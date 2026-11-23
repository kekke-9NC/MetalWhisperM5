// Build from the package root with the macOS 26 SDK or newer.
// swiftc -O -swift-version 6 -D METAL_WHISPER_DECODER_BENCHMARK -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
//   Sources/MetalWhisperM5/*.swift Benchmarks/Microbenchmarks/DecoderStepBenchmark.swift \
//   -o "/tmp/metal-whisper-decoder-bench"
// Run /tmp/metal-whisper-decoder-bench --help for the required inputs.

import Accelerate
import Foundation

@main
struct MetalWhisperDecoderStepBenchmark {
    private static let packCount = 24
    private static let beamSize = 5
    private static let warmups = 5
    private static let repeats = 15
    private static let numericalLimit: Float = 0.002
    private static let backends: [MetalWhisperDecoderBackend] = [.mps, .tensorF16, .tensorQ5]

    static func main() {
        do {
            try run()
        } catch {
            FileHandle.standardError.write(Data("Decoder step benchmark: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func run() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let usage = "Usage: metal-whisper-decoder-bench <model.bin> <audio>"
        if arguments == ["--help"] {
            print(usage)
            return
        }
        guard arguments.count == 2 else { throw BenchmarkError(usage) }
        let modelPath = arguments[0]
        let audioPath = arguments[1]
        let engine = try MetalWhisperEngine(
            modelURL: URL(fileURLWithPath: modelPath), maxBatchPacks: packCount,
            maxBeamsPerPack: beamSize, decoderBackend: .tensorQ5
        )
        let model = engine.model
        let hp = model.hparams
        guard engine.decoderBackend == .tensorQ5, hp.nTextState == 1280, hp.nTextLayer == 32,
              let head = model.decQ5Head else {
            throw BenchmarkError("The benchmark requires Apple10 Metal 4 and large-v3 Q5_0")
        }

        let tokenizer = model.tokenizer
        let specials = tokenizer.specials
        let prefix = [specials.sot, tokenizer.specials.languageTokenID(for: "en")!, specials.transcribe]
        let context = tokenizer.encodePrompt(
            "This synthetic prefix supplies decoder history for the benchmark. Every backend sees the same audio and context.",
            maxTokens: 48
        )
        let history = prefix + [specials.timestampBegin] + context
        let continuations = Array(tokenizer.encodePrompt(" The next tokens provide a fixed set of beam candidates.", maxTokens: 16).prefix(beamSize))
        try engine.prepareDecoderBenchmark(packs: try readPacks(audioPath), history: history, stepTokens: continuations)

        let denseBytes = model.decBlocks.reduce(model.decTokenEmb_F16.length) { sum, block in
            sum + block.selfQKV_F16.length + block.selfOut_F16.length + block.crossQ_F16.length
                + block.crossOut_F16.length + block.fc1_F16.length + block.fc2_F16.length
        }
        let q5Bytes = model.decQ5Blocks.reduce(head.buffer.length) { sum, block in
            sum + block.selfQKV.buffer.length + block.selfOut.buffer.length + block.crossQ.buffer.length
                + block.crossOut.buffer.length + block.fc1.buffer.length + block.fc2.buffer.length
        }
        print("\(engine.device.name), \(hp.nTextLayer) real layers + LM head, \(history.count) prior tokens per beam")
        print("Projection weight bytes per path: MPS \(denseBytes), A \(denseBytes - model.decTokenEmb_F16.length + head.buffer.length), B \(q5Bytes)")
        print("Cache condition: each timed step streams all 32 layers plus real Cross-KV from \(audioPath.split(separator: "/").last ?? "")")
        print("Rotated A/B order, \(warmups) warmups and \(repeats) timed samples, preparation and checks excluded")
        fflush(nil)

        let referenceWeights = makeReferenceWeights(engine)
        for b in [10, 40, 80, 120] {
            let variants = b <= 48 ? backends + [.mpsTensorHead] : backends
            var samples = [[MetalWhisperDecoderStepSample]](repeating: [], count: variants.count)
            for iteration in 0..<(warmups + repeats) {
                for n in variants.indices {
                    let index = (iteration + n) % variants.count
                    let sample = try engine.benchmarkDecoderStep(beams: b, backend: variants[index])
                    if iteration >= warmups { samples[index].append(sample) }
                }
            }
            for (index, backend) in variants.enumerated() {
                print(String(
                    format: "B=%3d %-8@ step GPU %.4f ms, wall %.4f ms, CPU encode %.4f ms",
                    b, backend.rawValue as NSString, median(samples[index].map(\.gpuMs)),
                    median(samples[index].map(\.elapsedMs)), median(samples[index].map(\.encodeMs))
                ))
            }
            fflush(nil)
            try compareOutputs(engine: engine, beams: b, weights: referenceWeights, variants: variants)
            try measureHead(engine: engine, beams: b)
            fflush(nil)
        }
    }

    private static func makeReferenceWeights(_ engine: MetalWhisperEngine) -> [Float] {
        let hp = engine.model.hparams
        let weightCount = hp.nVocab * hp.nTextState
        let half = engine.model.decTokenEmb_F16.contents().assumingMemoryBound(to: Float16.self)
        let weights = (0..<weightCount).map { Float(half[$0]) }
        var transposed = [Float](repeating: 0, count: weightCount)
        vDSP_mtrans(weights, 1, &transposed, 1, vDSP_Length(hp.nTextState), vDSP_Length(hp.nVocab))
        return transposed
    }

    private static func compareOutputs(
        engine: MetalWhisperEngine, beams: Int, weights: [Float], variants: [MetalWhisperDecoderBackend]
    ) throws {
        let hp = engine.model.hparams
        var baseline: MetalWhisperDecoderStepSample?
        var baselineTopTokens: [UInt32] = []
        for backend in variants {
            let raw = try engine.benchmarkDecoderStep(beams: beams, backend: backend, rawLogits: true)
            let filtered = try engine.benchmarkDecoderStep(beams: beams, backend: backend, capture: true)
            let input = raw.lmInput.map(Float.init)
            var reference = [Float](repeating: 0, count: beams * hp.nVocab)
            vDSP_mmul(
                input, 1, weights, 1, &reference, 1,
                vDSP_Length(beams), vDSP_Length(hp.nVocab), vDSP_Length(hp.nTextState)
            )
            let headError = try maxError(raw.rawLogits, reference)
            if baseline == nil {
                baseline = raw
                baselineTopTokens = filtered.topTokens
            }
            let stepDelta = try maxError(raw.rawLogits, baseline!.rawLogits)
            let changed = zip(filtered.topTokens, baselineTopTokens).count { $0 != $1 }
            print(String(
                format: "        %-8@ LM head FP32 max abs %.7g, full-step logit delta vs MPS %.7g, Top-8 differences %d/%d",
                backend.rawValue as NSString, headError, stepDelta, changed, beams * 8
            ))
            guard headError < numericalLimit else {
                throw BenchmarkError("\(backend.rawValue) B=\(beams) exceeded the LM head FP32 error gate")
            }
            if backend == .mpsTensorHead, stepDelta >= numericalLimit || changed != 0 {
                throw BenchmarkError("Hybrid decoder output drift at B=\(beams)")
            }
        }
    }

    private static func measureHead(engine: MetalWhisperEngine, beams: Int) throws {
        var mps: [Double] = []
        var q5: [Double] = []
        // A full step by the other backend streams its weights before each
        // isolated head and supplies a real Final LN activation.
        for i in 0..<(warmups + repeats) {
            _ = try engine.benchmarkDecoderStep(beams: beams, backend: .tensorF16)
            let mpsHead = try engine.benchmarkDecoderStep(beams: beams, backend: .mps, headOnly: true)
            _ = try engine.benchmarkDecoderStep(beams: beams, backend: .mps)
            let q5Head = try engine.benchmarkDecoderStep(beams: beams, backend: .tensorF16, headOnly: true)
            if i >= warmups {
                mps.append(mpsHead.gpuMs)
                q5.append(q5Head.gpuMs)
            }
        }
        print(String(format: "        Isolated LM head GPU MPS %.4f ms, Q5_0 %.4f ms", median(mps), median(q5)))
    }

    private static func maxError(_ result: [Float], _ reference: [Float]) throws -> Float {
        guard result.count == reference.count else { throw BenchmarkError("Unexpected logit count") }
        var maximum: Float = 0
        for i in result.indices {
            let error = abs(result[i] - reference[i])
            guard error.isFinite else { throw BenchmarkError("Non-finite decoder logit") }
            maximum = max(maximum, error)
        }
        return maximum
    }

    private static func median(_ values: [Double]) -> Double {
        values.sorted()[values.count / 2]
    }

    private static func readPacks(_ path: String) throws -> [[Float]] {
        let audio = try AudioLoader.load16kHzMono(from: URL(fileURLWithPath: path))
        let ranges = AudioLoader.chunkRanges(sampleCount: audio.count)
        guard ranges.count >= packCount, ranges[packCount - 1].count == 480_000 else {
            throw BenchmarkError("Expected at least 12 minutes of audio for 24 complete 30-second packs")
        }
        return ranges.prefix(packCount).map { Array(audio[$0]) }
    }

    private struct BenchmarkError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }
}

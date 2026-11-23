// Build from the package root with the macOS 26 SDK or newer.
// swiftc -O -swift-version 6 -D METAL_WHISPER_DECODER_BENCHMARK -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
//   Sources/MetalWhisperM5/*.swift Benchmarks/Microbenchmarks/CrossKVQ8Benchmark.swift \
//   -o "/tmp/metal-whisper-cross-kv-bench"
// Run /tmp/metal-whisper-cross-kv-bench --help for the required inputs.

import Foundation
import Metal

@main
struct MetalWhisperCrossKVQ8Benchmark {
    private static let packCount = 24
    private static let beamCount = 5
    private static let batchSizes = [10, 40, 80, 120]

    private struct Measurement {
        let crossMs: Double
        let stepGpuMs: Double
        let stepWallMs: Double
        let crossOutput: [Float16]
        let top8: [UInt32]
    }

    private struct BenchmarkError: LocalizedError {
        let errorDescription: String?
        init(_ reason: String) { errorDescription = reason }
    }

    static func main() {
        do {
            try run()
        } catch {
            FileHandle.standardError.write(Data("Cross-KV benchmark: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func run() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let usage = "Usage: metal-whisper-cross-kv-bench <model.bin> <audio> [q8Packed|q8|q8Keys|q8Values ...]"
        if arguments == ["--help"] {
            print(usage)
            return
        }
        guard arguments.count >= 2 else { throw BenchmarkError(usage) }
        let model = URL(fileURLWithPath: arguments[0])
        let packs = try readPacks(arguments[1])
        let candidates = try arguments.dropFirst(2).map { value -> MetalWhisperCrossKVFormat in
            guard let format = MetalWhisperCrossKVFormat(rawValue: value), format.quantizesKeys || format.quantizesValues
            else { throw BenchmarkError("Unknown Cross-KV format: \(value)") }
            return format
        }
        let testFormats: [MetalWhisperCrossKVFormat] = candidates.isEmpty
            ? [.q8Packed, .q8, .q8Keys, .q8Values] : candidates
        let (reference, query) = try measure(.f16, model: model, packs: packs, sharedQuery: nil)
        for format in testFormats where format != .f16 {
            let (results, _) = try measure(format, model: model, packs: packs, sharedQuery: query)
            var passesNumericalGate = true
            for batch in batchSizes {
                let base = reference[batch]!
                let result = results[batch]!
                let (maxError, cosine) = compare(base.crossOutput, result.crossOutput)
                passesNumericalGate = passesNumericalGate && cosine > 0.9999
                let positions = zip(base.top8, result.top8).filter(==).count
                let rows = (0..<batch).filter { b in
                    let start = b * 8
                    return base.top8[start..<start + 8].elementsEqual(result.top8[start..<start + 8])
                }.count
                print(String(
                    format: "%@ B=%3d cross GPU %.4f / %.4f ms, decoder GPU %.4f / %.4f ms, wall %.4f / %.4f ms, maxAbs %.7g, cosine %.8f, Top-8 positions %d/%d, exact rows %d/%d",
                    format.rawValue, batch, base.crossMs, result.crossMs,
                    base.stepGpuMs, result.stepGpuMs, base.stepWallMs, result.stepWallMs,
                    maxError, cosine, positions, batch * 8, rows, batch
                ))
            }
            print("\(format.rawValue) cosine > 0.9999: \(passesNumericalGate)")
            fflush(nil)
        }
    }

    private static func measure(
        _ format: MetalWhisperCrossKVFormat, model: URL, packs: [[Float]], sharedQuery: MTLBuffer?
    ) throws -> ([Int: Measurement], MTLBuffer) {
        let engine = try MetalWhisperEngine(
            modelURL: model, maxBatchPacks: packCount, maxBeamsPerPack: beamCount,
            decoderBackend: .mpsTensorHead, crossKVFormat: format
        )
        let hp = engine.model.hparams
        guard hp.nTextLayer == 32, hp.nTextState == 1280, engine.crossKVFormat == format else {
            throw BenchmarkError("Expected an Apple10 Metal GPU and large-v3 model with the requested Cross-KV format")
        }
        let vocabulary = engine.model.tokenizer
        let sp = vocabulary.specials
        let language = sp.languageTokenID(for: "en")!
        let history = [sp.sot, language, sp.transcribe, sp.timestampBegin] + vocabulary.encodePrompt(
            "This synthetic prefix supplies decoder history for the benchmark. Every backend sees the same audio and context.",
            maxTokens: 48
        )
        let next = vocabulary.encodePrompt(" The next tokens provide a fixed set of beam candidates.", maxTokens: 16)
        try engine.prepareDecoderBenchmark(packs: packs, history: history, stepTokens: Array(next.prefix(beamCount)))
        let beams = packCount * beamCount
        _ = try engine.benchmarkDecoderStep(beams: beams, backend: .mps)
        let query = try sharedQuery ?? engine.copyBenchmarkCrossQuery(beams: beams)
        print("\(format.rawValue) Cross-KV bytes for \(packCount) packs: \(engine.benchmarkCrossKVBytes)")
        fflush(nil)
        var results: [Int: Measurement] = [:]
        for batch in batchSizes {
            let elements = hp.nTextLayer * batch * hp.nTextState
            guard let output = engine.device.makeBuffer(length: elements * 2, options: .storageModeShared) else {
                throw BenchmarkError("Could not allocate Cross-Attention output")
            }
            var crossTimes: [Double] = []
            var stepGpuTimes: [Double] = []
            var stepWallTimes: [Double] = []
            for run in 0..<17 {
                let cross = try engine.benchmarkCrossAttention(beams: batch, query: query, output: output)
                let step = try engine.benchmarkDecoderStep(beams: batch, backend: .mps)
                if run >= 4 {
                    crossTimes.append(cross)
                    stepGpuTimes.append(step.gpuMs)
                    stepWallTimes.append(step.elapsedMs)
                }
            }
            let step = try engine.benchmarkDecoderStep(beams: batch, backend: .mps, capture: true)
            let data = UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float16.self), count: elements)
            results[batch] = Measurement(
                crossMs: median(crossTimes), stepGpuMs: median(stepGpuTimes), stepWallMs: median(stepWallTimes),
                crossOutput: Array(data), top8: step.topTokens
            )
        }
        return (results, query)
    }

    private static func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

    private static func compare(_ a: [Float16], _ b: [Float16]) -> (Float, Double) {
        precondition(a.count == b.count)
        var maxError: Float = 0
        var dot = 0.0, squaredA = 0.0, squaredB = 0.0
        for index in a.indices {
            let x = Double(a[index]), y = Double(b[index])
            precondition(x.isFinite && y.isFinite)
            maxError = max(maxError, Float(abs(x - y)))
            dot += x * y
            squaredA += x * x
            squaredB += y * y
        }
        return (maxError, dot / sqrt(squaredA * squaredB))
    }

    private static func readPacks(_ path: String) throws -> [[Float]] {
        let audio = try AudioLoader.load16kHzMono(from: URL(fileURLWithPath: path))
        let ranges = AudioLoader.chunkRanges(sampleCount: audio.count)
        guard ranges.count >= packCount, ranges[packCount - 1].count == 480_000 else {
            throw BenchmarkError("Expected at least 12 minutes of audio for 24 complete 30-second packs")
        }
        return ranges.prefix(packCount).map { Array(audio[$0]) }
    }
}

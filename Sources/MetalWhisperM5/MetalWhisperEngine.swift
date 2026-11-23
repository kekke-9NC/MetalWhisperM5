import Accelerate
import Compression
import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - Metal Uniform Structs (exact byte layout match with MSL)

private struct Im2ColParams {
    var P: UInt32
    var T_in: UInt32
    var T_out: UInt32
    var C_in: UInt32
    var stride: UInt32
}

private struct BiasGeluPosParams {
    var rows: UInt32
    var T: UInt32
    var D: UInt32
    var usePos: UInt32
}

private struct BiasGeluParams {
    var rows: UInt32
    var D: UInt32
}

private struct SplitQKVParams {
    var P: UInt32
    var T: UInt32
    var nHeads: UInt32
    var headDim: UInt32
}

private struct SplitCrossKVParams {
    var P: UInt32
    var T: UInt32
    var nHeads: UInt32
    var headDim: UInt32
    var layerOffset: UInt32
    var slot0: UInt32
    var slot1: UInt32
    var slot2: UInt32
    var slot3: UInt32
}

private struct EmbedParams {
    var B: UInt32
    var D: UInt32
}

private struct SelfAttnTreeParams {
    var B: UInt32
    var nHeads: UInt32
    var headDim: UInt32
    var maxTextCtx: UInt32
    var layerOffset: UInt32
}

private struct SelfAttnPrefillParams {
    var totalTokens: UInt32
    var nHeads: UInt32
    var headDim: UInt32
    var layerOffset: UInt32
}

private struct PackCrossGroup {
    var packIdx: UInt32
    var count: UInt32
    var r0: UInt32
    var r1: UInt32
    var r2: UInt32
    var r3: UInt32
    var r4: UInt32
    var pad: UInt32

    init(crossSlot: Int, rows: Range<Int>) {
        precondition((1...5).contains(rows.count))
        let base = rows.lowerBound
        packIdx = UInt32(crossSlot)
        count = UInt32(rows.count)
        r0 = UInt32(base)
        r1 = rows.count > 1 ? UInt32(base + 1) : 0
        r2 = rows.count > 2 ? UInt32(base + 2) : 0
        r3 = rows.count > 3 ? UInt32(base + 3) : 0
        r4 = rows.count > 4 ? UInt32(base + 4) : 0
        pad = 0
    }
}

private struct CrossAttnFusedParams {
    var numGroups: UInt32
    var P: UInt32
    var T: UInt32
    var nHeads: UInt32
    var headDim: UInt32
    var layerOffset: UInt32
}

private struct LogitsFilterRule {
    var step0: UInt32
    var allowText: UInt32
    var allowEot: UInt32
    var allowTs: UInt32
    var minTs: UInt32
    var maxTs: UInt32
    var suppressNonSpeech: UInt32
    var pad: UInt32
}

private struct LogitsFilterGlobal {
    var nVocab: UInt32
    var eot: UInt32
    var tsBegin: UInt32
    var topK: UInt32
}

private struct MPSGemmKey: Hashable {
    let m: Int
    let n: Int
    let k: Int
    let outF32: Bool
}

/// Each live pack owns one Cross-KV slot and every Self-KV slot allocated during
/// its current attempt, including ancestors shared by multiple beam hypotheses.
struct MetalWhisperKVSlotPool {
    let selfCapacity: Int
    private(set) var selfHighWaterMark = 0
    private var freeCrossSlots: [Int]
    private var freeSelfSlots: [UInt32]
    private var ownedSelfSlots: [[UInt32]]

    init(crossCount: Int, selfCount: Int) {
        selfCapacity = selfCount
        freeCrossSlots = Array((0..<crossCount).reversed())
        freeSelfSlots = (0..<selfCount).reversed().map(UInt32.init)
        ownedSelfSlots = Array(repeating: [], count: crossCount)
    }

    var freeCrossCount: Int { freeCrossSlots.count }
    var freeSelfCount: Int { freeSelfSlots.count }

    mutating func takeCrossSlot() -> Int? {
        freeCrossSlots.popLast()
    }

    mutating func takeSelfSlot(crossSlot: Int) throws -> UInt32 {
        guard let slot = freeSelfSlots.popLast() else {
            throw MetalWhisperError.bufferAllocationFailed("Self-KV slot pool")
        }
        ownedSelfSlots[crossSlot].append(slot)
        selfHighWaterMark = max(selfHighWaterMark, Int(slot) + 1)
        return slot
    }

    mutating func resetSelfSlots(crossSlot: Int) {
        freeSelfSlots.append(contentsOf: ownedSelfSlots[crossSlot].reversed())
        ownedSelfSlots[crossSlot].removeAll(keepingCapacity: true)
    }

    mutating func release(crossSlot: Int) {
        resetSelfSlots(crossSlot: crossSlot)
        freeCrossSlots.append(crossSlot)
    }
}

/// CPU workers compute a bounded number of mels ahead of the GPU. Only the
/// decoder thread copies finished mels into the shared Metal staging buffer.
private final class MetalWhisperMelPrefetcher: @unchecked Sendable {
    private let extractor: WhisperLogMelExtractor
    private let queue = DispatchQueue(label: "MetalWhisperM5.log-mel", qos: .userInitiated, attributes: .concurrent)
    private let condition = NSCondition()
    private let workers = DispatchGroup()
    private var ready: [Int: [Float]] = [:]
    private var scheduled = 0

    init(extractor: WhisperLogMelExtractor) {
        self.extractor = extractor
    }

    func schedule(packs: [MetalWhisperPackInput], upTo limit: Int) {
        let end = min(limit, packs.count)
        while scheduled < end {
            let index = scheduled
            let makeSamples = packs[index].makeSamples
            scheduled += 1
            queue.async(group: workers) {
                let mel = self.extractor.extractTimeMajor(samples: makeSamples())
                self.condition.lock()
                self.ready[index] = mel
                self.condition.signal()
                self.condition.unlock()
            }
        }
    }

    func take(index: Int) -> [Float] {
        condition.lock()
        defer { condition.unlock() }
        while ready[index] == nil {
            condition.wait()
        }
        return ready.removeValue(forKey: index)!
    }

    func waitForWorkers() {
        workers.wait()
    }
}

/// Optional per-call profiling. The MPS encoder submits its GPU
/// stages separately on the same queue while profiling.
private final class MetalWhisperProfile {
    enum EncoderStage {
        case convolution, attention, linear, crossKV, fused
    }
    private(set) var encodedPacks = 0
    private var encoderSamples = 0
    private var encoderGpuMs = 0.0
    private var convGpuMs = 0.0
    private var flashGpuMs = 0.0
    private var encoderGemmAndEpilogueGpuMs = 0.0
    private var crossKVGpuMs = 0.0
    private var decodeSteps = 0
    private var decodeRows = 0
    private var decodeAheadHits = 0
    private var decodeAheadPrepared = 0
    private var decodeAheadCpuMs = 0.0
    private var decodeStepCpuEncodeMs = 0.0
    private var decodeStepSubmitMs = 0.0
    private var decodeStepGpuWaitMs = 0.0
    private var decodeStepGpuMs = 0.0
    var melWaitMs = 0.0
    var encodeTotalMs = 0.0
    var prefillTotalMs = 0.0
    var decodeStepTotalMs = 0.0
    var peakSelfKVSlots = 0
    var peakUsedSelfKVSlots = 0

    func recordEncoder(_ stages: [(EncoderStage, MTLCommandBuffer)], packs: Int) throws {
        encodedPacks += packs
        for (stage, command) in stages {
            if let error = command.error { throw error }
            let gpuMs = (command.gpuEndTime - command.gpuStartTime) * 1000
            encoderGpuMs += gpuMs
            switch stage {
            case .convolution: convGpuMs += gpuMs
            case .attention: flashGpuMs += gpuMs
            case .linear: encoderGemmAndEpilogueGpuMs += gpuMs
            case .crossKV: crossKVGpuMs += gpuMs
            case .fused: break
            }
        }
        if stages.count > 1 { encoderSamples += 1 }
    }

    func recordDecodeStep(
        _ command: MTLCommandBuffer,
        rows: Int,
        reused: Bool,
        aheadCpuMs: Double?,
        start: Double,
        submit: Double,
        wait: Double,
        end: Double
    ) {
        decodeSteps += 1
        if reused { decodeAheadHits += 1 }
        if let aheadCpuMs {
            decodeAheadPrepared += 1
            decodeAheadCpuMs += aheadCpuMs
        }
        decodeRows += rows
        decodeStepCpuEncodeMs += (submit - start) * 1000
        decodeStepSubmitMs += (wait - submit) * 1000
        decodeStepGpuWaitMs += (end - wait) * 1000
        decodeStepGpuMs += (command.gpuEndTime - command.gpuStartTime) * 1000
    }

    func report(
        initMs: Double, audioLoadMs: Double, totalMs: Double,
        encoderBackend: MetalWhisperEncoderBackend, encoderPacksPerPass: Int,
        decoderBackend: MetalWhisperDecoderBackend, crossKVFormat: MetalWhisperCrossKVFormat, maxBatchPacks: Int
    ) -> MetalWhisperProfileReport {
        MetalWhisperProfileReport(
            initMs: initMs,
            audioLoadMs: audioLoadMs,
            totalMs: totalMs,
            melWaitMs: melWaitMs,
            encodedPacks: encodedPacks,
            encoderBackend: encoderBackend,
            encoderPacksPerPass: encoderPacksPerPass,
            encodeTotalMs: encodeTotalMs,
            encoderGpuMs: encoderGpuMs,
            convGpuMs: encoderSamples > 0 ? convGpuMs : nil,
            flashGpuMs: encoderSamples > 0 ? flashGpuMs : nil,
            encoderGemmAndEpilogueGpuMs: encoderSamples > 0 ? encoderGemmAndEpilogueGpuMs : nil,
            crossKVGpuMs: encoderSamples > 0 ? crossKVGpuMs : nil,
            prefillTotalMs: prefillTotalMs,
            decoderBackend: decoderBackend,
            crossKVFormat: crossKVFormat,
            maxBatchPacks: maxBatchPacks,
            decodeStepTotalMs: decodeStepTotalMs,
            decodeSteps: decodeSteps,
            decodeAheadHits: decodeAheadHits,
            decodeAheadPrepared: decodeAheadPrepared,
            decodeAheadCpuMs: decodeAheadCpuMs,
            averageDecodeRows: Double(decodeRows) / Double(max(1, decodeSteps)),
            peakSelfKVSlots: peakSelfKVSlots,
            peakUsedSelfKVSlots: peakUsedSelfKVSlots,
            decodeStepCpuEncodeMs: decodeStepCpuEncodeMs,
            decodeStepSubmitMs: decodeStepSubmitMs,
            decodeStepGpuWaitMs: decodeStepGpuWaitMs,
            decodeStepGpuMs: decodeStepGpuMs
        )
    }
}

/// A reusable Whisper model and Metal inference context. Concurrent calls on one
/// engine serialize access to its mutable GPU workspace.
public final class MetalWhisperEngine: @unchecked Sendable {
    let model: MetalWhisperModel
    let kernels: MetalWhisperKernels
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    public var modelParameters: WhisperHParams { model.hparams }
    public let maxBatchPacks: Int
    let maxBeamsPerPack: Int
    public let encoderBackend: MetalWhisperEncoderBackend
    let encoderPacksPerPass: Int
    /// The available decoder path. The profile reports the path used for a particular input.
    public let decoderBackend: MetalWhisperDecoderBackend
    public let crossKVFormat: MetalWhisperCrossKVFormat
    private let configuration: MetalWhisperConfiguration
    private let initMs: Double
    private var profile: MetalWhisperProfile?
    private var activeDecoderBackend: MetalWhisperDecoderBackend = .mps
    private let tensorDecoder: MetalWhisperDecoderTensorKernels?

    private let maxDecodeB: Int

    // Preallocated Encoder buffers
    private let melInBuf: MTLBuffer
    private let im2colBuf: MTLBuffer
    private let conv1OutBuf: MTLBuffer
    private let encHiddenBuf: MTLBuffer
    private let encNormBuf: MTLBuffer
    private let encTmpBuf: MTLBuffer
    private let encQKVBuf: MTLBuffer
    private let encQBuf: MTLBuffer
    private let encKBuf: MTLBuffer
    private let encVBuf: MTLBuffer
    private let encMLPBuf: MTLBuffer
    private let encCrossKVScratchBuf: MTLBuffer
    private let encResidualMoments: MTLBuffer?

    private struct KVBuffers {
        let k: MTLBuffer
        let v: MTLBuffer
        let slotCapacity: Int
    }

    private struct CrossKVBuffers {
        let k: MTLBuffer
        let v: MTLBuffer
        let kScale: MTLBuffer?
        let vScale: MTLBuffer?
        let slotCapacity: Int
    }

    // Cross-KV [layers, slots, heads, audioCtx, headDim] and Self-KV [layers, slots, D].
    // Allocate only the slots needed for this recording and reuse them in Pass 2.
    private var crossKV: CrossKVBuffers?
    private var selfKV: KVBuffers?

    // Decoder metadata & scratch buffers
    private let decTokenIDsBuf: MTLBuffer
    private let decPositionsBuf: MTLBuffer
    private let decWriteSlotsBuf: MTLBuffer
    private let decSeqLensBuf: MTLBuffer
    private let decSlotTableBuf: MTLBuffer
    private let decSeqStartsBuf: MTLBuffer
    private let decPosInSeqBuf: MTLBuffer
    private let decCrossGroupsBuf: MTLBuffer

    private let decHiddenBuf: MTLBuffer
    private let decNormBuf: MTLBuffer
    private let decQKVBuf: MTLBuffer
    private let decAttnOutBuf: MTLBuffer
    private let decTmpBuf: MTLBuffer
    private let decCrossQBuf: MTLBuffer
    private let decMLPBuf: MTLBuffer
    private let decResidualMoments: MTLBuffer?
    private let decLogitsInBuf: MTLBuffer
    private let decLogitsF32Buf: MTLBuffer

    // GPU-side Logits Filter + Top-8 + Raw Acoustic Token Timestamp buffers
    private let tokenMaskBuf: MTLBuffer
    private let logitsRulesBuf: MTLBuffer
    private let topKTokensBuf: MTLBuffer
    private let topKLogProbsBuf: MTLBuffer
    private let bestRawTsBuf: MTLBuffer

    // Cached MPS multiplication descriptors for the active matrix shapes.
    private var mpsCache: [MPSGemmKey: MPSMatrixMultiplication] = [:]

    private struct PreparedDecoderStep {
        let rows: Int
        let crossGroups: Int
        let command: MTLCommandBuffer
    }

    private let prepareDecoderAhead: Bool
    private var preparedDecoderStep: PreparedDecoderStep?

    private let serialLock = NSLock()

    public convenience init(modelURL: URL, configuration: MetalWhisperConfiguration = .init()) throws {
        try self.init(modelURL: modelURL, configuration: configuration, maxBeamsPerPack: 5, encoderPacksPerPass: 2)
    }

    // Explicit tuning for the standalone microbenchmarks.
    convenience init(
        modelURL: URL,
        maxBatchPacks: Int?,
        maxBeamsPerPack: Int = 5,
        encoderBackend: MetalWhisperEncoderBackend = .mps,
        encoderPacksPerPass: Int = 2,
        decoderBackend: MetalWhisperDecoderBackend = .mps,
        crossKVFormat: MetalWhisperCrossKVFormat = .f16,
        prepareDecoderAhead: Bool = false
    ) throws {
        try self.init(
            modelURL: modelURL,
            configuration: .init(
                encoderBackend: encoderBackend, decoderBackend: decoderBackend, crossKVFormat: crossKVFormat,
                maxBatchPacks: maxBatchPacks, prepareDecoderAhead: prepareDecoderAhead
            ),
            maxBeamsPerPack: maxBeamsPerPack, encoderPacksPerPass: encoderPacksPerPass
        )
    }

    private init(
        modelURL: URL, configuration: MetalWhisperConfiguration, maxBeamsPerPack: Int, encoderPacksPerPass: Int
    ) throws {
        let start = ProcessInfo.processInfo.systemUptime
        if let requested = configuration.maxBatchPacks, !(1...64).contains(requested) {
            throw MetalWhisperError.invalidConfiguration("maxBatchPacks must be between 1 and 64")
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw MetalWhisperError.metalUnavailable
        }
        self.device = device
        self.commandQueue = queue
        self.configuration = configuration
        self.prepareDecoderAhead = configuration.prepareDecoderAhead

        let supportsOptimizations = MetalWhisperKernels.supportsApple10Optimizations(device: device)
        let requestedDecoder = configuration.decoderBackend
        let useTensorDecoder = supportsOptimizations && requestedDecoder != .mps
        let model = try MetalWhisperModel(
            modelURL: modelURL, device: device, loadQ5Head: useTensorDecoder,
            loadQ5Decoder: useTensorDecoder && requestedDecoder == .tensorQ5
        )
        self.model = model
        let hp = model.hparams
        let hasDecoderWeights = model.decQ5Head != nil
            && (requestedDecoder != .tensorQ5 || model.decQ5Blocks.count == hp.nTextLayer)
        if #available(macOS 26.0, *), useTensorDecoder, hasDecoderWeights {
            self.tensorDecoder = try? MetalWhisperDecoderTensorKernels(
                device: device, headOnly: !requestedDecoder.hasTensorLayers
            )
        } else {
            self.tensorDecoder = nil
        }
        if tensorDecoder == nil {
            self.decoderBackend = .mps
        } else {
            self.decoderBackend = requestedDecoder == .auto ? .mpsTensorHead : requestedDecoder
        }

        let selected = try Self.compileKernels(
            device: device, hparams: hp, configuration: configuration, supportsOptimizations: supportsOptimizations
        )
        self.encoderBackend = selected.encoder
        self.crossKVFormat = selected.crossKV
        self.kernels = selected.kernels
        self.encoderPacksPerPass = min(4, max(1, encoderPacksPerPass))
        self.maxBeamsPerPack = min(5, max(1, maxBeamsPerPack))

        let crossValuesPerPack = hp.nTextLayer * hp.nAudioCtx * hp.nTextState
        let maxPacksByBuffer = min(
            device.maxBufferLength / (crossValuesPerPack * 2),
            Int(UInt32.max) / (crossValuesPerPack * 2)
        )
        if let requested = configuration.maxBatchPacks, requested > maxPacksByBuffer {
            throw MetalWhisperError.invalidConfiguration("maxBatchPacks exceeds the GPU buffer limit for this model")
        }
        guard maxPacksByBuffer >= 1 else {
            throw MetalWhisperError.bufferAllocationFailed("Cross-KV")
        }
        self.maxBatchPacks = configuration.maxBatchPacks
            ?? min(maxPacksByBuffer, Self.recommendedBatchPacks(workingSetSize: device.recommendedMaxWorkingSetSize))

        let D = hp.nAudioState
        let T = hp.nAudioCtx // 1500
        let T_mel = T * 2    // 3000
        let P = self.maxBatchPacks

        self.maxDecodeB = P * self.maxBeamsPerPack
        let maxPrefix = Self.promptTokenLimit(context: hp.nTextCtx) + 4
        let maxPrefillB = P * maxPrefix

        func makeSharedBuf(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard bytes <= device.maxBufferLength,
                  let buf = device.makeBuffer(length: max(256, bytes), options: .storageModeShared) else {
                throw MetalWhisperError.bufferAllocationFailed(label)
            }
            return buf
        }
        func makePrivateBuf(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard bytes <= device.maxBufferLength,
                  let buf = device.makeBuffer(length: max(256, bytes), options: .storageModePrivate) else {
                throw MetalWhisperError.bufferAllocationFailed(label)
            }
            return buf
        }

        // Encoder buffers (GPU-private for Apple Silicon lossless bandwidth compression & zero CPU page overhead)
        let encP = min(P, self.encoderPacksPerPass)
        self.melInBuf = try makeSharedBuf(encP * T_mel * hp.nMels * 4, "melIn")
        let im2colBytes = max(encP * T_mel * (3 * hp.nMels), encP * T * (3 * D)) * 2
        self.im2colBuf = try makePrivateBuf(im2colBytes, "im2col")
        self.conv1OutBuf = try makePrivateBuf(encP * T_mel * D * 2, "conv1Out")
        self.encHiddenBuf = try makePrivateBuf(encP * T * D * 2, "encHidden")
        self.encNormBuf = try makePrivateBuf(encP * T * D * 2, "encNorm")
        self.encTmpBuf = try makePrivateBuf(encP * T * D * 2, "encTmp")
        self.encQKVBuf = try makePrivateBuf(encP * T * (3 * D) * 2, "encQKV")
        self.encQBuf = try makePrivateBuf(encP * T * D * 2, "encQ")
        self.encKBuf = try makePrivateBuf(encP * T * D * 2, "encK")
        self.encVBuf = try makePrivateBuf(encP * T * D * 2, "encV")
        self.encMLPBuf = try makePrivateBuf(encP * T * (4 * D) * 2, "encMLP")
        self.encCrossKVScratchBuf = try makePrivateBuf(encP * T * (2 * D) * 2, "encCrossKVScratch")
        self.encResidualMoments = self.encoderBackend == .tensor
            ? try makePrivateBuf(encP * T * D * 4, "encResidualMoments") : nil

        // Decoder metadata (shared CPU->GPU) & activations (GPU-private)
        let maxB = max(self.maxDecodeB, maxPrefillB)
        self.decTokenIDsBuf = try makeSharedBuf(maxB * 4, "decTokenIDs")
        self.decPositionsBuf = try makeSharedBuf(maxB * 4, "decPositions")
        self.decWriteSlotsBuf = try makeSharedBuf(maxB * 4, "decWriteSlots")
        self.decSeqLensBuf = try makeSharedBuf(maxB * 4, "decSeqLens")
        self.decSlotTableBuf = try makeSharedBuf(self.maxDecodeB * hp.nTextCtx * 4, "decSlotTable")
        self.decSeqStartsBuf = try makeSharedBuf(maxB * 4, "decSeqStarts")
        self.decPosInSeqBuf = try makeSharedBuf(maxB * 4, "decPosInSeq")
        let maxCrossGroups = max(P, (maxB + 4) / 5 + P)
        self.decCrossGroupsBuf = try makeSharedBuf(maxCrossGroups * MemoryLayout<PackCrossGroup>.stride, "decCrossGroups")

        self.decHiddenBuf = try makePrivateBuf(maxB * D * 2, "decHidden")
        self.decNormBuf = try makePrivateBuf(maxB * D * 2, "decNorm")
        self.decQKVBuf = try makePrivateBuf(maxB * (3 * D) * 2, "decQKV")
        self.decAttnOutBuf = try makePrivateBuf(maxB * D * 2, "decAttnOut")
        self.decTmpBuf = try makePrivateBuf(maxB * D * 2, "decTmp")
        self.decCrossQBuf = try makePrivateBuf(maxB * D * 2, "decCrossQ")
        self.decMLPBuf = try makePrivateBuf(maxB * (4 * D) * 2, "decMLP")
        self.decResidualMoments = self.decoderBackend.hasTensorLayers
            ? try makePrivateBuf(self.maxDecodeB * hp.nTextState * 4, "decResidualMoments") : nil
        let maxLogitsRows = max(self.maxDecodeB, P * 2)
        self.decLogitsInBuf = try makePrivateBuf(maxLogitsRows * D * 2, "decLogitsIn")
        self.decLogitsF32Buf = try makeSharedBuf(maxLogitsRows * hp.nVocab * 4, "decLogitsF32")

        // Static GPU token mask (bit 0 = always-suppressed special token, bit 1 = non-speech token)
        var maskBytes = [UInt8](repeating: 0, count: hp.nVocab)
        let sp = model.tokenizer.specials
        let alwaysSuppress = [sp.noTimestamps, sp.sot, sp.prev, sp.solm, sp.transcribe, sp.translate, sp.noSpeech]
        for id in alwaysSuppress where id < hp.nVocab { maskBytes[id] |= 1 }
        let langStart = sp.sot + 1
        let langEnd = min(sp.timestampBegin, langStart + sp.numLanguages)
        for id in langStart..<langEnd { maskBytes[id] |= 1 }
        for id in WhisperSpecialTokens.suppressNonSpeechTokenIDs where id < sp.timestampBegin {
            maskBytes[id] |= 2
        }
        self.tokenMaskBuf = try makeSharedBuf(hp.nVocab, "tokenMask")
        self.tokenMaskBuf.contents().copyMemory(from: maskBytes, byteCount: hp.nVocab)

        self.logitsRulesBuf = try makeSharedBuf(maxLogitsRows * MemoryLayout<LogitsFilterRule>.stride, "logitsRules")
        self.topKTokensBuf = try makeSharedBuf(maxLogitsRows * 8 * 4, "topKTokens")
        self.topKLogProbsBuf = try makeSharedBuf(maxLogitsRows * 8 * 4, "topKLogProbs")
        self.bestRawTsBuf = try makeSharedBuf(maxLogitsRows * 4, "bestRawTs")
        self.initMs = (ProcessInfo.processInfo.systemUptime - start) * 1_000
    }

    static func recommendedBatchPacks(workingSetSize: UInt64) -> Int {
        switch workingSetSize / (1 << 30) {
        case 48...: 24
        case 40...: 16
        case 32...: 12
        case 24...: 8
        case 16...: 6
        default: 4
        }
    }

    static func supportsCooperativeAttention(hparams: WhisperHParams) -> Bool {
        hparams.nAudioCtx == 1_500 && hparams.nAudioHead > 0 && hparams.nAudioState > 0
            && hparams.nAudioState.isMultiple(of: 64) && hparams.nAudioState / 64 == hparams.nAudioHead
    }

    private static func compileKernels(
        device: MTLDevice, hparams: WhisperHParams, configuration: MetalWhisperConfiguration,
        supportsOptimizations: Bool
    ) throws -> (
        kernels: MetalWhisperKernels, encoder: MetalWhisperEncoderBackend, crossKV: MetalWhisperCrossKVFormat
    ) {
        let encoder: MetalWhisperEncoderBackend
        switch configuration.encoderBackend {
        case .auto, .cooperativeAttention:
            encoder = supportsOptimizations && supportsCooperativeAttention(hparams: hparams)
                ? .cooperativeAttention : .mps
        case .tensor, .tensorAttention:
            encoder = supportsOptimizations ? configuration.encoderBackend : .mps
        case .mps:
            encoder = .mps
        }
        let crossKV: MetalWhisperCrossKVFormat
        if !supportsOptimizations || hparams.headDim != 64 {
            crossKV = .f16
        } else if configuration.crossKVFormat == .auto {
            crossKV = encoder == .cooperativeAttention ? .q8Packed : .f16
        } else {
            crossKV = configuration.crossKVFormat
        }

        if encoder != .mps || crossKV != .f16 {
            let usesTensorEncoder = encoder == .tensor || encoder == .tensorAttention
            let usesCooperative = encoder == .cooperativeAttention
            if let optimized = try? MetalWhisperKernels(
                device: device, enableTensorEncoder: usesTensorEncoder,
                enableCooperativeAttention: usesCooperative, crossKVFormat: crossKV,
                enableRegisterAttention: encoder != .mps
            ), (!usesTensorEncoder || optimized.tensorEncoder != nil),
               (!usesCooperative || optimized.encoderFlashAttentionCooperative != nil) {
                return (optimized, encoder, crossKV)
            }
        }
        return (
            try MetalWhisperKernels(device: device, enableTensorEncoder: false, enableRegisterAttention: false),
            .mps, .f16
        )
    }

    // MARK: - Public transcription API

    /// Decodes an audio file supported by AVAudioFile using consecutive model windows.
    public func transcribe(
        audioURL: URL, options: MetalWhisperDecodeOptions = .init()
    ) throws -> MetalWhisperTranscription {
        let start = ProcessInfo.processInfo.systemUptime
        let samples = try AudioLoader.load16kHzMono(from: audioURL)
        let audioLoadMs = (ProcessInfo.processInfo.systemUptime - start) * 1_000
        return try transcribeSamples(samples, options: options, start: start, audioLoadMs: audioLoadMs)
    }

    /// Decodes mono Float32 PCM sampled at 16 kHz. Segment and token times are relative to the input.
    public func transcribe(
        samples16kHz: [Float], options: MetalWhisperDecodeOptions = .init()
    ) throws -> MetalWhisperTranscription {
        let start = ProcessInfo.processInfo.systemUptime
        guard samples16kHz.allSatisfy(\.isFinite) else { throw MetalWhisperError.invalidAudioInput }
        return try transcribeSamples(samples16kHz, options: options, start: start, audioLoadMs: 0)
    }

    private func transcribeSamples(
        _ samples: [Float], options: MetalWhisperDecodeOptions, start: Double, audioLoadMs: Double
    ) throws -> MetalWhisperTranscription {
        guard !samples.isEmpty else { throw MetalWhisperError.invalidAudioInput }
        let sampleRate = Double(AudioLoader.sampleRate)
        let duration = Double(samples.count) / sampleRate
        let ranges = AudioLoader.chunkRanges(
            sampleCount: samples.count, samplesPerWindow: model.melExtractor.samplesPerWindow
        )
        let packs = ranges.map { range in
            MetalWhisperPackInput(durationSeconds: Double(range.count) / sampleRate, initialPrompt: options.initialPrompt) {
                Array(samples[range])
            }
        }

        let (results, report) = try decodePacks(
            packs, durationSeconds: duration, options: options, start: start, audioLoadMs: audioLoadMs
        )
        let segments = zip(ranges, results).flatMap { range, result in
            let offset = Double(range.lowerBound) / sampleRate
            return result.segments.map { segment in
                MetalWhisperSegment(
                    start: segment.start + offset, end: segment.end + offset, text: segment.text,
                    avgLogProb: segment.avgLogProb, noSpeechProb: segment.noSpeechProb,
                    tokens: segment.tokens.map { token in
                        MetalWhisperToken(
                            id: token.id, text: token.text, start: min(segment.end, token.start) + offset,
                            end: min(segment.end, token.end) + offset,
                            probability: token.probability, isTimestamp: token.isTimestamp
                        )
                    }
                )
            }
        }
        let language = results.first { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }?.detectedLanguage
            ?? results.first?.detectedLanguage ?? options.languageCode
        return MetalWhisperTranscription(
            text: results.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines),
            detectedLanguage: language, durationSeconds: duration, segments: segments, profile: report
        )
    }

    /// Decodes caller-supplied windows in input order using continuous batching.
    public func transcribeBatch(
        packs: [MetalWhisperPackInput], options: MetalWhisperDecodeOptions = .init()
    ) throws -> [MetalWhisperDecodeResult] {
        guard !packs.isEmpty else { return [] }
        let duration = packs.reduce(0.0) { $0 + $1.durationSeconds }
        return try decodePacks(
            packs, durationSeconds: duration, options: options,
            start: ProcessInfo.processInfo.systemUptime, audioLoadMs: 0
        ).0
    }

    private func decodePacks(
        _ packs: [MetalWhisperPackInput], durationSeconds: Double,
        options: MetalWhisperDecodeOptions, start: Double, audioLoadMs: Double
    ) throws -> ([MetalWhisperDecodeResult], MetalWhisperProfileReport?) {
        guard packs.allSatisfy({ (0...30).contains($0.durationSeconds) }) else {
            throw MetalWhisperError.invalidAudioInput
        }
        guard (0...30).contains(options.maxInitialTimestamp),
              options.temperatureSchedule.allSatisfy({ (0...1_000_000).contains($0) }) else {
            throw MetalWhisperError.invalidConfiguration("timestamp limit or sampling temperature is out of range")
        }
        try Task.checkCancellation()
        serialLock.lock()
        defer { serialLock.unlock() }
        try Task.checkCancellation()

        activeDecoderBackend = configuration.decoderBackend == .auto && durationSeconds > 120 ? .mps : decoderBackend
        profile = configuration.enableProfiling ? MetalWhisperProfile() : nil
        defer { profile = nil }

        let results = try transcribeContinuousBatch(packs: packs, options: options)
        let report = profile?.report(
            initMs: initMs, audioLoadMs: audioLoadMs,
            totalMs: (ProcessInfo.processInfo.systemUptime - start) * 1_000,
            encoderBackend: encoderBackend, encoderPacksPerPass: encoderPacksPerPass,
            decoderBackend: activeDecoderBackend, crossKVFormat: crossKVFormat, maxBatchPacks: maxBatchPacks
        )
        return (results, report)
    }

    func transcribe(
        samples: [Float], durationSeconds: Double, initialPrompt: String = "",
        options: MetalWhisperDecodeOptions = .init()
    ) throws -> MetalWhisperDecodeResult {
        let input = MetalWhisperPackInput(samples: samples, durationSeconds: durationSeconds, initialPrompt: initialPrompt)
        return try transcribeBatch(packs: [input], options: options)[0]
    }

    // MARK: - Cached MPS Matrix Multiplication Helper

    private func getMPSGemm(M: Int, N: Int, K: Int, outF32: Bool = false) -> MPSMatrixMultiplication {
        let key = MPSGemmKey(m: M, n: N, k: K, outF32: outF32)
        if let existing = mpsCache[key] { return existing }
        let mm = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: M,
            resultColumns: N,
            interiorColumns: K,
            alpha: 1.0,
            beta: 0.0
        )
        mpsCache[key] = mm
        return mm
    }

    private func encodeMPSGemm(
        cb: MTLCommandBuffer,
        M: Int, N: Int, K: Int,
        aBuf: MTLBuffer, aOffset: Int = 0,
        bBuf: MTLBuffer,
        cBuf: MTLBuffer, cOffset: Int = 0,
        outF32: Bool = false
    ) {
        let descA = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * 2, dataType: .float16)
        let descB = MPSMatrixDescriptor(rows: N, columns: K, rowBytes: K * 2, dataType: .float16)
        let descC = MPSMatrixDescriptor(
            rows: M, columns: N,
            rowBytes: N * (outF32 ? 4 : 2),
            dataType: outF32 ? .float32 : .float16
        )
        let matA = MPSMatrix(buffer: aBuf, offset: aOffset, descriptor: descA)
        let matB = MPSMatrix(buffer: bBuf, descriptor: descB)
        let matC = MPSMatrix(buffer: cBuf, offset: cOffset, descriptor: descC)
        let mm = getMPSGemm(M: M, N: N, K: K, outF32: outF32)
        mm.encode(commandBuffer: cb, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    }

    // MARK: - Continuous pack scheduling

    private static func promptTokenLimit(context: Int) -> Int {
        max(0, min(200, context / 2 - 8))
    }

    private static func selfSlotsPerPack(context: Int, beams: Int, decodeTokens: Int) -> Int {
        // SOT, language and transcribe are shared. The first sampled token only
        // receives a KV slot if another token is decoded after it.
        let generatedSlots = min(max(1, decodeTokens) - 1, max(0, context - 4))
        let prefixSlots = min(promptTokenLimit(context: context) + 4, context - 1 - generatedSlots)
        return prefixSlots + beams * generatedSlots
    }

    private func kvBuffers(
        reusing existing: KVBuffers?,
        slots: Int,
        bytesPerSlot: Int,
        label: String
    ) throws -> KVBuffers {
        if let existing, existing.slotCapacity >= slots { return existing }
        let bytes = slots * bytesPerSlot
        guard bytes <= min(device.maxBufferLength, Int(UInt32.max)),
              let k = device.makeBuffer(length: bytes, options: .storageModePrivate),
              let v = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw MetalWhisperError.bufferAllocationFailed(label)
        }
        return KVBuffers(k: k, v: v, slotCapacity: slots)
    }

    private func reserveSelfKV(for slots: MetalWhisperKVSlotPool) throws {
        guard let current = selfKV, slots.selfHighWaterMark > current.slotCapacity else { return }
        preparedDecoderStep = nil
        let hp = model.hparams
        let bytesPerRow = hp.nTextState * MemoryLayout<Float16>.stride
        let capacity = min(
            slots.selfCapacity, max(slots.selfHighWaterMark, current.slotCapacity + max(256, current.slotCapacity / 2))
        )
        let expanded = try kvBuffers(
            reusing: nil, slots: capacity, bytesPerSlot: hp.nTextLayer * bytesPerRow, label: "Self-KV growth"
        )
        guard let cb = commandQueue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
            throw MetalWhisperError.metalUnavailable
        }

        // A slot keeps its ID when the layer stride changes. All previous
        // decoder commands have completed before the copy begins.
        let oldStride = current.slotCapacity * bytesPerRow
        let newStride = expanded.slotCapacity * bytesPerRow
        for layer in 0..<hp.nTextLayer {
            blit.copy(
                from: current.k, sourceOffset: layer * oldStride,
                to: expanded.k, destinationOffset: layer * newStride, size: oldStride
            )
            blit.copy(
                from: current.v, sourceOffset: layer * oldStride,
                to: expanded.v, destinationOffset: layer * newStride, size: oldStride
            )
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        selfKV = expanded
    }

    private func reserveCrossKV(slots: Int) throws {
        if let crossKV, crossKV.slotCapacity >= slots { return }
        preparedDecoderStep = nil
        let hp = model.hparams
        let frames = hp.nTextLayer * slots * hp.nTextHead * hp.nAudioCtx

        func allocate(bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
                throw MetalWhisperError.bufferAllocationFailed(label)
            }
            return buffer
        }

        let keysQ8 = crossKVFormat.quantizesKeys
        let valuesQ8 = crossKVFormat.quantizesValues
        let samples = frames * hp.headDim
        if crossKVFormat == .q8Packed {
            let data = try allocate(bytes: samples * 2, label: "Packed Q8 Cross-KV")
            let scales = try allocate(bytes: frames * 4, label: "Packed Q8 Cross-KV scales")
            crossKV = CrossKVBuffers(k: data, v: data, kScale: scales, vScale: scales, slotCapacity: slots)
            return
        }
        crossKV = try CrossKVBuffers(
            k: allocate(bytes: samples * (keysQ8 ? 1 : 2), label: "Cross-KV keys"),
            v: allocate(bytes: samples * (valuesQ8 ? 1 : 2), label: "Cross-KV values"),
            kScale: keysQ8 ? allocate(bytes: frames * 2, label: "Cross-KV key scales") : nil,
            vScale: valuesQ8 ? allocate(bytes: frames * 2, label: "Cross-KV value scales") : nil,
            slotCapacity: slots
        )
    }

    private func transcribeContinuousBatch(
        packs: [MetalWhisperPackInput], options: MetalWhisperDecodeOptions
    ) throws -> [MetalWhisperDecodeResult] {
        preparedDecoderStep = nil
        defer { preparedDecoderStep = nil }
        let hp = model.hparams
        let specials = model.tokenizer.specials
        let beamSize = min(maxBeamsPerPack, max(1, options.beamSize))
        let crossCount = min(maxBatchPacks, packs.count)
        let maxSteps = max(1, options.maxDecodeTokens)
        let requiredSelfSlots = crossCount * Self.selfSlotsPerPack(
            context: hp.nTextCtx, beams: beamSize, decodeTokens: maxSteps
        )
        // Reserve prompt space and the first 48 generated positions per beam.
        let initialSelfSlots = min(requiredSelfSlots, crossCount * Self.selfSlotsPerPack(
            context: hp.nTextCtx, beams: beamSize, decodeTokens: min(maxSteps, 49)
        ))
        let bytesPerToken = hp.nTextLayer * hp.nTextState * MemoryLayout<Float16>.stride
        try reserveCrossKV(slots: crossCount)
        selfKV = try kvBuffers(
            reusing: selfKV, slots: initialSelfSlots, bytesPerSlot: bytesPerToken, label: "Self-KV"
        )
        var slots = MetalWhisperKVSlotPool(crossCount: crossCount, selfCount: requiredSelfSlots)
        defer {
            if let profile {
                profile.peakSelfKVSlots = max(profile.peakSelfKVSlots, selfKV?.slotCapacity ?? 0)
                profile.peakUsedSelfKVSlots = max(profile.peakUsedSelfKVSlots, slots.selfHighWaterMark)
            }
        }
        let temperatures: [Float] = options.temperatureSchedule.isEmpty ? [0] : options.temperatureSchedule
        let explicitLanguage = specials.languageTokenID(for: options.languageCode)
        let defaultLanguage = explicitLanguage ?? specials.sot + 1
        let promptLimit = Self.promptTokenLimit(context: hp.nTextCtx)

        let prefetcher = MetalWhisperMelPrefetcher(extractor: model.melExtractor)
        let melStride = model.melExtractor.frameCount * hp.nMels
        let prefetchDepth = min(18, crossCount + 2)
        prefetcher.schedule(packs: packs, upTo: prefetchDepth)
        defer { prefetcher.waitForWorkers() }

        var active: [PackDecodeState] = []
        var awaitingPrefill: [PackDecodeState] = []
        active.reserveCapacity(crossCount)
        awaitingPrefill.reserveCapacity(crossCount)
        var results = [MetalWhisperDecodeResult?](repeating: nil, count: packs.count)
        var nextInput = 0
        while nextInput < packs.count || !active.isEmpty || !awaitingPrefill.isEmpty {
            try Task.checkCancellation()

            while nextInput < packs.count && slots.freeCrossCount > 0 {
                let count = min(encoderPacksPerPass, packs.count - nextInput, slots.freeCrossCount)
                let melStart = CFAbsoluteTimeGetCurrent()
                var crossSlots: [Int] = []
                for local in 0..<count {
                    let crossSlot = slots.takeCrossSlot()!
                    crossSlots.append(crossSlot)
                    let mel = prefetcher.take(index: nextInput + local)
                    let byteOffset = local * melStride * MemoryLayout<Float>.stride
                    mel.withUnsafeBytes { bytes in
                        melInBuf.contents().advanced(by: byteOffset)
                            .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
                    }
                }
                let stagingMs = (CFAbsoluteTimeGetCurrent() - melStart) * 1000
                profile?.melWaitMs += stagingMs
                let melMs = stagingMs / Double(count)
                prefetcher.schedule(packs: packs, upTo: nextInput + count + prefetchDepth)

                let encodeStart = CFAbsoluteTimeGetCurrent()
                try encodeChunkIntoCrossKV(slots: crossSlots)
                let elapsedEncodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000
                profile?.encodeTotalMs += elapsedEncodeMs
                let encodeMs = elapsedEncodeMs / Double(count)

                for crossSlot in crossSlots {
                    let input = packs[nextInput]
                    let tokens = model.tokenizer.encodePrompt(input.initialPrompt, maxTokens: 512)
                    awaitingPrefill.append(PackDecodeState(
                        inputIndex: nextInput,
                        crossSlot: crossSlot,
                        durationSeconds: input.durationSeconds,
                        melMs: melMs,
                        encodeMs: encodeMs,
                        emptySpeechRecoveryThreshold: input.emptySpeechRecoveryThreshold,
                        promptTokens: Array(tokens.suffix(promptLimit)),
                        languageID: defaultLanguage,
                        temperatures: temperatures
                    ))
                    nextInput += 1
                }
            }

            if !awaitingPrefill.isEmpty {
                let needsLanguage: (PackDecodeState) -> Bool = {
                    hp.isMultilingual && explicitLanguage == nil && $0.isBeamSearch
                }
                for detectLanguage in [false, true] {
                    var group = awaitingPrefill.filter { needsLanguage($0) == detectLanguage }
                    guard !group.isEmpty else { continue }
                    let started = CFAbsoluteTimeGetCurrent()
                    try prefill(&group, detectLanguage: detectLanguage, beamSize: beamSize, options: options, slots: &slots)
                    let prefillMs = (CFAbsoluteTimeGetCurrent() - started) * 1000
                    profile?.prefillTotalMs += prefillMs
                    let perPack = prefillMs / Double(group.count)
                    for i in group.indices { group[i].decodeMs += perPack }
                    active.append(contentsOf: group)
                }
                awaitingPrefill.removeAll(keepingCapacity: true)
            }

            let decodeStart = CFAbsoluteTimeGetCurrent()
            try advanceDecode(
                &active, beamSize: beamSize, maxSteps: maxSteps, options: options, slots: &slots
            )
            let elapsedDecodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
            profile?.decodeStepTotalMs += elapsedDecodeMs
            let decodeMs = elapsedDecodeMs / Double(max(1, active.count))

            for i in active.indices.reversed() {
                active[i].decodeMs += decodeMs
                if active[i].steps < maxSteps && active[i].beams.contains(where: { !$0.finished }) {
                    continue
                }
                var pack = active.remove(at: i)
                let pool = pack.completed.isEmpty ? pack.beams : pack.completed
                let best = pool.max(by: { $0.score < $1.score }) ?? BeamHyp(
                    genTokens: [], genLogProbs: [], genRawTs: [], kvSlots: [], sumLogProb: -10, finished: true
                )
                var outcome = buildOutcome(
                    beam: best,
                    durationSeconds: pack.durationSeconds,
                    noSpeechProb: pack.noSpeechProb,
                    temperatureUsed: pack.temperature
                )

                let recoveryThreshold = pack.originalOutcome == nil ? nil : pack.emptySpeechRecoveryThreshold
                let noSpeechThreshold = recoveryThreshold ?? options.noSpeechThreshold
                let isSilent = outcome.noSpeechProb > noSpeechThreshold
                    && outcome.avgLogProb < options.logprobThreshold
                let needsFallback = !isSilent && (outcome.compressionRatio > options.compressionRatioThreshold
                    || outcome.avgLogProb < options.logprobThreshold)
                if needsFallback && pack.temperatureIndex + 1 < pack.temperatures.count {
                    pack.temperatureIndex += 1
                } else if outcome.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                          pack.originalOutcome == nil, pack.emptySpeechRecoveryThreshold != nil {
                    pack.originalOutcome = (outcome, pack.languageID)
                    pack.promptTokens = []
                    pack.languageID = defaultLanguage
                    pack.temperatures = [0.2, 0.4, 0.6, 0.8]
                    pack.temperatureIndex = 0
                } else {
                    slots.release(crossSlot: pack.crossSlot)
                    if let original = pack.originalOutcome,
                       outcome.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        outcome = original.outcome
                        pack.languageID = original.languageID
                    }
                    results[pack.inputIndex] = MetalWhisperDecodeResult(
                        text: outcome.text,
                        detectedLanguage: hp.isMultilingual
                            ? specials.languageCode(forTokenID: pack.languageID) ?? options.languageCode : "en",
                        noSpeechProb: outcome.noSpeechProb,
                        avgLogProb: outcome.avgLogProb,
                        compressionRatio: outcome.compressionRatio,
                        temperatureUsed: outcome.temperatureUsed,
                        segments: outcome.segments,
                        melMs: pack.melMs,
                        encodeMs: pack.encodeMs,
                        decodeMs: pack.decodeMs
                    )
                    continue
                }
                slots.resetSelfSlots(crossSlot: pack.crossSlot)
                awaitingPrefill.append(pack)
            }
        }

        precondition(slots.freeSelfCount == requiredSelfSlots && slots.freeCrossCount == crossCount)
        return results.map { $0! }
    }

    // MARK: - Encoder + Cross-KV Precomputation

    private func encodeChunkIntoCrossKV(slots: [Int]) throws {
        if encoderBackend == .tensor, let tensorEncoder = kernels.tensorEncoder {
            try encodeTensorChunkIntoCrossKV(slots: slots, kernels: tensorEncoder)
        } else {
            try encodeMPSChunkIntoCrossKV(slots: slots)
        }
    }


    private func encodeTensorChunkIntoCrossKV(
        slots: [Int],
        kernels tensor: MetalWhisperEncoderTensorKernels
    ) throws {
        guard let command = commandQueue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder(dispatchType: .serial) else {
            throw MetalWhisperError.metalUnavailable
        }
        let hp = model.hparams
        let d = hp.nAudioState
        let t = hp.nAudioCtx
        let rows = slots.count * t
        let residualMoments = encResidualMoments!
        let crossKV = crossKV!
        let slotIDs = (0..<4).map { $0 < slots.count ? UInt32(slots[$0]) : 0 }
        let oneRowPerGroup = MTLSize(width: rows, height: 1, depth: 1)
        let normThreads = MTLSize(width: 256, height: 1, depth: 1)

        func linear(
            _ input: MTLBuffer, _ weights: MTLBuffer, _ output: MTLBuffer,
            rows: Int, columns: Int, inner: Int,
            epilogue: MetalWhisperLinearParams.Epilogue, bias: MTLBuffer?,
            auxiliary: MTLBuffer? = nil, extra: MTLBuffer? = nil, layerOffset: Int = 0
        ) {
            var params = MetalWhisperLinearParams(rows: rows, columns: columns, inner: inner, epilogue: epilogue)
            params.context = UInt32(t)
            params.heads = UInt32(hp.nAudioHead)
            params.headDim = UInt32(hp.headDim)
            params.layerOffset = UInt32(layerOffset)
            params.slot0 = slotIDs[0]
            params.slot1 = slotIDs[1]
            params.slot2 = slotIDs[2]
            params.slot3 = slotIDs[3]
            tensor.dispatchLinear(
                encoder, params: params, input: input, weights: weights,
                output: output, bias: bias, auxiliary: auxiliary, extra: extra
            )
        }

        func im2col(
            _ pipeline: MTLComputePipelineState, _ input: MTLBuffer,
            frames: Int, channels: Int, stride: Int
        ) {
            let outFrames = frames / stride
            var params = Im2ColParams(
                P: UInt32(slots.count), T_in: UInt32(frames), T_out: UInt32(outFrames),
                C_in: UInt32(channels), stride: UInt32(stride)
            )
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(im2colBuf, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<Im2ColParams>.stride, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: channels * 3, height: outFrames, depth: slots.count),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
            )
        }

        func residualAndNorm(
            _ input: MTLBuffer, _ weights: MTLBuffer, _ bias: MTLBuffer,
            inner: Int, normWeight: MTLBuffer, normBias: MTLBuffer
        ) {
            linear(
                input, weights, residualMoments, rows: rows, columns: d, inner: inner,
                epilogue: .residual, bias: bias, auxiliary: encHiddenBuf
            )
            encoder.setComputePipelineState(tensor.residualLayerNorm)
            encoder.setBuffer(residualMoments, offset: 0, index: 0)
            encoder.setBuffer(encHiddenBuf, offset: 0, index: 1)
            encoder.setBuffer(encNormBuf, offset: 0, index: 2)
            encoder.setBuffer(normWeight, offset: 0, index: 3)
            encoder.setBuffer(normBias, offset: 0, index: 4)
            var width = UInt32(d)
            encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.dispatchThreadgroups(oneRowPerGroup, threadsPerThreadgroup: normThreads)
        }

        im2col(kernels.im2colConv1DF32ToF16, melInBuf, frames: t * 2, channels: hp.nMels, stride: 1)
        linear(
            im2colBuf, model.conv1W_F16, conv1OutBuf, rows: rows * 2, columns: d, inner: hp.nMels * 3,
            epilogue: .gelu, bias: model.conv1B_F32
        )
        im2col(kernels.im2colConv1DF16ToF16, conv1OutBuf, frames: t * 2, channels: d, stride: 2)
        linear(
            im2colBuf, model.conv2W_F16, encHiddenBuf, rows: rows, columns: d, inner: d * 3,
            epilogue: .geluPosition, bias: model.conv2B_F32, auxiliary: model.encPosEmb_F32
        )

        encoder.setComputePipelineState(kernels.layerNormF16)
        encoder.setBuffer(encHiddenBuf, offset: 0, index: 0)
        encoder.setBuffer(encNormBuf, offset: 0, index: 1)
        encoder.setBuffer(model.encBlocks[0].attnLnW, offset: 0, index: 2)
        encoder.setBuffer(model.encBlocks[0].attnLnB, offset: 0, index: 3)
        var width = UInt32(d)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(oneRowPerGroup, threadsPerThreadgroup: normThreads)

        var attentionParams = SplitQKVParams(
            P: UInt32(slots.count), T: UInt32(t), nHeads: UInt32(hp.nAudioHead), headDim: UInt32(hp.headDim)
        )
        let attentionGrid = MTLSize(width: (t + 31) / 32, height: hp.nAudioHead, depth: slots.count)
        let attentionThreads = MTLSize(width: 64, height: 1, depth: 1)
        for (layer, block) in model.encBlocks.enumerated() {
            linear(
                encNormBuf, block.qkvW_F16, encQBuf, rows: rows, columns: d * 3, inner: d,
                epilogue: .splitQKV, bias: block.qkvB_F32, auxiliary: encKBuf, extra: encVBuf
            )
            encoder.setComputePipelineState(tensor.attention)
            encoder.setBuffer(encQBuf, offset: 0, index: 0)
            encoder.setBuffer(encKBuf, offset: 0, index: 1)
            encoder.setBuffer(encVBuf, offset: 0, index: 2)
            encoder.setBuffer(encTmpBuf, offset: 0, index: 3)
            encoder.setBytes(&attentionParams, length: MemoryLayout<SplitQKVParams>.stride, index: 4)
            encoder.dispatchThreadgroups(attentionGrid, threadsPerThreadgroup: attentionThreads)

            residualAndNorm(
                encTmpBuf, block.outW_F16, block.outB_F32, inner: d,
                normWeight: block.mlpLnW, normBias: block.mlpLnB
            )
            linear(
                encNormBuf, block.fc1W_F16, encMLPBuf, rows: rows, columns: d * 4, inner: d,
                epilogue: .gelu, bias: block.fc1B_F32
            )
            let next = layer + 1 < model.encBlocks.count ? model.encBlocks[layer + 1] : nil
            residualAndNorm(
                encMLPBuf, block.fc2W_F16, block.fc2B_F32, inner: d * 4,
                normWeight: next?.attnLnW ?? model.encLnPostW, normBias: next?.attnLnB ?? model.encLnPostB
            )
        }
        let crossLayerStride = crossKV.slotCapacity * t * d
        for (layer, block) in model.decBlocks.enumerated() {
            if crossKVFormat == .f16 {
                linear(
                    encNormBuf, block.crossKV_F16, crossKV.k, rows: rows, columns: d * 2, inner: d,
                    epilogue: .splitCrossKV, bias: block.crossKV_B, auxiliary: crossKV.v,
                    layerOffset: layer * crossLayerStride
                )
            } else {
                linear(
                    encNormBuf, block.crossKV_F16, encCrossKVScratchBuf, rows: rows, columns: d * 2, inner: d,
                    epilogue: .half, bias: nil
                )
                encodeCrossKVProjection(
                    encoder, bias: block.crossKV_B, slots: slots.count, slotIDs: slotIDs,
                    layerOffset: layer * crossLayerStride, cache: crossKV
                )
            }
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        try profile?.recordEncoder([(.fused, command)], packs: slots.count)
    }



    private func encodeMPSChunkIntoCrossKV(slots: [Int]) throws {
        let subP = slots.count
        guard var cb = commandQueue.makeCommandBuffer() else {
            throw MetalWhisperError.metalUnavailable
        }

        var profiledStages: [(MetalWhisperProfile.EncoderStage, MTLCommandBuffer)] = []
        var stage = MetalWhisperProfile.EncoderStage.convolution
        func startStage(_ next: MetalWhisperProfile.EncoderStage) throws {
            guard profile != nil else { return }
            cb.commit()
            profiledStages.append((stage, cb))
            guard let command = commandQueue.makeCommandBuffer() else {
                cb.waitUntilCompleted()
                throw MetalWhisperError.metalUnavailable
            }
            cb = command
            stage = next
        }

        let hp = model.hparams
        let D = hp.nAudioState
        let T = hp.nAudioCtx
        let T_mel = T * 2
        let nHeads = hp.nAudioHead
        let headDim = hp.headDim
        let rowsMel = subP * T_mel
        let rowsEnc = subP * T

        // Conv1 im2col + GEMM
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(kernels.im2colConv1DF32ToF16)
            var p1 = Im2ColParams(
                P: UInt32(subP), T_in: UInt32(T_mel), T_out: UInt32(T_mel),
                C_in: UInt32(hp.nMels), stride: 1
            )
            enc.setBuffer(melInBuf, offset: 0, index: 0)
            enc.setBuffer(im2colBuf, offset: 0, index: 1)
            enc.setBytes(&p1, length: MemoryLayout<Im2ColParams>.stride, index: 2)
            enc.dispatchThreads(
                MTLSize(width: 3 * hp.nMels, height: T_mel, depth: subP),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
            )
            enc.endEncoding()
        }
        encodeMPSGemm(cb: cb, M: rowsMel, N: D, K: 3 * hp.nMels, aBuf: im2colBuf, bBuf: model.conv1W_F16, cBuf: conv1OutBuf)

        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(kernels.addBiasGeluPosF16)
            var bg1 = BiasGeluPosParams(rows: UInt32(rowsMel), T: UInt32(T_mel), D: UInt32(D), usePos: 0)
            enc.setBuffer(conv1OutBuf, offset: 0, index: 0)
            enc.setBuffer(model.conv1B_F32, offset: 0, index: 1)
            enc.setBuffer(model.encPosEmb_F32, offset: 0, index: 2)
            enc.setBytes(&bg1, length: MemoryLayout<BiasGeluPosParams>.stride, index: 3)
            enc.dispatchThreads(
                MTLSize(width: D, height: rowsMel, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
            )

            enc.setComputePipelineState(kernels.im2colConv1DF16ToF16)
            var p2 = Im2ColParams(
                P: UInt32(subP), T_in: UInt32(T_mel), T_out: UInt32(T),
                C_in: UInt32(D), stride: 2
            )
            enc.setBuffer(conv1OutBuf, offset: 0, index: 0)
            enc.setBuffer(im2colBuf, offset: 0, index: 1)
            enc.setBytes(&p2, length: MemoryLayout<Im2ColParams>.stride, index: 2)
            enc.dispatchThreads(
                MTLSize(width: 3 * D, height: T, depth: subP),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
            )
            enc.endEncoding()
        }

        // Conv2 GEMM + bias + erf-GELU + posEmb + initial LN
        encodeMPSGemm(cb: cb, M: rowsEnc, N: D, K: 3 * D, aBuf: im2colBuf, bBuf: model.conv2W_F16, cBuf: encHiddenBuf)

        var d32 = UInt32(D)
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(kernels.addBiasGeluPosF16)
            var bg2 = BiasGeluPosParams(rows: UInt32(rowsEnc), T: UInt32(T), D: UInt32(D), usePos: 1)
            enc.setBuffer(encHiddenBuf, offset: 0, index: 0)
            enc.setBuffer(model.conv2B_F32, offset: 0, index: 1)
            enc.setBuffer(model.encPosEmb_F32, offset: 0, index: 2)
            enc.setBytes(&bg2, length: MemoryLayout<BiasGeluPosParams>.stride, index: 3)
            enc.dispatchThreads(
                MTLSize(width: D, height: rowsEnc, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
            )

            enc.setComputePipelineState(kernels.layerNormF16)
            enc.setBuffer(encHiddenBuf, offset: 0, index: 0)
            enc.setBuffer(encNormBuf, offset: 0, index: 1)
            enc.setBuffer(model.encBlocks[0].attnLnW, offset: 0, index: 2)
            enc.setBuffer(model.encBlocks[0].attnLnB, offset: 0, index: 3)
            enc.setBytes(&d32, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: rowsEnc, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
        }

        try startStage(.linear)

        // 32 Encoder Transformer Layers
        var splitParams = SplitQKVParams(
            P: UInt32(subP), T: UInt32(T), nHeads: UInt32(nHeads), headDim: UInt32(headDim)
        )
        let faGrid = MTLSize(width: (T + 31) / 32, height: nHeads, depth: subP)
        let tensorAttention = encoderBackend == .tensorAttention ? kernels.tensorEncoder?.attention : nil
        let attention = tensorAttention ?? kernels.encoderFlashAttentionRegisterF16 ?? kernels.encoderFlashAttentionF16
        let faThreads = MTLSize(width: tensorAttention == nil ? 128 : 64, height: 1, depth: 1)
        let cooperativeAttention = kernels.encoderFlashAttentionCooperative
        let tileRows = MetalWhisperKernels.CooperativeAttention.rowsPerGroup
        let completeGroups = (T - 1) / tileRows

        for layer in 0..<hp.nAudioLayer {
            let blk = model.encBlocks[layer]
            encodeMPSGemm(cb: cb, M: rowsEnc, N: 3 * D, K: D, aBuf: encNormBuf, bBuf: blk.qkvW_F16, cBuf: encQKVBuf)

            do {
                var enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(kernels.addBiasSplitQKVF16)
                enc.setBuffer(encQKVBuf, offset: 0, index: 0)
                enc.setBuffer(blk.qkvB_F32, offset: 0, index: 1)
                enc.setBuffer(encQBuf, offset: 0, index: 2)
                enc.setBuffer(encKBuf, offset: 0, index: 3)
                enc.setBuffer(encVBuf, offset: 0, index: 4)
                enc.setBytes(&splitParams, length: MemoryLayout<SplitQKVParams>.stride, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: D, height: T, depth: subP),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
                )

                if profile != nil {
                    enc.endEncoding()
                    try startStage(.attention)
                    enc = cb.makeComputeCommandEncoder()!
                }
                enc.setBuffer(encQBuf, offset: 0, index: 0)
                enc.setBuffer(encKBuf, offset: 0, index: 1)
                enc.setBuffer(encVBuf, offset: 0, index: 2)
                enc.setBuffer(encTmpBuf, offset: 0, index: 3)
                enc.setBytes(&splitParams, length: MemoryLayout<SplitQKVParams>.stride, index: 4)
                if let cooperativeAttention {
                    // Keep the final query tile on the reference accumulation path.
                    if completeGroups > 0 {
                        enc.setComputePipelineState(cooperativeAttention.tiles)
                        enc.dispatchThreadgroups(
                            MTLSize(width: completeGroups, height: nHeads, depth: subP),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                        )
                    }
                    enc.setComputePipelineState(cooperativeAttention.finalTile)
                    enc.dispatchThreadgroups(
                        MTLSize(width: faGrid.width - completeGroups * (tileRows / 32), height: nHeads, depth: subP),
                        threadsPerThreadgroup: faThreads
                    )
                } else {
                    enc.setComputePipelineState(attention)
                    enc.dispatchThreadgroups(faGrid, threadsPerThreadgroup: faThreads)
                }
                enc.endEncoding()
            }

            try startStage(.linear)
            encodeMPSGemm(cb: cb, M: rowsEnc, N: D, K: D, aBuf: encTmpBuf, bBuf: blk.outW_F16, cBuf: encNormBuf)

            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(kernels.addBiasResidualLayerNormF16)
                enc.setBuffer(encHiddenBuf, offset: 0, index: 0)
                enc.setBuffer(encNormBuf, offset: 0, index: 1)
                enc.setBuffer(blk.outB_F32, offset: 0, index: 2)
                enc.setBuffer(encTmpBuf, offset: 0, index: 3)
                enc.setBuffer(blk.mlpLnW, offset: 0, index: 4)
                enc.setBuffer(blk.mlpLnB, offset: 0, index: 5)
                enc.setBytes(&d32, length: 4, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: rowsEnc, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            encodeMPSGemm(cb: cb, M: rowsEnc, N: 4 * D, K: D, aBuf: encTmpBuf, bBuf: blk.fc1W_F16, cBuf: encMLPBuf)

            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(kernels.addBiasGeluF16)
                var mlpParams = BiasGeluParams(rows: UInt32(rowsEnc), D: UInt32(4 * D))
                enc.setBuffer(encMLPBuf, offset: 0, index: 0)
                enc.setBuffer(blk.fc1B_F32, offset: 0, index: 1)
                enc.setBytes(&mlpParams, length: MemoryLayout<BiasGeluParams>.stride, index: 2)
                enc.dispatchThreads(
                    MTLSize(width: 4 * D, height: rowsEnc, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
                )
                enc.endEncoding()
            }

            encodeMPSGemm(cb: cb, M: rowsEnc, N: D, K: 4 * D, aBuf: encMLPBuf, bBuf: blk.fc2W_F16, cBuf: encTmpBuf)

            let nextLnW = (layer + 1 < hp.nAudioLayer) ? model.encBlocks[layer + 1].attnLnW : model.encLnPostW
            let nextLnB = (layer + 1 < hp.nAudioLayer) ? model.encBlocks[layer + 1].attnLnB : model.encLnPostB
            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(kernels.addBiasResidualLayerNormF16)
                enc.setBuffer(encHiddenBuf, offset: 0, index: 0)
                enc.setBuffer(encTmpBuf, offset: 0, index: 1)
                enc.setBuffer(blk.fc2B_F32, offset: 0, index: 2)
                enc.setBuffer(encNormBuf, offset: 0, index: 3)
                enc.setBuffer(nextLnW, offset: 0, index: 4)
                enc.setBuffer(nextLnB, offset: 0, index: 5)
                enc.setBytes(&d32, length: 4, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: rowsEnc, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }
        }

        try startStage(.crossKV)

        // Populate the assigned ring slots independently of decoder row order.
        let crossKV = crossKV!
        let layerElems = crossKV.slotCapacity * nHeads * T * headDim
        let slotIDs = (0..<4).map { $0 < slots.count ? UInt32(slots[$0]) : 0 }
        for layer in 0..<hp.nTextLayer {
            let dBlk = model.decBlocks[layer]
            encodeMPSGemm(
                cb: cb, M: rowsEnc, N: 2 * D, K: D,
                aBuf: encNormBuf, bBuf: dBlk.crossKV_F16, cBuf: encCrossKVScratchBuf
            )
            let enc = cb.makeComputeCommandEncoder()!
            encodeCrossKVProjection(
                enc, bias: dBlk.crossKV_B, slots: subP, slotIDs: slotIDs,
                layerOffset: layer * layerElems, cache: crossKV
            )
            enc.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        if let profile {
            profiledStages.append((stage, cb))
            try profile.recordEncoder(profiledStages, packs: subP)
        }
    }

    private func encodeCrossKVProjection(
        _ encoder: MTLComputeCommandEncoder, bias: MTLBuffer, slots: Int, slotIDs: [UInt32],
        layerOffset: Int, cache: CrossKVBuffers
    ) {
        let hp = model.hparams
        var params = SplitCrossKVParams(
            P: UInt32(slots), T: UInt32(hp.nAudioCtx), nHeads: UInt32(hp.nAudioHead), headDim: UInt32(hp.headDim),
            layerOffset: UInt32(layerOffset),
            slot0: slotIDs[0], slot1: slotIDs[1], slot2: slotIDs[2], slot3: slotIDs[3]
        )
        encoder.setComputePipelineState(kernels.splitCrossKVQ8 ?? kernels.splitCrossKVF16)
        encoder.setBuffer(encCrossKVScratchBuf, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        encoder.setBuffer(cache.k, offset: 0, index: 2)
        encoder.setBuffer(cache.v, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<SplitCrossKVParams>.stride, index: 4)
        if kernels.splitCrossKVQ8 != nil {
            encoder.setBuffer(cache.kScale, offset: 0, index: 5)
            encoder.setBuffer(cache.vScale, offset: 0, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (hp.nAudioCtx + 7) / 8, height: hp.nAudioHead, depth: slots),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        } else {
            encoder.dispatchThreads(
                MTLSize(width: hp.nAudioState, height: hp.nAudioCtx, depth: slots),
                threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
            )
        }
    }

    private func encodeCrossAttention(
        _ encoder: MTLComputeCommandEncoder, cache: CrossKVBuffers, layer: Int, groups: Int,
        query: MTLBuffer, output: MTLBuffer, outputOffset: Int = 0
    ) {
        let hp = model.hparams
        let stride = cache.slotCapacity * hp.nAudioCtx * hp.nTextState
        var params = CrossAttnFusedParams(
            numGroups: UInt32(groups), P: UInt32(cache.slotCapacity), T: UInt32(hp.nAudioCtx),
            nHeads: UInt32(hp.nTextHead), headDim: UInt32(hp.headDim), layerOffset: UInt32(layer * stride)
        )
        encoder.setComputePipelineState(kernels.decoderCrossAttnPackFusedQ8 ?? kernels.decoderCrossAttnPackFusedF16)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(cache.k, offset: 0, index: 1)
        encoder.setBuffer(cache.v, offset: 0, index: 2)
        encoder.setBuffer(decCrossGroupsBuf, offset: 0, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<CrossAttnFusedParams>.stride, index: 5)
        encoder.setBuffer(model.decBlocks[layer].crossQ_B, offset: 0, index: 6)
        if kernels.decoderCrossAttnPackFusedQ8 != nil {
            encoder.setBuffer(cache.kScale, offset: 0, index: 7)
            encoder.setBuffer(cache.vScale, offset: 0, index: 8)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: hp.nTextHead, height: groups, depth: 1),
            threadsPerThreadgroup: MTLSize(width: crossKVFormat == .q8Packed ? 192 : 256, height: 1, depth: 1)
        )
    }

    // MARK: - Decoder Forward Pass (Prefill & Step)

    private func encodeDecoderBody(
        cb: MTLCommandBuffer,
        B: Int,
        numCrossGroups: Int,
        isPrefill: Bool,
        sharedEncoder: MTLComputeCommandEncoder? = nil,
        useQ5Layers: Bool = false
    ) {
        let hp = model.hparams
        let D = hp.nTextState
        let nHeads = hp.nTextHead
        let headDim = hp.headDim
        var d32 = UInt32(D)
        let selfKV = selfKV!
        let crossKV = crossKV!
        let selfLayerStride = selfKV.slotCapacity * D

        func beginPass() -> MTLComputeCommandEncoder {
            sharedEncoder ?? cb.makeComputeCommandEncoder()!
        }

        func finishPass(_ encoder: MTLComputeCommandEncoder) {
            if sharedEncoder == nil { encoder.endEncoding() }
        }

        func gemm(
            _ columns: Int, inner: Int, input: MTLBuffer, weights: MTLBuffer, q5: MetalWhisperQ5Weights?,
            output: MTLBuffer, epilogue: MetalWhisperDecoderEpilogue = .projection, bias: MTLBuffer? = nil
        ) {
            if let encoder = sharedEncoder, let tensorDecoder {
                tensorDecoder.dispatch(
                    encoder, rows: B, columns: columns, inner: inner,
                    input: input, f16: weights, q5: q5, output: output,
                    epilogue: epilogue, bias: bias, hidden: decHiddenBuf
                )
            } else {
                encodeMPSGemm(cb: cb, M: B, N: columns, K: inner, aBuf: input, bBuf: weights, cBuf: output)
            }
        }

        func residualProjectionAndLayerNorm(
            inner: Int, input: MTLBuffer, weights: MTLBuffer, q5: MetalWhisperQ5Weights?,
            projectionBias: MTLBuffer, lnWeights: MTLBuffer, lnBias: MTLBuffer
        ) {
            let moments = sharedEncoder == nil ? nil : decResidualMoments
            gemm(
                D, inner: inner, input: input, weights: weights, q5: q5,
                output: moments ?? decTmpBuf, epilogue: .residual, bias: projectionBias
            )
            let encoder = beginPass()
            if let moments, let residualLayerNorm = tensorDecoder?.residualLayerNorm {
                encoder.setComputePipelineState(residualLayerNorm)
                encoder.setBuffer(moments, offset: 0, index: 0)
                encoder.setBuffer(decHiddenBuf, offset: 0, index: 1)
                encoder.setBuffer(decNormBuf, offset: 0, index: 2)
                encoder.setBuffer(lnWeights, offset: 0, index: 3)
                encoder.setBuffer(lnBias, offset: 0, index: 4)
                encoder.setBytes(&d32, length: 4, index: 5)
            } else {
                encoder.setComputePipelineState(kernels.addBiasResidualLayerNormF16)
                encoder.setBuffer(decHiddenBuf, offset: 0, index: 0)
                encoder.setBuffer(decTmpBuf, offset: 0, index: 1)
                encoder.setBuffer(projectionBias, offset: 0, index: 2)
                encoder.setBuffer(decNormBuf, offset: 0, index: 3)
                encoder.setBuffer(lnWeights, offset: 0, index: 4)
                encoder.setBuffer(lnBias, offset: 0, index: 5)
                encoder.setBytes(&d32, length: 4, index: 6)
            }
            encoder.dispatchThreadgroups(
                MTLSize(width: B, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            finishPass(encoder)
        }

        // Embed lookup + initial Self-Attn LN for layer 0
        do {
            let enc = beginPass()
            var embP = EmbedParams(B: UInt32(B), D: UInt32(D))
            enc.setComputePipelineState(kernels.decoderEmbedLookupF16)
            enc.setBuffer(decTokenIDsBuf, offset: 0, index: 0)
            enc.setBuffer(decPositionsBuf, offset: 0, index: 1)
            enc.setBuffer(model.decTokenEmb_F16, offset: 0, index: 2)
            enc.setBuffer(model.decPosEmb_F32, offset: 0, index: 3)
            enc.setBuffer(decHiddenBuf, offset: 0, index: 4)
            enc.setBytes(&embP, length: MemoryLayout<EmbedParams>.stride, index: 5)
            enc.dispatchThreads(
                MTLSize(width: D, height: B, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: min(8, B), depth: 1)
            )

            enc.setComputePipelineState(kernels.layerNormF16)
            enc.setBuffer(decHiddenBuf, offset: 0, index: 0)
            enc.setBuffer(decNormBuf, offset: 0, index: 1)
            enc.setBuffer(model.decBlocks[0].selfLnW, offset: 0, index: 2)
            enc.setBuffer(model.decBlocks[0].selfLnB, offset: 0, index: 3)
            enc.setBytes(&d32, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: B, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            finishPass(enc)
        }

        for layer in 0..<hp.nTextLayer {
            let blk = model.decBlocks[layer]
            let q5 = useQ5Layers ? model.decQ5Blocks[layer] : nil

            // Self QKV: [B, D] * [3D, D]^T
            gemm(3 * D, inner: D, input: decNormBuf, weights: blk.selfQKV_F16, q5: q5?.selfQKV, output: decQKVBuf)

            // Self-Attention with fused QKV bias (Prefill or Zero-Copy Beam-Tree)
            do {
                let enc = beginPass()
                if isPrefill {
                    var saP = SelfAttnPrefillParams(
                        totalTokens: UInt32(B), nHeads: UInt32(nHeads), headDim: UInt32(headDim),
                        layerOffset: UInt32(layer * selfLayerStride)
                    )
                    enc.setComputePipelineState(kernels.decoderSelfAttnPrefillF16)
                    enc.setBuffer(decQKVBuf, offset: 0, index: 0)
                    enc.setBuffer(selfKV.k, offset: 0, index: 1)
                    enc.setBuffer(selfKV.v, offset: 0, index: 2)
                    enc.setBuffer(decWriteSlotsBuf, offset: 0, index: 3)
                    enc.setBuffer(decSeqStartsBuf, offset: 0, index: 4)
                    enc.setBuffer(decPosInSeqBuf, offset: 0, index: 5)
                    enc.setBuffer(decAttnOutBuf, offset: 0, index: 6)
                    enc.setBytes(&saP, length: MemoryLayout<SelfAttnPrefillParams>.stride, index: 7)
                    enc.setBuffer(blk.selfQKV_B, offset: 0, index: 8)
                    enc.dispatchThreadgroups(
                        MTLSize(width: nHeads, height: B, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                    )
                } else {
                    var saP = SelfAttnTreeParams(
                        B: UInt32(B), nHeads: UInt32(nHeads), headDim: UInt32(headDim),
                        maxTextCtx: UInt32(hp.nTextCtx), layerOffset: UInt32(layer * selfLayerStride)
                    )
                    enc.setComputePipelineState(kernels.decoderSelfAttnBeamTreeF16)
                    enc.setBuffer(decQKVBuf, offset: 0, index: 0)
                    enc.setBuffer(selfKV.k, offset: 0, index: 1)
                    enc.setBuffer(selfKV.v, offset: 0, index: 2)
                    enc.setBuffer(decWriteSlotsBuf, offset: 0, index: 3)
                    enc.setBuffer(decSeqLensBuf, offset: 0, index: 4)
                    enc.setBuffer(decSlotTableBuf, offset: 0, index: 5)
                    enc.setBuffer(decAttnOutBuf, offset: 0, index: 6)
                    enc.setBytes(&saP, length: MemoryLayout<SelfAttnTreeParams>.stride, index: 7)
                    enc.setBuffer(blk.selfQKV_B, offset: 0, index: 8)
                    enc.dispatchThreadgroups(
                        MTLSize(width: nHeads, height: B, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                    )
                }
                finishPass(enc)
            }

            residualProjectionAndLayerNorm(
                inner: D, input: decAttnOutBuf, weights: blk.selfOut_F16, q5: q5?.selfOut,
                projectionBias: blk.selfOut_B, lnWeights: blk.crossLnW, lnBias: blk.crossLnB
            )

            gemm(D, inner: D, input: decNormBuf, weights: blk.crossQ_F16, q5: q5?.crossQ, output: decCrossQBuf)

            // Pack-Fused 5-Beam Cross-Attention with fused Q bias (5x less Cross-KV DRAM traffic!)
            do {
                let enc = beginPass()
                encodeCrossAttention(
                    enc, cache: crossKV, layer: layer, groups: numCrossGroups,
                    query: decCrossQBuf, output: decAttnOutBuf
                )
                finishPass(enc)
            }

            residualProjectionAndLayerNorm(
                inner: D, input: decAttnOutBuf, weights: blk.crossOut_F16, q5: q5?.crossOut,
                projectionBias: blk.crossOut_B, lnWeights: blk.mlpLnW, lnBias: blk.mlpLnB
            )

            // The Metal 4 FC1 applies bias and erf-GELU in its epilogue.
            gemm(
                4 * D, inner: D, input: decNormBuf, weights: blk.fc1_F16, q5: q5?.fc1,
                output: decMLPBuf, epilogue: .gelu, bias: blk.fc1_B
            )
            if sharedEncoder == nil {
                let enc = beginPass()
                var mlpBP = BiasGeluParams(rows: UInt32(B), D: UInt32(4 * D))
                enc.setComputePipelineState(kernels.addBiasGeluF16)
                enc.setBuffer(decMLPBuf, offset: 0, index: 0)
                enc.setBuffer(blk.fc1_B, offset: 0, index: 1)
                enc.setBytes(&mlpBP, length: MemoryLayout<BiasGeluParams>.stride, index: 2)
                enc.dispatchThreads(
                    MTLSize(width: 4 * D, height: B, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: min(8, B), depth: 1)
                )
                finishPass(enc)
            }

            // FC2 residual feeds the next Self-LN or the final decoder LN.
            let nextLnW = (layer + 1 < hp.nTextLayer) ? model.decBlocks[layer + 1].selfLnW : model.decLnW
            let nextLnB = (layer + 1 < hp.nTextLayer) ? model.decBlocks[layer + 1].selfLnB : model.decLnB
            residualProjectionAndLayerNorm(
                inner: 4 * D, input: decMLPBuf, weights: blk.fc2_F16, q5: q5?.fc2,
                projectionBias: blk.fc2_B, lnWeights: nextLnW, lnBias: nextLnB
            )
        }
    }

    /// Runs batched causal prefill over `totalTokens` and computes raw FP32 LM Head logits for `logitRowIndices`.
    private func runDecoderPrefill(
        tokenIDs: [UInt32],
        positions: [UInt32],
        writeSlots: [UInt32],
        seqStarts: [UInt32],
        posInSeq: [UInt32],
        crossGroups: [PackCrossGroup],
        logitRowIndices: [Int]
    ) throws {
        let B = tokenIDs.count
        guard B > 0 else { return }
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MetalWhisperError.metalUnavailable
        }

        let hp = model.hparams
        let D = hp.nTextState

        decTokenIDsBuf.contents().copyMemory(from: tokenIDs, byteCount: B * 4)
        decPositionsBuf.contents().copyMemory(from: positions, byteCount: B * 4)
        decWriteSlotsBuf.contents().copyMemory(from: writeSlots, byteCount: B * 4)
        decSeqStartsBuf.contents().copyMemory(from: seqStarts, byteCount: B * 4)
        decPosInSeqBuf.contents().copyMemory(from: posInSeq, byteCount: B * 4)
        decCrossGroupsBuf.contents().copyMemory(
            from: crossGroups,
            byteCount: crossGroups.count * MemoryLayout<PackCrossGroup>.stride
        )

        encodeDecoderBody(cb: cb, B: B, numCrossGroups: crossGroups.count, isPrefill: true)

        if !logitRowIndices.isEmpty {
            let rowBytes = D * 2
            let blit = cb.makeBlitCommandEncoder()!
            for (dstIdx, srcRow) in logitRowIndices.enumerated() {
                blit.copy(
                    from: decNormBuf, sourceOffset: srcRow * rowBytes,
                    to: decLogitsInBuf, destinationOffset: dstIdx * rowBytes,
                    size: rowBytes
                )
            }
            blit.endEncoding()

            encodeMPSGemm(
                cb: cb, M: logitRowIndices.count, N: hp.nVocab, K: D,
                aBuf: decLogitsInBuf, bBuf: model.decTokenEmb_F16, cBuf: decLogitsF32Buf,
                outF32: true
            )
        }

        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
    }

    private func runDecoderStep(
        B: Int,
        numCrossGroups: Int,
        runGPUFilterTopK: Bool
    ) throws {
        guard B > 0 else { return }
        let timing = runGPUFilterTopK ? profile : nil
        let start = timing == nil ? 0 : CFAbsoluteTimeGetCurrent()
        let decoderBackend = activeDecoderBackend
        let canPrepare = prepareDecoderAhead && !decoderBackend.hasTensorLayers && runGPUFilterTopK
        let prepared = preparedDecoderStep
        preparedDecoderStep = nil
        let reused = canPrepare && prepared?.rows == B && prepared?.crossGroups == numCrossGroups

        let command: MTLCommandBuffer
        if reused, let prepared {
            command = prepared.command
        } else {
            guard let fresh = commandQueue.makeCommandBuffer() else {
                throw MetalWhisperError.metalUnavailable
            }
            command = fresh
            try encodeDecoderStep(
                cb: command, B: B, numCrossGroups: numCrossGroups, backend: decoderBackend,
                runGPUFilterTopK: runGPUFilterTopK
            )
        }

        let submit = timing == nil ? 0 : CFAbsoluteTimeGetCurrent()
        command.commit()
        let wait = timing == nil ? 0 : CFAbsoluteTimeGetCurrent()
        var aheadCpuMs: Double?
        if canPrepare, let next = commandQueue.makeCommandBuffer() {
            // Commands bind shared buffers. The next step fills their contents before submitting them.
            do {
                try encodeDecoderStep(
                    cb: next, B: B, numCrossGroups: numCrossGroups, backend: decoderBackend, runGPUFilterTopK: true
                )
                preparedDecoderStep = PreparedDecoderStep(rows: B, crossGroups: numCrossGroups, command: next)
                if timing != nil { aheadCpuMs = (CFAbsoluteTimeGetCurrent() - wait) * 1_000 }
            } catch {
                // A future step can encode its commands normally.
            }
        }
        command.waitUntilCompleted()
        if let err = command.error { throw err }
        timing?.recordDecodeStep(
            command, rows: B, reused: reused, aheadCpuMs: aheadCpuMs,
            start: start, submit: submit, wait: wait, end: CFAbsoluteTimeGetCurrent()
        )
    }

    private func encodeDecoderStep(
        cb: MTLCommandBuffer, B: Int, numCrossGroups: Int, backend: MetalWhisperDecoderBackend, runGPUFilterTopK: Bool
    ) throws {
        if !backend.hasTensorLayers {
            encodeDecoderBody(cb: cb, B: B, numCrossGroups: numCrossGroups, isPrefill: false)
            let useTensorHead = backend == .mpsTensorHead && B <= 48
            if !useTensorHead {
                encodeDecoderLMHead(cb: cb, B: B)
            }
            if useTensorHead || runGPUFilterTopK {
                guard let encoder = cb.makeComputeCommandEncoder() else { throw MetalWhisperError.metalUnavailable }
                if useTensorHead {
                    encodeDecoderLMHead(cb: cb, B: B, encoder: encoder)
                }
                if runGPUFilterTopK {
                    encodeGPUFilter(encoder, B: B)
                }
                encoder.endEncoding()
            }
        } else {
            guard let encoder = cb.makeComputeCommandEncoder(dispatchType: .serial) else {
                throw MetalWhisperError.metalUnavailable
            }
            encodeDecoderBody(
                cb: cb, B: B, numCrossGroups: numCrossGroups, isPrefill: false,
                sharedEncoder: encoder, useQ5Layers: backend == .tensorQ5
            )
            encodeDecoderLMHead(cb: cb, B: B, encoder: encoder)
            if runGPUFilterTopK {
                encodeGPUFilter(encoder, B: B)
            }
            encoder.endEncoding()
        }
    }

    private func encodeDecoderLMHead(cb: MTLCommandBuffer, B: Int, encoder: MTLComputeCommandEncoder? = nil) {
        let hp = model.hparams
        if let encoder, let tensorDecoder, let weights = model.decQ5Head {
            tensorDecoder.dispatch(
                encoder, rows: B, columns: hp.nVocab, inner: hp.nTextState,
                input: decNormBuf, f16: model.decTokenEmb_F16, q5: weights,
                output: decLogitsF32Buf, epilogue: .logits
            )
        } else {
            encodeMPSGemm(
                cb: cb, M: B, N: hp.nVocab, K: hp.nTextState,
                aBuf: decNormBuf, bBuf: model.decTokenEmb_F16, cBuf: decLogitsF32Buf, outF32: true
            )
        }
    }

    /// Runs GPU filter/log-softmax/Top-8 standalone on `B` logit rows already in `decLogitsF32Buf`.
    private func runGPUFilterOnly(B: Int, rowByteOffset: Int = 0) throws {
        guard B > 0, let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw MetalWhisperError.metalUnavailable
        }
        encodeGPUFilter(enc, B: B, rowByteOffset: rowByteOffset)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
    }

    private func encodeGPUFilter(_ enc: MTLComputeCommandEncoder, B: Int, rowByteOffset: Int = 0) {
        let hp = model.hparams
        let sp = model.tokenizer.specials
        var glob = LogitsFilterGlobal(
            nVocab: UInt32(hp.nVocab),
            eot: UInt32(sp.eot),
            tsBegin: UInt32(sp.timestampBegin),
            topK: 8
        )
        enc.setComputePipelineState(kernels.whisperLogitsFilterTopKF32)
        enc.setBuffer(decLogitsF32Buf, offset: rowByteOffset, index: 0)
        enc.setBuffer(tokenMaskBuf, offset: 0, index: 1)
        enc.setBuffer(logitsRulesBuf, offset: 0, index: 2)
        enc.setBuffer(topKTokensBuf, offset: 0, index: 3)
        enc.setBuffer(topKLogProbsBuf, offset: 0, index: 4)
        enc.setBytes(&glob, length: MemoryLayout<LogitsFilterGlobal>.stride, index: 5)
        enc.setBuffer(bestRawTsBuf, offset: 0, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: B, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    // MARK: - Batched Beam Search 5 & Best-of-5 Sampling

    private struct BeamHyp {
        var genTokens: [Int]
        var genLogProbs: [Float]
        var genRawTs: [Int]
        var kvSlots: [UInt32]
        var sumLogProb: Float
        var finished: Bool

        var score: Float {
            sumLogProb / Float(max(1, genTokens.count))
        }
    }

    private struct PackDecodeState {
        let inputIndex: Int
        let crossSlot: Int
        let durationSeconds: Double
        let melMs: Double
        let encodeMs: Double
        let emptySpeechRecoveryThreshold: Double?
        var promptTokens: [Int]
        var languageID: Int
        var temperatures: [Float]
        var temperatureIndex = 0
        var decodeMs = 0.0
        var noSpeechProb = 0.0
        var steps = 1
        var beams: [BeamHyp] = []
        var completed: [BeamHyp] = []
        var stepsSinceFirstTopEOT = 0
        var rng = SplitMix64(seed: 0)
        var originalOutcome: (outcome: PackDecodeOutcome, languageID: Int)?

        var temperature: Float { temperatures[temperatureIndex] }
        var isBeamSearch: Bool { temperature <= 1e-6 }
    }

    private struct PackDecodeOutcome {
        let text: String
        let noSpeechProb: Double
        let avgLogProb: Double
        let compressionRatio: Double
        let temperatureUsed: Float
        let segments: [MetalWhisperSegment]
    }

    private func makeFilterRule(
        genTokens: [Int],
        durationSeconds: Double,
        options: MetalWhisperDecodeOptions
    ) -> LogitsFilterRule {
        let sp = model.tokenizer.specials
        let nVocab = model.hparams.nVocab
        let tsBegin = sp.timestampBegin
        let maxAudioTs = min(
            nVocab - 1,
            tsBegin + max(50, Int((min(30.0, durationSeconds + 0.5) / 0.02).rounded(.up)))
        )

        if genTokens.isEmpty {
            let maxInit = min(maxAudioTs, tsBegin + Int((options.maxInitialTimestamp / 0.02).rounded()))
            return LogitsFilterRule(
                step0: 1, allowText: 0, allowEot: 0, allowTs: 1,
                minTs: UInt32(tsBegin), maxTs: UInt32(maxInit),
                suppressNonSpeech: options.suppressNonSpeech ? 1 : 0, pad: 0
            )
        }

        let lastTok = genTokens[genTokens.count - 1]
        let lastWasTs = lastTok >= tsBegin
        let penultimateWasTs = genTokens.count < 2 || genTokens[genTokens.count - 2] >= tsBegin

        var allowText: UInt32 = 1
        let allowEot: UInt32 = 1
        var allowTs: UInt32 = 1

        if lastWasTs {
            if penultimateWasTs {
                allowTs = 0
            } else {
                allowText = 0
            }
        }

        var minAllowedTs = tsBegin
        if let lastTs = genTokens.last(where: { $0 >= tsBegin }) {
            minAllowedTs = (lastWasTs && !penultimateWasTs) ? lastTs : (lastTs + 1)
        }
        if minAllowedTs > maxAudioTs {
            allowTs = 0
        }

        return LogitsFilterRule(
            step0: 0, allowText: allowText, allowEot: allowEot, allowTs: allowTs,
            minTs: UInt32(min(nVocab, minAllowedTs)), maxTs: UInt32(maxAudioTs),
            suppressNonSpeech: options.suppressNonSpeech ? 1 : 0, pad: 0
        )
    }

    private func prefill(
        _ states: inout [PackDecodeState],
        detectLanguage: Bool,
        beamSize: Int,
        options: MetalWhisperDecodeOptions,
        slots: inout MetalWhisperKVSlotPool
    ) throws {
        let specials = model.tokenizer.specials
        let nVocab = model.hparams.nVocab
        let count = states.count
        var tokenIDs: [UInt32] = []
        var positions: [UInt32] = []
        var writeSlots: [UInt32] = []
        var seqStarts: [UInt32] = []
        var posInSeq: [UInt32] = []
        var crossGroups: [PackCrossGroup] = []
        var packSlots: [[UInt32]] = []
        var sotRows: [Int] = []
        var step0Rows: [Int] = []

        for state in states {
            var prefix: [Int] = []
            if !state.promptTokens.isEmpty {
                prefix.append(specials.prev)
                prefix.append(contentsOf: state.promptTokens)
            }
            let start = tokenIDs.count
            sotRows.append(start + prefix.count)
            prefix.append(specials.sot)
            if !detectLanguage {
                if specials.numLanguages > 0 {
                    prefix.append(state.languageID)
                    prefix.append(specials.transcribe)
                }
                step0Rows.append(start + prefix.count - 1)
            }

            var prefixSlots: [UInt32] = []
            prefixSlots.reserveCapacity(prefix.count + 2)
            for (pos, token) in prefix.enumerated() {
                let slot = try slots.takeSelfSlot(crossSlot: state.crossSlot)
                prefixSlots.append(slot)
                tokenIDs.append(UInt32(token))
                positions.append(UInt32(pos))
                writeSlots.append(slot)
                seqStarts.append(UInt32(start))
                posInSeq.append(UInt32(pos))
            }
            packSlots.append(prefixSlots)

            for row in stride(from: start, to: tokenIDs.count, by: 5) {
                crossGroups.append(PackCrossGroup(
                    crossSlot: state.crossSlot, rows: row..<min(row + 5, tokenIDs.count)
                ))
            }
        }

        let logitRows = sotRows + step0Rows
        try reserveSelfKV(for: slots)
        try runDecoderPrefill(
            tokenIDs: tokenIDs, positions: positions, writeSlots: writeSlots,
            seqStarts: seqStarts, posInSeq: posInSeq, crossGroups: crossGroups, logitRowIndices: logitRows
        )

        let logits = decLogitsF32Buf.contents().bindMemory(to: Float.self, capacity: max(maxDecodeB, logitRows.count) * nVocab)
        for i in states.indices {
            let row = logits.advanced(by: i * nVocab)
            states[i].noSpeechProb = softmaxProb(logits: row, count: nVocab, targetIndex: specials.noSpeech)
            if detectLanguage {
                let start = specials.sot + 1
                let end = min(nVocab, start + specials.numLanguages)
                states[i].languageID = (start..<end).max(by: { row[$0] < row[$1] }) ?? start
            }
        }

        let step0RowOffset: Int
        if detectLanguage {
            try feedSharedTokenStep(
                states: states, tokens: states.map { UInt32($0.languageID) }, packSlots: &packSlots, slots: &slots
            )
            try feedSharedTokenStep(
                states: states, tokens: Array(repeating: UInt32(specials.transcribe), count: count),
                packSlots: &packSlots, slots: &slots
            )
            step0RowOffset = 0
        } else {
            step0RowOffset = count
        }

        let rules = logitsRulesBuf.contents().bindMemory(to: LogitsFilterRule.self, capacity: maxDecodeB)
        for i in states.indices {
            rules[i] = makeFilterRule(genTokens: [], durationSeconds: states[i].durationSeconds, options: options)
        }
        try runGPUFilterOnly(B: count, rowByteOffset: step0RowOffset * nVocab * MemoryLayout<Float>.stride)

        let topTokens = topKTokensBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB * 8)
        let topLogProbs = topKLogProbsBuf.contents().bindMemory(to: Float.self, capacity: maxDecodeB * 8)
        let bestRawTs = bestRawTsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)

        for i in states.indices {
            var state = states[i]
            state.steps = 1
            state.stepsSinceFirstTopEOT = 0
            state.completed.removeAll(keepingCapacity: true)
            state.beams.removeAll(keepingCapacity: true)
            state.beams.reserveCapacity(beamSize)
            // Each pack consumes its own random stream across interleaved decode steps.
            state.rng = SplitMix64(seed: UInt64(bitPattern: Int64(state.temperature * 10000)) ^ 0x9E3779B97F4A7C15)
            let rawTs = Int(bestRawTs[i])
            let logProbs = logits.advanced(by: (step0RowOffset + i) * nVocab)

            for k in 0..<beamSize {
                let token: Int
                let logProb: Float
                if state.isBeamSearch {
                    token = Int(topTokens[i * 8 + k])
                    logProb = topLogProbs[i * 8 + k]
                    if !logProb.isFinite { continue }
                } else {
                    (token, logProb) = sampleFromLogProbs(
                        logProbs: logProbs, count: nVocab, temperature: state.temperature, rng: &state.rng
                    )
                }
                state.beams.append(BeamHyp(
                    genTokens: [token], genLogProbs: [logProb], genRawTs: [rawTs],
                    kvSlots: packSlots[i], sumLogProb: logProb, finished: token == specials.eot
                ))
            }
            if state.beams.isEmpty {
                state.beams.append(BeamHyp(
                    genTokens: [specials.eot], genLogProbs: [-1], genRawTs: [specials.timestampBegin],
                    kvSlots: packSlots[i], sumLogProb: -1, finished: true
                ))
            }
            states[i] = state
        }
    }

    private func advanceDecode(
        _ states: inout [PackDecodeState],
        beamSize: Int,
        maxSteps: Int,
        options: MetalWhisperDecodeOptions,
        slots: inout MetalWhisperKVSlotPool
    ) throws {
        let hp = model.hparams
        let specials = model.tokenizer.specials
        let tokens = decTokenIDsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let positions = decPositionsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let writeSlots = decWriteSlotsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let seqLens = decSeqLensBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let slotTable = decSlotTableBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB * hp.nTextCtx)
        let crossGroups = decCrossGroupsBuf.contents().bindMemory(to: PackCrossGroup.self, capacity: maxBatchPacks)
        let rules = logitsRulesBuf.contents().bindMemory(to: LogitsFilterRule.self, capacity: maxDecodeB)

        var beamIndices: [Int] = []
        beamIndices.reserveCapacity(maxDecodeB)
        var rowsByState: [Range<Int>] = []
        rowsByState.reserveCapacity(states.count)
        var numCrossGroups = 0
        for i in states.indices {
            let firstRow = beamIndices.count
            if states[i].steps < maxSteps {
                for b in states[i].beams.indices where !states[i].beams[b].finished {
                    let beam = states[i].beams[b]
                    let pos = beam.kvSlots.count
                    if pos >= hp.nTextCtx - 1 {
                        states[i].beams[b].finished = true
                        states[i].completed.append(states[i].beams[b])
                        continue
                    }
                    let slot = try slots.takeSelfSlot(crossSlot: states[i].crossSlot)
                    let row = beamIndices.count
                    tokens[row] = UInt32(beam.genTokens.last!)
                    positions[row] = UInt32(pos)
                    writeSlots[row] = slot
                    seqLens[row] = UInt32(pos + 1)
                    rules[row] = makeFilterRule(
                        genTokens: beam.genTokens, durationSeconds: states[i].durationSeconds, options: options
                    )
                    let tableRow = slotTable.advanced(by: row * hp.nTextCtx)
                    beam.kvSlots.withUnsafeBufferPointer { tableRow.update(from: $0.baseAddress!, count: pos) }
                    tableRow[pos] = slot
                    states[i].beams[b].kvSlots.append(slot)
                    beamIndices.append(b)
                }
            }

            let rows = firstRow..<beamIndices.count
            rowsByState.append(rows)
            if !rows.isEmpty {
                crossGroups[numCrossGroups] = PackCrossGroup(crossSlot: states[i].crossSlot, rows: rows)
                numCrossGroups += 1
            }
        }

        guard !beamIndices.isEmpty else { return }
        try reserveSelfKV(for: slots)
        try runDecoderStep(B: beamIndices.count, numCrossGroups: numCrossGroups, runGPUFilterTopK: true)

        let topTokens = topKTokensBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB * 8)
        let topLogProbs = topKLogProbsBuf.contents().bindMemory(to: Float.self, capacity: maxDecodeB * 8)
        let bestRawTs = bestRawTsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let logits = decLogitsF32Buf.contents().bindMemory(to: Float.self, capacity: maxDecodeB * hp.nVocab)
        for (i, rows) in rowsByState.enumerated() where !rows.isEmpty {
            states[i].steps += 1
            if states[i].isBeamSearch {
                updateBeamSearch(
                    &states[i], rows: rows, beamIndices: beamIndices, beamSize: beamSize, eot: specials.eot,
                    topTokens: topTokens, topLogProbs: topLogProbs, bestRawTs: bestRawTs
                )
            } else {
                let temperature = states[i].temperature
                for row in rows {
                    let (token, logProb) = sampleFromLogProbs(
                        logProbs: logits.advanced(by: row * hp.nVocab), count: hp.nVocab,
                        temperature: temperature, rng: &states[i].rng
                    )
                    let b = beamIndices[row]
                    states[i].beams[b].genTokens.append(token)
                    states[i].beams[b].genLogProbs.append(logProb)
                    states[i].beams[b].genRawTs.append(Int(bestRawTs[row]))
                    states[i].beams[b].sumLogProb += logProb
                    if token == specials.eot {
                        states[i].beams[b].finished = true
                        states[i].completed.append(states[i].beams[b])
                    }
                }
            }
        }
    }

    private func updateBeamSearch(
        _ state: inout PackDecodeState,
        rows: Range<Int>,
        beamIndices: [Int],
        beamSize: Int,
        eot: Int,
        topTokens: UnsafePointer<UInt32>,
        topLogProbs: UnsafePointer<Float>,
        bestRawTs: UnsafePointer<UInt32>
    ) {
        struct Candidate {
            let parent: Int
            let token: Int
            let logProb: Float
            let rawTs: Int
            let score: Float
        }
        var candidates: [Candidate] = []
        candidates.reserveCapacity(rows.count * 8)
        for row in rows {
            let parent = beamIndices[row]
            for k in 0..<min(8, beamSize + 2) {
                let logProb = topLogProbs[row * 8 + k]
                if !logProb.isFinite { break }
                candidates.append(Candidate(
                    parent: parent, token: Int(topTokens[row * 8 + k]), logProb: logProb,
                    rawTs: Int(bestRawTs[row]), score: state.beams[parent].sumLogProb + logProb
                ))
            }
        }

        candidates.sort { $0.score > $1.score }
        let bestScore = candidates.first?.score ?? -.infinity
        var next: [BeamHyp] = []
        next.reserveCapacity(beamSize)
        for (rank, candidate) in candidates.enumerated() {
            var child = state.beams[candidate.parent]
            child.genTokens.append(candidate.token)
            child.genLogProbs.append(candidate.logProb)
            child.genRawTs.append(candidate.rawTs)
            child.sumLogProb = candidate.score
            if candidate.token == eot {
                if rank < beamSize || candidate.score >= bestScore - 2.5 {
                    child.finished = true
                    state.completed.append(child)
                }
            } else if next.count < beamSize {
                next.append(child)
            }
            if next.count >= beamSize && state.completed.count >= beamSize { break }
        }

        if !state.completed.isEmpty { state.stepsSinceFirstTopEOT += 1 }
        state.beams = state.completed.count >= beamSize || state.stepsSinceFirstTopEOT >= 8 ? [] : next
    }

    private func feedSharedTokenStep(
        states: [PackDecodeState],
        tokens: [UInt32],
        packSlots: inout [[UInt32]],
        slots: inout MetalWhisperKVSlotPool
    ) throws {
        let hp = model.hparams
        let tokPtr = decTokenIDsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let posPtr = decPositionsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let wSlotPtr = decWriteSlotsBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let seqLenPtr = decSeqLensBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB)
        let slotTblPtr = decSlotTableBuf.contents().bindMemory(to: UInt32.self, capacity: maxDecodeB * hp.nTextCtx)
        let crossGrpPtr = decCrossGroupsBuf.contents().bindMemory(to: PackCrossGroup.self, capacity: maxBatchPacks)

        for i in states.indices {
            let pos = packSlots[i].count
            let slot = try slots.takeSelfSlot(crossSlot: states[i].crossSlot)
            tokPtr[i] = tokens[i]
            posPtr[i] = UInt32(pos)
            wSlotPtr[i] = slot
            seqLenPtr[i] = UInt32(pos + 1)
            crossGrpPtr[i] = PackCrossGroup(crossSlot: states[i].crossSlot, rows: i..<(i + 1))
            let row = slotTblPtr.advanced(by: i * hp.nTextCtx)
            packSlots[i].withUnsafeBufferPointer { row.update(from: $0.baseAddress!, count: pos) }
            row[pos] = slot
            packSlots[i].append(slot)
        }
        try reserveSelfKV(for: slots)
        try runDecoderStep(B: states.count, numCrossGroups: states.count, runGPUFilterTopK: false)
    }

    // MARK: - Helper Math & Sampling

    private func softmaxProb(logits: UnsafePointer<Float>, count: Int, targetIndex: Int) -> Double {
        var maxVal: Float = -Float.infinity
        vDSP_maxv(logits, 1, &maxVal, vDSP_Length(count))
        guard maxVal.isFinite else { return 0.0 }
        var sumExp: Double = 0
        for i in 0..<count {
            sumExp += exp(Double(logits[i] - maxVal))
        }
        return exp(Double(logits[targetIndex] - maxVal)) / max(sumExp, 1e-30)
    }

    private func sampleFromLogProbs(
        logProbs: UnsafePointer<Float>,
        count: Int,
        temperature: Float,
        rng: inout SplitMix64
    ) -> (Int, Float) {
        let invT = 1.0 / max(temperature, 1e-5)
        var maxScaled: Float = -Float.infinity
        for i in 0..<count {
            let v = logProbs[i]
            if v.isFinite {
                let s = v * invT
                if s > maxScaled { maxScaled = s }
            }
        }
        guard maxScaled.isFinite else { return (model.tokenizer.specials.eot, -10.0) }
        var sumExp: Double = 0
        for i in 0..<count {
            let v = logProbs[i]
            if v.isFinite { sumExp += exp(Double(v * invT - maxScaled)) }
        }
        let u = rng.nextUnitDouble() * sumExp
        var cum: Double = 0
        var chosen = model.tokenizer.specials.eot
        for i in 0..<count {
            let v = logProbs[i]
            if !v.isFinite { continue }
            cum += exp(Double(v * invT - maxScaled))
            if cum >= u {
                chosen = i
                break
            }
        }
        return (chosen, logProbs[chosen])
    }

    // MARK: - Token Sequence -> Segments & Quality Metrics

    private func buildOutcome(
        beam: BeamHyp,
        durationSeconds: Double,
        noSpeechProb: Double,
        temperatureUsed: Float
    ) -> PackDecodeOutcome {
        let specials = model.tokenizer.specials
        let tsBegin = specials.timestampBegin
        let tokenizer = model.tokenizer

        var toks = beam.genTokens
        var lps = beam.genLogProbs
        var rawTsArr = beam.genRawTs
        if toks.last == specials.eot {
            toks.removeLast()
            lps.removeLast()
            if !rawTsArr.isEmpty { rawTsArr.removeLast() }
        }

        var segments: [MetalWhisperSegment] = []
        var curStart: Double = 0.0
        var curTextTokenIDs: [Int] = []
        var curTokenEntries: [(id: Int, lp: Float, rawTs: Double)] = []

        // Pool-Adjacent-Violators (Isotonic Regression) projects noisy per-token raw
        // acoustic timestamp predictions onto the optimal non-decreasing curve inside [segStart, segEnd].
        func isotonicTimestamps(_ raw: [Double], lo: Double, hi: Double) -> [Double] {
            let n = raw.count
            guard n > 0 else { return [] }
            struct Block { var sum: Double; var count: Int; var startIdx: Int }
            var stack: [Block] = []
            stack.reserveCapacity(n)
            for i in 0..<n {
                let v = max(lo, min(hi, raw[i]))
                stack.append(Block(sum: v, count: 1, startIdx: i))
                while stack.count >= 2 {
                    let b2 = stack[stack.count - 1]
                    let b1 = stack[stack.count - 2]
                    if b1.sum / Double(b1.count) > b2.sum / Double(b2.count) {
                        stack.removeLast()
                        stack[stack.count - 1].sum += b2.sum
                        stack[stack.count - 1].count += b2.count
                    } else {
                        break
                    }
                }
            }
            var out = [Double](repeating: lo, count: n)
            for blk in stack {
                let mean = blk.sum / Double(blk.count)
                for j in blk.startIdx..<(blk.startIdx + blk.count) {
                    out[j] = mean
                }
            }
            return out
        }

        func flushSegment(endTime: Double) {
            guard !curTextTokenIDs.isEmpty else { return }
            let segText = tokenizer.decode(curTextTokenIDs)
            let clampedStart = max(0.0, min(durationSeconds, curStart))
            let clampedEnd = max(clampedStart, min(durationSeconds, endTime))
            let segSpan = max(0.02, clampedEnd - clampedStart)

            // Blend Whisper's monotonic acoustic token timestamps with smooth linear distribution
            // across flat plateaus so every token has strictly increasing, acoustically anchored timing.
            let n = curTokenEntries.count
            let isoRaw = isotonicTimestamps(curTokenEntries.map(\.rawTs), lo: clampedStart, hi: clampedEnd)
            let hasAcousticSpread = (isoRaw.last ?? clampedStart) - (isoRaw.first ?? clampedStart) > 0.10
            var tokenEnds = [Double](repeating: clampedEnd, count: n)
            if hasAcousticSpread && n > 1 {
                // Smooth flat plateaus linearly between adjacent distinct anchor points
                var smoothed = isoRaw
                var i = 0
                while i < n {
                    var j = i + 1
                    while j < n && abs(isoRaw[j] - isoRaw[i]) < 0.005 { j += 1 }
                    if j - i > 1 {
                        let loAnchor = (i == 0) ? clampedStart : smoothed[i - 1]
                        let hiAnchor = (j == n) ? clampedEnd : isoRaw[j]
                        let count = Double(j - i)
                        for k in i..<j {
                            let frac = Double(k - i + 1) / (count + (j == n ? 0.0 : 1.0))
                            smoothed[k] = loAnchor + (hiAnchor - loAnchor) * frac
                        }
                    }
                    i = j
                }
                for idx in 0..<n {
                    let linEnd = clampedStart + segSpan * Double(idx + 1) / Double(n)
                    let blended = 0.65 * smoothed[idx] + 0.35 * linEnd
                    let prev = (idx == 0) ? clampedStart : tokenEnds[idx - 1]
                    tokenEnds[idx] = max(prev, min(clampedEnd, blended))
                }
                tokenEnds[n - 1] = clampedEnd
            } else {
                for idx in 0..<n {
                    tokenEnds[idx] = clampedStart + segSpan * Double(idx + 1) / Double(n)
                }
            }

            // Merge multi-byte UTF-8 BPE fragments (e.g., Japanese kanji split across 2-3 BPE tokens)
            // into valid complete UTF-8 token spans so zero U+FFFD replacement characters ever appear!
            var outTokens: [MetalWhisperToken] = []
            outTokens.reserveCapacity(n)
            var pendingBytes: [UInt8] = []
            var pendingStart: Double = clampedStart
            var pendingLogProbSum: Double = 0
            var pendingCount = 0
            var pendingFirstID = 0

            for (idx, entry) in curTokenEntries.enumerated() {
                let t0 = (idx == 0) ? clampedStart : tokenEnds[idx - 1]
                let t1 = max(t0, tokenEnds[idx])

                let b = (entry.id >= 0 && entry.id < tokenizer.tokenBytes.count) ? tokenizer.tokenBytes[entry.id] : []
                if pendingBytes.isEmpty {
                    pendingStart = t0
                    pendingFirstID = entry.id
                }
                pendingBytes.append(contentsOf: b)
                pendingLogProbSum += Double(entry.lp)
                pendingCount += 1

                if let validText = String(bytes: pendingBytes, encoding: .utf8) {
                    outTokens.append(MetalWhisperToken(
                        id: pendingFirstID,
                        text: validText,
                        start: pendingStart,
                        end: t1,
                        probability: exp(pendingLogProbSum / Double(max(1, pendingCount))),
                        isTimestamp: false
                    ))
                    pendingBytes.removeAll(keepingCapacity: true)
                    pendingLogProbSum = 0
                    pendingCount = 0
                }
            }
            if !pendingBytes.isEmpty {
                let fallbackText = String(decoding: pendingBytes, as: UTF8.self)
                outTokens.append(MetalWhisperToken(
                    id: pendingFirstID,
                    text: fallbackText,
                    start: pendingStart,
                    end: clampedEnd,
                    probability: exp(pendingLogProbSum / Double(max(1, pendingCount))),
                    isTimestamp: false
                ))
            }

            let meanLP = curTokenEntries.reduce(0.0) { $0 + Double($1.lp) } / Double(max(1, n))
            segments.append(MetalWhisperSegment(
                start: clampedStart,
                end: clampedEnd,
                text: segText,
                avgLogProb: meanLP,
                noSpeechProb: noSpeechProb,
                tokens: outTokens
            ))
            curTextTokenIDs.removeAll(keepingCapacity: true)
            curTokenEntries.removeAll(keepingCapacity: true)
        }

        var lastTimestamp: Double = 0.0
        for i in 0..<toks.count {
            let id = toks[i]
            let lp = lps[i]
            let rTs = (i < rawTsArr.count) ? Double(rawTsArr[i] - tsBegin) * 0.02 : lastTimestamp
            if id >= tsBegin {
                let ts = Double(id - tsBegin) * 0.02
                if curTextTokenIDs.isEmpty {
                    curStart = ts
                } else {
                    flushSegment(endTime: ts)
                    curStart = ts
                }
                lastTimestamp = ts
            } else if id < specials.eot {
                curTextTokenIDs.append(id)
                curTokenEntries.append((id, lp, rTs))
            }
        }
        if !curTextTokenIDs.isEmpty {
            flushSegment(endTime: max(lastTimestamp + 0.02, durationSeconds))
        }

        let fullText = segments.map(\.text).joined()
        let avgLP = Double(beam.sumLogProb) / Double(max(1, beam.genTokens.count))
        let compRatio = Self.zlibCompressionRatio(fullText)

        return PackDecodeOutcome(
            text: fullText,
            noSpeechProb: noSpeechProb,
            avgLogProb: avgLP,
            compressionRatio: compRatio,
            temperatureUsed: temperatureUsed,
            segments: segments
        )
    }

    static func zlibCompressionRatio(_ text: String) -> Double {
        let bytes = Array(text.utf8)
        guard bytes.count >= 16 else { return 1.0 }
        let dstCap = max(64, bytes.count * 2)
        var dst = [UInt8](repeating: 0, count: dstCap)
        let compressedSize = bytes.withUnsafeBufferPointer { src in
            dst.withUnsafeMutableBufferPointer { out in
                compression_encode_buffer(
                    out.baseAddress!, dstCap,
                    src.baseAddress!, bytes.count,
                    nil,
                    COMPRESSION_ZLIB
                )
            }
        }
        guard compressedSize > 0 else { return 1.0 }
        return Double(bytes.count) / Double(compressedSize)
    }
}

#if METAL_WHISPER_DECODER_BENCHMARK
struct MetalWhisperDecoderStepSample {
    let gpuMs: Double
    let encodeMs: Double
    let elapsedMs: Double
    let topTokens: [UInt32]
    let rawLogits: [Float]
    let lmInput: [Float16]
}

extension MetalWhisperEngine {
    var benchmarkCrossKVBytes: Int {
        guard let cache = crossKV else { return 0 }
        if crossKVFormat == .q8Packed { return cache.k.length + (cache.kScale?.length ?? 0) }
        return cache.k.length + cache.v.length + (cache.kScale?.length ?? 0) + (cache.vScale?.length ?? 0)
    }

    func copyBenchmarkCrossQuery(beams: Int) throws -> MTLBuffer {
        let bytes = beams * model.hparams.nTextState * MemoryLayout<Float16>.stride
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared),
              let command = commandQueue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
            throw MetalWhisperError.metalUnavailable
        }
        blit.copy(from: decCrossQBuf, sourceOffset: 0, to: buffer, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        return buffer
    }

    func benchmarkCrossAttention(beams: Int, query: MTLBuffer, output: MTLBuffer) throws -> Double {
        guard let cache = crossKV, beams > 0, beams.isMultiple(of: maxBeamsPerPack),
              let command = commandQueue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw MetalWhisperError.invalidAudioInput
        }
        let stride = beams * model.hparams.nTextState * MemoryLayout<Float16>.stride
        for layer in 0..<model.hparams.nTextLayer {
            encodeCrossAttention(
                encoder, cache: cache, layer: layer, groups: beams / maxBeamsPerPack,
                query: query, output: output, outputOffset: layer * stride
            )
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        return (command.gpuEndTime - command.gpuStartTime) * 1_000
    }

    func prepareDecoderBenchmark(packs: [[Float]], history: [Int], stepTokens: [Int]) throws {
        guard tensorDecoder != nil, !packs.isEmpty, packs.count <= maxBatchPacks,
              !history.isEmpty, history.count < model.hparams.nTextCtx,
              stepTokens.count == maxBeamsPerPack else {
            throw MetalWhisperError.invalidAudioInput
        }
        let hp = model.hparams
        let bytesPerToken = hp.nTextLayer * hp.nTextState * MemoryLayout<Float16>.stride
        let historySlots = packs.count * history.count
        let rows = packs.count * maxBeamsPerPack
        try reserveCrossKV(slots: packs.count)
        selfKV = try kvBuffers(
            reusing: nil, slots: historySlots + rows, bytesPerSlot: bytesPerToken, label: "benchmark Self-KV"
        )

        let melBytes = model.melExtractor.frameCount * hp.nMels * MemoryLayout<Float>.stride
        for first in stride(from: 0, to: packs.count, by: encoderPacksPerPass) {
            let packRange = first..<min(first + encoderPacksPerPass, packs.count)
            for pack in packRange {
                let mel = model.melExtractor.extractTimeMajor(samples: packs[pack])
                melInBuf.contents().advanced(by: (pack - first) * melBytes).copyMemory(from: mel, byteCount: melBytes)
            }
            try encodeChunkIntoCrossKV(slots: Array(packRange))
        }

        let prefixPositions = Array((0..<history.count).map(UInt32.init))
        let groups = packs.indices.flatMap { pack in
            let start = pack * history.count
            return stride(from: start, to: start + history.count, by: 5).map { first in
                PackCrossGroup(crossSlot: pack, rows: first..<min(first + 5, start + history.count))
            }
        }
        try runDecoderPrefill(
            tokenIDs: packs.flatMap { _ in history.map(UInt32.init) },
            positions: packs.flatMap { _ in prefixPositions },
            writeSlots: (0..<historySlots).map(UInt32.init),
            seqStarts: packs.indices.flatMap { pack in Array(repeating: UInt32(pack * history.count), count: history.count) },
            posInSeq: packs.flatMap { _ in prefixPositions }, crossGroups: groups, logitRowIndices: []
        )

        let tokens = decTokenIDsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let positions = decPositionsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let writeSlots = decWriteSlotsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let seqLens = decSeqLensBuf.contents().assumingMemoryBound(to: UInt32.self)
        let table = decSlotTableBuf.contents().assumingMemoryBound(to: UInt32.self)
        let crossGroups = decCrossGroupsBuf.contents().assumingMemoryBound(to: PackCrossGroup.self)
        let rules = logitsRulesBuf.contents().assumingMemoryBound(to: LogitsFilterRule.self)
        let options = MetalWhisperDecodeOptions()
        for pack in packs.indices {
            let first = pack * maxBeamsPerPack
            crossGroups[pack] = PackCrossGroup(crossSlot: pack, rows: first..<(first + maxBeamsPerPack))
            for (beam, token) in stepTokens.enumerated() {
                let row = first + beam
                let slot = UInt32(historySlots + row)
                tokens[row] = UInt32(token)
                positions[row] = UInt32(history.count)
                writeSlots[row] = slot
                seqLens[row] = UInt32(history.count + 1)
                for i in 0..<history.count {
                    table[row * hp.nTextCtx + i] = UInt32(pack * history.count + i)
                }
                table[row * hp.nTextCtx + history.count] = slot
                rules[row] = makeFilterRule(genTokens: [token], durationSeconds: 30, options: options)
            }
        }
    }

    func benchmarkDecoderStep(
        beams: Int, backend: MetalWhisperDecoderBackend, capture: Bool = false,
        rawLogits: Bool = false, headOnly: Bool = false
    ) throws -> MetalWhisperDecoderStepSample {
        let hp = model.hparams
        let hiddenBytes = beams * hp.nTextState * MemoryLayout<Float16>.stride
        let lmInput: MTLBuffer?
        if rawLogits {
            guard let buffer = device.makeBuffer(length: hiddenBytes, options: .storageModeShared) else {
                throw MetalWhisperError.bufferAllocationFailed("benchmark Final LN")
            }
            lmInput = buffer
        } else {
            lmInput = nil
        }
        let start = CFAbsoluteTimeGetCurrent()
        guard let cb = commandQueue.makeCommandBuffer() else { throw MetalWhisperError.metalUnavailable }
        if headOnly {
            if backend == .mps {
                encodeDecoderLMHead(cb: cb, B: beams)
            } else {
                guard let encoder = cb.makeComputeCommandEncoder() else { throw MetalWhisperError.metalUnavailable }
                encodeDecoderLMHead(cb: cb, B: beams, encoder: encoder)
                encoder.endEncoding()
            }
        } else {
            try encodeDecoderStep(
                cb: cb, B: beams, numCrossGroups: beams / maxBeamsPerPack,
                backend: backend, runGPUFilterTopK: !rawLogits
            )
        }
        if let lmInput {
            guard let blit = cb.makeBlitCommandEncoder() else { throw MetalWhisperError.metalUnavailable }
            blit.copy(from: decNormBuf, sourceOffset: 0, to: lmInput, destinationOffset: 0, size: hiddenBytes)
            blit.endEncoding()
        }
        let submit = CFAbsoluteTimeGetCurrent()
        cb.commit()
        cb.waitUntilCompleted()
        let end = CFAbsoluteTimeGetCurrent()
        if let error = cb.error { throw error }
        guard cb.gpuEndTime > cb.gpuStartTime else {
            throw MetalWhisperError.shaderCompilationFailed("Missing decoder benchmark GPU timestamps")
        }

        let topTokens = capture && !rawLogits && !headOnly ? Array(UnsafeBufferPointer(
            start: topKTokensBuf.contents().assumingMemoryBound(to: UInt32.self), count: beams * 8
        )) : []
        let logits = rawLogits ? Array(UnsafeBufferPointer(
            start: decLogitsF32Buf.contents().assumingMemoryBound(to: Float.self), count: beams * hp.nVocab
        )) : []
        let hidden = lmInput.map {
            Array(UnsafeBufferPointer(start: $0.contents().assumingMemoryBound(to: Float16.self), count: beams * hp.nTextState))
        } ?? []
        return MetalWhisperDecoderStepSample(
            gpuMs: (cb.gpuEndTime - cb.gpuStartTime) * 1_000,
            encodeMs: (submit - start) * 1_000, elapsedMs: (end - start) * 1_000,
            topTokens: topTokens, rawLogits: logits, lmInput: hidden
        )
    }
}
#endif

private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 0x123456789ABCDEF : seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextUnitDouble() -> Double {
        Double(next() >> 11) * (1.0 / Double(1 << 53))
    }
}

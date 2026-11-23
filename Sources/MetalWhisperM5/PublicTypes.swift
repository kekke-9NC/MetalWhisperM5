import Foundation

public enum MetalWhisperError: LocalizedError, Sendable {
    case metalUnavailable
    case invalidModelHeader(String)
    case missingTensor(String)
    case unsupportedQuantization(Int32, String)
    case shaderCompilationFailed(String)
    case bufferAllocationFailed(String)
    case invalidAudioInput
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            "An Apple Metal GPU is unavailable."
        case .invalidModelHeader(let detail):
            "Could not read the Whisper GGML model: \(detail)"
        case .missingTensor(let name):
            "Required model tensor is missing: \(name)"
        case .unsupportedQuantization(let type, let name):
            "Unsupported GGML tensor type \(type) for \(name)."
        case .shaderCompilationFailed(let detail):
            "Metal shader compilation failed: \(detail)"
        case .bufferAllocationFailed(let name):
            "Could not allocate a Metal buffer for \(name)."
        case .invalidAudioInput:
            "The audio input is empty or invalid."
        case .invalidConfiguration(let detail):
            "Invalid transcription configuration: \(detail)"
        }
    }
}

public enum MetalWhisperEncoderBackend: String, Sendable, Codable {
    case auto, cooperativeAttention, mps

    /// Experimental Metal tensor paths. Automatic selection uses the cooperative path.
    case tensorAttention, tensor
}

public enum MetalWhisperDecoderBackend: String, Sendable, Codable {
    case auto, mps, mpsTensorHead

    /// Experimental Metal tensor paths for decoder layers.
    case tensorF16, tensorQ5

    var hasTensorLayers: Bool { self == .tensorF16 || self == .tensorQ5 }
}

public enum MetalWhisperCrossKVFormat: String, Sendable, Codable {
    case auto, q8Packed, f16

    /// Experimental variants used for profiling individual projections.
    case q8, q8Keys, q8Values

    var quantizesKeys: Bool { self == .q8 || self == .q8Keys || self == .q8Packed }
    var quantizesValues: Bool { self == .q8 || self == .q8Values || self == .q8Packed }
}

public struct MetalWhisperConfiguration: Sendable {
    public var encoderBackend: MetalWhisperEncoderBackend
    public var decoderBackend: MetalWhisperDecoderBackend
    public var crossKVFormat: MetalWhisperCrossKVFormat
    /// The in-flight pack limit. A nil value uses the GPU's recommended working set.
    public var maxBatchPacks: Int?
    public var prepareDecoderAhead: Bool
    public var enableProfiling: Bool

    public init(
        encoderBackend: MetalWhisperEncoderBackend = .auto,
        decoderBackend: MetalWhisperDecoderBackend = .auto,
        crossKVFormat: MetalWhisperCrossKVFormat = .auto,
        maxBatchPacks: Int? = nil,
        prepareDecoderAhead: Bool = true,
        enableProfiling: Bool = false
    ) {
        self.encoderBackend = encoderBackend
        self.decoderBackend = decoderBackend
        self.crossKVFormat = crossKVFormat
        self.maxBatchPacks = maxBatchPacks
        self.prepareDecoderAhead = prepareDecoderAhead
        self.enableProfiling = enableProfiling
    }
}

public struct WhisperHParams: Sendable {
    public let nVocab: Int
    public let nAudioCtx: Int
    public let nAudioState: Int
    public let nAudioHead: Int
    public let nAudioLayer: Int
    public let nTextCtx: Int
    public let nTextState: Int
    public let nTextHead: Int
    public let nTextLayer: Int
    public let nMels: Int
    public let ftype: Int

    public var headDim: Int { nAudioState / nAudioHead }
    public var nFFN: Int { nAudioState * 4 }
    public var isMultilingual: Bool { nVocab >= 51_865 }
}

/// Audio and context for one model window. Run the sample provider on demand.
public struct MetalWhisperPackInput: Sendable {
    public let durationSeconds: Double
    public let initialPrompt: String
    public let emptySpeechRecoveryThreshold: Double?
    public let makeSamples: @Sendable () -> [Float]

    public init(
        durationSeconds: Double,
        initialPrompt: String = "",
        emptySpeechRecoveryThreshold: Double? = nil,
        makeSamples: @escaping @Sendable () -> [Float]
    ) {
        self.durationSeconds = durationSeconds
        self.initialPrompt = initialPrompt
        self.emptySpeechRecoveryThreshold = emptySpeechRecoveryThreshold
        self.makeSamples = makeSamples
    }

    public init(
        samples: [Float],
        durationSeconds: Double,
        initialPrompt: String = "",
        emptySpeechRecoveryThreshold: Double? = nil
    ) {
        self.init(
            durationSeconds: durationSeconds,
            initialPrompt: initialPrompt,
            emptySpeechRecoveryThreshold: emptySpeechRecoveryThreshold,
            makeSamples: { samples }
        )
    }
}

public struct MetalWhisperDecodeOptions: Sendable {
    public var languageCode: String
    public var beamSize: Int
    public var temperatureSchedule: [Float]
    public var compressionRatioThreshold: Double
    public var logprobThreshold: Double
    public var noSpeechThreshold: Double
    public var maxInitialTimestamp: Double
    public var suppressNonSpeech: Bool
    public var maxDecodeTokens: Int
    /// Prompt for the high-level APIs. Batch inputs carry their own per-pack prompts.
    public var initialPrompt: String

    public init(
        languageCode: String = "auto",
        beamSize: Int = 5,
        temperatureSchedule: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compressionRatioThreshold: Double = 2.4,
        logprobThreshold: Double = -1.0,
        noSpeechThreshold: Double = 0.6,
        maxInitialTimestamp: Double = 1.0,
        suppressNonSpeech: Bool = true,
        maxDecodeTokens: Int = 224,
        initialPrompt: String = ""
    ) {
        self.languageCode = languageCode
        self.beamSize = beamSize
        self.temperatureSchedule = temperatureSchedule
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logprobThreshold = logprobThreshold
        self.noSpeechThreshold = noSpeechThreshold
        self.maxInitialTimestamp = maxInitialTimestamp
        self.suppressNonSpeech = suppressNonSpeech
        self.maxDecodeTokens = maxDecodeTokens
        self.initialPrompt = initialPrompt
    }
}

public struct MetalWhisperToken: Sendable, Codable, Equatable {
    public let id: Int
    public let text: String
    public let start: Double
    public let end: Double
    public let probability: Double
    public let isTimestamp: Bool

    public init(id: Int, text: String, start: Double, end: Double, probability: Double, isTimestamp: Bool) {
        self.id = id
        self.text = text
        self.start = start
        self.end = end
        self.probability = probability
        self.isTimestamp = isTimestamp
    }
}

/// Timestamps are seconds relative to the transcription's source audio, or the input pack for a batch result.
public struct MetalWhisperSegment: Sendable, Codable, Equatable {
    public let start: Double
    public let end: Double
    public let text: String
    public let avgLogProb: Double
    public let noSpeechProb: Double
    public let tokens: [MetalWhisperToken]

    public init(
        start: Double, end: Double, text: String, avgLogProb: Double,
        noSpeechProb: Double, tokens: [MetalWhisperToken]
    ) {
        self.start = start
        self.end = end
        self.text = text
        self.avgLogProb = avgLogProb
        self.noSpeechProb = noSpeechProb
        self.tokens = tokens
    }
}

public struct MetalWhisperDecodeResult: Sendable, Codable {
    public let text: String
    public let detectedLanguage: String
    public let noSpeechProb: Double
    public let avgLogProb: Double
    public let compressionRatio: Double
    public let temperatureUsed: Float
    public let segments: [MetalWhisperSegment]
    public let melMs: Double
    public let encodeMs: Double
    public let decodeMs: Double
}

public struct MetalWhisperTranscription: Sendable, Codable {
    public let text: String
    public let detectedLanguage: String
    public let durationSeconds: Double
    public let segments: [MetalWhisperSegment]
    public let profile: MetalWhisperProfileReport?

    public var tokens: [MetalWhisperToken] { segments.flatMap(\.tokens) }

    public init(
        text: String, detectedLanguage: String, durationSeconds: Double,
        segments: [MetalWhisperSegment], profile: MetalWhisperProfileReport? = nil
    ) {
        self.text = text
        self.detectedLanguage = detectedLanguage
        self.durationSeconds = durationSeconds
        self.segments = segments
        self.profile = profile
    }
}

/// Per-transcription timings in milliseconds. CPU and GPU phases can overlap.
public struct MetalWhisperProfileReport: Sendable, Codable {
    public let initMs: Double
    public let audioLoadMs: Double
    public let totalMs: Double
    public let melWaitMs: Double
    public let encodedPacks: Int
    public let encoderBackend: MetalWhisperEncoderBackend
    public let encoderPacksPerPass: Int
    public let encodeTotalMs: Double
    public let encoderGpuMs: Double
    public let convGpuMs: Double?
    public let flashGpuMs: Double?
    public let encoderGemmAndEpilogueGpuMs: Double?
    public let crossKVGpuMs: Double?
    public let prefillTotalMs: Double
    public let decoderBackend: MetalWhisperDecoderBackend
    public let crossKVFormat: MetalWhisperCrossKVFormat
    public let maxBatchPacks: Int
    public let decodeStepTotalMs: Double
    public let decodeSteps: Int
    public let decodeAheadHits: Int
    public let decodeAheadPrepared: Int
    public let decodeAheadCpuMs: Double
    public let averageDecodeRows: Double
    public let peakSelfKVSlots: Int
    public let peakUsedSelfKVSlots: Int
    public let decodeStepCpuEncodeMs: Double
    public let decodeStepSubmitMs: Double
    public let decodeStepGpuWaitMs: Double
    public let decodeStepGpuMs: Double
}

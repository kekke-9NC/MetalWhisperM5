import Foundation
import MetalWhisperM5

enum MetalWhisperCommand {
    static let version = "metal-whisper 0.1.0"
    static let usage = """
    Usage: metal-whisper --model <path> --audio <path> [options]

      --model <path>                    Whisper GGML .bin model
                                       Defaults to METAL_WHISPER_MODEL_PATH
      --audio <path>                    WAV, M4A, MP3, FLAC, or another AVAudioFile format
      --language <code>                 Language code or auto (default)
      --prompt <text>                   Initial prompt for decoding
      --beam-size <1...5>                Beam size (default: 5)
      --backend auto|cooperative|mps    Encoder backend (default: auto)
                                       mps also forces the MPS decoder
      --cross-kv auto|q8|f16             Cross-attention KV format (default: auto)
      --max-batch <count>                Maximum in-flight audio windows (default: automatic)
      --format text|json|srt|vtt         Transcript format (default: text)
      --output <path>                   Write the transcript to a file
                                       Defaults to standard output
      --profile                         Write CPU and GPU timings as JSON to standard error
      --version                         Print the version
      --help, -h                        Show usage
    """

    static func run(arguments: [String]) -> Int32 {
        do {
            let command = try Arguments.parse(arguments, environment: ProcessInfo.processInfo.environment)
            switch command {
            case .help:
                print(usage)
            case .version:
                print(version)
            case .transcribe(let invocation):
                let engine = try MetalWhisperEngine(
                    modelURL: invocation.modelURL, configuration: invocation.configuration
                )
                let result = try engine.transcribe(audioURL: invocation.audioURL, options: invocation.decodeOptions)
                let data = try invocation.format.render(result)
                if let outputURL = invocation.outputURL {
                    try data.write(to: outputURL, options: .atomic)
                } else {
                    try FileHandle.standardOutput.write(contentsOf: data)
                }
                if let profile = result.profile {
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.sortedKeys]
                    let json = String(decoding: try encoder.encode(profile), as: UTF8.self)
                    try FileHandle.standardError.write(contentsOf: Data("MetalWhisper profile \(json)\n".utf8))
                }
            }
            return 0
        } catch let error as Arguments.Error {
            let message = "metal-whisper: \(error.localizedDescription)\n\n\(usage)\n"
            FileHandle.standardError.write(Data(message.utf8))
            return 2
        } catch {
            FileHandle.standardError.write(Data("metal-whisper: \(error.localizedDescription)\n".utf8))
            return 1
        }
    }
}

enum Arguments {
    struct Error: LocalizedError {
        let errorDescription: String?

        init(_ message: String) {
            errorDescription = message
        }
    }

    struct Invocation {
        let modelURL: URL
        let audioURL: URL
        let outputURL: URL?
        let format: OutputFormat
        let configuration: MetalWhisperConfiguration
        let decodeOptions: MetalWhisperDecodeOptions
    }

    case help
    case version
    case transcribe(Invocation)

    static func parse(_ arguments: [String], environment: [String: String]) throws -> Arguments {
        guard !arguments.isEmpty else { return .help }
        var modelPath = environment["METAL_WHISPER_MODEL_PATH"]
        var audioPath: String?
        var outputPath: String?
        var format: OutputFormat = .text
        var configuration = MetalWhisperConfiguration()
        var options = MetalWhisperDecodeOptions()
        var iterator = arguments.makeIterator()

        func value(for option: String, allowOptionPrefix: Bool = false) throws -> String {
            guard let value = iterator.next(), allowOptionPrefix || !value.hasPrefix("--") else {
                throw Error("Missing value for \(option).")
            }
            return value
        }

        while let option = iterator.next() {
            switch option {
            case "--help", "-h":
                return .help
            case "--version":
                return .version
            case "--model":
                modelPath = try value(for: option)
            case "--audio":
                audioPath = try value(for: option)
            case "--output":
                outputPath = try value(for: option)
            case "--language":
                options.languageCode = try value(for: option).lowercased()
                guard !options.languageCode.isEmpty else { throw Error("Expected a language code or auto.") }
            case "--prompt":
                options.initialPrompt = try value(for: option, allowOptionPrefix: true)
            case "--beam-size":
                guard let count = Int(try value(for: option)), (1...5).contains(count) else {
                    throw Error("--beam-size must be an integer between 1 and 5.")
                }
                options.beamSize = count
            case "--backend":
                switch try value(for: option) {
                case "auto":
                    configuration.encoderBackend = .auto
                    configuration.decoderBackend = .auto
                case "cooperative":
                    configuration.encoderBackend = .cooperativeAttention
                    configuration.decoderBackend = .auto
                case "mps":
                    configuration.encoderBackend = .mps
                    configuration.decoderBackend = .mps
                default:
                    throw Error("--backend expects auto, cooperative, or mps.")
                }
            case "--cross-kv":
                switch try value(for: option) {
                case "auto": configuration.crossKVFormat = .auto
                case "q8": configuration.crossKVFormat = .q8Packed
                case "f16": configuration.crossKVFormat = .f16
                default: throw Error("--cross-kv expects auto, q8, or f16.")
                }
            case "--max-batch":
                guard let count = Int(try value(for: option)), count > 0 else {
                    throw Error("--max-batch must be a positive integer.")
                }
                configuration.maxBatchPacks = count
            case "--format":
                guard let parsed = OutputFormat(rawValue: try value(for: option)) else {
                    throw Error("--format expects text, json, srt, or vtt.")
                }
                format = parsed
            case "--profile":
                configuration.enableProfiling = true
            default:
                throw Error("Unknown option \(option).")
            }
        }

        guard let modelPath, !modelPath.isEmpty else {
            throw Error("Specify --model or set METAL_WHISPER_MODEL_PATH.")
        }
        guard let audioPath, !audioPath.isEmpty else { throw Error("Specify --audio.") }
        if outputPath?.isEmpty == true { throw Error("--output expects a file path.") }

        let modelURL = URL(fileURLWithPath: modelPath)
        let audioURL = URL(fileURLWithPath: audioPath)
        let outputURL = outputPath.map { URL(fileURLWithPath: $0) }
        if let outputURL {
            let destination = outputURL.standardizedFileURL.resolvingSymlinksInPath()
            guard destination != modelURL.standardizedFileURL.resolvingSymlinksInPath(),
                  destination != audioURL.standardizedFileURL.resolvingSymlinksInPath() else {
                throw Error("--output must use a different file from the model and audio.")
            }
        }

        return .transcribe(Invocation(
            modelURL: modelURL, audioURL: audioURL, outputURL: outputURL, format: format,
            configuration: configuration, decodeOptions: options
        ))
    }
}

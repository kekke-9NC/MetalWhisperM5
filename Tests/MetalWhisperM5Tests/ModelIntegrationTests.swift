import CryptoKit
import Foundation
import XCTest
@testable import MetalWhisperM5

final class ModelIntegrationTests: XCTestCase {
    private static let sampleSHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
    private static let sampleURL = URL(string:
        "https://raw.githubusercontent.com/ggml-org/whisper.cpp/2eeeba56e9edd762b4b38467bab96c2517163158/samples/jfk.wav"
    )!

    @MainActor
    func testPublicSpeechAutoFallbackBatchAndTimestamps() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = environment["METAL_WHISPER_MODEL_PATH"] ?? ""
        try XCTSkipIf(modelPath.isEmpty, "Set METAL_WHISPER_MODEL_PATH to a Whisper large-v3 Q5_0 GGML model")
        _ = try requireMetalDevice()
        let modelURL = URL(fileURLWithPath: modelPath)

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let audioURL = try await Self.verifiedSample(environment: environment, temporaryDirectory: directory)

        let automatic = try MetalWhisperEngine(
            modelURL: modelURL, configuration: .init(maxBatchPacks: 2, enableProfiling: true)
        )
        let fallback = try MetalWhisperEngine(
            modelURL: modelURL,
            configuration: .init(
                encoderBackend: .mps, decoderBackend: .mps, crossKVFormat: .f16,
                maxBatchPacks: 2, enableProfiling: true
            )
        )
        let options = MetalWhisperDecodeOptions(languageCode: "en")
        let result = try automatic.transcribe(audioURL: audioURL, options: options)
        XCTAssertTrue(result.text.lowercased().contains("ask not what your country can do for you"), result.text)
        XCTAssertTrue(result.text.lowercased().contains("ask what you can do for your country"), result.text)
        XCTAssertEqual(result.detectedLanguage, "en")
        XCTAssertEqual(result.durationSeconds, 11, accuracy: 0.001)
        XCTAssertFalse(result.segments.isEmpty)
        XCTAssertFalse(result.tokens.isEmpty)
        XCTAssertEqual(result.profile?.encodedPacks, 1)

        let portable = try fallback.transcribe(audioURL: audioURL, options: options)
        // Quantized and F16 caches can choose different punctuation for the same words.
        XCTAssertEqual(Self.words(in: portable.text), Self.words(in: result.text))
        let profile = try XCTUnwrap(portable.profile)
        XCTAssertEqual(profile.encoderBackend, .mps)
        XCTAssertEqual(profile.decoderBackend, .mps)
        XCTAssertEqual(profile.crossKVFormat, .f16)
        XCTAssertEqual(profile.encodedPacks, 1)

        let samples = try AudioLoader.load16kHzMono(from: audioURL)
        let full = MetalWhisperPackInput(samples: samples, durationSeconds: result.durationSeconds)
        let prefixSamples = Array(samples.prefix(80_000))
        let prefix = MetalWhisperPackInput(samples: prefixSamples, durationSeconds: 5)
        let fullText = try XCTUnwrap(automatic.transcribeBatch(packs: [full], options: options).first?.text)
        let prefixText = try XCTUnwrap(automatic.transcribeBatch(packs: [prefix], options: options).first?.text)
        XCTAssertEqual(fullText.trimmingCharacters(in: .whitespacesAndNewlines), result.text)
        XCTAssertNotEqual(fullText, prefixText)
        let batch = try automatic.transcribeBatch(packs: [full, prefix, full], options: options)
        XCTAssertEqual(batch.map(\.text), [fullText, prefixText, fullText])

        let repeated = samples + [Float](repeating: 0, count: 480_000 - samples.count) + samples
        let long = try automatic.transcribe(samples16kHz: repeated, options: options)
        XCTAssertEqual(long.durationSeconds, 41, accuracy: 0.001)
        XCTAssertEqual(long.profile?.encodedPacks, 2)
        XCTAssertTrue(long.segments.contains { $0.start >= 30 && $0.end <= 41.001 })
        XCTAssertTrue(long.tokens.contains { $0.start >= 30 })
        XCTAssertEqual(long.segments.map(\.start), long.segments.map(\.start).sorted())
    }

    private static func words(in text: String) -> [Substring] {
        text.split { $0.isWhitespace || $0.isPunctuation }
    }

    private static func verifiedSample(environment: [String: String], temporaryDirectory: URL) async throws -> URL {
        let checkoutSample = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Benchmarks/samples/jfk.wav")
        let override = environment["METAL_WHISPER_TEST_AUDIO_PATH"].flatMap {
            $0.isEmpty ? nil : URL(fileURLWithPath: $0)
        }
        let cachedSample = FileManager.default.fileExists(atPath: checkoutSample.path) ? checkoutSample : nil
        if let localSample = override ?? cachedSample {
            try verifySampleDigest(Data(contentsOf: localSample))
            return localSample
        }

        let request = URLRequest(url: sampleURL, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
        let (download, response) = try await URLSession.shared.data(for: request)
        let status = try XCTUnwrap((response as? HTTPURLResponse)?.statusCode)
        guard status == 200 else { throw NSError(domain: "Public sample HTTP response", code: status) }
        try verifySampleDigest(download)
        let sample = temporaryDirectory.appendingPathComponent("jfk.wav")
        try download.write(to: sample, options: .atomic)
        return sample
    }

    private static func verifySampleDigest(_ data: Data) throws {
        let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard actual == sampleSHA256 else {
            throw NSError(domain: "JFK sample SHA-256 mismatch", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Expected JFK sample SHA-256 \(sampleSHA256), got \(actual)"
            ])
        }
    }
}

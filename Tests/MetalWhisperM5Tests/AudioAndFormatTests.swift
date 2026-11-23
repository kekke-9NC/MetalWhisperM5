import AVFoundation
import Foundation
import XCTest
@testable import MetalWhisperCLI
@testable import MetalWhisperM5

final class AudioAndFormatTests: XCTestCase {
    func testStereoWAVResamplesAndSplitsAtThirtySeconds() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("sine.wav")
        let sourceRate = 44_100.0
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: sourceRate, channels: 2))
        let sourceFrames = AVAudioFrameCount(31.25 * sourceRate)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: sourceFrames))
        buffer.frameLength = sourceFrames
        let channels = try XCTUnwrap(buffer.floatChannelData)
        for sample in 0..<Int(sourceFrames) {
            let angle = 2 * Double.pi * 440 * Double(sample) / sourceRate
            channels[0][sample] = Float(0.2 * sin(angle))
            channels[1][sample] = Float(0.2 * cos(angle))
        }

        do {
            var settings = format.settings
            settings.removeValue(forKey: AVLinearPCMIsNonInterleaved)
            let file = try AVAudioFile(
                forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false
            )
            try file.write(from: buffer)
        }

        let samples = try AudioLoader.load16kHzMono(from: url)
        XCTAssertEqual(samples.count, 500_000)
        XCTAssertEqual(AudioLoader.chunkRanges(sampleCount: samples.count), [0..<480_000, 480_000..<500_000])
        XCTAssertEqual(AudioLoader.chunkRanges(sampleCount: 480_000), [0..<480_000])
        XCTAssertTrue(AudioLoader.chunkRanges(sampleCount: 0).isEmpty)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))

        // Measure both quadratures to detect a missing channel as well as a wrong sample rate.
        let measurementCount = 1600
        let start = 1600
        guard samples.count >= start + measurementCount else { return XCTFail("The converted buffer is too short") }
        var sineAmplitude = 0.0
        var cosineAmplitude = 0.0
        for index in start..<(start + measurementCount) {
            let angle = 2 * Double.pi * 440 * Double(index) / 16_000
            sineAmplitude += Double(samples[index]) * sin(angle)
            cosineAmplitude += Double(samples[index]) * cos(angle)
        }
        XCTAssertEqual(sineAmplitude * 2 / Double(measurementCount), 0.1, accuracy: 0.025)
        XCTAssertEqual(cosineAmplitude * 2 / Double(measurementCount), 0.1, accuracy: 0.025)
    }

    func testTokenizerJoinsBytesAcrossUTF8TokenBoundaries() {
        let tokenizer = WhisperVocabTokenizer(
            tokenBytes: [[0xE6], [0x97, 0xA5], [0xF0, 0x9F], [0x8C], [0x8A]],
            nVocab: 51_866
        )
        XCTAssertEqual(tokenizer.decode([0, 1, 2, 3, 4]), "日🌊")
    }

    func testTranscriptFormats() throws {
        let token = MetalWhisperToken(
            id: 109, text: "日本語 & <work>", start: 1.25, end: 2.5, probability: 0.8, isTimestamp: false
        )
        let first = MetalWhisperSegment(
            start: 1.25, end: 2.5, text: token.text, avgLogProb: -0.25, noSpeechProb: 0.04, tokens: [token]
        )
        let second = MetalWhisperSegment(
            start: 3661.9996, end: 3662.25, text: "First line\n\nSecond line", avgLogProb: -0.5,
            noSpeechProb: 0.05, tokens: []
        )
        let transcript = MetalWhisperTranscription(
            text: "日本語 & <work> First line Second line", detectedLanguage: "ja",
            durationSeconds: 3663, segments: [first, second]
        )
        XCTAssertEqual(String(decoding: try OutputFormat.text.render(transcript), as: UTF8.self), transcript.text + "\n")
        let json = try OutputFormat.json.render(transcript)
        let decoded = try JSONDecoder().decode(MetalWhisperTranscription.self, from: json)
        XCTAssertEqual(decoded.text, transcript.text)
        XCTAssertEqual(decoded.detectedLanguage, "ja")
        XCTAssertEqual(decoded.durationSeconds, 3663)
        XCTAssertEqual(decoded.segments, [first, second])
        XCTAssertEqual(decoded.tokens, [token])
        XCTAssertEqual(
            String(decoding: try OutputFormat.srt.render(transcript), as: UTF8.self),
            "1\n00:00:01,250 --> 00:00:02,500\n日本語 & <work>\n\n2\n01:01:02,000 --> 01:01:02,250\nFirst line\nSecond line\n\n"
        )
        XCTAssertEqual(
            String(decoding: try OutputFormat.vtt.render(transcript), as: UTF8.self),
            "WEBVTT\n\n00:00:01.250 --> 00:00:02.500\n日本語 &amp; &lt;work&gt;\n\n01:01:02.000 --> 01:01:02.250\nFirst line\nSecond line\n\n"
        )
    }

    func testCLIArgumentsAndFallbackOptions() throws {
        let arguments = [
            "--audio", "sample with spaces.wav", "--language", "JA", "--prompt", "日本語",
            "--beam-size", "3", "--backend", "mps", "--cross-kv", "f16",
            "--max-batch", "4", "--profile", "--format", "json", "--output", "result.json"
        ]
        let command = try Arguments.parse(arguments, environment: ["METAL_WHISPER_MODEL_PATH": "model.bin"])
        guard case .transcribe(let invocation) = command else { return XCTFail("Expected a transcription command") }
        XCTAssertEqual(invocation.modelURL.lastPathComponent, "model.bin")
        XCTAssertEqual(invocation.audioURL.lastPathComponent, "sample with spaces.wav")
        XCTAssertEqual(invocation.outputURL?.lastPathComponent, "result.json")
        XCTAssertEqual(invocation.format, .json)
        XCTAssertEqual(invocation.decodeOptions.languageCode, "ja")
        XCTAssertEqual(invocation.decodeOptions.initialPrompt, "日本語")
        XCTAssertEqual(invocation.decodeOptions.beamSize, 3)
        XCTAssertEqual(invocation.configuration.encoderBackend, .mps)
        XCTAssertEqual(invocation.configuration.decoderBackend, .mps)
        XCTAssertEqual(invocation.configuration.crossKVFormat, .f16)
        XCTAssertEqual(invocation.configuration.maxBatchPacks, 4)
        XCTAssertTrue(invocation.configuration.enableProfiling)

        XCTAssertThrowsError(try Arguments.parse(["--beam-size", "6"], environment: [:]))
        XCTAssertThrowsError(try Arguments.parse(["--max-batch", "0"], environment: [:]))
        XCTAssertThrowsError(try Arguments.parse(["--model", "--audio", "sample.wav"], environment: [:]))
        XCTAssertThrowsError(try Arguments.parse(
            ["--model", "model.bin", "--audio", "sample.wav", "--output", "sample.wav"], environment: [:]
        ))
    }
}

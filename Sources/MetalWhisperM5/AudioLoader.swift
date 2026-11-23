import AudioToolbox
import AVFoundation
import Foundation

enum AudioLoader {
    static let sampleRate = 16_000

    static func load16kHzMono(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let source = file.processingFormat
        let sourceFrames = file.length
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 32_768) else {
            throw MetalWhisperError.invalidAudioInput
        }

        var samples: [Float] = []
        if source.sampleRate == Double(sampleRate), source.channelCount == 1 {
            while true {
                try Task.checkCancellation()
                guard try read(file, into: input, totalFrames: sourceFrames, requested: input.frameCapacity) else {
                    break
                }
                try appendMonoSamples(from: input, to: &samples)
            }
        } else {
            guard let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false
            ), let converter = AVAudioConverter(from: source, to: target),
               let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 32_768) else {
                throw MetalWhisperError.invalidAudioInput
            }
            converter.downmix = true
            let reader = ConverterInput(file: file, buffer: input, totalFrames: sourceFrames)
            while true {
                try Task.checkCancellation()
                var conversionError: NSError?
                let status = converter.convert(to: output, error: &conversionError) { packetCount, inputStatus in
                    reader.next(requested: packetCount, status: inputStatus)
                }
                if let readError = reader.error { throw readError }
                if let conversionError { throw conversionError }
                if status == .error { throw MetalWhisperError.invalidAudioInput }
                try appendMonoSamples(from: output, to: &samples)
                if status == .endOfStream { break }
                if status == .inputRanDry && reader.reachedEnd && output.frameLength == 0 { break }
            }
        }

        guard !samples.isEmpty else { throw MetalWhisperError.invalidAudioInput }
        return samples
    }

    static func chunkRanges(sampleCount: Int, samplesPerWindow: Int = 480_000) -> [Range<Int>] {
        guard sampleCount > 0, samplesPerWindow > 0 else { return [] }
        return stride(from: 0, to: sampleCount, by: samplesPerWindow).map { start in
            start..<(start + min(samplesPerWindow, sampleCount - start))
        }
    }

    // AVAudioConverter requests input synchronously. This state is local to one converter
    // and is inspected by the caller only after each conversion call returns.
    private final class ConverterInput: @unchecked Sendable {
        private let file: AVAudioFile
        private let buffer: AVAudioPCMBuffer
        private let totalFrames: AVAudioFramePosition
        private(set) var reachedEnd = false
        private(set) var error: Error?

        init(file: AVAudioFile, buffer: AVAudioPCMBuffer, totalFrames: AVAudioFramePosition) {
            self.file = file
            self.buffer = buffer
            self.totalFrames = totalFrames
        }

        func next(
            requested: AVAudioFrameCount, status: UnsafeMutablePointer<AVAudioConverterInputStatus>
        ) -> AVAudioPCMBuffer? {
            if !reachedEnd {
                do {
                    if try AudioLoader.read(file, into: buffer, totalFrames: totalFrames, requested: requested) {
                        status.pointee = .haveData
                        return buffer
                    }
                } catch {
                    self.error = error
                }
                reachedEnd = true
            }
            status.pointee = .endOfStream
            return nil
        }
    }

    private static func read(
        _ file: AVAudioFile, into buffer: AVAudioPCMBuffer, totalFrames: AVAudioFramePosition,
        requested: AVAudioFrameCount
    ) throws -> Bool {
        let remaining = totalFrames - file.framePosition
        guard remaining > 0 else { return false }
        let frames = AVAudioFrameCount(min(remaining, Int64(requested), Int64(buffer.frameCapacity)))
        guard frames > 0 else { return false }
        buffer.frameLength = 0
        do {
            try file.read(into: buffer, frameCount: frames)
        } catch let error as NSError where error.domain == NSOSStatusErrorDomain
            && error.code == Int(kAudioFileEndOfFileError) {
            // Some decoders return an EOF status instead of a zero-length buffer.
        }
        return buffer.frameLength > 0
    }

    private static func appendMonoSamples(from buffer: AVAudioPCMBuffer, to samples: inout [Float]) throws {
        guard let channel = buffer.floatChannelData?.pointee else { throw MetalWhisperError.invalidAudioInput }
        let frames = UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))
        guard frames.allSatisfy(\.isFinite) else { throw MetalWhisperError.invalidAudioInput }
        samples.append(contentsOf: frames)
    }
}

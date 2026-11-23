import Foundation
import MetalWhisperM5

enum OutputFormat: String {
    case text, json, srt, vtt

    func render(_ result: MetalWhisperTranscription) throws -> Data {
        switch self {
        case .text:
            return Data((result.text + "\n").utf8)
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let transcript = MetalWhisperTranscription(
                text: result.text, detectedLanguage: result.detectedLanguage,
                durationSeconds: result.durationSeconds, segments: result.segments
            )
            var data = try encoder.encode(transcript)
            data.append(0x0A)
            return data
        case .srt, .vtt:
            let webVTT = self == .vtt
            var blocks: [String] = webVTT ? ["WEBVTT"] : []
            var cueNumber = 0
            for segment in result.segments where segment.start.isFinite && segment.end.isFinite {
                let start = Self.milliseconds(segment.start)
                let end = Self.milliseconds(segment.end)
                let lines = segment.text.split(whereSeparator: \.isNewline).map {
                    $0.trimmingCharacters(in: .whitespaces)
                }.filter { !$0.isEmpty }
                guard end > start, !lines.isEmpty else { continue }

                let separator = webVTT ? "." : ","
                let range = "\(Self.timestamp(start, separator: separator)) --> \(Self.timestamp(end, separator: separator))"
                let cue = lines.joined(separator: "\n")
                cueNumber += 1
                if webVTT {
                    let escaped = cue.replacingOccurrences(of: "&", with: "&amp;")
                        .replacingOccurrences(of: "<", with: "&lt;")
                        .replacingOccurrences(of: ">", with: "&gt;")
                    blocks.append(range + "\n" + escaped)
                } else {
                    blocks.append("\(cueNumber)\n" + range + "\n" + cue)
                }
            }
            guard !blocks.isEmpty else { return Data() }
            return Data((blocks.joined(separator: "\n\n") + "\n\n").utf8)
        }
    }

    private static func milliseconds(_ seconds: Double) -> Int64 {
        // Leave headroom when converting the rounded Double to Int64.
        Int64((min(max(0, seconds), Double(Int64.max / 2048)) * 1000).rounded())
    }

    private static func timestamp(_ total: Int64, separator: String) -> String {
        let hours = total / 3_600_000
        let minutes = total / 60_000 % 60
        let seconds = total / 1_000 % 60
        return String(format: "%02lld:%02lld:%02lld%@%03lld", hours, minutes, seconds, separator, total % 1_000)
    }
}

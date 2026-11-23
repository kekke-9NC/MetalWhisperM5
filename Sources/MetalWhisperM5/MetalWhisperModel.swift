import Accelerate
import Foundation
import Metal

struct WhisperSpecialTokens: Sendable {
    let eot: Int
    let sot: Int
    let translate: Int
    let transcribe: Int
    let solm: Int
    let prev: Int
    let noSpeech: Int
    let noTimestamps: Int
    let timestampBegin: Int
    let numLanguages: Int

    static let languageCodes: [String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id",
        "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg",
        "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
        "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so",
        "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt",
        "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"
    ]

    static let suppressNonSpeechTokenIDs: [Int] = [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59,
        60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902,
        918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183,
        4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
        15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650,
        32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362
    ]

    init(nVocab: Int) {
        if nVocab >= 51_865 {
            let langs = max(98, nVocab - 51_765 - 1)
            let dt = langs - 98
            self.eot = 50_257
            self.sot = 50_258
            self.translate = 50_357 + dt
            self.transcribe = 50_358 + dt
            self.solm = 50_359 + dt
            self.prev = 50_360 + dt
            self.noSpeech = 50_361 + dt
            self.noTimestamps = 50_362 + dt
            self.timestampBegin = 50_363 + dt
            self.numLanguages = langs
        } else {
            self.eot = 50_256
            self.sot = 50_257
            self.translate = 50_357
            self.transcribe = 50_358
            self.solm = 50_359
            self.prev = 50_360
            self.noSpeech = 50_361
            self.noTimestamps = 50_362
            self.timestampBegin = 50_363
            self.numLanguages = 0
        }
    }

    func languageTokenID(for code: String) -> Int? {
        let normalized = code.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty, normalized != "auto" else { return nil }
        guard let idx = Self.languageCodes.firstIndex(of: normalized), idx < numLanguages else {
            return nil
        }
        return sot + 1 + idx
    }

    func languageCode(forTokenID id: Int) -> String? {
        let idx = id - (sot + 1)
        guard idx >= 0, idx < min(numLanguages, Self.languageCodes.count) else { return nil }
        return Self.languageCodes[idx]
    }
}

/// Exact Whisper byte-level vocabulary decoder and longest-match prompt encoder
/// backed directly by the vocabulary embedded in the GGML `.bin` file.
final class WhisperVocabTokenizer: @unchecked Sendable {
    let tokenBytes: [[UInt8]]
    let specials: WhisperSpecialTokens
    private let byteLookup: [[UInt8]: Int]
    private let maxTokenLength: Int

    init(tokenBytes: [[UInt8]], nVocab: Int) {
        self.tokenBytes = tokenBytes
        self.specials = WhisperSpecialTokens(nVocab: nVocab)
        var lookup: [[UInt8]: Int] = [:]
        lookup.reserveCapacity(tokenBytes.count)
        var maxLen = 1
        for (id, bytes) in tokenBytes.enumerated() where !bytes.isEmpty {
            if lookup[bytes] == nil {
                lookup[bytes] = id
            }
            if bytes.count > maxLen { maxLen = bytes.count }
        }
        self.byteLookup = lookup
        self.maxTokenLength = min(maxLen, 64)
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)
        for id in ids where id >= 0 && id < tokenBytes.count {
            bytes.append(contentsOf: tokenBytes[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    func encodePrompt(_ text: String, maxTokens: Int = 200) -> [Int] {
        let cleaned = text
            .replacingOccurrences(of: "<|", with: " ")
            .replacingOccurrences(of: "|>", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return [] }
        let bytes = Array((" " + cleaned).utf8)
        var ids: [Int] = []
        ids.reserveCapacity(min(maxTokens, bytes.count))

        var i = 0
        while i < bytes.count && ids.count < maxTokens {
            let upper = min(bytes.count, i + maxTokenLength)
            var matchedID: Int?
            var matchedLen = 1
            for len in stride(from: upper - i, through: 1, by: -1) {
                let slice = Array(bytes[i..<(i + len)])
                if let id = byteLookup[slice] {
                    matchedID = id
                    matchedLen = len
                    break
                }
            }
            if let id = matchedID {
                ids.append(id)
                i += matchedLen
            } else {
                i += 1
            }
        }
        return ids
    }
}

/// Whisper log-mel spectrograms computed with Accelerate matrix multiplication.
final class WhisperLogMelExtractor: @unchecked Sendable {
    static let sampleRate = AudioLoader.sampleRate
    static let nFFT = 400
    static let hopLength = 160
    static let chunkSamples = 30 * sampleRate // 480,000
    static let nFrames = 3_000
    static let nFreqBins = nFFT / 2 + 1 // 201

    let nMels: Int
    let frameCount: Int
    var samplesPerWindow: Int { frameCount * Self.hopLength }
    private let melFiltersT: [Float] // [201, nMels] pre-transposed for vDSP_mmul
    private let hannWindow: [Float]  // [400]
    private let cosTableT: [Float]   // [400, 201] pre-transposed for vDSP_mmul
    private let sinTableT: [Float]   // [400, 201]

    init(nMels: Int, melFilters: [Float], frameCount: Int = WhisperLogMelExtractor.nFrames) {
        self.nMels = nMels
        self.frameCount = frameCount

        // Pre-transpose melFilters [nMels, 201] -> [201, nMels]
        var mfT = [Float](repeating: 0, count: Self.nFreqBins * nMels)
        for m in 0..<nMels {
            for k in 0..<Self.nFreqBins {
                mfT[k * nMels + m] = melFilters[m * Self.nFreqBins + k]
            }
        }
        self.melFiltersT = mfT

        // Periodic Hann window matching PyTorch torch.hann_window(400)
        var hann = [Float](repeating: 0, count: Self.nFFT)
        for i in 0..<Self.nFFT {
            hann[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(Self.nFFT)))
        }
        self.hannWindow = hann

        // Precompute exact 400-point real DFT basis matrices [400, 201] for vDSP_mmul (AMX, zero deprecation)
        var cosT = [Float](repeating: 0, count: Self.nFFT * Self.nFreqBins)
        var sinT = [Float](repeating: 0, count: Self.nFFT * Self.nFreqBins)
        for n in 0..<Self.nFFT {
            let row = n * Self.nFreqBins
            for k in 0..<Self.nFreqBins {
                let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(Self.nFFT)
                cosT[row + k] = Float(cos(angle))
                sinT[row + k] = Float(-sin(angle))
            }
        }
        self.cosTableT = cosT
        self.sinTableT = sinT
    }

    /// Returns time-major Float32 log-mel frames with Whisper normalization.
    func extractTimeMajor(samples: [Float]) -> [Float] {
        let chunkSamples = samplesPerWindow
        let nSamples = min(samples.count, chunkSamples)
        let pad = Self.nFFT / 2 // 200
        let paddedLen = chunkSamples + 2 * pad
        var padded = [Float](repeating: 0, count: paddedLen)

        if nSamples > 0 {
            samples.withUnsafeBufferPointer { src in
                padded.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.advanced(by: pad).update(from: src.baseAddress!, count: nSamples)
                }
            }
            for i in 0..<pad {
                let reflectIdx = min(chunkSamples - 1, pad - i)
                padded[i] = padded[pad + reflectIdx]
            }
            let endBase = pad + chunkSamples
            for i in 0..<pad {
                padded[endBase + i] = padded[endBase - 2 - i]
            }
        }

        let nFrames = frameCount
        let nFFT = Self.nFFT
        let nBins = Self.nFreqBins
        var frames = [Float](repeating: 0, count: nFrames * nFFT)
        padded.withUnsafeBufferPointer { padPtr in
            hannWindow.withUnsafeBufferPointer { winPtr in
                frames.withUnsafeMutableBufferPointer { dstPtr in
                    let padBase = padPtr.baseAddress!
                    let winBase = winPtr.baseAddress!
                    let dstBase = dstPtr.baseAddress!
                    for f in 0..<nFrames {
                        vDSP_vmul(
                            padBase.advanced(by: f * Self.hopLength), 1,
                            winBase, 1,
                            dstBase.advanced(by: f * nFFT), 1,
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }
        }

        // Exact 400-point DFT via Apple AMX vDSP_mmul:
        // real[3000, 201] = frames[3000, 400] * cosTableT[400, 201]
        var realPart = [Float](repeating: 0, count: nFrames * nBins)
        var imagPart = [Float](repeating: 0, count: nFrames * nBins)
        vDSP_mmul(frames, 1, cosTableT, 1, &realPart, 1, vDSP_Length(nFrames), vDSP_Length(nBins), vDSP_Length(nFFT))
        vDSP_mmul(frames, 1, sinTableT, 1, &imagPart, 1, vDSP_Length(nFrames), vDSP_Length(nBins), vDSP_Length(nFFT))

        var magSq = [Float](repeating: 0, count: nFrames * nBins)
        vDSP_vsq(realPart, 1, &realPart, 1, vDSP_Length(magSq.count))
        vDSP_vsq(imagPart, 1, &imagPart, 1, vDSP_Length(magSq.count))
        vDSP_vadd(realPart, 1, imagPart, 1, &magSq, 1, vDSP_Length(magSq.count))

        // Time-major Mel projection via AMX vDSP_mmul:
        // mel[3000, nMels] = magSq[3000, 201] * melFiltersT[201, nMels]
        var mel = [Float](repeating: 0, count: nFrames * nMels)
        vDSP_mmul(magSq, 1, melFiltersT, 1, &mel, 1, vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // log10(max(mel, 1e-10)), clamp to max - 8.0, then (x + 4.0) / 4.0
        var floorVal: Float = 1e-10
        var count32 = Int32(mel.count)
        vDSP_vthr(mel, 1, &floorVal, &mel, 1, vDSP_Length(mel.count))
        vvlog10f(&mel, mel, &count32)

        var maxVal: Float = -Float.greatestFiniteMagnitude
        vDSP_maxv(mel, 1, &maxVal, vDSP_Length(mel.count))
        var clampLow = maxVal - 8.0
        vDSP_vthr(mel, 1, &clampLow, &mel, 1, vDSP_Length(mel.count))

        var addFour: Float = 4.0
        var divFour: Float = 0.25
        vDSP_vsadd(mel, 1, &addFour, &mel, 1, vDSP_Length(mel.count))
        vDSP_vsmul(mel, 1, &divFour, &mel, 1, vDSP_Length(mel.count))
        return mel
    }
}

struct EncoderBlockWeights: @unchecked Sendable {
    let attnLnW: MTLBuffer   // F32 [D]
    let attnLnB: MTLBuffer   // F32 [D]
    let qkvW_F16: MTLBuffer  // F16 [3D, D] (fused Q, K, V)
    let qkvB_F32: MTLBuffer  // F32 [3D] (K bias = 0)
    let outW_F16: MTLBuffer  // F16 [D, D]
    let outB_F32: MTLBuffer  // F32 [D]
    let mlpLnW: MTLBuffer    // F32 [D]
    let mlpLnB: MTLBuffer    // F32 [D]
    let fc1W_F16: MTLBuffer  // F16 [4D, D]
    let fc1B_F32: MTLBuffer  // F32 [4D]
    let fc2W_F16: MTLBuffer  // F16 [D, 4D]
    let fc2B_F32: MTLBuffer  // F32 [D]
}

/// Decoder weights pre-dequantized bit-exactly to F16 in Unified Memory so
/// Apple M5 Pro's hardware `MPSMatrixMultiplication` runs at ~286 GB/s across
/// B = 5..40 simultaneous beams (8x-15x faster than scalar Q5_0 dequant!).
struct DecoderBlockWeights: @unchecked Sendable {
    let selfLnW: MTLBuffer     // F32 [D]
    let selfLnB: MTLBuffer     // F32 [D]
    let selfQKV_F16: MTLBuffer // F16 [3D, D] fused Q, K, V
    let selfQKV_B: MTLBuffer   // F32 [3D] (K bias = 0)
    let selfOut_F16: MTLBuffer // F16 [D, D]
    let selfOut_B: MTLBuffer   // F32 [D]

    let crossLnW: MTLBuffer    // F32 [D]
    let crossLnB: MTLBuffer    // F32 [D]
    let crossQ_F16: MTLBuffer  // F16 [D, D]
    let crossQ_B: MTLBuffer    // F32 [D]
    let crossKV_F16: MTLBuffer // F16 [2D, D]
    let crossKV_B: MTLBuffer   // F32 [2D] (K bias = 0, V bias = v.bias)
    let crossOut_F16: MTLBuffer // F16 [D, D]
    let crossOut_B: MTLBuffer  // F32 [D]

    let mlpLnW: MTLBuffer      // F32 [D]
    let mlpLnB: MTLBuffer      // F32 [D]
    let fc1_F16: MTLBuffer     // F16 [4D, D]
    let fc1_B: MTLBuffer       // F32 [4D]
    let fc2_F16: MTLBuffer     // F16 [D, 4D]
    let fc2_B: MTLBuffer       // F32 [D]
}

struct MetalWhisperQ5Weights: @unchecked Sendable {
    let buffer: MTLBuffer
    let highBitsOffset: Int
    let scalesOffset: Int
}

struct MetalWhisperQ5DecoderBlock: @unchecked Sendable {
    let selfQKV: MetalWhisperQ5Weights
    let selfOut: MetalWhisperQ5Weights
    let crossQ: MetalWhisperQ5Weights
    let crossOut: MetalWhisperQ5Weights
    let fc1: MetalWhisperQ5Weights
    let fc2: MetalWhisperQ5Weights
}

private struct RawTensorSlice: Sendable {
    let name: String
    let dims: [Int]
    let ftype: Int32
    let offset: Int
}

private final class GGMLTensorReader: @unchecked Sendable {
    let fileData: Data
    let table: [String: RawTensorSlice]
    let device: MTLDevice

    init(fileData: Data, table: [String: RawTensorSlice], device: MTLDevice) {
        self.fileData = fileData
        self.table = table
        self.device = device
    }

    func requireSlice(_ name: String, shape: [Int]? = nil) throws -> RawTensorSlice {
        guard let slice = table[name] else { throw MetalWhisperError.missingTensor(name) }
        if let shape, slice.dims.filter({ $0 != 1 }) != shape.filter({ $0 != 1 }) {
            throw MetalWhisperError.invalidModelHeader("unexpected dimensions for \(name)")
        }
        return slice
    }

    func makeF32Buffer(_ name: String, shape: [Int]) throws -> MTLBuffer {
        let s = try requireSlice(name, shape: shape)
        let ne = s.dims.reduce(1, *)
        guard ne * 4 <= device.maxBufferLength,
              let buf = device.makeBuffer(length: ne * 4, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(s.name)
        }
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: ne)
        copyAsF32(slice: s, into: ptr, count: ne)
        return buf
    }

    func makeF16Buffer(_ name: String, shape: [Int]) throws -> MTLBuffer {
        let s = try requireSlice(name, shape: shape)
        let ne = s.dims.reduce(1, *)
        guard ne * 2 <= device.maxBufferLength,
              let buf = device.makeBuffer(length: ne * 2, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(s.name)
        }
        let ptr = buf.contents().bindMemory(to: Float16.self, capacity: ne)
        copyAsF16(slice: s, into: ptr, count: ne)
        return buf
    }

    func makeConcatF16Buffer(_ names: [String], shape: [Int]) throws -> MTLBuffer {
        let slices = try names.map { try requireSlice($0, shape: shape) }
        let totalNe = slices.reduce(0) { $0 + $1.dims.reduce(1, *) }
        guard totalNe * 2 <= device.maxBufferLength,
              let buf = device.makeBuffer(length: totalNe * 2, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(names.first ?? "concat_f16")
        }
        let basePtr = buf.contents().bindMemory(to: Float16.self, capacity: totalNe)
        var offset = 0
        for s in slices {
            let ne = s.dims.reduce(1, *)
            copyAsF16(slice: s, into: basePtr.advanced(by: offset), count: ne)
            offset += ne
        }
        return buf
    }

    func makeQ5Weights(_ names: [String]) throws -> MetalWhisperQ5Weights {
        let slices = try names.map { try requireSlice($0) }
        guard let first = slices.first, first.dims.count == 2 else {
            throw MetalWhisperError.invalidModelHeader("Expected Q5_0 weight matrices")
        }
        let inner = first.dims[0]
        for slice in slices {
            guard slice.ftype == 6, slice.dims.count == 2,
                  slice.dims[0] == inner, inner.isMultiple(of: 32) else {
                throw MetalWhisperError.unsupportedQuantization(slice.ftype, slice.name)
            }
        }
        let columns = slices.reduce(0) { $0 + $1.dims[1] }
        let blocksPerRow = inner / 32
        let blocks = columns * blocksPerRow
        let highBitsOffset = blocks * MemoryLayout<SIMD4<UInt32>>.stride
        let scalesOffset = (highBitsOffset + blocks * MemoryLayout<UInt32>.stride + 15) & ~15
        let bytes = scalesOffset + blocks * MemoryLayout<UInt16>.stride
        guard bytes <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(first.name)
        }
        let nibbles = buffer.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: blocks)
        let high = buffer.contents().advanced(by: highBitsOffset).bindMemory(to: UInt32.self, capacity: blocks)
        let scales = buffer.contents().advanced(by: scalesOffset).bindMemory(to: UInt16.self, capacity: blocks)

        // Transpose whole Q5_0 blocks into K-major planes. The source bytes are
        // preserved, including the FP16 scale and all five bits of each quant.
        fileData.withUnsafeBytes { raw in
            var firstColumn = 0
            for slice in slices {
                let src = raw.baseAddress!.advanced(by: slice.offset)
                for block in 0..<blocksPerRow {
                    let destination = block * columns + firstColumn
                    for column in 0..<slice.dims[1] {
                        let from = (column * blocksPerRow + block) * 22
                        let to = destination + column
                        scales[to] = src.loadUnaligned(fromByteOffset: from, as: UInt16.self)
                        high[to] = src.loadUnaligned(fromByteOffset: from + 2, as: UInt32.self)
                        nibbles[to] = src.loadUnaligned(fromByteOffset: from + 6, as: SIMD4<UInt32>.self)
                    }
                }
                firstColumn += slice.dims[1]
            }
        }
        return MetalWhisperQ5Weights(buffer: buffer, highBitsOffset: highBitsOffset, scalesOffset: scalesOffset)
    }

    func makeConcatQKVBiasF32(qName: String, d: Int, vName: String) throws -> MTLBuffer {
        let qSlice = try requireSlice(qName, shape: [d])
        let vSlice = try requireSlice(vName, shape: [d])
        guard let buf = device.makeBuffer(length: 3 * d * 4, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(qName)
        }
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: 3 * d)
        copyAsF32(slice: qSlice, into: ptr, count: d)
        memset(ptr.advanced(by: d), 0, d * 4)
        copyAsF32(slice: vSlice, into: ptr.advanced(by: 2 * d), count: d)
        return buf
    }

    func makeConcatKVBiasF32(d: Int, vName: String) throws -> MTLBuffer {
        let vSlice = try requireSlice(vName, shape: [d])
        guard let buf = device.makeBuffer(length: 2 * d * 4, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(vName)
        }
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: 2 * d)
        memset(ptr, 0, d * 4)
        copyAsF32(slice: vSlice, into: ptr.advanced(by: d), count: d)
        return buf
    }

    func makeConvWeightF16(_ name: String, cIn: Int, cOut: Int) throws -> MTLBuffer {
        let s = try requireSlice(name, shape: [3, cIn, cOut])
        let total = cOut * cIn * 3
        var tmpF32 = [Float](repeating: 0, count: total)
        tmpF32.withUnsafeMutableBufferPointer { dst in
            copyAsF32(slice: s, into: dst.baseAddress!, count: total)
        }
        guard let buf = device.makeBuffer(length: total * 2, options: .storageModeShared) else {
            throw MetalWhisperError.bufferAllocationFailed(name)
        }
        let outPtr = buf.contents().bindMemory(to: Float16.self, capacity: total)
        for oc in 0..<cOut {
            let srcRow = oc * cIn * 3
            let dstRow = oc * 3 * cIn
            for k in 0..<3 {
                for ic in 0..<cIn {
                    outPtr[dstRow + k * cIn + ic] = Float16(tmpF32[srcRow + ic * 3 + k])
                }
            }
        }
        return buf
    }

    private func copyAsF32(slice: RawTensorSlice, into dst: UnsafeMutablePointer<Float>, count: Int) {
        fileData.withUnsafeBytes { raw in
            let srcBase = raw.baseAddress!.advanced(by: slice.offset)
            switch slice.ftype {
            case 0:
                memcpy(dst, srcBase, count * 4)
            case 1:
                for i in 0..<count {
                    dst[i] = Float(srcBase.loadUnaligned(fromByteOffset: i * 2, as: Float16.self))
                }
            case 6:
                MetalWhisperModel.dequantizeQ5_0ToF32(src: srcBase, dst: dst, count: count)
            case 8:
                MetalWhisperModel.dequantizeQ8_0ToF32(src: srcBase, dst: dst, count: count)
            default:
                break
            }
        }
    }

    private func copyAsF16(slice: RawTensorSlice, into dst: UnsafeMutablePointer<Float16>, count: Int) {
        fileData.withUnsafeBytes { raw in
            let srcBase = raw.baseAddress!.advanced(by: slice.offset)
            switch slice.ftype {
            case 0:
                for i in 0..<count {
                    dst[i] = Float16(srcBase.loadUnaligned(fromByteOffset: i * 4, as: Float.self))
                }
            case 1:
                memcpy(dst, srcBase, count * 2)
            case 6:
                MetalWhisperModel.dequantizeQ5_0ToF16(src: srcBase, dst: dst, count: count)
            case 8:
                var tmp = [Float](repeating: 0, count: count)
                tmp.withUnsafeMutableBufferPointer { ptr in
                    MetalWhisperModel.dequantizeQ8_0ToF32(src: srcBase, dst: ptr.baseAddress!, count: count)
                }
                for i in 0..<count { dst[i] = Float16(tmp[i]) }
            default:
                break
            }
        }
    }
}

private final class ParallelCollector<T>: @unchecked Sendable {
    private let lock = NSLock()
    var items: [T?]
    var firstError: Error?

    init(count: Int) {
        self.items = [T?](repeating: nil, count: count)
    }

    func set(_ index: Int, _ value: T) {
        lock.lock()
        items[index] = value
        lock.unlock()
    }

    func recordError(_ error: Error) {
        lock.lock()
        if firstError == nil { firstError = error }
        lock.unlock()
    }
}

/// A memory-mapped GGML Whisper model converted to shared Metal weight buffers.
final class MetalWhisperModel: @unchecked Sendable {
    let device: MTLDevice
    let hparams: WhisperHParams
    let tokenizer: WhisperVocabTokenizer
    let melExtractor: WhisperLogMelExtractor

    // Encoder stem & top
    let conv1W_F16: MTLBuffer    // [D, 3 * nMels] F16
    let conv1B_F32: MTLBuffer    // [D]
    let conv2W_F16: MTLBuffer    // [D, 3 * D] F16
    let conv2B_F32: MTLBuffer    // [D]
    let encPosEmb_F32: MTLBuffer // [1500, D] F32
    let encBlocks: [EncoderBlockWeights]
    let encLnPostW: MTLBuffer    // [D]
    let encLnPostB: MTLBuffer    // [D]

    // Decoder embeddings, blocks & tied LM head
    let decTokenEmb_F16: MTLBuffer // [nVocab, D] F16 (shared by embedding lookup & MPS-16 LM Head!)
    let decPosEmb_F32: MTLBuffer   // [448, D] F32
    let decBlocks: [DecoderBlockWeights]
    let decLnW: MTLBuffer          // [D]
    let decLnB: MTLBuffer          // [D]
    let decQ5Head: MetalWhisperQ5Weights?
    let decQ5Blocks: [MetalWhisperQ5DecoderBlock]

    init(modelURL: URL, device: MTLDevice, loadQ5Head: Bool = false, loadQ5Decoder: Bool = false) throws {
        self.device = device
        let fileData = try Data(contentsOf: modelURL, options: .mappedIfSafe)
        let totalSize = fileData.count
        guard totalSize > 64 else {
            throw MetalWhisperError.invalidModelHeader("file too small")
        }

        var cursor = 0
        func readUInt32() throws -> UInt32 {
            guard cursor + 4 <= totalSize else { throw MetalWhisperError.invalidModelHeader("unexpected EOF") }
            let val = fileData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self) }
            cursor += 4
            return UInt32(littleEndian: val)
        }
        func readInt32() throws -> Int32 {
            Int32(bitPattern: try readUInt32())
        }

        let magic = try readUInt32()
        guard magic == 0x67676d6c else {
            throw MetalWhisperError.invalidModelHeader("bad magic \(String(format: "0x%08x", magic))")
        }

        let hp = WhisperHParams(
            nVocab: Int(try readInt32()),
            nAudioCtx: Int(try readInt32()),
            nAudioState: Int(try readInt32()),
            nAudioHead: Int(try readInt32()),
            nAudioLayer: Int(try readInt32()),
            nTextCtx: Int(try readInt32()),
            nTextState: Int(try readInt32()),
            nTextHead: Int(try readInt32()),
            nTextLayer: Int(try readInt32()),
            nMels: Int(try readInt32()),
            ftype: Int(try readInt32())
        )
        // The attention kernels use 64 features per head and the workspace is
        // shared between encoder and decoder projections of equal width.
        guard (51_864...51_866).contains(hp.nVocab),
              (1...1_500).contains(hp.nAudioCtx), (8...448).contains(hp.nTextCtx),
              (1...32).contains(hp.nAudioHead), hp.nTextHead == hp.nAudioHead,
              hp.nAudioState == hp.nAudioHead * 64, hp.nTextState == hp.nAudioState,
              (1...32).contains(hp.nAudioLayer), (1...32).contains(hp.nTextLayer),
              [80, 128].contains(hp.nMels) else {
            throw MetalWhisperError.invalidModelHeader("unsupported Whisper model dimensions")
        }
        self.hparams = hp

        // Mel filter bank [n_mel, n_fft]
        let filterRows = Int(try readInt32())
        let filterCols = Int(try readInt32())
        guard filterRows == hp.nMels, filterCols == WhisperLogMelExtractor.nFreqBins else {
            throw MetalWhisperError.invalidModelHeader("invalid mel filter dimensions")
        }
        let filterCount = filterRows * filterCols
        guard filterCount * 4 <= totalSize - cursor else {
            throw MetalWhisperError.invalidModelHeader("truncated mel filters")
        }
        var melFilters = [Float](repeating: 0, count: filterCount)
        fileData.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: cursor)
            melFilters.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: base, count: filterCount * 4))
            }
        }
        cursor += filterCount * 4
        self.melExtractor = WhisperLogMelExtractor(nMels: filterRows, melFilters: melFilters, frameCount: hp.nAudioCtx * 2)

        // Vocabulary
        let vocabCount = Int(try readInt32())
        let textTokenCount = WhisperSpecialTokens(nVocab: hp.nVocab).eot
        guard (textTokenCount...hp.nVocab).contains(vocabCount), vocabCount <= (totalSize - cursor) / 4 else {
            throw MetalWhisperError.invalidModelHeader("invalid vocabulary size")
        }
        var tokenBytes: [[UInt8]] = []
        tokenBytes.reserveCapacity(vocabCount)
        for _ in 0..<vocabCount {
            let len = Int(try readInt32())
            guard (0...1_024).contains(len), len <= totalSize - cursor else {
                throw MetalWhisperError.invalidModelHeader("truncated vocab")
            }
            if len == 0 {
                tokenBytes.append([])
            } else {
                var bytes = [UInt8](repeating: 0, count: len)
                fileData.copyBytes(to: &bytes, from: cursor..<(cursor + len))
                tokenBytes.append(bytes)
                cursor += len
            }
        }
        self.tokenizer = WhisperVocabTokenizer(tokenBytes: tokenBytes, nVocab: hp.nVocab)

        // Index all tensors (zero-copy over mmap)
        var table: [String: RawTensorSlice] = [:]
        table.reserveCapacity(1400)
        while cursor < totalSize {
            let nDims = Int(try readInt32())
            let nameLen = Int(try readInt32())
            let ftype = try readInt32()
            guard (1...4).contains(nDims), (1...512).contains(nameLen) else {
                throw MetalWhisperError.invalidModelHeader("invalid tensor header at \(cursor)")
            }
            var dims: [Int] = []
            dims.reserveCapacity(nDims)
            var ne = 1
            for _ in 0..<nDims {
                let d = Int(try readInt32())
                let (count, overflow) = ne.multipliedReportingOverflow(by: d)
                guard d > 0, !overflow else {
                    throw MetalWhisperError.invalidModelHeader("invalid tensor dimensions at \(cursor)")
                }
                dims.append(d)
                ne = count
            }
            guard nameLen <= totalSize - cursor else {
                throw MetalWhisperError.invalidModelHeader("truncated tensor name")
            }
            let nameData = fileData.subdata(in: cursor..<(cursor + nameLen))
            let name = String(decoding: nameData, as: UTF8.self)
            cursor += nameLen

            let blockSize: Int
            let blockBytes: Int
            switch ftype {
            case 0: (blockSize, blockBytes) = (1, 4)
            case 1: (blockSize, blockBytes) = (1, 2)
            case 6: (blockSize, blockBytes) = (32, 22)
            case 8: (blockSize, blockBytes) = (32, 34)
            default:
                throw MetalWhisperError.unsupportedQuantization(ftype, name)
            }
            guard dims[0].isMultiple(of: blockSize) else {
                throw MetalWhisperError.invalidModelHeader("unaligned quantized rows for \(name)")
            }
            let (byteCount, overflow) = (ne / blockSize).multipliedReportingOverflow(by: blockBytes)
            guard !overflow, byteCount <= totalSize - cursor else {
                throw MetalWhisperError.invalidModelHeader("truncated tensor payload for \(name)")
            }
            guard table[name] == nil else {
                throw MetalWhisperError.invalidModelHeader("duplicate tensor \(name)")
            }
            table[name] = RawTensorSlice(name: name, dims: dims, ftype: ftype, offset: cursor)
            cursor += byteCount
        }

        let reader = GGMLTensorReader(fileData: fileData, table: table, device: device)
        let d = hp.nAudioState

        self.conv1W_F16 = try reader.makeConvWeightF16("encoder.conv1.weight", cIn: hp.nMels, cOut: d)
        self.conv1B_F32 = try reader.makeF32Buffer("encoder.conv1.bias", shape: [d])
        self.conv2W_F16 = try reader.makeConvWeightF16("encoder.conv2.weight", cIn: d, cOut: d)
        self.conv2B_F32 = try reader.makeF32Buffer("encoder.conv2.bias", shape: [d])
        self.encPosEmb_F32 = try reader.makeF32Buffer("encoder.positional_embedding", shape: [d, hp.nAudioCtx])
        self.encLnPostW = try reader.makeF32Buffer("encoder.ln_post.weight", shape: [d])
        self.encLnPostB = try reader.makeF32Buffer("encoder.ln_post.bias", shape: [d])

        self.decTokenEmb_F16 = try reader.makeF16Buffer("decoder.token_embedding.weight", shape: [d, hp.nVocab])
        if (loadQ5Head || loadQ5Decoder), hp.nTextState.isMultiple(of: 64),
           table["decoder.token_embedding.weight"]?.ftype == 6 {
            self.decQ5Head = try? reader.makeQ5Weights(["decoder.token_embedding.weight"])
        } else {
            self.decQ5Head = nil
        }
        self.decPosEmb_F32 = try reader.makeF32Buffer("decoder.positional_embedding", shape: [d, hp.nTextCtx])
        self.decLnW = try reader.makeF32Buffer("decoder.ln.weight", shape: [d])
        self.decLnB = try reader.makeF32Buffer("decoder.ln.bias", shape: [d])

        let q5LayerWeights = [
            "attn.query.weight", "attn.key.weight", "attn.value.weight", "attn.out.weight",
            "cross_attn.query.weight", "cross_attn.out.weight", "mlp.0.weight", "mlp.2.weight"
        ]
        let needsQ5Blocks = loadQ5Decoder && self.decQ5Head != nil && (0..<hp.nTextLayer).allSatisfy { layer in
            q5LayerWeights.allSatisfy { table["decoder.blocks.\(layer).\($0)"]?.ftype == 6 }
        }
        // Convert independent transformer layers concurrently.
        let encCollector = ParallelCollector<EncoderBlockWeights>(count: hp.nAudioLayer)
        let decCollector = ParallelCollector<DecoderBlockWeights>(count: hp.nTextLayer)
        let q5Collector = ParallelCollector<MetalWhisperQ5DecoderBlock>(count: needsQ5Blocks ? hp.nTextLayer : 0)
        let totalLayers = hp.nAudioLayer + hp.nTextLayer
        let nEnc = hp.nAudioLayer

        DispatchQueue.concurrentPerform(iterations: totalLayers) { idx in
            do {
                if idx < nEnc {
                    let p = "encoder.blocks.\(idx)"
                    let block = EncoderBlockWeights(
                        attnLnW: try reader.makeF32Buffer("\(p).attn_ln.weight", shape: [d]),
                        attnLnB: try reader.makeF32Buffer("\(p).attn_ln.bias", shape: [d]),
                        qkvW_F16: try reader.makeConcatF16Buffer([
                            "\(p).attn.query.weight",
                            "\(p).attn.key.weight",
                            "\(p).attn.value.weight"
                        ], shape: [d, d]),
                        qkvB_F32: try reader.makeConcatQKVBiasF32(
                            qName: "\(p).attn.query.bias", d: d, vName: "\(p).attn.value.bias"
                        ),
                        outW_F16: try reader.makeF16Buffer("\(p).attn.out.weight", shape: [d, d]),
                        outB_F32: try reader.makeF32Buffer("\(p).attn.out.bias", shape: [d]),
                        mlpLnW: try reader.makeF32Buffer("\(p).mlp_ln.weight", shape: [d]),
                        mlpLnB: try reader.makeF32Buffer("\(p).mlp_ln.bias", shape: [d]),
                        fc1W_F16: try reader.makeF16Buffer("\(p).mlp.0.weight", shape: [d, 4 * d]),
                        fc1B_F32: try reader.makeF32Buffer("\(p).mlp.0.bias", shape: [4 * d]),
                        fc2W_F16: try reader.makeF16Buffer("\(p).mlp.2.weight", shape: [4 * d, d]),
                        fc2B_F32: try reader.makeF32Buffer("\(p).mlp.2.bias", shape: [d])
                    )
                    encCollector.set(idx, block)
                } else {
                    let layer = idx - nEnc
                    let p = "decoder.blocks.\(layer)"
                    let selfQKV = ["\(p).attn.query.weight", "\(p).attn.key.weight", "\(p).attn.value.weight"]
                    let block = DecoderBlockWeights(
                        selfLnW: try reader.makeF32Buffer("\(p).attn_ln.weight", shape: [d]),
                        selfLnB: try reader.makeF32Buffer("\(p).attn_ln.bias", shape: [d]),
                        selfQKV_F16: try reader.makeConcatF16Buffer(selfQKV, shape: [d, d]),
                        selfQKV_B: try reader.makeConcatQKVBiasF32(
                            qName: "\(p).attn.query.bias", d: d, vName: "\(p).attn.value.bias"
                        ),
                        selfOut_F16: try reader.makeF16Buffer("\(p).attn.out.weight", shape: [d, d]),
                        selfOut_B: try reader.makeF32Buffer("\(p).attn.out.bias", shape: [d]),
                        crossLnW: try reader.makeF32Buffer("\(p).cross_attn_ln.weight", shape: [d]),
                        crossLnB: try reader.makeF32Buffer("\(p).cross_attn_ln.bias", shape: [d]),
                        crossQ_F16: try reader.makeF16Buffer("\(p).cross_attn.query.weight", shape: [d, d]),
                        crossQ_B: try reader.makeF32Buffer("\(p).cross_attn.query.bias", shape: [d]),
                        crossKV_F16: try reader.makeConcatF16Buffer([
                            "\(p).cross_attn.key.weight",
                            "\(p).cross_attn.value.weight"
                        ], shape: [d, d]),
                        crossKV_B: try reader.makeConcatKVBiasF32(d: d, vName: "\(p).cross_attn.value.bias"),
                        crossOut_F16: try reader.makeF16Buffer("\(p).cross_attn.out.weight", shape: [d, d]),
                        crossOut_B: try reader.makeF32Buffer("\(p).cross_attn.out.bias", shape: [d]),
                        mlpLnW: try reader.makeF32Buffer("\(p).mlp_ln.weight", shape: [d]),
                        mlpLnB: try reader.makeF32Buffer("\(p).mlp_ln.bias", shape: [d]),
                        fc1_F16: try reader.makeF16Buffer("\(p).mlp.0.weight", shape: [d, 4 * d]),
                        fc1_B: try reader.makeF32Buffer("\(p).mlp.0.bias", shape: [4 * d]),
                        fc2_F16: try reader.makeF16Buffer("\(p).mlp.2.weight", shape: [4 * d, d]),
                        fc2_B: try reader.makeF32Buffer("\(p).mlp.2.bias", shape: [d])
                    )
                    decCollector.set(layer, block)
                    if needsQ5Blocks {
                        let quantized = try MetalWhisperQ5DecoderBlock(
                            selfQKV: reader.makeQ5Weights(selfQKV),
                            selfOut: reader.makeQ5Weights(["\(p).attn.out.weight"]),
                            crossQ: reader.makeQ5Weights(["\(p).cross_attn.query.weight"]),
                            crossOut: reader.makeQ5Weights(["\(p).cross_attn.out.weight"]),
                            fc1: reader.makeQ5Weights(["\(p).mlp.0.weight"]),
                            fc2: reader.makeQ5Weights(["\(p).mlp.2.weight"])
                        )
                        q5Collector.set(layer, quantized)
                    }
                }
            } catch {
                if idx < nEnc { encCollector.recordError(error) } else { decCollector.recordError(error) }
            }
        }

        if let err = encCollector.firstError { throw err }
        if let err = decCollector.firstError { throw err }
        self.encBlocks = encCollector.items.compactMap { $0 }
        self.decBlocks = decCollector.items.compactMap { $0 }
        self.decQ5Blocks = q5Collector.items.compactMap { $0 }
    }

    // MARK: - Bit-Exact GGML Dequantization

    static func dequantizeQ5_0ToF32(src: UnsafeRawPointer, dst: UnsafeMutablePointer<Float>, count: Int) {
        let nb = count / 32
        let bytes = src.assumingMemoryBound(to: UInt8.self)
        for b in 0..<nb {
            let off = b * 22
            let dBits = UInt16(bytes[off]) | (UInt16(bytes[off + 1]) << 8)
            let d = Float(Float16(bitPattern: dBits))
            let qh = UInt32(bytes[off + 2])
                | (UInt32(bytes[off + 3]) << 8)
                | (UInt32(bytes[off + 4]) << 16)
                | (UInt32(bytes[off + 5]) << 24)
            let base = b * 32
            for j in 0..<16 {
                let qByte = bytes[off + 6 + j]
                let xh0 = ((qh >> j) & 1) << 4
                let xh1 = ((qh >> (j + 16)) & 1) << 4
                let x0 = Int32(UInt32(qByte & 0x0F) | xh0) - 16
                let x1 = Int32(UInt32(qByte >> 4) | xh1) - 16
                dst[base + j] = Float(x0) * d
                dst[base + j + 16] = Float(x1) * d
            }
        }
    }

    static func dequantizeQ5_0ToF16(src: UnsafeRawPointer, dst: UnsafeMutablePointer<Float16>, count: Int) {
        let nb = count / 32
        let bytes = src.assumingMemoryBound(to: UInt8.self)
        for b in 0..<nb {
            let off = b * 22
            let dBits = UInt16(bytes[off]) | (UInt16(bytes[off + 1]) << 8)
            let d = Float(Float16(bitPattern: dBits))
            let qh = UInt32(bytes[off + 2])
                | (UInt32(bytes[off + 3]) << 8)
                | (UInt32(bytes[off + 4]) << 16)
                | (UInt32(bytes[off + 5]) << 24)
            let base = b * 32
            for j in 0..<16 {
                let qByte = bytes[off + 6 + j]
                let xh0 = ((qh >> j) & 1) << 4
                let xh1 = ((qh >> (j + 16)) & 1) << 4
                let x0 = Int32(UInt32(qByte & 0x0F) | xh0) - 16
                let x1 = Int32(UInt32(qByte >> 4) | xh1) - 16
                dst[base + j] = Float16(Float(x0) * d)
                dst[base + j + 16] = Float16(Float(x1) * d)
            }
        }
    }

    static func dequantizeQ8_0ToF32(src: UnsafeRawPointer, dst: UnsafeMutablePointer<Float>, count: Int) {
        let nb = count / 32
        let bytes = src.assumingMemoryBound(to: UInt8.self)
        for b in 0..<nb {
            let off = b * 34
            let dBits = UInt16(bytes[off]) | (UInt16(bytes[off + 1]) << 8)
            let d = Float(Float16(bitPattern: dBits))
            let base = b * 32
            for j in 0..<32 {
                let q = Int8(bitPattern: bytes[off + 2 + j])
                dst[base + j] = Float(q) * d
            }
        }
    }
}

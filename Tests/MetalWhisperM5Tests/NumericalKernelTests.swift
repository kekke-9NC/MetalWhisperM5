import Foundation
import Metal
import XCTest
@testable import MetalWhisperM5

func requireMetalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw XCTSkip("A Metal GPU is required for this test")
    }
    return device
}

final class NumericalKernelTests: XCTestCase {
    func testSeparateCrossKVQ8RemapsSlotsAndHandlesZeroFrames() throws {
        try verifyCrossKVQ8(packed: false)
    }

    func testPackedCrossKVQ8RemapsSlotsAndHandlesZeroFrames() throws {
        try verifyCrossKVQ8(packed: true)
    }

    func testPortableFlashAttentionMatchesFP32() throws {
        let device = try requireMetalDevice()
        let kernels = try MetalWhisperKernels(device: device, enableTensorEncoder: false)
        try verifyFlashAttentionAccuracy(
            device: device, pipeline: kernels.encoderFlashAttentionF16,
            label: "Portable FlashAttention", threads: 128, context: 70
        )
    }

    func testRegisterFlashAttentionMatchesFP32AndPortableHalfOutput() throws {
        let device = try requireMetalDevice()
        try requireApple10(device)
        let kernels = try MetalWhisperKernels(device: device, enableTensorEncoder: false)
        let pipeline = try XCTUnwrap(kernels.encoderFlashAttentionRegisterF16)
        for context in [70, 1500] {
            try verifyFlashAttentionAccuracy(
                device: device, pipeline: pipeline, label: "Register FlashAttention",
                threads: 128, context: context, referencePipeline: kernels.encoderFlashAttentionF16
            )
        }
    }

    func testCooperativeFlashAttentionMatchesFP32AndPortableFinalTile() throws {
        let device = try requireMetalDevice()
        try requireApple10(device)
        let kernels = try MetalWhisperKernels(
            device: device, enableTensorEncoder: false, enableCooperativeAttention: true
        )
        let cooperative = try XCTUnwrap(kernels.encoderFlashAttentionCooperative)
        for context in [70, 1500] {
            try verifyFlashAttentionAccuracy(
                device: device, pipeline: cooperative.tiles, label: "Cooperative FlashAttention",
                threads: 64, context: context, referencePipeline: kernels.encoderFlashAttentionF16,
                finalTile: cooperative.finalTile
            )
        }
    }

    func testF16TensorMatmulMatchesFP32() throws {
        let device = try requireMetalDevice()
        try requireApple10(device)
        let kernels = try MetalWhisperKernels(device: device)
        try verifyTensorLinearAccuracy(device: device, kernels: XCTUnwrap(kernels.tensorEncoder))
    }

    func testQ5TensorMatmulMatchesFP32() throws {
        let device = try requireMetalDevice()
        guard #available(macOS 26, *) else { throw XCTSkip("Metal 4 requires macOS 26") }
        try requireApple10(device)
        let kernels = try MetalWhisperDecoderTensorKernels(device: device)
        try verifyQ5TensorAccuracy(device: device, kernels: kernels)
    }

    private func requireApple10(_ device: MTLDevice) throws {
        guard MetalWhisperKernels.supportsApple10Optimizations(device: device) else {
            throw XCTSkip("Apple10 and macOS 26 are required for this Metal 4 kernel")
        }
    }

    func testKVSlotsRecycleAcross89PacksAndRetries() throws {
        let crossCount = 8
        let selfCount = crossCount * 1319
        var pool = MetalWhisperKVSlotPool(crossCount: crossCount, selfCount: selfCount)
        var live: [Int: Set<UInt32>] = [:]

        for pack in 0..<89 {
            if pool.freeCrossCount == 0 {
                let finished = live.keys.sorted()[pack % live.count]
                live.removeValue(forKey: finished)
                pool.release(crossSlot: finished)
            }
            let crossSlot = try XCTUnwrap(pool.takeCrossSlot())
            XCTAssertTrue(live[crossSlot] == nil)
            let occupied = Set(live.values.flatMap { $0 })
            var allocated = Set<UInt32>()
            for _ in 0..<(204 + (19 + pack % 205) * 5) {
                let slot = try pool.takeSelfSlot(crossSlot: crossSlot)
                XCTAssertTrue(!occupied.contains(slot) && allocated.insert(slot).inserted)
            }
            if pack.isMultiple(of: 3) {
                let freeCrossBeforeRetry = pool.freeCrossCount
                pool.resetSelfSlots(crossSlot: crossSlot)
                XCTAssertTrue(pool.freeCrossCount == freeCrossBeforeRetry)
                allocated.removeAll()
                for _ in 0..<67 {
                    let slot = try pool.takeSelfSlot(crossSlot: crossSlot)
                    XCTAssertTrue(!occupied.contains(slot) && allocated.insert(slot).inserted)
                }
            }
            live[crossSlot] = allocated
            XCTAssertTrue(pool.freeSelfCount + live.values.reduce(0) { $0 + $1.count } == selfCount)
            XCTAssertTrue(live.values.allSatisfy { slots in
                slots.allSatisfy { Int($0) < pool.selfHighWaterMark }
            })
        }

        for crossSlot in live.keys {
            pool.release(crossSlot: crossSlot)
        }
        XCTAssertTrue(pool.freeCrossCount == crossCount && pool.freeSelfCount == selfCount)

        // Exhaust the recycled free lists to verify that every slot comes back exactly once.
        let recycled = try (0..<crossCount).map { _ in try XCTUnwrap(pool.takeCrossSlot()) }
        XCTAssertTrue(Set(recycled).count == crossCount)
        var selfSlots = Set<UInt32>()
        for i in 0..<selfCount {
            let slot = try pool.takeSelfSlot(crossSlot: recycled[i % crossCount])
            XCTAssertTrue(slot < UInt32(selfCount) && selfSlots.insert(slot).inserted)
        }
        for crossSlot in recycled {
            pool.release(crossSlot: crossSlot)
        }
        XCTAssertTrue(pool.freeCrossCount == crossCount && pool.freeSelfCount == selfCount)
        XCTAssertTrue(pool.selfHighWaterMark == selfCount)
    }

    func testErfAndGELUMatchLibm() throws {
        let device = try requireMetalDevice()
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let library = try MetalWhisperKernels.makeLibrary(device: device, appending: """
        kernel void verify_erf_gelu(
            device const float *inputs [[buffer(0)]],
            device float2 *results [[buffer(1)]],
            uint i [[thread_position_in_grid]]
        ) {
            results[i] = float2(whisper_erf(inputs[i]), gelu_erf(inputs[i]));
        }
        """)
        let function = try XCTUnwrap(library.makeFunction(name: "verify_erf_gelu"))
        let pipeline = try device.makeComputePipelineState(function: function)
        var values = (0...65536).map { Float($0) / 4096 - 8 }
        values += (0...512).flatMap { k -> [Float] in
            let boundary = (Float(k) + 0.5) / 128
            return [boundary.nextDown, boundary, boundary.nextUp,
                    -boundary.nextDown, -boundary, -boundary.nextUp]
        }
        let inputs = try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
        let results = try XCTUnwrap(device.makeBuffer(length: values.count * 8, options: .storageModeShared))
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputs, offset: 0, index: 0)
        enc.setBuffer(results, offset: 0, index: 1)
        enc.dispatchThreads(
            MTLSize(width: values.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }

        let got = results.contents().bindMemory(to: SIMD2<Float>.self, capacity: values.count)
        var maxErfError: Float = 0
        var maxGeluError: Float = 0
        for (i, x) in values.enumerated() {
            let exact = Double(x)
            let refErf = Float(erf(exact))
            let refGelu = Float(0.5 * exact * (1 + erf(exact / sqrt(2))))
            XCTAssertTrue(got[i].x.isFinite && got[i].y.isFinite)
            maxErfError = max(maxErfError, abs(got[i].x - refErf))
            maxGeluError = max(maxGeluError, abs(got[i].y - refGelu))
        }
        XCTAssertLessThan(maxErfError, 1.5e-7)
        XCTAssertLessThan(maxGeluError, 1e-6)
    }

    private func verifyCrossKVQ8(packed: Bool) throws {
        let device = try requireMetalDevice()
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let packs = 2, frames = 13, heads = 2, headDim = 64
        let width = heads * headDim
        let slots = [2, 0]
        let layerOffset = 3 * heads * frames * headDim
        let sampleCount = 2 * layerOffset
        let frameCount = sampleCount / headDim
        let bias = (0..<(2 * width)).map { i in
            i % width < headDim ? Float(0) : sinf(Float(i) * 0.081) * 0.12
        }
        var input = (0..<(packs * frames * 2 * width)).map { i in
            Float16(sinf(Float(i) * 0.071) * cosf(Float(i) * 0.023))
        }
        for offset in 0..<headDim {
            input[offset] = 0
            input[width + offset] = 0
        }
        let src = try XCTUnwrap(device.makeBuffer(bytes: input, length: input.count * 2, options: .storageModeShared))
        let biases = try XCTUnwrap(device.makeBuffer(bytes: bias, length: bias.count * 4, options: .storageModeShared))
        let format: MetalWhisperCrossKVFormat = packed ? .q8Packed : .q8
        let quantized = try MetalWhisperKernels(device: device, enableTensorEncoder: false, crossKVFormat: format)
        let keys = try XCTUnwrap(device.makeBuffer(length: packed ? sampleCount * 2 : sampleCount, options: .storageModeShared))
        let values = packed ? keys : try XCTUnwrap(device.makeBuffer(length: sampleCount, options: .storageModeShared))
        let keyScales = try XCTUnwrap(device.makeBuffer(length: packed ? frameCount * 4 : frameCount * 2, options: .storageModeShared))
        let valueScales = packed ? keyScales : try XCTUnwrap(device.makeBuffer(length: frameCount * 2, options: .storageModeShared))
        let distinctBuffers = packed ? [keys, keyScales] : [keys, values, keyScales, valueScales]
        for buffer in distinctBuffers {
            memset(buffer.contents(), 0x80, buffer.length)
        }
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(try XCTUnwrap(quantized.splitCrossKVQ8))
        for (index, buffer) in [src, biases, keys, values].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var params: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (
            UInt32(packs), UInt32(frames), UInt32(heads), UInt32(headDim),
            UInt32(layerOffset), UInt32(slots[0]), UInt32(slots[1]), 0, 0
        )
        encoder.setBytes(&params, length: 36, index: 4)
        encoder.setBuffer(keyScales, offset: 0, index: 5)
        encoder.setBuffer(valueScales, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (frames + 7) / 8, height: heads, depth: packs),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }

        for (plane, pair) in [(keys, keyScales), (values, valueScales)].enumerated() {
            let codes = pair.0.contents().assumingMemoryBound(to: Int8.self)
            let scales = pair.1.contents().assumingMemoryBound(to: Float16.self)
            let samplesPerSlot = heads * frames * headDim
            let untouched = [0..<layerOffset, (layerOffset + samplesPerSlot)..<(layerOffset + 2 * samplesPerSlot)]
            for range in untouched {
                let codeBytesPerSample = packed ? 2 : 1
                let codeBytes = UnsafeRawBufferPointer(
                    start: pair.0.contents().advanced(by: range.lowerBound * codeBytesPerSample),
                    count: range.count * codeBytesPerSample
                )
                let scaleBytesPerFrame = packed ? 4 : 2
                let scaleBytes = UnsafeRawBufferPointer(
                    start: pair.1.contents().advanced(by: range.lowerBound / headDim * scaleBytesPerFrame),
                    count: range.count / headDim * scaleBytesPerFrame
                )
                XCTAssertTrue(codeBytes.allSatisfy { $0 == 0x80 })
                XCTAssertTrue(scaleBytes.allSatisfy { $0 == 0x80 })
            }
            for pack in 0..<packs {
                for frame in 0..<frames {
                    for head in 0..<heads {
                        let projection = head * headDim + plane * width
                        let start = (pack * frames + frame) * 2 * width + projection
                        let reference = (0..<headDim).map { d in
                            Float(Float16(Float(input[start + d]) + bias[projection + d]))
                        }
                        let maximum = reference.map(abs).max() ?? 0
                        let scale = maximum > 0 ? maximum / 127 : 0
                        let reciprocal = maximum > 0 ? 127 / maximum : 0
                        let offset = layerOffset + ((slots[pack] * heads + head) * frames + frame) * headDim
                        let scaleIndex = packed ? offset / 32 + plane : offset / headDim
                        XCTAssertTrue(scales[scaleIndex].bitPattern == Float16(scale).bitPattern)
                        for d in 0..<headDim {
                            let code = Int((reference[d] * reciprocal).rounded(.toNearestOrEven))
                            let codeIndex = packed
                                ? offset * 2 + (d % 16) * 8 + (d / 16 % 2) * 4 + plane * 2 + d / 32
                                : offset + (d % 32) * 2 + d / 32
                            XCTAssertTrue(codes[codeIndex] == Int8(max(-127, min(127, code))))
                        }
                    }
                }
            }
        }
    }

    private func verifyFlashAttentionAccuracy(
        device: MTLDevice, pipeline: MTLComputePipelineState, label: String, threads: Int, context: Int,
        referencePipeline: MTLComputePipelineState? = nil, finalTile: MTLComputePipelineState? = nil
    ) throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let P = 1, T = context, nHeads = 2, headDim = 64
        let count = P * nHeads * T * headDim
        var qH = [Float16](repeating: 0, count: count)
        var kH = [Float16](repeating: 0, count: count)
        var vH = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            qH[i] = Float16(sin(Float(i) * 0.017) * 0.7)
            kH[i] = Float16(cos(Float(i) * 0.019) * 0.7)
            vH[i] = Float16(sin(Float(i) * 0.023 + 0.5) * 0.7)
        }
        let qBuf = try XCTUnwrap(device.makeBuffer(bytes: qH, length: count * 2, options: .storageModeShared))
        let kBuf = try XCTUnwrap(device.makeBuffer(bytes: kH, length: count * 2, options: .storageModeShared))
        let vBuf = try XCTUnwrap(device.makeBuffer(bytes: vH, length: count * 2, options: .storageModeShared))
        let completeRows = (T - 1) / MetalWhisperKernels.CooperativeAttention.rowsPerGroup
            * MetalWhisperKernels.CooperativeAttention.rowsPerGroup
        func runFlash(
            _ pipeline: MTLComputePipelineState, threads: Int, finalTile: MTLComputePipelineState? = nil
        ) throws -> MTLBuffer {
            let output = try XCTUnwrap(device.makeBuffer(length: count * 2, options: .storageModeShared))
            let cb = try XCTUnwrap(queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pipeline)
            var params: (UInt32, UInt32, UInt32, UInt32) = (UInt32(P), UInt32(T), UInt32(nHeads), UInt32(headDim))
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2)
            enc.setBuffer(output, offset: 0, index: 3)
            enc.setBytes(&params, length: 16, index: 4)
            let groups = (T + 31) / 32
            if let finalTile {
                if completeRows > 0 {
                    enc.dispatchThreadgroups(
                        MTLSize(width: completeRows / MetalWhisperKernels.CooperativeAttention.rowsPerGroup,
                                height: nHeads, depth: P),
                        threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
                    )
                }
                enc.setComputePipelineState(finalTile)
                enc.dispatchThreadgroups(
                    MTLSize(width: groups - completeRows / 32, height: nHeads, depth: P),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
            } else {
                enc.dispatchThreadgroups(
                    MTLSize(width: groups, height: nHeads, depth: P),
                    threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
                )
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            return output
        }

        let oBuf = try runFlash(pipeline, threads: threads, finalTile: finalTile)
        if let referencePipeline {
            let reference = try runFlash(referencePipeline, threads: 128)
            let start = finalTile == nil ? 0 : completeRows * nHeads * headDim
            XCTAssertTrue(
                memcmp(oBuf.contents().advanced(by: start * 2), reference.contents().advanced(by: start * 2),
                       (count - start) * 2) == 0,
                "FlashAttention changed the reference F16 output at T=\(T)"
            )
        }

        let oPtr = oBuf.contents().bindMemory(to: Float16.self, capacity: count)
        var maxErr: Float = 0
        let scale: Float = 1.0 / 8.0
        for h in 0..<nHeads {
            let hBase = h * T * headDim
            for tq in [0, 15, 31, 32, 63, 64, T - 1] {
                var scores = [Float](repeating: 0, count: T)
                var maxS: Float = -Float.infinity
                for tk in 0..<T {
                    var s: Float = 0
                    for d in 0..<headDim {
                        s += Float(qH[hBase + tq * headDim + d]) * Float(kH[hBase + tk * headDim + d])
                    }
                    s *= scale
                    scores[tk] = s
                    if s > maxS { maxS = s }
                }
                var sumE: Float = 0
                for tk in 0..<T { scores[tk] = exp(scores[tk] - maxS); sumE += scores[tk] }
                for d in 0..<headDim {
                    var ref: Float = 0
                    for tk in 0..<T {
                        ref += (scores[tk] / sumE) * Float(vH[hBase + tk * headDim + d])
                    }
                    let got = Float(oPtr[tq * (nHeads * headDim) + h * headDim + d])
                    maxErr = max(maxErr, abs(ref - got))
                }
            }
        }
        XCTAssertLessThan(maxErr, 1e-4, "\(label), context \(T)")
    }

    private func verifyTensorLinearAccuracy(
        device: MTLDevice, kernels: MetalWhisperEncoderTensorKernels
    ) throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let m = 5, n = 64, k = 1280
        let a = (0..<(m * k)).map { Float16(sin(Float($0) * 0.025) * 0.35) }
        let b = (0..<(n * k)).map { Float16(cos(Float($0) * 0.047) * 0.08) }
        let bias = (0..<n).map { sin(Float($0) * 0.15) * 0.03 }
        let residual = (0..<(m * n)).map { Float16(cos(Float($0) * 0.03) * 0.2) }
        let aBuf = try XCTUnwrap(device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared))
        let bBuf = try XCTUnwrap(device.makeBuffer(bytes: b, length: b.count * 2, options: .storageModeShared))
        let biasBuf = try XCTUnwrap(device.makeBuffer(bytes: bias, length: bias.count * 4, options: .storageModeShared))
        let resBuf = try XCTUnwrap(device.makeBuffer(bytes: residual, length: residual.count * 2, options: .storageModeShared))
        let out = try XCTUnwrap(device.makeBuffer(length: m * n * 4, options: .storageModeShared))
        var reference = [Float](repeating: 0, count: m * n)
        for row in 0..<m {
            for col in 0..<n {
                for j in 0..<k {
                    reference[row * n + col] += Float(a[row * k + j]) * Float(b[col * k + j])
                }
            }
        }
        var maxError: Float = 0
        for mode: MetalWhisperLinearParams.Epilogue in [.half, .gelu, .residual, .logits] {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            kernels.dispatchLinear(
                encoder,
                params: MetalWhisperLinearParams(rows: m, columns: n, inner: k, epilogue: mode),
                input: aBuf, weights: bBuf, output: out, bias: biasBuf, auxiliary: resBuf, extra: nil
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw error }

            let result16 = out.contents().assumingMemoryBound(to: Float16.self)
            let result32 = out.contents().assumingMemoryBound(to: Float.self)
            let resultResidual = resBuf.contents().assumingMemoryBound(to: Float16.self)
            for index in reference.indices {
                let rounded = Float(Float16(reference[index]))
                let expected: Float
                let actual: Float
                switch mode {
                case .half:
                    expected = reference[index]
                    actual = Float(result16[index])
                case .gelu:
                    let x = Double(rounded + bias[index % n])
                    expected = Float(0.5 * x * (1 + erf(x / sqrt(2))))
                    actual = Float(result16[index])
                case .residual:
                    expected = Float(residual[index]) + rounded + bias[index % n]
                    actual = result32[index]
                    XCTAssertTrue(resultResidual[index] == Float16(actual))
                case .logits:
                    expected = reference[index]
                    actual = result32[index]
                default:
                    return XCTFail("Unexpected linear epilogue")
                }
                XCTAssertTrue(actual.isFinite)
                maxError = max(maxError, abs(expected - actual))
            }
        }
        XCTAssertLessThan(maxError, 0.002)
    }

    private func verifyQ5TensorAccuracy(
        device: MTLDevice, kernels: MetalWhisperDecoderTensorKernels
    ) throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let rows = 40, columns = 97, inner = 1280
        let blocksPerRow = inner / 32
        let count = columns * blocksPerRow
        let highOffset = count * 16
        let scalesOffset = (highOffset + count * 4 + 15) & ~15
        let q5 = try XCTUnwrap(device.makeBuffer(length: scalesOffset + count * 2, options: .storageModeShared))
        let packed = q5.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: count)
        let high = q5.contents().advanced(by: highOffset).bindMemory(to: UInt32.self, capacity: count)
        let scales = q5.contents().advanced(by: scalesOffset).bindMemory(to: UInt16.self, capacity: count)
        var ggml = [UInt8](repeating: 0, count: count * 22)

        func write<T>(_ value: T, at offset: Int, in bytes: UnsafeMutableRawBufferPointer) {
            withUnsafeBytes(of: value) { raw in
                bytes.baseAddress!.advanced(by: offset).copyMemory(from: raw.baseAddress!, byteCount: raw.count)
            }
        }

        ggml.withUnsafeMutableBytes { bytes in
            for column in 0..<columns {
                for block in 0..<blocksPerRow {
                    let rowMajor = column * blocksPerRow + block
                    let blockMajor = block * columns + column
                    let scale = Float16(Float(rowMajor % 29 - 14) / 4096).bitPattern
                    let seed = UInt32(rowMajor + 1)
                    let qh = seed &* 0x91F28753
                    let qs = SIMD4<UInt32>(seed &* 0xAD167FC3, seed &* 0xA46BCC59, ~qh, qh ^ 0xF8CAED12)
                    write(scale, at: rowMajor * 22, in: bytes)
                    write(qh, at: rowMajor * 22 + 2, in: bytes)
                    write(qs, at: rowMajor * 22 + 6, in: bytes)
                    packed[blockMajor] = qs
                    high[blockMajor] = qh
                    scales[blockMajor] = scale
                }
            }
        }
        var dense = [Float16](repeating: 0, count: columns * inner)
        ggml.withUnsafeBytes { src in
            dense.withUnsafeMutableBufferPointer { dst in
                MetalWhisperModel.dequantizeQ5_0ToF16(src: src.baseAddress!, dst: dst.baseAddress!, count: dst.count)
            }
        }
        let q5Weights = MetalWhisperQ5Weights(buffer: q5, highBitsOffset: highOffset, scalesOffset: scalesOffset)
        let input = (0..<(rows * inner)).map { Float16(sin(Float($0) * 0.37) * 0.25) }
        let inputBuffer = try XCTUnwrap(device.makeBuffer(bytes: input, length: input.count * 2, options: .storageModeShared))
        let output = try XCTUnwrap(device.makeBuffer(length: rows * columns * 4, options: .storageModeShared))
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        kernels.dispatch(
            encoder, rows: rows, columns: columns, inner: inner,
            input: inputBuffer, f16: q5, q5: q5Weights, output: output, epilogue: .logits
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }

        let result = output.contents().assumingMemoryBound(to: Float.self)
        var maxError: Float = 0
        for row in 0..<rows {
            for column in 0..<columns {
                var reference: Float = 0
                for k in 0..<inner {
                    reference.addProduct(Float(input[row * inner + k]), Float(dense[column * inner + k]))
                }
                let actual = result[row * columns + column]
                XCTAssertTrue(actual.isFinite)
                maxError = max(maxError, abs(actual - reference))
            }
        }
        XCTAssertLessThan(maxError, 0.002)
    }
}

// Build from the package root with the macOS 26 SDK or newer.
// swiftc -O -swift-version 6 -Xcc -DACCELERATE_NEW_LAPACK -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
//   Sources/MetalWhisperM5/*.swift Benchmarks/Microbenchmarks/Q5MMABenchmark.swift \
//   -o "/tmp/metal-whisper-q5-mma"
// Run /tmp/metal-whisper-q5-mma --help for the required inputs.

import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

private struct BenchmarkError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

private struct TensorSlice {
    let width: Int
    let rows: Int
    let type: Int
    let offset: Int
}

private func readDecoderTensors(_ data: Data, names: Set<String>) throws -> [String: TensorSlice] {
    try data.withUnsafeBytes { bytes in
        var cursor = 0
        func readWord() throws -> Int {
            guard cursor <= bytes.count - 4 else { throw BenchmarkError("Truncated GGML header") }
            let value = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: cursor, as: UInt32.self))
            cursor += 4
            return Int(value)
        }
        guard try readWord() == 0x67676d6c else { throw BenchmarkError("Invalid GGML magic") }
        cursor += 11 * 4
        let melRows = try readWord()
        let melColumns = try readWord()
        guard melRows <= 128, melColumns <= 201 else { throw BenchmarkError("Invalid mel filter shape") }
        cursor += melRows * melColumns * 4
        let vocabulary = try readWord()
        guard vocabulary <= 100_000 else { throw BenchmarkError("Invalid vocabulary size") }
        for _ in 0..<vocabulary {
            let length = try readWord()
            cursor += length
        }

        var tensors: [String: TensorSlice] = [:]
        while cursor < bytes.count && tensors.count < names.count {
            let dimensions = try readWord()
            let nameLength = try readWord()
            let type = try readWord()
            guard (1...4).contains(dimensions), (1...1024).contains(nameLength) else {
                throw BenchmarkError("Invalid GGML tensor header")
            }
            let width = try readWord()
            var rows = 1
            for _ in 1..<dimensions {
                let dim = try readWord()
                guard dim > 0, dim <= 100_000, rows <= bytes.count / dim else {
                    throw BenchmarkError("Invalid GGML tensor shape")
                }
                rows *= dim
            }
            guard width > 0, rows <= bytes.count / width, nameLength <= bytes.count - cursor else {
                throw BenchmarkError("Truncated GGML tensor name or shape")
            }
            let name = String(decoding: bytes[cursor..<(cursor + nameLength)], as: UTF8.self)
            cursor += nameLength
            let count = width * rows
            let length: Int
            switch type {
            case 0: length = count * 4
            case 1: length = count * 2
            case 6: length = count / 32 * 22
            case 8: length = count / 32 * 34
            default: throw BenchmarkError("Unsupported GGML type \(type)")
            }
            guard length <= bytes.count - cursor else { throw BenchmarkError("Truncated tensor \(name)") }
            if names.contains(name) {
                tensors[name] = TensorSlice(width: width, rows: rows, type: type, offset: cursor)
            }
            cursor += length
        }
        guard tensors.count == names.count else { throw BenchmarkError("Missing Q5_0 benchmark tensors") }
        return tensors
    }
}

private func makeBuffer(_ device: MTLDevice, bytes: Int) throws -> MTLBuffer {
    let aligned = (bytes + 15) & ~15
    guard let buffer = device.makeBuffer(length: aligned, options: .storageModeShared) else {
        throw BenchmarkError("Could not allocate \(aligned) bytes for the benchmark")
    }
    return buffer
}

private struct Q5Matrix {
    let n: Int
    let k: Int
    let scales: MTLBuffer
    let highBits: MTLBuffer
    let nibbles: MTLBuffer
    let weightsF16: MTLBuffer
    let bias: MTLBuffer

    init(device: MTLDevice, data: Data, weights: TensorSlice, biases: TensorSlice?) throws {
        n = weights.rows
        k = weights.width
        guard k.isMultiple(of: 32), weights.type == 6 else {
            throw BenchmarkError("A Q5_0 matrix with K divisible by 32 is required")
        }
        let blocksPerRow = k / 32
        let blocks = n * blocksPerRow
        scales = try makeBuffer(device, bytes: blocks * 2)
        highBits = try makeBuffer(device, bytes: blocks * 4)
        nibbles = try makeBuffer(device, bytes: blocks * 16)
        weightsF16 = try makeBuffer(device, bytes: n * k * 2)
        bias = try makeBuffer(device, bytes: n * 4)

        let d = scales.contents().bindMemory(to: UInt16.self, capacity: blocks)
        let qh = highBits.contents().bindMemory(to: UInt32.self, capacity: blocks)
        let qs = nibbles.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: blocks)
        let half = weightsF16.contents().bindMemory(to: Float16.self, capacity: n * k)
        try data.withUnsafeBytes { raw in
            let source = raw.baseAddress!.advanced(by: weights.offset)
            MetalWhisperModel.dequantizeQ5_0ToF16(src: source, dst: half, count: n * k)

            // Transpose whole quantization blocks. The scales, high bits and
            // packed nibbles stay bit for bit identical to the mapped GGML.
            for row in 0..<n {
                for block in 0..<blocksPerRow {
                    let from = (row * blocksPerRow + block) * 22
                    let to = block * n + row
                    d[to] = source.loadUnaligned(fromByteOffset: from, as: UInt16.self)
                    qh[to] = source.loadUnaligned(fromByteOffset: from + 2, as: UInt32.self)
                    qs[to] = source.loadUnaligned(fromByteOffset: from + 6, as: SIMD4<UInt32>.self)
                }
            }

            if let biases {
                guard biases.type == 0, biases.width == n, biases.rows == 1 else {
                    throw BenchmarkError("Invalid FC1 bias tensor shape")
                }
                bias.contents().copyMemory(from: raw.baseAddress!.advanced(by: biases.offset), byteCount: n * 4)
            } else {
                memset(bias.contents(), 0, n * 4)
            }
        }
    }
}

private struct Tile {
    let m: Int
    let n: Int
    let simdRows: Int
    let simdColumns: Int
    let stride: Int
    var tensor = false
    var kBlocks = 1
    var threads: Int { simdRows * simdColumns * 32 }
    var name: String { "q5_\(tensor ? "tensor_" : "")m\(m)_n\(n)_r\(simdRows)_c\(simdColumns)_s\(stride)_k\(kBlocks * 32)" }

    var wrapper: String {
        """
        kernel void \(name)(
            device const half* inputs [[buffer(0)]],
            device const half* scales [[buffer(1)]],
            device const uint* highBits [[buffer(2)]],
            device const uint* quants [[buffer(3)]],
            device const float* bias [[buffer(4)]],
            device const half* residual [[buffer(5)]],
            device uchar* output [[buffer(6)]],
            constant Q5BenchParams& params [[buffer(7)]],
            uint2 group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]
        ) {
            threadgroup half inputTile[\(m * stride)];
            threadgroup half weightTile[\(n * stride)];
            threadgroup float outputTile[\(m * n)];
            \(tensor ? "q5_gemm_tensor" : "q5_gemm")<\(m), \(n), \(simdRows), \(simdColumns), \(stride)\(tensor ? ", \(kBlocks)" : "")>(
                inputs, scales, highBits, quants, bias, residual, output, params, group, tid, simd,
                inputTile, weightTile, outputTile
            );
        }
        """
    }
}

private let gemmSource = #"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct Q5BenchParams {
    uint rows;
    uint columns;
    uint inner;
    uint epilogue;
};

template <uint BM, uint BN, uint THREADS, uint STRIDE>
inline void stage_q5_tile(
    device const half* inputs, device const half* scales, device const uint* highBits, device const uint* quants,
    constant Q5BenchParams& params, uint firstRow, uint firstCol, uint block, uint tid,
    threadgroup half* inputTile, threadgroup half* weightTile
) {
    constexpr uint BK = 32;
    for (uint i = tid; i < BM * 8; i += THREADS) {
        const uint r = i / 8;
        const uint k = (i % 8) * 4;
        const uint row = firstRow + r;
        const half4 value = row < params.rows
            ? *(device const half4*)(inputs + row * params.inner + block * BK + k)
            : half4(0.0h);
        *(threadgroup half4*)(inputTile + r * STRIDE + k) = value;
    }
    for (uint i = tid; i < BN * 4; i += THREADS) {
        const uint r = i / 4;
        const uint word = i & 3;
        const uint col = firstCol + r;
        half4 lower = half4(0);
        half4 upper = half4(0);
        if (col < params.columns) {
            const uint idx = block * params.columns + col;
            const uint packed = quants[idx * 4 + word];
            const uint high = highBits[idx];
            const float scale = float(scales[idx]);
            const uint4 lane = word * 4 + uint4(0, 1, 2, 3);
            const uint4 packedBytes = uint4(packed) >> uint4(0, 8, 16, 24);
            const int4 lo = int4((packedBytes & 15u) | (((uint4(high) >> lane) & 1u) << 4u)) - 16;
            const int4 hi = int4(((packedBytes >> 4u) & 15u) | (((uint4(high) >> (lane + 16u)) & 1u) << 4u)) - 16;
            lower = half4(float4(lo) * scale);
            upper = half4(float4(hi) * scale);
        }
        threadgroup half* dst = weightTile + r * STRIDE + word * 4;
        *(threadgroup half4*)dst = lower;
        *(threadgroup half4*)(dst + 16) = upper;
    }
}

inline void store_q5_output(
    device uchar* output, device const float* bias, device const half* residual,
    constant Q5BenchParams& params, uint row, uint col, float sum
) {
    if (row >= params.rows || col >= params.columns) return;
    const uint index = row * params.columns + col;
    if (params.epilogue == 3) {
        ((device float*)output)[index] = sum;
    } else {
        float value = sum + bias[col];
        if (params.epilogue == 1) value = gelu_erf(value);
        if (params.epilogue == 2) value += float(residual[index]);
        ((device half*)output)[index] = half(value);
    }
}

template <uint BM, uint BN, uint SIMD_ROWS, uint SIMD_COLS, uint STRIDE>
inline void q5_gemm(
    device const half* inputs, device const half* scales, device const uint* highBits,
    device const uint* quants, device const float* bias, device const half* residual,
    device uchar* output, constant Q5BenchParams& params, uint2 group, uint tid, uint simd,
    threadgroup half* inputTile, threadgroup half* weightTile, threadgroup float* outputTile
) {
    constexpr uint BK = 32;
    constexpr uint THREADS = SIMD_ROWS * SIMD_COLS * 32;
    constexpr uint RM = BM / SIMD_ROWS / 8;
    constexpr uint RN = BN / SIMD_COLS / 8;
    const uint firstRow = group.x * BM;
    const uint firstCol = group.y * BN;
    const uint simdRow = (simd / SIMD_COLS) * RM * 8;
    const uint simdCol = (simd % SIMD_COLS) * RN * 8;
    simdgroup_matrix<float, 8, 8> acc[RM][RN];

    #pragma clang loop unroll(full)
    for (uint a = 0; a < RM; ++a) {
        #pragma clang loop unroll(full)
        for (uint b = 0; b < RN; ++b) {
            acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint block = 0; block < params.inner / BK; ++block) {
        stage_q5_tile<BM, BN, THREADS, STRIDE>(
            inputs, scales, highBits, quants, params, firstRow, firstCol, block, tid, inputTile, weightTile
        );
        if (THREADS == 32) simdgroup_barrier(mem_flags::mem_threadgroup);
        else threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint k = 0; k < BK; k += 8) {
            simdgroup_matrix<half, 8, 8> x[RM];
            simdgroup_matrix<half, 8, 8> w[RN];
            #pragma clang loop unroll(full)
            for (uint a = 0; a < RM; ++a) {
                simdgroup_load(x[a], inputTile + (simdRow + a * 8) * STRIDE + k, STRIDE);
            }
            #pragma clang loop unroll(full)
            for (uint b = 0; b < RN; ++b) {
                simdgroup_load(w[b], weightTile + (simdCol + b * 8) * STRIDE + k, STRIDE, ulong2(0), true);
            }
            #pragma clang loop unroll(full)
            for (uint a = 0; a < RM; ++a) {
                #pragma clang loop unroll(full)
                for (uint b = 0; b < RN; ++b) {
                    simdgroup_multiply_accumulate(acc[a][b], x[a], w[b], acc[a][b]);
                }
            }
        }
        if (THREADS == 32) simdgroup_barrier(mem_flags::mem_threadgroup);
        else threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint a = 0; a < RM; ++a) {
        #pragma clang loop unroll(full)
        for (uint b = 0; b < RN; ++b) {
            simdgroup_store(acc[a][b], outputTile + (simdRow + a * 8) * BN + simdCol + b * 8, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BM * BN; i += THREADS) {
        const uint row = firstRow + i / BN;
        const uint col = firstCol + i % BN;
        store_q5_output(output, bias, residual, params, row, col, outputTile[i]);
    }
}

template <uint BM, uint BN, uint SIMD_ROWS, uint SIMD_COLS, uint STRIDE, uint K_BLOCKS>
inline void q5_gemm_tensor(
    device const half* inputs, device const half* scales, device const uint* highBits,
    device const uint* quants, device const float* bias, device const half* residual,
    device uchar* output, constant Q5BenchParams& params, uint2 group, uint tid, uint simd,
    threadgroup half* inputTile, threadgroup half* weightTile, threadgroup float* outputTile
) {
    using namespace mpp::tensor_ops;
    constexpr int BK = 32 * K_BLOCKS;
    constexpr uint SIMD_GROUPS = SIMD_ROWS * SIMD_COLS;
    constexpr uint THREADS = SIMD_GROUPS * 32;
    const uint firstRow = group.x * BM;
    const uint firstCol = group.y * BN;
    using LeftShape = extents<int32_t, BK, BM>;
    using RightShape = extents<int32_t, BK, BN>;
    using Left = tensor<threadgroup half, LeftShape, tensor_inline>;
    using Right = tensor<threadgroup half, RightShape, tensor_inline>;
    const array<int32_t, 2> strides = {1, STRIDE};
    auto a = Left(inputTile, LeftShape{}, strides);
    auto b = Right(weightTile, RightShape{}, strides);

    constexpr auto descriptor = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<SIMD_GROUPS>> op;
    auto acc = op.template get_destination_cooperative_tensor<Left, Right, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }
    for (uint block = 0; block < params.inner / BK; ++block) {
        #pragma clang loop unroll(full)
        for (uint part = 0; part < K_BLOCKS; ++part) {
            stage_q5_tile<BM, BN, THREADS, STRIDE>(
                inputs, scales, highBits, quants, params, firstRow, firstCol, block * K_BLOCKS + part, tid,
                inputTile + part * 32, weightTile + part * 32
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a, b, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto xy = acc.get_multidimensional_index(i);
        store_q5_output(output, bias, residual, params, firstRow + xy[1], firstCol + xy[0], acc[i]);
    }
}
"""#

private func gpuMilliseconds(
    _ queue: MTLCommandQueue,
    repeats: Int = 20,
    encode: (MTLCommandBuffer) -> Void
) throws -> Double {
    var samples: [Double] = []
    for i in 0..<(repeats + 5) {
        guard let command = queue.makeCommandBuffer() else { throw BenchmarkError("No command buffer") }
        encode(command)
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        guard command.gpuEndTime > command.gpuStartTime else { throw BenchmarkError("Missing GPU timestamps") }
        if i >= 5 {
            samples.append((command.gpuEndTime - command.gpuStartTime) * 1000)
        }
    }
    return samples.sorted()[samples.count / 2]
}

private func verify(
    reference: [Float], output: MTLBuffer, bias: MTLBuffer, residual: MTLBuffer,
    cols: Int, mode: UInt32
) throws -> (absolute: Float, rounded: Float) {
    let count = reference.count
    let out32 = output.contents().assumingMemoryBound(to: Float.self)
    let out16 = output.contents().assumingMemoryBound(to: Float16.self)
    let b = bias.contents().bindMemory(to: Float.self, capacity: cols)
    let r = residual.contents().bindMemory(to: Float16.self, capacity: count)
    var absolute: Float = 0
    var rounded: Float = 0
    for i in 0..<count {
        var expected = reference[i]
        if mode != 3 {
            expected += b[i % cols]
            if mode == 1 {
                let x = Double(expected)
                expected = Float(0.5 * x * (1 + erf(x / sqrt(2))))
            } else if mode == 2 {
                expected += Float(r[i])
            }
        }
        let actual = mode == 3 ? out32[i] : Float(out16[i])
        guard actual.isFinite else { throw BenchmarkError("Non-finite Q5_0 output") }
        absolute = max(absolute, abs(expected - actual))
        let expectedOutput = mode == 3 ? expected : Float(Float16(expected))
        rounded = max(rounded, abs(expectedOutput - actual))
    }
    return (absolute, rounded)
}

@main
struct MetalWhisperQ5MMABenchmark {
    static func main() {
        do {
            try run()
        } catch {
            FileHandle.standardError.write(Data("Q5_0 benchmark: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func run() throws {
        guard #available(macOS 26.0, *) else {
            throw BenchmarkError("The Metal 4 benchmark requires macOS 26 or later")
        }
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 1, arguments[0] != "--help" else {
            if arguments == ["--help"] {
                print("Usage: metal-whisper-q5-mma <model.bin>")
                return
            }
            throw BenchmarkError("Usage: metal-whisper-q5-mma <model.bin>")
        }
        let modelPath = arguments[0]
        let data = try Data(contentsOf: URL(fileURLWithPath: modelPath), options: .mappedIfSafe)
        let fc1 = "decoder.blocks.0.mlp.0.weight"
        let bias = "decoder.blocks.0.mlp.0.bias"
        let lmHead = "decoder.token_embedding.weight"
        let slices = try readDecoderTensors(data, names: [fc1, bias, lmHead])
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw BenchmarkError("Metal device or command queue unavailable")
        }
        guard MetalWhisperKernels.supportsApple10Optimizations(device: device) else {
            throw BenchmarkError("An Apple10 Metal GPU is required")
        }
        let variants = [
            Tile(m: 16, n: 32, simdRows: 2, simdColumns: 2, stride: 32),
            Tile(m: 16, n: 64, simdRows: 2, simdColumns: 4, stride: 40),
            Tile(m: 32, n: 32, simdRows: 4, simdColumns: 2, stride: 32),
            Tile(m: 32, n: 64, simdRows: 4, simdColumns: 2, stride: 32),
            Tile(m: 32, n: 64, simdRows: 2, simdColumns: 2, stride: 40),
            Tile(m: 32, n: 64, simdRows: 4, simdColumns: 2, stride: 40),
            Tile(m: 16, n: 32, simdRows: 1, simdColumns: 1, stride: 40),
            Tile(m: 32, n: 32, simdRows: 2, simdColumns: 1, stride: 40),
            Tile(m: 32, n: 64, simdRows: 2, simdColumns: 1, stride: 40),
            Tile(m: 16, n: 32, simdRows: 2, simdColumns: 2, stride: 32, tensor: true),
            Tile(m: 16, n: 64, simdRows: 2, simdColumns: 2, stride: 32, tensor: true),
            Tile(m: 32, n: 32, simdRows: 2, simdColumns: 2, stride: 32, tensor: true),
            Tile(m: 32, n: 64, simdRows: 2, simdColumns: 2, stride: 32, tensor: true),
            Tile(m: 32, n: 32, simdRows: 2, simdColumns: 1, stride: 32, tensor: true),
            Tile(m: 32, n: 64, simdRows: 2, simdColumns: 1, stride: 32, tensor: true),
            Tile(m: 32, n: 32, simdRows: 2, simdColumns: 2, stride: 64, tensor: true, kBlocks: 2),
            Tile(m: 32, n: 64, simdRows: 2, simdColumns: 2, stride: 64, tensor: true, kBlocks: 2),
            Tile(m: 32, n: 64, simdRows: 2, simdColumns: 2, stride: 128, tensor: true, kBlocks: 4),
            Tile(m: 32, n: 32, simdRows: 4, simdColumns: 2, stride: 64, tensor: true, kBlocks: 2),
            Tile(m: 32, n: 64, simdRows: 4, simdColumns: 2, stride: 64, tensor: true, kBlocks: 2),
            Tile(m: 48, n: 32, simdRows: 2, simdColumns: 2, stride: 32, tensor: true),
            Tile(m: 48, n: 64, simdRows: 2, simdColumns: 2, stride: 64, tensor: true, kBlocks: 2),
            Tile(m: 64, n: 32, simdRows: 2, simdColumns: 2, stride: 32, tensor: true),
            Tile(m: 64, n: 64, simdRows: 2, simdColumns: 2, stride: 64, tensor: true, kBlocks: 2)
        ]
        let sources = gemmSource + "\n" + variants.map(\.wrapper).joined(separator: "\n")
        let library = try MetalWhisperKernels.makeLibrary(
            device: device, appending: sources, languageVersion: .version4_0
        )
        let pipelines = try variants.map { variant in
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = library.makeFunction(name: variant.name)!
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
            return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        }
        print("Device \(device.name), GPU timestamp medians from 20 warm single-dispatch command buffers")

        var maxError: Float = 0
        for (label, name, epilogue) in [("FC1", fc1, UInt32(1)), ("LMHead", lmHead, UInt32(3))] {
            let matrix = try Q5Matrix(device: device, data: data, weights: slices[name]!, biases: label == "FC1" ? slices[bias]! : nil)
            let n = matrix.n
            let k = matrix.k
            let expectedRows = label == "FC1" ? 5120 : 51866
            guard k == 1280, n == expectedRows else {
                throw BenchmarkError("Unexpected large-v3 matrix shape for \(label): \(n) × \(k)")
            }
            let largestB = 40
            let input = try makeBuffer(device, bytes: largestB * k * 2)
            let residual = try makeBuffer(device, bytes: largestB * n * 2)
            let output = try makeBuffer(device, bytes: largestB * n * 4)
            let mpsOutput = try makeBuffer(device, bytes: largestB * n * 4)
            let input16 = input.contents().bindMemory(to: Float16.self, capacity: largestB * k)
            let residual16 = residual.contents().bindMemory(to: Float16.self, capacity: largestB * n)
            for i in 0..<(largestB * k) {
                input16[i] = Float16(sin(Float(i) * 0.137) * 1.2 + cos(Float(i) * 0.033) * 0.7)
            }
            for i in 0..<(largestB * n) {
                residual16[i] = Float16(Float(i % 31 - 15) / 60)
            }
            let weight16 = matrix.weightsF16.contents().bindMemory(to: Float16.self, capacity: n * k)
            let weight32 = (0..<(n * k)).map { Float(weight16[$0]) }
            let input32 = (0..<(largestB * k)).map { Float(input16[$0]) }

            for b in [1, 5, 16, 20, 40] {
                let a = MPSMatrix(buffer: input, descriptor: .init(rows: b, columns: k, rowBytes: k * 2, dataType: .float16))
                let w = MPSMatrix(buffer: matrix.weightsF16, descriptor: .init(rows: n, columns: k, rowBytes: k * 2, dataType: .float16))
                let c = MPSMatrix(
                    buffer: mpsOutput,
                    descriptor: .init(rows: b, columns: n, rowBytes: n * (epilogue == 3 ? 4 : 2), dataType: epilogue == 3 ? .float32 : .float16)
                )
                let mps = MPSMatrixMultiplication(
                    device: device, transposeLeft: false, transposeRight: true,
                    resultRows: b, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0
                )
                let mpsMs = try gpuMilliseconds(queue) {
                    mps.encode(commandBuffer: $0, leftMatrix: a, rightMatrix: w, resultMatrix: c)
                }
                func encode(_ cb: MTLCommandBuffer, variant: Int, mode: UInt32) {
                    let tile = variants[variant]
                    let encoder = cb.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipelines[variant])
                    let buffers = [input, matrix.scales, matrix.highBits, matrix.nibbles, matrix.bias, residual, output]
                    for (i, buffer) in buffers.enumerated() {
                        encoder.setBuffer(buffer, offset: 0, index: i)
                    }
                    var params = SIMD4<UInt32>(UInt32(b), UInt32(n), UInt32(k), mode)
                    encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 7)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (b + tile.m - 1) / tile.m, height: (n + tile.n - 1) / tile.n, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tile.threads, height: 1, depth: 1)
                    )
                    encoder.endEncoding()
                }

                let timings = try variants.indices.map { index in
                    try gpuMilliseconds(queue) { encode($0, variant: index, mode: epilogue) }
                }
                let winner = timings.indices.min(by: { timings[$0] < timings[$1] })!
                let details = timings.map { String(format: "%.4f", $0) }.joined(separator: "/")
                print(String(
                    format: "%@ B=%d MPS %.4f ms Q5 %.4f ms (%@), all %@",
                    label, b, mpsMs, timings[winner], variants[winner].name, details
                ))

                var reference = [Float](repeating: 0, count: b * n)
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(b), Int32(n), Int32(k), 1, input32, Int32(k), weight32, Int32(k), 0, &reference, Int32(n)
                )
                for mode in UInt32(0)...3 {
                    let command = queue.makeCommandBuffer()!
                    encode(command, variant: winner, mode: mode)
                    command.commit()
                    command.waitUntilCompleted()
                    if let error = command.error { throw error }
                    let error = try verify(reference: reference, output: output, bias: matrix.bias, residual: residual, cols: n, mode: mode)
                    maxError = max(maxError, error.absolute)
                    print(String(format: "  mode %d max abs %.7g, storage-rounded %.7g", mode, error.absolute, error.rounded))
                }
            }
        }
        print(String(format: "Maximum FP32 reference error: %.7g", maxError))
        let numericPass = maxError < 0.002
        print("Q5_0 MMA FP32 error < 0.002: \(numericPass)")
        if !numericPass { exit(2) }
    }
}

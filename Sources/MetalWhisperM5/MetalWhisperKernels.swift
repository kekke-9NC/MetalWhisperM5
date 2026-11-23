import Foundation
import Metal

/// Compiles the portable and supported accelerated Metal pipelines at runtime.
final class MetalWhisperKernels: @unchecked Sendable {
    let device: MTLDevice

    // Encoder kernels
    let im2colConv1DF32ToF16: MTLComputePipelineState
    let im2colConv1DF16ToF16: MTLComputePipelineState
    let addBiasGeluPosF16: MTLComputePipelineState
    let layerNormF16: MTLComputePipelineState
    let addBiasF16: MTLComputePipelineState
    let addBiasResidualF16: MTLComputePipelineState
    let addBiasResidualLayerNormF16: MTLComputePipelineState
    let addBiasGeluF16: MTLComputePipelineState
    let addBiasSplitQKVF16: MTLComputePipelineState
    let encoderFlashAttentionF16: MTLComputePipelineState
    let encoderFlashAttentionRegisterF16: MTLComputePipelineState?
    struct CooperativeAttention {
        static let rowsPerGroup = 64

        let tiles: MTLComputePipelineState
        let finalTile: MTLComputePipelineState
    }
    let encoderFlashAttentionCooperative: CooperativeAttention?
    let splitCrossKVF16: MTLComputePipelineState
    let splitCrossKVQ8: MTLComputePipelineState?
    let tensorEncoder: MetalWhisperEncoderTensorKernels?

    // Decoder kernels
    let decoderEmbedLookupF16: MTLComputePipelineState
    let decoderSelfAttnBeamTreeF16: MTLComputePipelineState
    let decoderSelfAttnPrefillF16: MTLComputePipelineState
    let decoderCrossAttnPackFusedF16: MTLComputePipelineState
    let decoderCrossAttnPackFusedQ8: MTLComputePipelineState?
    let whisperLogitsFilterTopKF32: MTLComputePipelineState

    init(
        device: MTLDevice, enableTensorEncoder: Bool = true, enableCooperativeAttention: Bool = false,
        crossKVFormat: MetalWhisperCrossKVFormat = .f16, enableRegisterAttention: Bool = true
    ) throws {
        self.device = device
        let supportsOptimizations = Self.supportsApple10Optimizations(device: device)
        let crossKVFormat = crossKVFormat == .auto ? .f16 : crossKVFormat
        let crossKVSource: String
        switch crossKVFormat {
        case .f16:
            crossKVSource = ""
        case .q8Packed:
            crossKVSource = Self.packedQ8Source
        default:
            crossKVSource = Self.quantizedCrossKVSource
        }
        let registerLibrary: MTLLibrary?
        if enableRegisterAttention, supportsOptimizations {
            registerLibrary = try? Self.makeLibrary(device: device, appending: Self.registerFlashSource + "\n" + crossKVSource)
        } else {
            registerLibrary = nil
        }
        let library = try registerLibrary ?? Self.makeLibrary(device: device, appending: crossKVSource)

        func makePSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw MetalWhisperError.shaderCompilationFailed("Missing Metal function \(name)")
            }
            do {
                return try device.makeComputePipelineState(function: fn)
            } catch {
                throw MetalWhisperError.shaderCompilationFailed("\(name): \(error.localizedDescription)")
            }
        }

        self.im2colConv1DF32ToF16 = try makePSO("im2col_conv1d_f32_to_f16")
        self.im2colConv1DF16ToF16 = try makePSO("im2col_conv1d_f16_to_f16")
        self.addBiasGeluPosF16 = try makePSO("add_bias_gelu_pos_f16")
        self.layerNormF16 = try makePSO("layernorm_f16")
        self.addBiasF16 = try makePSO("add_bias_f16")
        self.addBiasResidualF16 = try makePSO("add_bias_residual_f16")
        self.addBiasResidualLayerNormF16 = try makePSO("add_bias_residual_layernorm_f16")
        self.addBiasGeluF16 = try makePSO("add_bias_gelu_f16")
        self.addBiasSplitQKVF16 = try makePSO("add_bias_split_qkv_f16")
        self.encoderFlashAttentionF16 = try makePSO("encoder_flash_attention_f16")
        self.encoderFlashAttentionRegisterF16 = registerLibrary.flatMap { _ in
            try? makePSO("encoder_flash_attention_register_f16")
        }
        self.splitCrossKVF16 = try makePSO("split_cross_kv_f16")

        if #available(macOS 26.0, *), enableCooperativeAttention, supportsOptimizations {
            self.encoderFlashAttentionCooperative = try? Self.makeCooperativeAttention(device: device)
        } else {
            self.encoderFlashAttentionCooperative = nil
        }

        if #available(macOS 26.0, *), enableTensorEncoder, supportsOptimizations {
            self.tensorEncoder = try? MetalWhisperEncoderTensorKernels(device: device)
        } else {
            self.tensorEncoder = nil
        }

        self.decoderEmbedLookupF16 = try makePSO("decoder_embed_lookup_f16")
        self.decoderSelfAttnBeamTreeF16 = try makePSO("decoder_self_attn_beam_tree_f16")
        self.decoderSelfAttnPrefillF16 = try makePSO("decoder_self_attn_prefill_f16")
        self.decoderCrossAttnPackFusedF16 = try makePSO("decoder_cross_attn_pack_fused_f16")
        self.whisperLogitsFilterTopKF32 = try makePSO("whisper_logits_filter_logsoftmax_topk_f32")
        if crossKVFormat == .q8Packed {
            splitCrossKVQ8 = try makePSO("split_cross_kv_q8_packed")
            decoderCrossAttnPackFusedQ8 = try makePSO("decoder_cross_attn_q8_packed")
        } else if crossKVFormat != .f16 {
            let constants = MTLFunctionConstantValues()
            var quantizeKeys = crossKVFormat.quantizesKeys
            var quantizeValues = crossKVFormat.quantizesValues
            constants.setConstantValue(&quantizeKeys, type: .bool, index: 0)
            constants.setConstantValue(&quantizeValues, type: .bool, index: 1)
            func makeQuantizedPSO(_ name: String) throws -> MTLComputePipelineState {
                let function = try library.makeFunction(name: name, constantValues: constants)
                let descriptor = MTLComputePipelineDescriptor()
                descriptor.computeFunction = function
                descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
                return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
            }
            splitCrossKVQ8 = try makeQuantizedPSO("split_cross_kv_q8")
            decoderCrossAttnPackFusedQ8 = try makeQuantizedPSO("decoder_cross_attn_pack_fused_q8")
        } else {
            splitCrossKVQ8 = nil
            decoderCrossAttnPackFusedQ8 = nil
        }
    }

    static func supportsApple10Optimizations(device: MTLDevice) -> Bool {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            return device.supportsFamily(.apple10)
        }
        #endif
        return false
    }

    @available(macOS 26.0, *)
    static func makeMetal4Library(device: MTLDevice, appending source: String) throws -> MTLLibrary {
        #if compiler(>=6.2)
        return try makeLibrary(device: device, appending: source, languageVersion: .version4_0)
        #else
        throw MetalWhisperError.shaderCompilationFailed("Metal 4 requires the macOS 26 SDK or newer.")
        #endif
    }

    @available(macOS 26.0, *)
    private static func makeCooperativeAttention(device: MTLDevice) throws -> CooperativeAttention {
        let library = try makeMetal4Library(
            device: device, appending: registerFlashSource + "\n" + cooperativeFlashSource
        )
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalWhisperError.shaderCompilationFailed("Missing Metal function \(name)")
            }
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
            return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        }
        return CooperativeAttention(
            tiles: try pipeline("encoder_flash_attention_cooperative_f16"),
            finalTile: try pipeline("encoder_flash_attention_final_tile_f16")
        )
    }

    static func makeLibrary(
        device: MTLDevice,
        appending source: String = "",
        languageVersion: MTLLanguageVersion? = nil
    ) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .safe // Strict IEEE-754 precision for zero accuracy drift
            options.languageVersion = languageVersion ?? .version3_2
        } else {
            options.fastMathEnabled = false
            options.languageVersion = languageVersion ?? .version3_1
        }

        do {
            return try device.makeLibrary(source: metalSource + "\n" + source, options: options)
        } catch {
            throw MetalWhisperError.shaderCompilationFailed(error.localizedDescription)
        }
    }

    // Sample libm in Double and retain the rounding residual. The shader evaluates the
    // local erf Taylor series on a grid with at most 1/256 distance to the nearest node.
    private static let erfTableSource: String = {
        let rows = (0...512).map { i in
            let x = Double(i) / 128
            let value = erf(x)
            let high = Float(value)
            let low = Float(value - Double(high))
            let slope = Float((2 / sqrt(Double.pi)) * exp(-x * x))
            return "float3(\(high)f, \(low)f, \(slope)f)"
        }
        return "constant float3 erf_table[513] = {\n" + rows.joined(separator: ",\n") + "\n};"
    }()

    // MARK: - Metal Shading Language (MSL) Kernels

    // The 32-key softmax and F16 probability boundary match the Metal 3.2 kernel.
    private static let registerFlashSource = #"""
    inline void encoder_flash_attention_register(
        device const half* q, device const half* k, device const half* v, device half* o,
        constant SplitQKVParams& p, uint3 group, uint tid, uint lane, uint sg,
        threadgroup half* tileK, threadgroup half* tileV
    ) {
        constexpr uint HD = 64;
        constexpr uint QueryRows = 32;
        constexpr uint KeyRows = 32;
        constexpr uint THREADS = 128;
        // Pad each row by eight half values to reduce threadgroup bank conflicts.
        constexpr uint STRIDE_K = KeyRows + 8;
        constexpr uint STRIDE_V = HD + 8;

        // Apple10 assigns two consecutive columns of each 8x8 fragment to each lane.
        const uint rowInFragment = ((lane >> 1) & 3) | ((lane >> 2) & 4);
        const uint colInFragment = ((lane & 1) << 1) | ((lane & 8) >> 1);
        const uint firstRow = group.x * QueryRows + sg * 8;
        const uint row = firstRow + rowInFragment;
        const uint headOffset = (group.z * p.nHeads + group.y) * p.T * HD;

        simdgroup_half8x8 query[8];
        if (firstRow + 8 <= p.T) {
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                simdgroup_load(query[d], q + headOffset + firstRow * HD + d * 8, HD);
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                query[d] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                if (row < p.T) {
                    const device half* in = q + headOffset + row * HD + d * 8 + colInFragment;
                    query[d].thread_elements()[0] = in[0];
                    query[d].thread_elements()[1] = in[1];
                }
            }
        }

        simdgroup_float8x8 accumulators[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            accumulators[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        float rowMax = -INFINITY;
        float rowSum = 0.0f;

        for (uint block = 0; block < p.T; block += KeyRows) {
            for (uint index = tid * 4; index < KeyRows * HD; index += THREADS * 4) {
                const uint source = headOffset + block * HD + index;
                const bool valid = block + index / HD < p.T;
                const uint rowInTile = index / HD;
                const uint colInTile = index % HD;
                const half4 a = valid ? *(device const half4*)(k + source) : half4(0.0h);
                #pragma clang loop unroll(full)
                for (uint x = 0; x < 4; ++x) {
                    tileK[(colInTile + x) * STRIDE_K + rowInTile] = a[x];
                }
                *(threadgroup half4*)(tileV + rowInTile * STRIDE_V + colInTile)
                    = valid ? *(device const half4*)(v + source) : half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 dots[4];
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) dots[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                #pragma clang loop unroll(full)
                for (uint n = 0; n < 4; ++n) {
                    simdgroup_half8x8 matrixSlice;
                    simdgroup_load(matrixSlice, tileK + d * 8 * STRIDE_K + n * 8, STRIDE_K);
                    simdgroup_multiply_accumulate(dots[n], query[d], matrixSlice, dots[n]);
                }
            }

            float2 scores[4];
            float maximum = -INFINITY;
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                const uint key = block + n * 8 + colInFragment;
                const auto x = dots[n].thread_elements();
                scores[n] = float2(key < p.T ? x[0] * 0.125f : -INFINITY, key + 1 < p.T ? x[1] * 0.125f : -INFINITY);
                maximum = max(maximum, max(scores[n].x, scores[n].y));
            }
            maximum = max(maximum, simd_shuffle_xor(maximum, ushort(1)));
            maximum = max(maximum, simd_shuffle_xor(maximum, ushort(8)));
            const float nextMax = max(rowMax, maximum);
            const float alpha = exp(rowMax - nextMax);

            simdgroup_half8x8 weights[4];
            float4 pairs;
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                const float2 prob = exp(scores[n] - float2(nextMax));
                pairs[n] = prob.x + prob.y;
                weights[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                weights[n].thread_elements()[0] = half(prob.x);
                weights[n].thread_elements()[1] = half(prob.y);
            }
            // Keep the original 32-key simd_sum order for the FP32 denominator.
            pairs += simd_shuffle_xor(pairs, ushort(1));
            pairs += simd_shuffle_xor(pairs, ushort(8));
            const float sum = (pairs.x + pairs.y) + (pairs.z + pairs.w);
            const float scale = rowSum > 0.0f ? alpha : 1.0f;
            rowMax = nextMax;
            rowSum = rowSum * alpha + sum;

            #pragma clang loop unroll(full)
            for (uint n = 0; n < 8; ++n) {
                accumulators[n].thread_elements() *= scale;
                #pragma clang loop unroll(full)
                for (uint d = 0; d < 4; ++d) {
                    simdgroup_half8x8 matrixSlice;
                    simdgroup_load(matrixSlice, tileV + d * 8 * STRIDE_V + n * 8, STRIDE_V);
                    simdgroup_multiply_accumulate(accumulators[n], weights[d], matrixSlice, accumulators[n]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < p.T) {
            const float inv = 1.0f / rowSum;
            device half* dst = o + (group.z * p.T + row) * (p.nHeads * HD) + group.y * HD + colInFragment;
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 8; ++n) {
                const auto x = accumulators[n].thread_elements();
                dst[n * 8] = half(x[0] * inv);
                dst[n * 8 + 1] = half(x[1] * inv);
            }
        }
    }

    kernel void encoder_flash_attention_register_f16(
        device const half* q [[buffer(0)]],
        device const half* k [[buffer(1)]],
        device const half* v [[buffer(2)]],
        device half* o [[buffer(3)]],
        constant SplitQKVParams& p [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup half tileK[64 * 40];
        threadgroup half tileV[32 * 72];
        encoder_flash_attention_register(q, k, v, o, p, group, tid, lane, sg, tileK, tileV);
    }
    """#

    private static let cooperativeFlashSource = #"""
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

    template<uint BR, uint BC, uint S>
    inline void flash_cooperative(
        device half* q, device half* k, device half* v, device half* o,
        constant SplitQKVParams& p, uint3 group, uint simd, uint lane, threadgroup float4* scratch
    ) {
        using namespace mpp::tensor_ops;
        constexpr uint HD = 64u;
        const uint row0 = group.x * BR * S + simd * BR;
        if (row0 >= p.T) return;
        const uint rows = min(BR, p.T - row0);
        const uint base = (group.z * p.nHeads + group.y) * p.T * HD;
        using Shape = extents<int32_t, HD, dynamic_extent>;
        using Head = tensor<device half, Shape, tensor_inline>;
        const auto query = Head(q + base + row0 * HD, Shape(rows));
        constexpr auto scoreDesc = matmul2d_descriptor(BR, BC, HD, false, true, false);
        constexpr auto outDesc = matmul2d_descriptor(BR, HD, BC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<scoreDesc, execution_simdgroup> scoreOp;
        matmul2d<outDesc, execution_simdgroup> outOp;
        auto score = scoreOp.template get_destination_cooperative_tensor<Head, Head, float>();
        using Probability = decltype(outOp.template get_left_input_cooperative_tensor<half, half, float>());
        auto output = outOp.template get_destination_cooperative_tensor<Probability, Head, float>();
        auto reduced = scoreOp.template get_row_reduction_destination_cooperative_tensor<Head, Head, float>();
        threadgroup float4* stat = scratch + simd * BR;
        for (uint r = lane; r < BR; r += 32) stat[r] = float4(-INFINITY, 0.0f, -INFINITY, 1.0f);
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < output.get_capacity(); ++i) {
            if (output.is_valid_element(i)) output[i] = 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint start = 0; start < p.T; start += BC) {
            const uint cols = min(BC, p.T - start);
            const auto key = Head(k + base + start * HD, Shape(cols));
            const auto val = Head(v + base + start * HD, Shape(cols));
            scoreOp.run(query, key, score);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < score.get_capacity(); ++i) {
                if (score.is_valid_element(i)) {
                    const auto xy = score.get_multidimensional_index(i);
                    score[i] = uint(xy[0]) < cols ? score[i] * 0.125f : -INFINITY;
                }
            }
            reduce_rows(score, reduced, reduction_operation::max, -INFINITY);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < reduced.get_capacity(); ++i) {
                if (reduced.is_valid_element(i)) {
                    const auto r = reduced.get_multidimensional_index(i)[0];
                    const float4 old = stat[r];
                    const float next = max(old.x, reduced[i]);
                    stat[r] = float4(next, old.y, next, exp(old.x - next));
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < score.get_capacity(); ++i) {
                if (score.is_valid_element(i)) {
                    const auto xy = score.get_multidimensional_index(i);
                    score[i] = exp(score[i] - stat[xy[1]].x);
                }
            }
            reduce_rows(score, reduced);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < reduced.get_capacity(); ++i) {
                if (reduced.is_valid_element(i)) {
                    const auto r = reduced.get_multidimensional_index(i)[0];
                    stat[r].y = stat[r].y * stat[r].w + reduced[i];
                }
            }
            auto rounded = scoreOp.template get_destination_cooperative_tensor<Head, Head, half>();
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < score.get_capacity(); ++i) {
                if (score.is_valid_element(i)) rounded[i] = half(score[i]);
            }
            auto probability = outOp.template get_left_input_cooperative_tensor<half, half, float>(rounded);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < output.get_capacity(); ++i) {
                if (output.is_valid_element(i)) {
                    const auto xy = output.get_multidimensional_index(i);
                    output[i] *= stat[xy[1]].w;
                }
            }
            outOp.run(probability, val, output);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < output.get_capacity(); ++i) {
            if (output.is_valid_element(i)) {
                const auto xy = output.get_multidimensional_index(i);
                const uint r = xy[1];
                if (r < rows) {
                    const uint outRow = group.z * p.T + row0 + r;
                    o[outRow * p.nHeads * HD + group.y * HD + xy[0]] = half(output[i] * (1.0f / stat[r].y));
                }
            }
        }
    }

    kernel void encoder_flash_attention_cooperative_f16(
        device half* q [[buffer(0)]], device half* k [[buffer(1)]], device half* v [[buffer(2)]],
        device half* out [[buffer(3)]], constant SplitQKVParams& p [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float4 work[64];
        flash_cooperative<32, 32, 2>(q, k, v, out, p, group, simd, lane, work);
    }

    kernel void encoder_flash_attention_final_tile_f16(
        device const half* q [[buffer(0)]],
        device const half* k [[buffer(1)]],
        device const half* v [[buffer(2)]],
        device half* o [[buffer(3)]],
        constant SplitQKVParams& p [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        group.x += ((p.T - 1) / 64) * 2;
        threadgroup half tileK[64 * 40];
        threadgroup half tileV[32 * 72];
        encoder_flash_attention_register(q, k, v, o, p, group, tid, lane, sg, tileK, tileV);
    }
    """#

    // Each pair of 16-lane subgroups consumes two adjacent frames. A lane loads four
    // K and four V codes together, with one pair of half scales per head and frame.
    private static let packedQ8Source = #"""
    kernel void split_cross_kv_q8_packed(
        device const half* input [[buffer(0)]], device const float* bias [[buffer(1)]],
        device char4* packedKV [[buffer(2)]], constant SplitCrossKVParams& p [[buffer(4)]],
        device half2* scales [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        const uint frame = group.x * 8 + sg;
        if (frame >= p.T) return;
        const uint width = p.nHeads * 64;
        const uint feature = group.y * 64 + lane;
        const uint row = (group.z * p.T + frame) * 2 * width + feature;
        const float2 k = float2(half2(
            half(float(input[row]) + bias[feature]),
            half(float(input[row + 32]) + bias[feature + 32])
        ));
        const float2 v = float2(half2(
            half(float(input[row + width]) + bias[feature + width]),
            half(float(input[row + width + 32]) + bias[feature + width + 32])
        ));
        const float kMax = simd_max(max(abs(k.x), abs(k.y)));
        const float vMax = simd_max(max(abs(v.x), abs(v.y)));
        const float kInv = kMax > 0 ? 127.0f / kMax : 0.0f;
        const float vInv = vMax > 0 ? 127.0f / vMax : 0.0f;
        const char2 qk = char2(clamp(rint(k * kInv), float2(-127), float2(127)));
        const char2 qv = char2(clamp(rint(v * vInv), float2(-127), float2(127)));
        const uint offset = p.layerOffset + ((p.slots[group.z] * p.nHeads + group.y) * p.T + frame) * 64;
        packedKV[offset / 2 + (lane & 15) * 2 + lane / 16] = char4(qk, qv);
        if (lane == 0) scales[offset / 64] = half2(half(kMax / 127.0f), half(vMax / 127.0f));
    }

    template<uint Beams>
    inline void cross_q8_half_simd(
        device const half* q, device const uint2* kv, device const half2* scales,
        PackCrossGroup pack, device half* output, constant CrossAttnFusedParams& p,
        device const float* bias, uint head, uint lane, uint sg,
        threadgroup float2* stats, threadgroup float* values
    ) {
        constexpr uint Partitions = 12;
        const uint part = sg * 2 + lane / 16;
        const uint d = lane % 16;
        const uint feature = head * 64 + d;
        const uint width = p.nHeads * 64;
        const float4 qb = float4(bias[feature], bias[feature + 16], bias[feature + 32], bias[feature + 48]);
        float4 Q[Beams], acc[Beams];
        float m[Beams], l[Beams];
        #pragma clang loop unroll(full)
        for (uint b = 0; b < Beams; ++b) {
            const uint offset = pack.rows[b] * width + feature;
            const float4 raw = float4(q[offset], q[offset + 16], q[offset + 32], q[offset + 48]);
            Q[b] = (raw + qb) * 0.125f;
            m[b] = -INFINITY;
            l[b] = 0.0f;
            acc[b] = float4(0.0f);
        }
        const uint headBase = p.layerOffset + (pack.packIdx * p.nHeads + head) * p.T * 64;
        for (uint t = part; t < p.T; t += Partitions) {
            const uint2 pair = kv[headBase / 4 + t * 16 + d];
            const float4 low = float4(as_type<char4>(pair.x));
            const float4 high = float4(as_type<char4>(pair.y));
            const float2 s = float2(scales[headBase / 64 + t]);
            const float4 key = float4(low.x, high.x, low.y, high.y);
            const float4 value = float4(low.z, high.z, low.w, high.w);
            #pragma clang loop unroll(full)
            for (uint b = 0; b < Beams; ++b) {
                const float2 pairDots = Q[b].xy * key.xy + Q[b].zw * key.zw;
                float dot = pairDots.x + pairDots.y;
                dot += simd_shuffle_xor(dot, ushort(1));
                dot += simd_shuffle_xor(dot, ushort(2));
                dot += simd_shuffle_xor(dot, ushort(4));
                dot += simd_shuffle_xor(dot, ushort(8));
                const float score = dot * s.x;
                const float2 w = cross_attention_weights(m[b], score);
                l[b] = l[b] * w.x + w.y;
                acc[b] = acc[b] * w.x + (w.y * s.y) * value;
                m[b] = max(m[b], score);
            }
        }
        #pragma clang loop unroll(full)
        for (uint b = 0; b < Beams; ++b) {
            if (d == 0) stats[part] = float2(m[b], l[b]);
            values[part * 64 + d] = acc[b].x;
            values[part * 64 + d + 16] = acc[b].y;
            values[part * 64 + d + 32] = acc[b].z;
            values[part * 64 + d + 48] = acc[b].w;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                float maxScore = -INFINITY;
                #pragma clang loop unroll(full)
                for (uint p = 0; p < Partitions; ++p) maxScore = max(maxScore, stats[p].x);
                float total = 0.0f;
                float2 sums = float2(0.0f);
                #pragma clang loop unroll(full)
                for (uint p = 0; p < Partitions; ++p) {
                    const float w = exp(stats[p].x - maxScore);
                    total += stats[p].y * w;
                    sums.x += values[p * 64 + lane] * w;
                    sums.y += values[p * 64 + lane + 32] * w;
                }
                const float inv = 1.0f / total;
                const uint offset = pack.rows[b] * width + head * 64 + lane;
                const float2 result = sums * inv;
                output[offset] = half(result.x);
                output[offset + 32] = half(result.y);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    kernel void decoder_cross_attn_q8_packed(
        device const half* Q [[buffer(0)]], device const uint2* cache [[buffer(1)]],
        device const PackCrossGroup* groups [[buffer(3)]], device half* output [[buffer(4)]],
        constant CrossAttnFusedParams& params [[buffer(5)]], device const float* bias [[buffer(6)]],
        device const half2* scales [[buffer(7)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float2 stats[12];
        threadgroup float values[12 * 64];
        const PackCrossGroup pack = groups[group.y];
        #define BEAM_CASE(B) case B: cross_q8_half_simd<B>(Q, cache, scales, pack, output, params, bias, group.x, lane, sg, stats, values); break;
        switch (pack.count) {
            BEAM_CASE(1)
            BEAM_CASE(2)
            BEAM_CASE(3)
            BEAM_CASE(4)
            BEAM_CASE(5)
        }
        #undef BEAM_CASE
    }
    """#

    private static let quantizedCrossKVSource = #"""
    constant bool crossKVQuantizeKeys [[function_constant(0)]];
    constant bool crossKVQuantizeValues [[function_constant(1)]];

    template <bool Quantized>
    inline void store_cross_kv(
        float2 values, device uchar* data, device half* scales, uint frame, uint lane
    ) {
        if constexpr (Quantized) {
            const float maximum = simd_max(max(abs(values.x), abs(values.y)));
            const float scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
            const float reciprocal = maximum > 0.0f ? 127.0f / maximum : 0.0f;
            const float2 q = clamp(rint(values * reciprocal), float2(-127.0f), float2(127.0f));
            // The two values read by one lane are adjacent within each 64-byte frame.
            ((device char2*)data)[frame / 2 + lane] = char2(q);
            if (lane == 0) scales[frame / 64] = half(scale);
        } else {
            device half* output = ((device half*)data) + frame + lane;
            output[0] = half(values.x);
            output[32] = half(values.y);
        }
    }

    template <bool QuantizedKeys, bool QuantizedValues>
    inline void project_cross_kv(
        device const half* input, device const float* bias, device uchar* keys, device uchar* values,
        device half* keyScales, device half* valueScales, constant SplitCrossKVParams& p,
        uint frame, uint head, uint pack, uint lane
    ) {
        const uint width = p.nHeads * 64;
        const uint feature = head * 64 + lane;
        const uint row = (pack * p.T + frame) * 2 * width + feature;
        const float2 k = float2(half2(
            half(float(input[row]) + bias[feature]),
            half(float(input[row + 32]) + bias[feature + 32])
        ));
        const float2 v = float2(half2(
            half(float(input[row + width]) + bias[feature + width]),
            half(float(input[row + width + 32]) + bias[feature + width + 32])
        ));
        const uint offset = p.layerOffset + ((p.slots[pack] * p.nHeads + head) * p.T + frame) * 64;
        store_cross_kv<QuantizedKeys>(k, keys, keyScales, offset, lane);
        store_cross_kv<QuantizedValues>(v, values, valueScales, offset, lane);
    }

    kernel void split_cross_kv_q8(
        device const half* input [[buffer(0)]], device const float* bias [[buffer(1)]],
        device uchar* keys [[buffer(2)]], device uchar* values [[buffer(3)]],
        constant SplitCrossKVParams& p [[buffer(4)]],
        device half* keyScales [[buffer(5)]], device half* valueScales [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        const uint frame = group.x * 8 + sg;
        if (frame >= p.T) return;
        if (crossKVQuantizeKeys && crossKVQuantizeValues) {
            project_cross_kv<true, true>(input, bias, keys, values, keyScales, valueScales, p, frame, group.y, group.z, lane);
        } else if (crossKVQuantizeKeys) {
            project_cross_kv<true, false>(input, bias, keys, values, keyScales, valueScales, p, frame, group.y, group.z, lane);
        } else {
            project_cross_kv<false, true>(input, bias, keys, values, keyScales, valueScales, p, frame, group.y, group.z, lane);
        }
    }

    kernel void decoder_cross_attn_pack_fused_q8(
        device const half* Q [[buffer(0)]],
        device const uchar* keys [[buffer(1)]], device const uchar* values [[buffer(2)]],
        device const PackCrossGroup* groups [[buffer(3)]], device half* output [[buffer(4)]],
        constant CrossAttnFusedParams& params [[buffer(5)]], device const float* bias [[buffer(6)]],
        device const half* keyScales [[buffer(7)]], device const half* valueScales [[buffer(8)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        const PackCrossGroup pack = groups[group.y];
        threadgroup float2 partialStats[8];
        threadgroup float partialValues[8 * 64];
        if (crossKVQuantizeKeys && crossKVQuantizeValues) {
            dispatch_cross_attention<true, true>(
                Q, keys, values, keyScales, valueScales, pack, output, params, bias,
                group.x, lane, sg, partialStats, partialValues
            );
        } else if (crossKVQuantizeKeys) {
            dispatch_cross_attention<true, false>(
                Q, keys, values, keyScales, valueScales, pack, output, params, bias,
                group.x, lane, sg, partialStats, partialValues
            );
        } else {
            dispatch_cross_attention<false, true>(
                Q, keys, values, keyScales, valueScales, pack, output, params, bias,
                group.x, lane, sg, partialStats, partialValues
            );
        }
    }
    """#

    private static let metalSource: String = #"""
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    \#(erfTableSource)

    inline float whisper_erf(float x) {
        const float a = abs(x);
        if (a >= 4.0f) return copysign(1.0f, x);
        if (isnan(a)) return x;

        const uint index = uint(rint(a * 128.0f));
        const float center = float(index) * (1.0f / 128.0f);
        const float d = a - center;
        const float cubic = fma(2.0f * center, center, -1.0f) * (1.0f / 3.0f);
        const float delta = d * fma(d, fma(d, cubic, -center), 1.0f);
        const float3 row = erf_table[index];
        return copysign(row.x + fma(row.z, delta, row.y), x);
    }

    inline float gelu_erf(float x) {
        return 0.5f * x * (1.0f + whisper_erf(x * 0.70710678118654752440f));
    }

    // =========================================================================
    // 1. Encoder Stem: im2col for Conv1D (k=3, pad=1, stride=1 or 2)
    // =========================================================================
    struct Im2ColParams {
        uint P;
        uint T_in;
        uint T_out;
        uint C_in;
        uint stride;
    };

    kernel void im2col_conv1d_f32_to_f16(
        device const float* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant Im2ColParams& p [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint col = gid.x;
        uint t_out = gid.y;
        uint pack = gid.z;
        uint outCols = 3 * p.C_in;
        if (col >= outCols || t_out >= p.T_out || pack >= p.P) return;

        uint k = col / p.C_in;
        uint c = col - k * p.C_in;
        int t_in = int(t_out * p.stride + k) - 1;
        float val = 0.0f;
        if (t_in >= 0 && t_in < int(p.T_in)) {
            val = src[(pack * p.T_in + uint(t_in)) * p.C_in + c];
        }
        dst[(pack * p.T_out + t_out) * outCols + col] = half(val);
    }

    kernel void im2col_conv1d_f16_to_f16(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant Im2ColParams& p [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint col = gid.x;
        uint t_out = gid.y;
        uint pack = gid.z;
        uint outCols = 3 * p.C_in;
        if (col >= outCols || t_out >= p.T_out || pack >= p.P) return;

        uint k = col / p.C_in;
        uint c = col - k * p.C_in;
        int t_in = int(t_out * p.stride + k) - 1;
        half val = half(0.0h);
        if (t_in >= 0 && t_in < int(p.T_in)) {
            val = src[(pack * p.T_in + uint(t_in)) * p.C_in + c];
        }
        dst[(pack * p.T_out + t_out) * outCols + col] = val;
    }

    // =========================================================================
    // 2. Shared Elementwise & Fused LayerNorm Kernels
    // =========================================================================
    struct BiasGeluPosParams {
        uint rows;
        uint T;
        uint D;
        uint usePos;
    };

    kernel void add_bias_gelu_pos_f16(
        device half* x [[buffer(0)]],
        device const float* bias [[buffer(1)]],
        device const float* posEmb [[buffer(2)]],
        constant BiasGeluPosParams& p [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint row = gid.y;
        if (d >= p.D || row >= p.rows) return;
        uint idx = row * p.D + d;
        float v = gelu_erf(float(x[idx]) + bias[d]);
        if (p.usePos != 0) {
            uint t = row % p.T;
            v += posEmb[t * p.D + d];
        }
        x[idx] = half(v);
    }

    struct BiasGeluParams {
        uint rows;
        uint D;
    };

    kernel void add_bias_f16(
        device half* x [[buffer(0)]],
        device const float* bias [[buffer(1)]],
        constant BiasGeluParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint row = gid.y;
        if (d >= p.D || row >= p.rows) return;
        uint idx = row * p.D + d;
        x[idx] = half(float(x[idx]) + bias[d]);
    }

    kernel void add_bias_gelu_f16(
        device half* x [[buffer(0)]],
        device const float* bias [[buffer(1)]],
        constant BiasGeluParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint row = gid.y;
        if (d >= p.D || row >= p.rows) return;
        uint idx = row * p.D + d;
        x[idx] = half(gelu_erf(float(x[idx]) + bias[d]));
    }

    kernel void add_bias_residual_f16(
        device half* residual [[buffer(0)]],
        device const half* y [[buffer(1)]],
        device const float* bias [[buffer(2)]],
        constant BiasGeluParams& p [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint row = gid.y;
        if (d >= p.D || row >= p.rows) return;
        uint idx = row * p.D + d;
        residual[idx] = half(float(residual[idx]) + float(y[idx]) + bias[d]);
    }

    kernel void layernorm_f16(
        device const half* x [[buffer(0)]],
        device half* out [[buffer(1)]],
        device const float* weight [[buffer(2)]],
        device const float* bias [[buffer(3)]],
        constant uint& D [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        device const half* rowIn = x + row * D;
        device half* rowOut = out + row * D;

        float sum = 0.0f;
        float sumSq = 0.0f;
        for (uint i = tid; i < D; i += tgs) {
            float v = float(rowIn[i]);
            sum += v;
            sumSq += v * v;
        }
        sum = simd_sum(sum);
        sumSq = simd_sum(sumSq);

        threadgroup float sMean[8];
        threadgroup float sVar[8];
        if (lane == 0) {
            sMean[sg] = sum;
            sVar[sg] = sumSq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            float totalSum = (lane < 8) ? sMean[lane] : 0.0f;
            float totalSq = (lane < 8) ? sVar[lane] : 0.0f;
            totalSum = simd_sum(totalSum);
            totalSq = simd_sum(totalSq);
            if (lane == 0) {
                float m = totalSum / float(D);
                float v = max(0.0f, totalSq / float(D) - m * m);
                sMean[0] = m;
                sVar[0] = rsqrt(v + 1e-5f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = sMean[0];
        float invStd = sVar[0];

        for (uint i = tid; i < D; i += tgs) {
            float norm = (float(rowIn[i]) - mean) * invStd;
            rowOut[i] = half(norm * weight[i] + bias[i]);
        }
    }

    // Fused: residual[row] += y[row] + addBias; normOut[row] = LN(residual[row])
    kernel void add_bias_residual_layernorm_f16(
        device half* residual [[buffer(0)]],
        device const half* y [[buffer(1)]],
        device const float* addBias [[buffer(2)]],
        device half* normOut [[buffer(3)]],
        device const float* lnW [[buffer(4)]],
        device const float* lnB [[buffer(5)]],
        constant uint& D [[buffer(6)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgs [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        device half* rowRes = residual + row * D;
        device const half* rowY = y + row * D;
        device half* rowNorm = normOut + row * D;

        float sum = 0.0f;
        float sumSq = 0.0f;
        for (uint i = tid; i < D; i += tgs) {
            float v = float(rowRes[i]) + float(rowY[i]) + addBias[i];
            rowRes[i] = half(v);
            sum += v;
            sumSq += v * v;
        }
        sum = simd_sum(sum);
        sumSq = simd_sum(sumSq);

        threadgroup float sMean[8];
        threadgroup float sVar[8];
        if (lane == 0) {
            sMean[sg] = sum;
            sVar[sg] = sumSq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            float totalSum = (lane < 8) ? sMean[lane] : 0.0f;
            float totalSq = (lane < 8) ? sVar[lane] : 0.0f;
            totalSum = simd_sum(totalSum);
            totalSq = simd_sum(totalSq);
            if (lane == 0) {
                float m = totalSum / float(D);
                float v = max(0.0f, totalSq / float(D) - m * m);
                sMean[0] = m;
                sVar[0] = rsqrt(v + 1e-5f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = sMean[0];
        float invStd = sVar[0];

        for (uint i = tid; i < D; i += tgs) {
            float norm = (float(rowRes[i]) - mean) * invStd;
            rowNorm[i] = half(norm * lnW[i] + lnB[i]);
        }
    }

    // =========================================================================
    // 3. Encoder QKV Split & 16 KB High-Occupancy FlashAttention-2 (MMA)
    // =========================================================================
    struct SplitQKVParams {
        uint P;
        uint T;
        uint nHeads;
        uint headDim;
    };

    kernel void add_bias_split_qkv_f16(
        device const half* qkvIn [[buffer(0)]],
        device const float* qkvBias [[buffer(1)]],
        device half* qOut [[buffer(2)]],
        device half* kOut [[buffer(3)]],
        device half* vOut [[buffer(4)]],
        constant SplitQKVParams& p [[buffer(5)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint t = gid.y;
        uint pack = gid.z;
        uint D = p.nHeads * p.headDim;
        if (d >= D || t >= p.T || pack >= p.P) return;

        uint h = d / p.headDim;
        uint hd = d - h * p.headDim;
        uint inBase = (pack * p.T + t) * (3 * D);
        uint outIdx = ((pack * p.nHeads + h) * p.T + t) * p.headDim + hd;

        qOut[outIdx] = half(float(qkvIn[inBase + d]) + qkvBias[d]);
        kOut[outIdx] = half(float(qkvIn[inBase + D + d]) + qkvBias[D + d]);
        vOut[outIdx] = half(float(qkvIn[inBase + 2 * D + d]) + qkvBias[2 * D + d]);
    }

    // Keep each SIMD group's FP32 output fragments live across the key tiles.
    // Use the legacy 32-key softmax updates and F16 probability boundary.
    kernel void encoder_flash_attention_f16(
        device const half* q [[buffer(0)]],
        device const half* k [[buffer(1)]],
        device const half* v [[buffer(2)]],
        device half* o [[buffer(3)]],
        constant SplitQKVParams& p [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]
    ) {
        constexpr uint headDim = 64;
        constexpr uint tile = 32;
        const uint firstRow = group.x * tile + simd * 8;
        const uint headOffset = (group.z * p.nHeads + group.y) * p.T * headDim;

        threadgroup half tileK[tile * headDim];
        threadgroup half tileV[tile * headDim];
        threadgroup float scoreBuffer[tile * tile];
        threadgroup float scaleBuffer[tile * 8];
        threadgroup float* scores = scoreBuffer + simd * 8 * tile;
        threadgroup half* probabilities = (threadgroup half*)scores;
        threadgroup float* rowScales = scaleBuffer + simd * 8 * 8;

        simdgroup_half8x8 query[8];
        if (firstRow + 8 <= p.T) {
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                simdgroup_load(query[d], q + headOffset + firstRow * headDim + d * 8, headDim);
            }
        } else {
            for (uint index = lane * 4; index < 8 * headDim; index += 32 * 4) {
                const uint token = firstRow + index / headDim;
                *(threadgroup half4*)(probabilities + index) = token < p.T
                    ? *(device const half4*)(q + headOffset + firstRow * headDim + index) : half4(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                simdgroup_load(query[d], probabilities + d * 8, headDim);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_float8x8 accumulators[8];
        float maxByRow[8], sumByRow[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            accumulators[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            maxByRow[i] = -INFINITY;
            sumByRow[i] = 0.0f;
        }

        for (uint block = 0; block < p.T; block += tile) {
            for (uint index = tid * 4; index < tile * headDim; index += 128 * 4) {
                const uint source = headOffset + block * headDim + index;
                const bool valid = block + index / headDim < p.T;
                *(threadgroup half4*)(tileK + index) = valid ? *(device const half4*)(k + source) : half4(0.0h);
                *(threadgroup half4*)(tileV + index) = valid ? *(device const half4*)(v + source) : half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 dot[4];
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                dot[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                #pragma clang loop unroll(full)
                for (uint n = 0; n < 4; ++n) {
                    simdgroup_half8x8 key;
                    simdgroup_load(key, tileK + n * 8 * headDim + d * 8, headDim, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(dot[n], query[d], key, dot[n]);
                }
            }
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                simdgroup_store(dot[n], scores + n * 8, tile);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const bool valid = block + lane < p.T;
            half rowProbs[8];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                const float score = valid ? scores[r * tile + lane] * 0.125f : -INFINITY;
                const float nextMax = max(maxByRow[r], simd_max(score));
                const float alpha = exp(maxByRow[r] - nextMax);
                const float prob = valid ? exp(score - nextMax) : 0.0f;
                rowProbs[r] = half(prob);
                if (lane < 8) rowScales[r * 8 + lane] = sumByRow[r] > 0.0f ? alpha : 1.0f;
                maxByRow[r] = nextMax;
                sumByRow[r] = sumByRow[r] * alpha + simd_sum(prob);
            }

            // Each SIMD group can reuse its scores after all its lanes have read them.
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                probabilities[r * tile + lane] = rowProbs[r];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 pMat[4];
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 4; ++d) {
                simdgroup_load(pMat[d], probabilities + d * 8, tile);
            }
            simdgroup_float8x8 scaleFragment;
            simdgroup_load(scaleFragment, rowScales, 8);

            #pragma clang loop unroll(full)
            for (uint n = 0; n < 8; ++n) {
                accumulators[n].thread_elements() *= scaleFragment.thread_elements();
                #pragma clang loop unroll(full)
                for (uint d = 0; d < 4; ++d) {
                    simdgroup_half8x8 value;
                    simdgroup_load(value, tileV + d * 8 * headDim + n * 8, headDim);
                    simdgroup_multiply_accumulate(accumulators[n], pMat[d], value, accumulators[n]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint width = p.nHeads * headDim;
        const uint outBase = (group.z * p.T + firstRow) * width + group.y * headDim;
        #pragma clang loop unroll(full)
        for (uint halfTile = 0; halfTile < 2; ++halfTile) {
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                simdgroup_store(accumulators[halfTile * 4 + n], scores + n * 8, tile);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 8; ++r) {
                if (firstRow + r < p.T) {
                    const float inv = 1.0f / sumByRow[r];
                    o[outBase + r * width + halfTile * tile + lane] = half(scores[r * tile + lane] * inv);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    struct SplitCrossKVParams {
        uint P;
        uint T;
        uint nHeads;
        uint headDim;
        uint layerOffset;
        uint slots[4];
    };

    kernel void split_cross_kv_f16(
        device const half* kvIn [[buffer(0)]],
        device const float* kvBias [[buffer(1)]],
        device half* crossK [[buffer(2)]],
        device half* crossV [[buffer(3)]],
        constant SplitCrossKVParams& p [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint t = gid.y;
        uint pack = gid.z;
        uint D = p.nHeads * p.headDim;
        if (d >= D || t >= p.T || pack >= p.P) return;

        uint h = d / p.headDim;
        uint hd = d - h * p.headDim;
        uint inBase = (pack * p.T + t) * (2 * D);
        uint slot = p.slots[pack];
        uint outIdx = p.layerOffset + ((slot * p.nHeads + h) * p.T + t) * p.headDim + hd;

        crossK[outIdx] = half(float(kvIn[inBase + d]) + kvBias[d]);
        crossV[outIdx] = half(float(kvIn[inBase + D + d]) + kvBias[D + d]);
    }

    // =========================================================================
    // 4. Decoder Embedding & Zero-Copy Beam-Tree / Prefill Self-Attention
    // =========================================================================
    struct EmbedParams {
        uint B;
        uint D;
    };

    kernel void decoder_embed_lookup_f16(
        device const uint* tokenIDs [[buffer(0)]],
        device const uint* positions [[buffer(1)]],
        device const half* tokenEmb [[buffer(2)]],
        device const float* posEmb [[buffer(3)]],
        device half* out [[buffer(4)]],
        constant EmbedParams& p [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint d = gid.x;
        uint b = gid.y;
        if (d >= p.D || b >= p.B) return;
        uint tok = tokenIDs[b];
        uint pos = positions[b];
        out[b * p.D + d] = half(float(tokenEmb[tok * p.D + d]) + posEmb[pos * p.D + d]);
    }

    struct SelfAttnTreeParams {
        uint B;
        uint nHeads;
        uint headDim;
        uint maxTextCtx;
        uint layerOffset;
    };

    kernel void decoder_self_attn_beam_tree_f16(
        device const half* qkv [[buffer(0)]],
        device half* selfK [[buffer(1)]],
        device half* selfV [[buffer(2)]],
        device const uint* writeSlots [[buffer(3)]],
        device const uint* seqLens [[buffer(4)]],
        device const uint* slotTable [[buffer(5)]],
        device half* attnOut [[buffer(6)]],
        constant SelfAttnTreeParams& p [[buffer(7)]],
        device const float* qkvBias [[buffer(8)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        const uint head = tgid.x;
        const uint b = tgid.y;
        const uint D = p.nHeads * 64;
        if (head >= p.nHeads || b >= p.B) return;

        const uint qkvBase = b * (3 * D);
        const uint hOff = head * 64;
        const float scale = 0.125f;

        float q0 = (float(qkv[qkvBase + hOff + lane]) + qkvBias[hOff + lane]) * scale;
        float q1 = (float(qkv[qkvBase + hOff + lane + 32]) + qkvBias[hOff + lane + 32]) * scale;

        half kCur0 = qkv[qkvBase + D + hOff + lane]; // K has no bias in Whisper
        half kCur1 = qkv[qkvBase + D + hOff + lane + 32];
        half vCur0 = half(float(qkv[qkvBase + 2 * D + hOff + lane]) + qkvBias[2 * D + hOff + lane]);
        half vCur1 = half(float(qkv[qkvBase + 2 * D + hOff + lane + 32]) + qkvBias[2 * D + hOff + lane + 32]);

        const uint wSlot = writeSlots[b];
        const uint wBase = p.layerOffset + wSlot * D + hOff;
        selfK[wBase + lane] = kCur0;
        selfK[wBase + lane + 32] = kCur1;
        selfV[wBase + lane] = vCur0;
        selfV[wBase + lane + 32] = vCur1;

        const uint len = seqLens[b];
        device const uint* slots = slotTable + b * p.maxTextCtx;

        float mPrev = -INFINITY;
        float lSum = 0.0f;
        float o0 = 0.0f;
        float o1 = 0.0f;

        for (uint t = 0; t + 1 < len; ++t) {
            uint slot = slots[t];
            uint kvBase = p.layerOffset + slot * D + hOff;
            float k0 = float(selfK[kvBase + lane]);
            float k1 = float(selfK[kvBase + lane + 32]);
            float score = simd_sum(q0 * k0 + q1 * k1);

            float mNew = max(mPrev, score);
            float alpha = exp(mPrev - mNew);
            float prob = exp(score - mNew);
            lSum = lSum * alpha + prob;

            float v0 = float(selfV[kvBase + lane]);
            float v1 = float(selfV[kvBase + lane + 32]);
            o0 = o0 * alpha + prob * v0;
            o1 = o1 * alpha + prob * v1;
            mPrev = mNew;
        }

        {
            float score = simd_sum(q0 * float(kCur0) + q1 * float(kCur1));
            float mNew = max(mPrev, score);
            float alpha = exp(mPrev - mNew);
            float prob = exp(score - mNew);
            lSum = lSum * alpha + prob;
            o0 = o0 * alpha + prob * float(vCur0);
            o1 = o1 * alpha + prob * float(vCur1);
        }

        float invL = 1.0f / lSum;
        uint outBase = b * D + hOff;
        attnOut[outBase + lane] = half(o0 * invL);
        attnOut[outBase + lane + 32] = half(o1 * invL);
    }

    struct SelfAttnPrefillParams {
        uint totalTokens;
        uint nHeads;
        uint headDim;
        uint layerOffset;
    };

    kernel void decoder_self_attn_prefill_f16(
        device const half* qkv [[buffer(0)]],
        device half* selfK [[buffer(1)]],
        device half* selfV [[buffer(2)]],
        device const uint* writeSlots [[buffer(3)]],
        device const uint* seqStarts [[buffer(4)]],
        device const uint* posInSeq [[buffer(5)]],
        device half* attnOut [[buffer(6)]],
        constant SelfAttnPrefillParams& p [[buffer(7)]],
        device const float* qkvBias [[buffer(8)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        const uint head = tgid.x;
        const uint b = tgid.y;
        const uint D = p.nHeads * 64;
        if (head >= p.nHeads || b >= p.totalTokens) return;

        const uint hOff = head * 64;
        const uint qkvBase = b * (3 * D);
        const float scale = 0.125f;
        const float qb0 = qkvBias[hOff + lane];
        const float qb1 = qkvBias[hOff + lane + 32];
        const float vb0 = qkvBias[2 * D + hOff + lane];
        const float vb1 = qkvBias[2 * D + hOff + lane + 32];

        const uint wSlot = writeSlots[b];
        const uint wBase = p.layerOffset + wSlot * D + hOff;
        selfK[wBase + lane] = qkv[qkvBase + D + hOff + lane];
        selfK[wBase + lane + 32] = qkv[qkvBase + D + hOff + lane + 32];
        selfV[wBase + lane] = half(float(qkv[qkvBase + 2 * D + hOff + lane]) + vb0);
        selfV[wBase + lane + 32] = half(float(qkv[qkvBase + 2 * D + hOff + lane + 32]) + vb1);

        float q0 = (float(qkv[qkvBase + hOff + lane]) + qb0) * scale;
        float q1 = (float(qkv[qkvBase + hOff + lane + 32]) + qb1) * scale;

        const uint startIdx = seqStarts[b];
        const uint pos = posInSeq[b];

        float mPrev = -INFINITY;
        float lSum = 0.0f;
        float o0 = 0.0f;
        float o1 = 0.0f;

        for (uint t = 0; t <= pos; ++t) {
            uint srcBase = (startIdx + t) * (3 * D);
            float k0 = float(qkv[srcBase + D + hOff + lane]);
            float k1 = float(qkv[srcBase + D + hOff + lane + 32]);
            float score = simd_sum(q0 * k0 + q1 * k1);

            float mNew = max(mPrev, score);
            float alpha = exp(mPrev - mNew);
            float prob = exp(score - mNew);
            lSum = lSum * alpha + prob;

            float v0 = float(qkv[srcBase + 2 * D + hOff + lane]) + vb0;
            float v1 = float(qkv[srcBase + 2 * D + hOff + lane + 32]) + vb1;
            o0 = o0 * alpha + prob * v0;
            o1 = o1 * alpha + prob * v1;
            mPrev = mNew;
        }

        float invL = 1.0f / lSum;
        uint outBase = b * D + hOff;
        attnOut[outBase + lane] = half(o0 * invL);
        attnOut[outBase + lane + 32] = half(o1 * invL);
    }

    // =========================================================================
    // 5. Pack-Fused 5-Beam Cross-Attention (5x LESS Cross-KV DRAM Traffic!)
    //    1 threadgroup (256 threads = 8 simdgroups) processes up to 5 beams
    //    of the SAME pack simultaneously, reading crossK/crossV ONCE!
    // =========================================================================
    struct PackCrossGroup {
        uint packIdx;
        uint count;      // 1..5
        uint rows[5];    // batch row indices in Q / attnOut
        uint pad;
    };

    struct CrossAttnFusedParams {
        uint numGroups;
        uint P;
        uint T;
        uint nHeads;
        uint headDim;
        uint layerOffset;
    };

    inline float2 cross_attention_weights(float previousMax, float score) {
        // The term associated with the new maximum is exp(0) = 1.
        const float attenuation = exp(-abs(score - previousMax));
        return score > previousMax ? float2(attenuation, 1.0f) : float2(1.0f, attenuation);
    }

    template <uint BeamCount, bool QuantizedKeys, bool QuantizedValues>
    inline void cross_attention_for_beams(
        device const half* Q, device const uchar* crossK, device const uchar* crossV,
        device const half* keyScales, device const half* valueScales,
        PackCrossGroup grp, device half* attnOut, constant CrossAttnFusedParams& p,
        device const float* bias, uint head, uint lane, uint sg,
        threadgroup float2* partialStats, threadgroup float* partialValues
    ) {
        const uint D = p.nHeads * 64;
        const uint feature = head * 64;
        const float2 qb = float2(bias[feature + lane], bias[feature + lane + 32]);
        float2 query[BeamCount], acc[BeamCount];
        float rowMax[BeamCount], rowSum[BeamCount];
        #pragma clang loop unroll(full)
        for (uint b = 0; b < BeamCount; ++b) {
            const uint base = grp.rows[b] * D + feature;
            query[b] = (float2(float(Q[base + lane]), float(Q[base + lane + 32])) + qb) * 0.125f;
            rowMax[b] = -INFINITY;
            rowSum[b] = 0.0f;
            acc[b] = float2(0.0f);
        }
        const uint headBase = p.layerOffset + (grp.packIdx * p.nHeads + head) * p.T * 64;
        for (uint t = sg; t < p.T; t += 8) {
            const uint frame = headBase + t * 64;
            float2 k, v;
            float keyScale = 1.0f;
            float valueScale = 1.0f;
            if constexpr (QuantizedKeys) {
                k = float2(((device const char2*)crossK)[frame / 2 + lane]);
                keyScale = float(keyScales[frame / 64]);
            } else {
                const device half* keys = ((device const half*)crossK) + frame + lane;
                k = float2(float(keys[0]), float(keys[32]));
            }
            if constexpr (QuantizedValues) {
                v = float2(((device const char2*)crossV)[frame / 2 + lane]);
                valueScale = float(valueScales[frame / 64]);
            } else {
                const device half* values = ((device const half*)crossV) + frame + lane;
                v = float2(float(values[0]), float(values[32]));
            }
            #pragma clang loop unroll(full)
            for (uint b = 0; b < BeamCount; ++b) {
                float score = simd_sum(query[b].x * k.x + query[b].y * k.y);
                if constexpr (QuantizedKeys) score *= keyScale;
                const float2 w = cross_attention_weights(rowMax[b], score);
                rowSum[b] = rowSum[b] * w.x + w.y;
                float valueWeight = w.y;
                if constexpr (QuantizedValues) valueWeight *= valueScale;
                acc[b].x = acc[b].x * w.x + valueWeight * v.x;
                acc[b].y = acc[b].y * w.x + valueWeight * v.y;
                rowMax[b] = max(rowMax[b], score);
            }
        }

        // Merge the eight SIMD groups in their original order for each beam.
        #pragma clang loop unroll(full)
        for (uint b = 0; b < BeamCount; ++b) {
            if (lane == 0) partialStats[sg] = float2(rowMax[b], rowSum[b]);
            partialValues[sg * 64 + lane] = acc[b].x;
            partialValues[sg * 64 + lane + 32] = acc[b].y;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0) {
                float maximum = partialStats[0].x;
                #pragma clang loop unroll(full)
                for (uint s = 1; s < 8; ++s) maximum = max(maximum, partialStats[s].x);
                float denominator = 0.0f;
                float2 numerator = float2(0.0f);
                #pragma clang loop unroll(full)
                for (uint s = 0; s < 8; ++s) {
                    const float scale = exp(partialStats[s].x - maximum);
                    denominator += partialStats[s].y * scale;
                    numerator.x += partialValues[s * 64 + lane] * scale;
                    numerator.y += partialValues[s * 64 + lane + 32] * scale;
                }
                const float inv = 1.0f / denominator;
                const uint out = grp.rows[b] * D + feature + lane;
                attnOut[out] = half(numerator.x * inv);
                attnOut[out + 32] = half(numerator.y * inv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    template <bool QuantizedKeys, bool QuantizedValues>
    inline void dispatch_cross_attention(
        device const half* Q, device const uchar* crossK, device const uchar* crossV,
        device const half* keyScales, device const half* valueScales, PackCrossGroup pack,
        device half* output, constant CrossAttnFusedParams& params, device const float* bias,
        uint head, uint lane, uint sg, threadgroup float2* partialStats, threadgroup float* partialValues
    ) {
        // Specialize once per threadgroup, outside the loop over audio frames.
        switch (pack.count) {
            case 1:
                cross_attention_for_beams<1, QuantizedKeys, QuantizedValues>(
                    Q, crossK, crossV, keyScales, valueScales, pack, output, params, bias, head, lane, sg,
                    partialStats, partialValues
                );
                break;
            case 2:
                cross_attention_for_beams<2, QuantizedKeys, QuantizedValues>(
                    Q, crossK, crossV, keyScales, valueScales, pack, output, params, bias, head, lane, sg,
                    partialStats, partialValues
                );
                break;
            case 3:
                cross_attention_for_beams<3, QuantizedKeys, QuantizedValues>(
                    Q, crossK, crossV, keyScales, valueScales, pack, output, params, bias, head, lane, sg,
                    partialStats, partialValues
                );
                break;
            case 4:
                cross_attention_for_beams<4, QuantizedKeys, QuantizedValues>(
                    Q, crossK, crossV, keyScales, valueScales, pack, output, params, bias, head, lane, sg,
                    partialStats, partialValues
                );
                break;
            case 5:
                cross_attention_for_beams<5, QuantizedKeys, QuantizedValues>(
                    Q, crossK, crossV, keyScales, valueScales, pack, output, params, bias, head, lane, sg,
                    partialStats, partialValues
                );
                break;
        }
    }

    kernel void decoder_cross_attn_pack_fused_f16(
        device const half* Q [[buffer(0)]],
        device const half* crossK [[buffer(1)]],
        device const half* crossV [[buffer(2)]],
        device const PackCrossGroup* groups [[buffer(3)]],
        device half* output [[buffer(4)]],
        constant CrossAttnFusedParams& params [[buffer(5)]],
        device const float* bias [[buffer(6)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        if (group.x >= params.nHeads || group.y >= params.numGroups) return;
        const PackCrossGroup pack = groups[group.y];
        threadgroup float2 partialStats[8];
        threadgroup float partialValues[8 * 64];

        dispatch_cross_attention<false, false>(
            Q, (device const uchar*)crossK, (device const uchar*)crossV, nullptr, nullptr,
            pack, output, params, bias, group.x, lane, sg, partialStats, partialValues
        );
    }

    // =========================================================================
    // 6. GPU-Side Exact Whisper Timestamp Rules + FP32 Log-Softmax + Top-8
    //    Eliminates 100% of CPU exp() bottleneck (~0.08 ms for 40 beams!).
    // =========================================================================
    struct LogitsFilterRule {
        uint step0;
        uint allowText;
        uint allowEot;
        uint allowTs;
        uint minTs;
        uint maxTs;
        uint suppressNonSpeech;
        uint pad;
    };

    struct LogitsFilterGlobal {
        uint nVocab;
        uint eot;
        uint tsBegin;
        uint topK; // 8
    };

    inline bool tokenAllowed(uint i, LogitsFilterRule r, uchar maskByte, LogitsFilterGlobal g) {
        if ((maskByte & 1u) != 0u) return false;
        if (r.suppressNonSpeech != 0u && (maskByte & 2u) != 0u && i < g.tsBegin) return false;
        if (r.step0 != 0u) {
            return (i >= r.minTs && i <= r.maxTs);
        }
        if (i < g.eot) return (r.allowText != 0u);
        if (i == g.eot) return (r.allowEot != 0u);
        if (i >= g.tsBegin) return (r.allowTs != 0u && i >= r.minTs && i <= r.maxTs);
        return false;
    }

    kernel void whisper_logits_filter_logsoftmax_topk_f32(
        device float* logits [[buffer(0)]],              // [B, nVocab] in-place -> logProbs
        device const uchar* tokenMask [[buffer(1)]],     // [nVocab] bit0=special, bit1=nonSpeech
        device const LogitsFilterRule* rules [[buffer(2)]], // [B]
        device uint* outTopKTokens [[buffer(3)]],        // [B, 8]
        device float* outTopKLogProbs [[buffer(4)]],     // [B, 8]
        constant LogitsFilterGlobal& g [[buffer(5)]],
        device uint* outBestRawTs [[buffer(6)]],         // [B] argmax raw timestamp token ID
        uint b [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]
    ) {
        device float* row = logits + b * g.nVocab;
        const LogitsFilterRule r = rules[b];

        threadgroup float sBuf0[8];
        threadgroup float sBuf1[8];
        threadgroup float sBuf2[8];
        threadgroup uint sIdxBuf[8];

        // Pass 1: Apply mask & find globalMax, maxText, AND argmax raw timestamp logit!
        float locMax = -INFINITY;
        float locMaxText = -INFINITY;
        float locRawTsMax = -INFINITY;
        uint locRawTsIdx = g.tsBegin;
        for (uint i = tid; i < g.nVocab; i += 256) {
            float rawV = row[i];
            if (i >= g.tsBegin && i <= r.maxTs && rawV > locRawTsMax) {
                locRawTsMax = rawV;
                locRawTsIdx = i;
            }
            float v = tokenAllowed(i, r, tokenMask[i], g) ? rawV : -INFINITY;
            row[i] = v;
            locMax = max(locMax, v);
            if (i < g.tsBegin) locMaxText = max(locMaxText, v);
        }
        locMax = simd_max(locMax);
        locMaxText = simd_max(locMaxText);
        float sgRawTsMax = simd_max(locRawTsMax);
        uint sgRawTsIdx = simd_min((locRawTsMax == sgRawTsMax && isfinite(locRawTsMax)) ? locRawTsIdx : 0xFFFFFFFFu);
        if (lane == 0) {
            sBuf0[sg] = locMax;
            sBuf1[sg] = locMaxText;
            sBuf2[sg] = sgRawTsMax;
            sIdxBuf[sg] = sgRawTsIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float m = (lane < 8) ? sBuf0[lane] : -INFINITY;
            float mt = (lane < 8) ? sBuf1[lane] : -INFINITY;
            m = simd_max(m);
            mt = simd_max(mt);
            if (lane == 0) {
                sBuf0[0] = m;
                sBuf1[0] = mt;
                float bestTsV = -INFINITY;
                uint bestTsI = g.tsBegin;
                for (uint s = 0; s < 8; ++s) {
                    if (sBuf2[s] > bestTsV || (sBuf2[s] == bestTsV && sIdxBuf[s] < bestTsI)) {
                        bestTsV = sBuf2[s];
                        bestTsI = sIdxBuf[s];
                    }
                }
                outBestRawTs[b] = bestTsI;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float globalMax = sBuf0[0];
        const float globalMaxText = sBuf1[0];

        // Pass 2: Compute sumAll and sumTs (for OpenAI Whisper tsLogProb > maxTextLogProb rule!)
        float locSumAll = 0.0f;
        float locSumTs = 0.0f;
        if (isfinite(globalMax)) {
            for (uint i = tid; i < g.nVocab; i += 256) {
                float v = row[i];
                if (isfinite(v)) {
                    float e = exp(v - globalMax);
                    locSumAll += e;
                    if (i >= g.tsBegin) locSumTs += e;
                }
            }
        }
        locSumAll = simd_sum(locSumAll);
        locSumTs = simd_sum(locSumTs);
        if (lane == 0) {
            sBuf0[sg] = locSumAll;
            sBuf1[sg] = locSumTs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float sa = (lane < 8) ? sBuf0[lane] : 0.0f;
            float st = (lane < 8) ? sBuf1[lane] : 0.0f;
            sa = simd_sum(sa);
            st = simd_sum(st);
            if (lane == 0) {
                bool forceTs = (r.step0 == 0u) && (st > 0.0f) && ((globalMax + log(st)) > globalMaxText);
                float logZ = isfinite(globalMax) ? (globalMax + log(max(forceTs ? st : sa, 1e-30f))) : 0.0f;
                sBuf0[0] = logZ;
                sBuf2[0] = forceTs ? 1.0f : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float logZ = sBuf0[0];
        const bool forceTs = (sBuf2[0] > 0.5f);

        // Pass 3: Write exact logProbs in-place & track per-thread Top-8 in registers!
        float topV[8];
        uint topI[8];
        for (uint k = 0; k < 8; ++k) { topV[k] = -INFINITY; topI[k] = 0u; }

        for (uint i = tid; i < g.nVocab; i += 256) {
            float v = (forceTs && i < g.tsBegin) ? -INFINITY : row[i];
            float lp = isfinite(v) ? (v - logZ) : -INFINITY;
            row[i] = lp;
            if (lp > topV[7]) {
                // Insertion into sorted 8-element register array
                uint pos = 7;
                while (pos > 0 && lp > topV[pos - 1]) {
                    topV[pos] = topV[pos - 1];
                    topI[pos] = topI[pos - 1];
                    --pos;
                }
                topV[pos] = lp;
                topI[pos] = i;
            }
        }

        // Pass 4: Exact global Top-8 merge across 256 threads (deterministic tie-break by smallest ID)
        uint cursor = 0;
        for (uint k = 0; k < 8; ++k) {
            float myV = (cursor < 8) ? topV[cursor] : -INFINITY;
            uint myI = (cursor < 8) ? topI[cursor] : 0xFFFFFFFFu;

            float sgMax = simd_max(myV);
            uint sgWinI = simd_min((myV == sgMax && isfinite(myV)) ? myI : 0xFFFFFFFFu);
            if (lane == 0) {
                sBuf0[sg] = sgMax;
                sIdxBuf[sg] = sgWinI;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0 && lane == 0) {
                float bestV = -INFINITY;
                uint bestI = g.eot;
                for (uint s = 0; s < 8; ++s) {
                    if (sBuf0[s] > bestV || (sBuf0[s] == bestV && sIdxBuf[s] < bestI)) {
                        bestV = sBuf0[s];
                        bestI = sIdxBuf[s];
                    }
                }
                sBuf0[0] = bestV;
                sIdxBuf[0] = bestI;
                outTopKTokens[b * 8 + k] = bestI;
                outTopKLogProbs[b * 8 + k] = bestV;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (myI == sIdxBuf[0] && myV == sBuf0[0] && cursor < 8) {
                cursor += 1;
            }
        }
    }
    """#
}


struct MetalWhisperLinearParams {
    enum Epilogue: UInt32 {
        case half, gelu, geluPosition, residual, splitQKV, splitCrossKV, logits
    }

    var rows: UInt32
    var columns: UInt32
    var inner: UInt32
    var epilogue: UInt32
    var context: UInt32 = 0
    var heads: UInt32 = 0
    var headDim: UInt32 = 0
    var layerOffset: UInt32 = 0
    var slot0: UInt32 = 0
    var slot1: UInt32 = 0
    var slot2: UInt32 = 0
    var slot3: UInt32 = 0

    init(rows: Int, columns: Int, inner: Int, epilogue: Epilogue) {
        self.rows = UInt32(rows)
        self.columns = UInt32(columns)
        self.inner = UInt32(inner)
        self.epilogue = epilogue.rawValue
    }
}

/// Metal 4 encoder pipelines compiled separately from the Metal 3.2 fallback.
struct MetalWhisperEncoderTensorKernels {
    struct Tile {
        let pipeline: MTLComputePipelineState
        let rows: Int
        let columns: Int
    }

    let attention: MTLComputePipelineState
    let residualLayerNorm: MTLComputePipelineState
    let narrow: Tile
    let square: Tile
    let tall: Tile

    @available(macOS 26.0, *)
    init(device: MTLDevice) throws {
        let library = try MetalWhisperKernels.makeMetal4Library(
            device: device, appending: Self.source + "\n" + Self.residualLayerNormSource
        )

        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let kernel = library.makeFunction(name: name) else {
                throw MetalWhisperError.shaderCompilationFailed("Missing Metal function \(name)")
            }
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = kernel
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
            return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        }

        attention = try pipeline("encoder_flash_attention_tensor_f16")
        residualLayerNorm = try pipeline("layernorm_after_residual_f32")
        narrow = Tile(pipeline: try pipeline("encoder_linear_tall_narrow"), rows: 128, columns: 32)
        square = Tile(pipeline: try pipeline("encoder_linear_square"), rows: 64, columns: 64)
        tall = Tile(pipeline: try pipeline("encoder_linear_tall"), rows: 128, columns: 64)
    }

    func dispatchLinear(
        _ encoder: MTLComputeCommandEncoder,
        params: MetalWhisperLinearParams,
        input: MTLBuffer,
        weights: MTLBuffer,
        output: MTLBuffer,
        bias: MTLBuffer?,
        auxiliary: MTLBuffer?,
        extra: MTLBuffer?
    ) {
        let rows = Int(params.rows)
        let columns = Int(params.columns)
        let tile = params.inner >= params.columns * 2 ? tall : (rows <= 1_500 || columns >= 5_120 ? narrow : square)
        encoder.setComputePipelineState(tile.pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(bias ?? weights, offset: 0, index: 3)
        encoder.setBuffer(auxiliary ?? output, offset: 0, index: 4)
        encoder.setBuffer(extra ?? output, offset: 0, index: 5)
        var uniforms = params
        encoder.setBytes(&uniforms, length: MemoryLayout<MetalWhisperLinearParams>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + tile.rows - 1) / tile.rows, height: (columns + tile.columns - 1) / tile.columns, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    private static var source: String {
        #"""
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        template <uint BR, uint NSG>
        inline void attention_tensor64(
            device half* Q, device half* K, device half* V, device half* O,
            constant SplitQKVParams& params, uint3 group, uint tid, uint lane, uint simd,
            threadgroup float* scores, threadgroup half* probabilities, threadgroup float2* rowStats,
            threadgroup float* rescale
        ) {
            using namespace mpp::tensor_ops;
            constexpr uint BC = 64;
            constexpr uint HD = 64;
            const uint tokens = params.T;
            const uint queryStart = group.x * BR;
            const uint queryCount = min(BR, tokens - queryStart);
            const uint headBase = (group.z * params.nHeads + group.y) * tokens * HD;

            using HeadShape = extents<int32_t, HD, dynamic_extent>;
            using HeadTensor = tensor<device half, HeadShape, tensor_inline>;
            using ProbabilityShape = extents<int32_t, BC, BR>;
            using ProbabilityTensor = tensor<threadgroup half, ProbabilityShape, tensor_inline>;
            auto q = HeadTensor(Q + headBase + queryStart * HD, HeadShape(queryCount));
            auto p = ProbabilityTensor(probabilities, ProbabilityShape{});

            constexpr auto scoreDesc = matmul2d_descriptor(BR, BC, HD, false, true, false);
            constexpr auto outputDesc = matmul2d_descriptor(
                BR, HD, BC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate
            );
            matmul2d<scoreDesc, execution_simdgroups<NSG>> scoreOp;
            matmul2d<outputDesc, execution_simdgroups<NSG>> outputOp;
            auto score = scoreOp.template get_destination_cooperative_tensor<HeadTensor, HeadTensor, float>();
            auto output = outputOp.template get_destination_cooperative_tensor<ProbabilityTensor, HeadTensor, float>();

            if (tid < BR) rowStats[tid] = float2(-INFINITY, 0.0f);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < output.get_capacity(); ++i) {
                if (output.is_valid_element(i)) output[i] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint keyStart = 0; keyStart < tokens; keyStart += BC) {
                const uint keyCount = min(BC, tokens - keyStart);
                auto k = HeadTensor(K + headBase + keyStart * HD, HeadShape(keyCount));
                auto v = HeadTensor(V + headBase + keyStart * HD, HeadShape(keyCount));
                scoreOp.run(q, k, score);
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < score.get_capacity(); ++i) {
                    if (score.is_valid_element(i)) {
                        const auto xy = score.get_multidimensional_index(i);
                        scores[xy[1] * BC + xy[0]] = score[i];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                #pragma clang loop unroll(full)
                for (uint r = simd; r < BR; r += NSG) {
                    const float s0 = lane < keyCount ? (r < queryCount ? scores[r * BC + lane] * 0.125f : 0.0f) : -INFINITY;
                    const float s1 = lane + 32 < keyCount ? (r < queryCount ? scores[r * BC + lane + 32] * 0.125f : 0.0f) : -INFINITY;
                    const float mTile = simd_max(max(s0, s1));
                    const float2 old = rowStats[r];
                    const float m = max(old.x, mTile);
                    const float alpha = exp(old.x - m);
                    const float p0 = exp(s0 - m);
                    const float p1 = exp(s1 - m);
                    const float l = old.y * alpha + simd_sum(p0 + p1);
                    probabilities[r * BC + lane] = half(p0);
                    probabilities[r * BC + lane + 32] = half(p1);
                    if (lane == 0) {
                        rescale[r] = alpha;
                        rowStats[r] = float2(m, l);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < output.get_capacity(); ++i) {
                    if (output.is_valid_element(i)) {
                        const auto xy = output.get_multidimensional_index(i);
                        output[i] *= rescale[xy[1]];
                    }
                }
                outputOp.run(p, v, output);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint outWidth = params.nHeads * HD;
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < output.get_capacity(); ++i) {
                if (output.is_valid_element(i)) {
                    const auto xy = output.get_multidimensional_index(i);
                    const uint r = xy[1];
                    if (r < queryCount) {
                        const uint outRow = group.z * tokens + queryStart + r;
                        O[outRow * outWidth + group.y * HD + xy[0]] = half(output[i] / rowStats[r].y);
                    }
                }
            }
        }

        kernel void encoder_flash_attention_tensor_f16(
            device half* q [[buffer(0)]],
            device half* k [[buffer(1)]],
            device half* v [[buffer(2)]],
            device half* out [[buffer(3)]],
            constant SplitQKVParams& params [[buffer(4)]],
            uint3 group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]
        ) {
            threadgroup float scores[32 * 64];
            threadgroup half probabilities[32 * 64];
            threadgroup float2 rowStats[32];
            threadgroup float rescale[32];
            attention_tensor64<32, 2>(
                q, k, v, out, params, group, tid, lane, simd, scores, probabilities, rowStats, rescale
            );
        }


        struct TensorLinearParams {
            uint rows;
            uint columns;
            uint inner;
            uint epilogue;
            uint context;
            uint heads;
            uint headDim;
            uint layerOffset;
            uint slots[4];
        };

        inline void store_linear_epilogue(
            float sum, uint row, uint col, constant TensorLinearParams& p,
            device uchar* output, device const float* bias, device uchar* auxiliary, device uchar* extra
        ) {
            const uint index = row * p.columns + col;
            if (p.epilogue == 6) {
                ((device float*)output)[index] = sum;
                return;
            }
            // The MPS path rounds the projection to F16 before its epilogue.
            const half rounded = half(sum);
            if (p.epilogue == 0) {
                ((device half*)output)[index] = rounded;
            } else if (p.epilogue == 1 || p.epilogue == 2) {
                const float activated = gelu_erf(float(rounded) + bias[col]);
                const float position = p.epilogue == 2 ? ((device const float*)auxiliary)[(row % p.context) * p.columns + col] : 0.0f;
                ((device half*)output)[index] = half(activated + position);
            } else if (p.epilogue == 3) {
                device half* residual = (device half*)auxiliary;
                const float updated = float(residual[index]) + float(rounded) + bias[col];
                ((device float*)output)[index] = updated;
                residual[index] = half(updated);
            } else {
                const uint width = p.heads * p.headDim;
                const uint feature = col % width;
                const uint head = feature / p.headDim;
                const uint pack = row / p.context;
                const uint slot = p.epilogue == 5 ? p.slots[pack] : pack;
                const uint dst = p.layerOffset + ((slot * p.heads + head) * p.context + row % p.context) * p.headDim + feature % p.headDim;
                device uchar* projection = col < width ? output : (col < 2 * width ? auxiliary : extra);
                ((device half*)projection)[dst] = half(float(rounded) + bias[col]);
            }
        }

        template <uint BM, uint BN>
        inline void encoder_tensor_linear(
            device half* x, device half* w, device uchar* y, device const float* bias,
            device uchar* auxiliary, device uchar* extra, constant TensorLinearParams& params, uint2 group
        ) {
            using namespace mpp::tensor_ops;
            using Shape = dextents<int32_t, 2>;
            using Matrix = tensor<device half, Shape, tensor_inline>;
            const uint row = group.x * BM;
            const uint col = group.y * BN;
            const uint rowCount = min(BM, params.rows - row);
            const uint colCount = min(BN, params.columns - col);
            auto a = Matrix(x + row * params.inner, Shape(params.inner, rowCount));
            auto b = Matrix(w + col * params.inner, Shape(params.inner, colCount));
            constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
            matmul2d<desc, execution_simdgroups<4>> matmul;
            auto sum = matmul.template get_destination_cooperative_tensor<Matrix, Matrix, float>();
            matmul.run(a, b, sum);

            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
                if (!sum.is_valid_element(i)) continue;
                const auto xy = sum.get_multidimensional_index(i);
                if (uint(xy[0]) < colCount && uint(xy[1]) < rowCount) {
                    store_linear_epilogue(sum[i], row + xy[1], col + xy[0], params, y, bias, auxiliary, extra);
                }
            }
        }

        #define ENCODER_LINEAR_KERNEL(NAME, BM, BN) \
        kernel void NAME( \
            device half* x [[buffer(0)]], device half* w [[buffer(1)]], device uchar* y [[buffer(2)]], \
            device const float* bias [[buffer(3)]], device uchar* auxiliary [[buffer(4)]], \
            device uchar* extra [[buffer(5)]], constant TensorLinearParams& params [[buffer(6)]], \
            uint2 group [[threadgroup_position_in_grid]] \
        ) { \
            encoder_tensor_linear<BM, BN>(x, w, y, bias, auxiliary, extra, params, group); \
        }

        ENCODER_LINEAR_KERNEL(encoder_linear_tall_narrow, 128, 32)
        ENCODER_LINEAR_KERNEL(encoder_linear_square, 64, 64)
        ENCODER_LINEAR_KERNEL(encoder_linear_tall, 128, 64)
        #undef ENCODER_LINEAR_KERNEL
        """#
    }

    static let residualLayerNormSource = #"""
        // Preserve the legacy LayerNorm boundary: FP32 residual moments and F16
        // residual values for normalization, followed by a single F16 output rounding.
        kernel void layernorm_after_residual_f32(
            device const float* moments [[buffer(0)]],
            device const half* residual [[buffer(1)]],
            device half* output [[buffer(2)]],
            device const float* weight [[buffer(3)]],
            device const float* bias [[buffer(4)]],
            constant uint& width [[buffer(5)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]
        ) {
            const uint base = row * width;
            float sum = 0.0f;
            float squares = 0.0f;
            for (uint i = tid; i < width; i += 256) {
                const float value = moments[base + i];
                sum += value;
                squares += value * value;
            }
            sum = simd_sum(sum);
            squares = simd_sum(squares);
            threadgroup float sums[8];
            threadgroup float sumsOfSquares[8];
            if (lane == 0) {
                sums[simd] = sum;
                sumsOfSquares[simd] = squares;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd == 0) {
                const float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
                const float totalSquares = simd_sum(lane < 8 ? sumsOfSquares[lane] : 0.0f);
                if (lane == 0) {
                    const float mean = total / float(width);
                    const float variance = max(0.0f, totalSquares / float(width) - mean * mean);
                    sums[0] = mean;
                    sumsOfSquares[0] = rsqrt(variance + 1e-5f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float mean = sums[0];
            const float inverseStdDev = sumsOfSquares[0];
            for (uint i = tid; i < width; i += 256) {
                const float normalized = (float(residual[base + i]) - mean) * inverseStdDev;
                output[base + i] = half(normalized * weight[i] + bias[i]);
            }
        }
        """#
}

enum MetalWhisperDecoderEpilogue: UInt16 {
    case projection, gelu, residual, logits
}

struct MetalWhisperDecoderTensorKernels {
    private struct Tile {
        let rows: Int
        let columns: Int
        let simdgroups: Int
        let pipelines: [MetalWhisperDecoderEpilogue: MTLComputePipelineState]
    }

    private struct Params {
        let rows: UInt32
        let columns: UInt32
        let inner: UInt32
        let highBitsOffset: UInt32
        let scalesOffset: UInt32
    }

    let residualLayerNorm: MTLComputePipelineState?
    private let f16Small: Tile?
    private let f16Large: Tile?
    private let q5Small: Tile
    private let q5Medium: Tile?
    private let q5Large: Tile?
    private let q5LogitsMedium: Tile

    @available(macOS 26.0, *)
    init(device: MTLDevice, headOnly: Bool = false) throws {
        let library = try MetalWhisperKernels.makeMetal4Library(
            device: device, appending: Self.source + "\n" + MetalWhisperEncoderTensorKernels.residualLayerNormSource
        )

        func pipeline(_ name: String, epilogue: MetalWhisperDecoderEpilogue? = nil) throws -> MTLComputePipelineState {
            let function: MTLFunction
            if let epilogue {
                let constants = MTLFunctionConstantValues()
                var mode = epilogue.rawValue
                constants.setConstantValue(&mode, type: .ushort, index: 0)
                function = try library.makeFunction(name: name, constantValues: constants)
            } else {
                guard let kernel = library.makeFunction(name: name) else {
                    throw MetalWhisperError.shaderCompilationFailed("Missing Metal function \(name)")
                }
                function = kernel
            }
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
            return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        }

        let projections: [MetalWhisperDecoderEpilogue] = [.projection, .gelu, .residual]
        func tile(
            _ name: String, _ rows: Int, _ columns: Int, _ simdgroups: Int, modes: [MetalWhisperDecoderEpilogue]
        ) throws -> Tile {
            let pipelines = try Dictionary(uniqueKeysWithValues: modes.map { ($0, try pipeline(name, epilogue: $0)) })
            return Tile(rows: rows, columns: columns, simdgroups: simdgroups, pipelines: pipelines)
        }

        residualLayerNorm = headOnly ? nil : try pipeline("layernorm_after_residual_f32")
        f16Small = headOnly ? nil : try tile("decoder_f16_16x32", 16, 32, 2, modes: projections)
        f16Large = headOnly ? nil : try tile("decoder_f16_64x32", 64, 32, 4, modes: projections)
        q5Small = try tile("decoder_q5_16x64", 16, 64, 4, modes: headOnly ? [.logits] : projections + [.logits])
        q5Medium = headOnly ? nil : try tile("decoder_q5_32x32", 32, 32, 2, modes: projections)
        q5Large = headOnly ? nil : try tile("decoder_q5_128x32", 128, 32, 4, modes: projections + [.logits])
        q5LogitsMedium = try tile("decoder_q5_48x128", 48, 128, 4, modes: [.logits])
    }

    func dispatch(
        _ encoder: MTLComputeCommandEncoder, rows: Int, columns: Int, inner: Int,
        input: MTLBuffer, f16: MTLBuffer, q5: MetalWhisperQ5Weights?, output: MTLBuffer,
        epilogue: MetalWhisperDecoderEpilogue = .projection, bias: MTLBuffer? = nil, hidden: MTLBuffer? = nil
    ) {
        let tile: Tile
        if q5 == nil {
            tile = rows <= 16 ? f16Small! : f16Large!
        } else if rows <= 16 {
            tile = q5Small
        } else if epilogue == .logits && rows <= 48 {
            tile = q5LogitsMedium
        } else if epilogue == .logits || rows > 64 {
            tile = q5Large!
        } else {
            tile = q5Medium!
        }
        encoder.setComputePipelineState(tile.pipelines[epilogue]!)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(q5?.buffer ?? f16, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(bias ?? f16, offset: 0, index: 3)
        encoder.setBuffer(hidden ?? output, offset: 0, index: 4)
        var params = Params(
            rows: UInt32(rows), columns: UInt32(columns), inner: UInt32(inner),
            highBitsOffset: UInt32(q5?.highBitsOffset ?? 0), scalesOffset: UInt32(q5?.scalesOffset ?? 0)
        )
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + tile.rows - 1) / tile.rows, height: (columns + tile.columns - 1) / tile.columns, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tile.simdgroups * 32, height: 1, depth: 1)
        )
    }

    private static let source = #"""
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        struct DecoderLinearParams {
            uint rows;
            uint columns;
            uint inner;
            uint highBitsOffset;
            uint scalesOffset;
        };

        constant ushort decoderEpilogue [[function_constant(0)]];

        inline void decoder_store_projection(
            float sum, uint row, uint column, constant DecoderLinearParams& p,
            device uchar* output, device const float* bias, device half* hidden
        ) {
            const uint index = row * p.columns + column;
            if (decoderEpilogue == 3) {
                ((device float*)output)[index] = sum;
            } else {
                // Attention, GELU and residuals consume the same F16-rounded
                // projection as they do on the MPS path.
                const half projected = half(sum);
                if (decoderEpilogue == 0) {
                    ((device half*)output)[index] = projected;
                } else if (decoderEpilogue == 1) {
                    ((device half*)output)[index] = half(gelu_erf(float(projected) + bias[column]));
                } else {
                    const float updated = float(hidden[index]) + float(projected) + bias[column];
                    ((device float*)output)[index] = updated;
                    hidden[index] = half(updated);
                }
            }
        }

        template <uint BM, uint BN, uint NSG>
        inline void decoder_linear_f16(
            device half* input, device uchar* weights, device uchar* output, device const float* bias,
            device half* hidden, constant DecoderLinearParams& p, uint2 group
        ) {
            using namespace mpp::tensor_ops;
            using Shape = dextents<int32_t, 2>;
            using Matrix = tensor<device half, Shape, tensor_inline>;
            const uint firstRow = group.x * BM;
            const uint firstColumn = group.y * BN;
            const uint rows = min(BM, p.rows - firstRow);
            const uint columns = min(BN, p.columns - firstColumn);
            auto a = Matrix(input + firstRow * p.inner, Shape(p.inner, rows));
            auto b = Matrix(((device half*)weights) + firstColumn * p.inner, Shape(p.inner, columns));
            constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
            matmul2d<desc, execution_simdgroups<NSG>> op;
            auto acc = op.template get_destination_cooperative_tensor<Matrix, Matrix, float>();
            op.run(a, b, acc);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto xy = acc.get_multidimensional_index(i);
                if (uint(xy[1]) < rows && uint(xy[0]) < columns) {
                    decoder_store_projection(acc[i], firstRow + xy[1], firstColumn + xy[0], p, output, bias, hidden);
                }
            }
        }

        template <uint BK, uint BN, uint THREADS>
        inline void stage_q5_weights(
            device const uchar4* nibbles, device const uint* high, device const half* scales,
            threadgroup half* tile, uint columns, uint firstColumn, uint firstBlock, uint tid
        ) {
            for (uint i = tid; i < (BK / 32) * BN * 4; i += THREADS) {
                const uint colInTile = (i / 4) % BN;
                const uint kBlock = i / (BN * 4);
                const uint word = i % 4;
                const uint column = firstColumn + colInTile;
                half4 lower = half4(0.0h);
                half4 upper = half4(0.0h);
                if (column < columns) {
                    const uint block = (firstBlock + kBlock) * columns + column;
                    const uint4 loBits = uint4(nibbles[block * 4 + word]);
                    const uint4 shifts = word * 4 + uint4(0, 1, 2, 3);
                    const uint4 highBits = uint4(high[block]) >> shifts;
                    const float scale = float(scales[block]);
                    const int4 lo = int4((loBits & 15u) | ((highBits & 1u) << 4u)) - 16;
                    const int4 hi = int4((loBits >> 4u) | ((highBits >> 12u) & 16u)) - 16;
                    lower = half4(float4(lo) * scale);
                    upper = half4(float4(hi) * scale);
                }
                threadgroup half* dst = tile + colInTile * BK + kBlock * 32 + word * 4;
                *(threadgroup half4*)dst = lower;
                *(threadgroup half4*)(dst + 16) = upper;
            }
        }

        template <uint BM, uint BN, uint BK, uint NSG, bool DOUBLE_BUFFERED>
        inline void decoder_linear_q5(
            device half* input, device uchar* weights, device uchar* output, device const float* bias,
            device half* hidden, constant DecoderLinearParams& p, uint2 group, uint tid, threadgroup half* tile
        ) {
            using namespace mpp::tensor_ops;
            using LeftShape = dextents<int32_t, 2>;
            using RightShape = extents<int32_t, BK, BN>;
            using Left = tensor<device half, LeftShape, tensor_inline>;
            using Right = tensor<threadgroup half, RightShape, tensor_inline>;
            const uint firstRow = group.x * BM;
            const uint firstColumn = group.y * BN;
            const uint rows = min(BM, p.rows - firstRow);
            const uint columns = min(BN, p.columns - firstColumn);
            const auto packed = (device const uchar4*)weights;
            const auto high = (device const uint*)(weights + p.highBitsOffset);
            const auto scales = (device const half*)(weights + p.scalesOffset);
            const array<int32_t, 2> aStrides = {1, int32_t(p.inner)};
            constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc, execution_simdgroups<NSG>> op;
            auto acc = op.template get_destination_cooperative_tensor<Left, Right, float>();
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) acc[i] = 0.0f;
            }
            for (uint k = 0; k < p.inner; k += BK) {
                threadgroup half* staged = DOUBLE_BUFFERED ? tile + ((k / BK) & 1) * BK * BN : tile;
                stage_q5_weights<BK, BN, NSG * 32>(packed, high, scales, staged, p.columns, firstColumn, k / 32, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto a = Left(input + firstRow * p.inner + k, LeftShape(BK, rows), aStrides);
                auto b = Right(staged, RightShape{});
                op.run(a, b, acc);
                if constexpr (!DOUBLE_BUFFERED) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (decoderEpilogue == 3) {
                using OutputShape = dextents<int32_t, 2>;
                using Output = tensor<device float, OutputShape, tensor_inline>;
                const array<int32_t, 2> stride = {1, int32_t(p.columns)};
                auto dst = Output(((device float*)output) + firstRow * p.columns + firstColumn,
                                  OutputShape(columns, rows), stride);
                acc.store(dst);
                return;
            }
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto xy = acc.get_multidimensional_index(i);
                if (uint(xy[1]) < rows && uint(xy[0]) < columns) {
                    decoder_store_projection(acc[i], firstRow + xy[1], firstColumn + xy[0], p, output, bias, hidden);
                }
            }
        }

        #define DECODER_LINEAR_ARGS \
            device half* input [[buffer(0)]], device uchar* weights [[buffer(1)]], device uchar* output [[buffer(2)]], \
            device const float* bias [[buffer(3)]], device half* hidden [[buffer(4)]], \
            constant DecoderLinearParams& params [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]]

        #define DECODER_F16(NAME, BM, BN, NSG) \
        kernel void NAME(DECODER_LINEAR_ARGS) { \
            decoder_linear_f16<BM, BN, NSG>(input, weights, output, bias, hidden, params, group); \
        }

        #define DECODER_Q5(NAME, BM, BN, BK, NSG, DOUBLE_BUFFERED) \
        kernel void NAME(DECODER_LINEAR_ARGS, uint tid [[thread_index_in_threadgroup]]) { \
            threadgroup half tile[(DOUBLE_BUFFERED ? 2 : 1) * BN * BK]; \
            decoder_linear_q5<BM, BN, BK, NSG, DOUBLE_BUFFERED>(input, weights, output, bias, hidden, params, group, tid, tile); \
        }

        DECODER_F16(decoder_f16_16x32, 16, 32, 2)
        DECODER_F16(decoder_f16_64x32, 64, 32, 4)
        DECODER_Q5(decoder_q5_16x64, 16, 64, 64, 4, false)
        DECODER_Q5(decoder_q5_32x32, 32, 32, 64, 2, false)
        DECODER_Q5(decoder_q5_128x32, 128, 32, 64, 4, false)
        DECODER_Q5(decoder_q5_48x128, 48, 128, 32, 4, true)
        #undef DECODER_Q5
        #undef DECODER_F16
        #undef DECODER_LINEAR_ARGS
        """#
}

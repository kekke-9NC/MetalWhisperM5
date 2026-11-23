import Metal
import XCTest
@testable import MetalWhisperM5

final class HardwareFallbackTests: XCTestCase {
    func testWorkingSetLimits() {
        let gibibyte: UInt64 = 1 << 30
        let tiers: [(UInt64, Int)] = [(0, 4), (16, 6), (24, 8), (32, 12), (40, 16), (48, 24)]
        for (size, expected) in tiers {
            XCTAssertEqual(MetalWhisperEngine.recommendedBatchPacks(workingSetSize: size * gibibyte), expected)
        }
        for index in tiers.indices.dropFirst() {
            let size = tiers[index].0 * gibibyte
            XCTAssertEqual(MetalWhisperEngine.recommendedBatchPacks(workingSetSize: size - 1), tiers[index - 1].1)
        }
    }

    func testCooperativeAttentionUsesModelShape() {
        func parameters(context: Int, width: Int, heads: Int) -> WhisperHParams {
            WhisperHParams(
                nVocab: 51_866, nAudioCtx: context, nAudioState: width, nAudioHead: heads, nAudioLayer: 32,
                nTextCtx: 448, nTextState: width, nTextHead: heads, nTextLayer: 32, nMels: 128, ftype: 8
            )
        }
        XCTAssertTrue(MetalWhisperEngine.supportsCooperativeAttention(hparams: parameters(context: 1500, width: 1280, heads: 20)))
        XCTAssertFalse(MetalWhisperEngine.supportsCooperativeAttention(hparams: parameters(context: 500, width: 1280, heads: 20)))
        XCTAssertFalse(MetalWhisperEngine.supportsCooperativeAttention(hparams: parameters(context: 1500, width: 60, heads: 1)))
        XCTAssertFalse(MetalWhisperEngine.supportsCooperativeAttention(hparams: parameters(context: 1500, width: 1280, heads: 0)))
    }

    func testPortablePipelinesWithMetal4Disabled() throws {
        let device = try requireMetalDevice()
        let kernels = try MetalWhisperKernels(
            device: device, enableTensorEncoder: false, enableCooperativeAttention: false,
            crossKVFormat: .f16, enableRegisterAttention: false
        )
        XCTAssertNil(kernels.encoderFlashAttentionRegisterF16)
        XCTAssertNil(kernels.encoderFlashAttentionCooperative)
        XCTAssertNil(kernels.tensorEncoder)
        XCTAssertNil(kernels.splitCrossKVQ8)
        XCTAssertNil(kernels.decoderCrossAttnPackFusedQ8)
        XCTAssertGreaterThan(kernels.encoderFlashAttentionF16.threadExecutionWidth, 0)
        XCTAssertGreaterThan(kernels.decoderCrossAttnPackFusedF16.threadExecutionWidth, 0)
    }
}

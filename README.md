# MetalWhisperM5

[![Swift CI](https://github.com/kekke-9NC/MetalWhisperM5/actions/workflows/ci.yml/badge.svg)](https://github.com/kekke-9NC/MetalWhisperM5/actions/workflows/ci.yml)

[日本語](README.ja.md)

**Fastest in our Apple M5 Pro benchmark for Whisper large-v3 Q5_0.** MetalWhisperM5 transcribed 10 minutes and 38 seconds of English audio in **6.409 seconds (99.531× real time)** at beam size 5. With the same audio, model, and beam size, `whisper-cli` took 50.925 seconds and the first `whisper-server` request after model loading took 51.030 seconds. [Benchmark results and reproduction steps](Benchmarks/README.md)

MetalWhisperM5 runs Whisper speech recognition locally on Apple Silicon. It is a Swift library and a command-line tool that read GGML models directly and use the Metal, Metal Performance Shaders (MPS), Accelerate, and AVFoundation frameworks. The Swift package has zero third-party dependencies. Pass an audio file or 16 kHz mono PCM to get text and timestamped segments and tokens.

## Requirements

- Apple Silicon Mac with macOS 14 or later and a Metal GPU. Building requires Xcode 16 or later or the matching Command Line Tools.
- A compatible Whisper GGML `.bin` model. The loader supports F32, F16, Q5_0, and Q8_0 tensors. The release configuration uses `ggml-large-v3-q5_0.bin`, supplied separately. [Download from the whisper.cpp model repository](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-q5_0.bin).
- Metal 4 optimizations require a build with Xcode 26 or later, macOS 26 or later at runtime, an Apple10-capable GPU such as the M5 family, compatible model dimensions, and successful shader compilation. The portable path uses MPS, F16 Cross-KV, and Metal 3.1 on macOS 14 or Metal 3.2 from macOS 15 onward. Physical M1–M4 and macOS 14–15 configurations remain experimental pending hardware verification.

## Quick start

From the repository root, download the public 11-second JFK sample and run the CLI with your model. The CLI also accepts WAV, M4A, MP3, FLAC, and other formats that AVAudioFile can open.

```sh
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
Benchmarks/download_public_samples.sh --short-only
swift run -c release metal-whisper \
  --model /path/to/ggml-large-v3-q5_0.bin \
  --audio Benchmarks/samples/jfk.wav --language en
```

```sh
swift run -c release metal-whisper \
  --model /path/to/ggml-large-v3-q5_0.bin \
  --audio audio.m4a --language ja --format srt --output transcript.srt \
  --backend mps --cross-kv f16 --profile
```

| Option | Behavior |
| --- | --- |
| `--model <path>` | Pass the model path or set `METAL_WHISPER_MODEL_PATH`. |
| `--audio <path>` | Pass the source audio file. |
| `--language <code>` | Use auto detection by default or specify a code such as `en` or `ja`. |
| `--prompt <text>` | Supply an initial prompt for decoding. |
| `--beam-size <1...5>` | Set beam width. The default is 5. |
| `--backend auto\|cooperative\|mps` | Select the encoder backend. `mps` also selects the MPS decoder. The default is `auto`. |
| `--cross-kv auto\|q8\|f16` | Select Cross-KV storage. The default is `auto`. |
| `--max-batch <count>` | Limit the number of audio windows in flight. The default uses the GPU working set. |
| `--format text\|json\|srt\|vtt` | Select the transcript format. The default is `text`. |
| `--output <path>` | Write the transcript to a file. The default is standard output. |
| `--profile` | Emit a `MetalWhisper profile {JSON}` line to standard error with phase timings and the selected backends. |
| `--help`, `--version` | Show usage or version. |

## Use from Swift

Add the package and library product to your project's `Package.swift` declarations. For a local checkout, use `.package(path: "../MetalWhisperM5")` in the same dependency list.

```swift
// Package dependencies
.package(url: "https://github.com/kekke-9NC/MetalWhisperM5.git", branch: "main")

// Dependencies for the importing target
.product(name: "MetalWhisperM5", package: "MetalWhisperM5")
```

```swift
import Foundation
import MetalWhisperM5

let engine = try MetalWhisperEngine(
    modelURL: URL(fileURLWithPath: "ggml-large-v3-q5_0.bin"),
    configuration: .init(enableProfiling: true)
)
let transcript = try engine.transcribe(
    audioURL: URL(fileURLWithPath: "audio.wav"),
    options: .init(languageCode: "en")
)
print(transcript.text)
for segment in transcript.segments {
    print(segment.start, segment.end, segment.text)
}
```

- Reuse the engine to keep model buffers and compiled pipelines loaded. Inspect the model dimensions through `engine.modelParameters`.
- Use `transcribe(samples16kHz:options:)` for mono Float32 PCM. Both high-level methods split audio into consecutive model windows, normally 30 seconds, and return segment and token timestamps in seconds relative to the original audio.
- Use `transcribeBatch(packs:options:)` for caller-defined windows, such as speech regions from a voice activity detector. Each `MetalWhisperPackInput` supplies its own prompt and a sample array or deferred provider. Results follow input order and timestamps are relative to each pack.
- `MetalWhisperConfiguration` selects encoder and decoder backends, Cross-KV storage, the in-flight limit, decode-ahead preparation, and profiling. `MetalWhisperDecodeOptions` controls language, prompt, beam search, and temperature retries. `MetalWhisperProfileReport` gives phase timings and the path used for that transcription.

## Inference design

- **Cooperative FlashAttention.** Metal 4 tensor operations process full encoder tiles. The final query tile retains its higher-precision accumulation path.
- **Packed Q8 Cross-KV.** Unrotated keys and values use per-frame, per-head scales. Beams for the same audio share the Cross-KV cache.
- **Continuous batching.** As windows finish, recycled KV slots admit new windows and keep decoder rows active.
- **Decode-ahead MPS.** The CPU prepares the next command buffer while the GPU runs the current decode step. The engine reuses it when the next batch shape matches.
- **Native GGML loading.** The model loader reads weights from the local file into Metal buffers. Accelerate computes Whisper log-mel features.

[Architecture and precision details](Docs/ARCHITECTURE.md).

## Benchmarks

The benchmark compares the same model, English audio, and Beam 5 across the Swift CLI and the available whisper.cpp executables. The timed CLI runs include process startup and model loading. The server row times the first HTTP request after the model has loaded. RT× is audio duration divided by wall time. GPU utilization and memory are system-wide ioreg samples. RSS is measured per process. [Measurement details and reproduction commands](Benchmarks/README.md).

| US public-domain sample | Duration | Benchmark WAV SHA-256 |
| --- | ---: | --- |
| John F. Kennedy's 1961 US inaugural address excerpt | 11.000 s | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| Jane Austen's *Pride and Prejudice*, chapter 7, read by Elizabeth Klett for LibriVox | 637.933 s | `0082ea67e628a60f90b3d0516c949f990a1dda0f2319839031bbedbdae6e2476` |

[Inaugural address and recording](https://www.jfklibrary.org/archives/other-resources/john-f-kennedy-speeches/inaugural-address-19610120) · [Pinned whisper.cpp excerpt](https://github.com/ggml-org/whisper.cpp/blob/2eeeba56e9edd762b4b38467bab96c2517163158/samples/jfk.wav) · [LibriVox catalog and public-domain status](https://librivox.org/4023).

Measured on November 23, 2026, on an Apple M5 Pro (18 CPU cores, 20 GPU cores, 64 GiB) on AC power with macOS 27.0.1 (26A434), Xcode and macOS SDK 27.2, and Swift 6.4.2. Each row is one run using `ggml-large-v3-q5_0.bin`, English, and beam size 5. MetalWhisperM5 used a release build with automatic backend, Cross-KV and batch selection. The comparison used whisper.cpp 1.9.1 with Metal, FlashAttention and four CPU threads. Each timed launch or request began after an ioreg reading of 0% GPU utilization.

| Sample | Engine | Wall time (s) | RT× | GPU mean / peak | Peak GPU memory (GiB) | Peak RSS (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JFK | `metal-whisper` | 0.657 | 16.733 | 60.67% / 96% | 4.516 | 3.973 |
| JFK | `whisper-cli` | 1.045 | 10.530 | 56.40% / 94% | 3.184 | 1.948 |
| JFK | `whisper-server` (HTTP) | 0.703 | 15.655 | 68.50% / 96% | 3.215 | 1.951 |
| LibriVox ch. 7 | `metal-whisper` | 6.409 | 99.531 | 91.69% / 99% | 9.065 | 3.973 |
| LibriVox ch. 7 | `whisper-cli` | 50.925 | 12.527 | 86.57% / 100% | 3.497 | 2.072 |
| LibriVox ch. 7 | `whisper-server` (HTTP) | 51.030 | 12.501 | 86.96% / 100% | 3.518 | 2.140 |

## Development and license

Run `swift test` for model-free tests. GPU tests skip when Metal or the required GPU feature is unavailable. Set `METAL_WHISPER_MODEL_PATH` to enable model integration tests. [Contributor guide](CONTRIBUTING.md).

MIT licensed. Thanks to OpenAI for Whisper and to the whisper.cpp contributors for the GGML model format, conversion tools, and comparison executables. [License](LICENSE) · [Whisper](https://github.com/openai/whisper) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

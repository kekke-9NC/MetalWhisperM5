# Contributing

## Build and test

Use an Apple Silicon Mac and Xcode 16 or later, or the matching Command Line Tools. Use Xcode 26 or later to verify the Metal 4 paths. From the repository root, run the release build and the model-free suite.

```sh
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
swift build -c release
env -u METAL_WHISPER_MODEL_PATH -u METAL_WHISPER_TEST_AUDIO_PATH swift test
```

The suite runs CPU checks for audio conversion, timestamps, output formats, KV slot recycling, and backend selection. GPU tests exercise numerical kernels when the required GPU is available. GitHub Actions runs the model-free build and tests on Apple Silicon runners for macOS 26 and macOS 15 with Xcode 16.4. [CI workflow](.github/workflows/ci.yml).

Run the model integration test on a Mac with a Metal GPU. The test verifies the public JFK sample against its pinned SHA-256, then checks transcription and exact text across batch sizes. It compares the word sequence across the automatic and MPS/F16 paths because the two paths can produce different punctuation. It loads the sample from `Benchmarks/samples/jfk.wav` or a local path supplied in `METAL_WHISPER_TEST_AUDIO_PATH`. If both are absent, the opt-in test downloads the pinned sample.

```sh
Benchmarks/download_public_samples.sh --short-only
METAL_WHISPER_MODEL_PATH=/path/to/ggml-large-v3-q5_0.bin swift test
```

## Benchmark individual kernels

Each Swift source under `Benchmarks/Microbenchmarks` lists its build command. Run the resulting executable with `--help` for inputs. [Microbenchmarks](Benchmarks/Microbenchmarks).

## Describe the change

- For a bug report, include the macOS and SDK versions, chip, relevant model and quantization, a command or minimal Swift example, and the observed result. Use synthetic or redistributable public audio for reproduction.
- In a pull request, describe the behavior changed and the tests run. When editing Metal paths, exercise the affected optimized and portable routes. Include precision results for changes to reductions, layout, or quantization.
- For a performance change, use the same public audio, model SHA-256, language, and beam size before and after. Include wall time, RT×, and the run settings. Identify any phases measured separately, such as server startup. [Benchmark guide](Benchmarks/README.md).
- Keep runtime changes within Swift and system frameworks. Reference public fixtures with pinned hashes and keep audio, model weights, transcripts, and build products out of commits.

## Source layout

| Location | Contents |
| --- | --- |
| `Sources/MetalWhisperM5` | Public API, audio loading, GGML loading, shader sources, and inference scheduling |
| `Sources/MetalWhisperCLI` | CLI argument parsing and transcript formats |
| `Tests/MetalWhisperM5Tests` | CPU, GPU, and model integration tests |
| `Benchmarks` | Public sample downloads, end-to-end comparison, and kernel microbenchmarks |

Contributions are distributed under the project's MIT license. [License](LICENSE).

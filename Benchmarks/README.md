# Benchmarks

The benchmark measures one 11-second speech clip and one complete audiobook chapter. Each engine receives the same 16 kHz mono PCM16 WAV and GGML model, using English and beam size 5.

## Public-domain samples

| Sample | Source | Duration | SHA-256 of benchmark WAV |
| --- | --- | ---: | --- |
| `jfk.wav` | John F. Kennedy's January 20, 1961 US presidential inaugural address, excerpt distributed by whisper.cpp | 11.000000 s | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| `pride-and-prejudice-ch07.wav` | Jane Austen, *Pride and Prejudice* (version 4), chapter 7, read by Elizabeth Klett for LibriVox | 637.933437 s | `0082ea67e628a60f90b3d0516c949f990a1dda0f2319839031bbedbdae6e2476` |

The inaugural address recording is in the US public domain. The script fetches the WAV at whisper.cpp commit `2eeeba56e9edd762b4b38467bab96c2517163158` and verifies its SHA-256. [National Archives address](https://www.archives.gov/milestone-documents/president-john-f-kennedys-inaugural-address) · [Recording provenance and rights](https://commons.wikimedia.org/wiki/File:JFK_inaugural_address.ogg) · [Pinned whisper.cpp clip](https://github.com/ggml-org/whisper.cpp/blob/2eeeba56e9edd762b4b38467bab96c2517163158/samples/jfk.wav)

LibriVox releases the chapter recording into the public domain in the US. The original chapter MP3, `prideandprejudice_07_austen.mp3`, has SHA-256 `46e69a138a2b7b4b775a235d0f11508e7c8d8bf757d033f44631c045f514f2d6`. The downloader verifies the source bytes, removes metadata and converts to PCM16. The table shows the output from FFmpeg 8.1.2. Check the printed WAV SHA-256 when using a different FFmpeg version. [LibriVox catalog](https://librivox.org/4023) · [Internet Archive source](https://archive.org/download/prideandprejudice_1005_librivox/prideandprejudice_07_austen.mp3)

## Reproduce

Run from the repository root on Apple Silicon macOS with a compatible Swift toolchain and macOS SDK. The scripts use Bash, Python 3.9 or newer, curl, shasum, ffmpeg, and ffprobe. For comparison, install Metal-enabled `whisper-cli` and `whisper-server` from whisper.cpp and put them on `PATH`. [whisper.cpp build instructions](https://github.com/ggml-org/whisper.cpp#quick-start)

```bash
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
swift build -c release
Benchmarks/download_public_samples.sh
Benchmarks/run_benchmark.sh --model /path/to/ggml-large-v3-q5_0.bin
```

The downloader saves files to `Benchmarks/samples` by default. To use a separate directory or fetch the short fixture alone, pass a destination and optionally `--short-only`. The benchmark accepts repeatable `--audio` arguments. Pass `--metal-whisper` to select a separate release build or `--metal-only` to run the Swift executable alone.

```bash
Benchmarks/download_public_samples.sh --short-only /tmp/whisper-samples
Benchmarks/run_benchmark.sh --model /path/to/ggml-large-v3-q5_0.bin \
  --audio /tmp/whisper-samples/jfk.wav --output-dir /tmp/whisper-results
```

For another public WAV, use `ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le input-16k.wav` and pass `--audio input-16k.wav`. Record the source, rights, duration and SHA-256 alongside any published result.

### Measurement

- CLI wall times cover a fresh process from launch to exit, including model loading and transcript output. The OS page cache may already contain the model. Swift uses automatic backend, Cross-KV storage and batch size by default. The whisper.cpp tools use Metal with FlashAttention, four CPU threads and best-of 5. Each implementation uses its own windowing and decode logic.
- The server binds to loopback and loads the model once per clip. Startup is timed until the health endpoint reports ready. The first model-loaded request is timed separately, including HTTP upload, inference and response. The harness records multipart upload and response-body bytes. Reported server RSS covers its full lifetime including startup.
- Before launching each timed process and before sending the request, the harness waits for the `IOAccelerator` GPU device utilization counter to report 0%. The idle wait is outside wall time and defaults to 60 seconds. Pause unrelated GPU work before testing.
- The harness samples `IOAccelerator` device utilization and system memory in use roughly every 200 ms and records readings before, during and after. Mean utilization and peaks use the in-window samples. These readings are system-wide and can include compositor activity. The sampled GPU memory peak and the process peak RSS from macOS `/usr/bin/time -l` describe different measurements.
- RT× is audio duration divided by wall time. The server row uses the HTTP request duration. Empty fields mean the metric was unavailable or did not apply.
- `results.csv` holds one row per sample and engine. `environment.json` records the system, SDK, settings, and model, audio and executable hashes. Per-run files contain ioreg samples, transcripts and timing logs. The default output is a dated directory under `Benchmarks/results`.
- The standard run leaves Swift phase profiling disabled. Pass `--profile` for instrumented measurements with effective backend settings and detailed phase timing in the Swift stderr log.

## Measured results

Measured on November 23, 2026, on an Apple M5 Pro (18 CPU cores, 20 GPU cores, 64 GiB) on AC power with macOS 27.0.1 (26A434), Xcode and macOS SDK 27.2, and Swift 6.4.2. Each row is one run using `ggml-large-v3-q5_0.bin`, English, and beam size 5. MetalWhisperM5 used a release build with automatic backend, Cross-KV and batch selection. The comparison used whisper.cpp 1.9.1 with Metal, FlashAttention and four CPU threads. Each timed launch or request began after an ioreg reading of 0% GPU utilization.

Separate profiled runs confirmed cooperative FlashAttention and packed Q8 Cross-KV on both samples, with the Q5 tensor decoder head for JFK and the MPS decoder for the 22-window LibriVox chapter. The automatic in-flight limit was 24 packs.

The model SHA-256 was `d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1`. The long WAV was produced with FFmpeg 8.1.2.

| Sample | Engine | Wall time (s) | RT× | GPU mean / peak | Peak GPU memory (GiB) | Peak RSS (GiB) | GPU samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JFK | `metal-whisper` | 0.657 | 16.733 | 60.67% / 96% | 4.516 | 3.973 | 3 |
| JFK | `whisper-cli` | 1.045 | 10.530 | 56.40% / 94% | 3.184 | 1.948 | 5 |
| JFK | `whisper-server` (HTTP) | 0.703 | 15.655 | 68.50% / 96% | 3.215 | 1.951 | 4 |
| LibriVox ch. 7 | `metal-whisper` | 6.409 | 99.531 | 91.69% / 99% | 9.065 | 3.973 | 29 |
| LibriVox ch. 7 | `whisper-cli` | 50.925 | 12.527 | 86.57% / 100% | 3.497 | 2.072 | 224 |
| LibriVox ch. 7 | `whisper-server` (HTTP) | 51.030 | 12.501 | 86.96% / 100% | 3.518 | 2.140 | 224 |

| Sample | Server ready (s) | HTTP upload (bytes) | Response body (bytes) |
| --- | ---: | ---: | ---: |
| JFK | 0.387 | 352,390 | 112 |
| LibriVox ch. 7 | 0.434 | 20,414,226 | 11,472 |

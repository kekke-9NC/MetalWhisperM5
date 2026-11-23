#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$script_dir" "$@" <<'PY'
import argparse
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

BENCHMARKS = Path(sys.argv[1])
BEAMS = 5
GPU_INTERVAL = 0.2


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_benchmark.sh",
        description="Measure Whisper transcription on 16 kHz mono PCM16 WAV files on macOS.",
    )
    parser.add_argument("--model", default=os.getenv("METAL_WHISPER_MODEL_PATH"),
                        help="GGML model path (defaults to METAL_WHISPER_MODEL_PATH)")
    parser.add_argument("--audio", type=Path, action="append",
                        help="input WAV, repeat to add files (defaults to both public samples)")
    parser.add_argument("--metal-whisper", default=str(BENCHMARKS.parent / ".build/release/metal-whisper"),
                        help="path to a release build of metal-whisper")
    parser.add_argument("--output-dir", type=Path, help="new or empty results directory")
    parser.add_argument("--language", default="en")
    parser.add_argument("--backend", choices=["auto", "cooperative", "mps"], default="auto")
    parser.add_argument("--cross-kv", choices=["auto", "q8", "f16"], default="auto")
    parser.add_argument("--max-batch", type=int)
    parser.add_argument("--profile", action="store_true", help="enable metal-whisper phase profiling")
    parser.add_argument("--metal-only", action="store_true", help="omit the optional whisper.cpp comparators")
    parser.add_argument("--threads", type=int, default=4, help="threads for whisper.cpp")
    parser.add_argument("--idle-timeout", type=int, default=60, help="seconds to wait for a 0%% GPU sample")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds allowed per execution or HTTP request")
    args = parser.parse_args(sys.argv[2:])
    if args.model is None:
        parser.error("provide --model or set METAL_WHISPER_MODEL_PATH")
    if min(args.idle_timeout, args.timeout, args.threads) < 1 or (args.max_batch is not None and args.max_batch < 1):
        parser.error("timeouts, threads, and max batch must be positive")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def output(*command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def audio_duration(path):
    probe = json.loads(output(
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=format_name", "-of", "json", str(path),
    ))
    streams = probe["streams"]
    if probe["format"]["format_name"] != "wav" or len(streams) != 1 or any(
        streams[0].get(key) != value
        for key, value in (("codec_name", "pcm_s16le"), ("sample_rate", "16000"), ("channels", 1))
    ):
        raise RuntimeError(f"Expected a 16 kHz mono PCM16 WAV: {path}")
    duration = float(streams[0]["duration"])
    if duration <= 0:
        raise RuntimeError(f"Expected positive audio duration: {path}")
    return duration


def gpu_snapshot():
    try:
        plist = subprocess.check_output(
            ["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-a"],
            stderr=subprocess.DEVNULL, timeout=3,
        )
        stats = [entry.get("PerformanceStatistics", {}) for entry in plistlib.loads(plist)]
    except (OSError, ValueError, plistlib.InvalidFileException, subprocess.SubprocessError):
        return None, None
    utilization = [entry["Device Utilization %"] for entry in stats if "Device Utilization %" in entry]
    memory = [entry["In use system memory"] for entry in stats if "In use system memory" in entry]
    return max(utilization, default=None), sum(memory) if memory else None


def idle_gpu(seconds):
    deadline = time.monotonic() + seconds
    last_usage = None
    while time.monotonic() < deadline:
        usage, memory = gpu_snapshot()
        if usage == 0:
            return usage, memory
        if usage is not None:
            last_usage = usage
        time.sleep(GPU_INTERVAL)
    if last_usage is None:
        raise RuntimeError("IOAccelerator did not report Device Utilization %")
    raise RuntimeError(f"GPU did not report 0% before launch (last reading {last_usage}%)")


@contextmanager
def capture_gpu(path, idle_timeout):
    before = idle_gpu(idle_timeout)
    start = time.monotonic()
    samples = [("before", 0.0, *before)]
    done = threading.Event()

    def sample_during():
        while not done.is_set():
            usage, memory = gpu_snapshot()
            if done.is_set():
                break
            samples.append(("during", time.monotonic() - start, usage, memory))
            done.wait(GPU_INTERVAL)

    worker = threading.Thread(target=sample_during)
    worker.start()
    try:
        yield samples
    finally:
        done.set()
        worker.join()
        after = gpu_snapshot()
        samples.append(("after", time.monotonic() - start, *after))
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("phase", "elapsed_s", "gpu_utilization_pct", "gpu_in_use_bytes"))
            writer.writerows((phase, f"{elapsed:.6f}", usage, memory) for phase, elapsed, usage, memory in samples)


@contextmanager
def timed_process(command, logs, stdout):
    with stdout.open("w") as out, logs.with_suffix(".stderr.log").open("w") as err:
        process = subprocess.Popen(
            ["/usr/bin/time", "-l", "-o", str(logs.with_suffix(".time.txt")), *command],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def measure_command(command, logs, transcript, args):
    print(f"Running {logs.name}", file=sys.stderr, flush=True)
    with capture_gpu(logs.with_suffix(".ioreg.tsv"), args.idle_timeout) as samples:
        start = time.perf_counter()
        with timed_process(command, logs, transcript) as process:
            status = process.wait(args.timeout)
        elapsed = time.perf_counter() - start
        if status:
            raise RuntimeError(f"{logs.name} exited with status {status}. See {logs.with_suffix('.stderr.log')}.")
    return elapsed, samples, None, None, None


def wait_for_server(process, base_url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError("whisper-server exited during startup. See whisper-server.stderr.log.")
        try:
            with opener.open(base_url + "/health", timeout=0.25) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError("whisper-server did not report ready after 60 seconds.")


def measure_server(binary, shared_flags, audio, logs, args):
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]
    request_path = "/benchmark-" + os.urandom(8).hex()
    base_url = f"http://127.0.0.1:{port}{request_path}"
    public_dir = logs.parent / "empty-public"
    public_dir.mkdir()
    command = [binary, *shared_flags, "--host", "127.0.0.1", "--port", str(port),
               "--public", str(public_dir), "--request-path", request_path]
    # Keep the process alive across startup and inference, retaining one lifetime RSS measurement.
    print("Starting whisper-server", file=sys.stderr, flush=True)
    with ExitStack() as cleanup:
        with capture_gpu(logs.with_suffix(".startup-ioreg.tsv"), args.idle_timeout):
            start = time.perf_counter()
            server = cleanup.enter_context(timed_process(command, logs, logs.with_suffix(".stdout.log")))
            wait_for_server(server, base_url)
            ready = time.perf_counter() - start
        with capture_gpu(logs.with_suffix(".ioreg.tsv"), args.idle_timeout) as samples:
            with audio.open("rb") as wav:
                start = time.perf_counter()
                upload, download = subprocess.check_output(
                    [
                        "/usr/bin/curl", "--noproxy", "*", "--silent", "--show-error", "--fail",
                        "--max-time", str(args.timeout), "--output", str(logs.with_suffix(".txt")),
                        "--write-out", "%{size_upload} %{size_download}",
                        "--form", "file=@-;filename=input.wav;type=audio/wav",
                        "--form-string", "response_format=text", base_url + "/inference",
                    ],
                    stdin=wav, text=True, timeout=args.timeout + 5,
                ).split()
                elapsed = time.perf_counter() - start
    return elapsed, samples, ready, int(upload), int(download)


def metrics(engine, audio, duration, logs, measurement):
    elapsed, samples, ready, upload, download = measurement
    time_log = logs.with_suffix(".time.txt").read_text()
    rss_match = re.search(r"^\s*(\d+)\s+maximum resident set size", time_log, re.MULTILINE)
    peak_rss = int(rss_match[1]) if rss_match else None
    usage = [usage for phase, _, usage, _ in samples if phase == "during" and usage is not None]
    memory = [memory for phase, _, _, memory in samples if phase == "during" and memory is not None]
    return {
        "sample": audio.name,
        "duration_s": duration,
        "engine": engine,
        "wall_s": round(elapsed, 6),
        "realtime_x": round(duration / elapsed, 3),
        "gpu_before_pct": samples[0][2],
        "gpu_mean_pct": round(statistics.mean(usage), 2) if usage else None,
        "gpu_peak_pct": max(usage, default=None),
        "gpu_after_pct": samples[-1][2],
        "gpu_samples": len(usage),
        "gpu_peak_in_use_bytes": max(memory, default=None),
        "peak_rss_bytes": peak_rss,
        "server_ready_s": round(ready, 6) if ready is not None else None,
        "http_upload_bytes": upload,
        "http_response_bytes": download,
    }


def main():
    args = parse_args()
    if sys.platform != "darwin":
        raise RuntimeError("This benchmark requires macOS.")
    metal = shutil.which(args.metal_whisper)
    if not metal:
        raise RuntimeError("Build metal-whisper with swift build -c release or pass --metal-whisper.")
    cli = None if args.metal_only else shutil.which("whisper-cli")
    server = None if args.metal_only else shutil.which("whisper-server")
    for binary, executable in (("whisper-cli", cli), ("whisper-server", server)):
        if executable is None and not args.metal_only:
            print(f"{binary} is unavailable; omitting its benchmark.", file=sys.stderr)
    model = Path(args.model).resolve(strict=True)
    audios = args.audio or [
        BENCHMARKS / "samples/jfk.wav", BENCHMARKS / "samples/pride-and-prejudice-ch07.wav",
    ]
    audios = [path.resolve(strict=True) for path in audios]
    durations = [audio_duration(path) for path in audios]
    results = args.output_dir or BENCHMARKS / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    results = results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    if any(results.iterdir()):
        raise RuntimeError(f"Results directory is not empty: {results}")
    sdk = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-version"],
                         capture_output=True, text=True, timeout=15)
    metadata = {
        "chip": output("sysctl", "-n", "machdep.cpu.brand_string"),
        "memoryBytes": int(output("sysctl", "-n", "hw.memsize")),
        "macOS": output("sw_vers", "-productVersion"),
        "macOSBuild": output("sw_vers", "-buildVersion"),
        "sdk": sdk.stdout.strip() if sdk.returncode == 0 else None,
        "model": model.name,
        "modelSHA256": sha256(model),
        "beamSize": BEAMS,
        "language": args.language,
        "whisperCppThreads": args.threads,
        "metalWhisper": {"backend": args.backend, "crossKV": args.cross_kv,
                         "maxBatch": args.max_batch, "profile": args.profile},
        "binariesSHA256": {
            name: sha256(Path(path)) for name, path in
            (("metal-whisper", metal), ("whisper-cli", cli), ("whisper-server", server)) if path
        },
        "audio": [{"name": path.name, "sha256": sha256(path), "durationSeconds": duration}
                  for path, duration in zip(audios, durations)],
    }
    (results / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cpp = ["--model", str(model), "--language", args.language, "--beam-size", str(BEAMS),
           "--threads", str(args.threads), "--best-of", str(BEAMS), "--suppress-nst", "--flash-attn"]
    metal_flags = ["--model", str(model), "--language", args.language, "--beam-size", str(BEAMS),
                   "--backend", args.backend, "--cross-kv", args.cross_kv, "--format", "text"]
    if args.max_batch:
        metal_flags += ["--max-batch", str(args.max_batch)]
    if args.profile:
        metal_flags += ["--profile"]

    print(f"Results directory: {results}", file=sys.stderr, flush=True)
    with (results / "results.csv").open("w", newline="") as result_file:
        writer = None
        for index, (audio, duration) in enumerate(zip(audios, durations), start=1):
            run_dir = results / f"sample-{index:02d}"
            run_dir.mkdir()
            commands = [("metal-whisper", [metal, *metal_flags, "--audio", str(audio)])]
            if cli:
                commands.append(("whisper-cli", [cli, *cpp, "--file", str(audio)]))
            for name, command in commands + ([("whisper-server", None)] if server else []):
                logs = run_dir / name
                if command is None:
                    measurement = measure_server(server, cpp, audio, logs, args)
                else:
                    measurement = measure_command(command, logs, logs.with_suffix(".txt"), args)
                row = metrics(name, audio, duration, logs, measurement)
                if writer is None:
                    writer = csv.DictWriter(result_file, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                result_file.flush()
                gpu = row["gpu_mean_pct"]
                print(f"{audio.name:<32} {name:<15} {row['wall_s']:9.3f} s  "
                      f"{row['realtime_x']:8.2f}×  GPU mean {str(gpu) + '%' if gpu is not None else 'n/a'}")


def interrupted(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Benchmark interrupted.")
    except (RuntimeError, OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        sys.exit(f"Benchmark failed: {error}")
PY

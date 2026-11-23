#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--short-only] [destination]\n' "${0##*/}"
}

short_only=0
destination=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --short-only) short_only=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *)
            if [[ -n "$destination" ]]; then usage >&2; exit 2; fi
            destination="$1"
            ;;
    esac
    shift
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${destination:-"$script_dir/samples"}"
mkdir -p -- "$destination"
destination="$(cd -- "$destination" && pwd)"
staging="$(mktemp -d "$destination/.download.XXXXXX")"
cleanup() {
    local file
    for file in "$staging/download" "$staging/chapter.wav"; do
        if [[ -e "$file" ]]; then rm -- "$file"; fi
    done
    rmdir -- "$staging"
}
trap cleanup EXIT

verify_sha256() {
    local file="$1" expected="$2" actual
    actual="$(shasum -a 256 < "$file")"
    if [[ "${actual%% *}" != "$expected" ]]; then
        printf 'SHA-256 mismatch for %s\n' "$file" >&2
        return 1
    fi
}

download() {
    local url="$1" file="$2" expected="$3"
    if [[ -f "$file" ]]; then
        verify_sha256 "$file" "$expected"
        return
    fi
    curl --fail --silent --show-error --location --retry 3 --connect-timeout 10 \
        --max-time 120 --output "$staging/download" "$url"
    verify_sha256 "$staging/download" "$expected"
    mv -- "$staging/download" "$file"
}

download 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/2eeeba56e9edd762b4b38467bab96c2517163158/samples/jfk.wav' \
    "$destination/jfk.wav" '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
shasum -a 256 "$destination/jfk.wav"

if [[ "$short_only" == 1 ]]; then exit 0; fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'Install ffmpeg to convert the LibriVox sample, or use --short-only.\n' >&2
    exit 1
fi

download 'https://archive.org/download/prideandprejudice_1005_librivox/prideandprejudice_07_austen.mp3' \
    "$destination/prideandprejudice_07_austen.mp3" '46e69a138a2b7b4b775a235d0f11508e7c8d8bf757d033f44631c045f514f2d6'
output="$destination/pride-and-prejudice-ch07.wav"
ffmpeg -v error -nostdin -i "$destination/prideandprejudice_07_austen.mp3" \
    -map 0:a:0 -map_metadata -1 -map_chapters -1 -fflags +bitexact -flags:a +bitexact \
    -ar 16000 -ac 1 -c:a pcm_s16le -f wav "$staging/chapter.wav"
mv -- "$staging/chapter.wav" "$output"
shasum -a 256 "$destination/prideandprejudice_07_austen.mp3" "$output"

# MetalWhisperM5

[![Swift CI](https://github.com/kekke-9NC/MetalWhisperM5/actions/workflows/ci.yml/badge.svg)](https://github.com/kekke-9NC/MetalWhisperM5/actions/workflows/ci.yml)

[English](README.md)

**Whisper large-v3 Q5_0の実測比較で最速（Apple M5 Pro）。** ビーム幅5で10分38秒の英語音声を **6.409秒（実時間の99.531倍速）** で文字起こししました。同じ音声・モデル・ビーム幅で、`whisper-cli` は50.925秒、`whisper-server` はモデル読み込み後の最初のリクエストに51.030秒かかりました。[実測結果と再現手順](Benchmarks/README.md)

MetalWhisperM5は、Apple Silicon MacでWhisperの音声認識をローカル実行するSwiftライブラリとCLIです。GGMLモデルを直接読み込み、Metal、Metal Performance Shaders（MPS）、Accelerate、AVFoundationで推論します。Swift Packageの外部依存はありません。音声ファイルまたは16 kHzモノラルPCMから、文字起こしとタイムスタンプ付きのセグメント・トークンを取得できます。

## 動作環境

- macOS 14以降のApple Silicon MacとMetal対応GPU。ビルドにはXcode 16以降、または対応するCommand Line Toolsが必要です。
- Whisper用のGGML `.bin` モデル。F32、F16、Q5_0、Q8_0のテンソル形式に対応します。リリースの検証には別途取得する `ggml-large-v3-q5_0.bin` を使います。[whisper.cppのモデル配布元](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-q5_0.bin)
- Metal 4最適化には、Xcode 26以降でのビルド、実行環境のmacOS 26以降、M5世代などのApple10対応GPU、適合するモデル形状、実行時のシェーダーコンパイル成功が必要です。フォールバックはMPSとF16 Cross-KVを使い、macOS 14ではMetal 3.1、macOS 15以降ではMetal 3.2で動作します。M1〜M4およびmacOS 14〜15での動作は実験的で、該当環境での実機検証が残っています。

## クイックスタート

リポジトリのルートで、公開されているJFK就任演説の11秒の抜粋を取得し、手元のモデルで文字起こしします。CLIはWAV、M4A、MP3、FLACなど、AVAudioFileで開ける音声形式を読み込めます。

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

| オプション | 動作 |
| --- | --- |
| `--model <path>` | モデルのパスを指定します。環境変数 `METAL_WHISPER_MODEL_PATH` でも指定できます。 |
| `--audio <path>` | 入力音声を指定します。 |
| `--language <code>` | 既定は自動判定です。`en` や `ja` などの言語コードを指定できます。 |
| `--prompt <text>` | デコード時の初期プロンプトを指定します。 |
| `--beam-size <1...5>` | ビーム幅を指定します。既定は5です。 |
| `--backend auto\|cooperative\|mps` | エンコーダーを選択します。`mps` はデコーダーもMPSに設定します。既定は `auto` です。 |
| `--cross-kv auto\|q8\|f16` | Cross-KVの格納形式を指定します。既定は `auto` です。 |
| `--max-batch <count>` | 同時処理する音声ウィンドウ数の上限を指定します。既定はGPUの推奨ワーキングセットに基づきます。 |
| `--format text\|json\|srt\|vtt` | 出力形式を指定します。既定は `text` です。 |
| `--output <path>` | 指定先に結果を書き込みます。既定は標準出力です。 |
| `--profile` | 各工程の処理時間と使用したバックエンドを `MetalWhisper profile {JSON}` の形式で標準エラー出力に出します。 |
| `--help`、`--version` | ヘルプまたはバージョンを表示します。 |

## Swiftから使う

パッケージとライブラリ製品を、利用側の `Package.swift` の該当する依存関係に追加します。ローカルのチェックアウトを使う場合は `.package(path: "../MetalWhisperM5")` を指定できます。

```swift
// パッケージの依存関係
.package(url: "https://github.com/kekke-9NC/MetalWhisperM5.git", branch: "main")

// インポート元ターゲットの依存関係
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
    options: .init(languageCode: "ja")
)
print(transcript.text)
for segment in transcript.segments {
    print(segment.start, segment.end, segment.text)
}
```

- 同じエンジンを再利用すると、モデルのバッファとコンパイル済みのパイプラインを保持できます。モデル形状は `engine.modelParameters` で確認できます。
- モノラルFloat32 PCMは `transcribe(samples16kHz:options:)` に渡せます。ファイルとPCMのどちらもモデルのウィンドウ幅（通常は30秒）で分割し、セグメントとトークンの時刻を元の音声の先頭からの秒数で返します。
- 音声区間検出などで分割済みのウィンドウには `transcribeBatch(packs:options:)` を使います。`MetalWhisperPackInput` ごとにプロンプトとサンプル配列、またはサンプルを後から供給するクロージャーを指定でき、入力順で結果を受け取れます。時刻は各ウィンドウの先頭からの秒数です。
- `MetalWhisperConfiguration` でバックエンド、Cross-KV、同時実行数、デコードの先行エンコード、プロファイルの有効化を設定します。`MetalWhisperDecodeOptions` は言語、プロンプト、ビーム探索、温度フォールバックを制御します。`MetalWhisperProfileReport` には推論で使用した経路と工程別の時間が入ります。

## 推論の仕組み

- **Cooperative FlashAttention** は完全なエンコーダータイルをMetal 4のテンソル演算で処理し、末尾のクエリタイルは高精度の積和経路で処理します。
- **Packed Q8 Cross-KV** は回転なしのキー・バリューをフレームとヘッドごとのスケールで格納します。同一音声のビームでCross-KVを共有します。
- **Continuous Batching** は完了したウィンドウのKVスロットを回収し、待機中のウィンドウを投入してデコーダーを稼働させます。
- **Decode-Ahead MPS** はGPUでデコード中に、CPUで次のコマンドバッファを準備します。次のバッチ形状が一致したときに再利用します。
- **GGMLの直接読み込み** でローカルの重みをMetalバッファへ展開し、AccelerateでWhisperのログメル特徴量を計算します。

[アーキテクチャと数値精度の詳細（英語）](Docs/ARCHITECTURE.md)

## ベンチマーク

Swift CLIとwhisper.cppの実行ファイルを、同一モデル、英語音声、Beam 5で比較します。CLIの実行時間にはプロセス起動とモデル読み込みが含まれます。サーバーはモデル読み込み後の最初のHTTPリクエストを計測します。RT×は音声の長さを実行時間で割った値です。GPU稼働率とメモリはioregで取得したシステム全体の値です。RSSはプロセス単位で計測します。[計測方法と再現コマンド](Benchmarks/README.md)

| 米国でパブリックドメインの音声 | 長さ | 測定用WAVのSHA-256 |
| --- | ---: | --- |
| ジョン・F・ケネディ米大統領の1961年就任演説の抜粋 | 11.000秒 | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| ジェイン・オースティン『高慢と偏見』第7章、Elizabeth KlettによるLibriVox朗読 | 637.933秒 | `0082ea67e628a60f90b3d0516c949f990a1dda0f2319839031bbedbdae6e2476` |

[JFK大統領図書館の演説資料](https://www.jfklibrary.org/archives/other-resources/john-f-kennedy-speeches/inaugural-address-19610120) · [whisper.cppの固定版抜粋](https://github.com/ggml-org/whisper.cpp/blob/2eeeba56e9edd762b4b38467bab96c2517163158/samples/jfk.wav) · [LibriVoxの出典と利用条件](https://librivox.org/4023)

2026年11月23日、AC電源接続のApple M5 Pro（CPU 18コア、GPU 20コア、64 GiB）、macOS 27.0.1（26A434）、Xcode・macOS SDK 27.2、Swift 6.4.2で実測しました。各行は1回の測定です。全経路で `ggml-large-v3-q5_0.bin`、英語、ビーム幅5を使用しました。MetalWhisperM5はリリースビルドでバックエンド・Cross-KV・バッチ数を自動選択し、whisper.cpp 1.9.1はMetal・FlashAttention・CPU 4スレッドで実行しました。各プロセス起動・HTTPリクエストの前に `ioreg` でGPU稼働率0%を確認しました。

| 音声 | エンジン | 実時間（秒） | RT× | GPU平均 / 最大 | GPUメモリ最大（GiB） | ピークRSS（GiB） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JFK | `metal-whisper` | 0.657 | 16.733 | 60.67% / 96% | 4.516 | 3.973 |
| JFK | `whisper-cli` | 1.045 | 10.530 | 56.40% / 94% | 3.184 | 1.948 |
| JFK | `whisper-server`（HTTP） | 0.703 | 15.655 | 68.50% / 96% | 3.215 | 1.951 |
| LibriVox 第7章 | `metal-whisper` | 6.409 | 99.531 | 91.69% / 99% | 9.065 | 3.973 |
| LibriVox 第7章 | `whisper-cli` | 50.925 | 12.527 | 86.57% / 100% | 3.497 | 2.072 |
| LibriVox 第7章 | `whisper-server`（HTTP） | 51.030 | 12.501 | 86.96% / 100% | 3.518 | 2.140 |

## 開発・ライセンス

モデルなしのテストは `swift test` で実行します。Metal GPUや必要なGPU機能が利用できない場合は、該当するテストをスキップします。`METAL_WHISPER_MODEL_PATH` を設定すると、モデルを使う統合テストも実行します。[コントリビューションガイド（英語）](CONTRIBUTING.md)

MIT Licenseで公開します。Whisperを公開したOpenAIと、GGMLモデル形式・変換ツール・比較用の実行ファイルを整備したwhisper.cppの開発者に感謝します。[ライセンス](LICENSE) · [Whisper](https://github.com/openai/whisper) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

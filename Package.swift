// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MetalWhisperM5",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MetalWhisperM5", targets: ["MetalWhisperM5"]),
        .executable(name: "metal-whisper", targets: ["MetalWhisperCLI"])
    ],
    targets: [
        .target(name: "MetalWhisperM5"),
        .executableTarget(name: "MetalWhisperCLI", dependencies: ["MetalWhisperM5"]),
        .testTarget(name: "MetalWhisperM5Tests", dependencies: ["MetalWhisperM5", "MetalWhisperCLI"])
    ]
)

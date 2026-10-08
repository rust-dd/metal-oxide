#!/usr/bin/env python3
"""Build and measure matching Rust/MSL compute workloads on both Metal backends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = {
    "vec-add": "examples/vec-add/kernels/vec_add.metal",
    "reduction": "benchmarks/reference-msl/reduction.metal",
    "cooperation": "benchmarks/reference-msl/cooperation.metal",
    "particle-update": "examples/particle-update/reference.metal",
    "matmul": "benchmarks/reference-msl/matmul.metal",
}


def command(args, *, env=None, log=None):
    args = list(map(str, args))
    start = time.perf_counter_ns()
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=1800)
    duration = time.perf_counter_ns() - start
    if log:
        log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{args!r} failed ({result.returncode}):\n{result.stdout}")
    return result.stdout.strip(), duration


def metal_command():
    prefix = ["xcrun", "-sdk", "macosx", "metal"]
    try:
        command(prefix + ["--version"])
        return prefix
    except RuntimeError:
        info, _ = command(["xcodebuild", "-showComponent", "MetalToolchain", "-json"])
        directory = Path(json.loads(info)["toolchainSearchPath"]) / "Metal.xctoolchain/usr/metal"
        candidates = sorted((p for p in directory.iterdir() if p.name.isdigit()),
                            key=lambda p: int(p.name), reverse=True)
        for candidate in candidates:
            binary = candidate / "bin/metal"
            if binary.is_file():
                command([binary, "--version"])
                return [str(binary)]
        raise RuntimeError("install Metal with xcodebuild -downloadComponent MetalToolchain")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_binary(package, environment, log):
    output, _ = command(["cargo", "build", "-p", package, "--release", "--locked",
                         "--message-format=json"], env=environment, log=log)
    binaries = []
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (message.get("reason") == "compiler-artifact"
                and message["target"]["name"] == package and message.get("executable")):
            binaries.append(Path(message["executable"]))
    if len(binaries) != 1:
        raise RuntimeError(f"Cargo did not report exactly one {package} executable; see {log}")
    return binaries[0]


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive, default=20)
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--elements", type=positive, default=1_000_003)
    parser.add_argument("--matrix", nargs=3, type=positive, default=[129, 131, 127],
                        metavar=("ROWS", "COLUMNS", "INNER"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("benchmarks require macOS on Apple Silicon")
    if args.elements > 16_000_000 or max(args.matrix) > 1024:
        parser.error("elements must be <= 16000000; matrix dimensions must be <= 1024")
    parent = ROOT / "target/benchmarks"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    output = args.output.resolve() if args.output else run / "report.json"
    print(f"Benchmark work directory: {run}", flush=True)
    cli = build_binary("cargo-metal", os.environ, run / "cli.log")
    environment = dict(os.environ, CARGO_TARGET_DIR=str(run / "build"))
    artifacts, references, builds, identities = {}, {}, {}, {}
    metal = metal_command()
    sdk, _ = command(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
    metal_env = dict(os.environ, SDKROOT=sdk)
    for package, source in REFERENCES.items():
        print(f"Building {package}: first use and cache hit", flush=True)
        build_command = [cli, "build", "-p", package, "--release"]
        first, cold_ns = command(build_command, env=environment, log=run / f"{package}-first.log")
        match = re.search(r"Built Metal artifact (.+)", first)
        if not match:
            raise RuntimeError(f"expected fresh artifact for {package}:\n{first}")
        artifact = Path(match[1].strip())
        warm, warm_ns = command(build_command, env=environment, log=run / f"{package}-warm.log")
        if f"Cached Metal artifact {artifact}" not in warm:
            raise RuntimeError(f"expected artifact cache hit for {package}:\n{warm}")
        artifacts[package] = str(artifact)
        manifest = json.loads((artifact / "manifest.json").read_text())
        reference = run / f"{package}.metallib"
        air = run / f"{package}.ir"
        compile_command = metal + manifest["build"]["metal_flags"] + ["-c", str(ROOT / source), "-o", str(air)]
        _, compile_ns = command(compile_command, env=metal_env, log=run / f"{package}-reference.log")
        _, link_ns = command(metal + [air, "-o", reference], env=metal_env)
        references[package] = str(reference)
        builds[package] = {"command": list(map(str, build_command)),
                           "first_use_ns": cold_ns, "cache_hit_ns": warm_ns,
                           "reference_compile_ns": compile_ns, "reference_link_ns": link_ns,
                           "reference_command": compile_command}
        identities[package] = {"manifest": manifest, "reference_source": source,
                               "reference_sha256": digest(ROOT / source),
                               "reference_metallib_sha256": digest(reference)}
    binding = str(Path(artifacts["particle-update"]) / "bindings.rs")
    host_env = dict(os.environ, METAL_OXIDE_BENCH_BINDINGS=binding)
    host = build_binary("metal-oxide-bench", host_env, run / "host.log")
    config = {"artifacts": artifacts, "references": references, "samples": args.samples,
              "warmup": args.warmup, "elements": args.elements, "matrix": args.matrix}
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    backends = []
    for backend in ["classic", "metal4"]:
        print(f"Measuring {backend}", flush=True)
        result = run / f"{backend}.json"
        command([host, config_path, result],
                env=dict(host_env, METAL_OXIDE_BACKEND=backend), log=run / f"{backend}.log")
        backends.append(json.loads(result.read_text()))
    revision, _ = command(["git", "rev-parse", "HEAD"])
    status, _ = command(["git", "status", "--porcelain"])
    rustc, _ = command(["rustc", "-vV"])
    xcode, _ = command(["xcodebuild", "-version"])
    os_version, _ = command(["sw_vers"])
    report = {"schema": 1, "revision": revision, "dirty": bool(status),
              "timestamp_unix": time.time(), "os": os_version, "xcode": xcode,
              "host_rustc": rustc, "host_binary_sha256": digest(host),
              "host_build_environment": {k: v for k, v in host_env.items()
                  if k in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET")
                  or k.startswith("CARGO_PROFILE_RELEASE_")},
              "config": config, "builds": builds,
              "identities": identities, "backends": backends,
              "validation_environment": {k: v for k, v in os.environ.items()
                  if k.startswith(("MTL_", "METAL_", "DYLD_"))},
              "timing": {"unit": "ns", "host": "monotonic Instant/perf_counter",
                  "gpu": "Metal completion GPUStartTime/GPUEndTime; null when unavailable",
                  "build": "fresh artifact directory; shared dependencies warm after first package",
                  "submit": "includes encoding; overlaps GPU work and is not additive with GPU time",
                  "wait": "host blocking after commit, not isolated queue latency",
                  "cpu": "serial reference including result allocation",
                  "percentiles": "p10 lower, p90 upper rank at fraction*(n-1); median midpoint"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()

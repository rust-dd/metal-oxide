#!/usr/bin/env python3
"""Build and measure matching Rust/MSL compute workloads on both Metal backends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
# Each package with the source file that holds its kernels and the handwritten reference MSL.
WORKLOADS = {
    "vec-add": ("examples/vec-add/src/main.rs", "crates/metal-oxide/tests/kernels/vec_add.metal"),
    "reduction": ("examples/reduction/src/main.rs", "benchmarks/reference-msl/reduction.metal"),
    "cooperation": ("examples/cooperation/host/src/main.rs",
                    "benchmarks/reference-msl/cooperation.metal"),
    "particle-update": ("examples/particle-update/src/lib.rs",
                        "examples/particle-update/reference.metal"),
    "matmul": ("examples/matmul/host/src/main.rs", "benchmarks/reference-msl/matmul.metal"),
    "gemm": ("examples/gemm/host/src/main.rs", "benchmarks/reference-msl/gemm.metal"),
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


def build_kernels(package, source, environment, log, arguments=()):
    """Builds `package` with plain Cargo, then again after touching `source`, which reuses its
    kernels; returns the artifact that the kernel macros compiled, the command and both times."""
    listing = log.with_suffix(".artifacts")
    listing.unlink(missing_ok=True)
    environment = dict(environment, METAL_OXIDE_INSPECT=str(listing))
    build = ["cargo", "build", "-p", package, "--release", "--locked", *arguments]

    def artifacts():
        lines = listing.read_text().splitlines() if listing.exists() else []
        return {fields[0] for fields in map(lambda line: line.split("\t"), lines)
                if fields[1] == package}

    _, first_ns = command(build, env=environment, log=log)
    found = artifacts()
    if len(found) != 1:
        raise RuntimeError(f"expected one Metal artifact for {package}, found {sorted(found)}")
    os.utime(ROOT / source)
    _, rebuild_ns = command(build, env=environment, log=log.with_name(f"{log.stem}-rebuild.log"))
    if artifacts() != found:
        raise RuntimeError(f"rebuilding {package} with unchanged kernels changed its artifact")
    return Path(found.pop()), build, first_ns, rebuild_ns


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive, default=20)
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--elements", type=positive, default=1_000_003)
    parser.add_argument("--matrix", nargs=3, type=positive, default=[129, 131, 127],
                        metavar=("ROWS", "COLUMNS", "INNER"))
    parser.add_argument("--gemm", nargs=3, type=positive, default=[1024, 1024, 1024],
                        metavar=("M", "N", "K"))
    parser.add_argument("--gemm-repeat", type=positive, default=8, metavar="R")
    parser.add_argument("--output", type=Path)
    return parser


def check_bounds(parser, args):
    if args.elements > 16_000_000 or max(args.matrix) > 1024:
        parser.error("elements must be <= 16000000; matrix dimensions must be <= 1024")
    if max(args.gemm) > 8192:
        parser.error("gemm dimensions must be <= 8192")
    if args.gemm_repeat > 256:
        parser.error("gemm repeat must be <= 256")


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("benchmarks require macOS on Apple Silicon")
    check_bounds(parser, args)
    parent = ROOT / "target/benchmarks"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    output = args.output.resolve() if args.output else run / "report.json"
    print(f"Benchmark work directory: {run}", flush=True)
    environment = dict(os.environ, CARGO_TARGET_DIR=str(run / "build"))
    artifacts, references, builds, identities = {}, {}, {}, {}
    metal = metal_command()
    sdk, _ = command(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
    metal_env = dict(os.environ, SDKROOT=sdk)
    for package, (kernels, source) in WORKLOADS.items():
        print(f"Building {package}: first build and a rebuild that reuses its kernels", flush=True)
        artifact, build_command, first_ns, rebuild_ns = build_kernels(
            package, kernels, environment, run / f"{package}-build.log")
        artifacts[package] = str(artifact)
        manifest = json.loads((artifact / "manifest.json").read_text())
        reference = run / f"{package}.metallib"
        air = run / f"{package}.ir"
        compile_command = metal + manifest["build"]["metal_flags"] + ["-c", str(ROOT / source), "-o", str(air)]
        _, compile_ns = command(compile_command, env=metal_env, log=run / f"{package}-reference.log")
        _, link_ns = command(metal + [air, "-o", reference], env=metal_env)
        references[package] = str(reference)
        builds[package] = {"command": build_command,
                           "first_build_ns": first_ns, "rebuild_ns": rebuild_ns,
                           "reference_compile_ns": compile_ns, "reference_link_ns": link_ns,
                           "reference_command": compile_command}
        identities[package] = {"manifest": manifest, "reference_source": source,
                               "reference_sha256": digest(ROOT / source),
                               "reference_metallib_sha256": digest(reference)}
    host_env = dict(os.environ)
    host = build_binary("metal-oxide-bench", host_env, run / "host.log")
    config = {"artifacts": artifacts, "references": references, "samples": args.samples,
              "warmup": args.warmup, "elements": args.elements, "matrix": args.matrix,
              "gemm": args.gemm, "gemm_repeat": args.gemm_repeat}
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    backends = []
    for backend in ["classic", "metal4"]:
        print(f"Measuring {backend}", flush=True)
        result = run / f"{backend}.json"
        command([host, config_path, result],
                env=dict(host_env, METAL_OXIDE_BACKEND=backend), log=run / f"{backend}.log")
        backends.extend(json.loads(result.read_text()))
    # The stable sort keeps classic and metal4 at indices 0 and 1.
    backends.sort(key=lambda report: report["backend"] == "mps")
    revision, _ = command(["git", "rev-parse", "HEAD"])
    status, _ = command(["git", "status", "--porcelain"])
    rustc, _ = command(["rustc", "-vV"])
    xcode, _ = command(["xcodebuild", "-version"])
    os_version, _ = command(["sw_vers"])
    report = {"schema": 2, "revision": revision, "dirty": bool(status),
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
                  "build": "fresh target directory, shared dependencies warm after the first "
                           "package; the rebuild touches the kernels' file and reuses their artifact",
                  "submit": "includes encoding; overlaps GPU work and is not additive with GPU time",
                  "wait": "host blocking after commit, not isolated queue latency",
                  "cpu": "serial reference including result allocation; null for gemm, whose "
                         "exact result is computed once per shape before the samples",
                  "gemm": "C is set to NaN before each sample, outside every timed phase, then "
                          "compared bitwise with the exact result; gemm-f32 takes f32 A and B, "
                          "gemm-f16 takes F16 A and B, and every variant returns an f32 C",
                  "repeat": "each case reports repeat, the runs per sample: 1, except gemm, which "
                            "encodes, submits and waits for that many back-to-back GEMMs in one "
                            "command buffer. Its encode, submit, wait and gpu times cover all of "
                            "them; gpu_per_gemm_ns is gpu_ns / repeat rounded to 1 ns, summarized "
                            "over the samples",
                  "mps": "MPSMatrixMultiplication cases form the backend mps, measured in the "
                         "classic run only, on their own queue and shared buffers with MPS's row "
                         "stride (padding is not checked); block is null, pipeline is the kernel "
                         "object's creation, upload is 0, and encode is MPS's own encoding where "
                         "rust and msl record only, so compare submit",
                  "percentiles": "p10 lower, p90 upper rank at fraction*(n-1); median midpoint"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()

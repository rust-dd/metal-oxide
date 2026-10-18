#!/usr/bin/env python3
"""Build and measure matching Rust/MSL compute workloads on both Metal backends."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
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
BACKENDS = ("classic", "metal4")
# A command's limit; GEMM runs scale theirs with their work at an assumed slowest GEMM rate.
TIMEOUT_S = 1800
MIN_GEMM_FLOPS = 0.5e12
# A runner process measures rust, msl and, in the classic run, mps, for f32 and f16 inputs.
RUNNER_GEMM_CASES = 6
MLX_VERSION = "0.32.4"
MLX_VENV = ROOT / "target/benchmarks/mlx-venv"
MLX_INPUTS = ("f32", "f16")
INTERLEAVE_TIMING = ("per input type, the gemm candidates of one runner process (rust and msl, "
                     "and mps in the classic process) take turns: every warmup and timed round "
                     "runs one sample of each, the first candidate rotating every round, so drift "
                     "within the process reaches them alike. classic and metal4 run in separate "
                     "processes and MLX in its own process afterwards, so MLX and cross-backend "
                     "ratios carry drift between runs: repeat runs and report the spread. rust "
                     "and msl run kernels with the same hot loops, so their ratio is the run's "
                     "drift control")
MLX_TIMING = ("host_ns is the host time to build, evaluate and synchronize `repeat` independent "
              "GPU matmuls. Unlike the runner's GEMMs, which chain through one C, they may overlap "
              "on the GPU; a dependent chain measured within 1-5 % of them. The samples run back "
              "to back, since checks between samples were observed to double the time of 1024^3 "
              "samples. MLX exposes no GPU timestamps, so compare host_per_gemm_ns with the other "
              "backends' host_per_gemm_ns, not their gpu_per_gemm_ns. Before the warmup and after "
              "the last sample, the first result is checked against the CPU stream's exact f32 "
              "product, bitwise for gemm-f32 and within f16 rounding for gemm-f16, which returns "
              "an f16 C since MLX has no f32-output GEMM for f16 inputs, and the other results "
              "against the first, bitwise")


def command(args, *, env=None, log=None, timeout=TIMEOUT_S):
    args = list(map(str, args))
    start = time.perf_counter_ns()
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout)
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


def backend_list(value):
    names = value.split(",")
    if not set(names) <= set(BACKENDS) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a list of distinct backends from {','.join(BACKENDS)}")
    return [name for name in BACKENDS if name in names]


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive, default=20)
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--elements", type=positive, default=1_000_003)
    parser.add_argument("--matrix", nargs=3, type=positive, default=[129, 131, 127],
                        metavar=("ROWS", "COLUMNS", "INNER"))
    parser.add_argument("--gemm", nargs=3, type=positive, default=[1024, 1024, 1024],
                        metavar=("M", "N", "K"),
                        help="GEMM shape, each dimension at most 8192 (default: 1024 1024 1024)")
    parser.add_argument("--gemm-repeat", type=positive, default=64, metavar="R",
                        help="GEMMs per sample, back to back in one command buffer "
                             "(default: 64, at most 256)")
    parser.add_argument("--backends", type=backend_list, default=list(BACKENDS), metavar="LIST",
                        help="comma-separated backends to measure (default: classic,metal4)")
    parser.add_argument("--mlx", action="store_true",
                        help=f"also measure the GEMM with MLX {MLX_VERSION}")
    parser.add_argument("--output", type=Path)
    return parser


def check_bounds(parser, args):
    if args.elements > 16_000_000 or max(args.matrix) > 1024:
        parser.error("elements must be <= 16000000; matrix dimensions must be <= 1024")
    if max(args.gemm) > 8192:
        parser.error("gemm dimensions must be <= 8192")
    if args.gemm_repeat > 256:
        parser.error("gemm repeat must be <= 256")


def mlx_python(venv=MLX_VENV):
    """The interpreter of a virtual environment that has the pinned MLX, created on demand."""
    python = venv / "bin/python"
    if not python.exists():
        command([sys.executable, "-m", "venv", venv])
    try:
        installed, _ = command(
            [python, "-c", "import importlib.metadata as m; print(m.version('mlx'))"])
    except RuntimeError:
        installed = None
    if installed != MLX_VERSION:
        command([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                 f"mlx=={MLX_VERSION}"])
    return python


def gemm_timeout(args, gemms):
    """Seconds for a command that runs `gemms` GEMMs of the configured shape: their work at
    MIN_GEMM_FLOPS, and never less than TIMEOUT_S."""
    m, n, k = args.gemm
    return max(TIMEOUT_S, math.ceil(2 * m * n * k * gemms / MIN_GEMM_FLOPS))


def runner_timeout(args):
    return gemm_timeout(args, args.gemm_repeat * (args.warmup + args.samples) * RUNNER_GEMM_CASES)


def mlx_timeout(args):
    # bench_mlx.py runs one checked sample per input type before its warmup.
    rounds = 1 + args.warmup + args.samples
    return gemm_timeout(args, args.gemm_repeat * rounds * len(MLX_INPUTS))


def timed_out(what, args, timeout):
    """The error for `what` running longer than `timeout` seconds."""
    m, n, k = args.gemm
    return RuntimeError(f"{what} did not finish within {timeout} s at --gemm {m} {n} {k} "
                        f"--gemm-repeat {args.gemm_repeat}; lower --gemm-repeat or the shape")


def measure_mlx(interpreter, args, run):
    """Runs scripts/bench_mlx.py on the GEMM options of `args` and returns its result."""
    print("Measuring MLX", flush=True)
    result = run / "mlx.json"
    timeout = mlx_timeout(args)
    try:
        command([interpreter, ROOT / "scripts/bench_mlx.py", "--shape", *args.gemm,
                 "--dtypes", *MLX_INPUTS, "--samples", args.samples, "--warmup", args.warmup,
                 "--repeat", args.gemm_repeat, "--output", result], log=run / "mlx.log",
                timeout=timeout)
    except subprocess.TimeoutExpired:
        raise timed_out("MLX", args, timeout) from None
    return json.loads(result.read_text())


def per_gemm_ns(sample_ns, repeat):
    """One GEMM's share of a sample, rounded half up to 1 ns like the runner's gpu_per_gemm_ns."""
    return (sample_ns + repeat // 2) // repeat


def summarize(samples):
    """The runner's Summary (benchmarks/runner/src/stats.rs) of integer samples."""
    ordered = sorted(samples)
    last = len(ordered) - 1
    lower, upper = ordered[last // 2], ordered[(last + 1) // 2]
    return {"min": ordered[0], "p10": ordered[last // 10],
            "median": lower + (upper - lower) // 2, "p90": ordered[(last * 9 + 9) // 10],
            "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def merge_mlx(report, mlx):
    """The report with the MLX result and its timing note, once the cases are known to have run
    the report's GEMM shape, repeat and sample count."""
    config = report["config"]
    expected = (config["gemm"], config["gemm_repeat"], config["samples"])
    for case in mlx["cases"]:
        measured = (case["shape"], case["repeat"], len(case["samples"]))
        if measured != expected:
            raise RuntimeError(f"mlx {case['name']} measured {measured}, not {expected}")
    return dict(report, mlx=mlx, timing=dict(report["timing"], mlx=MLX_TIMING))


def write_report(output, report, mlx=None):
    """Writes `report`, then, if `mlx` measures MLX, rewrites it with that result; an MLX failure
    leaves the written report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if mlx is None:
        return report
    try:
        report = merge_mlx(report, mlx())
    except Exception as error:
        raise RuntimeError(f"MLX failed: {error}\n{output} holds the report without MLX") from error
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def tool_output(args):
    """The standard output of `args`, or None when the tool is missing, fails or prints nothing."""
    try:
        return command(args)[0] or None
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None


def positive_integer(text):
    try:
        number = int(text)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def gpu_cores(profile):
    """The core count that `system_profiler SPDisplaysDataType -json` lists for the first GPU."""
    try:
        return positive_integer(json.loads(profile)["SPDisplaysDataType"][0]["sppci_cores"])
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def machine_identity(query=tool_output):
    """The chip, model, memory and GPU cores of this host; what it cannot report is None."""
    return {"chip": query(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "model": query(["sysctl", "-n", "hw.model"]),
            "memory_bytes": positive_integer(query(["sysctl", "-n", "hw.memsize"])),
            "gpu_cores": gpu_cores(query(["system_profiler", "SPDisplaysDataType", "-json"]))}


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("benchmarks require macOS on Apple Silicon")
    check_bounds(parser, args)
    interpreter = None
    if args.mlx:
        print(f"Preparing MLX {MLX_VERSION} in {MLX_VENV}", flush=True)
        interpreter = mlx_python()
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
    timeout = runner_timeout(args)
    for backend in args.backends:
        print(f"Measuring {backend}", flush=True)
        result = run / f"{backend}.json"
        try:
            command([host, config_path, result], env=dict(host_env, METAL_OXIDE_BACKEND=backend),
                    log=run / f"{backend}.log", timeout=timeout)
        except subprocess.TimeoutExpired:
            raise timed_out(f"the {backend} runner", args, timeout) from None
        backends.extend(json.loads(result.read_text()))
    # The stable sort moves mps after the runtime backends, which stay in their order.
    backends.sort(key=lambda report: report["backend"] == "mps")
    revision, _ = command(["git", "rev-parse", "HEAD"])
    status, _ = command(["git", "status", "--porcelain"])
    rustc, _ = command(["rustc", "-vV"])
    xcode, _ = command(["xcodebuild", "-version"])
    os_version, _ = command(["sw_vers"])
    report = {"schema": 2, "revision": revision, "dirty": bool(status),
              "timestamp_unix": time.time(), "os": os_version, "xcode": xcode,
              "host_rustc": rustc, "machine": machine_identity(),
              "host_binary_sha256": digest(host),
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
                            "command buffer, ordered by their writes of one C. Its encode, submit, "
                            "wait and gpu times cover all of them; per sample, gpu_per_gemm_ns is "
                            "gpu_ns / repeat and host_per_gemm_ns is (submit_ns + wait_ns) / "
                            "repeat, rounded to 1 ns and summarized over the samples. Compare "
                            "host_per_gemm_ns with MLX's",
                  "interleave": INTERLEAVE_TIMING,
                  "mps": "MPSMatrixMultiplication cases form the backend mps, measured in the "
                         "classic run only, on their own queue and shared buffers with MPS's row "
                         "stride (padding is not checked); block is null, pipeline is the kernel "
                         "object's creation, upload is 0, and encode is MPS's own encoding where "
                         "rust and msl record only, so compare submit. Hazard tracking of the "
                         "write to C serializes its repeats, and the first MPS case's pipeline "
                         "time includes MPS's one-time library load",
                  "percentiles": "p10 lower, p90 upper rank at fraction*(n-1); median midpoint"}}
    mlx = (lambda: measure_mlx(interpreter, args, run)) if args.mlx else None
    write_report(output, report, mlx)
    print(output, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and measure the spectral example against its native Metal reference."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from bench import ROOT, build_kernels, command, digest, positive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive, default=20)
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--max-rows", type=positive, default=16384)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = ROOT / "target/benchmarks"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="spectral-", dir=parent))
    output = args.output.resolve() if args.output else run / "report.json"
    print(f"Spectral benchmark directory: {run}", flush=True)
    environment = dict(os.environ, CARGO_TARGET_DIR=str(run / "build"))
    modes = []
    for mode, features in (("precise", []), ("fast", ["--features", "fast"])):
        print(f"Building and validating {mode}", flush=True)
        artifact, _, cold_ns, rebuild_ns = build_kernels(
            "spectral", "examples/spectral/host/src/main.rs", environment,
            run / f"{mode}-build.log", features)
        host = run / "build/release/spectral"
        results = []
        for backend in ("classic", "metal4"):
            selected = dict(environment, METAL_OXIDE_BACKEND=backend)
            command([host], env=selected, log=run / f"{mode}-{backend}-validation.log")
            result = run / f"{mode}-{backend}.json"
            print(f"Measuring {mode}, {backend}", flush=True)
            command([host, "--bench", result, args.samples, args.warmup, args.max_rows],
                    env=selected, log=run / f"{mode}-{backend}.log")
            results.append(json.loads(result.read_text()))
        modes.append({"mode": mode, "first_build_ns": cold_ns, "rebuild_ns": rebuild_ns,
                      "manifest": json.loads((artifact / "manifest.json").read_text()),
                      "host_sha256": digest(host), "backends": results})
    revision, _ = command(["git", "rev-parse", "HEAD"])
    status, _ = command(["git", "status", "--porcelain"])
    rustc, _ = command(["rustc", "-vV"])
    xcode, _ = command(["xcodebuild", "-version"])
    os_version, _ = command(["sw_vers"])
    sources = [ROOT / "examples/spectral/host" / name
               for name in ("common.metal", "fgn.metal", "sheet.metal", "euler.metal")]
    report = {"schema": 2, "revision": revision, "dirty": bool(status),
              "timestamp_unix": time.time(), "os": os_version, "xcode": xcode,
              "host_rustc": rustc, "modes": modes,
              "references": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "validation_environment": {k: v for k, v in os.environ.items()
                                         if k.startswith(("MTL_", "METAL_", "DYLD_"))},
              "timing": {"unit": "ns", "build": "fresh target, shared dependencies warm after "
                         "precise; the rebuild touches the host source and reuses the kernels",
                         "total": "encode, submit, wait and zero-copy CPU view; reusable allocations",
                         "setup": "f64 eigenvalues and bit reversal; separately measured",
                         "allocation_upload": "initialized shared storage and setup upload; separate",
                         "submit": "includes encoding; can overlap GPU execution",
                         "gpu": "native completion timestamps; null when unavailable",
                         "comparison": "paired alternating order, identical chunks, native classic setBytes",
                         "percentiles": "nearest rank at fraction*(n-1) of paired total ratios"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure the GEMM with MLX on the GPU, checked against the CPU stream's exact result."""
import argparse
import json
from pathlib import Path
import platform
import time

import mlx.core as mx

from bench import per_gemm_ns, positive, summarize

INPUTS = {"f32": mx.float32, "f16": mx.float16}
NAMES = {dtype: name for name, dtype in INPUTS.items()}


def operands(m, n, k, dtype):
    """A and B of benchmarks/runner/src/gpu/gemm.rs in `dtype`, built on the CPU stream."""
    with mx.stream(mx.cpu):
        a = (mx.arange(m * k) % 17 - 8).astype(mx.float32) * 0.125
        b = (mx.arange(k * n) % 13 - 6).astype(mx.float32) * 0.25
        a, b = a.reshape(m, k).astype(dtype), b.reshape(k, n).astype(dtype)
        mx.eval(a, b)
    return a, b


def exact_product(a, b):
    """A * B on the CPU stream; multiples of 1/8 and 1/4 keep every partial sum exact in f32."""
    with mx.stream(mx.cpu):
        c = mx.matmul(a, b)
        mx.eval(c)
    return c


def bits(values):
    return values.view(mx.uint32 if values.dtype == mx.float32 else mx.uint16)


def wrong(c, exact):
    """True where f32 C differs in any bit, or f16 C is farther than f16 rounding, from exact."""
    if c.dtype == mx.float32:
        return bits(c) != bits(exact)
    rounding = mx.abs(exact.astype(c.dtype).astype(mx.float32) - exact)
    return mx.logical_not(mx.abs(c.astype(mx.float32) - exact) <= rounding)


def check(name, results, exact):
    """Raises unless the first result is the exact product and the others equal it bitwise."""
    first = results[0]
    with mx.stream(mx.cpu):
        bad = wrong(first, exact).reshape(-1)
        if mx.any(bad).item():
            row, column = divmod(mx.argmax(bad).item(), first.shape[1])
            raise RuntimeError(f"{name}: C[{row}][{column}] is {first[row, column].item()} "
                               f"instead of {exact[row, column].item()}")
        for index, other in enumerate(results[1:], 1):
            if not mx.array_equal(bits(other), bits(first)).item():
                raise RuntimeError(f"{name}: result {index} differs from result 0")


def run_sample(a, b, repeat):
    """The host ns to build, evaluate and synchronize `repeat` independent GPU matmuls, and
    their results."""
    start = time.perf_counter_ns()
    results = [mx.matmul(a, b, stream=mx.gpu) for _ in range(repeat)]
    mx.eval(results)
    mx.synchronize(mx.gpu)
    return time.perf_counter_ns() - start, results


def measure(token, a, b, exact, args):
    name = f"gemm-{token}"
    _, results = run_sample(a, b, args.repeat)
    check(name, results, exact)
    times = []
    # Checks between samples were observed to double the time of 1024^3 samples.
    for iteration in range(args.warmup + args.samples):
        results = None  # frees the previous buffers before the clock starts
        elapsed, results = run_sample(a, b, args.repeat)
        if iteration >= args.warmup:
            times.append(elapsed)
    check(name, results, exact)
    samples = [{"host_ns": ns, "host_per_gemm_ns": per_gemm_ns(ns, args.repeat)} for ns in times]
    return {"name": name, "variant": "mlx", "shape": args.shape, "repeat": args.repeat,
            "input": token, "output": NAMES[results[0].dtype], "samples": samples,
            "summary_ns": {key: summarize([sample[key] for sample in samples])
                           for key in samples[0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=positive, required=True, metavar=("M", "N", "K"))
    parser.add_argument("--dtypes", nargs="+", choices=list(INPUTS), required=True)
    parser.add_argument("--samples", type=positive, required=True)
    parser.add_argument("--warmup", type=positive, required=True)
    parser.add_argument("--repeat", type=positive, required=True, metavar="R")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exact = exact_product(*operands(*args.shape, mx.float32))
    cases = [measure(token, *operands(*args.shape, INPUTS[token]), exact, args)
             for token in args.dtypes]
    report = {"version": mx.__version__, "python": platform.python_version(),
              "device": mx.device_info(mx.gpu).get("device_name"), "cases": cases}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

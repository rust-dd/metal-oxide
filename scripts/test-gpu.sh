#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! cargo nextest --version >/dev/null 2>&1; then
    printf 'The GPU tests run with cargo-nextest: cargo install cargo-nextest --locked\n' >&2
    exit 1
fi

# The gpu profile runs the ignored hardware tests one at a time, numbering each finished one.
nextest=(nextest run --locked --profile gpu --run-ignored only)
cargo "${nextest[@]}" -p metal-oxide-build

# Arguments name the native APIs to test. By default Metal 4 runs from macOS 26 on;
# virtual GPUs without the Metal 4 family pass `classic`.
backends="$*"
compiler_filter=(-E 'all()')
if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 27 ]; then
    compiler_filter=(-E 'not test(msl41_)')
    printf 'MSL 4.1 acquire/release hardware tests require macOS 27.\n'
fi
if [ -z "$backends" ]; then
    backends="classic"
    if [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ]; then
        backends="classic metal4"
    fi
fi
for backend in $backends; do
    export METAL_OXIDE_BACKEND="$backend"
    printf 'Testing %s on %s\n' "$backend" "$(uname -m)"
    cargo "${nextest[@]}" -p metal-oxide -p metal-oxide-kernels -p cargo-metal
    (
        cd crates/metal-oxide-compiler
        cargo "${nextest[@]}" --features rustc-private --target-dir ../../target/compiler \
            --test gpu --test differential --test nostd --test generated_numeric \
            --test vector --test generated_vectors "${compiler_filter[@]}"
    )
    for package in cooperation vec-add saxpy reduction matmul gemm pipeline particle-update sort fft stencil; do
        cargo run -p cargo-metal --locked -- test --validate -p "$package" --locked -- --ignored --test-threads=1
    done
    for features in "" fast; do
        cargo run -p cargo-metal --locked -- test --validate -p spectral --release --locked ${features:+--features "$features"} -- --ignored --test-threads=1
    done
done

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo test -p metal-oxide-build --locked -- --ignored --test-threads=1

# Arguments name the native APIs to test. By default Metal 4 runs from macOS 26 on;
# virtual GPUs without the Metal 4 family pass `classic`.
backends="$*"
gpu_test_args=(--ignored --test-threads=1)
if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 27 ]; then
    gpu_test_args+=(--skip msl41_)
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
    cargo test -p metal-oxide --locked -- --ignored --test-threads=1
    cargo test -p metal-oxide-kernels --locked -- --ignored --test-threads=1
    cargo test -p cargo-metal --locked -- --ignored --test-threads=1
    (
        cd crates/metal-oxide-compiler
        cargo test --features rustc-private --locked --target-dir ../../target/compiler \
            --test gpu --test differential --test nostd --test generated_numeric \
            --test vector --test generated_vectors -- "${gpu_test_args[@]}"
    )
    for package in cooperation vec-add saxpy reduction matmul pipeline particle-update; do
        cargo run -p cargo-metal --locked -- test --validate -p "$package" --locked -- --ignored --test-threads=1
    done
    for features in "" fast; do
        cargo run -p cargo-metal --locked -- test --validate -p spectral --release --locked ${features:+--features "$features"} -- --ignored --test-threads=1
    done
done

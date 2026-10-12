#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo test -p cargo-metal --locked -- --ignored --test-threads=1 --skip numerical_modes_isolate_artifacts_and_match_source_compilation --skip embedded_artifacts_load_lazily_in_a_relocated_stable_host
cargo test -p metal-oxide-build --locked -- --ignored --test-threads=1

# Metal 4 needs macOS 26; earlier systems run the classic path only.
backends="classic"
if [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ]; then
    backends="classic metal4"
fi
for backend in $backends; do
    export METAL_OXIDE_BACKEND="$backend"
    printf 'Testing %s on %s\n' "$backend" "$(uname -m)"
    cargo test -p cargo-metal --locked --test math_mode -- --ignored --test-threads=1
    cargo test -p cargo-metal --locked --test embedded -- --ignored --test-threads=1
    cargo test -p metal-oxide --locked -- --ignored --test-threads=1
    (
        cd crates/metal-oxide-compiler
        cargo test --features rustc-private --locked --target-dir ../../target/compiler --test gpu --test differential -- --ignored --test-threads=1
    )
    for package in cooperation vec-add reduction matmul pipeline particle-update; do
        cargo run -p cargo-metal --locked -- test -p "$package"
    done
    for mode in precise fast; do
        cargo run -p cargo-metal --locked -- test -p spectral --release --math-mode "$mode"
    done
done

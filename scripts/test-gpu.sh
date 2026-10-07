#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo test -p cargo-metal --locked --test build -- --ignored --test-threads=1

for backend in classic metal4; do
    export METAL_OXIDE_BACKEND="$backend"
    printf 'Testing %s on %s\n' "$backend" "$(uname -m)"
    cargo test -p metal-oxide --locked -- --ignored --test-threads=1
    (
        cd crates/metal-oxide-compiler
        cargo test --features rustc-private --locked --target-dir ../../target/compiler --test gpu -- --ignored --test-threads=1
    )
    for package in vec-add reduction matmul pipeline particle-update; do
        cargo run -p cargo-metal --locked -- test -p "$package"
    done
done

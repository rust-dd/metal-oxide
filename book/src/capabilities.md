# Capability requirements

This table is generated from `metal-oxide-artifact::CAPABILITIES`, the same
requirements used by artifact validation and runtime loading. Regenerate both the
README and this chapter from the repository root:

```sh
cargo run -p metal-oxide-artifact --example capabilities --locked
cargo run -p metal-oxide-artifact --example capabilities --locked -- --check
```

<!-- capabilities -->
| Artifact feature | MSL | macOS | Apple GPU family | Mac2 |
| --- | --- | --- | --- | --- |
| `float32_atomics` | 3.0 | 13 | 7 | no |
| `uint64_atomics` | 2.4 | 12 | 8 | no |
| `atomic_acquire_release` | 4.1 | 27 | 7 | no |
| `int32_atomics` | 1.0 | 11 | 1 | yes |
| `simd_groups` | 2.0 | 11 | 7 | yes |
| `simdgroup_matrix` | 2.3 | 11 | 7 | no |
| `dynamic_threadgroup_memory` | 1.0 | 11 | 1 | yes |
| `logging` | 3.2 | 15 | 6 | yes |
| `bfloat16` | 3.1 | 14 | 6 | no |
<!-- /capabilities -->

The compiler baseline is MSL 3.2 and macOS 15 even when a feature's minimum is
lower. The runtime accepts the listed Apple GPU family or, where marked yes,
Mac2 support. Requirements follow Apple's
[Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
and [feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).
`uint64_atomics` covers the supported min/max operations only. Acquire/release
gates are tested, but execution on the required OS has not been validated.

These are eligibility requirements. See the [execution evidence](index.md#execution-evidence)
for the actual tested hardware and [supported Rust](supported-rust.md) for accepted
language constructs and operation restrictions.

# Compiler internals

```text
Rust #[kernel] / #[kernels]
          |
   rustc typed MIR
          |
   metal-oxide typed IR
          |             \
          MSL           ABI and generated launch functions
          |                       |
   Apple Metal compiler           |
          |                       |
      .metallib --------> stable Rust host runtime
```

The kernel marker does not translate Rust bodies. The pinned `rustc_driver`
frontend imports concrete entrypoints and reachable monomorphized helpers, with
device-target `core` metadata. Typed IR keeps layouts, source locations, reference
address spaces and access modes independent of rustc internals. Structured lowering
preserves checks and rejects unsupported operations explicitly.

The compiler proves checks where possible; remaining fallible operations report
through a dispatch failure word. Generated kernels also have a gate that skips
effects after an upstream failure. Participation analysis checks barriers and SIMD
operations transitively. MSL `#line` directives preserve Rust diagnostics and debug
locations. Apple's compiler produces AIR and the final library; direct AIR
generation remains deferred.

## Host and artifact boundary

The host macros generate launch functions from the kernel signature, safety,
contracts and block requirements. `metal-oxide-build` drives the compiler and
stores the compiled library with its versioned ABI and input manifest. Source,
dependencies, compiler/toolchain/SDK settings and numerical mode participate in
artifact identity. Version and ABI checks remain enabled during development;
`ABI_VERSION` is 4 until the first publication.

Inline modules share host/device types and generate codecs for shared records.
Separate `no_std` kernel crates can share helpers and dependencies without putting
rustc internals in the stable runtime dependency graph. The runtime validates
artifact requirements, bindings, contracts and shape before encoding. It selects
classic Metal or Metal 4 internally from OS/device capabilities, with separate
synchronization and residency implementations behind the same public API.

## Development verification

Portable workspace tests cover layouts, artifact validation and runtime contracts.
Compiler tests run from `crates/metal-oxide-compiler` using its pinned nightly and
`--features rustc-private`. They test source diagnostics, IR/MSL and host-executed
MSL against a test-only shim. Explicit hardware suites execute the emitted shaders
and compare with CPU and MSL references; compilation alone is not execution
evidence. The [contributing guide](https://github.com/rust-dd/metal-oxide/blob/main/CONTRIBUTING.md)
lists commands, and [CONCEPT.md](https://github.com/rust-dd/metal-oxide/blob/main/CONCEPT.md)
describes the implementation in depth.

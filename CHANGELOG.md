# Changelog

## Unreleased

### Added

- Kernels call math with Rust's method syntax, such as `x.sqrt()`, `a.mul_add(b, c)` and
  `y.atan2(x)`, through the device `Float` trait for `f32`, `F16` and `BF16`. New methods:
  `exp2`, `log2`, `log10`, `tan`, `asin`, `acos`, `atan2`, `hypot`, `rsqrt` and `powi`.
  `F16` and `BF16` also have `abs`, `min` and `max`.
- `Abi::minimum_macos` returns the oldest macOS that runs an artifact's kernels.
- `i64` works in kernels, kernel parameters, buffers, records and `log!`, so core methods
  that widen through it, such as `i32::midpoint`, compile. It has no atomics.
- Kernels call integer `ilog`, `ilog2`, `ilog10` and signed `isqrt`, which fail with Rust's
  message on invalid arguments; `to_le_bytes`, `to_be_bytes`, `to_ne_bytes` and their
  `from_*` counterparts; `mem::swap`, `ptr::swap`, `ptr::copy_nonoverlapping`, and slice
  `swap`, `reverse` and `copy_from_slice`. `==` and `starts_with` work on arrays and
  slices of integers: core's bytewise comparisons run element by element.
- Slice patterns, rest bindings included, and `first`, `last`, `split_first` and
  `split_last` on slices. The prover checks each pattern's length test.
- Arrays iterate by value: `for v in array`, and `into_iter` with `next` and adapters that
  call it. Folds of the array iterator, behind `sum` and `count`, are rejected.
- `<[T; N]>::map` and `array::from_fn` build arrays from closures.
  Transmutes between an integer and an array of integers as wide are supported.
- The book gains troubleshooting and organizing-kernels chapters, the API reference
  documents every public item of `metal-oxide` and `metal-oxide-device`, and a hardware
  report form asks for `cargo metal setup --check` and the generated MSL.
- `buffer.copy_from(&source)` and `buffer.fill(value)` copy and fill buffers on the GPU as
  lazy operations that follow earlier work like a launch, through built-in kernels that
  match a blit encoder's copy bandwidth on an M4 Max.
- `buffer.slice(range)` and `buffer.slice_mut(range)` pass part of a buffer to a launch at an
  offset. The kernel, `requires` and `len()` see only those elements; dependency tracking
  covers the whole buffer.
- Kernel modules, inline or with a `path`, define `KERNELS` and `pipeline(&device, name)`,
  which returns a kernel's pipeline from the device's cache for code that encodes its own
  dispatches.
- `Device::native_api` reports whether a context uses classic Metal or Metal 4, and
  `cargo metal setup --check` prints it.


### Changed

- Kernel build errors point at the kernel's own code. A rejection inside core or another
  crate reports at the kernel call that reached it, with a note naming the code it
  reached, which kernels in host crates show too; proof failures in called functions do
  likewise. `log!` errors point at the `log!` call. Unsupported operations are described
  instead of printed as MIR, such as calls that never return, tail calls and
  pointer-integer casts.
- The device `math` module is removed; its functions are `Float` methods with std's names,
  and `math::fma(a, b, c)` is `a.mul_add(b, c)`.
- Saved artifacts change only with their kernels or recorded inputs. Another compiler
  build, machine or Metal toolchain that produces the same kernels leaves `metal/`
  unchanged. Manifests name the metal-oxide compiler version instead of a compiler
  binary hash, and recorded path packages no longer end in a space.
- Kernel items inside inline modules save their artifacts below the module names, such
  as `metal/src/lib.rs/tests/fill-precise/`, so same-named items in one file no longer
  overwrite each other's artifacts.
- `Device::system_default` returns the default GPU's context while any clone of it, a buffer
  or a pipeline of it lives, and opens a new one only after they all dropped. Buffers keep
  their context alive. `Device::all` and `Device::from_native` still open independent ones.
- Launch errors name their kernel, shapes, limits and types: `InvalidLaunch` and
  `InvalidBuffer` carry a `String`, argument count and type mismatches are `InvalidLaunch`
  instead of `Artifact`, an invalid `METAL_OXIDE_BACKEND` is `Error::Environment`, and I/O
  errors of `Module::from_artifact` and `from_metallib` name the file.
- `Argument::read`, `write`, `read_write` and `atomic` take any launch argument by
  reference: a buffer, an `Arc`, or a slice.


### Fixed

- `Result::unwrap_unchecked` compiles: its `Err` arm, which runs `mem::forget` before the
  unreachable hint, is pruned like `Option`'s `None` arm.
- Repeated integer casts retain proven slice bounds, including before threadgroup
  barriers. Changed indices and different casts keep their runtime checks.
- Without the Metal compiler, kernel builds use a current saved artifact instead of
  failing after the Rust build. Reused builds rebuild after a Metal compiler or SDK
  change, and hosts track the toolchain's version files, so Cargo expands their kernels
  again after an Xcode or Metal Toolchain update.
- `dependencies(...)` crates build with the features and `default-features` setting that
  the kernels' package declares, and renamed dependencies resolve to their package.
  They previously built with default features only.
- Metal libraries target the minimum macOS that their kernels need. Linking omitted the
  deployment target, so libraries built with the macOS 26 SDK used AIR 2.8 and macOS 15
  refused to load them, saved artifacts included. The next build rewrites their
  `kernels.metallib`.
- Values of 3, 5, 6 or 7 bytes, such as a `[bool; 3]` that a closure captures, are copied
  member by member. LLVM copied a 3-byte value as a 24-bit integer, which Apple's GPU
  compilers misread, so `any` over such a capture could return `false` on the GPU.
- Folds nested in closures, such as `take(..).fold(..)` inside another fold, no longer
  exhaust the numerical proof budget. Loops keep the facts of their first 8 iterations
  apart before merging them, instead of 64, so calls in loop bodies are analyzed fewer
  times.
- Signed 16- and 32-bit `saturating_sub` and `saturating_neg` of constants, such as
  `1_i16.saturating_neg()`, return their negative results. Apple's GPU compilers folded
  them to zero where the difference was negative.
- Loops with an early `return` and several exits that continue at the same code compile.
  That code was emitted once per exit, so programs with a few such loops were rejected with
  "control flow requires excessive or cooperative block duplication".
- Checked `i8` additions and subtractions, including `+` and `-` with overflow checks,
  compile for every Apple GPU. LLVM could turn their overflow test into
  `llvm.sadd.with.overflow.i8` or `llvm.ssub.with.overflow.i8`, which Apple's GPU compilers
  cannot lower, so creating the pipeline failed.
- Float-to-integer `as` casts convert without branches and keep Rust's saturation and NaN
  rules. M1 and M2 GPU compilers crashed on kernels that converted one `f32` to `u64` twice,
  such as `x as u64` next to `1 + x as u64`.
- Shader `log!` messages no longer abort the process when stderr is closed, such as a pipe
  whose reader exited. Metal's logging thread printed them with `eprintln!`, which panics when
  writing fails.
- An array indexed past its end by a constant compiles to the bounds check that fails at run
  time, as in Rust. It was rejected as "invalid aggregate field, index or dereference" even
  where the index never runs.

### Upgrading

- Saved inputs use format 4. Build once with the kernel toolchain and commit the
  regenerated `metal/` directories; locked and toolchain-less builds report format 3
  artifacts as unsupported until then.
- `metal_oxide_build::KernelCrate` takes the module's `location` instead of its `file`.
- Replace `math::sqrt(x)` with `x.sqrt()`, `math::fma(a, b, c)` with `a.mul_add(b, c)` and
  the other functions likewise. Kernel crates import `metal_oxide_device::Float`; kernels in
  host crates need no import.
- `cargo semver-checks` against `0.1.0-alpha.1` finds no breaking change in `metal-oxide`
  and `metal-oxide-codegen`. Besides the two changes above, `Scalar` of `metal-oxide-artifact`
  and `metal-oxide-ir` gains `I64`; `metal-oxide-ir` also gains `Constant::I64`, math
  operations and slice-pattern projections, and its `Error` a `calls` field. Exhaustive matches
  on these enums and `Error` literals need the new cases.
- Kernel modules no longer define `module(&device)`; replace
  `Pipeline::new(&device, &kernels::module(&device)?, name)` with
  `kernels::pipeline(&device, name)`.
- `ReadArgument::buffer` and `WriteArgument::buffer_mut` are `view` and `view_mut`, which
  return the bound elements; `Argument::read(input.buffer())` becomes `Argument::read(input)`.
- Code that matched `Error::InvalidLaunch` or `InvalidBuffer` messages as `&str` compares
  `String`s, and code that expected `Error::Artifact` for wrong launch arguments expects
  `Error::InvalidLaunch`.

## 0.1.0-alpha.1

Prereleases use dotted identifiers from `0.1.0-alpha.1` on. SemVer orders
`0.1.0-alpha.1` below `0.1.0-alpha0`, so `0.1.0-alpha0` is yanked once
`0.1.0-alpha.1` is published.

### Added

- Kernels support integer powers, checked division and remainder, comparisons and
  byte swaps, integer and float clamps and signs, and `f32::copysign` and `total_cmp`.
- `Device`, `Buffer`, `Pipeline`, `Module`, `Launch` and `Batch` implement `Debug`
  without reading buffer contents or waiting for GPU work.
- The examples index lists the command for each runnable example; crate metadata
  links to the hosted book.

### Changed

- `Error` and `KernelFailure` are non-exhaustive. `Result<T, E = Error>` also accepts
  a custom error type.
- BF16 buffers expose direct shared-storage CPU views, avoiding mirror copies.
- Generated-program tests compare both halves of `u64` results and detect missing
  writes and overwritten output padding. F16 graphs run independently of BF16;
  selecting a BF16 test on an unsupported device fails instead of passing.

### Fixed

- Generated MSL omits stores that nothing reads and keeps their checks. Kernels with
  such stores failed with "Compilation failed" in the macOS 15 Metal compiler.
- A missing or mismatched kernel compiler reports the matching `cargo-metal`
  install command.
- Kernel failures retain the original dispatch push index when earlier dispatches
  have empty grids.
- The book and README explain release-profile overflow checks and the scope of
  threadgroup barriers. CI verifies nextest downloads and enforces its version.

### Upgrading from 0.1.0-alpha0

1. Install the matching tools with
   `cargo install cargo-metal --version '=0.1.0-alpha.1' --locked`, then run
   `cargo metal setup`.
2. Change every metal-oxide dependency to `=0.1.0-alpha.1`.
3. Build once with the kernel toolchain and commit the regenerated `metal/`
   directories. Saved artifacts record the locked package versions, so locked and
   toolchain-less builds of your own crates report them as stale until then.
4. Add a wildcard arm to exhaustive `Error` matches, and `..` to `KernelFailure`
   patterns. Constructing `KernelFailure` outside the runtime is no longer supported.
   Compiler integrations matching `metal-oxide-ir` operation enums must handle
   their new comparison, byte-swap and copy-sign variants.

Artifacts retain ABI 4 and saved alpha0 dependencies remain loadable. The hardware
coverage and limitations recorded for alpha0 still apply; the two-week scheduled
fuzz observation period remains outstanding.

## 0.1.0-alpha0

### Recorded hardware coverage

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal and Metal 4 |
| Apple Paravirtual device in GitHub-hosted M1 VMs | 15.7 and 26.6 | 16.4 and 26.6 | classic Metal |

The host supports Rust 1.89 and later. Development uses Rust 1.99; kernel authoring
uses nightly-2026-10-04 with rustc-dev, rust-src and llvm-tools. Saved artifacts
allow hosts and dependents to build without the kernel compiler. Artifacts use
ABI 4; incompatible contract or format changes after this release increment it.

### Known limitations

This is an early alpha. The API can change between releases.

Native M1–M3 GPUs and another Metal 4 GPU generation have not been verified;
Paravirtual execution does not establish native M1 coverage. Acquire/release
atomics require macOS 27/MSL 4.1 and still await hardware validation.
Automated rust-analyzer checks pass; editor rendering and Rust source display
in Xcode remain unverified. Two weeks of green scheduled fuzz runs have not yet
accumulated; that observation continues after this alpha.

Rust tensor primitives remain deferred after the MPP comparison. Unsupported
Rust operations, including recursion, trait-object calls, destructors, `f64`,
`i64` and 128-bit integers, are rejected explicitly. Fast math permits
reassociation and does not promise identical floating-point bits across compiler
versions. See [supported Rust](book/src/supported-rust.md),
[numerical modes](book/src/performance.md#numerical-modes) and
[configuration limitations](book/src/installation.md).

### Working features

- Ordinary Rust kernels in host functions, shared modules or separate `no_std`
  crates, compiled from rustc MIR through typed IR to Metal Shading Language.
- CUDA-style thread and block coordinates, static and dynamic block shapes,
  shared memory, SIMD groups, matrices and explicitly instantiated generic kernels.
- Typed buffers, checked launch contracts, GPU assertion reporting, async launches,
  operation chains and multiple launches in one command buffer.
- One runtime API for classic Metal and Metal 4, with device capability checks,
  native buffer ownership, validation, source inspection and GPU capture.
- Precise and fast numerical modes, explicit conversions, narrow integers,
  half types, supported atomics and small `core::simd` vectors.
- An authoring book, generated capability reference, packaged-installation checks,
  CPU/MSL/GPU differential tests and interleaved benchmarks.

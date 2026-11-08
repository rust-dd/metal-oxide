# Troubleshooting

Start with `cargo metal setup --check`. It reports macOS, both Rust toolchains, the
compiler components, the metal-oxide compiler, Xcode, the macOS SDK, the Metal compiler
and the Metal device, and marks what is missing.

## The Metal compiler is missing

Kernel builds compile MSL with Apple's Metal compiler, which comes with Xcode; the
Command Line Tools alone have no Metal compiler, so `xcrun metal` fails while they are
the selected developer directory. Install Xcode 16 or later, select it and run setup
again:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
cargo metal setup
```

When Xcode has no Metal compiler, setup downloads Apple's Metal Toolchain with
`xcodebuild -downloadComponent MetalToolchain`. Without a Metal compiler, a kernel build
uses the package's saved artifact and fails only when that artifact is stale or missing.

## Pipeline creation fails

Creating a pipeline compiles the kernel library for the GPU at hand, and Apple's GPU
compiler can fail on a kernel for one GPU family that it compiles for another. The
failure is an `Error::Pipeline` with a message such as `Compilation failed` or
`XPC_ERROR_CONNECTION_INTERRUPTED`. The compiler avoids the failures it knows of, and
the [verified configurations](index.md) list the GPUs that the test suites ran on. Report
a pipeline failure with the details below.

## The native API

On macOS 26 with a GPU of the Metal 4 family, devices use Metal 4; otherwise they use
classic Metal. `METAL_OXIDE_BACKEND=classic` selects classic Metal, which shows whether
a failure is specific to Metal 4. `METAL_OXIDE_BACKEND=metal4` fails with
`Error::UnsupportedDevice` without Metal 4, and other values fail with `Error::Command`.
`Device::native_api()` and `cargo metal setup --check` report the API in use.

## A kernel failed

`Error::KernelFailed` reports a check that failed in a kernel thread, like a Rust panic:
the kernel, the position of its dispatch in the command buffer, the Rust location and the
message. Kernels check indexing that the compiler could not prove in bounds, division by
zero, literal panics and failed `unwrap` or `expect`. They check arithmetic overflow only
with `overflow-checks = true` in `[profile.release]`, because kernels build with the
release profile. The dispatches of the same command buffer that wait for a failed one
are skipped, and later operations that read its writes fail with
`Error::DependencyFailed`.

`cargo metal inspect --emit msl` shows the checks that a kernel performs at run time:
each is a `metal_oxide_fail` call. See [debugging](debugging.md) for validation and
capture.

## Report a hardware problem

Open a [hardware report](https://github.com/rust-dd/metal-oxide/issues/new?template=hardware-report.yml)
with the output of `cargo metal setup --check`, whether the failure persists with
`METAL_OXIDE_BACKEND=classic`, and the generated MSL of the failing kernel:

```sh
cargo metal inspect --emit msl -p PACKAGE --kernel KERNEL
```

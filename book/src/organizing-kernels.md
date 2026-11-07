# Organizing kernels

Kernels take three forms. Each builds with the kernel toolchain during `cargo build`, saves
its artifact in `metal/` beside the host's sources and becomes launch functions in the host
crate.

## Kernel functions

A `#[metal_oxide::kernel]` function compiles on its own, as in [your first
kernel](first-kernel.md). It sees the device prelude, `core`, the items it defines and
`no_std` crates by their full paths.

## Kernel modules

`#[metal_oxide::kernels]` compiles the `#[kernel]` functions of an inline module as one
library. The module's other items stay on the host as well, so kernels and host code share
helpers, and the `#[repr(C)]` records that kernels take become `GpuValue`s. `math = "fast"`
relaxes floating-point math for the module.

Kernels that call another `no_std` crate name it in `dependencies(...)`. The host package
depends on that crate as usual, and the kernel build uses the name, features and
`default-features` setting that the host's manifest gives it:

```rust,ignore
#[metal_oxide::kernels(dependencies(libm))]
mod kernels {
    use metal_oxide::device::{DisjointBuffer, thread_index};

    #[kernel]
    pub fn sines(mut out: DisjointBuffer<f32>) {
        let index = thread_index();
        if let Some(out) = out.get_mut(&index) {
            *out = libm::sinf(index.get() as f32);
        }
    }
}
```

## Kernel crates

A separate `no_std` library keeps kernels and their helpers out of the host crate. It
depends on `metal-oxide-device` at exactly the version of `metal-oxide` and `cargo-metal`,
and its kernels import `kernel` from it:

```sh
cargo new --lib kernels
cargo add --manifest-path kernels/Cargo.toml metal-oxide-device@=0.1.0-alpha.1
```

```rust,ignore
#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

/// Adds two buffers of equal length element by element.
#[kernel(requires(a.len() == out.len(), b.len() == out.len()))]
pub fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get();
        *out = a.as_slice()[i] + b.as_slice()[i];
    }
}
```

The host package does not depend on the kernel crate. An empty module names its directory,
relative to the host package, and gets a launch function for each kernel and a host struct
for each record that the kernels take:

```rust,ignore
#[metal_oxide::kernels(path = "../kernels")]
mod kernels {}
```

The kernel crate's dependencies build for the GPU with it. Build scripts and proc macros
other than `#[kernel]` are unsupported there. The cooperation, matmul, gemm and spectral
examples use kernel crates.

## Publishing

Commit each `metal/` directory and include it in published packages. A package fetched from
a registry builds from its saved artifacts, so its dependents build without the kernel
toolchain.

# Installation

GPU execution requires macOS 15 or later on Apple Silicon. The host runtime supports
Rust 1.89 or later; this checkout pins Rust 1.99 for development. Editing kernels
also requires Xcode with its Metal Toolchain and the compiler's pinned nightly,
including `rustc-dev`, `rust-src` and `llvm-tools`. Install Xcode and select its
developer directory before setup.

## Install the alpha

In an existing Rust application, install the Cargo command and matching runtime:

```sh
cargo install cargo-metal --version '=0.1.0-alpha0' --locked
cargo metal setup
cargo add metal-oxide@=0.1.0-alpha0
```

Specify the prerelease version explicitly: plain `cargo install cargo-metal`
skips alpha versions. Keep the runtime and Cargo command at the same version.
Copy the tested source from [your first kernel](first-kernel.md) into
`src/main.rs`, then run `cargo run`. No application build script is required.

Setup installs the pinned compiler toolchain and downloads Apple's Metal Toolchain
if missing, then checks the tools and reports the Metal device.

## Run a development checkout

With Rust and Git installed, run:

```sh
git clone https://github.com/rust-dd/metal-oxide.git
cd metal-oxide
cargo run -p cargo-metal --locked -- setup
cargo run -p vec-add --locked
```

The vec-add example verifies several sizes against CPU results, including zero
and partial blocks. To use the released source, check out `v0.1.0-alpha0` before
running setup and the example.

Install the Cargo command from the same checkout to use it in other applications:

```sh
cargo install --locked --path crates/cargo-metal
cargo metal setup
cargo metal setup --check
```

## Use a development dependency

In an existing Cargo application, choose one dependency source:

```sh
cargo add metal-oxide --path /absolute/path/to/metal-oxide/crates/metal-oxide
```

Or use the repository directly:

```sh
cargo add metal-oxide --git https://github.com/rust-dd/metal-oxide.git
```

Keep `cargo-metal` and the dependency at the same repository revision. Copy the
tested source from [your first kernel](first-kernel.md) into `src/main.rs` and use
plain `cargo run`. No application build script is required. For a reproducible git
dependency, pin its revision and commit `Cargo.lock`.

## Saved artifacts

Each kernel build saves `metal/<file>/<item>-<math mode>/` beside the application's
sources. Commit that directory and include it when distributing a kernel crate.
An unchanged saved artifact allows dependents and hosts without the kernel
toolchain to build. Editing kernels or their dependencies can make it stale and
requires the kernel toolchain to regenerate it.

```sh
METAL_OXIDE_ARTIFACTS=locked cargo build --locked
```

This mode requires current saved artifacts and fails instead of rebuilding stale
ones. Saved libraries still validate their version, ABI and device requirements.

Release-profile environment settings and existing Cargo configuration files are
tracked when building kernels. After adding a previously absent `.cargo/config`
or `.cargo/config.toml` that changes only the release profile, touch a kernel
source file or run `cargo clean` so Cargo expands the kernel macro again.
Cargo configuration `include` directives are unsupported for kernel builds; put
kernel settings directly in a discovered configuration file.

Kernel builds track ambient values read by `env!` and `option_env!`, including
values supplied by a host build script. Cargo environment defaults retain ambient
override tracking; forced values are validated through the configuration. Build
scripts in device crates and their dependencies remain unsupported. Fetched
packages use their published saved kernels.

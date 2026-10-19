# Your first kernel

This is the complete vec-add executable:

```rust,ignore
{{#include ../../examples/vec-add/src/main.rs}}
```

The `#[kernel]` macro marks a Rust entrypoint for the compiler and produces a host
launch function of the same name. Its arguments begin with the device and launch
configuration, followed by the kernel's arguments. `requires` relations are
checked before encoding; a violation returns `Error::Contract`.

`ReadBuffer` exposes a checked shared slice. `DisjointBuffer::get_mut` uses a
`thread_index()` witness to give each thread exclusive access to its own element.
Extra threads get `None`. Here the length contracts prove the input indexing safe.

The host allocates initialized buffers, constructs a configuration, waits for the
launch and reads the output. Import `Operation` to use `.wait()`. The hardware
tests run only when explicitly requested; ordinary workspace tests do not run them.

For an in-place update, the tested SAXPY example uses the same ownership rules:

```rust,ignore
{{#include ../../examples/saxpy/src/main.rs}}
```

Run it from the checkout with `cargo run -p saxpy --locked`. For larger programs,
`#[metal_oxide::kernels]` shares types and helpers in an inline module;
`#[metal_oxide::kernels(path = "../kernels")]` imports a separate `no_std` kernel
crate. See [compiler internals](compiler.md) for their build boundary.

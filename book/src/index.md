# Rust kernels for Metal

metal-oxide compiles ordinary Rust functions into Metal compute kernels. The
host uses stable Rust, typed buffers and lazy launch operations. Kernel compilation
uses a separate pinned nightly and Apple's Metal compiler.

Start with [installation](installation.md) and the [first kernel](first-kernel.md).
The examples in this book are included from the repository's executable examples;
their hardware tests are explicit, separate from portable tests.

There is no published release yet. Use a checkout, path dependency or git dependency.
The API and development artifact format can change before publication.

## Execution evidence

The full hardware suite has execution evidence on these devices only:

| GPU | macOS | Xcode | Native APIs |
| --- | --- | --- | --- |
| M4 Max | 26.2 | 26.6 | classic Metal and Metal 4 |
| Apple Paravirtual device in GitHub-hosted M1 VMs | 15.7 and 26.6 | 16.4 and 26.6 | classic Metal |

Capability checks describe requirements, not proof of execution on every GPU
family. The Paravirtual device lacks the Metal 4 family and Apple7 matrix support.
Acquire/release atomics have compiler and capability-gate tests, but hardware
validation is pending. Metal Performance Primitives (MPP) is a benchmark reference;
Rust kernels expose SIMD-group matrices and do not expose a tensor API.

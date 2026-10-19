# Launch geometry

`LaunchConfig<X, Y, Z>` stores the threads per block in its type; omitted axes are
one. Its runtime `grid: Dim3` counts blocks. Each block is a Metal threadgroup.
`LaunchConfig::<256>::for_elements(257)` launches two blocks of 256 threads.
The kernel must guard the 255 padding threads.

`DynamicLaunchConfig` carries both grid and block dimensions at runtime. It shares
the const configuration's validation: nonzero block dimensions, device and
pipeline limits, total threadgroup storage and nonoverflowing global dimensions.
An empty grid executes no threads but still validates its bindings and geometry.
Scalar kernel parameters do not set the grid automatically.

## Device coordinates

| Rust function | Meaning | CUDA equivalent |
| --- | --- | --- |
| `thread_idx()` | Thread within its block | `threadIdx` |
| `block_idx()` | Block within the grid | `blockIdx` |
| `block_dim()` | Threads per block | `blockDim` |
| `grid_dim()` | Blocks in the grid | `gridDim` |

All four return x/y/z coordinates. The usual global x index is
`block_idx().x * block_dim().x + thread_idx().x`. `thread_index()` instead returns
an ownership witness for the global x index. Kernels using it, including through
helpers, require one-dimensional launches; other shapes are rejected.

## Required shapes

`#[kernel(block = (128, 1, 1))]` records an exact block requirement. Its generated
host function accepts the corresponding const configuration. Host const generics
alone do not specialize handwritten MSL.

These tested pipeline kernels use two required shapes and a shape-independent
final stage:

```rust,ignore
{{#include ../../examples/pipeline/src/main.rs}}
```

The [async chapter](async.md) includes their host launches. SIMD-group matrix
kernels require a declared shape whose thread count is a multiple of 32. Barriers
and collectives also impose participation rules; see [memory and safety](memory.md).

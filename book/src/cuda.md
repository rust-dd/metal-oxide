# Migrating from CUDA

Keep the launch's grid and block meanings when porting a CUDA kernel:

| CUDA | metal-oxide |
| --- | --- |
| `__global__` function | Rust function marked `#[metal_oxide::kernel]` |
| `threadIdx`, `blockIdx` | `thread_idx()`, `block_idx()` |
| `blockDim`, `gridDim` | `block_dim()`, `grid_dim()` |
| `<<<grid, block>>>` | generated launch with `LaunchConfig` or `DynamicLaunchConfig` |
| Device allocation and copy | initialized typed `Buffer<T>` and CPU views |
| Block shared memory | `ThreadgroupBuffer` or `DynamicThreadgroupBuffer` |
| Kernel chain in a stream | ordered buffer dependencies, often in `sequence` |

Metal threadgroups implement blocks; the grid counts blocks. Port the indexing
formula and guard padding threads. For an independent one-dimensional map, prefer
the `thread_index()` and `DisjointBuffer` pattern in [your first kernel](first-kernel.md).
For two or three dimensions, use the explicit coordinates and uphold each unsafe
buffer-access contract.

There is no implicit CUDA stream handle in the public launch API. Operations are
lazy and buffer dependencies order them across runtime queues. Await or wait for
results before observing them on the CPU. `sequence` groups launches in one
command buffer; a successful continuation can launch later work with `then`.

Port warp algorithms using the device SIMD-group API and its participation rules.
Matrix operations use 32-lane groups and 8x8 matrices on supported Apple GPUs;
CUDA tensor-core tile sizes and numerical behavior do not transfer automatically.
MPP remains a benchmark reference. Check both the [Rust subset](supported-rust.md)
and [capabilities](capabilities.md) before translating an intrinsic.

Shared physical memory does not remove synchronization or ownership requirements.
Use typed views and launch borrows instead of retaining unrestricted native aliases.
Compare CPU results, numerical tolerances and performance on the target GPU.

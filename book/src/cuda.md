# Migrating from CUDA

Keep the launch's grid and block meanings when porting a CUDA kernel:

| CUDA | metal-oxide |
| --- | --- |
| `__global__` function | Rust function marked `#[metal_oxide::kernel]` |
| `threadIdx`, `blockIdx` | `thread_idx()`, `block_idx()` |
| `blockDim`, `gridDim` | `block_dim()`, `grid_dim()` |
| `<<<grid, block>>>` | generated launch with `LaunchConfig` or `DynamicLaunchConfig` |
| Device allocation and copy | initialized typed `Buffer<T>` and CPU views |
| `__shared__ T values[N]` | `threadgroup::shared::<T, N>()` returning `ThreadgroupBuffer<T, N>` |
| `extern __shared__` memory | `DynamicThreadgroupBuffer<T>` kernel parameter with a launch-time element count |
| `__syncthreads()` | `unsafe { threadgroup::barrier() }`; orders threadgroup memory only |
| Device atomics such as `atomicAdd` | `AtomicBuffer<T>` and its supported operations and orderings |
| Warp lane ID | `simdgroup::lane_id()` |
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

All threads in a block must reach `threadgroup::barrier()` in the same loop
iteration. It does not order device-buffer accesses; port synchronization that
uses global memory separately.

Shared physical memory does not remove synchronization or ownership requirements.
Use typed views and launch borrows instead of retaining unrestricted native aliases.
Compare CPU results, numerical tolerances and performance on the target GPU.

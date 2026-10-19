# Memory and safety

`Buffer<T>` owns initialized shared Metal storage. `buffer_from_slice` uploads
values and `buffer_zeroed` creates initialized zero values through `GpuValue`.
The allocation's element count is distinct from the number of launched threads.
Allocation sizes and device limits are checked.

`GpuValue` encodes and decodes the canonical GPU layout. Built-in primitive scalars
can use direct CPU views; custom types use a typed CPU mirror. Matching Rust size
or `#[repr(C)]` alone does not authorize interpreting arbitrary bytes as a Rust
value. A `#[kernels]` module generates codecs for its `#[repr(C)]` records used
in the kernel interface.

## Host access and ownership

`buffer.read().await` returns a shared CPU view after pending GPU access completes.
`buffer.write().await` requires an exclusive borrow and returns an exclusive view.
Both also support `.wait()`. Check the launch result for GPU failures: CPU views
wait for access but do not report command errors.

Read launch arguments accept shared borrows, ownership and `Arc`. Write arguments
require exclusive borrows or ownership. Operations retain their arguments through
execution. Devices, pipelines and buffers can cross threads; each operation's
thread traits also depend on its arguments and closures.

Within a sequence, a CPU read panics while a write is recorded. CPU writes, upload
and native export panic while any access is recorded. Conflicting nested batches
return `InvalidBuffer`. These recording reservations also protect the interval
before native submission.

## Device access

Prefer checked slices and `DisjointBuffer` for independent element updates.
Unchecked buffer loads/stores and raw `Launch::new` require explicit unsafe
contracts covering bounds, initialization, aliasing, race freedom and cooperative
participation. An unsafe kernel produces an unsafe host launch function.

Thread-local, threadgroup and device memory are distinct address spaces.
Threadgroup storage must be initialized before reading, and all participating
threads must reach the required barriers. The compiler checks participation
through helper calls and rejects varying paths that bypass barriers or collectives.
It does not replace an unsafe caller's bounds or lane-validity obligations.

`ThreadgroupBuffer` supplies static block-local storage. A
`DynamicThreadgroupBuffer<T>` parameter takes an element count at launch; the
runtime checks the combined static and dynamic storage against the device limit.
Relaxed atomics do not order unrelated memory. Every concurrent access to an atomic
element must be atomic; supported operations are listed in
[supported Rust](supported-rust.md#atomics-and-cooperation).

## Native integration

`Device::from_native` retains a native device and creates its own queues. Unsafe
`Buffer::from_native` is an owning handoff of initialized shared, hazard-tracked
storage on the same physical device. External work must finish first; native
aliases must stay idle through CPU views and runtime accesses. Consuming
`Buffer::into_native` waits for pending access and uploads CPU changes. Dropping a
buffer or future alone does not authorize access through a native alias.
Imported storage must match the element extent, alignment and canonical layout,
and remain non-purgeable and non-aliasable while the runtime uses it.

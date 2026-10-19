# Async operations

A generated launch returns a lazy `Launch`: construction submits no work. Await
it with an executor, or import `Operation` and call `.wait()` to block. First-use
library and pipeline preparation runs on a worker. Awaiting yields while that
preparation, command capacity or GPU execution is pending.

Successful launches return arguments in the form passed. Owned and `Arc` read
buffers can therefore travel through `'static` futures; borrowed buffers remain
borrowed until the operation resolves or is dropped. CPU readback is also async.

## Sequences and dependencies

`sequence(|batch| ...)` puts pushed launches in one command buffer. Buffer accesses
order dependent dispatches; independent dispatches may run concurrently. The
runtime also orders conflicting operations across its queues. Device clones share
that scheduling context; separate device contexts do not share resources implicitly.

The pipeline example verifies two stages against CPU results:

```rust,ignore
{{#include ../../examples/pipeline/src/run.rs}}
```

Its small executor is included because it is part of the tested source. An
application may await the same operation using its existing async executor.

## Composition and cancellation

`Operation::then` starts a fallible continuation after success. `map` transforms a
successful result on the CPU; an error skips subsequent stages. `shared()` drives
an operation once for multiple consumers, whose clones receive the same `Arc` of
its result or error. If both `futures::FutureExt` and `Operation` are imported, use
`Operation::map(op, f)` to disambiguate their methods.

Dropping an operation before submission releases its recording reservations and
submits no work. Submitted GPU work continues, retaining its resources until
completion; dropping the future does not block. Dropping a chain starts none of its
later stages. Dropping every shared consumer stops driving the operation, while
already submitted work completes.

A failed generated dispatch gates its dependents in the command buffer. A new
batch that reads an unfinished writer's output also carries its failure dependency.
See [debugging](debugging.md) for the returned errors.

# Debugging

Start with the launch's `Result`. `Error::Contract` reports a failed host relation;
`InvalidLaunch` reports geometry errors; unsupported artifact requirements fail
before execution. An unproved bounds check, checked overflow, division by zero,
literal panic/assertion or failed unwrap/expect can return `Error::KernelFailed`
with the kernel, dispatch, Rust location and message. Dependent generated dispatches
are gated; a batch depending on an unfinished failed writer returns
`Error::DependencyFailed`.

The compiler must prove a check before a barrier or collective, or show that its
failure is uniform for the participating threads. Unsupported operations fail
compilation with Rust source diagnostics rather than silently changing semantics.

## Validation and capture

After [installing cargo-metal](installation.md), run these from the checkout:

```sh
cargo metal run --validate -p vec-add --locked
cargo metal run --capture target/vec-add.gputrace -p vec-add --locked
```

Validation enables Metal API and shader validation. Capture records the first
runtime device's GPU work into a trace for Xcode; use a fresh path because Metal
does not replace an existing trace. `cargo metal test` supports the same options;
select a single hardware test when capturing tests.

Fresh local capture builds retain Rust line tables and sources in a separate debug
artifact. Saved dependency artifacts can be captured but lack those Rust line
tables. Device `log!` messages use Metal shader logging and appear on stderr; this
requires the [logging capability](capabilities.md).

## Inspect generated code

```sh
cargo metal inspect --emit ir -p vec-add --locked
cargo metal inspect --emit msl -p vec-add --locked
cargo metal inspect --emit air -p vec-add --locked
```

Inspection builds the selected package and prints its kernel libraries. Add
`--kernel NAME` to select a kernel. IR shows the typed representation, MSL shows
lowered control flow and checks, and AIR shows Apple's compiled representation.
`cargo metal setup --check` reports tool versions, installation problems and device
limits without installing tools.

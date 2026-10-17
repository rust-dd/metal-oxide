#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

/// A pending chain or a shared clone keeps the buffers its launches borrow.
#[test]
fn composed_operations_keep_their_borrows() {
    trybuild::TestCases::new().compile_fail("tests/borrows/*.rs");
}

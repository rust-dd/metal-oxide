#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn signed_narrow_values_preserve_rust_wrapping_and_casts() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/signed_narrow.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let actual = support::execute_msl(
        &directory,
        "char a[8] = {}; short b[8] = {}; signed_narrow(a, b, char(-128), short(-32768), 9u, {}, {}, {}, {}); for (auto value : a) std::cout << int(value) << ','; for (auto value : b) std::cout << int(value) << ',';",
    );
    let a = i8::MIN;
    let b = i16::MIN;
    let expected = [
        a.wrapping_add(127) as i32,
        a.wrapping_sub(127) as i32,
        a.wrapping_mul(-127) as i32,
        a.wrapping_shl(9) as i32,
        a.wrapping_shr(9) as i32,
        a.wrapping_neg() as i32,
        !a as i32,
        b as i8 as i32,
        b.wrapping_add(32767) as i32,
        b.wrapping_sub(32767) as i32,
        b.wrapping_mul(-32767) as i32,
        b.wrapping_shl(9) as i32,
        b.wrapping_shr(9) as i32,
        b.wrapping_neg() as i32,
        !b as i32,
        a as i16 as i32,
    ]
    .map(|value| format!("{value},"))
    .concat();
    assert_eq!(actual, expected);
}

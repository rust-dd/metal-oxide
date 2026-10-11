#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn saturating_casts_execute_with_rust_nan_and_range_rules() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/float_casts.rs",
        &[],
    );
    support::checked(output);
    let result = support::execute_msl(
        &directory,
        "float input[4] = {as_type<float>(0x7fc00000u), as_type<float>(0x7f800000u), as_type<float>(0xff800000u), -129.75f}; int s[4]={}; uint u[4]={}, high[4]={}; char a[4]={}; uchar b[4]={}; short c[4]={}; ushort d[4]={}; for(uint i=0;i<4;i++) float_casts(input,s,u,a,b,c,d,high,4u,{i,0,0},{},{4,1,1},{1,1,1}); for(uint i=0;i<4;i++) std::cout<<s[i]<<','<<u[i]<<','<<int(a[i])<<','<<uint(b[i])<<','<<c[i]<<','<<d[i]<<','<<high[i]<<';';",
    );
    assert_eq!(
        result,
        "0,0,0,0,0,0,0;2147483647,4294967295,127,255,32767,65535,4294967295;-2147483648,0,-128,0,-32768,0,0;-129,0,-128,0,-129,0,0;"
    );
}

#[test]
fn unchecked_float_conversions_skip_saturation() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/unchecked_casts.rs",
        &[],
    );
    support::checked(output);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    assert!(!msl.contains("isnan"), "{msl}");
    let result = support::execute_msl(
        &directory,
        "float input[1] = {7.75f}; uint u[1] = {}; int s[1] = {}; unchecked_casts(input, u, s, {}, {}, {}, {}); std::cout << u[0] << ',' << s[0];",
    );
    assert_eq!(result, "7,-7");
}

#[test]
fn precise_math_keeps_explicit_fma_separate_from_ordinary_arithmetic() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/float_math.rs",
        &[],
    );
    support::checked(output);
    let result = support::execute_msl(
        &directory,
        "float out[7]={}; float_math(out, 1.0f+as_type<float>(0x34000000u), 1.0f-as_type<float>(0x34000000u), -1.0f, {}, {}, {}, {}); std::cout << as_type<uint>(out[4]) << ',' << as_type<uint>(out[5]) << ',' << as_type<uint>(out[6]);",
    );
    let a = 1.0_f32 + f32::EPSILON;
    let b = 1.0_f32 - f32::EPSILON;
    assert_eq!(
        result,
        format!(
            "{},{},{}",
            a.mul_add(b, -1.0).to_bits(),
            (a * b - 1.0).to_bits(),
            a.to_bits()
        )
    );
}

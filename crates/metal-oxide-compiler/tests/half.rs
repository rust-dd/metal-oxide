#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn half_storage_and_explicit_conversions_execute() {
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/half.rs", &[]);
    support::checked(output);
    let actual = support::execute_msl(
        &directory,
        "half input[4]={as_type<half>(ushort(0x3c00)),as_type<half>(ushort(0x8000)),as_type<half>(ushort(1)),as_type<half>(ushort(0x7c00))}; ushort bits[4]={}; float wide[4]={}; half copy[4]={}; for(uint i=0;i<4;i++) half_bits(input,bits,wide,copy,4u,{i,0,0},{},{4,1,1},{1,1,1}); for(uint i=0;i<4;i++) std::cout<<bits[i]<<','<<as_type<uint>(wide[i])<<','<<as_type<ushort>(copy[i])<<';';",
    );
    assert_eq!(
        actual,
        "15360,1065353216,15360;32768,2147483648,32768;1,864026624,1;31744,2139095040,31744;"
    );
}

use super::Kernel;

#[test]
fn launch_functions_follow_the_kernel_signature() {
    let function = syn::parse_quote! {
        /// Scales in place.
        pub fn scale(
            input: ReadBuffer<f32>,
            mut output: device::DisjointBuffer<f32>,
            totals: AtomicBuffer<u32>,
            device: f32,
            shared: DynamicThreadgroupBuffer<f32>,
        ) {}
    };
    let kernel = Kernel::new(&function, Some([64, 1, 1])).unwrap();
    let launcher = kernel
        .launcher(&quote::format_ident!("LIBRARY"))
        .to_string()
        .replace(' ', "");
    for expected in [
        "pubfnscale<MetalOxideBuffer0,MetalOxideBuffer1,MetalOxideBuffer2>",
        "config:::metal_oxide::LaunchConfig<64u32,1u32,1u32>",
        "argument_3:f32",
        "shared:::core::primitive::usize",
        "MetalOxideBuffer1:::metal_oxide::WriteArgument<f32>",
        "::metal_oxide::Argument::atomic(::metal_oxide::WriteArgument::buffer_mut(totals))",
        "::metal_oxide::Argument::threadgroup::<f32>(*shared)?",
        "::metal_oxide::Launch<(MetalOxideBuffer0,MetalOxideBuffer1,MetalOxideBuffer2,f32,::core::primitive::usize,),5usize>",
    ] {
        assert!(launcher.contains(expected), "{expected} in {launcher}");
    }
    let unsafe_kernel = syn::parse_quote! { unsafe fn raw(n: u32) {} };
    let launcher = Kernel::new(&unsafe_kernel, None)
        .unwrap()
        .launcher(&quote::format_ident!("LIBRARY"))
        .to_string();
    assert!(launcher.contains("unsafe fn raw"));
    assert!(launcher.contains("# Safety"));
    let generic = syn::parse_quote! { fn generic<T>(value: T) {} };
    assert!(Kernel::new(&generic, None).is_err());
}

use proc_macro::TokenStream;

/// Marks a concrete, unsafe kernel entrypoint for the Metal compiler.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, item: TokenStream) -> TokenStream {
    if !attributes.is_empty() {
        return "compile_error!(\"#[kernel] takes no arguments\");"
            .parse()
            .unwrap();
    }
    let mut output = "#[cfg_attr(target_env = \"metal\", metal_oxide::kernel)]"
        .parse::<TokenStream>()
        .unwrap();
    output.extend(item);
    output
}

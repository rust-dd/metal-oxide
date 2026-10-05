use proc_macro::{Delimiter, TokenStream, TokenTree};

/// Marks a concrete, unsafe kernel entrypoint for the Metal compiler.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let mut output = "#[cfg_attr(target_env = \"metal\", metal_oxide::kernel)]"
        .parse::<TokenStream>()
        .unwrap();
    if !attributes.is_empty() {
        let Some(shape) = block_shape(attributes) else {
            return "compile_error!(\"expected #[kernel(block = (X, Y, Z))] with positive integer literals\");".parse().unwrap();
        };
        output.extend(
            format!(
                "#[cfg_attr(target_env = \"metal\", metal_oxide::block_shape = \"{},{},{}\")]",
                shape[0], shape[1], shape[2],
            )
            .parse::<TokenStream>()
            .unwrap(),
        );
    }
    output.extend(item);
    output
}

fn block_shape(attributes: TokenStream) -> Option<[u32; 3]> {
    let tokens = attributes.into_iter().collect::<Vec<_>>();
    let [
        TokenTree::Ident(name),
        TokenTree::Punct(equal),
        TokenTree::Group(values),
    ] = tokens.as_slice()
    else {
        return None;
    };
    if name.to_string() != "block"
        || equal.as_char() != '='
        || values.delimiter() != Delimiter::Parenthesis
    {
        return None;
    }
    let values = values.stream().into_iter().collect::<Vec<_>>();
    let [
        TokenTree::Literal(x),
        TokenTree::Punct(a),
        TokenTree::Literal(y),
        TokenTree::Punct(b),
        TokenTree::Literal(z),
    ] = values.as_slice()
    else {
        return None;
    };
    if a.as_char() != ',' || b.as_char() != ',' {
        return None;
    }
    let parse = |v: &proc_macro::Literal| v.to_string().replace('_', "").parse::<u32>().ok();
    let shape = [parse(x)?, parse(y)?, parse(z)?];
    shape.into_iter().try_fold(1_u32, |n, axis| {
        (axis != 0).then(|| n.checked_mul(axis)).flatten()
    })?;
    Some(shape)
}

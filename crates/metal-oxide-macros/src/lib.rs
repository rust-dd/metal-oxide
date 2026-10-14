use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

/// Marks a concrete kernel entrypoint for the Metal compiler; a safe `fn` gets a safe launch
/// method. `block = (X, Y, Z)` fixes the block shape, and `requires(...)` lists relations
/// between buffer lengths and integer parameters that every launch checks.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let mut output = "#[cfg_attr(target_env = \"metal\", metal_oxide::kernel)]"
        .parse::<TokenStream>()
        .unwrap();
    let tokens = attributes.into_iter().collect::<Vec<_>>();
    for option in tokens.split(|token| matches!(token, TokenTree::Punct(p) if p.as_char() == ',')) {
        let attribute = match option {
            [] => continue,
            [TokenTree::Ident(name), TokenTree::Group(relations)]
                if name.to_string() == "requires"
                    && relations.delimiter() == Delimiter::Parenthesis =>
            {
                let mut attribute = "metal_oxide::requires".parse::<TokenStream>().unwrap();
                attribute.extend([TokenTree::Group(relations.clone())]);
                attribute
            }
            _ => match block_shape(option) {
                Some([x, y, z]) => format!("metal_oxide::block_shape = \"{x},{y},{z}\"")
                    .parse()
                    .unwrap(),
                None => {
                    return "compile_error!(\"expected #[kernel(block = (X, Y, Z), requires(...))] with positive integer block literals\");"
                        .parse()
                        .unwrap();
                }
            },
        };
        let mut cfg = "target_env = \"metal\",".parse::<TokenStream>().unwrap();
        cfg.extend(attribute);
        let mut cfg_attr = "cfg_attr".parse::<TokenStream>().unwrap();
        cfg_attr.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, cfg))]);
        output.extend("#".parse::<TokenStream>().unwrap());
        output.extend([TokenTree::Group(Group::new(Delimiter::Bracket, cfg_attr))]);
    }
    output.extend(item);
    output
}

fn block_shape(tokens: &[TokenTree]) -> Option<[u32; 3]> {
    let [
        TokenTree::Ident(name),
        TokenTree::Punct(equal),
        TokenTree::Group(values),
    ] = tokens
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

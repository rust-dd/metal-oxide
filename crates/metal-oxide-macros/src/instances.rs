use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use std::collections::HashSet;

pub(super) fn markers(arguments: &Group, item: &TokenStream) -> Result<TokenStream, &'static str> {
    let tokens = item.clone().into_iter().collect::<Vec<_>>();
    let function = tokens
        .windows(2)
        .find_map(|pair| match pair {
            [TokenTree::Ident(keyword), TokenTree::Ident(name)] if keyword.to_string() == "fn" => {
                Some(name.clone())
            }
            _ => None,
        })
        .ok_or("kernel instances require a free function")?;
    let tokens = arguments.stream().into_iter().collect::<Vec<_>>();
    let mut names = HashSet::new();
    let mut output = TokenStream::new();
    for instance in tokens.split(|token| matches!(token, TokenTree::Punct(p) if p.as_char() == ','))
    {
        if instance.is_empty() && !names.is_empty() {
            continue;
        }
        let [TokenTree::Ident(name), TokenTree::Group(arguments)] = instance else {
            return Err("expected instances(name(Type, CONST), ...) with at least one instance");
        };
        if arguments.delimiter() != Delimiter::Parenthesis || arguments.stream().is_empty() {
            return Err("a kernel instance needs explicit generic arguments");
        }
        let name = name.to_string();
        let name = name.strip_prefix("r#").unwrap_or(&name);
        if !names.insert(name.to_owned()) {
            return Err("duplicate kernel instance name");
        }
        output.extend(format!(
            "#[cfg(target_env = \"metal\")] #[metal_oxide_compiler::instance = {name:?}] const _: () ="
        ).parse::<TokenStream>().unwrap());
        let mut body = "let _ =".parse::<TokenStream>().unwrap();
        body.extend([TokenTree::Ident(function.clone())]);
        body.extend("::<".parse::<TokenStream>().unwrap());
        body.extend(arguments.stream());
        body.extend(">;".parse::<TokenStream>().unwrap());
        output.extend([TokenTree::Group(Group::new(Delimiter::Brace, body))]);
        output.extend(";".parse::<TokenStream>().unwrap());
    }
    Ok(output)
}

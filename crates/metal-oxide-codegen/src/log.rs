use metal_oxide_ir::{Error, LogPiece, Scalar, SourceLocation, log_pieces};

/// `os_log_default.log` with printf specifiers for each argument; floats print round-trip digits.
pub(crate) fn expression(
    format: &str,
    arguments: &[(String, Scalar)],
    source: &SourceLocation,
) -> Result<String, Error> {
    let pieces = log_pieces(format).map_err(|message| Error::new(source, message))?;
    let mut printf = String::new();
    let mut values = Vec::new();
    let mut next = arguments.iter();
    for piece in pieces {
        match piece {
            LogPiece::Text(text) => {
                for c in text.chars() {
                    match c {
                        '%' => printf.push_str("%%"),
                        '"' => printf.push_str("\\\""),
                        '\\' => printf.push_str("\\\\"),
                        '\n' => printf.push_str("\\n"),
                        '\t' => printf.push_str("\\t"),
                        c if c.is_control() => printf.push_str(&format!("\\{:03o}", c as u32)),
                        c => printf.push(c),
                    }
                }
            }
            LogPiece::Argument => {
                let (value, scalar) = next.next().unwrap();
                let (specifier, value) = match scalar {
                    Scalar::U8 => ("%hhu", value.clone()),
                    Scalar::U16 => ("%hu", value.clone()),
                    Scalar::U32 => ("%u", value.clone()),
                    Scalar::U64 | Scalar::Usize => ("%lu", value.clone()),
                    Scalar::I8 => ("%hhd", value.clone()),
                    Scalar::I16 => ("%hd", value.clone()),
                    Scalar::I32 => ("%d", value.clone()),
                    Scalar::I64 | Scalar::Isize => ("%ld", value.clone()),
                    Scalar::F32 => ("%.9g", value.clone()),
                    Scalar::F16 | Scalar::BF16 => ("%.5g", format!("float({value})")),
                    Scalar::Bool => unreachable!("validated log argument"),
                };
                printf.push_str(specifier);
                values.push(value);
            }
        }
    }
    let values = values
        .iter()
        .map(|value| format!(", {value}"))
        .collect::<String>();
    Ok(format!("os_log_default.log(\"{printf}\"{values})"))
}

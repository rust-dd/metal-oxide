/// A piece of a device log format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogPiece {
    Text(String),
    /// The next argument, printed with its type's default format.
    Argument,
}

/// Parses a Rust format string with `{}` placeholders and `{{`/`}}` escapes.
pub fn log_pieces(format: &str) -> Result<Vec<LogPiece>, String> {
    let mut pieces = Vec::new();
    let mut text = String::new();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, chars.peek()) {
            ('{', Some('{')) | ('}', Some('}')) => {
                chars.next();
                text.push(c);
            }
            ('{', Some('}')) => {
                chars.next();
                if !text.is_empty() {
                    pieces.push(LogPiece::Text(std::mem::take(&mut text)));
                }
                pieces.push(LogPiece::Argument);
            }
            ('{' | '}', _) => {
                return Err("device log formats support only `{}` placeholders".into());
            }
            _ => text.push(c),
        }
    }
    if !text.is_empty() {
        pieces.push(LogPiece::Text(text));
    }
    Ok(pieces)
}

use proc_macro2::Span;
use std::path::Path;

#[cfg(test)]
mod tests;

type Position = (usize, usize);

/// The start position of every token of the macro input, to place device errors on them.
pub(crate) struct Positions(Vec<(Position, Span)>);

impl Positions {
    /// No tokens, for errors in another crate's files.
    pub(crate) fn empty() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn new(item: &proc_macro::TokenStream) -> Self {
        fn collect(stream: proc_macro::TokenStream, positions: &mut Vec<(Position, Span)>) {
            for token in stream {
                let span = token.span();
                positions.push(((span.line(), span.column()), span.into()));
                if let proc_macro::TokenTree::Group(group) = token {
                    collect(group.stream(), positions);
                }
            }
        }
        let mut positions = Vec::new();
        collect(item.clone(), &mut positions);
        positions.sort_by_key(|(position, _)| *position);
        Self(positions)
    }

    /// The token that starts at or last before `position`.
    fn at(&self, position: Position) -> Option<Span> {
        let index = self.0.partition_point(|(start, _)| *start <= position);
        index.checked_sub(1).map(|index| self.0[index].1)
    }
}

/// The device build's errors on the input tokens they name, or the build's whole message on
/// `fallback` when it names none.
pub(crate) fn errors(
    message: &str,
    file: &Path,
    positions: &Positions,
    fallback: Span,
) -> syn::Error {
    let file = file.display().to_string();
    let lines = message
        .lines()
        .map(|line| line.strip_prefix("rustup: ").unwrap_or(line))
        .collect::<Vec<_>>();
    let mut errors = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        // rustc: `error[E0277]: message` followed by `--> file:line:column` and its notes.
        let details = &lines[index + 1..];
        let details = &details[..details
            .iter()
            .position(|next| next.starts_with("error"))
            .unwrap_or(details.len())];
        if let Some(header) = line.strip_prefix("error")
            && let Some((_, text)) = header.split_once(": ")
            && let Some(location) = details
                .iter()
                .find_map(|next| next.trim_start().strip_prefix("--> "))
            && let Some(position) = position(location, &file)
        {
            errors.push((position, format!("{text}{}", notes(details))));
        }
        // Apple's compiler through `#line`: `file:line:column: error: message`.
        if let Some(rest) = line.strip_prefix(&format!("{file}:"))
            && let Some((location, text)) = rest.split_once(": error: ")
            && let Some(position) = position(&format!("{file}:{location}"), &file)
        {
            errors.push((position, text.to_owned()));
        }
        // The metal-oxide compiler: `error: file:line:column: message`.
        if let Some(rest) = line.strip_prefix(&format!("error: {file}:"))
            && let Some((location, text)) = rest.split_once(": ")
            && let Some(position) = position(&format!("{file}:{location}"), &file)
        {
            errors.push((position, text.to_owned()));
        }
    }
    let placed = errors
        .into_iter()
        .filter_map(|(position, text)| {
            positions
                .at(position)
                .map(|span| syn::Error::new(span, format!("GPU build: {text}")))
        })
        .reduce(|mut combined, error| {
            combined.combine(error);
            combined
        });
    placed.unwrap_or_else(|| {
        let text = lines
            .iter()
            .filter(|line| {
                let trimmed = line.trim_start();
                !["Checking ", "Compiling ", "Blocking "]
                    .iter()
                    .any(|progress| trimmed.starts_with(progress))
            })
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        syn::Error::new(fallback, format!("GPU build failed:\n{text}"))
    })
}

/// The notes of one rustc diagnostic, each on its own line with the location it names, such as
/// the code in another crate that a kernel's call reached.
fn notes(details: &[&str]) -> String {
    let mut notes = String::new();
    for (index, line) in details.iter().enumerate() {
        let line = line.trim_start();
        let Some(note) = line
            .strip_prefix("note: ")
            .or_else(|| line.strip_prefix("= note: "))
        else {
            continue;
        };
        let location = details[index + 1..]
            .iter()
            .map(|next| next.trim_start())
            .take_while(|next| !next.starts_with("note: ") && !next.starts_with("= note: "))
            .find_map(|next| next.strip_prefix("--> "));
        notes.push_str("\nnote: ");
        notes.push_str(note);
        if let Some(location) = location {
            notes.push_str(&format!(" ({location})"));
        }
    }
    notes
}

/// The line and column of `file:line:column` when it names `file`.
fn position(location: &str, file: &str) -> Option<Position> {
    let rest = location.trim().strip_prefix(file)?.strip_prefix(':')?;
    let mut parts = rest.split(':');
    let line = parts.next()?.parse().ok()?;
    let column = parts.next()?.trim().parse().ok()?;
    Some((line, column))
}

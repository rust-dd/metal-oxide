#![cfg(feature = "rustc-private")]

mod support;

use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

/// The calls that report a rejection of kernel source or options, each followed by its message.
const REPORTS: [&str; 7] = [
    "Error::new(",
    "span_err(",
    "span_fatal(",
    "Err(\"",
    "Err(format!(\"",
    "error(\"",
    "error(&format!(\"",
];

/// Where the reports in `text` start their arguments: the calls in [`REPORTS`], and
/// `(span, message)` errors, including a span field and a message on the next line.
fn reports(text: &str) -> impl Iterator<Item = usize> + '_ {
    let calls = REPORTS.iter().flat_map(move |report| {
        text.match_indices(report)
            .map(|(at, _)| at + report.len() - 1)
    });
    let spans = text
        .match_indices("span,")
        .filter(move |(at, _)| {
            text[..*at]
                .trim_end_matches(|ch: char| ch.is_alphanumeric() || matches!(ch, '_' | '.'))
                .trim_end()
                .ends_with('(')
        })
        .map(|(at, _)| at + "span".len());
    calls.chain(spans)
}

#[test]
fn span_error_inventory_includes_local_and_field_spans() {
    let text = r#"
        Err((span, "plain".into()));
        Err((argument.span, "field".into()));
        Err((fn_span, "named".into()));
    "#;
    let messages = reports(text)
        .filter_map(|at| literal(&text[at..]))
        .collect::<Vec<_>>();
    assert_eq!(messages, ["plain", "field", "named"]);
}

/// The rejection messages in the compiler, IR and codegen sources, with a site of each.
fn messages() -> BTreeMap<String, String> {
    let mut found = BTreeMap::new();
    for crate_name in [
        "metal-oxide-compiler",
        "metal-oxide-ir",
        "metal-oxide-codegen",
    ] {
        let root = support::root().join("crates").join(crate_name).join("src");
        // Unit tests, which live in tests.rs modules, build errors of their own.
        for path in sources(&root)
            .into_iter()
            .filter(|path| !path.ends_with("tests.rs"))
        {
            let text = std::fs::read_to_string(&path).unwrap();
            for at in reports(&text) {
                if let Some(message) = literal(&text[at..]) {
                    let line = text[..at].lines().count();
                    let site = format!("{}:{line}", path.display());
                    found.entry(message).or_insert(site);
                }
            }
        }
    }
    found
}

fn sources(directory: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(sources(&path));
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
    found
}

/// The string literal that starts the arguments in `text`, before any other statement.
fn literal(text: &str) -> Option<String> {
    let start = text.find('"')?;
    if text[..start].contains([';', '{', '}']) || start > 80 {
        return None;
    }
    let mut message = String::new();
    let mut characters = text[start + 1..].chars();
    while let Some(character) = characters.next() {
        match character {
            '"' => return Some(message),
            // A line continuation drops the line break and the indentation after it.
            '\\' => match characters.next()? {
                '\n' => {
                    while characters.clone().next().is_some_and(char::is_whitespace) {
                        characters.next();
                    }
                }
                escaped => {
                    message.push('\\');
                    message.push(escaped);
                }
            },
            other => message.push(other),
        }
    }
    None
}

/// The entries of `tests/rejections.json` by message.
fn inventory() -> BTreeMap<String, Value> {
    let path = support::root().join("crates/metal-oxide-compiler/tests/rejections.json");
    let entries = serde_json::from_slice::<Vec<Value>>(&std::fs::read(path).unwrap()).unwrap();
    entries
        .into_iter()
        .map(|entry| (entry["message"].as_str().unwrap().to_owned(), entry))
        .collect()
}

#[test]
fn every_rejection_in_the_compiler_has_an_inventory_entry() {
    let found = messages();
    let inventory = inventory();
    let missing = found
        .iter()
        .filter(|(message, _)| !inventory.contains_key(*message))
        .collect::<Vec<_>>();
    let stale = inventory
        .keys()
        .filter(|message| !found.contains_key(*message))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "rejections without an entry in tests/rejections.json: {missing:#?}\nentries without a rejection: {stale:#?}"
    );
    for (message, entry) in &inventory {
        let fixture = entry.get("fixture").is_some();
        let internal = entry.get("internal").is_some();
        assert!(
            fixture != internal,
            "{message}: an entry has a fixture or an internal reason"
        );
    }
}

/// The text of `message` before its first placeholder, which every report of it contains.
fn fixed(message: &str) -> &str {
    let prefix = message.split('{').next().unwrap();
    if prefix.is_empty() {
        message.split_once('}').map_or(message, |(_, rest)| rest)
    } else {
        prefix
    }
}

fn check(message: &str, entry: &Value) {
    let fixture = entry["fixture"].as_str().unwrap();
    let options = entry
        .get("options")
        .and_then(Value::as_array)
        .map(|options| {
            options
                .iter()
                .map(|option| option.as_str().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let path = format!("crates/metal-oxide-compiler/tests/fixtures/{fixture}");
    let (output, directory) = support::emit(&path, &options);
    let error = String::from_utf8_lossy(&output.stderr).into_owned();
    support::rejected(output, fixed(message));
    // Command-line options have no source location.
    if entry.get("line").is_some() {
        let name = Path::new(fixture).file_name().unwrap().to_str().unwrap();
        let location = format!("{name}:{}:{}", entry["line"], entry["column"]);
        assert!(
            error.contains(&location),
            "{message}: expected {location}:\n{error}"
        );
    }
    assert!(!directory.join("kernels.metal").exists(), "{message}");
}

#[test]
fn reachable_rejections_report_at_their_fixtures() {
    let inventory = inventory();
    let reachable = inventory
        .iter()
        .filter(|(_, entry)| entry.get("fixture").is_some())
        .collect::<Vec<_>>();
    std::thread::scope(|scope| {
        for chunk in reachable.chunks(reachable.len().div_ceil(8).max(1)) {
            scope.spawn(move || {
                for (message, entry) in chunk {
                    check(message, entry);
                }
            });
        }
    });
}

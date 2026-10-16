use metal_oxide_artifact::CAPABILITIES;
use std::fmt::Write;

fn main() {
    let mut table = String::from(
        "| Artifact feature | MSL | macOS | Apple GPU family | Mac2 |\n| --- | --- | --- | --- | --- |\n",
    );
    for feature in CAPABILITIES {
        writeln!(
            table,
            "| `{}` | {}.{} | {} | {} | {} |",
            feature.name,
            feature.msl.0,
            feature.msl.1,
            feature.macos,
            feature.apple_family,
            if feature.mac2 { "yes" } else { "no" }
        )
        .unwrap();
    }
    let readme = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
    let text = std::fs::read_to_string(&readme).unwrap();
    let (before, rest) = text.split_once("<!-- capabilities -->\n").unwrap();
    let (_, after) = rest.split_once("<!-- /capabilities -->").unwrap();
    let expected = format!("{before}<!-- capabilities -->\n{table}<!-- /capabilities -->{after}");
    if std::env::args().any(|arg| arg == "--check") {
        assert_eq!(
            text, expected,
            "run cargo run -p metal-oxide-artifact --example capabilities"
        );
    } else {
        std::fs::write(readme, expected).unwrap();
    }
}

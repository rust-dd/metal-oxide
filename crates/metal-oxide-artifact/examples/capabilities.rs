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
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let check = std::env::args().any(|arg| arg == "--check");
    for file in ["README.md", "book/src/capabilities.md"] {
        let path = root.join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        let (before, rest) = text.split_once("<!-- capabilities -->\n").unwrap();
        let (_, after) = rest.split_once("<!-- /capabilities -->").unwrap();
        let expected =
            format!("{before}<!-- capabilities -->\n{table}<!-- /capabilities -->{after}");
        if check {
            assert_eq!(
                text, expected,
                "{file}: run cargo run -p metal-oxide-artifact --example capabilities"
            );
        } else if text != expected {
            std::fs::write(path, expected).unwrap();
        }
    }
}

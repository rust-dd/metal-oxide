use metal_oxide_artifact::MathMode;
use rustc_session::{EarlyDiagCtxt, config};
use std::path::PathBuf;

pub(crate) struct Options {
    pub(crate) directory: Option<PathBuf>,
    pub(crate) math_mode: MathMode,
}

pub(crate) fn take_options(arguments: &mut Vec<String>) -> Result<Options, String> {
    let mut directory = None;
    let mut math_mode = None;
    let mut index = 1;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if option == "--" {
            break;
        }
        if !matches!(option, "--metal-output" | "--metal-math-mode") {
            index += 1;
            continue;
        }
        let value = arguments
            .get(index + 1)
            .filter(|v| !v.starts_with('-') && !v.is_empty())
            .ok_or_else(|| match option {
                "--metal-output" => "--metal-output requires a directory".to_owned(),
                _ => "--metal-math-mode requires precise or fast".to_owned(),
            })?;
        match option {
            "--metal-output" if directory.is_none() => directory = Some(PathBuf::from(value)),
            "--metal-math-mode" if math_mode.is_none() => {
                math_mode = Some(value.parse::<MathMode>().map_err(|e| e.to_string())?);
            }
            _ => return Err(format!("{option} must be specified once")),
        }
        arguments.drain(index..index + 2);
    }
    Ok(Options {
        directory,
        math_mode: math_mode.unwrap_or_default(),
    })
}

pub(crate) fn help(arguments: &[String]) -> bool {
    let diagnostics = EarlyDiagCtxt::new(config::ErrorOutputType::default());
    let expanded = rustc_driver::args::arg_expand_all(&diagnostics, &arguments[1..]);
    let mut options = Default::default();
    for option in config::rustc_optgroups() {
        option.apply(&mut options);
    }
    let Ok(matches) = options.parse(&expanded) else {
        return false;
    };
    expanded.is_empty()
        || matches.opt_present("help")
        || ["C", "Z"]
            .iter()
            .any(|option| matches.opt_strs(option).iter().any(|value| value == "help"))
}

use rustc_session::{EarlyDiagCtxt, config};
use std::path::PathBuf;

pub(crate) fn take_directory(arguments: &mut Vec<String>) -> Result<Option<PathBuf>, String> {
    let mut directory = None;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] != "--metal-output" {
            index += 1;
            continue;
        }
        if directory.is_some() {
            return Err("--metal-output must be specified once".into());
        }
        let value = arguments
            .get(index + 1)
            .filter(|v| !v.starts_with('-') && !v.is_empty())
            .ok_or("--metal-output requires a directory")?;
        directory = Some(PathBuf::from(value));
        arguments.drain(index..index + 2);
    }
    Ok(directory)
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

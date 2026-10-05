use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    Build,
    Run,
    Test,
    Inspect,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) action: Action,
    pub(crate) manifest: PathBuf,
    pub(crate) package: Option<String>,
    pub(crate) release: bool,
    pub(crate) arguments: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Command {
    Doctor,
    Help,
    Version,
    Build(Options),
}

pub(crate) fn parse(arguments: &[&str]) -> Result<Command, String> {
    let arguments = arguments.strip_prefix(&["metal"]).unwrap_or(arguments);
    match arguments {
        [] | ["--help" | "-h" | "help"] => return Ok(Command::Help),
        ["--version" | "-V"] => return Ok(Command::Version),
        ["doctor"] => return Ok(Command::Doctor),
        _ => {}
    }
    let action = match arguments.first().copied() {
        Some("build") => Action::Build,
        Some("run") => Action::Run,
        Some("test") => Action::Test,
        Some("inspect") => Action::Inspect,
        _ => return Err("available commands: build, run, test, inspect, doctor".into()),
    };
    let mut options = Options {
        action,
        manifest: "Cargo.toml".into(),
        package: None,
        release: false,
        arguments: vec![],
    };
    let mut index = 1;
    let mut emit = false;
    let mut manifest = false;
    while index < arguments.len() {
        match arguments[index] {
            "--release" if !options.release => options.release = true,
            "--manifest-path" | "--package" | "-p" | "--emit" => {
                let option = arguments[index];
                index += 1;
                let value = arguments
                    .get(index)
                    .filter(|v| !v.is_empty() && !v.starts_with('-'))
                    .ok_or_else(|| format!("{option} requires a value"))?;
                match option {
                    "--manifest-path" if !manifest => {
                        options.manifest = PathBuf::from(value);
                        manifest = true;
                    }
                    "--package" | "-p" if options.package.is_none() => {
                        options.package = Some((*value).to_owned())
                    }
                    "--emit" if action == Action::Inspect && *value == "msl" && !emit => {
                        emit = true
                    }
                    _ => return Err(format!("invalid or repeated option {option}")),
                }
            }
            "--" if matches!(action, Action::Run | Action::Test) => {
                options.arguments = arguments[index + 1..]
                    .iter()
                    .map(|v| (*v).to_owned())
                    .collect();
                break;
            }
            option => return Err(format!("unknown option {option}")),
        }
        index += 1;
    }
    if action == Action::Inspect && !emit {
        return Err("inspect requires --emit msl".into());
    }
    Ok(Command::Build(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_cargo_prefix_and_forwards_only_host_arguments() {
        assert_eq!(parse(&["metal", "doctor"]), Ok(Command::Doctor));
        let Command::Build(options) =
            parse(&["metal", "run", "-p", "host", "--release", "--", "--n", "10"]).unwrap()
        else {
            panic!()
        };
        assert_eq!(options.package.as_deref(), Some("host"));
        assert_eq!(options.arguments, ["--n", "10"]);
        assert!(options.release);
    }

    #[test]
    fn rejects_ambiguous_or_unimplemented_options() {
        for args in [
            &["doctor", "--unknown"][..],
            &["build", "--"][..],
            &["inspect"][..],
            &["inspect", "--emit", "air"][..],
            &["run", "-p"][..],
            &["run", "-p", "a", "-p", "b"][..],
            &["run", "-p", ""][..],
            &["run", "--manifest-path", "a", "--manifest-path", "b"][..],
        ] {
            assert!(parse(args).is_err(), "{args:?}");
        }
    }
}

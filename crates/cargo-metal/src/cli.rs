use metal_oxide_artifact::MathMode;
use metal_oxide_build::Emit;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    Build,
    Run,
    Test,
    Inspect(Emit),
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) action: Action,
    pub(crate) manifest: PathBuf,
    pub(crate) package: Option<String>,
    pub(crate) release: bool,
    pub(crate) math_mode: MathMode,
    /// Runs the host under Metal API and shader validation.
    pub(crate) validate: bool,
    /// Records the host's GPU work into this `.gputrace`, with line tables in the kernels.
    pub(crate) capture: Option<PathBuf>,
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
        Some("inspect") => Action::Inspect(Emit::Msl),
        _ => return Err("available commands: build, run, test, inspect, doctor".into()),
    };
    let mut options = Options {
        action,
        manifest: "Cargo.toml".into(),
        package: None,
        release: false,
        math_mode: MathMode::default(),
        validate: false,
        capture: None,
        arguments: vec![],
    };
    let mut index = 1;
    let mut emit = None;
    let mut manifest = false;
    let mut math_mode = false;
    while index < arguments.len() {
        match arguments[index] {
            "--release" if !options.release => options.release = true,
            "--validate" if matches!(action, Action::Run | Action::Test) && !options.validate => {
                options.validate = true
            }
            "--manifest-path" | "--package" | "-p" | "--emit" | "--math-mode" | "--capture" => {
                let option = arguments[index];
                index += 1;
                let value = arguments
                    .get(index)
                    .filter(|v| !v.is_empty() && !v.starts_with('-'))
                    .ok_or_else(|| format!("{option} requires a value"))?;
                match option {
                    "--math-mode" if !math_mode => {
                        options.math_mode = value.parse::<MathMode>().map_err(|e| e.to_string())?;
                        math_mode = true;
                    }
                    "--manifest-path" if !manifest => {
                        options.manifest = PathBuf::from(value);
                        manifest = true;
                    }
                    "--package" | "-p" if options.package.is_none() => {
                        options.package = Some((*value).to_owned())
                    }
                    "--capture"
                        if matches!(action, Action::Run | Action::Test)
                            && options.capture.is_none() =>
                    {
                        options.capture = Some(PathBuf::from(value))
                    }
                    "--emit" if matches!(action, Action::Inspect(_)) && emit.is_none() => {
                        emit = Some(match *value {
                            "ir" => Emit::Ir,
                            "msl" => Emit::Msl,
                            "air" => Emit::Air,
                            _ => return Err("--emit takes ir, msl or air".into()),
                        });
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
    if let Action::Inspect(_) = options.action {
        options.action = Action::Inspect(emit.ok_or("inspect requires --emit ir|msl|air")?);
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
        assert_eq!(options.math_mode, MathMode::Precise);
        assert!(!options.validate);
        assert_eq!(options.capture, None);
        for action in ["run", "test"] {
            let Ok(Command::Build(options)) =
                parse(&[action, "--validate", "--capture", "kernels.gputrace"])
            else {
                panic!()
            };
            assert!(options.validate);
            assert_eq!(options.capture, Some(PathBuf::from("kernels.gputrace")));
        }
    }

    #[test]
    fn accepts_explicit_numerical_modes() {
        for mode in ["precise", "fast"] {
            let Ok(Command::Build(options)) = parse(&["build", "--math-mode", mode]) else {
                panic!()
            };
            assert_eq!(options.math_mode.as_str(), mode);
        }
        for arguments in [
            vec!["build", "--math-mode"],
            vec!["build", "--math-mode", "relaxed"],
            vec!["build", "--math-mode", "fast", "--math-mode", "precise"],
        ] {
            assert!(parse(&arguments).is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn rejects_ambiguous_or_unimplemented_options() {
        for args in [
            &["doctor", "--unknown"][..],
            &["build", "--"][..],
            &["inspect"][..],
            &["inspect", "--emit", "llvm"][..],
            &["build", "--validate"][..],
            &["test", "--validate", "--validate"][..],
            &["build", "--capture", "kernels.gputrace"][..],
            &["inspect", "--emit", "air", "--capture", "kernels.gputrace"][..],
            &["run", "--capture"][..],
            &["test", "--capture", "a.gputrace", "--capture", "b.gputrace"][..],
            &["inspect", "--emit", "ir", "--emit", "air"][..],
            &["run", "-p"][..],
            &["run", "-p", "a", "-p", "b"][..],
            &["run", "-p", ""][..],
            &["run", "--manifest-path", "a", "--manifest-path", "b"][..],
        ] {
            assert!(parse(args).is_err(), "{args:?}");
        }
    }
}

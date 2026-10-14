use std::path::PathBuf;

/// What `cargo metal` was asked to do.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Command {
    Help,
    Version,
    /// Installs the kernel toolchain, or with `check` only reports it.
    Setup {
        check: bool,
    },
    Cargo(Cargo),
    Inspect(Inspect),
}

/// `cargo run` or `cargo test` with Metal debugging.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Cargo {
    pub(crate) test: bool,
    /// Runs the host under Metal API and shader validation.
    pub(crate) validate: bool,
    /// Records the host's GPU work into this `.gputrace`, with line tables in the kernels.
    pub(crate) capture: Option<PathBuf>,
    /// Arguments for Cargo.
    pub(crate) arguments: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Emit {
    Ir,
    Msl,
    Air,
}

/// Prints the kernels that a package's build compiles.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Inspect {
    pub(crate) emit: Emit,
    /// Prints only the library that holds this kernel.
    pub(crate) kernel: Option<String>,
    /// Arguments for `cargo build`, such as `-p` and `--manifest-path`.
    pub(crate) arguments: Vec<String>,
}

pub(crate) fn parse(arguments: &[&str]) -> Result<Command, String> {
    let arguments = arguments.strip_prefix(&["metal"]).unwrap_or(arguments);
    match arguments {
        [] | ["--help" | "-h" | "help"] => Ok(Command::Help),
        ["--version" | "-V"] => Ok(Command::Version),
        ["setup"] => Ok(Command::Setup { check: false }),
        ["setup", "--check"] => Ok(Command::Setup { check: true }),
        [action @ ("run" | "test"), rest @ ..] => {
            cargo(*action == "test", rest).map(Command::Cargo)
        }
        ["inspect", rest @ ..] => inspect(rest).map(Command::Inspect),
        _ => Err("available commands: setup, run, test, inspect".into()),
    }
}

fn value<'a>(
    arguments: &mut impl Iterator<Item = &'a &'a str>,
    option: &str,
) -> Result<&'a str, String> {
    arguments
        .next()
        .copied()
        .filter(|value| !value.is_empty() && !value.starts_with('-'))
        .ok_or_else(|| format!("{option} requires a value"))
}

fn cargo(test: bool, arguments: &[&str]) -> Result<Cargo, String> {
    let mut options = Cargo {
        test,
        validate: false,
        capture: None,
        arguments: Vec::new(),
    };
    let mut arguments = arguments.iter();
    while let Some(&argument) = arguments.next() {
        match argument {
            "--" => {
                options.arguments.push(argument.into());
                options.arguments.extend(arguments.map(|v| (*v).to_owned()));
                break;
            }
            "--validate" if !options.validate => options.validate = true,
            "--capture" if options.capture.is_none() => {
                options.capture = Some(value(&mut arguments, argument)?.into());
            }
            "--validate" | "--capture" => return Err(format!("{argument} is given twice")),
            other => options.arguments.push(other.to_owned()),
        }
    }
    Ok(options)
}

fn inspect(arguments: &[&str]) -> Result<Inspect, String> {
    let mut emit = None;
    let mut kernel = None;
    let mut cargo = Vec::new();
    let mut arguments = arguments.iter();
    while let Some(&argument) = arguments.next() {
        match argument {
            "--emit" if emit.is_none() => {
                emit = Some(match value(&mut arguments, argument)? {
                    "ir" => Emit::Ir,
                    "msl" => Emit::Msl,
                    "air" => Emit::Air,
                    _ => return Err("--emit takes ir, msl or air".into()),
                });
            }
            "--kernel" if kernel.is_none() => {
                kernel = Some(value(&mut arguments, argument)?.to_owned());
            }
            "--emit" | "--kernel" => return Err(format!("{argument} is given twice")),
            other => cargo.push(other.to_owned()),
        }
    }
    Ok(Inspect {
        emit: emit.ok_or("inspect requires --emit ir|msl|air")?,
        kernel,
        arguments: cargo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_and_test_keep_cargo_arguments_and_take_metal_options() {
        let Ok(Command::Cargo(options)) = parse(&[
            "metal",
            "test",
            "-p",
            "vec-add",
            "--validate",
            "--release",
            "--capture",
            "kernels.gputrace",
            "--",
            "--ignored",
            "--validate",
        ]) else {
            panic!()
        };
        assert!(options.test && options.validate);
        assert_eq!(options.capture, Some(PathBuf::from("kernels.gputrace")));
        assert_eq!(
            options.arguments,
            [
                "-p",
                "vec-add",
                "--release",
                "--",
                "--ignored",
                "--validate"
            ]
        );
        let Ok(Command::Cargo(options)) = parse(&["run"]) else {
            panic!()
        };
        assert!(!options.test && !options.validate && options.arguments.is_empty());
    }

    #[test]
    fn inspect_takes_an_emit_kind_and_an_optional_kernel() {
        let Ok(Command::Inspect(options)) = parse(&[
            "inspect", "-p", "saxpy", "--emit", "msl", "--kernel", "saxpy",
        ]) else {
            panic!()
        };
        assert_eq!(options.emit, Emit::Msl);
        assert_eq!(options.kernel.as_deref(), Some("saxpy"));
        assert_eq!(options.arguments, ["-p", "saxpy"]);
        assert_eq!(
            parse(&["setup", "--check"]),
            Ok(Command::Setup { check: true })
        );
    }

    #[test]
    fn rejects_unknown_commands_and_repeated_or_incomplete_options() {
        for arguments in [
            &["build"][..],
            &["doctor"][..],
            &["inspect"][..],
            &["inspect", "--emit", "llvm"][..],
            &["inspect", "--emit", "ir", "--emit", "air"][..],
            &["inspect", "--emit", "ir", "--kernel"][..],
            &["run", "--capture"][..],
            &["test", "--validate", "--validate"][..],
            &["test", "--capture", "a.gputrace", "--capture", "b.gputrace"][..],
            &["setup", "--unknown"][..],
        ] {
            assert!(parse(arguments).is_err(), "{arguments:?}");
        }
    }
}

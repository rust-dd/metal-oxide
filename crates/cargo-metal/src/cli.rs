use clap::{Args, Parser, ValueEnum};
use std::{ffi::OsString, path::PathBuf};

/// Toolchain setup, Metal debugging and kernel inspection for metal-oxide.
#[derive(Debug, PartialEq, Eq, Parser)]
#[command(bin_name = "cargo metal", version)]
pub(crate) enum Command {
    /// Installs the pinned nightly, Metal Toolchain and metal-oxide compiler, then checks them.
    Setup {
        /// Only checks the toolchain.
        #[arg(long)]
        check: bool,
    },
    /// Runs `cargo run` under Metal API and shader validation, or recording its GPU work.
    Run(Cargo),
    /// Runs `cargo test` under Metal API and shader validation, or recording its GPU work.
    Test(Cargo),
    /// Builds the packages and prints the kernels their builds compiled.
    Inspect(Inspect),
}

#[derive(Debug, PartialEq, Eq, Args)]
pub(crate) struct Cargo {
    /// Runs the host under Metal API and shader validation.
    #[arg(long)]
    pub(crate) validate: bool,
    /// Records the first Metal device's GPU work with Rust line tables.
    #[arg(long, value_name = "FILE.gputrace")]
    pub(crate) capture: Option<PathBuf>,
    /// Arguments for Cargo.
    #[arg(value_name = "CARGO ARGUMENTS", allow_hyphen_values = true)]
    cargo: Vec<String>,
    /// Arguments for the program or the tests.
    #[arg(value_name = "ARGUMENTS", last = true)]
    program: Vec<String>,
}

impl Cargo {
    /// The arguments for Cargo in their given order. Clap takes a `--` right after the options as
    /// its own separator, while a later `--` stays among the Cargo arguments.
    pub(crate) fn arguments(&self) -> impl Iterator<Item = &str> {
        let separator = (!self.program.is_empty()).then_some("--");
        let program = self.program.iter().map(String::as_str);
        self.cargo
            .iter()
            .map(String::as_str)
            .chain(separator)
            .chain(program)
    }
}

#[derive(Debug, PartialEq, Eq, Args)]
pub(crate) struct Inspect {
    /// The form in which to print the kernels.
    #[arg(long)]
    pub(crate) emit: Emit,
    /// Prints only the library that holds this kernel.
    #[arg(long, value_name = "NAME")]
    pub(crate) kernel: Option<String>,
    /// Arguments for `cargo build`, such as `-p` and `--manifest-path`.
    #[arg(value_name = "CARGO BUILD ARGUMENTS", allow_hyphen_values = true)]
    pub(crate) arguments: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum Emit {
    /// The metal-oxide IR.
    Ir,
    /// The Metal Shading Language source.
    Msl,
    /// Apple's AIR, disassembled.
    Air,
}

/// Parses the arguments of `cargo-metal`, dropping the `metal` that Cargo passes first when it
/// runs `cargo metal`.
pub(crate) fn parse(
    arguments: impl IntoIterator<Item = impl Into<OsString>>,
) -> Result<Command, clap::Error> {
    let mut arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
    if arguments.get(1).is_some_and(|argument| argument == "metal") {
        arguments.remove(1);
    }
    Command::try_parse_from(arguments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    fn cli(arguments: &[&str]) -> Result<Command, clap::Error> {
        parse(std::iter::once("cargo-metal").chain(arguments.iter().copied()))
    }

    #[test]
    fn run_and_test_take_metal_options_and_pass_cargo_arguments_on_in_order() {
        let Ok(Command::Test(options)) = cli(&[
            "metal",
            "test",
            "--validate",
            "--capture",
            "kernels.gputrace",
            "-p",
            "vec-add",
            "--release",
            "--",
            "--ignored",
            "--validate",
        ]) else {
            panic!()
        };
        assert!(options.validate);
        assert_eq!(options.capture, Some(PathBuf::from("kernels.gputrace")));
        assert_eq!(
            options.arguments().collect::<Vec<_>>(),
            [
                "-p",
                "vec-add",
                "--release",
                "--",
                "--ignored",
                "--validate"
            ]
        );
        let Ok(Command::Test(options)) = cli(&["test", "--validate", "--", "--ignored"]) else {
            panic!()
        };
        assert_eq!(options.arguments().collect::<Vec<_>>(), ["--", "--ignored"]);
        let Ok(Command::Run(options)) = cli(&["run"]) else {
            panic!()
        };
        assert!(!options.validate && options.capture.is_none());
        assert_eq!(options.arguments().count(), 0);
    }

    #[test]
    fn inspect_takes_an_emit_kind_and_an_optional_kernel() {
        let Ok(Command::Inspect(options)) = cli(&[
            "inspect", "--emit", "msl", "--kernel", "saxpy", "-p", "saxpy",
        ]) else {
            panic!()
        };
        assert_eq!(options.emit, Emit::Msl);
        assert_eq!(options.kernel.as_deref(), Some("saxpy"));
        assert_eq!(options.arguments, ["-p", "saxpy"]);
        assert_eq!(
            cli(&["setup", "--check"]).ok(),
            Some(Command::Setup { check: true })
        );
    }

    #[test]
    fn rejects_missing_or_unknown_commands_and_repeated_or_incomplete_options() {
        for (arguments, kind) in [
            (&[][..], ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand),
            (&["build"][..], ErrorKind::InvalidSubcommand),
            (&["doctor"][..], ErrorKind::InvalidSubcommand),
            (&["inspect"][..], ErrorKind::MissingRequiredArgument),
            (&["inspect", "--emit", "llvm"][..], ErrorKind::InvalidValue),
            (
                &["inspect", "--emit", "ir", "--emit", "air"][..],
                ErrorKind::ArgumentConflict,
            ),
            (
                &["inspect", "--emit", "ir", "--kernel"][..],
                ErrorKind::InvalidValue,
            ),
            (&["run", "--capture"][..], ErrorKind::InvalidValue),
            (
                &["test", "--validate", "--validate"][..],
                ErrorKind::ArgumentConflict,
            ),
            (
                &["test", "--capture", "a.gputrace", "--capture", "b.gputrace"][..],
                ErrorKind::ArgumentConflict,
            ),
            (&["setup", "--unknown"][..], ErrorKind::UnknownArgument),
        ] {
            let Err(error) = cli(arguments) else {
                panic!("{arguments:?}")
            };
            assert_eq!(error.kind(), kind, "{arguments:?}");
        }
    }
}

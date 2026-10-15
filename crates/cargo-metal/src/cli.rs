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
mod tests;

mod cli;
mod compiler;
mod inspect;
mod run;
mod setup;

use cli::Command;
use std::process::ExitCode;

const HELP: &str = "\
cargo metal setup [--check]
    Installs the pinned nightly, Apple's Metal Toolchain and the metal-oxide compiler, then
    checks them; --check only checks.
cargo metal run|test [--validate] [--capture FILE.gputrace] [CARGO ARGUMENTS]
    cargo run or cargo test under Metal API and shader validation, or recording the GPU work
    of the first Metal device with Rust line tables.
cargo metal inspect --emit ir|msl|air [--kernel NAME] [CARGO BUILD ARGUMENTS]
    Builds the packages and prints the kernels their builds compiled.";

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let borrowed = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    let result = match cli::parse(&borrowed) {
        Ok(Command::Help) => {
            println!("{HELP}");
            return ExitCode::SUCCESS;
        }
        Ok(Command::Version) => {
            println!("cargo-metal {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Ok(Command::Setup { check }) => return setup::run(check),
        Ok(Command::Cargo(options)) => run::execute(&options),
        Ok(Command::Inspect(options)) => inspect::execute(&options).map(|()| ExitCode::SUCCESS),
        Err(error) => {
            eprintln!("cargo-metal: {error}");
            return ExitCode::from(2);
        }
    };
    result.unwrap_or_else(|error| {
        eprintln!("cargo-metal: {error}");
        ExitCode::FAILURE
    })
}

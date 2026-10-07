mod build;
mod cache;
mod cli;
mod doctor;
mod inputs;
mod metadata;
mod process;
mod rust;
mod toolchain;

use cli::Command;
use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let borrowed = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    match cli::parse(&borrowed) {
        Ok(Command::Doctor) => doctor::run(),
        Ok(Command::Help) => {
            println!(
                "cargo metal build|run|test [-p HOST] [--manifest-path PATH] [--release]\ncargo metal inspect --emit msl [-p HOST] [--manifest-path PATH]\ncargo metal doctor\n\nrun and test accept host arguments after --."
            );
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("cargo-metal {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Build(options)) => match build::execute(&options) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("cargo-metal: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("cargo-metal: {error}");
            ExitCode::from(2)
        }
    }
}

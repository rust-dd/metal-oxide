mod build;
mod cli;
mod compiler;
mod doctor;
mod metadata;

use cli::Command;
use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let borrowed = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    match cli::parse(&borrowed) {
        Ok(Command::Doctor) => doctor::run(),
        Ok(Command::Help) => {
            println!(
                "cargo metal build|run|test [-p HOST] [--manifest-path PATH] [--release] [--math-mode precise|fast]\ncargo metal inspect --emit ir|msl|air [-p HOST] [--manifest-path PATH] [--math-mode precise|fast]\ncargo metal doctor\n\nMath defaults to precise. Fast permits reassociation and contraction with fast math functions; arithmetic honors NaN and infinity.\nrun and test accept host arguments after --, and --validate enables Metal API and shader validation."
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

mod cli;
mod compiler;
mod inspect;
mod run;
mod setup;

use cli::Command;
use std::process::ExitCode;

fn main() -> ExitCode {
    let command = cli::parse(std::env::args_os()).unwrap_or_else(|error| error.exit());
    let result = match command {
        Command::Setup { check } => return setup::run(check),
        Command::Run(options) => run::execute("run", &options),
        Command::Test(options) => run::execute("test", &options),
        Command::Inspect(options) => inspect::execute(&options).map(|()| ExitCode::SUCCESS),
    };
    result.unwrap_or_else(|error| {
        eprintln!("cargo-metal: {error}");
        ExitCode::FAILURE
    })
}

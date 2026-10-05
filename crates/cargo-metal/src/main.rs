mod doctor;

use std::process::ExitCode;

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Doctor,
    Help,
    Version,
}

fn parse(arguments: &[&str]) -> Result<Command, String> {
    let arguments = arguments.strip_prefix(&["metal"]).unwrap_or(arguments);
    match arguments {
        [] | ["--help" | "-h" | "help"] => Ok(Command::Help),
        ["--version" | "-V"] => Ok(Command::Version),
        ["doctor"] => Ok(Command::Doctor),
        _ => Err("available commands: doctor, --help, --version".into()),
    }
}

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let borrowed = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    match parse(&borrowed) {
        Ok(Command::Doctor) => doctor::run(),
        Ok(Command::Help) => {
            println!(
                "cargo metal doctor\n\nCheck the Rust, Xcode, Metal toolchain, and GPU environment."
            );
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("cargo-metal {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cargo-metal: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, parse};

    #[test]
    fn accepts_cargo_and_direct_invocations() {
        assert_eq!(parse(&["metal", "doctor"]), Ok(Command::Doctor));
        assert_eq!(parse(&["doctor"]), Ok(Command::Doctor));
        assert_eq!(parse(&[]), Ok(Command::Help));
        assert_eq!(parse(&["--version"]), Ok(Command::Version));
    }

    #[test]
    fn rejects_unimplemented_commands_and_extra_arguments() {
        assert!(parse(&["build"]).is_err());
        assert!(parse(&["doctor", "--unknown"]).is_err());
    }
}

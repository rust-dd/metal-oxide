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

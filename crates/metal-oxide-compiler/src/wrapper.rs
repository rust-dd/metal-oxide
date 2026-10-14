use std::process::{Command, ExitCode};

fn option<'a>(arguments: &'a [String], name: &str) -> Option<&'a str> {
    arguments.iter().enumerate().find_map(|(index, value)| {
        if value == name {
            arguments.get(index + 1).map(String::as_str)
        } else {
            value.strip_prefix(&format!("{name}="))
        }
    })
}

pub(crate) fn forward(arguments: &mut Vec<String>) -> Option<ExitCode> {
    let target = std::env::var("METAL_OXIDE_CARGO_TARGET").ok()?;
    let rustc = arguments.get(1)?.clone();
    arguments.remove(1);
    if option(arguments, "--target") != Some(target.as_str()) {
        return Some(match Command::new(rustc).args(&arguments[1..]).status() {
            Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
            Err(error) => {
                eprintln!("error: cannot execute host rustc: {error}");
                ExitCode::FAILURE
            }
        });
    }
    for argument in arguments.iter_mut() {
        if let Some(emit) = argument.strip_prefix("--emit=") {
            let outputs = emit.split(',').filter(|v| *v != "link").collect::<Vec<_>>();
            *argument = format!("--emit={}", outputs.join(","));
        }
        if argument.ends_with(".rlib") && argument.contains('=') {
            *argument = format!("{}rmeta", argument.strip_suffix("rlib").unwrap());
        }
    }
    let kernel = std::env::var("METAL_OXIDE_CARGO_KERNEL").ok();
    if kernel.as_deref() == option(arguments, "--crate-name")
        && option(arguments, "--emit").is_some_and(|emit| emit.split(',').any(|v| v == "metadata"))
        && let Ok(output) = std::env::var("METAL_OXIDE_CARGO_OUTPUT")
    {
        arguments.extend(["--metal-output".into(), output]);
        if let Ok(mode) = std::env::var("METAL_OXIDE_CARGO_MATH_MODE") {
            arguments.extend(["--metal-math-mode".into(), mode]);
        }
        // Kernels copied from a host crate keep its positions, so their crate root cannot
        // declare `no_std`; the host build already reports their warnings.
        if std::env::var_os("METAL_OXIDE_CARGO_SOURCE").is_some() {
            arguments.extend([
                "-Zcrate-attr=no_std".into(),
                "-Zcrate-attr=allow(warnings)".into(),
            ]);
        }
    }
    None
}

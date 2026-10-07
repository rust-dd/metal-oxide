#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_codegen_ssa;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

mod backend;
mod collect;
mod driver;
mod import;
mod intrinsics;
mod metadata_abi;
mod monomorphize;
mod output;
mod target;
mod types;
mod wrapper;

fn main() -> std::process::ExitCode {
    let mut arguments = std::env::args().collect::<Vec<_>>();
    if let Some(status) = wrapper::forward(&mut arguments) {
        return status;
    }
    let output = match output::take_directory(&mut arguments) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("error: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    rustc_driver::catch_with_exit_code(|| {
        let expanded = {
            let diagnostics = rustc_session::EarlyDiagCtxt::new(
                rustc_session::config::ErrorOutputType::default(),
            );
            rustc_driver::args::arg_expand_all(&diagnostics, &arguments[1..])
        };
        let help = expanded.is_empty()
            || expanded
                .iter()
                .enumerate()
                .take_while(|(_, argument)| argument.as_str() != "--")
                .any(|(index, argument)| {
                    matches!(argument.as_str(), "--help" | "-h" | "-Chelp" | "-Zhelp")
                        || (argument == "help"
                            && index > 0
                            && matches!(expanded[index - 1].as_str(), "-C" | "-Z"))
                });
        rustc_driver::compiler_entrypoint(&arguments, &mut driver::Frontend { output, help });
    })
}

fn trace(arguments: std::fmt::Arguments<'_>) {
    if std::env::var_os("METAL_OXIDE_CARGO_TARGET").is_none() {
        println!("{arguments}");
    }
}

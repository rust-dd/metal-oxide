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

fn main() -> std::process::ExitCode {
    let mut arguments = std::env::args().collect::<Vec<_>>();
    let output = match output::take_directory(&mut arguments) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("error: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Some(directory) = &output
        && let Err(error) = output::prepare(directory)
    {
        eprintln!("error: {error}");
        return std::process::ExitCode::FAILURE;
    }
    rustc_driver::catch_with_exit_code(|| {
        rustc_driver::compiler_entrypoint(&arguments, &mut driver::Frontend { output });
    })
}

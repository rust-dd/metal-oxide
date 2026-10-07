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

mod arguments;
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
    let output = match arguments::take_directory(&mut arguments) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("error: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    rustc_driver::catch_with_exit_code(|| {
        let help = arguments::help(&arguments);
        rustc_driver::compiler_entrypoint(&arguments, &mut driver::Frontend { output, help });
    })
}

fn trace(emit: impl FnOnce()) {
    if std::env::var_os("METAL_OXIDE_CARGO_TARGET").is_none() {
        emit();
    }
}

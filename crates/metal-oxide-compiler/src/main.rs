#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_codegen_ssa;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_target;

mod backend;
mod driver;
mod metadata_abi;
mod target;

fn main() -> std::process::ExitCode {
    rustc_driver::catch_with_exit_code(|| {
        let arguments = std::env::args().collect::<Vec<_>>();
        rustc_driver::compiler_entrypoint(&arguments, &mut driver::Frontend);
    })
}

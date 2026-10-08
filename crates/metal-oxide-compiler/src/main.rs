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
    if arguments.len() == 2 && arguments[1] == "--metal-compiler-info" {
        return compiler_info();
    }
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
        if !help {
            arguments.insert(1, "-Zunstable-options".into());
        }
        let mut frontend = driver::Frontend {
            output,
            help,
            arguments: arguments[1..].to_vec(),
        };
        rustc_driver::compiler_entrypoint(&arguments, &mut frontend);
    })
}

fn compiler_info() -> std::process::ExitCode {
    let result = || -> Result<String, Box<dyn std::error::Error>> {
        let binary = std::fs::read(std::env::current_exe()?)?;
        Ok(metal_oxide_artifact::CompilerInfo::new(
            include_str!(concat!(env!("OUT_DIR"), "/rustc-version")),
            &binary,
        )
        .to_json()?)
    };
    match result() {
        Ok(info) => {
            println!("{info}");
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("compiler identity: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn trace(emit: impl FnOnce()) {
    if std::env::var_os("METAL_OXIDE_CARGO_TARGET").is_none() {
        emit();
    }
}

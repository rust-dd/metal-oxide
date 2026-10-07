use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::{Compiler, Config};
use rustc_middle::ty::TyCtxt;

pub(crate) struct Frontend {
    pub(crate) output: Option<std::path::PathBuf>,
}

impl Callbacks for Frontend {
    fn config(&mut self, config: &mut Config) {
        config.make_codegen_backend = Some(Box::new(|_| Box::new(crate::backend::FrontendBackend)));
        config.opts.unstable_opts.crate_attr.extend([
            "feature(register_tool,rustc_attrs)".into(),
            "register_tool(metal_oxide)".into(),
            "allow(internal_features)".into(),
        ]);
        config.opts.unstable_opts.mir_opt_level = Some(0);
        config.opts.unstable_opts.always_encode_mir = true;
        config.opts.unstable_opts.enforce_type_length_limit = true;
    }

    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        if let Some(directory) = &self.output
            && let Err(error) = crate::output::prepare(directory)
        {
            tcx.dcx().fatal(error.to_string());
        }
        crate::target::validate(tcx);
        let (entries, instances) = crate::collect::kernels(tcx);
        tcx.dcx().abort_if_errors();
        if let Some(directory) = &self.output {
            match crate::import::ModuleImporter::new(tcx).import(&entries, &instances) {
                Ok(module) => {
                    if let Err(error) = crate::output::write(directory, &module, tcx, &entries) {
                        tcx.dcx().err(error.to_string());
                    }
                }
                Err((span, error)) => {
                    tcx.dcx().span_err(span, error);
                }
            }
            tcx.dcx().abort_if_errors();
        }
        Compilation::Continue
    }
}

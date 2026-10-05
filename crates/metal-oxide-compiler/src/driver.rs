use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::{Compiler, Config};
use rustc_middle::ty::TyCtxt;

pub(crate) struct Frontend;

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
    }

    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        crate::target::validate(tcx);
        tcx.dcx().abort_if_errors();
        Compilation::Continue
    }
}

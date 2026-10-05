use std::any::Any;

use rustc_codegen_ssa::{CompiledModules, CrateInfo, TargetConfig, traits::CodegenBackend};
use rustc_middle::{dep_graph::WorkProductMap, ty::TyCtxt};
use rustc_session::{EarlySession, IncrCompSession, Session, config::OutputFilenames};

pub(crate) struct FrontendBackend;

impl CodegenBackend for FrontendBackend {
    fn name(&self) -> &'static str {
        "metal-oxide-frontend"
    }

    fn target_config(&self, _session: &EarlySession) -> TargetConfig {
        TargetConfig {
            internal_target_features: Default::default(),
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f16b: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn target_cpu(&self, _session: &Session) -> String {
        "generic".into()
    }

    fn provide(&self, providers: &mut rustc_middle::util::Providers) {
        crate::metadata_abi::provide(providers);
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        tcx.dcx()
            .fatal("native code generation is not supported; MSL emission is not implemented yet")
    }

    fn join_codegen(
        &self,
        _ongoing: Box<dyn Any>,
        session: &Session,
        _incremental: Option<&IncrCompSession>,
        _outputs: &OutputFilenames,
        _info: &CrateInfo,
    ) -> (CompiledModules, WorkProductMap) {
        session
            .dcx()
            .fatal("native code generation is not supported")
    }
}

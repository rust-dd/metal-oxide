use std::sync::OnceLock;

use rustc_hir::def::DefKind;
use rustc_middle::{
    ty::{self, Ty, TyCtxt, layout::FnAbiError},
    util::Providers,
};
use rustc_target::callconv::{AbiMap, ArgAbi, ArgAttributes, FnAbi};

type AbiResult<'tcx> = Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>>;
type PointerQuery<'tcx> =
    ty::PseudoCanonicalInput<'tcx, (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>)>;
type InstanceQuery<'tcx> =
    ty::PseudoCanonicalInput<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>;
type PointerProvider = for<'tcx> fn(TyCtxt<'tcx>, PointerQuery<'tcx>) -> AbiResult<'tcx>;
type InstanceProvider = for<'tcx> fn(TyCtxt<'tcx>, InstanceQuery<'tcx>) -> AbiResult<'tcx>;

static POINTER_PROVIDER: OnceLock<PointerProvider> = OnceLock::new();
static INSTANCE_PROVIDER: OnceLock<InstanceProvider> = OnceLock::new();

pub(crate) fn provide(providers: &mut Providers) {
    POINTER_PROVIDER.get_or_init(|| providers.queries.fn_abi_of_fn_ptr);
    INSTANCE_PROVIDER.get_or_init(|| providers.queries.fn_abi_of_instance_no_deduced_attrs);
    providers.queries.fn_abi_of_fn_ptr = pointer_abi;
    providers.queries.fn_abi_of_instance_no_deduced_attrs = instance_abi;
}

fn pointer_abi<'tcx>(tcx: TyCtxt<'tcx>, query: PointerQuery<'tcx>) -> AbiResult<'tcx> {
    let signature = tcx.instantiate_bound_regions_with_erased(query.value.0);
    if signature.abi().is_rustic_abi() {
        return POINTER_PROVIDER.get().unwrap()(tcx, query);
    }
    foreign_metadata_abi(tcx, query.typing_env, signature, query.value.1)
}

fn instance_abi<'tcx>(tcx: TyCtxt<'tcx>, query: InstanceQuery<'tcx>) -> AbiResult<'tcx> {
    let instance = query.value.0;
    if let ty::InstanceKind::Item(definition) = instance.def
        && matches!(tcx.def_kind(definition), DefKind::Fn | DefKind::AssocFn)
    {
        let signature = tcx
            .fn_sig(definition)
            .instantiate(tcx, instance.args)
            .skip_binder();
        if !signature.abi().is_rustic_abi() {
            return foreign_metadata_abi(tcx, query.typing_env, signature, query.value.1);
        }
    }
    INSTANCE_PROVIDER.get().unwrap()(tcx, query)
}

fn foreign_metadata_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    environment: ty::TypingEnv<'tcx>,
    signature: ty::FnSig<'tcx>,
    extra: &[Ty<'tcx>],
) -> AbiResult<'tcx> {
    // core needs foreign function layouts for metadata/CTFE; this never defines a callable GPU ABI.
    let argument = |ty| {
        let layout = tcx
            .layout_of(environment.as_query_input(ty))
            .map_err(|error| &*tcx.arena.alloc(FnAbiError::Layout(*error)))?;
        Ok(ArgAbi::new(layout, |_, _| ArgAttributes::new()))
    };
    let abi = FnAbi {
        ret: argument(signature.output())?,
        args: signature
            .inputs()
            .iter()
            .chain(extra)
            .copied()
            .map(argument)
            .collect::<Result<_, _>>()?,
        c_variadic: signature.c_variadic(),
        fixed_count: signature.inputs().len() as u32,
        conv: AbiMap::from_target(&tcx.sess.target)
            .canonize_abi(signature.abi(), signature.c_variadic())
            .unwrap(),
        can_unwind: false,
    };
    Ok(tcx.arena.alloc(abi))
}

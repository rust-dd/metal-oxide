use metal_oxide_ir::{
    Access, AtomicOp, BinaryOp, Builtin, IntegerOp, MathOp, MatrixOp, Scalar, SimdBuiltin, SimdOp,
};
mod device;
mod signatures;
mod vector;
pub(crate) use vector::VectorOp;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::sym;

pub(crate) use device::{DeviceType, device_type};

#[derive(Clone, Copy)]
pub(crate) enum Intrinsic {
    Coordinates(Builtin),
    SimdCoordinate(SimdBuiltin),
    ThreadIndex,
    BufferLoad,
    BufferStore,
    ThreadgroupAlloc(Access),
    ThreadgroupBarrier,
    Uninit,
    Simd(SimdOp),
    Matrix(MatrixOp),
    Vector(VectorOp),
    Atomic(AtomicOp),
    WrappingShift(BinaryOp),
    Math(MathOp),
    Integer(IntegerOp),
    Bitcast(Scalar),
    FloatConvert {
        from: Scalar,
        to: Scalar,
    },
    FloatBinary(Scalar, BinaryOp),
    UncheckedFloatToInt,
    Log,
    /// `<[T]>::len`.
    SliceLen,
    /// `Index::index` or `IndexMut::index_mut` on a slice or array; the index type selects the
    /// range.
    Subslice,
    /// A hint or adapter that returns its argument.
    Identity,
    /// The conservative result of `is_val_statically_known`.
    StaticallyKnown,
    /// `len`, `is_empty` and `as_slice` of a kernel buffer parameter.
    Buffer(BufferQuery),
    /// `ptr_offset_from_unsigned`: the element count between two pointers.
    OffsetFrom,
    /// `ptr::swap` and `typed_swap_nonoverlapping`, behind `mem::swap`: exchanges two pointees.
    Swap,
    /// `ptr::copy_nonoverlapping`: copies `count` elements, which its caller keeps valid.
    CopyNonoverlapping,
    /// `IntoIterator::into_iter` of an array: an iterator at its first element.
    ArrayIntoIter,
    /// `<[T; N]>::map` (`true`) or `array::from_fn` (`false`): a loop that calls a closure.
    ArrayFromClosure(bool),
}

/// The iterators whose `Iterator::next` the importer expands inline, so the prover sees the
/// loop bound or the chunk length; other iterator methods run core's MIR.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Expanded {
    /// `Range<T>` of an integer `T`.
    Range,
    /// `core::slice::ChunksExact`.
    Chunks,
    /// `core::array::IntoIter`, held as its array and the index of the next element.
    Array,
}

impl Expanded {
    pub(crate) fn of(tcx: TyCtxt<'_>, ty: Ty<'_>) -> Option<Self> {
        let ty::Adt(definition, arguments) = ty.kind() else {
            return None;
        };
        let did = definition.did();
        if tcx.lang_items().range_struct() == Some(did) {
            return arguments.type_at(0).is_integral().then_some(Self::Range);
        }
        if tcx.crate_name(did.krate).as_str() != "core" {
            return None;
        }
        let path = tcx.def_path_str(did);
        match tcx.item_name(did).as_str() {
            "ChunksExact" if path.starts_with("core::slice::") => Some(Self::Chunks),
            "IntoIter" if path.starts_with("core::array::") => Some(Self::Array),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferQuery {
    Length,
    IsEmpty,
    Slice,
    /// `DisjointBuffer::get_mut` with the calling thread's index.
    Element,
}

impl Intrinsic {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Coordinates(Builtin::ThreadIdx) => "thread_idx",
            Self::Coordinates(Builtin::BlockIdx) => "block_idx",
            Self::Coordinates(Builtin::BlockDim) => "block_dim",
            Self::Coordinates(Builtin::GridDim) => "grid_dim",
            Self::SimdCoordinate(SimdBuiltin::Lane) => "simd_lane",
            Self::SimdCoordinate(SimdBuiltin::Size) => "simd_size",
            Self::SimdCoordinate(SimdBuiltin::Group) => "simd_group",
            Self::SimdCoordinate(SimdBuiltin::Count) => "simd_count",
            Self::ThreadIndex => "thread_index",
            Self::BufferLoad => "buffer_load",
            Self::BufferStore => "buffer_store",
            Self::ThreadgroupAlloc(_) => "threadgroup_alloc",
            Self::ThreadgroupBarrier => "threadgroup_barrier",
            Self::Uninit => "uninit_array",
            Self::Simd(SimdOp::Sum) => "simd_sum",
            Self::Simd(SimdOp::Shuffle) => "simd_shuffle",
            Self::Simd(_) => "simd",
            Self::Matrix(_) => "simdgroup_matrix",
            Self::Vector(_) => "vector",
            Self::Atomic(AtomicOp::Add) => "atomic_add",
            Self::Atomic(_) => "atomic",
            Self::WrappingShift(_) => "wrapping_shift",
            Self::Math(_) => "float_math",
            Self::Integer(_) => "integer_op",
            Self::Bitcast(_) => "scalar_bitcast",
            Self::FloatConvert { .. } => "float_conversion",
            Self::FloatBinary(..) => "float_arithmetic",
            Self::UncheckedFloatToInt => "unchecked_float_to_int",
            Self::Log => "log",
            Self::SliceLen => "slice_len",
            Self::Subslice => "subslice",
            Self::Identity => "identity",
            Self::StaticallyKnown => "statically_known",
            Self::Buffer(BufferQuery::Length) => "buffer_length",
            Self::Buffer(BufferQuery::IsEmpty) => "buffer_is_empty",
            Self::Buffer(BufferQuery::Slice) => "buffer_slice",
            Self::Buffer(BufferQuery::Element) => "disjoint_element",
            Self::OffsetFrom => "offset_from",
            Self::Swap => "swap",
            Self::CopyNonoverlapping => "copy_nonoverlapping",
            Self::ArrayIntoIter => "array_into_iter",
            Self::ArrayFromClosure(_) => "array_from_closure",
        }
    }
}

/// Core entry points of a panic with a literal message; the importer replaces them with a
/// `Panic` terminator, so their bodies are never imported.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanicEntry {
    Panic,
    PanicFmt,
    LiteralArguments,
    /// A core helper that panics with the message of its argument at this index.
    Argument(usize),
    /// A core helper that panics with this message.
    Fixed(&'static str),
}

pub(crate) fn panic_entry(tcx: TyCtxt<'_>, definition: DefId) -> Option<PanicEntry> {
    let items = tcx.lang_items();
    if items.panic_fn() == Some(definition) {
        return Some(PanicEntry::Panic);
    }
    if items.panic_fmt() == Some(definition) {
        return Some(PanicEntry::PanicFmt);
    }
    if tcx.crate_name(definition.krate).as_str() == "core"
        && tcx.item_name(definition).as_str() == "do_panic"
        && tcx.item_name(tcx.parent(definition)).as_str() == "clamp"
    {
        let signature = tcx.fn_sig(definition).instantiate_identity().skip_binder();
        if signature.output().is_never() {
            return Some(PanicEntry::Fixed(
                if signature
                    .inputs()
                    .iter()
                    .any(|ty| matches!(ty.kind(), ty::Float(_)))
                {
                    "min > max, or either was NaN"
                } else {
                    "min > max"
                },
            ));
        }
    }
    // The cold helpers of `unwrap` and `expect`; a `Result`'s error is not formatted.
    match tcx.def_path_str(definition).as_str() {
        "core::option::unwrap_failed" => {
            return Some(PanicEntry::Fixed(
                "called `Option::unwrap()` on a `None` value",
            ));
        }
        "core::option::expect_failed" | "core::result::unwrap_failed" => {
            return Some(PanicEntry::Argument(0));
        }
        "core::num::imp::overflow_panic::pow" => {
            return Some(PanicEntry::Fixed("attempt to exponentiate with overflow"));
        }
        "core::num::imp::int_log10::panic_for_nonpositive_argument" => {
            return Some(PanicEntry::Fixed(
                "argument of integer logarithm must be positive",
            ));
        }
        "core::num::imp::int_sqrt::panic_for_negative_argument" => {
            return Some(PanicEntry::Fixed(
                "argument of integer square root cannot be negative",
            ));
        }
        // `const_panic!` formats the lengths only outside const evaluation; this is its const message.
        "core::slice::copy_from_slice_impl::len_mismatch_fail" => {
            return Some(PanicEntry::Fixed(
                "copy_from_slice: source slice length does not match destination slice length",
            ));
        }
        _ => {}
    }
    let implementation = tcx.inherent_impl_of_assoc(definition)?;
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    (items.format_arguments() == Some(ty.ty_adt_def()?.did())
        && tcx.item_name(definition).as_str() == "from_str")
        .then_some(PanicEntry::LiteralArguments)
}

/// The iterator whose `Iterator::next` a call invokes, if the importer expands it.
pub(crate) fn expanded_next<'tcx>(
    tcx: TyCtxt<'tcx>,
    definition: DefId,
    arguments: ty::GenericArgsRef<'tcx>,
) -> Option<Expanded> {
    if tcx.lang_items().next_fn() != Some(definition) {
        return None;
    }
    Expanded::of(tcx, arguments.type_at(0))
}

pub(crate) fn builtin(tcx: TyCtxt<'_>, definition: DefId) -> Option<Intrinsic> {
    if let Some(operation) = device::operation(tcx, definition) {
        if !signatures::valid(tcx, definition, operation) {
            tcx.dcx().span_fatal(
                tcx.def_span(definition),
                "invalid device intrinsic signature",
            );
        }
        return Some(operation);
    }
    if tcx.crate_name(definition.krate).as_str() != "core" {
        return None;
    }
    if let Some(operation) = vector::operation(tcx, definition) {
        return Some(Intrinsic::Vector(operation));
    }
    if matches!(
        tcx.def_path_str(definition).as_str(),
        "core::intrinsics::likely" | "core::intrinsics::unlikely"
    ) {
        return Some(Intrinsic::Identity);
    }
    // Their bodies copy through a `MaybeUninit` or the `copy_nonoverlapping` statement, which
    // the importer replaces with element copies.
    match tcx.def_path_str(definition).as_str() {
        "core::array::from_fn" => return Some(Intrinsic::ArrayFromClosure(false)),
        "core::ptr::swap" => return Some(Intrinsic::Swap),
        "core::ptr::copy_nonoverlapping" => return Some(Intrinsic::CopyNonoverlapping),
        _ => {}
    }
    if let Some(intrinsic) = tcx.intrinsic(definition) {
        if intrinsic.name.as_str().starts_with("simd_") {
            return Some(Intrinsic::Vector(VectorOp::Intrinsic(intrinsic.name)));
        }
        return Some(Intrinsic::Integer(match intrinsic.name.as_str() {
            "integer_min" => IntegerOp::Min,
            "integer_max" => IntegerOp::Max,
            "saturating_add" => IntegerOp::SaturatingAdd,
            "saturating_sub" => IntegerOp::SaturatingSub,
            "ctlz" | "ctlz_nonzero" => IntegerOp::LeadingZeros,
            "cttz" | "cttz_nonzero" => IntegerOp::TrailingZeros,
            "ctpop" => IntegerOp::CountOnes,
            "bitreverse" => IntegerOp::ReverseBits,
            "bswap" => IntegerOp::SwapBytes,
            "rotate_left" => IntegerOp::RotateLeft,
            "rotate_right" => IntegerOp::RotateRight,
            "ptr_offset_from_unsigned" => return Some(Intrinsic::OffsetFrom),
            "copysignf32" => return Some(Intrinsic::Math(MathOp::CopySign)),
            "is_val_statically_known" => return Some(Intrinsic::StaticallyKnown),
            "typed_swap_nonoverlapping" => return Some(Intrinsic::Swap),
            _ => return None,
        }));
    }
    if let Some(index) = slice_index(tcx, definition) {
        return Some(index);
    }
    if iterator_identity(tcx, definition) {
        return Some(Intrinsic::Identity);
    }
    if array_into_iter(tcx, definition) {
        return Some(Intrinsic::ArrayIntoIter);
    }
    let implementation = tcx.inherent_impl_of_assoc(definition)?;
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    if matches!(ty.kind(), ty::Slice(_)) && tcx.item_name(definition).as_str() == "len" {
        return Some(Intrinsic::SliceLen);
    }
    if matches!(ty.kind(), ty::Array(..)) && tcx.item_name(definition).as_str() == "map" {
        return Some(Intrinsic::ArrayFromClosure(true));
    }
    if matches!(ty.kind(), ty::Float(ty::FloatTy::F32)) {
        match tcx.item_name(definition).as_str() {
            "abs" => return Some(Intrinsic::Math(MathOp::Abs)),
            "min" => return Some(Intrinsic::Math(MathOp::Min)),
            "max" => return Some(Intrinsic::Math(MathOp::Max)),
            "to_bits" => return Some(Intrinsic::Bitcast(Scalar::U32)),
            "from_bits" => return Some(Intrinsic::Bitcast(Scalar::F32)),
            "to_int_unchecked" => return Some(Intrinsic::UncheckedFloatToInt),
            _ => {}
        }
    }
    if matches!(ty.kind(), ty::Int(_) | ty::Uint(_)) {
        match tcx.item_name(definition).as_str() {
            "wrapping_shl" => return Some(Intrinsic::WrappingShift(BinaryOp::Shl)),
            "wrapping_shr" => return Some(Intrinsic::WrappingShift(BinaryOp::Shr)),
            _ => {}
        }
    }
    None
}

/// The `Index` and `IndexMut` implementations of slices and arrays, which index by ranges.
fn slice_index(tcx: TyCtxt<'_>, definition: DefId) -> Option<Intrinsic> {
    let items = tcx.lang_items();
    let implementation = tcx.impl_of_assoc(definition)?;
    let trait_ = tcx.impl_opt_trait_id(implementation)?;
    if Some(trait_) != items.index_trait() && Some(trait_) != items.index_mut_trait() {
        return None;
    }
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    matches!(ty.kind(), ty::Slice(_) | ty::Array(..)).then_some(Intrinsic::Subslice)
}

/// Whether `definition` is a method of `core::array::IntoIter`, whose `MaybeUninit` elements have
/// no device representation; the importer expands its `next` instead.
pub(crate) fn array_iterator_method(tcx: TyCtxt<'_>, definition: DefId) -> bool {
    tcx.impl_of_assoc(definition).is_some_and(|implementation| {
        let ty = tcx.normalize_erasing_regions(
            ty::TypingEnv::non_body_analysis(tcx, implementation),
            tcx.type_of(implementation).instantiate_identity(),
        );
        Expanded::of(tcx, ty) == Some(Expanded::Array)
    })
}

/// Whether dropping `ty` runs code. An array iterator drops only the elements it has not
/// yielded, so it, and a value that holds it, need no drop when those elements need none.
pub(crate) fn needs_drop<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let environment = ty::TypingEnv::fully_monomorphized();
    if !ty.needs_drop(tcx, environment) {
        return false;
    }
    match ty.kind() {
        ty::Adt(_, arguments) if Expanded::of(tcx, ty) == Some(Expanded::Array) => {
            needs_drop(tcx, arguments.type_at(0))
        }
        ty::Adt(definition, arguments) => {
            definition.has_dtor(tcx)
                || definition.all_fields().any(|field| {
                    needs_drop(
                        tcx,
                        tcx.normalize_erasing_regions(environment, field.ty(tcx, arguments)),
                    )
                })
        }
        ty::Tuple(fields) => fields.iter().any(|field| needs_drop(tcx, field)),
        ty::Array(element, _) => needs_drop(tcx, *element),
        ty::Closure(_, arguments) => arguments
            .as_closure()
            .upvar_tys()
            .iter()
            .any(|upvar| needs_drop(tcx, upvar)),
        _ => true,
    }
}

/// The closure that `<[T; N]>::map::<F, U>` (`mapped`) or `array::from_fn::<T, N, F>` calls
/// through `FnMut`, and what it takes: an element or an index. Core fills the array through
/// `MaybeUninit` and a drop guard.
pub(crate) fn array_closure<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    mapped: bool,
) -> Option<(ty::Instance<'tcx>, Ty<'tcx>)> {
    let input = if mapped {
        instance.args.type_at(0)
    } else {
        tcx.types.usize
    };
    let call_mut = tcx
        .associated_item_def_ids(tcx.lang_items().fn_mut_trait()?)
        .iter()
        .copied()
        .find(|item| tcx.item_name(*item).as_str() == "call_mut")?;
    let arguments = tcx.mk_args(&[
        instance.args.type_at(2).into(),
        Ty::new_tup(tcx, &[input]).into(),
    ]);
    let closure = crate::monomorphize::resolve(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        call_mut,
        arguments,
    )
    .ok()??;
    tcx.is_closure_like(closure.def_id())
        .then_some((closure, input))
}

/// `<[T; N] as IntoIterator>::into_iter`; core's iterator holds `MaybeUninit` elements.
fn array_into_iter(tcx: TyCtxt<'_>, definition: DefId) -> bool {
    tcx.opt_item_name(definition)
        .is_some_and(|name| name.as_str() == "into_iter")
        && tcx.impl_of_assoc(definition).is_some_and(|implementation| {
            let ty = tcx.normalize_erasing_regions(
                ty::TypingEnv::non_body_analysis(tcx, implementation),
                tcx.type_of(implementation).instantiate_identity(),
            );
            tcx.impl_opt_trait_id(implementation) == tcx.get_diagnostic_item(sym::IntoIterator)
                && matches!(ty.kind(), ty::Array(..))
        })
}

/// The blanket `impl<I: Iterator> IntoIterator for I`, whose `into_iter` returns `self`.
fn iterator_identity(tcx: TyCtxt<'_>, definition: DefId) -> bool {
    let Some(implementation) = tcx.impl_of_assoc(definition) else {
        return false;
    };
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    tcx.lang_items()
        .into_iter_fn()
        .is_some_and(|method| tcx.associated_item(definition).trait_item_def_id() == Some(method))
        && matches!(ty.kind(), ty::Param(_))
}

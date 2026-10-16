use metal_oxide_ir::{
    Access, AtomicOp, BinaryOp, Builtin, IntegerOp, MathOp, Scalar, SimdBuiltin, SimdOp,
};
mod device;
mod signatures;
mod vector;
pub(crate) use vector::VectorOp;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyCtxt};

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
    /// `IntoIterator::into_iter` of an iterator, which returns its argument.
    Identity,
    /// `len`, `is_empty` and `as_slice` of a kernel buffer parameter.
    Buffer(BufferQuery),
    /// `ptr_offset_from_unsigned`: the element count between two pointers.
    OffsetFrom,
}

/// The iterators whose `Iterator::next` the importer expands inline, so the prover sees the
/// loop bound or the chunk length; other iterator methods run core's MIR.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Expanded {
    /// `Range<T>` of an integer `T`.
    Range,
    /// `core::slice::ChunksExact`.
    Chunks,
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
        (tcx.crate_name(did.krate).as_str() == "core"
            && tcx.def_path_str(did).starts_with("core::slice::")
            && tcx.item_name(did).as_str() == "ChunksExact")
            .then_some(Self::Chunks)
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
            Self::Buffer(BufferQuery::Length) => "buffer_length",
            Self::Buffer(BufferQuery::IsEmpty) => "buffer_is_empty",
            Self::Buffer(BufferQuery::Slice) => "buffer_slice",
            Self::Buffer(BufferQuery::Element) => "disjoint_element",
            Self::OffsetFrom => "offset_from",
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
    if let Some(intrinsic) = tcx.intrinsic(definition) {
        if intrinsic.name.as_str().starts_with("simd_") {
            return Some(Intrinsic::Vector(VectorOp::Intrinsic(intrinsic.name)));
        }
        return Some(Intrinsic::Integer(match intrinsic.name.as_str() {
            "integer_min" => IntegerOp::Min,
            "integer_max" => IntegerOp::Max,
            "saturating_add" => IntegerOp::SaturatingAdd,
            "saturating_sub" => IntegerOp::SaturatingSub,
            "ctlz" => IntegerOp::LeadingZeros,
            "cttz" => IntegerOp::TrailingZeros,
            "ctpop" => IntegerOp::CountOnes,
            "bitreverse" => IntegerOp::ReverseBits,
            "rotate_left" => IntegerOp::RotateLeft,
            "rotate_right" => IntegerOp::RotateRight,
            "ptr_offset_from_unsigned" => return Some(Intrinsic::OffsetFrom),
            _ => return None,
        }));
    }
    if let Some(index) = slice_index(tcx, definition) {
        return Some(index);
    }
    if iterator_identity(tcx, definition) {
        return Some(Intrinsic::Identity);
    }
    let implementation = tcx.inherent_impl_of_assoc(definition)?;
    let ty = tcx.normalize_erasing_regions(
        ty::TypingEnv::non_body_analysis(tcx, implementation),
        tcx.type_of(implementation).instantiate_identity(),
    );
    if matches!(ty.kind(), ty::Slice(_)) && tcx.item_name(definition).as_str() == "len" {
        return Some(Intrinsic::SliceLen);
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

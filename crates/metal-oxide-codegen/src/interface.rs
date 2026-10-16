use metal_oxide_artifact::{self as artifact, ParameterType};
use metal_oxide_ir::{
    Access, AddressSpace, Error, Function, FunctionAnalysis, Module, Scalar, Type,
};

#[derive(Debug)]
pub(crate) struct KernelInterfaces {
    pub(crate) kernels: Vec<Option<artifact::Kernel>>,
    pub(crate) simd: bool,
    pub(crate) logging: bool,
    atomics: bool,
    dynamic_shared: bool,
    bfloat: bool,
}

impl KernelInterfaces {
    pub(crate) fn new(module: &Module, analyses: &[FunctionAnalysis]) -> Result<Self, Error> {
        let kernels = module
            .functions
            .iter()
            .zip(analyses)
            .map(|(function, analysis)| {
                if !function.kernel {
                    return Ok(None);
                }
                crate::names::validate_name(function)?;
                let mut slots = [0; 2];
                let parameters = function.locals[1..1 + function.parameters]
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| {
                        let ty = match *ty {
                            Type::Scalar(_) | Type::Aggregate(_) => ParameterType::Value {
                                layout: layout(module, *ty, function)?,
                            },
                            Type::Buffer {
                                element,
                                address_space: AddressSpace::Threadgroup,
                                ..
                            } => {
                                let element = layout(module, element.ty(), function)?;
                                ParameterType::Threadgroup {
                                    stride: element.size,
                                    element,
                                }
                            }
                            Type::Buffer {
                                element, access, ..
                            } => {
                                let element = layout(module, element.ty(), function)?;
                                ParameterType::Buffer {
                                    stride: element.size,
                                    element,
                                    access: match access {
                                        Access::Read => artifact::Access::Read,
                                        Access::Write => artifact::Access::Write,
                                        Access::Atomic => artifact::Access::Atomic,
                                        Access::ReadWrite => artifact::Access::ReadWrite,
                                    },
                                }
                            }
                            _ => {
                                return Err(Error::new(
                                    &function.source,
                                    "unsupported kernel parameter layout",
                                ));
                            }
                        };
                        let namespace =
                            usize::from(matches!(ty, ParameterType::Threadgroup { .. }));
                        let binding = slots[namespace];
                        slots[namespace] += 1;
                        if slots[namespace] > 31 {
                            return Err(Error::new(
                                &function.source,
                                "Metal supports at most 31 bindings per argument table",
                            ));
                        }
                        Ok(artifact::Parameter {
                            name: format!("arg_{index}"),
                            binding,
                            ty,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                // A fixed limit, independent of which checks the proofs remove.
                if slots[0] > 28 {
                    return Err(Error::new(
                        &function.source,
                        "kernels take at most 28 buffer and value parameters; the next slots carry the gate, failure and length words",
                    ));
                }
                Ok(Some(artifact::Kernel {
                    name: function.name.clone(),
                    parameters,
                    required_block: function.required_block,
                    gate_binding: slots[0],
                    fallible: analysis.fallible,
                    lengths: reads_lengths(module, function, analysis),
                    one_dimensional: analysis.one_dimensional,
                    safe: false,
                    requires: function.requires.clone(),
                }))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            bfloat: module
                .functions
                .iter()
                .flat_map(|f| &f.locals)
                .any(|ty| contains_scalar(module, *ty, Scalar::BF16)),
            dynamic_shared: kernels.iter().flatten().any(|kernel| {
                kernel
                    .parameters
                    .iter()
                    .any(|p| matches!(p.ty, ParameterType::Threadgroup { .. }))
            }),
            kernels,
            simd: uses(module, |value| {
                matches!(
                    value,
                    metal_oxide_ir::Expression::SimdCoordinate(_)
                        | metal_oxide_ir::Expression::Simd { .. }
                )
            }),
            logging: uses(module, |value| {
                matches!(value, metal_oxide_ir::Expression::Log { .. })
            }),
            atomics: module
                .functions
                .iter()
                .flat_map(|function| &function.locals)
                .any(|ty| {
                    matches!(
                        ty,
                        Type::Buffer {
                            access: Access::Atomic,
                            ..
                        }
                    )
                }),
        })
    }

    pub(crate) fn abi(
        &self,
        math_mode: artifact::MathMode,
    ) -> Result<artifact::Abi, artifact::Error> {
        let abi = artifact::Abi {
            version: artifact::ABI_VERSION,
            math_mode,
            required_features: [
                (self.atomics, "int32_atomics"),
                (self.simd, "simd_groups"),
                (self.dynamic_shared, "dynamic_threadgroup_memory"),
                (self.logging, "logging"),
                (self.bfloat, "bfloat16"),
            ]
            .into_iter()
            .filter(|(required, _)| *required)
            .map(|(_, name)| name.to_owned())
            .collect(),
            kernels: self.kernels.iter().flatten().cloned().collect(),
        };
        abi.validate()?;
        Ok(abi)
    }
}

fn contains_scalar(module: &Module, ty: Type, scalar: Scalar) -> bool {
    match ty {
        Type::Scalar(value) => value == scalar,
        Type::Aggregate(id) => module
            .types
            .get(id)
            .unwrap()
            .component_types()
            .any(|ty| contains_scalar(module, ty, scalar)),
        Type::Buffer { element, .. } => contains_scalar(module, element.ty(), scalar),
        Type::Pointer(_) => {
            contains_scalar(module, module.types.pointer(ty).unwrap().target, scalar)
        }
        _ => false,
    }
}

fn uses(module: &Module, operation: impl Fn(&metal_oxide_ir::Expression) -> bool) -> bool {
    module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.statements)
        .any(|s| operation(&s.value))
}

fn scalar(value: Scalar, function: &Function) -> Result<artifact::Scalar, Error> {
    match value {
        Scalar::F32 => Ok(artifact::Scalar::F32),
        Scalar::F16 => Ok(artifact::Scalar::F16),
        Scalar::BF16 => Ok(artifact::Scalar::BF16),
        Scalar::U32 => Ok(artifact::Scalar::U32),
        Scalar::U64 => Ok(artifact::Scalar::U64),
        Scalar::I32 => Ok(artifact::Scalar::I32),
        Scalar::U8 => Ok(artifact::Scalar::U8),
        Scalar::U16 => Ok(artifact::Scalar::U16),
        Scalar::I8 => Ok(artifact::Scalar::I8),
        Scalar::I16 => Ok(artifact::Scalar::I16),
        Scalar::Bool | Scalar::Usize | Scalar::Isize => Err(Error::new(
            &function.source,
            format!("{} cannot cross the kernel ABI", value.name()),
        )),
    }
}

fn layout(module: &Module, ty: Type, function: &Function) -> Result<artifact::Layout, Error> {
    use artifact::Layout;
    use metal_oxide_ir::Aggregate;
    let result = match ty {
        Type::Scalar(value) => return Ok(Layout::scalar(scalar(value, function)?)),
        Type::Aggregate(id) => match module.types.get(id).unwrap() {
            Aggregate::Array { element, length } => {
                Layout::array(layout(module, *element, function)?, *length)
            }
            Aggregate::Tuple(fields) => Layout::tuple(
                fields
                    .iter()
                    .map(|ty| layout(module, *ty, function))
                    .collect::<Result<_, _>>()?,
            ),
            Aggregate::Record { name, fields } => Layout::record(
                name,
                fields
                    .iter()
                    .map(|field| Ok((field.name.clone(), layout(module, field.ty, function)?)))
                    .collect::<Result<_, Error>>()?,
            ),
            Aggregate::Enum { name, .. } => {
                return Err(Error::new(
                    &function.source,
                    format!("enum {name} cannot cross the kernel ABI"),
                ));
            }
        },
        _ => {
            return Err(Error::new(
                &function.source,
                "unsupported value in kernel ABI layout",
            ));
        }
    };
    result.map_err(|error| Error::new(&function.source, error.to_string()))
}

/// Whether a kernel reads the length of one of its buffer parameters.
fn reads_lengths(module: &Module, function: &Function, analysis: &FunctionAnalysis) -> bool {
    analysis
        .cfg
        .reachable_blocks()
        .flat_map(|block| &function.blocks[block].statements)
        .any(|statement| {
            matches!(
                &statement.value,
                metal_oxide_ir::Expression::Slice(data) | metal_oxide_ir::Expression::Len(data)
                    if matches!(
                        metal_oxide_ir::operand_type(module, function, data, &statement.source),
                        Ok(Type::Buffer { .. })
                    )
            )
        })
}

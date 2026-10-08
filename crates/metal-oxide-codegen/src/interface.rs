use metal_oxide_artifact::{self as artifact, ParameterType};
use metal_oxide_ir::{Access, Error, Function, Module, Scalar, Type};

#[derive(Debug)]
pub(crate) struct KernelInterfaces {
    pub(crate) kernels: Vec<Option<artifact::Kernel>>,
    pub(crate) simd: bool,
    atomics: bool,
}

impl KernelInterfaces {
    pub(crate) fn new(module: &Module) -> Result<Self, Error> {
        let kernels = module
            .functions
            .iter()
            .map(|function| {
                if !function.kernel {
                    return Ok(None);
                }
                crate::names::validate_name(function)?;
                if function.parameters > 31 {
                    return Err(Error::new(
                        &function.source,
                        "Metal supports at most 31 kernel parameters",
                    ));
                }
                let parameters = function.locals[1..1 + function.parameters]
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| {
                        let ty = match *ty {
                            Type::Scalar(_) | Type::Aggregate(_) => ParameterType::Value {
                                layout: layout(module, *ty, function)?,
                            },
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
                                        Access::ReadWrite => {
                                            return Err(Error::new(
                                                &function.source,
                                                "threadgroup buffers cannot be kernel arguments",
                                            ));
                                        }
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
                        Ok(artifact::Parameter {
                            name: format!("arg_{index}"),
                            binding: index as u32,
                            ty,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Some(artifact::Kernel {
                    name: function.name.clone(),
                    parameters,
                    required_block: function.required_block,
                }))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            kernels,
            simd: uses_simd(module),
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

    pub(crate) fn abi(&self) -> Result<artifact::Abi, artifact::Error> {
        let abi = artifact::Abi {
            version: artifact::ABI_VERSION,
            required_features: [(self.atomics, "int32_atomics"), (self.simd, "simd_groups")]
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

fn uses_simd(module: &Module) -> bool {
    module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.statements)
        .any(|s| {
            matches!(
                s.value,
                metal_oxide_ir::Expression::SimdCoordinate(_)
                    | metal_oxide_ir::Expression::Simd { .. }
            )
        })
}

fn scalar(value: Scalar, function: &Function) -> Result<artifact::Scalar, Error> {
    match value {
        Scalar::F32 => Ok(artifact::Scalar::F32),
        Scalar::F16 => Ok(artifact::Scalar::F16),
        Scalar::U32 => Ok(artifact::Scalar::U32),
        Scalar::I32 => Ok(artifact::Scalar::I32),
        Scalar::U8 => Ok(artifact::Scalar::U8),
        Scalar::U16 => Ok(artifact::Scalar::U16),
        Scalar::I8 => Ok(artifact::Scalar::I8),
        Scalar::I16 => Ok(artifact::Scalar::I16),
        Scalar::Bool | Scalar::Usize => Err(Error::new(
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

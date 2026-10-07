use metal_oxide_artifact::{self as artifact, ParameterType};
use metal_oxide_ir::{Access, Error, Function, Module, Scalar, Type};

#[derive(Debug)]
pub(crate) struct KernelInterfaces {
    pub(crate) kernels: Vec<Option<artifact::Kernel>>,
    pub(crate) simd: bool,
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
                            Type::Scalar(value) => ParameterType::Scalar {
                                scalar: scalar(value, function)?,
                            },
                            Type::Buffer {
                                element, access, ..
                            } => ParameterType::Buffer {
                                element: scalar(element, function)?,
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
                            },
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
        })
    }

    pub(crate) fn abi(&self) -> Result<artifact::Abi, artifact::Error> {
        let abi = artifact::Abi {
            version: artifact::ABI_VERSION,
            required_features: if self.simd {
                vec!["simd_groups".into()]
            } else {
                vec![]
            },
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
                    | metal_oxide_ir::Expression::SimdSum(_)
                    | metal_oxide_ir::Expression::SimdShuffle { .. }
            )
        })
}

fn scalar(value: Scalar, function: &Function) -> Result<artifact::Scalar, Error> {
    match value {
        Scalar::F32 => Ok(artifact::Scalar::F32),
        Scalar::U32 => Ok(artifact::Scalar::U32),
        Scalar::I32 => Ok(artifact::Scalar::I32),
        Scalar::U8 => Ok(artifact::Scalar::U8),
        Scalar::U16 => Ok(artifact::Scalar::U16),
        Scalar::Bool => Err(Error::new(
            &function.source,
            "bool cannot cross the kernel ABI",
        )),
    }
}

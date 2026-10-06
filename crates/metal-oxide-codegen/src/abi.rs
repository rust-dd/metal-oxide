use metal_oxide_artifact::{self as artifact, ParameterType};
use metal_oxide_ir::{Access, Module, Scalar, Type};

pub fn abi(module: &Module) -> Result<artifact::Abi, artifact::Error> {
    metal_oxide_ir::validate(module).map_err(|e| artifact::Error(e.to_string()))?;
    let kernels = module
        .functions
        .iter()
        .filter(|f| f.kernel)
        .map(|f| {
            let parameters = f.locals[1..1 + f.parameters]
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    let ty = match *ty {
                        Type::Scalar(value) => ParameterType::Scalar {
                            scalar: scalar(value)?,
                        },
                        Type::Buffer {
                            element, access, ..
                        } => ParameterType::Buffer {
                            element: scalar(element)?,
                            access: match access {
                                Access::Read => artifact::Access::Read,
                                Access::Write => artifact::Access::Write,
                                Access::Atomic => artifact::Access::Atomic,
                                Access::ReadWrite => {
                                    return Err(artifact::Error(
                                        "threadgroup buffers cannot be kernel arguments".into(),
                                    ));
                                }
                            },
                        },
                        _ => {
                            return Err(artifact::Error(
                                "unsupported kernel parameter layout".into(),
                            ));
                        }
                    };
                    Ok(artifact::Parameter {
                        name: format!("arg_{index}"),
                        binding: index as u32,
                        ty,
                    })
                })
                .collect::<Result<Vec<_>, artifact::Error>>()?;
            Ok(artifact::Kernel {
                name: f.name.clone(),
                parameters,
                required_block: f.required_block,
            })
        })
        .collect::<Result<Vec<_>, artifact::Error>>()?;
    let abi = artifact::Abi {
        version: artifact::ABI_VERSION,
        required_features: if uses_simd(module) {
            vec!["simd_groups".into()]
        } else {
            vec![]
        },
        kernels,
    };
    abi.validate()?;
    Ok(abi)
}

pub(crate) fn uses_simd(module: &Module) -> bool {
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

fn scalar(value: Scalar) -> Result<artifact::Scalar, artifact::Error> {
    match value {
        Scalar::F32 => Ok(artifact::Scalar::F32),
        Scalar::U32 => Ok(artifact::Scalar::U32),
        Scalar::I32 => Ok(artifact::Scalar::I32),
        Scalar::Bool => Err(artifact::Error("bool cannot cross the kernel ABI".into())),
    }
}

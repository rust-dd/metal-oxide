//! Serialized ABI form: kernels share interned layouts and parameter lists.

use serde::{Deserialize, Serialize};

use crate::{
    Abi, Access, Error, FieldLayout, Kernel, Layout, LayoutKind, MathMode, Parameter,
    ParameterType, Scalar,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AbiTables {
    version: u32,
    math_mode: MathMode,
    required_features: Vec<String>,
    /// Layouts; nested layouts refer to earlier entries.
    types: Vec<TypeEntry>,
    signatures: Vec<Vec<ParameterEntry>>,
    kernels: Vec<KernelEntry>,
}

#[derive(PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypeEntry {
    size: u64,
    alignment: u64,
    kind: KindEntry,
}

#[derive(PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum KindEntry {
    Scalar {
        scalar: Scalar,
    },
    Array {
        element: usize,
        length: u32,
        stride: u64,
    },
    Tuple {
        fields: Vec<FieldEntry>,
    },
    Record {
        name: String,
        fields: Vec<FieldEntry>,
    },
}

#[derive(PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldEntry {
    name: String,
    offset: u64,
    layout: usize,
}

#[derive(PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterEntry {
    name: String,
    binding: u32,
    ty: ParameterTypeEntry,
}

#[derive(PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ParameterTypeEntry {
    Value {
        layout: usize,
    },
    Buffer {
        element: usize,
        stride: u64,
        access: Access,
    },
    Threadgroup {
        element: usize,
        stride: u64,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelEntry {
    name: String,
    signature: usize,
    required_block: Option<[u32; 3]>,
    gate_binding: u32,
    fallible: bool,
    lengths: bool,
    one_dimensional: bool,
    safe: bool,
    requires: Vec<crate::Contract>,
}

impl From<Abi> for AbiTables {
    fn from(abi: Abi) -> Self {
        let mut types = Vec::new();
        let mut signatures = Vec::new();
        let kernels = abi
            .kernels
            .into_iter()
            .map(|kernel| {
                let signature = kernel
                    .parameters
                    .iter()
                    .map(|parameter| ParameterEntry {
                        name: parameter.name.clone(),
                        binding: parameter.binding,
                        ty: match &parameter.ty {
                            ParameterType::Value { layout } => ParameterTypeEntry::Value {
                                layout: intern_layout(&mut types, layout),
                            },
                            ParameterType::Buffer {
                                element,
                                stride,
                                access,
                            } => ParameterTypeEntry::Buffer {
                                element: intern_layout(&mut types, element),
                                stride: *stride,
                                access: *access,
                            },
                            ParameterType::Threadgroup { element, stride } => {
                                ParameterTypeEntry::Threadgroup {
                                    element: intern_layout(&mut types, element),
                                    stride: *stride,
                                }
                            }
                        },
                    })
                    .collect();
                KernelEntry {
                    name: kernel.name,
                    signature: intern(&mut signatures, signature),
                    required_block: kernel.required_block,
                    gate_binding: kernel.gate_binding,
                    fallible: kernel.fallible,
                    lengths: kernel.lengths,
                    one_dimensional: kernel.one_dimensional,
                    safe: kernel.safe,
                    requires: kernel.requires,
                }
            })
            .collect();
        Self {
            version: abi.version,
            math_mode: abi.math_mode,
            required_features: abi.required_features,
            types,
            signatures,
            kernels,
        }
    }
}

impl TryFrom<AbiTables> for Abi {
    type Error = Error;

    fn try_from(tables: AbiTables) -> Result<Self, Error> {
        let mut layouts = Vec::<Layout>::with_capacity(tables.types.len());
        for entry in tables.types {
            let kind = match entry.kind {
                KindEntry::Scalar { scalar } => LayoutKind::Scalar { scalar },
                KindEntry::Array {
                    element,
                    length,
                    stride,
                } => LayoutKind::Array {
                    element: Box::new(resolve(&layouts, element)?),
                    length,
                    stride,
                },
                KindEntry::Tuple { fields } => LayoutKind::Tuple {
                    fields: resolve_fields(&layouts, fields)?,
                },
                KindEntry::Record { name, fields } => LayoutKind::Record {
                    name,
                    fields: resolve_fields(&layouts, fields)?,
                },
            };
            layouts.push(Layout {
                size: entry.size,
                alignment: entry.alignment,
                kind,
            });
        }
        let signatures = tables
            .signatures
            .into_iter()
            .map(|parameters| {
                parameters
                    .into_iter()
                    .map(|parameter| {
                        Ok(Parameter {
                            name: parameter.name,
                            binding: parameter.binding,
                            ty: match parameter.ty {
                                ParameterTypeEntry::Value { layout } => ParameterType::Value {
                                    layout: resolve(&layouts, layout)?,
                                },
                                ParameterTypeEntry::Buffer {
                                    element,
                                    stride,
                                    access,
                                } => ParameterType::Buffer {
                                    element: resolve(&layouts, element)?,
                                    stride,
                                    access,
                                },
                                ParameterTypeEntry::Threadgroup { element, stride } => {
                                    ParameterType::Threadgroup {
                                        element: resolve(&layouts, element)?,
                                        stride,
                                    }
                                }
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let kernels = tables
            .kernels
            .into_iter()
            .map(|kernel| {
                Ok(Kernel {
                    parameters: signatures
                        .get(kernel.signature)
                        .cloned()
                        .ok_or_else(|| Error("ABI kernel refers to a missing signature".into()))?,
                    name: kernel.name,
                    required_block: kernel.required_block,
                    gate_binding: kernel.gate_binding,
                    fallible: kernel.fallible,
                    lengths: kernel.lengths,
                    one_dimensional: kernel.one_dimensional,
                    safe: kernel.safe,
                    requires: kernel.requires,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            version: tables.version,
            math_mode: tables.math_mode,
            required_features: tables.required_features,
            kernels,
        })
    }
}

fn intern_layout(types: &mut Vec<TypeEntry>, layout: &Layout) -> usize {
    let mut fields = |fields: &[FieldLayout]| {
        fields
            .iter()
            .map(|field| FieldEntry {
                name: field.name.clone(),
                offset: field.offset,
                layout: intern_layout(types, &field.layout),
            })
            .collect()
    };
    let kind = match &layout.kind {
        LayoutKind::Scalar { scalar } => KindEntry::Scalar { scalar: *scalar },
        LayoutKind::Array {
            element,
            length,
            stride,
        } => KindEntry::Array {
            element: intern_layout(types, element),
            length: *length,
            stride: *stride,
        },
        LayoutKind::Tuple { fields: entries } => KindEntry::Tuple {
            fields: fields(entries),
        },
        LayoutKind::Record {
            name,
            fields: entries,
        } => KindEntry::Record {
            name: name.clone(),
            fields: fields(entries),
        },
    };
    intern(
        types,
        TypeEntry {
            size: layout.size,
            alignment: layout.alignment,
            kind,
        },
    )
}

fn intern<T: PartialEq>(table: &mut Vec<T>, entry: T) -> usize {
    table
        .iter()
        .position(|existing| *existing == entry)
        .unwrap_or_else(|| {
            table.push(entry);
            table.len() - 1
        })
}

/// Only earlier entries are resolved, so references cannot form cycles.
fn resolve(layouts: &[Layout], index: usize) -> Result<Layout, Error> {
    layouts
        .get(index)
        .cloned()
        .ok_or_else(|| Error("ABI layout refers to a later or missing type".into()))
}

fn resolve_fields(layouts: &[Layout], fields: Vec<FieldEntry>) -> Result<Vec<FieldLayout>, Error> {
    fields
        .into_iter()
        .map(|field| {
            Ok(FieldLayout {
                name: field.name,
                offset: field.offset,
                layout: resolve(layouts, field.layout)?,
            })
        })
        .collect()
}

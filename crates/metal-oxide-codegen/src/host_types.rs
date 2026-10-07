use metal_oxide_artifact::{Abi, Error, Layout, LayoutKind, ParameterType};
use std::{collections::HashSet, fmt::Write};

pub(crate) struct HostTypes {
    records: Vec<(Layout, String)>,
    names: HashSet<String>,
}

impl HostTypes {
    pub(crate) fn new(abi: &Abi) -> Result<Self, Error> {
        let mut types = Self {
            records: Vec::new(),
            names: [
                "Kernels",
                "std",
                "metal_oxide",
                "f32",
                "u32",
                "i32",
                "u8",
                "u16",
                "i8",
                "i16",
                "usize",
                "load",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        };
        for parameter in abi.kernels.iter().flat_map(|kernel| &kernel.parameters) {
            types.collect(match &parameter.ty {
                ParameterType::Value { layout } => layout,
                ParameterType::Buffer { element, .. } => element,
            })?;
        }
        Ok(types)
    }

    fn collect(&mut self, layout: &Layout) -> Result<(), Error> {
        match &layout.kind {
            LayoutKind::Scalar { .. } => {}
            LayoutKind::Array { element, .. } => self.collect(element)?,
            LayoutKind::Tuple { fields } | LayoutKind::Record { fields, .. } => {
                if matches!(layout.kind, LayoutKind::Tuple { .. }) && fields.len() > 12 {
                    return Err(Error("host ABI tuples support at most 12 fields".into()));
                }
                for field in fields {
                    self.collect(&field.layout)?;
                }
            }
        }
        if let LayoutKind::Record { name, .. } = &layout.kind
            && !self.records.iter().any(|(other, _)| other == layout)
        {
            let mut candidate = name.clone();
            let mut suffix = 2;
            while !self.names.insert(candidate.clone()) {
                candidate = format!("{name}{suffix}");
                suffix += 1;
            }
            self.records.push((layout.clone(), candidate));
        }
        Ok(())
    }

    pub(crate) fn name(&self, layout: &Layout) -> String {
        match &layout.kind {
            LayoutKind::Scalar { scalar } => scalar.rust_name().into(),
            LayoutKind::Array {
                element, length, ..
            } => format!("[{}; {length}]", self.name(element)),
            LayoutKind::Tuple { fields } => format!(
                "({},)",
                fields
                    .iter()
                    .map(|field| self.name(&field.layout))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            LayoutKind::Record { .. } => format!(
                "r#{}",
                self.records
                    .iter()
                    .find(|(other, _)| other == layout)
                    .unwrap()
                    .1
            ),
        }
    }

    pub(crate) fn emit(&self, output: &mut String) {
        for (layout, name) in &self.records {
            let LayoutKind::Record {
                name: source_name,
                fields,
            } = &layout.kind
            else {
                unreachable!()
            };
            writeln!(
                output,
                "#[derive(Clone, Copy, Debug, PartialEq)]\npub struct r#{name} {{"
            )
            .unwrap();
            for field in fields {
                writeln!(
                    output,
                    "    pub r#{}: {},",
                    field.name,
                    self.name(&field.layout)
                )
                .unwrap();
            }
            writeln!(output, "}}\n\nimpl metal_oxide::GpuValue for r#{name} {{\n    const SIZE: std::primitive::usize = {};\n    const ALIGNMENT: std::primitive::usize = {};", layout.size, layout.alignment).unwrap();
            writeln!(output, "    fn layout() -> metal_oxide::Result<metal_oxide::Layout> {{\n        Ok(metal_oxide::Layout::record({source_name:?}, vec![").unwrap();
            for field in fields {
                writeln!(
                    output,
                    "            ({:?}.into(), <{} as metal_oxide::GpuValue>::layout()?),",
                    field.name,
                    self.name(&field.layout)
                )
                .unwrap();
            }
            output.push_str("        ])?)\n    }\n    fn zeroed() -> Self {\n        Self {\n");
            for field in fields {
                writeln!(
                    output,
                    "            r#{}: <{} as metal_oxide::GpuValue>::zeroed(),",
                    field.name,
                    self.name(&field.layout)
                )
                .unwrap();
            }
            output.push_str("        }\n    }\n    fn encode(self, bytes: &mut [u8]) {\n        assert_eq!(bytes.len(), Self::SIZE);\n        bytes.fill(0);\n");
            for field in fields {
                writeln!(
                    output,
                    "        <{} as metal_oxide::GpuValue>::encode(self.r#{}, &mut bytes[{}..{}]);",
                    self.name(&field.layout),
                    field.name,
                    field.offset,
                    field.offset + field.layout.size
                )
                .unwrap();
            }
            output.push_str("    }\n    fn decode(bytes: &[u8]) -> Self {\n        assert_eq!(bytes.len(), Self::SIZE);\n        Self {\n");
            for field in fields {
                writeln!(
                    output,
                    "            r#{}: <{} as metal_oxide::GpuValue>::decode(&bytes[{}..{}]),",
                    field.name,
                    self.name(&field.layout),
                    field.offset,
                    field.offset + field.layout.size
                )
                .unwrap();
            }
            output.push_str("        }\n    }\n}\n\n");
        }
    }
}

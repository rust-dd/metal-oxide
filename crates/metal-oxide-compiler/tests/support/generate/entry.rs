use super::{
    ast::Function,
    types::{Type, render},
};

/// A generated kernel's entry: its input element type, the `elements` values of the output
/// type that each thread stores, and its block, which the kernel requires when set.
pub struct Entry<'a> {
    pub input: &'a Type,
    pub output: &'a Type,
    pub elements: u32,
    pub block: [u32; 3],
    pub required: bool,
}

impl Entry<'_> {
    fn threads(&self) -> u32 {
        self.block.iter().product()
    }

    /// The `block = (x, y, z), ` attribute of a kernel that requires its block.
    fn attribute(&self) -> String {
        let [x, y, z] = self.block;
        if self.required {
            format!("block = ({x}, {y}, {z}), ")
        } else {
            String::new()
        }
    }

    /// The statements that number the thread `i` in the grid, a row of blocks; threads are
    /// numbered within a block x first, as Metal assigns SIMD lanes.
    fn index(&self) -> String {
        let [x, y, z] = self.block;
        let threads = self.threads();
        match (y * z, self.required) {
            (1, true) => format!("let i = block_idx().x * {x} + thread_idx().x;"),
            (1, false) => "let i = block_idx().x * block_dim().x + thread_idx().x;".into(),
            (_, true) => format!(
                "let i = block_idx().x * {threads} + (thread_idx().z * {y} + thread_idx().y) * {x} + thread_idx().x;"
            ),
            (_, false) => "let (block, thread) = (block_dim(), thread_idx());\n    let i = block_idx().x * (block.x * block.y * block.z) + (thread.z * block.y + thread.y) * block.x + thread.x;".into(),
        }
    }

    /// The statements of a cooperative kernel that number its thread `i` and its `lane` in the
    /// block. Before barriers, checks must be proved, and only the lane's are.
    fn lanes(&self) -> (String, &'static str) {
        let [x, y, _] = self.block;
        if self.block[1] * self.block[2] == 1 {
            let index = format!("let i = block_idx().x * {x} + thread_idx().x;");
            return (index, "thread_idx().x");
        }
        let threads = self.threads();
        let index = format!(
            "let lane = (thread_idx().z * {y} + thread_idx().y) * {x} + thread_idx().x;\n    let i = block_idx().x.wrapping_mul({threads}).wrapping_add(lane);"
        );
        (index, "lane")
    }

    /// The CPU's text for the words of the output element `word`: its bits, low word first.
    fn printed(&self) -> &'static str {
        match self.output {
            Type::Float => "word.to_bits().to_string()",
            Type::Int(int) => match (int.bits, int.signed) {
                (64, false) => r#"format!("{} {}", word as u32, word >> 32)"#,
                (64, true) => r#"format!("{} {}", word as u32, word as u64 >> 32)"#,
                (8, true) => "(word as u8).to_string()",
                (16, true) => "(word as u16).to_string()",
                (_, true) => "(word as u32).to_string()",
                (_, false) => "word.to_string()",
            },
            _ => unreachable!("buffers hold scalars"),
        }
    }

    /// The input of threads past the end, which run `cooperate` without one.
    fn zero(&self) -> String {
        match self.input {
            Type::Float => "0.0_f32".into(),
            ty => format!("0_{}", render(ty, &[])),
        }
    }
}

/// Switches the CPU program's floats to Metal's handling of subnormals, which flushes them to
/// zero, when the oracle asks for it on an AArch64 host.
const FLUSH: &str = r#"
#[cfg(cpu_reference)]
fn flush_subnormals() {
    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("METAL_OXIDE_FLUSH_SUBNORMALS").is_some() {
        let control: u64;
        // SAFETY: FPCR.FZ only makes this thread's float operations flush subnormals to zero.
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) control);
            core::arch::asm!("msr fpcr, {}", in(reg) control | 1 << 24);
        }
    }
}
"#;

/// The kernel that runs `run` for each thread with an input and, with `--cfg cpu_reference`, a
/// program that prints what each thread of `inputs` computes, one thread at a time.
pub fn independent(entry: &Entry, inputs: &str) -> String {
    let elements = entry.elements;
    let stores = stores(elements, "ELEMENTS", 2);
    let (input, output) = (render(entry.input, &[]), render(entry.output, &[]));
    let (printed, attribute, index) = (entry.printed(), entry.attribute(), entry.index());
    format!(
        r#"
#[cfg(not(cpu_reference))]
use metal_oxide_device::{{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx}};

#[cfg(not(cpu_reference))]
#[kernel({attribute}instances(generated({elements})))]
pub unsafe fn generated_body<const ELEMENTS: u32>(input: ReadBuffer<{input}>, output: WriteBuffer<{output}>, n: u32) {{
    {index}
    if i < n {{
        let words = run(unsafe {{ input.load_unchecked(i) }});
        unsafe {{
{stores}        }}
    }}
}}

#[cfg(cpu_reference)]
fn main() {{
    flush_subnormals();
    let location = std::sync::Arc::new(std::sync::Mutex::new((0, 0)));
    let recorded = std::sync::Arc::clone(&location);
    std::panic::set_hook(Box::new(move |info| {{
        let panicked = info.location().unwrap();
        *recorded.lock().unwrap() = (panicked.line(), panicked.column());
    }}));
    let inputs = [{inputs}];
    for input in inputs {{
        match std::panic::catch_unwind(|| run(input as {input})) {{
            Ok(words) => println!("{{}}", words.map(|word| {printed}).join(" ")),
            Err(_) => {{
                let (line, column) = *location.lock().unwrap();
                println!("panic {{line}}:{{column}}");
            }}
        }}
    }}
}}
{FLUSH}"#
    )
}

/// The kernel of a cooperative program, where every thread runs `cooperate` and those with an
/// input run `run`; with `--cfg cpu_reference`, the CPU model of blocks runs the same code.
pub fn cooperative(
    cooperation: &Function,
    counters: Option<&str>,
    entry: &Entry,
    inputs: &str,
) -> String {
    let elements = entry.elements;
    let stores = stores(elements, &elements.to_string(), 2);
    let (input, output) = (render(entry.input, &[]), render(entry.output, &[]));
    let (printed, zero) = (entry.printed(), entry.zero());
    let ([x, y, z], threads) = (entry.block, entry.threads());
    let (index, lane) = entry.lanes();
    let allocations = |indent: usize| {
        cooperation
            .parameters
            .iter()
            .filter_map(|(name, ty)| {
                let allocation = match ty {
                    Type::Threadgroup(element, length) => {
                        let element = render(element, &[]);
                        format!("threadgroup::shared::<{element}, {length}>()")
                    }
                    Type::AtomicThreadgroup(int, length) => {
                        format!("threadgroup::shared_atomic::<{}, {length}>()", int.name())
                    }
                    _ => return None,
                };
                Some(format!(
                    "{}let {name} = {allocation};\n",
                    "    ".repeat(indent)
                ))
            })
            .collect::<String>()
    };
    let (kernel, cpu) = (allocations(1), allocations(2));
    let arguments = |lane: &str| {
        let arguments = cooperation
            .parameters
            .iter()
            .map(|(name, _)| match name.as_str() {
                "input" => "value".to_owned(),
                "lane" => lane.to_owned(),
                buffer => buffer.to_owned(),
            });
        arguments.collect::<Vec<_>>().join(", ")
    };
    let (kernel_arguments, cpu_arguments) = (arguments(lane), arguments("thread_idx().x"));
    let Type::Array(_, values) = cooperation.result else {
        unreachable!("cooperate returns an array of words")
    };
    let values = (0..values)
        .map(|value| format!("k[{value}]"))
        .collect::<Vec<_>>()
        .join(", ");
    let parameter = counters
        .map(|int| format!(", counters: AtomicBuffer<{int}>"))
        .unwrap_or_default();
    let (create, print) = match counters {
        Some(int) => (
            format!("    let counters = model::AtomicBuffer::<{int}>::new(&model::COUNTERS);\n"),
            "    let words = counters.words().iter().map(u32::to_string).collect::<Vec<_>>();\n    \
             println!(\"counters {}\", words.join(\" \"));\n"
                .to_owned(),
        ),
        None => Default::default(),
    };
    let model = include_str!("model.rs");
    format!(
        r#"
#[cfg(not(cpu_reference))]
use metal_oxide_device::{{AtomicBuffer, AtomicThreadgroupBuffer, Float, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup}};

#[cfg(not(cpu_reference))]
#[kernel(block = ({x}, {y}, {z}))]
pub unsafe fn generated(input: ReadBuffer<{input}>, output: WriteBuffer<{output}>, n: u32{parameter}) {{
    {index}
    let value = if i < n {{ unsafe {{ input.load_unchecked(i) }} }} else {{ {zero} }};
{kernel}    let k = cooperate({kernel_arguments});
    if i < n {{
        let words = run(value, {values});
        unsafe {{
{stores}        }}
    }}
}}

#[cfg(cpu_reference)]
use model::{{AtomicBuffer, AtomicThreadgroupBuffer, ThreadgroupBuffer, simdgroup, thread_idx, threadgroup}};

#[cfg(cpu_reference)]
mod model {{
{model}}}

#[cfg(cpu_reference)]
fn main() {{
    let inputs = [{inputs}];
    let n = inputs.len() as u32;
{create}    let outcomes = model::launch({threads}, n, |i| {{
        flush_subnormals();
        let value = if i < n {{ inputs[i as usize] as {input} }} else {{ {zero} }};
{cpu}        let k = cooperate({cpu_arguments});
        (i < n).then(|| run(value, {values}))
    }});
    for outcome in outcomes {{
        match outcome {{
            Ok(Some(words)) => println!("{{}}", words.map(|word| {printed}).join(" ")),
            Ok(None) => {{}}
            Err((line, column)) => println!("panic {{line}}:{{column}}"),
        }}
    }}
{print}}}
{FLUSH}"#
    )
}

/// The stores of a thread's `elements` values into `output`, at `indent` levels.
fn stores(elements: u32, stride: &str, indent: usize) -> String {
    (0..elements)
        .map(|element| {
            let indent = "    ".repeat(indent);
            format!("{indent}output.store_unchecked(i * {stride} + {element}, words[{element}]);\n")
        })
        .collect()
}

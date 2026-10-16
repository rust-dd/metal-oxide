use super::{
    ast::Function,
    cooperation::BLOCK,
    types::{Type, render},
};

/// The kernel that runs `run` for each thread with an input and, with `--cfg cpu_reference`, a
/// program that prints what each thread of `inputs` computes, one thread at a time.
pub fn independent(words: usize, inputs: &str) -> String {
    let stores = stores(words, "WORDS", 2);
    format!(
        r#"
#[cfg(not(cpu_reference))]
use metal_oxide_device::{{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx}};

#[cfg(not(cpu_reference))]
#[kernel(instances(generated({words})))]
pub unsafe fn generated_body<const WORDS: u32>(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32) {{
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {{
        let words = run(unsafe {{ input.load_unchecked(i) }});
        unsafe {{
{stores}        }}
    }}
}}

#[cfg(cpu_reference)]
fn main() {{
    let location = std::sync::Arc::new(std::sync::Mutex::new((0, 0)));
    let recorded = std::sync::Arc::clone(&location);
    std::panic::set_hook(Box::new(move |info| {{
        let panicked = info.location().unwrap();
        *recorded.lock().unwrap() = (panicked.line(), panicked.column());
    }}));
    let inputs = [{inputs}];
    for input in inputs {{
        match std::panic::catch_unwind(|| run(input)) {{
            Ok(words) => println!("{{}}", words.map(|word| word.to_string()).join(" ")),
            Err(_) => {{
                let (line, column) = *location.lock().unwrap();
                println!("panic {{line}}:{{column}}");
            }}
        }}
    }}
}}
"#
    )
}

/// The kernel of a cooperative program, where every thread runs `cooperate` and those with an
/// input run `run`; with `--cfg cpu_reference`, the CPU model of blocks runs the same code.
pub fn cooperative(
    cooperation: &Function,
    counters: Option<&str>,
    words: usize,
    inputs: &str,
) -> String {
    let stores = stores(words, &words.to_string(), 2);
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
    let arguments = cooperation
        .parameters
        .iter()
        .map(|(name, _)| match name.as_str() {
            "input" => "value",
            "lane" => "thread_idx().x",
            buffer => buffer,
        })
        .collect::<Vec<_>>()
        .join(", ");
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
use metal_oxide_device::{{AtomicBuffer, AtomicThreadgroupBuffer, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup}};

#[cfg(not(cpu_reference))]
#[kernel(block = ({BLOCK}, 1, 1))]
pub unsafe fn generated(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32{parameter}) {{
    let i = block_idx().x * {BLOCK} + thread_idx().x;
    let value = if i < n {{ unsafe {{ input.load_unchecked(i) }} }} else {{ 0 }};
{kernel}    let k = cooperate({arguments});
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
{create}    let outcomes = model::launch({BLOCK}, n, |i| {{
        let value = if i < n {{ inputs[i as usize] }} else {{ 0 }};
{cpu}        let k = cooperate({arguments});
        (i < n).then(|| run(value, {values}))
    }});
    for outcome in outcomes {{
        match outcome {{
            Ok(Some(words)) => println!("{{}}", words.map(|word| word.to_string()).join(" ")),
            Ok(None) => {{}}
            Err((line, column)) => println!("panic {{line}}:{{column}}"),
        }}
    }}
{print}}}
"#
    )
}

/// The stores of a thread's `words` into `output`, at `indent` levels.
fn stores(words: usize, stride: &str, indent: usize) -> String {
    (0..words)
        .map(|word| {
            let indent = "    ".repeat(indent);
            format!("{indent}output.store_unchecked(i * {stride} + {word}, words[{word}]);\n")
        })
        .collect()
}

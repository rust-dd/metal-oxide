#![cfg(feature = "rustc-private")]

mod support;

use std::{process::Command, time::Duration};
use support::{
    differential::{Outcome, Run},
    generate::Program,
    oracle::Failure,
};

#[test]
fn differential_process_deadline_stops_nonterminating_programs() {
    let error = support::process::output(
        Command::new("sh").args(["-c", "while :; do :; done"]),
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    let output = support::process::output(
        Command::new("sh").args(["-c", "printf complete"]),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(support::checked(output), "complete");
}

#[test]
fn generated_programs_match_cpu_and_emitted_msl() {
    for seed in support::differential::SEEDS {
        let case = support::differential::Case::generated(seed);
        let (_, generated) = case
            .check()
            .unwrap_or_else(|failure| panic!("{}: {failure}", case.source.display()));
        let original = metal_oxide_artifact::COMPILER_OUTPUTS
            .map(|file| std::fs::read(generated.join(file.name())).unwrap());
        let rebuilt = case.compile().unwrap();
        for (file, expected) in metal_oxide_artifact::COMPILER_OUTPUTS
            .into_iter()
            .zip(original)
        {
            assert_eq!(
                std::fs::read(rebuilt.join(file.name())).unwrap(),
                expected,
                "nondeterministic {} for seed {seed:#x}",
                file.name()
            );
        }
    }
}

#[test]
fn a_kernel_that_writes_other_words_is_reported() {
    let (case, expected) = (5..)
        .map(|seed| {
            let case = support::differential::Case::generated(seed);
            let expected = case.cpu().unwrap();
            (case, expected)
        })
        .find(|(case, expected)| {
            !case.cooperative
                && case.output != "f32"
                && expected
                    .threads
                    .iter()
                    .all(|outcome| matches!(outcome, Outcome::Words(_)))
        })
        .unwrap();
    let source = std::fs::read_to_string(&case.source).unwrap();
    let changed = source.replacen("words[0]);", "words[0] ^ 1);", 1);
    assert_ne!(changed, source);
    std::fs::write(&case.source, changed).unwrap();
    assert!(matches!(case.check(), Err(Failure::Msl(_))));
    let generated = case.compile().unwrap();
    assert!(
        case.mismatch(&expected.threads, &case.msl(&generated).unwrap())
            .is_some()
    );
}

#[test]
fn compiler_failures_match_by_their_first_diagnostic_wherever_it_points() {
    let rejected = |location: &str| {
        Failure::Compiler(format!(
            "stdout:\nstderr:\nerror: {location}: integer division/remainder requires a proved nonzero divisor\n"
        ))
    };
    assert!(rejected("/a/case.rs:20:138").same_as(&rejected("/b/case.rs:3:9")));
    let other = Failure::Compiler("stderr:\nerror: unsupported device type: &&()\n".into());
    assert!(!rejected("/a/case.rs:1:1").same_as(&other));
    assert!(Failure::Msl("thread 3".into()).same_as(&Failure::Msl("thread 9".into())));
    assert!(!Failure::Msl("thread 3".into()).same_as(&Failure::Gpu("thread 3".into())));
    let gpu = |detail: &str| Failure::Gpu(detail.into());
    assert!(gpu("thread 3 (input 0x1)").same_as(&gpu("thread 9 (input 0x2)")));
    assert!(gpu("the GPU completed, the CPU panicked at").same_as(&gpu("the GPU completed")));
    assert!(!gpu("thread 3 (input 0x1)").same_as(&gpu("counters [1], expected [2]")));
    assert!(!gpu("the GPU completed").same_as(&gpu("the GPU failed Panic")));
    assert!(!gpu("thread 3").same_as(&Failure::Device("Compilation failed".into())));
    let device = |detail: &str| Failure::Device(detail.into());
    assert!(device("Compilation failed").same_as(&device("Compilation failed")));
    assert!(!device("Compilation failed").same_as(&device("GPU process timed out")));
}

#[test]
fn shrinking_keeps_what_fails_and_drops_the_rest() {
    let fails = |program: &Program| program.source(&[]).contains("\"generated assertion\"");
    // No simplification removes the entry, about 1.5 kB, or the CPU model of cooperative sources.
    let program = (0..)
        .map(Program::generate)
        .find(|program| {
            !program.cooperative() && fails(program) && program.source(&[]).len() > 4000
        })
        .unwrap();
    let size = program.source(&[]).len();
    let shrunk = program.shrink(fails);
    assert!(fails(&shrunk));
    let source = shrunk.source(&[]);
    assert!(source.len() < size / 2, "{source}");
}

/// Minimized programs that once failed, which must keep matching the CPU.
#[test]
fn generated_fixtures_match_cpu_and_emitted_msl() {
    for case in support::differential::fixtures() {
        if let Err(failure) = case.check() {
            panic!("{}: {failure}", case.source.display());
        }
    }
}

#[test]
fn bit_reversals_match_cpu_and_host_msl() {
    let case = support::differential::Case::saved(
        &support::root()
            .join("crates/metal-oxide-compiler/tests/fixtures/generated/narrow_reverse_float.rs"),
    );
    let (expected, _) = case.check().unwrap();
    for outcome in expected.threads {
        assert_eq!(outcome, Outcome::Words(vec![0, 0x41d8_0000, 0, 0]));
    }
    for name in [
        "reverse_constant_floats",
        "narrow_reverse_conversions",
        "narrow_reverse_dynamic",
    ] {
        let path = support::root().join(format!(
            "crates/metal-oxide-compiler/tests/fixtures/generated/{name}.rs"
        ));
        let case = support::differential::Case::saved(&path);
        case.check()
            .unwrap_or_else(|failure| panic!("{}: {failure}", path.display()));
    }
}

/// GPU results with subnormals flushed to zero agree with the CPU reference that flushes them.
#[test]
#[cfg(target_arch = "aarch64")]
fn the_cpu_reference_flushes_subnormals_when_asked() {
    let case = support::differential::Case::saved(
        &support::root()
            .join("crates/metal-oxide-compiler/tests/fixtures/generated/flushed_subnormals.rs"),
    );
    let (ieee, flushed) = (case.cpu().unwrap(), case.cpu_flushed().unwrap());
    for (ieee, flushed) in ieee.threads.iter().zip(&flushed.threads) {
        let (Outcome::Words(ieee), Outcome::Words(flushed)) = (ieee, flushed) else {
            panic!("the fixture completes");
        };
        assert_eq!(ieee[0], 1);
        assert_eq!(flushed, &[0, 0]);
    }
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[ignore = "requires a Metal GPU"]
fn bit_reversals_match_cpu_on_gpu() {
    let device = metal_oxide::Device::system_default().unwrap();
    let mut cases = [
        "narrow_reverse_float",
        "reverse_constant_floats",
        "narrow_reverse_conversions",
        "narrow_reverse_dynamic",
    ]
    .map(|name| {
        support::differential::Case::saved(&support::root().join(format!(
            "crates/metal-oxide-compiler/tests/fixtures/generated/{name}.rs"
        )))
    })
    .into_iter()
    .collect::<Vec<_>>();
    cases.insert(
        1,
        support::differential::Case::generated(0x44b6_8b45_d22d_f4c9),
    );
    let mut failures = Vec::new();
    for case in cases {
        eprintln!("checking {}", case.source.display());
        let (expected, generated) = match case.check() {
            Ok(checked) => checked,
            Err(failure) => {
                failures.push(format!("{}: {failure}", case.source.display()));
                continue;
            }
        };
        let actual = match case.gpu(&device, &generated) {
            Ok(actual) => actual,
            Err(failure) => {
                failures.push(format!("{}: {failure}", case.source.display()));
                continue;
            }
        };
        eprintln!(
            "thread 0: expected {:x?}, got {:x?}",
            expected.threads.first(),
            actual.words.get(..case.words.len())
        );
        if let Some(failure) =
            case.gpu_mismatch(&expected, actual.failure, &actual.words, &actual.counters)
        {
            failures.push(format!("{}: {failure}", case.source.display()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn failures_are_compared_by_location_and_completed_threads_by_words() {
    let mut case = support::differential::Case::generated(0);
    case.output = "u32".into();
    let words = case.words.len();
    let panic = Outcome::Panic { line: 9, column: 5 };
    let run = |threads: &[Outcome], counters: &[u32]| Run {
        threads: threads.to_vec(),
        counters: counters.to_vec(),
    };
    let threads = [Outcome::Words(vec![0; words]), panic.clone()];
    let expected = run(&threads, &[]);
    let layout = support::output::Layout {
        threads: 2,
        ..case.layout()
    };
    let mut zeros = layout.initial();
    zeros[..2 * words].fill(0);
    assert_eq!(
        case.gpu_mismatch(&expected, Some(panic.clone()), &zeros, &[]),
        None
    );
    assert!(case.gpu_mismatch(&expected, None, &zeros, &[]).is_some());
    let elsewhere = Outcome::Panic { line: 9, column: 6 };
    assert!(
        case.gpu_mismatch(&expected, Some(elsewhere.clone()), &zeros, &[])
            .is_some()
    );
    let mut changed = zeros.clone();
    changed[0] = 1;
    assert!(
        case.gpu_mismatch(&expected, Some(panic.clone()), &changed, &[])
            .is_some()
    );
    let completed = [
        Outcome::Words(vec![0; words]),
        Outcome::Words(vec![0; words]),
    ];
    assert!(
        case.gpu_mismatch(&run(&completed, &[]), Some(panic.clone()), &zeros, &[])
            .is_some()
    );
    let counted = run(&threads, &[1, 2]);
    assert_eq!(
        case.gpu_mismatch(&counted, Some(panic.clone()), &zeros, &[1, 2]),
        None
    );
    assert!(
        case.gpu_mismatch(&counted, Some(panic.clone()), &zeros, &[1, 3])
            .is_some()
    );
    assert!(
        case.mismatch(&threads, &[threads[0].clone(), elsewhere])
            .is_some()
    );
    assert!(case.mismatch(&threads, &completed).is_some());
}

#[test]
fn generated_programs_cover_the_supported_constructs() {
    let sources = (0..300)
        .map(|seed| support::generate::Program::generate(seed).source(&[]))
        .collect::<String>();
    for construct in [
        ".wrapping_add(",
        ".wrapping_sub(",
        ".wrapping_mul(",
        ".wrapping_shl(",
        ".wrapping_shr(",
        ".rotate_left(",
        ".rotate_right(",
        ".wrapping_neg()",
        ".reverse_bits()",
        ".count_ones()",
        ".leading_zeros()",
        ".trailing_zeros()",
        ".min(",
        ".max(",
        ".abs()",
        " ^ ",
        " & ",
        " | ",
        " / ",
        " % ",
        " + ",
        " - ",
        " * ",
        " == ",
        " != ",
        " < ",
        " <= ",
        " > ",
        " >= ",
        " && ",
        " || ",
        "(!",
        "(-",
        "_u8",
        "_u16",
        "_u32",
        "_u64",
        "_i8",
        "_i16",
        "_i32",
        "_f32",
        "::MIN",
        "::MAX",
        " as f32)",
        " as u8)",
        " as i32)",
        "(if ",
        "(match ",
        ": for c",
        "break 'o",
        "continue 'o",
        "struct R",
        "R0 { ",
        ",)",
        "% 1]",
        "% 4]",
        "fn h",
        "<const K: u32>",
        "::<",
        "let mut ",
        "\"generated assertion\"",
        " as usize]",
        " as usize) % 5]",
        " << ",
        " >> ",
        "(-(",
        "enum E0 { V0",
        "Some(b",
        "None => ",
        "None::<",
        ".then_some(",
        "Ok::<",
        "Err::<",
        ".unwrap_or(",
        ".unwrap()",
        ".expect(\"generated expectation\")",
        ".ok_or(",
        "?)",
        "matches!(",
        ".is_some()",
        ".is_err()",
        ": &R",
        ": &mut ",
        ": &[",
        ": &mut [",
        "(*p",
        ".len() as ",
        "(&l",
        "(&mut l",
        "[0..1]",
        "..((",
        ": impl Fn(",
        " = (move |a",
        "(move || -> ",
        "// cooperative",
        "simdgroup::sum(",
        "simdgroup::inclusive_sum(",
        "simdgroup::exclusive_sum(",
        "simdgroup::ballot(",
        "simdgroup::all(",
        "simdgroup::shuffle(",
        "simdgroup::shuffle_up(",
        "simdgroup::shuffle_xor(",
        "store_unchecked(lane ^ ",
        "threadgroup::barrier()",
        ".fetch_sub(",
        ".fetch_min(",
        "ThreadgroupBuffer<f32, 64>",
        "AtomicThreadgroupBuffer<i32, ",
        "counters: AtomicBuffer<i32>",
        "counters.fetch_max(",
        "// counters: u32",
        "    'o",
        "input: ReadBuffer<i8>",
        "input: ReadBuffer<f32>",
        "output: WriteBuffer<u16>",
        "output: WriteBuffer<i64>",
        "output: WriteBuffer<f32>",
        "run(input as u64)",
        "as f32 } else { 0.0_f32 }",
        ": while w",
        " += 1;",
        "return [",
        "return Some(",
        "(0_u32 .. ((",
        "(0_u32 ..= ((",
        "_u32) .. ((",
        ".checked_add(",
        ".checked_div(",
        ".checked_shl(",
        ".checked_pow(",
        ".checked_neg()",
        ".overflowing_mul(",
        ".overflowing_shr(",
        ").1",
        ".saturating_sub(",
        ".saturating_pow(",
        ".wrapping_rem(",
        ".wrapping_pow(",
        ".pow(",
        "_usize",
        "isize::MIN",
        "_f32 / ",
        "_f32 % ",
        ".sqrt()",
        ".floor()",
        ".signum()",
        ".mul_add(",
        ".copysign(",
        ".clamp(",
        ".powi(",
        ".exp2()",
        ".ln()",
        ".atan2(",
        ".powf(",
        ".hypot(",
        ".sin_cos().",
        " approx",
    ] {
        assert!(
            sources.contains(construct),
            "no generated program uses {construct:?}"
        );
    }
}

#[test]
fn the_cpu_model_runs_the_threads_of_a_block_together() {
    use support::generate::model::{launch, simdgroup, thread_idx, threadgroup};
    let outcomes = launch(64, 96, |i| {
        let shared = threadgroup::shared::<u32, 64>();
        let counts = threadgroup::shared_atomic::<u32, 2>();
        let lane = thread_idx().x;
        // SAFETY: every thread of the block reaches each operation, and each stores its own cell.
        let words = unsafe {
            shared.store_unchecked(lane ^ 1, i);
            counts.store(lane % 2, 0, core::sync::atomic::Ordering::Relaxed);
            threadgroup::barrier();
            counts.fetch_add(lane % 2, lane, core::sync::atomic::Ordering::Relaxed);
            threadgroup::barrier();
            [
                shared.load_unchecked(lane),
                counts.load(1, core::sync::atomic::Ordering::Relaxed),
                simdgroup::sum(i),
                simdgroup::exclusive_sum(1_u32),
                simdgroup::shuffle_up(i, 1),
                simdgroup::ballot(lane.is_multiple_of(3))[0],
            ]
        };
        assert!(i != 70, "thread 70 fails after the cooperative operations");
        words
    });
    assert_eq!(outcomes.len(), 128);
    for (i, outcome) in (0..).zip(&outcomes) {
        let (first, lane) = (i / 64 * 64, i % 64);
        let group = first + lane / 32 * 32;
        let expected = [
            first + (lane ^ 1),
            (1..64).step_by(2).sum(),
            (group..group + 32).sum(),
            lane % 32,
            if lane % 32 == 0 { i } else { i - 1 },
            0x4924_9249 << (lane / 32),
        ];
        match outcome {
            Ok(words) => assert_eq!(*words, expected, "thread {i}"),
            Err((line, _)) => assert!(i == 70 && *line > 0, "thread {i} panicked"),
        }
    }
    assert!(outcomes[70].is_err());
}

#[test]
fn generated_programs_are_deterministic() {
    for seed in [0, 1, u64::MAX] {
        let first = support::generate::Program::generate(seed);
        let second = support::generate::Program::generate(seed);
        assert_eq!(first.source(&[1, 2]), second.source(&[1, 2]));
        assert_eq!(first.words, second.words);
    }
    let sources = (0..32)
        .map(|seed| support::generate::Program::generate(seed).source(&[]))
        .collect::<std::collections::BTreeSet<_>>();
    assert!(sources.len() > 28, "seeds should yield distinct programs");
}

#[test]
fn unsupported_generated_sources_keep_locations_and_never_panic() {
    let directory = support::differential::scratch("unsupported");
    for (name, body, diagnostic) in [
        (
            "formatted_panic",
            "if n > 3 { panic!(\"n = {}\", n); } output.store_unchecked(0, n);",
            "literal message",
        ),
        (
            "pointer",
            "let address = n as *const u32; output.store_unchecked(0, *address);",
            "unsupported",
        ),
        (
            "varying_check",
            "let x = n / metal_oxide_device::thread_idx().x; metal_oxide_device::threadgroup::barrier(); output.store_unchecked(0, x);",
            "must be proved or uniform",
        ),
    ] {
        let source = directory.join(format!("{name}.rs"));
        std::fs::write(&source, format!("#![no_std]\nuse metal_oxide_device::{{kernel, WriteBuffer}};\n#[kernel]\npub unsafe fn invalid(output: WriteBuffer<u32>, n: u32) {{\nunsafe {{ {body} }}\n}}\n")).unwrap();
        let (output, _) = support::emit(source.to_str().unwrap(), &["-C", "overflow-checks=off"]);
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.to_lowercase().contains(diagnostic), "{error}");
        assert!(error.contains(&format!("{name}.rs:5:")), "{error}");
        assert!(
            !error.contains("internal compiler error") && !error.contains("panicked"),
            "{error}"
        );
    }
}

#[test]
fn duplicate_entrypoint_names_have_source_diagnostics() {
    let source = support::differential::scratch("duplicate").join("duplicate.rs");
    std::fs::write(&source, "#![no_std]\nmod first { use metal_oxide_device::kernel; #[kernel] pub unsafe fn repeated() {} }\nmod second { use metal_oxide_device::kernel; #[kernel] pub unsafe fn repeated() {} }\n").unwrap();
    let (output, _) = support::emit(source.to_str().unwrap(), &[]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("duplicate") && error.contains("duplicate.rs:"),
        "{error}"
    );
    assert!(
        !error.contains("internal compiler error") && !error.contains("panicked"),
        "{error}"
    );
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[ignore = "requires a Metal GPU"]
fn generated_programs_and_fixtures_match_cpu_on_gpu() {
    let device = metal_oxide::Device::system_default().unwrap();
    let seeds = support::differential::SEEDS.map(support::differential::Case::generated);
    for case in seeds.into_iter().chain(support::differential::fixtures()) {
        let result = case
            .check()
            .and_then(|(expected, generated)| case.check_gpu(&device, &expected, &generated));
        if let Err(failure) = result {
            panic!("{}: {failure}", case.source.display());
        }
    }
}

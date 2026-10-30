use super::*;
use metal_oxide::{Error, KernelFailure};
use objc2_metal::{MTLDevice, MTLGPUFamily};

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/runtime_checks.rs";

fn module(device: &Device) -> metal_oxide::Result<Module> {
    let (output, directory) = support::emit(FIXTURE, &["-C", "overflow-checks=on"]);
    support::checked(output);
    support::artifact::module(device, &directory)
}

/// The fixture line that contains `text`, as a check location reports it.
fn line(text: &str) -> u32 {
    let source = include_str!("../fixtures/runtime_checks.rs");
    source.lines().position(|line| line.contains(text)).unwrap() as u32 + 1
}

fn failure<T>(result: metal_oxide::Result<T>) -> KernelFailure {
    match result {
        Err(Error::KernelFailed(failure)) => *failure,
        Err(other) => panic!("expected a kernel failure, got {other:?}"),
        Ok(_) => panic!("expected a kernel failure"),
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn failed_bounds_check_reports_its_rust_location() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let lookup = Pipeline::new(&device, &module(&device)?, "lookup")?;
    let indices = device.buffer_from_slice(&[0_u32, 1, 2, 3, 2, 1, 0, 3, 9, 1])?;
    let mut out = device.buffer_from_slice(&[0_u32; 10])?;
    // SAFETY: both buffers have ten elements and each thread writes its own element.
    let result = unsafe {
        Launch::new(
            Ok(&lookup),
            LaunchConfig::<4>::for_elements(10)?,
            (&indices, &mut out, 10_u32),
            |(indices, out, n)| {
                Ok([
                    Argument::read(indices),
                    Argument::write(out),
                    Argument::value(*n)?,
                ])
            },
        )
        .wait()
    };
    let failure = failure(result);
    assert_eq!(failure.kernel, "lookup");
    assert_eq!(failure.dispatch, 0);
    assert_eq!(failure.message, "index out of bounds");
    assert!(
        failure.file.ends_with("fixtures/runtime_checks.rs"),
        "{failure}"
    );
    assert_eq!(failure.line, line("values[indices"));
    // Threads that pass their checks finish; the failing thread stops before its store.
    assert_eq!(out.read().wait(), [10, 20, 30, 40, 30, 20, 10, 40, 0, 20]);
    assert!(
        failure
            .to_string()
            .starts_with("kernel `lookup` panicked at ")
    );
    Ok(())
}

/// Launches `lookup` and `fill` in one command buffer; `indices` decide which lookups fail.
fn lookups_and_fills(
    device: &Device,
    module: &Module,
    indices: [u32; 2],
) -> metal_oxide::Result<(metal_oxide::Result<()>, [Vec<u32>; 3])> {
    let lookup = Pipeline::new(device, module, "lookup")?;
    let fill = Pipeline::new(device, module, "fill")?;
    let indices = indices.map(|index| device.buffer_from_slice(&[index; 4]));
    let [first, second] = [&indices[0], &indices[1]].map(|indices| indices.as_ref().unwrap());
    let mut looked_up = [
        device.buffer_zeroed::<u32>(4)?,
        device.buffer_zeroed::<u32>(4)?,
    ];
    let mut filled = device.buffer_zeroed::<u32>(4)?;
    let config = LaunchConfig::<4>::for_elements(4)?;
    let [early, late] = &mut looked_up;
    // SAFETY: every buffer covers four elements with one writer per element.
    let result = sequence(|batch| unsafe {
        for (indices, out, value) in [(first, early, 1_u32), (second, late, 2)] {
            batch.push(Launch::new(
                Ok(&lookup),
                config,
                (indices, out, 4_u32),
                |(indices, out, n)| {
                    Ok([
                        Argument::read(indices),
                        Argument::write(out),
                        Argument::value(*n)?,
                    ])
                },
            ))?;
            batch.push(Launch::new(
                Ok(&fill),
                config,
                (&mut filled, value, 4_u32),
                |(filled, value, n)| {
                    Ok([
                        Argument::write(filled),
                        Argument::value(*value)?,
                        Argument::value(*n)?,
                    ])
                },
            ))?;
        }
        Ok(())
    })
    .wait();
    Ok((
        result,
        [
            looked_up[0].read().wait().to_vec(),
            looked_up[1].read().wait().to_vec(),
            filled.read().wait().to_vec(),
        ],
    ))
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_failed_dispatch_skips_the_dispatches_ordered_after_it() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    // The first lookup, the first fill and the second lookup share no buffer, so they run
    // together; the second fill waits for the first and gates on both lookups.
    let (result, [early, late, filled]) = lookups_and_fills(&device, &module, [5, 1])?;
    let first = failure(result);
    assert_eq!((first.kernel.as_str(), first.dispatch), ("lookup", 0));
    assert_eq!(
        [early, late, filled],
        [[0; 4], [20; 4], [1; 4]].map(Vec::from)
    );
    let (result, [early, late, filled]) = lookups_and_fills(&device, &module, [1, 5])?;
    let second = failure(result);
    assert_eq!((second.kernel.as_str(), second.dispatch), ("lookup", 2));
    assert_eq!(
        [early, late, filled],
        [[20; 4], [0; 4], [1; 4]].map(Vec::from)
    );
    // The next command buffer starts from clear words.
    let (result, [early, late, filled]) = lookups_and_fills(&device, &module, [2, 3])?;
    result?;
    assert_eq!(
        [early, late, filled],
        [[30; 4], [40; 4], [2; 4]].map(Vec::from)
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn empty_dispatches_preserve_the_failure_push_index() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let lookup = Pipeline::new(&device, &module(&device)?, "lookup")?;
    let invalid = device.buffer_from_slice(&[4_u32])?;
    let mut out = device.buffer_from_slice(&[99_u32])?;
    for empty_dispatches in [1, 3] {
        // SAFETY: only the one-thread dispatch runs; both buffers cover its single element.
        let result = sequence(|batch| unsafe {
            batch.push(sequence(|batch| {
                for dispatch in 0..=empty_dispatches + 1 {
                    let grid = u32::from(dispatch == empty_dispatches);
                    batch.push(Launch::new(
                        Ok(&lookup),
                        LaunchConfig::<1>::new(Dim3::x(grid)),
                        (&invalid, &mut out, 1_u32),
                        |(indices, out, n)| {
                            Ok([
                                Argument::read(indices),
                                Argument::write(out),
                                Argument::value(*n)?,
                            ])
                        },
                    ))?;
                }
                Ok(())
            }))
        })
        .wait();
        let failure = failure(result);
        assert_eq!(failure.dispatch, empty_dispatches);
        assert_eq!(failure.kernel, "lookup");
        assert_eq!(failure.message, "index out of bounds");
        assert_eq!(out.read().wait(), [99]);
    }
    Ok(())
}

/// Commits `lookup` of `indices` into `looked_up` beside a long kernel in one command buffer
/// and drops its future, so the command buffer still runs when the caller commits more work.
fn detached_lookup(
    device: &Device,
    module: &Module,
    indices: &metal_oxide::Buffer<u32>,
    looked_up: &mut metal_oxide::Buffer<u32>,
) -> metal_oxide::Result<()> {
    let spin = Pipeline::new(device, module, "spin")?;
    let lookup = Pipeline::new(device, module, "lookup")?;
    let mut slow = device.buffer_zeroed::<u32>(256)?;
    let slow_config = LaunchConfig::<256>::for_elements(256)?;
    let config = LaunchConfig::<4>::for_elements(4)?;
    // SAFETY: each thread writes its own element of buffers that cover the grids.
    drop(started(sequence(|batch| unsafe {
        batch.push(Launch::new(
            Ok(&spin),
            slow_config,
            (&mut slow, 3_000_000_u32),
            |(slow, rounds)| Ok([Argument::write(slow), Argument::value(*rounds)?]),
        ))?;
        batch.push(Launch::new(
            Ok(&lookup),
            config,
            (indices, &mut *looked_up, 4_u32),
            |(indices, out, n)| {
                Ok([
                    Argument::read(indices),
                    Argument::write(out),
                    Argument::value(*n)?,
                ])
            },
        ))?;
        Ok(())
    })));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_failure_skips_the_running_reads_of_its_writes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let arithmetic = Pipeline::new(&device, &module, "arithmetic")?;
    let fill = Pipeline::new(&device, &module, "fill")?;
    let config = LaunchConfig::<4>::for_elements(4)?;
    let invalid = device.buffer_from_slice(&[5_u32; 4])?;
    let valid = device.buffer_from_slice(&[2_u32; 4])?;
    let mut looked_up = device.buffer_zeroed::<u32>(4)?;
    let mut out = device.buffer_zeroed::<u32>(4)?;
    // SAFETY: four threads each read and write their own element of four-element buffers.
    let increment = |looked_up: &metal_oxide::Buffer<u32>, out: &mut metal_oxide::Buffer<u32>| unsafe {
        Launch::new(
            Ok(&arithmetic),
            config,
            (looked_up, out, 1_u32, 4_u32),
            |(values, out, divisor, n)| {
                Ok([
                    Argument::read(values),
                    Argument::write(out),
                    Argument::value(*divisor)?,
                    Argument::value(*n)?,
                ])
            },
        )
        .wait()
        .map(drop)
    };

    detached_lookup(&device, &module, &invalid, &mut looked_up)?;
    let failure = match increment(&looked_up, &mut out) {
        Err(Error::DependencyFailed(failure)) => *failure,
        other => panic!("expected a dependency failure, got {other:?}"),
    };
    assert_eq!(
        (
            failure.kernel.as_str(),
            failure.dispatch,
            failure.message.as_str()
        ),
        ("lookup", 1, "index out of bounds")
    );
    assert_eq!(out.read().wait(), [0; 4]);

    // An overwrite does not read the failed writes.
    detached_lookup(&device, &module, &invalid, &mut looked_up)?;
    // SAFETY: four threads each write their own element of a four-element buffer.
    unsafe {
        Launch::new(
            Ok(&fill),
            config,
            (&mut looked_up, 3_u32, 4_u32),
            |(out, value, n)| {
                Ok([
                    Argument::write(out),
                    Argument::value(*value)?,
                    Argument::value(*n)?,
                ])
            },
        )
        .wait()?
    };
    assert_eq!(looked_up.read().wait(), [3; 4]);

    detached_lookup(&device, &module, &valid, &mut looked_up)?;
    increment(&looked_up, &mut out)?;
    assert_eq!(out.read().wait(), [31; 4]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn overflow_division_and_assertion_messages_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let arithmetic = Pipeline::new(&device, &module(&device)?, "arithmetic")?;
    let run = |values: &[u32], divisor: u32| -> metal_oxide::Result<Vec<u32>> {
        let values = device.buffer_from_slice(values)?;
        let mut out = device.buffer_zeroed::<u32>(values.len())?;
        // SAFETY: both buffers cover n elements and each thread writes its own element.
        unsafe {
            Launch::new(
                Ok(&arithmetic),
                LaunchConfig::<32>::for_elements(values.len() as u32)?,
                (&values, &mut out, divisor, values.len() as u32),
                |(values, out, divisor, n)| {
                    Ok([
                        Argument::read(values),
                        Argument::write(out),
                        Argument::value(*divisor)?,
                        Argument::value(*n)?,
                    ])
                },
            )
        }
        .wait()?;
        Ok(out.read().wait().to_vec())
    };
    assert_eq!(run(&[1, 5, 9], 2)?, [1, 3, 5]);
    for (values, divisor, message, text) in [
        (
            &[u32::MAX][..],
            1,
            "attempt to add with overflow",
            "(value + 1)",
        ),
        (&[3][..], 0, "attempt to divide by zero", "(value + 1)"),
        (&[1, 7][..], 1, "seven is not allowed", "assert!"),
    ] {
        let failure = failure(run(values, divisor).map(|_| ()));
        assert_eq!(failure.message, message);
        assert_eq!(failure.line, line(text), "{failure}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_uniform_check_before_a_barrier_stops_every_thread() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = Pipeline::new(&device, &module(&device)?, "reverse_scaled")?;
    let mut out = device.buffer_zeroed::<u32>(128)?;
    for divisor in [2_u32, 0] {
        // SAFETY: two 64-thread blocks write their own 64 elements after the barrier.
        let result = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::new(Dim3::x(2)),
                (&mut out, divisor),
                |(out, divisor)| Ok([Argument::write(out), Argument::value(*divisor)?]),
            )
            .wait()
        };
        if divisor == 0 {
            assert_eq!(failure(result).message, "attempt to divide by zero");
        } else {
            result?;
            let expected = (0..128).map(|i| (63 - i % 64) * 32).collect::<Vec<_>>();
            assert_eq!(out.read().wait(), expected);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn failure_words_span_chunks_and_are_reused_without_leaking_state() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let lookup = Pipeline::new(&device, &module(&device)?, "lookup")?;
    let valid = device.buffer_from_slice(&[3_u32])?;
    let invalid = device.buffer_from_slice(&[4_u32])?;
    let mut out = device.buffer_zeroed::<u32>(1)?;
    let config = LaunchConfig::<1>::new(Dim3::x(1));
    // Every dispatch writes `out` after the previous one and takes a failure word, and every
    // one after the second a snapshot gate, so the words span three chunks.
    let dispatches = 1100;
    for failing in [true, false, true] {
        // SAFETY: one thread writes the single element in dispatch order.
        let result = sequence(|batch| unsafe {
            for dispatch in 0..dispatches {
                let indices = if failing && dispatch == dispatches - 1 {
                    &invalid
                } else {
                    &valid
                };
                batch.push(Launch::new(
                    Ok(&lookup),
                    config,
                    (indices, &mut out, 1_u32),
                    |(indices, out, n)| {
                        Ok([
                            Argument::read(indices),
                            Argument::write(out),
                            Argument::value(*n)?,
                        ])
                    },
                ))?;
            }
            Ok(())
        })
        .wait();
        if failing {
            assert_eq!(failure(result).dispatch, dispatches - 1);
        } else {
            result?;
        }
        assert_eq!(out.read().wait(), [40]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn device_log_messages_reach_standard_error() {
    const CHILD: &str = "METAL_OXIDE_LOG_CHILD";
    let device = Device::system_default().unwrap();
    // Apple GPUs log; the Mac2-only paravirtual device of hosted macOS VMs cannot.
    let logs = device.as_native().supportsFamily(MTLGPUFamily::Apple7);
    if std::env::var_os(CHILD).is_some() {
        let (output, directory) =
            support::emit("crates/metal-oxide-compiler/tests/fixtures/logging.rs", &[]);
        support::checked(output);
        let module = support::artifact::module(&device, &directory);
        if !logs {
            let Err(Error::UnsupportedDevice(reason)) = &module else {
                panic!("expected no shader logging, got {:?}", module.err());
            };
            assert!(reason.contains("shader logging"), "{reason}");
            return;
        }
        let module = module.unwrap();
        let pipeline = Pipeline::new(&device, &module, "logging").unwrap();
        let mut out = device.buffer_zeroed::<u32>(3).unwrap();
        // SAFETY: three threads write their own elements.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<3>::new(Dim3::x(1)),
                (&mut out, 0.1_f32, metal_oxide::F16::from_f32(0.5)),
                |(out, scale, half)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(*scale)?,
                        Argument::value(*half)?,
                    ])
                },
            )
            .wait()
            .unwrap();
        }
        // Metal delivers shader logs asynchronously around completion.
        std::thread::sleep(std::time::Duration::from_millis(500));
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "checks::device_log_messages_reach_standard_error",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    for thread in (0..3).filter(|_| logs) {
        let message = format!(
            "thread {thread} scale 0.100000001 half 0.5 wide 1099511627776 signed -3 -1099511627776 {{braces}} 100%"
        );
        assert!(stderr.contains(&message), "{stderr}");
    }
}

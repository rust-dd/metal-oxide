use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/range_loops.rs";

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn range_loops_match_rust_for_signed_empty_and_extreme_bounds() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(&device, FIXTURE, "range_sums")?;
    let mut bounds = vec![
        (0, 0),
        (3, 1),
        (-3, 3),
        (5, 6),
        (-10, -2),
        (0, 1000),
        (i32::MAX - 2, i32::MAX),
        (i32::MIN, i32::MIN + 3),
    ];
    // The fixture indexes by `thread_idx`, so one 256-thread block covers every pair.
    bounds.extend((0..248).map(|k| (k - 125, 2 * k - 200)));
    let n = bounds.len();
    let flat = bounds
        .iter()
        .flat_map(|&(start, end)| [start, end])
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&flat)?;
    let mut sums = device.buffer_zeroed::<i32>(n)?;
    let mut counts = device.buffer_zeroed::<u32>(n)?;
    // SAFETY: the launch covers exactly one bounds pair and one output pair per thread.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::new(Dim3::x(1)),
            (&input, &mut sums, &mut counts),
            |(input, sums, counts)| {
                Ok([
                    Argument::read(input),
                    Argument::write(sums),
                    Argument::write(counts),
                ])
            },
        )
    }
    .wait()?;
    for (i, &(start, end)) in bounds.iter().enumerate() {
        let want = (start..end).fold(0_i32, i32::wrapping_add);
        assert_eq!(sums.read().wait()[i], want, "{start}..{end}");
        assert_eq!(
            counts.read().wait()[i],
            (start..end).count() as u32,
            "{start}..{end}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn nested_byte_and_explicit_range_loops_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let nested = pipeline(&device, FIXTURE, "nested_ranges")?;
    let bytes = pipeline(&device, FIXTURE, "byte_ranges")?;
    let manual = pipeline(&device, FIXTURE, "manual_next")?;
    let (offset, n) = (11_u32, 64_u32);
    let limits = (0..=255_u8).collect::<Vec<_>>();
    let limit_buffer = device.buffer_from_slice(&limits)?;
    let mut tables = device.buffer_zeroed::<u32>(16 * 64)?;
    let mut sums = device.buffer_zeroed::<u32>(3 * limits.len())?;
    let mut products = device.buffer_zeroed::<u32>(n as usize)?;
    // SAFETY: each single-block launch has one thread per output the buffers hold.
    unsafe {
        Launch::new(
            Ok(&nested),
            LaunchConfig::<64>::new(Dim3::x(1)),
            (&mut tables, offset),
            |(tables, offset)| Ok([Argument::write(tables), Argument::value(*offset)?]),
        )
        .wait()?;
        Launch::new(
            Ok(&bytes),
            LaunchConfig::<256>::new(Dim3::x(1)),
            (&limit_buffer, &mut sums),
            |(limit_buffer, sums)| Ok([Argument::read(limit_buffer), Argument::write(sums)]),
        )
        .wait()?;
        Launch::new(
            Ok(&manual),
            LaunchConfig::<64>::new(Dim3::x(1)),
            (&mut products, n),
            |(products, n)| Ok([Argument::write(products), Argument::value(*n)?]),
        )
        .wait()?;
    }
    let mut table = [0_u32; 16];
    for row in 0..4 {
        for column in row..4 {
            table[row * 4 + column] = (row * 10 + column) as u32 + offset;
        }
    }
    for chunk in tables.read().wait().chunks(16) {
        assert_eq!(chunk, table);
    }
    for (i, &limit) in limits.iter().enumerate() {
        let want = [
            (250_u8..255).map(u32::from).sum::<u32>(),
            (limit..u8::MAX).map(|k| u32::from(k) + 1).sum(),
            (5_u8..limit).map(|k| u32::from(k) * 3).sum(),
        ];
        assert_eq!(&sums.read().wait()[3 * i..3 * i + 3], want, "limit {limit}");
    }
    for i in 0..n {
        let want = (i..n).fold(1_u32, |product, k| product.wrapping_mul(k + 2));
        assert_eq!(products.read().wait()[i as usize], want, "start {i}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn range_loops_keep_threadgroup_barriers_uniform() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/range_barrier.rs",
        "shared_rotations",
    )?;
    let blocks = 3_usize;
    let input = (0..64 * blocks as u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect::<Vec<_>>();
    let input_buffer = device.buffer_from_slice(&input)?;
    for rounds in [0_u32, 1, 5, 64, 70] {
        let mut output = device.buffer_zeroed::<u32>(input.len())?;
        // SAFETY: each 64-thread block reads and writes its own 64 values.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::new(Dim3::x(blocks as u32)),
                (&input_buffer, &mut output, rounds),
                |(input_buffer, output, rounds)| {
                    Ok([
                        Argument::read(input_buffer),
                        Argument::write(output),
                        Argument::value(*rounds)?,
                    ])
                },
            )
        }
        .wait()?;
        for (i, &actual) in output.read().wait().iter().enumerate() {
            let (block, lane) = (i / 64, i % 64);
            let source = input[64 * block + (lane + rounds as usize) % 64];
            assert_eq!(
                actual,
                source.wrapping_add(rounds),
                "rounds {rounds}, i {i}"
            );
        }
    }
    Ok(())
}

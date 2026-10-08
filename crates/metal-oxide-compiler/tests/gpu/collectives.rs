use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn numeric_collectives_match_active_lane_references() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let source = "crates/metal-oxide-compiler/tests/fixtures/simd_ops.rs";
    let unsigned = pipeline(&device, source, "collect_u32")?;
    let signed = pipeline(&device, source, "collect_i32")?;
    let float = pipeline(&device, source, "collect_f32")?;
    let width = unsigned.thread_execution_width() as u32;
    for block in [1, width - 1, width, width + 3, width * 2, width * 3 - 1] {
        let config = DynamicLaunchConfig::new(Dim3::x(2), Dim3::x(block));
        let mut u = device.buffer_zeroed::<u32>((block * 2 * 8) as usize)?;
        let mut i = device.buffer_zeroed::<i32>(u.len())?;
        let mut f = device.buffer_zeroed::<f32>(u.len())?;
        // SAFETY: all block threads participate; every shuffle reads an active lane or itself.
        unsafe {
            device.launch(&unsigned, config, &[Argument::write(&mut u)])?;
            device.launch(&signed, config, &[Argument::write(&mut i)])?;
            device.launch(&float, config, &[Argument::write(&mut f)])?;
        }
        for thread in 0..block * 2 {
            let lane = thread % block % width;
            let active = width.min(block - (thread % block / width) * width);
            let prefix = lane * (lane + 1) / 2;
            let expected = [
                active * (active + 1) / 2,
                1,
                active,
                prefix + lane + 1,
                prefix,
                lane + 1,
                lane.max(1),
                lane + 1,
            ];
            let offset = thread as usize * 8;
            assert_eq!(
                &u.as_slice()[offset..offset + 8],
                &expected,
                "block={block}, lane={lane}"
            );
            assert_eq!(
                &i.as_slice()[offset..offset + 8],
                &expected.map(|x| x as i32)
            );
            assert_eq!(
                &f.as_slice()[offset..offset + 8],
                &expected.map(|x| x as f32)
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn signed_collectives_compare_negative_values_and_wrap_at_extremes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/simd_ops.rs",
        "collect_i32_edges",
    )?;
    let width = pipeline.thread_execution_width() as usize;
    for block in [1, width - 1, width, width + 3, width * 2] {
        for pattern in [
            [i32::MAX, 1, -1, i32::MIN],
            [i32::MIN, -1, 1, i32::MAX],
            [-17, 5, -2, 11],
            [i32::MAX, i32::MAX, i32::MIN, i32::MIN],
        ] {
            let values = (0..block * 2)
                .map(|index| pattern[index % pattern.len()])
                .collect::<Vec<_>>();
            let input = device.buffer_from_slice(&values)?;
            let mut output = device.buffer_zeroed::<i32>(values.len() * 5)?;
            // SAFETY: every active lane participates; separate buffers cover every thread index.
            unsafe {
                device.launch(
                    &pipeline,
                    DynamicLaunchConfig::new(Dim3::x(2), Dim3::x(block as u32)),
                    &[Argument::read(&input), Argument::write(&mut output)],
                )?;
            }
            for (block_index, block_values) in values.chunks(block).enumerate() {
                for (group, lanes) in block_values.chunks(width).enumerate() {
                    let total = lanes.iter().copied().fold(0_i32, i32::wrapping_add);
                    let minimum = *lanes.iter().min().unwrap();
                    let maximum = *lanes.iter().max().unwrap();
                    let mut prefix = 0_i32;
                    for (lane, value) in lanes.iter().copied().enumerate() {
                        let index = block_index * block + group * width + lane;
                        let inclusive = prefix.wrapping_add(value);
                        assert_eq!(
                            &output.as_slice()[index * 5..(index + 1) * 5],
                            &[total, minimum, maximum, inclusive, prefix],
                            "block={block}, group={group}, lane={lane}, pattern={pattern:?}"
                        );
                        prefix = inclusive;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn votes_clear_unused_bits_and_bit_reductions_preserve_all_lanes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/simd_ops.rs",
        "vote",
    )?;
    let width = pipeline.thread_execution_width() as u32;
    for block in [1, width - 1, width, width + 3, width * 2] {
        let mut out = device.buffer_zeroed::<u32>((block * 8) as usize)?;
        // SAFETY: all active lanes vote and reduce; xor uses the current lane as its source.
        unsafe {
            device.launch(
                &pipeline,
                DynamicLaunchConfig::new(Dim3::x(1), Dim3::x(block)),
                &[Argument::write(&mut out)],
            )?;
        }
        for thread in 0..block {
            let lane = thread % width;
            let active = width.min(block - (thread / width) * width);
            let bits = (0..active)
                .filter(|i| i % 2 == 0)
                .fold(0_u64, |mask, i| mask | (1_u64 << i));
            let and = (1..=active).fold(u32::MAX, |a, b| a & b);
            let or = (1..=active).fold(0, |a, b| a | b);
            let xor = (1..=active).fold(0, |a, b| a ^ b);
            assert_eq!(
                &out.as_slice()[thread as usize * 8..(thread as usize + 1) * 8],
                &[
                    bits as u32,
                    (bits >> 32) as u32,
                    1,
                    u32::from(active == 1),
                    and,
                    or,
                    xor,
                    lane
                ]
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn integer_permutations_match_full_group_sources() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/simd_ops.rs",
        "full_permute",
    )?;
    let width = pipeline.thread_execution_width() as u32;
    let mut out = device.buffer_zeroed::<i32>((width * 2 * 3) as usize)?;
    // SAFETY: two full groups participate; down retains the last lane and XOR 1 selects active lanes.
    unsafe {
        device.launch(
            &pipeline,
            DynamicLaunchConfig::new(Dim3::x(1), Dim3::x(width * 2)),
            &[Argument::write(&mut out)],
        )?;
    }
    for thread in 0..width * 2 {
        let lane = thread % width;
        let next = (lane + 1).min(width - 1);
        assert_eq!(
            &out.as_slice()[thread as usize * 3..(thread as usize + 1) * 3],
            &[-1, -(next as i32) - 1, -((lane ^ 1) as i32) - 1]
        );
    }
    Ok(())
}

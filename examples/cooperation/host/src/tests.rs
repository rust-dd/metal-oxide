use super::{program::Program, verify};
use metal_oxide::{Device, sequence};

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn ordered_workloads_match_cpu_and_handwritten_msl() {
    verify::all().unwrap();
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn scan_propagates_offsets_across_three_levels() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let program = Program::generated(&device)?;
    let values = (0..1_000_003_u32)
        .map(|i| if i % 7 == 0 { u32::MAX } else { i % 11 })
        .collect::<Vec<_>>();
    verify::scan(&device, &program, &values)
}

#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn compaction_retains_resources_after_its_operation_is_forgotten() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let program = Program::generated(&device)?;
    let n = 1_000_003;
    let values = (0..n).map(|i| i % 127).collect::<Vec<u32>>();
    let input = device.buffer_from_slice(&values)?;
    let mut flags = device.buffer_zeroed::<u32>(values.len())?;
    let mut scan = super::workloads::Scan::new(&device, n)?;
    let mut output = device.buffer_zeroed::<u32>(values.len())?;
    let mut count = device.buffer_zeroed::<u32>(1)?;
    // SAFETY: ordered passes use disjoint full-sized buffers and unique prefix positions.
    std::mem::forget(verify::started(sequence(|batch| unsafe {
        program.mark(batch, &input, &mut flags, n)?;
        scan.encode(batch, &program, &flags)?;
        program.scatter(
            batch,
            &input,
            &flags,
            scan.output(),
            &mut output,
            &mut count,
            n,
        )
    })));
    drop(input);
    drop(flags);
    drop(scan);
    drop(program);
    let expected = values
        .into_iter()
        .filter(|value| value & 1 == 1)
        .collect::<Vec<_>>();
    assert_eq!(count.read().wait()[0] as usize, expected.len());
    assert_eq!(&output.read().wait()[..expected.len()], expected);
    Ok(())
}

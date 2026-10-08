use super::{program::Program, workloads::Scan};
use metal_oxide::{Device, Result};

pub(super) fn prefix(values: &[u32]) -> (Vec<u32>, u32) {
    let mut sum = 0_u32;
    let prefix = values
        .iter()
        .map(|&value| {
            let previous = sum;
            sum = sum.wrapping_add(value);
            previous
        })
        .collect::<Vec<_>>();
    (prefix, sum)
}

pub(super) fn scan(device: &Device, program: &Program<'_>, values: &[u32]) -> Result<()> {
    let input = device.buffer_from_slice(values)?;
    let mut scan = Scan::new(device, values.len() as u32)?;
    // SAFETY: complete blocks use neutral padding, and every global dependency follows its producer.
    let submission = unsafe { device.submit(|batch| scan.enqueue(batch, program, &input)) }?;
    drop(submission);
    let (expected, total) = prefix(values);
    assert_eq!(scan.output().as_slice(), expected);
    assert_eq!(scan.total(), total);
    Ok(())
}

pub(super) fn histogram(device: &Device, program: &Program<'_>, values: &[u32]) -> Result<()> {
    let input = device.buffer_from_slice(values)?;
    let mut bins = device.buffer_zeroed::<u32>(16)?;
    // SAFETY: initialized bins are exclusively atomic during the submitted 1D launch.
    let submission = unsafe {
        device.submit(|batch| program.histogram(batch, &input, &mut bins, values.len() as u32))
    }?;
    drop(submission);
    let mut expected = [0_u32; 16];
    for &value in values {
        expected[(value & 15) as usize] += 1;
    }
    assert_eq!(bins.as_slice(), expected);
    Ok(())
}

pub(super) fn compact(device: &Device, program: &Program<'_>, values: &[u32]) -> Result<()> {
    let n = values.len() as u32;
    let input = device.buffer_from_slice(values)?;
    let mut flags = device.buffer_zeroed::<u32>(values.len())?;
    let mut scan = Scan::new(device, n)?;
    let mut output = device.buffer_from_slice(&vec![u32::MAX; values.len()])?;
    let mut count = device.buffer_zeroed::<u32>(1)?;
    // SAFETY: ordered flag/scan/scatter passes give every selected input a distinct output position.
    let submission = unsafe {
        device.submit(|batch| {
            program.mark(batch, &input, &mut flags, n)?;
            scan.enqueue(batch, program, &flags)?;
            program.scatter(
                batch,
                &input,
                &flags,
                scan.output(),
                &mut output,
                &mut count,
                n,
            )
        })
    }?;
    drop(submission);
    drop(input);
    drop(flags);
    drop(scan);
    let expected = values
        .iter()
        .copied()
        .filter(|value| value & 1 == 1)
        .collect::<Vec<_>>();
    assert_eq!(count.as_slice()[0] as usize, expected.len());
    assert_eq!(&output.as_slice()[..expected.len()], expected);
    assert!(
        output.as_slice()[expected.len()..]
            .iter()
            .all(|&value| value == u32::MAX)
    );
    Ok(())
}

pub(super) fn all() -> Result<()> {
    let device = Device::system_default()?;
    for program in [Program::generated(&device)?, Program::reference(&device)?] {
        for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
            let values = (0..n).map(|i| i % 17).collect::<Vec<_>>();
            histogram(&device, &program, &values)?;
            scan(&device, &program, &values)?;
            compact(&device, &program, &values)?;
        }
        histogram(&device, &program, &vec![13; 1_000_003])?;
        scan(&device, &program, &[u32::MAX, 1, 9, u32::MAX])?;
        compact(&device, &program, &vec![1; 257])?;
        compact(&device, &program, &vec![2; 257])?;
    }
    println!(
        "histogram, scan, compaction: CPU/MSL verified on {}",
        device.name()
    );
    Ok(())
}

use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    task::{Context, Waker},
};

use super::{program::Program, workloads::Scan};
use metal_oxide::{Device, Result, sequence};

/// Polls an operation once, which encodes and commits it, and returns its future.
pub(super) fn started<O: IntoFuture<IntoFuture: Unpin>>(operation: O) -> O::IntoFuture {
    let mut future = operation.into_future();
    let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
    future
}

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

pub(super) fn scan(device: &Device, program: &Program, values: &[u32]) -> Result<()> {
    let input = device.buffer_from_slice(values)?;
    let mut scan = Scan::new(device, values.len() as u32)?;
    // SAFETY: complete blocks use neutral padding, and every global dependency follows its producer.
    drop(started(sequence(|batch| unsafe {
        scan.encode(batch, program, &input)
    })));
    let (expected, total) = prefix(values);
    assert_eq!(scan.output().read().wait(), expected);
    assert_eq!(scan.total(), total);
    Ok(())
}

pub(super) fn histogram(device: &Device, program: &Program, values: &[u32]) -> Result<()> {
    let input = device.buffer_from_slice(values)?;
    let mut bins = device.buffer_zeroed::<u32>(16)?;
    // SAFETY: initialized bins are exclusively atomic during the 1D launch.
    drop(started(sequence(|batch| unsafe {
        program.histogram(batch, &input, &mut bins, values.len() as u32)
    })));
    let mut expected = [0_u32; 16];
    for &value in values {
        expected[(value & 15) as usize] += 1;
    }
    assert_eq!(bins.read().wait(), expected);
    Ok(())
}

pub(super) fn compact(device: &Device, program: &Program, values: &[u32]) -> Result<()> {
    let n = values.len() as u32;
    let input = device.buffer_from_slice(values)?;
    let mut flags = device.buffer_zeroed::<u32>(values.len())?;
    let mut scan = Scan::new(device, n)?;
    let mut output = device.buffer_from_slice(&vec![u32::MAX; values.len()])?;
    let mut count = device.buffer_zeroed::<u32>(1)?;
    // SAFETY: ordered flag/scan/scatter passes give every selected input a distinct output position.
    drop(started(sequence(|batch| unsafe {
        program.mark(batch, &input, &mut flags, n)?;
        scan.encode(batch, program, &flags)?;
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
    let expected = values
        .iter()
        .copied()
        .filter(|value| value & 1 == 1)
        .collect::<Vec<_>>();
    assert_eq!(count.read().wait()[0] as usize, expected.len());
    assert_eq!(&output.read().wait()[..expected.len()], expected);
    assert!(
        output.read().wait()[expected.len()..]
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

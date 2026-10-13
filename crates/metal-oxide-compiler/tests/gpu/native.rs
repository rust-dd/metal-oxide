use super::*;
use metal_oxide::Buffer;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void increments(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < 33 * 65) {
        uint a = (i + 1) * 2654435761u;
        uint b = (i + 7) * 2246822519u;
        float u = float(a >> 8) * 0x1p-24f + 0x1p-24f;
        float v = float(b >> 8) * 0x1p-24f;
        values[i] = sqrt(-2.0f * log(u)) * cos(6.28318530718f * v) * 0.125f;
    } else {
        values[i] = -8192.0f;
    }
}
"#;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn native_increments_feed_generated_integration_without_staging() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/integrate.rs",
        "integrate",
    )?;
    let native = device.as_native();
    let library = native
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
        .unwrap();
    let function = library
        .newFunctionWithName(&NSString::from_str("increments"))
        .unwrap();
    let producer = native
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap();
    let queue = native.newCommandQueue().unwrap();
    let len = 33 * 65 + 32;
    for completion in 0..3 {
        let raw = native
            .newBufferWithLength_options(len * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let address = raw.contents().as_ptr();
        let command = queue.commandBuffer().unwrap();
        let encoder = command.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&producer);
        // SAFETY: every native thread initializes its unique f32 in the allocation.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&raw), 0, 0);
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: len,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: external work completed; all f32 values are initialized and
        // ownership is handed over without any externally accessible buffer alias.
        let mut values = unsafe { Buffer::<f32>::from_native(&device, raw, len)? };
        // SAFETY: each active row owns 65 initialized elements; padded lanes do nothing.
        let operation = sequence(|batch| unsafe {
            batch
                .push(Launch::new(
                    Ok(&pipeline),
                    LaunchConfig::<32>::for_elements(33)?,
                    (&mut values, 33_u32, 65_u32, 0.015625_f32),
                    |(values, v1, v2, v3)| {
                        Ok([
                            Argument::read_write(values),
                            Argument::value(*v1)?,
                            Argument::value(*v2)?,
                            Argument::value(*v3)?,
                        ])
                    },
                ))
                .map(drop)?;
            Ok(())
        });
        match completion {
            0 => {
                operation.wait()?;
            }
            1 => drop(started(operation)),
            _ => std::mem::forget(started(operation)),
        }
        let raw = values.into_native();
        assert_eq!(raw.contents().as_ptr(), address);
        // SAFETY: export completed every access and the allocation contains len f32 values.
        let actual =
            unsafe { std::slice::from_raw_parts(raw.contents().cast::<f32>().as_ptr(), len) };
        for row in 0..33 {
            let mut state = 1.0_f32;
            for step in 0..65 {
                let i = (row * 65 + step) as u32;
                let a = (i + 1).wrapping_mul(2654435761);
                let b = (i + 7).wrapping_mul(2246822519);
                let u = ((a >> 8) as f32 + 1.0) / 16777216.0;
                let v = (b >> 8) as f32 / 16777216.0;
                let increment = (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos() * 0.125;
                state = state + 0.25 * state * 0.015625 + increment;
                assert!((actual[i as usize] - state).abs() < 3e-5 * state.abs().max(1.0));
            }
        }
        assert!(actual[33 * 65..].iter().all(|&x| x == -8192.0));
    }
    Ok(())
}

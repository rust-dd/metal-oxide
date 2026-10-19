#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide::{
    Argument, Buffer, Device, Dim3, Launch, LaunchConfig, Module, Operation, Pipeline,
};
use metal_oxide_artifact::{Manifest, MathMode, sha256};
use std::path::Path;
use support::Project;

const RUST: &str =
    include_str!("../../../crates/metal-oxide-compiler/tests/fixtures/transcendentals.rs");
const MSL: &str =
    include_str!("../../../crates/metal-oxide-compiler/tests/fixtures/transcendentals.metal");
const FLOAT: &str =
    include_str!("../../../crates/metal-oxide-compiler/tests/fixtures/float_math.rs");

const HOST: &str = r#"
#[metal_oxide::kernels(path = "kernels")]
mod precise {}

#[metal_oxide::kernels(path = "kernels", math = "fast")]
mod fast {}

fn main() {}
"#;

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn math_modes_build_separate_artifacts_that_match_the_reference() -> metal_oxide::Result<()> {
    let project = Project::new(HOST);
    project.kernel_crate(&format!("{RUST}\nmod float_math;\n"));
    std::fs::write(
        project.0.join("kernels/src/float_math.rs"),
        FLOAT.replace("#![no_std]\n", ""),
    )?;
    let build = || {
        let mut command = project.cargo(&["build", "--quiet"]);
        project.built(command.env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", "false"))
    };
    let built = build();
    let (precise, fast) = (&built["precise"], &built["fast"]);
    assert_ne!(precise, fast);
    // An edit outside the kernels reuses both artifacts.
    project.write(&format!("{HOST}// edited\n"));
    assert_eq!(build(), built);

    let device = Device::system_default()?;
    let mut reports = Vec::new();
    let mut abi_hashes = Vec::<String>::new();
    for (mode, directory) in [(MathMode::Precise, precise), (MathMode::Fast, fast)] {
        let manifest =
            Manifest::from_json(&std::fs::read_to_string(directory.join("manifest.json"))?)?;
        assert_eq!(manifest.abi.math_mode, mode);
        let abi_hash = sha256(manifest.abi.to_json()?.as_bytes());
        let offline = Module::from_artifact(&device, directory)?;
        offline.verify_abi(&abi_hash)?;
        for other in &abi_hashes {
            assert!(offline.verify_abi(other).is_err());
        }
        abi_hashes.push(abi_hash);
        let reference = if mode == MathMode::Fast {
            format!("#define METAL_OXIDE_FAST_MATH\n{MSL}")
        } else {
            MSL.into()
        };
        let reference = Module::from_source(&device, &reference)?;
        verify_fusion(&device, &offline, mode)?;
        let pipelines = [&offline, &reference]
            .map(|module| Pipeline::new(&device, module, "box_muller_step"))
            .into_iter()
            .collect::<metal_oxide::Result<Vec<_>>>()?;
        let mut timing = Vec::new();
        for n in [0, 1, 255, 256, 257, 1_000_003] {
            timing = box_muller(&device, &pipelines, mode, n)?;
        }
        reports.push(serde_json::json!({
            "math_mode": mode, "manifest": manifest, "elements": 1_000_003,
            "warmup": 5, "variants": ["offline", "reference"],
            "gpu_samples_ns": timing,
        }));
    }
    let backend = std::env::var("METAL_OXIDE_BACKEND").unwrap_or_else(|_| "auto".into());
    let report = serde_json::json!({"device": device.name(), "backend": backend, "cases": reports});
    let output = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("math-modes-{backend}.json"));
    std::fs::write(&output, serde_json::to_vec_pretty(&report).unwrap())?;
    eprintln!(
        "Math correctness and matched-mode timings: {}",
        output.display()
    );
    Ok(())
}

fn verify_fusion(device: &Device, module: &Module, mode: MathMode) -> metal_oxide::Result<()> {
    let pipeline = Pipeline::new(device, module, "float_math")?;
    let mut output = device.buffer_zeroed::<f32>(7)?;
    let a = 1.0_f32 + f32::EPSILON;
    let b = 1.0_f32 - f32::EPSILON;
    // SAFETY: one thread owns all seven elements; the remaining parameters are f32.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&mut output, a, b, -1.0_f32),
            |(output, a, b, c)| {
                Ok([
                    Argument::write(output),
                    Argument::value(*a)?,
                    Argument::value(*b)?,
                    Argument::value(*c)?,
                ])
            },
        )
    }
    .wait()?;
    let fused = a.mul_add(b, -1.0);
    assert_eq!(output.read().wait()[4], fused);
    assert_eq!(
        output.read().wait()[5],
        if mode == MathMode::Fast { fused } else { 0.0 }
    );
    Ok(())
}

fn box_muller(
    device: &Device,
    pipelines: &[Pipeline],
    mode: MathMode,
    n: u32,
) -> metal_oxide::Result<Vec<Vec<Option<u64>>>> {
    let uniforms = (0..n)
        .map(|i| (i % 1024 + 1) as f32 / 1024.0)
        .collect::<Vec<_>>();
    // Keeping angles in [0, pi] stays within fast cosine's stated error bound.
    let angles = (0..n)
        .map(|i| (i % 997) as f32 / 1994.0)
        .collect::<Vec<_>>();
    let expected = uniforms
        .iter()
        .zip(&angles)
        .map(|(&u, &v)| {
            let z =
                (-2.0 * (u as f64).ln()).sqrt() * (core::f32::consts::TAU as f64 * v as f64).cos();
            (100.0 * (0.01_f32 as f64 + 0.2_f32 as f64 * z).exp()) as f32
        })
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&uniforms)?;
    let angles = device.buffer_from_slice(&angles)?;
    let mut output = device.buffer_from_slice(&vec![-8192.0_f32; n as usize + 128])?;
    let mut results = Vec::new();
    for pipeline in pipelines {
        launch(pipeline, &input, &angles, &mut output, n)?;
        let actual = output.read().wait();
        assert!(actual[n as usize..].iter().all(|&x| x == -8192.0));
        let tolerance = if mode == MathMode::Fast { 2e-4 } else { 2e-5 };
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs(),
                "{mode:?}, n={n}, i={index}: {actual} != {expected}"
            );
        }
        results.push(actual[..n as usize].to_vec());
    }
    for result in &results[1..] {
        for (&actual, &expected) in result.iter().zip(&results[0]) {
            if mode == MathMode::Precise {
                assert_eq!(actual, expected);
            } else {
                assert!((actual - expected).abs() <= 2e-6 * expected.abs());
            }
        }
    }
    let mut samples = vec![Vec::new(); pipelines.len()];
    if n > 1_000_000 {
        for iteration in 0..25 {
            for offset in 0..pipelines.len() {
                let index = (iteration + offset) % pipelines.len();
                let time = launch(&pipelines[index], &input, &angles, &mut output, n)?;
                if iteration >= 5 {
                    samples[index].push(time);
                }
            }
        }
    }
    Ok(samples)
}

fn launch(
    pipeline: &Pipeline,
    input: &Buffer<f32>,
    angles: &Buffer<f32>,
    output: &mut Buffer<f32>,
    n: u32,
) -> metal_oxide::Result<Option<u64>> {
    let config = LaunchConfig::<128>::for_elements(n)?;
    // SAFETY: disjoint buffers cover n; each guarded 1D thread owns one output element.
    let operation = unsafe {
        Launch::new(
            Ok(pipeline),
            config,
            (input, angles, output, n),
            |(input, angles, output, n)| {
                Ok([
                    Argument::read(input),
                    Argument::read(angles),
                    Argument::write(output),
                    Argument::value(*n)?,
                ])
            },
        )
    };
    let (_, time) = operation.timed().wait()?;
    Ok(time.map(|duration| duration.as_nanos().try_into().unwrap()))
}

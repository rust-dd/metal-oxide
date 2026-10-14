#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide::{Argument, Device, Error, Launch, LaunchConfig, Module, Operation, Pipeline};
use metal_oxide_artifact::{Manifest, sha256};
use std::process::Command;
use support::{Project, checked};

/// The program that launches the kernels: generated functions for the whole library, lazy
/// loading, pipeline reuse, and argument checks that run before anything is encoded.
const HOST: &str = r###"#![deny(warnings)]

#[metal_oxide::kernels(path = "kernels")]
#[allow(dead_code)]
mod kernels {}

fn main() -> metal_oxide::Result<()> {
    use metal_oxide::{
        Argument, Device, Dim3, Error, Launch, LaunchConfig, Operation, Pipeline, sequence,
    };
    let device = Device::system_default()?;
    let start = std::time::Instant::now();
    let module = kernels::module(&device)?;
    println!("embedded load: {:?}", start.elapsed());
    let mut values = device.buffer_from_slice(&[0xffff_ffff_0000_0000_u64; 25])?;
    let config = LaunchConfig::<8>::for_elements(17)?;
    let start = std::time::Instant::now();
    // SAFETY: 17 distinct lanes update initialized elements; padding remains untouched.
    unsafe { kernels::in_place(&device, config, &mut values, 7, 17) }.wait()?;
    println!("first pipeline and dispatch: {:?}", start.elapsed());
    let start = std::time::Instant::now();
    for _ in 0..10 {
        // SAFETY: the same in-bounds launch reuses its initialized pipeline and buffer.
        unsafe { kernels::in_place(&device, config, &mut values, 0, 17) }.wait()?;
    }
    println!("ten cached dispatches: {:?}", start.elapsed());
    // SAFETY: ordered dispatches reuse the same exclusive buffer and stay within 17 elements.
    sequence(|batch| unsafe {
        batch.push(kernels::in_place(&device, config, &mut values, 9, 17))?;
        batch.push(kernels::in_place(&device, config, &mut values, 11, 17))?;
        Ok(())
    })
    .wait()?;
    // SAFETY: this entry has no arguments or memory accesses.
    unsafe { kernels::k317(&device, LaunchConfig::<1>::new(Dim3::x(1))) }.wait()?;
    assert_eq!(&values.read().wait()[..17], &[0xffff_ffff_0000_001b; 17]);
    assert_eq!(&values.read().wait()[17..], &[0xffff_ffff_0000_0000; 8]);

    assert!(module.verify_abi(&"0".repeat(64)).is_err());
    let input = device.buffer_from_slice(&[1.0_f32; 17])?;
    let mut reductions = device.buffer_zeroed::<f32>(3)?;
    let blocks = LaunchConfig::<8>::for_elements(17)?;
    // SAFETY: both scratch counts match the block width, all lanes initialize
    // their cells and participate, and three blocks own separate output elements.
    unsafe { kernels::dynamic_reduce(&device, blocks, &input, 8, 0.5, &mut reductions, 8, 17) }
        .wait()?;
    // SAFETY: as above.
    sequence(|batch| unsafe {
        batch.push(kernels::dynamic_reduce(
            &device,
            blocks,
            &input,
            8,
            0.5,
            &mut reductions,
            8,
            17,
        ))?;
        Ok(())
    })
    .wait()?;
    assert_eq!(reductions.read().wait(), [75.0, 203.0, 40.5]);
    let dynamic = Pipeline::new(&device, &module, "dynamic_reduce")?;
    let empty = LaunchConfig::<8>::new(Dim3::x(0));
    // SAFETY: the empty grid accesses no memory; mismatched scratch types and
    // counts must still be rejected through the artifact contract.
    unsafe {
        let scratch = Launch::new(
            Ok(&dynamic),
            empty,
            (&input, &mut reductions),
            |(input, reductions)| {
                Ok([
                    Argument::read(input),
                    Argument::threadgroup::<u32>(8)?,
                    Argument::value(0.5_f32)?,
                    Argument::write(reductions),
                    Argument::threadgroup::<(u8, u64)>(8)?,
                    Argument::value(17_u32)?,
                ])
            },
        );
        assert!(matches!(scratch.wait(), Err(Error::Artifact(_))));
        let count = Launch::new(
            Ok(&dynamic),
            empty,
            (&input, &mut reductions),
            |(input, reductions)| {
                Ok([
                    Argument::read(input),
                    Argument::value(8_u32)?,
                    Argument::value(0.5_f32)?,
                    Argument::write(reductions),
                    Argument::threadgroup::<(u8, u64)>(8)?,
                    Argument::value(17_u32)?,
                ])
            },
        );
        assert!(matches!(count.wait(), Err(Error::Artifact(_))));
        let none = Launch::new(Ok(&dynamic), empty, (), |()| Ok([]));
        assert!(matches!(none.wait(), Err(Error::Artifact(_))));
    }
    let pipeline = Pipeline::new(&device, &module, "in_place")?;
    // SAFETY: empty grids cannot access memory; invalid access, scalar layout and
    // specialization must still be rejected before the empty-launch fast path.
    unsafe {
        let read = Launch::new(Ok(&pipeline), empty, (&values,), |(values,)| {
            Ok([
                Argument::read(values),
                Argument::value(1_u64)?,
                Argument::value(17_u32)?,
            ])
        });
        assert!(matches!(read.wait(), Err(Error::Artifact(_))));
        let write = Launch::new(Ok(&pipeline), empty, (&mut values,), |(values,)| {
            Ok([
                Argument::write(values),
                Argument::value(1_u64)?,
                Argument::value(17_u32)?,
            ])
        });
        assert!(matches!(write.wait(), Err(Error::Artifact(_))));
        let layout = Launch::new(Ok(&pipeline), empty, (&mut values,), |(values,)| {
            Ok([
                Argument::read_write(values),
                Argument::value(1_u32)?,
                Argument::value(17_u32)?,
            ])
        });
        assert!(matches!(layout.wait(), Err(Error::Artifact(_))));
        let block = LaunchConfig::<32>::new(Dim3::x(0));
        let shape = Launch::new(Ok(&pipeline), block, (&mut values,), |(values,)| {
            Ok([
                Argument::read_write(values),
                Argument::value(1_u64)?,
                Argument::value(17_u32)?,
            ])
        });
        assert!(matches!(shape.wait(), Err(Error::InvalidLaunch(_))));
    }
    println!("embedded consumer passed");
    Ok(())
}
"###;

fn kernels() -> String {
    let mut kernels = String::from(
        r#"#![no_std]
use metal_oxide_device::{ReadWriteBuffer, block_dim, block_idx, kernel, thread_idx};
#[kernel(block = (8, 1, 1))]
pub unsafe fn in_place(values: ReadWriteBuffer<u64>, increment: u64, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n { unsafe { values.store_unchecked(i, values.load_unchecked(i).wrapping_add(increment)); } }
}
"#,
    );
    kernels.push_str("mod cooperation {\n");
    kernels.push_str(
        include_str!("../../metal-oxide-compiler/tests/fixtures/dynamic_shared.rs")
            .trim_start_matches("#![no_std]"),
    );
    kernels.push_str("\n}\n");
    for index in 1..483 {
        kernels.push_str(&format!("#[kernel]\npub unsafe fn k{index}() {{}}\n"));
    }
    kernels
}

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn kernel_libraries_load_lazily_in_a_relocated_host_built_on_stable_rust() -> metal_oxide::Result<()>
{
    let project = Project::new(HOST);
    let kernels = kernels();
    project.kernel_crate(&kernels);
    let directory = project
        .built(&mut project.cargo(&["build", "--release", "--quiet"]))
        .remove("kernels")
        .unwrap();
    let manifest_text = std::fs::read_to_string(directory.join("manifest.json"))?;
    let mut manifest = Manifest::from_json(&manifest_text)?;
    assert_eq!(manifest.abi.kernels.len(), 484);
    let library = std::fs::read(directory.join("kernels.metallib"))?;
    let device = Device::system_default()?;
    let module = Module::from_bytes(&device, &manifest_text, &library)?;
    module.verify_abi(&sha256(manifest.abi.to_json()?.as_bytes()))?;
    let pipeline = Pipeline::new(&device, &module, "in_place")?;
    verify_argument_types(&device, &pipeline)?;
    verify_rejections(&device, &manifest_text, &library);

    // Another library fails the digest; under a matching digest, its missing entrypoint fails
    // when it is requested, and the others still load.
    let source = project.0.join("kernels/src/lib.rs");
    std::fs::write(&source, kernels.replace("fn k482()", "fn renamed()"))?;
    let other = project
        .built(&mut project.cargo(&["build", "--release", "--quiet"]))
        .remove("kernels")
        .unwrap();
    let other_library = std::fs::read(other.join("kernels.metallib"))?;
    assert!(matches!(
        Module::from_bytes(&device, &manifest_text, &other_library),
        Err(Error::Artifact(_))
    ));
    manifest.files.metallib = sha256(&other_library);
    let swapped = Module::from_bytes(&device, &manifest.to_json()?, &other_library)?;
    assert!(matches!(
        Pipeline::new(&device, &swapped, "k482"),
        Err(Error::KernelNotFound(name)) if name == "k482"
    ));
    Pipeline::new(&device, &swapped, "in_place")?;
    std::fs::write(&source, &kernels)?;
    checked(project.cargo(&["build", "--release", "--quiet"]).output()?);

    // The saved artifact builds the host on the oldest supported Rust, without the kernel
    // toolchain; the moved binary runs with an empty environment.
    let stable = project.0.join("stable");
    checked(
        project
            .cargo(&["build", "--release", "--locked", "--offline", "--quiet"])
            .env("CARGO_TARGET_DIR", &stable)
            .env("RUSTUP_TOOLCHAIN", "1.89.0")
            .env("METAL_OXIDE_ARTIFACTS", "locked")
            .output()?,
    );
    let moved = project.0.join("moved");
    std::fs::create_dir(&moved)?;
    let binary = moved.join("consumer");
    std::fs::copy(stable.join("release/host"), &binary)?;
    std::fs::remove_dir_all(stable)?;
    let output = checked(
        Command::new(&binary)
            .current_dir(&moved)
            .env_clear()
            .env("PATH", "")
            .env(
                "METAL_OXIDE_BACKEND",
                std::env::var("METAL_OXIDE_BACKEND").unwrap_or_default(),
            )
            .output()?,
    );
    let report = String::from_utf8(output.stdout).unwrap();
    assert!(report.contains("embedded consumer passed"), "{report}");
    eprintln!("{report}");
    Ok(())
}

/// A slot checked for one Rust type still rejects another type in a later launch.
fn verify_argument_types(device: &Device, pipeline: &Pipeline) -> metal_oxide::Result<()> {
    let mut values = device.buffer_from_slice(&[1_u64, 2, 3])?;
    let mut narrow = device.buffer_from_slice(&[1_u32, 2, 3])?;
    let config = LaunchConfig::<8>::for_elements(3)?;
    for _ in 0..2 {
        // SAFETY: three guarded threads update their own elements in place.
        unsafe {
            Launch::new(
                Ok(pipeline),
                config,
                (&mut values, 5_u64, 3_u32),
                |(values, increment, n)| {
                    Ok([
                        Argument::read_write(values),
                        Argument::value(*increment)?,
                        Argument::value(*n)?,
                    ])
                },
            )
        }
        .wait()?;
    }
    assert_eq!(values.read().wait(), [11, 12, 13]);
    // SAFETY: validation rejects the arguments before anything is encoded.
    let scalar = unsafe {
        Launch::new(Ok(pipeline), config, (&mut values,), |(values,)| {
            Ok([
                Argument::read_write(values),
                Argument::value(5_u32)?,
                Argument::value(3_u32)?,
            ])
        })
    };
    assert!(matches!(scalar.wait(), Err(Error::Artifact(_))));
    // SAFETY: as above.
    let element = unsafe {
        Launch::new(Ok(pipeline), config, (&mut narrow,), |(narrow,)| {
            Ok([
                Argument::read_write(narrow),
                Argument::value(5_u64)?,
                Argument::value(3_u32)?,
            ])
        })
    };
    assert!(matches!(element.wait(), Err(Error::Artifact(_))));
    assert_eq!(values.read().wait(), [11, 12, 13]);
    Ok(())
}

fn verify_rejections(device: &Device, manifest: &str, library: &[u8]) {
    for bytes in [&[][..], &library[..library.len() / 2]] {
        assert!(matches!(
            Module::from_bytes(device, manifest, bytes),
            Err(Error::Artifact(_))
        ));
    }
    assert!(Module::from_bytes(device, "{}", library).is_err());
    let original = serde_json::from_str::<serde_json::Value>(manifest).unwrap();
    for (key, value) in [
        ("version", serde_json::json!(999)),
        ("math_mode", serde_json::json!("unknown")),
        ("required_features", serde_json::json!(["unknown"])),
    ] {
        let mut invalid = original.clone();
        invalid["abi"][key] = value;
        assert!(matches!(
            Module::from_bytes(device, &invalid.to_string(), library),
            Err(Error::Artifact(_))
        ));
    }
}

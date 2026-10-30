#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use support::{Project, checked};

const FORMS: &str = r#"
use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};
use metal_oxide::{Device, LaunchConfig, Operation};

#[metal_oxide::kernel(requires(a.len() == out.len()))]
pub fn twice(a: ReadBuffer<u32>, mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        *out = a.as_slice()[index.get()] * 2;
    }
}

/// Math methods resolve to std on the host and to the device trait in the kernel build.
#[metal_oxide::kernel(requires(sides.len() == out.len()))]
pub fn hypotenuse(sides: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        let side = sides.as_slice()[index.get()];
        *out = side.hypot(4.0) + side.mul_add(2.0, 1.0).sqrt();
    }
}

#[metal_oxide::kernels]
mod bodies {
    use metal_oxide::device::*;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Body {
        pub mass: f32,
        pub position: [f32; 2],
    }

    pub fn moved(body: Body, by: f32) -> Body {
        Body { position: [body.position[0] + by, body.position[1] - by], ..body }
    }

    #[kernel(block = (32, 1, 1))]
    pub fn push(mut bodies: DisjointBuffer<Body>, by: f32) {
        let index = thread_index();
        if let Some(body) = bodies.get_mut(&index) {
            *body = moved(*body, by);
        }
    }
}

#[metal_oxide::kernels(path = "{root}/crates/metal-oxide-compiler/tests/crates/vec-add-kernels")]
mod added {}

fn main() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let a = device.buffer_from_slice(&[1_u32, 2, 3])?;
    let mut out = device.buffer_zeroed::<u32>(3)?;
    twice(&device, LaunchConfig::<64>::for_elements(3)?, &a, &mut out).wait()?;
    assert_eq!(out.read().wait(), [2, 4, 6]);
    let short = device.buffer_zeroed::<u32>(2)?;
    let violated = twice(&device, LaunchConfig::<64>::for_elements(3)?, &short, &mut out).wait();
    assert!(matches!(violated, Err(metal_oxide::Error::Contract { .. })));

    let sides = device.buffer_from_slice(&[3.0_f32, 0.0])?;
    let mut lengths = device.buffer_zeroed::<f32>(2)?;
    hypotenuse(&device, LaunchConfig::<64>::for_elements(2)?, &sides, &mut lengths).wait()?;
    for (length, expected) in lengths.read().wait().iter().zip([5.0 + 7.0_f32.sqrt(), 5.0]) {
        assert!((length - expected).abs() <= 1e-6, "{length} != {expected}");
    }

    let body = bodies::Body { mass: 1.0, position: [0.0, 0.0] };
    let mut all = device.buffer_from_slice(&[body; 5])?;
    bodies::push(&device, LaunchConfig::<32>::for_elements(5)?, &mut all, 0.5).wait()?;
    assert_eq!(all.read().wait(), [bodies::moved(body, 0.5); 5]);

    let x = device.buffer_from_slice(&[1.0_f32, 2.0])?;
    let mut y = device.buffer_zeroed::<f32>(2)?;
    added::vec_add(&device, LaunchConfig::<64>::for_elements(2)?, &x, &x, &mut y).wait()?;
    assert_eq!(y.read().wait(), [2.0, 4.0]);
    metal_oxide::Pipeline::new(&device, &added::module(&device)?, "vec_add")?;
    println!("forms passed");
    Ok(())
}
"#;

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn kernels_in_functions_modules_and_crates_launch_and_share_records() {
    let project = Project::new(FORMS);
    let output = checked(project.cargo(&["run", "--quiet"]).output().unwrap());
    assert!(String::from_utf8_lossy(&output.stdout).contains("forms passed"));

    // A moved binary embeds its kernels and needs no kernel toolchain.
    checked(
        project
            .cargo(&["build", "--release", "--quiet"])
            .output()
            .unwrap(),
    );
    let moved = project.0.join("moved");
    std::fs::create_dir(&moved).unwrap();
    std::fs::copy(Project::target().join("release/host"), moved.join("host")).unwrap();
    let output = checked(
        std::process::Command::new(moved.join("host"))
            .current_dir(&moved)
            .env_clear()
            .env(
                "METAL_OXIDE_BACKEND",
                std::env::var("METAL_OXIDE_BACKEND").unwrap_or_default(),
            )
            .output()
            .unwrap(),
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("forms passed"));
}

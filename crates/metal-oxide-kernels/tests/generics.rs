#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use support::{Project, checked, failure};

const KERNEL: &str = r#"
#[kernel(
    instances(scale_f32(f32, 2), scale_i32(i32, 3), scale_u32(u32, 1), r#type(u32, 1)),
    block = (32, 1, 1),
    requires(input.len() == output.len())
)]
pub fn scale<T: Copy + core::ops::Add<Output = T>, const N: usize>(
    r#input: ReadBuffer<T>, mut output: DisjointBuffer<T>,
) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let mut value = input.as_slice()[index.get()];
        for _ in 0..N { value = value + value; }
        *output = value;
    }
}
"#;

const MAIN: &str = r#"
fn main() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let config = LaunchConfig::<32>::for_elements(3)?;
    let floats = device.buffer_from_slice(&[1.5_f32, -2.0, 0.0])?;
    let integers = device.buffer_from_slice(&[-3_i32, 4, 0])?;
    let unsigned = device.buffer_from_slice(&[7_u32, 0, u32::MAX])?;
    let mut f = device.buffer_zeroed::<f32>(3)?;
    let mut i = device.buffer_zeroed::<i32>(3)?;
    let mut u = device.buffer_zeroed::<u32>(3)?;
    scale_f32(&device, config, &floats, &mut f).wait()?;
    assert_eq!(f.read().wait(), [6.0, -8.0, 0.0]);
    inline::scale_i32(&device, config, &integers, &mut i).wait()?;
    assert_eq!(i.read().wait(), [-24, 32, 0]);
    imported::scale_u32(&device, config, &unsigned, &mut u).wait()?;
    assert_eq!(u.read().wait(), [14, 0, u32::MAX - 1]);
    inline::r#type(&device, config, &unsigned, &mut u).wait()?;
    imported::r#type(&device, config, &unsigned, &mut u).wait()?;
    assert_eq!(u.read().wait(), [14, 0, u32::MAX - 1]);
    let short = device.buffer_zeroed::<f32>(2)?;
    assert!(matches!(scale_f32(&device, config, &short, &mut f).wait(),
        Err(metal_oxide::Error::Contract { .. })));
    let pairs = device.buffer_from_slice(&[[1.5_f32, -2.0], [3.0, 4.5]])?;
    let mut copied = device.buffer_zeroed::<[f32; 2]>(2)?;
    arrays::copy_pairs(&device, config, &pairs, &mut copied).wait()?;
    assert_eq!(copied.read().wait(), [[1.5, -2.0], [3.0, 4.5]]);
    println!("generic launches passed");
    Ok(())
}
"#;

const ARRAYS: &str = r#"
const PAIR: usize = 2;

pub trait Element { type Value: Copy; }
impl Element for u32 { type Value = f32; }

#[kernel(instances(copy_pairs(u32, PAIR)), requires(input.len() == output.len()))]
pub fn copy<T: Element, const N: usize>(
    input: ReadBuffer<[<T as Element>::Value; N]>,
    mut output: DisjointBuffer<[<T as Element>::Value; N]>,
) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = input.as_slice()[index.get()];
    }
}
"#;

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn generic_instances_launch_in_all_authoring_forms_and_from_saved_artifacts() {
    let source = format!(
        "use metal_oxide::{{Device, LaunchConfig, Operation}};\n\
         use metal_oxide::device::*;\n{}\n\
         #[metal_oxide::kernels] mod inline {{ use metal_oxide::device::*; {KERNEL} }}\n\
         #[metal_oxide::kernels] mod arrays {{ use metal_oxide::device::*; {ARRAYS} }}\n\
         #[metal_oxide::kernels(path = \"kernels\")] mod imported {{}}\n{MAIN}",
        KERNEL.replace("#[kernel(", "#[metal_oxide::kernel(")
    );
    let project = Project::new(&source);
    project.kernel_crate(&format!("#![no_std]\nuse metal_oxide_device::*;\n{KERNEL}"));
    let output = checked(project.cargo(&["run", "--quiet"]).output().unwrap());
    assert!(String::from_utf8_lossy(&output.stdout).contains("generic launches passed"));
    checked(
        project
            .cargo(&["run", "--quiet"])
            .env("RUSTUP_TOOLCHAIN", "1.89.0")
            .env("METAL_OXIDE_ARTIFACTS", "locked")
            .output()
            .unwrap(),
    );
    project.write(&source.replace(
        "let config = LaunchConfig::<32>",
        "let config = LaunchConfig::<64>",
    ));
    let error = failure(project.cargo(&["check", "--quiet"]).output().unwrap());
    assert!(error.contains("mismatched types"), "{error}");
}

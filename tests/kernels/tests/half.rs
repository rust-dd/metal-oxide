#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use objc2_metal::{MTLDevice, MTLGPUFamily};

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn half_types_launch_from_host_modules_and_saved_kernel_crates() {
    let functions = r#"
        #[kernel(requires(input.len() == output.len()))]
        pub fn calculate(input: ReadBuffer<BF16>, mut output: DisjointBuffer<BF16>) {
            let index = thread_index();
            if let Some(output) = output.get_mut(&index) {
                let mut value = input.as_slice()[index.get()];
                value += BF16::from_f32(2.0);
                value *= BF16::from_f32(3.0);
                value -= BF16::from_f32(1.0);
                value /= BF16::from_f32(2.0);
                value %= BF16::from_f32(5.0);
                *output = value;
            }
        }
    "#;
    let source = format!(
        r#"
        use metal_oxide::{{BF16, Device, LaunchConfig, Operation}};
        #[metal_oxide::kernels]
        mod inline {{
            use metal_oxide::BF16;
            use metal_oxide::device::{{ReadBuffer, DisjointBuffer, thread_index}};
            {functions}
        }}
        #[metal_oxide::kernels(path = "kernels")] mod imported {{}}
        fn main() -> metal_oxide::Result<()> {{
            let device = Device::system_default()?;
            let input = device.buffer_from_slice(&[BF16::from_f32(2.0), BF16::from_f32(-4.0)])?;
            let mut output = device.buffer_zeroed::<BF16>(2)?;
            let config = LaunchConfig::<32>::for_elements(2)?;
            inline::calculate(&device, config, &input, &mut output).wait()?;
            assert_eq!(output.read().wait().iter().map(|v| v.to_f32()).collect::<Vec<_>>(), [0.5, -3.5]);
            imported::calculate(&device, config, &input, &mut output).wait()?;
            assert_eq!(output.read().wait().iter().map(|v| v.to_f32()).collect::<Vec<_>>(), [0.5, -3.5]);
            Ok(())
        }}
    "#
    );
    let project = support::Project::new(&source);
    project.kernel_crate(&format!(
        "#![no_std]\nuse metal_oxide_device::*;\n{functions}"
    ));
    // The Mac2-only paravirtual device of hosted macOS VMs lacks bfloat16 and must reject it.
    let device = metal_oxide::Device::system_default().unwrap();
    let bfloat = device.as_native().supportsFamily(MTLGPUFamily::Apple6);
    let mut saved = project.cargo(&["run", "--quiet"]);
    saved
        .env("RUSTUP_TOOLCHAIN", "1.89.0")
        .env("METAL_OXIDE_ARTIFACTS", "locked");
    for mut run in [project.cargo(&["run", "--quiet"]), saved] {
        let output = run.output().unwrap();
        if bfloat {
            support::checked(output);
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success()
                    && stderr.contains("UnsupportedDevice(\"artifact requires bfloat16\")"),
                "{stderr}"
            );
        }
    }
}

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use support::{Project, checked};

/// Runs `program` against the host records that the kernel macros generate: mirrored from
/// the ABI of `fixture` as a kernel crate, or `#[repr(C)]` records of a `#[kernels]` module.
fn run(kernels: &str, program: &str, fixture: Option<&str>) {
    let project = Project::new(&format!(
        "#![allow(non_camel_case_types, dead_code)]\n{kernels}\n{program}"
    ));
    if let Some(fixture) = fixture {
        let source = std::fs::read_to_string(support::root().join(fixture)).unwrap();
        project.kernel_crate(&source);
    }
    checked(project.cargo(&["run", "--quiet"]).output().unwrap());
}

fn mirrored(fixture: &str, program: &str) {
    let kernels = "#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nuse kernels::*;";
    run(kernels, program, Some(fixture));
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn mirrored_record_codecs_encode_the_gpu_layout() {
    mirrored(
        "crates/metal-oxide-compiler/tests/fixtures/structured.rs",
        r#"
use metal_oxide::GpuValue;
fn main() {
    assert_eq!((Particle::SIZE, Particle::ALIGNMENT), (24, 4));
    assert_eq!(Config::SIZE, 12);
    let value = Particle { tag: 7, flags: 0x0302, id: 0x0706_0504, position: (1.0, 2.0), velocity: [3.0, 4.0] };
    let mut bytes = [0xff; 24];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 2, 3, 4, 5, 6, 7, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]);
    assert_eq!(Particle::decode(&bytes), value);
    bytes[1] = 0xff;
    assert_eq!(Particle::decode(&bytes), value);
    assert_eq!(Particle::zeroed().position, (0.0, 0.0));
    let mut array = [0xff; 48];
    [value; 2].encode(&mut array);
    assert_eq!(<[Particle; 2]>::decode(&array), [value; 2]);
    assert_eq!(&array[0..24], &array[24..48]);
}
"#,
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn mirrored_record_names_do_not_shadow_what_launches_use() {
    mirrored(
        "crates/metal-oxide-compiler/tests/fixtures/host_names.rs",
        r#"
use metal_oxide::GpuValue;
fn main() {
    let value = usize { value: 7 };
    let mut bytes = [0; 4];
    value.encode(&mut bytes);
    assert_eq!(usize::decode(&bytes), value);
    assert_eq!(AsRef::SIZE, 4);
    assert_eq!(Ok::SIZE, 4);
    assert_eq!(Into::SIZE, 4);
}
"#,
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn mirrored_signed_record_codecs_preserve_bits_and_padding() {
    mirrored(
        "crates/metal-oxide-compiler/tests/fixtures/signed_narrow.rs",
        r#"
use metal_oxide::GpuValue;
fn main() {
    assert_eq!((SignedRecord::SIZE, SignedRecord::ALIGNMENT), (8, 2));
    let value = SignedRecord { small: -128, wide: -32768, pair: (-1, 0x1234) };
    let mut bytes = [0xff; 8];
    value.encode(&mut bytes);
    assert_eq!(bytes, [128, 0, 0, 128, 255, 0, 52, 18]);
    assert_eq!(SignedRecord::decode(&bytes), value);
    bytes[1] = 0xaa;
    bytes[5] = 0xbb;
    assert_eq!(SignedRecord::decode(&bytes), value);
}
"#,
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn mirrored_half_record_codecs_encode_the_gpu_layout() {
    mirrored(
        "crates/metal-oxide-compiler/tests/fixtures/half.rs",
        r#"
use metal_oxide::{F16, GpuValue};
fn main() {
    assert_eq!((HalfRecord::SIZE, HalfRecord::ALIGNMENT), (10, 2));
    let value = HalfRecord { tag: 7, pair: (F16::from_bits(0x3c00), F16::from_bits(0xbc00)), samples: [F16::from_bits(1), F16::INFINITY] };
    let mut bytes = [0xff; 10];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 0, 60, 0, 188, 1, 0, 0, 124]);
    bytes[1] = 0xaa;
    assert_eq!(HalfRecord::decode(&bytes), value);
    let layout = HalfRecord::layout().unwrap();
    layout.validate().unwrap();
    assert_eq!((layout.size, layout.alignment), (10, 2));
}
"#,
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn module_record_codecs_encode_the_gpu_layout() {
    let kernels = r#"
#[metal_oxide::kernels]
mod kernels {
    use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Particle {
        pub tag: u8,
        pub flags: u16,
        pub id: u32,
        pub position: (f32, f32),
        pub velocity: [f32; 2],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Config {
        pub dt: f32,
        pub acceleration: (f32, f32),
    }

    #[kernel]
    pub fn step(input: ReadBuffer<Particle>, mut out: DisjointBuffer<Particle>, config: Config) {
        let index = thread_index();
        let input = input.as_slice();
        if let Some(out) = out.get_mut(&index)
            && index.get() < input.len()
        {
            let mut value = input[index.get()];
            value.position.0 += value.velocity[0] * config.dt;
            *out = value;
        }
    }
}
use kernels::*;"#;
    run(
        kernels,
        r#"
use metal_oxide::GpuValue;
fn main() {
    assert_eq!((Particle::SIZE, Particle::ALIGNMENT), (24, 4));
    assert_eq!(Config::SIZE, 12);
    let value = Particle { tag: 7, flags: 0x0302, id: 0x0706_0504, position: (1.0, 2.0), velocity: [3.0, 4.0] };
    let mut bytes = [0xff; 24];
    value.encode(&mut bytes);
    assert_eq!(bytes, [7, 0, 2, 3, 4, 5, 6, 7, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]);
    assert_eq!(Particle::decode(&bytes), value);
    bytes[1] = 0xff;
    assert_eq!(Particle::decode(&bytes), value);
    assert_eq!(Particle::zeroed().position, (0.0, 0.0));
    let mut array = [0xff; 48];
    [value; 2].encode(&mut array);
    assert_eq!(<[Particle; 2]>::decode(&array), [value; 2]);
    assert_eq!(&array[0..24], &array[24..48]);
}
"#,
        None,
    );
}

use crate::kernels::Kernels;
use metal_oxide::{Argument, Device, Dim3, F16, LaunchConfig, Module, Pipeline};

pub(super) fn verify(device: &Device, kernels: &Kernels<'_>) -> metal_oxide::Result<()> {
    let module = Module::from_source(
        device,
        include_str!("../../../../benchmarks/reference-msl/matmul_half.metal"),
    )?;
    let reference = Pipeline::new(device, &module, "matmul_half_reference")?;
    const SENTINEL: f32 = -8192.0;
    for (rows, columns, inner) in [
        (0_u32, 7_u32, 19_u32),
        (5, 0, 19),
        (3, 7, 0),
        (1, 1, 1),
        (16, 16, 16),
        (17, 31, 19),
        (64, 48, 33),
        (9, 23, 127),
        (7, 19, 257),
        (3, 17, 513),
    ] {
        let a = (0..rows * inner)
            .map(|i| F16::from_f32((i % 97) as f32 / 37.0 - 1.25))
            .collect::<Vec<_>>();
        let b = (0..inner * columns)
            .map(|i| F16::from_f32((i % 89) as f32 / 31.0 - 1.375))
            .collect::<Vec<_>>();
        let input_a = device.buffer_from_slice(&a)?;
        let input_b = device.buffer_from_slice(&b)?;
        let count = (rows * columns) as usize;
        let mut output = device.buffer_from_slice(&vec![SENTINEL; count + 8])?;
        let mut handwritten = device.buffer_from_slice(&vec![SENTINEL; count + 8])?;
        let launch = LaunchConfig::<16, 16>::new(Dim3::xy(columns.div_ceil(16), rows.div_ceil(16)));
        // SAFETY: distinct buffers cover the matrix sizes; the 16x16 blocks guard padded cells.
        unsafe {
            kernels.matmul_half(
                launch,
                &input_a,
                &input_b,
                &mut output,
                rows,
                columns,
                inner,
            )?;
            device.launch(
                &reference,
                launch,
                &[
                    Argument::read(&input_a),
                    Argument::read(&input_b),
                    Argument::write(&mut handwritten),
                    Argument::value(rows)?,
                    Argument::value(columns)?,
                    Argument::value(inner)?,
                ],
            )?;
        }
        let actual = output.as_slice();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            handwritten
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(&actual[count..], [SENTINEL; 8]);
        assert_eq!(input_a.as_slice(), a);
        assert_eq!(input_b.as_slice(), b);
        let mut max_error = 0.0_f64;
        let operations = f64::from(inner.div_ceil(16) * 16) * 2.0;
        let rounding = operations * f64::from(f32::EPSILON) * 0.5;
        let gamma = rounding / (1.0 - rounding);
        for row in 0..rows {
            for column in 0..columns {
                let mut expected = 0.0_f32;
                let mut precise = 0.0_f64;
                let mut absolute_products = 0.0_f64;
                for k in 0..inner {
                    let x = a[(row * inner + k) as usize].to_f32();
                    let y = b[(k * columns + column) as usize].to_f32();
                    expected += x * y;
                    let product = f64::from(x) * f64::from(y);
                    precise += product;
                    absolute_products += product.abs();
                }
                let value = actual[(row * columns + column) as usize];
                assert_eq!(value.to_bits(), expected.to_bits());
                let error = (f64::from(value) - precise).abs();
                let bound = 2.0e-6 + gamma * absolute_products;
                assert!(
                    error <= bound,
                    "{rows}x{columns} inner={inner}: {error} > {bound}"
                );
                max_error = max_error.max(error);
            }
        }
        if rows == 17 {
            let mut queued = device.buffer_from_slice(&vec![SENTINEL; count + 8])?;
            // SAFETY: the same checked sizes and block shape apply; wait retains all arguments.
            unsafe {
                device.submit(|batch| {
                    kernels.enqueue_matmul_half(
                        batch,
                        launch,
                        &input_a,
                        &input_b,
                        &mut queued,
                        rows,
                        columns,
                        inner,
                    )
                })?
            }
            .wait()?;
            assert_eq!(queued.as_slice(), actual);
        }
        println!(
            "matmul-half {rows}x{inner} x {inner}x{columns}: CPU/MSL agree, max error {max_error:.3e}"
        );
    }
    Ok(())
}

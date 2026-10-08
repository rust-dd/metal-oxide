use super::*;
use metal_oxide::{Argument, Dim3, LaunchConfig};

pub(super) fn run(device: &Device, config: &Config, variant: &str) -> Result<Case> {
    let [rows, columns, inner] = config.matrix;
    let mut case = Case::new("matmul", variant, config.matrix.to_vec());
    let entry = if variant == "rust" {
        "matmul"
    } else {
        "matmul_reference"
    };
    let pipelines = case.pipelines(device, config, "matmul", &[entry])?;
    let a_values = (0..rows * inner)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect::<Vec<_>>();
    let b_values = (0..inner * columns)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
        .collect::<Vec<_>>();
    let start = Instant::now();
    let mut a = device.buffer_zeroed::<f32>(a_values.len())?;
    let mut b = device.buffer_zeroed::<f32>(b_values.len())?;
    let mut output = device.buffer_zeroed::<f32>((rows * columns) as usize)?;
    case.allocation_ns = elapsed(start);
    let launch = LaunchConfig::<16, 16>::new(Dim3::xy(columns.div_ceil(16), rows.div_ceil(16)));
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        let mut expected = vec![0.0_f32; (rows * columns) as usize];
        for row in 0..rows {
            for column in 0..columns {
                for k in 0..inner {
                    expected[(row * columns + column) as usize] += a_values
                        [(row * inner + k) as usize]
                        * b_values[(k * columns + column) as usize];
                }
            }
        }
        let cpu = elapsed(start);
        let start = Instant::now();
        a.as_mut_slice().copy_from_slice(&a_values);
        b.as_mut_slice().copy_from_slice(&b_values);
        let copy = elapsed(start);
        let start = Instant::now();
        a.upload();
        b.upload();
        output.upload();
        let upload = elapsed(start);
        // SAFETY: matrices cover the declared dimensions; complete 16x16 blocks initialize both shared tiles.
        let mut sample = unsafe {
            dispatch(device, |batch| {
                batch.launch(
                    &pipelines[0],
                    launch,
                    &[
                        Argument::read(&a),
                        Argument::read(&b),
                        Argument::write(&mut output),
                        Argument::value(rows)?,
                        Argument::value(columns)?,
                        Argument::value(inner)?,
                    ],
                )
            })?
        };
        let start = Instant::now();
        let result = output.as_slice();
        sample.readback_ns = elapsed(start);
        assert_eq!(result, expected, "matmul {variant}");
        sample.cpu_reference_ns = cpu;
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    case.finish()
}

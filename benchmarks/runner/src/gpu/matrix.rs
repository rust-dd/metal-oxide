use super::*;
use metal_oxide::{Argument, Buffer, Dim3, LaunchConfig};

pub(super) struct Matmul {
    case: Case,
    pipeline: Pipeline,
    shape: [u32; 3],
    a_values: Vec<f32>,
    b_values: Vec<f32>,
    launch: LaunchConfig<16, 16>,
    buffers: (Buffer<f32>, Buffer<f32>, Buffer<f32>),
}

impl Matmul {
    pub(super) fn new(device: &Device, config: &Config, variant: &str) -> Result<Self> {
        let [rows, columns, inner] = config.matrix;
        let mut case = Case::new("matmul", variant, config.matrix.to_vec());
        let entry = if variant == "rust" {
            "matmul"
        } else {
            "matmul_reference"
        };
        let pipeline = case
            .pipelines(device, config, "matmul", &[entry])?
            .remove(0);
        let a_values = (0..rows * inner)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let b_values = (0..inner * columns)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect::<Vec<_>>();
        let start = Instant::now();
        let a = device.buffer_zeroed::<f32>(a_values.len())?;
        let b = device.buffer_zeroed::<f32>(b_values.len())?;
        let output = device.buffer_zeroed::<f32>((rows * columns) as usize)?;
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipeline,
            shape: config.matrix,
            a_values,
            b_values,
            launch: LaunchConfig::<16, 16>::new(Dim3::xy(columns.div_ceil(16), rows.div_ceil(16))),
            buffers: (a, b, output),
        })
    }
}

impl Candidate for Matmul {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let [rows, columns, inner] = self.shape;
        let (a, b, output) = &mut self.buffers;
        let start = Instant::now();
        let mut expected = vec![0.0_f32; (rows * columns) as usize];
        for row in 0..rows {
            for column in 0..columns {
                for k in 0..inner {
                    expected[(row * columns + column) as usize] += self.a_values
                        [(row * inner + k) as usize]
                        * self.b_values[(k * columns + column) as usize];
                }
            }
        }
        let cpu = elapsed(start);
        let start = Instant::now();
        a.write().wait().copy_from_slice(&self.a_values);
        b.write().wait().copy_from_slice(&self.b_values);
        let copy = elapsed(start);
        let start = Instant::now();
        a.upload();
        b.upload();
        output.upload();
        let upload = elapsed(start);
        let (pipeline, launch) = (&self.pipeline, self.launch);
        let mut sample = dispatch(|batch| {
            // SAFETY: matrices cover the declared dimensions; complete 16x16 blocks initialize
            // both shared tiles.
            unsafe {
                batch
                    .push(Launch::new(
                        Ok(pipeline),
                        launch,
                        (&*a, &*b, &mut *output, rows, columns, inner),
                        |(a, b, output, rows, columns, inner)| {
                            Ok([
                                Argument::read(a),
                                Argument::read(b),
                                Argument::write(output),
                                Argument::value(*rows)?,
                                Argument::value(*columns)?,
                                Argument::value(*inner)?,
                            ])
                        },
                    ))
                    .map(drop)
            }
        })?;
        let start = Instant::now();
        let result = output.read().wait();
        sample.readback_ns = elapsed(start);
        let case = &mut self.case;
        assert_eq!(result, expected, "{} {}", case.name, case.variant);
        sample.cpu_reference_ns = Some(cpu);
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if keep {
            case.samples.push(sample);
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<Case> {
        self.case.finish()
    }
}

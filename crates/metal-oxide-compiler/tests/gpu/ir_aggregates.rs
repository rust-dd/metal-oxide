use super::{Argument, Device, Dim3, LaunchConfig, Module, Pipeline};
use metal_oxide_ir::{
    self as ir, Access, AddressSpace, Aggregate, Block, Constant, Expression, Function, Operand,
    Place, Scalar, SourceLocation, Statement, Terminator, Type, TypeTable,
};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn ir_owned_aggregate_values_and_copies_execute_on_metal() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let ir = aggregate_module();
    let msl = metal_oxide_codegen::Codegen::new(&ir)
        .unwrap()
        .emit()
        .unwrap();
    let module = Module::from_source(&device, &msl)?;
    let pipeline = Pipeline::new(&device, &module, "aggregate_values")?;
    for seed in [0, 17, u32::MAX] {
        let mut integers = device.buffer_zeroed::<u32>(5)?;
        let mut floats = device.buffer_zeroed::<f32>(1)?;
        // SAFETY: one thread writes five u32 values and one f32 in distinct initialized buffers.
        unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::write(&mut integers),
                    Argument::write(&mut floats),
                    Argument::value::<u32>(seed)?,
                ],
            )?;
        }
        assert_eq!(integers.as_slice(), [20, seed, seed, 30, seed]);
        assert_eq!(floats.as_slice(), [2.5]);
    }
    Ok(())
}

fn source() -> SourceLocation {
    SourceLocation {
        file: "aggregate-ir.rs".into(),
        line: 1,
        column: 1,
    }
}

fn block(statements: Vec<(usize, Expression)>) -> Block {
    Block {
        statements: statements
            .into_iter()
            .map(|(destination, value)| Statement {
                destination: Place::local(destination),
                value,
                source: source(),
            })
            .collect(),
        terminator: Terminator::Return,
        source: source(),
    }
}

fn field(local: usize, field: u32) -> Operand {
    Operand::Place(Place {
        local,
        projection: vec![ir::Projection::Field(field)],
    })
}

fn uint(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

fn construct(ty: Type, fields: Vec<Operand>) -> Expression {
    let Type::Aggregate(ty) = ty else {
        panic!("expected aggregate type")
    };
    Expression::Aggregate { ty, fields }
}

fn update(local: usize, field: u32, value: usize) -> Expression {
    Expression::AggregateUpdate {
        aggregate: Operand::local(local),
        field,
        value: Operand::local(value),
    }
}

fn store(buffer: usize, index: u32, value: Operand) -> Expression {
    Expression::BufferStore {
        buffer: Operand::local(buffer),
        index: uint(index),
        value,
    }
}

fn aggregate_module() -> ir::Module {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Scalar(Scalar::U32),
        length: 3,
    });
    let tuple = types.intern(Aggregate::Tuple(vec![Type::Scalar(Scalar::U32), array]));
    let record = types.intern(Aggregate::record(
        "Record",
        vec![tuple, Type::Scalar(Scalar::F32)],
    ));
    let buffer = |element| Type::Buffer {
        element: ir::Element::Scalar(element),
        access: Access::Write,
        address_space: AddressSpace::Device,
    };
    let kernel = Function {
        name: "aggregate_values".into(),
        kernel: true,
        required_block: None,
        parameters: 3,
        locals: vec![
            Type::Unit,
            buffer(Scalar::U32),
            buffer(Scalar::F32),
            Type::Scalar(Scalar::U32),
            array,
            tuple,
            record,
            record,
            tuple,
            array,
            tuple,
            array,
            Type::Unit,
        ],
        blocks: vec![block(vec![
            (
                4,
                construct(array, vec![Operand::local(3), uint(20), uint(30)]),
            ),
            (
                5,
                construct(tuple, vec![Operand::local(3), Operand::local(4)]),
            ),
            (
                6,
                construct(
                    record,
                    vec![
                        Operand::local(5),
                        Operand::Constant(Constant::F32(2.5_f32.to_bits())),
                    ],
                ),
            ),
            (
                7,
                Expression::Call {
                    function: 1,
                    arguments: vec![Operand::local(6), Operand::local(3)],
                },
            ),
            (8, Expression::Use(field(6, 0))),
            (9, Expression::Use(field(8, 1))),
            (10, Expression::Use(field(7, 0))),
            (11, Expression::Use(field(10, 1))),
            (12, store(1, 0, field(9, 1))),
            (12, store(1, 1, field(11, 1))),
            (12, store(1, 2, field(11, 0))),
            (12, store(1, 3, field(11, 2))),
            (12, store(1, 4, field(10, 0))),
            (12, store(2, 0, field(7, 1))),
        ])],
        source: source(),
    };
    let helper = Function {
        name: "update_nested".into(),
        kernel: false,
        required_block: None,
        parameters: 2,
        locals: vec![
            record,
            record,
            Type::Scalar(Scalar::U32),
            tuple,
            array,
            array,
            tuple,
        ],
        blocks: vec![block(vec![
            (3, Expression::Use(field(1, 0))),
            (4, Expression::Use(field(3, 1))),
            (5, update(4, 1, 2)),
            (6, update(3, 1, 5)),
            (0, update(1, 0, 6)),
        ])],
        source: source(),
    };
    ir::Module {
        types,
        functions: vec![kernel, helper],
    }
}

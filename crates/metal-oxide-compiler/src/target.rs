use rustc_middle::ty::{TyCtxt, TypingEnv};

pub(crate) fn validate(tcx: TyCtxt<'_>) {
    let target = &tcx.sess.target;
    if target.arch.to_string() != "metal"
        || target.env.to_string() != "metal"
        || target.os.to_string() != "none"
    {
        tcx.dcx()
            .fatal("expected the metal64-unknown-none device target");
    }
    if target.pointer_width != 64 || target.endian != rustc_abi::Endian::Little {
        tcx.dcx()
            .fatal("Metal device pointers and usize must be 64-bit and little-endian");
    }
    let environment = TypingEnv::fully_monomorphized();
    for (name, ty, bytes) in [
        ("f32", tcx.types.f32, 4),
        ("u32", tcx.types.u32, 4),
        ("i32", tcx.types.i32, 4),
        ("usize", tcx.types.usize, 8),
    ] {
        let layout = tcx.layout_of(environment.as_query_input(ty)).unwrap();
        if layout.size.bytes() != bytes || layout.align.abi.bytes() != bytes {
            tcx.dcx().fatal(format!(
                "unexpected {name} layout for the Metal device target"
            ));
        }
        println!(
            "{name}: size={} align={}",
            layout.size.bytes(),
            layout.align.abi.bytes()
        );
    }
    println!("target: pointer=64 usize=64 endian=little");
}

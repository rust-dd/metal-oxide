use super::super::types::{I32, Type};
use super::{Local, Scope};

fn parameters() -> Scope {
    let array = Type::Array(Box::new(I32), 2);
    let parameters = [
        ("p0", Type::Slice(Box::new(array.clone()), true)),
        ("p1", Type::Ref(Box::new(array), true)),
        ("p2", Type::Slice(Box::new(I32), false)),
    ];
    let locals = parameters.into_iter().map(|(name, ty)| Local {
        name: name.into(),
        ty,
        mutable: false,
    });
    Scope {
        locals: locals.collect(),
        ..Scope::default()
    }
}

/// The locals whose places assignments and `&mut` borrows can target.
fn written(scope: &Scope) -> Vec<String> {
    let targets = scope.targets(&[]).into_iter();
    let mut names = targets.map(|(name, ..)| name).collect::<Vec<_>>();
    names.dedup();
    names
}

fn captured(scope: &Scope) -> Vec<String> {
    let locals = scope.closure(&[]).locals.into_iter();
    locals.map(|local| local.name).collect()
}

#[test]
fn loop_bodies_neither_write_nor_capture_the_mut_references_they_iterate() {
    let scope = parameters();
    assert_eq!(written(&scope), ["p0", "p1"]);
    for (iterated, other) in [("p0", "p1"), ("p1", "p0")] {
        let body = scope.freeze(iterated);
        assert_eq!(written(&body), [other], "{iterated}");
        assert_eq!(captured(&body), ["p2"], "{iterated}");
    }
    assert_eq!(captured(&scope.freeze("p2")), ["p2"]);
}

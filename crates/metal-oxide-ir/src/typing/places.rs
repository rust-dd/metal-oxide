use crate::*;

/// What a place denotes: a value, or the elements behind a dereferenced slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Located {
    Value(Type),
    Elements(Pointer),
}

#[derive(Clone, Copy, Debug)]
pub struct PlaceInfo {
    pub located: Located,
    pub space: AddressSpace,
    /// Writes and mutable references need every pointer on the path to be mutable.
    pub writable: bool,
}

pub fn place_info(
    module: &Module,
    function: &Function,
    place: &Place,
    source: &SourceLocation,
) -> Result<PlaceInfo, Error> {
    let ty = *function
        .locals
        .get(place.local)
        .ok_or_else(|| Error::new(source, "invalid local reference"))?;
    if ty == Type::Never {
        return Err(Error::new(source, "never-typed values cannot be read"));
    }
    let mut info = PlaceInfo {
        located: Located::Value(ty),
        space: AddressSpace::Thread,
        writable: true,
    };
    for projection in &place.projection {
        let invalid = match projection {
            Projection::Field(_) => "invalid aggregate field projection",
            Projection::Index(_) => "invalid array or slice index projection",
            Projection::Deref => "dereference requires a pointer",
        };
        info.located = match (*projection, info.located) {
            (Projection::Field(field), Located::Value(ty)) => {
                module.types.field(ty, field).map(Located::Value)
            }
            (Projection::Index(index), located) => {
                if function.locals.get(index) != Some(&Type::Scalar(Scalar::Usize)) {
                    return Err(Error::new(source, "array indices require a usize local"));
                }
                match located {
                    Located::Value(Type::Aggregate(id)) => match module.types.get(id) {
                        Some(Aggregate::Array { element, .. }) => Some(Located::Value(*element)),
                        _ => None,
                    },
                    Located::Elements(slice) => Some(Located::Value(slice.target)),
                    Located::Value(_) => None,
                }
            }
            (Projection::Deref, Located::Value(ty)) => module.types.pointer(ty).map(|pointer| {
                info.space = pointer.space;
                info.writable &= pointer.mutable;
                if pointer.slice {
                    Located::Elements(pointer)
                } else {
                    Located::Value(pointer.target)
                }
            }),
            _ => None,
        }
        .ok_or_else(|| Error::new(source, invalid))?;
    }
    Ok(info)
}

pub fn place_type(
    module: &Module,
    function: &Function,
    place: &Place,
    source: &SourceLocation,
) -> Result<Type, Error> {
    match place_info(module, function, place, source)?.located {
        Located::Value(ty) => Ok(ty),
        Located::Elements(_) => Err(Error::new(source, "slice elements are not one value")),
    }
}

//! Messages from kernel threads to the host through Metal shader logging (macOS 15).

/// Writes one message per calling thread; each `{}` takes the next argument: an integer,
/// `f32` or [`F16`](crate::F16). Floats print round-trip digits.
#[macro_export]
macro_rules! log {
    ($format:literal $(, $value:expr)* $(,)?) => {
        $crate::log::write($format, ($($value,)*))
    };
}

/// The compiler lowers this call to Metal's `os_log`; kernels use [`log!`](crate::log!).
#[doc(hidden)]
pub fn write<A: Arguments>(format: &'static str, arguments: A) {
    // SAFETY: the bridge reads both values only during the call.
    unsafe {
        crate::intrinsics::__metal_log(
            format.as_ptr().cast(),
            format.len(),
            (&raw const arguments).cast(),
            core::mem::size_of::<A>(),
        )
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A value a message can print.
pub trait Value: Copy + sealed::Sealed {}

/// The arguments of one message.
pub trait Arguments: Copy + sealed::Sealed {}

macro_rules! values {
    ($($ty:ty),*) => {
        $(
            impl sealed::Sealed for $ty {}
            impl Value for $ty {}
        )*
    };
}

values!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, crate::F16);

macro_rules! arguments {
    ($($name:ident),*) => {
        impl<$($name: Value),*> sealed::Sealed for ($($name,)*) {}
        impl<$($name: Value),*> Arguments for ($($name,)*) {}
    };
}

arguments!();
arguments!(A);
arguments!(A, B);
arguments!(A, B, C);
arguments!(A, B, C, D);
arguments!(A, B, C, D, E);
arguments!(A, B, C, D, E, F);
arguments!(A, B, C, D, E, F, G);
arguments!(A, B, C, D, E, F, G, H);

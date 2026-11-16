//! Unified Rust reimplementation of NumPy (`numpy-rs`).
pub use numpy_types::*;

#[cfg(feature = "core")]
pub use numpy_core as core;
#[cfg(feature = "extra")]
pub use numpy_extra as extra;
#[cfg(feature = "fft")]
pub use numpy_fft as fft;
#[cfg(feature = "lib")]
pub use numpy_lib as lib;
#[cfg(feature = "linalg")]
pub use numpy_linalg as linalg;
#[cfg(feature = "random")]
pub use numpy_random as random;
pub use numpy_types as types;
#[cfg(feature = "ufunc")]
pub use numpy_ufunc as ufunc;

/// Common array types and construction helpers.
pub mod prelude {
    #[cfg(feature = "core")]
    pub use numpy_core::{arange, full, linspace, ones, zeros};
    pub use numpy_types::{
        Complex32, Complex64, DType, DynArray, NdArray, NumPyError, Order, Result, Scalar, Slice,
        SliceArg, f16, slice,
    };
}

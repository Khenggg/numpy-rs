//! Foundational types, 64-byte SIMD-aligned buffers, strided zero-copy `NdArray<T>`,
//! borrowed views (`ArrayView`, `ArrayViewMut`), NumPy 2.x `DType` promotion/casting,
//! and dynamic `DynArray` wrapper.

pub mod array;
pub mod buffer;
pub mod dtype;
pub mod dyn_array;
pub mod error;
pub mod shape;

pub use array::{ArrayView, ArrayViewMut, MemoryLayout, NdArray, NdIter};
pub use buffer::{AlignedBuffer, AlignedUninit, SIMD_ALIGN};
pub use dtype::{
    can_cast, f16, promote_types, result_type, Casting, Complex, Complex32, Complex64, DType,
    DTypeKind, FInfo, FloatElement, IInfo, Scalar,
};
pub use dyn_array::DynArray;
pub use error::{NumPyError, Result};
pub use shape::{
    broadcast_many_shapes, broadcast_shapes, broadcast_strides, check_non_overlapping,
    compute_c_strides, compute_f_strides, compute_strides, is_c_contiguous, is_f_contiguous,
    normalize_axes, normalize_axis, normalize_index, resolve_reshape_dims, shape_size, Order,
    Slice, SliceArg,
};

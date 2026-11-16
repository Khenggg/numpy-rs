use numpy_types::{
    AlignedBuffer, ArrayView, ArrayViewMut, Complex32, Complex64, DType, DynArray, NumPyError,
    Order, Result, Scalar, f16, shape_size,
};
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::ptr;

/// Minimum element count before dispatching Rayon parallel workers.
pub const RAYON_THRESHOLD: usize = 32_768;

/// Dynamic scalar value supporting all 14 NumPy core dtypes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynScalar {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(f16),
    F32(f32),
    F64(f64),
    C64(Complex32),
    C128(Complex64),
}

impl DynScalar {
    #[inline]
    pub const fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::C64(_) => DType::C64,
            Self::C128(_) => DType::C128,
        }
    }

    #[inline]
    pub fn to_f64_lossy(self) -> f64 {
        match self {
            Self::Bool(v) => v.to_f64_lossy(),
            Self::I8(v) => v as f64,
            Self::I16(v) => v as f64,
            Self::I32(v) => v as f64,
            Self::I64(v) => v as f64,
            Self::U8(v) => v as f64,
            Self::U16(v) => v as f64,
            Self::U32(v) => v as f64,
            Self::U64(v) => v as f64,
            Self::F16(v) => v.to_f64(),
            Self::F32(v) => v as f64,
            Self::F64(v) => v,
            Self::C64(v) => v.re as f64,
            Self::C128(v) => v.re,
        }
    }

    #[inline]
    pub fn to_complex64(self) -> Complex64 {
        match self {
            Self::C64(v) => Complex64::new(v.re as f64, v.im as f64),
            Self::C128(v) => v,
            other => Complex64::new(other.to_f64_lossy(), 0.0),
        }
    }

    #[inline]
    pub fn to_bool(self) -> bool {
        match self {
            Self::Bool(v) => v,
            Self::I8(v) => v != 0,
            Self::I16(v) => v != 0,
            Self::I32(v) => v != 0,
            Self::I64(v) => v != 0,
            Self::U8(v) => v != 0,
            Self::U16(v) => v != 0,
            Self::U32(v) => v != 0,
            Self::U64(v) => v != 0,
            Self::F16(v) => v.is_truthy(),
            Self::F32(v) => v != 0.0,
            Self::F64(v) => v != 0.0,
            Self::C64(v) => v.re != 0.0 || v.im != 0.0,
            Self::C128(v) => v.re != 0.0 || v.im != 0.0,
        }
    }

    #[inline]
    pub fn to_i64_lossy(self) -> i64 {
        match self {
            Self::Bool(v) => i64::from(v),
            Self::I8(v) => v as i64,
            Self::I16(v) => v as i64,
            Self::I32(v) => v as i64,
            Self::I64(v) => v,
            Self::U8(v) => v as i64,
            Self::U16(v) => v as i64,
            Self::U32(v) => v as i64,
            Self::U64(v) => v as i64,
            Self::F16(v) => v.to_f64() as i64,
            Self::F32(v) => v as i64,
            Self::F64(v) => v as i64,
            Self::C64(v) => v.re as i64,
            Self::C128(v) => v.re as i64,
        }
    }

    #[inline]
    pub fn to_u64_lossy(self) -> u64 {
        match self {
            Self::Bool(v) => u64::from(v),
            Self::I8(v) => v as u64,
            Self::I16(v) => v as u64,
            Self::I32(v) => v as u64,
            Self::I64(v) => v as u64,
            Self::U8(v) => v as u64,
            Self::U16(v) => v as u64,
            Self::U32(v) => v as u64,
            Self::U64(v) => v,
            Self::F16(v) => v.to_f64() as u64,
            Self::F32(v) => v as u64,
            Self::F64(v) => v as u64,
            Self::C64(v) => v.re as u64,
            Self::C128(v) => v.re as u64,
        }
    }

    #[inline]
    pub fn cast_to<T: Scalar>(self) -> T {
        match T::DTYPE {
            DType::Bool => unsafe { mem_transmute_copy::<bool, T>(self.to_bool()) },
            DType::I8 => unsafe { mem_transmute_copy::<i8, T>(self.to_i64_lossy() as i8) },
            DType::I16 => unsafe { mem_transmute_copy::<i16, T>(self.to_i64_lossy() as i16) },
            DType::I32 => unsafe { mem_transmute_copy::<i32, T>(self.to_i64_lossy() as i32) },
            DType::I64 => unsafe { mem_transmute_copy::<i64, T>(self.to_i64_lossy()) },
            DType::U8 => unsafe { mem_transmute_copy::<u8, T>(self.to_u64_lossy() as u8) },
            DType::U16 => unsafe { mem_transmute_copy::<u16, T>(self.to_u64_lossy() as u16) },
            DType::U32 => unsafe { mem_transmute_copy::<u32, T>(self.to_u64_lossy() as u32) },
            DType::U64 => unsafe { mem_transmute_copy::<u64, T>(self.to_u64_lossy()) },
            DType::F16 => T::from_f64_lossy(self.to_f64_lossy()),
            DType::F32 => T::from_f64_lossy(self.to_f64_lossy()),
            DType::F64 => T::from_f64_lossy(self.to_f64_lossy()),
            DType::C64 | DType::C128 => T::from_complex64_lossy(self.to_complex64()),
        }
    }
}

#[inline]
unsafe fn mem_transmute_copy<A: Copy, B: Copy>(a: A) -> B {
    debug_assert_eq!(std::mem::size_of::<A>(), std::mem::size_of::<B>());
    unsafe { ptr::read(&a as *const A as *const B) }
}

macro_rules! impl_from_dyn_scalar {
    ($t:ty, $var:ident) => {
        impl From<$t> for DynScalar {
            #[inline]
            fn from(v: $t) -> Self {
                Self::$var(v)
            }
        }
    };
}
impl_from_dyn_scalar!(bool, Bool);
impl_from_dyn_scalar!(i8, I8);
impl_from_dyn_scalar!(i16, I16);
impl_from_dyn_scalar!(i32, I32);
impl_from_dyn_scalar!(i64, I64);
impl_from_dyn_scalar!(u8, U8);
impl_from_dyn_scalar!(u16, U16);
impl_from_dyn_scalar!(u32, U32);
impl_from_dyn_scalar!(u64, U64);
impl_from_dyn_scalar!(f16, F16);
impl_from_dyn_scalar!(f32, F32);
impl_from_dyn_scalar!(f64, F64);
impl_from_dyn_scalar!(Complex32, C64);
impl_from_dyn_scalar!(Complex64, C128);

/// Copy a view in logical C-order into an uninitialized slice, using Rayon for large arrays.
pub(crate) fn copy_view_to_uninit<T: Copy + Send + Sync>(
    src: &ArrayView<'_, T>,
    dst: &mut [MaybeUninit<T>],
) {
    let len = src.len();
    debug_assert_eq!(len, dst.len());
    if len == 0 {
        return;
    }
    if let Some(sl) = src.as_slice() {
        if len >= RAYON_THRESHOLD {
            const CHUNK: usize = 4096;
            dst.par_chunks_mut(CHUNK)
                .zip(sl.par_chunks(CHUNK))
                .for_each(|(d, s)| unsafe {
                    ptr::copy_nonoverlapping(s.as_ptr(), d.as_mut_ptr() as *mut T, s.len());
                });
        } else {
            unsafe {
                ptr::copy_nonoverlapping(sl.as_ptr(), dst.as_mut_ptr() as *mut T, len);
            }
        }
        return;
    }

    let shape = src.shape();
    let strides = src.strides();
    let ndim = shape.len();
    let base_addr = src.as_ptr() as usize;

    if len >= RAYON_THRESHOLD && ndim > 0 {
        const CHUNK: usize = 2048;
        dst.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_linear = chunk_idx * CHUNK;
                let mut coords = vec![0usize; ndim];
                linear_to_coords(start_linear, shape, &mut coords);
                let mut cur_ptr = (base_addr as *const T).wrapping_offset(coords_offset(&coords, strides));
                let out_len = out_chunk.len();
                for (i, slot) in out_chunk.iter_mut().enumerate() {
                    slot.write(unsafe { *cur_ptr });
                    if i + 1 < out_len {
                        advance_coords_ptr(&mut coords, &mut cur_ptr, shape, strides);
                    }
                }
            });
    } else {
        for (slot, val) in dst.iter_mut().zip(src.iter()) {
            slot.write(val);
        }
    }
}

/// Write a slice of values in logical C-order into a mutable strided view.
pub(crate) fn copy_slice_to_view_mut<T: Copy + Send + Sync>(
    src: &[T],
    dst: &mut ArrayViewMut<'_, T>,
) {
    let len = dst.len();
    debug_assert_eq!(len, src.len());
    if len == 0 {
        return;
    }
    if let Some(sl) = dst.as_slice_mut() {
        if len >= RAYON_THRESHOLD {
            const CHUNK: usize = 4096;
            sl.par_chunks_mut(CHUNK)
                .zip(src.par_chunks(CHUNK))
                .for_each(|(d, s)| d.copy_from_slice(s));
        } else {
            sl.copy_from_slice(src);
        }
        return;
    }
    let _ = dst.copy_from_slice(src);
}

/// Fallibly fill a 64-byte aligned buffer with `value`, using Rayon parallel chunks when large.
pub(crate) fn try_aligned_filled_fast<T: Copy + Send + Sync>(
    len: usize,
    value: T,
) -> Result<AlignedBuffer<T>> {
    if len < RAYON_THRESHOLD {
        return AlignedBuffer::try_filled(len, value);
    }
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(len)?;
    uninit
        .as_mut_slice()
        .par_chunks_mut(4096)
        .for_each(|chunk| {
            for slot in chunk {
                slot.write(value);
            }
        });
    Ok(unsafe { uninit.assume_init() })
}

/// Convert a C-order flat linear index into coordinates for `shape`.
#[inline]
pub(crate) fn linear_to_coords(mut linear: usize, shape: &[usize], coords: &mut [usize]) {
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        if dim > 0 {
            coords[d] = linear % dim;
            linear /= dim;
        } else {
            coords[d] = 0;
        }
    }
}

/// Compute element offset from `coords` and `strides`.
#[inline]
pub(crate) fn coords_offset(coords: &[usize], strides: &[isize]) -> isize {
    let mut off = 0isize;
    for (&c, &st) in coords.iter().zip(strides.iter()) {
        off += (c as isize) * st;
    }
    off
}

/// Advance C-order `coords` and strided pointer `ptr` by 1 logical element.
#[inline]
pub(crate) fn advance_coords_ptr<T>(
    coords: &mut [usize],
    ptr: &mut *const T,
    shape: &[usize],
    strides: &[isize],
) {
    for d in (0..shape.len()).rev() {
        coords[d] += 1;
        if coords[d] < shape[d] {
            *ptr = unsafe { ptr.offset(strides[d]) };
            return;
        }
        let back = (coords[d] - 1) as isize * strides[d];
        *ptr = unsafe { ptr.offset(-back) };
        coords[d] = 0;
    }
}

/// Convert any integer `DynArray` into `NdArray<i64>` index array (rejecting floats/complex/bools).
pub(crate) fn dyn_to_i64_indices(arr: &DynArray) -> Result<numpy_types::NdArray<i64>> {
    match arr {
        DynArray::I8(a) => Ok(a.map(|x| x as i64)),
        DynArray::I16(a) => Ok(a.map(|x| x as i64)),
        DynArray::I32(a) => Ok(a.map(|x| x as i64)),
        DynArray::I64(a) => Ok(a.clone()),
        DynArray::U8(a) => Ok(a.map(|x| x as i64)),
        DynArray::U16(a) => Ok(a.map(|x| x as i64)),
        DynArray::U32(a) => Ok(a.map(|x| x as i64)),
        DynArray::U64(a) => {
            let mut out = Vec::with_capacity(a.len());
            for v in a.iter() {
                let iv = i64::try_from(v).map_err(|_| {
                    NumPyError::IndexOutOfBounds(format!("index {v} overflows i64"))
                })?;
                out.push(iv);
            }
            numpy_types::NdArray::from_shape_vec(a.shape(), out)
        }
        other => Err(NumPyError::DTypeError(format!(
            "arrays used as indices must be of integer type, got '{}'",
            other.dtype()
        ))),
    }
}

/// Convert a boolean `DynArray` into `&NdArray<bool>`, rejecting non-bool dtypes.
pub(crate) fn dyn_as_bool(arr: &DynArray) -> Result<&numpy_types::NdArray<bool>> {
    match arr {
        DynArray::Bool(b) => Ok(b),
        other => Err(NumPyError::DTypeError(format!(
            "boolean mask array must have dtype 'bool', got '{}'",
            other.dtype()
        ))),
    }
}

/// Helper to build a C-order `NdArray<T>` from an initialized `AlignedBuffer<T>`.
#[inline]
pub(crate) fn finish_c_array<T>(buf: AlignedBuffer<T>, shape: &[usize]) -> Result<numpy_types::NdArray<T>> {
    let _ = shape_size(shape)?;
    numpy_types::NdArray::from_aligned_buffer(buf, shape, Order::C)
}

/// Helper for resolving a scalar `Order` (`C` or `F`, treating `A`/`K` as `C` for new arrays).
#[inline]
pub(crate) fn resolve_new_order(order: Order) -> Order {
    match order {
        Order::F => Order::F,
        Order::C | Order::A | Order::K => Order::C,
    }
}

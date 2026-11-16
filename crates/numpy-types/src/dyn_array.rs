use crate::array::NdArray;
use crate::dtype::{can_cast, promote_types, Casting, Complex32, Complex64, DType, Scalar, f16};
use crate::error::{NumPyError, Result};
use crate::shape::{broadcast_shapes, Order, SliceArg};

/// Dynamically-typed NumPy array supporting runtime `DType` dispatch, casting,
/// and automatic NumPy 2.x type promotion while wrapping zero-copy `NdArray<T>`.
#[derive(Clone, Debug, PartialEq)]
pub enum DynArray {
    Bool(NdArray<bool>),
    I8(NdArray<i8>),
    I16(NdArray<i16>),
    I32(NdArray<i32>),
    I64(NdArray<i64>),
    U8(NdArray<u8>),
    U16(NdArray<u16>),
    U32(NdArray<u32>),
    U64(NdArray<u64>),
    F16(NdArray<f16>),
    F32(NdArray<f32>),
    F64(NdArray<f64>),
    C64(NdArray<Complex32>),
    C128(NdArray<Complex64>),
}

/// Dispatch macro over any `DynArray` value: `dispatch_dyn!(arr, inner => expr)`.
#[macro_export]
macro_rules! dispatch_dyn {
    ($val:expr, $ident:ident => $body:expr) => {
        match $val {
            $crate::DynArray::Bool($ident) => $body,
            $crate::DynArray::I8($ident) => $body,
            $crate::DynArray::I16($ident) => $body,
            $crate::DynArray::I32($ident) => $body,
            $crate::DynArray::I64($ident) => $body,
            $crate::DynArray::U8($ident) => $body,
            $crate::DynArray::U16($ident) => $body,
            $crate::DynArray::U32($ident) => $body,
            $crate::DynArray::U64($ident) => $body,
            $crate::DynArray::F16($ident) => $body,
            $crate::DynArray::F32($ident) => $body,
            $crate::DynArray::F64($ident) => $body,
            $crate::DynArray::C64($ident) => $body,
            $crate::DynArray::C128($ident) => $body,
        }
    };
}

impl DynArray {
    #[inline]
    pub fn dtype(&self) -> DType {
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
    pub fn shape(&self) -> &[usize] {
        dispatch_dyn!(self, a => a.shape())
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        dispatch_dyn!(self, a => a.strides())
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        dispatch_dyn!(self, a => a.ndim())
    }

    #[inline]
    pub fn len(&self) -> usize {
        dispatch_dyn!(self, a => a.len())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        dispatch_dyn!(self, a => a.is_empty())
    }

    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        dispatch_dyn!(self, a => a.is_c_contiguous())
    }

    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        dispatch_dyn!(self, a => a.is_f_contiguous())
    }

    #[inline]
    pub fn itemsize(&self) -> usize {
        self.dtype().itemsize()
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.len().saturating_mul(self.itemsize())
    }

    /// Create a zero-filled `DynArray` of `dtype`.
    pub fn zeros(shape: &[usize], dtype: DType, order: Order) -> Result<Self> {
        Ok(match dtype {
            DType::Bool => Self::Bool(NdArray::zeros_with_order(shape, order)?),
            DType::I8 => Self::I8(NdArray::zeros_with_order(shape, order)?),
            DType::I16 => Self::I16(NdArray::zeros_with_order(shape, order)?),
            DType::I32 => Self::I32(NdArray::zeros_with_order(shape, order)?),
            DType::I64 => Self::I64(NdArray::zeros_with_order(shape, order)?),
            DType::U8 => Self::U8(NdArray::zeros_with_order(shape, order)?),
            DType::U16 => Self::U16(NdArray::zeros_with_order(shape, order)?),
            DType::U32 => Self::U32(NdArray::zeros_with_order(shape, order)?),
            DType::U64 => Self::U64(NdArray::zeros_with_order(shape, order)?),
            DType::F16 => Self::F16(NdArray::zeros_with_order(shape, order)?),
            DType::F32 => Self::F32(NdArray::zeros_with_order(shape, order)?),
            DType::F64 => Self::F64(NdArray::zeros_with_order(shape, order)?),
            DType::C64 => Self::C64(NdArray::zeros_with_order(shape, order)?),
            DType::C128 => Self::C128(NdArray::zeros_with_order(shape, order)?),
        })
    }

    /// Create a one-filled `DynArray` of `dtype`.
    pub fn ones(shape: &[usize], dtype: DType, order: Order) -> Result<Self> {
        Ok(match dtype {
            DType::Bool => Self::Bool(NdArray::full_with_order(shape, true, order)?),
            DType::I8 => Self::I8(NdArray::full_with_order(shape, 1, order)?),
            DType::I16 => Self::I16(NdArray::full_with_order(shape, 1, order)?),
            DType::I32 => Self::I32(NdArray::full_with_order(shape, 1, order)?),
            DType::I64 => Self::I64(NdArray::full_with_order(shape, 1, order)?),
            DType::U8 => Self::U8(NdArray::full_with_order(shape, 1, order)?),
            DType::U16 => Self::U16(NdArray::full_with_order(shape, 1, order)?),
            DType::U32 => Self::U32(NdArray::full_with_order(shape, 1, order)?),
            DType::U64 => Self::U64(NdArray::full_with_order(shape, 1, order)?),
            DType::F16 => Self::F16(NdArray::full_with_order(shape, f16::ONE, order)?),
            DType::F32 => Self::F32(NdArray::full_with_order(shape, 1.0, order)?),
            DType::F64 => Self::F64(NdArray::full_with_order(shape, 1.0, order)?),
            DType::C64 => Self::C64(NdArray::full_with_order(shape, Complex32::new(1.0, 0.0), order)?),
            DType::C128 => Self::C128(NdArray::full_with_order(shape, Complex64::new(1.0, 0.0), order)?),
        })
    }

    pub fn slice(&self, args: &[SliceArg]) -> Result<Self> {
        Ok(match self {
            Self::Bool(a) => Self::Bool(a.slice(args)?),
            Self::I8(a) => Self::I8(a.slice(args)?),
            Self::I16(a) => Self::I16(a.slice(args)?),
            Self::I32(a) => Self::I32(a.slice(args)?),
            Self::I64(a) => Self::I64(a.slice(args)?),
            Self::U8(a) => Self::U8(a.slice(args)?),
            Self::U16(a) => Self::U16(a.slice(args)?),
            Self::U32(a) => Self::U32(a.slice(args)?),
            Self::U64(a) => Self::U64(a.slice(args)?),
            Self::F16(a) => Self::F16(a.slice(args)?),
            Self::F32(a) => Self::F32(a.slice(args)?),
            Self::F64(a) => Self::F64(a.slice(args)?),
            Self::C64(a) => Self::C64(a.slice(args)?),
            Self::C128(a) => Self::C128(a.slice(args)?),
        })
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self> {
        Ok(match self {
            Self::Bool(a) => Self::Bool(a.broadcast_to(target_shape)?),
            Self::I8(a) => Self::I8(a.broadcast_to(target_shape)?),
            Self::I16(a) => Self::I16(a.broadcast_to(target_shape)?),
            Self::I32(a) => Self::I32(a.broadcast_to(target_shape)?),
            Self::I64(a) => Self::I64(a.broadcast_to(target_shape)?),
            Self::U8(a) => Self::U8(a.broadcast_to(target_shape)?),
            Self::U16(a) => Self::U16(a.broadcast_to(target_shape)?),
            Self::U32(a) => Self::U32(a.broadcast_to(target_shape)?),
            Self::U64(a) => Self::U64(a.broadcast_to(target_shape)?),
            Self::F16(a) => Self::F16(a.broadcast_to(target_shape)?),
            Self::F32(a) => Self::F32(a.broadcast_to(target_shape)?),
            Self::F64(a) => Self::F64(a.broadcast_to(target_shape)?),
            Self::C64(a) => Self::C64(a.broadcast_to(target_shape)?),
            Self::C128(a) => Self::C128(a.broadcast_to(target_shape)?),
        })
    }

    pub fn reshape(&self, spec: &[isize]) -> Result<Self> {
        Ok(match self {
            Self::Bool(a) => Self::Bool(a.reshape(spec)?),
            Self::I8(a) => Self::I8(a.reshape(spec)?),
            Self::I16(a) => Self::I16(a.reshape(spec)?),
            Self::I32(a) => Self::I32(a.reshape(spec)?),
            Self::I64(a) => Self::I64(a.reshape(spec)?),
            Self::U8(a) => Self::U8(a.reshape(spec)?),
            Self::U16(a) => Self::U16(a.reshape(spec)?),
            Self::U32(a) => Self::U32(a.reshape(spec)?),
            Self::U64(a) => Self::U64(a.reshape(spec)?),
            Self::F16(a) => Self::F16(a.reshape(spec)?),
            Self::F32(a) => Self::F32(a.reshape(spec)?),
            Self::F64(a) => Self::F64(a.reshape(spec)?),
            Self::C64(a) => Self::C64(a.reshape(spec)?),
            Self::C128(a) => Self::C128(a.reshape(spec)?),
        })
    }

    pub fn transpose(&self, axes: Option<&[isize]>) -> Result<Self> {
        Ok(match self {
            Self::Bool(a) => Self::Bool(a.transpose(axes)?),
            Self::I8(a) => Self::I8(a.transpose(axes)?),
            Self::I16(a) => Self::I16(a.transpose(axes)?),
            Self::I32(a) => Self::I32(a.transpose(axes)?),
            Self::I64(a) => Self::I64(a.transpose(axes)?),
            Self::U8(a) => Self::U8(a.transpose(axes)?),
            Self::U16(a) => Self::U16(a.transpose(axes)?),
            Self::U32(a) => Self::U32(a.transpose(axes)?),
            Self::U64(a) => Self::U64(a.transpose(axes)?),
            Self::F16(a) => Self::F16(a.transpose(axes)?),
            Self::F32(a) => Self::F32(a.transpose(axes)?),
            Self::F64(a) => Self::F64(a.transpose(axes)?),
            Self::C64(a) => Self::C64(a.transpose(axes)?),
            Self::C128(a) => Self::C128(a.transpose(axes)?),
        })
    }

    pub fn to_c_contiguous(&self) -> Self {
        match self {
            Self::Bool(a) => Self::Bool(a.to_c_contiguous()),
            Self::I8(a) => Self::I8(a.to_c_contiguous()),
            Self::I16(a) => Self::I16(a.to_c_contiguous()),
            Self::I32(a) => Self::I32(a.to_c_contiguous()),
            Self::I64(a) => Self::I64(a.to_c_contiguous()),
            Self::U8(a) => Self::U8(a.to_c_contiguous()),
            Self::U16(a) => Self::U16(a.to_c_contiguous()),
            Self::U32(a) => Self::U32(a.to_c_contiguous()),
            Self::U64(a) => Self::U64(a.to_c_contiguous()),
            Self::F16(a) => Self::F16(a.to_c_contiguous()),
            Self::F32(a) => Self::F32(a.to_c_contiguous()),
            Self::F64(a) => Self::F64(a.to_c_contiguous()),
            Self::C64(a) => Self::C64(a.to_c_contiguous()),
            Self::C128(a) => Self::C128(a.to_c_contiguous()),
        }
    }

    /// Cast to `to` dtype respecting `casting` rules (lossless exact conversion for integers up to `i64`/`u64`).
    pub fn astype(&self, to: DType, casting: Casting) -> Result<Self> {
        let from = self.dtype();
        if !can_cast(from, to, casting) {
            return Err(NumPyError::DTypeError(format!(
                "Cannot cast array data from dtype('{from}') to dtype('{to}') according to the rule '{casting:?}'"
            )));
        }
        if from == to {
            return Ok(self.clone());
        }
        Ok(match to {
            DType::Bool => Self::Bool(cast_to_bool(self)),
            DType::I8 => Self::I8(cast_to_i64(self).map(|v| v as i8)),
            DType::I16 => Self::I16(cast_to_i64(self).map(|v| v as i16)),
            DType::I32 => Self::I32(cast_to_i32(self)),
            DType::I64 => Self::I64(cast_to_i64(self)),
            DType::U8 => Self::U8(cast_to_narrow_uint(self, |v| v as u8)),
            DType::U16 => Self::U16(cast_to_narrow_uint(self, |v| v as u16)),
            DType::U32 => Self::U32(cast_to_narrow_uint(self, |v| v as u32)),
            DType::U64 => Self::U64(cast_to_u64(self)),
            DType::F16 => Self::F16(cast_to_typed(self, f16::from_f64_lossy)),
            DType::F32 => Self::F32(cast_to_typed(self, f32::from_f64_lossy)),
            DType::F64 => Self::F64(cast_to_typed(self, f64::from_f64_lossy)),
            DType::C64 => Self::C64(cast_to_complex(self, |c| Complex32::new(c.re as f32, c.im as f32))),
            DType::C128 => Self::C128(cast_to_complex(self, |c| c)),
        })
    }

    /// Promote dtypes and broadcast shapes of a binary pair according to NumPy 2.x rules.
    pub fn broadcast_and_promote(a: &Self, b: &Self) -> Result<(Self, Self)> {
        let out_dt = promote_types(a.dtype(), b.dtype());
        let out_shape = broadcast_shapes(a.shape(), b.shape())?;
        let a_cast = if a.dtype() == out_dt {
            a.clone()
        } else {
            a.astype(out_dt, Casting::Unsafe)?
        };
        let b_cast = if b.dtype() == out_dt {
            b.clone()
        } else {
            b.astype(out_dt, Casting::Unsafe)?
        };
        let a_b = if a_cast.shape() == out_shape.as_slice() {
            a_cast
        } else {
            a_cast.broadcast_to(&out_shape)?
        };
        let b_b = if b_cast.shape() == out_shape.as_slice() {
            b_cast
        } else {
            b_cast.broadcast_to(&out_shape)?
        };
        Ok((a_b, b_b))
    }

    /// Promote to a common float/complex dtype (at least `F64` for integers/bools, matching NumPy ufunc float outputs).
    pub fn promote_float_pair(a: &Self, b: &Self) -> Result<(Self, Self)> {
        let dt = promote_types(a.dtype(), b.dtype());
        let target = if dt.is_inexact() { dt } else { DType::F64 };
        let out_shape = broadcast_shapes(a.shape(), b.shape())?;
        let a_b = a.astype(target, Casting::Unsafe)?.broadcast_to(&out_shape)?;
        let b_b = b.astype(target, Casting::Unsafe)?.broadcast_to(&out_shape)?;
        Ok((a_b, b_b))
    }
}

fn cast_to_bool(src: &DynArray) -> NdArray<bool> {
    match src {
        DynArray::Bool(a) => a.clone(),
        DynArray::I8(a) => a.map(|x| x.is_truthy()),
        DynArray::I16(a) => a.map(|x| x.is_truthy()),
        DynArray::I32(a) => a.map(|x| x.is_truthy()),
        DynArray::I64(a) => a.map(|x| x.is_truthy()),
        DynArray::U8(a) => a.map(|x| x.is_truthy()),
        DynArray::U16(a) => a.map(|x| x.is_truthy()),
        DynArray::U32(a) => a.map(|x| x.is_truthy()),
        DynArray::U64(a) => a.map(|x| x.is_truthy()),
        DynArray::F16(a) => a.map(|x| x.is_truthy()),
        DynArray::F32(a) => a.map(|x| x.is_truthy()),
        DynArray::F64(a) => a.map(|x| x.is_truthy()),
        DynArray::C64(a) => a.map(|x| x.is_truthy()),
        DynArray::C128(a) => a.map(|x| x.is_truthy()),
    }
}

fn cast_to_typed<U: Scalar>(src: &DynArray, mut conv: impl FnMut(f64) -> U) -> NdArray<U> {
    match src {
        DynArray::Bool(a) => a.map(|x| conv(x.to_f64_lossy())),
        DynArray::I8(a) => a.map(|x| conv(x as f64)),
        DynArray::I16(a) => a.map(|x| conv(x as f64)),
        DynArray::I32(a) => a.map(|x| conv(x as f64)),
        DynArray::I64(a) => a.map(|x| conv(x as f64)),
        DynArray::U8(a) => a.map(|x| conv(x as f64)),
        DynArray::U16(a) => a.map(|x| conv(x as f64)),
        DynArray::U32(a) => a.map(|x| conv(x as f64)),
        DynArray::U64(a) => a.map(|x| conv(x as f64)),
        DynArray::F16(a) => a.map(|x| conv(x.to_f64())),
        DynArray::F32(a) => a.map(|x| conv(x as f64)),
        DynArray::F64(a) => a.map(|x| conv(x)),
        DynArray::C64(a) => a.map(|x| conv(x.re as f64)),
        DynArray::C128(a) => a.map(|x| conv(x.re)),
    }
}

#[inline]
fn f64_to_numpy_i64(x: f64) -> i64 {
    if x.is_finite() && (-9223372036854775808.0..9223372036854775808.0).contains(&x) {
        x.trunc() as i64
    } else {
        i64::MIN
    }
}

#[inline]
fn f64_to_numpy_i32(x: f64) -> i32 {
    let t = x.trunc();
    if t.is_finite() && (-2147483648.0..2147483648.0).contains(&t) {
        t as i32
    } else {
        i32::MIN
    }
}

#[inline]
fn f64_to_numpy_u64(x: f64) -> u64 {
    if !x.is_finite() {
        1u64 << 63
    } else if (0.0..18446744073709551616.0).contains(&x) {
        x.trunc() as u64
    } else if (-9223372036854775808.0..0.0).contains(&x) {
        (x.trunc() as i64) as u64
    } else {
        1u64 << 63
    }
}

fn cast_to_i64(src: &DynArray) -> NdArray<i64> {
    match src {
        DynArray::Bool(a) => a.map(|x| i64::from(x)),
        DynArray::I8(a) => a.map(|x| x as i64),
        DynArray::I16(a) => a.map(|x| x as i64),
        DynArray::I32(a) => a.map(|x| x as i64),
        DynArray::I64(a) => a.clone(),
        DynArray::U8(a) => a.map(|x| x as i64),
        DynArray::U16(a) => a.map(|x| x as i64),
        DynArray::U32(a) => a.map(|x| x as i64),
        DynArray::U64(a) => a.map(|x| x as i64),
        DynArray::F16(a) => a.map(|x| f64_to_numpy_i64(x.to_f64())),
        DynArray::F32(a) => a.map(|x| f64_to_numpy_i64(x as f64)),
        DynArray::F64(a) => a.map(f64_to_numpy_i64),
        DynArray::C64(a) => a.map(|x| f64_to_numpy_i64(x.re as f64)),
        DynArray::C128(a) => a.map(|x| f64_to_numpy_i64(x.re)),
    }
}

fn cast_to_i32(src: &DynArray) -> NdArray<i32> {
    match src {
        DynArray::Bool(a) => a.map(|x| i32::from(x)),
        DynArray::I8(a) => a.map(|x| x as i32),
        DynArray::I16(a) => a.map(|x| x as i32),
        DynArray::I32(a) => a.clone(),
        DynArray::I64(a) => a.map(|x| x as i32),
        DynArray::U8(a) => a.map(|x| x as i32),
        DynArray::U16(a) => a.map(|x| x as i32),
        DynArray::U32(a) => a.map(|x| x as i32),
        DynArray::U64(a) => a.map(|x| x as i32),
        DynArray::F16(a) => a.map(|x| f64_to_numpy_i32(x.to_f64())),
        DynArray::F32(a) => a.map(|x| f64_to_numpy_i32(x as f64)),
        DynArray::F64(a) => a.map(f64_to_numpy_i32),
        DynArray::C64(a) => a.map(|x| f64_to_numpy_i32(x.re as f64)),
        DynArray::C128(a) => a.map(|x| f64_to_numpy_i32(x.re)),
    }
}

fn cast_to_narrow_uint<U: Scalar>(src: &DynArray, mut conv: impl FnMut(u64) -> U) -> NdArray<U> {
    match src {
        DynArray::Bool(a) => a.map(|x| conv(u64::from(x))),
        DynArray::I8(a) => a.map(|x| conv(x as u64)),
        DynArray::I16(a) => a.map(|x| conv(x as u64)),
        DynArray::I32(a) => a.map(|x| conv(x as u64)),
        DynArray::I64(a) => a.map(|x| conv(x as u64)),
        DynArray::U8(a) => a.map(|x| conv(x as u64)),
        DynArray::U16(a) => a.map(|x| conv(x as u64)),
        DynArray::U32(a) => a.map(|x| conv(x as u64)),
        DynArray::U64(a) => a.map(|x| conv(x)),
        DynArray::F16(a) => a.map(|x| conv(f64_to_numpy_i64(x.to_f64()) as u64)),
        DynArray::F32(a) => a.map(|x| conv(f64_to_numpy_i64(x as f64) as u64)),
        DynArray::F64(a) => a.map(|x| conv(f64_to_numpy_i64(x) as u64)),
        DynArray::C64(a) => a.map(|x| conv(f64_to_numpy_i64(x.re as f64) as u64)),
        DynArray::C128(a) => a.map(|x| conv(f64_to_numpy_i64(x.re) as u64)),
    }
}

fn cast_to_u64(src: &DynArray) -> NdArray<u64> {
    match src {
        DynArray::Bool(a) => a.map(|x| u64::from(x)),
        DynArray::I8(a) => a.map(|x| x as u64),
        DynArray::I16(a) => a.map(|x| x as u64),
        DynArray::I32(a) => a.map(|x| x as u64),
        DynArray::I64(a) => a.map(|x| x as u64),
        DynArray::U8(a) => a.map(|x| x as u64),
        DynArray::U16(a) => a.map(|x| x as u64),
        DynArray::U32(a) => a.map(|x| x as u64),
        DynArray::U64(a) => a.clone(),
        DynArray::F16(a) => a.map(|x| f64_to_numpy_u64(x.to_f64())),
        DynArray::F32(a) => a.map(|x| f64_to_numpy_u64(x as f64)),
        DynArray::F64(a) => a.map(f64_to_numpy_u64),
        DynArray::C64(a) => a.map(|x| f64_to_numpy_u64(x.re as f64)),
        DynArray::C128(a) => a.map(|x| f64_to_numpy_u64(x.re)),
    }
}

fn cast_to_complex<U: Scalar>(src: &DynArray, mut conv: impl FnMut(Complex64) -> U) -> NdArray<U> {
    match src {
        DynArray::Bool(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::I8(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::I16(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::I32(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::I64(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::U8(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::U16(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::U32(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::U64(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::F16(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::F32(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::F64(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::C64(a) => a.map(|x| conv(x.to_complex64())),
        DynArray::C128(a) => a.map(|x| conv(x)),
    }
}

macro_rules! impl_from_typed {
    ($t:ty, $var:ident) => {
        impl From<NdArray<$t>> for DynArray {
            #[inline]
            fn from(arr: NdArray<$t>) -> Self {
                Self::$var(arr)
            }
        }
        impl TryFrom<DynArray> for NdArray<$t> {
            type Error = NumPyError;
            fn try_from(dyn_arr: DynArray) -> Result<Self> {
                match dyn_arr {
                    DynArray::$var(a) => Ok(a),
                    other => Err(NumPyError::DTypeError(format!(
                        "expected dtype '{}', got '{}'",
                        <$t as Scalar>::DTYPE,
                        other.dtype()
                    ))),
                }
            }
        }
    };
}

impl_from_typed!(bool, Bool);
impl_from_typed!(i8, I8);
impl_from_typed!(i16, I16);
impl_from_typed!(i32, I32);
impl_from_typed!(i64, I64);
impl_from_typed!(u8, U8);
impl_from_typed!(u16, U16);
impl_from_typed!(u32, U32);
impl_from_typed!(u64, U64);
impl_from_typed!(f16, F16);
impl_from_typed!(f32, F32);
impl_from_typed!(f64, F64);
impl_from_typed!(Complex32, C64);
impl_from_typed!(Complex64, C128);

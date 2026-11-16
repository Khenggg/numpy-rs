//! Higher-order NumPy ufunc methods: `reduce`, `accumulate`, `outer`, and `at`.
//!
//! Supports both:
//! - [`DynArray`] with exact NumPy 2.x promotion, integer-widening (`Add`/`Multiply`
//!   widen `bool/i8/i16/i32 -> i64` and `u8/u16/u32 -> u64`), `.identity`, reorderability,
//!   `keepdims`, multi-axis reductions, and unbuffered `.at` indexing.
//! - Zero-cost typed [`NdArray<T>`] variants (`*_nd` and closure-generic `*_with`).

use crate::dyn_ufunc;
use crate::exec::{dtype_err, empty_like_shape, PARALLEL_THRESHOLD};
use crate::scalar_ops::*;
use crate::simd;
use crate::typed_ufunc::UfuncScalar;
use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{
    broadcast_many_shapes, broadcast_shapes, dispatch_dyn, normalize_axis, promote_types,
    shape_size, AlignedBuffer, Casting, DType, DynArray, NdArray, NumPyError, Order, Result,
    Scalar,
};
use rayon::prelude::*;
use std::borrow::Cow;

const PARALLEL_CHUNK: usize = 16_384;

#[derive(Copy, Clone)]
struct SendPtr<T>(*const T);

unsafe impl<T: Sync> Send for SendPtr<T> {}
unsafe impl<T: Sync> Sync for SendPtr<T> {}

impl<T: Copy> SendPtr<T> {
    #[inline(always)]
    unsafe fn add_offset(self, off: isize) -> T {
        unsafe { *self.0.offset(off) }
    }
}

/// Scalar constant across all 14 supported NumPy dtypes (used for `initial`, `.identity()`, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
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

impl From<bool> for ScalarValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i8> for ScalarValue {
    fn from(v: i8) -> Self {
        Self::I8(v)
    }
}
impl From<i16> for ScalarValue {
    fn from(v: i16) -> Self {
        Self::I16(v)
    }
}
impl From<i32> for ScalarValue {
    fn from(v: i32) -> Self {
        Self::I32(v)
    }
}
impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self {
        Self::I64(v)
    }
}
impl From<u8> for ScalarValue {
    fn from(v: u8) -> Self {
        Self::U8(v)
    }
}
impl From<u16> for ScalarValue {
    fn from(v: u16) -> Self {
        Self::U16(v)
    }
}
impl From<u32> for ScalarValue {
    fn from(v: u32) -> Self {
        Self::U32(v)
    }
}
impl From<u64> for ScalarValue {
    fn from(v: u64) -> Self {
        Self::U64(v)
    }
}
impl From<f16> for ScalarValue {
    fn from(v: f16) -> Self {
        Self::F16(v)
    }
}
impl From<f32> for ScalarValue {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}
impl From<f64> for ScalarValue {
    fn from(v: f64) -> Self {
        Self::F64(v)
    }
}
impl From<Complex32> for ScalarValue {
    fn from(v: Complex32) -> Self {
        Self::C64(v)
    }
}
impl From<Complex64> for ScalarValue {
    fn from(v: Complex64) -> Self {
        Self::C128(v)
    }
}

impl ScalarValue {
    #[inline]
    pub fn dtype(self) -> DType {
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
    pub fn cast_to(self, target: DType) -> Self {
        if self.dtype() == target {
            return self;
        }
        macro_rules! from_num {
            ($x:expr, $target:expr) => {{
                let v = $x;
                match $target {
                    DType::Bool => Self::Bool((v as f64) != 0.0),
                    DType::I8 => Self::I8(v as i8),
                    DType::I16 => Self::I16(v as i16),
                    DType::I32 => Self::I32(v as i32),
                    DType::I64 => Self::I64(v as i64),
                    DType::U8 => Self::U8(v as u8),
                    DType::U16 => Self::U16(v as u16),
                    DType::U32 => Self::U32(v as u32),
                    DType::U64 => Self::U64(v as u64),
                    DType::F16 => Self::F16(f16::from_f64(v as f64)),
                    DType::F32 => Self::F32(v as f32),
                    DType::F64 => Self::F64(v as f64),
                    DType::C64 => Self::C64(Complex32::new(v as f32, 0.0)),
                    DType::C128 => Self::C128(Complex64::new(v as f64, 0.0)),
                }
            }};
        }
        match self {
            Self::Bool(b) => {
                let x = if b { 1i64 } else { 0i64 };
                match target {
                    DType::Bool => Self::Bool(b),
                    DType::I8 => Self::I8(x as i8),
                    DType::I16 => Self::I16(x as i16),
                    DType::I32 => Self::I32(x as i32),
                    DType::I64 => Self::I64(x),
                    DType::U8 => Self::U8(x as u8),
                    DType::U16 => Self::U16(x as u16),
                    DType::U32 => Self::U32(x as u32),
                    DType::U64 => Self::U64(x as u64),
                    DType::F16 => Self::F16(f16::from_f32(x as f32)),
                    DType::F32 => Self::F32(x as f32),
                    DType::F64 => Self::F64(x as f64),
                    DType::C64 => Self::C64(Complex32::new(x as f32, 0.0)),
                    DType::C128 => Self::C128(Complex64::new(x as f64, 0.0)),
                }
            }
            Self::I8(v) => from_num!(v, target),
            Self::I16(v) => from_num!(v, target),
            Self::I32(v) => from_num!(v, target),
            Self::I64(v) => from_num!(v, target),
            Self::U8(v) => from_num!(v, target),
            Self::U16(v) => from_num!(v, target),
            Self::U32(v) => from_num!(v, target),
            Self::U64(v) => from_num!(v, target),
            Self::F16(v) => from_num!(v.to_f64(), target),
            Self::F32(v) => from_num!(v, target),
            Self::F64(v) => from_num!(v, target),
            Self::C64(c) => match target {
                DType::Bool => Self::Bool(c.re != 0.0 || c.im != 0.0),
                DType::C64 => Self::C64(c),
                DType::C128 => Self::C128(Complex64::new(c.re as f64, c.im as f64)),
                _ => Self::F32(c.re).cast_to(target),
            },
            Self::C128(c) => match target {
                DType::Bool => Self::Bool(c.re != 0.0 || c.im != 0.0),
                DType::C64 => Self::C64(Complex32::new(c.re as f32, c.im as f32)),
                DType::C128 => Self::C128(c),
                _ => Self::F64(c.re).cast_to(target),
            },
        }
    }

    /// Strict NumPy-compatible conversion for `reduce(..., initial=...)` matching NumPy's
    /// scalar-to-dtype validation (rejecting NaN / infinity / out-of-bounds ints instead of saturating).
    pub fn try_cast_initial(self, target: DType) -> Result<Self> {
        if self.dtype() == target {
            return Ok(self);
        }
        if matches!(self, Self::C64(_) | Self::C128(_)) && !target.is_complex() && target != DType::Bool
        {
            return Err(NumPyError::DTypeError(format!(
                "Cannot cast scalar complex initial value to dtype('{target}')"
            )));
        }
        if !target.is_integer() {
            return Ok(self.cast_to(target));
        }

        let val_i128: Option<i128> = match self {
            Self::Bool(b) => Some(i128::from(b)),
            Self::I8(v) => Some(i128::from(v)),
            Self::I16(v) => Some(i128::from(v)),
            Self::I32(v) => Some(i128::from(v)),
            Self::I64(v) => Some(i128::from(v)),
            Self::U8(v) => Some(i128::from(v)),
            Self::U16(v) => Some(i128::from(v)),
            Self::U32(v) => Some(i128::from(v)),
            Self::U64(v) => Some(i128::from(v)),
            Self::F16(_) | Self::F32(_) | Self::F64(_) => None,
            Self::C64(_) | Self::C128(_) => unreachable!(),
        };

        if let Some(iv) = val_i128 {
            return match target {
                DType::I8 => i8::try_from(iv).map(Self::I8),
                DType::I16 => i16::try_from(iv).map(Self::I16),
                DType::I32 => i32::try_from(iv).map(Self::I32),
                DType::I64 => i64::try_from(iv).map(Self::I64),
                DType::U8 => u8::try_from(iv).map(Self::U8),
                DType::U16 => u16::try_from(iv).map(Self::U16),
                DType::U32 => u32::try_from(iv).map(Self::U32),
                DType::U64 => u64::try_from(iv).map(Self::U64),
                _ => unreachable!(),
            }
            .map_err(|_| {
                NumPyError::InvalidValue(format!("initial integer {iv} out of bounds for {target}"))
            });
        }

        let fv = match self {
            Self::F16(v) => v.to_f64(),
            Self::F32(v) => v as f64,
            Self::F64(v) => v,
            _ => unreachable!(),
        };
        if fv.is_nan() {
            return Err(NumPyError::InvalidValue(
                "cannot convert float NaN to integer".into(),
            ));
        }
        if !fv.is_finite() {
            return Err(NumPyError::InvalidValue(
                "cannot convert float infinity to integer".into(),
            ));
        }
        let t = fv.trunc();
        let in_range = match target {
            DType::I8 => (-128.0..=127.0).contains(&t),
            DType::I16 => (-32768.0..=32767.0).contains(&t),
            DType::I32 => (-2147483648.0..=2147483647.0).contains(&t),
            DType::I64 => (-9223372036854775808.0..9223372036854775808.0).contains(&t),
            DType::U8 => (0.0..=255.0).contains(&t),
            DType::U16 => (0.0..=65535.0).contains(&t),
            DType::U32 => (0.0..=4294967295.0).contains(&t),
            DType::U64 => (0.0..18446744073709551616.0).contains(&t),
            _ => unreachable!(),
        };
        if !in_range {
            return Err(NumPyError::InvalidValue(format!(
                "initial float {fv} too big to convert to {target}"
            )));
        }
        Ok(match target {
            DType::I8 => Self::I8(t as i8),
            DType::I16 => Self::I16(t as i16),
            DType::I32 => Self::I32(t as i32),
            DType::I64 => Self::I64(t as i64),
            DType::U8 => Self::U8(t as u8),
            DType::U16 => Self::U16(t as u16),
            DType::U32 => Self::U32(t as u32),
            DType::U64 => Self::U64(t as u64),
            _ => unreachable!(),
        })
    }
}

/// Binary ufunc descriptor supporting `.reduce`, `.accumulate`, `.outer`, and `.at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    TrueDivide,
    FloorDivide,
    Remainder,
    Fmod,
    Power,
    FloatPower,
    Logaddexp,
    Logaddexp2,
    Arctan2,
    Hypot,
    Copysign,
    Nextafter,
    Heaviside,
    Maximum,
    Minimum,
    Fmax,
    Fmin,
    Gcd,
    Lcm,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
}

/// Alias for [`BinaryOp`] emphasizing its role as a ufunc object.
pub type BinaryUfunc = BinaryOp;

pub const ADD: BinaryOp = BinaryOp::Add;
pub const SUBTRACT: BinaryOp = BinaryOp::Subtract;
pub const MULTIPLY: BinaryOp = BinaryOp::Multiply;
pub const DIVIDE: BinaryOp = BinaryOp::TrueDivide;
pub const TRUE_DIVIDE: BinaryOp = BinaryOp::TrueDivide;
pub const FLOOR_DIVIDE: BinaryOp = BinaryOp::FloorDivide;
pub const REMAINDER: BinaryOp = BinaryOp::Remainder;
pub const MODULO: BinaryOp = BinaryOp::Remainder;
pub const MOD: BinaryOp = BinaryOp::Remainder;
pub const FMOD: BinaryOp = BinaryOp::Fmod;
pub const POWER: BinaryOp = BinaryOp::Power;
pub const FLOAT_POWER: BinaryOp = BinaryOp::FloatPower;
pub const LOGADDEXP: BinaryOp = BinaryOp::Logaddexp;
pub const LOGADDEXP2: BinaryOp = BinaryOp::Logaddexp2;
pub const ARCTAN2: BinaryOp = BinaryOp::Arctan2;
pub const ATAN2: BinaryOp = BinaryOp::Arctan2;
pub const HYPOT: BinaryOp = BinaryOp::Hypot;
pub const COPYSIGN: BinaryOp = BinaryOp::Copysign;
pub const NEXTAFTER: BinaryOp = BinaryOp::Nextafter;
pub const HEAVISIDE: BinaryOp = BinaryOp::Heaviside;
pub const MAXIMUM: BinaryOp = BinaryOp::Maximum;
pub const MINIMUM: BinaryOp = BinaryOp::Minimum;
pub const FMAX: BinaryOp = BinaryOp::Fmax;
pub const FMIN: BinaryOp = BinaryOp::Fmin;
pub const GCD: BinaryOp = BinaryOp::Gcd;
pub const LCM: BinaryOp = BinaryOp::Lcm;
pub const BITWISE_AND: BinaryOp = BinaryOp::BitwiseAnd;
pub const BITWISE_OR: BinaryOp = BinaryOp::BitwiseOr;
pub const BITWISE_XOR: BinaryOp = BinaryOp::BitwiseXor;
pub const LEFT_SHIFT: BinaryOp = BinaryOp::LeftShift;
pub const RIGHT_SHIFT: BinaryOp = BinaryOp::RightShift;
pub const EQUAL: BinaryOp = BinaryOp::Equal;
pub const NOT_EQUAL: BinaryOp = BinaryOp::NotEqual;
pub const LESS: BinaryOp = BinaryOp::Less;
pub const LESS_EQUAL: BinaryOp = BinaryOp::LessEqual;
pub const GREATER: BinaryOp = BinaryOp::Greater;
pub const GREATER_EQUAL: BinaryOp = BinaryOp::GreaterEqual;
pub const LOGICAL_AND: BinaryOp = BinaryOp::LogicalAnd;
pub const LOGICAL_OR: BinaryOp = BinaryOp::LogicalOr;
pub const LOGICAL_XOR: BinaryOp = BinaryOp::LogicalXor;

/// Unary ufunc descriptor supporting unbuffered `.at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Negative,
    Positive,
    Abs,
    Fabs,
    Square,
    Reciprocal,
    Sqrt,
    Cbrt,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Tanh,
    Arcsinh,
    Arccosh,
    Arctanh,
    Deg2Rad,
    Rad2Deg,
    Floor,
    Ceil,
    Trunc,
    Rint,
    Sign,
    Spacing,
    Real,
    Imag,
    Conj,
    Invert,
    LogicalNot,
    Sinc,
    I0,
}

/// Alias for [`UnaryOp`] emphasizing its role as a unary ufunc object.
pub type UnaryUfunc = UnaryOp;

pub const NEGATIVE: UnaryOp = UnaryOp::Negative;
pub const POSITIVE: UnaryOp = UnaryOp::Positive;
pub const ABS: UnaryOp = UnaryOp::Abs;
pub const ABSOLUTE: UnaryOp = UnaryOp::Abs;
pub const FABS: UnaryOp = UnaryOp::Fabs;
pub const SQUARE: UnaryOp = UnaryOp::Square;
pub const RECIPROCAL: UnaryOp = UnaryOp::Reciprocal;
pub const SQRT: UnaryOp = UnaryOp::Sqrt;
pub const CBRT: UnaryOp = UnaryOp::Cbrt;
pub const EXP: UnaryOp = UnaryOp::Exp;
pub const EXP2: UnaryOp = UnaryOp::Exp2;
pub const EXPM1: UnaryOp = UnaryOp::Expm1;
pub const LOG: UnaryOp = UnaryOp::Log;
pub const LOG2: UnaryOp = UnaryOp::Log2;
pub const LOG10: UnaryOp = UnaryOp::Log10;
pub const LOG1P: UnaryOp = UnaryOp::Log1p;
pub const SIN: UnaryOp = UnaryOp::Sin;
pub const COS: UnaryOp = UnaryOp::Cos;
pub const TAN: UnaryOp = UnaryOp::Tan;
pub const ARCSIN: UnaryOp = UnaryOp::Arcsin;
pub const ASIN: UnaryOp = UnaryOp::Arcsin;
pub const ARCCOS: UnaryOp = UnaryOp::Arccos;
pub const ACOS: UnaryOp = UnaryOp::Arccos;
pub const ARCTAN: UnaryOp = UnaryOp::Arctan;
pub const ATAN: UnaryOp = UnaryOp::Arctan;
pub const SINH: UnaryOp = UnaryOp::Sinh;
pub const COSH: UnaryOp = UnaryOp::Cosh;
pub const TANH: UnaryOp = UnaryOp::Tanh;
pub const ARCSINH: UnaryOp = UnaryOp::Arcsinh;
pub const ASINH: UnaryOp = UnaryOp::Arcsinh;
pub const ARCCOSH: UnaryOp = UnaryOp::Arccosh;
pub const ACOSH: UnaryOp = UnaryOp::Arccosh;
pub const ARCTANH: UnaryOp = UnaryOp::Arctanh;
pub const ATANH: UnaryOp = UnaryOp::Arctanh;
pub const DEG2RAD: UnaryOp = UnaryOp::Deg2Rad;
pub const RADIANS: UnaryOp = UnaryOp::Deg2Rad;
pub const RAD2DEG: UnaryOp = UnaryOp::Rad2Deg;
pub const DEGREES: UnaryOp = UnaryOp::Rad2Deg;
pub const FLOOR: UnaryOp = UnaryOp::Floor;
pub const CEIL: UnaryOp = UnaryOp::Ceil;
pub const TRUNC: UnaryOp = UnaryOp::Trunc;
pub const FIX: UnaryOp = UnaryOp::Trunc;
pub const RINT: UnaryOp = UnaryOp::Rint;
pub const SIGN: UnaryOp = UnaryOp::Sign;
pub const SPACING: UnaryOp = UnaryOp::Spacing;
pub const REAL: UnaryOp = UnaryOp::Real;
pub const IMAG: UnaryOp = UnaryOp::Imag;
pub const CONJ: UnaryOp = UnaryOp::Conj;
pub const CONJUGATE: UnaryOp = UnaryOp::Conj;
pub const INVERT: UnaryOp = UnaryOp::Invert;
pub const BITWISE_NOT: UnaryOp = UnaryOp::Invert;
pub const LOGICAL_NOT: UnaryOp = UnaryOp::LogicalNot;
pub const SINC: UnaryOp = UnaryOp::Sinc;
pub const I0: UnaryOp = UnaryOp::I0;

/// Indexing specification accepted by [`BinaryOp::at`] and [`UnaryOp::at`].
pub enum AtIndex<'a> {
    /// A single index array (integer array for axis 0, or boolean mask covering prefix axes).
    Array(&'a DynArray),
    /// An owned `DynArray` index (used by typed `&NdArray<T>` conversions via O(1) `Arc` clone).
    OwnedArray(DynArray),
    /// A typed `&NdArray<isize>` index array along axis 0.
    NdArrayIsize(&'a NdArray<isize>),
    /// A typed `&NdArray<usize>` index array along axis 0.
    NdArrayUsize(&'a NdArray<usize>),
    /// Multiple index arrays indexing leading axes `(idx_0, idx_1, ...)`.
    Tuple(&'a [&'a DynArray]),
    /// Flat integer slice along axis 0.
    SliceIsize(&'a [isize]),
    /// Flat `i64` slice along axis 0.
    SliceI64(&'a [i64]),
    /// Flat `i32` slice along axis 0.
    SliceI32(&'a [i32]),
    /// Flat `usize` slice along axis 0.
    SliceUsize(&'a [usize]),
}

impl<'a> From<&'a DynArray> for AtIndex<'a> {
    fn from(v: &'a DynArray) -> Self {
        Self::Array(v)
    }
}
impl<'a> From<&'a NdArray<bool>> for AtIndex<'a> {
    fn from(v: &'a NdArray<bool>) -> Self {
        Self::OwnedArray(DynArray::Bool(v.clone()))
    }
}
impl<'a> From<&'a NdArray<i8>> for AtIndex<'a> {
    fn from(v: &'a NdArray<i8>) -> Self {
        Self::OwnedArray(DynArray::I8(v.clone()))
    }
}
impl<'a> From<&'a NdArray<i16>> for AtIndex<'a> {
    fn from(v: &'a NdArray<i16>) -> Self {
        Self::OwnedArray(DynArray::I16(v.clone()))
    }
}
impl<'a> From<&'a NdArray<i32>> for AtIndex<'a> {
    fn from(v: &'a NdArray<i32>) -> Self {
        Self::OwnedArray(DynArray::I32(v.clone()))
    }
}
impl<'a> From<&'a NdArray<i64>> for AtIndex<'a> {
    fn from(v: &'a NdArray<i64>) -> Self {
        Self::OwnedArray(DynArray::I64(v.clone()))
    }
}
impl<'a> From<&'a NdArray<u8>> for AtIndex<'a> {
    fn from(v: &'a NdArray<u8>) -> Self {
        Self::OwnedArray(DynArray::U8(v.clone()))
    }
}
impl<'a> From<&'a NdArray<u16>> for AtIndex<'a> {
    fn from(v: &'a NdArray<u16>) -> Self {
        Self::OwnedArray(DynArray::U16(v.clone()))
    }
}
impl<'a> From<&'a NdArray<u32>> for AtIndex<'a> {
    fn from(v: &'a NdArray<u32>) -> Self {
        Self::OwnedArray(DynArray::U32(v.clone()))
    }
}
impl<'a> From<&'a NdArray<u64>> for AtIndex<'a> {
    fn from(v: &'a NdArray<u64>) -> Self {
        Self::OwnedArray(DynArray::U64(v.clone()))
    }
}
impl<'a> From<&'a NdArray<isize>> for AtIndex<'a> {
    fn from(v: &'a NdArray<isize>) -> Self {
        Self::NdArrayIsize(v)
    }
}
impl<'a> From<&'a NdArray<usize>> for AtIndex<'a> {
    fn from(v: &'a NdArray<usize>) -> Self {
        Self::NdArrayUsize(v)
    }
}
impl<'a> From<&'a [&'a DynArray]> for AtIndex<'a> {
    fn from(v: &'a [&'a DynArray]) -> Self {
        Self::Tuple(v)
    }
}
impl<'a, const N: usize> From<&'a [&'a DynArray; N]> for AtIndex<'a> {
    fn from(v: &'a [&'a DynArray; N]) -> Self {
        Self::Tuple(v.as_slice())
    }
}
impl<'a> From<&'a [isize]> for AtIndex<'a> {
    fn from(v: &'a [isize]) -> Self {
        Self::SliceIsize(v)
    }
}
impl<'a, const N: usize> From<&'a [isize; N]> for AtIndex<'a> {
    fn from(v: &'a [isize; N]) -> Self {
        Self::SliceIsize(v.as_slice())
    }
}
impl<'a> From<&'a [i64]> for AtIndex<'a> {
    fn from(v: &'a [i64]) -> Self {
        Self::SliceI64(v)
    }
}
impl<'a, const N: usize> From<&'a [i64; N]> for AtIndex<'a> {
    fn from(v: &'a [i64; N]) -> Self {
        Self::SliceI64(v.as_slice())
    }
}
impl<'a> From<&'a [i32]> for AtIndex<'a> {
    fn from(v: &'a [i32]) -> Self {
        Self::SliceI32(v)
    }
}
impl<'a, const N: usize> From<&'a [i32; N]> for AtIndex<'a> {
    fn from(v: &'a [i32; N]) -> Self {
        Self::SliceI32(v.as_slice())
    }
}
impl<'a> From<&'a [usize]> for AtIndex<'a> {
    fn from(v: &'a [usize]) -> Self {
        Self::SliceUsize(v)
    }
}
impl<'a, const N: usize> From<&'a [usize; N]> for AtIndex<'a> {
    fn from(v: &'a [usize; N]) -> Self {
        Self::SliceUsize(v.as_slice())
    }
}

impl BinaryOp {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::TrueDivide => "true_divide",
            Self::FloorDivide => "floor_divide",
            Self::Remainder => "remainder",
            Self::Fmod => "fmod",
            Self::Power => "power",
            Self::FloatPower => "float_power",
            Self::Logaddexp => "logaddexp",
            Self::Logaddexp2 => "logaddexp2",
            Self::Arctan2 => "arctan2",
            Self::Hypot => "hypot",
            Self::Copysign => "copysign",
            Self::Nextafter => "nextafter",
            Self::Heaviside => "heaviside",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::Fmax => "fmax",
            Self::Fmin => "fmin",
            Self::Gcd => "gcd",
            Self::Lcm => "lcm",
            Self::BitwiseAnd => "bitwise_and",
            Self::BitwiseOr => "bitwise_or",
            Self::BitwiseXor => "bitwise_xor",
            Self::LeftShift => "left_shift",
            Self::RightShift => "right_shift",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
            Self::LogicalAnd => "logical_and",
            Self::LogicalOr => "logical_or",
            Self::LogicalXor => "logical_xor",
        }
    }

    /// Whether multi-axis `.reduce(...)` is supported (matches NumPy `ufunc.reduce` reorderable flag).
    #[inline]
    pub const fn is_reorderable(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Multiply
                | Self::Logaddexp
                | Self::Logaddexp2
                | Self::Hypot
                | Self::Maximum
                | Self::Minimum
                | Self::Fmax
                | Self::Fmin
                | Self::Gcd
                | Self::BitwiseAnd
                | Self::BitwiseOr
                | Self::BitwiseXor
                | Self::LogicalAnd
                | Self::LogicalOr
                | Self::LogicalXor
        )
    }

    #[inline]
    const fn is_associative(self) -> bool {
        self.is_reorderable() || matches!(self, Self::Lcm)
    }

    #[inline]
    const fn can_error_per_element(self) -> bool {
        matches!(self, Self::Power)
    }

    /// Return NumPy's identity element for this ufunc at `dtype`, if it has one.
    pub fn identity(self, dtype: DType) -> Result<Option<ScalarValue>> {
        let comp_dt = self.reduce_comp_dtype(dtype, None)?;
        Ok(raw_identity(self, comp_dt))
    }

    fn validate_comp_dtype(self, dt: DType) -> Result<()> {
        match self {
            Self::Add | Self::Multiply | Self::Maximum | Self::Minimum | Self::Fmax | Self::Fmin => {
                Ok(())
            }
            Self::Power => {
                if dt == DType::Bool {
                    Err(dtype_err("power", "boolean dtype not supported for power loop"))
                } else {
                    Ok(())
                }
            }
            Self::LogicalAnd | Self::LogicalOr | Self::LogicalXor => {
                if dt == DType::Bool {
                    Ok(())
                } else {
                    Err(dtype_err(
                        self.name(),
                        "logical reduction requires boolean dtype",
                    ))
                }
            }
            Self::Subtract => {
                if dt == DType::Bool {
                    Err(dtype_err(self.name(), "boolean inputs"))
                } else {
                    Ok(())
                }
            }
            Self::TrueDivide => {
                if dt.is_inexact() {
                    Ok(())
                } else {
                    Err(dtype_err(self.name(), "requires floating/complex dtype"))
                }
            }
            Self::FloatPower => {
                if matches!(dt, DType::F64 | DType::C128) {
                    Ok(())
                } else {
                    Err(dtype_err(self.name(), "requires float64 or complex128"))
                }
            }
            Self::FloorDivide | Self::Remainder | Self::Fmod => {
                if dt.is_complex() || dt == DType::Bool {
                    Err(dtype_err(self.name(), "not supported for bool or complex"))
                } else {
                    Ok(())
                }
            }
            Self::Logaddexp
            | Self::Logaddexp2
            | Self::Arctan2
            | Self::Hypot
            | Self::Copysign
            | Self::Nextafter
            | Self::Heaviside => {
                if dt.is_float() {
                    Ok(())
                } else {
                    Err(dtype_err(self.name(), "requires real floating dtype"))
                }
            }
            Self::Gcd | Self::Lcm => {
                if dt.is_integer() {
                    Ok(())
                } else {
                    Err(dtype_err(self.name(), "requires integer dtype"))
                }
            }
            Self::BitwiseAnd | Self::BitwiseOr | Self::BitwiseXor => {
                if dt == DType::Bool || dt.is_integer() {
                    Ok(())
                } else {
                    Err(dtype_err(self.name(), "requires integer or boolean dtype"))
                }
            }
            Self::LeftShift | Self::RightShift => {
                if dt.is_integer() {
                    Ok(())
                } else {
                    Err(dtype_err(self.name(), "requires integer dtype"))
                }
            }
            Self::Equal
            | Self::NotEqual
            | Self::Less
            | Self::LessEqual
            | Self::Greater
            | Self::GreaterEqual => {
                if dt == DType::Bool {
                    Ok(())
                } else {
                    Err(dtype_err(
                        self.name(),
                        "reduction/accumulation on comparisons requires boolean dtype",
                    ))
                }
            }
        }
    }

    fn reduce_comp_dtype(self, in_dt: DType, explicit: Option<DType>) -> Result<DType> {
        if let Some(dt) = explicit {
            if matches!(
                self,
                Self::Equal
                    | Self::NotEqual
                    | Self::Less
                    | Self::LessEqual
                    | Self::Greater
                    | Self::GreaterEqual
            ) && in_dt != DType::Bool
            {
                return Err(dtype_err(
                    self.name(),
                    "comparison reduction only supports boolean arrays",
                ));
            }
            self.validate_comp_dtype(dt)?;
            return Ok(dt);
        }

        let dt = match self {
            Self::Add | Self::Multiply => match in_dt {
                DType::Bool | DType::I8 | DType::I16 | DType::I32 => DType::I64,
                DType::U8 | DType::U16 | DType::U32 => DType::U64,
                other => other,
            },
            Self::Subtract => {
                if in_dt == DType::Bool {
                    return Err(dtype_err("subtract", "boolean inputs"));
                }
                in_dt
            }
            Self::TrueDivide => {
                if in_dt.is_inexact() {
                    in_dt
                } else {
                    DType::F64
                }
            }
            Self::FloatPower => {
                if in_dt.is_complex() {
                    DType::C128
                } else {
                    DType::F64
                }
            }
            Self::FloorDivide | Self::Remainder | Self::Fmod | Self::Power => {
                if matches!(self, Self::FloorDivide | Self::Remainder | Self::Fmod)
                    && in_dt.is_complex()
                {
                    return Err(dtype_err(self.name(), "complex inputs"));
                }
                if in_dt == DType::Bool {
                    DType::I8
                } else {
                    in_dt
                }
            }
            Self::Logaddexp
            | Self::Logaddexp2
            | Self::Arctan2
            | Self::Hypot
            | Self::Copysign
            | Self::Nextafter
            | Self::Heaviside => {
                if in_dt.is_complex() {
                    return Err(dtype_err(self.name(), "complex inputs"));
                }
                real_float_target(in_dt)
            }
            Self::Maximum | Self::Minimum | Self::Fmax | Self::Fmin => in_dt,
            Self::Gcd | Self::Lcm => {
                if !in_dt.is_integer() {
                    return Err(dtype_err(self.name(), "requires integer inputs"));
                }
                in_dt
            }
            Self::BitwiseAnd | Self::BitwiseOr | Self::BitwiseXor => {
                if !(in_dt == DType::Bool || in_dt.is_integer()) {
                    return Err(dtype_err(self.name(), "requires integer or bool inputs"));
                }
                in_dt
            }
            Self::LeftShift | Self::RightShift => {
                if in_dt == DType::Bool {
                    DType::I8
                } else if in_dt.is_integer() {
                    in_dt
                } else {
                    return Err(dtype_err(self.name(), "requires integer or bool inputs"));
                }
            }
            Self::Equal
            | Self::NotEqual
            | Self::Less
            | Self::LessEqual
            | Self::Greater
            | Self::GreaterEqual => {
                if in_dt != DType::Bool {
                    return Err(dtype_err(
                        self.name(),
                        "comparison reduction only supports boolean arrays",
                    ));
                }
                DType::Bool
            }
            Self::LogicalAnd | Self::LogicalOr | Self::LogicalXor => DType::Bool,
        };
        Ok(dt)
    }

    fn at_comp_pair(self, a_dt: DType, b_dt: DType) -> Result<(DType, DType)> {
        let p = promote_types(a_dt, b_dt);
        match self {
            Self::Add | Self::Multiply | Self::Maximum | Self::Minimum | Self::Fmax | Self::Fmin => {
                Ok((p, p))
            }
            Self::Subtract => {
                if p == DType::Bool {
                    Err(dtype_err("subtract", "boolean inputs"))
                } else {
                    Ok((p, p))
                }
            }
            Self::TrueDivide => {
                let t = if p.is_inexact() { p } else { DType::F64 };
                Ok((t, t))
            }
            Self::FloatPower => {
                let t = if p.is_complex() { DType::C128 } else { DType::F64 };
                Ok((t, t))
            }
            Self::FloorDivide | Self::Remainder | Self::Fmod | Self::Power => {
                if matches!(self, Self::FloorDivide | Self::Remainder | Self::Fmod) && p.is_complex() {
                    return Err(dtype_err(self.name(), "complex inputs"));
                }
                let t = if p == DType::Bool { DType::I8 } else { p };
                Ok((t, t))
            }
            Self::Logaddexp
            | Self::Logaddexp2
            | Self::Arctan2
            | Self::Hypot
            | Self::Copysign
            | Self::Nextafter
            | Self::Heaviside => {
                if a_dt.is_complex() || b_dt.is_complex() {
                    return Err(dtype_err(self.name(), "complex inputs"));
                }
                let t = promote_types(real_float_target(a_dt), real_float_target(b_dt));
                Ok((t, t))
            }
            Self::Gcd | Self::Lcm => {
                if !a_dt.is_integer() || !b_dt.is_integer() || !p.is_integer() {
                    return Err(dtype_err(self.name(), "requires integer inputs"));
                }
                Ok((p, p))
            }
            Self::BitwiseAnd | Self::BitwiseOr | Self::BitwiseXor => {
                if !(p == DType::Bool || p.is_integer()) {
                    return Err(dtype_err(self.name(), "requires integer or bool inputs"));
                }
                Ok((p, p))
            }
            Self::LeftShift | Self::RightShift => {
                if !((a_dt == DType::Bool || a_dt.is_integer())
                    && (b_dt == DType::Bool || b_dt.is_integer()))
                {
                    return Err(dtype_err(self.name(), "requires integer or bool inputs"));
                }
                let t = if p == DType::Bool { DType::I8 } else { p };
                Ok((t, t))
            }
            Self::Equal
            | Self::NotEqual
            | Self::Less
            | Self::LessEqual
            | Self::Greater
            | Self::GreaterEqual => Ok((p, DType::Bool)),
            Self::LogicalAnd | Self::LogicalOr | Self::LogicalXor => Ok((DType::Bool, DType::Bool)),
        }
    }

    /// Reduce `a` over a single axis (`Some(axis)`) or all axes (`None`).
    pub fn reduce(
        self,
        a: &DynArray,
        axis: Option<isize>,
        dtype: Option<DType>,
        keepdims: bool,
        initial: Option<ScalarValue>,
    ) -> Result<DynArray> {
        let comp_dt = self.reduce_comp_dtype(a.dtype(), dtype)?;
        let axes: Vec<usize> = match axis {
            None => (0..a.ndim()).collect(),
            Some(ax) => vec![normalize_axis(ax, a.ndim())?],
        };
        reduce_dyn_impl(self, a, &axes, comp_dt, keepdims, initial)
    }

    /// Reduce `a` over multiple explicit `axes`.
    pub fn reduce_axes(
        self,
        a: &DynArray,
        axes: &[isize],
        dtype: Option<DType>,
        keepdims: bool,
        initial: Option<ScalarValue>,
    ) -> Result<DynArray> {
        let comp_dt = self.reduce_comp_dtype(a.dtype(), dtype)?;
        let mut norm = Vec::with_capacity(axes.len());
        let mut seen = vec![false; a.ndim()];
        for &ax in axes {
            let u = normalize_axis(ax, a.ndim())?;
            if seen[u] {
                return Err(NumPyError::InvalidValue("duplicate value in 'axis'".into()));
            }
            seen[u] = true;
            norm.push(u);
        }
        reduce_dyn_impl(self, a, &norm, comp_dt, keepdims, initial)
    }

    /// Cumulative reduction along `axis`.
    pub fn accumulate(
        self,
        a: &DynArray,
        axis: isize,
        dtype: Option<DType>,
    ) -> Result<DynArray> {
        if a.ndim() == 0 {
            return Err(NumPyError::InvalidValue("cannot accumulate on a scalar".into()));
        }
        let ax = normalize_axis(axis, a.ndim())?;
        let comp_dt = self.reduce_comp_dtype(a.dtype(), dtype)?;
        let cast_buf = if a.dtype() == comp_dt {
            Cow::Borrowed(a)
        } else {
            Cow::Owned(a.astype(comp_dt, Casting::Unsafe)?)
        };
        dispatch_dyn!(cast_buf.as_ref(), arr => Ok(DynArray::from(accumulate_homogeneous(self, arr, ax)?)))
    }

    /// Generalized outer application `op(a[i], b[j])` with shape `a.shape + b.shape`.
    pub fn outer(self, a: &DynArray, b: &DynArray) -> Result<DynArray> {
        let mut shape_a = Vec::with_capacity(a.ndim() + b.ndim());
        shape_a.extend_from_slice(a.shape());
        shape_a.resize(a.ndim() + b.ndim(), 1);
        let mut shape_b = Vec::with_capacity(a.ndim() + b.ndim());
        shape_b.resize(a.ndim(), 1);
        shape_b.extend_from_slice(b.shape());

        let spec_a: Vec<isize> = shape_a.iter().map(|&d| d as isize).collect();
        let spec_b: Vec<isize> = shape_b.iter().map(|&d| d as isize).collect();

        let a_view = if a.is_c_contiguous() {
            a.reshape(&spec_a)?
        } else {
            a.to_c_contiguous().reshape(&spec_a)?
        };
        let b_view = if b.is_c_contiguous() {
            b.reshape(&spec_b)?
        } else {
            b.to_c_contiguous().reshape(&spec_b)?
        };
        eval_dyn_binary(self, &a_view, &b_view)
    }

    /// Perform unbuffered in-place `a[indices] op= b`.
    pub fn at<'a>(
        self,
        a: &mut DynArray,
        indices: impl Into<AtIndex<'a>>,
        b: &DynArray,
    ) -> Result<()> {
        let idx = indices.into();
        let (comp_in_dt, comp_out_dt) = self.at_comp_pair(a.dtype(), b.dtype())?;
        let plan = resolve_at_plan(a.shape(), &idx)?;

        // Ensure `b` broadcasts cleanly INTO `plan.bcast_shape` without expanding `a`.
        let b_check = broadcast_shapes(&plan.bcast_shape, b.shape())?;
        if b_check != plan.bcast_shape {
            return Err(NumPyError::ShapeMismatch(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                plan.bcast_shape,
                b.shape()
            )));
        }

        let b_cast = if b.dtype() == comp_in_dt {
            Cow::Borrowed(b)
        } else {
            Cow::Owned(b.astype(comp_in_dt, Casting::Unsafe)?)
        };
        let b_bcast = b_cast.broadcast_to(&plan.bcast_shape)?;

        // Own the metadata: the loop mutates `a` through copy-on-write below.
        let trailing_shape = a.shape()[plan.consumed_axes..].to_vec();
        let trailing_len = shape_size(&trailing_shape)?;
        if plan.coords.is_empty() || trailing_len == 0 {
            return Ok(());
        }

        let a_dtype = a.dtype();
        let ndim = a.ndim();
        let mut full_coord = vec![0isize; ndim];
        let mut b_coord = vec![0isize; plan.bcast_shape.len()];
        let idx_prefix_len = plan.idx_bcast_shape.len();

        for (i_idx, prefix) in plan.coords.iter().enumerate() {
            for (k, &c) in prefix.iter().enumerate() {
                full_coord[k] = c;
            }
            if idx_prefix_len > 0 {
                linear_to_isize_coords(i_idx, &plan.idx_bcast_shape, &mut b_coord[..idx_prefix_len]);
            }
            for t_idx in 0..trailing_len {
                if !trailing_shape.is_empty() {
                    linear_to_isize_coords(
                        t_idx,
                        &trailing_shape,
                        &mut full_coord[plan.consumed_axes..],
                    );
                    let b_tail_off = idx_prefix_len;
                    b_coord[b_tail_off..]
                        .copy_from_slice(&full_coord[plan.consumed_axes..]);
                }
                let a_cur = dyn_get_scalar(a, &full_coord)?;
                let b_val = dyn_get_scalar(&b_bcast, &b_coord)?;
                let a_comp = a_cur.cast_to(comp_in_dt);
                let out_val = if comp_out_dt == DType::Bool
                    && matches!(
                        self,
                        Self::Equal
                            | Self::NotEqual
                            | Self::Less
                            | Self::LessEqual
                            | Self::Greater
                            | Self::GreaterEqual
                    ) {
                    ScalarValue::Bool(apply_cmp_scalar(self, comp_in_dt, a_comp, b_val)?)
                } else {
                    apply_binary_scalar(self, comp_in_dt, a_comp, b_val)?
                };
                dyn_set_scalar(a, &full_coord, out_val.cast_to(a_dtype))?;
            }
        }
        Ok(())
    }

    /// Typed homogeneous reduction on `NdArray<T>` preserving `T`.
    pub fn reduce_nd<T: HigherOrderScalar>(
        self,
        a: &NdArray<T>,
        axis: Option<isize>,
        keepdims: bool,
        initial: Option<T>,
    ) -> Result<NdArray<T>> {
        self.validate_comp_dtype(T::DTYPE)?;
        let axes: Vec<usize> = match axis {
            None => (0..a.ndim()).collect(),
            Some(ax) => vec![normalize_axis(ax, a.ndim())?],
        };
        let init_sv = initial.map(|v| v.to_scalar_value());
        reduce_homogeneous(self, a, &axes, keepdims, init_sv)
    }

    /// Typed homogeneous cumulative reduction on `NdArray<T>` preserving `T`.
    pub fn accumulate_nd<T: HigherOrderScalar>(
        self,
        a: &NdArray<T>,
        axis: isize,
    ) -> Result<NdArray<T>> {
        if a.ndim() == 0 {
            return Err(NumPyError::InvalidValue("cannot accumulate on a scalar".into()));
        }
        self.validate_comp_dtype(T::DTYPE)?;
        let ax = normalize_axis(axis, a.ndim())?;
        accumulate_homogeneous(self, a, ax)
    }

    /// Typed homogeneous outer application on `NdArray<T>` when output dtype == `T`.
    pub fn outer_nd<T: HigherOrderScalar>(
        self,
        a: &NdArray<T>,
        b: &NdArray<T>,
    ) -> Result<NdArray<T>> {
        self.validate_comp_dtype(T::DTYPE)?;
        let mut shape_a = Vec::with_capacity(a.ndim() + b.ndim());
        shape_a.extend_from_slice(a.shape());
        shape_a.resize(a.ndim() + b.ndim(), 1);
        let mut shape_b = Vec::with_capacity(a.ndim() + b.ndim());
        shape_b.resize(a.ndim(), 1);
        shape_b.extend_from_slice(b.shape());

        let spec_a: Vec<isize> = shape_a.iter().map(|&d| d as isize).collect();
        let spec_b: Vec<isize> = shape_b.iter().map(|&d| d as isize).collect();

        let a_view = if a.is_c_contiguous() {
            a.reshape(&spec_a)?
        } else {
            a.to_c_contiguous().reshape(&spec_a)?
        };
        let b_view = if b.is_c_contiguous() {
            b.reshape(&spec_b)?
        } else {
            b.to_c_contiguous().reshape(&spec_b)?
        };
        if self.can_error_per_element() {
            crate::exec::try_eval_binary(&a_view, &b_view, |x, y| T::ho_bin_try(self, x, y))
        } else {
            crate::exec::eval_binary(&a_view, &b_view, |x, y| T::ho_bin_fast(self, x, y))
        }
    }

    /// Typed unbuffered `.at` on `NdArray<T>` and `&NdArray<T>`.
    pub fn at_nd<'a, T: HigherOrderScalar>(
        self,
        a: &mut NdArray<T>,
        indices: impl Into<AtIndex<'a>>,
        b: &NdArray<T>,
    ) -> Result<()> {
        let idx = indices.into();
        let plan = resolve_at_plan(a.shape(), &idx)?;
        let (comp_in_dt, comp_out_dt) = self.at_comp_pair(T::DTYPE, T::DTYPE)?;

        let b_check = broadcast_shapes(&plan.bcast_shape, b.shape())?;
        if b_check != plan.bcast_shape {
            return Err(NumPyError::ShapeMismatch(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                plan.bcast_shape,
                b.shape()
            )));
        }
        let b_bcast = b.broadcast_to(&plan.bcast_shape)?;
        let trailing_shape = a.shape()[plan.consumed_axes..].to_vec();
        let trailing_len = shape_size(&trailing_shape)?;
        if plan.coords.is_empty() || trailing_len == 0 {
            return Ok(());
        }

        let ndim = a.ndim();
        let mut full_coord = vec![0isize; ndim];
        let mut b_coord = vec![0isize; plan.bcast_shape.len()];
        let idx_prefix_len = plan.idx_bcast_shape.len();

        for (i_idx, prefix) in plan.coords.iter().enumerate() {
            for (k, &c) in prefix.iter().enumerate() {
                full_coord[k] = c;
            }
            if idx_prefix_len > 0 {
                linear_to_isize_coords(i_idx, &plan.idx_bcast_shape, &mut b_coord[..idx_prefix_len]);
            }
            for t_idx in 0..trailing_len {
                if !trailing_shape.is_empty() {
                    linear_to_isize_coords(
                        t_idx,
                        &trailing_shape,
                        &mut full_coord[plan.consumed_axes..],
                    );
                    b_coord[idx_prefix_len..].copy_from_slice(&full_coord[plan.consumed_axes..]);
                }
                let a_cur = a.get(&full_coord)?;
                let b_typed = b_bcast.get(&b_coord)?;
                let out_val = if comp_in_dt == T::DTYPE && comp_out_dt == T::DTYPE {
                    T::ho_bin_try(self, a_cur, b_typed)?
                } else {
                    let b_val = b_typed.to_scalar_value();
                    let a_comp = a_cur.to_scalar_value().cast_to(comp_in_dt);
                    let res = if comp_out_dt == DType::Bool
                        && matches!(
                            self,
                            Self::Equal
                                | Self::NotEqual
                                | Self::Less
                                | Self::LessEqual
                                | Self::Greater
                                | Self::GreaterEqual
                        ) {
                        ScalarValue::Bool(apply_cmp_scalar(
                            self,
                            comp_in_dt,
                            a_comp,
                            b_val.cast_to(comp_in_dt),
                        )?)
                    } else {
                        apply_binary_scalar(self, comp_in_dt, a_comp, b_val.cast_to(comp_in_dt))?
                    };
                    T::from_scalar_value(res.cast_to(T::DTYPE))?
                };
                a.set(&full_coord, out_val)?;
            }
        }
        Ok(())
    }
}

impl UnaryOp {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Negative => "negative",
            Self::Positive => "positive",
            Self::Abs => "abs",
            Self::Fabs => "fabs",
            Self::Square => "square",
            Self::Reciprocal => "reciprocal",
            Self::Sqrt => "sqrt",
            Self::Cbrt => "cbrt",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Expm1 => "expm1",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Arcsin => "arcsin",
            Self::Arccos => "arccos",
            Self::Arctan => "arctan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Arcsinh => "arcsinh",
            Self::Arccosh => "arccosh",
            Self::Arctanh => "arctanh",
            Self::Deg2Rad => "deg2rad",
            Self::Rad2Deg => "rad2deg",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Rint => "rint",
            Self::Sign => "sign",
            Self::Spacing => "spacing",
            Self::Real => "real",
            Self::Imag => "imag",
            Self::Conj => "conj",
            Self::Invert => "invert",
            Self::LogicalNot => "logical_not",
            Self::Sinc => "sinc",
            Self::I0 => "i0",
        }
    }

    fn comp_target(self, dt: DType) -> Result<DType> {
        match self {
            Self::Negative => {
                if dt == DType::Bool {
                    Err(dtype_err("negative", "boolean inputs"))
                } else {
                    Ok(dt)
                }
            }
            Self::Positive => {
                if dt == DType::Bool {
                    Err(dtype_err("positive", "boolean inputs"))
                } else {
                    Ok(dt)
                }
            }
            Self::Abs => Ok(dt),
            Self::Fabs => {
                if dt.is_complex() {
                    Err(dtype_err("fabs", "complex inputs"))
                } else {
                    Ok(real_float_target(dt))
                }
            }
            Self::Square | Self::Reciprocal | Self::Conj => {
                Ok(if dt == DType::Bool { DType::I8 } else { dt })
            }
            Self::Floor | Self::Ceil | Self::Trunc => {
                if dt.is_complex() || dt == DType::Bool {
                    Err(dtype_err(self.name(), "bool or complex inputs"))
                } else {
                    Ok(dt)
                }
            }
            Self::Sign => {
                if dt == DType::Bool {
                    Err(dtype_err("sign", "boolean inputs"))
                } else {
                    Ok(dt)
                }
            }
            Self::Sqrt
            | Self::Exp
            | Self::Exp2
            | Self::Expm1
            | Self::Log
            | Self::Log2
            | Self::Log10
            | Self::Log1p
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Arcsin
            | Self::Arccos
            | Self::Arctan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Arcsinh
            | Self::Arccosh
            | Self::Arctanh
            | Self::Rint => Ok(float_or_complex_target(dt)),
            Self::Sinc => Ok(true_div_target(dt)),
            Self::I0 => {
                if dt.is_complex() {
                    Err(dtype_err("i0", "complex inputs"))
                } else {
                    Ok(true_div_target(dt))
                }
            }
            Self::Cbrt | Self::Deg2Rad | Self::Rad2Deg | Self::Spacing => {
                if dt.is_complex() {
                    Err(dtype_err(self.name(), "complex inputs"))
                } else {
                    Ok(real_float_target(dt))
                }
            }
            Self::Real | Self::Imag => Ok(dt),
            Self::Invert => {
                if dt == DType::Bool || dt.is_integer() {
                    Ok(dt)
                } else {
                    Err(dtype_err("invert", "requires integer or boolean inputs"))
                }
            }
            Self::LogicalNot => Ok(DType::Bool),
        }
    }

    /// Perform unbuffered in-place `a[indices] = op(a[indices])`.
    pub fn at<'a>(self, a: &mut DynArray, indices: impl Into<AtIndex<'a>>) -> Result<()> {
        let idx = indices.into();
        let comp_dt = self.comp_target(a.dtype())?;
        let plan = resolve_at_plan(a.shape(), &idx)?;
        let trailing_shape = a.shape()[plan.consumed_axes..].to_vec();
        let trailing_len = shape_size(&trailing_shape)?;
        if plan.coords.is_empty() || trailing_len == 0 {
            return Ok(());
        }
        let a_dtype = a.dtype();
        let mut full_coord = vec![0isize; a.ndim()];
        for prefix in &plan.coords {
            for (k, &c) in prefix.iter().enumerate() {
                full_coord[k] = c;
            }
            for t_idx in 0..trailing_len {
                if !trailing_shape.is_empty() {
                    linear_to_isize_coords(
                        t_idx,
                        &trailing_shape,
                        &mut full_coord[plan.consumed_axes..],
                    );
                }
                let cur = dyn_get_scalar(a, &full_coord)?;
                let res = apply_unary_scalar(self, comp_dt, cur.cast_to(comp_dt))?;
                dyn_set_scalar(a, &full_coord, res.cast_to(a_dtype))?;
            }
        }
        Ok(())
    }

    /// Typed unbuffered in-place unary `.at` on `NdArray<T>`.
    pub fn at_nd<'a, T: HigherOrderScalar>(
        self,
        a: &mut NdArray<T>,
        indices: impl Into<AtIndex<'a>>,
    ) -> Result<()> {
        let idx = indices.into();
        let comp_dt = self.comp_target(T::DTYPE)?;
        let plan = resolve_at_plan(a.shape(), &idx)?;
        let trailing_shape = a.shape()[plan.consumed_axes..].to_vec();
        let trailing_len = shape_size(&trailing_shape)?;
        if plan.coords.is_empty() || trailing_len == 0 {
            return Ok(());
        }
        let mut full_coord = vec![0isize; a.ndim()];
        for prefix in &plan.coords {
            for (k, &c) in prefix.iter().enumerate() {
                full_coord[k] = c;
            }
            for t_idx in 0..trailing_len {
                if !trailing_shape.is_empty() {
                    linear_to_isize_coords(
                        t_idx,
                        &trailing_shape,
                        &mut full_coord[plan.consumed_axes..],
                    );
                }
                let cur = a.get(&full_coord)?;
                let out_val = if comp_dt == T::DTYPE {
                    T::ho_unary(self, cur)?
                } else {
                    let res = apply_unary_scalar(self, comp_dt, cur.to_scalar_value().cast_to(comp_dt))?;
                    T::from_scalar_value(res.cast_to(T::DTYPE))?
                };
                a.set(&full_coord, out_val)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Free functions (top-level convenience wrappers)
// ============================================================================

/// Reduce `a` over `axis` (`None` = all axes) with optional dtype override, `keepdims`, and `initial`.
#[inline]
pub fn reduce(
    op: BinaryOp,
    a: &DynArray,
    axis: Option<isize>,
    dtype: Option<DType>,
    keepdims: bool,
    initial: Option<ScalarValue>,
) -> Result<DynArray> {
    op.reduce(a, axis, dtype, keepdims, initial)
}

/// Reduce `a` over multiple `axes`.
#[inline]
pub fn reduce_axes(
    op: BinaryOp,
    a: &DynArray,
    axes: &[isize],
    dtype: Option<DType>,
    keepdims: bool,
    initial: Option<ScalarValue>,
) -> Result<DynArray> {
    op.reduce_axes(a, axes, dtype, keepdims, initial)
}

/// Cumulative reduction of `a` along `axis`.
#[inline]
pub fn accumulate(
    op: BinaryOp,
    a: &DynArray,
    axis: isize,
    dtype: Option<DType>,
) -> Result<DynArray> {
    op.accumulate(a, axis, dtype)
}

/// Outer binary ufunc application.
#[inline]
pub fn outer(op: BinaryOp, a: &DynArray, b: &DynArray) -> Result<DynArray> {
    op.outer(a, b)
}

/// Unbuffered in-place binary ufunc application `a[indices] op= b`.
#[inline]
pub fn at<'a>(
    op: BinaryOp,
    a: &mut DynArray,
    indices: impl Into<AtIndex<'a>>,
    b: &DynArray,
) -> Result<()> {
    op.at(a, indices, b)
}

/// Unbuffered in-place unary ufunc application `a[indices] = op(a[indices])`.
#[inline]
pub fn unary_at<'a>(
    op: UnaryOp,
    a: &mut DynArray,
    indices: impl Into<AtIndex<'a>>,
) -> Result<()> {
    op.at(a, indices)
}

/// Zero-cost generic closure reduction on `NdArray<T>`.
pub fn reduce_with<T: Scalar>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    identity: Option<T>,
    op: impl Fn(T, T) -> T + Sync,
) -> Result<NdArray<T>> {
    let axes: Vec<usize> = match axis {
        None => (0..a.ndim()).collect(),
        Some(ax) => vec![normalize_axis(ax, a.ndim())?],
    };
    reduce_closure_nd(a, &axes, keepdims, identity, identity, true, &op)
}

/// Zero-cost generic closure accumulation on `NdArray<T>`.
pub fn accumulate_with<T: Scalar>(
    a: &NdArray<T>,
    axis: isize,
    op: impl Fn(T, T) -> T + Sync,
) -> Result<NdArray<T>> {
    if a.ndim() == 0 {
        return Err(NumPyError::InvalidValue("cannot accumulate on a scalar".into()));
    }
    let ax = normalize_axis(axis, a.ndim())?;
    accumulate_closure_nd(a, ax, &op)
}

/// Zero-cost generic closure outer application on `NdArray<A>` and `NdArray<B>`.
pub fn outer_with<A: Scalar, B: Scalar, Out: Scalar>(
    a: &NdArray<A>,
    b: &NdArray<B>,
    op: impl Fn(A, B) -> Out + Sync,
) -> Result<NdArray<Out>> {
    let mut shape_a = Vec::with_capacity(a.ndim() + b.ndim());
    shape_a.extend_from_slice(a.shape());
    shape_a.resize(a.ndim() + b.ndim(), 1);
    let mut shape_b = Vec::with_capacity(a.ndim() + b.ndim());
    shape_b.resize(a.ndim(), 1);
    shape_b.extend_from_slice(b.shape());

    let spec_a: Vec<isize> = shape_a.iter().map(|&d| d as isize).collect();
    let spec_b: Vec<isize> = shape_b.iter().map(|&d| d as isize).collect();

    let a_view = if a.is_c_contiguous() {
        a.reshape(&spec_a)?
    } else {
        a.to_c_contiguous().reshape(&spec_a)?
    };
    let b_view = if b.is_c_contiguous() {
        b.reshape(&spec_b)?
    } else {
        b.to_c_contiguous().reshape(&spec_b)?
    };
    crate::exec::eval_binary(&a_view, &b_view, op)
}

/// Zero-cost generic unbuffered binary `.at` on `NdArray<T>`.
pub fn at_with<'a, T: Scalar, B: HigherOrderScalar>(
    a: &mut NdArray<T>,
    indices: impl Into<AtIndex<'a>>,
    b: &NdArray<B>,
    op: impl Fn(T, B) -> T,
) -> Result<()> {
    let idx = indices.into();
    let plan = resolve_at_plan(a.shape(), &idx)?;
    let b_check = broadcast_shapes(&plan.bcast_shape, b.shape())?;
    if b_check != plan.bcast_shape {
        return Err(NumPyError::ShapeMismatch(format!(
            "operands could not be broadcast together with shapes {:?} {:?}",
            plan.bcast_shape,
            b.shape()
        )));
    }
    let b_bcast = b.broadcast_to(&plan.bcast_shape)?;
    let trailing_shape = a.shape()[plan.consumed_axes..].to_vec();
    let trailing_len = shape_size(&trailing_shape)?;
    if plan.coords.is_empty() || trailing_len == 0 {
        return Ok(());
    }
    let mut full_coord = vec![0isize; a.ndim()];
    let mut b_coord = vec![0isize; plan.bcast_shape.len()];
    let idx_prefix_len = plan.idx_bcast_shape.len();
    for (i_idx, prefix) in plan.coords.iter().enumerate() {
        for (k, &c) in prefix.iter().enumerate() {
            full_coord[k] = c;
        }
        if idx_prefix_len > 0 {
            linear_to_isize_coords(i_idx, &plan.idx_bcast_shape, &mut b_coord[..idx_prefix_len]);
        }
        for t_idx in 0..trailing_len {
            if !trailing_shape.is_empty() {
                linear_to_isize_coords(
                    t_idx,
                    &trailing_shape,
                    &mut full_coord[plan.consumed_axes..],
                );
                b_coord[idx_prefix_len..].copy_from_slice(&full_coord[plan.consumed_axes..]);
            }
            let cur = a.get(&full_coord)?;
            let b_val = b_bcast.get(&b_coord)?;
            a.set(&full_coord, op(cur, b_val))?;
        }
    }
    Ok(())
}

/// Zero-cost generic unbuffered unary `.at` on `NdArray<T>`.
pub fn unary_at_with<'a, T: Scalar>(
    a: &mut NdArray<T>,
    indices: impl Into<AtIndex<'a>>,
    op: impl Fn(T) -> T,
) -> Result<()> {
    let idx = indices.into();
    let plan = resolve_at_plan(a.shape(), &idx)?;
    let trailing_shape = a.shape()[plan.consumed_axes..].to_vec();
    let trailing_len = shape_size(&trailing_shape)?;
    if plan.coords.is_empty() || trailing_len == 0 {
        return Ok(());
    }
    let mut full_coord = vec![0isize; a.ndim()];
    for prefix in &plan.coords {
        for (k, &c) in prefix.iter().enumerate() {
            full_coord[k] = c;
        }
        for t_idx in 0..trailing_len {
            if !trailing_shape.is_empty() {
                linear_to_isize_coords(
                    t_idx,
                    &trailing_shape,
                    &mut full_coord[plan.consumed_axes..],
                );
            }
            let cur = a.get(&full_coord)?;
            a.set(&full_coord, op(cur))?;
        }
    }
    Ok(())
}

// ============================================================================
// Internal promotion / dispatch helpers
// ============================================================================

#[inline]
fn real_float_target(dt: DType) -> DType {
    match dt {
        DType::Bool | DType::I8 | DType::U8 | DType::F16 => DType::F16,
        DType::I16 | DType::U16 | DType::F32 => DType::F32,
        _ => DType::F64,
    }
}

#[inline]
fn float_or_complex_target(dt: DType) -> DType {
    match dt {
        DType::C64 | DType::C128 => dt,
        other => real_float_target(other),
    }
}

#[inline]
fn true_div_target(dt: DType) -> DType {
    if dt.is_inexact() {
        dt
    } else {
        DType::F64
    }
}

fn raw_identity(op: BinaryOp, dt: DType) -> Option<ScalarValue> {
    match op {
        BinaryOp::Add | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor | BinaryOp::Gcd => {
            Some(ScalarValue::I8(0).cast_to(dt))
        }
        BinaryOp::Multiply => Some(ScalarValue::I8(1).cast_to(dt)),
        BinaryOp::Logaddexp | BinaryOp::Logaddexp2 => match dt {
            DType::F16 => Some(ScalarValue::F16(f16::NEG_INFINITY)),
            DType::F32 => Some(ScalarValue::F32(f32::NEG_INFINITY)),
            DType::F64 => Some(ScalarValue::F64(f64::NEG_INFINITY)),
            _ => None,
        },
        BinaryOp::Hypot => Some(ScalarValue::I8(0).cast_to(dt)),
        BinaryOp::BitwiseAnd => match dt {
            DType::Bool => Some(ScalarValue::Bool(true)),
            DType::I8 => Some(ScalarValue::I8(-1)),
            DType::I16 => Some(ScalarValue::I16(-1)),
            DType::I32 => Some(ScalarValue::I32(-1)),
            DType::I64 => Some(ScalarValue::I64(-1)),
            DType::U8 => Some(ScalarValue::U8(u8::MAX)),
            DType::U16 => Some(ScalarValue::U16(u16::MAX)),
            DType::U32 => Some(ScalarValue::U32(u32::MAX)),
            DType::U64 => Some(ScalarValue::U64(u64::MAX)),
            _ => None,
        },
        BinaryOp::LogicalAnd => Some(ScalarValue::Bool(true)),
        BinaryOp::LogicalOr | BinaryOp::LogicalXor => Some(ScalarValue::Bool(false)),
        _ => None,
    }
}

fn eval_dyn_binary(op: BinaryOp, a: &DynArray, b: &DynArray) -> Result<DynArray> {
    match op {
        BinaryOp::Add => dyn_ufunc::add(a, b),
        BinaryOp::Subtract => dyn_ufunc::subtract(a, b),
        BinaryOp::Multiply => dyn_ufunc::multiply(a, b),
        BinaryOp::TrueDivide => dyn_ufunc::true_divide(a, b),
        BinaryOp::FloorDivide => dyn_ufunc::floor_divide(a, b),
        BinaryOp::Remainder => dyn_ufunc::remainder(a, b),
        BinaryOp::Fmod => dyn_ufunc::fmod(a, b),
        BinaryOp::Power => dyn_ufunc::power(a, b),
        BinaryOp::FloatPower => dyn_ufunc::float_power(a, b),
        BinaryOp::Logaddexp => dyn_ufunc::logaddexp(a, b),
        BinaryOp::Logaddexp2 => dyn_ufunc::logaddexp2(a, b),
        BinaryOp::Arctan2 => dyn_ufunc::arctan2(a, b),
        BinaryOp::Hypot => dyn_ufunc::hypot(a, b),
        BinaryOp::Copysign => dyn_ufunc::copysign(a, b),
        BinaryOp::Nextafter => dyn_ufunc::nextafter(a, b),
        BinaryOp::Heaviside => dyn_ufunc::heaviside(a, b),
        BinaryOp::Maximum => dyn_ufunc::maximum(a, b),
        BinaryOp::Minimum => dyn_ufunc::minimum(a, b),
        BinaryOp::Fmax => dyn_ufunc::fmax(a, b),
        BinaryOp::Fmin => dyn_ufunc::fmin(a, b),
        BinaryOp::Gcd => dyn_ufunc::gcd(a, b),
        BinaryOp::Lcm => dyn_ufunc::lcm(a, b),
        BinaryOp::BitwiseAnd => dyn_ufunc::bitwise_and(a, b),
        BinaryOp::BitwiseOr => dyn_ufunc::bitwise_or(a, b),
        BinaryOp::BitwiseXor => dyn_ufunc::bitwise_xor(a, b),
        BinaryOp::LeftShift => dyn_ufunc::left_shift(a, b),
        BinaryOp::RightShift => dyn_ufunc::right_shift(a, b),
        BinaryOp::Equal => dyn_ufunc::equal(a, b),
        BinaryOp::NotEqual => dyn_ufunc::not_equal(a, b),
        BinaryOp::Less => dyn_ufunc::less(a, b),
        BinaryOp::LessEqual => dyn_ufunc::less_equal(a, b),
        BinaryOp::Greater => dyn_ufunc::greater(a, b),
        BinaryOp::GreaterEqual => dyn_ufunc::greater_equal(a, b),
        BinaryOp::LogicalAnd => dyn_ufunc::logical_and(a, b),
        BinaryOp::LogicalOr => dyn_ufunc::logical_or(a, b),
        BinaryOp::LogicalXor => dyn_ufunc::logical_xor(a, b),
    }
}

// ============================================================================
// Reduction / Accumulation execution
// ============================================================================

fn reduce_dyn_impl(
    op: BinaryOp,
    a: &DynArray,
    axes: &[usize],
    comp_dt: DType,
    keepdims: bool,
    initial: Option<ScalarValue>,
) -> Result<DynArray> {
    if axes.len() > 1 && !op.is_reorderable() {
        return Err(NumPyError::InvalidValue(format!(
            "reduction operation '{}' is not reorderable",
            op.name()
        )));
    }
    let cast_buf = if a.dtype() == comp_dt {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(a.astype(comp_dt, Casting::Unsafe)?)
    };
    dispatch_dyn!(cast_buf.as_ref(), arr => Ok(DynArray::from(reduce_homogeneous(op, arr, axes, keepdims, initial)?)))
}

fn build_reduced_shape(shape: &[usize], axes: &[usize], keepdims: bool) -> (Vec<usize>, Vec<bool>) {
    let ndim = shape.len();
    let mut is_red = vec![false; ndim];
    for &ax in axes {
        is_red[ax] = true;
    }
    let mut out_shape = Vec::with_capacity(ndim);
    for (i, &d) in shape.iter().enumerate() {
        if is_red[i] {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(d);
        }
    }
    (out_shape, is_red)
}

#[inline]
fn linear_to_coords_off(
    mut idx: usize,
    axes: &[usize],
    shape: &[usize],
    strides: &[isize],
) -> isize {
    let mut off = 0isize;
    for &ax in axes.iter().rev() {
        let dim = shape[ax];
        let c = idx % dim;
        idx /= dim;
        off += (c as isize) * strides[ax];
    }
    off
}

fn reduce_homogeneous<T: HigherOrderScalar>(
    op: BinaryOp,
    a: &NdArray<T>,
    axes: &[usize],
    keepdims: bool,
    initial: Option<ScalarValue>,
) -> Result<NdArray<T>> {
    if axes.len() > 1 && !op.is_reorderable() {
        return Err(NumPyError::InvalidValue(format!(
            "reduction operation '{}' is not reorderable",
            op.name()
        )));
    }
    let init_typed: Option<T> = match initial {
        Some(v) => Some(T::from_scalar_value(v.try_cast_initial(T::DTYPE)?)?),
        None => None,
    };
    let ident_typed: Option<T> = match raw_identity(op, T::DTYPE) {
        Some(v) => Some(T::from_scalar_value(v)?),
        None => None,
    };

    if axes.is_empty() {
        let out = match init_typed {
            None => {
                if a.is_c_contiguous() {
                    a.clone()
                } else {
                    a.to_c_contiguous()
                }
            }
            Some(init) => {
                if op.can_error_per_element() {
                    crate::exec::try_eval_unary(a, |x| T::ho_bin_try(op, init, x))?
                } else {
                    crate::exec::eval_unary(a, |x| T::ho_bin_fast(op, init, x))?
                }
            }
        };
        return Ok(out);
    }

    if !op.can_error_per_element() {
        // Fast SIMD whole-array path for contiguous full reduction.
        if axes.len() == a.ndim() && init_typed.is_none() {
            if let Some(slice) = a.as_slice() {
                if !slice.is_empty() && T::supports_simd_reduce(op) {
                    let res = if slice.len() >= PARALLEL_THRESHOLD {
                        let partials: Vec<T> = slice
                            .par_chunks(PARALLEL_CHUNK)
                            .map(|ch| T::simd_reduce(op, ch).unwrap())
                            .collect();
                        T::simd_reduce(op, &partials).unwrap()
                    } else {
                        T::simd_reduce(op, slice).unwrap()
                    };
                    let out_shape = if keepdims {
                        vec![1; a.ndim()]
                    } else {
                        Vec::new()
                    };
                    return NdArray::full(&out_shape, res);
                }
            }
        }

        return reduce_closure_nd(
            a,
            axes,
            keepdims,
            init_typed,
            ident_typed,
            op.is_associative(),
            &|x, y| T::ho_bin_fast(op, x, y),
        );
    }

    // Fallible per-element path (e.g., integer Power with negative exponents).
    try_reduce_nd(a, axes, keepdims, init_typed, ident_typed, &|x, y| {
        T::ho_bin_try(op, x, y)
    })
}

fn reduce_closure_nd<T: Scalar>(
    a: &NdArray<T>,
    axes: &[usize],
    keepdims: bool,
    init: Option<T>,
    ident: Option<T>,
    associative: bool,
    op: &(impl Fn(T, T) -> T + Sync),
) -> Result<NdArray<T>> {
    let shape = a.shape();
    let (out_shape, is_red) = build_reduced_shape(shape, axes, keepdims);
    let out_len = shape_size(&out_shape)?;
    let red_count: usize = axes.iter().map(|&ax| shape[ax]).product();

    if red_count == 0 {
        let fill_val = init.or(ident).ok_or_else(|| {
            NumPyError::InvalidValue("zero-size array to reduction operation which has no identity".into())
        })?;
        return NdArray::full(&out_shape, fill_val);
    }
    if out_len == 0 {
        return empty_like_shape(&out_shape);
    }

    let mut uninit = AlignedBuffer::<T>::new_uninit(out_len);
    let seed = init.or(ident);

    // Whole-array contiguous fast path.
    if axes.len() == shape.len() {
        if let Some(slice) = a.as_slice() {
            let val = if init.is_none() && associative && slice.len() >= PARALLEL_THRESHOLD {
                let partials: Vec<T> = slice
                    .par_chunks(PARALLEL_CHUNK)
                    .map(|ch| {
                        let mut acc = ch[0];
                        for &x in &ch[1..] {
                            acc = op(acc, x);
                        }
                        acc
                    })
                    .collect();
                let (mut acc, rest) = match seed {
                    Some(v) => (v, partials.as_slice()),
                    None => (partials[0], &partials[1..]),
                };
                for &x in rest {
                    acc = op(acc, x);
                }
                acc
            } else {
                let (mut acc, rest) = match seed {
                    Some(v) => (v, slice),
                    None => (slice[0], &slice[1..]),
                };
                for &x in rest {
                    acc = op(acc, x);
                }
                acc
            };
            uninit.as_mut_slice()[0].write(val);
            let buf = unsafe { uninit.assume_init() };
            return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
        }
    }

    // Single-axis C-contiguous fast path (innermost or outer/middle slab).
    if axes.len() == 1 {
        if let Some(slice) = a.as_slice() {
            let ax = axes[0];
            let outer: usize = shape[..ax].iter().product();
            let k = shape[ax];
            let inner: usize = shape[ax + 1..].iter().product();
            let total = slice.len();
            let out_slice = uninit.as_mut_slice();

            if inner == 1 {
                let do_row = |o: usize| -> T {
                    let row = &slice[o * k..(o + 1) * k];
                    let (mut acc, rest) = match seed {
                        Some(v) => (v, row),
                        None => (row[0], &row[1..]),
                    };
                    for &x in rest {
                        acc = op(acc, x);
                    }
                    acc
                };
                if total >= PARALLEL_THRESHOLD && outer >= 2 {
                    out_slice
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(o, slot)| {
                            slot.write(do_row(o));
                        });
                } else {
                    for (o, slot) in out_slice.iter_mut().enumerate() {
                        slot.write(do_row(o));
                    }
                }
            } else {
                let do_outer = |(o, out_row): (usize, &mut [std::mem::MaybeUninit<T>])| {
                    let row0 = &slice[(o * k) * inner..(o * k + 1) * inner];
                    match seed {
                        Some(v) => {
                            for j in 0..inner {
                                out_row[j].write(op(v, row0[j]));
                            }
                        }
                        None => {
                            for j in 0..inner {
                                out_row[j].write(row0[j]);
                            }
                        }
                    }
                    // out_row is now fully initialized for 0..inner.
                    let init_row = unsafe {
                        std::slice::from_raw_parts_mut(
                            out_row.as_mut_ptr() as *mut T,
                            inner,
                        )
                    };
                    for r in 1..k {
                        let row_r = &slice[(o * k + r) * inner..(o * k + r + 1) * inner];
                        for j in 0..inner {
                            init_row[j] = op(init_row[j], row_r[j]);
                        }
                    }
                };
                if total >= PARALLEL_THRESHOLD && outer >= 2 {
                    out_slice.par_chunks_mut(inner).enumerate().for_each(do_outer);
                } else {
                    out_slice.chunks_mut(inner).enumerate().for_each(do_outer);
                }
            }
            let buf = unsafe { uninit.assume_init() };
            return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
        }
    }

    // General strided / multi-axis path.
    let kept_axes: Vec<usize> = (0..shape.len()).filter(|&i| !is_red[i]).collect();
    let mut red_axes: Vec<usize> = axes.to_vec();
    red_axes.sort_unstable();
    let strides = a.strides();
    let ptr = unsafe { SendPtr(a.storage_ptr().add(a.offset())) };
    let total_work = out_len.saturating_mul(red_count);
    let out_slice = uninit.as_mut_slice();

    let compute_elem = |out_idx: usize| -> T {
        let base_off = linear_to_coords_off(out_idx, &kept_axes, shape, strides);
        let first_off = base_off;
        let (mut acc, start_r) = match seed {
            Some(v) => (v, 0usize),
            None => (unsafe { ptr.add_offset(first_off) }, 1usize),
        };
        for r in start_r..red_count {
            let r_off = linear_to_coords_off(r, &red_axes, shape, strides);
            let x = unsafe { ptr.add_offset(base_off + r_off) };
            acc = op(acc, x);
        }
        acc
    };

    if total_work >= PARALLEL_THRESHOLD && out_len >= 2 {
        out_slice.par_iter_mut().enumerate().for_each(|(i, slot)| {
            slot.write(compute_elem(i));
        });
    } else {
        for (i, slot) in out_slice.iter_mut().enumerate() {
            slot.write(compute_elem(i));
        }
    }
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &out_shape, Order::C)
}

fn try_reduce_nd<T: Scalar>(
    a: &NdArray<T>,
    axes: &[usize],
    keepdims: bool,
    init: Option<T>,
    ident: Option<T>,
    op: &(impl Fn(T, T) -> Result<T> + Sync),
) -> Result<NdArray<T>> {
    let shape = a.shape();
    let (out_shape, is_red) = build_reduced_shape(shape, axes, keepdims);
    let out_len = shape_size(&out_shape)?;
    let red_count: usize = axes.iter().map(|&ax| shape[ax]).product();

    if red_count == 0 {
        let fill_val = init.or(ident).ok_or_else(|| {
            NumPyError::InvalidValue("zero-size array to reduction operation which has no identity".into())
        })?;
        return NdArray::full(&out_shape, fill_val);
    }
    if out_len == 0 {
        return empty_like_shape(&out_shape);
    }

    let kept_axes: Vec<usize> = (0..shape.len()).filter(|&i| !is_red[i]).collect();
    let mut red_axes: Vec<usize> = axes.to_vec();
    red_axes.sort_unstable();
    let strides = a.strides();
    let ptr = unsafe { SendPtr(a.storage_ptr().add(a.offset())) };
    let mut uninit = AlignedBuffer::<T>::new_uninit(out_len);
    let out_slice = uninit.as_mut_slice();
    let seed = init.or(ident);

    for (out_idx, slot) in out_slice.iter_mut().enumerate() {
        let base_off = linear_to_coords_off(out_idx, &kept_axes, shape, strides);
        let (mut acc, start_r) = match seed {
            Some(v) => (v, 0usize),
            None => (unsafe { ptr.add_offset(base_off) }, 1usize),
        };
        for r in start_r..red_count {
            let r_off = linear_to_coords_off(r, &red_axes, shape, strides);
            let x = unsafe { ptr.add_offset(base_off + r_off) };
            acc = op(acc, x)?;
        }
        slot.write(acc);
    }
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &out_shape, Order::C)
}

fn accumulate_homogeneous<T: HigherOrderScalar>(
    op: BinaryOp,
    a: &NdArray<T>,
    axis: usize,
) -> Result<NdArray<T>> {
    if !op.can_error_per_element() {
        return accumulate_closure_nd(a, axis, &|x, y| T::ho_bin_fast(op, x, y));
    }
    let shape = a.shape();
    let len = a.len();
    if len == 0 {
        return empty_like_shape(shape);
    }
    let other_axes: Vec<usize> = (0..shape.len()).filter(|&i| i != axis).collect();
    let outer_count: usize = other_axes.iter().map(|&ax| shape[ax]).product();
    let k = shape[axis];
    let in_strides = a.strides();
    let out_c_strides = numpy_types::compute_c_strides(shape);
    let ptr = unsafe { SendPtr(a.storage_ptr().add(a.offset())) };
    let mut buf = AlignedBuffer::<T>::new_zeroed(len);
    let out_slice = buf.as_mut_slice();
    for o in 0..outer_count {
        let in_base = linear_to_coords_off(o, &other_axes, shape, in_strides);
        let out_base = linear_to_coords_off(o, &other_axes, shape, &out_c_strides) as usize;
        let mut acc = unsafe { ptr.add_offset(in_base) };
        out_slice[out_base] = acc;
        for r in 1..k {
            let x = unsafe { ptr.add_offset(in_base + (r as isize) * in_strides[axis]) };
            acc = T::ho_bin_try(op, acc, x)?;
            out_slice[out_base + r * (out_c_strides[axis] as usize)] = acc;
        }
    }
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

fn accumulate_closure_nd<T: Scalar>(
    a: &NdArray<T>,
    axis: usize,
    op: &(impl Fn(T, T) -> T + Sync),
) -> Result<NdArray<T>> {
    let shape = a.shape();
    let len = a.len();
    if len == 0 {
        return empty_like_shape(shape);
    }

    let mut uninit = AlignedBuffer::<T>::new_uninit(len);
    if let Some(slice) = a.as_slice() {
        let outer: usize = shape[..axis].iter().product();
        let k = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let out_slice = uninit.as_mut_slice();

        if inner == 1 {
            let do_lane = |(o, out_row): (usize, &mut [std::mem::MaybeUninit<T>])| {
                let row = &slice[o * k..(o + 1) * k];
                let mut acc = row[0];
                out_row[0].write(acc);
                for r in 1..k {
                    acc = op(acc, row[r]);
                    out_row[r].write(acc);
                }
            };
            if len >= PARALLEL_THRESHOLD && outer >= 2 {
                out_slice.par_chunks_mut(k).enumerate().for_each(do_lane);
            } else {
                out_slice.chunks_mut(k).enumerate().for_each(do_lane);
            }
        } else {
            let block = k * inner;
            let do_block = |(o, out_blk): (usize, &mut [std::mem::MaybeUninit<T>])| {
                let src_blk = &slice[o * block..(o + 1) * block];
                for j in 0..inner {
                    out_blk[j].write(src_blk[j]);
                }
                for r in 1..k {
                    let prev_start = (r - 1) * inner;
                    let cur_start = r * inner;
                    for j in 0..inner {
                        let prev = unsafe { out_blk[prev_start + j].assume_init() };
                        out_blk[cur_start + j].write(op(prev, src_blk[cur_start + j]));
                    }
                }
            };
            if len >= PARALLEL_THRESHOLD && outer >= 2 {
                out_slice.par_chunks_mut(block).enumerate().for_each(do_block);
            } else {
                out_slice.chunks_mut(block).enumerate().for_each(do_block);
            }
        }
        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, shape, Order::C);
    }

    // General strided fallback.
    let other_axes: Vec<usize> = (0..shape.len()).filter(|&i| i != axis).collect();
    let outer_count: usize = other_axes.iter().map(|&ax| shape[ax]).product();
    let k = shape[axis];
    let in_strides = a.strides();
    let out_c_strides = numpy_types::compute_c_strides(shape);
    let ptr = unsafe { SendPtr(a.storage_ptr().add(a.offset())) };
    let mut buf = AlignedBuffer::<T>::new_zeroed(len);
    let out_slice = buf.as_mut_slice();
    for o in 0..outer_count {
        let in_base = linear_to_coords_off(o, &other_axes, shape, in_strides);
        let out_base = linear_to_coords_off(o, &other_axes, shape, &out_c_strides) as usize;
        let mut acc = unsafe { ptr.add_offset(in_base) };
        out_slice[out_base] = acc;
        for r in 1..k {
            let x = unsafe { ptr.add_offset(in_base + (r as isize) * in_strides[axis]) };
            acc = op(acc, x);
            out_slice[out_base + r * (out_c_strides[axis] as usize)] = acc;
        }
    }
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

// ============================================================================
// `.at` planning & element helpers
// ============================================================================

struct AtPlan {
    coords: Vec<Vec<isize>>,
    idx_bcast_shape: Vec<usize>,
    bcast_shape: Vec<usize>,
    consumed_axes: usize,
}

#[inline]
fn linear_to_isize_coords(mut idx: usize, shape: &[usize], out: &mut [isize]) {
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        out[d] = (idx % dim) as isize;
        idx /= dim;
    }
}

fn dyn_to_i64_array(arr: &DynArray) -> Result<NdArray<i64>> {
    fn conv<T: Scalar>(src: &NdArray<T>, f: impl Fn(T) -> i64 + Sync) -> Result<NdArray<i64>> {
        crate::exec::eval_unary(src, f)
    }
    match arr {
        DynArray::I8(x) => conv(x, |v| v as i64),
        DynArray::I16(x) => conv(x, |v| v as i64),
        DynArray::I32(x) => conv(x, |v| v as i64),
        DynArray::I64(x) => Ok(x.clone()),
        DynArray::U8(x) => conv(x, |v| v as i64),
        DynArray::U16(x) => conv(x, |v| v as i64),
        DynArray::U32(x) => conv(x, |v| v as i64),
        DynArray::U64(x) => {
            crate::exec::try_eval_unary(x, |v| {
                i64::try_from(v).map_err(|_| {
                    NumPyError::IndexOutOfBounds(format!(
                        "index {v} is out of bounds for 64-bit signed indexing"
                    ))
                })
            })
        }
        _ => Err(NumPyError::DTypeError(
            "arrays used as integer indices must be of integer type".into(),
        )),
    }
}

fn collect_bool_mask_coords(mask: &NdArray<bool>, axis_start: usize, target_shape: &[usize]) -> Result<(Vec<Vec<isize>>, usize)> {
    let m_ndim = mask.ndim();
    if axis_start + m_ndim > target_shape.len() {
        return Err(NumPyError::InvalidValue(format!(
            "boolean index did not match indexed array along dimension {axis_start}"
        )));
    }
    for d in 0..m_ndim {
        if mask.shape()[d] != target_shape[axis_start + d] {
            return Err(NumPyError::ShapeMismatch(format!(
                "boolean index did not match indexed array along axis {}; shape is {:?} but corresponding boolean dimension has shape {:?}",
                axis_start + d,
                &target_shape[axis_start..axis_start + m_ndim],
                mask.shape()
            )));
        }
    }
    let mut coords = Vec::new();
    let total = mask.len();
    let mut coord_buf = vec![0isize; m_ndim];
    let m_shape = mask.shape();
    for i in 0..total {
        linear_to_isize_coords(i, m_shape, &mut coord_buf);
        if mask.get(&coord_buf)? {
            coords.push(coord_buf.clone());
        }
    }
    Ok((coords, m_ndim))
}

fn resolve_at_dyn_array(target_shape: &[usize], arr: &DynArray) -> Result<AtPlan> {
    let ndim = target_shape.len();
    if let DynArray::Bool(mask) = arr {
        let (coords, consumed) = collect_bool_mask_coords(mask, 0, target_shape)?;
        let idx_bcast_shape = vec![coords.len()];
        let mut bcast_shape = idx_bcast_shape.clone();
        bcast_shape.extend_from_slice(&target_shape[consumed..]);
        return Ok(AtPlan {
            coords,
            idx_bcast_shape,
            bcast_shape,
            consumed_axes: consumed,
        });
    }
    if ndim == 0 {
        return Err(NumPyError::IndexOutOfBounds(
            "too many indices for array: array is 0-dimensional".into(),
        ));
    }
    let idx_arr = dyn_to_i64_array(arr)?;
    let idx_shape = idx_arr.shape().to_vec();
    let dim = target_shape[0];
    let total = idx_arr.len();
    let mut coords = Vec::with_capacity(total);
    let mut tmp = vec![0isize; idx_shape.len()];
    for i in 0..total {
        if !idx_shape.is_empty() {
            linear_to_isize_coords(i, &idx_shape, &mut tmp);
        }
        let raw = idx_arr.get(&tmp)? as isize;
        let d = dim as isize;
        let norm = if raw < 0 { raw + d } else { raw };
        if norm < 0 || norm >= d {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "index {raw} is out of bounds for axis 0 with size {dim}"
            )));
        }
        coords.push(vec![norm]);
    }
    let mut bcast_shape = idx_shape.clone();
    bcast_shape.extend_from_slice(&target_shape[1..]);
    Ok(AtPlan {
        coords,
        idx_bcast_shape: idx_shape,
        bcast_shape,
        consumed_axes: 1,
    })
}

fn resolve_at_plan(target_shape: &[usize], idx: &AtIndex<'_>) -> Result<AtPlan> {
    let ndim = target_shape.len();
    let normalize_idx = |raw: isize, ax: usize, dim: usize| -> Result<isize> {
        let d = dim as isize;
        let norm = if raw < 0 { raw + d } else { raw };
        if norm < 0 || norm >= d {
            Err(NumPyError::IndexOutOfBounds(format!(
                "index {raw} is out of bounds for axis {ax} with size {dim}"
            )))
        } else {
            Ok(norm)
        }
    };

    match idx {
        AtIndex::SliceIsize(sl) => {
            if ndim == 0 {
                return Err(NumPyError::IndexOutOfBounds(
                    "too many indices for array: array is 0-dimensional".into(),
                ));
            }
            let dim = target_shape[0];
            let mut coords = Vec::with_capacity(sl.len());
            for &raw in *sl {
                coords.push(vec![normalize_idx(raw, 0, dim)?]);
            }
            let idx_bcast_shape = vec![sl.len()];
            let mut bcast_shape = idx_bcast_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[1..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape,
                bcast_shape,
                consumed_axes: 1,
            })
        }
        AtIndex::SliceI64(sl) => {
            if ndim == 0 {
                return Err(NumPyError::IndexOutOfBounds(
                    "too many indices for array: array is 0-dimensional".into(),
                ));
            }
            let dim = target_shape[0];
            let mut coords = Vec::with_capacity(sl.len());
            for &raw in *sl {
                coords.push(vec![normalize_idx(raw as isize, 0, dim)?]);
            }
            let idx_bcast_shape = vec![sl.len()];
            let mut bcast_shape = idx_bcast_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[1..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape,
                bcast_shape,
                consumed_axes: 1,
            })
        }
        AtIndex::SliceI32(sl) => {
            if ndim == 0 {
                return Err(NumPyError::IndexOutOfBounds(
                    "too many indices for array: array is 0-dimensional".into(),
                ));
            }
            let dim = target_shape[0];
            let mut coords = Vec::with_capacity(sl.len());
            for &raw in *sl {
                coords.push(vec![normalize_idx(raw as isize, 0, dim)?]);
            }
            let idx_bcast_shape = vec![sl.len()];
            let mut bcast_shape = idx_bcast_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[1..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape,
                bcast_shape,
                consumed_axes: 1,
            })
        }
        AtIndex::SliceUsize(sl) => {
            if ndim == 0 {
                return Err(NumPyError::IndexOutOfBounds(
                    "too many indices for array: array is 0-dimensional".into(),
                ));
            }
            let dim = target_shape[0];
            let mut coords = Vec::with_capacity(sl.len());
            for &raw in *sl {
                if raw >= dim {
                    return Err(NumPyError::IndexOutOfBounds(format!(
                        "index {raw} is out of bounds for axis 0 with size {dim}"
                    )));
                }
                coords.push(vec![raw as isize]);
            }
            let idx_bcast_shape = vec![sl.len()];
            let mut bcast_shape = idx_bcast_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[1..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape,
                bcast_shape,
                consumed_axes: 1,
            })
        }
        AtIndex::Array(arr) => resolve_at_dyn_array(target_shape, arr),
        AtIndex::OwnedArray(arr) => resolve_at_dyn_array(target_shape, arr),
        AtIndex::NdArrayIsize(arr) => {
            if ndim == 0 {
                return Err(NumPyError::IndexOutOfBounds(
                    "too many indices for array: array is 0-dimensional".into(),
                ));
            }
            let idx_shape = arr.shape().to_vec();
            let dim = target_shape[0];
            let total = arr.len();
            let mut coords = Vec::with_capacity(total);
            let mut tmp = vec![0isize; idx_shape.len()];
            for i in 0..total {
                if !idx_shape.is_empty() {
                    linear_to_isize_coords(i, &idx_shape, &mut tmp);
                }
                let raw = arr.get(&tmp)?;
                coords.push(vec![normalize_idx(raw, 0, dim)?]);
            }
            let mut bcast_shape = idx_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[1..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape: idx_shape,
                bcast_shape,
                consumed_axes: 1,
            })
        }
        AtIndex::NdArrayUsize(arr) => {
            if ndim == 0 {
                return Err(NumPyError::IndexOutOfBounds(
                    "too many indices for array: array is 0-dimensional".into(),
                ));
            }
            let idx_shape = arr.shape().to_vec();
            let dim = target_shape[0];
            let total = arr.len();
            let mut coords = Vec::with_capacity(total);
            let mut tmp = vec![0isize; idx_shape.len()];
            for i in 0..total {
                if !idx_shape.is_empty() {
                    linear_to_isize_coords(i, &idx_shape, &mut tmp);
                }
                let raw = arr.get(&tmp)?;
                if raw >= dim {
                    return Err(NumPyError::IndexOutOfBounds(format!(
                        "index {raw} is out of bounds for axis 0 with size {dim}"
                    )));
                }
                coords.push(vec![raw as isize]);
            }
            let mut bcast_shape = idx_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[1..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape: idx_shape,
                bcast_shape,
                consumed_axes: 1,
            })
        }
        AtIndex::Tuple(items) => {
            if items.is_empty() {
                return Ok(AtPlan {
                    coords: vec![Vec::new()],
                    idx_bcast_shape: Vec::new(),
                    bcast_shape: target_shape.to_vec(),
                    consumed_axes: 0,
                });
            }
            let mut axis_cursor = 0usize;
            let mut expanded_axes: Vec<NdArray<i64>> = Vec::new();
            let mut scalar_bool_shapes: Vec<Vec<usize>> = Vec::new();
            for item in *items {
                if let DynArray::Bool(mask) = item {
                    let (mask_coords, consumed) =
                        collect_bool_mask_coords(mask, axis_cursor, target_shape)?;
                    let n_true = mask_coords.len();
                    for ax_off in 0..consumed {
                        let col: Vec<i64> = mask_coords.iter().map(|c| c[ax_off] as i64).collect();
                        expanded_axes.push(NdArray::from_vec(col));
                    }
                    if consumed == 0 {
                        // 0-D bool scalar contributes a 1-D axis of length 0 (False) or 1 (True)
                        // without consuming a target axis, and participates in tuple broadcasting.
                        scalar_bool_shapes.push(vec![n_true]);
                    }
                    axis_cursor += consumed;
                } else {
                    if axis_cursor >= ndim {
                        return Err(NumPyError::IndexOutOfBounds(format!(
                            "too many indices for array: array is {ndim}-dimensional, but {} were indexed",
                            axis_cursor + 1
                        )));
                    }
                    expanded_axes.push(dyn_to_i64_array(item)?);
                    axis_cursor += 1;
                }
            }
            let mut shapes: Vec<&[usize]> = expanded_axes.iter().map(|x| x.shape()).collect();
            for s in &scalar_bool_shapes {
                shapes.push(s.as_slice());
            }
            let idx_bcast_shape = broadcast_many_shapes(&shapes)?;
            let bcast_arrays: Vec<NdArray<i64>> = expanded_axes
                .iter()
                .map(|x| x.broadcast_to(&idx_bcast_shape))
                .collect::<Result<Vec<_>>>()?;
            let total = shape_size(&idx_bcast_shape)?;
            let mut coords = Vec::with_capacity(total);
            let mut tmp = vec![0isize; idx_bcast_shape.len()];
            for i in 0..total {
                if !idx_bcast_shape.is_empty() {
                    linear_to_isize_coords(i, &idx_bcast_shape, &mut tmp);
                }
                let mut tuple_coord = Vec::with_capacity(axis_cursor);
                for (ax, b_arr) in bcast_arrays.iter().enumerate() {
                    let raw = b_arr.get(&tmp)? as isize;
                    tuple_coord.push(normalize_idx(raw, ax, target_shape[ax])?);
                }
                coords.push(tuple_coord);
            }
            let mut bcast_shape = idx_bcast_shape.clone();
            bcast_shape.extend_from_slice(&target_shape[axis_cursor..]);
            Ok(AtPlan {
                coords,
                idx_bcast_shape,
                bcast_shape,
                consumed_axes: axis_cursor,
            })
        }
    }
}

fn dyn_get_scalar(arr: &DynArray, coord: &[isize]) -> Result<ScalarValue> {
    Ok(match arr {
        DynArray::Bool(x) => ScalarValue::Bool(x.get(coord)?),
        DynArray::I8(x) => ScalarValue::I8(x.get(coord)?),
        DynArray::I16(x) => ScalarValue::I16(x.get(coord)?),
        DynArray::I32(x) => ScalarValue::I32(x.get(coord)?),
        DynArray::I64(x) => ScalarValue::I64(x.get(coord)?),
        DynArray::U8(x) => ScalarValue::U8(x.get(coord)?),
        DynArray::U16(x) => ScalarValue::U16(x.get(coord)?),
        DynArray::U32(x) => ScalarValue::U32(x.get(coord)?),
        DynArray::U64(x) => ScalarValue::U64(x.get(coord)?),
        DynArray::F16(x) => ScalarValue::F16(x.get(coord)?),
        DynArray::F32(x) => ScalarValue::F32(x.get(coord)?),
        DynArray::F64(x) => ScalarValue::F64(x.get(coord)?),
        DynArray::C64(x) => ScalarValue::C64(x.get(coord)?),
        DynArray::C128(x) => ScalarValue::C128(x.get(coord)?),
    })
}

fn dyn_set_scalar(arr: &mut DynArray, coord: &[isize], val: ScalarValue) -> Result<()> {
    match (arr, val) {
        (DynArray::Bool(x), ScalarValue::Bool(v)) => x.set(coord, v),
        (DynArray::I8(x), ScalarValue::I8(v)) => x.set(coord, v),
        (DynArray::I16(x), ScalarValue::I16(v)) => x.set(coord, v),
        (DynArray::I32(x), ScalarValue::I32(v)) => x.set(coord, v),
        (DynArray::I64(x), ScalarValue::I64(v)) => x.set(coord, v),
        (DynArray::U8(x), ScalarValue::U8(v)) => x.set(coord, v),
        (DynArray::U16(x), ScalarValue::U16(v)) => x.set(coord, v),
        (DynArray::U32(x), ScalarValue::U32(v)) => x.set(coord, v),
        (DynArray::U64(x), ScalarValue::U64(v)) => x.set(coord, v),
        (DynArray::F16(x), ScalarValue::F16(v)) => x.set(coord, v),
        (DynArray::F32(x), ScalarValue::F32(v)) => x.set(coord, v),
        (DynArray::F64(x), ScalarValue::F64(v)) => x.set(coord, v),
        (DynArray::C64(x), ScalarValue::C64(v)) => x.set(coord, v),
        (DynArray::C128(x), ScalarValue::C128(v)) => x.set(coord, v),
        _ => unreachable!(),
    }
}

fn apply_binary_scalar(
    op: BinaryOp,
    dt: DType,
    a: ScalarValue,
    b: ScalarValue,
) -> Result<ScalarValue> {
    macro_rules! go {
        ($t:ty, $var:ident) => {{
            let x = <$t>::from_scalar_value(a)?;
            let y = <$t>::from_scalar_value(b)?;
            Ok(ScalarValue::$var(<$t>::ho_bin_try(op, x, y)?))
        }};
    }
    match dt {
        DType::Bool => go!(bool, Bool),
        DType::I8 => go!(i8, I8),
        DType::I16 => go!(i16, I16),
        DType::I32 => go!(i32, I32),
        DType::I64 => go!(i64, I64),
        DType::U8 => go!(u8, U8),
        DType::U16 => go!(u16, U16),
        DType::U32 => go!(u32, U32),
        DType::U64 => go!(u64, U64),
        DType::F16 => go!(f16, F16),
        DType::F32 => go!(f32, F32),
        DType::F64 => go!(f64, F64),
        DType::C64 => go!(Complex32, C64),
        DType::C128 => go!(Complex64, C128),
    }
}

fn apply_cmp_scalar(op: BinaryOp, dt: DType, a: ScalarValue, b: ScalarValue) -> Result<bool> {
    macro_rules! go {
        ($t:ty) => {{
            let x = <$t>::from_scalar_value(a)?;
            let y = <$t>::from_scalar_value(b)?;
            Ok(<$t>::ho_cmp(op, x, y))
        }};
    }
    match dt {
        DType::Bool => go!(bool),
        DType::I8 => go!(i8),
        DType::I16 => go!(i16),
        DType::I32 => go!(i32),
        DType::I64 => go!(i64),
        DType::U8 => go!(u8),
        DType::U16 => go!(u16),
        DType::U32 => go!(u32),
        DType::U64 => go!(u64),
        DType::F16 => go!(f16),
        DType::F32 => go!(f32),
        DType::F64 => go!(f64),
        DType::C64 => go!(Complex32),
        DType::C128 => go!(Complex64),
    }
}

fn apply_unary_scalar(op: UnaryOp, dt: DType, a: ScalarValue) -> Result<ScalarValue> {
    macro_rules! go {
        ($t:ty, $var:ident) => {{
            let x = <$t>::from_scalar_value(a)?;
            Ok(<$t>::ho_unary(op, x)?.to_scalar_value())
        }};
    }
    match op {
        UnaryOp::Abs if dt == DType::C64 => {
            let x = Complex32::from_scalar_value(a)?;
            return Ok(ScalarValue::F32(x.re.hypot(x.im)));
        }
        UnaryOp::Abs if dt == DType::C128 => {
            let x = Complex64::from_scalar_value(a)?;
            return Ok(ScalarValue::F64(x.re.hypot(x.im)));
        }
        UnaryOp::Real if dt == DType::C64 => {
            return Ok(ScalarValue::F32(Complex32::from_scalar_value(a)?.re));
        }
        UnaryOp::Real if dt == DType::C128 => {
            return Ok(ScalarValue::F64(Complex64::from_scalar_value(a)?.re));
        }
        UnaryOp::Imag if dt == DType::C64 => {
            return Ok(ScalarValue::F32(Complex32::from_scalar_value(a)?.im));
        }
        UnaryOp::Imag if dt == DType::C128 => {
            return Ok(ScalarValue::F64(Complex64::from_scalar_value(a)?.im));
        }
        UnaryOp::Imag => {
            return Ok(ScalarValue::I8(0).cast_to(dt));
        }
        UnaryOp::Real => {
            return Ok(a);
        }
        _ => {}
    }
    match dt {
        DType::Bool => go!(bool, Bool),
        DType::I8 => go!(i8, I8),
        DType::I16 => go!(i16, I16),
        DType::I32 => go!(i32, I32),
        DType::I64 => go!(i64, I64),
        DType::U8 => go!(u8, U8),
        DType::U16 => go!(u16, U16),
        DType::U32 => go!(u32, U32),
        DType::U64 => go!(u64, U64),
        DType::F16 => go!(f16, F16),
        DType::F32 => go!(f32, F32),
        DType::F64 => go!(f64, F64),
        DType::C64 => go!(Complex32, C64),
        DType::C128 => go!(Complex64, C128),
    }
}

// ============================================================================
// HigherOrderScalar trait (scalar kernels for all 14 concrete types)
// ============================================================================

#[doc(hidden)]
pub trait HigherOrderScalar: UfuncScalar {
    fn to_scalar_value(self) -> ScalarValue;
    fn from_scalar_value(v: ScalarValue) -> Result<Self>;
    fn ho_bin_fast(op: BinaryOp, a: Self, b: Self) -> Self;
    fn ho_bin_try(op: BinaryOp, a: Self, b: Self) -> Result<Self> {
        Ok(Self::ho_bin_fast(op, a, b))
    }
    fn ho_cmp(op: BinaryOp, a: Self, b: Self) -> bool;
    fn ho_unary(op: UnaryOp, a: Self) -> Result<Self>;
    #[inline]
    fn supports_simd_reduce(_op: BinaryOp) -> bool {
        false
    }
    #[inline]
    fn simd_reduce(_op: BinaryOp, _slice: &[Self]) -> Option<Self> {
        None
    }
}

impl HigherOrderScalar for bool {
    #[inline]
    fn to_scalar_value(self) -> ScalarValue {
        ScalarValue::Bool(self)
    }
    #[inline]
    fn from_scalar_value(v: ScalarValue) -> Result<Self> {
        match v {
            ScalarValue::Bool(b) => Ok(b),
            _ => Err(dtype_err("scalar", "expected bool scalar")),
        }
    }
    #[inline]
    fn ho_bin_fast(op: BinaryOp, a: bool, b: bool) -> bool {
        match op {
            BinaryOp::Add
            | BinaryOp::Maximum
            | BinaryOp::Fmax
            | BinaryOp::BitwiseOr
            | BinaryOp::LogicalOr => a | b,
            BinaryOp::Multiply
            | BinaryOp::Minimum
            | BinaryOp::Fmin
            | BinaryOp::BitwiseAnd
            | BinaryOp::LogicalAnd => a & b,
            BinaryOp::Power => a | !b,
            BinaryOp::BitwiseXor | BinaryOp::LogicalXor | BinaryOp::NotEqual => a ^ b,
            BinaryOp::Equal => a == b,
            BinaryOp::Less => !a & b,
            BinaryOp::LessEqual => !a | b,
            BinaryOp::Greater => a & !b,
            BinaryOp::GreaterEqual => a | !b,
            _ => unreachable!("invalid BinaryOp for bool"),
        }
    }
    #[inline]
    fn ho_cmp(op: BinaryOp, a: bool, b: bool) -> bool {
        match op {
            BinaryOp::Equal => a == b,
            BinaryOp::NotEqual => a != b,
            BinaryOp::Less => !a & b,
            BinaryOp::LessEqual => !a | b,
            BinaryOp::Greater => a & !b,
            BinaryOp::GreaterEqual => a | !b,
            _ => unreachable!(),
        }
    }
    #[inline]
    fn ho_unary(op: UnaryOp, a: bool) -> Result<bool> {
        match op {
            UnaryOp::Abs | UnaryOp::Real => Ok(a),
            UnaryOp::Imag => Ok(false),
            UnaryOp::Invert | UnaryOp::LogicalNot => Ok(!a),
            _ => Err(dtype_err(op.name(), "boolean input")),
        }
    }
}

macro_rules! impl_ho_signed_int {
    (
        $t:ty, $var:ident,
        $fdiv:ident, $frem:ident, $pow_fn:ident, $lsh:ident, $rsh:ident, $gcd:ident, $lcm:ident, $recip:ident,
        $simd_flag:expr, $simd_fn:expr
    ) => {
        impl HigherOrderScalar for $t {
            #[inline]
            fn to_scalar_value(self) -> ScalarValue {
                ScalarValue::$var(self)
            }
            #[inline]
            fn from_scalar_value(v: ScalarValue) -> Result<Self> {
                match v {
                    ScalarValue::$var(x) => Ok(x),
                    _ => Err(dtype_err("scalar", "scalar dtype mismatch")),
                }
            }
            #[inline]
            fn ho_bin_fast(op: BinaryOp, a: Self, b: Self) -> Self {
                match op {
                    BinaryOp::Add => a.wrapping_add(b),
                    BinaryOp::Subtract => a.wrapping_sub(b),
                    BinaryOp::Multiply => a.wrapping_mul(b),
                    BinaryOp::FloorDivide => $fdiv(a, b),
                    BinaryOp::Remainder => $frem(a, b),
                    BinaryOp::Fmod => if b == 0 { 0 } else { a.wrapping_rem(b) },
                    BinaryOp::Maximum | BinaryOp::Fmax => a.max(b),
                    BinaryOp::Minimum | BinaryOp::Fmin => a.min(b),
                    BinaryOp::Gcd => $gcd(a, b),
                    BinaryOp::Lcm => $lcm(a, b),
                    BinaryOp::BitwiseAnd => a & b,
                    BinaryOp::BitwiseOr => a | b,
                    BinaryOp::BitwiseXor => a ^ b,
                    BinaryOp::LeftShift => $lsh(a, b),
                    BinaryOp::RightShift => $rsh(a, b),
                    BinaryOp::Power => $pow_fn(a, b).unwrap_or(0),
                    BinaryOp::LogicalAnd => {
                        if Scalar::is_truthy(a) && Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalOr => {
                        if Scalar::is_truthy(a) || Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalXor => {
                        if Scalar::is_truthy(a) ^ Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    _ => unreachable!("invalid op {:?} for int", op),
                }
            }
            #[inline]
            fn ho_bin_try(op: BinaryOp, a: Self, b: Self) -> Result<Self> {
                match op {
                    BinaryOp::Power => $pow_fn(a, b),
                    _ => Ok(Self::ho_bin_fast(op, a, b)),
                }
            }
            #[inline]
            fn ho_cmp(op: BinaryOp, a: Self, b: Self) -> bool {
                match op {
                    BinaryOp::Equal => a == b,
                    BinaryOp::NotEqual => a != b,
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    BinaryOp::GreaterEqual => a >= b,
                    _ => unreachable!(),
                }
            }
            #[inline]
            fn ho_unary(op: UnaryOp, a: Self) -> Result<Self> {
                Ok(match op {
                    UnaryOp::Negative => a.wrapping_neg(),
                    UnaryOp::Positive | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc | UnaryOp::Conj | UnaryOp::Real => a,
                    UnaryOp::Imag => 0,
                    UnaryOp::Abs => a.wrapping_abs(),
                    UnaryOp::Square => a.wrapping_mul(a),
                    UnaryOp::Reciprocal => $recip(a),
                    UnaryOp::Sign => a.signum(),
                    UnaryOp::Invert => !a,
                    _ => return Err(dtype_err(op.name(), "integer input")),
                })
            }
            #[inline]
            fn supports_simd_reduce(op: BinaryOp) -> bool {
                $simd_flag && matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Maximum | BinaryOp::Minimum | BinaryOp::Fmax | BinaryOp::Fmin
                )
            }
            #[inline]
            fn simd_reduce(op: BinaryOp, slice: &[Self]) -> Option<Self> {
                $simd_fn(op, slice)
            }
        }
    };
}

impl_ho_signed_int!(i8, I8, floor_div_i8, floor_rem_i8, pow_i8, lshift_i8, rshift_i8, gcd_i8, lcm_i8, int_recip_i8, false, |_, _| None);
impl_ho_signed_int!(i16, I16, floor_div_i16, floor_rem_i16, pow_i16, lshift_i16, rshift_i16, gcd_i16, lcm_i16, int_recip_i16, false, |_, _| None);
impl_ho_signed_int!(
    i32, I32, floor_div_i32, floor_rem_i32, pow_i32, lshift_i32, rshift_i32, gcd_i32, lcm_i32, int_recip_i32,
    true,
    |op, slice: &[i32]| match op {
        BinaryOp::Add => Some(simd::reduce_sum_i32(slice)),
        BinaryOp::Maximum | BinaryOp::Fmax => Some(simd::reduce_max_i32(slice)),
        BinaryOp::Minimum | BinaryOp::Fmin => Some(simd::reduce_min_i32(slice)),
        _ => None,
    }
);
impl_ho_signed_int!(
    i64, I64, floor_div_i64, floor_rem_i64, pow_i64, lshift_i64, rshift_i64, gcd_i64, lcm_i64, int_recip_i64,
    true,
    |op, slice: &[i64]| match op {
        BinaryOp::Add => Some(simd::reduce_sum_i64(slice)),
        BinaryOp::Maximum | BinaryOp::Fmax => Some(simd::reduce_max_i64(slice)),
        BinaryOp::Minimum | BinaryOp::Fmin => Some(simd::reduce_min_i64(slice)),
        _ => None,
    }
);

macro_rules! impl_ho_unsigned_int {
    ($t:ty, $var:ident, $fdiv:ident, $frem:ident, $pow_fn:ident, $lsh:ident, $rsh:ident, $gcd:ident, $lcm:ident, $recip:ident) => {
        impl HigherOrderScalar for $t {
            #[inline]
            fn to_scalar_value(self) -> ScalarValue {
                ScalarValue::$var(self)
            }
            #[inline]
            fn from_scalar_value(v: ScalarValue) -> Result<Self> {
                match v {
                    ScalarValue::$var(x) => Ok(x),
                    _ => Err(dtype_err("scalar", "scalar dtype mismatch")),
                }
            }
            #[inline]
            fn ho_bin_fast(op: BinaryOp, a: Self, b: Self) -> Self {
                match op {
                    BinaryOp::Add => a.wrapping_add(b),
                    BinaryOp::Subtract => a.wrapping_sub(b),
                    BinaryOp::Multiply => a.wrapping_mul(b),
                    BinaryOp::FloorDivide => $fdiv(a, b),
                    BinaryOp::Remainder | BinaryOp::Fmod => $frem(a, b),
                    BinaryOp::Maximum | BinaryOp::Fmax => a.max(b),
                    BinaryOp::Minimum | BinaryOp::Fmin => a.min(b),
                    BinaryOp::Gcd => $gcd(a, b),
                    BinaryOp::Lcm => $lcm(a, b),
                    BinaryOp::BitwiseAnd => a & b,
                    BinaryOp::BitwiseOr => a | b,
                    BinaryOp::BitwiseXor => a ^ b,
                    BinaryOp::LeftShift => $lsh(a, b),
                    BinaryOp::RightShift => $rsh(a, b),
                    BinaryOp::Power => $pow_fn(a, b),
                    BinaryOp::LogicalAnd => {
                        if Scalar::is_truthy(a) && Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalOr => {
                        if Scalar::is_truthy(a) || Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalXor => {
                        if Scalar::is_truthy(a) ^ Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    _ => unreachable!("invalid op {:?} for uint", op),
                }
            }
            #[inline]
            fn ho_cmp(op: BinaryOp, a: Self, b: Self) -> bool {
                match op {
                    BinaryOp::Equal => a == b,
                    BinaryOp::NotEqual => a != b,
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    BinaryOp::GreaterEqual => a >= b,
                    _ => unreachable!(),
                }
            }
            #[inline]
            fn ho_unary(op: UnaryOp, a: Self) -> Result<Self> {
                Ok(match op {
                    UnaryOp::Negative => a.wrapping_neg(),
                    UnaryOp::Positive | UnaryOp::Abs | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Trunc | UnaryOp::Conj | UnaryOp::Real => a,
                    UnaryOp::Imag => 0,
                    UnaryOp::Square => a.wrapping_mul(a),
                    UnaryOp::Reciprocal => $recip(a),
                    UnaryOp::Sign => if a == 0 { 0 } else { 1 },
                    UnaryOp::Invert => !a,
                    _ => return Err(dtype_err(op.name(), "unsigned integer input")),
                })
            }
        }
    };
}

impl_ho_unsigned_int!(u8, U8, floor_div_u8, floor_rem_u8, pow_u8, lshift_u8, rshift_u8, gcd_u8, lcm_u8, int_recip_u8);
impl_ho_unsigned_int!(u16, U16, floor_div_u16, floor_rem_u16, pow_u16, lshift_u16, rshift_u16, gcd_u16, lcm_u16, int_recip_u16);
impl_ho_unsigned_int!(u32, U32, floor_div_u32, floor_rem_u32, pow_u32, lshift_u32, rshift_u32, gcd_u32, lcm_u32, int_recip_u32);
impl_ho_unsigned_int!(u64, U64, floor_div_u64, floor_rem_u64, pow_u64, lshift_u64, rshift_u64, gcd_u64, lcm_u64, int_recip_u64);

impl HigherOrderScalar for f16 {
    #[inline]
    fn to_scalar_value(self) -> ScalarValue {
        ScalarValue::F16(self)
    }
    #[inline]
    fn from_scalar_value(v: ScalarValue) -> Result<Self> {
        match v {
            ScalarValue::F16(x) => Ok(x),
            _ => Err(dtype_err("scalar", "scalar dtype mismatch")),
        }
    }
    #[inline]
    fn ho_bin_fast(op: BinaryOp, a: f16, b: f16) -> f16 {
        let x = a.to_f32();
        let y = b.to_f32();
        f16::from_f32(match op {
            BinaryOp::Add => x + y,
            BinaryOp::Subtract => x - y,
            BinaryOp::Multiply => x * y,
            BinaryOp::TrueDivide => x / y,
            BinaryOp::FloorDivide => floor_divide_f32(x, y),
            BinaryOp::Remainder => remainder_f32(x, y),
            BinaryOp::Fmod => x % y,
            BinaryOp::Power => x.powf(y),
            BinaryOp::Logaddexp => logaddexp_f32(x, y),
            BinaryOp::Logaddexp2 => logaddexp2_f32(x, y),
            BinaryOp::Arctan2 => x.atan2(y),
            BinaryOp::Hypot => x.hypot(y),
            BinaryOp::Copysign => {
                return f16::from_bits((a.to_bits() & 0x7FFF) | (b.to_bits() & 0x8000))
            }
            BinaryOp::Nextafter => return nextafter_f16(a, b),
            BinaryOp::Heaviside => heaviside_f32(x, y),
            BinaryOp::Maximum => return max_prop_nan_f16(a, b),
            BinaryOp::Minimum => return min_prop_nan_f16(a, b),
            BinaryOp::Fmax => return fmax_f16(a, b),
            BinaryOp::Fmin => return fmin_f16(a, b),
            BinaryOp::LogicalAnd => {
                return if Scalar::is_truthy(a) && Scalar::is_truthy(b) {
                    f16::ONE
                } else {
                    f16::ZERO
                };
            }
            BinaryOp::LogicalOr => {
                return if Scalar::is_truthy(a) || Scalar::is_truthy(b) {
                    f16::ONE
                } else {
                    f16::ZERO
                };
            }
            BinaryOp::LogicalXor => {
                return if Scalar::is_truthy(a) ^ Scalar::is_truthy(b) {
                    f16::ONE
                } else {
                    f16::ZERO
                };
            }
            _ => unreachable!("invalid BinaryOp {:?} for f16", op),
        })
    }
    #[inline]
    fn ho_cmp(op: BinaryOp, a: f16, b: f16) -> bool {
        let (x, y) = (a.to_f32(), b.to_f32());
        match op {
            BinaryOp::Equal => x == y,
            BinaryOp::NotEqual => x != y,
            BinaryOp::Less => x < y,
            BinaryOp::LessEqual => x <= y,
            BinaryOp::Greater => x > y,
            BinaryOp::GreaterEqual => x >= y,
            _ => unreachable!(),
        }
    }
    #[inline]
    fn ho_unary(op: UnaryOp, a: f16) -> Result<f16> {
        let x = a.to_f32();
        Ok(match op {
            UnaryOp::Negative => f16::from_bits(a.to_bits() ^ 0x8000),
            UnaryOp::Positive | UnaryOp::Conj | UnaryOp::Real => a,
            UnaryOp::Imag => f16::ZERO,
            UnaryOp::Abs | UnaryOp::Fabs => f16::from_bits(a.to_bits() & 0x7FFF),
            UnaryOp::Square => f16::from_f32(x * x),
            UnaryOp::Reciprocal => f16::from_f32(1.0 / x),
            UnaryOp::Sqrt => f16::from_f32(x.sqrt()),
            UnaryOp::Cbrt => f16::from_f32(x.cbrt()),
            UnaryOp::Exp => f16::from_f32(x.exp()),
            UnaryOp::Exp2 => f16::from_f32(x.exp2()),
            UnaryOp::Expm1 => f16::from_f32(x.exp_m1()),
            UnaryOp::Log => f16::from_f32(x.ln()),
            UnaryOp::Log2 => f16::from_f32(x.log2()),
            UnaryOp::Log10 => f16::from_f32(x.log10()),
            UnaryOp::Log1p => f16::from_f32(x.ln_1p()),
            UnaryOp::Sin => f16::from_f32(x.sin()),
            UnaryOp::Cos => f16::from_f32(x.cos()),
            UnaryOp::Tan => f16::from_f32(x.tan()),
            UnaryOp::Arcsin => f16::from_f32(x.asin()),
            UnaryOp::Arccos => f16::from_f32(x.acos()),
            UnaryOp::Arctan => f16::from_f32(x.atan()),
            UnaryOp::Sinh => f16::from_f32(x.sinh()),
            UnaryOp::Cosh => f16::from_f32(x.cosh()),
            UnaryOp::Tanh => f16::from_f32(x.tanh()),
            UnaryOp::Arcsinh => f16::from_f32(x.asinh()),
            UnaryOp::Arccosh => f16::from_f32(x.acosh()),
            UnaryOp::Arctanh => f16::from_f32(x.atanh()),
            UnaryOp::Deg2Rad => f16::from_f32(x * (std::f32::consts::PI / 180.0)),
            UnaryOp::Rad2Deg => f16::from_f32(x * (180.0 / std::f32::consts::PI)),
            UnaryOp::Floor => f16::from_f32(x.floor()),
            UnaryOp::Ceil => f16::from_f32(x.ceil()),
            UnaryOp::Trunc => f16::from_f32(x.trunc()),
            UnaryOp::Rint => f16::from_f32(x.round_ties_even()),
            UnaryOp::Sign => f16::from_f32(sign_f32(x)),
            UnaryOp::Spacing => spacing_f16(a),
            UnaryOp::Sinc => sinc_f16(a),
            UnaryOp::I0 => f16::from_f64(i0_f64(a.to_f64())),
            _ => return Err(dtype_err(op.name(), "float16 input")),
        })
    }
}

macro_rules! impl_ho_float {
    (
        $t:ty, $var:ident, $pi:expr,
        $fdiv:ident, $frem:ident, $lae:ident, $lae2:ident, $next:ident, $heav:ident,
        $max:ident, $min:ident, $fmax:ident, $fmin:ident, $sign:ident, $spac:ident, $sinc:ident,
        $sum_simd:ident, $prod_simd:ident, $max_simd:ident, $min_simd:ident
    ) => {
        impl HigherOrderScalar for $t {
            #[inline]
            fn to_scalar_value(self) -> ScalarValue {
                ScalarValue::$var(self)
            }
            #[inline]
            fn from_scalar_value(v: ScalarValue) -> Result<Self> {
                match v {
                    ScalarValue::$var(x) => Ok(x),
                    _ => Err(dtype_err("scalar", "scalar dtype mismatch")),
                }
            }
            #[inline]
            fn ho_bin_fast(op: BinaryOp, a: Self, b: Self) -> Self {
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Subtract => a - b,
                    BinaryOp::Multiply => a * b,
                    BinaryOp::TrueDivide => a / b,
                    BinaryOp::FloorDivide => $fdiv(a, b),
                    BinaryOp::Remainder => $frem(a, b),
                    BinaryOp::Fmod => a % b,
                    BinaryOp::Power | BinaryOp::FloatPower => a.powf(b),
                    BinaryOp::Logaddexp => $lae(a, b),
                    BinaryOp::Logaddexp2 => $lae2(a, b),
                    BinaryOp::Arctan2 => a.atan2(b),
                    BinaryOp::Hypot => a.hypot(b),
                    BinaryOp::Copysign => a.copysign(b),
                    BinaryOp::Nextafter => $next(a, b),
                    BinaryOp::Heaviside => $heav(a, b),
                    BinaryOp::Maximum => $max(a, b),
                    BinaryOp::Minimum => $min(a, b),
                    BinaryOp::Fmax => $fmax(a, b),
                    BinaryOp::Fmin => $fmin(a, b),
                    BinaryOp::LogicalAnd => {
                        if Scalar::is_truthy(a) && Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalOr => {
                        if Scalar::is_truthy(a) || Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalXor => {
                        if Scalar::is_truthy(a) ^ Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    _ => unreachable!("invalid BinaryOp {:?} for float", op),
                }
            }
            #[inline]
            fn ho_cmp(op: BinaryOp, a: Self, b: Self) -> bool {
                match op {
                    BinaryOp::Equal => a == b,
                    BinaryOp::NotEqual => a != b,
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    BinaryOp::GreaterEqual => a >= b,
                    _ => unreachable!(),
                }
            }
            #[inline]
            fn ho_unary(op: UnaryOp, x: Self) -> Result<Self> {
                Ok(match op {
                    UnaryOp::Negative => -x,
                    UnaryOp::Positive | UnaryOp::Conj | UnaryOp::Real => x,
                    UnaryOp::Imag => 0.0,
                    UnaryOp::Abs | UnaryOp::Fabs => x.abs(),
                    UnaryOp::Square => x * x,
                    UnaryOp::Reciprocal => 1.0 / x,
                    UnaryOp::Sqrt => x.sqrt(),
                    UnaryOp::Cbrt => x.cbrt(),
                    UnaryOp::Exp => x.exp(),
                    UnaryOp::Exp2 => x.exp2(),
                    UnaryOp::Expm1 => x.exp_m1(),
                    UnaryOp::Log => x.ln(),
                    UnaryOp::Log2 => x.log2(),
                    UnaryOp::Log10 => x.log10(),
                    UnaryOp::Log1p => x.ln_1p(),
                    UnaryOp::Sin => x.sin(),
                    UnaryOp::Cos => x.cos(),
                    UnaryOp::Tan => x.tan(),
                    UnaryOp::Arcsin => x.asin(),
                    UnaryOp::Arccos => x.acos(),
                    UnaryOp::Arctan => x.atan(),
                    UnaryOp::Sinh => x.sinh(),
                    UnaryOp::Cosh => x.cosh(),
                    UnaryOp::Tanh => x.tanh(),
                    UnaryOp::Arcsinh => x.asinh(),
                    UnaryOp::Arccosh => x.acosh(),
                    UnaryOp::Arctanh => x.atanh(),
                    UnaryOp::Deg2Rad => x * ($pi / 180.0),
                    UnaryOp::Rad2Deg => x * (180.0 / $pi),
                    UnaryOp::Floor => x.floor(),
                    UnaryOp::Ceil => x.ceil(),
                    UnaryOp::Trunc => x.trunc(),
                    UnaryOp::Rint => x.round_ties_even(),
                    UnaryOp::Sign => $sign(x),
                    UnaryOp::Spacing => $spac(x),
                    UnaryOp::Sinc => $sinc(x),
                    UnaryOp::I0 => i0_f64(x as f64) as Self,
                    _ => return Err(dtype_err(op.name(), "float input")),
                })
            }
            #[inline]
            fn supports_simd_reduce(op: BinaryOp) -> bool {
                matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Multiply | BinaryOp::Maximum | BinaryOp::Minimum
                )
            }
            #[inline]
            fn simd_reduce(op: BinaryOp, slice: &[Self]) -> Option<Self> {
                match op {
                    BinaryOp::Add => Some(simd::$sum_simd(slice)),
                    BinaryOp::Multiply => Some(simd::$prod_simd(slice)),
                    BinaryOp::Maximum => Some(simd::$max_simd(slice)),
                    BinaryOp::Minimum => Some(simd::$min_simd(slice)),
                    _ => None,
                }
            }
        }
    };
}

impl_ho_float!(
    f32, F32, std::f32::consts::PI,
    floor_divide_f32, remainder_f32, logaddexp_f32, logaddexp2_f32, nextafter_f32, heaviside_f32,
    max_prop_nan_f32, min_prop_nan_f32, fmax_f32, fmin_f32, sign_f32, spacing_f32, sinc_f32,
    reduce_sum_f32, reduce_prod_f32, reduce_max_f32, reduce_min_f32
);

impl_ho_float!(
    f64, F64, std::f64::consts::PI,
    floor_divide_f64, remainder_f64, logaddexp_f64, logaddexp2_f64, nextafter_f64, heaviside_f64,
    max_prop_nan_f64, min_prop_nan_f64, fmax_f64, fmin_f64, sign_f64, spacing_f64, sinc_f64,
    reduce_sum_f64, reduce_prod_f64, reduce_max_f64, reduce_min_f64
);

macro_rules! impl_ho_complex {
    (
        $c:ty, $var:ident, $div:ident, $pow:ident, $recip:ident,
        $exp2:ident, $expm1:ident, $log2:ident, $log10:ident, $log1p:ident, $rint:ident, $sign:ident, $sinc:ident
    ) => {
        impl HigherOrderScalar for $c {
            #[inline]
            fn to_scalar_value(self) -> ScalarValue {
                ScalarValue::$var(self)
            }
            #[inline]
            fn from_scalar_value(v: ScalarValue) -> Result<Self> {
                match v {
                    ScalarValue::$var(x) => Ok(x),
                    _ => Err(dtype_err("scalar", "scalar dtype mismatch")),
                }
            }
            #[inline]
            fn ho_bin_fast(op: BinaryOp, a: Self, b: Self) -> Self {
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Subtract => a - b,
                    BinaryOp::Multiply => a * b,
                    BinaryOp::TrueDivide => $div(a, b),
                    BinaryOp::Power | BinaryOp::FloatPower => $pow(a, b),
                    BinaryOp::Maximum => c_max(a, b),
                    BinaryOp::Minimum => c_min(a, b),
                    BinaryOp::Fmax => c_fmax(a, b),
                    BinaryOp::Fmin => c_fmin(a, b),
                    BinaryOp::LogicalAnd => {
                        if Scalar::is_truthy(a) && Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalOr => {
                        if Scalar::is_truthy(a) || Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    BinaryOp::LogicalXor => {
                        if Scalar::is_truthy(a) ^ Scalar::is_truthy(b) {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }
                    }
                    _ => unreachable!("invalid BinaryOp {:?} for complex", op),
                }
            }
            #[inline]
            fn ho_cmp(op: BinaryOp, a: Self, b: Self) -> bool {
                match op {
                    BinaryOp::Equal => c_eq(a, b),
                    BinaryOp::NotEqual => c_ne(a, b),
                    BinaryOp::Less => c_lt(a, b),
                    BinaryOp::LessEqual => c_le(a, b),
                    BinaryOp::Greater => c_gt(a, b),
                    BinaryOp::GreaterEqual => c_ge(a, b),
                    _ => unreachable!(),
                }
            }
            #[inline]
            fn ho_unary(op: UnaryOp, z: Self) -> Result<Self> {
                Ok(match op {
                    UnaryOp::Negative => -z,
                    UnaryOp::Positive => z,
                    UnaryOp::Abs | UnaryOp::Fabs => Self::new(z.norm(), 0.0),
                    UnaryOp::Real => Self::new(z.re, 0.0),
                    UnaryOp::Imag => Self::new(z.im, 0.0),
                    UnaryOp::Conj => z.conj(),
                    UnaryOp::Square => z * z,
                    UnaryOp::Reciprocal => $recip(z),
                    UnaryOp::Sqrt => z.sqrt(),
                    UnaryOp::Exp => z.exp(),
                    UnaryOp::Exp2 => $exp2(z),
                    UnaryOp::Expm1 => $expm1(z),
                    UnaryOp::Log => z.ln(),
                    UnaryOp::Log2 => $log2(z),
                    UnaryOp::Log10 => $log10(z),
                    UnaryOp::Log1p => $log1p(z),
                    UnaryOp::Sin => z.sin(),
                    UnaryOp::Cos => z.cos(),
                    UnaryOp::Tan => z.tan(),
                    UnaryOp::Arcsin => z.asin(),
                    UnaryOp::Arccos => z.acos(),
                    UnaryOp::Arctan => z.atan(),
                    UnaryOp::Sinh => z.sinh(),
                    UnaryOp::Cosh => z.cosh(),
                    UnaryOp::Tanh => z.tanh(),
                    UnaryOp::Arcsinh => z.asinh(),
                    UnaryOp::Arccosh => z.acosh(),
                    UnaryOp::Arctanh => z.atanh(),
                    UnaryOp::Rint => $rint(z),
                    UnaryOp::Sign => $sign(z),
                    UnaryOp::Sinc => $sinc(z),
                    _ => return Err(dtype_err(op.name(), "complex input")),
                })
            }
        }
    };
}

impl_ho_complex!(
    Complex32, C64, c64_div, c64_pow, c64_recip,
    c64_exp2, c64_expm1, c64_log2, c64_log10, c64_log1p, c64_rint, c64_sign, c64_sinc
);
impl_ho_complex!(
    Complex64, C128, c128_div, c128_pow, c128_recip,
    c128_exp2, c128_expm1, c128_log2, c128_log10, c128_log1p, c128_rint, c128_sign, c128_sinc
);

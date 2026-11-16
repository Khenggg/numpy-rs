//! Dynamic `DynArray` ufuncs with exact NumPy 2.x type promotion and broadcasting.

use crate::exec::dtype_err;
use crate::typed_ufunc::{self as typed, UfuncScalar};
use numpy_types::{
    broadcast_shapes, promote_types, result_type, Casting, DType, DynArray, NdArray, Result,
};
use std::borrow::Cow;

#[inline]
fn cast_borrowed<'a>(a: &'a DynArray, target: DType) -> Result<Cow<'a, DynArray>> {
    if a.dtype() == target {
        Ok(Cow::Borrowed(a))
    } else {
        Ok(Cow::Owned(a.astype(target, Casting::Unsafe)?))
    }
}

#[inline]
fn unary_float_target(dt: DType) -> Result<DType> {
    match dt {
        DType::Bool | DType::I8 | DType::U8 | DType::F16 => Ok(DType::F16),
        DType::I16 | DType::U16 | DType::F32 => Ok(DType::F32),
        DType::I32 | DType::U32 | DType::I64 | DType::U64 | DType::F64 => Ok(DType::F64),
        DType::C64 => Ok(DType::C64),
        DType::C128 => Ok(DType::C128),
    }
}

#[inline]
fn unary_real_float_target(op: &str, dt: DType) -> Result<DType> {
    if dt.is_complex() {
        return Err(dtype_err(op, &format!("{dt:?} input")));
    }
    unary_float_target(dt)
}

macro_rules! dyn_unary_dispatch {
    ($a:expr, $fn_name:ident) => {
        match $a {
            DynArray::Bool(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::I8(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::I16(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::I32(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::I64(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::U8(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::U16(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::U32(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::U64(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::F16(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::F32(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::F64(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::C64(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
            DynArray::C128(x) => Ok(DynArray::from(typed::$fn_name(x)?)),
        }
    };
}

macro_rules! dyn_unary_bool_dispatch {
    ($a:expr, $fn_name:ident) => {
        match $a {
            DynArray::Bool(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::I8(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::I16(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::I32(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::I64(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::U8(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::U16(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::U32(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::U64(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::F16(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::F32(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::F64(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::C64(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
            DynArray::C128(x) => Ok(DynArray::Bool(typed::$fn_name(x)?)),
        }
    };
}

macro_rules! dyn_bin_same_dispatch {
    ($a:expr, $b:expr, $fn_name:ident) => {{
        let dt = promote_types($a.dtype(), $b.dtype());
        let ca = cast_borrowed($a, dt)?;
        let cb = cast_borrowed($b, dt)?;
        match (ca.as_ref(), cb.as_ref()) {
            (DynArray::Bool(x), DynArray::Bool(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::I8(x), DynArray::I8(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::I16(x), DynArray::I16(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::I32(x), DynArray::I32(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::I64(x), DynArray::I64(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::U8(x), DynArray::U8(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::U16(x), DynArray::U16(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::U32(x), DynArray::U32(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::U64(x), DynArray::U64(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::F16(x), DynArray::F16(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::F32(x), DynArray::F32(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::F64(x), DynArray::F64(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::C64(x), DynArray::C64(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::C128(x), DynArray::C128(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            _ => unreachable!(),
        }
    }};
}

macro_rules! dyn_bin_real_float_dispatch {
    ($op_name:expr, $a:expr, $b:expr, $fn_name:ident) => {{
        let fa = unary_real_float_target($op_name, $a.dtype())?;
        let fb = unary_real_float_target($op_name, $b.dtype())?;
        let dt = promote_types(fa, fb);
        let ca = cast_borrowed($a, dt)?;
        let cb = cast_borrowed($b, dt)?;
        match (ca.as_ref(), cb.as_ref()) {
            (DynArray::F16(x), DynArray::F16(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::F32(x), DynArray::F32(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            (DynArray::F64(x), DynArray::F64(y)) => Ok(DynArray::from(typed::$fn_name(x, y)?)),
            _ => unreachable!(),
        }
    }};
}

macro_rules! dyn_bin_bool_dispatch {
    ($a:expr, $b:expr, $fn_name:ident) => {{
        let dt = promote_types($a.dtype(), $b.dtype());
        let ca = cast_borrowed($a, dt)?;
        let cb = cast_borrowed($b, dt)?;
        match (ca.as_ref(), cb.as_ref()) {
            (DynArray::Bool(x), DynArray::Bool(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::I8(x), DynArray::I8(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::I16(x), DynArray::I16(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::I32(x), DynArray::I32(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::I64(x), DynArray::I64(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::U8(x), DynArray::U8(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::U16(x), DynArray::U16(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::U32(x), DynArray::U32(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::U64(x), DynArray::U64(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::F16(x), DynArray::F16(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::F32(x), DynArray::F32(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::F64(x), DynArray::F64(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::C64(x), DynArray::C64(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            (DynArray::C128(x), DynArray::C128(y)) => Ok(DynArray::Bool(typed::$fn_name(x, y)?)),
            _ => unreachable!(),
        }
    }};
}

// ============================================================================
// 1. Arithmetic
// ============================================================================

#[inline]
pub fn add(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, add)
}

#[inline]
pub fn subtract(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, subtract)
}

#[inline]
pub fn multiply(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, multiply)
}

#[inline]
pub fn true_divide(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    let dt = promote_types(a.dtype(), b.dtype());
    let target = if dt.is_inexact() { dt } else { DType::F64 };
    let ca = cast_borrowed(a, target)?;
    let cb = cast_borrowed(b, target)?;
    match (ca.as_ref(), cb.as_ref()) {
        (DynArray::F16(x), DynArray::F16(y)) => Ok(DynArray::from(typed::true_divide(x, y)?)),
        (DynArray::F32(x), DynArray::F32(y)) => Ok(DynArray::from(typed::true_divide(x, y)?)),
        (DynArray::F64(x), DynArray::F64(y)) => Ok(DynArray::from(typed::true_divide(x, y)?)),
        (DynArray::C64(x), DynArray::C64(y)) => Ok(DynArray::from(typed::true_divide(x, y)?)),
        (DynArray::C128(x), DynArray::C128(y)) => Ok(DynArray::from(typed::true_divide(x, y)?)),
        _ => unreachable!(),
    }
}

pub use true_divide as divide;

#[inline]
pub fn floor_divide(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, floor_divide)
}

#[inline]
pub fn remainder(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, remainder)
}

pub use remainder as modulo;
pub use remainder as r#mod;

#[inline]
pub fn fmod(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, fmod)
}

#[inline]
pub fn divmod(a: &DynArray, b: &DynArray) -> Result<(DynArray, DynArray)> {
    let dt = promote_types(a.dtype(), b.dtype());
    let ca = cast_borrowed(a, dt)?;
    let cb = cast_borrowed(b, dt)?;
    match (ca.as_ref(), cb.as_ref()) {
        (DynArray::Bool(x), DynArray::Bool(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::I8(x), DynArray::I8(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::I16(x), DynArray::I16(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::I32(x), DynArray::I32(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::I64(x), DynArray::I64(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::U8(x), DynArray::U8(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::U16(x), DynArray::U16(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::U32(x), DynArray::U32(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::U64(x), DynArray::U64(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::F16(x), DynArray::F16(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::F32(x), DynArray::F32(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::F64(x), DynArray::F64(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::C64(x), DynArray::C64(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        (DynArray::C128(x), DynArray::C128(y)) => {
            let (q, r) = typed::divmod(x, y)?;
            Ok((DynArray::from(q), DynArray::from(r)))
        }
        _ => unreachable!(),
    }
}

#[inline]
pub fn power(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, power)
}

#[inline]
pub fn float_power(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    let target = if a.dtype().is_complex() || b.dtype().is_complex() {
        DType::C128
    } else {
        DType::F64
    };
    let ca = cast_borrowed(a, target)?;
    let cb = cast_borrowed(b, target)?;
    match (ca.as_ref(), cb.as_ref()) {
        (DynArray::F64(x), DynArray::F64(y)) => Ok(DynArray::from(typed::float_power(x, y)?)),
        (DynArray::C128(x), DynArray::C128(y)) => Ok(DynArray::from(typed::float_power(x, y)?)),
        _ => unreachable!(),
    }
}

#[inline]
pub fn negative(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, negative)
}

#[inline]
pub fn positive(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, positive)
}

#[inline]
pub fn abs(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, abs)
}

pub use abs as absolute;

#[inline]
pub fn fabs(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, fabs)
}

#[inline]
pub fn square(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, square)
}

#[inline]
pub fn reciprocal(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, reciprocal)
}

#[inline]
pub fn sqrt(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, sqrt)
}

#[inline]
pub fn cbrt(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, cbrt)
}

// ============================================================================
// 2. Exp / Log
// ============================================================================

#[inline]
pub fn exp(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, exp)
}
#[inline]
pub fn exp2(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, exp2)
}
#[inline]
pub fn expm1(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, expm1)
}
#[inline]
pub fn log(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, log)
}
#[inline]
pub fn log2(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, log2)
}
#[inline]
pub fn log10(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, log10)
}
#[inline]
pub fn log1p(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, log1p)
}
#[inline]
pub fn logaddexp(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("logaddexp", a, b, logaddexp)
}
#[inline]
pub fn logaddexp2(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("logaddexp2", a, b, logaddexp2)
}

// ============================================================================
// 3. Trig / Hyperbolic
// ============================================================================

#[inline]
pub fn sin(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, sin)
}
#[inline]
pub fn cos(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, cos)
}
#[inline]
pub fn tan(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, tan)
}
#[inline]
pub fn arcsin(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, arcsin)
}
pub use arcsin as asin;
#[inline]
pub fn arccos(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, arccos)
}
pub use arccos as acos;
#[inline]
pub fn arctan(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, arctan)
}
pub use arctan as atan;
#[inline]
pub fn arctan2(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("arctan2", a, b, arctan2)
}
pub use arctan2 as atan2;
#[inline]
pub fn hypot(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("hypot", a, b, hypot)
}
#[inline]
pub fn sinh(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, sinh)
}
#[inline]
pub fn cosh(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, cosh)
}
#[inline]
pub fn tanh(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, tanh)
}
#[inline]
pub fn arcsinh(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, arcsinh)
}
pub use arcsinh as asinh;
#[inline]
pub fn arccosh(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, arccosh)
}
pub use arccosh as acosh;
#[inline]
pub fn arctanh(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, arctanh)
}
pub use arctanh as atanh;
#[inline]
pub fn deg2rad(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, deg2rad)
}
pub use deg2rad as radians;
#[inline]
pub fn rad2deg(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, rad2deg)
}
pub use rad2deg as degrees;

// ============================================================================
// 4. Rounding / Float / Special
// ============================================================================

#[inline]
pub fn floor(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, floor)
}
#[inline]
pub fn ceil(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, ceil)
}
#[inline]
pub fn trunc(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, trunc)
}
pub use trunc as fix;
#[inline]
pub fn rint(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, rint)
}
#[inline]
pub fn round(a: &DynArray, decimals: i32) -> Result<DynArray> {
    match a {
        DynArray::Bool(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::I8(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::I16(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::I32(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::I64(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::U8(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::U16(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::U32(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::U64(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::F16(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::F32(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::F64(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::C64(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
        DynArray::C128(x) => Ok(DynArray::from(typed::round(x, decimals)?)),
    }
}
pub use round as around;
#[inline]
pub fn sign(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, sign)
}
#[inline]
pub fn signbit(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, signbit)
}
#[inline]
pub fn copysign(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("copysign", a, b, copysign)
}
#[inline]
pub fn spacing(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, spacing)
}
#[inline]
pub fn nextafter(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("nextafter", a, b, nextafter)
}

#[inline]
fn dyn_to_ldexp_i64(exp: &DynArray) -> Result<NdArray<i64>> {
    match exp {
        DynArray::Bool(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::I8(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::I16(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::I32(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::I64(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::U8(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::U16(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        DynArray::U32(x) => crate::exec::try_eval_unary(x, |e| e.u_to_ldexp_exp()),
        other => Err(dtype_err("ldexp", &format!("{:?} exponent", other.dtype()))),
    }
}

#[inline]
pub fn ldexp(a: &DynArray, exp: &DynArray) -> Result<DynArray> {
    let target = unary_real_float_target("ldexp", a.dtype())?;
    let ca = cast_borrowed(a, target)?;
    let exp_i64 = dyn_to_ldexp_i64(exp)?;
    match ca.as_ref() {
        DynArray::F16(x) => Ok(DynArray::from(typed::ldexp(x, &exp_i64)?)),
        DynArray::F32(x) => Ok(DynArray::from(typed::ldexp(x, &exp_i64)?)),
        DynArray::F64(x) => Ok(DynArray::from(typed::ldexp(x, &exp_i64)?)),
        _ => unreachable!(),
    }
}

#[inline]
pub fn frexp(a: &DynArray) -> Result<(DynArray, DynArray)> {
    match a {
        DynArray::Bool(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::I8(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::I16(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::I32(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::I64(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::U8(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::U16(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::U32(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::U64(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::F16(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::F32(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::F64(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::C64(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
        DynArray::C128(x) => { let (m, e) = typed::frexp(x)?; Ok((DynArray::from(m), DynArray::from(e))) }
    }
}

#[inline]
pub fn modf(a: &DynArray) -> Result<(DynArray, DynArray)> {
    match a {
        DynArray::Bool(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::I8(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::I16(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::I32(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::I64(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::U8(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::U16(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::U32(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::U64(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::F16(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::F32(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::F64(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::C64(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
        DynArray::C128(x) => { let (f, i) = typed::modf(x)?; Ok((DynArray::from(f), DynArray::from(i))) }
    }
}

#[inline]
pub fn isfinite(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, isfinite)
}
#[inline]
pub fn isinf(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, isinf)
}
#[inline]
pub fn isnan(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, isnan)
}
#[inline]
pub fn isneginf(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, isneginf)
}
#[inline]
pub fn isposinf(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, isposinf)
}

#[inline]
pub fn nan_to_num(
    a: &DynArray,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> Result<DynArray> {
    match a {
        DynArray::Bool(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::I8(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::I16(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::I32(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::I64(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::U8(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::U16(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::U32(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::U64(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::F16(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::F32(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::F64(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::C64(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
        DynArray::C128(x) => Ok(DynArray::from(typed::nan_to_num(x, nan, posinf, neginf)?)),
    }
}

#[inline]
pub fn clip(
    a: &DynArray,
    a_min: Option<&DynArray>,
    a_max: Option<&DynArray>,
) -> Result<DynArray> {
    let mut dtypes = vec![a.dtype()];
    if let Some(lo) = a_min {
        dtypes.push(lo.dtype());
    }
    if let Some(hi) = a_max {
        dtypes.push(hi.dtype());
    }
    let dt = result_type(&dtypes)?;
    let ca = cast_borrowed(a, dt)?;
    let clo = match a_min {
        Some(lo) => Some(cast_borrowed(lo, dt)?),
        None => None,
    };
    let chi = match a_max {
        Some(hi) => Some(cast_borrowed(hi, dt)?),
        None => None,
    };

    macro_rules! clip_arm {
        ($var:ident) => {{
            let DynArray::$var(ax) = ca.as_ref() else { unreachable!() };
            let lx = clo.as_ref().map(|c| {
                let DynArray::$var(v) = c.as_ref() else { unreachable!() };
                v
            });
            let hx = chi.as_ref().map(|c| {
                let DynArray::$var(v) = c.as_ref() else { unreachable!() };
                v
            });
            Ok(DynArray::from(typed::clip(ax, lx, hx)?))
        }};
    }

    match dt {
        DType::Bool => clip_arm!(Bool),
        DType::I8 => clip_arm!(I8),
        DType::I16 => clip_arm!(I16),
        DType::I32 => clip_arm!(I32),
        DType::I64 => clip_arm!(I64),
        DType::U8 => clip_arm!(U8),
        DType::U16 => clip_arm!(U16),
        DType::U32 => clip_arm!(U32),
        DType::U64 => clip_arm!(U64),
        DType::F16 => clip_arm!(F16),
        DType::F32 => clip_arm!(F32),
        DType::F64 => clip_arm!(F64),
        DType::C64 => clip_arm!(C64),
        DType::C128 => clip_arm!(C128),
    }
}

#[inline]
pub fn heaviside(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_real_float_dispatch!("heaviside", a, b, heaviside)
}
#[inline]
pub fn sinc(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, sinc)
}
#[inline]
pub fn i0(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, i0)
}
#[inline]
pub fn maximum(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, maximum)
}
#[inline]
pub fn minimum(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, minimum)
}
#[inline]
pub fn fmax(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, fmax)
}
#[inline]
pub fn fmin(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, fmin)
}
#[inline]
pub fn gcd(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, gcd)
}
#[inline]
pub fn lcm(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, lcm)
}

// ============================================================================
// 5. Complex / Bitwise / Comparison / Logical
// ============================================================================

#[inline]
pub fn real(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, real)
}
#[inline]
pub fn imag(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, imag)
}
#[inline]
pub fn conj(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, conj)
}
pub use conj as conjugate;

#[inline]
pub fn angle(a: &DynArray, deg: bool) -> Result<DynArray> {
    match a {
        DynArray::Bool(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::I8(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::I16(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::I32(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::I64(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::U8(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::U16(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::U32(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::U64(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::F16(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::F32(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::F64(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::C64(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
        DynArray::C128(x) => Ok(DynArray::from(typed::angle(x, deg)?)),
    }
}

#[inline]
pub fn bitwise_and(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, bitwise_and)
}
#[inline]
pub fn bitwise_or(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, bitwise_or)
}
#[inline]
pub fn bitwise_xor(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, bitwise_xor)
}
#[inline]
pub fn invert(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, invert)
}
pub use invert as bitwise_not;
#[inline]
pub fn left_shift(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, left_shift)
}
#[inline]
pub fn right_shift(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_same_dispatch!(a, b, right_shift)
}
#[inline]
pub fn bitwise_count(a: &DynArray) -> Result<DynArray> {
    dyn_unary_dispatch!(a, bitwise_count)
}

#[inline]
pub fn equal(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_bool_dispatch!(a, b, equal)
}
#[inline]
pub fn not_equal(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_bool_dispatch!(a, b, not_equal)
}
#[inline]
pub fn less(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_bool_dispatch!(a, b, less)
}
#[inline]
pub fn less_equal(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_bool_dispatch!(a, b, less_equal)
}
#[inline]
pub fn greater(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_bool_dispatch!(a, b, greater)
}
#[inline]
pub fn greater_equal(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dyn_bin_bool_dispatch!(a, b, greater_equal)
}

#[inline]
pub fn isclose(
    a: &DynArray,
    b: &DynArray,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<DynArray> {
    let dt = promote_types(a.dtype(), b.dtype());
    let ca = cast_borrowed(a, dt)?;
    let cb = cast_borrowed(b, dt)?;
    let out = match (ca.as_ref(), cb.as_ref()) {
        (DynArray::Bool(x), DynArray::Bool(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::I8(x), DynArray::I8(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::I16(x), DynArray::I16(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::I32(x), DynArray::I32(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::I64(x), DynArray::I64(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::U8(x), DynArray::U8(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::U16(x), DynArray::U16(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::U32(x), DynArray::U32(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::U64(x), DynArray::U64(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::F16(x), DynArray::F16(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::F32(x), DynArray::F32(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::F64(x), DynArray::F64(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::C64(x), DynArray::C64(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        (DynArray::C128(x), DynArray::C128(y)) => typed::isclose(x, y, rtol, atol, equal_nan)?,
        _ => unreachable!(),
    };
    Ok(DynArray::Bool(out))
}

#[inline]
pub fn allclose(
    a: &DynArray,
    b: &DynArray,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<bool> {
    let DynArray::Bool(mask) = isclose(a, b, rtol, atol, equal_nan)? else {
        unreachable!()
    };
    Ok(crate::exec::bool_all(&mask))
}

#[inline]
pub fn array_equal(a: &DynArray, b: &DynArray, equal_nan: bool) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    if a.is_empty() {
        return true;
    }
    if a.dtype() == b.dtype() {
        return match (a, b) {
            (DynArray::Bool(x), DynArray::Bool(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::I8(x), DynArray::I8(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::I16(x), DynArray::I16(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::I32(x), DynArray::I32(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::I64(x), DynArray::I64(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::U8(x), DynArray::U8(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::U16(x), DynArray::U16(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::U32(x), DynArray::U32(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::U64(x), DynArray::U64(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::F16(x), DynArray::F16(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::F32(x), DynArray::F32(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::F64(x), DynArray::F64(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::C64(x), DynArray::C64(y)) => typed::array_equal(x, y, equal_nan),
            (DynArray::C128(x), DynArray::C128(y)) => typed::array_equal(x, y, equal_nan),
            _ => false,
        };
    }
    let dt = promote_types(a.dtype(), b.dtype());
    let (Ok(ca), Ok(cb)) = (cast_borrowed(a, dt), cast_borrowed(b, dt)) else {
        return false;
    };
    array_equal(ca.as_ref(), cb.as_ref(), equal_nan)
}

#[inline]
pub fn array_equiv(a: &DynArray, b: &DynArray) -> bool {
    if broadcast_shapes(a.shape(), b.shape()).is_err() {
        return false;
    }
    match equal(a, b) {
        Ok(DynArray::Bool(mask)) => crate::exec::bool_all(&mask),
        _ => false,
    }
}

#[inline]
fn dyn_to_bool_arr(a: &DynArray) -> Result<Cow<'_, NdArray<bool>>> {
    if let DynArray::Bool(b) = a {
        Ok(Cow::Borrowed(b))
    } else {
        let converted = a.astype(DType::Bool, Casting::Unsafe)?;
        let DynArray::Bool(b) = converted else { unreachable!() };
        Ok(Cow::Owned(b))
    }
}

#[inline]
pub fn logical_and(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    let ba = dyn_to_bool_arr(a)?;
    let bb = dyn_to_bool_arr(b)?;
    Ok(DynArray::Bool(typed::logical_and(ba.as_ref(), bb.as_ref())?))
}

#[inline]
pub fn logical_or(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    let ba = dyn_to_bool_arr(a)?;
    let bb = dyn_to_bool_arr(b)?;
    Ok(DynArray::Bool(typed::logical_or(ba.as_ref(), bb.as_ref())?))
}

#[inline]
pub fn logical_xor(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    let ba = dyn_to_bool_arr(a)?;
    let bb = dyn_to_bool_arr(b)?;
    Ok(DynArray::Bool(typed::logical_xor(ba.as_ref(), bb.as_ref())?))
}

#[inline]
pub fn logical_not(a: &DynArray) -> Result<DynArray> {
    dyn_unary_bool_dispatch!(a, logical_not)
}

//! Zero-cost generic / specialized ufunc implementations for `NdArray<T>`.
//!
//! Resolves exact NumPy 2.x output element types at compile time via [`UfuncScalar`],
//! and routes `f64`, `f32`, `i32`, `i64` fast paths to AVX2+FMA kernels in [`crate::simd`]
//! with Rayon chunking (`len >= 32_768`) over `AlignedUninit` outputs.

use crate::exec::{
    bool_all, dtype_err, eval_binary, eval_binary2, eval_binary_simd, eval_ternary, eval_unary,
    eval_unary2, eval_unary_slice, try_eval_binary,
};
use crate::scalar_ops::*;
use crate::simd;
use half::f16;
use num_complex::{Complex, Complex32, Complex64};
use numpy_types::{broadcast_shapes, NdArray, Result, Scalar};
use std::f64::consts::PI;

mod private {
    pub trait Sealed {}
    impl Sealed for bool {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for half::f16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

#[inline]
fn round_signed_neg(x: i128, decimals: i32) -> i128 {
    let neg_d = decimals.saturating_neg() as u32;
    if neg_d >= 20 {
        return 0;
    }
    let step = 10_i128.pow(neg_d);
    let q = x / step;
    let r = (x % step).abs();
    let inc = match (r * 2).cmp(&step) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => (q & 1) != 0,
        std::cmp::Ordering::Less => false,
    };
    let rq = if inc {
        if x >= 0 { q + 1 } else { q - 1 }
    } else {
        q
    };
    rq.wrapping_mul(step)
}

#[inline]
fn round_unsigned_neg(x: u128, decimals: i32) -> u128 {
    let neg_d = decimals.saturating_neg() as u32;
    if neg_d >= 20 {
        return 0;
    }
    let step = 10_u128.pow(neg_d);
    let q = x / step;
    let r = x % step;
    let inc = match (r * 2).cmp(&step) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => (q & 1) != 0,
        std::cmp::Ordering::Less => false,
    };
    let rq = if inc { q + 1 } else { q };
    rq.wrapping_mul(step)
}

#[inline]
fn nan_to_num_f64_val(x: f64, nan: f64, posinf: f64, neginf: f64) -> f64 {
    if x.is_nan() {
        nan
    } else if x == f64::INFINITY {
        posinf
    } else if x == f64::NEG_INFINITY {
        neginf
    } else {
        x
    }
}

#[inline]
fn nan_to_num_f32_val(x: f32, nan: f32, posinf: f32, neginf: f32) -> f32 {
    if x.is_nan() {
        nan
    } else if x == f32::INFINITY {
        posinf
    } else if x == f32::NEG_INFINITY {
        neginf
    } else {
        x
    }
}

#[inline]
fn nan_to_num_f16_val(x: f16, nan: f16, posinf: f16, neginf: f16) -> f16 {
    if x.is_nan() {
        nan
    } else if x == f16::INFINITY {
        posinf
    } else if x == f16::NEG_INFINITY {
        neginf
    } else {
        x
    }
}

#[inline]
fn clip_cmax<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    let gt = if a.re == b.re { a.im > b.im } else { a.re > b.re };
    if a.re.is_nan() || a.im.is_nan() || gt {
        a
    } else {
        b
    }
}

#[inline]
fn clip_cmin<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    let lt = if a.re == b.re { a.im < b.im } else { a.re < b.re };
    if a.re.is_nan() || a.im.is_nan() || lt {
        a
    } else {
        b
    }
}

#[inline]
fn c_clip_elem<T: num_traits::Float>(x: Complex<T>, lo: Complex<T>, hi: Complex<T>) -> Complex<T> {
    clip_cmin(clip_cmax(x, lo), hi)
}

#[inline]
fn clip_arrays_ord<T: UfuncScalar + PartialOrd>(
    a: &NdArray<T>,
    a_min: Option<&NdArray<T>>,
    a_max: Option<&NdArray<T>>,
) -> Result<NdArray<T>> {
    match (a_min, a_max) {
        (None, None) => eval_unary(a, |x| x),
        (Some(lo), None) => T::u_maximum(a, lo),
        (None, Some(hi)) => T::u_minimum(a, hi),
        (Some(lo), Some(hi)) => {
            if lo.len() == 1 && hi.len() == 1 {
                return T::u_clip_scalar(
                    a,
                    Some(unsafe { *lo.storage_ptr().add(lo.offset()) }),
                    Some(unsafe { *hi.storage_ptr().add(hi.offset()) }),
                );
            }
            eval_ternary(a, lo, hi, clip_ord)
        }
    }
}

#[inline]
fn array_equal_exact<T: Scalar + PartialEq>(a: &NdArray<T>, b: &NdArray<T>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    if a.is_empty() {
        return true;
    }
    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        return sa == sb;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

trait AngleConv<O> {
    fn from_f64_ang(v: f64) -> O;
}
impl AngleConv<f16> for i8 {
    #[inline]
    fn from_f64_ang(v: f64) -> f16 {
        f16::from_f64(v)
    }
}
impl AngleConv<f16> for u8 {
    #[inline]
    fn from_f64_ang(v: f64) -> f16 {
        f16::from_f64(v)
    }
}
impl AngleConv<f32> for i16 {
    #[inline]
    fn from_f64_ang(v: f64) -> f32 {
        v as f32
    }
}
impl AngleConv<f32> for u16 {
    #[inline]
    fn from_f64_ang(v: f64) -> f32 {
        v as f32
    }
}
impl AngleConv<f64> for i32 {
    #[inline]
    fn from_f64_ang(v: f64) -> f64 {
        v
    }
}
impl AngleConv<f64> for u32 {
    #[inline]
    fn from_f64_ang(v: f64) -> f64 {
        v
    }
}
impl AngleConv<f64> for i64 {
    #[inline]
    fn from_f64_ang(v: f64) -> f64 {
        v
    }
}
impl AngleConv<f64> for u64 {
    #[inline]
    fn from_f64_ang(v: f64) -> f64 {
        v
    }
}

/// Core trait unifying static zero-cost ufunc specialization across all 14 NumPy scalar types.
pub trait UfuncScalar: Scalar + private::Sealed {
    type FloatOut: Scalar;
    type TrueDivOut: Scalar;
    type BoolToI8Out: Scalar;
    type FloatPowerOut: Scalar;
    type RealOut: Scalar;
    type AngleOut: Scalar;
    type RoundOut: Scalar;

    fn u_add(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_sub(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_mul(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_true_div(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::TrueDivOut>>;
    fn u_floor_div(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_rem(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_fmod(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_divmod(
        a: &NdArray<Self>,
        b: &NdArray<Self>,
    ) -> Result<(NdArray<Self::BoolToI8Out>, NdArray<Self::BoolToI8Out>)>;
    fn u_pow(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_float_pow(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatPowerOut>>;
    fn u_neg(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_pos(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_abs(a: &NdArray<Self>) -> Result<NdArray<Self::RealOut>>;
    fn u_fabs(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_square(a: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_recip(a: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_sqrt(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_cbrt(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;

    fn u_exp(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_exp2(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_expm1(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_log(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_log2(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_log10(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_log1p(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_logaddexp(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_logaddexp2(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;

    fn u_sin(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_cos(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_tan(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arcsin(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arccos(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arctan(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arctan2(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_hypot(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_sinh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_cosh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_tanh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arcsinh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arccosh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_arctanh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_deg2rad(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_rad2deg(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;

    fn u_floor(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_ceil(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_trunc(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_rint(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_round(a: &NdArray<Self>, decimals: i32) -> Result<NdArray<Self::RoundOut>>;
    fn u_sign(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_signbit(a: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_copysign(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_spacing(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_nextafter(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_to_ldexp_exp(self) -> Result<i64>;
    fn u_ldexp_with_i64(a: &NdArray<Self>, exp: &NdArray<i64>) -> Result<NdArray<Self::FloatOut>>;
    fn u_frexp(a: &NdArray<Self>) -> Result<(NdArray<Self::FloatOut>, NdArray<i32>)>;
    fn u_modf(a: &NdArray<Self>) -> Result<(NdArray<Self::FloatOut>, NdArray<Self::FloatOut>)>;
    fn u_isfinite(a: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_isinf(a: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_isnan(a: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_isneginf(a: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_isposinf(a: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_nan_to_num(
        a: &NdArray<Self>,
        nan: Option<f64>,
        posinf: Option<f64>,
        neginf: Option<f64>,
    ) -> Result<NdArray<Self>>;
    fn u_clip_scalar(
        a: &NdArray<Self>,
        a_min: Option<Self>,
        a_max: Option<Self>,
    ) -> Result<NdArray<Self>>;
    fn u_clip_arrays(
        a: &NdArray<Self>,
        a_min: Option<&NdArray<Self>>,
        a_max: Option<&NdArray<Self>>,
    ) -> Result<NdArray<Self>>;
    fn u_heaviside(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>>;
    fn u_sinc(a: &NdArray<Self>) -> Result<NdArray<Self::TrueDivOut>>;
    fn u_i0(a: &NdArray<Self>) -> Result<NdArray<Self::TrueDivOut>>;
    fn u_maximum(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_minimum(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_fmax(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_fmin(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_gcd(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_lcm(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;

    fn u_real(a: &NdArray<Self>) -> Result<NdArray<Self::RealOut>>;
    fn u_imag(a: &NdArray<Self>) -> Result<NdArray<Self::RealOut>>;
    fn u_conj(a: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_angle(a: &NdArray<Self>, deg: bool) -> Result<NdArray<Self::AngleOut>>;
    fn u_bitand(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_bitor(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_bitxor(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_invert(a: &NdArray<Self>) -> Result<NdArray<Self>>;
    fn u_lshift(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_rshift(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::BoolToI8Out>>;
    fn u_bitwise_count(a: &NdArray<Self>) -> Result<NdArray<u8>>;
    fn u_eq(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_ne(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_lt(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_le(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_gt(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_ge(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>>;
    fn u_isclose(
        a: &NdArray<Self>,
        b: &NdArray<Self>,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<NdArray<bool>>;
    fn u_array_equal(a: &NdArray<Self>, b: &NdArray<Self>, equal_nan: bool) -> bool;
    fn u_logical_not(a: &NdArray<Self>) -> Result<NdArray<bool>>;
}

// ============================================================================
// FloatOut via f64 helper macro (for bool, i8..u64)
// ============================================================================

macro_rules! impl_float_out_via_f64 {
    ($conv_in:expr, $conv_out:expr) => {
        fn u_sqrt(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).sqrt()))
        }
        fn u_cbrt(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).cbrt()))
        }
        fn u_exp(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).exp()))
        }
        fn u_exp2(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).exp2()))
        }
        fn u_expm1(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).exp_m1()))
        }
        fn u_log(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).ln()))
        }
        fn u_log2(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).log2()))
        }
        fn u_log10(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).log10()))
        }
        fn u_log1p(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).ln_1p()))
        }
        fn u_logaddexp(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_binary(a, b, |x, y| $conv_out(logaddexp_f64($conv_in(x), $conv_in(y))))
        }
        fn u_logaddexp2(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_binary(a, b, |x, y| $conv_out(logaddexp2_f64($conv_in(x), $conv_in(y))))
        }
        fn u_sin(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).sin()))
        }
        fn u_cos(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).cos()))
        }
        fn u_tan(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).tan()))
        }
        fn u_arcsin(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).asin()))
        }
        fn u_arccos(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).acos()))
        }
        fn u_arctan(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).atan()))
        }
        fn u_arctan2(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_binary(a, b, |x, y| $conv_out(($conv_in(x)).atan2($conv_in(y))))
        }
        fn u_hypot(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_binary(a, b, |x, y| $conv_out(($conv_in(x)).hypot($conv_in(y))))
        }
        fn u_sinh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).sinh()))
        }
        fn u_cosh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).cosh()))
        }
        fn u_tanh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).tanh()))
        }
        fn u_arcsinh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).asinh()))
        }
        fn u_arccosh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).acosh()))
        }
        fn u_arctanh(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).atanh()))
        }
        fn u_deg2rad(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).to_radians()))
        }
        fn u_rad2deg(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).to_degrees()))
        }
        fn u_rint(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).round_ties_even()))
        }
        fn u_copysign(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_binary(a, b, |x, y| $conv_out(($conv_in(x)).copysign($conv_in(y))))
        }
        fn u_heaviside(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_binary(a, b, |x, y| $conv_out(heaviside_f64($conv_in(x), $conv_in(y))))
        }
        fn u_fabs(a: &NdArray<Self>) -> Result<NdArray<Self::FloatOut>> {
            eval_unary(a, |x| $conv_out(($conv_in(x)).abs()))
        }
    };
}

macro_rules! impl_int_float_promote {
    (f16) => {
        impl_float_out_via_f64!(|v: Self| v as f64, f16::from_f64);
        fn u_spacing(a: &NdArray<Self>) -> Result<NdArray<f16>> {
            eval_unary(a, |x| spacing_f16(f16::from_f64(x as f64)))
        }
        fn u_nextafter(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<f16>> {
            eval_binary(a, b, |x, y| {
                nextafter_f16(f16::from_f64(x as f64), f16::from_f64(y as f64))
            })
        }
        fn u_ldexp_with_i64(a: &NdArray<Self>, exp: &NdArray<i64>) -> Result<NdArray<f16>> {
            eval_binary(a, exp, |x, e| f16::from_f32(ldexp_f32(x as f32, e)))
        }
        fn u_frexp(a: &NdArray<Self>) -> Result<(NdArray<f16>, NdArray<i32>)> {
            eval_unary2(a, |x| {
                let (m, e) = frexp_f32(x as f32);
                (f16::from_f32(m), e)
            })
        }
        fn u_modf(a: &NdArray<Self>) -> Result<(NdArray<f16>, NdArray<f16>)> {
            eval_unary2(a, |x| {
                let (fr, it) = modf_f32(x as f32);
                (f16::from_f32(fr), f16::from_f32(it))
            })
        }
    };
    (f32) => {
        impl_float_out_via_f64!(|v: Self| v as f64, |v: f64| v as f32);
        fn u_spacing(a: &NdArray<Self>) -> Result<NdArray<f32>> {
            eval_unary(a, |x| spacing_f32(x as f32))
        }
        fn u_nextafter(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<f32>> {
            eval_binary(a, b, |x, y| nextafter_f32(x as f32, y as f32))
        }
        fn u_ldexp_with_i64(a: &NdArray<Self>, exp: &NdArray<i64>) -> Result<NdArray<f32>> {
            eval_binary(a, exp, |x, e| ldexp_f32(x as f32, e))
        }
        fn u_frexp(a: &NdArray<Self>) -> Result<(NdArray<f32>, NdArray<i32>)> {
            eval_unary2(a, |x| frexp_f32(x as f32))
        }
        fn u_modf(a: &NdArray<Self>) -> Result<(NdArray<f32>, NdArray<f32>)> {
            eval_unary2(a, |x| modf_f32(x as f32))
        }
    };
    (f64) => {
        impl_float_out_via_f64!(|v: Self| v as f64, |v: f64| v);
        fn u_spacing(a: &NdArray<Self>) -> Result<NdArray<f64>> {
            eval_unary(a, |x| spacing_f64(x as f64))
        }
        fn u_nextafter(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<f64>> {
            eval_binary(a, b, |x, y| nextafter_f64(x as f64, y as f64))
        }
        fn u_ldexp_with_i64(a: &NdArray<Self>, exp: &NdArray<i64>) -> Result<NdArray<f64>> {
            eval_binary(a, exp, |x, e| ldexp_f64(x as f64, e))
        }
        fn u_frexp(a: &NdArray<Self>) -> Result<(NdArray<f64>, NdArray<i32>)> {
            eval_unary2(a, |x| frexp_f64(x as f64))
        }
        fn u_modf(a: &NdArray<Self>) -> Result<(NdArray<f64>, NdArray<f64>)> {
            eval_unary2(a, |x| modf_f64(x as f64))
        }
    };
}

// ============================================================================
// bool
// ============================================================================

impl UfuncScalar for bool {
    type FloatOut = f16;
    type TrueDivOut = f64;
    type BoolToI8Out = i8;
    type FloatPowerOut = f64;
    type RealOut = bool;
    type AngleOut = f64;
    type RoundOut = f16;

    fn u_add(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x || y)
    }
    fn u_sub(_a: &NdArray<bool>, _b: &NdArray<bool>) -> Result<NdArray<bool>> {
        Err(dtype_err("subtract", "boolean inputs"))
    }
    fn u_mul(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x && y)
    }
    fn u_true_div(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<f64>> {
        eval_binary(a, b, |x, y| (x as u8 as f64) / (y as u8 as f64))
    }
    fn u_floor_div(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_binary(a, b, |x, y| floor_div_i8(x as i8, y as i8))
    }
    fn u_rem(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_binary(a, b, |x, y| floor_rem_i8(x as i8, y as i8))
    }
    fn u_fmod(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_binary(a, b, |x, y| {
            let yi = y as i8;
            if yi == 0 { 0 } else { (x as i8) % yi }
        })
    }
    fn u_divmod(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<(NdArray<i8>, NdArray<i8>)> {
        eval_binary2(a, b, |x, y| divmod_i8(x as i8, y as i8))
    }
    fn u_pow(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<i8>> {
        try_eval_binary(a, b, |x, y| pow_i8(x as i8, y as i8))
    }
    fn u_float_pow(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<f64>> {
        eval_binary(a, b, |x, y| (x as u8 as f64).powf(y as u8 as f64))
    }
    fn u_neg(_a: &NdArray<bool>) -> Result<NdArray<bool>> {
        Err(dtype_err("negative", "boolean inputs"))
    }
    fn u_pos(_a: &NdArray<bool>) -> Result<NdArray<bool>> {
        Err(dtype_err("positive", "boolean inputs"))
    }
    fn u_abs(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x)
    }
    fn u_square(a: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_unary(a, |x| x as i8)
    }
    fn u_recip(a: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_unary(a, |x| int_recip_i8(x as i8))
    }
    impl_float_out_via_f64!(|v: bool| (v as u8) as f64, f16::from_f64);
    fn u_floor(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x)
    }
    fn u_ceil(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x)
    }
    fn u_trunc(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x)
    }
    fn u_round(a: &NdArray<bool>, decimals: i32) -> Result<NdArray<f16>> {
        if decimals != 0 {
            return Err(dtype_err("round", "boolean with non-zero decimals"));
        }
        eval_unary(a, |x| if x { f16::ONE } else { f16::ZERO })
    }
    fn u_sign(_a: &NdArray<bool>) -> Result<NdArray<bool>> {
        Err(dtype_err("sign", "boolean inputs"))
    }
    fn u_signbit(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |_| false)
    }
    fn u_spacing(a: &NdArray<bool>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| spacing_f16(if x { f16::ONE } else { f16::ZERO }))
    }
    fn u_nextafter(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| {
            nextafter_f16(
                if x { f16::ONE } else { f16::ZERO },
                if y { f16::ONE } else { f16::ZERO },
            )
        })
    }
    fn u_to_ldexp_exp(self) -> Result<i64> {
        Ok(self as i64)
    }
    fn u_ldexp_with_i64(a: &NdArray<bool>, exp: &NdArray<i64>) -> Result<NdArray<f16>> {
        eval_binary(a, exp, |x, e| f16::from_f32(ldexp_f32((x as u8) as f32, e)))
    }
    fn u_frexp(a: &NdArray<bool>) -> Result<(NdArray<f16>, NdArray<i32>)> {
        eval_unary2(a, |x| {
            let (m, e) = frexp_f32((x as u8) as f32);
            (f16::from_f32(m), e)
        })
    }
    fn u_modf(a: &NdArray<bool>) -> Result<(NdArray<f16>, NdArray<f16>)> {
        eval_unary2(a, |x| {
            let (fr, it) = modf_f32((x as u8) as f32);
            (f16::from_f32(fr), f16::from_f32(it))
        })
    }
    fn u_isfinite(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |_| true)
    }
    fn u_isinf(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |_| false)
    }
    fn u_isnan(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |_| false)
    }
    fn u_isneginf(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |_| false)
    }
    fn u_isposinf(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |_| false)
    }
    fn u_nan_to_num(
        a: &NdArray<bool>,
        _nan: Option<f64>,
        _posinf: Option<f64>,
        _neginf: Option<f64>,
    ) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x)
    }
    fn u_clip_scalar(
        a: &NdArray<bool>,
        a_min: Option<bool>,
        a_max: Option<bool>,
    ) -> Result<NdArray<bool>> {
        match (a_min, a_max) {
            (None, None) => eval_unary(a, |x| x),
            (Some(lo), None) => eval_unary(a, |x| x | lo),
            (None, Some(hi)) => eval_unary(a, |x| x & hi),
            (Some(lo), Some(hi)) => eval_unary(a, |x| clip_ord(x, lo, hi)),
        }
    }
    fn u_clip_arrays(
        a: &NdArray<bool>,
        a_min: Option<&NdArray<bool>>,
        a_max: Option<&NdArray<bool>>,
    ) -> Result<NdArray<bool>> {
        clip_arrays_ord(a, a_min, a_max)
    }
    fn u_sinc(a: &NdArray<bool>) -> Result<NdArray<f64>> {
        eval_unary(a, |x| sinc_f64((x as u8) as f64))
    }
    fn u_i0(a: &NdArray<bool>) -> Result<NdArray<f64>> {
        eval_unary(a, |x| i0_f64((x as u8) as f64))
    }
    fn u_maximum(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x | y)
    }
    fn u_minimum(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x & y)
    }
    fn u_fmax(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x | y)
    }
    fn u_fmin(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x & y)
    }
    fn u_gcd(_a: &NdArray<bool>, _b: &NdArray<bool>) -> Result<NdArray<bool>> {
        Err(dtype_err("gcd", "boolean inputs"))
    }
    fn u_lcm(_a: &NdArray<bool>, _b: &NdArray<bool>) -> Result<NdArray<bool>> {
        Err(dtype_err("lcm", "boolean inputs"))
    }
    fn u_real(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x)
    }
    fn u_imag(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        NdArray::zeros(a.shape())
    }
    fn u_conj(a: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_unary(a, |x| x as i8)
    }
    fn u_angle(a: &NdArray<bool>, _deg: bool) -> Result<NdArray<f64>> {
        eval_unary(a, |_| 0.0)
    }
    fn u_bitand(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x & y)
    }
    fn u_bitor(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x | y)
    }
    fn u_bitxor(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x ^ y)
    }
    fn u_invert(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| !x)
    }
    fn u_lshift(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_binary(a, b, |x, y| lshift_i8(x as i8, y as i8))
    }
    fn u_rshift(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<i8>> {
        eval_binary(a, b, |x, y| rshift_i8(x as i8, y as i8))
    }
    fn u_bitwise_count(a: &NdArray<bool>) -> Result<NdArray<u8>> {
        eval_unary(a, |x| x as u8)
    }
    fn u_eq(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x == y)
    }
    fn u_ne(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x != y)
    }
    fn u_lt(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| !x & y)
    }
    fn u_le(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x <= y)
    }
    fn u_gt(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x & !y)
    }
    fn u_ge(a: &NdArray<bool>, b: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x >= y)
    }
    fn u_isclose(
        a: &NdArray<bool>,
        b: &NdArray<bool>,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| {
            isclose_f64_elem((x as u8) as f64, (y as u8) as f64, rtol, atol, equal_nan)
        })
    }
    fn u_array_equal(a: &NdArray<bool>, b: &NdArray<bool>, _equal_nan: bool) -> bool {
        array_equal_exact(a, b)
    }
    fn u_logical_not(a: &NdArray<bool>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| !x)
    }
}

// ============================================================================
// Integer macro (with scalar variants for i8/i16/u8/u16/u32/u64, and SIMD for i32/i64)
// ============================================================================

macro_rules! impl_int_common_body {
    (
        $t:ty, $float_tag:ident, $float_out:ty, $signed:expr,
        $fdiv:ident, $frem:ident, $divmod:ident, $gcd:ident, $lcm:ident, $recip:ident
    ) => {
        type FloatOut = $float_out;
        type TrueDivOut = f64;
        type BoolToI8Out = $t;
        type FloatPowerOut = f64;
        type RealOut = $t;
        type AngleOut = $float_out;
        type RoundOut = $t;

        fn u_true_div(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<f64>> {
            eval_binary(a, b, |x, y| (x as f64) / (y as f64))
        }
        fn u_floor_div(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, $fdiv)
        }
        fn u_rem(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, $frem)
        }
        fn u_fmod(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| if y == 0 { 0 } else { x.wrapping_rem(y) })
        }
        fn u_divmod(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<(NdArray<$t>, NdArray<$t>)> {
            eval_binary2(a, b, $divmod)
        }
        fn u_float_pow(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<f64>> {
            eval_binary(a, b, |x, y| (x as f64).powf(y as f64))
        }
        fn u_pos(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_recip(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, $recip)
        }
        impl_int_float_promote!($float_tag);
        fn u_floor(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_ceil(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_trunc(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_round(a: &NdArray<Self>, decimals: i32) -> Result<NdArray<$t>> {
            if decimals >= 0 {
                return eval_unary(a, |x| x);
            }
            if $signed {
                eval_unary(a, |x| round_signed_neg(x as i128, decimals) as $t)
            } else {
                eval_unary(a, |x| round_unsigned_neg(x as u128, decimals) as $t)
            }
        }
        fn u_sign(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            if $signed {
                eval_unary(a, |x| {
                    let xi = x as i64;
                    if xi > 0 { 1 } else if xi < 0 { (!0) as $t } else { 0 }
                })
            } else {
                eval_unary(a, |x| if x > 0 { 1 } else { 0 })
            }
        }
        fn u_signbit(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            if $signed {
                eval_unary(a, |x| (x as i64) < 0)
            } else {
                eval_unary(a, |_| false)
            }
        }
        fn u_isfinite(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_unary(a, |_| true)
        }
        fn u_isinf(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_unary(a, |_| false)
        }
        fn u_isnan(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_unary(a, |_| false)
        }
        fn u_isneginf(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_unary(a, |_| false)
        }
        fn u_isposinf(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_unary(a, |_| false)
        }
        fn u_nan_to_num(
            a: &NdArray<Self>,
            _nan: Option<f64>,
            _posinf: Option<f64>,
            _neginf: Option<f64>,
        ) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_clip_arrays(
            a: &NdArray<Self>,
            a_min: Option<&NdArray<Self>>,
            a_max: Option<&NdArray<Self>>,
        ) -> Result<NdArray<$t>> {
            clip_arrays_ord(a, a_min, a_max)
        }
        fn u_sinc(a: &NdArray<Self>) -> Result<NdArray<f64>> {
            eval_unary(a, |x| sinc_f64(x as f64))
        }
        fn u_i0(a: &NdArray<Self>) -> Result<NdArray<f64>> {
            eval_unary(a, |x| i0_f64(x as f64))
        }
        fn u_fmax(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            Self::u_maximum(a, b)
        }
        fn u_fmin(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            Self::u_minimum(a, b)
        }
        fn u_gcd(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, $gcd)
        }
        fn u_lcm(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, $lcm)
        }
        fn u_real(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_imag(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            NdArray::zeros(a.shape())
        }
        fn u_conj(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x)
        }
        fn u_angle(a: &NdArray<Self>, deg: bool) -> Result<NdArray<$float_out>> {
            let factor = if deg { 180.0 / PI } else { 1.0 };
            if $signed {
                eval_unary(a, |x| {
                    let ang = if (x as i64) < 0 { PI * factor } else { 0.0 };
                    <Self as AngleConv<$float_out>>::from_f64_ang(ang)
                })
            } else {
                eval_unary(a, |_| <Self as AngleConv<$float_out>>::from_f64_ang(0.0))
            }
        }
        fn u_bitwise_count(a: &NdArray<Self>) -> Result<NdArray<u8>> {
            if $signed {
                eval_unary(a, |x| (x as i64).unsigned_abs().count_ones() as u8)
            } else {
                eval_unary(a, |x| x.count_ones() as u8)
            }
        }
        fn u_isclose(
            a: &NdArray<Self>,
            b: &NdArray<Self>,
            rtol: f64,
            atol: f64,
            equal_nan: bool,
        ) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| {
                isclose_f64_elem(x as f64, y as f64, rtol, atol, equal_nan)
            })
        }
        fn u_array_equal(a: &NdArray<Self>, b: &NdArray<Self>, _equal_nan: bool) -> bool {
            array_equal_exact(a, b)
        }
    };
}

macro_rules! impl_scalar_int_full {
    (
        signed, $t:ty, $float_tag:ident, $float_out:ty,
        $fdiv:ident, $frem:ident, $divmod:ident, $pow_fn:ident,
        $lsh:ident, $rsh:ident, $gcd:ident, $lcm:ident, $recip:ident
    ) => {
        impl UfuncScalar for $t {
            impl_int_common_body!(
                $t, $float_tag, $float_out, true, $fdiv, $frem, $divmod, $gcd, $lcm, $recip
            );
            fn u_pow(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
                try_eval_binary(a, b, $pow_fn)
            }
            fn u_neg(a: &NdArray<Self>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x.wrapping_neg())
            }
            fn u_abs(a: &NdArray<Self>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x.wrapping_abs())
            }
            fn u_to_ldexp_exp(self) -> Result<i64> {
                Ok(self as i64)
            }
            impl_scalar_int_ops_rest!($t, $lsh, $rsh);
        }
    };
    (
        unsigned, $t:ty, $float_tag:ident, $float_out:ty, $allow_ldexp_exp:expr,
        $fdiv:ident, $frem:ident, $divmod:ident, $pow_fn:ident,
        $lsh:ident, $rsh:ident, $gcd:ident, $lcm:ident, $recip:ident
    ) => {
        impl UfuncScalar for $t {
            impl_int_common_body!(
                $t, $float_tag, $float_out, false, $fdiv, $frem, $divmod, $gcd, $lcm, $recip
            );
            fn u_pow(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $pow_fn)
            }
            fn u_neg(a: &NdArray<Self>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x.wrapping_neg())
            }
            fn u_abs(a: &NdArray<Self>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x)
            }
            fn u_to_ldexp_exp(self) -> Result<i64> {
                if $allow_ldexp_exp {
                    Ok(self as i64)
                } else {
                    Err(dtype_err("ldexp", "uint64 exponent"))
                }
            }
            impl_scalar_int_ops_rest!($t, $lsh, $rsh);
        }
    };
}

macro_rules! impl_scalar_int_ops_rest {
    ($t:ty, $lsh:ident, $rsh:ident) => {
        fn u_add(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x.wrapping_add(y))
        }
        fn u_sub(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x.wrapping_sub(y))
        }
        fn u_mul(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x.wrapping_mul(y))
        }
        fn u_square(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| x.wrapping_mul(x))
        }
        fn u_clip_scalar(
            a: &NdArray<Self>,
            a_min: Option<$t>,
            a_max: Option<$t>,
        ) -> Result<NdArray<$t>> {
            match (a_min, a_max) {
                (None, None) => eval_unary(a, |x| x),
                (Some(lo), None) => eval_unary(a, |x| x.max(lo)),
                (None, Some(hi)) => eval_unary(a, |x| x.min(hi)),
                (Some(lo), Some(hi)) => eval_unary(a, |x| clip_ord(x, lo, hi)),
            }
        }
        fn u_maximum(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x.max(y))
        }
        fn u_minimum(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x.min(y))
        }
        fn u_bitand(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x & y)
        }
        fn u_bitor(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x | y)
        }
        fn u_bitxor(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, |x, y| x ^ y)
        }
        fn u_invert(a: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_unary(a, |x| !x)
        }
        fn u_lshift(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, $lsh)
        }
        fn u_rshift(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
            eval_binary(a, b, $rsh)
        }
        fn u_eq(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| x == y)
        }
        fn u_ne(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| x != y)
        }
        fn u_lt(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| x < y)
        }
        fn u_le(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| x <= y)
        }
        fn u_gt(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| x > y)
        }
        fn u_ge(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_binary(a, b, |x, y| x >= y)
        }
        fn u_logical_not(a: &NdArray<Self>) -> Result<NdArray<bool>> {
            eval_unary(a, |x| x == 0)
        }
    };
}

impl_scalar_int_full!(
    signed, i8, f16, f16, floor_div_i8, floor_rem_i8, divmod_i8, pow_i8,
    lshift_i8, rshift_i8, gcd_i8, lcm_i8, int_recip_i8
);
impl_scalar_int_full!(
    signed, i16, f32, f32, floor_div_i16, floor_rem_i16, divmod_i16, pow_i16,
    lshift_i16, rshift_i16, gcd_i16, lcm_i16, int_recip_i16
);
impl_scalar_int_full!(
    unsigned, u8, f16, f16, true, floor_div_u8, floor_rem_u8, divmod_u8, pow_u8,
    lshift_u8, rshift_u8, gcd_u8, lcm_u8, int_recip_u8
);
impl_scalar_int_full!(
    unsigned, u16, f32, f32, true, floor_div_u16, floor_rem_u16, divmod_u16, pow_u16,
    lshift_u16, rshift_u16, gcd_u16, lcm_u16, int_recip_u16
);
impl_scalar_int_full!(
    unsigned, u32, f64, f64, true, floor_div_u32, floor_rem_u32, divmod_u32, pow_u32,
    lshift_u32, rshift_u32, gcd_u32, lcm_u32, int_recip_u32
);
impl_scalar_int_full!(
    unsigned, u64, f64, f64, false, floor_div_u64, floor_rem_u64, divmod_u64, pow_u64,
    lshift_u64, rshift_u64, gcd_u64, lcm_u64, int_recip_u64
);

// ============================================================================
// i32 & i64 (with runtime AVX2 kernels)
// ============================================================================

macro_rules! impl_simd_signed_int {
    (
        $t:ty,
        $fdiv:ident, $frem:ident, $divmod:ident, $pow_fn:ident,
        $lsh:ident, $rsh:ident, $gcd:ident, $lcm:ident, $recip:ident,
        $s_add:ident, $s_add_s:ident,
        $s_sub:ident, $s_sub_s:ident, $s_rsub_s:ident,
        $s_mul:ident, $s_mul_s:ident,
        $s_neg:ident, $s_abs:ident, $s_square:ident, $s_inv:ident, $s_lnot:ident,
        $s_and:ident, $s_or:ident, $s_xor:ident, $s_lsh:ident, $rsh_simd:expr, $s_max:ident, $s_min:ident, $s_clip:ident,
        $s_eq:ident, $s_eq_s:ident, $s_ne:ident, $s_ne_s:ident,
        $s_lt:ident, $s_lt_s:ident, $s_le:ident, $s_le_s:ident,
        $s_gt:ident, $s_gt_s:ident, $s_ge:ident, $s_ge_s:ident
    ) => {
        impl UfuncScalar for $t {
            impl_int_common_body!(
                $t, f64, f64, true, $fdiv, $frem, $divmod, $gcd, $lcm, $recip
            );
            fn u_pow(a: &NdArray<Self>, b: &NdArray<Self>) -> Result<NdArray<$t>> {
                try_eval_binary(a, b, $pow_fn)
            }
            fn u_to_ldexp_exp(self) -> Result<i64> {
                Ok(self as i64)
            }
            fn u_add(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_add, simd::$s_add_s,
                    |s, sb, out| simd::$s_add_s(sb, s, out),
                    <$t>::wrapping_add,
                )
            }
            fn u_sub(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_sub, simd::$s_sub_s,
                    |s, sb, out| simd::$s_rsub_s(sb, s, out),
                    <$t>::wrapping_sub,
                )
            }
            fn u_mul(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_mul, simd::$s_mul_s,
                    |s, sb, out| simd::$s_mul_s(sb, s, out),
                    <$t>::wrapping_mul,
                )
            }
            fn u_neg(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_neg, <$t>::wrapping_neg)
            }
            fn u_abs(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_abs, <$t>::wrapping_abs)
            }
            fn u_square(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_square, |x| x.wrapping_mul(x))
            }
            fn u_invert(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_inv, |x| !x)
            }
            fn u_logical_not(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary_slice(a, simd::$s_lnot, |x| x == 0)
            }
            fn u_clip_scalar(
                a: &NdArray<$t>,
                a_min: Option<$t>,
                a_max: Option<$t>,
            ) -> Result<NdArray<$t>> {
                match (a_min, a_max) {
                    (None, None) => eval_unary(a, |x| x),
                    (Some(lo), None) => eval_unary(a, |x| x.max(lo)),
                    (None, Some(hi)) => eval_unary(a, |x| x.min(hi)),
                    (Some(lo), Some(hi)) => {
                        eval_unary_slice(
                            a,
                            |src, out| simd::$s_clip(src, lo, hi, out),
                            |x| clip_ord(x, lo, hi),
                        )
                    }
                }
            }
            fn u_maximum(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_max,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write(x.max(s)); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write(s.max(y)); },
                    <$t>::max,
                )
            }
            fn u_minimum(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_min,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write(x.min(s)); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write(s.min(y)); },
                    <$t>::min,
                )
            }
            fn u_bitand(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_and,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write(x & s); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write(s & y); },
                    |x, y| x & y,
                )
            }
            fn u_bitor(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_or,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write(x | s); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write(s | y); },
                    |x, y| x | y,
                )
            }
            fn u_bitxor(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_xor,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write(x ^ s); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write(s ^ y); },
                    |x, y| x ^ y,
                )
            }
            fn u_lshift(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_lsh,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write($lsh(x, s)); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write($lsh(s, y)); },
                    $lsh,
                )
            }
            fn u_rshift(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, $rsh_simd,
                    |sa, s, out| for (slot, &x) in out.iter_mut().zip(sa.iter()) { slot.write($rsh(x, s)); },
                    |s, sb, out| for (slot, &y) in out.iter_mut().zip(sb.iter()) { slot.write($rsh(s, y)); },
                    $rsh,
                )
            }
            fn u_eq(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_eq, simd::$s_eq_s,
                    |s, sb, out| simd::$s_eq_s(sb, s, out),
                    |x, y| x == y,
                )
            }
            fn u_ne(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_ne, simd::$s_ne_s,
                    |s, sb, out| simd::$s_ne_s(sb, s, out),
                    |x, y| x != y,
                )
            }
            fn u_lt(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_lt, simd::$s_lt_s,
                    |s, sb, out| simd::$s_gt_s(sb, s, out),
                    |x, y| x < y,
                )
            }
            fn u_le(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_le, simd::$s_le_s,
                    |s, sb, out| simd::$s_ge_s(sb, s, out),
                    |x, y| x <= y,
                )
            }
            fn u_gt(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_gt, simd::$s_gt_s,
                    |s, sb, out| simd::$s_lt_s(sb, s, out),
                    |x, y| x > y,
                )
            }
            fn u_ge(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_ge, simd::$s_ge_s,
                    |s, sb, out| simd::$s_le_s(sb, s, out),
                    |x, y| x >= y,
                )
            }
        }
    };
}

impl_simd_signed_int!(
    i32, floor_div_i32, floor_rem_i32, divmod_i32, pow_i32,
    lshift_i32, rshift_i32, gcd_i32, lcm_i32, int_recip_i32,
    add_i32, add_i32_scalar, sub_i32, sub_i32_scalar, rsub_i32_scalar,
    mul_i32, mul_i32_scalar, neg_i32, abs_i32, square_i32, invert_i32, logical_not_i32,
    bitand_i32, bitor_i32, bitxor_i32, lshift_i32, simd::rshift_i32, maximum_i32, minimum_i32, clip_scalar_i32,
    eq_i32, eq_i32_scalar, ne_i32, ne_i32_scalar,
    lt_i32, lt_i32_scalar, le_i32, le_i32_scalar,
    gt_i32, gt_i32_scalar, ge_i32, ge_i32_scalar
);

impl_simd_signed_int!(
    i64, floor_div_i64, floor_rem_i64, divmod_i64, pow_i64,
    lshift_i64, rshift_i64, gcd_i64, lcm_i64, int_recip_i64,
    add_i64, add_i64_scalar, sub_i64, sub_i64_scalar, rsub_i64_scalar,
    mul_i64, mul_i64_scalar, neg_i64, abs_i64, square_i64, invert_i64, logical_not_i64,
    bitand_i64, bitor_i64, bitxor_i64, lshift_i64,
    |sa, sb, out| for ((slot, &x), &y) in out.iter_mut().zip(sa.iter()).zip(sb.iter()) { slot.write(rshift_i64(x, y)); },
    maximum_i64, minimum_i64, clip_scalar_i64,
    eq_i64, eq_i64_scalar, ne_i64, ne_i64_scalar,
    lt_i64, lt_i64_scalar, le_i64, le_i64_scalar,
    gt_i64, gt_i64_scalar, ge_i64, ge_i64_scalar
);

// ============================================================================
// f16
// ============================================================================

impl UfuncScalar for f16 {
    type FloatOut = f16;
    type TrueDivOut = f16;
    type BoolToI8Out = f16;
    type FloatPowerOut = f64;
    type RealOut = f16;
    type AngleOut = f16;
    type RoundOut = f16;

    fn u_add(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32() + y.to_f32()))
    }
    fn u_sub(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32() - y.to_f32()))
    }
    fn u_mul(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32() * y.to_f32()))
    }
    fn u_true_div(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32() / y.to_f32()))
    }
    fn u_floor_div(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(floor_divide_f32(x.to_f32(), y.to_f32())))
    }
    fn u_rem(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(remainder_f32(x.to_f32(), y.to_f32())))
    }
    fn u_fmod(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32() % y.to_f32()))
    }
    fn u_divmod(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<(NdArray<f16>, NdArray<f16>)> {
        eval_binary2(a, b, |x, y| {
            let (q, r) = divmod_f32(x.to_f32(), y.to_f32());
            (f16::from_f32(q), f16::from_f32(r))
        })
    }
    fn u_pow(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32().powf(y.to_f32())))
    }
    fn u_float_pow(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f64>> {
        eval_binary(a, b, |x, y| x.to_f64().powf(y.to_f64()))
    }
    fn u_neg(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_bits(x.to_bits() ^ 0x8000))
    }
    fn u_pos(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| x)
    }
    fn u_abs(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_bits(x.to_bits() & 0x7FFF))
    }
    fn u_fabs(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        Self::u_abs(a)
    }
    fn u_square(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| {
            let v = x.to_f32();
            f16::from_f32(v * v)
        })
    }
    fn u_recip(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(1.0_f32 / x.to_f32()))
    }
    fn u_sqrt(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().sqrt()))
    }
    fn u_cbrt(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().cbrt()))
    }
    fn u_exp(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().exp()))
    }
    fn u_exp2(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().exp2()))
    }
    fn u_expm1(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().exp_m1()))
    }
    fn u_log(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().ln()))
    }
    fn u_log2(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().log2()))
    }
    fn u_log10(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().log10()))
    }
    fn u_log1p(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().ln_1p()))
    }
    fn u_logaddexp(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(logaddexp_f32(x.to_f32(), y.to_f32())))
    }
    fn u_logaddexp2(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(logaddexp2_f32(x.to_f32(), y.to_f32())))
    }
    fn u_sin(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().sin()))
    }
    fn u_cos(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().cos()))
    }
    fn u_tan(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().tan()))
    }
    fn u_arcsin(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().asin()))
    }
    fn u_arccos(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().acos()))
    }
    fn u_arctan(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().atan()))
    }
    fn u_arctan2(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32().atan2(y.to_f32())))
    }
    fn u_hypot(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(x.to_f32().hypot(y.to_f32())))
    }
    fn u_sinh(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().sinh()))
    }
    fn u_cosh(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().cosh()))
    }
    fn u_tanh(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().tanh()))
    }
    fn u_arcsinh(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().asinh()))
    }
    fn u_arccosh(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().acosh()))
    }
    fn u_arctanh(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().atanh()))
    }
    fn u_deg2rad(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().to_radians()))
    }
    fn u_rad2deg(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().to_degrees()))
    }
    fn u_floor(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().floor()))
    }
    fn u_ceil(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().ceil()))
    }
    fn u_trunc(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().trunc()))
    }
    fn u_rint(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(x.to_f32().round_ties_even()))
    }
    fn u_round(a: &NdArray<f16>, decimals: i32) -> Result<NdArray<f16>> {
        eval_unary(a, |x| round_f16(x, decimals))
    }
    fn u_sign(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f32(sign_f32(x.to_f32())))
    }
    fn u_signbit(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| (x.to_bits() & 0x8000) != 0)
    }
    fn u_copysign(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| {
            f16::from_bits((x.to_bits() & 0x7FFF) | (y.to_bits() & 0x8000))
        })
    }
    fn u_spacing(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, spacing_f16)
    }
    fn u_nextafter(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, nextafter_f16)
    }
    fn u_to_ldexp_exp(self) -> Result<i64> {
        Err(dtype_err("ldexp", "float16 exponent"))
    }
    fn u_ldexp_with_i64(a: &NdArray<f16>, exp: &NdArray<i64>) -> Result<NdArray<f16>> {
        eval_binary(a, exp, |x, e| f16::from_f32(ldexp_f32(x.to_f32(), e)))
    }
    fn u_frexp(a: &NdArray<f16>) -> Result<(NdArray<f16>, NdArray<i32>)> {
        eval_unary2(a, |x| {
            let (m, e) = frexp_f32(x.to_f32());
            (f16::from_f32(m), e)
        })
    }
    fn u_modf(a: &NdArray<f16>) -> Result<(NdArray<f16>, NdArray<f16>)> {
        eval_unary2(a, |x| {
            let (fr, it) = modf_f32(x.to_f32());
            (f16::from_f32(fr), f16::from_f32(it))
        })
    }
    fn u_isfinite(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, f16::is_finite)
    }
    fn u_isinf(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, f16::is_infinite)
    }
    fn u_isnan(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, f16::is_nan)
    }
    fn u_isneginf(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x == f16::NEG_INFINITY)
    }
    fn u_isposinf(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x == f16::INFINITY)
    }
    fn u_nan_to_num(
        a: &NdArray<f16>,
        nan: Option<f64>,
        posinf: Option<f64>,
        neginf: Option<f64>,
    ) -> Result<NdArray<f16>> {
        let nv = f16::from_f64(nan.unwrap_or(0.0));
        let pv = posinf.map(f16::from_f64).unwrap_or(f16::MAX);
        let qv = neginf.map(f16::from_f64).unwrap_or(f16::MIN);
        eval_unary(a, |x| nan_to_num_f16_val(x, nv, pv, qv))
    }
    fn u_clip_scalar(
        a: &NdArray<f16>,
        a_min: Option<f16>,
        a_max: Option<f16>,
    ) -> Result<NdArray<f16>> {
        match (a_min, a_max) {
            (None, None) => eval_unary(a, |x| x),
            (Some(lo), None) => eval_unary(a, |x| max_prop_nan_f16(x, lo)),
            (None, Some(hi)) => eval_unary(a, |x| min_prop_nan_f16(x, hi)),
            (Some(lo), Some(hi)) => eval_unary(a, |x| clip_f16_elem(x, lo, hi)),
        }
    }
    fn u_clip_arrays(
        a: &NdArray<f16>,
        a_min: Option<&NdArray<f16>>,
        a_max: Option<&NdArray<f16>>,
    ) -> Result<NdArray<f16>> {
        match (a_min, a_max) {
            (None, None) => eval_unary(a, |x| x),
            (Some(lo), None) => Self::u_maximum(a, lo),
            (None, Some(hi)) => Self::u_minimum(a, hi),
            (Some(lo), Some(hi)) => eval_ternary(a, lo, hi, clip_f16_elem),
        }
    }
    fn u_heaviside(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, |x, y| f16::from_f32(heaviside_f32(x.to_f32(), y.to_f32())))
    }
    fn u_sinc(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, sinc_f16)
    }
    fn u_i0(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| f16::from_f64(i0_f64(x.to_f64())))
    }
    fn u_maximum(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, max_prop_nan_f16)
    }
    fn u_minimum(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, min_prop_nan_f16)
    }
    fn u_fmax(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, fmax_f16)
    }
    fn u_fmin(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_binary(a, b, fmin_f16)
    }
    fn u_gcd(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("gcd", "float16 inputs"))
    }
    fn u_lcm(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("lcm", "float16 inputs"))
    }
    fn u_real(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| x)
    }
    fn u_imag(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        NdArray::zeros(a.shape())
    }
    fn u_conj(a: &NdArray<f16>) -> Result<NdArray<f16>> {
        eval_unary(a, |x| x)
    }
    fn u_angle(a: &NdArray<f16>, deg: bool) -> Result<NdArray<f16>> {
        let factor = if deg { 180.0 / PI } else { 1.0 };
        eval_unary(a, |x| f16::from_f64(0.0_f64.atan2(x.to_f64()) * factor))
    }
    fn u_bitand(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("bitwise_and", "float16 inputs"))
    }
    fn u_bitor(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("bitwise_or", "float16 inputs"))
    }
    fn u_bitxor(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("bitwise_xor", "float16 inputs"))
    }
    fn u_invert(_a: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("invert", "float16 inputs"))
    }
    fn u_lshift(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("left_shift", "float16 inputs"))
    }
    fn u_rshift(_a: &NdArray<f16>, _b: &NdArray<f16>) -> Result<NdArray<f16>> {
        Err(dtype_err("right_shift", "float16 inputs"))
    }
    fn u_bitwise_count(_a: &NdArray<f16>) -> Result<NdArray<u8>> {
        Err(dtype_err("bitwise_count", "float16 inputs"))
    }
    fn u_eq(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x == y)
    }
    fn u_ne(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x != y)
    }
    fn u_lt(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x < y)
    }
    fn u_le(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x <= y)
    }
    fn u_gt(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x > y)
    }
    fn u_ge(a: &NdArray<f16>, b: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| x >= y)
    }
    fn u_isclose(
        a: &NdArray<f16>,
        b: &NdArray<f16>,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<NdArray<bool>> {
        eval_binary(a, b, |x, y| {
            isclose_f64_elem(x.to_f64(), y.to_f64(), rtol, atol, equal_nan)
        })
    }
    fn u_array_equal(a: &NdArray<f16>, b: &NdArray<f16>, equal_nan: bool) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        if !equal_nan {
            return array_equal_exact(a, b);
        }
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }
    fn u_logical_not(a: &NdArray<f16>) -> Result<NdArray<bool>> {
        eval_unary(a, |x| x.to_f32() == 0.0)
    }
}

// ============================================================================
// f32 & f64 (runtime AVX2 + FMA kernels)
// ============================================================================

macro_rules! impl_simd_float {
    (
        $t:ty, $name_str:expr,
        $fdiv:ident, $frem:ident, $divmod:ident, $nextafter:ident, $spacing:ident,
        $frexp:ident, $modf:ident, $ldexp:ident, $sign_fn:ident, $logaddexp:ident,
        $logaddexp2:ident, $heaviside:ident, $sinc_fn:ident, $i0_fn:expr,
        $round_fn:ident, $nan_to_num_val:ident, $clip_elem:ident,
        $max_nan:ident, $min_nan:ident, $fmax_fn:ident, $fmin_fn:ident,
        $isclose_elem:ident,
        $s_add:ident, $s_add_s:ident,
        $s_sub:ident, $s_sub_s:ident, $s_rsub_s:ident,
        $s_mul:ident, $s_mul_s:ident,
        $s_div:ident, $s_div_s:ident, $s_rdiv_s:ident,
        $s_max:ident, $s_max_s:ident, $s_rmax_s:ident,
        $s_min:ident, $s_min_s:ident, $s_rmin_s:ident,
        $s_fmax:ident, $s_fmax_s:ident, $s_rfmax_s:ident,
        $s_fmin:ident, $s_fmin_s:ident, $s_rfmin_s:ident,
        $s_neg:ident, $s_abs:ident, $s_square:ident, $s_sqrt:ident, $s_recip:ident,
        $s_floor:ident, $s_ceil:ident, $s_trunc:ident, $s_rint:ident,
        $s_d2r:ident, $s_r2d:ident, $s_clip:ident,
        $s_isfin:ident, $s_isinf:ident, $s_isnan:ident, $s_signbit:ident, $s_lnot:ident,
        $s_eq:ident, $s_eq_s:ident, $s_ne:ident, $s_ne_s:ident,
        $s_lt:ident, $s_lt_s:ident, $s_le:ident, $s_le_s:ident,
        $s_gt:ident, $s_gt_s:ident, $s_ge:ident, $s_ge_s:ident,
        $s_isclose:ident
    ) => {
        impl UfuncScalar for $t {
            type FloatOut = $t;
            type TrueDivOut = $t;
            type BoolToI8Out = $t;
            type FloatPowerOut = f64;
            type RealOut = $t;
            type AngleOut = $t;
            type RoundOut = $t;

            fn u_add(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_add, simd::$s_add_s,
                    |s, sb, out| simd::$s_add_s(sb, s, out),
                    |x, y| x + y,
                )
            }
            fn u_sub(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_sub, simd::$s_sub_s,
                    |s, sb, out| simd::$s_rsub_s(sb, s, out),
                    |x, y| x - y,
                )
            }
            fn u_mul(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_mul, simd::$s_mul_s,
                    |s, sb, out| simd::$s_mul_s(sb, s, out),
                    |x, y| x * y,
                )
            }
            fn u_true_div(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_div, simd::$s_div_s,
                    |s, sb, out| simd::$s_rdiv_s(sb, s, out),
                    |x, y| x / y,
                )
            }
            fn u_floor_div(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $fdiv)
            }
            fn u_rem(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $frem)
            }
            fn u_fmod(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, |x, y| x % y)
            }
            fn u_divmod(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<(NdArray<$t>, NdArray<$t>)> {
                eval_binary2(a, b, $divmod)
            }
            fn u_pow(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, |x, y| x.powf(y))
            }
            fn u_float_pow(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<f64>> {
                eval_binary(a, b, |x, y| (x as f64).powf(y as f64))
            }
            fn u_neg(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_neg, |x| -x)
            }
            fn u_pos(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x)
            }
            fn u_abs(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_abs, <$t>::abs)
            }
            fn u_fabs(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                Self::u_abs(a)
            }
            fn u_square(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_square, |x| x * x)
            }
            fn u_recip(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_recip, |x| 1.0 / x)
            }
            fn u_sqrt(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_sqrt, <$t>::sqrt)
            }
            fn u_cbrt(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::cbrt)
            }
            fn u_exp(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::exp)
            }
            fn u_exp2(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::exp2)
            }
            fn u_expm1(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::exp_m1)
            }
            fn u_log(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::ln)
            }
            fn u_log2(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::log2)
            }
            fn u_log10(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::log10)
            }
            fn u_log1p(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::ln_1p)
            }
            fn u_logaddexp(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $logaddexp)
            }
            fn u_logaddexp2(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $logaddexp2)
            }
            fn u_sin(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::sin)
            }
            fn u_cos(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::cos)
            }
            fn u_tan(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::tan)
            }
            fn u_arcsin(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::asin)
            }
            fn u_arccos(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::acos)
            }
            fn u_arctan(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::atan)
            }
            fn u_arctan2(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, <$t>::atan2)
            }
            fn u_hypot(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, <$t>::hypot)
            }
            fn u_sinh(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::sinh)
            }
            fn u_cosh(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::cosh)
            }
            fn u_tanh(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::tanh)
            }
            fn u_arcsinh(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::asinh)
            }
            fn u_arccosh(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::acosh)
            }
            fn u_arctanh(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, <$t>::atanh)
            }
            fn u_deg2rad(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_d2r, <$t>::to_radians)
            }
            fn u_rad2deg(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_r2d, <$t>::to_degrees)
            }
            fn u_floor(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_floor, <$t>::floor)
            }
            fn u_ceil(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_ceil, <$t>::ceil)
            }
            fn u_trunc(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_trunc, <$t>::trunc)
            }
            fn u_rint(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary_slice(a, simd::$s_rint, <$t>::round_ties_even)
            }
            fn u_round(a: &NdArray<$t>, decimals: i32) -> Result<NdArray<$t>> {
                if decimals == 0 {
                    return Self::u_rint(a);
                }
                eval_unary(a, |x| $round_fn(x, decimals))
            }
            fn u_sign(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, $sign_fn)
            }
            fn u_signbit(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary_slice(a, simd::$s_signbit, <$t>::is_sign_negative)
            }
            fn u_copysign(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, <$t>::copysign)
            }
            fn u_spacing(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, $spacing)
            }
            fn u_nextafter(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $nextafter)
            }
            fn u_to_ldexp_exp(self) -> Result<i64> {
                Err(dtype_err("ldexp", concat!($name_str, " exponent")))
            }
            fn u_ldexp_with_i64(a: &NdArray<$t>, exp: &NdArray<i64>) -> Result<NdArray<$t>> {
                eval_binary(a, exp, $ldexp)
            }
            fn u_frexp(a: &NdArray<$t>) -> Result<(NdArray<$t>, NdArray<i32>)> {
                eval_unary2(a, $frexp)
            }
            fn u_modf(a: &NdArray<$t>) -> Result<(NdArray<$t>, NdArray<$t>)> {
                eval_unary2(a, $modf)
            }
            fn u_isfinite(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary_slice(a, simd::$s_isfin, <$t>::is_finite)
            }
            fn u_isinf(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary_slice(a, simd::$s_isinf, <$t>::is_infinite)
            }
            fn u_isnan(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary_slice(a, simd::$s_isnan, <$t>::is_nan)
            }
            fn u_isneginf(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary(a, |x| x == <$t>::NEG_INFINITY)
            }
            fn u_isposinf(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary(a, |x| x == <$t>::INFINITY)
            }
            fn u_nan_to_num(
                a: &NdArray<$t>,
                nan: Option<f64>,
                posinf: Option<f64>,
                neginf: Option<f64>,
            ) -> Result<NdArray<$t>> {
                let nv = nan.unwrap_or(0.0) as $t;
                let pv = posinf.map(|v| v as $t).unwrap_or(<$t>::MAX);
                let qv = neginf.map(|v| v as $t).unwrap_or(<$t>::MIN);
                eval_unary(a, |x| $nan_to_num_val(x, nv, pv, qv))
            }
            fn u_clip_scalar(
                a: &NdArray<$t>,
                a_min: Option<$t>,
                a_max: Option<$t>,
            ) -> Result<NdArray<$t>> {
                match (a_min, a_max) {
                    (None, None) => eval_unary(a, |x| x),
                    (Some(lo), None) => eval_unary_slice(
                        a,
                        |src, out| simd::$s_max_s(src, lo, out),
                        |x| $max_nan(x, lo),
                    ),
                    (None, Some(hi)) => eval_unary_slice(
                        a,
                        |src, out| simd::$s_min_s(src, hi, out),
                        |x| $min_nan(x, hi),
                    ),
                    (Some(lo), Some(hi)) => eval_unary_slice(
                        a,
                        |src, out| simd::$s_clip(src, lo, hi, out),
                        |x| $clip_elem(x, lo, hi),
                    ),
                }
            }
            fn u_clip_arrays(
                a: &NdArray<$t>,
                a_min: Option<&NdArray<$t>>,
                a_max: Option<&NdArray<$t>>,
            ) -> Result<NdArray<$t>> {
                match (a_min, a_max) {
                    (None, None) => eval_unary(a, |x| x),
                    (Some(lo), None) => Self::u_maximum(a, lo),
                    (None, Some(hi)) => Self::u_minimum(a, hi),
                    (Some(lo), Some(hi)) => {
                        if lo.len() == 1 && hi.len() == 1 {
                            return Self::u_clip_scalar(
                                a,
                                Some(unsafe { *lo.storage_ptr().add(lo.offset()) }),
                                Some(unsafe { *hi.storage_ptr().add(hi.offset()) }),
                            );
                        }
                        eval_ternary(a, lo, hi, $clip_elem)
                    }
                }
            }
            fn u_heaviside(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary(a, b, $heaviside)
            }
            fn u_sinc(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, $sinc_fn)
            }
            fn u_i0(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, $i0_fn)
            }
            fn u_maximum(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_max, simd::$s_max_s,
                    |s, sb, out| simd::$s_rmax_s(sb, s, out),
                    $max_nan,
                )
            }
            fn u_minimum(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_min, simd::$s_min_s,
                    |s, sb, out| simd::$s_rmin_s(sb, s, out),
                    $min_nan,
                )
            }
            fn u_fmax(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_fmax, simd::$s_fmax_s,
                    |s, sb, out| simd::$s_rfmax_s(sb, s, out),
                    $fmax_fn,
                )
            }
            fn u_fmin(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_binary_simd(
                    a, b, simd::$s_fmin, simd::$s_fmin_s,
                    |s, sb, out| simd::$s_rfmin_s(sb, s, out),
                    $fmin_fn,
                )
            }
            fn u_gcd(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("gcd", concat!($name_str, " inputs")))
            }
            fn u_lcm(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("lcm", concat!($name_str, " inputs")))
            }
            fn u_real(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x)
            }
            fn u_imag(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                NdArray::zeros(a.shape())
            }
            fn u_conj(a: &NdArray<$t>) -> Result<NdArray<$t>> {
                eval_unary(a, |x| x)
            }
            fn u_angle(a: &NdArray<$t>, deg: bool) -> Result<NdArray<$t>> {
                let factor = (if deg { 180.0 / PI } else { 1.0 }) as $t;
                eval_unary(a, |x| (0.0 as $t).atan2(x) * factor)
            }
            fn u_bitand(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("bitwise_and", concat!($name_str, " inputs")))
            }
            fn u_bitor(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("bitwise_or", concat!($name_str, " inputs")))
            }
            fn u_bitxor(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("bitwise_xor", concat!($name_str, " inputs")))
            }
            fn u_invert(_a: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("invert", concat!($name_str, " inputs")))
            }
            fn u_lshift(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("left_shift", concat!($name_str, " inputs")))
            }
            fn u_rshift(_a: &NdArray<$t>, _b: &NdArray<$t>) -> Result<NdArray<$t>> {
                Err(dtype_err("right_shift", concat!($name_str, " inputs")))
            }
            fn u_bitwise_count(_a: &NdArray<$t>) -> Result<NdArray<u8>> {
                Err(dtype_err("bitwise_count", concat!($name_str, " inputs")))
            }
            fn u_eq(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_eq, simd::$s_eq_s,
                    |s, sb, out| simd::$s_eq_s(sb, s, out),
                    |x, y| x == y,
                )
            }
            fn u_ne(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_ne, simd::$s_ne_s,
                    |s, sb, out| simd::$s_ne_s(sb, s, out),
                    |x, y| x != y,
                )
            }
            fn u_lt(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_lt, simd::$s_lt_s,
                    |s, sb, out| simd::$s_gt_s(sb, s, out),
                    |x, y| x < y,
                )
            }
            fn u_le(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_le, simd::$s_le_s,
                    |s, sb, out| simd::$s_ge_s(sb, s, out),
                    |x, y| x <= y,
                )
            }
            fn u_gt(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_gt, simd::$s_gt_s,
                    |s, sb, out| simd::$s_lt_s(sb, s, out),
                    |x, y| x > y,
                )
            }
            fn u_ge(a: &NdArray<$t>, b: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_binary_simd(
                    a, b, simd::$s_ge, simd::$s_ge_s,
                    |s, sb, out| simd::$s_le_s(sb, s, out),
                    |x, y| x >= y,
                )
            }
            fn u_isclose(
                a: &NdArray<$t>,
                b: &NdArray<$t>,
                rtol: f64,
                atol: f64,
                equal_nan: bool,
            ) -> Result<NdArray<bool>> {
                let rt = rtol as $t;
                let at = atol as $t;
                eval_binary_simd(
                    a,
                    b,
                    |sa, sb, out| simd::$s_isclose(sa, sb, rt, at, equal_nan, out),
                    |sa, y, out| {
                        for (slot, &x) in out.iter_mut().zip(sa.iter()) {
                            slot.write($isclose_elem(x, y, rt, at, equal_nan));
                        }
                    },
                    |x, sb, out| {
                        for (slot, &y) in out.iter_mut().zip(sb.iter()) {
                            slot.write($isclose_elem(x, y, rt, at, equal_nan));
                        }
                    },
                    |x, y| $isclose_elem(x, y, rt, at, equal_nan),
                )
            }
            fn u_array_equal(a: &NdArray<$t>, b: &NdArray<$t>, equal_nan: bool) -> bool {
                if a.shape() != b.shape() {
                    return false;
                }
                if !equal_nan {
                    return array_equal_exact(a, b);
                }
                a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            fn u_logical_not(a: &NdArray<$t>) -> Result<NdArray<bool>> {
                eval_unary_slice(a, simd::$s_lnot, |x| x == 0.0)
            }
        }
    };
}

impl_simd_float!(
    f32, "float32",
    floor_divide_f32, remainder_f32, divmod_f32, nextafter_f32, spacing_f32,
    frexp_f32, modf_f32, ldexp_f32, sign_f32, logaddexp_f32,
    logaddexp2_f32, heaviside_f32, sinc_f32, |x: f32| i0_f64(x as f64) as f32,
    round_f32, nan_to_num_f32_val, clip_f32_elem,
    max_prop_nan_f32, min_prop_nan_f32, fmax_f32, fmin_f32,
    isclose_f32_elem,
    add_f32, add_f32_scalar, sub_f32, sub_f32_scalar, rsub_f32_scalar,
    mul_f32, mul_f32_scalar, div_f32, div_f32_scalar, rdiv_f32_scalar,
    maximum_f32, maximum_f32_scalar, rmaximum_f32_scalar,
    minimum_f32, minimum_f32_scalar, rminimum_f32_scalar,
    fmax_f32, fmax_f32_scalar, rfmax_f32_scalar,
    fmin_f32, fmin_f32_scalar, rfmin_f32_scalar,
    neg_f32, abs_f32, square_f32, sqrt_f32, recip_f32,
    floor_f32, ceil_f32, trunc_f32, rint_f32, deg2rad_f32, rad2deg_f32, clip_scalar_f32,
    isfinite_f32, isinf_f32, isnan_f32, signbit_f32, logical_not_f32,
    eq_f32, eq_f32_scalar, ne_f32, ne_f32_scalar,
    lt_f32, lt_f32_scalar, le_f32, le_f32_scalar,
    gt_f32, gt_f32_scalar, ge_f32, ge_f32_scalar,
    isclose_f32
);

impl_simd_float!(
    f64, "float64",
    floor_divide_f64, remainder_f64, divmod_f64, nextafter_f64, spacing_f64,
    frexp_f64, modf_f64, ldexp_f64, sign_f64, logaddexp_f64,
    logaddexp2_f64, heaviside_f64, sinc_f64, i0_f64,
    round_f64, nan_to_num_f64_val, clip_f64_elem,
    max_prop_nan_f64, min_prop_nan_f64, fmax_f64, fmin_f64,
    isclose_f64_elem,
    add_f64, add_f64_scalar, sub_f64, sub_f64_scalar, rsub_f64_scalar,
    mul_f64, mul_f64_scalar, div_f64, div_f64_scalar, rdiv_f64_scalar,
    maximum_f64, maximum_f64_scalar, rmaximum_f64_scalar,
    minimum_f64, minimum_f64_scalar, rminimum_f64_scalar,
    fmax_f64, fmax_f64_scalar, rfmax_f64_scalar,
    fmin_f64, fmin_f64_scalar, rfmin_f64_scalar,
    neg_f64, abs_f64, square_f64, sqrt_f64, recip_f64,
    floor_f64, ceil_f64, trunc_f64, rint_f64, deg2rad_f64, rad2deg_f64, clip_scalar_f64,
    isfinite_f64, isinf_f64, isnan_f64, signbit_f64, logical_not_f64,
    eq_f64, eq_f64_scalar, ne_f64, ne_f64_scalar,
    lt_f64, lt_f64_scalar, le_f64, le_f64_scalar,
    gt_f64, gt_f64_scalar, ge_f64, ge_f64_scalar,
    isclose_f64
);

// ============================================================================
// Complex32 & Complex64
// ============================================================================

macro_rules! impl_complex {
    (
        $c:ty, $real:ty, $name_str:expr,
        $cdiv:ident, $crecip:ident, $csign:ident, $cpow:ident,
        $cexp2:ident, $cexpm1:ident, $clog2:ident, $clog10:ident, $clog1p:ident,
        $csinc:ident, $crint:ident, $round_real:ident, $nan_real:ident
    ) => {
        impl UfuncScalar for $c {
            type FloatOut = $c;
            type TrueDivOut = $c;
            type BoolToI8Out = $c;
            type FloatPowerOut = Complex64;
            type RealOut = $real;
            type AngleOut = $real;
            type RoundOut = $c;

            fn u_add(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, |x, y| x + y)
            }
            fn u_sub(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, |x, y| x - y)
            }
            fn u_mul(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, |x, y| x * y)
            }
            fn u_true_div(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, $cdiv)
            }
            fn u_floor_div(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("floor_divide", concat!($name_str, " inputs")))
            }
            fn u_rem(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("remainder", concat!($name_str, " inputs")))
            }
            fn u_fmod(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("fmod", concat!($name_str, " inputs")))
            }
            fn u_divmod(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<(NdArray<$c>, NdArray<$c>)> {
                Err(dtype_err("divmod", concat!($name_str, " inputs")))
            }
            fn u_pow(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, $cpow)
            }
            fn u_float_pow(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<Complex64>> {
                eval_binary(a, b, |x, y| {
                    c128_pow(
                        Complex64::new(x.re as f64, x.im as f64),
                        Complex64::new(y.re as f64, y.im as f64),
                    )
                })
            }
            fn u_neg(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| -z)
            }
            fn u_pos(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z)
            }
            fn u_abs(a: &NdArray<$c>) -> Result<NdArray<$real>> {
                eval_unary(a, |z| z.re.hypot(z.im))
            }
            fn u_fabs(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("fabs", concat!($name_str, " inputs")))
            }
            fn u_square(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z * z)
            }
            fn u_recip(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $crecip)
            }
            fn u_sqrt(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.sqrt())
            }
            fn u_cbrt(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("cbrt", concat!($name_str, " inputs")))
            }
            fn u_exp(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.exp())
            }
            fn u_exp2(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $cexp2)
            }
            fn u_expm1(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $cexpm1)
            }
            fn u_log(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.ln())
            }
            fn u_log2(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $clog2)
            }
            fn u_log10(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $clog10)
            }
            fn u_log1p(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $clog1p)
            }
            fn u_logaddexp(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("logaddexp", concat!($name_str, " inputs")))
            }
            fn u_logaddexp2(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("logaddexp2", concat!($name_str, " inputs")))
            }
            fn u_sin(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.sin())
            }
            fn u_cos(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.cos())
            }
            fn u_tan(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.tan())
            }
            fn u_arcsin(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.asin())
            }
            fn u_arccos(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.acos())
            }
            fn u_arctan(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.atan())
            }
            fn u_arctan2(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("arctan2", concat!($name_str, " inputs")))
            }
            fn u_hypot(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("hypot", concat!($name_str, " inputs")))
            }
            fn u_sinh(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.sinh())
            }
            fn u_cosh(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.cosh())
            }
            fn u_tanh(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.tanh())
            }
            fn u_arcsinh(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.asinh())
            }
            fn u_arccosh(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.acosh())
            }
            fn u_arctanh(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.atanh())
            }
            fn u_deg2rad(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("deg2rad", concat!($name_str, " inputs")))
            }
            fn u_rad2deg(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("rad2deg", concat!($name_str, " inputs")))
            }
            fn u_floor(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("floor", concat!($name_str, " inputs")))
            }
            fn u_ceil(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("ceil", concat!($name_str, " inputs")))
            }
            fn u_trunc(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("trunc", concat!($name_str, " inputs")))
            }
            fn u_rint(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $crint)
            }
            fn u_round(a: &NdArray<$c>, decimals: i32) -> Result<NdArray<$c>> {
                eval_unary(a, |z| Complex::new($round_real(z.re, decimals), $round_real(z.im, decimals)))
            }
            fn u_sign(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $csign)
            }
            fn u_signbit(_a: &NdArray<$c>) -> Result<NdArray<bool>> {
                Err(dtype_err("signbit", concat!($name_str, " inputs")))
            }
            fn u_copysign(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("copysign", concat!($name_str, " inputs")))
            }
            fn u_spacing(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("spacing", concat!($name_str, " inputs")))
            }
            fn u_nextafter(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("nextafter", concat!($name_str, " inputs")))
            }
            fn u_to_ldexp_exp(self) -> Result<i64> {
                Err(dtype_err("ldexp", concat!($name_str, " exponent")))
            }
            fn u_ldexp_with_i64(_a: &NdArray<$c>, _exp: &NdArray<i64>) -> Result<NdArray<$c>> {
                Err(dtype_err("ldexp", concat!($name_str, " inputs")))
            }
            fn u_frexp(_a: &NdArray<$c>) -> Result<(NdArray<$c>, NdArray<i32>)> {
                Err(dtype_err("frexp", concat!($name_str, " inputs")))
            }
            fn u_modf(_a: &NdArray<$c>) -> Result<(NdArray<$c>, NdArray<$c>)> {
                Err(dtype_err("modf", concat!($name_str, " inputs")))
            }
            fn u_isfinite(a: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_unary(a, |z| z.re.is_finite() && z.im.is_finite())
            }
            fn u_isinf(a: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_unary(a, |z| z.re.is_infinite() || z.im.is_infinite())
            }
            fn u_isnan(a: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_unary(a, |z| z.re.is_nan() || z.im.is_nan())
            }
            fn u_isneginf(_a: &NdArray<$c>) -> Result<NdArray<bool>> {
                Err(dtype_err("isneginf", concat!($name_str, " inputs")))
            }
            fn u_isposinf(_a: &NdArray<$c>) -> Result<NdArray<bool>> {
                Err(dtype_err("isposinf", concat!($name_str, " inputs")))
            }
            fn u_nan_to_num(
                a: &NdArray<$c>,
                nan: Option<f64>,
                posinf: Option<f64>,
                neginf: Option<f64>,
            ) -> Result<NdArray<$c>> {
                let nv = nan.unwrap_or(0.0) as $real;
                let pv = posinf.map(|v| v as $real).unwrap_or(<$real>::MAX);
                let qv = neginf.map(|v| v as $real).unwrap_or(<$real>::MIN);
                eval_unary(a, |z| {
                    Complex::new($nan_real(z.re, nv, pv, qv), $nan_real(z.im, nv, pv, qv))
                })
            }
            fn u_clip_scalar(
                a: &NdArray<$c>,
                a_min: Option<$c>,
                a_max: Option<$c>,
            ) -> Result<NdArray<$c>> {
                match (a_min, a_max) {
                    (None, None) => eval_unary(a, |x| x),
                    (Some(lo), None) => eval_unary(a, |x| c_max(x, lo)),
                    (None, Some(hi)) => eval_unary(a, |x| c_min(x, hi)),
                    (Some(lo), Some(hi)) => eval_unary(a, |x| c_clip_elem(x, lo, hi)),
                }
            }
            fn u_clip_arrays(
                a: &NdArray<$c>,
                a_min: Option<&NdArray<$c>>,
                a_max: Option<&NdArray<$c>>,
            ) -> Result<NdArray<$c>> {
                match (a_min, a_max) {
                    (None, None) => eval_unary(a, |x| x),
                    (Some(lo), None) => Self::u_maximum(a, lo),
                    (None, Some(hi)) => Self::u_minimum(a, hi),
                    (Some(lo), Some(hi)) => eval_ternary(a, lo, hi, c_clip_elem),
                }
            }
            fn u_heaviside(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("heaviside", concat!($name_str, " inputs")))
            }
            fn u_sinc(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, $csinc)
            }
            fn u_i0(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("i0", concat!($name_str, " inputs")))
            }
            fn u_maximum(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, c_max)
            }
            fn u_minimum(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, c_min)
            }
            fn u_fmax(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, c_fmax)
            }
            fn u_fmin(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_binary(a, b, c_fmin)
            }
            fn u_gcd(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("gcd", concat!($name_str, " inputs")))
            }
            fn u_lcm(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("lcm", concat!($name_str, " inputs")))
            }
            fn u_real(a: &NdArray<$c>) -> Result<NdArray<$real>> {
                eval_unary(a, |z| z.re)
            }
            fn u_imag(a: &NdArray<$c>) -> Result<NdArray<$real>> {
                eval_unary(a, |z| z.im)
            }
            fn u_conj(a: &NdArray<$c>) -> Result<NdArray<$c>> {
                eval_unary(a, |z| z.conj())
            }
            fn u_angle(a: &NdArray<$c>, deg: bool) -> Result<NdArray<$real>> {
                let factor = (if deg { 180.0 / PI } else { 1.0 }) as $real;
                eval_unary(a, |z| z.im.atan2(z.re) * factor)
            }
            fn u_bitand(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("bitwise_and", concat!($name_str, " inputs")))
            }
            fn u_bitor(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("bitwise_or", concat!($name_str, " inputs")))
            }
            fn u_bitxor(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("bitwise_xor", concat!($name_str, " inputs")))
            }
            fn u_invert(_a: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("invert", concat!($name_str, " inputs")))
            }
            fn u_lshift(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("left_shift", concat!($name_str, " inputs")))
            }
            fn u_rshift(_a: &NdArray<$c>, _b: &NdArray<$c>) -> Result<NdArray<$c>> {
                Err(dtype_err("right_shift", concat!($name_str, " inputs")))
            }
            fn u_bitwise_count(_a: &NdArray<$c>) -> Result<NdArray<u8>> {
                Err(dtype_err("bitwise_count", concat!($name_str, " inputs")))
            }
            fn u_eq(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_binary(a, b, c_eq)
            }
            fn u_ne(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_binary(a, b, c_ne)
            }
            fn u_lt(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_binary(a, b, c_lt)
            }
            fn u_le(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_binary(a, b, c_le)
            }
            fn u_gt(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_binary(a, b, c_gt)
            }
            fn u_ge(a: &NdArray<$c>, b: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_binary(a, b, c_ge)
            }
            fn u_isclose(
                a: &NdArray<$c>,
                b: &NdArray<$c>,
                rtol: f64,
                atol: f64,
                equal_nan: bool,
            ) -> Result<NdArray<bool>> {
                let rt = rtol as $real;
                let at = atol as $real;
                eval_binary(a, b, |x, y| {
                    let xn = x.re.is_nan() || x.im.is_nan();
                    let yn = y.re.is_nan() || y.im.is_nan();
                    if xn && yn {
                        equal_nan
                    } else if xn || yn {
                        false
                    } else if c_eq(x, y) {
                        true
                    } else if y.re.is_finite() && y.im.is_finite() {
                        let d = x - y;
                        d.re.hypot(d.im) <= at + rt * y.re.hypot(y.im)
                    } else {
                        false
                    }
                })
            }
            fn u_array_equal(a: &NdArray<$c>, b: &NdArray<$c>, equal_nan: bool) -> bool {
                if a.shape() != b.shape() {
                    return false;
                }
                if !equal_nan {
                    return array_equal_exact(a, b);
                }
                a.iter().zip(b.iter()).all(|(x, y)| {
                    c_eq(x, y)
                        || ((x.re.is_nan() || x.im.is_nan()) && (y.re.is_nan() || y.im.is_nan()))
                })
            }
            fn u_logical_not(a: &NdArray<$c>) -> Result<NdArray<bool>> {
                eval_unary(a, |z| z.re == 0.0 && z.im == 0.0)
            }
        }
    };
}

impl_complex!(
    Complex32, f32, "complex64",
    c64_div, c64_recip, c64_sign, c64_pow,
    c64_exp2, c64_expm1, c64_log2, c64_log10, c64_log1p,
    c64_sinc, c64_rint, round_f32, nan_to_num_f32_val
);

impl_complex!(
    Complex64, f64, "complex128",
    c128_div, c128_recip, c128_sign, c128_pow,
    c128_exp2, c128_expm1, c128_log2, c128_log10, c128_log1p,
    c128_sinc, c128_rint, round_f64, nan_to_num_f64_val
);

// ============================================================================
// Public typed ufunc entry points (`numpy_ufunc::typed::*`)
// ============================================================================

macro_rules! def_typed_bin_same {
    ($fn_name:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<T: UfuncScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
            T::$method(a, b)
        }
    };
}

macro_rules! def_typed_bin_assoc {
    ($fn_name:ident, $method:ident, $assoc:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<T: UfuncScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T::$assoc>> {
            T::$method(a, b)
        }
    };
}

macro_rules! def_typed_bin_bool {
    ($fn_name:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<T: UfuncScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<bool>> {
            T::$method(a, b)
        }
    };
}

macro_rules! def_typed_unary_same {
    ($fn_name:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<T: UfuncScalar>(a: &NdArray<T>) -> Result<NdArray<T>> {
            T::$method(a)
        }
    };
}

macro_rules! def_typed_unary_assoc {
    ($fn_name:ident, $method:ident, $assoc:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<T: UfuncScalar>(a: &NdArray<T>) -> Result<NdArray<T::$assoc>> {
            T::$method(a)
        }
    };
}

macro_rules! def_typed_unary_bool {
    ($fn_name:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<T: UfuncScalar>(a: &NdArray<T>) -> Result<NdArray<bool>> {
            T::$method(a)
        }
    };
}

// Arithmetic
def_typed_bin_same!(add, u_add, "Element-wise addition.");
def_typed_bin_same!(subtract, u_sub, "Element-wise subtraction.");
def_typed_bin_same!(multiply, u_mul, "Element-wise multiplication.");
def_typed_bin_assoc!(true_divide, u_true_div, TrueDivOut, "Element-wise true division.");
pub use true_divide as divide;
def_typed_bin_assoc!(floor_divide, u_floor_div, BoolToI8Out, "Element-wise floor division.");
def_typed_bin_assoc!(remainder, u_rem, BoolToI8Out, "Element-wise remainder (NumPy `remainder`/`mod`).");
pub use remainder as modulo;
pub use remainder as r#mod;
def_typed_bin_assoc!(fmod, u_fmod, BoolToI8Out, "Element-wise C `fmod` remainder.");
#[inline]
pub fn divmod<T: UfuncScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
) -> Result<(NdArray<T::BoolToI8Out>, NdArray<T::BoolToI8Out>)> {
    T::u_divmod(a, b)
}
def_typed_bin_assoc!(power, u_pow, BoolToI8Out, "Element-wise exponentiation.");
def_typed_bin_assoc!(float_power, u_float_pow, FloatPowerOut, "Element-wise double/complex exponentiation.");
def_typed_unary_same!(negative, u_neg, "Element-wise negation.");
def_typed_unary_same!(positive, u_pos, "Element-wise positive (copy).");
def_typed_unary_assoc!(abs, u_abs, RealOut, "Element-wise absolute value.");
pub use abs as absolute;
def_typed_unary_assoc!(fabs, u_fabs, FloatOut, "Element-wise real floating absolute value.");
def_typed_unary_assoc!(square, u_square, BoolToI8Out, "Element-wise square.");
def_typed_unary_assoc!(reciprocal, u_recip, BoolToI8Out, "Element-wise reciprocal.");
def_typed_unary_assoc!(sqrt, u_sqrt, FloatOut, "Element-wise square root.");
def_typed_unary_assoc!(cbrt, u_cbrt, FloatOut, "Element-wise cube root.");

// Exp / Log
def_typed_unary_assoc!(exp, u_exp, FloatOut, "Element-wise `e^x`.");
def_typed_unary_assoc!(exp2, u_exp2, FloatOut, "Element-wise `2^x`.");
def_typed_unary_assoc!(expm1, u_expm1, FloatOut, "Element-wise `e^x - 1`.");
def_typed_unary_assoc!(log, u_log, FloatOut, "Element-wise natural logarithm.");
def_typed_unary_assoc!(log2, u_log2, FloatOut, "Element-wise base-2 logarithm.");
def_typed_unary_assoc!(log10, u_log10, FloatOut, "Element-wise base-10 logarithm.");
def_typed_unary_assoc!(log1p, u_log1p, FloatOut, "Element-wise `ln(1 + x)`.");
def_typed_bin_assoc!(logaddexp, u_logaddexp, FloatOut, "Element-wise `ln(exp(x) + exp(y))`.");
def_typed_bin_assoc!(logaddexp2, u_logaddexp2, FloatOut, "Element-wise `log2(2^x + 2^y)`.");

// Trig / Hyperbolic
def_typed_unary_assoc!(sin, u_sin, FloatOut, "Element-wise sine.");
def_typed_unary_assoc!(cos, u_cos, FloatOut, "Element-wise cosine.");
def_typed_unary_assoc!(tan, u_tan, FloatOut, "Element-wise tangent.");
def_typed_unary_assoc!(arcsin, u_arcsin, FloatOut, "Element-wise inverse sine.");
pub use arcsin as asin;
def_typed_unary_assoc!(arccos, u_arccos, FloatOut, "Element-wise inverse cosine.");
pub use arccos as acos;
def_typed_unary_assoc!(arctan, u_arctan, FloatOut, "Element-wise inverse tangent.");
pub use arctan as atan;
def_typed_bin_assoc!(arctan2, u_arctan2, FloatOut, "Element-wise 2-argument arc tangent.");
pub use arctan2 as atan2;
def_typed_bin_assoc!(hypot, u_hypot, FloatOut, "Element-wise Euclidean hypotenuse.");
def_typed_unary_assoc!(sinh, u_sinh, FloatOut, "Element-wise hyperbolic sine.");
def_typed_unary_assoc!(cosh, u_cosh, FloatOut, "Element-wise hyperbolic cosine.");
def_typed_unary_assoc!(tanh, u_tanh, FloatOut, "Element-wise hyperbolic tangent.");
def_typed_unary_assoc!(arcsinh, u_arcsinh, FloatOut, "Element-wise inverse hyperbolic sine.");
pub use arcsinh as asinh;
def_typed_unary_assoc!(arccosh, u_arccosh, FloatOut, "Element-wise inverse hyperbolic cosine.");
pub use arccosh as acosh;
def_typed_unary_assoc!(arctanh, u_arctanh, FloatOut, "Element-wise inverse hyperbolic tangent.");
pub use arctanh as atanh;
def_typed_unary_assoc!(deg2rad, u_deg2rad, FloatOut, "Convert degrees to radians.");
pub use deg2rad as radians;
def_typed_unary_assoc!(rad2deg, u_rad2deg, FloatOut, "Convert radians to degrees.");
pub use rad2deg as degrees;

// Rounding / Float / Special
def_typed_unary_same!(floor, u_floor, "Element-wise floor.");
def_typed_unary_same!(ceil, u_ceil, "Element-wise ceiling.");
def_typed_unary_same!(trunc, u_trunc, "Element-wise truncation toward zero.");
pub use trunc as fix;
def_typed_unary_assoc!(rint, u_rint, FloatOut, "Element-wise round to nearest integer (ties to even).");
#[inline]
pub fn round<T: UfuncScalar>(a: &NdArray<T>, decimals: i32) -> Result<NdArray<T::RoundOut>> {
    T::u_round(a, decimals)
}
pub use round as around;
def_typed_unary_same!(sign, u_sign, "Element-wise sign.");
def_typed_unary_bool!(signbit, u_signbit, "Element-wise IEEE sign bit.");
def_typed_bin_assoc!(copysign, u_copysign, FloatOut, "Copy sign of second argument to magnitude of first.");
def_typed_unary_assoc!(spacing, u_spacing, FloatOut, "Distance to nearest adjacent floating-point number.");
def_typed_bin_assoc!(nextafter, u_nextafter, FloatOut, "Next representable float value after x towards y.");
#[inline]
pub fn ldexp<T: UfuncScalar, E: UfuncScalar>(
    a: &NdArray<T>,
    exp: &NdArray<E>,
) -> Result<NdArray<T::FloatOut>> {
    let exp_i64: NdArray<i64> = crate::exec::try_eval_unary(exp, |e| e.u_to_ldexp_exp())?;
    T::u_ldexp_with_i64(a, &exp_i64)
}
#[inline]
pub fn frexp<T: UfuncScalar>(a: &NdArray<T>) -> Result<(NdArray<T::FloatOut>, NdArray<i32>)> {
    T::u_frexp(a)
}
#[inline]
pub fn modf<T: UfuncScalar>(
    a: &NdArray<T>,
) -> Result<(NdArray<T::FloatOut>, NdArray<T::FloatOut>)> {
    T::u_modf(a)
}
def_typed_unary_bool!(isfinite, u_isfinite, "Test finiteness element-wise.");
def_typed_unary_bool!(isinf, u_isinf, "Test infinity element-wise.");
def_typed_unary_bool!(isnan, u_isnan, "Test NaN element-wise.");
def_typed_unary_bool!(isneginf, u_isneginf, "Test negative infinity element-wise.");
def_typed_unary_bool!(isposinf, u_isposinf, "Test positive infinity element-wise.");
#[inline]
pub fn nan_to_num<T: UfuncScalar>(
    a: &NdArray<T>,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> Result<NdArray<T>> {
    T::u_nan_to_num(a, nan, posinf, neginf)
}
#[inline]
pub fn clip_scalar<T: UfuncScalar>(
    a: &NdArray<T>,
    a_min: Option<T>,
    a_max: Option<T>,
) -> Result<NdArray<T>> {
    T::u_clip_scalar(a, a_min, a_max)
}
#[inline]
pub fn clip<T: UfuncScalar>(
    a: &NdArray<T>,
    a_min: Option<&NdArray<T>>,
    a_max: Option<&NdArray<T>>,
) -> Result<NdArray<T>> {
    T::u_clip_arrays(a, a_min, a_max)
}
def_typed_bin_assoc!(heaviside, u_heaviside, FloatOut, "Heaviside step function.");
def_typed_unary_assoc!(sinc, u_sinc, TrueDivOut, "Normalized sinc function `sin(pi*x)/(pi*x)`.");
def_typed_unary_assoc!(i0, u_i0, TrueDivOut, "Modified Bessel function of the first kind, order 0.");
def_typed_bin_same!(maximum, u_maximum, "Element-wise maximum (propagates NaNs).");
def_typed_bin_same!(minimum, u_minimum, "Element-wise minimum (propagates NaNs).");
def_typed_bin_same!(fmax, u_fmax, "Element-wise maximum (ignores NaNs).");
def_typed_bin_same!(fmin, u_fmin, "Element-wise minimum (ignores NaNs).");
def_typed_bin_same!(gcd, u_gcd, "Element-wise greatest common divisor.");
def_typed_bin_same!(lcm, u_lcm, "Element-wise least common multiple.");

// Complex / Bitwise / Comparison / Logical
def_typed_unary_assoc!(real, u_real, RealOut, "Real part of elements.");
def_typed_unary_assoc!(imag, u_imag, RealOut, "Imaginary part of elements.");
def_typed_unary_assoc!(conj, u_conj, BoolToI8Out, "Complex conjugate element-wise.");
pub use conj as conjugate;
#[inline]
pub fn angle<T: UfuncScalar>(a: &NdArray<T>, deg: bool) -> Result<NdArray<T::AngleOut>> {
    T::u_angle(a, deg)
}
def_typed_bin_same!(bitwise_and, u_bitand, "Element-wise bitwise AND.");
def_typed_bin_same!(bitwise_or, u_bitor, "Element-wise bitwise OR.");
def_typed_bin_same!(bitwise_xor, u_bitxor, "Element-wise bitwise XOR.");
def_typed_unary_same!(invert, u_invert, "Element-wise bitwise NOT / invert.");
pub use invert as bitwise_not;
def_typed_bin_assoc!(left_shift, u_lshift, BoolToI8Out, "Element-wise left bit shift.");
def_typed_bin_assoc!(right_shift, u_rshift, BoolToI8Out, "Element-wise right bit shift.");
#[inline]
pub fn bitwise_count<T: UfuncScalar>(a: &NdArray<T>) -> Result<NdArray<u8>> {
    T::u_bitwise_count(a)
}
def_typed_bin_bool!(equal, u_eq, "Element-wise `==`.");
def_typed_bin_bool!(not_equal, u_ne, "Element-wise `!=`.");
def_typed_bin_bool!(less, u_lt, "Element-wise `<`.");
def_typed_bin_bool!(less_equal, u_le, "Element-wise `<=`.");
def_typed_bin_bool!(greater, u_gt, "Element-wise `>`.");
def_typed_bin_bool!(greater_equal, u_ge, "Element-wise `>=`.");

#[inline]
pub fn isclose<T: UfuncScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<NdArray<bool>> {
    T::u_isclose(a, b, rtol, atol, equal_nan)
}

#[inline]
pub fn allclose<T: UfuncScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<bool> {
    Ok(bool_all(&T::u_isclose(a, b, rtol, atol, equal_nan)?))
}

#[inline]
pub fn array_equal<T: UfuncScalar>(a: &NdArray<T>, b: &NdArray<T>, equal_nan: bool) -> bool {
    T::u_array_equal(a, b, equal_nan)
}

#[inline]
pub fn array_equiv<T: UfuncScalar>(a: &NdArray<T>, b: &NdArray<T>) -> bool {
    if broadcast_shapes(a.shape(), b.shape()).is_err() {
        return false;
    }
    T::u_eq(a, b).map(|eq| bool_all(&eq)).unwrap_or(false)
}

#[inline]
pub fn logical_and<A: UfuncScalar, B: UfuncScalar>(
    a: &NdArray<A>,
    b: &NdArray<B>,
) -> Result<NdArray<bool>> {
    eval_binary(a, b, |x, y| x.is_truthy() && y.is_truthy())
}

#[inline]
pub fn logical_or<A: UfuncScalar, B: UfuncScalar>(
    a: &NdArray<A>,
    b: &NdArray<B>,
) -> Result<NdArray<bool>> {
    eval_binary(a, b, |x, y| x.is_truthy() || y.is_truthy())
}

#[inline]
pub fn logical_xor<A: UfuncScalar, B: UfuncScalar>(
    a: &NdArray<A>,
    b: &NdArray<B>,
) -> Result<NdArray<bool>> {
    eval_binary(a, b, |x, y| x.is_truthy() ^ y.is_truthy())
}

def_typed_unary_bool!(logical_not, u_logical_not, "Element-wise logical NOT.");

//! Polymorphic top-level ufunc traits so `numpy_ufunc::<op>(...)` works uniformly
//! on both zero-cost typed [`NdArray<T>`] and dynamic [`DynArray`].

use crate::dyn_ufunc as dyn_api;
use crate::typed_ufunc::{self as typed, UfuncScalar};
use numpy_types::{DynArray, NdArray, NumPyError, Result};

mod private {
    pub trait Sealed {}
    impl<T: super::UfuncScalar> Sealed for super::NdArray<T> {}
    impl Sealed for super::DynArray {}
}

/// Unified array trait implemented for [`NdArray<T>`] (`T: UfuncScalar`) and [`DynArray`].
pub trait UfuncArray: private::Sealed + Sized {
    type SameOut;
    type FloatOut;
    type TrueDivOut;
    type BoolToI8Out;
    type FloatPowerOut;
    type RealOut;
    type AngleOut;
    type RoundOut;
    type BoolOut;
    type PopcntOut;

    fn ua_add(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_sub(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_mul(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_true_div(&self, b: &Self) -> Result<Self::TrueDivOut>;
    fn ua_floor_div(&self, b: &Self) -> Result<Self::BoolToI8Out>;
    fn ua_rem(&self, b: &Self) -> Result<Self::BoolToI8Out>;
    fn ua_fmod(&self, b: &Self) -> Result<Self::BoolToI8Out>;
    fn ua_divmod(&self, b: &Self) -> Result<(Self::BoolToI8Out, Self::BoolToI8Out)>;
    fn ua_pow(&self, b: &Self) -> Result<Self::BoolToI8Out>;
    fn ua_float_pow(&self, b: &Self) -> Result<Self::FloatPowerOut>;
    fn ua_neg(&self) -> Result<Self::SameOut>;
    fn ua_pos(&self) -> Result<Self::SameOut>;
    fn ua_abs(&self) -> Result<Self::RealOut>;
    fn ua_fabs(&self) -> Result<Self::FloatOut>;
    fn ua_square(&self) -> Result<Self::BoolToI8Out>;
    fn ua_recip(&self) -> Result<Self::BoolToI8Out>;
    fn ua_sqrt(&self) -> Result<Self::FloatOut>;
    fn ua_cbrt(&self) -> Result<Self::FloatOut>;

    fn ua_exp(&self) -> Result<Self::FloatOut>;
    fn ua_exp2(&self) -> Result<Self::FloatOut>;
    fn ua_expm1(&self) -> Result<Self::FloatOut>;
    fn ua_log(&self) -> Result<Self::FloatOut>;
    fn ua_log2(&self) -> Result<Self::FloatOut>;
    fn ua_log10(&self) -> Result<Self::FloatOut>;
    fn ua_log1p(&self) -> Result<Self::FloatOut>;
    fn ua_logaddexp(&self, b: &Self) -> Result<Self::FloatOut>;
    fn ua_logaddexp2(&self, b: &Self) -> Result<Self::FloatOut>;

    fn ua_sin(&self) -> Result<Self::FloatOut>;
    fn ua_cos(&self) -> Result<Self::FloatOut>;
    fn ua_tan(&self) -> Result<Self::FloatOut>;
    fn ua_arcsin(&self) -> Result<Self::FloatOut>;
    fn ua_arccos(&self) -> Result<Self::FloatOut>;
    fn ua_arctan(&self) -> Result<Self::FloatOut>;
    fn ua_arctan2(&self, b: &Self) -> Result<Self::FloatOut>;
    fn ua_hypot(&self, b: &Self) -> Result<Self::FloatOut>;
    fn ua_sinh(&self) -> Result<Self::FloatOut>;
    fn ua_cosh(&self) -> Result<Self::FloatOut>;
    fn ua_tanh(&self) -> Result<Self::FloatOut>;
    fn ua_arcsinh(&self) -> Result<Self::FloatOut>;
    fn ua_arccosh(&self) -> Result<Self::FloatOut>;
    fn ua_arctanh(&self) -> Result<Self::FloatOut>;
    fn ua_deg2rad(&self) -> Result<Self::FloatOut>;
    fn ua_rad2deg(&self) -> Result<Self::FloatOut>;

    fn ua_floor(&self) -> Result<Self::SameOut>;
    fn ua_ceil(&self) -> Result<Self::SameOut>;
    fn ua_trunc(&self) -> Result<Self::SameOut>;
    fn ua_rint(&self) -> Result<Self::FloatOut>;
    fn ua_round(&self, decimals: i32) -> Result<Self::RoundOut>;
    fn ua_sign(&self) -> Result<Self::SameOut>;
    fn ua_signbit(&self) -> Result<Self::BoolOut>;
    fn ua_copysign(&self, b: &Self) -> Result<Self::FloatOut>;
    fn ua_spacing(&self) -> Result<Self::FloatOut>;
    fn ua_nextafter(&self, b: &Self) -> Result<Self::FloatOut>;
    fn ua_frexp(&self) -> Result<(Self::FloatOut, Self::PopcntOut)>;
    fn ua_modf(&self) -> Result<(Self::FloatOut, Self::FloatOut)>;
    fn ua_isfinite(&self) -> Result<Self::BoolOut>;
    fn ua_isinf(&self) -> Result<Self::BoolOut>;
    fn ua_isnan(&self) -> Result<Self::BoolOut>;
    fn ua_isneginf(&self) -> Result<Self::BoolOut>;
    fn ua_isposinf(&self) -> Result<Self::BoolOut>;
    fn ua_nan_to_num(
        &self,
        nan: Option<f64>,
        posinf: Option<f64>,
        neginf: Option<f64>,
    ) -> Result<Self::SameOut>;
    fn ua_clip(&self, a_min: Option<&Self>, a_max: Option<&Self>) -> Result<Self::SameOut>;
    fn ua_heaviside(&self, b: &Self) -> Result<Self::FloatOut>;
    fn ua_sinc(&self) -> Result<Self::TrueDivOut>;
    fn ua_i0(&self) -> Result<Self::TrueDivOut>;
    fn ua_maximum(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_minimum(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_fmax(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_fmin(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_gcd(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_lcm(&self, b: &Self) -> Result<Self::SameOut>;

    fn ua_real(&self) -> Result<Self::RealOut>;
    fn ua_imag(&self) -> Result<Self::RealOut>;
    fn ua_conj(&self) -> Result<Self::BoolToI8Out>;
    fn ua_angle(&self, deg: bool) -> Result<Self::AngleOut>;
    fn ua_bitand(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_bitor(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_bitxor(&self, b: &Self) -> Result<Self::SameOut>;
    fn ua_invert(&self) -> Result<Self::SameOut>;
    fn ua_lshift(&self, b: &Self) -> Result<Self::BoolToI8Out>;
    fn ua_rshift(&self, b: &Self) -> Result<Self::BoolToI8Out>;
    fn ua_bitwise_count(&self) -> Result<Self::PopcntOut>;
    fn ua_eq(&self, b: &Self) -> Result<Self::BoolOut>;
    fn ua_ne(&self, b: &Self) -> Result<Self::BoolOut>;
    fn ua_lt(&self, b: &Self) -> Result<Self::BoolOut>;
    fn ua_le(&self, b: &Self) -> Result<Self::BoolOut>;
    fn ua_gt(&self, b: &Self) -> Result<Self::BoolOut>;
    fn ua_ge(&self, b: &Self) -> Result<Self::BoolOut>;
    fn ua_isclose(
        &self,
        b: &Self,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<Self::BoolOut>;
    fn ua_allclose(&self, b: &Self, rtol: f64, atol: f64, equal_nan: bool) -> Result<bool>;
    fn ua_array_equal(&self, b: &Self, equal_nan: bool) -> bool;
    fn ua_array_equiv(&self, b: &Self) -> bool;
    fn ua_logical_not(&self) -> Result<Self::BoolOut>;
}

/// Separate helper trait for `frexp`'s `i32` output vs `bitwise_count`'s `u8` output.
pub trait UfuncArrayExt: UfuncArray {
    type FrexpExpOut;
    fn ua_frexp_pair(&self) -> Result<(Self::FloatOut, Self::FrexpExpOut)>;
}

impl<T: UfuncScalar> UfuncArray for NdArray<T> {
    type SameOut = NdArray<T>;
    type FloatOut = NdArray<T::FloatOut>;
    type TrueDivOut = NdArray<T::TrueDivOut>;
    type BoolToI8Out = NdArray<T::BoolToI8Out>;
    type FloatPowerOut = NdArray<T::FloatPowerOut>;
    type RealOut = NdArray<T::RealOut>;
    type AngleOut = NdArray<T::AngleOut>;
    type RoundOut = NdArray<T::RoundOut>;
    type BoolOut = NdArray<bool>;
    type PopcntOut = NdArray<u8>;

    #[inline] fn ua_add(&self, b: &Self) -> Result<Self::SameOut> { typed::add(self, b) }
    #[inline] fn ua_sub(&self, b: &Self) -> Result<Self::SameOut> { typed::subtract(self, b) }
    #[inline] fn ua_mul(&self, b: &Self) -> Result<Self::SameOut> { typed::multiply(self, b) }
    #[inline] fn ua_true_div(&self, b: &Self) -> Result<Self::TrueDivOut> { typed::true_divide(self, b) }
    #[inline] fn ua_floor_div(&self, b: &Self) -> Result<Self::BoolToI8Out> { typed::floor_divide(self, b) }
    #[inline] fn ua_rem(&self, b: &Self) -> Result<Self::BoolToI8Out> { typed::remainder(self, b) }
    #[inline] fn ua_fmod(&self, b: &Self) -> Result<Self::BoolToI8Out> { typed::fmod(self, b) }
    #[inline] fn ua_divmod(&self, b: &Self) -> Result<(Self::BoolToI8Out, Self::BoolToI8Out)> { typed::divmod(self, b) }
    #[inline] fn ua_pow(&self, b: &Self) -> Result<Self::BoolToI8Out> { typed::power(self, b) }
    #[inline] fn ua_float_pow(&self, b: &Self) -> Result<Self::FloatPowerOut> { typed::float_power(self, b) }
    #[inline] fn ua_neg(&self) -> Result<Self::SameOut> { typed::negative(self) }
    #[inline] fn ua_pos(&self) -> Result<Self::SameOut> { typed::positive(self) }
    #[inline] fn ua_abs(&self) -> Result<Self::RealOut> { typed::abs(self) }
    #[inline] fn ua_fabs(&self) -> Result<Self::FloatOut> { typed::fabs(self) }
    #[inline] fn ua_square(&self) -> Result<Self::BoolToI8Out> { typed::square(self) }
    #[inline] fn ua_recip(&self) -> Result<Self::BoolToI8Out> { typed::reciprocal(self) }
    #[inline] fn ua_sqrt(&self) -> Result<Self::FloatOut> { typed::sqrt(self) }
    #[inline] fn ua_cbrt(&self) -> Result<Self::FloatOut> { typed::cbrt(self) }

    #[inline] fn ua_exp(&self) -> Result<Self::FloatOut> { typed::exp(self) }
    #[inline] fn ua_exp2(&self) -> Result<Self::FloatOut> { typed::exp2(self) }
    #[inline] fn ua_expm1(&self) -> Result<Self::FloatOut> { typed::expm1(self) }
    #[inline] fn ua_log(&self) -> Result<Self::FloatOut> { typed::log(self) }
    #[inline] fn ua_log2(&self) -> Result<Self::FloatOut> { typed::log2(self) }
    #[inline] fn ua_log10(&self) -> Result<Self::FloatOut> { typed::log10(self) }
    #[inline] fn ua_log1p(&self) -> Result<Self::FloatOut> { typed::log1p(self) }
    #[inline] fn ua_logaddexp(&self, b: &Self) -> Result<Self::FloatOut> { typed::logaddexp(self, b) }
    #[inline] fn ua_logaddexp2(&self, b: &Self) -> Result<Self::FloatOut> { typed::logaddexp2(self, b) }

    #[inline] fn ua_sin(&self) -> Result<Self::FloatOut> { typed::sin(self) }
    #[inline] fn ua_cos(&self) -> Result<Self::FloatOut> { typed::cos(self) }
    #[inline] fn ua_tan(&self) -> Result<Self::FloatOut> { typed::tan(self) }
    #[inline] fn ua_arcsin(&self) -> Result<Self::FloatOut> { typed::arcsin(self) }
    #[inline] fn ua_arccos(&self) -> Result<Self::FloatOut> { typed::arccos(self) }
    #[inline] fn ua_arctan(&self) -> Result<Self::FloatOut> { typed::arctan(self) }
    #[inline] fn ua_arctan2(&self, b: &Self) -> Result<Self::FloatOut> { typed::arctan2(self, b) }
    #[inline] fn ua_hypot(&self, b: &Self) -> Result<Self::FloatOut> { typed::hypot(self, b) }
    #[inline] fn ua_sinh(&self) -> Result<Self::FloatOut> { typed::sinh(self) }
    #[inline] fn ua_cosh(&self) -> Result<Self::FloatOut> { typed::cosh(self) }
    #[inline] fn ua_tanh(&self) -> Result<Self::FloatOut> { typed::tanh(self) }
    #[inline] fn ua_arcsinh(&self) -> Result<Self::FloatOut> { typed::arcsinh(self) }
    #[inline] fn ua_arccosh(&self) -> Result<Self::FloatOut> { typed::arccosh(self) }
    #[inline] fn ua_arctanh(&self) -> Result<Self::FloatOut> { typed::arctanh(self) }
    #[inline] fn ua_deg2rad(&self) -> Result<Self::FloatOut> { typed::deg2rad(self) }
    #[inline] fn ua_rad2deg(&self) -> Result<Self::FloatOut> { typed::rad2deg(self) }

    #[inline] fn ua_floor(&self) -> Result<Self::SameOut> { typed::floor(self) }
    #[inline] fn ua_ceil(&self) -> Result<Self::SameOut> { typed::ceil(self) }
    #[inline] fn ua_trunc(&self) -> Result<Self::SameOut> { typed::trunc(self) }
    #[inline] fn ua_rint(&self) -> Result<Self::FloatOut> { typed::rint(self) }
    #[inline] fn ua_round(&self, decimals: i32) -> Result<Self::RoundOut> { typed::round(self, decimals) }
    #[inline] fn ua_sign(&self) -> Result<Self::SameOut> { typed::sign(self) }
    #[inline] fn ua_signbit(&self) -> Result<Self::BoolOut> { typed::signbit(self) }
    #[inline] fn ua_copysign(&self, b: &Self) -> Result<Self::FloatOut> { typed::copysign(self, b) }
    #[inline] fn ua_spacing(&self) -> Result<Self::FloatOut> { typed::spacing(self) }
    #[inline] fn ua_nextafter(&self, b: &Self) -> Result<Self::FloatOut> { typed::nextafter(self, b) }
    #[inline] fn ua_frexp(&self) -> Result<(Self::FloatOut, Self::PopcntOut)> {
        let _ = self;
        Err(NumPyError::DTypeError(
            "frexp is only supported for real floating-point arrays".into(),
        ))
    }
    #[inline] fn ua_modf(&self) -> Result<(Self::FloatOut, Self::FloatOut)> { typed::modf(self) }
    #[inline] fn ua_isfinite(&self) -> Result<Self::BoolOut> { typed::isfinite(self) }
    #[inline] fn ua_isinf(&self) -> Result<Self::BoolOut> { typed::isinf(self) }
    #[inline] fn ua_isnan(&self) -> Result<Self::BoolOut> { typed::isnan(self) }
    #[inline] fn ua_isneginf(&self) -> Result<Self::BoolOut> { typed::isneginf(self) }
    #[inline] fn ua_isposinf(&self) -> Result<Self::BoolOut> { typed::isposinf(self) }
    #[inline]
    fn ua_nan_to_num(&self, nan: Option<f64>, posinf: Option<f64>, neginf: Option<f64>) -> Result<Self::SameOut> {
        typed::nan_to_num(self, nan, posinf, neginf)
    }
    #[inline] fn ua_clip(&self, a_min: Option<&Self>, a_max: Option<&Self>) -> Result<Self::SameOut> { typed::clip(self, a_min, a_max) }
    #[inline] fn ua_heaviside(&self, b: &Self) -> Result<Self::FloatOut> { typed::heaviside(self, b) }
    #[inline] fn ua_sinc(&self) -> Result<Self::TrueDivOut> { typed::sinc(self) }
    #[inline] fn ua_i0(&self) -> Result<Self::TrueDivOut> { typed::i0(self) }
    #[inline] fn ua_maximum(&self, b: &Self) -> Result<Self::SameOut> { typed::maximum(self, b) }
    #[inline] fn ua_minimum(&self, b: &Self) -> Result<Self::SameOut> { typed::minimum(self, b) }
    #[inline] fn ua_fmax(&self, b: &Self) -> Result<Self::SameOut> { typed::fmax(self, b) }
    #[inline] fn ua_fmin(&self, b: &Self) -> Result<Self::SameOut> { typed::fmin(self, b) }
    #[inline] fn ua_gcd(&self, b: &Self) -> Result<Self::SameOut> { typed::gcd(self, b) }
    #[inline] fn ua_lcm(&self, b: &Self) -> Result<Self::SameOut> { typed::lcm(self, b) }

    #[inline] fn ua_real(&self) -> Result<Self::RealOut> { typed::real(self) }
    #[inline] fn ua_imag(&self) -> Result<Self::RealOut> { typed::imag(self) }
    #[inline] fn ua_conj(&self) -> Result<Self::BoolToI8Out> { typed::conj(self) }
    #[inline] fn ua_angle(&self, deg: bool) -> Result<Self::AngleOut> { typed::angle(self, deg) }
    #[inline] fn ua_bitand(&self, b: &Self) -> Result<Self::SameOut> { typed::bitwise_and(self, b) }
    #[inline] fn ua_bitor(&self, b: &Self) -> Result<Self::SameOut> { typed::bitwise_or(self, b) }
    #[inline] fn ua_bitxor(&self, b: &Self) -> Result<Self::SameOut> { typed::bitwise_xor(self, b) }
    #[inline] fn ua_invert(&self) -> Result<Self::SameOut> { typed::invert(self) }
    #[inline] fn ua_lshift(&self, b: &Self) -> Result<Self::BoolToI8Out> { typed::left_shift(self, b) }
    #[inline] fn ua_rshift(&self, b: &Self) -> Result<Self::BoolToI8Out> { typed::right_shift(self, b) }
    #[inline] fn ua_bitwise_count(&self) -> Result<Self::PopcntOut> { typed::bitwise_count(self) }
    #[inline] fn ua_eq(&self, b: &Self) -> Result<Self::BoolOut> { typed::equal(self, b) }
    #[inline] fn ua_ne(&self, b: &Self) -> Result<Self::BoolOut> { typed::not_equal(self, b) }
    #[inline] fn ua_lt(&self, b: &Self) -> Result<Self::BoolOut> { typed::less(self, b) }
    #[inline] fn ua_le(&self, b: &Self) -> Result<Self::BoolOut> { typed::less_equal(self, b) }
    #[inline] fn ua_gt(&self, b: &Self) -> Result<Self::BoolOut> { typed::greater(self, b) }
    #[inline] fn ua_ge(&self, b: &Self) -> Result<Self::BoolOut> { typed::greater_equal(self, b) }
    #[inline]
    fn ua_isclose(&self, b: &Self, rtol: f64, atol: f64, equal_nan: bool) -> Result<Self::BoolOut> {
        typed::isclose(self, b, rtol, atol, equal_nan)
    }
    #[inline]
    fn ua_allclose(&self, b: &Self, rtol: f64, atol: f64, equal_nan: bool) -> Result<bool> {
        typed::allclose(self, b, rtol, atol, equal_nan)
    }
    #[inline] fn ua_array_equal(&self, b: &Self, equal_nan: bool) -> bool { typed::array_equal(self, b, equal_nan) }
    #[inline] fn ua_array_equiv(&self, b: &Self) -> bool { typed::array_equiv(self, b) }
    #[inline] fn ua_logical_not(&self) -> Result<Self::BoolOut> { typed::logical_not(self) }
}

impl<T: UfuncScalar> UfuncArrayExt for NdArray<T> {
    type FrexpExpOut = NdArray<i32>;
    #[inline]
    fn ua_frexp_pair(&self) -> Result<(Self::FloatOut, Self::FrexpExpOut)> {
        typed::frexp(self)
    }
}

impl UfuncArray for DynArray {
    type SameOut = DynArray;
    type FloatOut = DynArray;
    type TrueDivOut = DynArray;
    type BoolToI8Out = DynArray;
    type FloatPowerOut = DynArray;
    type RealOut = DynArray;
    type AngleOut = DynArray;
    type RoundOut = DynArray;
    type BoolOut = DynArray;
    type PopcntOut = DynArray;

    #[inline] fn ua_add(&self, b: &Self) -> Result<DynArray> { dyn_api::add(self, b) }
    #[inline] fn ua_sub(&self, b: &Self) -> Result<DynArray> { dyn_api::subtract(self, b) }
    #[inline] fn ua_mul(&self, b: &Self) -> Result<DynArray> { dyn_api::multiply(self, b) }
    #[inline] fn ua_true_div(&self, b: &Self) -> Result<DynArray> { dyn_api::true_divide(self, b) }
    #[inline] fn ua_floor_div(&self, b: &Self) -> Result<DynArray> { dyn_api::floor_divide(self, b) }
    #[inline] fn ua_rem(&self, b: &Self) -> Result<DynArray> { dyn_api::remainder(self, b) }
    #[inline] fn ua_fmod(&self, b: &Self) -> Result<DynArray> { dyn_api::fmod(self, b) }
    #[inline] fn ua_divmod(&self, b: &Self) -> Result<(DynArray, DynArray)> { dyn_api::divmod(self, b) }
    #[inline] fn ua_pow(&self, b: &Self) -> Result<DynArray> { dyn_api::power(self, b) }
    #[inline] fn ua_float_pow(&self, b: &Self) -> Result<DynArray> { dyn_api::float_power(self, b) }
    #[inline] fn ua_neg(&self) -> Result<DynArray> { dyn_api::negative(self) }
    #[inline] fn ua_pos(&self) -> Result<DynArray> { dyn_api::positive(self) }
    #[inline] fn ua_abs(&self) -> Result<DynArray> { dyn_api::abs(self) }
    #[inline] fn ua_fabs(&self) -> Result<DynArray> { dyn_api::fabs(self) }
    #[inline] fn ua_square(&self) -> Result<DynArray> { dyn_api::square(self) }
    #[inline] fn ua_recip(&self) -> Result<DynArray> { dyn_api::reciprocal(self) }
    #[inline] fn ua_sqrt(&self) -> Result<DynArray> { dyn_api::sqrt(self) }
    #[inline] fn ua_cbrt(&self) -> Result<DynArray> { dyn_api::cbrt(self) }

    #[inline] fn ua_exp(&self) -> Result<DynArray> { dyn_api::exp(self) }
    #[inline] fn ua_exp2(&self) -> Result<DynArray> { dyn_api::exp2(self) }
    #[inline] fn ua_expm1(&self) -> Result<DynArray> { dyn_api::expm1(self) }
    #[inline] fn ua_log(&self) -> Result<DynArray> { dyn_api::log(self) }
    #[inline] fn ua_log2(&self) -> Result<DynArray> { dyn_api::log2(self) }
    #[inline] fn ua_log10(&self) -> Result<DynArray> { dyn_api::log10(self) }
    #[inline] fn ua_log1p(&self) -> Result<DynArray> { dyn_api::log1p(self) }
    #[inline] fn ua_logaddexp(&self, b: &Self) -> Result<DynArray> { dyn_api::logaddexp(self, b) }
    #[inline] fn ua_logaddexp2(&self, b: &Self) -> Result<DynArray> { dyn_api::logaddexp2(self, b) }

    #[inline] fn ua_sin(&self) -> Result<DynArray> { dyn_api::sin(self) }
    #[inline] fn ua_cos(&self) -> Result<DynArray> { dyn_api::cos(self) }
    #[inline] fn ua_tan(&self) -> Result<DynArray> { dyn_api::tan(self) }
    #[inline] fn ua_arcsin(&self) -> Result<DynArray> { dyn_api::arcsin(self) }
    #[inline] fn ua_arccos(&self) -> Result<DynArray> { dyn_api::arccos(self) }
    #[inline] fn ua_arctan(&self) -> Result<DynArray> { dyn_api::arctan(self) }
    #[inline] fn ua_arctan2(&self, b: &Self) -> Result<DynArray> { dyn_api::arctan2(self, b) }
    #[inline] fn ua_hypot(&self, b: &Self) -> Result<DynArray> { dyn_api::hypot(self, b) }
    #[inline] fn ua_sinh(&self) -> Result<DynArray> { dyn_api::sinh(self) }
    #[inline] fn ua_cosh(&self) -> Result<DynArray> { dyn_api::cosh(self) }
    #[inline] fn ua_tanh(&self) -> Result<DynArray> { dyn_api::tanh(self) }
    #[inline] fn ua_arcsinh(&self) -> Result<DynArray> { dyn_api::arcsinh(self) }
    #[inline] fn ua_arccosh(&self) -> Result<DynArray> { dyn_api::arccosh(self) }
    #[inline] fn ua_arctanh(&self) -> Result<DynArray> { dyn_api::arctanh(self) }
    #[inline] fn ua_deg2rad(&self) -> Result<DynArray> { dyn_api::deg2rad(self) }
    #[inline] fn ua_rad2deg(&self) -> Result<DynArray> { dyn_api::rad2deg(self) }

    #[inline] fn ua_floor(&self) -> Result<DynArray> { dyn_api::floor(self) }
    #[inline] fn ua_ceil(&self) -> Result<DynArray> { dyn_api::ceil(self) }
    #[inline] fn ua_trunc(&self) -> Result<DynArray> { dyn_api::trunc(self) }
    #[inline] fn ua_rint(&self) -> Result<DynArray> { dyn_api::rint(self) }
    #[inline] fn ua_round(&self, decimals: i32) -> Result<DynArray> { dyn_api::round(self, decimals) }
    #[inline] fn ua_sign(&self) -> Result<DynArray> { dyn_api::sign(self) }
    #[inline] fn ua_signbit(&self) -> Result<DynArray> { dyn_api::signbit(self) }
    #[inline] fn ua_copysign(&self, b: &Self) -> Result<DynArray> { dyn_api::copysign(self, b) }
    #[inline] fn ua_spacing(&self) -> Result<DynArray> { dyn_api::spacing(self) }
    #[inline] fn ua_nextafter(&self, b: &Self) -> Result<DynArray> { dyn_api::nextafter(self, b) }
    #[inline] fn ua_frexp(&self) -> Result<(DynArray, DynArray)> { dyn_api::frexp(self) }
    #[inline] fn ua_modf(&self) -> Result<(DynArray, DynArray)> { dyn_api::modf(self) }
    #[inline] fn ua_isfinite(&self) -> Result<DynArray> { dyn_api::isfinite(self) }
    #[inline] fn ua_isinf(&self) -> Result<DynArray> { dyn_api::isinf(self) }
    #[inline] fn ua_isnan(&self) -> Result<DynArray> { dyn_api::isnan(self) }
    #[inline] fn ua_isneginf(&self) -> Result<DynArray> { dyn_api::isneginf(self) }
    #[inline] fn ua_isposinf(&self) -> Result<DynArray> { dyn_api::isposinf(self) }
    #[inline]
    fn ua_nan_to_num(&self, nan: Option<f64>, posinf: Option<f64>, neginf: Option<f64>) -> Result<DynArray> {
        dyn_api::nan_to_num(self, nan, posinf, neginf)
    }
    #[inline] fn ua_clip(&self, a_min: Option<&Self>, a_max: Option<&Self>) -> Result<DynArray> { dyn_api::clip(self, a_min, a_max) }
    #[inline] fn ua_heaviside(&self, b: &Self) -> Result<DynArray> { dyn_api::heaviside(self, b) }
    #[inline] fn ua_sinc(&self) -> Result<DynArray> { dyn_api::sinc(self) }
    #[inline] fn ua_i0(&self) -> Result<DynArray> { dyn_api::i0(self) }
    #[inline] fn ua_maximum(&self, b: &Self) -> Result<DynArray> { dyn_api::maximum(self, b) }
    #[inline] fn ua_minimum(&self, b: &Self) -> Result<DynArray> { dyn_api::minimum(self, b) }
    #[inline] fn ua_fmax(&self, b: &Self) -> Result<DynArray> { dyn_api::fmax(self, b) }
    #[inline] fn ua_fmin(&self, b: &Self) -> Result<DynArray> { dyn_api::fmin(self, b) }
    #[inline] fn ua_gcd(&self, b: &Self) -> Result<DynArray> { dyn_api::gcd(self, b) }
    #[inline] fn ua_lcm(&self, b: &Self) -> Result<DynArray> { dyn_api::lcm(self, b) }

    #[inline] fn ua_real(&self) -> Result<DynArray> { dyn_api::real(self) }
    #[inline] fn ua_imag(&self) -> Result<DynArray> { dyn_api::imag(self) }
    #[inline] fn ua_conj(&self) -> Result<DynArray> { dyn_api::conj(self) }
    #[inline] fn ua_angle(&self, deg: bool) -> Result<DynArray> { dyn_api::angle(self, deg) }
    #[inline] fn ua_bitand(&self, b: &Self) -> Result<DynArray> { dyn_api::bitwise_and(self, b) }
    #[inline] fn ua_bitor(&self, b: &Self) -> Result<DynArray> { dyn_api::bitwise_or(self, b) }
    #[inline] fn ua_bitxor(&self, b: &Self) -> Result<DynArray> { dyn_api::bitwise_xor(self, b) }
    #[inline] fn ua_invert(&self) -> Result<DynArray> { dyn_api::invert(self) }
    #[inline] fn ua_lshift(&self, b: &Self) -> Result<DynArray> { dyn_api::left_shift(self, b) }
    #[inline] fn ua_rshift(&self, b: &Self) -> Result<DynArray> { dyn_api::right_shift(self, b) }
    #[inline] fn ua_bitwise_count(&self) -> Result<DynArray> { dyn_api::bitwise_count(self) }
    #[inline] fn ua_eq(&self, b: &Self) -> Result<DynArray> { dyn_api::equal(self, b) }
    #[inline] fn ua_ne(&self, b: &Self) -> Result<DynArray> { dyn_api::not_equal(self, b) }
    #[inline] fn ua_lt(&self, b: &Self) -> Result<DynArray> { dyn_api::less(self, b) }
    #[inline] fn ua_le(&self, b: &Self) -> Result<DynArray> { dyn_api::less_equal(self, b) }
    #[inline] fn ua_gt(&self, b: &Self) -> Result<DynArray> { dyn_api::greater(self, b) }
    #[inline] fn ua_ge(&self, b: &Self) -> Result<DynArray> { dyn_api::greater_equal(self, b) }
    #[inline]
    fn ua_isclose(&self, b: &Self, rtol: f64, atol: f64, equal_nan: bool) -> Result<DynArray> {
        dyn_api::isclose(self, b, rtol, atol, equal_nan)
    }
    #[inline]
    fn ua_allclose(&self, b: &Self, rtol: f64, atol: f64, equal_nan: bool) -> Result<bool> {
        dyn_api::allclose(self, b, rtol, atol, equal_nan)
    }
    #[inline] fn ua_array_equal(&self, b: &Self, equal_nan: bool) -> bool { dyn_api::array_equal(self, b, equal_nan) }
    #[inline] fn ua_array_equiv(&self, b: &Self) -> bool { dyn_api::array_equiv(self, b) }
    #[inline] fn ua_logical_not(&self) -> Result<DynArray> { dyn_api::logical_not(self) }
}

impl UfuncArrayExt for DynArray {
    type FrexpExpOut = DynArray;
    #[inline]
    fn ua_frexp_pair(&self) -> Result<(DynArray, DynArray)> {
        dyn_api::frexp(self)
    }
}

/// Heterogeneous operand pair for `ldexp(a, exp)` across `NdArray<T>` and `DynArray`.
pub trait LdexpOperands<E>: private::Sealed {
    type Output;
    fn ua_ldexp(&self, exp: &E) -> Result<Self::Output>;
}

impl<T: UfuncScalar, E: UfuncScalar> LdexpOperands<NdArray<E>> for NdArray<T> {
    type Output = NdArray<T::FloatOut>;
    #[inline]
    fn ua_ldexp(&self, exp: &NdArray<E>) -> Result<Self::Output> {
        typed::ldexp(self, exp)
    }
}

impl LdexpOperands<DynArray> for DynArray {
    type Output = DynArray;
    #[inline]
    fn ua_ldexp(&self, exp: &DynArray) -> Result<DynArray> {
        dyn_api::ldexp(self, exp)
    }
}

/// Heterogeneous operand pair for `logical_and` / `logical_or` / `logical_xor`.
pub trait LogicalOperands<B>: private::Sealed {
    type Output;
    fn ua_logical_and(&self, b: &B) -> Result<Self::Output>;
    fn ua_logical_or(&self, b: &B) -> Result<Self::Output>;
    fn ua_logical_xor(&self, b: &B) -> Result<Self::Output>;
}

impl<A: UfuncScalar, B: UfuncScalar> LogicalOperands<NdArray<B>> for NdArray<A> {
    type Output = NdArray<bool>;
    #[inline] fn ua_logical_and(&self, b: &NdArray<B>) -> Result<NdArray<bool>> { typed::logical_and(self, b) }
    #[inline] fn ua_logical_or(&self, b: &NdArray<B>) -> Result<NdArray<bool>> { typed::logical_or(self, b) }
    #[inline] fn ua_logical_xor(&self, b: &NdArray<B>) -> Result<NdArray<bool>> { typed::logical_xor(self, b) }
}

impl LogicalOperands<DynArray> for DynArray {
    type Output = DynArray;
    #[inline] fn ua_logical_and(&self, b: &DynArray) -> Result<DynArray> { dyn_api::logical_and(self, b) }
    #[inline] fn ua_logical_or(&self, b: &DynArray) -> Result<DynArray> { dyn_api::logical_or(self, b) }
    #[inline] fn ua_logical_xor(&self, b: &DynArray) -> Result<DynArray> { dyn_api::logical_xor(self, b) }
}

macro_rules! pub_bin_fn {
    ($fn_name:ident, $method:ident, $assoc:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<A: UfuncArray>(a: &A, b: &A) -> Result<A::$assoc> {
            a.$method(b)
        }
    };
}

macro_rules! pub_un_fn {
    ($fn_name:ident, $method:ident, $assoc:ident, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $fn_name<A: UfuncArray>(a: &A) -> Result<A::$assoc> {
            a.$method()
        }
    };
}

// 1. Arithmetic
pub_bin_fn!(add, ua_add, SameOut, "Element-wise addition.");
pub_bin_fn!(subtract, ua_sub, SameOut, "Element-wise subtraction.");
pub_bin_fn!(multiply, ua_mul, SameOut, "Element-wise multiplication.");
pub_bin_fn!(true_divide, ua_true_div, TrueDivOut, "Element-wise true division.");
pub use true_divide as divide;
pub_bin_fn!(floor_divide, ua_floor_div, BoolToI8Out, "Element-wise floor division.");
pub_bin_fn!(remainder, ua_rem, BoolToI8Out, "Element-wise remainder (NumPy `remainder`/`mod`).");
pub use remainder as modulo;
pub use remainder as r#mod;
pub_bin_fn!(fmod, ua_fmod, BoolToI8Out, "Element-wise C `fmod` remainder.");
#[inline]
pub fn divmod<A: UfuncArray>(a: &A, b: &A) -> Result<(A::BoolToI8Out, A::BoolToI8Out)> {
    a.ua_divmod(b)
}
pub_bin_fn!(power, ua_pow, BoolToI8Out, "Element-wise exponentiation.");
pub_bin_fn!(float_power, ua_float_pow, FloatPowerOut, "Element-wise float64/complex128 exponentiation.");
pub_un_fn!(negative, ua_neg, SameOut, "Element-wise negation.");
pub_un_fn!(positive, ua_pos, SameOut, "Element-wise positive (copy).");
pub_un_fn!(abs, ua_abs, RealOut, "Element-wise absolute value.");
pub use abs as absolute;
pub_un_fn!(fabs, ua_fabs, FloatOut, "Element-wise real floating absolute value.");
pub_un_fn!(square, ua_square, BoolToI8Out, "Element-wise square.");
pub_un_fn!(reciprocal, ua_recip, BoolToI8Out, "Element-wise reciprocal.");
pub_un_fn!(sqrt, ua_sqrt, FloatOut, "Element-wise square root.");
pub_un_fn!(cbrt, ua_cbrt, FloatOut, "Element-wise cube root.");

// 2. Exp / Log
pub_un_fn!(exp, ua_exp, FloatOut, "Element-wise `e^x`.");
pub_un_fn!(exp2, ua_exp2, FloatOut, "Element-wise `2^x`.");
pub_un_fn!(expm1, ua_expm1, FloatOut, "Element-wise `e^x - 1`.");
pub_un_fn!(log, ua_log, FloatOut, "Element-wise natural logarithm.");
pub_un_fn!(log2, ua_log2, FloatOut, "Element-wise base-2 logarithm.");
pub_un_fn!(log10, ua_log10, FloatOut, "Element-wise base-10 logarithm.");
pub_un_fn!(log1p, ua_log1p, FloatOut, "Element-wise `ln(1 + x)`.");
pub_bin_fn!(logaddexp, ua_logaddexp, FloatOut, "Element-wise `ln(exp(x) + exp(y))`.");
pub_bin_fn!(logaddexp2, ua_logaddexp2, FloatOut, "Element-wise `log2(2^x + 2^y)`.");

// 3. Trig / Hyperbolic
pub_un_fn!(sin, ua_sin, FloatOut, "Element-wise sine.");
pub_un_fn!(cos, ua_cos, FloatOut, "Element-wise cosine.");
pub_un_fn!(tan, ua_tan, FloatOut, "Element-wise tangent.");
pub_un_fn!(arcsin, ua_arcsin, FloatOut, "Element-wise inverse sine.");
pub use arcsin as asin;
pub_un_fn!(arccos, ua_arccos, FloatOut, "Element-wise inverse cosine.");
pub use arccos as acos;
pub_un_fn!(arctan, ua_arctan, FloatOut, "Element-wise inverse tangent.");
pub use arctan as atan;
pub_bin_fn!(arctan2, ua_arctan2, FloatOut, "Element-wise 2-argument arc tangent.");
pub use arctan2 as atan2;
pub_bin_fn!(hypot, ua_hypot, FloatOut, "Element-wise Euclidean hypotenuse.");
pub_un_fn!(sinh, ua_sinh, FloatOut, "Element-wise hyperbolic sine.");
pub_un_fn!(cosh, ua_cosh, FloatOut, "Element-wise hyperbolic cosine.");
pub_un_fn!(tanh, ua_tanh, FloatOut, "Element-wise hyperbolic tangent.");
pub_un_fn!(arcsinh, ua_arcsinh, FloatOut, "Element-wise inverse hyperbolic sine.");
pub use arcsinh as asinh;
pub_un_fn!(arccosh, ua_arccosh, FloatOut, "Element-wise inverse hyperbolic cosine.");
pub use arccosh as acosh;
pub_un_fn!(arctanh, ua_arctanh, FloatOut, "Element-wise inverse hyperbolic tangent.");
pub use arctanh as atanh;
pub_un_fn!(deg2rad, ua_deg2rad, FloatOut, "Convert degrees to radians.");
pub use deg2rad as radians;
pub_un_fn!(rad2deg, ua_rad2deg, FloatOut, "Convert radians to degrees.");
pub use rad2deg as degrees;

// 4. Rounding / Float / Special
pub_un_fn!(floor, ua_floor, SameOut, "Element-wise floor.");
pub_un_fn!(ceil, ua_ceil, SameOut, "Element-wise ceiling.");
pub_un_fn!(trunc, ua_trunc, SameOut, "Element-wise truncation toward zero.");
pub use trunc as fix;
pub_un_fn!(rint, ua_rint, FloatOut, "Element-wise round to nearest integer (ties to even).");
#[inline]
pub fn round<A: UfuncArray>(a: &A, decimals: i32) -> Result<A::RoundOut> {
    a.ua_round(decimals)
}
pub use round as around;
pub_un_fn!(sign, ua_sign, SameOut, "Element-wise sign.");
pub_un_fn!(signbit, ua_signbit, BoolOut, "Element-wise IEEE sign bit.");
pub_bin_fn!(copysign, ua_copysign, FloatOut, "Copy sign of second argument to magnitude of first.");
pub_un_fn!(spacing, ua_spacing, FloatOut, "Distance to nearest adjacent float.");
pub_bin_fn!(nextafter, ua_nextafter, FloatOut, "Next representable float after x towards y.");
#[inline]
pub fn ldexp<A: LdexpOperands<E>, E>(a: &A, exp: &E) -> Result<A::Output> {
    a.ua_ldexp(exp)
}
#[inline]
pub fn frexp<A: UfuncArrayExt>(a: &A) -> Result<(A::FloatOut, A::FrexpExpOut)> {
    a.ua_frexp_pair()
}
#[inline]
pub fn modf<A: UfuncArray>(a: &A) -> Result<(A::FloatOut, A::FloatOut)> {
    a.ua_modf()
}
pub_un_fn!(isfinite, ua_isfinite, BoolOut, "Test finiteness element-wise.");
pub_un_fn!(isinf, ua_isinf, BoolOut, "Test infinity element-wise.");
pub_un_fn!(isnan, ua_isnan, BoolOut, "Test NaN element-wise.");
pub_un_fn!(isneginf, ua_isneginf, BoolOut, "Test negative infinity element-wise.");
pub_un_fn!(isposinf, ua_isposinf, BoolOut, "Test positive infinity element-wise.");
#[inline]
pub fn nan_to_num<A: UfuncArray>(
    a: &A,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> Result<A::SameOut> {
    a.ua_nan_to_num(nan, posinf, neginf)
}
#[inline]
pub fn clip<A: UfuncArray>(a: &A, a_min: Option<&A>, a_max: Option<&A>) -> Result<A::SameOut> {
    a.ua_clip(a_min, a_max)
}
pub_bin_fn!(heaviside, ua_heaviside, FloatOut, "Heaviside step function.");
pub_un_fn!(sinc, ua_sinc, TrueDivOut, "Normalized sinc function.");
pub_un_fn!(i0, ua_i0, TrueDivOut, "Modified Bessel function of the first kind, order 0.");
pub_bin_fn!(maximum, ua_maximum, SameOut, "Element-wise maximum (propagates NaNs).");
pub_bin_fn!(minimum, ua_minimum, SameOut, "Element-wise minimum (propagates NaNs).");
pub_bin_fn!(fmax, ua_fmax, SameOut, "Element-wise maximum (ignores NaNs).");
pub_bin_fn!(fmin, ua_fmin, SameOut, "Element-wise minimum (ignores NaNs).");
pub_bin_fn!(gcd, ua_gcd, SameOut, "Element-wise greatest common divisor.");
pub_bin_fn!(lcm, ua_lcm, SameOut, "Element-wise least common multiple.");

// 5. Complex / Bitwise / Comparison / Logical
pub_un_fn!(real, ua_real, RealOut, "Real part of elements.");
pub_un_fn!(imag, ua_imag, RealOut, "Imaginary part of elements.");
pub_un_fn!(conj, ua_conj, BoolToI8Out, "Complex conjugate element-wise.");
pub use conj as conjugate;
#[inline]
pub fn angle<A: UfuncArray>(a: &A, deg: bool) -> Result<A::AngleOut> {
    a.ua_angle(deg)
}
pub_bin_fn!(bitwise_and, ua_bitand, SameOut, "Element-wise bitwise AND.");
pub_bin_fn!(bitwise_or, ua_bitor, SameOut, "Element-wise bitwise OR.");
pub_bin_fn!(bitwise_xor, ua_bitxor, SameOut, "Element-wise bitwise XOR.");
pub_un_fn!(invert, ua_invert, SameOut, "Element-wise bitwise NOT / invert.");
pub use invert as bitwise_not;
pub_bin_fn!(left_shift, ua_lshift, BoolToI8Out, "Element-wise left bit shift.");
pub_bin_fn!(right_shift, ua_rshift, BoolToI8Out, "Element-wise right bit shift.");
pub_un_fn!(bitwise_count, ua_bitwise_count, PopcntOut, "Count 1-bits (popcount) in integers.");
pub_bin_fn!(equal, ua_eq, BoolOut, "Element-wise `==`.");
pub_bin_fn!(not_equal, ua_ne, BoolOut, "Element-wise `!=`.");
pub_bin_fn!(less, ua_lt, BoolOut, "Element-wise `<`.");
pub_bin_fn!(less_equal, ua_le, BoolOut, "Element-wise `<=`.");
pub_bin_fn!(greater, ua_gt, BoolOut, "Element-wise `>`.");
pub_bin_fn!(greater_equal, ua_ge, BoolOut, "Element-wise `>=`.");

#[inline]
pub fn isclose<A: UfuncArray>(
    a: &A,
    b: &A,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<A::BoolOut> {
    a.ua_isclose(b, rtol, atol, equal_nan)
}

#[inline]
pub fn allclose<A: UfuncArray>(
    a: &A,
    b: &A,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<bool> {
    a.ua_allclose(b, rtol, atol, equal_nan)
}

#[inline]
pub fn array_equal<A: UfuncArray>(a: &A, b: &A, equal_nan: bool) -> bool {
    a.ua_array_equal(b, equal_nan)
}

#[inline]
pub fn array_equiv<A: UfuncArray>(a: &A, b: &A) -> bool {
    a.ua_array_equiv(b)
}

#[inline]
pub fn logical_and<A: LogicalOperands<B>, B>(a: &A, b: &B) -> Result<A::Output> {
    a.ua_logical_and(b)
}

#[inline]
pub fn logical_or<A: LogicalOperands<B>, B>(a: &A, b: &B) -> Result<A::Output> {
    a.ua_logical_or(b)
}

#[inline]
pub fn logical_xor<A: LogicalOperands<B>, B>(a: &A, b: &B) -> Result<A::Output> {
    a.ua_logical_xor(b)
}

pub_un_fn!(logical_not, ua_logical_not, BoolOut, "Element-wise logical NOT.");

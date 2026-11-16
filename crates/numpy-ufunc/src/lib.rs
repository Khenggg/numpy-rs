//! Universal functions (ufuncs) for `numpy-rs`.
//!
//! Provides both:
//! - Zero-cost generic/specialized functions for [`NdArray<T>`] (`T: UfuncScalar`)
//! - Dynamic functions for [`DynArray`] with exact NumPy 2.x type promotion & broadcasting
//! - Higher-order ufunc operations: `reduce`, `reduce_axes`, `accumulate`, `outer`, `at`
//! - Runtime-dispatched x86_64 AVX2+FMA kernels in [`simd`] and Rayon parallel chunking

pub mod dyn_ufunc;
mod exec;
pub mod higher_order;
mod scalar_ops;
pub mod simd;
pub mod traits;
pub mod typed_ufunc;

pub use dyn_ufunc as dyn_api;
pub use typed_ufunc as typed;

pub use half::f16;
pub use num_complex::{Complex32, Complex64};
pub use numpy_types::*;

pub use higher_order::{
    accumulate, accumulate_with, at, at_with, outer, outer_with, reduce, reduce_axes,
    reduce_with, unary_at, unary_at_with, AtIndex, BinaryOp, BinaryUfunc, HigherOrderScalar,
    ScalarValue, UnaryOp, UnaryUfunc, ABS, ABSOLUTE, ACOS, ACOSH, ADD, ARCCOS, ARCCOSH, ARCSIN,
    ARCSINH, ARCTAN, ARCTAN2, ARCTANH, ASIN, ASINH, ATAN, ATAN2, ATANH, BITWISE_AND, BITWISE_NOT,
    BITWISE_OR, BITWISE_XOR, CBRT, CEIL, CONJ, CONJUGATE, COPYSIGN, COS, COSH, DEG2RAD, DEGREES,
    DIVIDE, EQUAL, EXP, EXP2, EXPM1, FABS, FIX, FLOAT_POWER, FLOOR, FLOOR_DIVIDE, FMAX, FMIN, FMOD,
    GCD, GREATER, GREATER_EQUAL, HEAVISIDE, HYPOT, I0, IMAG, INVERT, LCM, LEFT_SHIFT, LESS,
    LESS_EQUAL, LOG, LOG10, LOG1P, LOG2, LOGADDEXP, LOGADDEXP2, LOGICAL_AND, LOGICAL_NOT,
    LOGICAL_OR, LOGICAL_XOR, MAXIMUM, MINIMUM, MOD, MODULO, MULTIPLY, NEGATIVE, NEXTAFTER,
    NOT_EQUAL, POSITIVE, POWER, RAD2DEG, RADIANS, REAL, RECIPROCAL, REMAINDER, RIGHT_SHIFT, RINT,
    SIGN, SIN, SINC, SINH, SPACING, SQRT, SQUARE, SUBTRACT, TAN, TANH, TRUE_DIVIDE, TRUNC,
};
pub use traits::{
    abs, absolute, acos, acosh, add, allclose, angle, arccos, arccosh, arcsin, arcsinh, arctan,
    arctan2, arctanh, around, array_equal, array_equiv, asin, asinh, atan, atan2, atanh,
    bitwise_and, bitwise_count, bitwise_not, bitwise_or, bitwise_xor, cbrt, ceil, clip, conj,
    conjugate, copysign, cos, cosh, deg2rad, degrees, divide, divmod, equal, exp, exp2, expm1,
    fabs, fix, float_power, floor, floor_divide, fmax, fmin, fmod, frexp, gcd, greater,
    greater_equal, heaviside, hypot, i0, imag, invert, isclose, isfinite, isinf, isnan, isneginf,
    isposinf, lcm, ldexp, left_shift, less, less_equal, log, log10, log1p, log2, logaddexp,
    logaddexp2, logical_and, logical_not, logical_or, logical_xor, maximum, minimum, modf,
    modulo, multiply, nan_to_num, negative, nextafter, not_equal, positive, power, r#mod, rad2deg,
    radians, real, reciprocal, remainder, right_shift, rint, round, sign, signbit, sin, sinc, sinh,
    spacing, sqrt, square, subtract, tan, tanh, true_divide, trunc, LdexpOperands,
    LogicalOperands, UfuncArray, UfuncArrayExt,
};
pub use typed_ufunc::{clip_scalar, UfuncScalar};

//! Exact NumPy 2.x scalar math kernels and special-value semantics.

use half::f16;
use num_complex::{Complex, Complex32, Complex64};
use numpy_types::{NumPyError, Result};
use std::f64::consts::{LN_2, LOG10_E, LOG2_E, PI};

// ============================================================================
// Integer Floor Divide / Remainder / Divmod / Power / Shifts / Bitwise Count
// ============================================================================

macro_rules! impl_signed_divmod {
    ($fn_div:ident, $fn_rem:ident, $fn_divmod:ident, $t:ty) => {
        #[inline]
        pub fn $fn_div(a: $t, b: $t) -> $t {
            if b == 0 {
                return 0;
            }
            let q = a.wrapping_div(b);
            let r = a.wrapping_rem(b);
            if r != 0 && ((r < 0) != (b < 0)) {
                q.wrapping_sub(1)
            } else {
                q
            }
        }

        #[inline]
        pub fn $fn_rem(a: $t, b: $t) -> $t {
            if b == 0 {
                return 0;
            }
            let r = a.wrapping_rem(b);
            if r != 0 && ((r < 0) != (b < 0)) {
                r.wrapping_add(b)
            } else {
                r
            }
        }

        #[inline]
        pub fn $fn_divmod(a: $t, b: $t) -> ($t, $t) {
            if b == 0 {
                return (0, 0);
            }
            let q = a.wrapping_div(b);
            let r = a.wrapping_rem(b);
            if r != 0 && ((r < 0) != (b < 0)) {
                (q.wrapping_sub(1), r.wrapping_add(b))
            } else {
                (q, r)
            }
        }
    };
}

impl_signed_divmod!(floor_div_i8, floor_rem_i8, divmod_i8, i8);
impl_signed_divmod!(floor_div_i16, floor_rem_i16, divmod_i16, i16);
impl_signed_divmod!(floor_div_i32, floor_rem_i32, divmod_i32, i32);
impl_signed_divmod!(floor_div_i64, floor_rem_i64, divmod_i64, i64);

macro_rules! impl_unsigned_divmod {
    ($fn_div:ident, $fn_rem:ident, $fn_divmod:ident, $t:ty) => {
        #[inline]
        pub fn $fn_div(a: $t, b: $t) -> $t {
            if b == 0 { 0 } else { a / b }
        }

        #[inline]
        pub fn $fn_rem(a: $t, b: $t) -> $t {
            if b == 0 { 0 } else { a % b }
        }

        #[inline]
        pub fn $fn_divmod(a: $t, b: $t) -> ($t, $t) {
            if b == 0 { (0, 0) } else { (a / b, a % b) }
        }
    };
}

impl_unsigned_divmod!(floor_div_u8, floor_rem_u8, divmod_u8, u8);
impl_unsigned_divmod!(floor_div_u16, floor_rem_u16, divmod_u16, u16);
impl_unsigned_divmod!(floor_div_u32, floor_rem_u32, divmod_u32, u32);
impl_unsigned_divmod!(floor_div_u64, floor_rem_u64, divmod_u64, u64);

macro_rules! impl_signed_pow {
    ($fn_pow:ident, $t:ty) => {
        #[inline]
        pub fn $fn_pow(mut base: $t, mut exp: $t) -> Result<$t> {
            if exp < 0 {
                return Err(NumPyError::InvalidValue(
                    "Integers to negative integer powers are not allowed.".into(),
                ));
            }
            if exp == 0 || base == 1 {
                return Ok(1);
            }
            if exp == 1 {
                return Ok(base);
            }
            if exp == 2 {
                return Ok(base.wrapping_mul(base));
            }
            let first_bit = (exp & 1) != 0;
            exp >>= 1;
            let mut out: $t = if first_bit { base } else { 1 };
            while exp > 0 {
                base = base.wrapping_mul(base);
                if (exp & 1) != 0 {
                    out = out.wrapping_mul(base);
                }
                exp >>= 1;
            }
            Ok(out)
        }
    };
}

impl_signed_pow!(pow_i8, i8);
impl_signed_pow!(pow_i16, i16);
impl_signed_pow!(pow_i32, i32);
impl_signed_pow!(pow_i64, i64);

macro_rules! impl_unsigned_pow {
    ($fn_pow:ident, $t:ty) => {
        #[inline]
        pub fn $fn_pow(mut base: $t, mut exp: $t) -> $t {
            if exp == 0 || base == 1 {
                return 1;
            }
            if exp == 1 {
                return base;
            }
            if exp == 2 {
                return base.wrapping_mul(base);
            }
            let first_bit = (exp & 1) != 0;
            exp >>= 1;
            let mut out: $t = if first_bit { base } else { 1 };
            while exp > 0 {
                base = base.wrapping_mul(base);
                if (exp & 1) != 0 {
                    out = out.wrapping_mul(base);
                }
                exp >>= 1;
            }
            out
        }
    };
}

impl_unsigned_pow!(pow_u8, u8);
impl_unsigned_pow!(pow_u16, u16);
impl_unsigned_pow!(pow_u32, u32);
impl_unsigned_pow!(pow_u64, u64);

macro_rules! impl_int_shifts {
    ($fn_lsh:ident, $fn_rsh:ident, $t:ty, $is_signed:expr) => {
        #[inline]
        pub fn $fn_lsh(a: $t, b: $t) -> $t {
            let bits = (std::mem::size_of::<$t>() * 8) as u64;
            if (b as u64) < bits {
                a.wrapping_shl(b as u32)
            } else {
                0
            }
        }

        #[inline]
        #[allow(unused_comparisons)]
        pub fn $fn_rsh(a: $t, b: $t) -> $t {
            let bits = (std::mem::size_of::<$t>() * 8) as u64;
            if (b as u64) < bits {
                a.wrapping_shr(b as u32)
            } else if $is_signed && a < 0 {
                !0 as $t
            } else {
                0
            }
        }
    };
}

impl_int_shifts!(lshift_i8, rshift_i8, i8, true);
impl_int_shifts!(lshift_i16, rshift_i16, i16, true);
impl_int_shifts!(lshift_i32, rshift_i32, i32, true);
impl_int_shifts!(lshift_i64, rshift_i64, i64, true);
impl_int_shifts!(lshift_u8, rshift_u8, u8, false);
impl_int_shifts!(lshift_u16, rshift_u16, u16, false);
impl_int_shifts!(lshift_u32, rshift_u32, u32, false);
impl_int_shifts!(lshift_u64, rshift_u64, u64, false);

macro_rules! impl_gcd_lcm {
    ($fn_gcd:ident, $fn_lcm:ident, $t:ty, $ut:ty) => {
        #[inline]
        pub fn $fn_gcd(a: $t, b: $t) -> $t {
            let mut x = a.unsigned_abs() as $ut;
            let mut y = b.unsigned_abs() as $ut;
            while x != 0 {
                let c = x;
                x = y % x;
                y = c;
            }
            y as $t
        }

        #[inline]
        pub fn $fn_lcm(a: $t, b: $t) -> $t {
            let x = a.unsigned_abs() as $ut;
            let y = b.unsigned_abs() as $ut;
            let mut gx = x;
            let mut gy = y;
            while gx != 0 {
                let c = gx;
                gx = gy % gx;
                gy = c;
            }
            if gy == 0 {
                0
            } else {
                ((x / gy).wrapping_mul(y)) as $t
            }
        }
    };
}

impl_gcd_lcm!(gcd_i8, lcm_i8, i8, u8);
impl_gcd_lcm!(gcd_i16, lcm_i16, i16, u16);
impl_gcd_lcm!(gcd_i32, lcm_i32, i32, u32);
impl_gcd_lcm!(gcd_i64, lcm_i64, i64, u64);

macro_rules! impl_ugcd_ulcm {
    ($fn_gcd:ident, $fn_lcm:ident, $t:ty) => {
        #[inline]
        pub fn $fn_gcd(mut a: $t, mut b: $t) -> $t {
            while a != 0 {
                let c = a;
                a = b % a;
                b = c;
            }
            b
        }

        #[inline]
        pub fn $fn_lcm(a: $t, b: $t) -> $t {
            let g = $fn_gcd(a, b);
            if g == 0 { 0 } else { (a / g).wrapping_mul(b) }
        }
    };
}

impl_ugcd_ulcm!(gcd_u8, lcm_u8, u8);
impl_ugcd_ulcm!(gcd_u16, lcm_u16, u16);
impl_ugcd_ulcm!(gcd_u32, lcm_u32, u32);
impl_ugcd_ulcm!(gcd_u64, lcm_u64, u64);

#[inline]
pub fn int_recip_i8(a: i8) -> i8 {
    if a == 0 { i8::MIN } else { (1.0 / (a as f64)) as i8 }
}
#[inline]
pub fn int_recip_i16(a: i16) -> i16 {
    if a == 0 { i16::MIN } else { (1.0 / (a as f64)) as i16 }
}
#[inline]
pub fn int_recip_i32(a: i32) -> i32 {
    if a == 0 { i32::MIN } else { (1.0 / (a as f64)) as i32 }
}
#[inline]
pub fn int_recip_i64(a: i64) -> i64 {
    if a == 0 { i64::MIN } else { (1.0 / (a as f64)) as i64 }
}
#[inline]
pub fn int_recip_u8(a: u8) -> u8 {
    if a == 0 { 0 } else { (1.0 / (a as f64)) as u8 }
}
#[inline]
pub fn int_recip_u16(a: u16) -> u16 {
    if a == 0 { 0 } else { (1.0 / (a as f64)) as u16 }
}
#[inline]
pub fn int_recip_u32(a: u32) -> u32 {
    if a == 0 { 0 } else { (1.0 / (a as f64)) as u32 }
}
#[inline]
pub fn int_recip_u64(a: u64) -> u64 {
    if a == 0 { 0 } else { (1.0 / (a as f64)) as u64 }
}

// ============================================================================
// Float Divmod / Floor Divide / Remainder / Spacing / Nextafter / Frexp / Modf
// ============================================================================

#[inline]
pub fn divmod_f64(a: f64, b: f64) -> (f64, f64) {
    let mut modulus = a % b;
    if b == 0.0 {
        return (a / b, modulus);
    }
    if a.is_nan() || b.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let div = (a - modulus) / b;
    if modulus != 0.0 {
        if (b < 0.0) != (modulus < 0.0) {
            modulus += b;
            let mut d = div - 1.0;
            let floordiv = d.floor();
            if d - floordiv > 0.5 {
                d = floordiv + 1.0;
            } else {
                d = floordiv;
            }
            return (d, modulus);
        }
    } else {
        modulus = 0.0_f64.copysign(b);
    }
    let floordiv = if div != 0.0 {
        let fd = div.floor();
        if div - fd > 0.5 { fd + 1.0 } else { fd }
    } else {
        0.0_f64.copysign(a / b)
    };
    (floordiv, modulus)
}

#[inline]
pub fn floor_divide_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return a / b;
    }
    divmod_f64(a, b).0
}

#[inline]
pub fn remainder_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return a % b;
    }
    divmod_f64(a, b).1
}

#[inline]
pub fn divmod_f32(a: f32, b: f32) -> (f32, f32) {
    let mut modulus = a % b;
    if b == 0.0 {
        return (a / b, modulus);
    }
    if a.is_nan() || b.is_nan() {
        return (f32::NAN, f32::NAN);
    }
    let div = (a - modulus) / b;
    if modulus != 0.0 {
        if (b < 0.0) != (modulus < 0.0) {
            modulus += b;
            let d = div - 1.0;
            let fd = d.floor();
            let floordiv = if d - fd > 0.5 { fd + 1.0 } else { fd };
            return (floordiv, modulus);
        }
    } else {
        modulus = 0.0_f32.copysign(b);
    }
    let floordiv = if div != 0.0 {
        let fd = div.floor();
        if div - fd > 0.5 { fd + 1.0 } else { fd }
    } else {
        0.0_f32.copysign(a / b)
    };
    (floordiv, modulus)
}

#[inline]
pub fn floor_divide_f32(a: f32, b: f32) -> f32 {
    if b == 0.0 { a / b } else { divmod_f32(a, b).0 }
}

#[inline]
pub fn remainder_f32(a: f32, b: f32) -> f32 {
    if b == 0.0 { a % b } else { divmod_f32(a, b).1 }
}

#[inline]
pub fn nextafter_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f64::from_bits(1).copysign(y);
    }
    let bits = x.to_bits();
    let move_away = (x < y) == (x > 0.0);
    let next_bits = if move_away { bits + 1 } else { bits - 1 };
    f64::from_bits(next_bits)
}

#[inline]
pub fn nextafter_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f32::from_bits(1).copysign(y);
    }
    let bits = x.to_bits();
    let move_away = (x < y) == (x > 0.0);
    let next_bits = if move_away { bits + 1 } else { bits - 1 };
    f32::from_bits(next_bits)
}

#[inline]
pub fn nextafter_f16(x: f16, y: f16) -> f16 {
    if x.is_nan() || y.is_nan() {
        return f16::NAN;
    }
    if x == y {
        return y;
    }
    if x == f16::ZERO {
        let sign = y.to_bits() & 0x8000;
        return f16::from_bits(sign | 1);
    }
    let xb = x.to_bits();
    let yb = y.to_bits();
    let next_bits = if (xb & 0x8000) == 0 {
        if (xb as i16) > (yb as i16) { xb - 1 } else { xb + 1 }
    } else if (yb & 0x8000) == 0 || (xb & 0x7fff) > (yb & 0x7fff) {
        xb - 1
    } else {
        xb + 1
    };
    f16::from_bits(next_bits)
}

#[inline]
pub fn spacing_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    let next = f64::from_bits(bits + 1);
    next - x
}

#[inline]
pub fn spacing_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    let next = f32::from_bits(bits + 1);
    next - x
}

#[inline]
pub fn spacing_f16(h: f16) -> f16 {
    let bits = h.to_bits();
    let h_exp = bits & 0x7c00;
    let h_sig = bits & 0x03ff;
    let ret = if h_exp == 0x7c00 {
        if h_sig == 0 { 0x7e00 } else { bits }
    } else if bits == 0x7bff {
        0x7c00 // +inf
    } else if (bits & 0x8000) != 0 && h_sig == 0 {
        if h_exp > 0x2c00 {
            h_exp - 0x2c00
        } else if h_exp > 0x0400 {
            1u16 << ((h_exp >> 10) - 2)
        } else {
            0x0001
        }
    } else if h_exp > 0x2800 {
        h_exp - 0x2800
    } else if h_exp > 0x0400 {
        1u16 << ((h_exp >> 10) - 1)
    } else {
        0x0001
    };
    f16::from_bits(ret)
}

#[inline]
pub fn frexp_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let mut bits = x.to_bits();
    let sign = bits & (1u64 << 63);
    let mut exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // subnormal: scale by 2^54
        let scaled = x * f64::from_bits(0x4350_0000_0000_0000);
        bits = scaled.to_bits();
        exp = ((bits >> 52) & 0x7ff) as i32 - 54;
    }
    let e = exp - 1022;
    let mant_bits = sign | (0x3fe_u64 << 52) | (bits & ((1u64 << 52) - 1));
    (f64::from_bits(mant_bits), e)
}

#[inline]
pub fn frexp_f32(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let mut bits = x.to_bits();
    let sign = bits & (1u32 << 31);
    let mut exp = ((bits >> 23) & 0xff) as i32;
    if exp == 0 {
        let scaled = x * f32::from_bits(0x4c00_0000); // 2^25
        bits = scaled.to_bits();
        exp = ((bits >> 23) & 0xff) as i32 - 25;
    }
    let e = exp - 126;
    let mant_bits = sign | (0x7e_u32 << 23) | (bits & ((1u32 << 23) - 1));
    (f32::from_bits(mant_bits), e)
}

#[inline]
pub fn ldexp_f64(x: f64, exp: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    if exp > 2098 {
        return if x.is_sign_negative() { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if exp < -2098 {
        return 0.0_f64.copysign(x);
    }
    // Split exp into up to three factors of 2^k that stay normal
    let mut res = x;
    let mut rem = exp as i32;
    while rem > 1023 {
        res *= f64::from_bits((1023 + 1023_u64) << 52); // 2^1023
        rem -= 1023;
    }
    while rem < -1022 {
        res *= f64::from_bits(1_u64 << 52); // 2^-1022
        rem += 1022;
    }
    let factor = f64::from_bits(((rem + 1023) as u64) << 52);
    res * factor
}

#[inline]
pub fn ldexp_f32(x: f32, exp: i64) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    if exp > 278 {
        return if x.is_sign_negative() { f32::NEG_INFINITY } else { f32::INFINITY };
    }
    if exp < -278 {
        return 0.0_f32.copysign(x);
    }
    let mut res = x;
    let mut rem = exp as i32;
    while rem > 127 {
        res *= f32::from_bits((127 + 127_u32) << 23);
        rem -= 127;
    }
    while rem < -126 {
        res *= f32::from_bits(1_u32 << 23); // 2^-126
        rem += 126;
    }
    let factor = f32::from_bits(((rem + 127) as u32) << 23);
    res * factor
}

#[inline]
pub fn modf_f64(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x.is_infinite() {
        return (0.0_f64.copysign(x), x);
    }
    let int_part = x.trunc();
    let frac_part = (x - int_part).copysign(x);
    (frac_part, int_part)
}

#[inline]
pub fn modf_f32(x: f32) -> (f32, f32) {
    if x.is_nan() {
        return (f32::NAN, f32::NAN);
    }
    if x.is_infinite() {
        return (0.0_f32.copysign(x), x);
    }
    let int_part = x.trunc();
    let frac_part = (x - int_part).copysign(x);
    (frac_part, int_part)
}

// ============================================================================
// Float Special Ops (sign, logaddexp, logaddexp2, heaviside, sinc, i0, round, max/min)
// ============================================================================

#[inline]
pub fn sign_f64(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

#[inline]
pub fn sign_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

#[inline]
pub fn logaddexp_f64(x: f64, y: f64) -> f64 {
    if x == y {
        x + LN_2
    } else {
        let tmp = x - y;
        if tmp > 0.0 {
            x + (-tmp).exp().ln_1p()
        } else if tmp <= 0.0 {
            y + tmp.exp().ln_1p()
        } else {
            tmp
        }
    }
}

#[inline]
pub fn logaddexp_f32(x: f32, y: f32) -> f32 {
    if x == y {
        x + (LN_2 as f32)
    } else {
        let tmp = x - y;
        if tmp > 0.0 {
            x + (-tmp).exp().ln_1p()
        } else if tmp <= 0.0 {
            y + tmp.exp().ln_1p()
        } else {
            tmp
        }
    }
}

#[inline]
pub fn logaddexp2_f64(x: f64, y: f64) -> f64 {
    if x == y {
        x + 1.0
    } else {
        let tmp = x - y;
        if tmp > 0.0 {
            x + LOG2_E * (-tmp).exp2().ln_1p()
        } else if tmp <= 0.0 {
            y + LOG2_E * tmp.exp2().ln_1p()
        } else {
            tmp
        }
    }
}

#[inline]
pub fn logaddexp2_f32(x: f32, y: f32) -> f32 {
    if x == y {
        x + 1.0
    } else {
        let tmp = x - y;
        if tmp > 0.0 {
            x + (LOG2_E as f32) * (-tmp).exp2().ln_1p()
        } else if tmp <= 0.0 {
            y + (LOG2_E as f32) * tmp.exp2().ln_1p()
        } else {
            tmp
        }
    }
}

#[inline]
pub fn heaviside_f64(x: f64, h0: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        h0
    } else if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

#[inline]
pub fn heaviside_f32(x: f32, h0: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x == 0.0 {
        h0
    } else if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

#[inline]
pub fn sinc_f64(x: f64) -> f64 {
    let px = PI * x;
    if px == 0.0 {
        // NumPy uses eps = finfo(float64).eps so sin(eps)/eps == 1.0
        1.0
    } else {
        px.sin() / px
    }
}

#[inline]
pub fn sinc_f32(x: f32) -> f32 {
    let px = (PI as f32) * x;
    if px == 0.0 {
        1.0
    } else {
        px.sin() / px
    }
}

#[inline]
pub fn sinc_f16(x: f16) -> f16 {
    let px = (PI as f32) * x.to_f32();
    let v = if px == 0.0 { 1.0 } else { px.sin() / px };
    f16::from_f32(v)
}

const I0_A: [f64; 30] = [
    -4.4153416464793395e-18,
    3.3307945188222384e-17,
    -2.431279846547955e-16,
    1.715391285555133e-15,
    -1.1685332877993451e-14,
    7.676185498604936e-14,
    -4.856446783111929e-13,
    2.95505266312964e-12,
    -1.726826291441556e-11,
    9.675809035373237e-11,
    -5.189795601635263e-10,
    2.6598237246823866e-09,
    -1.300025009986248e-08,
    6.046995022541919e-08,
    -2.670793853940612e-07,
    1.1173875391201037e-06,
    -4.4167383584587505e-06,
    1.6448448070728896e-05,
    -5.754195010082104e-05,
    0.00018850288509584165,
    -0.0005763755745385824,
    0.0016394756169413357,
    -0.004324309995050576,
    0.010546460394594998,
    -0.02373741480589947,
    0.04930528423967071,
    -0.09490109704804764,
    0.17162090152220877,
    -0.3046826723431984,
    0.6767952744094761,
];

const I0_B: [f64; 25] = [
    -7.233180487874754e-18,
    -4.830504485944182e-18,
    4.46562142029676e-17,
    3.461222867697461e-17,
    -2.8276239805165836e-16,
    -3.425485619677219e-16,
    1.7725601330565263e-15,
    3.8116806693526224e-15,
    -9.554846698828307e-15,
    -4.150569347287222e-14,
    1.54008621752141e-14,
    3.8527783827421426e-13,
    7.180124451383666e-13,
    -1.7941785315068062e-12,
    -1.3215811840447713e-11,
    -3.1499165279632416e-11,
    1.1889147107846439e-11,
    4.94060238822497e-10,
    3.3962320257083865e-09,
    2.266668990498178e-08,
    2.0489185894690638e-07,
    2.8913705208347567e-06,
    6.889758346916825e-05,
    0.0033691164782556943,
    0.8044904110141088,
];

#[inline]
fn chbevl(x: f64, vals: &[f64]) -> f64 {
    let mut b0 = vals[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in &vals[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}

#[inline]
pub fn i0_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax <= 8.0 {
        ax.exp() * chbevl(ax / 2.0 - 2.0, &I0_A)
    } else {
        ax.exp() * chbevl(32.0 / ax - 2.0, &I0_B) / ax.sqrt()
    }
}

#[inline]
pub(crate) fn isclose_f64_elem(x: f64, y: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if x.is_nan() && y.is_nan() {
        equal_nan
    } else if x == y {
        true
    } else if y.is_finite() {
        (x - y).abs() <= atol + rtol * y.abs()
    } else {
        false
    }
}

#[inline]
pub(crate) fn isclose_f32_elem(x: f32, y: f32, rtol: f32, atol: f32, equal_nan: bool) -> bool {
    if x.is_nan() && y.is_nan() {
        equal_nan
    } else if x == y {
        true
    } else if y.is_finite() {
        (x - y).abs() <= atol + rtol * y.abs()
    } else {
        false
    }
}

#[inline]
pub(crate) fn clip_ord<T: PartialOrd + Copy>(x: T, lo: T, hi: T) -> T {
    let t = if x < lo { lo } else { x };
    if t > hi { hi } else { t }
}

#[inline]
pub(crate) fn clip_f64_elem(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() || lo.is_nan() || hi.is_nan() {
        f64::NAN
    } else {
        clip_ord(x, lo, hi)
    }
}

#[inline]
pub(crate) fn clip_f32_elem(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() || lo.is_nan() || hi.is_nan() {
        f32::NAN
    } else {
        clip_ord(x, lo, hi)
    }
}

#[inline]
pub(crate) fn clip_f16_elem(x: f16, lo: f16, hi: f16) -> f16 {
    if x.is_nan() || lo.is_nan() || hi.is_nan() {
        f16::NAN
    } else {
        clip_ord(x, lo, hi)
    }
}

#[inline]
pub fn round_f64(x: f64, decimals: i32) -> f64 {
    if decimals == 0 {
        return if !x.is_finite() { x } else { x.round_ties_even() };
    }
    if decimals > 0 {
        if decimals > 308 {
            return f64::NAN;
        }
        if !x.is_finite() {
            return x;
        }
        let scale = 10.0_f64.powi(decimals);
        (x * scale).round_ties_even() / scale
    } else {
        let neg_d = decimals.unsigned_abs();
        if neg_d > 308 {
            return f64::NAN;
        }
        if !x.is_finite() {
            return x;
        }
        let scale = 10.0_f64.powi(neg_d as i32);
        (x / scale).round_ties_even() * scale
    }
}

#[inline]
pub fn round_f32(x: f32, decimals: i32) -> f32 {
    if decimals == 0 {
        return if !x.is_finite() { x } else { x.round_ties_even() };
    }
    if decimals > 0 {
        if decimals > 38 {
            return f32::NAN;
        }
        if !x.is_finite() {
            return x;
        }
        let scale = 10.0_f32.powi(decimals);
        (x * scale).round_ties_even() / scale
    } else {
        let neg_d = decimals.unsigned_abs();
        if neg_d > 38 {
            return f32::NAN;
        }
        if !x.is_finite() {
            return x;
        }
        let scale = 10.0_f32.powi(neg_d as i32);
        (x / scale).round_ties_even() * scale
    }
}

#[inline]
pub fn round_f16(x: f16, decimals: i32) -> f16 {
    if decimals == 0 {
        return if !x.is_finite() {
            x
        } else {
            f16::from_f32(x.to_f32().round_ties_even())
        };
    }
    if decimals.unsigned_abs() > 4 {
        return f16::NAN;
    }
    if !x.is_finite() {
        return x;
    }
    if decimals > 0 {
        let scale = 10.0_f64.powi(decimals);
        f16::from_f64((x.to_f64() * scale).round_ties_even() / scale)
    } else {
        let scale = 10.0_f64.powi(decimals.abs());
        f16::from_f64((x.to_f64() / scale).round_ties_even() * scale)
    }
}

#[inline]
pub fn max_prop_nan_f64(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() { a } else { b }
}
#[inline]
pub fn min_prop_nan_f64(a: f64, b: f64) -> f64 {
    if a <= b || a.is_nan() { a } else { b }
}
#[inline]
pub fn fmax_f64(a: f64, b: f64) -> f64 {
    if b.is_nan() {
        a
    } else if a.is_nan() {
        b
    } else if a >= b {
        a
    } else {
        b
    }
}
#[inline]
pub fn fmin_f64(a: f64, b: f64) -> f64 {
    if b.is_nan() {
        a
    } else if a.is_nan() {
        b
    } else if a <= b {
        a
    } else {
        b
    }
}

#[inline]
pub fn max_prop_nan_f32(a: f32, b: f32) -> f32 {
    if a >= b || a.is_nan() { a } else { b }
}
#[inline]
pub fn min_prop_nan_f32(a: f32, b: f32) -> f32 {
    if a <= b || a.is_nan() { a } else { b }
}
#[inline]
pub fn fmax_f32(a: f32, b: f32) -> f32 {
    if b.is_nan() {
        a
    } else if a.is_nan() {
        b
    } else if a >= b {
        a
    } else {
        b
    }
}
#[inline]
pub fn fmin_f32(a: f32, b: f32) -> f32 {
    if b.is_nan() {
        a
    } else if a.is_nan() {
        b
    } else if a <= b {
        a
    } else {
        b
    }
}

#[inline]
pub fn max_prop_nan_f16(a: f16, b: f16) -> f16 {
    let af = a.to_f32();
    let bf = b.to_f32();
    if af >= bf || af.is_nan() { a } else { b }
}
#[inline]
pub fn min_prop_nan_f16(a: f16, b: f16) -> f16 {
    let af = a.to_f32();
    let bf = b.to_f32();
    if af <= bf || af.is_nan() { a } else { b }
}
#[inline]
pub fn fmax_f16(a: f16, b: f16) -> f16 {
    if b.is_nan() {
        a
    } else if a.is_nan() {
        b
    } else if a.to_f32() >= b.to_f32() {
        a
    } else {
        b
    }
}
#[inline]
pub fn fmin_f16(a: f16, b: f16) -> f16 {
    if b.is_nan() {
        a
    } else if a.is_nan() {
        b
    } else if a.to_f32() <= b.to_f32() {
        a
    } else {
        b
    }
}

// ============================================================================
// Complex Scalar Kernels (exact NumPy C99 / 2.x semantics)
// ============================================================================

#[inline]
pub fn c64_div(a: Complex32, b: Complex32) -> Complex32 {
    let br = b.re.abs();
    let bi = b.im.abs();
    if br >= bi {
        if br == 0.0 {
            Complex32::new(a.re / br, a.im / bi)
        } else {
            let rat = b.im / b.re;
            let scl = 1.0_f32 / (b.re + b.im * rat);
            Complex32::new((a.re + a.im * rat) * scl, (a.im - a.re * rat) * scl)
        }
    } else {
        let rat = b.re / b.im;
        let scl = 1.0_f32 / (b.im + b.re * rat);
        Complex32::new((a.re * rat + a.im) * scl, (a.im * rat - a.re) * scl)
    }
}

#[inline]
pub fn c128_div(a: Complex64, b: Complex64) -> Complex64 {
    let br = b.re.abs();
    let bi = b.im.abs();
    if br >= bi {
        if br == 0.0 {
            Complex64::new(a.re / br, a.im / bi)
        } else {
            let rat = b.im / b.re;
            let scl = 1.0_f64 / (b.re + b.im * rat);
            Complex64::new((a.re + a.im * rat) * scl, (a.im - a.re * rat) * scl)
        }
    } else {
        let rat = b.re / b.im;
        let scl = 1.0_f64 / (b.im + b.re * rat);
        Complex64::new((a.re * rat + a.im) * scl, (a.im * rat - a.re) * scl)
    }
}

#[inline]
pub fn c64_recip(a: Complex32) -> Complex32 {
    let ar = a.re.abs();
    let ai = a.im.abs();
    if ai <= ar {
        if ar == 0.0 {
            Complex32::new(1.0 / ar, 0.0 / ai)
        } else {
            let r = a.im / a.re;
            let d = a.re + a.im * r;
            Complex32::new(1.0 / d, -r / d)
        }
    } else {
        let r = a.re / a.im;
        let d = a.re * r + a.im;
        Complex32::new(r / d, -1.0 / d)
    }
}

#[inline]
pub fn c128_recip(a: Complex64) -> Complex64 {
    let ar = a.re.abs();
    let ai = a.im.abs();
    if ai <= ar {
        if ar == 0.0 {
            Complex64::new(1.0 / ar, 0.0 / ai)
        } else {
            let r = a.im / a.re;
            let d = a.re + a.im * r;
            Complex64::new(1.0 / d, -r / d)
        }
    } else {
        let r = a.re / a.im;
        let d = a.re * r + a.im;
        Complex64::new(r / d, -1.0 / d)
    }
}

#[inline]
pub fn c64_sign(z: Complex32) -> Complex32 {
    let abs = z.re.hypot(z.im);
    if abs.is_nan() {
        Complex32::new(f32::NAN, f32::NAN)
    } else if abs.is_infinite() {
        if z.re.is_infinite() {
            if z.im.is_infinite() {
                Complex32::new(f32::NAN, f32::NAN)
            } else {
                Complex32::new(if z.re > 0.0 { 1.0 } else { -1.0 }, 0.0)
            }
        } else {
            Complex32::new(0.0, if z.im > 0.0 { 1.0 } else { -1.0 })
        }
    } else if abs == 0.0 {
        Complex32::new(0.0, 0.0)
    } else {
        Complex32::new(z.re / abs, z.im / abs)
    }
}

#[inline]
pub fn c128_sign(z: Complex64) -> Complex64 {
    let abs = z.re.hypot(z.im);
    if abs.is_nan() {
        Complex64::new(f64::NAN, f64::NAN)
    } else if abs.is_infinite() {
        if z.re.is_infinite() {
            if z.im.is_infinite() {
                Complex64::new(f64::NAN, f64::NAN)
            } else {
                Complex64::new(if z.re > 0.0 { 1.0 } else { -1.0 }, 0.0)
            }
        } else {
            Complex64::new(0.0, if z.im > 0.0 { 1.0 } else { -1.0 })
        }
    } else if abs == 0.0 {
        Complex64::new(0.0, 0.0)
    } else {
        Complex64::new(z.re / abs, z.im / abs)
    }
}

#[inline]
pub fn c64_pow(a: Complex32, b: Complex32) -> Complex32 {
    if b.re == 0.0 && b.im == 0.0 {
        return Complex32::new(1.0, 0.0);
    }
    if a.re == 0.0 && a.im == 0.0 && b.re > 0.0 && b.im == 0.0 {
        return Complex32::new(0.0, 0.0);
    }
    a.powc(b)
}

#[inline]
pub fn c128_pow(a: Complex64, b: Complex64) -> Complex64 {
    if b.re == 0.0 && b.im == 0.0 {
        return Complex64::new(1.0, 0.0);
    }
    if a.re == 0.0 && a.im == 0.0 && b.re > 0.0 && b.im == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    a.powc(b)
}

#[inline]
pub fn c64_exp2(z: Complex32) -> Complex32 {
    (z * (LN_2 as f32)).exp()
}
#[inline]
pub fn c128_exp2(z: Complex64) -> Complex64 {
    (z * LN_2).exp()
}

#[inline]
pub fn c64_expm1(z: Complex32) -> Complex32 {
    let a = (z.im * 0.5).sin();
    Complex32::new(
        z.re.exp_m1() * z.im.cos() - 2.0 * a * a,
        z.re.exp() * z.im.sin(),
    )
}
#[inline]
pub fn c128_expm1(z: Complex64) -> Complex64 {
    let a = (z.im * 0.5).sin();
    Complex64::new(
        z.re.exp_m1() * z.im.cos() - 2.0 * a * a,
        z.re.exp() * z.im.sin(),
    )
}

#[inline]
pub fn c64_log2(z: Complex32) -> Complex32 {
    z.ln() * (LOG2_E as f32)
}
#[inline]
pub fn c128_log2(z: Complex64) -> Complex64 {
    z.ln() * LOG2_E
}

#[inline]
pub fn c64_log10(z: Complex32) -> Complex32 {
    z.ln() * (LOG10_E as f32)
}
#[inline]
pub fn c128_log10(z: Complex64) -> Complex64 {
    z.ln() * LOG10_E
}

#[inline]
pub fn c64_log1p(z: Complex32) -> Complex32 {
    let re = if z.re.abs() < 1e-3 && z.im.abs() < 1e-3 {
        0.5 * (z.re * (2.0 + z.re) + z.im * z.im).ln_1p()
    } else {
        (z.re + 1.0).hypot(z.im).ln()
    };
    let im = z.im.atan2(z.re + 1.0);
    Complex32::new(re, im)
}

#[inline]
pub fn c128_log1p(z: Complex64) -> Complex64 {
    let re = if z.re.abs() < 1e-4 && z.im.abs() < 1e-4 {
        0.5 * (z.re * (2.0 + z.re) + z.im * z.im).ln_1p()
    } else {
        (z.re + 1.0).hypot(z.im).ln()
    };
    let im = z.im.atan2(z.re + 1.0);
    Complex64::new(re, im)
}

#[inline]
pub fn c64_sinc(z: Complex32) -> Complex32 {
    let pz = z * (PI as f32);
    if pz.re == 0.0 && pz.im == 0.0 {
        Complex32::new(1.0, 0.0)
    } else {
        c64_div(pz.sin(), pz)
    }
}

#[inline]
pub fn c128_sinc(z: Complex64) -> Complex64 {
    let pz = z * PI;
    if pz.re == 0.0 && pz.im == 0.0 {
        Complex64::new(1.0, 0.0)
    } else {
        c128_div(pz.sin(), pz)
    }
}

#[inline]
pub fn c64_rint(z: Complex32) -> Complex32 {
    Complex32::new(z.re.round_ties_even(), z.im.round_ties_even())
}
#[inline]
pub fn c128_rint(z: Complex64) -> Complex64 {
    Complex64::new(z.re.round_ties_even(), z.im.round_ties_even())
}

// Exact NumPy C99 lexicographical complex comparisons from loops.c.src
#[inline]
pub fn c_lt<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> bool {
    (a.re < b.re && !a.im.is_nan() && !b.im.is_nan()) || (a.re == b.re && a.im < b.im)
}
#[inline]
pub fn c_le<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> bool {
    (a.re < b.re && !a.im.is_nan() && !b.im.is_nan()) || (a.re == b.re && a.im <= b.im)
}
#[inline]
pub fn c_gt<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> bool {
    (a.re > b.re && !a.im.is_nan() && !b.im.is_nan()) || (a.re == b.re && a.im > b.im)
}
#[inline]
pub fn c_ge<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> bool {
    (a.re > b.re && !a.im.is_nan() && !b.im.is_nan()) || (a.re == b.re && a.im >= b.im)
}
#[inline]
pub fn c_eq<T: PartialEq>(a: Complex<T>, b: Complex<T>) -> bool {
    a.re == b.re && a.im == b.im
}
#[inline]
pub fn c_ne<T: PartialEq>(a: Complex<T>, b: Complex<T>) -> bool {
    a.re != b.re || a.im != b.im
}

#[inline]
pub fn c_max<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if a.re.is_nan() || a.im.is_nan() || c_ge(a, b) { a } else { b }
}
#[inline]
pub fn c_min<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if a.re.is_nan() || a.im.is_nan() || c_le(a, b) { a } else { b }
}
#[inline]
pub fn c_fmax<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if b.re.is_nan() || b.im.is_nan() || c_ge(a, b) { a } else { b }
}
#[inline]
pub fn c_fmin<T: num_traits::Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if b.re.is_nan() || b.im.is_nan() || c_le(a, b) { a } else { b }
}

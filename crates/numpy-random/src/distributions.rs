use crate::bit_generator::BitGenerator;
use crate::distributions_tables::LOGFACT_TABLE;
use crate::ziggurat_constants::{
    FE_DOUBLE, FE_FLOAT, FI_DOUBLE, FI_FLOAT, KE_DOUBLE, KE_FLOAT, KI_DOUBLE, KI_FLOAT, WE_DOUBLE,
    WE_FLOAT, WI_DOUBLE, WI_FLOAT, ZIGGURAT_EXP_R, ZIGGURAT_EXP_R_F32, ZIGGURAT_NOR_INV_R,
    ZIGGURAT_NOR_INV_R_F32, ZIGGURAT_NOR_R, ZIGGURAT_NOR_R_F32,
};
use numpy_types::{NumPyError, Result};
use std::f64::consts::PI;

const BETA_TINY_THRESHOLD: f64 = 3e-103;
const SIGNS_F64: [f64; 2] = [1.0, -1.0];
const SIGNS_F32: [f32; 2] = [1.0, -1.0];
const HYPERGEOM_D1: f64 = 1.715_527_769_921_413_5;
const HYPERGEOM_D2: f64 = 0.898_916_162_058_898_8;
const LEGACY_SIGNS: [f64; 2] = [1.0, -1.0];

pub(crate) const POISSON_LAM_MAX: f64 = 9.223_372_006_484_77e18;
pub(crate) const HYPERGEOM_MAX: i64 = 1_000_000_000;

/// Cached parameters for repeated draws from a Binomial(n, p) distribution.
#[derive(Debug, Clone, Default)]
pub struct BinomialState {
    pub has_binomial: bool,
    pub psave: f64,
    pub nsave: i64,
    pub r: f64,
    pub q: f64,
    pub fm: f64,
    pub m: i64,
    pub p1: f64,
    pub xm: f64,
    pub xl: f64,
    pub xr: f64,
    pub c: f64,
    pub laml: f64,
    pub lamr: f64,
    pub p2: f64,
    pub p3: f64,
    pub p4: f64,
}

impl BinomialState {
    pub fn reset(&mut self) {
        self.has_binomial = false;
    }
}

/// Kahan compensated summation matching NumPy's `kahan_sum` probability check.
#[inline]
pub(crate) fn kahan_sum(vals: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &x in vals {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Exact NumPy `random_loggam(x)` approximation to `ln(Gamma(x))`.
#[inline]
pub(crate) fn random_loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333_333_333_333_333e-02,
        -2.777_777_777_777_778e-03,
        7.936_507_936_507_937e-04,
        -5.952_380_952_380_952e-04,
        8.417_508_417_508_418e-04,
        -1.917_526_917_526_918e-03,
        6.410_256_410_256_41e-03,
        -2.955_065_359_477_124e-02,
        1.796_443_723_688_307e-01,
        -1.392_432_216_905_9e+00,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let n: i64 = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + (n as f64);
    let inv = 1.0 / x0;
    let x2 = inv * inv;
    let lg2pi = 1.837_877_066_409_345_3e+00;
    let mut gl0 = A[9];
    for &ak in A[..9].iter().rev() {
        gl0 *= x2;
        gl0 += ak;
    }
    let mut gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 1..=n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

/// Exact NumPy `logfactorial(k)` table + Stirling series.
#[inline]
pub(crate) fn logfactorial(k: i64) -> f64 {
    const HALFLN2PI: f64 = 0.918_938_533_204_672_8;
    if k < 0 {
        return 0.0;
    }
    if (k as usize) < LOGFACT_TABLE.len() {
        return LOGFACT_TABLE[k as usize];
    }
    let kf = k as f64;
    (kf + 0.5) * kf.ln() - kf + (HALFLN2PI + (1.0 / kf) * (1.0 / 12.0 - 1.0 / (360.0 * kf * kf)))
}

// ---------------------------------------------------------------------------
// Bounded integers (Lemire + Masked + buffered 8/16/1-bit)
// ---------------------------------------------------------------------------

/// Compute smallest power-of-two bitmask `>= rng`.
#[inline]
pub(crate) fn gen_mask_u64(mut v: u64) -> u64 {
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v
}

#[inline]
pub(crate) fn random_interval<B: BitGenerator>(rng: &mut B, max: u64) -> u64 {
    if max == 0 {
        return 0;
    }
    let mask = gen_mask_u64(max);
    if max <= u32::MAX as u64 {
        let mask32 = mask as u32;
        let max32 = max as u32;
        loop {
            let val = rng.next_u32() & mask32;
            if val <= max32 {
                return val as u64;
            }
        }
    } else {
        loop {
            let val = rng.next_u64() & mask;
            if val <= max {
                return val;
            }
        }
    }
}

#[inline]
pub(crate) fn bounded_lemire_u64<B: BitGenerator>(rng: &mut B, rng_max: u64) -> u64 {
    if rng_max == u64::MAX {
        return rng.next_u64();
    }
    let rng_excl = rng_max + 1;
    let mut m = (rng.next_u64() as u128) * (rng_excl as u128);
    let mut leftover = m as u64;
    if leftover < rng_excl {
        let threshold = (u64::MAX - rng_max) % rng_excl;
        while leftover < threshold {
            m = (rng.next_u64() as u128) * (rng_excl as u128);
            leftover = m as u64;
        }
    }
    (m >> 64) as u64
}

#[inline]
pub(crate) fn bounded_lemire_u32<B: BitGenerator>(rng: &mut B, rng_max: u32) -> u32 {
    if rng_max == u32::MAX {
        return rng.next_u32();
    }
    let rng_excl = (rng_max as u64) + 1;
    let mut m = (rng.next_u32() as u64) * rng_excl;
    let mut leftover = m as u32;
    if (leftover as u64) < rng_excl {
        let threshold = (u32::MAX - rng_max) % (rng_excl as u32);
        while leftover < threshold {
            m = (rng.next_u32() as u64) * rng_excl;
            leftover = m as u32;
        }
    }
    (m >> 32) as u32
}

#[inline]
fn buffered_uint16<B: BitGenerator>(rng: &mut B, bcnt: &mut i32, buf: &mut u32) -> u16 {
    if *bcnt == 0 {
        *buf = rng.next_u32();
        *bcnt = 1;
        (*buf & 0xffff) as u16
    } else {
        *bcnt -= 1;
        *buf >>= 16;
        (*buf & 0xffff) as u16
    }
}

#[inline]
pub(crate) fn buffered_bounded_lemire_u16<B: BitGenerator>(
    rng: &mut B,
    rng_max: u16,
    bcnt: &mut i32,
    buf: &mut u32,
) -> u16 {
    if rng_max == u16::MAX {
        return buffered_uint16(rng, bcnt, buf);
    }
    let rng_excl = (rng_max as u32) + 1;
    let mut m = (buffered_uint16(rng, bcnt, buf) as u32) * rng_excl;
    let mut leftover = m as u16;
    if (leftover as u32) < rng_excl {
        let threshold = (u16::MAX - rng_max) % (rng_excl as u16);
        while leftover < threshold {
            m = (buffered_uint16(rng, bcnt, buf) as u32) * rng_excl;
            leftover = m as u16;
        }
    }
    (m >> 16) as u16
}

#[inline]
fn buffered_uint8<B: BitGenerator>(rng: &mut B, bcnt: &mut i32, buf: &mut u32) -> u8 {
    if *bcnt == 0 {
        *buf = rng.next_u32();
        *bcnt = 3;
        (*buf & 0xff) as u8
    } else {
        *bcnt -= 1;
        *buf >>= 8;
        (*buf & 0xff) as u8
    }
}

#[inline]
pub(crate) fn buffered_bounded_lemire_u8<B: BitGenerator>(
    rng: &mut B,
    rng_max: u8,
    bcnt: &mut i32,
    buf: &mut u32,
) -> u8 {
    if rng_max == u8::MAX {
        return buffered_uint8(rng, bcnt, buf);
    }
    let rng_excl = (rng_max as u16) + 1;
    let mut m = (buffered_uint8(rng, bcnt, buf) as u16) * rng_excl;
    let mut leftover = m as u8;
    if (leftover as u16) < rng_excl {
        let threshold = (u8::MAX - rng_max) % (rng_excl as u8);
        while leftover < threshold {
            m = (buffered_uint8(rng, bcnt, buf) as u16) * rng_excl;
            leftover = m as u8;
        }
    }
    (m >> 8) as u8
}

#[inline]
pub(crate) fn buffered_uint1<B: BitGenerator>(rng: &mut B, bcnt: &mut i32, buf: &mut u32) -> bool {
    if *bcnt == 0 {
        *buf = rng.next_u32();
        *bcnt = 31;
        (*buf & 1) != 0
    } else {
        *bcnt -= 1;
        *buf >>= 1;
        (*buf & 1) != 0
    }
}

#[inline]
pub(crate) fn random_bounded_uint64<B: BitGenerator>(rng: &mut B, off: u64, rng_max: u64) -> u64 {
    if rng_max == 0 {
        off
    } else if rng_max <= u32::MAX as u64 {
        off.wrapping_add(bounded_lemire_u32(rng, rng_max as u32) as u64)
    } else {
        off.wrapping_add(bounded_lemire_u64(rng, rng_max))
    }
}

#[inline]
pub(crate) fn bounded_masked_u64<B: BitGenerator>(rng: &mut B, rng_max: u64, mask: u64) -> u64 {
    if rng_max == 0 {
        return 0;
    }
    loop {
        let val = rng.next_u64() & mask;
        if val <= rng_max {
            return val;
        }
    }
}

#[inline]
pub(crate) fn bounded_masked_u32<B: BitGenerator>(rng: &mut B, rng_max: u32, mask: u32) -> u32 {
    if rng_max == 0 {
        return 0;
    }
    loop {
        let val = rng.next_u32() & mask;
        if val <= rng_max {
            return val;
        }
    }
}

// ---------------------------------------------------------------------------
// Continuous distributions (f64 + Ziggurat f32)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn random_standard_exponential<B: BitGenerator>(rng: &mut B) -> f64 {
    let mut ri = rng.next_u64();
    ri >>= 3;
    let idx = (ri & 0xff) as usize;
    ri >>= 8;
    let x = (ri as f64) * WE_DOUBLE[idx];
    if ri < KE_DOUBLE[idx] {
        return x;
    }
    standard_exponential_unlikely(rng, idx, x)
}

#[cold]
#[inline(never)]
fn standard_exponential_unlikely<B: BitGenerator>(rng: &mut B, mut idx: usize, mut x: f64) -> f64 {
    loop {
        if idx == 0 {
            return ZIGGURAT_EXP_R - (-rng.next_double()).ln_1p();
        }
        if ((FE_DOUBLE[idx - 1] - FE_DOUBLE[idx]) * rng.next_double() + FE_DOUBLE[idx])
            < (-x).exp()
        {
            return x;
        }
        let mut ri = rng.next_u64();
        ri >>= 3;
        idx = (ri & 0xff) as usize;
        ri >>= 8;
        x = (ri as f64) * WE_DOUBLE[idx];
        if ri < KE_DOUBLE[idx] {
            return x;
        }
    }
}

#[inline]
pub(crate) fn random_standard_exponential_f32<B: BitGenerator>(rng: &mut B) -> f32 {
    let mut ri = rng.next_u32();
    ri >>= 1;
    let idx = (ri & 0xff) as usize;
    ri >>= 8;
    let x = (ri as f32) * WE_FLOAT[idx];
    if ri < KE_FLOAT[idx] {
        return x;
    }
    standard_exponential_unlikely_f32(rng, idx, x)
}

#[cold]
#[inline(never)]
fn standard_exponential_unlikely_f32<B: BitGenerator>(
    rng: &mut B,
    mut idx: usize,
    mut x: f32,
) -> f32 {
    loop {
        if idx == 0 {
            return ZIGGURAT_EXP_R_F32 - (-rng.next_float()).ln_1p();
        }
        if ((FE_FLOAT[idx - 1] - FE_FLOAT[idx]) * rng.next_float() + FE_FLOAT[idx]) < (-x).exp() {
            return x;
        }
        let mut ri = rng.next_u32();
        ri >>= 1;
        idx = (ri & 0xff) as usize;
        ri >>= 8;
        x = (ri as f32) * WE_FLOAT[idx];
        if ri < KE_FLOAT[idx] {
            return x;
        }
    }
}

#[inline]
pub(crate) fn random_standard_normal<B: BitGenerator>(rng: &mut B) -> f64 {
    let mut r = rng.next_u64();
    let idx = (r & 0xff) as usize;
    r >>= 8;
    let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
    let x = (rabs as f64) * WI_DOUBLE[idx] * SIGNS_F64[(r & 1) as usize];
    if rabs < KI_DOUBLE[idx] {
        return x;
    }
    standard_normal_unlikely(rng, idx, rabs, x)
}

#[cold]
#[inline(never)]
fn standard_normal_unlikely<B: BitGenerator>(
    rng: &mut B,
    mut idx: usize,
    mut rabs: u64,
    mut x: f64,
) -> f64 {
    loop {
        if idx == 0 {
            loop {
                let xx = -ZIGGURAT_NOR_INV_R * (-rng.next_double()).ln_1p();
                let yy = -(-rng.next_double()).ln_1p();
                if yy + yy > xx * xx {
                    return if ((rabs >> 8) & 1) != 0 {
                        -(ZIGGURAT_NOR_R + xx)
                    } else {
                        ZIGGURAT_NOR_R + xx
                    };
                }
            }
        } else if ((FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * rng.next_double() + FI_DOUBLE[idx])
            < (-0.5 * x * x).exp()
        {
            return x;
        }

        let mut r = rng.next_u64();
        idx = (r & 0xff) as usize;
        r >>= 8;
        rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
        x = (rabs as f64) * WI_DOUBLE[idx] * SIGNS_F64[(r & 1) as usize];
        if rabs < KI_DOUBLE[idx] {
            return x;
        }
    }
}

#[inline]
pub(crate) fn random_standard_normal_f32<B: BitGenerator>(rng: &mut B) -> f32 {
    let r = rng.next_u32();
    let idx = (r & 0xff) as usize;
    let rabs = (r >> 9) & 0x007f_ffff;
    let x = (rabs as f32) * WI_FLOAT[idx] * SIGNS_F32[((r >> 8) & 1) as usize];
    if rabs < KI_FLOAT[idx] {
        return x;
    }
    standard_normal_unlikely_f32(rng, r, idx, x)
}

#[cold]
#[inline(never)]
fn standard_normal_unlikely_f32<B: BitGenerator>(
    rng: &mut B,
    mut r: u32,
    mut idx: usize,
    mut x: f32,
) -> f32 {
    loop {
        if idx == 0 {
            loop {
                let xx = -ZIGGURAT_NOR_INV_R_F32 * (-rng.next_float()).ln_1p();
                let yy = -(-rng.next_float()).ln_1p();
                if yy + yy > xx * xx {
                    return if ((r >> 8) & 1) != 0 {
                        -(ZIGGURAT_NOR_R_F32 + xx)
                    } else {
                        ZIGGURAT_NOR_R_F32 + xx
                    };
                }
            }
        } else if ((FI_FLOAT[idx - 1] - FI_FLOAT[idx]) * rng.next_float() + FI_FLOAT[idx])
            < (-0.5f32 * x * x).exp()
        {
            return x;
        }

        r = rng.next_u32();
        idx = (r & 0xff) as usize;
        let rabs = (r >> 9) & 0x007f_ffff;
        x = (rabs as f32) * WI_FLOAT[idx] * SIGNS_F32[((r >> 8) & 1) as usize];
        if rabs < KI_FLOAT[idx] {
            return x;
        }
    }
}

#[inline]
pub(crate) fn random_normal<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loc + scale * random_standard_normal(rng)
}

#[inline]
pub(crate) fn random_exponential<B: BitGenerator>(rng: &mut B, scale: f64) -> f64 {
    scale * random_standard_exponential(rng)
}

#[inline]
pub(crate) fn random_uniform<B: BitGenerator>(rng: &mut B, lower: f64, range: f64) -> f64 {
    lower + range * rng.next_double()
}

pub(crate) fn random_standard_gamma<B: BitGenerator>(rng: &mut B, shape: f64) -> f64 {
    if shape == 1.0 {
        return random_standard_exponential(rng);
    }
    if shape == 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        loop {
            let u = rng.next_double();
            let v = random_standard_exponential(rng);
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return x;
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                if x <= v + y {
                    return x;
                }
            }
        }
    }
    let b = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * b).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = random_standard_normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_double();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return b * v;
        }
        if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
            return b * v;
        }
    }
}

pub(crate) fn random_standard_gamma_f32<B: BitGenerator>(rng: &mut B, shape: f32) -> f32 {
    if shape == 1.0 {
        return random_standard_exponential_f32(rng);
    }
    if shape == 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        loop {
            let u = rng.next_float();
            let v = random_standard_exponential_f32(rng);
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return x;
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                if x <= v + y {
                    return x;
                }
            }
        }
    }
    let b = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * b).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = random_standard_normal_f32(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_float();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return b * v;
        }
        if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
            return b * v;
        }
    }
}

#[inline]
pub(crate) fn random_gamma<B: BitGenerator>(rng: &mut B, shape: f64, scale: f64) -> f64 {
    scale * random_standard_gamma(rng, shape)
}

pub(crate) fn random_beta<B: BitGenerator>(rng: &mut B, a: f64, b: f64) -> f64 {
    if a <= 1.0 && b <= 1.0 {
        if a < BETA_TINY_THRESHOLD && b < BETA_TINY_THRESHOLD {
            let u = rng.next_double();
            return if (a + b) * u < a { 1.0 } else { 0.0 };
        }
        loop {
            let u = rng.next_double();
            let v = rng.next_double();
            let x = u.powf(1.0 / a);
            let y = v.powf(1.0 / b);
            let xpy = x + y;
            if xpy <= 1.0 && (u + v > 0.0) {
                if x > 0.0 && y > 0.0 {
                    return x / xpy;
                } else {
                    let log_x = u.ln() / a;
                    let log_y = v.ln() / b;
                    let delta = log_x - log_y;
                    if delta > 0.0 {
                        return (-((-delta).exp().ln_1p())).exp();
                    } else {
                        return (delta - (delta.exp().ln_1p())).exp();
                    }
                }
            }
        }
    } else {
        let ga = random_standard_gamma(rng, a);
        let gb = random_standard_gamma(rng, b);
        ga / (ga + gb)
    }
}

#[inline]
pub(crate) fn random_chisquare<B: BitGenerator>(rng: &mut B, df: f64) -> f64 {
    2.0 * random_standard_gamma(rng, 0.5 * df)
}

pub(crate) fn random_noncentral_chisquare<B: BitGenerator>(
    rng: &mut B,
    df: f64,
    nonc: f64,
) -> f64 {
    if nonc == 0.0 {
        return random_chisquare(rng, df);
    }
    if 1.0 < df {
        let chi2 = random_chisquare(rng, df - 1.0);
        let n = random_standard_normal(rng) + nonc.sqrt();
        chi2 + n * n
    } else {
        let i = random_poisson(rng, nonc / 2.0);
        random_chisquare(rng, df + 2.0 * (i as f64))
    }
}

#[inline]
pub(crate) fn random_f<B: BitGenerator>(rng: &mut B, dfnum: f64, dfden: f64) -> f64 {
    (random_chisquare(rng, dfnum) * dfden) / (random_chisquare(rng, dfden) * dfnum)
}

#[inline]
pub(crate) fn random_noncentral_f<B: BitGenerator>(
    rng: &mut B,
    dfnum: f64,
    dfden: f64,
    nonc: f64,
) -> f64 {
    let t = random_noncentral_chisquare(rng, dfnum, nonc) * dfden;
    t / (random_chisquare(rng, dfden) * dfnum)
}

#[inline]
pub(crate) fn random_standard_t<B: BitGenerator>(rng: &mut B, df: f64) -> f64 {
    let num = random_standard_normal(rng);
    let denom = random_standard_gamma(rng, df / 2.0);
    (df / 2.0).sqrt() * num / denom.sqrt()
}

#[inline]
pub(crate) fn random_standard_cauchy<B: BitGenerator>(rng: &mut B) -> f64 {
    let n1 = random_standard_normal(rng);
    let n2 = random_standard_normal(rng);
    n1 / n2
}

#[inline]
pub(crate) fn random_pareto<B: BitGenerator>(rng: &mut B, a: f64) -> f64 {
    (random_standard_exponential(rng) / a).exp_m1()
}

#[inline]
pub(crate) fn random_weibull<B: BitGenerator>(rng: &mut B, a: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else {
        random_standard_exponential(rng).powf(1.0 / a)
    }
}

#[inline]
pub(crate) fn random_power<B: BitGenerator>(rng: &mut B, a: f64) -> f64 {
    (-(-random_standard_exponential(rng)).exp_m1()).powf(1.0 / a)
}

#[inline]
pub(crate) fn random_laplace<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loop {
        let u = rng.next_double();
        if u >= 0.5 {
            return loc - scale * (2.0 - 2.0 * u).ln();
        } else if u > 0.0 {
            return loc + scale * (2.0 * u).ln();
        }
    }
}

#[inline]
pub(crate) fn random_gumbel<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loop {
        let u = 1.0 - rng.next_double();
        if u < 1.0 {
            return loc - scale * (-u.ln()).ln();
        }
    }
}

#[inline]
pub(crate) fn random_logistic<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loop {
        let u = rng.next_double();
        if u > 0.0 {
            return loc + scale * (u / (1.0 - u)).ln();
        }
    }
}

#[inline]
pub(crate) fn random_lognormal<B: BitGenerator>(rng: &mut B, mean: f64, sigma: f64) -> f64 {
    random_normal(rng, mean, sigma).exp()
}

#[inline]
pub(crate) fn random_rayleigh<B: BitGenerator>(rng: &mut B, scale: f64) -> f64 {
    scale * (2.0 * random_standard_exponential(rng)).sqrt()
}

#[inline]
pub(crate) fn random_triangular<B: BitGenerator>(
    rng: &mut B,
    left: f64,
    mode: f64,
    right: f64,
) -> f64 {
    let base = right - left;
    let leftbase = mode - left;
    let ratio = leftbase / base;
    let leftprod = leftbase * base;
    let rightprod = (right - mode) * base;
    let u = rng.next_double();
    if u <= ratio {
        left + (u * leftprod).sqrt()
    } else {
        right - ((1.0 - u) * rightprod).sqrt()
    }
}

pub(crate) fn random_vonmises<B: BitGenerator>(rng: &mut B, mu: f64, kappa: f64) -> f64 {
    if kappa.is_nan() {
        return f64::NAN;
    }
    if kappa < 1e-8 {
        return PI * (2.0 * rng.next_double() - 1.0);
    }
    if kappa > 1e6 {
        let mut res = mu + (1.0 / kappa).sqrt() * random_standard_normal(rng);
        if res > PI {
            res -= 2.0 * PI;
        } else if res < -PI {
            res += 2.0 * PI;
        }
        return res;
    }
    let s = if kappa < 1e-5 {
        1.0 / kappa + kappa
    } else {
        let r = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
        let rho = (r - (2.0 * r).sqrt()) / (2.0 * kappa);
        (1.0 + rho * rho) / (2.0 * rho)
    };

    let w;
    loop {
        let u = rng.next_double();
        let z = (PI * u).cos();
        let w_cand = (1.0 + s * z) / (s + z);
        let y = kappa * (s - w_cand);
        let v = rng.next_double();
        if y * (2.0 - y) - v >= 0.0 || (y / v).ln() + 1.0 - y >= 0.0 {
            w = w_cand;
            break;
        }
    }

    let u = rng.next_double();
    let mut result = w.acos();
    if u < 0.5 {
        result = -result;
    }
    result += mu;
    let neg = result < 0.0;
    let mut m = result.abs();
    let modulus = 2.0 * PI;
    if m > modulus {
        m %= modulus;
    }
    if m > PI {
        m -= modulus;
    }
    if neg {
        -m
    } else {
        m
    }
}

pub(crate) fn random_wald<B: BitGenerator>(rng: &mut B, mean: f64, scale: f64) -> f64 {
    let mut y = random_standard_normal(rng);
    y = mean * y * y;
    let d = 1.0 + (1.0 + 4.0 * scale / y).sqrt();
    let x = mean * (1.0 - 2.0 / d);
    let u = rng.next_double();
    if u <= mean / (mean + x) {
        x
    } else {
        mean * mean / x
    }
}

// ---------------------------------------------------------------------------
// Discrete distributions
// ---------------------------------------------------------------------------

pub(crate) fn random_poisson<B: BitGenerator>(rng: &mut B, lam: f64) -> i64 {
    if lam >= 10.0 {
        random_poisson_ptrs(rng, lam)
    } else if lam == 0.0 {
        0
    } else {
        random_poisson_mult(rng, lam)
    }
}

fn random_poisson_mult<B: BitGenerator>(rng: &mut B, lam: f64) -> i64 {
    let enlam = (-lam).exp();
    let mut x = 0i64;
    let mut prod = 1.0f64;
    loop {
        let u = rng.next_double();
        prod *= u;
        if prod > enlam {
            x += 1;
        } else {
            return x;
        }
    }
}

fn random_poisson_ptrs<B: BitGenerator>(rng: &mut B, lam: f64) -> i64 {
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.next_double() - 0.5;
        let v = rng.next_double();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor() as i64;
        if us >= 0.07 && v <= vr {
            return k;
        }
        if k < 0 || (us < 0.013 && v > us) {
            continue;
        }
        if (v.ln() + invalpha.ln() - (a / (us * us) + b).ln())
            <= (-lam + (k as f64) * loglam - random_loggam((k as f64) + 1.0))
        {
            return k;
        }
    }
}

#[inline]
pub(crate) fn random_negative_binomial<B: BitGenerator>(rng: &mut B, n: f64, p: f64) -> i64 {
    let y = random_gamma(rng, n, (1.0 - p) / p);
    random_poisson(rng, y)
}

pub(crate) fn random_binomial<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    p: f64,
    binomial: &mut BinomialState,
) -> i64 {
    binomial_impl(rng, n, p, binomial, false)
}

pub(crate) fn legacy_binomial<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    p: f64,
    binomial: &mut BinomialState,
) -> i64 {
    binomial_impl(rng, n, p, binomial, true)
}

fn binomial_impl<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    p: f64,
    binomial: &mut BinomialState,
    legacy_stirling: bool,
) -> i64 {
    if !legacy_stirling && (n == 0 || p == 0.0) {
        return 0;
    }
    if p <= 0.5 {
        if p * (n as f64) <= 30.0 {
            random_binomial_inversion(rng, n, p, binomial)
        } else {
            random_binomial_btpe(rng, n, p, binomial, legacy_stirling)
        }
    } else {
        let q = 1.0 - p;
        if q * (n as f64) <= 30.0 {
            n - random_binomial_inversion(rng, n, q, binomial)
        } else {
            n - random_binomial_btpe(rng, n, q, binomial, legacy_stirling)
        }
    }
}

fn random_binomial_inversion<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    p: f64,
    binomial: &mut BinomialState,
) -> i64 {
    let q;
    let qn;
    let bound;
    if !binomial.has_binomial || binomial.nsave != n || binomial.psave != p {
        binomial.nsave = n;
        binomial.psave = p;
        binomial.has_binomial = true;
        q = 1.0 - p;
        binomial.q = q;
        qn = ((n as f64) * (-p).ln_1p()).exp();
        binomial.r = qn;
        let np = (n as f64) * p;
        binomial.c = np;
        bound = (n as f64).min(np + 10.0 * (np * q + 1.0).sqrt()) as i64;
        binomial.m = bound;
    } else {
        q = binomial.q;
        qn = binomial.r;
        bound = binomial.m;
    }
    let mut x = 0i64;
    let mut px = qn;
    let mut u = rng.next_double();
    while u > px {
        x += 1;
        if x > bound {
            x = 0;
            px = qn;
            u = rng.next_double();
        } else {
            u -= px;
            px = (((n - x) as f64 + 1.0) * p * px) / ((x as f64) * q);
        }
    }
    x
}

fn random_binomial_btpe<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    p: f64,
    binomial: &mut BinomialState,
    legacy_stirling: bool,
) -> i64 {
    let r;
    let q;
    let fm;
    let m;
    let p1;
    let xm;
    let xl;
    let xr;
    let c;
    let laml;
    let lamr;
    let p2;
    let p3;
    let p4;
    if !binomial.has_binomial || binomial.nsave != n || binomial.psave != p {
        binomial.nsave = n;
        binomial.psave = p;
        binomial.has_binomial = true;
        r = p.min(1.0 - p);
        binomial.r = r;
        q = 1.0 - r;
        binomial.q = q;
        fm = (n as f64) * r + r;
        binomial.fm = fm;
        m = fm.floor() as i64;
        binomial.m = m;
        p1 = (2.195 * ((n as f64) * r * q).sqrt() - 4.6 * q).floor() + 0.5;
        binomial.p1 = p1;
        xm = (m as f64) + 0.5;
        binomial.xm = xm;
        xl = xm - p1;
        binomial.xl = xl;
        xr = xm + p1;
        binomial.xr = xr;
        c = 0.134 + 20.5 / (15.3 + (m as f64));
        binomial.c = c;
        let al = (fm - xl) / (fm - xl * r);
        laml = al * (1.0 + al / 2.0);
        binomial.laml = laml;
        let ar = (xr - fm) / (xr * q);
        lamr = ar * (1.0 + ar / 2.0);
        binomial.lamr = lamr;
        p2 = p1 * (1.0 + 2.0 * c);
        binomial.p2 = p2;
        p3 = p2 + c / laml;
        binomial.p3 = p3;
        p4 = p3 + c / lamr;
        binomial.p4 = p4;
    } else {
        r = binomial.r;
        q = binomial.q;
        m = binomial.m;
        p1 = binomial.p1;
        xm = binomial.xm;
        xl = binomial.xl;
        xr = binomial.xr;
        c = binomial.c;
        laml = binomial.laml;
        lamr = binomial.lamr;
        p2 = binomial.p2;
        p3 = binomial.p3;
        p4 = binomial.p4;
    }

    let nrq = (n as f64) * r * q;
    loop {
        let u = rng.next_double() * p4;
        let mut v = rng.next_double();
        let y: i64;
        if u <= p1 {
            y = (xm - p1 * v + u).floor() as i64;
            return if p > 0.5 { n - y } else { y };
        } else if u <= p2 {
            let x = xl + (u - p1) / c;
            v = v * c + 1.0 - ((m as f64) - x + 0.5).abs() / p1;
            if v > 1.0 {
                continue;
            }
            y = x.floor() as i64;
        } else if u <= p3 {
            y = (xl + v.ln() / laml).floor() as i64;
            if y < 0 || v == 0.0 {
                continue;
            }
            v = v * (u - p2) * laml;
        } else {
            y = (xr - v.ln() / lamr).floor() as i64;
            if y > n || v == 0.0 {
                continue;
            }
            v = v * (u - p3) * lamr;
        }

        let k = (y - m).abs();
        if (k <= 20) || ((k as f64) >= nrq / 2.0 - 1.0) {
            let s = r / q;
            let a = s * ((n as f64) + 1.0);
            let mut f = 1.0f64;
            if m < y {
                for i in (m + 1)..=y {
                    f *= a / (i as f64) - s;
                }
            } else if m > y {
                for i in (y + 1)..=m {
                    f /= a / (i as f64) - s;
                }
            }
            if v > f {
                continue;
            }
            return if p > 0.5 { n - y } else { y };
        } else {
            let kf = k as f64;
            let rho =
                (kf / nrq) * ((kf * (kf / 3.0 + 0.625) + 0.166_666_666_666_666_66) / nrq + 0.5);
            let t = -kf * kf / (2.0 * nrq);
            let big_a = v.ln();
            if big_a < t - rho {
                return if p > 0.5 { n - y } else { y };
            }
            if big_a > t + rho {
                continue;
            }
            let x1 = (y as f64) + 1.0;
            let f1 = (m as f64) + 1.0;
            let z = (n as f64) + 1.0 - (m as f64);
            let w = (n as f64) - (y as f64) + 1.0;
            let x2 = x1 * x1;
            let f2 = f1 * f1;
            let z2 = z * z;
            let w2 = w * w;
            let stirling_coeff = if legacy_stirling { 13680.0 } else { 13860.0 };
            let stirling = |d1: f64, d2: f64| {
                (stirling_coeff - (462.0 - (132.0 - (99.0 - 140.0 / d2) / d2) / d2) / d2)
                    / d1
                    / 166320.0
            };
            let sgn = if legacy_stirling { 1.0 } else { -1.0 };
            let rhs = xm * (f1 / x1).ln()
                + ((n - m) as f64 + 0.5) * (z / w).ln()
                + ((y - m) as f64) * (w * r / (x1 * q)).ln()
                + stirling(f1, f2)
                + stirling(z, z2)
                + sgn * stirling(x1, x2)
                + sgn * stirling(w, w2);
            if big_a > rhs {
                continue;
            }
            return if p > 0.5 { n - y } else { y };
        }
    }
}

pub(crate) fn random_geometric<B: BitGenerator>(rng: &mut B, p: f64) -> i64 {
    if p >= 0.333_333_333_333_333_3 {
        let mut x = 1i64;
        let mut sum = p;
        let mut prod = p;
        let q = 1.0 - p;
        let u = rng.next_double();
        while u > sum {
            prod *= q;
            sum += prod;
            x += 1;
        }
        x
    } else {
        let z = (-random_standard_exponential(rng) / (-p).ln_1p()).ceil();
        if z >= 9.223_372_036_854_776e+18 {
            i64::MAX
        } else {
            z as i64
        }
    }
}

pub(crate) fn random_hypergeometric<B: BitGenerator>(
    rng: &mut B,
    good: i64,
    bad: i64,
    sample: i64,
) -> i64 {
    if sample >= 10 && sample <= good + bad - 10 {
        hypergeometric_hrua(rng, good, bad, sample)
    } else {
        hypergeometric_sample(rng, good, bad, sample)
    }
}

fn hypergeometric_sample<B: BitGenerator>(
    rng: &mut B,
    good: i64,
    bad: i64,
    sample: i64,
) -> i64 {
    let total = good + bad;
    let mut computed_sample = if sample > total / 2 {
        total - sample
    } else {
        sample
    };
    let mut remaining_total = total;
    let mut remaining_good = good;
    while computed_sample > 0 && remaining_good > 0 && remaining_total > remaining_good {
        remaining_total -= 1;
        if (random_interval(rng, remaining_total as u64) as i64) < remaining_good {
            remaining_good -= 1;
        }
        computed_sample -= 1;
    }
    if remaining_total == remaining_good {
        remaining_good -= computed_sample;
    }
    if sample > total / 2 {
        remaining_good
    } else {
        good - remaining_good
    }
}

fn hypergeometric_hrua<B: BitGenerator>(
    rng: &mut B,
    good: i64,
    bad: i64,
    sample: i64,
) -> i64 {
    let popsize = good + bad;
    let computed_sample = sample.min(popsize - sample);
    let mingoodbad = good.min(bad);
    let maxgoodbad = good.max(bad);
    let p = (mingoodbad as f64) / (popsize as f64);
    let q = (maxgoodbad as f64) / (popsize as f64);
    let mu = (computed_sample as f64) * p;
    let a = mu + 0.5;
    let var = ((popsize - computed_sample) as f64) * (computed_sample as f64) * p * q
        / ((popsize - 1) as f64);
    let c = (var + 0.5).sqrt();
    let h = HYPERGEOM_D1 * c + HYPERGEOM_D2;
    let m = (((computed_sample + 1) as f64) * ((mingoodbad + 1) as f64) / ((popsize + 2) as f64))
        .floor() as i64;
    let g = logfactorial(m)
        + logfactorial(mingoodbad - m)
        + logfactorial(computed_sample - m)
        + logfactorial(maxgoodbad - computed_sample + m);
    let b = ((computed_sample.min(mingoodbad) + 1) as f64).min((a + 16.0 * c).floor());

    let mut k_val: i64;
    loop {
        let u = rng.next_double();
        let v = rng.next_double();
        let x = a + h * (v - 0.5) / u;
        if x < 0.0 || x >= b {
            continue;
        }
        k_val = x.floor() as i64;
        let gp = logfactorial(k_val)
            + logfactorial(mingoodbad - k_val)
            + logfactorial(computed_sample - k_val)
            + logfactorial(maxgoodbad - computed_sample + k_val);
        let t = g - gp;
        if (u * (4.0 - u) - 3.0) <= t {
            break;
        }
        if u * (u - t) >= 1.0 {
            continue;
        }
        if 2.0 * u.ln() <= t {
            break;
        }
    }

    if good > bad {
        k_val = computed_sample - k_val;
    }
    if computed_sample < sample {
        k_val = good - k_val;
    }
    k_val
}

pub(crate) fn random_logseries<B: BitGenerator>(rng: &mut B, p: f64) -> i64 {
    let r = (-p).ln_1p();
    loop {
        let v = rng.next_double();
        if v >= p {
            return 1;
        }
        let u = rng.next_double();
        let q = -(r * u).exp_m1();
        if v <= q * q {
            let result = (1.0 + v.ln() / q.ln()).floor() as i64;
            if result >= 1 && v != 0.0 {
                return result;
            }
            continue;
        }
        if v >= q {
            return 1;
        }
        return 2;
    }
}

pub(crate) fn random_zipf<B: BitGenerator>(rng: &mut B, a: f64) -> i64 {
    if a >= 1025.0 {
        return 1;
    }
    let am1 = a - 1.0;
    let b = 2.0f64.powf(am1);
    let umin = (i64::MAX as f64).powf(-am1);
    loop {
        let u01 = rng.next_double();
        let u = u01 * umin + (1.0 - u01);
        let v = rng.next_double();
        let x = u.powf(-1.0 / am1).floor();
        if x > (i64::MAX as f64) || x < 1.0 {
            continue;
        }
        let t = (1.0 + 1.0 / x).powf(am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b {
            return x as i64;
        }
    }
}

pub(crate) fn random_multinomial<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    mnix: &mut [i64],
    pix: &[f64],
    binomial: &mut BinomialState,
) {
    multinomial_impl(rng, n, mnix, pix, binomial, false);
}

pub(crate) fn legacy_multinomial<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    mnix: &mut [i64],
    pix: &[f64],
    binomial: &mut BinomialState,
) {
    multinomial_impl(rng, n, mnix, pix, binomial, true);
}

fn multinomial_impl<B: BitGenerator>(
    rng: &mut B,
    n: i64,
    mnix: &mut [i64],
    pix: &[f64],
    binomial: &mut BinomialState,
    legacy: bool,
) {
    let d = pix.len();
    let mut remaining_p = 1.0f64;
    let mut dn = n;
    for slot in mnix.iter_mut() {
        *slot = 0;
    }
    if d <= 1 {
        if d == 1 {
            mnix[0] = dn;
        }
        return;
    }
    for j in 0..(d - 1) {
        let p = pix[j] / remaining_p;
        if p >= 1.0 {
            mnix[j] = dn;
            dn = 0;
            break;
        }
        let drawn = binomial_impl(rng, dn, p, binomial, legacy);
        mnix[j] = drawn;
        dn -= drawn;
        if dn <= 0 {
            break;
        }
        remaining_p -= pix[j];
    }
    if dn > 0 {
        mnix[d - 1] = dn;
    }
}

// ---------------------------------------------------------------------------
// Legacy RandomState distribution kernels (matching legacy-distributions.c)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn legacy_double<B: BitGenerator>(rng: &mut B) -> f64 {
    let a = (rng.next_u32() >> 5) as f64;
    let b = (rng.next_u32() >> 6) as f64;
    (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
}

pub(crate) fn legacy_gauss<B: BitGenerator>(
    rng: &mut B,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> f64 {
    if *has_gauss {
        let tmp = *gauss;
        *has_gauss = false;
        *gauss = 0.0;
        return tmp;
    }
    loop {
        let x1 = 2.0 * legacy_double(rng) - 1.0;
        let x2 = 2.0 * legacy_double(rng) - 1.0;
        let r2 = x1 * x1 + x2 * x2;
        if r2 < 1.0 && r2 != 0.0 {
            let f = (-2.0 * r2.ln() / r2).sqrt();
            *gauss = f * x1;
            *has_gauss = true;
            return f * x2;
        }
    }
}

#[inline]
pub(crate) fn legacy_standard_exponential<B: BitGenerator>(rng: &mut B) -> f64 {
    -(1.0 - legacy_double(rng)).ln()
}

pub(crate) fn legacy_standard_gamma<B: BitGenerator>(
    rng: &mut B,
    shape: f64,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> f64 {
    if shape == 1.0 {
        return legacy_standard_exponential(rng);
    }
    if shape == 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        loop {
            let u = legacy_double(rng);
            let v = legacy_standard_exponential(rng);
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return x;
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                if x <= v + y {
                    return x;
                }
            }
        }
    }
    let b = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * b).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = legacy_gauss(rng, has_gauss, gauss);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = legacy_double(rng);
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return b * v;
        }
        if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
            return b * v;
        }
    }
}

#[inline]
pub(crate) fn legacy_chisquare<B: BitGenerator>(
    rng: &mut B,
    df: f64,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> f64 {
    2.0 * legacy_standard_gamma(rng, 0.5 * df, has_gauss, gauss)
}

#[inline]
pub(crate) fn legacy_noncentral_chisquare<B: BitGenerator>(
    rng: &mut B,
    df: f64,
    nonc: f64,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> f64 {
    if nonc == 0.0 {
        return legacy_chisquare(rng, df, has_gauss, gauss);
    }
    if 1.0 < df {
        let chi2 = legacy_chisquare(rng, df - 1.0, has_gauss, gauss);
        let n = legacy_gauss(rng, has_gauss, gauss) + nonc.sqrt();
        chi2 + n * n
    } else {
        let i = random_poisson(rng, nonc / 2.0);
        legacy_chisquare(rng, df + 2.0 * (i as f64), has_gauss, gauss)
    }
}

#[inline]
pub(crate) fn legacy_standard_t<B: BitGenerator>(
    rng: &mut B,
    df: f64,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> f64 {
    let num = legacy_gauss(rng, has_gauss, gauss);
    let denom = legacy_standard_gamma(rng, df / 2.0, has_gauss, gauss);
    (df / 2.0).sqrt() * num / denom.sqrt()
}

pub(crate) fn legacy_beta<B: BitGenerator>(
    rng: &mut B,
    a: f64,
    b: f64,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> f64 {
    if a <= 1.0 && b <= 1.0 {
        loop {
            let u = legacy_double(rng);
            let v = legacy_double(rng);
            let x = u.powf(1.0 / a);
            let y = v.powf(1.0 / b);
            if x + y <= 1.0 {
                if x + y > 0.0 {
                    return x / (x + y);
                } else {
                    let log_x = u.ln() / a;
                    let log_y = v.ln() / b;
                    let log_m = log_x.max(log_y);
                    let lx = (log_x - log_m).exp();
                    let ly = (log_y - log_m).exp();
                    return lx / (lx + ly);
                }
            }
        }
    } else {
        let ga = legacy_standard_gamma(rng, a, has_gauss, gauss);
        let gb = legacy_standard_gamma(rng, b, has_gauss, gauss);
        ga / (ga + gb)
    }
}

#[inline]
pub(crate) fn legacy_pareto<B: BitGenerator>(rng: &mut B, a: f64) -> f64 {
    (legacy_standard_exponential(rng) / a).exp_m1()
}

#[inline]
pub(crate) fn legacy_weibull<B: BitGenerator>(rng: &mut B, a: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else {
        legacy_standard_exponential(rng).powf(1.0 / a)
    }
}

#[inline]
pub(crate) fn legacy_power<B: BitGenerator>(rng: &mut B, a: f64) -> f64 {
    (-(-legacy_standard_exponential(rng)).exp_m1()).powf(1.0 / a)
}

#[inline]
pub(crate) fn legacy_rayleigh<B: BitGenerator>(rng: &mut B, scale: f64) -> f64 {
    scale * (2.0 * legacy_standard_exponential(rng)).sqrt()
}

#[inline]
pub(crate) fn legacy_laplace<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loop {
        let u = legacy_double(rng);
        if u >= 0.5 {
            return loc - scale * (2.0 - 2.0 * u).ln();
        } else if u > 0.0 {
            return loc + scale * (2.0 * u).ln();
        }
    }
}

#[inline]
pub(crate) fn legacy_gumbel<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loop {
        let u = 1.0 - legacy_double(rng);
        if u < 1.0 {
            return loc - scale * (-u.ln()).ln();
        }
    }
}

#[inline]
pub(crate) fn legacy_logistic<B: BitGenerator>(rng: &mut B, loc: f64, scale: f64) -> f64 {
    loop {
        let u = legacy_double(rng);
        if u > 0.0 {
            return loc + scale * (u / (1.0 - u)).ln();
        }
    }
}

#[inline]
pub(crate) fn legacy_vonmises<B: BitGenerator>(rng: &mut B, mu: f64, kappa: f64) -> f64 {
    if kappa.is_nan() {
        return f64::NAN;
    }
    if kappa < 1e-8 {
        return PI * (2.0 * legacy_double(rng) - 1.0);
    }
    if kappa.is_infinite() {
        let mut res = mu;
        if res > PI {
            res -= 2.0 * PI;
        } else if res < -PI {
            res += 2.0 * PI;
        }
        return res;
    }
    let s = if kappa < 1e-5 {
        1.0 / kappa + kappa
    } else {
        let r = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
        let rho = (r - (2.0 * r).sqrt()) / (2.0 * kappa);
        (1.0 + rho * rho) / (2.0 * rho)
    };
    let w;
    loop {
        let u = legacy_double(rng);
        let z = (PI * u).cos();
        let w_cand = (1.0 + s * z) / (s + z);
        let y = kappa * (s - w_cand);
        let v = legacy_double(rng);
        if y * (2.0 - y) - v >= 0.0 || (y / v).ln() + 1.0 - y >= 0.0 {
            w = w_cand;
            break;
        }
    }
    let u = legacy_double(rng);
    let mut result = w.acos();
    if u < 0.5 {
        result = -result;
    }
    result += mu;
    let neg = result < 0.0;
    let mut m = result.abs();
    let modulus = 2.0 * PI;
    if m > modulus {
        m %= modulus;
    }
    if m > PI {
        m -= modulus;
    }
    if neg {
        LEGACY_SIGNS[1] * m
    } else {
        LEGACY_SIGNS[0] * m
    }
}

#[inline]
pub(crate) fn legacy_logseries<B: BitGenerator>(rng: &mut B, p: f64) -> i64 {
    let r = (1.0 - p).ln();
    loop {
        let v = legacy_double(rng);
        if v >= p {
            return 1;
        }
        let u = legacy_double(rng);
        let q = 1.0 - (r * u).exp();
        if v <= q * q {
            return (1.0 + v.ln() / q.ln()).floor() as i64;
        }
        if v >= q {
            return 1;
        }
        return 2;
    }
}

#[inline]
pub(crate) fn legacy_geometric<B: BitGenerator>(rng: &mut B, p: f64) -> i64 {
    if p >= 0.333_333_333_333_333_3 {
        let mut x = 1i64;
        let mut sum = p;
        let mut prod = p;
        let q = 1.0 - p;
        let u = legacy_double(rng);
        while u > sum {
            prod *= q;
            sum += prod;
            x += 1;
        }
        x
    } else {
        // RandomState preserves the legacy inverse-CDF formula and its exact
        // uniform consumption, independently of Generator's exponential sampler.
        ((-legacy_double(rng)).ln_1p() / (1.0 - p).ln()).ceil() as i64
    }
}

#[inline]
pub(crate) fn legacy_negative_binomial<B: BitGenerator>(
    rng: &mut B,
    n: f64,
    p: f64,
    has_gauss: &mut bool,
    gauss: &mut f64,
) -> i64 {
    let y = ((1.0 - p) / p) * legacy_standard_gamma(rng, n, has_gauss, gauss);
    random_poisson(rng, y)
}

fn legacy_hypergeometric_hyp<B: BitGenerator>(rng: &mut B, good: i64, bad: i64, sample: i64) -> i64 {
    let d1 = bad + good - sample;
    let d2 = bad.min(good) as f64;
    let mut y = d2;
    let mut k = sample;
    while y > 0.0 {
        let u = legacy_double(rng);
        let step = (u + y / ((d1 + k) as f64)).floor() as i64;
        y -= step as f64;
        k -= 1;
        if k == 0 {
            break;
        }
    }
    let mut z = (d2 - y) as i64;
    if good > bad {
        z = sample - z;
    }
    z
}

fn legacy_hypergeometric_hrua<B: BitGenerator>(
    rng: &mut B,
    good: i64,
    bad: i64,
    sample: i64,
) -> i64 {
    let mingoodbad = good.min(bad);
    let popsize = good + bad;
    let maxgoodbad = good.max(bad);
    let m = sample.min(popsize - sample);
    let d4 = (mingoodbad as f64) / (popsize as f64);
    let d5 = 1.0 - d4;
    let d6 = (m as f64) * d4 + 0.5;
    let d7 = (((popsize - m) as f64) * (sample as f64) * d4 * d5 / ((popsize - 1) as f64) + 0.5)
        .sqrt();
    let d8 = HYPERGEOM_D1 * d7 + HYPERGEOM_D2;
    let d9 = (((m + 1) as f64) * ((mingoodbad + 1) as f64) / ((popsize + 2) as f64)).floor() as i64;
    let d10 = random_loggam((d9 + 1) as f64)
        + random_loggam((mingoodbad - d9 + 1) as f64)
        + random_loggam((m - d9 + 1) as f64)
        + random_loggam((maxgoodbad - m + d9 + 1) as f64);
    let d11 = ((m.min(mingoodbad) + 1) as f64).min((d6 + 16.0 * d7).floor());

    let mut z: i64;
    loop {
        let x = legacy_double(rng);
        let y = legacy_double(rng);
        let w = d6 + d8 * (y - 0.5) / x;
        if !(0.0..d11).contains(&w) {
            continue;
        }
        z = w.floor() as i64;
        let t = d10
            - (random_loggam((z + 1) as f64)
                + random_loggam((mingoodbad - z + 1) as f64)
                + random_loggam((m - z + 1) as f64)
                + random_loggam((maxgoodbad - m + z + 1) as f64));
        if (x * (4.0 - x) - 3.0) <= t {
            break;
        }
        if x * (x - t) >= 1.0 {
            continue;
        }
        if 2.0 * x.ln() <= t {
            break;
        }
    }

    if good > bad {
        z = m - z;
    }
    if m < sample {
        z = good - z;
    }
    z
}

pub(crate) fn legacy_hypergeometric<B: BitGenerator>(
    rng: &mut B,
    good: i64,
    bad: i64,
    sample: i64,
) -> i64 {
    if sample > 10 {
        legacy_hypergeometric_hrua(rng, good, bad, sample)
    } else if sample > 0 {
        legacy_hypergeometric_hyp(rng, good, bad, sample)
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Dense pure-Rust matrix factorizations for multivariate_normal
// ---------------------------------------------------------------------------

/// Validation mode when checking if covariance matrix is positive semidefinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckValid {
    /// Check PSD and continue silently (legacy default behavior where warnings cannot propagate).
    #[default]
    Warn,
    /// Return [`NumPyError::LinAlgError`] if covariance fails the PSD check.
    Raise,
    /// Skip the PSD validation check.
    Ignore,
}

/// Covariance factorization algorithm for [`Generator::multivariate_normal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MvNormalMethod {
    /// Singular value decomposition (NumPy default).
    #[default]
    Svd,
    /// Symmetric eigendecomposition.
    Eigh,
    /// Cholesky lower triangular factor.
    Cholesky,
}

/// Method used by [`Generator::multivariate_hypergeometric`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MvHypergeomMethod {
    /// Sequential univariate hypergeometric draws.
    #[default]
    Marginals,
    /// Reservoir-free element count table sampling when total items fit reasonably.
    Count,
}

/// Pure-Rust Cholesky decomposition returning lower-triangular `L` (row-major `n x n`)
/// such that `L * L^T = a`.
pub(crate) fn cholesky_lower(n: usize, a: &[f64]) -> Result<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(NumPyError::LinAlgError(
                        "Matrix is not positive definite".into(),
                    ));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Cyclic Jacobi symmetric eigendecomposition returning `(eigenvalues, U)` where `A = U diag(w) U^T`
/// and `U` is row-major `n x n`.
pub(crate) fn symmetric_eigh(n: usize, a_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = 0.5 * (a_in[i * n + j] + a_in[j * n + i]);
        }
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n <= 1 {
        let evals = if n == 1 { vec![a[0]] } else { vec![] };
        return (evals, v);
    }

    let max_sweeps = 64;
    for _ in 0..max_sweeps {
        let mut off_norm = 0.0f64;
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                off_norm += apq * apq;
            }
        }
        if off_norm <= 1e-30 {
            break;
        }
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-16 * (a[p * n + p].abs() + a[q * n + q].abs() + 1e-30) {
                    continue;
                }
                let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        let np = c * arp - s * arq;
                        let nq = s * arp + c * arq;
                        a[r * n + p] = np;
                        a[p * n + r] = np;
                        a[r * n + q] = nq;
                        a[q * n + r] = nq;
                    }
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut evals = Vec::with_capacity(n);
    let mut u = vec![0.0f64; n * n];
    for (col_new, &(val, col_old)) in pairs.iter().enumerate() {
        evals.push(val);
        for row in 0..n {
            u[row * n + col_new] = v[row * n + col_old];
        }
    }
    (evals, u)
}

/// One-sided Hestenes Jacobi SVD on `n x n` matrix `cov`, returning `A = U * diag(s) * V^T`.
pub(crate) fn jacobi_svd(n: usize, a_in: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u = a_in.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_sweeps = 64;
    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for r in 0..n {
                    let up = u[r * n + p];
                    let uq = u[r * n + q];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                let limit = 1e-15 * (alpha * beta).sqrt();
                if gamma.abs() <= limit {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for r in 0..n {
                    let up = u[r * n + p];
                    let uq = u[r * n + q];
                    u[r * n + p] = c * up - s * uq;
                    u[r * n + q] = s * up + c * uq;

                    let vp = v[r * n + p];
                    let vq = v[r * n + q];
                    v[r * n + p] = c * vp - s * vq;
                    v[r * n + q] = s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut cols: Vec<(f64, usize)> = (0..n)
        .map(|j| {
            let mut norm2 = 0.0;
            for i in 0..n {
                norm2 += u[i * n + j] * u[i * n + j];
            }
            (norm2.sqrt(), j)
        })
        .collect();
    cols.sort_by(|x, y| y.0.total_cmp(&x.0));

    let mut u_out = vec![0.0f64; n * n];
    let mut s_out = vec![0.0f64; n];
    let mut vt_out = vec![0.0f64; n * n];
    for (new_j, &(sigma, old_j)) in cols.iter().enumerate() {
        s_out[new_j] = sigma;
        for i in 0..n {
            u_out[i * n + new_j] = if sigma > 0.0 {
                u[i * n + old_j] / sigma
            } else if i == new_j {
                1.0
            } else {
                0.0
            };
            vt_out[new_j * n + i] = v[i * n + old_j];
        }
    }
    (u_out, s_out, vt_out)
}

/// Build the linear factor `A` (row-major `n x n`) so `X = mean + Z * A^T` has covariance `cov`.
pub(crate) fn compute_cov_factor(
    n: usize,
    cov: &[f64],
    method: MvNormalMethod,
    check_valid: CheckValid,
    tol: f64,
) -> Result<Vec<f64>> {
    for &v in cov {
        if !v.is_finite() {
            return Err(NumPyError::InvalidValue(
                "covariance matrix must contain only finite values".into(),
            ));
        }
    }
    match method {
        MvNormalMethod::Cholesky => cholesky_lower(n, cov),
        MvNormalMethod::Eigh => {
            let (evals, u) = symmetric_eigh(n, cov);
            if check_valid != CheckValid::Ignore {
                let mut sym_ok = true;
                for i in 0..n {
                    for j in (i + 1)..n {
                        let a = cov[i * n + j];
                        let b = cov[j * n + i];
                        if (a - b).abs() > tol + tol * b.abs() {
                            sym_ok = false;
                            break;
                        }
                    }
                }
                let psd = sym_ok && !evals.iter().any(|&s| s < -tol);
                if !psd && check_valid == CheckValid::Raise {
                    return Err(NumPyError::LinAlgError(
                        "covariance is not symmetric positive-semidefinite.".into(),
                    ));
                }
            }
            let mut factor = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    factor[i * n + j] = u[i * n + j] * evals[j].abs().sqrt();
                }
            }
            Ok(factor)
        }
        MvNormalMethod::Svd => {
            let (u, s, vt) = jacobi_svd(n, cov);
            if check_valid != CheckValid::Ignore {
                // Verify cov ≈ vt^T * diag(s) * vt within atol=tol, rtol=tol
                let mut psd = true;
                for i in 0..n {
                    for j in 0..n {
                        let mut recon = 0.0;
                        for k in 0..n {
                            recon += vt[k * n + i] * s[k] * vt[k * n + j];
                        }
                        let target = cov[i * n + j];
                        if (recon - target).abs() > tol + tol * target.abs() {
                            psd = false;
                            break;
                        }
                    }
                    if !psd {
                        break;
                    }
                }
                if !psd && check_valid == CheckValid::Raise {
                    return Err(NumPyError::LinAlgError(
                        "covariance is not symmetric positive-semidefinite.".into(),
                    ));
                }
            }
            let mut factor = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    factor[i * n + j] = u[i * n + j] * s[j].sqrt();
                }
            }
            Ok(factor)
        }
    }
}

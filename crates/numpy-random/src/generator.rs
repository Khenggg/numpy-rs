use crate::bit_generator::BitGenerator;
use crate::distributions::{
    bounded_lemire_u32, bounded_lemire_u64, buffered_bounded_lemire_u16,
    buffered_bounded_lemire_u8, buffered_uint1, compute_cov_factor, gen_mask_u64, kahan_sum,
    random_beta, random_binomial, random_bounded_uint64, random_chisquare, random_exponential,
    random_f, random_gamma,
    random_geometric, random_gumbel, random_hypergeometric, random_interval, random_laplace,
    random_logistic, random_lognormal, random_logseries, random_multinomial,
    random_negative_binomial, random_noncentral_chisquare, random_noncentral_f, random_normal,
    random_pareto, random_poisson, random_power, random_rayleigh, random_standard_cauchy,
    random_standard_exponential, random_standard_exponential_f32, random_standard_gamma,
    random_standard_gamma_f32, random_standard_normal, random_standard_normal_f32,
    random_standard_t, random_triangular, random_uniform, random_vonmises, random_wald,
    random_weibull, random_zipf, BinomialState, CheckValid, MvHypergeomMethod, MvNormalMethod,
    HYPERGEOM_MAX, POISSON_LAM_MAX,
};
use crate::pcg64::PCG64;
use crate::seed_sequence::SeedInput;
use numpy_types::{
    broadcast_many_shapes, broadcast_shapes, normalize_axis, shape_size, AlignedBuffer, NdArray,
    NumPyError, Order, Result, Scalar,
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::mem::MaybeUninit;

/// Minimum element count before Rayon parallel chunks spin up.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 65_536;

/// Trait for integer scalar types supported by [`Generator::integers_as`].
pub trait RandInt: Scalar + Ord {
    fn random_fill<B: BitGenerator>(
        rng: &mut B,
        low: Self,
        high: Option<Self>,
        endpoint: bool,
        out: &mut [MaybeUninit<Self>],
    ) -> Result<()>;
}

macro_rules! impl_signed_rand_int {
    ($ty:ty, $uty:ty) => {
        impl RandInt for $ty {
            fn random_fill<B: BitGenerator>(
                rng: &mut B,
                low: Self,
                high: Option<Self>,
                endpoint: bool,
                out: &mut [MaybeUninit<Self>],
            ) -> Result<()> {
                let (lo, hi) = match high {
                    Some(h) => (low, h),
                    None => (0 as $ty, low),
                };
                if endpoint {
                    if lo > hi {
                        return Err(NumPyError::InvalidValue(format!(
                            "low ({lo}) > high ({hi})"
                        )));
                    }
                    let span = (hi as i128 - lo as i128) as u128;
                    if span > u64::MAX as u128 {
                        return Err(NumPyError::InvalidValue(
                            "integer range exceeds 64-bit unsigned range".into(),
                        ));
                    }
                    fill_unsigned_range(rng, span as u64, out, |u| {
                        (lo as i128 + u as i128) as $ty
                    });
                } else {
                    if lo >= hi {
                        return Err(NumPyError::InvalidValue(format!(
                            "low ({lo}) >= high ({hi})"
                        )));
                    }
                    let span = (hi as i128 - lo as i128 - 1) as u128;
                    if span > u64::MAX as u128 {
                        return Err(NumPyError::InvalidValue(
                            "integer range exceeds 64-bit unsigned range".into(),
                        ));
                    }
                    fill_unsigned_range(rng, span as u64, out, |u| {
                        (lo as i128 + u as i128) as $ty
                    });
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_unsigned_rand_int {
    ($ty:ty) => {
        impl RandInt for $ty {
            fn random_fill<B: BitGenerator>(
                rng: &mut B,
                low: Self,
                high: Option<Self>,
                endpoint: bool,
                out: &mut [MaybeUninit<Self>],
            ) -> Result<()> {
                let (lo, hi) = match high {
                    Some(h) => (low, h),
                    None => (0 as $ty, low),
                };
                if endpoint {
                    if lo > hi {
                        return Err(NumPyError::InvalidValue(format!(
                            "low ({lo}) > high ({hi})"
                        )));
                    }
                    let span = (hi - lo) as u64;
                    fill_unsigned_range(rng, span, out, |u| lo.wrapping_add(u as $ty));
                } else {
                    if lo >= hi {
                        return Err(NumPyError::InvalidValue(format!(
                            "low ({lo}) >= high ({hi})"
                        )));
                    }
                    let span = (hi - lo - 1) as u64;
                    fill_unsigned_range(rng, span, out, |u| lo.wrapping_add(u as $ty));
                }
                Ok(())
            }
        }
    };
}

fn fill_unsigned_range<B: BitGenerator, T, F: Fn(u64) -> T>(
    rng: &mut B,
    rng_max: u64,
    out: &mut [MaybeUninit<T>],
    map: F,
) {
    if rng_max == 0 {
        for slot in out {
            slot.write(map(0));
        }
        return;
    }
    match std::mem::size_of::<T>() {
        1 => {
            let max8 = rng_max as u8;
            let mut bcnt = 0i32;
            let mut buf = 0u32;
            for slot in out {
                let v = buffered_bounded_lemire_u8(rng, max8, &mut bcnt, &mut buf);
                slot.write(map(v as u64));
            }
        }
        2 => {
            let max16 = rng_max as u16;
            let mut bcnt = 0i32;
            let mut buf = 0u32;
            for slot in out {
                let v = buffered_bounded_lemire_u16(rng, max16, &mut bcnt, &mut buf);
                slot.write(map(v as u64));
            }
        }
        4 => {
            let max32 = rng_max as u32;
            for slot in out {
                let v = bounded_lemire_u32(rng, max32);
                slot.write(map(v as u64));
            }
        }
        _ => {
            if rng_max <= u32::MAX as u64 {
                let max32 = rng_max as u32;
                for slot in out {
                    let v = bounded_lemire_u32(rng, max32);
                    slot.write(map(v as u64));
                }
            } else {
                for slot in out {
                    let v = bounded_lemire_u64(rng, rng_max);
                    slot.write(map(v));
                }
            }
        }
    }
}

impl_signed_rand_int!(i8, u8);
impl_signed_rand_int!(i16, u16);
impl_signed_rand_int!(i32, u32);
impl_signed_rand_int!(i64, u64);
impl_unsigned_rand_int!(u8);
impl_unsigned_rand_int!(u16);
impl_unsigned_rand_int!(u32);
impl_unsigned_rand_int!(u64);

impl RandInt for bool {
    fn random_fill<B: BitGenerator>(
        rng: &mut B,
        low: Self,
        high: Option<Self>,
        endpoint: bool,
        out: &mut [MaybeUninit<Self>],
    ) -> Result<()> {
        let (lo, hi) = match high {
            Some(h) => (low as u8, h as u8),
            None => (0u8, low as u8),
        };
        if endpoint {
            if lo > hi {
                return Err(NumPyError::InvalidValue("low > high".into()));
            }
            if lo == hi {
                for slot in out {
                    slot.write(lo != 0);
                }
            } else {
                let mut bcnt = 0i32;
                let mut buf = 0u32;
                for slot in out {
                    slot.write(buffered_uint1(rng, &mut bcnt, &mut buf));
                }
            }
        } else {
            if lo >= hi {
                return Err(NumPyError::InvalidValue("low >= high".into()));
            }
            for slot in out {
                slot.write(lo != 0);
            }
        }
        Ok(())
    }
}

/// Construct a default [`Generator<PCG64>`] seeded from OS entropy (matches `np.random.default_rng()`).
pub fn default_rng() -> Generator<PCG64> {
    Generator::new(PCG64::new())
}

/// Construct a default [`Generator<PCG64>`] from a seed (`np.random.default_rng(seed)`).
pub fn default_rng_from_seed(seed: impl Into<SeedInput>) -> Result<Generator<PCG64>> {
    Ok(Generator::new(PCG64::from_seed(seed)?))
}

#[inline]
pub(crate) fn alloc_and_fill<T: Scalar, F: FnMut(&mut [MaybeUninit<T>])>(
    shape: &[usize],
    mut fill: F,
) -> Result<NdArray<T>> {
    let len = shape_size(shape)?;
    let mut uninit = AlignedBuffer::<T>::new_uninit(len);
    fill(uninit.as_mut_slice());
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

#[inline]
pub(crate) fn alloc_and_par_fill<
    T: Scalar + Send,
    B: BitGenerator,
    F: Fn(&mut B, &mut [MaybeUninit<T>]) + Sync,
>(
    bitgen: &mut B,
    shape: &[usize],
    threshold: usize,
    fill_chunk: F,
) -> Result<NdArray<T>> {
    let len = shape_size(shape)?;
    if len < threshold {
        return alloc_and_fill(shape, |slice| fill_chunk(bitgen, slice));
    }
    let n_threads = rayon::current_num_threads().max(1);
    if n_threads <= 1 {
        return alloc_and_fill(shape, |slice| fill_chunk(bitgen, slice));
    }
    let chunk_size = len.div_ceil(n_threads).max(16_384);
    let n_chunks = len.div_ceil(chunk_size);
    let children = bitgen.spawn_or_fork(n_chunks)?;
    let mut uninit = AlignedBuffer::<T>::new_uninit(len);
    uninit
        .as_mut_slice()
        .par_chunks_mut(chunk_size)
        .zip(children.into_par_iter())
        .for_each(|(chunk, mut child_bg)| {
            fill_chunk(&mut child_bg, chunk);
        });
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

#[inline]
fn validate_non_negative(name: &str, v: f64) -> Result<()> {
    if v.is_nan() || v < 0.0 {
        return Err(NumPyError::InvalidValue(format!("{name} < 0")));
    }
    Ok(())
}

#[inline]
fn validate_positive(name: &str, v: f64) -> Result<()> {
    if v.is_nan() || v <= 0.0 {
        return Err(NumPyError::InvalidValue(format!("{name} <= 0")));
    }
    Ok(())
}

#[inline]
fn validate_finite(name: &str, v: f64) -> Result<()> {
    if !v.is_finite() {
        return Err(NumPyError::InvalidValue(format!("{name} must be finite")));
    }
    Ok(())
}

#[inline]
fn validate_prob_0_1(name: &str, p: f64) -> Result<()> {
    if p.is_nan() || p < 0.0 || p > 1.0 {
        return Err(NumPyError::InvalidValue(format!(
            "{name} < 0, {name} > 1 or {name} is NaN"
        )));
    }
    Ok(())
}

#[inline]
fn validate_prob_gt_0_1(name: &str, p: f64) -> Result<()> {
    if p.is_nan() || p <= 0.0 || p > 1.0 {
        return Err(NumPyError::InvalidValue(format!(
            "{name} <= 0, {name} > 1 or {name} contains NaNs"
        )));
    }
    Ok(())
}

#[inline]
fn validate_prob_lt_0_1(name: &str, p: f64) -> Result<()> {
    if p.is_nan() || p < 0.0 || p >= 1.0 {
        return Err(NumPyError::InvalidValue(format!(
            "{name} < 0, {name} >= 1 or {name} contains NaNs"
        )));
    }
    Ok(())
}

#[inline]
fn check_broadcast_output_shape(param_shape: &[usize], size: Option<&[usize]>) -> Result<Vec<usize>> {
    match size {
        None => Ok(param_shape.to_vec()),
        Some(target) => {
            let b = broadcast_shapes(param_shape, target)?;
            if b != target {
                return Err(NumPyError::BroadcastError(format!(
                    "Output size {target:?} is not compatible with broadcast dimensions of inputs {b:?}."
                )));
            }
            Ok(target.to_vec())
        }
    }
}

/// Primary NumPy `Generator` backed by a statically typed [`BitGenerator`] (default [`PCG64`]).
#[derive(Debug, Clone)]
pub struct Generator<B: BitGenerator = PCG64> {
    bit_generator: B,
    binomial: BinomialState,
}

impl<B: BitGenerator> Generator<B> {
    /// Wrap any [`BitGenerator`] in a [`Generator`].
    pub fn new(bit_generator: B) -> Self {
        Self {
            bit_generator,
            binomial: BinomialState::default(),
        }
    }

    /// Borrow the underlying [`BitGenerator`].
    pub fn bit_generator(&self) -> &B {
        &self.bit_generator
    }

    /// Mutably borrow the underlying [`BitGenerator`].
    pub fn bit_generator_mut(&mut self) -> &mut B {
        self.binomial.reset();
        &mut self.bit_generator
    }

    /// Extract the underlying [`BitGenerator`].
    pub fn into_bit_generator(self) -> B {
        self.bit_generator
    }

    /// Spawn `n_children` independent child [`Generator`] streams via [`SeedSequence::spawn`].
    pub fn spawn(&mut self, n_children: usize) -> Result<Vec<Generator<B>>> {
        let child_bgs = self.bit_generator.spawn(n_children)?;
        Ok(child_bgs.into_iter().map(Generator::new).collect())
    }

    // -----------------------------------------------------------------------
    // Scalar draws (zero allocation)
    // -----------------------------------------------------------------------

    #[inline]
    pub fn random_scalar(&mut self) -> f64 {
        self.bit_generator.next_double()
    }

    #[inline]
    pub fn random_scalar_f32(&mut self) -> f32 {
        self.bit_generator.next_float()
    }

    #[inline]
    pub fn standard_normal_scalar(&mut self) -> f64 {
        random_standard_normal(&mut self.bit_generator)
    }

    #[inline]
    pub fn standard_normal_scalar_f32(&mut self) -> f32 {
        random_standard_normal_f32(&mut self.bit_generator)
    }

    #[inline]
    pub fn standard_exponential_scalar(&mut self) -> f64 {
        random_standard_exponential(&mut self.bit_generator)
    }

    #[inline]
    pub fn standard_exponential_scalar_f32(&mut self) -> f32 {
        random_standard_exponential_f32(&mut self.bit_generator)
    }

    // -----------------------------------------------------------------------
    // Core array draws into 64-byte aligned NdArray<T>
    // -----------------------------------------------------------------------

    /// Draw uniform `f64` samples in `[0.0, 1.0)`.
    pub fn random(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(bg.next_double());
            }
        })
    }

    /// Draw uniform `f32` samples in `[0.0, 1.0)`.
    pub fn random_f32(&mut self, shape: &[usize]) -> Result<NdArray<f32>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(bg.next_float());
            }
        })
    }

    /// Generate `length` random bytes (identical to NumPy `Generator.bytes`).
    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        let n_u32 = length.div_ceil(4);
        let mut out = Vec::with_capacity(n_u32 * 4);
        for _ in 0..n_u32 {
            out.extend_from_slice(&self.bit_generator.next_u32().to_le_bytes());
        }
        out.truncate(length);
        out
    }

    /// Generate `length` random bytes in a 64-byte aligned 1D `NdArray<u8>`.
    pub fn bytes_array(&mut self, length: usize) -> Result<NdArray<u8>> {
        let n_u32 = length.div_ceil(4);
        let bg = &mut self.bit_generator;
        alloc_and_fill(&[length], |slice| {
            let mut idx = 0usize;
            for _ in 0..n_u32 {
                let bytes = bg.next_u32().to_le_bytes();
                let take = (length - idx).min(4);
                for &b in &bytes[..take] {
                    slice[idx].write(b);
                    idx += 1;
                }
            }
        })
    }

    /// Draw unbiased random integers in `[low, high)` (half-open) as `NdArray<i64>` using Lemire's method.
    pub fn integers(&mut self, low: i64, high: i64, shape: &[usize]) -> Result<NdArray<i64>> {
        self.integers_as::<i64>(low, Some(high), shape, false)
    }

    /// Draw unbiased random integers in `[low, high)` or `[low, high]` when `endpoint` is `true`.
    pub fn integers_inclusive(
        &mut self,
        low: i64,
        high: i64,
        shape: &[usize],
        endpoint: bool,
    ) -> Result<NdArray<i64>> {
        self.integers_as::<i64>(low, Some(high), shape, endpoint)
    }

    /// Draw unbiased random integers of arbitrary integer dtype `T: RandInt`.
    pub fn integers_as<T: RandInt>(
        &mut self,
        low: T,
        high: Option<T>,
        shape: &[usize],
        endpoint: bool,
    ) -> Result<NdArray<T>> {
        let len = shape_size(shape)?;
        let mut uninit = AlignedBuffer::<T>::new_uninit(len);
        T::random_fill(
            &mut self.bit_generator,
            low,
            high,
            endpoint,
            uninit.as_mut_slice(),
        )?;
        let buf = unsafe { uninit.assume_init() };
        NdArray::from_aligned_buffer(buf, shape, Order::C)
    }

    /// Draw uniform `f64` samples in `[low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_finite("low", low)?;
        validate_finite("high", high)?;
        let range = high - low;
        if !range.is_finite() {
            return Err(NumPyError::InvalidValue("Range exceeds finite bounds".into()));
        }
        validate_non_negative("high - low", range)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_uniform(bg, low, range));
            }
        })
    }

    /// Draw standard normal (`N(0, 1)`) `f64` samples using Ziggurat.
    pub fn standard_normal(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_normal(bg));
            }
        })
    }

    /// Draw standard normal (`N(0, 1)`) `f32` samples using Ziggurat.
    pub fn standard_normal_f32(&mut self, shape: &[usize]) -> Result<NdArray<f32>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_normal_f32(bg));
            }
        })
    }

    /// Draw normal (`N(loc, scale^2)`) `f64` samples.
    pub fn normal(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_normal(bg, loc, scale));
            }
        })
    }

    /// Draw standard exponential (`scale = 1.0`) `f64` samples using Ziggurat.
    pub fn standard_exponential(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_exponential(bg));
            }
        })
    }

    /// Draw standard exponential (`scale = 1.0`) `f32` samples using Ziggurat.
    pub fn standard_exponential_f32(&mut self, shape: &[usize]) -> Result<NdArray<f32>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_exponential_f32(bg));
            }
        })
    }

    /// Draw exponential samples with parameter `scale >= 0`.
    pub fn exponential(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_exponential(bg, scale));
            }
        })
    }

    /// Draw standard gamma (`scale = 1.0`) `f64` samples using Marsaglia-Tsang / Ahrens-Dieter.
    pub fn standard_gamma(&mut self, shape_param: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("shape", shape_param)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_gamma(bg, shape_param));
            }
        })
    }

    /// Draw standard gamma (`scale = 1.0`) `f32` samples.
    pub fn standard_gamma_f32(&mut self, shape_param: f32, shape: &[usize]) -> Result<NdArray<f32>> {
        if shape_param.is_nan() || shape_param < 0.0 {
            return Err(NumPyError::InvalidValue("shape < 0".into()));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_gamma_f32(bg, shape_param));
            }
        })
    }

    /// Draw gamma samples with parameters `shape_param >= 0` and `scale >= 0`.
    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("shape", shape_param)?;
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_gamma(bg, shape_param, scale));
            }
        })
    }

    /// Draw beta samples with parameters `a > 0`, `b > 0`.
    pub fn beta(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("a", a)?;
        validate_positive("b", b)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_beta(bg, a, b));
            }
        })
    }

    /// Draw binomial integer samples (`n >= 0`, `0 <= p <= 1`).
    pub fn binomial(&mut self, n: i64, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if n < 0 {
            return Err(NumPyError::InvalidValue("n < 0".into()));
        }
        validate_prob_0_1("p", p)?;
        let bg = &mut self.bit_generator;
        let bin = &mut self.binomial;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_binomial(bg, n, p, bin));
            }
        })
    }

    /// Draw negative binomial integer samples (`n > 0`, `0 < p <= 1`).
    pub fn negative_binomial(&mut self, n: f64, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        validate_positive("n", n)?;
        validate_prob_gt_0_1("p", p)?;
        let max_lam = ((1.0 - p) / p) * (n + 10.0 * n.sqrt());
        if max_lam > POISSON_LAM_MAX {
            return Err(NumPyError::InvalidValue(
                "n too large or p too small".into(),
            ));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_negative_binomial(bg, n, p));
            }
        })
    }

    /// Draw Poisson integer samples (`0 <= lam <= POISSON_LAM_MAX`).
    pub fn poisson(&mut self, lam: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if lam.is_nan() || lam < 0.0 || lam > POISSON_LAM_MAX {
            return Err(NumPyError::InvalidValue(
                "lam value too large or negative".into(),
            ));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_poisson(bg, lam));
            }
        })
    }

    /// Draw geometric integer samples (`0 < p <= 1`).
    pub fn geometric(&mut self, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        validate_prob_gt_0_1("p", p)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_geometric(bg, p));
            }
        })
    }

    /// Draw hypergeometric integer samples (`ngood >= 0`, `nbad >= 0`, `0 <= nsample <= ngood + nbad`).
    pub fn hypergeometric(
        &mut self,
        ngood: i64,
        nbad: i64,
        nsample: i64,
        shape: &[usize],
    ) -> Result<NdArray<i64>> {
        if ngood < 0 || nbad < 0 || nsample < 0 {
            return Err(NumPyError::InvalidValue(
                "ngood, nbad, and nsample must be non-negative".into(),
            ));
        }
        if ngood >= HYPERGEOM_MAX || nbad >= HYPERGEOM_MAX {
            return Err(NumPyError::InvalidValue(format!(
                "both ngood and nbad must be less than {HYPERGEOM_MAX}"
            )));
        }
        let total = ngood
            .checked_add(nbad)
            .ok_or_else(|| NumPyError::InvalidValue("ngood + nbad overflows i64".into()))?;
        if total < nsample {
            return Err(NumPyError::InvalidValue("ngood + nbad < nsample".into()));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_hypergeometric(bg, ngood, nbad, nsample));
            }
        })
    }

    /// Draw log-normal samples (`scale >= 0`).
    pub fn lognormal(&mut self, mean: f64, sigma: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("sigma", sigma)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_lognormal(bg, mean, sigma));
            }
        })
    }

    /// Draw logarithmic series integer samples (`0 <= p < 1`).
    pub fn logseries(&mut self, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        validate_prob_lt_0_1("p", p)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_logseries(bg, p));
            }
        })
    }

    /// Draw chi-square samples (`df > 0`).
    pub fn chisquare(&mut self, df: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("df", df)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_chisquare(bg, df));
            }
        })
    }

    /// Draw noncentral chi-square samples (`df > 0`, `nonc >= 0`).
    pub fn noncentral_chisquare(
        &mut self,
        df: f64,
        nonc: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        validate_positive("df", df)?;
        validate_non_negative("nonc", nonc)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_noncentral_chisquare(bg, df, nonc));
            }
        })
    }

    /// Draw F-distribution samples (`dfnum > 0`, `dfden > 0`).
    pub fn f(&mut self, dfnum: f64, dfden: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("dfnum", dfnum)?;
        validate_positive("dfden", dfden)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_f(bg, dfnum, dfden));
            }
        })
    }

    /// Draw noncentral F-distribution samples (`dfnum > 0`, `dfden > 0`, `nonc >= 0`).
    pub fn noncentral_f(
        &mut self,
        dfnum: f64,
        dfden: f64,
        nonc: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        validate_positive("dfnum", dfnum)?;
        validate_positive("dfden", dfden)?;
        validate_non_negative("nonc", nonc)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_noncentral_f(bg, dfnum, dfden, nonc));
            }
        })
    }

    /// Draw Student's t samples (`df > 0`).
    pub fn standard_t(&mut self, df: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("df", df)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_t(bg, df));
            }
        })
    }

    /// Draw standard Cauchy samples (`loc = 0`, `scale = 1`).
    pub fn standard_cauchy(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_standard_cauchy(bg));
            }
        })
    }

    /// Draw Laplace samples (`scale >= 0`).
    pub fn laplace(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_laplace(bg, loc, scale));
            }
        })
    }

    /// Draw logistic samples (`scale >= 0`).
    pub fn logistic(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_logistic(bg, loc, scale));
            }
        })
    }

    /// Draw Gumbel samples (`scale >= 0`).
    pub fn gumbel(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_gumbel(bg, loc, scale));
            }
        })
    }

    /// Draw Rayleigh samples (`scale >= 0`).
    pub fn rayleigh(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_rayleigh(bg, scale));
            }
        })
    }

    /// Draw Weibull samples (`a >= 0`).
    pub fn weibull(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("a", a)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_weibull(bg, a));
            }
        })
    }

    /// Draw Pareto II (Lomax) samples (`a > 0`).
    pub fn pareto(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("a", a)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_pareto(bg, a));
            }
        })
    }

    /// Draw power distribution samples (`a > 0`).
    pub fn power(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("a", a)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_power(bg, a));
            }
        })
    }

    /// Draw triangular distribution samples (`left <= mode <= right`, `left < right`).
    pub fn triangular(
        &mut self,
        left: f64,
        mode: f64,
        right: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        validate_finite("left", left)?;
        validate_finite("mode", mode)?;
        validate_finite("right", right)?;
        if left > mode {
            return Err(NumPyError::InvalidValue("left > mode".into()));
        }
        if mode > right {
            return Err(NumPyError::InvalidValue("mode > right".into()));
        }
        if left == right {
            return Err(NumPyError::InvalidValue("left == right".into()));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_triangular(bg, left, mode, right));
            }
        })
    }

    /// Draw von Mises distribution samples (`kappa >= 0`).
    pub fn vonmises(&mut self, mu: f64, kappa: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        if kappa < 0.0 {
            return Err(NumPyError::InvalidValue("kappa < 0".into()));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_vonmises(bg, mu, kappa));
            }
        })
    }

    /// Draw Wald (inverse Gaussian) samples (`mean > 0`, `scale > 0`).
    pub fn wald(&mut self, mean: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_positive("mean", mean)?;
        validate_positive("scale", scale)?;
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_wald(bg, mean, scale));
            }
        })
    }

    /// Draw Zipf integer samples (`a > 1`).
    pub fn zipf(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if a.is_nan() || a <= 1.0 {
            return Err(NumPyError::InvalidValue("a <= 1".into()));
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_zipf(bg, a));
            }
        })
    }

    // -----------------------------------------------------------------------
    // Multivariate distributions
    // -----------------------------------------------------------------------

    /// Draw Dirichlet samples (`alpha >= 0`), returning shape `size + [k]` (or `[k]` if `size` is `None`).
    pub fn dirichlet(&mut self, alpha: &[f64], size: Option<&[usize]>) -> Result<NdArray<f64>> {
        for &a in alpha {
            validate_non_negative("alpha", a)?;
        }
        let k = alpha.len();
        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(k);
        if k == 0 {
            return NdArray::zeros(&out_shape);
        }
        let max_alpha = alpha.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bg = &mut self.bit_generator;
        if max_alpha < 0.1 {
            let mut alpha_csum = vec![0.0f64; k];
            let mut csum = 0.0;
            for j in (0..k).rev() {
                csum += alpha[j];
                alpha_csum[j] = csum;
            }
            if csum == 0.0 {
                return alloc_and_fill(&out_shape, |slice| {
                    for slot in slice {
                        slot.write(0.0);
                    }
                });
            }
            alloc_and_fill(&out_shape, |slice| {
                for chunk in slice.chunks_exact_mut(k) {
                    let mut acc = 1.0;
                    let mut early_end = false;
                    for j in 0..(k - 1) {
                        if alpha_csum[j + 1] == 0.0 {
                            chunk[j].write(acc);
                            for rest in &mut chunk[(j + 1)..] {
                                rest.write(0.0);
                            }
                            early_end = true;
                            break;
                        }
                        let v = random_beta(bg, alpha[j], alpha_csum[j + 1]);
                        chunk[j].write(acc * v);
                        acc *= 1.0 - v;
                    }
                    if !early_end {
                        chunk[k - 1].write(acc);
                    }
                }
            })
        } else {
            alloc_and_fill(&out_shape, |slice| {
                for chunk in slice.chunks_exact_mut(k) {
                    let mut acc = 0.0;
                    for j in 0..k {
                        let g = random_standard_gamma(bg, alpha[j]);
                        acc += g;
                        chunk[j].write(g);
                    }
                    let inv = 1.0 / acc;
                    for slot in chunk {
                        unsafe {
                            let val = *slot.assume_init_ref();
                            slot.write(val * inv);
                        }
                    }
                }
            })
        }
    }

    /// Draw multinomial samples for scalar `n` and 1D `pvals`, returning shape `size + [k]`.
    pub fn multinomial(
        &mut self,
        n: i64,
        pvals: &[f64],
        size: Option<&[usize]>,
    ) -> Result<NdArray<i64>> {
        if n < 0 {
            return Err(NumPyError::InvalidValue("n < 0".into()));
        }
        let d = pvals.len();
        if d == 0 {
            return Err(NumPyError::InvalidValue(
                "pvals must have at least 1 element".into(),
            ));
        }
        for &p in pvals {
            validate_prob_0_1("pvals", p)?;
        }
        if kahan_sum(&pvals[..(d - 1)]) > 1.0 + 1e-12 {
            return Err(NumPyError::InvalidValue("sum(pvals[:-1]) > 1.0".into()));
        }
        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(d);
        let bg = &mut self.bit_generator;
        let bin = &mut self.binomial;
        alloc_and_fill(&out_shape, |slice| {
            let mut row_buf = vec![0i64; d];
            for chunk in slice.chunks_exact_mut(d) {
                random_multinomial(bg, n, &mut row_buf, pvals, bin);
                for (slot, &val) in chunk.iter_mut().zip(row_buf.iter()) {
                    slot.write(val);
                }
            }
        })
    }

    /// Draw multinomial samples with array broadcasting across `n` and `pvals`.
    pub fn multinomial_array(
        &mut self,
        n: &NdArray<i64>,
        pvals: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<i64>> {
        if pvals.ndim() < 1 || pvals.shape()[pvals.ndim() - 1] == 0 {
            return Err(NumPyError::InvalidValue(
                "pvals must have at least 1 dimension and last dimension > 0".into(),
            ));
        }
        let p_contig = pvals.to_c_contiguous();
        let p_slice = p_contig.as_slice().ok_or_else(|| {
            NumPyError::InvalidValue("failed to materialize contiguous pvals".into())
        })?;
        for nv in n.iter() {
            if nv < 0 {
                return Err(NumPyError::InvalidValue("n < 0".into()));
            }
        }
        for &pv in p_slice {
            validate_prob_0_1("pvals", pv)?;
        }
        let d = pvals.shape()[pvals.ndim() - 1];
        for p_row in p_slice.chunks_exact(d) {
            if kahan_sum(&p_row[..(d - 1)]) > 1.0 + 1e-12 {
                return Err(NumPyError::InvalidValue(
                    "sum(pvals[..., :-1]) > 1.0".into(),
                ));
            }
        }
        let p_batch_shape = &pvals.shape()[..(pvals.ndim() - 1)];
        let bcast = broadcast_shapes(n.shape(), p_batch_shape)?;
        let out_batch_shape = check_broadcast_output_shape(&bcast, size)?;
        let n_bcast = n.broadcast_to(&out_batch_shape)?;
        let p_bcast_shape = [out_batch_shape.as_slice(), &[d]].concat();
        let p_bcast = p_contig.broadcast_to(&p_bcast_shape)?;

        let mut out_shape = out_batch_shape;
        out_shape.push(d);
        let n_vals: Vec<i64> = n_bcast.iter().collect();
        let p_vals: Vec<f64> = p_bcast.iter().collect();

        let bg = &mut self.bit_generator;
        let bin = &mut self.binomial;
        alloc_and_fill(&out_shape, |slice| {
            let mut row_buf = vec![0i64; d];
            for (i, chunk) in slice.chunks_exact_mut(d).enumerate() {
                random_multinomial(
                    bg,
                    n_vals[i],
                    &mut row_buf,
                    &p_vals[(i * d)..((i + 1) * d)],
                    bin,
                );
                for (slot, &val) in chunk.iter_mut().zip(row_buf.iter()) {
                    slot.write(val);
                }
            }
        })
    }

    /// Draw multivariate normal samples with default SVD factorization, `CheckValid::Raise`, `tol = 1e-8`.
    pub fn multivariate_normal(
        &mut self,
        mean: &[f64],
        cov: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<f64>> {
        self.multivariate_normal_with(
            mean,
            cov,
            size,
            CheckValid::Raise,
            1e-8,
            MvNormalMethod::Svd,
        )
    }

    /// Draw multivariate normal samples with explicit factorization `method` and PSD checking.
    pub fn multivariate_normal_with(
        &mut self,
        mean: &[f64],
        cov: &NdArray<f64>,
        size: Option<&[usize]>,
        check_valid: CheckValid,
        tol: f64,
        method: MvNormalMethod,
    ) -> Result<NdArray<f64>> {
        if cov.ndim() != 2 || cov.shape()[0] != cov.shape()[1] {
            return Err(NumPyError::ShapeMismatch(
                "cov must be 2 dimensional and square".into(),
            ));
        }
        let d = mean.len();
        if cov.shape()[0] != d {
            return Err(NumPyError::ShapeMismatch(
                "mean and cov must have same length".into(),
            ));
        }
        if d == 0 {
            return Err(NumPyError::InvalidValue(
                "mean and cov must be non-empty".into(),
            ));
        }
        let cov_contig = cov.to_c_contiguous();
        let cov_slice = cov_contig
            .as_slice()
            .ok_or_else(|| NumPyError::InvalidValue("failed to materialize cov".into()))?;
        let factor = compute_cov_factor(d, cov_slice, method, check_valid, tol)?;

        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(d);

        let bg = &mut self.bit_generator;
        let mut z_row = vec![0.0f64; d];
        alloc_and_fill(&out_shape, |slice| {
            for chunk in slice.chunks_exact_mut(d) {
                for z in &mut z_row {
                    *z = random_standard_normal(bg);
                }
                for c in 0..d {
                    let mut sum = mean[c];
                    let row_off = c * d;
                    for k in 0..d {
                        sum += z_row[k] * factor[row_off + k];
                    }
                    chunk[c].write(sum);
                }
            }
        })
    }

    /// Draw multivariate hypergeometric samples (`colors >= 0`, `0 <= nsample <= sum(colors)`).
    pub fn multivariate_hypergeometric(
        &mut self,
        colors: &[i64],
        nsample: i64,
        size: Option<&[usize]>,
        method: MvHypergeomMethod,
    ) -> Result<NdArray<i64>> {
        let k = colors.len();
        if nsample < 0 {
            return Err(NumPyError::InvalidValue("nsample < 0".into()));
        }
        let mut total: i64 = 0;
        for &c in colors {
            if c < 0 {
                return Err(NumPyError::InvalidValue("colors < 0".into()));
            }
            total = total
                .checked_add(c)
                .ok_or_else(|| NumPyError::InvalidValue("sum(colors) overflows i64".into()))?;
        }
        if nsample > total {
            return Err(NumPyError::InvalidValue("nsample > sum(colors)".into()));
        }
        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(k);
        if k == 0 {
            return alloc_and_fill(&out_shape, |_| {});
        }
        let bg = &mut self.bit_generator;

        match method {
            MvHypergeomMethod::Marginals => {
                if total >= HYPERGEOM_MAX {
                    return Err(NumPyError::InvalidValue(format!(
                        "When method is \"marginals\", sum(colors) must be less than {HYPERGEOM_MAX}."
                    )));
                }
                alloc_and_fill(&out_shape, |slice| {
                    for chunk in slice.chunks_exact_mut(k) {
                        if nsample == 0 {
                            for slot in chunk {
                                slot.write(0);
                            }
                            continue;
                        }
                        if total == nsample {
                            for (slot, &c) in chunk.iter_mut().zip(colors.iter()) {
                                slot.write(c);
                            }
                            continue;
                        }
                        let mut more = nsample;
                        let mut bad = total;
                        let mut chosen = 0i64;
                        for (j, &c) in colors.iter().enumerate() {
                            if j == k - 1 || more == 0 {
                                chunk[j].write(more);
                                more = 0;
                                continue;
                            }
                            bad -= c;
                            if bad <= 0 {
                                chunk[j].write(more);
                                more = 0;
                                continue;
                            }
                            chosen = random_hypergeometric(bg, c, bad, more);
                            chunk[j].write(chosen);
                            more -= chosen;
                        }
                        let _ = chosen;
                    }
                })
            }
            MvHypergeomMethod::Count => {
                let max_count_pop = (isize::MAX as usize) / std::mem::size_of::<usize>();
                let total_usize = usize::try_from(total)
                    .ok()
                    .filter(|&t| t <= max_count_pop)
                    .ok_or_else(|| {
                        NumPyError::InvalidValue(format!(
                            "sum(colors)={total} is too large for method='count'"
                        ))
                    })?;
                let mut pop: Vec<usize> = Vec::new();
                pop.try_reserve_exact(total_usize).map_err(|_| {
                    NumPyError::InvalidValue(format!(
                        "insufficient memory for multivariate_hypergeometric with method='count' and sum(colors)={total}"
                    ))
                })?;
                for (color_idx, &count) in colors.iter().enumerate() {
                    pop.resize(pop.len() + count as usize, color_idx);
                }
                alloc_and_fill(&out_shape, |slice| {
                    let n_tot = pop.len();
                    let mut counts = vec![0i64; k];
                    for chunk in slice.chunks_exact_mut(k) {
                        counts.fill(0);
                        let draw = nsample as usize;
                        for i in 0..draw {
                            let j = i + random_interval(bg, (n_tot - 1 - i) as u64) as usize;
                            pop.swap(i, j);
                            counts[pop[i]] += 1;
                        }
                        for (slot, &cnt) in chunk.iter_mut().zip(counts.iter()) {
                            slot.write(cnt);
                        }
                    }
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Sampling, Permutations, Shuffling
    // -----------------------------------------------------------------------

    /// Draw random samples from `0..pop_size` with or without replacement and optional probabilities `p`.
    pub fn choice_range(
        &mut self,
        pop_size: usize,
        size: &[usize],
        replace: bool,
        p: Option<&[f64]>,
        shuffle: bool,
    ) -> Result<NdArray<i64>> {
        let out_total = shape_size(size)?;
        if pop_size == 0 {
            if out_total == 0 && p.is_none() {
                return alloc_and_fill(size, |_| {});
            }
            return Err(NumPyError::InvalidValue(
                "population size must be greater than 0".into(),
            ));
        }

        if let Some(probs) = p {
            if probs.len() != pop_size {
                return Err(NumPyError::ShapeMismatch(
                    "a and p must have same size".into(),
                ));
            }
            let p_sum = kahan_sum(probs);
            if p_sum.is_nan() {
                return Err(NumPyError::InvalidValue("Probabilities contain NaN".into()));
            }
            for &pv in probs {
                if pv < 0.0 {
                    return Err(NumPyError::InvalidValue(
                        "Probabilities are not non-negative".into(),
                    ));
                }
            }
            const ATOL: f64 = 1.490_116_119_384_765_6e-8;
            if (p_sum - 1.0).abs() > ATOL {
                return Err(NumPyError::InvalidValue(
                    "Probabilities do not sum to 1".into(),
                ));
            }

            if replace {
                let mut cdf = Vec::with_capacity(pop_size);
                let mut run = 0.0f64;
                for &pv in probs {
                    run += pv;
                    cdf.push(run);
                }
                let last = *cdf.last().unwrap_or(&1.0);
                for c in &mut cdf {
                    *c /= last;
                }
                let u_arr = self.random(size)?;
                let u_slice = u_arr.as_slice().unwrap();
                return alloc_and_fill(size, |slice| {
                    for (slot, &u) in slice.iter_mut().zip(u_slice.iter()) {
                        let idx = cdf.partition_point(|&v| v <= u).min(pop_size - 1);
                        slot.write(idx as i64);
                    }
                });
            } else {
                if out_total > pop_size {
                    return Err(NumPyError::InvalidValue(
                        "Cannot take a larger sample than population when replace is False"
                            .into(),
                    ));
                }
                let nz = probs.iter().filter(|&&v| v > 0.0).count();
                if nz < out_total {
                    return Err(NumPyError::InvalidValue(
                        "Fewer non-zero entries in p than size".into(),
                    ));
                }
                let mut p_work = probs.to_vec();
                let mut found = Vec::with_capacity(out_total);
                while found.len() < out_total {
                    let rem = out_total - found.len();
                    let mut cdf = Vec::with_capacity(pop_size);
                    let mut run = 0.0;
                    for &pv in &p_work {
                        run += pv;
                        cdf.push(run);
                    }
                    let last = *cdf.last().unwrap_or(&1.0);
                    for c in &mut cdf {
                        *c /= last;
                    }
                    let x = self.random(&[rem])?;
                    let mut seen_batch = HashSet::new();
                    for &u in x.as_slice().unwrap() {
                        let idx = cdf.partition_point(|&v| v <= u).min(pop_size - 1);
                        if seen_batch.insert(idx) && p_work[idx] > 0.0 {
                            found.push(idx as i64);
                            if found.len() == out_total {
                                break;
                            }
                        }
                    }
                    for &idx in &seen_batch {
                        p_work[idx] = 0.0;
                    }
                }
                return alloc_and_fill(size, |slice| {
                    for (slot, &v) in slice.iter_mut().zip(found.iter()) {
                        slot.write(v);
                    }
                });
            }
        }

        if replace {
            self.integers(0, pop_size as i64, size)
        } else {
            if out_total > pop_size {
                return Err(NumPyError::InvalidValue(
                    "Cannot take a larger sample than population when replace is False".into(),
                ));
            }
            let cutoff = if shuffle { 50 } else { 20 };
            if pop_size > 10_000 && out_total > pop_size / cutoff {
                let mut idx: Vec<i64> = (0..(pop_size as i64)).collect();
                let start = (pop_size - out_total).max(1);
                for i in (start..pop_size).rev() {
                    let j = random_bounded_uint64(&mut self.bit_generator, 0, i as u64) as usize;
                    idx.swap(i, j);
                }
                let tail = &idx[(pop_size - out_total)..];
                alloc_and_fill(size, |slice| {
                    for (slot, &v) in slice.iter_mut().zip(tail.iter()) {
                        slot.write(v);
                    }
                })
            } else {
                // Floyd's algorithm with open-addressed hash table matching NumPy.
                let set_min = ((1.2f64 * (out_total as f64)) as u64).max(1);
                let mask = gen_mask_u64(set_min) as usize;
                let set_size = mask + 1;
                let mut hash_set = vec![u64::MAX; set_size];
                let bg = &mut self.bit_generator;
                alloc_and_fill(size, |slice| {
                    let base = pop_size - out_total;
                    for j in base..pop_size {
                        let val = random_bounded_uint64(bg, 0, j as u64);
                        let mut loc = (val as usize) & mask;
                        while hash_set[loc] != u64::MAX && hash_set[loc] != val {
                            loc = (loc + 1) & mask;
                        }
                        if hash_set[loc] == u64::MAX {
                            hash_set[loc] = val;
                            slice[j - base].write(val as i64);
                        } else {
                            let j_u = j as u64;
                            let mut loc_j = j & mask;
                            while hash_set[loc_j] != u64::MAX {
                                loc_j = (loc_j + 1) & mask;
                            }
                            hash_set[loc_j] = j_u;
                            slice[j - base].write(j as i64);
                        }
                    }
                    if shuffle && out_total > 1 {
                        for i in (1..out_total).rev() {
                            let j = random_bounded_uint64(bg, 0, i as u64) as usize;
                            slice.swap(i, j);
                        }
                    }
                })
            }
        }
    }

    /// Sample elements from an `NdArray<T>` along `axis`.
    pub fn choice<T: Scalar>(
        &mut self,
        a: &NdArray<T>,
        size: &[usize],
        replace: bool,
        p: Option<&[f64]>,
        axis: usize,
        shuffle: bool,
    ) -> Result<NdArray<T>> {
        if a.ndim() == 0 {
            return Err(NumPyError::InvalidAxis {
                axis: axis as isize,
                ndim: 0,
            });
        }
        let ax = normalize_axis(axis as isize, a.ndim())?;
        let pop_size = a.shape()[ax];
        let idx = self.choice_range(pop_size, size, replace, p, shuffle)?;
        let idx_slice = idx.as_slice().unwrap();

        let a_contig = a.to_c_contiguous();
        let a_slice = a_contig.as_slice().unwrap();

        let outer: usize = a_contig.shape()[..ax].iter().product();
        let inner: usize = a_contig.shape()[(ax + 1)..].iter().product();

        let mut out_shape = Vec::with_capacity(a.ndim() - 1 + size.len());
        out_shape.extend_from_slice(&a_contig.shape()[..ax]);
        out_shape.extend_from_slice(size);
        out_shape.extend_from_slice(&a_contig.shape()[(ax + 1)..]);

        alloc_and_fill(&out_shape, |slice| {
            let mut dst_pos = 0usize;
            for o in 0..outer {
                let base_src = o * pop_size * inner;
                for &sel in idx_slice {
                    let s = sel as usize;
                    let src_off = base_src + s * inner;
                    for k in 0..inner {
                        slice[dst_pos].write(a_slice[src_off + k]);
                        dst_pos += 1;
                    }
                }
            }
        })
    }

    /// Shuffle an `NdArray<T>` in place along `axis` (NumPy `Generator.shuffle`).
    pub fn shuffle<T: Scalar>(&mut self, x: &mut NdArray<T>, axis: usize) -> Result<()> {
        if x.ndim() == 0 {
            return Err(NumPyError::InvalidAxis {
                axis: axis as isize,
                ndim: 0,
            });
        }
        let ax = normalize_axis(axis as isize, x.ndim())?;
        let ax_len = x.shape()[ax];
        if x.len() == 0 || ax_len <= 1 {
            return Ok(());
        }

        if !x.is_c_contiguous() || x.as_slice_mut().is_none() {
            *x = x.to_c_contiguous();
        }
        let shape = x.shape().to_vec();
        let outer: usize = shape[..ax].iter().product();
        let inner: usize = shape[(ax + 1)..].iter().product();
        let data = x.as_slice_mut().ok_or_else(|| {
            NumPyError::InvalidValue("array buffer is not uniquely mutable".into())
        })?;

        for i in (1..ax_len).rev() {
            let j = random_interval(&mut self.bit_generator, i as u64) as usize;
            if i == j {
                continue;
            }
            for o in 0..outer {
                let base = o * ax_len * inner;
                let lo = j.min(i);
                let hi = j.max(i);
                let (left, right) = data.split_at_mut(base + hi * inner);
                let a = &mut left[(base + lo * inner)..(base + (lo + 1) * inner)];
                let b = &mut right[..inner];
                a.swap_with_slice(b);
            }
        }
        Ok(())
    }

    /// Return a shuffled copy of `0..x` (NumPy `Generator.permutation(n)`).
    pub fn permutation_range(&mut self, x: usize) -> Result<NdArray<i64>> {
        let mut arr = alloc_and_fill(&[x], |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(i as i64);
            }
        })?;
        self.shuffle(&mut arr, 0)?;
        Ok(arr)
    }

    /// Return a new array with `x` shuffled along `axis` (NumPy `Generator.permutation(arr, axis=axis)`).
    pub fn permutation<T: Scalar>(&mut self, x: &NdArray<T>, axis: usize) -> Result<NdArray<T>> {
        if x.ndim() == 0 {
            return Err(NumPyError::InvalidAxis {
                axis: axis as isize,
                ndim: 0,
            });
        }
        let mut out = x.deep_copy(Order::C);
        self.shuffle(&mut out, axis)?;
        Ok(out)
    }

    /// Shuffle each 1D slice along `axis` independently (or flattened if `axis` is `None`),
    /// returning a new array (NumPy `Generator.permuted`).
    pub fn permuted<T: Scalar>(&mut self, x: &NdArray<T>, axis: Option<usize>) -> Result<NdArray<T>> {
        if x.ndim() == 0 {
            return Err(NumPyError::InvalidAxis {
                axis: axis.unwrap_or(0) as isize,
                ndim: 0,
            });
        }
        let mut out = x.to_c_contiguous().deep_copy(Order::C);
        match axis {
            None => {
                let n = out.len();
                if n <= 1 {
                    return Ok(out);
                }
                let slice = out.as_slice_mut().unwrap();
                for i in (1..n).rev() {
                    let j = random_interval(&mut self.bit_generator, i as u64) as usize;
                    slice.swap(i, j);
                }
            }
            Some(ax_in) => {
                if out.ndim() == 0 {
                    return Err(NumPyError::InvalidAxis {
                        axis: ax_in as isize,
                        ndim: 0,
                    });
                }
                let ax = normalize_axis(ax_in as isize, out.ndim())?;
                let shape = out.shape().to_vec();
                let ax_len = shape[ax];
                if ax_len <= 1 {
                    return Ok(out);
                }
                let outer: usize = shape[..ax].iter().product();
                let inner: usize = shape[(ax + 1)..].iter().product();
                let slice = out.as_slice_mut().unwrap();
                for o in 0..outer {
                    let base = o * ax_len * inner;
                    for k in 0..inner {
                        for i in (1..ax_len).rev() {
                            let j = random_interval(&mut self.bit_generator, i as u64) as usize;
                            if i != j {
                                slice.swap(base + i * inner + k, base + j * inner + k);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Parameter-broadcasting array variants for continuous & discrete ops
    // -----------------------------------------------------------------------

    /// Draw `normal` samples with array parameters `loc` and `scale` broadcasted cleanly.
    pub fn normal_array(
        &mut self,
        loc: &NdArray<f64>,
        scale: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<f64>> {
        for s in scale.iter() {
            validate_non_negative("scale", s)?;
        }
        let bshape = broadcast_shapes(loc.shape(), scale.shape())?;
        let out_shape = check_broadcast_output_shape(&bshape, size)?;
        let loc_b = loc.broadcast_to(&out_shape)?;
        let scale_b = scale.broadcast_to(&out_shape)?;
        let loc_vals: Vec<f64> = loc_b.iter().collect();
        let scale_vals: Vec<f64> = scale_b.iter().collect();
        let bg = &mut self.bit_generator;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_normal(bg, loc_vals[i], scale_vals[i]));
            }
        })
    }

    /// Draw `uniform` samples with array parameters `low` and `high`.
    pub fn uniform_array(
        &mut self,
        low: &NdArray<f64>,
        high: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<f64>> {
        let bshape = broadcast_shapes(low.shape(), high.shape())?;
        let out_shape = check_broadcast_output_shape(&bshape, size)?;
        let low_b = low.broadcast_to(&out_shape)?;
        let high_b = high.broadcast_to(&out_shape)?;
        let low_vals: Vec<f64> = low_b.iter().collect();
        let high_vals: Vec<f64> = high_b.iter().collect();
        for (&l, &h) in low_vals.iter().zip(high_vals.iter()) {
            validate_finite("low", l)?;
            validate_finite("high", h)?;
            let range = h - l;
            if !range.is_finite() {
                return Err(NumPyError::InvalidValue("Range exceeds finite bounds".into()));
            }
            validate_non_negative("high - low", range)?;
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_uniform(bg, low_vals[i], high_vals[i] - low_vals[i]));
            }
        })
    }

    /// Draw `gamma` samples with broadcasted `shape_param` and `scale` arrays.
    pub fn gamma_array(
        &mut self,
        shape_param: &NdArray<f64>,
        scale: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<f64>> {
        for a in shape_param.iter() {
            validate_non_negative("shape", a)?;
        }
        for s in scale.iter() {
            validate_non_negative("scale", s)?;
        }
        let bshape = broadcast_shapes(shape_param.shape(), scale.shape())?;
        let out_shape = check_broadcast_output_shape(&bshape, size)?;
        let a_b: Vec<f64> = shape_param.broadcast_to(&out_shape)?.iter().collect();
        let s_b: Vec<f64> = scale.broadcast_to(&out_shape)?.iter().collect();
        let bg = &mut self.bit_generator;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_gamma(bg, a_b[i], s_b[i]));
            }
        })
    }

    /// Draw `beta` samples with broadcasted `a` and `b` arrays.
    pub fn beta_array(
        &mut self,
        a: &NdArray<f64>,
        b: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<f64>> {
        for av in a.iter() {
            validate_positive("a", av)?;
        }
        for bv in b.iter() {
            validate_positive("b", bv)?;
        }
        let bshape = broadcast_shapes(a.shape(), b.shape())?;
        let out_shape = check_broadcast_output_shape(&bshape, size)?;
        let a_b: Vec<f64> = a.broadcast_to(&out_shape)?.iter().collect();
        let b_b: Vec<f64> = b.broadcast_to(&out_shape)?.iter().collect();
        let bg = &mut self.bit_generator;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_beta(bg, a_b[i], b_b[i]));
            }
        })
    }

    /// Draw `binomial` samples with broadcasted `n` and `p` arrays.
    pub fn binomial_array(
        &mut self,
        n: &NdArray<i64>,
        p: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<i64>> {
        for nv in n.iter() {
            if nv < 0 {
                return Err(NumPyError::InvalidValue("n < 0".into()));
            }
        }
        for pv in p.iter() {
            validate_prob_0_1("p", pv)?;
        }
        let bshape = broadcast_shapes(n.shape(), p.shape())?;
        let out_shape = check_broadcast_output_shape(&bshape, size)?;
        let n_b: Vec<i64> = n.broadcast_to(&out_shape)?.iter().collect();
        let p_b: Vec<f64> = p.broadcast_to(&out_shape)?.iter().collect();
        let bg = &mut self.bit_generator;
        let bin = &mut self.binomial;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_binomial(bg, n_b[i], p_b[i], bin));
            }
        })
    }

    /// Draw `poisson` samples for an array of `lam` rates.
    pub fn poisson_array(
        &mut self,
        lam: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<i64>> {
        for lv in lam.iter() {
            if lv.is_nan() || lv < 0.0 || lv > POISSON_LAM_MAX {
                return Err(NumPyError::InvalidValue(
                    "lam value too large or negative".into(),
                ));
            }
        }
        let out_shape = check_broadcast_output_shape(lam.shape(), size)?;
        let l_b: Vec<f64> = lam.broadcast_to(&out_shape)?.iter().collect();
        let bg = &mut self.bit_generator;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_poisson(bg, l_b[i]));
            }
        })
    }

    /// Draw `triangular` samples with broadcasted `left`, `mode`, `right` arrays.
    pub fn triangular_array(
        &mut self,
        left: &NdArray<f64>,
        mode: &NdArray<f64>,
        right: &NdArray<f64>,
        size: Option<&[usize]>,
    ) -> Result<NdArray<f64>> {
        let bshape = broadcast_many_shapes(&[left.shape(), mode.shape(), right.shape()])?;
        let out_shape = check_broadcast_output_shape(&bshape, size)?;
        let l_b: Vec<f64> = left.broadcast_to(&out_shape)?.iter().collect();
        let m_b: Vec<f64> = mode.broadcast_to(&out_shape)?.iter().collect();
        let r_b: Vec<f64> = right.broadcast_to(&out_shape)?.iter().collect();
        for ((&lv, &mv), &rv) in l_b.iter().zip(m_b.iter()).zip(r_b.iter()) {
            validate_finite("left", lv)?;
            validate_finite("mode", mv)?;
            validate_finite("right", rv)?;
            if lv > mv {
                return Err(NumPyError::InvalidValue("left > mode".into()));
            }
            if mv > rv {
                return Err(NumPyError::InvalidValue("mode > right".into()));
            }
            if lv == rv {
                return Err(NumPyError::InvalidValue("left == right".into()));
            }
        }
        let bg = &mut self.bit_generator;
        alloc_and_fill(&out_shape, |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(random_triangular(bg, l_b[i], m_b[i], r_b[i]));
            }
        })
    }

    // -----------------------------------------------------------------------
    // Parallel stream generation via SeedSequence::spawn + Rayon + AlignedUninit
    // -----------------------------------------------------------------------

    /// Generate uniform `f64` samples in `[0, 1)` across parallel spawned streams when large.
    pub fn par_random(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(bg.next_double());
                }
            },
        )
    }

    /// Generate uniform `f32` samples in `[0, 1)` across parallel spawned streams when large.
    pub fn par_random_f32(&mut self, shape: &[usize]) -> Result<NdArray<f32>> {
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(bg.next_float());
                }
            },
        )
    }

    /// Generate standard normal (`N(0, 1)`) `f64` samples across parallel spawned streams when large.
    pub fn par_standard_normal(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_standard_normal(bg));
                }
            },
        )
    }

    /// Generate standard normal (`N(0, 1)`) `f32` samples across parallel spawned streams when large.
    pub fn par_standard_normal_f32(&mut self, shape: &[usize]) -> Result<NdArray<f32>> {
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_standard_normal_f32(bg));
                }
            },
        )
    }

    /// Generate standard exponential `f64` samples across parallel spawned streams when large.
    pub fn par_standard_exponential(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_standard_exponential(bg));
                }
            },
        )
    }

    /// Generate standard exponential `f32` samples across parallel spawned streams when large.
    pub fn par_standard_exponential_f32(&mut self, shape: &[usize]) -> Result<NdArray<f32>> {
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_standard_exponential_f32(bg));
                }
            },
        )
    }

    /// Generate normal (`N(loc, scale^2)`) `f64` samples across parallel spawned streams when large.
    pub fn par_normal(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_normal(bg, loc, scale));
                }
            },
        )
    }

    /// Generate uniform `[low, high)` `f64` samples across parallel spawned streams when large.
    pub fn par_uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_finite("low", low)?;
        validate_finite("high", high)?;
        let range = high - low;
        if !range.is_finite() {
            return Err(NumPyError::InvalidValue("Range exceeds finite bounds".into()));
        }
        validate_non_negative("high - low", range)?;
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_uniform(bg, low, range));
                }
            },
        )
    }

    /// Generate exponential `f64` samples across parallel spawned streams when large.
    pub fn par_exponential(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        validate_non_negative("scale", scale)?;
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_exponential(bg, scale));
                }
            },
        )
    }

    /// Generate gamma `f64` samples across parallel spawned streams when large.
    pub fn par_gamma(
        &mut self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        validate_non_negative("shape", shape_param)?;
        validate_non_negative("scale", scale)?;
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                for slot in slice {
                    slot.write(random_gamma(bg, shape_param, scale));
                }
            },
        )
    }

    /// Generate unbiased random `i64` integers `[low, high)` across parallel spawned streams when large.
    pub fn par_integers(&mut self, low: i64, high: i64, shape: &[usize]) -> Result<NdArray<i64>> {
        if low >= high {
            return Err(NumPyError::InvalidValue(format!(
                "low ({low}) >= high ({high})"
            )));
        }
        let span = (high as i128 - low as i128 - 1) as u128;
        if span > u64::MAX as u128 {
            return Err(NumPyError::InvalidValue(
                "integer range exceeds 64-bit unsigned range".into(),
            ));
        }
        let span64 = span as u64;
        alloc_and_par_fill(
            &mut self.bit_generator,
            shape,
            DEFAULT_PARALLEL_THRESHOLD,
            |bg, slice| {
                fill_unsigned_range(bg, span64, slice, |u| (low as i128 + u as i128) as i64);
            },
        )
    }
}

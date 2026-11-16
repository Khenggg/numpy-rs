use crate::bit_generator::BitGenerator;
use crate::distributions::{
    bounded_masked_u32, bounded_masked_u64, compute_cov_factor, gen_mask_u64, kahan_sum,
    legacy_beta, legacy_binomial, legacy_chisquare, legacy_double, legacy_gauss, legacy_geometric,
    legacy_gumbel, legacy_hypergeometric, legacy_laplace, legacy_logistic, legacy_logseries,
    legacy_multinomial, legacy_negative_binomial, legacy_noncentral_chisquare, legacy_pareto,
    legacy_power, legacy_rayleigh, legacy_standard_exponential, legacy_standard_gamma,
    legacy_standard_t, legacy_vonmises, legacy_weibull, random_interval, random_poisson,
    random_triangular, random_zipf, BinomialState, CheckValid, MvNormalMethod, HYPERGEOM_MAX,
    POISSON_LAM_MAX,
};
use crate::generator::alloc_and_fill;
use crate::mt19937::{Mt19937State, MT19937};
use crate::seed_sequence::SeedSequence;
use numpy_types::{NdArray, NumPyError, Result, Scalar};
use std::collections::HashSet;

/// Snapshot of legacy [`RandomState`] internal state (equivalent to `RandomState.get_state()`).
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyState {
    pub mt: Mt19937State,
    pub has_gauss: bool,
    pub gauss: f64,
}

/// Legacy NumPy `RandomState` container backed by [`MT19937`] and legacy distributions
/// (Box-Muller Gaussian, `-ln(1 - U)` exponential, masked rejection `randint`).
#[derive(Debug, Clone)]
pub struct RandomState {
    mt: MT19937,
    has_gauss: bool,
    gauss: f64,
    binomial: BinomialState,
}

impl RandomState {
    /// Construct a `RandomState` seeded from OS entropy.
    pub fn new() -> Self {
        let ss = SeedSequence::new();
        let mt = MT19937::from_seed_seq(ss).expect("OS SeedSequence is valid");
        Self {
            mt,
            has_gauss: false,
            gauss: 0.0,
            binomial: BinomialState::default(),
        }
    }

    /// Construct a `RandomState` using NumPy's legacy 32-bit scalar seeding (`np.random.RandomState(seed)`).
    pub fn from_seed(seed: u32) -> Self {
        Self {
            mt: MT19937::from_legacy_seed(seed),
            has_gauss: false,
            gauss: 0.0,
            binomial: BinomialState::default(),
        }
    }

    /// Construct a `RandomState` using legacy key-array seeding (`init_by_array`).
    pub fn from_seed_slice(key: &[u32]) -> Self {
        Self {
            mt: MT19937::from_legacy_key(key),
            has_gauss: false,
            gauss: 0.0,
            binomial: BinomialState::default(),
        }
    }

    /// Construct a `RandomState` backed by an explicit [`SeedSequence`].
    pub fn from_seed_seq(seed_seq: SeedSequence) -> Result<Self> {
        Ok(Self {
            mt: MT19937::from_seed_seq(seed_seq)?,
            has_gauss: false,
            gauss: 0.0,
            binomial: BinomialState::default(),
        })
    }

    /// Wrap an existing [`MT19937`] instance.
    pub fn from_mt19937(mt: MT19937) -> Self {
        Self {
            mt,
            has_gauss: false,
            gauss: 0.0,
            binomial: BinomialState::default(),
        }
    }

    /// Re-seed this `RandomState` in place using a scalar 32-bit seed.
    pub fn seed(&mut self, seed: u32) {
        self.mt.legacy_seed(seed);
        self.has_gauss = false;
        self.gauss = 0.0;
        self.binomial.reset();
    }

    /// Re-seed this `RandomState` in place using an array of 32-bit seed words.
    pub fn seed_slice(&mut self, key: &[u32]) {
        self.mt.legacy_seed_array(key);
        self.has_gauss = false;
        self.gauss = 0.0;
        self.binomial.reset();
    }

    /// Return a snapshot of the internal state (`get_state`).
    pub fn get_state(&self) -> LegacyState {
        LegacyState {
            mt: self.mt.state(),
            has_gauss: self.has_gauss,
            gauss: self.gauss,
        }
    }

    /// Restore the internal state (`set_state`).
    pub fn set_state(&mut self, state: LegacyState) -> Result<()> {
        self.mt.set_state(state.mt)?;
        self.has_gauss = state.has_gauss;
        self.gauss = state.gauss;
        self.binomial.reset();
        Ok(())
    }

    /// Borrow underlying [`MT19937`] BitGenerator.
    pub fn bit_generator(&self) -> &MT19937 {
        &self.mt
    }

    /// Mutably borrow underlying [`MT19937`] BitGenerator.
    pub fn bit_generator_mut(&mut self) -> &mut MT19937 {
        self.has_gauss = false;
        self.gauss = 0.0;
        self.binomial.reset();
        &mut self.mt
    }

    #[inline]
    fn check_non_neg(name: &str, v: f64) -> Result<()> {
        if v.is_nan() || v < 0.0 {
            return Err(NumPyError::InvalidValue(format!("{name} < 0")));
        }
        Ok(())
    }

    #[inline]
    fn check_pos(name: &str, v: f64) -> Result<()> {
        if v.is_nan() || v <= 0.0 {
            return Err(NumPyError::InvalidValue(format!("{name} <= 0")));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Legacy core methods: rand, randn, randint, random_sample, tomaxint, bytes
    // -----------------------------------------------------------------------

    /// Uniform floats in `[0.0, 1.0)` matching `RandomState.random_sample`.
    pub fn random_sample(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_double(mt));
            }
        })
    }

    /// Alias of [`Self::random_sample`] (`RandomState.random`).
    pub fn random(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        self.random_sample(shape)
    }

    /// Alias of [`Self::random_sample`] (`RandomState.ranf`).
    pub fn ranf(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        self.random_sample(shape)
    }

    /// Alias of [`Self::random_sample`] (`RandomState.sample`).
    pub fn sample(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        self.random_sample(shape)
    }

    /// Uniform floats in `[0.0, 1.0)` with explicit `dims` shape (`RandomState.rand`).
    pub fn rand(&mut self, dims: &[usize]) -> Result<NdArray<f64>> {
        self.random_sample(dims)
    }

    /// Legacy Box-Muller `N(0, 1)` samples (`RandomState.randn`).
    pub fn randn(&mut self, dims: &[usize]) -> Result<NdArray<f64>> {
        self.standard_normal(dims)
    }

    /// Legacy masked rejection integers in `[low, high)` (or `[0, low)` when `high` is `None`).
    pub fn randint(
        &mut self,
        low: i64,
        high: Option<i64>,
        shape: &[usize],
    ) -> Result<NdArray<i64>> {
        let (lo, hi) = match high {
            Some(h) => (low, h),
            None => (0, low),
        };
        if lo >= hi {
            return Err(NumPyError::InvalidValue("low >= high".into()));
        }
        let span = (hi as i128 - lo as i128 - 1) as u128;
        if span > u64::MAX as u128 {
            return Err(NumPyError::InvalidValue(
                "high is out of bounds for i64".into(),
            ));
        }
        let rng_max = span as u64;
        let mask = gen_mask_u64(rng_max);
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            if rng_max == 0 {
                for slot in slice {
                    slot.write(lo);
                }
            } else if rng_max <= u32::MAX as u64 {
                let rng32 = rng_max as u32;
                let mask32 = mask as u32;
                for slot in slice {
                    let v = bounded_masked_u32(mt, rng32, mask32) as i64;
                    slot.write(lo + v);
                }
            } else {
                for slot in slice {
                    let v = bounded_masked_u64(mt, rng_max, mask);
                    slot.write((lo as i128 + v as i128) as i64);
                }
            }
        })
    }

    /// Legacy inclusive integers in `[low, high]` (or `[1, low]` when `high` is `None`).
    pub fn random_integers(
        &mut self,
        low: i64,
        high: Option<i64>,
        shape: &[usize],
    ) -> Result<NdArray<i64>> {
        let (lo, hi_excl) = match high {
            Some(h) => {
                let he = h.checked_add(1).ok_or_else(|| {
                    NumPyError::InvalidValue("high + 1 overflows i64".into())
                })?;
                (low, he)
            }
            None => {
                let he = low.checked_add(1).ok_or_else(|| {
                    NumPyError::InvalidValue("low + 1 overflows i64".into())
                })?;
                (1, he)
            }
        };
        self.randint(lo, Some(hi_excl), shape)
    }

    /// Legacy positive random integers matching `RandomState.tomaxint`
    /// (`[0, i32::MAX]` on Windows/32-bit where C `long` is 32-bit, `[0, i64::MAX]` on 64-bit Unix).
    pub fn tomaxint(&mut self, shape: &[usize]) -> Result<NdArray<i64>> {
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                #[cfg(any(windows, target_pointer_width = "32"))]
                let v = (mt.next_u32() >> 1) as i64;
                #[cfg(not(any(windows, target_pointer_width = "32")))]
                let v = (mt.next_u64() >> 1) as i64;
                slot.write(v);
            }
        })
    }

    /// Legacy random bytes (`RandomState.bytes`).
    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        let n_u32 = length.div_ceil(4);
        let mut out = Vec::with_capacity(n_u32 * 4);
        for _ in 0..n_u32 {
            out.extend_from_slice(&self.mt.next_u32().to_le_bytes());
        }
        out.truncate(length);
        out
    }

    // -----------------------------------------------------------------------
    // Legacy continuous & discrete distributions
    // -----------------------------------------------------------------------

    pub fn standard_normal(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_gauss(mt, has_gauss, gauss));
            }
        })
    }

    pub fn normal(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(loc + scale * legacy_gauss(mt, has_gauss, gauss));
            }
        })
    }

    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        if !low.is_finite() || !high.is_finite() || !(high - low).is_finite() {
            return Err(NumPyError::InvalidValue("Range exceeds finite bounds".into()));
        }
        let range = high - low;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(low + range * legacy_double(mt));
            }
        })
    }

    pub fn standard_exponential(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_standard_exponential(mt));
            }
        })
    }

    pub fn exponential(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(scale * legacy_standard_exponential(mt));
            }
        })
    }

    pub fn standard_gamma(&mut self, shape_param: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("shape", shape_param)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_standard_gamma(mt, shape_param, has_gauss, gauss));
            }
        })
    }

    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("shape", shape_param)?;
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(scale * legacy_standard_gamma(mt, shape_param, has_gauss, gauss));
            }
        })
    }

    pub fn beta(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("a", a)?;
        Self::check_pos("b", b)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_beta(mt, a, b, has_gauss, gauss));
            }
        })
    }

    pub fn chisquare(&mut self, df: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("df", df)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_chisquare(mt, df, has_gauss, gauss));
            }
        })
    }

    pub fn noncentral_chisquare(
        &mut self,
        df: f64,
        nonc: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        Self::check_pos("df", df)?;
        Self::check_non_neg("nonc", nonc)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_noncentral_chisquare(
                    mt, df, nonc, has_gauss, gauss,
                ));
            }
        })
    }

    pub fn f(&mut self, dfnum: f64, dfden: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("dfnum", dfnum)?;
        Self::check_pos("dfden", dfden)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                let c1 = legacy_chisquare(mt, dfnum, has_gauss, gauss);
                let c2 = legacy_chisquare(mt, dfden, has_gauss, gauss);
                slot.write((c1 * dfden) / (c2 * dfnum));
            }
        })
    }

    pub fn noncentral_f(
        &mut self,
        dfnum: f64,
        dfden: f64,
        nonc: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        Self::check_pos("dfnum", dfnum)?;
        Self::check_pos("dfden", dfden)?;
        Self::check_non_neg("nonc", nonc)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                let t = legacy_noncentral_chisquare(mt, dfnum, nonc, has_gauss, gauss) * dfden;
                let c2 = legacy_chisquare(mt, dfden, has_gauss, gauss);
                slot.write(t / (c2 * dfnum));
            }
        })
    }

    pub fn standard_t(&mut self, df: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("df", df)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_standard_t(mt, df, has_gauss, gauss));
            }
        })
    }

    pub fn standard_cauchy(&mut self, shape: &[usize]) -> Result<NdArray<f64>> {
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                let n1 = legacy_gauss(mt, has_gauss, gauss);
                let n2 = legacy_gauss(mt, has_gauss, gauss);
                slot.write(n1 / n2);
            }
        })
    }

    pub fn laplace(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_laplace(mt, loc, scale));
            }
        })
    }

    pub fn logistic(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_logistic(mt, loc, scale));
            }
        })
    }

    pub fn gumbel(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_gumbel(mt, loc, scale));
            }
        })
    }

    pub fn lognormal(&mut self, mean: f64, sigma: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("sigma", sigma)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write((mean + sigma * legacy_gauss(mt, has_gauss, gauss)).exp());
            }
        })
    }

    pub fn rayleigh(&mut self, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("scale", scale)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_rayleigh(mt, scale));
            }
        })
    }

    pub fn weibull(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_non_neg("a", a)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_weibull(mt, a));
            }
        })
    }

    pub fn pareto(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("a", a)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_pareto(mt, a));
            }
        })
    }

    pub fn power(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("a", a)?;
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_power(mt, a));
            }
        })
    }

    pub fn triangular(
        &mut self,
        left: f64,
        mode: f64,
        right: f64,
        shape: &[usize],
    ) -> Result<NdArray<f64>> {
        if !left.is_finite() || !mode.is_finite() || !right.is_finite() {
            return Err(NumPyError::InvalidValue(
                "triangular bounds must be finite".into(),
            ));
        }
        if left > mode || mode > right || left == right {
            return Err(NumPyError::InvalidValue(
                "triangular requires left <= mode <= right and left < right".into(),
            ));
        }
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_triangular(mt, left, mode, right));
            }
        })
    }

    pub fn vonmises(&mut self, mu: f64, kappa: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        if kappa < 0.0 {
            return Err(NumPyError::InvalidValue("kappa < 0".into()));
        }
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_vonmises(mt, mu, kappa));
            }
        })
    }

    pub fn wald(&mut self, mean: f64, scale: f64, shape: &[usize]) -> Result<NdArray<f64>> {
        Self::check_pos("mean", mean)?;
        Self::check_pos("scale", scale)?;
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                let mut y = legacy_gauss(mt, has_gauss, gauss);
                y = mean * y * y;
                let d = 1.0 + (1.0 + 4.0 * scale / y).sqrt();
                let x = mean * (1.0 - 2.0 / d);
                let u = legacy_double(mt);
                let val = if u <= mean / (mean + x) {
                    x
                } else {
                    mean * mean / x
                };
                slot.write(val);
            }
        })
    }

    pub fn binomial(&mut self, n: i64, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if n < 0 {
            return Err(NumPyError::InvalidValue("n < 0".into()));
        }
        if p.is_nan() || p < 0.0 || p > 1.0 {
            return Err(NumPyError::InvalidValue(
                "p < 0, p > 1 or p is NaN".into(),
            ));
        }
        let mt = &mut self.mt;
        let bin = &mut self.binomial;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_binomial(mt, n, p, bin));
            }
        })
    }

    pub fn negative_binomial(&mut self, n: f64, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        Self::check_pos("n", n)?;
        if p.is_nan() || p <= 0.0 || p > 1.0 {
            return Err(NumPyError::InvalidValue(
                "p <= 0, p > 1 or p is NaN".into(),
            ));
        }
        let max_lam = ((1.0 - p) / p) * (n + 10.0 * n.sqrt());
        if max_lam > POISSON_LAM_MAX {
            return Err(NumPyError::InvalidValue(
                "n too large or p too small".into(),
            ));
        }
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_negative_binomial(mt, n, p, has_gauss, gauss));
            }
        })
    }

    pub fn poisson(&mut self, lam: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if lam.is_nan() || lam < 0.0 || lam > POISSON_LAM_MAX {
            return Err(NumPyError::InvalidValue(
                "lam value too large or negative".into(),
            ));
        }
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_poisson(mt, lam));
            }
        })
    }

    pub fn geometric(&mut self, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if p.is_nan() || p <= 0.0 || p > 1.0 {
            return Err(NumPyError::InvalidValue(
                "p <= 0, p > 1 or p is NaN".into(),
            ));
        }
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_geometric(mt, p));
            }
        })
    }

    pub fn hypergeometric(
        &mut self,
        ngood: i64,
        nbad: i64,
        nsample: i64,
        shape: &[usize],
    ) -> Result<NdArray<i64>> {
        if ngood < 0 || nbad < 0 {
            return Err(NumPyError::InvalidValue(
                "ngood and nbad must be non-negative".into(),
            ));
        }
        if nsample < 1 {
            return Err(NumPyError::InvalidValue("nsample < 1".into()));
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
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_hypergeometric(mt, ngood, nbad, nsample));
            }
        })
    }

    pub fn logseries(&mut self, p: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if p.is_nan() || p < 0.0 || p >= 1.0 {
            return Err(NumPyError::InvalidValue(
                "p < 0, p >= 1 or p is NaN".into(),
            ));
        }
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(legacy_logseries(mt, p));
            }
        })
    }

    pub fn zipf(&mut self, a: f64, shape: &[usize]) -> Result<NdArray<i64>> {
        if a.is_nan() || a <= 1.0 {
            return Err(NumPyError::InvalidValue("a <= 1".into()));
        }
        let mt = &mut self.mt;
        alloc_and_fill(shape, |slice| {
            for slot in slice {
                slot.write(random_zipf(mt, a));
            }
        })
    }

    pub fn dirichlet(&mut self, alpha: &[f64], size: Option<&[usize]>) -> Result<NdArray<f64>> {
        for &a in alpha {
            Self::check_pos("alpha", a)?;
        }
        let k = alpha.len();
        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(k);
        if k == 0 {
            return NdArray::zeros(&out_shape);
        }
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        alloc_and_fill(&out_shape, |slice| {
            for chunk in slice.chunks_exact_mut(k) {
                let mut acc = 0.0;
                for j in 0..k {
                    let g = legacy_standard_gamma(mt, alpha[j], has_gauss, gauss);
                    acc += g;
                    chunk[j].write(g);
                }
                let inv = 1.0 / acc;
                for slot in chunk {
                    unsafe {
                        let v = *slot.assume_init_ref();
                        slot.write(v * inv);
                    }
                }
            }
        })
    }

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
            if p.is_nan() || p < 0.0 || p > 1.0 {
                return Err(NumPyError::InvalidValue("invalid pval".into()));
            }
        }
        if kahan_sum(&pvals[..(d - 1)]) > 1.0 + 1e-12 {
            return Err(NumPyError::InvalidValue("sum(pvals[:-1]) > 1.0".into()));
        }
        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(d);
        let mt = &mut self.mt;
        let bin = &mut self.binomial;
        alloc_and_fill(&out_shape, |slice| {
            let mut row = vec![0i64; d];
            for chunk in slice.chunks_exact_mut(d) {
                legacy_multinomial(mt, n, &mut row, pvals, bin);
                for (slot, &val) in chunk.iter_mut().zip(row.iter()) {
                    slot.write(val);
                }
            }
        })
    }

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
        let cov_slice = cov_contig.as_slice().unwrap();
        let factor = compute_cov_factor(d, cov_slice, method, check_valid, tol)?;
        let mut out_shape = size.map(|s| s.to_vec()).unwrap_or_default();
        out_shape.push(d);
        let mt = &mut self.mt;
        let has_gauss = &mut self.has_gauss;
        let gauss = &mut self.gauss;
        let mut z_row = vec![0.0f64; d];
        alloc_and_fill(&out_shape, |slice| {
            for chunk in slice.chunks_exact_mut(d) {
                for z in &mut z_row {
                    *z = legacy_gauss(mt, has_gauss, gauss);
                }
                for c in 0..d {
                    let mut sum = mean[c];
                    let off = c * d;
                    for k in 0..d {
                        sum += z_row[k] * factor[off + k];
                    }
                    chunk[c].write(sum);
                }
            }
        })
    }

    /// Legacy `RandomState.choice` over `0..pop_size`.
    pub fn choice_range(
        &mut self,
        pop_size: usize,
        size: &[usize],
        replace: bool,
        p: Option<&[f64]>,
    ) -> Result<NdArray<i64>> {
        let out_total = numpy_types::shape_size(size)?;
        if pop_size == 0 {
            if out_total == 0 && p.is_none() {
                return alloc_and_fill(size, |_| {});
            }
            return Err(NumPyError::InvalidValue("a must be > 0".into()));
        }
        if let Some(probs) = p {
            if probs.len() != pop_size {
                return Err(NumPyError::ShapeMismatch(
                    "a and p must have same size".into(),
                ));
            }
            let p_sum = kahan_sum(probs);
            if p_sum.is_nan() || (p_sum - 1.0).abs() > 1.490_116_119_384_765_6e-8 {
                return Err(NumPyError::InvalidValue(
                    "probabilities do not sum to 1".into(),
                ));
            }
            for &pv in probs {
                if pv < 0.0 {
                    return Err(NumPyError::InvalidValue(
                        "probabilities must be non-negative".into(),
                    ));
                }
            }
            if replace {
                let mut cdf = Vec::with_capacity(pop_size);
                let mut run = 0.0;
                for &pv in probs {
                    run += pv;
                    cdf.push(run);
                }
                let last = *cdf.last().unwrap_or(&1.0);
                for c in &mut cdf {
                    *c /= last;
                }
                let u_arr = self.random_sample(size)?;
                let u_slice = u_arr.as_slice().unwrap();
                return alloc_and_fill(size, |slice| {
                    for (slot, &u) in slice.iter_mut().zip(u_slice.iter()) {
                        slot.write(cdf.partition_point(|&v| v <= u).min(pop_size - 1) as i64);
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
                    let u_arr = self.random_sample(&[rem])?;
                    let mut seen = HashSet::new();
                    for &u in u_arr.as_slice().unwrap() {
                        let idx = cdf.partition_point(|&v| v <= u).min(pop_size - 1);
                        if seen.insert(idx) && p_work[idx] > 0.0 {
                            found.push(idx as i64);
                            if found.len() == out_total {
                                break;
                            }
                        }
                    }
                    for &idx in &seen {
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
            self.randint(0, Some(pop_size as i64), size)
        } else {
            if out_total > pop_size {
                return Err(NumPyError::InvalidValue(
                    "Cannot take a larger sample than population when replace is False".into(),
                ));
            }
            let mut perm = self.permutation_range(pop_size)?;
            let tail = &perm.as_slice_mut().unwrap()[..out_total];
            alloc_and_fill(size, |slice| {
                for (slot, &v) in slice.iter_mut().zip(tail.iter()) {
                    slot.write(v);
                }
            })
        }
    }

    /// Legacy `RandomState.choice` sampling from 1D `NdArray<T>`.
    pub fn choice<T: Scalar>(
        &mut self,
        a: &NdArray<T>,
        size: &[usize],
        replace: bool,
        p: Option<&[f64]>,
    ) -> Result<NdArray<T>> {
        if a.ndim() != 1 {
            return Err(NumPyError::ShapeMismatch(
                "a must be 1-dimensional in legacy RandomState.choice".into(),
            ));
        }
        let idx = self.choice_range(a.len(), size, replace, p)?;
        let a_contig = a.to_c_contiguous();
        let a_slice = a_contig.as_slice().unwrap();
        let idx_slice = idx.as_slice().unwrap();
        alloc_and_fill(size, |slice| {
            for (slot, &i) in slice.iter_mut().zip(idx_slice.iter()) {
                slot.write(a_slice[i as usize]);
            }
        })
    }

    /// Legacy `RandomState.shuffle` (in-place Fisher-Yates along axis 0).
    pub fn shuffle<T: Scalar>(&mut self, x: &mut NdArray<T>) -> Result<()> {
        if x.ndim() == 0 {
            return Err(NumPyError::InvalidAxis { axis: 0, ndim: 0 });
        }
        let n = x.shape()[0];
        if x.len() == 0 || n <= 1 {
            return Ok(());
        }
        if !x.is_c_contiguous() || x.as_slice_mut().is_none() {
            *x = x.to_c_contiguous();
        }
        let inner: usize = x.shape()[1..].iter().product();
        let data = x.as_slice_mut().ok_or_else(|| {
            NumPyError::InvalidValue("array buffer is not uniquely mutable".into())
        })?;
        for i in (1..n).rev() {
            let j = random_interval(&mut self.mt, i as u64) as usize;
            if i == j {
                continue;
            }
            let lo = j.min(i);
            let hi = j.max(i);
            let (left, right) = data.split_at_mut(hi * inner);
            left[(lo * inner)..((lo + 1) * inner)].swap_with_slice(&mut right[..inner]);
        }
        Ok(())
    }

    /// Legacy `RandomState.permutation(n)`.
    pub fn permutation_range(&mut self, n: usize) -> Result<NdArray<i64>> {
        let mut arr = alloc_and_fill(&[n], |slice| {
            for (i, slot) in slice.iter_mut().enumerate() {
                slot.write(i as i64);
            }
        })?;
        self.shuffle(&mut arr)?;
        Ok(arr)
    }

    /// Legacy `RandomState.permutation(arr)` (copy & shuffle along axis 0).
    pub fn permutation<T: Scalar>(&mut self, x: &NdArray<T>) -> Result<NdArray<T>> {
        let mut out = x.deep_copy(numpy_types::Order::C);
        self.shuffle(&mut out)?;
        Ok(out)
    }
}

impl Default for RandomState {
    fn default() -> Self {
        Self::new()
    }
}

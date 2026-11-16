use num_complex::Complex;
use numpy_types::FloatElement;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftDirection, FftNum, FftPlanner};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, OnceLock, RwLock};

/// Normalization mode for discrete Fourier transforms, matching NumPy's `norm` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Norm {
    /// Forward transforms are unscaled; inverse transforms are scaled by `1 / n` (NumPy default).
    #[default]
    Backward,
    /// Both forward and inverse transforms are scaled by `1 / sqrt(n)` (unitary).
    Ortho,
    /// Forward transforms are scaled by `1 / n`; inverse transforms are unscaled.
    Forward,
}

impl Norm {
    /// Canonical NumPy string representation.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Backward => "backward",
            Self::Ortho => "ortho",
            Self::Forward => "forward",
        }
    }

    /// Compute the scalar multiplier for an FFT/IFFT with total transform product `total_n`.
    #[inline]
    pub fn scale_factor(self, is_forward: bool, total_n: f64) -> f64 {
        let effective = if is_forward { self } else { self.swapped() };
        match effective {
            Self::Backward => 1.0,
            Self::Ortho => 1.0 / total_n.sqrt(),
            Self::Forward => 1.0 / total_n,
        }
    }

    /// Swap normalization direction (`Backward` <-> `Forward`, `Ortho` unchanged).
    #[inline]
    pub const fn swapped(self) -> Self {
        match self {
            Self::Backward => Self::Forward,
            Self::Ortho => Self::Ortho,
            Self::Forward => Self::Backward,
        }
    }
}

impl fmt::Display for Norm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Norm {
    type Err = numpy_types::NumPyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "backward" => Ok(Self::Backward),
            "ortho" => Ok(Self::Ortho),
            "forward" => Ok(Self::Forward),
            _ => Err(numpy_types::NumPyError::InvalidValue(format!(
                "Invalid norm value '{s}'; should be \"backward\", \"ortho\" or \"forward\"."
            ))),
        }
    }
}

struct PlannerCache<T: FftNum> {
    fft_planner: FftPlanner<T>,
    real_planner: RealFftPlanner<T>,
    c2c_fwd: HashMap<usize, Arc<dyn Fft<T>>>,
    c2c_inv: HashMap<usize, Arc<dyn Fft<T>>>,
    r2c: HashMap<usize, Arc<dyn RealToComplex<T>>>,
    c2r: HashMap<usize, Arc<dyn ComplexToReal<T>>>,
}

impl<T: FftNum> PlannerCache<T> {
    fn new() -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            real_planner: RealFftPlanner::new(),
            c2c_fwd: HashMap::new(),
            c2c_inv: HashMap::new(),
            r2c: HashMap::new(),
            c2r: HashMap::new(),
        }
    }
}

fn get_c2c_cached<T: FftNum>(
    lock: &RwLock<PlannerCache<T>>,
    len: usize,
    direction: FftDirection,
) -> Arc<dyn Fft<T>> {
    {
        let guard = lock.read().unwrap_or_else(|e| e.into_inner());
        let map = match direction {
            FftDirection::Forward => &guard.c2c_fwd,
            FftDirection::Inverse => &guard.c2c_inv,
        };
        if let Some(plan) = map.get(&len) {
            return Arc::clone(plan);
        }
    }
    let mut guard = lock.write().unwrap_or_else(|e| e.into_inner());
    let plan = guard.fft_planner.plan_fft(len, direction);
    let map = match direction {
        FftDirection::Forward => &mut guard.c2c_fwd,
        FftDirection::Inverse => &mut guard.c2c_inv,
    };
    Arc::clone(map.entry(len).or_insert(plan))
}

fn get_r2c_cached<T: FftNum>(
    lock: &RwLock<PlannerCache<T>>,
    len: usize,
) -> Arc<dyn RealToComplex<T>> {
    {
        let guard = lock.read().unwrap_or_else(|e| e.into_inner());
        if let Some(plan) = guard.r2c.get(&len) {
            return Arc::clone(plan);
        }
    }
    let mut guard = lock.write().unwrap_or_else(|e| e.into_inner());
    if let Some(plan) = guard.r2c.get(&len) {
        return Arc::clone(plan);
    }
    let plan = guard.real_planner.plan_fft_forward(len);
    guard.r2c.insert(len, Arc::clone(&plan));
    plan
}

fn get_c2r_cached<T: FftNum>(
    lock: &RwLock<PlannerCache<T>>,
    len: usize,
) -> Arc<dyn ComplexToReal<T>> {
    {
        let guard = lock.read().unwrap_or_else(|e| e.into_inner());
        if let Some(plan) = guard.c2r.get(&len) {
            return Arc::clone(plan);
        }
    }
    let mut guard = lock.write().unwrap_or_else(|e| e.into_inner());
    if let Some(plan) = guard.c2r.get(&len) {
        return Arc::clone(plan);
    }
    let plan = guard.real_planner.plan_fft_inverse(len);
    guard.c2r.insert(len, Arc::clone(&plan));
    plan
}

static CACHE_F32: OnceLock<RwLock<PlannerCache<f32>>> = OnceLock::new();
static CACHE_F64: OnceLock<RwLock<PlannerCache<f64>>> = OnceLock::new();

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Real floating precision supported natively by `rustfft` and `realfft` (`f32` and `f64`).
pub trait FftFloat: sealed::Sealed + FftNum + FloatElement + 'static {
    fn from_f64_val(v: f64) -> Self;
    fn to_f64_val(self) -> f64;
    fn nan_val() -> Self;
    fn is_finite_val(self) -> bool;
    fn plan_c2c(len: usize, direction: FftDirection) -> Arc<dyn Fft<Self>>;
    fn plan_r2c(len: usize) -> Arc<dyn RealToComplex<Self>>;
    fn plan_c2r(len: usize) -> Arc<dyn ComplexToReal<Self>>;
    fn c2c_from_slice_val(c: Complex<Self>) -> Complex<Self> {
        c
    }
}

impl FftFloat for f32 {
    #[inline]
    fn from_f64_val(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64_val(self) -> f64 {
        self as f64
    }
    #[inline]
    fn nan_val() -> Self {
        f32::NAN
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn plan_c2c(len: usize, direction: FftDirection) -> Arc<dyn Fft<Self>> {
        let lock = CACHE_F32.get_or_init(|| RwLock::new(PlannerCache::new()));
        get_c2c_cached(lock, len, direction)
    }
    #[inline]
    fn plan_r2c(len: usize) -> Arc<dyn RealToComplex<Self>> {
        let lock = CACHE_F32.get_or_init(|| RwLock::new(PlannerCache::new()));
        get_r2c_cached(lock, len)
    }
    #[inline]
    fn plan_c2r(len: usize) -> Arc<dyn ComplexToReal<Self>> {
        let lock = CACHE_F32.get_or_init(|| RwLock::new(PlannerCache::new()));
        get_c2r_cached(lock, len)
    }
}

impl FftFloat for f64 {
    #[inline]
    fn from_f64_val(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64_val(self) -> f64 {
        self
    }
    #[inline]
    fn nan_val() -> Self {
        f64::NAN
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn plan_c2c(len: usize, direction: FftDirection) -> Arc<dyn Fft<Self>> {
        let lock = CACHE_F64.get_or_init(|| RwLock::new(PlannerCache::new()));
        get_c2c_cached(lock, len, direction)
    }
    #[inline]
    fn plan_r2c(len: usize) -> Arc<dyn RealToComplex<Self>> {
        let lock = CACHE_F64.get_or_init(|| RwLock::new(PlannerCache::new()));
        get_r2c_cached(lock, len)
    }
    #[inline]
    fn plan_c2r(len: usize) -> Arc<dyn ComplexToReal<Self>> {
        let lock = CACHE_F64.get_or_init(|| RwLock::new(PlannerCache::new()));
        get_c2r_cached(lock, len)
    }
}

use crate::{
    SortElement,
    lanes::{Lanes, invalid},
};
use numpy_types::{Complex32, Complex64, NdArray, Result, Scalar, f16};
use std::ops::{Add, Sub};

fn strict_axis<T>(a: &NdArray<T>, axis: Option<isize>) -> Result<()> {
    if let Some(axis) = axis {
        numpy_types::normalize_axis(axis, a.ndim())?;
    }
    Ok(())
}

#[doc(hidden)]
pub trait StatWork: Copy + Add<Output = Self> + Sub<Output = Self> {
    fn zero() -> Self;
    fn divide(self, n: f64) -> Self;
    fn scale(self, n: f64) -> Self;
    fn norm_squared(self) -> f64;
    fn is_nan(self) -> bool;
}
impl StatWork for f64 {
    fn zero() -> Self {
        0.0
    }
    fn divide(self, n: f64) -> Self {
        self / n
    }
    fn scale(self, n: f64) -> Self {
        self * n
    }
    fn norm_squared(self) -> f64 {
        self * self
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}
impl StatWork for Complex64 {
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn divide(self, n: f64) -> Self {
        Self::new(self.re / n, self.im / n)
    }
    fn scale(self, n: f64) -> Self {
        Self::new(self.re * n, self.im * n)
    }
    fn norm_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}
/// Floating/complex inputs retain their dtype; integer/bool statistics use f64.
/// Accumulators use f64 precision (complex128 for complex inputs).
pub trait StatisticsElement: SortElement {
    type Mean: Scalar;
    type Real: Scalar;
    type Work: StatWork;
    fn work(self) -> Self::Work;
    fn mean_value(value: Self::Work) -> Self::Mean;
}
macro_rules! real_stats {
    ($mean:ty; $($t:ty),*) => {$(impl StatisticsElement for $t {
        type Mean = $mean; type Real = $mean; type Work = f64;
        fn work(self) -> f64 { self.to_f64_lossy() }
        fn mean_value(value: f64) -> $mean { <$mean>::from_f64_lossy(value) }
    })*};
}
real_stats!(f64; bool, i8, i16, i32, i64, u8, u16, u32, u64, f64);
real_stats!(f32; f32);
real_stats!(f16; f16);
macro_rules! complex_stats {
    ($t:ty, $real:ty) => {
        impl StatisticsElement for $t {
            type Mean = $t;
            type Real = $real;
            type Work = Complex64;
            fn work(self) -> Complex64 {
                self.to_complex64()
            }
            fn mean_value(value: Complex64) -> $t {
                <$t>::from_complex64_lossy(value)
            }
        }
    };
}
complex_stats!(Complex32, f32);
complex_stats!(Complex64, f64);

fn pairwise<T: StatisticsElement>(xs: &[T], skip_nan: bool) -> (T::Work, usize) {
    if xs.len() > 128 {
        let mid = xs.len() / 2;
        let (a, n) = pairwise(&xs[..mid], skip_nan);
        let (b, m) = pairwise(&xs[mid..], skip_nan);
        (a + b, n + m)
    } else {
        let mut sum = T::Work::zero();
        let mut count = 0;
        for &x in xs {
            let x = x.work();
            if skip_nan && x.is_nan() {
                continue;
            }
            sum = sum + x;
            count += 1;
        }
        (sum, count)
    }
}
fn means<T: StatisticsElement>(p: &Lanes<'_, T>, skip_nan: bool) -> (Vec<T::Work>, Vec<usize>) {
    let mut sums = vec![T::Work::zero(); p.count()];
    let mut counts = vec![0usize; p.count()];
    if p.inner == 1 {
        for out in 0..p.count() {
            let start = out * p.width;
            (sums[out], counts[out]) = pairwise(&p.data[start..start + p.width], skip_nan);
        }
    } else {
        for outer in 0..p.outer {
            let sums = &mut sums[outer * p.inner..(outer + 1) * p.inner];
            let counts = &mut counts[outer * p.inner..(outer + 1) * p.inner];
            for k in 0..p.width {
                let base = (outer * p.width + k) * p.inner;
                for ((sum, count), &x) in sums
                    .iter_mut()
                    .zip(counts.iter_mut())
                    .zip(&p.data[base..base + p.inner])
                {
                    let x = x.work();
                    if skip_nan && x.is_nan() {
                        continue;
                    }
                    *sum = *sum + x;
                    *count += 1;
                }
            }
        }
    }
    for (sum, &n) in sums.iter_mut().zip(&counts) {
        *sum = sum.divide(n as f64);
    }
    (sums, counts)
}
fn mean_impl<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    skip_nan: bool,
) -> Result<NdArray<T::Mean>> {
    if !skip_nan {
        strict_axis(a, axis)?;
    }
    let p = Lanes::new(a, axis, keepdims)?;
    NdArray::from_shape_vec(
        &p.reduced_shape,
        means(&p, skip_nan)
            .0
            .into_iter()
            .map(T::mean_value)
            .collect(),
    )
}
pub fn mean<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Mean>> {
    mean_impl(a, axis, keepdims, false)
}
pub fn nanmean<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Mean>> {
    mean_impl(a, axis, keepdims, true)
}

fn variance<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    ddof: usize,
    skip_nan: bool,
    sqrt: bool,
) -> Result<NdArray<T::Real>> {
    if !skip_nan {
        strict_axis(a, axis)?;
    }
    let p = Lanes::new(a, axis, keepdims)?;
    let (means, counts) = means(&p, skip_nan);
    let mut sums = vec![0.0f64; p.count()];
    if p.inner == 1 {
        for out in 0..p.count() {
            let mean = means[out];
            let start = out * p.width;
            let mut sum = 0.0;
            for &x in &p.data[start..start + p.width] {
                let x = x.work();
                if skip_nan && x.is_nan() {
                    continue;
                }
                sum += (x - mean).norm_squared();
            }
            sums[out] = sum;
        }
    } else {
        for outer in 0..p.outer {
            let start = outer * p.inner;
            for k in 0..p.width {
                let base = (outer * p.width + k) * p.inner;
                for ((sum, &mean), &x) in sums[start..start + p.inner]
                    .iter_mut()
                    .zip(&means[start..start + p.inner])
                    .zip(&p.data[base..base + p.inner])
                {
                    let x = x.work();
                    if skip_nan && x.is_nan() {
                        continue;
                    }
                    *sum += (x - mean).norm_squared();
                }
            }
        }
    }
    let data = sums
        .into_iter()
        .zip(counts)
        .map(|(s, n)| {
            let mut v = if skip_nan && n <= ddof {
                f64::NAN
            } else {
                s / n.saturating_sub(ddof) as f64
            };
            if sqrt {
                v = v.sqrt();
            }
            T::Real::from_f64_lossy(v)
        })
        .collect();
    NdArray::from_shape_vec(&p.reduced_shape, data)
}
pub fn var<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    ddof: usize,
) -> Result<NdArray<T::Real>> {
    variance(a, axis, keepdims, ddof, false, false)
}
pub fn std<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    ddof: usize,
) -> Result<NdArray<T::Real>> {
    variance(a, axis, keepdims, ddof, false, true)
}
pub fn nanvar<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    ddof: usize,
) -> Result<NdArray<T::Real>> {
    variance(a, axis, keepdims, ddof, true, false)
}
pub fn nanstd<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    ddof: usize,
) -> Result<NdArray<T::Real>> {
    variance(a, axis, keepdims, ddof, true, true)
}

#[derive(Debug, Clone, Copy, Default)]
pub enum QuantileMethod {
    #[default]
    Linear,
    Lower,
    Higher,
    Midpoint,
    Nearest,
}
/// Real-valued quantiles; `q` is a 1-D list and forms the leading output axis.
/// Interpolated results are f64. Empty q preserves a leading axis of length zero.
pub fn quantile<T: SortElement>(
    a: &NdArray<T>,
    q: &[f64],
    axis: Option<isize>,
    keepdims: bool,
    method: QuantileMethod,
) -> Result<NdArray<f64>> {
    quantiles(a, q, axis, keepdims, method, false)
}
pub fn nanquantile<T: SortElement>(
    a: &NdArray<T>,
    q: &[f64],
    axis: Option<isize>,
    keepdims: bool,
    method: QuantileMethod,
) -> Result<NdArray<f64>> {
    quantiles(a, q, axis, keepdims, method, true)
}
pub fn percentile<T: SortElement>(
    a: &NdArray<T>,
    q: &[f64],
    axis: Option<isize>,
    keepdims: bool,
    method: QuantileMethod,
) -> Result<NdArray<f64>> {
    if q.iter().any(|&q| !(0.0..=100.0).contains(&q)) {
        return Err(invalid("percentiles must be in [0, 100]"));
    }
    quantile(
        a,
        &q.iter().map(|q| q / 100.0).collect::<Vec<_>>(),
        axis,
        keepdims,
        method,
    )
}
pub fn nanpercentile<T: SortElement>(
    a: &NdArray<T>,
    q: &[f64],
    axis: Option<isize>,
    keepdims: bool,
    method: QuantileMethod,
) -> Result<NdArray<f64>> {
    if q.iter().any(|&q| !(0.0..=100.0).contains(&q)) {
        return Err(invalid("percentiles must be in [0, 100]"));
    }
    nanquantile(
        a,
        &q.iter().map(|q| q / 100.0).collect::<Vec<_>>(),
        axis,
        keepdims,
        method,
    )
}

fn quantiles<T: SortElement>(
    a: &NdArray<T>,
    q: &[f64],
    axis: Option<isize>,
    keepdims: bool,
    method: QuantileMethod,
    skip_nan: bool,
) -> Result<NdArray<f64>> {
    strict_axis(a, axis)?;
    if matches!(T::DTYPE, numpy_types::DType::C64 | numpy_types::DType::C128)
        || (T::DTYPE == numpy_types::DType::Bool
            && matches!(method, QuantileMethod::Linear | QuantileMethod::Midpoint))
    {
        return Err(invalid(
            "interpolated quantiles require a real numeric dtype",
        ));
    }
    if q.iter().any(|&q| !(0.0..=1.0).contains(&q)) {
        return Err(invalid("quantiles must be in [0, 1]"));
    }
    let p = Lanes::new(a, axis, keepdims)?;
    if skip_nan && a.is_empty() {
        return NdArray::from_shape_vec(&p.reduced_shape, vec![f64::NAN; p.count()]);
    }
    if p.width == 0 && !skip_nan {
        return Err(invalid("cannot take a quantile from an empty axis"));
    }
    let mut shape = vec![q.len()];
    shape.extend_from_slice(&p.reduced_shape);
    let len = numpy_types::shape_size(&shape)?;
    let mut result = vec![f64::NAN; len];
    if p.count() == 0 || q.is_empty() {
        return NdArray::from_shape_vec(&shape, result);
    }
    let mut lane = Vec::with_capacity(p.width);
    for out in 0..p.count() {
        lane.clear();
        lane.extend(p.lane(out).filter(|&x| !skip_nan || !x.is_nan()));
        if lane.is_empty() || lane.iter().any(|&x| x.is_nan()) {
            continue;
        }
        // Small q uses O(n) selection; many q amortize one O(n log n) sort.
        let sorted = q.len() > 4;
        if sorted {
            lane.sort_unstable_by(T::numpy_cmp);
        }
        for (qi, &q) in q.iter().enumerate() {
            let h = (lane.len() - 1) as f64 * q;
            let lower = h.floor() as usize;
            let higher = h.ceil() as usize;
            let nearest = h.round_ties_even() as usize;
            let (lo, hi, t) = match method {
                QuantileMethod::Lower => (lower, lower, 0.0),
                QuantileMethod::Higher => (higher, higher, 0.0),
                QuantileMethod::Nearest => (nearest, nearest, 0.0),
                QuantileMethod::Midpoint => (
                    lower,
                    (lower + 1).min(lane.len() - 1),
                    if lower == higher { 0.0 } else { 0.5 },
                ),
                QuantileMethod::Linear => {
                    (lower, (lower + 1).min(lane.len() - 1), h - lower as f64)
                }
            };
            if !sorted {
                lane.select_nth_unstable_by(hi, T::numpy_cmp);
            }
            let upper = lane[hi];
            let lower = if lo == hi {
                upper
            } else if sorted {
                lane[lo]
            } else {
                lane[..hi].select_nth_unstable_by(lo, T::numpy_cmp);
                lane[lo]
            };
            let (a, b) = (lower.to_f64_lossy(), upper.to_f64_lossy());
            let value = match method {
                QuantileMethod::Lower | QuantileMethod::Higher | QuantileMethod::Nearest => a,
                _ => {
                    let d = upper.interpolation_delta(lower);
                    if t < 0.5 {
                        a + d * t
                    } else {
                        b - d * (1.0 - t)
                    }
                }
            };
            result[qi * p.count() + out] = value;
        }
    }
    NdArray::from_shape_vec(&shape, result)
}
fn medians<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
    skip_nan: bool,
) -> Result<NdArray<T::Mean>> {
    strict_axis(a, axis)?;
    let p = Lanes::new(a, axis, keepdims)?;
    if p.count() == 0 {
        return NdArray::from_shape_vec(&p.reduced_shape, vec![]);
    }
    let mut result = Vec::with_capacity(p.count());
    let mut lane = Vec::with_capacity(p.width);
    for out in 0..p.count() {
        lane.clear();
        lane.extend(p.lane(out).filter(|&x| !skip_nan || !x.is_nan()));
        let value = if lane.is_empty() {
            if let Some(last) = p.lane(out).last() {
                T::mean_value(last.work())
            } else {
                T::Mean::from_complex64_lossy(Complex64::new(f64::NAN, f64::NAN))
            }
        } else if let Some(&nan) = lane
            .iter()
            .filter(|&&x| x.is_nan())
            .max_by(|a, b| a.numpy_cmp(b))
        {
            // NumPy's median NaN check selects the partitioned final element;
            // preserve its finite complex component instead of inventing NaN+0j.
            T::mean_value(nan.work())
        } else {
            let mid = lane.len() / 2;
            lane.select_nth_unstable_by(mid, T::numpy_cmp);
            let b = lane[mid].work();
            if lane.len() % 2 == 0 {
                let a = lane[..mid]
                    .iter()
                    .max_by(|a, b| a.numpy_cmp(b))
                    .copied()
                    .ok_or_else(|| invalid("empty median"))?
                    .work();
                T::mean_value((a + b).divide(2.0))
            } else {
                T::mean_value(b)
            }
        };
        result.push(value);
    }
    NdArray::from_shape_vec(&p.reduced_shape, result)
}
pub fn median<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Mean>> {
    medians(a, axis, keepdims, false)
}
pub fn nanmedian<T: StatisticsElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Mean>> {
    medians(a, axis, keepdims, true)
}

/// Weighted average. Weights either match the input shape or the reduced axis.
pub fn average<T: StatisticsElement>(
    a: &NdArray<T>,
    weights: &NdArray<f64>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<NdArray<T::Work>>
where
    T::Work: Scalar,
{
    strict_axis(a, axis)?;
    let p = Lanes::new(a, axis, keepdims)?;
    let axis_weights = weights.ndim() == 1 && weights.len() == p.width;
    if weights.shape() != a.shape() && (!axis_weights || axis.is_none()) {
        return Err(invalid(
            "weights must match the input or reduced axis shape",
        ));
    }
    let w = weights.to_vec();
    let mut result = Vec::with_capacity(p.count());
    for out in 0..p.count() {
        let mut total = 0.0;
        let mut sum = <T::Work as StatWork>::zero();
        for (k, x) in p.lane(out).enumerate() {
            let weight = if weights.shape() == a.shape() {
                w[p.base(out) + k * p.inner]
            } else {
                w[k]
            };
            sum = sum + x.work().scale(weight);
            total += weight;
        }
        if total == 0.0 {
            return Err(invalid("weights sum to zero"));
        }
        result.push(sum.divide(total));
    }
    NdArray::from_shape_vec(&p.reduced_shape, result)
}
/// Covariance of real variables. `rowvar=true` means rows are variables.
pub fn cov<T: Scalar>(a: &NdArray<T>, rowvar: bool, ddof: usize) -> Result<NdArray<f64>> {
    if a.ndim() > 2 {
        return Err(invalid("cov expects a vector or matrix"));
    }
    if matches!(T::DTYPE, numpy_types::DType::C64 | numpy_types::DType::C128) {
        return Err(invalid("real cov does not accept complex data"));
    }
    let (variables, samples) = match a.ndim() {
        0 => (1, 1),
        1 => (1, a.len()),
        _ if rowvar => (a.shape()[0], a.shape()[1]),
        _ => (a.shape()[1], a.shape()[0]),
    };
    let data = a.to_vec();
    let value = |i: usize, j: usize| {
        if a.ndim() < 2 || rowvar {
            data[i * samples + j].to_f64_lossy()
        } else {
            data[j * variables + i].to_f64_lossy()
        }
    };
    let means: Vec<f64> = (0..variables)
        .map(|i| (0..samples).map(|j| value(i, j)).sum::<f64>() / samples as f64)
        .collect();
    let len = variables
        .checked_mul(variables)
        .ok_or_else(|| invalid("covariance shape overflow"))?;
    let mut result = vec![0.0; len];
    for i in 0..variables {
        for j in 0..=i {
            let sum: f64 = (0..samples)
                .map(|k| (value(i, k) - means[i]) * (value(j, k) - means[j]))
                .sum();
            let v = sum / samples.saturating_sub(ddof) as f64;
            result[i * variables + j] = v;
            result[j * variables + i] = v;
        }
    }
    let shape = if variables == 1 {
        vec![]
    } else {
        vec![variables, variables]
    };
    NdArray::from_shape_vec(&shape, result)
}
pub fn corrcoef<T: Scalar>(a: &NdArray<T>, rowvar: bool) -> Result<NdArray<f64>> {
    let covariance = cov(a, rowvar, 1)?;
    let data = covariance.to_vec();
    if covariance.ndim() == 0 {
        return Ok(NdArray::from_scalar(data[0] / data[0]));
    }
    let n = covariance.shape()[0];
    let mut out = vec![0.0; data.len()];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = (data[i * n + j] / data[i * n + i].sqrt() / data[j * n + j].sqrt())
                .clamp(-1.0, 1.0);
        }
    }
    NdArray::from_shape_vec(covariance.shape(), out)
}

use crate::lanes::{Lanes, invalid};
use numpy_types::{Complex32, Complex64, NdArray, Result, Scalar, f16};
use std::cmp::Ordering;

/// NumPy's enhanced ordering: NaNs sort last and signed zeros compare equal.
pub trait SortElement: Scalar {
    fn numpy_cmp(&self, other: &Self) -> Ordering;
    fn is_nan(self) -> bool {
        false
    }
    /// NumPy performs interpolation subtraction in the source dtype.
    fn interpolation_delta(self, lower: Self) -> f64 {
        self.to_f64_lossy() - lower.to_f64_lossy()
    }
}
macro_rules! ordered {
    ($($t:ty),*) => {$(impl SortElement for $t {
        fn numpy_cmp(&self, other: &Self) -> Ordering { self.cmp(other) }
        fn interpolation_delta(self, lower: Self) -> f64 { self.wrapping_sub(lower) as f64 }
    })*};
}
ordered!(i8, i16, i32, i64, u8, u16, u32, u64);
impl SortElement for bool {
    fn numpy_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}
fn real_cmp(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}
macro_rules! float_ordered {
    ($($t:ty),*) => {$(impl SortElement for $t {
        fn numpy_cmp(&self, other: &Self) -> Ordering {
            real_cmp(self.to_f64_lossy(), other.to_f64_lossy())
        }
        fn is_nan(self) -> bool { self.to_f64_lossy().is_nan() }
        fn interpolation_delta(self, lower: Self) -> f64 { (self - lower).to_f64_lossy() }
    })*};
}
float_ordered!(f16, f32, f64);
fn complex_cmp(a: Complex64, b: Complex64) -> Ordering {
    let category = |z: Complex64| 2 * u8::from(z.re.is_nan()) + u8::from(z.im.is_nan());
    category(a)
        .cmp(&category(b))
        .then_with(|| match category(a) {
            0 => real_cmp(a.re, b.re).then_with(|| real_cmp(a.im, b.im)),
            1 => real_cmp(a.re, b.re),
            2 => real_cmp(a.im, b.im),
            _ => Ordering::Equal,
        })
}
macro_rules! complex_ordered {
    ($($t:ty),*) => {$(impl SortElement for $t {
        fn numpy_cmp(&self, other: &Self) -> Ordering {
            complex_cmp(self.to_complex64(), other.to_complex64())
        }
        fn is_nan(self) -> bool { self.re.is_nan() || self.im.is_nan() }
    })*};
}
complex_ordered!(Complex32, Complex64);

/// Stable sort along an axis, or flatten with `axis=None`.
pub fn sort<T: SortElement>(a: &NdArray<T>, axis: Option<isize>) -> Result<NdArray<T>> {
    if a.ndim() == 0 && axis.is_some() {
        return Err(invalid(
            "sort requires an array with at least one dimension",
        ));
    }
    let plan = Lanes::new(a, axis, false)?;
    if a.is_empty() {
        return NdArray::from_shape_vec(&plan.shape, vec![]);
    }
    let mut data = plan.data.to_vec();
    if plan.inner == 1 {
        if plan.width > 0 {
            for lane in data.chunks_mut(plan.width) {
                lane.sort_by(T::numpy_cmp);
            }
        }
    } else {
        let mut lane = Vec::with_capacity(plan.width);
        for out in 0..plan.count() {
            lane.clear();
            lane.extend(plan.lane(out));
            lane.sort_by(T::numpy_cmp);
            let base = plan.base(out);
            for (k, &v) in lane.iter().enumerate() {
                data[base + k * plan.inner] = v;
            }
        }
    }
    NdArray::from_shape_vec(&plan.shape, data)
}

pub fn argsort<T: SortElement>(a: &NdArray<T>, axis: Option<isize>) -> Result<NdArray<i64>> {
    indexed_sort(a, axis, None)
}
fn indexed_sort<T: SortElement>(
    a: &NdArray<T>,
    axis: Option<isize>,
    kth: Option<isize>,
) -> Result<NdArray<i64>> {
    let axis = if a.ndim() == 0 && matches!(axis, Some(0 | -1)) {
        None
    } else {
        axis
    };
    if a.ndim() == 0 && axis.is_some() {
        return Err(invalid("axis out of bounds"));
    }
    let p = Lanes::new(a, axis, false)?;
    let k = kth.map(|k| normalized_k(k, p.width)).transpose()?;
    if a.is_empty() {
        return NdArray::from_shape_vec(&p.shape, vec![]);
    }
    let mut result = vec![0i64; p.data.len()];
    let mut indices: Vec<usize> = (0..p.width).collect();
    for out in 0..p.count() {
        for (i, slot) in indices.iter_mut().enumerate() {
            *slot = i;
        }
        let base = p.base(out);
        let cmp = |&i: &usize, &j: &usize| {
            p.data[base + i * p.inner].numpy_cmp(&p.data[base + j * p.inner])
        };
        if let Some(k) = k {
            if p.width > 0 {
                indices.select_nth_unstable_by(k, cmp);
            }
        } else {
            indices.sort_by(cmp);
        }
        for (j, &i) in indices.iter().enumerate() {
            result[base + j * p.inner] = i as i64;
        }
    }
    NdArray::from_shape_vec(&p.shape, result)
}
fn normalized_k(k: isize, n: usize) -> Result<usize> {
    if n == 0 {
        return Ok(0);
    }
    let n_signed =
        isize::try_from(n).map_err(|_| invalid("axis length exceeds signed index range"))?;
    let k = if k < 0 {
        k.checked_add(n_signed)
            .ok_or_else(|| invalid("kth overflow"))?
    } else {
        k
    };
    if k < 0 || k as usize >= n {
        return Err(invalid("kth is out of bounds"));
    }
    Ok(k as usize)
}
pub fn partition<T: SortElement>(
    a: &NdArray<T>,
    kth: isize,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    if a.ndim() == 0 && axis.is_some() {
        return Err(invalid("partition requires at least one dimension"));
    }
    let p = Lanes::new(a, axis, false)?;
    let k = normalized_k(kth, p.width)?;
    if a.is_empty() {
        return NdArray::from_shape_vec(&p.shape, vec![]);
    }
    let mut data = p.data.to_vec();
    if p.width > 0 {
        if p.inner == 1 {
            for lane in data.chunks_mut(p.width) {
                lane.select_nth_unstable_by(k, T::numpy_cmp);
            }
        } else {
            let mut lane = Vec::with_capacity(p.width);
            for out in 0..p.count() {
                lane.clear();
                lane.extend(p.lane(out));
                lane.select_nth_unstable_by(k, T::numpy_cmp);
                let base = p.base(out);
                for (j, &v) in lane.iter().enumerate() {
                    data[base + j * p.inner] = v;
                }
            }
        }
    }
    NdArray::from_shape_vec(&p.shape, data)
}
pub fn argpartition<T: SortElement>(
    a: &NdArray<T>,
    kth: isize,
    axis: Option<isize>,
) -> Result<NdArray<i64>> {
    indexed_sort(a, axis, Some(kth))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSide {
    Left,
    Right,
}
pub fn searchsorted<T: SortElement>(
    a: &NdArray<T>,
    values: &NdArray<T>,
    side: SearchSide,
) -> Result<NdArray<i64>> {
    if a.ndim() != 1 {
        return Err(invalid(
            "searchsorted requires a one-dimensional sorted array",
        ));
    }
    let sorted = a.to_c_contiguous();
    let data = sorted
        .as_slice()
        .ok_or_else(|| invalid("expected contiguous data"))?;
    Ok(values.map(|v| {
        data.partition_point(|x| {
            let c = x.numpy_cmp(&v);
            c == Ordering::Less || (side == SearchSide::Right && c == Ordering::Equal)
        }) as i64
    }))
}
/// Stable indirect sort; the last key is the primary key.
pub fn lexsort<T: SortElement>(keys: &[&NdArray<T>], axis: isize) -> Result<NdArray<i64>> {
    let first = keys
        .first()
        .ok_or_else(|| invalid("lexsort requires at least one key"))?;
    if keys.iter().any(|a| a.shape() != first.shape()) {
        return Err(invalid("all keys must have the same shape"));
    }
    let plans = keys
        .iter()
        .map(|a| Lanes::new(a, Some(axis), false))
        .collect::<Result<Vec<_>>>()?;
    let p = &plans[0];
    if first.is_empty() {
        return NdArray::from_shape_vec(first.shape(), vec![]);
    }
    let mut result = vec![0i64; first.len()];
    let mut indices: Vec<usize> = (0..p.width).collect();
    for out in 0..p.count() {
        for (i, v) in indices.iter_mut().enumerate() {
            *v = i;
        }
        let base = p.base(out);
        indices.sort_by(|&i, &j| {
            for plan in plans.iter().rev() {
                let c = plan.data[base + i * p.inner].numpy_cmp(&plan.data[base + j * p.inner]);
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        });
        for (j, &v) in indices.iter().enumerate() {
            result[base + j * p.inner] = v as i64;
        }
    }
    NdArray::from_shape_vec(first.shape(), result)
}
/// Largest `k` values, sorted descending, along `axis`.
pub fn top_k<T: SortElement>(a: &NdArray<T>, k: usize, axis: isize) -> Result<NdArray<T>> {
    let p = Lanes::new(a, Some(axis), false)?;
    if k > p.width {
        return Err(invalid("k exceeds the axis length"));
    }
    let ax = numpy_types::normalize_axis(axis, a.ndim())?;
    let mut shape = a.shape().to_vec();
    shape[ax] = k;
    if a.is_empty() || k == 0 {
        return NdArray::from_shape_vec(&shape, vec![]);
    }
    let mut data = vec![T::zero(); p.count() * k];
    let mut lane = Vec::with_capacity(p.width);
    for out in 0..p.count() {
        lane.clear();
        lane.extend(p.lane(out));
        if k > 0 {
            lane.select_nth_unstable_by(k - 1, |a, b| b.numpy_cmp(a));
            lane[..k].sort_by(|a, b| b.numpy_cmp(a));
        }
        let base = (out / p.inner) * k * p.inner + out % p.inner;
        for j in 0..k {
            data[base + j * p.inner] = lane[j];
        }
    }
    NdArray::from_shape_vec(&shape, data)
}
pub fn sort_complex<T: Scalar>(a: &NdArray<T>) -> Result<NdArray<Complex64>> {
    sort(&a.map(Scalar::to_complex64), Some(-1))
}

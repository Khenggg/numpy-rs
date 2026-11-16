use crate::lanes::{Lanes, invalid};
use numpy_types::{NdArray, Result, Scalar, normalize_axis};

#[derive(Clone, Copy, Debug)]
pub enum ConvolveMode {
    Full,
    Same,
    Valid,
}
/// Direct 1-D convolution. The summation order is deterministic.
pub fn convolve(a: &NdArray<f64>, b: &NdArray<f64>, mode: ConvolveMode) -> Result<NdArray<f64>> {
    if a.ndim() != 1 || b.ndim() != 1 || a.is_empty() || b.is_empty() {
        return Err(invalid("convolve requires nonempty one-dimensional arrays"));
    }
    let (a, b) = if a.len() >= b.len() {
        (a.to_vec(), b.to_vec())
    } else {
        (b.to_vec(), a.to_vec())
    };
    let len = a
        .len()
        .checked_add(b.len())
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| invalid("convolution size overflow"))?;
    let (start, count) = match mode {
        ConvolveMode::Full => (0, len),
        ConvolveMode::Same => ((b.len() - 1) / 2, a.len()),
        ConvolveMode::Valid => (b.len() - 1, a.len() - b.len() + 1),
    };
    let out = (start..start + count)
        .map(|k| {
            let lo = k.saturating_sub(b.len() - 1);
            let hi = (k + 1).min(a.len());
            (lo..hi).map(|j| a[j] * b[k - j]).sum()
        })
        .collect();
    Ok(NdArray::from_vec(out))
}
pub fn correlate(a: &NdArray<f64>, b: &NdArray<f64>, mode: ConvolveMode) -> Result<NdArray<f64>> {
    if a.ndim() != 1 || b.ndim() != 1 || a.is_empty() || b.is_empty() {
        return Err(invalid(
            "correlate requires nonempty one-dimensional arrays",
        ));
    }
    let a = a.to_vec();
    let b = b.to_vec();
    let full = a
        .len()
        .checked_add(b.len())
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| invalid("correlation size overflow"))?;
    let (start, count) = match mode {
        ConvolveMode::Full => (0, full),
        ConvolveMode::Same => (
            (a.len().min(b.len()) - usize::from(a.len() >= b.len())) / 2,
            a.len().max(b.len()),
        ),
        ConvolveMode::Valid => (a.len().min(b.len()) - 1, a.len().abs_diff(b.len()) + 1),
    };
    let out = (start..start + count)
        .map(|k| {
            let lag = k as isize - (b.len() - 1) as isize;
            let lo = (-lag).max(0) as usize;
            let hi = (a.len() as isize - lag).min(b.len() as isize).max(0) as usize;
            (lo..hi)
                .map(|j| a[(j as isize + lag) as usize] * b[j])
                .sum()
        })
        .collect();
    Ok(NdArray::from_vec(out))
}
pub fn interp(
    x: &NdArray<f64>,
    xp: &NdArray<f64>,
    fp: &NdArray<f64>,
    left: Option<f64>,
    right: Option<f64>,
) -> Result<NdArray<f64>> {
    if xp.ndim() != 1 || fp.shape() != xp.shape() || xp.is_empty() {
        return Err(invalid(
            "xp and fp must be nonempty vectors of equal length",
        ));
    }
    let xp = xp.to_vec();
    let fp = fp.to_vec();
    Ok(x.map(|x| {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < xp[0] {
            return left.unwrap_or(fp[0]);
        }
        let last = xp.len() - 1;
        if x > xp[last] {
            return right.unwrap_or(fp[last]);
        }
        let k = xp.partition_point(|&v| v <= x).saturating_sub(1).min(last);
        if k == last || x == xp[k] {
            return fp[k];
        }
        let slope = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k]);
        let value = fp[k] + slope * (x - xp[k]);
        if value.is_nan() {
            let value = fp[k + 1] + slope * (x - xp[k + 1]);
            if value.is_nan() && fp[k] == fp[k + 1] {
                fp[k]
            } else {
                value
            }
        } else {
            value
        }
    }))
}
pub fn diff(a: &NdArray<f64>, n: usize, axis: isize) -> Result<NdArray<f64>> {
    if n == 0 {
        return Ok(a.clone());
    }
    let ax = normalize_axis(axis, a.ndim())?;
    let mut current = a.clone();
    for _ in 0..n.min(a.shape()[ax]) {
        let p = Lanes::new(&current, Some(axis), false)?;
        let mut shape = current.shape().to_vec();
        shape[ax] = p.width.saturating_sub(1);
        let mut out = Vec::with_capacity(numpy_types::shape_size(&shape)?);
        for outer in 0..p.outer {
            for k in 1..p.width {
                for i in 0..p.inner {
                    let j = (outer * p.width + k) * p.inner + i;
                    out.push(p.data[j] - p.data[j - p.inner]);
                }
            }
        }
        current = NdArray::from_shape_vec(&shape, out)?;
    }
    Ok(current)
}
pub fn gradient(
    a: &NdArray<f64>,
    spacing: f64,
    axis: isize,
    edge_order: usize,
) -> Result<NdArray<f64>> {
    if !matches!(edge_order, 1 | 2) {
        return Err(invalid("edge_order must be 1 or 2"));
    }
    let p = Lanes::new(a, Some(axis), false)?;
    if p.width < edge_order + 1 {
        return Err(invalid("array is too short for the requested edge_order"));
    }
    let mut result = vec![0.0; a.len()];
    for out in 0..p.count() {
        let base = p.base(out);
        let x = |k: usize| p.data[base + k * p.inner];
        for k in 0..p.width {
            let v = if k == 0 {
                if edge_order == 1 {
                    (x(1) - x(0)) / spacing
                } else {
                    (-1.5 * x(0) + 2.0 * x(1) - 0.5 * x(2)) / spacing
                }
            } else if k == p.width - 1 {
                if edge_order == 1 {
                    (x(k) - x(k - 1)) / spacing
                } else {
                    (1.5 * x(k) - 2.0 * x(k - 1) + 0.5 * x(k - 2)) / spacing
                }
            } else {
                (x(k + 1) - x(k - 1)) / (2.0 * spacing)
            };
            result[base + k * p.inner] = v;
        }
    }
    NdArray::from_shape_vec(a.shape(), result)
}
pub fn trapezoid<T: Scalar>(
    a: &NdArray<T>,
    x: Option<&NdArray<f64>>,
    dx: f64,
    axis: isize,
) -> Result<NdArray<f64>> {
    normalize_axis(axis, a.ndim())?;
    if matches!(T::DTYPE, numpy_types::DType::C64 | numpy_types::DType::C128) {
        return Err(invalid("real trapezoid requires real input"));
    }
    let p = Lanes::new(a, Some(axis), false)?;
    let xp = x
        .map(|x| Lanes::new(x, if x.ndim() == 1 { Some(0) } else { Some(axis) }, false))
        .transpose()?;
    if let Some(xp) = &xp {
        if xp.width != p.width || (xp.shape.len() != 1 && xp.shape != a.shape()) {
            return Err(invalid("x must match input shape or reduced axis length"));
        }
    }
    let mut result = vec![0.0; p.count()];
    for (out, slot) in result.iter_mut().enumerate() {
        let base = p.base(out);
        for k in 1..p.width {
            let width = if let Some(xp) = &xp {
                let xb = if xp.shape.len() == 1 { 0 } else { xp.base(out) };
                xp.data[xb + k * xp.inner] - xp.data[xb + (k - 1) * xp.inner]
            } else {
                dx
            };
            *slot += (p.data[base + k * p.inner].to_f64_lossy()
                + p.data[base + (k - 1) * p.inner].to_f64_lossy())
                * 0.5
                * width;
        }
    }
    NdArray::from_shape_vec(&p.reduced_shape, result)
}
pub fn unwrap(
    a: &NdArray<f64>,
    discont: Option<f64>,
    axis: isize,
    period: f64,
) -> Result<NdArray<f64>> {
    normalize_axis(axis, a.ndim())?;
    if !period.is_finite() || period <= 0.0 {
        return Err(invalid("period must be finite and positive"));
    }
    let p = Lanes::new(a, Some(axis), false)?;
    let threshold = discont.unwrap_or(period / 2.0).max(period / 2.0);
    let mut result = p.data.to_vec();
    for out in 0..p.count() {
        let base = p.base(out);
        let mut correction = 0.0;
        for k in 1..p.width {
            let index = base + k * p.inner;
            let delta = p.data[index] - p.data[index - p.inner];
            let mut adjusted = (delta + period / 2.0).rem_euclid(period) - period / 2.0;
            if adjusted == -period / 2.0 && delta > 0.0 {
                adjusted = period / 2.0;
            }
            if !(delta.abs() < threshold) {
                correction += adjusted - delta;
            }
            result[index] += correction;
        }
    }
    NdArray::from_shape_vec(a.shape(), result)
}

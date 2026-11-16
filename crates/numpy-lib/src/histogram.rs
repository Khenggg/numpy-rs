use crate::lanes::invalid;
use numpy_types::{NdArray, Result, Scalar};

#[derive(Clone, Copy, Debug)]
pub enum Bins<'a> {
    Count(usize),
    Edges(&'a [f64]),
}
pub struct Histogram {
    pub counts: NdArray<f64>,
    pub edges: NdArray<f64>,
}

pub fn histogram_bin_edges<T: Scalar>(
    a: &NdArray<T>,
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
) -> Result<NdArray<f64>> {
    if matches!(T::DTYPE, numpy_types::DType::C64 | numpy_types::DType::C128) {
        return Err(invalid("real histogram edges require real input"));
    }
    match bins {
        Bins::Edges(edges) => {
            if edges.len() < 2 || edges.windows(2).any(|w| w[0] > w[1]) {
                return Err(invalid(
                    "bin edges must be monotonically nondecreasing and have length >= 2",
                ));
            }
            Ok(NdArray::from_vec(edges.to_vec()))
        }
        Bins::Count(n) => {
            if n == 0 {
                return Err(invalid("bins must be positive"));
            }
            let (mut lo, mut hi) = match range {
                Some(range) => range,
                None if a.is_empty() => (0.0, 1.0),
                None => {
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for x in a.iter() {
                        let x = x.to_f64_lossy();
                        if !x.is_finite() {
                            return Err(invalid("autodetected histogram range is not finite"));
                        }
                        lo = lo.min(x);
                        hi = hi.max(x);
                    }
                    (lo, hi)
                }
            };
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(invalid("histogram range must be finite and increasing"));
            }
            if lo == hi {
                lo -= 0.5;
                hi += 0.5;
            }
            let count = n
                .checked_add(1)
                .ok_or_else(|| invalid("bin count overflow"))?;
            let span = hi - lo;
            let step = span / n as f64;
            let edges = (0..count)
                .map(|i| {
                    if i == n {
                        hi
                    } else {
                        if span.is_finite() {
                            lo + step * i as f64
                        } else {
                            // Fall back only when the finite endpoints' difference overflows.
                            let t = i as f64 / n as f64;
                            lo * (1.0 - t) + hi * t
                        }
                    }
                })
                .collect();
            Ok(NdArray::from_vec(edges))
        }
    }
}
fn bin_index(x: f64, edges: &[f64]) -> Option<usize> {
    if x.is_nan() || x < edges[0] || x > edges[edges.len() - 1] {
        return None;
    }
    let i = edges.partition_point(|&e| e <= x);
    if i == 0 {
        None
    } else {
        Some((i - 1).min(edges.len() - 2))
    }
}
pub fn histogram<T: Scalar>(
    a: &NdArray<T>,
    bins: Bins<'_>,
    range: Option<(f64, f64)>,
    weights: Option<&NdArray<f64>>,
    density: bool,
) -> Result<Histogram> {
    if let Some(w) = weights {
        if w.shape() != a.shape() {
            return Err(invalid("weights must match the input shape"));
        }
    }
    let edges = histogram_bin_edges(a, bins, range)?;
    let e = edges.to_vec();
    let mut counts = vec![0.0; e.len() - 1];
    let mut w = weights.map(|w| w.iter());
    for x in a.iter() {
        let weight = w.as_mut().and_then(Iterator::next).unwrap_or(1.0);
        if let Some(i) = bin_index(x.to_f64_lossy(), &e) {
            counts[i] += weight;
        }
    }
    if density {
        let total: f64 = counts.iter().sum();
        for (i, c) in counts.iter_mut().enumerate() {
            *c = *c / (e[i + 1] - e[i]) / total;
        }
    }
    Ok(Histogram {
        counts: NdArray::from_vec(counts),
        edges,
    })
}
pub fn bincount(
    a: &NdArray<i64>,
    weights: Option<&NdArray<f64>>,
    minlength: usize,
) -> Result<NdArray<f64>> {
    if a.ndim() != 1 || a.iter().any(|v| v < 0) {
        return Err(invalid(
            "bincount expects one-dimensional nonnegative integers",
        ));
    }
    if weights.is_some_and(|w| w.shape() != a.shape()) {
        return Err(invalid("weights must match input shape"));
    }
    let max = a.iter().max();
    let len = match max {
        Some(max) => usize::try_from(max)
            .ok()
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| invalid("bin count overflow"))?
            .max(minlength),
        None => minlength,
    };
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("bin count too large"))?;
    data.resize(len, 0.0);
    let mut w = weights.map(|w| w.iter());
    for index in a.iter() {
        data[index as usize] += w.as_mut().and_then(Iterator::next).unwrap_or(1.0);
    }
    Ok(NdArray::from_vec(data))
}
pub fn digitize<T: Scalar>(
    a: &NdArray<T>,
    bins: &NdArray<f64>,
    right: bool,
) -> Result<NdArray<i64>> {
    if matches!(T::DTYPE, numpy_types::DType::C64 | numpy_types::DType::C128) {
        return Err(invalid("digitize requires real input"));
    }
    if bins.ndim() != 1 {
        return Err(invalid("bins must be one-dimensional"));
    }
    let bins = bins.to_vec();
    let increasing = bins.windows(2).all(|w| w[0] <= w[1]);
    let decreasing = bins.windows(2).all(|w| w[0] >= w[1]);
    if !increasing && !decreasing {
        return Err(invalid("bins must be monotonic"));
    }
    Ok(a.map(|x| {
        let x = x.to_f64_lossy();
        if increasing {
            bins.partition_point(|&e| x.is_nan() || if right { e < x } else { e <= x }) as i64
        } else {
            bins.partition_point(|&e| if right { e >= x && !x.is_nan() } else { e > x }) as i64
        }
    }))
}
pub struct HistogramDD {
    pub counts: NdArray<f64>,
    pub edges: Vec<NdArray<f64>>,
}
/// `sample` has shape [observations, dimensions].
pub fn histogramdd<T: Scalar>(
    sample: &NdArray<T>,
    bins: &[Bins<'_>],
    weights: Option<&NdArray<f64>>,
    density: bool,
) -> Result<HistogramDD> {
    if sample.ndim() != 2 || sample.shape()[1] != bins.len() || bins.is_empty() {
        return Err(invalid(
            "sample must have shape [N, bins.len()] with at least one dimension",
        ));
    }
    let n = sample.shape()[0];
    let d = bins.len();
    if weights.is_some_and(|w| w.ndim() != 1 || w.len() != n) {
        return Err(invalid("weights must have shape [N]"));
    }
    let data = sample.to_vec();
    let mut edges = Vec::with_capacity(d);
    let mut shape = Vec::with_capacity(d);
    for (j, &bin) in bins.iter().enumerate() {
        let column = NdArray::from_vec((0..n).map(|i| data[i * d + j]).collect());
        let e = histogram_bin_edges(&column, bin, None)?;
        shape.push(e.len() - 1);
        edges.push(e);
    }
    let edge_data: Vec<Vec<f64>> = edges.iter().map(|e| e.to_vec()).collect();
    let mut counts = vec![0.0; numpy_types::shape_size(&shape)?];
    let mut w = weights.map(|w| w.iter());
    for i in 0..n {
        let weight = w.as_mut().and_then(Iterator::next).unwrap_or(1.0);
        let mut index = 0;
        let mut valid = true;
        for j in 0..d {
            if let Some(k) = bin_index(data[i * d + j].to_f64_lossy(), &edge_data[j]) {
                index = index * shape[j] + k;
            } else {
                valid = false;
                break;
            }
        }
        if valid {
            counts[index] += weight;
        }
    }
    if density {
        let total: f64 = counts.iter().sum();
        for (index, count) in counts.iter_mut().enumerate() {
            let mut coord = index;
            let mut volume = 1.0;
            for j in (0..d).rev() {
                let k = coord % shape[j];
                coord /= shape[j];
                volume *= edge_data[j][k + 1] - edge_data[j][k];
            }
            *count = *count / volume / total;
        }
    }
    Ok(HistogramDD {
        counts: NdArray::from_shape_vec(&shape, counts)?,
        edges,
    })
}
pub fn histogram2d<T: Scalar>(
    x: &NdArray<T>,
    y: &NdArray<T>,
    bins: [Bins<'_>; 2],
    weights: Option<&NdArray<f64>>,
    density: bool,
) -> Result<HistogramDD> {
    if x.ndim() != 1 || y.shape() != x.shape() {
        return Err(invalid("x and y must be vectors of equal length"));
    }
    let data = x.iter().zip(y.iter()).flat_map(|(x, y)| [x, y]).collect();
    histogramdd(
        &NdArray::from_shape_vec(&[x.len(), 2], data)?,
        &bins,
        weights,
        density,
    )
}

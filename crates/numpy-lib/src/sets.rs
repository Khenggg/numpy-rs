use crate::SortElement;
use numpy_types::{NdArray, Result};

#[derive(Debug, Clone)]
pub struct UniqueResult<T: Copy> {
    pub values: NdArray<T>,
    pub indices: NdArray<i64>,
    pub inverse: NdArray<i64>,
    pub counts: NdArray<i64>,
}
/// Sorted unique flattened values. NaNs compare equal by default.
pub fn unique<T: SortElement>(a: &NdArray<T>) -> Result<NdArray<T>> {
    let mut data = a.to_vec();
    data.sort_by(T::numpy_cmp);
    data.dedup_by(|a, b| *a == *b || (a.is_nan() && b.is_nan()));
    Ok(NdArray::from_vec(data))
}
pub fn unique_values<T: SortElement>(a: &NdArray<T>) -> Result<NdArray<T>> {
    let mut data = a.to_vec();
    data.sort_by(T::numpy_cmp);
    data.dedup();
    Ok(NdArray::from_vec(data))
}
pub fn unique_all<T: SortElement>(a: &NdArray<T>) -> Result<UniqueResult<T>> {
    let data = a.to_vec();
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&i, &j| data[i].numpy_cmp(&data[j]));
    let mut values = Vec::new();
    let mut indices = Vec::new();
    let mut counts = Vec::new();
    let mut inverse = vec![0i64; data.len()];
    for index in order {
        let value = data[index];
        if values.last().is_none_or(|&prev: &T| prev != value) {
            values.push(value);
            indices.push(index as i64);
            counts.push(0i64);
        }
        let group = values.len() - 1;
        counts[group] += 1;
        inverse[index] = group as i64;
    }
    Ok(UniqueResult {
        values: NdArray::from_vec(values),
        indices: NdArray::from_vec(indices),
        inverse: NdArray::from_shape_vec(a.shape(), inverse)?,
        counts: NdArray::from_vec(counts),
    })
}
pub fn unique_counts<T: SortElement>(a: &NdArray<T>) -> Result<(NdArray<T>, NdArray<i64>)> {
    let u = unique_all(a)?;
    Ok((u.values, u.counts))
}
pub fn unique_inverse<T: SortElement>(a: &NdArray<T>) -> Result<(NdArray<T>, NdArray<i64>)> {
    let u = unique_all(a)?;
    Ok((u.values, u.inverse))
}
pub fn isin<T: SortElement>(
    elements: &NdArray<T>,
    test: &NdArray<T>,
    invert: bool,
) -> Result<NdArray<bool>> {
    let mut sorted = test.to_vec();
    sorted.sort_unstable_by(T::numpy_cmp);
    Ok(elements.map(|x| {
        let i = sorted.partition_point(|v| v.numpy_cmp(&x).is_lt());
        (i < sorted.len() && sorted[i] == x) != invert
    }))
}
pub fn union1d<T: SortElement>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let mut data = a.to_vec();
    data.extend(b.iter());
    unique(&NdArray::from_vec(data))
}
pub fn intersect1d<T: SortElement>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let a = unique(a)?;
    let mask = isin(&a, b, false)?;
    Ok(NdArray::from_vec(
        a.iter()
            .zip(mask.iter())
            .filter_map(|(x, keep)| keep.then_some(x))
            .collect(),
    ))
}
pub fn setdiff1d<T: SortElement>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let a = unique(a)?;
    let mask = isin(&a, b, true)?;
    Ok(NdArray::from_vec(
        a.iter()
            .zip(mask.iter())
            .filter_map(|(x, keep)| keep.then_some(x))
            .collect(),
    ))
}
pub fn setxor1d<T: SortElement>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    // NaNs from separate inputs remain distinct, as in NumPy's adjacent equality test.
    let mut data = unique(a)?.to_vec();
    data.extend(unique(b)?.iter());
    data.sort_by(T::numpy_cmp);
    let out = data
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| {
            ((i == 0 || data[i - 1] != x) && (i + 1 == data.len() || data[i + 1] != x)).then_some(x)
        })
        .collect();
    Ok(NdArray::from_vec(out))
}
pub fn ediff1d<T: numpy_ufunc::HigherOrderScalar>(a: &NdArray<T>) -> Result<NdArray<T>> {
    let values = a.to_vec();
    numpy_ufunc::subtract(
        &NdArray::from_vec(values.iter().skip(1).copied().collect()),
        &NdArray::from_vec(
            values
                .iter()
                .take(values.len().saturating_sub(1))
                .copied()
                .collect(),
        ),
    )
}

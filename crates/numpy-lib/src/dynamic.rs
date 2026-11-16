//! Runtime dtype dispatch, including NumPy integer accumulation promotion.
use numpy_types::{DType, DynArray, Result, dispatch_dyn};
use numpy_ufunc::BinaryOp;

fn axis(a: &DynArray, axis: Option<isize>) -> Option<isize> {
    if a.ndim() == 0 && matches!(axis, Some(0 | -1)) {
        None
    } else {
        axis
    }
}
pub fn sum(
    a: &DynArray,
    ax: Option<isize>,
    dtype: Option<DType>,
    keepdims: bool,
) -> Result<DynArray> {
    BinaryOp::Add.reduce(a, axis(a, ax), dtype, keepdims, None)
}
pub fn prod(
    a: &DynArray,
    ax: Option<isize>,
    dtype: Option<DType>,
    keepdims: bool,
) -> Result<DynArray> {
    BinaryOp::Multiply.reduce(a, axis(a, ax), dtype, keepdims, None)
}
pub fn min(a: &DynArray, ax: Option<isize>, keepdims: bool) -> Result<DynArray> {
    BinaryOp::Minimum.reduce(a, axis(a, ax), None, keepdims, None)
}
pub fn max(a: &DynArray, ax: Option<isize>, keepdims: bool) -> Result<DynArray> {
    BinaryOp::Maximum.reduce(a, axis(a, ax), None, keepdims, None)
}
macro_rules! stats_dispatch {
    ($($name:ident),*) => {$(pub fn $name(a: &DynArray, axis: Option<isize>, keepdims: bool) -> Result<DynArray> {
        dispatch_dyn!(a, arr => Ok(DynArray::from(crate::$name(arr, axis, keepdims)?)))
    })*};
}
stats_dispatch!(
    mean,
    nanmean,
    median,
    nanmedian,
    nanmin,
    nanmax,
    all,
    any,
    count_nonzero
);
pub fn var(a: &DynArray, axis: Option<isize>, keepdims: bool, ddof: usize) -> Result<DynArray> {
    dispatch_dyn!(a, arr => Ok(DynArray::from(crate::var(arr, axis, keepdims, ddof)?)))
}
pub fn std(a: &DynArray, axis: Option<isize>, keepdims: bool, ddof: usize) -> Result<DynArray> {
    dispatch_dyn!(a, arr => Ok(DynArray::from(crate::std(arr, axis, keepdims, ddof)?)))
}
pub fn sort(a: &DynArray, axis: Option<isize>) -> Result<DynArray> {
    dispatch_dyn!(a, arr => Ok(DynArray::from(crate::sort(arr, axis)?)))
}
pub fn histogram(
    a: &DynArray,
    bins: crate::Bins<'_>,
    range: Option<(f64, f64)>,
    weights: Option<&DynArray>,
    density: bool,
) -> Result<crate::DynHistogram> {
    crate::histogram_dyn(a, bins, range, weights, density)
}
pub fn bincount(
    a: &numpy_types::NdArray<i64>,
    weights: Option<&DynArray>,
    minlength: usize,
) -> Result<DynArray> {
    crate::bincount_dyn(a, weights, minlength)
}

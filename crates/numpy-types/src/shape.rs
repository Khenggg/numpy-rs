use crate::error::{NumPyError, Result};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Memory layout order matching NumPy (`'C'`, `'F'`, `'A'`, `'K'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Order {
    #[default]
    C,
    F,
    A,
    K,
}

/// A Python/NumPy-style slice specification `start:stop:step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slice {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

impl Slice {
    #[inline]
    pub const fn new(start: Option<isize>, stop: Option<isize>, step: isize) -> Self {
        Self { start, stop, step }
    }

    #[inline]
    pub const fn full() -> Self {
        Self {
            start: None,
            stop: None,
            step: 1,
        }
    }

    #[inline]
    pub const fn step_by(mut self, step: isize) -> Self {
        self.step = step;
        self
    }

    /// Resolve slice against a dimension of length `dim_len`, returning `(first_idx, count, step)`.
    pub fn resolve(self, dim_len: usize) -> Result<(usize, usize, isize)> {
        if self.step == 0 {
            return Err(NumPyError::InvalidValue("slice step cannot be zero".into()));
        }
        if dim_len == 0 {
            return Ok((0, 0, self.step));
        }
        let n = dim_len as isize;
        if self.step > 0 {
            let start = match self.start {
                None => 0,
                Some(s) => {
                    let s = if s < 0 { s + n } else { s };
                    s.clamp(0, n)
                }
            };
            let stop = match self.stop {
                None => n,
                Some(s) => {
                    let s = if s < 0 { s + n } else { s };
                    s.clamp(0, n)
                }
            };
            if start >= stop {
                Ok((0, 0, self.step))
            } else {
                let count = ((stop - start - 1) / self.step + 1) as usize;
                Ok((start as usize, count, self.step))
            }
        } else {
            let start = match self.start {
                None => n - 1,
                Some(s) => {
                    let s = if s < 0 { s + n } else { s };
                    s.clamp(-1, n - 1)
                }
            };
            let stop = match self.stop {
                None => -1,
                Some(s) => {
                    let s = if s < 0 { s + n } else { s };
                    s.clamp(-1, n - 1)
                }
            };
            if start <= stop {
                Ok((0, 0, self.step))
            } else {
                let count = ((start - stop - 1) / (-self.step) + 1) as usize;
                Ok((start as usize, count, self.step))
            }
        }
    }
}

/// Indexing element for `NdArray::slice` / `ArrayView::slice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceArg {
    Index(isize),
    Slice(Slice),
    NewAxis,
    Ellipsis,
}

impl From<isize> for SliceArg {
    #[inline]
    fn from(idx: isize) -> Self {
        Self::Index(idx)
    }
}

impl From<usize> for SliceArg {
    #[inline]
    fn from(idx: usize) -> Self {
        Self::Index(idx as isize)
    }
}

impl From<i32> for SliceArg {
    #[inline]
    fn from(idx: i32) -> Self {
        Self::Index(idx as isize)
    }
}

impl From<Slice> for SliceArg {
    #[inline]
    fn from(s: Slice) -> Self {
        Self::Slice(s)
    }
}

impl From<Range<isize>> for SliceArg {
    #[inline]
    fn from(r: Range<isize>) -> Self {
        Self::Slice(Slice::new(Some(r.start), Some(r.end), 1))
    }
}

impl From<RangeFrom<isize>> for SliceArg {
    #[inline]
    fn from(r: RangeFrom<isize>) -> Self {
        Self::Slice(Slice::new(Some(r.start), None, 1))
    }
}

impl From<RangeTo<isize>> for SliceArg {
    #[inline]
    fn from(r: RangeTo<isize>) -> Self {
        Self::Slice(Slice::new(None, Some(r.end), 1))
    }
}

impl From<RangeFull> for SliceArg {
    #[inline]
    fn from(_: RangeFull) -> Self {
        Self::Slice(Slice::full())
    }
}

impl From<RangeInclusive<isize>> for SliceArg {
    #[inline]
    fn from(r: RangeInclusive<isize>) -> Self {
        let stop = (*r.end()).saturating_add(1);
        Self::Slice(Slice::new(Some(*r.start()), Some(stop), 1))
    }
}

impl From<RangeToInclusive<isize>> for SliceArg {
    #[inline]
    fn from(r: RangeToInclusive<isize>) -> Self {
        let stop = r.end.saturating_add(1);
        Self::Slice(Slice::new(None, Some(stop), 1))
    }
}

/// Convenience macro for building `&[SliceArg]` expressions:
/// `slice![0..5, .., -1, NewAxis]` or `slice![..; -1]`.
#[macro_export]
macro_rules! slice {
    ($($item:tt),* $(,)?) => {
        &[$($crate::slice_arg!($item)),*]
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! slice_arg {
    (..) => {
        $crate::SliceArg::Slice($crate::Slice::full())
    };
    (...) => {
        $crate::SliceArg::Ellipsis
    };
    (NewAxis) => {
        $crate::SliceArg::NewAxis
    };
    ($e:expr) => {
        $crate::SliceArg::from($e)
    };
}

/// Checked total number of elements for `shape`. Returns `1` for 0-D scalars.
#[inline]
pub fn shape_size(shape: &[usize]) -> Result<usize> {
    let mut total = 1usize;
    for &dim in shape {
        total = total
            .checked_mul(dim)
            .ok_or_else(|| NumPyError::ShapeMismatch(format!("shape {shape:?} overflows usize")))?;
    }
    Ok(total)
}

/// Compute row-major (C-order) element strides for `shape`.
pub fn compute_c_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    let mut strides = vec![0isize; ndim];
    let mut acc = 1isize;
    for i in (0..ndim).rev() {
        strides[i] = acc;
        let dim = shape[i].max(1) as isize;
        acc = acc.saturating_mul(dim);
    }
    strides
}

/// Compute column-major (Fortran-order) element strides for `shape`.
pub fn compute_f_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    let mut strides = vec![0isize; ndim];
    let mut acc = 1isize;
    for i in 0..ndim {
        strides[i] = acc;
        let dim = shape[i].max(1) as isize;
        acc = acc.saturating_mul(dim);
    }
    strides
}

/// Compute strides for `Order::C` or `Order::F` (treating `A`/`K` as `C` for fresh allocations).
#[inline]
pub fn compute_strides(shape: &[usize], order: Order) -> Vec<isize> {
    match order {
        Order::F => compute_f_strides(shape),
        Order::C | Order::A | Order::K => compute_c_strides(shape),
    }
}

/// Normalize a possibly negative axis index against `ndim`.
#[inline]
pub fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let n = ndim as isize;
    let resolved = if axis < 0 { axis + n } else { axis };
    if resolved < 0 || resolved >= n {
        Err(NumPyError::InvalidAxis { axis, ndim })
    } else {
        Ok(resolved as usize)
    }
}

/// Normalize a slice of axes and verify there are no duplicates.
pub fn normalize_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(axes.len());
    for &ax in axes {
        let norm = normalize_axis(ax, ndim)?;
        if out.contains(&norm) {
            return Err(NumPyError::InvalidValue(format!("repeated axis {ax} in axes {axes:?}")));
        }
        out.push(norm);
    }
    Ok(out)
}

/// Normalize an index along a single dimension of size `dim_len`.
#[inline]
pub fn normalize_index(idx: isize, dim_len: usize) -> Result<usize> {
    let n = dim_len as isize;
    let resolved = if idx < 0 { idx + n } else { idx };
    if resolved < 0 || resolved >= n {
        Err(NumPyError::IndexOutOfBounds(format!(
            "index {idx} is out of bounds for axis with size {dim_len}"
        )))
    } else {
        Ok(resolved as usize)
    }
}

/// Broadcast two shapes according to NumPy broadcasting rules.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1usize; ndim];
    for i in 0..ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db {
            out[ndim - 1 - i] = da;
        } else if da == 1 {
            out[ndim - 1 - i] = db;
        } else if db == 1 {
            out[ndim - 1 - i] = da;
        } else {
            return Err(NumPyError::BroadcastError(format!(
                "operands could not be broadcast together with shapes {a:?} {b:?}"
            )));
        }
    }
    Ok(out)
}

/// Broadcast multiple shapes together.
pub fn broadcast_many_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut acc = shapes[0].to_vec();
    for s in &shapes[1..] {
        acc = broadcast_shapes(&acc, s)?;
    }
    Ok(acc)
}

/// Compute strided view strides to broadcast `(src_shape, src_strides)` to `target_shape`.
pub fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    target_shape: &[usize],
) -> Result<Vec<isize>> {
    if src_shape.len() > target_shape.len() {
        return Err(NumPyError::BroadcastError(format!(
            "cannot broadcast shape {src_shape:?} to smaller rank {target_shape:?}"
        )));
    }
    let pad = target_shape.len() - src_shape.len();
    let mut out = vec![0isize; target_shape.len()];
    for i in 0..src_shape.len() {
        let s_dim = src_shape[i];
        let t_dim = target_shape[pad + i];
        if s_dim == t_dim {
            out[pad + i] = src_strides[i];
        } else if s_dim == 1 {
            out[pad + i] = 0;
        } else {
            return Err(NumPyError::BroadcastError(format!(
                "cannot broadcast shape {src_shape:?} to {target_shape:?}"
            )));
        }
    }
    Ok(out)
}

/// Return `true` iff every distinct multi-index within `shape` maps to a unique memory offset.
/// Guarantees zero aliasing for mutable strided views (`ArrayViewMut`).
pub fn check_non_overlapping(shape: &[usize], strides: &[isize]) -> bool {
    if shape.iter().any(|&d| d == 0) {
        return true;
    }
    let mut active: Vec<(usize, usize)> = Vec::with_capacity(shape.len());
    for (&dim, &st) in shape.iter().zip(strides.iter()) {
        if dim > 1 {
            if st == 0 {
                return false;
            }
            active.push((st.unsigned_abs(), dim));
        }
    }
    active.sort_unstable_by_key(|&(abs_st, _)| abs_st);
    let mut max_span = 0usize;
    for (abs_st, dim) in active {
        if abs_st <= max_span {
            return false;
        }
        let Some(add) = (dim - 1).checked_mul(abs_st) else {
            return false;
        };
        let Some(next) = max_span.checked_add(add) else {
            return false;
        };
        max_span = next;
    }
    true
}

/// Check if `(shape, strides)` represents a C-contiguous layout (NumPy semantics: empty or size-1 dims don't break contiguity).
pub fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.is_empty() || shape.iter().any(|&d| d == 0) {
        return true;
    }
    let mut expected = 1isize;
    for (&dim, &st) in shape.iter().zip(strides.iter()).rev() {
        if dim > 1 {
            if st != expected {
                return false;
            }
            expected *= dim as isize;
        }
    }
    true
}

/// Check if `(shape, strides)` represents a Fortran-contiguous layout.
pub fn is_f_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.is_empty() || shape.iter().any(|&d| d == 0) {
        return true;
    }
    let mut expected = 1isize;
    for (&dim, &st) in shape.iter().zip(strides.iter()) {
        if dim > 1 {
            if st != expected {
                return false;
            }
            expected *= dim as isize;
        }
    }
    true
}

/// Resolve a dynamic reshape spec (allowing at most one `-1` dimension) into concrete positive dims.
pub fn resolve_reshape_dims(current_len: usize, spec: &[isize]) -> Result<Vec<usize>> {
    let mut neg_idx: Option<usize> = None;
    let mut known_prod: usize = 1;
    let mut out = vec![0usize; spec.len()];
    for (i, &d) in spec.iter().enumerate() {
        if d == -1 {
            if neg_idx.is_some() {
                return Err(NumPyError::InvalidValue("can only specify one unknown dimension (-1)".into()));
            }
            neg_idx = Some(i);
        } else if d < 0 {
            return Err(NumPyError::InvalidValue(format!("invalid reshape dimension {d}")));
        } else {
            let ud = d as usize;
            out[i] = ud;
            known_prod = known_prod
                .checked_mul(ud)
                .ok_or_else(|| NumPyError::ShapeMismatch("reshape size overflow".into()))?;
        }
    }
    if let Some(idx) = neg_idx {
        if known_prod == 0 {
            return Err(NumPyError::ShapeMismatch(format!(
                "cannot reshape array of size {current_len} into shape {spec:?}"
            )));
        }
        if current_len % known_prod != 0 {
            return Err(NumPyError::ShapeMismatch(format!(
                "cannot reshape array of size {current_len} into shape {spec:?}"
            )));
        }
        out[idx] = current_len / known_prod;
    } else if known_prod != current_len {
        return Err(NumPyError::ShapeMismatch(format!(
            "cannot reshape array of size {current_len} into shape {spec:?}"
        )));
    }
    Ok(out)
}

//! Array manipulation routines (shape views, joining, splitting, tiling, flipping,
//! rolling, rotating, padding, resizing, inserting, deleting, trimming) for both
//! `NdArray<T>` and `DynArray`.

use crate::creation::CreationElement;
use crate::util::{
    RAYON_THRESHOLD, advance_coords_ptr, copy_view_to_uninit, coords_offset, dyn_as_bool,
    finish_c_array, linear_to_coords, DynScalar,
};
use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{
    normalize_axis, shape_size, AlignedBuffer, Casting, DType, DynArray, NdArray, NumPyError,
    Order, Result, Scalar, Slice, SliceArg, dispatch_dyn,
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::mem::MaybeUninit;

// =============================================================================
// 1. Parameter Enums
// =============================================================================

/// Specification of how to split an array along an axis.
#[derive(Debug, Clone, Copy)]
pub enum SplitSpec<'a> {
    /// Split into N equal sections (for `split`/`hsplit`/`vsplit`/`dsplit`) or nearly-equal sections (`array_split`).
    Sections(usize),
    /// Split at the specified slice boundaries (1-D sorted cut indices, negative indices allowed).
    Indices(&'a [isize]),
}

/// Reflection boundary condition for `pad(..., PadMode::Reflect | PadMode::Symmetric)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReflectType {
    /// Standard reflection (`Even`).
    #[default]
    Even,
    /// Odd reflection (`2 * edge - reflected`).
    Odd,
}

/// Padding widths specification for `pad`.
#[derive(Debug, Clone)]
pub enum PadWidth {
    /// Same padding before and after every axis.
    Uniform(usize),
    /// Same `(before, after)` pair applied to every axis.
    Pair(usize, usize),
    /// Explicit `(before, after)` pair per axis (length must equal `arr.ndim()`, or `1`).
    Axes(Vec<(usize, usize)>),
}

impl PadWidth {
    fn resolve(&self, ndim: usize) -> Result<Vec<(usize, usize)>> {
        if ndim == 0 {
            return match self {
                PadWidth::Uniform(_) | PadWidth::Pair(_, _) => Ok(Vec::new()),
                PadWidth::Axes(v) if v.len() <= 1 => Ok(Vec::new()),
                PadWidth::Axes(v) => Err(NumPyError::InvalidValue(format!(
                    "pad_width length {} does not match array ndim 0",
                    v.len()
                ))),
            };
        }
        match self {
            PadWidth::Uniform(w) => Ok(vec![(*w, *w); ndim]),
            PadWidth::Pair(b, a) => Ok(vec![(*b, *a); ndim]),
            PadWidth::Axes(v) => {
                if v.len() == ndim {
                    Ok(v.clone())
                } else if v.len() == 1 {
                    Ok(vec![v[0]; ndim])
                } else {
                    Err(NumPyError::InvalidValue(format!(
                        "pad_width length {} does not match array ndim {ndim}",
                        v.len()
                    )))
                }
            }
        }
    }
}

/// Window length specification for statistical padding modes (`Maximum`, `Mean`, `Median`, `Minimum`).
#[derive(Debug, Clone, Default)]
pub enum StatLength {
    /// Use the full axis extent (`None` in NumPy).
    #[default]
    Full,
    /// Uniform window size at both edges across all axes.
    Uniform(usize),
    /// `(before_len, after_len)` across all axes.
    Pair(usize, usize),
    /// Per-axis optional `(before_len, after_len)`.
    Axes(Vec<(Option<usize>, Option<usize>)>),
}

impl StatLength {
    fn resolve(&self, ndim: usize) -> Result<Vec<(Option<usize>, Option<usize>)>> {
        match self {
            StatLength::Full => Ok(vec![(None, None); ndim]),
            StatLength::Uniform(n) => Ok(vec![(Some(*n), Some(*n)); ndim]),
            StatLength::Pair(b, a) => Ok(vec![(Some(*b), Some(*a)); ndim]),
            StatLength::Axes(v) => {
                if v.len() == ndim {
                    Ok(v.clone())
                } else if v.len() == 1 && ndim > 0 {
                    Ok(vec![v[0]; ndim])
                } else {
                    Err(NumPyError::InvalidValue(format!(
                        "stat_length len {} does not match ndim {ndim}",
                        v.len()
                    )))
                }
            }
        }
    }
}

/// Padding strategy for `pad`.
#[derive(Debug, Clone)]
pub enum PadMode<T> {
    /// Pad with a uniform constant value.
    Constant(T),
    /// Pad with per-axis `(before, after)` constant values.
    ConstantPairs(Vec<(T, T)>),
    /// Replicate edge values.
    Edge,
    /// Linearly ramp between `end_value` and the array edge value.
    LinearRamp(T),
    /// Linearly ramp with per-axis `(end_before, end_after)` values.
    LinearRampPairs(Vec<(T, T)>),
    /// Pad with the maximum along the statistic window.
    Maximum(StatLength),
    /// Pad with the mean along the statistic window.
    Mean(StatLength),
    /// Pad with the median along the statistic window.
    Median(StatLength),
    /// Pad with the minimum along the statistic window.
    Minimum(StatLength),
    /// Mirror values excluding edge element.
    Reflect(ReflectType),
    /// Mirror values including edge element.
    Symmetric(ReflectType),
    /// Circularly wrap values from the opposite end.
    Wrap,
    /// Uninitialized-equivalent zeroed padding.
    Empty,
}

/// Dynamic padding mode for `dyn_pad`.
#[derive(Debug, Clone)]
pub enum DynPadMode {
    Constant(DynScalar),
    ConstantPairs(Vec<(DynScalar, DynScalar)>),
    Edge,
    LinearRamp(DynScalar),
    LinearRampPairs(Vec<(DynScalar, DynScalar)>),
    Maximum(StatLength),
    Mean(StatLength),
    Median(StatLength),
    Minimum(StatLength),
    Reflect(ReflectType),
    Symmetric(ReflectType),
    Wrap,
    Empty,
}

/// Target object specification for `insert`.
#[derive(Debug, Clone)]
pub enum InsertObj<'a> {
    /// Insert before a single index (supports negative indexing; `dim` appends at end).
    Index(isize),
    /// Insert before indices generated by a slice.
    Slice(Slice),
    /// Insert before each specified index.
    Indices(&'a [isize]),
    /// Insert where mask is `true`.
    BoolMask(&'a NdArray<bool>),
}

/// Target object specification for `delete`.
#[derive(Debug, Clone)]
pub enum DeleteObj<'a> {
    /// Delete a single index (supports negative indexing).
    Index(isize),
    /// Delete indices matched by a slice.
    Slice(Slice),
    /// Delete each specified index.
    Indices(&'a [isize]),
    /// Delete where mask is `true`.
    BoolMask(&'a NdArray<bool>),
}

/// Trim ends selection for `trim_zeros`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrimDirection {
    /// Trim both leading and trailing zeros (`"fb"` / `"bf"`).
    #[default]
    Both,
    /// Trim leading zeros only (`"f"`).
    Front,
    /// Trim trailing zeros only (`"b"`).
    Back,
}

impl TrimDirection {
    /// Parses `"fb"`, `"bf"`, `"f"`, `"b"` (case-insensitive) matching NumPy `trim_zeros`.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fb" | "bf" => Ok(Self::Both),
            "f" => Ok(Self::Front),
            "b" => Ok(Self::Back),
            _ => Err(NumPyError::InvalidValue(format!(
                "trim must be one of 'fb', 'bf', 'f', 'b', got '{s}'"
            ))),
        }
    }
}

/// Nested sequence input node for `block`.
#[derive(Debug, Clone)]
pub enum BlockNode<'a, T: Scalar> {
    Array(&'a NdArray<T>),
    List(Vec<BlockNode<'a, T>>),
}

/// Nested sequence input node for `dyn_block`.
#[derive(Debug, Clone)]
pub enum DynBlockNode<'a> {
    Array(&'a DynArray),
    List(Vec<DynBlockNode<'a>>),
}

/// Trait providing comparison/averaging operations needed by `pad` for all 14 scalar dtypes.
pub trait PadElement: CreationElement {
    /// Total ordering for `Minimum` / `Maximum` / `Median` statistics.
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering;
    /// Exact average `(a + b) / 2` in `Self` (matching NumPy round-to-even for integers).
    fn average_two(a: Self, b: Self) -> Self;
    /// Mean over a non-empty `f64` sum / count (round-to-even for integers).
    fn mean_from_sum(sum: Complex64, count: usize) -> Self;
    /// Linear ramp helper `edge + (end_val - edge) * t` (`t` in `[0, 1]`).
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self;
}

impl PadElement for bool {
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
    fn average_two(a: Self, b: Self) -> Self {
        a || b
    }
    fn mean_from_sum(sum: Complex64, _count: usize) -> Self {
        sum.re != 0.0
    }
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
        let a = if end_val { 1.0 } else { 0.0 };
        let b = if edge_val { 1.0 } else { 0.0 };
        (a + (b - a) * frac).round_ties_even() != 0.0
    }
}

macro_rules! impl_pad_int {
    ($($ty:ty),*) => {
        $(
            impl PadElement for $ty {
                fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.cmp(other)
                }
                fn average_two(a: Self, b: Self) -> Self {
                    let avg = ((a as f64) + (b as f64)) * 0.5;
                    avg.round_ties_even() as $ty
                }
                fn mean_from_sum(sum: Complex64, count: usize) -> Self {
                    (sum.re / (count as f64)).round_ties_even() as $ty
                }
                fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
                    let a = end_val as f64;
                    let b = edge_val as f64;
                    (a + (b - a) * frac).floor() as $ty
                }
            }
        )*
    };
}

impl_pad_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl PadElement for f16 {
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
        f64::from(*self).total_cmp(&f64::from(*other))
    }
    fn average_two(a: Self, b: Self) -> Self {
        f16::from_f64((f64::from(a) + f64::from(b)) * 0.5)
    }
    fn mean_from_sum(sum: Complex64, count: usize) -> Self {
        f16::from_f64(sum.re / (count as f64))
    }
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
        let a = f64::from(end_val);
        let b = f64::from(edge_val);
        f16::from_f64(a + (b - a) * frac)
    }
}

impl PadElement for f32 {
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.total_cmp(other)
    }
    fn average_two(a: Self, b: Self) -> Self {
        (a + b) * 0.5
    }
    fn mean_from_sum(sum: Complex64, count: usize) -> Self {
        (sum.re / (count as f64)) as f32
    }
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
        let a = f64::from(end_val);
        let b = f64::from(edge_val);
        (a + (b - a) * frac) as f32
    }
}

impl PadElement for f64 {
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.total_cmp(other)
    }
    fn average_two(a: Self, b: Self) -> Self {
        (a + b) * 0.5
    }
    fn mean_from_sum(sum: Complex64, count: usize) -> Self {
        sum.re / (count as f64)
    }
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
        end_val + (edge_val - end_val) * frac
    }
}

impl PadElement for Complex32 {
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.re
            .total_cmp(&other.re)
            .then_with(|| self.im.total_cmp(&other.im))
    }
    fn average_two(a: Self, b: Self) -> Self {
        (a + b) * 0.5
    }
    fn mean_from_sum(sum: Complex64, count: usize) -> Self {
        let m = sum / (count as f64);
        Complex32::new(m.re as f32, m.im as f32)
    }
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
        let a = Complex64::new(f64::from(end_val.re), f64::from(end_val.im));
        let b = Complex64::new(f64::from(edge_val.re), f64::from(edge_val.im));
        let c = a + (b - a) * frac;
        Complex32::new(c.re as f32, c.im as f32)
    }
}

impl PadElement for Complex64 {
    fn stat_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.re
            .total_cmp(&other.re)
            .then_with(|| self.im.total_cmp(&other.im))
    }
    fn average_two(a: Self, b: Self) -> Self {
        (a + b) * 0.5
    }
    fn mean_from_sum(sum: Complex64, count: usize) -> Self {
        sum / (count as f64)
    }
    fn ramp_lerp(end_val: Self, edge_val: Self, frac: f64) -> Self {
        end_val + (edge_val - end_val) * frac
    }
}

// =============================================================================
// 2. Shape, Axis & View Manipulations (Zero-Copy wherever possible)
// =============================================================================

/// Gives a new shape to an array without changing its data when stride-compatible (zero-copy).
pub fn reshape<T: Scalar>(arr: &NdArray<T>, newshape: &[isize]) -> Result<NdArray<T>> {
    arr.reshape(newshape)
}

/// Reshapes using an explicit element read/write order (`Order::C`, `Order::F`, or `Order::A`).
pub fn reshape_with_order<T: Scalar>(
    arr: &NdArray<T>,
    newshape: &[isize],
    order: Order,
) -> Result<NdArray<T>> {
    match order {
        Order::K => Err(NumPyError::InvalidValue(
            "order 'K' is not permitted for reshaping".into(),
        )),
        Order::F => reshape_f_order(arr, newshape),
        Order::A => {
            if arr.is_f_contiguous() && !arr.is_c_contiguous() {
                reshape_f_order(arr, newshape)
            } else {
                arr.reshape(newshape)
            }
        }
        Order::C => arr.reshape(newshape),
    }
}

fn reshape_f_order<T: Scalar>(arr: &NdArray<T>, newshape: &[isize]) -> Result<NdArray<T>> {
    let f_arr = arr.to_f_contiguous();
    let dims = numpy_types::resolve_reshape_dims(f_arr.len(), newshape)?;
    let mut rev_dims = dims.clone();
    rev_dims.reverse();
    let rev_i: Vec<isize> = rev_dims.iter().map(|&d| d as isize).collect();
    let ft = f_arr.t();
    let reshaped_t = ft.reshape(&rev_i)?;
    let rev_axes: Vec<isize> = (0..dims.len()).rev().map(|a| a as isize).collect();
    reshaped_t.transpose(Some(&rev_axes))
}

/// Returns a contiguous flattened 1-D array, returning a zero-copy view whenever already C-contiguous.
pub fn ravel<T: Scalar>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    Ok(arr.ravel(Order::C))
}

/// `ravel` with explicit `Order`.
pub fn ravel_with_order<T: Scalar>(arr: &NdArray<T>, order: Order) -> Result<NdArray<T>> {
    Ok(arr.ravel(order))
}

/// Always returns an owned flattened 1-D copy (64-byte aligned, Rayon-parallel for large arrays).
pub fn flatten<T: Scalar + Send + Sync>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    flatten_with_order(arr, Order::C)
}

/// Always returns an owned flattened 1-D copy in the specified traversal `Order`.
pub fn flatten_with_order<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    order: Order,
) -> Result<NdArray<T>> {
    let flat_view = arr.ravel(order);
    let mut uninit = AlignedBuffer::<T>::new_uninit(flat_view.len());
    copy_view_to_uninit(&flat_view.view(), uninit.as_mut_slice());
    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &[flat_view.len()])
}

/// Permutes array axes (always zero-copy view).
pub fn transpose<T: Scalar>(arr: &NdArray<T>, axes: Option<&[usize]>) -> Result<NdArray<T>> {
    match axes {
        None => arr.transpose(None),
        Some(ax) => {
            let ax_i: Vec<isize> = ax.iter().map(|&a| a as isize).collect();
            arr.transpose(Some(&ax_i))
        }
    }
}

/// Permutes axes allowing negative axis indices (always zero-copy view).
pub fn transpose_isize<T: Scalar>(
    arr: &NdArray<T>,
    axes: Option<&[isize]>,
) -> Result<NdArray<T>> {
    match axes {
        None => arr.transpose(None),
        Some(ax) => {
            let norm = numpy_types::normalize_axes(ax, arr.ndim())?;
            let norm_i: Vec<isize> = norm.into_iter().map(|a| a as isize).collect();
            arr.transpose(Some(&norm_i))
        }
    }
}

/// Interchanges two axes of an array (always zero-copy view).
pub fn swapaxes<T: Scalar>(arr: &NdArray<T>, axis1: isize, axis2: isize) -> Result<NdArray<T>> {
    arr.swapaxes(axis1, axis2)
}

/// Moves axes of an array to new positions (always zero-copy view).
pub fn moveaxis<T: Scalar>(
    arr: &NdArray<T>,
    source: &[isize],
    destination: &[isize],
) -> Result<NdArray<T>> {
    if source.len() != destination.len() {
        return Err(NumPyError::InvalidValue(format!(
            "source ({}) and destination ({}) must have the same number of elements",
            source.len(),
            destination.len()
        )));
    }
    let ndim = arr.ndim();
    let src = numpy_types::normalize_axes(source, ndim)?;
    let dst = numpy_types::normalize_axes(destination, ndim)?;

    let mut order: Vec<isize> = (0..ndim)
        .filter(|a| !src.contains(a))
        .map(|a| a as isize)
        .collect();
    let mut pairs: Vec<(usize, usize)> = dst.into_iter().zip(src).collect();
    pairs.sort_by_key(|&(d, _)| d);
    for (d, s) in pairs {
        order.insert(d, s as isize);
    }
    arr.transpose(Some(&order))
}

/// Rolls the specified axis backwards until it lies in the `start` position (always zero-copy view).
pub fn rollaxis<T: Scalar>(arr: &NdArray<T>, axis: isize, start: isize) -> Result<NdArray<T>> {
    let ndim = arr.ndim();
    let ax = normalize_axis(axis, ndim)?;
    let mut st = if start < 0 {
        start + (ndim as isize)
    } else {
        start
    };
    if st < 0 || (st as usize) > ndim {
        return Err(NumPyError::InvalidAxis {
            axis: start,
            ndim,
        });
    }
    let mut st_u = st as usize;
    if ax < st_u {
        st -= 1;
        st_u = st as usize;
    }
    if ax == st_u {
        return Ok(arr.clone());
    }
    let mut axes: Vec<isize> = (0..ndim as isize).collect();
    axes.remove(ax);
    axes.insert(st_u, ax as isize);
    arr.transpose(Some(&axes))
}

/// Removes axes of length one from `arr` (always zero-copy view).
pub fn squeeze<T: Scalar>(arr: &NdArray<T>, axis: Option<&[isize]>) -> Result<NdArray<T>> {
    match axis {
        None => arr.squeeze(None),
        Some(ax) => {
            if ax.is_empty() {
                return Ok(arr.clone());
            }
            let norm = numpy_types::normalize_axes(ax, arr.ndim())?;
            for &a in &norm {
                if arr.shape()[a] != 1 {
                    return Err(NumPyError::InvalidValue(format!(
                        "cannot select an axis to squeeze out which has size {} != 1 (axis {a})",
                        arr.shape()[a]
                    )));
                }
            }
            let args: Vec<SliceArg> = (0..arr.ndim())
                .map(|d| {
                    if norm.contains(&d) {
                        SliceArg::Index(0)
                    } else {
                        SliceArg::Slice(Slice::full())
                    }
                })
                .collect();
            arr.slice(&args)
        }
    }
}

/// Expands the shape of an array by inserting new length-1 axes (always zero-copy view).
pub fn expand_dims<T: Scalar>(arr: &NdArray<T>, axes: &[isize]) -> Result<NdArray<T>> {
    if axes.is_empty() {
        return Ok(arr.clone());
    }
    let out_ndim = arr.ndim() + axes.len();
    let norm = numpy_types::normalize_axes(axes, out_ndim)?;
    let new_set: HashSet<usize> = norm.into_iter().collect();
    let mut args = Vec::with_capacity(out_ndim);
    for pos in 0..out_ndim {
        if new_set.contains(&pos) {
            args.push(SliceArg::NewAxis);
        } else {
            args.push(SliceArg::Slice(Slice::full()));
        }
    }
    arr.slice(&args)
}

/// Converts input to at least 1-D zero-copy view.
pub fn atleast_1d<T: Scalar>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    if arr.ndim() == 0 {
        arr.reshape(&[1])
    } else {
        Ok(arr.clone())
    }
}

/// Converts input to at least 2-D zero-copy view (`(N,)` -> `(1, N)`).
pub fn atleast_2d<T: Scalar>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    match arr.ndim() {
        0 => arr.reshape(&[1, 1]),
        1 => expand_dims(arr, &[0]),
        _ => Ok(arr.clone()),
    }
}

/// Converts input to at least 3-D zero-copy view (`(N,)` -> `(1, N, 1)`, `(M, N)` -> `(M, N, 1)`).
pub fn atleast_3d<T: Scalar>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    match arr.ndim() {
        0 => arr.reshape(&[1, 1, 1]),
        1 => expand_dims(arr, &[0, 2]),
        2 => expand_dims(arr, &[2]),
        _ => Ok(arr.clone()),
    }
}

// =============================================================================
// 3. Joining Arrays (concatenate/concat, stack, vstack, hstack, dstack, column_stack, block)
// =============================================================================

/// Joins a sequence of arrays along an existing axis (or flattened if `axis == None`).
/// Uses 64-byte `AlignedUninit` and Rayon parallel copies above `RAYON_THRESHOLD`.
pub fn concatenate<T: Scalar + Send + Sync>(
    arrays: &[&NdArray<T>],
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue(
            "need at least one array to concatenate".into(),
        ));
    }
    let ax = match axis {
        None => {
            let ravels: Vec<NdArray<T>> = arrays.iter().map(|a| a.ravel(Order::C)).collect();
            let refs: Vec<&NdArray<T>> = ravels.iter().collect();
            return concatenate(&refs, Some(0));
        }
        Some(a) => a,
    };

    let ndim = arrays[0].ndim();
    if ndim == 0 {
        return Err(NumPyError::InvalidValue(
            "zero-dimensional arrays cannot be concatenated".into(),
        ));
    }
    let norm_ax = normalize_axis(ax, ndim)?;

    let mut out_shape = arrays[0].shape().to_vec();
    for (idx, &a) in arrays.iter().enumerate().skip(1) {
        if a.ndim() != ndim {
            return Err(NumPyError::ShapeMismatch(format!(
                "all input arrays must have the same number of dimensions, array 0 has {ndim}-D, array {idx} has {}-D",
                a.ndim()
            )));
        }
        for d in 0..ndim {
            if d == norm_ax {
                out_shape[norm_ax] = out_shape[norm_ax].checked_add(a.shape()[d]).ok_or_else(|| {
                    NumPyError::ShapeMismatch("concatenated dimension overflows usize".into())
                })?;
            } else if a.shape()[d] != out_shape[d] {
                return Err(NumPyError::ShapeMismatch(format!(
                    "all the input array dimensions except for the concatenation axis must match exactly, \
                     along dimension {d}, array 0 has size {} and array {idx} has size {}",
                    out_shape[d],
                    a.shape()[d]
                )));
            }
        }
    }

    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let outer: usize = out_shape[..norm_ax].iter().product();
    let inner: usize = out_shape[norm_ax + 1..].iter().product();
    let out_mid = out_shape[norm_ax] * inner;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    if outer == 1 {
        let mut offset = 0usize;
        for &a in arrays {
            let len = a.len();
            copy_view_to_uninit(&a.view(), &mut dst[offset..offset + len]);
            offset += len;
        }
    } else {
        // Ensure each source is readable as contiguous blocks of length `a.shape()[norm_ax] * inner`.
        let c_arrays: Vec<NdArray<T>> = arrays.iter().map(|a| a.to_c_contiguous()).collect();

        let mut mid_offsets = Vec::with_capacity(arrays.len());
        let mut acc = 0usize;
        for ca in &c_arrays {
            let m = ca.shape()[norm_ax] * inner;
            mid_offsets.push((acc, m, ca.as_slice().unwrap_or(&[])));
            acc += m;
        }

        if total >= RAYON_THRESHOLD {
            dst.par_chunks_mut(out_mid)
                .enumerate()
                .for_each(|(o, out_row)| {
                    for &(mid_off, mid_len, src_sl) in &mid_offsets {
                        if mid_len == 0 {
                            continue;
                        }
                        let src_block = &src_sl[o * mid_len..(o + 1) * mid_len];
                        let dst_block = &mut out_row[mid_off..mid_off + mid_len];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src_block.as_ptr(),
                                dst_block.as_mut_ptr() as *mut T,
                                mid_len,
                            );
                        }
                    }
                });
        } else {
            for (o, out_row) in dst.chunks_mut(out_mid).enumerate() {
                for &(mid_off, mid_len, src_sl) in &mid_offsets {
                    if mid_len == 0 {
                        continue;
                    }
                    let src_block = &src_sl[o * mid_len..(o + 1) * mid_len];
                    let dst_block = &mut out_row[mid_off..mid_off + mid_len];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_block.as_ptr(),
                            dst_block.as_mut_ptr() as *mut T,
                            mid_len,
                        );
                    }
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Alias for `concatenate` (NumPy 2.0 `np.concat`).
pub fn concat<T: Scalar + Send + Sync>(
    arrays: &[&NdArray<T>],
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    concatenate(arrays, axis)
}

/// Joins a sequence of identically shaped arrays along a new `axis`.
pub fn stack<T: Scalar + Send + Sync>(arrays: &[&NdArray<T>], axis: isize) -> Result<NdArray<T>> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to stack".into()));
    }
    let expected = arrays[0].shape();
    for &a in arrays.iter().skip(1) {
        if a.shape() != expected {
            return Err(NumPyError::ShapeMismatch(
                "all input arrays must have the same shape".into(),
            ));
        }
    }
    let out_ndim = expected.len() + 1;
    let norm_ax = normalize_axis(axis, out_ndim)?;
    let ax_i = norm_ax as isize;
    let expanded: Vec<NdArray<T>> = arrays
        .iter()
        .map(|a| expand_dims(a, &[ax_i]))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<T>> = expanded.iter().collect();
    concatenate(&refs, Some(ax_i))
}

/// Stacks arrays sequentially vertically (row-wise, axis 0 after `atleast_2d`).
pub fn vstack<T: Scalar + Send + Sync>(arrays: &[&NdArray<T>]) -> Result<NdArray<T>> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to vstack".into()));
    }
    let promoted: Vec<NdArray<T>> = arrays
        .iter()
        .map(|a| atleast_2d(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<T>> = promoted.iter().collect();
    concatenate(&refs, Some(0))
}

/// Stacks arrays sequentially horizontally (column-wise, axis 0 if 1-D else axis 1).
pub fn hstack<T: Scalar + Send + Sync>(arrays: &[&NdArray<T>]) -> Result<NdArray<T>> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to hstack".into()));
    }
    let promoted: Vec<NdArray<T>> = arrays
        .iter()
        .map(|a| atleast_1d(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<T>> = promoted.iter().collect();
    if promoted[0].ndim() == 1 {
        concatenate(&refs, Some(0))
    } else {
        concatenate(&refs, Some(1))
    }
}

/// Stacks arrays sequentially depth-wise along the 3rd axis (after `atleast_3d`).
pub fn dstack<T: Scalar + Send + Sync>(arrays: &[&NdArray<T>]) -> Result<NdArray<T>> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to dstack".into()));
    }
    let promoted: Vec<NdArray<T>> = arrays
        .iter()
        .map(|a| atleast_3d(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<T>> = promoted.iter().collect();
    concatenate(&refs, Some(2))
}

/// Stacks 1-D arrays as columns into a 2-D array.
pub fn column_stack<T: Scalar + Send + Sync>(arrays: &[&NdArray<T>]) -> Result<NdArray<T>> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue(
            "need at least one array to column_stack".into(),
        ));
    }
    let mut prepared = Vec::with_capacity(arrays.len());
    for &a in arrays {
        if a.ndim() < 2 {
            let a1 = atleast_1d(a)?;
            prepared.push(expand_dims(&a1, &[1])?);
        } else {
            prepared.push(a.clone());
        }
    }
    let refs: Vec<&NdArray<T>> = prepared.iter().collect();
    concatenate(&refs, Some(1))
}

fn inspect_block_tree<T: Scalar>(
    node: &BlockNode<'_, T>,
    depth: usize,
    max_list_depth: &mut usize,
    max_arr_ndim: &mut usize,
) -> Result<()> {
    match node {
        BlockNode::Array(a) => {
            *max_list_depth = (*max_list_depth).max(depth);
            *max_arr_ndim = (*max_arr_ndim).max(a.ndim());
            Ok(())
        }
        BlockNode::List(items) => {
            if items.is_empty() {
                return Err(NumPyError::InvalidValue(format!(
                    "List at depth {depth} cannot be empty"
                )));
            }
            for item in items {
                inspect_block_tree(item, depth + 1, max_list_depth, max_arr_ndim)?;
            }
            Ok(())
        }
    }
}

fn prepend_ones_to_ndim<T: Scalar>(arr: &NdArray<T>, target_ndim: usize) -> Result<NdArray<T>> {
    if arr.ndim() >= target_ndim {
        return Ok(arr.clone());
    }
    let add = target_ndim - arr.ndim();
    let new_axes: Vec<isize> = (0..add as isize).collect();
    expand_dims(arr, &new_axes)
}

fn eval_block_node<T: Scalar + Send + Sync>(
    node: &BlockNode<'_, T>,
    depth: usize,
    max_list_depth: usize,
    final_ndim: usize,
) -> Result<NdArray<T>> {
    match node {
        BlockNode::Array(a) => prepend_ones_to_ndim(a, final_ndim),
        BlockNode::List(items) => {
            let mut sub_arrays = Vec::with_capacity(items.len());
            for item in items {
                sub_arrays.push(eval_block_node(item, depth + 1, max_list_depth, final_ndim)?);
            }
            let axis = (final_ndim - max_list_depth + depth) as isize;
            let refs: Vec<&NdArray<T>> = sub_arrays.iter().collect();
            concatenate(&refs, Some(axis))
        }
    }
}

/// Assembles an nd-array from nested lists of blocks (`BlockNode`).
pub fn block<T: Scalar + Send + Sync>(root: &BlockNode<'_, T>) -> Result<NdArray<T>> {
    let mut max_list_depth = 0usize;
    let mut max_arr_ndim = 0usize;
    inspect_block_tree(root, 0, &mut max_list_depth, &mut max_arr_ndim)?;
    let final_ndim = max_list_depth.max(max_arr_ndim);
    eval_block_node(root, 0, max_list_depth, final_ndim)
}

// =============================================================================
// 4. Splitting Arrays (ALL ZERO-COPY VIEWS)
// =============================================================================

fn slice_along_axis<T: Scalar>(
    arr: &NdArray<T>,
    axis: usize,
    start: usize,
    stop: usize,
) -> Result<NdArray<T>> {
    let mut args = vec![SliceArg::Slice(Slice::full()); arr.ndim()];
    args[axis] = SliceArg::Slice(Slice::new(Some(start as isize), Some(stop as isize), 1));
    arr.slice(&args)
}

fn split_core<T: Scalar>(
    arr: &NdArray<T>,
    spec: SplitSpec<'_>,
    axis: isize,
    require_equal: bool,
) -> Result<Vec<NdArray<T>>> {
    if arr.ndim() == 0 {
        return Err(NumPyError::InvalidValue("cannot split a 0-D array".into()));
    }
    let norm_ax = normalize_axis(axis, arr.ndim())?;
    let dim = arr.shape()[norm_ax];

    match spec {
        SplitSpec::Sections(n) => {
            if n == 0 {
                return Err(NumPyError::InvalidValue(
                    "number sections must be larger than 0".into(),
                ));
            }
            if require_equal && dim % n != 0 {
                return Err(NumPyError::ShapeMismatch(format!(
                    "array split does not result in an equal division ({dim} not divisible by {n})"
                )));
            }
            let neach = dim / n;
            let extras = dim % n;
            let mut out = Vec::with_capacity(n);
            let mut cur = 0usize;
            for i in 0..n {
                let size = neach + usize::from(i < extras);
                out.push(slice_along_axis(arr, norm_ax, cur, cur + size)?);
                cur += size;
            }
            Ok(out)
        }
        SplitSpec::Indices(idxs) => {
            let mut bounds = Vec::with_capacity(idxs.len() + 2);
            bounds.push(0usize);
            for &raw in idxs {
                let norm = if raw < 0 {
                    (dim as isize + raw).max(0) as usize
                } else {
                    (raw as usize).min(dim)
                };
                bounds.push(norm);
            }
            bounds.push(dim);
            let mut out = Vec::with_capacity(bounds.len() - 1);
            for w in bounds.windows(2) {
                let start = w[0];
                let stop = w[1].max(start);
                out.push(slice_along_axis(arr, norm_ax, start, stop)?);
            }
            Ok(out)
        }
    }
}

/// Splits an array into multiple sub-arrays as zero-copy views.
pub fn split<T: Scalar>(
    arr: &NdArray<T>,
    spec: SplitSpec<'_>,
    axis: isize,
) -> Result<Vec<NdArray<T>>> {
    split_core(arr, spec, axis, true)
}

/// Splits an array into multiple sub-arrays (allowing unequal sections), zero-copy views.
pub fn array_split<T: Scalar>(
    arr: &NdArray<T>,
    spec: SplitSpec<'_>,
    axis: isize,
) -> Result<Vec<NdArray<T>>> {
    split_core(arr, spec, axis, false)
}

/// Splits an array horizontally (axis 0 for 1-D, axis 1 for >=2-D), zero-copy views.
pub fn hsplit<T: Scalar>(arr: &NdArray<T>, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>> {
    match arr.ndim() {
        0 => Err(NumPyError::InvalidValue("cannot hsplit a 0-D array".into())),
        1 => split(arr, spec, 0),
        _ => split(arr, spec, 1),
    }
}

/// Splits an array vertically (axis 0, requires >=2-D), zero-copy views.
pub fn vsplit<T: Scalar>(arr: &NdArray<T>, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>> {
    if arr.ndim() < 2 {
        return Err(NumPyError::InvalidValue(
            "vsplit only works on arrays of 2 or more dimensions".into(),
        ));
    }
    split(arr, spec, 0)
}

/// Splits an array depth-wise (axis 2, requires >=3-D), zero-copy views.
pub fn dsplit<T: Scalar>(arr: &NdArray<T>, spec: SplitSpec<'_>) -> Result<Vec<NdArray<T>>> {
    if arr.ndim() < 3 {
        return Err(NumPyError::InvalidValue(
            "dsplit only works on arrays of 3 or more dimensions".into(),
        ));
    }
    split(arr, spec, 2)
}

/// Unstacks an array along `axis` into a list of zero-copy `(N-1)-D` views.
pub fn unstack<T: Scalar>(arr: &NdArray<T>, axis: isize) -> Result<Vec<NdArray<T>>> {
    if arr.ndim() == 0 {
        return Err(NumPyError::InvalidValue(
            "Input array must be at least 1-d for unstack".into(),
        ));
    }
    let norm_ax = normalize_axis(axis, arr.ndim())?;
    let dim = arr.shape()[norm_ax];
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        out.push(arr.index_axis(norm_ax as isize, i as isize)?);
    }
    Ok(out)
}

// =============================================================================
// 5. Tiling & Repeating
// =============================================================================

/// Constructs an array by repeating `arr` the number of times given by `reps`.
pub fn tile<T: Scalar + Send + Sync>(arr: &NdArray<T>, reps: &[usize]) -> Result<NdArray<T>> {
    if reps.is_empty() {
        return Ok(arr.deep_copy(Order::C));
    }
    let d = reps.len();
    let arr_ndim = arr.ndim();
    let out_ndim = d.max(arr_ndim);

    let mut padded_shape = vec![1usize; out_ndim];
    let s_off = out_ndim - arr_ndim;
    padded_shape[s_off..].copy_from_slice(arr.shape());

    let mut padded_reps = vec![1usize; out_ndim];
    let r_off = out_ndim - d;
    padded_reps[r_off..].copy_from_slice(reps);

    let mut out_shape = Vec::with_capacity(out_ndim);
    for i in 0..out_ndim {
        let dim = padded_shape[i].checked_mul(padded_reps[i]).ok_or_else(|| {
            NumPyError::ShapeMismatch("tile: output dimension overflows usize".into())
        })?;
        out_shape.push(dim);
    }

    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let padded_shape_i: Vec<isize> = padded_shape.iter().map(|&x| x as isize).collect();
    let reshaped = arr.reshape(&padded_shape_i)?;

    let mut b_in = Vec::with_capacity(out_ndim * 2);
    let mut b_out = Vec::with_capacity(out_ndim * 2);
    for i in 0..out_ndim {
        b_in.push(1isize);
        b_in.push(padded_shape[i] as isize);
        b_out.push(padded_reps[i]);
        b_out.push(padded_shape[i]);
    }
    let expanded = reshaped.reshape(&b_in)?;
    let bcast = expanded.broadcast_to(&b_out)?;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    copy_view_to_uninit(&bcast.view(), uninit.as_mut_slice());
    let buf = unsafe { uninit.assume_init() };
    let tiled_c = finish_c_array(buf, &b_out)?;
    let out_shape_i: Vec<isize> = out_shape.iter().map(|&x| x as isize).collect();
    tiled_c.reshape(&out_shape_i)
}

/// Repeats elements of an array along an `axis` (or flattened if `axis == None`).
pub fn repeat<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    repeats: &[usize],
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    if axis.is_none() {
        let flat = arr.ravel(Order::C);
        return repeat(&flat, repeats, Some(0));
    }
    let ax = axis.unwrap_or(0);
    if arr.ndim() == 0 {
        let scalar_1d = arr.reshape(&[1])?;
        return repeat(&scalar_1d, repeats, Some(0));
    }
    let norm_ax = normalize_axis(ax, arr.ndim())?;
    let dim = arr.shape()[norm_ax];

    let rep_counts: Vec<usize> = if repeats.len() == 1 {
        vec![repeats[0]; dim]
    } else if repeats.len() == dim {
        repeats.to_vec()
    } else {
        return Err(NumPyError::InvalidValue(format!(
            "repeats length {} could not be broadcast with axis size {dim}",
            repeats.len()
        )));
    };

    let out_dim: usize = rep_counts.iter().try_fold(0usize, |acc, &r| {
        acc.checked_add(r)
            .ok_or_else(|| NumPyError::ShapeMismatch("repeat output dimension overflows".into()))
    })?;
    let mut out_shape = arr.shape().to_vec();
    out_shape[norm_ax] = out_dim;
    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let c_arr = arr.to_c_contiguous();
    let src_sl = c_arr.as_slice().unwrap_or(&[]);
    let outer: usize = out_shape[..norm_ax].iter().product();
    let inner: usize = out_shape[norm_ax + 1..].iter().product();
    let src_row_len = dim * inner;
    let dst_row_len = out_dim * inner;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    let copy_row = |src_row: &[T], dst_row: &mut [MaybeUninit<T>]| {
        let mut dst_pos = 0usize;
        for (i, &r) in rep_counts.iter().enumerate() {
            if r == 0 {
                continue;
            }
            let elem_slice = &src_row[i * inner..(i + 1) * inner];
            if inner == 1 {
                let val = elem_slice[0];
                for slot in &mut dst_row[dst_pos..dst_pos + r] {
                    slot.write(val);
                }
                dst_pos += r;
            } else {
                for _ in 0..r {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            elem_slice.as_ptr(),
                            dst_row[dst_pos..].as_mut_ptr() as *mut T,
                            inner,
                        );
                    }
                    dst_pos += inner;
                }
            }
        }
    };

    if total >= RAYON_THRESHOLD && outer > 1 {
        dst.par_chunks_mut(dst_row_len)
            .zip(src_sl.par_chunks(src_row_len))
            .for_each(|(d, s)| copy_row(s, d));
    } else if dst_row_len > 0 {
        for (d, s) in dst.chunks_mut(dst_row_len).zip(src_sl.chunks(src_row_len)) {
            copy_row(s, d);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

// =============================================================================
// 6. Flipping, Rotating & Rolling
// =============================================================================

/// Reverses the order of elements along the specified axes (or all axes when `axis == None`).
/// Always returns a zero-copy view.
pub fn flip<T: Scalar>(arr: &NdArray<T>, axis: Option<&[isize]>) -> Result<NdArray<T>> {
    arr.flip(axis)
}

/// Flips an array in the left/right direction (along axis 1, zero-copy view). Requires `ndim >= 2`.
pub fn fliplr<T: Scalar>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    if arr.ndim() < 2 {
        return Err(NumPyError::InvalidValue(format!(
            "Input must be >= 2-d for fliplr, got {}-D",
            arr.ndim()
        )));
    }
    arr.flip(Some(&[1]))
}

/// Flips an array in the up/down direction (along axis 0, zero-copy view). Requires `ndim >= 1`.
pub fn flipud<T: Scalar>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    if arr.ndim() < 1 {
        return Err(NumPyError::InvalidValue("Input must be >= 1-d for flipud".into()));
    }
    arr.flip(Some(&[0]))
}

/// Rotates an array by 90 degrees `k` times in the plane specified by `axes` (always zero-copy view).
pub fn rot90<T: Scalar>(
    arr: &NdArray<T>,
    k: isize,
    axes: (isize, isize),
) -> Result<NdArray<T>> {
    let ndim = arr.ndim();
    if ndim < 2 {
        return Err(NumPyError::InvalidValue(format!(
            "Input must be >= 2-d for rot90, got {ndim}-D"
        )));
    }
    let ax0 = normalize_axis(axes.0, ndim)?;
    let ax1 = normalize_axis(axes.1, ndim)?;
    if ax0 == ax1 {
        return Err(NumPyError::InvalidValue("Axes must be different".into()));
    }
    let k_mod = k.rem_euclid(4);
    match k_mod {
        0 => Ok(arr.clone()),
        1 => arr
            .flip(Some(&[ax1 as isize]))?
            .swapaxes(ax0 as isize, ax1 as isize),
        2 => arr.flip(Some(&[ax0 as isize, ax1 as isize])),
        3 => arr
            .swapaxes(ax0 as isize, ax1 as isize)?
            .flip(Some(&[ax1 as isize])),
        _ => unreachable!(),
    }
}

/// Rolls array elements along a given axis (or flattened array when `axis == None`).
pub fn roll<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    shift: isize,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    roll_axes(arr, &[shift], axis.as_ref().map(std::slice::from_ref))
}

/// Rolls array elements along multiple axes with per-axis shifts.
pub fn roll_axes<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    shifts: &[isize],
    axes: Option<&[isize]>,
) -> Result<NdArray<T>> {
    if axes.is_none() {
        if shifts.len() != 1 {
            return Err(NumPyError::InvalidValue(
                "shift must be an integer when axis is None".into(),
            ));
        }
        let flat = arr.ravel(Order::C);
        let rolled = roll_axes(&flat, shifts, Some(&[0]))?;
        let shape_i: Vec<isize> = arr.shape().iter().map(|&d| d as isize).collect();
        return rolled.reshape(&shape_i);
    }

    let ax_raw = axes.unwrap_or(&[]);
    if arr.ndim() == 0 {
        if ax_raw.is_empty() {
            return Ok(arr.deep_copy(Order::C));
        }
        let _ = normalize_axis(ax_raw[0], 0)?;
    }

    let broadcast_shifts: Vec<isize> = if shifts.len() == 1 && !ax_raw.is_empty() {
        vec![shifts[0]; ax_raw.len()]
    } else if shifts.len() == ax_raw.len() {
        shifts.to_vec()
    } else {
        return Err(NumPyError::InvalidValue(
            "shifts and axes must have the same length".into(),
        ));
    };

    let ndim = arr.ndim();
    let shape = arr.shape();
    let mut net_shifts = vec![0usize; ndim];
    for (&raw_ax, &sh) in ax_raw.iter().zip(broadcast_shifts.iter()) {
        let ax = normalize_axis(raw_ax, ndim)?;
        let dim = shape[ax];
        if dim > 0 {
            let s_mod = sh.rem_euclid(dim as isize) as usize;
            net_shifts[ax] = (net_shifts[ax] + s_mod) % dim;
        }
    }

    if arr.is_empty() {
        return Ok(arr.deep_copy(Order::C));
    }

    if net_shifts.iter().all(|&s| s == 0) {
        return Ok(arr.deep_copy(Order::C));
    }

    let total = arr.len();
    // Fast path for 1-D or single-axis contiguous roll
    if ndim == 1 && arr.is_c_contiguous() {
        let n = shape[0];
        let k = net_shifts[0];
        let src = arr.as_slice().unwrap_or(&[]);
        let mut uninit = AlignedBuffer::<T>::new_uninit(n);
        let dst = uninit.as_mut_slice();
        unsafe {
            std::ptr::copy_nonoverlapping(
                src[n - k..].as_ptr(),
                dst[..k].as_mut_ptr() as *mut T,
                k,
            );
            std::ptr::copy_nonoverlapping(
                src[..n - k].as_ptr(),
                dst[k..].as_mut_ptr() as *mut T,
                n - k,
            );
        }
        let buf = unsafe { uninit.assume_init() };
        return finish_c_array(buf, shape);
    }

    let view = arr.view();
    let strides = view.strides();
    let base_addr = view.as_ptr() as usize;
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    let eval_coords = |dst_coords: &[usize]| -> T {
        let mut off = 0isize;
        for d in 0..ndim {
            let dim = shape[d];
            let s_idx = (dst_coords[d] + dim - net_shifts[d]) % dim;
            off += (s_idx as isize) * strides[d];
        }
        unsafe { *((base_addr as *const T).offset(off)) }
    };

    if total >= RAYON_THRESHOLD {
        const CHUNK: usize = 2048;
        dst.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * CHUNK;
                let mut coords = vec![0usize; ndim];
                linear_to_coords(start, shape, &mut coords);
                let chunk_len = chunk.len();
                for (i, slot) in chunk.iter_mut().enumerate() {
                    slot.write(eval_coords(&coords));
                    if i + 1 < chunk_len {
                        for d in (0..ndim).rev() {
                            coords[d] += 1;
                            if coords[d] < shape[d] {
                                break;
                            }
                            coords[d] = 0;
                        }
                    }
                }
            });
    } else {
        let mut coords = vec![0usize; ndim];
        for (i, slot) in dst.iter_mut().enumerate() {
            slot.write(eval_coords(&coords));
            if i + 1 < total {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < shape[d] {
                        break;
                    }
                    coords[d] = 0;
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, shape)
}

// =============================================================================
// 7. Padding (`pad`)
// =============================================================================

fn reflect_index(pos: isize, dim: usize, include_edge: bool) -> usize {
    if dim <= 1 {
        return 0;
    }
    if include_edge {
        let period = (2 * dim) as isize;
        let m = if pos < 0 {
            (-pos - 1).rem_euclid(period)
        } else {
            pos.rem_euclid(period)
        } as usize;
        if m < dim {
            m
        } else {
            2 * dim - 1 - m
        }
    } else {
        let period = (2 * (dim - 1)) as isize;
        let m = pos.rem_euclid(period) as usize;
        if m < dim {
            m
        } else {
            2 * (dim - 1) - m
        }
    }
}

fn fill_odd_reflect<T: CreationElement>(
    line: &[T],
    out: &mut [MaybeUninit<T>],
    before: usize,
    after: usize,
    include_edge: bool,
) {
    let dim = line.len();
    for i in 0..dim {
        out[before + i].write(line[i]);
    }
    if dim == 1 && include_edge {
        // Odd symmetric with size 1: edge - (val - edge) = edge
        for slot in &mut out[..before] {
            slot.write(line[0]);
        }
        for slot in &mut out[before + dim..] {
            slot.write(line[0]);
        }
        return;
    }
    if dim == 1 {
        for slot in &mut out[..before] {
            slot.write(line[0]);
        }
        for slot in &mut out[before + dim..] {
            slot.write(line[0]);
        }
        return;
    }
    let two = T::one().elem_add(T::one());
    // Fill left iteratively outward so pads larger than `dim` reflect cleanly across the boundary
    let mut left_done = 0usize;
    while left_done < before {
        let edge_idx = before - left_done;
        let edge_val = unsafe { out[edge_idx].assume_init() };
        let chunk = (dim - usize::from(!include_edge)).min(before - left_done);
        let src_start = usize::from(!include_edge);
        for c in 0..chunk {
            let ref_val = unsafe { out[edge_idx + src_start + c].assume_init() };
            let val = two.elem_mul(edge_val).elem_sub(ref_val);
            out[edge_idx - 1 - c].write(val);
        }
        left_done += chunk;
    }

    let mut right_done = 0usize;
    while right_done < after {
        let edge_idx = before + dim - 1 + right_done;
        let edge_val = unsafe { out[edge_idx].assume_init() };
        let chunk = (dim - usize::from(!include_edge)).min(after - right_done);
        let src_off = usize::from(!include_edge);
        for c in 0..chunk {
            let ref_val = unsafe { out[edge_idx - src_off - c].assume_init() };
            let val = two.elem_mul(edge_val).elem_sub(ref_val);
            out[edge_idx + 1 + c].write(val);
        }
        right_done += chunk;
    }
}

fn compute_stat_1d<T: PadElement>(slice: &[T], mode_kind: u8) -> T {
    // 0: max, 1: min, 2: mean, 3: median
    match mode_kind {
        0 => {
            let mut best = slice[0];
            for &v in &slice[1..] {
                if v.stat_cmp(&best) == std::cmp::Ordering::Greater {
                    best = v;
                }
            }
            best
        }
        1 => {
            let mut best = slice[0];
            for &v in &slice[1..] {
                if v.stat_cmp(&best) == std::cmp::Ordering::Less {
                    best = v;
                }
            }
            best
        }
        2 => {
            let mut sum = Complex64::new(0.0, 0.0);
            for &v in slice {
                sum += v.to_c64();
            }
            T::mean_from_sum(sum, slice.len())
        }
        3 => {
            let mut tmp = slice.to_vec();
            let mid = tmp.len() / 2;
            tmp.sort_by(|a, b| a.stat_cmp(b));
            if tmp.len() % 2 == 1 {
                tmp[mid]
            } else {
                T::average_two(tmp[mid - 1], tmp[mid])
            }
        }
        _ => unreachable!(),
    }
}

fn pad_single_axis<T: PadElement>(
    arr: &NdArray<T>,
    axis: usize,
    before: usize,
    after: usize,
    mode: &PadMode<T>,
    ax_idx: usize,
    stat_lengths: &[(Option<usize>, Option<usize>)],
) -> Result<NdArray<T>> {
    if before == 0 && after == 0 {
        return Ok(arr.clone());
    }
    let shape = arr.shape();
    let dim_in = shape[axis];
    let dim_out = before
        .checked_add(dim_in)
        .and_then(|v| v.checked_add(after))
        .ok_or_else(|| NumPyError::ShapeMismatch("padded dimension overflows usize".into()))?;

    let mut out_shape = shape.to_vec();
    out_shape[axis] = dim_out;
    let total = shape_size(&out_shape)?;

    if dim_in == 0
        && (before > 0 || after > 0)
        && !matches!(
            mode,
            PadMode::Constant(_) | PadMode::ConstantPairs(_) | PadMode::Empty
        )
    {
        return Err(NumPyError::InvalidValue(
            "can't extend empty axis using non-constant padding mode".into(),
        ));
    }

    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let c_arr = arr.to_c_contiguous();
    let src_sl = c_arr.as_slice().unwrap_or(&[]);

    let src_block_len = dim_in * inner;
    let dst_block_len = dim_out * inner;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    let (c_before, c_after) = match mode {
        PadMode::Constant(v) => (*v, *v),
        PadMode::ConstantPairs(pairs) => {
            if pairs.len() == 1 {
                pairs[0]
            } else if ax_idx < pairs.len() {
                pairs[ax_idx]
            } else {
                return Err(NumPyError::InvalidValue(
                    "constant_values length does not match array ndim".into(),
                ));
            }
        }
        PadMode::LinearRamp(v) => (*v, *v),
        PadMode::LinearRampPairs(pairs) => {
            if pairs.len() == 1 {
                pairs[0]
            } else if ax_idx < pairs.len() {
                pairs[ax_idx]
            } else {
                return Err(NumPyError::InvalidValue(
                    "end_values length does not match array ndim".into(),
                ));
            }
        }
        PadMode::Empty => (T::zero(), T::zero()),
        _ => (T::zero(), T::zero()),
    };

    let (sl_before, sl_after) = if ax_idx < stat_lengths.len() {
        stat_lengths[ax_idx]
    } else {
        (None, None)
    };
    if matches!(sl_before, Some(0)) || matches!(sl_after, Some(0)) {
        return Err(NumPyError::InvalidValue(
            "stat_length must be greater than 0".into(),
        ));
    }

    let process_outer_chunk = |s_blk: &[T], d_blk: &mut [MaybeUninit<T>]| {
        // Copy middle intact region first (unless Odd reflection fills in place cleanly)
        match mode {
            PadMode::Constant(_) | PadMode::ConstantPairs(_) | PadMode::Empty => {
                for b in 0..before {
                    for j in 0..inner {
                        d_blk[b * inner + j].write(c_before);
                    }
                }
                if src_block_len > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s_blk.as_ptr(),
                            d_blk[before * inner..].as_mut_ptr() as *mut T,
                            src_block_len,
                        );
                    }
                }
                for a in 0..after {
                    let row = (before + dim_in + a) * inner;
                    for j in 0..inner {
                        d_blk[row + j].write(c_after);
                    }
                }
            }
            PadMode::Edge => {
                for b in 0..before {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s_blk[..inner].as_ptr(),
                            d_blk[b * inner..].as_mut_ptr() as *mut T,
                            inner,
                        );
                    }
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        s_blk.as_ptr(),
                        d_blk[before * inner..].as_mut_ptr() as *mut T,
                        src_block_len,
                    );
                }
                let last_src = &s_blk[(dim_in - 1) * inner..dim_in * inner];
                for a in 0..after {
                    let row = (before + dim_in + a) * inner;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            last_src.as_ptr(),
                            d_blk[row..].as_mut_ptr() as *mut T,
                            inner,
                        );
                    }
                }
            }
            PadMode::LinearRamp(_) | PadMode::LinearRampPairs(_) => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        s_blk.as_ptr(),
                        d_blk[before * inner..].as_mut_ptr() as *mut T,
                        src_block_len,
                    );
                }
                let first_src = &s_blk[..inner];
                for b in 0..before {
                    let frac = (b as f64) / (before as f64);
                    let row = b * inner;
                    for j in 0..inner {
                        d_blk[row + j].write(T::ramp_lerp(c_before, first_src[j], frac));
                    }
                }
                let last_src = &s_blk[(dim_in - 1) * inner..dim_in * inner];
                for a in 0..after {
                    let frac = ((a + 1) as f64) / (after as f64);
                    let row = (before + dim_in + a) * inner;
                    for j in 0..inner {
                        d_blk[row + j].write(T::ramp_lerp(last_src[j], c_after, frac));
                    }
                }
            }
            PadMode::Wrap => {
                for pos in 0..before {
                    let x = (pos as isize) - (before as isize);
                    let src_idx = x.rem_euclid(dim_in as isize) as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s_blk[src_idx * inner..].as_ptr(),
                            d_blk[pos * inner..].as_mut_ptr() as *mut T,
                            inner,
                        );
                    }
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        s_blk.as_ptr(),
                        d_blk[before * inner..].as_mut_ptr() as *mut T,
                        src_block_len,
                    );
                }
                for a in 0..after {
                    let src_idx = (dim_in + a) % dim_in;
                    let row = (before + dim_in + a) * inner;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s_blk[src_idx * inner..].as_ptr(),
                            d_blk[row..].as_mut_ptr() as *mut T,
                            inner,
                        );
                    }
                }
            }
            PadMode::Reflect(rtype) | PadMode::Symmetric(rtype) => {
                let include_edge = matches!(mode, PadMode::Symmetric(_));
                if *rtype == ReflectType::Even {
                    for pos in 0..before {
                        let x = (pos as isize) - (before as isize);
                        let src_idx = reflect_index(x, dim_in, include_edge);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                s_blk[src_idx * inner..].as_ptr(),
                                d_blk[pos * inner..].as_mut_ptr() as *mut T,
                                inner,
                            );
                        }
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s_blk.as_ptr(),
                            d_blk[before * inner..].as_mut_ptr() as *mut T,
                            src_block_len,
                        );
                    }
                    for a in 0..after {
                        let x = (dim_in + a) as isize;
                        let src_idx = reflect_index(x, dim_in, include_edge);
                        let row = (before + dim_in + a) * inner;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                s_blk[src_idx * inner..].as_ptr(),
                                d_blk[row..].as_mut_ptr() as *mut T,
                                inner,
                            );
                        }
                    }
                } else {
                    let mut line = vec![T::zero(); dim_in];
                    let mut out_line = vec![MaybeUninit::<T>::uninit(); dim_out];
                    for j in 0..inner {
                        for i in 0..dim_in {
                            line[i] = s_blk[i * inner + j];
                        }
                        fill_odd_reflect(&line, &mut out_line, before, after, include_edge);
                        for i in 0..dim_out {
                            d_blk[i * inner + j] = out_line[i];
                        }
                    }
                }
            }
            PadMode::Maximum(_) | PadMode::Minimum(_) | PadMode::Mean(_) | PadMode::Median(_) => {
                let kind = match mode {
                    PadMode::Maximum(_) => 0u8,
                    PadMode::Minimum(_) => 1u8,
                    PadMode::Mean(_) => 2u8,
                    PadMode::Median(_) => 3u8,
                    _ => unreachable!(),
                };
                let wb = sl_before.unwrap_or(dim_in).min(dim_in);
                let wa = sl_after.unwrap_or(dim_in).min(dim_in);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        s_blk.as_ptr(),
                        d_blk[before * inner..].as_mut_ptr() as *mut T,
                        src_block_len,
                    );
                }
                let mut col_buf = Vec::with_capacity(wb.max(wa));
                for j in 0..inner {
                    if before > 0 {
                        col_buf.clear();
                        for i in 0..wb {
                            col_buf.push(s_blk[i * inner + j]);
                        }
                        let val_b = compute_stat_1d(&col_buf, kind);
                        for b in 0..before {
                            d_blk[b * inner + j].write(val_b);
                        }
                    }
                    if after > 0 {
                        col_buf.clear();
                        for i in (dim_in - wa)..dim_in {
                            col_buf.push(s_blk[i * inner + j]);
                        }
                        let val_a = compute_stat_1d(&col_buf, kind);
                        for a in 0..after {
                            d_blk[(before + dim_in + a) * inner + j].write(val_a);
                        }
                    }
                }
            }
        }
    };

    if total >= RAYON_THRESHOLD && outer > 1 && src_block_len > 0 {
        dst.par_chunks_mut(dst_block_len)
            .zip(src_sl.par_chunks(src_block_len))
            .for_each(|(d, s)| process_outer_chunk(s, d));
    } else if total >= RAYON_THRESHOLD && outer > 1 {
        dst.par_chunks_mut(dst_block_len)
            .for_each(|d| process_outer_chunk(&[], d));
    } else if dst_block_len > 0 {
        for o in 0..outer {
            let s = if src_block_len == 0 {
                &[]
            } else {
                &src_sl[o * src_block_len..(o + 1) * src_block_len]
            };
            let d = &mut dst[o * dst_block_len..(o + 1) * dst_block_len];
            process_outer_chunk(s, d);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Pads an array according to `pad_width` and `mode` (matching NumPy axis-by-axis padding semantics).
pub fn pad<T: PadElement>(
    arr: &NdArray<T>,
    pad_width: &PadWidth,
    mode: &PadMode<T>,
) -> Result<NdArray<T>> {
    let ndim = arr.ndim();
    let widths = pad_width.resolve(ndim)?;
    let stat_lengths = match mode {
        PadMode::Maximum(sl)
        | PadMode::Minimum(sl)
        | PadMode::Mean(sl)
        | PadMode::Median(sl) => sl.resolve(ndim)?,
        _ => Vec::new(),
    };

    if ndim == 0 {
        return Ok(arr.deep_copy(Order::C));
    }

    let mut current = arr.clone();
    for (ax, &(before, after)) in widths.iter().enumerate() {
        current = pad_single_axis(&current, ax, before, after, mode, ax, &stat_lengths)?;
    }
    if !current.shares_storage(arr) {
        Ok(current)
    } else {
        Ok(current.deep_copy(Order::C))
    }
}

// =============================================================================
// 8. Resizing, Appending, Inserting, Deleting, Trimming
// =============================================================================

/// Returns a new array with `new_shape`, repeating or truncating C-order elements as needed.
pub fn resize<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    new_shape: &[usize],
) -> Result<NdArray<T>> {
    let total = shape_size(new_shape)?;
    if total == 0 || arr.is_empty() {
        return finish_c_array(AlignedBuffer::try_new_zeroed(total)?, new_shape);
    }
    let src_c = arr.to_c_contiguous();
    let src_sl = src_c.as_slice().unwrap_or(&[]);
    let src_len = src_sl.len();
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD {
        const CHUNK: usize = 4096;
        dst.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * CHUNK;
                for (i, slot) in chunk.iter_mut().enumerate() {
                    slot.write(src_sl[(start + i) % src_len]);
                }
            });
    } else {
        for (i, slot) in dst.iter_mut().enumerate() {
            slot.write(src_sl[i % src_len]);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, new_shape)
}

/// Appends values to the end of an array (flattened if `axis == None`, else along `axis`).
pub fn append<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    values: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    match axis {
        None => {
            let a1 = arr.ravel(Order::C);
            let v1 = values.ravel(Order::C);
            concatenate(&[&a1, &v1], Some(0))
        }
        Some(ax) => concatenate(&[arr, values], Some(ax)),
    }
}

/// Inserts values along `axis` before the given indices `obj`.
pub fn insert<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    obj: &InsertObj<'_>,
    values: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    let (work, norm_ax) = match axis {
        None => (arr.ravel(Order::C), 0usize),
        Some(ax) => {
            if arr.ndim() == 0 {
                return Err(NumPyError::InvalidAxis { axis: ax, ndim: 0 });
            }
            let a = normalize_axis(ax, arr.ndim())?;
            (arr.clone(), a)
        }
    };

    let shape = work.shape();
    let dim = shape[norm_ax];
    let ndim = shape.len();

    let (raw_indices, is_single_scalar) = match obj {
        InsertObj::Index(idx) => {
            let norm = if *idx < 0 {
                (dim as isize) + *idx
            } else {
                *idx
            };
            if norm < 0 || (norm as usize) > dim {
                return Err(NumPyError::IndexOutOfBounds(format!(
                    "index {idx} is out of bounds for axis {norm_ax} with size {dim}"
                )));
            }
            (vec![norm as usize], true)
        }
        InsertObj::Slice(sl) => {
            let (start, len, step) = sl.resolve(dim)?;
            let mut v = Vec::with_capacity(len);
            let mut cur = start as i128;
            let step128 = step as i128;
            for _ in 0..len {
                v.push(cur as usize);
                cur += step128;
            }
            (v, false)
        }
        InsertObj::Indices(idxs) => {
            let mut v = Vec::with_capacity(idxs.len());
            for &idx in *idxs {
                let norm = if idx < 0 {
                    (dim as isize) + idx
                } else {
                    idx
                };
                if norm < 0 || (norm as usize) > dim {
                    return Err(NumPyError::IndexOutOfBounds(format!(
                        "index {idx} is out of bounds for axis {norm_ax} with size {dim}"
                    )));
                }
                v.push(norm as usize);
            }
            (v, false)
        }
        InsertObj::BoolMask(mask) => {
            if mask.ndim() != 1 || mask.len() != dim {
                return Err(NumPyError::ShapeMismatch(format!(
                    "boolean array argument must be 1D with length {dim}"
                )));
            }
            let mut v = Vec::new();
            for (i, b) in mask.iter().enumerate() {
                if b {
                    v.push(i);
                }
            }
            (v, false)
        }
    };

    let num_new = raw_indices.len();
    let mut new_shape = shape.to_vec();
    new_shape[norm_ax] = dim + num_new;
    let total = shape_size(&new_shape)?;

    // Broadcast/adapt `values` to the inserted block shape
    let mut slice_shape = shape.to_vec();
    slice_shape[norm_ax] = num_new;

    let vals_prepared = if num_new == 0 {
        NdArray::<T>::zeros(&slice_shape)?
    } else if num_new == 1 {
        if axis.is_none() && is_single_scalar && (values.ndim() > 1 || values.len() > 1) {
            return insert_single_flat(&work, raw_indices[0], &values.ravel(Order::C));
        }
        let mut promoted = values.clone();
        if promoted.ndim() < ndim {
            let mut pad_shape = vec![1isize; ndim - promoted.ndim()];
            pad_shape.extend(promoted.shape().iter().map(|&d| d as isize));
            promoted = promoted.to_c_contiguous().reshape(&pad_shape)?;
        }
        if is_single_scalar {
            promoted = moveaxis(&promoted, &[0], &[norm_ax as isize])?;
        }
        let ins_len = promoted.shape()[norm_ax];
        if ins_len != 1 {
            let before_s = slice_along_axis(&work, norm_ax, 0, raw_indices[0])?;
            let after_s = slice_along_axis(&work, norm_ax, raw_indices[0], dim)?;
            let mut target_shape = shape.to_vec();
            target_shape[norm_ax] = ins_len;
            let vb = promoted.broadcast_to(&target_shape)?.to_c_contiguous();
            return concatenate(&[&before_s, &vb, &after_s], Some(norm_ax as isize));
        }
        promoted.broadcast_to(&slice_shape)?.to_c_contiguous()
    } else {
        values.broadcast_to(&slice_shape)?.to_c_contiguous()
    };

    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &new_shape);
    }

    // Build sorted placement schedule preserving relative order of identical insertion indices
    let mut indexed_inserts: Vec<(usize, usize)> = raw_indices
        .into_iter()
        .enumerate()
        .map(|(v_idx, pos)| (pos, v_idx))
        .collect();
    indexed_inserts.sort_by_key(|&(pos, v_idx)| (pos, v_idx));

    // Each element in schedule is (is_new: bool, src_index: usize)
    let mut schedule = Vec::with_capacity(dim + num_new);
    let mut ins_cursor = 0usize;
    for orig_pos in 0..=dim {
        while ins_cursor < indexed_inserts.len() && indexed_inserts[ins_cursor].0 == orig_pos {
            schedule.push((true, indexed_inserts[ins_cursor].1));
            ins_cursor += 1;
        }
        if orig_pos < dim {
            schedule.push((false, orig_pos));
        }
    }

    let work_c = work.to_c_contiguous();
    let work_sl = work_c.as_slice().unwrap_or(&[]);
    let val_sl = vals_prepared.as_slice().unwrap_or(&[]);

    let outer: usize = new_shape[..norm_ax].iter().product();
    let inner: usize = new_shape[norm_ax + 1..].iter().product();
    let work_row_len = dim * inner;
    let val_row_len = num_new * inner;
    let out_row_len = new_shape[norm_ax] * inner;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    let fill_row = |w_row: &[T], v_row: &[T], d_row: &mut [MaybeUninit<T>]| {
        for (out_idx, &(is_new, src_idx)) in schedule.iter().enumerate() {
            let src = if is_new {
                &v_row[src_idx * inner..(src_idx + 1) * inner]
            } else {
                &w_row[src_idx * inner..(src_idx + 1) * inner]
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    d_row[out_idx * inner..].as_mut_ptr() as *mut T,
                    inner,
                );
            }
        }
    };

    if total >= RAYON_THRESHOLD && outer > 1 {
        dst.par_chunks_mut(out_row_len)
            .enumerate()
            .for_each(|(o, d_row)| {
                let w_row = if work_row_len == 0 {
                    &[]
                } else {
                    &work_sl[o * work_row_len..(o + 1) * work_row_len]
                };
                let v_row = if val_row_len == 0 {
                    &[]
                } else {
                    &val_sl[o * val_row_len..(o + 1) * val_row_len]
                };
                fill_row(w_row, v_row, d_row);
            });
    } else if out_row_len > 0 {
        for o in 0..outer {
            let w_row = if work_row_len == 0 {
                &[]
            } else {
                &work_sl[o * work_row_len..(o + 1) * work_row_len]
            };
            let v_row = if val_row_len == 0 {
                &[]
            } else {
                &val_sl[o * val_row_len..(o + 1) * val_row_len]
            };
            let d_row = &mut dst[o * out_row_len..(o + 1) * out_row_len];
            fill_row(w_row, v_row, d_row);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &new_shape)
}

fn insert_single_flat<T: Scalar + Send + Sync>(
    flat: &NdArray<T>,
    pos: usize,
    values_flat: &NdArray<T>,
) -> Result<NdArray<T>> {
    let before = slice_along_axis(flat, 0, 0, pos)?;
    let after = slice_along_axis(flat, 0, pos, flat.len())?;
    concatenate(&[&before, values_flat, &after], Some(0))
}

/// Deletes sub-arrays along `axis` matching `obj`.
pub fn delete<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    obj: &DeleteObj<'_>,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    let (work, norm_ax) = match axis {
        None => (arr.ravel(Order::C), 0usize),
        Some(ax) => {
            if arr.ndim() == 0 {
                return Err(NumPyError::InvalidAxis { axis: ax, ndim: 0 });
            }
            let a = normalize_axis(ax, arr.ndim())?;
            (arr.clone(), a)
        }
    };

    let shape = work.shape();
    let dim = shape[norm_ax];
    let mut keep = vec![true; dim];

    match obj {
        DeleteObj::Index(idx) => {
            let norm = numpy_types::normalize_index(*idx, dim)?;
            keep[norm] = false;
        }
        DeleteObj::Slice(sl) => {
            let (start, len, step) = sl.resolve(dim)?;
            let mut cur = start as i128;
            let step128 = step as i128;
            for _ in 0..len {
                keep[cur as usize] = false;
                cur += step128;
            }
        }
        DeleteObj::Indices(idxs) => {
            for &idx in *idxs {
                let norm = numpy_types::normalize_index(idx, dim)?;
                keep[norm] = false;
            }
        }
        DeleteObj::BoolMask(mask) => {
            if mask.ndim() != 1 || mask.len() != dim {
                return Err(NumPyError::ShapeMismatch(format!(
                    "boolean array argument must be 1D with length {dim}"
                )));
            }
            for (i, b) in mask.iter().enumerate() {
                if b {
                    keep[i] = false;
                }
            }
        }
    }

    let kept_indices: Vec<usize> = keep
        .into_iter()
        .enumerate()
        .filter_map(|(i, k)| if k { Some(i) } else { None })
        .collect();

    let mut new_shape = shape.to_vec();
    new_shape[norm_ax] = kept_indices.len();
    let total = shape_size(&new_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &new_shape);
    }

    let work_c = work.to_c_contiguous();
    let work_sl = work_c.as_slice().unwrap_or(&[]);
    let outer: usize = new_shape[..norm_ax].iter().product();
    let inner: usize = new_shape[norm_ax + 1..].iter().product();
    let in_row_len = dim * inner;
    let out_row_len = kept_indices.len() * inner;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    let copy_row = |w_row: &[T], d_row: &mut [MaybeUninit<T>]| {
        for (dst_idx, &src_idx) in kept_indices.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    w_row[src_idx * inner..].as_ptr(),
                    d_row[dst_idx * inner..].as_mut_ptr() as *mut T,
                    inner,
                );
            }
        }
    };

    if total >= RAYON_THRESHOLD && outer > 1 {
        dst.par_chunks_mut(out_row_len)
            .zip(work_sl.par_chunks(in_row_len))
            .for_each(|(d, s)| copy_row(s, d));
    } else if out_row_len > 0 {
        for (d, s) in dst.chunks_mut(out_row_len).zip(work_sl.chunks(in_row_len)) {
            copy_row(s, d);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &new_shape)
}

/// Trims leading and/or trailing zeros along the specified `axis` (or all axes when `None`).
/// Always returns a zero-copy view.
pub fn trim_zeros<T: Scalar>(
    arr: &NdArray<T>,
    trim: TrimDirection,
    axis: Option<&[isize]>,
) -> Result<NdArray<T>> {
    let ndim = arr.ndim();
    if ndim == 0 {
        return Ok(arr.clone());
    }
    let trim_axes: Vec<usize> = match axis {
        None => (0..ndim).collect(),
        Some(ax) => numpy_types::normalize_axes(ax, ndim)?,
    };

    let shape = arr.shape();
    let mut bounds: Vec<(usize, usize)> = shape.iter().map(|&d| (0, d)).collect();
    let trim_front = matches!(trim, TrimDirection::Both | TrimDirection::Front);
    let trim_back = matches!(trim, TrimDirection::Both | TrimDirection::Back);

    if arr.is_empty() {
        return Ok(arr.clone());
    }

    let view = arr.view();
    let strides = view.strides();
    let base = view.as_ptr();

    let slice_has_nonzero = |ax: usize, idx: usize| -> bool {
        let mut sub_shape = Vec::with_capacity(ndim - 1);
        let mut sub_strides = Vec::with_capacity(ndim - 1);
        for d in 0..ndim {
            if d != ax {
                sub_shape.push(shape[d]);
                sub_strides.push(strides[d]);
            }
        }
        let sub_ptr_base = unsafe { base.offset((idx as isize) * strides[ax]) };
        let sub_len: usize = sub_shape.iter().product();
        if sub_len == 0 {
            return false;
        }
        let mut coords = vec![0usize; sub_shape.len()];
        let mut ptr = sub_ptr_base;
        for step in 0..sub_len {
            if unsafe { *ptr }.is_truthy() {
                return true;
            }
            if step + 1 < sub_len {
                advance_coords_ptr(&mut coords, &mut ptr, &sub_shape, &sub_strides);
            }
        }
        false
    };

    for &ax in &trim_axes {
        let dim = shape[ax];
        if dim == 0 {
            bounds[ax] = (0, 0);
            continue;
        }
        let mut first = 0usize;
        if trim_front {
            let mut found = false;
            for i in 0..dim {
                if slice_has_nonzero(ax, i) {
                    first = i;
                    found = true;
                    break;
                }
            }
            if !found {
                bounds[ax] = (0, 0);
                continue;
            }
        }
        let mut stop = dim;
        if trim_back {
            let mut found = false;
            for i in (first..dim).rev() {
                if slice_has_nonzero(ax, i) {
                    stop = i + 1;
                    found = true;
                    break;
                }
            }
            if !found {
                bounds[ax] = (0, 0);
                continue;
            }
        }
        bounds[ax] = (first, stop);
    }

    let args: Vec<SliceArg> = bounds
        .into_iter()
        .map(|(s, e)| SliceArg::Slice(Slice::new(Some(s as isize), Some(e as isize), 1)))
        .collect();
    let _ = coords_offset(&[], &[]);
    arr.slice(&args)
}

// =============================================================================
// 9. DynArray Manipulation Functions
// =============================================================================

/// Dynamic `reshape`.
pub fn dyn_reshape(arr: &DynArray, newshape: &[isize]) -> Result<DynArray> {
    arr.reshape(newshape)
}

/// Dynamic `reshape` with explicit `Order`.
pub fn dyn_reshape_with_order(
    arr: &DynArray,
    newshape: &[isize],
    order: Order,
) -> Result<DynArray> {
    dispatch_dyn!(arr, a => reshape_with_order(a, newshape, order).map(DynArray::from))
}

/// Dynamic `ravel` (zero-copy view when contiguous).
pub fn dyn_ravel(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => ravel(a).map(DynArray::from))
}

/// Dynamic `ravel` with explicit `Order`.
pub fn dyn_ravel_with_order(arr: &DynArray, order: Order) -> Result<DynArray> {
    dispatch_dyn!(arr, a => ravel_with_order(a, order).map(DynArray::from))
}

/// Dynamic `flatten` (always returns owning C-contiguous copy).
pub fn dyn_flatten(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => flatten(a).map(DynArray::from))
}

/// Dynamic `flatten` with explicit `Order`.
pub fn dyn_flatten_with_order(arr: &DynArray, order: Order) -> Result<DynArray> {
    dispatch_dyn!(arr, a => flatten_with_order(a, order).map(DynArray::from))
}

/// Dynamic `transpose` (zero-copy view).
pub fn dyn_transpose(arr: &DynArray, axes: Option<&[usize]>) -> Result<DynArray> {
    match axes {
        None => arr.transpose(None),
        Some(ax) => {
            let ax_i: Vec<isize> = ax.iter().map(|&a| a as isize).collect();
            arr.transpose(Some(&ax_i))
        }
    }
}

/// Dynamic `swapaxes` (zero-copy view).
pub fn dyn_swapaxes(arr: &DynArray, axis1: isize, axis2: isize) -> Result<DynArray> {
    dispatch_dyn!(arr, a => swapaxes(a, axis1, axis2).map(DynArray::from))
}

/// Dynamic `moveaxis` (zero-copy view).
pub fn dyn_moveaxis(
    arr: &DynArray,
    source: &[isize],
    destination: &[isize],
) -> Result<DynArray> {
    dispatch_dyn!(arr, a => moveaxis(a, source, destination).map(DynArray::from))
}

/// Dynamic `rollaxis` (zero-copy view).
pub fn dyn_rollaxis(arr: &DynArray, axis: isize, start: isize) -> Result<DynArray> {
    dispatch_dyn!(arr, a => rollaxis(a, axis, start).map(DynArray::from))
}

/// Dynamic `squeeze` (zero-copy view).
pub fn dyn_squeeze(arr: &DynArray, axis: Option<&[isize]>) -> Result<DynArray> {
    dispatch_dyn!(arr, a => squeeze(a, axis).map(DynArray::from))
}

/// Dynamic `expand_dims` (zero-copy view).
pub fn dyn_expand_dims(arr: &DynArray, axes: &[isize]) -> Result<DynArray> {
    dispatch_dyn!(arr, a => expand_dims(a, axes).map(DynArray::from))
}

/// Dynamic `atleast_1d` (zero-copy view).
pub fn dyn_atleast_1d(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => atleast_1d(a).map(DynArray::from))
}

/// Dynamic `atleast_2d` (zero-copy view).
pub fn dyn_atleast_2d(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => atleast_2d(a).map(DynArray::from))
}

/// Dynamic `atleast_3d` (zero-copy view).
pub fn dyn_atleast_3d(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => atleast_3d(a).map(DynArray::from))
}

fn promote_dyn_slice(arrays: &[&DynArray]) -> Result<(DType, Vec<DynArray>)> {
    let dtypes: Vec<DType> = arrays.iter().map(|a| a.dtype()).collect();
    let target = numpy_types::result_type(&dtypes)?;
    let mut cast = Vec::with_capacity(arrays.len());
    for &a in arrays {
        if a.dtype() == target {
            cast.push(a.clone());
        } else {
            cast.push(a.astype(target, Casting::Unsafe)?);
        }
    }
    Ok((target, cast))
}

macro_rules! dispatch_homogeneous_concat {
    ($target:expr, $cast:expr, $axis:expr, $op:ident) => {
        match $target {
            DType::Bool => {
                let typed: Vec<&NdArray<bool>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::Bool(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::Bool($op(&typed, $axis)?))
            }
            DType::I8 => {
                let typed: Vec<&NdArray<i8>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::I8(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::I8($op(&typed, $axis)?))
            }
            DType::I16 => {
                let typed: Vec<&NdArray<i16>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::I16(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::I16($op(&typed, $axis)?))
            }
            DType::I32 => {
                let typed: Vec<&NdArray<i32>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::I32(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::I32($op(&typed, $axis)?))
            }
            DType::I64 => {
                let typed: Vec<&NdArray<i64>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::I64(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::I64($op(&typed, $axis)?))
            }
            DType::U8 => {
                let typed: Vec<&NdArray<u8>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::U8(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::U8($op(&typed, $axis)?))
            }
            DType::U16 => {
                let typed: Vec<&NdArray<u16>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::U16(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::U16($op(&typed, $axis)?))
            }
            DType::U32 => {
                let typed: Vec<&NdArray<u32>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::U32(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::U32($op(&typed, $axis)?))
            }
            DType::U64 => {
                let typed: Vec<&NdArray<u64>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::U64(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::U64($op(&typed, $axis)?))
            }
            DType::F16 => {
                let typed: Vec<&NdArray<f16>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::F16(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::F16($op(&typed, $axis)?))
            }
            DType::F32 => {
                let typed: Vec<&NdArray<f32>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::F32(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::F32($op(&typed, $axis)?))
            }
            DType::F64 => {
                let typed: Vec<&NdArray<f64>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::F64(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::F64($op(&typed, $axis)?))
            }
            DType::C64 => {
                let typed: Vec<&NdArray<Complex32>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::C64(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::C64($op(&typed, $axis)?))
            }
            DType::C128 => {
                let typed: Vec<&NdArray<Complex64>> = $cast
                    .iter()
                    .map(|d| match d {
                        DynArray::C128(a) => Ok(a),
                        _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                    })
                    .collect::<Result<_>>()?;
                Ok(DynArray::C128($op(&typed, $axis)?))
            }
        }
    };
}

/// Dynamic `concatenate` (automatically promotes heterogeneous input dtypes via `result_type`).
pub fn dyn_concatenate(arrays: &[&DynArray], axis: Option<isize>) -> Result<DynArray> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue(
            "need at least one array to concatenate".into(),
        ));
    }
    let (target, cast) = promote_dyn_slice(arrays)?;
    dispatch_homogeneous_concat!(target, cast, axis, concatenate)
}

/// Dynamic alias for `dyn_concatenate`.
pub fn dyn_concat(arrays: &[&DynArray], axis: Option<isize>) -> Result<DynArray> {
    dyn_concatenate(arrays, axis)
}

/// Dynamic `stack`.
pub fn dyn_stack(arrays: &[&DynArray], axis: isize) -> Result<DynArray> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to stack".into()));
    }
    let (target, cast) = promote_dyn_slice(arrays)?;
    dispatch_homogeneous_concat!(target, cast, axis, stack)
}

/// Dynamic `vstack`.
pub fn dyn_vstack(arrays: &[&DynArray]) -> Result<DynArray> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to vstack".into()));
    }
    let promoted: Vec<DynArray> = arrays
        .iter()
        .map(|a| dyn_atleast_2d(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&DynArray> = promoted.iter().collect();
    dyn_concatenate(&refs, Some(0))
}

/// Dynamic `hstack`.
pub fn dyn_hstack(arrays: &[&DynArray]) -> Result<DynArray> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to hstack".into()));
    }
    let promoted: Vec<DynArray> = arrays
        .iter()
        .map(|a| dyn_atleast_1d(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&DynArray> = promoted.iter().collect();
    if promoted[0].ndim() == 1 {
        dyn_concatenate(&refs, Some(0))
    } else {
        dyn_concatenate(&refs, Some(1))
    }
}

/// Dynamic `dstack`.
pub fn dyn_dstack(arrays: &[&DynArray]) -> Result<DynArray> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue("need at least one array to dstack".into()));
    }
    let promoted: Vec<DynArray> = arrays
        .iter()
        .map(|a| dyn_atleast_3d(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&DynArray> = promoted.iter().collect();
    dyn_concatenate(&refs, Some(2))
}

/// Dynamic `column_stack`.
pub fn dyn_column_stack(arrays: &[&DynArray]) -> Result<DynArray> {
    if arrays.is_empty() {
        return Err(NumPyError::InvalidValue(
            "need at least one array to column_stack".into(),
        ));
    }
    let mut prepared = Vec::with_capacity(arrays.len());
    for &a in arrays {
        if a.ndim() < 2 {
            let a1 = dyn_atleast_1d(a)?;
            prepared.push(dyn_expand_dims(&a1, &[1])?);
        } else {
            prepared.push(a.clone());
        }
    }
    let refs: Vec<&DynArray> = prepared.iter().collect();
    dyn_concatenate(&refs, Some(1))
}

fn inspect_dyn_block_tree(
    node: &DynBlockNode<'_>,
    depth: usize,
    max_list_depth: &mut usize,
    max_arr_ndim: &mut usize,
    dtypes: &mut Vec<DType>,
) -> Result<()> {
    match node {
        DynBlockNode::Array(a) => {
            *max_list_depth = (*max_list_depth).max(depth);
            *max_arr_ndim = (*max_arr_ndim).max(a.ndim());
            dtypes.push(a.dtype());
            Ok(())
        }
        DynBlockNode::List(items) => {
            if items.is_empty() {
                return Err(NumPyError::InvalidValue(format!(
                    "List at depth {depth} cannot be empty"
                )));
            }
            for item in items {
                inspect_dyn_block_tree(item, depth + 1, max_list_depth, max_arr_ndim, dtypes)?;
            }
            Ok(())
        }
    }
}

fn eval_dyn_block_node(
    node: &DynBlockNode<'_>,
    depth: usize,
    max_list_depth: usize,
    final_ndim: usize,
) -> Result<DynArray> {
    match node {
        DynBlockNode::Array(a) => {
            if a.ndim() >= final_ndim {
                Ok((*a).clone())
            } else {
                let add = final_ndim - a.ndim();
                let new_axes: Vec<isize> = (0..add as isize).collect();
                dyn_expand_dims(a, &new_axes)
            }
        }
        DynBlockNode::List(items) => {
            let mut sub = Vec::with_capacity(items.len());
            for item in items {
                sub.push(eval_dyn_block_node(item, depth + 1, max_list_depth, final_ndim)?);
            }
            let axis = (final_ndim - max_list_depth + depth) as isize;
            let refs: Vec<&DynArray> = sub.iter().collect();
            dyn_concatenate(&refs, Some(axis))
        }
    }
}

/// Dynamic `block` (promotes heterogeneous leaf dtypes and returns unified `DynArray`).
pub fn dyn_block(root: &DynBlockNode<'_>) -> Result<DynArray> {
    let mut max_list_depth = 0usize;
    let mut max_arr_ndim = 0usize;
    let mut dtypes = Vec::new();
    inspect_dyn_block_tree(root, 0, &mut max_list_depth, &mut max_arr_ndim, &mut dtypes)?;
    let final_ndim = max_list_depth.max(max_arr_ndim);
    let _ = numpy_types::result_type(&dtypes);
    eval_dyn_block_node(root, 0, max_list_depth, final_ndim)
}

/// Dynamic `split` (returns zero-copy `DynArray` views).
pub fn dyn_split(arr: &DynArray, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<DynArray>> {
    dispatch_dyn!(arr, a => {
        Ok(split(a, spec, axis)?.into_iter().map(DynArray::from).collect())
    })
}

/// Dynamic `array_split` (zero-copy `DynArray` views).
pub fn dyn_array_split(arr: &DynArray, spec: SplitSpec<'_>, axis: isize) -> Result<Vec<DynArray>> {
    dispatch_dyn!(arr, a => {
        Ok(array_split(a, spec, axis)?.into_iter().map(DynArray::from).collect())
    })
}

/// Dynamic `hsplit` (zero-copy `DynArray` views).
pub fn dyn_hsplit(arr: &DynArray, spec: SplitSpec<'_>) -> Result<Vec<DynArray>> {
    dispatch_dyn!(arr, a => {
        Ok(hsplit(a, spec)?.into_iter().map(DynArray::from).collect())
    })
}

/// Dynamic `vsplit` (zero-copy `DynArray` views).
pub fn dyn_vsplit(arr: &DynArray, spec: SplitSpec<'_>) -> Result<Vec<DynArray>> {
    dispatch_dyn!(arr, a => {
        Ok(vsplit(a, spec)?.into_iter().map(DynArray::from).collect())
    })
}

/// Dynamic `dsplit` (zero-copy `DynArray` views).
pub fn dyn_dsplit(arr: &DynArray, spec: SplitSpec<'_>) -> Result<Vec<DynArray>> {
    dispatch_dyn!(arr, a => {
        Ok(dsplit(a, spec)?.into_iter().map(DynArray::from).collect())
    })
}

/// Dynamic `unstack` (zero-copy `DynArray` views).
pub fn dyn_unstack(arr: &DynArray, axis: isize) -> Result<Vec<DynArray>> {
    dispatch_dyn!(arr, a => {
        Ok(unstack(a, axis)?.into_iter().map(DynArray::from).collect())
    })
}

/// Dynamic `tile`.
pub fn dyn_tile(arr: &DynArray, reps: &[usize]) -> Result<DynArray> {
    dispatch_dyn!(arr, a => tile(a, reps).map(DynArray::from))
}

/// Dynamic `repeat`.
pub fn dyn_repeat(arr: &DynArray, repeats: &[usize], axis: Option<isize>) -> Result<DynArray> {
    dispatch_dyn!(arr, a => repeat(a, repeats, axis).map(DynArray::from))
}

/// Dynamic `flip` (zero-copy view).
pub fn dyn_flip(arr: &DynArray, axis: Option<&[isize]>) -> Result<DynArray> {
    dispatch_dyn!(arr, a => flip(a, axis).map(DynArray::from))
}

/// Dynamic `fliplr` (zero-copy view).
pub fn dyn_fliplr(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => fliplr(a).map(DynArray::from))
}

/// Dynamic `flipud` (zero-copy view).
pub fn dyn_flipud(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => flipud(a).map(DynArray::from))
}

/// Dynamic `rot90` (zero-copy view).
pub fn dyn_rot90(arr: &DynArray, k: isize, axes: (isize, isize)) -> Result<DynArray> {
    dispatch_dyn!(arr, a => rot90(a, k, axes).map(DynArray::from))
}

/// Dynamic `roll`.
pub fn dyn_roll(arr: &DynArray, shift: isize, axis: Option<isize>) -> Result<DynArray> {
    dispatch_dyn!(arr, a => roll(a, shift, axis).map(DynArray::from))
}

/// Dynamic `roll_axes`.
pub fn dyn_roll_axes(
    arr: &DynArray,
    shifts: &[isize],
    axes: Option<&[isize]>,
) -> Result<DynArray> {
    dispatch_dyn!(arr, a => roll_axes(a, shifts, axes).map(DynArray::from))
}

fn convert_pad_mode<T: Scalar>(mode: &DynPadMode) -> PadMode<T> {
    match mode {
        DynPadMode::Constant(s) => PadMode::Constant(s.cast_to::<T>()),
        DynPadMode::ConstantPairs(pairs) => PadMode::ConstantPairs(
            pairs
                .iter()
                .map(|(a, b)| (a.cast_to::<T>(), b.cast_to::<T>()))
                .collect(),
        ),
        DynPadMode::Edge => PadMode::Edge,
        DynPadMode::LinearRamp(s) => PadMode::LinearRamp(s.cast_to::<T>()),
        DynPadMode::LinearRampPairs(pairs) => PadMode::LinearRampPairs(
            pairs
                .iter()
                .map(|(a, b)| (a.cast_to::<T>(), b.cast_to::<T>()))
                .collect(),
        ),
        DynPadMode::Maximum(sl) => PadMode::Maximum(sl.clone()),
        DynPadMode::Mean(sl) => PadMode::Mean(sl.clone()),
        DynPadMode::Median(sl) => PadMode::Median(sl.clone()),
        DynPadMode::Minimum(sl) => PadMode::Minimum(sl.clone()),
        DynPadMode::Reflect(rt) => PadMode::Reflect(*rt),
        DynPadMode::Symmetric(rt) => PadMode::Symmetric(*rt),
        DynPadMode::Wrap => PadMode::Wrap,
        DynPadMode::Empty => PadMode::Empty,
    }
}

/// Dynamic `pad`.
pub fn dyn_pad(arr: &DynArray, pad_width: &PadWidth, mode: &DynPadMode) -> Result<DynArray> {
    dispatch_dyn!(arr, a => pad(a, pad_width, &convert_pad_mode(mode)).map(DynArray::from))
}

/// Dynamic `resize`.
pub fn dyn_resize(arr: &DynArray, new_shape: &[usize]) -> Result<DynArray> {
    dispatch_dyn!(arr, a => resize(a, new_shape).map(DynArray::from))
}

/// Dynamic `append` (promotes dtype if `arr` and `values` differ).
pub fn dyn_append(
    arr: &DynArray,
    values: &DynArray,
    axis: Option<isize>,
) -> Result<DynArray> {
    let target = numpy_types::promote_types(arr.dtype(), values.dtype());
    let a_c = if arr.dtype() == target {
        arr.clone()
    } else {
        arr.astype(target, Casting::Unsafe)?
    };
    let v_c = if values.dtype() == target {
        values.clone()
    } else {
        values.astype(target, Casting::Unsafe)?
    };
    match axis {
        None => {
            let a1 = dyn_ravel(&a_c)?;
            let v1 = dyn_ravel(&v_c)?;
            dyn_concatenate(&[&a1, &v1], Some(0))
        }
        Some(ax) => dyn_concatenate(&[&a_c, &v_c], Some(ax)),
    }
}

/// Dynamic `insert` specification (`DynInsertObj` converts `DynArray` boolean masks).
pub enum DynInsertObj<'a> {
    Index(isize),
    Slice(Slice),
    Indices(&'a [isize]),
    BoolMask(&'a DynArray),
}

/// Dynamic `delete` specification.
pub enum DynDeleteObj<'a> {
    Index(isize),
    Slice(Slice),
    Indices(&'a [isize]),
    BoolMask(&'a DynArray),
}

/// Dynamic `insert`.
pub fn dyn_insert(
    arr: &DynArray,
    obj: &DynInsertObj<'_>,
    values: &DynArray,
    axis: Option<isize>,
) -> Result<DynArray> {
    let mask_owned;
    let typed_obj = match obj {
        DynInsertObj::Index(i) => InsertObj::Index(*i),
        DynInsertObj::Slice(s) => InsertObj::Slice(*s),
        DynInsertObj::Indices(ix) => InsertObj::Indices(ix),
        DynInsertObj::BoolMask(m) => {
            mask_owned = dyn_as_bool(m)?;
            InsertObj::BoolMask(mask_owned)
        }
    };
    let target = numpy_types::promote_types(arr.dtype(), values.dtype());
    let a_c = if arr.dtype() == target {
        arr.clone()
    } else {
        arr.astype(target, Casting::Unsafe)?
    };
    let v_c = if values.dtype() == target {
        values.clone()
    } else {
        values.astype(target, Casting::Unsafe)?
    };

    macro_rules! do_insert {
        ($t:ty, $var:ident) => {
            match (&a_c, &v_c) {
                (DynArray::$var(a), DynArray::$var(v)) => {
                    Ok(DynArray::$var(insert(a, &typed_obj, v, axis)?))
                }
                _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
            }
        };
    }

    match target {
        DType::Bool => do_insert!(bool, Bool),
        DType::I8 => do_insert!(i8, I8),
        DType::I16 => do_insert!(i16, I16),
        DType::I32 => do_insert!(i32, I32),
        DType::I64 => do_insert!(i64, I64),
        DType::U8 => do_insert!(u8, U8),
        DType::U16 => do_insert!(u16, U16),
        DType::U32 => do_insert!(u32, U32),
        DType::U64 => do_insert!(u64, U64),
        DType::F16 => do_insert!(f16, F16),
        DType::F32 => do_insert!(f32, F32),
        DType::F64 => do_insert!(f64, F64),
        DType::C64 => do_insert!(Complex32, C64),
        DType::C128 => do_insert!(Complex64, C128),
    }
}

/// Dynamic `delete`.
pub fn dyn_delete(
    arr: &DynArray,
    obj: &DynDeleteObj<'_>,
    axis: Option<isize>,
) -> Result<DynArray> {
    let mask_owned;
    let typed_obj = match obj {
        DynDeleteObj::Index(i) => DeleteObj::Index(*i),
        DynDeleteObj::Slice(s) => DeleteObj::Slice(*s),
        DynDeleteObj::Indices(ix) => DeleteObj::Indices(ix),
        DynDeleteObj::BoolMask(m) => {
            mask_owned = dyn_as_bool(m)?;
            DeleteObj::BoolMask(mask_owned)
        }
    };
    dispatch_dyn!(arr, a => delete(a, &typed_obj, axis).map(DynArray::from))
}

/// Dynamic `trim_zeros` (zero-copy view).
pub fn dyn_trim_zeros(
    arr: &DynArray,
    trim: TrimDirection,
    axis: Option<&[isize]>,
) -> Result<DynArray> {
    dispatch_dyn!(arr, a => trim_zeros(a, trim, axis).map(DynArray::from))
}

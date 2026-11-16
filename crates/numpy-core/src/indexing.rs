//! Indexing, selection, coordinate conversion, and diagonal extraction routines
//! (`take`, `take_along_axis`, `put`, `put_along_axis`, `putmask`, `compress`, `extract`,
//! `choose`, `where_select`, `select`, `nonzero`, `argwhere`, `flatnonzero`, `diagonal`,
//! `fill_diagonal`, `unravel_index`, `ravel_multi_index`, fancy integer & boolean mask indexing)
//! for both `NdArray<T>` and `DynArray`.

use crate::util::{
    RAYON_THRESHOLD, advance_coords_ptr, coords_offset, copy_view_to_uninit, dyn_as_bool,
    dyn_to_i64_indices, finish_c_array, linear_to_coords, DynScalar,
};
use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{
    broadcast_many_shapes, broadcast_shapes, normalize_axis, shape_size, AlignedBuffer, Casting,
    DType, DynArray, NdArray, NumPyError, Order, Result, Scalar, Slice, SliceArg, dispatch_dyn,
};
use rayon::prelude::*;

// =============================================================================
// 1. Index Modes & Mixed Fancy Indexing Spec
// =============================================================================

/// Bounds-handling policy matching NumPy `mode='raise' | 'wrap' | 'clip'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexMode {
    /// Return `IndexOutOfBounds` on out-of-range indices (default).
    #[default]
    Raise,
    /// Wrap indices modulo axis size.
    Wrap,
    /// Clamp indices to `[0, axis_size - 1]`.
    Clip,
}

/// Unified indexing component for mixed slice / scalar / fancy integer / boolean / newaxis indexing.
#[derive(Debug, Clone)]
pub enum FancyArg<'a> {
    Int(isize),
    Slice(Slice),
    NewAxis,
    IntArray(&'a NdArray<i64>),
    BoolMask(&'a NdArray<bool>),
}

/// Dynamic variant of `FancyArg` accepting `DynArray` index/mask arrays.
#[derive(Debug, Clone)]
pub enum DynFancyArg<'a> {
    Int(isize),
    Slice(Slice),
    NewAxis,
    IntArray(&'a DynArray),
    BoolMask(&'a DynArray),
}

#[inline]
fn resolve_index_mode(
    idx: i64,
    dim: usize,
    mode: IndexMode,
    allow_negative_in_raise: bool,
) -> Result<usize> {
    if dim == 0 {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "cannot index axis of size 0 with index {idx}"
        )));
    }
    match mode {
        IndexMode::Raise => {
            if idx < 0 && !allow_negative_in_raise {
                return Err(NumPyError::IndexOutOfBounds(format!(
                    "index {idx} is out of bounds for axis with size {dim}"
                )));
            }
            let normalized = if idx < 0 {
                (dim as i64).checked_add(idx).ok_or_else(|| {
                    NumPyError::IndexOutOfBounds(format!("index {idx} overflows"))
                })?
            } else {
                idx
            };
            if normalized < 0 || (normalized as u64) >= (dim as u64) {
                return Err(NumPyError::IndexOutOfBounds(format!(
                    "index {idx} is out of bounds for axis with size {dim}"
                )));
            }
            Ok(normalized as usize)
        }
        IndexMode::Wrap => Ok(idx.rem_euclid(dim as i64) as usize),
        IndexMode::Clip => Ok(idx.clamp(0, (dim - 1) as i64) as usize),
    }
}

fn compute_c_unit_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for d in (0..shape.len()).rev() {
        s[d] = acc;
        acc = acc.saturating_mul(shape[d]);
    }
    s
}

// =============================================================================
// 2. take / take_along_axis / put / put_along_axis / putmask
// =============================================================================

/// Takes elements from an array along `axis` (or flattened array if `axis == None`).
pub fn take<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    indices: &NdArray<i64>,
    axis: Option<isize>,
    mode: IndexMode,
) -> Result<NdArray<T>> {
    if axis.is_none() {
        let flat_arr = arr.to_c_contiguous();
        let src = flat_arr.as_slice().unwrap_or(&[]);
        let dim = src.len();
        let out_shape = indices.shape().to_vec();
        let out_len = indices.len();
        if out_len == 0 {
            return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
        }
        let idx_c = indices.to_c_contiguous();
        let idx_sl = idx_c.as_slice().unwrap_or(&[]);
        let mut resolved = Vec::with_capacity(out_len);
        for &raw in idx_sl {
            resolved.push(resolve_index_mode(raw, dim, mode, true)?);
        }
        let mut uninit = AlignedBuffer::<T>::new_uninit(out_len);
        let dst = uninit.as_mut_slice();
        if out_len >= RAYON_THRESHOLD {
            dst.par_iter_mut()
                .zip(resolved.par_iter())
                .for_each(|(slot, &r)| {
                    slot.write(src[r]);
                });
        } else {
            for (slot, &r) in dst.iter_mut().zip(resolved.iter()) {
                slot.write(src[r]);
            }
        }
        let buf = unsafe { uninit.assume_init() };
        return finish_c_array(buf, &out_shape);
    }

    let ax = axis.unwrap_or(0);
    if arr.ndim() == 0 {
        return Err(NumPyError::InvalidAxis { axis: ax, ndim: 0 });
    }
    let norm_ax = normalize_axis(ax, arr.ndim())?;
    let dim = arr.shape()[norm_ax];

    let mut out_shape = Vec::with_capacity(arr.ndim() - 1 + indices.ndim());
    out_shape.extend_from_slice(&arr.shape()[..norm_ax]);
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&arr.shape()[norm_ax + 1..]);

    let total = shape_size(&out_shape)?;
    if total == 0 {
        // Still validate indices if indices is non-empty and dim == 0? No: if inner or outer is 0, total is 0.
        if !indices.is_empty() && dim == 0 {
            let idx0 = indices.iter().next().unwrap_or(0);
            let _ = resolve_index_mode(idx0, dim, mode, true)?;
        }
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let idx_c = indices.to_c_contiguous();
    let idx_sl = idx_c.as_slice().unwrap_or(&[]);
    let mut resolved = Vec::with_capacity(idx_sl.len());
    for &raw in idx_sl {
        resolved.push(resolve_index_mode(raw, dim, mode, true)?);
    }

    let arr_c = arr.to_c_contiguous();
    let src = arr_c.as_slice().unwrap_or(&[]);
    let outer: usize = arr.shape()[..norm_ax].iter().product();
    let inner: usize = arr.shape()[norm_ax + 1..].iter().product();
    let src_row_len = dim * inner;
    let dst_row_len = resolved.len() * inner;

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD && outer > 1 {
        dst.par_chunks_mut(dst_row_len)
            .zip(src.par_chunks(src_row_len))
            .for_each(|(d_row, s_row)| {
                for (j, &r) in resolved.iter().enumerate() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            s_row[r * inner..].as_ptr(),
                            d_row[j * inner..].as_mut_ptr() as *mut T,
                            inner,
                        );
                    }
                }
            });
    } else if total >= RAYON_THRESHOLD && outer == 1 && inner >= 1 {
        dst.par_chunks_mut(inner)
            .zip(resolved.par_iter())
            .for_each(|(d_blk, &r)| {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src[r * inner..].as_ptr(),
                        d_blk.as_mut_ptr() as *mut T,
                        inner,
                    );
                }
            });
    } else {
        for o in 0..outer {
            let s_row = &src[o * src_row_len..(o + 1) * src_row_len];
            let d_row = &mut dst[o * dst_row_len..(o + 1) * dst_row_len];
            for (j, &r) in resolved.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        s_row[r * inner..].as_ptr(),
                        d_row[j * inner..].as_mut_ptr() as *mut T,
                        inner,
                    );
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

fn check_along_axis_shapes(
    arr_shape: &[usize],
    idx_shape: &[usize],
    axis: usize,
) -> Result<Vec<usize>> {
    if arr_shape.len() != idx_shape.len() {
        return Err(NumPyError::ShapeMismatch(format!(
            "`indices` and `arr` must have the same number of dimensions ({} vs {})",
            idx_shape.len(),
            arr_shape.len()
        )));
    }
    let ndim = arr_shape.len();
    let mut a_rest = arr_shape.to_vec();
    let mut i_rest = idx_shape.to_vec();
    a_rest[axis] = 1;
    i_rest[axis] = 1;
    let mut out_shape = broadcast_shapes(&a_rest, &i_rest)?;
    if ndim > 0 {
        out_shape[axis] = idx_shape[axis];
    }
    Ok(out_shape)
}

/// Takes values from input array by matching 1-D index and data slices along `axis`.
pub fn take_along_axis<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    indices: &NdArray<i64>,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    let ax = match axis {
        None => {
            if indices.ndim() != 1 {
                return Err(NumPyError::InvalidValue(
                    "indices must be 1D when axis=None in take_along_axis".into(),
                ));
            }
            let flat = arr.ravel(Order::C);
            return take_along_axis(&flat, indices, Some(0));
        }
        Some(a) => a,
    };

    if arr.ndim() == 0 {
        return Err(NumPyError::InvalidAxis { axis: ax, ndim: 0 });
    }
    let norm_ax = normalize_axis(ax, arr.ndim())?;
    let out_shape = check_along_axis_shapes(arr.shape(), indices.shape(), norm_ax)?;
    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let dim = arr.shape()[norm_ax];
    let mut idx_bshape = out_shape.clone();
    idx_bshape[norm_ax] = indices.shape()[norm_ax];
    let idx_b = indices.broadcast_to(&idx_bshape)?.to_c_contiguous();
    let idx_sl = idx_b.as_slice().unwrap_or(&[]);
    let mut resolved = Vec::with_capacity(total);
    for &raw in idx_sl {
        resolved.push(resolve_index_mode(raw, dim, IndexMode::Raise, true)?);
    }

    let mut arr_bshape = out_shape.clone();
    arr_bshape[norm_ax] = dim;
    let arr_b = arr.broadcast_to(&arr_bshape)?.to_c_contiguous();
    let src = arr_b.as_slice().unwrap_or(&[]);

    let outer: usize = out_shape[..norm_ax].iter().product();
    let idx_ax_len = out_shape[norm_ax];
    let inner: usize = out_shape[norm_ax + 1..].iter().product();

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    let src_row_stride = dim * inner;
    let dst_row_stride = idx_ax_len * inner;

    if total >= RAYON_THRESHOLD && outer > 1 {
        dst.par_chunks_mut(dst_row_stride)
            .enumerate()
            .for_each(|(o, d_row)| {
                let s_base = o * src_row_stride;
                let r_base = o * dst_row_stride;
                for i in 0..idx_ax_len {
                    for j in 0..inner {
                        let sel = resolved[r_base + i * inner + j];
                        d_row[i * inner + j].write(src[s_base + sel * inner + j]);
                    }
                }
            });
    } else {
        for o in 0..outer {
            let s_base = o * src_row_stride;
            let r_base = o * dst_row_stride;
            for i in 0..idx_ax_len {
                for j in 0..inner {
                    let sel = resolved[r_base + i * inner + j];
                    dst[r_base + i * inner + j].write(src[s_base + sel * inner + j]);
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Replaces specified elements of `arr` with `v` at flat C-order indices `ind`.
pub fn put<T: Scalar>(
    arr: &mut NdArray<T>,
    ind: &NdArray<i64>,
    v: &NdArray<T>,
    mode: IndexMode,
) -> Result<()> {
    let n_ind = ind.len();
    if n_ind == 0 || v.is_empty() {
        return Ok(());
    }
    let total = arr.len();
    let idx_c = ind.to_c_contiguous();
    let idx_sl = idx_c.as_slice().unwrap_or(&[]);
    let mut resolved = Vec::with_capacity(n_ind);
    for &raw in idx_sl {
        resolved.push(resolve_index_mode(raw, total, mode, true)?);
    }

    let v_c = v.to_c_contiguous();
    let v_sl = v_c.as_slice().unwrap_or(&[]);
    let v_len = v_sl.len();

    let mut view_mut = arr.view_mut()?;
    if let Some(dst_sl) = view_mut.as_slice_mut() {
        for (k, &pos) in resolved.iter().enumerate() {
            dst_sl[pos] = v_sl[k % v_len];
        }
    } else {
        let shape = view_mut.shape().to_vec();
        let strides = view_mut.strides().to_vec();
        let base = view_mut.as_mut_ptr();
        let mut coords = vec![0usize; shape.len()];
        for (k, &pos) in resolved.iter().enumerate() {
            linear_to_coords(pos, &shape, &mut coords);
            let off = coords_offset(&coords, &strides);
            unsafe {
                *base.offset(off) = v_sl[k % v_len];
            }
        }
    }
    Ok(())
}

/// Puts values into destination array by matching 1-D index and data slices along `axis`.
pub fn put_along_axis<T: Scalar>(
    arr: &mut NdArray<T>,
    indices: &NdArray<i64>,
    values: &NdArray<T>,
    axis: Option<isize>,
) -> Result<()> {
    if axis.is_none() {
        if indices.ndim() != 1 || arr.ndim() != 1 {
            return Err(NumPyError::InvalidValue(
                "axis=None in put_along_axis requires 1-D arrays".into(),
            ));
        }
        return put_along_axis(arr, indices, values, Some(0));
    }
    let ax = axis.unwrap_or(0);
    if arr.ndim() == 0 {
        return Err(NumPyError::InvalidAxis { axis: ax, ndim: 0 });
    }
    let norm_ax = normalize_axis(ax, arr.ndim())?;
    let dim = arr.shape()[norm_ax];
    if indices.ndim() != arr.ndim() {
        return Err(NumPyError::ShapeMismatch(format!(
            "indices must have same ndim as arr ({} vs {})",
            indices.ndim(),
            arr.ndim()
        )));
    }
    let mut b_shape = arr.shape().to_vec();
    b_shape[norm_ax] = indices.shape()[norm_ax];
    let idx_b = indices.broadcast_to(&b_shape)?.to_c_contiguous();
    let val_b = values.broadcast_to(&b_shape)?.to_c_contiguous();
    let idx_sl = idx_b.as_slice().unwrap_or(&[]);
    let val_sl = val_b.as_slice().unwrap_or(&[]);

    let mut resolved = Vec::with_capacity(idx_sl.len());
    for &raw in idx_sl {
        resolved.push(resolve_index_mode(raw, dim, IndexMode::Raise, true)?);
    }

    let mut view_mut = arr.view_mut()?;
    let shape = view_mut.shape().to_vec();
    let strides = view_mut.strides().to_vec();
    let base = view_mut.as_mut_ptr();
    let mut coords = vec![0usize; shape.len()];
    for (linear, (&sel, &val)) in resolved.iter().zip(val_sl.iter()).enumerate() {
        linear_to_coords(linear, &b_shape, &mut coords);
        coords[norm_ax] = sel;
        let off = coords_offset(&coords, &strides);
        unsafe {
            *base.offset(off) = val;
        }
    }
    Ok(())
}

/// Changes elements of `arr` based on `mask` and `values`.
/// Matches NumPy `np.putmask`: at C-order flat index `i` where `mask[i]` is true,
/// assigns `values[i % values.len()]`.
pub fn putmask<T: Scalar>(
    arr: &mut NdArray<T>,
    mask: &NdArray<bool>,
    values: &NdArray<T>,
) -> Result<()> {
    let shape = arr.shape().to_vec();
    if mask.len() != arr.len() {
        return Err(NumPyError::InvalidValue(
            "putmask: mask and data must be the same size".into(),
        ));
    }
    let mask_b = mask.ravel(Order::C);
    let mask_sl = mask_b.as_slice().unwrap_or(&[]);
    if !mask_sl.iter().any(|&b| b) {
        return Ok(());
    }
    if values.is_empty() {
        return Err(NumPyError::InvalidValue(
            "putmask: values array cannot be empty when mask has True entries".into(),
        ));
    }
    let v_c = values.to_c_contiguous();
    let v_sl = v_c.as_slice().unwrap_or(&[]);
    let v_len = v_sl.len();

    let mut view_mut = arr.view_mut()?;
    if let Some(dst_sl) = view_mut.as_slice_mut() {
        for (flat_i, (&m, slot)) in mask_sl.iter().zip(dst_sl.iter_mut()).enumerate() {
            if m {
                *slot = v_sl[flat_i % v_len];
            }
        }
    } else {
        let strides = view_mut.strides().to_vec();
        let base = view_mut.as_mut_ptr();
        let mut coords = vec![0usize; shape.len()];
        for (flat_i, &m) in mask_sl.iter().enumerate() {
            if m {
                linear_to_coords(flat_i, &shape, &mut coords);
                let off = coords_offset(&coords, &strides);
                unsafe {
                    *base.offset(off) = v_sl[flat_i % v_len];
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// 3. Selection: compress, extract, choose, where_select, select
// =============================================================================

/// Returns selected slices of `a` along `axis` (or flattened `a` when `axis == None`).
pub fn compress<T: Scalar + Send + Sync>(
    condition: &NdArray<bool>,
    a: &NdArray<T>,
    axis: Option<isize>,
) -> Result<NdArray<T>> {
    if condition.ndim() != 1 {
        return Err(NumPyError::InvalidValue(
            "condition must be a 1-D array".into(),
        ));
    }
    let (work, norm_ax) = match axis {
        None => (a.ravel(Order::C), 0usize),
        Some(ax) => {
            if a.ndim() == 0 {
                return Err(NumPyError::InvalidAxis { axis: ax, ndim: 0 });
            }
            let na = normalize_axis(ax, a.ndim())?;
            (a.clone(), na)
        }
    };

    let dim = work.shape()[norm_ax];
    if condition.len() > dim {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "boolean index did not match indexed axis along dimension {norm_ax}; \
             dimension is {dim} but corresponding boolean dimension is {}",
            condition.len()
        )));
    }
    let cond_c = condition.to_c_contiguous();
    let cond_sl = cond_c.as_slice().unwrap_or(&[]);
    let selected: Vec<i64> = cond_sl
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i as i64) } else { None })
        .collect();
    let idx_arr = NdArray::from_vec(selected);
    take(&work, &idx_arr, Some(norm_ax as isize), IndexMode::Raise)
}

/// Returns elements of `arr` satisfying `condition` as a 1-D array.
pub fn extract<T: Scalar + Send + Sync>(
    condition: &NdArray<bool>,
    arr: &NdArray<T>,
) -> Result<NdArray<T>> {
    let flat_arr = arr.ravel(Order::C);
    let flat_cond = condition.ravel(Order::C);
    compress(&flat_cond, &flat_arr, Some(0))
}

/// Construct an array from an index array `a` and a set of `choices` arrays to choose from.
pub fn choose<T: Scalar + Send + Sync>(
    a: &NdArray<i64>,
    choices: &[&NdArray<T>],
    mode: IndexMode,
) -> Result<NdArray<T>> {
    if choices.is_empty() {
        return Err(NumPyError::InvalidValue(
            "choices sequence cannot be empty".into(),
        ));
    }
    let n_choices = choices.len();
    let mut shapes: Vec<&[usize]> = Vec::with_capacity(n_choices + 1);
    shapes.push(a.shape());
    for &c in choices {
        shapes.push(c.shape());
    }
    let out_shape = broadcast_many_shapes(&shapes)?;
    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let a_b = a.broadcast_to(&out_shape)?.to_c_contiguous();
    let a_sl = a_b.as_slice().unwrap_or(&[]);
    let mut resolved = Vec::with_capacity(total);
    for &raw in a_sl {
        resolved.push(resolve_index_mode(raw, n_choices, mode, false)?);
    }

    let c_bcast: Vec<NdArray<T>> = choices
        .iter()
        .map(|c| Ok(c.broadcast_to(&out_shape)?.to_c_contiguous()))
        .collect::<Result<_>>()?;
    let c_slices: Vec<&[T]> = c_bcast
        .iter()
        .map(|c| c.as_slice().unwrap_or(&[]))
        .collect();

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD {
        dst.par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                slot.write(c_slices[resolved[i]][i]);
            });
    } else {
        for (i, slot) in dst.iter_mut().enumerate() {
            slot.write(c_slices[resolved[i]][i]);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Element-wise ternary selection `condition ? x : y` (equivalent to `numpy.where(condition, x, y)`).
pub fn where_select<T: Scalar + Send + Sync>(
    condition: &NdArray<bool>,
    x: &NdArray<T>,
    y: &NdArray<T>,
) -> Result<NdArray<T>> {
    let out_shape = broadcast_many_shapes(&[condition.shape(), x.shape(), y.shape()])?;
    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let c_b = condition.broadcast_to(&out_shape)?;
    let x_b = x.broadcast_to(&out_shape)?;
    let y_b = y.broadcast_to(&out_shape)?;

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    if let (Some(cs), Some(xs), Some(ys)) = (c_b.as_slice(), x_b.as_slice(), y_b.as_slice()) {
        if total >= RAYON_THRESHOLD {
            dst.par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * 4096;
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        let k = base + j;
                        slot.write(if cs[k] { xs[k] } else { ys[k] });
                    }
                });
        } else {
            for i in 0..total {
                dst[i].write(if cs[i] { xs[i] } else { ys[i] });
            }
        }
    } else {
        let c_v = c_b.view();
        let x_v = x_b.view();
        let y_v = y_b.view();
        let ndim = out_shape.len();
        let c_st = c_v.strides();
        let x_st = x_v.strides();
        let y_st = y_v.strides();
        let c_base = c_v.as_ptr() as usize;
        let x_base = x_v.as_ptr() as usize;
        let y_base = y_v.as_ptr() as usize;

        if total >= RAYON_THRESHOLD && ndim > 0 {
            const CHUNK: usize = 2048;
            dst.par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * CHUNK;
                    let mut coords = vec![0usize; ndim];
                    linear_to_coords(start, &out_shape, &mut coords);
                    let mut cp = (c_base as *const bool).wrapping_offset(coords_offset(&coords, c_st));
                    let mut xp = (x_base as *const T).wrapping_offset(coords_offset(&coords, x_st));
                    let mut yp = (y_base as *const T).wrapping_offset(coords_offset(&coords, y_st));
                    let chunk_len = chunk.len();
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let val = unsafe { if *cp { *xp } else { *yp } };
                        slot.write(val);
                        if i + 1 < chunk_len {
                            let prev = coords.clone();
                            for d in (0..ndim).rev() {
                                coords[d] += 1;
                                if coords[d] < out_shape[d] {
                                    break;
                                }
                                coords[d] = 0;
                            }
                            for d in 0..ndim {
                                let delta = (coords[d] as isize) - (prev[d] as isize);
                                if delta != 0 {
                                    cp = unsafe { cp.offset(delta * c_st[d]) };
                                    xp = unsafe { xp.offset(delta * x_st[d]) };
                                    yp = unsafe { yp.offset(delta * y_st[d]) };
                                }
                            }
                        }
                    }
                });
        } else {
            let mut coords = vec![0usize; ndim];
            let mut cp = c_v.as_ptr();
            let mut xp = x_v.as_ptr();
            let mut yp = y_v.as_ptr();
            for (i, slot) in dst.iter_mut().enumerate() {
                slot.write(unsafe { if *cp { *xp } else { *yp } });
                if i + 1 < total {
                    let mut c1 = coords.clone();
                    let mut c2 = coords.clone();
                    advance_coords_ptr(&mut coords, &mut cp, &out_shape, c_st);
                    advance_coords_ptr(&mut c1, &mut xp, &out_shape, x_st);
                    advance_coords_ptr(&mut c2, &mut yp, &out_shape, y_st);
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Multi-way conditional selection matching `numpy.select(condlist, choicelist, default)`.
pub fn select<T: Scalar + Send + Sync>(
    condlist: &[&NdArray<bool>],
    choicelist: &[&NdArray<T>],
    default: T,
) -> Result<NdArray<T>> {
    if condlist.len() != choicelist.len() {
        return Err(NumPyError::InvalidValue(format!(
            "list of cases ({}) must be same length as list of conditions ({})",
            choicelist.len(),
            condlist.len()
        )));
    }
    if condlist.is_empty() {
        return Err(NumPyError::InvalidValue(
            "select with an empty condition list is not possible".into(),
        ));
    }

    let mut shapes: Vec<&[usize]> = Vec::with_capacity(condlist.len() * 2);
    for &c in condlist {
        shapes.push(c.shape());
    }
    for &ch in choicelist {
        shapes.push(ch.shape());
    }
    let out_shape = broadcast_many_shapes(&shapes)?;
    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let conds_b: Vec<NdArray<bool>> = condlist
        .iter()
        .map(|c| Ok(c.broadcast_to(&out_shape)?.to_c_contiguous()))
        .collect::<Result<_>>()?;
    let choices_b: Vec<NdArray<T>> = choicelist
        .iter()
        .map(|ch| Ok(ch.broadcast_to(&out_shape)?.to_c_contiguous()))
        .collect::<Result<_>>()?;
    let c_slices: Vec<&[bool]> = conds_b
        .iter()
        .map(|c| c.as_slice().unwrap_or(&[]))
        .collect();
    let v_slices: Vec<&[T]> = choices_b
        .iter()
        .map(|c| c.as_slice().unwrap_or(&[]))
        .collect();

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD {
        dst.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * 4096;
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let i = base + j;
                    let mut chosen = default;
                    for (cs, vs) in c_slices.iter().zip(v_slices.iter()) {
                        if cs[i] {
                            chosen = vs[i];
                            break;
                        }
                    }
                    slot.write(chosen);
                }
            });
    } else {
        for (i, slot) in dst.iter_mut().enumerate() {
            let mut chosen = default;
            for (cs, vs) in c_slices.iter().zip(v_slices.iter()) {
                if cs[i] {
                    chosen = vs[i];
                    break;
                }
            }
            slot.write(chosen);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

// =============================================================================
// 4. Non-zero Searches: nonzero, argwhere, flatnonzero
// =============================================================================

/// Returns indices of elements that are non-zero (`truthy`), one 1-D `NdArray<i64>` per axis.
pub fn nonzero<T: Scalar + Send + Sync>(arr: &NdArray<T>) -> Result<Vec<NdArray<i64>>> {
    let ndim = arr.ndim();
    if ndim == 0 {
        return Err(NumPyError::InvalidValue(
            "Calling nonzero on 0d arrays is not allowed. Use atleast_1d(arr) instead.".into(),
        ));
    }
    let total = arr.len();
    if total == 0 {
        return (0..ndim)
            .map(|_| finish_c_array(AlignedBuffer::<i64>::new_zeroed(0), &[0]))
            .collect();
    }

    let c_arr = arr.to_c_contiguous();
    let sl = c_arr.as_slice().unwrap_or(&[]);
    let shape = c_arr.shape();

    if total >= RAYON_THRESHOLD {
        const CHUNK: usize = 4096;
        let per_chunk: Vec<Vec<usize>> = sl
            .par_chunks(CHUNK)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base = chunk_idx * CHUNK;
                let mut local = Vec::new();
                for (i, v) in chunk.iter().enumerate() {
                    if v.is_truthy() {
                        local.push(base + i);
                    }
                }
                local
            })
            .collect();

        let nnz: usize = per_chunk.iter().map(|v| v.len()).sum();
        if nnz == 0 {
            return (0..ndim)
                .map(|_| finish_c_array(AlignedBuffer::<i64>::new_zeroed(0), &[0]))
                .collect();
        }
        let mut uninits: Vec<_> = (0..ndim)
            .map(|_| AlignedBuffer::<i64>::new_uninit(nnz))
            .collect();
        let mut cursor = 0usize;
        let mut coords = vec![0usize; ndim];
        for chunk_idxs in per_chunk {
            for linear in chunk_idxs {
                linear_to_coords(linear, shape, &mut coords);
                for d in 0..ndim {
                    uninits[d].as_mut_slice()[cursor].write(coords[d] as i64);
                }
                cursor += 1;
            }
        }
        return uninits
            .into_iter()
            .map(|u| {
                let b = unsafe { u.assume_init() };
                finish_c_array(b, &[nnz])
            })
            .collect();
    }

    let mut flat_hits = Vec::new();
    for (i, v) in sl.iter().enumerate() {
        if v.is_truthy() {
            flat_hits.push(i);
        }
    }
    let nnz = flat_hits.len();
    let mut bufs: Vec<_> = (0..ndim)
        .map(|_| AlignedBuffer::<i64>::new_uninit(nnz))
        .collect();
    let mut coords = vec![0usize; ndim];
    for (pos, linear) in flat_hits.into_iter().enumerate() {
        linear_to_coords(linear, shape, &mut coords);
        for d in 0..ndim {
            bufs[d].as_mut_slice()[pos].write(coords[d] as i64);
        }
    }
    bufs.into_iter()
        .map(|u| {
            let b = unsafe { u.assume_init() };
            finish_c_array(b, &[nnz])
        })
        .collect()
}

/// Returns indices of elements that are non-zero, grouped by element as `(nnz, ndim)` `NdArray<i64>`.
pub fn argwhere<T: Scalar + Send + Sync>(arr: &NdArray<T>) -> Result<NdArray<i64>> {
    let ndim = arr.ndim();
    if ndim == 0 {
        let v = arr.get(&[])?;
        let rows = usize::from(v.is_truthy());
        return finish_c_array(AlignedBuffer::<i64>::new_zeroed(0), &[rows, 0]);
    }
    let nz = nonzero(arr)?;
    let nnz = nz[0].len();
    if nnz == 0 {
        return finish_c_array(AlignedBuffer::<i64>::new_zeroed(0), &[0, ndim]);
    }
    let total = nnz * ndim;
    let mut uninit = AlignedBuffer::<i64>::new_uninit(total);
    let dst = uninit.as_mut_slice();
    let col_slices: Vec<&[i64]> = nz.iter().map(|c| c.as_slice().unwrap_or(&[])).collect();

    if total >= RAYON_THRESHOLD {
        dst.par_chunks_mut(ndim)
            .enumerate()
            .for_each(|(r, row)| {
                for d in 0..ndim {
                    row[d].write(col_slices[d][r]);
                }
            });
    } else {
        for r in 0..nnz {
            let off = r * ndim;
            for d in 0..ndim {
                dst[off + d].write(col_slices[d][r]);
            }
        }
    }
    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &[nnz, ndim])
}

/// Returns flat C-order indices of non-zero elements in `arr`.
pub fn flatnonzero<T: Scalar + Send + Sync>(arr: &NdArray<T>) -> Result<NdArray<i64>> {
    let flat = arr.ravel(Order::C);
    let mut res = nonzero(&flat)?;
    Ok(res.swap_remove(0))
}

// =============================================================================
// 5. Diagonals: diagonal, fill_diagonal
// =============================================================================

/// Returns specified diagonals (`axis1`, `axis2` moved to last axis), matching `numpy.diagonal`.
pub fn diagonal<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<NdArray<T>> {
    let ndim = arr.ndim();
    if ndim < 2 {
        return Err(NumPyError::InvalidValue(format!(
            "diagonal requires an array of at least two dimensions, got {ndim}-D"
        )));
    }
    let ax1 = normalize_axis(axis1, ndim)?;
    let ax2 = normalize_axis(axis2, ndim)?;
    if ax1 == ax2 {
        return Err(NumPyError::InvalidValue(
            "axis1 and axis2 cannot be the same".into(),
        ));
    }

    let shape = arr.shape();
    let d1 = shape[ax1];
    let d2 = shape[ax2];
    let r0 = if offset < 0 { offset.unsigned_abs() } else { 0 };
    let c0 = if offset > 0 { offset as usize } else { 0 };
    let diag_len = if r0 >= d1 || c0 >= d2 {
        0
    } else {
        (d1 - r0).min(d2 - c0)
    };

    let mut outer_axes = Vec::with_capacity(ndim - 2);
    let mut out_shape = Vec::with_capacity(ndim - 1);
    for d in 0..ndim {
        if d != ax1 && d != ax2 {
            outer_axes.push(d);
            out_shape.push(shape[d]);
        }
    }
    out_shape.push(diag_len);

    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let view = arr.view();
    let strides = view.strides();
    let s1 = strides[ax1];
    let s2 = strides[ax2];
    let diag_step = s1 + s2;
    let base_ptr = unsafe { view.as_ptr().offset((r0 as isize) * s1 + (c0 as isize) * s2) };

    let outer_shape: Vec<usize> = outer_axes.iter().map(|&d| shape[d]).collect();
    let outer_strides: Vec<isize> = outer_axes.iter().map(|&d| strides[d]).collect();
    let outer_count: usize = outer_shape.iter().product();

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD && outer_count > 1 {
        let base_addr = base_ptr as usize;
        dst.par_chunks_mut(diag_len)
            .enumerate()
            .for_each(|(o, d_row)| {
                let mut coords = vec![0usize; outer_shape.len()];
                linear_to_coords(o, &outer_shape, &mut coords);
                let row_ptr = (base_addr as *const T)
                    .wrapping_offset(coords_offset(&coords, &outer_strides));
                for (k, slot) in d_row.iter_mut().enumerate() {
                    slot.write(unsafe { *row_ptr.offset((k as isize) * diag_step) });
                }
            });
    } else if outer_count == 1 && diag_len >= RAYON_THRESHOLD {
        let base_addr = base_ptr as usize;
        dst.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * 4096;
                let ptr = base_addr as *const T;
                for (i, slot) in chunk.iter_mut().enumerate() {
                    slot.write(unsafe { *ptr.offset(((start + i) as isize) * diag_step) });
                }
            });
    } else {
        let mut coords = vec![0usize; outer_shape.len()];
        let mut row_ptr = base_ptr;
        for o in 0..outer_count {
            let d_row = &mut dst[o * diag_len..(o + 1) * diag_len];
            for (k, slot) in d_row.iter_mut().enumerate() {
                slot.write(unsafe { *row_ptr.offset((k as isize) * diag_step) });
            }
            if o + 1 < outer_count {
                advance_coords_ptr(&mut coords, &mut row_ptr, &outer_shape, &outer_strides);
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Fills the main diagonal of an array (`ndim >= 2`) in place.
pub fn fill_diagonal<T: Scalar>(
    arr: &mut NdArray<T>,
    val: &NdArray<T>,
    wrap: bool,
) -> Result<()> {
    let ndim = arr.ndim();
    if ndim < 2 {
        return Err(NumPyError::InvalidValue(format!(
            "array must be at least 2-d, got {ndim}-D"
        )));
    }
    let shape = arr.shape().to_vec();
    if ndim > 2 && shape.iter().any(|&d| d != shape[0]) {
        return Err(NumPyError::ShapeMismatch(
            "All dimensions of input must be of equal length for ndim > 2".into(),
        ));
    }
    if val.is_empty() || arr.is_empty() {
        return Ok(());
    }

    let val_c = val.to_c_contiguous();
    let v_sl = val_c.as_slice().unwrap_or(&[]);
    let v_len = v_sl.len();
    let mut view_mut = arr.view_mut()?;
    let strides = view_mut.strides().to_vec();
    let base = view_mut.as_mut_ptr();

    if ndim == 2 {
        let rows = shape[0];
        let cols = shape[1];
        if !wrap || rows <= cols {
            let count = rows.min(cols);
            let step = strides[0] + strides[1];
            for i in 0..count {
                unsafe {
                    *base.offset((i as isize) * step) = v_sl[i % v_len];
                }
            }
        } else {
            // Wrap down tall 2-D matrix (`step = cols + 1` in flat C-order)
            let flat_len = rows * cols;
            let flat_step = cols + 1;
            let mut k = 0usize;
            let mut flat_idx = 0usize;
            while flat_idx < flat_len {
                let r = flat_idx / cols;
                let c = flat_idx % cols;
                let off = (r as isize) * strides[0] + (c as isize) * strides[1];
                unsafe {
                    *base.offset(off) = v_sl[k % v_len];
                }
                k += 1;
                flat_idx += flat_step;
            }
        }
    } else {
        let step: isize = strides.iter().sum();
        for i in 0..shape[0] {
            unsafe {
                *base.offset((i as isize) * step) = v_sl[i % v_len];
            }
        }
    }
    Ok(())
}

// =============================================================================
// 6. Coordinate Conversions: unravel_index, ravel_multi_index
// =============================================================================

/// Converts flat C/F-order indices into a tuple of coordinate arrays for `shape`.
pub fn unravel_index(
    indices: &NdArray<i64>,
    shape: &[usize],
    order: Order,
) -> Result<Vec<NdArray<i64>>> {
    if !matches!(order, Order::C | Order::F) {
        return Err(NumPyError::InvalidValue(
            "only 'C' or 'F' order is permitted".into(),
        ));
    }
    let ndim = shape.len();
    let size = shape_size(shape)?;
    let out_shape = indices.shape().to_vec();
    let n = indices.len();

    let idx_c = indices.to_c_contiguous();
    let idx_sl = idx_c.as_slice().unwrap_or(&[]);

    if ndim == 0 {
        for &raw in idx_sl {
            if raw != 0 {
                return Err(NumPyError::IndexOutOfBounds(format!(
                    "index {raw} is out of bounds for array with size 1"
                )));
            }
        }
        return Ok(Vec::new());
    }

    for &raw in idx_sl {
        if raw < 0 || (raw as u64) >= (size as u64) {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "index {raw} is out of bounds for array with size {size}"
            )));
        }
    }

    if n == 0 {
        return (0..ndim)
            .map(|_| finish_c_array(AlignedBuffer::<i64>::new_zeroed(0), &out_shape))
            .collect();
    }

    let mut out_vecs: Vec<AlignedBuffer<i64>> = (0..ndim)
        .map(|_| AlignedBuffer::<i64>::new_zeroed(n))
        .collect();

    let use_f = matches!(order, Order::F);

    if n * ndim >= RAYON_THRESHOLD {
        // Build transposed chunk results cleanly then copy or write per-axis parallel
        for (axis, buf) in out_vecs.iter_mut().enumerate() {
            let dim = shape[axis] as i64;
            let div: i64 = if use_f {
                shape[..axis].iter().product::<usize>() as i64
            } else {
                shape[axis + 1..].iter().product::<usize>() as i64
            };
            buf.as_mut_slice()
                .par_iter_mut()
                .zip(idx_sl.par_iter())
                .for_each(|(slot, &lin)| {
                    *slot = (lin / div) % dim;
                });
        }
    } else {
        for (axis, buf) in out_vecs.iter_mut().enumerate() {
            let dim = shape[axis] as i64;
            let div: i64 = if use_f {
                shape[..axis].iter().product::<usize>() as i64
            } else {
                shape[axis + 1..].iter().product::<usize>() as i64
            };
            let sl = buf.as_mut_slice();
            for (k, &lin) in idx_sl.iter().enumerate() {
                sl[k] = (lin / div) % dim;
            }
        }
    }

    out_vecs
        .into_iter()
        .map(|b| finish_c_array(b, &out_shape))
        .collect()
}

/// Converts a tuple of index arrays into an array of flat indices in `dims` of given `order`.
pub fn ravel_multi_index(
    multi_index: &[&NdArray<i64>],
    dims: &[usize],
    mode: IndexMode,
    order: Order,
) -> Result<NdArray<i64>> {
    if multi_index.len() != dims.len() {
        return Err(NumPyError::InvalidValue(format!(
            "parameter multi_index must be a sequence of length {}, got {}",
            dims.len(),
            multi_index.len()
        )));
    }
    if dims.is_empty() {
        return finish_c_array(AlignedBuffer::<i64>::new_zeroed(1), &[]);
    }

    let shapes: Vec<&[usize]> = multi_index.iter().map(|a| a.shape()).collect();
    let out_shape = broadcast_many_shapes(&shapes)?;

    // Reject oversized dims (product of non-zero dimensions > isize::MAX) before empty short-circuit,
    // matching NumPy's check even when multi_index arrays or other dims are empty.
    let mut nonzero_prod: i128 = 1;
    for &d in dims {
        if d != 0 {
            nonzero_prod = nonzero_prod
                .checked_mul(d as i128)
                .filter(|&p| p <= isize::MAX as i128)
                .ok_or_else(|| {
                    NumPyError::InvalidValue(
                        "invalid dims: array size defined by dims is larger than the maximum possible size."
                            .into(),
                    )
                })?;
        }
    }

    let n = shape_size(&out_shape)?;
    if n == 0 {
        return finish_c_array(AlignedBuffer::<i64>::new_zeroed(0), &out_shape);
    }
    if dims.contains(&0) {
        return Err(NumPyError::InvalidValue(
            "cannot unravel if shape has zero entries (is empty).".into(),
        ));
    }

    let bcast: Vec<NdArray<i64>> = multi_index
        .iter()
        .map(|a| Ok(a.broadcast_to(&out_shape)?.to_c_contiguous()))
        .collect::<Result<_>>()?;
    let idx_slices: Vec<&[i64]> = bcast
        .iter()
        .map(|a| a.as_slice().unwrap_or(&[]))
        .collect();

    let mut strides = vec![1i64; dims.len()];
    if matches!(order, Order::F) {
        let mut acc: i128 = 1;
        for d in 0..dims.len() {
            strides[d] = acc as i64;
            acc *= dims[d] as i128;
        }
    } else {
        let mut acc: i128 = 1;
        for d in (0..dims.len()).rev() {
            strides[d] = acc as i64;
            acc *= dims[d] as i128;
        }
    }

    // Validate all indices before assembling linear offsets
    for (d, (&dim, &sl)) in dims.iter().zip(idx_slices.iter()).enumerate() {
        let _ = d;
        for &raw in sl {
            let _ = resolve_index_mode(raw, dim, mode, false)?;
        }
    }

    let mut uninit = AlignedBuffer::<i64>::new_uninit(n);
    let dst = uninit.as_mut_slice();

    if n * dims.len() >= RAYON_THRESHOLD {
        dst.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * 4096;
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let k = base + j;
                    let mut sum: i128 = 0;
                    for (d, (&dim, &sl)) in dims.iter().zip(idx_slices.iter()).enumerate() {
                        let r = match mode {
                            IndexMode::Raise => sl[k],
                            IndexMode::Wrap => sl[k].rem_euclid(dim as i64),
                            IndexMode::Clip => sl[k].clamp(0, (dim - 1) as i64),
                        };
                        sum += (r as i128) * (strides[d] as i128);
                    }
                    slot.write(sum as i64);
                }
            });
    } else {
        for (k, slot) in dst.iter_mut().enumerate() {
            let mut sum: i128 = 0;
            for (d, (&dim, &sl)) in dims.iter().zip(idx_slices.iter()).enumerate() {
                let r = match mode {
                    IndexMode::Raise => sl[k],
                    IndexMode::Wrap => sl[k].rem_euclid(dim as i64),
                    IndexMode::Clip => sl[k].clamp(0, (dim - 1) as i64),
                };
                sum += (r as i128) * (strides[d] as i128);
            }
            slot.write(sum as i64);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

// =============================================================================
// 7. Fancy Integer & Boolean Mask Indexing
// =============================================================================

/// Boolean mask indexing `arr[mask]`, collecting elements along the masked prefix dimensions.
pub fn boolean_mask<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    mask: &NdArray<bool>,
) -> Result<NdArray<T>> {
    if mask.ndim() == 0 {
        let flag = mask.get(&[])?;
        let mut out_shape = Vec::with_capacity(arr.ndim() + 1);
        out_shape.push(usize::from(flag));
        out_shape.extend_from_slice(arr.shape());
        if !flag || arr.is_empty() {
            return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
        }
        let mut uninit = AlignedBuffer::<T>::new_uninit(arr.len());
        copy_view_to_uninit(&arr.view(), uninit.as_mut_slice());
        let buf = unsafe { uninit.assume_init() };
        return finish_c_array(buf, &out_shape);
    }

    let m_ndim = mask.ndim();
    if m_ndim > arr.ndim() {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "too many boolean indices for array: mask has {m_ndim} dims, array has {}",
            arr.ndim()
        )));
    }
    for d in 0..m_ndim {
        if mask.shape()[d] != arr.shape()[d] {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "boolean index did not match indexed array along axis {d}; size of axis is {} but size of corresponding boolean axis is {}",
                arr.shape()[d],
                mask.shape()[d]
            )));
        }
    }

    let tail_shape = &arr.shape()[m_ndim..];
    let tail_len: usize = tail_shape.iter().product();

    let mask_c = mask.to_c_contiguous();
    let mask_sl = mask_c.as_slice().unwrap_or(&[]);
    let true_positions: Vec<usize> = mask_sl
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    let mut out_shape = Vec::with_capacity(1 + tail_shape.len());
    out_shape.push(true_positions.len());
    out_shape.extend_from_slice(tail_shape);

    let total = shape_size(&out_shape)?;
    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let arr_c = arr.to_c_contiguous();
    let src = arr_c.as_slice().unwrap_or(&[]);

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD {
        if tail_len == 1 {
            dst.par_iter_mut()
                .zip(true_positions.par_iter())
                .for_each(|(slot, &pos)| {
                    slot.write(src[pos]);
                });
        } else {
            dst.par_chunks_mut(tail_len)
                .zip(true_positions.par_iter())
                .for_each(|(d_blk, &pos)| unsafe {
                    std::ptr::copy_nonoverlapping(
                        src[pos * tail_len..].as_ptr(),
                        d_blk.as_mut_ptr() as *mut T,
                        tail_len,
                    );
                });
        }
    } else if tail_len == 1 {
        for (slot, &pos) in dst.iter_mut().zip(true_positions.iter()) {
            slot.write(src[pos]);
        }
    } else {
        for (d_blk, &pos) in dst.chunks_mut(tail_len).zip(true_positions.iter()) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src[pos * tail_len..].as_ptr(),
                    d_blk.as_mut_ptr() as *mut T,
                    tail_len,
                );
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Boolean mask assignment `arr[mask] = values` (broadcasting `values`).
pub fn boolean_mask_set<T: Scalar>(
    arr: &mut NdArray<T>,
    mask: &NdArray<bool>,
    values: &NdArray<T>,
) -> Result<()> {
    let m_ndim = mask.ndim();
    if m_ndim > arr.ndim() {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "too many boolean indices for array: mask has {m_ndim} dims, array has {}",
            arr.ndim()
        )));
    }
    for d in 0..m_ndim {
        if mask.shape()[d] != arr.shape()[d] {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "boolean index did not match indexed array along axis {d}"
            )));
        }
    }

    if m_ndim == 0 {
        if !mask.get(&[])? {
            return Ok(());
        }
        let target_shape = arr.shape().to_vec();
        let val_b = values.broadcast_to(&target_shape)?.to_c_contiguous();
        let val_sl = val_b.as_slice().unwrap_or(&[]);
        let mut view_mut = arr.view_mut()?;
        if let Some(dst) = view_mut.as_slice_mut() {
            dst.copy_from_slice(val_sl);
        } else {
            let strides = view_mut.strides().to_vec();
            let base = view_mut.as_mut_ptr();
            let mut coords = vec![0usize; target_shape.len()];
            for (i, &v) in val_sl.iter().enumerate() {
                linear_to_coords(i, &target_shape, &mut coords);
                unsafe {
                    *base.offset(coords_offset(&coords, &strides)) = v;
                }
            }
        }
        return Ok(());
    }

    let mask_c = mask.to_c_contiguous();
    let mask_sl = mask_c.as_slice().unwrap_or(&[]);
    let true_positions: Vec<usize> = mask_sl
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    let tail_shape = &arr.shape()[m_ndim..];
    let mut target_shape = Vec::with_capacity(1 + tail_shape.len());
    target_shape.push(true_positions.len());
    target_shape.extend_from_slice(tail_shape);

    let val_b = values.broadcast_to(&target_shape)?.to_c_contiguous();
    let val_sl = val_b.as_slice().unwrap_or(&[]);
    if true_positions.is_empty() || arr.is_empty() {
        return Ok(());
    }

    let tail_len: usize = tail_shape.iter().product();
    let arr_shape = arr.shape().to_vec();
    let mut view_mut = arr.view_mut()?;
    if let Some(dst) = view_mut.as_slice_mut() {
        for (k, &pos) in true_positions.iter().enumerate() {
            dst[pos * tail_len..(pos + 1) * tail_len]
                .copy_from_slice(&val_sl[k * tail_len..(k + 1) * tail_len]);
        }
    } else {
        let strides = view_mut.strides().to_vec();
        let base = view_mut.as_mut_ptr();
        let mut coords = vec![0usize; arr_shape.len()];
        for (k, &pos) in true_positions.iter().enumerate() {
            let flat_start = pos * tail_len;
            for j in 0..tail_len {
                linear_to_coords(flat_start + j, &arr_shape, &mut coords);
                let off = coords_offset(&coords, &strides);
                unsafe {
                    *base.offset(off) = val_sl[k * tail_len + j];
                }
            }
        }
    }
    Ok(())
}

/// Advanced integer array indexing `arr[idx_0, idx_1, ...]` along the leading axes.
pub fn index_fancy<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    indices: &[&NdArray<i64>],
) -> Result<NdArray<T>> {
    if indices.is_empty() {
        return Ok(arr.clone());
    }
    let k = indices.len();
    if k > arr.ndim() {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "too many indices for array: array is {}-dimensional, but {k} were indexed",
            arr.ndim()
        )));
    }

    let idx_shapes: Vec<&[usize]> = indices.iter().map(|i| i.shape()).collect();
    let bshape = broadcast_many_shapes(&idx_shapes)?;
    let n_idx = shape_size(&bshape)?;

    let mut out_shape = bshape.clone();
    out_shape.extend_from_slice(&arr.shape()[k..]);
    let total = shape_size(&out_shape)?;

    // Broadcast and validate all index arrays upfront
    let bcast_idxs: Vec<NdArray<i64>> = indices
        .iter()
        .map(|i| Ok(i.broadcast_to(&bshape)?.to_c_contiguous()))
        .collect::<Result<_>>()?;
    let idx_slices: Vec<&[i64]> = bcast_idxs
        .iter()
        .map(|i| i.as_slice().unwrap_or(&[]))
        .collect();

    let mut resolved_per_axis: Vec<Vec<usize>> = Vec::with_capacity(k);
    for (ax, sl) in idx_slices.iter().enumerate() {
        let dim = arr.shape()[ax];
        let mut r = Vec::with_capacity(n_idx);
        for &raw in *sl {
            r.push(resolve_index_mode(raw, dim, IndexMode::Raise, true)?);
        }
        resolved_per_axis.push(r);
    }

    if total == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &out_shape);
    }

    let tail_shape = &arr.shape()[k..];
    let tail_len: usize = tail_shape.iter().product();
    let head_strides = compute_c_unit_strides(&arr.shape()[..k]);

    let arr_c = arr.to_c_contiguous();
    let src = arr_c.as_slice().unwrap_or(&[]);

    let mut uninit = AlignedBuffer::<T>::new_uninit(total);
    let dst = uninit.as_mut_slice();

    let compute_flat_head = |i: usize| -> usize {
        let mut off = 0usize;
        for ax in 0..k {
            off += resolved_per_axis[ax][i] * head_strides[ax];
        }
        off
    };

    if total >= RAYON_THRESHOLD {
        if tail_len == 1 {
            dst.par_iter_mut()
                .enumerate()
                .for_each(|(i, slot)| {
                    slot.write(src[compute_flat_head(i)]);
                });
        } else {
            dst.par_chunks_mut(tail_len)
                .enumerate()
                .for_each(|(i, d_blk)| {
                    let head_off = compute_flat_head(i) * tail_len;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src[head_off..].as_ptr(),
                            d_blk.as_mut_ptr() as *mut T,
                            tail_len,
                        );
                    }
                });
        }
    } else if tail_len == 1 {
        for (i, slot) in dst.iter_mut().enumerate() {
            slot.write(src[compute_flat_head(i)]);
        }
    } else {
        for (i, d_blk) in dst.chunks_mut(tail_len).enumerate() {
            let head_off = compute_flat_head(i) * tail_len;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src[head_off..].as_ptr(),
                    d_blk.as_mut_ptr() as *mut T,
                    tail_len,
                );
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Advanced integer array assignment `arr[idx_0, idx_1, ...] = values` (last write wins on duplicates).
pub fn index_fancy_set<T: Scalar>(
    arr: &mut NdArray<T>,
    indices: &[&NdArray<i64>],
    values: &NdArray<T>,
) -> Result<()> {
    if indices.is_empty() {
        let target_shape = arr.shape().to_vec();
        let val_b = values.broadcast_to(&target_shape)?.to_c_contiguous();
        let val_sl = val_b.as_slice().unwrap_or(&[]);
        let mut vm = arr.view_mut()?;
        if let Some(dst) = vm.as_slice_mut() {
            dst.copy_from_slice(val_sl);
        } else {
            let strides = vm.strides().to_vec();
            let base = vm.as_mut_ptr();
            let mut coords = vec![0usize; target_shape.len()];
            for (i, &v) in val_sl.iter().enumerate() {
                linear_to_coords(i, &target_shape, &mut coords);
                unsafe {
                    *base.offset(coords_offset(&coords, &strides)) = v;
                }
            }
        }
        return Ok(());
    }

    let k = indices.len();
    if k > arr.ndim() {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "too many indices for array: array is {}-dimensional, but {k} were indexed",
            arr.ndim()
        )));
    }

    let idx_shapes: Vec<&[usize]> = indices.iter().map(|i| i.shape()).collect();
    let bshape = broadcast_many_shapes(&idx_shapes)?;
    let n_idx = shape_size(&bshape)?;

    let mut target_shape = bshape.clone();
    target_shape.extend_from_slice(&arr.shape()[k..]);

    let bcast_idxs: Vec<NdArray<i64>> = indices
        .iter()
        .map(|i| Ok(i.broadcast_to(&bshape)?.to_c_contiguous()))
        .collect::<Result<_>>()?;
    let idx_slices: Vec<&[i64]> = bcast_idxs
        .iter()
        .map(|i| i.as_slice().unwrap_or(&[]))
        .collect();

    let mut resolved_per_axis: Vec<Vec<usize>> = Vec::with_capacity(k);
    for (ax, sl) in idx_slices.iter().enumerate() {
        let dim = arr.shape()[ax];
        let mut r = Vec::with_capacity(n_idx);
        for &raw in *sl {
            r.push(resolve_index_mode(raw, dim, IndexMode::Raise, true)?);
        }
        resolved_per_axis.push(r);
    }

    let val_b = values.broadcast_to(&target_shape)?.to_c_contiguous();
    let val_sl = val_b.as_slice().unwrap_or(&[]);

    let tail_shape = arr.shape()[k..].to_vec();
    let tail_len: usize = tail_shape.iter().product();
    if n_idx == 0 || tail_len == 0 {
        return Ok(());
    }

    let arr_shape = arr.shape().to_vec();
    let head_strides = compute_c_unit_strides(&arr_shape[..k]);
    let mut vm = arr.view_mut()?;

    if let Some(dst) = vm.as_slice_mut() {
        for i in 0..n_idx {
            let mut head_off = 0usize;
            for ax in 0..k {
                head_off += resolved_per_axis[ax][i] * head_strides[ax];
            }
            dst[head_off * tail_len..(head_off + 1) * tail_len]
                .copy_from_slice(&val_sl[i * tail_len..(i + 1) * tail_len]);
        }
    } else {
        let strides = vm.strides().to_vec();
        let base = vm.as_mut_ptr();
        let mut coords = vec![0usize; arr_shape.len()];
        for i in 0..n_idx {
            for ax in 0..k {
                coords[ax] = resolved_per_axis[ax][i];
            }
            for j in 0..tail_len {
                linear_to_coords(j, &tail_shape, &mut coords[k..]);
                let off = coords_offset(&coords, &strides);
                unsafe {
                    *base.offset(off) = val_sl[i * tail_len + j];
                }
            }
        }
    }
    Ok(())
}

/// Full mixed advanced indexing supporting `Int`, `Slice`, `NewAxis`, `IntArray`, and `BoolMask`.
/// Follows NumPy's basic-vs-advanced subspace placement rules.
pub fn index_advanced<T: Scalar + Send + Sync>(
    arr: &NdArray<T>,
    args: &[FancyArg<'_>],
) -> Result<NdArray<T>> {
    // Fast zero-copy path when all args are basic (Int, Slice, NewAxis)
    let has_advanced = args
        .iter()
        .any(|a| matches!(a, FancyArg::IntArray(_) | FancyArg::BoolMask(_)));
    if !has_advanced {
        let basic: Vec<SliceArg> = args
            .iter()
            .map(|a| match a {
                FancyArg::Int(i) => SliceArg::Index(*i),
                FancyArg::Slice(s) => SliceArg::Slice(*s),
                FancyArg::NewAxis => SliceArg::NewAxis,
                FancyArg::IntArray(_) | FancyArg::BoolMask(_) => unreachable!(),
            })
            .collect();
        return arr.slice(&basic);
    }

    // Convert BoolMask / IntArray / NewAxis / Slice / Int specs using exact NumPy subspace rules
    let mut consumed_dims = 0usize;
    for a in args {
        match a {
            FancyArg::Int(_) | FancyArg::Slice(_) | FancyArg::IntArray(_) => consumed_dims += 1,
            FancyArg::BoolMask(m) => {
                if m.ndim() == 0 {
                    // Treated as NewAxis-like filter
                    consumed_dims += 0;
                } else {
                    consumed_dims += m.ndim();
                }
            }
            FancyArg::NewAxis => {}
        }
    }
    if consumed_dims > arr.ndim() {
        return Err(NumPyError::IndexOutOfBounds(format!(
            "too many indices for array: array is {}-D, but {consumed_dims} axes indexed",
            arr.ndim()
        )));
    }

    // First apply basic slices and `Int` reductions across axes that only have basic Slice/Int,
    // leaving fancy axes intact; wait: transpose fancy axes to front when separated by Slice/NewAxis.
    // Implement exact NumPy subspace rule directly:
    let mut cur_axis = 0usize;
    let mut slice_args = Vec::new();
    let mut int_arrays: Vec<(usize, NdArray<i64>)> = Vec::new();
    let mut adv_out_positions = Vec::new();
    let mut adv_arg_positions = Vec::new();
    let mut out_axis = 0usize;

    for (arg_pos, a) in args.iter().enumerate() {
        match a {
            FancyArg::NewAxis => {
                slice_args.push(SliceArg::NewAxis);
                out_axis += 1;
            }
            FancyArg::Int(idx) => {
                slice_args.push(SliceArg::Index(*idx));
                adv_arg_positions.push(arg_pos);
                cur_axis += 1;
            }
            FancyArg::Slice(sl) => {
                slice_args.push(SliceArg::Slice(*sl));
                cur_axis += 1;
                out_axis += 1;
            }
            FancyArg::IntArray(ia) => {
                slice_args.push(SliceArg::Slice(Slice::full()));
                int_arrays.push((out_axis, (*ia).clone()));
                adv_out_positions.push(out_axis);
                adv_arg_positions.push(arg_pos);
                cur_axis += 1;
                out_axis += 1;
            }
            FancyArg::BoolMask(m) => {
                if m.ndim() == 0 {
                    let flag = m.get(&[])?;
                    slice_args.push(SliceArg::NewAxis);
                    let empty_or_zero = if flag {
                        NdArray::from_vec(vec![0i64])
                    } else {
                        NdArray::from_vec(Vec::<i64>::new())
                    };
                    // 0D bool creates 1-D size 0 or 1 broadcast dimension of index 0 into NewAxis(size 1)
                    int_arrays.push((out_axis, empty_or_zero));
                    adv_out_positions.push(out_axis);
                    adv_arg_positions.push(arg_pos);
                    out_axis += 1;
                } else {
                    for d in 0..m.ndim() {
                        if m.shape()[d] != arr.shape()[cur_axis + d] {
                            return Err(NumPyError::IndexOutOfBounds(format!(
                                "boolean index did not match axis {}; expected {}, got {}",
                                cur_axis + d,
                                arr.shape()[cur_axis + d],
                                m.shape()[d]
                            )));
                        }
                        slice_args.push(SliceArg::Slice(Slice::full()));
                    }
                    let nz = nonzero(m)?;
                    for nz_ax in nz {
                        int_arrays.push((out_axis, nz_ax));
                        adv_out_positions.push(out_axis);
                        out_axis += 1;
                    }
                    adv_arg_positions.push(arg_pos);
                    cur_axis += m.ndim();
                }
            }
        }
    }

    // Apply all basic Slice/Int/NewAxis transforms first (zero-copy view)
    let sliced = arr.slice(&slice_args)?;
    let s_ndim = sliced.ndim();

    // Check if all advanced/scalar-int indices form a contiguous block in `args`
    let is_contiguous = adv_arg_positions
        .windows(2)
        .all(|w| w[1] == w[0] + 1);
    let insert_pos = if is_contiguous {
        adv_out_positions[0]
    } else {
        0
    };

    // Move all `adv_out_positions` to the front of `sliced` so `index_fancy` can gather directly
    let mut perm: Vec<isize> = Vec::with_capacity(s_ndim);
    for &p in &adv_out_positions {
        perm.push(p as isize);
    }
    for d in 0..s_ndim {
        if !adv_out_positions.contains(&d) {
            perm.push(d as isize);
        }
    }
    let transposed = sliced.transpose(Some(&perm))?;
    let i_refs: Vec<&NdArray<i64>> = int_arrays.iter().map(|(_, a)| a).collect();
    let gathered = index_fancy(&transposed, &i_refs)?;

    let b_ndim = gathered.ndim() - (s_ndim - adv_out_positions.len());
    if insert_pos == 0 || b_ndim == 0 {
        return Ok(gathered);
    }

    // Reorder the front `b_ndim` broadcast dimensions back to `insert_pos` (zero-copy transpose!)
    let non_adv_before = adv_out_positions[0];
    let g_ndim = gathered.ndim();
    let mut final_perm: Vec<isize> = Vec::with_capacity(g_ndim);
    for i in 0..non_adv_before {
        final_perm.push((b_ndim + i) as isize);
    }
    for i in 0..b_ndim {
        final_perm.push(i as isize);
    }
    for i in (b_ndim + non_adv_before)..g_ndim {
        final_perm.push(i as isize);
    }
    gathered.transpose(Some(&final_perm))
}

// =============================================================================
// 8. DynArray Indexing & Selection Functions
// =============================================================================

/// Dynamic `take`.
pub fn dyn_take(
    arr: &DynArray,
    indices: &DynArray,
    axis: Option<isize>,
    mode: IndexMode,
) -> Result<DynArray> {
    let idx = dyn_to_i64_indices(indices)?;
    dispatch_dyn!(arr, a => take(a, &idx, axis, mode).map(DynArray::from))
}

/// Dynamic `take_along_axis`.
pub fn dyn_take_along_axis(
    arr: &DynArray,
    indices: &DynArray,
    axis: Option<isize>,
) -> Result<DynArray> {
    let idx = dyn_to_i64_indices(indices)?;
    dispatch_dyn!(arr, a => take_along_axis(a, &idx, axis).map(DynArray::from))
}

/// Dynamic `put` (casts `v` to `arr.dtype()` cleanly).
pub fn dyn_put(
    arr: &mut DynArray,
    ind: &DynArray,
    v: &DynArray,
    mode: IndexMode,
) -> Result<()> {
    let idx = dyn_to_i64_indices(ind)?;
    let target_dt = arr.dtype();
    let v_cast = if v.dtype() == target_dt {
        v.clone()
    } else {
        v.astype(target_dt, Casting::Unsafe)?
    };
    macro_rules! do_put {
        ($var:ident) => {
            match (arr, v_cast) {
                (DynArray::$var(a), DynArray::$var(vc)) => put(a, &idx, &vc, mode),
                _ => Err(NumPyError::DTypeError("internal dtype mismatch".into())),
            }
        };
    }
    match target_dt {
        DType::Bool => do_put!(Bool),
        DType::I8 => do_put!(I8),
        DType::I16 => do_put!(I16),
        DType::I32 => do_put!(I32),
        DType::I64 => do_put!(I64),
        DType::U8 => do_put!(U8),
        DType::U16 => do_put!(U16),
        DType::U32 => do_put!(U32),
        DType::U64 => do_put!(U64),
        DType::F16 => do_put!(F16),
        DType::F32 => do_put!(F32),
        DType::F64 => do_put!(F64),
        DType::C64 => do_put!(C64),
        DType::C128 => do_put!(C128),
    }
}

/// Dynamic `put_along_axis`.
pub fn dyn_put_along_axis(
    arr: &mut DynArray,
    indices: &DynArray,
    values: &DynArray,
    axis: Option<isize>,
) -> Result<()> {
    let idx = dyn_to_i64_indices(indices)?;
    let target_dt = arr.dtype();
    let v_cast = if values.dtype() == target_dt {
        values.clone()
    } else {
        values.astype(target_dt, Casting::Unsafe)?
    };
    macro_rules! do_put_aa {
        ($var:ident) => {
            match (arr, v_cast) {
                (DynArray::$var(a), DynArray::$var(vc)) => put_along_axis(a, &idx, &vc, axis),
                _ => Err(NumPyError::DTypeError("internal dtype mismatch".into())),
            }
        };
    }
    match target_dt {
        DType::Bool => do_put_aa!(Bool),
        DType::I8 => do_put_aa!(I8),
        DType::I16 => do_put_aa!(I16),
        DType::I32 => do_put_aa!(I32),
        DType::I64 => do_put_aa!(I64),
        DType::U8 => do_put_aa!(U8),
        DType::U16 => do_put_aa!(U16),
        DType::U32 => do_put_aa!(U32),
        DType::U64 => do_put_aa!(U64),
        DType::F16 => do_put_aa!(F16),
        DType::F32 => do_put_aa!(F32),
        DType::F64 => do_put_aa!(F64),
        DType::C64 => do_put_aa!(C64),
        DType::C128 => do_put_aa!(C128),
    }
}

/// Dynamic `putmask`.
pub fn dyn_putmask(arr: &mut DynArray, mask: &DynArray, values: &DynArray) -> Result<()> {
    let m_bool = if mask.dtype() == DType::Bool {
        mask.clone()
    } else {
        mask.astype(DType::Bool, Casting::Unsafe)?
    };
    let mask_ref = dyn_as_bool(&m_bool)?;
    let target_dt = arr.dtype();
    let v_cast = if values.dtype() == target_dt {
        values.clone()
    } else {
        values.astype(target_dt, Casting::Unsafe)?
    };
    macro_rules! do_pm {
        ($var:ident) => {
            match (arr, v_cast) {
                (DynArray::$var(a), DynArray::$var(vc)) => putmask(a, mask_ref, &vc),
                _ => Err(NumPyError::DTypeError("internal dtype mismatch".into())),
            }
        };
    }
    match target_dt {
        DType::Bool => do_pm!(Bool),
        DType::I8 => do_pm!(I8),
        DType::I16 => do_pm!(I16),
        DType::I32 => do_pm!(I32),
        DType::I64 => do_pm!(I64),
        DType::U8 => do_pm!(U8),
        DType::U16 => do_pm!(U16),
        DType::U32 => do_pm!(U32),
        DType::U64 => do_pm!(U64),
        DType::F16 => do_pm!(F16),
        DType::F32 => do_pm!(F32),
        DType::F64 => do_pm!(F64),
        DType::C64 => do_pm!(C64),
        DType::C128 => do_pm!(C128),
    }
}

/// Dynamic `compress`.
pub fn dyn_compress(
    condition: &DynArray,
    a: &DynArray,
    axis: Option<isize>,
) -> Result<DynArray> {
    let cond = if condition.dtype() == DType::Bool {
        condition.clone()
    } else {
        condition.astype(DType::Bool, Casting::Unsafe)?
    };
    let c_ref = dyn_as_bool(&cond)?;
    dispatch_dyn!(a, arr => compress(c_ref, arr, axis).map(DynArray::from))
}

/// Dynamic `extract`.
pub fn dyn_extract(condition: &DynArray, arr: &DynArray) -> Result<DynArray> {
    let cond = if condition.dtype() == DType::Bool {
        condition.clone()
    } else {
        condition.astype(DType::Bool, Casting::Unsafe)?
    };
    let c_ref = dyn_as_bool(&cond)?;
    dispatch_dyn!(arr, a => extract(c_ref, a).map(DynArray::from))
}

/// Dynamic `choose` (promotes choice array dtypes via `result_type`).
pub fn dyn_choose(a: &DynArray, choices: &[&DynArray], mode: IndexMode) -> Result<DynArray> {
    if choices.is_empty() {
        return Err(NumPyError::InvalidValue("choices sequence cannot be empty".into()));
    }
    let idx = dyn_to_i64_indices(a)?;
    let dtypes: Vec<DType> = choices.iter().map(|c| c.dtype()).collect();
    let target = numpy_types::result_type(&dtypes)?;
    let cast: Vec<DynArray> = choices
        .iter()
        .map(|&c| {
            if c.dtype() == target {
                Ok(c.clone())
            } else {
                c.astype(target, Casting::Unsafe)
            }
        })
        .collect::<Result<_>>()?;

    macro_rules! do_choose {
        ($t:ty, $var:ident) => {{
            let refs: Vec<&NdArray<$t>> = cast
                .iter()
                .map(|d| match d {
                    DynArray::$var(arr) => Ok(arr),
                    _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                })
                .collect::<Result<_>>()?;
            Ok(DynArray::$var(choose(&idx, &refs, mode)?))
        }};
    }

    match target {
        DType::Bool => do_choose!(bool, Bool),
        DType::I8 => do_choose!(i8, I8),
        DType::I16 => do_choose!(i16, I16),
        DType::I32 => do_choose!(i32, I32),
        DType::I64 => do_choose!(i64, I64),
        DType::U8 => do_choose!(u8, U8),
        DType::U16 => do_choose!(u16, U16),
        DType::U32 => do_choose!(u32, U32),
        DType::U64 => do_choose!(u64, U64),
        DType::F16 => do_choose!(f16, F16),
        DType::F32 => do_choose!(f32, F32),
        DType::F64 => do_choose!(f64, F64),
        DType::C64 => do_choose!(Complex32, C64),
        DType::C128 => do_choose!(Complex64, C128),
    }
}

/// Dynamic `where_select` (promotes `x` and `y` via `promote_types`).
pub fn dyn_where_select(
    condition: &DynArray,
    x: &DynArray,
    y: &DynArray,
) -> Result<DynArray> {
    let cond = if condition.dtype() == DType::Bool {
        condition.clone()
    } else {
        condition.astype(DType::Bool, Casting::Unsafe)?
    };
    let c_ref = dyn_as_bool(&cond)?;
    let target = numpy_types::promote_types(x.dtype(), y.dtype());
    let xc = if x.dtype() == target {
        x.clone()
    } else {
        x.astype(target, Casting::Unsafe)?
    };
    let yc = if y.dtype() == target {
        y.clone()
    } else {
        y.astype(target, Casting::Unsafe)?
    };

    macro_rules! do_ws {
        ($var:ident) => {
            match (&xc, &yc) {
                (DynArray::$var(xa), DynArray::$var(ya)) => {
                    Ok(DynArray::$var(where_select(c_ref, xa, ya)?))
                }
                _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
            }
        };
    }

    match target {
        DType::Bool => do_ws!(Bool),
        DType::I8 => do_ws!(I8),
        DType::I16 => do_ws!(I16),
        DType::I32 => do_ws!(I32),
        DType::I64 => do_ws!(I64),
        DType::U8 => do_ws!(U8),
        DType::U16 => do_ws!(U16),
        DType::U32 => do_ws!(U32),
        DType::U64 => do_ws!(U64),
        DType::F16 => do_ws!(F16),
        DType::F32 => do_ws!(F32),
        DType::F64 => do_ws!(F64),
        DType::C64 => do_ws!(C64),
        DType::C128 => do_ws!(C128),
    }
}

/// Dynamic `select`.
pub fn dyn_select(
    condlist: &[&DynArray],
    choicelist: &[&DynArray],
    default: DynScalar,
) -> Result<DynArray> {
    if condlist.len() != choicelist.len() {
        return Err(NumPyError::InvalidValue(
            "list of cases must be same length as list of conditions".into(),
        ));
    }
    if condlist.is_empty() {
        return Err(NumPyError::InvalidValue(
            "select with an empty condition list is not possible".into(),
        ));
    }

    let conds: Vec<NdArray<bool>> = condlist
        .iter()
        .map(|&c| {
            let b = if c.dtype() == DType::Bool {
                c.clone()
            } else {
                c.astype(DType::Bool, Casting::Unsafe)?
            };
            Ok(dyn_as_bool(&b)?.clone())
        })
        .collect::<Result<_>>()?;
    let c_refs: Vec<&NdArray<bool>> = conds.iter().collect();

    let mut dtypes: Vec<DType> = choicelist.iter().map(|c| c.dtype()).collect();
    dtypes.push(default.dtype());
    let target = numpy_types::result_type(&dtypes)?;

    let cast: Vec<DynArray> = choicelist
        .iter()
        .map(|&c| {
            if c.dtype() == target {
                Ok(c.clone())
            } else {
                c.astype(target, Casting::Unsafe)
            }
        })
        .collect::<Result<_>>()?;

    macro_rules! do_sel {
        ($t:ty, $var:ident) => {{
            let refs: Vec<&NdArray<$t>> = cast
                .iter()
                .map(|d| match d {
                    DynArray::$var(arr) => Ok(arr),
                    _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
                })
                .collect::<Result<_>>()?;
            Ok(DynArray::$var(select(&c_refs, &refs, default.cast_to::<$t>())?))
        }};
    }

    match target {
        DType::Bool => do_sel!(bool, Bool),
        DType::I8 => do_sel!(i8, I8),
        DType::I16 => do_sel!(i16, I16),
        DType::I32 => do_sel!(i32, I32),
        DType::I64 => do_sel!(i64, I64),
        DType::U8 => do_sel!(u8, U8),
        DType::U16 => do_sel!(u16, U16),
        DType::U32 => do_sel!(u32, U32),
        DType::U64 => do_sel!(u64, U64),
        DType::F16 => do_sel!(f16, F16),
        DType::F32 => do_sel!(f32, F32),
        DType::F64 => do_sel!(f64, F64),
        DType::C64 => do_sel!(Complex32, C64),
        DType::C128 => do_sel!(Complex64, C128),
    }
}

/// Dynamic `nonzero` returning `Vec<DynArray::I64>`.
pub fn dyn_nonzero(arr: &DynArray) -> Result<Vec<DynArray>> {
    let parts = dispatch_dyn!(arr, a => nonzero(a))?;
    Ok(parts.into_iter().map(DynArray::I64).collect())
}

/// Dynamic `argwhere` returning `DynArray::I64`.
pub fn dyn_argwhere(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => argwhere(a).map(DynArray::I64))
}

/// Dynamic `flatnonzero` returning `DynArray::I64`.
pub fn dyn_flatnonzero(arr: &DynArray) -> Result<DynArray> {
    dispatch_dyn!(arr, a => flatnonzero(a).map(DynArray::I64))
}

/// Dynamic `diagonal`.
pub fn dyn_diagonal(
    arr: &DynArray,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<DynArray> {
    dispatch_dyn!(arr, a => diagonal(a, offset, axis1, axis2).map(DynArray::from))
}

/// Dynamic `fill_diagonal`.
pub fn dyn_fill_diagonal(arr: &mut DynArray, val: &DynArray, wrap: bool) -> Result<()> {
    let target = arr.dtype();
    let vc = if val.dtype() == target {
        val.clone()
    } else {
        val.astype(target, Casting::Unsafe)?
    };
    macro_rules! do_fd {
        ($var:ident) => {
            match (arr, vc) {
                (DynArray::$var(a), DynArray::$var(v)) => fill_diagonal(a, &v, wrap),
                _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
            }
        };
    }
    match target {
        DType::Bool => do_fd!(Bool),
        DType::I8 => do_fd!(I8),
        DType::I16 => do_fd!(I16),
        DType::I32 => do_fd!(I32),
        DType::I64 => do_fd!(I64),
        DType::U8 => do_fd!(U8),
        DType::U16 => do_fd!(U16),
        DType::U32 => do_fd!(U32),
        DType::U64 => do_fd!(U64),
        DType::F16 => do_fd!(F16),
        DType::F32 => do_fd!(F32),
        DType::F64 => do_fd!(F64),
        DType::C64 => do_fd!(C64),
        DType::C128 => do_fd!(C128),
    }
}

/// Dynamic `unravel_index`.
pub fn dyn_unravel_index(
    indices: &DynArray,
    shape: &[usize],
    order: Order,
) -> Result<Vec<DynArray>> {
    let idx = dyn_to_i64_indices(indices)?;
    Ok(unravel_index(&idx, shape, order)?
        .into_iter()
        .map(DynArray::I64)
        .collect())
}

/// Dynamic `ravel_multi_index`.
pub fn dyn_ravel_multi_index(
    multi_index: &[&DynArray],
    dims: &[usize],
    mode: IndexMode,
    order: Order,
) -> Result<DynArray> {
    let conv: Vec<NdArray<i64>> = multi_index
        .iter()
        .map(|&a| dyn_to_i64_indices(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<i64>> = conv.iter().collect();
    ravel_multi_index(&refs, dims, mode, order).map(DynArray::I64)
}

/// Dynamic `boolean_mask`.
pub fn dyn_boolean_mask(arr: &DynArray, mask: &DynArray) -> Result<DynArray> {
    let m = dyn_as_bool(mask)?;
    dispatch_dyn!(arr, a => boolean_mask(a, m).map(DynArray::from))
}

/// Dynamic `boolean_mask_set`.
pub fn dyn_boolean_mask_set(
    arr: &mut DynArray,
    mask: &DynArray,
    values: &DynArray,
) -> Result<()> {
    let m = dyn_as_bool(mask)?;
    let target = arr.dtype();
    let vc = if values.dtype() == target {
        values.clone()
    } else {
        values.astype(target, Casting::Unsafe)?
    };
    macro_rules! do_bms {
        ($var:ident) => {
            match (arr, vc) {
                (DynArray::$var(a), DynArray::$var(v)) => boolean_mask_set(a, m, &v),
                _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
            }
        };
    }
    match target {
        DType::Bool => do_bms!(Bool),
        DType::I8 => do_bms!(I8),
        DType::I16 => do_bms!(I16),
        DType::I32 => do_bms!(I32),
        DType::I64 => do_bms!(I64),
        DType::U8 => do_bms!(U8),
        DType::U16 => do_bms!(U16),
        DType::U32 => do_bms!(U32),
        DType::U64 => do_bms!(U64),
        DType::F16 => do_bms!(F16),
        DType::F32 => do_bms!(F32),
        DType::F64 => do_bms!(F64),
        DType::C64 => do_bms!(C64),
        DType::C128 => do_bms!(C128),
    }
}

/// Dynamic `index_fancy`.
pub fn dyn_index_fancy(arr: &DynArray, indices: &[&DynArray]) -> Result<DynArray> {
    let conv: Vec<NdArray<i64>> = indices
        .iter()
        .map(|&a| dyn_to_i64_indices(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<i64>> = conv.iter().collect();
    dispatch_dyn!(arr, a => index_fancy(a, &refs).map(DynArray::from))
}

/// Dynamic `index_fancy_set`.
pub fn dyn_index_fancy_set(
    arr: &mut DynArray,
    indices: &[&DynArray],
    values: &DynArray,
) -> Result<()> {
    let conv: Vec<NdArray<i64>> = indices
        .iter()
        .map(|&a| dyn_to_i64_indices(a))
        .collect::<Result<_>>()?;
    let refs: Vec<&NdArray<i64>> = conv.iter().collect();
    let target = arr.dtype();
    let vc = if values.dtype() == target {
        values.clone()
    } else {
        values.astype(target, Casting::Unsafe)?
    };
    macro_rules! do_ifs {
        ($var:ident) => {
            match (arr, vc) {
                (DynArray::$var(a), DynArray::$var(v)) => index_fancy_set(a, &refs, &v),
                _ => Err(NumPyError::DTypeError("internal cast mismatch".into())),
            }
        };
    }
    match target {
        DType::Bool => do_ifs!(Bool),
        DType::I8 => do_ifs!(I8),
        DType::I16 => do_ifs!(I16),
        DType::I32 => do_ifs!(I32),
        DType::I64 => do_ifs!(I64),
        DType::U8 => do_ifs!(U8),
        DType::U16 => do_ifs!(U16),
        DType::U32 => do_ifs!(U32),
        DType::U64 => do_ifs!(U64),
        DType::F16 => do_ifs!(F16),
        DType::F32 => do_ifs!(F32),
        DType::F64 => do_ifs!(F64),
        DType::C64 => do_ifs!(C64),
        DType::C128 => do_ifs!(C128),
    }
}

/// Dynamic mixed `index_advanced`.
pub fn dyn_index_advanced(arr: &DynArray, args: &[DynFancyArg<'_>]) -> Result<DynArray> {
    enum OwnedArg {
        Int(isize),
        Slice(Slice),
        NewAxis,
        IntArray(NdArray<i64>),
        BoolMask(NdArray<bool>),
    }
    let owned: Vec<OwnedArg> = args
        .iter()
        .map(|a| match a {
            DynFancyArg::Int(i) => Ok(OwnedArg::Int(*i)),
            DynFancyArg::Slice(s) => Ok(OwnedArg::Slice(*s)),
            DynFancyArg::NewAxis => Ok(OwnedArg::NewAxis),
            DynFancyArg::IntArray(ia) => Ok(OwnedArg::IntArray(dyn_to_i64_indices(ia)?)),
            DynFancyArg::BoolMask(m) => Ok(OwnedArg::BoolMask(dyn_as_bool(m)?.clone())),
        })
        .collect::<Result<_>>()?;
    let borrowed: Vec<FancyArg<'_>> = owned
        .iter()
        .map(|a| match a {
            OwnedArg::Int(i) => FancyArg::Int(*i),
            OwnedArg::Slice(s) => FancyArg::Slice(*s),
            OwnedArg::NewAxis => FancyArg::NewAxis,
            OwnedArg::IntArray(ia) => FancyArg::IntArray(ia),
            OwnedArg::BoolMask(m) => FancyArg::BoolMask(m),
        })
        .collect();
    dispatch_dyn!(arr, a => index_advanced(a, &borrowed).map(DynArray::from))
}

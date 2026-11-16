use numpy_types::{
    normalize_axis, AlignedBuffer, ArrayView, FloatElement, NdArray, NumPyError, Order, Result,
    Scalar,
};
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::ptr;

const SHIFT_RAYON_THRESHOLD: usize = 65_536;

fn fftfreq_typed<T: FloatElement>(n: usize, d: T) -> Result<NdArray<T>> {
    if n == 0 || d == <T as Scalar>::zero() {
        return Err(NumPyError::InvalidValue(
            "n and d must be non-zero for fftfreq".into(),
        ));
    }
    let nf = T::from_f64(n as f64);
    let val = <T as Scalar>::one() / (nf * d);
    let mid = (n - 1) / 2 + 1;
    let neg_start = -((n / 2) as isize);

    let mut uninit = AlignedBuffer::<T>::new_uninit(n);
    let dst = uninit.as_mut_slice();
    for (i, slot) in dst[..mid].iter_mut().enumerate() {
        slot.write(T::from_f64(i as f64) * val);
    }
    for (k, slot) in dst[mid..].iter_mut().enumerate() {
        slot.write(T::from_f64((neg_start + k as isize) as f64) * val);
    }
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &[n], Order::C)
}

fn rfftfreq_typed<T: FloatElement>(n: usize, d: T) -> Result<NdArray<T>> {
    if n == 0 || d == <T as Scalar>::zero() {
        return Err(NumPyError::InvalidValue(
            "n and d must be non-zero for rfftfreq".into(),
        ));
    }
    let nf = T::from_f64(n as f64);
    let val = <T as Scalar>::one() / (nf * d);
    let out_len = n / 2 + 1;

    let mut uninit = AlignedBuffer::<T>::new_uninit(out_len);
    let dst = uninit.as_mut_slice();
    for (i, slot) in dst.iter_mut().enumerate() {
        slot.write(T::from_f64(i as f64) * val);
    }
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &[out_len], Order::C)
}

/// Return the Discrete Fourier Transform sample frequencies (`numpy.fft.fftfreq`).
pub fn fftfreq(n: usize, d: f64) -> Result<NdArray<f64>> {
    fftfreq_typed(n, d)
}

/// Return the Discrete Fourier Transform sample frequencies for `rfft`/`irfft` (`numpy.fft.rfftfreq`).
pub fn rfftfreq(n: usize, d: f64) -> Result<NdArray<f64>> {
    rfftfreq_typed(n, d)
}

/// Single-precision variant of [`fftfreq`].
pub fn fftfreq_f32(n: usize, d: f32) -> Result<NdArray<f32>> {
    fftfreq_typed(n, d)
}

/// Single-precision variant of [`rfftfreq`].
pub fn rfftfreq_f32(n: usize, d: f32) -> Result<NdArray<f32>> {
    rfftfreq_typed(n, d)
}

pub(crate) fn shift_view<T: Scalar>(
    view: &ArrayView<'_, T>,
    axes: Option<&[isize]>,
    forward: bool,
) -> Result<NdArray<T>> {
    let ndim = view.ndim();
    let shape = view.shape();
    if ndim == 0 && axes.is_none_or(|axes| axes.is_empty()) {
        return Err(NumPyError::InvalidValue("cannot shift an array with no axes".into()));
    }
    let mut shifts = vec![0usize; ndim];

    match axes {
        None => {
            for ax in 0..ndim {
                let dim = shape[ax];
                if dim > 1 {
                    let s = dim / 2;
                    shifts[ax] = if forward { s } else { dim - s };
                }
            }
        }
        Some(ax_list) => {
            for &raw_ax in ax_list {
                let ax = normalize_axis(raw_ax, ndim)?;
                let dim = shape[ax];
                if dim > 1 {
                    let delta = if forward { dim / 2 } else { dim - (dim / 2) };
                    shifts[ax] = (shifts[ax] + delta) % dim;
                }
            }
        }
    }

    if view.is_empty() {
        return NdArray::zeros(shape);
    }
    if shifts.iter().all(|&s| s == 0) {
        return Ok(view.to_owned());
    }

    let total_len = view.len();
    let mut uninit = AlignedBuffer::<T>::new_uninit(total_len);
    let dst = uninit.as_mut_slice();

    if ndim == 1 {
        let n = shape[0];
        let split = (n - shifts[0]) % n;
        let stride = view.strides()[0];
        let src_ptr = view.as_ptr();
        copy_shifted_1d(src_ptr, stride, n, split, dst);
    } else {
        let inner_len = shape[ndim - 1];
        let inner_split = (inner_len - shifts[ndim - 1]) % inner_len;
        let inner_stride = view.strides()[ndim - 1];
        let outer_dims = &shape[..ndim - 1];
        let outer_strides = &view.strides()[..ndim - 1];
        let outer_shifts = &shifts[..ndim - 1];
        let num_rows = total_len / inner_len;
        let base_ptr = SendConstPtr(view.as_ptr());

        if total_len >= SHIFT_RAYON_THRESHOLD && num_rows > 1 {
            let chunk_rows = (4096 / inner_len.max(1)).clamp(1, (num_rows / 4).max(1));
            let chunk_elems = chunk_rows * inner_len;
            dst.par_chunks_mut(chunk_elems)
                .enumerate()
                .for_each(|(chunk_idx, dst_chunk)| {
                    copy_shifted_rows(
                        base_ptr,
                        outer_dims,
                        outer_strides,
                        outer_shifts,
                        inner_stride,
                        inner_len,
                        inner_split,
                        chunk_idx * chunk_rows,
                        dst_chunk,
                    );
                });
        } else {
            copy_shifted_rows(
                base_ptr,
                outer_dims,
                outer_strides,
                outer_shifts,
                inner_stride,
                inner_len,
                inner_split,
                0,
                dst,
            );
        }
    }

    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);

// SAFETY: `ArrayView<'_, T>` guarantees `T: Sync` and immutable shared access for the duration of the borrow.
unsafe impl<T: Sync> Send for SendConstPtr<T> {}
unsafe impl<T: Sync> Sync for SendConstPtr<T> {}

#[allow(clippy::too_many_arguments)]
fn copy_shifted_rows<T: Copy + Sync>(
    base_ptr: SendConstPtr<T>,
    outer_dims: &[usize],
    outer_strides: &[isize],
    outer_shifts: &[usize],
    inner_stride: isize,
    inner_len: usize,
    inner_split: usize,
    start_row: usize,
    dst_slice: &mut [MaybeUninit<T>],
) {
    let rows_in_chunk = dst_slice.len() / inner_len;
    let mut coords = vec![0usize; outer_dims.len()];
    let mut rem = start_row;
    for d in (0..outer_dims.len()).rev() {
        let dim = outer_dims[d];
        coords[d] = rem % dim;
        rem /= dim;
    }
    let raw_base = base_ptr.0;
    for row_offset in 0..rows_in_chunk {
        let mut src_off = 0isize;
        for d in 0..outer_dims.len() {
            let dim = outer_dims[d];
            let src_c = (coords[d] + dim - outer_shifts[d]) % dim;
            src_off += (src_c as isize) * outer_strides[d];
        }
        let row_ptr = unsafe { raw_base.offset(src_off) };
        let dst_row = &mut dst_slice[row_offset * inner_len..(row_offset + 1) * inner_len];
        copy_shifted_1d(row_ptr, inner_stride, inner_len, inner_split, dst_row);

        for d in (0..outer_dims.len()).rev() {
            coords[d] += 1;
            if coords[d] < outer_dims[d] {
                break;
            }
            coords[d] = 0;
        }
    }
}

#[inline]
fn copy_shifted_1d<T: Copy>(
    src_ptr: *const T,
    stride: isize,
    len: usize,
    split: usize,
    dst: &mut [MaybeUninit<T>],
) {
    let first_part = len - split;
    if stride == 1 {
        unsafe {
            let dst_raw = dst.as_mut_ptr() as *mut T;
            if first_part > 0 {
                ptr::copy_nonoverlapping(src_ptr.add(split), dst_raw, first_part);
            }
            if split > 0 {
                ptr::copy_nonoverlapping(src_ptr, dst_raw.add(first_part), split);
            }
        }
    } else {
        for i in 0..first_part {
            let val = unsafe { *src_ptr.offset(((split + i) as isize) * stride) };
            dst[i].write(val);
        }
        for i in 0..split {
            let val = unsafe { *src_ptr.offset((i as isize) * stride) };
            dst[first_part + i].write(val);
        }
    }
}

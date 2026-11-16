//! Shape validation, strided/batched views, 64-byte aligned buffers, and Rayon batch execution.

use faer::{MatMut, MatRef, Par};
use numpy_types::{
    AlignedBuffer, NdArray, NumPyError, Order, Result, Scalar, broadcast_shapes,
};

use crate::traits::LinalgScalar;

/// Minimum `batch_count * work_per_item` before spawning Rayon tasks across O(N^3) batches.
pub const DECOMP_RAYON_THRESHOLD: usize = 4_096;

/// Minimum `batch_count * m * k * n` before spawning Rayon tasks across GEMM batches.
pub const GEMM_RAYON_THRESHOLD: usize = 32_768;

/// Minimum single-matrix `m * k * n` to internally split rows across Rayon for GEMM.
pub const SINGLE_GEMM_RAYON_THRESHOLD: usize = 262_144;

/// Minimum single-matrix `m * n` before giving `faer` multi-threaded `Par`.
pub const SINGLE_FAER_PAR_THRESHOLD: usize = 16_384;

/// Thread-safe read-only pointer wrapper for sharing immutable `NdArray` storage across Rayon workers.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *const T);

unsafe impl<T: Sync> Send for SyncPtr<T> {}
unsafe impl<T: Sync> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    #[inline(always)]
    pub unsafe fn offset(self, count: isize) -> *const T {
        unsafe { self.0.offset(count) }
    }
}

/// Validates that `a` has at least `min_ndim` dimensions (`LinAlgError`, matching NumPy).
#[inline]
pub fn assert_min_ndim<T: Scalar>(a: &NdArray<T>, min_ndim: usize) -> Result<()> {
    if a.ndim() < min_ndim {
        return Err(NumPyError::LinAlgError(format!(
            "{}-dimensional array given. Array must be at least {}-dimensional",
            a.ndim(),
            min_ndim
        )));
    }
    Ok(())
}

/// Validates stacked 2-D matrices `[..., M, N]`, returning `(batch_shape, m, n)`.
#[inline]
pub fn parse_stacked_2d<T: Scalar>(a: &NdArray<T>) -> Result<(&[usize], usize, usize)> {
    assert_min_ndim(a, 2)?;
    let s = a.shape();
    let nd = s.len();
    Ok((&s[..nd - 2], s[nd - 2], s[nd - 1]))
}

/// Validates stacked square matrices `[..., N, N]`, returning `(batch_shape, n)`.
#[inline]
pub fn parse_stacked_square<T: Scalar>(a: &NdArray<T>) -> Result<(&[usize], usize)> {
    let (batch, m, n) = parse_stacked_2d(a)?;
    if m != n {
        return Err(NumPyError::LinAlgError(format!(
            "Last 2 dimensions of the array must be square, got ({m}, {n})"
        )));
    }
    Ok((batch, n))
}

/// Computes product of batch dimensions (returns `1` for empty batch slice `[]`).
#[inline]
pub fn batch_len(batch_shape: &[usize]) -> usize {
    batch_shape
        .iter()
        .copied()
        .fold(1usize, |acc, d| acc.saturating_mul(d))
}

/// Computes element offset in a strided array for a flat `linear_idx` over `shape` & `strides`.
#[inline]
pub fn linear_strided_offset(mut linear_idx: usize, shape: &[usize], strides: &[isize]) -> isize {
    let mut off = 0isize;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim > 1 {
            let coord = linear_idx % dim;
            linear_idx /= dim;
            off += (coord as isize) * strides[i];
        } else {
            linear_idx /= dim.max(1);
        }
    }
    off
}

/// Extracts base pointer, `(row_stride, col_stride)`, batch shape, and batch strides for a matrix stack.
/// Note: when `batch_shape.len() > ndim`, leading axes are prepended with `0` batch stride (for broadcasting).
#[inline]
pub fn matrix_stack_descriptor<T: Scalar>(
    a: &NdArray<T>,
) -> (SyncPtr<T>, isize, isize, &[usize], &[isize]) {
    let nd = a.ndim();
    debug_assert!(nd >= 2);
    let shape = a.shape();
    let strides = a.strides();
    let base = unsafe { a.storage_ptr().add(a.offset()) };
    (
        SyncPtr(base),
        strides[nd - 2],
        strides[nd - 1],
        &shape[..nd - 2],
        &strides[..nd - 2],
    )
}

/// Returns `true` if `arr` is C-contiguous over its whole buffer, returning its slice directly.
#[inline]
pub fn as_c_slice<T: Scalar>(a: &NdArray<T>) -> Option<&[T]> {
    if a.is_c_contiguous() {
        a.as_slice()
    } else {
        None
    }
}

/// Builds a 64-byte aligned C-contiguous `NdArray<T>` from an initialized `AlignedBuffer<T>`.
#[inline]
pub fn ndarray_from_aligned<T: Scalar>(buf: AlignedBuffer<T>, shape: &[usize]) -> Result<NdArray<T>> {
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

/// Allocates a 64-byte aligned zeroed `AlignedBuffer<T>` and converts to `NdArray<T>` after mutation.
#[inline]
pub fn alloc_zeroed_buf<T: Scalar>(len: usize) -> AlignedBuffer<T> {
    AlignedBuffer::new_zeroed(len)
}

/// Fallibly allocates a 64-byte aligned zeroed `AlignedBuffer<T>`, returning `Err` if too big.
#[inline]
pub fn try_alloc_zeroed_buf<T: Scalar>(len: usize) -> Result<AlignedBuffer<T>> {
    AlignedBuffer::try_new_zeroed(len)
}

/// Selects single-matrix `faer` parallelism based on matrix work size.
#[inline]
pub fn single_mat_par(work: usize) -> Par {
    if work >= SINGLE_FAER_PAR_THRESHOLD {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

/// Returns `true` if batch execution should fan out across Rayon threads.
#[inline]
pub fn should_par_batches(batch_count: usize, work_per_item: usize, threshold: usize) -> bool {
    batch_count > 1 && batch_count.saturating_mul(work_per_item) >= threshold
}

/// Constructs a read-only zero-copy `faer::MatRef` at the given pointer and matrix strides.
///
/// # Safety
/// Caller must ensure `ptr` points to a valid readable matrix of dimensions `(m, n)` with
/// element strides `(rsa, csa)` and `m > 0 && n > 0`.
#[inline]
pub unsafe fn mat_ref_from_raw<'a, T>(
    ptr: *const T,
    m: usize,
    n: usize,
    rsa: isize,
    csa: isize,
) -> MatRef<'a, T> {
    unsafe { MatRef::from_raw_parts(ptr, m, n, rsa, csa) }
}

/// Constructs a mutable `faer::MatMut` over a C-contiguous slice of length `m * n`.
#[inline]
pub fn mat_mut_from_c_slice<'a, T>(slice: &'a mut [T], m: usize, n: usize) -> MatMut<'a, T> {
    debug_assert_eq!(slice.len(), m * n);
    unsafe { MatMut::from_raw_parts_mut(slice.as_mut_ptr(), m, n, n as isize, 1) }
}

/// Copies a strided `(m, n)` matrix into a column-major `faer::Mat<T>`.
///
/// # Safety
/// `src_ptr` must be valid for all offsets in `0..m` x `0..n` using `(rsa, csa)`.
#[inline]
pub unsafe fn copy_strided_to_faer_mat<T: Copy>(
    src_ptr: *const T,
    m: usize,
    n: usize,
    rsa: isize,
    csa: isize,
    dst: &mut faer::Mat<T>,
) {
    for c in 0..n {
        let col_ptr = unsafe { src_ptr.offset(c as isize * csa) };
        for r in 0..m {
            dst[(r, c)] = unsafe { *col_ptr.offset(r as isize * rsa) };
        }
    }
}

/// Copies a strided `(m, n)` matrix into a C-contiguous `&mut [T]` slice of length `m * n`.
///
/// # Safety
/// `src_ptr` must be valid for all offsets in `0..m` x `0..n` using `(rsa, csa)`.
#[inline]
pub unsafe fn copy_strided_mat_to_c<T: Copy>(
    src_ptr: *const T,
    m: usize,
    n: usize,
    rsa: isize,
    csa: isize,
    dst: &mut [T],
) {
    debug_assert_eq!(dst.len(), m * n);
    if rsa == n as isize && csa == 1 {
        let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, m * n) };
        dst.copy_from_slice(src_slice);
        return;
    }
    for i in 0..m {
        let row_base = unsafe { src_ptr.offset(i as isize * rsa) };
        let dst_row = &mut dst[i * n..(i + 1) * n];
        if csa == 1 {
            let src_row = unsafe { std::slice::from_raw_parts(row_base, n) };
            dst_row.copy_from_slice(src_row);
        } else {
            for j in 0..n {
                dst_row[j] = unsafe { *row_base.offset(j as isize * csa) };
            }
        }
    }
}

/// Checks whether every entry in a strided `(m, n)` matrix is finite (`!isnan && !isinf`).
///
/// # Safety
/// `src_ptr` must be valid for all offsets in `0..m` x `0..n` using `(rsa, csa)`.
#[inline]
pub unsafe fn mat_all_finite<T: LinalgScalar>(
    src_ptr: *const T,
    m: usize,
    n: usize,
    rsa: isize,
    csa: isize,
) -> bool {
    if m == 0 || n == 0 {
        return true;
    }
    if rsa == n as isize && csa == 1 {
        let s = unsafe { std::slice::from_raw_parts(src_ptr, m * n) };
        return s.iter().all(|x| x.is_finite_elem());
    }
    for i in 0..m {
        let row_ptr = unsafe { src_ptr.offset(i as isize * rsa) };
        for j in 0..n {
            if !unsafe { *row_ptr.offset(j as isize * csa) }.is_finite_elem() {
                return false;
            }
        }
    }
    true
}

/// Checks whether any entry in a strided `(m, n)` matrix is `NaN`.
///
/// # Safety
/// `src_ptr` must be valid for all offsets in `0..m` x `0..n` using `(rsa, csa)`.
#[inline]
pub unsafe fn mat_any_nan<T: LinalgScalar>(
    src_ptr: *const T,
    m: usize,
    n: usize,
    rsa: isize,
    csa: isize,
) -> bool {
    if m == 0 || n == 0 {
        return false;
    }
    if rsa == n as isize && csa == 1 {
        let s = unsafe { std::slice::from_raw_parts(src_ptr, m * n) };
        return s.iter().any(|x| x.is_nan_elem());
    }
    for i in 0..m {
        let row_ptr = unsafe { src_ptr.offset(i as isize * rsa) };
        for j in 0..n {
            if unsafe { *row_ptr.offset(j as isize * csa) }.is_nan_elem() {
                return true;
            }
        }
    }
    false
}

/// Broadcasts `batch_a` against `batch_b`, returning `(out_batch_shape, padded_strides_a, padded_strides_b)`.
pub fn broadcast_batch_dims(
    shape_a: &[usize],
    strides_a: &[isize],
    shape_b: &[usize],
    strides_b: &[isize],
) -> Result<(Vec<usize>, Vec<isize>, Vec<isize>)> {
    let out_shape = broadcast_shapes(shape_a, shape_b)?;
    let nd = out_shape.len();
    let pad_strides = |src_shape: &[usize], src_strides: &[isize]| -> Vec<isize> {
        let mut res = vec![0isize; nd];
        let diff = nd - src_shape.len();
        for i in 0..src_shape.len() {
            res[diff + i] = if src_shape[i] == 1 && out_shape[diff + i] > 1 {
                0
            } else {
                src_strides[i]
            };
        }
        res
    };
    let sa = pad_strides(shape_a, strides_a);
    let sb = pad_strides(shape_b, strides_b);
    Ok((out_shape, sa, sb))
}

/// Fills an `(n, n)` C-contiguous buffer slice with the identity matrix.
#[inline]
pub fn fill_identity_c<T: Scalar>(buf: &mut [T], n: usize) {
    buf.fill(T::zero());
    for i in 0..n {
        buf[i * n + i] = T::one();
    }
}

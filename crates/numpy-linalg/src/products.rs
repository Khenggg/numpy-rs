//! Products and tensor operations: dot, vdot, inner, outer, matmul, matvec, vecmat,
//! vecdot, tensordot, kron, cross, multi_dot, and trace.

use std::collections::HashSet;

use numpy_types::{
    DType, DynArray, NdArray, NumPyError, Result, normalize_axis, result_type, shape_size,
};
use rayon::prelude::*;

use crate::batch::{
    GEMM_RAYON_THRESHOLD, SINGLE_GEMM_RAYON_THRESHOLD, SyncPtr, alloc_zeroed_buf,
    broadcast_batch_dims, linear_strided_offset, ndarray_from_aligned, should_par_batches,
    try_alloc_zeroed_buf,
};
use crate::traits::{ProductScalar, TensordotAxes, trace_default_dtype};
use crate::{dispatch_product_dyn, dispatch_product_pair_dyn};

const VEC_RAYON_THRESHOLD: usize = 65_536;

/// Returns a strided view with `src_axis` moved to `dst_axis` (preserving relative order of others).
pub(crate) fn move_axis<T: ProductScalar>(
    a: &NdArray<T>,
    src_axis: isize,
    dst_axis: isize,
) -> Result<NdArray<T>> {
    let nd = a.ndim();
    let src = normalize_axis(src_axis, nd)?;
    let dst = normalize_axis(dst_axis, nd)?;
    if src == dst {
        return Ok(a.clone());
    }
    let mut axes: Vec<usize> = (0..nd).collect();
    let removed = axes.remove(src);
    axes.insert(dst, removed);
    let perm: Vec<isize> = axes.into_iter().map(|x| x as isize).collect();
    a.transpose(Some(&perm))
}

#[inline]
unsafe fn dot_1d_raw<T: ProductScalar>(
    len: usize,
    a_ptr: *const T,
    sa: isize,
    b_ptr: *const T,
    sb: isize,
    conj_a: bool,
) -> T {
    if sa == 1 && sb == 1 && len >= 8 {
        let mut a0 = T::ZERO;
        let mut a1 = T::ZERO;
        let mut a2 = T::ZERO;
        let mut a3 = T::ZERO;
        let mut i = 0usize;
        let n4 = len & !3;
        if conj_a {
            while i < n4 {
                a0 = a0.add_elem(unsafe { *a_ptr.add(i) }.conj_elem().mul_elem(unsafe { *b_ptr.add(i) }));
                a1 = a1.add_elem(unsafe { *a_ptr.add(i + 1) }.conj_elem().mul_elem(unsafe { *b_ptr.add(i + 1) }));
                a2 = a2.add_elem(unsafe { *a_ptr.add(i + 2) }.conj_elem().mul_elem(unsafe { *b_ptr.add(i + 2) }));
                a3 = a3.add_elem(unsafe { *a_ptr.add(i + 3) }.conj_elem().mul_elem(unsafe { *b_ptr.add(i + 3) }));
                i += 4;
            }
            while i < len {
                a0 = a0.add_elem(unsafe { *a_ptr.add(i) }.conj_elem().mul_elem(unsafe { *b_ptr.add(i) }));
                i += 1;
            }
        } else {
            while i < n4 {
                a0 = a0.add_elem(unsafe { *a_ptr.add(i) }.mul_elem(unsafe { *b_ptr.add(i) }));
                a1 = a1.add_elem(unsafe { *a_ptr.add(i + 1) }.mul_elem(unsafe { *b_ptr.add(i + 1) }));
                a2 = a2.add_elem(unsafe { *a_ptr.add(i + 2) }.mul_elem(unsafe { *b_ptr.add(i + 2) }));
                a3 = a3.add_elem(unsafe { *a_ptr.add(i + 3) }.mul_elem(unsafe { *b_ptr.add(i + 3) }));
                i += 4;
            }
            while i < len {
                a0 = a0.add_elem(unsafe { *a_ptr.add(i) }.mul_elem(unsafe { *b_ptr.add(i) }));
                i += 1;
            }
        }
        return a0.add_elem(a1).add_elem(a2.add_elem(a3));
    }

    let mut acc = T::ZERO;
    if conj_a {
        for i in 0..len {
            let av = unsafe { *a_ptr.offset(i as isize * sa) }.conj_elem();
            let bv = unsafe { *b_ptr.offset(i as isize * sb) };
            acc = acc.add_elem(av.mul_elem(bv));
        }
    } else {
        for i in 0..len {
            let av = unsafe { *a_ptr.offset(i as isize * sa) };
            let bv = unsafe { *b_ptr.offset(i as isize * sb) };
            acc = acc.add_elem(av.mul_elem(bv));
        }
    }
    acc
}

/// Core 2-D matrix multiply `(m, k) @ (k, n) -> (m, n)` writing into C-contiguous `c_buf`.
fn gemm_2d_to_c<T: ProductScalar>(
    m: usize,
    k: usize,
    n: usize,
    a_ptr: SyncPtr<T>,
    rsa: isize,
    csa: isize,
    b_ptr: SyncPtr<T>,
    rsb: isize,
    csb: isize,
    c_buf: &mut [T],
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c_buf.fill(T::ZERO);
        return;
    }
    let work = m.saturating_mul(k).saturating_mul(n);
    if work >= SINGLE_GEMM_RAYON_THRESHOLD && m >= 8 {
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_rows = m.div_ceil(nthreads).max(1);
        c_buf
            .par_chunks_mut(chunk_rows * n)
            .enumerate()
            .for_each(|(chunk_idx, c_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = c_chunk.len() / n;
                unsafe {
                    T::strided_gemm(
                        rows,
                        k,
                        n,
                        a_ptr.offset(row_start as isize * rsa),
                        rsa,
                        csa,
                        b_ptr.offset(0),
                        rsb,
                        csb,
                        c_chunk.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
            });
    } else if work >= SINGLE_GEMM_RAYON_THRESHOLD && m == 1 && n >= 64 {
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_cols = n.div_ceil(nthreads).max(1);
        c_buf
            .par_chunks_mut(chunk_cols)
            .enumerate()
            .for_each(|(chunk_idx, c_chunk)| {
                let col_start = chunk_idx * chunk_cols;
                let cols = c_chunk.len();
                unsafe {
                    T::strided_gemm(
                        1,
                        k,
                        cols,
                        a_ptr.offset(0),
                        rsa,
                        csa,
                        b_ptr.offset(col_start as isize * csb),
                        rsb,
                        csb,
                        c_chunk.as_mut_ptr(),
                        cols as isize,
                        1,
                    );
                }
            });
    } else {
        unsafe {
            T::strided_gemm(
                m,
                k,
                n,
                a_ptr.0,
                rsa,
                csa,
                b_ptr.0,
                rsb,
                csb,
                c_buf.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

fn scalar_mul_array<T: ProductScalar>(scalar: T, arr: &NdArray<T>) -> Result<NdArray<T>> {
    let c = arr.to_c_contiguous();
    let src = c.as_slice().unwrap_or(&[]);
    let mut out = alloc_zeroed_buf::<T>(src.len());
    let dst = out.as_mut_slice();
    if src.len() >= VEC_RAYON_THRESHOLD {
        dst.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(d, &s)| *d = scalar.mul_elem(s));
    } else {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = scalar.mul_elem(s);
        }
    }
    ndarray_from_aligned(out, c.shape())
}

fn array_mul_scalar<T: ProductScalar>(arr: &NdArray<T>, scalar: T) -> Result<NdArray<T>> {
    let c = arr.to_c_contiguous();
    let src = c.as_slice().unwrap_or(&[]);
    let mut out = alloc_zeroed_buf::<T>(src.len());
    let dst = out.as_mut_slice();
    if src.len() >= VEC_RAYON_THRESHOLD {
        dst.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(d, &s)| *d = s.mul_elem(scalar));
    } else {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s.mul_elem(scalar);
        }
    }
    ndarray_from_aligned(out, c.shape())
}

/// Matrix / dot product matching `numpy.dot`.
pub fn dot<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() == 0 {
        let s = unsafe { *a.storage_ptr().add(a.offset()) };
        return scalar_mul_array(s, b);
    }
    if b.ndim() == 0 {
        let s = unsafe { *b.storage_ptr().add(b.offset()) };
        return array_mul_scalar(a, s);
    }
    if a.ndim() == 1 && b.ndim() == 1 {
        let k = a.len();
        if k != b.len() {
            return Err(NumPyError::ShapeMismatch(format!(
                "shapes ({},) and ({},) not aligned: {} (dim 0) != {} (dim 0)",
                k,
                b.len(),
                k,
                b.len()
            )));
        }
        let val = unsafe {
            dot_1d_raw(
                k,
                a.storage_ptr().add(a.offset()),
                a.strides()[0],
                b.storage_ptr().add(b.offset()),
                b.strides()[0],
                false,
            )
        };
        return Ok(NdArray::from_scalar(val));
    }
    if a.ndim() == 2 && b.ndim() == 2 {
        return matmul(a, b);
    }
    let k = a.shape()[a.ndim() - 1];
    if b.ndim() == 1 {
        if k != b.len() {
            return Err(NumPyError::ShapeMismatch(format!(
                "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim 0)",
                a.shape(),
                b.shape(),
                k,
                a.ndim() - 1,
                b.len()
            )));
        }
        let a_c = a.to_c_contiguous();
        let m_a: usize = a_c.shape()[..a_c.ndim() - 1].iter().product();
        let out_shape = &a_c.shape()[..a_c.ndim() - 1];
        let mut out = alloc_zeroed_buf::<T>(m_a);
        let a_ptr = SyncPtr(unsafe { a_c.storage_ptr().add(a_c.offset()) });
        let b_ptr = SyncPtr(unsafe { b.storage_ptr().add(b.offset()) });
        gemm_2d_to_c(
            m_a,
            k,
            1,
            a_ptr,
            k as isize,
            1,
            b_ptr,
            b.strides()[0],
            1,
            out.as_mut_slice(),
        );
        return ndarray_from_aligned(out, out_shape);
    }

    // General N-D x M-D (M >= 2): contract last axis of `a` with second-to-last axis of `b`.
    let b_k = b.shape()[b.ndim() - 2];
    if k != b_k {
        return Err(NumPyError::ShapeMismatch(format!(
            "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim {})",
            a.shape(),
            b.shape(),
            k,
            a.ndim() - 1,
            b_k,
            b.ndim() - 2
        )));
    }
    let mut out_shape = Vec::with_capacity(a.ndim() - 1 + b.ndim() - 1);
    out_shape.extend_from_slice(&a.shape()[..a.ndim() - 1]);
    out_shape.extend_from_slice(&b.shape()[..b.ndim() - 2]);
    out_shape.push(b.shape()[b.ndim() - 1]);

    let a_c = a.to_c_contiguous();
    let b_moved = move_axis(b, (b.ndim() - 2) as isize, 0)?.to_c_contiguous();
    let m_a: usize = a_c.shape()[..a_c.ndim() - 1].iter().product();
    let n_b: usize = b_moved.shape()[1..].iter().product();
    let out_total = shape_size(&out_shape)?;
    let mut out = try_alloc_zeroed_buf::<T>(out_total)?;
    let a_ptr = SyncPtr(unsafe { a_c.storage_ptr().add(a_c.offset()) });
    let b_ptr = SyncPtr(unsafe { b_moved.storage_ptr().add(b_moved.offset()) });
    gemm_2d_to_c(
        m_a,
        k,
        n_b,
        a_ptr,
        k as isize,
        1,
        b_ptr,
        n_b as isize,
        1,
        out.as_mut_slice(),
    );
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `dot` matching NumPy dtype promotion.
pub fn dot_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(dot(va, vb)?)))
}

#[inline]
fn to_dyn<T: ProductScalar>(arr: NdArray<T>) -> DynArray {
    T::product_into_dyn(arr)
}

/// Vector dot product flattening both operands and conjugating `a`, matching `numpy.vdot`.
pub fn vdot<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.len() != b.len() {
        return Err(NumPyError::ShapeMismatch(format!(
            "cannot compute vdot of arrays with sizes {} and {}",
            a.len(),
            b.len()
        )));
    }
    let len = a.len();
    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        if len >= VEC_RAYON_THRESHOLD {
            let chunk_size = 16_384;
            let partial: Vec<T> = sa
                .par_chunks(chunk_size)
                .zip(sb.par_chunks(chunk_size))
                .map(|(ca, cb)| unsafe {
                    dot_1d_raw(ca.len(), ca.as_ptr(), 1, cb.as_ptr(), 1, true)
                })
                .collect();
            let total = partial.into_iter().fold(T::ZERO, |acc, x| acc.add_elem(x));
            return Ok(NdArray::from_scalar(total));
        }
        let val = unsafe { dot_1d_raw(len, sa.as_ptr(), 1, sb.as_ptr(), 1, true) };
        return Ok(NdArray::from_scalar(val));
    }
    let a_c = a.to_c_contiguous();
    let b_c = b.to_c_contiguous();
    let sa = a_c.as_slice().unwrap_or(&[]);
    let sb = b_c.as_slice().unwrap_or(&[]);
    let val = unsafe { dot_1d_raw(len, sa.as_ptr(), 1, sb.as_ptr(), 1, true) };
    Ok(NdArray::from_scalar(val))
}

/// Dynamic `vdot`.
pub fn vdot_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(vdot(va, vb)?)))
}

/// Inner product over the last axis of both arrays (non-conjugated), matching `numpy.inner`.
pub fn inner<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() == 0 {
        let s = unsafe { *a.storage_ptr().add(a.offset()) };
        return scalar_mul_array(s, b);
    }
    if b.ndim() == 0 {
        let s = unsafe { *b.storage_ptr().add(b.offset()) };
        return array_mul_scalar(a, s);
    }
    let k = a.shape()[a.ndim() - 1];
    let k_b = b.shape()[b.ndim() - 1];
    if k != k_b {
        return Err(NumPyError::ShapeMismatch(format!(
            "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim {})",
            a.shape(),
            b.shape(),
            k,
            a.ndim() - 1,
            k_b,
            b.ndim() - 1
        )));
    }
    if a.ndim() == 1 && b.ndim() == 1 {
        let val = unsafe {
            dot_1d_raw(
                k,
                a.storage_ptr().add(a.offset()),
                a.strides()[0],
                b.storage_ptr().add(b.offset()),
                b.strides()[0],
                false,
            )
        };
        return Ok(NdArray::from_scalar(val));
    }
    let mut out_shape = Vec::with_capacity(a.ndim() + b.ndim() - 2);
    out_shape.extend_from_slice(&a.shape()[..a.ndim() - 1]);
    out_shape.extend_from_slice(&b.shape()[..b.ndim() - 1]);
    let m_a: usize = a.shape()[..a.ndim() - 1].iter().product();
    let n_b: usize = b.shape()[..b.ndim() - 1].iter().product();
    let a_c = a.to_c_contiguous();
    let b_c = b.to_c_contiguous();
    let out_total = shape_size(&out_shape)?;
    let mut out = try_alloc_zeroed_buf::<T>(out_total)?;
    let a_ptr = SyncPtr(unsafe { a_c.storage_ptr().add(a_c.offset()) });
    let b_ptr = SyncPtr(unsafe { b_c.storage_ptr().add(b_c.offset()) });
    // C = A * B^T: B is (N_b, K) C-contiguous so B^T has row stride 1 and col stride K.
    gemm_2d_to_c(
        m_a,
        k,
        n_b,
        a_ptr,
        k as isize,
        1,
        b_ptr,
        1,
        k as isize,
        out.as_mut_slice(),
    );
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `inner`.
pub fn inner_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(inner(va, vb)?)))
}

/// Outer product flattening both arrays (matching `numpy.outer`).
pub fn outer<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let m = a.len();
    let n = b.len();
    let a_c = a.to_c_contiguous();
    let b_c = b.to_c_contiguous();
    let sa = a_c.as_slice().unwrap_or(&[]);
    let sb = b_c.as_slice().unwrap_or(&[]);
    let total = shape_size(&[m, n])?;
    let mut out = try_alloc_zeroed_buf::<T>(total)?;
    let dst = out.as_mut_slice();
    if m > 0 && n > 0 {
        if m.saturating_mul(n) >= VEC_RAYON_THRESHOLD {
            dst.par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    let av = sa[i];
                    for (d, &bv) in row.iter_mut().zip(sb.iter()) {
                        *d = av.mul_elem(bv);
                    }
                });
        } else {
            for (i, row) in dst.chunks_exact_mut(n).enumerate() {
                let av = sa[i];
                for (d, &bv) in row.iter_mut().zip(sb.iter()) {
                    *d = av.mul_elem(bv);
                }
            }
        }
    }
    ndarray_from_aligned(out, &[m, n])
}

/// Strict 1-D Array API `numpy.linalg.outer`.
pub fn linalg_outer<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumPyError::ShapeMismatch(format!(
            "Only one-dimensional arrays allowed, got x1.ndim={} and x2.ndim={}",
            a.ndim(),
            b.ndim()
        )));
    }
    outer(a, b)
}

/// Dynamic `outer` (`numpy.outer`).
pub fn outer_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(outer(va, vb)?)))
}

/// Strict 1-D dynamic `numpy.linalg.outer`.
pub fn linalg_outer_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumPyError::ShapeMismatch(format!(
            "Only one-dimensional arrays allowed, got x1.ndim={} and x2.ndim={}",
            a.ndim(),
            b.ndim()
        )));
    }
    outer_dyn(a, b)
}

/// Matrix product supporting 1-D vector promotion and N-D stacked broadcasting, matching `numpy.matmul`.
pub fn matmul<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(NumPyError::ShapeMismatch(
            "matmul: Input operand does not have enough dimensions (scalars not allowed)".into(),
        ));
    }
    let prepend_a = a.ndim() == 1;
    let append_b = b.ndim() == 1;

    let (m, ka, rsa, csa, batch_a, bstr_a) = if prepend_a {
        (
            1usize,
            a.len(),
            0isize,
            a.strides()[0],
            &[][..],
            &[][..],
        )
    } else {
        let nd = a.ndim();
        (
            a.shape()[nd - 2],
            a.shape()[nd - 1],
            a.strides()[nd - 2],
            a.strides()[nd - 1],
            &a.shape()[..nd - 2],
            &a.strides()[..nd - 2],
        )
    };
    let (kb, n, rsb, csb, batch_b, bstr_b) = if append_b {
        (
            b.len(),
            1usize,
            b.strides()[0],
            0isize,
            &[][..],
            &[][..],
        )
    } else {
        let nd = b.ndim();
        (
            b.shape()[nd - 2],
            b.shape()[nd - 1],
            b.strides()[nd - 2],
            b.strides()[nd - 1],
            &b.shape()[..nd - 2],
            &b.strides()[..nd - 2],
        )
    };

    if ka != kb {
        return Err(NumPyError::ShapeMismatch(format!(
            "matmul: Input operand 1 has a mismatch in its core dimension 0 (size {} is different from {})",
            kb, ka
        )));
    }
    let k = ka;
    let (out_batch, pad_sa, pad_sb) = broadcast_batch_dims(batch_a, bstr_a, batch_b, bstr_b)?;
    let mut out_shape = out_batch.clone();
    if !prepend_a {
        out_shape.push(m);
    }
    if !append_b {
        out_shape.push(n);
    }

    let bcount = shape_size(&out_batch)?;
    let out_total = shape_size(&out_shape)?;
    let mat_len = m.checked_mul(n).unwrap_or(usize::MAX);
    let mut out = try_alloc_zeroed_buf::<T>(out_total)?;
    if bcount == 0 || m == 0 || n == 0 || k == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    let a_base = SyncPtr(unsafe { a.storage_ptr().add(a.offset()) });
    let b_base = SyncPtr(unsafe { b.storage_ptr().add(b.offset()) });

    if bcount == 1 {
        gemm_2d_to_c(
            m,
            k,
            n,
            a_base,
            rsa,
            csa,
            b_base,
            rsb,
            csb,
            out.as_mut_slice(),
        );
    } else {
        let work_per_item = m.saturating_mul(k).saturating_mul(n);
        let run_one = |batch_idx: usize, c_slice: &mut [T]| {
            let oa = linear_strided_offset(batch_idx, &out_batch, &pad_sa);
            let ob = linear_strided_offset(batch_idx, &out_batch, &pad_sb);
            unsafe {
                T::strided_gemm(
                    m,
                    k,
                    n,
                    a_base.offset(oa),
                    rsa,
                    csa,
                    b_base.offset(ob),
                    rsb,
                    csb,
                    c_slice.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        };
        if should_par_batches(bcount, work_per_item, GEMM_RAYON_THRESHOLD) {
            out.as_mut_slice()
                .par_chunks_mut(mat_len)
                .enumerate()
                .for_each(|(i, c_slice)| run_one(i, c_slice));
        } else {
            for (i, c_slice) in out.as_mut_slice().chunks_exact_mut(mat_len).enumerate() {
                run_one(i, c_slice);
            }
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `matmul`.
pub fn matmul_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(matmul(va, vb)?)))
}

/// Matrix-vector product `(..., M, K), (..., K) -> (..., M)` (no conjugation), matching `numpy.matvec`.
pub fn matvec<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() < 2 {
        return Err(NumPyError::ShapeMismatch(format!(
            "matvec: Input operand 0 does not have enough dimensions (has {}, requires 2)",
            a.ndim()
        )));
    }
    if b.ndim() < 1 {
        return Err(NumPyError::ShapeMismatch(
            "matvec: Input operand 1 does not have enough dimensions (has 0, requires 1)".into(),
        ));
    }
    let nda = a.ndim();
    let ndb = b.ndim();
    let m = a.shape()[nda - 2];
    let k = a.shape()[nda - 1];
    let kb = b.shape()[ndb - 1];
    if k != kb {
        return Err(NumPyError::ShapeMismatch(format!(
            "matvec: Input operand 1 core dimension mismatch ({kb} != {k})"
        )));
    }
    let rsa = a.strides()[nda - 2];
    let csa = a.strides()[nda - 1];
    let rsb = b.strides()[ndb - 1];

    let (out_batch, pad_sa, pad_sb) = broadcast_batch_dims(
        &a.shape()[..nda - 2],
        &a.strides()[..nda - 2],
        &b.shape()[..ndb - 1],
        &b.strides()[..ndb - 1],
    )?;
    let mut out_shape = out_batch.clone();
    out_shape.push(m);

    let bcount = shape_size(&out_batch)?;
    let mut out = alloc_zeroed_buf::<T>(bcount.saturating_mul(m));
    if bcount == 0 || m == 0 || k == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    let a_base = SyncPtr(unsafe { a.storage_ptr().add(a.offset()) });
    let b_base = SyncPtr(unsafe { b.storage_ptr().add(b.offset()) });
    if bcount == 1 {
        gemm_2d_to_c(m, k, 1, a_base, rsa, csa, b_base, rsb, 0, out.as_mut_slice());
    } else {
        let run_one = |batch_idx: usize, c_slice: &mut [T]| {
            let oa = linear_strided_offset(batch_idx, &out_batch, &pad_sa);
            let ob = linear_strided_offset(batch_idx, &out_batch, &pad_sb);
            unsafe {
                T::strided_gemm(
                    m,
                    k,
                    1,
                    a_base.offset(oa),
                    rsa,
                    csa,
                    b_base.offset(ob),
                    rsb,
                    0,
                    c_slice.as_mut_ptr(),
                    1,
                    1,
                );
            }
        };
        if should_par_batches(bcount, m.saturating_mul(k), GEMM_RAYON_THRESHOLD) {
            out.as_mut_slice()
                .par_chunks_mut(m)
                .enumerate()
                .for_each(|(i, c)| run_one(i, c));
        } else {
            for (i, c) in out.as_mut_slice().chunks_exact_mut(m).enumerate() {
                run_one(i, c);
            }
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `matvec`.
pub fn matvec_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(matvec(va, vb)?)))
}

/// Vector-matrix product `(..., K), (..., K, N) -> (..., N)`, conjugating `a` for complex types
/// matching `numpy.vecmat` (`sum_i conj(a_i) * B_{i,j}`).
pub fn vecmat<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() < 1 {
        return Err(NumPyError::ShapeMismatch(
            "vecmat: Input operand 0 does not have enough dimensions (has 0, requires 1)".into(),
        ));
    }
    if b.ndim() < 2 {
        return Err(NumPyError::ShapeMismatch(format!(
            "vecmat: Input operand 1 does not have enough dimensions (has {}, requires 2)",
            b.ndim()
        )));
    }
    let nda = a.ndim();
    let ndb = b.ndim();
    let k = a.shape()[nda - 1];
    let kb = b.shape()[ndb - 2];
    let n = b.shape()[ndb - 1];
    if k != kb {
        return Err(NumPyError::ShapeMismatch(format!(
            "vecmat: Input operand 1 core dimension mismatch ({kb} != {k})"
        )));
    }
    let csa = a.strides()[nda - 1];
    let rsb = b.strides()[ndb - 2];
    let csb = b.strides()[ndb - 1];

    let (out_batch, pad_sa, pad_sb) = broadcast_batch_dims(
        &a.shape()[..nda - 1],
        &a.strides()[..nda - 1],
        &b.shape()[..ndb - 2],
        &b.strides()[..ndb - 2],
    )?;
    let mut out_shape = out_batch.clone();
    out_shape.push(n);

    let bcount = shape_size(&out_batch)?;
    let mut out = alloc_zeroed_buf::<T>(bcount.saturating_mul(n));
    if bcount == 0 || k == 0 || n == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    let a_base = SyncPtr(unsafe { a.storage_ptr().add(a.offset()) });
    let b_base = SyncPtr(unsafe { b.storage_ptr().add(b.offset()) });
    if bcount == 1 {
        unsafe {
            T::strided_gemm_conj_lhs(
                1,
                k,
                n,
                a_base.0,
                0,
                csa,
                b_base.0,
                rsb,
                csb,
                out.as_mut_slice().as_mut_ptr(),
                n as isize,
                1,
            );
        }
    } else {
        let run_one = |batch_idx: usize, c_slice: &mut [T]| {
            let oa = linear_strided_offset(batch_idx, &out_batch, &pad_sa);
            let ob = linear_strided_offset(batch_idx, &out_batch, &pad_sb);
            unsafe {
                T::strided_gemm_conj_lhs(
                    1,
                    k,
                    n,
                    a_base.offset(oa),
                    0,
                    csa,
                    b_base.offset(ob),
                    rsb,
                    csb,
                    c_slice.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        };
        if should_par_batches(bcount, k.saturating_mul(n), GEMM_RAYON_THRESHOLD) {
            out.as_mut_slice()
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, c)| run_one(i, c));
        } else {
            for (i, c) in out.as_mut_slice().chunks_exact_mut(n).enumerate() {
                run_one(i, c);
            }
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `vecmat`.
pub fn vecmat_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(vecmat(va, vb)?)))
}

/// Complex-conjugating vector dot product over `axis`, matching `numpy.linalg.vecdot`.
pub fn vecdot<T: ProductScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    axis: isize,
) -> Result<NdArray<T>> {
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(NumPyError::ShapeMismatch(
            "vecdot: Input operand does not have enough dimensions (scalars not allowed)".into(),
        ));
    }
    let a_moved = move_axis(a, axis, -1)?;
    let b_moved = move_axis(b, axis, -1)?;
    let nda = a_moved.ndim();
    let ndb = b_moved.ndim();
    let k = a_moved.shape()[nda - 1];
    let kb = b_moved.shape()[ndb - 1];
    if k != kb {
        return Err(NumPyError::ShapeMismatch(format!(
            "vecdot: Core dimension mismatch ({kb} != {k})"
        )));
    }
    let sa_k = a_moved.strides()[nda - 1];
    let sb_k = b_moved.strides()[ndb - 1];
    let (out_shape, pad_sa, pad_sb) = broadcast_batch_dims(
        &a_moved.shape()[..nda - 1],
        &a_moved.strides()[..nda - 1],
        &b_moved.shape()[..ndb - 1],
        &b_moved.strides()[..ndb - 1],
    )?;
    let bcount = shape_size(&out_shape)?;
    let mut out = alloc_zeroed_buf::<T>(bcount);
    if bcount == 0 || k == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    let a_base = SyncPtr(unsafe { a_moved.storage_ptr().add(a_moved.offset()) });
    let b_base = SyncPtr(unsafe { b_moved.storage_ptr().add(b_moved.offset()) });
    if bcount == 1 {
        let v = unsafe { dot_1d_raw(k, a_base.0, sa_k, b_base.0, sb_k, true) };
        out.as_mut_slice()[0] = v;
    } else if should_par_batches(bcount, k, GEMM_RAYON_THRESHOLD) {
        out.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dst)| {
                let oa = linear_strided_offset(i, &out_shape, &pad_sa);
                let ob = linear_strided_offset(i, &out_shape, &pad_sb);
                *dst = unsafe {
                    dot_1d_raw(k, a_base.offset(oa), sa_k, b_base.offset(ob), sb_k, true)
                };
            });
    } else {
        for (i, dst) in out.as_mut_slice().iter_mut().enumerate() {
            let oa = linear_strided_offset(i, &out_shape, &pad_sa);
            let ob = linear_strided_offset(i, &out_shape, &pad_sb);
            *dst = unsafe {
                dot_1d_raw(k, a_base.offset(oa), sa_k, b_base.offset(ob), sb_k, true)
            };
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `vecdot`.
pub fn vecdot_dyn(a: &DynArray, b: &DynArray, axis: isize) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(vecdot(va, vb, axis)?)))
}

/// Tensor contraction matching `numpy.tensordot` / `numpy.linalg.tensordot`.
pub fn tensordot<T: ProductScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    axes: impl Into<TensordotAxes>,
) -> Result<NdArray<T>> {
    let nda = a.ndim();
    let ndb = b.ndim();
    let (axes_a, axes_b) = match axes.into() {
        TensordotAxes::Num(n) => {
            if n > nda || n > ndb {
                return Err(NumPyError::ShapeMismatch(format!(
                    "tensordot: axes={n} exceeds array dimensions ({nda}, {ndb})"
                )));
            }
            ((nda - n..nda).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>())
        }
        TensordotAxes::Axes(raw_a, raw_b) => {
            if raw_a.len() != raw_b.len() {
                return Err(NumPyError::ShapeMismatch(
                    "tensordot: axis lists must have equal length".into(),
                ));
            }
            let mut va = Vec::with_capacity(raw_a.len());
            let mut vb = Vec::with_capacity(raw_b.len());
            let mut seen_a = HashSet::with_capacity(raw_a.len());
            let mut seen_b = HashSet::with_capacity(raw_b.len());
            for (&ax_a, &ax_b) in raw_a.iter().zip(raw_b.iter()) {
                let na = normalize_axis(ax_a, nda)?;
                let nb = normalize_axis(ax_b, ndb)?;
                if !seen_a.insert(na) {
                    return Err(NumPyError::InvalidValue(format!(
                        "repeated axis {na} in tensordot first operand"
                    )));
                }
                if !seen_b.insert(nb) {
                    return Err(NumPyError::InvalidValue(format!(
                        "repeated axis {nb} in tensordot second operand"
                    )));
                }
                va.push(na);
                vb.push(nb);
            }
            (va, vb)
        }
    };

    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        if a.shape()[ax_a] != b.shape()[ax_b] {
            return Err(NumPyError::ShapeMismatch(format!(
                "shape-mismatch for sum: dim {} ({}) != dim {} ({})",
                ax_a,
                a.shape()[ax_a],
                ax_b,
                b.shape()[ax_b]
            )));
        }
    }

    let notin_a: Vec<usize> = (0..nda).filter(|k| !axes_a.contains(k)).collect();
    let notin_b: Vec<usize> = (0..ndb).filter(|k| !axes_b.contains(k)).collect();

    let mut out_shape = Vec::with_capacity(notin_a.len() + notin_b.len());
    for &ax in &notin_a {
        out_shape.push(a.shape()[ax]);
    }
    for &ax in &notin_b {
        out_shape.push(b.shape()[ax]);
    }

    let k_size: usize = axes_a.iter().map(|&ax| a.shape()[ax]).product();
    let m_size: usize = notin_a.iter().map(|&ax| a.shape()[ax]).product();
    let n_size: usize = notin_b.iter().map(|&ax| b.shape()[ax]).product();

    let mut perm_a: Vec<isize> = notin_a.iter().map(|&x| x as isize).collect();
    perm_a.extend(axes_a.iter().map(|&x| x as isize));
    let mut perm_b: Vec<isize> = axes_b.iter().map(|&x| x as isize).collect();
    perm_b.extend(notin_b.iter().map(|&x| x as isize));

    let at = a.transpose(Some(&perm_a))?.to_c_contiguous();
    let bt = b.transpose(Some(&perm_b))?.to_c_contiguous();

    let out_total = shape_size(&[m_size, n_size])?;
    let mut out = try_alloc_zeroed_buf::<T>(out_total)?;
    let a_ptr = SyncPtr(unsafe { at.storage_ptr().add(at.offset()) });
    let b_ptr = SyncPtr(unsafe { bt.storage_ptr().add(bt.offset()) });
    gemm_2d_to_c(
        m_size,
        k_size,
        n_size,
        a_ptr,
        k_size as isize,
        1,
        b_ptr,
        n_size as isize,
        1,
        out.as_mut_slice(),
    );
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `tensordot`.
pub fn tensordot_dyn(
    a: &DynArray,
    b: &DynArray,
    axes: impl Into<TensordotAxes>,
) -> Result<DynArray> {
    let axes = axes.into();
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(tensordot(va, vb, axes.clone())?)))
}

/// Kronecker product matching `numpy.kron`.
pub fn kron<T: ProductScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    if a.ndim() == 0 && b.ndim() == 0 {
        let va = unsafe { *a.storage_ptr().add(a.offset()) };
        let vb = unsafe { *b.storage_ptr().add(b.offset()) };
        return Ok(NdArray::from_scalar(va.mul_elem(vb)));
    }
    let nd = a.ndim().max(b.ndim()).max(1);
    let mut a_shape = vec![1usize; nd];
    let mut b_shape = vec![1usize; nd];
    let mut a_strides = vec![0isize; nd];
    let mut b_strides = vec![0isize; nd];

    let off_a = nd - a.ndim();
    for i in 0..a.ndim() {
        a_shape[off_a + i] = a.shape()[i];
        a_strides[off_a + i] = a.strides()[i];
    }
    let off_b = nd - b.ndim();
    for i in 0..b.ndim() {
        b_shape[off_b + i] = b.shape()[i];
        b_strides[off_b + i] = b.strides()[i];
    }

    let out_shape: Vec<usize> = a_shape
        .iter()
        .zip(b_shape.iter())
        .map(|(&sa, &sb)| {
            sa.checked_mul(sb)
                .ok_or_else(|| NumPyError::InvalidValue("kron output shape overflow".into()))
        })
        .collect::<Result<Vec<usize>>>()?;
    let total = shape_size(&out_shape)?;
    let mut out = try_alloc_zeroed_buf::<T>(total)?;
    if total == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    let last_a = a_shape[nd - 1];
    let last_b = b_shape[nd - 1];
    let last_out = out_shape[nd - 1];
    let row_count = total / last_out;

    let sa_last = a_strides[nd - 1];
    let sb_last = b_strides[nd - 1];
    let a_outer_shape = &a_shape[..nd - 1];
    let b_outer_shape = &b_shape[..nd - 1];
    let a_outer_strides = &a_strides[..nd - 1];
    let b_outer_strides = &b_strides[..nd - 1];
    let a_base = SyncPtr(unsafe { a.storage_ptr().add(a.offset()) });
    let b_base = SyncPtr(unsafe { b.storage_ptr().add(b.offset()) });

    let fill_row = |row_idx: usize, dst_row: &mut [T]| {
        let mut r = row_idx;
        let mut oa = 0isize;
        let mut ob = 0isize;
        for d in (0..nd - 1).rev() {
            let dim_b = b_outer_shape[d];
            let dim_out = dim_b * a_outer_shape[d];
            let coord = r % dim_out;
            r /= dim_out;
            let ca = coord / dim_b;
            let cb = coord % dim_b;
            oa += ca as isize * a_outer_strides[d];
            ob += cb as isize * b_outer_strides[d];
        }
        let a_row_ptr = unsafe { a_base.offset(oa) };
        let b_row_ptr = unsafe { b_base.offset(ob) };
        for ia in 0..last_a {
            let av = unsafe { *a_row_ptr.offset(ia as isize * sa_last) };
            let slice = &mut dst_row[ia * last_b..(ia + 1) * last_b];
            for jb in 0..last_b {
                let bv = unsafe { *b_row_ptr.offset(jb as isize * sb_last) };
                slice[jb] = av.mul_elem(bv);
            }
        }
    };

    if total >= VEC_RAYON_THRESHOLD && row_count > 1 {
        out.as_mut_slice()
            .par_chunks_mut(last_out)
            .enumerate()
            .for_each(|(row_idx, dst_row)| fill_row(row_idx, dst_row));
    } else {
        for (row_idx, dst_row) in out.as_mut_slice().chunks_exact_mut(last_out).enumerate() {
            fill_row(row_idx, dst_row);
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `kron`.
pub fn kron_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(kron(va, vb)?)))
}

/// Cross product of 3-vectors along `axis` (default `-1`), matching `numpy.linalg.cross`.
pub fn cross<T: ProductScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    axis: isize,
) -> Result<NdArray<T>> {
    if T::DTYPE == DType::Bool {
        return Err(NumPyError::DTypeError(
            "numpy boolean subtract, the '-' operator, is not supported in cross".into(),
        ));
    }
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(NumPyError::ShapeMismatch(
            "Both input arrays must be at least 1-dimensional with vector size 3".into(),
        ));
    }
    let a_moved = move_axis(a, axis, -1)?;
    let b_moved = move_axis(b, axis, -1)?;
    let nda = a_moved.ndim();
    let ndb = b_moved.ndim();
    let da = a_moved.shape()[nda - 1];
    let db = b_moved.shape()[ndb - 1];
    if da != 3 || db != 3 {
        return Err(NumPyError::ShapeMismatch(format!(
            "Both input arrays must be (arrays of) 3-dimensional vectors along axis {axis}, got sizes {da} and {db}"
        )));
    }
    let sa3 = a_moved.strides()[nda - 1];
    let sb3 = b_moved.strides()[ndb - 1];
    let (out_batch, pad_sa, pad_sb) = broadcast_batch_dims(
        &a_moved.shape()[..nda - 1],
        &a_moved.strides()[..nda - 1],
        &b_moved.shape()[..ndb - 1],
        &b_moved.strides()[..ndb - 1],
    )?;

    let mut tmp_shape = out_batch.clone();
    tmp_shape.push(3);
    let bcount = shape_size(&out_batch)?;
    let total = shape_size(&tmp_shape)?;
    let mut out = try_alloc_zeroed_buf::<T>(total)?;
    if bcount > 0 {
        let a_base = SyncPtr(unsafe { a_moved.storage_ptr().add(a_moved.offset()) });
        let b_base = SyncPtr(unsafe { b_moved.storage_ptr().add(b_moved.offset()) });
        let compute_vec = |idx: usize, dst: &mut [T]| {
            let oa = linear_strided_offset(idx, &out_batch, &pad_sa);
            let ob = linear_strided_offset(idx, &out_batch, &pad_sb);
            let pa = unsafe { a_base.offset(oa) };
            let pb = unsafe { b_base.offset(ob) };
            let a0 = unsafe { *pa };
            let a1 = unsafe { *pa.offset(sa3) };
            let a2 = unsafe { *pa.offset(2 * sa3) };
            let b0 = unsafe { *pb };
            let b1 = unsafe { *pb.offset(sb3) };
            let b2 = unsafe { *pb.offset(2 * sb3) };
            dst[0] = a1.mul_elem(b2).sub_elem(a2.mul_elem(b1));
            dst[1] = a2.mul_elem(b0).sub_elem(a0.mul_elem(b2));
            dst[2] = a0.mul_elem(b1).sub_elem(a1.mul_elem(b0));
        };
        if bcount * 3 >= VEC_RAYON_THRESHOLD {
            out.as_mut_slice()
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i, dst)| compute_vec(i, dst));
        } else {
            for (i, dst) in out.as_mut_slice().chunks_exact_mut(3).enumerate() {
                compute_vec(i, dst);
            }
        }
    }
    let res = ndarray_from_aligned(out, &tmp_shape)?;
    let out_nd = res.ndim();
    let target_ax = normalize_axis(axis, out_nd)? as isize;
    if target_ax as usize == out_nd - 1 {
        Ok(res)
    } else {
        Ok(move_axis(&res, -1, target_ax)?.to_c_contiguous())
    }
}

/// Dynamic `cross`.
pub fn cross_dyn(a: &DynArray, b: &DynArray, axis: isize) -> Result<DynArray> {
    dispatch_product_pair_dyn!(a, b, (va, vb) => Ok(to_dyn(cross(va, vb, axis)?)))
}

/// Optimal chain matrix multiplication using dynamic programming (`O(n^3)`), matching `numpy.linalg.multi_dot`.
pub fn multi_dot<T: ProductScalar>(arrays: &[&NdArray<T>]) -> Result<NdArray<T>> {
    let n = arrays.len();
    if n < 2 {
        return Err(NumPyError::InvalidValue(
            "multi_dot expects at least 2 arrays".into(),
        ));
    }

    let squeeze_first = arrays[0].ndim() == 1;
    let squeeze_last = arrays[n - 1].ndim() == 1;
    let mut mats: Vec<NdArray<T>> = Vec::with_capacity(n);

    for (idx, &arr) in arrays.iter().enumerate() {
        let nd = arr.ndim();
        if nd == 1 && (idx == 0 || idx == n - 1) {
            let new_shape = if idx == 0 {
                [1isize, arr.len() as isize]
            } else {
                [arr.len() as isize, 1isize]
            };
            mats.push(arr.reshape(&new_shape)?);
        } else if nd == 2 {
            mats.push(arr.clone());
        } else {
            return Err(NumPyError::LinAlgError(format!(
                "Expecting 2-D arrays (except optional 1-D first/last), got {nd}-D at index {idx}"
            )));
        }
    }

    for i in 0..n - 1 {
        if mats[i].shape()[1] != mats[i + 1].shape()[0] {
            return Err(NumPyError::ShapeMismatch(format!(
                "shapes {:?} and {:?} not aligned at chain index {i}",
                mats[i].shape(),
                mats[i + 1].shape()
            )));
        }
    }

    let prod = if n == 2 {
        matmul(&mats[0], &mats[1])?
    } else if n == 3 {
        let (a0, a1, a2) = (&mats[0], &mats[1], &mats[2]);
        let m0 = a0.shape()[0] as u128;
        let m1 = a1.shape()[0] as u128;
        let m2 = a2.shape()[0] as u128;
        let m3 = a2.shape()[1] as u128;
        let cost1 = m0
            .saturating_mul(m1)
            .saturating_mul(m2)
            .saturating_add(m0.saturating_mul(m2).saturating_mul(m3));
        let cost2 = m1
            .saturating_mul(m2)
            .saturating_mul(m3)
            .saturating_add(m0.saturating_mul(m1).saturating_mul(m3));
        if cost1 < cost2 {
            matmul(&matmul(a0, a1)?, a2)?
        } else {
            matmul(a0, &matmul(a1, a2)?)?
        }
    } else {
        // Cormen Matrix-Chain-Order DP over dimensions p[0..=n].
        let mut p = Vec::with_capacity(n + 1);
        p.push(mats[0].shape()[0] as u128);
        for mat in &mats {
            p.push(mat.shape()[1] as u128);
        }
        let mut cost = vec![vec![0u128; n]; n];
        let mut split = vec![vec![0usize; n]; n];
        for len in 1..n {
            for i in 0..n - len {
                let j = i + len;
                cost[i][j] = u128::MAX;
                for k in i..j {
                    let q = cost[i][k]
                        .saturating_add(cost[k + 1][j])
                        .saturating_add(p[i].saturating_mul(p[k + 1]).saturating_mul(p[j + 1]));
                    if q < cost[i][j] {
                        cost[i][j] = q;
                        split[i][j] = k;
                    }
                }
            }
        }
        fn eval_chain<T: ProductScalar>(
            mats: &[NdArray<T>],
            split: &[Vec<usize>],
            i: usize,
            j: usize,
        ) -> Result<NdArray<T>> {
            if i == j {
                Ok(mats[i].clone())
            } else {
                let k = split[i][j];
                let left = eval_chain(mats, split, i, k)?;
                let right = eval_chain(mats, split, k + 1, j)?;
                matmul(&left, &right)
            }
        }
        eval_chain(&mats, &split, 0, n - 1)?
    };

    match (squeeze_first, squeeze_last) {
        (true, true) => {
            let scalar = unsafe { *prod.storage_ptr().add(prod.offset()) };
            Ok(NdArray::from_scalar(scalar))
        }
        (true, false) => prod.reshape(&[prod.shape()[1] as isize]),
        (false, true) => prod.reshape(&[prod.shape()[0] as isize]),
        (false, false) => Ok(prod),
    }
}

/// Dynamic `multi_dot` promoting all inputs via `numpy_types::result_type`.
pub fn multi_dot_dyn(arrays: &[&DynArray]) -> Result<DynArray> {
    if arrays.len() < 2 {
        return Err(NumPyError::InvalidValue(
            "multi_dot expects at least 2 arrays".into(),
        ));
    }
    let dtypes: Vec<DType> = arrays.iter().map(|a| a.dtype()).collect();
    let target = result_type(&dtypes)?;
    let casted: Vec<DynArray> = arrays
        .iter()
        .map(|a| a.astype(target, numpy_types::Casting::Unsafe))
        .collect::<Result<Vec<_>>>()?;

    match target {
        DType::Bool => {
            let refs = extract_typed_refs!(casted, Bool);
            Ok(DynArray::Bool(multi_dot(&refs)?))
        }
        DType::I8 => {
            let refs = extract_typed_refs!(casted, I8);
            Ok(DynArray::I8(multi_dot(&refs)?))
        }
        DType::I16 => {
            let refs = extract_typed_refs!(casted, I16);
            Ok(DynArray::I16(multi_dot(&refs)?))
        }
        DType::I32 => {
            let refs = extract_typed_refs!(casted, I32);
            Ok(DynArray::I32(multi_dot(&refs)?))
        }
        DType::I64 => {
            let refs = extract_typed_refs!(casted, I64);
            Ok(DynArray::I64(multi_dot(&refs)?))
        }
        DType::U8 => {
            let refs = extract_typed_refs!(casted, U8);
            Ok(DynArray::U8(multi_dot(&refs)?))
        }
        DType::U16 => {
            let refs = extract_typed_refs!(casted, U16);
            Ok(DynArray::U16(multi_dot(&refs)?))
        }
        DType::U32 => {
            let refs = extract_typed_refs!(casted, U32);
            Ok(DynArray::U32(multi_dot(&refs)?))
        }
        DType::U64 => {
            let refs = extract_typed_refs!(casted, U64);
            Ok(DynArray::U64(multi_dot(&refs)?))
        }
        DType::F16 => {
            let refs = extract_typed_refs!(casted, F16);
            Ok(DynArray::F16(multi_dot(&refs)?))
        }
        DType::F32 => {
            let refs = extract_typed_refs!(casted, F32);
            Ok(DynArray::F32(multi_dot(&refs)?))
        }
        DType::F64 => {
            let refs = extract_typed_refs!(casted, F64);
            Ok(DynArray::F64(multi_dot(&refs)?))
        }
        DType::C64 => {
            let refs = extract_typed_refs!(casted, C64);
            Ok(DynArray::C64(multi_dot(&refs)?))
        }
        DType::C128 => {
            let refs = extract_typed_refs!(casted, C128);
            Ok(DynArray::C128(multi_dot(&refs)?))
        }
    }
}

macro_rules! extract_typed_refs {
    ($vec:expr, $variant:ident) => {{
        let mut refs = Vec::with_capacity($vec.len());
        for item in &$vec {
            match item {
                DynArray::$variant(a) => refs.push(a),
                _ => unreachable!("result_type guaranteed matching variant"),
            }
        }
        refs
    }};
}
use extract_typed_refs;

/// Sum along diagonal of `axis1` and `axis2` with diagonal offset `offset`.
pub fn trace_axes<T: ProductScalar>(
    a: &NdArray<T>,
    offset: isize,
    axis1: isize,
    axis2: isize,
) -> Result<NdArray<T>> {
    let nd = a.ndim();
    if nd < 2 {
        return Err(NumPyError::LinAlgError(format!(
            "trace requires at least 2 dimensions, got {nd}"
        )));
    }
    let ax1 = normalize_axis(axis1, nd)?;
    let ax2 = normalize_axis(axis2, nd)?;
    if ax1 == ax2 {
        return Err(NumPyError::InvalidValue(
            "axis1 and axis2 cannot be the same".into(),
        ));
    }

    let len1 = a.shape()[ax1];
    let len2 = a.shape()[ax2];
    let s1 = a.strides()[ax1];
    let s2 = a.strides()[ax2];

    let (start_r, start_c, diag_len) = if offset >= 0 {
        let off = offset as usize;
        let len = if off >= len2 {
            0
        } else {
            len1.min(len2 - off)
        };
        (0usize, off, len)
    } else {
        let off = offset.unsigned_abs();
        let len = if off >= len1 {
            0
        } else {
            (len1 - off).min(len2)
        };
        (off, 0usize, len)
    };

    let mut out_shape = Vec::with_capacity(nd - 2);
    let mut out_strides = Vec::with_capacity(nd - 2);
    for d in 0..nd {
        if d != ax1 && d != ax2 {
            out_shape.push(a.shape()[d]);
            out_strides.push(a.strides()[d]);
        }
    }

    let bcount = shape_size(&out_shape)?;
    let mut out = alloc_zeroed_buf::<T>(bcount);
    if bcount == 0 || diag_len == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    let base_diag_off = start_r as isize * s1 + start_c as isize * s2;
    let diag_step = s1 + s2;
    let a_base = SyncPtr(unsafe { a.storage_ptr().add(a.offset()).offset(base_diag_off) });

    if bcount == 1 {
        let mut acc = T::ZERO;
        for k in 0..diag_len {
            acc = acc.add_elem(unsafe { *a_base.offset(k as isize * diag_step) });
        }
        out.as_mut_slice()[0] = acc;
    } else if should_par_batches(bcount, diag_len, VEC_RAYON_THRESHOLD) {
        out.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dst)| {
                let oa = linear_strided_offset(i, &out_shape, &out_strides);
                let ptr = unsafe { a_base.offset(oa) };
                let mut acc = T::ZERO;
                for k in 0..diag_len {
                    acc = acc.add_elem(unsafe { *ptr.offset(k as isize * diag_step) });
                }
                *dst = acc;
            });
    } else {
        for (i, dst) in out.as_mut_slice().iter_mut().enumerate() {
            let oa = linear_strided_offset(i, &out_shape, &out_strides);
            let ptr = unsafe { a_base.offset(oa) };
            let mut acc = T::ZERO;
            for k in 0..diag_len {
                acc = acc.add_elem(unsafe { *ptr.offset(k as isize * diag_step) });
            }
            *dst = acc;
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Sum along diagonal of last two axes (`axis1 = -2, axis2 = -1`), matching `numpy.linalg.trace`.
#[inline]
pub fn trace<T: ProductScalar>(a: &NdArray<T>, offset: isize) -> Result<NdArray<T>> {
    trace_axes(a, offset, -2, -1)
}

/// Dynamic `trace_axes` with optional explicit output `dtype` (defaults to NumPy's `trace` promotion).
pub fn trace_axes_dyn(
    a: &DynArray,
    offset: isize,
    axis1: isize,
    axis2: isize,
    dtype: Option<DType>,
) -> Result<DynArray> {
    let target = dtype.unwrap_or_else(|| trace_default_dtype(a.dtype()));
    let casted = a.astype(target, numpy_types::Casting::Unsafe)?;
    dispatch_product_dyn!(&casted, va => Ok(to_dyn(trace_axes(va, offset, axis1, axis2)?)))
}

/// Dynamic `linalg.trace` (`axis1 = -2, axis2 = -1`).
#[inline]
pub fn trace_dyn(a: &DynArray, offset: isize) -> Result<DynArray> {
    trace_axes_dyn(a, offset, -2, -1, None)
}

/// Positive/zero integer matrix power over stacked square matrices `[..., M, M]`.
pub(crate) fn matrix_power_nonneg<T: ProductScalar>(a: &NdArray<T>, n: u64) -> Result<NdArray<T>> {
    let nd = a.ndim();
    if nd < 2 {
        return Err(NumPyError::LinAlgError(format!(
            "{nd}-dimensional array given. Array must be at least two-dimensional"
        )));
    }
    let m = a.shape()[nd - 2];
    let k = a.shape()[nd - 1];
    if m != k {
        return Err(NumPyError::LinAlgError(format!(
            "Last 2 dimensions of the array must be square, got ({m}, {k})"
        )));
    }
    if n == 0 {
        let shape = a.shape();
        let total = shape_size(shape)?;
        let mat_len = m.saturating_mul(m);
        let mut out = try_alloc_zeroed_buf::<T>(total)?;
        if mat_len > 0 {
            for batch in out.as_mut_slice().chunks_exact_mut(mat_len) {
                for i in 0..m {
                    batch[i * m + i] = T::ONE;
                }
            }
        }
        return ndarray_from_aligned(out, shape);
    }
    if n == 1 {
        return Ok(a.to_c_contiguous());
    }
    if n == 2 {
        return matmul(a, a);
    }
    if n == 3 {
        let a2 = matmul(a, a)?;
        return matmul(&a2, a);
    }

    // Binary exponentiation
    let mut base = a.to_c_contiguous();
    let mut exp = n;
    let mut result: Option<NdArray<T>> = None;
    while exp > 0 {
        if exp & 1 == 1 {
            result = Some(match result {
                None => base.clone(),
                Some(acc) => matmul(&acc, &base)?,
            });
        }
        exp >>= 1;
        if exp > 0 {
            base = matmul(&base, &base)?;
        }
    }
    Ok(result.expect("n >= 4 guarantees non-empty result"))
}

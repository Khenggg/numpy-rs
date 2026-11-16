//! Execution engine for NumPy ufuncs on `NdArray<T>`.
//!
//! Prioritizes:
//! 1. C-contiguous `as_slice()` fast path (`out_len == a.len()`), including scalar-broadcast
//!    fast path when one input has `len() == 1`.
//! 2. Inner-contiguous 2D+ broadcast fast path (`broadcast_strides.last() in {0, 1}`).
//! 3. General N-D strided odometer fallback (supports F-order, negative strides, arbitrary views).
//! 4. Rayon parallel chunking (`out_len >= PARALLEL_THRESHOLD = 32_768`) over `AlignedUninit<U>`.

use numpy_types::{
    broadcast_many_shapes, broadcast_shapes, broadcast_strides, shape_size, AlignedBuffer, NdArray,
    NumPyError, Order, Result, Scalar,
};
use rayon::prelude::*;
use std::mem::MaybeUninit;

/// Minimum output element count before dispatching across Rayon worker threads.
pub const PARALLEL_THRESHOLD: usize = 32_768;

/// Chunk size per Rayon task to stay cache-resident while amortizing scheduling overhead.
const PARALLEL_CHUNK: usize = 16_384;

#[derive(Copy, Clone)]
struct SendPtr<T>(*const T);

unsafe impl<T: Sync> Send for SendPtr<T> {}
unsafe impl<T: Sync> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline(always)]
    unsafe fn add_offset(self, off: isize) -> T
    where
        T: Copy,
    {
        unsafe { *self.0.offset(off) }
    }

    #[inline(always)]
    unsafe fn slice_from(self, off: isize, len: usize) -> &'static [T] {
        // Sound only within the lifetime of the borrowed NdArray backing buffer.
        unsafe { std::slice::from_raw_parts(self.0.offset(off), len) }
    }
}

#[inline]
pub(crate) fn empty_like_shape<U: Scalar>(shape: &[usize]) -> Result<NdArray<U>> {
    NdArray::from_aligned_buffer(AlignedBuffer::empty(), shape, Order::C)
}

#[inline]
unsafe fn array_base_ptr<T: Scalar>(arr: &NdArray<T>) -> SendPtr<T> {
    unsafe { SendPtr(arr.storage_ptr().add(arr.offset())) }
}

#[inline]
fn scalar_elem<T: Scalar>(arr: &NdArray<T>) -> T {
    debug_assert_eq!(arr.len(), 1);
    unsafe { *arr.storage_ptr().add(arr.offset()) }
}

#[inline]
fn linear_to_coords_and_off1(
    mut idx: usize,
    shape: &[usize],
    s0: &[isize],
    coords: &mut [usize],
) -> isize {
    let mut off = 0isize;
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        let c = idx % dim;
        idx /= dim;
        coords[d] = c;
        off += (c as isize) * s0[d];
    }
    off
}

#[inline]
fn linear_to_coords_and_off2(
    mut idx: usize,
    shape: &[usize],
    s0: &[isize],
    s1: &[isize],
    coords: &mut [usize],
) -> (isize, isize) {
    let mut off0 = 0isize;
    let mut off1 = 0isize;
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        let c = idx % dim;
        idx /= dim;
        coords[d] = c;
        let ci = c as isize;
        off0 += ci * s0[d];
        off1 += ci * s1[d];
    }
    (off0, off1)
}

#[inline]
fn linear_to_coords_and_off3(
    mut idx: usize,
    shape: &[usize],
    s0: &[isize],
    s1: &[isize],
    s2: &[isize],
    coords: &mut [usize],
) -> (isize, isize, isize) {
    let mut off0 = 0isize;
    let mut off1 = 0isize;
    let mut off2 = 0isize;
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        let c = idx % dim;
        idx /= dim;
        coords[d] = c;
        let ci = c as isize;
        off0 += ci * s0[d];
        off1 += ci * s1[d];
        off2 += ci * s2[d];
    }
    (off0, off1, off2)
}

#[inline]
fn linear_to_off1(mut idx: usize, shape: &[usize], s0: &[isize]) -> isize {
    let mut off = 0isize;
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        let c = idx % dim;
        idx /= dim;
        off += (c as isize) * s0[d];
    }
    off
}

#[inline]
fn linear_to_off2(mut idx: usize, shape: &[usize], s0: &[isize], s1: &[isize]) -> (isize, isize) {
    let mut off0 = 0isize;
    let mut off1 = 0isize;
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        let c = idx % dim;
        idx /= dim;
        let ci = c as isize;
        off0 += ci * s0[d];
        off1 += ci * s1[d];
    }
    (off0, off1)
}

#[inline]
fn run_unary_slice_parallel<A: Scalar, U: Scalar, K>(
    src: &[A],
    out: &mut [MaybeUninit<U>],
    kernel: &K,
) where
    K: Fn(&[A], &mut [MaybeUninit<U>]) + Sync,
{
    if out.len() >= PARALLEL_THRESHOLD {
        src.par_chunks(PARALLEL_CHUNK)
            .zip(out.par_chunks_mut(PARALLEL_CHUNK))
            .for_each(|(cs, co)| kernel(cs, co));
    } else {
        kernel(src, out);
    }
}

#[inline]
fn run_binary_slice_parallel<A: Scalar, B: Scalar, U: Scalar, K>(
    sa: &[A],
    sb: &[B],
    out: &mut [MaybeUninit<U>],
    kernel: &K,
) where
    K: Fn(&[A], &[B], &mut [MaybeUninit<U>]) + Sync,
{
    if out.len() >= PARALLEL_THRESHOLD {
        sa.par_chunks(PARALLEL_CHUNK)
            .zip(sb.par_chunks(PARALLEL_CHUNK))
            .zip(out.par_chunks_mut(PARALLEL_CHUNK))
            .for_each(|((ca, cb), co)| kernel(ca, cb, co));
    } else {
        kernel(sa, sb, out);
    }
}

#[inline]
fn run_rhs_scalar_parallel<A: Scalar, B: Scalar, U: Scalar, K>(
    sa: &[A],
    s: B,
    out: &mut [MaybeUninit<U>],
    kernel: &K,
) where
    K: Fn(&[A], B, &mut [MaybeUninit<U>]) + Sync,
{
    if out.len() >= PARALLEL_THRESHOLD {
        sa.par_chunks(PARALLEL_CHUNK)
            .zip(out.par_chunks_mut(PARALLEL_CHUNK))
            .for_each(|(ca, co)| kernel(ca, s, co));
    } else {
        kernel(sa, s, out);
    }
}

#[inline]
fn run_lhs_scalar_parallel<A: Scalar, B: Scalar, U: Scalar, K>(
    s: A,
    sb: &[B],
    out: &mut [MaybeUninit<U>],
    kernel: &K,
) where
    K: Fn(A, &[B], &mut [MaybeUninit<U>]) + Sync,
{
    if out.len() >= PARALLEL_THRESHOLD {
        sb.par_chunks(PARALLEL_CHUNK)
            .zip(out.par_chunks_mut(PARALLEL_CHUNK))
            .for_each(|(cb, co)| kernel(s, cb, co));
    } else {
        kernel(s, sb, out);
    }
}

/// Evaluate a unary ufunc using a slice kernel on contiguous regions and `elem_f` elsewhere.
pub(crate) fn eval_unary_slice<A: Scalar, U: Scalar, K, F>(
    a: &NdArray<A>,
    slice_k: K,
    elem_f: F,
) -> Result<NdArray<U>>
where
    K: Fn(&[A], &mut [MaybeUninit<U>]) + Sync,
    F: Fn(A) -> U + Sync,
{
    let shape = a.shape();
    let out_len = a.len();
    if out_len == 0 {
        return empty_like_shape(shape);
    }

    let mut uninit = AlignedBuffer::<U>::new_uninit(out_len);
    if let Some(sa) = a.as_slice() {
        run_unary_slice_parallel(sa, uninit.as_mut_slice(), &slice_k);
        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, shape, Order::C);
    }

    let strides = a.strides();
    let ndim = shape.len();
    let ptr_a = unsafe { array_base_ptr(a) };

    if ndim >= 2 && (strides[ndim - 1] == 0 || strides[ndim - 1] == 1) {
        let last_dim = shape[ndim - 1];
        let outer_shape = &shape[..ndim - 1];
        let outer_sa = &strides[..ndim - 1];
        let n_rows = out_len / last_dim;
        if strides[ndim - 1] == 1 {
            if last_dim >= PARALLEL_THRESHOLD {
                for (r, out_row) in uninit.as_mut_slice().chunks_mut(last_dim).enumerate() {
                    let off = linear_to_off1(r, outer_shape, outer_sa);
                    let row_a = unsafe { ptr_a.slice_from(off, last_dim) };
                    run_unary_slice_parallel(row_a, out_row, &slice_k);
                }
            } else if out_len >= PARALLEL_THRESHOLD {
                uninit
                    .as_mut_slice()
                    .par_chunks_mut(last_dim)
                    .enumerate()
                    .for_each(|(r, out_row)| {
                        let off = linear_to_off1(r, outer_shape, outer_sa);
                        let row_a = unsafe { ptr_a.slice_from(off, last_dim) };
                        slice_k(row_a, out_row);
                    });
            } else {
                for r in 0..n_rows {
                    let off = linear_to_off1(r, outer_shape, outer_sa);
                    let row_a = unsafe { ptr_a.slice_from(off, last_dim) };
                    let start = r * last_dim;
                    slice_k(row_a, &mut uninit.as_mut_slice()[start..start + last_dim]);
                }
            }
        } else {
            // strides[ndim - 1] == 0: constant along each row
            let fill_row = |r: usize, out_row: &mut [MaybeUninit<U>]| {
                let off = linear_to_off1(r, outer_shape, outer_sa);
                let val = elem_f(unsafe { ptr_a.add_offset(off) });
                out_row.fill(MaybeUninit::new(val));
            };
            if out_len >= PARALLEL_THRESHOLD {
                uninit
                    .as_mut_slice()
                    .par_chunks_mut(last_dim)
                    .enumerate()
                    .for_each(|(r, out_row)| fill_row(r, out_row));
            } else {
                for (r, out_row) in uninit.as_mut_slice().chunks_mut(last_dim).enumerate() {
                    fill_row(r, out_row);
                }
            }
        }
        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, shape, Order::C);
    }

    // Fast path for 1D strided views (avoids heap allocation & N-D odometer)
    if ndim == 1 {
        let s0 = strides[0];
        let process_1d = |start: usize, chunk: &mut [MaybeUninit<U>]| {
            let mut off = (start as isize) * s0;
            for slot in chunk.iter_mut() {
                slot.write(elem_f(unsafe { ptr_a.add_offset(off) }));
                off += s0;
            }
        };
        if out_len >= PARALLEL_THRESHOLD {
            uninit
                .as_mut_slice()
                .par_chunks_mut(PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| process_1d(ci * PARALLEL_CHUNK, chunk));
        } else {
            process_1d(0, uninit.as_mut_slice());
        }
        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, shape, Order::C);
    }

    // General strided fallback
    eval_unary_strided_into(ptr_a, shape, strides, uninit.as_mut_slice(), &elem_f);
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

/// Evaluate a scalar-closure unary ufunc on `NdArray<A>`.
pub(crate) fn eval_unary<A: Scalar, U: Scalar, F>(a: &NdArray<A>, f: F) -> Result<NdArray<U>>
where
    F: Fn(A) -> U + Sync,
{
    eval_unary_slice(
        a,
        |src, out| {
            for (slot, &x) in out.iter_mut().zip(src.iter()) {
                slot.write(f(x));
            }
        },
        &f,
    )
}

/// Evaluate a fallible scalar-closure unary ufunc on `NdArray<A>`.
pub(crate) fn try_eval_unary<A: Scalar, U: Scalar, F>(a: &NdArray<A>, f: F) -> Result<NdArray<U>>
where
    F: Fn(A) -> Result<U> + Sync,
{
    let shape = a.shape();
    let out_len = a.len();
    if out_len == 0 {
        return empty_like_shape(shape);
    }

    let mut uninit = AlignedBuffer::<U>::new_uninit(out_len);
    if let Some(sa) = a.as_slice() {
        if out_len >= PARALLEL_THRESHOLD {
            sa.par_chunks(PARALLEL_CHUNK)
                .zip(uninit.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                .try_for_each(|(cs, co)| -> Result<()> {
                    for (slot, &x) in co.iter_mut().zip(cs.iter()) {
                        slot.write(f(x)?);
                    }
                    Ok(())
                })?;
        } else {
            for (slot, &x) in uninit.as_mut_slice().iter_mut().zip(sa.iter()) {
                slot.write(f(x)?);
            }
        }
        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, shape, Order::C);
    }

    let strides = a.strides();
    let ptr_a = unsafe { array_base_ptr(a) };
    let process_chunk = |start: usize, chunk: &mut [MaybeUninit<U>]| -> Result<()> {
        let ndim = shape.len();
        let mut coords = vec![0usize; ndim];
        let mut off = linear_to_coords_and_off1(start, shape, strides, &mut coords);
        let back: Vec<isize> = shape
            .iter()
            .zip(strides.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let n = chunk.len();
        for (k, slot) in chunk.iter_mut().enumerate() {
            slot.write(f(unsafe { ptr_a.add_offset(off) })?);
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < shape[d] {
                        off += strides[d];
                        break;
                    }
                    coords[d] = 0;
                    off -= back[d];
                }
            }
        }
        Ok(())
    };

    if out_len >= PARALLEL_THRESHOLD {
        uninit
            .as_mut_slice()
            .par_chunks_mut(PARALLEL_CHUNK)
            .enumerate()
            .try_for_each(|(ci, chunk)| process_chunk(ci * PARALLEL_CHUNK, chunk))?;
    } else {
        process_chunk(0, uninit.as_mut_slice())?;
    }

    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, shape, Order::C)
}

/// Evaluate a 2-output unary ufunc (e.g., `frexp`, `modf`).
pub(crate) fn eval_unary2<A: Scalar, U1: Scalar, U2: Scalar, F>(
    a: &NdArray<A>,
    f: F,
) -> Result<(NdArray<U1>, NdArray<U2>)>
where
    F: Fn(A) -> (U1, U2) + Sync,
{
    let shape = a.shape();
    let out_len = a.len();
    if out_len == 0 {
        return Ok((empty_like_shape(shape)?, empty_like_shape(shape)?));
    }

    let mut u1 = AlignedBuffer::<U1>::new_uninit(out_len);
    let mut u2 = AlignedBuffer::<U2>::new_uninit(out_len);

    if let Some(sa) = a.as_slice() {
        if out_len >= PARALLEL_THRESHOLD {
            sa.par_chunks(PARALLEL_CHUNK)
                .zip(u1.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                .zip(u2.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                .for_each(|((cs, c1), c2)| {
                    for ((s1, s2), &x) in c1.iter_mut().zip(c2.iter_mut()).zip(cs.iter()) {
                        let (r1, r2) = f(x);
                        s1.write(r1);
                        s2.write(r2);
                    }
                });
        } else {
            for ((s1, s2), &x) in u1
                .as_mut_slice()
                .iter_mut()
                .zip(u2.as_mut_slice().iter_mut())
                .zip(sa.iter())
            {
                let (r1, r2) = f(x);
                s1.write(r1);
                s2.write(r2);
            }
        }
        let b1 = unsafe { u1.assume_init() };
        let b2 = unsafe { u2.assume_init() };
        return Ok((
            NdArray::from_aligned_buffer(b1, shape, Order::C)?,
            NdArray::from_aligned_buffer(b2, shape, Order::C)?,
        ));
    }

    let strides = a.strides();
    let ptr_a = unsafe { array_base_ptr(a) };
    let process_chunk = |start: usize, c1: &mut [MaybeUninit<U1>], c2: &mut [MaybeUninit<U2>]| {
        let ndim = shape.len();
        let mut coords = vec![0usize; ndim];
        let mut off = linear_to_coords_and_off1(start, shape, strides, &mut coords);
        let back: Vec<isize> = shape
            .iter()
            .zip(strides.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let n = c1.len();
        for (k, (s1, s2)) in c1.iter_mut().zip(c2.iter_mut()).enumerate() {
            let (r1, r2) = f(unsafe { ptr_a.add_offset(off) });
            s1.write(r1);
            s2.write(r2);
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < shape[d] {
                        off += strides[d];
                        break;
                    }
                    coords[d] = 0;
                    off -= back[d];
                }
            }
        }
    };

    if out_len >= PARALLEL_THRESHOLD {
        u1.as_mut_slice()
            .par_chunks_mut(PARALLEL_CHUNK)
            .zip(u2.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
            .enumerate()
            .for_each(|(ci, (c1, c2))| process_chunk(ci * PARALLEL_CHUNK, c1, c2));
    } else {
        process_chunk(0, u1.as_mut_slice(), u2.as_mut_slice());
    }

    let b1 = unsafe { u1.assume_init() };
    let b2 = unsafe { u2.assume_init() };
    Ok((
        NdArray::from_aligned_buffer(b1, shape, Order::C)?,
        NdArray::from_aligned_buffer(b2, shape, Order::C)?,
    ))
}

fn eval_unary_strided_into<A: Scalar, U: Scalar, F>(
    ptr_a: SendPtr<A>,
    shape: &[usize],
    strides: &[isize],
    out: &mut [MaybeUninit<U>],
    f: &F,
) where
    F: Fn(A) -> U + Sync,
{
    let out_len = out.len();
    let process_chunk = |start: usize, chunk: &mut [MaybeUninit<U>]| {
        let ndim = shape.len();
        let mut coords = vec![0usize; ndim];
        let mut off = linear_to_coords_and_off1(start, shape, strides, &mut coords);
        let back: Vec<isize> = shape
            .iter()
            .zip(strides.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let n = chunk.len();
        for (k, slot) in chunk.iter_mut().enumerate() {
            slot.write(f(unsafe { ptr_a.add_offset(off) }));
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < shape[d] {
                        off += strides[d];
                        break;
                    }
                    coords[d] = 0;
                    off -= back[d];
                }
            }
        }
    };

    if out_len >= PARALLEL_THRESHOLD {
        out.par_chunks_mut(PARALLEL_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| process_chunk(ci * PARALLEL_CHUNK, chunk));
    } else {
        process_chunk(0, out);
    }
}

/// Evaluate a binary ufunc with explicit contiguous, scalar-broadcast, inner-contiguous,
/// and general N-D strided kernels.
pub(crate) fn eval_binary_simd<A: Scalar, B: Scalar, U: Scalar, K, KR, KL, F>(
    a: &NdArray<A>,
    b: &NdArray<B>,
    bin_k: K,
    rhs_scalar_k: KR,
    lhs_scalar_k: KL,
    elem_f: F,
) -> Result<NdArray<U>>
where
    K: Fn(&[A], &[B], &mut [MaybeUninit<U>]) + Sync,
    KR: Fn(&[A], B, &mut [MaybeUninit<U>]) + Sync,
    KL: Fn(A, &[B], &mut [MaybeUninit<U>]) + Sync,
    F: Fn(A, B) -> U + Sync,
{
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let out_len = shape_size(&out_shape)?;
    if out_len == 0 {
        return empty_like_shape(&out_shape);
    }

    let mut uninit = AlignedBuffer::<U>::new_uninit(out_len);

    // Fast Path 1: Both inputs C-contiguous covering the full output shape.
    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        if sa.len() == out_len && sb.len() == out_len {
            run_binary_slice_parallel(sa, sb, uninit.as_mut_slice(), &bin_k);
            let buf = unsafe { uninit.assume_init() };
            return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
        }
    }

    // Fast Path 2: One input is a 1-element scalar broadcast onto a full C-contiguous array.
    if b.len() == 1 {
        if let Some(sa) = a.as_slice() {
            if sa.len() == out_len {
                let sb_val = scalar_elem(b);
                run_rhs_scalar_parallel(sa, sb_val, uninit.as_mut_slice(), &rhs_scalar_k);
                let buf = unsafe { uninit.assume_init() };
                return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
            }
        }
    }
    if a.len() == 1 {
        if let Some(sb) = b.as_slice() {
            if sb.len() == out_len {
                let sa_val = scalar_elem(a);
                run_lhs_scalar_parallel(sa_val, sb, uninit.as_mut_slice(), &lhs_scalar_k);
                let buf = unsafe { uninit.assume_init() };
                return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
            }
        }
    }

    let sa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let sb = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let ptr_a = unsafe { array_base_ptr(a) };
    let ptr_b = unsafe { array_base_ptr(b) };
    let ndim = out_shape.len();

    // Fast Path 3: Innermost axis is contiguous or scalar-broadcast on both inputs.
    if ndim >= 1
        && (sa[ndim - 1] == 0 || sa[ndim - 1] == 1)
        && (sb[ndim - 1] == 0 || sb[ndim - 1] == 1)
    {
        let last_dim = out_shape[ndim - 1];
        let outer_shape = &out_shape[..ndim - 1];
        let outer_sa = &sa[..ndim - 1];
        let outer_sb = &sb[..ndim - 1];
        let sa_last = sa[ndim - 1];
        let sb_last = sb[ndim - 1];

        let run_row = |r: usize, out_row: &mut [MaybeUninit<U>], parallel_inner: bool| {
            let (off_a, off_b) = linear_to_off2(r, outer_shape, outer_sa, outer_sb);
            match (sa_last, sb_last) {
                (1, 1) => {
                    let row_a = unsafe { ptr_a.slice_from(off_a, last_dim) };
                    let row_b = unsafe { ptr_b.slice_from(off_b, last_dim) };
                    if parallel_inner {
                        run_binary_slice_parallel(row_a, row_b, out_row, &bin_k);
                    } else {
                        bin_k(row_a, row_b, out_row);
                    }
                }
                (1, 0) => {
                    let row_a = unsafe { ptr_a.slice_from(off_a, last_dim) };
                    let bv = unsafe { ptr_b.add_offset(off_b) };
                    if parallel_inner {
                        run_rhs_scalar_parallel(row_a, bv, out_row, &rhs_scalar_k);
                    } else {
                        rhs_scalar_k(row_a, bv, out_row);
                    }
                }
                (0, 1) => {
                    let av = unsafe { ptr_a.add_offset(off_a) };
                    let row_b = unsafe { ptr_b.slice_from(off_b, last_dim) };
                    if parallel_inner {
                        run_lhs_scalar_parallel(av, row_b, out_row, &lhs_scalar_k);
                    } else {
                        lhs_scalar_k(av, row_b, out_row);
                    }
                }
                _ => {
                    let av = unsafe { ptr_a.add_offset(off_a) };
                    let bv = unsafe { ptr_b.add_offset(off_b) };
                    out_row.fill(MaybeUninit::new(elem_f(av, bv)));
                }
            }
        };

        if last_dim >= PARALLEL_THRESHOLD {
            for (r, out_row) in uninit.as_mut_slice().chunks_mut(last_dim).enumerate() {
                run_row(r, out_row, true);
            }
        } else if out_len >= PARALLEL_THRESHOLD {
            uninit
                .as_mut_slice()
                .par_chunks_mut(last_dim)
                .enumerate()
                .for_each(|(r, out_row)| run_row(r, out_row, false));
        } else {
            for (r, out_row) in uninit.as_mut_slice().chunks_mut(last_dim).enumerate() {
                run_row(r, out_row, false);
            }
        }

        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
    }

    // Fast path for 1D strided / 1D-broadcast views (zero heap allocation, simple linear step)
    if ndim == 1 {
        let sa0 = sa[0];
        let sb0 = sb[0];
        let process_1d = |start: usize, chunk: &mut [MaybeUninit<U>]| {
            let mut oa = (start as isize) * sa0;
            let mut ob = (start as isize) * sb0;
            for slot in chunk.iter_mut() {
                let av = unsafe { ptr_a.add_offset(oa) };
                let bv = unsafe { ptr_b.add_offset(ob) };
                slot.write(elem_f(av, bv));
                oa += sa0;
                ob += sb0;
            }
        };
        if out_len >= PARALLEL_THRESHOLD {
            uninit
                .as_mut_slice()
                .par_chunks_mut(PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| process_1d(ci * PARALLEL_CHUNK, chunk));
        } else {
            process_1d(0, uninit.as_mut_slice());
        }
        let buf = unsafe { uninit.assume_init() };
        return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
    }

    // Fallback: General N-D strided odometer (supports F-order, negative strides, arbitrary axes).
    let process_chunk = |start: usize, chunk: &mut [MaybeUninit<U>]| {
        let mut coords = vec![0usize; ndim];
        let (mut off_a, mut off_b) =
            linear_to_coords_and_off2(start, &out_shape, &sa, &sb, &mut coords);
        let back_a: Vec<isize> = out_shape
            .iter()
            .zip(sa.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let back_b: Vec<isize> = out_shape
            .iter()
            .zip(sb.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let n = chunk.len();
        for (k, slot) in chunk.iter_mut().enumerate() {
            let av = unsafe { ptr_a.add_offset(off_a) };
            let bv = unsafe { ptr_b.add_offset(off_b) };
            slot.write(elem_f(av, bv));
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < out_shape[d] {
                        off_a += sa[d];
                        off_b += sb[d];
                        break;
                    }
                    coords[d] = 0;
                    off_a -= back_a[d];
                    off_b -= back_b[d];
                }
            }
        }
    };

    if out_len >= PARALLEL_THRESHOLD {
        uninit
            .as_mut_slice()
            .par_chunks_mut(PARALLEL_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| process_chunk(ci * PARALLEL_CHUNK, chunk));
    } else {
        process_chunk(0, uninit.as_mut_slice());
    }

    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &out_shape, Order::C)
}

/// Evaluate a scalar-closure binary ufunc with broadcasting & Rayon parallelism.
pub(crate) fn eval_binary<A: Scalar, B: Scalar, U: Scalar, F>(
    a: &NdArray<A>,
    b: &NdArray<B>,
    f: F,
) -> Result<NdArray<U>>
where
    F: Fn(A, B) -> U + Sync,
{
    eval_binary_simd(
        a,
        b,
        |sa, sb, out| {
            for ((slot, &x), &y) in out.iter_mut().zip(sa.iter()).zip(sb.iter()) {
                slot.write(f(x, y));
            }
        },
        |sa, y, out| {
            for (slot, &x) in out.iter_mut().zip(sa.iter()) {
                slot.write(f(x, y));
            }
        },
        |x, sb, out| {
            for (slot, &y) in out.iter_mut().zip(sb.iter()) {
                slot.write(f(x, y));
            }
        },
        &f,
    )
}

/// Evaluate a fallible scalar-closure binary ufunc with broadcasting & Rayon parallelism.
pub(crate) fn try_eval_binary<A: Scalar, B: Scalar, U: Scalar, F>(
    a: &NdArray<A>,
    b: &NdArray<B>,
    f: F,
) -> Result<NdArray<U>>
where
    F: Fn(A, B) -> Result<U> + Sync,
{
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let out_len = shape_size(&out_shape)?;
    if out_len == 0 {
        return empty_like_shape(&out_shape);
    }

    let mut uninit = AlignedBuffer::<U>::new_uninit(out_len);

    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        if sa.len() == out_len && sb.len() == out_len {
            if out_len >= PARALLEL_THRESHOLD {
                sa.par_chunks(PARALLEL_CHUNK)
                    .zip(sb.par_chunks(PARALLEL_CHUNK))
                    .zip(uninit.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                    .try_for_each(|((ca, cb), co)| -> Result<()> {
                        for ((slot, &x), &y) in co.iter_mut().zip(ca.iter()).zip(cb.iter()) {
                            slot.write(f(x, y)?);
                        }
                        Ok(())
                    })?;
            } else {
                for ((slot, &x), &y) in uninit
                    .as_mut_slice()
                    .iter_mut()
                    .zip(sa.iter())
                    .zip(sb.iter())
                {
                    slot.write(f(x, y)?);
                }
            }
            let buf = unsafe { uninit.assume_init() };
            return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
        }
    }

    let sa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let sb = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let ptr_a = unsafe { array_base_ptr(a) };
    let ptr_b = unsafe { array_base_ptr(b) };
    let ndim = out_shape.len();

    let process_chunk = |start: usize, chunk: &mut [MaybeUninit<U>]| -> Result<()> {
        let mut coords = vec![0usize; ndim];
        let (mut off_a, mut off_b) =
            linear_to_coords_and_off2(start, &out_shape, &sa, &sb, &mut coords);
        let back_a: Vec<isize> = out_shape
            .iter()
            .zip(sa.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let back_b: Vec<isize> = out_shape
            .iter()
            .zip(sb.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let n = chunk.len();
        for (k, slot) in chunk.iter_mut().enumerate() {
            let av = unsafe { ptr_a.add_offset(off_a) };
            let bv = unsafe { ptr_b.add_offset(off_b) };
            slot.write(f(av, bv)?);
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < out_shape[d] {
                        off_a += sa[d];
                        off_b += sb[d];
                        break;
                    }
                    coords[d] = 0;
                    off_a -= back_a[d];
                    off_b -= back_b[d];
                }
            }
        }
        Ok(())
    };

    if out_len >= PARALLEL_THRESHOLD {
        uninit
            .as_mut_slice()
            .par_chunks_mut(PARALLEL_CHUNK)
            .enumerate()
            .try_for_each(|(ci, chunk)| process_chunk(ci * PARALLEL_CHUNK, chunk))?;
    } else {
        process_chunk(0, uninit.as_mut_slice())?;
    }

    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &out_shape, Order::C)
}

/// Evaluate a 2-output binary ufunc (e.g., `divmod`).
pub(crate) fn eval_binary2<A: Scalar, B: Scalar, U1: Scalar, U2: Scalar, F>(
    a: &NdArray<A>,
    b: &NdArray<B>,
    f: F,
) -> Result<(NdArray<U1>, NdArray<U2>)>
where
    F: Fn(A, B) -> (U1, U2) + Sync,
{
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let out_len = shape_size(&out_shape)?;
    if out_len == 0 {
        return Ok((empty_like_shape(&out_shape)?, empty_like_shape(&out_shape)?));
    }

    let mut u1 = AlignedBuffer::<U1>::new_uninit(out_len);
    let mut u2 = AlignedBuffer::<U2>::new_uninit(out_len);

    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        if sa.len() == out_len && sb.len() == out_len {
            if out_len >= PARALLEL_THRESHOLD {
                sa.par_chunks(PARALLEL_CHUNK)
                    .zip(sb.par_chunks(PARALLEL_CHUNK))
                    .zip(u1.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                    .zip(u2.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                    .for_each(|(((ca, cb), c1), c2)| {
                        for (((s1, s2), &x), &y) in
                            c1.iter_mut().zip(c2.iter_mut()).zip(ca.iter()).zip(cb.iter())
                        {
                            let (r1, r2) = f(x, y);
                            s1.write(r1);
                            s2.write(r2);
                        }
                    });
            } else {
                for (((s1, s2), &x), &y) in u1
                    .as_mut_slice()
                    .iter_mut()
                    .zip(u2.as_mut_slice().iter_mut())
                    .zip(sa.iter())
                    .zip(sb.iter())
                {
                    let (r1, r2) = f(x, y);
                    s1.write(r1);
                    s2.write(r2);
                }
            }
            let b1 = unsafe { u1.assume_init() };
            let b2 = unsafe { u2.assume_init() };
            return Ok((
                NdArray::from_aligned_buffer(b1, &out_shape, Order::C)?,
                NdArray::from_aligned_buffer(b2, &out_shape, Order::C)?,
            ));
        }
    }

    let sa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let sb = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let ptr_a = unsafe { array_base_ptr(a) };
    let ptr_b = unsafe { array_base_ptr(b) };
    let ndim = out_shape.len();

    let process_chunk = |start: usize, c1: &mut [MaybeUninit<U1>], c2: &mut [MaybeUninit<U2>]| {
        let mut coords = vec![0usize; ndim];
        let (mut off_a, mut off_b) =
            linear_to_coords_and_off2(start, &out_shape, &sa, &sb, &mut coords);
        let back_a: Vec<isize> = out_shape
            .iter()
            .zip(sa.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let back_b: Vec<isize> = out_shape
            .iter()
            .zip(sb.iter())
            .map(|(&dim, &s)| (dim as isize - 1) * s)
            .collect();
        let n = c1.len();
        for (k, (s1, s2)) in c1.iter_mut().zip(c2.iter_mut()).enumerate() {
            let av = unsafe { ptr_a.add_offset(off_a) };
            let bv = unsafe { ptr_b.add_offset(off_b) };
            let (r1, r2) = f(av, bv);
            s1.write(r1);
            s2.write(r2);
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < out_shape[d] {
                        off_a += sa[d];
                        off_b += sb[d];
                        break;
                    }
                    coords[d] = 0;
                    off_a -= back_a[d];
                    off_b -= back_b[d];
                }
            }
        }
    };

    if out_len >= PARALLEL_THRESHOLD {
        u1.as_mut_slice()
            .par_chunks_mut(PARALLEL_CHUNK)
            .zip(u2.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
            .enumerate()
            .for_each(|(ci, (c1, c2))| process_chunk(ci * PARALLEL_CHUNK, c1, c2));
    } else {
        process_chunk(0, u1.as_mut_slice(), u2.as_mut_slice());
    }

    let b1 = unsafe { u1.assume_init() };
    let b2 = unsafe { u2.assume_init() };
    Ok((
        NdArray::from_aligned_buffer(b1, &out_shape, Order::C)?,
        NdArray::from_aligned_buffer(b2, &out_shape, Order::C)?,
    ))
}

/// Evaluate a 3-input broadcasted elementwise op (e.g. `clip`).
pub(crate) fn eval_ternary<A: Scalar, B: Scalar, C: Scalar, U: Scalar, F>(
    a: &NdArray<A>,
    b: &NdArray<B>,
    c: &NdArray<C>,
    f: F,
) -> Result<NdArray<U>>
where
    F: Fn(A, B, C) -> U + Sync,
{
    let out_shape = broadcast_many_shapes(&[a.shape(), b.shape(), c.shape()])?;
    let out_len = shape_size(&out_shape)?;
    if out_len == 0 {
        return empty_like_shape(&out_shape);
    }

    let mut uninit = AlignedBuffer::<U>::new_uninit(out_len);
    if let (Some(sa), Some(sb), Some(sc)) = (a.as_slice(), b.as_slice(), c.as_slice()) {
        if sa.len() == out_len && sb.len() == out_len && sc.len() == out_len {
            if out_len >= PARALLEL_THRESHOLD {
                sa.par_chunks(PARALLEL_CHUNK)
                    .zip(sb.par_chunks(PARALLEL_CHUNK))
                    .zip(sc.par_chunks(PARALLEL_CHUNK))
                    .zip(uninit.as_mut_slice().par_chunks_mut(PARALLEL_CHUNK))
                    .for_each(|(((ca, cb), cc), co)| {
                        for (((slot, &x), &y), &z) in
                            co.iter_mut().zip(ca.iter()).zip(cb.iter()).zip(cc.iter())
                        {
                            slot.write(f(x, y, z));
                        }
                    });
            } else {
                for (((slot, &x), &y), &z) in uninit
                    .as_mut_slice()
                    .iter_mut()
                    .zip(sa.iter())
                    .zip(sb.iter())
                    .zip(sc.iter())
                {
                    slot.write(f(x, y, z));
                }
            }
            let buf = unsafe { uninit.assume_init() };
            return NdArray::from_aligned_buffer(buf, &out_shape, Order::C);
        }
    }

    let sa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let sb = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let sc = broadcast_strides(c.shape(), c.strides(), &out_shape)?;
    let ptr_a = unsafe { array_base_ptr(a) };
    let ptr_b = unsafe { array_base_ptr(b) };
    let ptr_c = unsafe { array_base_ptr(c) };
    let ndim = out_shape.len();

    let process_chunk = |start: usize, chunk: &mut [MaybeUninit<U>]| {
        let mut coords = vec![0usize; ndim];
        let (mut off_a, mut off_b, mut off_c) =
            linear_to_coords_and_off3(start, &out_shape, &sa, &sb, &sc, &mut coords);
        let back_a: Vec<isize> = out_shape
            .iter()
            .zip(sa.iter())
            .map(|(&d, &s)| (d as isize - 1) * s)
            .collect();
        let back_b: Vec<isize> = out_shape
            .iter()
            .zip(sb.iter())
            .map(|(&d, &s)| (d as isize - 1) * s)
            .collect();
        let back_c: Vec<isize> = out_shape
            .iter()
            .zip(sc.iter())
            .map(|(&d, &s)| (d as isize - 1) * s)
            .collect();
        let n = chunk.len();
        for (k, slot) in chunk.iter_mut().enumerate() {
            let av = unsafe { ptr_a.add_offset(off_a) };
            let bv = unsafe { ptr_b.add_offset(off_b) };
            let cv = unsafe { ptr_c.add_offset(off_c) };
            slot.write(f(av, bv, cv));
            if k + 1 < n {
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < out_shape[d] {
                        off_a += sa[d];
                        off_b += sb[d];
                        off_c += sc[d];
                        break;
                    }
                    coords[d] = 0;
                    off_a -= back_a[d];
                    off_b -= back_b[d];
                    off_c -= back_c[d];
                }
            }
        }
    };

    if out_len >= PARALLEL_THRESHOLD {
        uninit
            .as_mut_slice()
            .par_chunks_mut(PARALLEL_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| process_chunk(ci * PARALLEL_CHUNK, chunk));
    } else {
        process_chunk(0, uninit.as_mut_slice());
    }

    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &out_shape, Order::C)
}

/// Fast `all` reduction over a `bool` array (empty returns `true`).
#[inline]
pub(crate) fn bool_all(a: &NdArray<bool>) -> bool {
    if let Some(s) = a.as_slice() {
        if s.len() >= PARALLEL_THRESHOLD {
            return s.par_chunks(PARALLEL_CHUNK).all(|c| c.iter().all(|&x| x));
        }
        return s.iter().all(|&x| x);
    }
    a.iter().all(|x| x)
}

/// Fast `any` reduction over a `bool` array (empty returns `false`).
#[inline]
#[allow(dead_code)]
pub(crate) fn bool_any(a: &NdArray<bool>) -> bool {
    if let Some(s) = a.as_slice() {
        if s.len() >= PARALLEL_THRESHOLD {
            return s.par_chunks(PARALLEL_CHUNK).any(|c| c.iter().any(|&x| x));
        }
        return s.iter().any(|&x| x);
    }
    a.iter().any(|x| x)
}

/// Allocate an empty error-checking helper for unsupported dtype combinations.
#[inline]
pub(crate) fn dtype_err(op: &str, msg: &str) -> NumPyError {
    NumPyError::DTypeError(format!("ufunc '{op}' not supported for {msg}"))
}

//! Determinants, norms, condition number, and matrix rank:
//! det, slogdet, matrix_rank, norm, vector_norm, matrix_norm, cond.

use std::collections::HashSet;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, Par};
use numpy_types::{DynArray, NdArray, NumPyError, Result, Scalar, normalize_axis};
use rayon::prelude::*;

use crate::batch::{
    DECOMP_RAYON_THRESHOLD, alloc_zeroed_buf, batch_len, copy_strided_to_faer_mat,
    linear_strided_offset, mat_any_nan, matrix_stack_descriptor, ndarray_from_aligned,
    parse_stacked_2d, parse_stacked_square, should_par_batches, single_mat_par,
};
use crate::decomp::{eigvalsh, svdvals};
use crate::dispatch_linalg_dyn;
use crate::solve::inv;
use crate::traits::{
    LinalgReal, LinalgScalar, NormOrd, NormRealScalar, NormScalar, SlogdetOutput, SlogdetResult,
    Uplo,
};

const NORM_RAYON_THRESHOLD: usize = 65_536;

/// Compute `(sign, logabsdet)` of stacked square matrices `[..., N, N]`, matching `numpy.linalg.slogdet`.
pub fn slogdet<T: LinalgScalar>(a: &NdArray<T>) -> Result<SlogdetOutput<T>> {
    let (batch_shape_ref, n) = parse_stacked_square(a)?;
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);

    if bcount == 0 {
        return Ok(SlogdetOutput {
            sign: NdArray::zeros(&batch_shape)?,
            logabsdet: NdArray::zeros(&batch_shape)?,
        });
    }
    if n == 0 {
        return Ok(SlogdetOutput {
            sign: NdArray::ones(&batch_shape)?,
            logabsdet: NdArray::zeros(&batch_shape)?,
        });
    }

    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
    let mut sign_buf = alloc_zeroed_buf::<T>(bcount);
    let mut log_buf = alloc_zeroed_buf::<T::Real>(bcount);

    struct LuWs<T: LinalgScalar> {
        lu_mat: Mat<T>,
        perm_fwd: Vec<usize>,
        perm_bwd: Vec<usize>,
        mem: MemBuffer,
    }

    let make_ws = |par: Par| -> LuWs<T> {
        let req = T::faer_lu_scratch(n, n, par);
        LuWs {
            lu_mat: Mat::full(n, n, T::ZERO),
            perm_fwd: vec![0; n],
            perm_bwd: vec![0; n],
            mem: MemBuffer::new(req),
        }
    };

    let run_one = |idx: usize, dst_sign: &mut T, dst_log: &mut T::Real, par: Par, ws: &mut LuWs<T>| {
        let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
        let src_ptr = unsafe { a_base.offset(oa) };
        let has_nan = unsafe { mat_any_nan(src_ptr, n, n, rsa, csa) };
        if has_nan {
            *dst_sign = if T::is_complex_type() {
                T::nan_scalar()
            } else {
                T::ONE
            };
            *dst_log = T::Real::nan_val();
            return;
        }
        unsafe {
            copy_strided_to_faer_mat(src_ptr, n, n, rsa, csa, &mut ws.lu_mat);
        }
        let (tcount, _) = T::faer_lu_in_place(
            ws.lu_mat.as_mut(),
            &mut ws.perm_fwd,
            &mut ws.perm_bwd,
            par,
            MemStack::new(&mut ws.mem),
        );

        let mut acc_sign = if tcount % 2 != 0 {
            T::ONE.negate_val()
        } else {
            T::ONE
        };
        let mut acc_log = 0.0f64;
        let mut any_zero = false;
        let mut diag_nan = false;

        for i in 0..n {
            let d = ws.lu_mat[(i, i)];
            if d.is_nan_elem() {
                diag_nan = true;
                break;
            }
            if T::is_complex_type() {
                let re = d.real_part().to_f64_lossy();
                let im = d.imag_part().to_f64_lossy();
                let m = re.abs().max(im.abs());
                if m == 0.0 {
                    any_zero = true;
                    break;
                }
                if !m.is_finite() {
                    acc_sign = T::nan_scalar();
                    acc_log += f64::INFINITY;
                } else {
                    let scale_unit = d.div_real(T::Real::from_f64_lossy(m));
                    let sr = scale_unit.real_part().to_f64_lossy();
                    let si = scale_unit.imag_part().to_f64_lossy();
                    let smag = sr.hypot(si);
                    let unit = scale_unit.div_real(T::Real::from_f64_lossy(smag));
                    acc_sign = acc_sign.mul_val(unit);
                    acc_log += m.ln() + smag.ln();
                }
            } else {
                let val = d.real_part().to_f64_lossy();
                if val == 0.0 {
                    any_zero = true;
                    break;
                }
                if val < 0.0 {
                    acc_sign = acc_sign.negate_val();
                }
                acc_log += val.abs().ln();
            }
        }

        if diag_nan {
            *dst_sign = if T::is_complex_type() {
                T::nan_scalar()
            } else {
                T::ONE
            };
            *dst_log = T::Real::nan_val();
        } else if any_zero {
            *dst_sign = T::ZERO;
            *dst_log = T::Real::neg_inf();
        } else {
            if T::is_complex_type() {
                let re = acc_sign.real_part().to_f64_lossy();
                let im = acc_sign.imag_part().to_f64_lossy();
                let mag = re.hypot(im);
                if mag > 0.0 && mag.is_finite() {
                    acc_sign = acc_sign.div_real(T::Real::from_f64_lossy(mag));
                }
            }
            *dst_sign = acc_sign;
            *dst_log = T::Real::from_f64_lossy(acc_log);
        }
    };

    let work = n.saturating_mul(n).saturating_mul(n);
    if bcount == 1 {
        let par = single_mat_par(n * n);
        let mut ws = make_ws(par);
        run_one(
            0,
            &mut sign_buf.as_mut_slice()[0],
            &mut log_buf.as_mut_slice()[0],
            par,
            &mut ws,
        );
    } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
        sign_buf
            .as_mut_slice()
            .par_iter_mut()
            .zip(log_buf.as_mut_slice().par_iter_mut())
            .enumerate()
            .for_each_init(
                || make_ws(Par::Seq),
                |ws, (i, (ds, dl))| run_one(i, ds, dl, Par::Seq, ws),
            );
    } else {
        let mut ws = make_ws(Par::Seq);
        for (i, (ds, dl)) in sign_buf
            .as_mut_slice()
            .iter_mut()
            .zip(log_buf.as_mut_slice().iter_mut())
            .enumerate()
        {
            run_one(i, ds, dl, Par::Seq, &mut ws);
        }
    }

    Ok(SlogdetOutput {
        sign: ndarray_from_aligned(sign_buf, &batch_shape)?,
        logabsdet: ndarray_from_aligned(log_buf, &batch_shape)?,
    })
}

/// Dynamic `slogdet`.
pub fn slogdet_dyn(a: &DynArray) -> Result<SlogdetResult> {
    dispatch_linalg_dyn!(a, va => {
        let out = slogdet(va)?;
        Ok(SlogdetResult {
            sign: LinalgScalar::into_dyn(out.sign),
            logabsdet: LinalgReal::real_into_dyn(out.logabsdet),
        })
    })
}

/// Determinant of stacked square matrices `[..., N, N]`, matching `numpy.linalg.det`.
pub fn det<T: LinalgScalar>(a: &NdArray<T>) -> Result<NdArray<T>> {
    let SlogdetOutput { sign, logabsdet } = slogdet(a)?;
    let s_slice = sign.as_slice().unwrap_or(&[]);
    let l_slice = logabsdet.as_slice().unwrap_or(&[]);
    let mut out = alloc_zeroed_buf::<T>(s_slice.len());
    for ((dst, &sg), &lg) in out
        .as_mut_slice()
        .iter_mut()
        .zip(s_slice.iter())
        .zip(l_slice.iter())
    {
        if lg.is_nan_val() {
            *dst = T::nan_scalar();
        } else if sg == T::ZERO {
            *dst = T::ZERO;
        } else {
            *dst = sg.mul_val(T::from_real_val(lg.exp_val()));
        }
    }
    ndarray_from_aligned(out, sign.shape())
}

/// Dynamic `det`.
pub fn det_dyn(a: &DynArray) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgScalar::into_dyn(det(va)?)))
}

/// Matrix numerical rank via SVD (or Hermitian EVD), matching `numpy.linalg.matrix_rank`.
///
/// Note: In NumPy, inputs with `ndim < 2` return a Python `int` (`int(not all(a == 0))`).
/// In Rust's array-typed API, this returns a 0-D `NdArray<i64>` (`[]` shape) with `0` or `1` (`F-B2-07`).
pub fn matrix_rank<T: LinalgScalar>(
    a: &NdArray<T>,
    tol: Option<f64>,
    rtol: Option<f64>,
    hermitian: bool,
) -> Result<NdArray<i64>> {
    if tol.is_some() && rtol.is_some() {
        return Err(NumPyError::InvalidValue(
            "tol and rtol cannot both be specified".into(),
        ));
    }
    if a.ndim() < 2 {
        let nonzero = a.iter().any(|v| v != T::ZERO);
        return Ok(NdArray::from_scalar(if nonzero { 1 } else { 0 }));
    }
    let (batch_shape_ref, m, n) = parse_stacked_2d(a)?;
    if hermitian && m != n {
        return Err(NumPyError::LinAlgError(format!(
            "Last 2 dimensions of the array must be square when hermitian=True, got ({m}, {n})"
        )));
    }
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    if bcount == 0 {
        return Ok(NdArray::zeros(&batch_shape)?);
    }
    let k = m.min(n);
    if k == 0 {
        return Ok(NdArray::zeros(&batch_shape)?);
    }

    let s = if hermitian {
        let evals = eigvalsh(a, Uplo::Lower)?;
        let mut s_buf = alloc_zeroed_buf::<T::Real>(bcount * k);
        let src = evals.to_vec();
        for (src_chunk, dst_chunk) in src
            .chunks_exact(k)
            .zip(s_buf.as_mut_slice().chunks_exact_mut(k))
        {
            let mut abs_vals: Vec<T::Real> = src_chunk.iter().map(|v| v.abs_val()).collect();
            abs_vals.sort_by(|x, y| {
                y.to_f64_lossy()
                    .partial_cmp(&x.to_f64_lossy())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            dst_chunk.copy_from_slice(&abs_vals);
        }
        ndarray_from_aligned(s_buf, &[bcount, k])?
    } else {
        svdvals(a)?
    };

    let s_slice = s.as_slice().expect("s contiguous");
    let mut out = alloc_zeroed_buf::<i64>(bcount);
    let rtol_eff = match (tol, rtol) {
        (Some(_), _) => 0.0,
        (None, Some(r)) => r,
        (None, None) => (m.max(n) as f64) * T::Real::eps().to_f64_lossy(),
    };

    for (dst, s_b) in out.as_mut_slice().iter_mut().zip(s_slice.chunks_exact(k)) {
        let smax = s_b[0].to_f64_lossy();
        let cutoff = match tol {
            Some(t) => t,
            None => smax * rtol_eff,
        };
        let count = s_b.iter().filter(|&&v| v.to_f64_lossy() > cutoff).count();
        *dst = count as i64;
    }

    ndarray_from_aligned(out, &batch_shape)
}

/// Dynamic `matrix_rank`.
pub fn matrix_rank_dyn(
    a: &DynArray,
    tol: Option<f64>,
    rtol: Option<f64>,
    hermitian: bool,
) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(DynArray::I64(matrix_rank(va, tol, rtol, hermitian)?)))
}

#[inline]
fn eval_vector_norm_1d<T: NormScalar, F>(len: usize, mut get: F, ord: f64) -> Result<T::NormReal>
where
    F: FnMut(usize) -> T,
{
    if ord.is_nan() {
        return Ok(T::NormReal::NAN_R);
    }
    if len == 0 {
        return if ord == f64::NEG_INFINITY {
            Err(NumPyError::InvalidValue(
                "zero-size array to reduction operation minimum which has no identity".into(),
            ))
        } else if ord < 0.0 {
            Ok(T::NormReal::POS_INF_R)
        } else {
            Ok(T::NormReal::ZERO_R)
        };
    }

    if ord == 0.0 {
        let mut cnt = 0usize;
        for i in 0..len {
            if !get(i).is_zero_n() {
                cnt += 1;
            }
        }
        return Ok(T::NormReal::from_f64_r(cnt as f64));
    }

    if ord == f64::INFINITY {
        let mut mx = T::NormReal::ZERO_R;
        for i in 0..len {
            let v = get(i);
            if v.is_nan_n() {
                return Ok(T::NormReal::NAN_R);
            }
            let m = v.abs_n();
            if m.gt_r(mx) {
                mx = m;
            }
        }
        return Ok(mx);
    }

    if ord == f64::NEG_INFINITY {
        let mut mn = T::NormReal::POS_INF_R;
        for i in 0..len {
            let v = get(i);
            if v.is_nan_n() {
                return Ok(T::NormReal::NAN_R);
            }
            let m = v.abs_n();
            if m.lt_r(mn) {
                mn = m;
            }
        }
        return Ok(mn);
    }

    // FIX(F-B2-02): unscaled accumulation in input real dtype matches NumPy overflow/underflow semantics.
    if ord == 1.0 {
        let mut sum = T::NormReal::ZERO_R;
        for i in 0..len {
            let v = get(i);
            if v.is_nan_n() {
                return Ok(T::NormReal::NAN_R);
            }
            sum = sum.add_r(v.abs_n());
        }
        return Ok(sum);
    }

    if ord == 2.0 {
        let mut ssq = T::NormReal::ZERO_R;
        for i in 0..len {
            let v = get(i);
            if v.is_nan_n() {
                return Ok(T::NormReal::NAN_R);
            }
            let a = v.abs_n();
            ssq = ssq.add_r(a.mul_r(a));
        }
        return Ok(ssq.sqrt_r());
    }

    // General finite nonzero real power `ord`: accumulate sum(|x|^ord) then raise to 1/ord in T::NormReal.
    let mut acc = T::NormReal::ZERO_R;
    for i in 0..len {
        let v = get(i);
        if v.is_nan_n() {
            return Ok(T::NormReal::NAN_R);
        }
        acc = acc.add_r(v.abs_n().powf_r(ord));
    }
    Ok(acc.powf_r(1.0 / ord))
}

#[inline]
fn frobenius_norm_strided<T: NormScalar>(
    src_ptr: *const T,
    m: usize,
    n: usize,
    rsa: isize,
    csa: isize,
) -> T::NormReal {
    if m == 0 || n == 0 {
        return T::NormReal::ZERO_R;
    }
    // FIX(F-B2-02): unscaled Frobenius accumulation in input real dtype matches NumPy overflow/underflow.
    let mut ssq = T::NormReal::ZERO_R;
    for i in 0..m {
        let row_ptr = unsafe { src_ptr.offset(i as isize * rsa) };
        for j in 0..n {
            let v = unsafe { *row_ptr.offset(j as isize * csa) };
            if v.is_nan_n() {
                return T::NormReal::NAN_R;
            }
            let a = v.abs_n();
            ssq = ssq.add_r(a.mul_r(a));
        }
    }
    ssq.sqrt_r()
}

fn vector_norm_impl<T: NormScalar>(
    a: &NdArray<T>,
    ord: f64,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<NdArray<T::NormReal>> {
    let nd = a.ndim();
    let norm_axes: Vec<usize> = match axis {
        None => (0..nd).collect(),
        Some(raw) => {
            let mut seen = HashSet::with_capacity(raw.len());
            let mut res = Vec::with_capacity(raw.len());
            for &ax in raw {
                let nax = normalize_axis(ax, nd)?;
                if !seen.insert(nax) {
                    return Err(NumPyError::InvalidValue(format!(
                        "duplicate value {nax} in axis"
                    )));
                }
                res.push(nax);
            }
            res
        }
    };

    let mut out_shape = Vec::with_capacity(nd);
    for d in 0..nd {
        if norm_axes.contains(&d) {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(a.shape()[d]);
        }
    }

    // Empty `axis=Some(&[])` case: elementwise |x| (or 0/1 for ord=0) with original shape!
    if norm_axes.is_empty() {
        let a_c = a.to_c_contiguous();
        let src = a_c.as_slice().unwrap_or(&[]);
        let mut out = alloc_zeroed_buf::<T::NormReal>(src.len());
        for (dst, &v) in out.as_mut_slice().iter_mut().zip(src.iter()) {
            *dst = eval_vector_norm_1d(1, |_| v, ord)?;
        }
        return ndarray_from_aligned(out, &out_shape);
    }

    // Full reduction (`axis=None` or all axes reduced): fast contiguous slice path!
    if norm_axes.len() == nd {
        if let Some(sl) = a.as_slice() {
            let val = eval_vector_norm_1d(sl.len(), |i| sl[i], ord)?;
            let mut out = alloc_zeroed_buf::<T::NormReal>(1);
            out.as_mut_slice()[0] = val;
            return ndarray_from_aligned(out, &out_shape);
        }
        let a_c = a.to_c_contiguous();
        let sl = a_c.as_slice().unwrap_or(&[]);
        let val = eval_vector_norm_1d(sl.len(), |i| sl[i], ord)?;
        let mut out = alloc_zeroed_buf::<T::NormReal>(1);
        out.as_mut_slice()[0] = val;
        return ndarray_from_aligned(out, &out_shape);
    }

    // Move reduced axes to the end, keep remaining axes at front, convert to contiguous `(out_count, vec_len)`.
    let kept: Vec<usize> = (0..nd).filter(|d| !norm_axes.contains(d)).collect();
    let mut perm: Vec<isize> = kept.iter().map(|&x| x as isize).collect();
    perm.extend(norm_axes.iter().map(|&x| x as isize));

    let moved = a.transpose(Some(&perm))?.to_c_contiguous();
    let out_count: usize = kept.iter().map(|&d| a.shape()[d]).product();
    let vec_len: usize = norm_axes.iter().map(|&d| a.shape()[d]).product();

    let mut out = alloc_zeroed_buf::<T::NormReal>(out_count);
    if out_count == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    if vec_len == 0 {
        let empty_val = eval_vector_norm_1d::<T, _>(0, |_| T::ZERO_N, ord)?;
        out.as_mut_slice().fill(empty_val);
        return ndarray_from_aligned(out, &out_shape);
    }

    let src = moved.as_slice().expect("to_c_contiguous slice");
    if out_count * vec_len >= NORM_RAYON_THRESHOLD && out_count > 1 {
        out.as_mut_slice()
            .par_iter_mut()
            .zip(src.par_chunks_exact(vec_len))
            .try_for_each(|(dst, chunk)| -> Result<()> {
                *dst = eval_vector_norm_1d(vec_len, |i| chunk[i], ord)?;
                Ok(())
            })?;
    } else {
        for (dst, chunk) in out.as_mut_slice().iter_mut().zip(src.chunks_exact(vec_len)) {
            *dst = eval_vector_norm_1d(vec_len, |i| chunk[i], ord)?;
        }
    }
    ndarray_from_aligned(out, &out_shape)
}

/// Vector norm matching `numpy.linalg.vector_norm`.
pub fn vector_norm<T: LinalgScalar>(
    a: &NdArray<T>,
    ord: f64,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<NdArray<T::Real>> {
    vector_norm_impl(a, ord, axis, keepdims)
}

/// Dynamic `vector_norm`. Preserves `float16`, `float32`, `float64` real output dtypes (`FIX(F-B2-05)`).
pub fn vector_norm_dyn(
    a: &DynArray,
    ord: f64,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<DynArray> {
    if let DynArray::F16(va) = a {
        return Ok(DynArray::F16(vector_norm_impl(va, ord, axis, keepdims)?));
    }
    dispatch_linalg_dyn!(a, va => Ok(LinalgReal::real_into_dyn(vector_norm(va, ord, axis, keepdims)?)))
}

fn matrix_norm_core_impl<T: NormScalar>(
    a: &NdArray<T>,
    ord: NormOrd,
    row_axis: isize,
    col_axis: isize,
    keepdims: bool,
) -> Result<NdArray<T::NormReal>> {
    let nd = a.ndim();
    if nd < 2 {
        return Err(NumPyError::LinAlgError(format!(
            "matrix_norm requires at least 2 dimensions, got {nd}"
        )));
    }
    let ax0 = normalize_axis(row_axis, nd)?;
    let ax1 = normalize_axis(col_axis, nd)?;
    if ax0 == ax1 {
        return Err(NumPyError::InvalidValue(
            "Duplicate axes given for matrix norm".into(),
        ));
    }

    // Move `(ax0, ax1)` to trailing `(-2, -1)` preserving relative order of remaining axes.
    let kept: Vec<usize> = (0..nd).filter(|&d| d != ax0 && d != ax1).collect();
    let mut perm: Vec<isize> = kept.iter().map(|&d| d as isize).collect();
    perm.push(ax0 as isize);
    perm.push(ax1 as isize);
    let moved = a.transpose(Some(&perm))?;

    let mut out_shape = Vec::with_capacity(nd);
    for d in 0..nd {
        if d == ax0 || d == ax1 {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(a.shape()[d]);
        }
    }

    let (batch_shape_ref, m, n) = parse_stacked_2d(&moved)?;
    if matches!(ord, NormOrd::Zero | NormOrd::P(_)) {
        return Err(NumPyError::InvalidValue(
            "Invalid norm order for matrices.".into(),
        ));
    }
    if (ord == NormOrd::LNeg2 && m.min(n) == 0)
        || (ord == NormOrd::NegInf && m == 0)
        || (ord == NormOrd::LNeg1 && n == 0)
    {
        return Err(NumPyError::InvalidValue(
            "zero-size array to reduction operation minimum which has no identity".into(),
        ));
    }
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    if bcount == 0 {
        if !T::supports_svd_n() && matches!(ord, NormOrd::Nuc | NormOrd::L2 | NormOrd::LNeg2) {
            return Err(NumPyError::DTypeError(
                "array type float16 is unsupported in linalg".into(),
            ));
        }
        return Ok(NdArray::zeros(&out_shape)?);
    }

    match ord {
        NormOrd::Default | NormOrd::Fro => {
            let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(&moved);
            let mut out = alloc_zeroed_buf::<T::NormReal>(bcount);
            let work = m.saturating_mul(n);
            if should_par_batches(bcount, work, NORM_RAYON_THRESHOLD) {
                out.as_mut_slice()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, dst)| {
                        let oa = linear_strided_offset(i, &batch_shape, bstr_a);
                        *dst = frobenius_norm_strided(unsafe { a_base.offset(oa) }, m, n, rsa, csa);
                    });
            } else {
                for (i, dst) in out.as_mut_slice().iter_mut().enumerate() {
                    let oa = linear_strided_offset(i, &batch_shape, bstr_a);
                    *dst = frobenius_norm_strided(unsafe { a_base.offset(oa) }, m, n, rsa, csa);
                }
            }
            ndarray_from_aligned(out, &out_shape)
        }
        NormOrd::Nuc | NormOrd::L2 | NormOrd::LNeg2 => {
            let k = m.min(n);
            if k == 0 {
                if !T::supports_svd_n() {
                    return Err(NumPyError::DTypeError(
                        "array type float16 is unsupported in linalg".into(),
                    ));
                }
                if ord == NormOrd::LNeg2 {
                    return Err(NumPyError::InvalidValue(
                        "zero-size array to reduction operation minimum which has no identity"
                            .into(),
                    ));
                }
                return Ok(NdArray::zeros(&out_shape)?);
            }
            let s = T::svdvals_for_norm(&moved)?;
            let s_slice = s.as_slice().expect("s contiguous");
            let mut out = alloc_zeroed_buf::<T::NormReal>(bcount);
            for (dst, s_b) in out.as_mut_slice().iter_mut().zip(s_slice.chunks_exact(k)) {
                *dst = match ord {
                    NormOrd::Nuc => s_b.iter().copied().fold(T::NormReal::ZERO_R, |a, b| a.add_r(b)),
                    NormOrd::L2 => s_b[0],
                    NormOrd::LNeg2 => s_b[k - 1],
                    _ => unreachable!(),
                };
            }
            ndarray_from_aligned(out, &out_shape)
        }
        NormOrd::PosInf | NormOrd::NegInf => {
            if m == 0 {
                return Ok(NdArray::zeros(&out_shape)?);
            }
            let is_max = ord == NormOrd::PosInf;
            let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(&moved);
            let mut out = alloc_zeroed_buf::<T::NormReal>(bcount);
            for (idx, dst) in out.as_mut_slice().iter_mut().enumerate() {
                let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
                let src_ptr = unsafe { a_base.offset(oa) };
                let mut best = if is_max {
                    T::NormReal::ZERO_R
                } else {
                    T::NormReal::POS_INF_R
                };
                let mut nan_seen = false;
                for i in 0..m {
                    let row_ptr = unsafe { src_ptr.offset(i as isize * rsa) };
                    let mut row_sum = T::NormReal::ZERO_R;
                    for j in 0..n {
                        let v = unsafe { *row_ptr.offset(j as isize * csa) };
                        if v.is_nan_n() {
                            nan_seen = true;
                            break;
                        }
                        row_sum = row_sum.add_r(v.abs_n());
                    }
                    if nan_seen {
                        break;
                    }
                    if is_max {
                        if row_sum.gt_r(best) {
                            best = row_sum;
                        }
                    } else if row_sum.lt_r(best) {
                        best = row_sum;
                    }
                }
                *dst = if nan_seen {
                    T::NormReal::NAN_R
                } else {
                    best
                };
            }
            ndarray_from_aligned(out, &out_shape)
        }
        NormOrd::L1 | NormOrd::LNeg1 => {
            if n == 0 {
                return Ok(NdArray::zeros(&out_shape)?);
            }
            let is_max = ord == NormOrd::L1;
            let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(&moved);
            let mut out = alloc_zeroed_buf::<T::NormReal>(bcount);
            for (idx, dst) in out.as_mut_slice().iter_mut().enumerate() {
                let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
                let src_ptr = unsafe { a_base.offset(oa) };
                let mut best = if is_max {
                    T::NormReal::ZERO_R
                } else {
                    T::NormReal::POS_INF_R
                };
                let mut nan_seen = false;
                for j in 0..n {
                    let col_ptr = unsafe { src_ptr.offset(j as isize * csa) };
                    let mut col_sum = T::NormReal::ZERO_R;
                    for i in 0..m {
                        let v = unsafe { *col_ptr.offset(i as isize * rsa) };
                        if v.is_nan_n() {
                            nan_seen = true;
                            break;
                        }
                        col_sum = col_sum.add_r(v.abs_n());
                    }
                    if nan_seen {
                        break;
                    }
                    if is_max {
                        if col_sum.gt_r(best) {
                            best = col_sum;
                        }
                    } else if col_sum.lt_r(best) {
                        best = col_sum;
                    }
                }
                *dst = if nan_seen {
                    T::NormReal::NAN_R
                } else {
                    best
                };
            }
            ndarray_from_aligned(out, &out_shape)
        }
        NormOrd::Zero | NormOrd::P(_) => Err(NumPyError::InvalidValue(
            "Invalid norm order for matrices".into(),
        )),
    }
}

fn matrix_norm_core<T: LinalgScalar>(
    a: &NdArray<T>,
    ord: NormOrd,
    row_axis: isize,
    col_axis: isize,
    keepdims: bool,
) -> Result<NdArray<T::Real>> {
    matrix_norm_core_impl(a, ord, row_axis, col_axis, keepdims)
}

/// Matrix norm over last two axes `(-2, -1)`, matching `numpy.linalg.matrix_norm`.
pub fn matrix_norm<T: LinalgScalar>(
    a: &NdArray<T>,
    ord: NormOrd,
    keepdims: bool,
) -> Result<NdArray<T::Real>> {
    matrix_norm_core(a, ord, -2, -1, keepdims)
}

/// Dynamic `matrix_norm`. Preserves `float16` for Frobenius/1/-1/inf/-inf and rejects `nuc`/2/-2 (`FIX(F-B2-05)`).
pub fn matrix_norm_dyn(a: &DynArray, ord: NormOrd, keepdims: bool) -> Result<DynArray> {
    if let DynArray::F16(va) = a {
        return Ok(DynArray::F16(matrix_norm_core_impl(va, ord, -2, -1, keepdims)?));
    }
    dispatch_linalg_dyn!(a, va => Ok(LinalgReal::real_into_dyn(matrix_norm(va, ord, keepdims)?)))
}

fn norm_impl<T: NormScalar>(
    a: &NdArray<T>,
    ord: NormOrd,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<NdArray<T::NormReal>> {
    let vec_ord_to_f64 = |o: NormOrd| -> Result<f64> {
        match o {
            NormOrd::Default | NormOrd::L2 => Ok(2.0),
            NormOrd::LNeg2 => Ok(-2.0),
            NormOrd::L1 => Ok(1.0),
            NormOrd::LNeg1 => Ok(-1.0),
            NormOrd::PosInf => Ok(f64::INFINITY),
            NormOrd::NegInf => Ok(f64::NEG_INFINITY),
            NormOrd::Zero => Ok(0.0),
            NormOrd::P(p) => Ok(p),
            NormOrd::Fro | NormOrd::Nuc => Err(NumPyError::InvalidValue(
                "Invalid norm order for vectors".into(),
            )),
        }
    };

    match axis {
        None => {
            if a.ndim() == 1 {
                let p = vec_ord_to_f64(ord)?;
                vector_norm_impl(a, p, Some(&[0]), keepdims)
            } else if a.ndim() == 2 {
                matrix_norm_core_impl(a, ord, 0, 1, keepdims)
            } else if ord == NormOrd::Default {
                vector_norm_impl(a, 2.0, None, keepdims)
            } else {
                Err(NumPyError::InvalidValue(
                    "Improper number of dimensions to norm".into(),
                ))
            }
        }
        Some(axes) => match axes.len() {
            1 => {
                let p = vec_ord_to_f64(ord)?;
                vector_norm_impl(a, p, Some(axes), keepdims)
            }
            2 => matrix_norm_core_impl(a, ord, axes[0], axes[1], keepdims),
            _ => Err(NumPyError::InvalidValue(
                "Improper number of dimensions to norm".into(),
            )),
        },
    }
}

/// General matrix or vector norm matching `numpy.linalg.norm`.
pub fn norm<T: LinalgScalar>(
    a: &NdArray<T>,
    ord: NormOrd,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<NdArray<T::Real>> {
    norm_impl(a, ord, axis, keepdims)
}

/// Dynamic `norm`. Preserves `float16` output dtype matching NumPy (`FIX(F-B2-05)`).
pub fn norm_dyn(
    a: &DynArray,
    ord: NormOrd,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<DynArray> {
    if let DynArray::F16(va) = a {
        return Ok(DynArray::F16(norm_impl(va, ord, axis, keepdims)?));
    }
    dispatch_linalg_dyn!(a, va => Ok(LinalgReal::real_into_dyn(norm(va, ord, axis, keepdims)?)))
}

/// Condition number of stacked matrices, matching `numpy.linalg.cond`.
pub fn cond<T: LinalgScalar>(a: &NdArray<T>, p: NormOrd) -> Result<NdArray<T::Real>> {
    let (batch_shape_ref, m, n) = parse_stacked_2d(a)?;
    if m == 0 || n == 0 {
        return Err(NumPyError::LinAlgError(
            "cond is not defined on empty arrays".into(),
        ));
    }
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);

    match p {
        NormOrd::Default | NormOrd::L2 | NormOrd::LNeg2 => {
            if bcount == 0 {
                return Ok(NdArray::zeros(&batch_shape)?);
            }
            let k = m.min(n);
            let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
            let mut out = alloc_zeroed_buf::<T::Real>(bcount);

            for (idx, dst) in out.as_mut_slice().iter_mut().enumerate() {
                let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
                let src_ptr = unsafe { a_base.offset(oa) };
                if unsafe { mat_any_nan(src_ptr, m, n, rsa, csa) } {
                    *dst = T::Real::nan_val();
                    continue;
                }
                // Extract 2-D matrix view and call svdvals cleanly.
                let mut mat_c = alloc_zeroed_buf::<T>(m * n);
                unsafe {
                    crate::batch::copy_strided_mat_to_c(
                        src_ptr,
                        m,
                        n,
                        rsa,
                        csa,
                        mat_c.as_mut_slice(),
                    );
                }
                let mat = ndarray_from_aligned(mat_c, &[m, n])?;
                match svdvals(&mat) {
                    Ok(sv) => {
                        let sl = sv.as_slice().expect("sv contiguous");
                        let s_max = sl[0].to_f64_lossy();
                        let s_min = sl[k - 1].to_f64_lossy();
                        let ratio = if p == NormOrd::LNeg2 {
                            s_min / s_max
                        } else {
                            s_max / s_min
                        };
                        *dst = if ratio.is_nan() {
                            T::Real::pos_inf()
                        } else {
                            T::Real::from_f64_lossy(ratio)
                        };
                    }
                    Err(_) => {
                        // Input had Inf (no NaN was checked above).
                        *dst = T::Real::pos_inf();
                    }
                }
            }
            ndarray_from_aligned(out, &batch_shape)
        }
        NormOrd::Fro | NormOrd::Nuc | NormOrd::PosInf | NormOrd::NegInf | NormOrd::L1 | NormOrd::LNeg1 => {
            if m != n {
                return Err(NumPyError::LinAlgError(format!(
                    "Last 2 dimensions of the array must be square for p={p:?}, got ({m}, {n})"
                )));
            }
            if bcount == 0 {
                return Ok(NdArray::zeros(&batch_shape)?);
            }
            let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
            let mut out = alloc_zeroed_buf::<T::Real>(bcount);

            for (idx, dst) in out.as_mut_slice().iter_mut().enumerate() {
                let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
                let src_ptr = unsafe { a_base.offset(oa) };
                if unsafe { mat_any_nan(src_ptr, n, n, rsa, csa) } {
                    *dst = T::Real::nan_val();
                    continue;
                }
                let mut mat_c = alloc_zeroed_buf::<T>(n * n);
                unsafe {
                    crate::batch::copy_strided_mat_to_c(
                        src_ptr,
                        n,
                        n,
                        rsa,
                        csa,
                        mat_c.as_mut_slice(),
                    );
                }
                let mat = ndarray_from_aligned(mat_c, &[n, n])?;
                let norm_a = matrix_norm(&mat, p, false)?;
                let na_val = norm_a.as_slice().unwrap()[0].to_f64_lossy();
                match inv(&mat) {
                    Ok(inv_m) => {
                        let norm_inv = matrix_norm(&inv_m, p, false)?;
                        let ninv_val = norm_inv.as_slice().unwrap()[0].to_f64_lossy();
                        let prod = na_val * ninv_val;
                        *dst = if prod.is_nan() {
                            T::Real::pos_inf()
                        } else {
                            T::Real::from_f64_lossy(prod)
                        };
                    }
                    Err(_) => {
                        *dst = T::Real::pos_inf();
                    }
                }
            }
            ndarray_from_aligned(out, &batch_shape)
        }
        NormOrd::Zero | NormOrd::P(_) => Err(NumPyError::InvalidValue(
            "Invalid norm order for condition number".into(),
        )),
    }
}

/// Dynamic `cond`.
pub fn cond_dyn(a: &DynArray, p: NormOrd) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgReal::real_into_dyn(cond(va, p)?)))
}

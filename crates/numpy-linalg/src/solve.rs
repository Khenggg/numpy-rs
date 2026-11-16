//! Linear systems and inverses: solve, tensorsolve, inv, tensorinv, pinv, lstsq, matrix_power.

use std::collections::HashSet;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, Par};
use numpy_types::{DynArray, NdArray, NumPyError, Result, Scalar, normalize_axis, shape_size};
use rayon::prelude::*;

use crate::batch::{
    DECOMP_RAYON_THRESHOLD, SyncPtr, alloc_zeroed_buf, batch_len, broadcast_batch_dims,
    copy_strided_to_faer_mat, linear_strided_offset, mat_all_finite, matrix_stack_descriptor,
    ndarray_from_aligned, parse_stacked_2d, parse_stacked_square, should_par_batches,
    single_mat_par,
};
use crate::decomp::svd;
use crate::products::matrix_power_nonneg;
use crate::traits::{LinalgReal, LinalgScalar, LstsqOutput, LstsqResult, ProductScalar};
use crate::{dispatch_linalg_dyn, dispatch_linalg_pair_dyn, dispatch_product_dyn};

/// Solves stacked linear systems `A @ X = B` where `A` has shape `(..., M, M)` and
/// `B` has shape `(M,)` (only if 1-D) or `(..., M, K)`, matching NumPy 2.0 `numpy.linalg.solve`.
pub fn solve<T: LinalgScalar>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let (batch_a_ref, m) = parse_stacked_square(a)?;
    let batch_a = batch_a_ref.to_vec();
    if b.ndim() == 0 {
        return Err(NumPyError::LinAlgError(
            "0-dimensional array given as RHS of solve; must be at least 1-dimensional".into(),
        ));
    }

    let b_is_1d = b.ndim() == 1;
    let (k, rsb, csb, batch_b_ref, bstr_b_ref) = if b_is_1d {
        if b.shape()[0] != m {
            return Err(NumPyError::ShapeMismatch(format!(
                "solve: Input operand 1 core dimension mismatch ({} != {m})",
                b.shape()[0]
            )));
        }
        (1usize, b.strides()[0], 0isize, &[][..], &[][..])
    } else {
        let ndb = b.ndim();
        if b.shape()[ndb - 2] != m {
            return Err(NumPyError::ShapeMismatch(format!(
                "solve: Input operand 1 core dimension mismatch ({} != {m})",
                b.shape()[ndb - 2]
            )));
        }
        (
            b.shape()[ndb - 1],
            b.strides()[ndb - 2],
            b.strides()[ndb - 1],
            &b.shape()[..ndb - 2],
            &b.strides()[..ndb - 2],
        )
    };

    let (a_base, rsa, csa, _, bstr_a_ref) = matrix_stack_descriptor(a);
    let b_base = SyncPtr(unsafe { b.storage_ptr().add(b.offset()) });

    let (out_batch, pad_sa, pad_sb) = if b_is_1d {
        (
            batch_a.clone(),
            bstr_a_ref.to_vec(),
            vec![0isize; batch_a.len()],
        )
    } else {
        broadcast_batch_dims(&batch_a, bstr_a_ref, batch_b_ref, bstr_b_ref)?
    };

    let mut out_shape = out_batch.clone();
    out_shape.push(m);
    if !b_is_1d {
        out_shape.push(k);
    }

    let bcount = batch_len(&out_batch);
    let rhs_len = m.saturating_mul(k);
    let mut out = alloc_zeroed_buf::<T>(bcount.saturating_mul(rhs_len));
    if bcount == 0 || m == 0 || k == 0 {
        return ndarray_from_aligned(out, &out_shape);
    }

    struct SolveWs<T: LinalgScalar> {
        lu_mat: Mat<T>,
        rhs_mat: Mat<T>,
        perm_fwd: Vec<usize>,
        perm_bwd: Vec<usize>,
        mem: MemBuffer,
    }

    let make_ws = |par: Par| -> SolveWs<T> {
        let req_lu = T::faer_lu_scratch(m, m, par);
        let req_solve = T::faer_solve_scratch(m, k, par);
        SolveWs {
            lu_mat: Mat::full(m, m, T::ZERO),
            rhs_mat: Mat::full(m, k, T::ZERO),
            perm_fwd: vec![0; m],
            perm_bwd: vec![0; m],
            mem: MemBuffer::new(req_lu.or(req_solve)),
        }
    };

    let run_one = |idx: usize, dst: &mut [T], par: Par, ws: &mut SolveWs<T>| -> Result<()> {
        let oa = linear_strided_offset(idx, &out_batch, &pad_sa);
        let ob = linear_strided_offset(idx, &out_batch, &pad_sb);
        unsafe {
            copy_strided_to_faer_mat(a_base.offset(oa), m, m, rsa, csa, &mut ws.lu_mat);
            copy_strided_to_faer_mat(b_base.offset(ob), m, k, rsb, csb, &mut ws.rhs_mat);
        }

        let (_, row_perm) = T::faer_lu_in_place(
            ws.lu_mat.as_mut(),
            &mut ws.perm_fwd,
            &mut ws.perm_bwd,
            par,
            MemStack::new(&mut ws.mem),
        );
        for i in 0..m {
            if ws.lu_mat[(i, i)] == T::ZERO {
                return Err(NumPyError::LinAlgError("Singular matrix".into()));
            }
        }
        T::faer_solve_in_place(
            ws.lu_mat.as_ref(),
            row_perm,
            ws.rhs_mat.as_mut(),
            par,
            MemStack::new(&mut ws.mem),
        );
        for r in 0..m {
            for c in 0..k {
                dst[r * k + c] = ws.rhs_mat[(r, c)];
            }
        }
        Ok(())
    };

    let work = m.saturating_mul(m).saturating_mul(m);
    if bcount == 1 {
        let par = single_mat_par(m * m);
        let mut ws = make_ws(par);
        run_one(0, out.as_mut_slice(), par, &mut ws)?;
    } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
        out.as_mut_slice()
            .par_chunks_mut(rhs_len)
            .enumerate()
            .try_for_each_init(
                || make_ws(Par::Seq),
                |ws, (i, dst)| run_one(i, dst, Par::Seq, ws),
            )?;
    } else {
        let mut ws = make_ws(Par::Seq);
        for (i, dst) in out.as_mut_slice().chunks_exact_mut(rhs_len).enumerate() {
            run_one(i, dst, Par::Seq, &mut ws)?;
        }
    }

    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `solve`.
pub fn solve_dyn(a: &DynArray, b: &DynArray) -> Result<DynArray> {
    dispatch_linalg_pair_dyn!(a, b, (va, vb) => Ok(LinalgScalar::into_dyn(solve(va, vb)?)))
}

/// Multiplicative inverse over stacked square matrices `[..., N, N]`, matching `numpy.linalg.inv`.
pub fn inv<T: LinalgScalar>(a: &NdArray<T>) -> Result<NdArray<T>> {
    let (batch_shape_ref, n) = parse_stacked_square(a)?;
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    let mat_len = n.saturating_mul(n);
    let mut out = alloc_zeroed_buf::<T>(bcount.saturating_mul(mat_len));
    if bcount == 0 || n == 0 {
        return ndarray_from_aligned(out, a.shape());
    }

    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);

    struct InvWs<T: LinalgScalar> {
        lu_mat: Mat<T>,
        inv_mat: Mat<T>,
        perm_fwd: Vec<usize>,
        perm_bwd: Vec<usize>,
        mem: MemBuffer,
    }

    let make_ws = |par: Par| -> InvWs<T> {
        let req_lu = T::faer_lu_scratch(n, n, par);
        let req_inv = T::faer_inverse_scratch(n, par);
        InvWs {
            lu_mat: Mat::full(n, n, T::ZERO),
            inv_mat: Mat::full(n, n, T::ZERO),
            perm_fwd: vec![0; n],
            perm_bwd: vec![0; n],
            mem: MemBuffer::new(req_lu.or(req_inv)),
        }
    };

    let run_one = |idx: usize, dst: &mut [T], par: Par, ws: &mut InvWs<T>| -> Result<()> {
        let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
        unsafe {
            copy_strided_to_faer_mat(a_base.offset(oa), n, n, rsa, csa, &mut ws.lu_mat);
        }

        let (_, row_perm) = T::faer_lu_in_place(
            ws.lu_mat.as_mut(),
            &mut ws.perm_fwd,
            &mut ws.perm_bwd,
            par,
            MemStack::new(&mut ws.mem),
        );
        for i in 0..n {
            if ws.lu_mat[(i, i)] == T::ZERO {
                return Err(NumPyError::LinAlgError("Singular matrix".into()));
            }
        }
        T::faer_inverse(
            ws.inv_mat.as_mut(),
            ws.lu_mat.as_ref(),
            row_perm,
            par,
            MemStack::new(&mut ws.mem),
        );
        for r in 0..n {
            for c in 0..n {
                dst[r * n + c] = ws.inv_mat[(r, c)];
            }
        }
        Ok(())
    };

    let work = n.saturating_mul(n).saturating_mul(n);
    if bcount == 1 {
        let par = single_mat_par(n * n);
        let mut ws = make_ws(par);
        run_one(0, out.as_mut_slice(), par, &mut ws)?;
    } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
        out.as_mut_slice()
            .par_chunks_mut(mat_len)
            .enumerate()
            .try_for_each_init(
                || make_ws(Par::Seq),
                |ws, (i, dst)| run_one(i, dst, Par::Seq, ws),
            )?;
    } else {
        let mut ws = make_ws(Par::Seq);
        for (i, dst) in out.as_mut_slice().chunks_exact_mut(mat_len).enumerate() {
            run_one(i, dst, Par::Seq, &mut ws)?;
        }
    }

    ndarray_from_aligned(out, a.shape())
}

/// Dynamic `inv`.
pub fn inv_dyn(a: &DynArray) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgScalar::into_dyn(inv(va)?)))
}

/// Solves tensor equation `a x = b` (`tensordot(a, x, x.ndim) == b`), matching `numpy.linalg.tensorsolve`.
pub fn tensorsolve<T: LinalgScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    axes: Option<&[isize]>,
) -> Result<NdArray<T>> {
    let a_moved = if let Some(ax_list) = axes {
        let nda = a.ndim();
        let mut norm_ax = Vec::with_capacity(ax_list.len());
        let mut seen = HashSet::with_capacity(ax_list.len());
        for &ax in ax_list {
            let na = normalize_axis(ax, nda)?;
            if !seen.insert(na) {
                return Err(NumPyError::InvalidValue(format!(
                    "repeated axis {na} in tensorsolve"
                )));
            }
            norm_ax.push(na);
        }
        let mut perm: Vec<isize> = (0..nda)
            .filter(|k| !seen.contains(k))
            .map(|k| k as isize)
            .collect();
        for na in norm_ax {
            perm.push(na as isize);
        }
        a.transpose(Some(&perm))?
    } else {
        a.clone()
    };

    let q = b.ndim();
    if a_moved.ndim() < q || &a_moved.shape()[..q] != b.shape() {
        return Err(NumPyError::LinAlgError(format!(
            "Input arrays have incompatible shapes: a={:?}, b={:?}",
            a_moved.shape(),
            b.shape()
        )));
    }
    let outshape = a_moved.shape()[q..].to_vec();
    let n = shape_size(b.shape())?;
    let prod_out = shape_size(&outshape)?;
    if n != prod_out {
        return Err(NumPyError::LinAlgError(format!(
            "After reshaping, matrix must be square: ({n}, {prod_out})"
        )));
    }
    if n == 0 {
        return NdArray::zeros(&outshape);
    }

    let a_2d = a_moved.to_c_contiguous().reshape(&[n as isize, n as isize])?;
    let b_1d = b.to_c_contiguous().reshape(&[n as isize])?;
    let sol = solve(&a_2d, &b_1d)?;
    let target_dims: Vec<isize> = outshape.into_iter().map(|d| d as isize).collect();
    sol.reshape(&target_dims)
}

/// Dynamic `tensorsolve`.
pub fn tensorsolve_dyn(a: &DynArray, b: &DynArray, axes: Option<&[isize]>) -> Result<DynArray> {
    dispatch_linalg_pair_dyn!(a, b, (va, vb) => Ok(LinalgScalar::into_dyn(tensorsolve(va, vb, axes)?)))
}

/// Computes the "inverse" of an N-dimensional array, matching `numpy.linalg.tensorinv`.
pub fn tensorinv<T: LinalgScalar>(a: &NdArray<T>, ind: usize) -> Result<NdArray<T>> {
    if ind == 0 {
        return Err(NumPyError::InvalidValue(
            "Invalid ind argument: must be >= 1".into(),
        ));
    }
    let nd = a.ndim();
    if ind > nd {
        return Err(NumPyError::InvalidValue(format!(
            "Invalid ind={ind} for {nd}-dimensional array"
        )));
    }
    let oldshape = a.shape();
    let head = shape_size(&oldshape[..ind])?;
    let tail = shape_size(&oldshape[ind..])?;
    if head != tail {
        return Err(NumPyError::LinAlgError(format!(
            "Inconsistent shape for tensorinv: first {ind} axes product {head} != remaining product {tail}"
        )));
    }
    let mut invshape: Vec<isize> = oldshape[ind..].iter().map(|&d| d as isize).collect();
    invshape.extend(oldshape[..ind].iter().map(|&d| d as isize));

    let a_2d = a.to_c_contiguous().reshape(&[head as isize, tail as isize])?;
    let inv_2d = inv(&a_2d)?;
    inv_2d.reshape(&invshape)
}

/// Dynamic `tensorinv`.
pub fn tensorinv_dyn(a: &DynArray, ind: usize) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgScalar::into_dyn(tensorinv(va, ind)?)))
}

/// Moore-Penrose pseudoinverse via SVD (or `eigh` when `hermitian=true`), matching `numpy.linalg.pinv`.
/// `rcond` defaults to `1e-15` when `None`, matching NumPy.
pub fn pinv<T: LinalgScalar>(
    a: &NdArray<T>,
    rcond: Option<f64>,
    hermitian: bool,
) -> Result<NdArray<T>> {
    let (batch_shape_ref, m, n) = parse_stacked_2d(a)?;
    if hermitian && m != n {
        return Err(NumPyError::LinAlgError(format!(
            "Last 2 dimensions of the array must be square when hermitian=True, got ({m}, {n})"
        )));
    }
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    let mut out_shape = batch_shape;
    out_shape.push(n);
    out_shape.push(m);

    if bcount == 0 || m == 0 || n == 0 {
        return NdArray::zeros(&out_shape);
    }

    let rcond_val = rcond.unwrap_or(1e-15);
    let svd_out = svd(a, false, true, hermitian)?;
    let u_arr = svd_out.u.expect("svd compute_uv=true");
    let s_arr = svd_out.s;
    let vh_arr = svd_out.vh.expect("svd compute_uv=true");

    let k = m.min(n);
    let u_data = u_arr.as_slice().expect("svd u is C-contiguous");
    let s_data = s_arr.as_slice().expect("svd s is C-contiguous");
    let vh_data = vh_arr.as_slice().expect("svd vh is C-contiguous");

    let out_mat_len = n * m;
    let u_mat_len = m * k;
    let vh_mat_len = k * n;
    let mut out = alloc_zeroed_buf::<T>(bcount * out_mat_len);

    let compute_batch = |idx: usize, dst: &mut [T], scratch: &mut (Vec<T>, Vec<T>)| {
        let s_b = &s_data[idx * k..(idx + 1) * k];
        let u_b = &u_data[idx * u_mat_len..(idx + 1) * u_mat_len];
        let vh_b = &vh_data[idx * vh_mat_len..(idx + 1) * vh_mat_len];

        let smax = s_b[0].to_f64_lossy();
        let cutoff = rcond_val * smax;

        // Build v_scaled (n, k) = conj(vh^T) * diag(inv_s) and u_conj_t (k, m) = conj(u^T).
        let (v_scaled, u_conj_t) = scratch;
        for r in 0..n {
            for j in 0..k {
                let sv = s_b[j].to_f64_lossy();
                let inv_sv = if sv > cutoff { 1.0 / sv } else { 0.0 };
                let vh_conj = vh_b[j * n + r].conj_val();
                v_scaled[r * k + j] = vh_conj.mul_val(T::from_real_val(T::Real::from_f64_lossy(inv_sv)));
            }
        }
        for j in 0..k {
            for c in 0..m {
                u_conj_t[j * m + c] = u_b[c * k + j].conj_val();
            }
        }
        unsafe {
            T::strided_gemm(
                n,
                k,
                m,
                v_scaled.as_ptr(),
                k as isize,
                1,
                u_conj_t.as_ptr(),
                m as isize,
                1,
                dst.as_mut_ptr(),
                m as isize,
                1,
            );
        }
    };

    let work = m.saturating_mul(n).saturating_mul(k);
    if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
        out.as_mut_slice()
            .par_chunks_mut(out_mat_len)
            .enumerate()
            .for_each_init(
                || (vec![T::ZERO; n * k], vec![T::ZERO; k * m]),
                |ws, (i, dst)| compute_batch(i, dst, ws),
            );
    } else {
        let mut ws = (vec![T::ZERO; n * k], vec![T::ZERO; k * m]);
        for (i, dst) in out.as_mut_slice().chunks_exact_mut(out_mat_len).enumerate() {
            compute_batch(i, dst, &mut ws);
        }
    }

    ndarray_from_aligned(out, &out_shape)
}

/// Dynamic `pinv`.
pub fn pinv_dyn(a: &DynArray, rcond: Option<f64>, hermitian: bool) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgScalar::into_dyn(pinv(va, rcond, hermitian)?)))
}

/// Linear least-squares solution to `A @ x = b`, matching `numpy.linalg.lstsq`.
pub fn lstsq<T: LinalgScalar>(
    a: &NdArray<T>,
    b: &NdArray<T>,
    rcond: Option<f64>,
) -> Result<LstsqOutput<T>> {
    if a.ndim() != 2 {
        return Err(NumPyError::LinAlgError(format!(
            "{}-dimensional array given for a. Array must be two-dimensional",
            a.ndim()
        )));
    }
    if b.ndim() != 1 && b.ndim() != 2 {
        return Err(NumPyError::LinAlgError(format!(
            "{}-dimensional array given for b. Array must be 1- or 2-dimensional",
            b.ndim()
        )));
    }
    let m = a.shape()[0];
    let n = a.shape()[1];
    let b_is_1d = b.ndim() == 1;
    let k = if b_is_1d { 1 } else { b.shape()[1] };
    if b.shape()[0] != m {
        return Err(NumPyError::LinAlgError(format!(
            "Incompatible dimensions: a has {m} rows, b has {} rows",
            b.shape()[0]
        )));
    }

    let x_shape: Vec<usize> = if b_is_1d { vec![n] } else { vec![n, k] };
    let min_mn = m.min(n);

    // Check finite values upfront on both `a` and `b`.
    let a_c = a.to_c_contiguous();
    let b_2d = if b_is_1d {
        b.to_c_contiguous().reshape(&[m as isize, 1])?
    } else {
        b.to_c_contiguous()
    };
    let a_slice = a_c.as_slice().unwrap_or(&[]);
    let b_slice = b_2d.as_slice().unwrap_or(&[]);
    if !unsafe { mat_all_finite(a_slice.as_ptr(), m, n, n as isize, 1) }
        || !unsafe { mat_all_finite(b_slice.as_ptr(), m, k, k as isize, 1) }
    {
        return Err(NumPyError::LinAlgError(
            "Array must not contain infs or NaNs".into(),
        ));
    }

    if m == 0 || n == 0 || k == 0 {
        let x = NdArray::zeros(&x_shape)?;
        let s = NdArray::zeros(&[min_mn])?;
        let rank = NdArray::from_scalar(0i64);
        let residuals = if m > n && n == 0 && k > 0 {
            // rank == n == 0 and m > n: residual is ||B[:, col]||_2^2.
            let res_len = if b_is_1d { 1 } else { k };
            let mut rbuf = alloc_zeroed_buf::<T::Real>(res_len);
            for col in 0..k {
                let mut acc = 0.0f64;
                for row in 0..m {
                    let v = b_slice[row * k + col];
                    let mag = v.abs_real().to_f64_lossy();
                    acc += mag * mag;
                }
                rbuf.as_mut_slice()[col] = T::Real::from_f64_lossy(acc);
            }
            ndarray_from_aligned(rbuf, &[res_len])?
        } else {
            NdArray::zeros(&[0])?
        };
        return Ok(LstsqOutput {
            x,
            residuals,
            rank,
            s,
        });
    }

    let rcond_eff = match rcond {
        Some(r) if r < 0.0 => T::Real::eps().to_f64_lossy(),
        Some(r) => r,
        None => (m.max(n) as f64) * T::Real::eps().to_f64_lossy(),
    };

    let svd_out = svd(&a_c, false, true, false)?;
    let u_arr = svd_out.u.expect("svd u");
    let s_arr = svd_out.s;
    let vh_arr = svd_out.vh.expect("svd vh");
    let u_data = u_arr.as_slice().expect("u contiguous");
    let s_data = s_arr.as_slice().expect("s contiguous");
    let vh_data = vh_arr.as_slice().expect("vh contiguous");

    let smax = s_data[0].to_f64_lossy();
    let cutoff = rcond_eff * smax;
    let mut rank_val: i64 = 0;
    for &sv in s_data {
        if sv.to_f64_lossy() > cutoff {
            rank_val += 1;
        }
    }

    // 1) u_h_b = U^H @ B, shape (min_mn, k)
    let mut u_conj_t = vec![T::ZERO; min_mn * m];
    for j in 0..min_mn {
        for r in 0..m {
            u_conj_t[j * m + r] = u_data[r * min_mn + j].conj_val();
        }
    }
    let mut uh_b = vec![T::ZERO; min_mn * k];
    unsafe {
        T::strided_gemm(
            min_mn,
            m,
            k,
            u_conj_t.as_ptr(),
            m as isize,
            1,
            b_slice.as_ptr(),
            k as isize,
            1,
            uh_b.as_mut_ptr(),
            k as isize,
            1,
        );
    }

    // 2) Scale rows by 1 / s[j] if s[j] > cutoff else 0.
    for j in 0..min_mn {
        let sv = s_data[j].to_f64_lossy();
        let inv_sv = if sv > cutoff { 1.0 / sv } else { 0.0 };
        let scale = T::from_real_val(T::Real::from_f64_lossy(inv_sv));
        for col in 0..k {
            uh_b[j * k + col] = uh_b[j * k + col].mul_val(scale);
        }
    }

    // 3) x = V @ scaled_uh_b = (Vh^H) @ uh_b, shape (n, k)
    let mut v_mat = vec![T::ZERO; n * min_mn];
    for r in 0..n {
        for j in 0..min_mn {
            v_mat[r * min_mn + j] = vh_data[j * n + r].conj_val();
        }
    }
    let mut x_buf = alloc_zeroed_buf::<T>(n * k);
    unsafe {
        T::strided_gemm(
            n,
            min_mn,
            k,
            v_mat.as_ptr(),
            min_mn as isize,
            1,
            uh_b.as_ptr(),
            k as isize,
            1,
            x_buf.as_mut_slice().as_mut_ptr(),
            k as isize,
            1,
        );
    }

    // 4) Residuals: nonempty only if rank == n and m > n.
    let residuals = if rank_val as usize == n && m > n {
        let mut ax = vec![T::ZERO; m * k];
        unsafe {
            T::strided_gemm(
                m,
                n,
                k,
                a_slice.as_ptr(),
                n as isize,
                1,
                x_buf.as_slice().as_ptr(),
                k as isize,
                1,
                ax.as_mut_ptr(),
                k as isize,
                1,
            );
        }
        let res_len = if b_is_1d { 1 } else { k };
        let mut rbuf = alloc_zeroed_buf::<T::Real>(res_len);
        for col in 0..k {
            let mut acc = 0.0f64;
            for row in 0..m {
                let diff = b_slice[row * k + col].sub_elem(ax[row * k + col]);
                let mag = diff.abs_real().to_f64_lossy();
                acc += mag * mag;
            }
            rbuf.as_mut_slice()[col] = T::Real::from_f64_lossy(acc);
        }
        ndarray_from_aligned(rbuf, &[res_len])?
    } else {
        NdArray::zeros(&[0])?
    };

    Ok(LstsqOutput {
        x: ndarray_from_aligned(x_buf, &x_shape)?,
        residuals,
        rank: NdArray::from_scalar(rank_val),
        s: s_arr,
    })
}

/// Dynamic `lstsq`.
pub fn lstsq_dyn(a: &DynArray, b: &DynArray, rcond: Option<f64>) -> Result<LstsqResult> {
    dispatch_linalg_pair_dyn!(a, b, (va, vb) => {
        let out = lstsq(va, vb, rcond)?;
        Ok(LstsqResult {
            x: LinalgScalar::into_dyn(out.x),
            residuals: LinalgReal::real_into_dyn(out.residuals),
            rank: DynArray::I64(out.rank),
            s: LinalgReal::real_into_dyn(out.s),
        })
    })
}

/// Raise a square matrix (or stack of square matrices) to integer power `n`, matching `numpy.linalg.matrix_power`.
pub fn matrix_power<T: ProductScalar>(a: &NdArray<T>, n: i64) -> Result<NdArray<T>> {
    if n >= 0 {
        matrix_power_nonneg(a, n as u64)
    } else {
        let inv_a = T::invert_stacked(a)?;
        matrix_power_nonneg(&inv_a, n.unsigned_abs())
    }
}

/// Dynamic `matrix_power`. For `n < 0`, integers/bool/`f16` auto-promote to `f64` matching NumPy.
pub fn matrix_power_dyn(a: &DynArray, n: i64) -> Result<DynArray> {
    if n >= 0 {
        dispatch_product_dyn!(a, va => Ok(ProductScalar::product_into_dyn(matrix_power_nonneg(va, n as u64)?)))
    } else {
        dispatch_linalg_dyn!(a, va => Ok(LinalgScalar::into_dyn(matrix_power(va, n)?)))
    }
}

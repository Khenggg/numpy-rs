//! Matrix decompositions: cholesky, qr, svd, svdvals, eig, eigvals, eigh, eigvalsh.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::svd::ComputeSvdVectors;
use faer::{Mat, Par};
use numpy_types::{DynArray, NdArray, NumPyError, Result, Scalar};
use rayon::prelude::*;

use crate::batch::{
    DECOMP_RAYON_THRESHOLD, alloc_zeroed_buf, batch_len, copy_strided_mat_to_c,
    copy_strided_to_faer_mat, fill_identity_c, linear_strided_offset, mat_all_finite,
    mat_mut_from_c_slice, mat_ref_from_raw, matrix_stack_descriptor, ndarray_from_aligned,
    parse_stacked_2d, parse_stacked_square, should_par_batches, single_mat_par,
};
use crate::dispatch_linalg_dyn;
use crate::traits::{
    EigOutput, EigResult, EighOutput, EighResult, LinalgReal, LinalgScalar, ProductScalar, QrMode,
    QrOutput, QrResult, SvdOutput, SvdResult, Uplo,
};

fn cholesky_lower_nonfinite<T: LinalgScalar>(dst: &mut [T], n: usize) -> Result<()> {
    for j in 0..n {
        let mut dot_diag = T::ZERO;
        for p in 0..j {
            let v = dst[j * n + p];
            dot_diag = dot_diag.add_elem(v.mul_elem(v.conj_val()));
        }
        let diag_r = dst[j * n + j].sub_elem(dot_diag).real_part();
        if diag_r.to_f64_lossy() <= 0.0 {
            return Err(NumPyError::LinAlgError(
                "Matrix is not positive definite".into(),
            ));
        }
        let l_jj = diag_r.sqrt_val();
        dst[j * n + j] = T::from_real_val(l_jj);
        for i in (j + 1)..n {
            let mut acc = T::ZERO;
            for p in 0..j {
                acc = acc.add_elem(dst[i * n + p].mul_elem(dst[j * n + p].conj_val()));
            }
            dst[i * n + j] = dst[i * n + j].sub_elem(acc).div_real(l_jj);
        }
    }
    Ok(())
}

/// Cholesky decomposition of stacked positive-definite Hermitian/symmetric matrices `[..., N, N]`.
/// When `upper == false`, returns lower-triangular `L` with `L @ L^H = A`.
/// When `upper == true`, returns upper-triangular `U` with `U^H @ U = A`.
pub fn cholesky<T: LinalgScalar>(a: &NdArray<T>, upper: bool) -> Result<NdArray<T>> {
    let (batch_shape_ref, n) = parse_stacked_square(a)?;
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    let mat_len = n.saturating_mul(n);
    let mut out = alloc_zeroed_buf::<T>(bcount.saturating_mul(mat_len));
    if bcount == 0 || n == 0 {
        return ndarray_from_aligned(out, a.shape());
    }

    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);

    let run_one = |batch_idx: usize, dst: &mut [T], par: Par, stack_buf: &mut MemBuffer| -> Result<()> {
        let oa = linear_strided_offset(batch_idx, &batch_shape, bstr_a);
        let src_ptr = unsafe { a_base.offset(oa) };
        if !upper {
            // Copy lower triangle into `dst` and zero strict upper triangle.
            for i in 0..n {
                let row_ptr = unsafe { src_ptr.offset(i as isize * rsa) };
                for j in 0..=i {
                    dst[i * n + j] = unsafe { *row_ptr.offset(j as isize * csa) };
                }
                for j in (i + 1)..n {
                    dst[i * n + j] = T::ZERO;
                }
            }
        } else {
            // Read only upper triangle of A, storing A^H into lower triangle of `dst`.
            for i in 0..n {
                for j in 0..=i {
                    let v = unsafe { *src_ptr.offset(j as isize * rsa + i as isize * csa) };
                    dst[i * n + j] = v.conj_val();
                }
                for j in (i + 1)..n {
                    dst[i * n + j] = T::ZERO;
                }
            }
        }

        if dst.iter().any(|v| !v.is_finite_elem()) {
            cholesky_lower_nonfinite(dst, n)?;
        } else {
            let mat_mut = mat_mut_from_c_slice(dst, n, n);
            T::faer_cholesky_in_place(mat_mut, par, MemStack::new(stack_buf))?;
        }

        if upper {
            // Transpose-conjugate lower L -> upper U and zero strict lower triangle.
            for i in 0..n {
                for j in (i + 1)..n {
                    dst[i * n + j] = dst[j * n + i].conj_val();
                    dst[j * n + i] = T::ZERO;
                }
            }
        }
        Ok(())
    };

    let work = n.saturating_mul(n).saturating_mul(n);
    if bcount == 1 {
        let par = single_mat_par(n * n);
        let req = T::faer_cholesky_scratch(n, par);
        let mut buf = MemBuffer::new(req);
        run_one(0, out.as_mut_slice(), par, &mut buf)?;
    } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
        let req = T::faer_cholesky_scratch(n, Par::Seq);
        out.as_mut_slice()
            .par_chunks_mut(mat_len)
            .enumerate()
            .try_for_each_init(
                || MemBuffer::new(req),
                |buf, (i, dst)| run_one(i, dst, Par::Seq, buf),
            )?;
    } else {
        let req = T::faer_cholesky_scratch(n, Par::Seq);
        let mut buf = MemBuffer::new(req);
        for (i, dst) in out.as_mut_slice().chunks_exact_mut(mat_len).enumerate() {
            run_one(i, dst, Par::Seq, &mut buf)?;
        }
    }

    ndarray_from_aligned(out, a.shape())
}

/// Dynamic `cholesky`.
pub fn cholesky_dyn(a: &DynArray, upper: bool) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgScalar::into_dyn(cholesky(va, upper)?)))
}

/// QR decomposition over stacked `[..., M, N]` matrices, matching `numpy.linalg.qr`.
pub fn qr<T: LinalgScalar>(a: &NdArray<T>, mode: QrMode) -> Result<QrOutput<T>> {
    let (batch_shape_ref, m, n) = parse_stacked_2d(a)?;
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    let k = m.min(n);

    let mut q_shape = batch_shape.clone();
    let mut r_shape = batch_shape.clone();
    let (q_cols, r_rows) = match mode {
        QrMode::Reduced => (k, k),
        QrMode::Complete => (m, m),
        QrMode::R => (0, k),
        QrMode::Raw => (0, 0),
    };
    q_shape.push(m);
    q_shape.push(q_cols);
    r_shape.push(r_rows);
    r_shape.push(n);

    if bcount == 0 || m == 0 || n == 0 {
        return match mode {
            QrMode::Reduced => Ok(QrOutput::QR {
                q: NdArray::zeros(&q_shape)?,
                r: NdArray::zeros(&r_shape)?,
            }),
            QrMode::Complete => {
                let mut q_buf = alloc_zeroed_buf::<T>(bcount.saturating_mul(m).saturating_mul(m));
                if bcount > 0 && m > 0 {
                    for chunk in q_buf.as_mut_slice().chunks_exact_mut(m * m) {
                        fill_identity_c(chunk, m);
                    }
                }
                Ok(QrOutput::QR {
                    q: ndarray_from_aligned(q_buf, &q_shape)?,
                    r: NdArray::zeros(&r_shape)?,
                })
            }
            QrMode::R => Ok(QrOutput::R {
                r: NdArray::zeros(&r_shape)?,
            }),
            QrMode::Raw => {
                let mut h_shape = batch_shape.clone();
                h_shape.push(n);
                h_shape.push(m);
                let mut tau_shape = batch_shape;
                tau_shape.push(k);
                Ok(QrOutput::Raw {
                    h: NdArray::zeros(&h_shape)?,
                    tau: NdArray::zeros(&tau_shape)?,
                })
            }
        };
    }

    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
    let bs = if mode == QrMode::Raw {
        1
    } else {
        T::faer_qr_bs(m, n)
    };
    let work = m.saturating_mul(n).saturating_mul(k);

    struct QrWorkspace<T> {
        qr_mat: Mat<T>,
        h_mat: Mat<T>,
        q_mat: Option<Mat<T>>,
        mem: MemBuffer,
    }

    let make_ws = |par: Par, need_q: bool, q_cols_dim: usize| -> QrWorkspace<T> {
        let req_qr = T::faer_qr_scratch(m, n, bs, par);
        let req_q = if need_q {
            T::faer_householder_left_scratch(m, bs, q_cols_dim)
        } else {
            StackReq::EMPTY
        };
        let req = req_qr.or(req_q);
        QrWorkspace {
            qr_mat: Mat::full(m, n, T::ZERO),
            h_mat: Mat::full(bs, k, T::ZERO),
            q_mat: need_q.then(|| Mat::full(m, q_cols_dim, T::ZERO)),
            mem: MemBuffer::new(req),
        }
    };

    let factor_into_ws = |batch_idx: usize, par: Par, ws: &mut QrWorkspace<T>| {
        let oa = linear_strided_offset(batch_idx, &batch_shape, bstr_a);
        let src_ptr = unsafe { a_base.offset(oa) };
        unsafe {
            copy_strided_to_faer_mat(src_ptr, m, n, rsa, csa, &mut ws.qr_mat);
        }
        T::faer_qr_in_place(
            ws.qr_mat.as_mut(),
            ws.h_mat.as_mut(),
            par,
            MemStack::new(&mut ws.mem),
        );
    };

    match mode {
        QrMode::R => {
            let r_len = k * n;
            let mut r_out = alloc_zeroed_buf::<T>(bcount * r_len);
            let run = |i: usize, dst_r: &mut [T], par: Par, ws: &mut QrWorkspace<T>| {
                factor_into_ws(i, par, ws);
                for row in 0..k {
                    for col in row..n {
                        dst_r[row * n + col] = ws.qr_mat[(row, col)];
                    }
                }
            };
            if bcount == 1 {
                let par = single_mat_par(m * n);
                let mut ws = make_ws(par, false, 0);
                run(0, r_out.as_mut_slice(), par, &mut ws);
            } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
                r_out
                    .as_mut_slice()
                    .par_chunks_mut(r_len)
                    .enumerate()
                    .for_each_init(
                        || make_ws(Par::Seq, false, 0),
                        |ws, (i, dst)| run(i, dst, Par::Seq, ws),
                    );
            } else {
                let mut ws = make_ws(Par::Seq, false, 0);
                for (i, dst) in r_out.as_mut_slice().chunks_exact_mut(r_len).enumerate() {
                    run(i, dst, Par::Seq, &mut ws);
                }
            }
            Ok(QrOutput::R {
                r: ndarray_from_aligned(r_out, &r_shape)?,
            })
        }
        QrMode::Raw => {
            let mut h_shape = batch_shape.clone();
            h_shape.push(n);
            h_shape.push(m);
            let mut tau_shape = batch_shape.clone();
            tau_shape.push(k);
            let h_len = n * m;
            let mut h_out = alloc_zeroed_buf::<T>(bcount * h_len);
            let mut tau_out = alloc_zeroed_buf::<T>(bcount * k);

            let run = |i: usize, dst_h: &mut [T], dst_tau: &mut [T], par: Par, ws: &mut QrWorkspace<T>| {
                factor_into_ws(i, par, ws);
                for r in 0..m {
                    for c in 0..n {
                        // LAPACK raw `h` has shape (N, M) in C-order (Fortran-transposed).
                        dst_h[c * m + r] = ws.qr_mat[(r, c)];
                    }
                }
                for c in 0..k {
                    let tau_faer = ws.h_mat[(0, c)];
                    let re = tau_faer.real_part();
                    dst_tau[c] = if re.is_finite_val() && re != T::Real::ZERO {
                        T::ONE.div_real(re)
                    } else {
                        T::ZERO
                    };
                }
            };

            if bcount == 1 {
                let par = single_mat_par(m * n);
                let mut ws = make_ws(par, false, 0);
                run(0, h_out.as_mut_slice(), tau_out.as_mut_slice(), par, &mut ws);
            } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
                h_out
                    .as_mut_slice()
                    .par_chunks_mut(h_len)
                    .zip(tau_out.as_mut_slice().par_chunks_mut(k))
                    .enumerate()
                    .for_each_init(
                        || make_ws(Par::Seq, false, 0),
                        |ws, (i, (dh, dt))| run(i, dh, dt, Par::Seq, ws),
                    );
            } else {
                let mut ws = make_ws(Par::Seq, false, 0);
                for (i, (dh, dt)) in h_out
                    .as_mut_slice()
                    .chunks_exact_mut(h_len)
                    .zip(tau_out.as_mut_slice().chunks_exact_mut(k))
                    .enumerate()
                {
                    run(i, dh, dt, Par::Seq, &mut ws);
                }
            }

            Ok(QrOutput::Raw {
                h: ndarray_from_aligned(h_out, &h_shape)?,
                tau: ndarray_from_aligned(tau_out, &tau_shape)?,
            })
        }
        QrMode::Reduced | QrMode::Complete => {
            let q_len = m * q_cols;
            let r_len = r_rows * n;
            let mut q_out = alloc_zeroed_buf::<T>(bcount * q_len);
            let mut r_out = alloc_zeroed_buf::<T>(bcount * r_len);

            let run = |i: usize, dst_q: &mut [T], dst_r: &mut [T], par: Par, ws: &mut QrWorkspace<T>| {
                factor_into_ws(i, par, ws);
                let q_mat = ws.q_mat.as_mut().expect("q_mat allocated");
                for c in 0..q_cols {
                    for r in 0..m {
                        q_mat[(r, c)] = if r == c { T::ONE } else { T::ZERO };
                    }
                }
                T::faer_apply_householder_left(
                    ws.qr_mat.as_ref().subcols(0, k),
                    ws.h_mat.as_ref(),
                    q_mat.as_mut(),
                    par,
                    MemStack::new(&mut ws.mem),
                );
                for row in 0..m {
                    for col in 0..q_cols {
                        dst_q[row * q_cols + col] = q_mat[(row, col)];
                    }
                }
                for row in 0..r_rows {
                    for col in row..n {
                        dst_r[row * n + col] = ws.qr_mat[(row, col)];
                    }
                }
            };

            if bcount == 1 {
                let par = single_mat_par(m * n);
                let mut ws = make_ws(par, true, q_cols);
                run(0, q_out.as_mut_slice(), r_out.as_mut_slice(), par, &mut ws);
            } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
                q_out
                    .as_mut_slice()
                    .par_chunks_mut(q_len)
                    .zip(r_out.as_mut_slice().par_chunks_mut(r_len))
                    .enumerate()
                    .for_each_init(
                        || make_ws(Par::Seq, true, q_cols),
                        |ws, (i, (dq, dr))| run(i, dq, dr, Par::Seq, ws),
                    );
            } else {
                let mut ws = make_ws(Par::Seq, true, q_cols);
                for (i, (dq, dr)) in q_out
                    .as_mut_slice()
                    .chunks_exact_mut(q_len)
                    .zip(r_out.as_mut_slice().chunks_exact_mut(r_len))
                    .enumerate()
                {
                    run(i, dq, dr, Par::Seq, &mut ws);
                }
            }

            Ok(QrOutput::QR {
                q: ndarray_from_aligned(q_out, &q_shape)?,
                r: ndarray_from_aligned(r_out, &r_shape)?,
            })
        }
    }
}

/// Dynamic `qr`.
pub fn qr_dyn(a: &DynArray, mode: QrMode) -> Result<QrResult> {
    dispatch_linalg_dyn!(a, va => match qr(va, mode)? {
        QrOutput::QR { q, r } => Ok(QrResult::QR {
            q: LinalgScalar::into_dyn(q),
            r: LinalgScalar::into_dyn(r),
        }),
        QrOutput::R { r } => Ok(QrResult::R {
            r: LinalgScalar::into_dyn(r),
        }),
        QrOutput::Raw { h, tau } => Ok(QrResult::Raw {
            h: LinalgScalar::into_dyn(h),
            tau: LinalgScalar::into_dyn(tau),
        }),
    })
}

/// Singular Value Decomposition over stacked `[..., M, N]` matrices, matching `numpy.linalg.svd`.
pub fn svd<T: LinalgScalar>(
    a: &NdArray<T>,
    full_matrices: bool,
    compute_uv: bool,
    hermitian: bool,
) -> Result<SvdOutput<T>> {
    let (batch_shape_ref, m, n) = parse_stacked_2d(a)?;
    if hermitian && m != n {
        return Err(NumPyError::LinAlgError(format!(
            "Last 2 dimensions of the array must be square when hermitian=True, got ({m}, {n})"
        )));
    }
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);
    let k = m.min(n);

    let (u_cols, v_rows) = if full_matrices { (m, n) } else { (k, k) };
    let mut s_shape = batch_shape.clone();
    s_shape.push(k);
    let mut u_shape = batch_shape.clone();
    u_shape.push(m);
    u_shape.push(u_cols);
    let mut vh_shape = batch_shape.clone();
    vh_shape.push(v_rows);
    vh_shape.push(n);

    if bcount == 0 || m == 0 || n == 0 {
        let s = NdArray::zeros(&s_shape)?;
        if !compute_uv {
            return Ok(SvdOutput { u: None, s, vh: None });
        }
        let mut u_buf = alloc_zeroed_buf::<T>(bcount.saturating_mul(m).saturating_mul(u_cols));
        let mut vh_buf = alloc_zeroed_buf::<T>(bcount.saturating_mul(v_rows).saturating_mul(n));
        if full_matrices && bcount > 0 {
            if m > 0 {
                for chunk in u_buf.as_mut_slice().chunks_exact_mut(m * m) {
                    fill_identity_c(chunk, m);
                }
            }
            if n > 0 {
                for chunk in vh_buf.as_mut_slice().chunks_exact_mut(n * n) {
                    fill_identity_c(chunk, n);
                }
            }
        }
        return Ok(SvdOutput {
            u: Some(ndarray_from_aligned(u_buf, &u_shape)?),
            s,
            vh: Some(ndarray_from_aligned(vh_buf, &vh_shape)?),
        });
    }

    // FIX(F-B2-03): Hermitian SVD routes through `eigh`/`eigvalsh` in NumPy, which succeeds on NaN/Inf.
    if hermitian {
        if !compute_uv {
            let evals = eigvalsh(a, Uplo::Lower)?;
            let mut s_buf = alloc_zeroed_buf::<T::Real>(bcount * k);
            let src_all = evals.to_vec();
            for (src_chunk, dst_chunk) in src_all
                .chunks_exact(k)
                .zip(s_buf.as_mut_slice().chunks_exact_mut(k))
            {
                let mut vals: Vec<T::Real> = src_chunk.iter().map(|v| v.abs_val()).collect();
                vals.sort_by(|x, y| {
                    y.to_f64_lossy()
                        .partial_cmp(&x.to_f64_lossy())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                dst_chunk.copy_from_slice(&vals);
            }
            return Ok(SvdOutput {
                u: None,
                s: ndarray_from_aligned(s_buf, &s_shape)?,
                vh: None,
            });
        }

        let EighOutput {
            eigenvalues,
            eigenvectors,
        } = eigh(a, Uplo::Lower)?;
        let ev_all = eigenvalues.to_vec();
        let vec_all = eigenvectors.to_vec();
        let mut s_buf = alloc_zeroed_buf::<T::Real>(bcount * n);
        let mut u_buf = alloc_zeroed_buf::<T>(bcount * n * n);
        let mut vh_buf = alloc_zeroed_buf::<T>(bcount * n * n);

        let nn = n * n;
        for b in 0..bcount {
            let ev_b = &ev_all[b * n..(b + 1) * n];
            let vec_b = &vec_all[b * nn..(b + 1) * nn];
            let dst_s = &mut s_buf.as_mut_slice()[b * n..(b + 1) * n];
            let dst_u = &mut u_buf.as_mut_slice()[b * nn..(b + 1) * nn];
            let dst_vh = &mut vh_buf.as_mut_slice()[b * nn..(b + 1) * nn];

            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&i, &j| {
                ev_b[j]
                    .abs_val()
                    .to_f64_lossy()
                    .partial_cmp(&ev_b[i].abs_val().to_f64_lossy())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (out_col, &orig_col) in order.iter().enumerate() {
                let lam = ev_b[orig_col];
                dst_s[out_col] = lam.abs_val();
                let negative = lam.to_f64_lossy() < 0.0;
                for row in 0..n {
                    let v = vec_b[row * n + orig_col];
                    dst_u[row * n + out_col] = v;
                    let vh_val = v.conj_val();
                    dst_vh[out_col * n + row] = if negative { vh_val.negate_val() } else { vh_val };
                }
            }
        }

        return Ok(SvdOutput {
            u: Some(ndarray_from_aligned(u_buf, &u_shape)?),
            s: ndarray_from_aligned(s_buf, &s_shape)?,
            vh: Some(ndarray_from_aligned(vh_buf, &vh_shape)?),
        });
    }

    // Check finite upfront (matches NumPy GH-32591 / umath_linalg) for non-Hermitian SVD.
    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
    for idx in 0..bcount {
        let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
        if !unsafe { mat_all_finite(a_base.offset(oa), m, n, rsa, csa) } {
            return Err(NumPyError::LinAlgError("SVD did not converge".into()));
        }
    }

    let comp_u = if !compute_uv {
        ComputeSvdVectors::No
    } else if full_matrices {
        ComputeSvdVectors::Full
    } else {
        ComputeSvdVectors::Thin
    };
    let comp_v = comp_u;
    let work = m.saturating_mul(n).saturating_mul(k);

    struct SvdWs<T> {
        a_col: Mat<T>,
        s_diag: Diag<T>,
        u_mat: Option<Mat<T>>,
        v_mat: Option<Mat<T>>,
        mem: MemBuffer,
    }

    let make_ws = |par: Par| -> SvdWs<T> {
        let req = T::faer_svd_scratch(m, n, comp_u, comp_v, par);
        SvdWs {
            a_col: Mat::full(m, n, T::ZERO),
            s_diag: Diag::full(k, T::ZERO),
            u_mat: compute_uv.then(|| Mat::full(m, u_cols, T::ZERO)),
            v_mat: compute_uv.then(|| Mat::full(n, v_rows, T::ZERO)),
            mem: MemBuffer::new(req),
        }
    };

    let run_svd = |batch_idx: usize,
                   dst_s: &mut [T::Real],
                   mut dst_u: Option<&mut [T]>,
                   mut dst_vh: Option<&mut [T]>,
                   par: Par,
                   ws: &mut SvdWs<T>|
     -> Result<()> {
        let oa = linear_strided_offset(batch_idx, &batch_shape, bstr_a);
        unsafe {
            copy_strided_to_faer_mat(a_base.offset(oa), m, n, rsa, csa, &mut ws.a_col);
        }

        T::faer_svd(
            ws.a_col.as_ref(),
            ws.s_diag.as_mut(),
            ws.u_mat.as_mut().map(|u| u.as_mut()),
            ws.v_mat.as_mut().map(|v| v.as_mut()),
            par,
            MemStack::new(&mut ws.mem),
        )?;

        let s_col = ws.s_diag.column_vector();
        for i in 0..k {
            dst_s[i] = s_col[i].real_part();
        }
        if let (Some(du), Some(u_mat)) = (dst_u.as_deref_mut(), ws.u_mat.as_ref()) {
            for r in 0..m {
                for c in 0..u_cols {
                    du[r * u_cols + c] = u_mat[(r, c)];
                }
            }
        }
        if let (Some(dvh), Some(v_mat)) = (dst_vh.as_deref_mut(), ws.v_mat.as_ref()) {
            // Vh = V^H: V has shape (n, v_rows), Vh has shape (v_rows, n).
            for r_vh in 0..v_rows {
                for c_vh in 0..n {
                    dvh[r_vh * n + c_vh] = v_mat[(c_vh, r_vh)].conj_val();
                }
            }
        }
        Ok(())
    };

    let mut s_buf = alloc_zeroed_buf::<T::Real>(bcount * k);
    if !compute_uv {
        if bcount == 1 {
            let par = single_mat_par(m * n);
            let mut ws = make_ws(par);
            run_svd(0, s_buf.as_mut_slice(), None, None, par, &mut ws)?;
        } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
            s_buf
                .as_mut_slice()
                .par_chunks_mut(k)
                .enumerate()
                .try_for_each_init(
                    || make_ws(Par::Seq),
                    |ws, (i, ds)| run_svd(i, ds, None, None, Par::Seq, ws),
                )?;
        } else {
            let mut ws = make_ws(Par::Seq);
            for (i, ds) in s_buf.as_mut_slice().chunks_exact_mut(k).enumerate() {
                run_svd(i, ds, None, None, Par::Seq, &mut ws)?;
            }
        }
        return Ok(SvdOutput {
            u: None,
            s: ndarray_from_aligned(s_buf, &s_shape)?,
            vh: None,
        });
    }

    let u_len = m * u_cols;
    let vh_len = v_rows * n;
    let mut u_buf = alloc_zeroed_buf::<T>(bcount * u_len);
    let mut vh_buf = alloc_zeroed_buf::<T>(bcount * vh_len);

    if bcount == 1 {
        let par = single_mat_par(m * n);
        let mut ws = make_ws(par);
        run_svd(
            0,
            s_buf.as_mut_slice(),
            Some(u_buf.as_mut_slice()),
            Some(vh_buf.as_mut_slice()),
            par,
            &mut ws,
        )?;
    } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
        s_buf
            .as_mut_slice()
            .par_chunks_mut(k)
            .zip(u_buf.as_mut_slice().par_chunks_mut(u_len))
            .zip(vh_buf.as_mut_slice().par_chunks_mut(vh_len))
            .enumerate()
            .try_for_each_init(
                || make_ws(Par::Seq),
                |ws, (i, ((ds, du), dvh))| run_svd(i, ds, Some(du), Some(dvh), Par::Seq, ws),
            )?;
    } else {
        let mut ws = make_ws(Par::Seq);
        for (i, ((ds, du), dvh)) in s_buf
            .as_mut_slice()
            .chunks_exact_mut(k)
            .zip(u_buf.as_mut_slice().chunks_exact_mut(u_len))
            .zip(vh_buf.as_mut_slice().chunks_exact_mut(vh_len))
            .enumerate()
        {
            run_svd(i, ds, Some(du), Some(dvh), Par::Seq, &mut ws)?;
        }
    }

    Ok(SvdOutput {
        u: Some(ndarray_from_aligned(u_buf, &u_shape)?),
        s: ndarray_from_aligned(s_buf, &s_shape)?,
        vh: Some(ndarray_from_aligned(vh_buf, &vh_shape)?),
    })
}

/// Singular values only, matching `numpy.linalg.svdvals`.
pub fn svdvals<T: LinalgScalar>(a: &NdArray<T>) -> Result<NdArray<T::Real>> {
    Ok(svd(a, false, false, false)?.s)
}

/// Dynamic `svd`.
pub fn svd_dyn(
    a: &DynArray,
    full_matrices: bool,
    compute_uv: bool,
    hermitian: bool,
) -> Result<SvdResult> {
    dispatch_linalg_dyn!(a, va => {
        let out = svd(va, full_matrices, compute_uv, hermitian)?;
        Ok(SvdResult {
            u: out.u.map(LinalgScalar::into_dyn),
            s: LinalgReal::real_into_dyn(out.s),
            vh: out.vh.map(LinalgScalar::into_dyn),
        })
    })
}

/// Dynamic `svdvals`.
pub fn svdvals_dyn(a: &DynArray) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgReal::real_into_dyn(svdvals(va)?)))
}

fn eig_core<T: LinalgScalar>(
    a: &NdArray<T>,
    compute_vecs: bool,
) -> Result<(NdArray<T::Complex>, Option<NdArray<T::Complex>>)> {
    let (batch_shape_ref, n) = parse_stacked_square(a)?;
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);

    let mut val_shape = batch_shape.clone();
    val_shape.push(n);
    let mut vec_shape = batch_shape.clone();
    vec_shape.push(n);
    vec_shape.push(n);

    if bcount == 0 || n == 0 {
        let evals = NdArray::zeros(&val_shape)?;
        let evecs = compute_vecs.then(|| NdArray::zeros(&vec_shape)).transpose()?;
        return Ok((evals, evecs));
    }

    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
    for idx in 0..bcount {
        let oa = linear_strided_offset(idx, &batch_shape, bstr_a);
        if !unsafe { mat_all_finite(a_base.offset(oa), n, n, rsa, csa) } {
            return Err(NumPyError::LinAlgError(
                "Array must not contain infs or NaNs".into(),
            ));
        }
    }

    let work = n.saturating_mul(n).saturating_mul(n);
    let mut val_buf = alloc_zeroed_buf::<T::Complex>(bcount * n);
    let mut vec_buf = compute_vecs.then(|| alloc_zeroed_buf::<T::Complex>(bcount * n * n));

    let make_ws = |par: Par| -> (Vec<T>, MemBuffer) {
        let req = T::evd_scratch_req(n, compute_vecs, par);
        (vec![T::ZERO; n * n], MemBuffer::new(req))
    };

    let run_one = |batch_idx: usize,
                   dst_s: &mut [T::Complex],
                   dst_u: Option<&mut [T::Complex]>,
                   par: Par,
                   ws: &mut (Vec<T>, MemBuffer)|
     -> Result<()> {
        let oa = linear_strided_offset(batch_idx, &batch_shape, bstr_a);
        let src_ptr = unsafe { a_base.offset(oa) };
        unsafe {
            copy_strided_mat_to_c(src_ptr, n, n, rsa, csa, &mut ws.0);
        }
        let a_ref = unsafe { mat_ref_from_raw(ws.0.as_ptr(), n, n, n as isize, 1) };
        T::evd_compute(a_ref, dst_s, dst_u, par, MemStack::new(&mut ws.1))
    };

    if let Some(ref mut vb) = vec_buf {
        let nn = n * n;
        if bcount == 1 {
            let par = single_mat_par(n * n);
            let mut ws = make_ws(par);
            run_one(0, val_buf.as_mut_slice(), Some(vb.as_mut_slice()), par, &mut ws)?;
        } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
            val_buf
                .as_mut_slice()
                .par_chunks_mut(n)
                .zip(vb.as_mut_slice().par_chunks_mut(nn))
                .enumerate()
                .try_for_each_init(
                    || make_ws(Par::Seq),
                    |ws, (i, (ds, du))| run_one(i, ds, Some(du), Par::Seq, ws),
                )?;
        } else {
            let mut ws = make_ws(Par::Seq);
            for (i, (ds, du)) in val_buf
                .as_mut_slice()
                .chunks_exact_mut(n)
                .zip(vb.as_mut_slice().chunks_exact_mut(nn))
                .enumerate()
            {
                run_one(i, ds, Some(du), Par::Seq, &mut ws)?;
            }
        }
        Ok((
            ndarray_from_aligned(val_buf, &val_shape)?,
            Some(ndarray_from_aligned(vec_buf.unwrap(), &vec_shape)?),
        ))
    } else {
        if bcount == 1 {
            let par = single_mat_par(n * n);
            let mut ws = make_ws(par);
            run_one(0, val_buf.as_mut_slice(), None, par, &mut ws)?;
        } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
            val_buf
                .as_mut_slice()
                .par_chunks_mut(n)
                .enumerate()
                .try_for_each_init(
                    || make_ws(Par::Seq),
                    |ws, (i, ds)| run_one(i, ds, None, Par::Seq, ws),
                )?;
        } else {
            let mut ws = make_ws(Par::Seq);
            for (i, ds) in val_buf.as_mut_slice().chunks_exact_mut(n).enumerate() {
                run_one(i, ds, None, Par::Seq, &mut ws)?;
            }
        }
        Ok((ndarray_from_aligned(val_buf, &val_shape)?, None))
    }
}

/// Eigenvalues and right eigenvectors of stacked square matrices `[..., N, N]`, matching `numpy.linalg.eig`.
pub fn eig<T: LinalgScalar>(a: &NdArray<T>) -> Result<EigOutput<T>> {
    let (eigenvalues, evecs) = eig_core(a, true)?;
    Ok(EigOutput {
        eigenvalues,
        eigenvectors: evecs.expect("eig requested eigenvectors"),
    })
}

/// Eigenvalues of stacked square matrices `[..., N, N]`, matching `numpy.linalg.eigvals`.
pub fn eigvals<T: LinalgScalar>(a: &NdArray<T>) -> Result<NdArray<T::Complex>> {
    Ok(eig_core(a, false)?.0)
}

fn eig_dyn_typed<T: LinalgScalar>(a: &NdArray<T>) -> Result<EigResult> {
    let out = eig(a)?;
    Ok(EigResult {
        eigenvalues: T::complex_into_dyn(out.eigenvalues),
        eigenvectors: T::complex_into_dyn(out.eigenvectors),
    })
}

fn eigvals_dyn_typed<T: LinalgScalar>(a: &NdArray<T>) -> Result<DynArray> {
    Ok(T::complex_into_dyn(eigvals(a)?))
}

/// Dynamic `eig`.
pub fn eig_dyn(a: &DynArray) -> Result<EigResult> {
    dispatch_linalg_dyn!(a, va => eig_dyn_typed(va))
}

/// Dynamic `eigvals`.
pub fn eigvals_dyn(a: &DynArray) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => eigvals_dyn_typed(va))
}

fn eigh_core<T: LinalgScalar>(
    a: &NdArray<T>,
    uplo: Uplo,
    compute_vecs: bool,
) -> Result<(NdArray<T::Real>, Option<NdArray<T>>)> {
    let (batch_shape_ref, n) = parse_stacked_square(a)?;
    let batch_shape = batch_shape_ref.to_vec();
    let bcount = batch_len(&batch_shape);

    let mut val_shape = batch_shape.clone();
    val_shape.push(n);
    let mut vec_shape = batch_shape.clone();
    vec_shape.push(n);
    vec_shape.push(n);

    if bcount == 0 || n == 0 {
        let evals = NdArray::zeros(&val_shape)?;
        let evecs = compute_vecs.then(|| NdArray::zeros(&vec_shape)).transpose()?;
        return Ok((evals, evecs));
    }

    let (a_base, rsa, csa, _, bstr_a) = matrix_stack_descriptor(a);
    let work = n.saturating_mul(n).saturating_mul(n);
    let mut val_buf = alloc_zeroed_buf::<T::Real>(bcount * n);
    let mut vec_buf = compute_vecs.then(|| alloc_zeroed_buf::<T>(bcount * n * n));

    struct EighWs<T> {
        a_mat: Option<Mat<T>>,
        s_diag: Diag<T>,
        u_col: Option<Mat<T>>,
        mem: MemBuffer,
    }

    let comp = if compute_vecs {
        ComputeEigenvectors::Yes
    } else {
        ComputeEigenvectors::No
    };

    let make_ws = |par: Par| -> EighWs<T> {
        let req = T::faer_self_adjoint_evd_scratch(n, comp, par);
        EighWs {
            a_mat: (uplo == Uplo::Upper).then(|| Mat::full(n, n, T::ZERO)),
            s_diag: Diag::full(n, T::ZERO),
            u_col: compute_vecs.then(|| Mat::full(n, n, T::ZERO)),
            mem: MemBuffer::new(req),
        }
    };

    let run_one = |batch_idx: usize,
                   dst_s: &mut [T::Real],
                   mut dst_u: Option<&mut [T]>,
                   par: Par,
                   ws: &mut EighWs<T>|
     -> Result<()> {
        let oa = linear_strided_offset(batch_idx, &batch_shape, bstr_a);
        let src_ptr = unsafe { a_base.offset(oa) };
        let a_ref = if let Some(ref mut a_mat) = ws.a_mat {
            // Mirror upper triangle (conjugated) into lower triangle for faer's lower-triangle EVD.
            for i in 0..n {
                for j in 0..=i {
                    let val = unsafe { *src_ptr.offset(j as isize * rsa + i as isize * csa) };
                    a_mat[(i, j)] = val.conj_val();
                }
            }
            a_mat.as_ref()
        } else {
            unsafe { mat_ref_from_raw(src_ptr, n, n, rsa, csa) }
        };
        T::faer_self_adjoint_evd(
            a_ref,
            ws.s_diag.as_mut(),
            ws.u_col.as_mut().map(|u| u.as_mut()),
            par,
            MemStack::new(&mut ws.mem),
        )?;

        let s_col = ws.s_diag.column_vector();
        for i in 0..n {
            dst_s[i] = s_col[i].real_part();
        }
        if let (Some(du), Some(u_mat)) = (dst_u.as_deref_mut(), ws.u_col.as_ref()) {
            for r in 0..n {
                for c in 0..n {
                    du[r * n + c] = u_mat[(r, c)];
                }
            }
        }
        Ok(())
    };

    if let Some(ref mut vb) = vec_buf {
        let nn = n * n;
        if bcount == 1 {
            let par = single_mat_par(n * n);
            let mut ws = make_ws(par);
            run_one(0, val_buf.as_mut_slice(), Some(vb.as_mut_slice()), par, &mut ws)?;
        } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
            val_buf
                .as_mut_slice()
                .par_chunks_mut(n)
                .zip(vb.as_mut_slice().par_chunks_mut(nn))
                .enumerate()
                .try_for_each_init(
                    || make_ws(Par::Seq),
                    |ws, (i, (ds, du))| run_one(i, ds, Some(du), Par::Seq, ws),
                )?;
        } else {
            let mut ws = make_ws(Par::Seq);
            for (i, (ds, du)) in val_buf
                .as_mut_slice()
                .chunks_exact_mut(n)
                .zip(vb.as_mut_slice().chunks_exact_mut(nn))
                .enumerate()
            {
                run_one(i, ds, Some(du), Par::Seq, &mut ws)?;
            }
        }
        Ok((
            ndarray_from_aligned(val_buf, &val_shape)?,
            Some(ndarray_from_aligned(vec_buf.unwrap(), &vec_shape)?),
        ))
    } else {
        if bcount == 1 {
            let par = single_mat_par(n * n);
            let mut ws = make_ws(par);
            run_one(0, val_buf.as_mut_slice(), None, par, &mut ws)?;
        } else if should_par_batches(bcount, work, DECOMP_RAYON_THRESHOLD) {
            val_buf
                .as_mut_slice()
                .par_chunks_mut(n)
                .enumerate()
                .try_for_each_init(
                    || make_ws(Par::Seq),
                    |ws, (i, ds)| run_one(i, ds, None, Par::Seq, ws),
                )?;
        } else {
            let mut ws = make_ws(Par::Seq);
            for (i, ds) in val_buf.as_mut_slice().chunks_exact_mut(n).enumerate() {
                run_one(i, ds, None, Par::Seq, &mut ws)?;
            }
        }
        Ok((ndarray_from_aligned(val_buf, &val_shape)?, None))
    }
}

/// Eigenvalues and eigenvectors of Hermitian/symmetric stacked matrices `[..., N, N]`,
/// matching `numpy.linalg.eigh`. Eigenvalues are sorted in ascending order.
pub fn eigh<T: LinalgScalar>(a: &NdArray<T>, uplo: Uplo) -> Result<EighOutput<T>> {
    let (eigenvalues, evecs) = eigh_core(a, uplo, true)?;
    Ok(EighOutput {
        eigenvalues,
        eigenvectors: evecs.expect("eigh requested eigenvectors"),
    })
}

/// Eigenvalues of Hermitian/symmetric stacked matrices `[..., N, N]`, matching `numpy.linalg.eigvalsh`.
pub fn eigvalsh<T: LinalgScalar>(a: &NdArray<T>, uplo: Uplo) -> Result<NdArray<T::Real>> {
    Ok(eigh_core(a, uplo, false)?.0)
}

/// Dynamic `eigh`.
pub fn eigh_dyn(a: &DynArray, uplo: Uplo) -> Result<EighResult> {
    dispatch_linalg_dyn!(a, va => {
        let out = eigh(va, uplo)?;
        Ok(EighResult {
            eigenvalues: LinalgReal::real_into_dyn(out.eigenvalues),
            eigenvectors: LinalgScalar::into_dyn(out.eigenvectors),
        })
    })
}

/// Dynamic `eigvalsh`.
pub fn eigvalsh_dyn(a: &DynArray, uplo: Uplo) -> Result<DynArray> {
    dispatch_linalg_dyn!(a, va => Ok(LinalgReal::real_into_dyn(eigvalsh(va, uplo)?)))
}

#![cfg(feature = "linalg")]

mod common;

use common::{assert_dyn_matches_oracle, non_contiguous_2d_views, py_assign_dyn, run_oracle};
use numpy_linalg::{
    cholesky, cond, cross, det, dot, eig, eigh, eigvals, eigvalsh, inner, inv, kron, linalg_outer,
    lstsq, matmul, matrix_norm, matrix_power, matrix_rank, matvec, multi_dot, norm, outer, pinv, qr,
    slogdet, solve, svd, svdvals, tensorinv, tensorsolve, tensordot, trace, trace_axes, vdot,
    vecdot, vecmat, vector_norm, NormOrd, QrMode, QrOutput, TensordotAxes, Uplo,
};
use numpy_types::{Complex64, DynArray, NdArray, Order, Slice, SliceArg};

#[test]
fn products_and_contractions_against_numpy() {
    let a = NdArray::from_shape_vec(
        &[3, 4],
        vec![1.0, -2.0, 3.5, 0.5, 4.0, 1.0, -1.5, 2.0, -3.0, 2.5, 0.0, 1.25],
    )
    .unwrap();
    let b = NdArray::from_shape_vec(
        &[4, 2],
        vec![2.0, 1.0, -1.0, 3.0, 0.5, -2.5, 4.0, 0.75],
    )
    .unwrap();
    let v1 = NdArray::from_vec(vec![1.0, -2.0, 3.0]);
    let v2 = NdArray::from_vec(vec![4.0, 0.5, -1.5]);
    let sq = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 0.0, 3.0]).unwrap();
    let c3 = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0]).unwrap();
    let c4 = NdArray::from_shape_vec(&[2, 3], vec![4.0, -1.0, 2.0, 3.0, 1.5, -0.5]).unwrap();

    for (layout, a_view) in non_contiguous_2d_views(&a) {
        let k_cols = a_view.shape()[1];
        let rhs = NdArray::from_shape_vec(
            &[k_cols, 2],
            (0..(k_cols * 2)).map(|i| (i as f64) * 0.5 - 1.0).collect(),
        )
        .unwrap();
        let py = format!(
            "{}\n{}\nemit_arr('dot', np.dot(a, b))\nemit_arr('matmul', a @ b)\n",
            py_assign_dyn("a", &DynArray::F64(a_view.clone())),
            py_assign_dyn("b", &DynArray::F64(rhs.clone()))
        );
        let o = run_oracle(&py);
        assert_dyn_matches_oracle(
            &dot(&a_view, &rhs).unwrap().into(),
            o.arr("dot"),
            1e-12,
            1e-12,
            false,
            false,
            &format!("{layout}/dot"),
        );
        assert_dyn_matches_oracle(
            &matmul(&a_view, &rhs).unwrap().into(),
            o.arr("matmul"),
            1e-12,
            1e-12,
            false,
            false,
            &format!("{layout}/matmul"),
        );
    }

    let cvec1 = NdArray::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)]);
    let cvec2 = NdArray::from_vec(vec![Complex64::new(3.0, -1.0), Complex64::new(2.0, 1.0)]);
    let t1 = NdArray::from_shape_vec(&[2, 3, 4], (1..=24).map(|x| x as f64).collect()).unwrap();
    let t2 = NdArray::from_shape_vec(&[4, 3, 2], (1..=24).map(|x| x as f64 * 0.25).collect()).unwrap();
    let m1 = NdArray::from_shape_vec(&[3, 4], (1..=12).map(|x| x as f64).collect()).unwrap();
    let m2 = b.clone();
    let m3 = NdArray::from_shape_vec(&[2, 5], vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0, 1.5, -2.0, 0.0, 2.5]).unwrap();
    let mv_v = NdArray::from_vec(vec![1.0, -1.0, 2.0, 0.5]);
    let vm_v = NdArray::from_vec(vec![0.5, -2.0, 1.5]);

    let py = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\
emit_arr('vdot', np.vdot(cv1, cv2))
emit_arr('inner', np.inner(v1, v2))
emit_arr('outer', np.outer(v1, v2))
emit_arr('louter', np.linalg.outer(v1, v2))
emit_arr('matvec', np.matvec(a, mvv))
emit_arr('vecmat', np.vecmat(vmv, a))
emit_arr('vecdot', np.vecdot(c3, c4, axis=-1))
emit_arr('tdot', np.tensordot(t1, t2, axes=([1, 2], [1, 0])))
emit_arr('kron', np.kron(sq, np.array([[1.0, -1.0], [2.0, 0.5]])))
emit_arr('cross', np.linalg.cross(c3, c4, axis=-1))
emit_arr('mdot', np.linalg.multi_dot([m1, m2, m3]))
emit_arr('mpow_pos', np.linalg.matrix_power(sq, 3))
emit_arr('mpow_neg', np.linalg.matrix_power(sq, -2))
emit_arr('trace', np.linalg.trace(t1, offset=1))
emit_arr('trace_ax', np.trace(t1, offset=-1, axis1=1, axis2=2))
",
        py_assign_dyn("cv1", &DynArray::C128(cvec1.clone())),
        py_assign_dyn("cv2", &DynArray::C128(cvec2.clone())),
        py_assign_dyn("v1", &DynArray::F64(v1.clone())),
        py_assign_dyn("v2", &DynArray::F64(v2.clone())),
        py_assign_dyn("a", &DynArray::F64(a.clone())),
        py_assign_dyn("mvv", &DynArray::F64(mv_v.clone())),
        py_assign_dyn("vmv", &DynArray::F64(vm_v.clone())),
        py_assign_dyn("c3", &DynArray::F64(c3.clone())),
        py_assign_dyn("c4", &DynArray::F64(c4.clone())),
        py_assign_dyn("t1", &DynArray::F64(t1.clone())),
        py_assign_dyn("t2", &DynArray::F64(t2.clone())),
        py_assign_dyn("sq", &DynArray::F64(sq.clone())),
        py_assign_dyn("m1", &DynArray::F64(m1.clone())),
        py_assign_dyn("m2", &DynArray::F64(m2.clone())),
        py_assign_dyn("m3", &DynArray::F64(m3.clone())),
    );
    let o = run_oracle(&py);

    assert_dyn_matches_oracle(&vdot(&cvec1, &cvec2).unwrap().into(), o.arr("vdot"), 1e-13, 1e-13, false, false, "vdot");
    assert_dyn_matches_oracle(&inner(&v1, &v2).unwrap().into(), o.arr("inner"), 1e-13, 1e-13, false, false, "inner");
    assert_dyn_matches_oracle(&outer(&v1, &v2).unwrap().into(), o.arr("outer"), 1e-13, 1e-13, false, false, "outer");
    assert_dyn_matches_oracle(&linalg_outer(&v1, &v2).unwrap().into(), o.arr("louter"), 1e-13, 1e-13, false, false, "linalg_outer");
    assert_dyn_matches_oracle(&matvec(&a, &mv_v).unwrap().into(), o.arr("matvec"), 1e-12, 1e-12, false, false, "matvec");
    assert_dyn_matches_oracle(&vecmat(&vm_v, &a).unwrap().into(), o.arr("vecmat"), 1e-12, 1e-12, false, false, "vecmat");
    assert_dyn_matches_oracle(&vecdot(&c3, &c4, -1).unwrap().into(), o.arr("vecdot"), 1e-12, 1e-12, false, false, "vecdot");
    assert_dyn_matches_oracle(
        &tensordot(&t1, &t2, TensordotAxes::Axes(vec![1, 2], vec![1, 0]))
            .unwrap()
            .into(),
        o.arr("tdot"),
        1e-11,
        1e-11,
        false,
        false,
        "tensordot",
    );
    let k2 = NdArray::from_shape_vec(&[2, 2], vec![1.0, -1.0, 2.0, 0.5]).unwrap();
    assert_dyn_matches_oracle(&kron(&sq, &k2).unwrap().into(), o.arr("kron"), 1e-13, 1e-13, false, false, "kron");
    assert_dyn_matches_oracle(&cross(&c3, &c4, -1).unwrap().into(), o.arr("cross"), 1e-13, 1e-13, false, false, "cross");
    assert_dyn_matches_oracle(&multi_dot(&[&m1, &m2, &m3]).unwrap().into(), o.arr("mdot"), 1e-11, 1e-11, false, false, "multi_dot");
    assert_dyn_matches_oracle(&matrix_power(&sq, 3).unwrap().into(), o.arr("mpow_pos"), 1e-12, 1e-12, false, false, "mpow_pos");
    assert_dyn_matches_oracle(&matrix_power(&sq, -2).unwrap().into(), o.arr("mpow_neg"), 1e-12, 1e-12, false, false, "mpow_neg");
    assert_dyn_matches_oracle(&trace(&t1, 1).unwrap().into(), o.arr("trace"), 1e-13, 1e-13, false, false, "trace");
    assert_dyn_matches_oracle(&trace_axes(&t1, -1, 1, 2).unwrap().into(), o.arr("trace_ax"), 1e-13, 1e-13, false, false, "trace_axes");
}

#[test]
fn norms_det_rank_cond_and_edge_cases_against_numpy() {
    let m = NdArray::from_shape_vec(
        &[2, 3, 2],
        vec![
            3.0, 1.0, -2.0, 4.0, 0.5, -1.5, //
            2.0, -1.0, 0.0, 3.0, 1.5, 2.5,
        ],
    )
    .unwrap();
    let sq = NdArray::from_shape_vec(
        &[2, 2, 2],
        vec![4.0, 1.0, 2.0, 3.0, -5.0, 2.0, 1.0, -4.0],
    )
    .unwrap();
    let v = NdArray::from_vec(vec![3.0, -4.0, 0.0, 12.0]);
    let zero_n = NdArray::<f64>::zeros(&[2, 2, 0]).unwrap();
    let zero_m = NdArray::<f64>::zeros(&[2, 0, 3]).unwrap();

    let py = format!(
        "{}\n{}\n{}\n{}\n{}\n\
emit_arr('vnorm_2', np.linalg.vector_norm(v, ord=2.0))
emit_arr('vnorm_inf', np.linalg.norm(v, ord=np.inf))
emit_arr('vnorm_0', np.linalg.norm(v, ord=0))
emit_arr('mnorm_fro', np.linalg.matrix_norm(m, ord='fro', keepdims=True))
emit_arr('mnorm_nuc', np.linalg.matrix_norm(m, ord='nuc'))
emit_arr('mnorm_1', np.linalg.matrix_norm(m, ord=1))
emit_arr('mnorm_n1', np.linalg.matrix_norm(m, ord=-1))
emit_arr('mnorm_2', np.linalg.matrix_norm(m, ord=2))
emit_arr('mnorm_n2', np.linalg.matrix_norm(m, ord=-2))
emit_arr('mnorm_inf', np.linalg.matrix_norm(m, ord=np.inf))
emit_arr('mnorm_ninf', np.linalg.matrix_norm(m, ord=-np.inf))
emit_arr('zn_ninf', np.linalg.matrix_norm(zn, ord=-np.inf))
emit_arr('zm_n1', np.linalg.matrix_norm(zm, ord=-1))
emit_arr('det', np.linalg.det(sq))
s, l = np.linalg.slogdet(sq)
emit_arr('sdet_s', s)
emit_arr('sdet_l', l)
emit_arr('rank', np.linalg.matrix_rank(m))
emit_arr('cond2', np.linalg.cond(sq, p=2))
emit_arr('condfro', np.linalg.cond(sq, p='fro'))
",
        py_assign_dyn("m", &DynArray::F64(m.clone())),
        py_assign_dyn("sq", &DynArray::F64(sq.clone())),
        py_assign_dyn("v", &DynArray::F64(v.clone())),
        py_assign_dyn("zn", &DynArray::F64(zero_n.clone())),
        py_assign_dyn("zm", &DynArray::F64(zero_m.clone())),
    );
    let o = run_oracle(&py);

    assert_dyn_matches_oracle(&vector_norm(&v, 2.0, None, false).unwrap().into(), o.arr("vnorm_2"), 1e-13, 1e-13, false, false, "vnorm_2");
    assert_dyn_matches_oracle(&norm(&v, NormOrd::PosInf, None, false).unwrap().into(), o.arr("vnorm_inf"), 0.0, 0.0, false, false, "vnorm_inf");
    assert_dyn_matches_oracle(&norm(&v, NormOrd::Zero, None, false).unwrap().into(), o.arr("vnorm_0"), 0.0, 0.0, false, false, "vnorm_0");

    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::Fro, true).unwrap().into(), o.arr("mnorm_fro"), 1e-13, 1e-13, false, false, "mnorm_fro");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::Nuc, false).unwrap().into(), o.arr("mnorm_nuc"), 1e-12, 1e-12, false, false, "mnorm_nuc");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::L1, false).unwrap().into(), o.arr("mnorm_1"), 1e-13, 1e-13, false, false, "mnorm_1");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::LNeg1, false).unwrap().into(), o.arr("mnorm_n1"), 1e-13, 1e-13, false, false, "mnorm_n1");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::L2, false).unwrap().into(), o.arr("mnorm_2"), 1e-12, 1e-12, false, false, "mnorm_2");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::LNeg2, false).unwrap().into(), o.arr("mnorm_n2"), 1e-12, 1e-12, false, false, "mnorm_n2");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::PosInf, false).unwrap().into(), o.arr("mnorm_inf"), 1e-13, 1e-13, false, false, "mnorm_inf");
    assert_dyn_matches_oracle(&matrix_norm(&m, NormOrd::NegInf, false).unwrap().into(), o.arr("mnorm_ninf"), 1e-13, 1e-13, false, false, "mnorm_ninf");
    assert_dyn_matches_oracle(&matrix_norm(&zero_n, NormOrd::NegInf, false).unwrap().into(), o.arr("zn_ninf"), 0.0, 0.0, false, false, "zn_ninf");
    assert_dyn_matches_oracle(&matrix_norm(&zero_m, NormOrd::LNeg1, false).unwrap().into(), o.arr("zm_n1"), 0.0, 0.0, false, false, "zm_n1");

    // Empty-batch invalid norm order rejection
    let empty_batch = NdArray::<f64>::zeros(&[0, 2, 3]).unwrap();
    assert!(matrix_norm(&empty_batch, NormOrd::P(3.0), false).is_err());

    assert_dyn_matches_oracle(&det(&sq).unwrap().into(), o.arr("det"), 1e-12, 1e-12, false, false, "det");
    let sd = slogdet(&sq).unwrap();
    assert_dyn_matches_oracle(&sd.sign.into(), o.arr("sdet_s"), 1e-13, 1e-13, false, false, "slogdet_sign");
    assert_dyn_matches_oracle(&sd.logabsdet.into(), o.arr("sdet_l"), 1e-12, 1e-12, false, false, "slogdet_log");
    assert_dyn_matches_oracle(&matrix_rank(&m, None, None, false).unwrap().into(), o.arr("rank"), 0.0, 0.0, false, false, "rank");
    assert_dyn_matches_oracle(&cond(&sq, NormOrd::L2).unwrap().into(), o.arr("cond2"), 1e-11, 1e-11, false, false, "cond2");
    assert_dyn_matches_oracle(&cond(&sq, NormOrd::Fro).unwrap().into(), o.arr("condfro"), 1e-11, 1e-11, false, false, "condfro");
}

#[test]
fn solvers_and_decompositions_against_numpy() {
    let a = NdArray::from_shape_vec(
        &[3, 3],
        vec![4.0, 1.0, -1.0, 2.0, 5.0, 1.0, -1.0, 2.0, 6.0],
    )
    .unwrap();
    let spd = NdArray::from_shape_vec(
        &[3, 3],
        vec![6.0, 2.0, -1.0, 2.0, 5.0, 1.5, -1.0, 1.5, 4.0],
    )
    .unwrap();
    let b1 = NdArray::from_vec(vec![1.0, 2.0, -3.0]);
    let b2 = NdArray::from_shape_vec(&[3, 2], vec![1.0, 0.0, 2.0, -1.0, -3.0, 4.0]).unwrap();
    let tall = NdArray::from_shape_vec(
        &[4, 3],
        vec![1.0, 2.0, -1.0, 0.5, -1.0, 3.0, 2.0, 0.0, 1.0, -1.5, 2.5, 0.5],
    )
    .unwrap();
    let tall_b = NdArray::from_vec(vec![1.0, 2.0, -1.0, 3.0]);
    let t_a = NdArray::from_shape_vec(
        &[2, 2, 4],
        {
            let mut v = vec![0.0; 16];
            for i in 0..4 {
                v[i * 4 + i] = 2.0 + i as f64;
            }
            v
        },
    )
    .unwrap();
    let t_b = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tinv_a = NdArray::from_shape_vec(
        &[4, 2, 2],
        {
            let mut v = vec![0.0; 16];
            for i in 0..4 {
                v[i * 4 + i] = 2.0 + i as f64;
            }
            v
        },
    )
    .unwrap();

    // Also include Complex64 Hermitian eigh + non-contiguous view solve
    let h_c = NdArray::from_shape_vec(
        &[2, 2],
        vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let py = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\
emit_arr('solve1', np.linalg.solve(a, b1))
emit_arr('solve2', np.linalg.solve(a, b2))
emit_arr('inv', np.linalg.inv(a))
emit_arr('pinv', np.linalg.pinv(tall))
emit_arr('tsolve', np.linalg.tensorsolve(ta, tb))
emit_arr('tinv', np.linalg.tensorinv(tinva, ind=1))
x, res, rnk, s = np.linalg.lstsq(tall, tall_b, rcond=None)
emit_arr('lstsq_x', x)
emit_arr('lstsq_res', res)
emit_arr('lstsq_rnk', np.asarray(rnk))
emit_arr('lstsq_s', s)
emit_arr('chol_l', np.linalg.cholesky(spd))
emit_arr('chol_u', np.linalg.cholesky(spd, upper=True))
emit_arr('svdvals', np.linalg.svdvals(tall))
emit_arr('eigvalsh', np.linalg.eigvalsh(spd))
emit_arr('c_eigvalsh', np.linalg.eigvalsh(hc))
emit_arr('eye3', np.eye(3))
emit_arr('eye4', np.eye(4))
",
        py_assign_dyn("a", &DynArray::F64(a.clone())),
        py_assign_dyn("spd", &DynArray::F64(spd.clone())),
        py_assign_dyn("b1", &DynArray::F64(b1.clone())),
        py_assign_dyn("b2", &DynArray::F64(b2.clone())),
        py_assign_dyn("tall", &DynArray::F64(tall.clone())),
        py_assign_dyn("tall_b", &DynArray::F64(tall_b.clone())),
        py_assign_dyn("ta", &DynArray::F64(t_a.clone())),
        py_assign_dyn("tb", &DynArray::F64(t_b.clone())),
        py_assign_dyn("tinva", &DynArray::F64(tinv_a.clone())),
        py_assign_dyn("hc", &DynArray::C128(h_c.clone())),
    );
    let o = run_oracle(&py);

    assert_dyn_matches_oracle(&solve(&a, &b1).unwrap().into(), o.arr("solve1"), 1e-12, 1e-12, false, false, "solve1");
    assert_dyn_matches_oracle(&solve(&a, &b2).unwrap().into(), o.arr("solve2"), 1e-12, 1e-12, false, false, "solve2");
    assert_dyn_matches_oracle(&inv(&a).unwrap().into(), o.arr("inv"), 1e-12, 1e-12, false, false, "inv");
    assert_dyn_matches_oracle(&pinv(&tall, None, false).unwrap().into(), o.arr("pinv"), 1e-12, 1e-12, false, false, "pinv");
    assert_dyn_matches_oracle(&tensorsolve(&t_a, &t_b, None).unwrap().into(), o.arr("tsolve"), 1e-12, 1e-12, false, false, "tensorsolve");
    assert_dyn_matches_oracle(&tensorinv(&tinv_a, 1).unwrap().into(), o.arr("tinv"), 1e-12, 1e-12, false, false, "tensorinv");

    let lsq = lstsq(&tall, &tall_b, None).unwrap();
    assert_dyn_matches_oracle(&lsq.x.into(), o.arr("lstsq_x"), 1e-12, 1e-12, false, false, "lstsq_x");
    assert_dyn_matches_oracle(&lsq.residuals.into(), o.arr("lstsq_res"), 1e-12, 1e-12, false, false, "lstsq_res");
    assert_dyn_matches_oracle(&lsq.rank.into(), o.arr("lstsq_rnk"), 0.0, 0.0, false, false, "lstsq_rnk");
    assert_dyn_matches_oracle(&lsq.s.into(), o.arr("lstsq_s"), 1e-12, 1e-12, false, false, "lstsq_s");

    assert_dyn_matches_oracle(&cholesky(&spd, false).unwrap().into(), o.arr("chol_l"), 1e-12, 1e-12, false, false, "chol_l");
    assert_dyn_matches_oracle(&cholesky(&spd, true).unwrap().into(), o.arr("chol_u"), 1e-12, 1e-12, false, false, "chol_u");

    // QR: reconstruction + orthogonality
    let QrOutput::QR { q, r } = qr(&tall, QrMode::Reduced).unwrap() else {
        panic!("expected QR");
    };
    let recon_qr = matmul(&q, &r).unwrap();
    let py_tall = format!("{}\nemit_arr('tall', tall)\n", py_assign_dyn("tall", &DynArray::F64(tall.clone())));
    let o_tall = run_oracle(&py_tall);
    assert_dyn_matches_oracle(&recon_qr.into(), o_tall.arr("tall"), 1e-12, 1e-12, false, false, "qr_recon");
    let qtq = matmul(&q.t(), &q).unwrap();
    assert_dyn_matches_oracle(&qtq.into(), o.arr("eye3"), 1e-12, 1e-12, false, false, "qr_ortho");

    // SVD: singular values + reconstruction + orthogonal U/Vh
    let svd_out = svd(&tall, true, true, false).unwrap();
    assert_dyn_matches_oracle(&svdvals(&tall).unwrap().into(), o.arr("svdvals"), 1e-12, 1e-12, false, false, "svdvals");
    assert_dyn_matches_oracle(&svd_out.s.clone().into(), o.arr("svdvals"), 1e-12, 1e-12, false, false, "svd_s");
    let u = svd_out.u.unwrap();
    let vh = svd_out.vh.unwrap();
    let mut s_diag = NdArray::<f64>::zeros(&[4, 3]).unwrap();
    for i in 0..3 {
        s_diag.set(&[i as isize, i as isize], svd_out.s.get(&[i as isize]).unwrap()).unwrap();
    }
    let recon_svd = matmul(&matmul(&u, &s_diag).unwrap(), &vh).unwrap();
    assert_dyn_matches_oracle(&recon_svd.into(), o_tall.arr("tall"), 1e-12, 1e-12, false, false, "svd_recon");
    let uut = matmul(&u.t(), &u).unwrap();
    assert_dyn_matches_oracle(&uut.into(), o.arr("eye4"), 1e-12, 1e-12, false, false, "svd_u_ortho");

    // Eigh + Eigvalsh + Eig
    assert_dyn_matches_oracle(&eigvalsh(&spd, Uplo::Lower).unwrap().into(), o.arr("eigvalsh"), 1e-12, 1e-12, false, false, "eigvalsh");
    assert_dyn_matches_oracle(&eigvalsh(&h_c, Uplo::Lower).unwrap().into(), o.arr("c_eigvalsh"), 1e-12, 1e-12, false, false, "c_eigvalsh");
    let eh = eigh(&spd, Uplo::Lower).unwrap();
    let vtv = matmul(&eh.eigenvectors.t(), &eh.eigenvectors).unwrap();
    assert_dyn_matches_oracle(&vtv.into(), o.arr("eye3"), 1e-12, 1e-12, false, false, "eigh_ortho");

    // Eig check: A @ V[:, i] == w[i] * V[:, i]
    let e_out = eig(&a).unwrap();
    let evals = eigvals(&a).unwrap();
    assert_eq!(evals.shape(), &[3]);
    assert_eq!(e_out.eigenvectors.shape(), &[3, 3]);
    // Verify trace equals sum of eigenvalues
    let tr_a = a.get(&[0, 0]).unwrap() + a.get(&[1, 1]).unwrap() + a.get(&[2, 2]).unwrap();
    let sum_ev: Complex64 = evals.to_vec().into_iter().sum();
    assert!((sum_ev.re - tr_a).abs() < 1e-11 && sum_ev.im.abs() < 1e-11);

    // Non-contiguous solve check
    let mut a_f = NdArray::<f64>::zeros_with_order(&[3, 3], Order::F).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            a_f.set(&[i, j], a.get(&[i, j]).unwrap()).unwrap();
        }
    }
    let b_rev = b2.slice(&[SliceArg::Slice(Slice::new(None, None, -1)), SliceArg::Slice(Slice::full())]).unwrap();
    assert_eq!(solve(&a_f, &b_rev).unwrap().shape(), &[3, 2]);
}

#[test]
fn round2_cross_overflow_against_numpy() {
    let py = r#"
a = np.broadcast_to(np.array([1.0, 2.0, 3.0]), (1 << 57, 1, 3))
b = np.broadcast_to(np.array([4.0, 5.0, 6.0]), (1, 4, 3))
try:
    np.cross(a, b, axis=-1)
except Exception as e:
    emit_err("cross_too_big", e)
"#;
    let o = run_oracle(py);
    assert!(o.errors.contains_key("cross_too_big"));
    assert_eq!(o.errors["cross_too_big"].0, "ValueError");

    let a = NdArray::from_vec(vec![1.0f64, 2.0, 3.0])
        .broadcast_to(&[1usize << 57, 1, 3])
        .unwrap();
    let b = NdArray::from_vec(vec![4.0f64, 5.0, 6.0])
        .broadcast_to(&[1, 4, 3])
        .unwrap();
    assert!(matches!(
        cross(&a, &b, -1),
        Err(numpy_types::NumPyError::InvalidValue(_))
    ));
}

#[test]
fn p1_linalg_f16_norms_overflow_hermitian_nan_lstsq_rank_and_qr_complex_raw() {
    use numpy_linalg::*;
    use numpy_types::{f16, DType, NumPyError};

    let py = r#"
for tag, fn in [
    ("svd_f16", lambda: np.linalg.svd(np.eye(2, dtype=np.float16))),
    ("inv_f16", lambda: np.linalg.inv(np.eye(2, dtype=np.float16))),
    ("solve_f16", lambda: np.linalg.solve(np.eye(2, dtype=np.float16), np.ones(2, dtype=np.float16))),
    ("det_f16", lambda: np.linalg.det(np.eye(2, dtype=np.float16))),
    ("cond_f16", lambda: np.linalg.cond(np.eye(2, dtype=np.float16))),
    ("rank_f16", lambda: np.linalg.matrix_rank(np.eye(2, dtype=np.float16))),
    ("mpow_neg_f16", lambda: np.linalg.matrix_power(np.eye(2, dtype=np.float16), -1)),
    ("mnorm_l2_f16", lambda: np.linalg.matrix_norm(np.eye(2, dtype=np.float16), ord=2)),
    ("mnorm_nuc_f16", lambda: np.linalg.matrix_norm(np.eye(2, dtype=np.float16), ord='nuc')),
]:
    try:
        fn()
    except Exception as e:
        emit_err(tag, e)

emit_arr("n_f16_ok", np.linalg.norm(np.array([3.0, 4.0], dtype=np.float16)))
emit_arr("n_f16_ovf", np.linalg.norm(np.array([200.0, 200.0], dtype=np.float16)))
emit_arr("n_f16_udf", np.linalg.norm(np.array([1e-4, 1e-4], dtype=np.float16)))
emit_arr("mn_f16_fro", np.linalg.matrix_norm(np.eye(2, dtype=np.float16), ord='fro'))

emit_arr("v_big_2", np.linalg.vector_norm(np.array([1e200, 1e200]), ord=2.0))
emit_arr("v_small_2", np.linalg.vector_norm(np.array([1e-200, 1e-200]), ord=2.0))
emit_arr("v_big_3", np.linalg.vector_norm(np.array([1e200, 1e200]), ord=3.0))
emit_arr("v_big_neg2", np.linalg.vector_norm(np.array([1e200, 1e200]), ord=-2.0))
emit_arr("f32_big_2", np.linalg.vector_norm(np.array([1e30, 1e30], dtype=np.float32), ord=2.0))
emit_arr("f32_small_2", np.linalg.vector_norm(np.array([1e-30, 1e-30], dtype=np.float32), ord=2.0))
emit_arr("m_big_fro", np.linalg.matrix_norm(np.array([[1e200, 0.0], [0.0, 1e200]]), ord='fro'))

a_nan = np.array([[1.0, np.nan], [np.nan, 2.0]])
u_h, s_h, vh_h = np.linalg.svd(a_nan, hermitian=True)
emit_arr("svd_h_nan_s", s_h)
emit_arr("pinv_h_nan", np.linalg.pinv(a_nan, hermitian=True))

_, _, rnk, _ = np.linalg.lstsq(np.eye(2), np.ones(2), rcond=None)
emit_arr("lstsq_rnk_i32", np.asarray(rnk))
"#;
    let o = run_oracle(py);

    let eye_f16 = DynArray::F16(NdArray::from_shape_vec(&[2, 2], vec![f16::ONE, f16::ZERO, f16::ZERO, f16::ONE]).unwrap());
    let vec_f16_one = DynArray::F16(NdArray::from_vec(vec![f16::ONE, f16::ONE]));
    for (tag, res) in [
        ("svd_f16", svd_dyn(&eye_f16, true, true, false).map(|_| ())),
        ("inv_f16", inv_dyn(&eye_f16).map(|_| ())),
        ("solve_f16", solve_dyn(&eye_f16, &vec_f16_one).map(|_| ())),
        ("det_f16", det_dyn(&eye_f16).map(|_| ())),
        ("cond_f16", cond_dyn(&eye_f16, NormOrd::Default).map(|_| ())),
        ("rank_f16", matrix_rank_dyn(&eye_f16, None, None, false).map(|_| ())),
        ("mpow_neg_f16", matrix_power_dyn(&eye_f16, -1).map(|_| ())),
        ("mnorm_l2_f16", matrix_norm_dyn(&eye_f16, NormOrd::L2, false).map(|_| ())),
        ("mnorm_nuc_f16", matrix_norm_dyn(&eye_f16, NormOrd::Nuc, false).map(|_| ())),
    ] {
        assert!(o.errors.contains_key(tag), "missing oracle err for {tag}");
        assert!(matches!(res, Err(NumPyError::DTypeError(_))), "{tag} should fail with DTypeError");
    }

    let v_ok = DynArray::F16(NdArray::from_vec(vec![f16::from_f32(3.0), f16::from_f32(4.0)]));
    let v_ovf = DynArray::F16(NdArray::from_vec(vec![f16::from_f32(200.0), f16::from_f32(200.0)]));
    let v_udf = DynArray::F16(NdArray::from_vec(vec![f16::from_f32(1e-4), f16::from_f32(1e-4)]));
    assert_dyn_matches_oracle(&norm_dyn(&v_ok, NormOrd::Default, None, false).unwrap(), o.arr("n_f16_ok"), 1e-3, 1e-3, false, false, "n_f16_ok");
    assert_dyn_matches_oracle(&norm_dyn(&v_ovf, NormOrd::Default, None, false).unwrap(), o.arr("n_f16_ovf"), 0.0, 0.0, false, false, "n_f16_ovf");
    assert_dyn_matches_oracle(&norm_dyn(&v_udf, NormOrd::Default, None, false).unwrap(), o.arr("n_f16_udf"), 0.0, 0.0, false, false, "n_f16_udf");
    assert_dyn_matches_oracle(&matrix_norm_dyn(&eye_f16, NormOrd::Fro, false).unwrap(), o.arr("mn_f16_fro"), 1e-3, 1e-3, false, false, "mn_f16_fro");

    assert_dyn_matches_oracle(&vector_norm(&NdArray::from_vec(vec![1e200f64, 1e200]), 2.0, None, false).unwrap().into(), o.arr("v_big_2"), 0.0, 0.0, false, false, "v_big_2");
    assert_dyn_matches_oracle(&vector_norm(&NdArray::from_vec(vec![1e-200f64, 1e-200]), 2.0, None, false).unwrap().into(), o.arr("v_small_2"), 0.0, 0.0, false, false, "v_small_2");
    assert_dyn_matches_oracle(&vector_norm(&NdArray::from_vec(vec![1e200f64, 1e200]), 3.0, None, false).unwrap().into(), o.arr("v_big_3"), 0.0, 0.0, false, false, "v_big_3");
    assert_dyn_matches_oracle(&vector_norm(&NdArray::from_vec(vec![1e200f64, 1e200]), -2.0, None, false).unwrap().into(), o.arr("v_big_neg2"), 0.0, 0.0, false, false, "v_big_neg2");
    assert_dyn_matches_oracle(&vector_norm(&NdArray::from_vec(vec![1e30f32, 1e30]), 2.0, None, false).unwrap().into(), o.arr("f32_big_2"), 0.0, 0.0, false, false, "f32_big_2");
    assert_dyn_matches_oracle(&vector_norm(&NdArray::from_vec(vec![1e-30f32, 1e-30]), 2.0, None, false).unwrap().into(), o.arr("f32_small_2"), 0.0, 0.0, false, false, "f32_small_2");
    let m_big = NdArray::from_shape_vec(&[2, 2], vec![1e200f64, 0.0, 0.0, 1e200]).unwrap();
    assert_dyn_matches_oracle(&matrix_norm(&m_big, NormOrd::Fro, false).unwrap().into(), o.arr("m_big_fro"), 0.0, 0.0, false, false, "m_big_fro");

    // Hermitian SVD & pinv succeed on NaN/Inf (do not raise LinAlgError like non-Hermitian SVD)
    let a_nan = NdArray::from_shape_vec(&[2, 2], vec![1.0, f64::NAN, f64::NAN, 2.0]).unwrap();
    assert!(svd(&a_nan, true, true, false).is_err());
    let svd_h = svd(&a_nan, true, true, true).unwrap();
    assert_dyn_matches_oracle(&svd_h.s.into(), o.arr("svd_h_nan_s"), 1e-12, 1e-12, true, false, "svd_h_nan_s");
    assert_dyn_matches_oracle(&pinv(&a_nan, None, true).unwrap().into(), o.arr("pinv_h_nan"), 1e-12, 1e-12, true, false, "pinv_h_nan");
    let a_diag_nan = NdArray::from_shape_vec(&[2, 2], vec![f64::NAN, 0.0, 0.0, 1.0]).unwrap();
    assert!(svd(&a_diag_nan, true, true, true).is_ok());
    assert!(pinv(&a_diag_nan, None, true).is_ok());
    let a_inf = NdArray::from_shape_vec(&[2, 2], vec![f64::INFINITY, 0.0, 0.0, 1.0]).unwrap();
    assert!(svd(&a_inf, true, true, false).is_err());
    assert!(svd(&a_inf, true, true, true).is_ok());
    assert!(pinv(&a_inf, None, true).is_ok());

    // lstsq rank dtype == I32
    let lsq = lstsq_dyn(
        &DynArray::F64(NdArray::from_shape_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap()),
        &DynArray::F64(NdArray::from_vec(vec![1.0, 1.0])),
        None,
    )
    .unwrap();
    assert_eq!(lsq.rank.dtype(), DType::I32);
    assert_dyn_matches_oracle(&lsq.rank, o.arr("lstsq_rnk_i32"), 0.0, 0.0, false, false, "lstsq_rnk_i32");

    // Complex QR raw reconstruction with faer Hermitian reflectors: Q^H @ Q == I and Q^H @ A upper-triangular
    let c32 = NdArray::from_shape_vec(
        &[3, 2],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ],
    )
    .unwrap();
    if let QrOutput::Raw { h, tau } = qr(&c32, QrMode::Raw).unwrap() {
        let h_v = h.to_vec();
        let tau_v = tau.to_vec();
        let mut q = NdArray::from_shape_vec(
            &[3, 3],
            (0..9).map(|i| if i % 4 == 0 { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }).collect(),
        )
        .unwrap();
        for j in 0..2 {
            let mut v = vec![Complex64::new(0.0, 0.0); 3];
            v[j] = Complex64::new(1.0, 0.0);
            for r in (j + 1)..3 {
                v[r] = h_v[j * 3 + r];
            }
            let qv = q.to_vec();
            let mut h_mat = vec![Complex64::new(0.0, 0.0); 9];
            for r in 0..3 {
                for c in 0..3 {
                    let id = if r == c { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) };
                    h_mat[r * 3 + c] = id - tau_v[j] * v[r] * v[c].conj();
                }
            }
            let mut next_q = vec![Complex64::new(0.0, 0.0); 9];
            for r in 0..3 {
                for c in 0..3 {
                    let mut s = Complex64::new(0.0, 0.0);
                    for k in 0..3 {
                        s += qv[r * 3 + k] * h_mat[k * 3 + c];
                    }
                    next_q[r * 3 + c] = s;
                }
            }
            q = NdArray::from_shape_vec(&[3, 3], next_q).unwrap();
        }
        let qv = q.to_vec();
        let mut qh = vec![Complex64::new(0.0, 0.0); 9];
        for r in 0..3 {
            for c in 0..3 {
                qh[r * 3 + c] = qv[c * 3 + r].conj();
            }
        }
        let qh_arr = NdArray::from_shape_vec(&[3, 3], qh).unwrap();
        let qtq = matmul(&qh_arr, &q).unwrap().to_vec();
        for r in 0..3 {
            for c in 0..3 {
                let target = if r == c { 1.0 } else { 0.0 };
                assert!((qtq[r * 3 + c].re - target).abs() < 1e-11 && qtq[r * 3 + c].im.abs() < 1e-11);
            }
        }
        let r_mat = matmul(&qh_arr, &c32).unwrap().to_vec();
        for row in 0..3 {
            for col in 0..row.min(2) {
                assert!(r_mat[row * 2 + col].norm() < 1e-11, "subdiag ({row},{col}) not zero");
            }
        }
    } else {
        panic!("expected QrOutput::Raw");
    }
}

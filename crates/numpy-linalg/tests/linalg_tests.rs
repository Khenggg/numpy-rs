use num_complex::{Complex32, Complex64};
use numpy_linalg::*;
use numpy_types::{DynArray, NdArray, NumPyError, Order, Slice, SliceArg, f16};

fn arr_f64(shape: &[usize], data: &[f64]) -> NdArray<f64> {
    NdArray::from_shape_vec(shape, data.to_vec()).unwrap()
}

fn arr_f32(shape: &[usize], data: &[f32]) -> NdArray<f32> {
    NdArray::from_shape_vec(shape, data.to_vec()).unwrap()
}

fn arr_c64(shape: &[usize], data: &[(f32, f32)]) -> NdArray<Complex32> {
    let v: Vec<Complex32> = data.iter().map(|&(r, i)| Complex32::new(r, i)).collect();
    NdArray::from_shape_vec(shape, v).unwrap()
}

fn arr_c128(shape: &[usize], data: &[(f64, f64)]) -> NdArray<Complex64> {
    let v: Vec<Complex64> = data.iter().map(|&(r, i)| Complex64::new(r, i)).collect();
    NdArray::from_shape_vec(shape, v).unwrap()
}

fn assert_close_f64(actual: &NdArray<f64>, expected: &[f64], shape: &[usize], tol: f64) {
    assert_eq!(actual.shape(), shape, "shape mismatch");
    let v = actual.to_vec();
    assert_eq!(v.len(), expected.len());
    for (idx, (&a, &e)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at idx {idx}: got {a}, expected {e} (diff={})",
            (a - e).abs()
        );
    }
}

fn assert_close_f32(actual: &NdArray<f32>, expected: &[f32], shape: &[usize], tol: f32) {
    assert_eq!(actual.shape(), shape, "shape mismatch");
    let v = actual.to_vec();
    for (idx, (&a, &e)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at idx {idx}: got {a}, expected {e}"
        );
    }
}

fn assert_close_c128(actual: &NdArray<Complex64>, expected: &[Complex64], shape: &[usize], tol: f64) {
    assert_eq!(actual.shape(), shape, "shape mismatch");
    let v = actual.to_vec();
    for (idx, (&a, &e)) in v.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).norm();
        assert!(d <= tol, "mismatch at idx {idx}: got {a}, expected {e}, |diff|={d}");
    }
}

fn assert_close_c64(actual: &NdArray<Complex32>, expected: &[Complex32], shape: &[usize], tol: f32) {
    assert_eq!(actual.shape(), shape, "shape mismatch");
    let v = actual.to_vec();
    for (idx, (&a, &e)) in v.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).norm();
        assert!(d <= tol, "mismatch at idx {idx}: got {a}, expected {e}, |diff|={d}");
    }
}

fn eye_f64(n: usize) -> NdArray<f64> {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    NdArray::from_shape_vec(&[n, n], v).unwrap()
}

fn eye_c128(n: usize) -> NdArray<Complex64> {
    let mut v = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        v[i * n + i] = Complex64::new(1.0, 0.0);
    }
    NdArray::from_shape_vec(&[n, n], v).unwrap()
}

fn conj_transpose_c128(a: &NdArray<Complex64>) -> NdArray<Complex64> {
    let t = a.transpose(None).unwrap();
    let v: Vec<Complex64> = t.to_vec().into_iter().map(|z| z.conj()).collect();
    NdArray::from_shape_vec(t.shape(), v).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Products & Tensor Ops
// ---------------------------------------------------------------------------

#[test]
fn test_dot_vdot_inner_outer() {
    // Scalar * array dot
    let s = NdArray::from_scalar(3.0f64);
    let m = arr_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_close_f64(&dot(&s, &m).unwrap(), &[3.0, 6.0, 9.0, 12.0], &[2, 2], 1e-12);
    assert_close_f64(&dot(&m, &s).unwrap(), &[3.0, 6.0, 9.0, 12.0], &[2, 2], 1e-12);

    // 1D dot & ND dot
    let a = arr_f64(&[3], &[1.0, 2.0, 3.0]);
    let b = arr_f64(&[3], &[4.0, -5.0, 6.0]);
    assert_close_f64(&dot(&a, &b).unwrap(), &[12.0], &[], 1e-12);

    // ND (2, 3, 2) dot (4, 2, 5) -> (2, 3, 4, 5)
    let da = arr_f64(&[2, 3, 2], &(1..=12).map(|x| x as f64).collect::<Vec<_>>());
    let db = arr_f64(&[4, 2, 5], &(1..=40).map(|x| x as f64).collect::<Vec<_>>());
    let dc = dot(&da, &db).unwrap();
    assert_eq!(dc.shape(), &[2, 3, 4, 5]);

    // vdot conjugates first argument; dot does NOT conjugate
    let ca = arr_c128(&[2, 2], &[(1.0, 2.0), (3.0, -4.0), (0.0, 1.0), (-2.0, 0.0)]);
    let cb = arr_c128(&[2, 2], &[(4.0, 1.0), (-2.0, 3.0), (1.0, -1.0), (3.0, 2.0)]);
    let vd = vdot(&ca, &cb).unwrap();
    // conj(1+2j)*(4+1j) + conj(3-4j)*(-2+3j) + conj(0+1j)*(1-1j) + conj(-2)*(3+2j)
    // = (6-7j) + (-18+1j) + (-1-1j) + (-6-4j) = (-19-11j)
    assert_close_c128(&vd, &[Complex64::new(-19.0, -11.0)], &[], 1e-12);

    // Non-contiguous Fortran order vdot
    let ca_f = NdArray::from_shape_vec_order(&[2, 2], ca.to_vec(), Order::F).unwrap();
    let cb_f = NdArray::from_shape_vec_order(&[2, 2], cb.to_vec(), Order::F).unwrap();
    let mut expected_f = Complex64::new(0.0, 0.0);
    for (z1, z2) in ca_f.to_vec().iter().zip(cb_f.to_vec().iter()) {
        expected_f += z1.conj() * z2;
    }
    assert_close_c128(&vdot(&ca_f, &cb_f).unwrap(), &[expected_f], &[], 1e-12);

    // inner: contracts last axis of both without conjugation
    let ia = arr_f64(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ib = arr_f64(&[2, 3], &[7.0, 8.0, 9.0, 1.0, 0.0, -1.0]);
    assert_close_f64(
        &inner(&ia, &ib).unwrap(),
        &[50.0, -2.0, 122.0, -2.0],
        &[2, 2],
        1e-12,
    );

    // outer vs linalg_outer
    let o_res = outer(&ia, &a).unwrap();
    assert_eq!(o_res.shape(), &[6, 3]);
    assert!(matches!(linalg_outer(&ia, &a), Err(NumPyError::ShapeMismatch(_))));
    let lo_res = linalg_outer(&a, &b).unwrap();
    assert_close_f64(
        &lo_res,
        &[4.0, -5.0, 6.0, 8.0, -10.0, 12.0, 12.0, -15.0, 18.0],
        &[3, 3],
        1e-12,
    );
}

#[test]
fn test_matmul_matvec_vecmat_vecdot() {
    // 1D @ 1D -> scalar, 1D @ 2D -> 1D, 2D @ 1D -> 1D
    let v1 = arr_f64(&[3], &[1.0, 2.0, 3.0]);
    let v2 = arr_f64(&[3], &[4.0, 5.0, 6.0]);
    assert_close_f64(&matmul(&v1, &v2).unwrap(), &[32.0], &[], 1e-12);

    let m = arr_f64(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_close_f64(&matmul(&v1, &m).unwrap(), &[22.0, 28.0], &[2], 1e-12);

    // Broadcasted 4D batched matmul: [2, 1, 3, 4] @ [1, 3, 4, 2] -> [2, 3, 3, 2]
    let ba = arr_f64(&[2, 1, 3, 4], &(0..24).map(|x| x as f64 * 0.5).collect::<Vec<_>>());
    let bb = arr_f64(&[1, 3, 4, 2], &(0..24).map(|x| (24 - x) as f64 * 0.25).collect::<Vec<_>>());
    let bc = matmul(&ba, &bb).unwrap();
    assert_eq!(bc.shape(), &[2, 3, 3, 2]);

    // Transposed & reversed (negative-stride) matmul
    let a_sq = arr_f64(&[3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let a_t = a_sq.transpose(None).unwrap();
    let a_rev = a_sq
        .slice(&[
            SliceArg::Slice(Slice::new(None, None, -1)),
            SliceArg::Slice(Slice::new(None, None, -1)),
        ])
        .unwrap();
    let prod_strided = matmul(&a_t, &a_rev).unwrap();
    let prod_contig = matmul(&a_t.to_c_contiguous(), &a_rev.to_c_contiguous()).unwrap();
    assert_close_f64(&prod_strided, &prod_contig.to_vec(), &[3, 3], 1e-12);

    // Complex vecmat (CONJUGATES x1), matvec (NO conjugate), vecdot (CONJUGATES x1) vs NumPy 2.5.4
    let u = arr_c128(&[2], &[(1.0, 2.0), (3.0, -4.0)]);
    let bmat = arr_c128(&[2, 2], &[(2.0, -1.0), (0.0, 1.0), (5.0, 2.0), (-1.0, -3.0)]);
    assert_close_c128(
        &vecmat(&u, &bmat).unwrap(),
        &[Complex64::new(7.0, 21.0), Complex64::new(11.0, -12.0)],
        &[2],
        1e-12,
    );
    assert_close_c128(
        &matvec(&bmat, &u).unwrap(),
        &[Complex64::new(8.0, 6.0), Complex64::new(-14.0, 7.0)],
        &[2],
        1e-12,
    );
    let w = arr_c128(&[2], &[(4.0, 1.0), (-2.0, 3.0)]);
    assert_close_c128(&vecdot(&u, &w, -1).unwrap(), &[Complex64::new(-12.0, -6.0)], &[], 1e-12);

    // matvec/vecmat ndim checks
    assert!(matches!(matvec(&v1, &v2), Err(NumPyError::ShapeMismatch(_))));
    assert!(matches!(vecmat(&v1, &v2), Err(NumPyError::ShapeMismatch(_))));

    // Large batched GEMM (triggers Rayon across batch) + giant single GEMM
    let big_a = arr_f32(&[80, 16, 16], &vec![1.0; 80 * 256]);
    let big_b = arr_f32(&[80, 16, 16], &vec![2.0; 80 * 256]);
    let big_c = matmul(&big_a, &big_b).unwrap();
    assert_eq!(big_c.shape(), &[80, 16, 16]);
    assert_close_f32(&big_c, &vec![32.0; 80 * 256], &[80, 16, 16], 1e-4);

    let g1 = arr_f64(&[100, 100], &vec![1.0; 10_000]);
    let g2 = arr_f64(&[100, 100], &vec![0.5; 10_000]);
    let g3 = matmul(&g1, &g2).unwrap();
    assert_close_f64(&g3, &vec![50.0; 10_000], &[100, 100], 1e-10);
}

#[test]
fn test_tensordot_kron_cross_multi_dot_trace_power() {
    // tensordot
    let a = arr_f64(&[3, 4, 5], &(0..60).map(|x| x as f64).collect::<Vec<_>>());
    let b = arr_f64(&[4, 3, 2], &(0..24).map(|x| x as f64).collect::<Vec<_>>());
    let td = tensordot(&a, &b, (vec![1isize, 0], vec![0, 1])).unwrap();
    assert_eq!(td.shape(), &[5, 2]);
    let td_outer = tensordot(&arr_f64(&[2], &[1.0, 2.0]), &arr_f64(&[3], &[3.0, 4.0, 5.0]), 0).unwrap();
    assert_close_f64(&td_outer, &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0], &[2, 3], 1e-12);

    // kron
    let ka = arr_f64(&[2, 2], &[1.0, 0.0, 0.0, 2.0]);
    let kb = arr_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let kc = kron(&ka, &kb).unwrap();
    assert_close_f64(
        &kc,
        &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 6.0, 8.0],
        &[4, 4],
        1e-12,
    );

    // cross
    let cx = arr_f64(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let cy = arr_f64(&[3], &[0.0, 1.0, 0.0]);
    assert_close_f64(&cross(&cx, &cy, -1).unwrap(), &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 3], 1e-12);
    assert!(matches!(
        cross(&arr_f64(&[2], &[1.0, 2.0]), &arr_f64(&[2], &[3.0, 4.0]), 0),
        Err(NumPyError::ShapeMismatch(_))
    ));

    // multi_dot with DP parenthesization & 1-D ends
    let m0 = arr_f64(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let m1 = arr_f64(&[4, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let m2 = arr_f64(&[2, 5], &[1.0, 2.0, 0.0, 1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    let m3 = arr_f64(&[5], &[1.0, 1.0, 1.0, 1.0, 1.0]);
    let md_res = multi_dot(&[&m0, &m1, &m2, &m3]).unwrap();
    let manual = dot(&dot(&dot(&m0, &m1).unwrap(), &m2).unwrap(), &m3).unwrap();
    assert_close_f64(&md_res, &manual.to_vec(), &[], 1e-12);

    // trace + trace_axes + dynamic promotion
    let tr_m = arr_f64(&[3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_close_f64(&trace(&tr_m, 0).unwrap(), &[15.0], &[], 1e-12);
    assert_close_f64(&trace(&tr_m, 1).unwrap(), &[8.0], &[], 1e-12);
    assert_close_f64(&trace(&tr_m, -1).unwrap(), &[12.0], &[], 1e-12);

    let dyn_i32 = DynArray::I32(NdArray::from_shape_vec(&[2, 2], vec![1, 2, 3, 4]).unwrap());
    let tr_dyn = trace_dyn(&dyn_i32, 0).unwrap();
    assert_eq!(tr_dyn.dtype(), numpy_types::DType::I64);
    if let DynArray::I64(a) = &tr_dyn {
        assert_eq!(a.to_vec(), vec![5]);
    } else {
        panic!("expected I64");
    }

    // matrix_power (+, 0, -) & DynArray negative promotion
    let pmat = arr_f64(&[2, 2], &[1.0, 1.0, 0.0, 1.0]);
    assert_close_f64(&matrix_power(&pmat, 0).unwrap(), &[1.0, 0.0, 0.0, 1.0], &[2, 2], 1e-12);
    assert_close_f64(&matrix_power(&pmat, 5).unwrap(), &[1.0, 5.0, 0.0, 1.0], &[2, 2], 1e-12);
    assert_close_f64(&matrix_power(&pmat, -3).unwrap(), &[1.0, -3.0, 0.0, 1.0], &[2, 2], 1e-12);

    let p_dyn_pos = matrix_power_dyn(&dyn_i32, 2).unwrap();
    assert_eq!(p_dyn_pos.dtype(), numpy_types::DType::I32);
    if let DynArray::I32(a) = &p_dyn_pos {
        assert_eq!(a.to_vec(), vec![7, 10, 15, 22]);
    } else {
        panic!("expected I32");
    }
    let p_dyn_neg = matrix_power_dyn(&dyn_i32, -1).unwrap();
    assert_eq!(p_dyn_neg.dtype(), numpy_types::DType::F64);
    if let DynArray::F64(a) = &p_dyn_neg {
        assert_close_f64(a, &[-2.0, 1.0, 1.5, -0.5], &[2, 2], 1e-12);
    } else {
        panic!("expected F64");
    }
}

#[test]
fn test_product_dyn_promotions_all_dtypes() {
    let b = DynArray::Bool(NdArray::from_shape_vec(&[2], vec![true, false]).unwrap());
    let i = DynArray::I16(NdArray::from_shape_vec(&[2], vec![3, 4]).unwrap());
    let d = dot_dyn(&b, &i).unwrap();
    assert_eq!(d.dtype(), numpy_types::DType::I16);
    if let DynArray::I16(a) = &d {
        assert_eq!(a.to_vec(), vec![3]);
    } else {
        panic!("expected I16");
    }

    let h1 = DynArray::F16(
        NdArray::from_shape_vec(&[2], vec![f16::from_f32(1.5), f16::from_f32(2.0)]).unwrap(),
    );
    let h2 = DynArray::F16(
        NdArray::from_shape_vec(&[2], vec![f16::from_f32(2.0), f16::from_f32(3.0)]).unwrap(),
    );
    let hd = matmul_dyn(&h1, &h2).unwrap();
    assert_eq!(hd.dtype(), numpy_types::DType::F16);
    if let DynArray::F16(a) = &hd {
        assert!((a.to_vec()[0].to_f32() - 9.0).abs() < 1e-3);
    } else {
        panic!("expected F16");
    }
}

// ---------------------------------------------------------------------------
// 2. Linear Systems & Inverses
// ---------------------------------------------------------------------------

#[test]
fn test_solve_inv_all_types_and_batching() {
    // 1-D RHS with batched A: (2, 3, 3) & (3,) -> (2, 3)
    let a = arr_f64(
        &[2, 3, 3],
        &[
            2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
            4.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0,
        ],
    );
    let b1 = arr_f64(&[3], &[1.0, 2.0, 3.0]);
    let x1 = solve(&a, &b1).unwrap();
    assert_eq!(x1.shape(), &[2, 3]);
    let rec = matvec(&a, &x1).unwrap();
    assert_close_f64(&rec, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], 1e-12);

    // NumPy 2.0 rejects (2, 3, 3) solve (2, 3)
    let b2_bad = arr_f64(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(solve(&a, &b2_bad).is_err());

    // Batched 2D RHS
    let b2 = arr_f64(&[1, 3, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, -1.0]);
    let x2 = solve(&a, &b2).unwrap();
    assert_eq!(x2.shape(), &[2, 3, 2]);
    let ax2 = matmul(&a, &x2).unwrap();
    assert_close_f64(
        &ax2,
        &[1.0, 0.0, 0.0, 1.0, 2.0, -1.0, 1.0, 0.0, 0.0, 1.0, 2.0, -1.0],
        &[2, 3, 2],
        1e-12,
    );

    // Complex32 & Complex64 inv & singular LinAlgError
    let ca = arr_c128(&[2, 2], &[(1.0, 1.0), (2.0, 0.0), (0.0, -1.0), (3.0, 2.0)]);
    let cinv = inv(&ca).unwrap();
    assert_close_c128(&matmul(&ca, &cinv).unwrap(), &eye_c128(2).to_vec(), &[2, 2], 1e-12);

    let ca32 = arr_c64(&[2, 2], &[(2.0, 1.0), (1.0, 0.0), (0.0, 1.0), (3.0, -1.0)]);
    let cinv32 = inv(&ca32).unwrap();
    let id32 = matmul(&ca32, &cinv32).unwrap();
    assert_close_c64(
        &id32,
        &[Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)],
        &[2, 2],
        1e-5,
    );

    let singular = arr_f64(&[2, 2], &[1.0, 2.0, 2.0, 4.0]);
    assert!(matches!(inv(&singular), Err(NumPyError::LinAlgError(_))));
    assert!(matches!(solve(&singular, &arr_f64(&[2], &[1.0, 2.0])), Err(NumPyError::LinAlgError(_))));

    // Empty 0x0 inv
    let emp = arr_f64(&[0, 0], &[]);
    assert_eq!(inv(&emp).unwrap().shape(), &[0, 0]);

    // Rayon parallel batched solve (128 x 8x8)
    let mut big_a = vec![0.0; 128 * 64];
    let mut big_b = vec![1.0; 128 * 8];
    for batch in 0..128 {
        for i in 0..8 {
            big_a[batch * 64 + i * 8 + i] = 4.0 + (batch % 5) as f64;
            big_b[batch * 8 + i] = (i + 1) as f64;
        }
    }
    let ba = NdArray::from_shape_vec(&[128, 8, 8], big_a).unwrap();
    let inv_ba = inv(&ba).unwrap();
    assert_eq!(inv_ba.shape(), &[128, 8, 8]);
    assert!((inv_ba.to_vec()[0] - 0.25).abs() < 1e-12);
}

#[test]
fn test_tensorsolve_tensorinv() {
    let mut a_data = vec![0.0; 16];
    let eye4 = eye_f64(4).to_vec();
    for i in 0..16 {
        a_data[i] = eye4[i] + 0.1 * (i as f64);
    }
    let a_t = NdArray::from_shape_vec(&[2, 2, 2, 2], a_data).unwrap();
    let b_t = arr_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let sol = tensorsolve(&a_t, &b_t, None).unwrap();
    assert_close_f64(&sol, &[0.625, 0.625, 0.625, 0.625], &[2, 2], 1e-12);

    let a_inv = tensorinv(&a_t, 2).unwrap();
    assert_eq!(a_inv.shape(), &[2, 2, 2, 2]);
    let check_id = tensordot(&a_t, &a_inv, 2).unwrap();
    assert_close_f64(&check_id, &eye4, &[2, 2, 2, 2], 1e-12);

    assert!(matches!(tensorinv(&a_t, 0), Err(NumPyError::InvalidValue(_))));
    assert!(matches!(tensorinv(&a_t, 5), Err(NumPyError::InvalidValue(_))));
}

#[test]
fn test_pinv_and_lstsq() {
    // Rank-deficient (3x2) pinv & 4 Moore-Penrose axioms
    let a = arr_f64(&[3, 2], &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
    let ap = pinv(&a, None, false).unwrap();
    assert_eq!(ap.shape(), &[2, 3]);
    let a_ap_a = matmul(&matmul(&a, &ap).unwrap(), &a).unwrap();
    assert_close_f64(&a_ap_a, &a.to_vec(), &[3, 2], 1e-12);
    let ap_a_ap = matmul(&matmul(&ap, &a).unwrap(), &ap).unwrap();
    assert_close_f64(&ap_a_ap, &ap.to_vec(), &[2, 3], 1e-12);

    // Complex Hermitian pinv
    let h = arr_c128(&[2, 2], &[(2.0, 0.0), (0.0, 1.0), (0.0, -1.0), (2.0, 0.0)]);
    let hp = pinv(&h, None, true).unwrap();
    assert_close_c128(&matmul(&h, &hp).unwrap(), &eye_c128(2).to_vec(), &[2, 2], 1e-12);

    // Overdetermined lstsq matching NumPy 2.5.4
    let a_ls = arr_f64(&[3, 2], &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
    let b_ls = arr_f64(&[3], &[1.0, 2.0, 2.0]);
    let ls = lstsq(&a_ls, &b_ls, None).unwrap();
    assert_close_f64(&ls.x, &[2.0 / 3.0, 0.5], &[2], 1e-12);
    assert_close_f64(&ls.residuals, &[1.0 / 6.0], &[1], 1e-12);
    assert_eq!(ls.rank.to_vec(), vec![2]);
    assert_close_f64(&ls.s, &[4.079143328941734, 0.6004912172131635], &[2], 1e-11);

    // Empty (3, 0) lstsq residuals == sum of squares of B columns
    let a0 = arr_f64(&[3, 0], &[]);
    let b0 = arr_f64(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0]);
    let ls0 = lstsq(&a0, &b0, None).unwrap();
    assert_eq!(ls0.x.shape(), &[0, 2]);
    assert_close_f64(&ls0.residuals, &[10.0, 21.0], &[2], 1e-12);
    assert_eq!(ls0.rank.to_vec(), vec![0]);

    // DynArray auto-promotion for solve/inv
    let di = DynArray::I32(NdArray::from_shape_vec(&[2, 2], vec![4, 7, 2, 6]).unwrap());
    let di_inv = inv_dyn(&di).unwrap();
    assert_eq!(di_inv.dtype(), numpy_types::DType::F64);
    if let DynArray::F64(a) = &di_inv {
        assert_close_f64(a, &[0.6, -0.7, -0.2, 0.4], &[2, 2], 1e-12);
    } else {
        panic!("expected F64");
    }
}

// ---------------------------------------------------------------------------
// 3. Determinants, Norms & Rank
// ---------------------------------------------------------------------------

#[test]
fn test_det_slogdet_matrix_rank() {
    let a = arr_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_close_f64(&det(&a).unwrap(), &[-2.0], &[], 1e-12);
    let sl = slogdet(&a).unwrap();
    assert_close_f64(&sl.sign, &[-1.0], &[], 1e-12);
    assert_close_f64(&sl.logabsdet, &[2.0f64.ln()], &[], 1e-12);

    // 0x0 det=1, slogdet=(1, 0); singular=(0, -inf); NaN -> NaN
    let emp = arr_f64(&[0, 0], &[]);
    assert_close_f64(&det(&emp).unwrap(), &[1.0], &[], 1e-12);
    let zero = arr_f64(&[2, 2], &[1.0, 2.0, 2.0, 4.0]);
    let slz = slogdet(&zero).unwrap();
    assert_close_f64(&slz.sign, &[0.0], &[], 1e-12);
    assert_eq!(slz.logabsdet.to_vec()[0], f64::NEG_INFINITY);

    let nans = arr_f64(&[2, 2], &[f64::NAN, 1.0, 2.0, 3.0]);
    let sln = slogdet(&nans).unwrap();
    assert!(sln.logabsdet.to_vec()[0].is_nan());

    // Complex slogdet
    let ca = arr_c128(&[2, 2], &[(1.0, 2.0), (0.0, 1.0), (-1.0, 1.0), (3.0, 0.0)]);
    let cd = det(&ca).unwrap();
    // (1+2j)*3 - 1j*(-1+1j) = (3+6j) - (-1-1j) = (4+7j)
    assert_close_c128(&cd, &[Complex64::new(4.0, 7.0)], &[], 1e-12);

    // matrix_rank
    let rk = arr_f64(&[3, 3], &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0]);
    assert_eq!(matrix_rank(&rk, None, None, false).unwrap().to_vec(), vec![2]);
    assert_eq!(matrix_rank(&arr_f64(&[3], &[0.0, 0.0, 0.0]), None, None, false).unwrap().to_vec(), vec![0]);
    assert_eq!(matrix_rank(&arr_f64(&[3], &[0.0, 2.0, 0.0]), None, None, false).unwrap().to_vec(), vec![1]);
    assert!(matches!(
        matrix_rank(&rk, Some(1e-6), Some(1e-6), false),
        Err(NumPyError::InvalidValue(_))
    ));
}

#[test]
fn test_norms_and_cond() {
    // Vector norms
    let v = arr_f64(&[4], &[-3.0, 0.0, 4.0, 0.0]);
    assert_close_f64(&vector_norm(&v, 0.0, None, false).unwrap(), &[2.0], &[], 1e-12);
    assert_close_f64(&vector_norm(&v, 1.0, None, false).unwrap(), &[7.0], &[], 1e-12);
    assert_close_f64(&vector_norm(&v, 2.0, None, false).unwrap(), &[5.0], &[], 1e-12);
    assert_close_f64(&vector_norm(&v, f64::INFINITY, None, false).unwrap(), &[4.0], &[], 1e-12);
    assert_close_f64(&vector_norm(&v, f64::NEG_INFINITY, None, false).unwrap(), &[0.0], &[], 1e-12);

    let vpos = arr_f64(&[2], &[3.0, 4.0]);
    assert_close_f64(&vector_norm(&vpos, -1.0, None, false).unwrap(), &[12.0 / 7.0], &[], 1e-12);
    assert_close_f64(&vector_norm(&vpos, -2.0, None, false).unwrap(), &[2.4], &[], 1e-12);

    // Empty vector norm edge cases
    let ev = arr_f64(&[0], &[]);
    assert_eq!(vector_norm(&ev, -1.0, None, false).unwrap().to_vec()[0], f64::INFINITY);
    assert!(vector_norm(&ev, f64::NEG_INFINITY, None, false).is_err());

    // Matrix norms on complex 2x3 matrix matched against NumPy 2.5.4
    let m = arr_c128(
        &[2, 3],
        &[
            (1.0, 1.0), (2.0, -1.0), (0.0, 0.0),
            (0.0, -1.0), (3.0, 0.0), (1.0, 2.0),
        ],
    );
    let expected_mn = [
        (NormOrd::Fro, 4.69041575982343),
        (NormOrd::Nuc, 6.279776729642758),
        (NormOrd::PosInf, 6.23606797749979),
        (NormOrd::NegInf, 3.6502815398728847),
        (NormOrd::L1, 5.23606797749979),
        (NormOrd::LNeg1, 2.23606797749979),
        (NormOrd::L2, 4.208111682512642),
        (NormOrd::LNeg2, 2.071665047130116),
    ];
    for &(ord, exp) in &expected_mn {
        assert_close_f64(&matrix_norm(&m, ord, false).unwrap(), &[exp], &[], 1e-11);
    }

    // cond across all 8 NormOrd values matched against NumPy 2.5.4
    let cm = arr_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let expected_cond = [
        (NormOrd::Default, 14.933034373659268),
        (NormOrd::Fro, 14.999999999999998),
        (NormOrd::PosInf, 20.999999999999993),
        (NormOrd::NegInf, 5.999999999999998),
        (NormOrd::L1, 20.999999999999993),
        (NormOrd::LNeg1, 5.999999999999998),
        (NormOrd::L2, 14.933034373659268),
        (NormOrd::LNeg2, 0.06696562634074717),
    ];
    for &(ord, exp) in &expected_cond {
        assert_close_f64(&cond(&cm, ord).unwrap(), &[exp], &[], 1e-11);
    }
    assert!(matches!(cond(&arr_f64(&[0, 0], &[]), NormOrd::Default), Err(NumPyError::LinAlgError(_))));
    assert_eq!(cond(&arr_f64(&[2, 2], &[0.0; 4]), NormOrd::Default).unwrap().to_vec()[0], f64::INFINITY);
    assert!(cond(&arr_f64(&[2, 2], &[f64::NAN, 0.0, 0.0, 1.0]), NormOrd::Default).unwrap().to_vec()[0].is_nan());

    // norm w/ keepdims and axis
    let nres = norm(&cm, NormOrd::Fro, Some(&[0, 1]), true).unwrap();
    assert_close_f64(&nres, &[30.0f64.sqrt()], &[1, 1], 1e-12);
}

// ---------------------------------------------------------------------------
// 4. Decompositions
// ---------------------------------------------------------------------------

#[test]
fn test_cholesky() {
    let spd = arr_f64(&[3, 3], &[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]);
    let l = cholesky(&spd, false).unwrap();
    assert_close_f64(&matmul(&l, &l.transpose(None).unwrap()).unwrap(), &spd.to_vec(), &[3, 3], 1e-11);
    let u = cholesky(&spd, true).unwrap();
    assert_close_f64(&matmul(&u.transpose(None).unwrap(), &u).unwrap(), &spd.to_vec(), &[3, 3], 1e-11);

    // Complex Hermitian positive-definite
    let ch = arr_c128(
        &[2, 2],
        &[(4.0, 0.0), (1.0, -2.0), (1.0, 2.0), (5.0, 0.0)],
    );
    let cl = cholesky(&ch, false).unwrap();
    assert_close_c128(&matmul(&cl, &conj_transpose_c128(&cl)).unwrap(), &ch.to_vec(), &[2, 2], 1e-12);
    let cu = cholesky(&ch, true).unwrap();
    assert_close_c128(&matmul(&conj_transpose_c128(&cu), &cu).unwrap(), &ch.to_vec(), &[2, 2], 1e-12);

    // Non-positive-definite -> LinAlgError
    let npd = arr_f64(&[2, 2], &[1.0, 2.0, 2.0, 1.0]);
    assert!(matches!(cholesky(&npd, false), Err(NumPyError::LinAlgError(_))));
}

#[test]
fn test_qr_all_modes() {
    let a = arr_f64(
        &[4, 3],
        &[1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0],
    );

    if let QrOutput::QR { q, r } = qr(&a, QrMode::Reduced).unwrap() {
        assert_eq!(q.shape(), &[4, 3]);
        assert_eq!(r.shape(), &[3, 3]);
        assert_close_f64(&matmul(&q, &r).unwrap(), &a.to_vec(), &[4, 3], 1e-11);
        assert_close_f64(&matmul(&q.transpose(None).unwrap(), &q).unwrap(), &eye_f64(3).to_vec(), &[3, 3], 1e-11);
    } else {
        panic!("expected QR");
    }

    if let QrOutput::QR { q, r } = qr(&a, QrMode::Complete).unwrap() {
        assert_eq!(q.shape(), &[4, 4]);
        assert_eq!(r.shape(), &[4, 3]);
        assert_close_f64(&matmul(&q, &r).unwrap(), &a.to_vec(), &[4, 3], 1e-11);
        assert_close_f64(&matmul(&q.transpose(None).unwrap(), &q).unwrap(), &eye_f64(4).to_vec(), &[4, 4], 1e-11);
    } else {
        panic!("expected QR");
    }

    if let QrOutput::R { r } = qr(&a, QrMode::R).unwrap() {
        assert_eq!(r.shape(), &[3, 3]);
    } else {
        panic!("expected R");
    }

    if let QrOutput::Raw { h, tau } = qr(&a, QrMode::Raw).unwrap() {
        assert_eq!(h.shape(), &[3, 4]);
        assert_eq!(tau.shape(), &[3]);
        let h_vec = h.to_vec();
        let tau_vec = tau.to_vec();
        // Reconstruct Q_full = H_0 @ H_1 @ H_2 from raw Householder vectors + tau.
        let mut q_rec = eye_f64(4);
        for j in 0..3 {
            let mut v = vec![0.0; 4];
            v[j] = 1.0;
            for r in (j + 1)..4 {
                v[r] = h_vec[j * 4 + r];
            }
            let vv = outer(&arr_f64(&[4], &v), &arr_f64(&[4], &v)).unwrap();
            let scaled: Vec<f64> = eye_f64(4)
                .to_vec()
                .iter()
                .zip(vv.to_vec().iter())
                .map(|(&id, &o)| id - tau_vec[j] * o)
                .collect();
            let hk = arr_f64(&[4, 4], &scaled);
            q_rec = matmul(&q_rec, &hk).unwrap();
        }
        // Check Q_rec^T @ A matches R above diagonal (and zeros below diagonal)
        let r_from_raw = matmul(&q_rec.transpose(None).unwrap(), &a).unwrap();
        let r_v = r_from_raw.to_vec();
        for row in 0..4 {
            for col in 0..row.min(3) {
                assert!(r_v[row * 3 + col].abs() < 1e-11);
            }
        }
        assert_close_f64(&matmul(&q_rec.transpose(None).unwrap(), &q_rec).unwrap(), &eye_f64(4).to_vec(), &[4, 4], 1e-11);
    } else {
        panic!("expected Raw");
    }

    // Complex QR
    let ca = arr_c128(
        &[3, 2],
        &[(1.0, 1.0), (2.0, 0.0), (0.0, -1.0), (1.0, 2.0), (3.0, 0.0), (-1.0, 1.0)],
    );
    if let QrOutput::QR { q, r } = qr(&ca, QrMode::Reduced).unwrap() {
        assert_close_c128(&matmul(&q, &r).unwrap(), &ca.to_vec(), &[3, 2], 1e-11);
        assert_close_c128(&matmul(&conj_transpose_c128(&q), &q).unwrap(), &eye_c128(2).to_vec(), &[2, 2], 1e-11);
    } else {
        panic!("expected QR");
    }
}

#[test]
fn test_svd_eig_eigh_and_dyn_promotion() {
    // SVD full & reduced + hermitian + empty + NaN error
    let a = arr_f64(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let s_full = svd(&a, true, true, false).unwrap();
    let u = s_full.u.unwrap();
    let vh = s_full.vh.unwrap();
    assert_eq!(u.shape(), &[3, 3]);
    assert_eq!(vh.shape(), &[2, 2]);
    let s_vals = s_full.s.to_vec();
    let mut diag_s = vec![0.0; 6];
    diag_s[0] = s_vals[0];
    diag_s[3] = s_vals[1];
    let rec = matmul(&matmul(&u, &arr_f64(&[3, 2], &diag_s)).unwrap(), &vh).unwrap();
    assert_close_f64(&rec, &a.to_vec(), &[3, 2], 1e-11);
    assert!(matches!(
        svd(&arr_f64(&[2, 2], &[1.0, f64::NAN, 0.0, 1.0]), true, false, false),
        Err(NumPyError::LinAlgError(_))
    ));

    // Real eig with complex conjugate pair: rotation 90 deg -> eigenvalues +/- i
    let rot = arr_f64(&[2, 2], &[0.0, -1.0, 1.0, 0.0]);
    let eg = eig(&rot).unwrap();
    let evals = eg.eigenvalues.to_vec();
    let evecs = eg.eigenvectors.to_vec();
    // Verify A @ v_k == lambda_k * v_k and ||v_k|| == 1
    let rot_c = arr_c128(&[2, 2], &[(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]);
    for k in 0..2 {
        let vk = NdArray::from_shape_vec(&[2], vec![evecs[k], evecs[2 + k]]).unwrap();
        let norm_v: f64 = vk.to_vec().iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!((norm_v - 1.0).abs() < 1e-12);
        let lhs = matvec(&rot_c, &vk).unwrap();
        let rhs: Vec<Complex64> = vk.to_vec().into_iter().map(|z| z * evals[k]).collect();
        assert_close_c128(&lhs, &rhs, &[2], 1e-12);
    }

    // Complex eig
    let ceig = arr_c128(&[2, 2], &[(1.0, 2.0), (3.0, -1.0), (0.5, 1.0), (-2.0, 4.0)]);
    let ceg = eig(&ceig).unwrap();
    for k in 0..2 {
        let evs = ceg.eigenvectors.to_vec();
        let vk = NdArray::from_shape_vec(&[2], vec![evs[k], evs[2 + k]]).unwrap();
        let lhs = matvec(&ceig, &vk).unwrap();
        let lam = ceg.eigenvalues.to_vec()[k];
        let rhs: Vec<Complex64> = vk.to_vec().into_iter().map(|z| z * lam).collect();
        assert_close_c128(&lhs, &rhs, &[2], 1e-11);
    }

    // Complex Hermitian eigh Uplo::Lower & Uplo::Upper matched against NumPy 2.5.4
    let h_full = arr_c128(
        &[3, 3],
        &[
            (2.0, 0.0), (1.0, -2.0), (0.5, 0.5),
            (1.0, 2.0), (5.0, 0.0), (0.0, -1.0),
            (0.5, -0.5), (0.0, 1.0), (3.0, 0.0),
        ],
    );
    let eh_l = eigh(&h_full, Uplo::Lower).unwrap();
    let eh_u = eigh(&h_full, Uplo::Upper).unwrap();
    let expected_w = [0.35601362515901697, 3.34381564401749, 6.300170730823494];
    assert_close_f64(&eh_l.eigenvalues, &expected_w, &[3], 1e-11);
    assert_close_f64(&eh_u.eigenvalues, &expected_w, &[3], 1e-11);
    assert_close_c128(
        &matmul(&conj_transpose_c128(&eh_u.eigenvectors), &eh_u.eigenvectors).unwrap(),
        &eye_c128(3).to_vec(),
        &[3, 3],
        1e-11,
    );

    // DynArray auto-promotion of I32, U8, Bool, F16 -> F64 in svdvals_dyn & eigh_dyn
    let d_u8 = DynArray::U8(NdArray::from_shape_vec(&[2, 2], vec![3, 0, 0, 4]).unwrap());
    let s_u8 = svdvals_dyn(&d_u8).unwrap();
    assert_eq!(s_u8.dtype(), numpy_types::DType::F64);
    if let DynArray::F64(a) = &s_u8 {
        assert_close_f64(a, &[4.0, 3.0], &[2], 1e-12);
    } else {
        panic!("expected F64");
    }

    // Batched Rayon SVD (128 x 6x6)
    let mut bdata = vec![0.0f64; 128 * 36];
    for b in 0..128 {
        for i in 0..6 {
            bdata[b * 36 + i * 6 + i] = (i + 1) as f64;
        }
    }
    let b_arr = NdArray::from_shape_vec(&[128, 6, 6], bdata).unwrap();
    let b_s = svdvals(&b_arr).unwrap();
    assert_eq!(b_s.shape(), &[128, 6]);
    assert_close_f64(&b_s.slice(&[SliceArg::Index(5)]).unwrap(), &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[6], 1e-12);
}

#[test]
fn test_kron_overflow_multi_dot_large_cost_and_empty_batch_matrix_norm() {
    // kron shape overflow must return Err instead of wrapping around.
    let huge_a = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1usize << 40])
        .unwrap();
    let huge_b = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1usize << 40])
        .unwrap();
    assert!(matches!(
        kron(&huge_a, &huge_b),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("overflow")
    ));

    let too_big_a = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1usize << 59])
        .unwrap();
    let too_big_b = NdArray::from_vec(vec![1.0f64, 2.0]);
    assert!(matches!(
        kron(&too_big_a, &too_big_b),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));

    // multi_dot u128 cost saturation prevents overflow panic on huge chain dimensions.
    let m0 = NdArray::<f64>::zeros(&[0, 1usize << 58]).unwrap();
    let m1 = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1usize << 58, 1])
        .unwrap();
    let m2 = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1, 1usize << 58])
        .unwrap();
    let m3 = NdArray::<f64>::zeros(&[1usize << 58, 0]).unwrap();
    let chain = multi_dot(&[&m0, &m1, &m2, &m3]).unwrap();
    assert_eq!(chain.shape(), &[0, 0]);

    // 4-matrix chain reaching 2^129 > u128::MAX in DP does not panic on u128 multiplication.
    let o0 = NdArray::<f64>::zeros(&[1usize << 43, 0]).unwrap();
    let o1 = NdArray::<f64>::zeros(&[0, 1usize << 43]).unwrap();
    let o2 = NdArray::<f64>::zeros(&[1usize << 43, 0]).unwrap();
    let o3 = NdArray::<f64>::zeros(&[0, 1usize << 43]).unwrap();
    assert!(multi_dot(&[&o0, &o1, &o2, &o3]).is_err());

    // 3-matrix chain overflow protection (cost_left = 2^116, cost_right = 0)
    let c0 = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1usize << 58, 1])
        .unwrap();
    let c1 = NdArray::from_scalar(1.0f64)
        .broadcast_to(&[1, 1usize << 58])
        .unwrap();
    let c2 = NdArray::<f64>::zeros(&[1usize << 58, 0]).unwrap();
    let chain3 = multi_dot(&[&c0, &c1, &c2]).unwrap();
    assert_eq!(chain3.shape(), &[1usize << 58, 0]);

    // matrix_norm empty-batch validation matches NumPy 2.5.4.
    let empty_023 = NdArray::<f64>::zeros(&[0, 2, 3]).unwrap();
    assert!(matches!(
        matrix_norm(&empty_023, NormOrd::P(3.0), false),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        matrix_norm(&empty_023, NormOrd::Zero, false),
        Err(NumPyError::InvalidValue(_))
    ));
    assert_eq!(
        matrix_norm(&empty_023, NormOrd::L1, false).unwrap().shape(),
        &[0]
    );
    assert_eq!(
        matrix_norm(&empty_023, NormOrd::LNeg2, false).unwrap().shape(),
        &[0]
    );

    let empty_003 = NdArray::<f64>::zeros(&[0, 0, 3]).unwrap();
    assert_eq!(
        matrix_norm(&empty_003, NormOrd::LNeg1, false).unwrap().shape(),
        &[0]
    );
    assert!(matches!(
        matrix_norm(&empty_003, NormOrd::NegInf, false),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        matrix_norm(&empty_003, NormOrd::LNeg2, false),
        Err(NumPyError::InvalidValue(_))
    ));

    let empty_020 = NdArray::<f64>::zeros(&[0, 2, 0]).unwrap();
    assert_eq!(
        matrix_norm(&empty_020, NormOrd::NegInf, false).unwrap().shape(),
        &[0]
    );
    assert!(matches!(
        matrix_norm(&empty_020, NormOrd::LNeg1, false),
        Err(NumPyError::InvalidValue(_))
    ));

    let empty_220 = NdArray::<f64>::zeros(&[2, 2, 0]).unwrap();
    assert_eq!(
        matrix_norm(&empty_220, NormOrd::NegInf, false).unwrap().to_vec(),
        vec![0.0, 0.0]
    );
    assert!(matches!(
        matrix_norm(&empty_220, NormOrd::LNeg1, false),
        Err(NumPyError::InvalidValue(_))
    ));

    let empty_203 = NdArray::<f64>::zeros(&[2, 0, 3]).unwrap();
    assert_eq!(
        matrix_norm(&empty_203, NormOrd::LNeg1, false).unwrap().to_vec(),
        vec![0.0, 0.0]
    );
    assert!(matches!(
        matrix_norm(&empty_203, NormOrd::NegInf, false),
        Err(NumPyError::InvalidValue(_))
    ));
}

#[test]
fn test_round2_linalg_cross_bool_slogdet_inf_and_cholesky_nonfinite() {
    // Bug #7: trace_axes with offset = isize::MIN does not panic.
    let m22 = arr_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(trace(&m22, isize::MIN).unwrap().shape(), &[] as &[usize]);
    assert_eq!(trace(&m22, isize::MIN).unwrap().to_vec(), vec![0.0]);

    // Bug #23: cross and cross_dyn reject boolean dtype.
    let b1 = NdArray::from_vec(vec![true, false, true]);
    let b2 = NdArray::from_vec(vec![false, true, true]);
    assert!(matches!(cross(&b1, &b2, -1), Err(NumPyError::DTypeError(_))));
    assert!(matches!(
        cross_dyn(&DynArray::Bool(b1), &DynArray::Bool(b2), -1),
        Err(NumPyError::DTypeError(_))
    ));

    // Bug #24: complex slogdet with infinite entry and finite huge magnitude.
    let c_inf = arr_c128(&[2, 2], &[(f64::INFINITY, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]);
    let sd_inf = slogdet(&c_inf).unwrap();
    let s_val = sd_inf.sign.to_vec()[0];
    assert!(s_val.re.is_nan() && s_val.im.is_nan());
    assert_eq!(sd_inf.logabsdet.to_vec()[0], f64::INFINITY);

    let c_big = arr_c128(&[2, 2], &[(1e308, 1e308), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]);
    let sd_big = slogdet(&c_big).unwrap();
    let sb_val = sd_big.sign.to_vec()[0];
    assert!((sb_val.re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    assert!((sb_val.im - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    assert!((sd_big.logabsdet.to_vec()[0] - (1e308f64.ln() + 2.0f64.sqrt().ln())).abs() < 1e-10);

    // Bug #24: cholesky propagates NaN/+Inf like LAPACK/NumPy while rejecting negative-definite.
    let m_nan = arr_f64(&[2, 2], &[f64::NAN, 0.0, 0.0, 1.0]);
    let l_nan = cholesky(&m_nan, false).unwrap().to_vec();
    assert!(l_nan[0].is_nan() && l_nan[1] == 0.0 && l_nan[2].is_nan() && l_nan[3].is_nan());

    let m_off_nan = arr_f64(&[2, 2], &[1.0, f64::NAN, f64::NAN, 1.0]);
    let l_off = cholesky(&m_off_nan, false).unwrap().to_vec();
    assert!(l_off[0] == 1.0 && l_off[1] == 0.0 && l_off[2].is_nan() && l_off[3].is_nan());

    let m_inf = arr_f64(&[2, 2], &[f64::INFINITY, 0.0, 0.0, 1.0]);
    assert_eq!(cholesky(&m_inf, false).unwrap().to_vec(), vec![f64::INFINITY, 0.0, 0.0, 1.0]);

    let m_3nan = arr_f64(&[3, 3], &[2.0, 1.0, 0.0, 1.0, f64::NAN, 0.0, 0.0, 0.0, 1.0]);
    let l_3 = cholesky(&m_3nan, false).unwrap().to_vec();
    assert!((l_3[0] - 2.0f64.sqrt()).abs() < 1e-12);
    assert!((l_3[3] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    assert!(l_3[4].is_nan() && l_3[7].is_nan() && l_3[8].is_nan());

    let m_neg = arr_f64(&[2, 2], &[-1.0, 0.0, 0.0, 1.0]);
    assert!(matches!(cholesky(&m_neg, false), Err(NumPyError::LinAlgError(_))));
}

#[test]
fn test_cross_huge_broadcast_returns_err_instead_of_panicking() {
    let a = NdArray::from_vec(vec![1.0f64, 2.0, 3.0])
        .broadcast_to(&[1usize << 57, 1, 3])
        .unwrap();
    let b = NdArray::from_vec(vec![4.0f64, 5.0, 6.0])
        .broadcast_to(&[1, 4, 3])
        .unwrap();
    assert!(matches!(
        cross(&a, &b, -1),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));

    let au = NdArray::from_vec(vec![1u8, 2, 3])
        .broadcast_to(&[1usize << 61, 1, 3])
        .unwrap();
    let bu = NdArray::from_vec(vec![4u8, 5, 6])
        .broadcast_to(&[1, 2, 3])
        .unwrap();
    assert!(matches!(cross(&au, &bu, -1), Err(NumPyError::ShapeMismatch(_))));
}

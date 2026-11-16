use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{DType, DynArray, NdArray, NumPyError, Order, Slice, SliceArg};
use numpy_ufunc::{BinaryOp, ScalarValue, UfuncArray, UnaryOp};

#[test]
fn test_i64_simd_max_min_reductions_and_extremes() {
    for len in [1usize, 3, 4, 5, 8, 9, 17, 100, 40_000] {
        let mut data: Vec<i64> = (0..len).map(|i| ((i * 37) % 1000) as i64 - 500).collect();
        data[0] = i64::MIN;
        if len > 1 {
            data[len - 1] = i64::MAX;
        }
        let arr = NdArray::from_vec(data.clone());
        let expected_max = *data.iter().max().unwrap();
        let expected_min = *data.iter().min().unwrap();

        assert_eq!(
            BinaryOp::Maximum
                .reduce_nd(&arr, None, false, None)
                .unwrap()
                .to_vec(),
            vec![expected_max]
        );
        assert_eq!(
            BinaryOp::Fmax
                .reduce_nd(&arr, None, false, None)
                .unwrap()
                .to_vec(),
            vec![expected_max]
        );
        assert_eq!(
            BinaryOp::Minimum
                .reduce_nd(&arr, None, false, None)
                .unwrap()
                .to_vec(),
            vec![expected_min]
        );
        assert_eq!(
            BinaryOp::Fmin
                .reduce_nd(&arr, None, false, None)
                .unwrap()
                .to_vec(),
            vec![expected_min]
        );
    }

    let empty = NdArray::<i64>::zeros(&[0]).unwrap();
    assert!(BinaryOp::Maximum.reduce_nd(&empty, None, false, None).is_err());
    assert!(BinaryOp::Minimum.reduce_nd(&empty, None, false, None).is_err());
}

#[test]
fn test_validate_comp_dtype_and_no_unreachable_panics() {
    let bool_arr = NdArray::from_vec(vec![true, false, true]);
    assert!(matches!(
        BinaryOp::Power.reduce_nd(&bool_arr, None, false, None),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        BinaryOp::Power.accumulate_nd(&bool_arr, 0),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        BinaryOp::Power.outer_nd(&bool_arr, &bool_arr),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        BinaryOp::Power.reduce(&DynArray::Bool(bool_arr.clone()), None, Some(DType::Bool), false, None),
        Err(NumPyError::DTypeError(_))
    ));
    let dyn_pow = BinaryOp::Power
        .reduce(&DynArray::Bool(bool_arr.clone()), None, None, false, None)
        .unwrap();
    assert_eq!(dyn_pow.dtype(), DType::I8);

    // Logical ops require boolean dtype for reduce/accumulate/outer_nd, while DynArray auto-casts to bool when dtype=None.
    let int_arr = NdArray::from_vec(vec![1i64, 0, 2]);
    let int_empty = NdArray::<i64>::zeros(&[0]).unwrap();
    for op in [BinaryOp::LogicalAnd, BinaryOp::LogicalOr, BinaryOp::LogicalXor] {
        assert!(matches!(
            op.reduce_nd(&int_arr, None, false, None),
            Err(NumPyError::DTypeError(_))
        ));
        assert!(matches!(
            op.reduce_nd(&int_empty, None, false, None),
            Err(NumPyError::DTypeError(_))
        ));
        assert!(matches!(
            op.accumulate_nd(&int_arr, 0),
            Err(NumPyError::DTypeError(_))
        ));
        assert!(matches!(
            op.outer_nd(&int_arr, &int_arr),
            Err(NumPyError::DTypeError(_))
        ));
        assert!(matches!(
            op.reduce(&DynArray::I64(int_arr.clone()), None, Some(DType::I64), false, None),
            Err(NumPyError::DTypeError(_))
        ));
        let ok = op
            .reduce(&DynArray::I64(int_arr.clone()), None, None, false, None)
            .unwrap();
        assert_eq!(ok.dtype(), DType::Bool);
    }

    // Power.at_nd on bool: [0**0=1, 0**1=0, 1**0=1, 1**1=1]
    let mut b_out = NdArray::from_vec(vec![false, false, true, true]);
    let b_exp = NdArray::from_vec(vec![false, true, false, true]);
    let idx = NdArray::from_vec(vec![0i64, 1, 2, 3]);
    BinaryOp::Power.at_nd(&mut b_out, &idx, &b_exp).unwrap();
    assert_eq!(b_out.to_vec(), vec![true, false, true, true]);

    // Logical ops at_nd on non-bool types return 0/1 cleanly without panic.
    let mut i_out = NdArray::from_vec(vec![5i64, 0, 3]);
    let i_rhs = NdArray::from_vec(vec![0i64, 7, 9]);
    let idx3 = NdArray::from_vec(vec![0i64, 1, 2]);
    BinaryOp::LogicalAnd.at_nd(&mut i_out, &idx3, &i_rhs).unwrap();
    assert_eq!(i_out.to_vec(), vec![0, 0, 1]);

    let mut f_out = NdArray::from_vec(vec![0.0f64, 2.0, 0.0]);
    let f_rhs = NdArray::from_vec(vec![0.0f64, 0.0, f64::NAN]);
    BinaryOp::LogicalOr.at_nd(&mut f_out, &idx3, &f_rhs).unwrap();
    assert_eq!(f_out.to_vec(), vec![0.0, 1.0, 1.0]);

    let mut c_out = NdArray::from_vec(vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)]);
    let c_rhs = NdArray::from_vec(vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)]);
    let idx2 = NdArray::from_vec(vec![0i64, 1]);
    BinaryOp::LogicalXor.at_nd(&mut c_out, &idx2, &c_rhs).unwrap();
    assert_eq!(
        c_out.to_vec(),
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    );

    // UnaryOp::Real, Imag, Abs at_nd on bool, int, f16, complex without panic.
    let mut b_u = NdArray::from_vec(vec![true, false]);
    UnaryOp::Real.at_nd(&mut b_u, &idx2).unwrap();
    assert_eq!(b_u.to_vec(), vec![true, false]);
    UnaryOp::Imag.at_nd(&mut b_u, &idx2).unwrap();
    assert_eq!(b_u.to_vec(), vec![false, false]);

    let mut i_u = NdArray::from_vec(vec![7i32, -3]);
    UnaryOp::Real.at_nd(&mut i_u, &idx2).unwrap();
    assert_eq!(i_u.to_vec(), vec![7, -3]);
    UnaryOp::Imag.at_nd(&mut i_u, &idx2).unwrap();
    assert_eq!(i_u.to_vec(), vec![0, 0]);

    let mut h_u = NdArray::from_vec(vec![f16::from_f32(1.5), f16::from_f32(-2.5)]);
    UnaryOp::Real.at_nd(&mut h_u, &idx2).unwrap();
    assert_eq!(h_u.to_vec(), vec![f16::from_f32(1.5), f16::from_f32(-2.5)]);
    UnaryOp::Imag.at_nd(&mut h_u, &idx2).unwrap();
    assert_eq!(h_u.to_vec(), vec![f16::ZERO, f16::ZERO]);

    let mut c32_u = NdArray::from_vec(vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, -5.0)]);
    UnaryOp::Real.at_nd(&mut c32_u, &[0i64]).unwrap();
    assert_eq!(c32_u.to_vec()[0], Complex32::new(3.0, 0.0));
    UnaryOp::Imag.at_nd(&mut c32_u, &[1i64]).unwrap();
    assert_eq!(c32_u.to_vec()[1], Complex32::new(-5.0, 0.0));
    let mut c64_u = NdArray::from_vec(vec![Complex64::new(3.0, 4.0)]);
    UnaryOp::Abs.at_nd(&mut c64_u, &[0i64]).unwrap();
    assert_eq!(c64_u.to_vec()[0], Complex64::new(5.0, 0.0));

    // UfuncArray::ua_frexp returns DTypeError cleanly on NdArray<T>.
    assert!(matches!(
        NdArray::<i32>::zeros(&[4]).unwrap().ua_frexp(),
        Err(NumPyError::DTypeError(_))
    ));

    // reduce with empty axes and initial scalar does not panic.
    let scalar = NdArray::from_scalar(5i64);
    let r0 = BinaryOp::Add
        .reduce(&DynArray::I64(scalar), None, None, false, Some(ScalarValue::I64(10)))
        .unwrap();
    match r0 {
        DynArray::I64(v) => assert_eq!(v.to_vec(), vec![15]),
        other => panic!("expected I64, got {other:?}"),
    }
}

#[test]
fn test_at_nd_typed_index_arrays_contiguous_and_strided() {
    let base = NdArray::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);

    // &NdArray<i64> non-contiguous (flipped step -2)
    let raw_i64 = NdArray::from_vec(vec![4i64, 99, 0, 99, 0]);
    let idx_i64 = raw_i64
        .slice(&[SliceArg::Slice(Slice::full().step_by(-2))])
        .unwrap(); // [0, 0, 4]
    let mut out = base.copy(Order::C);
    BinaryOp::Add
        .at_nd(&mut out, &idx_i64, &NdArray::from_scalar(10.0))
        .unwrap();
    assert_eq!(out.to_vec(), vec![21.0, 2.0, 3.0, 4.0, 15.0]);

    // &NdArray<i32>
    let idx_i32 = NdArray::from_vec(vec![-1i32, 1, -1]);
    let mut out = base.copy(Order::C);
    BinaryOp::Add
        .at_nd(&mut out, &idx_i32, &NdArray::from_scalar(1.0))
        .unwrap();
    assert_eq!(out.to_vec(), vec![1.0, 3.0, 3.0, 4.0, 7.0]);

    // &NdArray<isize> non-contiguous (transposed 2x2)
    let mat_isize = NdArray::from_shape_vec(&[2, 2], vec![0isize, 2, 1, 2]).unwrap().t();
    let mut out = base.copy(Order::C);
    BinaryOp::Add
        .at_nd(&mut out, &mat_isize, &NdArray::from_scalar(10.0))
        .unwrap();
    assert_eq!(out.to_vec(), vec![11.0, 12.0, 23.0, 4.0, 5.0]);

    // &NdArray<usize>
    let idx_usize = NdArray::from_vec(vec![2usize, 2]);
    let mut out = base.copy(Order::C);
    BinaryOp::Add
        .at_nd(&mut out, &idx_usize, &NdArray::from_scalar(5.0))
        .unwrap();
    assert_eq!(out.to_vec(), vec![1.0, 2.0, 13.0, 4.0, 5.0]);

    // &NdArray<bool> non-contiguous (flipped mask)
    let raw_mask = NdArray::from_vec(vec![true, false, false, false, true]);
    let mask = raw_mask.flip(None).unwrap();
    let mut out = base.copy(Order::C);
    BinaryOp::Multiply
        .at_nd(&mut out, &mask, &NdArray::from_scalar(2.0))
        .unwrap();
    assert_eq!(out.to_vec(), vec![2.0, 2.0, 3.0, 4.0, 10.0]);
}

#[test]
fn test_round2_clip_inverted_bounds_isclose_nan_and_round_extreme() {
    use numpy_ufunc::{clip_scalar, isclose, round};

    // Bug #20: clip with lo > hi consistently yields hi across SIMD and scalar tails.
    for len in [3usize, 17, 65] {
        let vals_f64: Vec<f64> = (0..len).map(|i| (i as f64) - 2.0).collect();
        let a_f64 = NdArray::from_vec(vals_f64);
        let c_f64 = clip_scalar(&a_f64, Some(10.0), Some(5.0)).unwrap();
        assert!(c_f64.to_vec().iter().all(|&v| v == 5.0));

        let vals_f32: Vec<f32> = (0..len).map(|i| (i as f32) - 2.0).collect();
        let a_f32 = NdArray::from_vec(vals_f32);
        let c_f32 = clip_scalar(&a_f32, Some(10.0), Some(5.0)).unwrap();
        assert!(c_f32.to_vec().iter().all(|&v| v == 5.0));

        let vals_i32: Vec<i32> = (0..len).map(|i| (i as i32) - 2).collect();
        let a_i32 = NdArray::from_vec(vals_i32);
        let c_i32 = clip_scalar(&a_i32, Some(10), Some(5)).unwrap();
        assert!(c_i32.to_vec().iter().all(|&v| v == 5));

        let vals_i64: Vec<i64> = (0..len).map(|i| (i as i64) - 2).collect();
        let a_i64 = NdArray::from_vec(vals_i64);
        let c_i64 = clip_scalar(&a_i64, Some(10), Some(5)).unwrap();
        assert!(c_i64.to_vec().iter().all(|&v| v == 5));
    }
    let a_f16 = NdArray::from_vec(vec![f16::from_f32(1.0), f16::from_f32(7.0)]);
    let c_f16 = clip_scalar(&a_f16, Some(f16::from_f32(10.0)), Some(f16::from_f32(5.0))).unwrap();
    assert_eq!(c_f16.to_vec(), vec![f16::from_f32(5.0); 2]);
    let a_c = NdArray::from_vec(vec![Complex64::new(1.0, 1.0), Complex64::new(10.0, 0.0)]);
    let c_c = clip_scalar(&a_c, Some(Complex64::new(5.0, 0.0)), Some(Complex64::new(2.0, 0.0))).unwrap();
    assert_eq!(c_c.to_vec(), vec![Complex64::new(2.0, 0.0); 2]);

    // Bug #21: isclose returns true for x == y even when rtol or atol is NaN (SIMD + tails).
    for len in [3usize, 17, 65] {
        let xf64 = NdArray::from_vec(vec![1.0f64; len]);
        assert!(isclose(&xf64, &xf64, f64::NAN, 1e-8, false)
            .unwrap()
            .to_vec()
            .iter()
            .all(|&b| b));
        assert!(isclose(&xf64, &xf64, 1e-5, f64::NAN, false)
            .unwrap()
            .to_vec()
            .iter()
            .all(|&b| b));
        let xf32 = NdArray::from_vec(vec![1.0f32; len]);
        assert!(isclose(&xf32, &xf32, f64::NAN, 1e-8, false)
            .unwrap()
            .to_vec()
            .iter()
            .all(|&b| b));
    }

    // Bug #22: round with extreme positive/negative decimals overflows scale to inf -> NaN.
    let rf64 = NdArray::from_vec(vec![1.23, -0.0, f64::INFINITY, f64::NAN]);
    assert!(round(&rf64, 400).unwrap().to_vec().iter().all(|v| v.is_nan()));
    assert!(round(&rf64, -400).unwrap().to_vec().iter().all(|v| v.is_nan()));
    assert!(round(&rf64, 309).unwrap().to_vec().iter().all(|v| v.is_nan()));
    let rf32 = NdArray::from_vec(vec![1.23f32, -0.0, f32::INFINITY]);
    assert!(round(&rf32, 39).unwrap().to_vec().iter().all(|v| v.is_nan()));
    assert!(round(&rf32, -39).unwrap().to_vec().iter().all(|v| v.is_nan()));
    let rf16 = NdArray::from_vec(vec![f16::from_f32(1.23), f16::from_f32(0.0)]);
    assert!(round(&rf16, 5).unwrap().to_vec().iter().all(|v| v.is_nan()));
    assert!(round(&rf16, -5).unwrap().to_vec().iter().all(|v| v.is_nan()));
}

#[test]
fn test_at_tuple_scalar_bool_mask_regression() {
    let f0 = DynArray::from(NdArray::from_scalar(false));
    let t0 = DynArray::from(NdArray::from_scalar(true));
    let idx = DynArray::from(NdArray::from_vec(vec![0i64, 1]));

    // (False, [0, 1]): False covers 0 axes -> shape [0] vs [2] fails broadcast cleanly.
    let mut a_err = DynArray::from(NdArray::<f64>::zeros(&[2, 3]).unwrap());
    let res = BinaryOp::Add.at(
        &mut a_err,
        numpy_ufunc::AtIndex::Tuple(&[&f0, &idx]),
        &DynArray::from(NdArray::from_scalar(1.0f64)),
    );
    assert!(res.is_err());
    let DynArray::F64(ref a_err_inner) = a_err else { panic!() };
    assert_eq!(a_err_inner.to_vec(), vec![0.0; 6]);

    // (False,): value shape [0, 2, 3] performs 0 updates cleanly without OutOfBounds.
    let mut a_f = DynArray::from(NdArray::<f64>::zeros(&[2, 3]).unwrap());
    BinaryOp::Add
        .at(
            &mut a_f,
            numpy_ufunc::AtIndex::Tuple(&[&f0]),
            &DynArray::from(NdArray::<f64>::zeros(&[0, 2, 3]).unwrap()),
        )
        .unwrap();
    let DynArray::F64(ref a_f_inner) = a_f else { panic!() };
    assert_eq!(a_f_inner.to_vec(), vec![0.0; 6]);

    // (True,): value shape [1, 2, 3] adds slice across entire target.
    let mut a_t = DynArray::from(NdArray::<f64>::zeros(&[2, 3]).unwrap());
    let val_t = NdArray::from_shape_vec(&[1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    BinaryOp::Add
        .at(
            &mut a_t,
            numpy_ufunc::AtIndex::Tuple(&[&t0]),
            &DynArray::from(val_t),
        )
        .unwrap();
    let DynArray::F64(ref a_t_inner) = a_t else { panic!() };
    assert_eq!(a_t_inner.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // (True, [0, 2]) on [3, 4]: True consumes 0 axes -> [0, 2] selects rows 0 and 2.
    let mut a_t2 = DynArray::from(NdArray::<f64>::zeros(&[3, 4]).unwrap());
    let idx02 = DynArray::from(NdArray::from_vec(vec![0i64, 2]));
    BinaryOp::Add
        .at(
            &mut a_t2,
            numpy_ufunc::AtIndex::Tuple(&[&t0, &idx02]),
            &DynArray::from(NdArray::from_scalar(1.0f64)),
        )
        .unwrap();
    let DynArray::F64(ref a_t2_inner) = a_t2 else { panic!() };
    assert_eq!(
        a_t2_inner.to_vec(),
        vec![
            1.0, 1.0, 1.0, 1.0, // row 0
            0.0, 0.0, 0.0, 0.0, // row 1
            1.0, 1.0, 1.0, 1.0, // row 2
        ]
    );
}

#[test]
fn test_complex_clip_nan_parity_regression() {
    use numpy_types::{Complex32, Complex64};

    let c64 = NdArray::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(1.0, f64::NAN),
        Complex64::new(2.0, 0.0),
    ]);

    let r1 = numpy_ufunc::clip_scalar(
        &c64,
        Some(Complex64::new(f64::NAN, 0.0)),
        Some(Complex64::new(5.0, 0.0)),
    )
    .unwrap()
    .to_vec();
    assert!(r1[0].re.is_nan() && r1[0].im == 0.0);
    assert!(r1[1].re.is_nan() && r1[1].im == 0.0);
    assert_eq!(r1[2].re, 1.0);
    assert!(r1[2].im.is_nan());
    assert!(r1[3].re.is_nan() && r1[3].im == 0.0);

    let r2 = numpy_ufunc::clip_scalar(
        &c64,
        Some(Complex64::new(0.0, 0.0)),
        Some(Complex64::new(f64::NAN, 0.0)),
    )
    .unwrap()
    .to_vec();
    assert!(r2[0].re.is_nan() && r2[0].im == 0.0);
    assert!(r2[1].re.is_nan() && r2[1].im == 0.0);
    assert_eq!(r2[2].re, 1.0);
    assert!(r2[2].im.is_nan());
    assert!(r2[3].re.is_nan() && r2[3].im == 0.0);

    let r3 = numpy_ufunc::clip_scalar(
        &c64,
        Some(Complex64::new(0.0, f64::NAN)),
        Some(Complex64::new(5.0, 0.0)),
    )
    .unwrap()
    .to_vec();
    assert_eq!(r3[0], Complex64::new(1.0, 0.0));
    assert_eq!(r3[1], Complex64::new(1.0, 2.0));
    assert_eq!(r3[2].re, 1.0);
    assert!(r3[2].im.is_nan());
    assert_eq!(r3[3], Complex64::new(2.0, 0.0));

    let r4 = numpy_ufunc::clip_scalar(
        &c64,
        Some(Complex64::new(0.0, 0.0)),
        Some(Complex64::new(0.0, f64::NAN)),
    )
    .unwrap()
    .to_vec();
    assert!(r4[0].re == 0.0 && r4[0].im.is_nan());
    assert!(r4[1].re == 0.0 && r4[1].im.is_nan());
    assert_eq!(r4[2].re, 1.0);
    assert!(r4[2].im.is_nan());
    assert!(r4[3].re == 0.0 && r4[3].im.is_nan());

    // Array bounds version & single-bound Complex64 parity.
    let lo_arr = NdArray::from_scalar(Complex64::new(0.0, f64::NAN));
    let hi_arr = NdArray::from_scalar(Complex64::new(5.0, 0.0));
    let ra = numpy_ufunc::clip(&c64, Some(&lo_arr), Some(&hi_arr))
        .unwrap()
        .to_vec();
    assert_eq!(ra[0], Complex64::new(1.0, 0.0));
    assert_eq!(ra[1], Complex64::new(1.0, 2.0));
    assert_eq!(ra[2].re, 1.0);
    assert!(ra[2].im.is_nan());
    assert_eq!(ra[3], Complex64::new(2.0, 0.0));

    let rs_lo = numpy_ufunc::clip_scalar(&c64, Some(Complex64::new(0.0, f64::NAN)), None)
        .unwrap()
        .to_vec();
    assert!(rs_lo[0].re == 0.0 && rs_lo[0].im.is_nan());
    assert!(rs_lo[1].re == 0.0 && rs_lo[1].im.is_nan());
    assert_eq!(rs_lo[2].re, 1.0);
    assert!(rs_lo[2].im.is_nan());

    // Complex32 coverage too.
    let c32 = NdArray::from_vec(vec![Complex32::new(1.0, 2.0), Complex32::new(1.0, f32::NAN)]);
    let r32 = numpy_ufunc::clip_scalar(
        &c32,
        Some(Complex32::new(0.0, f32::NAN)),
        Some(Complex32::new(5.0, 0.0)),
    )
    .unwrap()
    .to_vec();
    assert_eq!(r32[0], Complex32::new(1.0, 2.0));
    assert_eq!(r32[1].re, 1.0);
    assert!(r32[1].im.is_nan());
}

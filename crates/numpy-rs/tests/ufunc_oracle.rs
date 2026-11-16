#![cfg(feature = "ufunc")]

mod common;

use common::{
    assert_dyn_matches_oracle, edge_c128_values, edge_f32_values, edge_f64_values,
    non_contiguous_2d_views, py_assign_dyn, run_oracle,
};
use numpy_types::{f16, Complex64, DType, DynArray, NdArray, Order};
use numpy_ufunc::{
    abs, accumulate, acos, acosh, add, allclose, angle, arccos, arcsin, arctan, arctan2,
    array_equal, array_equiv, asin, asinh, at, atan, atanh, bitwise_and, bitwise_count, bitwise_or,
    bitwise_xor, cbrt, ceil, clip, conj, copysign, cos, cosh, deg2rad, divmod, equal, exp, exp2,
    expm1, fabs, floor, floor_divide, fmax, fmin, fmod, frexp, gcd, greater, greater_equal,
    heaviside, hypot, i0, imag, invert, isclose, isfinite, isinf, isnan, isneginf, isposinf, lcm,
    ldexp, left_shift, less, less_equal, log, log10, log1p, log2, logaddexp, logaddexp2,
    logical_and, logical_not, logical_or, logical_xor, maximum, minimum, modf, multiply,
    nan_to_num, negative, nextafter, not_equal, outer, positive, power, rad2deg, real, reciprocal,
    reduce, reduce_axes, remainder, right_shift, rint, round, sign, signbit, sin, sinc, sinh,
    spacing, sqrt, square, subtract, tan, tanh, true_divide, trunc, unary_at, BinaryOp,
    ScalarValue, UnaryOp,
};

#[test]
fn unary_ufuncs_edge_values_and_strided_views_against_numpy() {
    let edges = edge_f64_values();
    let a = NdArray::from_vec(edges.clone());
    // General finite test matrix (non-contiguous views) + positive matrix for log/sqrt/acosh
    let mat = NdArray::from_shape_vec(
        &[3, 4],
        vec![-1.25, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 1.5, -2.0, 2.0, 3.0, -3.5],
    )
    .unwrap();
    let pos = NdArray::from_shape_vec(
        &[3, 4],
        vec![0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0],
    )
    .unwrap();
    let big = NdArray::from_vec(vec![1.0, 1.5, 2.5, 10.0, -1.5, -2.5]);

    let py = format!(
        "{}\n{}\n{}\n{}\n",
        // General edge/pos/mat setup
        py_assign_dyn("e", &DynArray::F64(a.clone())),
        py_assign_dyn("m", &DynArray::F64(mat.clone())),
        py_assign_dyn("p", &DynArray::F64(pos.clone())),
        py_assign_dyn("b", &DynArray::F64(big.slice(&[numpy_types::SliceArg::Slice(numpy_types::Slice::new(None, Some(4), 1))]).unwrap())),
    ) + r#"
for name in ['negative','positive','abs','fabs','square','reciprocal','cbrt','exp','exp2','expm1',
             'sin','cos','tan','sinh','cosh','tanh','arcsinh','floor','ceil','trunc','rint','sign',
             'signbit','spacing','isfinite','isinf','isnan','isneginf','isposinf','logical_not',
             'deg2rad','rad2deg','sinc','i0']:
    emit_arr('edge_' + name, getattr(np, name)(e))
for name in ['sqrt','log','log2','log10','log1p']:
    emit_arr('pos_' + name, getattr(np, name)(p))
for name in ['arcsin','arccos','arctan','arctanh']:
    emit_arr('unit_' + name, getattr(np, name)(np.clip(m, -0.95, 0.95)))
emit_arr('big_acosh', np.arccosh(b))
emit_arr('round_pos', np.round(m, 1))
emit_arr('round_neg', np.round(np.array([15.0, 25.0, 123.45]), -1))
emit_arr('nan_to_num', np.nan_to_num(e, nan=-99.0, posinf=123.0, neginf=-123.0))
emit_arr('frexp_m', np.frexp(e)[0])
emit_arr('frexp_e', np.frexp(e[np.isfinite(e)])[1])
emit_arr('modf_f', np.modf(e)[0])
emit_arr('modf_i', np.modf(e)[1])
"#;
    let oracle = run_oracle(&py);

    let pairs: [(&str, DynArray, f64); 34] = [
        ("edge_negative", negative(&a).unwrap().into(), 0.0),
        ("edge_positive", positive(&a).unwrap().into(), 0.0),
        ("edge_abs", abs(&a).unwrap().into(), 0.0),
        ("edge_fabs", fabs(&a).unwrap().into(), 0.0),
        ("edge_square", square(&a).unwrap().into(), 1e-14),
        ("edge_reciprocal", reciprocal(&a).unwrap().into(), 1e-14),
        ("edge_cbrt", cbrt(&a).unwrap().into(), 1e-14),
        ("edge_exp", exp(&a).unwrap().into(), 1e-13),
        ("edge_exp2", exp2(&a).unwrap().into(), 1e-13),
        ("edge_expm1", expm1(&a).unwrap().into(), 1e-13),
        ("edge_sin", sin(&a).unwrap().into(), 1e-14),
        ("edge_cos", cos(&a).unwrap().into(), 1e-14),
        ("edge_tan", tan(&a).unwrap().into(), 1e-13),
        ("edge_sinh", sinh(&a).unwrap().into(), 1e-13),
        ("edge_cosh", cosh(&a).unwrap().into(), 1e-13),
        ("edge_tanh", tanh(&a).unwrap().into(), 1e-14),
        ("edge_arcsinh", asinh(&a).unwrap().into(), 1e-13),
        ("edge_floor", floor(&a).unwrap().into(), 0.0),
        ("edge_ceil", ceil(&a).unwrap().into(), 0.0),
        ("edge_trunc", trunc(&a).unwrap().into(), 0.0),
        ("edge_rint", rint(&a).unwrap().into(), 0.0),
        ("edge_sign", sign(&a).unwrap().into(), 0.0),
        ("edge_signbit", signbit(&a).unwrap().into(), 0.0),
        ("edge_spacing", spacing(&a).unwrap().into(), 0.0),
        ("edge_isfinite", isfinite(&a).unwrap().into(), 0.0),
        ("edge_isinf", isinf(&a).unwrap().into(), 0.0),
        ("edge_isnan", isnan(&a).unwrap().into(), 0.0),
        ("edge_isneginf", isneginf(&a).unwrap().into(), 0.0),
        ("edge_isposinf", isposinf(&a).unwrap().into(), 0.0),
        ("edge_logical_not", logical_not(&a).unwrap().into(), 0.0),
        ("edge_deg2rad", deg2rad(&a).unwrap().into(), 1e-14),
        ("edge_rad2deg", rad2deg(&a).unwrap().into(), 1e-14),
        ("edge_sinc", sinc(&a).unwrap().into(), 1e-13),
        ("edge_i0", i0(&a).unwrap().into(), 1e-11),
    ];
    for (tag, res, tol) in pairs {
        assert_dyn_matches_oracle(&res, oracle.arr(tag), tol, tol, true, false, tag);
    }

    // Also test non-contiguous views on log/sqrt/trig
    for (layout, view) in non_contiguous_2d_views(&pos) {
        let p_py = format!(
            "{}\nemit_arr('sqrt', np.sqrt(p))\nemit_arr('log', np.log(p))\nemit_arr('exp', np.exp(p))\n",
            py_assign_dyn("p", &DynArray::F64(view.clone()))
        );
        let po = run_oracle(&p_py);
        assert_dyn_matches_oracle(&sqrt(&view).unwrap().into(), po.arr("sqrt"), 1e-14, 1e-14, true, false, &layout);
        assert_dyn_matches_oracle(&log(&view).unwrap().into(), po.arr("log"), 1e-14, 1e-14, true, false, &layout);
        assert_dyn_matches_oracle(&exp(&view).unwrap().into(), po.arr("exp"), 1e-13, 1e-13, true, false, &layout);
    }

    assert_dyn_matches_oracle(&sqrt(&pos).unwrap().into(), oracle.arr("pos_sqrt"), 1e-14, 1e-14, true, false, "pos_sqrt");
    assert_dyn_matches_oracle(&log(&pos).unwrap().into(), oracle.arr("pos_log"), 1e-14, 1e-14, true, false, "pos_log");
    assert_dyn_matches_oracle(&log2(&pos).unwrap().into(), oracle.arr("pos_log2"), 1e-14, 1e-14, true, false, "pos_log2");
    assert_dyn_matches_oracle(&log10(&pos).unwrap().into(), oracle.arr("pos_log10"), 1e-14, 1e-14, true, false, "pos_log10");
    assert_dyn_matches_oracle(&log1p(&pos).unwrap().into(), oracle.arr("pos_log1p"), 1e-14, 1e-14, true, false, "pos_log1p");

    let clipped = clip(
        &mat,
        Some(&NdArray::from_scalar(-0.95)),
        Some(&NdArray::from_scalar(0.95)),
    )
    .unwrap();
    assert_dyn_matches_oracle(&arcsin(&clipped).unwrap().into(), oracle.arr("unit_arcsin"), 1e-14, 1e-14, true, false, "arcsin");
    assert_dyn_matches_oracle(&arccos(&clipped).unwrap().into(), oracle.arr("unit_arccos"), 1e-14, 1e-14, true, false, "arccos");
    assert_dyn_matches_oracle(&arctan(&clipped).unwrap().into(), oracle.arr("unit_arctan"), 1e-14, 1e-14, true, false, "arctan");
    assert_dyn_matches_oracle(&atanh(&clipped).unwrap().into(), oracle.arr("unit_arctanh"), 1e-13, 1e-13, true, false, "arctanh");
    let _ = (asin::<NdArray<f64>>, acos::<NdArray<f64>>, atan::<NdArray<f64>>);

    let big_ge1 = big
        .slice(&[numpy_types::SliceArg::Slice(numpy_types::Slice::new(None, Some(4), 1))])
        .unwrap();
    assert_dyn_matches_oracle(&acosh(&big_ge1).unwrap().into(), oracle.arr("big_acosh"), 1e-13, 1e-13, true, false, "acosh");

    assert_dyn_matches_oracle(&round(&mat, 1).unwrap().into(), oracle.arr("round_pos"), 1e-14, 1e-14, false, false, "round_pos");
    let r_arr = NdArray::from_vec(vec![15.0, 25.0, 123.45]);
    assert_dyn_matches_oracle(&round(&r_arr, -1).unwrap().into(), oracle.arr("round_neg"), 0.0, 0.0, false, false, "round_neg");

    assert_dyn_matches_oracle(
        &nan_to_num(&a, Some(-99.0), Some(123.0), Some(-123.0))
            .unwrap()
            .into(),
        oracle.arr("nan_to_num"),
        0.0,
        0.0,
        true,
        false,
        "nan_to_num",
    );

    let (fm, fe) = frexp(&a).unwrap();
    assert_dyn_matches_oracle(&fm.into(), oracle.arr("frexp_m"), 0.0, 0.0, true, false, "frexp_m");
    let fe_fin = fe
        .slice(&[numpy_types::SliceArg::Slice(numpy_types::Slice::new(
            None,
            Some(16),
            1,
        ))])
        .unwrap();
    assert_dyn_matches_oracle(&fe_fin.into(), oracle.arr("frexp_e"), 0.0, 0.0, false, false, "frexp_e");

    let (mf, mi) = modf(&a).unwrap();
    assert_dyn_matches_oracle(&mf.into(), oracle.arr("modf_f"), 0.0, 0.0, true, false, "modf_f");
    assert_dyn_matches_oracle(&mi.into(), oracle.arr("modf_i"), 0.0, 0.0, true, false, "modf_i");

    // Complex128 and f32 / f16 unary parity
    let f32_arr = NdArray::from_vec(edge_f32_values());
    let c_arr = NdArray::from_vec(vec![
        Complex64::new(1.2, -0.8),
        Complex64::new(-0.5, 2.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 4.0),
    ]);
    let f16_arr = NdArray::from_vec(vec![f16::from_f32(1.5), f16::from_f32(-2.25)]);
    let pyc = format!(
        "{}\n{}\n{}\nemit_arr('f32_abs', np.abs(f))\nemit_arr('c_abs', np.abs(c))\nemit_arr('c_real', np.real(c))\nemit_arr('c_imag', np.imag(c))\nemit_arr('c_conj', np.conj(c))\nemit_arr('c_angle_deg', np.angle(c, deg=True))\nemit_arr('c_exp', np.exp(c))\nemit_arr('h_sqrt', np.sqrt(np.abs(h)))\n",
        py_assign_dyn("f", &DynArray::F32(f32_arr.clone())),
        py_assign_dyn("c", &DynArray::C128(c_arr.clone())),
        py_assign_dyn("h", &DynArray::F16(f16_arr.clone())),
    );
    let oc = run_oracle(&pyc);
    assert_dyn_matches_oracle(&abs(&f32_arr).unwrap().into(), oc.arr("f32_abs"), 1e-6, 1e-6, true, false, "f32_abs");
    assert_dyn_matches_oracle(&abs(&c_arr).unwrap().into(), oc.arr("c_abs"), 1e-13, 1e-13, true, false, "c_abs");
    assert_dyn_matches_oracle(&real(&c_arr).unwrap().into(), oc.arr("c_real"), 0.0, 0.0, true, false, "c_real");
    assert_dyn_matches_oracle(&imag(&c_arr).unwrap().into(), oc.arr("c_imag"), 0.0, 0.0, true, false, "c_imag");
    assert_dyn_matches_oracle(&conj(&c_arr).unwrap().into(), oc.arr("c_conj"), 0.0, 0.0, true, false, "c_conj");
    assert_dyn_matches_oracle(&angle(&c_arr, true).unwrap().into(), oc.arr("c_angle_deg"), 1e-12, 1e-12, true, false, "c_angle_deg");
    assert_dyn_matches_oracle(&exp(&c_arr).unwrap().into(), oc.arr("c_exp"), 1e-13, 1e-13, true, false, "c_exp");
    assert_dyn_matches_oracle(&sqrt(&abs(&f16_arr).unwrap()).unwrap().into(), oc.arr("h_sqrt"), 1e-2, 1e-2, true, false, "f16_sqrt");
    let _ = edge_c128_values();
}

#[test]
fn binary_promotion_broadcasting_bitwise_comparison_and_ldexp_against_numpy() {
    let a = NdArray::from_shape_vec(&[3, 1], vec![1.5, -2.5, 4.0]).unwrap();
    let b = NdArray::from_shape_vec(&[1, 4], vec![2.0, -3.0, 0.5, 1.0]).unwrap();
    let ia = NdArray::from_vec(vec![12i32, -15, 7, 0, -1]);
    let ib = NdArray::from_vec(vec![8i32, 6, 3, 5, 2]);
    let ua = NdArray::from_vec(vec![0b101101u8, 0xff, 0]);
    let exp_arr = NdArray::from_vec(vec![1i32, 2, -1, 0]);

    let py = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n\
emit_arr('add', a + b)
emit_arr('sub', a - b)
emit_arr('mul', a * b)
emit_arr('tdiv', np.true_divide(a, b))
emit_arr('fdiv', np.floor_divide(a, b))
emit_arr('rem', np.remainder(a, b))
emit_arr('fmod', np.fmod(a, b))
emit_arr('divmod_q', np.divmod(a, b)[0])
emit_arr('divmod_r', np.divmod(a, b)[1])
emit_arr('fpow', np.float_power(np.abs(a), b))
emit_arr('atan2', np.arctan2(a, b))
emit_arr('hypot', np.hypot(a, b))
emit_arr('copysign', np.copysign(a, b))
emit_arr('nextafter', np.nextafter(a, b))
emit_arr('heaviside', np.heaviside(a, b))
emit_arr('logaddexp', np.logaddexp(a, b))
emit_arr('logaddexp2', np.logaddexp2(a, b))
emit_arr('max', np.maximum(a, b))
emit_arr('min', np.minimum(a, b))
emit_arr('fmax', np.fmax(a, b))
emit_arr('fmin', np.fmin(a, b))
emit_arr('gcd', np.gcd(ia, ib))
emit_arr('lcm', np.lcm(ia, ib))
emit_arr('band', np.bitwise_and(ia, ib))
emit_arr('bor', np.bitwise_or(ia, ib))
emit_arr('bxor', np.bitwise_xor(ia, ib))
emit_arr('binv', np.invert(ia))
emit_arr('lsh', np.left_shift(ia, np.array([1, 2, 1, 0, 1], dtype=np.int32)))
emit_arr('rsh', np.right_shift(ia, np.array([1, 1, 1, 0, 1], dtype=np.int32)))
emit_arr('bcount', np.bitwise_count(ua))
emit_arr('ldexp', np.ldexp(b.ravel(), e))
emit_arr('eq', a == b)
emit_arr('ne', a != b)
emit_arr('lt', a < b)
emit_arr('le', a <= b)
emit_arr('gt', a > b)
emit_arr('ge', a >= b)
emit_arr('isclose', np.isclose(a, a + 1e-9, rtol=1e-5, atol=1e-8, equal_nan=False))
emit_arr('land', np.logical_and(ia, ib))
emit_arr('lor', np.logical_or(ia, ib))
emit_arr('lxor', np.logical_xor(ia, ib))
emit_arr('dyn_promo', ia.astype(np.int32) + np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32))
",
        py_assign_dyn("a", &DynArray::F64(a.clone())),
        py_assign_dyn("b", &DynArray::F64(b.clone())),
        py_assign_dyn("ia", &DynArray::I32(ia.clone())),
        py_assign_dyn("ib", &DynArray::I32(ib.clone())),
        py_assign_dyn("ua", &DynArray::U8(ua.clone())),
        py_assign_dyn("e", &DynArray::I32(exp_arr.clone())),
    );
    let o = run_oracle(&py);

    assert_dyn_matches_oracle(&add(&a, &b).unwrap().into(), o.arr("add"), 0.0, 0.0, false, false, "add");
    assert_dyn_matches_oracle(&subtract(&a, &b).unwrap().into(), o.arr("sub"), 0.0, 0.0, false, false, "sub");
    assert_dyn_matches_oracle(&multiply(&a, &b).unwrap().into(), o.arr("mul"), 0.0, 0.0, false, false, "mul");
    assert_dyn_matches_oracle(&true_divide(&a, &b).unwrap().into(), o.arr("tdiv"), 1e-14, 1e-14, false, false, "tdiv");
    assert_dyn_matches_oracle(&floor_divide(&a, &b).unwrap().into(), o.arr("fdiv"), 0.0, 0.0, false, false, "fdiv");
    assert_dyn_matches_oracle(&remainder(&a, &b).unwrap().into(), o.arr("rem"), 1e-14, 1e-14, false, false, "rem");
    assert_dyn_matches_oracle(&fmod(&a, &b).unwrap().into(), o.arr("fmod"), 1e-14, 1e-14, false, false, "fmod");
    let (dq, dr) = divmod(&a, &b).unwrap();
    assert_dyn_matches_oracle(&dq.into(), o.arr("divmod_q"), 0.0, 0.0, false, false, "divmod_q");
    assert_dyn_matches_oracle(&dr.into(), o.arr("divmod_r"), 1e-14, 1e-14, false, false, "divmod_r");
    assert_dyn_matches_oracle(&numpy_ufunc::float_power(&abs(&a).unwrap(), &b).unwrap().into(), o.arr("fpow"), 1e-13, 1e-13, false, false, "fpow");
    assert_dyn_matches_oracle(&arctan2(&a, &b).unwrap().into(), o.arr("atan2"), 1e-14, 1e-14, false, false, "atan2");
    assert_dyn_matches_oracle(&hypot(&a, &b).unwrap().into(), o.arr("hypot"), 1e-14, 1e-14, false, false, "hypot");
    assert_dyn_matches_oracle(&copysign(&a, &b).unwrap().into(), o.arr("copysign"), 0.0, 0.0, false, false, "copysign");
    assert_dyn_matches_oracle(&nextafter(&a, &b).unwrap().into(), o.arr("nextafter"), 0.0, 0.0, false, false, "nextafter");
    assert_dyn_matches_oracle(&heaviside(&a, &b).unwrap().into(), o.arr("heaviside"), 0.0, 0.0, false, false, "heaviside");
    assert_dyn_matches_oracle(&logaddexp(&a, &b).unwrap().into(), o.arr("logaddexp"), 1e-13, 1e-13, false, false, "logaddexp");
    assert_dyn_matches_oracle(&logaddexp2(&a, &b).unwrap().into(), o.arr("logaddexp2"), 1e-13, 1e-13, false, false, "logaddexp2");
    assert_dyn_matches_oracle(&maximum(&a, &b).unwrap().into(), o.arr("max"), 0.0, 0.0, false, false, "maximum");
    assert_dyn_matches_oracle(&minimum(&a, &b).unwrap().into(), o.arr("min"), 0.0, 0.0, false, false, "minimum");
    assert_dyn_matches_oracle(&fmax(&a, &b).unwrap().into(), o.arr("fmax"), 0.0, 0.0, false, false, "fmax");
    assert_dyn_matches_oracle(&fmin(&a, &b).unwrap().into(), o.arr("fmin"), 0.0, 0.0, false, false, "fmin");
    assert_dyn_matches_oracle(&gcd(&ia, &ib).unwrap().into(), o.arr("gcd"), 0.0, 0.0, false, false, "gcd");
    assert_dyn_matches_oracle(&lcm(&ia, &ib).unwrap().into(), o.arr("lcm"), 0.0, 0.0, false, false, "lcm");
    assert_dyn_matches_oracle(&bitwise_and(&ia, &ib).unwrap().into(), o.arr("band"), 0.0, 0.0, false, false, "band");
    assert_dyn_matches_oracle(&bitwise_or(&ia, &ib).unwrap().into(), o.arr("bor"), 0.0, 0.0, false, false, "bor");
    assert_dyn_matches_oracle(&bitwise_xor(&ia, &ib).unwrap().into(), o.arr("bxor"), 0.0, 0.0, false, false, "bxor");
    assert_dyn_matches_oracle(&invert(&ia).unwrap().into(), o.arr("binv"), 0.0, 0.0, false, false, "binv");
    let shl_b = NdArray::from_vec(vec![1i32, 2, 1, 0, 1]);
    let shr_b = NdArray::from_vec(vec![1i32, 1, 1, 0, 1]);
    assert_dyn_matches_oracle(&left_shift(&ia, &shl_b).unwrap().into(), o.arr("lsh"), 0.0, 0.0, false, false, "lsh");
    assert_dyn_matches_oracle(&right_shift(&ia, &shr_b).unwrap().into(), o.arr("rsh"), 0.0, 0.0, false, false, "rsh");
    assert_dyn_matches_oracle(&bitwise_count(&ua).unwrap().into(), o.arr("bcount"), 0.0, 0.0, false, false, "bcount");
    let b1 = b.ravel(Order::C);
    assert_dyn_matches_oracle(&ldexp(&b1, &exp_arr).unwrap().into(), o.arr("ldexp"), 0.0, 0.0, false, false, "ldexp");
    assert_dyn_matches_oracle(&equal(&a, &b).unwrap().into(), o.arr("eq"), 0.0, 0.0, false, false, "eq");
    assert_dyn_matches_oracle(&not_equal(&a, &b).unwrap().into(), o.arr("ne"), 0.0, 0.0, false, false, "ne");
    assert_dyn_matches_oracle(&less(&a, &b).unwrap().into(), o.arr("lt"), 0.0, 0.0, false, false, "lt");
    assert_dyn_matches_oracle(&less_equal(&a, &b).unwrap().into(), o.arr("le"), 0.0, 0.0, false, false, "le");
    assert_dyn_matches_oracle(&greater(&a, &b).unwrap().into(), o.arr("gt"), 0.0, 0.0, false, false, "gt");
    assert_dyn_matches_oracle(&greater_equal(&a, &b).unwrap().into(), o.arr("ge"), 0.0, 0.0, false, false, "ge");

    let a_eps = add(&a, &NdArray::from_scalar(1e-9)).unwrap();
    assert_dyn_matches_oracle(
        &isclose(&a, &a_eps, 1e-5, 1e-8, false).unwrap().into(),
        o.arr("isclose"),
        0.0,
        0.0,
        false,
        false,
        "isclose",
    );
    assert!(allclose(&a, &a_eps, 1e-5, 1e-8, false).unwrap());
    assert!(array_equal(&a, &a, false));
    assert!(array_equiv(&b, &b.broadcast_to(&[3, 4]).unwrap()));
    assert_dyn_matches_oracle(&logical_and(&ia, &ib).unwrap().into(), o.arr("land"), 0.0, 0.0, false, false, "land");
    assert_dyn_matches_oracle(&logical_or(&ia, &ib).unwrap().into(), o.arr("lor"), 0.0, 0.0, false, false, "lor");
    assert_dyn_matches_oracle(&logical_xor(&ia, &ib).unwrap().into(), o.arr("lxor"), 0.0, 0.0, false, false, "lxor");

    // Mixed dtype DynArray promotion (i32 + f32 -> f64)
    let f32_rhs = DynArray::F32(NdArray::from_vec(vec![1.5f32, 2.5, 3.5, 4.5, 5.5]));
    let dyn_sum = add(&DynArray::I32(ia.clone()), &f32_rhs).unwrap();
    assert_eq!(dyn_sum.dtype(), DType::F64);
    assert_dyn_matches_oracle(&dyn_sum, o.arr("dyn_promo"), 1e-6, 1e-6, false, false, "dyn_promo");
    let _ = power(&ia, &NdArray::from_vec(vec![2i32, 1, 2, 0, 3])).unwrap();
}

#[test]
fn higher_order_reduce_accumulate_outer_and_at_against_numpy() {
    let mat = NdArray::from_shape_vec(
        &[3, 4],
        vec![1i64, -4, 7, 2, 9, 0, -8, 5, 3, 6, -1, 10],
    )
    .unwrap();
    let dmat = DynArray::I64(mat.clone());
    let fvec = DynArray::F64(NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
    let fmat = DynArray::F64(NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let v1 = DynArray::F64(NdArray::from_vec(vec![1.0, 2.0, 3.0]));
    let v2 = DynArray::F64(NdArray::from_vec(vec![10.0, 20.0]));
    let b_arr = DynArray::Bool(NdArray::from_vec(vec![true, false, true, true]));

    let py = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n\
emit_arr('i64_max_ax0', np.maximum.reduce(m, axis=0))
emit_arr('i64_min_ax1', np.minimum.reduce(m, axis=1, keepdims=True))
emit_arr('i64_add_all', np.add.reduce(m, axis=None, initial=10))
emit_arr('i64_axes', np.add.reduce(m, axis=(0, 1)))
emit_arr('acc_mul', np.multiply.accumulate(fm, axis=1))
emit_arr('outer_sub', np.subtract.outer(v1, v2))
emit_arr('bool_land', np.logical_and.reduce(b))
emit_arr('bool_lxor', np.logical_xor.reduce(b))
a1 = fv.copy(); np.add.at(a1, [0, 1, 0, -1], 5.0); emit_arr('add_at', a1)
a2 = fv.copy(); np.negative.at(a2, [1, 1, 2]); emit_arr('neg_at', a2)
c1 = np.array([1.0+2.0j, -3.0+4.0j]); np.abs.at(c1, [0, 1]); emit_arr('cabs_at', c1)
f_at = np.array([1.0, 2.0, 3.0]); np.logical_and.at(f_at, [0, 2], 0.0); emit_arr('f_land_at', f_at)
",
        py_assign_dyn("m", &dmat),
        py_assign_dyn("fm", &fmat),
        py_assign_dyn("v1", &v1),
        py_assign_dyn("v2", &v2),
        py_assign_dyn("b", &b_arr),
        py_assign_dyn("fv", &fvec),
    );
    let o = run_oracle(&py);

    assert_dyn_matches_oracle(
        &reduce(BinaryOp::Maximum, &dmat, Some(0), None, false, None).unwrap(),
        o.arr("i64_max_ax0"),
        0.0,
        0.0,
        false,
        false,
        "i64_max_ax0",
    );
    assert_dyn_matches_oracle(
        &reduce(BinaryOp::Minimum, &dmat, Some(1), None, true, None).unwrap(),
        o.arr("i64_min_ax1"),
        0.0,
        0.0,
        false,
        false,
        "i64_min_ax1",
    );
    assert_dyn_matches_oracle(
        &reduce(BinaryOp::Add, &dmat, None, None, false, Some(ScalarValue::I64(10))).unwrap(),
        o.arr("i64_add_all"),
        0.0,
        0.0,
        false,
        false,
        "i64_add_all",
    );
    assert_dyn_matches_oracle(
        &reduce_axes(BinaryOp::Add, &dmat, &[0, 1], None, false, None).unwrap(),
        o.arr("i64_axes"),
        0.0,
        0.0,
        false,
        false,
        "i64_axes",
    );
    assert_dyn_matches_oracle(
        &accumulate(BinaryOp::Multiply, &fmat, 1, None).unwrap(),
        o.arr("acc_mul"),
        0.0,
        0.0,
        false,
        false,
        "acc_mul",
    );
    assert_dyn_matches_oracle(
        &outer(BinaryOp::Subtract, &v1, &v2).unwrap(),
        o.arr("outer_sub"),
        0.0,
        0.0,
        false,
        false,
        "outer_sub",
    );
    assert_dyn_matches_oracle(
        &reduce(BinaryOp::LogicalAnd, &b_arr, None, None, false, None).unwrap(),
        o.arr("bool_land"),
        0.0,
        0.0,
        false,
        false,
        "bool_land",
    );
    assert_dyn_matches_oracle(
        &reduce(BinaryOp::LogicalXor, &b_arr, None, None, false, None).unwrap(),
        o.arr("bool_lxor"),
        0.0,
        0.0,
        false,
        false,
        "bool_lxor",
    );

    let mut a1 = fvec.clone();
    let idx1 = NdArray::from_vec(vec![0i64, 1, 0, -1]);
    at(BinaryOp::Add, &mut a1, &idx1, &DynArray::F64(NdArray::from_scalar(5.0))).unwrap();
    assert_dyn_matches_oracle(&a1, o.arr("add_at"), 0.0, 0.0, false, false, "add_at");

    let mut a2 = fvec.clone();
    let idx2 = NdArray::from_vec(vec![1i64, 1, 2]);
    unary_at(UnaryOp::Negative, &mut a2, &idx2).unwrap();
    assert_dyn_matches_oracle(&a2, o.arr("neg_at"), 0.0, 0.0, false, false, "neg_at");

    let mut c1 = NdArray::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]);
    UnaryOp::Abs
        .at_nd(&mut c1, &NdArray::from_vec(vec![0i64, 1]))
        .unwrap();
    assert_dyn_matches_oracle(&c1.into(), o.arr("cabs_at"), 1e-13, 1e-13, false, false, "cabs_at");

    let mut f_at = NdArray::from_vec(vec![1.0f64, 2.0, 3.0]);
    BinaryOp::LogicalAnd
        .at_nd(
            &mut f_at,
            &NdArray::from_vec(vec![0i64, 2]),
            &NdArray::from_scalar(0.0f64),
        )
        .unwrap();
    assert_dyn_matches_oracle(&f_at.into(), o.arr("f_land_at"), 0.0, 0.0, false, false, "f_land_at");
}

#[test]
fn round2_complex_clip_and_tuple_at_against_numpy() {
    let py = r#"
c = np.array([1+0j, 1+2j, complex(1, np.nan), 2+0j], dtype=np.complex128)
emit_arr("clip_nan_lo", np.clip(c, complex(np.nan, 0.0), 5+0j))
emit_arr("clip_nan_hi", np.clip(c, 0+0j, complex(np.nan, 0.0)))
emit_arr("clip_nanj_lo", np.clip(c, complex(0.0, np.nan), 5+0j))
emit_arr("clip_nanj_hi", np.clip(c, 0+0j, complex(0.0, np.nan)))
emit_arr("clip_single_lo_nanj", np.clip(c, complex(0.0, np.nan), None))

try:
    np.add.at(np.zeros((2, 3)), (False, np.array([0, 1], dtype=np.int64)), 1.0)
except Exception as e:
    emit_err("at_false_idx", e)

af = np.zeros((2, 3), dtype=np.float64)
np.add.at(af, (False,), np.zeros((0, 2, 3), dtype=np.float64))
emit_arr("at_false_only", af)

at_t = np.zeros((2, 3), dtype=np.float64)
np.add.at(at_t, (True,), np.arange(1.0, 7.0).reshape((1, 2, 3)))
emit_arr("at_true_only", at_t)

at_t2 = np.zeros((3, 4), dtype=np.float64)
np.add.at(at_t2, (True, np.array([0, 2], dtype=np.int64)), 1.5)
emit_arr("at_true_idx", at_t2)
"#;
    let o = run_oracle(py);

    let c64 = NdArray::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(1.0, f64::NAN),
        Complex64::new(2.0, 0.0),
    ]);
    assert_dyn_matches_oracle(
        &numpy_ufunc::clip_scalar(
            &c64,
            Some(Complex64::new(f64::NAN, 0.0)),
            Some(Complex64::new(5.0, 0.0)),
        )
        .unwrap()
        .into(),
        o.arr("clip_nan_lo"),
        0.0,
        0.0,
        true,
        true,
        "clip_nan_lo",
    );
    assert_dyn_matches_oracle(
        &numpy_ufunc::clip_scalar(
            &c64,
            Some(Complex64::new(0.0, 0.0)),
            Some(Complex64::new(f64::NAN, 0.0)),
        )
        .unwrap()
        .into(),
        o.arr("clip_nan_hi"),
        0.0,
        0.0,
        true,
        true,
        "clip_nan_hi",
    );
    assert_dyn_matches_oracle(
        &numpy_ufunc::clip_scalar(
            &c64,
            Some(Complex64::new(0.0, f64::NAN)),
            Some(Complex64::new(5.0, 0.0)),
        )
        .unwrap()
        .into(),
        o.arr("clip_nanj_lo"),
        0.0,
        0.0,
        true,
        true,
        "clip_nanj_lo",
    );
    assert_dyn_matches_oracle(
        &numpy_ufunc::clip_scalar(
            &c64,
            Some(Complex64::new(0.0, 0.0)),
            Some(Complex64::new(0.0, f64::NAN)),
        )
        .unwrap()
        .into(),
        o.arr("clip_nanj_hi"),
        0.0,
        0.0,
        true,
        true,
        "clip_nanj_hi",
    );
    assert_dyn_matches_oracle(
        &numpy_ufunc::clip_scalar(&c64, Some(Complex64::new(0.0, f64::NAN)), None)
            .unwrap()
            .into(),
        o.arr("clip_single_lo_nanj"),
        0.0,
        0.0,
        true,
        true,
        "clip_single_lo_nanj",
    );

    assert!(o.errors.contains_key("at_false_idx"));
    let f0 = DynArray::from(NdArray::from_scalar(false));
    let t0 = DynArray::from(NdArray::from_scalar(true));
    let idx01 = DynArray::from(NdArray::from_vec(vec![0i64, 1]));
    let mut a_err = DynArray::from(NdArray::<f64>::zeros(&[2, 3]).unwrap());
    assert!(BinaryOp::Add
        .at(
            &mut a_err,
            numpy_ufunc::AtIndex::Tuple(&[&f0, &idx01]),
            &DynArray::from(NdArray::from_scalar(1.0f64)),
        )
        .is_err());

    let mut af = DynArray::from(NdArray::<f64>::zeros(&[2, 3]).unwrap());
    BinaryOp::Add
        .at(
            &mut af,
            numpy_ufunc::AtIndex::Tuple(&[&f0]),
            &DynArray::from(NdArray::<f64>::zeros(&[0, 2, 3]).unwrap()),
        )
        .unwrap();
    assert_dyn_matches_oracle(&af, o.arr("at_false_only"), 0.0, 0.0, false, true, "at_false_only");

    let mut at_t = DynArray::from(NdArray::<f64>::zeros(&[2, 3]).unwrap());
    let t_val = NdArray::from_shape_vec(&[1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    BinaryOp::Add
        .at(&mut at_t, numpy_ufunc::AtIndex::Tuple(&[&t0]), &DynArray::from(t_val))
        .unwrap();
    assert_dyn_matches_oracle(&at_t, o.arr("at_true_only"), 0.0, 0.0, false, true, "at_true_only");

    let mut at_t2 = DynArray::from(NdArray::<f64>::zeros(&[3, 4]).unwrap());
    let idx02 = DynArray::from(NdArray::from_vec(vec![0i64, 2]));
    BinaryOp::Add
        .at(
            &mut at_t2,
            numpy_ufunc::AtIndex::Tuple(&[&t0, &idx02]),
            &DynArray::from(NdArray::from_scalar(1.5f64)),
        )
        .unwrap();
    assert_dyn_matches_oracle(&at_t2, o.arr("at_true_idx"), 0.0, 0.0, false, true, "at_true_idx");
}

#[test]
fn p0_p1_int_reciprocal_neg_zero_max_min_reduce_and_initial_bounds() {
    let a64 = NdArray::from_vec(vec![0.0f64, -0.0, 0.0, f64::NAN, 1.0]);
    let b64 = NdArray::from_vec(vec![-0.0f64, 0.0, 0.0, 1.0, f64::NAN]);
    let a32 = NdArray::from_vec(vec![0.0f32, -0.0, 0.0, f32::NAN, 1.0]);
    let b32 = NdArray::from_vec(vec![-0.0f32, 0.0, 0.0, 1.0, f32::NAN]);
    let neg0 = NdArray::from_vec(vec![-0.0f64]);
    let i8_a = NdArray::from_vec(vec![0i8, 1, -2, 2]);
    let i16_a = NdArray::from_vec(vec![0i16, 1, -2, 2]);
    let i32_a = NdArray::from_vec(vec![0i32, 1, -2, 2]);
    let i64_a = NdArray::from_vec(vec![0i64, 1, -2, 2]);
    let u32_a = NdArray::from_vec(vec![0u32, 1, 2]);
    let u64_nz = NdArray::from_vec(vec![1u64, 2]);
    let u64_a = NdArray::from_vec(vec![0u64, 1, 2]);

    let oracle = run_oracle(&format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        py_assign_dyn("a64", &DynArray::F64(a64.clone())),
        py_assign_dyn("b64", &DynArray::F64(b64.clone())),
        py_assign_dyn("a32", &DynArray::F32(a32.clone())),
        py_assign_dyn("b32", &DynArray::F32(b32.clone())),
        py_assign_dyn("neg0", &DynArray::F64(neg0.clone())),
        py_assign_dyn("i8_a", &DynArray::I8(i8_a.clone())),
        py_assign_dyn("i16_a", &DynArray::I16(i16_a.clone())),
        py_assign_dyn("i32_a", &DynArray::I32(i32_a.clone())),
        py_assign_dyn("i64_a", &DynArray::I64(i64_a.clone())),
        py_assign_dyn("u32_a", &DynArray::U32(u32_a.clone())),
        py_assign_dyn("u64_nz", &DynArray::U64(u64_nz.clone())),
        py_assign_dyn("u64_a", &DynArray::U64(u64_a.clone())),
        r#"
emit_arr("rec_i8", np.reciprocal(i8_a))
emit_arr("rec_i16", np.reciprocal(i16_a))
emit_arr("rec_i32", np.reciprocal(i32_a))
emit_arr("rec_i64", np.reciprocal(i64_a))
emit_arr("rec_u32", np.reciprocal(u32_a))
emit_arr("rec_u64_nz", np.reciprocal(u64_nz))
emit_arr("rec_u64", np.reciprocal(u64_a))
emit_arr("max64", np.maximum(a64, b64))
emit_arr("min64", np.minimum(a64, b64))
emit_arr("max32", np.maximum(a32, b32))
emit_arr("min32", np.minimum(a32, b32))
emit_arr("fmax64", np.fmax(a64, b64))
emit_arr("fmin64", np.fmin(a64, b64))
emit_arr("add_neg0", np.array(np.add.reduce(neg0), dtype=np.float64))
emit_arr("mul_neg0", np.array(np.multiply.reduce(neg0), dtype=np.float64))
emit_arr("max_neg0", np.array(np.maximum.reduce(neg0), dtype=np.float64))
"#
    ));

    // FIX(F-A2-01): integer reciprocal at 0 (note: C `(uint64_t)(+Inf)` is compiler UB between MSVC `1<<63` and GCC `0`)
    assert_dyn_matches_oracle(&reciprocal(&DynArray::I8(i8_a)).unwrap(), oracle.arr("rec_i8"), 0., 0., false, false, "rec_i8");
    assert_dyn_matches_oracle(&reciprocal(&DynArray::I16(i16_a)).unwrap(), oracle.arr("rec_i16"), 0., 0., false, false, "rec_i16");
    assert_dyn_matches_oracle(&reciprocal(&DynArray::I32(i32_a)).unwrap(), oracle.arr("rec_i32"), 0., 0., false, false, "rec_i32");
    assert_dyn_matches_oracle(&reciprocal(&DynArray::I64(i64_a)).unwrap(), oracle.arr("rec_i64"), 0., 0., false, false, "rec_i64");
    assert_dyn_matches_oracle(&reciprocal(&DynArray::U32(u32_a)).unwrap(), oracle.arr("rec_u32"), 0., 0., false, false, "rec_u32");
    assert_dyn_matches_oracle(&reciprocal(&DynArray::U64(u64_nz)).unwrap(), oracle.arr("rec_u64_nz"), 0., 0., false, false, "rec_u64_nz");
    let rec_u64 = reciprocal(&DynArray::U64(u64_a)).unwrap();
    assert_eq!(NdArray::<u64>::try_from(rec_u64.clone()).unwrap().to_vec(), vec![1u64 << 63, 1, 0]);
    #[cfg(target_os = "windows")]
    assert_dyn_matches_oracle(&rec_u64, oracle.arr("rec_u64"), 0., 0., false, false, "rec_u64");

    // FIX(F-A2-02 + T-AUD-UFUNC-NAN): maximum/minimum/fmax/fmin signed zero & NaN
    let da64 = DynArray::F64(a64);
    let db64 = DynArray::F64(b64);
    let da32 = DynArray::F32(a32);
    let db32 = DynArray::F32(b32);
    assert_dyn_matches_oracle(&maximum(&da64, &db64).unwrap(), oracle.arr("max64"), 0., 0., true, true, "max64");
    assert_dyn_matches_oracle(&minimum(&da64, &db64).unwrap(), oracle.arr("min64"), 0., 0., true, true, "min64");
    assert_dyn_matches_oracle(&maximum(&da32, &db32).unwrap(), oracle.arr("max32"), 0., 0., true, true, "max32");
    assert_dyn_matches_oracle(&minimum(&da32, &db32).unwrap(), oracle.arr("min32"), 0., 0., true, true, "min32");
    assert_dyn_matches_oracle(&fmax(&da64, &db64).unwrap(), oracle.arr("fmax64"), 0., 0., true, true, "fmax64");
    assert_dyn_matches_oracle(&fmin(&da64, &db64).unwrap(), oracle.arr("fmin64"), 0., 0., true, true, "fmin64");

    // FIX(F-A2-03): singleton [-0.0] reduction seeds accumulator with identity
    let dneg0 = DynArray::F64(neg0);
    assert_dyn_matches_oracle(&reduce(BinaryOp::Add, &dneg0, None, None, false, None).unwrap(), oracle.arr("add_neg0"), 0., 0., false, true, "add_neg0");
    assert_dyn_matches_oracle(&reduce(BinaryOp::Multiply, &dneg0, None, None, false, None).unwrap(), oracle.arr("mul_neg0"), 0., 0., false, true, "mul_neg0");
    assert_dyn_matches_oracle(&reduce(BinaryOp::Maximum, &dneg0, None, None, false, None).unwrap(), oracle.arr("max_neg0"), 0., 0., false, true, "max_neg0");

    // FIX(F-A2-04): reduce initial NaN / Inf / out-of-range int rejected instead of saturating
    let i32_arr = DynArray::I32(NdArray::from_vec(vec![1, 2, 3]));
    assert!(reduce(BinaryOp::Add, &i32_arr, None, None, false, Some(ScalarValue::F64(f64::NAN))).is_err());
    assert!(reduce(BinaryOp::Add, &i32_arr, None, None, false, Some(ScalarValue::F64(1e300))).is_err());
    assert!(reduce(BinaryOp::Add, &i32_arr, None, Some(DType::I32), false, Some(ScalarValue::I64(i64::MAX))).is_err());
    assert!(reduce(BinaryOp::Maximum, &i32_arr, None, None, false, Some(ScalarValue::I64(i64::MAX))).is_err());
    assert!(reduce(BinaryOp::Add, &DynArray::F32(NdArray::from_vec(vec![1.0f32])), None, None, false, Some(ScalarValue::C64(num_complex::Complex32::new(1.0, 2.0)))).is_err());
}

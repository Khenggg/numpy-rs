mod common;

use common::{
    assert_aligned, assert_c128_close, assert_c64_close, assert_dyn_matches_oracle,
    assert_f32_close, assert_f64_bits_eq, assert_f64_close, non_contiguous_2d_views, py_assign_c128,
    py_assign_c64, py_assign_dyn, py_assign_f32, py_assign_f64, run_oracle,
};
use numpy_fft::{
    fft, fft2, fftfreq, fftfreq_f32, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift, ihfft,
    irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftfreq_f32, rfftn, Norm,
};
use numpy_types::{f16, Complex32, Complex64, DynArray, NdArray, Slice, SliceArg};

fn make_real_vec(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let x = i as f64 + 1.0 + seed;
            (x * 0.73).sin() * 2.5 + (x * 0.19).cos() * 0.75 - 0.1 * (i % 5) as f64
        })
        .collect()
}

fn make_complex_vec(len: usize, seed: f64) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            let x = i as f64 + 1.0 + seed;
            Complex64::new(
                (x * 0.61).sin() * 2.0 + 0.1 * ((i % 7) as f64 - 3.0),
                (x * 0.43).cos() * 1.75 - 0.15 * ((i % 4) as f64),
            )
        })
        .collect()
}

fn assert_c128_matches(
    actual: &NdArray<Complex64>,
    expected_tag: &str,
    oracle: &common::OracleOutput,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_aligned(actual, ctx);
    let exp = oracle.arr(expected_tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{ctx}: shape mismatch");
    assert_c128_close(&actual.to_vec(), &exp.to_c128_vec(), rtol, atol, equal_nan, ctx);
}

fn assert_c64_matches(
    actual: &NdArray<Complex32>,
    expected_tag: &str,
    oracle: &common::OracleOutput,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_aligned(actual, ctx);
    let exp = oracle.arr(expected_tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{ctx}: shape mismatch");
    assert_c64_close(&actual.to_vec(), &exp.to_c64_vec(), rtol, atol, equal_nan, ctx);
}

fn assert_f64_matches(
    actual: &NdArray<f64>,
    expected_tag: &str,
    oracle: &common::OracleOutput,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_aligned(actual, ctx);
    let exp = oracle.arr(expected_tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{ctx}: shape mismatch");
    assert_f64_close(&actual.to_vec(), &exp.to_f64_vec(), rtol, atol, equal_nan, ctx);
}

fn assert_f32_matches(
    actual: &NdArray<f32>,
    expected_tag: &str,
    oracle: &common::OracleOutput,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    ctx: &str,
) {
    assert_aligned(actual, ctx);
    let exp = oracle.arr(expected_tag);
    assert_eq!(actual.shape(), exp.shape.as_slice(), "{ctx}: shape mismatch");
    assert_f32_close(&actual.to_vec(), &exp.to_f32_vec(), rtol, atol, equal_nan, ctx);
}

#[test]
fn test_fft_ifft_1d_and_axis_norms_against_oracle() {
    let mut script = String::new();
    let mut real_cases: Vec<(usize, Option<usize>, NdArray<f64>)> = Vec::new();
    let mut cpx_cases: Vec<(usize, Option<usize>, NdArray<Complex64>)> = Vec::new();

    for (idx, &(len, n_opt)) in [
        (1usize, None),
        (2, None),
        (3, None),
        (5, None),
        (7, None),
        (8, None),
        (12, None),
        (17, None),
        (10, Some(4)),  // crop
        (6, Some(13)),  // prime zero-pad
        (12, Some(16)), // pow2 zero-pad
    ]
    .iter()
    .enumerate()
    {
        let r = NdArray::from_vec(make_real_vec(len, idx as f64));
        let c = NdArray::from_vec(make_complex_vec(len, idx as f64));
        script.push_str(&py_assign_f64(&format!("r_{idx}"), &r));
        script.push('\n');
        script.push_str(&py_assign_c128(&format!("c_{idx}"), &c));
        script.push('\n');
        let n_py = match n_opt {
            Some(n) => n.to_string(),
            None => "None".to_string(),
        };
        for norm in ["backward", "ortho", "forward"] {
            script.push_str(&format!(
                "emit_arr('rf_{idx}_{norm}', np.fft.fft(r_{idx}, n={n_py}, axis=-1, norm='{norm}'))\n\
                 emit_arr('cf_{idx}_{norm}', np.fft.fft(c_{idx}, n={n_py}, axis=-1, norm='{norm}'))\n\
                 emit_arr('ci_{idx}_{norm}', np.fft.ifft(c_{idx}, n={n_py}, axis=-1, norm='{norm}'))\n"
            ));
        }
        real_cases.push((idx, n_opt, r));
        cpx_cases.push((idx, n_opt, c));
    }

    // 3D batched along axis 0, 1, -1, plus f32/c64 checks
    let arr3 = NdArray::from_shape_vec(&[3, 5, 7], make_complex_vec(3 * 5 * 7, 9.0)).unwrap();
    let arr1_f32 = NdArray::from_vec(make_real_vec(11, 2.5).into_iter().map(|x| x as f32).collect());
    let arr1_c64 = NdArray::from_vec(
        make_complex_vec(11, 3.5)
            .into_iter()
            .map(|z| Complex32::new(z.re as f32, z.im as f32))
            .collect(),
    );
    script.push_str(&py_assign_c128("a3", &arr3));
    script.push('\n');
    script.push_str(&py_assign_f32("af32", &arr1_f32));
    script.push('\n');
    script.push_str(&py_assign_c64("ac64", &arr1_c64));
    script.push('\n');
    for ax in [0isize, 1, -1] {
        script.push_str(&format!(
            "emit_arr('a3_f_{ax}', np.fft.fft(a3, axis={ax}, norm='ortho'))\n\
             emit_arr('a3_i_{ax}', np.fft.ifft(a3, axis={ax}, norm='ortho'))\n"
        ));
    }
    script.push_str(
        "emit_arr('af32_fft', np.fft.fft(af32, norm='forward'))\n\
         emit_arr('ac64_fft', np.fft.fft(ac64, n=16, norm='ortho'))\n\
         emit_arr('ac64_ifft', np.fft.ifft(ac64, n=7, norm='backward'))\n",
    );

    let oracle = run_oracle(&script);
    let norms = [
        ("backward", Norm::Backward),
        ("ortho", Norm::Ortho),
        ("forward", Norm::Forward),
    ];

    for (idx, n_opt, r) in &real_cases {
        for &(norm_name, norm) in &norms {
            let out = fft(r, *n_opt, -1, norm).unwrap();
            assert_c128_matches(
                &out,
                &format!("rf_{idx}_{norm_name}"),
                &oracle,
                1e-12,
                1e-12,
                false,
                &format!("fft real case {idx} norm={norm_name}"),
            );
        }
    }

    for (idx, n_opt, c) in &cpx_cases {
        for &(norm_name, norm) in &norms {
            let f_out = fft(c, *n_opt, -1, norm).unwrap();
            assert_c128_matches(
                &f_out,
                &format!("cf_{idx}_{norm_name}"),
                &oracle,
                1e-12,
                1e-12,
                false,
                &format!("fft cpx case {idx} norm={norm_name}"),
            );
            let i_out = ifft(c, *n_opt, -1, norm).unwrap();
            assert_c128_matches(
                &i_out,
                &format!("ci_{idx}_{norm_name}"),
                &oracle,
                1e-12,
                1e-12,
                false,
                &format!("ifft cpx case {idx} norm={norm_name}"),
            );
            if n_opt.is_none() {
                let rt = ifft(&f_out, None, -1, norm).unwrap();
                assert_c128_close(&rt.to_vec(), &c.to_vec(), 1e-12, 1e-12, false, "fft/ifft roundtrip");
            }
        }
    }

    for ax in [0isize, 1, -1] {
        let f3 = fft(&arr3, None, ax, Norm::Ortho).unwrap();
        assert_c128_matches(&f3, &format!("a3_f_{ax}"), &oracle, 1e-12, 1e-12, false, &format!("a3_f_{ax}"));
        let i3 = ifft(&arr3, None, ax, Norm::Ortho).unwrap();
        assert_c128_matches(&i3, &format!("a3_i_{ax}"), &oracle, 1e-12, 1e-12, false, &format!("a3_i_{ax}"));
    }

    let rf32 = fft(&arr1_f32, None, -1, Norm::Forward).unwrap();
    assert_c64_matches(&rf32, "af32_fft", &oracle, 1e-5, 1e-5, false, "af32_fft");
    let rc64 = fft(&arr1_c64, Some(16), -1, Norm::Ortho).unwrap();
    assert_c64_matches(&rc64, "ac64_fft", &oracle, 1e-5, 1e-5, false, "ac64_fft");
    let ic64 = ifft(&arr1_c64, Some(7), -1, Norm::Backward).unwrap();
    assert_c64_matches(&ic64, "ac64_ifft", &oracle, 1e-5, 1e-5, false, "ac64_ifft");
}

#[test]
fn test_fft2_ifft2_fftn_ifftn_axes_s_and_repeated_against_oracle() {
    let a2 = NdArray::from_shape_vec(&[4, 5], make_complex_vec(20, 1.1)).unwrap();
    let a3 = NdArray::from_shape_vec(&[2, 3, 5], make_complex_vec(30, 2.2)).unwrap();
    let a3_f32 = NdArray::from_shape_vec(
        &[3, 4, 2],
        make_real_vec(24, 0.7).into_iter().map(|x| x as f32).collect(),
    )
    .unwrap();

    let mut script = String::new();
    script.push_str(&py_assign_c128("a2", &a2));
    script.push('\n');
    script.push_str(&py_assign_c128("a3", &a3));
    script.push('\n');
    script.push_str(&py_assign_f32("a3f", &a3_f32));
    script.push('\n');
    script.push_str(
        "emit_arr('f2_def', np.fft.fft2(a2))\n\
         emit_arr('i2_def', np.fft.ifft2(a2))\n\
         emit_arr('f2_s', np.fft.fft2(a2, s=(6, 3), axes=(-1, -2), norm='ortho'))\n\
         emit_arr('f2_neg1', np.fft.fft2(a2, s=(-1, 7), norm='forward'))\n\
         emit_arr('f2_rep', np.fft.fft2(a2, axes=(0, 0)))\n\
         emit_arr('i2_rep', np.fft.ifft2(a2, s=(3, 6), axes=(1, 1), norm='ortho'))\n\
         emit_arr('fn_def', np.fft.fftn(a3))\n\
         emit_arr('in_def', np.fft.ifftn(a3, norm='forward'))\n\
         emit_arr('fn_axes', np.fft.fftn(a3, s=(4, -1), axes=(2, 0), norm='ortho'))\n\
         emit_arr('fn_empty', np.fft.fftn(a3, axes=()))\n\
         emit_arr('in_empty', np.fft.ifftn(a3, axes=()))\n\
         emit_arr('fn_rep', np.fft.fftn(a3, s=(3, 4, 3), axes=(0, 2, 0)))\n\
         emit_arr('a3f_fn', np.fft.fftn(a3f, s=(5, 2), axes=(-2, -1)))\n",
    );
    let oracle = run_oracle(&script);

    assert_c128_matches(&fft2(&a2, None, None, Norm::Backward).unwrap(), "f2_def", &oracle, 1e-12, 1e-12, false, "f2_def");
    assert_c128_matches(&ifft2(&a2, None, None, Norm::Backward).unwrap(), "i2_def", &oracle, 1e-12, 1e-12, false, "i2_def");
    assert_c128_matches(
        &fft2(&a2, Some(&[6, 3]), Some(&[-1, -2]), Norm::Ortho).unwrap(),
        "f2_s",
        &oracle,
        1e-12,
        1e-12,
        false,
        "f2_s",
    );
    assert_c128_matches(
        &fft2(&a2, Some(&[-1, 7]), None, Norm::Forward).unwrap(),
        "f2_neg1",
        &oracle,
        1e-12,
        1e-12,
        false,
        "f2_neg1",
    );
    assert_c128_matches(
        &fft2(&a2, None, Some(&[0, 0]), Norm::Backward).unwrap(),
        "f2_rep",
        &oracle,
        1e-12,
        1e-12,
        false,
        "f2_rep",
    );
    assert_c128_matches(
        &ifft2(&a2, Some(&[3, 6]), Some(&[1, 1]), Norm::Ortho).unwrap(),
        "i2_rep",
        &oracle,
        1e-12,
        1e-12,
        false,
        "i2_rep",
    );
    assert_c128_matches(&fftn(&a3, None, None, Norm::Backward).unwrap(), "fn_def", &oracle, 1e-12, 1e-12, false, "fn_def");
    assert_c128_matches(&ifftn(&a3, None, None, Norm::Forward).unwrap(), "in_def", &oracle, 1e-12, 1e-12, false, "in_def");
    assert_c128_matches(
        &fftn(&a3, Some(&[4, -1]), Some(&[2, 0]), Norm::Ortho).unwrap(),
        "fn_axes",
        &oracle,
        1e-12,
        1e-12,
        false,
        "fn_axes",
    );
    assert_c128_matches(&fftn(&a3, None, Some(&[]), Norm::Backward).unwrap(), "fn_empty", &oracle, 1e-12, 1e-12, false, "fn_empty");
    assert_c128_matches(&ifftn(&a3, None, Some(&[]), Norm::Backward).unwrap(), "in_empty", &oracle, 1e-12, 1e-12, false, "in_empty");
    assert_c128_matches(
        &fftn(&a3, Some(&[3, 4, 3]), Some(&[0, 2, 0]), Norm::Backward).unwrap(),
        "fn_rep",
        &oracle,
        1e-12,
        1e-12,
        false,
        "fn_rep",
    );
    assert_c64_matches(
        &fftn(&a3_f32, Some(&[5, 2]), Some(&[-2, -1]), Norm::Backward).unwrap(),
        "a3f_fn",
        &oracle,
        1e-5,
        1e-5,
        false,
        "a3f_fn",
    );
}

#[test]
fn test_rfft_irfft_nd_and_odd_even_lengths_against_oracle() {
    let r_even = NdArray::from_vec(make_real_vec(8, 0.3));
    let r_odd = NdArray::from_vec(make_real_vec(7, 0.9));
    let r_one = NdArray::from_vec(vec![4.25]);
    let r2 = NdArray::from_shape_vec(&[5, 6], make_real_vec(30, 1.7)).unwrap();
    let r3 = NdArray::from_shape_vec(&[2, 3, 5], make_real_vec(30, 4.1)).unwrap();
    let r2_f32 = NdArray::from_shape_vec(
        &[4, 5],
        make_real_vec(20, 2.1).into_iter().map(|x| x as f32).collect(),
    )
    .unwrap();

    // Complex spectra for direct irfft tests (including n=1 and non-Hermitian inputs to match pocketfft symmetry handling)
    let c_spec = NdArray::from_vec(make_complex_vec(5, 5.5));
    let c_one = NdArray::from_vec(vec![Complex64::new(3.5, -7.25)]);
    let c2_spec = NdArray::from_shape_vec(&[4, 5], make_complex_vec(20, 6.2)).unwrap();

    let mut script = String::new();
    for (name, arr) in [("re", &r_even), ("ro", &r_odd), ("r1", &r_one), ("r2", &r2), ("r3", &r3)] {
        script.push_str(&py_assign_f64(name, arr));
        script.push('\n');
    }
    script.push_str(&py_assign_f32("r2f", &r2_f32));
    script.push('\n');
    script.push_str(&py_assign_c128("cs", &c_spec));
    script.push('\n');
    script.push_str(&py_assign_c128("c1", &c_one));
    script.push('\n');
    script.push_str(&py_assign_c128("c2", &c2_spec));
    script.push('\n');

    script.push_str(
        "emit_arr('rf_even', np.fft.rfft(re))\n\
         emit_arr('rf_odd', np.fft.rfft(ro, norm='ortho'))\n\
         emit_arr('rf_one', np.fft.rfft(r1, norm='forward'))\n\
         emit_arr('rf_crop', np.fft.rfft(re, n=5))\n\
         emit_arr('rf_pad', np.fft.rfft(ro, n=12, norm='forward'))\n\
         emit_arr('ir_def', np.fft.irfft(cs))\n\
         emit_arr('ir_odd', np.fft.irfft(cs, n=7, norm='ortho'))\n\
         emit_arr('ir_even', np.fft.irfft(cs, n=8, norm='forward'))\n\
         emit_arr('ir_crop', np.fft.irfft(cs, n=4))\n\
         emit_arr('ir_pad', np.fft.irfft(cs, n=13))\n\
         emit_arr('ir_n1', np.fft.irfft(c1, n=1, norm='ortho'))\n\
         emit_arr('r2_def', np.fft.rfft2(r2, norm='ortho'))\n\
         emit_arr('r2_ax', np.fft.rfft2(r2, s=(7, 4), axes=(1, 0), norm='forward'))\n\
         emit_arr('r2_neg1', np.fft.rfft2(r2, s=(-1, 8)))\n\
         emit_arr('ir2_def', np.fft.irfft2(c2))\n\
         emit_arr('ir2_odd', np.fft.irfft2(c2, s=(5, 7), norm='ortho'))\n\
         emit_arr('ir2_axes', np.fft.irfft2(c2, s=(4, -1), axes=(1, 0), norm='forward'))\n\
         emit_arr('rn_def', np.fft.rfftn(r3, norm='backward'))\n\
         emit_arr('rn_sub', np.fft.rfftn(r3, s=(4, 6), axes=(0, 1), norm='ortho'))\n\
         emit_arr('rn_rep', np.fft.rfftn(r2, s=(4, 5), axes=(0, 0)))\n\
         emit_arr('irn_def', np.fft.irfftn(np.fft.rfftn(r3, s=(2, 3, 5)), s=(2, 3, 5)))\n\
         emit_arr('irn_none', np.fft.irfftn(c2))\n\
         emit_arr('r2f_out', np.fft.rfft2(r2f, norm='ortho'))\n",
    );
    for (tag, expr) in [
        ("rn_empty", "np.fft.rfftn(r3, axes=())"),
        ("irn_empty", "np.fft.irfftn(c2, axes=())"),
    ] {
        script.push_str(&format!(
            "try:\n    emit_arr('{tag}', {expr})\nexcept Exception as e:\n    emit_err('{tag}', e)\n"
        ));
    }
    let oracle = run_oracle(&script);

    assert_c128_matches(&rfft(&r_even, None, -1, Norm::Backward).unwrap(), "rf_even", &oracle, 1e-12, 1e-12, false, "rf_even");
    assert_c128_matches(&rfft(&r_odd, None, -1, Norm::Ortho).unwrap(), "rf_odd", &oracle, 1e-12, 1e-12, false, "rf_odd");
    assert_c128_matches(&rfft(&r_one, None, -1, Norm::Forward).unwrap(), "rf_one", &oracle, 1e-12, 1e-12, false, "rf_one");
    assert_c128_matches(&rfft(&r_even, Some(5), -1, Norm::Backward).unwrap(), "rf_crop", &oracle, 1e-12, 1e-12, false, "rf_crop");
    assert_c128_matches(&rfft(&r_odd, Some(12), -1, Norm::Forward).unwrap(), "rf_pad", &oracle, 1e-12, 1e-12, false, "rf_pad");

    assert_f64_matches(&irfft(&c_spec, None, -1, Norm::Backward).unwrap(), "ir_def", &oracle, 1e-12, 1e-12, false, "ir_def");
    assert_f64_matches(&irfft(&c_spec, Some(7), -1, Norm::Ortho).unwrap(), "ir_odd", &oracle, 1e-12, 1e-12, false, "ir_odd");
    assert_f64_matches(&irfft(&c_spec, Some(8), -1, Norm::Forward).unwrap(), "ir_even", &oracle, 1e-12, 1e-12, false, "ir_even");
    assert_f64_matches(&irfft(&c_spec, Some(4), -1, Norm::Backward).unwrap(), "ir_crop", &oracle, 1e-12, 1e-12, false, "ir_crop");
    assert_f64_matches(&irfft(&c_spec, Some(13), -1, Norm::Backward).unwrap(), "ir_pad", &oracle, 1e-12, 1e-12, false, "ir_pad");
    assert_f64_matches(&irfft(&c_one, Some(1), -1, Norm::Ortho).unwrap(), "ir_n1", &oracle, 1e-12, 1e-12, false, "ir_n1");

    assert_c128_matches(&rfft2(&r2, None, None, Norm::Ortho).unwrap(), "r2_def", &oracle, 1e-12, 1e-12, false, "r2_def");
    assert_c128_matches(&rfft2(&r2, Some(&[7, 4]), Some(&[1, 0]), Norm::Forward).unwrap(), "r2_ax", &oracle, 1e-12, 1e-12, false, "r2_ax");
    assert_c128_matches(&rfft2(&r2, Some(&[-1, 8]), None, Norm::Backward).unwrap(), "r2_neg1", &oracle, 1e-12, 1e-12, false, "r2_neg1");
    assert_f64_matches(&irfft2(&c2_spec, None, None, Norm::Backward).unwrap(), "ir2_def", &oracle, 1e-12, 1e-12, false, "ir2_def");
    assert_f64_matches(&irfft2(&c2_spec, Some(&[5, 7]), None, Norm::Ortho).unwrap(), "ir2_odd", &oracle, 1e-12, 1e-12, false, "ir2_odd");
    assert_f64_matches(&irfft2(&c2_spec, Some(&[4, -1]), Some(&[1, 0]), Norm::Forward).unwrap(), "ir2_axes", &oracle, 1e-12, 1e-12, false, "ir2_axes");

    let r3_spec = rfftn(&r3, None, None, Norm::Backward).unwrap();
    assert_c128_matches(&r3_spec, "rn_def", &oracle, 1e-12, 1e-12, false, "rn_def");
    assert_c128_matches(&rfftn(&r3, Some(&[4, 6]), Some(&[0, 1]), Norm::Ortho).unwrap(), "rn_sub", &oracle, 1e-12, 1e-12, false, "rn_sub");
    if oracle.errors.contains_key("rn_empty") {
        assert!(rfftn(&r3, None, Some(&[]), Norm::Backward).is_err(), "rfftn axes=() should error matching NumPy {:?}", oracle.errors.get("rn_empty"));
    } else {
        assert_c128_matches(&rfftn(&r3, None, Some(&[]), Norm::Backward).unwrap(), "rn_empty", &oracle, 1e-12, 1e-12, false, "rn_empty");
    }
    assert_c128_matches(&rfftn(&r2, Some(&[4, 5]), Some(&[0, 0]), Norm::Backward).unwrap(), "rn_rep", &oracle, 1e-12, 1e-12, false, "rn_rep");
    assert_f64_matches(&irfftn(&r3_spec, Some(&[2, 3, 5]), None, Norm::Backward).unwrap(), "irn_def", &oracle, 1e-12, 1e-12, false, "irn_def");
    assert_f64_matches(&irfftn(&c2_spec, None, None, Norm::Backward).unwrap(), "irn_none", &oracle, 1e-12, 1e-12, false, "irn_none");
    if oracle.errors.contains_key("irn_empty") {
        assert!(irfftn(&c2_spec, None, Some(&[]), Norm::Backward).is_err(), "irfftn axes=() should error matching NumPy {:?}", oracle.errors.get("irn_empty"));
    } else {
        assert_f64_matches(&irfftn(&c2_spec, None, Some(&[]), Norm::Backward).unwrap(), "irn_empty", &oracle, 1e-12, 1e-12, false, "irn_empty");
    }

    let r2f_out = rfft2(&r2_f32, None, None, Norm::Ortho).unwrap();
    assert_c64_matches(&r2f_out, "r2f_out", &oracle, 1e-5, 1e-5, false, "r2f_out");
    let r2f_rt = irfft2(&r2f_out, Some(&[4, 5]), None, Norm::Ortho).unwrap();
    assert_f32_close(&r2f_rt.to_vec(), &r2_f32.to_vec(), 1e-5, 1e-5, false, "rfft2/irfft2 f32 roundtrip");

    // Complex input to rfft/rfft2/rfftn via DynArray must error
    let dyn_c = DynArray::C128(c_spec);
    assert!(rfft(&dyn_c, None, -1, Norm::Backward).is_err());
    assert!(rfft2(&dyn_c, None, None, Norm::Backward).is_err());
    assert!(rfftn(&dyn_c, None, None, Norm::Backward).is_err());
}

#[test]
fn test_hfft_and_ihfft_against_oracle() {
    let r1 = NdArray::from_vec(make_real_vec(9, 0.4));
    let r_even = NdArray::from_vec(make_real_vec(8, 1.2));
    let r_one = NdArray::from_vec(vec![-2.5]);
    let r2 = NdArray::from_shape_vec(&[3, 7], make_real_vec(21, 3.1)).unwrap();
    let c1 = NdArray::from_vec(make_complex_vec(5, 2.0));
    let c_one = NdArray::from_vec(vec![Complex64::new(6.0, -3.0)]);
    let c2 = NdArray::from_shape_vec(&[4, 5], make_complex_vec(20, 4.4)).unwrap();
    let r_f32 = NdArray::from_vec(make_real_vec(10, 0.8).into_iter().map(|x| x as f32).collect());
    let c_f32 = NdArray::from_vec(
        make_complex_vec(6, 1.9)
            .into_iter()
            .map(|z| Complex32::new(z.re as f32, z.im as f32))
            .collect(),
    );

    let mut script = String::new();
    for (name, arr) in [("r1", &r1), ("re", &r_even), ("rone", &r_one), ("r2", &r2)] {
        script.push_str(&py_assign_f64(name, arr));
        script.push('\n');
    }
    for (name, arr) in [("c1", &c1), ("cone", &c_one), ("c2", &c2)] {
        script.push_str(&py_assign_c128(name, arr));
        script.push('\n');
    }
    script.push_str(&py_assign_f32("rf32", &r_f32));
    script.push('\n');
    script.push_str(&py_assign_c64("cf32", &c_f32));
    script.push('\n');

    for norm in ["backward", "ortho", "forward"] {
        script.push_str(&format!(
            "emit_arr('ih_r1_{norm}', np.fft.ihfft(r1, norm='{norm}'))\n\
             emit_arr('ih_re_{norm}', np.fft.ihfft(re, norm='{norm}'))\n\
             emit_arr('hf_c1_def_{norm}', np.fft.hfft(c1, norm='{norm}'))\n\
             emit_arr('hf_c1_n7_{norm}', np.fft.hfft(c1, n=7, norm='{norm}'))\n\
             emit_arr('hf_c1_n12_{norm}', np.fft.hfft(c1, n=12, norm='{norm}'))\n"
        ));
    }
    script.push_str(
        "emit_arr('ih_n1', np.fft.ihfft(rone, n=1))\n\
         emit_arr('ih_pad', np.fft.ihfft(r1, n=14))\n\
         emit_arr('ih_crop', np.fft.ihfft(r1, n=4))\n\
         emit_arr('ih_ax0', np.fft.ihfft(r2, axis=0, norm='ortho'))\n\
         emit_arr('hf_n1', np.fft.hfft(cone, n=1, norm='forward'))\n\
         emit_arr('hf_ax0', np.fft.hfft(c2, axis=0, norm='ortho'))\n\
         emit_arr('ih_f32', np.fft.ihfft(rf32, norm='ortho'))\n\
         emit_arr('hf_f32', np.fft.hfft(cf32, n=9, norm='backward'))\n",
    );
    let oracle = run_oracle(&script);

    let norms = [
        ("backward", Norm::Backward),
        ("ortho", Norm::Ortho),
        ("forward", Norm::Forward),
    ];
    for &(norm_name, norm) in &norms {
        assert_c128_matches(&ihfft(&r1, None, -1, norm).unwrap(), &format!("ih_r1_{norm_name}"), &oracle, 1e-12, 1e-12, false, "ih_r1");
        assert_c128_matches(&ihfft(&r_even, None, -1, norm).unwrap(), &format!("ih_re_{norm_name}"), &oracle, 1e-12, 1e-12, false, "ih_re");
        assert_f64_matches(&hfft(&c1, None, -1, norm).unwrap(), &format!("hf_c1_def_{norm_name}"), &oracle, 1e-12, 1e-12, false, "hf_def");
        assert_f64_matches(&hfft(&c1, Some(7), -1, norm).unwrap(), &format!("hf_c1_n7_{norm_name}"), &oracle, 1e-12, 1e-12, false, "hf_n7");
        assert_f64_matches(&hfft(&c1, Some(12), -1, norm).unwrap(), &format!("hf_c1_n12_{norm_name}"), &oracle, 1e-12, 1e-12, false, "hf_n12");
    }

    assert_c128_matches(&ihfft(&r_one, Some(1), -1, Norm::Backward).unwrap(), "ih_n1", &oracle, 1e-12, 1e-12, false, "ih_n1");
    assert_c128_matches(&ihfft(&r1, Some(14), -1, Norm::Backward).unwrap(), "ih_pad", &oracle, 1e-12, 1e-12, false, "ih_pad");
    assert_c128_matches(&ihfft(&r1, Some(4), -1, Norm::Backward).unwrap(), "ih_crop", &oracle, 1e-12, 1e-12, false, "ih_crop");
    assert_c128_matches(&ihfft(&r2, None, 0, Norm::Ortho).unwrap(), "ih_ax0", &oracle, 1e-12, 1e-12, false, "ih_ax0");
    assert_f64_matches(&hfft(&c_one, Some(1), -1, Norm::Forward).unwrap(), "hf_n1", &oracle, 1e-12, 1e-12, false, "hf_n1");
    assert_f64_matches(&hfft(&c2, None, 0, Norm::Ortho).unwrap(), "hf_ax0", &oracle, 1e-12, 1e-12, false, "hf_ax0");

    assert_c64_matches(&ihfft(&r_f32, None, -1, Norm::Ortho).unwrap(), "ih_f32", &oracle, 1e-5, 1e-5, false, "ih_f32");
    assert_f32_matches(&hfft(&c_f32, Some(9), -1, Norm::Backward).unwrap(), "hf_f32", &oracle, 1e-5, 1e-5, false, "hf_f32");

    // Complex input to ihfft via DynArray must error
    assert!(ihfft(&DynArray::C128(c1), None, -1, Norm::Backward).is_err());
}

#[test]
fn test_fftfreq_rfftfreq_and_shifts_against_oracle() {
    let a1_even = NdArray::from_vec(make_real_vec(8, 0.1));
    let a1_odd = NdArray::from_vec(make_real_vec(7, 0.2));
    let a1_one = NdArray::from_vec(vec![9.5]);
    let a2 = NdArray::from_shape_vec(&[4, 5], make_real_vec(20, 0.5)).unwrap();
    let scalar0 = NdArray::from_scalar(5.0f64);

    let mut script = String::new();
    for (name, arr) in [("e1", &a1_even), ("o1", &a1_odd), ("one1", &a1_one), ("a2", &a2), ("s0", &scalar0)] {
        script.push_str(&py_assign_f64(name, arr));
        script.push('\n');
    }
    for (i, &(n, d)) in [(1usize, 1.0f64), (2, 0.5), (7, 0.25), (8, 0.125), (9, -0.5)].iter().enumerate() {
        script.push_str(&format!(
            "emit_arr('ff_{i}', np.fft.fftfreq({n}, d={d}))\n\
             emit_arr('rf_{i}', np.fft.rfftfreq({n}, d={d}))\n"
        ));
    }
    script.push_str(
        "emit_arr('fs_e1', np.fft.fftshift(e1))\n\
         emit_arr('is_e1', np.fft.ifftshift(e1))\n\
         emit_arr('fs_o1', np.fft.fftshift(o1))\n\
         emit_arr('is_o1', np.fft.ifftshift(o1))\n\
         emit_arr('fs_one1', np.fft.fftshift(one1))\n\
         emit_arr('is_one1', np.fft.ifftshift(one1))\n\
         emit_arr('fs_a2_all', np.fft.fftshift(a2))\n\
         emit_arr('is_a2_all', np.fft.ifftshift(a2))\n\
         emit_arr('fs_a2_0', np.fft.fftshift(a2, axes=(0,)))\n\
         emit_arr('fs_a2_neg1', np.fft.fftshift(a2, axes=(-1,)))\n\
         emit_arr('fs_a2_rep', np.fft.fftshift(a2, axes=(0, 0)))\n\
         emit_arr('is_a2_rep', np.fft.ifftshift(a2, axes=(1, 1)))\n\
         emit_arr('fs_a2_empty', np.fft.fftshift(a2, axes=()))\n\
         try:\n    emit_arr('fs_s0', np.fft.fftshift(s0))\n\
         except Exception as e:\n    emit_err('fs_s0', e)\n\
         try:\n    emit_arr('is_s0', np.fft.ifftshift(s0))\n\
         except Exception as e:\n    emit_err('is_s0', e)\n",
    );
    let oracle = run_oracle(&script);

    for (i, &(n, d)) in [(1usize, 1.0f64), (2, 0.5), (7, 0.25), (8, 0.125), (9, -0.5)].iter().enumerate() {
        let ff = fftfreq(n, d).unwrap();
        assert_f64_matches(&ff, &format!("ff_{i}"), &oracle, 1e-14, 1e-14, false, &format!("ff_{i}"));
        let rf = rfftfreq(n, d).unwrap();
        assert_f64_matches(&rf, &format!("rf_{i}"), &oracle, 1e-14, 1e-14, false, &format!("rf_{i}"));
        let ff32 = fftfreq_f32(n, d as f32).unwrap();
        let exp_f32: Vec<f32> = oracle.arr(&format!("ff_{i}")).to_f64_vec().into_iter().map(|x| x as f32).collect();
        assert_f32_close(&ff32.to_vec(), &exp_f32, 1e-6, 1e-6, false, &format!("ff32_{i}"));
        let rf32 = rfftfreq_f32(n, d as f32).unwrap();
        let exp_rf32: Vec<f32> = oracle.arr(&format!("rf_{i}")).to_f64_vec().into_iter().map(|x| x as f32).collect();
        assert_f32_close(&rf32.to_vec(), &exp_rf32, 1e-6, 1e-6, false, &format!("rf32_{i}"));
    }

    assert!(fftfreq(0, 1.0).is_err());
    assert!(fftfreq(4, 0.0).is_err());
    let nonfinite = run_oracle("emit_arr('nan_freq', np.fft.fftfreq(4, np.nan))");
    assert_f64_matches(&fftfreq(4, f64::NAN).unwrap(), "nan_freq", &nonfinite, 0.0, 0.0, true, "nan sample spacing");
    assert!(rfftfreq(0, 1.0).is_err());
    assert!(rfftfreq(4, 0.0).is_err());

    assert_f64_bits_eq(&fftshift(&a1_even, None).unwrap().to_vec(), &oracle.arr("fs_e1").to_f64_vec(), "fs_e1");
    assert_f64_bits_eq(&ifftshift(&a1_even, None).unwrap().to_vec(), &oracle.arr("is_e1").to_f64_vec(), "is_e1");
    assert_f64_bits_eq(&fftshift(&a1_odd, None).unwrap().to_vec(), &oracle.arr("fs_o1").to_f64_vec(), "fs_o1");
    assert_f64_bits_eq(&ifftshift(&a1_odd, None).unwrap().to_vec(), &oracle.arr("is_o1").to_f64_vec(), "is_o1");
    assert_f64_bits_eq(&fftshift(&a1_one, None).unwrap().to_vec(), &oracle.arr("fs_one1").to_f64_vec(), "fs_one1");
    assert_f64_bits_eq(&ifftshift(&a1_one, None).unwrap().to_vec(), &oracle.arr("is_one1").to_f64_vec(), "is_one1");
    assert_f64_bits_eq(&fftshift(&a2, None).unwrap().to_vec(), &oracle.arr("fs_a2_all").to_f64_vec(), "fs_a2_all");
    assert_f64_bits_eq(&ifftshift(&a2, None).unwrap().to_vec(), &oracle.arr("is_a2_all").to_f64_vec(), "is_a2_all");
    assert_f64_bits_eq(&fftshift(&a2, Some(&[0])).unwrap().to_vec(), &oracle.arr("fs_a2_0").to_f64_vec(), "fs_a2_0");
    assert_f64_bits_eq(&fftshift(&a2, Some(&[-1])).unwrap().to_vec(), &oracle.arr("fs_a2_neg1").to_f64_vec(), "fs_a2_neg1");
    assert_f64_bits_eq(&fftshift(&a2, Some(&[0, 0])).unwrap().to_vec(), &oracle.arr("fs_a2_rep").to_f64_vec(), "fs_a2_rep");
    assert_f64_bits_eq(&ifftshift(&a2, Some(&[1, 1])).unwrap().to_vec(), &oracle.arr("is_a2_rep").to_f64_vec(), "is_a2_rep");
    assert_f64_bits_eq(&fftshift(&a2, Some(&[])).unwrap().to_vec(), &oracle.arr("fs_a2_empty").to_f64_vec(), "fs_a2_empty");

    // 0-D scalar fftshift/ifftshift must match NumPy 2.5.4 (which raises ValueError)
    if oracle.errors.contains_key("fs_s0") {
        assert!(
            fftshift(&scalar0, None).is_err(),
            "fftshift on 0-D scalar should return Err matching NumPy 2.5.4 {:?}",
            oracle.errors.get("fs_s0")
        );
    }
    if oracle.errors.contains_key("is_s0") {
        assert!(
            ifftshift(&scalar0, None).is_err(),
            "ifftshift on 0-D scalar should return Err matching NumPy 2.5.4 {:?}",
            oracle.errors.get("is_s0")
        );
    }
}

#[test]
fn test_fft_non_contiguous_strided_and_broadcast_views_against_oracle() {
    let base = NdArray::from_shape_vec(&[6, 5], make_real_vec(30, 2.3)).unwrap();
    let views = non_contiguous_2d_views(&base);

    let step_slice = base
        .slice(&[
            SliceArg::Slice(Slice::full().step_by(-2)),
            SliceArg::Slice(Slice::full().step_by(2)),
        ])
        .unwrap();
    let row = NdArray::from_shape_vec(&[1, 5], make_real_vec(5, 0.9)).unwrap();
    let bcast = row.broadcast_to(&[4, 5]).unwrap();

    let mut script = String::new();
    for (name, v) in &views {
        script.push_str(&py_assign_f64(name, &v.to_c_contiguous()));
        script.push('\n');
        script.push_str(&format!(
            "emit_arr('fft_{name}', np.fft.fft({name}, axis=-1))\n\
             emit_arr('rfft_{name}', np.fft.rfft({name}, axis=0, norm='ortho'))\n\
             emit_arr('fft2_{name}', np.fft.fft2({name}, norm='ortho'))\n\
             emit_arr('shift_{name}', np.fft.fftshift({name}))\n"
        ));
    }
    script.push_str(&py_assign_f64("stepv", &step_slice.to_c_contiguous()));
    script.push('\n');
    script.push_str(&py_assign_f64("bcastv", &bcast.to_c_contiguous()));
    script.push('\n');
    script.push_str(
        "emit_arr('fft_step', np.fft.fft2(stepv))\n\
         emit_arr('rfft_step', np.fft.rfft2(stepv))\n\
         emit_arr('fft_bcast', np.fft.fft(bcastv, axis=0, norm='ortho'))\n\
         emit_arr('fft2_bcast', np.fft.fft2(bcastv))\n",
    );
    let oracle = run_oracle(&script);

    for (name, v) in &views {
        assert_c128_matches(&fft(v, None, -1, Norm::Backward).unwrap(), &format!("fft_{name}"), &oracle, 1e-12, 1e-12, false, &format!("fft_{name}"));
        assert_c128_matches(&rfft(v, None, 0, Norm::Ortho).unwrap(), &format!("rfft_{name}"), &oracle, 1e-12, 1e-12, false, &format!("rfft_{name}"));
        assert_c128_matches(&fft2(v, None, None, Norm::Ortho).unwrap(), &format!("fft2_{name}"), &oracle, 1e-12, 1e-12, false, &format!("fft2_{name}"));
        assert_f64_bits_eq(&fftshift(v, None).unwrap().to_vec(), &oracle.arr(&format!("shift_{name}")).to_f64_vec(), &format!("shift_{name}"));
    }

    assert_c128_matches(&fft2(&step_slice, None, None, Norm::Backward).unwrap(), "fft_step", &oracle, 1e-12, 1e-12, false, "fft_step");
    assert_c128_matches(&rfft2(&step_slice, None, None, Norm::Backward).unwrap(), "rfft_step", &oracle, 1e-12, 1e-12, false, "rfft_step");
    assert_c128_matches(&fft(&bcast, None, 0, Norm::Ortho).unwrap(), "fft_bcast", &oracle, 1e-12, 1e-12, false, "fft_bcast");
    assert_c128_matches(&fft2(&bcast, None, None, Norm::Backward).unwrap(), "fft2_bcast", &oracle, 1e-12, 1e-12, false, "fft2_bcast");
}

#[test]
fn test_fft_edge_cases_nan_inf_neg_zero_empty_scalar_dyn_and_large_batch() {
    let nz = NdArray::from_vec(vec![-0.0f64, -0.0, -0.0, -0.0]);
    let with_nan = NdArray::from_vec(vec![1.0f64, f64::NAN, 2.0, -3.0]);
    let with_inf = NdArray::from_vec(vec![1.0f64, f64::INFINITY, -2.0, 0.5]);
    // Non-finite imaginary parts on DC and Nyquist of complex input to irfft/hfft
    let ir_bad_im = NdArray::from_vec(vec![
        Complex64::new(2.0, f64::NAN),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, f64::INFINITY),
    ]);
    let ir_bad_im_odd = NdArray::from_vec(vec![
        Complex64::new(2.0, f64::NEG_INFINITY),
        Complex64::new(0.5, 1.0),
        Complex64::new(-1.0, 0.25),
    ]);

    let empty_0_4: NdArray<f64> = NdArray::zeros(&[0, 4]).unwrap();
    let empty_4_0: NdArray<f64> = NdArray::zeros(&[4, 0]).unwrap();
    let scalar0 = NdArray::from_scalar(3.5f64);

    // DynArray dtypes: Bool, I8, I32, U8, F16
    let dyn_bool = DynArray::Bool(NdArray::from_vec(vec![true, false, true, true]));
    let dyn_i8 = DynArray::I8(NdArray::from_vec(vec![-3i8, 0, 5, -2]));
    let dyn_i32 = DynArray::I32(NdArray::from_vec(vec![10i32, -20, 30, 40]));
    let dyn_u8 = DynArray::U8(NdArray::from_vec(vec![1u8, 2, 3, 4]));
    let dyn_f16 = DynArray::F16(NdArray::from_vec(vec![
        f16::from_f32(1.0),
        f16::from_f32(-2.0),
        f16::from_f32(0.5),
        f16::from_f32(3.5),
    ]));

    // Large batch > 16_384 elements to trigger Rayon parallel path + strided axis=0 tiled kernel (16x37 tail)
    let large = NdArray::from_shape_vec(&[64, 300], make_complex_vec(64 * 300, 1.23)).unwrap();
    let large_r = NdArray::from_shape_vec(&[64, 300], make_real_vec(64 * 300, 4.56)).unwrap();
    let tiled_c = NdArray::from_shape_vec(&[16, 37], make_complex_vec(16 * 37, 2.71)).unwrap();
    let tiled_r = NdArray::from_shape_vec(&[16, 37], make_real_vec(16 * 37, 3.14)).unwrap();

    let mut script = String::new();
    for (name, arr) in [("nz", &nz), ("wnan", &with_nan), ("winf", &with_inf), ("e04", &empty_0_4), ("e40", &empty_4_0), ("sc0", &scalar0)] {
        script.push_str(&py_assign_f64(name, arr));
        script.push('\n');
    }
    script.push_str(&py_assign_c128("irbad", &ir_bad_im));
    script.push('\n');
    script.push_str(&py_assign_c128("irbad_o", &ir_bad_im_odd));
    script.push('\n');
    for (name, d) in [("db", &dyn_bool), ("di8", &dyn_i8), ("di32", &dyn_i32), ("du8", &dyn_u8), ("df16", &dyn_f16)] {
        script.push_str(&py_assign_dyn(name, d));
        script.push('\n');
    }
    script.push_str(&py_assign_c128("lg", &large));
    script.push('\n');
    script.push_str(&py_assign_f64("lgr", &large_r));
    script.push('\n');
    script.push_str(&py_assign_c128("tc", &tiled_c));
    script.push('\n');
    script.push_str(&py_assign_f64("tr", &tiled_r));
    script.push('\n');

    script.push_str(
        "emit_arr('fft_nz', np.fft.fft(nz))\n\
         emit_arr('rfft_nz', np.fft.rfft(nz))\n\
         emit_arr('fft_nan', np.fft.fft(wnan))\n\
         emit_arr('rfft_nan', np.fft.rfft(wnan))\n\
         emit_arr('fft_inf', np.fft.fft(winf))\n\
         emit_arr('rfft_inf', np.fft.rfft(winf))\n\
         emit_arr('ir_bad_even', np.fft.irfft(irbad, n=4))\n\
         emit_arr('ir_bad_odd', np.fft.irfft(irbad_o, n=5))\n\
         emit_arr('hf_bad_even', np.fft.hfft(irbad, n=4))\n\
         emit_arr('e04_pad', np.fft.fft(e04, n=6, axis=0))\n\
         emit_arr('e04_ax1', np.fft.fft(e04, axis=1))\n\
         emit_arr('e40_pad_r', np.fft.rfft(e40, n=6, axis=1))\n\
         emit_arr('dyn_bool_fft', np.fft.fft(db))\n\
         emit_arr('dyn_i8_rfft', np.fft.rfft(di8))\n\
         emit_arr('dyn_i32_ihfft', np.fft.ihfft(di32))\n\
         emit_arr('dyn_u8_fft', np.fft.fft(du8))\n\
         emit_arr('dyn_f16_fft', np.fft.fft(df16))\n\
         emit_arr('lg_fft', np.fft.fft(lg, axis=1, norm='ortho'))\n\
         emit_arr('lg_fft_ax0', np.fft.fft(lg, axis=0, norm='ortho'))\n\
         emit_arr('lgr_rfft', np.fft.rfft(lgr, axis=1, norm='ortho'))\n\
         emit_arr('tc_fft_ax0', np.fft.fft(tc, axis=0))\n\
         emit_arr('tc_ifft_ax0', np.fft.ifft(tc, axis=0))\n\
         emit_arr('tr_rfft_ax0', np.fft.rfft(tr, axis=0))\n\
         emit_arr('tc_irfft_ax0', np.fft.irfft(tc, n=30, axis=0))\n",
    );
    for (tag, expr) in [
        ("e04_err", "np.fft.fft(e04, axis=0)"),
        ("e40_err", "np.fft.fft(e40, axis=1)"),
        ("sc0_fft", "np.fft.fft(sc0)"),
        ("sc0_rfft", "np.fft.rfft(sc0)"),
        ("sc0_fftn", "np.fft.fftn(sc0)"),
    ] {
        script.push_str(&format!(
            "try:\n    emit_arr('{tag}', {expr})\nexcept Exception as e:\n    emit_err('{tag}', e)\n"
        ));
    }

    let oracle = run_oracle(&script);

    assert_c128_matches(&fft(&nz, None, -1, Norm::Backward).unwrap(), "fft_nz", &oracle, 0.0, 0.0, false, "fft_nz");
    assert_c128_matches(&rfft(&nz, None, -1, Norm::Backward).unwrap(), "rfft_nz", &oracle, 0.0, 0.0, false, "rfft_nz");
    assert_c128_matches(&fft(&with_nan, None, -1, Norm::Backward).unwrap(), "fft_nan", &oracle, 1e-12, 1e-12, true, "fft_nan");
    assert_c128_matches(&rfft(&with_nan, None, -1, Norm::Backward).unwrap(), "rfft_nan", &oracle, 1e-12, 1e-12, true, "rfft_nan");
    assert_c128_matches(&fft(&with_inf, None, -1, Norm::Backward).unwrap(), "fft_inf", &oracle, 1e-12, 1e-12, true, "fft_inf");
    assert_c128_matches(&rfft(&with_inf, None, -1, Norm::Backward).unwrap(), "rfft_inf", &oracle, 1e-12, 1e-12, true, "rfft_inf");

    assert_f64_matches(&irfft(&ir_bad_im, Some(4), -1, Norm::Backward).unwrap(), "ir_bad_even", &oracle, 1e-12, 1e-12, true, "ir_bad_even");
    assert_f64_matches(&irfft(&ir_bad_im_odd, Some(5), -1, Norm::Backward).unwrap(), "ir_bad_odd", &oracle, 1e-12, 1e-12, true, "ir_bad_odd");
    assert_f64_matches(&hfft(&ir_bad_im, Some(4), -1, Norm::Backward).unwrap(), "hf_bad_even", &oracle, 1e-12, 1e-12, true, "hf_bad_even");

    // Empty arrays
    assert_c128_matches(&fft(&empty_0_4, Some(6), 0, Norm::Backward).unwrap(), "e04_pad", &oracle, 0.0, 0.0, false, "e04_pad");
    assert_c128_matches(&fft(&empty_0_4, None, 1, Norm::Backward).unwrap(), "e04_ax1", &oracle, 0.0, 0.0, false, "e04_ax1");
    assert_c128_matches(&rfft(&empty_4_0, Some(6), 1, Norm::Backward).unwrap(), "e40_pad_r", &oracle, 0.0, 0.0, false, "e40_pad_r");
    if oracle.errors.contains_key("e04_err") {
        assert!(fft(&empty_0_4, None, 0, Norm::Backward).is_err(), "empty fft axis=0 should error like NumPy");
    }
    if oracle.errors.contains_key("e40_err") {
        assert!(fft(&empty_4_0, None, 1, Norm::Backward).is_err(), "empty fft axis=1 should error like NumPy");
    }

    // 0-D scalars
    assert_eq!(fft(&scalar0, None, -1, Norm::Backward).is_err(), oracle.errors.contains_key("sc0_fft"), "0-D fft vs oracle");
    assert_eq!(rfft(&scalar0, None, -1, Norm::Backward).is_err(), oracle.errors.contains_key("sc0_rfft"), "0-D rfft vs oracle");
    if oracle.errors.contains_key("sc0_fftn") {
        assert!(fftn(&scalar0, None, None, Norm::Backward).is_err(), "0-D fftn should match NumPy error");
    } else if let Ok(out) = fftn(&scalar0, None, None, Norm::Backward) {
        // A scalar has no transform axes. NumPy preserves its real dtype;
        // the typed Rust FFT API has a statically complex return type.
        assert_eq!(out.shape(), oracle.arr("sc0_fftn").shape);
        let expected = oracle.arr("sc0_fftn").to_f64_vec();
        assert_c128_close(&out.to_vec(), &expected.iter().map(|&v| Complex64::new(v, 0.0)).collect::<Vec<_>>(), 1e-12, 1e-12, false, "sc0_fftn");
    }

    // DynArray promotions (Bool/Int -> C128)
    assert_dyn_matches_oracle(&fft(&dyn_bool, None, -1, Norm::Backward).unwrap(), oracle.arr("dyn_bool_fft"), 1e-12, 1e-12, false, false, "dyn_bool_fft");
    assert_dyn_matches_oracle(&rfft(&dyn_i8, None, -1, Norm::Backward).unwrap(), oracle.arr("dyn_i8_rfft"), 1e-12, 1e-12, false, false, "dyn_i8_rfft");
    assert_dyn_matches_oracle(&ihfft(&dyn_i32, None, -1, Norm::Backward).unwrap(), oracle.arr("dyn_i32_ihfft"), 1e-12, 1e-12, false, false, "dyn_i32_ihfft");
    assert_dyn_matches_oracle(&fft(&dyn_u8, None, -1, Norm::Backward).unwrap(), oracle.arr("dyn_u8_fft"), 1e-12, 1e-12, false, false, "dyn_u8_fft");
    assert_dyn_matches_oracle(&fft(&dyn_f16, None, -1, Norm::Backward).unwrap(), oracle.arr("dyn_f16_fft"), 1e-3, 1e-3, false, false, "dyn_f16_fft");

    // Large Rayon batch (> 16_384 elements) + strided axis=0 tiled kernel (16x37 tail)
    assert_c128_matches(&fft(&large, None, 1, Norm::Ortho).unwrap(), "lg_fft", &oracle, 1e-11, 1e-11, false, "lg_fft");
    assert_c128_matches(&fft(&large, None, 0, Norm::Ortho).unwrap(), "lg_fft_ax0", &oracle, 1e-11, 1e-11, false, "lg_fft_ax0");
    assert_c128_matches(&rfft(&large_r, None, 1, Norm::Ortho).unwrap(), "lgr_rfft", &oracle, 1e-11, 1e-11, false, "lgr_rfft");
    assert_c128_matches(&fft(&tiled_c, None, 0, Norm::Backward).unwrap(), "tc_fft_ax0", &oracle, 1e-12, 1e-12, false, "tc_fft_ax0");
    assert_c128_matches(&ifft(&tiled_c, None, 0, Norm::Backward).unwrap(), "tc_ifft_ax0", &oracle, 1e-12, 1e-12, false, "tc_ifft_ax0");
    assert_c128_matches(&rfft(&tiled_r, None, 0, Norm::Backward).unwrap(), "tr_rfft_ax0", &oracle, 1e-12, 1e-12, false, "tr_rfft_ax0");
    assert_f64_matches(&irfft(&tiled_c, Some(30), 0, Norm::Backward).unwrap(), "tc_irfft_ax0", &oracle, 1e-12, 1e-12, false, "tc_irfft_ax0");
}

#[test]
fn test_p1_fft2_explicit_none_axes_matches_fftn_oracle() {
    use numpy_fft::{fft2_with_axes, ifft2_with_axes, irfft2_with_axes, rfft2_with_axes, Fft2Axes};

    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let x3 = NdArray::from_shape_vec(&[2, 3, 4], data).unwrap();
    let oracle = run_oracle(&format!(
        "{}\n{}",
        py_assign_f64("x3", &x3),
        r#"
emit_arr("fft2_default", np.fft.fft2(x3))
emit_arr("fft2_none", np.fft.fft2(x3, axes=None))
emit_arr("ifft2_none", np.fft.ifft2(np.fft.fft2(x3, axes=None), axes=None))
emit_arr("rfft2_none", np.fft.rfft2(x3, axes=None))
emit_arr("irfft2_none", np.fft.irfft2(np.fft.rfft2(x3, axes=None), s=(2, 3, 4), axes=None))
"#,
    ));

    let f_def = fft2_with_axes(&x3, None, Fft2Axes::Default, Norm::Backward).unwrap();
    let f_none = fft2_with_axes(&x3, None, Fft2Axes::None, Norm::Backward).unwrap();
    let if_none = ifft2_with_axes(&f_none, None, Fft2Axes::None, Norm::Backward).unwrap();
    let rf_none = rfft2_with_axes(&x3, None, Fft2Axes::None, Norm::Backward).unwrap();
    let irf_none = irfft2_with_axes(&rf_none, Some(&[2, 3, 4]), Fft2Axes::None, Norm::Backward).unwrap();

    assert_c128_matches(&f_def, "fft2_default", &oracle, 1e-12, 1e-12, false, "fft2_default");
    assert_c128_matches(&f_none, "fft2_none", &oracle, 1e-12, 1e-12, false, "fft2_none");
    assert_c128_matches(&if_none, "ifft2_none", &oracle, 1e-12, 1e-12, false, "ifft2_none");
    assert_c128_matches(&rf_none, "rfft2_none", &oracle, 1e-12, 1e-12, false, "rfft2_none");
    assert_f64_matches(&irf_none, "irfft2_none", &oracle, 1e-12, 1e-12, false, "irfft2_none");
    assert_eq!(f_none.to_vec(), fftn(&x3, None, None, Norm::Backward).unwrap().to_vec());
}

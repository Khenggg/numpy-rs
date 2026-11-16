use numpy_fft::{
    f16, fft, fft2, fftfreq, fftfreq_f32, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift,
    ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftfreq_f32, rfftn, Complex32,
    Complex64, DynArray, NdArray, Norm, NumPyError, Order, Slice, SliceArg,
};

fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn c32(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

fn assert_c64_close(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let dr = (a.re - e.re).abs();
        let di = (a.im - e.im).abs();
        assert!(
            dr <= tol && di <= tol,
            "mismatch at idx {i}: actual={a:?}, expected={e:?}, diff=({dr}, {di}), tol={tol}"
        );
    }
}

fn assert_f64_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "mismatch at idx {i}: actual={a}, expected={e}, diff={diff}, tol={tol}"
        );
    }
}

fn assert_c32_close(actual: &[Complex32], expected: &[Complex32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let dr = (a.re - e.re).abs();
        let di = (a.im - e.im).abs();
        assert!(
            dr <= tol && di <= tol,
            "mismatch at idx {i}: actual={a:?}, expected={e:?}, diff=({dr}, {di}), tol={tol}"
        );
    }
}

fn assert_f32_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "mismatch at idx {i}: actual={a}, expected={e}, diff={diff}, tol={tol}"
        );
    }
}

#[test]
fn test_fft_1d_prime_norms_and_roundtrip_numpy_oracle() {
    let x = NdArray::from_vec(vec![1.0_f64, 2.0, -1.5, 4.25, 0.5, -3.0, 2.5]);

    let exp_b = vec![
        c64(5.75, 0.0),
        c64(0.5274461886312247, -2.6985401487295633),
        c64(7.014592261586456, 4.070157624022361),
        c64(-6.917038450217681, -2.2662905774210147),
        c64(-6.917038450217681, 2.2662905774210147),
        c64(7.014592261586456, -4.070157624022361),
        c64(0.5274461886312247, 2.6985401487295633),
    ];
    let exp_o = vec![
        c64(2.1732957198030562, 0.0),
        c64(0.19935592072672628, -1.0199523052088109),
        c64(2.651266667525128, 1.5383749814281),
        c64(-2.6143947926210873, -0.856577323780711),
        c64(-2.6143947926210873, 0.856577323780711),
        c64(2.651266667525128, -1.5383749814281),
        c64(0.19935592072672628, 1.0199523052088109),
    ];
    let exp_f = vec![
        c64(0.8214285714285714, 0.0),
        c64(0.07534945551874639, -0.3855057355327947),
        c64(1.002084608798065, 0.5814510891460515),
        c64(-0.9881483500310972, -0.32375579677443067),
        c64(-0.9881483500310972, 0.32375579677443067),
        c64(1.002084608798065, -0.5814510891460515),
        c64(0.07534945551874639, 0.3855057355327947),
    ];

    for (norm, expected) in [
        (Norm::Backward, &exp_b),
        (Norm::Ortho, &exp_o),
        (Norm::Forward, &exp_f),
    ] {
        let y = fft(&x, None, -1, norm).unwrap();
        assert_eq!(y.shape(), &[7]);
        assert_c64_close(&y.to_vec(), expected, 1e-11);

        let inv = ifft(&y, None, -1, norm).unwrap();
        let x_c: Vec<Complex64> = x.to_vec().into_iter().map(|v| c64(v, 0.0)).collect();
        assert_c64_close(&inv.to_vec(), &x_c, 1e-11);
    }
}

#[test]
fn test_rfft_irfft_hfft_ihfft_numpy_oracle() {
    let x = NdArray::from_vec(vec![1.0_f64, 2.0, -1.5, 4.25, 0.5, -3.0, 2.5]);

    let r7 = rfft(&x, None, -1, Norm::Backward).unwrap();
    assert_eq!(r7.shape(), &[4]);
    assert_c64_close(
        &r7.to_vec(),
        &[
            c64(5.75, 0.0),
            c64(0.5274461886312256, -2.6985401487295633),
            c64(7.014592261586455, 4.070157624022361),
            c64(-6.917038450217679, -2.2662905774210147),
        ],
        1e-11,
    );

    // Inverse roundtrip (requires explicit odd n=7)
    let rec7 = irfft(&r7, Some(7), -1, Norm::Backward).unwrap();
    assert_f64_close(&rec7.to_vec(), &x.to_vec(), 1e-12);

    // Cropping & zero-padding on rfft
    let r5 = rfft(&x, Some(5), -1, Norm::Backward).unwrap();
    assert_c64_close(
        &r5.to_vec(),
        &[
            c64(6.25, 0.0),
            c64(-0.4522542485937371, 1.9531804262389902),
            c64(-0.17274575140626314, -6.350252847135843),
        ],
        1e-11,
    );

    let r10 = rfft(&x, Some(10), -1, Norm::Backward).unwrap();
    assert_c64_close(
        &r10.to_vec(),
        &[
            c64(5.75, 0.0),
            c64(1.4141352879691058, -2.6154054198116725),
            c64(-2.679711762656369, -0.42446086449889364),
            c64(8.960864712030896, 1.6164094438043008),
            c64(-5.195288237343632, -7.819715977867025),
            c64(-0.75, 0.0),
        ],
        1e-11,
    );

    // ihfft
    let ih7 = ihfft(&x, None, -1, Norm::Backward).unwrap();
    assert_c64_close(
        &ih7.to_vec(),
        &[
            c64(0.8214285714285714, 0.0),
            c64(0.07534945551874651, 0.3855057355327947),
            c64(1.002084608798065, -0.5814510891460515),
            c64(-0.9881483500310969, 0.32375579677443067),
        ],
        1e-12,
    );
    let h_rec = hfft(&ih7, Some(7), -1, Norm::Backward).unwrap();
    assert_f64_close(&h_rec.to_vec(), &x.to_vec(), 1e-12);

    // Explicit irfft / hfft from Hermitian-spectrum input
    let cx = NdArray::from_vec(vec![
        c64(1.0, 2.0),
        c64(-3.0, 0.5),
        c64(0.25, -1.25),
        c64(4.0, 0.0),
    ]);
    let ir_def = irfft(&cx, None, -1, Norm::Backward).unwrap();
    assert_eq!(ir_def.shape(), &[6]);
    assert_f64_close(
        &ir_def.to_vec(),
        &[
            -0.08333333333333333,
            -0.8251603157205569,
            0.7864851811257442,
            0.5833333333333333,
            1.796848152207589,
            -1.2581730176127763,
        ],
        1e-12,
    );

    let ir7 = irfft(&cx, Some(7), -1, Norm::Backward).unwrap();
    assert_f64_close(
        &ir7.to_vec(),
        &[
            0.5,
            -1.20063741762334,
            0.6875603320972608,
            0.3641325987722475,
            1.0465504402829997,
            1.2760281206617516,
            -1.6736340741909201,
        ],
        1e-12,
    );

    let hf_def = hfft(&cx, None, -1, Norm::Backward).unwrap();
    assert_f64_close(
        &hf_def.to_vec(),
        &[
            -0.5,
            -7.549038105676658,
            10.781088913245535,
            3.5,
            4.718911086754465,
            -4.950961894323342,
        ],
        1e-11,
    );

    let hf5 = hfft(&cx, Some(5), -1, Norm::Backward).unwrap();
    assert_f64_close(
        &hf5.to_vec(),
        &[
            -4.5,
            -1.7770170778731877,
            8.974037006467515,
            3.0431839204068014,
            -0.740203849001129,
        ],
        1e-11,
    );
}

#[test]
fn test_fft2_rfft2_irfft2_fftn_irfftn_numpy_oracle() {
    let a2 = NdArray::from_vec((0..12).map(|v| v as f64).collect())
        .reshape(&[3, 4])
        .unwrap();

    let f2 = fft2(&a2, Some(&[3, 2]), None, Norm::Ortho).unwrap();
    assert_eq!(f2.shape(), &[3, 2]);
    assert_c64_close(
        &f2.to_vec(),
        &[
            c64(11.022703842524301, 0.0),
            c64(-1.2247448713915892, 0.0),
            c64(-4.898979485566357, 2.8284271247461903),
            c64(0.0, 0.0),
            c64(-4.898979485566357, -2.8284271247461903),
            c64(0.0, 0.0),
        ],
        1e-11,
    );

    let rf2 = rfft2(&a2, Some(&[2, 5]), None, Norm::Forward).unwrap();
    assert_eq!(rf2.shape(), &[2, 3]);
    assert_c64_close(
        &rf2.to_vec(),
        &[
            c64(2.8, 0.0),
            c64(-0.870820393249937, -0.4530768593185974),
            c64(0.47082039324993696, -0.5428824546345146),
            c64(-1.6, 0.0),
            c64(0.1236067977499789, 0.38042260651806137),
            c64(-0.323606797749979, 0.2351141009169893),
        ],
        1e-12,
    );

    let r43 = rfft2(&a2, Some(&[4, 3]), None, Norm::Backward).unwrap();
    let ir43 = irfft2(&r43, Some(&[4, 3]), None, Norm::Backward).unwrap();
    assert_eq!(ir43.shape(), &[4, 3]);
    assert_f64_close(
        &ir43.to_vec(),
        &[0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0],
        1e-12,
    );

    // Repeated axes (`gh-27159`)
    let r3 = NdArray::from_vec((0..24).map(|v| v as f64).collect())
        .reshape(&[2, 3, 4])
        .unwrap();
    let rep = rfftn(&r3, Some(&[3, 2, 5]), Some(&[0, 0, 1]), Norm::Ortho).unwrap();
    assert_eq!(rep.shape(), &[3, 3, 4]);
    let exp_rep = vec![
        c64(4.38178046004133, 0.0),
        c64(5.477225575051662, 0.0),
        c64(6.572670690061994, 0.0),
        c64(7.6681158050723255, 0.0),
        c64(-1.9119416959380582, -3.106137575296109),
        c64(-1.7293675101030033, -3.6680431413538326),
        c64(-1.5467933242679477, -4.229948707411559),
        c64(-1.3642191384328926, -4.791854273469283),
        c64(-0.2789485340826061, 1.9196985952661803),
        c64(-0.09637434824755084, 2.0523465057913017),
        c64(0.0861998375875046, 2.184994416316424),
        c64(0.2687740234225598, 2.3176423268415456),
        c64(14.240786495134321, 5.692099788303083),
        c64(15.336231610144653, 5.692099788303083),
        c64(16.431676725154983, 5.692099788303083),
        c64(17.527121840165314, 5.692099788303083),
        c64(2.6509729449806305, -7.214604371765123),
        c64(2.8335471308156857, -7.776509937822847),
        c64(3.01612131665074, -8.338415503880572),
        c64(3.1986955024857964, -8.900321069938295),
        c64(0.6749603767908086, 4.062213088042789),
        c64(0.8575345626258642, 4.19486099856791),
        c64(1.04010874846092, 4.327508909093033),
        c64(1.222682934295975, 4.460156819618153),
        c64(14.240786495134321, -5.692099788303083),
        c64(15.336231610144653, -5.692099788303083),
        c64(16.431676725154983, -5.692099788303083),
        c64(17.527121840165314, -5.692099788303083),
        c64(-3.18852099182575, -9.11197096786615),
        c64(-3.0059468059906953, -9.673876533923874),
        c64(-2.8233726201556406, -10.235782099981598),
        c64(-2.640798434320584, -10.797687666039323),
        c64(2.053477900074975, 2.164846491941762),
        c64(2.2360520859100315, 2.2974944024668824),
        c64(2.418626271745086, 2.430142312992005),
        c64(2.601200457580141, 2.5627902235171267),
    ];
    assert_c64_close(&rep.to_vec(), &exp_rep, 1e-11);

    // irfftn with `s = [2, -1]` uses `a.shape[axes[-1]]` (`4`), not `2*(4-1)=6`
    let c3 = NdArray::from_vec((0..24).map(|v| c64(v as f64, (v as f64) * 0.5)).collect())
        .reshape(&[2, 3, 4])
        .unwrap();
    let irn = irfftn(&c3, Some(&[2, -1]), Some(&[0, 2]), Norm::Backward).unwrap();
    assert_eq!(irn.shape(), &[2, 3, 4]);
    assert_f64_close(
        &irn.to_vec(),
        &[
            7.0, -2.25, 0.0, 1.25, 11.0, -3.25, 0.0, 2.25, 15.0, -4.25, 0.0, 3.25, -6.0, 1.5, 0.0,
            -1.5, -6.0, 1.5, 0.0, -1.5, -6.0, 1.5, 0.0, -1.5,
        ],
        1e-12,
    );

    // Explicit `fftn`/`ifftn` roundtrip + empty axes passthrough
    let full = fftn(&c3, None, None, Norm::Backward).unwrap();
    let rec = ifftn(&full, None, None, Norm::Backward).unwrap();
    assert_c64_close(&rec.to_vec(), &c3.to_vec(), 1e-11);

    let pass = fftn(&r3, None, Some(&[]), Norm::Backward).unwrap();
    assert_eq!(pass.shape(), r3.shape());
    let r3_c: Vec<Complex64> = r3.to_vec().into_iter().map(|v| c64(v, 0.0)).collect();
    assert_c64_close(&pass.to_vec(), &r3_c, 1e-15);
}

#[test]
fn test_f32_and_complex32_precision_preservation() {
    let x32 = NdArray::from_vec(vec![1.0_f32, -2.0, 3.5, 0.25, -1.5, 4.0, 0.0, -0.5])
        .reshape(&[2, 4])
        .unwrap();
    let r32 = rfft(&x32, None, 1, Norm::Backward).unwrap();
    assert_eq!(r32.shape(), &[2, 3]);
    let ir32 = irfft(&r32, Some(4), 1, Norm::Backward).unwrap();
    assert_f32_close(&ir32.to_vec(), &x32.to_vec(), 1e-5);

    let cx32 = NdArray::from_vec(vec![
        c32(1.0, -0.5),
        c32(2.0, 1.25),
        c32(-1.5, 0.75),
        c32(0.5, -2.0),
        c32(3.0, 0.0),
        c32(-0.25, 1.5),
    ])
    .reshape(&[2, 3])
    .unwrap();
    let f2_32 = fft2(&cx32, None, None, Norm::Ortho).unwrap();
    let if2_32 = ifft2(&f2_32, None, None, Norm::Ortho).unwrap();
    assert_c32_close(&if2_32.to_vec(), &cx32.to_vec(), 1e-5);

    let h32 = ihfft(&x32, None, -1, Norm::Ortho).unwrap();
    let rh32 = hfft(&h32, Some(4), -1, Norm::Ortho).unwrap();
    assert_f32_close(&rh32.to_vec(), &x32.to_vec(), 1e-5);

    // f32 scalar fftfreq / rfftfreq
    let ff32 = fftfreq_f32(5, 0.5).unwrap();
    assert_f32_close(&ff32.to_vec(), &[0.0, 0.4, 0.8, -0.8, -0.4], 1e-6);
    let rff32 = rfftfreq_f32(6, 2.0).unwrap();
    assert_f32_close(&rff32.to_vec(), &[0.0, 1.0 / 12.0, 1.0 / 6.0, 0.25], 1e-6);
}

#[test]
fn test_dyn_array_auto_promotion_and_dtype_errors() {
    // NumPy 2.5 promotes f16 to complex64 / float32.
    let x16 = NdArray::from_vec(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(-1.0),
        f16::from_f32(0.5),
    ]);
    let dyn_f16 = DynArray::F16(x16);

    let fft_dyn16 = fft(&dyn_f16, None, -1, Norm::Backward).unwrap();
    match fft_dyn16 {
        DynArray::C64(arr) => {
            assert_c32_close(
                &arr.to_vec(),
                &[c32(2.5, 0.0), c32(2.0, -1.5), c32(-2.5, 0.0), c32(2.0, 1.5)],
                1e-6,
            );
        }
        other => panic!("expected C64, got {:?}", other.dtype()),
    }

    let rfft_dyn16 = rfft(&dyn_f16, None, -1, Norm::Backward).unwrap();
    match &rfft_dyn16 {
        DynArray::C64(arr) => {
            assert_c32_close(
                &arr.to_vec(),
                &[c32(2.5, 0.0), c32(2.0, -1.5), c32(-2.5, 0.0)],
                1e-6,
            );
        }
        other => panic!("expected C64, got {:?}", other.dtype()),
    }

    // Integer, unsigned, and bool inputs promote to C128 / F64
    let dyn_cases = vec![
        DynArray::Bool(NdArray::from_vec(vec![true, false, true, false])),
        DynArray::I8(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::I16(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::I32(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::I64(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::U8(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::U16(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::U32(NdArray::from_vec(vec![1, 0, 1, 0])),
        DynArray::U64(NdArray::from_vec(vec![1, 0, 1, 0])),
    ];
    for da in dyn_cases {
        match fft(&da, None, -1, Norm::Backward).unwrap() {
            DynArray::C128(arr) => {
                assert_c64_close(
                    &arr.to_vec(),
                    &[c64(2.0, 0.0), c64(0.0, 0.0), c64(2.0, 0.0), c64(0.0, 0.0)],
                    1e-12,
                );
            }
            other => panic!("expected C128 for integer/bool fft, got {:?}", other.dtype()),
        }
        match irfft(&da, Some(4), -1, Norm::Backward).unwrap() {
            DynArray::F64(arr) => {
                assert_f64_close(&arr.to_vec(), &[0.5, 0.0, 0.5, 0.0], 1e-12);
            }
            other => panic!("expected F64 for integer/bool irfft, got {:?}", other.dtype()),
        }
    }

    // f32 DynArray stays C64 / F32; complex DynArray rejected on real transforms
    let dyn_f32 = DynArray::F32(NdArray::from_vec(vec![1.0_f32, 0.0, 1.0, 0.0]));
    match rfft(&dyn_f32, None, -1, Norm::Backward).unwrap() {
        DynArray::C64(arr) => assert_eq!(arr.shape(), &[3]),
        other => panic!("expected C64, got {:?}", other.dtype()),
    }

    let dyn_c64 = DynArray::C64(NdArray::from_vec(vec![c32(1.0, 0.0); 4]));
    assert!(matches!(
        rfft(&dyn_c64, None, -1, Norm::Backward),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        rfft2(&dyn_c64, Some(&[2, 2]), Some(&[0, 0]), Norm::Backward),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        rfftn(&dyn_c64, None, None, Norm::Backward),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        ihfft(&dyn_c64, None, -1, Norm::Backward),
        Err(NumPyError::DTypeError(_))
    ));
    assert!(matches!(
        irfft(&dyn_c64, None, -1, Norm::Backward),
        Ok(DynArray::F32(_))
    ));
}

#[test]
fn test_non_contiguous_strided_transposed_negative_and_broadcast_views() {
    // 3D sliced with negative stride (`::-1`) and step 2 (`::2`)
    let base = NdArray::from_vec((0..24).map(|v| v as f64).collect())
        .reshape(&[2, 3, 4])
        .unwrap();
    let sliced = base
        .slice(&[
            SliceArg::Slice(Slice::new(None, None, -1)),
            SliceArg::Slice(Slice::full()),
            SliceArg::Slice(Slice::new(None, None, 2)),
        ])
        .unwrap();
    assert_eq!(sliced.shape(), &[2, 3, 2]);
    let out = fftn(&sliced, None, Some(&[-1, 0]), Norm::Ortho).unwrap();
    assert_c64_close(
        &out.to_vec(),
        &[
            c64(14.0, 0.0),
            c64(-2.0, 0.0),
            c64(22.0, 0.0),
            c64(-2.0, 0.0),
            c64(30.0, 0.0),
            c64(-2.0, 0.0),
            c64(12.0, 0.0),
            c64(0.0, 0.0),
            c64(12.0, 0.0),
            c64(0.0, 0.0),
            c64(12.0, 0.0),
            c64(0.0, 0.0),
        ],
        1e-11,
    );

    // Transposed and Fortran-contiguous views match C-contiguous copies
    let mat = NdArray::from_vec((0..20).map(|v| v as f64).collect())
        .reshape(&[4, 5])
        .unwrap();
    let mat_t_view = mat.view().transpose(None).unwrap();
    let mat_t_owned = mat_t_view.to_owned();
    assert_c64_close(
        &fft(&mat_t_view, None, 0, Norm::Backward).unwrap().to_vec(),
        &fft(&mat_t_owned, None, 0, Norm::Backward).unwrap().to_vec(),
        1e-12,
    );
    let mat_f = mat.to_f_contiguous();
    assert_c64_close(
        &rfft2(&mat_f, None, None, Norm::Ortho).unwrap().to_vec(),
        &rfft2(&mat, None, None, Norm::Ortho).unwrap().to_vec(),
        1e-12,
    );

    // 0-stride broadcast view
    let row = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]).reshape(&[1, 4]).unwrap();
    let bcast = row.broadcast_to(&[3, 4]).unwrap();
    let b_fft = fft2(&bcast, None, None, Norm::Backward).unwrap();
    let b_owned = fft2(&bcast.to_c_contiguous(), None, None, Norm::Backward).unwrap();
    assert_c64_close(&b_fft.to_vec(), &b_owned.to_vec(), 1e-12);
}

#[test]
fn test_helpers_fftfreq_and_fftshift_oracle() {
    assert_f64_close(
        &fftfreq(5, 0.5).unwrap().to_vec(),
        &[0.0, 0.4, 0.8, -0.8, -0.4],
        1e-15,
    );
    assert_f64_close(
        &fftfreq(6, 2.0).unwrap().to_vec(),
        &[
            0.0,
            0.08333333333333333,
            0.16666666666666666,
            -0.25,
            -0.16666666666666666,
            -0.08333333333333333,
        ],
        1e-15,
    );
    assert_f64_close(&rfftfreq(5, 0.5).unwrap().to_vec(), &[0.0, 0.4, 0.8], 1e-15);
    assert_f64_close(
        &rfftfreq(6, 2.0).unwrap().to_vec(),
        &[0.0, 0.08333333333333333, 0.16666666666666666, 0.25],
        1e-15,
    );

    let x5 = NdArray::from_vec(vec![0_i32, 1, 2, 3, 4]);
    let s5 = fftshift(&x5, None).unwrap();
    assert_eq!(s5.to_vec(), vec![3, 4, 0, 1, 2]);
    let is5 = ifftshift(&s5, None).unwrap();
    assert_eq!(is5.to_vec(), vec![0, 1, 2, 3, 4]);
    assert_eq!(ifftshift(&x5, None).unwrap().to_vec(), vec![2, 3, 4, 0, 1]);

    let x2 = NdArray::from_vec((0..15_i64).collect())
        .reshape(&[3, 5])
        .unwrap();
    let s2 = fftshift(&x2, Some(&[0, 1])).unwrap();
    assert_eq!(
        s2.to_vec(),
        vec![13, 14, 10, 11, 12, 3, 4, 0, 1, 2, 8, 9, 5, 6, 7]
    );
    let is2 = ifftshift(&x2, Some(&[-2, -1])).unwrap();
    assert_eq!(
        is2.to_vec(),
        vec![7, 8, 9, 5, 6, 12, 13, 14, 10, 11, 2, 3, 4, 0, 1]
    );

    // Repeated axes and empty axes in fftshift/ifftshift + DynArray + non-contiguous
    let rep_shift = fftshift(&x5, Some(&[0, 0])).unwrap();
    assert_eq!(rep_shift.to_vec(), vec![1, 2, 3, 4, 0]);
    let empty_ax = fftshift(&x2, Some(&[])).unwrap();
    assert_eq!(empty_ax.to_vec(), x2.to_vec());

    let x2_t = x2.t();
    let s2_t = fftshift(&x2_t, None).unwrap();
    assert_eq!(s2_t.shape(), &[5, 3]);
    assert_eq!(ifftshift(&s2_t, None).unwrap().to_vec(), x2_t.to_c_contiguous().to_vec());

    let dyn_shift = fftshift(&DynArray::I64(x2), None).unwrap();
    match dyn_shift {
        DynArray::I64(arr) => assert_eq!(arr.to_vec(), s2.to_vec()),
        other => panic!("expected I64 DynArray, got {:?}", other.dtype()),
    }
}

#[test]
fn test_edge_cases_nan_nyquist_crop_single_and_empty() {
    // DC and Nyquist imaginary parts are always ignored by irfft (matching NumPy 2.5.4 pocketfft); cropped NaN ignored
    let crop_nan = NdArray::from_vec(vec![
        c64(4.0, 0.0),
        c64(-1.0, 0.0),
        c64(2.0, 0.0),
        c64(999.0, f64::NAN),
    ]);
    let out_crop = irfft(&crop_nan, Some(4), -1, Norm::Backward).unwrap();
    assert_f64_close(&out_crop.to_vec(), &[1.0, 0.5, 2.0, 0.5], 1e-12);

    let fin_im = NdArray::from_vec(vec![c64(1.0, 5.0), c64(0.0, 0.0), c64(2.0, 9.0)]);
    let out_fin = irfft(&fin_im, Some(4), -1, Norm::Backward).unwrap();
    assert_f64_close(&out_fin.to_vec(), &[0.75, -0.25, 0.75, -0.25], 1e-12);

    let dc_inf = NdArray::from_vec(vec![
        c64(1.0, f64::INFINITY),
        c64(0.0, 0.0),
        c64(0.0, 0.0),
    ]);
    let out_dc_inf = irfft(&dc_inf, Some(4), -1, Norm::Backward).unwrap();
    assert_f64_close(&out_dc_inf.to_vec(), &[0.25, 0.25, 0.25, 0.25], 1e-12);

    let nyq_nan = NdArray::from_vec(vec![
        c64(1.0, 0.0),
        c64(0.0, 0.0),
        c64(2.0, f64::NAN),
    ]);
    let out_nyq_nan = irfft(&nyq_nan, Some(4), -1, Norm::Backward).unwrap();
    assert_f64_close(&out_nyq_nan.to_vec(), &[0.75, -0.25, 0.75, -0.25], 1e-12);

    // Length 1 transforms
    let one_c = NdArray::from_vec(vec![c64(3.5, 1.25)]);
    assert_c64_close(&fft(&one_c, None, -1, Norm::Ortho).unwrap().to_vec(), &[c64(3.5, 1.25)], 1e-15);
    let one_h = NdArray::from_vec(vec![c64(3.5, 2.0)]);
    assert_f64_close(&hfft(&one_h, Some(1), -1, Norm::Backward).unwrap().to_vec(), &[3.5], 1e-15);
    let one_r = NdArray::from_vec(vec![3.5_f64]);
    assert_c64_close(&ihfft(&one_r, None, -1, Norm::Backward).unwrap().to_vec(), &[c64(3.5, 0.0)], 1e-15);

    // Empty arrays: untransformed dim == 0 returns zeros of target shape
    let empty04 = NdArray::<f64>::zeros_with_order(&[0, 4], Order::C).unwrap();
    assert_eq!(fft(&empty04, Some(6), 1, Norm::Backward).unwrap().shape(), &[0, 6]);
    assert_eq!(rfft(&empty04, Some(5), 1, Norm::Backward).unwrap().shape(), &[0, 3]);
    assert_eq!(fft(&empty04, Some(4), 0, Norm::Backward).unwrap().shape(), &[4, 4]);
}

#[test]
fn test_error_conditions() {
    let x = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    assert!(matches!(fft(&x, Some(0), -1, Norm::Backward), Err(NumPyError::InvalidValue(_))));
    assert!(matches!(rfft(&x, Some(0), -1, Norm::Backward), Err(NumPyError::InvalidValue(_))));
    assert!(matches!(fft(&x, None, 2, Norm::Backward), Err(NumPyError::InvalidAxis { .. })));
    assert!(matches!(fft(&x, None, -3, Norm::Backward), Err(NumPyError::InvalidAxis { .. })));
    assert!(matches!(
        fftn(&x, Some(&[2]), Some(&[0, 1]), Norm::Backward),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        rfftn(&x, None, Some(&[]), Norm::Backward),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        irfftn(&x, None, Some(&[]), Norm::Backward),
        Err(NumPyError::InvalidValue(_))
    ));
    // irfft on length-1 input with n=None -> default_n = 0 -> InvalidValue
    let len1 = NdArray::from_vec(vec![c64(1.0, 0.0)]);
    assert!(matches!(irfft(&len1, None, -1, Norm::Backward), Err(NumPyError::InvalidValue(_))));

    // 1D array passed to fft2 defaults axes=[-2, -1] -> InvalidAxis
    let v1 = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
    assert!(matches!(fft2(&v1, None, None, Norm::Backward), Err(NumPyError::InvalidAxis { .. })));

    // fftfreq/rfftfreq errors
    assert!(matches!(fftfreq(0, 1.0), Err(NumPyError::InvalidValue(_))));
    assert!(matches!(fftfreq(4, 0.0), Err(NumPyError::InvalidValue(_))));
    assert!(matches!(rfftfreq(0, 1.0), Err(NumPyError::InvalidValue(_))));
    assert!(matches!(rfftfreq_f32(4, 0.0), Err(NumPyError::InvalidValue(_))));
}

#[test]
fn test_large_rayon_batched_parseval_and_concurrency() {
    // Exceeds RAYON_WORK_THRESHOLD (16_384) across prime & composite dimensions
    let rows = 65_usize;
    let cols = 301_usize;
    let data: Vec<f64> = (0..(rows * cols))
        .map(|i| ((i % 97) as f64 - 48.0) * 0.125)
        .collect();
    let mat = NdArray::from_vec(data)
        .reshape(&[rows as isize, cols as isize])
        .unwrap();

    // Parseval's theorem for Norm::Ortho
    let energy_in: f64 = mat.iter().map(|v| v * v).sum();
    let spec_2d = fft2(&mat, None, None, Norm::Ortho).unwrap();
    let energy_out: f64 = spec_2d.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (energy_in - energy_out).abs() <= 1e-8 * energy_in,
        "Parseval mismatch: in={energy_in}, out={energy_out}"
    );

    // 2D Real roundtrip across non-contiguous transposed view (`rows x cols` -> `cols x rows`)
    let mat_t = mat.t();
    let rt = rfft2(&mat_t, None, None, Norm::Forward).unwrap();
    let rec_t = irfft2(&rt, Some(&[cols as isize, rows as isize]), None, Norm::Forward).unwrap();
    assert_eq!(rec_t.shape(), &[cols, rows]);
    assert_f64_close(&rec_t.to_vec(), &mat_t.to_c_contiguous().to_vec(), 1e-10);

    // Non-last-axis large Rayon batch (axis 0 across 301 lanes)
    let ax0_f = fft(&mat, None, 0, Norm::Ortho).unwrap();
    let ax0_inv = ifft(&ax0_f, None, 0, Norm::Ortho).unwrap();
    let mat_c: Vec<Complex64> = mat.iter().map(|v| c64(v, 0.0)).collect();
    assert_c64_close(&ax0_inv.to_vec(), &mat_c, 1e-10);

    // Concurrent calls sharing `OnceLock`/`RwLock` planner cache
    std::thread::scope(|s| {
        for t in 0..4 {
            let mat_ref = &mat;
            s.spawn(move || {
                let n = 256 + t * 17;
                let res = rfft(mat_ref, Some(n), 1, Norm::Backward).unwrap();
                let inv = irfft(&res, Some(n), 1, Norm::Backward).unwrap();
                assert_eq!(inv.shape(), &[rows, n]);
            });
        }
    });
}


#[test]
fn test_hfft_ihfft_norms_3d_crop_pad_and_repeated_irfftn() {
    let x = NdArray::from_vec(vec![1.0_f64, 2.0, -1.5, 4.25, 0.5, -3.0, 2.5]);
    let exp_ih_b = vec![
        c64(0.5416666666666666, 0.0),
        c64(-0.5416666666666666, 0.43301270189221924),
        c64(1.0416666666666665, 1.010362971081845),
        c64(-0.5416666666666666, 0.0),
    ];
    let exp_ih_o = vec![
        c64(1.326806944007555, 0.0),
        c64(-1.326806944007555, 1.0606601717798212),
        c64(2.551551815399144, 2.4748737341529163),
        c64(-1.326806944007555, 0.0),
    ];
    let exp_ih_f = vec![
        c64(3.25, 0.0),
        c64(-3.25, 2.5980762113533156),
        c64(6.25, 6.0621778264910695),
        c64(-3.25, 0.0),
    ];
    for (norm, expected) in [
        (Norm::Backward, &exp_ih_b),
        (Norm::Ortho, &exp_ih_o),
        (Norm::Forward, &exp_ih_f),
    ] {
        let out = ihfft(&x, Some(6), -1, norm).unwrap();
        assert_c64_close(&out.to_vec(), expected, 1e-11);
    }

    let cx = NdArray::from_vec(vec![
        c64(1.0, 2.0),
        c64(-3.0, 0.5),
        c64(0.25, -1.25),
        c64(4.0, 0.0),
    ]);
    let exp_h_b = vec![
        -4.5,
        -1.7770170778731877,
        8.974037006467515,
        3.0431839204068014,
        -0.740203849001129,
    ];
    let exp_h_o = vec![
        -2.0124611797498106,
        -0.794706196660497,
        4.013311355812017,
        1.3609532228127834,
        -0.33102922471470286,
    ];
    let exp_h_f = vec![
        -0.9,
        -0.35540341557463756,
        1.7948074012935031,
        0.6086367840813603,
        -0.14804076980022582,
    ];
    for (norm, expected) in [
        (Norm::Backward, &exp_h_b),
        (Norm::Ortho, &exp_h_o),
        (Norm::Forward, &exp_h_f),
    ] {
        let out = hfft(&cx, Some(5), -1, norm).unwrap();
        assert_f64_close(&out.to_vec(), expected, 1e-11);
    }

    // 3D mixed pad + crop (`[2, 3, 4]` -> `[3, 2, 5]`) with `Norm::Forward`
    let a = NdArray::from_vec((0..24).map(|v| v as f64).collect())
        .reshape(&[2, 3, 4])
        .unwrap();
    let f3 = fftn(&a, Some(&[3, 2, 5]), None, Norm::Forward).unwrap();
    assert_eq!(f3.shape(), &[3, 2, 5]);
    let exp_f3 = vec![
        c64(5.066666666666666, 0.0),
        c64(-0.8277605243332493, -1.0628964525818545),
        c64(0.9610938576665826, -0.8321498382569882),
        c64(0.9610938576665826, 0.8321498382569882),
        c64(-0.8277605243332493, 1.0628964525818545),
        c64(-1.0666666666666664, 0.0),
        c64(0.08240453183331928, 0.25361507101204084),
        c64(-0.21573786516665266, 0.1567427339446595),
        c64(-0.21573786516665266, -0.1567427339446595),
        c64(0.08240453183331928, -0.25361507101204084),
        c64(-1.1333333333333333, -3.57957166897568),
        c64(-0.8112332406410006, 0.7703872447248159),
        c64(-0.8090829662332055, -0.5517838639578009),
        c64(0.31880950181655976, -0.8410512475802745),
        c64(0.7681733717243128, 0.16056765146155882),
        c64(-0.2666666666666667, 0.46188021535170043),
        c64(0.13041968009784063, 0.027721558775701278),
        c64(0.013937128435687118, 0.13260291938243646),
        c64(-0.12180606101901345, 0.05423155241010669),
        c64(-0.08921741418118102, -0.09908597673031924),
        c64(-1.1333333333333333, 3.57957166897568),
        c64(0.7681733717243128, -0.16056765146155882),
        c64(0.31880950181655976, 0.8410512475802745),
        c64(-0.8090829662332055, 0.5517838639578009),
        c64(-0.8112332406410006, -0.7703872447248159),
        c64(-0.2666666666666667, -0.46188021535170043),
        c64(-0.08921741418118102, 0.09908597673031924),
        c64(-0.12180606101901345, -0.05423155241010669),
        c64(0.013937128435687118, -0.13260291938243646),
        c64(0.13041968009784063, -0.027721558775701278),
    ];
    assert_c64_close(&f3.to_vec(), &exp_f3, 1e-11);

    // Repeated axes on irfftn (`axes=[0, 0, 2]`, `s=[2, 3, 3]`, `Norm::Ortho`)
    let c_rep = NdArray::from_vec((1..=12).map(|v| c64(v as f64, 0.0)).collect())
        .reshape(&[2, 3, 2])
        .unwrap();
    let ir_rep = irfftn(&c_rep, Some(&[2, 3, 3]), Some(&[0, 0, 2]), Norm::Ortho).unwrap();
    assert_eq!(ir_rep.shape(), &[3, 3, 3]);
    let exp_ir_rep = vec![
        2.357022603955159,
        -0.47140452079103173,
        -0.47140452079103173,
        5.185449728701348,
        -0.47140452079103135,
        -0.47140452079103135,
        8.013876853447542,
        -0.4714045207910324,
        -0.4714045207910324,
        8.720983634634088,
        1.649915822768611,
        -2.5927248643506746,
        11.549410759380278,
        1.6499158227686115,
        -2.592724864350674,
        14.37783788412647,
        1.649915822768611,
        -2.5927248643506746,
        8.720983634634088,
        -2.5927248643506746,
        1.649915822768611,
        11.549410759380278,
        -2.592724864350674,
        1.6499158227686115,
        14.37783788412647,
        -2.5927248643506746,
        1.649915822768611,
    ];
    assert_f64_close(&ir_rep.to_vec(), &exp_ir_rep, 1e-11);

    // 0-D scalar error parity
    let scalar0 = NdArray::from_vec(vec![3.0_f64]).reshape(&[]).unwrap();
    assert!(matches!(
        fft(&scalar0, None, -1, Norm::Backward),
        Err(NumPyError::InvalidAxis { .. })
    ));
}


#[test]
fn test_tiled_strided_axis0_all_kernels_and_batch_tails() {
    // `cols = 17` triggers `batch = 16` + `batch = 1` tail; `rows = 24` exercises `n % 8 == 0` tile padding.
    let rows = 24_usize;
    let cols = 17_usize;
    let data: Vec<f64> = (0..(rows * cols))
        .map(|i| ((i * 13 + 7) % 53) as f64 * 0.25 - 4.0)
        .collect();
    let mat = NdArray::from_vec(data.clone())
        .reshape(&[rows as isize, cols as isize])
        .unwrap();

    // Tiled c2c_into_buffer (`axis = 0`) with zero-padding (`n = 32`) & cropping (`n = 16`)
    for (n, norm) in [(None, Norm::Backward), (Some(32), Norm::Ortho), (Some(16), Norm::Forward)] {
        let out = fft(&mat, n, 0, norm).unwrap();
        let expected_rows = n.unwrap_or(rows);
        assert_eq!(out.shape(), &[expected_rows, cols]);
        // Compare against transforming along contiguous axis 1 of transposed copy
        let mat_t = mat.t().to_c_contiguous();
        let ref_t = fft(&mat_t, n, 1, norm).unwrap().t().to_c_contiguous();
        assert_c64_close(&out.to_vec(), &ref_t.to_vec(), 1e-11);
    }

    // Tiled r2c_into_buffer + c2r_into_buffer (`rfft`/`irfft` + `ihfft`/`hfft` on `axis = 0`)
    for n in [rows, 19, 32] {
        let rf0 = rfft(&mat, Some(n), 0, Norm::Ortho).unwrap();
        let mat_t = mat.t().to_c_contiguous();
        let rf_ref = rfft(&mat_t, Some(n), 1, Norm::Ortho).unwrap().t().to_c_contiguous();
        assert_c64_close(&rf0.to_vec(), &rf_ref.to_vec(), 1e-11);

        let irf0 = irfft(&rf0, Some(n), 0, Norm::Ortho).unwrap();
        let irf_ref = irfft(&rf_ref.t().to_c_contiguous(), Some(n), 1, Norm::Ortho)
            .unwrap()
            .t()
            .to_c_contiguous();
        assert_f64_close(&irf0.to_vec(), &irf_ref.to_vec(), 1e-11);

        let ih0 = ihfft(&mat, Some(n), 0, Norm::Forward).unwrap();
        let ih_ref = ihfft(&mat_t, Some(n), 1, Norm::Forward)
            .unwrap()
            .t()
            .to_c_contiguous();
        assert_c64_close(&ih0.to_vec(), &ih_ref.to_vec(), 1e-11);

        let h0 = hfft(&ih0, Some(n), 0, Norm::Forward).unwrap();
        let h_ref = hfft(&ih_ref.t().to_c_contiguous(), Some(n), 1, Norm::Forward)
            .unwrap()
            .t()
            .to_c_contiguous();
        assert_f64_close(&h0.to_vec(), &h_ref.to_vec(), 1e-11);
    }

    // 3D `axis = 1` tiled strided path (`[3, 16, 17]` with `n = 16` + crop `n = 11`)
    let a3 = NdArray::from_vec((0..(3 * 16 * 17)).map(|i| (i as f64) * 0.1).collect())
        .reshape(&[3, 16, 17])
        .unwrap();
    let f3_ax1 = fft(&a3, Some(11), 1, Norm::Backward).unwrap();
    let f3_ref = fft(
        &a3.view().transpose(Some(&[0, 2, 1])).unwrap().to_owned(),
        Some(11),
        2,
        Norm::Backward,
    )
    .unwrap()
    .view()
    .transpose(Some(&[0, 2, 1]))
    .unwrap()
    .to_owned();
    assert_c64_close(&f3_ax1.to_vec(), &f3_ref.to_vec(), 1e-10);
}

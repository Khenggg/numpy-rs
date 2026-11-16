mod common;
use common::{assert_dyn_matches_oracle, py_assign_dyn, run_oracle};
use numpy_lib as lib;
use numpy_types::*;

#[test]
fn reductions_statistics_and_sorting_against_numpy() {
    let a = NdArray::from_shape_vec(
        &[3, 4],
        vec![4., 1., -2., 7., 0., 9., 3., -4., 8., 2., 5., 6.],
    )
    .unwrap();
    for (layout, a) in common::non_contiguous_2d_views(&a) {
        for axis in [None, Some(0), Some(1), Some(-1)] {
            for keep in [false, true] {
                let ax = axis.map_or("None".into(), |a| a.to_string());
                let py = format!(
                    "{}\nfor fn in ['sum','prod','min','max','mean','var','std','median','argmin','argmax','all','any','count_nonzero']:\n emit_arr(fn,getattr(np,fn)(a,axis={ax},keepdims={}))\n",
                    py_assign_dyn("a", &DynArray::F64(a.clone())),
                    if keep { "True" } else { "False" }
                );
                let oracle = run_oracle(&py);
                let outputs: Vec<(&str, DynArray)> = vec![
                    ("sum", lib::sum(&a, axis, keep).unwrap().into()),
                    ("prod", lib::prod(&a, axis, keep).unwrap().into()),
                    ("min", lib::min(&a, axis, keep).unwrap().into()),
                    ("max", lib::max(&a, axis, keep).unwrap().into()),
                    ("mean", lib::mean(&a, axis, keep).unwrap().into()),
                    ("var", lib::var(&a, axis, keep, 0).unwrap().into()),
                    ("std", lib::std(&a, axis, keep, 0).unwrap().into()),
                    ("median", lib::median(&a, axis, keep).unwrap().into()),
                    ("argmin", lib::argmin(&a, axis, keep).unwrap().into()),
                    ("argmax", lib::argmax(&a, axis, keep).unwrap().into()),
                    ("all", lib::all(&a, axis, keep).unwrap().into()),
                    ("any", lib::any(&a, axis, keep).unwrap().into()),
                    (
                        "count_nonzero",
                        lib::count_nonzero(&a, axis, keep).unwrap().into(),
                    ),
                ];
                for (tag, actual) in outputs {
                    assert_dyn_matches_oracle(
                        &actual,
                        &oracle.arrays[tag],
                        1e-12,
                        1e-12,
                        true,
                        false,
                        &format!("{layout} {axis:?} {keep} {tag}"),
                    );
                }
            }
            let ax = axis.map_or("None".into(), |a| a.to_string());
            let oracle = run_oracle(&format!(
                "{}\nemit_arr('sort',np.sort(a,axis={ax},kind='stable'))\nemit_arr('argsort',np.argsort(a,axis={ax},kind='stable'))\nemit_arr('cumsum',np.cumsum(a,axis={ax}))\nemit_arr('cumprod',np.cumprod(a,axis={ax}))",
                py_assign_dyn("a", &a.clone().into())
            ));
            for (tag, actual) in [
                ("sort", DynArray::from(lib::sort(&a, axis).unwrap())),
                ("argsort", lib::argsort(&a, axis).unwrap().into()),
                ("cumsum", lib::cumsum(&a, axis).unwrap().into()),
                ("cumprod", lib::cumprod(&a, axis).unwrap().into()),
            ] {
                assert_dyn_matches_oracle(
                    &actual,
                    &oracle.arrays[tag],
                    1e-12,
                    1e-12,
                    true,
                    false,
                    tag,
                );
            }
        }
    }
}

#[test]
fn dynamic_reduction_dtype_promotion_matches_numpy() {
    let inputs: Vec<DynArray> = vec![
        NdArray::from_vec(vec![true, false, true]).into(),
        NdArray::from_vec(vec![120i8, 120, 120]).into(),
        NdArray::from_vec(vec![2u8, 3, 4]).into(),
        NdArray::from_vec(vec![2i16, 3, 4]).into(),
        NdArray::from_vec(vec![2u16, 3, 4]).into(),
        NdArray::from_vec(vec![2i32, 3, 4]).into(),
        NdArray::from_vec(vec![2u32, 3, 4]).into(),
        NdArray::from_vec(vec![2i64, 3, 4]).into(),
        NdArray::from_vec(vec![2u64, 3, 4]).into(),
        NdArray::from_vec(vec![f16::from_f64(2.), f16::from_f64(3.)]).into(),
        NdArray::from_vec(vec![2f32, 3., 4.]).into(),
        NdArray::from_vec(vec![2f64, 3., 4.]).into(),
        NdArray::from_vec(vec![Complex32::new(2., 1.), Complex32::new(3., -1.)]).into(),
        NdArray::from_vec(vec![Complex64::new(2., 1.), Complex64::new(3., -1.)]).into(),
    ];
    for input in inputs {
        let oracle = run_oracle(&format!(
            "{}\nfor fn in ['sum','prod','mean','var','std']:\n emit_arr(fn,getattr(np,fn)(a))",
            py_assign_dyn("a", &input)
        ));
        for (tag, output) in [
            ("sum", lib::dynamic::sum(&input, None, None, false).unwrap()),
            (
                "prod",
                lib::dynamic::prod(&input, None, None, false).unwrap(),
            ),
            ("mean", lib::dynamic::mean(&input, None, false).unwrap()),
            ("var", lib::dynamic::var(&input, None, false, 0).unwrap()),
            ("std", lib::dynamic::std(&input, None, false, 0).unwrap()),
        ] {
            assert_dyn_matches_oracle(
                &output,
                &oracle.arrays[tag],
                1e-3,
                1e-5,
                true,
                false,
                &format!("{} {tag}", input.dtype()),
            );
        }
    }
}

#[test]
fn nanfunctions_quantile_and_setops_against_numpy() {
    let a = NdArray::from_shape_vec(
        &[2, 4],
        vec![0., -0., f64::NAN, 3., 1., f64::NAN, 7., f64::NAN],
    )
    .unwrap();
    let oracle = run_oracle(&format!(
        "{}\nfor fn in ['nansum','nanprod','nanmean','nanvar','nanstd','nanmedian','nanmin','nanmax','nanargmin','nanargmax']:\n emit_arr(fn,getattr(np,fn)(a,axis=1))\nemit_arr('unique',np.unique(a))\nu=np.unique_all(a)\nfor k in ['values','indices','inverse_indices','counts']: emit_arr(k,getattr(u,k))\nfor m in ['linear','lower','higher','midpoint','nearest']:\n emit_arr(m,np.nanquantile(a,[0.,.25,.5,.75,1.],axis=1,method=m))",
        py_assign_dyn("a", &a.clone().into())
    ));
    let outputs: Vec<(&str, DynArray)> = vec![
        ("nansum", lib::nansum(&a, Some(1), false).unwrap().into()),
        ("nanprod", lib::nanprod(&a, Some(1), false).unwrap().into()),
        ("nanmean", lib::nanmean(&a, Some(1), false).unwrap().into()),
        ("nanvar", lib::nanvar(&a, Some(1), false, 0).unwrap().into()),
        ("nanstd", lib::nanstd(&a, Some(1), false, 0).unwrap().into()),
        (
            "nanmedian",
            lib::nanmedian(&a, Some(1), false).unwrap().into(),
        ),
        ("nanmin", lib::nanmin(&a, Some(1), false).unwrap().into()),
        ("nanmax", lib::nanmax(&a, Some(1), false).unwrap().into()),
        (
            "nanargmin",
            lib::nanargmin(&a, Some(1), false).unwrap().into(),
        ),
        (
            "nanargmax",
            lib::nanargmax(&a, Some(1), false).unwrap().into(),
        ),
        ("unique", lib::unique(&a).unwrap().into()),
    ];
    for (tag, actual) in outputs {
        assert_dyn_matches_oracle(&actual, &oracle.arrays[tag], 1e-12, 1e-12, true, false, tag);
    }
    let u = lib::unique_all(&a).unwrap();
    for (tag, actual) in [
        ("values", DynArray::from(u.values)),
        ("indices", u.indices.into()),
        ("inverse_indices", u.inverse.into()),
        ("counts", u.counts.into()),
    ] {
        assert_dyn_matches_oracle(&actual, &oracle.arrays[tag], 0., 0., true, false, tag);
    }
    for (tag, method) in [
        ("linear", lib::QuantileMethod::Linear),
        ("lower", lib::QuantileMethod::Lower),
        ("higher", lib::QuantileMethod::Higher),
        ("midpoint", lib::QuantileMethod::Midpoint),
        ("nearest", lib::QuantileMethod::Nearest),
    ] {
        let output =
            lib::nanquantile(&a, &[0., 0.25, 0.5, 0.75, 1.], Some(1), false, method).unwrap();
        assert_dyn_matches_oracle(
            &output.into(),
            &oracle.arrays[tag],
            1e-12,
            1e-12,
            true,
            false,
            tag,
        );
    }
}

#[test]
fn histograms_and_calculus_against_numpy() {
    let a = NdArray::from_shape_vec(
        &[3, 4],
        vec![4., 1., -2., 7., 0., 9., 3., -4., 8., 2., 5., 6.],
    )
    .unwrap()
    .t();
    let oracle = run_oracle(&format!(
        "{}\nh,e=np.histogram(a,bins=5,range=(-4.,9.),density=True)\nemit_arr('hist',h)\nemit_arr('edges',e)\nfor ax in [0,1]:\n emit_arr(f'diff{{ax}}',np.diff(a,axis=ax))\n emit_arr(f'gradient{{ax}}',np.gradient(a,2.,axis=ax,edge_order=2))\n emit_arr(f'trap{{ax}}',np.trapezoid(a,dx=2.,axis=ax))\nemit_arr('cov',np.cov(a))\nemit_arr('corr',np.corrcoef(a))",
        py_assign_dyn("a", &a.clone().into())
    ));
    let h = lib::histogram(&a, lib::Bins::Count(5), Some((-4., 9.)), None, true).unwrap();
    for (tag, actual) in [
        ("hist", h.counts),
        ("edges", h.edges),
        ("cov", lib::cov(&a, true, 1).unwrap()),
        ("corr", lib::corrcoef(&a, true).unwrap()),
    ] {
        assert_dyn_matches_oracle(
            &actual.into(),
            &oracle.arrays[tag],
            1e-12,
            1e-12,
            true,
            false,
            tag,
        );
    }
    for ax in [0, 1] {
        for (tag, actual) in [
            (format!("diff{ax}"), lib::diff(&a, 1, ax).unwrap()),
            (
                format!("gradient{ax}"),
                lib::gradient(&a, 2., ax, 2).unwrap(),
            ),
            (
                format!("trap{ax}"),
                lib::trapezoid(&a, None, 2., ax).unwrap(),
            ),
        ] {
            assert_dyn_matches_oracle(
                &actual.into(),
                &oracle.arrays[&tag],
                1e-12,
                1e-12,
                true,
                false,
                &tag,
            );
        }
    }
}

#[test]
fn signal_modes_and_special_values_against_numpy() {
    for (n, m) in [(3, 2), (2, 3), (4, 2), (2, 4)] {
        let a = NdArray::from_vec((0..n).map(|i| (i + 1) as f64).collect());
        let b = NdArray::from_vec((0..m).map(|i| (i + 2) as f64).collect());
        let oracle = run_oracle(&format!(
            "{}\n{}\nfor fn in ['convolve','correlate']:\n for mode in ['full','same','valid']:\n  emit_arr(fn+mode,getattr(np,fn)(a,b,mode=mode))",
            py_assign_dyn("a", &a.clone().into()),
            py_assign_dyn("b", &b.clone().into())
        ));
        for (name, mode) in [
            ("full", lib::ConvolveMode::Full),
            ("same", lib::ConvolveMode::Same),
            ("valid", lib::ConvolveMode::Valid),
        ] {
            for (prefix, out) in [
                ("convolve", lib::convolve(&a, &b, mode).unwrap()),
                ("correlate", lib::correlate(&a, &b, mode).unwrap()),
            ] {
                let tag = format!("{prefix}{name}");
                assert_dyn_matches_oracle(
                    &out.into(),
                    &oracle.arrays[&tag],
                    0.,
                    0.,
                    false,
                    false,
                    &tag,
                );
            }
        }
    }
    let a = NdArray::from_vec(vec![0., f64::NAN, 1.]);
    let oracle = run_oracle(&format!(
        "{}\nemit_arr('unwrap',np.unwrap(a))\nemit_arr('digitize',np.digitize(a,[0.,1.,2.],right=True))",
        py_assign_dyn("a", &a.clone().into())
    ));
    assert_dyn_matches_oracle(
        &lib::unwrap(&a, None, 0, std::f64::consts::TAU)
            .unwrap()
            .into(),
        &oracle.arrays["unwrap"],
        0.,
        0.,
        true,
        false,
        "unwrap",
    );
    assert_dyn_matches_oracle(
        &lib::digitize(&a, &NdArray::from_vec(vec![0., 1., 2.]), true)
            .unwrap()
            .into(),
        &oracle.arrays["digitize"],
        0.,
        0.,
        false,
        false,
        "digitize",
    );
}

#[test]
fn polynomial_ops_fit_and_orthogonal_bases_against_numpy() {
    let x_std = NdArray::from_vec(vec![-2., -1., 0., 1., 2.]);
    let y_std = NdArray::from_vec(vec![12., 3., 2., 9., 24.]); // 2 + 3x + 4x^2
    let x_shift = NdArray::from_vec(vec![1000., 1001., 1002., 1003.]);
    let y_shift = NdArray::from_vec(vec![1., 2., 5., 10.]);
    let x_const = NdArray::from_vec(vec![1., 1.]);
    let y_const = NdArray::from_vec(vec![2., 4.]);
    let x_under = NdArray::from_vec(vec![0., 1., 2.]);
    let y_under = NdArray::from_vec(vec![1., 2., 3.]);
    let eval_pts = NdArray::from_vec(vec![-1.5, -0.25, 0.0, 0.75, 1.5]);

    let oracle = run_oracle(&format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        py_assign_dyn("x_std", &x_std.clone().into()),
        py_assign_dyn("y_std", &y_std.clone().into()),
        py_assign_dyn("x_shift", &x_shift.clone().into()),
        py_assign_dyn("y_shift", &y_shift.clone().into()),
        py_assign_dyn("x_const", &x_const.clone().into()),
        py_assign_dyn("y_const", &y_const.clone().into()),
        py_assign_dyn("x_under", &x_under.clone().into()),
        py_assign_dyn("y_under", &y_under.clone().into()),
        py_assign_dyn("eval_pts", &eval_pts.clone().into()),
        r#"
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE
p = Polynomial([1.0, 2.0, 3.0])
q = Polynomial([4.0, -2.0, -3.0])
emit_arr('p_add_q', (p + q).coef)
emit_arr('p_sub_q', (p - q).coef)
emit_arr('p_mul_q', (p * q).coef)
emit_arr('p_const_d1', Polynomial([5.0]).deriv(1).coef)
emit_arr('np_polyder_5', np.polyder([5.0], 1))
emit_arr('fit_std', Polynomial.fit(x_std, y_std, 2).convert().coef)
emit_arr('fit_shift', Polynomial.fit(x_shift, y_shift, 2).convert().coef)
emit_arr('fit_const', Polynomial.fit(x_const, y_const, 1).convert().coef)
emit_arr('fit_under', Polynomial.fit(x_under, y_under, 4).convert().coef)
emit_arr('fit_rcond', Polynomial.fit(x_std, y_std, 2, rcond=0.5).convert().coef)
classes = [('poly', Polynomial), ('cheb', Chebyshev), ('leg', Legendre), ('lag', Laguerre), ('herm', Hermite), ('herme', HermiteE)]
for name, cls in classes:
    s = cls([1.0, 2.0, 3.0, 4.0, -0.5])
    emit_arr(f'{name}_d0', s.deriv(0).coef)
    emit_arr(f'{name}_d1', s.deriv(1).coef)
    emit_arr(f'{name}_d2', s.deriv(2).coef)
    emit_arr(f'{name}_d5', s.deriv(5).coef)
    emit_arr(f'{name}_cd1', cls([5.0]).deriv(1).coef)
    emit_arr(f'{name}_i0', s.integ(0).coef)
    emit_arr(f'{name}_i1', s.integ(1, k=[4.0]).coef)
    emit_arr(f'{name}_i2_1', s.integ(2, k=[4.0]).coef)
    emit_arr(f'{name}_i2_2', s.integ(2, k=[4.0, 5.0]).coef)
    emit_arr(f'{name}_z_i1', cls([0.0]).integ(1, k=[4.0]).coef)
    emit_arr(f'{name}_eval', s(eval_pts))
    emit_arr(f'{name}_roots', np.sort_complex(s.roots()))
"#
    ));

    let p = lib::Polynomial::new(vec![1.0, 2.0, 3.0]);
    let q = lib::Polynomial::new(vec![4.0, -2.0, -3.0]);
    for (tag, res) in [
        ("p_add_q", (&p + &q)),
        ("p_add_q", (p.clone() + q.clone())),
        ("p_sub_q", (&p - &q)),
        ("p_sub_q", (p.clone() - q.clone())),
        ("p_mul_q", (&p * &q)),
        ("p_mul_q", (p * q)),
    ] {
        assert_dyn_matches_oracle(
            &NdArray::from_vec(res.coefficients().to_vec()).into(),
            &oracle.arrays[tag],
            1e-12,
            1e-12,
            false,
            false,
            tag,
        );
    }
    assert_eq!(lib::Polynomial::new(vec![]).coefficients(), &[0.0]);
    assert_dyn_matches_oracle(
        &NdArray::from_vec(lib::Polynomial::new(vec![5.0]).deriv(1).coefficients().to_vec()).into(),
        &oracle.arrays["p_const_d1"],
        0.,
        0.,
        false,
        false,
        "p_const_d1",
    );
    assert_dyn_matches_oracle(
        &NdArray::from_vec(lib::polyder(&[5.0], 1)).into(),
        &oracle.arrays["np_polyder_5"],
        0.,
        0.,
        false,
        false,
        "np_polyder_5",
    );

    for (tag, x, y, deg, rcond, tol) in [
        ("fit_std", &x_std, &y_std, 2usize, None, 1e-12),
        ("fit_shift", &x_shift, &y_shift, 2, None, 1e-5),
        ("fit_const", &x_const, &y_const, 1, None, 1e-12),
        ("fit_rcond", &x_std, &y_std, 2, Some(0.5), 1e-12),
    ] {
        let fitted = lib::Polynomial::fit(x, y, deg, rcond).unwrap();
        assert_dyn_matches_oracle(
            &NdArray::from_vec(fitted.coefficients().to_vec()).into(),
            &oracle.arrays[tag],
            tol,
            tol,
            false,
            false,
            tag,
        );
    }
    let fitted_under = lib::Polynomial::fit(&x_under, &y_under, 4, None).unwrap();
    let mut padded_exp = oracle.arrays["fit_under"].to_f64_vec();
    padded_exp.resize(fitted_under.coefficients().len(), 0.0);
    common::assert_f64_close(
        fitted_under.coefficients(),
        &padded_exp,
        1e-12,
        1e-12,
        false,
        "fit_under",
    );
    common::assert_f64_close(
        &fitted_under.evaluate(&x_under).to_vec(),
        &y_under.to_vec(),
        1e-12,
        1e-12,
        false,
        "fit_under_eval",
    );

    let sort_complex = |arr: NdArray<Complex64>| -> NdArray<Complex64> {
        let mut v = arr.to_vec();
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then_with(|| a.im.total_cmp(&b.im)));
        NdArray::from_vec(v)
    };

    macro_rules! check_series_oracle {
        ($ty:ident, $name:literal) => {{
            let c = vec![1.0, 2.0, 3.0, 4.0, -0.5];
            let s = lib::$ty::new(c);
            let sc = lib::$ty::new(vec![5.0]);
            let sz = lib::$ty::new(vec![0.0]);
            for (suffix, arr) in [
                ("d0", s.deriv(0).coefficients().to_vec()),
                ("d1", s.deriv(1).coefficients().to_vec()),
                ("d2", s.deriv(2).coefficients().to_vec()),
                ("d5", s.deriv(5).coefficients().to_vec()),
                ("cd1", sc.deriv(1).coefficients().to_vec()),
                ("i0", s.integ(0, &[]).unwrap().coefficients().to_vec()),
                ("i1", s.integ(1, &[4.0]).unwrap().coefficients().to_vec()),
                ("i2_1", s.integ(2, &[4.0]).unwrap().coefficients().to_vec()),
                (
                    "i2_2",
                    s.integ(2, &[4.0, 5.0]).unwrap().coefficients().to_vec(),
                ),
                ("z_i1", sz.integ(1, &[4.0]).unwrap().coefficients().to_vec()),
                ("eval", s.evaluate(&eval_pts).to_vec()),
            ] {
                let tag = format!("{}_{suffix}", $name);
                assert_dyn_matches_oracle(
                    &NdArray::from_vec(arr).into(),
                    &oracle.arrays[&tag],
                    1e-11,
                    1e-11,
                    false,
                    false,
                    &tag,
                );
            }
            let r = sort_complex(s.roots().unwrap());
            let rtag = format!("{}_roots", $name);
            assert_dyn_matches_oracle(
                &r.into(),
                &oracle.arrays[&rtag],
                1e-10,
                1e-10,
                false,
                false,
                &rtag,
            );
            assert!(sc.roots().unwrap().is_empty());
        }};
    }
    check_series_oracle!(Polynomial, "poly");
    check_series_oracle!(Chebyshev, "cheb");
    check_series_oracle!(Legendre, "leg");
    check_series_oracle!(Laguerre, "lag");
    check_series_oracle!(Hermite, "herm");
    check_series_oracle!(HermiteE, "herme");
}

#[test]
fn nan_complex_sorting_searching_histograms_interp_and_io_against_numpy() {
    let c_both_nan = NdArray::from_vec(vec![Complex64::new(f64::NAN, f64::NAN)]);
    let c_mixed_nan = NdArray::from_vec(vec![
        Complex64::new(1.0, f64::NAN),
        Complex64::new(f64::NAN, 2.0),
    ]);
    let c32_nan = NdArray::from_vec(vec![Complex32::new(f32::NAN, f32::NAN)]);

    let a_sort = NdArray::from_shape_vec(
        &[3, 5],
        vec![
            9., 1., 5., 3., 7., 2., 8., 4., 6., 0., -3., 11., 2., -1., 4.,
        ],
    )
    .unwrap();
    let ss_base = NdArray::from_vec(vec![-1.0, -0.0, 1.0, 1.0, 4.0, f64::NAN]);
    let ss_q = NdArray::from_vec(vec![-2.0, 0.0, 1.0, 2.5, 5.0, f64::NAN]);
    let k1 = NdArray::from_shape_vec(&[2, 4], vec![1i64, 2, 1, 2, 5, 5, 3, 3]).unwrap();
    let k2 = NdArray::from_shape_vec(&[2, 4], vec![9i64, 3, 3, 1, 4, 2, 4, 2]).unwrap();

    let hx = NdArray::from_vec(vec![0.1, 0.5, 1.2, 1.8, 2.0]);
    let hy = NdArray::from_vec(vec![0.2, 0.9, 1.5, 0.3, 1.9]);
    let hw = NdArray::from_vec(vec![1.0, 2.0, 1.5, 0.5, 3.0]);
    let bc_in = NdArray::from_vec(vec![0i64, 1, 1, 3, 2]);
    let bc_w = NdArray::from_vec(vec![0.5, 1.0, 2.5, 4.0, 0.25]);
    let ix = NdArray::from_vec(vec![-1.0, 0.0, 0.5, 1.0, 2.5, 4.0, f64::NAN]);
    let ixp = NdArray::from_vec(vec![0.0, 1.0, 3.0]);
    let ifp = NdArray::from_vec(vec![10.0, 20.0, 0.0]);

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp_dir =
        std::env::temp_dir().join(format!("numpy-rs-lib-oracle-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let py_npy = tmp_dir.join("from_py.npy");
    let py_npz = tmp_dir.join("from_py.npz");
    let rs_npy = tmp_dir.join("from_rs.npy");
    let rs_npz = tmp_dir.join("from_rs.npz");
    let py_txt = tmp_dir.join("from_py.csv");

    let npy_arr = NdArray::from_shape_vec(&[2, 3], vec![1.5, -0.0, 3.25, 4.0, f64::INFINITY, -2.0])
        .unwrap()
        .t();
    lib::save(&rs_npy, &npy_arr.clone().into()).unwrap();
    let npy_i64 = NdArray::from_vec(vec![i64::MIN, -1, 0, i64::MAX]);
    lib::savez_compressed(
        &rs_npz,
        &[("f", &npy_arr.clone().into()), ("i", &npy_i64.clone().into())],
    )
    .unwrap();

    let py_npy_s = py_npy.to_string_lossy().replace('\\', "/");
    let py_npz_s = py_npz.to_string_lossy().replace('\\', "/");
    let rs_npy_s = rs_npy.to_string_lossy().replace('\\', "/");
    let rs_npz_s = rs_npz.to_string_lossy().replace('\\', "/");
    let py_txt_s = py_txt.to_string_lossy().replace('\\', "/");

    let assigns = [
        py_assign_dyn("c_both", &c_both_nan.clone().into()),
        py_assign_dyn("c_mix", &c_mixed_nan.clone().into()),
        py_assign_dyn("c32", &c32_nan.clone().into()),
        py_assign_dyn("a_sort", &a_sort.clone().into()),
        py_assign_dyn("ss_base", &ss_base.clone().into()),
        py_assign_dyn("ss_q", &ss_q.clone().into()),
        py_assign_dyn("k1", &k1.clone().into()),
        py_assign_dyn("k2", &k2.clone().into()),
        py_assign_dyn("hx", &hx.clone().into()),
        py_assign_dyn("hy", &hy.clone().into()),
        py_assign_dyn("hw", &hw.clone().into()),
        py_assign_dyn("bc_in", &bc_in.clone().into()),
        py_assign_dyn("bc_w", &bc_w.clone().into()),
        py_assign_dyn("ix", &ix.clone().into()),
        py_assign_dyn("ixp", &ixp.clone().into()),
        py_assign_dyn("ifp", &ifp.clone().into()),
    ]
    .join("\n");

    let oracle = run_oracle(&format!(
        "{assigns}\n\
         emit_arr('c_both_min', np.nanmin(c_both))\n\
         emit_arr('c_both_max', np.nanmax(c_both))\n\
         emit_arr('c_mix_min', np.nanmin(c_mix))\n\
         emit_arr('c_mix_max', np.nanmax(c_mix))\n\
         emit_arr('c32_min', np.nanmin(c32))\n\
         emit_arr('sort_none', np.sort(a_sort, axis=None))\n\
         emit_arr('sort_ax0', np.sort(a_sort, axis=0))\n\
         emit_arr('sort_ax1', np.sort(a_sort, axis=1))\n\
         emit_arr('sort_t_none', np.sort(a_sort.T, axis=None))\n\
         emit_arr('sort_t_ax0', np.sort(a_sort.T, axis=0))\n\
         emit_arr('sort_t_ax1', np.sort(a_sort.T, axis=1))\n\
         emit_arr('ss_left', np.searchsorted(ss_base, ss_q, side='left'))\n\
         emit_arr('ss_right', np.searchsorted(ss_base, ss_q, side='right'))\n\
         emit_arr('lex_ax1', np.lexsort((k1, k2), axis=1))\n\
         emit_arr('lex_ax0', np.lexsort((k1, k2), axis=0))\n\
         emit_arr('topk_ax1', np.sort(a_sort, axis=1)[:, ::-1][:, :3])\n\
         h2, ex, ey = np.histogram2d(hx, hy, bins=[2, 3], weights=hw, density=True)\n\
         emit_arr('h2_c', h2)\n\
         emit_arr('h2_ex', ex)\n\
         emit_arr('h2_ey', ey)\n\
         sample = np.column_stack([hx, hy])\n\
         hdd, e_dd = np.histogramdd(sample, bins=[2, 2], density=False)\n\
         emit_arr('hdd_c', hdd)\n\
         emit_arr('bc', np.bincount(bc_in, weights=bc_w, minlength=7))\n\
         emit_arr('interp_def', np.interp(ix, ixp, ifp))\n\
         emit_arr('interp_lr', np.interp(ix, ixp, ifp, left=-5.0, right=99.0))\n\
         arr_f = np.asfortranarray(np.array([[1.5, -0.0, 3.25], [4.0, np.inf, -2.0]], dtype=np.float64).T)\n\
         np.save('{py_npy_s}', arr_f)\n\
         np.savez('{py_npz_s}', a=arr_f, b=np.array([True, False, True]))\n\
         np.savetxt('{py_txt_s}', np.array([[1.0, 2.5], [3.5, 4.0]]), delimiter=',')\n\
         emit_arr('rs_npy_loaded', np.load('{rs_npy_s}'))\n\
         z_rs = np.load('{rs_npz_s}')\n\
         emit_arr('rs_npz_f', z_rs['f'])\n\
         emit_arr('rs_npz_i', z_rs['i'])\n",
    ));

    // All-NaN complex nanmin/nanmax preserve NaN+NaNj / first NaN element
    for (tag, res) in [
        ("c_both_min", DynArray::from(lib::nanmin(&c_both_nan, None, false).unwrap())),
        ("c_both_max", lib::nanmax(&c_both_nan, None, false).unwrap().into()),
        ("c_mix_min", lib::nanmin(&c_mixed_nan, None, false).unwrap().into()),
        ("c_mix_max", lib::nanmax(&c_mixed_nan, None, false).unwrap().into()),
        ("c32_min", lib::nanmin(&c32_nan, None, false).unwrap().into()),
    ] {
        assert_dyn_matches_oracle(&res, &oracle.arrays[tag], 0., 0., true, false, tag);
    }

    // partition & argpartition invariants across axes & layouts
    for (layout, view) in common::non_contiguous_2d_views(&a_sort) {
        for (axis, suffix, kth) in [
            (None, "none", -4isize),
            (Some(0), "ax0", 1isize),
            (Some(1), "ax1", -2isize),
        ] {
            let p = lib::partition(&view, kth, axis).unwrap();
            let ap = lib::argpartition(&view, kth, axis).unwrap();
            let sorted_actual = lib::sort(&p, axis).unwrap();
            let sorted_expected = lib::sort(&view, axis).unwrap();
            assert_eq!(
                sorted_actual.shape(),
                sorted_expected.shape(),
                "partition shape {layout} {axis:?}"
            );
            assert_eq!(
                sorted_actual.to_vec(),
                sorted_expected.to_vec(),
                "partition multiset {layout} {axis:?}"
            );
            if matches!(layout, "c_contiguous" | "f_contiguous" | "transposed") {
                let prefix = if layout == "transposed" { "sort_t" } else { "sort" };
                let tag_sorted = format!("{prefix}_{suffix}");
                assert_dyn_matches_oracle(
                    &sorted_actual.into(),
                    &oracle.arrays[&tag_sorted],
                    0.,
                    0.,
                    false,
                    false,
                    &format!("partition_oracle {layout} {axis:?}"),
                );
            }
            let width = match axis {
                None => view.len(),
                Some(ax) => view.shape()[ax as usize],
            };
            let k_idx = ((kth % width as isize) + width as isize) as usize % width;
            // Verify pivot invariant along 1D flattened or axis=1 rows
            if axis == Some(1) {
                let pv = p.to_vec();
                let apv = ap.to_vec();
                let vv = view.to_vec();
                for r in 0..view.shape()[0] {
                    let row = &pv[r * width..(r + 1) * width];
                    let pivot = row[k_idx];
                    assert!(row[..k_idx].iter().all(|&v| v <= pivot));
                    assert!(row[k_idx + 1..].iter().all(|&v| v >= pivot));
                    let idx_row = &apv[r * width..(r + 1) * width];
                    let gathered: Vec<f64> =
                        idx_row.iter().map(|&idx| vv[r * width + idx as usize]).collect();
                    assert!(gathered[..k_idx].iter().all(|&v| v <= gathered[k_idx]));
                    assert!(gathered[k_idx + 1..].iter().all(|&v| v >= gathered[k_idx]));
                }
            }
        }
    }

    // searchsorted, lexsort, top_k
    assert_dyn_matches_oracle(
        &lib::searchsorted(&ss_base, &ss_q, lib::SearchSide::Left).unwrap().into(),
        &oracle.arrays["ss_left"],
        0.,
        0.,
        false,
        false,
        "ss_left",
    );
    assert_dyn_matches_oracle(
        &lib::searchsorted(&ss_base, &ss_q, lib::SearchSide::Right).unwrap().into(),
        &oracle.arrays["ss_right"],
        0.,
        0.,
        false,
        false,
        "ss_right",
    );
    assert_dyn_matches_oracle(
        &lib::lexsort(&[&k1, &k2], 1).unwrap().into(),
        &oracle.arrays["lex_ax1"],
        0.,
        0.,
        false,
        false,
        "lex_ax1",
    );
    assert_dyn_matches_oracle(
        &lib::lexsort(&[&k1, &k2], 0).unwrap().into(),
        &oracle.arrays["lex_ax0"],
        0.,
        0.,
        false,
        false,
        "lex_ax0",
    );
    assert_dyn_matches_oracle(
        &lib::top_k(&a_sort, 3, 1).unwrap().into(),
        &oracle.arrays["topk_ax1"],
        0.,
        0.,
        false,
        false,
        "topk_ax1",
    );

    // histogram2d, histogramdd, bincount, interp, complex histogram rejection
    let h2 = lib::histogram2d(&hx, &hy, [lib::Bins::Count(2), lib::Bins::Count(3)], Some(&hw), true)
        .unwrap();
    assert_dyn_matches_oracle(&h2.counts.into(), &oracle.arrays["h2_c"], 1e-12, 1e-12, false, false, "h2_c");
    assert_dyn_matches_oracle(&h2.edges[0].clone().into(), &oracle.arrays["h2_ex"], 1e-12, 1e-12, false, false, "h2_ex");
    assert_dyn_matches_oracle(&h2.edges[1].clone().into(), &oracle.arrays["h2_ey"], 1e-12, 1e-12, false, false, "h2_ey");

    let mut sample_data = Vec::with_capacity(hx.len() * 2);
    for (xv, yv) in hx.iter().zip(hy.iter()) {
        sample_data.push(xv);
        sample_data.push(yv);
    }
    let sample = NdArray::from_shape_vec(&[hx.len(), 2], sample_data).unwrap();
    let hdd = lib::histogramdd(&sample, &[lib::Bins::Count(2), lib::Bins::Count(2)], None, false).unwrap();
    assert_dyn_matches_oracle(&hdd.counts.into(), &oracle.arrays["hdd_c"], 1e-12, 1e-12, false, false, "hdd_c");

    assert!(lib::histogram(&c_mixed_nan, lib::Bins::Count(2), None, None, false).is_err());
    assert!(lib::histogram_bin_edges(&c_mixed_nan, lib::Bins::Count(2), None).is_err());

    assert_dyn_matches_oracle(
        &lib::bincount(&bc_in, Some(&bc_w), 7).unwrap().into(),
        &oracle.arrays["bc"],
        1e-12,
        1e-12,
        false,
        false,
        "bincount",
    );
    assert_dyn_matches_oracle(
        &lib::interp(&ix, &ixp, &ifp, None, None).unwrap().into(),
        &oracle.arrays["interp_def"],
        1e-12,
        1e-12,
        true,
        false,
        "interp_def",
    );
    assert_dyn_matches_oracle(
        &lib::interp(&ix, &ixp, &ifp, Some(-5.0), Some(99.0)).unwrap().into(),
        &oracle.arrays["interp_lr"],
        1e-12,
        1e-12,
        true,
        false,
        "interp_lr",
    );

    // Cross-language NPY / NPZ / text I/O roundtrips
    let py_loaded_npy = lib::load(&py_npy).unwrap();
    assert_dyn_matches_oracle(&py_loaded_npy, &oracle.arrays["rs_npy_loaded"], 0., 0., false, true, "py_npy");
    let py_loaded_npz = lib::loadz(&py_npz).unwrap();
    assert_dyn_matches_oracle(&py_loaded_npz["a"], &oracle.arrays["rs_npz_f"], 0., 0., false, true, "py_npz_a");
    assert_dyn_matches_oracle(
        &DynArray::from(npy_arr),
        &oracle.arrays["rs_npy_loaded"],
        0.,
        0.,
        false,
        true,
        "rs_npy_loaded",
    );
    assert_dyn_matches_oracle(
        &DynArray::from(npy_i64),
        &oracle.arrays["rs_npz_i"],
        0.,
        0.,
        false,
        false,
        "rs_npz_i",
    );
    let txt_loaded = lib::loadtxt(&py_txt, Some(','), 0).unwrap();
    assert_eq!(txt_loaded.shape(), &[2, 2]);
    assert_eq!(txt_loaded.to_vec(), vec![1.0, 2.5, 3.5, 4.0]);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn round2_statistics_scalar_axis_percentile_bounds_and_empty_nanquantile_against_numpy() {
    let s = NdArray::from_scalar(3.0f64);
    let empty03 = NdArray::<f64>::zeros(&[0, 3]).unwrap();
    let empty0 = NdArray::<f64>::zeros(&[0]).unwrap();
    let empty20 = NdArray::<f64>::zeros(&[2, 0]).unwrap();
    let non_empty = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let oracle = run_oracle(
        r#"
s = np.array(3.0)
for fn in ['nanmean', 'nanvar', 'nanstd']:
    for ax in [0, -1]:
        for kd in [False, True]:
            emit_arr(f'{fn}_{ax}_{kd}', getattr(np, fn)(s, axis=ax, keepdims=kd))
for fn in ['mean', 'var', 'std', 'median', 'nanmedian']:
    try:
        getattr(np, fn)(s, axis=0)
    except Exception as e:
        emit_err(f'{fn}_0d_ax0', e)

try:
    np.percentile([1.0, 2.0], [100.0 + 1e-14])
except Exception as e:
    emit_err('pct_hi', e)
try:
    np.nanpercentile([1.0, 2.0], [-1e-14])
except Exception as e:
    emit_err('nanpct_lo', e)

e03 = np.zeros((0, 3))
e0 = np.zeros((0,))
e20 = np.zeros((2, 0))
ne = np.array([[1.0, 2.0], [3.0, 4.0]])
emit_arr('nq_03_empty_ax0', np.nanquantile(e03, [], axis=0))
emit_arr('nq_03_empty_ax0_kd', np.nanquantile(e03, [], axis=0, keepdims=True))
emit_arr('nq_03_q2_ax0', np.nanquantile(e03, [0.2, 0.8], axis=0))
emit_arr('nq_0_empty_none', np.nanquantile(e0, [], axis=None))
emit_arr('nq_20_half_ax0', np.nanquantile(e20, [0.5], axis=0))
emit_arr('nq_ne_empty_ax0', np.nanquantile(ne, [], axis=0))
"#,
    );

    for (ax, ax_tag) in [(0isize, "0"), (-1isize, "-1")] {
        for (kd, kd_tag) in [(false, "False"), (true, "True")] {
            assert_dyn_matches_oracle(
                &lib::nanmean(&s, Some(ax), kd).unwrap().into(),
                &oracle.arrays[&format!("nanmean_{ax_tag}_{kd_tag}")],
                0.,
                0.,
                false,
                false,
                "nanmean_0d",
            );
            assert_dyn_matches_oracle(
                &lib::nanvar(&s, Some(ax), kd, 0).unwrap().into(),
                &oracle.arrays[&format!("nanvar_{ax_tag}_{kd_tag}")],
                0.,
                0.,
                false,
                false,
                "nanvar_0d",
            );
            assert_dyn_matches_oracle(
                &lib::nanstd(&s, Some(ax), kd, 0).unwrap().into(),
                &oracle.arrays[&format!("nanstd_{ax_tag}_{kd_tag}")],
                0.,
                0.,
                false,
                false,
                "nanstd_0d",
            );
        }
    }
    assert!(lib::nanmean(&s, Some(1), false).is_err());
    assert!(lib::nanvar(&s, Some(1), false, 0).is_err());
    assert!(lib::nanstd(&s, Some(-2), false, 0).is_err());

    assert!(oracle.errors.contains_key("mean_0d_ax0"));
    assert!(oracle.errors.contains_key("var_0d_ax0"));
    assert!(oracle.errors.contains_key("std_0d_ax0"));
    assert!(oracle.errors.contains_key("median_0d_ax0"));
    assert!(oracle.errors.contains_key("nanmedian_0d_ax0"));
    assert!(lib::mean(&s, Some(0), false).is_err());
    assert!(lib::mean(&s, Some(-1), false).is_err());
    assert!(lib::var(&s, Some(0), false, 0).is_err());
    assert!(lib::std(&s, Some(0), false, 0).is_err());
    assert!(lib::median(&s, Some(0), false).is_err());
    assert!(lib::nanmedian(&s, Some(0), false).is_err());
    assert!(lib::quantile(&s, &[0.5], Some(0), false, lib::QuantileMethod::Linear).is_err());
    assert!(lib::nanquantile(&s, &[0.5], Some(0), false, lib::QuantileMethod::Linear).is_err());

    let p_in = NdArray::from_vec(vec![1.0, 2.0]);
    assert!(oracle.errors.contains_key("pct_hi"));
    assert!(oracle.errors.contains_key("nanpct_lo"));
    assert!(lib::percentile(&p_in, &[100.0 + 1e-14], None, false, lib::QuantileMethod::Linear).is_err());
    assert!(lib::nanpercentile(&p_in, &[-1e-14], None, false, lib::QuantileMethod::Linear).is_err());

    assert_dyn_matches_oracle(
        &lib::nanquantile(&empty03, &[], Some(0), false, lib::QuantileMethod::Linear)
            .unwrap()
            .into(),
        &oracle.arrays["nq_03_empty_ax0"],
        0.,
        0.,
        true,
        false,
        "nq_03_empty_ax0",
    );
    assert_dyn_matches_oracle(
        &lib::nanquantile(&empty03, &[], Some(0), true, lib::QuantileMethod::Linear)
            .unwrap()
            .into(),
        &oracle.arrays["nq_03_empty_ax0_kd"],
        0.,
        0.,
        true,
        false,
        "nq_03_empty_ax0_kd",
    );
    assert_dyn_matches_oracle(
        &lib::nanquantile(&empty03, &[0.2, 0.8], Some(0), false, lib::QuantileMethod::Linear)
            .unwrap()
            .into(),
        &oracle.arrays["nq_03_q2_ax0"],
        0.,
        0.,
        true,
        false,
        "nq_03_q2_ax0",
    );
    assert_dyn_matches_oracle(
        &lib::nanquantile(&empty0, &[], None, false, lib::QuantileMethod::Linear)
            .unwrap()
            .into(),
        &oracle.arrays["nq_0_empty_none"],
        0.,
        0.,
        true,
        false,
        "nq_0_empty_none",
    );
    assert_dyn_matches_oracle(
        &lib::nanquantile(&empty20, &[0.5], Some(0), false, lib::QuantileMethod::Linear)
            .unwrap()
            .into(),
        &oracle.arrays["nq_20_half_ax0"],
        0.,
        0.,
        true,
        false,
        "nq_20_half_ax0",
    );
    assert_dyn_matches_oracle(
        &lib::nanquantile(&non_empty, &[], Some(0), false, lib::QuantileMethod::Linear)
            .unwrap()
            .into(),
        &oracle.arrays["nq_ne_empty_ax0"],
        0.,
        0.,
        true,
        false,
        "nq_ne_empty_ax0",
    );
    assert!(lib::nanquantile(&empty03, &[1.5], Some(0), false, lib::QuantileMethod::Linear).is_err());
}

#[test]
fn p1_numpy_lib_npy_native_descr_savetxt_polynomial_and_histogram_dtypes() {
    use std::io::Cursor;

    // FIX(F-A3-03): native NPY descr 'i4', 'f8' without endian marker
    let header = "{'descr': 'i4', 'fortran_order': False, 'shape': (2,), }\n";
    let pad = 64 - ((10 + header.len()) % 64);
    let mut raw = b"\x93NUMPY\x01\x00".to_vec();
    raw.extend_from_slice(&((header.len() + pad) as u16).to_le_bytes());
    raw.extend_from_slice(header.as_bytes());
    raw.resize(raw.len() + pad - 1, b' ');
    raw.push(b'\n');
    raw.extend_from_slice(&16843009i32.to_le_bytes());
    raw.extend_from_slice(&16843009i32.to_le_bytes());
    let decoded = lib::read_npy(&mut Cursor::new(&raw)).unwrap();
    assert_eq!(decoded.dtype(), DType::I32);
    if let DynArray::I32(arr) = decoded {
        assert_eq!(arr.to_vec(), vec![16843009, 16843009]);
    } else {
        panic!("expected I32");
    }

    // FIX(F-A3-08): savetxt uses %.18e and lowercase nan/inf/-inf matching np.savetxt
    let tmp = std::env::temp_dir().join("numpy_rs_savetxt_p1.txt");
    let mat = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.5, f64::NAN, f64::NEG_INFINITY]).unwrap();
    lib::savetxt(&tmp, &mat, " ").unwrap();
    let txt = std::fs::read_to_string(&tmp).unwrap().replace("\r\n", "\n");
    let _ = std::fs::remove_file(&tmp);

    let oracle = run_oracle(
        r#"
import io
b = io.StringIO()
np.savetxt(b, np.array([[1.0, 2.5], [np.nan, -np.inf]]))
emit_val("savetxt_hex", b.getvalue().encode("utf-8").hex())
q1, r1 = np.polydiv([0., 0., 1.], [1., 2.])
emit_arr("pd_q1", q1)
emit_arr("pd_r1", r1)
q2, r2 = np.polydiv([1., 2.], [0., 1.])
emit_arr("pd_q2", q2)
emit_arr("pd_r2", r2)
emit_arr("pi_extra", np.polyint([1., 2., 3.], m=2, k=[1., 2., 3.]))
emit_arr("pm_empty", np.polymul([], [1., 2.]))
emit_arr("pm_both_empty", np.polymul([], []))
hc0, he0 = np.histogram([1.0], bins=[0.0])
emit_arr("h_single_c", hc0)
emit_arr("h_single_e", he0)
hc_i, _ = np.histogram(np.array([1.0, 2.0, 2.0]), bins=2)
emit_arr("h_i64", hc_i)
hc_f32, _ = np.histogram(np.array([1.0, 2.0, 2.0]), bins=2, weights=np.array([1.0, 2.0, 3.0], dtype=np.float32))
emit_arr("h_w_f32", hc_f32)
bc_i = np.bincount(np.array([0, 1, 1], dtype=np.int64))
emit_arr("bc_i64", bc_i)
bc_w = np.bincount(np.array([0, 1, 1], dtype=np.int64), weights=np.array([1, 2, 3], dtype=np.int32))
emit_arr("bc_w_f64", bc_w)
"#,
    );
    let got_hex: String = txt.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got_hex, oracle.val("savetxt_hex"));

    // FIX(F-A3-04/05/09): polynomial parity
    let (q1, r1) = lib::polydiv(&[0., 0., 1.], &[1., 2.]).unwrap();
    assert_dyn_matches_oracle(&NdArray::from_vec(q1).into(), oracle.arr("pd_q1"), 0., 0., false, false, "pd_q1");
    assert_dyn_matches_oracle(&NdArray::from_vec(r1).into(), oracle.arr("pd_r1"), 0., 0., false, false, "pd_r1");
    let (q2, r2) = lib::polydiv(&[1., 2.], &[0., 1.]).unwrap();
    assert_dyn_matches_oracle(&NdArray::from_vec(q2).into(), oracle.arr("pd_q2"), 0., 0., true, false, "pd_q2");
    assert_dyn_matches_oracle(&NdArray::from_vec(r2).into(), oracle.arr("pd_r2"), 0., 0., true, false, "pd_r2");

    let pi = lib::polyint(&[1., 2., 3.], 2, &[1., 2., 3.]).unwrap();
    assert_dyn_matches_oracle(&NdArray::from_vec(pi).into(), oracle.arr("pi_extra"), 1e-12, 1e-12, false, false, "pi_extra");
    assert!(lib::polyint_with_constants(&[1., 2.], 1, Some(&[])).is_err());

    let pm1 = lib::polymul(&[], &[1., 2.]);
    assert_dyn_matches_oracle(&NdArray::from_vec(pm1).into(), oracle.arr("pm_empty"), 0., 0., false, false, "pm_empty");
    let pm0 = lib::polymul(&[], &[]);
    assert_dyn_matches_oracle(&NdArray::from_vec(pm0).into(), oracle.arr("pm_both_empty"), 0., 0., false, false, "pm_both_empty");

    // FIX(F-A3-02/06): histogram_dyn & bincount_dyn dtypes + single-edge bins
    let a1 = DynArray::F64(NdArray::from_vec(vec![1.0]));
    let hs = lib::histogram_dyn(&a1, lib::Bins::Edges(&[0.0]), None, None, false).unwrap();
    assert_dyn_matches_oracle(&hs.counts, oracle.arr("h_single_c"), 0., 0., false, false, "h_single_c");
    assert_dyn_matches_oracle(&hs.edges.into(), oracle.arr("h_single_e"), 0., 0., false, false, "h_single_e");

    let a3 = DynArray::F64(NdArray::from_vec(vec![1.0, 2.0, 2.0]));
    let hi = lib::histogram_dyn(&a3, lib::Bins::Count(2), None, None, false).unwrap();
    assert_dyn_matches_oracle(&hi.counts, oracle.arr("h_i64"), 0., 0., false, false, "h_i64");
    let w_f32 = DynArray::F32(NdArray::from_vec(vec![1.0, 2.0, 3.0]));
    let hw = lib::histogram_dyn(&a3, lib::Bins::Count(2), None, Some(&w_f32), false).unwrap();
    assert_dyn_matches_oracle(&hw.counts, oracle.arr("h_w_f32"), 1e-6, 1e-6, false, false, "h_w_f32");

    let bi = NdArray::from_vec(vec![0i64, 1, 1]);
    assert_dyn_matches_oracle(&lib::bincount_dyn(&bi, None, 0).unwrap(), oracle.arr("bc_i64"), 0., 0., false, false, "bc_i64");
    let bw = DynArray::I32(NdArray::from_vec(vec![1i32, 2, 3]));
    assert_dyn_matches_oracle(&lib::bincount_dyn(&bi, Some(&bw), 0).unwrap(), oracle.arr("bc_w_f64"), 0., 0., false, false, "bc_w_f64");
}

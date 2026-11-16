use numpy_lib::*;
use std::io::Cursor;
#[test]
fn npz_and_text_interchange_preserve_values_and_validate_before_writing() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("numpy-rs-io-{}-{unique}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let a: DynArray = NdArray::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.])
        .unwrap()
        .t()
        .into();
    let b: DynArray = NdArray::from_vec(vec![i64::MAX, i64::MIN]).into();
    for compressed in [false, true] {
        let path = dir.join(if compressed {
            "compressed.npz"
        } else {
            "stored.npz"
        });
        if compressed {
            savez_compressed(&path, &[("a", &a), ("b", &b)]).unwrap();
        } else {
            savez(&path, &[("a", &a), ("b", &b)]).unwrap();
        }
        let loaded = loadz(&path).unwrap();
        assert_eq!(loaded["a"], a);
        assert_eq!(loaded["b"], b);
        let before = std::fs::read(&path).unwrap();
        assert!(savez(&path, &[("a", &a), ("a", &b)]).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_file(path).unwrap();
    }
    let path = dir.join("values.csv");
    let values = NdArray::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    savetxt(&path, &values, ",").unwrap();
    assert_eq!(loadtxt(&path, Some(','), 0).unwrap(), values);
    std::fs::write(&path, "7\n").unwrap();
    assert_eq!(loadtxt(&path, None, 0).unwrap().shape(), &[]);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
}
#[test]
fn npy_roundtrips_all_types_layouts_and_bits() {
    let arrays: Vec<DynArray> = vec![
        NdArray::from_vec(vec![true, false]).into(),
        NdArray::from_vec(vec![i8::MIN, i8::MAX]).into(),
        NdArray::from_vec(vec![0u8, 255]).into(),
        NdArray::from_vec(vec![i16::MIN, i16::MAX]).into(),
        NdArray::from_vec(vec![0u16, u16::MAX]).into(),
        NdArray::from_vec(vec![i32::MIN, i32::MAX]).into(),
        NdArray::from_vec(vec![0u32, u32::MAX]).into(),
        NdArray::from_vec(vec![i64::MIN, i64::MAX]).into(),
        NdArray::from_vec(vec![0u64, u64::MAX]).into(),
        NdArray::from_vec(vec![f16::from_bits(0x8000), f16::INFINITY]).into(),
        NdArray::from_vec(vec![-0f32, f32::INFINITY]).into(),
        NdArray::from_vec(vec![-0f64, f64::INFINITY]).into(),
        NdArray::from_vec(vec![Complex32::new(-0., 1.), Complex32::new(2., 3.)]).into(),
        NdArray::from_vec(vec![Complex64::new(-0., 1.), Complex64::new(2., 3.)]).into(),
        NdArray::<f64>::zeros(&[2, 0, 3]).unwrap().into(),
        NdArray::from_scalar(7i64).into(),
        NdArray::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .t()
            .into(),
    ];
    for a in arrays {
        let mut first = Vec::new();
        write_npy(&mut first, &a).unwrap();
        let b = read_npy(&mut Cursor::new(&first)).unwrap();
        let mut second = Vec::new();
        write_npy(&mut second, &b).unwrap();
        assert_eq!(first, second);
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
    }
}
#[test]
fn malformed_npy_is_rejected_without_panic() {
    for bytes in [
        vec![],
        b"not numpy".to_vec(),
        b"\x93NUMPY\x02\x00\xff\xff\xff\xff".to_vec(),
    ] {
        assert!(read_npy(&mut Cursor::new(bytes)).is_err());
    }
    let mut bytes = Vec::new();
    write_npy(&mut bytes, &NdArray::from_vec(vec![1f64, 2.]).into()).unwrap();
    bytes.pop();
    assert!(read_npy(&mut Cursor::new(bytes)).is_err());
    for header in [
        "{'descr':'<f8','fortran_order':Falsex,'shape':(1,),}",
        "{'descr':'<f8','fortran_order':False,'shape':(1),}",
        "{'descr':'<f8','fortran_order':False,'shape':(1,,2),}",
        "{'descr':'<f8','fortran_order':False,'shape':(1,), 'shape':(2,),}",
    ] {
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1f64.to_le_bytes());
        assert!(read_npy(&mut Cursor::new(bytes)).is_err(), "{header}");
    }
}
#[test]
fn legacy_polynomial_arithmetic_and_fitting() {
    assert_eq!(poly(&[1., 2., 3.]), vec![1., -6., 11., -6.]);
    assert_eq!(polyadd(&[1., 2.], &[3., 4., 5.]), vec![3., 5., 7.]);
    assert_eq!(polyadd(&[1., 0.], &[-1., 2.]), vec![0., 2.]);
    assert_eq!(polysub(&[1., 0.], &[1., 2.]), vec![0., -2.]);
    assert_eq!(polymul(&[1., -1.], &[1., 1.]), vec![1., 0., -1.]);
    assert_eq!(
        polydiv(&[1., 0., -1.], &[1., -1.]).unwrap(),
        (vec![1., 1.], vec![0.])
    );
    assert!(polydiv(&[1.], &[0.]).is_err());
    assert_eq!(polyder(&[1., -6., 11., -6.], 1), vec![3., -12., 11.]);
    assert_eq!(polyint(&[2., 3.], 1, &[4.]).unwrap(), vec![1., 3., 4.]);
    let x = NdArray::from_vec(vec![-2., -1., 0., 1., 2.]);
    let y = polyval(&[2., 3., 4.], &x);
    let fit = polyfit(&x, &y, 2, None).unwrap();
    for (a, b) in fit.iter().zip([2., 3., 4.]) {
        assert!((a - b).abs() < 1e-10);
    }
    let r = roots(&[1., -3., 2.]).unwrap();
    let mut real: Vec<f64> = r
        .iter()
        .map(|z| {
            assert!(z.im.abs() < 1e-12);
            z.re
        })
        .collect();
    real.sort_by(f64::total_cmp);
    assert!((real[0] - 1.).abs() < 1e-12 && (real[1] - 2.).abs() < 1e-12);
}
#[test]
fn orthogonal_bases_convert_to_power_series() {
    for basis in [
        Basis::Power,
        Basis::Chebyshev,
        Basis::Legendre,
        Basis::Laguerre,
        Basis::Hermite,
        Basis::HermiteE,
    ] {
        let series = PolynomialSeries::new(vec![1., 2., 3., 4., 5.], basis);
        let power = series.to_polynomial();
        for x in [-2., -0.5, 0., 0.5, 2.] {
            assert!(
                (series.eval(x) - power.eval(x)).abs() < 1e-10,
                "{basis:?} {x}"
            );
        }
    }
    assert_eq!(
        Polynomial::new(vec![1., 2., 3.]).deriv(1).coefficients(),
        &[2., 6.]
    );
    assert_eq!(Polynomial::new(vec![]).coefficients(), &[0.0]);
    assert_eq!(polyder(&[5.0], 1), Vec::<f64>::new());
    assert_eq!(
        Polynomial::new(vec![5.0]).deriv(1).coefficients(),
        &[0.0]
    );
    assert_eq!(
        Polynomial::new(vec![0.0]).integ(1, &[4.0]).unwrap().coefficients(),
        &[4.0]
    );
    assert!(Polynomial::new(vec![1.0, 2.0]).integ(1, &[1.0, 2.0]).is_err());

    let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
    let q = Polynomial::new(vec![4.0, -2.0, -3.0]);
    assert_eq!((&p + &q).coefficients(), &[5.0]);
    assert_eq!((p.clone() + q.clone()).coefficients(), &[5.0]);
    assert_eq!((&p - &q).coefficients(), &[-3.0, 4.0, 6.0]);
    assert_eq!((p.clone() - q.clone()).coefficients(), &[-3.0, 4.0, 6.0]);
    assert_eq!(
        (&p * &q).coefficients(),
        &[4.0, 6.0, 5.0, -12.0, -9.0]
    );
    assert_eq!(
        (p * Polynomial::new(vec![0.0])).coefficients(),
        &[0.0]
    );

    let fx = NdArray::from_vec(vec![1000.0, 1001.0, 1002.0, 1003.0]);
    let fy = fx.map(|v| 2.0 + 3.0 * v + 4.0 * v * v);
    let fitted = Polynomial::fit(&fx, &fy, 2, None).unwrap();
    let eval_err = fitted
        .evaluate(&fx)
        .iter()
        .zip(fy.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(eval_err < 1e-6, "eval_err={eval_err}");

    macro_rules! check_ortho_const {
        ($ty:ident) => {
            assert_eq!($ty::new(vec![]).coefficients(), &[0.0]);
            assert_eq!($ty::new(vec![5.0]).deriv(1).coefficients(), &[0.0]);
            assert!($ty::new(vec![5.0]).roots().unwrap().is_empty());
        };
    }
    check_ortho_const!(Chebyshev);
    check_ortho_const!(Legendre);
    check_ortho_const!(Laguerre);
    check_ortho_const!(Hermite);
    check_ortho_const!(HermiteE);
}

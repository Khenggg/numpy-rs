use numpy_lib::*;

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() < 1e-12 || (a.is_nan() && e.is_nan()),
            "{a} != {e}"
        );
    }
}
#[test]
fn accumulation_promotes_and_wraps() {
    let a = NdArray::from_shape_vec(&[2, 3], vec![120i8; 6]).unwrap();
    let s: NdArray<i64> = sum(&a, Some(1), true).unwrap();
    assert_eq!(s.shape(), &[2, 1]);
    assert_eq!(s.to_vec(), vec![360, 360]);
    assert_eq!(
        sum(&NdArray::from_vec(vec![i64::MAX, 1]), None, false)
            .unwrap()
            .to_vec(),
        vec![i64::MIN]
    );
    assert_eq!(
        cumsum(&NdArray::from_scalar(7i32), Some(0))
            .unwrap()
            .to_vec(),
        vec![7]
    );
    assert_eq!(
        sum(&NdArray::from_scalar(7i32), Some(-1), true)
            .unwrap()
            .shape(),
        &[]
    );
    assert_eq!(
        sum(&NdArray::from_vec(vec![true, false, true]), None, false)
            .unwrap()
            .to_vec(),
        vec![2i64]
    );
}
#[test]
fn reductions_across_layouts_and_empty_axes() {
    let a = NdArray::from_shape_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    close(
        &mean(&a, Some(0), false).unwrap().to_vec(),
        &[2.5, 3.5, 4.5],
    );
    close(
        &mean(&a.t(), Some(1), false).unwrap().to_vec(),
        &[2.5, 3.5, 4.5],
    );
    close(
        &var(&a.flip(Some(&[1])).unwrap(), Some(1), true, 0)
            .unwrap()
            .to_vec(),
        &[2. / 3., 2. / 3.],
    );
    let empty = NdArray::<f64>::zeros(&[2, 0, 3]).unwrap();
    assert_eq!(sum(&empty, Some(1), false).unwrap().to_vec(), vec![0.; 6]);
    assert!(min(&empty, Some(1), false).is_err());
    assert!(min(&empty, Some(0), false).unwrap().is_empty());
    assert!(argmin(&empty, Some(0), false).unwrap().is_empty());
    assert!(mean(&NdArray::from_scalar(2.), Some(0), false).is_err());
}
#[test]
fn nan_statistics_and_degrees_of_freedom() {
    let a = NdArray::from_shape_vec(
        &[2, 3],
        vec![1., f64::NAN, 3., f64::NAN, f64::NAN, f64::NAN],
    )
    .unwrap();
    close(
        &nanmean(&a, Some(1), false).unwrap().to_vec(),
        &[2., f64::NAN],
    );
    close(
        &nanvar(&a, Some(1), false, 1).unwrap().to_vec(),
        &[2., f64::NAN],
    );
    assert!(nanargmin(&a, Some(1), false).is_err());
    assert_eq!(argmin(&a, Some(1), false).unwrap().to_vec(), vec![1, 0]);
    close(&nansum(&a, Some(1), false).unwrap().to_vec(), &[4., 0.]);
    let v = NdArray::from_vec(vec![1f64, 3.]);
    assert!(var(&v, None, false, 2).unwrap().to_vec()[0].is_infinite());
    assert!(nanvar(&v, None, false, 2).unwrap().to_vec()[0].is_nan());
}
#[test]
fn complex_statistics_keep_output_types() {
    let a = NdArray::from_vec(vec![Complex32::new(1., 2.), Complex32::new(3., 4.)]);
    let m: NdArray<Complex32> = mean(&a, None, false).unwrap();
    assert_eq!(m.to_vec(), vec![Complex32::new(2., 3.)]);
    let variance: NdArray<f32> = var(&a, None, false, 0).unwrap();
    assert_eq!(variance.to_vec(), vec![2.]);
    let nan = NdArray::from_vec(vec![Complex64::new(f64::NAN, 1.), Complex64::new(1., 2.)]);
    let median = median(&nan, None, false).unwrap().to_vec()[0];
    assert!(median.re.is_nan());
    assert_eq!(median.im, 1.);
    let all_nan = NdArray::from_vec(vec![
        Complex64::new(f64::NAN, 5.),
        Complex64::new(f64::NAN, 1.),
    ]);
    assert_eq!(nanmedian(&all_nan, None, false).unwrap().to_vec()[0].im, 1.);

    let both_nan = NdArray::from_vec(vec![Complex64::new(f64::NAN, f64::NAN)]);
    let rmin = nanmin(&both_nan, None, false).unwrap().to_vec()[0];
    let rmax = nanmax(&both_nan, None, false).unwrap().to_vec()[0];
    assert!(rmin.re.is_nan() && rmin.im.is_nan(), "rmin={rmin:?}");
    assert!(rmax.re.is_nan() && rmax.im.is_nan(), "rmax={rmax:?}");

    let first_nan = NdArray::from_vec(vec![
        Complex64::new(1.0, f64::NAN),
        Complex64::new(f64::NAN, 2.0),
    ]);
    let fmin = nanmin(&first_nan, None, false).unwrap().to_vec()[0];
    assert_eq!(fmin.re, 1.0);
    assert!(fmin.im.is_nan());
}
#[test]
fn stable_order_and_integer_precision() {
    let a = NdArray::from_vec(vec![0., -0., f64::NAN, 1., -1., f64::NAN]);
    assert_eq!(argsort(&a, None).unwrap().to_vec(), vec![4, 0, 1, 3, 2, 5]);
    let b = NdArray::from_vec(vec![i64::MAX, i64::MAX - 1, 1]);
    assert_eq!(
        sort(&b, None).unwrap().to_vec(),
        vec![1, i64::MAX - 1, i64::MAX]
    );
    assert_eq!(
        argsort(&NdArray::from_scalar(3.), Some(0)).unwrap().shape(),
        &[1]
    );
    assert!(sort(&NdArray::from_scalar(3.), Some(0)).is_err());
    assert_eq!(
        searchsorted(
            &sort(&a, None).unwrap(),
            &NdArray::from_vec(vec![-0., f64::NAN]),
            SearchSide::Right
        )
        .unwrap()
        .to_vec(),
        vec![3, 6]
    );
}
#[test]
fn partition_and_top_k_invariants() {
    let a = NdArray::from_shape_vec(&[3, 2], vec![5, 2, 1, 8, 3, 4]).unwrap();
    let p = partition(&a, 1, Some(0)).unwrap();
    assert_eq!(&p.to_vec()[2..4], &[3, 4]);
    assert_eq!(top_k(&a, 2, 0).unwrap().to_vec(), vec![5, 8, 3, 4]);
    assert_eq!(top_k(&a, 0, 1).unwrap().shape(), &[3, 0]);
    assert!(partition(&a, 3, Some(0)).is_err());
    assert!(
        partition(&NdArray::<f64>::from_vec(vec![]), 99, None)
            .unwrap()
            .is_empty()
    );
    let max_wide_cols = (isize::MAX as usize - 64) / std::mem::size_of::<f64>();
    let wide_empty = NdArray::<f64>::zeros(&[0, max_wide_cols]).unwrap();
    assert!(argsort(&wide_empty, Some(1)).unwrap().is_empty());
    assert!(partition(&wide_empty, -1, Some(1)).unwrap().is_empty());
    assert!(median(&wide_empty, Some(1), false).unwrap().is_empty());
    assert!(top_k(&wide_empty, 1, 1).unwrap().is_empty());
    assert!(NdArray::<f64>::zeros(&[0, isize::MAX as usize]).is_err());
    assert!(NdArray::<f64>::zeros(&[0, (isize::MAX as usize) + 1]).is_err());
}
#[test]
fn quantiles_selection_matches_sorting() {
    let a = NdArray::from_vec(vec![8., 1., 4., 2., 6., 3., 5., 7.]);
    close(
        &quantile(
            &a,
            &[0., 0.25, 0.5, 0.75, 1.],
            None,
            false,
            QuantileMethod::Linear,
        )
        .unwrap()
        .to_vec(),
        &[1., 2.75, 4.5, 6.25, 8.],
    );
    close(
        &quantile(&a, &[0.25, 0.5], None, true, QuantileMethod::Linear)
            .unwrap()
            .to_vec(),
        &[2.75, 4.5],
    );
    assert!(quantile(&a, &[f64::NAN], None, false, QuantileMethod::Linear).is_err());
    assert!(
        quantile(
            &NdArray::<f64>::from_vec(vec![]),
            &[],
            None,
            false,
            QuantileMethod::Linear
        )
        .is_err()
    );
    close(&median(&a, None, false).unwrap().to_vec(), &[4.5]);
    close(
        &nanmedian(&NdArray::from_vec(vec![1., f64::NAN, 3.]), None, false)
            .unwrap()
            .to_vec(),
        &[2.],
    );
    let extreme = NdArray::from_vec(vec![i64::MIN, i64::MAX]);
    let q = quantile(&extreme, &[0.5], None, false, QuantileMethod::Linear).unwrap();
    assert_eq!(q.to_vec(), vec![i64::MAX as f64]);
}
#[test]
fn weighted_average_and_covariance() {
    let a = NdArray::from_shape_vec(&[2, 3], vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
    let m: NdArray<f64> =
        average(&a, &NdArray::from_vec(vec![1., 2., 1.]), Some(1), false).unwrap();
    close(&m.to_vec(), &[2., 5.]);
    assert!(average(&a, &NdArray::from_vec(vec![0.; 3]), Some(1), false).is_err());
    close(&cov(&a, true, 1).unwrap().to_vec(), &[1., 1., 1., 1.]);
    close(&corrcoef(&a, true).unwrap().to_vec(), &[1., 1., 1., 1.]);
    let row = NdArray::from_shape_vec(&[1, 3], vec![1., 2., 3.]).unwrap();
    let c = cov(&row, false, 1).unwrap();
    assert_eq!(c.shape(), &[3, 3]);
    assert!(c.iter().all(f64::is_nan));
    assert_eq!(
        cov(&NdArray::<f64>::zeros(&[1, 0]).unwrap(), false, 1)
            .unwrap()
            .shape(),
        &[0, 0]
    );
}
#[test]
fn unique_inverse_and_nan_membership() {
    let a = NdArray::from_shape_vec(&[2, 3], vec![3., 1., 3., f64::NAN, -0., f64::NAN]).unwrap();
    let u = unique_all(&a).unwrap();
    assert_eq!(u.indices.to_vec(), vec![4, 1, 0, 3, 5]);
    assert_eq!(u.counts.to_vec(), vec![1, 1, 2, 1, 1]);
    assert_eq!(u.inverse.shape(), a.shape());
    let vals = u.values.to_vec();
    close(
        &u.inverse
            .iter()
            .map(|i| vals[i as usize])
            .collect::<Vec<_>>(),
        &a.to_vec(),
    );
    let nan = NdArray::from_vec(vec![f64::NAN]);
    assert_eq!(isin(&nan, &nan, false).unwrap().to_vec(), vec![false]);
    assert!(intersect1d(&nan, &nan).unwrap().is_empty());
    assert_eq!(setdiff1d(&nan, &nan).unwrap().len(), 1);
}
#[test]
fn histogram_edges_weights_and_density() {
    let a = NdArray::from_vec(vec![-1., 0., 0.5, 1., 2., f64::NAN]);
    close(
        &histogram(&a, Bins::Edges(&[0., 1., 2.]), None, None, false)
            .unwrap()
            .counts
            .to_vec(),
        &[2., 2.],
    );
    close(
        &histogram(&a, Bins::Edges(&[0., 0., 1.]), None, None, false)
            .unwrap()
            .counts
            .to_vec(),
        &[0., 3.],
    );
    let h = histogram(&a, Bins::Count(2), Some((0., 2.)), None, true).unwrap();
    close(&h.counts.to_vec(), &[0.5, 0.5]);
    assert!(histogram(&a, Bins::Count(2), None, None, false).is_err());
    assert_eq!(
        digitize(&a, &NdArray::from_vec(vec![2., 1., 0.]), false)
            .unwrap()
            .to_vec(),
        vec![3, 2, 2, 1, 0, 0]
    );
    close(
        &bincount(&NdArray::from_vec(vec![0, 1, 1, 3]), None, 5)
            .unwrap()
            .to_vec(),
        &[1., 2., 0., 1., 0.],
    );
    let edge_neighbor = (-4f64) * 0.8 + 9f64 * 0.2;
    assert_eq!(
        histogram(
            &NdArray::from_vec(vec![edge_neighbor]),
            Bins::Count(5),
            Some((-4., 9.)),
            None,
            false
        )
        .unwrap()
        .counts
        .to_vec(),
        vec![1., 0., 0., 0., 0.]
    );
}
#[test]
fn histogramdd_counts_last_edges() {
    let a = NdArray::from_shape_vec(&[3, 2], vec![0., 0., 1., 1., 2., 2.]).unwrap();
    let h = histogramdd(
        &a,
        &[Bins::Edges(&[0., 1., 2.]), Bins::Edges(&[0., 1., 2.])],
        None,
        false,
    )
    .unwrap();
    assert_eq!(h.counts.shape(), &[2, 2]);
    close(&h.counts.to_vec(), &[1., 0., 0., 2.]);
}
#[test]
fn calculus_and_signal() {
    let a = NdArray::from_vec(vec![1., 2., 3.]);
    let b = NdArray::from_vec(vec![1., 2.]);
    close(
        &convolve(&a, &b, ConvolveMode::Full).unwrap().to_vec(),
        &[1., 4., 7., 6.],
    );
    close(
        &correlate(&a, &b, ConvolveMode::Full).unwrap().to_vec(),
        &[2., 5., 8., 3.],
    );
    close(&diff(&a, 1, 0).unwrap().to_vec(), &[1., 1.]);
    close(&gradient(&a, 1., 0, 2).unwrap().to_vec(), &[1., 1., 1.]);
    close(&trapezoid(&a, None, 1., 0).unwrap().to_vec(), &[4.]);
    close(
        &interp(
            &NdArray::from_vec(vec![-1., 0.5, 2.]),
            &NdArray::from_vec(vec![0., 1.]),
            &NdArray::from_vec(vec![1., 3.]),
            None,
            None,
        )
        .unwrap()
        .to_vec(),
        &[1., 2., 3.],
    );
    close(
        &unwrap(
            &NdArray::from_vec(vec![0., 3., -3., -2.]),
            None,
            0,
            std::f64::consts::TAU,
        )
        .unwrap()
        .to_vec(),
        &[
            0.,
            3.,
            std::f64::consts::TAU - 3.,
            std::f64::consts::TAU - 2.,
        ],
    );
    assert!(unwrap(&NdArray::from_scalar(3.), None, 0, 6.).is_err());
    assert!(trapezoid(&NdArray::from_scalar(3.), None, 1., 0).is_err());
}

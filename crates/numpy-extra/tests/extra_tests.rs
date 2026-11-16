use numpy_extra::*;

#[test]
fn masked_variance_preserves_numpy_scalar_and_array_mask_rules() {
    let nan = MaskedArray::new(
        NdArray::from_shape_vec(&[2, 2], vec![f64::NAN, 1., f64::NAN, 2.]).unwrap(),
        NdArray::from_scalar(false),
    )
    .unwrap();
    let result = nan.var(Some(1), false, 0).unwrap();
    assert_eq!(result.data().to_vec(), vec![0., 0.]);
    assert_eq!(result.mask().to_vec(), vec![false, false]);
    assert_eq!(nan.var(None, false, 0).unwrap().mask().to_vec(), vec![true]);
    let overflow = MaskedArray::new(
        NdArray::from_shape_vec(&[2, 2], vec![1e200, -1e200, 1e200, -1e200]).unwrap(),
        NdArray::from_scalar(false),
    )
    .unwrap();
    assert!(
        overflow
            .var(Some(1), false, 0)
            .unwrap()
            .data()
            .iter()
            .all(f64::is_infinite)
    );
    assert_eq!(
        overflow.var(Some(1), false, 0).unwrap().mask().to_vec(),
        vec![false, false]
    );
    assert_eq!(
        overflow.std(Some(1), false, 0).unwrap().mask().to_vec(),
        vec![true, true]
    );
    let scalar = MaskedArray::new(
        NdArray::from_vec(vec![1f64, 3.]),
        NdArray::from_scalar(false),
    )
    .unwrap();
    assert_eq!(
        scalar.var(None, false, 3).unwrap().data().to_vec(),
        vec![-2.]
    );
    assert_eq!(
        scalar.var(None, false, 3).unwrap().mask().to_vec(),
        vec![false]
    );
    assert_eq!(
        scalar.std(None, false, 3).unwrap().mask().to_vec(),
        vec![true]
    );
}

#[test]
fn masked_reductions_exclude_masks_but_preserve_nan() {
    let data = NdArray::from_shape_vec(&[2, 3], vec![1f64, 2., 3., 4., f64::NAN, 6.]).unwrap();
    let mask =
        NdArray::from_shape_vec(&[2, 3], vec![false, true, false, true, true, true]).unwrap();
    let a = MaskedArray::new(data, mask).unwrap();
    assert_eq!(a.compressed().to_vec(), vec![1., 3.]);
    assert_eq!(a.count(Some(1), false).unwrap().to_vec(), vec![2, 0]);
    let mean = a.mean(Some(1), false).unwrap();
    assert_eq!(mean.data().to_vec()[0], 2.);
    assert_eq!(mean.mask().to_vec(), vec![false, true]);
    let var = a.var(Some(1), false, 0).unwrap();
    assert_eq!(var.data().to_vec()[0], 1.);
    assert_eq!(var.mask().to_vec(), vec![false, true]);
    assert_eq!(
        a.filled(Some(-1.)).unwrap().to_vec(),
        vec![1., -1., 3., -1., -1., -1.]
    );
    let nan = MaskedArray::new(
        NdArray::from_vec(vec![1f64, f64::NAN]),
        NdArray::from_scalar(false),
    )
    .unwrap();
    assert!(nan.mean(None, false).unwrap().data().to_vec()[0].is_nan());
    assert_eq!(nan.mean(None, false).unwrap().mask().to_vec(), vec![false]);
}
#[test]
fn masked_broadcasts_and_integer_sums_widen() {
    let a = MaskedArray::new(
        NdArray::from_shape_vec(&[2, 2], vec![120i8; 4]).unwrap(),
        NdArray::from_vec(vec![false, true]),
    )
    .unwrap();
    let sum: MaskedArray<i64> = a.sum(None, false).unwrap();
    assert_eq!(sum.data().to_vec(), vec![240]);
    assert_eq!(sum.mask().to_vec(), vec![false]);
    let b = MaskedArray::new(NdArray::from_scalar(2i8), NdArray::from_scalar(false)).unwrap();
    assert_eq!(a.add(&b).unwrap().data().to_vec(), vec![122i8; 4]);
    let mut a = a;
    a.set_mask(NdArray::from_scalar(false)).unwrap();
    assert_eq!(a.count(None, false).unwrap().to_vec(), vec![4]);
    assert_eq!(a.mean(None, false).unwrap().data().to_vec(), vec![120.]);
    assert!(a.set_mask(NdArray::from_vec(vec![true; 3])).is_err());
    assert_eq!(a.fill_value, 63);
}
#[test]
fn calendar_roundtrips_leap_days_negative_epoch_and_units() {
    for date in [
        "1970-01-01",
        "1969-12-31",
        "2000-02-29",
        "1900-03-01",
        "0000-01-01",
        "-0001-12-31",
        "2026-11-16",
    ] {
        let parsed = DateTime64::parse(date, TimeUnit::Day).unwrap();
        assert_eq!(
            parsed.to_string(),
            if date == "-0001-12-31" {
                "-001-12-31"
            } else {
                date
            }
        );
    }
    assert!(DateTime64::parse("1900-02-29", TimeUnit::Day).is_err());
    let timestamp =
        DateTime64::parse("1969-12-31T23:59:59.999999999", TimeUnit::Nanosecond).unwrap();
    assert_eq!(timestamp.value, -1);
    assert_eq!(timestamp.to_string(), "1969-12-31T23:59:59.999999999");
    assert_eq!(timestamp.cast(TimeUnit::Day).unwrap().value, -1);
    assert_eq!(
        DateTime64::parse("2026-11", TimeUnit::Month)
            .unwrap()
            .to_string(),
        "2026-11"
    );
    assert_eq!(
        TimeDelta64::new(-1, TimeUnit::Second)
            .cast(TimeUnit::Minute)
            .unwrap()
            .value,
        -1
    );
    assert!(
        TimeDelta64::new(1, TimeUnit::Month)
            .cast(TimeUnit::Day)
            .is_err()
    );
    assert!(
        DateTime64::new(i64::MAX, TimeUnit::Day)
            .cast(TimeUnit::Nanosecond)
            .is_err()
    );
}
#[test]
fn datetime_arithmetic_and_nat() {
    let day = DateTime64::parse("2026-11-16", TimeUnit::Day).unwrap();
    let hour = day.add(TimeDelta64::new(1, TimeUnit::Hour)).unwrap();
    assert_eq!(hour.unit, TimeUnit::Hour);
    assert_eq!(hour.to_string(), "2026-11-16T01");
    assert_eq!(hour.difference(day, TimeUnit::Hour).unwrap().value, 1);
    assert!(
        DateTime64::nat(TimeUnit::Day)
            .add(TimeDelta64::new(1, TimeUnit::Day))
            .unwrap()
            .is_nat()
    );
    assert_eq!(DateTime64::nat(TimeUnit::Day).to_string(), "NaT");
    assert_eq!(
        TimeDelta64::new(1, TimeUnit::Day)
            .add(TimeDelta64::new(1, TimeUnit::Hour))
            .unwrap(),
        TimeDelta64::new(25, TimeUnit::Hour)
    );
}
#[test]
fn business_calendars_roll_count_and_large_offsets() {
    let parse = |s| DateTime64::parse(s, TimeUnit::Day).unwrap();
    let calendar = BusinessCalendar::new(
        [true, true, true, true, true, false, false],
        &[parse("2026-11-17")],
    )
    .unwrap();
    assert!(!calendar.is_busday(parse("2026-11-15")).unwrap());
    assert_eq!(
        calendar
            .busday_count(parse("2026-11-16"), parse("2026-11-23"))
            .unwrap(),
        4
    );
    assert_eq!(
        calendar
            .busday_count(parse("2026-11-23"), parse("2026-11-16"))
            .unwrap(),
        -4
    );
    assert_eq!(
        calendar
            .busday_offset(parse("2026-11-16"), 1, Roll::Raise)
            .unwrap()
            .to_string(),
        "2026-11-18"
    );
    assert_eq!(
        calendar
            .busday_offset(parse("2026-11-16"), -1, Roll::Raise)
            .unwrap()
            .to_string(),
        "2026-11-13"
    );
    assert_eq!(
        calendar
            .busday_offset(parse("2026-11-15"), 0, Roll::Forward)
            .unwrap()
            .to_string(),
        "2026-11-16"
    );
    assert_eq!(
        calendar
            .busday_offset(parse("2026-10-31"), 0, Roll::ModifiedFollowing)
            .unwrap()
            .to_string(),
        "2026-10-30"
    );
    let start = parse("1970-01-01");
    let end = busday_offset(start, 1_000_000, Roll::Raise).unwrap();
    assert_eq!(busday_count(start, end).unwrap(), 1_000_000);
}
#[test]
fn unicode_strings_operations_and_broadcasting() {
    let a = StringArray::from_vec(vec!["  mèo  ".into(), "-42".into(), "Straße".into()]);
    assert_eq!(a.strip(None).data(), &["mèo", "-42", "Straße"]);
    assert_eq!(a.upper().data()[2], "STRASSE");
    assert_eq!(a.find("è").to_vec(), vec![3, -1, -1]);
    assert_eq!(a.zfill(6).data()[1], "-00042");
    assert_eq!(
        StringArray::from_scalar("ab").center(5, '-').data(),
        &["--ab-"]
    );
    let b = StringArray::new(&[2, 1], vec!["x".into(), "y".into()]).unwrap();
    let c = StringArray::from_vec(vec!["1".into(), "2".into(), "3".into()]);
    let result = b.add(&c).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.data(), &["x1", "x2", "x3", "y1", "y2", "y3"]);
    assert!(a.split(Some(""), None).is_err());
    assert_eq!(
        StringArray::from_scalar(" a  b c ")
            .split(None, Some(1))
            .unwrap(),
        vec![vec!["a".to_string(), "b c ".to_string()]]
    );
}
#[test]
fn structured_records_keep_unaligned_integer_and_float_bits() {
    let mut a = StructuredArray::zeros(
        &[2],
        &[
            ("tag", DType::U8),
            ("large", DType::I64),
            ("value", DType::F64),
        ],
    )
    .unwrap();
    assert_eq!(a.itemsize(), 17);
    assert_eq!(a.fields()[1].offset, 1);
    a.set_field("large", &NdArray::from_vec(vec![i64::MAX, i64::MIN + 1]))
        .unwrap();
    a.set_field("value", &NdArray::from_vec(vec![-0f64, f64::INFINITY]))
        .unwrap();
    a.set_field("tag", &NdArray::from_scalar(7u8)).unwrap();
    assert_eq!(
        a.field::<i64>("large").unwrap().to_vec(),
        vec![i64::MAX, i64::MIN + 1]
    );
    assert_eq!(a.field::<u8>("tag").unwrap().to_vec(), vec![7, 7]);
    assert_eq!(
        a.field::<f64>("value").unwrap().to_vec()[0].to_bits(),
        (-0f64).to_bits()
    );
    assert!(a.field::<i32>("large").is_err());
    assert!(StructuredArray::zeros(&[1], &[("a", DType::F64), ("a", DType::F64)]).is_err());
}

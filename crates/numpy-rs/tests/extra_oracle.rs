mod common;
use common::{assert_dyn_matches_oracle, py_assign_dyn, run_oracle};
use numpy_extra::*;

#[test]
fn masked_statistics_against_numpy() {
    let a = NdArray::from_shape_vec(
        &[3, 4],
        vec![1f64, 2., 3., 4., 5., 6., 7., 8., 9., f64::NAN, 11., 12.],
    )
    .unwrap();
    let mask = NdArray::from_shape_vec(
        &[3, 4],
        vec![
            false, true, false, false, true, true, true, true, false, false, true, false,
        ],
    )
    .unwrap();
    let masked = MaskedArray::new(a.clone(), mask.clone()).unwrap();
    for axis in [None, Some(0), Some(1)] {
        let ax = axis.map_or("None".into(), |a| a.to_string());
        let oracle = run_oracle(&format!(
            "{}\n{}\na=np.ma.array(a,mask=m)\nfor fn in ['sum','prod','mean','var','std','min','max']:\n r=getattr(a,fn)(axis={ax})\n emit_arr(fn,np.ma.filled(r,np.nan))\n emit_arr(fn+'mask',np.ma.getmaskarray(r))",
            py_assign_dyn("a", &a.clone().into()),
            py_assign_dyn("m", &mask.clone().into())
        ));
        let outputs: Vec<(&str, MaskedArray<f64>)> = vec![
            ("sum", masked.sum(axis, false).unwrap()),
            ("prod", masked.prod(axis, false).unwrap()),
            ("mean", masked.mean(axis, false).unwrap()),
            ("var", masked.var(axis, false, 0).unwrap()),
            ("std", masked.std(axis, false, 0).unwrap()),
            ("min", masked.min(axis, false).unwrap()),
            ("max", masked.max(axis, false).unwrap()),
        ];
        for (tag, output) in outputs {
            assert_dyn_matches_oracle(
                &output.filled(Some(f64::NAN)).unwrap().into(),
                &oracle.arrays[tag],
                1e-12,
                1e-12,
                true,
                false,
                tag,
            );
            assert_dyn_matches_oracle(
                &output.mask().clone().into(),
                &oracle.arrays[&format!("{tag}mask")],
                0.,
                0.,
                false,
                false,
                tag,
            );
        }
    }
}
#[test]
fn business_days_rolls_holidays_and_reverse_endpoints_against_numpy() {
    let parse = |s| DateTime64::parse(s, TimeUnit::Day).unwrap();
    let calendar = BusinessCalendar::new(
        [true, true, true, true, true, false, false],
        &[parse("2026-11-17"), parse("2026-12-01")],
    )
    .unwrap();
    let mut script = "hol=['2026-11-17','2026-12-01']\n".to_string();
    let dates = [
        "2026-10-31",
        "2026-11-01",
        "2026-11-16",
        "2026-11-17",
        "2026-11-28",
    ];
    for (i, date) in dates.iter().enumerate() {
        for (j, other) in dates.iter().enumerate() {
            script.push_str(&format!(
                "emit_val('c{i}_{j}',np.busday_count('{date}','{other}',holidays=hol))\n"
            ));
        }
    }
    let rolls = [
        ("forward", Roll::Forward),
        ("backward", Roll::Backward),
        ("modifiedfollowing", Roll::ModifiedFollowing),
        ("modifiedpreceding", Roll::ModifiedPreceding),
    ];
    for (i, date) in dates.iter().enumerate() {
        for (j, (roll, _)) in rolls.iter().enumerate() {
            for (k, offset) in [-30, -1, 0, 1, 30].iter().enumerate() {
                script.push_str(&format!("emit_val('o{i}_{j}_{k}',np.busday_offset('{date}',{offset},roll='{roll}',holidays=hol))\n"));
            }
        }
    }
    let oracle = run_oracle(&script);
    for (i, date) in dates.iter().enumerate() {
        for (j, other) in dates.iter().enumerate() {
            assert_eq!(
                calendar
                    .busday_count(parse(date), parse(other))
                    .unwrap()
                    .to_string(),
                oracle.values[&format!("c{i}_{j}")]
            );
        }
    }
    for (i, date) in dates.iter().enumerate() {
        for (j, (_, roll)) in rolls.iter().enumerate() {
            for (k, offset) in [-30, -1, 0, 1, 30].iter().enumerate() {
                assert_eq!(
                    calendar
                        .busday_offset(parse(date), *offset, *roll)
                        .unwrap()
                        .to_string(),
                    oracle.values[&format!("o{i}_{j}_{k}")]
                );
            }
        }
    }
}
#[test]
fn datetime_units_and_duration_arithmetic_against_numpy() {
    let mut script = String::new();
    let dates = [
        "1969-12-31T23:59:59.999999999",
        "1970-01-01T01:02:03.123456789",
        "2000-02-29T12:00:00",
        "-0001-12-31",
    ];
    let units = [
        ("Y", TimeUnit::Year),
        ("M", TimeUnit::Month),
        ("W", TimeUnit::Week),
        ("D", TimeUnit::Day),
        ("h", TimeUnit::Hour),
        ("s", TimeUnit::Second),
        ("ms", TimeUnit::Millisecond),
        ("us", TimeUnit::Microsecond),
        ("ns", TimeUnit::Nanosecond),
    ];
    for (i, date) in dates.iter().enumerate() {
        for (j, (name, _)) in units.iter().enumerate() {
            if i == 3 && *name == "ns" {
                continue;
            }
            script.push_str(&format!(
                "emit_val('v{i}_{j}',np.datetime64('{date}','{name}').astype(np.int64))\n"
            ));
        }
    }
    script.push_str("emit_val('duration',(np.timedelta64(1,'D')+np.timedelta64(1,'h')).astype(np.int64))\nemit_val('difference',((np.datetime64('2026-01-02T00','h')-np.datetime64('2026-01-01T23','h')).astype('timedelta64[D]')).astype(np.int64))\n");
    let oracle = run_oracle(&script);
    for (i, date) in dates.iter().enumerate() {
        for (j, (_, unit)) in units.iter().enumerate() {
            if i == 3 && *unit == TimeUnit::Nanosecond {
                continue;
            }
            let parsed = DateTime64::parse(date, *unit).unwrap();
            assert_eq!(
                parsed.value.to_string(),
                oracle.values[&format!("v{i}_{j}")],
                "{date} {unit:?}"
            );
        }
    }
    let duration = TimeDelta64::new(1, TimeUnit::Day)
        .add(TimeDelta64::new(1, TimeUnit::Hour))
        .unwrap();
    assert_eq!(duration.value.to_string(), oracle.values["duration"]);
    let difference = DateTime64::parse("2026-01-02T00", TimeUnit::Hour)
        .unwrap()
        .difference(
            DateTime64::parse("2026-01-01T23", TimeUnit::Hour).unwrap(),
            TimeUnit::Day,
        )
        .unwrap();
    assert_eq!(difference.value.to_string(), oracle.values["difference"]);
    assert_eq!(
        DateTime64::new(1, TimeUnit::Day),
        DateTime64::new(24, TimeUnit::Hour)
    );
    assert_ne!(
        DateTime64::nat(TimeUnit::Day),
        DateTime64::nat(TimeUnit::Day)
    );
}
#[test]
fn unicode_strings_and_packed_records_against_numpy() {
    let input = StringArray::from_vec(vec![
        "  mèo  ".into(),
        "-42".into(),
        "Straße".into(),
        "\u{1c}foo\u{1c}".into(),
    ]);
    let oracle = run_oracle(
        "a=np.array(['  mèo  ','-42','Straße','\\x1cfoo\\x1c'])\nfor fn in ['upper','lower','strip']:\n emit_val(fn,'|'.join(getattr(np.strings,fn)(a)))\nemit_val('zfill','|'.join(np.strings.zfill(a,6)))\na=np.zeros(2,dtype=[('tag','u1'),('large','<i8'),('value','<f8')])\na['tag']=7\na['large']=[9223372036854775807,-9223372036854775807]\na['value']=[-0.,np.inf]\nemit_val('records',a.tobytes().hex())",
    );
    for (tag, result) in [
        ("upper", input.upper()),
        ("lower", input.lower()),
        ("strip", input.strip(None)),
        ("zfill", input.zfill(6)),
    ] {
        assert_eq!(result.data().join("|"), oracle.values[tag]);
    }
    let mut a = StructuredArray::zeros(
        &[2],
        &[
            ("tag", DType::U8),
            ("large", DType::I64),
            ("value", DType::F64),
        ],
    )
    .unwrap();
    a.set_field("tag", &NdArray::from_scalar(7u8)).unwrap();
    a.set_field("large", &NdArray::from_vec(vec![i64::MAX, i64::MIN + 1]))
        .unwrap();
    a.set_field("value", &NdArray::from_vec(vec![-0f64, f64::INFINITY]))
        .unwrap();
    assert_eq!(common::bytes_to_hex(a.as_bytes()), oracle.values["records"]);
}

#[test]
fn masked_variance_ddof_edge_cases_against_numpy() {
    let v1 = NdArray::from_vec(vec![1.0, 2.0]);
    let m_false = NdArray::from_vec(vec![false, false]);
    let m_true = NdArray::from_vec(vec![true, true]);
    let v2 = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 6.0]).unwrap();
    let m2 = NdArray::from_shape_vec(&[2, 2], vec![false, false, false, false]).unwrap();

    let oracle = run_oracle(&format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        py_assign_dyn("v1", &v1.clone().into()),
        py_assign_dyn("mf", &m_false.clone().into()),
        py_assign_dyn("mt", &m_true.clone().into()),
        py_assign_dyn("v2", &v2.clone().into()),
        py_assign_dyn("m2", &m2.clone().into()),
        r#"
a_exp = np.ma.array(v1, mask=mf)
a_allm = np.ma.array(v1, mask=mt)
a2 = np.ma.array(v2, mask=m2)
for ddof in [0, 1, 2, 3]:
    for kd in [False, True]:
        r = a_exp.var(ddof=ddof, keepdims=kd)
        emit_arr(f'v1_var_{ddof}_{kd}', np.ma.filled(r, np.nan))
        emit_arr(f'v1_var_{ddof}_{kd}_m', np.ma.getmaskarray(r))
        s = a_exp.std(ddof=ddof, keepdims=kd)
        emit_arr(f'v1_std_{ddof}_{kd}', np.ma.filled(s, np.nan))
        emit_arr(f'v1_std_{ddof}_{kd}_m', np.ma.getmaskarray(s))
    rm = a_allm.var(ddof=ddof)
    emit_arr(f'allm_var_{ddof}_m', np.ma.getmaskarray(rm))
    r2 = a2.var(axis=1, ddof=ddof)
    emit_arr(f'v2_var_{ddof}', np.ma.filled(r2, np.nan))
    emit_arr(f'v2_var_{ddof}_m', np.ma.getmaskarray(r2))
"#
    ));

    let ma_exp = MaskedArray::new(v1.clone(), m_false).unwrap();
    let ma_allm = MaskedArray::new(v1, m_true).unwrap();
    let ma2 = MaskedArray::new(v2, m2).unwrap();

    for ddof in [0usize, 1, 2, 3] {
        for (kd_str, kd) in [("False", false), ("True", true)] {
            let rv = ma_exp.var(None, kd, ddof).unwrap();
            assert_dyn_matches_oracle(
                &rv.filled(Some(f64::NAN)).unwrap().into(),
                &oracle.arrays[&format!("v1_var_{ddof}_{kd_str}")],
                1e-12,
                1e-12,
                true,
                false,
                &format!("v1_var_{ddof}_{kd_str}"),
            );
            assert_dyn_matches_oracle(
                &rv.mask().clone().into(),
                &oracle.arrays[&format!("v1_var_{ddof}_{kd_str}_m")],
                0.,
                0.,
                false,
                false,
                &format!("v1_var_{ddof}_{kd_str}_m"),
            );
            let rs = ma_exp.std(None, kd, ddof).unwrap();
            assert_dyn_matches_oracle(
                &rs.filled(Some(f64::NAN)).unwrap().into(),
                &oracle.arrays[&format!("v1_std_{ddof}_{kd_str}")],
                1e-12,
                1e-12,
                true,
                false,
                &format!("v1_std_{ddof}_{kd_str}"),
            );
            assert_dyn_matches_oracle(
                &rs.mask().clone().into(),
                &oracle.arrays[&format!("v1_std_{ddof}_{kd_str}_m")],
                0.,
                0.,
                false,
                false,
                &format!("v1_std_{ddof}_{kd_str}_m"),
            );
        }
        let rm = ma_allm.var(None, false, ddof).unwrap();
        assert_dyn_matches_oracle(
            &rm.mask().clone().into(),
            &oracle.arrays[&format!("allm_var_{ddof}_m")],
            0.,
            0.,
            false,
            false,
            &format!("allm_var_{ddof}_m"),
        );
        let r2 = ma2.var(Some(1), false, ddof).unwrap();
        assert_dyn_matches_oracle(
            &r2.filled(Some(f64::NAN)).unwrap().into(),
            &oracle.arrays[&format!("v2_var_{ddof}")],
            1e-12,
            1e-12,
            true,
            false,
            &format!("v2_var_{ddof}"),
        );
        assert_dyn_matches_oracle(
            &r2.mask().clone().into(),
            &oracle.arrays[&format!("v2_var_{ddof}_m")],
            0.,
            0.,
            false,
            false,
            &format!("v2_var_{ddof}_m"),
        );
    }
}

#[test]
fn round2_datetime_busday_and_masked_scalar_axis_against_numpy() {
    let oracle = run_oracle(
        r#"
try:
    _ = np.datetime64('NaT', 'D') + np.timedelta64('NaT', 'M')
except Exception as e:
    emit_err('nat_d_plus_m', e)
try:
    _ = np.datetime64('NaT', 'D') + np.timedelta64(1, 'M')
except Exception as e:
    emit_err('nat_d_plus_1m', e)

emit_val('nat_d_s_nat', str(np.datetime64('NaT', 'D') + np.timedelta64('NaT', 's')))
emit_val('nat_d_s_unit', str(np.datetime_data(np.datetime64('NaT', 'D') + np.timedelta64('NaT', 's'))[0]))
emit_val('nat_m_d_unit', str(np.datetime_data(np.datetime64('NaT', 'M') + np.timedelta64('NaT', 'D'))[0]))
emit_val('nat_m_1d_unit', str(np.datetime_data(np.datetime64('NaT', 'M') + np.timedelta64(1, 'D'))[0]))

try:
    np.busday_offset(np.datetime64('2024-03-20T00', 'h'), 1)
except Exception as e:
    emit_err('bd_off_h', e)
try:
    np.busday_count(np.datetime64('2024-03-20T00', 'h'), np.datetime64('2024-03-21T00', 'h'))
except Exception as e:
    emit_err('bd_cnt_h', e)
emit_val('is_bd_h', int(np.is_busday(np.datetime64('2024-03-20T05', 'h'))))
emit_val('bd_max', np.busday_offset(np.datetime64((1 << 63) - 11, 'D'), 5).view(np.int64))

s0 = np.ma.array(3.0, mask=False)
for fn in ['sum', 'prod', 'mean', 'var', 'std', 'min', 'max']:
    for ax in [0, -1]:
        r = getattr(s0, fn)(axis=ax)
        emit_arr(f's0_{fn}_{ax}', np.ma.filled(r, np.nan))
        emit_arr(f's0_{fn}_{ax}_m', np.ma.getmaskarray(r))
"#,
    );

    // 13. DateTime64::add NaT unit compatibility & promotion
    assert!(oracle.errors.contains_key("nat_d_plus_m"));
    assert!(oracle.errors.contains_key("nat_d_plus_1m"));
    assert!(DateTime64::nat(TimeUnit::Day).add(TimeDelta64::nat(TimeUnit::Month)).is_err());
    assert!(DateTime64::nat(TimeUnit::Day).add(TimeDelta64::new(1, TimeUnit::Month)).is_err());

    let nat_ds = DateTime64::nat(TimeUnit::Day).add(TimeDelta64::nat(TimeUnit::Second)).unwrap();
    assert!(nat_ds.is_nat() && nat_ds.unit == TimeUnit::Second);
    assert_eq!(oracle.values["nat_d_s_unit"], "s");
    let nat_md = DateTime64::nat(TimeUnit::Month).add(TimeDelta64::nat(TimeUnit::Day)).unwrap();
    assert!(nat_md.is_nat() && nat_md.unit == TimeUnit::Day);
    assert_eq!(oracle.values["nat_m_d_unit"], "D");
    let nat_m1d = DateTime64::nat(TimeUnit::Month).add(TimeDelta64::new(1, TimeUnit::Day)).unwrap();
    assert!(nat_m1d.is_nat() && nat_m1d.unit == TimeUnit::Day);
    assert_eq!(oracle.values["nat_m_1d_unit"], "D");

    // 14. BusinessCalendar safe cast rejects sub-day units while is_busday allows them
    let h0 = DateTime64::parse("2024-03-20T05", TimeUnit::Hour).unwrap();
    let h1 = DateTime64::parse("2024-03-21T00", TimeUnit::Hour).unwrap();
    assert!(oracle.errors.contains_key("bd_off_h"));
    assert!(oracle.errors.contains_key("bd_cnt_h"));
    assert!(busday_offset(h0, 1, Roll::Forward).is_err());
    assert!(busday_count(h0, h1).is_err());
    assert!(BusinessCalendar::new([true, true, true, true, true, false, false], &[h0]).is_err());
    assert_eq!(is_busday(h0).unwrap(), oracle.values["is_bd_h"] == "1");
    let m0 = DateTime64::parse("2024-03", TimeUnit::Month).unwrap();
    assert!(busday_offset(m0, 1, Roll::Forward).is_ok());

    // 15. busday_offset near i64::MAX / i64::MIN
    let d_max = DateTime64::new(i64::MAX - 10, TimeUnit::Day);
    let r_max = busday_offset(d_max, 5, Roll::Forward).unwrap();
    assert_eq!(r_max.value.to_string(), oracle.values["bd_max"]);
    assert!(busday_offset(d_max, 50, Roll::Forward).is_err());
    let d_min = DateTime64::new(i64::MIN + 10, TimeUnit::Day);
    assert!(busday_offset(d_min, -3, Roll::Backward).is_ok());
    assert!(busday_offset(d_min, -50, Roll::Backward).is_err());

    // 16 & 17. MaskedArray 0-D reductions with axis=0/-1 and typed fill_value checks
    let ma0 = MaskedArray::new(NdArray::from_scalar(3.0f64), NdArray::from_scalar(false)).unwrap();
    assert_eq!(ma0.fill_value, 1e20);
    let ma_i64 = MaskedArray::new(NdArray::from_scalar(3i64), NdArray::from_scalar(false)).unwrap();
    assert_eq!(ma_i64.fill_value, 999999);

    for ax in [0isize, -1isize] {
        let ax_str = ax.to_string();
        let outs: Vec<(&str, MaskedArray<f64>)> = vec![
            ("sum", ma0.sum(Some(ax), false).unwrap()),
            ("prod", ma0.prod(Some(ax), false).unwrap()),
            ("mean", ma0.mean(Some(ax), false).unwrap()),
            ("var", ma0.var(Some(ax), false, 0).unwrap()),
            ("std", ma0.std(Some(ax), false, 0).unwrap()),
            ("min", ma0.min(Some(ax), false).unwrap()),
            ("max", ma0.max(Some(ax), false).unwrap()),
        ];
        for (tag, out) in outs {
            assert_dyn_matches_oracle(
                &out.filled(Some(f64::NAN)).unwrap().into(),
                &oracle.arrays[&format!("s0_{tag}_{ax_str}")],
                1e-12,
                1e-12,
                true,
                false,
                tag,
            );
            assert_dyn_matches_oracle(
                &out.mask().clone().into(),
                &oracle.arrays[&format!("s0_{tag}_{ax_str}_m")],
                0.,
                0.,
                false,
                false,
                tag,
            );
        }
    }
    assert!(ma0.mean(Some(1), false).is_err());
    assert!(ma0.var(Some(-2), false, 0).is_err());
}

#[test]
fn p1_timedelta_calendar_fixed_conversions_against_numpy() {
    let cases: &[(i64, &str, TimeUnit, &str, TimeUnit)] = &[
        (14, "M", TimeUnit::Month, "D", TimeUnit::Day),
        (-14, "M", TimeUnit::Month, "D", TimeUnit::Day),
        (1, "M", TimeUnit::Month, "D", TimeUnit::Day),
        (-1, "M", TimeUnit::Month, "D", TimeUnit::Day),
        (1, "Y", TimeUnit::Year, "D", TimeUnit::Day),
        (-1, "Y", TimeUnit::Year, "D", TimeUnit::Day),
        (14, "Y", TimeUnit::Year, "W", TimeUnit::Week),
        (-14, "Y", TimeUnit::Year, "W", TimeUnit::Week),
        (1, "M", TimeUnit::Month, "ns", TimeUnit::Nanosecond),
        (-1, "M", TimeUnit::Month, "ns", TimeUnit::Nanosecond),
        (30, "D", TimeUnit::Day, "M", TimeUnit::Month),
        (-30, "D", TimeUnit::Day, "M", TimeUnit::Month),
        (31, "D", TimeUnit::Day, "M", TimeUnit::Month),
        (-31, "D", TimeUnit::Day, "M", TimeUnit::Month),
        (365, "D", TimeUnit::Day, "M", TimeUnit::Month),
        (-365, "D", TimeUnit::Day, "M", TimeUnit::Month),
        (2_629_746, "s", TimeUnit::Second, "M", TimeUnit::Month),
        (-2_629_745, "s", TimeUnit::Second, "M", TimeUnit::Month),
    ];

    let mut py = String::new();
    for (i, &(val, src_name, _, dst_name, _)) in cases.iter().enumerate() {
        py.push_str(&format!(
            "emit_val('c{i}', np.timedelta64({val}, '{src_name}').astype('timedelta64[{dst_name}]').astype(np.int64))\n"
        ));
    }
    py.push_str("emit_val('diff_md', (np.datetime64('2021-03', 'M') - np.datetime64('2020-01', 'M')).astype('timedelta64[D]').astype(np.int64))\n");
    let oracle = run_oracle(&py);

    for (i, &(val, src_name, src_u, dst_name, dst_u)) in cases.iter().enumerate() {
        let got = TimeDelta64::new(val, src_u).cast(dst_u).unwrap().value;
        let exp: i64 = oracle.val(&format!("c{i}")).parse().unwrap();
        assert_eq!(got, exp, "({val}, {src_name}) -> {dst_name}");
    }

    let d1 = DateTime64::parse("2021-03", TimeUnit::Month).unwrap();
    let d0 = DateTime64::parse("2020-01", TimeUnit::Month).unwrap();
    let diff = d1.difference(d0, TimeUnit::Day).unwrap().value;
    let exp_diff: i64 = oracle.val("diff_md").parse().unwrap();
    assert_eq!(diff, exp_diff);
}

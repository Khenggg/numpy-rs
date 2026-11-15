mod common;

use common::{
    assert_aligned, assert_dyn_matches_oracle, assert_f64_bits_eq, edge_f64_values, run_oracle,
};
use numpy_types::*;

fn dtype_to_py_name(dt: DType) -> &'static str {
    dt.name()
}

#[test]
fn test_all_14x14_promote_types_against_numpy_oracle() {
    let oracle = run_oracle(
        r#"
dtypes = ['bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64','complex64','complex128']
for i, a in enumerate(dtypes):
    for j, b in enumerate(dtypes):
        emit_val(f"p_{i}_{j}", np.promote_types(a, b).name)
for k, combo in enumerate([
    ('bool', 'int8', 'uint8'),
    ('int16', 'uint16', 'float16'),
    ('int8', 'uint8', 'float16'),
    ('int8', 'uint16', 'float16'),
    ('int8', 'uint16', 'float32'),
    ('int8', 'uint16', 'complex64'),
    ('int16', 'uint16', 'float32'),
    ('int16', 'uint16', 'complex64'),
    ('int32', 'float32', 'complex64'),
    ('uint64', 'int8', 'float32'),
    ('bool', 'float16', 'complex64'),
]):
    emit_val(f"r_{k}", np.result_type(*combo).name)
"#,
    );

    for (i, &a) in DType::ALL.iter().enumerate() {
        for (j, &b) in DType::ALL.iter().enumerate() {
            let expected = DType::from_name(oracle.val(&format!("p_{i}_{j}"))).unwrap();
            let actual = promote_types(a, b);
            assert_eq!(
                actual, expected,
                "promote_types({a}, {b}): rust={actual} != numpy={expected}"
            );
            // Symmetry check
            assert_eq!(actual, promote_types(b, a), "promote_types not symmetric for {a}, {b}");
        }
    }

    let combos: Vec<&[DType]> = vec![
        &[DType::Bool, DType::I8, DType::U8],
        &[DType::I16, DType::U16, DType::F16],
        &[DType::I8, DType::U8, DType::F16],
        &[DType::I8, DType::U16, DType::F16],
        &[DType::I8, DType::U16, DType::F32],
        &[DType::I8, DType::U16, DType::C64],
        &[DType::I16, DType::U16, DType::F32],
        &[DType::I16, DType::U16, DType::C64],
        &[DType::I32, DType::F32, DType::C64],
        &[DType::U64, DType::I8, DType::F32],
        &[DType::Bool, DType::F16, DType::C64],
    ];
    for (k, combo) in combos.iter().enumerate() {
        let expected = DType::from_name(oracle.val(&format!("r_{k}"))).unwrap();
        let actual = result_type(combo).unwrap();
        assert_eq!(actual, expected, "result_type({combo:?}): rust={actual} != numpy={expected}");
    }
    assert!(result_type(&[]).is_err());
}

#[test]
fn test_all_14x14_can_cast_all_5_modes_against_numpy_oracle() {
    let oracle = run_oracle(
        r#"
dtypes = ['bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64','complex64','complex128']
modes = ['no', 'equiv', 'safe', 'same_kind', 'unsafe']
for m in modes:
    for i, a in enumerate(dtypes):
        for j, b in enumerate(dtypes):
            emit_val(f"{m}_{i}_{j}", int(np.can_cast(a, b, casting=m)))
"#,
    );

    let modes = [
        ("no", Casting::No),
        ("equiv", Casting::Equiv),
        ("safe", Casting::Safe),
        ("same_kind", Casting::SameKind),
        ("unsafe", Casting::Unsafe),
    ];

    let mut mismatches = Vec::new();
    for (m_name, mode) in modes {
        for (i, &from) in DType::ALL.iter().enumerate() {
            for (j, &to) in DType::ALL.iter().enumerate() {
                let expected = oracle.val(&format!("{m_name}_{i}_{j}")) == "1";
                let actual = can_cast(from, to, mode);
                if actual != expected {
                    mismatches.push(format!(
                        "can_cast({from}, {to}, {mode:?}): rust={actual} != numpy={expected}"
                    ));
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "can_cast mismatches vs NumPy 2.5.4 ({} total):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
fn test_finfo_and_iinfo_against_numpy_oracle() {
    let oracle = run_oracle(
        r#"
for dt in ['float16', 'float32', 'float64', 'complex64', 'complex128']:
    fi = np.finfo(dt)
    emit_val(f"fi_{dt}_bits", fi.bits)
    emit_val(f"fi_{dt}_eps", float(fi.eps).hex())
    emit_val(f"fi_{dt}_epsneg", float(fi.epsneg).hex())
    emit_val(f"fi_{dt}_max", float(fi.max).hex())
    emit_val(f"fi_{dt}_min", float(fi.min).hex())
    emit_val(f"fi_{dt}_snorm", float(fi.smallest_normal).hex())
    emit_val(f"fi_{dt}_ssub", float(fi.smallest_subnormal).hex())
    emit_val(f"fi_{dt}_res", float(fi.resolution).hex())
    emit_val(f"fi_{dt}_maxexp", fi.maxexp)
    emit_val(f"fi_{dt}_minexp", fi.minexp)
    emit_val(f"fi_{dt}_nmant", fi.nmant)
    emit_val(f"fi_{dt}_negep", fi.negep)
    emit_val(f"fi_{dt}_nexp", fi.nexp)
for dt in ['int8','int16','int32','int64','uint8','uint16','uint32','uint64']:
    ii = np.iinfo(dt)
    emit_val(f"ii_{dt}_bits", ii.bits)
    emit_val(f"ii_{dt}_min", ii.min)
    emit_val(f"ii_{dt}_max", ii.max)
"#,
    );

    let parse_py_hex = |s: &str| -> f64 {
        // Check exact decimal/hex via Python float.fromhex? Parse hex float manually:
        // Python hex format: [-]0x1.<hex>p[+-]<exp> or [-]0x0.<hex>p<exp>
        let neg = s.starts_with('-');
        let rest = s.trim_start_matches('-').strip_prefix("0x").unwrap();
        let (mant_str, exp_str) = rest.split_once('p').unwrap();
        let exp: i32 = exp_str.parse().unwrap();
        let (int_part, frac_part) = mant_str.split_once('.').unwrap_or((mant_str, ""));
        let int_val = u64::from_str_radix(int_part, 16).unwrap() as f64;
        let mut frac_val = 0.0f64;
        for (idx, ch) in frac_part.chars().enumerate() {
            let d = ch.to_digit(16).unwrap() as f64;
            frac_val += d * 16.0f64.powi(-((idx as i32) + 1));
        }
        let mag = (int_val + frac_val) * 2.0f64.powi(exp);
        if neg { -mag } else { mag }
    };

    for dt in [DType::F16, DType::F32, DType::F64, DType::C64, DType::C128] {
        let name = dtype_to_py_name(dt);
        let fi = dt.finfo().unwrap();
        assert_eq!(fi.bits, oracle.val(&format!("fi_{name}_bits")).parse::<usize>().unwrap(), "{name} bits");
        assert_eq!(fi.eps, parse_py_hex(oracle.val(&format!("fi_{name}_eps"))), "{name} eps");
        assert_eq!(fi.epsneg, parse_py_hex(oracle.val(&format!("fi_{name}_epsneg"))), "{name} epsneg");
        assert_eq!(fi.max, parse_py_hex(oracle.val(&format!("fi_{name}_max"))), "{name} max");
        assert_eq!(fi.min, parse_py_hex(oracle.val(&format!("fi_{name}_min"))), "{name} min");
        assert_eq!(
            fi.smallest_normal,
            parse_py_hex(oracle.val(&format!("fi_{name}_snorm"))),
            "{name} smallest_normal"
        );
        assert_eq!(
            fi.smallest_subnormal,
            parse_py_hex(oracle.val(&format!("fi_{name}_ssub"))),
            "{name} smallest_subnormal"
        );
        let py_res = parse_py_hex(oracle.val(&format!("fi_{name}_res")));
        assert!(
            (fi.resolution - py_res).abs() <= py_res * 1e-3,
            "{name} resolution {} vs {}",
            fi.resolution,
            py_res
        );
        assert_eq!(fi.maxexp, oracle.val(&format!("fi_{name}_maxexp")).parse::<i32>().unwrap());
        assert_eq!(fi.minexp, oracle.val(&format!("fi_{name}_minexp")).parse::<i32>().unwrap());
        assert_eq!(fi.nmant, oracle.val(&format!("fi_{name}_nmant")).parse::<usize>().unwrap());
        assert_eq!(fi.negep, oracle.val(&format!("fi_{name}_negep")).parse::<i32>().unwrap());
        assert_eq!(fi.nexp, oracle.val(&format!("fi_{name}_nexp")).parse::<usize>().unwrap());
    }

    for dt in [
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let name = dtype_to_py_name(dt);
        let ii = dt.iinfo().unwrap();
        assert_eq!(ii.bits, oracle.val(&format!("ii_{name}_bits")).parse::<usize>().unwrap());
        assert_eq!(ii.min, oracle.val(&format!("ii_{name}_min")).parse::<i128>().unwrap());
        assert_eq!(ii.max, oracle.val(&format!("ii_{name}_max")).parse::<u128>().unwrap());
    }

    assert!(DType::Bool.finfo().is_err());
    assert!(DType::I32.finfo().is_err());
    assert!(DType::F64.iinfo().is_err());
    assert!(DType::Bool.iinfo().is_err());
}

#[test]
fn test_dtype_metadata_and_from_name() {
    for &dt in &DType::ALL {
        assert_eq!(DType::from_name(dt.name()).unwrap(), dt);
        assert_eq!(DType::from_name(&dt.char_code().to_string()).unwrap(), dt);
    }
    assert_eq!(DType::from_name("<f8").unwrap(), DType::F64);
    assert_eq!(DType::from_name("|u1").unwrap(), DType::U8);
    assert_eq!(DType::from_name("=c16").unwrap(), DType::C128);
    assert!(DType::from_name("object").is_err());
    assert!(DType::from_name("datetime64").is_err());
}

#[test]
fn test_aligned_buffer_64_byte_all_dtypes_and_edge_lengths() {
    for len in [0usize, 1, 3, 7, 15, 17, 63, 64, 65, 255, 1023] {
        for &dt in &DType::ALL {
            let arr = DynArray::zeros(&[len], dt, Order::C).unwrap();
            assert_eq!(arr.len(), len);
            if len > 0 {
                let ptr_addr = match &arr {
                    DynArray::Bool(a) => a.storage_ptr() as usize,
                    DynArray::I8(a) => a.storage_ptr() as usize,
                    DynArray::I16(a) => a.storage_ptr() as usize,
                    DynArray::I32(a) => a.storage_ptr() as usize,
                    DynArray::I64(a) => a.storage_ptr() as usize,
                    DynArray::U8(a) => a.storage_ptr() as usize,
                    DynArray::U16(a) => a.storage_ptr() as usize,
                    DynArray::U32(a) => a.storage_ptr() as usize,
                    DynArray::U64(a) => a.storage_ptr() as usize,
                    DynArray::F16(a) => a.storage_ptr() as usize,
                    DynArray::F32(a) => a.storage_ptr() as usize,
                    DynArray::F64(a) => a.storage_ptr() as usize,
                    DynArray::C64(a) => a.storage_ptr() as usize,
                    DynArray::C128(a) => a.storage_ptr() as usize,
                };
                assert_eq!(ptr_addr % SIMD_ALIGN, 0, "unaligned {dt} len={len}");
            }
        }
        let mut spare = Vec::<f64>::with_capacity(len + 17);
        spare.extend((0..len).map(|x| x as f64));
        let arr = NdArray::from_vec(spare);
        assert_aligned(&arr, "from_vec_with_spare_cap");
        assert_eq!(arr.len(), len);
    }

    // Dropping AlignedUninit without assume_init must not leak or UB
    let mut uninit = AlignedBuffer::<u64>::new_uninit(128);
    uninit.as_mut_slice()[0].write(42);
    drop(uninit);
}

#[test]
fn test_slice_resolve_exhaustive_against_python_slice_indices() {
    let oracle = run_oracle(
        r#"
dims = [0, 1, 2, 5, 7]
bounds = [None, -10, -5, -1, 0, 1, 3, 5, 10]
steps = [-7, -2, -1, 1, 2, 7]
for d in dims:
    for start in bounds:
        for stop in bounds:
            for step in steps:
                s, e, st = slice(start, stop, step).indices(d)
                cnt = len(range(s, e, st))
                first = s if cnt > 0 else 0
                emit_val(f"{d}_{start}_{stop}_{step}", f"{first},{cnt},{st}")
"#,
    );

    let dims = [0usize, 1, 2, 5, 7];
    let bounds: [Option<isize>; 9] = [None, Some(-10), Some(-5), Some(-1), Some(0), Some(1), Some(3), Some(5), Some(10)];
    let steps = [-7isize, -2, -1, 1, 2, 7];

    for &d in &dims {
        for &start in &bounds {
            for &stop in &bounds {
                for &step in &steps {
                    let py_key = format!(
                        "{d}_{}_{}_{step}",
                        start.map_or_else(|| "None".into(), |v| v.to_string()),
                        stop.map_or_else(|| "None".into(), |v| v.to_string()),
                    );
                    let expected: Vec<isize> = oracle
                        .val(&py_key)
                        .split(',')
                        .map(|x| x.parse::<isize>().unwrap())
                        .collect();
                    let (first, count, rstep) = Slice::new(start, stop, step).resolve(d).unwrap();
                    assert_eq!(
                        (first as isize, count as isize, rstep),
                        (expected[0], expected[1], expected[2]),
                        "Slice::resolve({start:?}, {stop:?}, {step}) for dim {d}"
                    );
                }
            }
        }
    }
    assert!(Slice::new(None, None, 0).resolve(5).is_err());
}

#[test]
fn test_ndarray_slicing_negative_steps_ellipsis_newaxis_zero_d_empty() {
    let oracle = run_oracle(
        r#"
a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
emit_arr("s1", a[::-1, 1:, ::-2])
emit_arr("s2", a[..., None, -1])
emit_arr("s3", a[1, ..., ::-1])
emit_arr("s4", a[0, 1:1, :])
# Chained negative slicing
v = a[:, ::-1, ::-1]
emit_arr("s5", v[1::-1, 1:, 1::2])
# 0-D scalar slicing
sc = np.array(42, dtype=np.int32)
emit_arr("sc_newaxis", sc[np.newaxis])
emit_arr("sc_ellipsis", sc[...])
"#,
    );

    let a = NdArray::from_shape_vec(&[2, 3, 4], (0..24i32).collect()).unwrap();

    let s1 = a
        .slice(&[
            SliceArg::Slice(Slice::full().step_by(-1)),
            SliceArg::from(1..),
            SliceArg::Slice(Slice::full().step_by(-2)),
        ])
        .unwrap();
    assert!(a.shares_storage(&s1));
    assert_dyn_matches_oracle(&DynArray::I32(s1), oracle.arr("s1"), 0.0, 0.0, false, false, "s1");

    let s2 = a
        .slice(&[SliceArg::Ellipsis, SliceArg::NewAxis, SliceArg::from(-1)])
        .unwrap();
    assert!(a.shares_storage(&s2));
    assert_dyn_matches_oracle(&DynArray::I32(s2), oracle.arr("s2"), 0.0, 0.0, false, false, "s2");

    let s3 = a
        .slice(&[
            SliceArg::from(1),
            SliceArg::Ellipsis,
            SliceArg::Slice(Slice::full().step_by(-1)),
        ])
        .unwrap();
    assert!(a.shares_storage(&s3));
    assert_dyn_matches_oracle(&DynArray::I32(s3), oracle.arr("s3"), 0.0, 0.0, false, false, "s3");

    let s4 = a
        .slice(&[SliceArg::from(0), SliceArg::from(1..1), SliceArg::from(..)])
        .unwrap();
    assert!(s4.is_empty());
    assert_dyn_matches_oracle(&DynArray::I32(s4), oracle.arr("s4"), 0.0, 0.0, false, false, "s4");

    let v = a
        .slice(&[
            SliceArg::from(..),
            SliceArg::Slice(Slice::full().step_by(-1)),
            SliceArg::Slice(Slice::full().step_by(-1)),
        ])
        .unwrap();
    let s5 = v
        .slice(&[
            SliceArg::Slice(Slice::new(Some(1), None, -1)),
            SliceArg::from(1..),
            SliceArg::Slice(Slice::new(Some(1), None, 2)),
        ])
        .unwrap();
    assert!(a.shares_storage(&s5));
    assert_dyn_matches_oracle(&DynArray::I32(s5), oracle.arr("s5"), 0.0, 0.0, false, false, "s5");

    let sc = NdArray::from_scalar(42i32);
    let sc_na = sc.slice(&[SliceArg::NewAxis]).unwrap();
    assert!(sc.shares_storage(&sc_na));
    assert_dyn_matches_oracle(
        &DynArray::I32(sc_na),
        oracle.arr("sc_newaxis"),
        0.0,
        0.0,
        false,
        false,
        "sc_newaxis",
    );
    let sc_el = sc.slice(&[SliceArg::Ellipsis]).unwrap();
    assert_dyn_matches_oracle(
        &DynArray::I32(sc_el),
        oracle.arr("sc_ellipsis"),
        0.0,
        0.0,
        false,
        false,
        "sc_ellipsis",
    );

    // Error cases
    assert!(sc.slice(&[SliceArg::from(0)]).is_err());
    assert!(sc.slice(&[SliceArg::from(..)]).is_err());
    assert!(a.slice(&[SliceArg::Ellipsis, SliceArg::Ellipsis]).is_err());
    assert!(a.slice(&[SliceArg::from(0), SliceArg::from(0), SliceArg::from(0), SliceArg::from(0)]).is_err());
    let empty_dim = NdArray::<i32>::zeros(&[0, 4]).unwrap();
    assert!(empty_dim.slice(&[SliceArg::from(0)]).is_err());
}

#[test]
fn test_broadcast_shapes_and_broadcast_to_against_numpy_oracle() {
    let oracle = run_oracle(
        r#"
cases = [
    [(), (2, 3)],
    [(1, 3), (4, 1)],
    [(2, 1, 4), (3, 1)],
    [(0, 3), (1, 3)],
    [(1, 0), (4, 0)],
]
for idx, (s1, s2) in enumerate(cases):
    emit_val(f"bs_{idx}", ",".join(str(x) for x in np.broadcast_shapes(s1, s2)))

row = np.array([[10.0, 20.0, 30.0]])
emit_arr("bcast", np.broadcast_to(row, (3, 2, 3)))
empty_row = np.zeros((1, 0), dtype=np.float64)
emit_arr("bcast_empty", np.broadcast_to(empty_row, (4, 0)))
"#,
    );

    let cases: [(&[usize], &[usize]); 5] = [
        (&[], &[2, 3]),
        (&[1, 3], &[4, 1]),
        (&[2, 1, 4], &[3, 1]),
        (&[0, 3], &[1, 3]),
        (&[1, 0], &[4, 0]),
    ];
    for (idx, &(a, b)) in cases.iter().enumerate() {
        let expected: Vec<usize> = oracle
            .val(&format!("bs_{idx}"))
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(broadcast_shapes(a, b).unwrap(), expected);
    }

    assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
    assert!(broadcast_shapes(&[0, 3], &[2, 3]).is_err());

    let row = NdArray::from_shape_vec(&[1, 3], vec![10.0, 20.0, 30.0]).unwrap();
    let b = row.broadcast_to(&[3, 2, 3]).unwrap();
    assert!(row.shares_storage(&b));
    assert_eq!(b.strides(), &[0, 0, 1]);
    assert_dyn_matches_oracle(&DynArray::F64(b), oracle.arr("bcast"), 0.0, 0.0, false, false, "bcast");

    let empty_row = NdArray::<f64>::zeros(&[1, 0]).unwrap();
    let be = empty_row.broadcast_to(&[4, 0]).unwrap();
    assert_dyn_matches_oracle(
        &DynArray::F64(be),
        oracle.arr("bcast_empty"),
        0.0,
        0.0,
        false,
        false,
        "bcast_empty",
    );
    assert!(row.broadcast_to(&[3]).is_err());
}

#[test]
fn test_check_non_overlapping_and_view_mut_safety() {
    assert!(check_non_overlapping(&[2, 3], &[3, 1]));
    assert!(check_non_overlapping(&[2, 3], &[-3, -1]));
    assert!(check_non_overlapping(&[2, 3], &[2, 3]));
    assert!(check_non_overlapping(&[0, 4], &[0, 0]));
    assert!(!check_non_overlapping(&[2, 3], &[0, 1])); // broadcast 0-stride
    assert!(!check_non_overlapping(&[2, 3], &[2, 1])); // overlapping offsets
    assert!(!check_non_overlapping(&[2, 2], &[2, -2])); // self-overlapping signed strides

    // Mutating a flipped strided view via view_mut must succeed
    let mut a = NdArray::from_shape_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let mut flipped = a.flip(Some(&[1])).unwrap();
    {
        let mut vm = flipped.view_mut().unwrap();
        vm.copy_from_slice(&[30, 20, 10, 60, 50, 40]).unwrap();
    }
    assert_eq!(flipped.to_vec(), vec![30, 20, 10, 60, 50, 40]);
    // CoW: original `a` is unchanged
    assert_eq!(a.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    assert!(a.view_mut().is_ok());

    // Broadcast view must reject view_mut
    let mut b = NdArray::from_vec(vec![1, 2, 3]).broadcast_to(&[2, 3]).unwrap();
    assert!(b.view_mut().is_err());
}

#[test]
fn test_reshape_ravel_flatten_orders_and_nocopy_against_numpy_oracle() {
    let oracle = run_oracle(
        r#"
a = np.arange(12, dtype=np.int32).reshape(3, 4)
emit_arr("t", a.T)
emit_arr("t_res", a.T.reshape(2, 6))
emit_arr("rav_c", a.T.ravel('C'))
emit_arr("rav_f", a.T.ravel('F'))
emit_arr("rav_a", a.T.ravel('A'))
emit_arr("rav_k", a.T.ravel('K'))
emit_arr("flt_f", a.flatten('F'))
f_arr = np.asfortranarray(a)
emit_arr("f_rav_f", f_arr.ravel('F'))
emit_arr("f_rav_c", f_arr.ravel('C'))
"#,
    );

    let a = NdArray::from_shape_vec(&[3, 4], (0..12i32).collect()).unwrap();
    let t = a.t();
    assert!(a.shares_storage(&t));
    assert_dyn_matches_oracle(&DynArray::I32(t.clone()), oracle.arr("t"), 0.0, 0.0, false, false, "t");

    let t_res = t.reshape(&[2, 6]).unwrap();
    assert!(!t.shares_storage(&t_res)); // requires copy when transposed
    assert_dyn_matches_oracle(&DynArray::I32(t_res), oracle.arr("t_res"), 0.0, 0.0, false, false, "t_res");

    // Zero-copy reshape on flipped contiguous array and strided-leading-dim array
    let flipped = a.flip(None).unwrap();
    let f_res = flipped.reshape(&[12]).unwrap();
    assert!(flipped.shares_storage(&f_res));
    assert_eq!(f_res.to_vec(), (0..12i32).rev().collect::<Vec<_>>());

    let strided_col = a.slice(&[SliceArg::from(..), SliceArg::from(0..2)]).unwrap(); // shape [3,2], strides [4,1]
    let strided_exp = strided_col.reshape(&[3, 1, 2]).unwrap();
    assert!(strided_col.shares_storage(&strided_exp));

    // ravel orders
    assert_dyn_matches_oracle(&DynArray::I32(t.ravel(Order::C)), oracle.arr("rav_c"), 0.0, 0.0, false, false, "rav_c");
    let rav_f = t.ravel(Order::F);
    assert!(t.shares_storage(&rav_f)); // t is F-contiguous, so ravel('F') is zero-copy!
    assert_dyn_matches_oracle(&DynArray::I32(rav_f), oracle.arr("rav_f"), 0.0, 0.0, false, false, "rav_f");
    let rav_a = t.ravel(Order::A);
    assert!(t.shares_storage(&rav_a));
    assert_dyn_matches_oracle(&DynArray::I32(rav_a), oracle.arr("rav_a"), 0.0, 0.0, false, false, "rav_a");
    let rav_k = t.ravel(Order::K);
    assert!(t.shares_storage(&rav_k));
    assert_dyn_matches_oracle(&DynArray::I32(rav_k), oracle.arr("rav_k"), 0.0, 0.0, false, false, "rav_k");

    // flatten always copies
    let flt_f = a.flatten(Order::F);
    assert!(!a.shares_storage(&flt_f));
    assert_dyn_matches_oracle(&DynArray::I32(flt_f), oracle.arr("flt_f"), 0.0, 0.0, false, false, "flt_f");

    let f_arr = a.to_f_contiguous();
    assert!(f_arr.is_f_contiguous());
    let f_rav_f = f_arr.ravel(Order::F);
    assert!(f_arr.shares_storage(&f_rav_f));
    assert_dyn_matches_oracle(&DynArray::I32(f_rav_f), oracle.arr("f_rav_f"), 0.0, 0.0, false, false, "f_rav_f");
    let f_rav_c = f_arr.ravel(Order::C);
    assert!(!f_arr.shares_storage(&f_rav_c));
    assert_dyn_matches_oracle(&DynArray::I32(f_rav_c), oracle.arr("f_rav_c"), 0.0, 0.0, false, false, "f_rav_c");
}

#[test]
fn test_cow_isolation_views_iters_and_bounds_validation() {
    let mut a = NdArray::from_shape_vec(&[2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let b = a.clone();
    assert!(a.shares_storage(&b));
    assert!(!a.is_unique());
    a.as_slice_mut().unwrap()[0] = -99.0;
    assert!(!a.shares_storage(&b));
    assert!(a.is_unique());
    assert_eq!(b.get(&[0, 0]).unwrap(), 10.0);

    // Bounds checking in from_storage
    let buf = std::sync::Arc::new(AlignedBuffer::from_slice(&[1, 2, 3, 4]));
    assert!(NdArray::from_storage(buf.clone(), 0, vec![2, 2], vec![2, 1]).is_ok());
    assert!(NdArray::from_storage(buf.clone(), 0, vec![2, 2], vec![2, -1]).is_err()); // underflows 0
    assert!(NdArray::from_storage(buf.clone(), 0, vec![2, 3], vec![2, 1]).is_err()); // overflows len 4
    assert!(NdArray::from_storage(buf.clone(), 3, vec![2, 2], vec![-2, -1]).is_ok()); // offsets 3,2,1,0
    assert!(NdArray::from_storage(buf, 5, vec![0], vec![1]).is_err()); // empty offset > storage len
}

#[test]
fn test_dyn_array_astype_nan_inf_neg_zero_and_complex_bool() {
    let edges = edge_f64_values();
    let f64_arr = NdArray::from_vec(edges.clone());
    let dyn_f64 = DynArray::F64(f64_arr.clone());

    let oracle = run_oracle(&format!(
        "{}\n{}",
        common::py_assign_f64("x", &f64_arr),
        r#"
emit_arr("to_f32", x.astype(np.float32))
emit_arr("to_f16", x.astype(np.float16))
emit_arr("to_c128", x.astype(np.complex128))
emit_arr("to_bool", x.astype(np.bool_))
c = np.array([0+0j, -0.0+0j, 0+1j, 1+0j, 0+1j*np.nan, np.nan+0j], dtype=np.complex128)
emit_arr("c_to_bool", c.astype(np.bool_))
f16_nan = np.array([np.float16('nan'), np.float16(0.0), np.float16(-0.0)], dtype=np.float16)
emit_arr("f16_to_bool", f16_nan.astype(np.bool_))
"#
    ));

    // -0.0 preservation across f64 -> f32 -> f64
    let as_f32 = dyn_f64.astype(DType::F32, Casting::Unsafe).unwrap();
    assert_dyn_matches_oracle(&as_f32, oracle.arr("to_f32"), 0.0, 0.0, true, true, "f64->f32");

    let as_f16 = dyn_f64.astype(DType::F16, Casting::Unsafe).unwrap();
    assert_dyn_matches_oracle(&as_f16, oracle.arr("to_f16"), 0.0, 0.0, true, true, "f64->f16");

    let as_c128 = dyn_f64.astype(DType::C128, Casting::Safe).unwrap();
    assert_dyn_matches_oracle(&as_c128, oracle.arr("to_c128"), 0.0, 0.0, true, true, "f64->c128");

    let as_bool = dyn_f64.astype(DType::Bool, Casting::Unsafe).unwrap();
    assert_dyn_matches_oracle(&as_bool, oracle.arr("to_bool"), 0.0, 0.0, false, false, "f64->bool");

    // Complex with 0.0 + 1.0j -> Bool MUST be true!
    let c_arr = DynArray::C128(NdArray::from_vec(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, f64::NAN),
        Complex64::new(f64::NAN, 0.0),
    ]));
    let c_bool = c_arr.astype(DType::Bool, Casting::Unsafe).unwrap();
    assert_dyn_matches_oracle(&c_bool, oracle.arr("c_to_bool"), 0.0, 0.0, false, false, "c128->bool");

    // f16 NaN Scalar::is_truthy consistency check
    assert!(f16::NAN.is_truthy(), "f16::NAN.is_truthy() must match NumPy/f32/f64 (true)");

    // Signed zero round-trip through NdArray<f64>
    assert_f64_bits_eq(&f64_arr.to_vec(), &edges, "edge_f64 roundtrip");
}

#[test]
fn test_dyn_array_broadcast_and_promote_float_pair_against_numpy_oracle() {
    let a = NdArray::from_shape_vec(&[2, 1], vec![1i32, -2]).unwrap();
    let b = NdArray::from_shape_vec(&[1, 3], vec![0.5f32, -0.0, 2.5]).unwrap();
    let da = DynArray::I32(a.clone());
    let db = DynArray::F32(b.clone());

    let oracle = run_oracle(&format!(
        "{}\n{}\n{}",
        common::py_assign_dyn("a", &da),
        common::py_assign_dyn("b", &db),
        r#"
pa, pb = np.broadcast_arrays(a.astype(np.promote_types(a.dtype, b.dtype)), b.astype(np.promote_types(a.dtype, b.dtype)))
emit_arr("pa", pa)
emit_arr("pb", pb)
"#
    ));

    let (pa, pb) = DynArray::broadcast_and_promote(&da, &db).unwrap();
    assert_dyn_matches_oracle(&pa, oracle.arr("pa"), 0.0, 0.0, true, true, "pa");
    assert_dyn_matches_oracle(&pb, oracle.arr("pb"), 0.0, 0.0, true, true, "pb");

    let (fa, fb) = DynArray::promote_float_pair(
        &DynArray::I16(NdArray::from_vec(vec![1, 2])),
        &DynArray::U8(NdArray::from_vec(vec![3, 4])),
    )
    .unwrap();
    assert_eq!(fa.dtype(), DType::F64);
    assert_eq!(fb.dtype(), DType::F64);
}

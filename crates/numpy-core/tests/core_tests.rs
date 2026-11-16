use numpy_core::{empty, full, tile, Memmap, MemmapMode};
use numpy_types::{NdArray, NumPyError, Order, SliceArg};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn unique_temp_file(tag: &str) -> TempFileGuard {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "numpy_core_test_{}_{}_{}.bin",
        std::process::id(),
        tag,
        n
    ));
    TempFileGuard(p)
}

#[test]
fn test_huge_allocation_creation_and_tile_return_err() {
    assert!(matches!(
        empty::<f64>(&[1 << 61]),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));
    assert!(matches!(
        full::<f64>(&[1 << 60], 2.5),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));
    let small = NdArray::<f64>::ones(&[2]).unwrap();
    assert!(matches!(
        tile(&small, &[1 << 60]),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));
}

#[test]
fn test_memmap_aliasing_bounds_and_mutex_sync() {
    let guard = unique_temp_file("memmap_safety");
    {
        let mut m1 = Memmap::<i64>::open(&guard.0, MemmapMode::WriteNew, 0, Some(&[4]), Order::C)
            .unwrap();
        // Unique handle can borrow &mut [i64].
        {
            let sl = m1.as_slice_mut().unwrap().expect("unique aligned slice");
            sl.copy_from_slice(&[10, 20, 30, 40]);
        }

        // Cloned handle shares Arc storage -> as_slice_mut must return Ok(None) to avoid aliasing UB.
        let mut m2 = m1.clone();
        assert!(m1.as_slice_mut().unwrap().is_none());
        assert!(m2.as_slice_mut().unwrap().is_none());

        // Element set & write_from_ndarray still work cleanly via Mutex<MmapMut>.
        m2.set(&[1], 222).unwrap();
        assert_eq!(m1.get(&[1]).unwrap(), 222);

        let src = NdArray::from_vec(vec![1i64, 2, 3, 4]);
        m2.write_from_ndarray(&src).unwrap();
        assert_eq!(m1.to_ndarray().to_vec(), vec![1, 2, 3, 4]);

        // Out-of-bounds indices fail cleanly without panic/UB.
        assert!(matches!(
            m1.get(&[4]),
            Err(NumPyError::IndexOutOfBounds(_))
        ));
        assert!(matches!(
            m2.set(&[-5], 7),
            Err(NumPyError::IndexOutOfBounds(_))
        ));
        assert!(matches!(
            m1.get(&[0, 0]),
            Err(NumPyError::IndexOutOfBounds(_))
        ));

        // Concurrent writes across cloned handles stay synchronized through Mutex<MmapMut>.
        let mut h1 = m1.clone();
        let mut h2 = m2.clone();
        std::thread::scope(|s| {
            s.spawn(move || {
                for k in 0..100 {
                    h1.set(&[0], k).unwrap();
                }
            });
            s.spawn(move || {
                for k in 0..100 {
                    h2.set(&[3], k + 1000).unwrap();
                }
            });
        });
        assert_eq!(m1.get(&[0]).unwrap(), 99);
        assert_eq!(m1.get(&[3]).unwrap(), 1099);

        m1.flush().unwrap();
    }

    // Unaligned byte_offset=1 still supports unaligned get/set/write_from_ndarray cleanly.
    {
        let mut unaligned =
            Memmap::<i32>::open(&guard.0, MemmapMode::ReadWrite, 1, Some(&[3]), Order::C).unwrap();
        assert!(unaligned.as_slice().is_none());
        assert!(unaligned.as_slice_mut().unwrap().is_none());
        unaligned.set(&[0], 0x1234_5678).unwrap();
        assert_eq!(unaligned.get(&[0]).unwrap(), 0x1234_5678);
        let view = unaligned.slice(&[SliceArg::from(0isize..2isize)]).unwrap();
        assert_eq!(view.get(&[0]).unwrap(), 0x1234_5678);
    }

    // ReadOnly rejects set, as_slice_mut, and write_from_ndarray.
    {
        let mut ro =
            Memmap::<i64>::open(&guard.0, MemmapMode::ReadOnly, 0, None, Order::C).unwrap();
        assert!(matches!(
            ro.as_slice_mut(),
            Err(NumPyError::InvalidValue(_))
        ));
        assert!(matches!(ro.set(&[0], 1), Err(NumPyError::InvalidValue(_))));
        let src = NdArray::from_vec(vec![0i64; ro.len()]);
        assert!(matches!(
            ro.write_from_ndarray(&src),
            Err(NumPyError::InvalidValue(_))
        ));
    }
}

#[test]
fn test_round2_core_fixes_ub_overflow_and_semantics() {
    use num_complex::Complex64;
    use numpy_core::{
        arange, array_repr, diag, diagonal, einsum_path_shapes, eye_with_order, fill_diagonal,
        geomspace, index_advanced, insert, logspace, pad, tri, tril, triu, zeros, EinsumOptimize,
        FancyArg, InsertObj, PadMode, PadWidth, PrintOptions,
    };

    // Bug #1: parallel pad on dim_in == 0 initializes every slot safely.
    let empty_mid = zeros::<i32>(&[2, 0, 100_000]).unwrap();
    let padded_mid = pad(
        &empty_mid,
        &PadWidth::Axes(vec![(0, 0), (1, 0), (0, 0)]),
        &PadMode::Constant(7),
    )
    .unwrap();
    assert_eq!(padded_mid.shape(), &[2, 1, 100_000]);
    let sl = padded_mid.as_slice().unwrap();
    assert!(sl.iter().all(|&x| x == 7));

    // Bug #7 & #11: eye_with_order(..., Order::F) writes ones and k = isize::MIN does not panic.
    let eye_f = eye_with_order::<f64>(3, Some(4), 1, Order::F).unwrap();
    assert!(eye_f.is_f_contiguous());
    assert_eq!(
        eye_f.to_vec(),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    let eye_min = eye_with_order::<f64>(3, None, isize::MIN, Order::C).unwrap();
    assert_eq!(eye_min.to_vec(), vec![0.0; 9]);
    let d1 = NdArray::from_vec(vec![1i32, 2]);
    assert!(diag(&d1, isize::MIN).is_err());
    let d2 = NdArray::from_shape_vec(&[2, 2], vec![1i32, 2, 3, 4]).unwrap();
    assert_eq!(diag(&d2, isize::MIN).unwrap().len(), 0);
    assert_eq!(diagonal(&d2, isize::MIN, 0, 1).unwrap().len(), 0);

    // Bug #8: tri / tril / triu with extreme k = isize::MAX / isize::MIN.
    assert_eq!(tri::<i32>(2, Some(2), isize::MAX).unwrap().to_vec(), vec![1, 1, 1, 1]);
    assert_eq!(tri::<i32>(2, Some(2), isize::MIN).unwrap().to_vec(), vec![0, 0, 0, 0]);
    assert_eq!(tril(&d2, isize::MAX).unwrap().to_vec(), vec![1, 2, 3, 4]);
    assert_eq!(triu(&d2, isize::MIN).unwrap().to_vec(), vec![1, 2, 3, 4]);

    // Bug #9: complex arange rejects huge lengths instead of OOM/UB.
    assert!(arange(
        Complex64::new(0.0, 0.0),
        Complex64::new(1e200, 0.0),
        Complex64::new(1e-100, 0.0)
    )
    .is_err());

    // Bug #10: einsum_path_shapes checks shape overflow cleanly.
    assert!(einsum_path_shapes("ij,jk->ik", &[&[usize::MAX, 0], &[0, 2]], EinsumOptimize::Greedy, None).is_err());

    // Bug #13: index_advanced places separated advanced/Int indices at axis 0.
    let a4 = zeros::<i32>(&[3, 5, 6, 4]).unwrap();
    let idx = NdArray::from_vec(vec![1i64, 2]);
    let s = numpy_types::Slice::full();
    assert_eq!(
        index_advanced(&a4, &[FancyArg::Slice(s), FancyArg::IntArray(&idx), FancyArg::Slice(s), FancyArg::Int(0)])
            .unwrap()
            .shape(),
        &[2, 3, 6]
    );
    assert_eq!(
        index_advanced(&a4, &[FancyArg::Slice(s), FancyArg::IntArray(&idx), FancyArg::NewAxis, FancyArg::Int(0)])
            .unwrap()
            .shape(),
        &[2, 3, 1, 4]
    );
    assert_eq!(
        index_advanced(&a4, &[FancyArg::Slice(s), FancyArg::IntArray(&idx), FancyArg::Int(0)])
            .unwrap()
            .shape(),
        &[3, 2, 4]
    );

    // Bug #14: insert multi-index sequence and scalar-obj 2D broadcasting match NumPy.
    let z22 = zeros::<i64>(&[2, 2]).unwrap();
    let v1020 = NdArray::from_vec(vec![10i64, 20]);
    assert_eq!(
        insert(&z22, &InsertObj::Indices(&[1, 2]), &v1020, Some(0))
            .unwrap()
            .to_vec(),
        vec![0, 0, 10, 20, 0, 0, 10, 20]
    );
    let v21 = NdArray::from_shape_vec(&[2, 1], vec![10i64, 20]).unwrap();
    let ins_sc = insert(&z22, &InsertObj::Index(1), &v21, Some(1)).unwrap();
    assert_eq!(ins_sc.shape(), &[2, 4]);
    assert_eq!(ins_sc.to_vec(), vec![0, 10, 20, 0, 0, 10, 20, 0]);
    let v12 = NdArray::from_shape_vec(&[1, 2], vec![10i64, 20]).unwrap();
    let ins_sc12 = insert(&z22, &InsertObj::Index(1), &v12, Some(1)).unwrap();
    assert_eq!(ins_sc12.shape(), &[2, 3]);
    assert_eq!(ins_sc12.to_vec(), vec![0, 10, 0, 0, 20, 0]);

    // Bug #15 & #16: pad linear_ramp integer floor and empty-axis non-constant validation.
    let neg_v = NdArray::from_vec(vec![-1i64, -2]);
    let ramp = pad(
        &neg_v,
        &PadWidth::Uniform(3),
        &PadMode::LinearRampPairs(vec![(-10, 0)]),
    )
    .unwrap();
    assert_eq!(ramp.to_vec(), vec![-10, -7, -4, -1, -2, -2, -1, 0]);
    let empty00 = zeros::<f64>(&[0, 0]).unwrap();
    assert!(pad(&empty00, &PadWidth::Uniform(1), &PadMode::Edge).is_err());
    assert_eq!(
        pad(&empty00, &PadWidth::Uniform(1), &PadMode::Constant(0.0))
            .unwrap()
            .shape(),
        &[2, 2]
    );

    // Bug #17: fill_diagonal with empty val is a no-op.
    let mut ones33 = numpy_core::ones::<f64>(&[3, 3]).unwrap();
    let empty_val = NdArray::from_vec(Vec::<f64>::new());
    fill_diagonal(&mut ones33, &empty_val, false).unwrap();
    assert!(ones33.to_vec().iter().all(|&x| x == 1.0));

    // Bug #18: geomspace NaN and complex logspace with base <= 0.
    let g_nan = geomspace(f64::NAN, -10.0, 4, true).unwrap();
    assert!(g_nan.to_vec().iter().all(|x| x.is_nan()));
    let g_rnan = geomspace(1.0, f64::NAN, 4, true).unwrap();
    assert_eq!(g_rnan.to_vec()[0], 1.0);
    assert!(g_rnan.to_vec()[1..].iter().all(|x| x.is_nan()));

    let cl0 = logspace(
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        3,
        true,
        0.0,
    )
    .unwrap();
    assert_eq!(
        cl0.to_vec(),
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    let cl_neg = logspace(
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        3,
        true,
        -2.0,
    )
    .unwrap();
    assert_eq!(cl_neg.to_vec()[0], Complex64::new(1.0, 0.0));
    assert!(cl_neg.to_vec()[1].re.is_nan());
    assert_eq!(cl_neg.to_vec()[2], Complex64::new(-2.0, 0.0));

    // Bug #19: array_repr empty 1D vs 2D and default dtypes.
    let opts = PrintOptions::default();
    assert_eq!(
        array_repr(&zeros::<f64>(&[0]).unwrap(), &opts).unwrap(),
        "array([], dtype=float64)"
    );
    assert_eq!(
        array_repr(&zeros::<i64>(&[0, 3]).unwrap(), &opts).unwrap(),
        "array([], shape=(0, 3), dtype=int64)"
    );
    assert_eq!(
        array_repr(&NdArray::from_scalar(3.5f64), &opts).unwrap(),
        "array(3.5)"
    );
    assert_eq!(
        array_repr(&NdArray::from_scalar(true), &opts).unwrap(),
        "array(True)"
    );
    assert_eq!(
        array_repr(&NdArray::from_scalar(3.5f32), &opts).unwrap(),
        "array(3.5, dtype=float32)"
    );
}

#[test]
fn test_memmap_readonly_vs_writable_as_slice_and_concurrent_reads() {
    let guard = unique_temp_file("memmap_rw_ro_sync");
    {
        let mut m_rw = Memmap::<i64>::open(&guard.0, MemmapMode::WriteNew, 0, Some(&[4]), Order::C)
            .unwrap();
        let init = NdArray::from_vec(vec![10i64, 20, 30, 40]);
        m_rw.write_from_ndarray(&init).unwrap();

        // Writable Memmap must never expose unsynchronized &[T] via as_slice().
        assert!(m_rw.as_slice().is_none());

        // ReadOnly Memmap can expose &[T] via as_slice().
        let m_ro = Memmap::<i64>::open(&guard.0, MemmapMode::ReadOnly, 0, Some(&[4]), Order::C)
            .unwrap();
        assert_eq!(m_ro.as_slice(), Some(&[10i64, 20, 30, 40][..]));

        // Concurrent writer + readers calling get() and to_ndarray() must stay synchronized.
        let mut w = m_rw.clone();
        let r1 = m_rw.clone();
        let r2 = m_rw.clone();
        std::thread::scope(|s| {
            s.spawn(move || {
                for k in 0..200i64 {
                    w.set(&[0], k).unwrap();
                    w.set(&[3], k + 1000).unwrap();
                }
            });
            s.spawn(move || {
                for _ in 0..200 {
                    let v = r1.get(&[0]).unwrap();
                    assert!((0..=200).contains(&v));
                }
            });
            s.spawn(move || {
                for _ in 0..100 {
                    let arr = r2.to_ndarray();
                    assert_eq!(arr.shape(), &[4]);
                    let v = arr.to_vec();
                    assert!((0..=200).contains(&v[0]));
                    assert_eq!(v[1], 20);
                }
            });
        });

        assert_eq!(m_rw.get(&[0]).unwrap(), 199);
        assert_eq!(m_rw.get(&[3]).unwrap(), 1199);
    }

    // Large writable Memmap (> RAYON_THRESHOLD) concurrent to_ndarray() + set().
    let guard_big = unique_temp_file("memmap_rw_rayon_sync");
    {
        let len = 8192usize;
        let mut m_big =
            Memmap::<i64>::open(&guard_big.0, MemmapMode::WriteNew, 0, Some(&[len]), Order::C)
                .unwrap();
        let init = NdArray::from_vec((0..len as i64).collect());
        m_big.write_from_ndarray(&init).unwrap();
        assert!(m_big.as_slice().is_none());

        let mut w = m_big.clone();
        let r = m_big.clone();
        std::thread::scope(|s| {
            s.spawn(move || {
                for k in 0..64i64 {
                    w.set(&[10], k).unwrap();
                }
            });
            s.spawn(move || {
                for _ in 0..16 {
                    let out = r.to_ndarray();
                    assert_eq!(out.len(), len);
                }
            });
        });
        assert_eq!(m_big.get(&[10]).unwrap(), 63);
    }
}

#[test]
fn test_memmap_slice_flip_step_isize_min() {
    let guard = unique_temp_file("memmap_step_min");
    let mut mm =
        Memmap::<i32>::open(&guard.0, MemmapMode::WriteNew, 0, Some(&[1, 3]), Order::C).unwrap();
    let init = NdArray::from_shape_vec(&[1, 3], vec![1i32, 2, 3]).unwrap();
    mm.write_from_ndarray(&init).unwrap();

    let s = mm
        .slice(&[
            SliceArg::Slice(numpy_types::Slice::new(None, None, isize::MIN)),
            SliceArg::Slice(numpy_types::Slice::full()),
        ])
        .unwrap();
    assert_eq!(s.shape(), &[1, 3]);
    assert_eq!(s.to_ndarray().to_vec(), vec![1, 2, 3]);

    let f = s.flip(Some(&[0])).unwrap();
    assert_eq!(f.shape(), &[1, 3]);
    assert_eq!(f.to_ndarray().to_vec(), vec![1, 2, 3]);
}

#[test]
fn test_insert_delete_huge_slice_step_regression() {
    let arr = NdArray::from_vec(vec![10i64, 20, 30]);
    let val = NdArray::from_scalar(99i64);
    let obj_ins = numpy_core::InsertObj::Slice(numpy_types::Slice::new(Some(1), Some(2), isize::MAX));
    let ins = numpy_core::insert(&arr, &obj_ins, &val, None).unwrap();
    assert_eq!(ins.to_vec(), vec![10, 99, 20, 30]);

    let obj_del = numpy_core::DeleteObj::Slice(numpy_types::Slice::new(Some(1), Some(2), isize::MAX));
    let del = numpy_core::delete(&arr, &obj_del, None).unwrap();
    assert_eq!(del.to_vec(), vec![10, 30]);
}

#[test]
fn test_complex_logspace_base_zero_negative_real_nan() {
    use numpy_types::Complex64;

    let out = numpy_core::logspace(
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        3,
        true,
        0.0,
    )
    .unwrap();
    let v = out.to_vec();
    assert!(v[0].re.is_nan() && v[0].im.is_nan());
    assert_eq!(v[1], Complex64::new(1.0, 0.0));
    assert_eq!(v[2], Complex64::new(0.0, 0.0));

    let out2 = numpy_core::logspace(
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 1.0),
        2,
        true,
        0.0,
    )
    .unwrap();
    let v2 = out2.to_vec();
    assert!(v2[0].re.is_nan() && v2[0].im.is_nan());
    assert_eq!(v2[1], Complex64::new(0.0, 0.0));

    assert_eq!(
        numpy_core::logspace(Complex64::new(0.5, -3.0), Complex64::new(0.5, -3.0), 1, true, 0.0)
            .unwrap()
            .to_vec(),
        vec![Complex64::new(0.0, 0.0)]
    );
    assert_eq!(
        numpy_core::logspace(Complex64::new(1.0, 2.0), Complex64::new(1.0, 2.0), 1, true, 0.0)
            .unwrap()
            .to_vec(),
        vec![Complex64::new(0.0, 0.0)]
    );
    let neg_imag = numpy_core::logspace(
        Complex64::new(-1.0, 2.0),
        Complex64::new(-1.0, 2.0),
        1,
        true,
        0.0,
    )
    .unwrap()
    .to_vec();
    assert!(neg_imag[0].re.is_nan() && neg_imag[0].im.is_nan());

    assert_eq!(
        numpy_core::logspace(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), 1, true, 0.0)
            .unwrap()
            .to_vec(),
        vec![Complex64::new(1.0, 0.0)]
    );
}

#[test]
fn test_fill_diagonal_non_square_higher_dim_with_empty_val_errors() {
    let empty_val = NdArray::<i64>::zeros(&[0]).unwrap();
    let mut a3 = NdArray::<i64>::zeros(&[2, 3, 4]).unwrap();
    assert!(matches!(
        numpy_core::fill_diagonal(&mut a3, &empty_val, false),
        Err(NumPyError::ShapeMismatch(_))
    ));
    let mut a1 = NdArray::<i64>::zeros(&[4]).unwrap();
    assert!(matches!(
        numpy_core::fill_diagonal(&mut a1, &empty_val, false),
        Err(NumPyError::InvalidValue(_))
    ));
}

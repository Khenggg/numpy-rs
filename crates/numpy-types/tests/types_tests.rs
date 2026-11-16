use numpy_types::*;

#[test]
fn test_aligned_buffer_64_byte_alignment() {
    let a: NdArray<f64> = NdArray::zeros(&[100, 100]).unwrap();
    assert_eq!((a.storage_ptr() as usize) % SIMD_ALIGN, 0);
    assert_eq!(a.shape(), &[100, 100]);
    assert_eq!(a.strides(), &[100, 1]);
    assert_eq!(a.len(), 10_000);
    assert!(a.is_c_contiguous());
}

#[test]
fn test_slicing_negative_steps_and_zero_copy() {
    let a = NdArray::from_shape_vec(&[3, 4], (0..12).collect::<Vec<i32>>()).unwrap();
    let v = a.slice(&[SliceArg::from(0..2), SliceArg::Slice(Slice::full().step_by(-1))]).unwrap();
    assert!(a.shares_storage(&v));
    assert_eq!(v.shape(), &[2, 4]);
    assert_eq!(v.to_vec(), vec![3, 2, 1, 0, 7, 6, 5, 4]);

    let empty = a.slice(&[SliceArg::from(3..3), SliceArg::from(..)]).unwrap();
    assert_eq!(empty.shape(), &[0, 4]);
    assert!(empty.is_empty());
    assert_eq!(empty.to_vec(), Vec::<i32>::new());
}

#[test]
fn test_broadcast_flip_transpose_reshape() {
    let row = NdArray::from_shape_vec(&[1, 3], vec![10.0f64, 20.0, 30.0]).unwrap();
    let b = row.broadcast_to(&[4, 3]).unwrap();
    assert!(row.shares_storage(&b));
    assert_eq!(b.strides(), &[0, 1]);
    assert_eq!(b.to_vec(), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);

    let mat = NdArray::from_shape_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let t = mat.t();
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.to_vec(), vec![1, 4, 2, 5, 3, 6]);

    let f = mat.flip(Some(&[1])).unwrap();
    assert_eq!(f.to_vec(), vec![3, 2, 1, 6, 5, 4]);

    let r = mat.reshape(&[-1, 2]).unwrap();
    assert!(mat.shares_storage(&r));
    assert_eq!(r.shape(), &[3, 2]);
}

#[test]
fn test_non_overlapping_and_cow_mutation() {
    let mut a = NdArray::from_shape_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let b = a.clone();
    assert!(a.shares_storage(&b));
    a.set(&[1, 2], 99).unwrap();
    assert!(!a.shares_storage(&b));
    assert_eq!(a.get(&[-1, -1]).unwrap(), 99);
    assert_eq!(b.get(&[-1, -1]).unwrap(), 6);

    let bcast = NdArray::from_vec(vec![1, 2, 3]).broadcast_to(&[2, 3]).unwrap();
    let mut bcast_mut = bcast;
    assert!(bcast_mut.view_mut().is_err());

    let mut owned = NdArray::from_shape_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let vm = owned.view_mut().unwrap();
    let mut slices = vm.axis_iter_mut(0).unwrap();
    slices[1].copy_from_slice(&[40, 50, 60]).unwrap();
    assert_eq!(owned.to_vec(), vec![1, 2, 3, 40, 50, 60]);
}

#[test]
fn test_promote_types_and_dyn_array() {
    assert_eq!(promote_types(DType::U8, DType::I8), DType::I16);
    assert_eq!(promote_types(DType::U32, DType::I32), DType::I64);
    assert_eq!(promote_types(DType::U64, DType::I8), DType::F64);
    assert_eq!(promote_types(DType::I32, DType::F32), DType::F64);
    assert_eq!(promote_types(DType::I16, DType::F16), DType::F32);
    assert_eq!(promote_types(DType::C64, DType::F64), DType::C128);
    assert!(can_cast(DType::I16, DType::F32, Casting::Safe));
    assert!(!can_cast(DType::I32, DType::F32, Casting::Safe));
    assert!(can_cast(DType::I32, DType::F32, Casting::SameKind));

    let da = DynArray::I32(NdArray::from_vec(vec![1, 2, 3]));
    let db = DynArray::F32(NdArray::from_shape_vec(&[2, 1], vec![0.5, 1.5]).unwrap());
    let (pa, pb) = DynArray::broadcast_and_promote(&da, &db).unwrap();
    assert_eq!(pa.dtype(), DType::F64);
    assert_eq!(pb.dtype(), DType::F64);
    assert_eq!(pa.shape(), &[2, 3]);
    assert_eq!(pb.shape(), &[2, 3]);
}

#[test]
fn test_ndarray_copy_order_and_isolation() {
    let base = NdArray::from_shape_vec(&[2, 3], vec![1i32, 2, 3, 4, 5, 6]).unwrap();
    let c_copy = base.copy(Order::C);
    assert!(!base.shares_storage(&c_copy));
    assert!(c_copy.is_c_contiguous());
    assert_eq!(c_copy.to_vec(), vec![1, 2, 3, 4, 5, 6]);

    let f_copy = base.copy(Order::F);
    assert!(f_copy.is_f_contiguous());
    assert!(!f_copy.is_c_contiguous());
    assert_eq!(f_copy.to_vec(), vec![1, 2, 3, 4, 5, 6]);

    let t = base.t();
    let mut k_copy = t.copy(Order::K);
    assert!(!t.shares_storage(&k_copy));
    assert_eq!(k_copy.shape(), &[3, 2]);
    assert_eq!(k_copy.to_vec(), vec![1, 4, 2, 5, 3, 6]);
    k_copy.set(&[0, 0], 999).unwrap();
    assert_eq!(base.get(&[0, 0]).unwrap(), 1);
}

#[test]
fn test_huge_allocation_overflow_returns_err() {
    assert!(matches!(
        NdArray::<f64>::zeros(&[1 << 61]),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));
    assert!(matches!(
        NdArray::<f64>::zeros(&[1 << 60]),
        Err(NumPyError::InvalidValue(ref msg)) if msg.contains("array is too big")
    ));
    assert!(matches!(
        NdArray::<f64>::ones(&[1 << 61]),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        NdArray::<f64>::full(&[1 << 61], 1.5),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        AlignedBuffer::<f64>::try_new_zeroed(1 << 61),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        AlignedBuffer::<f64>::try_new_uninit(1 << 60),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(matches!(
        AlignedBuffer::<u8>::try_layout_for(isize::MAX as usize),
        Err(NumPyError::InvalidValue(_))
    ));
    assert!(NdArray::<f64>::zeros(&[1 << 40, 1 << 40]).is_err());

    // Confirm NumPy 2.5.4 SameKind integer rules remain intact.
    assert!(can_cast(DType::U8, DType::I8, Casting::SameKind));
    assert!(!can_cast(DType::I8, DType::U8, Casting::SameKind));
}

#[test]
fn test_round2_types_safety_slices_and_shape_guards() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Bug #3: ZST AlignedBuffer::from_vec must drop each element exactly once.
    struct ZstCounter(Arc<AtomicUsize>);
    impl Drop for ZstCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    {
        let vec: Vec<ZstCounter> = (0..5).map(|_| ZstCounter(Arc::clone(&drops))).collect();
        let buf = AlignedBuffer::from_vec(vec);
        assert_eq!(buf.len(), 5);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 5);

    // Bugs #2 & #4: shape_size, from_storage, broadcast_to, nbytes overflow guards.
    assert!(shape_size(&[usize::MAX]).is_err());
    assert!(shape_size(&[usize::MAX, 2, 0]).is_err());
    assert!(shape_size(&[1usize << 62, 2, 0]).is_err());
    let empty_f64 = NdArray::<f64>::zeros(&[0]).unwrap();
    assert!(empty_f64.broadcast_to(&[1usize << 62, 2, 0]).is_err());
    assert!(empty_f64.broadcast_to(&[1usize << 61, 0]).is_err());
    let empty_u8 = NdArray::<u8>::zeros(&[0]).unwrap();
    assert!(empty_u8.broadcast_to(&[1usize << 62, 0]).is_ok());
    assert_eq!(
        empty_u8.broadcast_to(&[1usize << 62, 0]).unwrap().nbytes(),
        0
    );
    let one_f64 = NdArray::from_scalar(1.0f64);
    assert!(one_f64.broadcast_to(&[1usize << 60]).is_err());

    let buf_small = Arc::new(AlignedBuffer::<f64>::from_vec(vec![1.0, 2.0, 3.0]));
    assert!(NdArray::from_storage(buf_small, 0, vec![usize::MAX], vec![-1]).is_err());

    // Bug #5: Slice::resolve with step = isize::MIN does not overflow.
    let s3 = NdArray::from_vec(vec![0i64, 1, 2]);
    let res = s3
        .slice(&[SliceArg::Slice(Slice::full().step_by(isize::MIN))])
        .unwrap();
    assert_eq!(res.to_vec(), vec![2]);

    // Bug #6: is_c_contiguous / is_f_contiguous handle overflowing strides without panic.
    assert!(!is_c_contiguous(&[usize::MAX, 2], &[2, 1]));
    assert!(!is_f_contiguous(&[2, usize::MAX], &[1, 2]));
    assert!(!is_c_contiguous(&[1usize << 62, 4], &[4, 1]));
    assert!(!is_f_contiguous(&[4, 1usize << 62], &[1, 4]));

    // Bug #12: RangeInclusive / RangeToInclusive ending at -1 include the last element.
    let v5 = NdArray::from_vec(vec![10i32, 20, 30, 40, 50]);
    assert_eq!(v5.slice(&[SliceArg::from(0..=-1)]).unwrap().to_vec(), vec![10, 20, 30, 40, 50]);
    assert_eq!(v5.slice(&[SliceArg::from(-3..=-1)]).unwrap().to_vec(), vec![30, 40, 50]);
    assert_eq!(v5.slice(&[SliceArg::from(..=-1)]).unwrap().to_vec(), vec![10, 20, 30, 40, 50]);
    assert_eq!(v5.slice(&[SliceArg::from(..=-2)]).unwrap().to_vec(), vec![10, 20, 30, 40]);
    assert_eq!(v5.slice(&[SliceArg::from(1..=3)]).unwrap().to_vec(), vec![20, 30, 40]);
}

#[test]
fn test_slice_flip_step_isize_min_regression() {
    let a = NdArray::from_vec(vec![1i32, 2, 3]).reshape(&[1, 3]).unwrap();
    let s = a
        .slice(&[
            SliceArg::Slice(Slice::new(None, None, isize::MIN)),
            SliceArg::Slice(Slice::full()),
        ])
        .unwrap();
    assert_eq!(s.shape(), &[1, 3]);
    assert_eq!(s.strides()[0], 0);
    assert_eq!(s.to_vec(), vec![1, 2, 3]);

    let f = s.flip(Some(&[0])).unwrap();
    assert_eq!(f.shape(), &[1, 3]);
    assert_eq!(f.to_vec(), vec![1, 2, 3]);

    let v = NdArray::from_vec(vec![10i32, 20, 30]);
    let sv = v
        .slice(&[SliceArg::Slice(Slice::new(None, None, isize::MIN))])
        .unwrap();
    assert_eq!(sv.shape(), &[1]);
    assert_eq!(sv.strides(), &[0]);
    assert_eq!(sv.to_vec(), vec![30]);
    let fv = sv.flip(Some(&[0])).unwrap();
    assert_eq!(fv.shape(), &[1]);
    assert_eq!(fv.to_vec(), vec![30]);

    let empty = v
        .slice(&[SliceArg::Slice(Slice::new(Some(0), Some(0), isize::MIN))])
        .unwrap();
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.flip(Some(&[0])).unwrap().shape(), &[0]);

    let buf = std::sync::Arc::new(AlignedBuffer::from_vec(vec![42i32]));
    let raw = NdArray::from_storage(buf, 0, vec![1], vec![isize::MIN]).unwrap();
    let raw_flipped = raw.flip(Some(&[0])).unwrap();
    assert_eq!(raw_flipped.shape(), &[1]);
    assert_eq!(raw_flipped.to_vec(), vec![42]);
}

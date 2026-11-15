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

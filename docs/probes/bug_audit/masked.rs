extern crate numpy_extra;
use numpy_extra::*;
fn main() {
    let mut a=MaskedArray::new(NdArray::from_vec(vec![1f64,2.,3.]),NdArray::from_scalar(false)).unwrap();
    a.set_mask(NdArray::from_scalar(false)).unwrap();
    assert_eq!(a.mask().shape(),a.data().shape());
    assert_eq!(a.count(None,false).unwrap().to_vec(),vec![3]);
    assert_eq!(a.mean(None,false).unwrap().data().to_vec(),vec![2.]);
    assert!(a.set_mask(NdArray::from_vec(vec![false;2])).is_err());
    assert_eq!(a.count(None,false).unwrap().to_vec(),vec![3]);
    println!("PASS B07: validated setters preserve mask/data invariant");
}

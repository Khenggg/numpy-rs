extern crate numpy_types;
#[path="../../../crates/numpy-lib/src/lanes.rs"] mod lanes;
#[path="../../../crates/numpy-lib/src/sorting.rs"] mod sorting;
use numpy_types::*;
fn main() {
    let values=[f64::NEG_INFINITY,-1.,-0.,0.,1.,f64::INFINITY,f64::NAN];
    let mut data=vec![];for &re in &values {for &im in &values {data.push(Complex64::new(re,im));}}
    let actual=sorting::argsort(&NdArray::from_vec(data),None).unwrap().to_vec();
    // Checked with NumPy 2.5.4 np.argsort(...,kind='stable').
    let expected=vec![0,1,2,3,4,5,7,8,9,10,11,12,14,21,15,22,16,17,23,24,18,25,19,26,28,29,30,31,32,33,35,36,37,38,39,40,6,13,20,27,34,41,42,43,44,45,46,47,48];
    assert_eq!(actual,expected);println!("PASS D04: complex enhanced-order mismatch disproven for 49 combinations");
}

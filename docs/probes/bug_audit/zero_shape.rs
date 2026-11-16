extern crate numpy_types;
#[path="../../../crates/numpy-lib/src/lanes.rs"] mod lanes;
#[path="../../../crates/numpy-lib/src/sorting.rs"] mod sorting;
use numpy_types::*;
fn main() {
    // Both paths test zero-element arrays: construction rejection for oversized spans
    // (B11) and zero-allocation fast paths on valid huge empty dimensions (B12).
    assert!(NdArray::<f64>::zeros(&[0, isize::MAX as usize]).is_err());
    assert!(NdArray::<f64>::zeros(&[0, (isize::MAX as usize) + 1]).is_err());

    let max_f64_cols = (isize::MAX as usize - 64) / std::mem::size_of::<f64>();
    let empty_f64 = NdArray::<f64>::zeros(&[0, max_f64_cols]).unwrap();
    let indices_f64 = sorting::argsort(&empty_f64, Some(1)).unwrap();
    assert_eq!(indices_f64.shape(), empty_f64.shape());
    assert!(indices_f64.is_empty());
    assert!(sorting::partition(&empty_f64, -1, Some(1)).unwrap().is_empty());

    let empty_u8 = NdArray::<u8>::zeros(&[0, isize::MAX as usize - 64]).unwrap();
    let indices_u8 = sorting::argsort(&empty_u8, Some(1)).unwrap();
    assert_eq!(indices_u8.shape(), empty_u8.shape());
    assert!(indices_u8.is_empty());

    println!("PASS B11 B12 (overflow checks enabled)");
}

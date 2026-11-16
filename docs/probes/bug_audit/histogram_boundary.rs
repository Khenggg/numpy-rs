extern crate numpy_types;
#[path="../../../crates/numpy-lib/src/lanes.rs"] mod lanes;
#[path="../../../crates/numpy-lib/src/histogram.rs"] mod histogram;
use numpy_types::*;
fn main() {
    let x=(-4f64)*0.8+9f64*0.2;
    assert_eq!(x.to_bits(),0xbff6666666666667);
    let histogram=histogram::histogram(&NdArray::from_vec(vec![x]),histogram::Bins::Count(5),Some((-4.,9.)),None,false).unwrap();
    assert_eq!(histogram.counts.to_vec(),vec![1.,0.,0.,0.,0.]);
    println!("PASS B13: uniform histogram boundary follows NumPy");
}

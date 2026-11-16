extern crate numpy_types;
#[path="../../../crates/numpy-lib/src/lanes.rs"] mod lanes;
#[path="../../../crates/numpy-lib/src/sorting.rs"] mod sorting;
pub use sorting::SortElement;
#[path="../../../crates/numpy-lib/src/statistics.rs"] mod statistics;
#[path="../../../crates/numpy-lib/src/histogram.rs"] mod histogram;
#[path="../../../crates/numpy-lib/src/signal.rs"] mod signal;
use numpy_types::*;
fn main() {
    let a=NdArray::from_shape_vec(&[1,3],vec![1f64,2.,3.]).unwrap();
    let covariance=statistics::cov(&a,false,1).unwrap();
    assert_eq!(covariance.shape(),&[3,3]);
    assert!(covariance.iter().all(f64::is_nan));
    let empty=NdArray::from_shape_vec(&[1,0],vec![] as Vec<f64>).unwrap();
    assert_eq!(statistics::cov(&empty,false,1).unwrap().shape(),&[0,0]);
    assert!(signal::unwrap(&NdArray::from_scalar(3.),None,0,std::f64::consts::TAU).is_err());
    assert!(signal::trapezoid(&NdArray::from_scalar(3.),None,1.,0).is_err());
    let extreme=statistics::quantile(&NdArray::from_vec(vec![i64::MIN,i64::MAX]),&[0.,0.5,1.],None,false,statistics::QuantileMethod::Linear).unwrap();
    assert_eq!(extreme.to_vec(),vec![i64::MIN as f64,i64::MAX as f64,i64::MAX as f64]);
    let complex=NdArray::from_vec(vec![Complex64::new(1.,2.),Complex64::new(1.,3.)]);
    assert!(histogram::digitize(&complex,&NdArray::from_vec(vec![0.,2.]),false).is_err());
    assert!(histogram::histogram(&complex,histogram::Bins::Count(2),None,None,false).is_err());
    let nan_edge=histogram::histogram(&NdArray::from_vec(vec![1.,2.]),histogram::Bins::Edges(&[0.,f64::NAN,3.]),None,None,false).unwrap();
    assert_eq!(nan_edge.counts.to_vec(),vec![2.,0.]);
    println!("PASS B01 B02 B03 B08 B09 B10; D01 remains disproven");
}

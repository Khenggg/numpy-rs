extern crate numpy_types;
extern crate zip;
#[path="../../crates/numpy-lib/src/lanes.rs"] mod lanes;
#[path="../../crates/numpy-lib/src/io.rs"] mod npy;
use numpy_types::*;
fn main(){
    for path in std::env::args().skip(1){
        match npy::load(&path){
            Ok(a)=>{npy::save(std::path::Path::new(&path).with_extension("rust.npy"),&a).unwrap();print!("{} {:?} {:?}",path,a.dtype(),a.shape());dispatch_dyn!(&a,x=>{for v in x.iter(){let v=v.to_complex64();print!(" {:?}:{:?}",v.re,v.im);}});println!();}
            Err(e)=>println!("{} ERROR {:?}",path,e)
        }
    }
}

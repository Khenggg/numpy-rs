extern crate numpy_lib;
extern crate numpy_types;
extern crate numpy_ufunc;
#[path="../../crates/numpy-extra/src/masked.rs"] mod masked;
use numpy_lib::NdArray;
use masked::MaskedArray;
use std::io::{self,BufRead};
fn main(){for line in io::stdin().lock().lines(){
    let line=line.unwrap();let p:Vec<_>=line.split(';').collect();
    let shape:Vec<usize>=p[0].split(',').filter(|x|!x.is_empty()).map(|x|x.parse().unwrap()).collect();
    let data:Vec<f64>=p[1].split(',').filter(|x|!x.is_empty()).map(|x|x.parse().unwrap()).collect();
    let mask:Vec<bool>=p[2].split(',').filter(|x|!x.is_empty()).map(|x|x=="1").collect();
    let a=MaskedArray::new(NdArray::from_shape_vec(&shape,data).unwrap(),NdArray::from_shape_vec(&shape,mask).unwrap()).unwrap();
    let axis=if p[4]=="none"{None}else{Some(p[4].parse().unwrap())};let ddof=p[5].parse().unwrap();
    let r=match p[3]{"mean"=>a.mean(axis,false),"var"=>a.var(axis,false,ddof),"std"=>a.std(axis,false,ddof),"min"=>a.min(axis,false),_=>a.max(axis,false)};
    match r{Ok(a)=>println!("{:?};{:?};{:?}",a.data().shape(),a.mask().to_vec(),a.data().to_vec()),Err(_)=>println!("ERROR")}
}}

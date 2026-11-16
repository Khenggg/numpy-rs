use numpy_types::{NdArray,Result,Scalar};
use serde_json::{json,Value};
use std::{hint::black_box,time::Instant};

#[derive(Clone,Debug)]
pub struct Spec{pub id:String,pub group:String,pub feature:String,pub n:usize,pub shape:Vec<usize>,pub dtype:String,pub layout:String,pub numpy_api:String,pub notes:String,pub equivalence:String}
impl Spec{
    pub fn new(group:&str,feature:&str,n:usize,shape:&[usize],dtype:&str)->Self{Self{id:format!("{group}/{feature}/{n}"),group:group.into(),feature:feature.into(),n,shape:shape.to_vec(),dtype:dtype.into(),layout:"C".into(),numpy_api:format!("np.{feature}"),notes:String::new(),equivalence:"equivalent".into()}}
    pub fn json(&self)->Value{json!({"id":self.id,"group":self.group,"feature":self.feature,"n":self.n,"shape":self.shape,"dtype":self.dtype,"layout":self.layout,"numpy_api":self.numpy_api,"notes":self.notes,"equivalence":self.equivalence})}
}
pub struct BenchCase{pub spec:Spec,pub run:Box<dyn FnMut()>,pub check:Value}
pub fn bench<R:'static>(spec:Spec,mut f:impl FnMut()->R+'static,snapshot:impl FnOnce(&R)->Value)->BenchCase{
    let value=f();let check=snapshot(&value);drop(value);
    BenchCase{spec,run:Box::new(move||{drop(black_box(f()));}),check}
}
pub fn bench_array<T:Scalar>(spec:Spec,mut f:impl FnMut()->Result<NdArray<T>>+'static)->BenchCase{
    bench(spec,move||f().expect("benchmark operation failed"),snap)
}
pub fn array(n:usize)->NdArray<f64>{NdArray::from_vec((0..n).map(|i|((i*37%1009)as f64-504.)/128.).collect())}
pub fn positive(n:usize)->NdArray<f64>{NdArray::from_vec((0..n).map(|i|1.+(i*37%1009)as f64/1024.).collect())}
pub fn matrix(r:usize,c:usize)->NdArray<f64>{array(r*c).reshape(&[r as isize,c as isize]).unwrap()}
pub fn number(x:f64)->Value{if x.is_nan(){json!("NaN")}else if x==f64::INFINITY{json!("Infinity")}else if x==f64::NEG_INFINITY{json!("-Infinity")}else{json!(x)}}
pub fn scalar(x:f64)->Value{number(x)}
pub fn snap<T:Scalar>(a:&NdArray<T>)->Value{
    let mut sr=0.;let mut si=0.;let mut norm2=0.;let mut nan=0;let mut pi=0;let mut ni=0;let mut sample=Vec::new();
    let n=a.len();let mut indices:Vec<usize>=if n==0{vec![]}else{(0..7).map(|i|i*n/7).chain([n-1]).collect()};indices.sort_unstable();indices.dedup();
    for(i,x)in a.iter().enumerate(){let z=x.to_complex64();if z.re.is_finite(){sr+=z.re;}if z.im.is_finite(){si+=z.im;}
        if z.re.is_finite()&&z.im.is_finite(){norm2+=z.re*z.re+z.im*z.im;}
        nan+=usize::from(z.re.is_nan()||z.im.is_nan());pi+=usize::from(z.re==f64::INFINITY||z.im==f64::INFINITY);ni+=usize::from(z.re==f64::NEG_INFINITY||z.im==f64::NEG_INFINITY);
        if indices.binary_search(&i).is_ok(){sample.push(json!([number(z.re),number(z.im)]));}
    }
    json!({"shape":a.shape(),"dtype":T::DTYPE.name(),"len":n,"sum_re":number(sr),"sum_im":number(si),"norm2":number(norm2),"nan":nan,"posinf":pi,"neginf":ni,"sample":sample})
}
pub fn snap_many<T:Scalar>(a:&Vec<NdArray<T>>)->Value{Value::Array(a.iter().map(snap).collect())}
pub fn measure(case:&mut BenchCase,samples:usize,target_ms:f64)->Value{
    for _ in 0..3{(case.run)();}
    let start=Instant::now();(case.run)();let one=start.elapsed().as_nanos().max(1)as f64;
    let target=(target_ms*1e6).max(100_000.);let mut inner=((target/one)as usize).clamp(1,100_000);
    if inner>4{let start=Instant::now();for _ in 0..inner{(case.run)();}let dt=start.elapsed().as_nanos().max(1)as f64;inner=((inner as f64*target/dt)as usize).clamp(1,100_000);}
    let mut times=Vec::new();for _ in 0..samples{let start=Instant::now();for _ in 0..inner{(case.run)();}times.push(start.elapsed().as_nanos()as f64/inner as f64);}
    let mut sorted=times.clone();sorted.sort_by(f64::total_cmp);let mid=sorted.len()/2;let median=if sorted.len()%2==1{sorted[mid]}else{(sorted[mid-1]+sorted[mid])/2.};
    json!({"samples_ns":times,"median_ns":median,"min_ns":sorted[0],"p95_ns":sorted[(sorted.len()-1)*95/100],"inner":inner,"samples":samples})
}

//! Extended benchmark registry. All fixtures are prepared before timed calls.
use crate::harness::{array, bench, bench_array, matrix, positive, scalar, snap, BenchCase, Spec};
use numpy_types::{Complex64, DType, DynArray, NdArray, Order, Scalar};
use numpy_lib as lib;
use numpy_extra as extra;
use numpy_fft as fft;
use numpy_random as random;
use random::{BitGenerator, CheckValid, MvHypergeomMethod, MvNormalMethod};
use serde_json::{json, Value};
use std::io::Cursor;

const LIB: &str = "sort argsort partition argpartition searchsorted lexsort top_k sort_complex sum prod min max ptp cumsum cumprod argmin argmax nanargmin nanargmax all any count_nonzero nansum nanprod nancumsum nancumprod nanmin nanmax mean nanmean var std nanvar nanstd quantile nanquantile percentile nanpercentile median nanmedian average cov corrcoef unique unique_all unique_counts unique_inverse unique_values isin union1d intersect1d setdiff1d setxor1d ediff1d histogram histogram_bin_edges histogram2d histogramdd bincount digitize convolve correlate interp diff gradient trapezoid unwrap polyval polyadd polysub polymul polydiv polyder polyint poly roots polyfit Polynomial Chebyshev Legendre Laguerre Hermite HermiteE polynomial_deriv polynomial_integ polynomial_roots series_to_polynomial poly1d write_npy read_npy save load savez savez_compressed loadz read_npz savetxt loadtxt dynamic_sum dynamic_mean dynamic_sort";
const DIST: &str = "random uniform standard_normal normal standard_exponential exponential standard_gamma gamma beta binomial negative_binomial poisson geometric hypergeometric lognormal logseries chisquare noncentral_chisquare f noncentral_f standard_t standard_cauchy laplace logistic gumbel rayleigh weibull pareto power triangular vonmises wald zipf dirichlet multinomial multivariate_normal";
const RANDOM_EXTRA: &str = "random_f32 standard_normal_f32 standard_exponential_f32 standard_gamma_f32 integers integers_inclusive integers_as_u32 bytes bytes_array random_scalar random_scalar_f32 standard_normal_scalar standard_normal_scalar_f32 standard_exponential_scalar standard_exponential_scalar_f32 choice_range choice_weighted choice_without_replacement choice permutation_range permutation permuted shuffle multinomial_array multivariate_normal_cholesky multivariate_normal_eigh multivariate_hypergeometric_marginals multivariate_hypergeometric_count normal_array uniform_array gamma_array beta_array binomial_array poisson_array triangular_array spawn par_random par_random_f32 par_standard_normal par_standard_normal_f32 par_standard_exponential par_standard_exponential_f32 par_normal par_uniform par_exponential par_gamma par_integers";
const BITGEN: &str = "PCG64 PCG64DXSM MT19937 Philox SFC64 seed_state_u32 seed_state_u64 seed_spawn pcg_advance pcg_jumped pcgdxsm_advance pcgdxsm_jumped mt_jumped philox_advance philox_jumped";
const LEGACY_EXTRA: &str = "randint random_integers tomaxint bytes choice_range choice permutation_range permutation shuffle rand randn random_sample ranf sample";
const FFT: &str = "fft ifft fft2 ifft2 fftn ifftn rfft irfft rfft2 irfft2 rfftn irfftn hfft ihfft fftshift ifftshift fftfreq rfftfreq fftfreq_f32 rfftfreq_f32";

fn extend(out: &mut Vec<Spec>, group: &str, feature: &str, n: usize, shape: Vec<usize>, dtype: &str) {
    let mut s = Spec::new(group, feature, n, &shape, dtype);
    if let Some(kind) = group.strip_prefix("extra.") {
        let prefix = match kind { "masked" => "np.ma.MaskedArray.", "strings" => "np.strings.", "datetime" => "np.datetime64 / np.busday_", _ => "np.ndarray structured dtype / " };
        s.numpy_api = format!("{prefix}{feature}");
    } else if group.starts_with("random") {
        s.numpy_api = format!("{}{}", if group == "random.legacy" { "np.random.RandomState(123)." } else { "np.random.Generator(np.random.PCG64(123))." }, feature);
    } else if group == "fft" {
        s.numpy_api = format!("np.fft.{}", feature.trim_end_matches("_f32"));
        s.notes = "Warm planner/cache; allocation included; norm=backward.".into();
    }
    if feature.starts_with("par_") { s.equivalence = "distribution_only".into(); s.notes = "Rust spawned parallel streams versus NumPy serial Generator; checks only distribution invariants, not equal streams.".into(); }
    if group == "extra.records" && feature == "field" { s.notes = "Both return an independent copy of the numeric field; NumPy field view alone is not equivalent.".into(); }
    if group == "extra.strings" { s.notes = "Variable-width UTF-8 StringDType; split uses object strings because np.strings.split has no StringDType loop.".into(); }
    out.push(s);
}

pub fn specs() -> Vec<Spec> {
    let mut out = vec![];
    for f in LIB.split_whitespace() {
        let bulk = "sort sum mean var std cumsum nanmean quantile unique histogram convolve".split_whitespace().any(|x| x == f);
        for n in if bulk { vec![1024,100000] } else { vec![1024] } {
            extend(&mut out,"lib",f,n,if ["cov","corrcoef","histogramdd"].contains(&f) { vec![8,n/8] } else { vec![n] },"float64");
        }
    }
    for (g,fs) in [
        ("extra.masked","filled compressed count add multiply sum prod min max mean var std map"),
        ("extra.strings","upper lower strip lstrip rstrip replace split find startswith endswith zfill center ljust rjust join add"),
        ("extra.datetime","parse format cast add difference timedelta_cast timedelta_add is_busday busday_count busday_offset"),
        ("extra.records","zeros from_bytes field set_field")
    ] { for f in fs.split_whitespace() { let n=if g=="extra.datetime"{1}else{1024};extend(&mut out,g,f,n,if n==1{vec![]}else{vec![n]},if g=="extra.strings"{"str"}else if g=="extra.datetime"{"datetime64[D]"}else{"float64"}); } }
    for f in DIST.split_whitespace().chain(RANDOM_EXTRA.split_whitespace()) {
        let small=f.ends_with("_scalar")||f.contains("_scalar_");
        for n in if small{vec![1]}else if ["random","standard_normal","integers","gamma","par_random"].contains(&f){vec![1024,100000]}else{vec![1024]} {
            let shape=if small{vec![]}else if ["dirichlet","multinomial","multinomial_array","multivariate_hypergeometric_marginals","multivariate_hypergeometric_count"].contains(&f){vec![n,3]}else if f.starts_with("multivariate_normal"){vec![n,2]}else{vec![n]};
            let dtype=if f.ends_with("_f32"){"float32"}else if ["integers","integers_inclusive","binomial","negative_binomial","poisson","geometric","hypergeometric","logseries","zipf"].contains(&f){"int64"}else{"float64"};
            extend(&mut out,"random",f,n,shape,dtype);
        }
    }
    for f in BITGEN.split_whitespace(){extend(&mut out,"random.bitgen",f,1024,vec![1024],"uint64");}
    for f in DIST.split_whitespace().chain(LEGACY_EXTRA.split_whitespace()){extend(&mut out,"random.legacy",f,1024,vec![1024],"float64");}
    for f in FFT.split_whitespace(){for n in if ["fft","rfft"].contains(&f){vec![1024,131072]}else{vec![1024]}{
        let shape=if ["fftn","ifftn","rfftn","irfftn"].contains(&f){vec![8,8,16]}else if ["fft2","ifft2","rfft2","irfft2"].contains(&f){vec![32,32]}else{vec![n]};
        extend(&mut out,"fft",f,n,shape,if f.ends_with("_f32"){"float32"}else{"float64"});
    }}
    out
}

pub fn make(id: &str) -> Option<BenchCase> {
    let s=specs().into_iter().find(|s|s.id==id)?;
    match s.group.as_str(){"lib"=>lib_case(s),"fft"=>fft_case(s),"random"=>rng_case(s),"random.legacy"=>legacy_case(s),"random.bitgen"=>bitgen_case(s),"extra.masked"=>masked_case(s),"extra.strings"=>strings_case(s),"extra.datetime"=>datetime_case(s),"extra.records"=>records_case(s),_=>None}
}

macro_rules! ba {($s:expr,$e:expr)=>{return Some(bench_array($s,move||$e))};}
macro_rules! bv {($s:expr,$e:expr,$snapshot:expr)=>{return Some(bench($s,move||$e,$snapshot))};}

fn linspace(n:usize,lo:f64,hi:f64)->NdArray<f64>{NdArray::from_vec((0..n).map(|i|lo+(hi-lo)*i as f64/(n-1)as f64).collect())}
fn sorted_complex(a:&NdArray<Complex64>)->NdArray<Complex64>{lib::sort(a,None).unwrap()}
fn raw_u64(a:&Vec<u64>)->Value{json!({"len":a.len(),"prefix":a.iter().take(16).copied().collect::<Vec<_>>()})}
fn raw_u32(a:&Vec<u32>)->Value{json!({"len":a.len(),"prefix":a.iter().take(16).copied().collect::<Vec<_>>()})}
fn raw_u8(a:&Vec<u8>)->Value{json!({"len":a.len(),"prefix":a.iter().take(16).copied().collect::<Vec<_>>()})}
fn invariants<T:Scalar>(a:&NdArray<T>)->Value{json!({"shape":a.shape(),"finite":a.iter().all(|x|{let z=x.to_complex64();z.re.is_finite()&&z.im.is_finite()})})}

fn lib_case(s:Spec)->Option<BenchCase>{
    let n=s.n;let f=s.feature.clone();let a=array(n);let p=positive(n).map(|x|1.+(x-1.)/1000.);
    let nan=NdArray::from_vec(a.iter().enumerate().map(|(i,x)|if i%17==0{f64::NAN}else{x}).collect());
    let ints=NdArray::from_vec((0..n).map(|i|(i*37%101)as i64).collect());
    match f.as_str(){
        "sort"=>ba!(s,lib::sort(&a,None)),"argsort"=>ba!(s,lib::argsort(&a,None)),
        "partition"|"argpartition"=>{
            let source=a.clone();let k=n/2;let indirect=f=="argpartition";
            if indirect{bv!(s,lib::argpartition(&a,k as isize,None).unwrap(),move|idx:&NdArray<i64>|{let indices=idx.to_vec();let mut ordered=indices.clone();ordered.sort();let v=source.to_vec();let values:Vec<_>=indices.iter().map(|&i|v[i as usize]).collect();json!({"kvalue":values[k],"partition_valid":values[..k].iter().all(|&x|x<=values[k])&&values[k..].iter().all(|&x|x>=values[k]),"permutation_valid":ordered.iter().enumerate().all(|(i,&v)|v==i as i64)})});}
            bv!(s,lib::partition(&a,k as isize,None).unwrap(),move|part:&NdArray<f64>|{let values=part.to_vec();let mut expected=source.to_vec();expected.sort_by(f64::total_cmp);let mut got=values.clone();got.sort_by(f64::total_cmp);json!({"kvalue":values[k],"partition_valid":values[..k].iter().all(|&x|x<=values[k])&&values[k..].iter().all(|&x|x>=values[k]),"permutation_valid":got==expected})});
        }
        "searchsorted"=>{let b=lib::sort(&a,None).unwrap();ba!(s,lib::searchsorted(&b,&a,lib::SearchSide::Right));}
        "lexsort"=>ba!(s,lib::lexsort(&[&a,&p],0)),"top_k"=>ba!(s,lib::top_k(&a,16,0)),"sort_complex"=>ba!(s,lib::sort_complex(&a)),
        "sum"=>ba!(s,lib::sum(&a,None,false)),"prod"=>ba!(s,lib::prod(&p,None,false)),"min"=>ba!(s,lib::min(&a,None,false)),"max"=>ba!(s,lib::max(&a,None,false)),"ptp"=>ba!(s,lib::ptp(&a,None,false)),
        "cumsum"=>ba!(s,lib::cumsum(&a,None)),"cumprod"=>ba!(s,lib::cumprod(&p,None)),"argmin"=>ba!(s,lib::argmin(&a,None,false)),"argmax"=>ba!(s,lib::argmax(&a,None,false)),
        "nanargmin"=>ba!(s,lib::nanargmin(&nan,None,false)),"nanargmax"=>ba!(s,lib::nanargmax(&nan,None,false)),"all"=>ba!(s,lib::all(&a,None,false)),"any"=>ba!(s,lib::any(&a,None,false)),"count_nonzero"=>ba!(s,lib::count_nonzero(&a,None,false)),
        "nansum"=>ba!(s,lib::nansum(&nan,None,false)),"nanprod"=>ba!(s,lib::nanprod(&nan,None,false)),"nancumsum"=>ba!(s,lib::nancumsum(&nan,None)),"nancumprod"=>ba!(s,lib::nancumprod(&nan,None)),"nanmin"=>ba!(s,lib::nanmin(&nan,None,false)),"nanmax"=>ba!(s,lib::nanmax(&nan,None,false)),
        "mean"=>ba!(s,lib::mean(&a,None,false)),"nanmean"=>ba!(s,lib::nanmean(&nan,None,false)),"var"=>ba!(s,lib::var(&a,None,false,0)),"std"=>ba!(s,lib::std(&a,None,false,0)),"nanvar"=>ba!(s,lib::nanvar(&nan,None,false,0)),"nanstd"=>ba!(s,lib::nanstd(&nan,None,false,0)),
        "quantile"=>ba!(s,lib::quantile(&a,&[0.1,0.5,0.9],None,false,lib::QuantileMethod::Linear)),"nanquantile"=>ba!(s,lib::nanquantile(&nan,&[0.1,0.5,0.9],None,false,lib::QuantileMethod::Linear)),"percentile"=>ba!(s,lib::percentile(&a,&[10.,50.,90.],None,false,lib::QuantileMethod::Linear)),"nanpercentile"=>ba!(s,lib::nanpercentile(&nan,&[10.,50.,90.],None,false,lib::QuantileMethod::Linear)),"median"=>ba!(s,lib::median(&a,None,false)),"nanmedian"=>ba!(s,lib::nanmedian(&nan,None,false)),
        "average"=>{let w=positive(n);ba!(s,lib::average(&a,&w,None,false));},"cov"=>{let m=matrix(8,n/8);ba!(s,lib::cov(&m,true,1));},"corrcoef"=>{let m=matrix(8,n/8);ba!(s,lib::corrcoef(&m,true));},
        "unique"=>ba!(s,lib::unique(&ints)),"unique_values"=>ba!(s,lib::unique_values(&ints)),
        "unique_all"=>bv!(s,lib::unique_all(&ints).unwrap(),|u:&lib::UniqueResult<i64>|json!({"values":snap(&u.values),"indices":snap(&u.indices),"inverse":snap(&u.inverse),"counts":snap(&u.counts)})),
        "unique_counts"=>bv!(s,lib::unique_counts(&ints).unwrap(),|u:&(NdArray<i64>,NdArray<i64>)|json!({"values":snap(&u.0),"other":snap(&u.1)})),"unique_inverse"=>bv!(s,lib::unique_inverse(&ints).unwrap(),|u:&(NdArray<i64>,NdArray<i64>)|json!({"values":snap(&u.0),"other":snap(&u.1)})),
        "isin"=>{let b=NdArray::from_vec((0..101).step_by(3).map(|x|x as i64).collect());ba!(s,lib::isin(&ints,&b,false));},
        "union1d"|"intersect1d"|"setdiff1d"|"setxor1d"=>{let b=NdArray::from_vec((50..150).map(|x|x as i64).collect());match f.as_str(){"union1d"=>ba!(s,lib::union1d(&ints,&b)),"intersect1d"=>ba!(s,lib::intersect1d(&ints,&b)),"setdiff1d"=>ba!(s,lib::setdiff1d(&ints,&b)),_=>ba!(s,lib::setxor1d(&ints,&b))}},"ediff1d"=>ba!(s,lib::ediff1d(&a)),
        "histogram"=>bv!(s,lib::histogram(&a,lib::Bins::Count(32),Some((-4.,4.)),None,false).unwrap(),|h:&lib::Histogram|json!({"counts":snap(&h.counts),"edges":snap(&h.edges)})),"histogram_bin_edges"=>ba!(s,lib::histogram_bin_edges(&a,lib::Bins::Count(32),Some((-4.,4.)))),
        "histogram2d"|"histogramdd"=>{let values=a.to_vec();let x=NdArray::from_vec(values[..n/2].to_vec());let y=NdArray::from_vec(values[n/2..].to_vec());let m=NdArray::from_shape_vec(&[n/2,2],x.iter().zip(y.iter()).flat_map(|(a,b)|[a,b]).collect()).unwrap();let check=|h:&lib::HistogramDD|json!({"counts":snap(&h.counts),"edges":h.edges.iter().map(snap).collect::<Vec<_>>()});if f=="histogram2d"{bv!(s,lib::histogram2d(&x,&y,[lib::Bins::Count(16),lib::Bins::Count(16)],None,false).unwrap(),check);}bv!(s,lib::histogramdd(&m,&[lib::Bins::Count(16),lib::Bins::Count(16)],None,false).unwrap(),check);}
        "bincount"=>ba!(s,lib::bincount(&ints,None,128)),"digitize"=>{let edges=linspace(33,-4.,4.);ba!(s,lib::digitize(&a,&edges,false));},
        "convolve"|"correlate"=>{let kernel=NdArray::from_vec((0..17).map(|i|i as f64/17.).collect());if f=="convolve"{ba!(s,lib::convolve(&a,&kernel,lib::ConvolveMode::Same));}ba!(s,lib::correlate(&a,&kernel,lib::ConvolveMode::Same));},
        "interp"=>{let xp=linspace(65,-4.,4.);let fp=xp.map(|x|x*x);ba!(s,lib::interp(&a,&xp,&fp,None,None));},"diff"=>ba!(s,lib::diff(&a,2,0)),"gradient"=>ba!(s,lib::gradient(&a,0.5,0,2)),"trapezoid"=>ba!(s,lib::trapezoid(&a,None,0.5,0)),"unwrap"=>ba!(s,lib::unwrap(&a,None,0,std::f64::consts::TAU)),
        "dynamic_sum"|"dynamic_mean"|"dynamic_sort"=>{let d=DynArray::from(a);bv!(s,match f.as_str(){"dynamic_sum"=>lib::dynamic::sum(&d,None,None,false).unwrap(),"dynamic_mean"=>lib::dynamic::mean(&d,None,false).unwrap(),_=>lib::dynamic::sort(&d,None).unwrap()},|d:&DynArray|numpy_types::dispatch_dyn!(d,arr=>snap(arr)));}
        _=>{}
    }
    let c=vec![1.,-2.,0.5,3.];let d=vec![1.,0.25,-0.5];
    match f.as_str(){
        "polyval"=>bv!(s,lib::polyval(&c,&a),|x:&NdArray<f64>|snap(x)),"polyadd"=>bv!(s,lib::polyadd(&c,&d),|x:&Vec<f64>|snap(&NdArray::from_vec(x.clone()))),"polysub"=>bv!(s,lib::polysub(&c,&d),|x:&Vec<f64>|snap(&NdArray::from_vec(x.clone()))),"polymul"=>bv!(s,lib::polymul(&c,&d),|x:&Vec<f64>|snap(&NdArray::from_vec(x.clone()))),
        "polydiv"=>bv!(s,lib::polydiv(&c,&d).unwrap(),|x:&(Vec<f64>,Vec<f64>)|json!([snap(&NdArray::from_vec(x.0.clone())),snap(&NdArray::from_vec(x.1.clone()))])),"polyder"=>bv!(s,lib::polyder(&c,2),|x:&Vec<f64>|snap(&NdArray::from_vec(x.clone()))),"polyint"=>bv!(s,lib::polyint(&c,2,&[0.]).unwrap(),|x:&Vec<f64>|snap(&NdArray::from_vec(x.clone()))),"poly"=>bv!(s,lib::poly(&[-2.,-1.,1.,2.]),|x:&Vec<f64>|snap(&NdArray::from_vec(x.clone()))),"roots"=>ba!(s,Ok(sorted_complex(&lib::roots(&[1.,0.,-1.]).unwrap()))),
        "polyfit"=>{let x=linspace(n,-1.,1.);let y=x.map(|x|2.*x*x-3.*x+1.);ba!(s,lib::polyfit(&x,&y,2,None));},
        "Polynomial"=>{let p=lib::Polynomial::new(c);bv!(s,p.evaluate(&a),|x:&NdArray<f64>|snap(x));},"Chebyshev"=>{let p=lib::Chebyshev::new(c);bv!(s,p.evaluate(&a),|x:&NdArray<f64>|snap(x));},"Legendre"=>{let p=lib::Legendre::new(c);bv!(s,p.evaluate(&a),|x:&NdArray<f64>|snap(x));},"Laguerre"=>{let p=lib::Laguerre::new(c);bv!(s,p.evaluate(&a),|x:&NdArray<f64>|snap(x));},"Hermite"=>{let p=lib::Hermite::new(c);bv!(s,p.evaluate(&a),|x:&NdArray<f64>|snap(x));},"HermiteE"=>{let p=lib::HermiteE::new(c);bv!(s,p.evaluate(&a),|x:&NdArray<f64>|snap(x));},
        "polynomial_deriv"=>{let p=lib::Polynomial::new(c);bv!(s,p.deriv(2),|p:&lib::Polynomial|snap(&NdArray::from_vec(p.coefficients().to_vec())));},"polynomial_integ"=>{let p=lib::Polynomial::new(c);bv!(s,p.integ(2,&[]).unwrap(),|p:&lib::Polynomial|snap(&NdArray::from_vec(p.coefficients().to_vec())));},"polynomial_roots"=>{let p=lib::Polynomial::new(c);ba!(s,Ok(sorted_complex(&p.roots().unwrap())));},"series_to_polynomial"=>{let p=lib::Chebyshev::new(c);bv!(s,p.to_polynomial(),|p:&lib::Polynomial|snap(&NdArray::from_vec(p.coefficients().to_vec())));},"poly1d"=>{let p=lib::poly1d(c);bv!(s,a.map(|x|p.eval(x)),|x:&NdArray<f64>|snap(x));},
        _=>io_case(s,a)
    }
}

fn dyn_snap(a:&DynArray)->Value{numpy_types::dispatch_dyn!(a,x=>snap(x))}
fn archive_snap(a:&std::collections::BTreeMap<String,DynArray>)->Value{Value::Object(a.iter().map(|(k,v)|(k.clone(),dyn_snap(v))).collect())}
fn io_case(mut s:Spec,a:NdArray<f64>)->Option<BenchCase>{
    let f=s.feature.clone();let n=s.n;let d=DynArray::from(a.clone());let path=std::env::temp_dir().join(format!("numpy-matrix-{}-{f}-{n}",std::process::id()));
    s.notes="Cached temporary-file I/O; fsync excluded. NPY memory cases include encoding/decoding, not setup.".into();
    if f=="write_npy"{bv!(s,{let mut b=vec![];lib::write_npy(&mut b,&d).unwrap();b},|raw:&Vec<u8>|dyn_snap(&lib::read_npy(&mut Cursor::new(raw)).unwrap()));}
    if f=="read_npy"{let mut raw=vec![];lib::write_npy(&mut raw,&d).unwrap();bv!(s,lib::read_npy(&mut Cursor::new(&raw)).unwrap(),dyn_snap);}
    match f.as_str(){
        "save"|"load"=>{let path=path.with_extension("npy");lib::save(&path,&d).unwrap();if f=="load"{bv!(s,lib::load(&path).unwrap(),dyn_snap);}let check=path.clone();bv!(s,lib::save(&path,&d).unwrap(),move|_:&()|dyn_snap(&lib::load(&check).unwrap()));}
        "savez"|"savez_compressed"|"loadz"|"read_npz"=>{let path=path.with_extension("npz");lib::savez(&path,&[("a",&d)]).unwrap();if f=="loadz"{bv!(s,lib::loadz(&path).unwrap(),archive_snap);}if f=="read_npz"{let bytes=std::fs::read(&path).unwrap();bv!(s,lib::read_npz(Cursor::new(&bytes)).unwrap(),archive_snap);}let check=path.clone();bv!(s,if f=="savez"{lib::savez(&path,&[("a",&d)]).unwrap()}else{lib::savez_compressed(&path,&[("a",&d)]).unwrap()},move|_:&()|archive_snap(&lib::loadz(&check).unwrap()));}
        "savetxt"|"loadtxt"=>{let path=path.with_extension("txt");lib::savetxt(&path,&a,",").unwrap();if f=="loadtxt"{ba!(s,lib::loadtxt(&path,Some(','),0));}let check=path.clone();bv!(s,lib::savetxt(&path,&a,",").unwrap(),move|_:&()|snap(&lib::loadtxt(&check,Some(','),0).unwrap()));}
        _=>None
    }
}

fn fft_case(s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();let shape=s.shape.clone();let a=array(n).reshape(&shape.iter().map(|&v|v as isize).collect::<Vec<_>>()).unwrap();let c=a.map(|x|Complex64::new(x,0.25));use fft::Norm::Backward as B;
    match f.as_str(){"fft"=>ba!(s,fft::fft(&c,None,-1,B)),"ifft"=>ba!(s,fft::ifft(&c,None,-1,B)),"fft2"=>ba!(s,fft::fft2(&c,None,None,B)),"ifft2"=>ba!(s,fft::ifft2(&c,None,None,B)),"fftn"=>ba!(s,fft::fftn(&c,None,None,B)),"ifftn"=>ba!(s,fft::ifftn(&c,None,None,B)),"rfft"=>ba!(s,fft::rfft(&a,None,-1,B)),"rfft2"=>ba!(s,fft::rfft2(&a,None,None,B)),"rfftn"=>ba!(s,fft::rfftn(&a,None,None,B)),"ihfft"=>ba!(s,fft::ihfft(&a,None,-1,B)),
    "irfft"|"hfft"=>{let spectrum=fft::rfft(&a,None,-1,B).unwrap();if f=="irfft"{ba!(s,fft::irfft(&spectrum,Some(n),-1,B));}ba!(s,fft::hfft(&spectrum,Some(n),-1,B));},"irfft2"=>{let spectrum=fft::rfft2(&a,None,None,B).unwrap();let dims=shape.iter().map(|&v|v as isize).collect::<Vec<_>>();ba!(s,fft::irfft2(&spectrum,Some(&dims),None,B));},"irfftn"=>{let spectrum=fft::rfftn(&a,None,None,B).unwrap();let dims=shape.iter().map(|&v|v as isize).collect::<Vec<_>>();ba!(s,fft::irfftn(&spectrum,Some(&dims),None,B));},"fftshift"=>ba!(s,fft::fftshift(&a,None)),"ifftshift"=>ba!(s,fft::ifftshift(&a,None)),"fftfreq"=>ba!(s,fft::fftfreq(n,0.25)),"rfftfreq"=>ba!(s,fft::rfftfreq(n,0.25)),"fftfreq_f32"=>ba!(s,fft::fftfreq_f32(n,0.25)),"rfftfreq_f32"=>ba!(s,fft::rfftfreq_f32(n,0.25)),_=>None}
}

fn masked_snap<T:Scalar>(a:&extra::MaskedArray<T>)->Value{json!({"data":snap(&a.filled(Some(T::zero())).unwrap()),"mask":snap(a.mask())})}
fn masked_case(s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();let a=extra::MaskedArray::new(array(n),NdArray::from_vec((0..n).map(|i|i%7==0).collect())).unwrap();let b=extra::MaskedArray::new(positive(n),NdArray::from_vec((0..n).map(|i|i%11==0).collect())).unwrap();
    match f.as_str(){"filled"=>ba!(s,a.filled(Some(-1.))),"compressed"=>bv!(s,a.compressed(),|x:&NdArray<f64>|snap(x)),"count"=>ba!(s,a.count(None,false)),"add"=>bv!(s,a.add(&b).unwrap(),masked_snap::<f64>),"multiply"=>bv!(s,a.multiply(&b).unwrap(),masked_snap::<f64>),"sum"=>bv!(s,a.sum(None,false).unwrap(),masked_snap::<f64>),"prod"=>bv!(s,a.prod(None,false).unwrap(),masked_snap::<f64>),"min"=>bv!(s,a.min(None,false).unwrap(),masked_snap::<f64>),"max"=>bv!(s,a.max(None,false).unwrap(),masked_snap::<f64>),"mean"=>bv!(s,a.mean(None,false).unwrap(),masked_snap::<f64>),"var"=>bv!(s,a.var(None,false,0).unwrap(),masked_snap::<f64>),"std"=>bv!(s,a.std(None,false,0).unwrap(),masked_snap::<f64>),"map"=>bv!(s,a.map(|x|x*2.).unwrap(),masked_snap::<f64>),_=>None}
}

fn strings_snap(a:&extra::StringArray)->Value{json!({"shape":a.shape(),"len":a.len(),"sample":a.data().iter().take(8).collect::<Vec<_>>()})}
fn strings_case(mut s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();let a=extra::StringArray::from_vec((0..n).map(|i|format!("  Ken {} café  ",i%17)).collect());let b=extra::StringArray::from_scalar("!");
    match f.as_str(){"upper"=>bv!(s,a.upper(),strings_snap),"lower"=>bv!(s,a.lower(),strings_snap),"strip"=>bv!(s,a.strip(None),strings_snap),"lstrip"=>bv!(s,a.lstrip(None),strings_snap),"rstrip"=>bv!(s,a.rstrip(None),strings_snap),"replace"=>bv!(s,a.replace("Ken","Pet",Some(1)),strings_snap),"split"=>bv!(s,a.split(None,Some(1)).unwrap(),|x:&Vec<Vec<String>>|json!({"len":x.len(),"sample":x.iter().take(8).collect::<Vec<_>>()})),"find"=>bv!(s,a.find("café"),|x:&NdArray<i64>|snap(x)),"startswith"=>bv!(s,a.startswith("  Ken"),|x:&NdArray<bool>|snap(x)),"endswith"=>bv!(s,a.endswith("  "),|x:&NdArray<bool>|snap(x)),"zfill"=>bv!(s,a.zfill(24),strings_snap),"center"=>bv!(s,a.center(24,'-'),strings_snap),"ljust"=>bv!(s,a.ljust(24,'-'),strings_snap),"rjust"=>bv!(s,a.rjust(24,'-'),strings_snap),"join"=>{s.notes.push_str(" Python str.join per element: NumPy 2.5 has no native StringDType join ufunc.");s.equivalence="python_fallback".into();bv!(s,a.join("-"),strings_snap);},"add"=>bv!(s,a.add(&b).unwrap(),strings_snap),_=>None}
}

fn datetime_case(s:Spec)->Option<BenchCase>{use extra::{DateTime64 as D,TimeDelta64 as T,TimeUnit as U};let f=s.feature.clone();let date=D::parse("2024-03-29",U::Day).unwrap();let other=D::parse("2024-04-08",U::Day).unwrap();let delta=T::new(3,U::Hour);
    match f.as_str(){"parse"=>bv!(s,D::parse("1969-12-31T23:59:59.123456789",U::Nanosecond).unwrap(),|v:&D|scalar(v.value as f64)),"format"=>bv!(s,date.to_string(),|v:&String|json!(v)),"cast"=>bv!(s,date.cast(U::Hour).unwrap(),|v:&D|scalar(v.value as f64)),"add"=>bv!(s,date.add(delta).unwrap(),|v:&D|scalar(v.value as f64)),"difference"=>bv!(s,other.difference(date,U::Day).unwrap(),|v:&T|scalar(v.value as f64)),"timedelta_cast"=>bv!(s,T::new(-25,U::Hour).cast(U::Day).unwrap(),|v:&T|scalar(v.value as f64)),"timedelta_add"=>bv!(s,T::new(1,U::Day).add(delta).unwrap(),|v:&T|scalar(v.value as f64)),"is_busday"=>bv!(s,extra::is_busday(date).unwrap(),|v:&bool|scalar(if *v{1.}else{0.})),"busday_count"=>bv!(s,extra::busday_count(date,other).unwrap(),|v:&i64|scalar(*v as f64)),"busday_offset"=>bv!(s,extra::busday_offset(date,100,extra::Roll::Forward).unwrap(),|v:&D|scalar(v.value as f64)),_=>None}
}

fn records_case(s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();let fields=[("tag",DType::U8),("value",DType::F64)];let mut a=extra::StructuredArray::zeros(&[n],&fields).unwrap();a.set_field("value",&array(n)).unwrap();let raw=a.as_bytes().to_vec();let v=positive(n);
    match f.as_str(){"zeros"=>bv!(s,extra::StructuredArray::zeros(&[n],&fields).unwrap(),|a:&extra::StructuredArray|json!({"len":a.shape().iter().product::<usize>(),"itemsize":a.itemsize(),"zero":a.as_bytes().iter().all(|&b|b==0)})),"from_bytes"=>bv!(s,extra::StructuredArray::from_bytes(&[n],&fields,raw.clone()).unwrap(),|a:&extra::StructuredArray|snap(&a.field::<f64>("value").unwrap())),"field"=>ba!(s,a.field::<f64>("value")),"set_field"=>{let v_snap=v.clone();bv!(s,a.set_field("value",&v).unwrap(),move|_:&()|snap(&v_snap));},_=>None}
}

// Common distribution APIs have matching signatures for Generator and RandomState.
macro_rules! common_distributions {($s:ident,$f:ident,$n:ident,$r:ident)=>{{match $f.as_str(){
    "random"=>ba!($s,$r.random(&[$n])),"uniform"=>ba!($s,$r.uniform(-2.,3.,&[$n])),"standard_normal"=>ba!($s,$r.standard_normal(&[$n])),"normal"=>ba!($s,$r.normal(1.5,2.,&[$n])),"standard_exponential"=>ba!($s,$r.standard_exponential(&[$n])),"exponential"=>ba!($s,$r.exponential(2.,&[$n])),"standard_gamma"=>ba!($s,$r.standard_gamma(2.,&[$n])),"gamma"=>ba!($s,$r.gamma(2.,3.,&[$n])),"beta"=>ba!($s,$r.beta(2.,5.,&[$n])),"binomial"=>ba!($s,$r.binomial(20,0.3,&[$n])),"negative_binomial"=>ba!($s,$r.negative_binomial(4.,0.4,&[$n])),"poisson"=>ba!($s,$r.poisson(8.,&[$n])),"geometric"=>ba!($s,$r.geometric(0.3,&[$n])),"hypergeometric"=>ba!($s,$r.hypergeometric(20,30,10,&[$n])),"lognormal"=>ba!($s,$r.lognormal(0.5,0.75,&[$n])),"logseries"=>ba!($s,$r.logseries(0.4,&[$n])),"chisquare"=>ba!($s,$r.chisquare(4.,&[$n])),"noncentral_chisquare"=>ba!($s,$r.noncentral_chisquare(4.,2.,&[$n])),"f"=>ba!($s,$r.f(5.,9.,&[$n])),"noncentral_f"=>ba!($s,$r.noncentral_f(5.,9.,2.,&[$n])),"standard_t"=>ba!($s,$r.standard_t(7.,&[$n])),"standard_cauchy"=>ba!($s,$r.standard_cauchy(&[$n])),"laplace"=>ba!($s,$r.laplace(1.,2.,&[$n])),"logistic"=>ba!($s,$r.logistic(1.,2.,&[$n])),"gumbel"=>ba!($s,$r.gumbel(1.,2.,&[$n])),"rayleigh"=>ba!($s,$r.rayleigh(2.,&[$n])),"weibull"=>ba!($s,$r.weibull(2.,&[$n])),"pareto"=>ba!($s,$r.pareto(3.,&[$n])),"power"=>ba!($s,$r.power(2.,&[$n])),"triangular"=>ba!($s,$r.triangular(-2.,0.5,3.,&[$n])),"vonmises"=>ba!($s,$r.vonmises(0.5,2.,&[$n])),"wald"=>ba!($s,$r.wald(2.,3.,&[$n])),"zipf"=>ba!($s,$r.zipf(2.,&[$n])),"dirichlet"=>ba!($s,$r.dirichlet(&[1.,2.,3.],Some(&[$n]))),"multinomial"=>ba!($s,$r.multinomial(20,&[0.2,0.3,0.5],Some(&[$n]))),"multivariate_normal"=>{let cov=NdArray::from_shape_vec(&[2,2],vec![1.,0.25,0.25,2.]).unwrap();ba!($s,$r.multivariate_normal(&[1.,2.],&cov,Some(&[$n])))},_=>{}
}}};}

fn rng_case(s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();let mut r=random::default_rng_from_seed(123u64).unwrap();common_distributions!(s,f,n,r);let a=array(n);
    match f.as_str(){
        "random_f32"=>ba!(s,r.random_f32(&[n])),"standard_normal_f32"=>ba!(s,r.standard_normal_f32(&[n])),"standard_exponential_f32"=>ba!(s,r.standard_exponential_f32(&[n])),"standard_gamma_f32"=>ba!(s,r.standard_gamma_f32(2.,&[n])),"integers"=>ba!(s,r.integers(0,100,&[n])),"integers_inclusive"=>ba!(s,r.integers_inclusive(0,100,&[n],true)),"integers_as_u32"=>ba!(s,r.integers_as::<u32>(0,Some(100),&[n],false)),
        "bytes"=>bv!(s,r.bytes(n),raw_u8),"bytes_array"=>bv!(s,r.bytes_array(n).unwrap(),|a:&NdArray<u8>|raw_u8(&a.to_vec())),
        "random_scalar"=>bv!(s,r.random_scalar(),|x:&f64|scalar(*x)),"random_scalar_f32"=>bv!(s,r.random_scalar_f32(),|x:&f32|scalar(*x as f64)),"standard_normal_scalar"=>bv!(s,r.standard_normal_scalar(),|x:&f64|scalar(*x)),"standard_normal_scalar_f32"=>bv!(s,r.standard_normal_scalar_f32(),|x:&f32|scalar(*x as f64)),"standard_exponential_scalar"=>bv!(s,r.standard_exponential_scalar(),|x:&f64|scalar(*x)),"standard_exponential_scalar_f32"=>bv!(s,r.standard_exponential_scalar_f32(),|x:&f32|scalar(*x as f64)),
        "choice_range"=>ba!(s,r.choice_range(n,&[n],true,None,true)),"choice_weighted"=>{let sum=136.;let p=(1..=16).map(|x|x as f64/sum).collect::<Vec<_>>();ba!(s,r.choice_range(16,&[n],true,Some(&p),true));},"choice_without_replacement"=>ba!(s,r.choice_range(n,&[64],false,None,true)),"choice"=>ba!(s,r.choice(&a,&[n],true,None,0,true)),"permutation_range"=>ba!(s,r.permutation_range(n)),"permutation"=>ba!(s,r.permutation(&a,0)),"permuted"=>ba!(s,r.permuted(&a,None)),
        "shuffle"=>{let mut a=a;let observed=std::rc::Rc::new(std::cell::RefCell::new(a.deep_copy(Order::C)));let check=observed.clone();bv!(s,{r.shuffle(&mut a,0).unwrap();*observed.borrow_mut()=a.clone();},move|_:&()|snap(&check.borrow()));},
        "multinomial_array"=>{let p=NdArray::from_shape_vec(&[n,3],(0..n).flat_map(|_|[0.2,0.3,0.5]).collect()).unwrap();let counts=NdArray::from_scalar(20i64);ba!(s,r.multinomial_array(&counts,&p,None));},
        "multivariate_normal_cholesky"|"multivariate_normal_eigh"=>{let cov=NdArray::from_shape_vec(&[2,2],vec![1.,0.25,0.25,2.]).unwrap();let method=if f.ends_with("cholesky"){MvNormalMethod::Cholesky}else{MvNormalMethod::Eigh};ba!(s,r.multivariate_normal_with(&[1.,2.],&cov,Some(&[n]),CheckValid::Raise,1e-8,method));},
        "multivariate_hypergeometric_marginals"|"multivariate_hypergeometric_count"=>{let method=if f.ends_with("count"){MvHypergeomMethod::Count}else{MvHypergeomMethod::Marginals};ba!(s,r.multivariate_hypergeometric(&[20,30,50],10,Some(&[n]),method));},
        "normal_array"|"uniform_array"|"gamma_array"|"beta_array"=>{let one=NdArray::full(&[n],1.).unwrap();let two=NdArray::full(&[n],2.).unwrap();match f.as_str(){"normal_array"=>ba!(s,r.normal_array(&one,&two,None)),"uniform_array"=>ba!(s,r.uniform_array(&one,&two,None)),"gamma_array"=>ba!(s,r.gamma_array(&one,&two,None)),_=>ba!(s,r.beta_array(&one,&two,None))}},"binomial_array"=>{let count=NdArray::full(&[n],20i64).unwrap();let p=NdArray::full(&[n],0.3).unwrap();ba!(s,r.binomial_array(&count,&p,None));},"poisson_array"=>{let rate=NdArray::full(&[n],8.).unwrap();ba!(s,r.poisson_array(&rate,None));},"triangular_array"=>{let l=NdArray::full(&[n],-2.).unwrap();let m=NdArray::full(&[n],0.5).unwrap();let h=NdArray::full(&[n],3.).unwrap();ba!(s,r.triangular_array(&l,&m,&h,None));},
        "spawn"=>bv!(s,r.spawn(4).unwrap(),|children:&Vec<random::Generator<random::PCG64>>|json!(children.iter().map(|c|{let mut c=c.clone();snap(&c.random(&[8]).unwrap())}).collect::<Vec<_>>())),
        "par_random"=>bv!(s,r.par_random(&[n]).unwrap(),invariants::<f64>),"par_random_f32"=>bv!(s,r.par_random_f32(&[n]).unwrap(),invariants::<f32>),"par_standard_normal"=>bv!(s,r.par_standard_normal(&[n]).unwrap(),invariants::<f64>),"par_standard_normal_f32"=>bv!(s,r.par_standard_normal_f32(&[n]).unwrap(),invariants::<f32>),"par_standard_exponential"=>bv!(s,r.par_standard_exponential(&[n]).unwrap(),invariants::<f64>),"par_standard_exponential_f32"=>bv!(s,r.par_standard_exponential_f32(&[n]).unwrap(),invariants::<f32>),"par_normal"=>bv!(s,r.par_normal(1.5,2.,&[n]).unwrap(),invariants::<f64>),"par_uniform"=>bv!(s,r.par_uniform(-2.,3.,&[n]).unwrap(),invariants::<f64>),"par_exponential"=>bv!(s,r.par_exponential(2.,&[n]).unwrap(),invariants::<f64>),"par_gamma"=>bv!(s,r.par_gamma(2.,3.,&[n]).unwrap(),invariants::<f64>),"par_integers"=>bv!(s,r.par_integers(0,100,&[n]).unwrap(),invariants::<i64>),_=>None
    }
}

fn legacy_case(s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();let mut r=random::RandomState::from_seed(123);common_distributions!(s,f,n,r);let a=array(n);
    match f.as_str(){"randint"=>ba!(s,r.randint(-20,Some(50),&[n])),"random_integers"=>ba!(s,r.random_integers(-20,Some(50),&[n])),"tomaxint"=>bv!(s,r.tomaxint(&[n]).unwrap(),|x:&NdArray<i64>|json!({"len":x.len(),"prefix":x.iter().take(16).collect::<Vec<_>>()})),"bytes"=>bv!(s,r.bytes(n),raw_u8),"rand"=>ba!(s,r.rand(&[n])),"randn"=>ba!(s,r.randn(&[n])),"random_sample"=>ba!(s,r.random_sample(&[n])),"ranf"=>ba!(s,r.ranf(&[n])),"sample"=>ba!(s,r.sample(&[n])),"choice_range"=>ba!(s,r.choice_range(n,&[n],true,None)),"choice"=>ba!(s,r.choice(&a,&[n],true,None)),"permutation_range"=>ba!(s,r.permutation_range(n)),"permutation"=>ba!(s,r.permutation(&a)),"shuffle"=>{let mut a=a;let observed=std::rc::Rc::new(std::cell::RefCell::new(a.deep_copy(Order::C)));let check=observed.clone();bv!(s,{r.shuffle(&mut a).unwrap();*observed.borrow_mut()=a.clone();},move|_:&()|snap(&check.borrow()));},_=>None}
}

fn bitgen_case(s:Spec)->Option<BenchCase>{let n=s.n;let f=s.feature.clone();
    macro_rules! raw{($ty:ty)=>{{let mut r=<$ty>::from_seed(123u64).unwrap();bv!(s,r.random_raw(n),raw_u64);}};}
    match f.as_str(){"PCG64"=>raw!(random::PCG64),"PCG64DXSM"=>raw!(random::PCG64DXSM),"MT19937"=>raw!(random::MT19937),"Philox"=>raw!(random::Philox),"SFC64"=>raw!(random::SFC64),"seed_state_u32"=>{let r=random::SeedSequence::from_seed(123u64).unwrap();bv!(s,r.generate_state_u32(n),raw_u32);},"seed_state_u64"=>{let r=random::SeedSequence::from_seed(123u64).unwrap();bv!(s,r.generate_state_u64(n),raw_u64);},"seed_spawn"=>{let mut r=random::SeedSequence::from_seed(123u64).unwrap();bv!(s,r.spawn(4).unwrap(),|v:&Vec<random::SeedSequence>|json!(v.iter().map(|c|raw_u32(&c.generate_state_u32(4))).collect::<Vec<_>>()));},"pcg_advance"=>{let mut r=random::PCG64::from_seed(123u64).unwrap();bv!(s,{r.advance(12345);r.random_raw(8)},raw_u64);},"pcgdxsm_advance"=>{let mut r=random::PCG64DXSM::from_seed(123u64).unwrap();bv!(s,{r.advance(12345);r.random_raw(8)},raw_u64);},"philox_advance"=>{let mut r=random::Philox::from_seed(123u64).unwrap();bv!(s,{r.advance_u128(12345);r.random_raw(8)},raw_u64);},"pcg_jumped"=>{let r=random::PCG64::from_seed(123u64).unwrap();bv!(s,r.jumped(1).random_raw(8),raw_u64);},"pcgdxsm_jumped"=>{let r=random::PCG64DXSM::from_seed(123u64).unwrap();bv!(s,r.jumped(1).random_raw(8),raw_u64);},"mt_jumped"=>{let r=random::MT19937::from_seed(123u64).unwrap();bv!(s,r.jumped(1).random_raw(8),raw_u64);},"philox_jumped"=>{let r=random::Philox::from_seed(123u64).unwrap();bv!(s,r.jumped(1).random_raw(8),raw_u64);},_=>None}
}

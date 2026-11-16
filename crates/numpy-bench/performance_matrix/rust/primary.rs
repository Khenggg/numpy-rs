use crate::harness::*;
use numpy_types::{NdArray,DynArray,Order,Slice,SliceArg,DType,Casting,Scalar};
use numpy_core as c;
use numpy_ufunc as u;
use numpy_linalg as l;
use serde_json::{json,Value};

const UNARY:&[&str]=&["absolute","negative","positive","reciprocal","square","sqrt","cbrt","exp","exp2","expm1","log","log2","log10","log1p","sin","cos","tan","arcsin","arccos","arctan","sinh","cosh","tanh","arcsinh","arccosh","arctanh","deg2rad","rad2deg","ceil","floor","trunc","rint","fix","sign","signbit","isfinite","isinf","isnan","isposinf","isneginf","sinc","i0","spacing","logical_not","conjugate","real","imag"];
const BINARY:&[&str]=&["add","subtract","multiply","divide","floor_divide","remainder","fmod","power","float_power","maximum","minimum","fmax","fmin","hypot","arctan2","copysign","nextafter","logaddexp","logaddexp2","heaviside","equal","not_equal","less","less_equal","greater","greater_equal","logical_and","logical_or","logical_xor"];
const INTS:&[&str]=&["bitwise_and","bitwise_or","bitwise_xor","invert","left_shift","right_shift","bitwise_count","gcd","lcm"];
const CORE:&[&str]=&["empty","zeros","ones","full","empty_like","zeros_like","ones_like","full_like","arange","linspace","logspace","geomspace","eye","identity","diag","diagflat","tri","tril","triu","vander","meshgrid","indices","from_fn","reshape","ravel","flatten","transpose","swapaxes","moveaxis","rollaxis","squeeze","expand_dims","atleast_1d","atleast_2d","atleast_3d","concatenate","stack","vstack","hstack","dstack","column_stack","block","split","array_split","hsplit","vsplit","dsplit","unstack","tile","repeat","flip","fliplr","flipud","roll","rot90","pad","resize","append","insert","delete","trim_zeros","take","take_along_axis","put","put_along_axis","putmask","compress","extract","choose","where_select","select","nonzero","argwhere","flatnonzero","diagonal","fill_diagonal","unravel_index","ravel_multi_index","boolean_mask","index_fancy","copyto","may_share_memory","shares_memory","array2string","einsum","einsum_path"];
const LINALG:&[&str]=&["dot","vdot","inner","outer","matmul","matvec","vecmat","vecdot","tensordot","kron","cross","multi_dot","matrix_power","trace","solve","tensorsolve","inv","tensorinv","pinv","lstsq","det","slogdet","matrix_rank","norm","vector_norm","matrix_norm","cond","cholesky","qr","svd","svdvals","eig","eigh","eigvals","eigvalsh"];
pub fn specs()->Vec<Spec>{
    let mut all=Vec::new();
    for n in [1024,100000]{
        for feature in ["zeros","full","slice","transpose","broadcast_to","reshape","ascontiguousarray","astype","map_fused"]{
            let mut s=Spec::new("types",feature,n,&[n/32,32],"float64");if feature=="map_fused"{s.equivalence="workload_fusion".into();s.notes="Rust one fused pass vs NumPy two operators and intermediate".into();}all.push(s);
        }
        for &feature in CORE{let mut s=Spec::new("core",feature,n,&[n/32,32],"float64");
            if feature=="empty"||feature=="empty_like"{s.equivalence="allocation_contract_differs".into();s.notes="Rust empty zero-initializes; NumPy empty is uninitialized. Metadata validation only; exclude direct speed rankings.".into();}
            if ["put","put_along_axis","putmask","fill_diagonal","copyto"].contains(&feature){s.notes="Timed work includes a fresh input copy on both sides, then mutation".into();}
            all.push(s);
        }
        for &feature in UNARY.iter().chain(BINARY).chain(INTS){let dtype=if INTS.contains(&feature){"int64"}else{"float64"};all.push(Spec::new("ufunc",feature,n,&[n],dtype));}
        for feature in ["round","ldexp","frexp","modf","divmod","nan_to_num","clip","angle","isclose","allclose","array_equal","array_equiv","reduce","accumulate","outer","at","add.f32","sin.f32","sqrt.f32","exp.f32","add.broadcast","add.strided","add.dynamic"]{all.push(Spec::new("ufunc",feature,n,&[n],if feature.ends_with("f32"){"float32"}else{"float64"}));}
    }
    for m in [32usize,128]{for &feature in LINALG{let mut s=Spec::new("linalg",feature,m,&[m,m],"float64");s.numpy_api=format!("np.linalg.{feature}");if ["qr","eig","eigh","svd"].contains(&feature){s.notes="Full decomposition timed; verification compares canonical invariant summaries, not arbitrary vector signs/bases".into();}all.push(s);}}
    for n in [1000000usize]{for feature in ["add","multiply","sqrt","sin","exp","add.f32"]{all.push(Spec::new("ufunc",feature,n,&[n],if feature.ends_with("f32"){"float32"}else{"float64"}));}}
    for n in [256usize,512]{for feature in ["matmul","matmul.f32","solve","cholesky","svd"]{all.push(Spec::new("linalg",feature,n,&[n,n],if feature.ends_with("f32"){"float32"}else{"float64"}));}}
    all
}
pub fn make(id:&str)->Option<BenchCase>{
    let s=specs().into_iter().find(|s|s.id==id)?;
    match s.group.as_str(){"types"=>types(s),"core"=>core(s),"ufunc"=>ufunc(s),"linalg"=>linalg(s),_=>None}
}
fn types(s:Spec)->Option<BenchCase>{
    let n=s.n;let a=matrix(n/32,32);let feature=s.feature.clone();
    Some(match feature.as_str(){
        "zeros"=>bench_array(s,move||NdArray::<f64>::zeros(&[n])),
        "full"=>bench_array(s,move||NdArray::<f64>::full(&[n],3.25)),
        "slice"=>bench_array(s,move||a.slice(&[SliceArg::Slice(Slice::full().step_by(2)),SliceArg::Slice(Slice::full())])),
        "transpose"=>bench_array(s,move||Ok(a.t())),
        "broadcast_to"=>{let row=positive(32).reshape(&[1,32]).unwrap();bench_array(s,move||row.broadcast_to(&[n/32,32]))},
        "reshape"=>bench_array(s,move||a.reshape(&[n as isize])),
        "ascontiguousarray"=>{let a=a.t();bench_array(s,move||Ok(a.to_c_contiguous()))},
        "astype"=>{let a=DynArray::F32(a.map(|x|x as f32));bench(s,move||a.astype(DType::F64,Casting::Unsafe).unwrap(),|r|match r{DynArray::F64(a)=>snap(a),_=>unreachable!()})},
        "map_fused"=>bench_array(s,move||Ok(a.map(|x|x*2.+1.))),
        _=>return None,
    })
}
fn core(s:Spec)->Option<BenchCase>{
    let n=s.n;let rows=n/32;let a=matrix(rows,32);let b=a.map(|x|x+1.);let v=array(n);let short=positive(32);let condition=v.map(|x|x>0.);let idx=NdArray::from_vec((0..16).map(|x|x as i64).collect());let feature=s.feature.clone();
    macro_rules! ba{($e:expr)=>{Some(bench_array(s,move||$e))};}
    macro_rules! many{($e:expr)=>{Some(bench(s,move||$e.unwrap(),snap_many))};}
    match feature.as_str(){
        "empty"=>Some(bench(s,move||c::empty::<f64>(&[n]).unwrap(),|r|json!({"shape":r.shape(),"dtype":"float64"}))),
        "empty_like"=>Some(bench(s,move||c::empty_like::<f64,f64>(&a).unwrap(),|r|json!({"shape":r.shape(),"dtype":"float64"}))),
        "zeros"=>ba!(c::zeros::<f64>(&[n])),"ones"=>ba!(c::ones::<f64>(&[n])),"full"=>ba!(c::full(&[n],3.25)),
        "zeros_like"=>ba!(c::zeros_like::<f64,f64>(&a)),"ones_like"=>ba!(c::ones_like::<f64,f64>(&a)),"full_like"=>ba!(c::full_like::<f64,f64>(&a,3.25)),
        "arange"=>ba!(c::arange(0f64,n as f64,1.)),"linspace"=>ba!(c::linspace(0f64,10.,n,true)),"logspace"=>ba!(c::logspace(0f64,2.,n,true,10.)),"geomspace"=>ba!(c::geomspace(1f64,100.,n,true)),
        "eye"=>ba!(c::eye::<f64>(rows,Some(32),0)),"identity"=>ba!(c::identity::<f64>(32)),"diag"=>ba!(c::diag(&a,0)),"diagflat"=>ba!(c::diagflat(&short,0)),"tri"=>ba!(c::tri::<f64>(rows,Some(32),0)),"tril"=>ba!(c::tril(&a,0)),"triu"=>ba!(c::triu(&a,0)),"vander"=>ba!(c::vander(&v,Some(4),true)),
        "meshgrid"=>{let y=positive(rows);many!(c::meshgrid(&[&short,&y],c::MeshgridIndexing::Ij,false,true))},
        "indices"=>ba!(c::indices(&[rows,32])),"from_fn"=>ba!(c::from_fn(&[n],|i|i[0]as f64)),
        "reshape"=>ba!(c::reshape(&a,&[n as isize])),"ravel"=>ba!(c::ravel(&a)),"flatten"=>ba!(c::flatten(&a)),"transpose"=>ba!(c::transpose(&a,None)),"swapaxes"=>ba!(c::swapaxes(&a,0,1)),"moveaxis"=>ba!(c::moveaxis(&a,&[0],&[1])),"rollaxis"=>ba!(c::rollaxis(&a,1,0)),
        "squeeze"=>{let a=a.reshape(&[1,rows as isize,32]).unwrap();ba!(c::squeeze(&a,None))},"expand_dims"=>ba!(c::expand_dims(&a,&[0])),"atleast_1d"=>ba!(c::atleast_1d(&a)),"atleast_2d"=>ba!(c::atleast_2d(&v)),"atleast_3d"=>ba!(c::atleast_3d(&a)),
        "concatenate"=>ba!(c::concatenate(&[&a,&b],Some(0))),"stack"=>ba!(c::stack(&[&a,&b],0)),"vstack"=>ba!(c::vstack(&[&a,&b])),"hstack"=>ba!(c::hstack(&[&a,&b])),"dstack"=>ba!(c::dstack(&[&a,&b])),"column_stack"=>ba!(c::column_stack(&[&v,&v])),
        "block"=>ba!(c::block(&c::BlockNode::List(vec![c::BlockNode::Array(&a),c::BlockNode::Array(&b)]))),
        "split"=>many!(c::split(&a,c::SplitSpec::Sections(4),1)),"array_split"=>many!(c::array_split(&a,c::SplitSpec::Sections(3),1)),"hsplit"=>many!(c::hsplit(&a,c::SplitSpec::Sections(4))),"vsplit"=>many!(c::vsplit(&a,c::SplitSpec::Indices(&[8]))),"dsplit"=>{let a=a.reshape(&[rows as isize,4,8]).unwrap();many!(c::dsplit(&a,c::SplitSpec::Sections(2)))},"unstack"=>many!(c::unstack(&a,1)),
        "tile"=>ba!(c::tile(&a,&[2,2])),"repeat"=>ba!(c::repeat(&v,&[2],None)),"flip"=>ba!(c::flip(&a,None)),"fliplr"=>ba!(c::fliplr(&a)),"flipud"=>ba!(c::flipud(&a)),"roll"=>ba!(c::roll(&a,7,Some(1))),"rot90"=>ba!(c::rot90(&a,1,(0,1))),"pad"=>ba!(c::pad(&a,&c::PadWidth::Uniform(2),&c::PadMode::Constant(0.))),
        "resize"=>ba!(c::resize(&v,&[n+100])),"append"=>ba!(c::append(&v,&short,None)),"insert"=>ba!(c::insert(&v,&c::InsertObj::Index(7),&NdArray::from_scalar(3.),None)),"delete"=>ba!(c::delete(&v,&c::DeleteObj::Index(7),None)),"trim_zeros"=>{let v=v.map(|x|if x.abs()<1.{0.}else{x});ba!(c::trim_zeros(&v,c::TrimDirection::Both,None))},
        "take"=>ba!(c::take(&a,&idx,Some(1),c::IndexMode::Raise)),"take_along_axis"=>{let idx=idx.reshape(&[1,16]).unwrap().broadcast_to(&[rows,16]).unwrap();ba!(c::take_along_axis(&a,&idx,Some(1)))},
        "put"=>ba!({let mut out=v.copy(Order::C);c::put(&mut out,&idx,&short,c::IndexMode::Raise)?;Ok(out)}),
        "put_along_axis"=>{let idx=idx.reshape(&[1,16]).unwrap().broadcast_to(&[rows,16]).unwrap();ba!({let mut out=a.copy(Order::C);c::put_along_axis(&mut out,&idx,&NdArray::from_scalar(3.),Some(1))?;Ok(out)})},
        "putmask"=>ba!({let mut out=v.copy(Order::C);c::putmask(&mut out,&condition,&short)?;Ok(out)}),
        "compress"=>ba!(c::compress(&condition,&v,None)),"extract"=>ba!(c::extract(&condition,&v)),"choose"=>{let selector=v.map(|x|i64::from(x>0.));ba!(c::choose(&selector,&[&v,&v.map(|x|x+1.)],c::IndexMode::Raise))},"where_select"=>{let other=v.map(|x|x+1.);ba!(c::where_select(&condition,&v,&other))},"select"=>{let other=v.map(|x|x+1.);ba!(c::select(&[&condition],&[&other],0.))},
        "nonzero"=>many!(c::nonzero(&v)),"argwhere"=>ba!(c::argwhere(&v)),"flatnonzero"=>ba!(c::flatnonzero(&v)),"diagonal"=>ba!(c::diagonal(&a,0,0,1)),
        "fill_diagonal"=>ba!({let mut out=a.copy(Order::C);c::fill_diagonal(&mut out,&NdArray::from_scalar(3.),false)?;Ok(out)}),
        "unravel_index"=>many!(c::unravel_index(&idx,&[rows,32],Order::C)),"ravel_multi_index"=>{let zero=idx.map(|_|0i64);ba!(c::ravel_multi_index(&[&zero,&idx],&[rows,32],c::IndexMode::Raise,Order::C))},"boolean_mask"=>ba!(c::boolean_mask(&v,&condition)),"index_fancy"=>ba!(c::index_fancy(&v,&[&idx])),
        "copyto"=>ba!({let mut out=a.copy(Order::C);c::copyto(&mut out,&b,None)?;Ok(out)}),
        "may_share_memory"=>Some(bench(s,move||c::may_share_memory(&a,&a.t()),|r|json!(r))),"shares_memory"=>Some(bench(s,move||c::shares_memory(&a,&a.t()),|r|json!(r))),
        "array2string"=>Some(bench(s,move||c::array2string_default(&v).unwrap(),|r|json!(r))),
        "einsum"=>{let a=matrix(32,32);let b=a.t();ba!(c::einsum_with_optimize("ij,jk->ik",&[&a,&b],c::EinsumOptimize::Greedy))},
        "einsum_path"=>Some(bench(s,move||c::einsum_path_shapes("ij,jk->ik",&[&[32,32],&[32,32]],c::EinsumOptimize::Greedy,None).unwrap(),|_|json!({"operands":2,"contractions":1}))),
        _=>None,
    }
}
fn ufunc(s:Spec)->Option<BenchCase>{
    let n=s.n;let a=positive(n);let b=NdArray::from_vec((0..n).map(|i|1.+((i*19+7)%997)as f64/1024.).collect());let signed=a.map(|x|x-1.5);let ia=a.map(|x|(x*8.)as i64);let ib=b.map(|x|(x*2.)as i64);let feature=s.feature.clone();
    macro_rules! unary{($($name:ident),*)=>{match feature.as_str(){$(stringify!($name)=>return Some(bench_array(s,move||u::$name(&a))),)*_=>{}}};}
    macro_rules! binary{($($name:ident),*)=>{match feature.as_str(){$(stringify!($name)=>return Some(bench_array(s,move||u::$name(&a,&b))),)*_=>{}}};}
    macro_rules! integer{($($name:ident),*)=>{match feature.as_str(){$(stringify!($name)=>return Some(bench_array(s,move||u::$name(&ia,&ib))),)*_=>{}}};}
    if ["arcsin","arccos","arctanh","sign","signbit"].contains(&feature.as_str()){
        return Some(match feature.as_str(){"arcsin"=>bench_array(s,move||u::arcsin(&signed)),"arccos"=>bench_array(s,move||u::arccos(&signed)),"arctanh"=>bench_array(s,move||u::arctanh(&signed)),"sign"=>bench_array(s,move||u::sign(&signed)),_=>bench_array(s,move||u::signbit(&signed))});
    }
    unary!(absolute,negative,positive,reciprocal,square,sqrt,cbrt,exp,exp2,expm1,log,log2,log10,log1p,sin,cos,tan,arctan,sinh,cosh,tanh,arcsinh,arccosh,deg2rad,rad2deg,ceil,floor,trunc,rint,fix,isfinite,isinf,isnan,isposinf,isneginf,sinc,i0,spacing,logical_not,conjugate,real,imag);
    binary!(add,subtract,multiply,divide,floor_divide,remainder,fmod,power,float_power,maximum,minimum,fmax,fmin,hypot,arctan2,copysign,nextafter,logaddexp,logaddexp2,heaviside,equal,not_equal,less,less_equal,greater,greater_equal,logical_and,logical_or,logical_xor);
    integer!(bitwise_and,bitwise_or,bitwise_xor,left_shift,right_shift,gcd,lcm);
    Some(match feature.as_str(){
        "invert"=>bench_array(s,move||u::invert(&ia)),"bitwise_count"=>bench_array(s,move||u::bitwise_count(&ia)),
        "round"=>bench_array(s,move||u::round(&a,2)),"ldexp"=>bench_array(s,move||u::ldexp(&a,&ib)),
        "frexp"=>bench(s,move||u::frexp(&a).unwrap(),|r|json!([snap(&r.0),snap(&r.1)])),"modf"=>bench(s,move||u::modf(&a).unwrap(),|r|json!([snap(&r.0),snap(&r.1)])),"divmod"=>bench(s,move||u::divmod(&a,&b).unwrap(),|r|json!([snap(&r.0),snap(&r.1)])),
        "nan_to_num"=>bench_array(s,move||u::nan_to_num(&a,None,None,None)),"clip"=>bench_array(s,move||u::clip_scalar(&a,Some(1.25),Some(1.75))),"angle"=>bench_array(s,move||u::angle(&a,false)),"isclose"=>bench_array(s,move||u::isclose(&a,&b,1e-5,1e-8,false)),"allclose"=>bench(s,move||u::allclose(&a,&b,1e-5,1e-8,false).unwrap(),|r|json!(r)),"array_equal"=>bench(s,move||u::array_equal(&a,&b,false),|r|json!(r)),"array_equiv"=>bench(s,move||u::array_equiv(&a,&b),|r|json!(r)),
        "reduce"=>bench_array(s,move||u::BinaryOp::Add.reduce_nd(&a,None,false,None)),"accumulate"=>bench_array(s,move||u::BinaryOp::Add.accumulate_nd(&a,0)),"outer"=>{let a=positive(128);bench_array(s,move||u::BinaryOp::Multiply.outer_nd(&a,&a))},
        "at"=>{let idx=NdArray::from_vec((0..n.min(1024)).map(|i|(i%128)as i64).collect());bench_array(s,move||{let mut out=a.copy(Order::C);u::BinaryOp::Add.at_nd(&mut out,&idx,&NdArray::from_scalar(1.))?;Ok(out)})},
        "add.f32"=>{let a=a.map(|x|x as f32);let b=b.map(|x|x as f32);bench_array(s,move||u::add(&a,&b))},"sin.f32"=>{let a=a.map(|x|x as f32);bench_array(s,move||u::sin(&a))},"sqrt.f32"=>{let a=a.map(|x|x as f32);bench_array(s,move||u::sqrt(&a))},"exp.f32"=>{let a=a.map(|x|x as f32);bench_array(s,move||u::exp(&a))},
        "add.broadcast"=>{let a=a.reshape(&[(n/32)as isize,32]).unwrap();let b=positive(32);bench_array(s,move||u::add(&a,&b))},"add.strided"=>{let a=positive(n*2).slice(&[SliceArg::Slice(Slice::full().step_by(2))]).unwrap();let b=b.clone();bench_array(s,move||u::add(&a,&b))},"add.dynamic"=>{let a=DynArray::F64(a);let b=DynArray::F64(b);bench(s,move||u::add(&a,&b).unwrap(),|r|match r{DynArray::F64(a)=>snap(a),_=>unreachable!()})},
        _=>return None,
    })
}
fn spd(n:usize)->NdArray<f64>{NdArray::from_shape_vec(&[n,n],(0..n*n).map(|i|{let(r,c)=(i/n,i%n);if r==c{n as f64+1.}else{(((r+c)*7)%23)as f64/32.}}).collect()).unwrap()}
fn sorted_complex<T:Scalar>(a:&NdArray<T>)->Value{let mut a:Vec<_>=a.iter().map(|x|x.to_complex64()).collect();a.sort_by(|a,b|a.re.total_cmp(&b.re).then_with(||a.im.total_cmp(&b.im)));snap(&NdArray::from_vec(a))}
fn linalg(s:Spec)->Option<BenchCase>{
    let n=s.n;let a=spd(n);let b=matrix(n,n);let v=positive(n);let feature=s.feature.clone();
    macro_rules! ba{($e:expr)=>{Some(bench_array(s,move||$e))};}
    match feature.as_str(){
        "matmul"=>ba!(l::matmul(&a,&b)),"matmul.f32"=>{let a=a.map(|x|x as f32);let b=b.map(|x|x as f32);ba!(l::matmul(&a,&b))},"dot"=>ba!(l::dot(&a,&b)),"vdot"=>ba!(l::vdot(&a,&b)),"inner"=>ba!(l::inner(&a,&b)),"outer"=>ba!(l::outer(&v,&v)),"matvec"=>ba!(l::matvec(&a,&v)),"vecmat"=>ba!(l::vecmat(&v,&a)),"vecdot"=>ba!(l::vecdot(&a,&b,-1)),"tensordot"=>ba!(l::tensordot(&a,&b,1usize)),"kron"=>{let tiny=matrix(2,2);ba!(l::kron(&a,&tiny))},"cross"=>{let a=matrix(n,3);let b=a.map(|x|x+1.);ba!(l::cross(&a,&b,-1))},"multi_dot"=>ba!(l::multi_dot(&[&a,&b,&a])),"matrix_power"=>ba!(l::matrix_power(&a,3)),"trace"=>ba!(l::trace(&a,0)),
        "solve"=>ba!(l::solve(&a,&v)),"tensorsolve"=>ba!(l::tensorsolve(&a,&v,None)),"inv"=>ba!(l::inv(&a)),"tensorinv"=>ba!(l::tensorinv(&a,1)),"pinv"=>ba!(l::pinv(&a,Some(1e-15),false)),"lstsq"=>Some(bench(s,move||l::lstsq(&a,&v,None).unwrap(),|r|json!([snap(&r.x),snap(&r.rank),snap(&r.s)]))),
        "det"=>ba!(l::det(&a)),"slogdet"=>Some(bench(s,move||l::slogdet(&a).unwrap(),|r|json!([snap(&r.sign),snap(&r.logabsdet)]))),"matrix_rank"=>ba!(l::matrix_rank(&a,None,None,false)),"norm"=>ba!(l::norm(&a,l::NormOrd::Default,None,false)),"vector_norm"=>ba!(l::vector_norm(&v,2.,None,false)),"matrix_norm"=>ba!(l::matrix_norm(&a,l::NormOrd::Fro,false)),"cond"=>ba!(l::cond(&a,l::NormOrd::Default)),"cholesky"=>ba!(l::cholesky(&a,false)),
        "qr"=>Some(bench(s,move||l::qr(&a,l::QrMode::Reduced).unwrap(),|r|match r{l::QrOutput::QR{q,r}=>json!({"q_shape":q.shape(),"r_abs":snap(&r.map(f64::abs))}),_=>unreachable!()})),
        "svd"=>Some(bench(s,move||l::svd(&a,false,true,false).unwrap(),|r|snap(&r.s))),"svdvals"=>ba!(l::svdvals(&a)),
        "eig"=>Some(bench(s,move||l::eig(&a).unwrap(),|r|sorted_complex(&r.eigenvalues))),"eigh"=>Some(bench(s,move||l::eigh(&a,l::Uplo::Lower).unwrap(),|r|snap(&r.eigenvalues))),"eigvals"=>Some(bench(s,move||l::eigvals(&a).unwrap(),sorted_complex)),"eigvalsh"=>ba!(l::eigvalsh(&a,l::Uplo::Lower)),_=>None,
    }
}

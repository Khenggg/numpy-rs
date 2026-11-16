mod harness;
mod primary;
mod extended;
use std::io::{self,BufRead,Write};
use serde_json::{json,Value};
fn main(){
    let _=rayon::ThreadPoolBuilder::new().build_global();
    if std::env::args().any(|a|a=="--list"){
        let mut specs=primary::specs();specs.extend(extended::specs());println!("{}",Value::Array(specs.iter().map(|s|s.json()).collect()));return;
    }
    for line in io::stdin().lock().lines(){let Ok(line)=line else{break;};
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
            let request:Value=serde_json::from_str(&line).expect("invalid benchmark request");let id=request["id"].as_str().expect("missing id");
            let mut case=primary::make(id).or_else(||extended::make(id)).expect("unknown case");
            let stats=if request["verify_only"].as_bool().unwrap_or(false){Value::Null}else{harness::measure(&mut case,request["samples"].as_u64().unwrap_or(11)as usize,request["sample_ms"].as_f64().unwrap_or(15.))};
            json!({"id":id,"spec":case.spec.json(),"stats":stats,"check":case.check})
        }));
        match result{Ok(value)=>println!("{value}"),Err(error)=>{let message=error.downcast_ref::<String>().cloned().or_else(||error.downcast_ref::<&str>().map(|s|s.to_string())).unwrap_or_else(||"panic".into());println!("{}",json!({"error":message}));}}
        io::stdout().flush().unwrap();
    }
}

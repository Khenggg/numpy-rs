extern crate numpy_types;
#[path="../../crates/numpy-extra/src/datetime.rs"] mod datetime;
use datetime::*;
use std::io::{self,BufRead};
fn main(){
    for line in io::stdin().lock().lines(){
        let line=line.unwrap();let p:Vec<_>=line.split(';').collect();
        let result=(||->numpy_types::Result<String>{
            match p[0]{
                "offset"=>{
                    let roll=match p[3]{"raise"=>Roll::Raise,"nat"=>Roll::NaT,"forward"=>Roll::Forward,"backward"=>Roll::Backward,"modifiedfollowing"=>Roll::ModifiedFollowing,_=>Roll::ModifiedPreceding};
                    let cal=if p.get(4)==Some(&"holiday"){
                        BusinessCalendar::new([true,true,true,true,true,false,false],&[DateTime64::parse("2024-03-29",TimeUnit::Day)?,DateTime64::parse("2024-04-01",TimeUnit::Day)?,DateTime64::parse("1970-01-01",TimeUnit::Day)?])?
                    }else{BusinessCalendar::default()};
                    Ok(cal.busday_offset(DateTime64::parse(p[1],TimeUnit::Day)?,p[2].parse().unwrap(),roll)?.to_string())
                }
                "count"=>Ok(busday_count(DateTime64::parse(p[1],TimeUnit::Day)?,DateTime64::parse(p[2],TimeUnit::Day)?)?.to_string()),
                "cast"=>Ok(DateTime64::parse(p[1],p[2].parse()?)?.cast(p[3].parse()?)?.to_string()),
                "durationadd"=>{
                    let d=TimeDelta64::new(p[1].parse().unwrap(),p[2].parse()?).add(TimeDelta64::new(p[3].parse().unwrap(),p[4].parse()?))?;
                    Ok(format!("{};{:?}",d.value,d.unit))
                }
                "difference"=>Ok(DateTime64::parse(p[1],p[2].parse()?)?.difference(DateTime64::parse(p[3],p[4].parse()?)?,p[5].parse()?)?.value.to_string()),
                "equaldate"=>Ok((DateTime64::new(p[1].parse().unwrap(),p[2].parse()?)==DateTime64::new(p[3].parse().unwrap(),p[4].parse()?)).to_string()),
                "equaldelta"=>Ok((TimeDelta64::new(p[1].parse().unwrap(),p[2].parse()?)==TimeDelta64::new(p[3].parse().unwrap(),p[4].parse()?)).to_string()),
                _=>unreachable!()
            }
        })();
        println!("{}",result.unwrap_or_else(|_|"ERROR".into()));
    }
}

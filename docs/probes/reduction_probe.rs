use std::hint::black_box;
use std::time::{Duration, Instant};

#[inline(never)]
fn columns(a: &[f64], outer: usize, k: usize, inner: usize) -> Vec<f64> {
    let mut out = vec![0.0; outer * inner];
    for o in 0..outer {
        for j in 0..inner {
            let mut acc = 0.0;
            for r in 0..k { acc += a[(o*k+r)*inner+j]; }
            out[o*inner+j] = acc;
        }
    }
    out
}

#[inline(never)]
fn rows(a: &[f64], outer: usize, k: usize, inner: usize) -> Vec<f64> {
    let mut out = vec![0.0; outer * inner];
    for (o, dst) in out.chunks_mut(inner).enumerate() {
        for r in 0..k {
            let src = &a[(o*k+r)*inner..(o*k+r+1)*inner];
            for (y,x) in dst.iter_mut().zip(src) { *y += *x; }
        }
    }
    out
}

fn time(mut f: impl FnMut() -> Vec<f64>) -> f64 {
    let mut samples=Vec::new();
    for _ in 0..7 {
        let start=Instant::now(); let mut n=0;
        while start.elapsed() < Duration::from_millis(50) { black_box(f()); n+=1; }
        samples.push(start.elapsed().as_secs_f64()*1e6/n as f64);
    }
    samples.sort_by(f64::total_cmp); samples[3]
}

fn main() {
    for (outer,k,inner) in [(1,64,64),(1,1024,1024),(32,64,512),(1,1_000_000,1),(1024,1024,1)] {
        let a:Vec<f64>=(0..outer*k*inner).map(|i|(i%101) as f64/10.).collect();
        assert_eq!(columns(&a,outer,k,inner),rows(&a,outer,k,inner));
        let col=time(||columns(black_box(&a),black_box(outer),black_box(k),black_box(inner)));
        let row=time(||rows(black_box(&a),black_box(outer),black_box(k),black_box(inner)));
        println!("outer={outer} k={k} inner={inner} gather_us={col:.3} stream_us={row:.3} speedup={:.2}",col/row);
    }
}

//! Criterion benchmarks for `numpy-linalg` (matmul, batched matmul, solve, inv, cholesky, qr, svd, eigh, norm).

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use numpy_bench::{default_criterion, gen_f32, gen_f64, gen_spd_f64, gen_well_cond_f64};
use numpy_linalg::{
    NormOrd, QrMode, Uplo, cholesky, det, eigh, inv, lstsq, matmul, norm, qr, solve, svd,
};

fn bench_linalg_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_matmul");

    for &n in &[64usize, 256, 512, 1024] {
        if n >= 512 {
            group.sample_size(15);
        } else {
            group.sample_size(30);
        }
        let a = gen_f64(&[n, n], 1);
        let b = gen_f64(&[n, n], 2);
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bencher, _| {
            bencher.iter(|| black_box(matmul(&a, &b).unwrap()));
        });
    }

    for &n in &[256usize, 512, 1024] {
        if n >= 512 {
            group.sample_size(15);
        } else {
            group.sample_size(30);
        }
        let a = gen_f32(&[n, n], 3);
        let b = gen_f32(&[n, n], 4);
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("f32", n), &n, |bencher, _| {
            bencher.iter(|| black_box(matmul(&a, &b).unwrap()));
        });
    }

    group.sample_size(20);
    let a256 = gen_f64(&[256, 256], 5);
    let bt256 = gen_f64(&[256, 256], 6).t();
    group.bench_function("f64_transposed_256", |bencher| {
        bencher.iter(|| black_box(matmul(&a256, &bt256).unwrap()));
    });

    let ba = gen_f64(&[16, 64, 64], 7);
    let bb = gen_f64(&[16, 64, 64], 8);
    group.bench_function("f64_batched_16x64x64", |bencher| {
        bencher.iter(|| black_box(matmul(&ba, &bb).unwrap()));
    });

    group.finish();
}

fn bench_linalg_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_solve");

    for &n in &[64usize, 256, 512] {
        if n >= 512 {
            group.sample_size(15);
        } else {
            group.sample_size(25);
        }
        let a = gen_well_cond_f64(n, 10);
        let rhs = gen_f64(&[n, 1], 11);
        group.bench_with_input(BenchmarkId::new("solve_f64", n), &n, |bencher, _| {
            bencher.iter(|| black_box(solve(&a, &rhs).unwrap()));
        });
    }

    group.sample_size(20);
    let w256 = gen_well_cond_f64(256, 12);
    group.bench_function("inv_f64_256", |bencher| {
        bencher.iter(|| black_box(inv(&w256).unwrap()));
    });
    group.bench_function("det_f64_256", |bencher| {
        bencher.iter(|| black_box(det(&w256).unwrap()));
    });

    for &n in &[256usize, 512] {
        if n >= 512 {
            group.sample_size(15);
        }
        let spd = gen_spd_f64(n, 13);
        group.bench_with_input(BenchmarkId::new("cholesky_f64", n), &n, |bencher, _| {
            bencher.iter(|| black_box(cholesky(&spd, false).unwrap()));
        });
    }

    group.finish();
}

fn bench_linalg_decomp(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_decomp");

    for &n in &[128usize, 256, 512] {
        if n >= 256 {
            group.sample_size(15);
        } else {
            group.sample_size(25);
        }
        let a = gen_well_cond_f64(n, 20);
        group.bench_with_input(BenchmarkId::new("qr_f64", n), &n, |bencher, _| {
            bencher.iter(|| black_box(qr(&a, QrMode::Reduced).unwrap()));
        });
    }

    for &n in &[64usize, 128, 256] {
        if n >= 256 {
            group.sample_size(15);
        } else {
            group.sample_size(20);
        }
        let a = gen_well_cond_f64(n, 21);
        let spd = gen_spd_f64(n, 22);
        group.bench_with_input(BenchmarkId::new("svd_f64", n), &n, |bencher, _| {
            bencher.iter(|| black_box(svd(&a, false, true, false).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("eigh_f64", n), &n, |bencher, _| {
            bencher.iter(|| black_box(eigh(&spd, Uplo::Lower).unwrap()));
        });
    }

    group.sample_size(15);
    let m = gen_f64(&[512, 128], 23);
    let y = gen_f64(&[512, 1], 24);
    group.bench_function("lstsq_512x128", |bencher| {
        bencher.iter(|| black_box(lstsq(&m, &y, None).unwrap()));
    });

    group.finish();
}

fn bench_linalg_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_norm");

    let v_100k = gen_f64(&[100_000], 30);
    let v_10m = gen_f64(&[10_000_000], 31);

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("norm_f64_100k", |bencher| {
        bencher.iter(|| black_box(norm(&v_100k, NormOrd::Default, None, false).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("norm_f64_10m", |bencher| {
        bencher.iter(|| black_box(norm(&v_10m, NormOrd::Default, None, false).unwrap()));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_linalg_matmul, bench_linalg_solve, bench_linalg_decomp, bench_linalg_norm
}
criterion_main!(benches);

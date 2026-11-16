//! Criterion benchmarks for `numpy-random` (PCG64, PCG64DXSM, Philox, SFC64, MT19937, bulk & parallel distributions).

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use numpy_bench::default_criterion;
use numpy_random::{Generator, MT19937, PCG64DXSM, Philox, SFC64, default_rng_from_seed};

fn bench_random_pcg64(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_pcg64");
    let mut rng = default_rng_from_seed(12345u64).unwrap();

    for &(name, len) in &[
        ("random_f64_1k", 1_000usize),
        ("random_f64_100k", 100_000),
    ] {
        group.throughput(Throughput::Bytes((len * 8) as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(rng.random(&[len]).unwrap()));
        });
    }

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("random_f64_10m", |b| {
        b.iter(|| black_box(rng.random(&[10_000_000]).unwrap()));
    });

    group.sample_size(20);
    for &(name, len) in &[
        ("standard_normal_f64_1k", 1_000usize),
        ("standard_normal_f64_100k", 100_000),
    ] {
        group.throughput(Throughput::Bytes((len * 8) as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(rng.standard_normal(&[len]).unwrap()));
        });
    }

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("standard_normal_f64_10m", |b| {
        b.iter(|| black_box(rng.standard_normal(&[10_000_000]).unwrap()));
    });

    group.sample_size(20);
    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("standard_exponential_f64_100k", |b| {
        b.iter(|| black_box(rng.standard_exponential(&[100_000]).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("standard_exponential_f64_10m", |b| {
        b.iter(|| black_box(rng.standard_exponential(&[10_000_000]).unwrap()));
    });

    group.sample_size(20);
    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("integers_i64_100k", |b| {
        b.iter(|| black_box(rng.integers(0, 1_000_000, &[100_000]).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("integers_i64_10m", |b| {
        b.iter(|| black_box(rng.integers(0, 1_000_000, &[10_000_000]).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("uniform_f64_1m", |b| {
        b.iter(|| black_box(rng.uniform(-5.0, 5.0, &[1_000_000]).unwrap()));
    });

    group.bench_function("gamma_f64_1m", |b| {
        b.iter(|| black_box(rng.gamma(2.5, 1.5, &[1_000_000]).unwrap()));
    });

    group.bench_function("poisson_i64_1m", |b| {
        b.iter(|| black_box(rng.poisson(5.0, &[1_000_000]).unwrap()));
    });

    group.bench_function("binomial_i64_1m", |b| {
        b.iter(|| black_box(rng.binomial(20, 0.3, &[1_000_000]).unwrap()));
    });

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("choice_no_replace_100k_from_1m", |b| {
        b.iter(|| {
            black_box(
                rng.choice_range(1_000_000, &[100_000], false, None, true)
                    .unwrap(),
            )
        });
    });

    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("permutation_1m", |b| {
        b.iter(|| black_box(rng.permutation_range(1_000_000).unwrap()));
    });

    group.finish();
}

fn bench_random_bitgens(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_bitgens");
    group.sample_size(15);
    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));

    let mut rng_dxsm = Generator::new(PCG64DXSM::from_seed(12345u64).unwrap());
    group.bench_function("pcg64dxsm_random_1m", |b| {
        b.iter(|| black_box(rng_dxsm.random(&[1_000_000]).unwrap()));
    });

    let mut rng_philox = Generator::new(Philox::from_seed(12345u64).unwrap());
    group.bench_function("philox_random_1m", |b| {
        b.iter(|| black_box(rng_philox.random(&[1_000_000]).unwrap()));
    });

    let mut rng_sfc = Generator::new(SFC64::from_seed(12345u64).unwrap());
    group.bench_function("sfc64_random_1m", |b| {
        b.iter(|| black_box(rng_sfc.random(&[1_000_000]).unwrap()));
    });

    let mut rng_mt = Generator::new(MT19937::from_seed(12345u64).unwrap());
    group.bench_function("mt19937_random_1m", |b| {
        b.iter(|| black_box(rng_mt.random(&[1_000_000]).unwrap()));
    });

    group.finish();
}

fn bench_random_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_par");
    let mut rng = default_rng_from_seed(12345u64).unwrap();

    group.sample_size(20);
    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("par_random_f64_100k", |b| {
        b.iter(|| black_box(rng.par_random(&[100_000]).unwrap()));
    });
    group.bench_function("par_standard_normal_f64_100k", |b| {
        b.iter(|| black_box(rng.par_standard_normal(&[100_000]).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("par_gamma_f64_1m", |b| {
        b.iter(|| black_box(rng.par_gamma(2.5, 1.5, &[1_000_000]).unwrap()));
    });

    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("par_random_f64_10m", |b| {
        b.iter(|| black_box(rng.par_random(&[10_000_000]).unwrap()));
    });
    group.bench_function("par_standard_normal_f64_10m", |b| {
        b.iter(|| black_box(rng.par_standard_normal(&[10_000_000]).unwrap()));
    });
    group.bench_function("par_standard_exponential_f64_10m", |b| {
        b.iter(|| black_box(rng.par_standard_exponential(&[10_000_000]).unwrap()));
    });
    group.bench_function("par_integers_i64_10m", |b| {
        b.iter(|| black_box(rng.par_integers(0, 1_000_000, &[10_000_000]).unwrap()));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_random_pcg64, bench_random_bitgens, bench_random_par
}
criterion_main!(benches);

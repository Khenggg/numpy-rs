use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use numpy_bench::{default_criterion, gen_c128, gen_f64};
use numpy_fft::{Norm, fft, fft2, rfft, rfft2};

fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_1d");

    let c_1k = gen_c128(&[1024], 101);
    let c_64k = gen_c128(&[65536], 102);
    let c_1m = gen_c128(&[1_048_576], 103);
    let r_64k = gen_f64(&[65536], 104);
    let c_1000 = gen_c128(&[1000], 105);
    let c_10000 = gen_c128(&[10000], 106);
    let c_997 = gen_c128(&[997], 107);

    group.throughput(Throughput::Bytes((1024 * 16) as u64));
    group.bench_function("c128_1024", |b| {
        b.iter(|| black_box(fft(&c_1k, None, -1, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((65536 * 16) as u64));
    group.bench_function("c128_65536", |b| {
        b.iter(|| black_box(fft(&c_64k, None, -1, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((65536 * 8) as u64));
    group.bench_function("f64_65536", |b| {
        b.iter(|| black_box(fft(&r_64k, None, -1, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((1000 * 16) as u64));
    group.bench_function("c128_1000", |b| {
        b.iter(|| black_box(fft(&c_1000, None, -1, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((10000 * 16) as u64));
    group.bench_function("c128_10000", |b| {
        b.iter(|| black_box(fft(&c_10000, None, -1, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((997 * 16) as u64));
    group.bench_function("c128_997_prime", |b| {
        b.iter(|| black_box(fft(&c_997, None, -1, Norm::Backward).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((1_048_576 * 16) as u64));
    group.bench_function("c128_1048576", |b| {
        b.iter(|| black_box(fft(&c_1m, None, -1, Norm::Backward).unwrap()));
    });

    group.finish();
}

fn bench_rfft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft_1d");

    let r_1k = gen_f64(&[1024], 110);
    let r_64k = gen_f64(&[65536], 111);
    let r_1m = gen_f64(&[1_048_576], 112);

    group.throughput(Throughput::Bytes((1024 * 8) as u64));
    group.bench_function("f64_1024", |b| {
        b.iter(|| black_box(rfft(&r_1k, None, -1, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((65536 * 8) as u64));
    group.bench_function("f64_65536", |b| {
        b.iter(|| black_box(rfft(&r_64k, None, -1, Norm::Backward).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((1_048_576 * 8) as u64));
    group.bench_function("f64_1048576", |b| {
        b.iter(|| black_box(rfft(&r_1m, None, -1, Norm::Backward).unwrap()));
    });

    group.finish();
}

fn bench_fft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_2d");

    let c_256 = gen_c128(&[256, 256], 120);
    let c_1024 = gen_c128(&[1024, 1024], 121);
    let r_512 = gen_f64(&[512, 512], 122);
    let r_1024 = gen_f64(&[1024, 1024], 123);

    group.throughput(Throughput::Bytes((256 * 256 * 16) as u64));
    group.bench_function("c128_256x256", |b| {
        b.iter(|| black_box(fft2(&c_256, None, None, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((512 * 512 * 8) as u64));
    group.bench_function("f64_512x512", |b| {
        b.iter(|| black_box(fft2(&r_512, None, None, Norm::Backward).unwrap()));
    });
    group.bench_function("rfft2_f64_512x512", |b| {
        b.iter(|| black_box(rfft2(&r_512, None, None, Norm::Backward).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((1024 * 1024 * 16) as u64));
    group.bench_function("c128_1024x1024", |b| {
        b.iter(|| black_box(fft2(&c_1024, None, None, Norm::Backward).unwrap()));
    });

    group.throughput(Throughput::Bytes((1024 * 1024 * 8) as u64));
    group.bench_function("rfft2_f64_1024x1024", |b| {
        b.iter(|| black_box(rfft2(&r_1024, None, None, Norm::Backward).unwrap()));
    });

    group.finish();
}

fn bench_fft_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_batch");
    group.sample_size(20);

    let c_batch = gen_c128(&[1000, 1024], 130);
    group.throughput(Throughput::Bytes((1000 * 1024 * 16) as u64));
    group.bench_function("c128_1000x1024_axis1", |b| {
        b.iter(|| black_box(fft(&c_batch, None, 1, Norm::Backward).unwrap()));
    });
    group.bench_function("c128_1000x1024_axis0", |b| {
        b.iter(|| black_box(fft(&c_batch, None, 0, Norm::Backward).unwrap()));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_fft_1d, bench_rfft_1d, bench_fft_2d, bench_fft_batch
}
criterion_main!(benches);

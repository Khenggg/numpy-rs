//! Criterion benchmarks for `numpy-core` (creation, manipulation, indexing, einsum).

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use numpy_bench::{default_criterion, gen_f64};
use numpy_core::{
    IndexMode, PadMode, PadWidth, arange, boolean_mask, concatenate, einsum, eye, linspace, pad,
    repeat, roll, stack, take, tile, where_select,
};
use numpy_types::NdArray;

fn bench_core_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_create");

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("arange_f64_100k", |b| {
        b.iter(|| black_box(arange(0.0f64, 100_000.0, 1.0).unwrap()));
    });
    group.bench_function("linspace_f64_100k", |b| {
        b.iter(|| black_box(linspace(0.0f64, 1.0, 100_000, true).unwrap()));
    });

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("arange_f64_10m", |b| {
        b.iter(|| black_box(arange(0.0f64, 10_000_000.0, 1.0).unwrap()));
    });

    group.throughput(Throughput::Bytes((1000 * 1000 * 8) as u64));
    group.bench_function("eye_f64_1000x1000", |b| {
        b.iter(|| black_box(eye::<f64>(1000, Some(1000), 0).unwrap()));
    });

    group.finish();
}

fn bench_core_manip(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_manip");

    let v_100k_a = gen_f64(&[100_000], 1);
    let v_100k_b = gen_f64(&[100_000], 2);
    let m_100 = gen_f64(&[100, 100], 3);
    let m_1000 = gen_f64(&[1000, 1000], 4);
    let v_1m = gen_f64(&[1_000_000], 5);

    group.throughput(Throughput::Bytes((200_000 * 8) as u64));
    group.bench_function("concatenate_2x_f64_100k", |b| {
        b.iter(|| black_box(concatenate(&[&v_100k_a, &v_100k_b], Some(0)).unwrap()));
    });

    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("tile_100x100_10x10", |b| {
        b.iter(|| black_box(tile(&m_100, &[10, 10]).unwrap()));
    });
    group.bench_function("repeat_100k_x10", |b| {
        b.iter(|| black_box(repeat(&v_100k_a, &[10], None).unwrap()));
    });
    group.bench_function("roll_1d_1m", |b| {
        b.iter(|| black_box(roll(&v_1m, 1024, Some(0)).unwrap()));
    });

    group.sample_size(15);
    let v_5m_a = gen_f64(&[5_000_000], 6);
    let v_5m_b = gen_f64(&[5_000_000], 7);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("concatenate_2x_f64_5m", |b| {
        b.iter(|| black_box(concatenate(&[&v_5m_a, &v_5m_b], Some(0)).unwrap()));
    });

    group.throughput(Throughput::Bytes((4 * 1000 * 1000 * 8) as u64));
    group.bench_function("stack_4x_f64_1000x1000", |b| {
        b.iter(|| black_box(stack(&[&m_1000, &m_1000, &m_1000, &m_1000], 0).unwrap()));
    });

    group.throughput(Throughput::Bytes((1004 * 1004 * 8) as u64));
    group.bench_function("pad_constant_1000x1000", |b| {
        b.iter(|| black_box(pad(&m_1000, &PadWidth::Uniform(2), &PadMode::Constant(0.0)).unwrap()));
    });

    group.finish();
}

fn bench_core_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_index");

    let a_100k = gen_f64(&[100_000], 10);
    let b_100k = gen_f64(&[100_000], 11);
    let cond_100k = a_100k.map(|x| x > 0.0);

    let a_1m = gen_f64(&[1_000_000], 12);
    let b_1m = gen_f64(&[1_000_000], 13);
    let cond_1m = a_1m.map(|x| x > 0.0);

    let m_1000 = gen_f64(&[1000, 1000], 14);
    let idx_100 = NdArray::from_vec((0..100).map(|i| (i * 9) as i64).collect());

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("where_select_f64_100k", |b| {
        b.iter(|| black_box(where_select(&cond_100k, &a_100k, &b_100k).unwrap()));
    });

    group.throughput(Throughput::Bytes((100 * 1000 * 8) as u64));
    group.bench_function("take_axis0_1000x1000_100idx", |b| {
        b.iter(|| black_box(take(&m_1000, &idx_100, Some(0), IndexMode::Raise).unwrap()));
    });

    group.sample_size(20);
    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("where_select_f64_1m", |b| {
        b.iter(|| black_box(where_select(&cond_1m, &a_1m, &b_1m).unwrap()));
    });
    group.bench_function("boolean_mask_f64_1m_50pct", |b| {
        b.iter(|| black_box(boolean_mask(&a_1m, &cond_1m).unwrap()));
    });

    group.finish();
}

fn bench_core_einsum(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_einsum");

    let a_256 = gen_f64(&[256, 256], 20);
    let b_256 = gen_f64(&[256, 256], 21);
    let ba = gen_f64(&[32, 64, 64], 22);
    let bb = gen_f64(&[32, 64, 64], 23);
    let m_1000a = gen_f64(&[1000, 1000], 24);
    let m_1000b = gen_f64(&[1000, 1000], 25);

    group.sample_size(20);
    group.bench_function("einsum_ij_jk_ik_256x256", |b| {
        b.iter(|| black_box(einsum("ij,jk->ik", &[&a_256, &b_256]).unwrap()));
    });
    group.bench_function("einsum_bij_bjk_bik_32x64x64", |b| {
        b.iter(|| black_box(einsum("bij,bjk->bik", &[&ba, &bb]).unwrap()));
    });
    group.bench_function("einsum_ij_ij_1000x1000", |b| {
        b.iter(|| black_box(einsum("ij,ij->", &[&m_1000a, &m_1000b]).unwrap()));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_core_create, bench_core_manip, bench_core_index, bench_core_einsum
}
criterion_main!(benches);

//! Criterion benchmarks for `numpy-ufunc` (1K, 100K, 10M f64/f32, broadcasted, strided, reductions).

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use numpy_bench::{default_criterion, gen_f32_pos, gen_f64, gen_f64_pos, gen_i32};
use numpy_types::{DynArray, Slice, SliceArg};
use numpy_ufunc::{
    BinaryOp, add, cos, exp, greater, isclose, log, multiply, power, sin, sqrt, tanh, true_divide,
};

fn bench_ufunc_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("ufunc_binary");

    for &(label, n) in &[("1k", 1_000usize), ("100k", 100_000), ("10m", 10_000_000)] {
        if n >= 10_000_000 {
            group.sample_size(15);
        } else {
            group.sample_size(40);
        }
        let a64 = gen_f64_pos(&[n], 1);
        let b64 = gen_f64_pos(&[n], 2);
        let a32 = gen_f32_pos(&[n], 3);
        let b32 = gen_f32_pos(&[n], 4);

        group.throughput(Throughput::Bytes((n * std::mem::size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::new("add_f64", label), &n, |b, _| {
            b.iter(|| black_box(add(&a64, &b64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("mul_f64", label), &n, |b, _| {
            b.iter(|| black_box(multiply(&a64, &b64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("div_f64", label), &n, |b, _| {
            b.iter(|| black_box(true_divide(&a64, &b64).unwrap()));
        });
        if n <= 100_000 {
            group.bench_with_input(BenchmarkId::new("pow_f64", label), &n, |b, _| {
                b.iter(|| black_box(power(&a64, &b64).unwrap()));
            });
        }

        group.throughput(Throughput::Bytes((n * std::mem::size_of::<f32>()) as u64));
        group.bench_with_input(BenchmarkId::new("add_f32", label), &n, |b, _| {
            b.iter(|| black_box(add(&a32, &b32).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("mul_f32", label), &n, |b, _| {
            b.iter(|| black_box(multiply(&a32, &b32).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("div_f32", label), &n, |b, _| {
            b.iter(|| black_box(true_divide(&a32, &b32).unwrap()));
        });
    }

    group.finish();
}

fn bench_ufunc_unary(c: &mut Criterion) {
    let mut group = c.benchmark_group("ufunc_unary");

    for &(label, n) in &[("1k", 1_000usize), ("100k", 100_000), ("10m", 10_000_000)] {
        if n >= 10_000_000 {
            group.sample_size(15);
        } else {
            group.sample_size(40);
        }
        let a64 = gen_f64_pos(&[n], 10);
        let a32 = gen_f32_pos(&[n], 11);

        group.throughput(Throughput::Bytes((n * std::mem::size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::new("sqrt_f64", label), &n, |b, _| {
            b.iter(|| black_box(sqrt(&a64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("sin_f64", label), &n, |b, _| {
            b.iter(|| black_box(sin(&a64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("cos_f64", label), &n, |b, _| {
            b.iter(|| black_box(cos(&a64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("exp_f64", label), &n, |b, _| {
            b.iter(|| black_box(exp(&a64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("log_f64", label), &n, |b, _| {
            b.iter(|| black_box(log(&a64).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("tanh_f64", label), &n, |b, _| {
            b.iter(|| black_box(tanh(&a64).unwrap()));
        });

        group.throughput(Throughput::Bytes((n * std::mem::size_of::<f32>()) as u64));
        group.bench_with_input(BenchmarkId::new("sqrt_f32", label), &n, |b, _| {
            b.iter(|| black_box(sqrt(&a32).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("sin_f32", label), &n, |b, _| {
            b.iter(|| black_box(sin(&a32).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("exp_f32", label), &n, |b, _| {
            b.iter(|| black_box(exp(&a32).unwrap()));
        });
    }

    group.finish();
}

fn bench_ufunc_strided_bcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("ufunc_strided_bcast");

    let mat = gen_f64(&[1000, 1000], 20);
    let mat_t = mat.t();
    let row = gen_f64(&[1, 1000], 21);
    let col = gen_f64(&[1000, 1], 22);
    let vec_200k = gen_f64(&[200_000], 23);
    let strided_100k = vec_200k
        .slice(&[SliceArg::Slice(Slice::full().step_by(2))])
        .unwrap();
    let contig_100k = gen_f64(&[100_000], 24);

    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("add_broadcast_row_1000x1000", |b| {
        b.iter(|| black_box(add(&mat, &row).unwrap()));
    });
    group.bench_function("add_broadcast_col_1000x1000", |b| {
        b.iter(|| black_box(add(&mat, &col).unwrap()));
    });
    group.bench_function("add_transposed_1000x1000", |b| {
        b.iter(|| black_box(add(&mat, &mat_t).unwrap()));
    });
    group.bench_function("sin_transposed_1000x1000", |b| {
        b.iter(|| black_box(sin(&mat_t).unwrap()));
    });

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("add_strided_100k", |b| {
        b.iter(|| black_box(add(&strided_100k, &contig_100k).unwrap()));
    });
    group.bench_function("sin_strided_100k", |b| {
        b.iter(|| black_box(sin(&strided_100k).unwrap()));
    });

    let big = gen_f64(&[1000, 10_000], 25);
    let big_row = gen_f64(&[1, 10_000], 26);
    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("add_broadcast_10m", |b| {
        b.iter(|| black_box(add(&big, &big_row).unwrap()));
    });

    group.finish();
}

fn bench_ufunc_higher(c: &mut Criterion) {
    let mut group = c.benchmark_group("ufunc_higher");

    let a_100k = gen_f64(&[100_000], 30);
    let b_100k = gen_f64(&[100_000], 31);
    let v_256 = gen_f64(&[256], 32);
    let dyn_i = DynArray::I32(gen_i32(&[100_000], 33));
    let dyn_f = DynArray::F64(a_100k.clone());

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("reduce_add_100k", |b| {
        b.iter(|| black_box(BinaryOp::Add.reduce_nd(&a_100k, None, false, None).unwrap()));
    });
    group.bench_function("accumulate_add_100k", |b| {
        b.iter(|| black_box(BinaryOp::Add.accumulate_nd(&a_100k, 0).unwrap()));
    });
    group.bench_function("outer_mul_256x256", |b| {
        b.iter(|| black_box(BinaryOp::Multiply.outer_nd(&v_256, &v_256).unwrap()));
    });
    group.bench_function("greater_100k", |b| {
        b.iter(|| black_box(greater(&a_100k, &b_100k).unwrap()));
    });
    group.bench_function("isclose_100k", |b| {
        b.iter(|| black_box(isclose(&a_100k, &b_100k, 1e-5, 1e-8, false).unwrap()));
    });
    group.bench_function("dyn_add_i32_f64_100k", |b| {
        b.iter(|| black_box(add(&dyn_i, &dyn_f).unwrap()));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_ufunc_binary, bench_ufunc_unary, bench_ufunc_strided_bcast, bench_ufunc_higher
}
criterion_main!(benches);

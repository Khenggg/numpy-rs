use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use numpy_bench::{default_criterion, gen_f32, gen_f64};
use numpy_types::{Casting, DType, DynArray, NdArray, Slice, SliceArg};

fn bench_zeros_and_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("types_zeros");

    for &n in &[1_000usize, 100_000, 10_000_000] {
        let label = match n {
            1_000 => "1k",
            100_000 => "100k",
            10_000_000 => "10m",
            _ => unreachable!(),
        };
        if n >= 10_000_000 {
            group.sample_size(15);
        } else {
            group.sample_size(40);
        }
        group.throughput(Throughput::Bytes((n * std::mem::size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::new("zeros_f64", label), &n, |b, &len| {
            b.iter(|| black_box(NdArray::<f64>::zeros(&[len]).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("full_f64", label), &n, |b, &len| {
            b.iter(|| black_box(NdArray::<f64>::full(&[len], 3.14159).unwrap()));
        });
    }

    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * std::mem::size_of::<f32>()) as u64));
    group.bench_with_input(BenchmarkId::new("zeros_f32", "10m"), &10_000_000usize, |b, &len| {
        b.iter(|| black_box(NdArray::<f32>::zeros(&[len]).unwrap()));
    });
    group.bench_with_input(BenchmarkId::new("full_f32", "10m"), &10_000_000usize, |b, &len| {
        b.iter(|| black_box(NdArray::<f32>::full(&[len], 3.14159f32).unwrap()));
    });

    group.finish();
}

fn bench_views(c: &mut Criterion) {
    let mut group = c.benchmark_group("types_view");
    let mat = gen_f64(&[1000, 1000], 1);
    let vec_100k = gen_f64(&[100_000], 2);
    let row = gen_f64(&[1, 1000], 3);

    let sub_args = [
        SliceArg::from(100isize..900isize),
        SliceArg::from(100isize..900isize),
    ];
    group.bench_function("slice_2d_1000x1000", |b| {
        b.iter(|| black_box(mat.slice(black_box(&sub_args)).unwrap()));
    });

    let strided_args = [SliceArg::Slice(Slice::full().step_by(2))];
    group.bench_function("slice_strided_100k_step2", |b| {
        b.iter(|| black_box(vec_100k.slice(black_box(&strided_args)).unwrap()));
    });

    let flip_args = [SliceArg::Slice(Slice::full().step_by(-1))];
    group.bench_function("slice_flip_100k", |b| {
        b.iter(|| black_box(vec_100k.slice(black_box(&flip_args)).unwrap()));
    });

    group.bench_function("transpose_1000x1000", |b| {
        b.iter(|| black_box(mat.t()));
    });

    group.bench_function("broadcast_1x1000_to_1000x1000", |b| {
        b.iter(|| black_box(row.broadcast_to(black_box(&[1000, 1000])).unwrap()));
    });

    group.bench_function("reshape_1000x1000_to_100x10000", |b| {
        b.iter(|| black_box(mat.reshape(black_box(&[100, 10_000])).unwrap()));
    });

    group.finish();
}

fn bench_to_c_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("types_to_c_contiguous");

    let mat_1k = gen_f64(&[1000, 1000], 10);
    let mat_t_1k = mat_1k.t();
    let mat_512 = gen_f64(&[512, 512], 11);
    let mat_t_512 = mat_512.t();
    let sub_800 = mat_1k
        .slice(&[
            SliceArg::from(100isize..900isize),
            SliceArg::from(100isize..900isize),
        ])
        .unwrap();
    let flipped_1k = mat_1k.flip(None).unwrap();
    let bcast_1k = gen_f64(&[1, 1000], 12).broadcast_to(&[1000, 1000]).unwrap();

    let vec_200k = gen_f64(&[200_000], 13);
    let strided_100k = vec_200k
        .slice(&[SliceArg::Slice(Slice::full().step_by(2))])
        .unwrap();

    group.bench_function("noop_1000x1000", |b| {
        b.iter(|| black_box(mat_1k.to_c_contiguous()));
    });

    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("transpose_1000x1000", |b| {
        b.iter(|| black_box(mat_t_1k.to_c_contiguous()));
    });

    group.throughput(Throughput::Bytes((512 * 512 * 8) as u64));
    group.bench_function("transpose_512x512", |b| {
        b.iter(|| black_box(mat_t_512.to_c_contiguous()));
    });

    group.throughput(Throughput::Bytes((100_000 * 8) as u64));
    group.bench_function("strided_100k_step2", |b| {
        b.iter(|| black_box(strided_100k.to_c_contiguous()));
    });

    group.throughput(Throughput::Bytes((640_000 * 8) as u64));
    group.bench_function("submatrix_800x800", |b| {
        b.iter(|| black_box(sub_800.to_c_contiguous()));
    });

    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("flip_1000x1000", |b| {
        b.iter(|| black_box(flipped_1k.to_c_contiguous()));
    });

    group.throughput(Throughput::Bytes((1_000_000 * 8) as u64));
    group.bench_function("broadcast_1000x1000", |b| {
        b.iter(|| black_box(bcast_1k.to_c_contiguous()));
    });

    let vec_20m = gen_f64(&[20_000_000], 14);
    let strided_10m = vec_20m
        .slice(&[SliceArg::Slice(Slice::full().step_by(2))])
        .unwrap();
    group.sample_size(15);
    group.throughput(Throughput::Bytes((10_000_000 * 8) as u64));
    group.bench_function("strided_10m_step2", |b| {
        b.iter(|| black_box(strided_10m.to_c_contiguous()));
    });

    group.finish();
}

fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("types_map");

    let a_1k = gen_f64(&[1_000], 20);
    let a_100k = gen_f64(&[100_000], 21);
    let a_10m = gen_f64(&[10_000_000], 22);
    let f32_10m = gen_f32(&[10_000_000], 23);

    group.throughput(Throughput::Elements(1_000));
    group.bench_function("contig_f64_1k", |b| {
        b.iter(|| black_box(a_1k.map(|x| x * 2.0 + 1.0)));
    });

    group.throughput(Throughput::Elements(100_000));
    group.bench_function("contig_f64_100k", |b| {
        b.iter(|| black_box(a_100k.map(|x| x * 2.0 + 1.0)));
    });

    let mat_t = gen_f64(&[1000, 1000], 24).t();
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("transpose_f64_1000x1000", |b| {
        b.iter(|| black_box(mat_t.map(|x| x * 2.0 + 1.0)));
    });

    let strided_100k = gen_f64(&[200_000], 25)
        .slice(&[SliceArg::Slice(Slice::full().step_by(2))])
        .unwrap();
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("strided_f64_100k_step2", |b| {
        b.iter(|| black_box(strided_100k.map(|x| x * 2.0 + 1.0)));
    });

    group.sample_size(15);
    group.throughput(Throughput::Elements(10_000_000));
    group.bench_function("contig_f64_10m", |b| {
        b.iter(|| black_box(a_10m.map(|x| x * 2.0 + 1.0)));
    });
    group.bench_function("contig_f32_10m", |b| {
        b.iter(|| black_box(f32_10m.map(|x| x * 2.0f32 + 1.0f32)));
    });

    group.finish();
}

fn bench_dyn_astype(c: &mut Criterion) {
    let mut group = c.benchmark_group("types_dyn");
    let d_100k = DynArray::F32(gen_f32(&[100_000], 30));
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("astype_f32_to_f64_100k", |b| {
        b.iter(|| black_box(d_100k.astype(DType::F64, Casting::Safe).unwrap()));
    });

    let d_10m = DynArray::F32(gen_f32(&[10_000_000], 31));
    group.sample_size(15);
    group.throughput(Throughput::Elements(10_000_000));
    group.bench_function("astype_f32_to_f64_10m", |b| {
        b.iter(|| black_box(d_10m.astype(DType::F64, Casting::Safe).unwrap()));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_zeros_and_full, bench_views, bench_to_c_contiguous, bench_map, bench_dyn_astype
}
criterion_main!(benches);

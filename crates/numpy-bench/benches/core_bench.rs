//! Criterion benchmarks for `numpy-core` (creation, manipulation, indexing, einsum).
//! Enabled with `--features bench-core` once `numpy-core` hands off.

use criterion::{Criterion, criterion_group, criterion_main};
use numpy_bench::default_criterion;

fn bench_core_placeholder(_c: &mut Criterion) {
    // Wired upon @core-dev handoff:
    // - core_create: arange/f64_100k, arange/f64_10m, linspace/f64_100k, eye/f64_1000x1000
    // - core_manip: concatenate/2x_f64_100k, concatenate/2x_f64_5m, stack/4x_f64_1000x1000,
    //               pad/constant_1000x1000, tile/100x100_10x10, repeat/100k_x10, roll/1d_1m
    // - core_index: where_select/f64_100k, where_select/f64_1m, take/axis0_1000x1000_100idx,
    //               boolean_mask/f64_1m_50pct
    // - core_einsum: einsum/ij_jk_ik_256x256, einsum/bij_bjk_bik_32x64x64, einsum/ij_ij_1000x1000
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_core_placeholder
}
criterion_main!(benches);

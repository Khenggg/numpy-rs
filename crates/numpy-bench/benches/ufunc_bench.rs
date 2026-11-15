//! Criterion benchmarks for `numpy-ufunc` (1K, 100K, 10M f64/f32, broadcasted, strided, reductions).
//! Enabled with `--features bench-ufunc` once `numpy-ufunc` hands off.

use criterion::{Criterion, criterion_group, criterion_main};
use numpy_bench::default_criterion;

fn bench_ufunc_placeholder(_c: &mut Criterion) {
    // Wired upon @ufunc-dev handoff:
    // - ufunc_binary: add/mul/div/pow across f64/f32 (1k, 100k, 10m)
    // - ufunc_unary: sqrt/sin/cos/exp/log/tanh across f64/f32 (1k, 100k, 10m)
    // - ufunc_strided_bcast: add_broadcast (row/col/10m), add_strided, add_transposed, sin_strided, sin_transposed
    // - ufunc_higher: reduce, accumulate, outer, greater, isclose, DynArray mixed promotion
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_ufunc_placeholder
}
criterion_main!(benches);

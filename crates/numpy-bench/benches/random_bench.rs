//! Criterion benchmarks for `numpy-random` (PCG64, PCG64DXSM, Philox, SFC64, MT19937, bulk distributions).
//! Enabled with `--features bench-random` once `numpy-random` hands off.

use criterion::{Criterion, criterion_group, criterion_main};
use numpy_bench::default_criterion;

fn bench_random_placeholder(_c: &mut Criterion) {
    // Wired upon @random-dev handoff:
    // - random_pcg64: random (1k, 100k, 10m), standard_normal (1k, 100k, 10m),
    //                 standard_exponential (100k, 10m), integers (100k, 10m),
    //                 uniform/gamma/poisson/binomial (1m), choice/permutation
    // - random_bitgens: pcg64dxsm/philox/sfc64/mt19937 random_1m
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_random_placeholder
}
criterion_main!(benches);

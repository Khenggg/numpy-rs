//! Criterion benchmarks for `numpy-fft` (1D power-of-2 & non-power-of-2, rfft, 2D fft2/rfft2, batched).
//! Enabled with `--features bench-fft` once `numpy-fft` hands off.

use criterion::{Criterion, criterion_group, criterion_main};
use numpy_bench::default_criterion;

fn bench_fft_placeholder(_c: &mut Criterion) {
    // Wired upon @fft-dev handoff:
    // - fft_1d: c128 (1024, 65536, 1048576, 1000, 10000, 997), f64_65536
    // - rfft_1d: f64 (1024, 65536, 1048576)
    // - fft_2d: c128 (256x256, 1024x1024), f64_512x512, rfft2 (512x512, 1024x1024)
    // - fft_batch: c128_1000x1024_axis1, c128_1000x1024_axis0
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_fft_placeholder
}
criterion_main!(benches);

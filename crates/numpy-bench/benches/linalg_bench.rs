//! Criterion benchmarks for `numpy-linalg` (matmul, batched matmul, solve, inv, cholesky, qr, svd, eigh, norm).
//! Enabled with `--features bench-linalg` once `numpy-linalg` hands off.

use criterion::{Criterion, criterion_group, criterion_main};
use numpy_bench::default_criterion;

fn bench_linalg_placeholder(_c: &mut Criterion) {
    // Wired upon @linalg-dev handoff:
    // - linalg_matmul: f64 (64, 256, 512, 1024), f32 (256, 512, 1024), transposed, batched
    // - linalg_solve: solve (64, 256, 512), inv (256), det (256), cholesky (256, 512)
    // - linalg_decomp: qr (128, 256, 512), svd (64, 128, 256, 512), eigh (64, 128, 256, 512), lstsq (512x128)
    // - linalg_norm: norm f64 (100k, 10m)
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_linalg_placeholder
}
criterion_main!(benches);

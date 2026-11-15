# numpy-rs Workspace Architecture & Multi-Agent Contract

## Overview
`numpy-rs` is a pure-Rust reimplementation of NumPy located in `D:\numpy-rs`, with upstream NumPy (`numpy/numpy`) cloned at `D:\numpy-rs\upstream-numpy` and a live Python NumPy 2.5.4 oracle at `D:\numpy-rs\.venv\Scripts\python.exe`.

## Crate Ownership Boundaries (Strict — Never Edit Outside Assigned Crate)
1. `crates/numpy-types/` (**Foundation — Phase 0 locked**):
   - `AlignedBuffer<T>` / `AlignedUninit<T>` (64-byte SIMD alignment)
   - `NdArray<T>`, `ArrayView<'a, T>`, `ArrayViewMut<'a, T>`, `NdIter<'a, T>`
   - `DynArray` + `dispatch_dyn!`
   - `DType` (14 core types: `Bool, I8..I64, U8..U64, F16, F32, F64, C64, C128`), `Scalar`, `FloatElement`, `promote_types`, `result_type`, `can_cast`, `finfo`, `iinfo`
   - `Order`, `Slice`, `SliceArg`, `slice!`, `broadcast_shapes`, `broadcast_strides`, `normalize_axis`, `check_non_overlapping`
   - `NumPyError`, `Result<T>`
2. `crates/numpy-core/` (**Owner: `core-dev`**):
   - Creation: `empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `identity`, `diag`, `diagflat`, `tri`, `tril`, `triu`, `vander`, `meshgrid`, `indices`, `from_fn`
   - Manipulation: `reshape`, `ravel`, `flatten`, `transpose`, `swapaxes`, `moveaxis`, `rollaxis`, `squeeze`, `expand_dims`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `concatenate`, `concat`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`, `block`, `split`, `array_split`, `hsplit`, `vsplit`, `dsplit`, `unstack`, `tile`, `repeat`, `flip`, `fliplr`, `flipud`, `roll`, `rot90`, `pad`, `resize`, `append`, `insert`, `delete`, `trim_zeros`
   - Indexing & Selection: `take`, `take_along_axis`, `put`, `put_along_axis`, `putmask`, `compress`, `extract`, `choose`, `where_select` (`np.where`), `select`, `nonzero`, `argwhere`, `flatnonzero`, `diagonal`, `fill_diagonal`, `unravel_index`, `ravel_multi_index`, fancy integer indexing & boolean mask indexing
   - Contraction & Memory: `einsum`, `einsum_path`, `copyto`, `may_share_memory`, `shares_memory`, `memmap` (read-only/read-write via `memmap2`), `array2string` / pretty-print formatting
3. `crates/numpy-ufunc/` (**Owner: `ufunc-dev`**):
   - Element-wise unary/binary arithmetic, math, trig, hyperbolic, exp/log, rounding, complex ops, bitwise, comparison, logical, `isclose`/`allclose`, `nan_to_num`, `clip`, `sign`, `copysign`, `ldexp`, `frexp`, `modf`, `divmod`, `gcd`, `lcm`, `hypot`, `arctan2`, `heaviside`, `sinc`, `i0`, `spacing`, `nextafter`, `logaddexp`, `logaddexp2`
   - Ufunc methods: `reduce`, `accumulate`, `outer`, `at`
   - Both typed (`NdArray<T>`) zero-cost APIs AND dynamic (`DynArray`) auto-promoting APIs
   - Performance: `as_slice()` fast path, `std::arch::x86_64` AVX2+FMA kernels (runtime `is_x86_feature_detected!("avx2")` + `fma`), Rayon parallel chunking above cache-aware thresholds (`len >= 32_768`)
4. `crates/numpy-linalg/` (**Owner: `linalg-dev`**):
   - `dot`, `vdot`, `inner`, `outer`, `matmul`, `matvec`, `vecmat`, `vecdot`, `tensordot`, `kron`, `cross`, `multi_dot`, `matrix_power`, `trace`
   - `solve`, `tensorsolve`, `inv`, `tensorinv`, `pinv`, `lstsq`, `det`, `slogdet`, `matrix_rank`, `norm`, `vector_norm`, `matrix_norm`, `cond`
   - `cholesky`, `qr`, `svd`, `svdvals`, `eig`, `eigh`, `eigvals`, `eigvalsh`
   - Backed by `faer` / `matrixmultiply` + batched N-D Rayon parallelism (`[..., M, N]`), supporting `f32`, `f64`, `Complex32`, `Complex64` and `DynArray` auto-promotion (ints -> `f64`)
5. `crates/numpy-fft/` (**Owner: `fft-dev`**):
   - `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`
   - Norm modes: `Norm::Backward`, `Norm::Ortho`, `Norm::Forward`
   - Backed by `rustfft` + `realfft` + planner caching + Rayon multi-lane batching along arbitrary axes
6. `crates/numpy-random/` (**Owner: `random-dev`**):
   - `SeedSequence`, `BitGenerator` trait, `PCG64`, `PCG64DXSM`, `MT19937`, `Philox` (4x32-10), `SFC64`
   - `Generator` (`default_rng`) + legacy `RandomState`
   - Distributions: `integers`, `random`, `uniform`, `normal`, `standard_normal`, `exponential`, `standard_exponential`, `gamma`, `standard_gamma`, `beta`, `binomial`, `negative_binomial`, `poisson`, `geometric`, `hypergeometric`, `lognormal`, `logseries`, `chisquare`, `noncentral_chisquare`, `f`, `noncentral_f`, `standard_t`, `standard_cauchy`, `laplace`, `logistic`, `gumbel`, `rayleigh`, `weibull`, `pareto`, `power`, `triangular`, `vonmises`, `wald`, `zipf`, `dirichlet`, `multinomial`, `multivariate_normal`, `choice`, `permutation`, `shuffle`, `bytes`
   - SIMD / Lemire unbiased bounded integers, Ziggurat for normal/exponential, fast bulk generation into 64-byte aligned `AlignedBuffer`
7. `crates/numpy-lib/` (**Owner: `lib-dev`**):
   - Sorting & Searching: `sort`, `argsort`, `lexsort`, `partition`, `argpartition`, `searchsorted`, `top_k`, `sort_complex`
   - Reductions & Stats: `sum`, `prod`, `cumsum`, `cumprod`, `min`, `max`, `argmin`, `argmax`, `ptp`, `all`, `any`, `count_nonzero`, `mean`, `var`, `std`, `average`, `median`, `percentile`, `quantile`, `cov`, `corrcoef`
   - NaN functions: `nansum`, `nanprod`, `nancumsum`, `nancumprod`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`, `nanmean`, `nanvar`, `nanstd`, `nanmedian`, `nanpercentile`, `nanquantile`
   - Histograms & SetOps: `histogram`, `histogram_bin_edges`, `histogram2d`, `histogramdd`, `bincount`, `digitize`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`, `isin`, `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`, `ediff1d`
   - Signal / Calculus / Poly / IO: `interp`, `gradient`, `diff`, `trapezoid`, `convolve`, `correlate`, `unwrap`, `poly`, `poly1d` (`Polynomial`, `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE`), `polyfit`, `polyval`, `roots`, `polyder`, `polyint`, `polyadd`, `polysub`, `polymul`, `polydiv`, NPY/NPZ (`save`, `load`, `savez`, `savez_compressed`), `loadtxt`, `savetxt`
8. `crates/numpy-extra/` (**Owner: `extra-dev`**):
   - `MaskedArray` (`numpy.ma` core operations, masks, fill_value, compressed, masked reductions)
   - `DateTime64` & `TimeDelta64` (`TimeUnit`, arithmetic, `busday_count`, `busday_offset`, `is_busday`, ISO-8601 parsing/formatting)
   - Vectorized `strings` / `char` operations (`upper`, `lower`, `strip`, `split`, `replace`, `find`, `startswith`, `endswith`, `join`, `zfill`, `center`, etc.)
   - `StructuredArray` / `rec` (named fields over byte buffers)
9. `crates/numpy-rs/` (**Owner: `verifier` for integration & oracle tests; Coordinator integrates final re-exports**):
   - Unified facade crate re-exporting all modules + `prelude` + Python NumPy 2.5.4 differential test suite
10. `crates/numpy-bench/` (**Owner: `perf`**):
    - Criterion benchmarks (`benches/`) + head-to-head Python NumPy 2.5.4 timing & throughput comparator + optimization reports

## Performance & Correctness Standards
- Zero-copy views wherever possible (`shares_storage` verified in tests).
- Never use `unwrap()` in public library code on user-controllable inputs; return `NumPyError`.
- Always benchmark / gate Rayon parallelism so small arrays (< threshold) run single-threaded without pool scheduling overhead.
- Run `cargo test -p <your_crate>` to verify your crate compiles cleanly and passes all unit tests.

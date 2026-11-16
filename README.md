# numpy-rs

[![CI](https://github.com/Khenggg/numpy-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Khenggg/numpy-rs/actions/workflows/ci.yml)

A pure Rust numerical-array workspace with aligned storage, signed-stride
views, broadcasting, ufuncs, linear algebra, Fourier transforms, random
generators and numerical utilities. An optional local `upstream-numpy/`
checkout and Python (`NUMPY_RS_PYTHON` or `.venv`) provide source references
and a NumPy 2.5.4 oracle.

## Using the facade

Enable `full` for all modules, or select individual features (`core`, `ufunc`,
`linalg`, `fft`, `random`, `lib`, `extra`). Default features are `fft` and
`random`. The foundational array types are always exported.

```toml
[dependencies]
numpy-rs = { path = "crates/numpy-rs", features = ["full"] }
```

```rust
use numpy_rs::{NdArray, Result, lib};

fn main() -> Result<()> {
    let a = NdArray::from_shape_vec(&[2, 3], vec![1f64, 2., 3., 4., 5., 6.])?;
    let column_means = lib::mean(&a, Some(0), false)?;
    assert_eq!(column_means.to_vec(), vec![2.5, 3.5, 4.5]);
    Ok(())
}
```

## Modules

| Module | Responsibilities |
|---|---|
| `types` | Array storage, views, slices, dtypes and promotion |
| `core` | Creation, manipulation, indexing, einsum and memory helpers |
| `ufunc` | Elementwise arithmetic/math, broadcasting and ufunc methods |
| `linalg` | Matrix products, solves, decompositions and norms |
| `fft` | Complex/real transforms, normalization and frequency helpers |
| `random` | Seed sequences, bit generators, distributions and RandomState |
| `lib` | Sorting/searching, reductions/statistics, sets, histograms, signal/calculus, polynomials, NPY/NPZ/text I/O |
| `extra` | MaskedArray, date/time scalars and business calendars, UTF-8 string arrays and packed structured records |

`lib` uses typed arrays with compile-time result types. Integer accumulation
widens to i64/u64; floating/complex statistics retain their output dtype and
integer statistics return f64. `lib::dynamic` dispatches selected reductions
and statistics on `DynArray`. Axes are signed, `None` reduces all axes, and
`keepdims` retains reduced dimensions. Quantile functions take a list of q
values; its length is the leading result dimension.

`MaskedArray` keeps its data/mask invariant private. Read them through
`data()`/`mask()` and update through `set_data()`/`set_mask()`, which validate
broadcasting. Masked variance follows NumPy's scalar/array mask behavior,
including the distinction between variance and standard-deviation overflow.

## Verification

```powershell
# Disabling debug symbols substantially reduces this workspace's generic
# ufunc compilation memory usage; numerical behavior is unchanged.
$env:CARGO_PROFILE_DEV_DEBUG = '0'
$env:CARGO_PROFILE_TEST_DEBUG = '0'
cargo test --workspace --all-features --no-fail-fast
cargo check --workspace --all-features --all-targets

# Build release libraries required by standalone probes (and debug artifacts for -IncludeArtifacts):
cargo build --release -p numpy-types -p numpy-core -p numpy-ufunc -p numpy-linalg -p numpy-lib -p numpy-extra
cargo build -p numpy-lib -p numpy-extra

# Focused reproducible semantic and performance checks, after building:
.venv/Scripts/python.exe docs/probes/run_review.py
powershell -ExecutionPolicy Bypass -File docs/probes/bug_audit/run.ps1 -IncludeArtifacts
```

The normal tests include live NumPy differential suites for types, FFT,
random generators, `lib` and `extra`. Focused probes compare NPY versions,
endian/layout combinations, calendar edge cases, and confirmed bug cases.

See [the implementation and validation record](docs/IMPLEMENTATION_STATUS.md),
[NumPy 2.5.4 API coverage and roadmap](docs/API_COVERAGE.md),
[performance measurements](docs/PERFORMANCE_REVIEW.md), and
[the hypothesis/evidence/fix audit](docs/BUG_AUDIT.md).

## Compatibility boundaries

This is a Rust API with a tested compatibility baseline, not complete NumPy
API coverage. In particular:

- Statistics accumulate in f64/complex128 precision before converting to
  their output dtype; float16/float32 rounding can differ from NumPy.
- Quantiles return f64 and support Linear, Lower, Higher, Midpoint and Nearest;
  their return dtype is fixed even for discontinuous methods.
- Histograms, covariance, `trapezoid`, `signal`, and `polynomial` currently
  expose real-valued APIs: `signal` (`convolve`, `correlate`, `interp`, `diff`,
  `gradient`, `unwrap`) and `polynomial` (`poly*`, `Poly1d`, `Polynomial`,
  `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE`, with `roots()`
  returning `Complex64`) are `f64`-only, while `histogram*`, `digitize`,
  `cov`/`corrcoef`, and `trapezoid` reject unsupported complex inputs rather
  than silently discarding imaginary parts.
- `Polynomial` and orthogonal basis series represent raw unmapped ascending
  coefficients (without `domain`/`window` attributes); `Polynomial::fit` fits in
  a `[-1, 1]`-mapped domain for conditioning and converts back to power-series
  coefficients.
- `MaskedArray` always carries an explicit boolean mask (`NdArray<bool>`, no
  `nomask` singleton), and stores `fill_value: T` in the element type `T` itself
  (so narrow integer `MaskedArray<i8>`/`i16`/`u8`/`u16` wrap NumPy's `999999`
  default fill value rather than widening to `i64`/`u64`).
- Numeric NPY v1/v2/v3 and NPZ are supported. Object/pickle and structured NPY
  descriptors are rejected. File functions use the exact supplied path.
  Text I/O handles rectangular real data with a fixed scientific output format.
- Structured records contain packed numeric fields in little-endian byte
  storage. Field access copies into aligned arrays.
- `DateTime64` and `TimeDelta64` are scalar-only (`BusinessCalendar` operates on
  scalar `DateTime64` values; vectorized `NdArray<DateTime64>` is not provided).
  Date/time parsing accepts calendar dates and ISO local times with optional
  `Z`; timezone offsets and conversions between calendar durations (Y/M) and
  fixed durations are not provided. Overflow returns an error instead of
  wrapping timestamps.
- The core, ufunc and linear-algebra Criterion files still contain benchmark
  placeholders; FFT/random benchmarks and standalone review probes are active.

Coverage of additional NumPy options and distributions should be established
with explicit oracle cases before claiming complete parity.

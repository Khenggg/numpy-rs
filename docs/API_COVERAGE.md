# NumPy 2.5.4 API Coverage & Community Roadmap

Generated from direct inventory of `D:\numpy-rs\crates\*\src\**\*.rs` (~51.0k LOC source, 12.6k LOC tests) against Python `numpy==2.5.4`.

---

## 1. Module-Level Parity Summary

| Module / Subsystem | NumPy `2.5.4` Surface | `numpy-rs` Coverage & Status | Primary Limitations / Remaining Gaps |
| :--- | :--- | :--- | :--- |
| **`numpy.fft`** (`numpy-fft`) | 18 public functions | **18 / 18** (`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftshift`, `ifftshift`, `fftfreq`, `rfftfreq`) | 1D `irfft`/`hfft` on `f16` returns `f32` instead of `f16` |
| **`numpy.random.Generator`** (`numpy-random`) | 44 generator methods | **44 / 44** + `BitGenerator`s (`PCG64`, `PCG64DXSM`, `MT19937`, `Philox`, `SFC64`), `SeedSequence`, `par_*` parallel streams | Single-threaded defaults on some `Generator` methods; `shuffle` uses rejection sampling instead of Lemire |
| **`numpy.random.RandomState`** (`numpy-random`) | 49 legacy methods | **49 / 49** | Legacy single-threaded API |
| **`numpy.linalg`** (`numpy-linalg`) | 32 public symbols | **30 / 32** (`matmul`, `dot`, `solve`, `inv`, `lstsq`, `pinv`, `qr`, `svd`, `eig`, `eigh`, `cholesky`, `det`, `slogdet`, `cond`, `norm`, `vector_norm`, `matrix_norm`, `matrix_rank`, `matrix_power`, `tensordot`, `kron`, `cross`, `multi_dot`, `trace`, `tensorsolve`, `tensorinv`, `vdot`, `vecdot`, `matvec`, `vecmat`, `outer`, etc.) | `LinAlgError` lives inside `NumPyError::LinAlgError`; standalone `matrix_transpose` alias maps to `NdArray::t` / `transpose` |
| **Top-level `numpy` Computational APIs** (`numpy-types`, `numpy-core`, `numpy-ufunc`, `numpy-lib`) | 375 non-Python-C-API exports *(out of 496 total `numpy` exports after excluding CPython types/exceptions/`f2py`/`ctypeslib`/`testing`)* | **~328 / 375 (~87%)** matched (including `ascontiguousarray`=`to_c_contiguous`, `asfortranarray`=`to_f_contiguous`, `isfortran`=`is_f_contiguous`, `pow`=`power`, `isnat`=`is_nat`) | **~43 missing top-level functions** (listed in Section 2); `lib::dynamic` currently wraps 16 runtime-dispatched functions; `f80`/`f128` (`longdouble`), `object` dtype not supported |
| **`numpy.polynomial`** (`numpy-lib`) | 6 basis classes + `poly1d` + legacy `poly*` | Power `poly*` (`polyval`, `polyadd`, `polysub`, `polymul`, `polydiv`, `polyder`, `polyint`, `poly`, `roots`, `polyfit`, `poly1d`) + `Polynomial`, `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE` (`evaluate`, `to_polynomial`, `from_polynomial`, `deriv`, `integ`, `roots`) + `Polynomial::{Add, Sub, Mul, fit}` | `f64`-only (except complex `roots()`); no `domain`/`window` attributes on series structs; orthogonal series lack direct `.fit()`, `div`, `pow`, `trim`, `cutdeg` |
| **`numpy.strings`** (`numpy-extra::StringArray`) | 46 methods / helpers | **16 core string operations** (`add`, `center`, `endswith`, `find`, `join`, `ljust`, `lower`, `lstrip`, `replace`, `rjust`, `rstrip`, `split`, `startswith`, `strip`, `upper`, `zfill`) | **21+ missing methods**: `capitalize`, `decode`, `encode`, `expandtabs`, `index`, `isalnum`, `isalpha`, `isdecimal`, `isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper`, `rfind`, `rindex`, `rpartition`, `str_len`, `swapcase`, `title`, `translate`, `partition`, `rsplit`; `split` drops array shape; UTF-8 only (no `bytes_`) |
| **`numpy.ma`** (`numpy-extra::MaskedArray`) | 226 module exports / methods | **19 struct methods** (`new`, getters/setters, `with_fill_value`, `filled`, `compressed`, `count`, `add`, `multiply`, `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `map`) | Explicit `NdArray<bool>` mask only (no `nomask` singleton); `fill_value: T` wraps for narrow integers instead of widening; missing general `MaskedArray` ufunc integration and ~200 `ma` helpers (`masked_where`, `masked_invalid`, `clump_*`, `fix_invalid`, etc.) |
| **DateTime & Structured** (`numpy-extra`) | Full `ndarray[datetime64]`, `ndarray[timedelta64]`, nested `void`/structured dtypes | Scalar `DateTime64`, `TimeDelta64`, `BusinessCalendar`; flat packed little-endian numeric `StructuredArray` | No `NdArray<DateTime64>` / `NdArray<TimeDelta64>`; no timezone offsets; `StructuredArray` has no nested/string/endian fields or zero-copy column views |

---

## 2. Explicit List of Missing Top-Level Computational Functions (~43)

Excluding CPython/typing/testing/deprecated exports (`f2py`, `ctypeslib`, `testing`, `matrix`, `bmat`, `asmatrix`, `errstate`, `from_dlpack`, `frompyfunc`, `show_config`, `sctypeDict`, `.pyi` stubs, scalar type classes):

- **Functional & Higher-Order Helpers (4):** `apply_along_axis`, `apply_over_axes`, `piecewise`, `vectorize`
- **Signal Windows (5):** `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`
- **Index Grids, Broadcasting & Array Placement (12):** `asarray_chkfinite`, `broadcast_arrays`, `diag_indices`, `diag_indices_from`, `ix_`, `mask_indices`, `place`, `require`, `tril_indices`, `tril_indices_from`, `triu_indices`, `triu_indices_from`
- **Bitwise Packing (2):** `packbits`, `unpackbits`
- **Complex & Dtype Predicates (9):** `common_type`, `iscomplex`, `iscomplexobj`, `isdtype`, `isreal`, `isrealobj`, `issubdtype`, `min_scalar_type`, `real_if_close`
- **Text Formatting, Date Metadata & Arbitrary Text/Binary Input (11):** `base_repr`, `binary_repr`, `datetime_as_string`, `datetime_data`, `format_float_positional`, `format_float_scientific`, `fromfile`, `fromregex`, `fromstring`, `genfromtxt`

---

## 3. Architectural & Quality Gaps Toward Production v1.0

1. **Compile Time & Code Size (`numpy-ufunc`):** `numpy-ufunc` (`12,838` LOC, heavy macro expansion across `14×14` dtypes) causes large debug artifact footprints (requiring `debug = 0` in `Cargo.toml`) and long release LTO links (>10 min / >4.5 GB RAM when compiling Criterion benches). Reducing monomorphization via shared strided kernels will make `cargo add numpy-rs` practical for Rust users.
2. **Performance Opportunities Still Unapplied:**
   - Replace `i-k-j` scalar loop and `.to_c_contiguous()` copies in `numpy-core::einsum` with `matrixmultiply` / `faer` GEMM dispatch.
   - Add SIMD kernels for transcendentals (`sin`, `cos`, `exp`, `log`, `tanh`, `pow`, `atan2`), `f16`, complex, and ARM64 NEON fallback (currently `x86_64` AVX2+FMA only).
   - Eliminate unnecessary pre-zeroing in `numpy-linalg::batch::alloc_zeroed_buf` / `numpy-fft` and full-array copy allocations in `numpy-lib::reductions` (`wide_acc!` / `without_nan`).
   - Run full 619-feature `performance_matrix` benchmarks end-to-end (`audit_coverage.py` currently has `0 / 619` result rows).
3. **Safety, Fuzzing & Documentation:**
   - Run `cargo miri test` across all `unsafe` modules (`numpy-types`, `Memmap`, `pad`, `simd`, `linalg`, `fft`, `random`) and add `proptest` / property-based fuzzers against NumPy.
   - Populate Rustdoc + runnable doctests across all 9 crates (currently `0` doctests) and resolve existing `cargo clippy` lints (including `clippy::excessive_precision` in constants and `clippy::reversed_empty_ranges` in `types_tests`).

---

## 4. Recommended Release Roadmap

- **Milestone 1 (Complete — Baseline Audited):** Zero-warning build, `144/144` unit/integration/oracle tests passing, `4,214/4,214` standalone probes passing against NumPy `2.5.4`, dual-OS (`ubuntu-latest` + `windows-latest`) GitHub Actions CI.
- **Milestone 2 (Core Parity & Ergonomics):** Implement the ~43 missing top-level functions, expand `StringArray` to full `np.strings` parity, add complex/`f32` support to `signal`/`polynomial`/`histogram`, add `NdArray<DateTime64>`/`TimeDelta64`, and complete ergonomic Rust operators & doctests.
- **Milestone 3 (Performance & Compile Speed):** Modularize `numpy-ufunc` dispatch to shrink LTO memory/time, implement `einsum` GEMM & transcendental SIMD + NEON, and run full 619-feature `performance_matrix`.
- **Milestone 4 (Community Harden & `0.1.0` Publish):** `proptest` fuzzing, Miri `unsafe` gate in CI, Clippy cleanliness (`-D warnings`), SemVer API audit, and `crates.io` release.

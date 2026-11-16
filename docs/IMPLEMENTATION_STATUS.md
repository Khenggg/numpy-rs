# Implementation and validation record

Date: 2026-11-16. This record describes the work completed in this continuation;
it does not claim exhaustive coverage of NumPy.

## Implemented

- Replaced the `numpy-lib` stub with typed sorting/searching, promoted integer
  reductions and cumulative operations, NaN functions, mean/variance/quantiles,
  weighted averages, covariance/correlation, sets, real histograms,
  signal/calculus, polynomial arithmetic/fitting/bases, and numeric
  NPY/NPZ/text I/O. Selected runtime-dtype APIs are in `lib::dynamic`.
- Replaced the `numpy-extra` stub with validated masked arrays and reductions,
  date/time and duration scalars, business calendars, UTF-8 string arrays and
  packed structured numeric records. Data/mask setters preserve broadcasting
  invariants instead of exposing independently mutable public fields.
- Added feature-gated module exports and a prelude to the `numpy-rs` facade.
- Fixed an existing ufunc borrow-check error in indexed updates, the sign and
  legacy uniform-consumption formula for `RandomState::geometric`, float16 FFT
  promotion to complex64/float32, and scalar FFT-shift rejection.
- Corrected two pre-existing FFT test assumptions using live NumPy evidence:
  NaN sample spacing produces NaN frequencies; a scalar with no FFT axes keeps
  its real dtype in Python, while the typed Rust FFT return type is complex.
- Added normal crate regression tests, `lib_oracle` and `extra_oracle`, plus
  durable standalone performance/semantic and deep-audit probes.
- Wave 2 fixes and test expansion:
  - `numpy-random`: `RandomState::dirichlet` and `Generator::dirichlet` return
    `size + [0]` zeros on empty `alpha` without `chunks_exact_mut(0)` panics;
    `RandomState::multivariate_normal_with` and
    `Generator::multivariate_normal_with` reject empty `mean` (`d == 0`);
    `RandomState::hypergeometric` enforces `ngood.checked_add(nbad)` and the
    shared `HYPERGEOM_MAX` (`1_000_000_000`) bound; `RandomState::shuffle`
    returns an error rather than panicking if `as_slice_mut()` is `None`;
    `RandomState::bit_generator_mut` resets `has_gauss`, `gauss`, and
    `binomial` state.
  - `numpy-lib`: `Polynomial::new` and orthogonal basis constructors normalize
    empty coefficient vectors to `vec![0.0]` (`Polynomial([5.0]).deriv(1)` ->
    `[0.0]`, while legacy `polyder(&[5.0], 1)` retains `vec![]`); added
    `Add`/`Sub`/`Mul` (owned and borrowed variants) and `[-1, 1]`-conditioned
    `Polynomial::fit` (converting back to ascending power coefficients); added
    `from_polynomial`, `.deriv`, `.integ`, and `.roots` on `Chebyshev`,
    `Legendre`, `Laguerre`, `Hermite`, and `HermiteE`; fixed `nanmin`/`nanmax`
    (`nan_extreme`) to preserve the first NaN element on all-NaN lanes
    (preserving `NaN + NaNj` on complex inputs).
  - Expanded `random_oracle`, `lib_oracle`, `extra_oracle`, `lib_tests`, and
    `io_polynomial_tests` to cover all fixed edge cases plus `io` cross-roundtrips,
    `partition`/`argpartition`, `searchsorted`, `lexsort`, `top_k`,
    `histogram2d`, `histogramdd`, `bincount`, `interp`, and `MaskedArray`
    `ddof >= n` variance/std behavior.
  - Round 2 fixes (`numpy-random`, `numpy-lib`, `numpy-extra`):
    - `numpy-random`: added `kappa > 1e6` wrapped-normal fallback and `NaN`/`Inf`
      handling in `vonmises`; fixed `i64::MAX` overflow in `random_binomial_btpe`
      and `random_binomial_inversion`; matched NumPy `Generator` zero-draw
      short-circuit in `binomial_impl`; added `POISSON_LAM_MAX` upper bound check in
      `negative_binomial`; allowed `p == 0.0` in `RandomState::logseries`;
      enforced `nsample >= 1` in `RandomState::hypergeometric`; handled empty
      `colors` (`k == 0, nsample == 0`), `Marginals` `sum(colors) < 1_000_000_000`,
      and `Count` reservation checks in `Generator::multivariate_hypergeometric`;
      allowed `choice`/`choice_range` on `pop_size == 0` with empty `size` and
      `p == None`; rejected 0-D scalars in `Generator::permuted`; propagated `NaN`
      means in `Generator::multivariate_normal_with`.
    - `numpy-lib`: allowed `axis = Some(0 | -1)` on 0-D scalars for `nanmean`/
      `nanvar`/`nanstd` while preserving `StrictAxis` errors on non-NaN and
      quantile/median APIs; validated `[0, 100]` bounds in `percentile`/
      `nanpercentile` before scaling; matched empty-input `nanquantile` shape
      to NumPy's `nanmean` fallback.
    - `numpy-extra`: fixed `DateTime64::add` `NaT` unit-compatibility and
      promotion; enforced NumPy's day-castability rules (`Y`/`M`/`W`/`D` allowed,
      sub-day rejected) in `BusinessCalendar::{new, busday_count, busday_offset}`
      while preserving sub-day `is_busday`; fixed `BusinessCalendar::busday_offset`
      binary search near `i64::MAX`/`i64::MIN`; allowed `axis = Some(0 | -1)`
      for 0-D `MaskedArray` reductions in `MaskedLanes::new`.

The checkout already contained extensive uncommitted implementations before
this continuation. Those changes were preserved; this record does not
attribute all dirty files to this continuation, and no commit was created.

## Validation

```powershell
$env:CARGO_PROFILE_DEV_DEBUG = '0'
$env:CARGO_PROFILE_TEST_DEBUG = '0'
cargo test -p numpy-fft -p numpy-random -p numpy-lib -p numpy-extra
cargo test -p numpy-rs --no-default-features
cargo test -p numpy-rs
cargo test -p numpy-rs --features full --test types_oracle --test fft_oracle --test random_oracle --test lib_oracle --test extra_oracle
```

Domain B crate suites pass with zero compiler warnings across owned crates:
51 tests across `numpy-extra` (8), `numpy-fft` (12), `numpy-lib` (17), and
`numpy-random` (14), plus 41 oracle tests in `numpy-rs` (`types_oracle` 14,
`fft_oracle` 7, `random_oracle` 6, `lib_oracle` 8, `extra_oracle` 6). Debug
symbols were disabled in `[profile.dev]` / `[profile.test]` (`Cargo.toml`) to
keep generic ufunc codegen memory low; release optimization settings were not
changed.

Additional evidence, independently run by the review agents:

| Check | Result |
|---|---|
| Date/time/business-calendar differential corpus | 1,882 / 1,882 pass |
| Python → Rust → Python NPY roundtrips | 604 / 604 pass |
| Masked-array extended differential corpus | 1,728 / 1,728 pass |
| Deep bug audit | 13 proven findings fixed and rechecked; 4 hypotheses disproven in their tested cases |
| NPZ stored/deflated, invalid-name destination preservation, text roundtrips | Normal integration tests pass |

The 4,214 standalone differential cases supplement the 144 normal tests across
the workspace (`--exclude numpy-bench --all-features`); they are separate counts.
The deep-audit artifact checks were run against newly built libraries, not stale
binaries. Detailed methods and commands are in
[PERFORMANCE_REVIEW.md](PERFORMANCE_REVIEW.md), [BUG_AUDIT.md](BUG_AUDIT.md),
and [probes/README.md](probes/README.md).

## Performance changes

Statistics use a contiguous-lane kernel when the reduced axis is last and a
row-streaming accumulator for other contiguous axis layouts. Arbitrary
strided inputs are gathered once into C order. Median and small quantile
requests use selection instead of fully sorting every lane; sorting scratch
storage is reused. Empty outputs return before allocating axis-sized scratch.

The reviewer compared the production variance implementation with a candidate
that uses a local accumulator for contiguous lanes, checked exact result
equality, and measured approximately 1.6–2.6x improvement in the tested
last-axis cases. This candidate is now applied. Results vary with machine
load; this is not a speedup claim for every dtype, size, layout or NumPy call.

## Remaining boundaries

[README.md](../README.md#compatibility-boundaries) documents API/dtype and
format limitations. In particular, the statistics working precision differs
from NumPy float16/float32 accumulation, `signal` and `polynomial` expose
`f64`-only APIs (`roots()` returns `Complex64`, `trapezoid` takes real generic
scalars and returns `f64`), `Polynomial` and orthogonal basis series represent
raw unmapped ascending coefficients (`Polynomial::fit` maps to `[-1, 1]`
internally and converts back), `DateTime64`/`TimeDelta64` are scalar-only,
`MaskedArray` always carries an explicit boolean mask (no `nomask` singleton)
and stores typed `fill_value: T` (narrow integers wrap `999999` into `T`),
several generic utilities reject complex inputs, and three existing Criterion
suites remain placeholders. Full NumPy parity has not been established outside
the stated corpora.

`numpy-types` now enforces byte-span bounds at array construction via
`AlignedBuffer::check_shape_bytes`; lane consumers also retain their own
dimension checks, and valid enormous empty dimensions return before scratch
allocation.

## Graph impact check

The required codebase-memory indexing, architecture, symbol/path discovery
and post-edit change detection were used. Repeated fast/full indexing returned
the original 3,811 nodes despite the new untracked modules; change detection
also omitted untracked files and returned an empty impacted-symbol list.
That output is insufficient to conclude a small blast radius. Exact source
inspection, full-workspace/all-target compilation, oracle tests and the
independent probes provide the actual validation evidence here.

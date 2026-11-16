# Deep bug audit

Audit date: 2026-11-16. Oracle: local NumPy 2.5.4. Production source is owned
by the implementation agent; this auditor writes only this report and probes
under `target/bug-audit/`.

Statuses: **PROVEN** = actual Rust output differs from a checked contract or
oracle, **FIXED** = the reproducer passed against the edited source,
**DISPROVEN** = the proposed failure did not reproduce in the stated cases,
**UNVERIFIED** = no sufficient runtime evidence yet. A static suspicion alone
is never marked proven.

| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| B01 | `cov` keeps a one-row matrix untransposed with `rowvar=false` | FIXED | `[[1,2,3]]` originally returned scalar `1`; NumPy returns a 3x3 NaN matrix. `[1,0]` originally returned scalar NaN; NumPy returns `[0,0]`. Recompiled source probe now matches both shapes and values. |
| B02 | Scalar reduction-axis exception leaks into `unwrap` | FIXED | `unwrap(array(3.), axis=0)` originally returned scalar `3`; NumPy raises ValueError. Recompiled source probe now returns `InvalidAxis`. Error category differs but both reject this unsupported axis. |
| B03 | Scalar reduction-axis exception leaks into `trapezoid` | FIXED | `trapezoid(array(3.), axis=0)` originally returned scalar `0`; NumPy raises IndexError. Recompiled source probe now returns `InvalidAxis`. |
| B04 | NPY reader accepts malformed Boolean/tuple header values | FIXED | Header `fortran_order: Falsex` and `shape: (1)` each originally returned I64 `[7]`; NumPy rejects both. The strict parser now rejects both; rebuilt-artifact assertions passed. |
| B05 | NPY reader resolves duplicate keys inconsistently | FIXED | Header contains `shape: (1,)` followed by `shape: (2,)`, payload has one I64. Rust originally read the first key; now it uses the last key and rejects the truncated two-element payload like NumPy. Rebuilt-artifact assertion passed. |
| B06 | Legacy polynomial add/subtract incorrectly trim leading zero coefficients | FIXED | `polyadd([1,0],[-1,2])` originally returned `[2]`, oracle `[0,2]`; `polysub([1,0],[1,2])` originally returned `[-2]`, oracle `[0,-2]`. Both now preserve leading coefficient zeros. Rebuilt-artifact assertions passed. |
| B07 | Public masked-array fields allow a scalar mask to trigger panic/wrong counts | FIXED | Assigning a scalar false mask originally produced count `1` and an indexing panic for three data values. Fields are now private; `set_mask(false)` broadcasts, count is `3`, mean is `2`, and an invalid setter returns Err without changing state. Rebuilt-artifact assertions passed. |
| B08 | Quantile integer interpolation differs at integer subtraction overflow | FIXED | For `[i64::MIN,i64::MAX]`, q `[0.,.5,1.]`, Linear/Midpoint, Rust midpoint originally was `0`; NumPy midpoint is approximately `9.223372036854776e18`. Source-dtype wrapping subtraction now matches NumPy. This matches its overflow semantics, not a claim that its result is mathematically preferable. Durable source assertion passed. |
| B09 | `digitize` silently discards complex imaginary components | FIXED | Input `[1+2j]`, bins `[0,2]` originally returned `[1]`; NumPy rejects complex x. The Rust real-valued API now rejects complex input explicitly. Durable source assertion passed. |
| B10 | Complex histogram inputs can be converted to real without changing counts | FIXED with explicit API limit | Input `[1+2j,1+3j]`, inferred 2 bins originally gave Rust `[0,2]`, NumPy `[1,1]` with complex edges. The f64-edge Rust API now rejects unsupported complex input instead of silently converting it. Durable source assertion passed. |
| B11 | Zero-sized shapes can carry out-of-range dimensions into signed-index arithmetic | FIXED at construction & consumer boundaries | `numpy-types` (`AlignedBuffer::check_shape_bytes`) now rejects `[0, isize::MAX]` and `[0, isize::MAX + 1]` at array construction with `"array is too big"`, matching NumPy 2.5.4; `Lanes` also retains dimension and signed `kth` checks. `zero_shape.rs` passes with overflow checks enabled. |
| B12 | Empty sorting outputs still allocate enormous per-axis scratch buffers | FIXED | `argsort(zeros([0, max_f64_cols]), axis=1)` (`max_f64_cols = (isize::MAX - 64) / 8`) and `argsort(zeros([0, isize::MAX - 64]: u8), axis=1)` return empty arrays immediately without allocating per-axis index/scratch buffers. `zero_shape.rs` passes with overflow checks enabled. |
| B13 | Weighted uniform-bin interpolation can move a histogram boundary by one ULP | FIXED | Input `x=(-4.)*.8+9.*.2`, exact bits `bff6666666666667`, 5 bins/range `(-4,9)`: Rust originally counted bin 1, NumPy bin 0. Uniform edge calculation now follows the finite-step linspace order; durable source assertion and live oracle passed. |
| D01 | NaN histogram edges cause an indexing panic or wrong simple-case count | DISPROVEN | Input `[1,2]`, edges `[0,NaN,3]` returns `[2,0]` in both implementations. This single case does not establish all non-monotonic/NaN-edge behavior. |
| D02 | Complex input is always rejected or incorrectly histogrammed after real conversion | DISPROVEN for stated case, superseded by B10 | In the original probe, `[1+2j]` with two inferred bins gave `[0,1]` in both implementations (NumPy emitting `ComplexWarning`). Subsequently per `B10`, the real-valued Rust `histogram` and `histogram_bin_edges` APIs now explicitly reject complex (`Complex32`/`Complex64`) inputs rather than silently discarding imaginary parts. |
| D03 | Single-value `loadtxt` loses scalar shape | DISPROVEN on current artifact | File containing `7\n` loads as shape `[]` in current Rust artifact and NumPy. |
| D04 | Complex enhanced sort order separates signed zeros or misorders NaN categories | DISPROVEN for 49 combinations | Stable argsort matches NumPy for the Cartesian product of real/imaginary `[-inf,-1,-0,+0,+1,+inf,NaN]`. Durable exact-source probe and live oracle passed. |

## Reproduction

Durable, assertion-based reproducers now live in `docs/probes/bug_audit/`:

```powershell
& docs/probes/bug_audit/run.ps1
# After rebuilding the owning debug crates:
& docs/probes/bug_audit/run.ps1 -IncludeArtifacts
```

Both commands passed (`zero_shape.rs`, `numeric_fixes.rs`, `complex_order.rs`,
`histogram_boundary.rs`, plus `io_polynomial.rs` and `masked.rs` under
`-IncludeArtifacts`, and `oracle.py` against live NumPy 2.5.4). All thirteen
proven findings have a passing reproducer after correction; four tested
suspicions were disproven within their stated cases. This audit does not assert
complete NumPy compatibility.

The first source probe (`docs/probes/bug_audit/numeric_fixes.rs`) includes the
actual `lanes`, `sorting`, `statistics`, `histogram` and `signal` modules,
rather than copied implementations:

```powershell
$auditTypes = Get-ChildItem target/release/deps -Filter 'libnumpy_types*.rlib' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
rustc --edition=2024 -Awarnings -O -C lto=thin docs/probes/bug_audit/numeric_fixes.rs `
    --extern "numpy_types=$auditTypes" -L dependency=target/release/deps `
    -o target/bug-audit/numeric_fixes.exe
target/bug-audit/numeric_fixes.exe
.venv/Scripts/python.exe docs/probes/bug_audit/oracle.py
```

`io_polynomial.rs` and `masked.rs` link the newest debug `.rlib` selected by
modification time with `rustc --extern ... -L dependency=target/debug/deps`.
The NPY probe creates `badbool.npy`, `badtuple.npy`, and `dupkey.npy` locally;
the oracle loads the exact same bytes.

After a fix, source-inclusion probes are recompiled. Artifact-linked probes
must be rebuilt only after the owning crate's new artifact exists. Existing
test success alone does not close an audit finding if it did not cover the
reproducer.

`docs/probes/bug_audit/zero_shape.rs` constructs only zero-element arrays and is
compiled with `-C overflow-checks=yes` in addition to the source-probe flags
above: it verifies `NdArray::<f64>::zeros` returns `Err` for oversized spans
(`[0, isize::MAX]` and `[0, isize::MAX + 1]`) and verifies `argsort` and
`partition` fast paths return empty outputs immediately on legal huge
zero-element shapes without attempting large allocations.

## Discovery limitations

The initial codebase-memory index predates these modules. `get_architecture`
reports the old stubs, `search_graph` finds no new audited symbols, and
`trace_path(read_npy)` returns function not found. Exact source files were read
as the required fallback. The coordinator should refresh the graph after the
edits and run `detect_changes`; untracked new files are not fully represented
by the old change detector.

## Round 2 Domain B findings (fixed and verified)

All 17 Round 2 bugs identified across `numpy-random`, `numpy-lib`, and
`numpy-extra` have been addressed and covered with live NumPy 2.5.4 oracle
tests (`random_oracle`, `lib_oracle`, `extra_oracle`):

1. `vonmises` (`distributions.rs`, `generator.rs`, `random_state.rs`): added
   NumPy's `kappa > 1e6` wrapped-normal fallback in `random_vonmises`, prevented
   `kappa = inf` hangs in `legacy_vonmises`, and aligned `NaN`/`Inf` handling on
   `mu`/`kappa` (`kappa < 0` check only).
2. `random_binomial_btpe` / `random_binomial_inversion` (`distributions.rs`):
   computed `(n as f64) + 1.0` instead of `(n + 1) as f64`, avoiding signed
   overflow at `n = i64::MAX` (NumPy's C `n + 1` wraps/UBs at `i64::MAX` for
   `btpe`, whereas Rust avoids overflow/panic and matches non-overflowing BTPE
   as well as NumPy at `p = 0.5` and inversion at `n = i64::MAX`).
3. `binomial_impl` (`distributions.rs`): short-circuited `n == 0 || p == 0.0`
   when `!legacy_stirling` before consuming RNG state, matching `Generator`.
4. `negative_binomial` (`generator.rs`, `random_state.rs`): enforced
   `((1.0 - p) / p) * (n + 10.0 * n.sqrt()) <= POISSON_LAM_MAX`.
5. `RandomState::logseries` (`random_state.rs`): allowed `p == 0.0` (`p < 0.0`
   rejected) returning `1`.
6. `RandomState::hypergeometric` (`random_state.rs`): enforced `nsample >= 1`
   while preserving `nsample == 0` support in `Generator::hypergeometric`.
7. `Generator::multivariate_hypergeometric` (`generator.rs`): handled empty
   `colors` with `nsample == 0` (`size + [0]` output), enforced `sum(colors) <
   1_000_000_000` for `Marginals`, and guarded `Count` with `max_index` +
   `try_reserve_exact`.
8. `choice_range` / `choice` (`generator.rs`, `random_state.rs`): allowed
   `pop_size == 0` when `out_total == 0` and `p.is_none()`.
9. `Generator::permuted` (`generator.rs`): rejected 0-D scalar inputs.
10. `Generator::multivariate_normal_with` (`generator.rs`): allowed `NaN` in
    `mean` to propagate rather than rejecting non-finite `mean`.
11. `nanmean`/`nanvar`/`nanstd` (`statistics.rs`): allowed `axis = Some(0 | -1)`
    on 0-D scalars while keeping `strict_axis` errors on non-NaN and
    quantile/median functions.
12. `percentile`/`nanpercentile`/`nanquantile` (`statistics.rs`): checked `[0,
    100]` bounds before dividing by `100.0`, and aligned empty-input
    `nanquantile` output shape to `nanmean` (`&p.reduced_shape`).
13. `DateTime64::add` (`datetime.rs`): promoted compatible units on `NaT`
    operands and rejected incompatible fixed + calendar unit combinations.
14. `BusinessCalendar` (`datetime.rs`): rejected sub-day units (`< Day`) in
    `new`, `busday_count`, and `busday_offset` while keeping `is_busday` valid
    on sub-day timestamps.
15. `BusinessCalendar::busday_offset` (`datetime.rs`): clamped binary-search
    upper bound to valid `i64` bounds and counted in `i128` to avoid spurious
    overflow near `i64::MAX`/`i64::MIN`.
16. `MaskedLanes::new` (`masked.rs`): allowed `axis = Some(0 | -1)` for 0-D
    `MaskedArray` reductions (`mean`, `var`, `std`, `min`, `max`).
17. `MaskedArray<T>::fill_value: T` (`masked.rs`, `README.md`,
    `IMPLEMENTATION_STATUS.md`): documented the typed `fill_value: T` design
    constraint (narrow integers wrap `999999` into `T`, covered by
    `extra_tests::masked_broadcasts_and_integer_sums_widen` and
    `extra_oracle::round2_datetime_busday_and_masked_scalar_axis_against_numpy`).

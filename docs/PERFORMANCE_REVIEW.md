# Performance and correctness review

Reviewed 2026-11-16 by the performance reviewer. This review does not claim complete NumPy compatibility. The local oracle is NumPy 2.5.4 at `.venv/Scripts/python.exe`.

## Current completion priorities

The code graph and exact sources show substantial implementations in `numpy-core`, `numpy-ufunc`, `numpy-linalg`, `numpy-fft`, and `numpy-random`. At the start of this review, `numpy-lib`, `numpy-extra`, and the `numpy-rs` facade only re-exported `numpy-types`. Common reductions/statistics/sorting in `numpy-lib`, followed by namespaced facade exports, are useful next steps. Specialized datetime, strings, and masked-array support remains a separate workstream.

`numpy-ufunc::higher_order` already implements reduction/accumulation, integer widening, contiguous SIMD paths, and strided fallback. Reusing those paths for sum/product/extrema avoids another independent implementation. Typed same-dtype reductions and dynamic NumPy-style promotion should be documented separately. A statistics API that always returns f64 is useful, but differs from NumPy float16/float32 statistics, which retain their dtype.

## Measured axis traversal

A standalone Rust probe compares two **algorithm kernels**, not public crate APIs:

* Gather: visit each output lane, then traverse the reduction dimension.
* Stream: visit successive contiguous input rows and update the corresponding output slice.

Both use the same left-to-right summation order for each output and assert exact equality. They allocate an output vector per invocation. Inputs and dimensions pass through `std::hint::black_box`. Each reported value is the median of seven samples, each running for at least 50 ms. No Cargo build was needed while the implementation agent ran its baseline tests.

Environment: Intel Core i5-10300H, 4 cores / 8 logical processors; rustc 1.99.1; `rustc -O`; no target-cpu override. Other work was running on the machine, so these are local directional measurements, not stable release performance claims.

| Layout (`outer`, `reduction`, `inner`) | Gather, µs | Stream, µs | Gather / stream |
| --- | ---: | ---: | ---: |
| 1, 64, 64 | 5.428 | 1.921 | 2.83 |
| 1, 1024, 1024 | 10,781.260 | 1,088.622 | 9.90 |
| 32, 64, 512 | 2,899.061 | 1,127.833 | 2.57 |
| 1, 1,000,000, 1 | 1,627.174 | 3,779.921 | 0.43 |
| 1024, 1024, 1 | 2,458.690 | 4,331.317 | 0.57 |

Recommendation: split contiguous reductions into `inner == 1` (fold each contiguous lane) and `inner > 1` (stream contiguous rows into output accumulators). For the latter, use slice iteration so LLVM can vectorize across independent output elements without reassociating each sum. Variance can use two streaming passes over each slab. Avoid allocating one Vec per sum/mean lane and avoid coordinate division/modulo inside the element loop.

For arbitrary strides, keep an allocation-free cursor after initial setup. A materialized C-order copy may pay off for repeated operations, but should be measured against direct traversal and should not be imposed on zero-copy consumers.

Reproduce the probe while it remains in the build directory:

```powershell
rustc -O target/perf-review/reduction_probe.rs -o target/perf-review/reduction_probe.exe
& target/perf-review/reduction_probe.exe
```

The probe is intentionally under ignored `target/`; production benchmarks should cover the actual public APIs with Criterion before setting regression thresholds. Large arrays should include contiguous, transposed, negative-stride, and broadcast views.

### Verified optimization in the new statistics implementation

Once `statistics.rs` existed, a standalone harness compiled the exact current production `lanes`, `sorting`, and `statistics` modules with `rustc -O -C lto=thin -C codegen-units=1`, linking the pre-existing release `numpy-types` artifact. A candidate copied only `statistics.rs` into the ignored build directory and added an `inner == 1` branch to the variance second pass. The branch computes each contiguous lane with a local mean and scalar sum; the existing streaming kernel remains for `inner > 1`. No production source was edited by the reviewer.

The comparison ran both implementations in the same binary, alternating measurement order across nine sample pairs of at least 70 ms each. Every output was checked for exact equality first. The speedup column is the median of paired ratios; timing columns are separate medians.

| Shape | Axis | Current variance, µs | Candidate, µs | Paired speedup |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64 | 0 | 5.234 | 4.847 | 1.11 |
| 64 × 64 | 1 | 23.640 | 10.043 | 2.61 |
| 1024 × 1024 | 0 | 2,202.558 | 2,070.029 | 1.06 |
| 1024 × 1024 | 1 | 5,723.062 | 2,766.519 | 2.00 |
| 32 × 64 × 512 | 0 | 2,474.307 | 2,557.436 | 1.00 |
| 32 × 64 × 512 | 1 | 2,369.337 | 2,695.096 | 0.98 |
| 32 × 64 × 512 | 2 | 5,853.225 | 2,843.588 | 2.05 |

The improvement is specific to contiguous reduction lanes; the other axes show machine-load noise. A second run of the frozen comparison measured speedups 2.08×, 1.64×, and 1.73× for the same three contiguous-lane cases; unchanged axes ranged 0.94–1.04×. The observed improvement across both runs is approximately 1.6–2.6×, with substantial absolute-timing variance under concurrent machine load.

The implementation agent adopted the `inner == 1` branch in production `statistics.rs` after this comparison. Frozen baseline/candidate source snapshots and the comparison harness live temporarily at `target/perf-review/statistics_baseline.rs`, `target/perf-review/statistics_candidate.rs`, and `target/perf-review/compare_variance.rs`. The harness uses those snapshots, so later production edits do not replace its baseline. This is algorithm-level evidence, not an end-to-end release benchmark or a guarantee of absolute timings.

## Correctness gates from the local oracle

### Extrema and ordering

* `min([0., -0.])` and `max([0., -0.])` return -0.; reversing input returns +0. Equal-value extrema select the later value in the tested scalar cases. SIMD regrouping can change signed-zero behavior, so preserve it deliberately or document the difference.
* `argmin` and `argmax` select the first equal value, and select the first NaN when NaNs occur. `argmin([1., NaN, 2.]) == argmax(...) == 1`.
* Ascending sort, partition, and searchsorted place NaN after finite values and infinities. Ordinary ±0 compare equal. Rust `total_cmp` alone differs because it distinguishes signed zero and orders negative NaNs separately.
* Integer comparison must stay integer-based. Converting to f64 misorders distinct int64/uint64 values beyond 2^53.
* Stable argsort is valuable for ties. If the API exposes a different sorting strategy, document its stability contract.

### Empty reductions and dimensions

* Empty sum has identity 0; product has identity 1. Empty min/max/argmin/argmax fail only when a reduced dimension is empty. Empty mean/variance/median yield NaN (NumPy also emits warnings). Ordinary quantile with an empty reduced dimension raises an error, including q=[]; empty non-reduced dimensions still produce valid empty outputs.
* For shape `(2, 0, 3)`, reduction along axis 0 or 2 has a valid empty output for min/max/argmin/argmax. Reduction along axis 1 fails for extrema. Do not reject solely because the total array length is zero.
* Scalar sum accepts `axis=None`, `axis=0`, and `axis=-1`.
* Scalar axis rules vary by API: min/max/argmin/argmax also accept 0/-1, whereas mean/variance/median reject an explicit 0/-1 axis. Scalar cumsum/cumprod produce a length-one vector for None/0/-1.
* A NumPy-compatible sum initializes with positive zero: `sum([-0.])` is +0.; product preserves the sign.
* The variance denominator is `max(N - ddof, 0)`. Thus `var([1.,3.], ddof=2)` and ddof=3 return +inf, while zero numerator divided by zero yields NaN. Returning NaN unconditionally whenever ddof >= N differs from NumPy.

### Partition and quantiles

* Negative kth values are normalized relative to lane length: -1 is the last and -n the first. Empty partition returns an empty array even for out-of-range scalar kth values (tested -99/-1/0/1/99).
* Scalar argsort/argpartition accept axis 0/-1 and produce a length-one index array. Scalar sort/partition reject an explicit axis; flattening axis=None is accepted.
* A single kth partition should use selection (`select_nth_unstable_by`) rather than sorting the entire lane. For one linear quantile, select the lower neighbor then find/select the adjacent upper value. This targets expected O(n) rather than O(n log n). For many quantiles, sorting once may be faster; benchmark the crossover.
* A partition is not required to be fully sorted. Tests should verify the kth value, ordering on either side, and preservation of values rather than exact unstable permutation.
* Any NaN in a lane causes an ordinary quantile to be NaN; sorting NaN to the end is insufficient unless the implementation also checks it. nanquantile must discard NaNs and use the remaining count.
* Upstream `_lerp` computes `d = b - a`, then `a + d*t` for t < 0.5 and `b - d*(1-t)` otherwise. The second branch reduces cancellation near the upper endpoint. Inf behavior is a direct consequence: quantile([1., inf], .25) is inf, but at .5/.75 it is NaN. Quantile([inf, inf], q) is NaN at the tested q=0/.5/1.
* Floating q uses upstream neighbor-index conventions as well as interpolation: for linear/midpoint the upper neighbor is `min(floor(h)+1,n-1)`, not `ceil(h)`. Thus quantile([1., inf], [0.,1.]) gives [NaN,NaN]. A Python integer q can take a different exact-index path: quantile([1., inf], 0) is 1 and quantile(...,1) is inf. Compare the Rust `&[f64]` API against floating/list q, not Python integer q.
* Median is not interchangeable with quantile(.5) at overflow: median([1e308, 1e308]) is inf, while quantile([1e308, 1e308], .5) is 1e308.
* NumPy accepts bool inputs for lower/higher/nearest quantile methods, but rejects bool interpolation in linear/midpoint. A real-numeric-only quantile API can choose to reject bool uniformly if documented.
* Complex median preserves the NaN-selected element, including its imaginary component: median([NaN+1j,1+2j]) is NaN+1j. Returning NaN+0j differs. All-NaN nanmedian also preserves a selected NaN element (the tested [NaN+1j,NaN+5j] returns NaN+5j).

### Histogram and set operations

* Explicit histogram edges may contain duplicates (zero-width bins), but must not decrease. All bins are left-inclusive/right-exclusive except the last bin, which includes its right endpoint. Values outside the edge range and NaN do not count.
* Uniform-bin direct indexing needs correction against the actual bin edges near floating-point boundaries. Arbitrary edges naturally use binary search, yielding O(n log b) instead of checking all b bins per value.
* Sort-based unique/set operations should compare integer values in their native dtype. Keep original positions while sorting if return_index/return_inverse are exposed. NaN equalness differs across operations, so one shared equality predicate is insufficient.
* NumPy `unique` defaults to equal_nan=True; its Array API-style `unique_all`, `unique_counts`, `unique_inverse`, and `unique_values` use equal_nan=False. Thus unique_all([NaN,NaN]) has two values/counts [1,1], while unique([NaN,NaN]) has one value.
* For complex unique with equal_nan=True, the representative follows enhanced sort ordering. `unique([NaN+2j,1+NaNj], return_index=True)` selects the representative 1+NaNj and index 1, rather than the earliest original position among all NaN variants.
* Increasing digitize maps NaN to bins.len() for both right=False and right=True; decreasing digitize maps NaN to zero.

### Signal operations

* Unwrap must propagate a NaN correction to subsequent elements: unwrap([0.,NaN,1.]) is [0.,NaN,NaN]. A conditional that only updates correction for abs(delta)>=threshold incorrectly skips NaN deltas.
* A generic trapezoid API must preserve complex components or reject complex input explicitly; casting every input to f64 silently loses information. NumPy trapezoid([1+2j,3+4j]) is 2+3j.

## Benchmark acceptance

Keep the correctness corpus fixed while comparing algorithms. Include size 0/1, small arrays, cache-sized arrays, and memory-sized arrays; at least one reduction on every axis; NaN-heavy and duplicate-heavy sorting inputs; values above 2^53; and noncontiguous arrays. Warm the Rayon pool outside timing and compare serial versus parallel on the same layouts before selecting thresholds. Do not enable Rayon solely from a generic element-count constant: lane count and contiguous work per task also matter.

## Follow-up review: I/O and extra subsystems

Standalone probes are now preserved under `docs/probes/`; their README documents requirements and commands. They compile current production modules directly and reuse existing dependencies without launching Cargo. Generated artifacts remain under ignored `target/perf-review/`.

### NPY interoperability

The reader passed 600 Python-generated cases across 25 dtype/endian variants, four shapes (scalar, vector, matrix, and zero dimension), three NPY versions, and C/F layouts. The extended **Python → Rust → Python roundtrip passed 604 cases** with exact shape, dtype kind/itemsize, values, and floating/complex sign bits. Extra cases included int64/uint64 extrema, NaN, infinities and signed zero. The current writer emits little-endian C-order NPY, while the reader preserves the logical values of big-endian and Fortran-order inputs.

Source review also checked bounded reads, checked payload-size arithmetic, descriptor rejection, and aligned scalar decoding. This review does not claim coverage of every malformed Python header literal. NPZ and text I/O still need their separate integration checks. The implementation agent corrected NPZ name validation before destination creation and the scalar shape of a one-value text file.

### Dates, durations and business calendars

After fixes, **1,882 differential cases passed with zero mismatches**. They include 1,764 business offsets across six roll modes, default/custom holiday calendars, negative and positive offsets through 100 days; fixed/calendar unit casts and BCE formatting; reverse business counts; finer-unit duration promotion; subtraction before downcast; cross-unit equality; and NaT inequality.

The initial review found and the implementation agent corrected reversed-count endpoint handling, loss of finer duration units during addition, timestamp truncation before subtraction, NaT equality, and BCE formatting. Negative fixed-unit casts and week alignment matched the oracle. Checked overflow errors are intentional; this corpus does not establish compatibility with every extreme-value overflow behavior.

### Masked arrays and strings

Source review found and the implementation agent corrected wrapped small-integer fill values and Python whitespace U+001C–U+001F. Variable-width UTF-8 case conversion should be compared against NumPy's `StringDType`, because fixed-width Unicode arrays can truncate expanded case mappings.

The extended masked probe contains 1,728 real-valued cases across scalar/matrix/empty layouts, masks, reduction axes, NaN, infinity, overflow, and degrees of freedom. Its initial run found 82 mismatches concentrated in shape-dependent masked mean/variance rules. The owning upstream source explains why scalar and array outputs differ; `docs/probes/MASKED_ALGORITHM.md` records the full algorithm and regression vectors. After the implementation agent's parity fixes, **all 1,728 cases passed**.

### Final standalone acceptance

The complete durable driver was rerun against the final formatted production source, including the stricter NPY header parser and updated masked algorithms. It exited successfully with **4,214 differential cases and zero mismatches**: 1,882 datetime/business-calendar cases, 604 NPY read/write roundtrips, and 1,728 masked-array cases. The standalone kernel probe also passed exact equality for all five measured layouts. Generated machine-readable results are written to `target/perf-review/review_results.json` on each complete run.

The baseline statistics layer deliberately uses f64/complex128 working precision. Retaining a float32/complex64 output dtype therefore does not promise bitwise NumPy float32 accumulation parity. Public crates and their normal integration/oracle tests still need to pass independently of these standalone probes.

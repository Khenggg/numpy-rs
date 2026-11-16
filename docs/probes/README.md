# Standalone review probes

Run from the repository with its NumPy Python environment:

```powershell
.\.venv\Scripts\python.exe docs/probes/run_review.py
```

Select a smaller review with `--only datetime`, `--only npy`, `--only masked`, or `--only kernels`. `--root PATH` selects another checkout. Generated fixtures, binaries, and `review_results.json` go under that checkout's ignored `target/perf-review/` directory.

The script never launches Cargo or edits production source. Semantic probes compile the current production Rust modules directly and reuse existing `numpy-types` and `zip` dependency artifacts from `target/release/deps` (preferred) or `target/debug/deps`. Run a normal workspace build first if those dependencies are missing. This avoids recompiling all ufunc dispatch instantiations during a focused review. The kernel benchmark only uses Rust's standard library.

`datetime_probe.rs` checks fixed-unit casts, BCE formatting, business-day offsets with six roll modes and two holiday calendars, reversed business counts, duration promotion, exact subtraction before downcast, cross-unit equality, and NaT inequality against live NumPy. The original extended suite contains 1,882 cases.

`io_probe.rs` reads Python-generated NPY files with the production reader and rewrites them with the production writer. The Python driver loads those outputs and checks shape, dtype kind/size, exact values, and floating/complex sign bits. The 604 cases cover 25 endian/dtype variants, scalar/vector/matrix/zero shapes, C/F order, NPY versions 1/2/3, integer extrema, NaN, infinity, and signed zero. This probes NPY; it does not replace NPZ or text-I/O integration tests.

`reduction_probe.rs` compares contiguous row streaming with strided column gathering. It is an algorithm microbenchmark, not a release benchmark of public crate APIs. It verifies exact equality first and reports seven-sample median timings. Concurrent machine load affects timings.

`masked_probe.rs` checks 1,728 real-valued cases against `numpy.ma`: scalar/matrix/empty shapes, five reductions, masks, NaN/infinity/overflow, and degrees of freedom. It compares shape, mask and unmasked values, and requires an existing build of the implemented `numpy-lib` crate. See `MASKED_ALGORITHM.md` for NumPy's scalar/array differences. A mismatch makes the driver exit with status 1; the driver does not silently whitelist current behavior.

The probes intentionally include production source by a relative `#[path]`, so changes to that source are exercised on the next run. They complement the workspace's normal tests; they do not substitute for building and testing the real crates.

The final review run passed all 4,214 differential cases (1,882 datetime, 604 NPY roundtrips, 1,728 masked arrays) with zero mismatches, plus exact-equality checks in the five-layout kernel probe. This records the reviewed state; running the driver again checks the current checkout.

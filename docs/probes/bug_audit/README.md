# Reproducible deep audit probes

Run from the repository after building the required release/debug crates:

```powershell
cargo build --release -p numpy-types
powershell -ExecutionPolicy Bypass -File docs/probes/bug_audit/run.ps1

cargo build -p numpy-lib -p numpy-extra
powershell -ExecutionPolicy Bypass -File docs/probes/bug_audit/run.ps1 -IncludeArtifacts
```

The first command compiles exact current production modules using the newest
release `numpy-types` artifact (by modification time). The second also checks
binary I/O, polynomial APIs and masked-array setters against the newest debug
`numpy-lib` / `numpy-extra` library artifacts (`-IncludeArtifacts`). Note that
`zero_shape.rs` verifies `NdArray::<f64>::zeros` rejects out-of-range
byte-span zero-element shapes (`[0, isize::MAX]` and `[0, isize::MAX + 1]`)
and tests `argsort`/`partition` fast paths on legal huge empty shapes.
Both run the local NumPy oracle. They never launch cargo or overwrite source.
Generated binaries and malformed NPY fixtures stay under ignored target/.

The assertions describe the corrected behavior. Historical failing output,
the hypothesis and its verification status are recorded in BUG_AUDIT.md.
Rejecting complex histogram inputs is the supported real-valued Rust API's
explicit limitation; it prevents silent conversion to incorrect real edges.

#!/usr/bin/env python3
"""Read-only benchmark coverage audit; never runs cargo or timing workloads.

The denominator is the repository's NumPy-facing operation surface, not NumPy's
complete API. The source inventory is conservative lexical extraction, including
the public-function macros used by this workspace; unresolved macros are listed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
GROUPS = ("types", "core", "ufunc", "linalg", "fft", "random", "lib", "extra")
PASS_STATUSES = {"pass", "passed", "ok", "verified", "exact", "tolerance", "invariant", "sampled_pass"}
ALIASES = {
    "abs": "absolute", "divide": "true_divide", "modulo": "remainder", "mod": "remainder",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2",
    "asinh": "arcsinh", "acosh": "arccosh", "atanh": "arctanh",
    "radians": "deg2rad", "degrees": "rad2deg", "fix": "trunc", "around": "round",
    "conjugate": "conj", "bitwise_not": "invert", "concat": "concatenate", "t": "transpose",
}
TYPE_OPERATIONS = {
    "zeros", "zeros_with_order", "ones", "full", "full_with_order", "slice", "transpose", "t",
    "reshape", "ravel", "flatten", "swapaxes", "moveaxis", "rollaxis", "squeeze", "expand_dims",
    "broadcast_to", "flip", "to_c_contiguous", "to_f_contiguous", "copy", "map", "astype",
    "promote_types", "result_type", "can_cast", "finfo", "iinfo", "broadcast_shapes", "broadcast_strides",
    "broadcast_many_shapes", "check_non_overlapping", "normalize_axis", "normalize_axes", "normalize_index",
    "as_slice", "as_slice_mut", "view", "view_mut", "to_vec", "get", "get_mut", "set", "item",
    "iter", "iter_mut", "get_flat", "set_flat", "shares_storage", "is_c_contiguous", "is_f_contiguous",
}
TRIVIAL_METHODS = {
    "new", "from_vec", "from_shape_vec", "from_shape_vec_order", "from_aligned_buffer", "from_storage",
    "from_scalar", "shape", "strides", "ndim", "len", "is_empty", "dtype", "itemsize", "nbytes", "offset",
    "storage_ptr", "as_ptr", "as_mut_ptr", "is_unique", "layout", "is_contiguous", "data", "mask", "fields",
    "coefficients", "as_bytes", "value", "unit", "is_nat", "nat", "with_fill_value", "set_mask", "set_data",
}


def mask_comments_strings(source: str) -> str:
    """Preserve offsets/newlines while hiding comments, quoted/raw strings, chars."""
    pattern = re.compile(r'//[^\n]*|/\*.*?\*/|(?:br|r)(?P<hashes>\#*)".*?"(?P=hashes)|b?"(?:\\.|[^"\\])*"|b?\'(?:\\.|[^\'\\])\'', re.S)
    return pattern.sub(lambda m: "".join("\n" if c == "\n" else " " for c in m.group()), source)


def matching(text: str, opening: int) -> int:
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack = [pairs[text[opening]]]
    for i in range(opening + 1, len(text)):
        char = text[i]
        if char in pairs:
            stack.append(pairs[char])
        elif char in ")}]":
            if not stack or char != stack.pop():
                return -1
            if not stack:
                return i
    return -1


def function_declarations(path: Path) -> tuple[list[dict], list[dict]]:
    source = path.read_text(encoding="utf-8")
    text = mask_comments_strings(source)
    macros = []
    for match in re.finditer(r"\bmacro_rules\s*!\s*(\w+)\s*\{", text):
        opening = text.find("{", match.start())
        end = matching(text, opening)
        if end >= 0:
            macros.append((match.group(1), match.start(), end + 1, text[opening + 1:end]))
    clean = list(text)
    for _, start, end, _ in macros:
        clean[start:end] = " " * (end - start)
    clean = "".join(clean)
    impls = []
    for match in re.finditer(r"\bimpl\b", clean):
        opening = clean.find("{", match.end())
        if opening < 0:
            continue
        header = clean[match.end():opening].strip()
        # Exclude impl Fn(...) argument bounds and trait implementations.
        if ";" in header or " for " in header or header.startswith(("Fn(", "FnMut(", "FnOnce(")):
            continue
        if header.startswith("<"):
            depth = 0
            for i, char in enumerate(header):
                depth += (char == "<") - (char == ">")
                if depth == 0:
                    header = header[i + 1:].lstrip()
                    break
        owner = re.match(r"([A-Za-z_]\w*)", header)
        end = matching(clean, opening)
        if owner and end >= 0:
            impls.append((opening, end, owner.group(1)))
    rows = []
    public_fn = re.compile(r"\bpub\s+(?:(?:const|async|unsafe)\s+)*fn\s+((?:r#)?[A-Za-z_]\w*)")
    for match in public_fn.finditer(clean):
        owners = [(end - start, owner) for start, end, owner in impls if start < match.start() < end]
        owner = min(owners)[1] if owners else None
        rows.append({"name": match.group(1).removeprefix("r#"), "owner": owner,
                     "line": source.count("\n", 0, match.start()) + 1, "macro": None})
    unresolved = []
    for macro, _, _, body in macros:
        if not public_fn.search(body) and not re.search(r"pub\s+fn\s+\$", body):
            continue
        placeholder_names = re.findall(r"pub\s+(?:(?:const|async|unsafe)\s+)*fn\s+\$(\w+)", body)
        literal_names = public_fn.findall(body)
        owner_placeholder = re.search(r"impl\s+\$(\w+)", body)
        invocations = list(re.finditer(r"\b" + re.escape(macro) + r"\s*!\s*\(", clean))
        for invocation in invocations:
            opening = clean.find("(", invocation.start())
            end = matching(clean, opening)
            if end < 0:
                unresolved.append({"macro": macro, "reason": "unbalanced invocation"})
                continue
            arguments = clean[opening + 1:end]
            first = re.match(r"\s*((?:r#)?[A-Za-z_]\w*)", arguments)
            if not first:
                unresolved.append({"macro": macro, "line": source.count("\n", 0, invocation.start()) + 1,
                                   "reason": "public-function macro arguments require manual review"})
                continue
            names = [first.group(1)]
            # The stats_dispatch repetition accepts a flat list of function names.
            if re.search(r"\$\s*\(\s*\$\w+\s*:\s*ident\s*\)\s*,\s*\*", body):
                names = re.findall(r"\b[A-Za-z_]\w*\b", arguments)
            if placeholder_names:
                for name in names:
                    rows.append({"name": name.removeprefix("r#"), "owner": None,
                                 "line": source.count("\n", 0, invocation.start()) + 1, "macro": macro})
            elif literal_names and owner_placeholder:
                for name in literal_names:
                    rows.append({"name": name.removeprefix("r#"), "owner": first.group(1),
                                 "line": source.count("\n", 0, invocation.start()) + 1, "macro": macro})
            else:
                unresolved.append({"macro": macro, "reason": "public literal methods cannot be attributed automatically"})
    # Simple public aliases remain explicit surface entries and share one feature.
    for match in re.finditer(r"\bpub\s+use\s+((?:r#)?\w+)\s+as\s+((?:r#)?\w+)\s*;", clean):
        rows.append({"name": match.group(2).removeprefix("r#"), "owner": None,
                     "line": source.count("\n", 0, match.start()) + 1, "macro": None,
                     "alias_of": match.group(1).removeprefix("r#")})
    return rows, unresolved


def classify(group: str, module: str, row: dict) -> tuple[str, str, str | None]:
    """Return canonical group, feature, exclusion reason. Rules are explicit."""
    name, owner = row["name"], row["owner"]
    feature = re.sub(r"^dyn_|_dyn$", "", name)
    if group in {"ufunc", "core", "types"}:
        feature = ALIASES.get(feature, feature)
    if module in {"exec", "scalar_ops", "simd", "lanes", "util", "traits", "batch"}:
        if not (group == "ufunc" and module == "traits"):
            return group, feature, "Implementation kernel/trait/helper surface; no independent NumPy operation claim."
    if name.startswith(("u_", "ua_", "ho_")):
        return group, feature, "Scalar/dispatch implementation method, not a separately timed NumPy-facing operation."
    if group == "ufunc" and owner in {"ScalarValue", "BinaryOp", "UnaryOp"} and name in {"cast_to", "dtype", "name", "identity", "is_reorderable"}:
        return group, feature, "Ufunc descriptor/scalar metadata helper; not a numerical-operation timing."
    if group == "ufunc" and feature.endswith(("_nd", "_with")):
        base = feature.removesuffix("_nd").removesuffix("_with")
        if base in {"reduce", "reduce_axes", "accumulate", "outer", "at", "unary_at"}:
            feature = base
    if group == "core":
        feature = feature.removesuffix("_with_order")
        feature = {"arange_stop": "arange", "linspace_retstep": "linspace", "linspace_arrays": "linspace",
                   "indices_dense": "indices", "indices_sparse": "indices", "transpose_isize": "transpose",
                   "einsum_with_optimize": "einsum", "einsum_path_shapes": "einsum_path", "roll_axes": "roll",
                   "memmap_typed": "memmap", "array2string_default": "array2string"}.get(feature, feature)
        if owner and name in TRIVIAL_METHODS | {"parse", "from_str_code", "mode", "order", "path", "is_writable", "byte_offset", "byte_bounds"}:
            return group, f"{owner}.{feature}", "Memory/format enum getter or representation helper; no standalone NumPy operation claim."
    if group == "types":
        if name not in TYPE_OPERATIONS:
            return group, feature, "Rust storage/metadata/construction helper; not included in operation denominator."
        feature = re.sub(r"_with_order$", "", feature)
    if group == "random":
        if owner == "RandomState":
            group = "random.legacy"
        elif owner == "Generator" or name in {"default_rng", "default_rng_from_seed"}:
            pass
        elif owner == "SeedSequence":
            mapping = {"generate_state_u32": "seed_state_u32", "generate_state_u64": "seed_state_u64", "spawn": "spawn"}
            if name not in mapping:
                return "random.bitgen", f"SeedSequence.{name}", "SeedSequence construction/state accessor; not a bulk-generation operation."
            return "random.bitgen", mapping[name], None
        elif owner in {"PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"}:
            if name in {"next_u64", "next_u32", "next_double", "random_raw"}:
                return "random.bitgen", owner, None
            if name in {"advance", "jumped"} and owner in {"PCG64", "PCG64DXSM"}:
                return "random.bitgen", "pcg_" + name, None
            return "random.bitgen", f"{owner}.{name}", "Bit-generator constructor/state accessor; listed separately from timed raw generation."
        else:
            return group, feature, "Low-level distribution/bit-generator helper rather than Generator/RandomState API."
    if group == "extra":
        if module == "masked":
            group = "extra.masked"
        elif module == "strings":
            group = "extra.strings"
            if name == "map":
                return group, feature, "Generic Rust closure helper; NumPy-facing string operations are inventoried individually."
        elif module == "records":
            group = "extra.records"
            if name in {"zeros", "from_bytes", "field", "set_field"}:
                return group, feature, None
        elif module == "datetime":
            group = "extra.datetime"
            if owner == "TimeDelta64":
                feature = "timedelta_" + name
            if name in {"parse", "cast", "add", "difference", "is_busday", "busday_count", "busday_offset"}:
                return group, feature, None
    if group == "fft" and module == "planner":
        return group, feature, "Planner/norm support API; public transform timing does not independently measure this helper."
    if owner and name in TRIVIAL_METHODS:
        return group, f"{owner}.{feature}", "Constructor/getter/setter or representation helper; not a numerical-operation benchmark."
    if group == "lib" and owner:
        feature = f"{owner}.{feature}"
    if group == "random" and owner == "Generator" and name in {"bit_generator", "bit_generator_mut", "into_bit_generator"}:
        return group, feature, "Generator accessor, not an independently timed distribution."
    return group, feature, None


def inventory(root: Path) -> dict:
    surfaces, unresolved, sources = [], [], []
    private_modules = {"ufunc": {"exec", "scalar_ops"}, "fft": {"engine"},
                       "random": {"distributions_tables", "mt19937_constants", "ziggurat_constants"}, "lib": {"lanes"}}
    for group in GROUPS:
        directory = root / "crates" / ("numpy-" + group) / "src"
        for path in sorted(directory.glob("*.rs")):
            if path.stem == "lib" or path.stem in private_modules.get(group, set()):
                continue
            relative = path.relative_to(root).as_posix()
            sources.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            functions, pending = function_declarations(path)
            unresolved.extend({"source": relative, **entry} for entry in pending)
            for row in functions:
                category, feature, reason = classify(group, path.stem, row)
                surfaces.append({**row, "source": relative, "module": path.stem,
                                 "surface": f"{group}.{path.stem}." + ((row["owner"] + ".") if row["owner"] else "") + row["name"],
                                 "group": category, "feature": feature, "excluded_reason": reason})
            if group == "random":
                text = mask_comments_strings(path.read_text(encoding="utf-8"))
                # Public trait implementations do not spell their methods `pub`.
                for match in re.finditer(r"\bimpl\s+BitGenerator\s+for\s+(PCG64DXSM|PCG64|MT19937|Philox|SFC64)\s*\{", text):
                    owner = match.group(1)
                    surfaces.append({"name": "next_u64", "owner": owner, "source": relative,
                                     "line": text.count("\n",0,match.start())+1, "module": path.stem,
                                     "macro": None, "surface": f"random.{path.stem}.{owner}.BitGenerator",
                                     "group": "random.bitgen", "feature": owner, "excluded_reason": None})
    # Display is implemented through fmt::Display rather than a pub fn.
    surfaces.append({"name": "fmt", "owner": "DateTime64", "source": "crates/numpy-extra/src/datetime.rs", "line": None,
                     "module": "datetime", "macro": None, "surface": "extra.datetime.DateTime64.Display",
                     "group": "extra.datetime", "feature": "format", "excluded_reason": None})
    return {"method": "conservative_source_inventory_v1", "scope": "NumPy-facing operations implemented in this workspace; not all NumPy",
            "surfaces": surfaces, "unresolved_macros": unresolved, "sources": sources}


def rows_from_json(value, *, manifest=False) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in (("cases", "manifest", "specs") if manifest else ("rows", "results", "cases")):
            if isinstance(value.get(key), list):
                return value[key]
        if all(isinstance(v, dict) for v in value.values()):
            return [{"id": k, **v} for k, v in value.items()]
    raise ValueError("Expected a list or an object containing cases/rows/results.")


def result_quality(row: dict) -> tuple[bool, str]:
    verification = row.get("verification", {})
    status = str(verification.get("status", "")).lower() if isinstance(verification, dict) else str(verification).lower()
    if status not in PASS_STATUSES:
        return False, "verification_not_passed: " + (status or "missing")
    for name in ("rust", "numpy"):
        stats = row.get(name)
        if not isinstance(stats, dict):
            return False, f"missing_{name}_timing"
        median = stats.get("median_ns")
        if not isinstance(median, (float, int)) or not math.isfinite(median) or median <= 0:
            return False, f"invalid_{name}_median"
        raw = stats.get("samples_ns")
        if not isinstance(raw, list) or len(raw) < 2:
            return False, f"missing_{name}_raw_samples"
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in raw):
            return False, f"invalid_{name}_raw_samples"
    return True, "timed_with_recorded_validation"


def audit(inv: dict, manifest: list[dict], results: list[dict], modes: list[str]) -> dict:
    issues, notes = [], []
    specs = {}
    for spec in manifest:
        if not isinstance(spec, dict) or not isinstance(spec.get("id"), str):
            issues.append("Manifest item lacks string id."); continue
        if spec["id"] in specs:
            issues.append("Duplicate manifest id: " + spec["id"])
        specs[spec["id"]] = spec
        missing = [k for k in ("group", "feature", "n", "shape", "dtype", "layout", "numpy_api", "notes", "equivalence") if k not in spec]
        if missing:
            issues.append(f"{spec['id']}: missing spec fields {','.join(missing)}")
    by_case = defaultdict(list)
    seen = set()
    for row in results:
        spec = row.get("spec", row)
        identity = spec.get("id", row.get("id"))
        mode = row.get("mode", "unspecified")
        if identity not in specs:
            issues.append(f"Result id not registered in manifest: {identity}")
            continue
        if (identity, mode) in seen:
            issues.append(f"Duplicate result: {identity} / {mode}")
        seen.add((identity, mode))
        by_case[identity].append(row)
        if mode not in modes:
            notes.append(f"Result {identity} uses mode {mode}; expected profiles {modes}.")
        for key in ("group", "feature", "n", "shape", "dtype", "layout"):
            if key in spec and spec[key] != specs[identity].get(key):
                issues.append(f"Result spec drift: {identity} field {key}")
    surface_features = defaultdict(list)
    excluded = []
    for surface in inv["surfaces"]:
        if surface["excluded_reason"]:
            excluded.append(surface)
        else:
            surface_features[(surface["group"], surface["feature"])].append(surface)
    registered = defaultdict(list)
    for spec in specs.values():
        registered[(spec.get("group"), spec.get("feature"))].append(spec)
    features = []
    all_keys = sorted(set(surface_features) | set(registered), key=lambda x: (str(x[0]), str(x[1])))
    for key in all_keys:
        cases = registered.get(key, [])
        case_results = []
        missing_profiles = []
        for spec in cases:
            outputs = by_case.get(spec["id"], [])
            for mode in modes:
                matches = [r for r in outputs if r.get("mode") == mode]
                if not matches:
                    missing_profiles.append({"id": spec["id"], "mode": mode, "reason": "no_result_record"})
                for result in matches:
                    valid, reason = result_quality(result)
                    case_results.append({"id": spec["id"], "mode": mode, "valid": valid, "reason": reason,
                                         "verification": result.get("verification"), "equivalence": spec.get("equivalence")})
        passed = sum(r["valid"] for r in case_results)
        expected = len(cases) * len(modes)
        if not cases:
            status, reason = "unregistered", "No exact group/feature case registered; no performance measurement claimed."
        elif not case_results:
            status, reason = "not_measured", "Cases are registered but no expected-profile result exists."
        elif passed == expected and not missing_profiles:
            status, reason = "measured", "All registered cases/profiles have both timings and recorded validation."
        else:
            status, reason = "partial", "Some registered cases/profiles are missing or have invalid/failed verification."
        features.append({"group": key[0], "feature": key[1], "status": status, "reason": reason,
                         "source_backed": key in surface_features, "surfaces": surface_features.get(key, []),
                         "case_ids": [s["id"] for s in cases], "expected_profile_results": expected,
                         "valid_profile_results": passed, "results": case_results, "missing_profiles": missing_profiles,
                         "dtype_cases": sorted({str(s.get("dtype")) for s in cases}),
                         "layout_cases": sorted({str(s.get("layout")) for s in cases}),
                         "sizes": sorted({int(s["n"]) for s in cases if isinstance(s.get("n"), int)})})
    groups = {}
    for group in sorted({f["group"] for f in features}):
        entries = [f for f in features if f["group"] == group]
        groups[group] = {"features": len(entries), **dict(Counter(f["status"] for f in entries))}
    return {"scope": inv["scope"], "method": inv["method"], "expected_modes": modes,
            "summary": {"manifest_cases": len(specs), "result_rows": len(results), "features": len(features),
                        "source_backed_features": len(surface_features), "excluded_surface_entries": len(excluded),
                        "unresolved_macro_patterns": len(inv["unresolved_macros"]),
                        **dict(Counter(f["status"] for f in features))},
            "groups": groups, "features": features, "excluded_surfaces": excluded,
            "unresolved_macros": inv["unresolved_macros"], "issues": sorted(set(issues)), "notes": sorted(set(notes)),
            "limitations": ["A measured feature means its registered cases only, not every dtype/layout/argument option.",
                            "Validation strength is inherited from each result's verification and equivalence metadata.",
                            "Lexical inventory is not a compiler-expanded Rust API inventory; unresolved macros remain explicit.",
                            "Aliases and typed/dynamic entry points may share a canonical feature; this is not variant-complete coverage.",
                            "Helper exclusions are explicit and remain reviewable in excluded_surfaces."]}


def write_markdown(report: dict, destination: Path) -> None:
    summary = report["summary"]
    lines = ["# Performance coverage audit", "", report["scope"], "",
             f"Registered cases: {summary['manifest_cases']}; result rows: {summary['result_rows']}; "
             f"canonical features: {summary['features']}; fully measured registered features: {summary.get('measured', 0)}.", "",
             "A measured feature covers its registered cases, not all input sizes, dtypes, layouts or options.", "",
             "| Group | Features | Measured | Partial | Not measured | Unregistered |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, data in report["groups"].items():
        lines.append(f"| {group} | {data['features']} | {data.get('measured',0)} | {data.get('partial',0)} | {data.get('not_measured',0)} | {data.get('unregistered',0)} |")
    lines += ["", "## Features without complete measurements", "", "| Feature | Status | Reason |", "|---|---|---|"]
    for feature in report["features"]:
        if feature["status"] != "measured":
            lines.append(f"| {feature['group']}.{feature['feature']} | {feature['status']} | {feature['reason']} |")
    lines += ["", "## Inventory limits", ""] + ["- " + note for note in report["limitations"]]
    if report["issues"]:
        lines += ["", "## Manifest/result issues", ""] + ["- " + issue for issue in report["issues"]]
    if report["unresolved_macros"]:
        lines += ["", "## Macro patterns needing manual review", ""]
        lines += [f"- {m['source']}: {m['macro']}: {m['reason']}" for m in report["unresolved_macros"]]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.rs"
        path.write_text('''// pub fn fake() {}\nimpl<T> Thing<T> { pub fn run(&self) {} }\npub fn real() {}\nmacro_rules! function { ($name:ident) => {pub fn $name() {}}; }\nfunction!(generated);\n''', encoding="utf-8")
        declarations, pending = function_declarations(path)
        assert {(r["owner"], r["name"]) for r in declarations} == {("Thing", "run"), (None, "real"), (None, "generated")}
        assert not pending
    good = {"verification": {"status": "pass"}, "rust": {"median_ns": 3., "samples_ns": [2.,4.]}, "numpy": {"median_ns": 6., "samples_ns": [5.,7.]}}
    assert result_quality(good)[0]
    assert not result_quality({**good,"verification":{"status":"failed"}})[0]
    assert not result_quality({**good,"rust":{"median_ns":0,"samples_ns":[1,2]}})[0]
    print("PASS coverage parser and result-quality self-tests")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs/performance-comparison")
    parser.add_argument("--modes", nargs="+", default=["single_thread", "native"])
    parser.add_argument("--strict", action="store_true", help="Fail on schema/drift/duplicate issues; missing feature cases remain explicit.")
    parser.add_argument("--require-all-features", action="store_true", help="Also fail unless every inventoried feature is measured.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    inv = inventory(args.repo.resolve())
    manifest = rows_from_json(json.loads(args.manifest.read_text(encoding="utf-8-sig")), manifest=True) if args.manifest else []
    results = []
    for path in args.results:
        results.extend(rows_from_json(json.loads(path.read_text(encoding="utf-8-sig"))))
    report = audit(inv, manifest, results, args.modes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feature_inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    (args.out_dir / "coverage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_dir / "COVERAGE_AUDIT.md")
    print(json.dumps(report["summary"], sort_keys=True))
    if args.strict and report["issues"]:
        return 1
    if args.require_all_features and any(f["status"] != "measured" for f in report["features"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

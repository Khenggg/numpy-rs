"""Repeat standalone reviews without compiling numpy-ufunc's full dispatch table.

Run with the repository's NumPy Python environment. Existing Cargo dependency
artifacts are reused; this script never launches Cargo or edits production code.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np


_SKIP_WALK_DIRS = frozenset({".fingerprint", "examples", "fingerprint", "incremental", "run"})
_DEP_SUFFIXES = frozenset({".rlib", ".rmeta", ".so", ".dll", ".dylib"})


def _profile_dir_info(profile_dir: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    """Collect `-L dependency=...` search dirs and `.rlib` candidates across Cargo <= 1.99
    (`deps/`) and Cargo >= 1.100 (`build/<pkg>/<meta>/out/`) layouts without launching Cargo
    or adding `<profile>/` root (which contains unhashed uplift `.rlib`s causing E0464)."""
    dep_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    rlibs: dict[str, list[Path]] = {}
    if not profile_dir.is_dir():
        return dep_dirs, rlibs
    for dirpath, dirnames, filenames in os.walk(profile_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_WALK_DIRS]
        current = Path(dirpath)
        if current == profile_dir:
            continue
        has_dep_artifact = False
        for fname in filenames:
            path = current / fname
            suffix = path.suffix
            if suffix in _DEP_SUFFIXES:
                has_dep_artifact = True
            if suffix == ".rlib" and fname.startswith("lib"):
                stem = path.stem[3:]
                crate = stem.rsplit("-", 1)[0] if "-" in stem else stem
                rlibs.setdefault(crate, []).append(path)
        if has_dep_artifact and current not in seen_dirs:
            seen_dirs.add(current)
            dep_dirs.append(current)
    return dep_dirs, rlibs


def _candidate_profiles(root: Path) -> list[tuple[str, list[Path], dict[str, list[Path]]]]:
    build_root = Path(
        os.environ.get("CARGO_BUILD_BUILD_DIR")
        or os.environ.get("CARGO_TARGET_DIR")
        or (root / "target")
    )
    profiles: list[tuple[str, list[Path], dict[str, list[Path]]]] = []
    for name in ("release", "debug"):
        dep_dirs, rlibs = _profile_dir_info(build_root / name)
        if dep_dirs or rlibs:
            profiles.append((name, dep_dirs, rlibs))
    return profiles


def latest_rlib(rlibs: dict[str, list[Path]], library: str, min_size: int = 0) -> Path | None:
    matches = [p for p in rlibs.get(library, ()) if p.stat().st_size >= min_size]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _dep_args(output: Path, name: str, dep_dirs: list[Path]) -> list[str]:
    """Write `-L dependency=...` lines to a BOM-free response file so Windows never hits 32 KB cmd limits."""
    if not dep_dirs:
        return []
    rsp = output / f"{name}.deps.rsp"
    lines = [line for d in dep_dirs for line in ("-L", f"dependency={d}")]
    rsp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [f"@{rsp}"]


def build(root: Path, output: Path, name: str, needs_zip: bool = False, needs_masked: bool = False) -> Path:
    profiles = _candidate_profiles(root)
    if needs_masked:
        required = ["numpy_types", "numpy_core", "numpy_ufunc", "numpy_linalg", "numpy_lib", "numpy_extra"]
        choices = [
            (prof_name, dep_dirs, rlibs)
            for prof_name, dep_dirs, rlibs in profiles
            if all(
                latest_rlib(rlibs, lib, min_size=100_000 if lib == "numpy_lib" else 0) is not None
                for lib in required
            )
        ]
        if not choices:
            raise RuntimeError(
                "Masked probe needs a complete release or debug build of numpy-types, numpy-core, "
                "numpy-ufunc, numpy-linalg, numpy-lib, and numpy-extra"
            )
        binary = output / (name + (".exe" if sys.platform == "win32" else ""))
        last_err: subprocess.CalledProcessError | None = None
        for prof_name, dep_dirs, rlibs in choices:
            release = prof_name == "release"
            base_cmd = ["rustc", "-O", "--edition", "2024", "-A", "dead_code"]
            if release:
                base_cmd += ["-C", "lto=thin", "-C", "codegen-units=1"]
            dep_flags = _dep_args(output, name, dep_dirs)
            lib_candidates = sorted(
                (p for p in rlibs.get("numpy_lib", ()) if p.stat().st_size >= 100_000),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            ufunc_all = rlibs.get("numpy_ufunc", [])
            types_all = rlibs.get("numpy_types", [])
            for lib_art in lib_candidates:
                lib_mtime = lib_art.stat().st_mtime
                by_proximity = lambda p: (
                    p.stat().st_mtime <= lib_mtime,
                    -abs(p.stat().st_mtime - lib_mtime),
                    p.stat().st_mtime,
                )
                ufunc_candidates = sorted(ufunc_all, key=by_proximity, reverse=True)
                types_candidates = sorted(types_all, key=by_proximity, reverse=True)
                for ufunc_art in ufunc_candidates:
                    for types_art in types_candidates:
                        command = (
                            base_cmd
                            + [
                                "--extern", f"numpy_types={types_art}",
                                "--extern", f"numpy_lib={lib_art}",
                                "--extern", f"numpy_ufunc={ufunc_art}",
                            ]
                            + dep_flags
                            + [
                                str(root / "docs" / "probes" / f"{name}.rs"),
                                "-o", str(binary),
                            ]
                        )
                        try:
                            subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
                            return binary
                        except subprocess.CalledProcessError as exc:
                            last_err = exc
        assert last_err is not None
        sys.stderr.write(last_err.stderr)
        raise last_err
    else:
        externs = ["numpy_types"] + (["zip"] if needs_zip else [])
        choices = [
            (prof_name, dep_dirs, rlibs)
            for prof_name, dep_dirs, rlibs in profiles
            if all(latest_rlib(rlibs, lib) is not None for lib in externs)
        ]
        if not choices:
            raise RuntimeError(
                f"Missing required artifact(s) ({', '.join(externs)}) in release or debug target artifacts; "
                "build workspace dependencies first"
            )
        prof_name, dep_dirs, rlibs = choices[0]

    release = prof_name == "release"
    command = ["rustc", "-O", "--edition", "2024", "-A", "dead_code"]
    if release:
        command += ["-C", "lto=thin", "-C", "codegen-units=1"]
    for library in externs:
        artifact = latest_rlib(rlibs, library)
        if artifact is None:
            raise RuntimeError(f"Missing {library} artifact in {prof_name} build; build workspace dependencies first")
        command += ["--extern", f"{library}={artifact}"]
    binary = output / (name + (".exe" if sys.platform == "win32" else ""))
    dep_flags = _dep_args(output, name, dep_dirs)
    command += dep_flags + [str(root / "docs" / "probes" / f"{name}.rs"), "-o", str(binary)]
    subprocess.run(command, cwd=root, check=True)
    return binary


def datetimes(root: Path, output: Path) -> dict:
    binary = build(root, output, "datetime_probe")
    commands, expected = [], []

    def add(command: str, value: object) -> None:
        commands.append(command)
        expected.append(str(value))

    for day in np.arange(np.datetime64("2024-03-20"), np.datetime64("2024-04-10")):
        for offset, roll, calendar in itertools.product(
            [-100, -10, -1, 0, 1, 10, 100],
            ["raise", "nat", "forward", "backward", "modifiedfollowing", "modifiedpreceding"],
            ["default", "holiday"],
        ):
            try:
                value = np.busday_offset(day, offset, roll=roll, holidays=(
                    ["2024-03-29", "2024-04-01", "1970-01-01"] if calendar == "holiday" else []
                ))
            except (ValueError, OverflowError):
                value = "ERROR"
            add(f"offset;{day};{offset};{roll};{calendar}", value)
    for date in ["-0001-12-31", "0000-01-01", "1582-10-10", "1900-02-28", "1969-12-31T23:59:59.999999999", "1970-01-01T00:00:00.000000001", "2000-02-29"]:
        source = "ns" if "T" in date else "D"
        for unit in ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"]:
            try:
                value = np.datetime64(date, source).astype(f"datetime64[{unit}]")
            except (ValueError, OverflowError):
                value = "ERROR"
            add(f"cast;{date};{source};{unit}", value)
    days = np.arange(np.datetime64("2024-03-28"), np.datetime64("2024-04-03"))
    for begin, end in itertools.product(days, repeat=2):
        add(f"count;{begin};{end}", np.busday_count(begin, end))
    units = {"Y": "Year", "M": "Month", "W": "Week", "D": "Day", "h": "Hour", "m": "Minute", "s": "Second", "ms": "Millisecond", "us": "Microsecond", "ns": "Nanosecond"}
    for a, au, b, bu in [(1, "D", 1, "h"), (-1, "h", 1, "D"), (1, "Y", 1, "M"), (1, "M", 1, "Y")]:
        value = np.timedelta64(a, au) + np.timedelta64(b, bu)
        unit = np.datetime_data(value.dtype)[0]
        add(f"durationadd;{a};{au};{b};{bu}", f"{value.astype('i8')};{units[unit]}")
    for begin, end in [("1970-01-02T00", "1970-01-01T23"), ("1970-01-01T23", "1970-01-02T00")]:
        value = (np.datetime64(begin) - np.datetime64(end)).astype("timedelta64[D]").astype("i8")
        add(f"difference;{begin};h;{end};h;D", value)
    for prefix, constructor in [("equaldate", np.datetime64), ("equaldelta", np.timedelta64)]:
        for a, au, b, bu in [(1, "D", 24, "h"), (1, "D", 23, "h"), (-(2**63), "D", -(2**63), "D")]:
            add(f"{prefix};{a};{au};{b};{bu}", str(bool(constructor(a, au) == constructor(b, bu))).lower())
    run = subprocess.run([str(binary)], input="\n".join(commands) + "\n", text=True, capture_output=True, check=True)
    actual = run.stdout.splitlines()
    bad = [(c, e, a) for c, e, a in zip(commands, expected, actual) if e != a]
    if len(actual) != len(expected):
        bad.append(("output count", len(expected), len(actual)))
    return {"cases": len(commands), "mismatches": bad}


def npy_roundtrip(root: Path, output: Path) -> dict:
    binary = build(root, output, "io_probe", needs_zip=True)
    fixtures = output / "io-cases"
    fixtures.mkdir(exist_ok=True)
    files = []
    dtypes = ["?", "i1", "u1"] + [endian + code for code in ["i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c8", "c16"] for endian in ["<", ">"]]
    for dtype, shape, version, order in itertools.product(dtypes, [(), (3,), (2, 3), (2, 0, 3)], [(1, 0), (2, 0), (3, 0)], ["C", "F"]):
        values = np.arange(np.prod(shape, dtype=int))
        if np.dtype(dtype).kind == "c":
            values = values + 1j * (values + 0.5)
        array = np.array(values, dtype=dtype).reshape(shape, order=order)
        if order == "F" and array.ndim:
            array = np.asfortranarray(array)
        path = fixtures / f"{len(files)}.npy"
        with path.open("wb") as stream:
            np.lib.format.write_array(stream, array, version=version, allow_pickle=False)
        files.append(path)
    specials = [np.array([np.iinfo("i8").min, np.iinfo("i8").max], dtype=">i8"), np.array([0, np.iinfo("u8").max], dtype="<u8"), np.array([np.nan, np.inf, -np.inf, -0.0, 0.0]), np.array([complex(np.nan, 1), complex(1, np.nan), complex(-0.0, -0.0)], dtype=">c16")]
    for i, array in enumerate(specials):
        path = fixtures / f"special{i}.npy"
        np.save(path, array)
        files.append(path)
    bad = []
    for start in range(0, len(files), 60):
        chunk = files[start:start + 60]
        run = subprocess.run([str(binary)] + [str(p) for p in chunk], text=True, capture_output=True, check=True)
        if " ERROR " in run.stdout:
            bad.extend(line for line in run.stdout.splitlines() if " ERROR " in line)
    for path in files:
        a, b = np.load(path), np.load(path.with_suffix(".rust.npy"))
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind or a.dtype.itemsize != b.dtype.itemsize or not np.array_equal(a, b, equal_nan=True):
            bad.append(str(path))
        if a.dtype.kind in "fc" and not np.array_equal(np.signbit(a.real), np.signbit(b.real)):
            bad.append(str(path) + " real sign bits")
        if a.dtype.kind == "c" and not np.array_equal(np.signbit(a.imag), np.signbit(b.imag)):
            bad.append(str(path) + " imaginary sign bits")
    return {"cases": len(files), "mismatches": bad}


def masked_arrays(root: Path, output: Path) -> dict:
    binary = build(root, output, "masked_probe", needs_masked=True)
    commands, expected = [], []
    patterns = [[1., 2., 3.], [np.nan, 1., 2.], [np.inf, 1., 2.], [1e200, -1e200, 0.], [-0., 0., 1.], [np.nan, np.nan, np.nan]]
    masks = [[False], [True], [False, True], [True, False, False]]
    for shape, pattern, mask_pattern in itertools.product([(), (2, 3), (2, 0, 3)], patterns, masks):
        size = int(np.prod(shape))
        data = np.resize(np.array(pattern), size).reshape(shape)
        mask = np.resize(np.array(mask_pattern), size).reshape(shape)
        array = np.ma.array(data, mask=mask)
        for operation, axis in itertools.product(["mean", "var", "std", "min", "max"], [None] + list(range(len(shape)))):
            for ddof in ([0, 1, 3] if operation in ["var", "std"] else [0]):
                try:
                    value = getattr(array, operation)(axis=axis, **({"ddof": ddof} if operation in ["var", "std"] else {}))
                    result = (list(value.shape), np.ma.getmaskarray(value).ravel().tolist(), np.asarray(np.ma.getdata(value)).ravel().tolist())
                except (ValueError, IndexError, ZeroDivisionError):
                    result = "ERROR"
                commands.append(";".join([",".join(map(str, shape)), ",".join(map(str, data.ravel())), ",".join("1" if x else "0" for x in mask.ravel()), operation, "none" if axis is None else str(axis), str(ddof)]))
                expected.append(result)
    run = subprocess.run([str(binary)], input="\n".join(commands) + "\n", capture_output=True, text=True, check=True)
    lines = run.stdout.splitlines()
    bad = []
    for command, wanted, line in zip(commands, expected, lines):
        if line == "ERROR":
            actual = line
        else:
            shape, mask, values = line.split(";")
            actual = (json.loads(shape), json.loads(mask), [float(x) for x in values.strip("[]").split(",") if x.strip()])
        if isinstance(wanted, str) or isinstance(actual, str):
            matches = wanted == actual
        else:
            matches = wanted[:2] == actual[:2] and all(mask or np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True) for mask, a, b in zip(wanted[1], wanted[2], actual[2]))
        if not matches:
            bad.append((command, wanted, actual))
    if len(lines) != len(expected):
        bad.append(("output count", len(expected), len(lines)))
    return {"cases": len(commands), "mismatches": bad}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--only", choices=["all", "datetime", "npy", "masked", "kernels"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "target" / "perf-review"
    output.mkdir(parents=True, exist_ok=True)
    results = {"numpy": np.__version__}
    warnings.simplefilter("ignore", RuntimeWarning)
    if args.only in ("all", "datetime"):
        results["datetime"] = datetimes(root, output)
    if args.only in ("all", "npy"):
        results["npy_roundtrip"] = npy_roundtrip(root, output)
    if args.only in ("all", "masked"):
        results["masked"] = masked_arrays(root, output)
    if args.only in ("all", "kernels"):
        binary = output / ("reduction_probe.exe" if sys.platform == "win32" else "reduction_probe")
        subprocess.run(["rustc", "-O", str(root / "docs/probes/reduction_probe.rs"), "-o", str(binary)], check=True)
        results["kernels"] = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout.splitlines()
    print(json.dumps(results, indent=2, ensure_ascii=True))
    (output / "review_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(isinstance(value, dict) and value.get("mismatches") for value in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

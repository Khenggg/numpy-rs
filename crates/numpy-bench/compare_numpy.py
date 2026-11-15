#!/usr/bin/env python3
"""Head-to-head Python NumPy 2.5.4 benchmark runner & Criterion comparator for numpy-rs.

Usage:
    D:\\numpy-rs\\.venv\\Scripts\\python.exe crates\\numpy-bench\\compare_numpy.py --suite all
    D:\\numpy-rs\\.venv\\Scripts\\python.exe crates\\numpy-bench\\compare_numpy.py --suite types
    D:\\numpy-rs\\.venv\\Scripts\\python.exe crates\\numpy-bench\\compare_numpy.py --suite ufunc --json results_ufunc.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def fmt_time(ns: float) -> str:
    if ns < 1_000.0:
        return f"{ns:7.1f} ns"
    if ns < 1_000_000.0:
        return f"{ns / 1_000.0:7.2f} us"
    if ns < 1_000_000_000.0:
        return f"{ns / 1_000_000.0:7.2f} ms"
    return f"{ns / 1_000_000_000.0:7.3f}  s"


def fmt_throughput(elems: Optional[int], nbytes: Optional[int], ns: float) -> str:
    if ns <= 0:
        return "-"
    sec = ns * 1e-9
    if nbytes is not None and nbytes >= 1024:
        gbps = (nbytes / 1e9) / sec
        return f"{gbps:6.2f} GB/s"
    if elems is not None and elems > 0:
        melems = (elems / 1e6) / sec
        return f"{melems:6.1f} Melem/s"
    return "-"


def bench_fn(
    fn: Callable[[], object],
    min_sample_ns: int = 15_000_000,
    samples: int = 15,
    warmup_calls: int = 3,
) -> Dict[str, float]:
    """Adaptive microbenchmark returning median/min/mean/p95 in nanoseconds."""
    for _ in range(warmup_calls):
        _res = fn()

    # Calibrate inner loop count so each sample takes >= min_sample_ns
    t0 = time.perf_counter_ns()
    _res = fn()
    dt = max(time.perf_counter_ns() - t0, 50)
    inner = max(1, min(100_000, int(min_sample_ns // dt)))

    # Refine inner count once if fast
    if inner > 4:
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            _res = fn()
        batch_dt = max(time.perf_counter_ns() - t0, 1)
        if batch_dt < min_sample_ns // 2:
            inner = max(inner, min(200_000, int(inner * (min_sample_ns / batch_dt))))

    per_call_ns: List[float] = []
    for _ in range(samples):
        if inner == 1:
            s0 = time.perf_counter_ns()
            _res = fn()
            s1 = time.perf_counter_ns()
            per_call_ns.append(float(s1 - s0))
        else:
            s0 = time.perf_counter_ns()
            for _ in range(inner):
                _res = fn()
            s1 = time.perf_counter_ns()
            per_call_ns.append(float(s1 - s0) / inner)

    per_call_ns.sort()
    p95_idx = min(len(per_call_ns) - 1, int(0.95 * (len(per_call_ns) - 1)))
    return {
        "median_ns": statistics.median(per_call_ns),
        "min_ns": per_call_ns[0],
        "mean_ns": statistics.fmean(per_call_ns),
        "p95_ns": per_call_ns[p95_idx],
        "inner": float(inner),
        "samples": float(samples),
    }


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def suite_types() -> List[Tuple[str, Optional[int], Optional[int], Callable[[], object]]]:
    rng = make_rng(1)
    mat_1k = rng.uniform(-1.0, 1.0, size=(1000, 1000))
    mat_t_1k = mat_1k.T
    mat_512 = rng.uniform(-1.0, 1.0, size=(512, 512))
    mat_t_512 = mat_512.T
    sub_800 = mat_1k[100:900, 100:900]
    flipped_1k = mat_1k[::-1, ::-1]
    row = rng.uniform(-1.0, 1.0, size=(1, 1000))
    bcast_1k = np.broadcast_to(row, (1000, 1000))

    vec_1k = rng.uniform(-1.0, 1.0, size=1_000)
    vec_100k = rng.uniform(-1.0, 1.0, size=100_000)
    vec_200k = rng.uniform(-1.0, 1.0, size=200_000)
    strided_100k = vec_200k[::2]
    vec_10m = rng.uniform(-1.0, 1.0, size=10_000_000)
    vec_20m = rng.uniform(-1.0, 1.0, size=20_000_000)
    strided_10m = vec_20m[::2]
    f32_100k = vec_100k.astype(np.float32)
    f32_10m = vec_10m.astype(np.float32)

    return [
        ("types_zeros/zeros_f64/1k", 1_000, 1_000 * 8, lambda: np.zeros(1_000, dtype=np.float64)),
        ("types_zeros/zeros_f64/100k", 100_000, 100_000 * 8, lambda: np.zeros(100_000, dtype=np.float64)),
        ("types_zeros/zeros_f64/10m", 10_000_000, 10_000_000 * 8, lambda: np.zeros(10_000_000, dtype=np.float64)),
        ("types_zeros/zeros_f32/10m", 10_000_000, 10_000_000 * 4, lambda: np.zeros(10_000_000, dtype=np.float32)),
        ("types_zeros/full_f64/1k", 1_000, 1_000 * 8, lambda: np.full(1_000, 3.14159, dtype=np.float64)),
        ("types_zeros/full_f64/100k", 100_000, 100_000 * 8, lambda: np.full(100_000, 3.14159, dtype=np.float64)),
        ("types_zeros/full_f64/10m", 10_000_000, 10_000_000 * 8, lambda: np.full(10_000_000, 3.14159, dtype=np.float64)),
        ("types_zeros/full_f32/10m", 10_000_000, 10_000_000 * 4, lambda: np.full(10_000_000, np.float32(3.14159), dtype=np.float32)),
        ("types_view/slice_2d_1000x1000", None, None, lambda: mat_1k[100:900, 100:900]),
        ("types_view/slice_strided_100k_step2", None, None, lambda: vec_100k[::2]),
        ("types_view/slice_flip_100k", None, None, lambda: vec_100k[::-1]),
        ("types_view/transpose_1000x1000", None, None, lambda: mat_1k.T),
        ("types_view/broadcast_1x1000_to_1000x1000", None, None, lambda: np.broadcast_to(row, (1000, 1000))),
        ("types_view/reshape_1000x1000_to_100x10000", None, None, lambda: mat_1k.reshape((100, 10_000))),
        ("types_to_c_contiguous/noop_1000x1000", None, None, lambda: np.ascontiguousarray(mat_1k)),
        ("types_to_c_contiguous/transpose_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.ascontiguousarray(mat_t_1k)),
        ("types_to_c_contiguous/transpose_512x512", 512 * 512, 512 * 512 * 8, lambda: np.ascontiguousarray(mat_t_512)),
        ("types_to_c_contiguous/strided_100k_step2", 100_000, 100_000 * 8, lambda: np.ascontiguousarray(strided_100k)),
        ("types_to_c_contiguous/strided_10m_step2", 10_000_000, 10_000_000 * 8, lambda: np.ascontiguousarray(strided_10m)),
        ("types_to_c_contiguous/submatrix_800x800", 640_000, 640_000 * 8, lambda: np.ascontiguousarray(sub_800)),
        ("types_to_c_contiguous/flip_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.ascontiguousarray(flipped_1k)),
        ("types_to_c_contiguous/broadcast_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.ascontiguousarray(bcast_1k)),
        ("types_map/contig_f64_1k", 1_000, 1_000 * 8, lambda: vec_1k * 2.0 + 1.0),
        ("types_map/contig_f64_100k", 100_000, 100_000 * 8, lambda: vec_100k * 2.0 + 1.0),
        ("types_map/contig_f64_10m", 10_000_000, 10_000_000 * 8, lambda: vec_10m * 2.0 + 1.0),
        ("types_map/contig_f32_10m", 10_000_000, 10_000_000 * 4, lambda: f32_10m * np.float32(2.0) + np.float32(1.0)),
        ("types_map/transpose_f64_1000x1000", 1_000_000, 1_000_000 * 8, lambda: mat_t_1k * 2.0 + 1.0),
        ("types_map/strided_f64_100k_step2", 100_000, 100_000 * 8, lambda: strided_100k * 2.0 + 1.0),
        ("types_dyn/astype_f32_to_f64_100k", 100_000, 100_000 * 8, lambda: f32_100k.astype(np.float64)),
        ("types_dyn/astype_f32_to_f64_10m", 10_000_000, 10_000_000 * 8, lambda: f32_10m.astype(np.float64)),
    ]


def suite_core() -> List[Tuple[str, Optional[int], Optional[int], Callable[[], object]]]:
    rng = make_rng(2)
    a_100k = rng.uniform(-1.0, 1.0, size=100_000)
    b_100k = rng.uniform(-1.0, 1.0, size=100_000)
    a_5m = rng.uniform(-1.0, 1.0, size=5_000_000)
    b_5m = rng.uniform(-1.0, 1.0, size=5_000_000)
    m1 = rng.uniform(-1.0, 1.0, size=(1000, 1000))
    m2 = rng.uniform(-1.0, 1.0, size=(1000, 1000))
    m3 = rng.uniform(-1.0, 1.0, size=(1000, 1000))
    m4 = rng.uniform(-1.0, 1.0, size=(1000, 1000))
    small_100 = rng.uniform(-1.0, 1.0, size=(100, 100))
    a_1m = rng.uniform(-1.0, 1.0, size=1_000_000)
    b_1m = rng.uniform(-1.0, 1.0, size=1_000_000)
    cond_100k = rng.random(100_000) > 0.5
    cond_1m = rng.random(1_000_000) > 0.5
    idx_100 = rng.integers(0, 1000, size=100)
    e_a = rng.uniform(-1.0, 1.0, size=(256, 256))
    e_b = rng.uniform(-1.0, 1.0, size=(256, 256))
    e_ba = rng.uniform(-1.0, 1.0, size=(32, 64, 64))
    e_bb = rng.uniform(-1.0, 1.0, size=(32, 64, 64))

    return [
        ("core_create/arange_f64_100k", 100_000, 100_000 * 8, lambda: np.arange(100_000, dtype=np.float64)),
        ("core_create/arange_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.arange(10_000_000, dtype=np.float64)),
        ("core_create/linspace_f64_100k", 100_000, 100_000 * 8, lambda: np.linspace(0.0, 100.0, 100_000)),
        ("core_create/eye_f64_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.eye(1000, dtype=np.float64)),
        ("core_manip/concatenate_2x_f64_100k", 200_000, 200_000 * 8, lambda: np.concatenate((a_100k, b_100k))),
        ("core_manip/concatenate_2x_f64_5m", 10_000_000, 10_000_000 * 8, lambda: np.concatenate((a_5m, b_5m))),
        ("core_manip/stack_4x_f64_1000x1000", 4_000_000, 4_000_000 * 8, lambda: np.stack((m1, m2, m3, m4))),
        ("core_manip/pad_constant_1000x1000", 1_020_100, 1_020_100 * 8, lambda: np.pad(m1, 5, mode="constant")),
        ("core_manip/tile_100x100_10x10", 1_000_000, 1_000_000 * 8, lambda: np.tile(small_100, (10, 10))),
        ("core_manip/repeat_100k_x10", 1_000_000, 1_000_000 * 8, lambda: np.repeat(a_100k, 10)),
        ("core_manip/roll_1d_1m", 1_000_000, 1_000_000 * 8, lambda: np.roll(a_1m, 12345)),
        ("core_index/where_select_f64_100k", 100_000, 100_000 * 8, lambda: np.where(cond_100k, a_100k, b_100k)),
        ("core_index/where_select_f64_1m", 1_000_000, 1_000_000 * 8, lambda: np.where(cond_1m, a_1m, b_1m)),
        ("core_index/take_axis0_1000x1000_100idx", 100_000, 100_000 * 8, lambda: np.take(m1, idx_100, axis=0)),
        ("core_index/boolean_mask_f64_1m_50pct", 1_000_000, None, lambda: a_1m[cond_1m]),
        ("core_einsum/ij_jk_ik_256x256", 256 * 256, None, lambda: np.einsum("ij,jk->ik", e_a, e_b)),
        ("core_einsum/bij_bjk_bik_32x64x64", 32 * 64 * 64, None, lambda: np.einsum("bij,bjk->bik", e_ba, e_bb)),
        ("core_einsum/ij_ij_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.einsum("ij,ij->", m1, m2)),
    ]


def suite_ufunc() -> List[Tuple[str, Optional[int], Optional[int], Callable[[], object]]]:
    rng = make_rng(3)
    f64_1k_a = rng.uniform(0.25, 2.25, size=1_000)
    f64_1k_b = rng.uniform(0.25, 2.25, size=1_000)
    f64_100k_a = rng.uniform(0.25, 2.25, size=100_000)
    f64_100k_b = rng.uniform(0.25, 2.25, size=100_000)
    f64_1m_a = rng.uniform(0.25, 2.25, size=1_000_000)
    f64_1m_b = rng.uniform(0.25, 2.25, size=1_000_000)
    f64_10m_a = rng.uniform(0.25, 2.25, size=10_000_000)
    f64_10m_b = rng.uniform(0.25, 2.25, size=10_000_000)

    f32_1k_a = f64_1k_a.astype(np.float32)
    f32_1k_b = f64_1k_b.astype(np.float32)
    f32_100k_a = f64_100k_a.astype(np.float32)
    f32_100k_b = f64_100k_b.astype(np.float32)
    f32_10m_a = f64_10m_a.astype(np.float32)
    f32_10m_b = f64_10m_b.astype(np.float32)

    mat_a = rng.uniform(0.25, 2.25, size=(1000, 1000))
    mat_b = rng.uniform(0.25, 2.25, size=(1000, 1000))
    mat_bt = mat_b.T
    row_1000 = rng.uniform(0.25, 2.25, size=1000)
    col_1000 = rng.uniform(0.25, 2.25, size=(1000, 1))
    mat_10m = rng.uniform(0.25, 2.25, size=(3162, 3162))
    row_3162 = rng.uniform(0.25, 2.25, size=3162)

    strided_a = rng.uniform(0.25, 2.25, size=200_000)[::2]
    strided_b = rng.uniform(0.25, 2.25, size=200_000)[::2]
    i32_100k = rng.integers(1, 1000, size=100_000, dtype=np.int32)
    vec_1000 = rng.uniform(0.25, 2.25, size=1000)

    return [
        # Binary contiguous
        ("ufunc_binary/add_f64_1k", 1_000, 1_000 * 8, lambda: np.add(f64_1k_a, f64_1k_b)),
        ("ufunc_binary/add_f64_100k", 100_000, 100_000 * 8, lambda: np.add(f64_100k_a, f64_100k_b)),
        ("ufunc_binary/add_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.add(f64_10m_a, f64_10m_b)),
        ("ufunc_binary/add_f32_1k", 1_000, 1_000 * 4, lambda: np.add(f32_1k_a, f32_1k_b)),
        ("ufunc_binary/add_f32_100k", 100_000, 100_000 * 4, lambda: np.add(f32_100k_a, f32_100k_b)),
        ("ufunc_binary/add_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.add(f32_10m_a, f32_10m_b)),
        ("ufunc_binary/mul_f64_100k", 100_000, 100_000 * 8, lambda: np.multiply(f64_100k_a, f64_100k_b)),
        ("ufunc_binary/mul_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.multiply(f64_10m_a, f64_10m_b)),
        ("ufunc_binary/mul_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.multiply(f32_10m_a, f32_10m_b)),
        ("ufunc_binary/div_f64_100k", 100_000, 100_000 * 8, lambda: np.divide(f64_100k_a, f64_100k_b)),
        ("ufunc_binary/div_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.divide(f64_10m_a, f64_10m_b)),
        ("ufunc_binary/div_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.divide(f32_10m_a, f32_10m_b)),
        ("ufunc_binary/pow_f64_100k", 100_000, 100_000 * 8, lambda: np.power(f64_100k_a, f64_100k_b)),
        ("ufunc_binary/pow_f64_1m", 1_000_000, 1_000_000 * 8, lambda: np.power(f64_1m_a, f64_1m_b)),
        # Unary contiguous
        ("ufunc_unary/sqrt_f64_1k", 1_000, 1_000 * 8, lambda: np.sqrt(f64_1k_a)),
        ("ufunc_unary/sqrt_f64_100k", 100_000, 100_000 * 8, lambda: np.sqrt(f64_100k_a)),
        ("ufunc_unary/sqrt_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.sqrt(f64_10m_a)),
        ("ufunc_unary/sqrt_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.sqrt(f32_10m_a)),
        ("ufunc_unary/sin_f64_1k", 1_000, 1_000 * 8, lambda: np.sin(f64_1k_a)),
        ("ufunc_unary/sin_f64_100k", 100_000, 100_000 * 8, lambda: np.sin(f64_100k_a)),
        ("ufunc_unary/sin_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.sin(f64_10m_a)),
        ("ufunc_unary/sin_f32_100k", 100_000, 100_000 * 4, lambda: np.sin(f32_100k_a)),
        ("ufunc_unary/sin_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.sin(f32_10m_a)),
        ("ufunc_unary/cos_f64_100k", 100_000, 100_000 * 8, lambda: np.cos(f64_100k_a)),
        ("ufunc_unary/cos_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.cos(f64_10m_a)),
        ("ufunc_unary/exp_f64_1k", 1_000, 1_000 * 8, lambda: np.exp(f64_1k_a)),
        ("ufunc_unary/exp_f64_100k", 100_000, 100_000 * 8, lambda: np.exp(f64_100k_a)),
        ("ufunc_unary/exp_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.exp(f64_10m_a)),
        ("ufunc_unary/exp_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.exp(f32_10m_a)),
        ("ufunc_unary/log_f64_100k", 100_000, 100_000 * 8, lambda: np.log(f64_100k_a)),
        ("ufunc_unary/log_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.log(f64_10m_a)),
        ("ufunc_unary/log_f32_10m", 10_000_000, 10_000_000 * 4, lambda: np.log(f32_10m_a)),
        ("ufunc_unary/tanh_f64_100k", 100_000, 100_000 * 8, lambda: np.tanh(f64_100k_a)),
        ("ufunc_unary/tanh_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.tanh(f64_10m_a)),
        # Broadcasted & Strided
        ("ufunc_strided_bcast/add_bcast_row_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.add(mat_a, row_1000)),
        ("ufunc_strided_bcast/add_bcast_col_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.add(mat_a, col_1000)),
        ("ufunc_strided_bcast/add_bcast_row_10m", 9_998_244, 9_998_244 * 8, lambda: np.add(mat_10m, row_3162)),
        ("ufunc_strided_bcast/add_strided_100k_step2", 100_000, 100_000 * 8, lambda: np.add(strided_a, strided_b)),
        ("ufunc_strided_bcast/add_transposed_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.add(mat_a, mat_bt)),
        ("ufunc_strided_bcast/sin_strided_100k_step2", 100_000, 100_000 * 8, lambda: np.sin(strided_a)),
        ("ufunc_strided_bcast/sin_transposed_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.sin(mat_bt)),
        # Comparisons & Higher-order & Dyn promotion
        ("ufunc_higher/greater_f64_100k", 100_000, None, lambda: np.greater(f64_100k_a, f64_100k_b)),
        ("ufunc_higher/greater_f64_10m", 10_000_000, None, lambda: np.greater(f64_10m_a, f64_10m_b)),
        ("ufunc_higher/isclose_f64_100k", 100_000, None, lambda: np.isclose(f64_100k_a, f64_100k_b)),
        ("ufunc_higher/add_reduce_axis1_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.add.reduce(mat_a, axis=1)),
        ("ufunc_higher/add_accumulate_1d_100k", 100_000, 100_000 * 8, lambda: np.add.accumulate(f64_100k_a)),
        ("ufunc_higher/multiply_outer_1000x1000", 1_000_000, 1_000_000 * 8, lambda: np.multiply.outer(vec_1000, vec_1000)),
        ("ufunc_higher/add_mixed_i32_f32_100k", 100_000, 100_000 * 8, lambda: np.add(i32_100k, f32_100k_a)),
    ]


def suite_linalg() -> List[Tuple[str, Optional[int], Optional[int], Callable[[], object]]]:
    rng = make_rng(4)

    def make_spd(n: int) -> np.ndarray:
        m = rng.uniform(-0.25, 0.25, size=(n, n))
        return m @ m.T + np.eye(n) * float(n)

    a64 = rng.uniform(-1.0, 1.0, size=(64, 64))
    b64 = rng.uniform(-1.0, 1.0, size=(64, 64))
    a256 = rng.uniform(-1.0, 1.0, size=(256, 256))
    b256 = rng.uniform(-1.0, 1.0, size=(256, 256))
    a512 = rng.uniform(-1.0, 1.0, size=(512, 512))
    b512 = rng.uniform(-1.0, 1.0, size=(512, 512))
    b512_t = b512.T
    a1024 = rng.uniform(-1.0, 1.0, size=(1024, 1024))
    b1024 = rng.uniform(-1.0, 1.0, size=(1024, 1024))

    f32_a256 = a256.astype(np.float32)
    f32_b256 = b256.astype(np.float32)
    f32_a512 = a512.astype(np.float32)
    f32_b512 = b512.astype(np.float32)
    f32_a1024 = a1024.astype(np.float32)
    f32_b1024 = b1024.astype(np.float32)

    ba32 = rng.uniform(-1.0, 1.0, size=(32, 64, 64))
    bb32 = rng.uniform(-1.0, 1.0, size=(32, 64, 64))
    ba64 = rng.uniform(-1.0, 1.0, size=(64, 128, 128))
    bb64 = rng.uniform(-1.0, 1.0, size=(64, 128, 128))

    spd64 = make_spd(64)
    spd128 = make_spd(128)
    spd256 = make_spd(256)
    spd512 = make_spd(512)
    rhs64 = rng.uniform(-1.0, 1.0, size=(64, 1))
    rhs256 = rng.uniform(-1.0, 1.0, size=(256, 1))
    rhs512 = rng.uniform(-1.0, 1.0, size=(512, 1))

    a128 = rng.uniform(-1.0, 1.0, size=(128, 128))
    rect_512_128 = rng.uniform(-1.0, 1.0, size=(512, 128))
    vec_100k = rng.uniform(-1.0, 1.0, size=100_000)
    vec_10m = rng.uniform(-1.0, 1.0, size=10_000_000)

    return [
        ("linalg_matmul/f64_64x64", None, None, lambda: a64 @ b64),
        ("linalg_matmul/f64_256x256", None, None, lambda: a256 @ b256),
        ("linalg_matmul/f64_512x512", None, None, lambda: a512 @ b512),
        ("linalg_matmul/f64_1024x1024", None, None, lambda: a1024 @ b1024),
        ("linalg_matmul/f32_256x256", None, None, lambda: f32_a256 @ f32_b256),
        ("linalg_matmul/f32_512x512", None, None, lambda: f32_a512 @ f32_b512),
        ("linalg_matmul/f32_1024x1024", None, None, lambda: f32_a1024 @ f32_b1024),
        ("linalg_matmul/f64_transposed_512x512", None, None, lambda: a512 @ b512_t),
        ("linalg_matmul/batched_f64_32x64x64", None, None, lambda: ba32 @ bb32),
        ("linalg_matmul/batched_f64_64x128x128", None, None, lambda: ba64 @ bb64),
        ("linalg_solve/solve_f64_64x64", None, None, lambda: np.linalg.solve(spd64, rhs64)),
        ("linalg_solve/solve_f64_256x256", None, None, lambda: np.linalg.solve(spd256, rhs256)),
        ("linalg_solve/solve_f64_512x512", None, None, lambda: np.linalg.solve(spd512, rhs512)),
        ("linalg_solve/inv_f64_256x256", None, None, lambda: np.linalg.inv(spd256)),
        ("linalg_solve/det_f64_256x256", None, None, lambda: np.linalg.det(spd256)),
        ("linalg_solve/cholesky_f64_256x256", None, None, lambda: np.linalg.cholesky(spd256)),
        ("linalg_solve/cholesky_f64_512x512", None, None, lambda: np.linalg.cholesky(spd512)),
        ("linalg_decomp/qr_f64_128x128", None, None, lambda: np.linalg.qr(a128)),
        ("linalg_decomp/qr_f64_256x256", None, None, lambda: np.linalg.qr(a256)),
        ("linalg_decomp/qr_f64_512x512", None, None, lambda: np.linalg.qr(a512)),
        ("linalg_decomp/svd_f64_64x64", None, None, lambda: np.linalg.svd(a64, full_matrices=False)),
        ("linalg_decomp/svd_f64_128x128", None, None, lambda: np.linalg.svd(a128, full_matrices=False)),
        ("linalg_decomp/svd_f64_256x256", None, None, lambda: np.linalg.svd(a256, full_matrices=False)),
        ("linalg_decomp/svd_f64_512x512", None, None, lambda: np.linalg.svd(a512, full_matrices=False)),
        ("linalg_decomp/eigh_f64_64x64", None, None, lambda: np.linalg.eigh(spd64)),
        ("linalg_decomp/eigh_f64_128x128", None, None, lambda: np.linalg.eigh(spd128)),
        ("linalg_decomp/eigh_f64_256x256", None, None, lambda: np.linalg.eigh(spd256)),
        ("linalg_decomp/eigh_f64_512x512", None, None, lambda: np.linalg.eigh(spd512)),
        ("linalg_decomp/lstsq_f64_512x128", None, None, lambda: np.linalg.lstsq(rect_512_128, rhs512, rcond=None)),
        ("linalg_norm/norm_f64_100k", 100_000, 100_000 * 8, lambda: np.linalg.norm(vec_100k)),
        ("linalg_norm/norm_f64_10m", 10_000_000, 10_000_000 * 8, lambda: np.linalg.norm(vec_10m)),
    ]


def suite_fft() -> List[Tuple[str, Optional[int], Optional[int], Callable[[], object]]]:
    rng = make_rng(5)

    def rand_c128(shape: Tuple[int, ...]) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=shape) + 1j * rng.uniform(-1.0, 1.0, size=shape)

    c_1k = rand_c128((1024,))
    c_64k = rand_c128((65536,))
    c_1m = rand_c128((1_048_576,))
    c_1000 = rand_c128((1000,))
    c_10000 = rand_c128((10000,))
    c_997 = rand_c128((997,))

    r_1k = rng.uniform(-1.0, 1.0, size=1024)
    r_64k = rng.uniform(-1.0, 1.0, size=65536)
    r_1m = rng.uniform(-1.0, 1.0, size=1_048_576)

    c_256x256 = rand_c128((256, 256))
    c_1024x1024 = rand_c128((1024, 1024))
    r_512x512 = rng.uniform(-1.0, 1.0, size=(512, 512))
    r_1024x1024 = rng.uniform(-1.0, 1.0, size=(1024, 1024))
    c_batch = rand_c128((1000, 1024))

    return [
        ("fft_1d/c128_1024", 1024, 1024 * 16, lambda: np.fft.fft(c_1k)),
        ("fft_1d/c128_65536", 65536, 65536 * 16, lambda: np.fft.fft(c_64k)),
        ("fft_1d/c128_1048576", 1_048_576, 1_048_576 * 16, lambda: np.fft.fft(c_1m)),
        ("fft_1d/f64_65536", 65536, 65536 * 8, lambda: np.fft.fft(r_64k)),
        ("fft_1d/c128_1000", 1000, 1000 * 16, lambda: np.fft.fft(c_1000)),
        ("fft_1d/c128_10000", 10000, 10000 * 16, lambda: np.fft.fft(c_10000)),
        ("fft_1d/c128_997_prime", 997, 997 * 16, lambda: np.fft.fft(c_997)),
        ("rfft_1d/f64_1024", 1024, 1024 * 8, lambda: np.fft.rfft(r_1k)),
        ("rfft_1d/f64_65536", 65536, 65536 * 8, lambda: np.fft.rfft(r_64k)),
        ("rfft_1d/f64_1048576", 1_048_576, 1_048_576 * 8, lambda: np.fft.rfft(r_1m)),
        ("fft_2d/c128_256x256", 256 * 256, 256 * 256 * 16, lambda: np.fft.fft2(c_256x256)),
        ("fft_2d/c128_1024x1024", 1024 * 1024, 1024 * 1024 * 16, lambda: np.fft.fft2(c_1024x1024)),
        ("fft_2d/f64_512x512", 512 * 512, 512 * 512 * 8, lambda: np.fft.fft2(r_512x512)),
        ("fft_2d/rfft2_f64_512x512", 512 * 512, 512 * 512 * 8, lambda: np.fft.rfft2(r_512x512)),
        ("fft_2d/rfft2_f64_1024x1024", 1024 * 1024, 1024 * 1024 * 8, lambda: np.fft.rfft2(r_1024x1024)),
        ("fft_batch/c128_1000x1024_axis1", 1000 * 1024, 1000 * 1024 * 16, lambda: np.fft.fft(c_batch, axis=1)),
        ("fft_batch/c128_1000x1024_axis0", 1000 * 1024, 1000 * 1024 * 16, lambda: np.fft.fft(c_batch, axis=0)),
    ]


def suite_random() -> List[Tuple[str, Optional[int], Optional[int], Callable[[], object]]]:
    rng_pcg = np.random.Generator(np.random.PCG64(12345))
    rng_dxsm = np.random.Generator(np.random.PCG64DXSM(12345))
    rng_philox = np.random.Generator(np.random.Philox(12345))
    rng_sfc = np.random.Generator(np.random.SFC64(12345))
    rng_mt = np.random.Generator(np.random.MT19937(12345))

    return [
        ("random_pcg64/random_f64_1k", 1_000, 1_000 * 8, lambda: rng_pcg.random(1_000)),
        ("random_pcg64/random_f64_100k", 100_000, 100_000 * 8, lambda: rng_pcg.random(100_000)),
        ("random_pcg64/random_f64_10m", 10_000_000, 10_000_000 * 8, lambda: rng_pcg.random(10_000_000)),
        ("random_pcg64/standard_normal_f64_1k", 1_000, 1_000 * 8, lambda: rng_pcg.standard_normal(1_000)),
        ("random_pcg64/standard_normal_f64_100k", 100_000, 100_000 * 8, lambda: rng_pcg.standard_normal(100_000)),
        ("random_pcg64/standard_normal_f64_10m", 10_000_000, 10_000_000 * 8, lambda: rng_pcg.standard_normal(10_000_000)),
        ("random_pcg64/standard_exponential_f64_100k", 100_000, 100_000 * 8, lambda: rng_pcg.standard_exponential(100_000)),
        ("random_pcg64/standard_exponential_f64_10m", 10_000_000, 10_000_000 * 8, lambda: rng_pcg.standard_exponential(10_000_000)),
        ("random_pcg64/integers_i64_100k", 100_000, 100_000 * 8, lambda: rng_pcg.integers(0, 1_000_000, size=100_000)),
        ("random_pcg64/integers_i64_10m", 10_000_000, 10_000_000 * 8, lambda: rng_pcg.integers(0, 1_000_000, size=10_000_000)),
        ("random_pcg64/uniform_f64_1m", 1_000_000, 1_000_000 * 8, lambda: rng_pcg.uniform(-5.0, 5.0, size=1_000_000)),
        ("random_pcg64/gamma_f64_1m", 1_000_000, 1_000_000 * 8, lambda: rng_pcg.gamma(2.5, 1.5, size=1_000_000)),
        ("random_pcg64/poisson_i64_1m", 1_000_000, 1_000_000 * 8, lambda: rng_pcg.poisson(5.0, size=1_000_000)),
        ("random_pcg64/binomial_i64_1m", 1_000_000, 1_000_000 * 8, lambda: rng_pcg.binomial(20, 0.3, size=1_000_000)),
        ("random_pcg64/choice_no_replace_100k_from_1m", 100_000, 100_000 * 8, lambda: rng_pcg.choice(1_000_000, size=100_000, replace=False)),
        ("random_pcg64/permutation_1m", 1_000_000, 1_000_000 * 8, lambda: rng_pcg.permutation(1_000_000)),
        ("random_bitgens/pcg64dxsm_random_1m", 1_000_000, 1_000_000 * 8, lambda: rng_dxsm.random(1_000_000)),
        ("random_bitgens/philox_random_1m", 1_000_000, 1_000_000 * 8, lambda: rng_philox.random(1_000_000)),
        ("random_bitgens/sfc64_random_1m", 1_000_000, 1_000_000 * 8, lambda: rng_sfc.random(1_000_000)),
        ("random_bitgens/mt19937_random_1m", 1_000_000, 1_000_000 * 8, lambda: rng_mt.random(1_000_000)),
    ]


SUITES = {
    "types": suite_types,
    "core": suite_core,
    "ufunc": suite_ufunc,
    "linalg": suite_linalg,
    "fft": suite_fft,
    "random": suite_random,
}


def load_criterion_estimates(criterion_dir: Path) -> Dict[str, Dict[str, float]]:
    """Scan target/criterion for */new/estimates.json and return normalized key -> ns stats."""
    out: Dict[str, Dict[str, float]] = {}
    if not criterion_dir.exists():
        return out
    for est_path in criterion_dir.rglob("new/estimates.json"):
        rel = est_path.relative_to(criterion_dir).parent.parent
        key = rel.as_posix()
        try:
            with est_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            out[key] = {
                "median_ns": float(data["median"]["point_estimate"]),
                "mean_ns": float(data["mean"]["point_estimate"]),
                "std_dev_ns": float(data["std_dev"]["point_estimate"]),
            }
        except Exception:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="NumPy 2.5.4 vs numpy-rs Criterion head-to-head benchmark")
    parser.add_argument(
        "--suite",
        default="types",
        choices=["types", "core", "ufunc", "linalg", "fft", "random", "all"],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--criterion-dir",
        default=str(Path(__file__).resolve().parents[2] / "target" / "criterion"),
        help="Path to Criterion output directory",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--samples", type=int, default=15, help="Samples per benchmark")
    parser.add_argument("--min-sample-ms", type=float, default=15.0, help="Target ms per sample")
    args = parser.parse_args()

    selected = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    criterion_dir = Path(args.criterion_dir)
    rust_results = load_criterion_estimates(criterion_dir)

    print(
        f"NumPy {np.__version__} | Python {sys.version.split()[0]} | "
        f"CPU threads={os.cpu_count()} | Criterion dir={criterion_dir}"
    )
    print("-" * 112)
    print(
        f"{'Benchmark':<44} | {'NumPy med':>10} | {'NumPy Tput':>12} | "
        f"{'Rust med':>10} | {'Rust Tput':>12} | {'Speedup (Py/Rs)':>15}"
    )
    print("-" * 112)

    all_records: Dict[str, dict] = {}
    min_ns = int(args.min_sample_ms * 1_000_000)

    for suite_name in selected:
        cases = SUITES[suite_name]()
        for key, elems, nbytes, fn in cases:
            py_stats = bench_fn(fn, min_sample_ns=min_ns, samples=args.samples)
            py_med = py_stats["median_ns"]
            py_tput = fmt_throughput(elems, nbytes, py_med)

            rs = rust_results.get(key)
            if rs is not None:
                rs_med = rs["median_ns"]
                rs_tput = fmt_throughput(elems, nbytes, rs_med)
                ratio = py_med / rs_med if rs_med > 0 else float("inf")
                if ratio >= 1.15:
                    speedup_str = f"{ratio:6.2f}x FASTER"
                elif ratio <= 0.85:
                    speedup_str = f"{ratio:6.2f}x SLOWER"
                else:
                    speedup_str = f"{ratio:6.2f}x PARITY"
                rs_str = fmt_time(rs_med)
            else:
                rs_med = None
                rs_tput = "-"
                rs_str = "   pending"
                speedup_str = "      -"

            print(
                f"{key:<44} | {fmt_time(py_med):>10} | {py_tput:>12} | "
                f"{rs_str:>10} | {rs_tput:>12} | {speedup_str:>15}",
                flush=True,
            )
            all_records[key] = {
                "suite": suite_name,
                "elems": elems,
                "nbytes": nbytes,
                "numpy": py_stats,
                "rust": rs,
                "speedup_py_over_rust": (py_med / rs_med) if rs_med else None,
            }

    print("-" * 112)
    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

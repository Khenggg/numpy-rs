//! Shared benchmark helpers, deterministic data generators, and layout diagnostics
//! for comparing `numpy-rs` against Python NumPy 2.5.4.

use criterion::Criterion;
use num_complex::{Complex32, Complex64};
use numpy_types::{AlignedBuffer, MemoryLayout, NdArray, Order, SIMD_ALIGN, Scalar, shape_size};
use std::fmt;
use std::time::Duration;

/// Standard Criterion configuration tuned for low noise and fast iteration across
/// microsecond (1K) and multi-millisecond (10M / 1024x1024) workloads.
pub fn default_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(400))
        .measurement_time(Duration::from_millis(1200))
        .sample_size(40)
        .noise_threshold(0.03)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
fn u64_to_unit_f64(bits: u64) -> f64 {
    // 53-bit uniform mantissa in [0.0, 1.0)
    ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Generate a deterministic `NdArray<f64>` with values in `[-1.0, 1.0)`.
pub fn gen_f64(shape: &[usize], seed: u64) -> NdArray<f64> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        2.0 * u64_to_unit_f64(state) - 1.0
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic strictly positive `NdArray<f64>` with values in `[0.25, 2.25)`
/// safe for `sqrt`, `log`, `divide`, and `power`.
pub fn gen_f64_pos(shape: &[usize], seed: u64) -> NdArray<f64> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        0.25 + 2.0 * u64_to_unit_f64(state)
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic `NdArray<f32>` with values in `[-1.0, 1.0)`.
pub fn gen_f32(shape: &[usize], seed: u64) -> NdArray<f32> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        (2.0 * u64_to_unit_f64(state) - 1.0) as f32
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic strictly positive `NdArray<f32>` in `[0.25, 2.25)`.
pub fn gen_f32_pos(shape: &[usize], seed: u64) -> NdArray<f32> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        (0.25 + 2.0 * u64_to_unit_f64(state)) as f32
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic `NdArray<Complex64>` with real/imag parts in `[-1.0, 1.0)`.
pub fn gen_c128(shape: &[usize], seed: u64) -> NdArray<Complex64> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        let re = 2.0 * u64_to_unit_f64(state) - 1.0;
        state = splitmix64(state);
        let im = 2.0 * u64_to_unit_f64(state) - 1.0;
        Complex64::new(re, im)
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic `NdArray<Complex32>` with real/imag parts in `[-1.0, 1.0)`.
pub fn gen_c64(shape: &[usize], seed: u64) -> NdArray<Complex32> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        let re = (2.0 * u64_to_unit_f64(state) - 1.0) as f32;
        state = splitmix64(state);
        let im = (2.0 * u64_to_unit_f64(state) - 1.0) as f32;
        Complex32::new(re, im)
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic `NdArray<i32>`.
pub fn gen_i32(shape: &[usize], seed: u64) -> NdArray<i32> {
    let len = shape_size(shape).expect("valid shape");
    let mut state = seed;
    let buf = AlignedBuffer::from_fn(len, |_| {
        state = splitmix64(state);
        (state as i32) % 1024
    });
    NdArray::from_aligned_buffer(buf, shape, Order::C).unwrap()
}

/// Generate a deterministic well-conditioned symmetric positive-definite `[n, n]` `NdArray<f64>`
/// (`M + n * I` where `M` is symmetric with off-diagonal entries in `[-0.5, 0.5]`).
pub fn gen_spd_f64(n: usize, seed: u64) -> NdArray<f64> {
    let mut uninit = AlignedBuffer::<f64>::new_uninit(n * n);
    let slice = uninit.as_mut_slice();
    let mut state = seed;
    for i in 0..n {
        for j in 0..=i {
            state = splitmix64(state);
            let v = u64_to_unit_f64(state) - 0.5;
            if i == j {
                slice[i * n + j].write((n as f64) + 1.0 + v.abs());
            } else {
                let sym = v * 0.25;
                slice[i * n + j].write(sym);
                slice[j * n + i].write(sym);
            }
        }
    }
    let buf = unsafe { uninit.assume_init() };
    NdArray::from_aligned_buffer(buf, &[n, n], Order::C).unwrap()
}

/// Diagnostic snapshot of an `NdArray<T>`'s memory alignment, offset, contiguity, and stride layout.
#[derive(Debug, Clone)]
pub struct LayoutReport {
    pub label: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub len: usize,
    pub nbytes: usize,
    pub ptr_addr: usize,
    pub aligned_64: bool,
    pub layout: MemoryLayout,
    pub has_as_slice: bool,
}

impl fmt::Display for LayoutReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<28} dtype={:<10} shape={:<14?} strides={:<14?} off={:<6} ptr=0x{:x} align64={} layout={:?} slice={}",
            self.label,
            self.dtype,
            self.shape,
            self.strides,
            self.offset,
            self.ptr_addr,
            self.aligned_64,
            self.layout,
            self.has_as_slice,
        )
    }
}

/// Inspect an `NdArray<T>`'s storage pointer alignment, strides, contiguity, and slice fast-path eligibility.
pub fn inspect_layout<T: Scalar>(label: &str, arr: &NdArray<T>) -> LayoutReport {
    let ptr_addr = arr.storage_ptr() as usize;
    LayoutReport {
        label: label.to_string(),
        dtype: T::DTYPE.name().to_string(),
        shape: arr.shape().to_vec(),
        strides: arr.strides().to_vec(),
        offset: arr.offset(),
        len: arr.len(),
        nbytes: arr.nbytes(),
        ptr_addr,
        aligned_64: ptr_addr % SIMD_ALIGN == 0,
        layout: arr.layout(),
        has_as_slice: arr.as_slice().is_some(),
    }
}

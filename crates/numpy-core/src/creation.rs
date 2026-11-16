//! Array creation routines (`empty`, `zeros`, `ones`, `full`, `arange`, `linspace`,
//! `logspace`, `geomspace`, `eye`, `identity`, `diag`, `diagflat`, `tri`, `tril`, `triu`,
//! `vander`, `meshgrid`, `indices`, `from_fn`) for both `NdArray<T>` and `DynArray`.

use crate::util::{
    RAYON_THRESHOLD, finish_c_array, linear_to_coords, resolve_new_order,
    try_aligned_filled_fast, DynScalar,
};
use half::f16;
use num_complex::{Complex32, Complex64};
use numpy_types::{
    shape_size, AlignedBuffer, DType, DynArray, NdArray, NumPyError, Order, Result, Scalar,
    dispatch_dyn,
};
use rayon::prelude::*;
use std::mem::MaybeUninit;

/// Indexing convention for `meshgrid`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshgridIndexing {
    /// Cartesian `(x, y)` indexing: swaps the first two axes (`(M, N)` for `x` length `N`, `y` length `M`).
    #[default]
    Xy,
    /// Matrix `(i, j)` indexing: preserves axis order `(N, M)` for `x` length `N`, `y` length `M`.
    Ij,
}

/// Trait unifying arithmetic needed across NumPy creation routines for all 14 scalar dtypes.
pub trait CreationElement: Scalar + Send + Sync {
    /// Converts a real `f64` using NumPy rounding/truncation semantics for `linspace`.
    fn from_linspace_f64(v: f64) -> Self;
    /// Converts a `Complex64` value into `Self`.
    fn from_c64(v: Complex64) -> Self;
    /// Converts `Self` into `(re, im)` `Complex64`.
    fn to_c64(self) -> Complex64;
    /// Computes `a * b` in `Self` (for `bool`, logical AND).
    fn elem_mul(self, rhs: Self) -> Self;
    /// Computes `a + b` in `Self` (for `bool`, logical OR).
    fn elem_add(self, rhs: Self) -> Self;
    /// Computes `a - b` in `Self`.
    fn elem_sub(self, rhs: Self) -> Self;
    /// Computes `arange` length and element evaluation.
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>>;
    /// Computes `linspace` sequence and returned step size.
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)>;
    /// Computes `logspace` sequence in `base`.
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>>;
    /// Computes `geomspace` sequence.
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>>;
}

fn finish_order_array<T>(buf: AlignedBuffer<T>, shape: &[usize], order: Order) -> Result<NdArray<T>> {
    let resolved = resolve_new_order(order);
    let _ = shape_size(shape)?;
    NdArray::from_aligned_buffer(buf, shape, resolved)
}

#[inline]
fn resolve_like_order<T>(arr: &NdArray<T>, order: Order) -> Order {
    match order {
        Order::C => Order::C,
        Order::F => Order::F,
        Order::A | Order::K => {
            if arr.is_f_contiguous() && !arr.is_c_contiguous() {
                Order::F
            } else {
                Order::C
            }
        }
    }
}

fn check_no_nan_arange(start: f64, stop: f64, step: f64) -> Result<()> {
    if start.is_nan() || stop.is_nan() || step.is_nan() {
        return Err(NumPyError::InvalidValue(
            "arange: cannot compute length for NaN bounds".into(),
        ));
    }
    if step == 0.0 {
        return Err(NumPyError::InvalidValue("arange: step must not be zero".into()));
    }
    Ok(())
}

fn float_arange_len(start: f64, stop: f64, step: f64) -> Result<usize> {
    check_no_nan_arange(start, stop, step)?;
    if !start.is_finite() || !stop.is_finite() || !step.is_finite() {
        return Err(NumPyError::InvalidValue(
            "arange: infinite bounds or step cannot produce finite length".into(),
        ));
    }
    let span = stop - start;
    if (step > 0.0 && span <= 0.0) || (step < 0.0 && span >= 0.0) {
        return Ok(0);
    }
    let count_f = (span / step).ceil();
    if count_f <= 0.0 {
        return Ok(0);
    }
    if count_f > (isize::MAX as f64) {
        return Err(NumPyError::InvalidValue(
            "arange: resulting array is too large".into(),
        ));
    }
    Ok(count_f as usize)
}

fn complex_arange_len(start: Complex64, stop: Complex64, step: Complex64) -> Result<usize> {
    if !start.is_finite() || !stop.is_finite() || !step.is_finite() {
        return Err(NumPyError::InvalidValue(
            "arange: non-finite length for complex bounds".into(),
        ));
    }
    if step.re == 0.0 && step.im == 0.0 {
        return Err(NumPyError::InvalidValue("arange: step must not be zero".into()));
    }
    let ratio = (stop - start) / step;
    if !ratio.re.is_finite() || !ratio.im.is_finite() {
        return Err(NumPyError::InvalidValue(
            "arange: non-finite length for complex bounds".into(),
        ));
    }
    let count_f = ratio.re.ceil();
    if count_f <= 0.0 {
        return Ok(0);
    }
    if count_f > (isize::MAX as f64) {
        return Err(NumPyError::InvalidValue(
            "arange: resulting array is too large".into(),
        ));
    }
    Ok(count_f as usize)
}

fn complex_logspace_elem(y: Complex64, base: f64) -> Complex64 {
    if base == 0.0 {
        if y.re == 0.0 && y.im == 0.0 {
            Complex64::new(1.0, 0.0)
        } else if y.re > 0.0 && y.re.is_finite() && y.im.is_finite() {
            Complex64::new(0.0, 0.0)
        } else {
            Complex64::new(f64::NAN, f64::NAN)
        }
    } else if base < 0.0 && y.im == 0.0 {
        Complex64::new(base.powf(y.re), 0.0)
    } else {
        let ln_b = Complex64::new(base, 0.0).ln();
        (y * ln_b).exp()
    }
}

fn int_arange_len(start: i128, stop: i128, step: i128) -> Result<usize> {
    if step == 0 {
        return Err(NumPyError::InvalidValue("arange: step must not be zero".into()));
    }
    if (step > 0 && stop <= start) || (step < 0 && stop >= start) {
        return Ok(0);
    }
    let dist: u128 = if step > 0 {
        (stop - start) as u128
    } else {
        (start - stop) as u128
    };
    let abs_step = step.unsigned_abs();
    let count = dist.div_ceil(abs_step);
    usize::try_from(count).map_err(|_| NumPyError::InvalidValue("arange: resulting array is too large".into()))
}

fn fill_arange_float<T: Scalar + Send + Sync>(
    len: usize,
    eval: impl Fn(usize) -> T + Sync,
) -> Result<NdArray<T>> {
    if len == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &[0]);
    }
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(len)?;
    let dst = uninit.as_mut_slice();
    if len >= RAYON_THRESHOLD {
        dst.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * 4096;
                for (i, slot) in chunk.iter_mut().enumerate() {
                    slot.write(eval(base + i));
                }
            });
    } else {
        for (i, slot) in dst.iter_mut().enumerate() {
            slot.write(eval(i));
        }
    }
    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &[len])
}

fn real_linspace_impl<T: CreationElement>(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<(NdArray<T>, f64)> {
    if num == 0 {
        let arr = finish_c_array(AlignedBuffer::new_zeroed(0), &[0])?;
        return Ok((arr, f64::NAN));
    }
    let div = if endpoint {
        (num - 1) as f64
    } else {
        num as f64
    };
    let delta = stop - start;
    let step = if div == 0.0 { f64::NAN } else { delta / div };

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(num)?;
    let dst = uninit.as_mut_slice();

    if num == 1 {
        dst[0].write(T::from_linspace_f64(start));
    } else {
        let use_mul_div = step == 0.0 || !step.is_finite();
        let eval = |i: usize| -> T {
            if endpoint && i == num - 1 {
                T::from_linspace_f64(stop)
            } else if use_mul_div {
                T::from_linspace_f64(start + (delta * (i as f64)) / div)
            } else {
                T::from_linspace_f64(start + (i as f64) * step)
            }
        };
        if num >= RAYON_THRESHOLD {
            dst.par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * 4096;
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        slot.write(eval(base + j));
                    }
                });
        } else {
            for (i, slot) in dst.iter_mut().enumerate() {
                slot.write(eval(i));
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    let arr = finish_c_array(buf, &[num])?;
    Ok((arr, step))
}

fn complex_linspace_impl<T: CreationElement>(
    start: Complex64,
    stop: Complex64,
    num: usize,
    endpoint: bool,
) -> Result<(NdArray<T>, f64)> {
    if num == 0 {
        let arr = finish_c_array(AlignedBuffer::new_zeroed(0), &[0])?;
        return Ok((arr, f64::NAN));
    }
    let div = if endpoint {
        (num - 1) as f64
    } else {
        num as f64
    };
    let delta = stop - start;
    let step_c = if div == 0.0 {
        Complex64::new(f64::NAN, f64::NAN)
    } else {
        delta / div
    };
    let ret_step = step_c.re;

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(num)?;
    let dst = uninit.as_mut_slice();

    if num == 1 {
        dst[0].write(T::from_c64(start));
    } else {
        let use_mul_div = (step_c.re == 0.0 && step_c.im == 0.0) || !step_c.is_finite();
        let eval = |i: usize| -> T {
            if endpoint && i == num - 1 {
                T::from_c64(stop)
            } else if use_mul_div {
                T::from_c64(start + (delta * (i as f64)) / div)
            } else {
                T::from_c64(start + step_c * (i as f64))
            }
        };
        if num >= RAYON_THRESHOLD {
            dst.par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * 4096;
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        slot.write(eval(base + j));
                    }
                });
        } else {
            for (i, slot) in dst.iter_mut().enumerate() {
                slot.write(eval(i));
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    let arr = finish_c_array(buf, &[num])?;
    Ok((arr, ret_step))
}

fn real_logspace_impl<T: CreationElement>(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    base: f64,
) -> Result<NdArray<T>> {
    let (lin, _) = real_linspace_impl::<f64>(start, stop, num, endpoint)?;
    let slice = lin.as_slice().unwrap_or(&[]);
    fill_arange_float(num, |i| T::from_f64_lossy(base.powf(slice[i])))
}

fn real_geomspace_impl<T: CreationElement>(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> Result<NdArray<T>> {
    if start == 0.0 || stop == 0.0 {
        return Err(NumPyError::InvalidValue(
            "Geometric sequence cannot include zero".into(),
        ));
    }
    if num == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &[0]);
    }
    let has_nan = start.is_nan() || stop.is_nan();
    if !has_nan && (start < 0.0) != (stop < 0.0) {
        return Err(NumPyError::InvalidValue(
            "geomspace: start and stop have opposite signs; use complex dtype".into(),
        ));
    }
    if has_nan {
        return fill_arange_float(num, |i| {
            if i == 0 && !start.is_nan() {
                T::from_f64_lossy(start)
            } else {
                T::from_f64_lossy(f64::NAN)
            }
        });
    }
    let sign = if start < 0.0 { -1.0 } else { 1.0 };
    let log_start = (start.abs()).log10();
    let log_stop = (stop.abs()).log10();
    let (lin, _) = real_linspace_impl::<f64>(log_start, log_stop, num, endpoint)?;
    let sl = lin.as_slice().unwrap_or(&[]);
    fill_arange_float(num, |i| {
        if i == 0 {
            T::from_f64_lossy(start)
        } else if endpoint && i == num - 1 {
            T::from_f64_lossy(stop)
        } else {
            T::from_f64_lossy(sign * 10.0f64.powf(sl[i]))
        }
    })
}

fn complex_geomspace_impl<T: CreationElement>(
    start: Complex64,
    stop: Complex64,
    num: usize,
    endpoint: bool,
) -> Result<NdArray<T>> {
    if (start.re == 0.0 && start.im == 0.0) || (stop.re == 0.0 && stop.im == 0.0) {
        return Err(NumPyError::InvalidValue(
            "Geometric sequence cannot include zero".into(),
        ));
    }
    if num == 0 {
        return finish_c_array(AlignedBuffer::new_zeroed(0), &[0]);
    }
    let start_nan = start.re.is_nan() || start.im.is_nan();
    let stop_nan = stop.re.is_nan() || stop.im.is_nan();
    if start_nan || stop_nan {
        return fill_arange_float(num, |i| {
            if i == 0 && !start_nan {
                T::from_c64(start)
            } else {
                T::from_c64(Complex64::new(f64::NAN, f64::NAN))
            }
        });
    }
    let out_sign = start / start.norm();
    let log_start = (start / out_sign).ln() / std::f64::consts::LN_10;
    let log_stop = (stop / out_sign).ln() / std::f64::consts::LN_10;
    let (lin, _) = complex_linspace_impl::<Complex64>(log_start, log_stop, num, endpoint)?;
    let sl = lin.as_slice().unwrap_or(&[]);
    let ln10 = std::f64::consts::LN_10;
    fill_arange_float(num, |i| {
        if i == 0 {
            T::from_c64(start)
        } else if endpoint && i == num - 1 {
            T::from_c64(stop)
        } else {
            T::from_c64(out_sign * (sl[i] * ln10).exp())
        }
    })
}

impl CreationElement for bool {
    #[inline]
    fn from_linspace_f64(v: f64) -> Self {
        v != 0.0
    }
    #[inline]
    fn from_c64(v: Complex64) -> Self {
        v.re != 0.0 || v.im != 0.0
    }
    #[inline]
    fn to_c64(self) -> Complex64 {
        Complex64::new(if self { 1.0 } else { 0.0 }, 0.0)
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self && rhs
    }
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self || rhs
    }
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self != rhs
    }
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
        let len = int_arange_len(i128::from(start), i128::from(stop), i128::from(step))?;
        if len > 2 {
            return Err(NumPyError::InvalidValue(
                "no fill function for bool arange of length > 2".into(),
            ));
        }
        let s0 = i128::from(start);
        let st = i128::from(step);
        fill_arange_float(len, |i| (s0 + (i as i128) * st) != 0)
    }
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
        real_linspace_impl(
            if start { 1.0 } else { 0.0 },
            if stop { 1.0 } else { 0.0 },
            num,
            endpoint,
        )
    }
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
        real_logspace_impl(
            if start { 1.0 } else { 0.0 },
            if stop { 1.0 } else { 0.0 },
            num,
            endpoint,
            base,
        )
    }
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
        real_geomspace_impl(
            if start { 1.0 } else { 0.0 },
            if stop { 1.0 } else { 0.0 },
            num,
            endpoint,
        )
    }
}

macro_rules! impl_creation_int {
    ($($ty:ty),*) => {
        $(
            impl CreationElement for $ty {
                #[inline]
                fn from_linspace_f64(v: f64) -> Self {
                    v.floor() as $ty
                }
                #[inline]
                fn from_c64(v: Complex64) -> Self {
                    v.re.floor() as $ty
                }
                #[inline]
                fn to_c64(self) -> Complex64 {
                    Complex64::new(self as f64, 0.0)
                }
                #[inline]
                fn elem_mul(self, rhs: Self) -> Self {
                    self.wrapping_mul(rhs)
                }
                #[inline]
                fn elem_add(self, rhs: Self) -> Self {
                    self.wrapping_add(rhs)
                }
                #[inline]
                fn elem_sub(self, rhs: Self) -> Self {
                    self.wrapping_sub(rhs)
                }
                fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
                    let s0 = start as i128;
                    let s1 = stop as i128;
                    let st = step as i128;
                    let len = int_arange_len(s0, s1, st)?;
                    fill_arange_float(len, |i| (s0 + (i as i128) * st) as $ty)
                }
                fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
                    real_linspace_impl(start as f64, stop as f64, num, endpoint)
                }
                fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
                    real_logspace_impl(start as f64, stop as f64, num, endpoint, base)
                }
                fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
                    real_geomspace_impl(start as f64, stop as f64, num, endpoint)
                }
            }
        )*
    };
}

impl_creation_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl CreationElement for f16 {
    #[inline]
    fn from_linspace_f64(v: f64) -> Self {
        f16::from_f64(v)
    }
    #[inline]
    fn from_c64(v: Complex64) -> Self {
        f16::from_f64(v.re)
    }
    #[inline]
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
        let s0 = f64::from(start);
        let s1 = f64::from(stop);
        let st = f64::from(step);
        let len = float_arange_len(s0, s1, st)?;
        fill_arange_float(len, |i| f16::from_f64(s0 + (i as f64) * st))
    }
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
        real_linspace_impl(f64::from(start), f64::from(stop), num, endpoint)
    }
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
        real_logspace_impl(f64::from(start), f64::from(stop), num, endpoint, base)
    }
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
        real_geomspace_impl(f64::from(start), f64::from(stop), num, endpoint)
    }
}

impl CreationElement for f32 {
    #[inline]
    fn from_linspace_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn from_c64(v: Complex64) -> Self {
        v.re as f32
    }
    #[inline]
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
        let s0 = f64::from(start);
        let s1 = f64::from(stop);
        let st = f64::from(step);
        let len = float_arange_len(s0, s1, st)?;
        fill_arange_float(len, |i| start + (i as f32) * step)
    }
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
        real_linspace_impl(f64::from(start), f64::from(stop), num, endpoint)
    }
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
        real_logspace_impl(f64::from(start), f64::from(stop), num, endpoint, base)
    }
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
        real_geomspace_impl(f64::from(start), f64::from(stop), num, endpoint)
    }
}

impl CreationElement for f64 {
    #[inline]
    fn from_linspace_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn from_c64(v: Complex64) -> Self {
        v.re
    }
    #[inline]
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
        let len = float_arange_len(start, stop, step)?;
        fill_arange_float(len, |i| start + (i as f64) * step)
    }
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
        real_linspace_impl(start, stop, num, endpoint)
    }
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
        real_logspace_impl(start, stop, num, endpoint, base)
    }
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
        real_geomspace_impl(start, stop, num, endpoint)
    }
}

impl CreationElement for Complex32 {
    #[inline]
    fn from_linspace_f64(v: f64) -> Self {
        Complex32::new(v as f32, 0.0)
    }
    #[inline]
    fn from_c64(v: Complex64) -> Self {
        Complex32::new(v.re as f32, v.im as f32)
    }
    #[inline]
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
        let len = complex_arange_len(start.to_c64(), stop.to_c64(), step.to_c64())?;
        fill_arange_float(len, |i| start + step * (i as f32))
    }
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
        complex_linspace_impl(start.to_c64(), stop.to_c64(), num, endpoint)
    }
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
        let (lin, _) = complex_linspace_impl::<Complex64>(start.to_c64(), stop.to_c64(), num, endpoint)?;
        let sl = lin.as_slice().unwrap_or(&[]);
        fill_arange_float(num, |i| Self::from_c64(complex_logspace_elem(sl[i], base)))
    }
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
        complex_geomspace_impl(start.to_c64(), stop.to_c64(), num, endpoint)
    }
}

impl CreationElement for Complex64 {
    #[inline]
    fn from_linspace_f64(v: f64) -> Self {
        Complex64::new(v, 0.0)
    }
    #[inline]
    fn from_c64(v: Complex64) -> Self {
        v
    }
    #[inline]
    fn to_c64(self) -> Complex64 {
        self
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn elem_add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn elem_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn arange_impl(start: Self, stop: Self, step: Self) -> Result<NdArray<Self>> {
        let len = complex_arange_len(start, stop, step)?;
        fill_arange_float(len, |i| start + step * (i as f64))
    }
    fn linspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<(NdArray<Self>, f64)> {
        complex_linspace_impl(start, stop, num, endpoint)
    }
    fn logspace_impl(start: Self, stop: Self, num: usize, endpoint: bool, base: f64) -> Result<NdArray<Self>> {
        let (lin, _) = complex_linspace_impl::<Complex64>(start, stop, num, endpoint)?;
        let sl = lin.as_slice().unwrap_or(&[]);
        fill_arange_float(num, |i| complex_logspace_elem(sl[i], base))
    }
    fn geomspace_impl(start: Self, stop: Self, num: usize, endpoint: bool) -> Result<NdArray<Self>> {
        complex_geomspace_impl(start, stop, num, endpoint)
    }
}

// =============================================================================
// 1. empty / zeros / ones / full (+ _with_order / _like)
// =============================================================================

/// Allocates a 64-byte aligned C-order `NdArray<T>` without user-specified fill values
/// (zero-initialized safely to preserve Rust memory-safety invariants for all types).
pub fn empty<T: Scalar>(shape: &[usize]) -> Result<NdArray<T>> {
    empty_with_order(shape, Order::C)
}

/// Allocates a 64-byte aligned `NdArray<T>` with explicit `Order`.
pub fn empty_with_order<T: Scalar>(shape: &[usize], order: Order) -> Result<NdArray<T>> {
    let size = AlignedBuffer::<T>::check_shape_bytes::<T>(shape)?;
    let buf = AlignedBuffer::<T>::try_new_zeroed(size)?;
    finish_order_array(buf, shape, order)
}

/// Returns a new array with the same shape, dtype, and memory order as `prototype`.
pub fn empty_like<T: Scalar, U: Scalar>(prototype: &NdArray<U>) -> Result<NdArray<T>> {
    empty_like_with_order(prototype, Order::K)
}

/// Returns a new array with the same shape and dtype as `prototype`, honoring `order`.
pub fn empty_like_with_order<T: Scalar, U: Scalar>(
    prototype: &NdArray<U>,
    order: Order,
) -> Result<NdArray<T>> {
    empty_with_order(prototype.shape(), resolve_like_order(prototype, order))
}

/// Creates a zero-filled C-order `NdArray<T>` with 64-byte alignment.
pub fn zeros<T: Scalar>(shape: &[usize]) -> Result<NdArray<T>> {
    zeros_with_order(shape, Order::C)
}

/// Creates a zero-filled `NdArray<T>` with explicit `Order`.
pub fn zeros_with_order<T: Scalar>(shape: &[usize], order: Order) -> Result<NdArray<T>> {
    NdArray::<T>::zeros_with_order(shape, resolve_new_order(order))
}

/// Creates a zero-filled array matching `prototype`'s shape.
pub fn zeros_like<T: Scalar, U: Scalar>(prototype: &NdArray<U>) -> Result<NdArray<T>> {
    zeros_like_with_order(prototype, Order::K)
}

/// Creates a zero-filled array matching `prototype`'s shape and requested `Order`.
pub fn zeros_like_with_order<T: Scalar, U: Scalar>(
    prototype: &NdArray<U>,
    order: Order,
) -> Result<NdArray<T>> {
    zeros_with_order(prototype.shape(), resolve_like_order(prototype, order))
}

/// Creates a one-filled C-order `NdArray<T>` using parallel Rayon initialization above `RAYON_THRESHOLD`.
pub fn ones<T: Scalar + Send + Sync>(shape: &[usize]) -> Result<NdArray<T>> {
    ones_with_order(shape, Order::C)
}

/// Creates a one-filled `NdArray<T>` with explicit `Order`.
pub fn ones_with_order<T: Scalar + Send + Sync>(shape: &[usize], order: Order) -> Result<NdArray<T>> {
    full_with_order(shape, T::one(), order)
}

/// Creates a one-filled array matching `prototype`'s shape.
pub fn ones_like<T: Scalar + Send + Sync, U: Scalar>(prototype: &NdArray<U>) -> Result<NdArray<T>> {
    ones_like_with_order(prototype, Order::K)
}

/// Creates a one-filled array matching `prototype`'s shape and requested `Order`.
pub fn ones_like_with_order<T: Scalar + Send + Sync, U: Scalar>(
    prototype: &NdArray<U>,
    order: Order,
) -> Result<NdArray<T>> {
    ones_with_order(prototype.shape(), resolve_like_order(prototype, order))
}

/// Creates a C-order `NdArray<T>` filled with `fill_value`.
pub fn full<T: Scalar + Send + Sync>(shape: &[usize], fill_value: T) -> Result<NdArray<T>> {
    full_with_order(shape, fill_value, Order::C)
}

/// Creates an `NdArray<T>` filled with `fill_value` and explicit `Order`.
pub fn full_with_order<T: Scalar + Send + Sync>(
    shape: &[usize],
    fill_value: T,
    order: Order,
) -> Result<NdArray<T>> {
    let size = AlignedBuffer::<T>::check_shape_bytes::<T>(shape)?;
    let buf = try_aligned_filled_fast(size, fill_value)?;
    finish_order_array(buf, shape, order)
}

/// Creates a filled array matching `prototype`'s shape.
pub fn full_like<T: Scalar + Send + Sync, U: Scalar>(
    prototype: &NdArray<U>,
    fill_value: T,
) -> Result<NdArray<T>> {
    full_like_with_order(prototype, fill_value, Order::K)
}

/// Creates a filled array matching `prototype`'s shape and requested `Order`.
pub fn full_like_with_order<T: Scalar + Send + Sync, U: Scalar>(
    prototype: &NdArray<U>,
    fill_value: T,
    order: Order,
) -> Result<NdArray<T>> {
    full_with_order(prototype.shape(), fill_value, resolve_like_order(prototype, order))
}

// =============================================================================
// 2. Numerical ranges: arange / linspace / logspace / geomspace
// =============================================================================

/// Evenly spaced values within `[start, stop)` separated by `step`.
pub fn arange<T: CreationElement>(start: T, stop: T, step: T) -> Result<NdArray<T>> {
    T::arange_impl(start, stop, step)
}

/// Evenly spaced values `[0, 1, ..., stop - 1]` with step `1`.
pub fn arange_stop<T: CreationElement>(stop: T) -> Result<NdArray<T>> {
    T::arange_impl(T::zero(), stop, T::one())
}

/// Evenly spaced numbers over `[start, stop]` (or `[start, stop)` when `endpoint == false`).
pub fn linspace<T: CreationElement>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
) -> Result<NdArray<T>> {
    Ok(T::linspace_impl(start, stop, num, endpoint)?.0)
}

/// Variant of `linspace` also returning the sample spacing `step`.
pub fn linspace_retstep<T: CreationElement>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
) -> Result<(NdArray<T>, f64)> {
    T::linspace_impl(start, stop, num, endpoint)
}

/// Array-valued `linspace` interpolating elementwise between array-like `start` and `stop` along `axis`.
pub fn linspace_arrays<T: CreationElement>(
    start: &NdArray<T>,
    stop: &NdArray<T>,
    num: usize,
    endpoint: bool,
    axis: isize,
) -> Result<NdArray<T>> {
    let out_ndim = start.ndim().max(stop.ndim()) + 1;
    let norm_ax = numpy_types::normalize_axis(axis, out_ndim)?;
    let bshape = numpy_types::broadcast_shapes(start.shape(), stop.shape())?;
    let s_b = start.broadcast_to(&bshape)?.to_c_contiguous();
    let e_b = stop.broadcast_to(&bshape)?.to_c_contiguous();
    let slice_len = s_b.len();

    let mut out_shape = bshape.clone();
    out_shape.insert(norm_ax, num);
    let total = shape_size(&out_shape)?;

    if total == 0 {
        return zeros(&out_shape);
    }

    let div = if endpoint && num > 1 {
        (num - 1) as f64
    } else if !endpoint {
        num as f64
    } else {
        1.0
    };

    let s_sl = s_b.as_slice().unwrap_or(&[]);
    let e_sl = e_b.as_slice().unwrap_or(&[]);

    // Build `(num, slice_len...)` first and move axis 0 to `norm_ax` if needed for cache-friendly fill.
    let mut tmp_shape = Vec::with_capacity(bshape.len() + 1);
    tmp_shape.push(num);
    tmp_shape.extend_from_slice(&bshape);

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    if total >= RAYON_THRESHOLD {
        dst.par_chunks_mut(slice_len.max(1))
            .enumerate()
            .for_each(|(i, chunk)| {
                if endpoint && num > 1 && i == num - 1 {
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        slot.write(e_sl[j]);
                    }
                } else {
                    let frac = if num <= 1 { 0.0 } else { (i as f64) / div };
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        let a = s_sl[j].to_c64();
                        let b = e_sl[j].to_c64();
                        let v = a + (b - a) * frac;
                        if T::DTYPE.is_complex() {
                            slot.write(T::from_c64(v));
                        } else {
                            slot.write(T::from_linspace_f64(v.re));
                        }
                    }
                }
            });
    } else {
        for i in 0..num {
            let off = i * slice_len;
            if endpoint && num > 1 && i == num - 1 {
                for j in 0..slice_len {
                    dst[off + j].write(e_sl[j]);
                }
            } else {
                let frac = if num <= 1 { 0.0 } else { (i as f64) / div };
                for j in 0..slice_len {
                    let a = s_sl[j].to_c64();
                    let b = e_sl[j].to_c64();
                    let v = a + (b - a) * frac;
                    if T::DTYPE.is_complex() {
                        dst[off + j].write(T::from_c64(v));
                    } else {
                        dst[off + j].write(T::from_linspace_f64(v.re));
                    }
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    let arr = finish_c_array(buf, &tmp_shape)?;
    if norm_ax == 0 {
        Ok(arr)
    } else {
        let mut perm: Vec<isize> = (1..=bshape.len() as isize).collect();
        perm.insert(norm_ax, 0);
        Ok(arr.transpose(Some(&perm))?.to_c_contiguous())
    }
}

/// Logarithmically spaced numbers (`base ** linspace(start, stop, num, endpoint)`).
pub fn logspace<T: CreationElement>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
    base: f64,
) -> Result<NdArray<T>> {
    T::logspace_impl(start, stop, num, endpoint, base)
}

/// Geometrically spaced numbers from `start` to `stop`.
pub fn geomspace<T: CreationElement>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
) -> Result<NdArray<T>> {
    T::geomspace_impl(start, stop, num, endpoint)
}

// =============================================================================
// 3. Matrix & Grid builders: eye, identity, diag, diagflat, tri, tril, triu, vander, meshgrid, indices, from_fn
// =============================================================================

/// 2-D array with ones on the `k`-th diagonal and zeros elsewhere.
pub fn eye<T: Scalar + Send + Sync>(n: usize, m: Option<usize>, k: isize) -> Result<NdArray<T>> {
    eye_with_order(n, m, k, Order::C)
}

/// `eye` with explicit memory `Order`.
pub fn eye_with_order<T: Scalar + Send + Sync>(
    n: usize,
    m: Option<usize>,
    k: isize,
    order: Order,
) -> Result<NdArray<T>> {
    let cols = m.unwrap_or(n);
    let resolved = resolve_new_order(order);
    let total = AlignedBuffer::<T>::check_shape_bytes::<T>(&[n, cols])?;
    let mut buf = AlignedBuffer::<T>::try_new_zeroed(total)?;
    if n > 0 && cols > 0 {
        let start_row = if k < 0 { k.unsigned_abs() } else { 0 };
        let start_col = if k > 0 { k as usize } else { 0 };
        if start_row < n && start_col < cols {
            let diag_len = (n - start_row).min(cols - start_col);
            let sl = buf.as_mut_slice();
            let one = T::one();
            match resolved {
                Order::F => {
                    for i in 0..diag_len {
                        let r = start_row + i;
                        let c = start_col + i;
                        sl[c * n + r] = one;
                    }
                }
                _ => {
                    for i in 0..diag_len {
                        let r = start_row + i;
                        let c = start_col + i;
                        sl[r * cols + c] = one;
                    }
                }
            }
        }
    }
    finish_order_array(buf, &[n, cols], resolved)
}

/// Square `n x n` identity matrix.
pub fn identity<T: Scalar + Send + Sync>(n: usize) -> Result<NdArray<T>> {
    eye(n, Some(n), 0)
}

/// Extract the `k`-th diagonal (when `v.ndim() == 2`) or construct a diagonal 2-D array (when `v.ndim() == 1`).
pub fn diag<T: Scalar + Send + Sync>(v: &NdArray<T>, k: isize) -> Result<NdArray<T>> {
    match v.ndim() {
        1 => {
            let n = v.len();
            let dim = n.checked_add(k.unsigned_abs()).ok_or_else(|| {
                NumPyError::ShapeMismatch("diag: resulting dimension overflows usize".into())
            })?;
            let mut out = zeros::<T>(&[dim, dim])?;
            if n == 0 {
                return Ok(out);
            }
            let r0 = if k < 0 { k.unsigned_abs() } else { 0 };
            let c0 = if k > 0 { k as usize } else { 0 };
            let sl = out.as_slice_mut().ok_or_else(|| {
                NumPyError::InvalidValue("internal error: newly allocated array is not contiguous".into())
            })?;
            let view = v.view();
            let stride0 = view.strides()[0];
            let base = view.as_ptr();
            for i in 0..n {
                let val = unsafe { *base.offset((i as isize) * stride0) };
                sl[(r0 + i) * dim + (c0 + i)] = val;
            }
            Ok(out)
        }
        2 => {
            let rows = v.shape()[0];
            let cols = v.shape()[1];
            let r0 = if k < 0 { k.unsigned_abs() } else { 0 };
            let c0 = if k > 0 { k as usize } else { 0 };
            if r0 >= rows || c0 >= cols {
                return zeros(&[0]);
            }
            let diag_len = (rows - r0).min(cols - c0);
            let view = v.view();
            let s0 = view.strides()[0];
            let s1 = view.strides()[1];
            let step = s0 + s1;
            let start_ptr = unsafe { view.as_ptr().offset((r0 as isize) * s0 + (c0 as isize) * s1) };

            let mut uninit = AlignedBuffer::<T>::try_new_uninit(diag_len)?;
            let dst = uninit.as_mut_slice();
            if diag_len >= RAYON_THRESHOLD {
                let base_addr = start_ptr as usize;
                dst.par_chunks_mut(4096)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base = chunk_idx * 4096;
                        let ptr = base_addr as *const T;
                        for (i, slot) in chunk.iter_mut().enumerate() {
                            slot.write(unsafe { *ptr.offset(((base + i) as isize) * step) });
                        }
                    });
            } else {
                for (i, slot) in dst.iter_mut().enumerate() {
                    slot.write(unsafe { *start_ptr.offset((i as isize) * step) });
                }
            }
            let buf = unsafe { uninit.assume_init() };
            finish_c_array(buf, &[diag_len])
        }
        _ => Err(NumPyError::InvalidValue(format!(
            "diag requires 1-D or 2-D array, got {}-D",
            v.ndim()
        ))),
    }
}

/// Create a 2-D array with the flattened input `v` as the `k`-th diagonal.
pub fn diagflat<T: Scalar + Send + Sync>(v: &NdArray<T>, k: isize) -> Result<NdArray<T>> {
    let flat = v.ravel(Order::C).to_c_contiguous();
    diag(&flat, k)
}

/// Lower-triangular matrix of ones (`1` at `(i, j)` where `j <= i + k`).
pub fn tri<T: Scalar + Send + Sync>(n: usize, m: Option<usize>, k: isize) -> Result<NdArray<T>> {
    let cols = m.unwrap_or(n);
    let total = shape_size(&[n, cols])?;
    if total == 0 {
        return zeros(&[n, cols]);
    }
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();
    let one = T::one();
    let zero = T::zero();

    let fill_row = |row_idx: usize, row_slice: &mut [MaybeUninit<T>]| {
        let limit = (row_idx as i128) + (k as i128) + 1;
        let ones_count = if limit <= 0 {
            0
        } else {
            limit.min(cols as i128) as usize
        };
        for slot in &mut row_slice[..ones_count] {
            slot.write(one);
        }
        for slot in &mut row_slice[ones_count..] {
            slot.write(zero);
        }
    };

    if total >= RAYON_THRESHOLD && cols > 0 {
        dst.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, chunk)| fill_row(r, chunk));
    } else if cols > 0 {
        for (r, chunk) in dst.chunks_mut(cols).enumerate() {
            fill_row(r, chunk);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &[n, cols])
}

fn tri_mask_apply<T: Scalar + Send + Sync>(
    m: &NdArray<T>,
    k: isize,
    keep_lower: bool,
) -> Result<NdArray<T>> {
    if m.ndim() == 0 {
        return Err(NumPyError::InvalidValue(
            "tril/triu require at least 1-D input".into(),
        ));
    }
    let src_owned;
    let src = if m.ndim() == 1 {
        let n = m.shape()[0];
        src_owned = m.broadcast_to(&[n, n])?.to_c_contiguous();
        &src_owned
    } else {
        m
    };

    let shape = src.shape();
    let ndim = shape.len();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let total = src.len();
    if total == 0 {
        return zeros(shape);
    }

    let c_arr = src.to_c_contiguous();
    let src_sl = c_arr.as_slice().unwrap_or(&[]);
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();
    let zero = T::zero();
    let mat_len = rows * cols;

    #[inline]
    fn write_tri_row<T: Copy>(
        r: usize,
        cols: usize,
        k: isize,
        keep_lower: bool,
        zero: T,
        s_row: &[T],
        d_row: &mut [MaybeUninit<T>],
    ) {
        let limit = (r as i128) + (k as i128);
        for c in 0..cols {
            let keep = if keep_lower {
                (c as i128) <= limit
            } else {
                (c as i128) >= limit
            };
            d_row[c].write(if keep { s_row[c] } else { zero });
        }
    }

    let process_matrix = |mat_src: &[T], mat_dst: &mut [MaybeUninit<T>]| {
        for r in 0..rows {
            let row_off = r * cols;
            write_tri_row(
                r,
                cols,
                k,
                keep_lower,
                zero,
                &mat_src[row_off..row_off + cols],
                &mut mat_dst[row_off..row_off + cols],
            );
        }
    };

    if mat_len > 0 {
        if total >= RAYON_THRESHOLD && (total / mat_len) > 1 {
            dst.par_chunks_mut(mat_len)
                .zip(src_sl.par_chunks(mat_len))
                .for_each(|(d, s)| process_matrix(s, d));
        } else if total >= RAYON_THRESHOLD && mat_len == total {
            dst.par_chunks_mut(cols)
                .zip(src_sl.par_chunks(cols))
                .enumerate()
                .for_each(|(r, (d_row, s_row))| {
                    write_tri_row(r, cols, k, keep_lower, zero, s_row, d_row);
                });
        } else {
            for (d, s) in dst.chunks_mut(mat_len).zip(src_sl.chunks(mat_len)) {
                process_matrix(s, d);
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, shape)
}

/// Lower triangle of an array: zeros elements above the `k`-th diagonal.
pub fn tril<T: Scalar + Send + Sync>(m: &NdArray<T>, k: isize) -> Result<NdArray<T>> {
    tri_mask_apply(m, k, true)
}

/// Upper triangle of an array: zeros elements below the `k`-th diagonal.
pub fn triu<T: Scalar + Send + Sync>(m: &NdArray<T>, k: isize) -> Result<NdArray<T>> {
    tri_mask_apply(m, k, false)
}

/// Vandermonde matrix from 1-D input array `x`.
pub fn vander<T: CreationElement>(
    x: &NdArray<T>,
    n: Option<usize>,
    increasing: bool,
) -> Result<NdArray<T>> {
    if x.ndim() != 1 {
        return Err(NumPyError::InvalidValue(format!(
            "x must be a one-dimensional array, got {}-D",
            x.ndim()
        )));
    }
    let rows = x.len();
    let cols = n.unwrap_or(rows);
    let total = shape_size(&[rows, cols])?;
    if total == 0 {
        return zeros(&[rows, cols]);
    }
    let x_c = x.to_c_contiguous();
    let x_sl = x_c.as_slice().unwrap_or(&[]);

    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();
    let one = T::one();

    let fill_row = |val: T, row: &mut [MaybeUninit<T>]| {
        if increasing {
            let mut cur = one;
            for (j, slot) in row.iter_mut().enumerate() {
                if j == 0 {
                    slot.write(one);
                } else {
                    cur = cur.elem_mul(val);
                    slot.write(cur);
                }
            }
        } else {
            let mut cur = one;
            for j in (0..cols).rev() {
                if j == cols - 1 {
                    row[j].write(one);
                } else {
                    cur = cur.elem_mul(val);
                    row[j].write(cur);
                }
            }
        }
    };

    if total >= RAYON_THRESHOLD {
        dst.par_chunks_mut(cols)
            .zip(x_sl.par_iter())
            .for_each(|(row, &val)| fill_row(val, row));
    } else {
        for (row, &val) in dst.chunks_mut(cols).zip(x_sl.iter()) {
            fill_row(val, row);
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &[rows, cols])
}

/// N-D coordinate arrays from 1-D coordinate vectors.
/// When `copy == false`, returns zero-copy broadcast views sharing storage with contiguous inputs.
pub fn meshgrid<T: Scalar + Send + Sync>(
    xi: &[&NdArray<T>],
    indexing: MeshgridIndexing,
    sparse: bool,
    copy: bool,
) -> Result<Vec<NdArray<T>>> {
    let ndim = xi.len();
    if ndim == 0 {
        return Ok(Vec::new());
    }

    let mut flat_vecs: Vec<NdArray<T>> = Vec::with_capacity(ndim);
    let mut base_shape = Vec::with_capacity(ndim);
    for &arr in xi {
        let f = arr.ravel(Order::C);
        base_shape.push(f.len());
        flat_vecs.push(f);
    }

    if ndim == 1 {
        let out0 = if copy {
            flat_vecs[0].deep_copy(Order::C)
        } else {
            flat_vecs[0].clone()
        };
        return Ok(vec![out0]);
    }

    let mut full_shape = base_shape.clone();
    if indexing == MeshgridIndexing::Xy {
        full_shape.swap(0, 1);
    }

    let mut out = Vec::with_capacity(ndim);
    for (idx, vec1d) in flat_vecs.into_iter().enumerate() {
        let out_axis = if indexing == MeshgridIndexing::Xy && idx < 2 {
            1 - idx
        } else {
            idx
        };
        let mut s = vec![1isize; ndim];
        s[out_axis] = vec1d.len() as isize;
        let reshaped = vec1d.reshape(&s)?;
        let grid = if sparse {
            reshaped
        } else {
            reshaped.broadcast_to(&full_shape)?
        };
        if copy {
            out.push(grid.to_c_contiguous());
        } else {
            out.push(grid);
        }
    }
    Ok(out)
}

/// Returns an `NdArray<i64>` representing a dense coordinate grid of shape `(N, d_0, ..., d_{N-1})`.
pub fn indices_dense(dimensions: &[usize]) -> Result<NdArray<i64>> {
    let n = dimensions.len();
    if n == 0 {
        return zeros::<i64>(&[0]);
    }
    let mut out_shape = Vec::with_capacity(n + 1);
    out_shape.push(n);
    out_shape.extend_from_slice(dimensions);
    let grid_size = shape_size(dimensions)?;
    let total = shape_size(&out_shape)?;
    if total == 0 {
        return zeros::<i64>(&out_shape);
    }

    let mut uninit = AlignedBuffer::<i64>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();

    for (axis, chunk) in dst.chunks_mut(grid_size).enumerate() {
        let dim = dimensions[axis];
        let inner: usize = dimensions[axis + 1..].iter().product();
        let outer: usize = dimensions[..axis].iter().product();
        let tile_len = dim * inner;

        if grid_size >= RAYON_THRESHOLD && outer >= 4 {
            chunk
                .par_chunks_mut(tile_len)
                .for_each(|tile| {
                    for idx in 0..dim {
                        let off = idx * inner;
                        let val = idx as i64;
                        for slot in &mut tile[off..off + inner] {
                            slot.write(val);
                        }
                    }
                });
        } else {
            for tile in chunk.chunks_mut(tile_len) {
                for idx in 0..dim {
                    let off = idx * inner;
                    let val = idx as i64;
                    for slot in &mut tile[off..off + inner] {
                        slot.write(val);
                    }
                }
            }
        }
    }

    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, &out_shape)
}

/// Returns a sparse list of `i64` coordinate arrays of shapes `(d_0, 1, ...)`, `(1, d_1, ...)`, etc.
pub fn indices_sparse(dimensions: &[usize]) -> Result<Vec<NdArray<i64>>> {
    let n = dimensions.len();
    let mut out = Vec::with_capacity(n);
    for (axis, &dim) in dimensions.iter().enumerate() {
        let mut s = vec![1isize; n];
        s[axis] = dim as isize;
        let arr = arange(0i64, dim as i64, 1)?;
        out.push(arr.reshape(&s)?);
    }
    Ok(out)
}

/// Convenience `indices` matching `np.indices(dimensions, sparse=false)` (dense stacked array).
pub fn indices(dimensions: &[usize]) -> Result<NdArray<i64>> {
    indices_dense(dimensions)
}

/// Construct an array by executing `f(&coords)` over each coordinate tuple.
pub fn from_fn<T, F>(shape: &[usize], mut f: F) -> Result<NdArray<T>>
where
    T: Scalar,
    F: FnMut(&[usize]) -> T,
{
    let total = shape_size(shape)?;
    if total == 0 {
        return zeros(shape);
    }
    let ndim = shape.len();
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();
    let mut coords = vec![0usize; ndim];
    for (linear, slot) in dst.iter_mut().enumerate() {
        slot.write(f(&coords));
        if linear + 1 < total {
            for d in (0..ndim).rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
    }
    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, shape)
}

/// Parallel version of `from_fn` using Rayon above `RAYON_THRESHOLD`.
pub fn from_fn_par<T, F>(shape: &[usize], f: F) -> Result<NdArray<T>>
where
    T: Scalar + Send + Sync,
    F: Fn(&[usize]) -> T + Sync,
{
    let total = shape_size(shape)?;
    if total == 0 {
        return zeros(shape);
    }
    if total < RAYON_THRESHOLD {
        return from_fn(shape, f);
    }
    let ndim = shape.len();
    let mut uninit = AlignedBuffer::<T>::try_new_uninit(total)?;
    let dst = uninit.as_mut_slice();
    const CHUNK: usize = 2048;
    dst.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start_linear = chunk_idx * CHUNK;
            let mut coords = vec![0usize; ndim];
            linear_to_coords(start_linear, shape, &mut coords);
            let chunk_len = chunk.len();
            for (i, slot) in chunk.iter_mut().enumerate() {
                slot.write(f(&coords));
                if i + 1 < chunk_len {
                    for d in (0..ndim).rev() {
                        coords[d] += 1;
                        if coords[d] < shape[d] {
                            break;
                        }
                        coords[d] = 0;
                    }
                }
            }
        });
    let buf = unsafe { uninit.assume_init() };
    finish_c_array(buf, shape)
}

// =============================================================================
// 4. DynArray Creation Functions
// =============================================================================

macro_rules! dispatch_creation_dtype {
    ($dtype:expr, $macro_fn:ident $(, $arg:expr)*) => {
        match $dtype {
            DType::Bool => $macro_fn!(bool, Bool $(, $arg)*),
            DType::I8 => $macro_fn!(i8, I8 $(, $arg)*),
            DType::I16 => $macro_fn!(i16, I16 $(, $arg)*),
            DType::I32 => $macro_fn!(i32, I32 $(, $arg)*),
            DType::I64 => $macro_fn!(i64, I64 $(, $arg)*),
            DType::U8 => $macro_fn!(u8, U8 $(, $arg)*),
            DType::U16 => $macro_fn!(u16, U16 $(, $arg)*),
            DType::U32 => $macro_fn!(u32, U32 $(, $arg)*),
            DType::U64 => $macro_fn!(u64, U64 $(, $arg)*),
            DType::F16 => $macro_fn!(f16, F16 $(, $arg)*),
            DType::F32 => $macro_fn!(f32, F32 $(, $arg)*),
            DType::F64 => $macro_fn!(f64, F64 $(, $arg)*),
            DType::C64 => $macro_fn!(Complex32, C64 $(, $arg)*),
            DType::C128 => $macro_fn!(Complex64, C128 $(, $arg)*),
        }
    };
}

/// Creates a `DynArray` with uninitialized-equivalent zeroed memory and given `DType` & `Order`.
pub fn dyn_empty(shape: &[usize], dtype: DType, order: Order) -> Result<DynArray> {
    macro_rules! do_empty {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(empty_with_order::<$t>(shape, order)?))
        };
    }
    dispatch_creation_dtype!(dtype, do_empty)
}

/// Creates an `empty_like` `DynArray` matching `prototype`'s shape, dtype (unless overridden), and order.
pub fn dyn_empty_like(
    prototype: &DynArray,
    dtype: Option<DType>,
    order: Order,
) -> Result<DynArray> {
    let dt = dtype.unwrap_or_else(|| prototype.dtype());
    let ord = match order {
        Order::C => Order::C,
        Order::F => Order::F,
        Order::A | Order::K => {
            if prototype.is_f_contiguous() && !prototype.is_c_contiguous() {
                Order::F
            } else {
                Order::C
            }
        }
    };
    dyn_empty(prototype.shape(), dt, ord)
}

/// Creates a zero-filled `DynArray` of given `DType` and `Order`.
pub fn dyn_zeros(shape: &[usize], dtype: DType, order: Order) -> Result<DynArray> {
    macro_rules! do_zeros {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(zeros_with_order::<$t>(shape, order)?))
        };
    }
    dispatch_creation_dtype!(dtype, do_zeros)
}

/// Creates a zero-filled `DynArray` matching `prototype`.
pub fn dyn_zeros_like(
    prototype: &DynArray,
    dtype: Option<DType>,
    order: Order,
) -> Result<DynArray> {
    let dt = dtype.unwrap_or_else(|| prototype.dtype());
    let ord = match order {
        Order::C => Order::C,
        Order::F => Order::F,
        Order::A | Order::K => {
            if prototype.is_f_contiguous() && !prototype.is_c_contiguous() {
                Order::F
            } else {
                Order::C
            }
        }
    };
    dyn_zeros(prototype.shape(), dt, ord)
}

/// Creates a one-filled `DynArray` of given `DType` and `Order`.
pub fn dyn_ones(shape: &[usize], dtype: DType, order: Order) -> Result<DynArray> {
    macro_rules! do_ones {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(ones_with_order::<$t>(shape, order)?))
        };
    }
    dispatch_creation_dtype!(dtype, do_ones)
}

/// Creates a one-filled `DynArray` matching `prototype`.
pub fn dyn_ones_like(
    prototype: &DynArray,
    dtype: Option<DType>,
    order: Order,
) -> Result<DynArray> {
    let dt = dtype.unwrap_or_else(|| prototype.dtype());
    let ord = match order {
        Order::C => Order::C,
        Order::F => Order::F,
        Order::A | Order::K => {
            if prototype.is_f_contiguous() && !prototype.is_c_contiguous() {
                Order::F
            } else {
                Order::C
            }
        }
    };
    dyn_ones(prototype.shape(), dt, ord)
}

/// Creates a `DynArray` filled with `fill_value` (inferred dtype if `dtype` is `None`).
pub fn dyn_full(
    shape: &[usize],
    fill_value: DynScalar,
    dtype: Option<DType>,
    order: Order,
) -> Result<DynArray> {
    let dt = dtype.unwrap_or_else(|| fill_value.dtype());
    macro_rules! do_full {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(full_with_order::<$t>(
                shape,
                fill_value.cast_to::<$t>(),
                order,
            )?))
        };
    }
    dispatch_creation_dtype!(dt, do_full)
}

/// Creates a filled `DynArray` matching `prototype`.
pub fn dyn_full_like(
    prototype: &DynArray,
    fill_value: DynScalar,
    dtype: Option<DType>,
    order: Order,
) -> Result<DynArray> {
    let dt = dtype.unwrap_or_else(|| prototype.dtype());
    let ord = match order {
        Order::C => Order::C,
        Order::F => Order::F,
        Order::A | Order::K => {
            if prototype.is_f_contiguous() && !prototype.is_c_contiguous() {
                Order::F
            } else {
                Order::C
            }
        }
    };
    dyn_full(prototype.shape(), fill_value, Some(dt), ord)
}

/// Dynamic `arange` supporting arbitrary scalar dtypes and NumPy type promotion when `dtype` is `None`.
pub fn dyn_arange(
    start: DynScalar,
    stop: Option<DynScalar>,
    step: Option<DynScalar>,
    dtype: Option<DType>,
) -> Result<DynArray> {
    let (s0, s1) = match stop {
        Some(b) => (start, b),
        None => {
            let zero = DynScalar::I64(0);
            (zero, start)
        }
    };
    let st = step.unwrap_or(DynScalar::I64(1));
    let target_dt = match dtype {
        Some(dt) => dt,
        None => numpy_types::result_type(&[s0.dtype(), s1.dtype(), st.dtype()])?,
    };

    macro_rules! do_arange {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(arange::<$t>(
                s0.cast_to::<$t>(),
                s1.cast_to::<$t>(),
                st.cast_to::<$t>(),
            )?))
        };
    }
    dispatch_creation_dtype!(target_dt, do_arange)
}

/// Dynamic `linspace` returning `(DynArray, step)` with `dtype` defaulting to `F64` (or `C128` if complex).
pub fn dyn_linspace_retstep(
    start: DynScalar,
    stop: DynScalar,
    num: usize,
    endpoint: bool,
    dtype: Option<DType>,
) -> Result<(DynArray, f64)> {
    let dt = match dtype {
        Some(dt) => dt,
        None => {
            let base = numpy_types::promote_types(start.dtype(), stop.dtype());
            if base.is_complex() {
                DType::C128
            } else {
                DType::F64
            }
        }
    };
    macro_rules! do_lin {
        ($t:ty, $var:ident) => {{
            let (arr, step) = linspace_retstep::<$t>(
                start.cast_to::<$t>(),
                stop.cast_to::<$t>(),
                num,
                endpoint,
            )?;
            Ok((DynArray::$var(arr), step))
        }};
    }
    dispatch_creation_dtype!(dt, do_lin)
}

/// Dynamic `linspace`.
pub fn dyn_linspace(
    start: DynScalar,
    stop: DynScalar,
    num: usize,
    endpoint: bool,
    dtype: Option<DType>,
) -> Result<DynArray> {
    Ok(dyn_linspace_retstep(start, stop, num, endpoint, dtype)?.0)
}

/// Dynamic `logspace`.
pub fn dyn_logspace(
    start: DynScalar,
    stop: DynScalar,
    num: usize,
    endpoint: bool,
    base: f64,
    dtype: Option<DType>,
) -> Result<DynArray> {
    let dt = match dtype {
        Some(dt) => dt,
        None => {
            let promoted = numpy_types::promote_types(start.dtype(), stop.dtype());
            if promoted.is_complex() {
                DType::C128
            } else {
                DType::F64
            }
        }
    };
    macro_rules! do_log {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(logspace::<$t>(
                start.cast_to::<$t>(),
                stop.cast_to::<$t>(),
                num,
                endpoint,
                base,
            )?))
        };
    }
    dispatch_creation_dtype!(dt, do_log)
}

/// Dynamic `geomspace`.
pub fn dyn_geomspace(
    start: DynScalar,
    stop: DynScalar,
    num: usize,
    endpoint: bool,
    dtype: Option<DType>,
) -> Result<DynArray> {
    let dt = match dtype {
        Some(dt) => dt,
        None => {
            let promoted = numpy_types::promote_types(start.dtype(), stop.dtype());
            if promoted.is_complex() {
                DType::C128
            } else {
                DType::F64
            }
        }
    };
    macro_rules! do_geom {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(geomspace::<$t>(
                start.cast_to::<$t>(),
                stop.cast_to::<$t>(),
                num,
                endpoint,
            )?))
        };
    }
    dispatch_creation_dtype!(dt, do_geom)
}

/// Dynamic `eye`.
pub fn dyn_eye(
    n: usize,
    m: Option<usize>,
    k: isize,
    dtype: DType,
    order: Order,
) -> Result<DynArray> {
    macro_rules! do_eye {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(eye_with_order::<$t>(n, m, k, order)?))
        };
    }
    dispatch_creation_dtype!(dtype, do_eye)
}

/// Dynamic `identity`.
pub fn dyn_identity(n: usize, dtype: DType) -> Result<DynArray> {
    dyn_eye(n, Some(n), 0, dtype, Order::C)
}

/// Dynamic `diag`.
pub fn dyn_diag(v: &DynArray, k: isize) -> Result<DynArray> {
    dispatch_dyn!(v, a => diag(a, k).map(DynArray::from))
}

/// Dynamic `diagflat`.
pub fn dyn_diagflat(v: &DynArray, k: isize) -> Result<DynArray> {
    dispatch_dyn!(v, a => diagflat(a, k).map(DynArray::from))
}

/// Dynamic `tri`.
pub fn dyn_tri(n: usize, m: Option<usize>, k: isize, dtype: DType) -> Result<DynArray> {
    macro_rules! do_tri {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(tri::<$t>(n, m, k)?))
        };
    }
    dispatch_creation_dtype!(dtype, do_tri)
}

/// Dynamic `tril`.
pub fn dyn_tril(m: &DynArray, k: isize) -> Result<DynArray> {
    dispatch_dyn!(m, a => tril(a, k).map(DynArray::from))
}

/// Dynamic `triu`.
pub fn dyn_triu(m: &DynArray, k: isize) -> Result<DynArray> {
    dispatch_dyn!(m, a => triu(a, k).map(DynArray::from))
}

/// Dynamic `vander`.
pub fn dyn_vander(x: &DynArray, n: Option<usize>, increasing: bool) -> Result<DynArray> {
    dispatch_dyn!(x, a => vander(a, n, increasing).map(DynArray::from))
}

/// Dynamic `meshgrid` handling heterogeneous dtypes directly via per-element broadcast views.
pub fn dyn_meshgrid(
    xi: &[&DynArray],
    indexing: MeshgridIndexing,
    sparse: bool,
    copy: bool,
) -> Result<Vec<DynArray>> {
    let ndim = xi.len();
    if ndim == 0 {
        return Ok(Vec::new());
    }
    if ndim == 1 {
        let flat = xi[0].reshape(&[-1])?;
        if copy {
            return Ok(vec![flat.to_c_contiguous()]);
        }
        return Ok(vec![flat]);
    }
    let mut base_shape = Vec::with_capacity(ndim);
    let mut flats = Vec::with_capacity(ndim);
    for &arr in xi {
        let f = arr.reshape(&[-1])?;
        base_shape.push(f.len());
        flats.push(f);
    }
    let mut full_shape = base_shape.clone();
    if indexing == MeshgridIndexing::Xy {
        full_shape.swap(0, 1);
    }
    let mut out = Vec::with_capacity(ndim);
    for (idx, arr) in flats.into_iter().enumerate() {
        let out_axis = if indexing == MeshgridIndexing::Xy && idx < 2 {
            1 - idx
        } else {
            idx
        };
        let mut s: Vec<isize> = vec![1; ndim];
        s[out_axis] = arr.len() as isize;
        let reshaped = arr.reshape(&s)?;
        let grid = if sparse {
            reshaped
        } else {
            reshaped.broadcast_to(&full_shape)?
        };
        out.push(if copy {
            grid.to_c_contiguous()
        } else {
            grid
        });
    }
    Ok(out)
}

/// Dynamic `indices` returning dense `DynArray` in requested `dtype` (defaults to `I64`).
pub fn dyn_indices(dimensions: &[usize], dtype: Option<DType>) -> Result<DynArray> {
    let base = indices_dense(dimensions)?;
    let dyn_i64 = DynArray::I64(base);
    match dtype {
        Some(dt) if dt != DType::I64 => dyn_i64.astype(dt, numpy_types::Casting::Unsafe),
        _ => Ok(dyn_i64),
    }
}

/// Dynamic sparse `indices`.
pub fn dyn_indices_sparse(dimensions: &[usize], dtype: Option<DType>) -> Result<Vec<DynArray>> {
    let parts = indices_sparse(dimensions)?;
    let dt = dtype.unwrap_or(DType::I64);
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        let d = DynArray::I64(p);
        if dt == DType::I64 {
            out.push(d);
        } else {
            out.push(d.astype(dt, numpy_types::Casting::Unsafe)?);
        }
    }
    Ok(out)
}

/// Dynamic `from_fn` constructing an array of `dtype` using `f(&coords) -> DynScalar`.
pub fn dyn_from_fn<F>(shape: &[usize], dtype: DType, mut f: F) -> Result<DynArray>
where
    F: FnMut(&[usize]) -> DynScalar,
{
    macro_rules! do_fn {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(from_fn::<$t, _>(shape, |c| {
                f(c).cast_to::<$t>()
            })?))
        };
    }
    dispatch_creation_dtype!(dtype, do_fn)
}

/// Parallel dynamic `from_fn` using Rayon above `RAYON_THRESHOLD`.
pub fn dyn_from_fn_par<F>(shape: &[usize], dtype: DType, f: F) -> Result<DynArray>
where
    F: Fn(&[usize]) -> DynScalar + Sync,
{
    macro_rules! do_fn {
        ($t:ty, $var:ident) => {
            Ok(DynArray::$var(from_fn_par::<$t, _>(shape, |c| {
                f(c).cast_to::<$t>()
            })?))
        };
    }
    dispatch_creation_dtype!(dtype, do_fn)
}

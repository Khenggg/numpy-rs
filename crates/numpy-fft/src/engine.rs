use crate::planner::{FftFloat, Norm};
use num_complex::Complex;
use numpy_types::{
    compute_c_strides, normalize_axis, shape_size, AlignedBuffer, ArrayView, NdArray, NumPyError,
    Order, Result, Scalar,
};
use rayon::prelude::*;
use rustfft::FftDirection;
use std::mem::MaybeUninit;

/// Minimum total work (`num_lanes * n`) before spawning Rayon tasks.
const RAYON_WORK_THRESHOLD: usize = 16_384;
/// Target elements per Rayon chunk for L2 cache locality.
const TARGET_CHUNK_ELEMS: usize = 32_768;
/// Number of adjacent lanes packed together into an L2 cache tile for strided axes.
const TILE_LANES: usize = 16;

#[derive(Clone, Copy)]
struct SendPtr<T>(*const T);
impl<T> SendPtr<T> {
    #[inline(always)]
    fn ptr(self) -> *const T {
        self.0
    }
}
// SAFETY: Shared read-only access over an immutable `ArrayView` borrow.
unsafe impl<T: Sync> Send for SendPtr<T> {}
unsafe impl<T: Sync> Sync for SendPtr<T> {}

#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
impl<T> SendMutPtr<T> {
    #[inline(always)]
    fn ptr(self) -> *mut T {
        self.0
    }
}
// SAFETY: Parallel workers only access strictly disjoint coordinates `(outer_coords, 0..axis_len)`
// mapped bijectively onto a unique contiguous output buffer.
unsafe impl<T: Send> Send for SendMutPtr<T> {}
unsafe impl<T: Send> Sync for SendMutPtr<T> {}

/// Trait mapping an input element scalar type into a target `FftFloat` real or complex value.
pub trait ToFftValue<F: FftFloat>: Scalar {
    fn to_real(self) -> F;
    fn to_complex(self) -> Complex<F>;
}

impl ToFftValue<f32> for f32 {
    #[inline]
    fn to_real(self) -> f32 {
        self
    }
    #[inline]
    fn to_complex(self) -> Complex<f32> {
        Complex::new(self, 0.0)
    }
}

impl ToFftValue<f32> for Complex<f32> {
    #[inline]
    fn to_real(self) -> f32 {
        self.re
    }
    #[inline]
    fn to_complex(self) -> Complex<f32> {
        self
    }
}

impl ToFftValue<f64> for f32 {
    #[inline]
    fn to_real(self) -> f64 {
        self as f64
    }
    #[inline]
    fn to_complex(self) -> Complex<f64> {
        Complex::new(self as f64, 0.0)
    }
}

impl ToFftValue<f64> for Complex<f32> {
    #[inline]
    fn to_real(self) -> f64 {
        self.re as f64
    }
    #[inline]
    fn to_complex(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
}

impl ToFftValue<f64> for f64 {
    #[inline]
    fn to_real(self) -> f64 {
        self
    }
    #[inline]
    fn to_complex(self) -> Complex<f64> {
        Complex::new(self, 0.0)
    }
}

impl ToFftValue<f64> for Complex<f64> {
    #[inline]
    fn to_real(self) -> f64 {
        self.re
    }
    #[inline]
    fn to_complex(self) -> Complex<f64> {
        self
    }
}

macro_rules! impl_to_fft_f64 {
    ($($t:ty),* $(,)?) => {
        $(
            impl ToFftValue<f64> for $t {
                #[inline]
                fn to_real(self) -> f64 {
                    self.to_f64_lossy()
                }
                #[inline]
                fn to_complex(self) -> Complex<f64> {
                    Complex::new(self.to_f64_lossy(), 0.0)
                }
            }
        )*
    };
}

impl_to_fft_f64!(bool, i8, i16, i32, i64, u8, u16, u32, u64, numpy_types::f16);

impl ToFftValue<f32> for numpy_types::f16 {
    #[inline]
    fn to_real(self) -> f32 { self.to_f32() }
    #[inline]
    fn to_complex(self) -> Complex<f32> { Complex::new(self.to_f32(), 0.0) }
}

struct LaneLayout {
    outer_dims: Vec<usize>,
    outer_in_strides: Vec<isize>,
    outer_out_strides: Vec<isize>,
    in_axis_len: usize,
    in_axis_stride: isize,
    out_axis_stride: isize,
    num_lanes: usize,
}

impl LaneLayout {
    fn new(
        active_shape: &[usize],
        in_strides: &[isize],
        out_strides: &[isize],
        axis: usize,
        in_axis_len: usize,
    ) -> Self {
        let ndim = active_shape.len();
        let mut outer_dims = Vec::with_capacity(ndim.saturating_sub(1));
        let mut outer_in_strides = Vec::with_capacity(ndim.saturating_sub(1));
        let mut outer_out_strides = Vec::with_capacity(ndim.saturating_sub(1));
        let mut num_lanes = 1usize;
        for d in 0..ndim {
            if d != axis {
                outer_dims.push(active_shape[d]);
                outer_in_strides.push(in_strides[d]);
                outer_out_strides.push(out_strides[d]);
                num_lanes *= active_shape[d];
            }
        }
        Self {
            outer_dims,
            outer_in_strides,
            outer_out_strides,
            in_axis_len,
            in_axis_stride: in_strides[axis],
            out_axis_stride: out_strides[axis],
            num_lanes,
        }
    }
}

struct LaneCursor {
    coords: Vec<usize>,
    in_off: isize,
    out_off: isize,
}

impl LaneCursor {
    fn at_lane(layout: &LaneLayout, mut lane_idx: usize) -> Self {
        let k = layout.outer_dims.len();
        let mut coords = vec![0usize; k];
        let mut in_off = 0isize;
        let mut out_off = 0isize;
        for d in (0..k).rev() {
            let dim = layout.outer_dims[d];
            let c = lane_idx % dim;
            lane_idx /= dim;
            coords[d] = c;
            in_off += (c as isize) * layout.outer_in_strides[d];
            out_off += (c as isize) * layout.outer_out_strides[d];
        }
        Self {
            coords,
            in_off,
            out_off,
        }
    }

    #[inline]
    fn advance(&mut self, layout: &LaneLayout) {
        for d in (0..layout.outer_dims.len()).rev() {
            self.coords[d] += 1;
            if self.coords[d] < layout.outer_dims[d] {
                self.in_off += layout.outer_in_strides[d];
                self.out_off += layout.outer_out_strides[d];
                return;
            }
            let reset = (self.coords[d] - 1) as isize;
            self.in_off -= reset * layout.outer_in_strides[d];
            self.out_off -= reset * layout.outer_out_strides[d];
            self.coords[d] = 0;
        }
    }

    #[inline]
    fn last_coord(&self) -> usize {
        self.coords.last().copied().unwrap_or(0)
    }

    #[inline]
    fn advance_by(&mut self, layout: &LaneLayout, count: usize) {
        if let Some(last) = self.coords.len().checked_sub(1) {
            if self.coords[last] + count < layout.outer_dims[last] {
                self.coords[last] += count;
                self.in_off += (count as isize) * layout.outer_in_strides[last];
                self.out_off += (count as isize) * layout.outer_out_strides[last];
                return;
            }
        }
        for _ in 0..count {
            self.advance(layout);
        }
    }
}

#[inline]
fn can_tile_outer(layout: &LaneLayout, require_in_unit: bool) -> Option<usize> {
    let last_dim = *layout.outer_dims.last()?;
    if last_dim <= 1 || layout.outer_out_strides.last() != Some(&1) {
        return None;
    }
    if require_in_unit && layout.outer_in_strides.last() != Some(&1) {
        return None;
    }
    Some(last_dim)
}

#[inline]
fn tile_dims(axis_len: usize) -> (usize, usize) {
    let tile_cap = TILE_LANES.min((65_536 / axis_len.max(1)).max(1));
    let padded = if axis_len % 8 == 0 {
        axis_len + 1
    } else {
        axis_len
    };
    (tile_cap, padded)
}

#[inline]
fn validate_fft_n(n: usize) -> Result<usize> {
    if n < 1 {
        return Err(NumPyError::InvalidValue(format!(
            "Invalid number of FFT data points ({n}) specified."
        )));
    }
    Ok(n)
}

#[inline]
fn compute_chunk_lanes(num_lanes: usize, n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let max_balanced = (num_lanes / (threads * 2)).max(1);
    let cache_target = (TARGET_CHUNK_ELEMS / n.max(1)).max(1);
    cache_target.min(max_balanced).max(1)
}

#[inline]
fn run_lane_chunks<F>(num_lanes: usize, n: usize, worker: F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    if num_lanes <= 1 || num_lanes * n < RAYON_WORK_THRESHOLD {
        worker(0, num_lanes);
        return;
    }
    let chunk_lanes = compute_chunk_lanes(num_lanes, n);
    let num_chunks = num_lanes.div_ceil(chunk_lanes);
    (0..num_chunks).into_par_iter().for_each(|c| {
        let start = c * chunk_lanes;
        let end = (start + chunk_lanes).min(num_lanes);
        worker(start, end);
    });
}

#[inline]
unsafe fn read_complex_lane_uninit<In: ToFftValue<F>, F: FftFloat>(
    src: *const In,
    stride: isize,
    copy_len: usize,
    dst: &mut [MaybeUninit<Complex<F>>],
) {
    if stride == 1 {
        let sl = unsafe { std::slice::from_raw_parts(src, copy_len) };
        for (d, &s) in dst[..copy_len].iter_mut().zip(sl.iter()) {
            d.write(s.to_complex());
        }
    } else {
        let mut p = src;
        for d in dst[..copy_len].iter_mut() {
            d.write(unsafe { (*p).to_complex() });
            p = unsafe { p.offset(stride) };
        }
    }
    if copy_len < dst.len() {
        dst[copy_len..].fill(MaybeUninit::new(Complex::new(
            <F as Scalar>::zero(),
            <F as Scalar>::zero(),
        )));
    }
}

#[inline]
unsafe fn read_complex_lane<In: ToFftValue<F>, F: FftFloat>(
    src: *const In,
    stride: isize,
    copy_len: usize,
    dst: &mut [Complex<F>],
) {
    let dst_uninit = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut MaybeUninit<Complex<F>>, dst.len())
    };
    unsafe { read_complex_lane_uninit(src, stride, copy_len, dst_uninit) };
}

#[inline]
unsafe fn read_real_lane<In: ToFftValue<F>, F: FftFloat>(
    src: *const In,
    stride: isize,
    copy_len: usize,
    dst: &mut [F],
) {
    if stride == 1 {
        let sl = unsafe { std::slice::from_raw_parts(src, copy_len) };
        for (d, &s) in dst[..copy_len].iter_mut().zip(sl.iter()) {
            *d = s.to_real();
        }
    } else {
        let mut p = src;
        for d in dst[..copy_len].iter_mut() {
            *d = unsafe { (*p).to_real() };
            p = unsafe { p.offset(stride) };
        }
    }
    if copy_len < dst.len() {
        dst[copy_len..].fill(<F as Scalar>::zero());
    }
}

#[inline]
unsafe fn write_complex_lane<F: FftFloat>(
    src: &[Complex<F>],
    dst: *mut Complex<F>,
    stride: isize,
    scale: Option<F>,
    conj: bool,
) {
    if stride == 1 {
        let out_sl = unsafe {
            std::slice::from_raw_parts_mut(dst as *mut MaybeUninit<Complex<F>>, src.len())
        };
        match (scale, conj) {
            (None, false) => unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            },
            (Some(s), false) => {
                for (d, &v) in out_sl.iter_mut().zip(src.iter()) {
                    d.write(Complex::new(v.re * s, v.im * s));
                }
            }
            (None, true) => {
                for (d, &v) in out_sl.iter_mut().zip(src.iter()) {
                    d.write(Complex::new(v.re, -v.im));
                }
            }
            (Some(s), true) => {
                for (d, &v) in out_sl.iter_mut().zip(src.iter()) {
                    d.write(Complex::new(v.re * s, -v.im * s));
                }
            }
        }
    } else {
        let mut p = dst;
        match (scale, conj) {
            (None, false) => {
                for &v in src {
                    unsafe { p.write(v) };
                    p = unsafe { p.offset(stride) };
                }
            }
            (Some(s), false) => {
                for &v in src {
                    unsafe { p.write(Complex::new(v.re * s, v.im * s)) };
                    p = unsafe { p.offset(stride) };
                }
            }
            (None, true) => {
                for &v in src {
                    unsafe { p.write(Complex::new(v.re, -v.im)) };
                    p = unsafe { p.offset(stride) };
                }
            }
            (Some(s), true) => {
                for &v in src {
                    unsafe { p.write(Complex::new(v.re * s, -v.im * s)) };
                    p = unsafe { p.offset(stride) };
                }
            }
        }
    }
}

#[inline]
unsafe fn scale_complex_slice<F: FftFloat>(slice: &mut [Complex<F>], scale: Option<F>, conj: bool) {
    match (scale, conj) {
        (None, false) => {}
        (Some(s), false) => {
            for v in slice.iter_mut() {
                v.re = v.re * s;
                v.im = v.im * s;
            }
        }
        (None, true) => {
            for v in slice.iter_mut() {
                v.im = -v.im;
            }
        }
        (Some(s), true) => {
            for v in slice.iter_mut() {
                v.re = v.re * s;
                v.im = -v.im * s;
            }
        }
    }
}

#[inline]
unsafe fn write_real_lane<F: FftFloat>(
    src: &[F],
    dst: *mut F,
    stride: isize,
    scale: Option<F>,
) {
    if stride == 1 {
        let out_sl =
            unsafe { std::slice::from_raw_parts_mut(dst as *mut MaybeUninit<F>, src.len()) };
        match scale {
            None => unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            },
            Some(s) => {
                for (d, &v) in out_sl.iter_mut().zip(src.iter()) {
                    d.write(v * s);
                }
            }
        }
    } else {
        let mut p = dst;
        match scale {
            None => {
                for &v in src {
                    unsafe { p.write(v) };
                    p = unsafe { p.offset(stride) };
                }
            }
            Some(s) => {
                for &v in src {
                    unsafe { p.write(v * s) };
                    p = unsafe { p.offset(stride) };
                }
            }
        }
    }
}

#[inline]
unsafe fn pack_complex_tile<In: ToFftValue<F>, F: FftFloat>(
    src: *const In,
    in_axis_stride: isize,
    copy_len: usize,
    n: usize,
    batch: usize,
    padded_n: usize,
    tile: &mut [Complex<F>],
) {
    for r in 0..copy_len {
        let row_ptr = unsafe { src.offset((r as isize) * in_axis_stride) };
        let row = unsafe { std::slice::from_raw_parts(row_ptr, batch) };
        for (b, &v) in row.iter().enumerate() {
            tile[b * padded_n + r] = v.to_complex();
        }
    }
    if copy_len < n {
        for b in 0..batch {
            tile[b * padded_n + copy_len..b * padded_n + n]
                .fill(Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()));
        }
    }
}

#[inline]
unsafe fn pack_real_tile<In: ToFftValue<F>, F: FftFloat>(
    src: *const In,
    in_axis_stride: isize,
    copy_len: usize,
    n: usize,
    batch: usize,
    padded_n: usize,
    tile: &mut [F],
) {
    for r in 0..copy_len {
        let row_ptr = unsafe { src.offset((r as isize) * in_axis_stride) };
        let row = unsafe { std::slice::from_raw_parts(row_ptr, batch) };
        for (b, &v) in row.iter().enumerate() {
            tile[b * padded_n + r] = v.to_real();
        }
    }
    if copy_len < n {
        for b in 0..batch {
            tile[b * padded_n + copy_len..b * padded_n + n].fill(<F as Scalar>::zero());
        }
    }
}

#[inline]
unsafe fn unpack_complex_tile<F: FftFloat>(
    tile: &[Complex<F>],
    padded_len: usize,
    axis_len: usize,
    batch: usize,
    dst: *mut Complex<F>,
    out_axis_stride: isize,
    scale: Option<F>,
    conj: bool,
) {
    match (scale, conj) {
        (None, false) => {
            for r in 0..axis_len {
                let dst_row = unsafe { dst.offset((r as isize) * out_axis_stride) };
                for b in 0..batch {
                    unsafe { dst_row.add(b).write(tile[b * padded_len + r]) };
                }
            }
        }
        (Some(s), false) => {
            for r in 0..axis_len {
                let dst_row = unsafe { dst.offset((r as isize) * out_axis_stride) };
                for b in 0..batch {
                    let v = tile[b * padded_len + r];
                    unsafe { dst_row.add(b).write(Complex::new(v.re * s, v.im * s)) };
                }
            }
        }
        (None, true) => {
            for r in 0..axis_len {
                let dst_row = unsafe { dst.offset((r as isize) * out_axis_stride) };
                for b in 0..batch {
                    let v = tile[b * padded_len + r];
                    unsafe { dst_row.add(b).write(Complex::new(v.re, -v.im)) };
                }
            }
        }
        (Some(s), true) => {
            for r in 0..axis_len {
                let dst_row = unsafe { dst.offset((r as isize) * out_axis_stride) };
                for b in 0..batch {
                    let v = tile[b * padded_len + r];
                    unsafe { dst_row.add(b).write(Complex::new(v.re * s, -v.im * s)) };
                }
            }
        }
    }
}

#[inline]
unsafe fn unpack_real_tile<F: FftFloat>(
    tile: &[F],
    padded_len: usize,
    axis_len: usize,
    batch: usize,
    dst: *mut F,
    out_axis_stride: isize,
    scale: Option<F>,
) {
    match scale {
        None => {
            for r in 0..axis_len {
                let dst_row = unsafe { dst.offset((r as isize) * out_axis_stride) };
                for b in 0..batch {
                    unsafe { dst_row.add(b).write(tile[b * padded_len + r]) };
                }
            }
        }
        Some(s) => {
            for r in 0..axis_len {
                let dst_row = unsafe { dst.offset((r as isize) * out_axis_stride) };
                for b in 0..batch {
                    unsafe { dst_row.add(b).write(tile[b * padded_len + r] * s) };
                }
            }
        }
    }
}

#[inline]
fn opt_scale<F: FftFloat>(scale_f64: f64) -> Option<F> {
    if (scale_f64 - 1.0).abs() == 0.0 {
        None
    } else {
        Some(F::from_f64_val(scale_f64))
    }
}

#[inline]
fn sanitize_c2r_scratch<F: FftFloat>(scratch: &mut [Complex<F>], n: usize) {
    scratch[0].im = <F as Scalar>::zero();
    if n % 2 == 0 {
        let nyq = scratch.len() - 1;
        scratch[nyq].im = <F as Scalar>::zero();
    }
}

fn c2c_into_buffer<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    active_shape: &[usize],
    out_ptr: SendMutPtr<Complex<F>>,
    out_strides: &[isize],
    axis: usize,
    n: usize,
    direction: FftDirection,
    scale_f64: f64,
) {
    let layout = LaneLayout::new(
        active_shape,
        view.strides(),
        out_strides,
        axis,
        view.shape()[axis],
    );
    if layout.num_lanes == 0 {
        return;
    }
    let copy_len = layout.in_axis_len.min(n);
    let plan = F::plan_c2c(n, direction);
    let scratch_len = plan.get_inplace_scratch_len();
    let scale = opt_scale::<F>(scale_f64);
    let in_base = SendPtr(view.as_ptr());

    // Fast path: output axis stride == 1 (contiguous lanes in destination)
    if layout.out_axis_stride == 1 {
        run_lane_chunks(layout.num_lanes, n, |start, end| {
            let mut fft_scratch =
                vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); scratch_len];
            let lanes_in_chunk = end - start;
            let is_dense_out = layout.outer_dims.is_empty()
                || (axis == active_shape.len() - 1
                    && layout.outer_out_strides.last() == Some(&(n as isize))
                    && {
                        let mut expected = n as isize;
                        let mut ok = true;
                        for (&d, &st) in layout
                            .outer_dims
                            .iter()
                            .zip(layout.outer_out_strides.iter())
                            .rev()
                        {
                            if d > 1 && st != expected {
                                ok = false;
                                break;
                            }
                            expected *= d as isize;
                        }
                        ok
                    });

            let mut cursor = LaneCursor::at_lane(&layout, start);
            if is_dense_out {
                let out_chunk_raw = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                let out_chunk_uninit = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_chunk_raw as *mut MaybeUninit<Complex<F>>,
                        lanes_in_chunk * n,
                    )
                };
                for lane_i in 0..lanes_in_chunk {
                    let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                    let dst_lane = &mut out_chunk_uninit[lane_i * n..(lane_i + 1) * n];
                    unsafe {
                        read_complex_lane_uninit(src, layout.in_axis_stride, copy_len, dst_lane)
                    };
                    if lane_i + 1 < lanes_in_chunk {
                        cursor.advance(&layout);
                    }
                }
                let out_chunk =
                    unsafe { std::slice::from_raw_parts_mut(out_chunk_raw, lanes_in_chunk * n) };
                plan.process_with_scratch(out_chunk, &mut fft_scratch);
                if scale.is_some() {
                    unsafe { scale_complex_slice(out_chunk, scale, false) };
                }
            } else {
                for lane_i in 0..lanes_in_chunk {
                    let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                    let dst_raw = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                    let dst_uninit = unsafe {
                        std::slice::from_raw_parts_mut(
                            dst_raw as *mut MaybeUninit<Complex<F>>,
                            n,
                        )
                    };
                    unsafe {
                        read_complex_lane_uninit(src, layout.in_axis_stride, copy_len, dst_uninit)
                    };
                    let dst_lane = unsafe { std::slice::from_raw_parts_mut(dst_raw, n) };
                    plan.process_with_scratch(dst_lane, &mut fft_scratch);
                    if scale.is_some() {
                        unsafe { scale_complex_slice(dst_lane, scale, false) };
                    }
                    if lane_i + 1 < lanes_in_chunk {
                        cursor.advance(&layout);
                    }
                }
            }
        });
        return;
    }

    let tile_outer = can_tile_outer(&layout, true);
    let (tile_cap, padded_n) = tile_dims(n);
    run_lane_chunks(layout.num_lanes, n, |start, end| {
        let mut fft_scratch =
            vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); scratch_len];
        let mut cursor = LaneCursor::at_lane(&layout, start);
        let mut remaining = end - start;
        if let (Some(last_dim), true) = (tile_outer, tile_cap > 1) {
            let mut tile_buf = vec![
                Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero());
                tile_cap * padded_n
            ];
            while remaining > 0 {
                let batch = tile_cap.min(remaining).min(last_dim - cursor.last_coord());
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                if batch == 1 {
                    let lane = &mut tile_buf[..n];
                    unsafe { read_complex_lane(src, layout.in_axis_stride, copy_len, lane) };
                    plan.process_with_scratch(lane, &mut fft_scratch);
                    unsafe { write_complex_lane(lane, dst, layout.out_axis_stride, scale, false) };
                } else {
                    unsafe {
                        pack_complex_tile(
                            src,
                            layout.in_axis_stride,
                            copy_len,
                            n,
                            batch,
                            padded_n,
                            &mut tile_buf,
                        )
                    };
                    for b in 0..batch {
                        plan.process_with_scratch(
                            &mut tile_buf[b * padded_n..b * padded_n + n],
                            &mut fft_scratch,
                        );
                    }
                    unsafe {
                        unpack_complex_tile(
                            &tile_buf,
                            padded_n,
                            n,
                            batch,
                            dst,
                            layout.out_axis_stride,
                            scale,
                            false,
                        )
                    };
                }
                remaining -= batch;
                if remaining > 0 {
                    cursor.advance_by(&layout, batch);
                }
            }
        } else {
            let mut lane_buf = vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); n];
            for lane_i in 0..remaining {
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                unsafe { read_complex_lane(src, layout.in_axis_stride, copy_len, &mut lane_buf) };
                plan.process_with_scratch(&mut lane_buf, &mut fft_scratch);
                unsafe {
                    write_complex_lane(&lane_buf, dst, layout.out_axis_stride, scale, false)
                };
                if lane_i + 1 < remaining {
                    cursor.advance(&layout);
                }
            }
        }
    });
}

fn c2c_inplace_strided<F: FftFloat>(
    buf_ptr: SendMutPtr<Complex<F>>,
    buf_strides: &[isize],
    active_shape: &[usize],
    axis: usize,
    in_len: usize,
    n: usize,
    direction: FftDirection,
    scale_f64: f64,
) where
    Complex<F>: ToFftValue<F>,
{
    let layout = LaneLayout::new(active_shape, buf_strides, buf_strides, axis, in_len);
    if layout.num_lanes == 0 {
        return;
    }
    let copy_len = in_len.min(n);
    let plan = F::plan_c2c(n, direction);
    let scratch_len = plan.get_inplace_scratch_len();
    let scale = opt_scale::<F>(scale_f64);

    if layout.out_axis_stride == 1 {
        run_lane_chunks(layout.num_lanes, n, |start, end| {
            let mut fft_scratch =
                vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); scratch_len];
            let mut cursor = LaneCursor::at_lane(&layout, start);
            let lanes_in_chunk = end - start;
            for lane_i in 0..lanes_in_chunk {
                let lane_ptr = unsafe { buf_ptr.ptr().offset(cursor.out_off) };
                let lane_sl = unsafe { std::slice::from_raw_parts_mut(lane_ptr, n) };
                if copy_len < n {
                    lane_sl[copy_len..]
                        .fill(Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()));
                }
                plan.process_with_scratch(lane_sl, &mut fft_scratch);
                if scale.is_some() {
                    unsafe { scale_complex_slice(lane_sl, scale, false) };
                }
                if lane_i + 1 < lanes_in_chunk {
                    cursor.advance(&layout);
                }
            }
        });
        return;
    }

    let tile_outer = can_tile_outer(&layout, false);
    let (tile_cap, padded_n) = tile_dims(n);
    run_lane_chunks(layout.num_lanes, n, |start, end| {
        let mut fft_scratch =
            vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); scratch_len];
        let mut cursor = LaneCursor::at_lane(&layout, start);
        let mut remaining = end - start;
        if let (Some(last_dim), true) = (tile_outer, tile_cap > 1) {
            let mut tile_buf = vec![
                Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero());
                tile_cap * padded_n
            ];
            while remaining > 0 {
                let batch = tile_cap.min(remaining).min(last_dim - cursor.last_coord());
                let lane_ptr = unsafe { buf_ptr.ptr().offset(cursor.out_off) };
                if batch == 1 {
                    let lane = &mut tile_buf[..n];
                    unsafe {
                        read_complex_lane(
                            lane_ptr as *const Complex<F>,
                            layout.out_axis_stride,
                            copy_len,
                            lane,
                        )
                    };
                    plan.process_with_scratch(lane, &mut fft_scratch);
                    unsafe {
                        write_complex_lane(lane, lane_ptr, layout.out_axis_stride, scale, false)
                    };
                } else {
                    unsafe {
                        pack_complex_tile(
                            lane_ptr as *const Complex<F>,
                            layout.out_axis_stride,
                            copy_len,
                            n,
                            batch,
                            padded_n,
                            &mut tile_buf,
                        )
                    };
                    for b in 0..batch {
                        plan.process_with_scratch(
                            &mut tile_buf[b * padded_n..b * padded_n + n],
                            &mut fft_scratch,
                        );
                    }
                    unsafe {
                        unpack_complex_tile(
                            &tile_buf,
                            padded_n,
                            n,
                            batch,
                            lane_ptr,
                            layout.out_axis_stride,
                            scale,
                            false,
                        )
                    };
                }
                remaining -= batch;
                if remaining > 0 {
                    cursor.advance_by(&layout, batch);
                }
            }
        } else {
            let mut lane_buf = vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); n];
            for lane_i in 0..remaining {
                let lane_ptr = unsafe { buf_ptr.ptr().offset(cursor.out_off) };
                unsafe {
                    read_complex_lane(
                        lane_ptr as *const Complex<F>,
                        layout.out_axis_stride,
                        copy_len,
                        &mut lane_buf,
                    )
                };
                plan.process_with_scratch(&mut lane_buf, &mut fft_scratch);
                unsafe {
                    write_complex_lane(&lane_buf, lane_ptr, layout.out_axis_stride, scale, false)
                };
                if lane_i + 1 < remaining {
                    cursor.advance(&layout);
                }
            }
        }
    });
}

fn r2c_into_buffer<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    active_shape: &[usize],
    out_ptr: SendMutPtr<Complex<F>>,
    out_strides: &[isize],
    axis: usize,
    n: usize,
    scale_f64: f64,
    conjugate_out: bool,
) {
    let out_axis_len = n / 2 + 1;
    let layout = LaneLayout::new(
        active_shape,
        view.strides(),
        out_strides,
        axis,
        view.shape()[axis],
    );
    if layout.num_lanes == 0 {
        return;
    }
    let copy_len = layout.in_axis_len.min(n);
    let plan = F::plan_r2c(n);
    let scratch_len = plan.get_scratch_len();
    let scale = opt_scale::<F>(scale_f64);
    let in_base = SendPtr(view.as_ptr());

    if layout.out_axis_stride == 1 {
        run_lane_chunks(layout.num_lanes, n, |start, end| {
            let mut in_buf = vec![<F as Scalar>::zero(); n];
            let mut fft_scratch = vec![
                Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero());
                scratch_len
            ];
            let mut cursor = LaneCursor::at_lane(&layout, start);
            let lanes_in_chunk = end - start;
            for lane_i in 0..lanes_in_chunk {
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                unsafe { read_real_lane(src, layout.in_axis_stride, copy_len, &mut in_buf) };
                let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst, out_axis_len) };
                let _ = plan.process_with_scratch(&mut in_buf, dst_sl, &mut fft_scratch);
                if scale.is_some() || conjugate_out {
                    unsafe { scale_complex_slice(dst_sl, scale, conjugate_out) };
                }
                if lane_i + 1 < lanes_in_chunk {
                    cursor.advance(&layout);
                }
            }
        });
        return;
    }

    let tile_outer = can_tile_outer(&layout, true);
    let (tile_cap, padded_n) = tile_dims(n);
    let (_, padded_out) = tile_dims(out_axis_len);
    run_lane_chunks(layout.num_lanes, n, |start, end| {
        let mut fft_scratch =
            vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); scratch_len];
        let mut cursor = LaneCursor::at_lane(&layout, start);
        let mut remaining = end - start;
        if let (Some(last_dim), true) = (tile_outer, tile_cap > 1) {
            let mut in_tile = vec![<F as Scalar>::zero(); tile_cap * padded_n];
            let mut out_tile = vec![
                Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero());
                tile_cap * padded_out
            ];
            while remaining > 0 {
                let batch = tile_cap.min(remaining).min(last_dim - cursor.last_coord());
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                if batch == 1 {
                    let in_lane = &mut in_tile[..n];
                    let out_lane = &mut out_tile[..out_axis_len];
                    unsafe { read_real_lane(src, layout.in_axis_stride, copy_len, in_lane) };
                    let _ = plan.process_with_scratch(in_lane, out_lane, &mut fft_scratch);
                    unsafe {
                        write_complex_lane(
                            out_lane,
                            dst,
                            layout.out_axis_stride,
                            scale,
                            conjugate_out,
                        )
                    };
                } else {
                    unsafe {
                        pack_real_tile(
                            src,
                            layout.in_axis_stride,
                            copy_len,
                            n,
                            batch,
                            padded_n,
                            &mut in_tile,
                        )
                    };
                    for b in 0..batch {
                        let _ = plan.process_with_scratch(
                            &mut in_tile[b * padded_n..b * padded_n + n],
                            &mut out_tile[b * padded_out..b * padded_out + out_axis_len],
                            &mut fft_scratch,
                        );
                    }
                    unsafe {
                        unpack_complex_tile(
                            &out_tile,
                            padded_out,
                            out_axis_len,
                            batch,
                            dst,
                            layout.out_axis_stride,
                            scale,
                            conjugate_out,
                        )
                    };
                }
                remaining -= batch;
                if remaining > 0 {
                    cursor.advance_by(&layout, batch);
                }
            }
        } else {
            let mut in_buf = vec![<F as Scalar>::zero(); n];
            let mut out_buf =
                vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); out_axis_len];
            for lane_i in 0..remaining {
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                unsafe { read_real_lane(src, layout.in_axis_stride, copy_len, &mut in_buf) };
                let _ = plan.process_with_scratch(&mut in_buf, &mut out_buf, &mut fft_scratch);
                unsafe {
                    write_complex_lane(
                        &out_buf,
                        dst,
                        layout.out_axis_stride,
                        scale,
                        conjugate_out,
                    )
                };
                if lane_i + 1 < remaining {
                    cursor.advance(&layout);
                }
            }
        }
    });
}

fn c2r_into_buffer<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    active_shape: &[usize],
    out_ptr: SendMutPtr<F>,
    out_strides: &[isize],
    axis: usize,
    n: usize,
    scale_f64: f64,
    conjugate_in: bool,
) {
    let c_len = n / 2 + 1;
    let layout = LaneLayout::new(
        active_shape,
        view.strides(),
        out_strides,
        axis,
        view.shape()[axis],
    );
    if layout.num_lanes == 0 {
        return;
    }
    let copy_len = layout.in_axis_len.min(c_len);
    let plan = F::plan_c2r(n);
    let scratch_len = plan.get_scratch_len();
    let scale = opt_scale::<F>(scale_f64);
    let in_base = SendPtr(view.as_ptr());

    if layout.out_axis_stride == 1 {
        run_lane_chunks(layout.num_lanes, n, |start, end| {
            let mut in_buf =
                vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); c_len];
            let mut fft_scratch = vec![
                Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero());
                scratch_len
            ];
            let mut cursor = LaneCursor::at_lane(&layout, start);
            let lanes_in_chunk = end - start;
            for lane_i in 0..lanes_in_chunk {
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                unsafe { read_complex_lane(src, layout.in_axis_stride, copy_len, &mut in_buf) };
                if conjugate_in {
                    for v in &mut in_buf[..copy_len] {
                        v.im = -v.im;
                    }
                }
                sanitize_c2r_scratch(&mut in_buf, n);
                let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst, n) };
                let _ = plan.process_with_scratch(&mut in_buf, dst_sl, &mut fft_scratch);
                if let Some(s) = scale {
                    for v in dst_sl.iter_mut() {
                        *v = *v * s;
                    }
                }
                if lane_i + 1 < lanes_in_chunk {
                    cursor.advance(&layout);
                }
            }
        });
        return;
    }

    let tile_outer = can_tile_outer(&layout, true);
    let (tile_cap, padded_n) = tile_dims(n);
    let (_, padded_c) = tile_dims(c_len);
    run_lane_chunks(layout.num_lanes, n, |start, end| {
        let mut fft_scratch =
            vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); scratch_len];
        let mut cursor = LaneCursor::at_lane(&layout, start);
        let mut remaining = end - start;
        if let (Some(last_dim), true) = (tile_outer, tile_cap > 1) {
            let mut in_tile = vec![
                Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero());
                tile_cap * padded_c
            ];
            let mut out_tile = vec![<F as Scalar>::zero(); tile_cap * padded_n];
            while remaining > 0 {
                let batch = tile_cap.min(remaining).min(last_dim - cursor.last_coord());
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                if batch == 1 {
                    let in_lane = &mut in_tile[..c_len];
                    let out_lane = &mut out_tile[..n];
                    unsafe { read_complex_lane(src, layout.in_axis_stride, copy_len, in_lane) };
                    if conjugate_in {
                        for v in &mut in_lane[..copy_len] {
                            v.im = -v.im;
                        }
                    }
                    sanitize_c2r_scratch(in_lane, n);
                    let _ = plan.process_with_scratch(in_lane, out_lane, &mut fft_scratch);
                    unsafe { write_real_lane(out_lane, dst, layout.out_axis_stride, scale) };
                } else {
                    unsafe {
                        pack_complex_tile(
                            src,
                            layout.in_axis_stride,
                            copy_len,
                            c_len,
                            batch,
                            padded_c,
                            &mut in_tile,
                        )
                    };
                    for b in 0..batch {
                        let lane_in = &mut in_tile[b * padded_c..b * padded_c + c_len];
                        if conjugate_in {
                            for v in &mut lane_in[..copy_len] {
                                v.im = -v.im;
                            }
                        }
                        sanitize_c2r_scratch(lane_in, n);
                        let lane_out = &mut out_tile[b * padded_n..b * padded_n + n];
                        let _ = plan.process_with_scratch(lane_in, lane_out, &mut fft_scratch);
                    }
                    unsafe {
                        unpack_real_tile(
                            &out_tile,
                            padded_n,
                            n,
                            batch,
                            dst,
                            layout.out_axis_stride,
                            scale,
                        )
                    };
                }
                remaining -= batch;
                if remaining > 0 {
                    cursor.advance_by(&layout, batch);
                }
            }
        } else {
            let mut in_buf =
                vec![Complex::new(<F as Scalar>::zero(), <F as Scalar>::zero()); c_len];
            let mut out_buf = vec![<F as Scalar>::zero(); n];
            for lane_i in 0..remaining {
                let src = unsafe { in_base.ptr().offset(cursor.in_off) };
                let dst = unsafe { out_ptr.ptr().offset(cursor.out_off) };
                unsafe { read_complex_lane(src, layout.in_axis_stride, copy_len, &mut in_buf) };
                if conjugate_in {
                    for v in &mut in_buf[..copy_len] {
                        v.im = -v.im;
                    }
                }
                sanitize_c2r_scratch(&mut in_buf, n);
                let _ = plan.process_with_scratch(&mut in_buf, &mut out_buf, &mut fft_scratch);
                unsafe { write_real_lane(&out_buf, dst, layout.out_axis_stride, scale) };
                if lane_i + 1 < remaining {
                    cursor.advance(&layout);
                }
            }
        }
    });
}

pub(crate) fn c2c_1d<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    n: Option<usize>,
    axis: isize,
    is_forward: bool,
    norm: Norm,
) -> Result<NdArray<Complex<F>>>
where
    Complex<F>: Scalar,
{
    let ax = normalize_axis(axis, view.ndim())?;
    let resolved_n = validate_fft_n(n.unwrap_or(view.shape()[ax]))?;
    let mut out_shape = view.shape().to_vec();
    out_shape[ax] = resolved_n;
    if view.is_empty() {
        return NdArray::zeros(&out_shape);
    }
    let out_len = shape_size(&out_shape)?;
    let mut out_uninit = AlignedBuffer::<Complex<F>>::new_uninit(out_len);
    let out_strides = compute_c_strides(&out_shape);
    let direction = if is_forward {
        FftDirection::Forward
    } else {
        FftDirection::Inverse
    };
    let scale = norm.scale_factor(is_forward, resolved_n as f64);
    c2c_into_buffer(
        view,
        &out_shape,
        SendMutPtr(out_uninit.as_mut_ptr() as *mut Complex<F>),
        &out_strides,
        ax,
        resolved_n,
        direction,
        scale,
    );
    let out_buf = unsafe { out_uninit.assume_init() };
    NdArray::from_aligned_buffer(out_buf, &out_shape, Order::C)
}

pub(crate) fn r2c_1d<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
    is_ihfft: bool,
) -> Result<NdArray<Complex<F>>>
where
    Complex<F>: Scalar,
{
    let ax = normalize_axis(axis, view.ndim())?;
    let resolved_n = validate_fft_n(n.unwrap_or(view.shape()[ax]))?;
    let mut out_shape = view.shape().to_vec();
    out_shape[ax] = resolved_n / 2 + 1;
    if view.is_empty() {
        return NdArray::zeros(&out_shape);
    }
    let out_len = shape_size(&out_shape)?;
    let out_strides = compute_c_strides(&out_shape);
    // `ihfft` uses swapped norm (equivalent to inverse direction scaling) and conjugates output.
    let scale = norm.scale_factor(!is_ihfft, resolved_n as f64);
    let out_buf = if out_strides[ax] == 1 {
        let mut buf = AlignedBuffer::<Complex<F>>::new_zeroed(out_len);
        r2c_into_buffer(
            view,
            &out_shape,
            SendMutPtr(buf.as_mut_ptr()),
            &out_strides,
            ax,
            resolved_n,
            scale,
            is_ihfft,
        );
        buf
    } else {
        let mut uninit = AlignedBuffer::<Complex<F>>::new_uninit(out_len);
        r2c_into_buffer(
            view,
            &out_shape,
            SendMutPtr(uninit.as_mut_ptr() as *mut Complex<F>),
            &out_strides,
            ax,
            resolved_n,
            scale,
            is_ihfft,
        );
        unsafe { uninit.assume_init() }
    };
    NdArray::from_aligned_buffer(out_buf, &out_shape, Order::C)
}

pub(crate) fn c2r_1d<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    n: Option<usize>,
    axis: isize,
    norm: Norm,
    is_hfft: bool,
) -> Result<NdArray<F>> {
    let ax = normalize_axis(axis, view.ndim())?;
    let default_n = view.shape()[ax].saturating_sub(1) * 2;
    let resolved_n = validate_fft_n(n.unwrap_or(default_n))?;
    let mut out_shape = view.shape().to_vec();
    out_shape[ax] = resolved_n;
    if view.is_empty() {
        return NdArray::zeros(&out_shape);
    }
    let out_len = shape_size(&out_shape)?;
    let out_strides = compute_c_strides(&out_shape);
    // `hfft` uses swapped norm (equivalent to forward direction scaling) and conjugates input.
    let scale = norm.scale_factor(is_hfft, resolved_n as f64);
    let out_buf = if out_strides[ax] == 1 {
        let mut buf = AlignedBuffer::<F>::new_zeroed(out_len);
        c2r_into_buffer(
            view,
            &out_shape,
            SendMutPtr(buf.as_mut_ptr()),
            &out_strides,
            ax,
            resolved_n,
            scale,
            is_hfft,
        );
        buf
    } else {
        let mut uninit = AlignedBuffer::<F>::new_uninit(out_len);
        c2r_into_buffer(
            view,
            &out_shape,
            SendMutPtr(uninit.as_mut_ptr() as *mut F),
            &out_strides,
            ax,
            resolved_n,
            scale,
            is_hfft,
        );
        unsafe { uninit.assume_init() }
    };
    NdArray::from_aligned_buffer(out_buf, &out_shape, Order::C)
}

fn cook_nd_args(
    shape: &[usize],
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    is_invreal: bool,
    default_last_two: bool,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let ndim = shape.len();
    let raw_axes_owned: Vec<isize>;
    let raw_axes: &[isize] = match axes {
        Some(ax) => ax,
        None => {
            if default_last_two {
                &[-2, -1]
            } else if let Some(s_vals) = s {
                if s_vals.len() > ndim {
                    return Err(NumPyError::InvalidAxis {
                        axis: -(s_vals.len() as isize),
                        ndim,
                    });
                }
                let start = ndim - s_vals.len();
                raw_axes_owned = (start..ndim).map(|i| i as isize).collect();
                &raw_axes_owned
            } else {
                raw_axes_owned = (0..ndim).map(|i| i as isize).collect();
                &raw_axes_owned
            }
        }
    };

    let mut norm_axes = Vec::with_capacity(raw_axes.len());
    for &ax in raw_axes {
        norm_axes.push(normalize_axis(ax, ndim)?);
    }

    let mut resolved_s = Vec::with_capacity(norm_axes.len());
    match s {
        Some(s_vals) => {
            if s_vals.len() != norm_axes.len() {
                return Err(NumPyError::InvalidValue(
                    "Shape and axes have different lengths.".into(),
                ));
            }
            for (&raw_s, &ax) in s_vals.iter().zip(norm_axes.iter()) {
                if raw_s == -1 {
                    resolved_s.push(validate_fft_n(shape[ax])?);
                } else if raw_s < 1 {
                    return Err(NumPyError::InvalidValue(format!(
                        "Invalid number of FFT data points ({raw_s}) specified."
                    )));
                } else {
                    resolved_s.push(raw_s as usize);
                }
            }
        }
        None => {
            for &ax in &norm_axes {
                resolved_s.push(shape[ax]);
            }
            if is_invreal && !resolved_s.is_empty() {
                let last_idx = resolved_s.len() - 1;
                let last_ax = norm_axes[last_idx];
                resolved_s[last_idx] = shape[last_ax].saturating_sub(1) * 2;
            }
            for &n in &resolved_s {
                validate_fft_n(n)?;
            }
        }
    }

    Ok((resolved_s, norm_axes))
}

#[inline]
fn has_duplicate_axes(axes: &[usize]) -> bool {
    for i in 0..axes.len() {
        for j in (i + 1)..axes.len() {
            if axes[i] == axes[j] {
                return true;
            }
        }
    }
    false
}

pub(crate) fn c2c_nd<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    is_forward: bool,
    norm: Norm,
    default_last_two: bool,
) -> Result<NdArray<Complex<F>>>
where
    Complex<F>: ToFftValue<F> + Scalar,
{
    let (resolved_s, norm_axes) = cook_nd_args(view.shape(), s, axes, false, default_last_two)?;
    if norm_axes.is_empty() {
        let out = if let Some(sl) = view.as_slice() {
            let mut uninit = AlignedBuffer::<Complex<F>>::new_uninit(sl.len());
            for (d, &v) in uninit.as_mut_slice().iter_mut().zip(sl.iter()) {
                d.write(v.to_complex());
            }
            unsafe { uninit.assume_init() }
        } else {
            let mut uninit = AlignedBuffer::<Complex<F>>::new_uninit(view.len());
            for (d, v) in uninit.as_mut_slice().iter_mut().zip(view.iter()) {
                d.write(v.to_complex());
            }
            unsafe { uninit.assume_init() }
        };
        return NdArray::from_aligned_buffer(out, view.shape(), Order::C);
    }

    if norm_axes.len() == 1 {
        return c2c_1d(
            view,
            Some(resolved_s[0]),
            norm_axes[0] as isize,
            is_forward,
            norm,
        );
    }

    if has_duplicate_axes(&norm_axes) {
        let last = norm_axes.len() - 1;
        let mut cur = c2c_1d(
            view,
            Some(resolved_s[last]),
            norm_axes[last] as isize,
            is_forward,
            norm,
        )?;
        for i in (0..last).rev() {
            cur = c2c_1d(
                &cur.view(),
                Some(resolved_s[i]),
                norm_axes[i] as isize,
                is_forward,
                norm,
            )?;
        }
        return Ok(cur);
    }

    let in_shape = view.shape();
    let mut final_shape = in_shape.to_vec();
    for (&ax, &sz) in norm_axes.iter().zip(resolved_s.iter()) {
        final_shape[ax] = sz;
    }
    if view.is_empty() || final_shape.contains(&0) {
        return NdArray::zeros(&final_shape);
    }

    let out_len = shape_size(&final_shape)?;
    let mut out_buf = AlignedBuffer::<Complex<F>>::new_zeroed(out_len);
    let out_strides = compute_c_strides(&final_shape);

    let mut active_shape = in_shape.to_vec();
    let last = norm_axes.len() - 1;
    for i in 0..last {
        let ax = norm_axes[i];
        active_shape[ax] = in_shape[ax].min(resolved_s[i]);
    }
    active_shape[norm_axes[last]] = resolved_s[last];

    let direction = if is_forward {
        FftDirection::Forward
    } else {
        FftDirection::Inverse
    };
    c2c_into_buffer(
        view,
        &active_shape,
        SendMutPtr(out_buf.as_mut_ptr()),
        &out_strides,
        norm_axes[last],
        resolved_s[last],
        direction,
        1.0,
    );

    let total_n: f64 = resolved_s.iter().fold(1.0, |acc, &x| acc * (x as f64));
    let total_scale = norm.scale_factor(is_forward, total_n);

    for i in (0..last).rev() {
        let ax = norm_axes[i];
        let n = resolved_s[i];
        let in_len = in_shape[ax].min(n);
        active_shape[ax] = n;
        let step_scale = if i == 0 { total_scale } else { 1.0 };
        c2c_inplace_strided(
            SendMutPtr(out_buf.as_mut_ptr()),
            &out_strides,
            &active_shape,
            ax,
            in_len,
            n,
            direction,
            step_scale,
        );
    }

    NdArray::from_aligned_buffer(out_buf, &final_shape, Order::C)
}

pub(crate) fn r2c_nd<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
    default_last_two: bool,
) -> Result<NdArray<Complex<F>>>
where
    Complex<F>: ToFftValue<F> + Scalar,
{
    let (resolved_s, norm_axes) = cook_nd_args(view.shape(), s, axes, false, default_last_two)?;
    if norm_axes.is_empty() {
        return Err(NumPyError::InvalidValue(
            "rfftn requires at least one axis to transform.".into(),
        ));
    }
    if norm_axes.len() == 1 {
        return r2c_1d(
            view,
            Some(resolved_s[0]),
            norm_axes[0] as isize,
            norm,
            false,
        );
    }

    if has_duplicate_axes(&norm_axes) {
        let last = norm_axes.len() - 1;
        let mut cur = r2c_1d(
            view,
            Some(resolved_s[last]),
            norm_axes[last] as isize,
            norm,
            false,
        )?;
        for i in (0..last).rev() {
            cur = c2c_1d(
                &cur.view(),
                Some(resolved_s[i]),
                norm_axes[i] as isize,
                true,
                norm,
            )?;
        }
        return Ok(cur);
    }

    let in_shape = view.shape();
    let last = norm_axes.len() - 1;
    let mut final_shape = in_shape.to_vec();
    for i in 0..last {
        final_shape[norm_axes[i]] = resolved_s[i];
    }
    final_shape[norm_axes[last]] = resolved_s[last] / 2 + 1;

    if view.is_empty() || final_shape.contains(&0) {
        return NdArray::zeros(&final_shape);
    }

    let out_len = shape_size(&final_shape)?;
    let mut out_buf = AlignedBuffer::<Complex<F>>::new_zeroed(out_len);
    let out_strides = compute_c_strides(&final_shape);

    let mut active_shape = in_shape.to_vec();
    for i in 0..last {
        let ax = norm_axes[i];
        active_shape[ax] = in_shape[ax].min(resolved_s[i]);
    }
    active_shape[norm_axes[last]] = final_shape[norm_axes[last]];

    r2c_into_buffer(
        view,
        &active_shape,
        SendMutPtr(out_buf.as_mut_ptr()),
        &out_strides,
        norm_axes[last],
        resolved_s[last],
        1.0,
        false,
    );

    let total_n: f64 = resolved_s.iter().fold(1.0, |acc, &x| acc * (x as f64));
    let total_scale = norm.scale_factor(true, total_n);

    for i in (0..last).rev() {
        let ax = norm_axes[i];
        let n = resolved_s[i];
        let in_len = in_shape[ax].min(n);
        active_shape[ax] = n;
        let step_scale = if i == 0 { total_scale } else { 1.0 };
        c2c_inplace_strided(
            SendMutPtr(out_buf.as_mut_ptr()),
            &out_strides,
            &active_shape,
            ax,
            in_len,
            n,
            FftDirection::Forward,
            step_scale,
        );
    }

    NdArray::from_aligned_buffer(out_buf, &final_shape, Order::C)
}

pub(crate) fn c2r_nd<In: ToFftValue<F>, F: FftFloat>(
    view: &ArrayView<'_, In>,
    s: Option<&[isize]>,
    axes: Option<&[isize]>,
    norm: Norm,
    default_last_two: bool,
) -> Result<NdArray<F>>
where
    Complex<F>: ToFftValue<F> + Scalar,
{
    let (resolved_s, norm_axes) = cook_nd_args(view.shape(), s, axes, true, default_last_two)?;
    if norm_axes.is_empty() {
        return Err(NumPyError::InvalidValue(
            "irfftn requires at least one axis to transform.".into(),
        ));
    }
    if norm_axes.len() == 1 {
        return c2r_1d(
            view,
            Some(resolved_s[0]),
            norm_axes[0] as isize,
            norm,
            false,
        );
    }

    if has_duplicate_axes(&norm_axes) {
        let last = norm_axes.len() - 1;
        let mut cur = c2c_1d(
            view,
            Some(resolved_s[0]),
            norm_axes[0] as isize,
            false,
            norm,
        )?;
        for i in 1..last {
            cur = c2c_1d(
                &cur.view(),
                Some(resolved_s[i]),
                norm_axes[i] as isize,
                false,
                norm,
            )?;
        }
        return c2r_1d(
            &cur.view(),
            Some(resolved_s[last]),
            norm_axes[last] as isize,
            norm,
            false,
        );
    }

    let in_shape = view.shape();
    let last = norm_axes.len() - 1;
    let mut final_shape = in_shape.to_vec();
    for (&ax, &sz) in norm_axes.iter().zip(resolved_s.iter()) {
        final_shape[ax] = sz;
    }
    if view.is_empty() || final_shape.contains(&0) {
        return NdArray::zeros(&final_shape);
    }

    let last_ax = norm_axes[last];
    let last_c_needed = in_shape[last_ax].min(resolved_s[last] / 2 + 1);
    let mut complex_shape = in_shape.to_vec();
    for i in 0..last {
        complex_shape[norm_axes[i]] = resolved_s[i];
    }
    complex_shape[last_ax] = last_c_needed;

    let complex_len = shape_size(&complex_shape)?;
    let mut complex_buf = AlignedBuffer::<Complex<F>>::new_zeroed(complex_len);
    let complex_strides = compute_c_strides(&complex_shape);

    let mut active_shape = in_shape.to_vec();
    active_shape[norm_axes[0]] = resolved_s[0];
    for i in 1..last {
        let ax = norm_axes[i];
        active_shape[ax] = in_shape[ax].min(resolved_s[i]);
    }
    active_shape[last_ax] = last_c_needed;

    c2c_into_buffer(
        view,
        &active_shape,
        SendMutPtr(complex_buf.as_mut_ptr()),
        &complex_strides,
        norm_axes[0],
        resolved_s[0],
        FftDirection::Inverse,
        1.0,
    );

    for i in 1..last {
        let ax = norm_axes[i];
        let n = resolved_s[i];
        let in_len = in_shape[ax].min(n);
        active_shape[ax] = n;
        c2c_inplace_strided(
            SendMutPtr(complex_buf.as_mut_ptr()),
            &complex_strides,
            &active_shape,
            ax,
            in_len,
            n,
            FftDirection::Inverse,
            1.0,
        );
    }

    let complex_arr = NdArray::from_aligned_buffer(complex_buf, &complex_shape, Order::C)?;
    let out_len = shape_size(&final_shape)?;
    let out_strides = compute_c_strides(&final_shape);
    let total_n: f64 = resolved_s.iter().fold(1.0, |acc, &x| acc * (x as f64));
    let total_scale = norm.scale_factor(false, total_n);

    let out_buf = if out_strides[last_ax] == 1 {
        let mut buf = AlignedBuffer::<F>::new_zeroed(out_len);
        c2r_into_buffer(
            &complex_arr.view(),
            &final_shape,
            SendMutPtr(buf.as_mut_ptr()),
            &out_strides,
            last_ax,
            resolved_s[last],
            total_scale,
            false,
        );
        buf
    } else {
        let mut uninit = AlignedBuffer::<F>::new_uninit(out_len);
        c2r_into_buffer(
            &complex_arr.view(),
            &final_shape,
            SendMutPtr(uninit.as_mut_ptr() as *mut F),
            &out_strides,
            last_ax,
            resolved_s[last],
            total_scale,
            false,
        );
        unsafe { uninit.assume_init() }
    };

    NdArray::from_aligned_buffer(out_buf, &final_shape, Order::C)
}

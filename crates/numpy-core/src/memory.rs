use crate::util::{
    RAYON_THRESHOLD, coords_offset, copy_slice_to_view_mut, dyn_as_bool, linear_to_coords,
};
use memmap2::{Mmap, MmapMut, MmapOptions};
use numpy_types::{
    AlignedBuffer, ArrayView, ArrayViewMut, Casting, Complex32, Complex64, DType, DynArray,
    NdArray, NumPyError, Order, Result, Scalar, SliceArg, can_cast, check_non_overlapping,
    compute_strides, f16, is_c_contiguous, is_f_contiguous, normalize_axis, normalize_index,
    shape_size,
};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Mutex};

/// Copy values from `src` into `dst`, broadcasting `src` and optional `where_mask` to `dst.shape()`.
pub fn copyto<T: Scalar + Send + Sync>(
    dst: &mut NdArray<T>,
    src: &NdArray<T>,
    where_mask: Option<&NdArray<bool>>,
) -> Result<()> {
    let target_shape = dst.shape().to_vec();
    let src_b = if src.shape() == target_shape.as_slice() {
        src.clone()
    } else {
        src.broadcast_to(&target_shape)?
    };
    let mask_b = match where_mask {
        Some(m) => {
            let mb = if m.shape() == target_shape.as_slice() {
                m.clone()
            } else {
                m.broadcast_to(&target_shape)?
            };
            Some(mb)
        }
        None => None,
    };

    if dst.is_empty() {
        return Ok(());
    }

    let same_src_buf = dst.storage_ptr() as usize == src_b.storage_ptr() as usize;
    let src_safe = if same_src_buf {
        src_b.to_c_contiguous()
    } else {
        src_b
    };
    let mask_safe = match mask_b {
        Some(m) => {
            if dst.storage_ptr() as usize == m.storage_ptr() as usize {
                Some(m.to_c_contiguous())
            } else {
                Some(m)
            }
        }
        None => None,
    };

    let mut dst_view = dst.view_mut()?;
    let src_view = src_safe.view();

    match &mask_safe {
        None => {
            if let (Some(d_sl), Some(s_sl)) = (dst_view.as_slice_mut(), src_view.as_slice()) {
                if d_sl.len() >= RAYON_THRESHOLD {
                    d_sl.par_chunks_mut(4096)
                        .zip(s_sl.par_chunks(4096))
                        .for_each(|(d, s)| d.copy_from_slice(s));
                } else {
                    d_sl.copy_from_slice(s_sl);
                }
                return Ok(());
            }
            if let Some(s_sl) = src_view.as_slice() {
                copy_slice_to_view_mut(s_sl, &mut dst_view);
                return Ok(());
            }
            let tmp = src_safe.to_c_contiguous();
            if let Some(s_sl) = tmp.as_slice() {
                copy_slice_to_view_mut(s_sl, &mut dst_view);
            }
            Ok(())
        }
        Some(mask) => {
            let sv = src_safe.view();
            let mv = mask.view();

            if let (Some(d_sl), Some(s_sl), Some(m_sl)) =
                (dst_view.as_slice_mut(), sv.as_slice(), mv.as_slice())
            {
                if d_sl.len() >= RAYON_THRESHOLD {
                    d_sl.par_chunks_mut(4096)
                        .zip(s_sl.par_chunks(4096))
                        .zip(m_sl.par_chunks(4096))
                        .for_each(|((d, s), m)| {
                            for i in 0..d.len() {
                                if m[i] {
                                    d[i] = s[i];
                                }
                            }
                        });
                } else {
                    for ((d, &s), &m) in d_sl.iter_mut().zip(s_sl.iter()).zip(m_sl.iter()) {
                        if m {
                            *d = s;
                        }
                    }
                }
                return Ok(());
            }

            let d_shape = dst_view.shape().to_vec();
            let d_strides = dst_view.strides().to_vec();
            let d_ptr = dst_view.as_mut_ptr();
            let mut coords = vec![0usize; d_shape.len()];
            let mut cur_d = d_ptr;
            let len = dst_view.len();
            let mut s_it = sv.iter();
            let mut m_it = mv.iter();
            for idx in 0..len {
                let s_val = s_it.next().unwrap_or_else(T::zero);
                let m_val = m_it.next().unwrap_or(false);
                if m_val {
                    unsafe { *cur_d = s_val };
                }
                if idx + 1 < len {
                    for d in (0..d_shape.len()).rev() {
                        coords[d] += 1;
                        if coords[d] < d_shape[d] {
                            cur_d = unsafe { cur_d.offset(d_strides[d]) };
                            break;
                        }
                        let back = (coords[d] - 1) as isize * d_strides[d];
                        cur_d = unsafe { cur_d.offset(-back) };
                        coords[d] = 0;
                    }
                }
            }
            Ok(())
        }
    }
}

/// Dynamic `copyto` with NumPy casting checks and broadcasting.
pub fn dyn_copyto(
    dst: &mut DynArray,
    src: &DynArray,
    casting: Casting,
    where_mask: Option<&DynArray>,
) -> Result<()> {
    if !can_cast(src.dtype(), dst.dtype(), casting) {
        return Err(NumPyError::DTypeError(format!(
            "Cannot cast array data from dtype('{}') to dtype('{}') according to the rule '{casting:?}'",
            src.dtype(),
            dst.dtype()
        )));
    }
    let mask_bool = match where_mask {
        Some(m) => Some(dyn_as_bool(m)?),
        None => None,
    };
    let src_cast = if src.dtype() == dst.dtype() {
        src.clone()
    } else {
        src.astype(dst.dtype(), casting)?
    };
    match (dst, src_cast) {
        (DynArray::Bool(d), DynArray::Bool(s)) => copyto(d, &s, mask_bool),
        (DynArray::I8(d), DynArray::I8(s)) => copyto(d, &s, mask_bool),
        (DynArray::I16(d), DynArray::I16(s)) => copyto(d, &s, mask_bool),
        (DynArray::I32(d), DynArray::I32(s)) => copyto(d, &s, mask_bool),
        (DynArray::I64(d), DynArray::I64(s)) => copyto(d, &s, mask_bool),
        (DynArray::U8(d), DynArray::U8(s)) => copyto(d, &s, mask_bool),
        (DynArray::U16(d), DynArray::U16(s)) => copyto(d, &s, mask_bool),
        (DynArray::U32(d), DynArray::U32(s)) => copyto(d, &s, mask_bool),
        (DynArray::U64(d), DynArray::U64(s)) => copyto(d, &s, mask_bool),
        (DynArray::F16(d), DynArray::F16(s)) => copyto(d, &s, mask_bool),
        (DynArray::F32(d), DynArray::F32(s)) => copyto(d, &s, mask_bool),
        (DynArray::F64(d), DynArray::F64(s)) => copyto(d, &s, mask_bool),
        (DynArray::C64(d), DynArray::C64(s)) => copyto(d, &s, mask_bool),
        (DynArray::C128(d), DynArray::C128(s)) => copyto(d, &s, mask_bool),
        _ => unreachable!("src_cast was cast to dst.dtype()"),
    }
}

/// Physical byte layout descriptor for memory overlap detection.
#[derive(Debug, Clone)]
pub struct MemorySpan {
    pub data_ptr: usize,
    pub shape: Vec<usize>,
    pub byte_strides: Vec<isize>,
    pub itemsize: usize,
    pub len: usize,
}

impl MemorySpan {
    /// Compute inclusive-start, exclusive-end byte bounding box `[min_byte, max_byte_end)`.
    pub fn byte_bounds(&self) -> Option<(i128, i128)> {
        if self.len == 0 || self.itemsize == 0 {
            return None;
        }
        let mut min_off = self.data_ptr as i128;
        let mut max_off = self.data_ptr as i128;
        for (&dim, &st) in self.shape.iter().zip(self.byte_strides.iter()) {
            if dim > 1 {
                let span = (dim as i128 - 1) * (st as i128);
                if span < 0 {
                    min_off += span;
                } else {
                    max_off += span;
                }
            }
        }
        Some((min_off, max_off + self.itemsize as i128))
    }

    fn normalized_positive(&self) -> (i128, Vec<(usize, usize)>) {
        let mut min_off = self.data_ptr as i128;
        let mut dims: Vec<(usize, usize)> = Vec::with_capacity(self.shape.len());
        for (&dim, &st) in self.shape.iter().zip(self.byte_strides.iter()) {
            if dim > 1 && st != 0 {
                if st < 0 {
                    min_off += (dim as i128 - 1) * (st as i128);
                }
                dims.push((dim, st.unsigned_abs()));
            }
        }
        dims.sort_unstable_by_key(|&(dim, st)| (st, dim));
        // Coalesce contiguous runs
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(dims.len());
        for (dim, st) in dims {
            if let Some(last) = merged.last_mut() {
                if last.0.checked_mul(last.1) == Some(st) {
                    if let Some(prod) = last.0.checked_mul(dim) {
                        last.0 = prod;
                        continue;
                    }
                }
            }
            merged.push((dim, st));
        }
        (min_off, merged)
    }
}

/// Trait implemented by any array or view type that can report its memory footprint.
pub trait MemoryInspect {
    fn memory_span(&self) -> MemorySpan;
}

impl<T> MemoryInspect for NdArray<T> {
    fn memory_span(&self) -> MemorySpan {
        let itemsize = mem::size_of::<T>();
        let data_ptr = if self.is_empty() {
            0
        } else {
            unsafe { self.storage_ptr().add(self.offset()) as usize }
        };
        let byte_strides = self
            .strides()
            .iter()
            .map(|&s| s.saturating_mul(itemsize as isize))
            .collect();
        MemorySpan {
            data_ptr,
            shape: self.shape().to_vec(),
            byte_strides,
            itemsize,
            len: self.len(),
        }
    }
}

impl<T> MemoryInspect for ArrayView<'_, T> {
    fn memory_span(&self) -> MemorySpan {
        let itemsize = mem::size_of::<T>();
        let data_ptr = if self.is_empty() {
            0
        } else {
            self.as_ptr() as usize
        };
        let byte_strides = self
            .strides()
            .iter()
            .map(|&s| s.saturating_mul(itemsize as isize))
            .collect();
        MemorySpan {
            data_ptr,
            shape: self.shape().to_vec(),
            byte_strides,
            itemsize,
            len: self.len(),
        }
    }
}

impl<T> MemoryInspect for ArrayViewMut<'_, T> {
    fn memory_span(&self) -> MemorySpan {
        self.as_view().memory_span()
    }
}

impl MemoryInspect for DynArray {
    fn memory_span(&self) -> MemorySpan {
        numpy_types::dispatch_dyn!(self, a => a.memory_span())
    }
}

impl<T> MemoryInspect for Memmap<T> {
    fn memory_span(&self) -> MemorySpan {
        let itemsize = mem::size_of::<T>();
        let data_ptr = if self.len == 0 {
            0
        } else {
            unsafe { (self.storage.as_ptr() as *const u8).add(self.byte_offset) as usize }
        };
        MemorySpan {
            data_ptr,
            shape: self.shape.clone(),
            byte_strides: self
                .strides
                .iter()
                .map(|&s| s.saturating_mul(itemsize as isize))
                .collect(),
            itemsize,
            len: self.len,
        }
    }
}

impl MemoryInspect for DynMemmap {
    fn memory_span(&self) -> MemorySpan {
        match self {
            Self::Bool(m) => m.memory_span(),
            Self::I8(m) => m.memory_span(),
            Self::I16(m) => m.memory_span(),
            Self::I32(m) => m.memory_span(),
            Self::I64(m) => m.memory_span(),
            Self::U8(m) => m.memory_span(),
            Self::U16(m) => m.memory_span(),
            Self::U32(m) => m.memory_span(),
            Self::U64(m) => m.memory_span(),
            Self::F16(m) => m.memory_span(),
            Self::F32(m) => m.memory_span(),
            Self::F64(m) => m.memory_span(),
            Self::C64(m) => m.memory_span(),
            Self::C128(m) => m.memory_span(),
        }
    }
}

/// Fast O(ndim) check if two arrays might share memory (bounding-box byte range overlap).
pub fn may_share_memory<A: MemoryInspect + ?Sized, B: MemoryInspect + ?Sized>(a: &A, b: &B) -> bool {
    let sa = a.memory_span();
    let sb = b.memory_span();
    let (Some((a_min, a_max)), Some((b_min, b_max))) = (sa.byte_bounds(), sb.byte_bounds()) else {
        return false;
    };
    a_max > b_min && b_max > a_min
}

/// Exact check whether two arrays share any overlapping element bytes.
pub fn shares_memory<A: MemoryInspect + ?Sized, B: MemoryInspect + ?Sized>(a: &A, b: &B) -> bool {
    let sa = a.memory_span();
    let sb = b.memory_span();
    let (Some((a_min, a_max)), Some((b_min, b_max))) = (sa.byte_bounds(), sb.byte_bounds()) else {
        return false;
    };
    if !(a_max > b_min && b_max > a_min) {
        return false;
    }

    let (a0, a_dims) = sa.normalized_positive();
    let (b0, b_dims) = sb.normalized_positive();
    let ia = sa.itemsize as i128;
    let ib = sb.itemsize as i128;

    // Trivial 0-active-dim (single element / broadcast-only) cases
    if a_dims.is_empty() && b_dims.is_empty() {
        return a0 + ia > b0 && b0 + ib > a0;
    }

    // Build combined Diophantine variables: a0 + sum(st_i * x_i) - b0 - sum(st_j * y_j) in (-ia, ib)
    // i.e. sum(c_k * x_k) in [b0 - a0 - ia + 1, b0 - a0 + ib - 1]
    let rhs_lo = b0 - a0 - ia + 1;
    let rhs_hi = b0 - a0 + ib - 1;

    // Fast GCD check across all non-zero strides
    let mut g = 0i128;
    for &(_, st) in &a_dims {
        g = gcd_i128(g, st as i128);
    }
    for &(_, st) in &b_dims {
        g = gcd_i128(g, st as i128);
    }
    if g > 0 {
        let first_mult = ceil_div_i128(rhs_lo, g) * g;
        if first_mult > rhs_hi {
            return false;
        }
    }

    // Exact 1-dim vs 1-dim solver
    if a_dims.len() <= 1 && b_dims.len() <= 1 {
        let (na, sta) = a_dims.first().copied().unwrap_or((1, 0));
        let (nb, stb) = b_dims.first().copied().unwrap_or((1, 0));
        return solve_1d_overlap(na, sta as i128, nb, stb as i128, rhs_lo, rhs_hi);
    }

    // General branch-and-bound Diophantine overlap solver (with suffix min/max pruning)
    let mut terms: Vec<(i128, usize)> = Vec::with_capacity(a_dims.len() + b_dims.len());
    for &(dim, st) in &a_dims {
        terms.push((st as i128, dim - 1));
    }
    for &(dim, st) in &b_dims {
        terms.push((-(st as i128), dim - 1));
    }
    // Sort by absolute stride descending for tightest pruning first
    terms.sort_unstable_by(|x, y| y.0.abs().cmp(&x.0.abs()));

    let m = terms.len();
    let mut suf_min = vec![0i128; m + 1];
    let mut suf_max = vec![0i128; m + 1];
    for i in (0..m).rev() {
        let (c, max_x) = terms[i];
        let contrib = c * (max_x as i128);
        suf_min[i] = suf_min[i + 1] + contrib.min(0);
        suf_max[i] = suf_max[i + 1] + contrib.max(0);
    }
    let mut suf_gcd = vec![0i128; m + 1];
    for i in (0..m).rev() {
        suf_gcd[i] = gcd_i128(suf_gcd[i + 1], terms[i].0.abs());
    }

    let mut steps = 0usize;
    match search_overlap(&terms, 0, 0, rhs_lo, rhs_hi, &suf_min, &suf_max, &suf_gcd, &mut steps) {
        Some(found) => found,
        None => true, // conservative fallback if budget exhausted (never false negative!)
    }
}

#[allow(clippy::too_many_arguments)]
fn search_overlap(
    terms: &[(i128, usize)],
    idx: usize,
    acc: i128,
    rhs_lo: i128,
    rhs_hi: i128,
    suf_min: &[i128],
    suf_max: &[i128],
    suf_gcd: &[i128],
    steps: &mut usize,
) -> Option<bool> {
    if acc + suf_max[idx] < rhs_lo || acc + suf_min[idx] > rhs_hi {
        return Some(false);
    }
    let g = suf_gcd[idx];
    if g > 0 {
        let lo = rhs_lo - acc;
        let hi = rhs_hi - acc;
        if ceil_div_i128(lo, g) * g > hi {
            return Some(false);
        }
    }
    if idx == terms.len() {
        return Some(acc >= rhs_lo && acc <= rhs_hi);
    }
    *steps += 1;
    if *steps > 200_000 {
        return None;
    }

    let (c, max_x) = terms[idx];
    // Compute tightened [x_lo, x_hi] bounds from remaining suffix range!
    let rem_min = suf_min[idx + 1];
    let rem_max = suf_max[idx + 1];
    let need_lo = rhs_lo - acc - rem_max;
    let need_hi = rhs_hi - acc - rem_min;
    let (x_lo, x_hi) = if c > 0 {
        let lo = ceil_div_i128(need_lo, c).max(0);
        let hi = floor_div_i128(need_hi, c).min(max_x as i128);
        (lo, hi)
    } else {
        let abs_c = -c;
        let lo = ceil_div_i128(-need_hi, abs_c).max(0);
        let hi = floor_div_i128(-need_lo, abs_c).min(max_x as i128);
        (lo, hi)
    };
    if x_lo > x_hi {
        return Some(false);
    }
    for x in x_lo..=x_hi {
        match search_overlap(
            terms,
            idx + 1,
            acc + c * x,
            rhs_lo,
            rhs_hi,
            suf_min,
            suf_max,
            suf_gcd,
            steps,
        ) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => return None,
        }
    }
    Some(false)
}

fn solve_1d_overlap(
    na: usize,
    sta: i128,
    nb: usize,
    stb: i128,
    rhs_lo: i128,
    rhs_hi: i128,
) -> bool {
    if sta == 0 && stb == 0 {
        return 0 >= rhs_lo && 0 <= rhs_hi;
    }
    if sta == 0 {
        // -stb * j in [rhs_lo, rhs_hi]
        let lo = ceil_div_i128(-rhs_hi, stb).max(0);
        let hi = floor_div_i128(-rhs_lo, stb).min((nb - 1) as i128);
        return lo <= hi;
    }
    if stb == 0 {
        let lo = ceil_div_i128(rhs_lo, sta).max(0);
        let hi = floor_div_i128(rhs_hi, sta).min((na - 1) as i128);
        return lo <= hi;
    }
    let (g, x0, y0) = ext_gcd(sta, -stb);
    let m_lo = ceil_div_i128(rhs_lo, g);
    let m_hi = floor_div_i128(rhs_hi, g);
    if m_lo > m_hi {
        return false;
    }
    // Check if there is k in Z, m in [m_lo, m_hi] with:
    // i = m*x0 + k*(stb/g) in [0, na-1] and j = m*y0 + k*(sta/g) in [0, nb-1]
    // Since step count of m_hi - m_lo is at most ia + ib (<= 32 bytes usually), iterate m in [m_lo, m_hi]!
    let step_i = stb / g;
    let step_j = sta / g;
    let max_i = (na - 1) as i128;
    let max_j = (nb - 1) as i128;
    for m in m_lo..=m_hi {
        let ix = m * x0;
        let iy = m * y0;
        let k_lo_i = ceil_div_i128(-ix, step_i);
        let k_hi_i = floor_div_i128(max_i - ix, step_i);
        let k_lo_j = ceil_div_i128(-iy, step_j);
        let k_hi_j = floor_div_i128(max_j - iy, step_j);
        if k_lo_i.max(k_lo_j) <= k_hi_i.min(k_hi_j) {
            return true;
        }
    }
    false
}

#[inline]
fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[inline]
fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
        return (a.abs(), if a >= 0 { 1 } else { -1 }, 0);
    }
    let (g, x1, y1) = ext_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

#[inline]
fn floor_div_i128(a: i128, b: i128) -> i128 {
    debug_assert!(b > 0);
    a.div_euclid(b)
}

#[inline]
fn ceil_div_i128(a: i128, b: i128) -> i128 {
    debug_assert!(b > 0);
    -((-a).div_euclid(b))
}

// ---------------------------------------------------------------------------
// Memory-mapped arrays via memmap2
// ---------------------------------------------------------------------------

/// File open mode for `Memmap` matching NumPy's `'r'`, `'r+'`, `'w+'`, `'c'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemmapMode {
    /// Open existing file for reading only (`'r'`).
    ReadOnly,
    /// Open existing file for reading and writing (`'r+'`).
    ReadWrite,
    /// Create or overwrite existing file for reading and writing (`'w+'`).
    WriteNew,
    /// Copy-on-write: writes modify private memory pages without touching disk (`'c'`).
    CopyOnWrite,
}

impl MemmapMode {
    pub fn from_str_code(s: &str) -> Result<Self> {
        match s {
            "r" | "readonly" => Ok(Self::ReadOnly),
            "r+" | "readwrite" => Ok(Self::ReadWrite),
            "w+" | "write" => Ok(Self::WriteNew),
            "c" | "copyonwrite" => Ok(Self::CopyOnWrite),
            _ => Err(NumPyError::InvalidValue(format!(
                "invalid memmap mode '{s}', expected 'r', 'r+', 'w+', or 'c'"
            ))),
        }
    }
}

enum MmapBacking {
    ReadOnly(Mmap),
    Writable(Mutex<MmapMut>),
}

struct MmapStorage {
    backing: MmapBacking,
    ptr: *const u8,
    byte_len: usize,
    mode: MemmapMode,
    path: PathBuf,
}

unsafe impl Send for MmapStorage {}
unsafe impl Sync for MmapStorage {}

impl MmapStorage {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    fn is_writable(&self) -> bool {
        !matches!(self.mode, MemmapMode::ReadOnly)
    }

    fn with_readable_bytes<R>(&self, f: impl FnOnce(&[u8]) -> Result<R>) -> Result<R> {
        match &self.backing {
            MmapBacking::ReadOnly(m) => f(&m[..]),
            MmapBacking::Writable(m) => {
                let guard = m
                    .lock()
                    .map_err(|_| NumPyError::IoError("memmap mutex poisoned".into()))?;
                f(&guard[..])
            }
        }
    }

    fn with_writable_bytes<R>(&self, f: impl FnOnce(&mut [u8]) -> Result<R>) -> Result<R> {
        match &self.backing {
            MmapBacking::ReadOnly(_) => Err(NumPyError::InvalidValue(
                "assignment destination is read-only".into(),
            )),
            MmapBacking::Writable(m) => {
                let mut guard = m
                    .lock()
                    .map_err(|_| NumPyError::IoError("memmap mutex poisoned".into()))?;
                f(&mut guard)
            }
        }
    }

    fn flush(&self) -> Result<()> {
        match &self.backing {
            MmapBacking::ReadOnly(_) => Ok(()),
            MmapBacking::Writable(m) => {
                let guard = m
                    .lock()
                    .map_err(|_| NumPyError::IoError("memmap mutex poisoned".into()))?;
                guard.flush().map_err(NumPyError::from)
            }
        }
    }

    fn flush_async(&self) -> Result<()> {
        match &self.backing {
            MmapBacking::ReadOnly(_) => Ok(()),
            MmapBacking::Writable(m) => {
                let guard = m
                    .lock()
                    .map_err(|_| NumPyError::IoError("memmap mutex poisoned".into()))?;
                guard.flush_async().map_err(NumPyError::from)
            }
        }
    }
}

/// Typed memory-mapped N-dimensional array backed by `memmap2`.
///
/// Supports zero-copy views (`slice`, `transpose`, `swapaxes`, `flip`, `squeeze`,
/// `expand_dims`), `flush()`, safe aligned slice access (`as_slice` / `as_slice_mut`),
/// unaligned-safe element indexing, and zero-copy or aligned `to_ndarray()` materialization.
#[derive(Clone)]
pub struct Memmap<T> {
    storage: Arc<MmapStorage>,
    byte_offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
    len: usize,
    order: Order,
    _marker: PhantomData<T>,
}

impl<T: Scalar> Memmap<T> {
    /// Open or create a memory-mapped array backed by `path`.
    ///
    /// If `shape` is `None`, infers a 1-D shape from `(file_len - byte_offset) / size_of::<T>()`
    /// (requires `mode != WriteNew` and remainder == 0).
    pub fn open<P: AsRef<Path>>(
        path: P,
        mode: MemmapMode,
        byte_offset: usize,
        shape: Option<&[usize]>,
        order: Order,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let itemsize = mem::size_of::<T>();
        if itemsize == 0 {
            return Err(NumPyError::DTypeError("cannot memmap zero-sized type".into()));
        }

        let file: File = match mode {
            MemmapMode::ReadOnly | MemmapMode::CopyOnWrite => OpenOptions::new().read(true).open(path_ref)?,
            MemmapMode::ReadWrite => OpenOptions::new().read(true).write(true).open(path_ref)?,
            MemmapMode::WriteNew => OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path_ref)?,
        };

        let meta_len = file.metadata()?.len() as usize;
        let resolved_shape: Vec<usize> = match shape {
            Some(s) => s.to_vec(),
            None => {
                if mode == MemmapMode::WriteNew {
                    return Err(NumPyError::InvalidValue(
                        "shape must be given when mode is WriteNew ('w+')".into(),
                    ));
                }
                if byte_offset > meta_len {
                    return Err(NumPyError::InvalidValue(format!(
                        "byte_offset {byte_offset} exceeds file length {meta_len}"
                    )));
                }
                let avail = meta_len - byte_offset;
                if avail % itemsize != 0 {
                    return Err(NumPyError::InvalidValue(
                        "Size of available data is not a multiple of the data-type size.".into(),
                    ));
                }
                vec![avail / itemsize]
            }
        };

        let len = shape_size(&resolved_shape)?;
        let data_bytes = len
            .checked_mul(itemsize)
            .ok_or_else(|| NumPyError::ShapeMismatch("memmap byte length overflow".into()))?;
        let needed_bytes = byte_offset
            .checked_add(data_bytes)
            .ok_or_else(|| NumPyError::ShapeMismatch("memmap byte offset overflow".into()))?;

        if matches!(mode, MemmapMode::WriteNew | MemmapMode::ReadWrite) && meta_len < needed_bytes {
            file.set_len(needed_bytes as u64)?;
        } else if matches!(mode, MemmapMode::ReadOnly | MemmapMode::CopyOnWrite)
            && needed_bytes > meta_len
        {
            return Err(NumPyError::InvalidValue(format!(
                "memmap requires {needed_bytes} bytes, but file only has {meta_len} bytes"
            )));
        }

        let mut opts = MmapOptions::new();
        opts.offset(0).len(needed_bytes);
        let (backing, ptr, mapped_len) = match mode {
            MemmapMode::ReadOnly => {
                let mm = unsafe { opts.map(&file)? };
                let p = mm.as_ptr();
                let l = mm.len();
                (MmapBacking::ReadOnly(mm), p, l)
            }
            MemmapMode::ReadWrite => {
                let mm = unsafe { opts.map_mut(&file)? };
                let p = mm.as_ptr();
                let l = mm.len();
                (MmapBacking::Writable(Mutex::new(mm)), p, l)
            }
            MemmapMode::WriteNew => {
                let mm = unsafe { opts.map_mut(&file)? };
                let p = mm.as_ptr();
                let l = mm.len();
                (MmapBacking::Writable(Mutex::new(mm)), p, l)
            }
            MemmapMode::CopyOnWrite => {
                let mm = unsafe { opts.map_copy(&file)? };
                let p = mm.as_ptr();
                let l = mm.len();
                (MmapBacking::Writable(Mutex::new(mm)), p, l)
            }
        };

        let resolved_order = match order {
            Order::F => Order::F,
            Order::C | Order::A | Order::K => Order::C,
        };
        let strides = compute_strides(&resolved_shape, resolved_order);
        let storage = Arc::new(MmapStorage {
            backing,
            ptr,
            byte_len: mapped_len,
            mode,
            path: path_ref.to_path_buf(),
        });

        Ok(Self {
            storage,
            byte_offset,
            shape: resolved_shape,
            strides,
            len,
            order: resolved_order,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn mode(&self) -> MemmapMode {
        self.storage.mode
    }

    #[inline]
    pub fn is_writable(&self) -> bool {
        self.storage.is_writable()
    }

    #[inline]
    pub fn path(&self) -> &Path {
        &self.storage.path
    }

    #[inline]
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    #[inline]
    pub fn order(&self) -> Order {
        self.order
    }

    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.shape, &self.strides)
    }

    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        is_f_contiguous(&self.shape, &self.strides)
    }

    #[inline]
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    #[inline]
    fn data_ptr(&self) -> *const u8 {
        unsafe { self.storage.as_ptr().add(self.byte_offset) }
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        (self.data_ptr() as usize) % mem::align_of::<T>() == 0
    }

    /// Return a zero-copy `&[T]` for read-only mappings when C-contiguous and naturally aligned for `T`.
    /// Returns `None` for writable mappings (use `as_slice_mut`, `get`, or `to_ndarray` for synchronized access).
    pub fn as_slice(&self) -> Option<&[T]> {
        if !matches!(self.storage.backing, MmapBacking::ReadOnly(_)) {
            return None;
        }
        if self.len == 0 {
            return Some(&[]);
        }
        if self.is_c_contiguous() && self.is_aligned() {
            let byte_end = self
                .len
                .checked_mul(mem::size_of::<T>())
                .and_then(|bytes| self.byte_offset.checked_add(bytes))?;
            if byte_end > self.storage.byte_len {
                return None;
            }
            let ptr = self.data_ptr() as *const T;
            Some(unsafe { std::slice::from_raw_parts(ptr, self.len) })
        } else {
            None
        }
    }

    /// Return a zero-copy `&mut [T]` if writable, C-contiguous, naturally aligned for `T`,
    /// and uniquely owned (`Arc::strong_count == 1`).
    pub fn as_slice_mut(&mut self) -> Result<Option<&mut [T]>> {
        if !self.is_writable() {
            return Err(NumPyError::InvalidValue(
                "assignment destination is read-only".into(),
            ));
        }
        if self.len == 0 {
            return Ok(Some(&mut []));
        }
        if !self.is_c_contiguous() || !self.is_aligned() {
            return Ok(None);
        }
        let itemsize = mem::size_of::<T>();
        let byte_end = self
            .len
            .checked_mul(itemsize)
            .and_then(|bytes| self.byte_offset.checked_add(bytes))
            .ok_or_else(|| {
                NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
            })?;
        if byte_end > self.storage.byte_len {
            return Err(NumPyError::IndexOutOfBounds(
                "memmap byte offset out of bounds".into(),
            ));
        }
        let Some(storage_mut) = Arc::get_mut(&mut self.storage) else {
            return Ok(None);
        };
        let ptr = match &mut storage_mut.backing {
            MmapBacking::ReadOnly(_) => {
                return Err(NumPyError::InvalidValue(
                    "assignment destination is read-only".into(),
                ));
            }
            MmapBacking::Writable(m) => {
                let mm = m
                    .get_mut()
                    .map_err(|_| NumPyError::IoError("memmap mutex poisoned".into()))?;
                unsafe { mm.as_mut_ptr().add(self.byte_offset) as *mut T }
            }
        };
        Ok(Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }))
    }

    fn checked_byte_pos(&self, idx: &[isize]) -> Result<usize> {
        if idx.len() != self.ndim() {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "expected {} indices, got {}",
                self.ndim(),
                idx.len()
            )));
        }
        let mut elem_off = 0isize;
        for (ax, &i) in idx.iter().enumerate() {
            let norm = normalize_index(i, self.shape[ax])?;
            let step = (norm as isize)
                .checked_mul(self.strides[ax])
                .ok_or_else(|| {
                    NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
                })?;
            elem_off = elem_off.checked_add(step).ok_or_else(|| {
                NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
            })?;
        }
        let itemsize = mem::size_of::<T>();
        let byte_delta = elem_off
            .checked_mul(itemsize as isize)
            .ok_or_else(|| {
                NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
            })?;
        let byte_pos = (self.byte_offset as isize)
            .checked_add(byte_delta)
            .ok_or_else(|| {
                NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
            })?;
        if byte_pos < 0
            || (byte_pos as usize)
                .checked_add(itemsize)
                .is_none_or(|end| end > self.storage.byte_len)
        {
            return Err(NumPyError::IndexOutOfBounds(
                "memmap byte offset out of bounds".into(),
            ));
        }
        Ok(byte_pos as usize)
    }

    /// Read element at multi-index `idx` (supports negative indices and unaligned byte offsets).
    pub fn get(&self, idx: &[isize]) -> Result<T> {
        let byte_pos = self.checked_byte_pos(idx)?;
        let itemsize = mem::size_of::<T>();
        self.storage.with_readable_bytes(|buf| {
            if byte_pos + itemsize > buf.len() {
                return Err(NumPyError::IndexOutOfBounds(
                    "memmap byte offset out of bounds".into(),
                ));
            }
            let p = unsafe { buf.as_ptr().add(byte_pos) } as *const T;
            Ok(unsafe { ptr::read_unaligned(p) })
        })
    }

    /// Write element at multi-index `idx`.
    pub fn set(&mut self, idx: &[isize], val: T) -> Result<()> {
        if !self.is_writable() {
            return Err(NumPyError::InvalidValue(
                "assignment destination is read-only".into(),
            ));
        }
        let byte_pos = self.checked_byte_pos(idx)?;
        let itemsize = mem::size_of::<T>();
        self.storage.with_writable_bytes(|buf| {
            if byte_pos + itemsize > buf.len() {
                return Err(NumPyError::IndexOutOfBounds(
                    "memmap byte offset out of bounds".into(),
                ));
            }
            let p = unsafe { buf.as_mut_ptr().add(byte_pos) } as *mut T;
            unsafe { ptr::write_unaligned(p, val) };
            Ok(())
        })
    }

    /// Flush modifications in memory back to disk.
    pub fn flush(&self) -> Result<()> {
        self.storage.flush()
    }

    /// Asynchronously flush modifications in memory back to disk.
    pub fn flush_async(&self) -> Result<()> {
        self.storage.flush_async()
    }

    /// O(1) zero-copy slice view into the mapped file.
    pub fn slice(&self, args: &[SliceArg]) -> Result<Self> {
        let has_ellipsis = args.iter().filter(|a| matches!(a, SliceArg::Ellipsis)).count();
        if has_ellipsis > 1 {
            return Err(NumPyError::IndexOutOfBounds(
                "an index can only have a single ellipsis ('...')".into(),
            ));
        }
        let consumed = args
            .iter()
            .filter(|a| !matches!(a, SliceArg::NewAxis | SliceArg::Ellipsis))
            .count();
        if consumed > self.ndim() {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "too many indices for array: array is {}-dimensional, but {consumed} were indexed",
                self.ndim()
            )));
        }
        let ellipsis_expand = self.ndim() - consumed;
        let mut expanded = Vec::with_capacity(self.ndim() + args.len());
        for &arg in args {
            match arg {
                SliceArg::Ellipsis => expanded.extend(std::iter::repeat_n(
                    SliceArg::Slice(numpy_types::Slice::full()),
                    ellipsis_expand,
                )),
                other => expanded.push(other),
            }
        }
        let remaining =
            self.ndim() - expanded.iter().filter(|a| !matches!(a, SliceArg::NewAxis)).count();
        expanded.extend(std::iter::repeat_n(
            SliceArg::Slice(numpy_types::Slice::full()),
            remaining,
        ));

        let itemsize = mem::size_of::<T>() as isize;
        let mut new_byte_off = self.byte_offset as isize;
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        let mut src_axis = 0usize;

        for arg in expanded {
            match arg {
                SliceArg::NewAxis => {
                    new_shape.push(1);
                    new_strides.push(0);
                }
                SliceArg::Index(idx) => {
                    let norm = normalize_index(idx, self.shape[src_axis])?;
                    new_byte_off += (norm as isize) * self.strides[src_axis] * itemsize;
                    src_axis += 1;
                }
                SliceArg::Slice(sl) => {
                    let (first, count, step) = sl.resolve(self.shape[src_axis])?;
                    if count > 0 {
                        new_byte_off += (first as isize) * self.strides[src_axis] * itemsize;
                    }
                    let new_stride = if count <= 1 {
                        0
                    } else {
                        self.strides[src_axis].checked_mul(step).ok_or_else(|| {
                            NumPyError::ShapeMismatch("slice stride overflow".into())
                        })?
                    };
                    new_shape.push(count);
                    new_strides.push(new_stride);
                    src_axis += 1;
                }
                SliceArg::Ellipsis => unreachable!(),
            }
        }
        let new_len = shape_size(&new_shape)?;
        let safe_off = if new_len == 0 {
            self.byte_offset
        } else {
            new_byte_off as usize
        };
        Ok(Self {
            storage: Arc::clone(&self.storage),
            byte_offset: safe_off,
            shape: new_shape,
            strides: new_strides,
            len: new_len,
            order: self.order,
            _marker: PhantomData,
        })
    }

    /// O(1) zero-copy transpose.
    pub fn transpose(&self, axes: Option<&[isize]>) -> Result<Self> {
        let ndim = self.ndim();
        let perm: Vec<usize> = match axes {
            None => (0..ndim).rev().collect(),
            Some(ax) => {
                if ax.len() != ndim {
                    return Err(NumPyError::InvalidValue(format!(
                        "axes don't match array: expected {ndim} axes, got {}",
                        ax.len()
                    )));
                }
                numpy_types::normalize_axes(ax, ndim)?
            }
        };
        Ok(Self {
            storage: Arc::clone(&self.storage),
            byte_offset: self.byte_offset,
            shape: perm.iter().map(|&i| self.shape[i]).collect(),
            strides: perm.iter().map(|&i| self.strides[i]).collect(),
            len: self.len,
            order: self.order,
            _marker: PhantomData,
        })
    }

    /// O(1) zero-copy swapaxes.
    pub fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<Self> {
        let a1 = normalize_axis(axis1, self.ndim())?;
        let a2 = normalize_axis(axis2, self.ndim())?;
        let mut out = self.clone();
        out.shape.swap(a1, a2);
        out.strides.swap(a1, a2);
        Ok(out)
    }

    /// O(1) zero-copy flip.
    pub fn flip(&self, axes: Option<&[isize]>) -> Result<Self> {
        let ax_list: Vec<usize> = match axes {
            None => (0..self.ndim()).collect(),
            Some(ax) => numpy_types::normalize_axes(ax, self.ndim())?,
        };
        let itemsize = mem::size_of::<T>() as isize;
        let mut new_strides = self.strides.clone();
        let mut new_byte_off = self.byte_offset as isize;
        for ax in ax_list {
            let dim = self.shape[ax];
            if dim > 1 && self.len > 0 {
                new_byte_off += ((dim - 1) as isize) * new_strides[ax] * itemsize;
            }
            new_strides[ax] = if dim <= 1 {
                new_strides[ax].checked_neg().unwrap_or(0)
            } else {
                new_strides[ax].checked_neg().ok_or_else(|| {
                    NumPyError::ShapeMismatch("flip stride overflow".into())
                })?
            };
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            byte_offset: if self.len == 0 {
                self.byte_offset
            } else {
                new_byte_off as usize
            },
            shape: self.shape.clone(),
            strides: new_strides,
            len: self.len,
            order: self.order,
            _marker: PhantomData,
        })
    }

    /// Materialize the mapped view into an owned, 64-byte aligned `NdArray<T>` (C-order).
    pub fn to_ndarray(&self) -> NdArray<T>
    where
        T: Send + Sync,
    {
        if self.len == 0 {
            return NdArray::from_aligned_buffer(AlignedBuffer::empty(), &self.shape, Order::C)
                .unwrap_or_else(|_| unreachable!());
        }
        let itemsize = mem::size_of::<T>();
        self.storage
            .with_readable_bytes(|buf| {
                let mut uninit = AlignedBuffer::<T>::try_new_uninit(self.len)?;
                let dst = uninit.as_mut_slice();
                if self.is_c_contiguous() && self.is_aligned() {
                    let byte_end = self
                        .len
                        .checked_mul(itemsize)
                        .and_then(|bytes| self.byte_offset.checked_add(bytes))
                        .ok_or_else(|| {
                            NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
                        })?;
                    if byte_end > buf.len() {
                        return Err(NumPyError::IndexOutOfBounds(
                            "memmap byte offset out of bounds".into(),
                        ));
                    }
                    let sl = unsafe {
                        std::slice::from_raw_parts(
                            buf.as_ptr().add(self.byte_offset) as *const T,
                            self.len,
                        )
                    };
                    if self.len >= RAYON_THRESHOLD {
                        dst.par_chunks_mut(4096)
                            .zip(sl.par_chunks(4096))
                            .for_each(|(d, s)| unsafe {
                                ptr::copy_nonoverlapping(
                                    s.as_ptr(),
                                    d.as_mut_ptr() as *mut T,
                                    s.len(),
                                );
                            });
                    } else {
                        unsafe {
                            ptr::copy_nonoverlapping(
                                sl.as_ptr(),
                                dst.as_mut_ptr() as *mut T,
                                self.len,
                            );
                        }
                    }
                    let buf_owned = unsafe { uninit.assume_init() };
                    return NdArray::from_aligned_buffer(buf_owned, &self.shape, Order::C);
                }

                if self.byte_offset > buf.len() {
                    return Err(NumPyError::IndexOutOfBounds(
                        "memmap byte offset out of bounds".into(),
                    ));
                }
                let itemsize_i = itemsize as isize;
                let base_addr = unsafe { buf.as_ptr().add(self.byte_offset) } as usize;
                let ndim = self.ndim();
                if self.len >= RAYON_THRESHOLD && ndim > 0 {
                    const CHUNK: usize = 2048;
                    let shape = &self.shape;
                    let strides = &self.strides;
                    dst.par_chunks_mut(CHUNK)
                        .enumerate()
                        .for_each(|(chunk_idx, out_chunk)| {
                            let mut coords = vec![0usize; ndim];
                            linear_to_coords(chunk_idx * CHUNK, shape, &mut coords);
                            let chunk_len = out_chunk.len();
                            for (i, slot) in out_chunk.iter_mut().enumerate() {
                                let elem_off = coords_offset(&coords, strides);
                                let p = unsafe {
                                    (base_addr as *const u8).offset(elem_off * itemsize_i)
                                } as *const T;
                                slot.write(unsafe { ptr::read_unaligned(p) });
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
                } else {
                    let mut coords = vec![0usize; ndim];
                    for (idx, slot) in dst.iter_mut().enumerate() {
                        let elem_off = coords_offset(&coords, &self.strides);
                        let p = unsafe { (base_addr as *const u8).offset(elem_off * itemsize_i) }
                            as *const T;
                        slot.write(unsafe { ptr::read_unaligned(p) });
                        if idx + 1 < self.len {
                            for d in (0..ndim).rev() {
                                coords[d] += 1;
                                if coords[d] < self.shape[d] {
                                    break;
                                }
                                coords[d] = 0;
                            }
                        }
                    }
                }
                let buf_owned = unsafe { uninit.assume_init() };
                NdArray::from_aligned_buffer(buf_owned, &self.shape, Order::C)
            })
            .unwrap_or_else(|e| panic!("memmap read failed: {e}"))
    }

    /// Write values from an `NdArray<T>` into this writable `Memmap<T>`.
    pub fn write_from_ndarray(&mut self, src: &NdArray<T>) -> Result<()>
    where
        T: Send + Sync,
    {
        if !self.is_writable() {
            return Err(NumPyError::InvalidValue(
                "assignment destination is read-only".into(),
            ));
        }
        if !check_non_overlapping(&self.shape, &self.strides) {
            return Err(NumPyError::InvalidValue(
                "cannot write to overlapping or broadcast-strided memmap".into(),
            ));
        }
        let src_b = if src.shape() == self.shape.as_slice() {
            src.clone()
        } else {
            src.broadcast_to(&self.shape)?
        };
        if self.len == 0 {
            return Ok(());
        }
        if let (Some(dst_sl), Some(src_sl)) = (self.as_slice_mut()?, src_b.as_slice()) {
            dst_sl.copy_from_slice(src_sl);
            return Ok(());
        }
        let itemsize = mem::size_of::<T>();
        let itemsize_i = itemsize as isize;
        let base_off = self.byte_offset as isize;
        let ndim = self.ndim();
        let shape = &self.shape;
        let strides = &self.strides;
        let len = self.len;
        self.storage.with_writable_bytes(|buf| {
            let mut coords = vec![0usize; ndim];
            for (idx, val) in src_b.iter().enumerate() {
                let elem_off = coords_offset(&coords, strides);
                let byte_pos = base_off
                    .checked_add(elem_off.checked_mul(itemsize_i).ok_or_else(|| {
                        NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
                    })?)
                    .ok_or_else(|| {
                        NumPyError::IndexOutOfBounds("memmap byte offset out of bounds".into())
                    })?;
                if byte_pos < 0
                    || (byte_pos as usize)
                        .checked_add(itemsize)
                        .is_none_or(|end| end > buf.len())
                {
                    return Err(NumPyError::IndexOutOfBounds(
                        "memmap byte offset out of bounds".into(),
                    ));
                }
                let p = unsafe { buf.as_mut_ptr().add(byte_pos as usize) } as *mut T;
                unsafe { ptr::write_unaligned(p, val) };
                if idx + 1 < len {
                    for d in (0..ndim).rev() {
                        coords[d] += 1;
                        if coords[d] < shape[d] {
                            break;
                        }
                        coords[d] = 0;
                    }
                }
            }
            Ok(())
        })
    }
}

/// Dynamically-typed memory-mapped array across all 14 `DType`s.
#[derive(Clone)]
pub enum DynMemmap {
    Bool(Memmap<bool>),
    I8(Memmap<i8>),
    I16(Memmap<i16>),
    I32(Memmap<i32>),
    I64(Memmap<i64>),
    U8(Memmap<u8>),
    U16(Memmap<u16>),
    U32(Memmap<u32>),
    U64(Memmap<u64>),
    F16(Memmap<f16>),
    F32(Memmap<f32>),
    F64(Memmap<f64>),
    C64(Memmap<Complex32>),
    C128(Memmap<Complex64>),
}

macro_rules! dispatch_memmap {
    ($val:expr, $ident:ident => $body:expr) => {
        match $val {
            DynMemmap::Bool($ident) => $body,
            DynMemmap::I8($ident) => $body,
            DynMemmap::I16($ident) => $body,
            DynMemmap::I32($ident) => $body,
            DynMemmap::I64($ident) => $body,
            DynMemmap::U8($ident) => $body,
            DynMemmap::U16($ident) => $body,
            DynMemmap::U32($ident) => $body,
            DynMemmap::U64($ident) => $body,
            DynMemmap::F16($ident) => $body,
            DynMemmap::F32($ident) => $body,
            DynMemmap::F64($ident) => $body,
            DynMemmap::C64($ident) => $body,
            DynMemmap::C128($ident) => $body,
        }
    };
}

impl DynMemmap {
    pub fn open<P: AsRef<Path>>(
        path: P,
        dtype: DType,
        mode: MemmapMode,
        byte_offset: usize,
        shape: Option<&[usize]>,
        order: Order,
    ) -> Result<Self> {
        Ok(match dtype {
            DType::Bool => Self::Bool(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::I8 => Self::I8(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::I16 => Self::I16(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::I32 => Self::I32(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::I64 => Self::I64(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::U8 => Self::U8(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::U16 => Self::U16(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::U32 => Self::U32(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::U64 => Self::U64(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::F16 => Self::F16(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::F32 => Self::F32(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::F64 => Self::F64(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::C64 => Self::C64(Memmap::open(path, mode, byte_offset, shape, order)?),
            DType::C128 => Self::C128(Memmap::open(path, mode, byte_offset, shape, order)?),
        })
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::C64(_) => DType::C64,
            Self::C128(_) => DType::C128,
        }
    }

    pub fn shape(&self) -> &[usize] {
        dispatch_memmap!(self, m => m.shape())
    }

    pub fn strides(&self) -> &[isize] {
        dispatch_memmap!(self, m => m.strides())
    }

    pub fn len(&self) -> usize {
        dispatch_memmap!(self, m => m.len())
    }

    pub fn is_empty(&self) -> bool {
        dispatch_memmap!(self, m => m.is_empty())
    }

    pub fn flush(&self) -> Result<()> {
        dispatch_memmap!(self, m => m.flush())
    }

    pub fn slice(&self, args: &[SliceArg]) -> Result<Self> {
        Ok(match self {
            Self::Bool(m) => Self::Bool(m.slice(args)?),
            Self::I8(m) => Self::I8(m.slice(args)?),
            Self::I16(m) => Self::I16(m.slice(args)?),
            Self::I32(m) => Self::I32(m.slice(args)?),
            Self::I64(m) => Self::I64(m.slice(args)?),
            Self::U8(m) => Self::U8(m.slice(args)?),
            Self::U16(m) => Self::U16(m.slice(args)?),
            Self::U32(m) => Self::U32(m.slice(args)?),
            Self::U64(m) => Self::U64(m.slice(args)?),
            Self::F16(m) => Self::F16(m.slice(args)?),
            Self::F32(m) => Self::F32(m.slice(args)?),
            Self::F64(m) => Self::F64(m.slice(args)?),
            Self::C64(m) => Self::C64(m.slice(args)?),
            Self::C128(m) => Self::C128(m.slice(args)?),
        })
    }

    pub fn transpose(&self, axes: Option<&[isize]>) -> Result<Self> {
        Ok(match self {
            Self::Bool(m) => Self::Bool(m.transpose(axes)?),
            Self::I8(m) => Self::I8(m.transpose(axes)?),
            Self::I16(m) => Self::I16(m.transpose(axes)?),
            Self::I32(m) => Self::I32(m.transpose(axes)?),
            Self::I64(m) => Self::I64(m.transpose(axes)?),
            Self::U8(m) => Self::U8(m.transpose(axes)?),
            Self::U16(m) => Self::U16(m.transpose(axes)?),
            Self::U32(m) => Self::U32(m.transpose(axes)?),
            Self::U64(m) => Self::U64(m.transpose(axes)?),
            Self::F16(m) => Self::F16(m.transpose(axes)?),
            Self::F32(m) => Self::F32(m.transpose(axes)?),
            Self::F64(m) => Self::F64(m.transpose(axes)?),
            Self::C64(m) => Self::C64(m.transpose(axes)?),
            Self::C128(m) => Self::C128(m.transpose(axes)?),
        })
    }

    pub fn to_dyn_array(&self) -> DynArray {
        match self {
            Self::Bool(m) => DynArray::Bool(m.to_ndarray()),
            Self::I8(m) => DynArray::I8(m.to_ndarray()),
            Self::I16(m) => DynArray::I16(m.to_ndarray()),
            Self::I32(m) => DynArray::I32(m.to_ndarray()),
            Self::I64(m) => DynArray::I64(m.to_ndarray()),
            Self::U8(m) => DynArray::U8(m.to_ndarray()),
            Self::U16(m) => DynArray::U16(m.to_ndarray()),
            Self::U32(m) => DynArray::U32(m.to_ndarray()),
            Self::U64(m) => DynArray::U64(m.to_ndarray()),
            Self::F16(m) => DynArray::F16(m.to_ndarray()),
            Self::F32(m) => DynArray::F32(m.to_ndarray()),
            Self::F64(m) => DynArray::F64(m.to_ndarray()),
            Self::C64(m) => DynArray::C64(m.to_ndarray()),
            Self::C128(m) => DynArray::C128(m.to_ndarray()),
        }
    }
}

/// Helper alias matching NumPy's `np.memmap(path, dtype, mode, offset, shape, order)`.
pub fn memmap<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    mode: MemmapMode,
    byte_offset: usize,
    shape: Option<&[usize]>,
    order: Order,
) -> Result<DynMemmap> {
    DynMemmap::open(path, dtype, mode, byte_offset, shape, order)
}

/// Typed helper `memmap_typed::<T>(...)`.
pub fn memmap_typed<T: Scalar, P: AsRef<Path>>(
    path: P,
    mode: MemmapMode,
    byte_offset: usize,
    shape: Option<&[usize]>,
    order: Order,
) -> Result<Memmap<T>> {
    Memmap::open(path, mode, byte_offset, shape, order)
}

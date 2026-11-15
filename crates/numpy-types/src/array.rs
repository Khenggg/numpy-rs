use crate::buffer::AlignedBuffer;
use crate::dtype::{DType, Scalar};
use crate::error::{NumPyError, Result};
use crate::shape::{
    broadcast_strides, check_non_overlapping, compute_c_strides, compute_f_strides, compute_strides,
    is_c_contiguous, is_f_contiguous, normalize_axes, normalize_axis, normalize_index,
    resolve_reshape_dims, shape_size, Order, Slice, SliceArg,
};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

/// Memory contiguity classification of an `NdArray` or view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    CContiguous,
    FContiguous,
    StridedPositive,
    StridedGeneral,
}

/// Core statically-typed N-dimensional array with 64-byte SIMD-aligned CoW storage
/// and NumPy-compatible signed striding (supporting O(1) views, negative strides,
/// and 0-stride broadcasting).
pub struct NdArray<T> {
    storage: Arc<AlignedBuffer<T>>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
    len: usize,
}

impl<T> Clone for NdArray<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            len: self.len,
        }
    }
}

impl<T> NdArray<T> {
    /// Wrap an `AlignedBuffer<T>` into an `NdArray<T>` with the requested `Order`.
    pub fn from_aligned_buffer(
        buffer: AlignedBuffer<T>,
        shape: &[usize],
        order: Order,
    ) -> Result<Self> {
        let len = shape_size(shape)?;
        if buffer.len() != len {
            return Err(NumPyError::ShapeMismatch(format!(
                "cannot create array of shape {shape:?} (size {len}) from buffer of len {}",
                buffer.len()
            )));
        }
        let strides = compute_strides(shape, order);
        Ok(Self {
            storage: Arc::new(buffer),
            offset: 0,
            shape: shape.to_vec(),
            strides,
            len,
        })
    }

    /// Construct from explicit `(storage, offset, shape, strides)` after verifying
    /// every valid coordinate stays strictly inside `storage`.
    pub fn from_storage(
        storage: Arc<AlignedBuffer<T>>,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(NumPyError::ShapeMismatch(format!(
                "shape len {} != strides len {}",
                shape.len(),
                strides.len()
            )));
        }
        let len = shape_size(&shape)?;
        if len > 0 {
            let mut min_off = offset as isize;
            let mut max_off = offset as isize;
            for (&d, &st) in shape.iter().zip(strides.iter()) {
                if d > 1 {
                    let span = (d as isize - 1)
                        .checked_mul(st)
                        .ok_or_else(|| NumPyError::ShapeMismatch("stride span overflow".into()))?;
                    if span < 0 {
                        min_off = min_off
                            .checked_add(span)
                            .ok_or_else(|| NumPyError::ShapeMismatch("offset underflow".into()))?;
                    } else {
                        max_off = max_off
                            .checked_add(span)
                            .ok_or_else(|| NumPyError::ShapeMismatch("offset overflow".into()))?;
                    }
                }
            }
            if min_off < 0 || (max_off as usize) >= storage.len() {
                return Err(NumPyError::IndexOutOfBounds(format!(
                    "strided view bounds [{min_off}, {max_off}] exceed storage len {}",
                    storage.len()
                )));
            }
        } else if offset > storage.len() {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "offset {offset} exceeds storage len {}",
                storage.len()
            )));
        }
        Ok(Self {
            storage,
            offset,
            shape,
            strides,
            len,
        })
    }

    /// Create a zero-filled array (`Order::C`).
    pub fn zeros(shape: &[usize]) -> Result<Self>
    where
        T: Scalar,
    {
        Self::zeros_with_order(shape, Order::C)
    }

    /// Create a zero-filled array with explicit `Order`.
    pub fn zeros_with_order(shape: &[usize], order: Order) -> Result<Self>
    where
        T: Scalar,
    {
        let len = shape_size(shape)?;
        let buf = AlignedBuffer::new_zeroed(len);
        Self::from_aligned_buffer(buf, shape, order)
    }

    /// Create a one-filled array (`Order::C`).
    pub fn ones(shape: &[usize]) -> Result<Self>
    where
        T: Scalar,
    {
        Self::full(shape, T::one())
    }

    /// Create an array filled with `value`.
    pub fn full(shape: &[usize], value: T) -> Result<Self>
    where
        T: Copy,
    {
        Self::full_with_order(shape, value, Order::C)
    }

    /// Create an array filled with `value` and explicit `Order`.
    pub fn full_with_order(shape: &[usize], value: T, order: Order) -> Result<Self>
    where
        T: Copy,
    {
        let len = shape_size(shape)?;
        let buf = AlignedBuffer::filled(len, value);
        Self::from_aligned_buffer(buf, shape, order)
    }

    /// Create a 0-dimensional scalar array.
    pub fn from_scalar(value: T) -> Self
    where
        T: Copy,
    {
        let buf = AlignedBuffer::filled(1, value);
        Self {
            storage: Arc::new(buf),
            offset: 0,
            shape: Vec::new(),
            strides: Vec::new(),
            len: 1,
        }
    }

    /// Create a 1-D array from a `Vec<T>`.
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len();
        let buf = AlignedBuffer::from_vec(vec);
        Self {
            storage: Arc::new(buf),
            offset: 0,
            shape: vec![len],
            strides: vec![1],
            len,
        }
    }

    /// Create an N-D array in C-order from `shape` and flat `Vec<T>`.
    pub fn from_shape_vec(shape: &[usize], vec: Vec<T>) -> Result<Self> {
        Self::from_shape_vec_order(shape, vec, Order::C)
    }

    /// Create an N-D array from `shape`, flat data `vec` (interpreted in `order`), and `order`.
    pub fn from_shape_vec_order(shape: &[usize], vec: Vec<T>, order: Order) -> Result<Self> {
        let len = shape_size(shape)?;
        if vec.len() != len {
            return Err(NumPyError::ShapeMismatch(format!(
                "cannot reshape vector of len {} into shape {shape:?} (size {len})",
                vec.len()
            )));
        }
        let buf = AlignedBuffer::from_vec(vec);
        Self::from_aligned_buffer(buf, shape, order)
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
    pub fn dtype(&self) -> DType
    where
        T: Scalar,
    {
        T::DTYPE
    }

    #[inline]
    pub fn itemsize(&self) -> usize {
        std::mem::size_of::<T>()
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.len * self.itemsize()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn storage_ptr(&self) -> *const T {
        self.storage.as_ptr()
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
    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous() || self.is_f_contiguous()
    }

    pub fn layout(&self) -> MemoryLayout {
        if self.is_c_contiguous() {
            MemoryLayout::CContiguous
        } else if self.is_f_contiguous() {
            MemoryLayout::FContiguous
        } else if self.strides.iter().all(|&s| s >= 0) {
            MemoryLayout::StridedPositive
        } else {
            MemoryLayout::StridedGeneral
        }
    }

    /// Return `true` if `self` and `other` reference the exact same underlying storage buffer.
    #[inline]
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Return `true` if this array owns the sole strong reference to its buffer.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.storage) == 1
    }

    /// Fast O(1) immutable contiguous slice if C-contiguous (works for 0-D scalars too).
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.len == 0 {
            return Some(&[]);
        }
        if self.is_c_contiguous() {
            unsafe {
                Some(std::slice::from_raw_parts(
                    self.storage.as_ptr().add(self.offset),
                    self.len,
                ))
            }
        } else {
            None
        }
    }

    /// Fast O(1) mutable contiguous slice (clones CoW buffer if shared, returns `None` if non-C-contiguous).
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]>
    where
        T: Clone,
    {
        if self.len == 0 {
            return Some(&mut []);
        }
        if !self.is_c_contiguous() {
            return None;
        }
        let buf = Arc::make_mut(&mut self.storage);
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                buf.as_mut_ptr().add(self.offset),
                self.len,
            ))
        }
    }

    /// Borrow a lightweight zero-copy `ArrayView<'_, T>`.
    #[inline]
    pub fn view(&self) -> ArrayView<'_, T> {
        let ptr = if self.len == 0 {
            NonNull::dangling()
        } else {
            unsafe { NonNull::new_unchecked(self.storage.as_ptr().add(self.offset) as *mut T) }
        };
        ArrayView {
            ptr,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Borrow a zero-copy `ArrayViewMut<'_, T>`, ensuring CoW uniqueness and non-overlapping strides.
    pub fn view_mut(&mut self) -> Result<ArrayViewMut<'_, T>>
    where
        T: Clone,
    {
        if !check_non_overlapping(&self.shape, &self.strides) {
            return Err(NumPyError::InvalidValue(
                "cannot mutably borrow an overlapping or broadcast-strided array".into(),
            ));
        }
        let ptr = if self.len == 0 {
            NonNull::dangling()
        } else {
            let buf = Arc::make_mut(&mut self.storage);
            unsafe { NonNull::new_unchecked(buf.as_mut_ptr().add(self.offset)) }
        };
        Ok(ArrayViewMut {
            ptr,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            len: self.len,
            _marker: PhantomData,
        })
    }

    /// Copy elements into a fresh C-contiguous array (if already compact C-contiguous, shares/clones cheaply).
    pub fn to_c_contiguous(&self) -> Self
    where
        T: Copy,
    {
        if self.is_c_contiguous() && self.offset == 0 && self.storage.len() == self.len {
            return self.clone();
        }
        self.deep_copy(Order::C)
    }

    /// Copy elements into a fresh Fortran-contiguous array.
    pub fn to_f_contiguous(&self) -> Self
    where
        T: Copy,
    {
        if self.is_f_contiguous() && self.offset == 0 && self.storage.len() == self.len {
            return self.clone();
        }
        self.deep_copy(Order::F)
    }

    /// Return a freshly allocated copy in `order` (`C`, `F`, `A`, `K`).
    pub fn deep_copy(&self, order: Order) -> Self
    where
        T: Copy,
    {
        let resolved_order = match order {
            Order::C => Order::C,
            Order::F => Order::F,
            Order::A | Order::K => {
                if self.is_f_contiguous() && !self.is_c_contiguous() {
                    Order::F
                } else {
                    Order::C
                }
            }
        };
        if self.len == 0 {
            return Self {
                storage: Arc::new(AlignedBuffer::empty()),
                offset: 0,
                shape: self.shape.clone(),
                strides: compute_strides(&self.shape, resolved_order),
                len: 0,
            };
        }
        if resolved_order == Order::C {
            if let Some(sl) = self.as_slice() {
                let buf = AlignedBuffer::from_slice(sl);
                return Self::from_aligned_buffer(buf, &self.shape, Order::C).unwrap();
            }
        }
        let mut uninit = AlignedBuffer::<T>::new_uninit(self.len);
        let dst = uninit.as_mut_slice();
        match resolved_order {
            Order::C => {
                for (i, val) in self.iter().enumerate() {
                    dst[i].write(val);
                }
            }
            Order::F => {
                let dst_strides = compute_f_strides(&self.shape);
                let mut idx = vec![0usize; self.ndim()];
                for val in self.iter() {
                    let mut pos = 0usize;
                    for d in 0..self.ndim() {
                        pos += idx[d] * (dst_strides[d] as usize);
                    }
                    dst[pos].write(val);
                    // increment idx in C-order to match `self.iter()`
                    for d in (0..self.ndim()).rev() {
                        idx[d] += 1;
                        if idx[d] < self.shape[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
            }
            _ => unreachable!(),
        }
        let buf = unsafe { uninit.assume_init() };
        Self::from_aligned_buffer(buf, &self.shape, resolved_order).unwrap()
    }

    /// O(1) zero-copy view slicing supporting integer indices, ranges, negative steps, `NewAxis`, and `Ellipsis`.
    pub fn slice(&self, args: &[SliceArg]) -> Result<Self> {
        let has_ellipsis = args.iter().filter(|a| matches!(a, SliceArg::Ellipsis)).count();
        if has_ellipsis > 1 {
            return Err(NumPyError::IndexOutOfBounds(
                "an index can only have a single ellipsis ('...')".into(),
            ));
        }
        let consumed_axes = args
            .iter()
            .filter(|a| !matches!(a, SliceArg::NewAxis | SliceArg::Ellipsis))
            .count();
        if consumed_axes > self.ndim() {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "too many indices for array: array is {}-dimensional, but {consumed_axes} were indexed",
                self.ndim()
            )));
        }

        let ellipsis_expand = self.ndim() - consumed_axes;
        let mut expanded = Vec::with_capacity(self.ndim() + args.len());
        for &arg in args {
            match arg {
                SliceArg::Ellipsis => {
                    for _ in 0..ellipsis_expand {
                        expanded.push(SliceArg::Slice(Slice::full()));
                    }
                }
                other => expanded.push(other),
            }
        }
        let remaining = self.ndim() - expanded.iter().filter(|a| !matches!(a, SliceArg::NewAxis)).count();
        for _ in 0..remaining {
            expanded.push(SliceArg::Slice(Slice::full()));
        }

        let mut new_offset = self.offset as isize;
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
                    let dim = self.shape[src_axis];
                    let norm = normalize_index(idx, dim)?;
                    new_offset += (norm as isize) * self.strides[src_axis];
                    src_axis += 1;
                }
                SliceArg::Slice(sl) => {
                    let dim = self.shape[src_axis];
                    let (first, count, step) = sl.resolve(dim)?;
                    if count > 0 {
                        new_offset += (first as isize) * self.strides[src_axis];
                    }
                    new_shape.push(count);
                    new_strides.push(self.strides[src_axis] * step);
                    src_axis += 1;
                }
                SliceArg::Ellipsis => unreachable!(),
            }
        }

        let new_len = shape_size(&new_shape)?;
        let safe_offset = if new_len == 0 { 0 } else { new_offset as usize };
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: safe_offset,
            shape: new_shape,
            strides: new_strides,
            len: new_len,
        })
    }

    /// O(1) view selecting a single index along `axis`, reducing dimensionality by 1.
    pub fn index_axis(&self, axis: isize, index: isize) -> Result<Self> {
        let ax = normalize_axis(axis, self.ndim())?;
        let idx = normalize_index(index, self.shape[ax])?;
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(ax);
        let st = new_strides.remove(ax);
        let new_len = shape_size(&new_shape)?;
        let new_offset = if new_len == 0 {
            0
        } else {
            ((self.offset as isize) + (idx as isize) * st) as usize
        };
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: new_offset,
            shape: new_shape,
            strides: new_strides,
            len: new_len,
        })
    }

    /// O(1) zero-copy transpose (reverses axes if `axes` is `None`).
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
                normalize_axes(ax, ndim)?
            }
        };
        let new_shape = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides = perm.iter().map(|&i| self.strides[i]).collect();
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            shape: new_shape,
            strides: new_strides,
            len: self.len,
        })
    }

    /// O(1) full transpose (`.T`).
    #[inline]
    pub fn t(&self) -> Self {
        self.transpose(None).unwrap()
    }

    /// O(1) matrix transpose (`.mT`, swaps the last two axes; errors if `ndim < 2`).
    pub fn m_t(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(NumPyError::InvalidValue(format!(
                "matrix transpose requires at least 2 dimensions, got {}",
                self.ndim()
            )));
        }
        self.swapaxes(-2, -1)
    }

    /// O(1) swap two axes.
    pub fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<Self> {
        let a1 = normalize_axis(axis1, self.ndim())?;
        let a2 = normalize_axis(axis2, self.ndim())?;
        let mut out = self.clone();
        out.shape.swap(a1, a2);
        out.strides.swap(a1, a2);
        Ok(out)
    }

    /// O(1) reverse elements along the given axes (all axes if `None`) using negative strides.
    pub fn flip(&self, axes: Option<&[isize]>) -> Result<Self> {
        let ax_list: Vec<usize> = match axes {
            None => (0..self.ndim()).collect(),
            Some(ax) => normalize_axes(ax, self.ndim())?,
        };
        let mut new_strides = self.strides.clone();
        let mut new_offset = self.offset as isize;
        for ax in ax_list {
            let dim = self.shape[ax];
            if dim > 1 && self.len > 0 {
                new_offset += ((dim - 1) as isize) * new_strides[ax];
            }
            new_strides[ax] = -new_strides[ax];
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: if self.len == 0 { 0 } else { new_offset as usize },
            shape: self.shape.clone(),
            strides: new_strides,
            len: self.len,
        })
    }

    /// O(1) remove size-1 dimensions.
    pub fn squeeze(&self, axes: Option<&[isize]>) -> Result<Self> {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        match axes {
            None => {
                for (&d, &st) in self.shape.iter().zip(self.strides.iter()) {
                    if d != 1 {
                        new_shape.push(d);
                        new_strides.push(st);
                    }
                }
            }
            Some(ax_list) => {
                let norm = normalize_axes(ax_list, self.ndim())?;
                for &ax in &norm {
                    if self.shape[ax] != 1 {
                        return Err(NumPyError::InvalidValue(format!(
                            "cannot select an axis to squeeze out which has size {} != 1",
                            self.shape[ax]
                        )));
                    }
                }
                for (i, (&d, &st)) in self.shape.iter().zip(self.strides.iter()).enumerate() {
                    if !norm.contains(&i) {
                        new_shape.push(d);
                        new_strides.push(st);
                    }
                }
            }
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            shape: new_shape,
            strides: new_strides,
            len: self.len,
        })
    }

    /// O(1) insert size-1 dimensions at the given output axis positions.
    pub fn expand_dims(&self, axes: &[isize]) -> Result<Self> {
        let out_ndim = self.ndim() + axes.len();
        let norm = normalize_axes(axes, out_ndim)?;
        let mut new_shape = vec![0usize; out_ndim];
        let mut new_strides = vec![0isize; out_ndim];
        let mut src = 0usize;
        for i in 0..out_ndim {
            if norm.contains(&i) {
                new_shape[i] = 1;
                new_strides[i] = 0;
            } else {
                new_shape[i] = self.shape[src];
                new_strides[i] = self.strides[src];
                src += 1;
            }
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            shape: new_shape,
            strides: new_strides,
            len: self.len,
        })
    }

    /// O(1) broadcast array to `target_shape` (using 0-strides).
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self> {
        let new_strides = broadcast_strides(&self.shape, &self.strides, target_shape)?;
        let new_len = shape_size(target_shape)?;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: if new_len == 0 { 0 } else { self.offset },
            shape: target_shape.to_vec(),
            strides: new_strides,
            len: new_len,
        })
    }

    /// Reshape array (supports `-1` dimension inference), zero-copy when C-contiguous or compatible strides.
    pub fn reshape(&self, spec: &[isize]) -> Result<Self>
    where
        T: Copy,
    {
        let new_shape = resolve_reshape_dims(self.len, spec)?;
        if new_shape == self.shape {
            return Ok(self.clone());
        }
        if self.len == 0 {
            return Ok(Self {
                storage: Arc::new(AlignedBuffer::empty()),
                offset: 0,
                strides: compute_c_strides(&new_shape),
                shape: new_shape,
                len: 0,
            });
        }
        if let Some(new_strides) = attempt_nocopy_reshape_c(&self.shape, &self.strides, &new_shape) {
            return Ok(Self {
                storage: Arc::clone(&self.storage),
                offset: self.offset,
                shape: new_shape,
                strides: new_strides,
                len: self.len,
            });
        }
        let contig = self.deep_copy(Order::C);
        let strides = compute_c_strides(&new_shape);
        Ok(Self {
            storage: contig.storage,
            offset: 0,
            shape: new_shape,
            strides,
            len: self.len,
        })
    }

    /// Return a contiguous 1-D view if possible, else copy (`Order`).
    pub fn ravel(&self, order: Order) -> Self
    where
        T: Copy,
    {
        if self.len == 0 {
            return Self {
                storage: Arc::new(AlignedBuffer::empty()),
                offset: 0,
                shape: vec![0],
                strides: vec![1],
                len: 0,
            };
        }
        let want_f = matches!(order, Order::F);
        let can_zero_copy = match order {
            Order::C => self.is_c_contiguous(),
            Order::F => self.is_f_contiguous(),
            Order::A | Order::K => self.is_contiguous(),
        };
        if can_zero_copy {
            return Self {
                storage: Arc::clone(&self.storage),
                offset: self.offset,
                shape: vec![self.len],
                strides: vec![1],
                len: self.len,
            };
        }
        let copied = if want_f {
            self.deep_copy(Order::F)
        } else {
            self.deep_copy(Order::C)
        };
        Self {
            storage: copied.storage,
            offset: 0,
            shape: vec![self.len],
            strides: vec![1],
            len: self.len,
        }
    }

    /// Always return a freshly allocated 1-D copy.
    pub fn flatten(&self, order: Order) -> Self
    where
        T: Copy,
    {
        let mut copied = self.deep_copy(order);
        copied.shape = vec![self.len];
        copied.strides = vec![1];
        copied
    }

    /// Read a single element at multi-index `idx` (supports negative indices).
    pub fn get(&self, idx: &[isize]) -> Result<T>
    where
        T: Copy,
    {
        if idx.len() != self.ndim() {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "expected {} indices, got {}",
                self.ndim(),
                idx.len()
            )));
        }
        let mut off = self.offset as isize;
        for (ax, &i) in idx.iter().enumerate() {
            let norm = normalize_index(i, self.shape[ax])?;
            off += (norm as isize) * self.strides[ax];
        }
        Ok(self.storage.as_slice()[off as usize])
    }

    /// Write a single element at multi-index `idx` (CoW unique if shared).
    pub fn set(&mut self, idx: &[isize], val: T) -> Result<()>
    where
        T: Clone,
    {
        if idx.len() != self.ndim() {
            return Err(NumPyError::IndexOutOfBounds(format!(
                "expected {} indices, got {}",
                self.ndim(),
                idx.len()
            )));
        }
        let mut off = self.offset as isize;
        for (ax, &i) in idx.iter().enumerate() {
            let norm = normalize_index(i, self.shape[ax])?;
            off += (norm as isize) * self.strides[ax];
        }
        let buf = Arc::make_mut(&mut self.storage);
        buf.as_mut_slice()[off as usize] = val;
        Ok(())
    }

    /// Logical C-order iterator over elements by value (`Copy`).
    pub fn iter(&self) -> NdIter<'_, T>
    where
        T: Copy,
    {
        self.view().iter()
    }

    /// Materialize elements in logical C-order into a `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Copy,
    {
        if let Some(sl) = self.as_slice() {
            return sl.to_vec();
        }
        self.iter().collect()
    }

    /// Apply a unary closure element-wise into a freshly allocated C-contiguous `NdArray<U>`.
    pub fn map<U, F>(&self, mut f: F) -> NdArray<U>
    where
        T: Copy,
        F: FnMut(T) -> U,
    {
        let mut uninit = AlignedBuffer::<U>::new_uninit(self.len);
        let dst = uninit.as_mut_slice();
        if let Some(sl) = self.as_slice() {
            for (slot, &x) in dst.iter_mut().zip(sl.iter()) {
                slot.write(f(x));
            }
        } else {
            for (slot, x) in dst.iter_mut().zip(self.iter()) {
                slot.write(f(x));
            }
        }
        let buf = unsafe { uninit.assume_init() };
        NdArray::from_aligned_buffer(buf, &self.shape, Order::C).unwrap()
    }
}

/// Zero-copy borrowed immutable view over strided elements.
#[derive(Clone)]
pub struct ArrayView<'a, T> {
    ptr: NonNull<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    len: usize,
    _marker: PhantomData<&'a T>,
}

unsafe impl<'a, T: Sync> Send for ArrayView<'a, T> {}
unsafe impl<'a, T: Sync> Sync for ArrayView<'a, T> {}

impl<'a, T> ArrayView<'a, T> {
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
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
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
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.len == 0 {
            return Some(&[]);
        }
        if self.is_c_contiguous() {
            Some(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
        } else {
            None
        }
    }

    /// Borrow a sub-view along `axis` at `index`, reducing rank by 1.
    pub fn index_axis(&self, axis: isize, index: isize) -> Result<Self> {
        let ax = normalize_axis(axis, self.ndim())?;
        let idx = normalize_index(index, self.shape[ax])?;
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(ax);
        let st = new_strides.remove(ax);
        let new_len = shape_size(&new_shape)?;
        let ptr = if new_len == 0 {
            NonNull::dangling()
        } else {
            unsafe { NonNull::new_unchecked(self.ptr.as_ptr().offset((idx as isize) * st)) }
        };
        Ok(Self {
            ptr,
            shape: new_shape,
            strides: new_strides,
            len: new_len,
            _marker: PhantomData,
        })
    }

    /// Iterate over sub-views along `axis`.
    pub fn axis_iter(&self, axis: isize) -> Result<impl Iterator<Item = ArrayView<'a, T>> + '_> {
        let ax = normalize_axis(axis, self.ndim())?;
        let count = self.shape[ax];
        Ok((0..count).map(move |i| self.index_axis(ax as isize, i as isize).unwrap()))
    }

    /// O(1) view broadcast.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self> {
        let new_strides = broadcast_strides(&self.shape, &self.strides, target_shape)?;
        let new_len = shape_size(target_shape)?;
        Ok(Self {
            ptr: if new_len == 0 {
                NonNull::dangling()
            } else {
                self.ptr
            },
            shape: target_shape.to_vec(),
            strides: new_strides,
            len: new_len,
            _marker: PhantomData,
        })
    }

    /// O(1) transpose.
    pub fn transpose(&self, axes: Option<&[isize]>) -> Result<Self> {
        let ndim = self.ndim();
        let perm: Vec<usize> = match axes {
            None => (0..ndim).rev().collect(),
            Some(ax) => {
                if ax.len() != ndim {
                    return Err(NumPyError::InvalidValue("axes don't match array".into()));
                }
                normalize_axes(ax, ndim)?
            }
        };
        Ok(Self {
            ptr: self.ptr,
            shape: perm.iter().map(|&i| self.shape[i]).collect(),
            strides: perm.iter().map(|&i| self.strides[i]).collect(),
            len: self.len,
            _marker: PhantomData,
        })
    }

    pub fn iter(&self) -> NdIter<'a, T>
    where
        T: Copy,
    {
        if self.is_c_contiguous() {
            let slice = self.as_slice().unwrap_or(&[]);
            NdIter::Contiguous(slice.iter())
        } else {
            NdIter::Strided(StridedCursor::new(self.ptr, &self.shape, &self.strides, self.len))
        }
    }

    pub fn to_owned(&self) -> NdArray<T>
    where
        T: Copy,
    {
        if self.len == 0 {
            return NdArray::from_aligned_buffer(AlignedBuffer::empty(), &self.shape, Order::C)
                .unwrap();
        }
        if let Some(sl) = self.as_slice() {
            return NdArray::from_aligned_buffer(
                AlignedBuffer::from_slice(sl),
                &self.shape,
                Order::C,
            )
            .unwrap();
        }
        let mut uninit = AlignedBuffer::<T>::new_uninit(self.len);
        for (slot, val) in uninit.as_mut_slice().iter_mut().zip(self.iter()) {
            slot.write(val);
        }
        let buf = unsafe { uninit.assume_init() };
        NdArray::from_aligned_buffer(buf, &self.shape, Order::C).unwrap()
    }
}

/// Zero-copy borrowed mutable view over strictly non-overlapping strided elements.
pub struct ArrayViewMut<'a, T> {
    ptr: NonNull<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Send> Send for ArrayViewMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for ArrayViewMut<'a, T> {}

impl<'a, T> ArrayViewMut<'a, T> {
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
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.shape, &self.strides)
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.len == 0 {
            return Some(&mut []);
        }
        if self.is_c_contiguous() {
            Some(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
        } else {
            None
        }
    }

    /// Borrow an immutable view.
    #[inline]
    pub fn as_view(&self) -> ArrayView<'_, T> {
        ArrayView {
            ptr: self.ptr,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Split into disjoint mutable sub-views along `axis` (safe for parallel Rayon writes).
    pub fn axis_iter_mut(mut self, axis: isize) -> Result<Vec<ArrayViewMut<'a, T>>> {
        let ax = normalize_axis(axis, self.ndim())?;
        let count = self.shape[ax];
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(ax);
        let st = new_strides.remove(ax);
        let new_len = shape_size(&new_shape)?;
        let base_ptr = self.ptr.as_ptr();
        let _ = &mut self.ptr;

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let ptr = if new_len == 0 {
                NonNull::dangling()
            } else {
                unsafe { NonNull::new_unchecked(base_ptr.offset((i as isize) * st)) }
            };
            out.push(ArrayViewMut {
                ptr,
                shape: new_shape.clone(),
                strides: new_strides.clone(),
                len: new_len,
                _marker: PhantomData,
            });
        }
        Ok(out)
    }

    /// Fill every element in C-order from a slice.
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if src.len() != self.len {
            return Err(NumPyError::ShapeMismatch(format!(
                "source slice len {} != target len {}",
                src.len(),
                self.len
            )));
        }
        if let Some(dst) = self.as_slice_mut() {
            dst.copy_from_slice(src);
            return Ok(());
        }
        let mut cur = StridedCursorMut::new(self.ptr, &self.shape, &self.strides, self.len);
        for &v in src {
            if let Some(slot) = cur.next() {
                *slot = v;
            }
        }
        Ok(())
    }
}

/// Fast logical C-order iterator with contiguous slice fast-path.
pub enum NdIter<'a, T> {
    Contiguous(std::slice::Iter<'a, T>),
    Strided(StridedCursor<'a, T>),
}

impl<'a, T: Copy> Iterator for NdIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self {
            Self::Contiguous(it) => it.next().copied(),
            Self::Strided(cur) => cur.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Contiguous(it) => it.size_hint(),
            Self::Strided(cur) => (cur.remaining, Some(cur.remaining)),
        }
    }
}

impl<'a, T: Copy> ExactSizeIterator for NdIter<'a, T> {}

pub struct StridedCursor<'a, T> {
    ptr: *const T,
    shape: Vec<usize>,
    strides: Vec<isize>,
    coords: Vec<usize>,
    remaining: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> StridedCursor<'a, T> {
    fn new(ptr: NonNull<T>, shape: &[usize], strides: &[isize], len: usize) -> Self {
        Self {
            ptr: ptr.as_ptr(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            coords: vec![0; shape.len()],
            remaining: len,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Copy> Iterator for StridedCursor<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let val = unsafe { *self.ptr };
        if self.remaining > 0 {
            for d in (0..self.shape.len()).rev() {
                self.coords[d] += 1;
                if self.coords[d] < self.shape[d] {
                    self.ptr = unsafe { self.ptr.offset(self.strides[d]) };
                    break;
                }
                let back = (self.coords[d] - 1) as isize * self.strides[d];
                self.ptr = unsafe { self.ptr.offset(-back) };
                self.coords[d] = 0;
            }
        }
        Some(val)
    }
}

struct StridedCursorMut<'a, T> {
    ptr: *mut T,
    shape: Vec<usize>,
    strides: Vec<isize>,
    coords: Vec<usize>,
    remaining: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> StridedCursorMut<'a, T> {
    fn new(ptr: NonNull<T>, shape: &[usize], strides: &[isize], len: usize) -> Self {
        Self {
            ptr: ptr.as_ptr(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            coords: vec![0; shape.len()],
            remaining: len,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let slot = unsafe { &mut *self.ptr };
        if self.remaining > 0 {
            for d in (0..self.shape.len()).rev() {
                self.coords[d] += 1;
                if self.coords[d] < self.shape[d] {
                    self.ptr = unsafe { self.ptr.offset(self.strides[d]) };
                    break;
                }
                let back = (self.coords[d] - 1) as isize * self.strides[d];
                self.ptr = unsafe { self.ptr.offset(-back) };
                self.coords[d] = 0;
            }
        }
        Some(slot)
    }
}

/// Attempt NumPy's zero-copy C-order reshape across compatible contiguous dimension runs.
fn attempt_nocopy_reshape_c(
    old_shape: &[usize],
    old_strides: &[isize],
    new_shape: &[usize],
) -> Option<Vec<isize>> {
    if is_c_contiguous(old_shape, old_strides) {
        return Some(compute_c_strides(new_shape));
    }
    // Squeeze out size-1 dims from old layout to examine contiguous chunks
    let mut nz_shape = Vec::with_capacity(old_shape.len());
    let mut nz_strides = Vec::with_capacity(old_strides.len());
    for (&d, &st) in old_shape.iter().zip(old_strides.iter()) {
        if d != 1 {
            nz_shape.push(d);
            nz_strides.push(st);
        }
    }
    let mut new_strides = vec![0isize; new_shape.len()];
    let mut oi = 0usize;
    let mut oj;
    let mut ni = 0usize;
    let mut nj;
    while ni < new_shape.len() && oi < nz_shape.len() {
        if new_shape[ni] == 1 {
            new_strides[ni] = 0;
            ni += 1;
            continue;
        }
        let mut np = new_shape[ni];
        let mut op = nz_shape[oi];
        nj = ni + 1;
        oj = oi + 1;
        while np != op {
            if np < op {
                if nj >= new_shape.len() {
                    return None;
                }
                np *= new_shape[nj];
                nj += 1;
            } else {
                if oj >= nz_shape.len() {
                    return None;
                }
                op *= nz_shape[oj];
                oj += 1;
            }
        }
        // Verify old chunk [oi..oj) is contiguous
        for k in oi..oj - 1 {
            if nz_strides[k] != (nz_shape[k + 1] as isize) * nz_strides[k + 1] {
                return None;
            }
        }
        let mut cur_stride = nz_strides[oj - 1];
        for k in (ni..nj).rev() {
            new_strides[k] = cur_stride;
            cur_stride *= new_shape[k].max(1) as isize;
        }
        ni = nj;
        oi = oj;
    }
    if oi != nz_shape.len() {
        return None;
    }
    while ni < new_shape.len() {
        if new_shape[ni] != 1 {
            return None;
        }
        new_strides[ni] = 0;
        ni += 1;
    }
    Some(new_strides)
}

impl<T: Copy + PartialEq> PartialEq for NdArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        if let (Some(a), Some(b)) = (self.as_slice(), other.as_slice()) {
            return a == b;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for NdArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdArray")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("data", &self.to_vec())
            .finish()
    }
}

impl<T: Copy + fmt::Display + fmt::Debug> fmt::Display for NdArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ndim() == 0 {
            return write!(f, "{}", self.iter().next().unwrap());
        }
        let vals = self.to_vec();
        write!(f, "array({vals:?}, shape={:?})", self.shape)
    }
}

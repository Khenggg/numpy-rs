use crate::dtype::Scalar;
use crate::error::{NumPyError, Result};
use std::alloc::{self, Layout};
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::slice;

/// 64-byte alignment guarantees cache-line & AVX2/AVX-512 alignment.
pub const SIMD_ALIGN: usize = 64;

/// Contiguous heap buffer aligned to at least 64 bytes.
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T> AlignedBuffer<T> {
    /// Compute the 64-byte aligned allocation `Layout` for `len` elements of `T`,
    /// returning `Err(NumPyError::InvalidValue("array is too big"))` if the total
    /// byte count overflows `usize` or exceeds `isize::MAX - 64`.
    #[inline]
    pub fn try_layout_for(len: usize) -> Result<Option<Layout>> {
        if len == 0 || mem::size_of::<T>() == 0 {
            return Ok(None);
        }
        let align = SIMD_ALIGN.max(mem::align_of::<T>());
        let max_bytes = (isize::MAX as usize).saturating_sub(align.max(64));
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .filter(|&b| b <= max_bytes)
            .ok_or_else(|| NumPyError::InvalidValue("array is too big".into()))?;
        let layout = Layout::from_size_align(bytes, align)
            .map_err(|_| NumPyError::InvalidValue("array is too big".into()))?;
        Ok(Some(layout))
    }

    #[inline]
    fn layout_for(len: usize) -> Option<Layout> {
        Self::try_layout_for(len).expect("array is too big")
    }

    /// Validate that `shape` element count and byte span (even across empty dimensions)
    /// fit within `isize::MAX - SIMD_ALIGN`, returning the total element count.
    #[inline]
    pub fn check_shape_bytes<U>(shape: &[usize]) -> Result<usize> {
        let total = crate::shape::shape_size(shape)?;
        let nonzero_elems = shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d.max(1))
                .ok_or_else(|| NumPyError::InvalidValue("array is too big".into()))
        })?;
        let _ = AlignedBuffer::<U>::try_layout_for(nonzero_elems)?;
        Ok(total)
    }

    /// Allocate an empty zero-length buffer without hitting the allocator.
    #[inline]
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
        }
    }

    /// Fallibly allocate a buffer filled with zeros (via OS `alloc_zeroed`).
    pub fn try_new_zeroed(len: usize) -> Result<Self>
    where
        T: Scalar,
    {
        let Some(layout) = Self::try_layout_for(len)? else {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
            });
        };
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(raw as *mut T) },
            len,
        })
    }

    /// Allocate a buffer filled with zeros (via OS `alloc_zeroed`).
    pub fn new_zeroed(len: usize) -> Self
    where
        T: Scalar,
    {
        Self::try_new_zeroed(len).expect("array is too big")
    }

    /// Fallibly allocate an uninitialized 64-byte aligned buffer for safe write-once initialization.
    pub fn try_new_uninit(len: usize) -> Result<AlignedUninit<T>> {
        let Some(layout) = Self::try_layout_for(len)? else {
            return Ok(AlignedUninit {
                ptr: NonNull::dangling(),
                len,
            });
        };
        let raw = unsafe { alloc::alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Ok(AlignedUninit {
            ptr: unsafe { NonNull::new_unchecked(raw as *mut MaybeUninit<T>) },
            len,
        })
    }

    /// Allocate an uninitialized 64-byte aligned buffer for safe write-once initialization.
    pub fn new_uninit(len: usize) -> AlignedUninit<T> {
        Self::try_new_uninit(len).expect("array is too big")
    }

    /// Fallibly create an aligned buffer filled with `value`.
    pub fn try_filled(len: usize, value: T) -> Result<Self>
    where
        T: Copy,
    {
        let mut uninit = Self::try_new_uninit(len)?;
        for slot in uninit.as_mut_slice() {
            slot.write(value);
        }
        Ok(unsafe { uninit.assume_init() })
    }

    /// Create an aligned buffer filled with `value`.
    pub fn filled(len: usize, value: T) -> Self
    where
        T: Copy,
    {
        Self::try_filled(len, value).expect("array is too big")
    }

    /// Fallibly create an aligned buffer initialized from a closure `f(index)`.
    pub fn try_from_fn<F>(len: usize, mut f: F) -> Result<Self>
    where
        F: FnMut(usize) -> T,
    {
        let mut uninit = Self::try_new_uninit(len)?;
        for (i, slot) in uninit.as_mut_slice().iter_mut().enumerate() {
            slot.write(f(i));
        }
        Ok(unsafe { uninit.assume_init() })
    }

    /// Create an aligned buffer initialized from a closure `f(index)`.
    pub fn from_fn<F>(len: usize, f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self::try_from_fn(len, f).expect("array is too big")
    }

    /// Copy elements from an existing slice into a 64-byte aligned buffer.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        let uninit = Self::new_uninit(src.len());
        if !src.is_empty() && mem::size_of::<T>() > 0 {
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), uninit.ptr.as_ptr() as *mut T, src.len());
            }
        }
        unsafe { uninit.assume_init() }
    }

    /// Move elements from a `Vec<T>` into a 64-byte aligned buffer.
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len();
        if mem::size_of::<T>() == 0 {
            let _ = mem::ManuallyDrop::new(vec);
            return Self {
                ptr: NonNull::dangling(),
                len,
            };
        }
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
        let uninit = Self::new_uninit(len);
        let mut md = mem::ManuallyDrop::new(vec);
        unsafe {
            ptr::copy_nonoverlapping(md.as_ptr(), uninit.ptr.as_ptr() as *mut T, len);
            let cap = md.capacity();
            let ptr = md.as_mut_ptr();
            if cap > 0 {
                let layout = Layout::array::<T>(cap).unwrap();
                alloc::dealloc(ptr as *mut u8, layout);
            }
            uninit.assume_init()
        }
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
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T: Clone> Clone for AlignedBuffer<T> {
    fn clone(&self) -> Self {
        let mut uninit = Self::new_uninit(self.len);
        for (dst, src) in uninit.as_mut_slice().iter_mut().zip(self.as_slice().iter()) {
            dst.write(src.clone());
        }
        unsafe { uninit.assume_init() }
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if mem::needs_drop::<T>() && self.len > 0 {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
            }
        }
        if let Some(layout) = Self::layout_for(self.len) {
            unsafe {
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

/// Uninitialized 64-byte aligned buffer wrapper; safely frees memory if dropped before `assume_init`.
pub struct AlignedUninit<T> {
    ptr: NonNull<MaybeUninit<T>>,
    len: usize,
}

unsafe impl<T: Send> Send for AlignedUninit<T> {}
unsafe impl<T: Sync> Sync for AlignedUninit<T> {}

impl<T> AlignedUninit<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.ptr.as_ptr()
    }

    /// Convert into initialized `AlignedBuffer<T>`.
    ///
    /// # Safety
    /// Every element in `0..self.len()` must have been initialized.
    #[inline]
    pub unsafe fn assume_init(self) -> AlignedBuffer<T> {
        let md = mem::ManuallyDrop::new(self);
        AlignedBuffer {
            ptr: unsafe { NonNull::new_unchecked(md.ptr.as_ptr() as *mut T) },
            len: md.len,
        }
    }
}

impl<T> Drop for AlignedUninit<T> {
    fn drop(&mut self) {
        if let Some(layout) = AlignedBuffer::<T>::layout_for(self.len) {
            unsafe {
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

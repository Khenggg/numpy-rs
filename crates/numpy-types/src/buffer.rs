use crate::dtype::Scalar;
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
    #[inline]
    fn layout_for(len: usize) -> Option<Layout> {
        if len == 0 || mem::size_of::<T>() == 0 {
            return None;
        }
        let align = SIMD_ALIGN.max(mem::align_of::<T>());
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .expect("buffer byte length overflow");
        Some(Layout::from_size_align(bytes, align).expect("invalid aligned layout"))
    }

    /// Allocate an empty zero-length buffer without hitting the allocator.
    #[inline]
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
        }
    }

    /// Allocate a buffer filled with zeros (via OS `alloc_zeroed`).
    pub fn new_zeroed(len: usize) -> Self
    where
        T: Scalar,
    {
        let Some(layout) = Self::layout_for(len) else {
            return Self {
                ptr: NonNull::dangling(),
                len,
            };
        };
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self {
            ptr: unsafe { NonNull::new_unchecked(raw as *mut T) },
            len,
        }
    }

    /// Allocate an uninitialized 64-byte aligned buffer for safe write-once initialization.
    pub fn new_uninit(len: usize) -> AlignedUninit<T> {
        let Some(layout) = Self::layout_for(len) else {
            return AlignedUninit {
                ptr: NonNull::dangling(),
                len,
            };
        };
        let raw = unsafe { alloc::alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        AlignedUninit {
            ptr: unsafe { NonNull::new_unchecked(raw as *mut MaybeUninit<T>) },
            len,
        }
    }

    /// Create an aligned buffer filled with `value`.
    pub fn filled(len: usize, value: T) -> Self
    where
        T: Copy,
    {
        let mut uninit = Self::new_uninit(len);
        for slot in uninit.as_mut_slice() {
            slot.write(value);
        }
        unsafe { uninit.assume_init() }
    }

    /// Create an aligned buffer initialized from a closure `f(index)`.
    pub fn from_fn<F>(len: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        let mut uninit = Self::new_uninit(len);
        for (i, slot) in uninit.as_mut_slice().iter_mut().enumerate() {
            slot.write(f(i));
        }
        unsafe { uninit.assume_init() }
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
        if len == 0 || mem::size_of::<T>() == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len,
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

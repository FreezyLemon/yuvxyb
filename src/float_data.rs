use core::{alloc::Layout, ptr::NonNull, mem::size_of, ops::{Deref, DerefMut}};
use std::{ptr::copy_nonoverlapping, alloc::dealloc};

const ALIGNMENT: usize = 256;

#[derive(Debug)]
pub struct FloatData {
    ptr: NonNull<[f32; 3]>,
    len: usize,
}

impl FloatData {
    fn layout(len: usize) -> Layout {
        assert!(ALIGNMENT.is_power_of_two()); // includes check for > 0
        assert!(len > 0, "cannot create zero-length FloatData");

        Layout::from_size_align(len * size_of::<[f32; 3]>(), ALIGNMENT)
            .unwrap()
            .pad_to_align()
    }

    /// Creates a new FloatData from the provided slice.
    /// Currently allocates, but might be able to reuse memory in the future.
    pub fn from_data(data: &[[f32; 3]]) -> Self {
        let mut fd = Self::new(data.len());
        fd.copy_from_slice(data);

        fd
    }

    /// Allocates zero-initialized and padded.
    pub fn new(len: usize) -> Self {
        unsafe {
            // SAFETY: We're initializing the data on the next line.
            let d = Self::new_uninitialized(len);
            // SAFETY: Validity & alignment are guaranteed by new_unitialized.
            core::ptr::write_bytes(d.ptr.as_ptr(), 0, d.len);
            d
        }
    }

    /// Allocates padded, but possibly uninitialized.
    pub unsafe fn new_uninitialized(len: usize) -> Self {
        let ptr = {
            let layout = Self::layout(len);
            // SAFETY: size_of::<f32> is 4, and len > 0 is checked above.
            let raw_ptr = unsafe { std::alloc::alloc(layout) };

            match NonNull::new(raw_ptr as *mut _) {
                Some(p) => p,
                None => std::alloc::handle_alloc_error(layout),
            }
        };

        Self {
            ptr,
            len,
        }
    }
}

impl Deref for FloatData {
    type Target = [[f32; 3]];

    fn deref(&self) -> &Self::Target {
        // SAFETY: The data is valid (len >0) and mutations are disallowed during returned lifetime 'a
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for FloatData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: See deref above.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Clone for FloatData {
    fn clone(&self) -> Self {
        unsafe {
            let fd = Self::new_uninitialized(self.len);
            // SAFETY: Alignment and size are guaranteed by new_unitialized.
            // Two FloatDatas cannot own the same memory, so they're guaranteed to not overlap.
            copy_nonoverlapping(self.ptr.as_ptr(), fd.ptr.as_ptr(), fd.len);
            fd
        }
    }
}

impl Drop for FloatData {
    fn drop(&mut self) {
        // SAFETY: Self::layout prevents us from deallocating the wrong amount of memory
        unsafe { dealloc(self.ptr.as_ptr() as *mut _, Self::layout(self.len)) }
    }
}

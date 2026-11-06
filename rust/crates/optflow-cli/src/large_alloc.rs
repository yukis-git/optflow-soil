//! The binary's allocator: blocks of at least [`THRESHOLD`] bytes are mapped and unmapped directly,
//! everything else goes to the system allocator. macOS's malloc keeps freed large blocks in a cache
//! with their pages resident (`MallocLargeCache`), so the full-resolution maps of one stage stay in
//! the process while the next stage takes fresh pages for its own. Unmapping on free returns them at once.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::{c_int, c_void};
use std::ptr;

const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_PRIVATE: c_int = 0x0002;
const MAP_ANON: c_int = 0x1000;
const MAP_FAILED: *mut c_void = !0 as *mut c_void;
/// The smallest block that is mapped. The large blocks of a run are full-resolution maps, tile
/// packs and filter responses (70 MB and more at full resolution); 16 MiB keeps no less of them
/// resident but faults 2.7x as many pages (the iteration's 10 to 24 MB maps would be unmapped too).
const THRESHOLD: usize = 64 << 20;
/// The alignment every mapping has (the arm64 page; 4 KiB pages are also 16 KiB-aligned here).
const MAP_ALIGN: usize = 16 * 1024;

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

fn mapped(layout: Layout) -> bool {
    layout.size() >= THRESHOLD && layout.align() <= MAP_ALIGN
}

/// The global allocator of `optflow` (see the module).
pub struct LargeMmap;

unsafe impl GlobalAlloc for LargeMmap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !mapped(layout) {
            return System.alloc(layout);
        }
        let p = mmap(
            ptr::null_mut(),
            layout.size(),
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0,
        );
        if p == MAP_FAILED {
            ptr::null_mut()
        } else {
            p.cast()
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !mapped(layout) {
            return System.alloc_zeroed(layout);
        }
        // a fresh anonymous mapping is zero
        self.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if mapped(layout) {
            munmap(ptr.cast(), layout.size());
        } else {
            System.dealloc(ptr, layout);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        if !mapped(layout) && !mapped(new_layout) {
            return System.realloc(ptr, layout, new_size);
        }
        let new = self.alloc(new_layout);
        if !new.is_null() {
            ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::THRESHOLD;

    /// Blocks on both sides of the threshold: zeroed, kept across a growing and a shrinking
    /// reallocation that crosses it.
    #[test]
    fn blocks_across_the_threshold_keep_their_bytes() {
        let big = vec![0u8; THRESHOLD];
        assert!(big.iter().all(|&b| b == 0));
        let mut v: Vec<u32> = (0..(THRESHOLD / 4 - 1) as u32).collect();
        v.reserve_exact(16);
        assert!(v.capacity() * 4 >= THRESHOLD);
        v.push(7);
        assert!(v[..v.len() - 1]
            .iter()
            .enumerate()
            .all(|(i, &x)| x == i as u32));
        v.truncate(1000);
        v.shrink_to_fit();
        assert_eq!(v[999], 999);
    }
}

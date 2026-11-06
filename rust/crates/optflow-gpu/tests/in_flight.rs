#![cfg(target_os = "macos")]
//! A buffer queued on a batch is refused to the CPU until the batch has finished, whoever owns it
//! by then (the closure of `Gpu::batch` may move it away before the GPU runs).
use std::panic::{catch_unwind, AssertUnwindSafe};

use optflow_gpu::device::{Buffer, Gpu};

const N: usize = 256;

/// Queues `test_two_sum` over `N` elements with `out` as its output.
fn two_sum(gpu: &Gpu, b: &mut optflow_gpu::device::Batch, out: &mut Buffer) -> (Buffer, Buffer) {
    let mut a = gpu.buffer(4 * N).unwrap();
    let mut c = gpu.buffer(4 * N).unwrap();
    a.as_mut_slice::<f32>().fill(1.5);
    c.as_mut_slice::<f32>().fill(0.25);
    b.dispatch(
        "test_two_sum",
        &mut [&mut a, &mut c, out],
        &(N as u32).to_le_bytes(),
        (N, 1, 1),
        (64, 1, 1),
    )
    .unwrap();
    (a, c)
}

#[test]
fn a_dispatched_buffer_is_refused_until_the_batch_returns() {
    let gpu = Gpu::new().unwrap();
    let mut out = gpu.buffer(4 * 3 * N).unwrap();
    let (a, c) = gpu
        .batch(|b| {
            let (a, c) = two_sum(&gpu, b, &mut out);
            assert!(catch_unwind(AssertUnwindSafe(|| a.as_slice::<f32>()[0])).is_err());
            (a, c)
        })
        .unwrap();
    // after the batch: readable, and the kernel has run
    assert_eq!(a.as_slice::<f32>()[0], 1.5);
    assert_eq!(c.as_slice::<f32>()[0], 0.25);
    assert_eq!(out.as_slice::<f32>()[0], 1.75);
}

#[test]
fn a_dispatched_buffer_moved_to_another_thread_is_refused() {
    let gpu = Gpu::new().unwrap();
    let mut out = gpu.buffer(4 * 3 * N).unwrap();
    gpu.batch(|b| {
        let (a, _c) = two_sum(&gpu, b, &mut out);
        let read = std::thread::spawn(move || a.as_slice::<f32>()[0]).join();
        assert!(read.is_err(), "the other thread read a buffer in flight");
    })
    .unwrap();
}

#[test]
fn a_dispatched_buffer_released_in_the_batch_is_not_pooled() {
    let gpu = Gpu::new().unwrap();
    let mut out = gpu.buffer(4 * 3 * N).unwrap();
    let before = gpu.pooled_bytes();
    gpu.batch(|b| {
        let (a, c) = two_sum(&gpu, b, &mut out);
        gpu.release(a);
        gpu.release(c);
        assert_eq!(gpu.pooled_bytes(), before);
        // a new buffer of that size is a fresh one, not the dispatched one
        let fresh = gpu.buffer(4 * N).unwrap();
        assert!(fresh.as_slice::<f32>().iter().all(|&x| x == 0.0));
    })
    .unwrap();
    // an idle buffer is still pooled
    let idle = gpu.buffer(4 * N).unwrap();
    let cap_before = gpu.pooled_bytes();
    gpu.release(idle);
    assert!(gpu.pooled_bytes() > cap_before);
}

#[test]
fn a_panicking_batch_ends_the_marks() {
    let gpu = Gpu::new().unwrap();
    let mut out = gpu.buffer(4 * 3 * N).unwrap();
    let r = catch_unwind(AssertUnwindSafe(|| {
        gpu.batch(|b| {
            let _ = two_sum(&gpu, b, &mut out);
            panic!("in the batch");
        })
    }));
    assert!(r.is_err());
    // never committed: the output is still zero and readable
    assert!(out.as_slice::<f32>().iter().all(|&x| x == 0.0));
}

/// A submitted batch's buffers stay refused until its wait; other buffers are usable meanwhile.
#[test]
fn a_submitted_buffer_is_refused_until_the_wait() {
    let gpu = Gpu::new().unwrap();
    let mut out = gpu.buffer(4 * 3 * N).unwrap();
    let mut other = gpu.buffer(4 * N).unwrap();
    let ((a, _c), pending) = gpu.submit(|b| two_sum(&gpu, b, &mut out)).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| out.as_slice::<f32>()[0])).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| a.as_slice::<f32>()[0])).is_err());
    other.as_mut_slice::<f32>().fill(2.0);
    pending.wait().unwrap();
    assert_eq!(out.as_slice::<f32>()[0], 1.75);
    assert_eq!(a.as_slice::<f32>()[0], 1.5);
    assert_eq!(other.as_slice::<f32>()[0], 2.0);
}

/// Dropping a submitted batch without its wait waits for the GPU before ending the marks.
#[test]
fn dropping_a_submitted_batch_waits_for_it() {
    let gpu = Gpu::new().unwrap();
    let mut out = gpu.buffer(4 * 3 * N).unwrap();
    let (_, pending) = gpu.submit(|b| two_sum(&gpu, b, &mut out)).unwrap();
    drop(pending);
    assert_eq!(out.as_slice::<f32>()[0], 1.75);
}

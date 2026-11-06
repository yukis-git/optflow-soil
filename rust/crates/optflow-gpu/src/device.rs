//! The Metal device, the compiled shader library, a size-keyed pool of shared buffers and
//! batched dispatch (one command buffer, one wait).

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLResourceOptions, MTLSize,
};

use crate::GpuError;

/// All shaders as one source; the pragmas and the include live in `common.metal` only.
const SOURCE: &str = concat!(
    include_str!("../shaders/common.metal"),
    include_str!("../shaders/fft.metal"),
    include_str!("../shaders/spline.metal"),
    include_str!("../shaders/phase.metal"),
    include_str!("../shaders/moments.metal")
);

/// Metal's limit for `setBytes`.
const MAX_SET_BYTES: usize = 4096;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: Mutex<HashMap<&'static str, Pipeline>>,
    pool: Mutex<Pool>,
    /// Per-kernel GPU time (`OPTFLOW_GPU_KERNELS=1`): every dispatch then runs in a command
    /// buffer of its own, and its GPU time is added to its kernel's `(seconds, calls)`.
    kernel_times: Option<Mutex<HashMap<&'static str, (f64, usize)>>>,
}

type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Free buffers by allocated size, and their total.
#[derive(Default)]
struct Pool {
    free: BTreeMap<usize, Vec<Buffer>>,
    bytes: usize,
}

/// The most the pool keeps for reuse; a buffer released above it is freed.
const POOL_CAP: usize = 2 << 30;

/// A request takes a free buffer up to this many times its size class (the smallest one that
/// fits), so that a window of a new shape reuses the buffers of an earlier, larger one instead
/// of allocating its own.
const REUSE_RATIO: usize = 2;

/// The allocated size for a request: the next power of two up to 64 MiB, then multiples of
/// 16 MiB, so that windows of slightly different sizes share buffers.
fn size_class(bytes: usize) -> usize {
    const MIB: usize = 1 << 20;
    if bytes <= 64 * MIB {
        bytes.max(256).next_power_of_two()
    } else {
        bytes.div_ceil(16 * MIB) * 16 * MIB
    }
}

// SAFETY: MTLDevice, MTLCommandQueue, MTLLibrary and MTLComputePipelineState are documented by
// Apple as thread-safe; the caches are behind mutexes. Command buffers and encoders are created
// and used inside one call (`batch`) and never shared.
unsafe impl Send for Gpu {}
// SAFETY: see above.
unsafe impl Sync for Gpu {}

/// A zero-initialised shared-storage buffer. CPU access goes through `as_slice`/`as_mut_slice`.
///
/// A buffer queued on a [`Batch`] is *in flight* until that batch's `Gpu::batch` call has
/// returned (its command buffers completed, or never committed). The CPU accessors panic on a
/// buffer in flight, and [`Gpu::release`] frees it instead of pooling it: the closure of
/// `Gpu::batch` owns what it captures and may move a dispatched buffer elsewhere (another
/// thread, the pool) before the GPU has run, so the borrow checker alone cannot keep CPU and GPU
/// access apart.
pub struct Buffer {
    inner: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// The requested size: all that the accessors expose.
    len: usize,
    /// The allocated size (the size class of `len`).
    cap: usize,
    /// The number of dispatches of unfinished batches that use the buffer; shared with those
    /// batches, which decrement it once their GPU work can no longer run.
    in_flight: Arc<AtomicUsize>,
}

// SAFETY: an MTLBuffer is a thread-safe object. CPU access to its contents goes through `bytes`
// and `bytes_mut`, which refuse a buffer in flight (see there), so it never overlaps GPU work.
unsafe impl Send for Buffer {}
// SAFETY: see above (shared access only reads).
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether a batch that has not finished uses the buffer.
    fn is_in_flight(&self) -> bool {
        // Acquire: pairs with the Release decrement made after the GPU work completed.
        self.in_flight.load(Ordering::Acquire) != 0
    }

    fn assert_idle(&self) {
        assert!(
            !self.is_in_flight(),
            "CPU access to a GPU buffer queued on a batch that has not finished"
        );
    }

    fn bytes(&self) -> &[u8] {
        self.assert_idle();
        // SAFETY: `contents()` of a shared-storage buffer points to at least `len` readable bytes
        // that live as long as `inner`. GPU work on the buffer is only queued by
        // `Batch::dispatch`, which takes it as `&mut Buffer` (so no slice of it is alive then)
        // and marks it in flight until its command buffer has completed or can no longer be
        // committed. The buffer is not in flight now, so no GPU work on it is pending or
        // executing; and none can be queued while this slice lives, since that needs `&mut`.
        unsafe { std::slice::from_raw_parts(self.inner.contents().as_ptr().cast::<u8>(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.assert_idle();
        // SAFETY: as in `bytes`; `&mut self` makes this the only access, CPU or GPU, to the
        // memory while the slice lives.
        unsafe {
            std::slice::from_raw_parts_mut(self.inner.contents().as_ptr().cast::<u8>(), self.len)
        }
    }

    /// The contents as `T`s. Panics if the length is not a multiple of `size_of::<T>()`.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(self.bytes())
    }

    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(self.bytes_mut())
    }
}

impl Gpu {
    pub fn new() -> Result<Gpu, GpuError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| GpuError::Command("cannot create a command queue".into()))?;
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        opts.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&opts))
            .map_err(|e| GpuError::Compile(e.localizedDescription().to_string()))?;
        Ok(Gpu {
            device,
            queue,
            library,
            pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(Pool::default()),
            kernel_times: std::env::var("OPTFLOW_GPU_KERNELS")
                .is_ok_and(|v| v == "1")
                .then(|| Mutex::new(HashMap::new())),
        })
    }

    /// A zeroed shared buffer of `bytes` bytes (allocated in a size class), from the pool when
    /// a free one of that class, or the smallest free one up to `REUSE_RATIO` times it, exists.
    /// A reused buffer is longer than its `len`; only the first `len` bytes are zeroed or visible.
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, GpuError> {
        if bytes as u64 > self.max_buffer_length() {
            return Err(GpuError::Command(format!(
                "cannot allocate a {bytes}-byte buffer (device limit {})",
                self.max_buffer_length()
            )));
        }
        let cap = size_class(bytes).min(self.max_buffer_length() as usize);
        let pooled = {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let key = pool
                .free
                .range(cap..=cap.saturating_mul(REUSE_RATIO))
                .find(|(_, v)| !v.is_empty())
                .map(|(&k, _)| k);
            let b = key.and_then(|k| {
                let v = pool.free.get_mut(&k)?;
                let b = v.pop();
                if v.is_empty() {
                    pool.free.remove(&k);
                }
                b
            });
            if let Some(b) = &b {
                pool.bytes -= b.cap;
            }
            b
        };
        if let Some(mut b) = pooled {
            b.len = bytes;
            b.bytes_mut().fill(0);
            return Ok(b);
        }
        let inner = self
            .device
            .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                GpuError::Command(format!("Metal could not allocate a {cap}-byte buffer"))
            })?;
        // a fresh shared buffer is zero-filled by Metal
        Ok(Buffer {
            inner,
            len: bytes,
            cap,
            in_flight: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Hand a buffer back to the pool; it is freed instead when the pool would hold more than
    /// `POOL_CAP` bytes, or when it is in flight (queued on a batch that has not finished: the
    /// command buffer keeps the memory alive until the GPU is done with it).
    pub fn release(&self, b: Buffer) {
        if b.is_in_flight() {
            return;
        }
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.bytes + b.cap > POOL_CAP {
            return;
        }
        pool.bytes += b.cap;
        pool.free.entry(b.cap).or_default().push(b);
    }

    /// Free every buffer in the pool. Buffers in flight were never pooled (`release` drops them),
    /// and one a caller still holds is not in the pool, so nothing in use is freed.
    pub fn trim_pool(&self) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.free.clear();
        pool.bytes = 0;
    }

    /// Bytes held by free buffers in the pool.
    #[doc(hidden)]
    pub fn pooled_bytes(&self) -> usize {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }

    fn pipeline(&self, kernel: &'static str) -> Result<Pipeline, GpuError> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(kernel) {
            return Ok(p.clone());
        }
        let f = self
            .library
            .newFunctionWithName(&NSString::from_str(kernel))
            .ok_or_else(|| GpuError::Command(format!("no kernel named {kernel}")))?;
        let p = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| {
                GpuError::Command(format!(
                    "pipeline for {kernel}: {}",
                    e.localizedDescription()
                ))
            })?;
        cache.insert(kernel, p.clone());
        Ok(p)
    }

    /// The most threads one threadgroup of `kernel` may have (the pipeline's
    /// `maxTotalThreadsPerThreadgroup`, which Metal lowers for register-heavy kernels).
    pub fn max_threads(&self, kernel: &'static str) -> Result<usize, GpuError> {
        Ok(self.pipeline(kernel)?.maxTotalThreadsPerThreadgroup())
    }

    /// One dispatch in its own command buffer; returns when it has completed.
    /// `params` are passed with `setBytes` at the buffer index after the last buffer.
    pub fn dispatch(
        &self,
        kernel: &'static str,
        buffers: &mut [&mut Buffer],
        params: &[u8],
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) -> Result<(), GpuError> {
        self.batch(|b| b.dispatch(kernel, buffers, params, grid, group))?
    }

    /// Runs `f`, which queues dispatches on one command buffer; commits it and waits. Errors of
    /// the command buffer become `GpuError::Command`.
    pub fn batch<R>(&self, f: impl FnOnce(&mut Batch) -> R) -> Result<R, GpuError> {
        Ok(self.batch_timed(f)?.0)
    }

    /// As [`Gpu::batch`], also returning the GPU time of the work in seconds (summed over the
    /// command buffers when per-kernel timing splits it).
    pub fn batch_timed<R>(&self, f: impl FnOnce(&mut Batch) -> R) -> Result<(R, f64), GpuError> {
        let (r, pending) = self.submit(f)?;
        Ok((r, pending.wait()?))
    }

    /// As [`Gpu::batch_timed`], but returns once the command buffer is committed: the wait (and
    /// its errors) come with [`Pending::wait`]. The buffers it dispatched stay in flight until
    /// then, so the CPU may meanwhile use any other buffer. Command buffers run in the order they
    /// are committed, and Metal orders their accesses to the same buffer.
    pub fn submit<R>(&self, f: impl FnOnce(&mut Batch) -> R) -> Result<(R, Pending), GpuError> {
        let (cmd, enc) = self.command()?;
        let mut batch = Batch {
            gpu: self,
            cmd,
            enc,
            seconds: 0.0,
            failed: None,
            in_flight: InFlight(Vec::new()),
        };
        // If `f` panics, the encoder is ended (Metal aborts the process on an encoder dropped
        // unended) and `batch` dropped with its command buffer never committed: dropping
        // `in_flight` then ends the marks.
        let r = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut batch))) {
            Ok(r) => r,
            Err(panic) => {
                batch.enc.endEncoding();
                drop(batch);
                std::panic::resume_unwind(panic)
            }
        };
        let Batch {
            cmd,
            enc,
            seconds,
            failed,
            in_flight,
            ..
        } = batch;
        enc.endEncoding();
        cmd.commit();
        Ok((
            r,
            Pending {
                cmd,
                waited: false,
                seconds,
                failed,
                in_flight,
            },
        ))
    }

    fn command(&self) -> Result<(CommandBuffer, Encoder), GpuError> {
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| GpuError::Command("cannot create a command buffer".into()))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| GpuError::Command("cannot create a compute encoder".into()))?;
        Ok((cmd, enc))
    }

    /// The per-kernel GPU times since the last call, `(kernel, seconds, calls)` sorted by
    /// time (empty unless `OPTFLOW_GPU_KERNELS=1`).
    #[doc(hidden)]
    pub fn take_kernel_times(&self) -> Vec<(&'static str, f64, usize)> {
        let Some(t) = &self.kernel_times else {
            return Vec::new();
        };
        let mut t = t.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = t.drain().map(|(k, (s, c))| (k, s, c)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Bytes the device has allocated to this process (Metal's `currentAllocatedSize`).
    #[doc(hidden)]
    pub fn allocated_bytes(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    pub fn max_buffer_length(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    pub fn recommended_working_set(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }
}

/// Commits `cmd`, waits for it and returns its GPU time in seconds.
fn finish(cmd: &CommandBuffer) -> Result<f64, GpuError> {
    cmd.commit();
    completed(cmd)
}

/// Waits for the committed `cmd` and returns its GPU time in seconds.
fn completed(cmd: &CommandBuffer) -> Result<f64, GpuError> {
    cmd.waitUntilCompleted();
    if cmd.status() != MTLCommandBufferStatus::Completed {
        let why = cmd
            .error()
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_else(|| format!("status {:?}", cmd.status()));
        return Err(GpuError::Command(why));
    }
    Ok(cmd.GPUEndTime() - cmd.GPUStartTime())
}

/// A committed command buffer of [`Gpu::submit`] not yet waited for. Dropping it waits too, before
/// its in-flight marks end (the fields drop after `drop`), so the marks never end while the GPU
/// may still run it.
pub struct Pending {
    cmd: CommandBuffer,
    waited: bool,
    /// GPU time of the command buffers already completed (per-kernel timing).
    seconds: f64,
    /// The first error of a completed command buffer (per-kernel timing).
    failed: Option<GpuError>,
    in_flight: InFlight,
}

impl Pending {
    /// Waits for the command buffer; returns the GPU time of the work in seconds, or the error of
    /// the command buffer, as [`Gpu::batch_timed`]. The buffers are no longer in flight after it.
    pub fn wait(mut self) -> Result<f64, GpuError> {
        let done = completed(&self.cmd);
        self.waited = true;
        // no GPU work of this batch can run any more: end the marks
        drop(std::mem::replace(&mut self.in_flight, InFlight(Vec::new())));
        let seconds = self.seconds + done?;
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        Ok(seconds)
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if !self.waited {
            self.cmd.waitUntilCompleted();
        }
    }
}

/// The in-flight marks a batch has put on buffers; dropping it removes them. It must be dropped
/// only when no GPU work of the batch can run any more (completed, or never committed).
struct InFlight(Vec<Arc<AtomicUsize>>);

impl Drop for InFlight {
    fn drop(&mut self) {
        for c in &self.0 {
            // Release: the CPU access that sees the count drop happens after the GPU work.
            c.fetch_sub(1, Ordering::Release);
        }
    }
}

/// The dispatches of one command buffer (see [`Gpu::batch`]). They run in order; Metal tracks
/// the hazards between them on the buffers.
pub struct Batch<'a> {
    gpu: &'a Gpu,
    cmd: CommandBuffer,
    enc: Encoder,
    /// GPU time of the command buffers already completed (per-kernel timing).
    seconds: f64,
    /// The first error of a completed command buffer (per-kernel timing).
    failed: Option<GpuError>,
    /// The buffers this batch has dispatched (one mark per dispatch and buffer).
    in_flight: InFlight,
}

impl Batch<'_> {
    /// The device the batch runs on.
    pub fn gpu(&self) -> &Gpu {
        self.gpu
    }

    /// Queue `kernel` over `grid` threads (not threadgroups) with threadgroups of `group`.
    pub fn dispatch(
        &mut self,
        kernel: &'static str,
        buffers: &mut [&mut Buffer],
        params: &[u8],
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) -> Result<(), GpuError> {
        if params.len() > MAX_SET_BYTES {
            return Err(GpuError::Command(format!(
                "{kernel}: {} parameter bytes exceed the {MAX_SET_BYTES}-byte setBytes limit",
                params.len()
            )));
        }
        let pso = self.gpu.pipeline(kernel)?;
        let threads = group.0 * group.1 * group.2;
        if threads == 0 || threads > pso.maxTotalThreadsPerThreadgroup() {
            return Err(GpuError::Command(format!(
                "{kernel}: threadgroup of {threads} threads (limit {})",
                pso.maxTotalThreadsPerThreadgroup()
            )));
        }
        self.enc.setComputePipelineState(&pso);
        for (i, b) in buffers.iter_mut().enumerate() {
            // Mark the buffer in flight before it is bound: from here until the batch's command
            // buffer has completed (or is dropped uncommitted) its CPU accessors refuse it,
            // whoever owns it by then.
            b.in_flight.fetch_add(1, Ordering::AcqRel);
            self.in_flight.0.push(Arc::clone(&b.in_flight));
            // SAFETY: `b.inner` is a live MTLBuffer and offset 0 is inside it (the command
            // buffer retains it, so it stays alive even if the `Buffer` is dropped before the GPU
            // runs). No CPU slice of it is alive now (`&mut Buffer`), and none can be made until
            // the mark above is removed, which happens only after the GPU work using it has
            // completed or can no longer be committed (`Gpu::batch_timed`, `InFlight`).
            unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.inner), 0, i) };
        }
        if !params.is_empty() {
            let ptr = NonNull::new(params.as_ptr().cast_mut().cast::<c_void>())
                .expect("a slice pointer is never null");
            // SAFETY: `ptr` points to `params.len()` readable bytes; Metal copies them during
            // the call.
            unsafe {
                self.enc
                    .setBytes_length_atIndex(ptr, params.len(), buffers.len())
            };
        }
        self.enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: grid.0,
                height: grid.1,
                depth: grid.2,
            },
            MTLSize {
                width: group.0,
                height: group.1,
                depth: group.2,
            },
        );
        if let Some(times) = &self.gpu.kernel_times {
            // per-kernel timing: complete this dispatch on its own and start a new buffer
            let (cmd, enc) = self.gpu.command()?;
            let old = std::mem::replace(&mut self.cmd, cmd);
            std::mem::replace(&mut self.enc, enc).endEncoding();
            match finish(&old) {
                Ok(s) => {
                    self.seconds += s;
                    let mut t = times.lock().unwrap_or_else(|e| e.into_inner());
                    let e = t.entry(kernel).or_insert((0.0, 0));
                    e.0 += s;
                    e.1 += 1;
                }
                Err(e) => {
                    let msg = e.to_string();
                    self.failed.get_or_insert(e);
                    return Err(GpuError::Command(msg));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A released larger buffer serves a smaller request within `REUSE_RATIO`, not beyond it.
    #[test]
    fn a_larger_free_buffer_is_reused_within_the_ratio_only() {
        let gpu = Gpu::new().unwrap();
        let big = 1usize << 20;
        gpu.release(gpu.buffer(big).unwrap());
        assert_eq!(gpu.pooled_bytes(), big);
        // within the ratio: the 1 MiB buffer is taken; its `len` is the request, zero-filled
        let within = big / REUSE_RATIO;
        let b = gpu.buffer(within).unwrap();
        assert_eq!(gpu.pooled_bytes(), 0, "the larger buffer was reused");
        assert_eq!((b.len, b.cap), (within, big));
        assert!(b.bytes().iter().all(|&x| x == 0));
        gpu.release(b);
        assert_eq!(gpu.pooled_bytes(), big);
        // beyond the ratio: a fresh buffer, the pooled one stays
        if REUSE_RATIO < 4096 {
            let beyond = big / (2 * REUSE_RATIO);
            let b = gpu.buffer(beyond).unwrap();
            assert_eq!(gpu.pooled_bytes(), big, "the pooled buffer is too big");
            assert_eq!(b.cap, size_class(beyond));
        }
    }

    /// The smallest free buffer that fits is the one taken.
    #[test]
    fn the_smallest_fitting_free_buffer_is_taken() {
        let gpu = Gpu::new().unwrap();
        let (a, b) = (gpu.buffer(1 << 20).unwrap(), gpu.buffer(2 << 20).unwrap());
        gpu.release(b);
        gpu.release(a);
        let got = gpu.buffer(1 << 19).unwrap();
        assert_eq!(got.cap, 1 << 20);
        assert_eq!(gpu.pooled_bytes(), 2 << 20);
    }
}

//! `GpuEngine`: the fine-stage iterations of one window on Metal, in the order of
//! `optflow_core::gpu_model::refine_raw_model` (GPU spec §3.3).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Instant;

use optflow_core::config::Config;
use optflow_core::engine::{EngineError, FineEngine, Prepared};
use optflow_core::fft::Fft2;
use optflow_core::gabor::{gabor_bank, Bank};
use optflow_core::gpu_model::df64::Df64;
use optflow_core::gpu_support::{
    low_confidence, median, plan_moments, ref_pack, tile_margin, untrusted_fill, Aperture,
    LowFreeze,
};
use optflow_core::image::Image;
use optflow_core::iter_trace;
use optflow_core::phaseflow::RawFlow;
use optflow_core::prof::{self, Stage};
use rayon::prelude::*;

use crate::device::{Batch, Buffer, Gpu, Pending};
use crate::fft::Fft2Plan;
use crate::kernels::{self, MomentPlanGpu, PAD};
use crate::GpuError;

/// Filters transformed together (the planes of one group are the working set of the FFTs).
const GROUP: usize = 4;
/// Entries of the bank cache; a third key clears it.
const CACHE_ENTRIES: usize = 2;
/// The most windows refined at once (each lane holds one window's buffers and its own banks).
const MAX_LANES: usize = 2;

type Key = (usize, usize, Vec<u64>, u64, u64, u64, u64);

/// The bank of one padded size on the device, for the responses.
struct Entry {
    dev: DevBank,
}

/// The bank of one padded size on the CPU with its transform, for the reference pack (which the
/// CPU computes, rustfft, so that it is the CPU's bit for bit). Shared: `prepare_extra` makes the
/// pack of the next tile with it on another thread.
struct CpuBank {
    bank: Bank,
    fft: Fft2,
}

/// [`FineEngine::prepare_extra`]'s data: the reference pack of the tile's `i1` and the bank it was
/// made with (which also gives the device bank if that is not cached).
struct Prefetched {
    pack: Vec<f32>,
    bank: Arc<CpuBank>,
    /// the bank's key (padded size and filter parameters), checked against the window's
    key: Key,
}

struct DevBank {
    plan: Fft2Plan,
    /// The filters, `GROUP` of them per buffer (`np` values each).
    groups: Vec<Buffer>,
    filters: usize,
}

#[derive(Default)]
struct State {
    cache: HashMap<Key, Entry>,
}

/// The fine stage on the default Metal device. Up to [`FineEngine::lanes`] windows at a time
/// (`OPTFLOW_GPU_LANES`, 1 or 2, default 2): each lane's state (its device banks) is behind a
/// mutex of its own, and the lanes share the device, its queue and the buffer pool. `refine_raw`
/// must not be called from more threads at once, e.g. from rayon workers; a call made while every
/// lane is running fails with an [`EngineError`] instead of waiting (the lock is held across
/// rayon work, so waiting from a rayon worker could deadlock).
pub struct GpuEngine {
    gpu: Gpu,
    /// One per lane.
    states: Vec<Mutex<State>>,
    /// The CPU banks by key (`CACHE_ENTRIES`, a third key clears it). Its own lock, held only to
    /// look up or insert (never across rayon work), so `prepare_extra` never touches `states`.
    cpu_banks: Mutex<HashMap<Key, Arc<CpuBank>>>,
    /// Reference packs made inside `refine` / taken from `prepare_extra` (for the tests).
    packs: [AtomicUsize; 2],
    profile: bool,
    /// GPU seconds and command buffers per stage (profile mode).
    gpu_stages: Mutex<HashMap<&'static str, (f64, usize)>>,
}

impl GpuEngine {
    /// The engine with the lanes of `OPTFLOW_GPU_LANES` (read here; 1 or 2, default 2).
    pub fn new() -> Result<GpuEngine, GpuError> {
        let lanes = std::env::var("OPTFLOW_GPU_LANES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MAX_LANES);
        GpuEngine::with_lanes(lanes)
    }

    /// The engine with `lanes` windows at a time (clamped to 1..=2).
    pub fn with_lanes(lanes: usize) -> Result<GpuEngine, GpuError> {
        Ok(GpuEngine {
            gpu: Gpu::new()?,
            states: (0..lanes.clamp(1, MAX_LANES))
                .map(|_| Mutex::new(State::default()))
                .collect(),
            cpu_banks: Mutex::new(HashMap::new()),
            packs: [AtomicUsize::new(0), AtomicUsize::new(0)],
            profile: std::env::var("OPTFLOW_GPU_PROFILE").is_ok_and(|v| v == "1"),
            gpu_stages: Mutex::new(HashMap::new()),
        })
    }

    /// GPU time per stage since the last call, `(stage, seconds, command buffers)` (filled in
    /// profile mode, `OPTFLOW_GPU_PROFILE=1`).
    #[doc(hidden)]
    pub fn take_stage_times(&self) -> Vec<(&'static str, f64, usize)> {
        let mut t = self.gpu_stages.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = t.drain().map(|(k, (s, c))| (k, s, c)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// GPU time per kernel since the last call (filled with `OPTFLOW_GPU_KERNELS=1`).
    #[doc(hidden)]
    pub fn take_kernel_times(&self) -> Vec<(&'static str, f64, usize)> {
        self.gpu.take_kernel_times()
    }

    fn book(&self, stage: Stage, seconds: f64) {
        let mut t = self.gpu_stages.lock().unwrap_or_else(|e| e.into_inner());
        let e = t.entry(stage.name()).or_insert((0.0, 0));
        e.0 += seconds;
        e.1 += 1;
    }

    /// Bytes held by free buffers in the pool (for the tests).
    #[doc(hidden)]
    pub fn pooled_bytes(&self) -> usize {
        self.gpu.pooled_bytes()
    }

    /// Bytes the device has allocated to this process (for the benchmarks).
    #[doc(hidden)]
    pub fn allocated_bytes(&self) -> u64 {
        self.gpu.allocated_bytes()
    }

    pub fn device_name(&self) -> String {
        crate::device_name().unwrap_or_else(|e| e.to_string())
    }

    /// Reference packs since the engine was made: `(made inside a refinement, prepared ahead by
    /// prepare_extra and used)` (for the tests).
    #[doc(hidden)]
    pub fn reference_packs(&self) -> (usize, usize) {
        (
            self.packs[0].load(Ordering::Relaxed),
            self.packs[1].load(Ordering::Relaxed),
        )
    }

    /// The CPU bank of a padded size, from the cache or made (outside the lock; when two threads
    /// make the same one, the first inserted is kept, and both are the same bits). The build is
    /// timed here, as `Reference`, and only here: no caller wraps this in a `Reference` timer.
    fn cpu_bank(&self, ph: usize, pw: usize, cfg: &Config) -> Arc<CpuBank> {
        let k = key(ph, pw, cfg);
        let lock = || self.cpu_banks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = lock().get(&k) {
            return Arc::clone(b);
        }
        let made = Arc::new(prof::time(Stage::Reference, || CpuBank {
            bank: gabor_bank(ph, pw, cfg),
            fft: Fft2::new(ph, pw),
        }));
        let mut cache = lock();
        if let Some(b) = cache.get(&k) {
            return Arc::clone(b);
        }
        if cache.len() >= CACHE_ENTRIES {
            cache.clear();
        }
        cache.insert(k, Arc::clone(&made));
        made
    }
}

fn key(ph: usize, pw: usize, cfg: &Config) -> Key {
    (
        ph,
        pw,
        cfg.scales.iter().map(|s| s.to_bits()).collect(),
        cfg.n_orient as u64,
        cfg.lam.to_bits(),
        cfg.bandwidth.to_bits(),
        cfg.dc_cut.to_bits(),
    )
}

fn upload<T: bytemuck::Pod>(gpu: &Gpu, v: &[T]) -> Result<Buffer, GpuError> {
    let mut b = gpu.buffer(std::mem::size_of_val(v))?;
    b.as_mut_slice::<T>().copy_from_slice(v);
    Ok(b)
}

fn make_entry(gpu: &Gpu, ph: usize, pw: usize, bank: &Bank) -> Result<Entry, GpuError> {
    let groups = bank
        .filters
        .chunks(GROUP)
        .map(|g| {
            let mut b = gpu.buffer(4 * ph * pw * g.len())?;
            for (dst, f) in b.as_mut_slice::<f32>().chunks_mut(ph * pw).zip(g) {
                dst.copy_from_slice(f);
            }
            Ok(b)
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
    let dev = DevBank {
        plan: Fft2Plan::new(gpu, ph, pw)?,
        groups,
        filters: bank.filters.len(),
    };
    Ok(Entry { dev })
}

/// CPU-side conversion buffers, allocated on first use and reused by every pre-warp / reset check.
#[derive(Default)]
struct Conv {
    energy: Vec<f64>,
    resid: Vec<f64>,
}

/// `dst[i] = f64(src[i])`, element-wise in parallel.
fn widen(dst: &mut Vec<f64>, src: &[f32]) {
    dst.resize(src.len(), 0.0);
    dst.par_iter_mut()
        .zip(src.par_iter())
        .for_each(|(d, &s)| *d = f64::from(s));
}

/// The buffers of one window.
struct Work<'a> {
    gpu: &'a Gpu,
    e: &'a mut DevBank,
    w: usize,
    h: usize,
    w2: usize,
    h2: usize,
    shift: (i64, i64),
    f: usize,
    // images and their spectra
    img2: Buffer,
    coeffs2: Buffer,
    scratch_n: Buffer,
    spec: Buffer,
    warped: Buffer,
    // reference and responses
    refpack: Buffer,
    responses: Buffer,
    ga: Buffer,
    gscratch: Buffer,
    // the field
    sx: Buffer,
    sy: Buffer,
    csx: Buffer,
    csy: Buffer,
    mid: Buffer,
    taps_s: Buffer,
    // the solve
    sums: Buffer,
    global: Buffer,
    mtaps: Buffer,
    r: usize,
    plan: MomentPlanGpu,
    u: [Buffer; 4],
    u_next: [Buffer; 4],
    energy: Buffer,
    resid: Buffer,
    slope: Buffer,
    erange: Buffer,
    /// The aperture prototypes of the solve (off by default).
    aperture: kernels::SolveAperture,
    conv: Conv,
}

impl Work<'_> {
    fn np(&self) -> usize {
        (self.w + 2 * PAD) * (self.h + 2 * PAD)
    }

    /// The coefficients of image 2.
    fn coeffs(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        kernels::prefilter_real(
            b,
            &mut self.img2,
            &mut self.scratch_n,
            &mut self.coeffs2,
            self.w2,
            self.h2,
        )
    }

    /// `sx`, `sy` from the displacement (first outer iteration, and later ones when the pre-warp
    /// rule fills nothing).
    fn smooth(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let (w, h) = (self.w, self.h);
        let [uxh, uxl, uyh, uyl] = &mut self.u;
        Self::blur(
            b,
            [uxh, uxl, uyh, uyl],
            &mut self.taps_s,
            &mut self.mid,
            [&mut self.sx, &mut self.sy],
            w,
            h,
        )
    }

    /// `sx`, `sy` from the field the pre-warp rule filled into `u_next` (`prewarp_fill`).
    fn smooth_filled(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let (w, h) = (self.w, self.h);
        let [uxh, uxl, uyh, uyl] = &mut self.u_next;
        Self::blur(
            b,
            [uxh, uxl, uyh, uyl],
            &mut self.taps_s,
            &mut self.mid,
            [&mut self.sx, &mut self.sy],
            w,
            h,
        )
    }

    fn blur(
        b: &mut Batch,
        [uxh, uxl, uyh, uyl]: [&mut Buffer; 4],
        taps_s: &mut Buffer,
        mid: &mut Buffer,
        [sx, sy]: [&mut Buffer; 2],
        w: usize,
        h: usize,
    ) -> Result<(), GpuError> {
        kernels::blur_field(b, uxh, uxl, taps_s, mid, sx, w, h)?;
        kernels::blur_field(b, uyh, uyl, taps_s, mid, sy, w, h)
    }

    /// Spline of the field and the warp of image 2.
    fn warp(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let (w, h) = (self.w, self.h);
        kernels::prefilter_field(b, &mut self.sx, &mut self.csx, w, h)?;
        kernels::prefilter_field(b, &mut self.sy, &mut self.csy, w, h)?;
        kernels::warp(
            b,
            &mut self.coeffs2,
            (self.w2, self.h2),
            &mut self.sx,
            &mut self.sy,
            (w, h),
            self.shift,
            &mut self.warped,
        )
    }

    /// Responses of the warped image: pad, FFT, filters, inverse and scale (one fused
    /// transform), spline, node layout.
    fn responses(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let (w, h, f) = (self.w, self.h, self.f);
        let (pw, ph, np) = (w + 2 * PAD, h + 2 * PAD, self.np());
        kernels::pad_reflect_complex(b, &mut self.warped, w, h, &mut self.spec)?;
        self.e.plan.forward(b, &mut self.spec, 1)?;
        // the same bits as filter_multiply, inverse and response_splines' `x * (1 / np)`
        let s = 1.0 / (np as f32);
        for (g, filt) in self.e.groups.iter_mut().enumerate() {
            let cnt = (f - g * GROUP).min(GROUP);
            self.e
                .plan
                .inverse_filtered(b, &mut self.spec, filt, np, cnt, &mut self.ga, s)?;
            kernels::response_splines(
                b,
                &mut self.ga,
                &mut self.gscratch,
                (pw, ph),
                cnt,
                1.0,
                (g * GROUP, f),
                &mut self.responses,
            )?;
        }
        Ok(())
    }

    fn accumulate(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let [uxh, uxl, uyh, uyl] = &mut self.u;
        kernels::accumulate(
            b,
            &mut self.refpack,
            &mut self.responses,
            self.f,
            &mut self.sx,
            &mut self.sy,
            &mut self.csx,
            &mut self.csy,
            uxh,
            uxl,
            uyh,
            uyl,
            self.w,
            self.h,
            &mut self.sums,
        )
    }

    /// Global window moments of the sums.
    fn moments(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let (w, h) = (self.w, self.h);
        kernels::moments_global(
            b,
            &mut self.sums,
            w,
            h,
            &mut self.mtaps,
            self.r,
            &mut self.plan,
            &mut self.global,
        )
    }

    /// The block solves (before the median reset).
    fn solve(&mut self, b: &mut Batch) -> Result<(), GpuError> {
        let (w, h) = (self.w, self.h);
        let [a0, a1, a2, a3] = &mut self.u;
        let [o0, o1, o2, o3] = &mut self.u_next;
        kernels::solve_blocks_ap(
            b,
            &mut self.sums,
            &mut self.global,
            [a0, a1, a2, a3],
            w,
            h,
            &mut self.mtaps,
            self.r,
            &mut self.plan,
            [o0, o1, o2, o3],
            (
                &mut self.energy,
                &mut self.resid,
                &mut self.slope,
                &mut self.erange,
            ),
            self.aperture,
        )
    }

    /// The reset of the pixels below the energy threshold, when the block ranges say the
    /// threshold can matter (the median is taken on the CPU).
    fn reset(&mut self, threshold: f64, b: &mut Batch) -> Result<(), GpuError> {
        let [a0, a1, a2, a3] = &mut self.u;
        let [o0, o1, o2, o3] = &mut self.u_next;
        kernels::reset_below(
            b,
            &mut self.energy,
            threshold,
            [a0, a1, a2, a3],
            [o0, o1, o2, o3],
            &mut self.resid,
            &mut self.slope,
            self.w * self.h,
        )
    }

    /// The pre-warp rule on the last solve, on the CPU (`gpu_support::untrusted_fill`, the rule of
    /// `prewarp` without its blur): when it fills untrusted pixels, the filled field goes to
    /// `u_next` (free until the next solve writes it) as Df64 and the result is true, for
    /// `smooth_filled`; otherwise the pre-warp is the plain blur of `u` (`smooth`).
    fn prewarp_fill(&mut self, cfg: &Config, it: usize, freeze: &mut LowFreeze) -> bool {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let cv = &mut self.conv;
        // The Vecs move into Images for the call and back afterwards, so they are allocated once.
        widen(&mut cv.energy, self.energy.as_slice::<f32>());
        widen(&mut cv.resid, self.resid.as_slice::<f32>());
        let energy = Image::from_vec(w, h, std::mem::take(&mut cv.energy));
        let resid = Image::from_vec(w, h, std::mem::take(&mut cv.resid));
        let low = freeze.low(it, || low_confidence(&energy, &resid, cfg));
        cv.energy = energy.data;
        cv.resid = resid.data;
        let [uxh, uxl, uyh, uyl] = self.u.each_ref().map(|b| b.as_slice::<f32>());
        let sl = self.slope.as_slice::<f32>();
        let at = |i: usize| {
            (
                f64::from(uxh[i]) + f64::from(uxl[i]),
                f64::from(uyh[i]) + f64::from(uyl[i]),
                std::array::from_fn(|k| f64::from(sl[k * n + i])),
            )
        };
        let Some(fill) = untrusted_fill(w, h, &low, cfg, at) else {
            return false;
        };
        for (dst, src) in self.u_next.iter_mut().zip(&self.u) {
            dst.as_mut_slice::<f32>()
                .par_chunks_mut(w)
                .zip(src.as_slice::<f32>().par_chunks(w))
                .for_each(|(d, s)| d.copy_from_slice(s));
        }
        let [oxh, oxl, oyh, oyl] = self.u_next.each_mut().map(|b| b.as_mut_slice::<f32>());
        for &(p, vx, vy) in &fill {
            let (p, dx, dy) = (p as usize, Df64::from_f64(vx), Df64::from_f64(vy));
            (oxh[p], oxl[p], oyh[p], oyl[p]) = (dx.hi, dx.lo, dy.hi, dy.lo);
        }
        true
    }

    fn release(self) {
        let g = self.gpu;
        self.plan.release(g);
        for b in [
            self.img2,
            self.coeffs2,
            self.scratch_n,
            self.spec,
            self.warped,
            self.refpack,
            self.responses,
            self.ga,
            self.gscratch,
            self.sx,
            self.sy,
            self.csx,
            self.csy,
            self.mid,
            self.taps_s,
            self.sums,
            self.global,
            self.mtaps,
            self.energy,
            self.resid,
            self.slope,
            self.erange,
        ]
        .into_iter()
        .chain(self.u)
        .chain(self.u_next)
        {
            g.release(b);
        }
    }
}

type StageFn = fn(&mut Work, &mut Batch) -> Result<(), GpuError>;

/// Runs `stages` in one command buffer, or (profile mode) one per stage, adding the wall time
/// to the stage each is told to.
fn run(
    eng: &GpuEngine,
    work: &mut Work,
    stages: &[(Stage, StageFn)],
    merged: Stage,
) -> Result<(), GpuError> {
    let gpu = &eng.gpu;
    if eng.profile {
        for &(stage, f) in stages {
            let t = Instant::now();
            let (r, s) = gpu.batch_timed(|b| f(work, b))?;
            r?;
            prof::add(stage, t);
            eng.book(stage, s);
        }
        Ok(())
    } else {
        let t = Instant::now();
        gpu.batch(|b| stages.iter().try_for_each(|&(_, f)| f(work, b)))??;
        prof::add(merged, t);
        Ok(())
    }
}

/// Command buffers committed by [`start`] and not yet waited for (none in profile mode).
struct Started {
    pending: Vec<Pending>,
    t: Instant,
    merged: Stage,
}

impl Started {
    /// Waits for every command buffer in order (dropping the rest waits for them too) and adds
    /// the wall time since the start to `merged`.
    fn wait(self) -> Result<(), GpuError> {
        let Started { pending, t, merged } = self;
        let timed = !pending.is_empty();
        for p in pending {
            p.wait()?;
        }
        if timed {
            prof::add(merged, t);
        }
        Ok(())
    }
}

/// [`run`] for each of `groups` (one command buffer each), committed without waiting: until
/// [`Started::wait`] the CPU may use every buffer the stages do not dispatch. Profile mode runs
/// and waits per stage as `run` does, so nothing is left to wait for.
fn start(
    eng: &GpuEngine,
    work: &mut Work,
    groups: &[&[(Stage, StageFn)]],
    merged: Stage,
) -> Result<Started, GpuError> {
    let mut started = Started {
        pending: Vec::new(),
        t: Instant::now(),
        merged,
    };
    for stages in groups {
        if eng.profile {
            run(eng, work, stages, merged)?;
        } else {
            let (r, p) = eng
                .gpu
                .submit(|b| stages.iter().try_for_each(|&(_, f)| f(work, b)))?;
            started.pending.push(p);
            r?;
        }
    }
    Ok(started)
}

impl GpuEngine {
    /// `pre`: the reference pack made ahead by `prepare_extra`, or None to make it here.
    #[allow(clippy::too_many_arguments)]
    fn refine(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
        pre: Option<Prefetched>,
    ) -> Result<RawFlow, GpuError> {
        let gpu = &self.gpu;
        let (w, h) = (i1.w, i1.h);
        let bad = |m: String| GpuError::Command(m);
        if w == 0 || h == 0 || i2.w == 0 || i2.h == 0 {
            return Err(bad(format!(
                "refine: empty image {w}x{h} / {}x{}",
                i2.w, i2.h
            )));
        }
        for (name, im) in [("ux", ux), ("uy", uy)] {
            if (im.w, im.h) != (w, h) {
                return Err(bad(format!(
                    "refine: i1 is {w}x{h} but {name} is {}x{}",
                    im.w, im.h
                )));
            }
        }
        let n = w * h;
        let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
        let np = pw * ph;

        // the first free lane. `try_lock`: the lock is held across rayon work (building the
        // filters on the CPU), so a call from a rayon worker could otherwise block that worker
        // on itself.
        let mut state = self
            .states
            .iter()
            .find_map(|m| match m.try_lock() {
                Ok(s) => Some(s),
                Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            })
            .ok_or_else(|| {
                bad(format!(
                    "GpuEngine::refine_raw called while every lane is running ({} window(s) at a \
                     time)",
                    self.states.len()
                ))
            })?;
        let k = key(ph, pw, cfg);
        // a pack made for another size or bank is not this window's: make it here instead
        let pre = pre.filter(|p| p.key == k);
        if !state.cache.contains_key(&k) {
            if state.cache.len() >= CACHE_ENTRIES {
                state.cache.clear();
            }
            let cpu = match &pre {
                Some(p) => Arc::clone(&p.bank),
                None => self.cpu_bank(ph, pw, cfg),
            };
            let entry = prof::time(Stage::Reference, || make_entry(gpu, ph, pw, &cpu.bank))?;
            state.cache.insert(k.clone(), entry);
        }
        let Entry { dev } = state.cache.get_mut(&k).expect("inserted above");
        let f = dev.filters;
        let g = GROUP.min(f);
        if let Some(p) = &pre {
            if p.pack.len() != n * f * 5 {
                return Err(bad(format!(
                    "refine: the prepared reference pack has {} values, the window needs {}",
                    p.pack.len(),
                    n * f * 5
                )));
            }
        }

        let (mtaps, r) = kernels::moment_taps_buffer(gpu, cfg.sigma_w)?;
        let mut u = [
            gpu.buffer(4 * n)?,
            gpu.buffer(4 * n)?,
            gpu.buffer(4 * n)?,
            gpu.buffer(4 * n)?,
        ];
        u[0].as_mut_slice::<f32>().copy_from_slice(&ux.data);
        u[2].as_mut_slice::<f32>().copy_from_slice(&uy.data);
        let mut energy = gpu.buffer(4 * n)?;
        energy.as_mut_slice::<f32>().fill(1.0);
        let mut resid = gpu.buffer(4 * n)?;
        resid
            .as_mut_slice::<f32>()
            .fill(std::f64::consts::PI as f32);
        let plan = MomentPlanGpu::new(gpu, cfg.shape_order)?;
        let n_global = plan.n_global;
        let mut wk = Work {
            gpu,
            e: dev,
            w,
            h,
            w2: i2.w,
            h2: i2.h,
            shift,
            f,
            img2: upload(gpu, &i2.data)?,
            coeffs2: gpu.buffer(4 * i2.w * i2.h)?,
            scratch_n: gpu.buffer(8 * i2.w * i2.h)?,
            spec: gpu.buffer(8 * np)?,
            warped: gpu.buffer(4 * n)?,
            refpack: gpu.buffer(4 * n * f * 5)?,
            responses: gpu.buffer(8 * np * f)?,
            ga: gpu.buffer(8 * np * g)?,
            gscratch: gpu.buffer(16 * np * g)?,
            sx: gpu.buffer(4 * n)?,
            sy: gpu.buffer(4 * n)?,
            csx: gpu.buffer(4 * n)?,
            csy: gpu.buffer(4 * n)?,
            mid: gpu.buffer(8 * n)?,
            taps_s: kernels::blur_taps(gpu, cfg.sigma_s)?,
            sums: gpu.buffer(28 * n)?,
            global: gpu.buffer(4 * n * n_global)?,
            mtaps,
            r,
            plan,
            u,
            u_next: [
                gpu.buffer(4 * n)?,
                gpu.buffer(4 * n)?,
                gpu.buffer(4 * n)?,
                gpu.buffer(4 * n)?,
            ],
            energy,
            resid,
            slope: gpu.buffer(16 * n)?,
            erange: gpu.buffer(12 * kernels::block_count(w, h))?,
            aperture: {
                let ap = Aperture::from_env();
                kernels::SolveAperture {
                    conf: ap.conf,
                    ridge: ap.ridge,
                    tol2: cfg.phase_tol * cfg.phase_tol,
                }
            },
            conv: Conv::default(),
        };

        // the first outer iteration up to the responses, committed without waiting: the
        // reference pack (which these stages do not use) is copied into its buffer meanwhile
        let first = |wk: &mut Work| {
            let coeffs: &[(Stage, StageFn)] = &[(Stage::Warp, |s, b| s.coeffs(b))];
            let iteration: &[(Stage, StageFn)] = &[
                (Stage::Smooth, |s, b| s.smooth(b)),
                (Stage::Warp, |s, b| s.warp(b)),
                (Stage::Responses, |s, b| s.responses(b)),
            ];
            if cfg.outer_iters > 0 {
                start(self, wk, &[coeffs, iteration], Stage::Responses)
            } else {
                start(self, wk, &[coeffs], Stage::Warp)
            }
        };
        // `OPTFLOW_ITER_TRACE`: the field after every outer iteration, read back (the first
        // iteration's time includes the reference pack)
        let trace = iter_trace::enabled();
        if trace {
            iter_trace::begin(w, &ux.data, &uy.data);
        }
        let mut t_it = trace.then(Instant::now);
        let fill = |wk: &mut Work, reference: Vec<f32>| {
            wk.refpack.as_mut_slice::<f32>().copy_from_slice(&reference);
        };
        match pre {
            // made one tile ahead (`prepare_extra`)
            Some(p) => {
                self.packs[1].fetch_add(1, Ordering::Relaxed);
                let started = first(&mut wk)?;
                fill(&mut wk, p.pack);
                started.wait()?;
            }
            // the reference pack on the CPU, while the GPU does the first outer iteration;
            // joined before the first accumulate
            None => {
                self.packs[0].fetch_add(1, Ordering::Relaxed);
                let cpu = self.cpu_bank(ph, pw, cfg);
                std::thread::scope(|sc| {
                    let pack = sc.spawn(|| {
                        prof::time(Stage::Reference, || ref_pack(i1, &cpu.bank, &cpu.fft))
                    });
                    let started = first(&mut wk);
                    let data = pack.join().map_err(|_| {
                        GpuError::Command("the reference-pack thread panicked".to_string())
                    });
                    // the GPU's error first (dropping `started` waits for its work)
                    let (started, data) = (started?, data?);
                    fill(&mut wk, data);
                    started.wait()
                })?;
            }
        }
        let mut freeze = LowFreeze::for_config(cfg);
        for it in 0..cfg.outer_iters {
            if it > 0 {
                t_it = t_it.map(|_| Instant::now());
                let filled = prof::time(Stage::Prewarp, || wk.prewarp_fill(cfg, it, &mut freeze));
                let smooth: StageFn = if filled {
                    |s, b| s.smooth_filled(b)
                } else {
                    |s, b| s.smooth(b)
                };
                run(
                    self,
                    &mut wk,
                    &[
                        (Stage::Smooth, smooth),
                        (Stage::Warp, |s, b| s.warp(b)),
                        (Stage::Responses, |s, b| s.responses(b)),
                    ],
                    Stage::Responses,
                )?;
            }
            for _ in 0..cfg.inner_iters {
                run(
                    self,
                    &mut wk,
                    &[
                        (Stage::Accumulate, |s, b| s.accumulate(b)),
                        (Stage::Moments, |s, b| s.moments(b)),
                        (Stage::Solve, |s, b| s.solve(b)),
                    ],
                    Stage::Accumulate,
                )?;
                let reset = prof::time(Stage::Solve, || fold_ranges(&wk.erange));
                if reset {
                    widen(&mut wk.conv.energy, wk.energy.as_slice::<f32>());
                    let threshold = 1e-9 * median(&wk.conv.energy) + 1e-30;
                    let t = Instant::now();
                    let (r, s) = gpu.batch_timed(|b| wk.reset(threshold, b))?;
                    r?;
                    prof::add(Stage::Solve, t);
                    if self.profile {
                        self.book(Stage::Solve, s);
                    }
                }
                std::mem::swap(&mut wk.u, &mut wk.u_next);
            }
            if let Some(t) = t_it {
                let secs = t.elapsed().as_secs_f64();
                let wide = |hi: &Buffer, lo: &Buffer| {
                    let (hi, lo) = (hi.as_slice::<f32>(), lo.as_slice::<f32>());
                    hi.iter()
                        .zip(lo)
                        .map(|(&a, &b)| f64::from(a) + f64::from(b))
                        .collect()
                };
                iter_trace::outer(wide(&wk.u[0], &wk.u[1]), wide(&wk.u[2], &wk.u[3]), secs);
            }
        }

        let df = |hi: &Buffer, lo: &Buffer| {
            Image::from_vec(
                w,
                h,
                hi.as_slice::<f32>()
                    .iter()
                    .zip(lo.as_slice::<f32>())
                    .map(|(&a, &b)| f64::from(a) + f64::from(b))
                    .collect(),
            )
        };
        let plane = |b: &Buffer| {
            Image::from_vec(
                w,
                h,
                b.as_slice::<f32>().iter().map(|&e| f64::from(e)).collect(),
            )
        };
        let out = RawFlow {
            ux: df(&wk.u[0], &wk.u[1]),
            uy: df(&wk.u[2], &wk.u[3]),
            energy: plane(&wk.energy),
            resid: plane(&wk.resid),
        };
        wk.release();
        Ok(out)
    }
}

/// True when the median threshold may reject a pixel (or the range is not usable), false
/// when no pixel can be below it: no NaN, finite highest, and `lowest > 1e-9 highest + 1e-30`.
fn fold_ranges(erange: &Buffer) -> bool {
    let (lo, hi, nan) = erange.as_slice::<f32>().chunks(3).fold(
        (f64::INFINITY, f64::NEG_INFINITY, false),
        |a, c| {
            (
                a.0.min(f64::from(c[0])),
                a.1.max(f64::from(c[1])),
                a.2 | (c[2] != 0.0),
            )
        },
    );
    let skip = !nan && hi <= f64::MAX / 2.0 && lo > 1e-9 * hi + 1e-30;
    !skip
}

impl FineEngine for GpuEngine {
    fn refine_raw(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<RawFlow, EngineError> {
        Ok(self.refine(i1, i2, shift, ux, uy, cfg, None)?)
    }

    /// The reference pack of `i1` on the CPU (rustfft, the bank from `cpu_banks`): no GPU work
    /// and no lane's lock, so it may run while another window is refined.
    fn prepare_extra(
        &self,
        i1: &Image<f32>,
        cfg: &Config,
    ) -> Option<Box<dyn std::any::Any + Send>> {
        if i1.w == 0 || i1.h == 0 {
            return None; // `refine` reports the empty window
        }
        let (ph, pw) = (i1.h + 2 * PAD, i1.w + 2 * PAD);
        let bank = self.cpu_bank(ph, pw, cfg);
        let pack = prof::time(Stage::Reference, || ref_pack(i1, &bank.bank, &bank.fft));
        Some(Box::new(Prefetched {
            pack,
            bank,
            key: key(ph, pw, cfg),
        }))
    }

    /// `refine_raw`, with the reference pack of `prepare_extra` when the tile has one (an extra
    /// of another type is ignored and the pack is made as in `refine_raw`).
    fn refine_prepared(&self, p: Prepared, cfg: &Config) -> Result<RawFlow, EngineError> {
        let pre = p
            .extra
            .and_then(|e| e.downcast::<Prefetched>().ok())
            .map(|b| *b);
        Ok(self.refine(&p.i1, &p.i2, p.offset, &p.ux, &p.uy, cfg, pre)?)
    }

    /// No: without a prepared pack `refine` makes it on another thread while the GPU runs the
    /// first outer iteration, instead of the GPU idling while the prefetch makes it.
    fn prefetch_first_tile(&self) -> bool {
        false
    }

    /// The GPU's `energy` and `resid` are `f32` buffers widened (`refine`).
    fn planes_exact_in_f32(&self) -> bool {
        true
    }

    /// The lanes of the engine (`OPTFLOW_GPU_LANES`): two tiles at once hide one tile's CPU work
    /// between its command buffers (the pre-warp rule, the copies, the launches) behind the
    /// other's GPU work. Each lane's buffers are its own, so the bits do not depend on the other.
    fn lanes(&self) -> usize {
        self.states.len()
    }

    /// Drops the device banks of every lane, the CPU banks and the pool's free buffers. Waits for
    /// a running refinement (the caller has none: it is called between the fine stage and the
    /// strain).
    fn release_caches(&self) {
        for state in &self.states {
            state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .cache
                .clear();
        }
        self.cpu_banks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.gpu.trim_pool();
    }

    fn check_window(&self, w: usize, h: usize, cfg: &Config) -> Result<(), EngineError> {
        let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
        let (n, np) = ((w * h) as u64, (pw * ph) as u64);
        // image 2 is the window moved by the coarse displacement and widened by the tile margin
        // on each side (`tile::displaced_window`): up to 2 margin plus twice the largest
        // displacement larger than the window (it is clipped to the whole image, unknown here)
        let extra = (2 * tile_margin(cfg) + 2 * cfg.max_disp.max(0.0).ceil() as usize + 2) as u64;
        let n2 = (w as u64 + extra) * (h as u64 + extra);
        let f = (cfg.scales.len() * cfg.n_orient) as u64;
        let g = f.min(GROUP as u64);
        let (unknowns, moments, ..) = plan_moments(cfg.shape_order);
        let global: Vec<_> = moments
            .iter()
            .filter(|m| [0u8, 1, 2, 6].contains(&m.0))
            .collect();
        let n_global = global.len() as u64;
        let blocks = kernels::block_count(w, h) as u64;
        let local_y = kernels::local_y_bytes(blocks as usize, kernels::MAX_LOCAL_COMBOS) as u64;
        let mut ykeys: Vec<(u8, u32)> = global.iter().map(|m| (m.0, m.1 .1)).collect();
        ykeys.sort_unstable();
        ykeys.dedup();
        let n_yplanes = ykeys.len() as u64;
        // the largest single buffers: responses, the reference pack
        let largest = (8 * np * f).max(20 * n * f);
        let need: u64 = 8 * np * f // responses
            + 20 * n * f // reference
            + 4 * np * f // bank
            + 8 * np // spectrum
            + (24 * np + 16 * np) * g // group planes and the spline scratch
            + 28 * n // sums
            + 4 * n * n_global // global moments
            + 4 * n * n_yplanes // their y pass
            + 4 * n * (unknowns as u64 + 1) // local moments
            + 32 * blocks // block anchors
            + local_y // local y pass
            + 32 * n // u, current and next (hi, lo, x, y)
            + 8 * n // energy, resid
            + 16 * n // slope
            + 4 * n * 5 // sx, sy, csx, csy, warped
            + 8 * n // blur intermediate
            + 16 * n2; // image 2, its coefficients and scratch
        let need = need + need / 20;
        let limit = self.gpu.recommended_working_set();
        if largest > self.gpu.max_buffer_length() || need > limit {
            return Err(GpuError::TooLarge {
                need: need.max(largest),
                limit: if largest > self.gpu.max_buffer_length() {
                    self.gpu.max_buffer_length()
                } else {
                    limit
                },
                w,
                h,
            }
            .into());
        }
        Ok(())
    }
}

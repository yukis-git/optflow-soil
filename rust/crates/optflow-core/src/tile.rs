//! Grid tiling of images into analysis windows, and the fine stage on tiles (large images).
//!
//! `refine` holds dozens of full-size `f64` maps, which does not fit for a multi-megapixel image.
//! [`refine_tiled`] runs [`refine_raw`] on one tile (core plus margin) at a time and keeps only
//! each tile's core; the confidence is formed afterwards from the joined fields, so it is that
//! of the whole image (median texture, border rule).

use rayon::prelude::*;

use crate::config::Config;
use crate::engine::{CpuEngine, EngineError, FineEngine, Prepared};
use crate::image::Image;
use crate::phaseflow::{
    border_zone, confidence, refine_low, reportable, texture_scale, Flow, RawFlow,
};

/// Half-open pixel rectangle `[x0, x1) x [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

/// A tile: the `core` it contributes to the result and the `ext` it is processed on (the core
/// plus a margin on every side, clipped to the image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub core: Rect,
    pub ext: Rect,
}

/// Number of tiles of size `t` along an axis of length `n`: `ceil(n / t)`. Needs `t >= 1`.
pub fn tile_count(n: usize, t: usize) -> usize {
    n.div_ceil(t)
}

/// Tile `k < tile_count(n, t)` along an axis of length `n`, as `(core0, core1, ext0, ext1)`:
/// the core is `[k t, min((k + 1) t, n))`, the extension adds `m` on each side within `[0, n]`.
/// The cores of consecutive tiles abut and together are exactly `[0, n)`.
pub fn tile_axis(n: usize, t: usize, m: usize, k: usize) -> (usize, usize, usize, usize) {
    let core0 = k * t;
    let core1 = n.min(core0.saturating_add(t));
    (
        core0,
        core1,
        core0.saturating_sub(m),
        n.min(core1.saturating_add(m)),
    )
}

/// The tiles of a `w` x `h` image, row by row: the product of the tiles of the two axes. Cores
/// have side `tile` (smaller at the right and bottom border) and partition the image; `ext` adds
/// `margin`. Needs `tile >= 1`; an empty image has no tiles.
pub fn tiles(w: usize, h: usize, tile: usize, margin: usize) -> Vec<Tile> {
    assert!(tile >= 1, "tiles: tile size must be at least 1");
    let mut out = Vec::with_capacity(tile_count(w, tile) * tile_count(h, tile));
    for ky in 0..tile_count(h, tile) {
        let (cy0, cy1, ey0, ey1) = tile_axis(h, tile, margin, ky);
        for kx in 0..tile_count(w, tile) {
            let (cx0, cx1, ex0, ex1) = tile_axis(w, tile, margin, kx);
            out.push(Tile {
                core: Rect {
                    x0: cx0,
                    y0: cy0,
                    x1: cx1,
                    y1: cy1,
                },
                ext: Rect {
                    x0: ex0,
                    y0: ey0,
                    x1: ex1,
                    y1: ey1,
                },
            });
        }
    }
    out
}

/// Margin around a tile core: at least 64 px and more than the reach of the Gaussian windows
/// (`4 sigma_w` of the local fit, `4 sigma_s` of the displacement smoothing), of the edge guard
/// and of the strain window.
pub fn tile_margin(cfg: &Config) -> usize {
    // `as usize` saturates and maps NaN to 0
    let ceil = |v: f64| v.ceil() as usize;
    [
        64,
        ceil(4.0 * cfg.sigma_w),
        ceil(4.0 * cfg.sigma_s),
        ceil(cfg.edge_guard_px()),
        cfg.strain_win,
    ]
    .into_iter()
    .max()
    .unwrap_or(64)
}

fn crop<T: Copy + Default>(img: &Image<T>, r: Rect) -> Image<T> {
    let w = r.x1 - r.x0;
    let mut data = Vec::with_capacity(w * (r.y1 - r.y0));
    for y in r.y0..r.y1 {
        data.extend_from_slice(&img.data[y * img.w + r.x0..y * img.w + r.x1]);
    }
    Image::from_vec(w, r.y1 - r.y0, data)
}

/// The part of the second image (`w` x `h`) a tile needs: the tile's extension moved by the
/// coarse displacement (the finite minimum and maximum of `ux`, `uy`, the start field on `ext`,
/// bound where the tile's pixels land), widened by `margin` on every side and clipped to the
/// image. Never empty: a window that would lie outside the image collapses to the border strip,
/// which is what a lookup outside the whole image clamps to as well.
fn displaced_window(
    ext: Rect,
    (ux, uy): (&Image<f32>, &Image<f32>),
    (w, h): (usize, usize),
    margin: usize,
) -> Rect {
    let range = |u: &Image<f32>| {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &u.data {
            if v.is_finite() {
                lo = lo.min(f64::from(v));
                hi = hi.max(f64::from(v));
            }
        }
        if lo > hi {
            (0.0, 0.0) // no finite displacement: the tile stays where it is
        } else {
            (lo, hi)
        }
    };
    let axis = |first: usize, end: usize, (lo, hi): (f64, f64), n: usize| {
        let m = margin as f64;
        let a = (first as f64 + lo).floor() - m;
        let b = ((end - 1) as f64 + hi).ceil() + 1.0 + m;
        // `a` can be NaN-free but huge: clamp in f64 before the cast
        let a = a.clamp(0.0, (n - 1) as f64) as usize;
        let b = b.clamp((a + 1) as f64, n as f64) as usize;
        (a, b)
    };
    let (x0, x1) = axis(ext.x0, ext.x1, range(ux), w);
    let (y0, y1) = axis(ext.y0, ext.y1, range(uy), h);
    Rect { x0, y0, x1, y1 }
}

/// [`refine`] on tiles of `tile` x `tile` pixels (core), for images that are too large for one
/// pass. The coarse field `(ux, uy)` is cut to each tile's extension; the second image is cut
/// to the extension displaced by the coarse field (see `displaced_window`). Only the tile cores
/// are kept, so the fields agree with the untiled result up to the influence of the tile border
/// on the margin's far end, which the margin is chosen to make negligible. An image with a single
/// tile is processed by [`refine`] itself.
pub fn refine_tiled(
    i1: &Image<f32>,
    i2: &Image<f32>,
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
    tile: usize,
) -> Flow {
    refine_tiled_low(i1, i2, ux, uy, cfg, tile).0
}

/// [`refine_tiled`], and `low`: whether the `f64` confidence is below `cfg.conf_min`
/// (see [`refine_low`]).
pub(crate) fn refine_tiled_low(
    i1: &Image<f32>,
    i2: &Image<f32>,
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
    tile: usize,
) -> (Flow, Image<bool>) {
    let (w, h) = (i1.w, i1.h);
    for (name, im) in [("i2", i2), ("ux", ux), ("uy", uy)] {
        assert!(
            (im.w, im.h) == (w, h),
            "refine_tiled: i1 is {w}x{h} but {name} is {}x{}",
            im.w,
            im.h
        );
    }
    refine_tiled_from_on(
        &CpuEngine,
        i1,
        i2,
        &|r| (crop(ux, r), crop(uy, r)),
        cfg,
        tile,
    )
    .expect("the CPU engine does not fail")
}

/// [`refine_tiled_low`] with the start field given per rectangle: `start(r)` is the field on
/// `r`, which the tiled driver asks for one tile's extension at a time (the start of the whole
/// image need not exist).
///
/// The tiles run largest extension first, for macOS libmalloc's large-allocation cache: it keeps
/// freed large blocks resident and, as measured, serves a later request from one only if the
/// request is more than half the block's size; a request larger than every cached block gets
/// fresh memory. In this order the buffers of every later (smaller) tile fit the blocks of the
/// first one instead of adding to the footprint. With another allocator the order is harmless,
/// only less useful. It does not change the result: a tile reads only the inputs and writes only
/// its own core.
pub(crate) fn refine_tiled_from_on(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    start: &StartFn<'_>,
    cfg: &Config,
    tile: usize,
) -> Result<(Flow, Image<bool>), EngineError> {
    // read once per call: `OPTFLOW_TILE_PREFETCH=0` turns the prefetch off (spec §7, memory)
    let prefetch = std::env::var("OPTFLOW_TILE_PREFETCH").map_or(true, |v| v != "0");
    refine_tiled_from_on_with(engine, i1, i2, start, cfg, tile, prefetch)
}

/// The start field of a rectangle, called from the prefetch thread as well.
pub(crate) type StartFn<'a> = dyn Fn(Rect) -> (Image<f32>, Image<f32>) + Sync + 'a;

/// [`refine_tiled_from_on`] with the prefetch given: `prefetch` prepares tile k+1 (its start
/// field, crops, displaced window and [`FineEngine::prepare_extra`]) on one other thread while
/// the engine refines tile k; otherwise one tile at a time, without an extra. The cores are
/// written in the same order by the same code either way, so the output is the same bit for bit.
pub(crate) fn refine_tiled_from_on_with(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    start: &StartFn<'_>,
    cfg: &Config,
    tile: usize,
    prefetch: bool,
) -> Result<(Flow, Image<bool>), EngineError> {
    let (w, h) = (i1.w, i1.h);
    assert!(
        (i2.w, i2.h) == (w, h),
        "refine_tiled: i1 is {w}x{h} but i2 is {}x{}",
        i2.w,
        i2.h
    );
    let margin = tile_margin(cfg);
    let ts = tiles(w, h, tile, margin);
    if ts.len() <= 1 {
        let (ux, uy) = start(Rect {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        });
        return refine_low(engine, i1, i2, &ux, &uy, cfg);
    }
    if engine.planes_exact_in_f32() {
        join_tiles::<f32>(engine, i1, i2, start, cfg, prefetch, &ts, margin)
    } else {
        join_tiles::<f64>(engine, i1, i2, start, cfg, prefetch, &ts, margin)
    }
}

/// A value kept per pixel between the tiles and the confidence: the `f64` of the iterations
/// itself, or, for an engine whose values are `f32` widened, that `f32` (half the memory).
trait Plane: Copy + Send + Sync + Default {
    /// Exact for `f64`, and for `f32` when `v` is a widened `f32`
    /// ([`FineEngine::planes_exact_in_f32`]).
    fn store(v: f64) -> Self;
    fn load(self) -> f64;
    fn to_f32(self) -> f32;
    /// [`texture_scale`] of an energy plane.
    fn texture_scale(energy: &[Self]) -> f64;
}

impl Plane for f64 {
    fn store(v: f64) -> Self {
        v
    }
    fn load(self) -> f64 {
        self
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn texture_scale(energy: &[Self]) -> f64 {
        texture_scale(energy)
    }
}

impl Plane for f32 {
    fn store(v: f64) -> Self {
        v as f32
    }
    fn load(self) -> f64 {
        f64::from(self)
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn texture_scale(energy: &[Self]) -> f64 {
        // the median reads `f64`; the widened copy lives only for this call
        let wide: Vec<f64> = energy.iter().map(|&v| f64::from(v)).collect();
        texture_scale(&wide)
    }
}

/// The tiles of [`refine_tiled_from_on_with`] (more than one) refined and joined, with the
/// residual and the energy kept as `P`.
#[allow(clippy::too_many_arguments)]
fn join_tiles<P: Plane>(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    start: &StartFn<'_>,
    cfg: &Config,
    prefetch: bool,
    ts: &[Tile],
    margin: usize,
) -> Result<(Flow, Image<bool>), EngineError> {
    let (w, h) = (i1.w, i1.h);
    // the cores joined: displacement as in `Flow`; the residual (in the `f64` of the iterations),
    // the energy and the border rule are kept for the confidence, which needs the median energy of
    // the whole image
    let n = w * h;
    let b = border_zone(cfg);
    let mut out_ux = vec![0.0f32; n];
    let mut out_uy = vec![0.0f32; n];
    let mut resid = vec![P::default(); n];
    let mut energy = vec![P::default(); n];
    let mut keep = vec![false; n];
    let area = |r: Rect| (r.x1 - r.x0) * (r.y1 - r.y0);
    let mut order: Vec<usize> = (0..ts.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(area(ts[k].ext))); // stable: row order among equals

    // what a tile needs before the engine runs: reads only the inputs (any thread)
    let prepare = |t: &Tile, extra: bool| {
        let (ux, uy) = start(t.ext);
        let win2 = displaced_window(t.ext, (&ux, &uy), (w, h), margin);
        let i1 = crop(i1, t.ext);
        let extra = if extra {
            engine.prepare_extra(&i1, cfg)
        } else {
            None
        };
        Prepared {
            i1,
            i2: crop(i2, win2),
            offset: (
                t.ext.x0 as i64 - win2.x0 as i64,
                t.ext.y0 as i64 - win2.y0 as i64,
            ),
            ux,
            uy,
            extra,
        }
    };
    let trace = crate::iter_trace::enabled();
    let mut traced = Vec::new();
    // the trace of the tile just refined on this thread
    let taken = |t: &Tile| {
        if trace {
            crate::iter_trace::take(t.core, t.ext)
        } else {
            None
        }
    };
    // the core of a refined tile, row by row in parallel
    let mut write = |t: &Tile, raw: RawFlow, tt: Traced| {
        traced.extend(tt);
        let RawFlow {
            ux: tux,
            uy: tuy,
            energy: ten,
            resid: tres,
        } = raw;
        let ew = t.ext.x1 - t.ext.x0;
        let (cx0, cx1) = (t.core.x0, t.core.x1);
        out_ux
            .par_chunks_mut(w)
            .zip(out_uy.par_chunks_mut(w))
            .zip(resid.par_chunks_mut(w))
            .zip(energy.par_chunks_mut(w))
            .zip(keep.par_chunks_mut(w))
            .enumerate()
            .skip(t.core.y0)
            .take(t.core.y1 - t.core.y0)
            .for_each(|(y, ((((r_ux, r_uy), r_res), r_en), r_keep))| {
                for x in cx0..cx1 {
                    let l = (y - t.ext.y0) * ew + (x - t.ext.x0);
                    let (u, v) = (tux.data[l], tuy.data[l]);
                    r_ux[x] = u as f32;
                    r_uy[x] = v as f32;
                    r_res[x] = P::store(tres.data[l]);
                    r_en[x] = P::store(ten.data[l]);
                    r_keep[x] = reportable(x, y, x as f64 + u, y as f64 + v, w, h, b);
                }
            });
    };
    // one at a time under the memory probe, whose tile marks must not overlap
    let lanes = if crate::prof::probing() {
        1
    } else {
        engine.lanes()
    };
    if prefetch && lanes > 1 {
        // `lanes` tiles refined at once (the same prefetch, the cores written in order)
        refine_in_lanes(
            engine,
            lanes,
            &order,
            !engine.prefetch_first_tile(),
            |pos, extra| prepare(&ts[order[pos]], extra),
            |pos| taken(&ts[order[pos]]),
            cfg,
            |pos, raw, tt| write(&ts[order[pos]], raw, tt),
        )?;
    } else if prefetch {
        // one producer prepares the tiles in order; the rendezvous channel (no buffer) lets it
        // finish tile k+1 while the engine refines tile k and then wait for its hand-over, so at
        // most one prepared tile waits (two tiles' CPU data at once, spec §7). Without
        // `prefetch_first_tile` the first tile is prepared here, without an extra, and the
        // producer starts with the second.
        let first = (!engine.prefetch_first_tile()).then_some(order[0]);
        let ahead = &order[usize::from(first.is_some())..];
        std::thread::scope(|sc| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Prepared>(0);
            let prepare = &prepare;
            let ts = &ts;
            let producer = sc.spawn(move || {
                for &k in ahead {
                    // a send fails when the consumer stopped (an engine error): stop as well
                    if tx.send(prepare(&ts[k], true)).is_err() {
                        return;
                    }
                }
            });
            // `rx` is dropped before the join (and on unwinding out of this closure), which
            // unblocks a producer waiting in `send`
            let refined = (|| {
                if let Some(k) = first {
                    crate::prof::tile(k, true);
                    let raw = engine.refine_prepared(prepare(&ts[k], false), cfg)?;
                    write(&ts[k], raw, taken(&ts[k]));
                    crate::prof::tile(k, false);
                }
                for &k in ahead {
                    crate::prof::tile(k, true);
                    // a closed channel before the end: the producer panicked (its join says how)
                    let Ok(p) = rx.recv() else {
                        return Ok(false);
                    };
                    let raw = engine.refine_prepared(p, cfg)?;
                    write(&ts[k], raw, taken(&ts[k]));
                    crate::prof::tile(k, false);
                }
                Ok(true)
            })();
            drop(rx);
            let joined = producer.join();
            match (refined, joined) {
                // the engine's error first: one at a time, the later tile is never prepared
                (Err(e), _) => Err(e),
                (Ok(_), Err(panic)) => std::panic::resume_unwind(panic),
                (Ok(true), Ok(())) => Ok(()),
                (Ok(false), Ok(())) => unreachable!("the producer stops early only by a panic"),
            }
        })?;
    } else {
        for &k in &order {
            let t = &ts[k];
            crate::prof::tile(k, true);
            let raw = engine.refine_prepared(prepare(t, false), cfg)?;
            write(t, raw, taken(t));
            crate::prof::tile(k, false);
        }
    }
    let t_conf = crate::prof::start(crate::prof::Stage::Confidence);
    let scale = P::texture_scale(&energy);
    let (conf, low): (Vec<f32>, Vec<bool>) = (0..n)
        .into_par_iter()
        .map(|i| {
            let c = confidence(
                energy[i].load(),
                resid[i].load(),
                keep[i],
                scale,
                cfg.phase_tol,
            );
            (c as f32, c < cfg.conf_min)
        })
        .unzip();
    let flow = Flow {
        ux: Image::from_vec(w, h, out_ux),
        uy: Image::from_vec(w, h, out_uy),
        conf: Image::from_vec(w, h, conf),
        resid: Image::from_vec(w, h, resid.par_iter().map(|&r| r.to_f32()).collect()),
    };
    crate::prof::add(crate::prof::Stage::Confidence, t_conf);
    let low = Image::from_vec(w, h, low);
    if trace {
        crate::iter_trace::finish(traced, &low);
    }
    Ok((flow, low))
}

/// Why a lane stopped: the engine's error, or a panic (preparing the first tile or refining).
enum Failed {
    Engine(EngineError),
    Panic(Box<dyn std::any::Any + Send>),
}

/// The iteration trace of a refined tile ([`crate::iter_trace::take`]).
type Traced = Option<crate::iter_trace::TileTrace>;

/// The tiles of `order` (positions `0..order.len()`; the tile indices are for the profile)
/// refined on `lanes` threads at once ([`FineEngine::lanes`]), each taking the next tile when it
/// is done. One producer prepares the tiles in order (`prepare(pos, true)`, from position 1 when
/// `first_here`: the first tile is then prepared without an extra by the lane that takes it) and
/// hands each over a rendezvous channel, so at most one prepared tile waits besides the ones
/// being refined. `taken(pos)` runs on the lane after the refinement. The results are written in
/// the order of the positions, as one at a time, and a lane takes position `p` only when
/// position `p - lanes` is written, so at most `lanes - 1` results wait for their turn. The first
/// position that fails decides the outcome as one at a time would: its engine error is returned,
/// its panic (also one of `prepare` or `taken`) resumed, a panic of the producer only when every
/// earlier tile was refined; after a failure no lane takes a later tile, and the ones running
/// are waited for.
///
/// Nothing waits forever: the producer waits only for a lane to take its tile or for the driver
/// to stop; a lane waits for the producer, for a write or for a failure. The position the driver
/// waits for (`written`) never waits for a write (it is below `written + lanes`), and the lanes
/// on earlier positions finish them. When the driver stops, also by unwinding out of `write`,
/// its guard marks every later position failed, wakes the lanes waiting for a write and drops
/// the receiver (unblocking the producer's `send`), so the lanes and the producer return and the
/// scope joins them.
#[allow(clippy::too_many_arguments)]
fn refine_in_lanes(
    engine: &dyn FineEngine,
    lanes: usize,
    order: &[usize],
    first_here: bool,
    prepare: impl Fn(usize, bool) -> Prepared + Sync,
    taken: impl Fn(usize) -> Traced + Sync,
    cfg: &Config,
    mut write: impl FnMut(usize, RawFlow, Traced),
) -> Result<(), EngineError> {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::mpsc::{channel, sync_channel, Receiver};
    use std::sync::{Condvar, Mutex};
    type Done = Result<(RawFlow, Traced), Failed>;
    /// The next tile for a lane: its position, and the producer's channel.
    struct Next {
        pos: usize,
        rx: Option<Receiver<(usize, Prepared)>>,
    }
    /// What the lanes wait on: the positions written (`0..written`) and the first that failed
    /// (`usize::MAX`: none; 0 once the driver stopped).
    struct Turn {
        written: Mutex<usize>,
        moved: Condvar,
        first_failed: AtomicUsize,
    }
    impl Turn {
        /// Position `pos` failed (the driver stopped: 0): no lane takes a later one.
        fn fail(&self, pos: usize) {
            self.first_failed.fetch_min(pos, SeqCst);
            // under the lock, so a lane that checked before it waits does not miss the wake-up
            let _g = self.written.lock().unwrap_or_else(|e| e.into_inner());
            self.moved.notify_all();
        }
        /// Whether `pos` comes after the first failed position.
        fn skips(&self, pos: usize) -> bool {
            pos > self.first_failed.load(SeqCst)
        }
        /// Waits until `pos - lanes` is written (true) or `pos` skips (false).
        fn may_take(&self, pos: usize, lanes: usize) -> bool {
            let mut w = self.written.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if self.skips(pos) {
                    return false;
                }
                if pos < *w + lanes {
                    return true;
                }
                w = self.moved.wait(w).unwrap_or_else(|e| e.into_inner());
            }
        }
    }
    /// Stops the lanes and the producer when the driver stops (returns or unwinds).
    struct Stop<'a> {
        turn: &'a Turn,
        next: &'a Mutex<Next>,
    }
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.turn.fail(0);
            // a lane waiting in `recv` holds the lock until the producer sends or returns
            self.next.lock().unwrap_or_else(|e| e.into_inner()).rx = None;
        }
    }
    let n = order.len();
    let turn = Turn {
        written: Mutex::new(0),
        moved: Condvar::new(),
        first_failed: AtomicUsize::new(usize::MAX),
    };
    let (tx, rx) = sync_channel::<(usize, Prepared)>(0);
    let next = Mutex::new(Next {
        pos: 0,
        rx: Some(rx),
    });
    std::thread::scope(|sc| {
        let stop = Stop {
            turn: &turn,
            next: &next,
        };
        let (done_tx, done_rx) = channel::<(usize, Done)>();
        let prepare = &prepare;
        let producer = sc.spawn(move || {
            for pos in usize::from(first_here)..n {
                // a send fails when the driver stopped (a failure): stop as well
                if tx.send((pos, prepare(pos, true))).is_err() {
                    return;
                }
            }
        });
        for _ in 0..lanes {
            let done_tx = done_tx.clone();
            let (next, turn, taken) = (&next, &turn, &taken);
            sc.spawn(move || loop {
                // the lock is held while waiting for the turn and for the producer, so the tiles
                // go out in order
                let (pos, p) = {
                    let mut g = next.lock().unwrap_or_else(|e| e.into_inner());
                    let pos = g.pos;
                    if pos == n || !turn.may_take(pos, lanes) {
                        return;
                    }
                    let p = if first_here && pos == 0 {
                        None
                    } else {
                        match g.rx.as_ref().map(Receiver::recv) {
                            Some(Ok((at, p))) => {
                                debug_assert_eq!(at, pos);
                                Some(p)
                            }
                            // the producer panicked, or the driver stopped
                            _ => return,
                        }
                    };
                    g.pos += 1;
                    (pos, p)
                };
                // a failure while this lane waited for the producer
                if turn.skips(pos) {
                    return;
                }
                let run = std::panic::AssertUnwindSafe(|| {
                    crate::prof::tile(order[pos], true);
                    let p = p.unwrap_or_else(|| prepare(pos, false));
                    let done = engine.refine_prepared(p, cfg).map(|raw| (raw, taken(pos)));
                    crate::prof::tile(order[pos], false);
                    done
                });
                let done = match std::panic::catch_unwind(run) {
                    Ok(Ok(d)) => Ok(d),
                    Ok(Err(e)) => Err(Failed::Engine(e)),
                    Err(panic) => Err(Failed::Panic(panic)),
                };
                if done.is_err() {
                    turn.fail(pos);
                }
                if done_tx.send((pos, done)).is_err() {
                    return;
                }
            });
        }
        drop(done_tx);
        // the results in order; one that comes early waits here (at most `lanes - 1` of them)
        let mut early = std::collections::HashMap::new();
        // Err(None): every lane returned before this position's result, the producer panicked
        let mut outcome = Ok(());
        for pos in 0..n {
            let done = loop {
                if let Some(d) = early.remove(&pos) {
                    break Some(d);
                }
                match done_rx.recv() {
                    Ok((p, d)) => {
                        early.insert(p, d);
                    }
                    Err(_) => break None,
                }
            };
            match done {
                Some(Ok((raw, tt))) => {
                    write(pos, raw, tt);
                    *turn.written.lock().unwrap_or_else(|e| e.into_inner()) = pos + 1;
                    turn.moved.notify_all();
                }
                Some(Err(f)) => {
                    outcome = Err(Some(f));
                    break;
                }
                None => {
                    outcome = Err(None);
                    break;
                }
            }
        }
        drop(stop);
        drop(done_rx);
        let joined = producer.join();
        match (outcome, joined) {
            (Err(Some(Failed::Engine(e))), _) => Err(e),
            (Err(Some(Failed::Panic(panic))), _) => std::panic::resume_unwind(panic),
            (_, Err(panic)) => std::panic::resume_unwind(panic),
            (Ok(()), Ok(())) => Ok(()),
            // a lane returns without a position's result only after a failure at an earlier one
            // (whose result came first) or when the producer returned, which it does early only
            // by a panic
            (Err(None), Ok(())) => unreachable!("the producer stops early only by a panic"),
        }
    })
}

/// [`refine_tiled_from_on`] with the CPU engine (for the tests of the modules that call it).
#[cfg(test)]
pub(crate) fn refine_tiled_from(
    i1: &Image<f32>,
    i2: &Image<f32>,
    start: &StartFn<'_>,
    cfg: &Config,
    tile: usize,
) -> (Flow, Image<bool>) {
    refine_tiled_from_on(&CpuEngine, i1, i2, start, cfg, tile)
        .expect("the CPU engine does not fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pixel is in exactly one core, every core lies in its extension, every extension in
    /// the image, for the sizes of the Kani harness and beyond.
    #[test]
    fn tiles_equal_the_product_of_the_axis_tiles_and_cover_each_pixel_once() {
        for (w, h, t, m) in [
            (1, 1, 4, 0),
            (7, 5, 4, 3),
            (32, 32, 16, 8),
            (33, 20, 16, 8),
            (31, 17, 5, 2),
            (100, 60, 64, 64),
            (20, 100, 7, 100),
            (16, 16, 16, 0),
        ] {
            let ts = tiles(w, h, t, m);
            assert_eq!(ts.len(), tile_count(w, t) * tile_count(h, t));
            let mut i = 0;
            for ky in 0..tile_count(h, t) {
                for kx in 0..tile_count(w, t) {
                    let (cx0, cx1, ex0, ex1) = tile_axis(w, t, m, kx);
                    let (cy0, cy1, ey0, ey1) = tile_axis(h, t, m, ky);
                    let want = Tile {
                        core: Rect {
                            x0: cx0,
                            y0: cy0,
                            x1: cx1,
                            y1: cy1,
                        },
                        ext: Rect {
                            x0: ex0,
                            y0: ey0,
                            x1: ex1,
                            y1: ey1,
                        },
                    };
                    assert_eq!(ts[i], want, "{w}x{h} t={t} m={m} tile ({kx}, {ky})");
                    i += 1;
                }
            }
            let mut hits = vec![0u32; w * h];
            for tl in &ts {
                let (c, e) = (tl.core, tl.ext);
                assert!(e.x0 <= c.x0 && c.x0 < c.x1 && c.x1 <= e.x1 && e.x1 <= w);
                assert!(e.y0 <= c.y0 && c.y0 < c.y1 && c.y1 <= e.y1 && e.y1 <= h);
                assert!(c.x0 - e.x0 <= m && e.x1 - c.x1 <= m);
                assert!(c.y0 - e.y0 <= m && e.y1 - c.y1 <= m);
                for y in c.y0..c.y1 {
                    for x in c.x0..c.x1 {
                        hits[y * w + x] += 1;
                    }
                }
            }
            assert!(
                hits.iter().all(|&n| n == 1),
                "{w}x{h} t={t} m={m}: some pixel is not covered exactly once"
            );
        }
    }

    #[test]
    fn tiles_of_an_image_within_one_tile_are_the_whole_image() {
        let ts = tiles(300, 200, 1024, 64);
        let whole = Rect {
            x0: 0,
            y0: 0,
            x1: 300,
            y1: 200,
        };
        assert_eq!(
            ts,
            vec![Tile {
                core: whole,
                ext: whole
            }]
        );
        assert!(tiles(0, 5, 4, 1).is_empty());
    }

    #[test]
    fn tile_margin_is_64_for_the_default_config_and_grows_with_the_reach() {
        let cfg = Config::default();
        assert_eq!(tile_margin(&cfg), 64);
        let wide = |f: fn(&mut Config)| {
            let mut c = Config::default();
            f(&mut c);
            tile_margin(&c)
        };
        assert_eq!(wide(|c| c.sigma_w = 20.0), 80);
        assert_eq!(wide(|c| c.sigma_s = 21.0), 84);
        assert_eq!(wide(|c| c.edge_guard = 100.5), 101);
        assert_eq!(wide(|c| c.strain_win = 99), 99);
        // the automatic guard is 3 sigma of the longest filter: 12.2 px by default
        assert_eq!(wide(|c| c.lam = 80.0), 123);
    }

    fn constant(w: usize, h: usize, v: f32) -> Image<f32> {
        Image::from_vec(w, h, vec![v; w * h])
    }

    /// The window of the second image a tile needs: the extension moved by the extreme coarse
    /// displacements, one pixel more at the far end (the last pixel's interpolation support),
    /// widened by the margin, clipped to the image.
    #[test]
    fn displaced_window_covers_the_moved_extension_plus_the_margin() {
        let ext = Rect {
            x0: 10,
            y0: 20,
            x1: 20,
            y1: 40,
        };
        let size = (100, 100);
        let window = |ux: &Image<f32>, uy: &Image<f32>, margin: usize| {
            displaced_window(ext, (ux, uy), size, margin)
        };
        let (zero, shifted) = (constant(100, 100, 0.0), constant(100, 100, 0.5));
        // no displacement: the extension and the pixel after its last
        let r = window(&zero, &zero, 0);
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (10, 20, 20, 40));
        // the margin on every side
        let r = window(&zero, &zero, 3);
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (7, 23, 17, 43));
        // displaced by half a pixel: floor at the near end, ceil at the far end, plus the one
        let r = window(&shifted, &shifted, 0);
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (10, 21, 20, 41));
        let back = constant(100, 100, -2.5);
        let r = window(&back, &zero, 1);
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (6, 19, 19, 41));
        // the extremes of a field count, NaN and the infinities do not
        let mut ramp = constant(100, 100, 1.0);
        ramp.data[0] = 4.0;
        ramp.data[1] = -3.0;
        ramp.data[2] = f32::NAN;
        ramp.data[3] = f32::INFINITY;
        ramp.data[4] = f32::NEG_INFINITY;
        let r = window(&ramp, &zero, 0);
        assert_eq!((r.x0, r.x1), (7, 24));
        // a field without a finite value leaves the tile where it is
        let none = constant(100, 100, f32::NAN);
        let r = window(&none, &none, 0);
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (10, 20, 20, 40));
    }

    /// A 100 x 60 image: x is clipped at the width, y at the height (not the other way round),
    /// at the far end, to the image, with the margin and a displacement.
    #[test]
    fn displaced_window_clips_each_axis_to_its_own_size() {
        let ext = Rect {
            x0: 70,
            y0: 10,
            x1: 95,
            y1: 25,
        };
        let size = (100, 60);
        let ux = constant(100, 60, 10.0);
        let uy = constant(100, 60, 30.0);
        // x: [floor(70 + 10) - 5, ceil(94 + 10) + 1 + 5) = [75, 110) -> [75, 100)
        // y: [floor(10 + 30) - 10, ceil(24 + 30) + 1 + 10) = [30, 65) -> [30, 60)
        let r = displaced_window(ext, (&ux, &uy), size, 5);
        assert_eq!((r.x0, r.x1), (75, 100));
        let r = displaced_window(ext, (&ux, &uy), size, 10);
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (70, 100, 30, 60));
        // inside both: not clipped
        let r = displaced_window(
            ext,
            (&constant(100, 60, 0.0), &constant(100, 60, 0.0)),
            size,
            3,
        );
        assert_eq!((r.x0, r.x1, r.y0, r.y1), (67, 98, 7, 28));
    }

    /// A window that would lie outside the image collapses to the border strip: one pixel wide
    /// at least, at either end, never empty.
    #[test]
    fn displaced_window_is_never_empty() {
        let ext = Rect {
            x0: 0,
            y0: 0,
            x1: 5,
            y1: 5,
        };
        for (d, want) in [(-100.0f32, (0, 1)), (500.0, (99, 100))] {
            let u = constant(100, 100, d);
            let r = displaced_window(ext, (&u, &u), (100, 100), 0);
            assert_eq!((r.x0, r.x1), want, "displacement {d}");
            assert_eq!((r.y0, r.y1), want, "displacement {d}");
        }
    }

    /// A smooth random texture and the same texture moved by `d` (sums of sinusoids).
    fn speckle_pair(w: usize, h: usize, d: (f64, f64)) -> (Image<f32>, Image<f32>) {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let waves: Vec<(f64, f64, f64)> = (0..24)
            .map(|_| {
                let (f, a) = (0.15 + 0.5 * rnd(), std::f64::consts::TAU * rnd());
                (f * a.cos(), f * a.sin(), std::f64::consts::TAU * rnd())
            })
            .collect();
        let at = |x: f64, y: f64| {
            waves
                .iter()
                .map(|&(kx, ky, p)| (kx * x + ky * y + p).sin())
                .sum::<f64>() as f32
        };
        let img = |dx: f64, dy: f64| {
            let data = (0..w * h)
                .map(|i| at((i % w) as f64 - dx, (i / w) as f64 - dy))
                .collect();
            Image::from_vec(w, h, data)
        };
        (img(0.0, 0.0), img(d.0, d.1))
    }

    fn zero_start(r: Rect) -> (Image<f32>, Image<f32>) {
        let (w, h) = (r.x1 - r.x0, r.y1 - r.y0);
        (constant(w, h, 0.0), constant(w, h, 0.0))
    }

    fn bits(im: &Image<f32>) -> Vec<u32> {
        im.data.iter().map(|v| v.to_bits()).collect()
    }

    /// Runs `f` on its own thread and fails if it takes longer than a minute (a deadlock).
    fn within_a_minute<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(std::time::Duration::from_secs(60))
            .expect("the tiled driver hung (or its thread died)")
    }

    /// The panic message of `f`, or None if it returned.
    fn panic_of<T>(f: impl FnOnce() -> T) -> Option<String> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(_) => None,
            Err(p) => Some(
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default(),
            ),
        }
    }

    /// The CPU engine, recording where `prepare_extra` runs and whether `refine_prepared` got the
    /// extra; fails the `fail_at`-th refinement (0-based) and panics preparing the extra of the
    /// tile whose extension starts at `panic_extra_at`; `first_here` answers no to
    /// `prefetch_first_tile` (as the GPU engine). `lanes` is what it answers to `lanes` (0 = 1);
    /// with more than one, the first refinements wait (up to 5 s) until two run at once, and
    /// `most` is the most that ran at once. `fail_on` fails (`false`) or panics (`true`) on the
    /// tile whose `i1` is `(w, h, first value's bits)` (counted in `fails`). With more than one
    /// lane, the tile `slow.0` waits (up to 2 s) until a `fail_on` tile failed, records in `seen`
    /// how many refinements had started, and then fails if `slow.1`. `short_on` returns a 1 x 1
    /// result for that tile (the driver's write panics).
    #[derive(Default)]
    struct Watch {
        prepared_on: std::sync::Mutex<Vec<std::thread::ThreadId>>,
        refined: std::sync::atomic::AtomicUsize,
        with_extra: std::sync::atomic::AtomicUsize,
        fail_at: Option<usize>,
        panic_extra_at: Option<(usize, usize)>,
        first_here: bool,
        lanes: usize,
        fail_on: Option<((usize, usize, u32), bool)>,
        running: std::sync::atomic::AtomicUsize,
        most: std::sync::atomic::AtomicUsize,
        fails: std::sync::atomic::AtomicUsize,
        slow: Option<((usize, usize, u32), bool)>,
        seen: std::sync::atomic::AtomicUsize,
        short_on: Option<(usize, usize, u32)>,
    }

    impl FineEngine for Watch {
        fn refine_raw(
            &self,
            i1: &Image<f32>,
            i2: &Image<f32>,
            shift: (i64, i64),
            ux: &Image<f32>,
            uy: &Image<f32>,
            cfg: &Config,
        ) -> Result<RawFlow, EngineError> {
            CpuEngine.refine_raw(i1, i2, shift, ux, uy, cfg)
        }

        fn prepare_extra(
            &self,
            i1: &Image<f32>,
            _cfg: &Config,
        ) -> Option<Box<dyn std::any::Any + Send>> {
            self.prepared_on
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            if self.panic_extra_at == Some((i1.w, i1.h)) {
                panic!("prepare_extra fails on a {}x{} tile", i1.w, i1.h);
            }
            Some(Box::new((i1.w, i1.h)))
        }

        fn refine_prepared(&self, p: Prepared, cfg: &Config) -> Result<RawFlow, EngineError> {
            use std::sync::atomic::Ordering::SeqCst;
            let k = self.refined.fetch_add(1, SeqCst);
            if let Some(e) = &p.extra {
                assert_eq!(e.downcast_ref::<(usize, usize)>(), Some(&(p.i1.w, p.i1.h)));
                self.with_extra.fetch_add(1, SeqCst);
            }
            if self.fail_at == Some(k) {
                return Err(EngineError(format!("tile {k} fails")));
            }
            let id = (p.i1.w, p.i1.h, p.i1.data[0].to_bits());
            match self.fail_on {
                Some((at, panics)) if at == id => {
                    self.fails.fetch_add(1, SeqCst);
                    if panics {
                        panic!("tile {id:?} panics");
                    }
                    return Err(EngineError(format!("tile {id:?} fails")));
                }
                _ => {}
            }
            let now = self.running.fetch_add(1, SeqCst) + 1;
            self.most.fetch_max(now, SeqCst);
            let t = std::time::Instant::now();
            while self.lanes > 1
                && self.fail_on.is_none()
                && self.most.load(SeqCst) < 2
                && t.elapsed() < std::time::Duration::from_secs(5)
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if let Some((at, fails)) = self.slow.filter(|s| s.0 == id) {
                let t = std::time::Instant::now();
                while self.lanes > 1
                    && self.fails.load(SeqCst) == 0
                    && t.elapsed() < std::time::Duration::from_secs(2)
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                self.seen.store(self.refined.load(SeqCst), SeqCst);
                if fails {
                    self.running.fetch_sub(1, SeqCst);
                    return Err(EngineError(format!("tile {at:?} fails")));
                }
            }
            let raw = self.refine_raw(&p.i1, &p.i2, p.offset, &p.ux, &p.uy, cfg);
            self.running.fetch_sub(1, SeqCst);
            if self.short_on == Some(id) {
                let one = || Image::from_vec(1, 1, vec![0.0f64]);
                return Ok(RawFlow {
                    ux: one(),
                    uy: one(),
                    energy: one(),
                    resid: one(),
                });
            }
            raw
        }

        fn prefetch_first_tile(&self) -> bool {
            !self.first_here
        }

        fn lanes(&self) -> usize {
            self.lanes.max(1)
        }
    }

    /// `(w, h, first value's bits)` of the `i1` of each tile of a 700 x 450 image cut at 256, in
    /// the order they are refined (`Watch::fail_on`).
    fn tile_ids(a: &Image<f32>, cfg: &Config) -> Vec<(usize, usize, u32)> {
        let ts = tiles(a.w, a.h, 256, tile_margin(cfg));
        let area = |r: Rect| (r.x1 - r.x0) * (r.y1 - r.y0);
        let mut order: Vec<usize> = (0..ts.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(area(ts[k].ext)));
        order
            .iter()
            .map(|&k| {
                let e = ts[k].ext;
                (
                    e.x1 - e.x0,
                    e.y1 - e.y0,
                    a.data[e.y0 * a.w + e.x0].to_bits(),
                )
            })
            .collect()
    }

    /// Two tiles refined at once (`lanes`), with the first tile prepared ahead or here, give the
    /// bits of one at a time; two did run at once, never more, each tile once.
    #[test]
    fn two_lanes_give_the_same_bits_as_one_at_a_time() {
        use std::sync::atomic::Ordering::SeqCst;
        let (w, h) = (700, 450);
        let (a, b) = speckle_pair(w, h, (0.7, -0.4));
        let cfg = Config::default();
        let n_tiles = tiles(w, h, 256, tile_margin(&cfg)).len();
        let run = |prefetch: bool, first_here: bool, lanes: usize| {
            let e = Watch {
                first_here,
                lanes,
                ..Watch::default()
            };
            let out =
                refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, prefetch).unwrap();
            (out, e)
        };
        let ((f0, k0), e0) = run(false, false, 2);
        assert_eq!(
            e0.most.load(SeqCst),
            1,
            "without the prefetch one at a time"
        );
        for first_here in [false, true] {
            let ((f, k), e) = run(true, first_here, 2);
            assert_eq!(bits(&f0.ux), bits(&f.ux));
            assert_eq!(bits(&f0.uy), bits(&f.uy));
            assert_eq!(bits(&f0.conf), bits(&f.conf));
            assert_eq!(bits(&f0.resid), bits(&f.resid));
            assert_eq!(k0.data, k.data);
            assert_eq!(e.most.load(SeqCst), 2, "first_here {first_here}");
            assert_eq!(e.refined.load(SeqCst), n_tiles);
            let ahead = n_tiles - usize::from(first_here);
            assert_eq!(e.with_extra.load(SeqCst), ahead);
            let on = e.prepared_on.lock().unwrap().clone();
            assert_eq!(on.len(), ahead);
            assert!(on
                .iter()
                .all(|&id| id == on[0] && id != std::thread::current().id()));
        }
    }

    /// An engine error or a panic of one lane is the error or the panic of one at a time (the
    /// first failing tile in the order, whatever the other lane does), also when the producer
    /// panics after it; nothing hangs.
    #[test]
    fn an_error_or_a_panic_in_a_lane_is_that_of_one_at_a_time() {
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            let ids = tile_ids(&a, &cfg);
            let mut distinct = ids.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), ids.len());
            let mut out = Vec::new();
            for (pos, panics, panic_extra_at) in [
                (0, false, None),
                (2, false, None),
                (5, false, None),
                (1, true, None),
                (5, true, None),
                // the bottom-left tile (position 3) fails to prepare after position 2 failed
                (2, false, Some((320, 258))),
                (2, true, Some((320, 258))),
            ] {
                let run = |prefetch: bool, first_here: bool, lanes: usize| {
                    let e = Watch {
                        fail_on: Some((ids[pos], panics)),
                        panic_extra_at,
                        first_here,
                        lanes,
                        ..Watch::default()
                    };
                    let mut err = None;
                    let panic = panic_of(|| {
                        err =
                            refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, prefetch)
                                .err();
                    });
                    let refined = e.refined.load(std::sync::atomic::Ordering::SeqCst);
                    ((panic, err), refined)
                };
                out.push((
                    pos,
                    panics,
                    run(false, false, 1),
                    run(true, false, 2),
                    run(true, true, 2),
                ));
            }
            out
        });
        for (pos, panics, (one, n_one), (two, n_two), (here, n_here)) in got {
            if panics {
                assert!(
                    one.0.as_deref().is_some_and(|m| m.ends_with("panics")),
                    "{one:?}"
                );
                assert_eq!(one.1, None);
            } else {
                assert_eq!(one.0, None);
                assert!(
                    one.1.as_ref().is_some_and(|e| e.0.ends_with("fails")),
                    "{one:?}"
                );
            }
            assert_eq!(one, two, "position {pos}");
            assert_eq!(one, here, "position {pos}, the first tile here");
            // one at a time refines up to the failing tile; the other lane at most one more
            assert_eq!(n_one, pos + 1);
            for n in [n_two, n_here] {
                assert!(n == pos + 1 || n == pos + 2, "position {pos}: {n} refined");
            }
        }
    }

    /// A panic on the driver's thread (writing a tile's result of the wrong size) is resumed as
    /// one at a time, with the lanes and the producer stopped: nothing hangs.
    #[test]
    fn a_panic_writing_a_tile_is_that_of_one_at_a_time_and_nothing_hangs() {
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            let ids = tile_ids(&a, &cfg);
            let run = |first_here: bool, lanes: usize| {
                let e = Watch {
                    short_on: Some(ids[0]),
                    first_here,
                    lanes,
                    ..Watch::default()
                };
                panic_of(|| refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, true))
            };
            [run(false, 1), run(false, 2), run(true, 2)]
        });
        assert!(
            got[0]
                .as_deref()
                .is_some_and(|m| m.contains("out of bounds")),
            "{got:?}"
        );
        assert_eq!(got[0], got[1]);
        assert_eq!(got[0], got[2]);
    }

    /// A slow tile and a later one failing fast while it runs: the outcome is that of one at a
    /// time (the later tile's error or panic, or the slow tile's error if it fails too).
    #[test]
    fn a_slow_tile_before_a_failing_one_gives_the_outcome_of_one_at_a_time() {
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            let ids = tile_ids(&a, &cfg);
            let mut out = Vec::new();
            for (slow_fails, panics) in [(false, false), (false, true), (true, false), (true, true)]
            {
                let run = |prefetch: bool, first_here: bool, lanes: usize| {
                    let e = Watch {
                        fail_on: Some((ids[1], panics)),
                        slow: Some((ids[0], slow_fails)),
                        first_here,
                        lanes,
                        ..Watch::default()
                    };
                    let mut err = None;
                    let panic = panic_of(|| {
                        err =
                            refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, prefetch)
                                .err();
                    });
                    (panic, err.map(|e| e.0))
                };
                out.push((
                    slow_fails,
                    panics,
                    ids[usize::from(!slow_fails)],
                    run(false, false, 1),
                    run(true, false, 2),
                    run(true, true, 2),
                ));
            }
            out
        });
        for (slow_fails, panics, id, one, two, here) in got {
            if panics && !slow_fails {
                assert_eq!(one, (Some(format!("tile {id:?} panics")), None));
            } else {
                assert_eq!(one, (None, Some(format!("tile {id:?} fails"))));
            }
            assert_eq!(one, two, "slow fails {slow_fails}, panics {panics}");
            assert_eq!(
                one, here,
                "slow fails {slow_fails}, panics {panics}, first here"
            );
        }
    }

    /// With two lanes and the first tile slow, the other lane refines the second tile and then
    /// waits until the first is written: at most `lanes` positions ahead of the writes.
    #[test]
    fn a_lane_takes_a_tile_only_when_the_one_lanes_earlier_is_written() {
        use std::sync::atomic::Ordering::SeqCst;
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            let ids = tile_ids(&a, &cfg);
            [false, true].map(|first_here| {
                let e = Watch {
                    slow: Some((ids[0], false)),
                    first_here,
                    lanes: 2,
                    ..Watch::default()
                };
                refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, true).unwrap();
                (e.seen.load(SeqCst), e.refined.load(SeqCst))
            })
        });
        assert_eq!(got, [(2, 6), (2, 6)]);
    }

    /// A panic on a lane after the refinement (taking the tile's trace) is that tile's failure,
    /// resumed with its payload after the earlier tiles are written.
    #[test]
    fn a_panic_taking_a_trace_on_a_lane_is_that_tiles_failure() {
        struct Zeros;
        impl FineEngine for Zeros {
            fn refine_raw(
                &self,
                i1: &Image<f32>,
                _i2: &Image<f32>,
                _shift: (i64, i64),
                _ux: &Image<f32>,
                _uy: &Image<f32>,
                _cfg: &Config,
            ) -> Result<RawFlow, EngineError> {
                let z = || Image::from_vec(i1.w, i1.h, vec![0.0f64; i1.w * i1.h]);
                Ok(RawFlow {
                    ux: z(),
                    uy: z(),
                    energy: z(),
                    resid: z(),
                })
            }
        }
        let got = within_a_minute(|| {
            [false, true].map(|first_here| {
                let order: Vec<usize> = (0..8).collect();
                let mut written = Vec::new();
                let prepare = |_: usize, _: bool| Prepared {
                    i1: constant(8, 8, 0.0),
                    i2: constant(8, 8, 0.0),
                    offset: (0, 0),
                    ux: constant(8, 8, 0.0),
                    uy: constant(8, 8, 0.0),
                    extra: None,
                };
                let taken = |pos: usize| {
                    assert_ne!(pos, 3, "taking the trace of position 3 panics");
                    None
                };
                let cfg = Config::default();
                let panic = panic_of(|| {
                    refine_in_lanes(
                        &Zeros,
                        2,
                        &order,
                        first_here,
                        prepare,
                        taken,
                        &cfg,
                        |pos, _, _| written.push(pos),
                    )
                });
                (panic, written)
            })
        });
        for (panic, written) in got {
            assert!(
                panic
                    .as_deref()
                    .is_some_and(|m| m.contains("position 3 panics")),
                "{panic:?}"
            );
            assert_eq!(written, [0, 1, 2]);
        }
    }

    /// The CPU engine with `energy` and `resid` rounded to `f32` and widened again, as the GPU
    /// engine returns them; `exact` is what it answers to `planes_exact_in_f32`.
    struct Widened {
        exact: bool,
    }

    impl FineEngine for Widened {
        fn refine_raw(
            &self,
            i1: &Image<f32>,
            i2: &Image<f32>,
            shift: (i64, i64),
            ux: &Image<f32>,
            uy: &Image<f32>,
            cfg: &Config,
        ) -> Result<RawFlow, EngineError> {
            let mut raw = CpuEngine.refine_raw(i1, i2, shift, ux, uy, cfg)?;
            for v in raw.energy.data.iter_mut().chain(&mut raw.resid.data) {
                *v = f64::from(*v as f32);
            }
            Ok(raw)
        }

        fn planes_exact_in_f32(&self) -> bool {
            self.exact
        }
    }

    #[test]
    fn keeping_widened_f32_planes_as_f32_gives_the_same_bits() {
        let (w, h) = (700, 450);
        let (a, b) = speckle_pair(w, h, (0.7, -0.4));
        let cfg = Config::default();
        let run = |exact: bool| {
            refine_tiled_from_on_with(&Widened { exact }, &a, &b, &zero_start, &cfg, 256, true)
                .unwrap()
        };
        let ((f64s, low64), (f32s, low32)) = (run(false), run(true));
        assert_eq!(bits(&f64s.ux), bits(&f32s.ux));
        assert_eq!(bits(&f64s.uy), bits(&f32s.uy));
        assert_eq!(bits(&f64s.conf), bits(&f32s.conf));
        assert_eq!(bits(&f64s.resid), bits(&f32s.resid));
        assert_eq!(low64.data, low32.data);
    }

    #[test]
    fn prefetching_tiles_gives_the_same_bits_as_one_at_a_time() {
        use std::sync::atomic::Ordering::SeqCst;
        // an image of 3 x 2 tiles with the CPU engine, prefetch on and off
        let (w, h) = (700, 450);
        let (a, b) = speckle_pair(w, h, (0.7, -0.4));
        let cfg = Config::default();
        let n_tiles = tiles(w, h, 256, tile_margin(&cfg)).len();
        assert_eq!(n_tiles, 6);
        let run = |prefetch: bool| {
            let e = Watch::default();
            let out =
                refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, prefetch).unwrap();
            (out, e)
        };
        let ((f0, k0), e0) = run(false);
        let ((f1, k1), e1) = run(true);
        assert_eq!(bits(&f0.ux), bits(&f1.ux));
        assert_eq!(bits(&f0.uy), bits(&f1.uy));
        assert_eq!(bits(&f0.conf), bits(&f1.conf));
        assert_eq!(bits(&f0.resid), bits(&f1.resid));
        assert_eq!(k0.data, k1.data);
        // and equal to the plain CPU engine
        let (fc, _) = refine_tiled_from(&a, &b, &zero_start, &cfg, 256);
        assert_eq!(bits(&f0.ux), bits(&fc.ux));
        // off: no extra, one at a time on this thread
        assert!(e0.prepared_on.lock().unwrap().is_empty());
        assert_eq!(e0.with_extra.load(SeqCst), 0);
        assert_eq!(e0.refined.load(SeqCst), n_tiles);
        // on: every tile prepared, with its extra, on one other thread
        let on = e1.prepared_on.lock().unwrap().clone();
        assert_eq!(on.len(), n_tiles);
        assert!(on
            .iter()
            .all(|&id| id == on[0] && id != std::thread::current().id()));
        assert_eq!(e1.with_extra.load(SeqCst), n_tiles);
        assert_eq!(e1.refined.load(SeqCst), n_tiles);
    }

    /// Without `prefetch_first_tile` the first tile is prepared on the caller's thread without an
    /// extra and the rest ahead with theirs; the bits are those of one at a time.
    #[test]
    fn the_first_tile_prepared_here_gives_the_same_bits_as_one_at_a_time() {
        use std::sync::atomic::Ordering::SeqCst;
        let (w, h) = (700, 450);
        let (a, b) = speckle_pair(w, h, (0.7, -0.4));
        let cfg = Config::default();
        let n_tiles = tiles(w, h, 256, tile_margin(&cfg)).len();
        let run = |prefetch: bool, first_here: bool| {
            let e = Watch {
                first_here,
                ..Watch::default()
            };
            let out =
                refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, prefetch).unwrap();
            (out, e)
        };
        let ((f0, k0), _) = run(false, false);
        let ((f1, k1), e1) = run(true, true);
        let ((f2, k2), _) = run(true, false);
        for (f, k) in [(&f1, &k1), (&f2, &k2)] {
            assert_eq!(bits(&f0.ux), bits(&f.ux));
            assert_eq!(bits(&f0.uy), bits(&f.uy));
            assert_eq!(bits(&f0.conf), bits(&f.conf));
            assert_eq!(bits(&f0.resid), bits(&f.resid));
            assert_eq!(k0.data, k.data);
        }
        // all but the first prepared ahead, on one other thread, with their extras
        let on = e1.prepared_on.lock().unwrap().clone();
        assert_eq!(on.len(), n_tiles - 1);
        assert!(on
            .iter()
            .all(|&id| id == on[0] && id != std::thread::current().id()));
        assert_eq!(e1.with_extra.load(SeqCst), n_tiles - 1);
        assert_eq!(e1.refined.load(SeqCst), n_tiles);
        // off: `first_here` changes nothing
        let ((f3, _), e3) = run(false, true);
        assert_eq!(bits(&f0.ux), bits(&f3.ux));
        assert!(e3.prepared_on.lock().unwrap().is_empty());
    }

    /// A panic forming a later tile's start field (on the prefetch thread) is the same panic as
    /// one at a time, and nothing hangs.
    #[test]
    fn a_panic_preparing_a_tile_ahead_is_the_same_panic_as_one_at_a_time() {
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            // the bottom-left tile is neither the first nor the last of the order
            let start = |r: Rect| {
                if r.x0 == 0 && r.y0 > 0 {
                    panic!("no start field at ({}, {})", r.x0, r.y0);
                }
                zero_start(r)
            };
            [(false, false), (true, false), (true, true)].map(|(prefetch, first_here)| {
                let e = Watch {
                    first_here,
                    ..Watch::default()
                };
                let p =
                    panic_of(|| refine_tiled_from_on_with(&e, &a, &b, &start, &cfg, 256, prefetch));
                (p, e.refined.load(std::sync::atomic::Ordering::SeqCst))
            })
        });
        assert_eq!(got[0].0.as_deref(), Some("no start field at (0, 192)"));
        assert_eq!(got[0], got[1], "one at a time vs prefetch");
        assert_eq!(got[0], got[2], "one at a time vs the first tile here");
        assert!(got[0].1 > 0 && got[0].1 < 6, "{got:?}");
    }

    /// A panic in `prepare_extra` on the prefetch thread reaches the caller, without a hang.
    #[test]
    fn a_panic_in_prepare_extra_reaches_the_caller() {
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            let e = Watch {
                panic_extra_at: Some((320, 258)), // the bottom-left tile's extension
                ..Watch::default()
            };
            panic_of(|| refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, true))
        });
        assert_eq!(
            got.as_deref(),
            Some("prepare_extra fails on a 320x258 tile")
        );
    }

    /// An engine error returns early with the same error, prefetch on or off, also when the
    /// prefetch thread panics after it (the error comes first, as one at a time); nothing hangs.
    /// In the last case the engine fails at order index 2 while the producer prepares index 3,
    /// the bottom-left tile, whose `prepare_extra` panics: so the panic happens whatever the timing.
    #[test]
    fn an_engine_error_with_prefetch_is_the_error_of_one_at_a_time() {
        let got = within_a_minute(|| {
            let (a, b) = speckle_pair(700, 450, (0.7, -0.4));
            let cfg = Config {
                outer_iters: 1,
                ..Config::default()
            };
            let mut out = Vec::new();
            for (fail_at, panic_extra_at) in
                [(0, None), (2, None), (5, None), (2, Some((320, 258)))]
            {
                let run = |prefetch: bool, first_here: bool| {
                    let e = Watch {
                        fail_at: Some(fail_at),
                        panic_extra_at,
                        first_here,
                        ..Watch::default()
                    };
                    let r = refine_tiled_from_on_with(&e, &a, &b, &zero_start, &cfg, 256, prefetch);
                    let prepared = e.prepared_on.lock().unwrap().len();
                    let refined = e.refined.load(std::sync::atomic::Ordering::SeqCst);
                    ((r.err(), refined), prepared)
                };
                out.push((
                    fail_at,
                    panic_extra_at,
                    run(false, false),
                    run(true, false),
                    run(true, true),
                ));
            }
            out
        });
        for (
            fail_at,
            panic_extra_at,
            (off, off_prepared),
            (on, on_prepared),
            (here, here_prepared),
        ) in got
        {
            assert_eq!(off.0, Some(EngineError(format!("tile {fail_at} fails"))));
            assert_eq!(off, on, "fail at {fail_at}");
            assert_eq!(off, here, "fail at {fail_at}, the first tile here");
            assert_eq!(off_prepared, 0);
            if panic_extra_at.is_some() {
                // tiles 0..=2 handed over, then tile 3 prepared (recorded before its panic);
                // with the first tile here, tiles 1..=3
                assert_eq!(on_prepared, 4, "the producer reached the panicking tile");
                assert_eq!(here_prepared, 3, "the producer reached the panicking tile");
            }
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// The tiles are the product of the per-axis tiles (`tiles` is built from `tile_axis`), so the
    /// partition of the image follows from the partition of one axis, proved here for a symbolic
    /// tile index instead of a loop over the returned `Vec`: cores abut, start at 0 and end at the
    /// axis length, every pixel is in the core of tile `p / t` and in no other, each core lies in
    /// its extension and each extension within the axis.
    #[kani::proof]
    #[kani::unwind(5)]
    fn k8_tiles_partition_the_image() {
        let n: usize = kani::any();
        let t: usize = kani::any();
        let m: usize = kani::any();
        kani::assume(n >= 1 && n <= 32 && t >= 4 && t <= 16 && m <= 8);
        let count = tile_count(n, t);
        assert!(count >= 1);
        let k: usize = kani::any();
        kani::assume(k < count);
        let (c0, c1, e0, e1) = tile_axis(n, t, m, k);
        assert!(e0 <= c0 && c0 < c1 && c1 <= e1 && e1 <= n);
        assert!(c0 - e0 <= m && e1 - c1 <= m);
        assert!(tile_axis(n, t, m, 0).0 == 0);
        assert!(tile_axis(n, t, m, count - 1).1 == n);
        if k + 1 < count {
            assert!(c1 == tile_axis(n, t, m, k + 1).0);
        }
        let p: usize = kani::any();
        kani::assume(p < n);
        let kp = p / t;
        assert!(kp < count);
        let (p0, p1, _, _) = tile_axis(n, t, m, kp);
        assert!(p0 <= p && p < p1);
        // no other tile contains p
        if c0 <= p && p < c1 {
            assert!(k == kp);
        }
    }
}

//! Change of the fine-stage field per outer iteration, for measurement only (spec
//! 2026-11-01-iteration-convergence).
//!
//! Off unless `OPTFLOW_ITER_TRACE` names a file (read once per process): then [`enabled`] is
//! false and the engines and drivers do nothing else, so the output is the same bit for bit.
//! On, an engine hands the field of its window to [`begin`] (the start) and to [`outer`] after
//! every outer iteration, with that iteration's wall time (the GPU engine reads its buffers back
//! for it, so the stage and fine times of a traced run are not valid); the driver keeps
//! `|u_k - u_(k-1)|` (px, `u_0` the start field) of each tile core and, once the confidence of
//! the fine stage is known, appends one JSON line per fine stage to the file: per tile and over
//! all tiles, per outer iteration, max, p99 and RMS over the trusted pixels (`conf >= conf_min`)
//! of the core.

use crate::image::Image;
use crate::tile::Rect;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Whether `OPTFLOW_ITER_TRACE` names a file.
pub fn enabled() -> bool {
    PATH.get_or_init(|| {
        std::env::var_os("OPTFLOW_ITER_TRACE")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    })
    .is_some()
}

/// The window an engine is refining: the last field and `|du|` per pixel and outer iteration.
pub(crate) struct Window {
    w: usize,
    prev: (Vec<f64>, Vec<f64>),
    deltas: Vec<Vec<f32>>,
    secs: Vec<f64>,
}

// the window of each thread refining one (an engine refines on its caller's thread; the tiled
// driver may refine two tiles at once on two threads, `FineEngine::lanes`)
static WINDOW: Mutex<Option<HashMap<ThreadId, Window>>> = Mutex::new(None);
// whether the fine stage being traced starts from the last pair's field (a sequence)
static WARM: Mutex<Option<bool>> = Mutex::new(None);
static STAGES: AtomicUsize = AtomicUsize::new(0);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn me() -> ThreadId {
    std::thread::current().id()
}

/// The start field of a window of width `w` (of this thread).
#[doc(hidden)]
pub fn begin(w: usize, ux: &[f32], uy: &[f32]) {
    let wide = |u: &[f32]| u.iter().map(|&v| f64::from(v)).collect();
    let window = Window {
        w,
        prev: (wide(ux), wide(uy)),
        deltas: Vec::new(),
        secs: Vec::new(),
    };
    lock(&WINDOW)
        .get_or_insert_with(HashMap::new)
        .insert(me(), window);
}

/// The field after an outer iteration that took `secs` (of this thread's window).
#[doc(hidden)]
pub fn outer(ux: Vec<f64>, uy: Vec<f64>, secs: f64) {
    let mut g = lock(&WINDOW);
    let win = g
        .as_mut()
        .and_then(|m| m.get_mut(&me()))
        .expect("iter_trace::outer before begin");
    let d = ux
        .iter()
        .zip(&uy)
        .zip(win.prev.0.iter().zip(&win.prev.1))
        .map(|((a, b), (p, q))| (a - p).hypot(b - q) as f32)
        .collect();
    win.deltas.push(d);
    win.secs.push(secs);
    win.prev = (ux, uy);
}

/// Whether the next fine stage starts from the last pair's field.
pub(crate) fn set_warm(warm: bool) {
    *lock(&WARM) = Some(warm);
}

/// One tile of a traced fine stage: its core (image coordinates) and `|du|` on it.
pub(crate) struct TileTrace {
    core: Rect,
    deltas: Vec<Vec<f32>>,
    secs: Vec<f64>,
}

/// The window this thread just refined, cut to `core` (given in the window's coordinates `ext`).
pub(crate) fn take(core: Rect, ext: Rect) -> Option<TileTrace> {
    let win = lock(&WINDOW).as_mut()?.remove(&me())?;
    let cut = |d: &Vec<f32>| {
        let mut out = Vec::with_capacity((core.x1 - core.x0) * (core.y1 - core.y0));
        for y in core.y0..core.y1 {
            let row = (y - ext.y0) * win.w;
            out.extend_from_slice(&d[row + core.x0 - ext.x0..row + core.x1 - ext.x0]);
        }
        out
    };
    Some(TileTrace {
        core,
        deltas: win.deltas.iter().map(cut).collect(),
        secs: win.secs,
    })
}

/// max, p99, RMS and count of `v` (only the count when empty); `v` is reordered.
fn stats(v: &mut [f32]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::json!({"n": 0});
    }
    let n = v.len();
    let max = v.iter().fold(0.0f32, |a, &b| a.max(b));
    let rms = (v.iter().map(|&d| f64::from(d).powi(2)).sum::<f64>() / n as f64).sqrt();
    let k = ((0.99 * (n - 1) as f64).round() as usize).min(n - 1);
    let (_, p99, _) = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    serde_json::json!({"n": n, "max": max, "p99": *p99, "rms": rms})
}

/// Appends the line of a fine stage whose tiles are `tiles` and whose low-confidence pixels
/// (image of width `low.w`) are `low`.
pub(crate) fn finish(tiles: Vec<TileTrace>, low: &Image<bool>) {
    let Some(Some(path)) = PATH.get() else {
        return;
    };
    let w = low.w;
    let iters = tiles.iter().map(|t| t.deltas.len()).max().unwrap_or(0);
    let trusted = |t: &TileTrace, d: &[f32]| -> Vec<f32> {
        let cw = t.core.x1 - t.core.x0;
        d.iter()
            .enumerate()
            .filter(|&(i, _)| !low.data[(t.core.y0 + i / cw) * w + t.core.x0 + i % cw])
            .map(|(_, &v)| v)
            .collect()
    };
    let mut all: Vec<Vec<f32>> = vec![Vec::new(); iters];
    let mut per_tile = Vec::new();
    for t in &tiles {
        let mut its = Vec::new();
        for (k, d) in t.deltas.iter().enumerate() {
            let mut v = trusted(t, d);
            all[k].extend_from_slice(&v);
            its.push(stats(&mut v));
        }
        per_tile.push(serde_json::json!({
            "core": [t.core.x0, t.core.y0, t.core.x1, t.core.y1],
            "pixels": (t.core.x1 - t.core.x0) * (t.core.y1 - t.core.y0),
            "secs": t.secs,
            "iters": its,
        }));
    }
    let line = serde_json::json!({
        "stage": STAGES.fetch_add(1, Ordering::Relaxed),
        "warm": lock(&WARM).take(),
        "w": w,
        "h": low.h,
        "iters": all.iter_mut().map(|v| stats(v)).collect::<Vec<_>>(),
        "tiles": per_tile,
    });
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("OPTFLOW_ITER_TRACE {}: {e}", path.display()));
    writeln!(f, "{line}").expect("OPTFLOW_ITER_TRACE: write");
}

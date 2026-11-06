//! Wall-clock time per pipeline stage, summed over all calls since the last [`reset`].
//!
//! For benchmarks (`examples/bench_pair.rs`): each stage is timed where the pipeline runs it,
//! around the whole (possibly parallel) step, so the sums are wall time of a single pipeline run.
//! The counters are process-wide; with several pipelines running at once they mix. Timing does
//! not touch any value the pipeline computes.
//!
//! A probe ([`set_probe`]) is told every stage boundary and tile of the tiled fine stage, so a
//! benchmark can sample its own counters there (the memory profile of `bench_pair --mem`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// The timed stages, in pipeline order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Coarse POC grid on the pyramid and the dense start field (for an image of several
    /// tiles, the start of each tile, formed when the fine stage gets to it; the margins shared
    /// by neighbouring tiles are formed, and counted, twice).
    Coarse,
    /// Filter bank and the reference phase of the first image.
    Reference,
    /// Trusted pre-warp from the second outer iteration on: the confidence of the last solve and
    /// the extrapolation of the untrusted displacements (per outer iteration).
    Prewarp,
    /// Gaussian smoothing of the field for the pre-warp and its spline (per outer iteration).
    Smooth,
    /// Pre-warp of the second image (its spline and the sampling).
    Warp,
    /// Bank responses of the warped image: FFT, filters, inverse FFTs and their splines.
    Responses,
    /// Sampling of the responses and the per-pixel sums over the filters (per inner iteration).
    Accumulate,
    /// Window moments of the sums and the local systems, fused row by row.
    Moments,
    /// The energy threshold of the local solve (median over the image).
    Solve,
    /// Confidence of the fine stage.
    Confidence,
    /// Validity, ROI and guard band of the result.
    Guard,
    /// Strain of the result.
    Strain,
    /// `crack_model = "xdic"`: the baseline's validity and strain for the detector, the jump test
    /// and the crack lines made of its points (`xdic::detect_lines`).
    XdicDetect,
    /// `crack_model = "xdic"`: the reference responses of I1 with the other side painted and the
    /// responses of the side's own pre-warped I2 (per side, per outer iteration).
    XdicResponses,
    /// `crack_model = "xdic"`: the extrapolation of each side, the per-side moments and the
    /// per-side solve (the baseline iterations of `refine_xdic` are timed by the stages above).
    XdicSolve,
    /// `crack_model = "xdic"`: the geometry update (`update_geometry`) and the fields of the new lines.
    XdicGeometry,
    /// `crack_model = "xdic"`: crack or band (`model_gain`), `advance_lines`, `lines_to_frame0` and
    /// `xdic::finish` (guard, one-sided strain, COD / CSD along the lines).
    XdicFinish,
}

/// Every stage, in pipeline order.
pub const STAGES: [Stage; 17] = [
    Stage::Coarse,
    Stage::Reference,
    Stage::Prewarp,
    Stage::Smooth,
    Stage::Warp,
    Stage::Responses,
    Stage::Accumulate,
    Stage::Moments,
    Stage::Solve,
    Stage::Confidence,
    Stage::Guard,
    Stage::Strain,
    Stage::XdicDetect,
    Stage::XdicResponses,
    Stage::XdicSolve,
    Stage::XdicGeometry,
    Stage::XdicFinish,
];

static NANOS: [AtomicU64; STAGES.len()] = [const { AtomicU64::new(0) }; STAGES.len()];

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Coarse => "coarse POC",
            Stage::Reference => "reference phase",
            Stage::Prewarp => "pre-warp rule",
            Stage::Smooth => "smooth field",
            Stage::Warp => "warp",
            Stage::Responses => "responses",
            Stage::Accumulate => "accumulate",
            Stage::Moments => "moments+solve",
            Stage::Solve => "solve threshold",
            Stage::Confidence => "confidence",
            Stage::Guard => "guard",
            Stage::Strain => "strain",
            Stage::XdicDetect => "xdic detect",
            Stage::XdicResponses => "xdic responses",
            Stage::XdicSolve => "xdic side solve",
            Stage::XdicGeometry => "xdic geometry",
            Stage::XdicFinish => "xdic finish",
        }
    }
}

/// A point of the pipeline reported to the probe of [`set_probe`]. For the benchmarks only.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// A stage starts.
    Enter(Stage),
    /// A stage ends.
    Exit(Stage),
    /// Tile `k` of the tiled fine stage starts.
    TileStart(usize),
    /// Tile `k` of the tiled fine stage ends.
    TileEnd(usize),
}

static PROBE: OnceLock<fn(Mark)> = OnceLock::new();

/// Installs `probe`, called at every [`Mark`] from the thread that runs the pipeline. At most one
/// probe per process: false if one is installed already. For the benchmarks only (the memory
/// profile of `examples/common/mem.rs`); nothing in the pipeline depends on it. Exceptions: the
/// tiled fine stage's prefetch thread (`OPTFLOW_TILE_PREFETCH`, on by default) forms the next
/// tile's start field, with its `Coarse` marks (and, on the GPU, `Reference` for the next tile's
/// reference pack), while the pipeline thread is inside the current tile; the GPU engine's
/// reference-pack thread marks `Reference` the same way. With a probe installed the tiles are
/// refined one at a time ([`crate::engine::FineEngine::lanes`] is not used), so the tile marks do
/// not overlap and the memory between them is that tile's (the same bits).
#[doc(hidden)]
pub fn set_probe(probe: fn(Mark)) -> bool {
    PROBE.set(probe).is_ok()
}

/// Whether a probe is installed ([`set_probe`]).
pub(crate) fn probing() -> bool {
    PROBE.get().is_some()
}

fn mark(m: Mark) {
    if let Some(probe) = PROBE.get() {
        probe(m);
    }
}

/// Runs `f` and adds its wall time to `stage`.
pub fn time<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
    let t = start(stage);
    let out = f();
    add(stage, t);
    out
}

/// `stage` starts: the time to pass to [`add`] when it ends.
pub(crate) fn start(stage: Stage) -> Instant {
    mark(Mark::Enter(stage));
    Instant::now()
}

/// Adds the wall time since `start` (from the crate's `start`) to `stage`, which ends.
pub fn add(stage: Stage, start: Instant) {
    let ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    NANOS[stage as usize].fetch_add(ns, Ordering::Relaxed);
    mark(Mark::Exit(stage));
}

/// Tile `k` of the tiled fine stage starts (`true`) or ends.
pub(crate) fn tile(k: usize, starts: bool) {
    mark(if starts {
        Mark::TileStart(k)
    } else {
        Mark::TileEnd(k)
    });
}

/// Sets every stage's sum to 0.
pub fn reset() {
    for n in &NANOS {
        n.store(0, Ordering::Relaxed);
    }
}

/// Seconds per stage since the last [`reset`], in the order of [`STAGES`].
pub fn snapshot() -> [f64; STAGES.len()] {
    let mut out = [0.0; STAGES.len()];
    for (o, n) in out.iter_mut().zip(&NANOS) {
        *o = n.load(Ordering::Relaxed) as f64 * 1e-9;
    }
    out
}

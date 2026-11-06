//! The hook of the fine stage: the iterations of [`crate::phaseflow::refine_raw`] on one window
//! can be run by another engine (the GPU backend) in place of the CPU code.

use crate::config::Config;
use crate::image::Image;
use crate::phaseflow::RawFlow;

/// What a tile needs before its refinement, made off the engine's thread (one tile ahead, see
/// `tile::refine_tiled_from_on`): the arguments of [`FineEngine::refine_raw`] (`offset` is its
/// `shift`) and the engine's own data from [`FineEngine::prepare_extra`].
pub struct Prepared {
    pub i1: Image<f32>,
    pub i2: Image<f32>,
    pub offset: (i64, i64),
    pub ux: Image<f32>,
    pub uy: Image<f32>,
    /// engine-specific data made from i1 alone (the GPU engine: the reference pack); None = none
    pub extra: Option<Box<dyn std::any::Any + Send>>,
}

/// Why an engine could not run the iterations (a device error, a window it cannot hold).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct EngineError(pub String);

/// The iterations of the fine stage on one window (`phaseflow::refine_raw`'s contract: `i1`,
/// `ux`, `uy` cover the window, `i2` the part of the second image it needs, `shift` places it).
/// The pipeline calls `refine_raw` for one window at a time (or [`FineEngine::lanes`] at once);
/// an engine need not accept more concurrent calls (the GPU engine refuses them with an error),
/// so do not call it from rayon workers.
pub trait FineEngine: Sync {
    fn refine_raw(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<RawFlow, EngineError>;

    /// Called once before tracking with the largest window the pipeline will pass (w, h), so an
    /// engine can refuse up front (GPU spec §6). The CPU accepts every size.
    fn check_window(&self, _w: usize, _h: usize, _cfg: &Config) -> Result<(), EngineError> {
        Ok(())
    }

    /// Engine-specific preparation from `i1` (may run on another thread while the engine works on
    /// the previous tile, so it must not need the engine's own lock). Default: none.
    fn prepare_extra(
        &self,
        _i1: &Image<f32>,
        _cfg: &Config,
    ) -> Option<Box<dyn std::any::Any + Send>> {
        None
    }

    /// Whether the tiled driver prepares the first tile of a call ahead too (on its prefetch
    /// thread, with the extra). An engine that makes the extra itself while its own work runs
    /// (the GPU: the reference pack during the first outer iteration) answers false: the first
    /// tile is then prepared on the caller's thread without an extra, so the engine starts at
    /// once, and the prefetch begins with the second tile. Either way the output is the same.
    /// Default: true.
    fn prefetch_first_tile(&self) -> bool {
        true
    }

    /// How many tiles the tiled driver may refine at once, each on a thread of its own (with the
    /// prefetch on; without it one at a time). An engine answering more than 1 must accept that
    /// many concurrent `refine_prepared` calls, and each call's result must not depend on the
    /// others. The cores are still joined in the order of one at a time, so the output is the
    /// same. Default: 1.
    fn lanes(&self) -> usize {
        1
    }

    /// Refinement with a prepared tile. Default: `refine_raw` (the extra is ignored).
    fn refine_prepared(&self, p: Prepared, cfg: &Config) -> Result<RawFlow, EngineError> {
        self.refine_raw(&p.i1, &p.i2, p.offset, &p.ux, &p.uy, cfg)
    }

    /// Whether [`FineEngine::refine_raw_with_slope`] works (the CPU code), i.e. whether
    /// `crack_model = "xdic"` can run on this engine.
    fn returns_slope(&self) -> bool {
        false
    }

    /// Whether this engine is the CPU code itself, so that a leg of `crack_model = "xdic"` can
    /// keep what the baseline prepared (the filter bank, the reference, the spline) for its
    /// per-side solve instead of building it again. Default: no.
    fn is_cpu_code(&self) -> bool {
        false
    }

    /// [`FineEngine::refine_raw`] on the whole image, and the `slope` of its last local solve
    /// (`[dux/dx, duy/dx, dux/dy, duy/dy]` per pixel, row-major), the baseline of
    /// `crack_model = "xdic"` (the per-side solve warm-starts from it and extrapolates with it).
    /// Default: an error; only the CPU code returns the slope.
    fn refine_raw_with_slope(
        &self,
        _i1: &Image<f32>,
        _i2: &Image<f32>,
        _ux: &Image<f32>,
        _uy: &Image<f32>,
        _cfg: &Config,
    ) -> Result<(RawFlow, Vec<[f64; 4]>), EngineError> {
        Err(EngineError(
            "crack_model 'xdic' runs on the CPU only (the per-side solve needs the slope of the \
             baseline solve, which this engine does not return); use --device cpu"
                .to_string(),
        ))
    }

    /// Frees what the engine keeps between windows (caches, pooled buffers) once the caller is
    /// done with the fine stage of a pair; the next `refine_raw` rebuilds what it needs and gives
    /// the same bits. Default: nothing to free.
    fn release_caches(&self) {}

    /// Whether the `energy` and `resid` of every [`RawFlow`] this engine returns are `f32` values
    /// widened to `f64`, so that the tiled driver may keep them as `f32` (half the memory) and
    /// widen them again with the same bits. Default: no (the CPU's are genuine `f64`).
    fn planes_exact_in_f32(&self) -> bool {
        false
    }
}

/// The CPU code: [`crate::phaseflow::refine_raw`] itself.
pub struct CpuEngine;

impl FineEngine for CpuEngine {
    fn refine_raw(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<RawFlow, EngineError> {
        Ok(crate::phaseflow::refine_raw(i1, i2, shift, ux, uy, cfg))
    }

    fn returns_slope(&self) -> bool {
        true
    }

    fn is_cpu_code(&self) -> bool {
        true
    }

    fn refine_raw_with_slope(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<(RawFlow, Vec<[f64; 4]>), EngineError> {
        Ok(crate::phaseflow::refine_raw_slope(
            i1,
            i2,
            (0, 0),
            ux,
            uy,
            cfg,
        ))
    }
}

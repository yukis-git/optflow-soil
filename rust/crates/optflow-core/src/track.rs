//! End-to-end tracking of image pairs and sequences (Lagrangian, cumulative).
//!
//! Port of `python/optflow_ref/track.py`; that file is the specification. A pair runs the coarse
//! stage on the whole image, the fine stage (tiled when the image is large, see [`crate::tile`]),
//! and then turns the confidence into validity: low confidence, the ROI exterior, a guard band
//! around large invalid regions and the image border become NaN in the output, and the strain is
//! computed from what remains. A sequence tracks every frame directly against a reference frame
//! and composes the displacements back to frame 0.
//!
//! Storage is `f32` as everywhere; the bilinear sampling and the accumulation of the link
//! confidence run in `f64` like the reference.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::Serialize;

use crate::config::{Config, ConfigError};
use crate::edt::{distance_to_invalid, large_components, within};
use crate::engine::{CpuEngine, EngineError, FineEngine};
use crate::image::Image;
use crate::phaseflow::{border_zone, confidence, refine_low, reportable, texture_scale};
use crate::poc::{coarse_flow, grid_sample, Grid};
use crate::prof::{self, Stage};
use crate::strain::{displacement_gradient, strain_fields_with, Gradient, Strain};
#[cfg(test)]
use crate::tile::refine_tiled_from;
use crate::tile::{refine_tiled_from_on, tile_margin, Rect};

mod hybrid;
pub mod resume;
mod xdic_leg;

use crate::xdic::{CrackOut, Poly};
use xdic_leg::XLeg;

/// Images with a side longer than this many pixels get their fine stage in tiles of this size.
pub const TILE: usize = 1024;

/// Most rows per band of the strain and guard computation. Both only look a bounded number of
/// rows away, so the bands (with a halo) give the result of the whole image with a fraction of
/// the memory.
pub const BAND_ROWS: usize = 512;

/// Pixels per band of the strain and guard computation: a wide image gets fewer rows per band
/// (see [`band_rows`]).
pub const BAND_PIXELS: usize = 1 << 19;

/// Rows per band for an image `w` pixels wide: `BAND_PIXELS / w`, at least 32 and at most
/// [`BAND_ROWS`]. The strain of a band holds about 30 `f64` maps of the band and its halo
/// (`strain_win / 2 + 1` rows either way), 790 MB for 512 rows of a 6240 px wide image and 150 MB
/// for the 84 rows it gets here; the result does not depend on the band height. Each stage
/// raises it to at least twice its halo (`stage_bands`), so that a wide strain window or guard
/// does not make the halo the bulk of the work.
pub fn band_rows(w: usize) -> usize {
    (BAND_PIXELS / w.max(1)).clamp(32, BAND_ROWS)
}

/// Rows either side of a band that the guard's distance transform of an image `h` rows high
/// reads (0 with the guard off; see [`invalid_pixels`]).
fn guard_halo(cfg: &Config, h: usize) -> usize {
    let guard = cfg.edge_guard_px();
    if guard > 0.0 {
        guard.ceil().min(h as f64) as usize + 1
    } else {
        0
    }
}

/// Rows either side of a band that the strain fit with window `win` reads (see
/// [`strain_banded`]).
fn strain_halo(win: usize) -> usize {
    win / 2 + 1
}

/// The band heights of the guard and of the strain for `band` rows asked for on an image `h`
/// rows high: each at least twice its stage's halo, so that a band never reads more than twice
/// its own rows. The results do not depend on the band height, so this changes only the cost.
fn stage_bands(cfg: &Config, h: usize, band: usize) -> (usize, usize) {
    (
        band.max(2 * guard_halo(cfg, h)),
        band.max(2 * strain_halo(cfg.strain_win)),
    )
}

/// `(rows, columns)` as Python prints an array shape.
fn shape_text(&(h, w): &(usize, usize)) -> String {
    format!("({h}, {w})")
}

/// Why a pair or a sequence cannot be tracked. The texts are those of the `ValueError`s of the
/// Python reference, except [`DataLength`](Self::DataLength), which only Rust has. Shapes are
/// `(rows, columns)`.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum TrackError {
    /// An input image (or the ROI mask) whose `data` does not have `w * h` values. The fields of
    /// [`Image`] are public, so a caller can build one; a NumPy array cannot disagree with its
    /// shape, so the Python reference has no such case (and this error no parity counterpart). It
    /// is found before any other check.
    #[error("image data has {len} values, not {w}x{h}")]
    DataLength { len: usize, w: usize, h: usize },
    #[error("image sizes differ: {} vs {}", shape_text(.0), shape_text(.1))]
    SizeMismatch((usize, usize), (usize, usize)),
    /// `what` is "image" for a pair, "frame" for a sequence.
    #[error(
        "ROI mask shape {} differs from the {what} shape {}",
        shape_text(.mask),
        shape_text(.image)
    )]
    MaskShape {
        mask: (usize, usize),
        image: (usize, usize),
        what: &'static str,
    },
    #[error("ROI mask is empty")]
    EmptyMask,
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The fine-stage engine failed (a device error of a GPU engine; the CPU engine never does).
    #[error("{0}")]
    Engine(String),
    #[error("image smaller than poc_win ({poc_win}): {}", shape_text(.shape))]
    TooSmall {
        poc_win: usize,
        shape: (usize, usize),
    },
    #[error("no frames")]
    NoFrames,
    #[error(
        "frame {k} has shape {}, expected {}",
        shape_text(.got),
        shape_text(.expected)
    )]
    FrameShape {
        k: usize,
        got: (usize, usize),
        expected: (usize, usize),
    },
}

impl From<EngineError> for TrackError {
    fn from(e: EngineError) -> Self {
        TrackError::Engine(e.0)
    }
}

/// Seconds spent per stage, and for a sequence frame the reference frame and the mean
/// confidence of its link (as the `timing` dict of the reference). `coarse` includes the dense
/// start field of the fine stage, as in the reference; for an image of several tiles that start
/// is formed tile by tile during the fine stage, and its time is moved from `fine` to `coarse`
/// (the margins shared by neighbouring tiles are formed, and counted, twice). With the tile
/// prefetch (`OPTFLOW_TILE_PREFETCH`, on by default) the start of the next tile is formed on the
/// prefetch thread while the engine refines the current one, so the moved time overlaps the fine
/// stage: `coarse` is over-counted and `fine` under-counted by that overlap, their sum (the wall
/// time of the two stages) is unchanged.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Timing {
    pub coarse: f64,
    pub fine: f64,
    pub strain: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_conf: Option<f64>,
    // hybrid sequences only: most consecutive frames a pixel was carried by the adjacent leg, the
    // shares of the ROI pixels that are valid by fallback, that were fallback candidates but failed
    // the spatial gate, and that were re-acquired by the direct leg
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_fb_max: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_frac: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fb_gate_rejected: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reacquired: Option<f64>,
    // crack_model "xdic" only: seconds of the crack detector, and of the per-side solve with its
    // checks (the reference's timing["xdic_detect"] and timing["xdic"]); the finer breakdown is in
    // `prof` (the `Xdic*` stages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xdic_detect: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xdic: Option<f64>,
}

/// Outcome of [`track_pair`] / one frame of [`track_sequence`].
///
/// `ux`, `uy` are NaN wherever the pixel is invalid (low confidence, outside the ROI, within the
/// guard distance of a guard source, near the image border). `conf` is the confidence *before*
/// any of that masking and guarding: it is not NaN-ed and not thresholded, so the valid pixels
/// are exactly the finite ones in `ux`. In a sequence it is the frame-0 confidence of the whole
/// link: the minimum over every leg. `raw_ux`, `raw_uy` are the unmasked field, used to
/// initialise the next frame.
#[derive(Clone, Debug)]
pub struct TrackResult {
    pub ux: Image<f32>,
    pub uy: Image<f32>,
    pub conf: Image<f32>,
    pub strain: Strain,
    pub raw_ux: Image<f32>,
    pub raw_uy: Image<f32>,
    pub timing: Timing,
    /// Hybrid sequences only: the consecutive frames each pixel has been carried by the adjacent
    /// leg (0 = direct); `None` otherwise.
    pub fb: Option<Image<i32>>,
    /// `crack_model = "xdic"` only: the crack maps and lines (None without a crack line, a band or
    /// a rejected line).
    pub crack: Option<Box<CrackOut>>,
}

fn shape_of<T>(im: &Image<T>) -> (usize, usize) {
    (im.h, im.w)
}

// ---------------------------------------------------------------------------------------------
// validity, guard and strain of the result

/// The pixels that are invalid in the output: low confidence, outside the ROI, within `guard`
/// pixels of a guard source (a 4-connected low-confidence component of at least
/// `edge_guard_min_area_px` pixels, or the ROI exterior) and, strictly, closer than `guard` to
/// the image border. The distance transform runs in bands of `band` rows with a halo of
/// `ceil(guard) + 1` rows, in which every source that can reach a band pixel lies, so the result
/// does not depend on `band`.
fn invalid_pixels(
    low: &Image<bool>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    band: usize,
) -> Image<bool> {
    assert!(band >= 1, "invalid_pixels: band height must be at least 1");
    let (w, h) = (low.w, low.h);
    let mut bad = low.clone();
    if let Some(m) = mask {
        for (b, &inside) in bad.data.iter_mut().zip(&m.data) {
            *b |= !inside;
        }
    }
    let guard = cfg.edge_guard_px();
    if guard > 0.0 {
        let mut src = large_components(low, cfg.edge_guard_min_area_px());
        if let Some(m) = mask {
            for (s, &inside) in src.data.iter_mut().zip(&m.data) {
                *s |= !inside;
            }
        }
        let halo = guard_halo(cfg, h);
        for r0 in (0..h).step_by(band) {
            let r1 = (r0 + band).min(h);
            let (a, b) = (r0.saturating_sub(halo), (r1 + halo).min(h));
            let sources = &src.data[a * w..b * w];
            if !sources.iter().any(|&s| s) {
                continue; // no source within reach of the band; the transform has nothing to measure
            }
            let valid = Image::from_vec(w, b - a, sources.iter().map(|&s| !s).collect());
            let dist = distance_to_invalid(&valid);
            for y in r0..r1 {
                let drow = &dist.data[(y - a) * w..(y - a + 1) * w];
                for (flag, &d) in bad.data[y * w..(y + 1) * w].iter_mut().zip(drow) {
                    *flag |= within(d, guard);
                }
            }
        }
        // the frame just outside the image is a guard source as well
        for y in 0..h {
            for x in 0..w {
                let edge = x.min(y).min(w - 1 - x).min(h - 1 - y);
                if (edge as f64) < guard {
                    bad.data[y * w + x] = true;
                }
            }
        }
    }
    bad
}

fn masked(u: &Image<f32>, bad: &Image<bool>) -> Image<f32> {
    Image::from_vec(
        u.w,
        u.h,
        u.data
            .iter()
            .zip(&bad.data)
            .map(|(&v, &b)| if b { f32::NAN } else { v })
            .collect(),
    )
}

/// Rows `lo..hi` of `im` as an image.
fn rows<T: Copy + Default>(im: &Image<T>, lo: usize, hi: usize) -> Image<T> {
    Image::from_vec(im.w, hi - lo, im.data[lo * im.w..hi * im.w].to_vec())
}

/// What the strain stage computes. The default is everything, as [`track_pair`] and
/// [`track_sequence`] return it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrainOpts {
    /// Compute the small-strain maps `exx`, `eyy`, `exy`. Off, they are left empty (`h == 0`)
    /// and nothing is held for them; the other strain maps are bit for bit the same.
    pub small: bool,
}

impl Default for StrainOpts {
    fn default() -> Self {
        Self { small: true }
    }
}

fn strain_parts(s: &mut Strain, small: bool) -> Vec<&mut Image<f32>> {
    let Strain {
        exx_gl,
        eyy_gl,
        exy_gl,
        exx,
        eyy,
        exy,
        gmax,
        vol,
    } = s;
    let mut parts = vec![exx_gl, eyy_gl, exy_gl];
    if small {
        parts.extend([exx, eyy, exy]);
    }
    parts.extend([gmax, vol]);
    parts
}

/// Strain of the displacement field in bands of `band` rows. The fit at a pixel reads `win / 2`
/// rows either way, so a halo of `win / 2 + 1` rows makes every band's core identical to the same
/// rows of the whole-image computation.
fn strain_banded(
    ux: &Image<f32>,
    uy: &Image<f32>,
    win: usize,
    band: usize,
    opts: StrainOpts,
) -> Strain {
    assert!(band >= 1, "strain_banded: band height must be at least 1");
    let (w, h) = (ux.w, ux.h);
    let halo = strain_halo(win);
    // room for every band from the start: appended band by band, the maps would be reallocated
    // several times, to up to twice their size, and macOS libmalloc's large-allocation cache keeps
    // the blocks left behind resident (measured: it reuses a block only for a request of more
    // than half its size). With another allocator this is harmless, only less useful
    let empty = |cap: usize| Image {
        w,
        h: 0,
        data: Vec::with_capacity(cap),
    };
    let small_cap = if opts.small { w * h } else { 0 };
    let mut out = Strain {
        exx_gl: empty(w * h),
        eyy_gl: empty(w * h),
        exy_gl: empty(w * h),
        exx: empty(small_cap),
        eyy: empty(small_cap),
        exy: empty(small_cap),
        gmax: empty(w * h),
        vol: empty(w * h),
    };
    for r0 in (0..h).step_by(band) {
        let r1 = (r0 + band).min(h);
        let (a, b) = (r0.saturating_sub(halo), (r1 + halo).min(h));
        let g = displacement_gradient(&rows(ux, a, b), &rows(uy, a, b), win);
        let core = Gradient {
            dudx: rows(&g.dudx, r0 - a, r1 - a),
            dudy: rows(&g.dudy, r0 - a, r1 - a),
            dvdx: rows(&g.dvdx, r0 - a, r1 - a),
            dvdy: rows(&g.dvdy, r0 - a, r1 - a),
        };
        let mut s = strain_fields_with(&core, opts.small);
        for (dst, src) in strain_parts(&mut out, opts.small)
            .into_iter()
            .zip(strain_parts(&mut s, opts.small))
        {
            dst.data.append(&mut src.data);
            dst.h += r1 - r0;
        }
    }
    out
}

/// Displacement and strain of the output (`_finish` of the reference): `low` marks the pixels
/// of too low confidence.
struct Finished {
    ux: Image<f32>,
    uy: Image<f32>,
    strain: Strain,
}

/// `band` is the band height asked for; each stage uses at least twice its halo
/// ([`stage_bands`]).
fn finish(
    ux: &Image<f32>,
    uy: &Image<f32>,
    low: &Image<bool>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    band: usize,
    opts: StrainOpts,
) -> Finished {
    let (guard_band, strain_band) = stage_bands(cfg, low.h, band);
    let (mx, my) = prof::time(Stage::Guard, || {
        let bad = invalid_pixels(low, cfg, mask, guard_band);
        (masked(ux, &bad), masked(uy, &bad))
    });
    let strain = prof::time(Stage::Strain, || {
        strain_banded(&mx, &my, cfg.strain_win, strain_band, opts)
    });
    Finished {
        ux: mx,
        uy: my,
        strain,
    }
}

// ---------------------------------------------------------------------------------------------
// pair

/// Coarse and fine stage of one pair, before any validity decision.
struct Pair {
    raw_ux: Image<f32>,
    raw_uy: Image<f32>,
    conf: Image<f32>,
    /// `conf < conf_min`, decided on the `f64` confidence the `f32` `conf` is rounded from.
    low: Image<bool>,
    /// Seconds, as in [`Timing`]: with the tile prefetch the start-field time, formed on the
    /// prefetch thread during the fine stage, is in `coarse` and taken out of `fine`, so `coarse`
    /// is over-counted and `fine` under-counted by the overlap (the sum is unchanged).
    coarse: f64,
    fine: f64,
    /// `crack_model = "xdic"` only: the crack lines of the leg.
    xleg: Option<Box<XLeg>>,
}

/// The coarse grid field as a dense start for the fine stage. With a dense guess `init` the
/// coarse stage corrected the guess at the grid points: the interpolated correction is added to
/// the guess.
#[cfg(test)]
fn dense_start(
    g: &Grid,
    init: Option<(&Image<f32>, &Image<f32>)>,
    (w, h): (usize, usize),
) -> (Image<f32>, Image<f32>) {
    let whole = Rect {
        x0: 0,
        y0: 0,
        x1: w,
        y1: h,
    };
    dense_start_on(&start_grid(g, init), init, w, whole)
}

/// The grid that [`dense_start_on`] interpolates: the coarse field `g`, or with a dense guess
/// `init` the coarse stage's correction of the guess at the grid points.
fn start_grid<'g>(g: &'g Grid, init: Option<(&Image<f32>, &Image<f32>)>) -> Cow<'g, Grid> {
    match init {
        None => Cow::Borrowed(g),
        Some((ix, iy)) => {
            let at = |u: &Image<f32>, init: &Image<f32>| {
                let mut data = Vec::with_capacity(u.data.len());
                for (j, &y) in g.ys.iter().enumerate() {
                    for (i, &x) in g.xs.iter().enumerate() {
                        data.push(
                            (f64::from(u.data[j * u.w + i]) - f64::from(init.get(x, y))) as f32,
                        );
                    }
                }
                Image::from_vec(u.w, u.h, data)
            };
            Cow::Owned(Grid {
                xs: g.xs.clone(),
                ys: g.ys.clone(),
                ux: at(&g.ux, ix),
                uy: at(&g.uy, iy),
                peak: g.peak.clone(),
            })
        }
    }
}

/// The dense start on the rectangle `r` of an image `w` pixels wide: per pixel the interpolated
/// `source` grid ([`start_grid`]), plus the guess `init` (images of the whole image) if there
/// is one. A pixel's value does not depend on `r`, so a tile's start equals the same part of the
/// start of the whole image.
fn dense_start_on(
    source: &Grid,
    init: Option<(&Image<f32>, &Image<f32>)>,
    w: usize,
    r: Rect,
) -> (Image<f32>, Image<f32>) {
    let rw = r.x1 - r.x0;
    let mut gx = Image::new(rw, r.y1 - r.y0);
    let mut gy = Image::new(rw, r.y1 - r.y0);
    gx.data
        .par_chunks_mut(rw.max(1))
        .zip(gy.data.par_chunks_mut(rw.max(1)))
        .enumerate()
        .for_each(|(j, (rx, ry))| {
            let y = r.y0 + j;
            for (i, x) in (r.x0..r.x1).enumerate() {
                let (cx, cy) = grid_sample(source, x as f64, y as f64);
                let (bx, by) = match init {
                    None => (0.0, 0.0),
                    Some((ix, iy)) => {
                        (f64::from(ix.data[y * w + x]), f64::from(iy.data[y * w + x]))
                    }
                };
                rx[i] = (bx + cx) as f32;
                ry[i] = (by + cy) as f32;
            }
        });
    (gx, gy)
}

/// [`pair_flow_on`] with the CPU engine (for the tests of this module).
#[cfg(test)]
fn pair_flow(
    i1: &Image<f32>,
    i2: &Image<f32>,
    cfg: &Config,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Pair {
    pair_flow_on(&CpuEngine, i1, i2, cfg, init).expect("the CPU engine does not fail")
}

fn pair_flow_on(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    cfg: &Config,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Result<Pair, TrackError> {
    if crate::iter_trace::enabled() {
        crate::iter_trace::set_warm(init.is_some());
    }
    let t = prof::start(Stage::Coarse);
    let g = coarse_flow(i1, i2, cfg, init);
    let source = start_grid(&g, init);
    let (w, h) = (i1.w, i1.h);
    // the tiled fine stage forms the dense start of each tile when it gets to it, so the start of
    // the whole image is never held
    let whole = (w.max(h) <= TILE).then(|| {
        let r = Rect {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        };
        dense_start_on(&source, init, w, r)
    });
    let coarse = t.elapsed();
    prof::add(Stage::Coarse, t);
    // the reference times the dense start with the coarse stage: so is each tile's (the margins
    // shared by neighbouring tiles are formed twice)
    // (a mutex: the tiled driver may form the next tile's start on its prefetch thread)
    let tile_starts = std::sync::Mutex::new(Duration::ZERO);
    let start = |r: Rect| {
        let t = prof::start(Stage::Coarse);
        let out = dense_start_on(&source, init, w, r);
        *tile_starts.lock().unwrap_or_else(|e| e.into_inner()) += t.elapsed();
        prof::add(Stage::Coarse, t);
        out
    };
    let t = Instant::now();
    let (flow, low) = match &whole {
        Some((gx, gy)) => refine_low(engine, i1, i2, gx, gy, cfg),
        None => refine_tiled_from_on(engine, i1, i2, &start, cfg, TILE),
    }?;
    drop(whole);
    let tile_starts = tile_starts.into_inner().unwrap_or_else(|e| e.into_inner());
    let fine = t.elapsed().saturating_sub(tile_starts);
    Ok(Pair {
        raw_ux: flow.ux,
        raw_uy: flow.uy,
        conf: flow.conf,
        low,
        coarse: (coarse + tile_starts).as_secs_f64(),
        fine: fine.as_secs_f64(),
        xleg: None,
    })
}

/// `data` has `w * h` values (a product that does not fit in `usize` is not their number either).
fn check_len<T>(i: &Image<T>) -> Result<(), TrackError> {
    if i.w.checked_mul(i.h) == Some(i.data.len()) {
        Ok(())
    } else {
        Err(TrackError::DataLength {
            len: i.data.len(),
            w: i.w,
            h: i.h,
        })
    }
}

fn check_size(i: &Image<f32>, cfg: &Config) -> Result<(), TrackError> {
    if i.w.min(i.h) < cfg.poc_win {
        return Err(TrackError::TooSmall {
            poc_win: cfg.poc_win,
            shape: shape_of(i),
        });
    }
    Ok(())
}

/// The ROI checks shared by pair and sequence: shape, then non-empty.
fn check_mask(
    mask: Option<&Image<bool>>,
    image: (usize, usize),
    what: &'static str,
) -> Result<(), TrackError> {
    if let Some(m) = mask {
        if shape_of(m) != image {
            return Err(TrackError::MaskShape {
                mask: shape_of(m),
                image,
                what,
            });
        }
        if !m.data.iter().any(|&b| b) {
            return Err(TrackError::EmptyMask);
        }
    }
    Ok(())
}

/// Displacement of `i1`'s pixels into `i2`. `mask` is the ROI (true inside); `init` is a dense
/// guess of the displacement, e.g. the previous frame's raw field. Panics if `init` does not
/// have the size of the images.
///
/// Images with a side over [`TILE`] pixels are refined in tiles; the strain and the guard are
/// computed in bands of [`band_rows`] rows. Errors, in the reference's order: sizes differ, mask
/// shape, empty mask, invalid config, image smaller than `poc_win`; before them all, in Rust only,
/// an image or the mask whose `data` is not `w * h` long ([`TrackError::DataLength`]).
pub fn track_pair(
    i1: &Image<f32>,
    i2: &Image<f32>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Result<TrackResult, TrackError> {
    track_pair_on(&CpuEngine, i1, i2, cfg, mask, init)
}

/// [`track_pair`] with the iterations of the fine stage run by `engine` (the CPU engine gives
/// `track_pair` bit for bit). An error of the engine is [`TrackError::Engine`].
pub fn track_pair_on(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Result<TrackResult, TrackError> {
    track_pair_cow(
        engine,
        Cow::Borrowed(i1),
        Cow::Borrowed(i2),
        cfg,
        mask,
        init,
        StrainOpts::default(),
    )
}

/// [`track_pair`] that takes the images and releases them as soon as the fine stage is done,
/// before the guard and the strain: for a caller that does not need them afterwards (the
/// command line tool), so that they are not held through the end of the pipeline. The result
/// is that of [`track_pair`].
pub fn track_pair_owned(
    i1: Image<f32>,
    i2: Image<f32>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Result<TrackResult, TrackError> {
    track_pair_owned_on(&CpuEngine, i1, i2, cfg, mask, init)
}

/// [`track_pair_owned`] with the iterations of the fine stage run by `engine`.
pub fn track_pair_owned_on(
    engine: &dyn FineEngine,
    i1: Image<f32>,
    i2: Image<f32>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Result<TrackResult, TrackError> {
    track_pair_owned_with(engine, i1, i2, cfg, mask, init, StrainOpts::default())
}

/// [`track_pair_owned_on`] with a choice of the strain maps to compute ([`StrainOpts`]).
pub fn track_pair_owned_with(
    engine: &dyn FineEngine,
    i1: Image<f32>,
    i2: Image<f32>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    init: Option<(&Image<f32>, &Image<f32>)>,
    opts: StrainOpts,
) -> Result<TrackResult, TrackError> {
    track_pair_cow(
        engine,
        Cow::Owned(i1),
        Cow::Owned(i2),
        cfg,
        mask,
        init,
        opts,
    )
}

/// The largest window the fine stage passes to an engine for images of `w x h`.
fn check_window(
    engine: &dyn FineEngine,
    w: usize,
    h: usize,
    cfg: &Config,
) -> Result<(), TrackError> {
    let side = TILE + 2 * tile_margin(cfg);
    Ok(engine.check_window(w.min(side), h.min(side), cfg)?)
}

fn track_pair_cow(
    engine: &dyn FineEngine,
    i1: Cow<'_, Image<f32>>,
    i2: Cow<'_, Image<f32>>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
    init: Option<(&Image<f32>, &Image<f32>)>,
    opts: StrainOpts,
) -> Result<TrackResult, TrackError> {
    check_len(&i1)?;
    check_len(&i2)?;
    if let Some(m) = mask {
        check_len(m)?;
    }
    if shape_of(&i1) != shape_of(&i2) {
        return Err(TrackError::SizeMismatch(shape_of(&i1), shape_of(&i2)));
    }
    check_mask(mask, shape_of(&i1), "image")?;
    cfg.validate()?;
    check_size(&i1, cfg)?;
    check_window(engine, i1.w, i1.h, cfg)?;
    if cfg.is_xdic() {
        let pair = xdic_leg::pair_flow_xdic(engine, &i1, &i2, cfg, init, mask, None)?;
        engine.release_caches();
        let w = i1.w;
        drop((i1, i2));
        let t = Instant::now();
        if pair.xleg.as_ref().is_some_and(|x| x.has_lines()) {
            return Ok(xdic_leg::pair_result_with_lines(pair, cfg, opts, t));
        }
        // no line: the baseline result (with the bands the detector saw, if any)
        let fin = finish(
            &pair.raw_ux,
            &pair.raw_uy,
            &pair.low,
            cfg,
            mask,
            band_rows(w),
            opts,
        );
        let (t_detect, t_xdic) = xdic_leg::leg_times(&pair);
        let crack = pair.xleg.as_ref().and_then(|x| x.no_line_crack());
        return Ok(TrackResult {
            ux: fin.ux,
            uy: fin.uy,
            conf: pair.conf,
            strain: fin.strain,
            raw_ux: pair.raw_ux,
            raw_uy: pair.raw_uy,
            timing: Timing {
                coarse: pair.coarse,
                fine: pair.fine,
                strain: t.elapsed().as_secs_f64(),
                xdic_detect: t_detect,
                xdic: t_xdic,
                ..Timing::default()
            },
            fb: None,
            crack,
        });
    }
    let pair = pair_flow_on(engine, &i1, &i2, cfg, init)?;
    // the engine's caches are not needed by the guard and the strain (a sequence keeps them: it
    // refines every frame)
    engine.release_caches();
    let w = i1.w;
    drop((i1, i2));
    let t = Instant::now();
    let fin = finish(
        &pair.raw_ux,
        &pair.raw_uy,
        &pair.low,
        cfg,
        mask,
        band_rows(w),
        opts,
    );
    let timing = Timing {
        coarse: pair.coarse,
        fine: pair.fine,
        strain: t.elapsed().as_secs_f64(),
        reference: None,
        mean_conf: None,
        ..Timing::default()
    };
    Ok(TrackResult {
        ux: fin.ux,
        uy: fin.uy,
        conf: pair.conf,
        strain: fin.strain,
        raw_ux: pair.raw_ux,
        raw_uy: pair.raw_uy,
        timing,
        fb: None,
        crack: None,
    })
}

// ---------------------------------------------------------------------------------------------
// sampling and composition

/// The lattice cell of a sample at coordinate `c` on an axis of `n >= 1` nodes, `c` in
/// `[0, n - 1]`: nodes `i0 <= i1 < n` and the fraction `t` in `[0, 1]` from `i0` towards `i1`.
fn axis(c: f64, n: usize) -> (usize, usize, f64) {
    if n == 1 {
        return (0, 0, 0.0);
    }
    let i0 = (c.floor() as usize).min(n - 2);
    (i0, i0 + 1, c - i0 as f64)
}

/// [`sample_bilinear`] in `f64`.
fn bilinear64(a: &Image<f32>, x: f64, y: f64) -> f64 {
    let (w, h) = (a.w, a.h);
    if w == 0
        || h == 0
        || !(x.is_finite() && y.is_finite())
        || x < 0.0
        || y < 0.0
        || x > (w - 1) as f64
        || y > (h - 1) as f64
    {
        return f64::NAN;
    }
    let (x0, x1, tx) = axis(x, w);
    let (y0, y1, ty) = axis(y, h);
    let at = |yy: usize, xx: usize| f64::from(a.data[yy * w + xx]);
    (1.0 - ty) * ((1.0 - tx) * at(y0, x0) + tx * at(y0, x1))
        + ty * ((1.0 - tx) * at(y1, x0) + tx * at(y1, x1))
}

/// Bilinear sample of `a` at column `x`, row `y`; NaN outside `[0, w - 1] x [0, h - 1]`, for a
/// non-finite coordinate, and where a NaN of `a` enters the interpolation.
pub fn sample_bilinear(a: &Image<f32>, x: f64, y: f64) -> f32 {
    bilinear64(a, x, y) as f32
}

/// `u_0k(X) = u_0r(X) + u_rk(X + u_0r(X))` with `a = u_0r`, `b = u_rk`: NaN where the point
/// leaves the image of `b`.
pub fn compose(
    ax: &Image<f32>,
    ay: &Image<f32>,
    bx: &Image<f32>,
    by: &Image<f32>,
) -> (Image<f32>, Image<f32>) {
    let w = ax.w;
    assert!(
        shape_of(ax) == shape_of(ay)
            && shape_of(ax) == shape_of(bx)
            && shape_of(ax) == shape_of(by),
        "compose: the four fields differ in size"
    );
    let mut cx = Image::new(ax.w, ax.h);
    let mut cy = Image::new(ax.w, ax.h);
    cx.data
        .par_chunks_mut(w.max(1))
        .zip(cy.data.par_chunks_mut(w.max(1)))
        .enumerate()
        .for_each(|(y, (rx, ry))| {
            for x in 0..w {
                let i = y * w + x;
                let (u, v) = (f64::from(ax.data[i]), f64::from(ay.data[i]));
                let (px, py) = (x as f64 + u, y as f64 + v);
                rx[x] = (u + bilinear64(bx, px, py)) as f32;
                ry[x] = (v + bilinear64(by, px, py)) as f32;
            }
        });
    (cx, cy)
}

/// `min(c0r, conf)` where `conf`, the confidence of a leg, is sampled at the positions the
/// frame-0 pixels have in that leg's reference (`xx + u0r`); a sample outside the image is NaN
/// and counts as 0.
fn link_conf(c0r: &[f64], conf: &Image<f32>, u0r: &(Image<f32>, Image<f32>)) -> Vec<f64> {
    let w = conf.w;
    let mut out = vec![0.0f64; c0r.len()];
    out.par_chunks_mut(w.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let c = bilinear64(
                    conf,
                    x as f64 + f64::from(u0r.0.data[i]),
                    y as f64 + f64::from(u0r.1.data[i]),
                );
                *o = c0r[i].min(if c.is_finite() { c } else { 0.0 });
            }
        });
    out
}

/// Mean confidence of a leg over the frame-0 pixels `X` that lie in the ROI (when given; the test
/// is on the frame-0 grid, at `X` itself) and that, in the leg's reference, sit on a reportable
/// pixel: one that the border rule of the fine stage (spec 2.4) did not zero. The border strips
/// are a geometric exclusion, not a tracking-quality signal; counting them pulled the mean of
/// frames of ~128 px under `ref_update_conf` and moved the reference at every frame.
///
/// `conf` is the leg's confidence and `raw` its raw displacement (what the border rule is decided
/// on); `u0r` maps frame 0 into the leg's reference. Both the confidence and the 0/1 image of the
/// reportable pixels are sampled bilinearly at `X + u0r(X)`; a pixel counts where the confidence
/// is finite (a sample outside the image is NaN) and the reportable sample exceeds 0.5. 0 if none
/// counts. The sum runs row by row in order, so it does not depend on the thread count.
fn mean_conf(
    conf: &Image<f32>,
    raw: (&Image<f32>, &Image<f32>),
    u0r: &(Image<f32>, Image<f32>),
    mask: Option<&Image<bool>>,
    cfg: &Config,
) -> f64 {
    let (w, h) = (conf.w, conf.h);
    let b = border_zone(cfg);
    let mut rep = Image::<f32>::new(w, h);
    rep.data
        .par_chunks_mut(w.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            for (x, r) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let (tx, ty) = (
                    x as f64 + f64::from(raw.0.data[i]),
                    y as f64 + f64::from(raw.1.data[i]),
                );
                *r = if reportable(x, y, tx, ty, w, h, b) {
                    1.0
                } else {
                    0.0
                };
            }
        });
    let rows: Vec<(f64, usize)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let (mut sum, mut n) = (0.0f64, 0usize);
            for x in 0..w {
                let i = y * w + x;
                let (px, py) = (
                    x as f64 + f64::from(u0r.0.data[i]),
                    y as f64 + f64::from(u0r.1.data[i]),
                );
                let c = bilinear64(conf, px, py);
                if c.is_finite() && bilinear64(&rep, px, py) > 0.5 && mask.is_none_or(|m| m.data[i])
                {
                    sum += c;
                    n += 1;
                }
            }
            (sum, n)
        })
        .collect();
    let (sum, n) = rows
        .iter()
        .fold((0.0f64, 0usize), |(s, n), &(rs, rn)| (s + rs, n + rn));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

// ---------------------------------------------------------------------------------------------
// sequence

/// State of a running sequence: the reference, the displacement and confidence of frame 0 into
/// it, and the previous frame with its pair.
struct Run {
    /// Frames taken after frame 0.
    k: usize,
    shape: (usize, usize),
    reference: Image<f32>,
    ref_idx: usize,
    /// Frame 0 -> reference.
    u0r: (Image<f32>, Image<f32>),
    /// Confidence of the frame 0 -> reference link, on the frame-0 grid.
    c0r: Vec<f64>,
    prev: Image<f32>,
    prev_pair: Option<Pair>,
    /// The extra state of `seq_mode = "hybrid"`.
    hybrid: Option<hybrid::Hybrid>,
}

enum State {
    Start,
    Running(Box<Run>),
    Done,
}

struct Sequence<'a, I> {
    engine: &'a dyn FineEngine,
    frames: I,
    cfg: &'a Config,
    mask: Option<&'a Image<bool>>,
    opts: StrainOpts,
    state: State,
}

impl<I: Iterator<Item = Image<f32>>> Sequence<'_, I> {
    /// One leg `i1 -> i2`. With `crack_model = "xdic"` the carried crack lines `hint()` and the ROI in
    /// `i1`'s coordinates `roi()` go with it; otherwise the call is exactly [`pair_flow_on`].
    fn leg(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        init: Option<(&Image<f32>, &Image<f32>)>,
        hint: impl FnOnce() -> Option<Vec<Poly>>,
        roi: impl FnOnce() -> Option<Image<bool>>,
    ) -> Result<Pair, TrackError> {
        if self.cfg.is_xdic() {
            let (h, r) = (hint(), roi());
            xdic_leg::pair_flow_xdic(
                self.engine,
                i1,
                i2,
                self.cfg,
                init,
                r.as_ref(),
                h.as_deref(),
            )
        } else {
            pair_flow_on(self.engine, i1, i2, self.cfg, init)
        }
    }

    fn start(&mut self) -> Result<Box<Run>, TrackError> {
        let f0 = self.frames.next().ok_or(TrackError::NoFrames)?;
        check_len(&f0)?;
        if let Some(m) = self.mask {
            check_len(m)?;
        }
        self.cfg.validate()?;
        let shape = shape_of(&f0);
        check_mask(self.mask, shape, "frame")?;
        let (w, h) = (f0.w, f0.h);
        Ok(Box::new(Run {
            k: 0,
            shape,
            reference: f0.clone(),
            ref_idx: 0,
            u0r: (Image::new(w, h), Image::new(w, h)),
            c0r: vec![1.0; w * h],
            prev: f0,
            prev_pair: None,
            hybrid: self.cfg.is_hybrid().then(|| hybrid::Hybrid::new(w, h)),
        }))
    }

    /// The next frame's result; `None` when the frames are used up.
    fn step(&mut self, run: &mut Run) -> Option<Result<(usize, TrackResult), TrackError>> {
        let fk = self.frames.next()?;
        run.k += 1;
        let k = run.k;
        if let Err(e) = check_len(&fk) {
            return Some(Err(e));
        }
        if shape_of(&fk) != run.shape {
            return Some(Err(TrackError::FrameShape {
                k,
                got: shape_of(&fk),
                expected: run.shape,
            }));
        }
        if let Err(e) = check_size(&run.reference, self.cfg) {
            return Some(Err(e));
        }
        if k == 1 {
            if let Err(e) = check_window(self.engine, fk.w, fk.h, self.cfg) {
                return Some(Err(e));
            }
        }
        if run.hybrid.is_some() {
            return Some(self.step_hybrid(run, fk).map(|r| (k, r)));
        }
        let cfg = self.cfg;
        let init = run.prev_pair.as_ref().map(|p| (&p.raw_ux, &p.raw_uy));
        let mut pair = match self.leg(
            &run.reference,
            &fk,
            init,
            || xdic_leg::same_ref_hint(run.prev_pair.as_ref()),
            || xdic_leg::roi_in(self.mask, &run.u0r),
        ) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let leg_mean = |pair: &Pair, u0r: &(Image<f32>, Image<f32>)| {
            mean_conf(
                &pair.conf,
                (&pair.raw_ux, &pair.raw_uy),
                u0r,
                self.mask,
                cfg,
            )
        };
        let mut cmean = leg_mean(&pair, &run.u0r);
        if cmean < cfg.ref_update_conf && run.ref_idx != k - 1 {
            // the link to frame k is poor: the reference moves to frame k - 1. prev_pair is the
            // leg old reference -> k - 1, its confidence extends the link of frame 0
            let prev = run
                .prev_pair
                .as_ref()
                .expect("a reference older than k - 1 means a pair was tracked before");
            run.c0r = link_conf(&run.c0r, &prev.conf, &run.u0r);
            run.u0r = compose(&run.u0r.0, &run.u0r.1, &prev.raw_ux, &prev.raw_uy);
            run.reference = std::mem::replace(&mut run.prev, Image::new(0, 0));
            run.ref_idx = k - 1;
            pair = match self.leg(
                &run.reference,
                &fk,
                None,
                || xdic_leg::hint(run.prev_pair.as_ref()),
                || xdic_leg::roi_in(self.mask, &run.u0r),
            ) {
                Ok(p) => p,
                Err(e) => return Some(Err(e)),
            };
            cmean = leg_mean(&pair, &run.u0r);
        }
        let (cx, cy) = compose(&run.u0r.0, &run.u0r.1, &pair.raw_ux, &pair.raw_uy);
        let conf0 = link_conf(&run.c0r, &pair.conf, &run.u0r);
        let (w, h) = (fk.w, fk.h);
        let low = Image::from_vec(w, h, conf0.iter().map(|&c| c < cfg.conf_min).collect());
        let t = Instant::now();
        let xo = if cfg.is_xdic() {
            xdic_leg::frame0_out(
                &pair, &run.u0r, &cx, &cy, &low.data, cfg, self.mask, self.opts,
            )
        } else {
            None
        };
        let (ux, uy, strain, crack) = match xo {
            Some((ux, uy, strain, crack)) => (ux, uy, strain, Some(crack)),
            None => {
                let fin = finish(&cx, &cy, &low, cfg, self.mask, band_rows(w), self.opts);
                (fin.ux, fin.uy, fin.strain, None)
            }
        };
        let (xdic_detect, xdic) = xdic_leg::leg_times(&pair);
        let result = TrackResult {
            ux,
            uy,
            conf: Image::from_vec(w, h, conf0.iter().map(|&c| c as f32).collect()),
            strain,
            raw_ux: cx,
            raw_uy: cy,
            timing: Timing {
                coarse: pair.coarse,
                fine: pair.fine,
                strain: t.elapsed().as_secs_f64(),
                reference: Some(run.ref_idx),
                mean_conf: Some(cmean),
                xdic_detect,
                xdic,
                ..Timing::default()
            },
            fb: None,
            crack,
        };
        run.prev = fk;
        run.prev_pair = Some(pair);
        Some(Ok((k, result)))
    }
}

impl<I: Iterator<Item = Image<f32>>> Iterator for Sequence<'_, I> {
    type Item = Result<(usize, TrackResult), TrackError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut run = match std::mem::replace(&mut self.state, State::Done) {
            State::Done => return None,
            State::Start => match self.start() {
                Ok(run) => run,
                Err(e) => return Some(Err(e)),
            },
            State::Running(run) => run,
        };
        let item = self.step(&mut run)?;
        if item.is_ok() {
            self.state = State::Running(run);
        }
        Some(item)
    }
}

/// Displacement from frame 0 to frame `k` for `k = 1, 2, ...`, one item per frame after the first.
///
/// Each frame is matched directly against the current reference frame (no per-step drift); the
/// reference moves to the previous frame when the mean confidence of the link drops below
/// `cfg.ref_update_conf`. The confidence of the output is that of the whole frame-0 -> `k` link:
/// the minimum of the confidences of all legs (frame 0 -> r1 -> ... -> current reference -> `k`),
/// each sampled where the frame-0 pixel lies in that leg's reference frame. Validity, guard and
/// strain are decided on the frame-0 grid.
///
/// The checks of the reference run when the first item is requested (no frames, invalid config,
/// ROI shape or emptiness) and as each frame arrives (shape, `poc_win`); after an error the
/// iterator ends. In Rust only, a frame or the mask whose `data` is not `w * h` long is an error
/// as well ([`TrackError::DataLength`]), found first: for the mask and the first frame when the
/// first item is requested, for every other frame as it arrives.
pub fn track_sequence<'a, I>(
    frames: I,
    cfg: &'a Config,
    mask: Option<&'a Image<bool>>,
) -> impl Iterator<Item = Result<(usize, TrackResult), TrackError>> + 'a
where
    I: Iterator<Item = Image<f32>> + 'a,
{
    track_sequence_on(&CpuEngine, frames, cfg, mask)
}

/// [`track_sequence`] with the iterations of the fine stage run by `engine`. The engine's
/// `check_window` is asked once, when the first frame after the first arrives (after its
/// `poc_win` check); an error of the engine ends the iterator like any other.
pub fn track_sequence_on<'a, I>(
    engine: &'a dyn FineEngine,
    frames: I,
    cfg: &'a Config,
    mask: Option<&'a Image<bool>>,
) -> impl Iterator<Item = Result<(usize, TrackResult), TrackError>> + 'a
where
    I: Iterator<Item = Image<f32>> + 'a,
{
    track_sequence_with(engine, frames, cfg, mask, StrainOpts::default())
}

/// [`track_sequence_on`] with a choice of the strain maps to compute ([`StrainOpts`]).
pub fn track_sequence_with<'a, I>(
    engine: &'a dyn FineEngine,
    frames: I,
    cfg: &'a Config,
    mask: Option<&'a Image<bool>>,
    opts: StrainOpts,
) -> impl Iterator<Item = Result<(usize, TrackResult), TrackError>> + 'a
where
    I: Iterator<Item = Image<f32>> + 'a,
{
    Sequence {
        engine,
        frames,
        cfg,
        mask,
        opts,
        state: State::Start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strain::strain_fields;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn bits(im: &Image<f32>) -> Vec<u32> {
        im.data.iter().map(|v| v.to_bits()).collect()
    }

    /// `got == want` elementwise; on a mismatch the message gives the count and the first
    /// differing index, not the arrays.
    fn assert_same<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[T], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let wrong: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        assert!(
            wrong.is_empty(),
            "{what}: {} of {} differ, first at {} ({:?} vs {:?})",
            wrong.len(),
            got.len(),
            wrong.first().copied().unwrap_or(0),
            wrong.first().map(|&i| &got[i]),
            wrong.first().map(|&i| &want[i])
        );
    }

    fn strain_images(s: &Strain) -> [&Image<f32>; 8] {
        [
            &s.exx_gl, &s.eyy_gl, &s.exy_gl, &s.exx, &s.eyy, &s.exy, &s.gmax, &s.vol,
        ]
    }

    // ---- sampling and composition

    #[test]
    fn sample_bilinear_interpolates_and_is_nan_outside() {
        // a(x, y) = 10 y + x on a 4 x 3 image is reproduced exactly by bilinear interpolation
        let a = Image::from_vec(
            4,
            3,
            (0..12).map(|i| (10 * (i / 4) + i % 4) as f32).collect(),
        );
        for (x, y) in [
            (0.0, 0.0),
            (3.0, 2.0),
            (1.25, 0.5),
            (2.9, 1.7),
            (3.0, 0.0),
            (0.0, 2.0),
        ] {
            let want = 10.0 * y + x;
            assert!(
                (f64::from(sample_bilinear(&a, x, y)) - want).abs() < 1e-5,
                "({x}, {y})"
            );
        }
        for (x, y) in [
            (-1e-9, 0.0),
            (0.0, -1e-9),
            (3.0 + 1e-9, 0.0),
            (0.0, 2.0 + 1e-9),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::NEG_INFINITY),
        ] {
            assert!(sample_bilinear(&a, x, y).is_nan(), "({x}, {y})");
        }
        // single row / column / pixel and the empty image
        let row = Image::from_vec(3, 1, vec![1.0f32, 2.0, 4.0]);
        assert_eq!(sample_bilinear(&row, 1.5, 0.0), 3.0);
        assert!(sample_bilinear(&row, 1.5, 0.5).is_nan());
        let one = Image::from_vec(1, 1, vec![7.0f32]);
        assert_eq!(sample_bilinear(&one, 0.0, 0.0), 7.0);
        assert!(sample_bilinear(&one, 0.5, 0.0).is_nan());
        assert!(sample_bilinear(&Image::<f32>::new(0, 0), 0.0, 0.0).is_nan());
    }

    #[test]
    fn sample_bilinear_propagates_a_nan_of_its_cell_even_with_weight_zero() {
        // like numpy: 0 * NaN is NaN, so every sample whose lattice cell contains the NaN is NaN
        let mut a = Image::from_vec(4, 4, vec![1.0f32; 16]);
        a.data[5] = f32::NAN; // (1, 1)
        assert!(sample_bilinear(&a, 1.0, 1.0).is_nan());
        assert!(sample_bilinear(&a, 0.0, 0.0).is_nan());
        assert!(sample_bilinear(&a, 1.5, 1.5).is_nan());
        assert!(sample_bilinear(&a, 1.999, 1.999).is_nan()); // the cell is [1, 2] x [1, 2]
        assert_eq!(sample_bilinear(&a, 2.0, 2.0), 1.0); // from 2 on the cell is [2, 3] x [2, 3]
        assert_eq!(sample_bilinear(&a, 3.0, 3.0), 1.0);
        assert_eq!(sample_bilinear(&a, 2.5, 2.5), 1.0);
        assert_eq!(sample_bilinear(&a, 3.0, 0.0), 1.0);
    }

    #[test]
    fn compose_of_translations_adds_and_is_nan_where_the_point_leaves() {
        let z = Image::new(20, 20);
        let fill = |v: f32| Image::from_vec(20, 20, vec![v; 400]);
        let (cx, cy) = compose(&fill(1.0), &fill(2.0), &fill(0.5), &fill(-1.0));
        for y in 0..10 {
            for x in 0..10 {
                assert!((cx.get(x, y) - 1.5).abs() < 1e-6 && (cy.get(x, y) - 1.0).abs() < 1e-6);
            }
        }
        // x + 1 > 19 for the last column: the sample of b is NaN, so is the sum
        assert!(cx.get(19, 5).is_nan() && cy.get(19, 5).is_nan());
        assert!(cx.get(18, 5).is_finite());
        // y + 2 > 19 for the last two rows
        assert!(cx.get(5, 18).is_nan() && cx.get(5, 19).is_nan() && cx.get(5, 17).is_finite());
        // a zero first leg returns the second
        let (px, _) = compose(&z, &z, &fill(0.25), &z);
        assert!(px.data.iter().all(|&v| v == 0.25));
    }

    // ---- errors

    fn text(e: TrackError) -> String {
        e.to_string()
    }

    fn zeros(w: usize, h: usize) -> Image<f32> {
        Image::new(w, h)
    }

    #[test]
    fn track_pair_errors_have_the_texts_and_order_of_the_reference() {
        let cfg = Config::default();
        let e = track_pair(&zeros(10, 10), &zeros(12, 10), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), "image sizes differ: (10, 10) vs (10, 12)");
        let ones = |w, h| Image::from_vec(w, h, vec![true; w * h]);
        let (a, b) = (zeros(64, 64), zeros(64, 64));
        let e = track_pair(&a, &b, &cfg, Some(&ones(65, 64)), None).unwrap_err();
        assert_eq!(
            text(e),
            "ROI mask shape (64, 65) differs from the image shape (64, 64)"
        );
        let none = Image::from_vec(64, 64, vec![false; 64 * 64]);
        // the empty mask is reported before the invalid config and the small image
        let bad_cfg = Config {
            poc_win: 48,
            ..Config::default()
        };
        let e = track_pair(&a, &b, &bad_cfg, Some(&none), None).unwrap_err();
        assert_eq!(text(e), "ROI mask is empty");
        let e = track_pair(&a, &b, &bad_cfg, None, None).unwrap_err();
        assert_eq!(text(e), "poc_win must be a power of two >= 16");
        let e = track_pair(&zeros(128, 32), &zeros(128, 32), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), "image smaller than poc_win (64): (32, 128)");
        let e = track_pair(&zeros(48, 128), &zeros(48, 128), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), "image smaller than poc_win (64): (128, 48)");
    }

    fn first_error(
        frames: Vec<Image<f32>>,
        cfg: &Config,
        mask: Option<&Image<bool>>,
    ) -> Option<String> {
        let mut it = track_sequence(frames.into_iter(), cfg, mask);
        let first = it.next()?.err().map(text);
        assert!(it.next().is_none(), "the iterator ends after an error");
        first
    }

    /// An image whose `data` is not `w * h` long: `Image`'s fields are public, so a caller can
    /// build one (`Image::from_vec` panics instead).
    fn short<T: Default + Clone>(w: usize, h: usize, len: usize) -> Image<T> {
        Image {
            w,
            h,
            data: vec![T::default(); len],
        }
    }

    // Rust only: Python's arrays cannot disagree with their shape, so these errors have no parity
    // counterpart (and no place in the reference's order: they come first)
    #[test]
    fn track_pair_rejects_an_image_whose_data_is_not_w_times_h_long() {
        let cfg = Config::default();
        let (ok, bad) = (zeros(64, 64), short::<f32>(64, 64, 64 * 64 - 1));
        let want = "image data has 4095 values, not 64x64";
        let e = track_pair(&bad, &ok, &cfg, None, None).unwrap_err();
        assert_eq!(text(e), want);
        let e = track_pair(&ok, &bad, &cfg, None, None).unwrap_err();
        assert_eq!(text(e), want);
        // too long as well, and before any other check: the sizes differ here too
        let long = short::<f32>(64, 64, 64 * 64 + 1);
        let e = track_pair(&long, &zeros(80, 64), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), "image data has 4097 values, not 64x64");
        // the mask is an input image as well; checked after both images
        let ones = |w, h| Image::from_vec(w, h, vec![true; w * h]);
        let bad_mask = short::<bool>(64, 64, 10);
        let e = track_pair(&ok, &ok, &cfg, Some(&bad_mask), None).unwrap_err();
        assert_eq!(text(e), "image data has 10 values, not 64x64");
        let e = track_pair(&bad, &ok, &cfg, Some(&bad_mask), None).unwrap_err();
        assert_eq!(text(e), want);
        // before the invalid config and the empty mask
        let bad_cfg = Config {
            poc_win: 48,
            ..Config::default()
        };
        let e = track_pair(&bad, &ok, &bad_cfg, Some(&ones(64, 64)), None).unwrap_err();
        assert_eq!(text(e), want);
        // the owned variant says the same
        let e = track_pair_owned(bad.clone(), ok.clone(), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), want);
        // sides whose product does not fit in usize are not a length either
        let huge = short::<f32>(usize::MAX, 2, 4);
        let e = track_pair(&huge, &huge, &cfg, None, None).unwrap_err();
        assert_eq!(
            text(e),
            format!("image data has 4 values, not {}x2", usize::MAX)
        );
        // an empty image with data is wrong, an empty one without is not this error
        let e = track_pair(&short::<f32>(0, 5, 3), &zeros(0, 5), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), "image data has 3 values, not 0x5");
        let e = track_pair(&zeros(0, 5), &zeros(0, 5), &cfg, None, None).unwrap_err();
        assert_eq!(text(e), "image smaller than poc_win (64): (5, 0)");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // tracks one pair: a full pipeline
    fn track_sequence_rejects_a_frame_or_mask_whose_data_is_not_w_times_h_long() {
        let cfg = Config::default();
        let (z, bad) = (zeros(64, 64), short::<f32>(64, 64, 100));
        let want = "image data has 100 values, not 64x64";
        assert_eq!(
            first_error(vec![bad.clone(), z.clone()], &cfg, None).unwrap(),
            want
        );
        // the first frame is checked before the config and the mask
        let bad_cfg = Config {
            grid_step: 0,
            ..Config::default()
        };
        assert_eq!(
            first_error(vec![bad.clone()], &bad_cfg, None).unwrap(),
            want
        );
        // the mask is checked at the start, before the config
        let bad_mask = short::<bool>(64, 64, 7);
        assert_eq!(
            first_error(vec![z.clone(), z.clone()], &cfg, Some(&bad_mask)).unwrap(),
            "image data has 7 values, not 64x64"
        );
        assert_eq!(
            first_error(vec![z.clone()], &bad_cfg, Some(&bad_mask)).unwrap(),
            "image data has 7 values, not 64x64"
        );
        // a later frame is checked when it arrives, before its shape: the earlier frame is tracked
        let mut it = track_sequence(
            vec![z.clone(), z.clone(), short::<f32>(48, 64, 5)].into_iter(),
            &cfg,
            None,
        );
        assert!(it.next().unwrap().is_ok());
        assert_eq!(
            text(it.next().unwrap().unwrap_err()),
            "image data has 5 values, not 48x64"
        );
        assert!(it.next().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn track_sequence_errors_have_the_texts_of_the_reference() {
        let cfg = Config::default();
        assert_eq!(first_error(vec![], &cfg, None).unwrap(), "no frames");
        let z = zeros(64, 64);
        assert_eq!(
            first_error(vec![z.clone(), zeros(48, 64)], &cfg, None).unwrap(),
            "frame 1 has shape (64, 48), expected (64, 64)"
        );
        let wide = Image::from_vec(48, 64, vec![true; 48 * 64]);
        assert_eq!(
            first_error(vec![z.clone(), z.clone()], &cfg, Some(&wide)).unwrap(),
            "ROI mask shape (64, 48) differs from the frame shape (64, 64)"
        );
        let empty = Image::from_vec(64, 64, vec![false; 64 * 64]);
        assert_eq!(
            first_error(vec![z.clone(), z.clone()], &cfg, Some(&empty)).unwrap(),
            "ROI mask is empty"
        );
        let bad_cfg = Config {
            grid_step: 0,
            ..Config::default()
        };
        assert_eq!(
            first_error(vec![z.clone()], &bad_cfg, None).unwrap(),
            "grid_step >= 1 and strain_win odd >= 5 required"
        );
        // the size check of the first pair
        let small = zeros(128, 32);
        assert_eq!(
            first_error(vec![small.clone(), small], &cfg, None).unwrap(),
            "image smaller than poc_win (64): (32, 128)"
        );
        // a shape mismatch is found when the frame arrives, after the earlier frames were tracked
        // (64 x 64 passes the size check of frame 1)
        let mut it = track_sequence(
            vec![z.clone(), z.clone(), zeros(48, 64)].into_iter(),
            &cfg,
            None,
        );
        assert!(it.next().unwrap().is_ok());
        assert_eq!(
            text(it.next().unwrap().unwrap_err()),
            "frame 2 has shape (64, 48), expected (64, 64)"
        );
        assert!(it.next().is_none());
        // frames too small for poc_win fail the size check at k = 1, before any shape check
        let mut it = track_sequence(
            vec![zeros(20, 20), zeros(20, 20), zeros(21, 20)].into_iter(),
            &cfg,
            None,
        );
        assert_eq!(
            text(it.next().unwrap().unwrap_err()),
            "image smaller than poc_win (64): (20, 20)"
        );
    }

    #[test]
    fn track_sequence_with_one_frame_yields_nothing() {
        let cfg = Config::default();
        assert!(first_error(vec![zeros(64, 64)], &cfg, None).is_none());
        let mut it = track_sequence(std::iter::once(zeros(64, 64)), &cfg, None);
        assert!(it.next().is_none() && it.next().is_none());
    }

    #[test]
    fn timing_serialises_like_the_python_dict() {
        let pair = Timing {
            coarse: 0.5,
            fine: 1.5,
            strain: 0.25,
            ..Timing::default()
        };
        assert_eq!(
            serde_json::to_string(&pair).unwrap(),
            r#"{"coarse":0.5,"fine":1.5,"strain":0.25}"#
        );
        let seq = Timing {
            reference: Some(2),
            mean_conf: Some(0.75),
            ..pair
        };
        assert_eq!(
            serde_json::to_string(&seq).unwrap(),
            r#"{"coarse":0.5,"fine":1.5,"strain":0.25,"reference":2,"mean_conf":0.75}"#
        );
    }

    // ---- dense start of the fine stage

    fn grid_of(xs: Vec<usize>, ys: Vec<usize>, f: impl Fn(usize, usize) -> (f32, f32)) -> Grid {
        let (nx, ny) = (xs.len(), ys.len());
        let (mut ux, mut uy) = (Image::new(nx, ny), Image::new(nx, ny));
        for (j, &y) in ys.iter().enumerate() {
            for (i, &x) in xs.iter().enumerate() {
                (ux.data[j * nx + i], uy.data[j * nx + i]) = f(x, y);
            }
        }
        let peak = Image::from_vec(nx, ny, vec![1.0; nx * ny]);
        Grid {
            xs,
            ys,
            ux,
            uy,
            peak,
        }
    }

    #[test]
    fn dense_start_interpolates_the_grid_and_adds_the_correction_to_a_guess() {
        let (w, h) = (50usize, 40usize);
        let g = grid_of(vec![10, 20, 30, 40], vec![8, 16, 24, 32], |x, y| {
            (0.1 * x as f32 + 0.02 * y as f32, 3.0 - 0.05 * y as f32)
        });
        // without a guess: the grid field, interpolated (and clamped outside the grid)
        let (gx, gy) = dense_start(&g, None, (w, h));
        for y in 0..h {
            for x in 0..w {
                let (u, v) = grid_sample(&g, x as f64, y as f64);
                assert_eq!((gx.get(x, y), gy.get(x, y)), (u as f32, v as f32));
            }
        }
        // a constant guess: guess + interpolated (field - guess) is the interpolated field
        let c = |v: f32| Image::from_vec(w, h, vec![v; w * h]);
        let (hx, hy) = dense_start(&g, Some((&c(5.0), &c(-2.0))), (w, h));
        for i in 0..w * h {
            assert!(
                (hx.data[i] - gx.data[i]).abs() < 1e-5 && (hy.data[i] - gy.data[i]).abs() < 1e-5
            );
        }
        // a guess that the grid confirms is returned as it is, between the nodes too
        let guess = |x: usize, y: usize| (0.3 * x as f32 - 0.2 * y as f32, 0.1 * x as f32 + 1.0);
        let g2 = grid_of(vec![10, 20, 30, 40], vec![8, 16, 24, 32], guess);
        let (mut ix, mut iy) = (Image::new(w, h), Image::new(w, h));
        for y in 0..h {
            for x in 0..w {
                (ix.data[y * w + x], iy.data[y * w + x]) = guess(x, y);
            }
        }
        let (kx, ky) = dense_start(&g2, Some((&ix, &iy)), (w, h));
        assert_eq!((kx.data, ky.data), (ix.data, iy.data));
    }

    #[test]
    fn dense_start_on_a_rectangle_equals_that_part_of_the_whole_image_start() {
        let (w, h) = (50usize, 40usize);
        let g = grid_of(vec![10, 20, 30, 40], vec![8, 16, 24, 32], |x, y| {
            (0.1 * x as f32 + 0.02 * y as f32, 3.0 - 0.05 * y as f32)
        });
        let mut rnd = lcg(11);
        let mut noise = || Image::from_vec(w, h, (0..w * h).map(|_| rnd() as f32 - 0.5).collect());
        let (ix, iy) = (noise(), noise());
        let rect = |x0, y0, x1, y1| Rect { x0, y0, x1, y1 };
        let rects = [
            rect(10, 5, 30, 25),  // interior
            rect(0, 3, 12, 20),   // left border
            rect(35, 6, 50, 30),  // right border
            rect(4, 0, 44, 9),    // top border
            rect(20, 31, 33, 40), // bottom border
            rect(17, 0, 18, 40),  // 1 px wide, full height
            rect(0, 22, 50, 23),  // 1 px high, full width
            rect(49, 39, 50, 40), // the last pixel
            rect(0, 0, 50, 40),   // the whole image
        ];
        for init in [None, Some((&ix, &iy))] {
            let (gx, gy) = dense_start(&g, init, (w, h));
            let source = start_grid(&g, init);
            for r in rects {
                let (rx, ry) = dense_start_on(&source, init, w, r);
                assert_eq!((rx.w, rx.h), (r.x1 - r.x0, r.y1 - r.y0));
                let part = |im: &Image<f32>| -> Vec<u32> {
                    (r.y0..r.y1)
                        .flat_map(|y| (r.x0..r.x1).map(move |x| (x, y)))
                        .map(|(x, y)| im.get(x, y).to_bits())
                        .collect()
                };
                let what = format!("{r:?}, init {}", init.is_some());
                assert_same(&bits(&rx), &part(&gx), &format!("x, {what}"));
                assert_same(&bits(&ry), &part(&gy), &format!("y, {what}"));
            }
        }
    }

    /// Blurred noise of `w` x `h` and the same content moved 2 px right and 1 px down.
    fn shifted_speckle(w: usize, h: usize) -> (Image<f32>, Image<f32>) {
        let (pw, ph) = (w + 8, h + 8);
        let speckle = crate::interp::gauss_blur(&crate::image::pseudo_random_image(pw, ph, 7), 1.5);
        let cut = |dx: usize, dy: usize| {
            let data = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| speckle.data[(y + dy) * pw + x + dx] as f32)
                .collect();
            Image::from_vec(w, h, data)
        };
        (cut(4, 4), cut(2, 3))
    }

    /// The tiled driver on 3 x 3 tiles (tile 128) with each tile's start formed on its own
    /// rectangle gives the bits of the whole-image start cut to the tiles, with and without a
    /// dense guess.
    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn the_tiled_driver_with_a_per_tile_start_equals_the_whole_image_start() {
        let (w, h) = (300usize, 260usize);
        let (i1, i2) = shifted_speckle(w, h);
        let cfg = Config {
            outer_iters: 1,
            ..Config::default()
        };
        let mut rnd = lcg(3);
        let mut guess =
            |c: f32| Image::from_vec(w, h, (0..w * h).map(|_| c + 0.2 * rnd() as f32).collect());
        let init_field = (guess(1.7), guess(0.6));
        for init in [None, Some((&init_field.0, &init_field.1))] {
            let g = coarse_flow(&i1, &i2, &cfg, init);
            let source = start_grid(&g, init);
            let start = |r: Rect| dense_start_on(&source, init, w, r);
            let (got, got_low) = refine_tiled_from(&i1, &i2, &start, &cfg, 128);
            let (gx, gy) = dense_start(&g, init, (w, h));
            let (want, want_low) = crate::tile::refine_tiled_low(&i1, &i2, &gx, &gy, &cfg, 128);
            let what = format!("init {}", init.is_some());
            assert_same(&bits(&got.ux), &bits(&want.ux), &format!("ux, {what}"));
            assert_same(&bits(&got.uy), &bits(&want.uy), &format!("uy, {what}"));
            assert_same(
                &bits(&got.conf),
                &bits(&want.conf),
                &format!("conf, {what}"),
            );
            assert_same(&got_low.data, &want_low.data, &format!("low, {what}"));
        }
    }

    /// A pair wider than `TILE` (two tiles), with and without a dense guess: `pair_flow`, which
    /// lets the tiled fine stage form each tile's start, gives the bits of the path before that
    /// change (the whole-image start cut to the tiles).
    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn a_tiled_pair_with_the_start_formed_per_tile_equals_the_whole_image_start() {
        let (w, h) = (TILE + 76, 300usize);
        // the content of the first image appears 2 px right and 1 px down in the second
        let (i1, i2) = shifted_speckle(w, h);
        let cfg = Config {
            outer_iters: 2,
            ..Config::default()
        };
        // a guess that varies from pixel to pixel, so that a start taken from the wrong place
        // shows (the coarse field of a pure shift is the same everywhere)
        let mut rnd = lcg(3);
        let mut guess =
            |c: f32| Image::from_vec(w, h, (0..w * h).map(|_| c + 0.2 * rnd() as f32).collect());
        let init_field = (guess(1.7), guess(0.6));
        for init in [None, Some((&init_field.0, &init_field.1))] {
            let got = pair_flow(&i1, &i2, &cfg, init);
            let g = coarse_flow(&i1, &i2, &cfg, init);
            let (gx, gy) = dense_start(&g, init, (w, h));
            let (want, want_low) = crate::tile::refine_tiled_low(&i1, &i2, &gx, &gy, &cfg, TILE);
            let what = format!("init {}", init.is_some());
            assert_same(&bits(&got.raw_ux), &bits(&want.ux), &format!("ux, {what}"));
            assert_same(&bits(&got.raw_uy), &bits(&want.uy), &format!("uy, {what}"));
            assert_same(
                &bits(&got.conf),
                &bits(&want.conf),
                &format!("conf, {what}"),
            );
            assert_same(&got.low.data, &want_low.data, &format!("low, {what}"));
            // the motion was found: the comparison is not between two failures
            let near = want
                .ux
                .data
                .iter()
                .filter(|&&u| (f64::from(u) - 2.0).abs() < 0.05)
                .count();
            assert!(
                near > w * h / 2,
                "{what}: {near} of {} px near ux = 2",
                w * h
            );
        }
    }

    // ---- band heights

    #[test]
    fn band_rows_is_band_pixels_over_the_width_between_32_and_band_rows() {
        for (w, want) in [
            (0, BAND_ROWS),
            (1, BAND_ROWS),
            (512, BAND_ROWS),
            (1024, BAND_ROWS),
            (1025, 511),
            (6002, 87),
            (6240, 84),
            (16384, 32),
            (16385, 32),
            (1 << 20, 32),
        ] {
            assert_eq!(band_rows(w), want, "w = {w}");
        }
    }

    #[test]
    fn every_stage_band_is_at_least_twice_its_halo() {
        let cfg = |strain_win: usize, edge_guard: f64| Config {
            strain_win,
            edge_guard,
            ..Config::default()
        };
        for c in [
            Config::default(),
            cfg(255, -1.0),
            cfg(15, 500.0),
            cfg(255, 500.0),
            cfg(15, 0.0),
        ] {
            for h in [64usize, 300, 4160] {
                for band in [1usize, 32, 84, 512, 5000] {
                    let (gb, sb) = stage_bands(&c, h, band);
                    let (gh, sh) = (guard_halo(&c, h), strain_halo(c.strain_win));
                    let what = format!(
                        "strain_win {} edge_guard {} h {h} band {band}",
                        c.strain_win, c.edge_guard
                    );
                    // never smaller than asked, and a band reads at most twice its own rows
                    assert!(gb >= band && gb >= 2 * gh, "guard {gb}: {what}");
                    assert!(sb >= band && sb >= 2 * sh, "strain {sb}: {what}");
                    assert!(gb + 2 * gh <= 2 * gb && sb + 2 * sh <= 2 * sb, "{what}");
                    // a band that is high enough already is kept
                    assert_eq!(gb, band.max(2 * gh), "{what}");
                    assert_eq!(sb, band.max(2 * sh), "{what}");
                }
            }
        }
        // the cases of the review: 6240 px wide (84 rows asked) with a 255 px strain window, and
        // with a 500 px guard on 4160 rows
        let rows = band_rows(6240);
        assert_eq!(stage_bands(&Config::default(), 4160, rows), (84, 84));
        assert_eq!(stage_bands(&cfg(255, -1.0), 4160, rows), (84, 256));
        assert_eq!(stage_bands(&cfg(15, 500.0), 4160, rows), (1002, 84));
        // the guard off: no halo
        assert_eq!(guard_halo(&cfg(15, 0.0), 4160), 0);
    }

    // ---- guard

    /// The rule of the reference written out with brute force distances: bad = low | outside the
    /// ROI | within `guard` of a source (large low component, ROI exterior) | closer than `guard`
    /// to the image border.
    fn reference_bad(low: &Image<bool>, cfg: &Config, mask: Option<&Image<bool>>) -> Vec<bool> {
        let (w, h) = (low.w, low.h);
        let guard = cfg.edge_guard_px();
        let mut src = large_components(low, cfg.edge_guard_min_area_px()).data;
        let mut bad = low.data.clone();
        if let Some(m) = mask {
            for i in 0..w * h {
                bad[i] |= !m.data[i];
                src[i] |= !m.data[i];
            }
        }
        if guard > 0.0 {
            let sources: Vec<(i64, i64)> = (0..w * h)
                .filter(|&i| src[i])
                .map(|i| ((i % w) as i64, (i / w) as i64))
                .collect();
            for (i, flag) in bad.iter_mut().enumerate() {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let near = sources
                    .iter()
                    .any(|&(sx, sy)| (((x - sx).pow(2) + (y - sy).pow(2)) as f64).sqrt() <= guard);
                let edge = x.min(y).min(w as i64 - 1 - x).min(h as i64 - 1 - y);
                *flag |= near || (edge as f64) < guard;
            }
        }
        bad
    }

    fn guard_cfg(guard: f64, min_area: f64) -> Config {
        Config {
            edge_guard: guard,
            edge_guard_min_area: min_area,
            ..Config::default()
        }
    }

    fn block(w: usize, h: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> Image<bool> {
        let data = (0..w * h)
            .map(|i| (x0..x1).contains(&(i % w)) && (y0..y1).contains(&(i / w)))
            .collect();
        Image::from_vec(w, h, data)
    }

    #[test]
    fn guard_surrounds_large_low_components_but_not_small_ones() {
        let (w, h) = (40, 30);
        let cfg = guard_cfg(3.0, 20.0);
        // 4 x 6 = 24 >= 20: a source; (24, 12) is 3 from the block, (25, 12) is 4
        let big = block(w, h, 18, 22, 10, 16);
        let bad = invalid_pixels(&big, &cfg, None, BAND_ROWS);
        assert!(bad.get(24, 12) && !bad.get(25, 12));
        assert!(bad.get(23, 17) && !bad.get(24, 16)); // sqrt(8) <= 3, sqrt(10) > 3
        assert!(bad.get(20, 12) && bad.get(20, 7) && !bad.get(20, 6));
        assert_same(&bad.data, &reference_bad(&big, &cfg, None), "big");
        // 4 x 4 = 16 < 20: only the pixels themselves (and the image border)
        let small = block(w, h, 18, 22, 10, 14);
        let bad = invalid_pixels(&small, &cfg, None, BAND_ROWS);
        assert!(bad.get(20, 12) && !bad.get(22, 12) && !bad.get(20, 14) && !bad.get(17, 11));
        assert_same(&bad.data, &reference_bad(&small, &cfg, None), "small");
    }

    #[test]
    fn guard_distance_is_inclusive_and_exact_for_pythagorean_triples() {
        // guard 5: (3, 4) and (5, 0) are at distance exactly 5 (invalid), (4, 4) is not
        let (w, h) = (41, 41);
        let cfg = guard_cfg(5.0, 1.0);
        let mut low = Image::new(w, h);
        low.set(20, 20, true);
        let bad = invalid_pixels(&low, &cfg, None, BAND_ROWS);
        assert!(bad.get(23, 24) && bad.get(25, 20) && bad.get(20, 15) && bad.get(17, 16));
        assert!(!bad.get(24, 24) && !bad.get(26, 20) && !bad.get(25, 22));
        assert_same(&bad.data, &reference_bad(&low, &cfg, None), "pythagorean");
    }

    #[test]
    fn guard_marks_the_image_border_strictly_closer_than_the_guard() {
        let (w, h) = (30, 20);
        let none = Image::new(w, h);
        for guard in [3.0, 3.5, 2.25] {
            let cfg = guard_cfg(guard, 1e9);
            let bad = invalid_pixels(&none, &cfg, None, BAND_ROWS);
            assert_same(
                &bad.data,
                &reference_bad(&none, &cfg, None),
                &format!("guard {guard}"),
            );
            // the pixel exactly `guard` from the border is fine when guard is a whole number
            // the pixel 3 px from the border is bad exactly when 3 < guard
            assert_eq!(bad.get(15, 3), 3.0 < guard, "guard {guard}");
            assert!(bad.get(15, 2) && bad.get(2, 10) && bad.get(w - 3, 10));
        }
        // guard 0: nothing but low confidence and the ROI
        let cfg = guard_cfg(0.0, 1.0);
        let mut low = Image::new(w, h);
        low.set(5, 5, true);
        let bad = invalid_pixels(&low, &cfg, None, BAND_ROWS);
        assert_eq!(bad.data.iter().filter(|&&b| b).count(), 1);
    }

    #[test]
    fn roi_exterior_is_a_guard_source() {
        let (w, h) = (40, 30);
        let cfg = guard_cfg(3.0, 1e9);
        let none = Image::new(w, h);
        let roi = block(w, h, 5, 35, 5, 25);
        let bad = invalid_pixels(&none, &cfg, Some(&roi), BAND_ROWS);
        // exterior pixel (4, 12) is at distance 1..: inside pixels up to distance 3 are invalid
        assert!(bad.get(4, 12) && bad.get(5, 12) && bad.get(7, 12) && !bad.get(8, 12));
        assert!(!bad.get(20, 12));
        assert_same(&bad.data, &reference_bad(&none, &cfg, Some(&roi)), "roi");
    }

    /// A displacement field with NaN holes, low-confidence regions of every kind (large across
    /// band boundaries, small, scattered) and an ROI with a hole.
    fn banded_inputs() -> (Image<f32>, Image<f32>, Image<bool>, Image<bool>) {
        let (w, h) = (300usize, 200usize);
        let mut rnd = lcg(5);
        let mut ux = Image::new(w, h);
        let mut uy = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f64, y as f64);
                ux.data[y * w + x] = (0.01 * xf + 0.002 * yf + 0.3 * (xf / 9.0).sin()) as f32;
                uy.data[y * w + x] = (-0.003 * xf + 1e-4 * yf * yf + 0.2 * (yf / 7.0).cos()) as f32;
            }
        }
        for (x0, x1, y0, y1) in [(30, 40, 20, 31), (200, 215, 60, 75), (100, 110, 120, 140)] {
            for y in y0..y1 {
                for x in x0..x1 {
                    ux.data[y * w + x] = f32::NAN;
                    uy.data[y * w + x] = f32::NAN;
                }
            }
        }
        for _ in 0..150 {
            let i = (rnd() * (w * h) as f64) as usize;
            ux.data[i] = f32::NAN;
        }
        let mut low = Image::new(w, h);
        for (x0, x1, y0, y1) in [
            (60, 90, 40, 65),    // large, inside one band
            (140, 150, 20, 180), // a long strip over every band boundary
            (250, 262, 60, 72),  // 144 px: too small to guard
            (20, 60, 150, 190),  // large, near the bottom
            (200, 230, 68, 90),  // large, its top edge 4 rows below the band edge at 64
            (70, 100, 136, 150), // large, its top edge 8 rows below the band edge at 128
        ] {
            for y in y0..y1 {
                for x in x0..x1 {
                    low.data[y * w + x] = true;
                }
            }
        }
        for _ in 0..300 {
            low.data[(rnd() * (w * h) as f64) as usize] = true;
        }
        let mut roi = block(w, h, 12, 285, 8, 190);
        for y in 100..116 {
            for x in 220..240 {
                roi.data[y * w + x] = false;
            }
        }
        (ux, uy, low, roi)
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn banded_guard_and_strain_equal_the_whole_image_bit_for_bit() {
        let (ux, uy, low, roi) = banded_inputs();
        let cfg = Config::default();
        let (w, h) = (ux.w, ux.h);
        // the whole-image computation, written out
        let want_bad = reference_bad(&low, &cfg, Some(&roi));
        let (wx, wy) = (
            masked(&ux, &Image::from_vec(w, h, want_bad.clone())),
            masked(&uy, &Image::from_vec(w, h, want_bad.clone())),
        );
        let want_strain = strain_fields(&displacement_gradient(&wx, &wy, cfg.strain_win));
        let n_bad = want_bad.iter().filter(|&&b| b).count();
        assert!(
            n_bad > 5000 && n_bad < w * h * 3 / 4,
            "unbalanced test field: {n_bad}"
        );
        let n_nan = want_strain.exx.data.iter().filter(|v| v.is_nan()).count();
        assert!(
            n_nan > 0 && n_nan < w * h,
            "strain is all or nothing: {n_nan}"
        );
        for band in [64, 1, 7, 13, 100, 199, 200, 5000] {
            let got = invalid_pixels(&low, &cfg, Some(&roi), band);
            assert_same(&got.data, &want_bad, &format!("guard, band {band}"));
            // the strain at this band height itself (finish raises small bands, see below)
            let direct = strain_banded(&wx, &wy, cfg.strain_win, band, StrainOpts::default());
            for (i, (g, e)) in strain_images(&direct)
                .into_iter()
                .zip(strain_images(&want_strain))
                .enumerate()
            {
                assert_eq!((g.w, g.h), (w, h));
                let what = format!("strain_banded field {i}, band {band}");
                assert_same(&bits(g), &bits(e), &what);
            }
            let fin = finish(
                &ux,
                &uy,
                &low,
                &cfg,
                Some(&roi),
                band,
                StrainOpts::default(),
            );
            assert_same(&bits(&fin.ux), &bits(&wx), &format!("ux, band {band}"));
            assert_same(&bits(&fin.uy), &bits(&wy), &format!("uy, band {band}"));
            for (i, (g, e)) in strain_images(&fin.strain)
                .into_iter()
                .zip(strain_images(&want_strain))
                .enumerate()
            {
                assert_eq!((g.w, g.h), (w, h));
                assert_same(
                    &bits(g),
                    &bits(e),
                    &format!("strain field {i}, band {band}"),
                );
            }
            // without the small-strain maps: those are empty, the other five are the same
            if band == 13 {
                let off = StrainOpts { small: false };
                let lean = strain_banded(&wx, &wy, cfg.strain_win, band, off);
                for (i, (g, e)) in strain_images(&lean)
                    .into_iter()
                    .zip(strain_images(&want_strain))
                    .enumerate()
                {
                    if (3..6).contains(&i) {
                        assert_eq!((g.w, g.h, g.data.len()), (w, 0, 0), "small field {i}");
                        assert_eq!(g.data.capacity(), 0, "small field {i} holds memory");
                    } else {
                        assert_same(&bits(g), &bits(e), &format!("lean field {i}"));
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn banded_guard_equals_the_whole_image_for_a_wide_guard() {
        // a guard of 40 px reaches across several bands of 16 rows
        let (_, _, low, roi) = banded_inputs();
        let cfg = guard_cfg(40.0, 100.0);
        let want = reference_bad(&low, &cfg, Some(&roi));
        for band in [16, 31, 200] {
            assert_eq!(
                invalid_pixels(&low, &cfg, Some(&roi), band).data,
                want,
                "band {band}"
            );
        }
    }

    #[test]
    fn within_restores_the_integer_square_distance() {
        // sqrt(26) = 5.0990195.. is stored as f32; the decision must be that of float64
        let valid = Image::from_vec(30, 30, (0..900).map(|i| i != 15 * 30 + 15).collect());
        let d = distance_to_invalid(&valid);
        for (x, y, d2) in [(20usize, 15usize, 25.0f64), (20, 16, 26.0), (18, 19, 25.0)] {
            let got = d.data[y * 30 + x];
            assert_eq!(f64::from(got).powi(2).round(), d2);
            assert_eq!(within(got, 5.0), d2 <= 25.0, "d2 = {d2}");
        }
        assert!(within(0.0, 0.5) && !within(1.0, 0.99));
    }

    /// Python's `test_mean_conf_averages_over_reportable_pixels_only`. The border strip (confidence
    /// 0 by the border rule of the fine stage) is a geometric exclusion, not a tracking-quality
    /// signal.
    #[test]
    fn mean_conf_averages_over_reportable_pixels_only() {
        let n = 128usize;
        let cfg = Config::default();
        assert_eq!(border_zone(&cfg), 16.0);
        let image = |f: &dyn Fn(usize, usize) -> f32| {
            Image::from_vec(n, n, (0..n * n).map(|i| f(i % n, i / n)).collect())
        };
        let inner = |x: usize, y: usize| (16..n - 16).contains(&x) && (16..n - 16).contains(&y);
        let conf = image(&|x, y| if inner(x, y) { 0.9 } else { 0.0 });
        let zero = Image::<f32>::new(n, n);
        let z = (zero.clone(), zero.clone());
        let mean = |conf: &Image<f32>, raw: (&Image<f32>, &Image<f32>), u0r: &_, roi| {
            mean_conf(conf, raw, u0r, roi, &cfg)
        };
        let near = |got: f64, want: f64| {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        };
        let still = (&zero, &zero);
        near(mean(&conf, still, &z, None), 0.9);
        // inside the reportable zone a low confidence does count
        let low = image(&|x, y| match (inner(x, y), x < 64) {
            (true, true) => 0.1,
            (true, false) => 0.9,
            _ => 0.0,
        });
        near(
            mean(&low, still, &z, None),
            (48.0 * 0.1 + 48.0 * 0.9) / 96.0,
        );
        // a zero inside the reportable zone is a tracking failure, not a border effect: it counts
        let hole = image(&|x, y| {
            if inner(x, y) && !((40..60).contains(&x) && (40..60).contains(&y)) {
                0.9
            } else {
                0.0
            }
        });
        near(
            mean(&hole, still, &z, None),
            0.9 * (96.0 * 96.0 - 400.0) / (96.0 * 96.0),
        );
        // the ROI restricts the average further; a ROI entirely in the strip has no reportable pixel
        let roi = Image::from_vec(n, n, (0..n * n).map(|i| i % n < 80).collect());
        near(
            mean(&low, still, &z, Some(&roi)),
            (48.0 * 0.1 + 16.0 * 0.9) / 64.0,
        );
        let strip = Image::from_vec(n, n, (0..n * n).map(|i| i % n < 10).collect());
        assert_eq!(mean(&conf, still, &z, Some(&strip)), 0.0);
        // matches that leave image 2 are not reportable either (border rule): x + 8 > 127 - 16 for
        // x >= 104, where the confidence is 0 and does not count
        let right = Image::from_vec(n, n, vec![8.0f32; n * n]);
        let conf_shift = image(&|x, y| if inner(x, y) && x < 104 { 0.9 } else { 0.0 });
        near(mean(&conf_shift, (&right, &zero), &z, None), 0.9);
        // u0r moves the frame-0 grid into the leg's reference: frame-0 pixels whose position falls in
        // the strip of the leg's image do not count
        let u0r = (Image::from_vec(n, n, vec![5.0f32; n * n]), zero.clone());
        near(mean(&conf, still, &u0r, None), 0.9);
        // the ROI is tested at the frame-0 pixel X, not at X + u0r(X): the confidence grows with
        // the column x' of the leg (x' / 200), frame-0 pixels x < 40 land on x' in [30, 70)
        let ramp = image(&|x, y| if inner(x, y) { x as f32 / 200.0 } else { 0.0 });
        let by30 = (Image::from_vec(n, n, vec![30.0f32; n * n]), zero.clone());
        let left = Image::from_vec(n, n, (0..n * n).map(|i| i % n < 40).collect());
        near(mean(&ramp, still, &by30, Some(&left)), 49.5 / 200.0);
        // a fractional u0r at the edge of the strip: the 0/1 reportable image is interpolated and a
        // pixel counts only where the sample exceeds 0.5. With u0r = 5.5 the frame-0 pixel x = 10
        // lands on 15.5, half way between the last column of the strip (15, mask 0) and the first
        // reportable one (16): the sample is exactly 0.5 and does not count (nor does x = 106,
        // half way to the strip on the right, 111.5). Every counted pixel has the confidence 0.9;
        // a `>= 0.5` cut would add the two edge columns at confidence 0.45 each.
        let half = (Image::from_vec(n, n, vec![5.5f32; n * n]), zero.clone());
        near(mean(&conf, still, &half, None), 0.9);
        // nothing reportable: 0
        let out = (Image::from_vec(n, n, vec![500.0f32; n * n]), zero.clone());
        assert_eq!(mean(&conf, still, &out, None), 0.0);
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// `sample_bilinear` indexes only inside the image (any out-of-bounds access fails the
    /// proof) and is NaN exactly outside `[0, w - 1] x [0, h - 1]` or for a non-finite
    /// coordinate, for any `f64` pair. The image is built without a loop (a constant array cut
    /// to size) so that no symbolic loop over a `Vec` enters the model.
    #[kani::proof]
    fn k3_bilinear_never_out_of_bounds() {
        let w: usize = kani::any();
        let h: usize = kani::any();
        kani::assume(w >= 1 && w <= 4 && h >= 1 && h <= 4);
        let mut data = Vec::from([1.0f32; 16]);
        data.truncate(w * h);
        let a = Image::from_vec(w, h, data);
        let x: f64 = kani::any();
        let y: f64 = kani::any();
        let v = sample_bilinear(&a, x, y);
        let inside = x.is_finite()
            && y.is_finite()
            && x >= 0.0
            && y >= 0.0
            && x <= (w - 1) as f64
            && y <= (h - 1) as f64;
        assert!(inside == !v.is_nan());
    }
}

//! Crack model "xdic" (experimental): a displacement field with a jump across explicit crack lines, solved per side of
//! each line by local-phase matching (the X-DIC idea).
//!
//! Port of `python/optflow_ref/xdic.py`; that file (its module doc, the docstrings and the
//! constants) is the specification, and `_track_pair_xdic` / the sequence legs of `track.py` say
//! how a leg is put together (`crate::track`). The parts:
//!
//! * [`geom`]: polylines (signed distance field, resampling, extension, merging).
//! * [`detect`]: the field-jump detector on the baseline result.
//! * [`solve`]: the per-side solve and the geometry update (`refine_xdic`).
//! * [`outputs`]: crack or band (`model_gain`), the carried geometry (`advance_lines`,
//!   `lines_to_frame0`) and the outputs of a leg (`jump_along`, the one-sided strain).
//! * [`params`]: all constants of the method, named as in `xdic.py`.
//!
//! The baseline iterations of the method run on the whole image at once and the per-side solve on
//! the strip around the lines, in `f64` (images and filter responses are `f32` as everywhere else),
//! so `crack_model = "xdic"` has no tiles: it takes images of at most [`MAX_PIXELS`] pixels (crop
//! to the region of interest with an ROI) and the CPU fine engine.

// A transcription of the numpy code of the reference: its index loops over several arrays at once and
// its long argument lists are kept as they are, so that the two read side by side.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

pub mod detect;
pub mod geom;
pub mod num;
pub mod outputs;
pub mod params;
pub mod solve;

/// Largest image of `crack_model = "xdic"`, in pixels (a 4K frame): the memory grows with the area,
/// about 1.1 GB per million pixels (measured on 1920 x 1080: 2.25 GB).
pub const MAX_PIXELS: usize = 3840 * 2160;

/// A crack polyline: vertices `(x, y)` (column, row) in the coordinates of a leg's reference.
pub type Poly = Vec<[f64; 2]>;

/// A per-pixel result of the baseline or the per-side solve (`phaseflow.Flow` of the reference,
/// the fields `refine_xdic` and `model_gain` use): displacement, confidence (before any
/// validity decision, 0 in the border strips), the residual of the last solve and its affine
/// part `slope` (`[dux/dx, duy/dx, dux/dy, duy/dy]`). Row-major `w * h`, `f64`.
#[derive(Clone, Debug)]
pub struct Flow {
    pub w: usize,
    pub h: usize,
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    pub conf: Vec<f64>,
    pub resid: Vec<f64>,
    pub slope: Vec<[f64; 4]>,
}

/// COD / CSD samples along one crack line (`jump_along`): the points every `LINE_STEP` px of arc
/// length inside the image, their unit normals and arc lengths, the opening `cod` and slip `csd`
/// (smoothed: median over `+-JUMP_SMOOTH` px), their raw values, and the extrapolated
/// displacement of either side at the points (`u_plus[0]`/`[1]` = x/y components).
#[derive(Clone, Debug)]
pub struct JumpLine {
    pub pts: Vec<[f64; 2]>,
    pub n: Vec<[f64; 2]>,
    pub s: Vec<f64>,
    pub cod: Vec<f64>,
    pub csd: Vec<f64>,
    pub cod_raw: Vec<f64>,
    pub csd_raw: Vec<f64>,
    pub u_plus: [Vec<f64>; 2],
    pub u_minus: [Vec<f64>; 2],
}

/// The crack geometry of a leg or a sequence frame (`crack_lines` of the reference's `Result`):
/// the lines in the coordinates of the result's reference, the same lines advanced into the
/// target frame (a leg's carried geometry; `None` for a sequence frame), the jump samples, the
/// lines the detector found that are continuous bands, and the lines dropped for a missing jump.
#[derive(Clone, Debug, Default)]
pub struct CrackLines {
    pub lines: Vec<Poly>,
    pub lines_next: Option<Vec<Poly>>,
    pub jumps: Vec<JumpLine>,
    pub bands: Vec<Poly>,
    pub rejected: Vec<Poly>,
}

/// The crack outputs of a result: the COD / CSD / normal maps on the line pixels and the lines
/// (the `crack` and `crack_lines` of the reference's `Result`).
#[derive(Clone, Debug)]
pub struct CrackOut {
    /// None when the result has no crack line (only bands or rejected lines).
    pub maps: Option<outputs::CrackMaps>,
    pub lines: CrackLines,
}

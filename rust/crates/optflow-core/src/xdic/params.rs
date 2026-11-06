//! Constants of `crack_model = "xdic"`, named as in `python/optflow_ref/xdic.py`: the detector
//! (`DET_*`), the geometry update (`UPD_*`), the COD / CSD estimate (`JUMP_*`, `LINE_STEP`), the
//! mixed-pixel layer (`MIX_PX`) and the crack-or-band decision (`CRACK_*`). A validation on new
//! data is running and these may change: keep them in step with `xdic.py`, and only here.

/// Radius of the half-disk fits [px].
pub const DET_R: f64 = 36.0;
/// Significance `|J| / sigma_J`.
pub const DET_ZMIN: f64 = 6.0;
/// Minimum jump [px].
pub const DET_JMIN: f64 = 0.3;
/// Candidates: pixels within this distance of a large invalid hole [px] ...
pub const DET_GMAX: f64 = 12.0;
/// ... and valid pixels with a max shear strain above this.
pub const DET_GMAX_STRAIN: f64 = 0.05;
pub const DET_NTHETA: usize = 16;
pub const DET_MINPTS: f64 = 30.0;
pub const DET_STEP: usize = 4;
/// Half-disk pixels closer than this to the dividing line are left out [px].
pub const DET_LINE_GUARD: f64 = 8.0;
pub const DET_BIN: f64 = 12.0;
/// Half width of the band of the geometry update [px].
pub const UPD_BAND: f64 = 8.0;
pub const UPD_BIN: f64 = 16.0;
/// A pixel votes in the geometry update when `|e+ - e-|` exceeds this times the noise level.
pub const UPD_INFO: f64 = 2.0;
pub const JUMP_R: f64 = 16.0;
pub const JUMP_MIN: usize = 12;
pub const JUMP_SMOOTH: f64 = 32.0;
pub const JUMP_SKIP: f64 = 4.0;
pub const LINE_STEP: f64 = 2.0;
/// Width of the mixed-pixel layer painted for both sides [px].
pub const MIX_PX: f64 = 2.0;
/// A crack: the per-side solve removes at least 2/3 of the baseline's excess residual at the line ...
pub const CRACK_GAIN: f64 = 3.0;
/// ... and that excess stands above the noise.
pub const CRACK_E0_MIN: f64 = 1.5;
/// Or: the mean residual within `CRACK_MEAN_PX` of the line falls by this factor.
pub const CRACK_MEAN: f64 = 2.0;
pub const CRACK_MEAN_PX: f64 = 8.0;

//! Coarse displacement on a grid by windowed, band-limited phase-only correlation (POC).
//!
//! Port of `python/optflow_ref/poc.py`; that file is the specification. Windows are cut from `f32`
//! images and handled in `f64` (windows, FFT, correlation surface); results are stored as `f32`.
//! A window is `(height, width)` everywhere, like Python's `(wy, wx)`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

use crate::config::Config;
use crate::image::Image;
use crate::interp::gauss_blur;
use crate::select::median;

/// Coarse displacement field on a regular grid; `ux`, `uy` and `peak` are `ys.len()` rows of
/// `xs.len()` columns.
#[derive(Clone, Debug)]
pub struct Grid {
    /// Pixel x of the grid columns.
    pub xs: Vec<usize>,
    /// Pixel y of the grid rows.
    pub ys: Vec<usize>,
    pub ux: Image<f32>,
    pub uy: Image<f32>,
    /// Normalised POC peak height, 0 = invalid.
    pub peak: Image<f32>,
}

/// POC re-centring iterations per grid point.
const POC_ITERS: usize = 3;
/// Most fill passes of `clean`.
const CLEAN_MAX_PASS: usize = 100;
/// Deepest pyramid `n_levels` can return (`h >> level` needs `level < 64`).
const MAX_LEVELS: u32 = 63;

/// Everything `poc_shift` needs for one window shape: FFT plans, Hann window and the low-pass mask
/// (`|f| <= 0.25` per axis), built once per shape.
struct PocPlan {
    h: usize,
    w: usize,
    row_fwd: Arc<dyn Fft<f64>>,
    row_inv: Arc<dyn Fft<f64>>,
    col_fwd: Arc<dyn Fft<f64>>,
    col_inv: Arc<dyn Fft<f64>>,
    scratch_len: usize,
    hann: Vec<f64>,
    lowpass: Vec<f64>,
    lowpass_mean: f64,
}

/// `0.5 - 0.5 cos(2 pi i / n)`, `i = 0..n`.
fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect()
}

/// `|numpy.fft.fftfreq(n)|`, computed the way numpy does (integer index times `1 / n`).
fn abs_fftfreq(n: usize) -> Vec<f64> {
    let val = 1.0 / n as f64;
    (0..n)
        .map(|k| {
            let m = if k < n.div_ceil(2) {
                k as f64
            } else {
                k as f64 - n as f64
            };
            (m * val).abs()
        })
        .collect()
}

impl PocPlan {
    fn new(h: usize, w: usize) -> Self {
        assert!(h >= 3 && w >= 3, "POC window {h}x{w} is too small");
        let mut planner = FftPlanner::<f64>::new();
        let (row_fwd, row_inv) = (planner.plan_fft_forward(w), planner.plan_fft_inverse(w));
        let (col_fwd, col_inv) = (planner.plan_fft_forward(h), planner.plan_fft_inverse(h));
        let scratch_len = [&row_fwd, &row_inv, &col_fwd, &col_inv]
            .iter()
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let (hy, hx) = (hann(h), hann(w));
        let hann = hy
            .iter()
            .flat_map(|&a| hx.iter().map(move |&b| a * b))
            .collect();
        let (fy, fx) = (abs_fftfreq(h), abs_fftfreq(w));
        let lowpass: Vec<f64> = fy
            .iter()
            .flat_map(|&a| {
                fx.iter()
                    .map(move |&b| if a <= 0.25 && b <= 0.25 { 1.0 } else { 0.0 })
            })
            .collect();
        let lowpass_mean = lowpass.iter().sum::<f64>() / (h * w) as f64;
        Self {
            h,
            w,
            row_fwd,
            row_inv,
            col_fwd,
            col_inv,
            scratch_len,
            hann,
            lowpass,
            lowpass_mean,
        }
    }

    /// In-place 2-D transform (rows, then columns) without normalisation.
    fn transform(&self, data: &mut [Complex64], inverse: bool) {
        let (row, col) = if inverse {
            (&self.row_inv, &self.col_inv)
        } else {
            (&self.row_fwd, &self.col_fwd)
        };
        let mut scratch = vec![Complex64::default(); self.scratch_len];
        row.process_with_scratch(data, &mut scratch);
        let mut t = vec![Complex64::default(); data.len()];
        for y in 0..self.h {
            for x in 0..self.w {
                t[x * self.h + y] = data[y * self.w + x];
            }
        }
        col.process_with_scratch(&mut t, &mut scratch);
        for y in 0..self.h {
            for x in 0..self.w {
                data[y * self.w + x] = t[x * self.h + y];
            }
        }
    }

    /// `fft2((win - mean) * hann)`.
    fn spectrum(&self, win: &[f64]) -> Vec<Complex64> {
        let mean = win.iter().sum::<f64>() / win.len() as f64;
        let mut data: Vec<Complex64> = win
            .iter()
            .zip(&self.hann)
            .map(|(&v, &h)| Complex64::new((v - mean) * h, 0.0))
            .collect();
        self.transform(&mut data, false);
        data
    }

    /// Shift `(dx, dy, peak)` of the second window against the first, from their spectra.
    fn shift(&self, fa: &[Complex64], fb: &[Complex64]) -> (f64, f64, f64) {
        let (h, w) = (self.h, self.w);
        let mut c: Vec<Complex64> = fa
            .iter()
            .zip(fb)
            .zip(&self.lowpass)
            .map(|((a, b), &m)| {
                let c = b * a.conj();
                (c / (c.norm() + 1e-12)) * m
            })
            .collect();
        self.transform(&mut c, true);
        let inv_n = 1.0 / (h * w) as f64;
        // fftshift: out[k] = in[(k - n / 2) mod n] along each axis
        let mut r = vec![0.0f64; h * w];
        for y in 0..h {
            let sy = (y + h - h / 2) % h;
            for x in 0..w {
                let sx = (x + w - w / 2) % w;
                r[y * w + x] = c[sy * w + sx].re * inv_n / self.lowpass_mean;
            }
        }
        // first maximum in row-major order; a NaN wins like in numpy
        let mut best = 0;
        for (k, &v) in r.iter().enumerate() {
            if v > r[best] || (v.is_nan() && !r[best].is_nan()) {
                best = k;
            }
        }
        let iy = (best / w).clamp(1, h - 2);
        let ix = (best % w).clamp(1, w - 2);
        let at = |y: usize, x: usize| r[y * w + x];
        let sx = parabola(at(iy, ix - 1), at(iy, ix), at(iy, ix + 1));
        let sy = parabola(at(iy - 1, ix), at(iy, ix), at(iy + 1, ix));
        (
            ix as f64 - (w / 2) as f64 + sx,
            iy as f64 - (h / 2) as f64 + sy,
            at(iy, ix),
        )
    }
}

/// Vertex offset of the parabola through three equally spaced samples.
fn parabola(m: f64, c: f64, p: f64) -> f64 {
    let den = m - 2.0 * c + p;
    if den == 0.0 {
        0.0
    } else {
        0.5 * (m - p) / den
    }
}

/// The shared `PocPlan` of a window shape (built on first use).
fn plan_for(h: usize, w: usize) -> Arc<PocPlan> {
    type PlanCache = Mutex<HashMap<(usize, usize), Arc<PocPlan>>>;
    static PLANS: OnceLock<PlanCache> = OnceLock::new();
    let mut plans = PLANS
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    plans
        .entry((h, w))
        .or_insert_with(|| Arc::new(PocPlan::new(h, w)))
        .clone()
}

/// The `wy` x `wx` window of `img` with top-left corner `(y0, x0)` as `f64`. The window must lie
/// inside the image.
fn window_f64(img: &Image<f32>, y0: usize, x0: usize, wy: usize, wx: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(wy * wx);
    for y in y0..y0 + wy {
        out.extend(
            img.data[y * img.w + x0..y * img.w + x0 + wx]
                .iter()
                .map(|&v| f64::from(v)),
        );
    }
    out
}

/// Shift `d` with `b(x) ~ a(x - d)` for equally shaped (possibly rectangular) windows.
/// Returns `(dx, dy, peak)` with `|dx| < width / 2`, `|dy| < height / 2`. Panics if the windows
/// differ in shape or a side is shorter than 3 px.
pub fn poc_shift(a: &Image<f32>, b: &Image<f32>) -> (f64, f64, f64) {
    assert_eq!((a.w, a.h), (b.w, b.h), "poc_shift: windows differ in shape");
    let plan = plan_for(a.h, a.w);
    let fa = plan.spectrum(&window_f64(a, 0, 0, a.h, a.w));
    let fb = plan.spectrum(&window_f64(b, 0, 0, b.h, b.w));
    plan.shift(&fa, &fb)
}

/// Grid positions along one axis of length `n` for a window of `win` px, as `(start, count)`: the
/// positions are `start + k * step` for `k < count`, `start = win / 2`, and the window
/// `[p - win / 2, p - win / 2 + win)` of each stays inside `[0, n)`. Panics if `step` is 0.
fn grid_axis(n: usize, win: usize, step: usize) -> (usize, usize) {
    // last position n - ceil(win / 2); Python uses n - floor(win / 2), which is the same for the
    // even windows used in practice and lets an odd window overrun the image by one pixel
    let stop = n.saturating_add(1).saturating_sub(win - win / 2);
    let start = win / 2;
    if start >= stop {
        return (start, 0);
    }
    (start, (stop - start - 1) / step + 1)
}

/// Grid columns `xs` and rows `ys` whose `win = (height, width)` window lies inside an `h` x `w`
/// image, `step` apart, starting at the first position with a full window. Panics if `step` is 0.
///
/// Identical to Python's `grid_points` for even windows; for an odd window the last position that
/// would overrun the image by a pixel is dropped. `coarse_flow` expects a validated `cfg.poc_win`
/// (a power of two, so even).
pub fn grid_points(
    h: usize,
    w: usize,
    step: usize,
    win: (usize, usize),
) -> (Vec<usize>, Vec<usize>) {
    assert!(step >= 1, "grid_points: step must be at least 1");
    let axis = |n: usize, win: usize| {
        let (start, count) = grid_axis(n, win, step);
        // start + k * step < n for k < count, so nothing overflows
        (0..count).map(|k| start + k * step).collect()
    };
    (axis(w, win.1), axis(h, win.0))
}

/// Top-left corner `(y0, x0)` of the `win = (height, width)` window centred on `(y, x)` and moved
/// by `(oy, ox)`, or `None` unless the whole window lies inside the `h` x `w` image.
pub fn shifted_window(
    y: usize,
    x: usize,
    win: (usize, usize),
    oy: i64,
    ox: i64,
    h: usize,
    w: usize,
) -> Option<(usize, usize)> {
    let corner = |c: usize, len: usize, o: i64| -> Option<usize> {
        let c = i64::try_from(c).ok()?;
        let half = i64::try_from(len / 2).ok()?;
        usize::try_from(c.checked_sub(half)?.checked_add(o)?).ok()
    };
    let y0 = corner(y, win.0, oy)?;
    let x0 = corner(x, win.1, ox)?;
    if y0.checked_add(win.0)? <= h && x0.checked_add(win.1)? <= w {
        Some((y0, x0))
    } else {
        None
    }
}

/// Python's `int(round(v))`: halves go to the even integer; NaN and infinities saturate.
fn round_half_even(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// POC displacement and peak at one grid point, starting from `start`.
fn poc_point(
    plan: &PocPlan,
    a: &Image<f32>,
    b: &Image<f32>,
    (y, x): (usize, usize),
    win: (usize, usize),
    start: (f64, f64),
) -> (f64, f64, f64) {
    let (mut dx, mut dy) = start;
    // a window of the first image that is not fully inside: nothing to correlate
    let Some((ay0, ax0)) = shifted_window(y, x, win, 0, 0, a.h, a.w) else {
        return (dx, dy, 0.0);
    };
    let fa = plan.spectrum(&window_f64(a, ay0, ax0, win.0, win.1));
    let mut pk = 0.0;
    for _ in 0..POC_ITERS {
        let (ox, oy) = (round_half_even(dx), round_half_even(dy));
        let Some((y0, x0)) = shifted_window(y, x, win, oy, ox, b.h, b.w) else {
            pk = 0.0;
            break;
        };
        let fb = plan.spectrum(&window_f64(b, y0, x0, win.0, win.1));
        let (sx, sy, p) = plan.shift(&fa, &fb);
        pk = p;
        dx = ox as f64 + sx;
        dy = oy as f64 + sy;
        if sx.abs() < 0.5 && sy.abs() < 0.5 {
            break;
        }
    }
    (dx, dy, pk)
}

/// POC at each grid point; `win = (height, width)`, `init = (ux, uy)` of `ys.len()` rows by
/// `xs.len()` columns or `None`. Each point starts at its `init` value, shifts the second window
/// by the rounded displacement and repeats up to 3 times; a window that leaves the image gives
/// peak 0 and keeps the displacement reached so far. A non-finite start value counts as 0 (Python
/// would raise). Panics if the images or `init` have the wrong shape.
pub fn poc_grid(
    a: &Image<f32>,
    b: &Image<f32>,
    xs: &[usize],
    ys: &[usize],
    win: (usize, usize),
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Grid {
    assert_eq!((a.w, a.h), (b.w, b.h), "poc_grid: images differ in size");
    let (ny, nx) = (ys.len(), xs.len());
    if let Some((iux, iuy)) = init {
        assert!(
            (iux.w, iux.h) == (nx, ny) && (iuy.w, iuy.h) == (nx, ny),
            "poc_grid: init must be {ny} rows by {nx} columns"
        );
    }
    let plan = plan_for(win.0, win.1);
    // one task per grid point; each point is computed sequentially, so the result does not depend
    // on the thread count
    let cells: Vec<(f64, f64, f64)> = (0..ny * nx)
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k / nx, k % nx);
            let start = init.map_or((0.0, 0.0), |(iux, iuy)| {
                (f64::from(iux.data[k]), f64::from(iuy.data[k]))
            });
            let start = (finite_or_zero(start.0), finite_or_zero(start.1));
            poc_point(&plan, a, b, (ys[i], xs[j]), win, start)
        })
        .collect();
    let mut g = Grid {
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        ux: Image::new(nx, ny),
        uy: Image::new(nx, ny),
        peak: Image::new(nx, ny),
    };
    for (k, (dx, dy, pk)) in cells.into_iter().enumerate() {
        g.ux.data[k] = dx as f32;
        g.uy.data[k] = dy as f32;
        g.peak.data[k] = pk as f32;
    }
    g
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Flat indices of the 8-neighbours of `(i, j)` that are inside the `ny` x `nx` grid and set in
/// `ok`, in Python's order.
fn neighbours(i: usize, j: usize, ny: usize, nx: usize, ok: &[bool]) -> Vec<usize> {
    const OFFS: [(i64, i64); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    OFFS.iter()
        .filter_map(|&(di, dj)| {
            let (ni, nj) = (i as i64 + di, j as i64 + dj);
            if ni < 0 || nj < 0 || ni >= ny as i64 || nj >= nx as i64 {
                return None;
            }
            let k = ni as usize * nx + nj as usize;
            ok[k].then_some(k)
        })
        .collect()
}

/// Marks weak peaks (`peak < peak_min`) and normalised-median outliers invalid, then fills them
/// from valid neighbours: each pass fills every invalid point that has a valid 8-neighbour with
/// the median of those neighbours, taken from the points valid when the pass began (at most 100
/// passes). The peak of an invalid point becomes 0.
///
/// If no point stays valid there is nothing to fill from: the result is `fallback = (ux, uy)`, the
/// grid's initial guess, with all peaks 0, or an all-zero field without one. A non-finite fallback
/// value counts as 0, like a non-finite start value of `poc_grid`. Panics if `fallback` does not
/// have the grid's shape.
pub fn clean(
    g: &Grid,
    peak_min: f64,
    thresh: f64,
    fallback: Option<(&Image<f32>, &Image<f32>)>,
) -> Grid {
    let (ny, nx) = (g.ux.h, g.ux.w);
    if let Some((fx, fy)) = fallback {
        assert!(
            (fx.w, fx.h) == (nx, ny) && (fy.w, fy.h) == (nx, ny),
            "clean: fallback must be {ny} rows by {nx} columns"
        );
    }
    let mut ux: Vec<f64> = g.ux.data.iter().map(|&v| f64::from(v)).collect();
    let mut uy: Vec<f64> = g.uy.data.iter().map(|&v| f64::from(v)).collect();
    let valid: Vec<bool> = g
        .peak
        .data
        .iter()
        .map(|&p| f64::from(p) >= peak_min)
        .collect();

    let mut out = valid.clone();
    for i in 0..ny {
        for j in 0..nx {
            let k = i * nx + j;
            if !valid[k] {
                continue;
            }
            let nb = neighbours(i, j, ny, nx, &valid);
            if nb.len() < 3 {
                continue;
            }
            for comp in [&ux, &uy] {
                let vals: Vec<f64> = nb.iter().map(|&p| comp[p]).collect();
                let med = median(&vals);
                let dev: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
                let res = median(&dev);
                if (comp[k] - med).abs() / (res + 0.1) > thresh {
                    out[k] = false;
                }
            }
        }
    }

    if !out.iter().any(|&o| o) {
        let field = |f: Option<&Image<f32>>| match f {
            Some(f) => Image::from_vec(
                nx,
                ny,
                f.data
                    .iter()
                    .map(|&v| finite_or_zero(f64::from(v)) as f32)
                    .collect(),
            ),
            None => Image::new(nx, ny),
        };
        let mut peak = g.peak.clone();
        for p in &mut peak.data {
            *p *= 0.0;
        }
        return Grid {
            xs: g.xs.clone(),
            ys: g.ys.clone(),
            ux: field(fallback.map(|f| f.0)),
            uy: field(fallback.map(|f| f.1)),
            peak,
        };
    }
    for _ in 0..CLEAN_MAX_PASS {
        if out.iter().all(|&o| o) {
            break;
        }
        let mut new = out.clone();
        for i in 0..ny {
            for j in 0..nx {
                let k = i * nx + j;
                if out[k] {
                    continue;
                }
                let nb = neighbours(i, j, ny, nx, &out);
                if !nb.is_empty() {
                    let vx: Vec<f64> = nb.iter().map(|&p| ux[p]).collect();
                    let vy: Vec<f64> = nb.iter().map(|&p| uy[p]).collect();
                    ux[k] = median(&vx);
                    uy[k] = median(&vy);
                    new[k] = true;
                }
            }
        }
        out = new;
    }

    let to_image = |v: &[f64]| Image::from_vec(nx, ny, v.iter().map(|&x| x as f32).collect());
    let mut peak = g.peak.clone();
    for (p, &ok) in peak.data.iter_mut().zip(&valid) {
        if !ok {
            *p = 0.0;
        }
    }
    Grid {
        xs: g.xs.clone(),
        ys: g.ys.clone(),
        ux: to_image(&ux),
        uy: to_image(&uy),
        peak,
    }
}

/// Position of `q` among the grid `nodes` (equally spaced): bilinear weight and the two node
/// indices, clamped to the grid.
fn frac(q: f64, nodes: &[usize]) -> (f64, usize, usize) {
    let n = nodes.len();
    if n == 1 {
        return (0.0, 0, 0);
    }
    let step = (nodes[1] as f64) - (nodes[0] as f64);
    let f = ((q - nodes[0] as f64) / step).clamp(0.0, (n - 1) as f64);
    let i0 = (f.floor() as usize).min(n - 2);
    (f - i0 as f64, i0, (i0 + 1).min(n - 1))
}

/// Bilinear interpolation of the grid field at pixel coordinates `(x, y)` (clamped to the grid).
/// Panics if the grid is empty.
pub fn grid_sample(g: &Grid, x: f64, y: f64) -> (f64, f64) {
    assert!(
        !g.xs.is_empty() && !g.ys.is_empty(),
        "grid_sample: empty grid"
    );
    let (tx, x0, x1) = frac(x, &g.xs);
    let (ty, y0, y1) = frac(y, &g.ys);
    let bil = |a: &Image<f32>| {
        let at = |yy: usize, xx: usize| f64::from(a.data[yy * a.w + xx]);
        (1.0 - ty) * ((1.0 - tx) * at(y0, x0) + tx * at(y0, x1))
            + ty * ((1.0 - tx) * at(y1, x0) + tx * at(y1, x1))
    };
    (bil(&g.ux), bil(&g.uy))
}

/// Gaussian blur (sigma 1) and every second pixel: `ceil(h / 2)` rows, `ceil(w / 2)` columns.
pub fn downsample(img: &Image<f32>) -> Image<f32> {
    let wide = Image::from_vec(
        img.w,
        img.h,
        img.data.iter().map(|&v| f64::from(v)).collect(),
    );
    let blur = gauss_blur(&wide, 1.0);
    let (w, h) = (img.w.div_ceil(2), img.h.div_ceil(2));
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.data[y * w + x] = blur.data[2 * y * img.w + 2 * x] as f32;
        }
    }
    out
}

/// Largest power of two not above `n`; 0 for 0.
pub fn pow2_floor(n: usize) -> usize {
    if n >= 1 {
        1 << (usize::BITS - 1 - n.leading_zeros())
    } else {
        0
    }
}

/// `floor(2 n / 3)` without overflow.
fn two_thirds(n: usize) -> usize {
    n / 3 * 2 + n % 3 * 2 / 3
}

/// POC window `(height, width)` for an `h` x `w` image. Along each axis the window may use at most
/// 2/3 of the image (power of two), leaving room to shift. Square `win` x `win` when it fits;
/// otherwise the short side gets what fits and the long side is widened to keep about `win^2`
/// pixels. `None` if a side would be shorter than 16 px.
pub fn level_window(h: usize, w: usize, win: usize) -> Option<(usize, usize)> {
    let (lim_h, lim_w) = (pow2_floor(two_thirds(h)), pow2_floor(two_thirds(w)));
    if win <= lim_h && win <= lim_w {
        return Some((win, win));
    }
    let area = win.saturating_mul(win);
    let (wy, wx) = if h <= w {
        let wy = win.min(lim_h);
        (wy, lim_w.min(win.max(area / wy.max(1))))
    } else {
        let wx = win.min(lim_w);
        (lim_h.min(win.max(area / wx.max(1))), wx)
    };
    (wy.min(wx) >= 16).then_some((wy, wx))
}

/// `ceil(log2(max(max_disp, 1e-9) / (win / 4)))`, at least 0 and at most `MAX_LEVELS`, computed
/// with floating-point `log2` exactly as Python does. NaN gives 0 (Python: `max(0, nan)`).
fn start_level(max_disp: f64, win: usize) -> u32 {
    if max_disp.is_nan() {
        return 0;
    }
    let l = (max_disp.max(1e-9) / (win as f64 / 4.0)).log2().ceil();
    if l > 0.0 {
        l.min(f64::from(MAX_LEVELS)) as u32
    } else {
        0
    }
}

/// Test oracle for `start_level`: the smallest `L` with `max(max_disp, 1e-9) <= (win / 4) 2^L`
/// (at most `MAX_LEVELS`), the exact value of the ceiling of the logarithm, from doublings only.
#[cfg(test)]
fn start_level_exact(max_disp: f64, win: usize) -> u32 {
    if max_disp.is_nan() {
        return 0;
    }
    let x = max_disp.max(1e-9);
    let mut t = win as f64 / 4.0;
    for l in 0..=MAX_LEVELS {
        if x <= t {
            return l;
        }
        t *= 2.0;
    }
    MAX_LEVELS
}

/// Pyramid levels so that `max_disp / 2^L <= win / 4`, while some window still fits: the start
/// level shrinks until `level_window` accepts the `h >> L` x `w >> L` image.
pub fn n_levels(max_disp: f64, win: usize, h: usize, w: usize) -> u32 {
    let mut l = start_level(max_disp, win);
    while l > 0 && level_window(h >> l, w >> l, win).is_none() {
        l -= 1;
    }
    l
}

/// The displacement the coarse pyramid is designed to capture along each axis, `(cy, cx)` [px]:
/// `window / 4 * 2^L` with `L = n_levels(max_disp, win, h, w)` and the window `coarse_flow` uses
/// at level `L`, which is `win` x `win` at `L = 0` and otherwise `level_window` of the level-`L`
/// image `downsample` makes (`ceil(n / 2)` per level, not `n >> L`). `level_window` shrinks that
/// window on a small or narrow image, so the range can be less than `max_disp`, which `n_levels`
/// sizes from the nominal `win`; the coarse stage then guarantees only this much. Both factors are
/// exact in binary floating point.
pub fn capture_range(max_disp: f64, win: usize, h: usize, w: usize) -> (f64, f64) {
    let levels = n_levels(max_disp, win, h, w);
    let (wy, wx) = if levels == 0 {
        (win, win)
    } else {
        let (mut lh, mut lw) = (h, w);
        for _ in 0..levels {
            lh = lh.div_ceil(2);
            lw = lw.div_ceil(2);
        }
        level_window(lh, lw, win).expect("n_levels leaves a window at every pyramid level")
    };
    let scale = (1u64 << levels) as f64;
    (wy as f64 / 4.0 * scale, wx as f64 / 4.0 * scale)
}

/// Grid displacement at full resolution. `init = (ux, uy)` is a dense guess of the image's size
/// (e.g. the previous frame); the full pyramid then searches for the correction relative to it.
/// Panics if the images or `init` differ in size.
pub fn coarse_flow(
    a: &Image<f32>,
    b: &Image<f32>,
    cfg: &Config,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Grid {
    assert_eq!((a.w, a.h), (b.w, b.h), "coarse_flow: images differ in size");
    if let Some((iux, iuy)) = init {
        assert!(
            (iux.w, iux.h) == (a.w, a.h) && (iuy.w, iuy.h) == (a.w, a.h),
            "coarse_flow: init must have the size of the images"
        );
    }
    let win0 = (cfg.poc_win, cfg.poc_win);
    let (xs, ys) = grid_points(a.h, a.w, cfg.grid_step, win0);
    let levels = n_levels(cfg.max_disp, cfg.poc_win, a.h, a.w) as usize;
    // pyr[k] is level k + 1; level 0 is the input pair
    let mut pyr: Vec<(Image<f32>, Image<f32>)> = Vec::with_capacity(levels);
    for _ in 0..levels {
        let next = match pyr.last() {
            Some((pa, pb)) => (downsample(pa), downsample(pb)),
            None => (downsample(a), downsample(b)),
        };
        pyr.push(next);
    }
    let mut g: Option<Grid> = None;
    for lev in (0..=levels).rev() {
        let (la, lb) = if lev == 0 {
            (a, b)
        } else {
            (&pyr[lev - 1].0, &pyr[lev - 1].1)
        };
        let (wl, gx, gy) = if lev == 0 {
            (win0, xs.clone(), ys.clone())
        } else {
            let wl = level_window(la.h, la.w, cfg.poc_win)
                .expect("n_levels leaves a window at every pyramid level");
            let (gx, gy) = grid_points(la.h, la.w, cfg.grid_step, wl);
            (wl, gx, gy)
        };
        let (nx, ny) = (gx.len(), gy.len());
        let mut init_u = None;
        if let Some(prev) = &g {
            // prev is on level lev + 1: halve the coordinates, double the displacement
            let (mut ux, mut uy) = (Image::new(nx, ny), Image::new(nx, ny));
            for (i, &y) in gy.iter().enumerate() {
                for (j, &x) in gx.iter().enumerate() {
                    let (px, py) = grid_sample(prev, x as f64 / 2.0, y as f64 / 2.0);
                    ux.data[i * nx + j] = (2.0 * px) as f32;
                    uy.data[i * nx + j] = (2.0 * py) as f32;
                }
            }
            init_u = Some((ux, uy));
        } else if let Some((dux, duy)) = init {
            // the coarsest level starts from the guess, scaled to this level
            let sc = 1usize << lev;
            let (mut ux, mut uy) = (Image::new(nx, ny), Image::new(nx, ny));
            for (i, &y) in gy.iter().enumerate() {
                for (j, &x) in gx.iter().enumerate() {
                    ux.data[i * nx + j] = (f64::from(dux.get(x * sc, y * sc)) / sc as f64) as f32;
                    uy.data[i * nx + j] = (f64::from(duy.get(x * sc, y * sc)) / sc as f64) as f32;
                }
            }
            init_u = Some((ux, uy));
        }
        // the level's guess starts the search and is what `clean` falls back to when no point
        // of this level matches
        let guess = init_u.as_ref().map(|(ux, uy)| (ux, uy));
        let raw = poc_grid(la, lb, &gx, &gy, wl, guess);
        g = Some(clean(&raw, cfg.poc_peak_min, cfg.median_thresh, guess));
    }
    g.expect("the loop runs at least for level 0")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic speckle: Gaussian blobs (as in python/optflow_ref/synth.py, without
    /// supersampling) moved by `(dx, dy)`, so the image is `texture(.., 0, 0)` shifted by `d`.
    fn texture(w: usize, h: usize, dx: f64, dy: f64) -> Image<f32> {
        let mut s = 12345u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let margin = 80.0;
        let n = (0.6 * (h as f64 + 2.0 * margin) * (w as f64 + 2.0 * margin) / (PI * 2.75 * 2.75))
            as usize;
        let mut acc = vec![0.0f64; w * h];
        for _ in 0..n {
            let cx = -margin + (w as f64 + 2.0 * margin) * rnd() + dx;
            let cy = -margin + (h as f64 + 2.0 * margin) * rnd() + dy;
            let r = 1.5 + 2.5 * rnd();
            let amp = (0.3 + 0.7 * rnd()) * if rnd() < 0.5 { -1.0 } else { 1.0 };
            let rad = 5.2 * r + 2.0;
            let (x0, x1) = ((cx - rad).floor().max(0.0), (cx + rad).ceil().min(w as f64));
            let (y0, y1) = ((cy - rad).floor().max(0.0), (cy + rad).ceil().min(h as f64));
            for y in y0 as usize..(y1.max(y0)) as usize {
                for x in x0 as usize..(x1.max(x0)) as usize {
                    let (ex, ey) = (x as f64 - cx, y as f64 - cy);
                    acc[y * w + x] += amp * (-(ex * ex + ey * ey) / (2.0 * r * r)).exp();
                }
            }
        }
        Image::from_vec(
            w,
            h,
            acc.iter().map(|v| (0.5 + 0.25 * v.tanh()) as f32).collect(),
        )
    }

    fn grid_of(xs: Vec<usize>, ys: Vec<usize>, ux: Vec<f32>, peak: Vec<f32>) -> Grid {
        let (nx, ny) = (xs.len(), ys.len());
        Grid {
            xs,
            ys,
            ux: Image::from_vec(nx, ny, ux),
            uy: Image::new(nx, ny),
            peak: Image::from_vec(nx, ny, peak),
        }
    }

    #[test]
    fn pow2_floor_examples() {
        let cases = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 2),
            (15, 8),
            (16, 16),
            (63, 32),
            (64, 64),
            (65, 64),
        ];
        for (n, want) in cases {
            assert_eq!(pow2_floor(n), want, "pow2_floor({n})");
        }
        assert_eq!(pow2_floor(usize::MAX), 1 << (usize::BITS - 1));
    }

    #[test]
    fn grid_points_examples() {
        assert_eq!(
            grid_points(128, 128, 16, (64, 64)),
            (vec![32, 48, 64, 80, 96], vec![32, 48, 64, 80, 96])
        );
        // rectangular window: (height, width)
        let (xs, ys) = grid_points(96, 512, 16, (64, 64));
        assert_eq!(
            (xs.first(), xs.last(), xs.len()),
            (Some(&32), Some(&480), 29)
        );
        assert_eq!(ys, vec![32, 48, 64]);
        let (xs, ys) = grid_points(40, 600, 16, (16, 256));
        assert_eq!((xs[0], ys[0]), (128, 8));
        // a window as large as the image has exactly one position; a larger one has none
        assert_eq!(grid_points(64, 64, 16, (64, 64)), (vec![32], vec![32]));
        assert_eq!(grid_points(60, 64, 16, (64, 64)), (vec![32], vec![]));
        // odd windows must not overrun the image
        assert_eq!(grid_points(3, 3, 1, (3, 3)), (vec![1], vec![1]));
    }

    #[test]
    fn grid_points_windows_stay_inside_the_image_and_miss_no_position() {
        // sampled over small shapes: the property of Kani harness K6 on the real Vec output
        for (h, w) in [(0, 0), (1, 5), (7, 3), (16, 16), (23, 40), (40, 23)] {
            for step in 1..=7 {
                for (wy, wx) in [(2, 2), (3, 5), (8, 8), (16, 4), (15, 16), (40, 40)] {
                    let (xs, ys) = grid_points(h, w, step, (wy, wx));
                    for (pts, win, n) in [(&xs, wx, w), (&ys, wy, h)] {
                        for &p in pts {
                            assert!(p >= win / 2 && p - win / 2 + win <= n, "{p} {win} {n}");
                        }
                        assert!(pts.windows(2).all(|d| d[1] - d[0] == step));
                        // the next position would not fit
                        if let Some(&last) = pts.last() {
                            assert!(last + step - win / 2 + win > n);
                        }
                        // no position is skipped: the first one that fits is the first returned
                        let fits = |p: usize| p >= win / 2 && p - win / 2 + win <= n;
                        let first_fit = (0..=n).find(|&p| fits(p));
                        assert_eq!(pts.first().copied(), first_fit);
                    }
                }
            }
        }
    }

    #[test]
    fn shifted_window_examples() {
        let win = (64, 64);
        assert_eq!(shifted_window(32, 32, win, 0, 0, 128, 128), Some((0, 0)));
        assert_eq!(shifted_window(32, 32, win, -1, 0, 128, 128), None);
        assert_eq!(shifted_window(32, 32, win, 0, -1, 128, 128), None);
        assert_eq!(shifted_window(96, 96, win, 0, 0, 128, 128), Some((64, 64)));
        assert_eq!(shifted_window(96, 96, win, 1, 0, 128, 128), None);
        assert_eq!(shifted_window(96, 96, win, 0, 1, 128, 128), None);
        assert_eq!(shifted_window(50, 60, win, -10, 7, 128, 128), Some((8, 35)));
        // huge offsets and coordinates must not overflow
        assert_eq!(shifted_window(5, 5, win, i64::MAX, 0, 128, 128), None);
        assert_eq!(shifted_window(5, 5, win, i64::MIN, 0, 128, 128), None);
        assert_eq!(shifted_window(usize::MAX, 5, win, 0, 0, 128, 128), None);
    }

    #[test]
    fn rounding_goes_to_the_even_integer_like_python() {
        let cases = [
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (-0.5, 0),
            (-1.5, -2),
            (-2.5, -2),
            (2.4, 2),
            (2.6, 3),
        ];
        for (v, want) in cases {
            assert_eq!(round_half_even(v), want, "round({v})");
        }
    }

    #[test]
    fn level_window_rectangular_for_strips() {
        assert_eq!(level_window(500, 500, 64), Some((64, 64)));
        assert_eq!(level_window(40, 600, 64), Some((16, 256)));
        assert_eq!(level_window(600, 40, 64), Some((256, 16)));
        assert_eq!(level_window(10, 600, 64), None);
        assert_eq!(level_window(0, 0, 64), None);
        // no overflow for absurd sizes
        assert_eq!(level_window(usize::MAX, usize::MAX, 64), Some((64, 64)));
    }

    #[test]
    fn levels_from_max_disp() {
        assert_eq!(n_levels(16.0, 64, 1000, 1000), 0);
        assert_eq!(n_levels(64.0, 64, 1000, 1000), 2);
        // rectangular 16 x 128 window at level 2
        assert_eq!(n_levels(64.0, 64, 100, 1000), 2);
        // coarser levels leave no room for a 16 px window
        assert_eq!(n_levels(64.0, 64, 40, 1000), 0);
        // tiny, zero, negative and NaN displacements give no pyramid
        for d in [1e-12, 0.0, -5.0, f64::NAN] {
            assert_eq!(n_levels(d, 64, 1000, 1000), 0, "max_disp {d}");
        }
        // huge or infinite displacements are limited by the image
        assert_eq!(n_levels(f64::INFINITY, 64, 1000, 1000), 5);
        assert_eq!(n_levels(1e300, 64, 4, 4), 0);
    }

    #[test]
    fn capture_range_is_the_window_over_four_times_two_to_the_levels() {
        // L = 3 and a 32 px window: only +-64 px, not the 128 asked for
        assert_eq!(capture_range(128.0, 64, 512, 512), (64.0, 64.0));
        // the default reaches max_disp exactly
        assert_eq!(capture_range(32.0, 64, 512, 512), (32.0, 32.0));
        // L = 0: the nominal window
        assert_eq!(capture_range(16.0, 64, 1000, 1000), (16.0, 16.0));
        assert_eq!(capture_range(8.0, 32, 1000, 1000), (8.0, 8.0));
        // a strip, (cy, cx): the window is 32 x 128 at L = 3
        assert_eq!(capture_range(128.0, 64, 421, 3002), (64.0, 256.0));
        // L = 0 although max_disp asks for more: no coarser level has a window (a 40 px tall image)
        assert_eq!(capture_range(128.0, 64, 40, 1000), (16.0, 16.0));
        // tiny, zero, negative and NaN displacements give no pyramid
        for d in [1e-12, 0.0, -5.0, f64::NAN] {
            assert_eq!(
                capture_range(d, 64, 1000, 1000),
                (16.0, 16.0),
                "max_disp {d}"
            );
        }
    }

    #[test]
    fn capture_range_takes_the_window_of_the_downsampled_level_not_of_shape_shifted_by_l() {
        // 95 >> 1 = 47 allows a 16 px window, but downsample keeps ceil(95 / 2) = 48 rows: 32 px fit
        assert_eq!(level_window(95 >> 1, 95 >> 1, 64), Some((16, 16)));
        let small = downsample(&Image::new(95, 95));
        assert_eq!((small.h, small.w), (48, 48));
        assert_eq!(level_window(48, 48, 64), Some((32, 32)));
        assert_eq!(capture_range(32.0, 64, 95, 95), (16.0, 16.0));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // blurs images of up to 1.3 million pixels
    fn capture_range_uses_the_window_of_the_images_downsample_makes() {
        // the same level shape as the pyramid of coarse_flow, from the real downsample
        for (d, h, w) in [
            (128.0, 512, 512),
            (128.0, 421, 3002),
            (100.0, 96, 512),
            (500.0, 700, 333),
            (32.0, 95, 95),
            (128.0, 40, 1000),
        ] {
            let levels = n_levels(d, 64, h, w);
            let mut img = Image::new(w, h);
            for _ in 0..levels {
                img = downsample(&img);
            }
            let (wy, wx) = if levels == 0 {
                (64, 64)
            } else {
                level_window(img.h, img.w, 64).unwrap()
            };
            let scale = f64::from(1u32 << levels);
            assert_eq!(
                capture_range(d, 64, h, w),
                (wy as f64 / 4.0 * scale, wx as f64 / 4.0 * scale),
                "({d}, {h}, {w})"
            );
        }
    }

    #[test]
    fn start_level_exact_agrees_with_the_log2_formula() {
        for win in [16usize, 32, 64, 128, 256, 512] {
            for d in 0..=5000u32 {
                let d = f64::from(d);
                assert_eq!(
                    start_level_exact(d, win),
                    start_level(d, win),
                    "max_disp {d}, win {win}"
                );
            }
            // exact powers of two and the values just around them
            for k in 0..40 {
                let p = 2.0f64.powi(k) * (win as f64 / 4.0);
                for d in [p, p * (1.0 + 1e-6), p * (1.0 - 1e-6)] {
                    assert_eq!(start_level_exact(d, win), start_level(d, win), "{d}, {win}");
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn poc_shift_recovers_a_sub_pixel_shift() {
        let (dx, dy) = (2.3, -1.6);
        let (a, b) = (texture(64, 64, 0.0, 0.0), texture(64, 64, dx, dy));
        let (sx, sy, pk) = poc_shift(&a, &b);
        assert!(
            (sx - dx).abs() < 0.1 && (sy - dy).abs() < 0.1,
            "got ({sx}, {sy})"
        );
        assert!(pk > 0.3, "peak {pk}");
        // rectangular window, height 32 and width 64
        let (a, b) = (texture(64, 32, 0.0, 0.0), texture(64, 32, dx, dy));
        let (sx, sy, _) = poc_shift(&a, &b);
        assert!(
            (sx - dx).abs() < 0.15 && (sy - dy).abs() < 0.15,
            "got ({sx}, {sy})"
        );
    }

    #[test]
    fn poc_shift_of_identical_windows_is_zero() {
        let a = texture(32, 32, 0.0, 0.0);
        let (sx, sy, pk) = poc_shift(&a, &a);
        assert!(sx.abs() < 1e-6 && sy.abs() < 1e-6, "got ({sx}, {sy})");
        assert!(pk > 0.5, "peak {pk}");
    }

    #[test]
    fn poc_grid_follows_the_initial_guess_and_flags_leaving_windows() {
        let (w, h) = (96, 96);
        let (a, b) = (texture(w, h, 0.0, 0.0), texture(w, h, 7.0, 3.0));
        let (xs, ys) = (vec![48usize], vec![48usize]);
        // no guess: 7 px is beyond the +-0.5 px reached by one step but within three iterations
        let g = poc_grid(&a, &b, &xs, &ys, (32, 32), None);
        assert!((g.ux.data[0] - 7.0).abs() < 0.2 && (g.uy.data[0] - 3.0).abs() < 0.2);
        assert!(g.peak.data[0] > 0.3);
        // a guess near the truth also lands on it
        let ix = Image::from_vec(1, 1, vec![6.0f32]);
        let iy = Image::from_vec(1, 1, vec![4.0f32]);
        let g = poc_grid(&a, &b, &xs, &ys, (32, 32), Some((&ix, &iy)));
        assert!((g.ux.data[0] - 7.0).abs() < 0.2 && (g.uy.data[0] - 3.0).abs() < 0.2);
        // a guess that moves the window out of the image: peak 0, guess kept
        let ix = Image::from_vec(1, 1, vec![60.0f32]);
        let iy = Image::from_vec(1, 1, vec![0.0f32]);
        let g = poc_grid(&a, &b, &xs, &ys, (32, 32), Some((&ix, &iy)));
        assert_eq!(
            (g.ux.data[0], g.uy.data[0], g.peak.data[0]),
            (60.0, 0.0, 0.0)
        );
        // a non-finite guess counts as zero
        let ix = Image::from_vec(1, 1, vec![f32::NAN]);
        let g = poc_grid(&a, &b, &xs, &ys, (32, 32), Some((&ix, &iy)));
        assert!((g.ux.data[0] - 7.0).abs() < 0.2);
        // empty grids give empty fields
        let g = poc_grid(&a, &b, &[], &ys, (32, 32), None);
        assert_eq!((g.ux.w, g.ux.h), (0, 1));
    }

    #[test]
    fn clean_replaces_an_outlier_with_the_neighbour_median() {
        let mut ux = vec![1.0f32; 25];
        ux[12] = 9.0;
        let g = grid_of((0..5).collect(), (0..5).collect(), ux, vec![1.0; 25]);
        let c = clean(&g, 0.1, 2.0, None);
        assert!((c.ux.data[12] - 1.0).abs() < 1e-6);
        assert!(c.ux.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // the outlier keeps its peak: only weak peaks are zeroed
        assert_eq!(c.peak.data[12], 1.0);
    }

    #[test]
    fn clean_fills_weak_points_from_valid_neighbours_and_zeroes_their_peak() {
        let mut ux = vec![2.0f32; 9];
        ux[4] = 100.0;
        let mut peak = vec![1.0f32; 9];
        peak[4] = 0.0;
        let c = clean(
            &grid_of((0..3).collect(), (0..3).collect(), ux, peak),
            0.1,
            2.0,
            None,
        );
        assert_eq!(c.ux.data[4], 2.0);
        assert_eq!(c.peak.data[4], 0.0);
        assert_eq!(c.peak.data[0], 1.0);
        // a corner has only 3 neighbours: a lone outlier there is still tested (3 >= 3)
        let mut ux = vec![1.0f32; 9];
        ux[0] = 50.0;
        let c = clean(
            &grid_of((0..3).collect(), (0..3).collect(), ux, vec![1.0; 9]),
            0.1,
            2.0,
            None,
        );
        assert_eq!(c.ux.data[0], 1.0);
    }

    #[test]
    fn clean_with_nothing_valid_gives_zero_field() {
        let g = grid_of(vec![0, 1], vec![0, 1], vec![3.0; 4], vec![0.0; 4]);
        let c = clean(&g, 0.1, 2.0, None);
        assert!(c
            .ux
            .data
            .iter()
            .chain(&c.uy.data)
            .chain(&c.peak.data)
            .all(|&v| v == 0.0));
        let empty = clean(&grid_of(vec![], vec![], vec![], vec![]), 0.1, 2.0, None);
        assert!(empty.ux.data.is_empty());
    }

    #[test]
    fn clean_without_a_valid_point_returns_the_fallback() {
        let (xs, ys) = (
            (0..3).map(|i| i * 16).collect(),
            (0..4).map(|i| i * 16).collect(),
        );
        let mut g = grid_of(xs, ys, vec![7.0; 12], vec![0.0; 12]); // every peak weak
        g.uy = Image::from_vec(3, 4, vec![-7.0; 12]);
        let (fx, fy) = (
            Image::from_vec(3, 4, vec![2.5f32; 12]),
            Image::from_vec(3, 4, vec![-1.5f32; 12]),
        );
        let c = clean(&g, 0.1, 2.0, Some((&fx, &fy)));
        assert_eq!(c.ux, fx);
        assert_eq!(c.uy, fy);
        assert!(c.peak.data.iter().all(|&p| p == 0.0));
        // without a guess: zeros
        let c = clean(&g, 0.1, 2.0, None);
        assert!(c.ux.data.iter().chain(&c.uy.data).all(|&v| v == 0.0));
        // a non-finite guess counts as 0, like a non-finite start value of poc_grid
        let nan = Image::from_vec(3, 4, vec![f32::NAN; 12]);
        let c = clean(&g, 0.1, 2.0, Some((&nan, &fy)));
        assert!(c.ux.data.iter().all(|&v| v == 0.0));
        assert_eq!(c.uy, fy);
        // the guess is not used when some point is valid: the valid neighbours fill the rest
        g.peak.data[0] = 1.0;
        let c = clean(&g, 0.1, 2.0, Some((&fx, &fy)));
        assert!(c.ux.data.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn clean_fills_a_block_pass_by_pass() {
        // A 5x5 field ux = 0.125 * column with a weak 3x3 block (ux = 77, peak 0) in the middle.
        // Pass 1 fills the 8 cells of the block's ring from the valid cells around them, as they
        // were when the pass began (in units of 0.125, neighbour values -> median):
        //   (1,1): 0 1 2 0 0 -> 0      (1,2): 1 2 3 -> 2      (1,3): 2 3 4 4 4 -> 4
        //   (2,1): 0 0 0 -> 0                                 (2,3): 4 4 4 -> 4
        //   (3,1): 0 0 0 1 2 -> 0      (3,2): 1 2 3 -> 2      (3,3): 2 3 4 4 4 -> 4
        // The centre has no valid neighbour before pass 2, which fills it from the ring values of
        // pass 1: 0 2 4 0 4 0 2 4 -> (2 + 2) / 2 = 2. A fill made in place (a cell filled in this
        // pass already counting as a neighbour) would give (1,2) = median(0 1 2 3) = 1.5 instead.
        let s = 0.125f32;
        let mut peak = vec![1.0f32; 25];
        let mut ux: Vec<f32> = (0..25).map(|k| s * (k % 5) as f32).collect();
        for i in 1..4 {
            for j in 1..4 {
                peak[i * 5 + j] = 0.0;
                ux[i * 5 + j] = 77.0;
            }
        }
        let c = clean(
            &grid_of((0..5).collect(), (0..5).collect(), ux, peak),
            0.1,
            2.0,
            None,
        );
        let col = [0.0f32, 2.0, 4.0]; // the filled block, the same in each of its three rows
        for i in 0..5 {
            for j in 0..5 {
                let want = if (1..4).contains(&i) && (1..4).contains(&j) {
                    s * col[j - 1]
                } else {
                    s * j as f32
                };
                assert_eq!(c.ux.data[i * 5 + j], want, "({i}, {j})");
            }
        }
        // the filled cells are flagged invalid, the ring keeps its peaks
        assert_eq!(c.peak.data[12], 0.0);
        assert_eq!(c.peak.data[0], 1.0);
    }

    #[test]
    fn grid_sample_is_bilinear_and_clamped() {
        let g = Grid {
            xs: vec![0, 10],
            ys: vec![0, 10],
            ux: Image::from_vec(2, 2, vec![0.0, 1.0, 2.0, 3.0]),
            uy: Image::new(2, 2),
            peak: Image::from_vec(2, 2, vec![1.0; 4]),
        };
        assert_eq!(grid_sample(&g, 5.0, 5.0).0, 1.5);
        // x = -3 clamps to 0, y = 20 clamps to 10: node (row 1, column 0)
        assert_eq!(grid_sample(&g, -3.0, 20.0).0, 2.0);
        assert_eq!(grid_sample(&g, 10.0, 0.0).0, 1.0);
        assert!((grid_sample(&g, 2.5, 7.5).0 - (0.25 * 1.0 + 1.5)).abs() < 1e-12);
        assert!(grid_sample(&g, f64::NAN, 0.0).0.is_nan());
        // single-node axes
        let one = grid_of(vec![5], vec![5], vec![4.0], vec![1.0]);
        assert_eq!(grid_sample(&one, -100.0, 100.0).0, 4.0);
    }

    #[test]
    fn downsample_halves_with_ceil_and_keeps_constants() {
        let img = Image::from_vec(5, 3, vec![2.5f32; 15]);
        let d = downsample(&img);
        assert_eq!((d.w, d.h), (3, 2));
        assert!(d.data.iter().all(|&v| (v - 2.5).abs() < 1e-6));
        let one = downsample(&Image::from_vec(1, 1, vec![7.0f32]));
        assert_eq!((one.w, one.h, one.data[0]), (1, 1, 7.0));
        // the blur comes before the decimation: an isolated bright pixel is spread out
        let mut spike = Image::new(8, 8);
        spike.set(2, 2, 1.0);
        let d = downsample(&spike);
        assert!(d.get(1, 1) > 0.1 && d.get(1, 1) < 0.2);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn coarse_flow_follows_a_shift_beyond_one_window() {
        let (w, h) = (256, 256);
        let (a, b) = (texture(w, h, 0.0, 0.0), texture(w, h, 21.4, -13.2));
        let g = coarse_flow(&a, &b, &Config::default(), None);
        let (nx, ny) = (g.xs.len(), g.ys.len());
        for i in 1..ny - 2 {
            for j in 1..nx - 2 {
                let (ux, uy) = (g.ux.data[i * nx + j], g.uy.data[i * nx + j]);
                assert!(
                    (ux - 21.4).abs() < 0.25 && (uy + 13.2).abs() < 0.25,
                    "({i}, {j}): {ux}, {uy}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn coarse_flow_is_deterministic() {
        let (a, b) = (texture(160, 128, 0.0, 0.0), texture(160, 128, 5.3, -2.7));
        let g1 = coarse_flow(&a, &b, &Config::default(), None);
        let g2 = coarse_flow(&a, &b, &Config::default(), None);
        assert_eq!(g1.ux, g2.ux);
        assert_eq!(g1.uy, g2.uy);
        assert_eq!(g1.peak, g2.peak);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn coarse_flow_keeps_the_guess_where_nothing_matches() {
        let n = 192;
        let a = texture(n, n, 0.0, 0.0);
        // a blank second image: no correlation peak anywhere, on any level
        let b = Image::from_vec(n, n, vec![0.5f32; n * n]);
        let (gx, gy) = (
            Image::from_vec(n, n, vec![3.0f32; n * n]),
            Image::from_vec(n, n, vec![-2.0f32; n * n]),
        );
        // two pyramid levels (192 -> 96 -> 48 px): the guess is scaled down to the coarsest level
        // and every finer level starts from the one above, so the resampling is exercised
        let cfg = Config {
            max_disp: 64.0,
            ..Config::default()
        };
        assert_eq!(n_levels(cfg.max_disp, cfg.poc_win, n, n), 2);
        let g = coarse_flow(&a, &b, &cfg, Some((&gx, &gy)));
        assert!(!g.ux.data.is_empty());
        assert!(g.peak.data.iter().all(|&p| p == 0.0));
        assert!(g.ux.data.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert!(g.uy.data.iter().all(|&v| (v + 2.0).abs() < 1e-6));
        // without a guess the same pair gives zeros
        let g = coarse_flow(&a, &b, &cfg, None);
        assert!(g.ux.data.iter().chain(&g.uy.data).all(|&v| v == 0.0));
    }

    /// The guess is sampled at the grid point itself (`x * sc`, `y * sc` at a level `sc` times
    /// coarser) and the grid it makes is carried down the pyramid: with a blank second image, so
    /// that no level finds a peak, a guess that is linear in `x` and in `y` comes back as itself.
    /// A constant guess cannot show a swapped or inverted sample position.
    #[test]
    #[cfg_attr(miri, ignore)] // a full pipeline: over 30 min under Miri
    fn coarse_flow_carries_a_sloped_guess_through_the_pyramid() {
        let n = 192;
        let a = texture(n, n, 0.0, 0.0);
        let b = Image::from_vec(n, n, vec![0.5f32; n * n]);
        let field = |f: &dyn Fn(f64, f64) -> f64| {
            let data = (0..n * n)
                .map(|i| f((i % n) as f64, (i / n) as f64) as f32)
                .collect();
            Image::from_vec(n, n, data)
        };
        let ux = |x: f64, y: f64| 1.0 + 0.01 * x + 0.03 * y;
        let uy = |x: f64, y: f64| -2.0 + 0.02 * x - 0.01 * y;
        let (gx, gy) = (field(&ux), field(&uy));
        let cfg = Config {
            max_disp: 64.0,
            ..Config::default()
        };
        assert_eq!(n_levels(cfg.max_disp, cfg.poc_win, n, n), 2);
        let g = coarse_flow(&a, &b, &cfg, Some((&gx, &gy)));
        assert!(g.peak.data.iter().all(|&p| p == 0.0));
        let (nx, ny) = (g.xs.len(), g.ys.len());
        assert!(nx >= 7 && ny >= 7);
        // the grid points whose neighbourhood lies within the coarse grids: the two outermost
        // rings of points read the edge values of the coarser grid (clamped, not extrapolated)
        let mut worst: f64 = 0.0;
        for (i, &y) in g.ys.iter().enumerate().take(ny - 2).skip(2) {
            for (j, &x) in g.xs.iter().enumerate().take(nx - 2).skip(2) {
                let (x, y) = (x as f64, y as f64);
                let got = (
                    f64::from(g.ux.data[i * nx + j]),
                    f64::from(g.uy.data[i * nx + j]),
                );
                worst = worst
                    .max((got.0 - ux(x, y)).abs())
                    .max((got.1 - uy(x, y)).abs());
            }
        }
        assert!(worst < 1e-3, "worst deviation from the guess {worst}");
    }

    /// `poc_shift` takes windows of any shape of at least 3 px a side; the production windows are
    /// powers of two, for which `n - n / 2` and `n / 2` agree, but the shift of the spectrum
    /// (`fftshift`) must also centre an odd axis on `n / 2`, as numpy does.
    #[test]
    #[cfg_attr(miri, ignore)] // textures of thousands of blobs: slow under Miri
    fn poc_shift_of_windows_with_an_odd_side_is_centred_on_half_the_side() {
        for (w, h) in [(37usize, 31usize), (36, 31), (37, 32), (33, 33)] {
            let a = texture(w, h, 0.0, 0.0);
            for (dx, dy) in [(3.0, -2.0), (-4.0, 3.0), (1.0, 0.0), (0.0, -1.0)] {
                let b = texture(w, h, dx, dy);
                let (sx, sy, peak) = poc_shift(&a, &b);
                assert!(
                    (sx - dx).abs() < 0.5 && (sy - dy).abs() < 0.5,
                    "{w}x{h} shift ({dx}, {dy}) found as ({sx}, {sy})"
                );
                assert!(peak > 0.3, "{w}x{h} shift ({dx}, {dy}): peak {peak}");
            }
        }
    }

    #[test]
    fn coarse_flow_on_an_image_smaller_than_the_window_is_empty() {
        let img = Image::new(40, 40);
        let g = coarse_flow(&img, &img, &Config::default(), None);
        assert!(g.xs.is_empty() && g.ys.is_empty() && g.ux.data.is_empty());
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn k4_level_window_rule() {
        let h: usize = kani::any();
        let w: usize = kani::any();
        let e: u32 = kani::any();
        kani::assume(h >= 1 && h <= 1 << 16 && w >= 1 && w <= 1 << 16 && e >= 4 && e <= 9);
        let win = 1usize << e;
        let (lim_h, lim_w) = (pow2_floor(2 * h / 3), pow2_floor(2 * w / 3));
        match level_window(h, w, win) {
            // a window exists exactly when both 2/3 limits allow at least 16 px (win >= 16)
            None => assert!(lim_h < 16 || lim_w < 16),
            Some((wy, wx)) => {
                assert!(wy >= 16 && wx >= 16);
                assert!(wy <= pow2_floor(2 * h / 3) && wx <= pow2_floor(2 * w / 3));
                assert!(wy.count_ones() == 1 && wx.count_ones() == 1);
                if win <= pow2_floor(2 * h / 3) && win <= pow2_floor(2 * w / 3) {
                    assert!(wy == win && wx == win);
                }
            }
        }
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn k5_n_levels_leaves_a_usable_window() {
        let h: usize = kani::any();
        let w: usize = kani::any();
        let d: u16 = kani::any();
        kani::assume(h >= 1 && h <= 1 << 14 && w >= 1 && w <= 1 << 14 && d >= 1 && d <= 1000);
        let l = n_levels(d as f64, 64, h, w);
        assert!(l <= 8);
        if l > 0 {
            assert!(level_window(h >> l, w >> l, 64).is_some());
        }
    }

    /// coarse_flow calls level_window at every level 1..=L on the ceil-sized shapes of repeated
    /// [::2] downsampling and does not handle None: every such level must have a window.
    #[kani::proof]
    #[kani::unwind(10)]
    fn k5b_every_pyramid_level_has_a_window() {
        let h: usize = kani::any();
        let w: usize = kani::any();
        let d: u16 = kani::any();
        kani::assume(h >= 1 && h <= 1 << 14 && w >= 1 && w <= 1 << 14 && d >= 1 && d <= 1000);
        let l = n_levels(d as f64, 64, h, w);
        let (mut hh, mut ww) = (h, w);
        for _ in 0..l {
            hh = hh.div_ceil(2);
            ww = ww.div_ceil(2);
            assert!(level_window(hh, ww, 64).is_some());
        }
    }

    /// `capture_range` finds its level-`L` window (the `expect` cannot fire) and, each window being
    /// at least 16 px, covers at least 4 px per axis; `L = 0` gives `win / 4` exactly.
    #[kani::proof]
    #[kani::unwind(10)]
    fn k5c_capture_range_has_a_window_and_a_positive_range() {
        let h: usize = kani::any();
        let w: usize = kani::any();
        let d: u16 = kani::any();
        kani::assume(h >= 1 && h <= 1 << 14 && w >= 1 && w <= 1 << 14 && d >= 1 && d <= 1000);
        let (cy, cx) = capture_range(d as f64, 64, h, w);
        assert!(cy >= 4.0 && cx >= 4.0);
        if n_levels(d as f64, 64, h, w) == 0 {
            assert!(cy == 16.0 && cx == 16.0);
        }
    }

    /// The grid positions are `start + i * step` for `i < count` (`grid_axis`, which
    /// `grid_points` expands); proved for a symbolic `i` instead of a loop over a `Vec`.
    #[kani::proof]
    fn k6_grid_windows_inside_image() {
        let h: usize = kani::any();
        let w: usize = kani::any();
        let step: usize = kani::any();
        let wy: usize = kani::any();
        let wx: usize = kani::any();
        kani::assume(
            h <= 64
                && w <= 64
                && step >= 1
                && step <= 16
                && wy >= 2
                && wy <= 64
                && wx >= 2
                && wx <= 64,
        );
        let (x0, nx) = grid_axis(w, wx, step);
        let (y0, ny) = grid_axis(h, wy, step);
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < nx && j < ny);
        let (x, y) = (x0 + i * step, y0 + j * step);
        assert!(x >= wx / 2 && x - wx / 2 + wx <= w);
        assert!(y >= wy / 2 && y - wy / 2 + wy <= h);
    }

    #[kani::proof]
    fn k7_shifted_window_in_bounds() {
        let (y, x, h, w): (usize, usize, usize, usize) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        let (wy, wx): (usize, usize) = (kani::any(), kani::any());
        let (oy, ox): (i64, i64) = (kani::any(), kani::any());
        kani::assume(
            h <= 1 << 16 && w <= 1 << 16 && y < h && x < w && wy <= 1 << 12 && wx <= 1 << 12,
        );
        kani::assume(oy.unsigned_abs() < 1 << 20 && ox.unsigned_abs() < 1 << 20);
        if let Some((y0, x0)) = shifted_window(y, x, (wy, wx), oy, ox, h, w) {
            assert!(y0 + wy <= h && x0 + wx <= w);
        }
    }
}

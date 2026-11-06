//! Dense sub-pixel refinement from the local phase of a multi-band complex Gabor filter bank.
//!
//! Port of `window_moment`, `shape_unknowns`, `local_solve` and `refine` of
//! `python/optflow_ref/phaseflow.py`; that file is the specification and describes the method.
//! The filter bank and the reference phase come from [`crate::gabor`].
//!
//! Storage is `f32` (images, filter responses, B-spline coefficients); the window moments, the
//! local least-squares systems, the residual and the displacement field carried through the
//! iterations are `f64`. Every output value is a sequential sum in a fixed order; rayon only
//! splits the work across independent rows, filters or moment maps, so the result does not
//! depend on the number of threads.
//!
//! Border rule (spec 2.4): the Gabor responses within about `edge_guard_px` of an image border
//! come from reflect padding and the Gaussian window of the local fit averages them over
//! `4 sigma_w`; the padded texture is self-consistent, so neither the residual nor the texture
//! factor can tell. With `B = 4 sigma_w` the confidence is therefore 0 where the pixel lies
//! within `B` of the first image's border and where its match `X'` lies within `B` of the second
//! image's border or outside it (a non-finite `X'` counts as outside). The rule applies only
//! while the edge guard is on (`edge_guard_px() > 0`); with the guard off `B = 0` and only the
//! matches that leave the second image get confidence 0. `track` gives these strips the usual
//! guard band, so a result starts about `B + guard` (28 px at the defaults) from a border.
//!
//! Trusted pre-warp (spec 2.3). From the second outer iteration on, the pre-warp leaves out the
//! displacement of the pixels that cannot be trusted and extrapolates it from the trusted ones,
//! so that a region where the coarse stage failed does not bias its good neighbours; see
//! the `prewarp` module for the rule, its reach (about 40 px, an assumption and not a bound) and
//! its tile behaviour.
//!
//! The per-pixel `n x n` systems (`n = 6` for the affine, `12` for the quadratic shape function)
//! are assembled one pixel at a time from the window-moment maps and solved by Cholesky
//! decomposition (numpy uses an LU solve; the results agree to rounding) after a tiny relative
//! ridge on the diagonal. A pixel whose energy vanishes, whose gradients span only one direction
//! (aperture problem) or whose system is not positive definite is degenerate: it keeps its input
//! displacement and gets the residual `pi`.

use std::f64::consts::PI;

use num_complex::Complex32;
use rayon::prelude::*;

use crate::atan2::atan2_all;
use crate::config::Config;
use crate::engine::{CpuEngine, EngineError, FineEngine};
use crate::fft::Fft2;
use crate::gabor::{
    gabor_bank, padded_spectrum, reference_inverses, reference_spectrum, reference_values, Bank,
    PAD,
};
use crate::image::Image;
use crate::interp::{
    correlate_sep, gauss_blur, gauss_kernel, prefilter_pair_lines, prefilter_rows, taps, taps_sum,
    BSpline, Boundary, Taps, MAX_KERNELS,
};
use crate::madd::madd;
use crate::prewarp::{low_confidence, prewarp};
use crate::prof::{self, Stage};
use crate::select::median;

/// Result of [`refine`], one value per pixel of the first image.
#[derive(Clone, Debug)]
pub struct Flow {
    /// Displacement of the matched point in x [px]: pixel `(x, y)` of the first image appears at
    /// `(x + ux, y + uy)` in the second.
    pub ux: Image<f32>,
    /// Displacement in y [px].
    pub uy: Image<f32>,
    /// Confidence in `[0, 1]`; 0 within `B = 4 sigma_w` of the first image's border and where the
    /// matched point is within `B` of the second image's border or outside it (module doc).
    pub conf: Image<f32>,
    /// Weighted RMS phase residual of the local model [rad]; `pi` where the system is degenerate.
    pub resid: Image<f32>,
}

/// `sum_q W(q - p) (qx - px)^px (qy - py)^py a(q)` with a Gaussian window `W` of width `sigma`
/// (radius `round(4 sigma)`), edges replicated. Along rows (y) the kernel is `k * t^py`, then
/// along columns (x) `k * t^px`, as in `window_moment` of the reference.
pub fn window_moment(a: &Image<f64>, sigma: f64, px: u32, py: u32) -> Image<f64> {
    let k = gauss_kernel(sigma);
    correlate_sep(
        a,
        &moment_kernel(&k, py),
        &moment_kernel(&k, px),
        Boundary::Nearest,
    )
}

/// The taps `k * t^p` for offsets `t = -r ..= r` (`r = k.len() / 2`).
fn moment_kernel(k: &[f64], p: u32) -> Vec<f64> {
    let r = (k.len() / 2) as i64;
    k.iter()
        .enumerate()
        .map(|(i, &kv)| kv * ((i as i64 - r) as f64).powi(p as i32))
        .collect()
}

/// [`moment_kernel`] of the Gaussian of `sigma` for `p = 0 ..= max_power` (for `gpu_support`).
pub(crate) fn moment_taps(sigma: f64, max_power: u32) -> Vec<Vec<f64>> {
    let k = gauss_kernel(sigma);
    (0..=max_power).map(|p| moment_kernel(&k, p)).collect()
}

/// Powers `(x, y)` of the offset `d = q - p` in the polynomial shape function of each order.
const POWERS_1: [(u32, u32); 3] = [(0, 0), (1, 0), (0, 1)];
const POWERS_2: [(u32, u32); 6] = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)];

/// Unknowns of the shape function as `(displacement component 0 = x / 1 = y, powers of the
/// offset d)`; `ux` and `uy` (the constant terms) come first. Panics unless `order` is 1 or 2.
pub fn shape_unknowns(order: u32) -> Vec<(usize, (u32, u32))> {
    let pw: &[(u32, u32)] = match order {
        1 => &POWERS_1,
        2 => &POWERS_2,
        _ => panic!("shape_unknowns: order must be 1 or 2, got {order}"),
    };
    let mut out = vec![(0, pw[0]), (1, pw[0])];
    for &p in &pw[1..] {
        out.extend([(0, p), (1, p)]);
    }
    out
}

/// The per-pixel sums of the inner iteration that the window moments are taken of.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Src {
    Gxx,
    Gxy,
    Gyy,
    Gxc,
    Gyc,
    Gcc,
    Gw,
}

/// Per-pixel sums `w g g^T`, `w g c`, `w c^2` and `w` over the filter bank (`gcc`, `gw`).
pub(crate) struct Sums {
    pub(crate) gxx: Image<f64>,
    pub(crate) gxy: Image<f64>,
    pub(crate) gyy: Image<f64>,
    pub(crate) gxc: Image<f64>,
    pub(crate) gyc: Image<f64>,
    pub(crate) gcc: Image<f64>,
    pub(crate) gw: Image<f64>,
}

impl Sums {
    /// Planes for an image of `w x h` (zero; `accumulate_into` writes every value).
    pub(crate) fn new(w: usize, h: usize) -> Self {
        let plane = || Image::from_vec(w, h, vec![0.0f64; w * h]);
        Self {
            gxx: plane(),
            gxy: plane(),
            gyy: plane(),
            gxc: plane(),
            gyc: plane(),
            gcc: plane(),
            gw: plane(),
        }
    }

    fn source(&self, s: Src) -> &Image<f64> {
        match s {
            Src::Gxx => &self.gxx,
            Src::Gxy => &self.gxy,
            Src::Gyy => &self.gyy,
            Src::Gxc => &self.gxc,
            Src::Gyc => &self.gyc,
            Src::Gcc => &self.gcc,
            Src::Gw => &self.gw,
        }
    }
}

/// Which window-moment maps the `n x n` system of one pixel is assembled from.
struct Plan {
    n: usize,
    /// The distinct `(source, powers)` moments, in order of first use.
    moments: Vec<(Src, (u32, u32))>,
    /// `moments` index of `M[i][j]` at `i * n + j`.
    m_idx: Vec<usize>,
    /// `moments` index of `b[i]`.
    b_idx: Vec<usize>,
    /// `moments` index of the window sums of `w c^2` and of `w`.
    gcc_idx: usize,
    gw_idx: usize,
}

fn plan(order: u32) -> Plan {
    let unk = shape_unknowns(order);
    let n = unk.len();
    let mut moments: Vec<(Src, (u32, u32))> = Vec::new();
    let mut index_of = |key: (Src, (u32, u32))| -> usize {
        moments.iter().position(|m| *m == key).unwrap_or_else(|| {
            moments.push(key);
            moments.len() - 1
        })
    };
    let mut m_idx = vec![0; n * n];
    for (i, &(ci, pi)) in unk.iter().enumerate() {
        for (j, &(cj, pj)) in unk.iter().enumerate() {
            let src = match (ci.min(cj), ci.max(cj)) {
                (0, 0) => Src::Gxx,
                (0, 1) => Src::Gxy,
                _ => Src::Gyy,
            };
            m_idx[i * n + j] = index_of((src, (pi.0 + pj.0, pi.1 + pj.1)));
        }
    }
    let b_idx = unk
        .iter()
        .map(|&(c, p)| index_of((if c == 0 { Src::Gxc } else { Src::Gyc }, p)))
        .collect();
    let gcc_idx = index_of((Src::Gcc, (0, 0)));
    let gw_idx = index_of((Src::Gw, (0, 0)));
    Plan {
        n,
        moments,
        m_idx,
        b_idx,
        gcc_idx,
        gw_idx,
    }
}

/// [`plan`] as plain data (for `gpu_support`): `(n, moments, m_idx, b_idx, gcc_idx, gw_idx)`, the
/// source of a moment numbered in the order of [`Src`].
#[allow(clippy::type_complexity)]
pub(crate) fn plan_moments(
    order: u32,
) -> (
    usize,
    Vec<(u8, (u32, u32))>,
    Vec<usize>,
    Vec<usize>,
    usize,
    usize,
) {
    let p = plan(order);
    let moments = p.moments.iter().map(|&(s, pw)| (s as u8, pw)).collect();
    (p.n, moments, p.m_idx, p.b_idx, p.gcc_idx, p.gw_idx)
}

/// The window moments of one source: per y power, the `(index in the plan, x power)` of each of
/// its moments.
struct SrcMoments {
    src: Src,
    ys: Vec<(u32, Vec<(usize, u32)>)>,
}

/// The window moments of a plan, computed one row at a time (see [`MomentRows::row`]).
struct MomentRows {
    /// Kernel radius: `2 r + 1` taps.
    r: usize,
    /// `k * t^p` (`moment_kernel`) for every power `p` a moment of the plan uses, at index `p`.
    taps: Vec<Vec<f64>>,
    sources: Vec<SrcMoments>,
    n_moments: usize,
}

impl MomentRows {
    fn new(plan: &Plan, sigma: f64) -> Self {
        let k = gauss_kernel(sigma);
        let max_p = plan.moments.iter().map(|&(_, (px, py))| px.max(py));
        let taps = (0..=max_p.max().unwrap_or(0))
            .map(|p| moment_kernel(&k, p))
            .collect();
        let mut sources: Vec<SrcMoments> = Vec::new();
        for (idx, &(src, (px, py))) in plan.moments.iter().enumerate() {
            let s = match sources.iter().position(|s| s.src == src) {
                Some(s) => s,
                None => {
                    sources.push(SrcMoments {
                        src,
                        ys: Vec::new(),
                    });
                    sources.len() - 1
                }
            };
            let ys = &mut sources[s].ys;
            match ys.iter_mut().find(|(p, _)| *p == py) {
                Some((_, items)) => items.push((idx, px)),
                None => ys.push((py, vec![(idx, px)])),
            }
        }
        Self {
            r: k.len() / 2,
            taps,
            sources,
            n_moments: plan.moments.len(),
        }
    }

    /// Length of the scratch `pad` that [`MomentRows::row`] needs for rows of `w` pixels.
    fn pad_len(&self, w: usize) -> usize {
        MAX_KERNELS * (w + 2 * self.r)
    }

    /// Row `y` of every window moment of the plan: `out[m * w + x]` is moment `m` at `(x, y)`,
    /// bit for bit the value of `window_moment` (the same taps and the same sequential sums: the
    /// pass along y, edges replicated, then the pass along x of the replicated-edge row). The
    /// moments of one source share the reads of the pass along y, those with one y power its
    /// result; each pass applies up to `MAX_KERNELS` kernels to every value it reads.
    fn row(&self, sums: &Sums, y: usize, pad: &mut [f64], out: &mut [f64]) {
        self.row_span(sums, y, (0, sums.gw.w), pad, out);
    }

    /// [`MomentRows::row`] for the pixels `x0 <= x < x1` of row `y`: `out[m * (x1 - x0) + x - x0]`
    /// is moment `m` at `(x, y)`, the same bits as `row` gives there. Only the values within the
    /// kernel of the span are read.
    fn row_span(
        &self,
        sums: &Sums,
        y: usize,
        (x0, x1): (usize, usize),
        pad: &mut [f64],
        out: &mut [f64],
    ) {
        let (w, h, r) = (sums.gw.w, sums.gw.h, self.r);
        let nt = 2 * r + 1;
        let pw = w + 2 * r;
        let len = x1 - x0;
        debug_assert_eq!(out.len(), self.n_moments * len);
        let (xa, xb) = (x0.saturating_sub(r), (x1 + r).min(w));
        for s in &self.sources {
            let a = &sums.source(s.src).data;
            let line = |t: usize| {
                let sy = (y + t).saturating_sub(r).min(h - 1);
                &a[sy * w + xa..sy * w + xb]
            };
            for ys in s.ys.chunks(MAX_KERNELS) {
                // along y into the middle of the padded rows, then replicate the edges
                let ks: Vec<&[f64]> = ys
                    .iter()
                    .map(|(py, _)| &self.taps[*py as usize][..])
                    .collect();
                taps_sum(nt, line, &ks, xb - xa, |p, x, v| {
                    let at = p * pw + r + xa + x;
                    pad[at..at + v.len()].copy_from_slice(v);
                });
                for p in 0..ys.len() {
                    let row = &mut pad[p * pw..(p + 1) * pw];
                    if xa == 0 {
                        let first = row[r];
                        row[..r].fill(first);
                    }
                    if xb == w {
                        let last = row[r + w - 1];
                        row[r + w..].fill(last);
                    }
                }
                // along x
                for (p, (_, items)) in ys.iter().enumerate() {
                    let row = &pad[p * pw..(p + 1) * pw];
                    for chunk in items.chunks(MAX_KERNELS) {
                        let ks: Vec<&[f64]> = chunk
                            .iter()
                            .map(|(_, px)| &self.taps[*px as usize][..])
                            .collect();
                        taps_sum(
                            nt,
                            |t| &row[t + x0..t + x1],
                            &ks,
                            len,
                            |q, x, v| {
                                let m = chunk[q].0;
                                out[m * len + x..m * len + x + v.len()].copy_from_slice(v);
                            },
                        );
                    }
                }
            }
        }
    }
}

/// Solves `M x = b` for a symmetric positive definite `M` by Cholesky decomposition. Only the
/// lower triangle of `m` is read; `m` is overwritten by the factor and `b` by the solution.
/// Returns false, leaving `b` meaningless, if a pivot is not positive and finite or the
/// solution is not finite.
#[cfg(test)]
fn cholesky_solve<const N: usize>(m: &mut [[f64; N]; N], b: &mut [f64; N]) -> bool {
    let mut ml = m.map(|row| row.map(|v| [v]));
    let mut bl = b.map(|v| [v]);
    let [ok] = cholesky_solve_lanes(&mut ml, &mut bl);
    *m = ml.map(|row| row.map(|[v]| v));
    *b = bl.map(|[v]| v);
    ok
}

/// [`cholesky_solve`] of `L` independent systems in lockstep (lane `l` of every entry is system
/// `l`), so that the operations vectorise across the systems. Per lane exactly the operations of
/// the scalar solve in the same order; a lane whose pivot fails goes on with meaningless values
/// (they stay in that lane) and reports false, like the scalar solve's early return.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // textbook index form of the factorisation, as in the scalar solve
fn cholesky_solve_lanes<const N: usize, const L: usize>(
    m: &mut [[[f64; L]; N]; N],
    b: &mut [[f64; L]; N],
) -> [bool; L] {
    let mut ok = [true; L];
    for j in 0..N {
        let mut d = m[j][j];
        for k in 0..j {
            let v = m[j][k];
            for l in 0..L {
                d[l] = madd(-v[l], v[l], d[l]);
            }
        }
        let mut ljj = [0.0f64; L];
        for l in 0..L {
            ok[l] &= d[l].is_finite() && d[l] > 0.0;
            ljj[l] = d[l].sqrt();
        }
        m[j][j] = ljj;
        for i in j + 1..N {
            let mut s = m[i][j];
            for k in 0..j {
                let (a, c) = (m[i][k], m[j][k]);
                for l in 0..L {
                    s[l] = madd(-a[l], c[l], s[l]);
                }
            }
            for l in 0..L {
                m[i][j][l] = s[l] / ljj[l];
            }
        }
    }
    // L y = b, then L^T x = y
    for i in 0..N {
        let mut s = b[i];
        for k in 0..i {
            let (a, c) = (m[i][k], b[k]);
            for l in 0..L {
                s[l] = madd(-a[l], c[l], s[l]);
            }
        }
        for l in 0..L {
            b[i][l] = s[l] / m[i][i][l];
        }
    }
    for i in (0..N).rev() {
        let mut s = b[i];
        for k in i + 1..N {
            let (a, c) = (m[k][i], b[k]);
            for l in 0..L {
                s[l] = madd(-a[l], c[l], s[l]);
            }
        }
        for l in 0..L {
            b[i][l] = s[l] / m[i][i][l];
        }
    }
    for l in 0..L {
        ok[l] &= b.iter().all(|v| v[l].is_finite());
    }
    ok
}

/// `np.maximum(v, 0.0)`: unlike `f64::max`, keeps NaN.
fn max0(v: f64) -> f64 {
    if v.is_nan() {
        v
    } else {
        v.max(0.0)
    }
}

#[cfg(test)]
struct Solved {
    ux: Image<f64>,
    uy: Image<f64>,
    energy: Image<f64>,
    resid: Image<f64>,
    /// Per pixel, row-major, the affine part of the fitted shape function at the window centre,
    /// `[dux/dx, duy/dx, dux/dy, duy/dy]` (`theta[2..6]`; for order 2 the first-order terms); all
    /// zero where the pixel kept its input displacement.
    slope: Vec<[f64; 4]>,
}

/// A pixel is solved only if the translation block `[[M00, M01], [M01, M11]]` of its system
/// has `det > APERTURE_TOL * energy^2` (energy = `M[0][0] + M[1][1]`): a window with a single
/// gradient direction (aperture problem) cannot determine the displacement.
pub(crate) const APERTURE_TOL: f64 = 1e-3;

/// Relative ridge `RIDGE * trace(M) / n` added to the diagonal of every system that is solved.
pub(crate) const RIDGE: f64 = 1e-12;

/// Prototypes against the drift along line textures, off by default (spec
/// 2026-11-03-convergence-fixes; read once per process, each a number in its range, otherwise
/// off). Both look at the translation block `T = [[M00, M01], [M01, M11]]` of a solved pixel,
/// its eigenvalues `l_min <= l_max` and `q = l_min / e` (`e = l_min + l_max`, the energy; `q` is
/// in `(0, 1/2]`, 1/2 for an isotropic window, about the eigenvalue ratio for a line):
///
/// - `OPTFLOW_APERTURE_CONF=q0` (`conf`, `q0 > 0`): the confidence gets the factor
///   `min(1, q / q0)`, carried in the residual (`resid^2 += phase_tol^2 ln(q0 / q)` where
///   `q < q0`), so that a line-like pixel is low for the pre-warp and invalid in the output.
/// - `OPTFLOW_APERTURE_RIDGE=a` (`ridge`, `0 < a < 1/2`): where `l_min < a e`, the weak direction
///   `v` gets the ridge `(a e - l_min) v v^T` towards the input displacement `u0` of the solve
///   (the term `(u - u0)^T R (u - u0)`, a damping of the step), which raises `l_min` to `a e`:
///   along a line each solve moves the displacement by about `l_min / (a e)` of the undamped
///   step. The residual stays the misfit of the data (with the tiny `RIDGE` term, as before),
///   not of the ridge. (A ridge towards the pre-warp `s` instead made the drift worse: `s`
///   carries the drift of the line itself; spec section 3.)
///
/// Zero is off. Off, every pixel's values are those without the prototypes, bit for bit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aperture {
    pub conf: f64,
    pub ridge: f64,
}

impl Aperture {
    /// The settings of `OPTFLOW_APERTURE_CONF` and `OPTFLOW_APERTURE_RIDGE`.
    pub fn from_env() -> Aperture {
        static AP: std::sync::OnceLock<Aperture> = std::sync::OnceLock::new();
        *AP.get_or_init(|| {
            let var = |name: &str, max: f64| {
                std::env::var(name)
                    .ok()
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .filter(|&v| v > 0.0 && v < max)
                    .unwrap_or(0.0)
            };
            Aperture {
                conf: var("OPTFLOW_APERTURE_CONF", f64::INFINITY),
                ridge: var("OPTFLOW_APERTURE_RIDGE", 0.5),
            }
        })
    }
}

/// `(l_min, l_max)` of `[[m00, m01], [m01, m11]]` with `det = m00 m11 - m01^2` (`l_min` as
/// `det / l_max`, free of the cancellation of `e / 2 - disc`).
#[inline(always)]
fn eigen2(m00: f64, m01: f64, m11: f64, det: f64) -> (f64, f64) {
    let half = 0.5 * (m00 - m11);
    let lmax = 0.5 * (m00 + m11) + (half * half + m01 * m01).sqrt();
    (det / lmax, lmax)
}

/// The ridge `R = [r00, r01, r11]` of `OPTFLOW_APERTURE_RIDGE=a` for `T` ([`Aperture`]): with
/// `P = (l_max I - T) / (l_max - l_min)` the projector on the weak direction, `(a e - l_min) P`
/// where `l_min < a e`, else `None`.
#[inline(always)]
fn aperture_ridge(
    a: f64,
    (m00, m01, m11): (f64, f64, f64),
    (lmin, lmax): (f64, f64),
) -> Option<[f64; 3]> {
    let lift = a * (lmin + lmax) - lmin;
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN lift is no ridge
    if !(lift > 0.0) {
        return None;
    }
    let c = lift / (lmax - lmin);
    Some([c * (lmax - m00), -c * m01, c * (lmax - m11)])
}

/// The residual `resid` with the confidence factor `min(1, q / q0)` of `OPTFLOW_APERTURE_CONF=q0`
/// folded in ([`Aperture`]): `sqrt(resid^2 + tol2 ln(q0 / q))` where `q < q0`.
#[inline(always)]
fn aperture_resid(resid: f64, q0: f64, q: f64, tol2: f64) -> f64 {
    if q < q0 {
        madd(resid, resid, tol2 * (q0 / q).ln()).sqrt()
    } else {
        resid
    }
}

/// What the local solve needs of the [`Aperture`] prototypes: the settings and `phase_tol^2`.
#[derive(Clone, Copy, Default)]
pub(crate) struct ApSolve {
    pub ap: Aperture,
    pub tol2: f64,
}

/// The row of each output of [`solve_row`].
struct RowOut<'a> {
    ux: &'a mut [f64],
    uy: &'a mut [f64],
    resid: &'a mut [f64],
    energy: &'a mut [f64],
    slope: &'a mut [[f64; 4]],
}

/// Pixels of a row whose systems [`solve_row`] solves in lockstep. 8 rather than 4: more
/// independent work for the long dependency chain of the factorisation (measured: moments+solve
/// at 512^2 on one thread 649 -> 602 ms; 2 lanes 713 ms, 16 lanes 593 ms).
const SOLVE_LANES: usize = 8;

/// Solves the `N x N` systems of one row (`N` is `plan.n`) from the row's window moments `m`
/// (`m[moment * w + x]`), given the input displacement of the row. Writes the energy
/// `M[0][0] + M[1][1]` of every pixel, and `(ux, uy, resid, slope)`: the solution, the residual
/// and the affine coefficients `theta[2..6]`, or the input displacement, `pi` and zeros where the
/// window spans one gradient direction only or the system is not positive definite. The energy
/// threshold of [`local_solve_into`] needs the whole image and is applied afterwards. `SOLVE_LANES`
/// neighbouring pixels are solved in lockstep (per pixel the operations of [`cholesky_solve`]); a
/// lane past the row's end repeats its last pixel and is discarded. `ap`: the [`Aperture`]
/// prototypes and `phase_tol^2`.
fn solve_row<const N: usize>(
    plan: &Plan,
    lower: &[(usize, usize, usize)],
    m: &[f64],
    (ux, uy): (&[f64], &[f64]),
    ap: (Aperture, f64),
    out: RowOut,
) {
    let RowOut {
        ux: out_ux,
        uy: out_uy,
        resid: out_resid,
        energy: out_energy,
        slope: out_slope,
    } = out;
    const L: usize = SOLVE_LANES;
    debug_assert_eq!(plan.n, N);
    let w = ux.len();
    let at = |idx: usize, x: usize| m[idx * w + x];
    let (aperture, tol2) = ap;
    for x0 in (0..w).step_by(L) {
        let xs: [usize; L] = std::array::from_fn(|l| (x0 + l).min(w - 1));
        let live = L.min(w - x0);
        let mut wanted = [false; L];
        // the eigenvalues of the translation block, for the aperture prototypes
        let mut eig = [(0.0f64, 0.0f64); L];
        let mut block = [(0.0f64, 0.0f64, 0.0f64); L];
        for (l, &x) in xs.iter().enumerate().take(live) {
            let (m00, m11) = (at(plan.m_idx[0], x), at(plan.m_idx[N + 1], x));
            let m01 = at(plan.m_idx[N], x);
            let e = m00 + m11;
            out_energy[x] = e;
            (out_ux[x], out_uy[x], out_resid[x]) = (ux[x], uy[x], PI);
            out_slope[x] = [0.0; 4];
            let det2 = m00 * m11 - m01 * m01;
            wanted[l] = det2 > APERTURE_TOL * e * e;
            if aperture != Aperture::default() {
                eig[l] = eigen2(m00, m01, m11, det2);
                block[l] = (m00, m01, m11);
            }
        }
        if !wanted.contains(&true) {
            continue;
        }
        let mut mat = [[[0.0f64; L]; N]; N];
        for &(i, j, idx) in lower {
            mat[i][j] = xs.map(|x| at(idx, x));
        }
        // ridge: keeps the system non-singular for the pixels that passed the checks
        for l in 0..L {
            let ridge = RIDGE * (0..N).map(|i| mat[i][i][l]).sum::<f64>() / N as f64;
            for (i, row) in mat.iter_mut().enumerate() {
                row[i][l] += ridge;
            }
        }
        let mut b = [[0.0f64; L]; N];
        for (bi, &idx) in b.iter_mut().zip(&plan.b_idx) {
            *bi = xs.map(|x| at(idx, x));
        }
        let mut theta = b;
        // the aperture ridge towards the input u0: R on the translation block, R u0 on b
        #[allow(clippy::type_complexity)]
        let mut ridged: [Option<([f64; 3], (f64, f64))>; L] = [None; L];
        if aperture.ridge > 0.0 {
            for (l, &x) in xs.iter().enumerate().take(live) {
                if !wanted[l] {
                    continue;
                }
                let Some(r) = aperture_ridge(aperture.ridge, block[l], eig[l]) else {
                    continue;
                };
                let s = (ux[x], uy[x]);
                mat[0][0][l] += r[0];
                mat[1][0][l] += r[1];
                mat[1][1][l] += r[2];
                theta[0][l] += r[0] * s.0 + r[1] * s.1;
                theta[1][l] += r[1] * s.0 + r[2] * s.1;
                ridged[l] = Some((r, s));
            }
        }
        let solved = cholesky_solve_lanes(&mut mat, &mut theta);
        for (l, &x) in xs.iter().enumerate().take(live) {
            if !(wanted[l] && solved[l]) {
                continue;
            }
            let mut fit = 0.0;
            for (t, bi) in theta.iter().zip(&b) {
                fit = madd(t[l], bi[l], fit);
            }
            let mut z = at(plan.gcc_idx, x) - fit;
            if let Some((r, s)) = ridged[l] {
                // the misfit of the data: + (u0 - u)^T R u, u the solved translation
                let (u0, u1) = (theta[0][l], theta[1][l]);
                let (d0, d1) = (s.0 - u0, s.1 - u1);
                z += d0 * (r[0] * u0 + r[1] * u1) + d1 * (r[1] * u0 + r[2] * u1);
            }
            let mut resid = (max0(z) / (at(plan.gw_idx, x) + 1e-30)).sqrt();
            if aperture.conf > 0.0 {
                let (lmin, lmax) = eig[l];
                resid = aperture_resid(resid, aperture.conf, lmin / (lmin + lmax), tol2);
            }
            (out_ux[x], out_uy[x], out_resid[x]) = (theta[0][l], theta[1][l], resid);
            out_slope[x] = [theta[2][l], theta[3][l], theta[4][l], theta[5][l]];
        }
    }
}

/// [`local_solve_into`] into fresh buffers (the tests' entry point).
#[cfg(test)]
fn local_solve(sums: &Sums, ux: &Image<f64>, uy: &Image<f64>, sigma: f64, order: u32) -> Solved {
    let (w, h) = (ux.w, ux.h);
    let plane = || vec![0.0f64; w * h];
    let (mut oux, mut ouy, mut ores, mut oen) = (plane(), plane(), plane(), plane());
    let mut oslope = vec![[0.0f64; 4]; w * h];
    let out = SolveOut {
        ux: &mut oux,
        uy: &mut ouy,
        energy: &mut oen,
        resid: &mut ores,
        slope: &mut oslope,
    };
    local_solve_into(sums, ux, uy, sigma, order, ApSolve::default(), out);
    Solved {
        ux: Image::from_vec(w, h, oux),
        uy: Image::from_vec(w, h, ouy),
        energy: Image::from_vec(w, h, oen),
        resid: Image::from_vec(w, h, ores),
        slope: oslope,
    }
}

/// Where [`local_solve_into`] writes its results, one value per pixel each (row-major); every
/// value is written, and none of them is read.
pub(crate) struct SolveOut<'a> {
    pub(crate) ux: &'a mut [f64],
    pub(crate) uy: &'a mut [f64],
    pub(crate) energy: &'a mut [f64],
    pub(crate) resid: &'a mut [f64],
    pub(crate) slope: &'a mut [[f64; 4]],
}

/// Per-pixel weighted least squares for the polynomial shape function of `order` (1 affine,
/// 2 quadratic) from the sums over the filter bank (`local_solve` of the reference).
///
/// `slope` is the affine part `theta[2..6]` of the solution, `(dux/dx, duy/dx, dux/dy, duy/dy)`
/// (the unknowns after `ux`, `uy` are ordered by power of the offset, then `x` before `y`), 0 where
/// the pixel is not solved. `energy` is `M[0][0] + M[1][1]`. A pixel is solved where
/// `energy > 1e-9 median(energy) + 1e-30` and `M[0][0] M[1][1] - M[0][1]^2 > APERTURE_TOL
/// energy^2` (the gradients in its window span two directions), after adding the ridge
/// `RIDGE trace(M) / n` to the diagonal, and if its system is positive definite with a finite
/// solution; all others keep the input displacement and get `resid = pi`. At the least-squares optimum the residual sum is
/// `sum W w c^2 - theta . b`, so `resid = sqrt(max(that, 0) / (sum W w + 1e-30))`.
///
/// One pass over the rows (in parallel) computes each row's window moments and solves its
/// systems at once, so no full-size moment map is held; the energy threshold, which needs the
/// median over the image, then resets the pixels below it. Every pixel's values are those of
/// moment maps of the whole image followed by the solve, bit for bit.
///
/// The results go into buffers of the caller (`out`, which the iterations of [`refine_raw`]
/// reuse). The moments pass also yields the smallest and the largest energy: when even the
/// smallest is above `1e-9 max + 1e-30`, the threshold cannot reset a pixel and the median is
/// skipped.
///
/// `ap`: the [`Aperture`] prototypes (off: [`ApSolve::default`]).
pub(crate) fn local_solve_into(
    sums: &Sums,
    ux: &Image<f64>,
    uy: &Image<f64>,
    sigma: f64,
    order: u32,
    ap: ApSolve,
    out: SolveOut,
) {
    let (w, h) = (ux.w, ux.h);
    let SolveOut {
        ux: oux,
        uy: ouy,
        energy: oen,
        resid: ores,
        slope: oslope,
    } = out;
    for len in [oux.len(), ouy.len(), oen.len(), ores.len(), oslope.len()] {
        assert_eq!(len, w * h, "local_solve: output of the wrong size");
    }
    let plan = plan(order);
    let rows = MomentRows::new(&plan, sigma);
    let n = plan.n;
    // the moments the lower triangle of M is read from
    let lower: Vec<(usize, usize, usize)> = (0..n)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .map(|(i, j)| (i, j, plan.m_idx[i * n + j]))
        .collect();
    // the smallest and the largest energy, and whether one is NaN (row by row, then reduced)
    let (lowest, highest, nan) = prof::time(Stage::Moments, || {
        let outs = (
            oux.par_chunks_mut(w),
            ouy.par_chunks_mut(w),
            ores.par_chunks_mut(w),
            oen.par_chunks_mut(w),
            oslope.par_chunks_mut(w),
        );
        outs.into_par_iter()
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0f64; rows.pad_len(w)],
                        vec![0.0f64; rows.n_moments * w],
                    )
                },
                |(pad, m), (y, (o_ux, o_uy, o_res, o_en, o_sl))| {
                    rows.row(sums, y, pad, m);
                    let input = (&ux.data[y * w..(y + 1) * w], &uy.data[y * w..(y + 1) * w]);
                    let aprow = (ap.ap, ap.tol2);
                    let out = RowOut {
                        ux: o_ux,
                        uy: o_uy,
                        resid: o_res,
                        energy: o_en,
                        slope: o_sl,
                    };
                    match n {
                        6 => solve_row::<6>(&plan, &lower, m, input, aprow, out),
                        12 => solve_row::<12>(&plan, &lower, m, input, aprow, out),
                        n => unreachable!("shape function with {n} unknowns"),
                    }
                    energy_range(o_en)
                },
            )
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY, false),
                |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 | b.2),
            )
    });
    // the threshold 1e-9 median + 1e-30 is at most 1e-9 max + 1e-30 (the median is at most the
    // largest value, and the rounded product and sum are monotonic): if even the smallest
    // energy is above that, no pixel is reset and the median is not needed. A NaN energy makes
    // the median NaN, which resets every pixel: then the full path below. The median of an even
    // count is the mean of the two middle values, at most the largest only while their sum does
    // not overflow: hence `highest <= f64::MAX / 2`
    if !nan && highest <= f64::MAX / 2.0 && lowest > 1e-9 * highest + 1e-30 {
        return;
    }
    prof::time(Stage::Solve, || {
        let threshold = 1e-9 * median(oen) + 1e-30;
        oux.par_chunks_mut(w)
            .zip(ouy.par_chunks_mut(w))
            .zip(ores.par_chunks_mut(w))
            .zip(oslope.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (((o_ux, o_uy), o_res), o_sl))| {
                for x in 0..w {
                    let p = y * w + x;
                    let textured = oen[p] > threshold; // false for NaN, like the reference
                    if !textured {
                        (o_ux[x], o_uy[x], o_res[x]) = (ux.data[p], uy.data[p], PI);
                        o_sl[x] = [0.0; 4];
                    }
                }
            });
    });
}

/// [`local_solve_into`] for some pixels of a rectangle of a larger image, with the same bits at
/// those pixels: `sums` holds the rectangle, `(ox, oy)` is its corner in `ux` and `uy`, and row `y`
/// of the rectangle is solved for `x` in `spans[y]` (an empty span skips the row). Every pixel
/// solved must have its whole window inside the rectangle, or the rectangle's edge on the image's.
/// The outputs have the size of the rectangle; other pixels are left as they are.
///
/// A pixel without texture is one with an energy at or below `threshold`
/// (`local_solve_into`: `1e-9` times the median energy plus `1e-30`, which needs the solve of the
/// whole image; the caller gives it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn local_solve_spans(
    sums: &Sums,
    (ux, uy): (&Image<f64>, &Image<f64>),
    (ox, oy): (usize, usize),
    spans: &[(usize, usize)],
    sigma: f64,
    order: u32,
    ap: ApSolve,
    threshold: f64,
    out: SolveOut,
) {
    let (w, h) = (sums.gw.w, sums.gw.h);
    assert_eq!(spans.len(), h, "local_solve_spans: one span per row");
    let SolveOut {
        ux: oux,
        uy: ouy,
        energy: oen,
        resid: ores,
        slope: oslope,
    } = out;
    let plan = plan(order);
    let rows = MomentRows::new(&plan, sigma);
    let n = plan.n;
    let lower: Vec<(usize, usize, usize)> = (0..n)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .map(|(i, j)| (i, j, plan.m_idx[i * n + j]))
        .collect();
    let (full_w, tol2) = (ux.w, ap.tol2);
    let outs = (
        oux.par_chunks_mut(w),
        ouy.par_chunks_mut(w),
        ores.par_chunks_mut(w),
        oen.par_chunks_mut(w),
        oslope.par_chunks_mut(w),
    );
    outs.into_par_iter().enumerate().for_each_init(
        || (vec![0.0f64; rows.pad_len(w)], Vec::new()),
        |(pad, m): &mut (Vec<f64>, Vec<f64>), (y, (o_ux, o_uy, o_res, o_en, o_sl))| {
            let (x0, x1) = spans[y];
            if x0 >= x1 {
                return;
            }
            let len = x1 - x0;
            m.resize(rows.n_moments * len, 0.0);
            rows.row_span(sums, y, (x0, x1), pad, m);
            let at = (y + oy) * full_w + ox;
            let input = (&ux.data[at + x0..at + x1], &uy.data[at + x0..at + x1]);
            let row_out = RowOut {
                ux: &mut o_ux[x0..x1],
                uy: &mut o_uy[x0..x1],
                resid: &mut o_res[x0..x1],
                energy: &mut o_en[x0..x1],
                slope: &mut o_sl[x0..x1],
            };
            match n {
                6 => solve_row::<6>(&plan, &lower, m, input, (ap.ap, tol2), row_out),
                12 => solve_row::<12>(&plan, &lower, m, input, (ap.ap, tol2), row_out),
                n => unreachable!("shape function with {n} unknowns"),
            }
            for x in x0..x1 {
                let textured = o_en[x] > threshold; // false for NaN, like the reference
                if !textured {
                    let k = x - x0;
                    (o_ux[x], o_uy[x], o_res[x]) = (input.0[k], input.1[k], PI);
                    o_sl[x] = [0.0; 4];
                }
            }
        },
    );
}

/// The smallest and the largest of `v` (NaN left out) and whether `v` holds a NaN.
fn energy_range(v: &[f64]) -> (f64, f64, bool) {
    v.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, false),
        |(lo, hi, nan), &e| (lo.min(e), hi.max(e), nan | e.is_nan()),
    )
}

pub(crate) fn to_f64(im: &Image<f32>) -> Image<f64> {
    Image::from_vec(
        im.w,
        im.h,
        im.data.par_iter().map(|&v| f64::from(v)).collect(),
    )
}

pub(crate) fn to_f32(im: &Image<f64>) -> Image<f32> {
    Image::from_vec(im.w, im.h, im.data.par_iter().map(|&v| v as f32).collect())
}

/// B-spline coefficients of the bank responses on the padded grid (not cropped), the `n_filters`
/// values of one grid node adjacent in memory: `data[(y * w + x) * n_filters + k]` for filter `k`.
pub(crate) struct Responses {
    pub(crate) w: usize,
    pub(crate) n_filters: usize,
    pub(crate) data: Vec<Complex32>,
}

impl Responses {
    /// Responses of `warped` to every filter of `bank`.
    ///
    /// Per filter: the inverse transform of the filtered spectrum, then the B-spline prefilter,
    /// with the values of `BSpline::new` of the `fft.inverse` result bit for bit. Up to `GROUP`
    /// filters are in flight at once, and every step splits into independent blocks of `STRIP`
    /// lines: the transforms along the rows, then per block of columns the transforms along the
    /// columns, the scale and the prefilter along y, then per block of rows the prefilter along
    /// x. A prefilter pass filters the lines of a block side by side (`prefilter_rows` filters
    /// every column of a block with the operations of the line filter), so both directions
    /// vectorise; the coefficients go straight into the node layout.
    #[cfg(test)]
    fn new(warped: &Image<f32>, bank: &Bank, fft: &Fft2) -> Self {
        Self::new_in(warped, bank, fft, Vec::new())
    }

    /// `Responses::new` stored in `reuse` if that has the size (the storage of the responses
    /// of the last outer iteration): every value is written, so its contents do not matter.
    pub(crate) fn new_in(
        warped: &Image<f32>,
        bank: &Bank,
        fft: &Fft2,
        reuse: Vec<Complex32>,
    ) -> Self {
        // blocks of `LINES` lines: the transforms pair the lines as `Fft2::inverse` does
        const STRIP: usize = crate::fft::LINES;
        const GROUP: usize = 4;
        let zero = Complex32::default();
        let spectrum = padded_spectrum(warped, fft);
        let (w, h, f) = (bank.w, bank.h, bank.filters.len());
        let (row_plan, col_plan, scale) = fft.inverse_parts();
        let (row_scratch, col_scratch) = (
            row_plan.get_inplace_scratch_len(),
            col_plan.get_inplace_scratch_len(),
        );
        let g = GROUP.min(f);
        let mut data = if reuse.len() == w * h * f {
            reuse
        } else {
            drop(reuse);
            vec![zero; w * h * f]
        };
        // per filter in flight: the transforms along the rows, `rows[y * w + x]`, and the
        // coefficients after the pass along y, column-major with the lanes adjacent,
        // `ty[(x * h + y) * 2 + lane]`
        let mut rows: Vec<Vec<Complex32>> = (0..g).map(|_| vec![zero; w * h]).collect();
        let mut ty: Vec<Vec<f64>> = (0..g).map(|_| vec![0.0; 2 * w * h]).collect();
        for k0 in (0..f).step_by(g) {
            let filters = &bank.filters[k0..(k0 + g).min(f)];
            let n_in = filters.len();

            // filtered spectrum and the transforms along the rows
            let mut items: Vec<(&[f32], &[Complex32], &mut [Complex32])> = Vec::new();
            for (gk, r) in filters.iter().zip(rows.iter_mut()) {
                let chunks = gk.chunks(w * STRIP).zip(spectrum.chunks(w * STRIP));
                for ((gc, sc), rc) in chunks.zip(r.chunks_mut(w * STRIP)) {
                    items.push((gc, sc, rc));
                }
            }
            items.into_par_iter().for_each_init(
                || vec![zero; row_scratch],
                |s, (gc, sc, rc)| {
                    for ((o, &sv), &gv) in rc.iter_mut().zip(sc).zip(gc) {
                        *o = sv * gv;
                    }
                    row_plan.process_with_scratch(rc, s);
                },
            );

            // per block of columns: transforms along the columns, scale, prefilter along y
            let mut items: Vec<(&[Complex32], usize, &mut [f64])> = Vec::new();
            for (r, t) in rows.iter().zip(ty.iter_mut()).take(n_in) {
                for (b, tc) in t.chunks_mut(2 * h * STRIP).enumerate() {
                    items.push((r, b, tc));
                }
            }
            items.into_par_iter().for_each_init(
                || (vec![zero; h * STRIP], vec![zero; col_scratch]),
                |(lines, s), (r, b, tc)| {
                    let (x0, n) = (b * STRIP, tc.len() / (2 * h));
                    let lines = &mut lines[..h * n];
                    for (j, line) in lines.chunks_mut(h).enumerate() {
                        for (y, v) in line.iter_mut().enumerate() {
                            *v = r[y * w + x0 + j];
                        }
                    }
                    col_plan.process_with_scratch(lines, s);
                    // scaled, as f64 lanes, straight into the strip of `ty`
                    for (line, out) in lines.chunks(h).zip(tc.chunks_mut(2 * h)) {
                        for (&v, o) in line.iter().zip(out.chunks_mut(2)) {
                            let v = v * scale;
                            o[0] = f64::from(v.re);
                            o[1] = f64::from(v.im);
                        }
                    }
                    prefilter_pair_lines(tc, n, h);
                },
            );

            // per block of rows: prefilter along x, into the node layout (`ty` holds the lanes of
            // consecutive rows adjacent)
            data.par_chunks_mut(w * f * STRIP)
                .enumerate()
                .for_each_init(
                    || vec![0.0f64; 2 * w * STRIP],
                    |buf, (b, dst)| {
                        let (y0, n) = (b * STRIP, dst.len() / (w * f));
                        let buf = &mut buf[..2 * w * n];
                        for (j, t) in ty.iter().take(n_in).enumerate() {
                            for (x, row) in buf.chunks_mut(2 * n).enumerate() {
                                row.copy_from_slice(&t[(x * h + y0) * 2..(x * h + y0 + n) * 2]);
                            }
                            prefilter_rows(buf, 2 * n, w);
                            for (r, out) in dst.chunks_mut(w * f).enumerate() {
                                for (x, node) in out.chunks_mut(f).enumerate() {
                                    let c = &buf[x * 2 * n + 2 * r..x * 2 * n + 2 * r + 2];
                                    node[k0 + j] = Complex32::new(c[0] as f32, c[1] as f32);
                                }
                            }
                        }
                    },
                );
        }
        Self {
            w,
            n_filters: f,
            data,
        }
    }

    /// The spline of every filter at the sample point `t`, as `(re, im)` in `out[k]`. Per
    /// filter exactly the arithmetic of `BSpline::sample` in `f64` (not rounded to `f32`); the
    /// taps are computed once for all filters and the loops run over the filters. `n` must be
    /// `n_filters` (passed so that a caller with a constant count gets the loops unrolled), and
    /// `scratch` and `out` hold `n` values.
    #[inline(always)]
    fn sample_all(&self, n: usize, t: &Taps, scratch: &mut [[f64; 2]], out: &mut [[f64; 2]]) {
        debug_assert_eq!(n, self.n_filters);
        let (scratch, out) = (&mut scratch[..n], &mut out[..n]);
        out.fill([0.0; 2]);
        for (&yj, &wyj) in t.yi.iter().zip(&t.wy) {
            let row = &self.data[yj * self.w * n..(yj + 1) * self.w * n];
            let (x0, wx0) = (t.xi[0], t.wx[0]);
            for (a, c) in scratch.iter_mut().zip(&row[x0 * n..(x0 + 1) * n]) {
                *a = [wx0 * f64::from(c.re), wx0 * f64::from(c.im)];
            }
            for (&xk, &wxk) in t.xi.iter().zip(&t.wx).skip(1) {
                for (a, c) in scratch.iter_mut().zip(&row[xk * n..(xk + 1) * n]) {
                    a[0] = madd(wxk, f64::from(c.re), a[0]);
                    a[1] = madd(wxk, f64::from(c.im), a[1]);
                }
            }
            for (o, a) in out.iter_mut().zip(scratch.iter()) {
                o[0] = madd(wyj, a[0], o[0]);
                o[1] = madd(wyj, a[1], o[1]);
            }
        }
    }
}

/// The field of the current outer iteration, smoothed and spline-ready.
pub(crate) struct Smoothed {
    pub(crate) sx: Image<f32>,
    pub(crate) sy: Image<f32>,
    pub(crate) csx: BSpline<f32>,
    pub(crate) csy: BSpline<f32>,
}

/// The reference phase of the first image packed per pixel for `accumulate`: for pixel `i` and
/// filter `k`, `data[(i * n_filters + k) * 5..][..5]` is `[r.re, r.im, gx, gy, amp]`.
pub(crate) struct RefPack {
    pub(crate) n_filters: usize,
    pub(crate) data: Vec<f32>,
}

impl RefPack {
    /// The reference phase of `img` ([`crate::gabor::reference_phase`]) packed. `GROUP` filters
    /// at a time are transformed in parallel and packed straight from their padded inverse
    /// transforms, so neither the unpacked filters nor their crops are held.
    pub(crate) fn new(img: &Image<f32>, bank: &Bank, fft: &Fft2) -> Self {
        const GROUP: usize = 4;
        let spectrum = reference_spectrum(img, bank, fft);
        let (w, f, pw) = (img.w, bank.filters.len(), bank.w);
        let mut data = vec![0.0f32; img.w * img.h * f * 5];
        for (gi, group) in bank.filters.chunks(GROUP).enumerate() {
            let inverses: Vec<[Vec<Complex32>; 3]> = group
                .par_iter()
                .map(|g| reference_inverses(&spectrum, g, bank, fft))
                .collect();
            data.par_chunks_mut(f * 5 * w.max(1))
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, px) in row.chunks_mut(f * 5).enumerate() {
                        let q = (y + PAD) * pw + x + PAD;
                        for (j, [r, rx, ry]) in inverses.iter().enumerate() {
                            let [gx, gy, amp] = reference_values(r[q], rx[q], ry[q]);
                            let k = gi * GROUP + j;
                            px[5 * k..5 * k + 5].copy_from_slice(&[r[q].re, r[q].im, gx, gy, amp]);
                        }
                    }
                });
        }
        Self { n_filters: f, data }
    }
}

/// Evaluates the splines with coefficients `a` and `b` (images of the same size) at the sample
/// point described by `t`, in `f64` and not rounded to `f32`: per image the arithmetic of
/// `BSpline::sample`, with the taps shared.
#[inline(always)]
fn sample_taps2(a: &Image<f32>, b: &Image<f32>, t: &Taps) -> (f64, f64) {
    let w = a.w;
    let (mut ta, mut tb) = (0.0, 0.0);
    for (&yj, &wyj) in t.yi.iter().zip(&t.wy) {
        let (ra, rb) = (&a.data[yj * w..(yj + 1) * w], &b.data[yj * w..(yj + 1) * w]);
        let mut aa = t.wx[0] * f64::from(ra[t.xi[0]]);
        let mut ab = t.wx[0] * f64::from(rb[t.xi[0]]);
        for (&xk, &wxk) in t.xi.iter().zip(&t.wx).skip(1) {
            aa = madd(wxk, f64::from(ra[xk]), aa);
            ab = madd(wxk, f64::from(rb[xk]), ab);
        }
        ta = madd(wyj, aa, ta);
        tb = madd(wyj, ab, tb);
    }
    (ta, tb)
}

/// Scratch of one pixel of `accumulate`, `n` filters long.
struct PixelScratch<'a> {
    acc: &'a mut [[f64; 2]],
    sampled: &'a mut [[f64; 2]],
    num: &'a mut [f64],
    den: &'a mut [f64],
    mag: &'a mut [f64],
    dphi: &'a mut [f64],
}

/// Pixels of a row whose fixed points `accumulate` runs in lockstep ([`fixed_points`]): the two
/// steps are a chain of dependent samples, latency-bound for one pixel alone (measured,
/// accumulate at 512^2 on one thread: 1 pixel 738 ms, 2 lanes 652 ms, 4 lanes 627 ms, 8 lanes
/// 628 ms).
const FIXED_POINT_LANES: usize = 4;

/// The positions in the pre-warped image of `L` pixels `(x, i, (u, v))` of row `y` (`i` the
/// pixel's index): `X' + s(X') = X + u(X)` by two fixed-point steps, the pixels in lockstep (per
/// pixel the operations of one alone), so that their chains of dependent samples overlap.
#[inline(always)]
fn fixed_points<const L: usize>(
    sm: &Smoothed,
    y: usize,
    px: [(usize, usize, (f64, f64)); L],
) -> [(f64, f64); L] {
    let (w, h) = (sm.sx.w, sm.sx.h);
    let t: [(f64, f64); L] = px.map(|(x, _, (u, v))| (x as f64 + u, y as f64 + v));
    let mut q: [(f64, f64); L] = std::array::from_fn(|l| {
        let i = px[l].1;
        (
            t[l].0 - f64::from(sm.sx.data[i]),
            t[l].1 - f64::from(sm.sy.data[i]),
        )
    });
    for _ in 0..2 {
        for l in 0..L {
            let (dx, dy) = match taps(q[l].0, q[l].1, w, h) {
                Some(tp) => sample_taps2(&sm.csx.coeffs, &sm.csy.coeffs, &tp),
                None => (f64::NAN, f64::NAN),
            };
            q[l] = (t[l].0 - dx, t[l].1 - dy);
        }
    }
    q
}

/// The sums of `accumulate` at the pixel whose position in the pre-warped image is `q`
/// ([`fixed_points`]); `(u, v)` is its displacement and `rp` its [`RefPack`] entry. Written once
/// for a filter count known at compile time and for any count (`n` = the slice lengths).
#[inline(always)]
fn pixel_sums_at(
    n: usize,
    responses: &Responses,
    sm: &Smoothed,
    (qx, qy): (f64, f64),
    (u, v): (f64, f64),
    rp: &[f32],
    s: &mut PixelScratch,
) -> [f64; 7] {
    let (w, h) = (sm.sx.w, sm.sx.h);
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let Some(t) = taps(qx + PAD as f64, qy + PAD as f64, pw, ph) else {
        return [f64::NAN; 7];
    };
    let (acc, sampled) = (&mut s.acc[..n], &mut s.sampled[..n]);
    responses.sample_all(n, &t, acc, sampled);
    // dphi = angle(r1 * conj(r2)) from its two components, all filters at once
    // (`crate::atan2`, the same on every platform); the weight's |r2|
    let (num, den, mag) = (&mut s.num[..n], &mut s.den[..n], &mut s.mag[..n]);
    for k in 0..n {
        let (a_re, a_im) = (f64::from(rp[5 * k]), f64::from(rp[5 * k + 1]));
        let [b_re, b_im] = sampled[k];
        num[k] = madd(a_im, b_re, -(a_re * b_im));
        den[k] = madd(a_re, b_re, a_im * b_im);
        mag[k] = madd(b_re, b_re, b_im * b_im).sqrt();
    }
    let dphi = &mut s.dphi[..n];
    atan2_all(num, den, dphi);
    let mut out = [0.0f64; 7];
    for k in 0..n {
        let dphi = dphi[k];
        let wk = f64::from(rp[5 * k + 4]) * mag[k];
        let (gx, gy) = (f64::from(rp[5 * k + 2]), f64::from(rp[5 * k + 3]));
        let c = madd(gy, v, madd(gx, u, dphi));
        out[0] = madd(wk * gx, gx, out[0]);
        out[1] = madd(wk * gx, gy, out[1]);
        out[2] = madd(wk * gy, gy, out[2]);
        out[3] = madd(wk * gx, c, out[3]);
        out[4] = madd(wk * gy, c, out[4]);
        out[5] = madd(wk * c, c, out[5]);
        out[6] += wk;
    }
    out
}

/// One inner iteration's sums, into `sums` (planes of the size of `ux`; every value is written):
/// samples every response at each pixel's residual position and accumulates the phase-gradient
/// normal equations over the filters (in filter order).
pub(crate) fn accumulate_into(
    reference: &RefPack,
    responses: &Responses,
    sm: &Smoothed,
    ux: &Image<f64>,
    uy: &Image<f64>,
    sums: &mut Sums,
) {
    assert_eq!(reference.n_filters, responses.n_filters);
    match responses.n_filters {
        12 => accumulate_with::<12>(reference, responses, sm, ux, uy, sums),
        _ => accumulate_with::<0>(reference, responses, sm, ux, uy, sums),
    }
}

/// `accumulate_into` for `F` filters (`F = 0`: any number, in heap scratch).
fn accumulate_with<const F: usize>(
    reference: &RefPack,
    responses: &Responses,
    sm: &Smoothed,
    ux: &Image<f64>,
    uy: &Image<f64>,
    sums: &mut Sums,
) {
    let (w, h) = (ux.w, ux.h);
    {
        let Sums {
            gxx,
            gxy,
            gyy,
            gxc,
            gyc,
            gcc,
            gw,
        } = sums;
        let [s0, s1, s2, s3, s4, s5, s6] = [gxx, gxy, gyy, gxc, gyc, gcc, gw].map(|p| {
            assert_eq!((p.w, p.h), (w, h), "accumulate: sums of the wrong size");
            &mut p.data
        });
        let rows = (
            s0.par_chunks_mut(w),
            s1.par_chunks_mut(w),
            s2.par_chunks_mut(w),
            s3.par_chunks_mut(w),
            s4.par_chunks_mut(w),
            s5.par_chunks_mut(w),
            s6.par_chunks_mut(w),
        );
        rows.into_par_iter()
            .enumerate()
            .for_each(|(y, (r0, r1, r2, r3, r4, r5, r6))| {
                // bound here, not captured: a constant for `F > 0`, so the loops over the
                // filters unroll
                let n = if F == 0 { responses.n_filters } else { F };
                let (mut acc_f, mut sampled_f) = ([[0.0f64; 2]; F], [[0.0f64; 2]; F]);
                let (mut num_f, mut den_f, mut mag_f) = ([0.0f64; F], [0.0f64; F], [0.0f64; F]);
                let mut dphi_f = [0.0f64; F];
                let mut heap = (
                    vec![[0.0f64; 2]; if F == 0 { n } else { 0 }],
                    vec![[0.0f64; 2]; if F == 0 { n } else { 0 }],
                    vec![0.0f64; if F == 0 { 4 * n } else { 0 }],
                );
                let mut scratch = if F == 0 {
                    let (num, rest) = heap.2.split_at_mut(n);
                    let (den, rest) = rest.split_at_mut(n);
                    let (mag, dphi) = rest.split_at_mut(n);
                    PixelScratch {
                        acc: &mut heap.0,
                        sampled: &mut heap.1,
                        num,
                        den,
                        mag,
                        dphi,
                    }
                } else {
                    PixelScratch {
                        acc: &mut acc_f,
                        sampled: &mut sampled_f,
                        num: &mut num_f,
                        den: &mut den_f,
                        mag: &mut mag_f,
                        dphi: &mut dphi_f,
                    }
                };
                let mut x = 0;
                while x < w {
                    // FIXED_POINT_LANES pixels at a time: their fixed points in lockstep (a lane
                    // past the end of the row repeats the last pixel and is not used)
                    let live = FIXED_POINT_LANES.min(w - x);
                    let px: [(usize, usize, (f64, f64)); FIXED_POINT_LANES] =
                        std::array::from_fn(|l| {
                            let xx = x + l.min(live - 1);
                            let i = y * w + xx;
                            (xx, i, (ux.data[i], uy.data[i]))
                        });
                    let q = fixed_points(sm, y, px);
                    for (&(xx, i, uv), &q) in px.iter().zip(&q).take(live) {
                        let rp = &reference.data[i * n * 5..(i + 1) * n * 5];
                        let v = pixel_sums_at(n, responses, sm, q, uv, rp, &mut scratch);
                        (r0[xx], r1[xx], r2[xx], r3[xx], r4[xx], r5[xx], r6[xx]) =
                            (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
                    }
                    x += FIXED_POINT_LANES;
                }
            });
    }
}

/// Iteration result before the confidence is formed, one value per pixel of the first image, in
/// the `f64` the iterations carry. [`refine`] turns it into a [`Flow`]; the tiled driver joins the
/// pieces of several windows first, because the confidence depends on the whole image.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct RawFlow {
    pub ux: Image<f64>,
    pub uy: Image<f64>,
    /// `M[0][0] + M[1][1]` of the last local solve (the texture measure of the confidence).
    pub energy: Image<f64>,
    pub resid: Image<f64>,
}

/// `img(X + shift + ux, Y + shift + uy)` on the pixel grid of `ux`, from `spline`, the
/// `BSpline` of `img` (`warp` of `interp` with the lookup window displaced by `shift`; `img`
/// need not have the size of `ux`). For a zero shift the arithmetic is that of `interp::warp`,
/// bit for bit.
pub(crate) fn warp_shifted(
    spline: &BSpline<f32>,
    ux: &Image<f32>,
    uy: &Image<f32>,
    shift: (i64, i64),
) -> Image<f32> {
    let w = ux.w;
    let (sx, sy) = (shift.0 as f64, shift.1 as f64);
    let mut out = Image::new(ux.w, ux.h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = y * w + x;
            *o = spline.sample(
                (x as f64 + sx) + f64::from(ux.data[i]),
                (y as f64 + sy) + f64::from(uy.data[i]),
            );
        }
    });
    out
}

/// The iterations of [`refine`] on a window of a larger problem: `i1`, `ux` and `uy` cover the
/// window, `i2` any part of the second image. Pixel `(x, y)` of `i1` sits at `(x + shift.0,
/// y + shift.1)` in `i2` before it is displaced by the field, so `i2` must cover every position
/// `X + shift + u(X)` (plus the support of the cubic spline); positions outside it are clamped to
/// its border. `ux`, `uy` are displacements of the full problem, `shift` only places the windows.
/// With `i2` of the size of `i1` and `shift = (0, 0)` this is [`refine`] without the confidence.
pub(crate) fn refine_raw(
    i1: &Image<f32>,
    i2: &Image<f32>,
    shift: (i64, i64),
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
) -> RawFlow {
    refine_raw_slope(i1, i2, shift, ux, uy, cfg).0
}

/// [`refine_raw`] and the `slope` of its last local solve (`[dux/dx, duy/dx, dux/dy, duy/dy]` per
/// pixel, row-major), which `crack_model = "xdic"` needs (`Flow.slope` of the reference). The
/// iterations are those of `refine_raw`, bit for bit.
pub(crate) fn refine_raw_slope(
    i1: &Image<f32>,
    i2: &Image<f32>,
    shift: (i64, i64),
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
) -> (RawFlow, Vec<[f64; 4]>) {
    let (w, h) = (i1.w, i1.h);
    assert!(w >= 1 && h >= 1, "refine: empty image {w}x{h}");
    assert!(i2.w >= 1 && i2.h >= 1, "refine: empty image i2");
    for (name, im) in [("ux", ux), ("uy", uy)] {
        assert!(
            (im.w, im.h) == (w, h),
            "refine: i1 is {w}x{h} but {name} is {}x{}",
            im.w,
            im.h
        );
    }
    let prep = Prepared::new(i1, i2, cfg);
    refine_raw_prepared(&prep, i1.w, i1.h, shift, ux, uy, cfg)
}

/// What the iterations of a pair need of its two images: the filter bank, the transforms, the
/// reference phase of the first and the spline of the second.
pub(crate) struct Prepared {
    pub(crate) bank: Bank,
    pub(crate) fft: Fft2,
    pub(crate) reference: RefPack,
    pub(crate) spline2: BSpline<f32>,
}

impl Prepared {
    pub(crate) fn new(i1: &Image<f32>, i2: &Image<f32>, cfg: &Config) -> Prepared {
        let (ph, pw) = (i1.h + 2 * PAD, i1.w + 2 * PAD);
        let (bank, fft, reference) = prof::time(Stage::Reference, || {
            let bank = gabor_bank(ph, pw, cfg);
            let fft = Fft2::new(ph, pw);
            let reference = RefPack::new(i1, &bank, &fft);
            (bank, fft, reference)
        });
        // the spline of the second image does not change over the iterations
        let spline2 = prof::time(Stage::Warp, || BSpline::new(i2));
        Prepared {
            bank,
            fft,
            reference,
            spline2,
        }
    }
}

/// [`refine_raw_slope`] on what is prepared for the pair (of a first image of `w x h`).
pub(crate) fn refine_raw_prepared(
    prep: &Prepared,
    w: usize,
    h: usize,
    shift: (i64, i64),
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
) -> (RawFlow, Vec<[f64; 4]>) {
    let (bank, fft, reference, spline2) = (&prep.bank, &prep.fft, &prep.reference, &prep.spline2);
    let (mut ux, mut uy) = (to_f64(ux), to_f64(uy));
    let mut resid = Image::from_vec(w, h, vec![PI; w * h]);
    let mut energy = Image::from_vec(w, h, vec![1.0; w * h]);
    let mut slope: Vec<[f64; 4]> = Vec::new();
    // the storage of the last outer iteration's responses, for the next one's
    let mut spent: Vec<Complex32> = Vec::new();
    let trace = crate::iter_trace::enabled();
    if trace {
        crate::iter_trace::begin(w, &to_f32(&ux).data, &to_f32(&uy).data);
    }
    let mut freeze = crate::prewarp::LowFreeze::for_config(cfg);
    let aperture = Aperture::from_env();
    for it in 0..cfg.outer_iters {
        let t_it = trace.then(std::time::Instant::now);
        let (sx, sy) = if it == 0 {
            // no solve yet to tell the trusted pixels from the rest
            prof::time(Stage::Smooth, || {
                (
                    to_f32(&gauss_blur(&ux, cfg.sigma_s)),
                    to_f32(&gauss_blur(&uy, cfg.sigma_s)),
                )
            })
        } else {
            let low = prof::time(Stage::Prewarp, || {
                freeze.low(it, || low_confidence(&energy, &resid, cfg))
            });
            let (px, py) = prewarp(&ux, &uy, &slope, &low, cfg);
            // the slope of the last solve is spent: the next solve allocates its own, and the
            // warp and the responses below are the memory peak of the iteration
            drop(std::mem::take(&mut slope));
            prof::time(Stage::Smooth, || (to_f32(&px), to_f32(&py)))
        };
        let warped = prof::time(Stage::Warp, || warp_shifted(spline2, &sx, &sy, shift));
        let responses = prof::time(Stage::Responses, || {
            Responses::new_in(&warped, bank, fft, std::mem::take(&mut spent))
        });
        drop(warped);
        let sm = prof::time(Stage::Smooth, || Smoothed {
            csx: BSpline::new(&sx),
            csy: BSpline::new(&sy),
            sx,
            sy,
        });
        // the buffers of the inner iterations, reused by each: the sums, and the displacement
        // the next solve writes while it reads the current one; the solve writes the energy,
        // the residual and the slope in place (it reads none of them), so no buffer is held
        // longer than before and none is reallocated within the outer iteration
        let (mut sums, mut next) = prof::time(Stage::Accumulate, || {
            let plane = || Image::from_vec(w, h, vec![0.0f64; w * h]);
            (Sums::new(w, h), (plane(), plane()))
        });
        for _ in 0..cfg.inner_iters {
            prof::time(Stage::Accumulate, || {
                accumulate_into(reference, &responses, &sm, &ux, &uy, &mut sums)
            });
            if slope.len() != w * h {
                slope = vec![[0.0f64; 4]; w * h];
            }
            let out = SolveOut {
                ux: &mut next.0.data,
                uy: &mut next.1.data,
                energy: &mut energy.data,
                resid: &mut resid.data,
                slope: &mut slope,
            };
            let ap = ApSolve {
                ap: aperture,
                tol2: cfg.phase_tol * cfg.phase_tol,
            };
            local_solve_into(&sums, &ux, &uy, cfg.sigma_w, cfg.shape_order, ap, out);
            std::mem::swap(&mut ux, &mut next.0);
            std::mem::swap(&mut uy, &mut next.1);
        }
        drop((sums, next, sm));
        spent = responses.data;
        if let Some(t) = t_it {
            let secs = t.elapsed().as_secs_f64();
            crate::iter_trace::outer(ux.data.clone(), uy.data.clone(), secs);
        }
    }
    (
        RawFlow {
            ux,
            uy,
            energy,
            resid,
        },
        slope,
    )
}

/// Scale of the texture factor of the confidence: a quarter of the median `energy` of the whole
/// image.
pub(crate) fn texture_scale(energy: &[f64]) -> f64 {
    0.25 * median(energy) + 1e-30
}

/// Width `B` of the strips along the borders in which the confidence is 0: `4 sigma_w` while
/// the edge guard is on, else 0 (module doc).
pub(crate) fn border_zone(cfg: &Config) -> f64 {
    if cfg.edge_guard_px() > 0.0 {
        4.0 * cfg.sigma_w
    } else {
        0.0
    }
}

/// Whether pixel `(x, y)` of a `w` x `h` first image, matched at `(tx, ty)` in the second image
/// of the same size, may carry confidence: at least `b` from the first image's border, and the
/// match at least `b` inside the second image's `[0, w - 1] x [0, h - 1]` (`b = 0`: not outside).
/// A NaN match fails every comparison, i.e. counts as outside.
pub(crate) fn reportable(x: usize, y: usize, tx: f64, ty: f64, w: usize, h: usize, b: f64) -> bool {
    let own = x.min(y).min(w - 1 - x).min(h - 1 - y) as f64;
    own >= b && tx >= b && ty >= b && (w - 1) as f64 - tx >= b && (h - 1) as f64 - ty >= b
}

/// Confidence of one pixel: a texture factor times `exp(-(resid / phase_tol)^2)`, clamped to
/// `[0, 1]`, and 0 where the pixel is not `reportable`. `f64`, like the reference's float64: the
/// stored `f32` is rounded from it and `conf < conf_min` is decided on it.
pub(crate) fn confidence(
    energy: f64,
    resid: f64,
    reportable: bool,
    scale: f64,
    phase_tol: f64,
) -> f64 {
    let ratio = energy / scale;
    let texture = if ratio.is_nan() {
        ratio
    } else {
        ratio.min(1.0)
    };
    let c = texture * (-(resid / phase_tol).powi(2)).exp();
    if reportable {
        c.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Dense sub-pixel displacement of `i2` relative to `i1`, refined from the initial field
/// `(ux, uy)` (`refine` of the reference). All four images must have the same size, at least
/// 1 x 1; the B-spline limitation of [`crate::interp`] applies to images narrower than about 20 px.
///
/// Per outer iteration the second image is warped with the Gaussian-smoothed field (width
/// `cfg.sigma_s`; from the second iteration on, smoothed from the displacement of the trusted pixels
/// only, see the module doc) and filtered; per inner iteration the responses are sampled at each pixel's
/// residual position and the local shape-function problem is solved. The confidence is a texture
/// factor times `exp(-(resid / phase_tol)^2)`, and 0 in the strips along the borders of the module
/// doc (within `4 sigma_w` of the first image's border, where `X + u` is within `4 sigma_w` of the
/// second image's border or leaves it).
///
/// # Panics
///
/// If an image is empty, or the four images differ in size. Also, for example, for a `cfg` that
/// [`Config::validate`] would reject, where the iteration first meets the invalid value:
/// a `shape_order` other than 1 or 2 (in the first local solve, after the first warp and filter
/// responses were computed), empty `scales` (the filter bank) or a `sigma_s` or `sigma_w` that is
/// not positive and finite (the Gaussian kernels). The pipeline validates the configuration
/// before it calls `refine`.
pub fn refine(
    i1: &Image<f32>,
    i2: &Image<f32>,
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
) -> Flow {
    refine_low(&CpuEngine, i1, i2, ux, uy, cfg)
        .expect("the CPU engine does not fail")
        .0
}

/// [`refine`], and `low`: whether the `f64` confidence is below `cfg.conf_min` (the reference
/// decides validity on its float64 confidence; the `f32` of `Flow::conf` is rounded from it).
pub(crate) fn refine_low(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
) -> Result<(Flow, Image<bool>), EngineError> {
    let (w, h) = (i1.w, i1.h);
    assert!(w >= 1 && h >= 1, "refine: empty image {w}x{h}");
    assert!(
        (i2.w, i2.h) == (w, h),
        "refine: i1 is {w}x{h} but i2 is {}x{}",
        i2.w,
        i2.h
    );
    let raw = engine.refine_raw(i1, i2, (0, 0), ux, uy, cfg)?;
    let t_conf = prof::start(Stage::Confidence);
    let scale = texture_scale(&raw.energy.data);
    let b = border_zone(cfg);
    let mut conf = vec![0.0f32; w * h];
    let mut low = vec![false; w * h];
    conf.par_chunks_mut(w)
        .zip(low.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (c_row, l_row))| {
            for x in 0..w {
                let i = y * w + x;
                let (tx, ty) = (x as f64 + raw.ux.data[i], y as f64 + raw.uy.data[i]);
                let c = confidence(
                    raw.energy.data[i],
                    raw.resid.data[i],
                    reportable(x, y, tx, ty, w, h, b),
                    scale,
                    cfg.phase_tol,
                );
                c_row[x] = c as f32;
                l_row[x] = c < cfg.conf_min;
            }
        });
    let flow = Flow {
        ux: to_f32(&raw.ux),
        uy: to_f32(&raw.uy),
        conf: Image::from_vec(w, h, conf),
        resid: to_f32(&raw.resid),
    };
    prof::add(Stage::Confidence, t_conf);
    let low = Image::from_vec(w, h, low);
    if crate::iter_trace::enabled() {
        let whole = crate::tile::Rect {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        };
        let traced = crate::iter_trace::take(whole, whole);
        crate::iter_trace::finish(traced.into_iter().collect(), &low);
    }
    Ok((flow, low))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::pseudo_random_image;
    use proptest::prelude::*;

    /// The bits of a float with every NaN mapped to one pattern, for bit-for-bit comparisons:
    /// Rust leaves the sign and payload of a NaN that arithmetic produces unspecified (Miri
    /// randomizes them), while every finite or infinite value keeps its exact bits.
    trait CanonBits {
        type Bits;
        fn canon_bits(self) -> Self::Bits;
    }

    impl CanonBits for f64 {
        type Bits = u64;
        fn canon_bits(self) -> u64 {
            if self.is_nan() { f64::NAN } else { self }.to_bits()
        }
    }

    impl CanonBits for f32 {
        type Bits = u32;
        fn canon_bits(self) -> u32 {
            if self.is_nan() { f32::NAN } else { self }.to_bits()
        }
    }

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// `A A^T + 0.05 I` for a random `N x N` matrix `A` (row-major `a`) with entries in `[-1, 1]`.
    fn spd_from<const N: usize>(a: &[f64]) -> [[f64; N]; N] {
        let mut m = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                m[i][j] = (0..N).map(|k| a[i * N + k] * a[j * N + k]).sum::<f64>()
                    + if i == j { 0.05 } else { 0.0 };
            }
        }
        m
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 64 }))]

        #[test]
        fn refine_cholesky_residual_is_tiny_for_spd_6x6(
            a in prop::collection::vec(-1.0f64..1.0, 36),
            b in prop::array::uniform6(-10.0f64..10.0),
        ) {
            check_cholesky::<6>(&a, &b)?;
        }

        #[test]
        fn refine_cholesky_residual_is_tiny_for_spd_12x12(
            a in prop::collection::vec(-1.0f64..1.0, 144),
            b in prop::array::uniform12(-10.0f64..10.0),
        ) {
            check_cholesky::<12>(&a, &b)?;
        }

        /// Lane independence, not correctness: the scalar `cholesky_solve` is the lane solve with
        /// one lane, so this shows that four systems in lockstep give, per lane, what each gives
        /// alone, including for lanes whose pivot fails (the residual proptests above check the
        /// factorisation itself).
        #[test]
        fn refine_cholesky_lanes_equal_one_system_at_a_time_bit_for_bit(
            a in prop::collection::vec(-1.0f64..1.0, 4 * 36),
            b in prop::collection::vec(-10.0f64..10.0, 4 * 6),
            shift in prop::array::uniform4(-1.0f64..0.5),
        ) {
            // A A^T + shift I: positive definite for a positive shift, often not otherwise
            let systems: Vec<([[f64; 6]; 6], [f64; 6])> = (0..4)
                .map(|l| {
                    let mut m = spd_from::<6>(&a[l * 36..(l + 1) * 36]);
                    for (i, row) in m.iter_mut().enumerate() {
                        row[i] += shift[l] - 0.05;
                    }
                    (m, std::array::from_fn(|i| b[l * 6 + i]))
                })
                .collect();
            let mut ml = [[[0.0f64; 4]; 6]; 6];
            let mut bl = [[0.0f64; 4]; 6];
            for (l, (m, rhs)) in systems.iter().enumerate() {
                for i in 0..6 {
                    for j in 0..6 {
                        ml[i][j][l] = m[i][j];
                    }
                    bl[i][l] = rhs[i];
                }
            }
            let ok = cholesky_solve_lanes(&mut ml, &mut bl);
            for (l, (m, rhs)) in systems.iter().enumerate() {
                let (mut m1, mut x) = (*m, *rhs);
                let ok1 = cholesky_solve(&mut m1, &mut x);
                prop_assert_eq!(ok[l], ok1);
                if ok1 {
                    for i in 0..6 {
                        prop_assert_eq!(bl[i][l].canon_bits(), x[i].canon_bits());
                    }
                }
            }
        }
    }

    fn check_cholesky<const N: usize>(a: &[f64], b: &[f64; N]) -> Result<(), TestCaseError> {
        let m = spd_from::<N>(a);
        let mut work = m;
        let mut x = *b;
        prop_assert!(cholesky_solve(&mut work, &mut x));
        let mut resid = [0.0; N];
        for i in 0..N {
            resid[i] = (0..N).map(|j| m[i][j] * x[j]).sum::<f64>() - b[i];
        }
        prop_assert!(
            norm(&resid) <= norm(b) * 1e-10,
            "|Mx - b| = {} for |b| = {}",
            norm(&resid),
            norm(b)
        );
        Ok(())
    }

    #[test]
    fn refine_cholesky_rejects_matrices_that_are_not_positive_definite() {
        assert!(!cholesky_solve(&mut [[0.0; 6]; 6], &mut [1.0; 6]));
        // indefinite: [[1, 2], [2, 1]] has eigenvalues 3 and -1
        assert!(!cholesky_solve(
            &mut [[1.0, 2.0], [2.0, 1.0]],
            &mut [1.0, 1.0]
        ));
        assert!(!cholesky_solve(
            &mut [[f64::NAN, 0.0], [0.0, 1.0]],
            &mut [1.0, 1.0]
        ));
        // only the lower triangle is read
        let mut b = [6.0, 5.0];
        assert!(cholesky_solve(&mut [[4.0, 99.0], [2.0, 3.0]], &mut b));
        assert!((b[0] - 1.0).abs() < 1e-14 && (b[1] - 1.0).abs() < 1e-14);
    }

    fn smooth_image(w: usize, h: usize) -> Image<f32> {
        let data = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                0.5 + 0.2 * (x * 0.31).sin() * (y * 0.23).cos() + 0.1 * (x * 0.07 + y * 0.11).sin()
            })
            .collect();
        Image::from_vec(w, h, data)
    }

    #[test]
    fn refine_warp_shifted_with_zero_shift_equals_interp_warp_bit_for_bit() {
        let (w, h) = (37usize, 29usize);
        let img = smooth_image(w, h);
        let ux = Image::from_vec(
            w,
            h,
            (0..w * h).map(|i| (i as f32 * 0.37).sin() * 3.1).collect(),
        );
        let uy = Image::from_vec(
            w,
            h,
            (0..w * h).map(|i| (i as f32 * 0.19).cos() * 2.3).collect(),
        );
        assert_eq!(
            warp_shifted(&BSpline::new(&img), &ux, &uy, (0, 0)).data,
            crate::interp::warp(&img, &ux, &uy).data
        );
    }

    #[test]
    fn refine_warp_shifted_window_matches_the_whole_image_lookup() {
        // i1 window of 24 x 20 at (30, 20); i2 window of 70 x 56 at (12, 8): shift = (18, 12)
        let full = smooth_image(120, 90);
        let (w, h) = (24usize, 20usize);
        let field = |x: usize, y: usize| (2.5 + 0.01 * x as f32, -1.5 + 0.02 * y as f32);
        let mut whole = (Image::new(120, 90), Image::new(120, 90));
        for y in 0..90 {
            for x in 0..120 {
                (whole.0.data[y * 120 + x], whole.1.data[y * 120 + x]) = field(x, y);
            }
        }
        let want = crate::interp::warp(&full, &whole.0, &whole.1);
        let crop = |im: &Image<f32>, x0: usize, y0: usize, cw: usize, ch: usize| {
            let mut d = Vec::new();
            for y in y0..y0 + ch {
                d.extend_from_slice(&im.data[y * im.w + x0..y * im.w + x0 + cw]);
            }
            Image::from_vec(cw, ch, d)
        };
        let got = warp_shifted(
            &BSpline::new(&crop(&full, 12, 8, 70, 56)),
            &crop(&whole.0, 30, 20, w, h),
            &crop(&whole.1, 30, 20, w, h),
            (18, 12),
        );
        for y in 0..h {
            for x in 0..w {
                let d = (got.data[y * w + x] - want.data[(y + 20) * 120 + x + 30]).abs();
                // only the prefilter near the window border differs, and that decays quickly
                assert!(d < 1e-5, "({x}, {y}): {d}");
            }
        }
    }

    #[test]
    fn refine_confidence_follows_the_formula() {
        // texture saturates at 1, the residual factor is a Gaussian of width phase_tol
        let c = confidence(2.0, 0.4, true, 1.0, 0.4);
        assert!((c - (-1.0f64).exp()).abs() < 1e-15);
        assert_eq!(confidence(2.0, 0.0, true, 1.0, 0.4), 1.0);
        assert_eq!(confidence(2.0, 0.0, false, 1.0, 0.4), 0.0);
        assert!((confidence(0.25, 0.0, true, 1.0, 0.4) - 0.25).abs() < 1e-15);
        assert!(confidence(f64::NAN, 0.0, true, 1.0, 0.4).is_nan());
    }

    #[test]
    fn refine_border_rule_follows_the_spec() {
        // the guard is on by default: B = 4 sigma_w = 16
        let b = border_zone(&Config::default());
        assert_eq!(b, 16.0);
        // the guard off switches the zone off
        let off = Config {
            edge_guard: 0.0,
            ..Config::default()
        };
        assert_eq!(border_zone(&off), 0.0);
        let wide = Config {
            sigma_w: 6.0,
            ..Config::default()
        };
        assert_eq!(border_zone(&wide), 24.0);
        // B = 0: only a match outside [0, w - 1] x [0, h - 1] is refused, as before
        assert!(reportable(0, 0, 0.0, 0.0, 10, 8, 0.0) && reportable(9, 7, 9.0, 7.0, 10, 8, 0.0));
        assert!(!reportable(0, 0, -1e-9, 0.0, 10, 8, 0.0));
        assert!(!reportable(0, 0, 9.0001, 7.0, 10, 8, 0.0));
        assert!(!reportable(0, 0, f64::NAN, 0.0, 10, 8, 0.0));
        assert!(!reportable(0, 0, 0.0, f64::INFINITY, 10, 8, 0.0));
        // B = 16 on a 100 x 80 image: the pixel itself must be 16 or more from its border ...
        let (w, h) = (100usize, 80usize);
        assert!(!reportable(15, 40, 40.0, 40.0, w, h, 16.0));
        assert!(reportable(16, 40, 40.0, 40.0, w, h, 16.0));
        assert!(!reportable(50, 15, 40.0, 40.0, w, h, 16.0));
        assert!(!reportable(w - 1 - 15, 40, 40.0, 40.0, w, h, 16.0));
        assert!(reportable(w - 1 - 16, 40, 40.0, 40.0, w, h, 16.0));
        assert!(!reportable(50, h - 1 - 15, 40.0, 40.0, w, h, 16.0));
        assert!(reportable(50, h - 1 - 16, 40.0, 40.0, w, h, 16.0));
        // ... and so must its match in the second image, on every side, and not be NaN
        assert!(reportable(50, 40, 16.0, 40.0, w, h, 16.0));
        assert!(!reportable(50, 40, 15.9, 40.0, w, h, 16.0));
        assert!(reportable(50, 40, 83.0, 40.0, w, h, 16.0));
        assert!(!reportable(50, 40, 83.1, 40.0, w, h, 16.0));
        assert!(!reportable(50, 40, 40.0, 15.9, w, h, 16.0));
        assert!(reportable(50, 40, 40.0, 63.0, w, h, 16.0));
        assert!(!reportable(50, 40, 40.0, 63.1, w, h, 16.0));
        assert!(!reportable(50, 40, f64::NAN, 40.0, w, h, 16.0));
        assert!(!reportable(50, 40, 40.0, f64::NAN, w, h, 16.0));
    }

    #[test]
    fn refine_sample_taps_equal_bspline_sample() {
        let (w, h) = (30usize, 25usize);
        let points = [
            (0.0, 0.0),
            (3.3, 4.7),
            (28.9, 24.0),
            (-2.0, 40.0),
            (15.5, 12.25),
        ];
        let a: Vec<f32> = (0..w * h).map(|i| (i as f32 * 0.23).sin()).collect();
        let b: Vec<f32> = (0..w * h).map(|i| (i as f32 * 0.31).cos()).collect();
        let sa = BSpline::new(&Image::from_vec(w, h, a));
        let sb = BSpline::new(&Image::from_vec(w, h, b));
        for (x, y) in points {
            let t = taps(x, y, w, h).unwrap();
            let (va, vb) = sample_taps2(&sa.coeffs, &sb.coeffs, &t);
            assert_eq!((va as f32, vb as f32), (sa.sample(x, y), sb.sample(x, y)));
        }
    }

    #[test]
    fn refine_accumulate_for_a_fixed_and_any_filter_count_agree_bit_for_bit() {
        let (w, h) = (40usize, 36usize);
        let i1 = smooth_image(w, h);
        let i2 = Image::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| i1.data[(i + 1) % (w * h)] * 0.9 + 0.05)
                .collect(),
        );
        let cfg = Config::default();
        let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
        let (bank, fft) = (gabor_bank(ph, pw, &cfg), Fft2::new(ph, pw));
        assert_eq!(bank.filters.len(), 12);
        let reference = RefPack::new(&i1, &bank, &fft);
        let responses = Responses::new(&i2, &bank, &fft);
        let field =
            |s: f32| Image::from_vec(w, h, (0..w * h).map(|i| s * (i % 7) as f32).collect());
        let (sx, sy) = (field(0.05), field(-0.03));
        let sm = Smoothed {
            csx: BSpline::new(&sx),
            csy: BSpline::new(&sy),
            sx,
            sy,
        };
        let (ux, uy) = (to_f64(&field(0.07)), to_f64(&field(-0.04)));
        // into sums holding other values: every value is written
        let mut fixed = Sums::new(w, h);
        fixed.gxy.data.fill(f64::NAN);
        let mut any = Sums::new(w, h);
        any.gcc.data.fill(7.0);
        accumulate_with::<12>(&reference, &responses, &sm, &ux, &uy, &mut fixed);
        accumulate_with::<0>(&reference, &responses, &sm, &ux, &uy, &mut any);
        let bits = |s: &Sums| -> Vec<u64> {
            [&s.gxx, &s.gxy, &s.gyy, &s.gxc, &s.gyc, &s.gcc, &s.gw]
                .iter()
                .flat_map(|im| im.data.iter().map(|v| v.canon_bits()))
                .collect()
        };
        assert_eq!(bits(&fixed), bits(&any));
        assert!(fixed.gw.data.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn refine_sample_all_equals_bspline_sample_per_filter() {
        let (w, h, f) = (30usize, 25usize, 5usize);
        let splines: Vec<BSpline<Complex32>> = (0..f)
            .map(|k| {
                let data: Vec<Complex32> = (0..w * h)
                    .map(|i| {
                        Complex32::new(
                            (i as f32 * 0.37 + k as f32).sin(),
                            (i as f32 * 0.11 - k as f32).cos(),
                        )
                    })
                    .collect();
                BSpline::new(&Image::from_vec(w, h, data))
            })
            .collect();
        let mut data = vec![Complex32::default(); w * h * f];
        for i in 0..w * h {
            for (k, s) in splines.iter().enumerate() {
                data[i * f + k] = s.coeffs.data[i];
            }
        }
        let resp = Responses {
            w,
            n_filters: f,
            data,
        };
        let (mut scratch, mut out) = (vec![[0.0; 2]; f], vec![[0.0; 2]; f]);
        for (x, y) in [
            (0.0, 0.0),
            (3.3, 4.7),
            (28.9, 24.0),
            (-2.0, 40.0),
            (15.5, 12.25),
        ] {
            let t = taps(x, y, w, h).unwrap();
            resp.sample_all(f, &t, &mut scratch, &mut out);
            for (k, s) in splines.iter().enumerate() {
                let want = s.sample(x, y);
                assert_eq!((out[k][0] as f32, out[k][1] as f32), (want.re, want.im));
            }
        }
    }

    /// A config with 6 filters (3 orientations at 2 scales: one group of 4 filters and one of 2),
    /// 2 under Miri, where the transforms are slow.
    fn bank_config() -> Config {
        if cfg!(miri) {
            Config {
                scales: vec![1.0],
                n_orient: 2,
                ..Config::default()
            }
        } else {
            Config {
                scales: vec![0.6, 1.0],
                n_orient: 3,
                ..Config::default()
            }
        }
    }

    /// A noisy `f32` image in `[0, 1)`.
    fn noisy_image(w: usize, h: usize, seed: u64) -> Image<f32> {
        let noise = pseudo_random_image(w, h, seed);
        Image::from_vec(
            w,
            h,
            noise.data.iter().map(|&v| (0.5 + 0.5 * v) as f32).collect(),
        )
    }

    /// `Responses::new` is, for every filter, `BSpline::new` of the inverse transform of the
    /// filtered spectrum, bit for bit, at a padded size (87 x 82) off the 8-line grid of the
    /// blocked transforms and with a filter count off the group of 4.
    #[test]
    fn refine_responses_equal_the_bspline_of_the_inverse_transform_bit_for_bit() {
        let (w, h) = (18usize, 23usize);
        let img = noisy_image(w, h, 5);
        let cfg = bank_config();
        let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
        assert!(ph % crate::fft::LINES != 0 && pw % crate::fft::LINES != 0);
        let (bank, fft) = (gabor_bank(ph, pw, &cfg), Fft2::new(ph, pw));
        let f = bank.filters.len();
        let got = Responses::new(&img, &bank, &fft);
        assert_eq!((got.w, got.n_filters, got.data.len()), (pw, f, pw * ph * f));
        let spectrum = padded_spectrum(&img, &fft);
        for (k, g) in bank.filters.iter().enumerate() {
            let mut r: Vec<Complex32> = spectrum.iter().zip(g).map(|(&s, &g)| s * g).collect();
            fft.inverse(&mut r);
            let want = BSpline::new(&Image::from_vec(pw, ph, r)).coeffs;
            for (i, want) in want.data.iter().enumerate() {
                let have = got.data[i * f + k];
                assert_eq!(
                    (have.re.canon_bits(), have.im.canon_bits()),
                    (want.re.canon_bits(), want.im.canon_bits()),
                    "filter {k} node {i}"
                );
            }
        }
    }

    /// `RefPack::new` packs, per pixel and filter, `[r.re, r.im, gx, gy, amp]` of
    /// `gabor::reference_phase`, bit for bit, at the same off-grid sizes.
    #[test]
    fn refine_refpack_equals_reference_phase_bit_for_bit() {
        let (w, h) = (18usize, 23usize);
        let img = noisy_image(w, h, 6);
        let cfg = bank_config();
        let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
        let (bank, fft) = (gabor_bank(ph, pw, &cfg), Fft2::new(ph, pw));
        let f = bank.filters.len();
        let got = RefPack::new(&img, &bank, &fft);
        assert_eq!((got.n_filters, got.data.len()), (f, w * h * f * 5));
        let want = crate::gabor::reference_phase(&img, &bank, &fft);
        assert_eq!(want.len(), f);
        for (k, rp) in want.iter().enumerate() {
            for i in 0..w * h {
                let have = &got.data[(i * f + k) * 5..][..5];
                let want = [
                    rp.r.data[i].re,
                    rp.r.data[i].im,
                    rp.gx.data[i],
                    rp.gy.data[i],
                    rp.amp.data[i],
                ];
                let bits = |v: &[f32]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
                assert_eq!(bits(have), bits(&want), "filter {k} pixel {i}");
            }
        }
    }

    #[test]
    fn refine_moment_rows_equal_window_moment_bit_for_bit() {
        // narrower than the window in one direction, wider in the other, and rows with and
        // without a partial block of 8 (fewer under Miri, where it is slow)
        let sizes: &[(usize, usize)] = if cfg!(miri) {
            &[(7, 9)]
        } else {
            &[(9, 37), (31, 6), (1, 5), (7, 1), (45, 20)]
        };
        for &(w, h) in sizes {
            let [gxx, gxy, gyy, gxc, gyc, gcc, gw] =
                [1u64, 2, 3, 4, 5, 6, 7].map(|seed| pseudo_random_image(w, h, seed));
            let sums = Sums {
                gxx,
                gxy,
                gyy,
                gxc,
                gyc,
                gcc,
                gw,
            };
            for order in [1, 2] {
                let plan = plan(order);
                let rows = MomentRows::new(&plan, 1.5);
                assert_eq!(rows.n_moments, plan.moments.len());
                let mut pad = vec![0.0; rows.pad_len(w)];
                let mut m = vec![0.0; rows.n_moments * w];
                let mut maps = vec![vec![0.0; w * h]; rows.n_moments];
                for y in 0..h {
                    rows.row(&sums, y, &mut pad, &mut m);
                    for (k, map) in maps.iter_mut().enumerate() {
                        map[y * w..(y + 1) * w].copy_from_slice(&m[k * w..(k + 1) * w]);
                    }
                }
                for (&(src, (px, py)), got) in plan.moments.iter().zip(&maps) {
                    let want = window_moment(sums.source(src), 1.5, px, py);
                    if cfg!(miri) {
                        // Miri perturbs `exp` (in `gauss_kernel`) differently on every call
                        let worst = got.iter().zip(&want.data);
                        let worst = worst.map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                        assert!(worst < 1e-12, "{w}x{h} order {order} ({px}, {py}): {worst}");
                    } else {
                        assert_eq!(got, &want.data, "{w}x{h} order {order} ({px}, {py})");
                    }
                }
            }
        }
    }

    /// Sums of four unit-gradient "filters" whose constraints `g . U(q) = c` are exact for the
    /// field `U(q) = (ux(q), uy(q))`, built per pixel like `accumulate` does.
    fn sums_for_field(w: usize, h: usize, field: impl Fn(f64, f64) -> (f64, f64)) -> Sums {
        sums_for_field_with(w, h, field, |_, _| (4, 1.0))
    }

    /// [`sums_for_field`] with `pixel(x, y) = (n, scale)`: only the first `n` of the four filters
    /// contribute at that pixel (one filter leaves a single gradient direction), all scaled by
    /// `scale`.
    fn sums_for_field_with(
        w: usize,
        h: usize,
        field: impl Fn(f64, f64) -> (f64, f64),
        pixel: impl Fn(usize, usize) -> (usize, f64),
    ) -> Sums {
        let mut s = [
            vec![0.0; w * h],
            vec![0.0; w * h],
            vec![0.0; w * h],
            vec![0.0; w * h],
            vec![0.0; w * h],
            vec![0.0; w * h],
            vec![0.0; w * h],
        ];
        for y in 0..h {
            for x in 0..w {
                let (qx, qy) = (x as f64, y as f64);
                let (u, v) = field(qx, qy);
                let (n_filters, scale) = pixel(x, y);
                for k in 0..n_filters {
                    let th = PI * k as f64 / 4.0;
                    let (gx, gy) = (th.cos(), th.sin());
                    let wk = scale * (1.0 + 0.5 * (0.3 * qx + 0.2 * qy + k as f64).sin());
                    let c = gx * u + gy * v;
                    let i = y * w + x;
                    s[0][i] += wk * gx * gx;
                    s[1][i] += wk * gx * gy;
                    s[2][i] += wk * gy * gy;
                    s[3][i] += wk * gx * c;
                    s[4][i] += wk * gy * c;
                    s[5][i] += wk * c * c;
                    s[6][i] += wk;
                }
            }
        }
        let [gxx, gxy, gyy, gxc, gyc, gcc, gw] = s.map(|v| Image::from_vec(w, h, v));
        Sums {
            gxx,
            gxy,
            gyy,
            gxc,
            gyc,
            gcc,
            gw,
        }
    }

    #[test]
    fn refine_local_solve_recovers_polynomial_fields_in_the_interior() {
        // smaller under Miri, where the whole local solve is slow
        let (w, h, sigma) = if cfg!(miri) {
            (26usize, 22usize, 1.0)
        } else {
            (48, 40, 2.0)
        };
        let zero = Image::from_vec(w, h, vec![0.0; w * h]);
        let affine = |x: f64, y: f64| (0.5 + 0.02 * x - 0.01 * y, -0.3 + 0.015 * x + 0.03 * y);
        let quad = |x: f64, y: f64| {
            (
                0.5 + 0.02 * x - 0.01 * y + 0.002 * x * x - 0.001 * x * y,
                -0.3 + 0.015 * x + 0.03 * y + 0.0015 * y * y + 0.001 * x * y - 0.0005 * x * x,
            )
        };
        let r = (4.0 * sigma + 0.5) as usize; // window radius: pixels this far from the border see no replicated edge
        for (order, field) in [
            (1u32, &affine as &dyn Fn(f64, f64) -> (f64, f64)),
            (2, &quad),
        ] {
            let sums = sums_for_field(w, h, field);
            let sol = local_solve(&sums, &zero, &zero, sigma, order);
            for y in r..h - r {
                for x in r..w - r {
                    let i = y * w + x;
                    let (u, v) = field(x as f64, y as f64);
                    assert!(
                        (sol.ux.data[i] - u).abs() < 1e-7 && (sol.uy.data[i] - v).abs() < 1e-7,
                        "order {order} ({x}, {y}): got ({}, {}) want ({u}, {v})",
                        sol.ux.data[i],
                        sol.uy.data[i]
                    );
                    // exact fit up to the ridge (relative 1e-12 of the mean diagonal)
                    assert!(sol.resid.data[i] < 1e-4, "resid {}", sol.resid.data[i]);
                }
            }
        }
        // an affine model cannot fit a quadratic field exactly
        let sums = sums_for_field(w, h, quad);
        let sol = local_solve(&sums, &zero, &zero, sigma, 1);
        let misfit = sol.resid.data[(h / 2) * w + w / 2];
        assert!(misfit > 1e-3, "misfit {misfit}");
    }

    #[test]
    fn refine_local_solve_rejects_a_one_dimensional_aperture() {
        // gradients almost purely along x: positive definite, but det2 = 1e-6 < 1e-3 energy^2
        let (w, h) = (16usize, 16usize);
        let constant = |v: f64| Image::from_vec(w, h, vec![v; w * h]);
        let sums = |gyy: f64| Sums {
            gxx: constant(1.0),
            gxy: constant(0.0),
            gyy: constant(gyy),
            gxc: constant(0.3),
            gyc: constant(0.1),
            gcc: constant(0.2),
            gw: constant(1.0),
        };
        let (ux, uy) = (constant(1.25), constant(-0.75));
        let thin = local_solve(&sums(1e-6), &ux, &uy, 2.0, 1);
        assert!(thin.ux.data.iter().all(|&v| v == 1.25));
        assert!(thin.uy.data.iter().all(|&v| v == -0.75));
        assert!(thin.resid.data.iter().all(|&v| v == PI));
        assert!(thin.slope.iter().all(|&g| g == [0.0; 4]));
        // the same system with gradients along both axes is solved
        let both = local_solve(&sums(1.0), &ux, &uy, 2.0, 1);
        assert!(both.resid.data.iter().all(|&v| v < PI));
        assert!((both.ux.data[8 * w + 8] - 0.3).abs() < 1e-9);
        assert!((both.uy.data[8 * w + 8] - 0.1).abs() < 1e-9);
        // constant data: the fitted slope vanishes (up to rounding)
        let centre = both.slope[8 * w + 8];
        assert!(centre.iter().all(|g| g.abs() < 1e-9), "{centre:?}");
    }

    /// The [`Aperture`] prototypes on a window of constant sums with gradients mostly along x
    /// (`q = gyy / (1 + gyy)`): off (also with a pre-warp given) the solve is the plain one bit for
    /// bit; the confidence factor `min(1, q / q0)` goes into the residual and leaves the rest; the
    /// ridge pulls the weak component (y) towards the input by `(a e - l_min) / (a e)`, keeps the
    /// residual the misfit of the data, and leaves a window with `q >= a` as it was.
    #[test]
    fn refine_aperture_prototypes_on_a_line_like_window() {
        let (w, h) = (16usize, 16usize);
        let constant = |v: f64| Image::from_vec(w, h, vec![v; w * h]);
        let (gxc, gyc, gcc) = (0.3, 0.1, 2.0);
        let sums = |gyy: f64| Sums {
            gxx: constant(1.0),
            gxy: constant(0.0),
            gyy: constant(gyy),
            gxc: constant(gxc),
            gyc: constant(gyc),
            gcc: constant(gcc),
            gw: constant(1.0),
        };
        let (ux, uy) = (constant(1.25), constant(-0.75));
        let tol2 = 0.4 * 0.4;
        let solve = |gyy: f64, ap: Aperture| {
            let plane = || vec![0.0f64; w * h];
            let (mut oux, mut ouy, mut ores, mut oen) = (plane(), plane(), plane(), plane());
            let mut oslope = vec![[0.0f64; 4]; w * h];
            let out = SolveOut {
                ux: &mut oux,
                uy: &mut ouy,
                energy: &mut oen,
                resid: &mut ores,
                slope: &mut oslope,
            };
            let aps = ApSolve { ap, tol2 };
            local_solve_into(&sums(gyy), &ux, &uy, 2.0, 1, aps, out);
            (oux, ouy, ores, oslope)
        };
        let bits = |v: &[f64]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
        let c = 8 * w + 8;
        for gyy in [0.01, 1.0] {
            let plain = local_solve(&sums(gyy), &ux, &uy, 2.0, 1);
            let off = solve(gyy, Aperture::default());
            assert_eq!(bits(&off.0), bits(&plain.ux.data));
            assert_eq!(bits(&off.1), bits(&plain.uy.data));
            assert_eq!(bits(&off.2), bits(&plain.resid.data));
            assert_eq!(bits(&off.3.concat()), bits(&plain.slope.concat()));
            // the confidence factor
            let (q, q0) = (gyy / (1.0 + gyy), 0.05);
            let conf = solve(
                gyy,
                Aperture {
                    conf: q0,
                    ridge: 0.0,
                },
            );
            assert_eq!(bits(&conf.0), bits(&plain.ux.data));
            assert_eq!(bits(&conf.1), bits(&plain.uy.data));
            let factor = |r: f64| (-r * r / tol2).exp();
            let (r0, r1) = (plain.resid.data[c], conf.2[c]);
            let want = factor(r0) * (q / q0).min(1.0);
            assert!(
                (factor(r1) / want - 1.0).abs() < 1e-9,
                "gyy {gyy}: {} vs {want}",
                factor(r1)
            );
            // the ridge
            let a = 0.05;
            let ridged = solve(
                gyy,
                Aperture {
                    conf: 0.0,
                    ridge: a,
                },
            );
            if q >= a {
                assert_eq!(bits(&ridged.0), bits(&plain.ux.data));
                assert_eq!(bits(&ridged.1), bits(&plain.uy.data));
                assert_eq!(bits(&ridged.2), bits(&plain.resid.data));
                continue;
            }
            // per unit window weight: l_min = gyy, e = 1 + gyy, the lift a e - gyy along y
            let lift = a * (1.0 + gyy) - gyy;
            let uy_want = (gyc + lift * -0.75) / (gyy + lift);
            assert!((ridged.0[c] - gxc).abs() < 1e-9, "ux {}", ridged.0[c]);
            assert!(
                (ridged.1[c] - uy_want).abs() < 1e-9,
                "uy {} want {uy_want}",
                ridged.1[c]
            );
            assert!((plain.uy.data[c] - gyc / gyy).abs() < 1e-6);
            // the residual is the data misfit at the ridged solution
            let (u, v) = (ridged.0[c], ridged.1[c]);
            let misfit = gcc - 2.0 * (u * gxc + v * gyc) + u * u + v * v * gyy;
            assert!(
                (ridged.2[c] - misfit.sqrt()).abs() < 1e-9,
                "{} vs {}",
                ridged.2[c],
                misfit.sqrt()
            );
        }
    }

    /// The solve into buffers that hold other values (as the iterations reuse them) writes the
    /// values of the solve into fresh ones, bit for bit, and so do the responses into the storage
    /// of earlier ones.
    #[test]
    fn refine_solve_and_responses_into_reused_buffers_equal_fresh_ones_bit_for_bit() {
        let (w, h) = (23usize, 19usize);
        let rnd = |seed: u64| pseudo_random_image(w, h, seed);
        let sq = |im: Image<f64>| Image::from_vec(w, h, im.data.iter().map(|v| v * v).collect());
        let sums = Sums {
            gxx: sq(rnd(1)),
            gxy: rnd(2),
            gyy: sq(rnd(3)),
            gxc: rnd(4),
            gyc: rnd(5),
            gcc: sq(rnd(6)),
            gw: sq(rnd(7)),
        };
        let (ux, uy) = (rnd(8), rnd(9));
        for order in [1u32, 2] {
            let want = local_solve(&sums, &ux, &uy, 1.5, order);
            let mut got = (rnd(10), rnd(11), rnd(12), rnd(13));
            let mut slope = vec![[f64::NAN; 4]; w * h];
            let out = SolveOut {
                ux: &mut got.0.data,
                uy: &mut got.1.data,
                energy: &mut got.2.data,
                resid: &mut got.3.data,
                slope: &mut slope,
            };
            local_solve_into(&sums, &ux, &uy, 1.5, order, ApSolve::default(), out);
            let bits = |v: &[f64]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got.0.data), bits(&want.ux.data), "order {order}");
            assert_eq!(bits(&got.1.data), bits(&want.uy.data));
            assert_eq!(bits(&got.2.data), bits(&want.energy.data));
            assert_eq!(bits(&got.3.data), bits(&want.resid.data));
            let flat = |v: &[[f64; 4]]| bits(&v.concat());
            assert_eq!(flat(&slope), flat(&want.slope));
        }
        let cfg = Config::default();
        let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
        let (bank, fft) = (gabor_bank(ph, pw, &cfg), Fft2::new(ph, pw));
        let img = Image::from_vec(w, h, rnd(14).data.iter().map(|&v| v as f32).collect());
        let want = Responses::new(&img, &bank, &fft);
        let old = vec![Complex32::new(f32::NAN, 3.0); want.data.len()];
        let got = Responses::new_in(&img, &bank, &fft, old);
        let bits = |v: &[Complex32]| {
            v.iter()
                .map(|c| (c.re.canon_bits(), c.im.canon_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&got.data), bits(&want.data));
    }

    /// Fixed points in lockstep give, per pixel, the position of each pixel alone, bit for bit,
    /// also where a sample leaves the image (NaN or clamped).
    #[test]
    fn refine_fixed_points_in_lockstep_equal_one_pixel_at_a_time_bit_for_bit() {
        let (w, h) = (31usize, 23usize);
        let f = |seed: u64, s: f64| {
            let im = pseudo_random_image(w, h, seed);
            Image::from_vec(w, h, im.data.iter().map(|&v| (s * v) as f32).collect())
        };
        let (sx, sy) = (f(1, 3.0), f(2, -2.0));
        let sm = Smoothed {
            csx: BSpline::new(&sx),
            csy: BSpline::new(&sy),
            sx,
            sy,
        };
        let (ux, uy) = (pseudo_random_image(w, h, 3), pseudo_random_image(w, h, 4));
        for y in [0, 7, h - 1] {
            for x0 in [0, 9, w - 4] {
                let px: [(usize, usize, (f64, f64)); 4] = std::array::from_fn(|l| {
                    let (x, i) = (x0 + l, y * w + x0 + l);
                    // one far outside the image (clamped samples), one NaN
                    let uv = match l {
                        1 => (40.0 * ux.data[i], uy.data[i]),
                        2 if y == 7 => (f64::NAN, 0.0),
                        _ => (ux.data[i], uy.data[i]),
                    };
                    (x, i, uv)
                });
                let lanes = fixed_points(&sm, y, px);
                for (l, &p) in px.iter().enumerate() {
                    let [one] = fixed_points(&sm, y, [p]);
                    assert_eq!(
                        (lanes[l].0.canon_bits(), lanes[l].1.canon_bits()),
                        (one.0.canon_bits(), one.1.canon_bits()),
                        "({}, {y}) lane {l}",
                        p.0
                    );
                }
            }
        }
    }

    /// `accumulate` with its rows in groups of `FIXED_POINT_LANES` equals the sums formed one
    /// pixel at a time, bit for bit, at widths whose last group is partial (the lanes past the
    /// end of the row repeat its last pixel and are not used), also where a pixel's position is
    /// NaN (its sums are NaN) and for any filter count.
    #[test]
    fn refine_accumulate_in_pixel_groups_equals_one_pixel_at_a_time_bit_for_bit() {
        for w in [37usize, 38, 39, 40] {
            let h = 9usize;
            let i1 = smooth_image(w, h);
            let i2 = Image::from_vec(
                w,
                h,
                (0..w * h)
                    .map(|i| i1.data[(i + 1) % (w * h)] * 0.9 + 0.05)
                    .collect(),
            );
            let cfg = Config::default();
            let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
            let (bank, fft) = (gabor_bank(ph, pw, &cfg), Fft2::new(ph, pw));
            let reference = RefPack::new(&i1, &bank, &fft);
            let responses = Responses::new(&i2, &bank, &fft);
            let field =
                |s: f32| Image::from_vec(w, h, (0..w * h).map(|i| s * (i % 7) as f32).collect());
            let (sx, sy) = (field(0.05), field(-0.03));
            let sm = Smoothed {
                csx: BSpline::new(&sx),
                csy: BSpline::new(&sy),
                sx,
                sy,
            };
            let (mut ux, uy) = (to_f64(&field(0.07)), to_f64(&field(-0.04)));
            // the last pixel of row 3 (in the partial group for w % 4 != 0) has no position
            ux.data[3 * w + w - 1] = f64::NAN;
            let n = bank.filters.len();
            let mut heap = (
                vec![[0.0f64; 2]; n],
                vec![[0.0f64; 2]; n],
                vec![0.0f64; 4 * n],
            );
            let (num, rest) = heap.2.split_at_mut(n);
            let (den, rest) = rest.split_at_mut(n);
            let (mag, dphi) = rest.split_at_mut(n);
            let mut scratch = PixelScratch {
                acc: &mut heap.0,
                sampled: &mut heap.1,
                num,
                den,
                mag,
                dphi,
            };
            let mut want = vec![[0.0f64; 7]; w * h];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let uv = (ux.data[i], uy.data[i]);
                    let [q] = fixed_points(&sm, y, [(x, i, uv)]);
                    let rp = &reference.data[i * n * 5..(i + 1) * n * 5];
                    want[i] = pixel_sums_at(n, &responses, &sm, q, uv, rp, &mut scratch);
                }
            }
            let mut fixed = Sums::new(w, h);
            accumulate_with::<12>(&reference, &responses, &sm, &ux, &uy, &mut fixed);
            let mut any = Sums::new(w, h);
            accumulate_with::<0>(&reference, &responses, &sm, &ux, &uy, &mut any);
            for got in [&fixed, &any] {
                let planes = [
                    &got.gxx, &got.gxy, &got.gyy, &got.gxc, &got.gyc, &got.gcc, &got.gw,
                ];
                for (k, plane) in planes.iter().enumerate() {
                    for (i, (v, want)) in plane.data.iter().zip(&want).enumerate() {
                        assert_eq!(
                            v.canon_bits(),
                            want[k].canon_bits(),
                            "w {w}: sum {k} at pixel {i}"
                        );
                    }
                }
            }
            assert!(fixed.gw.data[3 * w + w - 1].is_nan());
        }
    }

    /// The threshold's shortcut (no median when even the smallest energy is above
    /// `1e-9 max + 1e-30`) and the full path agree with the per-pixel reference: energies within
    /// a factor of a billion (the shortcut), one far below (reset), one NaN (the median is NaN and
    /// every pixel is reset).
    #[test]
    fn refine_local_solve_threshold_shortcut_and_full_path_equal_the_reference() {
        let (w, h) = (11usize, 7usize);
        let field = |x: f64, y: f64| (0.2 + 0.01 * x, -0.1 + 0.02 * y);
        let (ux, uy) = (pseudo_random_image(w, h, 31), pseudo_random_image(w, h, 32));
        let bits = |v: &[f64]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
        for weak in [1.0, 1e-13, f64::NAN] {
            let pixel = |x: usize, y: usize| {
                if (x, y) == (5, 3) {
                    (4, weak)
                } else {
                    (4, 1.0)
                }
            };
            let sums = sums_for_field_with(w, h, field, pixel);
            let got = local_solve(&sums, &ux, &uy, 0.13, 1);
            let want = local_solve_per_pixel(&sums, &ux, &uy, 0.13, 1);
            assert_eq!(bits(&got.ux.data), bits(&want.ux.data), "weak {weak}");
            assert_eq!(bits(&got.resid.data), bits(&want.resid.data), "weak {weak}");
            let reset = got.resid.data.iter().filter(|&&r| r == PI).count();
            let expect = if weak.is_nan() {
                w * h
            } else if weak < 1e-9 {
                1
            } else {
                0
            };
            assert_eq!(reset, expect, "weak {weak}");
        }
    }

    #[test]
    fn refine_local_solve_keeps_the_input_where_the_system_is_degenerate() {
        let (w, h) = (16usize, 16usize);
        let flat = Image::from_vec(w, h, vec![0.0; w * h]);
        let sums = Sums {
            gxx: flat.clone(),
            gxy: flat.clone(),
            gyy: flat.clone(),
            gxc: flat.clone(),
            gyc: flat.clone(),
            gcc: flat.clone(),
            gw: flat.clone(),
        };
        let ux = Image::from_vec(w, h, vec![1.25; w * h]);
        let uy = Image::from_vec(w, h, vec![-0.75; w * h]);
        for order in [1, 2] {
            let sol = local_solve(&sums, &ux, &uy, 2.0, order);
            assert!(sol.ux.data.iter().all(|&v| v == 1.25));
            assert!(sol.uy.data.iter().all(|&v| v == -0.75));
            assert!(sol.resid.data.iter().all(|&v| v == PI));
            assert_eq!(sol.slope.len(), w * h);
            assert!(sol.slope.iter().all(|&g| g == [0.0; 4]), "order {order}");
        }
    }

    /// The slope output is the affine part of the fitted shape function at the window centre, in the
    /// order `(dux/dx, duy/dx, dux/dy, duy/dy)` that [`prewarp`] reads, for a sheared, a rotated and a
    /// general field (four distinct coefficients: an exchanged pair is caught); for order 2 a
    /// quadratic term makes the gradient vary over the image and the slope is the one at the pixel
    /// itself (Python: `test_local_solve_slope_is_dux_dx_duy_dx_dux_dy_duy_dy`).
    #[test]
    fn refine_local_solve_slope_is_dux_dx_duy_dx_dux_dy_duy_dy() {
        // a width that is not a multiple of SOLVE_LANES (8): the last lanes of each row are padding
        let (w, h, sigma) = if cfg!(miri) {
            (26usize, 22usize, 1.0)
        } else {
            (50, 40, 2.0)
        };
        assert_ne!(w % SOLVE_LANES, 0);
        let zero = Image::from_vec(w, h, vec![0.0; w * h]);
        let r = (4.0 * sigma + 0.5) as usize;
        for (name, [a1, b1, a2, b2]) in [
            ("shear", [0.0, 0.0, 0.03, 0.0]),
            ("rotation", [0.0, 0.02, -0.02, 0.0]),
            ("general", [0.01, 0.03, -0.02, 0.005]),
        ] {
            for (order, q) in [(1u32, 0.0), (2, 4e-4)] {
                let field = |x: f64, y: f64| {
                    (
                        2.0 + a1 * x + a2 * y + q * x * x,
                        -1.5 + b1 * x + b2 * y + q * x * y,
                    )
                };
                let sums = sums_for_field(w, h, field);
                let sol = local_solve(&sums, &zero, &zero, sigma, order);
                assert_eq!(sol.slope.len(), w * h);
                for y in r..h - r {
                    for x in r..w - r {
                        let (px, py) = (x as f64, y as f64);
                        // [dux/dx, duy/dx, dux/dy, duy/dy] of the field at the pixel
                        let want = [a1 + 2.0 * q * px, b1 + q * py, a2, b2 + q * px];
                        let got = sol.slope[y * w + x];
                        for (g, t) in got.iter().zip(want) {
                            assert!(
                                (g - t).abs() < 1e-7,
                                "{name} order {order} ({x}, {y}): got {got:?} want {want:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The system of one pixel, as `local_solve` states it, solved with the scalar
    /// `cholesky_solve`: the solution `theta` and the residual, or `None` where the Cholesky
    /// factorisation fails.
    fn solve_pixel<const N: usize>(
        plan: &Plan,
        at: &dyn Fn(usize) -> f64,
    ) -> Option<([f64; N], f64)> {
        let mut m = [[0.0f64; N]; N];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate().take(i + 1) {
                *v = at(plan.m_idx[i * N + j]);
            }
        }
        let ridge = RIDGE * (0..N).map(|i| m[i][i]).sum::<f64>() / N as f64;
        for (i, row) in m.iter_mut().enumerate() {
            row[i] += ridge;
        }
        let b: [f64; N] = std::array::from_fn(|i| at(plan.b_idx[i]));
        let mut theta = b;
        if !cholesky_solve(&mut m, &mut theta) {
            return None;
        }
        let mut fit = 0.0;
        for (t, bi) in theta.iter().zip(&b) {
            fit = madd(*t, *bi, fit);
        }
        let resid = (max0(at(plan.gcc_idx) - fit) / (at(plan.gw_idx) + 1e-30)).sqrt();
        Some((theta, resid))
    }

    /// `local_solve` one pixel at a time: the whole-image moment maps of `window_moment`, the
    /// aperture rule, the system of each pixel assembled and solved alone by `solve_pixel`, then
    /// the energy threshold.
    fn local_solve_per_pixel(
        sums: &Sums,
        ux: &Image<f64>,
        uy: &Image<f64>,
        sigma: f64,
        order: u32,
    ) -> Solved {
        let plan = plan(order);
        let maps: Vec<Image<f64>> = plan
            .moments
            .iter()
            .map(|&(src, (px, py))| window_moment(sums.source(src), sigma, px, py))
            .collect();
        let (w, h, n) = (ux.w, ux.h, plan.n);
        let mut out = Solved {
            ux: ux.clone(),
            uy: uy.clone(),
            energy: Image::new(w, h),
            resid: Image::from_vec(w, h, vec![PI; w * h]),
            slope: vec![[0.0; 4]; w * h],
        };
        for p in 0..w * h {
            let at = |idx: usize| maps[idx].data[p];
            let (m00, m11, m01) = (at(plan.m_idx[0]), at(plan.m_idx[n + 1]), at(plan.m_idx[n]));
            let energy = m00 + m11;
            out.energy.data[p] = energy;
            let wanted = m00 * m11 - m01 * m01 > APERTURE_TOL * energy * energy;
            if !wanted {
                continue;
            }
            let solved: Option<(Vec<f64>, f64)> = match n {
                6 => solve_pixel::<6>(&plan, &at).map(|(t, r)| (t.to_vec(), r)),
                12 => solve_pixel::<12>(&plan, &at).map(|(t, r)| (t.to_vec(), r)),
                n => unreachable!("shape function with {n} unknowns"),
            };
            if let Some((t, resid)) = solved {
                out.ux.data[p] = t[0];
                out.uy.data[p] = t[1];
                out.resid.data[p] = resid;
                out.slope[p] = [t[2], t[3], t[4], t[5]];
            }
        }
        let threshold = 1e-9 * median(&out.energy.data) + 1e-30;
        for p in 0..w * h {
            let textured = out.energy.data[p] > threshold; // false for NaN
            if !textured {
                (out.ux.data[p], out.uy.data[p], out.resid.data[p]) = (ux.data[p], uy.data[p], PI);
                out.slope[p] = [0.0; 4];
            }
        }
        out
    }

    /// `local_solve` equals the per-pixel solve over `window_moment` bit for bit, for rows of
    /// 1 and 3 pixels (a single tail group), 5 and 7 (a full group, then a tail group), 8 (two
    /// full groups, no tail) and 50 (full groups, then a tail group of 2); in a tail group the
    /// lanes past the row's end repeat the last pixel and are dropped. The sums mix pixels that are solved,
    /// pixels with one gradient direction only (the aperture rule) and pixels with a weight of
    /// 1e-12, whose energy is far below the threshold of the whole image (`1e-9 median(energy)`);
    /// with the narrow window (`sigma` 0.13, radius 1, the neighbours weigh 1e-13) every pixel
    /// keeps its own fate (asserted: solved exactly where nothing rejects it, `energy` at most the
    /// threshold exactly for the light pixels), so the lanes of a group disagree on whether they
    /// are solved (asserted for the first group of 4 of every row with 4 pixels or more, which
    /// holds a pixel rejected by the aperture rule, so that the lanes disagree without the energy
    /// reset too).
    #[test]
    #[cfg_attr(miri, ignore)] // Miri perturbs `exp` in `gauss_kernel` differently on every call
    fn refine_local_solve_equals_the_per_pixel_solve_bit_for_bit() {
        let bits = |v: &[f64]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
        let slope_bits = |s: &Solved| bits(&s.slope.iter().flatten().copied().collect::<Vec<_>>());
        let field = |x: f64, y: f64| (0.5 + 0.02 * x - 0.01 * y, -0.3 + 0.015 * x + 0.03 * y);
        for (sigma, mixed) in [(0.13, true), (1.0, false)] {
            for w in [1usize, 3, 5, 7, 8, 50] {
                let h = 9usize;
                // 0: solved, 1: one filter (the aperture rule), 2: weight 1e-12 (the energy threshold)
                let kind = |x: usize, y: usize| match (mixed, x + 2 * y, 3 * x + y) {
                    (true, a, _) if a % 3 == 0 => 1,
                    (true, _, b) if b % 7 == 0 => 2,
                    _ => 0,
                };
                let pixel = |x: usize, y: usize| match kind(x, y) {
                    1 => (1, 1.0),
                    2 => (4, 1e-12),
                    _ => (4, 1.0),
                };
                let sums = sums_for_field_with(w, h, field, pixel);
                let (ux, uy) = (pseudo_random_image(w, h, 11), pseudo_random_image(w, h, 12));
                for order in [1u32, 2] {
                    let got = local_solve(&sums, &ux, &uy, sigma, order);
                    let want = local_solve_per_pixel(&sums, &ux, &uy, sigma, order);
                    let what = format!("sigma {sigma} {w}x{h} order {order}");
                    assert_eq!(bits(&got.ux.data), bits(&want.ux.data), "ux, {what}");
                    assert_eq!(bits(&got.uy.data), bits(&want.uy.data), "uy, {what}");
                    assert_eq!(
                        bits(&got.resid.data),
                        bits(&want.resid.data),
                        "resid, {what}"
                    );
                    assert_eq!(
                        bits(&got.energy.data),
                        bits(&want.energy.data),
                        "energy, {what}"
                    );
                    assert_eq!(slope_bits(&got), slope_bits(&want), "slope, {what}");
                    // not vacuous: some pixels are solved, and in the mixed case every pixel
                    // has the fate of its kind
                    let solved = got.resid.data.iter().filter(|&&r| r < PI).count();
                    assert!(solved > 0, "nothing solved, {what}");
                    if mixed {
                        let threshold = 1e-9 * crate::select::median(&got.energy.data) + 1e-30;
                        for p in 0..w * h {
                            let (x, y) = (p % w, p / w);
                            let k = kind(x, y);
                            let at = format!("pixel ({x}, {y}) of kind {k}, {what}");
                            assert_eq!(got.resid.data[p] < PI, k == 0, "solved, {at}");
                            assert_eq!(got.energy.data[p] <= threshold, k == 2, "energy, {at}");
                        }
                        if w >= 4 {
                            // the 4 lanes of the first group of every row disagree
                            for (y, row) in got.resid.data.chunks(w).enumerate() {
                                assert!(row[..4].iter().any(|&r| r < PI), "row {y}, {what}");
                                assert!(row[..4].contains(&PI), "row {y}, {what}");
                                // ... also through the aperture rule alone, not only through the
                                // energy reset: a kind-1 pixel of the group is rejected (its
                                // energy is above the threshold, asserted pixel by pixel above)
                                assert!(
                                    (0..4).any(|x| kind(x, y) == 1 && row[x] == PI),
                                    "row {y}: no aperture-rejected pixel in the first group, {what}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The energy threshold of `local_solve` with a non-zero median: a pixel with energy
    /// `<= 1e-9 median(energy)` keeps the input displacement (residual `pi`, zero slope) although
    /// its system is solvable, one above it is solved. The window is a single tap (`sigma` 0.1),
    /// so each pixel sees only its own sums, scaled by `f`: energy `2 f`, median 2, threshold 2e-9.
    #[test]
    fn refine_local_solve_resets_pixels_whose_energy_is_at_most_a_billionth_of_the_median() {
        let (w, h) = (9usize, 5usize);
        // gxx, gxy, gyy, gxc, gyc, gcc, gw: energy 2 and the solution (0.3, 0.1) at scale 1
        let base = [1.0, 0.0, 1.0, 0.3, 0.1, 0.2, 1.0];
        let scale = |p: usize| match p {
            10 => 0.9e-9,  // 1.8e-9, below the threshold
            11 => 1.0e-9,  // 2e-9, exactly the threshold: not above it
            12 => 1.1e-9,  // 2.2e-9, above
            13 => 1.0e-12, // far below
            _ => 1.0,
        };
        let sums_of =
            |i: usize| Image::from_vec(w, h, (0..w * h).map(|p| base[i] * scale(p)).collect());
        let [gxx, gxy, gyy, gxc, gyc, gcc, gw] = [0, 1, 2, 3, 4, 5, 6].map(sums_of);
        let sums = Sums {
            gxx,
            gxy,
            gyy,
            gxc,
            gyc,
            gcc,
            gw,
        };
        let (ux, uy) = (pseudo_random_image(w, h, 21), pseudo_random_image(w, h, 22));
        for order in [1u32, 2] {
            let sol = local_solve(&sums, &ux, &uy, 0.1, order);
            let what = format!("order {order}");
            let mut median_energy = sol.energy.data.clone();
            median_energy.sort_by(f64::total_cmp);
            assert_eq!(median_energy[w * h / 2], 2.0, "{what}");
            for p in 0..w * h {
                let reset = matches!(p, 10 | 11 | 13);
                if reset {
                    assert_eq!(
                        (sol.ux.data[p], sol.uy.data[p]),
                        (ux.data[p], uy.data[p]),
                        "{what} pixel {p}"
                    );
                    assert_eq!(sol.resid.data[p], PI, "{what} pixel {p}");
                    assert_eq!(sol.slope[p], [0.0; 4], "{what} pixel {p}");
                } else {
                    let (u, v) = (sol.ux.data[p], sol.uy.data[p]);
                    assert!(
                        (u - 0.3).abs() < 1e-9 && (v - 0.1).abs() < 1e-9,
                        "{what} pixel {p}: ({u}, {v})"
                    );
                    assert!(sol.resid.data[p] < PI, "{what} pixel {p}");
                }
            }
        }
    }

    fn size_check_cfg() -> Config {
        Config {
            outer_iters: 1,
            inner_iters: 1,
            ..Config::default()
        }
    }

    #[test]
    #[should_panic(expected = "refine: empty image")]
    fn refine_panics_for_an_empty_image() {
        let none = Image::<f32>::new(0, 0);
        refine(&none, &none, &none, &none, &size_check_cfg());
    }

    #[test]
    #[should_panic(expected = "refine: i1 is 24x24 but i2 is 23x24")]
    fn refine_panics_when_the_second_image_has_another_size() {
        let (a, b, zero) = (
            smooth_image(24, 24),
            smooth_image(23, 24),
            Image::new(24, 24),
        );
        refine(&a, &b, &zero, &zero, &size_check_cfg());
    }

    #[test]
    #[should_panic(expected = "refine: i1 is 24x24 but uy is 24x25")]
    fn refine_panics_when_the_displacement_has_another_size() {
        let (a, zero) = (smooth_image(24, 24), Image::new(24, 24));
        refine(&a, &a, &zero, &Image::new(24, 25), &size_check_cfg());
    }

    #[test]
    #[should_panic(expected = "order must be 1 or 2")]
    #[cfg_attr(miri, ignore)] // the panic comes after the first warp and filter responses: slow
    fn refine_panics_for_a_shape_order_other_than_1_or_2() {
        let (a, zero) = (smooth_image(24, 24), Image::new(24, 24));
        let cfg = Config {
            shape_order: 3,
            ..size_check_cfg()
        };
        refine(&a, &a, &zero, &zero, &cfg);
    }
}

/// Tests added after mutation testing (cargo-mutants, Task 11m).
#[cfg(test)]
mod mutation_gap_tests {
    use super::*;
    use crate::atan2::atan2;

    /// A system whose factorisation meets a pivot that is not a positive finite number is not
    /// solved. An infinite pivot is the case the finiteness check of the solution does not catch
    /// (dividing by infinity gives 0), so only the pivot check does.
    #[test]
    fn cholesky_solve_rejects_a_pivot_that_is_not_positive_and_finite() {
        let mut m = [[4.0, 2.0], [2.0, 3.0]];
        let mut b = [2.0, 1.0];
        assert!(cholesky_solve(&mut m, &mut b));
        // 4 x + 2 y = 2, 2 x + 3 y = 1 -> x = 0.5, y = 0
        assert!((b[0] - 0.5).abs() < 1e-12 && b[1].abs() < 1e-12, "{b:?}");
        for pivot in [f64::INFINITY, 0.0, -1.0, f64::NAN] {
            for at in 0..2 {
                let mut m = [[4.0, 0.0], [0.0, 3.0]];
                m[at][at] = pivot;
                let mut b = [1.0, 1.0];
                assert!(!cholesky_solve(&mut m, &mut b), "pivot {pivot} at {at}");
            }
        }
    }

    /// Deterministic coefficients of `n` filters on a `pw` x `ph` grid.
    fn responses(pw: usize, ph: usize, n: usize, seed: u64) -> Responses {
        let mut s = seed;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let data = (0..pw * ph * n)
            .map(|_| Complex32::new(rnd(), rnd()))
            .collect();
        Responses {
            w: pw,
            n_filters: n,
            data,
        }
    }

    /// The sums of `pixel_sums_at` at a position beyond the border of the image, but within its
    /// padding: the response splines are sampled on the padded grid (`w + 2 PAD` wide), so the
    /// position must not be clamped to a narrower one. One filter with a reference phase of 0
    /// (`r = 1`, `gx = 1`, `gy = 0`, `amp = 1`): then `dphi = atan2(-b.im, b.re)`, the weight is
    /// `|b|`, and the seven sums follow from the sample `b` alone, which `BSpline::sample` of the
    /// same coefficients gives (rounded to `f32`, as `sample_all` computes it in `f64`).
    #[test]
    fn pixel_sums_at_samples_the_padded_grid_beyond_the_border() {
        let (w, h) = (40usize, 30usize);
        let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
        let resp = responses(pw, ph, 1, 7);
        let oracle = BSpline {
            coeffs: Image::from_vec(pw, ph, resp.data.clone()),
        };
        let sm = Smoothed {
            sx: Image::new(w, h),
            sy: Image::new(w, h),
            csx: BSpline {
                coeffs: Image::new(w, h),
            },
            csy: BSpline {
                coeffs: Image::new(w, h),
            },
        };
        let rp = [1.0f32, 0.0, 1.0, 0.0, 1.0];
        let (mut acc, mut sampled) = ([[0.0f64; 2]; 1], [[0.0f64; 2]; 1]);
        let (mut num, mut den, mut mag, mut dphi) = ([0.0f64], [0.0f64], [0.0f64], [0.0f64]);
        let mut scratch = PixelScratch {
            acc: &mut acc,
            sampled: &mut sampled,
            num: &mut num,
            den: &mut den,
            mag: &mut mag,
            dphi: &mut dphi,
        };
        let (wf, hf) = (w as f64, h as f64);
        let positions = [
            (20.0, 15.0),
            (wf - 1.0, hf - 1.0),
            (wf + 0.5, 10.0),
            (wf + 1.5, hf + 1.5),
            (wf + 12.25, 8.5),
            (5.0, hf + 20.75),
            (wf + 28.5, hf + 28.5),
            (-0.5, -3.25),
            (-17.75, 12.0),
            (-31.0, -31.0),
        ];
        for (qx, qy) in positions {
            let got = pixel_sums_at(1, &resp, &sm, (qx, qy), (0.0, 0.0), &rp, &mut scratch);
            // the sample of the independent spline over the padded grid
            let t = taps(qx + PAD as f64, qy + PAD as f64, pw, ph).expect("finite");
            let mut b = [[0.0f64; 2]];
            let mut tmp = [[0.0f64; 2]];
            resp.sample_all(1, &t, &mut tmp, &mut b);
            let spline = oracle.sample(qx + PAD as f64, qy + PAD as f64);
            assert_eq!(
                (b[0][0] as f32).to_bits(),
                spline.re.to_bits(),
                "re at ({qx}, {qy})"
            );
            assert_eq!(
                (b[0][1] as f32).to_bits(),
                spline.im.to_bits(),
                "im at ({qx}, {qy})"
            );
            let wk = madd(b[0][0], b[0][0], b[0][1] * b[0][1]).sqrt();
            let c = atan2(-b[0][1], b[0][0]);
            let want = [
                madd(wk, 1.0, 0.0),
                0.0,
                0.0,
                madd(wk, c, 0.0),
                0.0,
                madd(wk * c, c, 0.0),
                wk,
            ];
            let bits = |v: &[f64; 7]| v.map(f64::to_bits);
            assert_eq!(bits(&got), bits(&want), "sums at ({qx}, {qy})");
        }
        // not finite: no sample
        let got = pixel_sums_at(
            1,
            &resp,
            &sm,
            (f64::NAN, 3.0),
            (0.0, 0.0),
            &rp,
            &mut scratch,
        );
        assert!(got.iter().all(|v| v.is_nan()));
    }
}

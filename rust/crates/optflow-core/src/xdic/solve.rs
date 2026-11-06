//! The per-side solve and the geometry update of `crack_model = "xdic"` (`side_extrapolate`,
//! `_Geo`, `_update_geometry`, `masked_reference`, `_moments_grid` and `refine_xdic` of
//! `xdic.py`; its module doc describes the method).
//!
//! Precision as in the rest of the port: images, filter responses and spline coefficients are
//! `f32`; the fields, the moments, the local systems and the residuals are `f64`.

use num_complex::Complex32;
use rayon::prelude::*;

use super::geom::{polyline_fields, resample, PolyFields};
use super::num::{lstsq, median, median3_nearest};
use super::params::*;
use super::{Flow, Poly};
use crate::config::Config;
use crate::edt::{distance_to_invalid, nearest_source, within};
use crate::fft::Fft2;
use crate::gabor::{padded_spectrum, reference_values, Bank, RefPhase, PAD};
use crate::image::Image;
use crate::interp::{gauss_blur, gauss_kernel, taps, BSpline};
use crate::phaseflow::{
    accumulate_into, confidence, local_solve_into, local_solve_spans, reportable, texture_scale,
    to_f32, warp_shifted, ApSolve, Aperture, Prepared, Responses, Smoothed, SolveOut, Sums,
};
use crate::prewarp::prewarp;
use crate::prof::{self, Stage};

/// The two sides of a line: index 0 is `+` (`phi >= 0`), index 1 is `-`.
const SIDES: [i8; 2] = [1, -1];

/// `bs` (the spline of an `f32` image) at `(x, y)` in `f64`, not rounded to `f32`; NaN for a
/// non-finite coordinate.
pub fn sample_f64(bs: &BSpline<f32>, x: f64, y: f64) -> f64 {
    let (w, h) = (bs.coeffs.w, bs.coeffs.h);
    let Some(t) = taps(x, y, w, h) else {
        return f64::NAN;
    };
    let mut total = 0.0;
    for (&yj, &wyj) in t.yi.iter().zip(&t.wy) {
        let row = &bs.coeffs.data[yj * w..(yj + 1) * w];
        let mut acc = t.wx[0] * f64::from(row[t.xi[0]]);
        for (&xk, &wxk) in t.xi.iter().zip(&t.wx).skip(1) {
            acc += wxk * f64::from(row[xk]);
        }
        total += wyj * acc;
    }
    total
}

/// The geometry of a set of lines for the solve (`_Geo`).
#[derive(Clone, Debug)]
pub struct Geo {
    pub w: usize,
    pub h: usize,
    pub lines: Vec<Poly>,
    pub f: PolyFields,
    /// +1 where `phi >= 0`, else -1.
    pub side: Vec<i8>,
    /// The strip `S`: interior and `|phi| <= R_B`.
    pub strip: Vec<bool>,
    /// Pixels painted for side `s` (index as `SIDES`): the strip of the other side and the
    /// mixed-pixel layer.
    pub fill: [Vec<bool>; 2],
    /// Pixels of the other side within `2 R_B`: not a source of side `s`'s pre-warp.
    pub fill_pw: [Vec<bool>; 2],
    /// Pixels of side `s` in the strip: what its solve replaces.
    pub sel: [Vec<bool>; 2],
}

impl Geo {
    pub fn new(lines: &[Poly], w: usize, h: usize, cfg: &Config) -> Geo {
        let f = polyline_fields(lines, w, h);
        let rb = cfg.xdic_rb_px();
        let n = w * h;
        let side: Vec<i8> = f
            .phi
            .par_iter()
            .map(|&p| if p >= 0.0 { 1 } else { -1 })
            .collect();
        let strip: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|i| f.interior[i] && f.phi[i].abs() <= rb)
            .collect();
        let mix = |i: usize| f.phi[i].abs() < MIX_PX;
        let mk = |g: &(dyn Fn(usize, i8) -> bool + Sync)| -> [Vec<bool>; 2] {
            [
                (0..n).into_par_iter().map(|i| g(i, SIDES[0])).collect(),
                (0..n).into_par_iter().map(|i| g(i, SIDES[1])).collect(),
            ]
        };
        let fill = mk(&|i, s| strip[i] && (side[i] == -s || mix(i)));
        let fill_pw = mk(&|i, s| f.interior[i] && side[i] == -s && f.phi[i].abs() <= 2.0 * rb);
        let sel = mk(&|i, s| strip[i] && side[i] == s);
        Geo {
            w,
            h,
            lines: lines.to_vec(),
            f,
            side,
            strip,
            fill,
            fill_pw,
            sel,
        }
    }
}

/// `distance_transform_edt(~low) > r` for a non-empty `low`.
fn far_from(low: &[bool], w: usize, h: usize, r: f64) -> Vec<bool> {
    let d = distance_to_invalid(&Image::from_vec(w, h, low.iter().map(|&b| !b).collect()));
    d.data.iter().map(|&v| !within(v, r)).collect()
}

/// `side_extrapolate`: the trusted displacement of one side (outside `fill`, `conf >= conf_min`
/// and farther than the automatic guard from the low-confidence pixels outside `fill`), every
/// other pixel the first-order extrapolation `u(N) + grad u(N) (X - N)` from its nearest trusted
/// pixel `N`. Returns `(ux, uy, ok)` (`ok` = trusted).
pub fn side_extrapolate(
    ux: &[f64],
    uy: &[f64],
    slope: &[[f64; 4]],
    conf: &[f64],
    cfg: &Config,
    fill: &[bool],
    w: usize,
    h: usize,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let n = w * h;
    let low: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| conf[i] < cfg.conf_min && !fill[i])
        .collect();
    let ok: Vec<bool> = if low.par_iter().any(|&b| b) {
        let far = far_from(&low, w, h, cfg.auto_edge_guard_px());
        (0..n).into_par_iter().map(|i| !fill[i] && far[i]).collect()
    } else {
        fill.par_iter().map(|&f| !f).collect()
    };
    let (mut ox, mut oy) = (ux.to_vec(), uy.to_vec());
    let any = ok.par_iter().any(|&b| b);
    if any && !ok.par_iter().all(|&b| b) {
        let nearest =
            nearest_source(&Image::from_vec(w, h, ok.clone())).expect("a trusted pixel exists");
        ox.par_chunks_mut(w)
            .zip(oy.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (rx, ry))| {
                for x in 0..w {
                    let i = y * w + x;
                    if ok[i] {
                        continue;
                    }
                    let nn = nearest[i] as usize;
                    let (dx, dy) = (x as f64 - (nn % w) as f64, y as f64 - (nn / w) as f64);
                    let g = slope[nn];
                    rx[x] = ux[nn] + g[0] * dx + g[2] * dy;
                    ry[x] = uy[nn] + g[1] * dx + g[3] * dy;
                }
            });
    }
    (ox, oy, ok)
}

/// The squared intensity difference between `i1` at `p` and `gain[0] I2 + gain[1]` at `p + u`.
fn intensity_resid(
    i1: &Image<f32>,
    c2: &BSpline<f32>,
    gain: [f64; 2],
    p: usize,
    u: f64,
    v: f64,
) -> f64 {
    let (x, y) = ((p % i1.w) as f64, (p / i1.w) as f64);
    let d = f64::from(i1.data[p]) - (gain[0] * sample_f64(c2, x + u, y + v) + gain[1]);
    d * d
}

/// `np.interp(x, xp, fp)` at the integers `0..n` (linear, constant outside `xp`).
fn interp_fill(vals: &[f64]) -> Vec<f64> {
    let known: Vec<usize> = (0..vals.len()).filter(|&i| vals[i].is_finite()).collect();
    (0..vals.len())
        .map(|i| {
            if i <= known[0] {
                return vals[known[0]];
            }
            let last = *known.last().unwrap();
            if i >= last {
                return vals[last];
            }
            let hi = known.partition_point(|&k| k < i);
            let (a, b) = (known[hi - 1], known[hi]);
            if known[hi] == i {
                return vals[i];
            }
            vals[a] + (vals[b] - vals[a]) * ((i - a) as f64 / (b - a) as f64)
        })
        .collect()
}

/// `np.arange(start, stop, step)` as numpy fills it.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    let delta = (start + step) - start;
    (0..n).map(|i| start + i as f64 * delta).collect()
}

/// Extrapolations of the two sides (for the pixels that change side).
pub type Ext = [(Vec<f64>, Vec<f64>); 2];

/// Spec 3.3 (`_update_geometry`): reassign the pixels near the line by the residual of either
/// side's motion and move the polylines. A pixel votes only when the two residuals differ by more
/// than `UPD_INFO` times the noise level `noise`. Returns the new lines and, when the update ran,
/// the extrapolation of each side.
#[allow(clippy::too_many_arguments)]
pub fn update_geometry(
    geo: &Geo,
    ux: &[f64],
    uy: &[f64],
    slope: &[[f64; 4]],
    conf: &[f64],
    cfg: &Config,
    i1: &Image<f32>,
    c2: &BSpline<f32>,
    gain: [f64; 2],
    noise: f64,
) -> (Vec<Poly>, Option<Ext>) {
    let (w, h) = (geo.w, geo.h);
    let n = w * h;
    let band: Vec<bool> = (0..n)
        .map(|i| geo.f.interior[i] && geo.f.phi[i].abs() <= UPD_BAND)
        .collect();
    if !band.iter().any(|&b| b) {
        return (geo.lines.clone(), None);
    }
    let bp: Vec<usize> = (0..n).filter(|&i| band[i]).collect();
    let blur_w = gauss_blur(
        &Image::from_vec(w, h, band.iter().map(|&b| f64::from(u8::from(b))).collect()),
        1.0,
    );
    let mut ext: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut e: Vec<Vec<f64>> = Vec::new();
    for k in 0..2 {
        // side s pixels of S beyond the band are the source
        let fill: Vec<bool> = (0..n)
            .map(|i| !(geo.sel[k][i] && geo.f.phi[i].abs() > UPD_BAND))
            .collect();
        let (ex, ey, ok) = side_extrapolate(ux, uy, slope, conf, cfg, &fill, w, h);
        if !ok.iter().any(|&b| b) {
            return (geo.lines.clone(), None);
        }
        let mut res = vec![0.0f64; n];
        for &p in &bp {
            res[p] = intensity_resid(i1, c2, gain, p, ex[p], ey[p]);
        }
        let blurred = gauss_blur(&Image::from_vec(w, h, res), 1.0);
        e.push(
            (0..n)
                .map(|i| blurred.data[i] / blur_w.data[i].max(1e-9))
                .collect(),
        );
        ext.push((ex, ey));
    }
    let lab_plus: Vec<bool> = (0..n).map(|i| e[0][i] < e[1][i]).collect();
    let wgt: Vec<f64> = (0..n)
        .map(|i| {
            let d = (e[0][i] - e[1][i]).abs();
            if d > UPD_INFO * noise {
                d
            } else {
                0.0
            }
        })
        .collect();
    let deltas = arange(-UPD_BAND, UPD_BAND + 1e-9, 0.25);
    let mut new = Vec::with_capacity(geo.lines.len());
    for (li, p) in geo.lines.iter().enumerate() {
        let m: Vec<usize> = bp
            .iter()
            .copied()
            .filter(|&i| geo.f.lid[i] == li as i32)
            .collect();
        let (pts, nrm, sarc) = resample(p, UPD_BIN);
        if m.is_empty() || sarc.len() < 2 {
            new.push(p.clone());
            continue;
        }
        let nb = sarc.len() - 1;
        let mut dl = vec![f64::NAN; nb];
        for (b, d) in dl.iter_mut().enumerate() {
            let mb: Vec<usize> = m
                .iter()
                .copied()
                .filter(|&i| {
                    let t = geo.f.tau[i];
                    t >= sarc[b]
                        && if b < nb - 1 {
                            t < sarc[b + 1]
                        } else {
                            t <= sarc[b + 1]
                        }
                })
                .collect();
            if mb.len() < 20 {
                continue;
            }
            // cost(delta) = weight of + labels with phi < delta plus weight of - labels with phi > delta
            let cost: Vec<f64> = deltas
                .iter()
                .map(|&dv| {
                    let mut c = 0.0;
                    for &i in &mb {
                        let ph = geo.f.phi[i];
                        if lab_plus[i] && ph < dv {
                            c += wgt[i];
                        }
                    }
                    for &i in &mb {
                        let ph = geo.f.phi[i];
                        if !lab_plus[i] && ph > dv {
                            c += wgt[i];
                        }
                    }
                    c
                })
                .collect();
            let cmin = cost.iter().copied().fold(f64::INFINITY, f64::min);
            let best: Vec<f64> = (0..deltas.len())
                .filter(|&k| cost[k] <= cmin + 1e-12)
                .map(|k| deltas[k])
                .collect();
            *d = median(&best);
        }
        if !dl.iter().any(|v| v.is_finite()) {
            new.push(p.clone());
            continue;
        }
        let mut dl = interp_fill(&dl);
        if nb >= 3 {
            dl = median3_nearest(&dl);
        }
        let mut dv = vec![dl[0]];
        dv.extend(dl.windows(2).map(|s| 0.5 * (s[0] + s[1])));
        dv.push(dl[nb - 1]);
        new.push(
            pts.iter()
                .zip(&nrm)
                .zip(&dv)
                .map(|((q, nn), d)| [q[0] + d * nn[0], q[1] + d * nn[1]])
                .collect(),
        );
    }
    let mut it = ext.into_iter();
    (new, Some([it.next().unwrap(), it.next().unwrap()]))
}

/// A filter below this fraction of its largest value counts as zero in [`filtered_rows`]: what it
/// leaves out is far below the rounding of the `f32` responses (checked in the tests of this
/// file).
const FILTER_ZERO: f32 = 1e-10;

/// The frequency columns where each filter of `bank` is not zero (see [`FILTER_ZERO`]).
fn active_columns(bank: &Bank) -> Vec<Vec<u32>> {
    bank.filters
        .par_iter()
        .map(|g| {
            let mut largest = vec![0.0f32; bank.w];
            for row in g.chunks(bank.w) {
                for (l, &v) in largest.iter_mut().zip(row) {
                    *l = l.max(v.abs());
                }
            }
            let top = largest.iter().copied().fold(0.0, f32::max);
            (0..bank.w as u32)
                .filter(|&x| largest[x as usize] > FILTER_ZERO * top)
                .collect()
        })
        .collect()
}

/// The rows `y0 <= y < y1` of the (unpadded) image of the inverse transform of `spec` times the
/// filter `g`, `w` values per row: the columns of the padded grid where the filter is not zero
/// (`active`), then only the rows asked for. The values of `Fft2::inverse` there, up to rounding
/// (the transforms run in the other order).
fn filtered_rows(
    spec: &[Complex32],
    g: &[f32],
    active: &[u32],
    fft: &Fft2,
    (y0, y1): (usize, usize),
    w: usize,
) -> Vec<Complex32> {
    let (ph, pw) = (fft.height(), fft.width());
    let (row_plan, col_plan, scale) = fft.inverse_parts();
    let zero = Complex32::default();
    let mut scratch = vec![zero; row_plan.get_inplace_scratch_len()];
    let mut cols = vec![zero; active.len() * ph];
    for (line, &x) in cols.chunks_mut(ph).zip(active) {
        for (y, v) in line.iter_mut().enumerate() {
            let i = y * pw + x as usize;
            *v = spec[i] * g[i];
        }
    }
    scratch.resize(col_plan.get_inplace_scratch_len().max(scratch.len()), zero);
    if !cols.is_empty() {
        col_plan.process_with_scratch(&mut cols, &mut scratch);
    }
    let n_rows = y1 - y0;
    let mut rows = vec![zero; n_rows * pw];
    for (line, &x) in cols.chunks(ph).zip(active) {
        let x = x as usize;
        for r in 0..n_rows {
            rows[r * pw + x] = line[y0 + PAD + r];
        }
    }
    drop(cols);
    if n_rows > 0 {
        row_plan.process_with_scratch(&mut rows, &mut scratch);
    }
    let mut out = Vec::with_capacity(n_rows * w);
    for row in rows.chunks(pw) {
        out.extend(row[PAD..PAD + w].iter().map(|&v| v * scale));
    }
    out
}

/// What the per-side solve of a leg needs besides the baseline's [`Prepared`], the same for all
/// geometries: the mean of `i1`, the spectral derivatives of the unpainted `i1` and the non-zero
/// columns of the filters.
pub(crate) struct SideData {
    m1: f64,
    dx: Image<f32>,
    dy: Image<f32>,
    active: Vec<Vec<u32>>,
}

impl SideData {
    pub(crate) fn new(i1: &Image<f32>, prep: &Prepared) -> SideData {
        let (w, h) = (i1.w, i1.h);
        let (bank, fft) = (&prep.bank, &prep.fft);
        let pw = bank.w;
        let spec = padded_spectrum(i1, fft);
        let deriv = |axis_x: bool| -> Image<f32> {
            let mut p: Vec<Complex32> = spec
                .par_iter()
                .enumerate()
                .map(|(i, &c)| {
                    let f = if axis_x {
                        bank.fx[i % pw]
                    } else {
                        bank.fy[i / pw]
                    };
                    Complex32::new(-c.im * f, c.re * f)
                })
                .collect();
            fft.inverse(&mut p);
            Image::from_vec(
                w,
                h,
                (0..w * h)
                    .map(|i| p[(i / w + PAD) * pw + i % w + PAD].re)
                    .collect(),
            )
        };
        SideData {
            m1: i1.data.iter().map(|&v| f64::from(v)).sum::<f64>() / (w * h) as f64,
            dx: deriv(true),
            dy: deriv(false),
            active: active_columns(bank),
        }
    }
}

/// `masked_reference`: per filter the response of `i1` with `fill` painted `m1`, and the phase
/// gradient of a shift of the side's texture inside the FIXED mask, `Im(conj(r) G*(M dI1)) /
/// |r|^2` with `M = 1 - fill` and `dI1` the spectral derivative of the unpainted `i1`, not the
/// gradient of `r` itself (which also holds the edge of the paint). Only the rows `rows.0 <= y <
/// rows.1` of the image: each image of the result has `rows.1 - rows.0` rows.
fn masked_reference(
    i1: &Image<f32>,
    fill: &[bool],
    data: &SideData,
    prep: &Prepared,
    rows: (usize, usize),
) -> Vec<RefPhase> {
    let (w, h) = (i1.w, i1.h);
    let (bank, fft) = (&prep.bank, &prep.fft);
    let m1 = data.m1 as f32;
    let painted = Image::from_vec(
        w,
        h,
        i1.data
            .iter()
            .zip(fill)
            .map(|(&v, &f)| if f { m1 } else { v })
            .collect(),
    );
    let spec = padded_spectrum(&painted, fft);
    // the derivative of the unpainted image along an axis, kept only outside the paint
    let kept = |d: &Image<f32>| -> Vec<Complex32> {
        let masked = d
            .data
            .iter()
            .zip(fill)
            .map(|(&v, &f)| if f { 0.0 } else { v })
            .collect();
        padded_spectrum(&Image::from_vec(w, h, masked), fft)
    };
    let (sdx, sdy) = (kept(&data.dx), kept(&data.dy));
    let n_rows = rows.1 - rows.0;
    bank.filters
        .par_iter()
        .zip(&data.active)
        .map(|(g, active)| {
            let filt = |s: &[Complex32]| filtered_rows(s, g, active, fft, rows, w);
            let (r, rx, ry) = (filt(&spec), filt(&sdx), filt(&sdy));
            let m = r.len();
            let mut gx = vec![0.0f32; m];
            let mut gy = vec![0.0f32; m];
            let mut amp = vec![0.0f32; m];
            for i in 0..m {
                [gx[i], gy[i], amp[i]] = reference_values(r[i], rx[i], ry[i]);
            }
            RefPhase {
                r: Image::from_vec(w, n_rows, r),
                gx: Image::from_vec(w, n_rows, gx),
                gy: Image::from_vec(w, n_rows, gy),
                amp: Image::from_vec(w, n_rows, amp),
            }
        })
        .collect()
}

/// What the solve of one side covers: the pixels `sel` to solve, per row of a rectangle of the
/// image the span holding them, and the span of the moments their windows read.
struct SidePlan {
    /// Corner of the rectangle in the image.
    ox: usize,
    oy: usize,
    rw: usize,
    rh: usize,
    solve: Vec<(usize, usize)>,
    moments: Vec<(usize, usize)>,
}

impl SidePlan {
    /// None when no pixel is selected. `r` is the radius of the window of the local solve.
    fn new(sel: &[bool], w: usize, h: usize, r: usize) -> Option<SidePlan> {
        let spans: Vec<Option<(usize, usize)>> = sel
            .chunks(w)
            .map(|row| {
                let first = row.iter().position(|&b| b)?;
                let last = row.iter().rposition(|&b| b)?;
                Some((first, last + 1))
            })
            .collect();
        let y_first = spans.iter().position(|s| s.is_some())?;
        let y_last = spans.iter().rposition(|s| s.is_some())?;
        let (oy, y_end) = (y_first.saturating_sub(r), (y_last + r + 1).min(h));
        let (lo, hi) = spans
            .iter()
            .flatten()
            .fold((w, 0), |(a, b), s| (a.min(s.0), b.max(s.1)));
        let (ox, x_end) = (lo.saturating_sub(r), (hi + r).min(w));
        let rect = |(a, b): (usize, usize)| (a - ox, b - ox);
        let solve = spans[oy..y_end]
            .iter()
            .map(|s| s.map_or((0, 0), rect))
            .collect();
        let moments = (oy..y_end)
            .map(|y| {
                let near = &spans[y.saturating_sub(r)..(y + r + 1).min(h)];
                let (a, b) = near
                    .iter()
                    .flatten()
                    .fold((w, 0), |(a, b), s| (a.min(s.0), b.max(s.1)));
                if a < b {
                    rect((a.saturating_sub(r).max(ox), (b + r).min(x_end)))
                } else {
                    (0, 0)
                }
            })
            .collect();
        Some(SidePlan {
            ox,
            oy,
            rw: x_end - ox,
            rh: y_end - oy,
            solve,
            moments,
        })
    }
}

/// `_moments_grid`: the seven per-pixel sums of the local solve for the responses `resp` of the
/// side's pre-warped `i2` (pre-warp `s`), taken at the grid position itself: `c = dphi + g' s`,
/// the pixels of `fill` weighing nothing. On the rectangle of `plan`, inside its moment spans
/// (the local solve of the plan reads nothing else); `rk` and `resp` start at the image row `y0`.
fn moments_grid(
    rk: &[RefPhase],
    resp: &[Vec<Complex32>],
    (sx, sy): (&[f64], &[f64]),
    fill: &[bool],
    plan: &SidePlan,
    (w, y0): (usize, usize),
) -> Sums {
    let (rw, rh) = (plan.rw, plan.rh);
    let mut acc = vec![[0.0f64; 7]; rw * rh];
    acc.par_chunks_mut(rw).enumerate().for_each(|(ry, row)| {
        let (a, b) = plan.moments[ry];
        let y = ry + plan.oy;
        for (rx, sum) in row.iter_mut().enumerate().take(b).skip(a) {
            let x = rx + plan.ox;
            let i = y * w + x;
            if fill[i] {
                continue;
            }
            let j = (y - y0) * w + x;
            for (rf, r2) in rk.iter().zip(resp) {
                let r1 = rf.r.data[j];
                let r2 = r2[j];
                let (a1, b1) = (f64::from(r1.re), f64::from(r1.im));
                let (a2, b2) = (f64::from(r2.re), f64::from(r2.im));
                // r1 * conj(r2)
                let (pr, pi) = (a1 * a2 + b1 * b2, b1 * a2 - a1 * b2);
                let dphi = pi.atan2(pr);
                let wk = f64::from(rf.amp.data[j]) * a2.hypot(b2);
                let (gx, gy) = (f64::from(rf.gx.data[j]), f64::from(rf.gy.data[j]));
                let c = dphi + gx * sx[i] + gy * sy[i];
                sum[0] += wk * gx * gx;
                sum[1] += wk * gx * gy;
                sum[2] += wk * gy * gy;
                sum[3] += wk * gx * c;
                sum[4] += wk * gy * c;
                sum[5] += wk * c * c;
                sum[6] += wk;
            }
        }
    });
    let plane = |k: usize| Image::from_vec(rw, rh, acc.iter().map(|a| a[k]).collect());
    Sums {
        gxx: plane(0),
        gxy: plane(1),
        gyy: plane(2),
        gxc: plane(3),
        gyc: plane(4),
        gcc: plane(5),
        gw: plane(6),
    }
}

/// The reference responses of the two sides for one geometry, on the image rows `rows`.
struct SideRefs {
    plans: [Option<SidePlan>; 2],
    phases: [Option<Vec<RefPhase>>; 2],
    rows: (usize, usize),
}

/// The state of one side of the strip in an outer iteration.
struct SideState<'a> {
    rk: &'a [RefPhase],
    respk: Vec<Vec<Complex32>>,
    skx: Vec<f64>,
    sky: Vec<f64>,
}

/// `_confidence` of the solve: texture factor times the phase residual factor (before the border
/// rule), in `[0, 1]`.
fn confidence_of(energy: &[f64], resid: &[f64], cfg: &Config) -> Vec<f64> {
    let scale = texture_scale(energy);
    energy
        .par_iter()
        .zip(resid)
        .map(|(&e, &r)| confidence(e, r, true, scale, cfg.phase_tol))
        .collect()
}

/// Spec 3.2 / 3.3 (`refine_xdic`): the per-side re-solve of the strip around `lines`, warm-started
/// from the baseline refine `fl0`. Returns the flow, the lines after the geometry updates and
/// their `Geo`.
pub fn refine_xdic(
    i1: &Image<f32>,
    i2: &Image<f32>,
    fl0: &Flow,
    cfg: &Config,
    lines: &[Poly],
) -> (Flow, Vec<Poly>, Geo) {
    let prep = Prepared::new(i1, i2, cfg);
    let data = prof::time(Stage::XdicResponses, || SideData::new(i1, &prep));
    refine_xdic_on(&prep, &data, i1, fl0, cfg, lines)
}

/// [`refine_xdic`] with what the baseline of the leg prepared, and the [`SideData`] of the leg.
pub(crate) fn refine_xdic_on(
    prep: &Prepared,
    data: &SideData,
    i1: &Image<f32>,
    fl0: &Flow,
    cfg: &Config,
    lines: &[Poly],
) -> (Flow, Vec<Poly>, Geo) {
    let (w, h) = (i1.w, i1.h);
    let n = w * h;
    let (bank, fft, reference, spline2) = (&prep.bank, &prep.fft, &prep.reference, &prep.spline2);
    let m1 = data.m1;
    // the radius of the window of the local solve
    let solve_r = gauss_kernel(cfg.sigma_w).len() / 2;
    let mut ux = Image::from_vec(w, h, fl0.ux.clone());
    let mut uy = Image::from_vec(w, h, fl0.uy.clone());
    let mut slope = fl0.slope.clone();
    let mut conf = fl0.conf.clone();
    let mut resid = Image::from_vec(w, h, fl0.resid.clone());
    let mut energy = Image::from_vec(w, h, vec![1.0f64; n]);
    let mut geo = prof::time(Stage::XdicGeometry, || Geo::new(lines, w, h, cfg));
    let xiters = usize::try_from(cfg.xdic_iters).unwrap_or(0);
    let n_outer = cfg.outer_iters.max(2 * xiters + 2);
    let sched: Vec<usize> = (0..xiters).map(|i| 2 + 2 * i).collect();
    let mut gain_done = false;
    let mut gain = [1.0, 0.0];
    let mut noise = 0.0;
    let aperture = Aperture::from_env();

    // before the first iteration the low-confidence pixels of S take the unblurred extrapolation
    // of their side (the second side sees the first one's result)
    prof::time(Stage::XdicSolve, || {
        for k in 0..2 {
            let (ex, ey, _) = side_extrapolate(
                &ux.data,
                &uy.data,
                &slope,
                &conf,
                cfg,
                &geo.fill_pw[k],
                w,
                h,
            );
            for i in 0..n {
                if geo.sel[k][i] && conf[i] < cfg.conf_min {
                    ux.data[i] = ex[i];
                    uy.data[i] = ey[i];
                }
            }
        }
    });
    let mut refs: Option<SideRefs> = None;
    let mut spent: Vec<Complex32> = Vec::new();
    for it in 0..n_outer {
        if sched.contains(&it) {
            if !gain_done {
                // brightness of I2 against I1 (gain, offset) from the confident pixels off the strip
                let ok: Vec<usize> = (0..n)
                    .filter(|&i| conf[i] >= 0.5 && !geo.strip[i])
                    .collect();
                if ok.len() > 100 {
                    let rows: Vec<[f64; 2]> = ok
                        .iter()
                        .map(|&i| {
                            let (x, y) = ((i % w) as f64, (i / w) as f64);
                            [sample_f64(spline2, x + ux.data[i], y + uy.data[i]), 1.0]
                        })
                        .collect();
                    let y: Vec<f64> = ok.iter().map(|&i| f64::from(i1.data[i])).collect();
                    gain = lstsq(&rows, &y);
                    noise = rows
                        .iter()
                        .zip(&y)
                        .map(|(r, &t)| {
                            let d = t - (r[0] * gain[0] + r[1] * gain[1]);
                            d * d
                        })
                        .sum::<f64>()
                        / ok.len() as f64;
                }
                gain_done = true;
            }
            let (newl, ext) = prof::time(Stage::XdicGeometry, || {
                let (nl, ext) = update_geometry(
                    &geo, &ux.data, &uy.data, &slope, &conf, cfg, i1, spline2, gain, noise,
                );
                let g2 = ext.as_ref().map(|_| Geo::new(&nl, w, h, cfg));
                (g2.map(|g| (g, nl)), ext)
            });
            if let (Some((g2, _)), Some(ext)) = (newl, ext) {
                for k in 0..2 {
                    for i in 0..n {
                        if g2.sel[k][i] && !geo.sel[k][i] {
                            ux.data[i] = ext[k].0[i];
                            uy.data[i] = ext[k].1[i];
                            conf[i] = 0.0;
                        }
                    }
                }
                geo = g2;
                refs = None;
            }
        }
        if refs.is_none() {
            refs = Some(prof::time(Stage::XdicResponses, || {
                let plans = [0, 1].map(|k| SidePlan::new(&geo.sel[k], w, h, solve_r));
                let y0 = plans.iter().flatten().map(|p| p.oy).min().unwrap_or(0);
                let y1 = plans
                    .iter()
                    .flatten()
                    .map(|p| p.oy + p.rh)
                    .max()
                    .unwrap_or(0);
                let side = |k: usize| {
                    plans[k]
                        .as_ref()
                        .map(|_| masked_reference(i1, &geo.fill[k], data, prep, (y0, y1)))
                };
                let phases = [side(0), side(1)];
                SideRefs {
                    plans,
                    phases,
                    rows: (y0, y1),
                }
            }));
        }
        // the pre-warp: the first iteration has the confidence of the baseline, later ones their own
        let low = Image::from_vec(w, h, conf.iter().map(|&c| c < cfg.conf_min).collect());
        let (psx, psy) = prewarp(&ux, &uy, &slope, &low, cfg);
        let (sx, sy) = prof::time(Stage::Smooth, || (to_f32(&psx), to_f32(&psy)));
        let warped = prof::time(Stage::Warp, || warp_shifted(spline2, &sx, &sy, (0, 0)));
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
        // the per-side pre-warps and the responses of the side's own pre-warped I2
        let ref_sets = refs.as_ref().unwrap();
        let mut states: Vec<Option<SideState>> = Vec::new();
        for k in 0..2 {
            let Some(rk) = ref_sets.phases[k].as_deref() else {
                states.push(None);
                continue;
            };
            let (ex, ey, _) = prof::time(Stage::XdicSolve, || {
                side_extrapolate(
                    &ux.data,
                    &uy.data,
                    &slope,
                    &conf,
                    cfg,
                    &geo.fill_pw[k],
                    w,
                    h,
                )
            });
            let (skx, sky) = prof::time(Stage::Smooth, || {
                (
                    gauss_blur(&Image::from_vec(w, h, ex), cfg.sigma_s),
                    gauss_blur(&Image::from_vec(w, h, ey), cfg.sigma_s),
                )
            });
            let respk = prof::time(Stage::XdicResponses, || {
                let (fx32, fy32) = (to_f32(&skx), to_f32(&sky));
                let wp = warp_shifted(spline2, &fx32, &fy32, (0, 0));
                let jk = Image::from_vec(
                    w,
                    h,
                    (0..n)
                        .map(|i| {
                            if geo.fill[k][i] {
                                m1 as f32
                            } else {
                                wp.data[i]
                            }
                        })
                        .collect(),
                );
                let spec = padded_spectrum(&jk, fft);
                bank.filters
                    .par_iter()
                    .zip(&data.active)
                    .map(|(g, active)| filtered_rows(&spec, g, active, fft, ref_sets.rows, w))
                    .collect::<Vec<_>>()
            });
            states.push(Some(SideState {
                rk,
                respk,
                skx: skx.data,
                sky: sky.data,
            }));
        }
        let (ux_in, uy_in) = (ux.clone(), uy.clone());
        // the inner iterations of the baseline solve over the whole image
        let mut sums = Sums::new(w, h);
        let mut next = (Image::<f64>::new(w, h), Image::<f64>::new(w, h));
        for _ in 0..cfg.inner_iters {
            prof::time(Stage::Accumulate, || {
                accumulate_into(reference, &responses, &sm, &ux, &uy, &mut sums)
            });
            if slope.len() != n {
                slope = vec![[0.0; 4]; n];
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
        // what the whole image's solve would call a pixel without texture: from the median energy
        // of the baseline solve just done, since the side's own is known only on the strip. The
        // two are of the same order, and a pixel near this threshold is far below the confidence
        // limit either way, so the choice does not reach a valid pixel
        let no_texture = 1e-9 * (4.0 * texture_scale(&energy.data)) + 1e-30;
        // one Gauss-Newton step of each side about its pre-warp (see `masked_reference`)
        prof::time(Stage::XdicSolve, || {
            for (k, st) in states.iter().enumerate() {
                let Some(st) = st else { continue };
                let plan = ref_sets.plans[k]
                    .as_ref()
                    .expect("a side in states has a plan");
                let momk = moments_grid(
                    st.rk,
                    &st.respk,
                    (&st.skx, &st.sky),
                    &geo.fill[k],
                    plan,
                    (w, ref_sets.rows.0),
                );
                let m = plan.rw * plan.rh;
                let (mut sux, mut suy) = (vec![0.0f64; m], vec![0.0f64; m]);
                let (mut sen, mut sre) = (vec![0.0f64; m], vec![0.0f64; m]);
                let mut ssl = vec![[0.0f64; 4]; m];
                local_solve_spans(
                    &momk,
                    (&ux_in, &uy_in),
                    (plan.ox, plan.oy),
                    &plan.solve,
                    cfg.sigma_w,
                    cfg.shape_order,
                    ApSolve::default(),
                    no_texture,
                    SolveOut {
                        ux: &mut sux,
                        uy: &mut suy,
                        energy: &mut sen,
                        resid: &mut sre,
                        slope: &mut ssl,
                    },
                );
                for ry in 0..plan.rh {
                    let (a, b) = plan.solve[ry];
                    for rx in a..b {
                        let i = (ry + plan.oy) * w + rx + plan.ox;
                        if geo.sel[k][i] {
                            let j = ry * plan.rw + rx;
                            ux.data[i] = sux[j];
                            uy.data[i] = suy[j];
                            energy.data[i] = sen[j];
                            resid.data[i] = sre[j];
                            slope[i] = ssl[j];
                        }
                    }
                }
            }
        });
        conf = confidence_of(&energy.data, &resid.data, cfg);
    }
    let b = crate::phaseflow::border_zone(cfg);
    let conf_out: Vec<f64> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if reportable(x, y, x as f64 + ux.data[i], y as f64 + uy.data[i], w, h, b) {
                conf[i].clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let fl = Flow {
        w,
        h,
        ux: ux.data,
        uy: uy.data,
        conf: conf_out,
        resid: resid.data,
        slope,
    };
    let lines = geo.lines.clone();
    (fl, lines, geo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gabor::gabor_bank;
    use crate::phaseflow::{local_solve_into, Sums};

    /// A smooth speckle with a low-frequency part, so that the responses resemble those of an
    /// image and the spectrum is not flat.
    fn speckle(w: usize, h: usize) -> Image<f32> {
        let data = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                0.5 + 0.2 * (0.31 * x + 0.17 * y).sin() * (0.23 * y - 0.11 * x).cos()
                    + 0.05 * ((i.wrapping_mul(2654435761) % 1009) as f32 / 1009.0)
                    + 0.1 * (0.013 * x).sin()
            })
            .collect();
        Image::from_vec(w, h, data)
    }

    /// `filtered_rows` gives the rows of `Fft2::inverse` up to the rounding of `f32`, on every
    /// filter; the columns it leaves out do not show.
    #[test]
    fn filtered_rows_are_the_rows_of_the_full_inverse() {
        let (w, h) = (97, 64);
        let cfg = Config::default();
        let im = speckle(w, h);
        let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
        let bank = gabor_bank(ph, pw, &cfg);
        let fft = Fft2::new(ph, pw);
        let spec = padded_spectrum(&im, &fft);
        let active = active_columns(&bank);
        assert!(active.iter().all(|a| !a.is_empty()));
        for rows in [(0, h), (20, 40), (63, 64), (5, 5)] {
            for (g, act) in bank.filters.iter().zip(&active) {
                let got = filtered_rows(&spec, g, act, &fft, rows, w);
                assert_eq!(got.len(), (rows.1 - rows.0) * w);
                let mut full: Vec<Complex32> = spec.iter().zip(g).map(|(&a, &b)| a * b).collect();
                fft.inverse(&mut full);
                let top = full.iter().map(|v| v.norm()).fold(0.0f32, f32::max);
                for r in 0..rows.1 - rows.0 {
                    for x in 0..w {
                        let want = full[(rows.0 + r + PAD) * pw + x + PAD];
                        let d = (got[r * w + x] - want).norm();
                        assert!(d <= 3e-6 * top, "row {r} x {x}: {d} of {top}");
                    }
                }
            }
        }
    }

    /// The solve of the spans of a [`SidePlan`] on the plan's rectangle gives the bits of the solve
    /// of the whole image at the selected pixels, the pixels without texture included.
    #[test]
    fn the_solve_of_a_side_plan_has_the_bits_of_the_whole_image_solve() {
        let (w, h) = (90, 70);
        let sigma = 2.0;
        let r = gauss_kernel(sigma).len() / 2;
        // the selected pixels: a slanted strip, reaching the left and bottom edges
        let sel: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (y - 0.5 * x - 40.0).abs() < 6.0
            })
            .collect();
        // some "fill" that weighs nothing, as in the real solve: the far side
        let weight: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                if y - 0.5 * x - 40.0 < -6.0 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let v = |i: usize, s: f64| ((i as f64 * s).sin() + 1.5) * weight[i];
        let plane = |s: f64| Image::from_vec(w, h, (0..w * h).map(|i| v(i, s)).collect());
        let mut sums = Sums::new(w, h);
        sums.gxx = plane(0.37);
        sums.gyy = plane(0.91);
        sums.gxy = Image::from_vec(w, h, (0..w * h).map(|i| 0.2 * v(i, 0.13)).collect());
        sums.gxc = plane(1.31);
        sums.gyc = plane(0.77);
        sums.gcc = plane(2.1);
        sums.gw = plane(0.5);
        // a dead region inside the strip: no energy, so no texture
        for y in 40..44 {
            for x in 2..8 {
                let i = y * w + x;
                for p in [
                    &mut sums.gxx,
                    &mut sums.gyy,
                    &mut sums.gxy,
                    &mut sums.gxc,
                    &mut sums.gyc,
                    &mut sums.gcc,
                    &mut sums.gw,
                ] {
                    p.data[i] = 0.0;
                }
            }
        }
        let ux = plane(0.05);
        let uy = plane(0.07);
        let n = w * h;
        let (mut fx, mut fy, mut fe, mut fr) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut fs = vec![[0.0; 4]; n];
        local_solve_into(
            &sums,
            &ux,
            &uy,
            sigma,
            1,
            ApSolve::default(),
            SolveOut {
                ux: &mut fx,
                uy: &mut fy,
                energy: &mut fe,
                resid: &mut fr,
                slope: &mut fs,
            },
        );
        let thr = 1e-9 * crate::xdic::num::median(&fe) + 1e-30;

        let plan = SidePlan::new(&sel, w, h, r).expect("pixels selected");
        let crop = |im: &Image<f64>, spans: &[(usize, usize)]| {
            let mut d = vec![f64::NAN; plan.rw * plan.rh];
            for ry in 0..plan.rh {
                let (a, b) = spans[ry];
                for rx in a..b {
                    d[ry * plan.rw + rx] = im.data[(ry + plan.oy) * w + rx + plan.ox];
                }
            }
            Image::from_vec(plan.rw, plan.rh, d)
        };
        // NaN outside the moment spans: a read there would show
        let m = &plan.moments;
        let rect = Sums {
            gxx: crop(&sums.gxx, m),
            gxy: crop(&sums.gxy, m),
            gyy: crop(&sums.gyy, m),
            gxc: crop(&sums.gxc, m),
            gyc: crop(&sums.gyc, m),
            gcc: crop(&sums.gcc, m),
            gw: crop(&sums.gw, m),
        };
        let k = plan.rw * plan.rh;
        let (mut sx, mut sy, mut se, mut sr) =
            (vec![0.0; k], vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        let mut ss = vec![[0.0; 4]; k];
        local_solve_spans(
            &rect,
            (&ux, &uy),
            (plan.ox, plan.oy),
            &plan.solve,
            sigma,
            1,
            ApSolve::default(),
            thr,
            SolveOut {
                ux: &mut sx,
                uy: &mut sy,
                energy: &mut se,
                resid: &mut sr,
                slope: &mut ss,
            },
        );
        let mut seen = 0;
        for i in 0..n {
            if !sel[i] {
                continue;
            }
            let (x, y) = (i % w, i / w);
            let j = (y - plan.oy) * plan.rw + x - plan.ox;
            assert_eq!(sx[j].to_bits(), fx[i].to_bits(), "ux at {x},{y}");
            assert_eq!(sy[j].to_bits(), fy[i].to_bits(), "uy at {x},{y}");
            assert_eq!(se[j].to_bits(), fe[i].to_bits(), "energy at {x},{y}");
            assert_eq!(sr[j].to_bits(), fr[i].to_bits(), "resid at {x},{y}");
            assert_eq!(ss[j].map(f64::to_bits), fs[i].map(f64::to_bits));
            seen += 1;
        }
        assert!(seen > 500, "{seen} pixels compared");
    }
}

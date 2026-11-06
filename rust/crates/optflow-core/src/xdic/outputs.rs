//! Crack or band, the outputs of a leg and the carried geometry (`model_gain`, `is_crack`,
//! `advance_lines`, `lines_to_frame0`, `jump_along`, `finish` of `xdic.py`).

use rayon::prelude::*;

use super::geom::{polyline_fields, resample, vertex_normals, PolyFields};
use super::num::{bilinear_nearest, lstsq, median, nanmedian};
use super::params::*;
use super::solve::sample_f64;
use super::{Flow, JumpLine, Poly};
use crate::config::Config;
use crate::edt::{distance_to_invalid, large_components, within};
use crate::image::Image;
use crate::interp::BSpline;
use crate::strain::{displacement_gradient_frac, strain_fields_with, Gradient, Strain};

/// Per line: how much better the jump explains the images than the continuous baseline
/// (`model_gain`). `e0` and `e1` are median squared intensity residuals in units of the noise
/// (baseline and per-side field), `gain = (e0 - 1) / max(e1 - 1, 0.1)`, `mean` the ratio of the
/// mean residuals within `CRACK_MEAN_PX` of the line, `n` the number of pixels there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelGain {
    pub gain: f64,
    pub e0: f64,
    pub e1: f64,
    pub mean: f64,
    pub n: usize,
}

impl ModelGain {
    fn none(n: usize) -> ModelGain {
        ModelGain {
            gain: 0.0,
            e0: 0.0,
            e1: 0.0,
            mean: 0.0,
            n,
        }
    }
}

/// A line found by the detector is a crack when the per-side solve explains the images better
/// (see [`model_gain`]): `gain >= CRACK_GAIN` and `e0 >= CRACK_E0_MIN`, or `mean >= CRACK_MEAN`.
pub fn is_crack(m: &ModelGain) -> bool {
    (m.gain >= CRACK_GAIN && m.e0 >= CRACK_E0_MIN) || m.mean >= CRACK_MEAN
}

/// Median |J| = hypot(COD, CSD) of one line's jump samples (0 when none is finite).
pub fn jump_magnitude(j: &JumpLine) -> f64 {
    let m: Vec<f64> = j.cod.iter().zip(&j.csd).map(|(a, b)| a.hypot(*b)).collect();
    if m.iter().any(|v| v.is_finite()) {
        nanmedian(&m)
    } else {
        0.0
    }
}

/// `model_gain`: within `reach` px of each line (interior pixels, inside 4 sigma_w of the border
/// and the ROI), the median squared intensity difference between `i1` at X and `gain I2 + offset`
/// at X + u under the baseline `fl0` and the per-side `flx`, against the same median on the
/// confident baseline pixels off the strip.
pub fn model_gain(
    i1: &Image<f32>,
    c2: &BSpline<f32>,
    fl0: &Flow,
    flx: &Flow,
    lines: &[Poly],
    cfg: &Config,
    mask: Option<&Image<bool>>,
    reach: f64,
) -> Vec<ModelGain> {
    let (w, h) = (i1.w, i1.h);
    let n = w * h;
    let f = polyline_fields(lines, w, h);
    let b = 4.0 * cfg.sigma_w;
    let inner: Vec<bool> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (x.min(y).min(w - 1 - x).min(h - 1 - y) as f64) >= b && mask.is_none_or(|m| m.data[i])
        })
        .collect();
    let rb = cfg.xdic_rb_px();
    let off: Vec<usize> = (0..n)
        .filter(|&i| fl0.conf[i] >= 0.5 && inner[i] && !(f.interior[i] && f.phi[i].abs() <= rb))
        .collect();
    if off.len() <= 100 {
        return lines.iter().map(|_| ModelGain::none(0)).collect();
    }
    let at = |i: usize, u: &[f64], v: &[f64]| -> f64 {
        let (x, y) = ((i % w) as f64, (i / w) as f64);
        sample_f64(c2, x + u[i], y + v[i])
    };
    let rows: Vec<[f64; 2]> = off
        .iter()
        .map(|&i| [at(i, &fl0.ux, &fl0.uy), 1.0])
        .collect();
    let y: Vec<f64> = off.iter().map(|&i| f64::from(i1.data[i])).collect();
    let g = lstsq(&rows, &y);
    let res0: Vec<f64> = rows
        .iter()
        .zip(&y)
        .map(|(r, &t)| {
            let d = t - (r[0] * g[0] + r[1] * g[1]);
            d * d
        })
        .collect();
    let bg = median(&res0).max(1e-12);
    let resid = |i: usize, fl: &Flow| -> f64 {
        let d = f64::from(i1.data[i]) - (g[0] * at(i, &fl.ux, &fl.uy) + g[1]);
        d * d
    };
    (0..lines.len())
        .map(|li| {
            let pick = |r: f64| -> Vec<usize> {
                (0..n)
                    .filter(|&i| {
                        f.interior[i] && f.lid[i] == li as i32 && inner[i] && f.phi[i].abs() <= r
                    })
                    .collect()
            };
            let sel = pick(reach);
            if sel.len() < 20 {
                return ModelGain::none(sel.len());
            }
            let e0 = median(&sel.iter().map(|&i| resid(i, fl0)).collect::<Vec<_>>()) / bg;
            let e1 = median(&sel.iter().map(|&i| resid(i, flx)).collect::<Vec<_>>()) / bg;
            let sel2 = pick(CRACK_MEAN_PX);
            let m0 = sel2.iter().map(|&i| resid(i, fl0)).sum::<f64>() / sel2.len() as f64;
            let m1 = sel2.iter().map(|&i| resid(i, flx)).sum::<f64>() / sel2.len() as f64;
            ModelGain {
                gain: (e0 - 1.0) / (e1 - 1.0).max(0.1),
                e0,
                e1,
                mean: m0 / m1.max(1e-12),
                n: sel2.len(),
            }
        })
        .collect()
}

/// The polylines of a leg (reference coordinates) moved into the leg's target frame: every vertex
/// by the mean of the raw displacement 1.5 px to either side of it (`advance_lines`).
pub fn advance_lines(lines: &[Poly], ux: &[f64], uy: &[f64], w: usize, h: usize) -> Vec<Poly> {
    lines
        .iter()
        .map(|p| {
            let nrm = vertex_normals(p);
            p.iter()
                .zip(&nrm)
                .map(|(q, nn)| {
                    let side = |a: &[f64], sgn: f64| {
                        bilinear_nearest(
                            a,
                            w,
                            h,
                            q[0] + sgn * 1.5 * nn[0],
                            q[1] + sgn * 1.5 * nn[1],
                        )
                    };
                    let mvx = 0.5 * (side(ux, 1.0) + side(ux, -1.0));
                    let mvy = 0.5 * (side(uy, 1.0) + side(uy, -1.0));
                    [
                        q[0] + if mvx.is_nan() { 0.0 } else { mvx },
                        q[1] + if mvy.is_nan() { 0.0 } else { mvy },
                    ]
                })
                .collect()
        })
        .collect()
}

/// Polylines of a leg's reference mapped back to the frame-0 grid (`lines_to_frame0`; `u0r` =
/// frame 0 -> reference, raw, NaN where it leaves the image). A vertex `P` lies in the crack of
/// the reference; the forward map jumps over `P` at the crack of frame 0, found along the normal
/// through `P` as the 1 px cell where `g(s) = (X(s) + u0r(X(s)) - P) . n` goes from < 0 to >= 0.
/// `X0` is continuous in `P` and `u0r`: `P - u` within 1/2 px of the zero of `g`, the zero beyond
/// 1 px, linear in between. A vertex without a cell moves with the nearest vertex that has one.
pub fn lines_to_frame0(
    lines: &[Poly],
    u0r: Option<(&[f64], &[f64])>,
    w: usize,
    h: usize,
    half: f64,
) -> Vec<Poly> {
    let Some((u0, v0)) = u0r else {
        return lines.to_vec();
    };
    let nz = |a: &[f64]| a.iter().any(|&v| v.is_finite() && v != 0.0);
    if !(nz(u0) || nz(v0)) {
        return lines.to_vec();
    }
    let ax: Vec<f64> = u0
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    let ay: Vec<f64> = v0
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    let fin: Vec<f64> = u0
        .iter()
        .zip(v0)
        .map(|(a, b)| f64::from(u8::from(a.is_finite() && b.is_finite())))
        .collect();
    let reach = ax
        .iter()
        .zip(&ay)
        .map(|(a, b)| a.hypot(*b))
        .fold(0.0f64, f64::max)
        .ceil() as i64
        + 2;
    let ss: Vec<f64> = (-reach..=reach).map(|v| v as f64).collect();
    let ns = ss.len();
    let val = |a: &[f64], x: f64, y: f64| bilinear_nearest(a, w, h, x, y);
    lines
        .iter()
        .map(|p| {
            let nrm = vertex_normals(p);
            let tan: Vec<[f64; 2]> = nrm.iter().map(|v| [-v[1], v[0]]).collect();
            let nv = p.len();
            let mut a_pt: Vec<[f64; 2]> = p.clone();
            let mut found = vec![false; nv];
            let mut has_sample = vec![false; nv];
            let mut crossv = vec![0.0; nv];
            let mut um = vec![[0.0f64; 2]; nv];
            for _ in 0..2 {
                for v in 0..nv {
                    let (a, nn, pp) = (a_pt[v], nrm[v], p[v]);
                    let mut g = vec![0.0f64; ns];
                    let mut good = vec![false; ns];
                    for (k, &s) in ss.iter().enumerate() {
                        let (qx, qy) = (a[0] + s * nn[0], a[1] + s * nn[1]);
                        let (vx, vy, ok) = (val(&ax, qx, qy), val(&ay, qx, qy), val(&fin, qx, qy));
                        g[k] = (qx + vx - pp[0]) * nn[0] + (qy + vy - pp[1]) * nn[1];
                        good[k] = ok > 1.0 - 1e-9;
                    }
                    // the largest rise from < 0 to >= 0 between two good samples (first on a tie)
                    let (mut bi, mut brise) = (0usize, -1.0f64);
                    for k in 0..ns - 1 {
                        let rise = if g[k] < 0.0 && g[k + 1] >= 0.0 && good[k] && good[k + 1] {
                            g[k + 1] - g[k]
                        } else {
                            -1.0
                        };
                        if rise > brise {
                            brise = rise;
                            bi = k;
                        }
                    }
                    let fnd = brise > 0.0;
                    let mut near = 0usize;
                    let mut nb = f64::INFINITY;
                    for k in 0..ns {
                        let v = if good[k] { g[k].abs() } else { f64::INFINITY };
                        if v < nb {
                            nb = v;
                            near = k;
                        }
                    }
                    let cross = if fnd {
                        ss[bi] - g[bi] / brise
                    } else {
                        ss[near]
                    };
                    let c = [a[0] + cross * nn[0], a[1] + cross * nn[1]];
                    let mean_at = |arr: &[f64]| {
                        0.5 * (val(arr, c[0] + half * nn[0], c[1] + half * nn[1])
                            + val(arr, c[0] - half * nn[0], c[1] - half * nn[1]))
                    };
                    um[v] = [mean_at(&ax), mean_at(&ay)];
                    found[v] = fnd;
                    has_sample[v] = good.iter().any(|&b| b);
                    crossv[v] = cross;
                }
                // no measurement of its own: the motion of the nearest vertex with a cell (the
                // lower one on a tie), no motion for a line without a cell or a finite sample
                let cells: Vec<usize> = (0..nv).filter(|&v| found[v]).collect();
                for v in 0..nv {
                    if found[v] {
                        continue;
                    }
                    match cells.iter().min_by_key(|&&c| c.abs_diff(v)) {
                        Some(&c) => um[v] = um[c],
                        None if !has_sample[v] => um[v] = [0.0, 0.0],
                        None => {}
                    }
                }
                for v in 0..nv {
                    let d = um[v][0] * tan[v][0] + um[v][1] * tan[v][1];
                    a_pt[v] = [p[v][0] - d * tan[v][0], p[v][1] - d * tan[v][1]];
                }
            }
            (0..nv)
                .map(|v| {
                    let x = [p[v][0] - um[v][0], p[v][1] - um[v][1]];
                    let sx = (x[0] - a_pt[v][0]) * nrm[v][0] + (x[1] - a_pt[v][1]) * nrm[v][1];
                    let sc = if found[v] {
                        let d = sx - crossv[v];
                        crossv[v] + (2.0 * (1.0 - d.abs())).clamp(0.0, 1.0) * d
                    } else {
                        sx
                    };
                    [a_pt[v][0] + sc * nrm[v][0], a_pt[v][1] + sc * nrm[v][1]]
                })
                .collect()
        })
        .collect()
}

/// COD / CSD at points every `LINE_STEP` px along each polyline (`jump_along`): per line the
/// samples, the per-side affine extrapolation of the field to each, `J = u+ - u-`, `COD = J . n`,
/// `CSD = J . t` and their median over `+-JUMP_SMOOTH` px along the line. No line entries at all
/// when no pixel of the field is valid.
pub fn jump_along(
    mx: &[f64],
    my: &[f64],
    lines: &[Poly],
    phi: &[f64],
    w: usize,
    h: usize,
) -> Vec<JumpLine> {
    let valid: Vec<bool> = (0..w * h)
        .map(|i| mx[i].is_finite() && my[i].is_finite())
        .collect();
    if !valid.iter().any(|&b| b) {
        return Vec::new();
    }
    let r = JUMP_R;
    let k = r.ceil() as i64;
    let mut out = Vec::new();
    for p in lines {
        let (pts, nrm, sarc) = resample(p, LINE_STEP);
        let inside: Vec<usize> = (0..pts.len())
            .filter(|&i| {
                pts[i][0] >= 0.0
                    && pts[i][0] <= (w - 1) as f64
                    && pts[i][1] >= 0.0
                    && pts[i][1] <= (h - 1) as f64
            })
            .collect();
        let pts: Vec<[f64; 2]> = inside.iter().map(|&i| pts[i]).collect();
        let nrm: Vec<[f64; 2]> = inside.iter().map(|&i| nrm[i]).collect();
        let sarc: Vec<f64> = inside.iter().map(|&i| sarc[i]).collect();
        let n = pts.len();
        let est: Vec<([f64; 2], [f64; 2])> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (pi, ni) = (pts[i], nrm[i]);
                // the valid pixels within r of the sample
                let mut nb: Vec<(f64, f64, usize)> = Vec::new();
                for yy in (pi[1].floor() as i64 - k)..=(pi[1].ceil() as i64 + k) {
                    for xx in (pi[0].floor() as i64 - k)..=(pi[0].ceil() as i64 + k) {
                        if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                            continue;
                        }
                        let q = yy as usize * w + xx as usize;
                        if !valid[q] {
                            continue;
                        }
                        let (dx, dy) = (xx as f64 - pi[0], yy as f64 - pi[1]);
                        if dx * dx + dy * dy <= r * r {
                            nb.push((dx, dy, q));
                        }
                    }
                }
                let mut up = [f64::NAN; 2];
                let mut um = [f64::NAN; 2];
                if nb.is_empty() {
                    return (up, um);
                }
                for (plus, dst) in [(true, &mut up), (false, &mut um)] {
                    let m: Vec<(f64, f64, usize)> = nb
                        .iter()
                        .copied()
                        .filter(|&(dx, dy, q)| {
                            let dn = dx * ni[0] + dy * ni[1];
                            (phi[q] >= 0.0) == plus && dn.abs() >= JUMP_SKIP
                        })
                        .collect();
                    if m.len() < JUMP_MIN {
                        continue;
                    }
                    let a: Vec<[f64; 3]> = m
                        .iter()
                        .map(|&(dx, dy, _)| {
                            [1.0, dx * ni[0] + dy * ni[1], -dx * ni[1] + dy * ni[0]]
                        })
                        .collect();
                    for (c, field) in [mx, my].into_iter().enumerate() {
                        let val: Vec<f64> = m.iter().map(|&(_, _, q)| field[q]).collect();
                        let mut coef = lstsq(&a, &val);
                        let res: Vec<f64> = a
                            .iter()
                            .zip(&val)
                            .map(|(r, &v)| v - (r[0] * coef[0] + r[1] * coef[1] + r[2] * coef[2]))
                            .collect();
                        let med = median(&res);
                        let mad = median(&res.iter().map(|v| (v - med).abs()).collect::<Vec<_>>());
                        let keep: Vec<bool> = res
                            .iter()
                            .map(|v| v.abs() <= 3.0 * 1.4826 * mad + 0.05)
                            .collect();
                        let nk = keep.iter().filter(|&&b| b).count();
                        if nk >= JUMP_MIN && nk != keep.len() {
                            let ak: Vec<[f64; 3]> = a
                                .iter()
                                .zip(&keep)
                                .filter(|(_, &k)| k)
                                .map(|(r, _)| *r)
                                .collect();
                            let vk: Vec<f64> = val
                                .iter()
                                .zip(&keep)
                                .filter(|(_, &k)| k)
                                .map(|(v, _)| *v)
                                .collect();
                            coef = lstsq(&ak, &vk);
                        }
                        dst[c] = coef[0];
                    }
                }
                (up, um)
            })
            .collect();
        let u_plus = [
            est.iter().map(|e| e.0[0]).collect::<Vec<_>>(),
            est.iter().map(|e| e.0[1]).collect::<Vec<_>>(),
        ];
        let u_minus = [
            est.iter().map(|e| e.1[0]).collect::<Vec<_>>(),
            est.iter().map(|e| e.1[1]).collect::<Vec<_>>(),
        ];
        let mut cod_raw = vec![0.0; n];
        let mut csd_raw = vec![0.0; n];
        for i in 0..n {
            let jx = u_plus[0][i] - u_minus[0][i];
            let jy = u_plus[1][i] - u_minus[1][i];
            cod_raw[i] = jx * nrm[i][0] + jy * nrm[i][1];
            csd_raw[i] = -jx * nrm[i][1] + jy * nrm[i][0];
        }
        let mut cod = vec![f64::NAN; n];
        let mut csd = vec![f64::NAN; n];
        for i in 0..n {
            let m: Vec<usize> = (0..n)
                .filter(|&j| (sarc[j] - sarc[i]).abs() <= JUMP_SMOOTH)
                .collect();
            if m.iter().any(|&j| cod_raw[j].is_finite()) {
                cod[i] = nanmedian(&m.iter().map(|&j| cod_raw[j]).collect::<Vec<_>>());
                csd[i] = nanmedian(&m.iter().map(|&j| csd_raw[j]).collect::<Vec<_>>());
            }
        }
        out.push(JumpLine {
            pts,
            n: nrm,
            s: sarc,
            cod,
            csd,
            cod_raw,
            csd_raw,
            u_plus,
            u_minus,
        });
    }
    out
}

/// The validity and the COD / CSD samples of a leg's outputs, before the strain
/// (`finish` of `xdic.py` up to the strain).
pub struct FinishCore {
    pub w: usize,
    pub h: usize,
    /// Pixels that are invalid in the output.
    pub bad: Vec<bool>,
    pub geo: PolyFields,
    pub jumps: Vec<JumpLine>,
}

/// COD / CSD / normal maps on the line pixels (`mask` = the pixels within 1 px of a line).
#[derive(Clone, Debug)]
pub struct CrackMaps {
    pub mask: Image<bool>,
    pub nx: Image<f32>,
    pub ny: Image<f32>,
    pub cod: Image<f32>,
    pub csd: Image<f32>,
}

/// Validity of the output of a leg with crack lines `lines`: low confidence (`low`), outside the
/// ROI, within the guard of a guard source (large low-confidence components outside the strip
/// `S`, the ROI exterior) and within the guard of the border; and the jump samples of the lines
/// from the field `(ux, uy)` masked by that validity.
pub fn finish_core(
    ux: &[f64],
    uy: &[f64],
    low: &[bool],
    cfg: &Config,
    mask: Option<&Image<bool>>,
    lines: &[Poly],
    w: usize,
    h: usize,
) -> FinishCore {
    let n = w * h;
    let geo = polyline_fields(lines, w, h);
    let rb = cfg.xdic_rb_px();
    let strip: Vec<bool> = (0..n)
        .map(|i| geo.interior[i] && geo.phi[i].abs() <= rb)
        .collect();
    let mut bad: Vec<bool> = (0..n)
        .map(|i| low[i] || mask.is_some_and(|m| !m.data[i]))
        .collect();
    let guard = cfg.edge_guard_px();
    if guard > 0.0 {
        let cand = Image::from_vec(w, h, (0..n).map(|i| low[i] && !strip[i]).collect());
        let mut src = large_components(&cand, cfg.edge_guard_min_area_px());
        if let Some(m) = mask {
            for (s, &inside) in src.data.iter_mut().zip(&m.data) {
                *s |= !inside;
            }
        }
        if src.data.iter().any(|&s| s) {
            let d = distance_to_invalid(&Image::from_vec(
                w,
                h,
                src.data.iter().map(|&s| !s).collect(),
            ));
            for i in 0..n {
                bad[i] |= within(d.data[i], guard);
            }
        }
        for (i, b) in bad.iter_mut().enumerate() {
            let (x, y) = (i % w, i / w);
            if (x.min(y).min(w - 1 - x).min(h - 1 - y) as f64) < guard {
                *b = true;
            }
        }
    }
    let mx: Vec<f64> = (0..n)
        .map(|i| if bad[i] { f64::NAN } else { ux[i] })
        .collect();
    let my: Vec<f64> = (0..n)
        .map(|i| if bad[i] { f64::NAN } else { uy[i] })
        .collect();
    let jumps = jump_along(&mx, &my, lines, &geo.phi, w, h);
    FinishCore {
        w,
        h,
        bad,
        geo,
        jumps,
    }
}

/// The masked displacement, the strain (one-sided within `strain_win / 2` of the line: the field
/// of the other side masked, `min_frac` 1/4) and the crack maps of a [`FinishCore`]. `small`
/// asks for the small-strain maps as well.
pub fn finish_outputs(
    core: &FinishCore,
    ux: &[f64],
    uy: &[f64],
    cfg: &Config,
    small: bool,
) -> (Image<f32>, Image<f32>, Strain, CrackMaps) {
    let (w, h) = (core.w, core.h);
    let n = w * h;
    let geo = &core.geo;
    let mask_f = |keep: &dyn Fn(usize) -> bool| -> (Image<f32>, Image<f32>) {
        let f = |u: &[f64]| {
            Image::from_vec(
                w,
                h,
                (0..n)
                    .map(|i| {
                        if core.bad[i] || !keep(i) {
                            f32::NAN
                        } else {
                            u[i] as f32
                        }
                    })
                    .collect(),
            )
        };
        (f(ux), f(uy))
    };
    let (mx, my) = mask_f(&|_| true);
    let full = crate::strain::displacement_gradient(&mx, &my, cfg.strain_win);
    let near: Vec<bool> = (0..n)
        .map(|i| geo.interior[i] && geo.phi[i].abs() < cfg.strain_win as f64 / 2.0 + 1.0)
        .collect();
    let mut grad = full;
    for s in [1i8, -1] {
        let on = |i: usize| {
            if s > 0 {
                geo.phi[i] >= 0.0
            } else {
                geo.phi[i] < 0.0
            }
        };
        let (kx, ky) = mask_f(&|i| on(i) || !geo.interior[i]);
        let gs = displacement_gradient_frac(&kx, &ky, cfg.strain_win, (1, 4));
        let pick = |dst: &mut Image<f32>, src: &Image<f32>| {
            for i in 0..n {
                if near[i] && on(i) {
                    dst.data[i] = src.data[i];
                }
            }
        };
        let Gradient {
            dudx,
            dudy,
            dvdx,
            dvdy,
        } = &mut grad;
        pick(dudx, &gs.dudx);
        pick(dudy, &gs.dudy);
        pick(dvdx, &gs.dvdx);
        pick(dvdy, &gs.dvdy);
    }
    let strain = strain_fields_with(&grad, small);
    let line_px: Vec<bool> = (0..n)
        .map(|i| geo.interior[i] && geo.phi[i].abs() <= 1.0)
        .collect();
    let mut cod_m = vec![f32::NAN; n];
    let mut csd_m = vec![f32::NAN; n];
    let samples: Vec<([f64; 2], f64, f64)> = core
        .jumps
        .iter()
        .flat_map(|j| (0..j.pts.len()).map(move |k| (j.pts[k], j.cod[k], j.csd[k])))
        .collect();
    if !samples.is_empty() {
        let lp: Vec<usize> = (0..n).filter(|&i| line_px[i]).collect();
        let nearest: Vec<usize> = lp
            .par_iter()
            .map(|&i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let mut best = (f64::INFINITY, 0usize);
                for (k, s) in samples.iter().enumerate() {
                    let d = (s.0[0] - x).powi(2) + (s.0[1] - y).powi(2);
                    if d < best.0 {
                        best = (d, k);
                    }
                }
                best.1
            })
            .collect();
        for (&i, &k) in lp.iter().zip(&nearest) {
            cod_m[i] = samples[k].1 as f32;
            csd_m[i] = samples[k].2 as f32;
        }
    }
    let nanf = |v: &[f64]| -> Image<f32> {
        Image::from_vec(
            w,
            h,
            (0..n)
                .map(|i| if line_px[i] { v[i] as f32 } else { f32::NAN })
                .collect(),
        )
    };
    let (nx, ny) = (nanf(&geo.nx), nanf(&geo.ny));
    let crack = CrackMaps {
        mask: Image::from_vec(w, h, line_px),
        nx,
        ny,
        cod: Image::from_vec(w, h, cod_m),
        csd: Image::from_vec(w, h, csd_m),
    };
    (mx, my, strain, crack)
}

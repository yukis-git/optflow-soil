//! The field-jump detector of `crack_model = "xdic"` (`jump_test`, `_lines_from_points` and
//! `detect_lines` of `xdic.py`; the module doc of `xdic.py` describes the method).

use rayon::prelude::*;

use super::geom::{extend_to_border, length, merge_collinear};
use super::num::{dilate3, inv3, label, major_axis, median, median3_nearest};
use super::params::*;
use super::Poly;
use crate::config::Config;
use crate::edt::{distance_to_invalid, large_components, within};
use crate::image::Image;

/// Offsets of the integer points of the disk of radius `r` (`_disk_offsets`).
fn disk_offsets(r: f64) -> (Vec<i64>, Vec<i64>) {
    let k = r.ceil() as i64;
    let (mut ox, mut oy) = (Vec::new(), Vec::new());
    for y in -k..=k {
        for x in -k..=k {
            if ((x * x + y * y) as f64) <= r * r {
                ox.push(x);
                oy.push(y);
            }
        }
    }
    (ox, oy)
}

/// Result of [`jump_test`] at each point: `j` = u+ - u- (NaN where no direction could be
/// tested), `z` and `theta` of the direction with the largest z.
pub struct JumpTest {
    pub j: Vec<[f64; 2]>,
    pub z: Vec<f64>,
    pub theta: Vec<f64>,
}

/// Sums over the valid pixels of one half disk: count and first/second moments of the offsets
/// (0..6), `sum u*(1, ox, oy)` (6..9) and `sum v*(1, ox, oy)` (9..12), `sum u^2`, `sum v^2`.
type Half = [f64; 14];

/// The fit of both halves per direction from their sums `acc` (`t`: the + half of direction `t`,
/// `ntheta + t`: its - half) and the direction with the largest z: `(J, z, theta)`, `(None, NaN,
/// NaN)` where no direction can be tested.
#[allow(clippy::too_many_arguments)]
fn best_direction(
    acc: &[Half],
    (cs, sn, th): (&[f64], &[f64], &[f64]),
    r: f64,
    minpts: f64,
    kappa: f64,
) -> (Option<[f64; 2]>, f64, f64) {
    let ntheta = th.len();
    // per half: the affine fit's constant term and its variance for both components
    let fit = |a: &Half| -> ([f64; 2], [f64; 2], bool, f64) {
        let cnt = a[0];
        let ok = cnt >= minpts;
        let mut m = if ok {
            [[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]]
        } else {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        };
        for (i, row) in m.iter_mut().enumerate() {
            row[i] += 1e-9;
        }
        let minv = inv3(&m);
        let mut a0 = [0.0; 2];
        let mut var = [0.0; 2];
        for c in 0..2 {
            let b = [a[6 + 3 * c], a[7 + 3 * c], a[8 + 3 * c]];
            let uu = a[12 + c];
            let mut coef = [0.0; 3];
            for i in 0..3 {
                coef[i] = minv[i][0] * b[0] + minv[i][1] * b[1] + minv[i][2] * b[2];
            }
            let ss = (uu - (coef[0] * b[0] + coef[1] * b[1] + coef[2] * b[2])).max(0.0);
            let s2 = ss / (cnt - 3.0).max(1.0);
            a0[c] = coef[0];
            var[c] = s2 * minv[0][0];
        }
        (a0, var, ok, cnt)
    };
    let mut best: Option<(f64, usize, [f64; 2])> = None;
    for t in 0..ntheta {
        let (ap, am) = (&acc[t], &acc[ntheta + t]);
        let (a0p, vp, okp, cp) = fit(ap);
        let (a0m, vm, okm, cm) = fit(am);
        let mt = |a: &Half, cnt: f64| (-sn[t] * a[1] + cs[t] * a[2]) / cnt.max(1.0);
        let ok = okp && okm && mt(ap, cp).abs() <= r / 4.0 && mt(am, cm).abs() <= r / 4.0;
        let jj = [a0p[0] - a0m[0], a0p[1] - a0m[1]];
        let vj = kappa * 0.5 * ((vp[0] + vm[0]) + (vp[1] + vm[1]));
        let zz = if ok {
            jj[0].hypot(jj[1]) / (vj + 1e-12).sqrt()
        } else {
            -1.0
        };
        // np.argmax: the first maximum
        if best.is_none_or(|b| zz > b.0) {
            best = Some((zz, t, jj));
        }
    }
    match best {
        Some((zz, t, jj)) if zz >= 0.0 => (Some(jj), zz, th[t]),
        _ => (None, f64::NAN, f64::NAN),
    }
}

fn directions(ntheta: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let th: Vec<f64> = (0..ntheta)
        .map(|k| std::f64::consts::PI * k as f64 / ntheta as f64)
        .collect();
    let (cs, sn) = th.iter().map(|t| (t.cos(), t.sin())).unzip();
    (cs, sn, th)
}

/// Half-disk jump test at the integer points `(px, py)`: the affine model fitted by least
/// squares to each half of the disk, J = u+ - u- at the point, z = |J| / sigma_J, over `ntheta`
/// normals, keeping the one with the largest z. `kappa` inflates the variance for the spatially
/// correlated errors. A direction is tested only if both halves hold `minpts` valid pixels
/// whose mean tangential offset from the point is within `r / 4`.
#[allow(clippy::too_many_arguments)]
pub fn jump_test(
    ux: &[f64],
    uy: &[f64],
    valid: &[bool],
    w: usize,
    h: usize,
    px: &[i64],
    py: &[i64],
    r: f64,
    ntheta: usize,
    guard: f64,
    minpts: f64,
    kappa: f64,
) -> JumpTest {
    let (ox, oy) = disk_offsets(r);
    let no = ox.len();
    let (cs, sn, th) = directions(ntheta);
    // side[o * ntheta + t]: +1 in the + half (s > guard), -1 in the - half (s < -guard), else 0
    let mut side = vec![0i8; no * ntheta];
    for o in 0..no {
        for t in 0..ntheta {
            let s = ox[o] as f64 * cs[t] + oy[o] as f64 * sn[t];
            side[o * ntheta + t] = if s > guard {
                1
            } else if s < -guard {
                -1
            } else {
                0
            };
        }
    }
    let rows: Vec<(Option<[f64; 2]>, f64, f64)> = px
        .par_iter()
        .zip(py.par_iter())
        .map(|(&cx, &cy)| {
            let mut acc = vec![[0.0f64; 14]; 2 * ntheta];
            for o in 0..no {
                let (qx, qy) = (cx + ox[o], cy + oy[o]);
                if qx < 0 || qx >= w as i64 || qy < 0 || qy >= h as i64 {
                    continue;
                }
                let q = qy as usize * w + qx as usize;
                if !valid[q] {
                    continue;
                }
                let (u, v) = (ux[q], uy[q]);
                let (fx, fy) = (ox[o] as f64, oy[o] as f64);
                let row = [1.0, fx, fy, fx * fx, fx * fy, fy * fy];
                for t in 0..ntheta {
                    let sd = side[o * ntheta + t];
                    if sd == 0 {
                        continue;
                    }
                    let a: &mut Half = &mut acc[if sd > 0 { t } else { ntheta + t }];
                    for (s, r) in a[..6].iter_mut().zip(&row) {
                        *s += r;
                    }
                    for k in 0..3 {
                        a[6 + k] += u * row[k];
                        a[9 + k] += v * row[k];
                    }
                    a[12] += u * u;
                    a[13] += v * v;
                }
            }
            best_direction(&acc, (&cs, &sn, &th), r, minpts, kappa)
        })
        .collect();
    JumpTest {
        j: rows
            .iter()
            .map(|r| r.0.unwrap_or([f64::NAN, f64::NAN]))
            .collect(),
        z: rows.iter().map(|r| r.1).collect(),
        theta: rows.iter().map(|r| r.2).collect(),
    }
}

/// [`jump_test`] with the sums of a half disk taken row by row from running sums along the image
/// rows, many times faster on many points. The sums differ from `jump_test`'s by rounding, so this
/// only screens the points: see [`detect_lines`]. The second result marks the points with a half
/// whose valid pixels lie almost on a line: their fit is rounding, and `z` means nothing.
#[allow(clippy::too_many_arguments)]
fn jump_test_screen(
    ux: &[f64],
    uy: &[f64],
    valid: &[bool],
    w: usize,
    h: usize,
    px: &[i64],
    py: &[i64],
    r: f64,
    ntheta: usize,
    guard: f64,
    minpts: f64,
    kappa: f64,
) -> (JumpTest, Vec<bool>) {
    let (ox, oy) = disk_offsets(r);
    let (cs, sn, th) = directions(ntheta);
    // per half (t: the + half of direction t, s > guard; ntheta + t: the - half, s < -guard) its
    // pixels row by row: a half disk meets a row in one run
    let mut runs: Vec<Vec<(i64, i64, i64)>> = vec![Vec::new(); 2 * ntheta];
    for t in 0..ntheta {
        for (&x, &y) in ox.iter().zip(&oy) {
            let s = x as f64 * cs[t] + y as f64 * sn[t];
            let half = if s > guard {
                t
            } else if s < -guard {
                ntheta + t
            } else {
                continue;
            };
            match runs[half].last_mut() {
                Some(run) if run.0 == y => run.2 = x + 1,
                _ => runs[half].push((y, x, x + 1)),
            }
        }
    }
    // per row the running sums over the valid pixels, of (1, x, x^2, u, u x, v, v x, u^2, v^2)
    let mut running = vec![[0.0f64; 9]; (w + 1) * h];
    running
        .par_chunks_mut(w + 1)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let q = y * w + x;
                let mut next = row[x];
                if valid[q] {
                    let (u, v, xf) = (ux[q], uy[q], x as f64);
                    let d = [1.0, xf, xf * xf, u, u * xf, v, v * xf, u * u, v * v];
                    for (s, d) in next.iter_mut().zip(d) {
                        *s += d;
                    }
                }
                row[x + 1] = next;
            }
        });
    let rows: Vec<((Option<[f64; 2]>, f64, f64), bool)> = px
        .par_iter()
        .zip(py.par_iter())
        .map(|(&cx, &cy)| {
            let mut acc = vec![[0.0f64; 14]; 2 * ntheta];
            let c = cx as f64;
            for (a, half) in acc.iter_mut().zip(&runs) {
                for &(dy, xa, xb) in half {
                    let qy = cy + dy;
                    if qy < 0 || qy >= h as i64 {
                        continue;
                    }
                    let (qa, qb) = ((cx + xa).max(0) as usize, (cx + xb).min(w as i64));
                    if qb <= qa as i64 {
                        continue;
                    }
                    let at = qy as usize * (w + 1);
                    let (lo, hi) = (&running[at + qa], &running[at + qb as usize]);
                    let d: [f64; 9] = std::array::from_fn(|k| hi[k] - lo[k]);
                    let fy = dy as f64;
                    let sx = d[1] - c * d[0];
                    a[0] += d[0];
                    a[1] += sx;
                    a[2] += fy * d[0];
                    a[3] += d[2] - 2.0 * c * d[1] + c * c * d[0];
                    a[4] += fy * sx;
                    a[5] += fy * fy * d[0];
                    a[6] += d[3];
                    a[7] += d[4] - c * d[3];
                    a[8] += fy * d[3];
                    a[9] += d[5];
                    a[10] += d[6] - c * d[5];
                    a[11] += fy * d[5];
                    a[12] += d[7];
                    a[13] += d[8];
                }
            }
            let flat = acc.iter().any(|a| {
                let n = a[0];
                let (mx, my) = (a[1] / n, a[2] / n);
                let (cxx, cyy, cxy) = (a[3] / n - mx * mx, a[5] / n - my * my, a[4] / n - mx * my);
                n >= minpts && cxx * cyy - cxy * cxy < 1.0
            });
            (
                best_direction(&acc, (&cs, &sn, &th), r, minpts, kappa),
                flat,
            )
        })
        .collect();
    (
        JumpTest {
            j: rows
                .iter()
                .map(|r| r.0 .0.unwrap_or([f64::NAN, f64::NAN]))
                .collect(),
            z: rows.iter().map(|r| r.0 .1).collect(),
            theta: rows.iter().map(|r| r.0 .2).collect(),
        },
        rows.iter().map(|r| r.1).collect(),
    )
}

/// `np.arange(start, stop, step)` as numpy fills it: `first + i * (second - first)`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    let delta = (start + step) - start;
    (0..n).map(|i| start + i as f64 * delta).collect()
}

/// Polylines through clustered crack points (`_lines_from_points`). `hole`: the invalid pixels of
/// the baseline, `(x, y)` lists; where a bin has invalid pixels within `DET_R` of the cluster
/// axis the vertex is their median normal offset.
pub fn lines_from_points(
    px: &[i64],
    py: &[i64],
    z: &[f64],
    w: usize,
    h: usize,
    hole: &[(i64, i64)],
) -> Vec<Poly> {
    if px.is_empty() {
        return Vec::new();
    }
    let step = DET_STEP as i64;
    let (gw, gh) = (w / DET_STEP + 1, h / DET_STEP + 1);
    let mut grid = vec![false; gw * gh];
    let cell: Vec<usize> = px
        .iter()
        .zip(py)
        .map(|(&x, &y)| (y / step) as usize * gw + (x / step) as usize)
        .collect();
    for &c in &cell {
        grid[c] = true;
    }
    let (lab, nlab) = label(&dilate3(&grid, gw, gh, 2), gw, gh, true);
    let mut lines: Vec<Poly> = Vec::new();
    for c in 1..=nlab {
        let sel: Vec<usize> = (0..px.len()).filter(|&i| lab[cell[i]] == c).collect();
        if sel.len() < 4 {
            continue;
        }
        let x: Vec<f64> = sel.iter().map(|&i| px[i] as f64).collect();
        let y: Vec<f64> = sel.iter().map(|&i| py[i] as f64).collect();
        let wt: Vec<f64> = sel.iter().map(|&i| z[i].min(50.0)).collect();
        let sw: f64 = wt.iter().sum();
        let mx = x.iter().zip(&wt).map(|(a, b)| a * b).sum::<f64>() / sw;
        let my = y.iter().zip(&wt).map(|(a, b)| a * b).sum::<f64>() / sw;
        // np.cov(..., aweights=w): sum w (x - m)(x - m)^T / (sum w - sum w^2 / sum w)
        let fact = sw - wt.iter().map(|a| a * a).sum::<f64>() / sw;
        let (mut cxx, mut cxy, mut cyy) = (0.0, 0.0, 0.0);
        for ((&a, &b), &wi) in x.iter().zip(&y).zip(&wt) {
            let (da, db) = (a - mx, b - my);
            cxx += da * wi * da;
            cxy += da * wi * db;
            cyy += db * wi * db;
        }
        let t = major_axis(cxx / fact, cxy / fact, cyy / fact);
        let nrm = [-t[1], t[0]];
        let tau: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| (a - mx) * t[0] + (b - my) * t[1])
            .collect();
        let nu: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| (a - mx) * nrm[0] + (b - my) * nrm[1])
            .collect();
        let htau: Vec<f64> = hole
            .iter()
            .map(|&(a, b)| (a as f64 - mx) * t[0] + (b as f64 - my) * t[1])
            .collect();
        let hnu: Vec<f64> = hole
            .iter()
            .map(|&(a, b)| (a as f64 - mx) * nrm[0] + (b as f64 - my) * nrm[1])
            .collect();
        let tmin = tau.iter().copied().fold(f64::INFINITY, f64::min);
        let tmax = tau.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let edges = arange(tmin, tmax + DET_BIN, DET_BIN);
        let mut verts: Vec<(f64, f64)> = Vec::new();
        for pair in edges.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let m: Vec<usize> = (0..tau.len())
                .filter(|&i| tau[i] >= a && tau[i] < b)
                .collect();
            if m.is_empty() {
                continue;
            }
            let zm: Vec<f64> = m.iter().map(|&i| wt[i]).collect();
            let zmax = zm.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mut sn, mut sd) = (0.0, 0.0);
            for (k, &i) in m.iter().enumerate() {
                if zm[k] >= 0.7 * zmax {
                    sn += nu[i] * zm[k];
                    sd += zm[k];
                }
            }
            let mut nu0 = sn / sd;
            let hm: Vec<f64> = (0..htau.len())
                .filter(|&i| htau[i] >= a && htau[i] < b && (hnu[i] - nu0).abs() <= DET_R)
                .map(|i| hnu[i])
                .collect();
            if hm.len() as f64 >= 0.25 * DET_BIN * DET_BIN {
                nu0 = median(&hm);
            }
            verts.push((0.5 * (a + b), nu0));
        }
        if verts.len() < 2 {
            continue;
        }
        if verts.len() >= 3 {
            let f = median3_nearest(&verts.iter().map(|v| v.1).collect::<Vec<_>>());
            for (v, fv) in verts.iter_mut().zip(f) {
                v.1 = fv;
            }
        }
        lines.push(
            verts
                .iter()
                .map(|&(tv, nv)| [mx + tv * t[0] + nv * nrm[0], my + tv * t[1] + nv * nrm[1]])
                .collect(),
        );
    }
    let lines = merge_collinear(lines, 3.0 * DET_R, 0.95, 8.0);
    lines
        .into_iter()
        .filter(|p| length(p) >= 2.0 * DET_R)
        .map(|p| extend_to_border(&p, w, h, 2.0 * DET_R))
        .collect()
}

/// What `detect_lines` saw besides the lines (for the tests and the reports).
#[derive(Clone, Debug, Default)]
pub struct DetectInfo {
    pub n_cand: usize,
    pub n_acc: usize,
}

/// Crack polylines of the baseline result: `mx`, `my` are NaN where invalid, `gmax` is its max
/// shear strain, `mask` the ROI (`detect_lines`).
pub fn detect_lines(
    mx: &[f64],
    my: &[f64],
    gmax: Option<&[f32]>,
    w: usize,
    h: usize,
    cfg: &Config,
    mask: Option<&Image<bool>>,
) -> (Vec<Poly>, DetectInfo) {
    let n = w * h;
    let valid: Vec<bool> = (0..n)
        .map(|i| mx[i].is_finite() && my[i].is_finite())
        .collect();
    let roi = |i: usize| mask.is_none_or(|m| m.data[i]);
    let hole_in = Image::from_vec(w, h, (0..n).map(|i| !valid[i] && roi(i)).collect());
    let holes = large_components(&hole_in, cfg.edge_guard_min_area_px());
    let mut cand = vec![false; n];
    if holes.data.iter().any(|&b| b) {
        let d = distance_to_invalid(&Image::from_vec(
            w,
            h,
            holes.data.iter().map(|&b| !b).collect(),
        ));
        for i in 0..n {
            cand[i] = within(d.data[i], DET_GMAX);
        }
    }
    if let Some(g) = gmax {
        for i in 0..n {
            let gv = if g[i].is_nan() { 0.0 } else { f64::from(g[i]) };
            cand[i] |= valid[i] && gv > DET_GMAX_STRAIN;
        }
    }
    let bw = crate::phaseflow::border_zone(cfg) + cfg.edge_guard_px();
    let mut inner: Vec<bool> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (x.min(y).min(w - 1 - x).min(h - 1 - y) as f64) >= bw
        })
        .collect();
    if let Some(m) = mask {
        if !m.data.iter().all(|&b| b) {
            // the guard band along the ROI edge is a hole for the same reason
            let d = distance_to_invalid(m);
            for i in 0..n {
                inner[i] &= !within(d.data[i], bw);
            }
        }
    }
    let (mut px, mut py) = (Vec::new(), Vec::new());
    for y in (0..h).step_by(DET_STEP) {
        for x in (0..w).step_by(DET_STEP) {
            let i = y * w + x;
            if cand[i] && roi(i) && inner[i] {
                px.push(x as i64);
                py.push(y as i64);
            }
        }
    }
    let kappa = 4.0 * std::f64::consts::PI * cfg.sigma_w * cfg.sigma_w;
    // the screen decides which points cannot pass (with a margin of 2 %, its worst difference on the real videos is 7e-5); the
    // points that may pass go through `jump_test`, the only one whose values are used
    let (screen, flat) = jump_test_screen(
        mx,
        my,
        &valid,
        w,
        h,
        &px,
        &py,
        DET_R,
        DET_NTHETA,
        DET_LINE_GUARD,
        DET_MINPTS,
        kappa,
    );
    let margin = 1.0 - 0.02;
    let (spx, spy): (Vec<i64>, Vec<i64>) = (0..px.len())
        .filter(|&i| {
            flat[i]
                || (screen.z[i].is_finite()
                    && screen.z[i] >= DET_ZMIN * margin
                    && screen.j[i][0].hypot(screen.j[i][1]) >= DET_JMIN * margin)
        })
        .map(|i| (px[i], py[i]))
        .unzip();
    let jt = jump_test(
        mx,
        my,
        &valid,
        w,
        h,
        &spx,
        &spy,
        DET_R,
        DET_NTHETA,
        DET_LINE_GUARD,
        DET_MINPTS,
        kappa,
    );
    let acc: Vec<usize> = (0..spx.len())
        .filter(|&i| {
            jt.z[i].is_finite() && jt.z[i] >= DET_ZMIN && jt.j[i][0].hypot(jt.j[i][1]) >= DET_JMIN
        })
        .collect();
    let apx: Vec<i64> = acc.iter().map(|&i| spx[i]).collect();
    let apy: Vec<i64> = acc.iter().map(|&i| spy[i]).collect();
    let az: Vec<f64> = acc.iter().map(|&i| jt.z[i]).collect();
    let hole: Vec<(i64, i64)> = (0..n)
        .filter(|&i| holes.data[i] && inner[i])
        .map(|i| ((i % w) as i64, (i / w) as i64))
        .collect();
    let lines = lines_from_points(&apx, &apy, &az, w, h, &hole);
    (
        lines,
        DetectInfo {
            n_cand: px.len(),
            n_acc: acc.len(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The screen of [`detect_lines`] agrees with [`jump_test`]: the same points testable, `z` and
    /// `J` far inside the margin of the screen, on a noisy field with a jump and a hole.
    #[test]
    fn the_screen_agrees_with_the_jump_test() {
        let (w, h) = (160, 140);
        let n = w * h;
        let mut seed = 7u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.02
        };
        let field = |x: f64, y: f64, a: f64, j: f64| {
            a * x * 0.01 + 3.0 * a + y * 0.002 + if y - 0.3 * x > 60.0 { j } else { 0.0 }
        };
        let (mut ux, mut uy) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            ux[i] = field(x, y, 7.0, 1.5) + noise();
            uy[i] = field(x, y, -4.0, -0.7) + noise();
        }
        let valid: Vec<bool> = (0..n)
            .map(|i| !((30..60).contains(&(i % w)) && (20..50).contains(&(i / w))))
            .collect();
        let (mut px, mut py) = (Vec::new(), Vec::new());
        for y in (0..h as i64).step_by(3) {
            for x in (0..w as i64).step_by(3) {
                px.push(x);
                py.push(y);
            }
        }
        let args = (
            DET_R,
            DET_NTHETA,
            DET_LINE_GUARD,
            DET_MINPTS,
            4.0 * std::f64::consts::PI * 16.0,
        );
        let a = jump_test(
            &ux, &uy, &valid, w, h, &px, &py, args.0, args.1, args.2, args.3, args.4,
        );
        let (b, flat) = jump_test_screen(
            &ux, &uy, &valid, w, h, &px, &py, args.0, args.1, args.2, args.3, args.4,
        );
        let mut tested = 0;
        for i in 0..px.len() {
            assert_eq!(a.z[i].is_nan(), b.z[i].is_nan(), "point {i}");
            if a.z[i].is_nan() || flat[i] {
                continue;
            }
            tested += 1;
            // the margin of the screen is 2 % of the thresholds
            let dz = (a.z[i] - b.z[i]).abs();
            assert!(dz <= 0.01 * a.z[i].max(DET_ZMIN), "z at {i}: {dz}");
            let (ja, jb) = (a.j[i][0].hypot(a.j[i][1]), b.j[i][0].hypot(b.j[i][1]));
            assert!((ja - jb).abs() <= 0.01 * ja.max(DET_JMIN), "J at {i}");
        }
        assert!(flat.iter().any(|&f| f), "the field has a clipped disk");
        assert!(tested > 500, "{tested} points tested");
    }
}

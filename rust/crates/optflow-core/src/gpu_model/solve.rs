//! The accumulate and the local solve of an inner iteration in the GPU's arithmetic (GPU spec
//! §4.2, §4.3): anchor-free f32 sums per pixel, window moments of the anchor-free part in f32, and
//! per block of output pixels an affine anchor `L_B` against which the displacement enters the
//! right-hand side as a small deviation. Everything is f32 except the displacement and the
//! coordinates (Df64).

use std::f64::consts::PI;

use num_complex::Complex32;
use rayon::prelude::*;

use super::atan2f::atan2f;
use super::df64::Df64;
use super::field::{axis_taps_df64, sample_centred};
use super::BLOCK;
use crate::gabor::PAD;
use crate::gpu_support::{moment_taps, plan_moments};
use crate::image::Image;
use crate::phaseflow::{Aperture, RefPack, Responses, APERTURE_TOL, RIDGE};

/// The smoothed field of the outer iteration: `s` and the (centred) coefficients of its spline.
pub(crate) struct ModelField<'a> {
    pub sx: &'a Image<f32>,
    pub sy: &'a Image<f32>,
    pub csx: &'a Image<f32>,
    pub csy: &'a Image<f32>,
}

/// Anchor-free per-pixel sums over the filters: `[gxx, gxy, gyy, hx, hy, s_phiphi, s_w]` with
/// `G = sum w g g^T`, `h = sum w g dphi`, `s_phiphi = sum w dphi^2`, `s_w = sum w`; all 7 NaN where
/// the sampling coordinate is not finite. `ux`, `uy` are the displacement of the pixels.
pub(crate) fn pixel_sums(
    reference: &RefPack,
    responses: &Responses,
    field: &ModelField,
    ux: &[Df64],
    uy: &[Df64],
) -> Vec<[f32; 7]> {
    let (w, h) = (field.sx.w, field.sx.h);
    let n = responses.n_filters;
    assert_eq!(reference.n_filters, n);
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let mut out = vec![[0.0f32; 7]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut acc = vec![[0.0f32; 2]; n];
        let mut sampled = vec![[0.0f32; 2]; n];
        for (x, o) in row.iter_mut().enumerate() {
            let i = y * w + x;
            // X' + s(X') = X + u(X) by two fixed-point steps, in Df64
            let tx = ux[i].add_f32(x as f32);
            let ty = uy[i].add_f32(y as f32);
            let mut qx = tx.add_f32(-field.sx.data[i]);
            let mut qy = ty.add_f32(-field.sy.data[i]);
            for _ in 0..2 {
                let dx = sample_centred(field.csx, qx, qy);
                let dy = sample_centred(field.csy, qx, qy);
                qx = tx.sub(dx);
                qy = ty.sub(dy);
            }
            let (px, py) = (qx.add_f32(PAD as f32), qy.add_f32(PAD as f32));
            if !(px.hi.is_finite() && py.hi.is_finite()) {
                *o = [f32::NAN; 7];
                continue;
            }
            let (xi, wx) = axis_taps_df64(px, pw);
            let (yi, wy) = axis_taps_df64(py, ph);
            sampled.fill([0.0; 2]);
            for (&yj, &wyj) in yi.iter().zip(&wy) {
                let line = &responses.data[yj * pw * n..(yj + 1) * pw * n];
                let c0: &[Complex32] = &line[xi[0] * n..(xi[0] + 1) * n];
                for (a, c) in acc.iter_mut().zip(c0) {
                    *a = [wx[0] * c.re, wx[0] * c.im];
                }
                for (&xk, &wxk) in xi.iter().zip(&wx).skip(1) {
                    for (a, c) in acc.iter_mut().zip(&line[xk * n..(xk + 1) * n]) {
                        a[0] += wxk * c.re;
                        a[1] += wxk * c.im;
                    }
                }
                for (s, a) in sampled.iter_mut().zip(&acc) {
                    s[0] += wyj * a[0];
                    s[1] += wyj * a[1];
                }
            }
            let rp = &reference.data[i * n * 5..(i + 1) * n * 5];
            let mut sums = [0.0f32; 7];
            for (k, b) in sampled.iter().enumerate() {
                let (a_re, a_im) = (rp[5 * k], rp[5 * k + 1]);
                let [b_re, b_im] = *b;
                let num = a_im * b_re - a_re * b_im;
                let den = a_re * b_re + a_im * b_im;
                let mag = (b_re * b_re + b_im * b_im).sqrt();
                let dphi = atan2f(num, den);
                let wk = rp[5 * k + 4] * mag;
                let (gx, gy) = (rp[5 * k + 2], rp[5 * k + 3]);
                sums[0] += (wk * gx) * gx;
                sums[1] += (wk * gx) * gy;
                sums[2] += (wk * gy) * gy;
                sums[3] += (wk * gx) * dphi;
                sums[4] += (wk * gy) * dphi;
                sums[5] += (wk * dphi) * dphi;
                sums[6] += wk;
            }
            *o = sums;
        }
    });
    out
}

/// Where [`local_solve_blocks`] writes, one value per pixel each (row-major); every value is
/// written and none is read.
pub struct ModelSolveOut<'a> {
    pub ux: &'a mut [Df64],
    pub uy: &'a mut [Df64],
    pub energy: &'a mut [f32],
    pub resid: &'a mut [f32],
    pub slope: &'a mut [[f32; 4]],
}

/// `sum_t k[t] a[j + t][c]` for the output rows `j` and every column `c` of the plane `a` (`pw`
/// wide, `nrows + 2 r` rows: the output rows with the `r = k.len() / 2` rows the window reads on
/// either side); a sequential f32 sum over `t` from 0.
fn pass_y(a: &[f32], pw: usize, nrows: usize, k: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; nrows * pw];
    out.par_chunks_mut(pw).enumerate().for_each(|(j, row)| {
        for (t, &kt) in k.iter().enumerate() {
            for (o, &v) in row.iter_mut().zip(&a[(j + t) * pw..(j + t + 1) * pw]) {
                *o += kt * v;
            }
        }
    });
    out
}

/// `sum_t k[t] row[j + t]` for the output columns `j < ncols` of every row of `a` (`ncols + 2 r`
/// wide); the result has `ncols` values per row.
fn pass_x(a: &[f32], ncols: usize, k: &[f32]) -> Vec<f32> {
    let pw = ncols + k.len() - 1;
    let mut out = vec![0.0f32; a.len() / pw * ncols];
    out.par_chunks_mut(ncols)
        .zip(a.par_chunks(pw))
        .for_each(|(o, row)| {
            for (j, o) in o.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (t, &kt) in k.iter().enumerate() {
                    acc += kt * row[j + t];
                }
                *o = acc;
            }
        });
    out
}

/// Window moments of one source plane: for each `(moment index, (px, py))` of `wanted`, the
/// window sum of `plane` (`ncols + 2 r` wide, `nrows + 2 r` high, see [`pass_y`]) with the offset
/// powers, the y pass shared by the moments of one `py`: `(m, nrows x ncols values)`.
fn moments_of(
    plane: &[f32],
    (nrows, ncols): (usize, usize),
    taps: &[Vec<f32>],
    wanted: &[(usize, (u32, u32))],
) -> Vec<(usize, Vec<f32>)> {
    let pw = ncols + taps[0].len() - 1;
    let mut out = Vec::with_capacity(wanted.len());
    let mut pys: Vec<u32> = wanted.iter().map(|&(_, (_, py))| py).collect();
    pys.sort_unstable();
    pys.dedup();
    for py in pys {
        let ya = pass_y(plane, pw, nrows, &taps[py as usize]);
        for &(m, (px, p)) in wanted {
            if p == py {
                out.push((m, pass_x(&ya, ncols, &taps[px as usize])));
            }
        }
    }
    out
}

/// `np.maximum(v, 0.0)`: keeps NaN.
fn max0(v: f32) -> f32 {
    if v.is_nan() {
        v
    } else {
        v.max(0.0)
    }
}

/// Cholesky solve of the leading `n x n` block of the symmetric `m` (lower triangle read) for
/// `b`, in f32, the operations and their order of `phaseflow::cholesky_solve_lanes`. False if a
/// pivot is not positive and finite or the solution is not finite.
#[allow(clippy::needless_range_loop)] // textbook index form, as on the CPU
fn cholesky_f32(m: &mut [[f32; 12]; 12], b: &mut [f32; 12], n: usize) -> bool {
    let mut ok = true;
    for j in 0..n {
        let mut d = m[j][j];
        for k in 0..j {
            let v = m[j][k];
            d += (-v) * v;
        }
        ok &= d.is_finite() && d > 0.0;
        let ljj = d.sqrt();
        m[j][j] = ljj;
        for i in j + 1..n {
            let mut s = m[i][j];
            for k in 0..j {
                s += (-m[i][k]) * m[j][k];
            }
            m[i][j] = s / ljj;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s += (-m[i][k]) * b[k];
        }
        b[i] = s / m[i][i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s += (-m[k][i]) * b[k];
        }
        b[i] = s / m[i][i];
    }
    ok && b[..n].iter().all(|v| v.is_finite())
}

/// The local solve in increment form with block anchors. Writes u (Df64), energy, resid, slope
/// with the semantics of `phaseflow::local_solve_into` (a pixel that is not solved keeps its
/// input displacement and gets `pi` and zero slope; so does every pixel whose energy is not above
/// `1e-9 median(energy) + 1e-30`).
///
/// Per block `B` of `BLOCK x BLOCK` output pixels the anchor is `L_B(q) = U_B + S_B (q - c_B)`
/// (`c_B` the central pixel, `U_B` its displacement, `S_B` the central difference there; 0 where
/// not finite). With `v_q = u_q - L_B(q)` the system of a pixel `p` of the block is
/// `(M + rho I) theta' = b' - rho lambda`, `theta = theta' + lambda`, `lambda = (L_B(p), S_B, 0..)`,
/// `M` from the anchor-free sums and `b'`, `Z` from `t_q = h_q + G_q v_q`,
/// `z_q = s_phiphi + 2 h v + v^T G v`; the residual is
/// `sqrt(max0(Z - theta' b' + rho lambda theta) / (sum W s_w + 1e-30))` (GPU spec §4.3).
pub fn local_solve_blocks(
    sums: &[[f32; 7]],
    u: (&[Df64], &[Df64]),
    w: usize,
    h: usize,
    sigma: f64,
    order: u32,
    out: ModelSolveOut,
) {
    let off = ModelAperture::default();
    local_solve_blocks_ap(sums, u, w, h, sigma, order, off, out)
}

/// The [`Aperture`] prototypes of the GPU's solve: the settings and `phase_tol^2` (used in f32).
/// Default: off.
#[derive(Clone, Copy, Default)]
pub struct ModelAperture {
    pub ap: Aperture,
    pub tol2: f64,
}

/// [`local_solve_blocks`] with the [`Aperture`] prototypes as in `phaseflow`, in f32 and in the
/// increment form: the ridge target is the input's deviation `d = u0 - lambda` (`v_p` of the
/// planes), and `(d - theta')^T R theta'` takes the ridge out of the residual.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(a > b)` is true for NaN, like the CPU's
#[allow(clippy::too_many_arguments)]
pub fn local_solve_blocks_ap(
    sums: &[[f32; 7]],
    u: (&[Df64], &[Df64]),
    w: usize,
    h: usize,
    sigma: f64,
    order: u32,
    ap: ModelAperture,
    out: ModelSolveOut,
) {
    let (aconf, aridge, tol2) = (ap.ap.conf as f32, ap.ap.ridge as f32, ap.tol2 as f32);
    let ModelSolveOut {
        ux: oux,
        uy: ouy,
        energy: oen,
        resid: ores,
        slope: oslope,
    } = out;
    let (ux, uy) = u;
    for len in [
        sums.len(),
        ux.len(),
        uy.len(),
        oux.len(),
        ouy.len(),
        oen.len(),
    ] {
        assert_eq!(
            len,
            w * h,
            "local_solve_blocks: input or output of the wrong size"
        );
    }
    assert_eq!(ores.len(), w * h);
    assert_eq!(oslope.len(), w * h);
    let (n, moments, m_idx, b_idx, gcc_idx, gw_idx) = plan_moments(order);
    let taps: Vec<Vec<f32>> = moment_taps(sigma, 4)
        .iter()
        .map(|t| t.iter().map(|&v| v as f32).collect())
        .collect();
    let r = taps[0].len() / 2;

    // the anchor-free window moments (Gxx, Gxy, Gyy -> M, Gw -> sum W s_w) over the whole image
    let mut global: Vec<Vec<f32>> = vec![Vec::new(); moments.len()];
    for (src, plane_of) in [(0u8, 0usize), (1, 1), (2, 2), (6, 6)] {
        let wanted: Vec<(usize, (u32, u32))> = moments
            .iter()
            .enumerate()
            .filter(|(_, m)| m.0 == src)
            .map(|(i, m)| (i, m.1))
            .collect();
        if wanted.is_empty() {
            continue;
        }
        // the plane with its edges replicated by the window radius
        let pw = w + 2 * r;
        let mut plane = Vec::with_capacity(pw * (h + 2 * r));
        for qy in 0..h + 2 * r {
            let sy = qy.saturating_sub(r).min(h - 1);
            plane
                .extend((0..pw).map(|qx| sums[sy * w + qx.saturating_sub(r).min(w - 1)][plane_of]));
        }
        for (m, v) in moments_of(&plane, (h, w), &taps, &wanted) {
            global[m] = v;
        }
    }

    // per block: the moments of the increment form, the systems of its pixels
    let band = BLOCK * w;
    let bands = (
        oux.par_chunks_mut(band),
        ouy.par_chunks_mut(band),
        oen.par_chunks_mut(band),
        ores.par_chunks_mut(band),
        oslope.par_chunks_mut(band),
    );
    bands
        .into_par_iter()
        .enumerate()
        .for_each(|(by, (o_ux, o_uy, o_en, o_res, o_sl))| {
            let (y0, y1) = (by * BLOCK, ((by + 1) * BLOCK).min(h));
            for x0 in (0..w).step_by(BLOCK) {
                let x1 = (x0 + BLOCK).min(w);
                // the anchor
                let (cx, cy) = ((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2);
                let c = cy * w + cx;
                let mut anchor = (ux[c], uy[c]);
                if !(anchor.0.hi.is_finite() && anchor.1.hi.is_finite()) {
                    anchor = (Df64::from_f32(0.0), Df64::from_f32(0.0));
                }
                // S = [dux/dx, duy/dx, dux/dy, duy/dy]: the central difference, one-sided at an edge
                let diff = |a: usize, b: usize, span: f32| {
                    [ux[a].diff_f32(ux[b]) / span, uy[a].diff_f32(uy[b]) / span]
                };
                let (xa, xb) = (c + usize::from(cx + 1 < w), c - usize::from(cx > 0));
                let (ya, yb) = (c + usize::from(cy + 1 < h) * w, c - usize::from(cy > 0) * w);
                let sx = if xa == xb {
                    [0.0; 2]
                } else {
                    diff(xa, xb, (xa - xb) as f32)
                };
                let sy = if ya == yb {
                    [0.0; 2]
                } else {
                    diff(ya, yb, ((ya - yb) / w) as f32)
                };
                let mut s = [sx[0], sx[1], sy[0], sy[1]];
                if !s.iter().all(|v| v.is_finite()) {
                    s = [0.0; 4];
                }
                // the planes of the increment form over the block and the radius of the windows
                // around it. Beyond the image the window sees the edge pixel again (replicated):
                // its sums and displacement, but the anchor at the position it is read at, as
                // the algebra of the increment form needs
                let (bw, bh) = (x1 - x0, y1 - y0);
                let (ew, eh) = (bw + 2 * r, bh + 2 * r);
                let mut planes = [
                    Vec::with_capacity(ew * eh),
                    Vec::with_capacity(ew * eh),
                    Vec::with_capacity(ew * eh),
                ];
                for j in 0..eh {
                    // image row of the sums, and the row of the position (maybe outside)
                    let qy = (y0 + j).saturating_sub(r).min(h - 1);
                    let dy = (y0 as f32 + j as f32 - r as f32) - cy as f32;
                    for i in 0..ew {
                        let qx = (x0 + i).saturating_sub(r).min(w - 1);
                        let dx = (x0 as f32 + i as f32 - r as f32) - cx as f32;
                        let q = qy * w + qx;
                        let vx = ux[q].diff_f32(anchor.0) - (s[0] * dx + s[2] * dy);
                        let vy = uy[q].diff_f32(anchor.1) - (s[1] * dx + s[3] * dy);
                        let [gxx, gxy, gyy, hx, hy, spp, _] = sums[q];
                        planes[0].push(hx + (gxx * vx + gxy * vy));
                        planes[1].push(hy + (gxy * vx + gyy * vy));
                        planes[2].push(
                            (spp + 2.0 * (hx * vx + hy * vy))
                                + (vx * (gxx * vx + gxy * vy) + vy * (gxy * vx + gyy * vy)),
                        );
                    }
                }
                // the moments of the block's pixels
                let mut local: Vec<Vec<f32>> = vec![Vec::new(); moments.len()];
                for (src, plane) in [(3u8, &planes[0]), (4, &planes[1]), (5, &planes[2])] {
                    let wanted: Vec<(usize, (u32, u32))> = moments
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| m.0 == src)
                        .map(|(i, m)| (i, m.1))
                        .collect();
                    if wanted.is_empty() {
                        continue;
                    }
                    for (m, v) in moments_of(plane, (bh, bw), &taps, &wanted) {
                        local[m] = v;
                    }
                }
                // the systems
                for py in y0..y1 {
                    for px in x0..x1 {
                        let p = py * w + px;
                        let o = (py - y0) * w + px;
                        let lp = (py - y0) * bw + (px - x0);
                        let at = |idx: usize| global[idx][p];
                        let (m00, m11, m01) = (at(m_idx[0]), at(m_idx[n + 1]), at(m_idx[n]));
                        let e = m00 + m11;
                        o_en[o] = e;
                        (o_ux[o], o_uy[o], o_res[o]) = (ux[p], uy[p], PI as f32);
                        o_sl[o] = [0.0; 4];
                        let det2 = m00 * m11 - m01 * m01;
                        if !(det2 > (APERTURE_TOL as f32) * e * e) {
                            continue;
                        }
                        let mut mat = [[0.0f32; 12]; 12];
                        for i in 0..n {
                            for j in 0..=i {
                                mat[i][j] = at(m_idx[i * n + j]);
                            }
                        }
                        let trace: f32 = (0..n).map(|i| mat[i][i]).sum();
                        let rho = (RIDGE as f32) * trace / n as f32;
                        for (i, row) in mat.iter_mut().enumerate().take(n) {
                            row[i] += rho;
                        }
                        // lambda without the anchor's displacement: (L_B(p) - U_B, S_B, 0, ..)
                        let (dx, dy) = ((px as f32) - (cx as f32), (py as f32) - (cy as f32));
                        let mut lam = [0.0f32; 12];
                        lam[0] = s[0] * dx + s[2] * dy;
                        lam[1] = s[1] * dx + s[3] * dy;
                        lam[2..6].copy_from_slice(&s);
                        let bp: [f32; 12] =
                            std::array::from_fn(|i| if i < n { local[b_idx[i]][lp] } else { 0.0 });
                        // lambda with the anchor's displacement (GPU spec 4.3), in f32
                        let (ax, ay) = (anchor.0.hi + anchor.0.lo, anchor.1.hi + anchor.1.lo);
                        let lam_full: [f32; 12] = std::array::from_fn(|i| match i {
                            0 => ax + lam[0],
                            1 => ay + lam[1],
                            _ => lam[i],
                        });
                        let mut theta: [f32; 12] =
                            std::array::from_fn(|i| bp[i] - rho * lam_full[i]);
                        // the eigenvalues of the translation block (as `phaseflow::eigen2`)
                        let half = 0.5f32 * (m00 - m11);
                        let lmax = 0.5f32 * (m00 + m11) + (half * half + m01 * m01).sqrt();
                        let lmin = det2 / lmax;
                        let mut ridged = None;
                        if aridge > 0.0 {
                            let lift = aridge * (lmin + lmax) - lmin;
                            if lift > 0.0 {
                                let c = lift / (lmax - lmin);
                                let r = [c * (lmax - m00), -c * m01, c * (lmax - m11)];
                                let d = (
                                    ux[p].diff_f32(anchor.0) - lam[0],
                                    uy[p].diff_f32(anchor.1) - lam[1],
                                );
                                mat[0][0] += r[0];
                                mat[1][0] += r[1];
                                mat[1][1] += r[2];
                                theta[0] += r[0] * d.0 + r[1] * d.1;
                                theta[1] += r[1] * d.0 + r[2] * d.1;
                                ridged = Some((r, d));
                            }
                        }
                        if !cholesky_f32(&mut mat, &mut theta, n) {
                            continue;
                        }
                        let mut fit = 0.0f32;
                        for i in 0..n {
                            fit += theta[i] * bp[i];
                        }
                        // rho lambda.theta with the anchor's displacement in lambda and theta
                        let (lx, ly) = (ax + lam[0], ay + lam[1]);
                        let mut lt = lx * (lx + theta[0]) + ly * (ly + theta[1]);
                        for i in 2..n {
                            lt += lam[i] * (lam[i] + theta[i]);
                        }
                        let z = local[gcc_idx][lp];
                        let sw = at(gw_idx);
                        let mut zr = (z - fit) + rho * lt;
                        if let Some((r, d)) = ridged {
                            let (e0, e1) = (d.0 - theta[0], d.1 - theta[1]);
                            zr += e0 * (r[0] * theta[0] + r[1] * theta[1])
                                + e1 * (r[1] * theta[0] + r[2] * theta[1]);
                        }
                        let mut resid = (max0(zr) / (sw + 1e-30)).sqrt();
                        if aconf > 0.0 {
                            let q = lmin / (lmin + lmax);
                            if q < aconf {
                                resid = (resid * resid + tol2 * (aconf / q).ln()).sqrt();
                            }
                        }
                        o_ux[o] = anchor.0.add_f32(lam[0]).add_f32(theta[0]);
                        o_uy[o] = anchor.1.add_f32(lam[1]).add_f32(theta[1]);
                        o_res[o] = resid;
                        o_sl[o] = [
                            theta[2] + s[0],
                            theta[3] + s[1],
                            theta[4] + s[2],
                            theta[5] + s[3],
                        ];
                    }
                }
            }
        });

    // the energy threshold needs the median over the image
    let energy: Vec<f64> = oen.iter().map(|&e| f64::from(e)).collect();
    let threshold = 1e-9 * crate::gpu_support::median(&energy) + 1e-30;
    for p in 0..w * h {
        if !(energy[p] > threshold) {
            (oux[p], ouy[p], ores[p]) = (ux[p], uy[p], PI as f32);
            oslope[p] = [0.0; 4];
        }
    }
}

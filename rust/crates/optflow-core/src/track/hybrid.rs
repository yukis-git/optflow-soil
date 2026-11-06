//! `seq_mode = "hybrid"`: direct-first sequence tracking with a spatially gated adjacent-frame
//! fallback. Port of `_track_hybrid` and `_local_affine_agreement` of `python/optflow_ref/track.py`
//! (the docstrings there, and the reports 2026-11-02-hybrid-sequence-phase1-report.md sections
//! "v2 direct-first" and "v3", are the specification).
//!
//! The direct leg is the chain machinery of the parent module (reference frame, `u0r`, the previous
//! direct raw as the start of the next); what differs is per pixel. A pixel is *direct-good* when
//! its link confidence (`c0K` of an anchored pixel, min the current leg) reaches `conf_min`; its
//! output is the chain value. Any other ROI pixel is carried by the adjacent leg `k - 1 -> k` for
//! at most `fb_max` consecutive frames, and is valid only if its value agrees with a robust local
//! affine model of the direct-good pixels around it (`fb_min_support > 0`). A pixel that was in
//! fallback when the reference moved is *unanchored* and returns to direct-good when a leg from a
//! reference older than `k - 1` is confident and agrees with the carried value (or, when there is
//! none, with the local model).

use super::*;

/// State of a hybrid run beyond the chain's `Run`.
pub(super) struct Hybrid {
    /// `c0K` is trusted (the pixel was direct-good when the reference moved).
    anch: Vec<bool>,
    /// Raw output of frame `k - 1`.
    u_prev: (Image<f32>, Image<f32>),
    /// Consecutive frames each pixel has been carried by the adjacent leg (capped at `fb_max + 1`).
    nfb: Vec<u32>,
    /// Direct-good at `k - 1` (frame 0 counts as good).
    good_prev: Vec<bool>,
    /// Raw displacement of the adjacent leg computed at `k - 1` (its start at `k`).
    prev_adj: Option<(Image<f32>, Image<f32>)>,
}

impl Hybrid {
    pub(super) fn new(w: usize, h: usize) -> Hybrid {
        Hybrid {
            anch: vec![true; w * h],
            u_prev: (Image::new(w, h), Image::new(w, h)),
            nfb: vec![0; w * h],
            good_prev: vec![true; w * h],
            prev_adj: None,
        }
    }
}

/// `conf` of a leg sampled at the positions the frame-0 pixels have in that leg's reference
/// (`x + u0r`); a sample outside the image is NaN and counts as 0 (`_conf_at`).
fn conf_at(conf: &Image<f32>, u0r: &(Image<f32>, Image<f32>)) -> Vec<f64> {
    let w = conf.w;
    let mut out = vec![0.0f64; conf.data.len()];
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
                *o = if c.is_finite() { c } else { 0.0 };
            }
        });
    out
}

fn to_f64(im: &Image<f32>) -> Vec<f64> {
    im.data.iter().map(|&v| f64::from(v)).collect()
}

/// Sum over the `(2 r + 1)^2` window around every pixel, zero outside the image (scipy's
/// `uniform_filter(a, size, mode="constant") * size^2`), by running sums along rows then columns.
fn box_sum(a: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut rows = vec![0.0f64; a.len()];
    rows.par_chunks_mut(w.max(1))
        .zip(a.par_chunks(w.max(1)))
        .for_each(|(out, inp)| {
            let mut s = 0.0;
            s += inp[..r.min(w)].iter().sum::<f64>();
            for x in 0..w {
                if x + r < w {
                    s += inp[x + r];
                }
                if x > r {
                    s -= inp[x - r - 1];
                }
                out[x] = s;
            }
        });
    // columns: out[y] = out[y - 1] + rows[y + r] - rows[y - r - 1], all x at once
    let mut out = vec![0.0f64; a.len()];
    let mut acc = vec![0.0f64; w];
    for y in 0..r.min(h) {
        for x in 0..w {
            acc[x] += rows[y * w + x];
        }
    }
    for y in 0..h {
        if y + r < h {
            for x in 0..w {
                acc[x] += rows[(y + r) * w + x];
            }
        }
        if y > r {
            for x in 0..w {
                acc[x] -= rows[(y - r - 1) * w + x];
            }
        }
        out[y * w..(y + 1) * w].copy_from_slice(&acc);
    }
    out
}

/// Solves the symmetric 3 x 3 system `m c = b0` and `m c' = b1` by Gaussian elimination with
/// partial pivoting; `None` for a singular matrix.
fn solve3(m: [[f64; 3]; 3], b: [[f64; 2]; 3]) -> Option<[[f64; 2]; 3]> {
    let mut a = [[0.0f64; 5]; 3];
    for i in 0..3 {
        a[i][..3].copy_from_slice(&m[i]);
        a[i][3] = b[i][0];
        a[i][4] = b[i][1];
    }
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c] == 0.0 || !a[p][c].is_finite() {
            return None;
        }
        a.swap(c, p);
        for i in c + 1..3 {
            let f = a[i][c] / a[c][c];
            let pivot = a[c];
            for (aj, pj) in a[i].iter_mut().zip(pivot).skip(c) {
                *aj -= f * pj;
            }
        }
    }
    let mut x = [[0.0f64; 2]; 3];
    for k in 0..2 {
        for i in (0..3).rev() {
            let mut s = a[i][3 + k];
            for j in i + 1..3 {
                s -= a[i][j] * x[j][k];
            }
            x[i][k] = s / a[i][i];
        }
    }
    Some(x)
}

/// Spatial-consistency test of the hybrid fallback (`_local_affine_agreement`): for every pixel an
/// affine model `u(q) = a + B (q - p)` is fitted by least squares (ridge on `B`) to the
/// displacement `(ux, uy)` of the `support` pixels in the square window of half-size
/// `round(radius)` around it. Two passes; in the second, support pixels whose residual against
/// their own local model of the first pass exceeds `3 tol` are dropped (robust to a displacement
/// step inside the window). `ok` where the last pass held at least `min_support` pixels in the
/// window and `|(tx, ty) - a| <= tol`. `ux`, `uy` are NaN off the support.
#[allow(clippy::too_many_arguments)]
fn local_affine_agreement(
    (w, h): (usize, usize),
    ux: &[f64],
    uy: &[f64],
    support: &[bool],
    (tx, ty): (&[f64], &[f64]),
    radius: f64,
    min_support: i64,
    tol: f64,
) -> Vec<bool> {
    let n = w * h;
    let r = (radius.round_ties_even() as i64).max(1) as usize;
    let ridge = 1e-3 * (r * r) as f64;
    let xs: Vec<f64> = (0..n).map(|i| (i % w) as f64).collect();
    let ys: Vec<f64> = (0..n).map(|i| (i / w) as f64).collect();
    let ux0: Vec<f64> = (0..n)
        .map(|i| if support[i] { ux[i] } else { 0.0 })
        .collect();
    let uy0: Vec<f64> = (0..n)
        .map(|i| if support[i] { uy[i] } else { 0.0 })
        .collect();
    let mut sel = support.to_vec();
    let (mut ax, mut ay) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    #[allow(unused_assignments)]
    let mut s1 = vec![0.0f64; n];
    let mut ok_fit = vec![false; n];
    for pass in 0..2 {
        let wt: Vec<f64> = sel.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect();
        let prod = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..n).map(f).collect() };
        s1 = box_sum(&wt, w, h, r);
        let sx = box_sum(&prod(&|i| wt[i] * xs[i]), w, h, r);
        let sy = box_sum(&prod(&|i| wt[i] * ys[i]), w, h, r);
        let sxx = box_sum(&prod(&|i| wt[i] * xs[i] * xs[i]), w, h, r);
        let sxy = box_sum(&prod(&|i| wt[i] * xs[i] * ys[i]), w, h, r);
        let syy = box_sum(&prod(&|i| wt[i] * ys[i] * ys[i]), w, h, r);
        let mut rhs: Vec<Vec<f64>> = Vec::with_capacity(6);
        for comp in [&ux0, &uy0] {
            rhs.push(box_sum(&prod(&|i| wt[i] * comp[i]), w, h, r));
            rhs.push(box_sum(&prod(&|i| wt[i] * xs[i] * comp[i]), w, h, r));
            rhs.push(box_sum(&prod(&|i| wt[i] * ys[i] * comp[i]), w, h, r));
        }
        let fits: Vec<(f64, f64, bool)> = (0..n)
            .into_par_iter()
            .map(|i| {
                if s1[i] < 3.0 {
                    return (f64::NAN, f64::NAN, false);
                }
                let (x, y) = (xs[i], ys[i]);
                let sxc = sx[i] - x * s1[i];
                let syc = sy[i] - y * s1[i];
                let sxxc = sxx[i] - 2.0 * x * sx[i] + x * x * s1[i];
                let sxyc = sxy[i] - x * sy[i] - y * sx[i] + x * y * s1[i];
                let syyc = syy[i] - 2.0 * y * sy[i] + y * y * s1[i];
                let m = [
                    [s1[i], sxc, syc],
                    [sxc, sxxc + ridge * s1[i], sxyc],
                    [syc, sxyc, syyc + ridge * s1[i]],
                ];
                let mut b = [[0.0f64; 2]; 3];
                for (c, row) in b.iter_mut().enumerate().take(3) {
                    for (k, v) in row.iter_mut().enumerate() {
                        let su = rhs[3 * k][i];
                        *v = match c {
                            0 => su,
                            1 => rhs[3 * k + 1][i] - x * su,
                            _ => rhs[3 * k + 2][i] - y * su,
                        };
                    }
                }
                match solve3(m, b) {
                    Some(c) => (c[0][0], c[0][1], true),
                    None => (f64::NAN, f64::NAN, true),
                }
            })
            .collect();
        for (i, &(a, b, ok)) in fits.iter().enumerate() {
            ax[i] = a;
            ay[i] = b;
            ok_fit[i] = ok;
        }
        if pass == 0 {
            for i in 0..n {
                let res = (ux[i] - ax[i]).hypot(uy[i] - ay[i]);
                sel[i] = support[i] && res <= 3.0 * tol;
            }
        }
    }
    (0..n)
        .map(|i| {
            s1[i] + 0.5 >= min_support as f64
                && ok_fit[i]
                && (tx[i] - ax[i]).hypot(ty[i] - ay[i]) <= tol
        })
        .collect()
}

impl<I: Iterator<Item = Image<f32>>> Sequence<'_, I> {
    /// The hybrid counterpart of the chain step, for frame `fk` (checked by the caller).
    pub(super) fn step_hybrid(
        &mut self,
        run: &mut Run,
        fk: Image<f32>,
    ) -> Result<TrackResult, TrackError> {
        let k = run.k;
        let cfg = self.cfg;
        let (w, h) = (fk.w, fk.h);
        let n = w * h;
        let roi: Vec<bool> = match self.mask {
            Some(m) => m.data.clone(),
            None => vec![true; n],
        };
        let n_roi = roi.iter().filter(|&&r| r).count();
        let mut hy = run.hybrid.take().expect("a hybrid run has its state");
        let init = run.prev_pair.as_ref().map(|p| (&p.raw_ux, &p.raw_uy));
        let mut pair = self.leg(
            &run.reference,
            &fk,
            init,
            || xdic_leg::same_ref_hint(run.prev_pair.as_ref()),
            || xdic_leg::roi_in(self.mask, &run.u0r),
        )?;
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
            // the reference moves to k - 1; c0K is replaced (not min-accumulated) by the confidence of
            // the leg old reference -> k - 1 where the pixel was direct-good at k - 1, and 0 (the pixel
            // is unanchored) where it was carried by the adjacent leg
            let prev = run
                .prev_pair
                .as_ref()
                .expect("a reference older than k - 1 means a pair was tracked before");
            let c = conf_at(&prev.conf, &run.u0r);
            run.c0r = (0..n)
                .map(|i| if hy.good_prev[i] { c[i] } else { 0.0 })
                .collect();
            hy.anch = hy.good_prev.clone();
            run.u0r = (hy.u_prev.0.clone(), hy.u_prev.1.clone());
            run.reference = std::mem::replace(&mut run.prev, Image::new(0, 0));
            run.ref_idx = k - 1;
            pair = self.leg(
                &run.reference,
                &fk,
                None,
                || xdic_leg::hint(run.prev_pair.as_ref()),
                || xdic_leg::roi_in(self.mask, &run.u0r),
            )?;
            cmean = leg_mean(&pair, &run.u0r);
        }
        let (cx, cy) = compose(&run.u0r.0, &run.u0r.1, &pair.raw_ux, &pair.raw_uy);
        let leg_c = conf_at(&pair.conf, &run.u0r);
        let conf_min = cfg.conf_min;
        let link: Vec<f64> = (0..n)
            .map(|i| {
                if hy.anch[i] {
                    run.c0r[i].min(leg_c[i])
                } else {
                    leg_c[i]
                }
            })
            .collect();
        let direct_ok: Vec<bool> = (0..n)
            .map(|i| cx.data[i].is_finite() && cy.data[i].is_finite() && link[i] >= conf_min)
            .collect();
        let mut good: Vec<bool> = (0..n).map(|i| direct_ok[i] && hy.anch[i]).collect();
        let reacq_ok = run.ref_idx + 1 < k; // the reference is older than k - 1
        let cand_re: Vec<bool> = (0..n)
            .map(|i| direct_ok[i] && !hy.anch[i] && roi[i] && reacq_ok)
            .collect();
        let need: Vec<bool> = (0..n)
            .map(|i| {
                !good[i]
                    && roi[i]
                    && hy.u_prev.0.data[i].is_finite()
                    && hy.u_prev.1.data[i].is_finite()
                    && i64::from(hy.nfb[i]) < cfg.fb_max
            })
            .collect();
        let (mut coarse, mut fine) = (pair.coarse, pair.fine);
        let mut adj_raw: Option<(Image<f32>, Image<f32>)> = None;
        let (ax, ay, ca): (Image<f32>, Image<f32>, Vec<f64>);
        if need.iter().any(|&b| b) {
            let (araw, aconf) = if run.ref_idx == k - 1 {
                // the direct leg is the adjacent leg
                (
                    (pair.raw_ux.clone(), pair.raw_uy.clone()),
                    pair.conf.clone(),
                )
            } else {
                let init = hy.prev_adj.as_ref().map(|(a, b)| (a, b));
                let adj = self.leg(
                    &run.prev,
                    &fk,
                    init,
                    || xdic_leg::hint(run.prev_pair.as_ref()),
                    || xdic_leg::roi_in(self.mask, &hy.u_prev),
                )?;
                coarse += adj.coarse;
                fine += adj.fine;
                ((adj.raw_ux, adj.raw_uy), adj.conf)
            };
            let (cax, cay) = compose(&hy.u_prev.0, &hy.u_prev.1, &araw.0, &araw.1);
            ca = conf_at(&aconf, &hy.u_prev);
            // where the composition leaves the image the carried value stays
            let fixed = |c: &Image<f32>, u: &Image<f32>| {
                Image::from_vec(
                    w,
                    h,
                    (0..n)
                        .map(|i| {
                            if cax.data[i].is_finite() && cay.data[i].is_finite() {
                                c.data[i]
                            } else {
                                u.data[i]
                            }
                        })
                        .collect(),
                )
            };
            ax = fixed(&cax, &hy.u_prev.0);
            ay = fixed(&cay, &hy.u_prev.1);
            adj_raw = Some(araw);
        } else {
            ax = hy.u_prev.0.clone();
            ay = hy.u_prev.1.clone();
            ca = vec![0.0; n];
        }
        let has_fb: Vec<bool> = (0..n).map(|i| need[i] && ca[i] >= conf_min).collect();
        let agree = |i: usize| {
            (f64::from(cx.data[i]) - f64::from(ax.data[i]))
                .hypot(f64::from(cy.data[i]) - f64::from(ay.data[i]))
                <= cfg.fb_tol
        };
        let gate_on = cfg.fb_min_support > 0;
        let radius = cfg.fb_radius_px();
        let masked_u = |good: &[bool]| -> (Vec<f64>, Vec<f64>) {
            (
                (0..n)
                    .map(|i| {
                        if good[i] {
                            f64::from(cx.data[i])
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
                (0..n)
                    .map(|i| {
                        if good[i] {
                            f64::from(cy.data[i])
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
            )
        };
        let mut re_nofb: Vec<bool> = (0..n).map(|i| cand_re[i] && !has_fb[i]).collect();
        if gate_on && (0..n).any(|i| re_nofb[i] && roi[i]) {
            // no fallback value to compare with: the direct value must agree with the local model of
            // the direct-good pixels (before this re-acquisition)
            let (gx, gy) = masked_u(&good);
            let support: Vec<bool> = (0..n).map(|i| good[i] && roi[i]).collect();
            let (txv, tyv) = (to_f64(&cx), to_f64(&cy));
            let gate0 = local_affine_agreement(
                (w, h),
                &gx,
                &gy,
                &support,
                (&txv, &tyv),
                radius,
                cfg.fb_min_support,
                cfg.fb_tol,
            );
            for i in 0..n {
                re_nofb[i] &= gate0[i];
            }
        }
        let re: Vec<bool> = (0..n)
            .map(|i| (cand_re[i] && has_fb[i] && agree(i)) || re_nofb[i])
            .collect();
        for i in 0..n {
            if re[i] {
                run.c0r[i] = leg_c[i];
                hy.anch[i] = true;
                good[i] = true;
            }
        }
        let cap = cfg.fb_max.saturating_add(1);
        for (f, &g) in hy.nfb.iter_mut().zip(&good) {
            *f = if g {
                0
            } else {
                u32::try_from((i64::from(*f) + 1).min(cap)).unwrap_or(u32::MAX)
            };
        }
        let fb_cand: Vec<bool> = (0..n)
            .map(|i| !good[i] && need[i] && i64::from(hy.nfb[i]) <= cfg.fb_max && ca[i] >= conf_min)
            .collect();
        let fb_ok: Vec<bool> = if gate_on && (0..n).any(|i| fb_cand[i] && roi[i]) {
            let (gx, gy) = masked_u(&good);
            let support: Vec<bool> = (0..n).map(|i| good[i] && roi[i]).collect();
            let (txv, tyv) = (to_f64(&ax), to_f64(&ay));
            let gate = local_affine_agreement(
                (w, h),
                &gx,
                &gy,
                &support,
                (&txv, &tyv),
                radius,
                cfg.fb_min_support,
                cfg.fb_tol,
            );
            (0..n).map(|i| fb_cand[i] && gate[i]).collect()
        } else {
            fb_cand.clone()
        };
        let pick = |c: &Image<f32>, a: &Image<f32>| {
            Image::from_vec(
                w,
                h,
                (0..n)
                    .map(|i| if good[i] { c.data[i] } else { a.data[i] })
                    .collect(),
            )
        };
        let (ux, uy) = (pick(&cx, &ax), pick(&cy, &ay));
        let conf: Vec<f32> = (0..n)
            .map(|i| {
                if good[i] {
                    link[i] as f32
                } else {
                    (ca[i] * (1.0 - hy.nfb[i] as f64 / cap as f64).max(0.0)) as f32
                }
            })
            .collect();
        let low = Image::from_vec(
            w,
            h,
            (0..n)
                .map(|i| !(good[i] || fb_ok[i]))
                .collect::<Vec<bool>>(),
        );
        let t = Instant::now();
        let xo = if cfg.is_xdic() {
            xdic_leg::frame0_out(
                &pair, &run.u0r, &ux, &uy, &low.data, cfg, self.mask, self.opts,
            )
        } else {
            None
        };
        let (fin_ux, fin_uy, fin_strain, crack) = match xo {
            Some((a, b, c, d)) => (a, b, c, Some(d)),
            None => {
                let fin = finish(&ux, &uy, &low, cfg, self.mask, band_rows(w), self.opts);
                (fin.ux, fin.uy, fin.strain, None)
            }
        };
        let (xdic_detect, xdic) = xdic_leg::leg_times(&pair);
        let share = |f: &dyn Fn(usize) -> bool| -> f64 {
            (0..n).filter(|&i| roi[i] && f(i)).count() as f64 / n_roi as f64
        };
        let timing = Timing {
            coarse,
            fine,
            strain: t.elapsed().as_secs_f64(),
            reference: Some(run.ref_idx),
            mean_conf: Some(cmean),
            n_fb_max: Some(i64::from(hy.nfb.iter().copied().max().unwrap_or(0))),
            fallback_frac: Some(share(&|i| fb_ok[i])),
            fb_gate_rejected: Some(share(&|i| fb_cand[i] && !fb_ok[i])),
            reacquired: Some(share(&|i| re[i])),
            xdic_detect,
            xdic,
        };
        let fb = Image::from_vec(
            w,
            h,
            hy.nfb
                .iter()
                .map(|&v| i32::try_from(v).unwrap_or(i32::MAX))
                .collect(),
        );
        let result = TrackResult {
            ux: fin_ux,
            uy: fin_uy,
            conf: Image::from_vec(w, h, conf),
            strain: fin_strain,
            raw_ux: ux.clone(),
            raw_uy: uy.clone(),
            timing,
            fb: Some(fb),
            crack,
        };
        hy.u_prev = (ux, uy);
        hy.good_prev = good;
        hy.prev_adj = adj_raw;
        run.hybrid = Some(hy);
        run.prev = fk;
        run.prev_pair = Some(pair);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_sum_is_the_zero_padded_window_sum() {
        let (w, h) = (9usize, 7usize);
        let a: Vec<f64> = (0..w * h).map(|i| ((i * 37) % 11) as f64 - 4.0).collect();
        for r in [1usize, 2, 3, 20] {
            let got = box_sum(&a, w, h, r);
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0;
                    for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                        for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                            s += a[yy * w + xx];
                        }
                    }
                    assert!((got[y * w + x] - s).abs() < 1e-9, "r {r} at ({x}, {y})");
                }
            }
        }
    }

    /// A plane of displacement on the support, a hole in the middle: the model predicts the hole;
    /// a target off the plane by more than `tol` fails, one within passes; too few support
    /// pixels fail.
    #[test]
    fn the_local_affine_gate_compares_with_the_model_of_the_support() {
        let (w, h) = (40usize, 40usize);
        let n = w * h;
        let plane = |i: usize| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            (0.5 + 0.01 * x - 0.02 * y, -1.0 + 0.015 * y)
        };
        let hole = |i: usize| (15..25).contains(&(i % w)) && (15..25).contains(&(i / w));
        let support: Vec<bool> = (0..n).map(|i| !hole(i)).collect();
        let ux: Vec<f64> = (0..n)
            .map(|i| if support[i] { plane(i).0 } else { f64::NAN })
            .collect();
        let uy: Vec<f64> = (0..n)
            .map(|i| if support[i] { plane(i).1 } else { f64::NAN })
            .collect();
        let centre = 20 * w + 20;
        let gate = |dx: f64, radius: f64, min_support: i64| {
            let tx: Vec<f64> = (0..n).map(|i| plane(i).0 + dx).collect();
            let ty: Vec<f64> = (0..n).map(|i| plane(i).1).collect();
            local_affine_agreement(
                (w, h),
                &ux,
                &uy,
                &support,
                (&tx, &ty),
                radius,
                min_support,
                0.5,
            )[centre]
        };
        assert!(gate(0.0, 12.0, 20));
        assert!(gate(0.4, 12.0, 20));
        assert!(!gate(0.6, 12.0, 20));
        // a window of half-size 2 holds no support pixel inside the hole
        assert!(!gate(0.0, 2.0, 20));
        // a window that holds some, but fewer than min_support of them
        assert!(gate(0.0, 8.0, 20));
        assert!(!gate(0.0, 8.0, 100_000));
    }
}

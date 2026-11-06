//! The smoothed displacement field of an outer iteration in the GPU's arithmetic (GPU spec §4.4):
//! a value is a Df64 base plus f32 deviations from it, so that the large displacement (hundreds of
//! pixels) keeps about 1e-7 px although every sum runs in f32.

use rayon::prelude::*;

use super::df64::{two_prod, Df64};
use crate::image::Image;
use crate::interp::{gauss_kernel, mirror};

/// Gaussian blur of a displacement held as Df64, centred: `s_p = u_p + sum_t k_t (u_{p+t} - u_p)`,
/// separable (y, then x), edges replicated, taps `gauss_kernel(sigma)` as f32; the pass along y
/// keeps its result as Df64, the result is rounded to f32 at the end. `sum k = 1` is the property
/// of the f64 kernel, so the offset of the values drops out of the f32 sums.
pub fn blur_centred(u: &[Df64], w: usize, h: usize, sigma: f64) -> Image<f32> {
    assert_eq!(
        u.len(),
        w * h,
        "blur_centred: {} values for {w}x{h}",
        u.len()
    );
    let k: Vec<f32> = gauss_kernel(sigma).iter().map(|&v| v as f32).collect();
    let r = k.len() / 2;
    let mut mid = vec![Df64::from_f32(0.0); w * h];
    mid.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let base = u[y * w + x];
            let mut acc = 0.0f32;
            for (t, &kt) in k.iter().enumerate() {
                let sy = (y + t).saturating_sub(r).min(h - 1);
                acc += kt * u[sy * w + x].diff_f32(base);
            }
            *o = base.add_f32(acc);
        }
    });
    let mut out = Image::<f32>::new(w, h);
    out.data
        .par_chunks_mut(w)
        .zip(mid.par_chunks(w))
        .for_each(|(out, row)| {
            for (x, o) in out.iter_mut().enumerate() {
                let base = row[x];
                let mut acc = 0.0f32;
                for (t, &kt) in k.iter().enumerate() {
                    let sx = (x + t).saturating_sub(r).min(w - 1);
                    acc += kt * row[sx].diff_f32(base);
                }
                let s = base.add_f32(acc);
                *o = s.hi + s.lo;
            }
        });
    out
}

/// The pole of the cubic B-spline prefilter, `sqrt(3) - 2`, and the start of its anti-causal pass,
/// as f32 constants (a shader embeds them as literals).
fn pole() -> f32 {
    (3.0f64.sqrt() - 2.0) as f32
}

fn anticausal_start() -> f32 {
    let z = 3.0f64.sqrt() - 2.0;
    (z / (z * z - 1.0)) as f32
}

/// Terms summed to start the causal pass (`interp::HORIZON`).
const HORIZON: usize = 21;

/// `interp::prefilter_line` in f32: the same operations in the same order, mirror boundary.
fn prefilter_line_f32(c: &mut [f32]) {
    let n = c.len();
    if n == 1 {
        return;
    }
    let z = pole();
    for v in c.iter_mut() {
        *v *= 6.0;
    }
    let h = n.min(HORIZON);
    let mut zk = 1.0f32;
    let mut sum = c[0];
    for &v in &c[1..h] {
        zk *= z;
        sum += zk * v;
    }
    c[0] = sum;
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    c[n - 1] = anticausal_start() * (c[n - 1] + z * c[n - 2]);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Cubic B-spline coefficients of an f32 field whose values may be large (the smoothed
/// displacement): per line the base C (first value of the line) is subtracted, the f32 prefilter
/// of `interp` (mirror boundary, pole sqrt(3) - 2, HORIZON terms) runs on the deviation, C is
/// added back; the pass along y keeps per-column (C, f32 deviation), the pass along x re-bases per
/// row. Rounded to f32 like `BSpline::new`.
pub fn spline_coeffs_centred(s: &Image<f32>) -> Image<f32> {
    let (w, h) = (s.w, s.h);
    assert!(
        w >= 1 && h >= 1,
        "spline_coeffs_centred: empty image {w}x{h}"
    );
    // along y, a column at a time: the deviation from the column's first value
    let base: Vec<f32> = s.data[..w].to_vec();
    let mut dev = vec![0.0f32; w * h];
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let mut line: Vec<f32> = (0..h).map(|y| s.data[y * w + x] - base[x]).collect();
            prefilter_line_f32(&mut line);
            line
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for (y, &v) in col.iter().enumerate() {
            dev[y * w + x] = v;
        }
    }
    // along x, a row at a time, re-based on the row's first value
    let mut out = Image::<f32>::new(w, h);
    out.data
        .par_chunks_mut(w)
        .zip(dev.par_chunks(w))
        .for_each(|(out, drow)| {
            let value = |x: usize| Df64::from_f32(base[x]).add_f32(drow[x]);
            let row_base = value(0);
            let mut line: Vec<f32> = (0..w).map(|x| value(x).diff_f32(row_base)).collect();
            prefilter_line_f32(&mut line);
            for (o, &e) in out.iter_mut().zip(&line) {
                let c = row_base.add_f32(e);
                *o = c.hi + c.lo;
            }
        });
    out
}

/// The four B-spline weights for the fraction `t` in `[0, 1)` in f32 (`interp::weights`).
pub(crate) fn weights_f32(t: f32) -> [f32; 4] {
    let u = 1.0 - t;
    let t2 = t * t;
    let t3 = t2 * t;
    [
        u * u * u / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// The coordinate `ci + t` (`ci` an integer, `t` in `[0, 1)`) clamped to `[0, n - 1]`, as integer
/// part and fraction (`interp::axis_taps`' clamp).
fn clamp_axis(ci: i64, t: f32, n: usize) -> (i64, f32) {
    if ci < 0 {
        (0, 0.0)
    } else if ci >= n as i64 - 1 {
        (n as i64 - 1, 0.0)
    } else {
        (ci, t)
    }
}

/// The four source indices around `c0`, mirrored into `[0, n)`.
fn mirrored(c0: i64, n: usize) -> [usize; 4] {
    let mut idx = [0usize; 4];
    for (k, slot) in idx.iter_mut().enumerate() {
        *slot = mirror(c0 + k as i64 - 1, n as i64) as usize;
    }
    idx
}

/// Source indices (mirrored into the image) and weights along one axis of length `n` for the
/// coordinate `ci + t` (`ci` an integer, `t` in `[0, 1)`), clamped to `[0, n - 1]` first
/// (`interp::axis_taps` with the coordinate split into integer part and fraction).
pub(crate) fn axis_taps_f32(ci: i64, t: f32, n: usize) -> ([usize; 4], [f32; 4]) {
    let (c0, t) = clamp_axis(ci, t, n);
    (mirrored(c0, n), weights_f32(t))
}

/// [`axis_taps_f32`] of a Df64 coordinate. Requires a finite coordinate.
pub(crate) fn axis_taps_df64(c: Df64, n: usize) -> ([usize; 4], [f32; 4]) {
    let (ci, t) = c.floor_frac();
    axis_taps_f32(i64::from(ci), t, n)
}

/// The spline of `coeffs` at `(x, y)` as Df64: `C + sum wx wy (c - C)` with `C` the coefficient of
/// the first tap, so that the f32 weights, which sum to 1 only to 1e-7, multiply deviations and
/// not the large value itself; NaN (both parts) for a non-finite coordinate. Coordinates outside
/// the image are clamped to its bounds.
pub fn sample_centred(coeffs: &Image<f32>, x: Df64, y: Df64) -> Df64 {
    if !(x.hi.is_finite() && y.hi.is_finite()) {
        return Df64 {
            hi: f32::NAN,
            lo: f32::NAN,
        };
    }
    let w = coeffs.w;
    let (xi, wx) = axis_taps_df64(x, w);
    let (yi, wy) = axis_taps_df64(y, coeffs.h);
    let base = coeffs.data[yi[0] * w + xi[0]];
    let mut total = 0.0f32;
    for (&yj, &wyj) in yi.iter().zip(&wy) {
        let row = &coeffs.data[yj * w..(yj + 1) * w];
        let mut acc = wx[0] * (row[xi[0]] - base);
        for (&xk, &wxk) in xi.iter().zip(&wx).skip(1) {
            acc += wxk * (row[xk] - base);
        }
        total += wyj * acc;
    }
    Df64::from_f32(base).add_f32(total)
}

/// `a * b` as a Df64 (exact barring under- and overflow).
fn exact_prod(a: f32, b: f32) -> Df64 {
    let (hi, lo) = two_prod(a, b);
    Df64 { hi, lo }
}

/// The four B-spline weights for the f32 fraction `t` in `[0, 1)` as Df64, to about 1e-14
/// relative (`interp::weights` in f64 is that exact value rounded).
fn weights_df64(t: f32) -> [Df64; 4] {
    let sixth = Df64::from_f64(1.0 / 6.0);
    let u = Df64::from_f32(1.0).add_f32(-t);
    let t2 = exact_prod(t, t);
    let t3 = t2.mul_f32(t);
    let u3 = u.mul(u).mul(u);
    [
        u3.mul(sixth),
        t3.mul_f32(3.0).sub(t2.mul_f32(6.0)).add_f32(4.0).mul(sixth),
        t2.mul_f32(3.0)
            .sub(t3.mul_f32(3.0))
            .add(Df64::from_f32(t).mul_f32(3.0))
            .add_f32(1.0)
            .mul(sixth),
        t3.mul(sixth),
    ]
}

/// `img(X + shift + s)` on the pixel grid of `sx`, from the coefficients `coeffs` of `img`: the
/// integer part of `X + shift + s` and the fraction of `s` are separate, and the spline sum runs
/// in Df64 with Df64 weights and is rounded to f32 at the end. That is the f64 sampling of
/// `phaseflow::warp_shifted` rounded to f32 (equal in all but about 1 pixel in 2000, always within 1 ulp): the warped
/// image feeds the FFTs, and at the pixels where a response is real by the mirror symmetry of the
/// padding (an image border) the phase difference is `+-pi` decided by the last bit of the image,
/// so a warp that is only close to the CPU's changes the border band of the result (GPU spec §4.4).
pub fn warp_exact(
    coeffs: &Image<f32>,
    sx: &Image<f32>,
    sy: &Image<f32>,
    shift: (i64, i64),
) -> Image<f32> {
    let w = sx.w;
    let mut out = Image::<f32>::new(sx.w, sx.h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let (px, py) = (sx.data[i], sy.data[i]);
            if !(px.is_finite() && py.is_finite()) {
                *o = f32::NAN;
                continue;
            }
            let (fx, fy) = (px.floor(), py.floor());
            let (cx, tx) = clamp_axis(x as i64 + shift.0 + fx as i64, px - fx, coeffs.w);
            let (cy, ty) = clamp_axis(y as i64 + shift.1 + fy as i64, py - fy, coeffs.h);
            let (xi, yi) = (mirrored(cx, coeffs.w), mirrored(cy, coeffs.h));
            let (wx, wy) = (weights_df64(tx), weights_df64(ty));
            let mut total = Df64::from_f32(0.0);
            for (&yj, &wyj) in yi.iter().zip(&wy) {
                let line = &coeffs.data[yj * coeffs.w..(yj + 1) * coeffs.w];
                let mut acc = wx[0].mul_f32(line[xi[0]]);
                for (&xk, &wxk) in xi.iter().zip(&wx).skip(1) {
                    acc = acc.add(wxk.mul_f32(line[xk]));
                }
                total = total.add(acc.mul(wyj));
            }
            *o = total.hi + total.lo;
        }
    });
    out
}

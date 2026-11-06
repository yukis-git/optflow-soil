//! Cubic B-spline interpolation and Gaussian blur.
//!
//! Port of `python/optflow_ref/interp.py`; that file is the specification. Images are indexed
//! `(x = column, y = row)`. Pixels are stored as `f32`, but the B-spline prefilter, the sampling
//! sums and the Gaussian blur run in `f64`. Every output value is a sequential sum in a fixed
//! order; rayon only splits the work across independent rows or pixels, so results do not depend
//! on the number of threads.
//!
//! # Limitation: narrow images
//!
//! The causal prefilter is started from the first `min(n, 21)` samples of each line without the
//! mirrored continuation beyond them (as in `interp.py`, whose results this port reproduces). For
//! an image narrower or shorter than about 15 to 21 pixels the spline therefore does not
//! interpolate the data exactly: the error at the pixel centres is about 1e-2 at `n = 3`, 3.6e-3
//! at `n = 5` and 5e-6 at `n = 10` (relative to values of order 1), and negligible from `n = 20`.
//! Every input in this system (images, filter responses, displacement fields) is at least 64
//! pixels in both directions, so the limitation does not arise in practice.

use num_complex::Complex32;
use rayon::prelude::*;

use crate::image::Image;
use crate::madd::madd;

/// The prefilter pole of the cubic B-spline, `sqrt(3) - 2` (computed exactly as numpy does).
fn pole() -> f64 {
    3.0f64.sqrt() - 2.0
}

/// Number of leading samples summed to start the causal prefilter: `ceil(ln 1e-12 / ln |pole|)`.
const HORIZON: usize = 21;

/// Reflects `i` into `[0, n)` with whole-sample symmetry (period `2 (n - 1)`). Requires `n >= 1`.
pub fn mirror(i: i64, n: i64) -> i64 {
    assert!(n >= 1, "mirror: n must be at least 1, got {n}");
    if (0..n).contains(&i) {
        return i;
    }
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = i.rem_euclid(period);
    if m >= n {
        period - m
    } else {
        m
    }
}

/// The four B-spline weights for the fractional offset `t` in `[0, 1)`.
fn weights(t: f64) -> [f64; 4] {
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

/// Source pixels and weights of one cubic B-spline sample: `xi`/`yi` are already mirrored into
/// the image, `wx`/`wy` are the separable weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Taps {
    pub xi: [usize; 4],
    pub yi: [usize; 4],
    pub wx: [f64; 4],
    pub wy: [f64; 4],
}

/// Indices and weights along one axis of length `n >= 1` for coordinate `c`, clamped to
/// `[0, n - 1]` first. Truncation after the clamp equals `floor` because the value is not negative.
#[inline(always)]
fn axis_taps(c: f64, n: usize) -> ([usize; 4], [f64; 4]) {
    let c = c.clamp(0.0, (n - 1) as f64);
    let c0 = c as usize;
    let t = c - c0 as f64;
    let idx = if c0 >= 1 && c0 + 2 < n {
        // all four inside: `mirror` is the identity there
        [c0 - 1, c0, c0 + 1, c0 + 2]
    } else {
        let mut idx = [0usize; 4];
        for (k, slot) in idx.iter_mut().enumerate() {
            *slot = mirror(c0 as i64 + k as i64 - 1, n as i64) as usize;
        }
        idx
    };
    (idx, weights(t))
}

/// Taps for sampling a `w` x `h` coefficient image at `(x, y)`. Coordinates outside the image
/// are clamped to its bounds. `None` if `x` or `y` is NaN or infinite. Requires `w, h >= 1`.
#[inline]
pub fn taps(x: f64, y: f64, w: usize, h: usize) -> Option<Taps> {
    assert!(w >= 1 && h >= 1, "taps: empty image {w}x{h}");
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let (xi, wx) = axis_taps(x, w);
    let (yi, wy) = axis_taps(y, h);
    Some(Taps { xi, yi, wx, wy })
}

mod sealed {
    use num_complex::Complex32;

    /// Keeps [`super::SplineValue`] to the two types below: the spline code stores the
    /// components of a value in arrays of length 2 (`LANES <= 2`).
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for Complex32 {}
}

/// A pixel type the B-spline can interpolate: `f32` or `Complex32` (the trait is sealed).
/// Interpolation and prefilter are linear with real coefficients, so each real component (lane)
/// is processed in `f64` alone.
pub trait SplineValue: sealed::Sealed + Copy + Default + Send + Sync {
    /// Number of real components: 1 for `f32`, 2 (re, im) for `Complex32`.
    const LANES: usize;
    /// Component `k < LANES`, widened to `f64`.
    fn lane(self, k: usize) -> f64;
    /// Builds a value from its components (`l[k]` for `k < LANES`; the rest is ignored).
    fn from_lanes(l: [f64; 2]) -> Self;
}

impl SplineValue for f32 {
    const LANES: usize = 1;
    fn lane(self, _k: usize) -> f64 {
        f64::from(self)
    }
    fn from_lanes(l: [f64; 2]) -> Self {
        l[0] as f32
    }
}

impl SplineValue for Complex32 {
    const LANES: usize = 2;
    fn lane(self, k: usize) -> f64 {
        f64::from(if k == 0 { self.re } else { self.im })
    }
    fn from_lanes(l: [f64; 2]) -> Self {
        Complex32::new(l[0] as f32, l[1] as f32)
    }
}

/// Cubic B-spline prefilter along axis 0 of a row-major plane of `n` rows and `w` columns, all
/// columns at once, in place. Mirror boundary. Each column is filtered with exactly the
/// operations (and order) of `prefilter_line`.
pub(crate) fn prefilter_rows(c: &mut [f64], w: usize, n: usize) {
    debug_assert_eq!(c.len(), w * n);
    if n == 1 {
        return;
    }
    let z = pole();
    for v in c.iter_mut() {
        *v *= 6.0;
    }
    // c[0] = sum_{k < h} z^k c[k]
    let h = n.min(HORIZON);
    let (first, rest) = c.split_at_mut(w);
    let mut zk = 1.0;
    for k in 1..h {
        zk *= z;
        for (a, &b) in first.iter_mut().zip(&rest[(k - 1) * w..k * w]) {
            *a = madd(zk, b, *a);
        }
    }
    // causal pass
    for i in 1..n {
        let (prev, cur) = c.split_at_mut(i * w);
        for (a, &b) in cur[..w].iter_mut().zip(&prev[(i - 1) * w..]) {
            *a = madd(z, b, *a);
        }
    }
    // start of the anti-causal pass
    let (prev, last) = c.split_at_mut((n - 1) * w);
    let scale = z / (z * z - 1.0);
    for (a, &b) in last.iter_mut().zip(&prev[(n - 2) * w..]) {
        *a = scale * (*a + z * b);
    }
    // anti-causal pass
    for i in (0..n - 1).rev() {
        let (cur, next) = c.split_at_mut((i + 1) * w);
        for (a, &b) in cur[i * w..].iter_mut().zip(&next[..w]) {
            *a = z * (b - *a);
        }
    }
}

/// The prefilter of `lines` lines of `n` pairs each, stored line after line with the two lanes
/// of a value adjacent (`c[(l * n + i) * 2 + lane]`): every lane of every line is filtered along
/// `i` with the operations of [`prefilter_rows`] for one column, the lines in lockstep.
pub(crate) fn prefilter_pair_lines(c: &mut [f64], lines: usize, n: usize) {
    debug_assert_eq!(c.len(), lines * n * 2);
    if n == 1 {
        return;
    }
    let z = pole();
    for v in c.iter_mut() {
        *v *= 6.0;
    }
    let at = |l: usize, i: usize| (l * n + i) * 2;
    // c[0] = sum_{k < h} z^k c[k]
    let h = n.min(HORIZON);
    let mut zk = 1.0;
    for k in 1..h {
        zk *= z;
        for l in 0..lines {
            let (a, b) = (at(l, 0), at(l, k));
            c[a] = madd(zk, c[b], c[a]);
            c[a + 1] = madd(zk, c[b + 1], c[a + 1]);
        }
    }
    // causal pass
    for i in 1..n {
        for l in 0..lines {
            let (a, b) = (at(l, i), at(l, i - 1));
            c[a] = madd(z, c[b], c[a]);
            c[a + 1] = madd(z, c[b + 1], c[a + 1]);
        }
    }
    // start of the anti-causal pass
    let scale = z / (z * z - 1.0);
    for l in 0..lines {
        let (a, b) = (at(l, n - 1), at(l, n - 2));
        c[a] = scale * (c[a] + z * c[b]);
        c[a + 1] = scale * (c[a + 1] + z * c[b + 1]);
    }
    // anti-causal pass
    for i in (0..n - 1).rev() {
        for l in 0..lines {
            let (a, b) = (at(l, i), at(l, i + 1));
            c[a] = z * (c[b] - c[a]);
            c[a + 1] = z * (c[b + 1] - c[a + 1]);
        }
    }
}

/// Cubic B-spline prefilter of one contiguous line, in place (mirror boundary). The definition
/// that [`prefilter_rows`] follows column by column (the reference of the tests).
#[cfg(test)]
fn prefilter_line(c: &mut [f64]) {
    let n = c.len();
    if n == 1 {
        return;
    }
    let z = pole();
    for v in c.iter_mut() {
        *v *= 6.0;
    }
    let h = n.min(HORIZON);
    let mut zk = 1.0;
    let mut sum = c[0];
    for &v in &c[1..h] {
        zk *= z;
        sum = madd(zk, v, sum);
    }
    c[0] = sum;
    for i in 1..n {
        c[i] = madd(z, c[i - 1], c[i]);
    }
    c[n - 1] = (z / (z * z - 1.0)) * (c[n - 1] + z * c[n - 2]);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Cubic B-spline coefficients of an image, ready for sampling.
#[derive(Clone, Debug)]
pub struct BSpline<T: SplineValue> {
    pub coeffs: Image<T>,
}

impl<T: SplineValue> BSpline<T> {
    /// Prefilters `img` (axis 0, then axis 1, mirror boundary) in `f64` and stores the
    /// coefficients in `T`.
    ///
    /// Both passes run in parallel: the pass along axis 0 on blocks of `STRIP` columns (each held
    /// as its own `h x STRIP` plane), the pass along axis 1 on blocks of 8 rows side by side;
    /// `prefilter_rows` filters every line of a block with the operations of the line filter, so
    /// the coefficients are those of the two passes over the whole image, bit for bit.
    ///
    /// # Panics
    ///
    /// If `img` is empty (`w` or `h` is 0).
    ///
    /// # Limitation
    ///
    /// The causal initialisation sums only the first `min(n, 21)` samples of a line, without the
    /// mirrored continuation (same as `interp.py`). For an image narrower or shorter than about
    /// 15 to 21 pixels the spline does not interpolate exactly: lattice error about 1e-2 at
    /// `n = 3`, 3.6e-3 at `n = 5`, 5e-6 at `n = 10`. All images in this system are at least 64
    /// pixels in both directions.
    pub fn new(img: &Image<T>) -> Self {
        const STRIP: usize = 16;
        let (w, h) = (img.w, img.h);
        assert!(w >= 1 && h >= 1, "BSpline::new: empty image {w}x{h}");
        // per lane, the pass along y: block `b` (columns `b STRIP ..`, `bw` wide) at
        // `cols[lane][b * h * STRIP..]`, row-major within the block
        let cols: Vec<Vec<f64>> = (0..T::LANES)
            .map(|lane| {
                let mut c = vec![0.0f64; w * h];
                c.par_chunks_mut(h * STRIP)
                    .enumerate()
                    .for_each(|(b, blk)| {
                        let (x0, bw) = (b * STRIP, blk.len() / h);
                        for (y, row) in blk.chunks_mut(bw).enumerate() {
                            let src = &img.data[y * w + x0..y * w + x0 + bw];
                            for (o, v) in row.iter_mut().zip(src) {
                                *o = v.lane(lane);
                            }
                        }
                        prefilter_rows(blk, bw, h);
                    });
                c
            })
            .collect();
        // the pass along x on blocks of `ROWS` rows in lockstep (the rows of a block side by
        // side, `buf[x * rows + r]`, filtered by `prefilter_rows` with the operations of the line
        // filter), every lane, and the coefficients in `T`
        const ROWS: usize = 8;
        let mut data = vec![T::default(); w * h];
        data.par_chunks_mut(w * ROWS).enumerate().for_each_init(
            || [vec![0.0f64; w * ROWS], vec![0.0f64; w * ROWS]],
            |bufs, (blk, out)| {
                let (y0, rows) = (blk * ROWS, out.len() / w);
                for (c, buf) in cols.iter().zip(bufs.iter_mut()) {
                    let buf = &mut buf[..w * rows];
                    for r in 0..rows {
                        let y = y0 + r;
                        for b in 0..w.div_ceil(STRIP) {
                            let bw = STRIP.min(w - b * STRIP);
                            let at = b * h * STRIP + y * bw;
                            for (j, &v) in c[at..at + bw].iter().enumerate() {
                                buf[(b * STRIP + j) * rows + r] = v;
                            }
                        }
                    }
                    prefilter_rows(buf, rows, w);
                }
                for (r, out) in out.chunks_mut(w).enumerate() {
                    for (x, o) in out.iter_mut().enumerate() {
                        let i = x * rows + r;
                        let im = if T::LANES > 1 { bufs[1][i] } else { 0.0 };
                        *o = T::from_lanes([bufs[0][i], im]);
                    }
                }
            },
        );
        Self {
            coeffs: Image::from_vec(w, h, data),
        }
    }

    /// The spline at `(x, y)`. Coordinates outside the image are clamped to its bounds;
    /// NaN (in every component) if `x` or `y` is not finite.
    pub fn sample(&self, x: f64, y: f64) -> T {
        let w = self.coeffs.w;
        let Some(t) = taps(x, y, w, self.coeffs.h) else {
            return T::from_lanes([f64::NAN; 2]);
        };
        let mut out = [0.0f64; 2];
        for (lane, o) in out.iter_mut().enumerate().take(T::LANES) {
            let mut total = 0.0;
            for (&yj, &wyj) in t.yi.iter().zip(&t.wy) {
                let row = &self.coeffs.data[yj * w..(yj + 1) * w];
                let mut acc = t.wx[0] * row[t.xi[0]].lane(lane);
                for (&xk, &wxk) in t.xi.iter().zip(&t.wx).skip(1) {
                    acc = madd(wxk, row[xk].lane(lane), acc);
                }
                total = madd(wyj, acc, total);
            }
            *o = total;
        }
        T::from_lanes(out)
    }
}

/// `img(X + ux, Y + uy)` on the pixel grid. All three images must have the same size.
///
/// Inherits the narrow-image limitation of [`BSpline::new`]: for an image narrower or shorter
/// than about 15 to 21 pixels the interpolation is not exact, even with zero displacement.
///
/// # Panics
///
/// If the three images differ in size, or are empty (there is nothing to build the spline of).
pub fn warp(img: &Image<f32>, ux: &Image<f32>, uy: &Image<f32>) -> Image<f32> {
    assert!(
        (ux.w, ux.h) == (img.w, img.h) && (uy.w, uy.h) == (img.w, img.h),
        "warp: image is {}x{}, ux is {}x{}, uy is {}x{}",
        img.w,
        img.h,
        ux.w,
        ux.h,
        uy.w,
        uy.h
    );
    let spline = BSpline::new(img);
    let w = img.w;
    let mut out = Image::new(img.w, img.h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = y * w + x;
            *o = spline.sample(
                x as f64 + f64::from(ux.data[i]),
                y as f64 + f64::from(uy.data[i]),
            );
        }
    });
    out
}

/// Normalised Gaussian kernel of radius `round(4 sigma)` (`2 r + 1` taps).
///
/// # Panics
///
/// Unless `sigma` is positive and finite.
pub fn gauss_kernel(sigma: f64) -> Vec<f64> {
    assert!(
        sigma.is_finite() && sigma > 0.0,
        "gauss_kernel: sigma must be positive and finite, got {sigma}"
    );
    let r = (4.0 * sigma + 0.5) as usize;
    let mut k: Vec<f64> = (0..=2 * r)
        .map(|i| {
            let u = (i as f64 - r as f64) / sigma;
            (-0.5 * (u * u)).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// What a correlation sees outside the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Edge pixels are replicated (scipy `mode="nearest"`).
    Nearest,
    /// Zeros (scipy `mode="constant", cval=0`).
    Constant0,
}

/// Index of the pixel read for position `i` of an axis of length `n`; `None` reads zero.
fn source_index(i: i64, n: usize, b: Boundary) -> Option<usize> {
    match b {
        Boundary::Nearest => Some(i.clamp(0, n as i64 - 1) as usize),
        Boundary::Constant0 => (0..n as i64).contains(&i).then_some(i as usize),
    }
}

/// Separable correlation: `ky` along axis 0 (rows, y), then `kx` along axis 1 (x), with the
/// semantics of `scipy.ndimage.correlate1d`: `out[i] = sum_k w[k] in[i + k - r]`, `r = len / 2`.
/// Both kernels must be non-empty.
pub fn correlate_sep(img: &Image<f64>, ky: &[f64], kx: &[f64], b: Boundary) -> Image<f64> {
    assert!(
        !ky.is_empty() && !kx.is_empty(),
        "correlate_sep: empty kernel"
    );
    let (w, h) = (img.w, img.h);
    if w == 0 || h == 0 {
        return img.clone();
    }
    let (ry, rx) = ((ky.len() / 2) as i64, (kx.len() / 2) as i64);

    let mut tmp = Image::new(w, h);
    tmp.data.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        for (k, &wk) in ky.iter().enumerate() {
            let Some(src) = source_index(y as i64 + k as i64 - ry, h, b) else {
                continue;
            };
            for (o, &v) in out.iter_mut().zip(&img.data[src * w..(src + 1) * w]) {
                *o = madd(wk, v, *o);
            }
        }
    });

    let mut out = Image::new(w, h);
    out.data
        .par_chunks_mut(w)
        .zip(tmp.data.par_chunks(w))
        .for_each_init(Vec::new, |pad: &mut Vec<f64>, (out, row)| {
            pad.clear();
            pad.extend(
                (0..w + kx.len() - 1)
                    .map(|j| source_index(j as i64 - rx, w, b).map_or(0.0, |s| row[s])),
            );
            // out[x] = sum_k kx[k] pad[x + k], a sequential sum per output
            taps_sum(
                kx.len(),
                |t| &pad[t..t + w],
                &[kx],
                w,
                |_, x, v| {
                    out[x..x + v.len()].copy_from_slice(v);
                },
            );
        });
    out
}

/// Most kernels one call of [`taps_sum`] applies at once.
pub(crate) const MAX_KERNELS: usize = 3;

/// For every kernel `ks[p]` (at most `MAX_KERNELS`, each at least `nt` taps) and `x < len`:
/// `sum_t ks[p][t] * line(t)[x]` over `t < nt`, a sequential sum over `t` starting from 0 (the
/// order of [`correlate_sep`]), handed to `put(p, x0, values from x0)`. Blocks of `B` outputs
/// stay in registers over all taps, and each value read serves every kernel.
pub(crate) fn taps_sum<'a>(
    nt: usize,
    line: impl Fn(usize) -> &'a [f64],
    ks: &[&[f64]],
    len: usize,
    put: impl FnMut(usize, usize, &[f64]),
) {
    match ks.len() {
        1 => taps_sum_n::<1>(nt, line, [ks[0]], len, put),
        2 => taps_sum_n::<2>(nt, line, [ks[0], ks[1]], len, put),
        3 => taps_sum_n::<3>(nt, line, [ks[0], ks[1], ks[2]], len, put),
        n => unreachable!("taps_sum: {n} kernels"),
    }
}

#[inline(always)]
fn taps_sum_n<'a, const P: usize>(
    nt: usize,
    line: impl Fn(usize) -> &'a [f64],
    ks: [&[f64]; P],
    len: usize,
    mut put: impl FnMut(usize, usize, &[f64]),
) {
    const B: usize = 8;
    let ks: [&[f64]; P] = ks.map(|k| &k[..nt]);
    let mut x0 = 0;
    while x0 + B <= len {
        let mut acc = [[0.0f64; B]; P];
        for t in 0..nt {
            let v: &[f64; B] = line(t)[x0..x0 + B].try_into().expect("block of B");
            for (a, k) in acc.iter_mut().zip(&ks) {
                let kt = k[t];
                for (a, &v) in a.iter_mut().zip(v) {
                    *a = madd(kt, v, *a);
                }
            }
        }
        for (p, a) in acc.iter().enumerate() {
            put(p, x0, a);
        }
        x0 += B;
    }
    for x in x0..len {
        let mut acc = [0.0f64; P];
        for t in 0..nt {
            let v = line(t)[x];
            for (a, k) in acc.iter_mut().zip(&ks) {
                *a = madd(k[t], v, *a);
            }
        }
        for (p, a) in acc.iter().enumerate() {
            put(p, x, std::slice::from_ref(a));
        }
    }
}

/// Separable Gaussian blur, radius `round(4 sigma)`, edge values replicated.
///
/// # Panics
///
/// Unless `sigma` is positive and finite (see [`gauss_kernel`]), also for an empty image.
pub fn gauss_blur(img: &Image<f64>, sigma: f64) -> Image<f64> {
    let k = gauss_kernel(sigma);
    correlate_sep(img, &k, &k, Boundary::Nearest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Deterministic pseudo-random values in `[-1, 1)` (no external RNG needed).
    fn lcg_values(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    fn f32_image(w: usize, h: usize, seed: u64) -> Image<f32> {
        let v = lcg_values(w * h, seed);
        Image::from_vec(w, h, v.iter().map(|&x| x as f32).collect())
    }

    #[test]
    fn interp_horizon_matches_its_definition() {
        let h = ((1e-12f64).ln() / pole().abs().ln()).ceil();
        assert_eq!(HORIZON as f64, h);
    }

    #[test]
    fn interp_mirror_reflects_with_whole_sample_symmetry() {
        // period 2 (n - 1) = 8: ... 3 2 1 | 0 1 2 3 4 3 2 1 | 0 1 2 ...; same values as interp._mirror
        let got: Vec<i64> = (-9..=9).map(|i| mirror(i, 5)).collect();
        assert_eq!(
            got,
            vec![1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1]
        );
        assert!((-5..5).all(|i| mirror(i, 1) == 0));
        assert_eq!(
            (0..6).map(|i| mirror(i - 2, 2)).collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0, 1]
        );
        assert_eq!(mirror(i64::MIN, 7), mirror(i64::MIN + 12, 7));
    }

    /// Taps at the last pixel `x = w - 1` and past it (`w - 1 + 0.5` is clamped to `w - 1`): at the
    /// border the four-tap support folds back with whole-sample symmetry, and the weights are
    /// those of the fractional offset 0: `[1/6, 2/3, 1/6, 0]`.
    #[test]
    fn interp_taps_at_the_last_pixel_and_half_a_pixel_beyond_it() {
        let (w, h) = (7usize, 5usize);
        let weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0];
        let last = taps(6.0, 2.0, w, h).unwrap();
        assert_eq!(last.xi, [5, 6, 5, 4]);
        assert_eq!(last.wx, weights);
        assert_eq!(taps(6.5, 2.0, w, h).unwrap(), last);
        assert_eq!(taps(1e9, 2.0, w, h).unwrap(), last);
        // the same along y at y = h - 1, and at the corner
        let bottom = taps(3.0, 4.0, w, h).unwrap();
        assert_eq!(bottom.yi, [3, 4, 3, 2]);
        assert_eq!(bottom.wy, weights);
        assert_eq!(taps(3.0, 4.5, w, h).unwrap(), bottom);
        let corner = taps(6.5, 4.5, w, h).unwrap();
        assert_eq!((corner.xi, corner.yi), (last.xi, bottom.yi));
        // one pixel before the last: the support still reaches past the border by one tap
        let before = taps(5.0, 2.0, w, h).unwrap();
        assert_eq!(before.xi, [4, 5, 6, 5]);
        assert_eq!(before.wx, weights);
        // a narrow axis: n = 2 reflects every tap
        let two = taps(1.5, 0.0, 2, 1).unwrap();
        assert_eq!((two.xi, two.yi), ([0, 1, 0, 1], [0; 4]));
        // and the sample itself: past the last pixel it equals the sample at the last pixel
        let s = BSpline::new(&f32_image(24, 24, 9));
        for y in [0.0, 7.25, 23.0] {
            assert_eq!(s.sample(23.0, y), s.sample(23.5, y));
            assert_eq!(s.sample(y, 23.0), s.sample(y, 23.5));
        }
    }

    #[test]
    #[should_panic(expected = "empty image")]
    fn interp_warp_rejects_an_empty_image() {
        let none = Image::<f32>::new(0, 0);
        let _ = warp(&none, &none, &none);
    }

    #[test]
    fn interp_gauss_blur_rejects_a_sigma_that_is_not_positive_and_finite() {
        let img = Image::from_vec(3, 3, vec![1.0f64; 9]);
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let r = std::panic::catch_unwind(|| gauss_blur(&img, sigma));
            assert!(r.is_err(), "sigma {sigma}");
        }
        // also for an empty image
        let none = Image::<f64>::new(0, 4);
        assert!(std::panic::catch_unwind(|| gauss_blur(&none, f64::NAN)).is_err());
    }

    #[test]
    fn interp_gauss_kernel_radius_and_normalisation() {
        // radius = int(4 sigma + 0.5), as numpy's int() truncates
        for (sigma, taps) in [
            (0.1, 1),
            (0.125, 3),
            (1.0, 9),
            (2.0, 17),
            (2.5, 21),
            (4.0, 33),
        ] {
            let k = gauss_kernel(sigma);
            assert_eq!(k.len(), taps, "sigma {sigma}");
            assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            assert!(k.iter().zip(k.iter().rev()).all(|(a, b)| a == b));
        }
        let k = gauss_kernel(1.0);
        // values printed by python/optflow_ref/interp.py gauss_kernel(1.0)
        assert!((k[4] - 0.398943469).abs() < 1e-9 && (k[0] - 1.33830625e-4).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "sigma")]
    fn interp_gauss_kernel_rejects_zero_sigma() {
        let _ = gauss_kernel(0.0);
    }

    #[test]
    fn interp_prefilter_rows_matches_line_filter_bitwise() {
        for n in [1usize, 2, 3, 5, 20, 21, 22, 50] {
            let w = 7;
            let plane = lcg_values(w * n, n as u64);
            let mut rows = plane.clone();
            prefilter_rows(&mut rows, w, n);
            for x in 0..w {
                let mut col: Vec<f64> = (0..n).map(|y| plane[y * w + x]).collect();
                prefilter_line(&mut col);
                let got: Vec<f64> = (0..n).map(|y| rows[y * w + x]).collect();
                assert_eq!(got, col, "n={n} column {x}");
            }
            // three lines of n pairs, line after line
            let pairs = lcg_values(3 * n * 2, 100 + n as u64);
            let mut lines = pairs.clone();
            prefilter_pair_lines(&mut lines, 3, n);
            for l in 0..3 {
                for lane in 0..2 {
                    let at = |i: usize| (l * n + i) * 2 + lane;
                    let mut line: Vec<f64> = (0..n).map(|i| pairs[at(i)]).collect();
                    prefilter_line(&mut line);
                    let got: Vec<f64> = (0..n).map(|i| lines[at(i)]).collect();
                    assert_eq!(got, line, "n={n} pair line {l} lane {lane}");
                }
            }
        }
    }

    // The Python prefilter starts the causal pass from the first min(n, 21) samples and ignores the
    // mirrored continuation beyond them, so for images narrower than about 15 pixels the spline
    // does not pass exactly through the data. These golden values come from interp.py itself.
    #[test]
    fn interp_small_image_matches_python_golden_values() {
        #[rustfmt::skip]
        let img = Image::from_vec(5, 3, vec![
            1.0f32, -2.0, 0.5, 3.0, 0.0,
            2.0, 1.0, -1.0, 0.25, 4.0,
            0.0, 3.0, 2.0, -3.0, 1.0,
        ]);
        let s = BSpline::new(&img);
        let golden = [
            (0.0, 0.0, 1.0398018662995405),
            (4.0, 2.0, 1.0),
            (1.0, 1.0, 1.0000000000000004),
            (2.3, 0.7, -0.32944405408591126),
            (3.9, 1.6, 2.016086592306819),
            (0.4, 1.95, 0.7889519509314864),
            (-1.0, 5.0, -0.004570472309550544), // clamped to (0, 2)
            (2.5, 1.0, -1.011134931967287),
        ];
        for (x, y, want) in golden {
            let got = f64::from(s.sample(x, y));
            assert!((got - want).abs() < 1e-6, "({x}, {y}): {got} vs {want}");
        }
        // a single row: the y direction has n = 1 and is skipped by the prefilter
        let row = BSpline::new(&Image::from_vec(3, 1, vec![1.0f32, 2.0, 4.0]));
        for (x, y, want) in [
            (0.0, 0.0, 1.0346030478972568),
            (0.5, 0.0, 1.239295559688996),
            (1.5, 0.3, 3.2780059642596338),
            (2.0, 0.0, 4.000000000000002),
            (1.0, 5.0, 1.9999999999999998),
        ] {
            let got = f64::from(row.sample(x, y));
            assert!((got - want).abs() < 1e-6, "({x}, {y}): {got} vs {want}");
        }
    }

    #[test]
    fn interp_complex_spline_is_two_real_splines() {
        let re = f32_image(23, 17, 1);
        let im = f32_image(23, 17, 2);
        let z = Image::from_vec(
            23,
            17,
            re.data
                .iter()
                .zip(&im.data)
                .map(|(&a, &b)| Complex32::new(a, b))
                .collect(),
        );
        let (sz, sr, si) = (BSpline::new(&z), BSpline::new(&re), BSpline::new(&im));
        for (x, y) in [(0.3, 0.9), (11.5, 8.25), (22.0, 16.0), (-4.0, 30.0)] {
            let v = sz.sample(x, y);
            assert_eq!((v.re, v.im), (sr.sample(x, y), si.sample(x, y)));
        }
        let nan = sz.sample(f64::NAN, 0.0);
        assert!(nan.re.is_nan() && nan.im.is_nan());
    }

    /// `BSpline::new` as defined: the prefilter along axis 0 over the whole plane, then along
    /// axis 1 line by line, one lane after the other.
    fn bspline_serial<T: SplineValue>(img: &Image<T>) -> Vec<T> {
        let (w, h) = (img.w, img.h);
        let planes: Vec<Vec<f64>> = (0..T::LANES)
            .map(|lane| {
                let mut c: Vec<f64> = img.data.iter().map(|v| v.lane(lane)).collect();
                prefilter_rows(&mut c, w, h);
                c.chunks_mut(w).for_each(prefilter_line);
                c
            })
            .collect();
        (0..w * h)
            .map(|i| T::from_lanes([planes[0][i], planes.get(1).map_or(0.0, |p| p[i])]))
            .collect()
    }

    #[test]
    fn interp_bspline_in_column_blocks_equals_the_whole_plane_passes_bit_for_bit() {
        // widths below, at and off the block width, one row and one column
        for (w, h) in [
            (1, 1),
            (5, 9),
            (16, 7),
            (17, 23),
            (37, 3),
            (64, 40),
            (1, 30),
            (45, 1),
        ] {
            let img = f32_image(w, h, (w * 31 + h) as u64);
            let got = BSpline::new(&img).coeffs.data;
            let want = bspline_serial(&img);
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "f32 {w}x{h}"
            );
            let im = f32_image(w, h, (w * 7 + h * 3) as u64);
            let z = Image::from_vec(
                w,
                h,
                img.data
                    .iter()
                    .zip(&im.data)
                    .map(|(&a, &b)| Complex32::new(a, b))
                    .collect(),
            );
            let got = BSpline::new(&z).coeffs.data;
            let want = bspline_serial(&z);
            assert!(
                got.iter().zip(&want).all(|(a, b)| {
                    (a.re.to_bits(), a.im.to_bits()) == (b.re.to_bits(), b.im.to_bits())
                }),
                "Complex32 {w}x{h}"
            );
        }
    }

    #[test]
    fn interp_warp_with_zero_displacement_reproduces_image() {
        let img = f32_image(40, 50, 7);
        let zero = Image::new(40, 50);
        let out = warp(&img, &zero, &zero);
        let max = out
            .data
            .iter()
            .zip(&img.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max < 1e-5, "{max}");
    }

    #[test]
    fn interp_warp_gives_nan_where_displacement_is_not_finite() {
        let img = f32_image(8, 8, 3);
        let mut ux = Image::new(8, 8);
        ux.set(2, 3, f32::NAN);
        let out = warp(&img, &ux, &Image::new(8, 8));
        assert!(out.get(2, 3).is_nan());
        assert_eq!(out.data.iter().filter(|v| v.is_nan()).count(), 1);
    }

    #[test]
    fn interp_correlate_sep_follows_scipy_correlate1d() {
        // values from scipy.ndimage.correlate1d on [1, 2, 3, 4]
        let row = Image::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let one = [1.0];
        let cases: [(&[f64], Boundary, [f64; 4]); 6] = [
            (&[1.0, 0.0], Boundary::Nearest, [1.0, 1.0, 2.0, 3.0]),
            (&[0.0, 1.0], Boundary::Nearest, [1.0, 2.0, 3.0, 4.0]),
            (&[1.0, 2.0, 3.0], Boundary::Nearest, [9.0, 14.0, 20.0, 23.0]),
            (&[1.0, 0.0], Boundary::Constant0, [0.0, 1.0, 2.0, 3.0]),
            (
                &[1.0, 2.0, 3.0],
                Boundary::Constant0,
                [8.0, 14.0, 20.0, 11.0],
            ),
            (
                &[1.0, 2.0, 3.0, 4.0],
                Boundary::Constant0,
                [11.0, 20.0, 30.0, 20.0],
            ),
        ];
        for (k, b, want) in cases {
            assert_eq!(correlate_sep(&row, &one, k, b).data, want, "{k:?} {b:?}");
            // the same kernel along y on the transposed image
            let col = Image::from_vec(1, 4, row.data.clone());
            assert_eq!(
                correlate_sep(&col, k, &one, b).data,
                want,
                "{k:?} {b:?} (y)"
            );
        }
    }

    /// An image with no pixel, however it is empty (no columns, no rows or neither), comes back
    /// as it is, whatever the boundary. The kernels must still be non-empty, but the early return
    /// does not read them; without it `par_chunks_mut(0)` would panic on an image with no columns
    /// but some rows.
    #[test]
    fn interp_correlate_sep_of_an_empty_image_is_that_image() {
        for (w, h) in [(0usize, 0usize), (0, 4), (4, 0)] {
            let img = Image::<f64>::new(w, h);
            for b in [Boundary::Nearest, Boundary::Constant0] {
                let out = correlate_sep(&img, &[1.0, 2.0, 3.0], &[0.5, 0.5], b);
                assert_eq!((out.w, out.h, out.data.len()), (w, h, 0), "{w}x{h} {b:?}");
            }
        }
    }

    #[test]
    fn interp_results_do_not_depend_on_thread_count() {
        let img = f32_image(61, 47, 11);
        let (ux, uy) = (f32_image(61, 47, 12), f32_image(61, 47, 13));
        let big = Image::from_vec(61, 47, img.data.iter().map(|&v| f64::from(v)).collect());
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    (
                        BSpline::new(&img).coeffs,
                        warp(&img, &ux, &uy),
                        gauss_blur(&big, 2.0),
                    )
                })
        };
        assert!(run(1) == run(4));
    }

    fn image_f32(lo: usize, hi: usize) -> impl Strategy<Value = Image<f32>> {
        (lo..=hi, lo..=hi).prop_flat_map(|(w, h)| {
            prop::collection::vec(-10.0f32..10.0, w * h).prop_map(move |v| Image::from_vec(w, h, v))
        })
    }

    fn coordinate() -> impl Strategy<Value = f64> {
        prop_oneof![
            -1e3f64..1e3,
            -1e300f64..1e300,
            Just(0.0),
            Just(f64::MAX),
            Just(f64::MIN),
        ]
    }

    /// Plain two-dimensional correlation with the same boundary rule (no separation).
    fn naive_correlate(img: &Image<f64>, ky: &[f64], kx: &[f64], b: Boundary) -> Image<f64> {
        let (ry, rx) = ((ky.len() / 2) as i64, (kx.len() / 2) as i64);
        let mut out = Image::new(img.w, img.h);
        for y in 0..img.h {
            for x in 0..img.w {
                let mut acc = 0.0;
                for (a, &wy) in ky.iter().enumerate() {
                    for (c, &wx) in kx.iter().enumerate() {
                        let sy = source_index(y as i64 + a as i64 - ry, img.h, b);
                        let sx = source_index(x as i64 + c as i64 - rx, img.w, b);
                        if let (Some(sy), Some(sx)) = (sy, sx) {
                            acc += wy * wx * img.get(sx, sy);
                        }
                    }
                }
                out.set(x, y, acc);
            }
        }
        out
    }

    proptest! {
        #[test]
        fn interp_sample_reproduces_lattice_values(img in image_f32(12, 20)) {
            let s = BSpline::new(&img);
            for y in 0..img.h {
                for x in 0..img.w {
                    let got = s.sample(x as f64, y as f64);
                    prop_assert!((got - img.get(x, y)).abs() < 1e-4,
                        "({x}, {y}): {got} vs {}", img.get(x, y));
                }
            }
        }

        // For images narrower than 12 pixels the reference prefilter is inexact (see the golden
        // test); the lattice error must still stay within the truncated tail of its start sum.
        #[test]
        fn interp_small_images_stay_within_truncation_envelope(img in image_f32(3, 11)) {
            let s = BSpline::new(&img);
            let zn = |n: usize| pole().abs().powi(n as i32);
            let peak = img.data.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
            let tol = 1e-4 + 10.0 * (zn(img.w) + zn(img.h)) * peak;
            for y in 0..img.h {
                for x in 0..img.w {
                    let got = f64::from(s.sample(x as f64, y as f64));
                    let want = f64::from(img.get(x, y));
                    prop_assert!((got - want).abs() <= tol, "({x}, {y}): {got} vs {want}, tol {tol}");
                }
            }
        }

        #[test]
        fn interp_complex_sample_reproduces_lattice_values(
            re in image_f32(12, 16), seed in 0u64..1000
        ) {
            let im = f32_image(re.w, re.h, seed);
            let z = Image::from_vec(re.w, re.h,
                re.data.iter().zip(&im.data).map(|(&a, &b)| Complex32::new(a, b)).collect());
            let s = BSpline::new(&z);
            for y in 0..z.h {
                for x in 0..z.w {
                    let got = s.sample(x as f64, y as f64);
                    prop_assert!((got - z.get(x, y)).norm() < 2e-4);
                }
            }
        }

        #[test]
        fn interp_taps_weights_sum_to_one_and_indices_are_in_bounds(
            x in coordinate(), y in coordinate(), w in 1usize..=64, h in 1usize..=64
        ) {
            let t = taps(x, y, w, h).unwrap();
            prop_assert!((t.wx.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            prop_assert!((t.wy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            prop_assert!(t.wx.iter().chain(&t.wy).all(|&v| v >= 0.0));
            prop_assert!(t.xi.iter().all(|&i| i < w) && t.yi.iter().all(|&i| i < h));
        }

        #[test]
        fn interp_taps_refuse_non_finite_coordinates(
            x in coordinate(), bad in prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
            w in 1usize..=8, h in 1usize..=8
        ) {
            prop_assert!(taps(bad, x, w, h).is_none() && taps(x, bad, w, h).is_none());
        }

        #[test]
        fn interp_correlate_sep_matches_naive_2d_correlation(
            (w, h) in (1usize..=12, 1usize..=12),
            ky in prop::collection::vec(-2.0f64..2.0, 1..=7),
            kx in prop::collection::vec(-2.0f64..2.0, 1..=7),
            seed in 0u64..1000,
            constant_zero in any::<bool>(),
        ) {
            let b = if constant_zero { Boundary::Constant0 } else { Boundary::Nearest };
            let img = Image::from_vec(w, h, lcg_values(w * h, seed));
            let got = correlate_sep(&img, &ky, &kx, b);
            let want = naive_correlate(&img, &ky, &kx, b);
            for (a, c) in got.data.iter().zip(&want.data) {
                prop_assert!((a - c).abs() < 1e-9, "{a} vs {c}");
            }
        }

        #[test]
        fn interp_gauss_blur_keeps_constant_images(
            (w, h) in (1usize..=10, 1usize..=10), sigma in 0.2f64..3.0, c in -100.0f64..100.0
        ) {
            let out = gauss_blur(&Image::from_vec(w, h, vec![c; w * h]), sigma);
            prop_assert!(out.data.iter().all(|&v| (v - c).abs() < 1e-10 * (1.0 + c.abs())));
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn k1_mirror_in_bounds() {
        let n: i64 = kani::any();
        let i: i64 = kani::any();
        kani::assume(n >= 1 && n <= 1 << 20);
        kani::assume(i > -(1 << 40) && i < (1 << 40));
        let m = mirror(i, n);
        assert!(m >= 0 && m < n);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn k2_taps_in_bounds_for_any_coordinate() {
        let w: usize = kani::any();
        let h: usize = kani::any();
        kani::assume(w >= 1 && w <= 64 && h >= 1 && h <= 64);
        let x: f64 = kani::any();
        let y: f64 = kani::any();
        match taps(x, y, w, h) {
            None => assert!(!x.is_finite() || !y.is_finite()),
            Some(t) => {
                for k in 0..4 {
                    assert!(t.xi[k] < w);
                    assert!(t.yi[k] < h);
                }
            }
        }
    }
}

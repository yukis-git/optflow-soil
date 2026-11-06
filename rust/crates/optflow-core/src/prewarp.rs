//! The trusted pre-warp of the fine stage (spec 2.3; `prewarp` of `python/optflow_ref/phaseflow.py`,
//! which is the specification).
//!
//! The pre-warp `s = G_sigma_s * u` of every outer iteration after the first leaves out the
//! displacement of pixels that cannot be trusted (see [`prewarp`]): a region where the coarse
//! stage failed holds displacements ~100 px wrong that the fine stage cannot correct, and blurred
//! into `s` they stretch the warped image of the good pixels beside them, whose phase is then
//! biased smoothly and nearly equally in all filters, so that the local fit absorbs it and they
//! stay confident at errors of up to 0.4 px. Untrusted pixels take the
//! first-order extrapolation `u(N) + grad u(N) (X - N)` from the nearest trusted pixel `N`
//! ([`crate::edt::nearest_source`], with scipy's tie rule); `grad u(N)` is the affine part
//! (`Solved::slope` of [`crate::phaseflow`]) of the local fit at `N`. Every tile of
//! [`crate::tile::refine_tiled`] does this on its own data. The reach of the rule, about
//! `R + 4 sigma_s + 12` = 40 px, is less than the 64 px tile margin, but that is not a bound; it
//! holds on the assumption that every reported pixel has a trusted pixel within about `R`. An
//! untrusted pixel far from every trusted one (the middle of a wide failed region) takes its
//! value from a distant `N`, which the edge of a tile can change; such pixels are not reported.
//! The texture scale of the confidence is the median energy of the tile, which differs from the
//! whole image's when the texture is uneven, so a pixel near `conf_min` can be trusted in one and
//! not in the other.

use rayon::prelude::*;

use crate::config::Config;
use crate::edt::{distance_to_invalid, nearest_source, within};
use crate::image::Image;
use crate::interp::gauss_blur;
use crate::phaseflow::{confidence, texture_scale};
use crate::prof::{self, Stage};

/// The pixels the pre-warp does not trust by their own confidence: the `f64` confidence of the
/// last local solve, before the border rule, is below `conf_min` (`_confidence` of the reference;
/// a NaN confidence is not low, like `conf < conf_min`). `confidence` clamps to `[0, 1]`, which
/// decides `< conf_min` (in `[0, 1]`) the same way as the unclipped value.
pub(crate) fn low_confidence(energy: &Image<f64>, resid: &Image<f64>, cfg: &Config) -> Image<bool> {
    let (w, h) = (energy.w, energy.h);
    let scale = texture_scale(&energy.data);
    let mut low = vec![false; w * h];
    low.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, l) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let c = confidence(energy.data[i], resid.data[i], true, scale, cfg.phase_tol);
            *l = c < cfg.conf_min;
        }
    });
    Image::from_vec(w, h, low)
}

/// Prototype, off by default (spec 2026-11-02-fullres-convergence): `OPTFLOW_PREWARP_FREEZE_AT=K`
/// (read once per process, `K >= 1`) keeps the low pixels of the pre-warp of outer iteration `K`
/// for every later outer iteration of the window, instead of recomputing them from each solve.
/// Pixels near `conf_min` that flip between low and confident from one solve to the next move
/// the nearest trusted pixel of the fill, the pre-warp jumps by up to tens of px there, and the
/// field near them settles into a 2-cycle that never converges; frozen, the fill rule stays the
/// same function of `u` and the slope. Unset (or not a positive integer): the low pixels of every
/// outer iteration are those of its own last solve, bit for bit as before.
pub fn freeze_at() -> Option<usize> {
    static AT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *AT.get_or_init(|| {
        std::env::var("OPTFLOW_PREWARP_FREEZE_AT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&k| k >= 1)
    })
}

/// The freeze iteration of a run: `cfg.prewarp_freeze_at` when it is positive (0 = off), else the
/// environment variable [`freeze_at`].
pub fn freeze_at_for(cfg: &Config) -> Option<usize> {
    match usize::try_from(cfg.prewarp_freeze_at) {
        Ok(k) if k >= 1 => Some(k),
        _ => freeze_at(),
    }
}

/// The low pixels of the pre-warp of one window over its outer iterations, under [`freeze_at_for`]
/// (`Default`: the environment variable only).
#[derive(Default)]
pub struct LowFreeze {
    at: Option<usize>,
    kept: Option<Image<bool>>,
}

impl LowFreeze {
    /// The freeze of a run with configuration `cfg`.
    pub fn for_config(cfg: &Config) -> LowFreeze {
        LowFreeze {
            at: freeze_at_for(cfg),
            kept: None,
        }
    }

    /// The low pixels for the pre-warp of outer iteration `it` (`>= 1`): `compute()` (the low
    /// pixels of the last solve) up to the freeze iteration, the kept ones after it. Off:
    /// `compute()`, nothing kept.
    pub fn low(&mut self, it: usize, compute: impl FnOnce() -> Image<bool>) -> Image<bool> {
        let at = self.at.or_else(freeze_at);
        match at {
            Some(k) if it > k => {
                if let Some(kept) = &self.kept {
                    return kept.clone();
                }
                compute()
            }
            Some(k) if it == k => {
                let low = compute();
                self.kept = Some(low.clone());
                low
            }
            _ => compute(),
        }
    }
}

/// The trusted pixels (farther than `radius` from every low pixel) and the nearest trusted pixel
/// of every pixel, or `None` if the fill changes nothing: no pixel is low, or no pixel is trusted,
/// or all are (see `prewarp`).
fn trusted_sources(low: &Image<bool>, radius: f64) -> Option<(Image<bool>, Vec<u32>)> {
    let (w, h) = (low.w, low.h);
    if !low.data.iter().any(|&l| l) {
        return None;
    }
    // trusted: farther than `radius` from every low pixel
    let valid = Image::from_vec(w, h, low.data.iter().map(|&l| !l).collect());
    let dist = distance_to_invalid(&valid);
    let trusted = Image::from_vec(
        w,
        h,
        dist.data.iter().map(|&d| !within(d, radius)).collect(),
    );
    let n_trusted = trusted.data.iter().filter(|&&t| t).count();
    if n_trusted == 0 || n_trusted == w * h {
        return None;
    }
    let nearest = nearest_source(&trusted)?;
    Some((trusted, nearest))
}

/// The untrusted pixels `p` of a `w` x `h` field and their values `(p, ux, uy)` in
/// [`extrapolated`], in raster order, or `None` where it is `None`. `at(n)` is `(ux, uy, slope)`
/// of pixel `n` (read only at trusted pixels), so the field can stay in the caller's layout (the
/// GPU backend's Df64 planes); the arithmetic is that of `extrapolated`, bit for bit.
pub(crate) fn untrusted_fill(
    w: usize,
    h: usize,
    low: &Image<bool>,
    radius: f64,
    at: impl Fn(usize) -> (f64, f64, [f64; 4]) + Sync,
) -> Option<Vec<(u32, f64, f64)>> {
    assert_eq!((low.w, low.h), (w, h), "untrusted_fill: mask size");
    let (trusted, nearest) = trusted_sources(low, radius)?;
    Some(
        (0..w * h)
            .into_par_iter()
            .filter(|&p| !trusted.data[p])
            .map(|p| {
                let (x, y) = (p % w, p / w);
                let n = nearest[p] as usize;
                let (dx, dy) = (x as f64 - (n % w) as f64, y as f64 - (n / w) as f64);
                let (ux, uy, g) = at(n);
                (
                    p as u32,
                    ux + g[0] * dx + g[2] * dy,
                    uy + g[1] * dx + g[3] * dy,
                )
            })
            .collect(),
    )
}

/// `(ux, uy)` with every untrusted pixel replaced by the first-order extrapolation from its
/// nearest trusted pixel, or `None` if that changes nothing: no pixel is low, or no pixel is
/// trusted, or all are (see `prewarp`).
fn extrapolated(
    ux: &Image<f64>,
    uy: &Image<f64>,
    slope: &[[f64; 4]],
    low: &Image<bool>,
    radius: f64,
) -> Option<(Image<f64>, Image<f64>)> {
    let w = ux.w;
    let (trusted, nearest) = trusted_sources(low, radius)?;
    let (mut fx, mut fy) = (ux.clone(), uy.clone());
    fx.data
        .par_chunks_mut(w)
        .zip(fy.data.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (rx, ry))| {
            for x in 0..w {
                let p = y * w + x;
                if trusted.data[p] {
                    continue;
                }
                let n = nearest[p] as usize;
                let (dx, dy) = (x as f64 - (n % w) as f64, y as f64 - (n / w) as f64);
                let g = slope[n];
                rx[x] = ux.data[n] + g[0] * dx + g[2] * dy;
                ry[x] = uy.data[n] + g[1] * dx + g[3] * dy;
            }
        });
    Some((fx, fy))
}

/// The smoothed pre-warp `s = G_sigma_s * u` that the fine stage warps the second image with,
/// built only from the displacement of trusted pixels (`prewarp` of the reference).
///
/// `low` marks the pixels with confidence below `conf_min` in the previous solve (before the
/// border rule). Trusted: not low, and farther than `R = cfg.auto_edge_guard_px()` (3 spatial
/// sigma of the longest filter, 12.2 px at the defaults; the automatic edge guard, not the user's
/// `edge_guard`) from every low pixel. Every other pixel `X` takes `u(N) + grad u(N) (X - N)` from
/// its nearest trusted pixel `N`, with `grad u(N)` the affine part `slope` of the shape function
/// fitted at `N`; the guard band counts as untrusted (trusting it leaves 0.05 px), and the fill
/// must be first order (a zeroth-order fill leaves ~600 pixels above 3e-3 px). With no trusted
/// pixel, or only trusted pixels, `s` is the plain `gauss_blur`, bit for bit.
pub(crate) fn prewarp(
    ux: &Image<f64>,
    uy: &Image<f64>,
    slope: &[[f64; 4]],
    low: &Image<bool>,
    cfg: &Config,
) -> (Image<f64>, Image<f64>) {
    let filled = prof::time(Stage::Prewarp, || {
        extrapolated(ux, uy, slope, low, cfg.auto_edge_guard_px())
    });
    let (ux, uy) = filled.as_ref().map_or((ux, uy), |(fx, fy)| (fx, fy));
    prof::time(Stage::Smooth, || {
        (gauss_blur(ux, cfg.sigma_s), gauss_blur(uy, cfg.sigma_s))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::pseudo_random_image;

    /// `low_confidence` is `conf < conf_min`, strictly: a pixel whose confidence is exactly
    /// `conf_min` is confident (Python: "conf_min itself is confident"), one just below is low.
    #[test]
    fn prewarp_low_confidence_is_strictly_below_conf_min() {
        let (w, h) = (8usize, 4usize);
        // uniform energy: the texture factor is 1, the confidence is exp(-(resid / phase_tol)^2)
        let energy = Image::from_vec(w, h, vec![5.0; w * h]);
        let scale = texture_scale(&energy.data);
        let phase_tol = Config::default().phase_tol;
        let conf_of = |resid: f64| confidence(5.0, resid, true, scale, phase_tol);
        let resid0 = 0.3;
        let conf_min = conf_of(resid0);
        let (larger, smaller) = (resid0 * (1.0 + 1e-9), resid0 * (1.0 - 1e-9));
        assert!(conf_min > 0.0 && conf_min < 1.0);
        assert!(conf_of(larger) < conf_min && conf_of(smaller) > conf_min);
        let mut resid = Image::from_vec(w, h, vec![resid0; w * h]);
        (resid.data[1], resid.data[2]) = (larger, smaller);
        let cfg = Config {
            conf_min,
            ..Config::default()
        };
        let low = low_confidence(&energy, &resid, &cfg);
        assert!(!low.data[0], "exactly conf_min is confident");
        assert!(low.data[1], "just below conf_min is low");
        assert!(!low.data[2], "just above conf_min is confident");
        assert_eq!(low.data.iter().filter(|&&l| l).count(), 1);
    }

    /// An affine displacement field of `w` x `h` pixels and its exact constant slope.
    fn affine_field(w: usize, h: usize) -> (Image<f64>, Image<f64>, Vec<[f64; 4]>) {
        let (mut ux, mut uy) = (Image::new(w, h), Image::new(w, h));
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                ux.data[y * w + x] = 3.0 + 0.02 * fx - 0.01 * fy;
                uy.data[y * w + x] = -2.0 + 0.005 * fx + 0.015 * fy;
            }
        }
        (ux, uy, vec![[0.02, 0.005, -0.01, 0.015]; w * h])
    }

    /// Port of Python's
    /// `test_prewarp_replaces_untrusted_pixels_by_the_first_order_extrapolation_of_the_trusted_ones`:
    /// a block where the coarse stage failed (low confidence, displacement +100 px) and a ring of
    /// pixels around it that look confident but lie closer than the filter reach (12.2 px) to the
    /// block and carry garbage too. Whatever the untrusted values are, they are replaced by
    /// `u(N) + grad u(N) (X - N)` from the nearest trusted pixel `N`, which reproduces an affine
    /// field exactly: the pre-warp is the plain blur of the clean field.
    #[test]
    fn prewarp_replaces_untrusted_pixels_by_the_first_order_extrapolation() {
        let cfg = Config::default();
        let (w, h) = (112usize, 96usize);
        let (ux, uy, slope) = affine_field(w, h);
        let block = Image::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| (40..80).contains(&(i % w)) && (30..60).contains(&(i / w)))
                .collect(),
        );
        // distance to the block (0 on it)
        let outside = Image::from_vec(w, h, block.data.iter().map(|&b| !b).collect());
        let dist = distance_to_invalid(&outside);
        let near = |p: usize, r: f64| f64::from(dist.data[p]) <= r;
        assert!(cfg.auto_edge_guard_px() > 8.0);
        assert!((0..w * h).any(|p| !block.data[p] && !near(p, cfg.auto_edge_guard_px())));
        let (mut bad_x, mut bad_y) = (ux.clone(), uy.clone());
        for p in 0..w * h {
            if block.data[p] {
                (bad_x.data[p], bad_y.data[p]) = (100.0, 100.0);
            } else if near(p, 8.0) {
                (bad_x.data[p], bad_y.data[p]) = (50.0, -50.0);
            }
        }
        // the ring is "confident" (not low), the block is low
        let (sx, sy) = prewarp(&bad_x, &bad_y, &slope, &block, &cfg);
        let (wx, wy) = (gauss_blur(&ux, cfg.sigma_s), gauss_blur(&uy, cfg.sigma_s));
        let worst = |a: &Image<f64>, b: &Image<f64>| {
            a.data
                .iter()
                .zip(&b.data)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        };
        assert!(worst(&sx, &wx) < 1e-12, "sx: {}", worst(&sx, &wx));
        assert!(worst(&sy, &wy) < 1e-12, "sy: {}", worst(&sy, &wy));
        // without the pre-warp's rule (the plain blur of the garbage) the field is far off
        let plain = gauss_blur(&bad_x, cfg.sigma_s);
        assert!(worst(&plain, &wx) > 1.0);
    }

    /// Python's `test_prewarp_without_untrusted_pixels_is_the_plain_blur`: with every pixel
    /// trusted, with none trusted (nothing to extrapolate from), the pre-warp is `gauss_blur`
    /// itself, bit for bit; a single low pixel untrusts its neighbourhood.
    #[test]
    fn prewarp_without_untrusted_or_without_trusted_pixels_is_the_plain_blur() {
        let cfg = Config::default();
        let (w, h) = (80usize, 64usize);
        let noise = |seed: u64| pseudo_random_image(w, h, seed);
        let (ux, uy) = (noise(1), noise(2));
        let slope: Vec<[f64; 4]> = noise(3)
            .data
            .iter()
            .map(|&v| [v, -v, 2.0 * v, 0.5])
            .collect();
        let (bx, by) = (gauss_blur(&ux, cfg.sigma_s), gauss_blur(&uy, cfg.sigma_s));
        let bits = |im: &Image<f64>| im.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let none_low = Image::from_vec(w, h, vec![false; w * h]);
        let all_low = Image::from_vec(w, h, vec![true; w * h]);
        for (what, low) in [
            ("every pixel trusted", &none_low),
            ("none trusted", &all_low),
        ] {
            let (sx, sy) = prewarp(&ux, &uy, &slope, low, &cfg);
            assert_eq!(bits(&sx), bits(&bx), "{what}");
            assert_eq!(bits(&sy), bits(&by), "{what}");
        }
        let mut one_low = none_low;
        one_low.data[10 * w + 10] = true;
        let (sx, _) = prewarp(&ux, &uy, &slope, &one_low, &cfg);
        assert_ne!(bits(&sx), bits(&bx));
    }

    /// The fill is skipped when no pixel is trusted or every one is, and only then: a count of
    /// trusted pixels that happens to be `w + h` (11 on a 6 x 5 image, not `w * h` = 30) is
    /// neither. With a radius of 0.5 the trusted pixels are exactly the ones that are not low.
    #[test]
    fn extrapolation_runs_when_the_trusted_count_is_width_plus_height() {
        let (w, h) = (6usize, 5usize);
        let (ux, uy, slope) = affine_field(w, h);
        let low_except = |trusted: usize| {
            Image::from_vec(
                w,
                h,
                (0..w * h).map(|i| i >= trusted).collect::<Vec<bool>>(),
            )
        };
        for trusted in [1, w, w + h - 1, w + h, w + h + 1, w * h - 1] {
            let filled = extrapolated(&ux, &uy, &slope, &low_except(trusted), 0.5);
            assert!(filled.is_some(), "{trusted} of {} pixels trusted", w * h);
        }
        for trusted in [0, w * h] {
            let filled = extrapolated(&ux, &uy, &slope, &low_except(trusted), 0.5);
            assert!(filled.is_none(), "{trusted} of {} pixels trusted", w * h);
        }
    }

    /// `untrusted_fill` (the GPU backend's form: the untrusted pixels only, the field read through
    /// a closure) is `extrapolated` bit for bit: the same pixels take the same values, and both
    /// skip the fill in the same cases.
    #[test]
    fn untrusted_fill_is_the_extrapolation_bit_for_bit() {
        let (w, h) = (96usize, 80usize);
        let noise = |seed: u64| pseudo_random_image(w, h, seed);
        let (ux, uy) = (noise(1), noise(2));
        let (s0, s1) = (noise(3), noise(4));
        let slope: Vec<[f64; 4]> = (0..w * h)
            .map(|i| [s0.data[i], s1.data[i], -s1.data[i], 0.5 * s0.data[i]])
            .collect();
        let radius = Config::default().auto_edge_guard_px();
        let block = |i: usize| (20..34).contains(&(i % w)) && (10..30).contains(&(i / w));
        let lows = [
            (0..w * h)
                .map(|i| block(i) || i % 997 == 5)
                .collect::<Vec<bool>>(),
            vec![false; w * h],
            vec![true; w * h],
        ];
        let bits = |im: &Image<f64>| im.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for low in lows {
            let low = Image::from_vec(w, h, low);
            let want = extrapolated(&ux, &uy, &slope, &low, radius);
            let got = untrusted_fill(w, h, &low, radius, |n| (ux.data[n], uy.data[n], slope[n]));
            match (want, got) {
                (None, None) => {}
                (Some((wx, wy)), Some(fill)) => {
                    let (mut gx, mut gy) = (ux.clone(), uy.clone());
                    for &(p, vx, vy) in &fill {
                        (gx.data[p as usize], gy.data[p as usize]) = (vx, vy);
                    }
                    assert_eq!(bits(&gx), bits(&wx));
                    assert_eq!(bits(&gy), bits(&wy));
                }
                (w, g) => panic!(
                    "extrapolated {:?}, untrusted_fill {:?}",
                    w.is_some(),
                    g.is_some()
                ),
            }
        }
    }
}

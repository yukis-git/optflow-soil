//! Automatic ROI crop (`python/optflow_ref/crop.py` is the specification): track on the ROI's
//! bounding box plus a margin instead of the whole image, and write full-size outputs (NaN outside
//! the crop, confidence 0, valid 0).
//!
//! The rectangle is in pixels of the scaled image, half-open: `[x0, x1) x [y0, y1)`. It starts at
//! the ROI mask's bounding box, grows by the margin on every side, has `x0` and `y0` rounded down
//! to a multiple of `A = grid_step * 2^L` (`L` = the pyramid depth POC uses for this config and
//! image, so the coarse grid of the crop lies on the coarse grid of the whole image) and is
//! clipped to the image.

use std::borrow::Cow;

use anyhow::{bail, Result};
use optflow_core::config::Config;
use optflow_core::gabor::PAD;
use optflow_core::image::Image;
use optflow_core::poc::n_levels;

/// Printed (to stderr) when the material moves close to the crop border.
pub const WARNING: &str = "warning: the material moves close to the crop border \u{2014} increase --crop-margin or use --crop off";

/// Warn when more than this share of the ROI pixels is lost at the crop border.
pub const WARN_FRACTION: f64 = 0.01;

/// `crop.json` when nothing is cropped.
pub const OFF_JSON: &str = "{\n  \"mode\": \"off\"\n}";

/// `--crop-margin` must not be negative.
pub fn check_margin(margin: Option<i64>) -> Result<()> {
    match margin {
        Some(m) if m < 0 => bail!("--crop-margin must be >= 0, got {m}"),
        _ => Ok(()),
    }
}

/// Largest displacement + the fine stage's averaging window (4 sigma_w) + the guard band + the
/// response padding + the footprint of the coarse stage: its grid points sit `poc_win / 2` inside
/// the image at the coarsest pyramid level and the outlier filter looks one grid step further,
/// `(poc_win / 2 + grid_step) * 2^L` at full resolution, so the coarse grid of the crop sees the
/// same context as in the whole `h` x `w` image. A sequence needs twice that: the material moves
/// cumulatively. `cfg` must be valid. The sum saturates at `usize::MAX` for absurd settings
/// (Python's integers do not; the crop is then the whole image either way).
pub fn default_margin(cfg: &Config, h: usize, w: usize, seq: bool) -> usize {
    let levels = n_levels(cfg.max_disp, cfg.poc_win, h, w);
    let scale = 1usize.checked_shl(levels).unwrap_or(usize::MAX);
    let footprint = (cfg.poc_win / 2)
        .saturating_add(cfg.grid_step)
        .saturating_mul(scale);
    // f64 -> usize casts saturate
    let m = [
        cfg.max_disp.ceil() as usize,
        (4.0 * cfg.sigma_w).ceil() as usize,
        cfg.edge_guard_px().ceil() as usize,
        PAD,
        footprint,
    ]
    .into_iter()
    .fold(0usize, usize::saturating_add);
    if seq {
        m.saturating_mul(2)
    } else {
        m
    }
}

/// The alignment `A = grid_step * 2^L` of the crop's start (`L` = the pyramid depth of the `h` x `w`
/// image), `None` when it does not fit in `usize`. Python's integers do not overflow; `validate`
/// bounds `grid_step` at 4096, so this only matters for a `Config` that skipped it.
fn alignment(cfg: &Config, h: usize, w: usize) -> Option<usize> {
    let scale = 1usize.checked_shl(n_levels(cfg.max_disp, cfg.poc_win, h, w))?;
    cfg.grid_step.checked_mul(scale)
}

/// `v // A * A`. With `A = None` (more than `usize::MAX`) every `v` is below `A`: 0, as in Python.
fn align_down(v: usize, a: Option<usize>) -> usize {
    a.map_or(0, |a| v / a * a)
}

/// A crop rectangle of the scaled image (half-open) and the margin it was made with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub margin: usize,
}

/// The crop for an ROI mask. `margin = None`: [`default_margin`]. `cfg` must be valid. Errors:
/// empty mask, negative margin, image or crop smaller than `poc_win`.
pub fn plan(mask: &Image<bool>, cfg: &Config, margin: Option<i64>, seq: bool) -> Result<Crop> {
    let (w, h) = (mask.w, mask.h);
    let (mut xs, mut ys) = ((usize::MAX, 0), (usize::MAX, 0));
    for (y, row) in mask.data.chunks_exact(w.max(1)).enumerate() {
        if let (Some(first), Some(last)) =
            (row.iter().position(|&b| b), row.iter().rposition(|&b| b))
        {
            xs = (xs.0.min(first), xs.1.max(last));
            ys = (ys.0.min(y), ys.1.max(y));
        }
    }
    if xs.0 == usize::MAX {
        bail!("ROI mask is empty");
    }
    check_margin(margin)?;
    if h.min(w) < cfg.poc_win {
        bail!("image smaller than poc_win ({}): ({h}, {w})", cfg.poc_win);
    }
    let margin = margin.map_or_else(|| default_margin(cfg, h, w, seq), |m| m as usize);
    let a = alignment(cfg, h, w);
    let crop = Crop {
        x0: align_down(xs.0.saturating_sub(margin), a),
        y0: align_down(ys.0.saturating_sub(margin), a),
        x1: (xs.1 + 1).saturating_add(margin).min(w),
        y1: (ys.1 + 1).saturating_add(margin).min(h),
        margin,
    };
    if crop.w().min(crop.h()) < cfg.poc_win {
        bail!(
            "the crop {}x{} is smaller than poc_win ({}): increase --crop-margin or use --crop off",
            crop.w(),
            crop.h(),
            cfg.poc_win
        );
    }
    Ok(crop)
}

impl Crop {
    pub fn w(&self) -> usize {
        self.x1 - self.x0
    }

    pub fn h(&self) -> usize {
        self.y1 - self.y0
    }

    /// `crop.json`, formatted as Python's `json.dumps(..., indent=2)`.
    pub fn json(&self) -> String {
        format!(
            "{{\n  \"x0\": {},\n  \"y0\": {},\n  \"x1\": {},\n  \"y1\": {},\n  \"margin\": {},\n  \"mode\": \"auto\"\n}}",
            self.x0, self.y0, self.x1, self.y1, self.margin
        )
    }

    /// Whether the crop is the whole `w` x `h` image (the default margin often makes it so).
    pub fn is_whole(&self, w: usize, h: usize) -> bool {
        (self.x0, self.y0, self.x1, self.y1) == (0, 0, w, h)
    }

    /// The part of a full-size image inside the crop.
    pub fn apply<T: Copy + Default>(&self, im: &Image<T>) -> Image<T> {
        let mut data = Vec::with_capacity(self.w() * self.h());
        for y in self.y0..self.y1 {
            data.extend_from_slice(&im.data[y * im.w + self.x0..y * im.w + self.x1]);
        }
        Image::from_vec(self.w(), self.h(), data)
    }

    /// `im` (the crop's size) placed in a `w` x `h` image filled with `fill`.
    fn put(&self, im: &Image<f32>, fill: f32, w: usize, h: usize) -> Image<f32> {
        let mut data = vec![fill; w * h];
        for (y, row) in im.data.chunks_exact(im.w.max(1)).enumerate() {
            let at = (self.y0 + y) * w + self.x0;
            data[at..at + im.w].copy_from_slice(row);
        }
        Image::from_vec(w, h, data)
    }

    /// A map of the result of tracking the crop as a map of the full `w` x `h` image: `fill`
    /// outside the crop (NaN, or 0 for the confidence). `im` itself when the crop is the whole
    /// image. The outputs are placed one map at a time as they are written, so the full-size
    /// copy of a whole result is never held.
    pub fn place<'a>(
        &self,
        im: &'a Image<f32>,
        fill: f32,
        w: usize,
        h: usize,
    ) -> Cow<'a, Image<f32>> {
        if self.is_whole(w, h) {
            Cow::Borrowed(im)
        } else {
            Cow::Owned(self.put(im, fill, w, h))
        }
    }

    /// Share of the ROI pixels whose match lies within `B = 4 sigma_w` of the crop border (or outside
    /// it), which the border rule of the fine stage sets to confidence 0. `raw_ux`, `raw_uy`: the
    /// unmasked displacement of the crop, `mask`: its ROI, `(w, h)`: the size of the whole image.
    /// Only sides the crop introduced count (not those on the border of the image, which are lost
    /// without a crop as well); 0 when there are none.
    pub fn border_fraction(
        &self,
        raw_ux: &Image<f32>,
        raw_uy: &Image<f32>,
        mask: &Image<bool>,
        (w, h): (usize, usize),
        cfg: &Config,
    ) -> f64 {
        let sides = [self.x0 > 0, self.y0 > 0, self.x1 < w, self.y1 < h];
        let total = mask.data.iter().filter(|&&b| b).count();
        if !sides.contains(&true) || total == 0 {
            return 0.0;
        }
        let b = if cfg.edge_guard_px() > 0.0 {
            4.0 * cfg.sigma_w
        } else {
            0.0
        };
        let (cw, ch) = (mask.w, mask.h);
        let mut lost = 0usize;
        for y in 0..ch {
            for x in 0..cw {
                let i = y * cw + x;
                if !mask.data[i] {
                    continue;
                }
                let tx = x as f64 + f64::from(raw_ux.data[i]);
                let ty = y as f64 + f64::from(raw_uy.data[i]);
                // NaN compares false: no match counts as outside
                let inside = |d: f64, introduced: bool| !introduced || d >= b;
                let ok = inside(tx, sides[0])
                    && inside(ty, sides[1])
                    && inside((cw - 1) as f64 - tx, sides[2])
                    && inside((ch - 1) as f64 - ty, sides[3])
                    && tx.is_finite()
                    && ty.is_finite();
                lost += usize::from(!ok);
            }
        }
        lost as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(
        w: usize,
        h: usize,
        x: std::ops::Range<usize>,
        y: std::ops::Range<usize>,
    ) -> Image<bool> {
        let mut m = Image::<bool>::new(w, h);
        for yy in y {
            for xx in x.clone() {
                m.set(xx, yy, true);
            }
        }
        m
    }

    #[test]
    fn default_margin_is_displacement_plus_window_plus_guard_plus_padding_plus_coarse_footprint() {
        // L = 1 level: (64 / 2 + 16) * 2 = 96 for the coarse stage
        let cfg = Config::default();
        assert_eq!(
            default_margin(&cfg, 768, 768, false),
            32 + 16 + 13 + 32 + 96
        );
        assert_eq!(default_margin(&cfg, 768, 768, true), 2 * 189);
        // L = 3 levels: 48 * 8 = 384
        let cfg = Config {
            max_disp: 100.5,
            sigma_w: 6.0,
            edge_guard: 5.0,
            ..Config::default()
        };
        assert_eq!(
            default_margin(&cfg, 768, 768, false),
            101 + 24 + 5 + 32 + 384
        );
        // a small image gets fewer levels
        let cfg = Config::default();
        assert_eq!(n_levels(cfg.max_disp, cfg.poc_win, 40, 40), 0);
        assert_eq!(default_margin(&cfg, 40, 40, false), 32 + 16 + 13 + 32 + 48);
    }

    #[test]
    fn rect_is_the_roi_bounding_box_plus_margin_aligned_down_to_the_coarse_grid() {
        // L = 1, A = 32: 300 - 189 = 111 rounds down to 96
        let c = plan(
            &mask(768, 768, 300..400, 300..400),
            &Config::default(),
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            c,
            Crop {
                x0: 96,
                y0: 96,
                x1: 589,
                y1: 589,
                margin: 189
            }
        );
        // max_disp = 128: L = 3 levels, A = 16 * 8 = 128, margin 128 + 16 + 13 + 32 + 48 * 8
        let cfg = Config {
            max_disp: 128.0,
            ..Config::default()
        };
        let c = plan(&mask(2048, 2048, 1000..1080, 1000..1080), &cfg, None, false).unwrap();
        assert_eq!(c.margin, 573);
        assert_eq!((c.x0, c.y0), (384, 384)); // 1000 - 573 = 427 rounds down to 384
        assert_eq!((c.x1, c.y1), (1080 + 573, 1080 + 573)); // only the start is aligned
        assert_eq!(
            c.json(),
            "{\n  \"x0\": 384,\n  \"y0\": 384,\n  \"x1\": 1653,\n  \"y1\": 1653,\n  \"margin\": 573,\n  \"mode\": \"auto\"\n}"
        );
    }

    #[test]
    fn alignment_beyond_usize_rounds_every_start_down_to_zero_like_pythons_big_integers() {
        // `validate` bounds grid_step at 4096, so these configs are rejected before they get here;
        // the arithmetic must not depend on that (2^63 << 1 used to wrap to 0 and divide by zero)
        let m = mask(400, 300, 350..400, 120..200);
        for grid_step in [1usize << 63, 1 << 62, (1 << 63) + 1, usize::MAX] {
            let cfg = Config {
                grid_step,
                ..Config::default()
            };
            assert_eq!(n_levels(cfg.max_disp, cfg.poc_win, 300, 400), 1);
            // the default margin is huge (it saturates): the crop is the whole image
            let c = plan(&m, &cfg, None, false).unwrap();
            assert!(c.margin > 1 << 62, "{c:?}");
            assert_eq!((c.x0, c.y0, c.x1, c.y1), (0, 0, 400, 300));
            // a given margin: 350 - 10 and 120 - 10 are below A = grid_step * 2 and round down to 0
            let c = plan(&m, &cfg, Some(10), false).unwrap();
            assert_eq!(
                c,
                Crop {
                    x0: 0,
                    y0: 0,
                    x1: 400,
                    y1: 210,
                    margin: 10
                }
            );
        }
        // an alignment that still fits in usize rounds as usual; one that does not is `None`
        let at = |grid_step| {
            alignment(
                &Config {
                    grid_step,
                    ..Config::default()
                },
                300,
                400,
            )
        };
        assert_eq!(at(1 << 62), Some(1 << 63)); // L = 1
        assert_eq!(at(1 << 63), None);
        assert_eq!(align_down(usize::MAX, at(1 << 62)), 1 << 63);
        assert_eq!(align_down(usize::MAX, at(1 << 63)), 0);
        assert_eq!(align_down(77, at(16)), 64);
    }

    #[test]
    fn rect_is_clipped_to_the_image_and_a_given_margin_replaces_the_default() {
        let m = mask(400, 300, 350..400, 0..80);
        let c = plan(&m, &Config::default(), None, false).unwrap();
        assert_eq!((c.x0, c.y0, c.x1, c.y1), (160, 0, 400, 269)); // 350 - 189 = 161 -> 160
        let c = plan(&m, &Config::default(), Some(10), false).unwrap();
        assert_eq!((c.x0, c.y0, c.x1, c.y1, c.margin), (320, 0, 400, 90, 10)); // 340 -> 320
        assert_eq!(
            plan(&m, &Config::default(), None, true).unwrap().margin,
            378
        );
    }

    #[test]
    fn plan_rejects_an_empty_roi_a_negative_margin_and_a_crop_below_poc_win() {
        let cfg = Config::default();
        let e = |m: &Image<bool>, margin| plan(m, &cfg, margin, false).unwrap_err().to_string();
        assert_eq!(e(&Image::<bool>::new(400, 300), None), "ROI mask is empty");
        assert_eq!(
            e(&mask(400, 300, 0..400, 0..300), Some(-1)),
            "--crop-margin must be >= 0, got -1"
        );
        assert_eq!(
            e(&mask(400, 300, 100..110, 100..110), Some(0)),
            "the crop 14x14 is smaller than poc_win (64): increase --crop-margin or use --crop off"
        );
        assert_eq!(
            e(&mask(40, 32, 0..40, 0..32), None),
            "image smaller than poc_win (64): (32, 40)"
        );
    }

    fn fraction(ux: f32, m: &Image<bool>, crop: Crop, size: (usize, usize), cfg: &Config) -> f64 {
        let u = Image::from_vec(m.w, m.h, vec![ux; m.w * m.h]);
        crop.border_fraction(&u, &Image::new(m.w, m.h), m, size, cfg)
    }

    #[test]
    fn border_fraction_counts_roi_pixels_whose_match_is_within_4_sigma_w_of_the_crop_border() {
        let cfg = Config::default();
        let crop = Crop {
            x0: 100,
            y0: 100,
            x1: 200,
            y1: 200,
            margin: 20,
        };
        let inner = mask(100, 100, 30..70, 30..70);
        let all = mask(100, 100, 0..100, 0..100);
        let f = |ux, m: &Image<bool>, cfg: &Config| fraction(ux, m, crop, (400, 400), cfg);
        assert_eq!(f(0.0, &inner, &cfg), 0.0);
        assert!(f(25.0, &inner, &cfg) > 0.2); // x + 25 > 99 - 16 for x >= 59
        assert_eq!(f(5.0, &inner, &cfg), 0.0);
        assert_eq!(f(f32::NAN, &inner, &cfg), 1.0); // no match at all counts as outside
        assert!((f(0.0, &all, &cfg) - (1.0 - 68.0 * 68.0 / 10000.0)).abs() < 1e-12);
        // the guard off: only a match outside the crop counts
        let off = Config {
            edge_guard: 0.0,
            ..Config::default()
        };
        assert_eq!(f(0.0, &all, &off), 0.0);
        assert!((f(3.0, &all, &off) - 0.03).abs() < 1e-12);
    }

    /// The displacement is read in both axes with its own sign: `x + ux`, `y + uy`, and the
    /// distances to the far sides `(w - 1) - tx`, `(h - 1) - ty`. The expected share is counted
    /// pixel by pixel here.
    #[test]
    fn border_fraction_follows_the_match_in_x_and_y() {
        let cfg = Config::default();
        let b = 4.0 * cfg.sigma_w;
        let crop = Crop {
            x0: 100,
            y0: 100,
            x1: 200,
            y1: 200,
            margin: 20,
        };
        // an ROI on one side only, displaced towards the inside of the image or away from it: the
        // shares differ with the sign of the displacement (for the whole image they would not)
        let cases = [
            (mask(100, 100, 0..100, 0..100), (12.0f32, -7.0f32)),
            (mask(100, 100, 0..100, 0..100), (0.0, 25.0)),
            (mask(100, 100, 0..30, 0..100), (20.0, 0.0)),
            (mask(100, 100, 0..30, 0..100), (-20.0, 0.0)),
            (mask(100, 100, 70..100, 0..100), (-20.0, 0.0)),
            (mask(100, 100, 0..100, 0..30), (0.0, 20.0)),
            (mask(100, 100, 0..100, 0..30), (0.0, -20.0)),
            (mask(100, 100, 0..100, 70..100), (0.0, -20.0)),
            (mask(100, 100, 0..100, 70..100), (0.0, 20.0)),
        ];
        for (roi, (dx, dy)) in &cases {
            let ux = Image::from_vec(100, 100, vec![*dx; 100 * 100]);
            let uy = Image::from_vec(100, 100, vec![*dy; 100 * 100]);
            let got = crop.border_fraction(&ux, &uy, roi, (400, 400), &cfg);
            let (mut lost, mut total) = (0usize, 0usize);
            for y in 0..100 {
                for x in 0..100 {
                    if !roi.data[y * 100 + x] {
                        continue;
                    }
                    let (tx, ty) = (x as f64 + f64::from(*dx), y as f64 + f64::from(*dy));
                    let ok = tx >= b && ty >= b && 99.0 - tx >= b && 99.0 - ty >= b;
                    lost += usize::from(!ok);
                    total += 1;
                }
            }
            let want = lost as f64 / total as f64;
            assert!((got - want).abs() < 1e-12, "({dx}, {dy}): {got} != {want}");
        }
        // a NaN in one component alone loses the pixel
        let ux = Image::from_vec(100, 100, vec![0.0f32; 100 * 100]);
        let uy = Image::from_vec(100, 100, vec![f32::NAN; 100 * 100]);
        let inner = mask(100, 100, 30..70, 30..70);
        assert_eq!(
            crop.border_fraction(&ux, &uy, &inner, (400, 400), &cfg),
            1.0
        );
    }

    /// With no side of the crop introduced by it (or no ROI pixel) the share is 0, whatever the
    /// displacement: nothing is divided by zero or counted from a NaN.
    #[test]
    fn border_fraction_is_zero_without_crop_sides_or_roi_pixels() {
        let cfg = Config::default();
        let all = mask(100, 100, 0..100, 0..100);
        let whole = Crop {
            x0: 0,
            y0: 0,
            x1: 100,
            y1: 100,
            margin: 20,
        };
        assert_eq!(fraction(f32::NAN, &all, whole, (100, 100), &cfg), 0.0);
        let inner = Crop {
            x0: 10,
            y0: 10,
            x1: 110,
            y1: 110,
            margin: 20,
        };
        let none = mask(100, 100, 0..0, 0..0);
        let got = fraction(0.0, &none, inner, (400, 400), &cfg);
        assert_eq!(got, 0.0, "{got}");
    }

    #[test]
    fn border_fraction_ignores_crop_sides_that_are_image_borders() {
        let cfg = Config::default();
        let all = mask(100, 100, 0..100, 0..100);
        let whole = Crop {
            x0: 0,
            y0: 0,
            x1: 100,
            y1: 100,
            margin: 20,
        };
        assert_eq!(fraction(0.0, &all, whole, (100, 100), &cfg), 0.0);
        // only the right side is a crop border (the image is 150 wide and 100 high)
        assert!((fraction(0.0, &all, whole, (150, 100), &cfg) - 0.16).abs() < 1e-12);
        // the left side is an image border, the others are not
        let c = Crop {
            x0: 0,
            y0: 100,
            x1: 100,
            y1: 200,
            margin: 20,
        };
        let want = 1.0 - 84.0 * 68.0 / 10000.0;
        assert!((fraction(0.0, &all, c, (400, 400), &cfg) - want).abs() < 1e-12);
    }

    #[test]
    fn place_is_put_and_borrows_for_a_whole_image_crop() {
        let part = Image::from_vec(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = Crop {
            x0: 2,
            y0: 1,
            x1: 5,
            y1: 3,
            margin: 0,
        };
        assert!(!c.is_whole(6, 4));
        let placed = c.place(&part, 0.0, 6, 4);
        assert!(matches!(placed, Cow::Owned(_)));
        assert_eq!(placed.data, c.put(&part, 0.0, 6, 4).data);
        let whole = Crop {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 2,
            margin: 7,
        };
        assert!(whole.is_whole(3, 2) && !whole.is_whole(4, 2));
        let placed = whole.place(&part, f32::NAN, 3, 2);
        assert!(matches!(placed, Cow::Borrowed(_)));
        // the copy a whole-image crop saves is the image itself
        assert_eq!(whole.put(&part, f32::NAN, 3, 2).data, part.data);
        assert_eq!(whole.apply(&part).data, part.data);
    }

    #[test]
    fn apply_and_put_are_inverse_inside_the_crop() {
        let c = Crop {
            x0: 2,
            y0: 1,
            x1: 5,
            y1: 3,
            margin: 0,
        };
        let full = Image::from_vec(6, 4, (0..24).map(|i| i as f32).collect());
        let part = c.apply(&full);
        assert_eq!((part.w, part.h), (3, 2));
        assert_eq!(part.data, [8.0, 9.0, 10.0, 14.0, 15.0, 16.0]);
        let back = c.put(&part, f32::NAN, 6, 4);
        assert_eq!(back.get(2, 1), 8.0);
        assert_eq!(back.get(4, 2), 16.0);
        assert!(back.get(1, 1).is_nan() && back.get(5, 2).is_nan() && back.get(3, 0).is_nan());
    }
}

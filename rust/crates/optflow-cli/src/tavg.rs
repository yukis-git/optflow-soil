//! Temporal averaging front-end of `seq` (`python/optflow_ref/tavg.py` and `_iter_averaged` of
//! `io.py`): each sampled frame is the mean of the `2 m + 1` frames around it (clipped to the
//! video, the window may reach past `stop`), taken after the integer downscale. With `align`, every
//! neighbour is first aligned to the centre frame with a motion estimated from the coarse POC flow
//! (`tavg_model`: one robust affine fit inside the ROI, or the dense coarse flow); a neighbour whose
//! motion cannot be estimated is left out of the mean.
//!
//! Deviation: the Python warp is scipy's cubic `map_coordinates` (mode `nearest`), here the
//! cubic B-spline of `optflow_core::interp::warp`; they agree away from the image border.

use std::collections::BTreeMap;

use anyhow::Result;
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::interp::warp;
use optflow_core::poc::{coarse_flow, grid_sample};

/// Grid points with a valid POC peak needed for the affine fit.
const MIN_POINTS: usize = 6;

/// np.median of a non-empty slice (the mean of the two middle values for an even count).
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Solves `a x = b` (3 x 3, Gaussian elimination with partial pivoting).
fn solve3(mut a: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
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
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut s = a[i][3];
        for j in i + 1..3 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Least-squares affine `v(x, y) = c0 + c1 x + c2 y` through the selected points (centred
/// coordinates for conditioning; the result is in absolute coordinates).
fn lstsq(xs: &[f64], ys: &[f64], v: &[f64], sel: &[bool]) -> Option<[f64; 3]> {
    let idx: Vec<usize> = (0..xs.len()).filter(|&i| sel[i]).collect();
    let n = idx.len() as f64;
    let (mx, my) = (
        idx.iter().map(|&i| xs[i]).sum::<f64>() / n,
        idx.iter().map(|&i| ys[i]).sum::<f64>() / n,
    );
    let mut m = [[0.0f64; 4]; 3];
    for &i in &idx {
        let row = [1.0, xs[i] - mx, ys[i] - my];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += row[r] * row[c];
            }
            m[r][3] += row[r] * v[i];
        }
    }
    let c = solve3(m)?;
    Some([c[0] - c[1] * mx - c[2] * my, c[1], c[2]])
}

/// Robust affine through scattered displacements: points whose residual exceeds
/// `max(3 x 1.4826 MAD, 0.05 px)` of the previous fit are dropped, 4 passes. `None` with fewer
/// than [`MIN_POINTS`] points left.
fn fit_affine(xs: &[f64], ys: &[f64], ux: &[f64], uy: &[f64]) -> Option<([f64; 3], [f64; 3])> {
    let mut sel = vec![true; xs.len()];
    let mut coef = None;
    for _ in 0..4 {
        if sel.iter().filter(|&&s| s).count() < MIN_POINTS {
            return None;
        }
        let cx = lstsq(xs, ys, ux, &sel)?;
        let cy = lstsq(xs, ys, uy, &sel)?;
        coef = Some((cx, cy));
        let r: Vec<f64> = (0..xs.len())
            .map(|i| {
                (cx[0] + cx[1] * xs[i] + cx[2] * ys[i] - ux[i])
                    .hypot(cy[0] + cy[1] * xs[i] + cy[2] * ys[i] - uy[i])
            })
            .collect();
        let mut rs: Vec<f64> = (0..xs.len()).filter(|&i| sel[i]).map(|i| r[i]).collect();
        let mad = 1.4826 * median(&mut rs);
        let lim = (3.0 * mad).max(0.05);
        sel = r.iter().map(|&v| v <= lim).collect();
    }
    coef
}

/// `other` resampled onto the pixel grid of `centre`, or `None` when no motion could be
/// estimated: `aligned(x) = other(x + u(x))` with `u` the affine fit to the coarse POC flow
/// centre -> other (only grid points inside `mask`, so that the motion of the region of interest
/// is the one removed), or with `cfg.tavg_model == "dense"` the coarse flow itself, interpolated
/// bilinearly.
pub fn align_to(
    centre: &Image<f32>,
    other: &Image<f32>,
    cfg: &Config,
    mask: Option<&Image<bool>>,
) -> Option<Image<f32>> {
    let (w, h) = (centre.w, centre.h);
    let g = coarse_flow(centre, other, cfg, None);
    let mut ux = Image::<f32>::new(w, h);
    let mut uy = Image::<f32>::new(w, h);
    if cfg.tavg_model == "dense" {
        for y in 0..h {
            for x in 0..w {
                let (a, b) = grid_sample(&g, x as f64, y as f64);
                ux.data[y * w + x] = a as f32;
                uy.data[y * w + x] = b as f32;
            }
        }
    } else {
        let (mut xs, mut ys, mut vx, mut vy) = (vec![], vec![], vec![], vec![]);
        for (iy, &y) in g.ys.iter().enumerate() {
            for (ix, &x) in g.xs.iter().enumerate() {
                let j = iy * g.xs.len() + ix;
                if g.peak.data[j] > 0.0 && mask.is_none_or(|m| m.data[y * w + x]) {
                    xs.push(x as f64);
                    ys.push(y as f64);
                    vx.push(f64::from(g.ux.data[j]));
                    vy.push(f64::from(g.uy.data[j]));
                }
            }
        }
        if xs.len() < MIN_POINTS {
            return None;
        }
        let (cx, cy) = fit_affine(&xs, &ys, &vx, &vy)?;
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                ux.data[y * w + x] = (cx[0] + cx[1] * fx + cx[2] * fy) as f32;
                uy.data[y * w + x] = (cy[0] + cy[1] * fx + cy[2] * fy) as f32;
            }
        }
    }
    Some(warp(other, &ux, &uy))
}

/// Mean of `frames` (the window around `frames[centre]`). With `align = Some((cfg, mask))` every
/// other frame is first aligned to the centre ([`align_to`]); one whose motion cannot be
/// estimated is left out. The sum runs in `f64`, the centre first, then the others in order.
pub fn average_window(
    frames: &[&Image<f32>],
    centre: usize,
    align: Option<(&Config, Option<&Image<bool>>)>,
) -> Image<f32> {
    let c = frames[centre];
    let mut used: Vec<Image<f32>> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        if i == centre {
            continue;
        }
        match align {
            Some((cfg, mask)) => {
                if let Some(a) = align_to(c, f, cfg, mask) {
                    used.push(a);
                }
            }
            None => used.push((*f).clone()),
        }
    }
    if used.is_empty() {
        return c.clone();
    }
    let n = (used.len() + 1) as f64;
    let data = (0..c.data.len())
        .map(|i| {
            let s: f64 =
                f64::from(c.data[i]) + used.iter().map(|u| f64::from(u.data[i])).sum::<f64>();
            (s / n) as f32
        })
        .collect();
    Image::from_vec(c.w, c.h, data)
}

/// The ROI mask of a `w` x `h` frame (`None`: no ROI).
pub type MaskFn = Box<dyn Fn(usize, usize) -> Result<Option<Image<bool>>>>;

/// The averaged frames `start, start + step, ... < stop` of `raw`, which yields every frame with
/// index `first, first + 1, ...` (downscaled), `first = max(0, start - m)`.
pub struct Averaged {
    raw: Box<dyn Iterator<Item = Result<Image<f32>>>>,
    m: usize,
    step: usize,
    stop: Option<i64>,
    align: Option<Config>,
    mask: MaskFn,
    buf: BTreeMap<usize, Image<f32>>,
    /// Index of the next frame to read.
    read: usize,
    /// Index of the last frame read (None before the first).
    last: Option<usize>,
    /// Index of the next sampled frame.
    nxt: usize,
    ended: bool,
}

impl Averaged {
    pub fn new(
        raw: Box<dyn Iterator<Item = Result<Image<f32>>>>,
        start: usize,
        stop: Option<i64>,
        step: usize,
        m: usize,
        align: Option<Config>,
        mask: MaskFn,
    ) -> Averaged {
        Averaged {
            raw,
            m,
            step,
            stop,
            align,
            mask,
            buf: BTreeMap::new(),
            read: start.saturating_sub(m),
            last: None,
            nxt: start,
            ended: false,
        }
    }

    /// Whether the window of sampled frame `nxt` is complete when frame `ready` is the last one
    /// that counts (and `nxt` is before `stop`).
    fn due(&self, ready: Option<usize>) -> bool {
        self.stop.is_none_or(|s| (self.nxt as i64) < s) && ready.is_some_and(|r| self.nxt <= r)
    }

    fn emit(&mut self) -> Result<Image<f32>> {
        let s = self.nxt;
        let last = self
            .last
            .expect("a window is emitted after a frame was read");
        let (lo, hi) = (s.saturating_sub(self.m), (s + self.m).min(last));
        let frames: Vec<&Image<f32>> = (lo..=hi).map(|i| &self.buf[&i]).collect();
        let out = if frames.len() == 1 {
            frames[0].clone()
        } else {
            let mask = match &self.align {
                Some(_) => (self.mask)(frames[0].w, frames[0].h)?,
                None => None,
            };
            let align = self.align.as_ref().map(|c| (c, mask.as_ref()));
            average_window(&frames, s - lo, align)
        };
        self.nxt += self.step;
        let keep_from = self.nxt.saturating_sub(self.m);
        self.buf.retain(|&j, _| j >= keep_from);
        Ok(out)
    }
}

impl Iterator for Averaged {
    type Item = Result<Image<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let ready = if self.ended {
                self.last
            } else {
                self.last.and_then(|l| l.checked_sub(self.m))
            };
            if self.due(ready) {
                let r = self.emit();
                if r.is_err() {
                    self.ended = true;
                    self.last = None;
                }
                return Some(r);
            }
            if self.ended {
                return None;
            }
            match self.raw.next() {
                None => self.ended = true,
                Some(Err(e)) => {
                    self.ended = true;
                    self.last = None;
                    return Some(Err(e));
                }
                Some(Ok(f)) => {
                    self.buf.insert(self.read, f);
                    self.last = Some(self.read);
                    self.read += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w` x 1... frames whose single value is the frame index.
    fn indexed(n: usize, first: usize) -> Box<dyn Iterator<Item = Result<Image<f32>>>> {
        Box::new((first..first + n).map(|i| Ok(Image::from_vec(2, 2, vec![i as f32; 4]))))
    }

    fn values(a: Averaged) -> Vec<f32> {
        a.map(|f| f.unwrap().data[0]).collect()
    }

    fn none_mask() -> MaskFn {
        Box::new(|_, _| Ok(None))
    }

    #[test]
    fn windows_are_centred_and_clipped_to_the_video() {
        // 10 frames 0..9, m = 2, every 3rd from 1: samples 1, 4, 7 (stop None: 10 would be next)
        let a = Averaged::new(indexed(10, 0), 1, None, 3, 2, None, none_mask());
        // windows [0..3], [2..6], [5..9]: means 1.5, 4, 7
        assert_eq!(values(a), vec![1.5, 4.0, 7.0]);
        // the last windows are clipped at the end of the video
        let a = Averaged::new(indexed(10, 0), 0, None, 4, 1, None, none_mask());
        // samples 0, 4, 8: [0..1] 0.5, [3..5] 4, [7..9] 8
        assert_eq!(values(a), vec![0.5, 4.0, 8.0]);
    }

    #[test]
    fn the_window_may_reach_past_stop_but_samples_stop_before_it() {
        // samples 2 and 4 (stop 6, step 2), m = 3: the first window is clipped at the start (0..5), the
        // second reaches past stop (1..7)
        let a = Averaged::new(indexed(9, 0), 2, Some(6), 2, 3, None, none_mask());
        assert_eq!(values(a), vec![2.5, 4.0]);
        // a reader that begins later (start - m = 3 for start 6) numbers its frames from there; the
        // windows at the end are clipped at the last frame (10)
        let a = Averaged::new(indexed(8, 3), 6, None, 1, 3, None, none_mask());
        assert_eq!(values(a), vec![6.0, 7.0, 7.5, 8.0, 8.5]);
    }

    #[test]
    fn a_single_frame_window_is_the_frame() {
        let a = Averaged::new(indexed(3, 0), 0, None, 1, 0, None, none_mask());
        assert_eq!(values(a), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn the_mean_of_noisy_frames_has_less_noise() {
        let mut s = 7u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32
        };
        let frames: Vec<Image<f32>> = (0..9)
            .map(|_| Image::from_vec(16, 16, (0..256).map(|_| rnd()).collect()))
            .collect();
        let refs: Vec<&Image<f32>> = frames.iter().collect();
        let mean = average_window(&refs, 4, None);
        let rms = |im: &Image<f32>| {
            (im.data.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 256.0).sqrt()
        };
        assert!(rms(&mean) < rms(&frames[4]) / 2.0);
    }

    #[test]
    fn affine_fit_recovers_a_plane_and_drops_outliers() {
        let (mut xs, mut ys, mut ux, mut uy) = (vec![], vec![], vec![], vec![]);
        for j in 0..6 {
            for i in 0..8 {
                let (x, y) = (16.0 * i as f64 + 20.0, 16.0 * j as f64 + 30.0);
                xs.push(x);
                ys.push(y);
                ux.push(0.5 + 0.001 * x - 0.002 * y);
                uy.push(-1.0 + 0.003 * y);
            }
        }
        ux[10] += 5.0; // an outlier
        let (cx, cy) = fit_affine(&xs, &ys, &ux, &uy).unwrap();
        assert!(
            (cx[0] - 0.5).abs() < 1e-9
                && (cx[1] - 0.001).abs() < 1e-9
                && (cx[2] + 0.002).abs() < 1e-9
        );
        assert!((cy[0] + 1.0).abs() < 1e-9 && cy[1].abs() < 1e-9 && (cy[2] - 0.003).abs() < 1e-9);
        assert!(fit_affine(&xs[..4], &ys[..4], &ux[..4], &uy[..4]).is_none());
    }
}

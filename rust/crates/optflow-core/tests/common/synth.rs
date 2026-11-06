//! Synthetic speckle images with an exactly known affine displacement (a small port of
//! `python/optflow_ref/synth.py`: Gaussian blobs evaluated analytically at inverse-mapped
//! coordinates; the random numbers are this file's own, not numpy's).
#![allow(dead_code)]

use std::f64::consts::PI;

use optflow_core::image::Image;

pub struct Pattern {
    cx: Vec<f64>,
    cy: Vec<f64>,
    r: Vec<f64>,
    a: Vec<f64>,
}

/// Blobs of radius 1.5 to 4 px, about 60 % area coverage, over the image plus `margin` px on every
/// side, from a seeded generator (the same pattern for the same arguments).
pub fn make_pattern(w: usize, h: usize, seed: u64, margin: f64) -> Pattern {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(12345);
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let (rmin, rmax) = (1.5, 4.0);
    let rmean = 0.5 * (rmin + rmax);
    let n = (0.6 * (h as f64 + 2.0 * margin) * (w as f64 + 2.0 * margin) / (PI * rmean * rmean))
        as usize;
    let mut p = Pattern {
        cx: Vec::with_capacity(n),
        cy: Vec::with_capacity(n),
        r: Vec::with_capacity(n),
        a: Vec::with_capacity(n),
    };
    for _ in 0..n {
        p.cx.push(-margin + (w as f64 + 2.0 * margin) * rnd());
        p.cy.push(-margin + (h as f64 + 2.0 * margin) * rnd());
        p.r.push(rmin + (rmax - rmin) * rnd());
        let sign = if rnd() < 0.5 { -1.0 } else { 1.0 };
        p.a.push((0.3 + 0.7 * rnd()) * sign);
    }
    p
}

/// `x = c + A (X - c) + t`.
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    pub a: [[f64; 2]; 2],
    pub t: [f64; 2],
    pub c: [f64; 2],
}

impl Affine {
    pub fn fwd(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.c[0], y - self.c[1]);
        (
            self.c[0] + self.a[0][0] * dx + self.a[0][1] * dy + self.t[0],
            self.c[1] + self.a[1][0] * dx + self.a[1][1] * dy + self.t[1],
        )
    }

    pub fn inv(&self, x: f64, y: f64) -> (f64, f64) {
        let [[a, b], [c, d]] = self.a;
        let det = a * d - b * c;
        let (dx, dy) = (x - self.c[0] - self.t[0], y - self.c[1] - self.t[1]);
        (
            self.c[0] + (d * dx - b * dy) / det,
            self.c[1] + (-c * dx + a * dy) / det,
        )
    }

    /// Displacement `(x - X, y - Y)` of the reference pixel `(X, Y)`.
    pub fn displacement(&self, x: f64, y: f64) -> (f64, f64) {
        let (fx, fy) = self.fwd(x, y);
        (fx - x, fy - y)
    }
}

/// `synth.uniform_strain(exx, eyy, center, t=t)` (no shear).
pub fn uniform_strain(exx: f64, eyy: f64, center: (f64, f64), t: (f64, f64)) -> Affine {
    Affine {
        a: [[1.0 + exx, 0.0], [0.0, 1.0 + eyy]],
        t: [t.0, t.1],
        c: [center.0, center.1],
    }
}

/// The pattern after the deformation `field` (`None`: the reference), grey values in `[0, 1]`.
pub fn render(p: &Pattern, w: usize, h: usize, field: Option<&Affine>) -> Image<f32> {
    let mut acc = vec![0.0f64; w * h];
    // an upper bound of how far the deformation stretches a distance
    let stretch = field.map_or(1.0, |f| {
        let s: f64 = f.a.iter().flatten().map(|v| v.abs()).sum();
        s.max(1.0) * 1.01
    });
    for k in 0..p.cx.len() {
        let (px, py) = field.map_or((p.cx[k], p.cy[k]), |f| f.fwd(p.cx[k], p.cy[k]));
        let rad = (5.2 * p.r[k] + 2.0) * stretch;
        let x0 = (px - rad).floor().max(0.0) as usize;
        let x1 = ((px + rad).ceil().max(0.0) as usize).min(w);
        let y0 = (py - rad).floor().max(0.0) as usize;
        let y1 = ((py + rad).ceil().max(0.0) as usize).min(h);
        let inv2r2 = 1.0 / (2.0 * p.r[k] * p.r[k]);
        for y in y0..y1 {
            for x in x0..x1 {
                let (sx, sy) = field.map_or((x as f64, y as f64), |f| f.inv(x as f64, y as f64));
                let (dx, dy) = (sx - p.cx[k], sy - p.cy[k]);
                acc[y * w + x] += p.a[k] * (-(dx * dx + dy * dy) * inv2r2).exp();
            }
        }
    }
    Image::from_vec(
        w,
        h,
        acc.iter().map(|v| (0.5 + 0.25 * v.tanh()) as f32).collect(),
    )
}

/// The exact displacement field of `field` on the `w` x `h` grid, as images.
pub fn displacement(field: &Affine, w: usize, h: usize) -> (Image<f32>, Image<f32>) {
    let (mut ux, mut uy) = (Image::new(w, h), Image::new(w, h));
    for y in 0..h {
        for x in 0..w {
            let (u, v) = field.displacement(x as f64, y as f64);
            ux.data[y * w + x] = u as f32;
            uy.data[y * w + x] = v as f32;
        }
    }
    (ux, uy)
}

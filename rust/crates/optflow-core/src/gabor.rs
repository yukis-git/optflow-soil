//! Multi-band complex Gabor filter bank.
//!
//! Port of `gabor_bank`, `_pad`, `_crop` and `reference_phase` of `python/optflow_ref/phaseflow.py`;
//! that file is the specification. Filters and responses are stored as `f32` (`Complex32`); the
//! filter values and the per-pixel phase gradient are computed in `f64`.

use std::f64::consts::{PI, TAU};

use num_complex::Complex32;
use rayon::prelude::*;

use crate::config::Config;
use crate::fft::Fft2;
use crate::image::Image;
use crate::interp::mirror;

/// Reflect padding added around an image before the filter bank is applied.
pub const PAD: usize = 32;

/// Frequency-domain filters for a padded `h` x `w` image (`h`, `w` include the `2 * PAD`).
#[derive(Clone, Debug)]
pub struct Bank {
    pub h: usize,
    pub w: usize,
    /// One real `h * w` array per filter in FFT layout (row-major, frequency 0 first); scales
    /// outer, orientations inner.
    pub filters: Vec<Vec<f32>>,
    /// Angular frequency `2 pi * fftfreq(w)` of each column [rad/px]; broadcast over the rows.
    pub fx: Vec<f32>,
    /// Angular frequency `2 pi * fftfreq(h)` of each row [rad/px]; broadcast over the columns.
    pub fy: Vec<f32>,
}

/// `2 pi * fftfreq(n)` in `f64`, evaluated like numpy: `k * (1 / n)` for `k = 0 .. (n-1)/2` and
/// then `-(n/2) .. -1`, times `2 pi`.
fn angular_freqs(n: usize) -> Vec<f64> {
    let val = 1.0 / n as f64;
    let positive = (n - 1) / 2 + 1;
    (0..n)
        .map(|i| {
            let k = if i < positive {
                i as f64
            } else {
                i as f64 - n as f64
            };
            k * val * 2.0 * PI
        })
        .collect()
}

/// One-sided Gaussian filters at the wavelengths `lam * scales` and `n_orient` orientations for a
/// padded `h` x `w` image, each multiplied by the DC-cut high-pass factor `1 - exp(-|f|^2 s^2 / 2)`
/// with `s = dc_cut * lam * max(scales) / (2 pi)`. Needs `h, w >= 1` and a non-empty `cfg.scales`.
pub fn gabor_bank(h: usize, w: usize, cfg: &Config) -> Bank {
    assert!(h >= 1 && w >= 1, "gabor_bank: empty size {h}x{w}");
    assert!(!cfg.scales.is_empty(), "gabor_bank: scales is empty");
    let (fx, fy) = (angular_freqs(w), angular_freqs(h));
    let max_scale = cfg.scales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let s_hp = cfg.dc_cut * cfg.lam * max_scale / TAU;
    let mut hp = vec![0.0f64; h * w];
    hp.par_chunks_mut(w).zip(&fy).for_each(|(row, &fyv)| {
        for (o, &fxv) in row.iter_mut().zip(&fx) {
            *o = 1.0 - (-(fxv * fxv + fyv * fyv) * s_hp * s_hp / 2.0).exp();
        }
    });
    let mut filters = Vec::with_capacity(cfg.scales.len() * cfg.n_orient);
    for &sc in &cfg.scales {
        let w0 = TAU / (cfg.lam * sc);
        let s = cfg.bandwidth * w0;
        for k in 0..cfg.n_orient {
            let th = PI * k as f64 / cfg.n_orient as f64;
            let (cx, cy) = (w0 * th.cos(), w0 * th.sin());
            // the Gaussian is separable: exp(-(dx^2 + dy^2) / 2 s^2) = gx[x] * gy[y]
            let gx: Vec<f64> = fx
                .iter()
                .map(|f| (-((f - cx) * (f - cx)) / (2.0 * s * s)).exp())
                .collect();
            let gy: Vec<f64> = fy
                .iter()
                .map(|f| (-((f - cy) * (f - cy)) / (2.0 * s * s)).exp())
                .collect();
            let mut filter = vec![0.0f32; h * w];
            filter
                .par_chunks_mut(w)
                .zip(hp.par_chunks(w))
                .zip(&gy)
                .for_each(|((row, hp_row), &gyv)| {
                    for ((o, &hpv), &gxv) in row.iter_mut().zip(hp_row).zip(&gx) {
                        *o = (hpv * (gxv * gyv)) as f32;
                    }
                });
            filters.push(filter);
        }
    }
    Bank {
        h,
        w,
        filters,
        fx: fx.iter().map(|&f| f as f32).collect(),
        fy: fy.iter().map(|&f| f as f32).collect(),
    }
}

/// For each of the `n + 2 pad` positions of an axis of length `n` padded by `pad` on both sides,
/// the index of the source pixel: whole-sample reflection (`mirror`, period `2 (n - 1)`), the
/// mapping of `np.pad(mode="reflect")`. The one mapping behind [`pad_reflect`] and
/// [`padded_spectrum`].
fn reflect_indices(n: usize, pad: usize) -> Vec<usize> {
    (0..n + 2 * pad)
        .map(|i| mirror(i as i64 - pad as i64, n as i64) as usize)
        .collect()
}

/// Pads `img` by `pad` pixels on every side like `np.pad(img, pad, mode="reflect")`: the edge
/// pixel is not repeated, and a pad wider than the image keeps reflecting (period `2 (n - 1)`).
pub fn pad_reflect(img: &Image<f32>, pad: usize) -> Image<f32> {
    let (w, h) = (img.w + 2 * pad, img.h + 2 * pad);
    let (rows, cols) = (reflect_indices(img.h, pad), reflect_indices(img.w, pad));
    let mut data = Vec::with_capacity(w * h);
    for &sy in &rows {
        let row = &img.data[sy * img.w..][..img.w];
        data.extend(cols.iter().map(|&x| row[x]));
    }
    Image { w, h, data }
}

/// The spectrum of `img` reflect-padded by `PAD` (`pad_reflect`, as complex values with zero
/// imaginary part, transformed by `fft`), the padding built row by row in parallel.
pub(crate) fn padded_spectrum(img: &Image<f32>, fft: &Fft2) -> Vec<Complex32> {
    let w = img.w + 2 * PAD;
    let (rows, cols) = (reflect_indices(img.h, PAD), reflect_indices(img.w, PAD));
    let mut spectrum = vec![Complex32::default(); w * (img.h + 2 * PAD)];
    spectrum.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &img.data[rows[y] * img.w..][..img.w];
        for (o, &x) in out.iter_mut().zip(&cols) {
            *o = Complex32::new(row[x], 0.0);
        }
    });
    fft.forward(&mut spectrum);
    spectrum
}

/// Response of one filter and its phase gradient at every pixel of the (unpadded) image.
#[derive(Clone, Debug)]
pub struct RefPhase {
    /// Complex response `r`.
    pub r: Image<Complex32>,
    /// `Im(conj(r) dr/dx) / (|r|^2 + 1e-12)` [rad/px].
    pub gx: Image<f32>,
    /// `Im(conj(r) dr/dy) / (|r|^2 + 1e-12)` [rad/px].
    pub gy: Image<f32>,
    /// `sqrt(|r|^2 + 1e-12)`.
    pub amp: Image<f32>,
}

/// Per filter: the response of `img` (reflect-padded by `PAD`), its phase gradient and its
/// amplitude. `bank` and `fft` must be for `(img.h + 2 PAD) x (img.w + 2 PAD)`.
pub fn reference_phase(img: &Image<f32>, bank: &Bank, fft: &Fft2) -> Vec<RefPhase> {
    let spectrum = reference_spectrum(img, bank, fft);
    bank.filters
        .par_iter()
        .map(|g| reference_filter(&spectrum, g, bank, fft, (img.h, img.w)))
        .collect()
}

/// The spectrum of `img` reflect-padded by `PAD`, the input of [`reference_filter`]. Panics
/// unless `bank` and `fft` are for `(img.h + 2 PAD) x (img.w + 2 PAD)`.
pub(crate) fn reference_spectrum(img: &Image<f32>, bank: &Bank, fft: &Fft2) -> Vec<Complex32> {
    let (ph, pw) = (img.h + 2 * PAD, img.w + 2 * PAD);
    assert!(
        bank.h == ph && bank.w == pw && fft.height() == ph && fft.width() == pw,
        "reference_phase: image {}x{} needs a {ph}x{pw} bank and FFT (bank {}x{}, FFT {}x{})",
        img.h,
        img.w,
        bank.h,
        bank.w,
        fft.height(),
        fft.width()
    );
    padded_spectrum(img, fft)
}

/// One entry of [`reference_phase`]: the response to the filter `g` of `bank` (one of
/// `bank.filters`) from the padded spectrum of [`reference_spectrum`], for an image of
/// `size = (h, w)`.
pub(crate) fn reference_filter(
    spectrum: &[Complex32],
    g: &[f32],
    bank: &Bank,
    fft: &Fft2,
    size: (usize, usize),
) -> RefPhase {
    let (h, w) = size;
    let [r_pad, rx_pad, ry_pad] = reference_inverses(spectrum, g, bank, fft);
    let crop = |p: &[Complex32]| {
        let stride = w + 2 * PAD;
        let mut data = vec![Complex32::default(); h * w];
        data.par_chunks_mut(w.max(1))
            .enumerate()
            .for_each(|(y, row)| {
                row.copy_from_slice(&p[(y + PAD) * stride + PAD..][..w]);
            });
        Image { w, h, data }
    };
    let (r, rx, ry) = (crop(&r_pad), crop(&rx_pad), crop(&ry_pad));
    drop((r_pad, rx_pad, ry_pad));
    let n = h * w;
    let (mut gx, mut gy, mut amp) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    gx.par_chunks_mut(w.max(1))
        .zip(gy.par_chunks_mut(w.max(1)))
        .zip(amp.par_chunks_mut(w.max(1)))
        .enumerate()
        .for_each(|(y, ((gx, gy), amp))| {
            for x in 0..w {
                let i = y * w + x;
                [gx[x], gy[x], amp[x]] = reference_values(r.data[i], rx.data[i], ry.data[i]);
            }
        });
    RefPhase {
        r,
        gx: Image::from_vec(w, h, gx),
        gy: Image::from_vec(w, h, gy),
        amp: Image::from_vec(w, h, amp),
    }
}

/// The inverse transforms behind one entry of [`reference_phase`], on the padded grid (not
/// cropped): the response `r` to the filter `g` and its derivatives along x and y (the
/// spectrum times `i f`), as `[r, rx, ry]`.
pub(crate) fn reference_inverses(
    spectrum: &[Complex32],
    g: &[f32],
    bank: &Bank,
    fft: &Fft2,
) -> [Vec<Complex32>; 3] {
    let pw = bank.w;
    let mut p: Vec<Complex32> = spectrum.par_iter().zip(g).map(|(&s, &g)| s * g).collect();
    // derivative along an axis: multiply the spectrum by i * f
    let deriv = |axis_x: bool| -> Vec<Complex32> {
        p.par_iter()
            .enumerate()
            .map(|(i, &c)| {
                let f = if axis_x {
                    bank.fx[i % pw]
                } else {
                    bank.fy[i / pw]
                };
                Complex32::new(-c.im * f, c.re * f)
            })
            .collect()
    };
    let mut rx = deriv(true);
    fft.inverse(&mut rx);
    let mut ry = deriv(false);
    fft.inverse(&mut ry);
    fft.inverse(&mut p);
    [p, rx, ry]
}

/// `[gx, gy, amp]` of [`RefPhase`] at one pixel from the response `r` and its derivatives:
/// `Im(conj(r) d) / (|r|^2 + 1e-12)` for `d = rx, ry` and `sqrt(|r|^2 + 1e-12)`, in `f64`.
#[inline]
pub(crate) fn reference_values(r: Complex32, rx: Complex32, ry: Complex32) -> [f32; 3] {
    let (re, im) = (f64::from(r.re), f64::from(r.im));
    let a2 = re * re + im * im + 1e-12;
    // Im(conj(r) * d) = r.re * d.im - r.im * d.re
    let grad = |d: Complex32| (re * f64::from(d.im) - im * f64::from(d.re)) / a2;
    [grad(rx) as f32, grad(ry) as f32, a2.sqrt() as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize) -> Image<f32> {
        let data = (0..w * h)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0 + (i as f32 * 0.37).sin())
            .collect();
        Image::from_vec(w, h, data)
    }

    /// `padded_spectrum` is the forward transform of `pad_reflect(img, PAD)` as a complex image,
    /// bit for bit, for images wider and narrower than the padding (a pad wider than the image
    /// keeps reflecting) and with sizes that are not multiples of the 8-line blocks.
    #[test]
    fn gabor_padded_spectrum_equals_the_transform_of_pad_reflect() {
        // fewer sizes under Miri, where the transforms are slow
        let sizes: &[(usize, usize)] = if cfg!(miri) {
            &[(5, 3)]
        } else {
            &[(18, 23), (5, 3), (1, 4), (40, 1)]
        };
        for &(w, h) in sizes {
            let img = textured(w, h);
            let padded = pad_reflect(&img, PAD);
            assert_eq!((padded.w, padded.h), (w + 2 * PAD, h + 2 * PAD));
            let fft = Fft2::new(padded.h, padded.w);
            let mut want: Vec<Complex32> = padded
                .data
                .iter()
                .map(|&v| Complex32::new(v, 0.0))
                .collect();
            fft.forward(&mut want);
            let got = padded_spectrum(&img, &fft);
            let bits = |v: &[Complex32]| -> Vec<(u32, u32)> {
                v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
            };
            assert_eq!(bits(&got), bits(&want), "{w}x{h}");
        }
    }

    #[test]
    fn gabor_pad_reflect_follows_numpy_reflect() {
        // np.pad(np.array([[1, 2, 3], [4, 5, 6]], np.float32), 2, mode="reflect")
        let img = Image::from_vec(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = pad_reflect(&img, 2);
        let rows: Vec<&[f32]> = p.data.chunks(p.w).collect();
        let (even, odd) = (
            [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
        );
        for (y, row) in rows.iter().enumerate() {
            assert_eq!(*row, if y % 2 == 0 { even } else { odd }, "row {y}");
        }
    }
}

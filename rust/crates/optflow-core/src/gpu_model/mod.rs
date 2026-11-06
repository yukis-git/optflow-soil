//! Pure-Rust models of the arithmetic the Metal kernels use (f32 pairs, an f32 `atan2`), so that
//! the kernels can be written as line-by-line ports and tested on the CPU first.
//! Not part of the API.

pub mod atan2f;
pub mod df64;
pub mod field;
pub mod solve;

use num_complex::Complex32;

use crate::config::Config;
use crate::engine::{EngineError, FineEngine};
use crate::fft::Fft2;
use crate::gabor::{gabor_bank, PAD};
use crate::gpu_support::{low_confidence, untrusted_fill};
use crate::image::Image;
use crate::interp::BSpline;
use crate::phaseflow::{RawFlow, RefPack, Responses};
use df64::Df64;
use field::{blur_centred, spline_coeffs_centred, warp_exact};
use solve::{local_solve_blocks, pixel_sums, ModelField, ModelSolveOut};

/// Side of the square output blocks of the block-local affine anchors (GPU spec §4.3).
pub const BLOCK: usize = 16;

/// The incremental-f32 numerics of the GPU backend run on the CPU ([`refine_raw_model`]).
pub struct ModelEngine;

impl FineEngine for ModelEngine {
    fn refine_raw(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<RawFlow, EngineError> {
        Ok(refine_raw_model(i1, i2, shift, ux, uy, cfg))
    }
}

/// [`ModelEngine`] whose stored responses and reference phase are each scaled by `1 + e`, `e`
/// a deterministic pseudo-random value in `[-2^-23, 2^-23]` (one f32 ulp) drawn from the seed:
/// the gate's stand-in for a GPU FFT that rounds differently from rustfft in the low bits.
#[doc(hidden)]
pub struct PerturbedModelEngine(pub u64);

impl FineEngine for PerturbedModelEngine {
    fn refine_raw(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<RawFlow, EngineError> {
        Ok(refine_raw_model_with(
            i1,
            i2,
            shift,
            ux,
            uy,
            cfg,
            Some(self.0),
        ))
    }
}

/// `v * (1 + e)` with `e` in `[-2^-23, 2^-23]` from a splitmix64 hash of `(seed, stream, index)`.
fn jitter(v: f32, seed: u64, stream: u64, index: usize) -> f32 {
    let mut z = seed
        .wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0; // [-1, 1)
    v * (1.0 + unit * f32::EPSILON)
}

/// [`crate::phaseflow::refine_raw`] with the arithmetic of the GPU backend (GPU spec §4): the
/// displacement and the coordinates are Df64, the sums, moments and solves f32 in increment form
/// against block-local affine anchors. The FFTs, the responses and the reference phase are the
/// CPU's (f32 storage); the pre-warp of the later outer iterations is the CPU's f64 rule.
pub fn refine_raw_model(
    i1: &Image<f32>,
    i2: &Image<f32>,
    shift: (i64, i64),
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
) -> RawFlow {
    refine_raw_model_with(i1, i2, shift, ux, uy, cfg, None)
}

/// [`refine_raw_model`], with the stored responses and reference phase jittered by one f32 ulp
/// (see [`PerturbedModelEngine`]) when `perturb` is a seed.
pub fn refine_raw_model_with(
    i1: &Image<f32>,
    i2: &Image<f32>,
    shift: (i64, i64),
    ux: &Image<f32>,
    uy: &Image<f32>,
    cfg: &Config,
    perturb: Option<u64>,
) -> RawFlow {
    let (w, h) = (i1.w, i1.h);
    assert!(w >= 1 && h >= 1, "refine: empty image {w}x{h}");
    assert!(i2.w >= 1 && i2.h >= 1, "refine: empty image i2");
    for (name, im) in [("ux", ux), ("uy", uy)] {
        assert!(
            (im.w, im.h) == (w, h),
            "refine: i1 is {w}x{h} but {name} is {}x{}",
            im.w,
            im.h
        );
    }
    let (ph, pw) = (h + 2 * PAD, w + 2 * PAD);
    let bank = gabor_bank(ph, pw, cfg);
    let fft = Fft2::new(ph, pw);
    let mut reference = RefPack::new(i1, &bank, &fft);
    if let Some(seed) = perturb {
        for (i, v) in reference.data.iter_mut().enumerate() {
            *v = jitter(*v, seed, 0, i);
        }
    }
    let spline2 = BSpline::new(i2);

    let to_df = |im: &Image<f32>| -> Vec<Df64> {
        im.data
            .iter()
            .map(|&v| Df64::from_f64(f64::from(v)))
            .collect()
    };
    let (mut ux, mut uy) = (to_df(ux), to_df(uy));
    let mut next = (ux.clone(), uy.clone());
    let mut resid = vec![std::f32::consts::PI; w * h];
    let mut energy = vec![1.0f32; w * h];
    let mut slope = vec![[0.0f32; 4]; w * h];
    let mut spent: Vec<Complex32> = Vec::new();
    for it in 0..cfg.outer_iters {
        let (sx, sy) = if it == 0 {
            (
                blur_centred(&ux, w, h, cfg.sigma_s),
                blur_centred(&uy, w, h, cfg.sigma_s),
            )
        } else {
            // the CPU's rule: the mask and the fill of the untrusted pixels in f64 (the GPU
            // backend runs them on the CPU too), the filled values as Df64, then the blur of the
            // first iteration (`prewarp` with `blur_centred` for its `gauss_blur`)
            let plane =
                |v: &[f32]| Image::from_vec(w, h, v.iter().map(|&e| f64::from(e)).collect());
            let low = low_confidence(&plane(&energy), &plane(&resid), cfg);
            let at = |n: usize| (ux[n].to_f64(), uy[n].to_f64(), slope[n].map(f64::from));
            match untrusted_fill(w, h, &low, cfg, at) {
                None => (
                    blur_centred(&ux, w, h, cfg.sigma_s),
                    blur_centred(&uy, w, h, cfg.sigma_s),
                ),
                Some(fill) => {
                    let (mut fx, mut fy) = (ux.clone(), uy.clone());
                    for &(p, vx, vy) in &fill {
                        fx[p as usize] = Df64::from_f64(vx);
                        fy[p as usize] = Df64::from_f64(vy);
                    }
                    (
                        blur_centred(&fx, w, h, cfg.sigma_s),
                        blur_centred(&fy, w, h, cfg.sigma_s),
                    )
                }
            }
        };
        let warped = warp_exact(&spline2.coeffs, &sx, &sy, shift);
        let mut responses = Responses::new_in(&warped, &bank, &fft, std::mem::take(&mut spent));
        if let Some(seed) = perturb {
            let stream = 1 + it as u64;
            for (i, c) in responses.data.iter_mut().enumerate() {
                c.re = jitter(c.re, seed, stream, 2 * i);
                c.im = jitter(c.im, seed, stream, 2 * i + 1);
            }
        }
        drop(warped);
        let (csx, csy) = (spline_coeffs_centred(&sx), spline_coeffs_centred(&sy));
        let field = ModelField {
            sx: &sx,
            sy: &sy,
            csx: &csx,
            csy: &csy,
        };
        for _ in 0..cfg.inner_iters {
            let sums = pixel_sums(&reference, &responses, &field, &ux, &uy);
            let out = ModelSolveOut {
                ux: &mut next.0,
                uy: &mut next.1,
                energy: &mut energy,
                resid: &mut resid,
                slope: &mut slope,
            };
            local_solve_blocks(&sums, (&ux, &uy), w, h, cfg.sigma_w, cfg.shape_order, out);
            std::mem::swap(&mut ux, &mut next.0);
            std::mem::swap(&mut uy, &mut next.1);
        }
        spent = responses.data;
    }
    let plane = |v: &[f32]| Image::from_vec(w, h, v.iter().map(|&e| f64::from(e)).collect());
    let to64 = |u: &[Df64]| Image::from_vec(w, h, u.iter().map(|d| d.to_f64()).collect());
    RawFlow {
        ux: to64(&ux),
        uy: to64(&uy),
        energy: plane(&energy),
        resid: plane(&resid),
    }
}

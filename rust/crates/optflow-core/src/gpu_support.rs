//! Items of the CPU pipeline that the GPU backend (`optflow-gpu`) reuses as they are, so that
//! both sides share one definition. Not part of the API; nothing here computes anything new.

use crate::config::Config;
use crate::image::Image;

pub use crate::phaseflow::Aperture;
pub use crate::prewarp::LowFreeze;
pub use crate::tile::tile_margin;
pub use crate::track::TILE;

/// `prewarp::low_confidence`: the pixels with confidence below `conf_min` of a solve.
pub fn low_confidence(energy: &Image<f64>, resid: &Image<f64>, cfg: &Config) -> Image<bool> {
    crate::prewarp::low_confidence(energy, resid, cfg)
}

/// `prewarp::untrusted_fill` at the radius of `prewarp` (`cfg.auto_edge_guard_px()`): the
/// untrusted pixels and their first-order extrapolation `(p, ux, uy)`, `at(n)` giving
/// `(ux, uy, slope)` of pixel `n`; `None` when `prewarp` is the plain blur.
pub fn untrusted_fill(
    w: usize,
    h: usize,
    low: &Image<bool>,
    cfg: &Config,
    at: impl Fn(usize) -> (f64, f64, [f64; 4]) + Sync,
) -> Option<Vec<(u32, f64, f64)>> {
    crate::prewarp::untrusted_fill(w, h, low, cfg.auto_edge_guard_px(), at)
}

/// `select::median`.
pub fn median(v: &[f64]) -> f64 {
    crate::select::median(v)
}

/// `phaseflow::texture_scale`.
pub fn texture_scale(energy: &[f64]) -> f64 {
    crate::phaseflow::texture_scale(energy)
}

/// `phaseflow::border_zone`.
pub fn border_zone(cfg: &Config) -> f64 {
    crate::phaseflow::border_zone(cfg)
}

/// `phaseflow::moment_kernel` of the Gaussian of `sigma` for the powers `p = 0 ..= max_power`.
pub fn moment_taps(sigma: f64, max_power: u32) -> Vec<Vec<f64>> {
    crate::phaseflow::moment_taps(sigma, max_power)
}

/// `phaseflow::plan`: `(n, moments, m_idx, b_idx, gcc_idx, gw_idx)`. A moment is `(source,
/// powers)`, the source numbered 0..7 in the order `Gxx, Gxy, Gyy, Gxc, Gyc, Gcc, Gw`.
#[allow(clippy::type_complexity)]
pub fn plan_moments(
    order: u32,
) -> (
    usize,
    Vec<(u8, (u32, u32))>,
    Vec<usize>,
    Vec<usize>,
    usize,
    usize,
) {
    crate::phaseflow::plan_moments(order)
}

/// `RefPack::new(img, bank, fft).data`: per pixel and filter `[r.re, r.im, gx, gy, amp]`.
#[doc(hidden)]
pub fn ref_pack(img: &Image<f32>, bank: &crate::gabor::Bank, fft: &crate::fft::Fft2) -> Vec<f32> {
    crate::phaseflow::RefPack::new(img, bank, fft).data
}

/// `Responses::new_in(img, bank, fft, vec![]).data`: node layout `[(y * pw + x) * f + k]` on the
/// padded grid.
#[doc(hidden)]
pub fn responses(
    img: &Image<f32>,
    bank: &crate::gabor::Bank,
    fft: &crate::fft::Fft2,
) -> Vec<num_complex::Complex32> {
    crate::phaseflow::Responses::new_in(img, bank, fft, Vec::new()).data
}

/// `gpu_model::solve::pixel_sums` on raw data: `reference` and `responses` as in [`ref_pack`] and
/// [`responses`] for `n_filters` filters, the smoothed field `s` with its centred coefficients,
/// and the displacement `(ux, uy)` as `(hi, lo)` pairs.
#[doc(hidden)]
pub fn model_pixel_sums(
    reference: Vec<f32>,
    responses: Vec<num_complex::Complex32>,
    n_filters: usize,
    field: [&Image<f32>; 4],
    ux: &[(f32, f32)],
    uy: &[(f32, f32)],
) -> Vec<[f32; 7]> {
    use crate::gpu_model::df64::Df64;
    use crate::gpu_model::solve::{pixel_sums, ModelField};
    let rp = crate::phaseflow::RefPack {
        n_filters,
        data: reference,
    };
    let rs = crate::phaseflow::Responses {
        w: field[0].w + 2 * crate::gabor::PAD,
        n_filters,
        data: responses,
    };
    let f = ModelField {
        sx: field[0],
        sy: field[1],
        csx: field[2],
        csy: field[3],
    };
    let df =
        |v: &[(f32, f32)]| -> Vec<Df64> { v.iter().map(|&(hi, lo)| Df64 { hi, lo }).collect() };
    pixel_sums(&rp, &rs, &f, &df(ux), &df(uy))
}

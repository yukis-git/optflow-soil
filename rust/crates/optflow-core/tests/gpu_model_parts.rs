use optflow_core::gpu_model::atan2f::atan2f;
use optflow_core::gpu_model::df64::{two_prod, two_sum, Df64};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

#[test]
fn two_sum_and_two_prod_are_exact() {
    let mut s = 1;
    for _ in 0..100_000 {
        let a = ((lcg(&mut s) - 0.5) * 1e3) as f32;
        let b = ((lcg(&mut s) - 0.5) * 1e-2) as f32;
        let (h, l) = two_sum(a, b);
        assert_eq!(f64::from(h) + f64::from(l), f64::from(a) + f64::from(b));
        let (h, l) = two_prod(a, b);
        assert_eq!(f64::from(h) + f64::from(l), f64::from(a) * f64::from(b));
    }
}

#[test]
fn df64_arithmetic_is_within_1e_minus_13_relative() {
    let mut s = 2;
    for _ in 0..100_000 {
        let (x, y) = ((lcg(&mut s) - 0.5) * 1000.0, (lcg(&mut s) - 0.5) * 1000.0);
        let v = (lcg(&mut s) - 0.5) as f32;
        let (dx, dy) = (Df64::from_f64(x), Df64::from_f64(y));
        // the reference is the exact value of the (rounded) inputs: from_f64 itself keeps only
        // about 48 bits, which is worse than 1e-13 of a small difference of two values near 500
        let (x, y) = (dx.to_f64(), dy.to_f64());
        let tol = |r: f64| 1e-13 * r.abs().max(1.0);
        assert!((dx.add(dy).to_f64() - (x + y)).abs() <= tol(x + y));
        assert!((dx.sub(dy).to_f64() - (x - y)).abs() <= tol(x - y));
        assert!((dx.mul_f32(v).to_f64() - x * f64::from(v)).abs() <= tol(x * f64::from(v)));
        let (i, f) = dx.floor_frac();
        assert_eq!(f64::from(i), x.floor());
        assert!((f64::from(f) - (x - x.floor())).abs() <= 1e-7, "{x}");
        assert!((0.0..1.0).contains(&f));
    }
}

#[test]
fn atan2f_is_within_2_ulp_and_keeps_the_special_cases() {
    let ulp = |v: f32| (f32::from_bits(v.abs().to_bits() + 1) - v.abs()) as f64;
    let mut s = 3;
    let mut worst = 0.0f64;
    for _ in 0..1_000_000 {
        let y = ((lcg(&mut s) - 0.5) * 10f64.powf(lcg(&mut s) * 8.0 - 4.0)) as f32;
        let x = ((lcg(&mut s) - 0.5) * 10f64.powf(lcg(&mut s) * 8.0 - 4.0)) as f32;
        let want = f64::from(y).atan2(f64::from(x));
        let got = f64::from(atan2f(y, x));
        let e = (got - want).abs() / ulp(want as f32);
        worst = worst.max(e);
        assert!(e <= 2.0, "atan2f({y}, {x}) = {got}, want {want}");
    }
    eprintln!("atan2f worst error: {worst} ulp");
    for (y, x) in [
        (0.0f32, 0.0f32),
        (-0.0, 0.0),
        (0.0, -0.0),
        (-0.0, -0.0),
        (1.0, 0.0),
        (f32::INFINITY, f32::INFINITY),
        (-1.0, f32::NEG_INFINITY),
        (f32::NAN, 1.0),
    ] {
        // f64 rounded to f32: the platform's f32 libm returns pi - 1 ulp for (0, -0) on macOS
        let want = f64::from(y).atan2(f64::from(x)) as f32;
        let got = atan2f(y, x);
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "{y} {x}"
        );
    }
}

// ---- Task 3: the field parts and the ModelEngine ----

mod common;

use common::synth;
use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, FineEngine};
use optflow_core::gpu_model::field::{
    blur_centred, sample_centred, spline_coeffs_centred, warp_exact,
};
use optflow_core::gpu_model::ModelEngine;
use optflow_core::image::Image;
use optflow_core::interp::{gauss_blur, taps, warp, BSpline};

#[test]
fn df64_difference_of_nearby_values_keeps_the_difference_to_1e_minus_7_relative() {
    let mut s = 4;
    for _ in 0..100_000 {
        let x = 300.0 + lcg(&mut s);
        let d = 1e-4 * (1.0 + lcg(&mut s));
        let (a, b) = (Df64::from_f64(x + d), Df64::from_f64(x));
        // the exact difference of the two df64 values
        let want = a.to_f64() - b.to_f64();
        let got = f64::from(a.diff_f32(b));
        assert!(
            (got - want).abs() <= 1e-7 * want.abs(),
            "{x} {d}: {got} vs {want}"
        );
    }
}

/// A large smooth field: 300 px plus 1% strain, the case where plain f32 loses ~1e-5 px.
fn big_field(w: usize, h: usize) -> Vec<f64> {
    (0..w * h)
        .map(|i| {
            300.0 + 0.01 * (i % w) as f64 - 0.003 * (i / w) as f64
                + 0.2 * ((i % w) as f64 * 0.05).sin()
        })
        .collect()
}

#[test]
fn centred_blur_matches_the_f64_blur_to_1e_minus_6_px() {
    let (w, h) = (70, 50);
    let u = big_field(w, h);
    let want = gauss_blur(&Image::from_vec(w, h, u.clone()), 4.0);
    let du: Vec<Df64> = u.iter().map(|&v| Df64::from_f64(v)).collect();
    let got = blur_centred(&du, w, h, 4.0);
    for (g, w) in got.data.iter().zip(&want.data) {
        // the f32 result rounds the f64 one: within half an ulp of 300 plus 1e-6
        assert!((f64::from(*g) - w).abs() <= 1.6e-5 + 1e-6);
        assert_eq!(*g, *w as f32, "rounding differs at {w}"); // equal in all but near-ties
    }
}

#[test]
fn centred_spline_coefficients_equal_bspline_new() {
    let (w, h) = (70, 50);
    let s = Image::from_vec(w, h, big_field(w, h).iter().map(|&v| v as f32).collect());
    let want = BSpline::new(&s).coeffs;
    let got = spline_coeffs_centred(&s);
    let differ = got
        .data
        .iter()
        .zip(&want.data)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differ * 1000 <= w * h,
        "{differ} of {} coefficients differ",
        w * h
    );
    for (a, b) in got.data.iter().zip(&want.data) {
        assert!((a - b).abs() <= 2.0 * f32::EPSILON * b.abs());
    }
}

/// The spline of `coeffs` at `(x, y)` in f64 arithmetic of the f32 coefficients (the sampling
/// of `phaseflow::sample_taps2`).
fn sample_f64(coeffs: &Image<f32>, x: f64, y: f64) -> f64 {
    let t = taps(x, y, coeffs.w, coeffs.h).unwrap();
    let mut total = 0.0;
    for (&yj, &wyj) in t.yi.iter().zip(&t.wy) {
        let row = &coeffs.data[yj * coeffs.w..(yj + 1) * coeffs.w];
        let mut acc = t.wx[0] * f64::from(row[t.xi[0]]);
        for (&xk, &wxk) in t.xi.iter().zip(&t.wx).skip(1) {
            acc += wxk * f64::from(row[xk]);
        }
        total += wyj * acc;
    }
    total
}

#[test]
fn centred_sampling_matches_the_f64_spline_to_1e_minus_7_px() {
    let (w, h) = (70, 50);
    let s = Image::from_vec(w, h, big_field(w, h).iter().map(|&v| v as f32).collect());
    let sp = BSpline::new(&s);
    let mut seed = 5;
    for _ in 0..10_000 {
        let x = 1.0 + lcg(&mut seed) * (w as f64 - 3.0);
        let y = 1.0 + lcg(&mut seed) * (h as f64 - 3.0);
        let want = sample_f64(&sp.coeffs, x, y);
        let got = sample_centred(&sp.coeffs, Df64::from_f64(x), Df64::from_f64(y)).to_f64();
        assert!((got - want).abs() <= 1e-7, "{x} {y}: {got} vs {want}");
    }
    // outside the image the coordinate is clamped; a non-finite one gives NaN
    let at = |x: f64, y: f64| sample_centred(&sp.coeffs, Df64::from_f64(x), Df64::from_f64(y));
    assert!((at(-5.0, 7.5).to_f64() - sample_f64(&sp.coeffs, -5.0, 7.5)).abs() <= 1e-7);
    assert!((at(900.0, 7.5).to_f64() - sample_f64(&sp.coeffs, 900.0, 7.5)).abs() <= 1e-7);
    let n = at(f64::NAN, 3.0);
    assert!(n.hi.is_nan() && n.lo.is_nan());
}

#[test]
fn exact_warp_is_the_f32_rounding_of_the_f64_warp() {
    let (w, h) = (80, 60);
    let mut seed = 6;
    let img = Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|_| lcg(&mut seed) as f32)
            .collect::<Vec<_>>(),
    );
    // displacements of both signs, a few pixels past the border included
    let mut field = |scale: f64| -> Image<f32> {
        Image::from_vec(
            w,
            h,
            (0..w * h)
                .map(|_| ((lcg(&mut seed) - 0.5) * scale) as f32)
                .collect::<Vec<_>>(),
        )
    };
    let (mut sx, sy) = (field(120.0), field(120.0));
    sx.data[7] = f32::NAN;
    let want = warp(&img, &sx, &sy);
    let got = warp_exact(&BSpline::new(&img).coeffs, &sx, &sy, (0, 0));
    assert!(got.data[7].is_nan() && want.data[7].is_nan());
    let differ = got
        .data
        .iter()
        .zip(&want.data)
        .filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
        .count();
    // equal in all but the last bit of a few near-ties (the Df64 sums are exact to ~1e-14)
    assert!(
        differ * 1000 <= w * h,
        "{differ} of {} pixels differ",
        w * h
    );
    for (a, b) in got.data.iter().zip(&want.data) {
        assert!(a.is_nan() && b.is_nan() || (a - b).abs() <= 1.5 * f32::EPSILON * b.abs());
    }
    // a window of the second image: the same values at the shifted place
    let sub = |x0: usize, y0: usize| {
        let (ww, hh) = (50, 40);
        let data = (0..hh)
            .flat_map(|y| (0..ww).map(move |x| (x, y)))
            .map(|(x, y)| img.data[(y + y0) * w + x + x0])
            .collect();
        Image::from_vec(ww, hh, data)
    };
    let (win, zero) = (sub(10, 5), Image::<f32>::new(50, 40));
    let a = warp_exact(&BSpline::new(&win).coeffs, &zero, &zero, (0, 0));
    assert_eq!(a.data[3 * 50 + 4], win.data[3 * 50 + 4]);
}

#[test]
fn model_engine_refines_like_the_cpu_on_a_small_pair() {
    let n = 96;
    let c = n as f64 / 2.0;
    let f = synth::uniform_strain(0.01, -0.003, (c, c), (3.4, -2.1));
    let p = synth::make_pattern(n, n, 1, 32.0);
    let (a, b) = (
        synth::render(&p, n, n, None),
        synth::render(&p, n, n, Some(&f)),
    );
    let (ux, uy) = (
        Image::from_vec(n, n, vec![3.0f32; n * n]),
        Image::from_vec(n, n, vec![-2.0f32; n * n]),
    );
    let cfg = Config::default();
    let want = CpuEngine
        .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
        .unwrap();
    let got = ModelEngine
        .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
        .unwrap();
    let rms = |x: &[f64], y: &[f64]| {
        let d: Vec<f64> = x
            .iter()
            .zip(y)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(a, b)| (a - b) * (a - b))
            .collect();
        (d.iter().sum::<f64>() / d.len() as f64).sqrt()
    };
    let (ex, ey) = (
        rms(&got.ux.data, &want.ux.data),
        rms(&got.uy.data, &want.uy.data),
    );
    println!("model vs cpu, 96^2: ux {ex:.2e} uy {ey:.2e} px");
    assert!(ex <= 1e-5 && ey <= 1e-5);
}

#[test]
fn model_engine_refines_like_the_cpu_with_a_quadratic_shape_function() {
    let n = 96;
    let c = n as f64 / 2.0;
    let f = synth::uniform_strain(0.02, 0.01, (c, c), (-5.3, 4.2));
    let p = synth::make_pattern(n, n, 2, 32.0);
    let (a, b) = (
        synth::render(&p, n, n, None),
        synth::render(&p, n, n, Some(&f)),
    );
    let (ux, uy) = (
        Image::from_vec(n, n, vec![-5.0f32; n * n]),
        Image::from_vec(n, n, vec![4.0f32; n * n]),
    );
    let cfg = Config {
        shape_order: 2,
        ..Config::default()
    };
    let want = CpuEngine
        .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
        .unwrap();
    let got = ModelEngine
        .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
        .unwrap();
    let rms = |x: &[f64], y: &[f64]| {
        let d: Vec<f64> = x
            .iter()
            .zip(y)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(a, b)| (a - b) * (a - b))
            .collect();
        (d.iter().sum::<f64>() / d.len() as f64).sqrt()
    };
    let (ex, ey) = (
        rms(&got.ux.data, &want.ux.data),
        rms(&got.uy.data, &want.uy.data),
    );
    println!("model vs cpu, 96^2, order 2: ux {ex:.2e} uy {ey:.2e} px");
    assert!(ex <= 1e-5 && ey <= 1e-5);
}

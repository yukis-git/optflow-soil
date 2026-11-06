//! Parity of the B-spline sampler, `warp` and `gauss_blur` with the Python fixtures
//! (`python/optflow_ref/interp.py`). Run with `-- --nocapture` to see the measured errors.
mod common;
use optflow_core::image::Image;
use optflow_core::interp::*;

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

#[test]
fn interp_sample_matches_python() {
    let img = img_f32("interp_img");
    let (_, xs) = common::load_f64("interp_x");
    let (_, ys) = common::load_f64("interp_y");
    let (_, want) = common::load_f64("interp_sample");
    let s = BSpline::new(&img);
    let mut max = 0.0f64;
    for i in 0..xs.len() {
        let got = s.sample(xs[i], ys[i]) as f64;
        assert!(
            (got - want[i]).abs() < 1e-5,
            "i={i} got {got} want {}",
            want[i]
        );
        max = max.max((got - want[i]).abs());
    }
    println!("sample: n={} max |diff| = {max:.3e}", xs.len());
}

#[test]
fn interp_warp_matches_python() {
    let img = img_f32("interp_img");
    let (ux, uy) = (img_f32("interp_warp_ux"), img_f32("interp_warp_uy"));
    let (_, want) = common::load_f64("interp_warp");
    let got = warp(&img, &ux, &uy);
    let rms = common::rms_diff(&got.data, &want, None);
    let max = got
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max);
    println!("warp: rms = {rms:.3e}, max |diff| = {max:.3e}");
    assert!(rms < 1e-5);
}

#[test]
fn interp_blur_matches_python() {
    let (shape, v) = common::load_f64("interp_img");
    let img = Image::from_vec(shape[1], shape[0], v);
    for s in [1.0, 2.0, 4.0] {
        let (_, want) = common::load_f64(&format!("interp_blur_{s}"));
        let got = gauss_blur(&img, s);
        let max = got
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("blur sigma {s}: max |diff| = {max:.3e}");
        // fixtures are float32 (Task 2 ruling): input and expected values carry ~6e-8 relative rounding
        assert!(max < 1e-6, "sigma {s}: {max}");
    }
}

#[test]
fn interp_non_finite_coordinates_give_nan() {
    let s = BSpline::new(&Image::from_vec(3, 3, vec![1.0f32; 9]));
    assert!(s.sample(f64::NAN, 1.0).is_nan() && s.sample(1.0, f64::INFINITY).is_nan());
}

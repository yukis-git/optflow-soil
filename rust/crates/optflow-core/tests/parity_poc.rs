//! Parity of the POC coarse stage with the Python reference (`python/optflow_ref/poc.py`). Run
//! with `-- --nocapture` to see the measured differences.
mod common;
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::poc::*;

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    assert_eq!(shape.len(), 2, "{name}: expected a 2-D array");
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

fn crop(img: &Image<f32>, y0: usize, y1: usize, x0: usize, x1: usize) -> Image<f32> {
    let mut data = Vec::new();
    for y in y0..y1 {
        data.extend_from_slice(&img.data[y * img.w + x0..y * img.w + x1]);
    }
    Image::from_vec(x1 - x0, y1 - y0, data)
}

fn usize_list(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect()
}

fn max_abs_diff(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max)
}

/// Compares a grid with the fixtures `<prefix>_ux`, `<prefix>_uy` (and `_peak` when given).
fn check_grid(label: &str, g: &Grid, prefix: &str, peak: bool) {
    for (name, got) in [("ux", &g.ux), ("uy", &g.uy)] {
        let (shape, want) = common::load_f64(&format!("{prefix}_{name}"));
        assert_eq!(
            (got.h, got.w),
            (shape[0], shape[1]),
            "{label}: {name} shape"
        );
        let rms = common::rms_diff(&got.data, &want, None);
        let max = max_abs_diff(&got.data, &want);
        println!("{label} {name}: rms = {rms:.3e}, max |diff| = {max:.3e}");
        assert!(rms < 1e-3, "{label} {name}: rms {rms}");
        assert!(max < 1e-3, "{label} {name}: max {max}");
    }
    if peak {
        let (_, want) = common::load_f64(&format!("{prefix}_peak"));
        let rms = common::rms_diff(&g.peak.data, &want, None);
        let max = max_abs_diff(&g.peak.data, &want);
        println!("{label} peak: rms = {rms:.3e}, max |diff| = {max:.3e}");
        assert!(rms < 1e-3, "{label} peak: rms {rms}");
    }
}

#[test]
fn poc_shift_matches_python() {
    let (a, b) = (img_f32("poc_a"), img_f32("poc_b"));
    let m = &common::meta()["poc_shift"];
    let w: Vec<usize> = usize_list(&m["window"]);
    let (wa, wb) = (
        crop(&a, w[0], w[1], w[2], w[3]),
        crop(&b, w[0], w[1], w[2], w[3]),
    );
    let (dx, dy, pk) = poc_shift(&wa, &wb);
    let (ex, ey, ep) = (
        m["dx"].as_f64().unwrap(),
        m["dy"].as_f64().unwrap(),
        m["peak"].as_f64().unwrap(),
    );
    println!(
        "poc_shift: |dx| diff = {:.3e}, |dy| diff = {:.3e}, |peak| diff = {:.3e}",
        (dx - ex).abs(),
        (dy - ey).abs(),
        (pk - ep).abs()
    );
    assert!((dx - ex).abs() < 1e-3, "dx {dx} vs {ex}");
    assert!((dy - ey).abs() < 1e-3, "dy {dy} vs {ey}");
    assert!((pk - ep).abs() < 1e-3, "peak {pk} vs {ep}");
}

#[test]
fn poc_grid_matches_python() {
    let (a, b) = (img_f32("poc_a"), img_f32("poc_b"));
    let g = coarse_flow(&a, &b, &Config::default(), None);
    let m = &common::meta()["poc_grid"];
    assert_eq!(g.xs, usize_list(&m["xs"]));
    assert_eq!(g.ys, usize_list(&m["ys"]));
    check_grid("poc_grid", &g, "poc_grid", true);
}

#[test]
fn poc_strip_matches_python() {
    let (a, b) = (img_f32("poc_strip_a"), img_f32("poc_strip_b"));
    assert_eq!((a.h, a.w), (96, 512));
    let m = &common::meta()["poc_strip"];
    let cfg = Config {
        max_disp: m["max_disp"].as_f64().unwrap(),
        ..Config::default()
    };
    let want_levels = m["n_levels"].as_u64().unwrap() as u32;
    assert_eq!(want_levels, 2);
    assert_eq!(n_levels(cfg.max_disp, cfg.poc_win, a.h, a.w), want_levels);
    let g = coarse_flow(&a, &b, &cfg, None);
    assert_eq!(g.xs, usize_list(&m["xs"]));
    assert_eq!(g.ys, usize_list(&m["ys"]));
    check_grid("poc_strip", &g, "poc_strip", false);
}

#[test]
fn level_window_matches_python_exactly() {
    let m = common::meta();
    let cases = m["level_window"].as_array().unwrap();
    assert!(cases.len() >= 9);
    for c in cases {
        let (h, w) = (
            c[0].as_u64().unwrap() as usize,
            c[1].as_u64().unwrap() as usize,
        );
        let want = c[2].as_array().map(|p| {
            (
                p[0].as_u64().unwrap() as usize,
                p[1].as_u64().unwrap() as usize,
            )
        });
        assert_eq!(level_window(h, w, 64), want, "level_window({h}, {w}, 64)");
    }
}

#[test]
fn n_levels_matches_python_exactly() {
    let m = common::meta();
    let cases = m["n_levels"].as_array().unwrap();
    assert!(cases.len() >= 16);
    for c in cases {
        let d = c[0].as_f64().unwrap();
        let (h, w) = (
            c[1].as_u64().unwrap() as usize,
            c[2].as_u64().unwrap() as usize,
        );
        let want = c[3].as_u64().unwrap() as u32;
        assert_eq!(n_levels(d, 64, h, w), want, "n_levels({d}, 64, {h}, {w})");
    }
}

#[test]
fn capture_range_matches_python_exactly() {
    let m = common::meta();
    let cases = m["capture_range"].as_array().unwrap();
    assert!(cases.len() >= 14);
    for c in cases {
        let (d, win) = (c[0].as_f64().unwrap(), c[1].as_u64().unwrap() as usize);
        let (h, w) = (
            c[2].as_u64().unwrap() as usize,
            c[3].as_u64().unwrap() as usize,
        );
        let want = (c[4].as_f64().unwrap(), c[5].as_f64().unwrap());
        // bit for bit: windows / 4 * 2^L are exact in binary floating point
        assert_eq!(
            capture_range(d, win, h, w),
            want,
            "capture_range({d}, {win}, {h}, {w})"
        );
    }
}

/// A constant dense guess image.
fn constant(w: usize, h: usize, v: f32) -> Image<f32> {
    Image::from_vec(w, h, vec![v; w * h])
}

#[test]
fn coarse_flow_with_a_sloped_guess_matches_python() {
    // the poc_a / poc_b pair (true shift (5.3, -2.7)) with the dense guess ux = 4 + 0.02 x,
    // uy = -2 + 0.01 y; the 128 x 128 image has two pyramid levels, the guess starts the coarse one.
    // This pair converges to the same grid with or without the guess (so it only shows that the
    // guess does no harm); `coarse_flow_with_a_sloped_guess_that_decides_the_result_matches_python`
    // is the discriminating case.
    let (a, b) = (img_f32("poc_a"), img_f32("poc_b"));
    let (gx, gy) = (img_f32("poc_init_guess_ux"), img_f32("poc_init_guess_uy"));
    let g = coarse_flow(&a, &b, &Config::default(), Some((&gx, &gy)));
    let m = &common::meta()["poc_init"];
    assert_eq!(g.xs, usize_list(&m["xs"]));
    assert_eq!(g.ys, usize_list(&m["ys"]));
    check_grid("poc_init", &g, "poc_init", true);
    let best = g.peak.data.iter().copied().fold(0.0f32, f32::max);
    assert!(
        best > 0.5,
        "best peak {best}: the search should have found the shift"
    );
}

/// The poc_strip pair (true shift (60.4, 1.1)) with `max_disp` 16: one level and a 64 px window, so
/// a shift of 60 px is out of reach without a guess. Python's result with the sloped guess
/// `ux = 45 + 0.05 x`, `uy = -3 + 0.12 y` (parameters in `meta.json`, not stored as arrays)
/// differs from its result without one by more than 10 px at every grid point (asserted by the
/// fixture generator, `make_parity_fixtures.py`, and below for the Rust result without a guess),
/// so matching it shows that the guess is sampled and used where it belongs.
#[test]
fn coarse_flow_with_a_sloped_guess_that_decides_the_result_matches_python() {
    let (a, b) = (img_f32("poc_strip_a"), img_f32("poc_strip_b"));
    let m = &common::meta()["poc_far"];
    let line = |key: &str| (m[key][0].as_f64().unwrap(), m[key][1].as_f64().unwrap());
    let ((x0, x1), (y0, y1)) = (line("guess_x"), line("guess_y"));
    let guess = |f: &dyn Fn(usize, usize) -> f64| {
        let data = (0..a.w * a.h).map(|i| f(i % a.w, i / a.w) as f32).collect();
        Image::from_vec(a.w, a.h, data)
    };
    let gx = guess(&|x, _| x0 + x1 * x as f64);
    let gy = guess(&|_, y| y0 + y1 * y as f64);
    let cfg = Config {
        max_disp: m["max_disp"].as_f64().unwrap(),
        ..Config::default()
    };
    assert_eq!(n_levels(cfg.max_disp, cfg.poc_win, a.h, a.w), 0);
    let g = coarse_flow(&a, &b, &cfg, Some((&gx, &gy)));
    assert_eq!(g.xs, usize_list(&m["xs"]));
    assert_eq!(g.ys, usize_list(&m["ys"]));
    check_grid("poc_far", &g, "poc_far", true);
    let found = g.peak.data.iter().filter(|&&p| p > 0.9).count();
    assert!(found >= 40, "only {found} points found the shift");
    // the guess is what decides it: without one every grid point is far from Python's result
    let none = coarse_flow(&a, &b, &cfg, None);
    let (_, want_ux) = common::load_f64("poc_far_ux");
    let (_, want_uy) = common::load_f64("poc_far_uy");
    let far = (0..want_ux.len())
        .filter(|&i| {
            (f64::from(none.ux.data[i]) - want_ux[i]).hypot(f64::from(none.uy.data[i]) - want_uy[i])
                > 10.0
        })
        .count();
    assert_eq!(far, want_ux.len(), "{far} of {} points", want_ux.len());
}

#[test]
fn coarse_flow_without_any_match_falls_back_to_the_guess_like_python() {
    // a blank second image has no correlation peak on any level: every level returns its guess,
    // so the result is the sloped guess resampled down and up the pyramid
    let a = img_f32("poc_a");
    let b = constant(a.w, a.h, 0.5);
    let (gx, gy) = (img_f32("poc_init_guess_ux"), img_f32("poc_init_guess_uy"));
    let g = coarse_flow(&a, &b, &Config::default(), Some((&gx, &gy)));
    let m = &common::meta()["poc_fallback"];
    assert_eq!(g.xs, usize_list(&m["xs"]));
    assert_eq!(g.ys, usize_list(&m["ys"]));
    check_grid("poc_fallback", &g, "poc_fallback", true);
    assert!(g.peak.data.iter().all(|&p| p == 0.0));
    // the guess ranges over 4.0..6.5 in x: this is not the zero field of a coarse_flow without one
    assert!(g.ux.data.iter().all(|&v| v > 4.0));
    let zero = coarse_flow(&a, &b, &Config::default(), None);
    assert!(zero.ux.data.iter().all(|&v| v == 0.0));
}

#[test]
fn coarse_flow_uses_the_initial_guess() {
    // Strip fixture: true shift (60.4, 1.1). With max_disp = 16 there is a single level (64 px
    // window, search range 16 px), so 60 px is out of reach without a guess; a guess within a few
    // pixels must carry the search. With max_disp = 100 the guess is scaled through the pyramid.
    let (a, b) = (img_f32("poc_strip_a"), img_f32("poc_strip_b"));
    let guess = (constant(a.w, a.h, 58.0), constant(a.w, a.h, 0.0));
    let near = |g: &Grid| {
        let ok =
            g.ux.data
                .iter()
                .zip(&g.uy.data)
                .filter(|(x, y)| (f64::from(**x) - 60.4).hypot(f64::from(**y) - 1.1) < 0.2)
                .count();
        ok as f64 / g.ux.data.len() as f64
    };
    for max_disp in [16.0, 100.0] {
        let cfg = Config {
            max_disp,
            ..Config::default()
        };
        let with = near(&coarse_flow(&a, &b, &cfg, Some((&guess.0, &guess.1))));
        let without = near(&coarse_flow(&a, &b, &cfg, None));
        println!(
            "initial guess, max_disp {max_disp}: {with:.3} correct with, {without:.3} without"
        );
        assert!(
            with > 0.9,
            "max_disp {max_disp}: only {with} correct with the guess"
        );
        if max_disp == 16.0 {
            assert!(without < 0.5, "the guess should be needed, got {without}");
        }
    }
}

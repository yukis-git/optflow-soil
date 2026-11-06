//! The tiled fine stage `tile::refine_tiled` against the untiled `phaseflow::refine` on synthetic
//! speckle images with a known motion, from the coarse stage's field as `track_pair` starts it.
//! Run with `-- --nocapture` to see the measured differences.
//!
//! The differences are measured over the pixels the pipeline reports (`common::bands::pipeline_valid`:
//! `conf >= conf_min`, farther than the edge guard from every large low-confidence component and
//! from the image border). Since the border rule (spec 2.4) the confidence is 0 within
//! `B = 4 sigma_w` of the border and wherever the match is within `B` of the second image's
//! border, so the layer of about 25 px along the border, where `refine` itself is ill-conditioned
//! (matched points and windows leave the image: any change of the computation that is not a change
//! of the result moves those pixels by 1e-2 to 1 px, tiled or not), is not reported.
//!
//! One more set is checked on its own, the reported pixels that are also 32 px from the border and
//! from any low-confidence pixel: there only the tile joins can differ, and they must not show
//! (max < 1e-3).
//!
//! The tiled result is also checked against the known motion in the bands of the spec
//! (`common::bands`), for all three motions. The third (92 px, beyond the tile margin, on a 512 px
//! image) is one where the coarse stage fails over a large region: before the trusted pre-warp
//! (spec 2.3) the pixels reported next to it differed by up to 1.6e-2 px between tiled and untiled
//! and were off by up to 0.13 px from the truth, at confidence 0.94 to 1.0. Each tile applies the
//! pre-warp rule on its own data; the reach of its fill (about 40 px) is below the 64 px margin,
//! which makes the tiles agree with the whole image where every reported pixel has a trusted pixel
//! within about `R` = 12.2 px (the `prewarp` module doc): an assumption that holds for these images,
//! not a bound.
mod common;
use common::bands::{check_bands, pipeline_valid};
use common::synth;
use optflow_core::config::Config;
use optflow_core::edt::distance_to_invalid;
use optflow_core::image::Image;
use optflow_core::phaseflow::{refine, Flow};
use optflow_core::poc::{coarse_flow, grid_sample};
use optflow_core::tile::refine_tiled;

const N: usize = 512;
const TILE: usize = 128;

/// The dense field `track_pair` hands to `refine`: the coarse grid interpolated to every pixel.
fn coarse_field(i1: &Image<f32>, i2: &Image<f32>, cfg: &Config) -> (Image<f32>, Image<f32>) {
    let g = coarse_flow(i1, i2, cfg, None);
    let (mut ux, mut uy) = (Image::new(i1.w, i1.h), Image::new(i1.w, i1.h));
    for y in 0..i1.h {
        for x in 0..i1.w {
            let (u, v) = grid_sample(&g, x as f64, y as f64);
            ux.data[y * i1.w + x] = u as f32;
            uy.data[y * i1.w + x] = v as f32;
        }
    }
    (ux, uy)
}

/// Distance of a pixel from the nearest image border (0 on the outermost row or column).
fn border_distance(i: usize) -> usize {
    let (x, y) = (i % N, i / N);
    x.min(y).min(N - 1 - x).min(N - 1 - y)
}

/// RMS and maximum of `a - b` over the selected values.
fn diff(a: &[f32], b: &[f32], mask: &[bool]) -> (f64, f64) {
    let (mut sum, mut count, mut worst) = (0.0f64, 0usize, 0.0f64);
    for ((&x, &y), _) in a.iter().zip(b).zip(mask).filter(|(_, &m)| m) {
        let d = f64::from(x) - f64::from(y);
        sum += d * d;
        count += 1;
        worst = worst.max(d.abs());
    }
    ((sum / count as f64).sqrt(), worst)
}

/// RMS of the displacement difference (x and y together) and the largest single difference.
fn disp_diff(tiled: &Flow, whole: &Flow, mask: &[bool]) -> (f64, f64) {
    let (rx, mx) = diff(&tiled.ux.data, &whole.ux.data, mask);
    let (ry, my) = diff(&tiled.uy.data, &whole.uy.data, mask);
    (((rx * rx + ry * ry) / 2.0).sqrt(), mx.max(my))
}

/// A pair of `N` x `N` speckle images related by `translation + uniform_strain(0.01, -0.003)`
/// about the image centre, the coarse field of the pair and the motion.
fn pair(
    t: (f64, f64),
    cfg: &Config,
) -> (
    Image<f32>,
    Image<f32>,
    Image<f32>,
    Image<f32>,
    synth::Affine,
) {
    let c = N as f64 / 2.0;
    let field = synth::uniform_strain(0.01, -0.003, (c, c), t);
    let pat = synth::make_pattern(N, N, 0, 120.0);
    let a = synth::render(&pat, N, N, None);
    let b = synth::render(&pat, N, N, Some(&field));
    let (ux, uy) = coarse_field(&a, &b, cfg);
    (a, b, ux, uy, field)
}

/// Reported pixels that are at least 32 px from the border and from any low-confidence pixel.
fn strict_mask(whole: &Flow, cfg: &Config, valid: &[bool]) -> Vec<bool> {
    let ok = Image::from_vec(
        N,
        N,
        whole
            .conf
            .data
            .iter()
            .map(|&c| f64::from(c) >= cfg.conf_min)
            .collect(),
    );
    let dist = distance_to_invalid(&ok);
    (0..N * N)
        .map(|i| valid[i] && f64::from(dist.data[i]) >= 32.0 && border_distance(i) >= 32)
        .collect()
}

/// `bands`: also check the tiled result against the known motion in the distance bands.
fn check(label: &str, t: (f64, f64), cfg: &Config, bands: bool) {
    let (a, b, ux, uy, field) = pair(t, cfg);
    let whole = refine(&a, &b, &ux, &uy, cfg);
    let tiled = refine_tiled(&a, &b, &ux, &uy, cfg, TILE);
    let valid = pipeline_valid(&whole.conf, cfg);
    let (rms, max) = disp_diff(&tiled, &whole, &valid);
    let all = vec![true; N * N];
    let (rms_all, max_all) = disp_diff(&tiled, &whole, &all);
    let (conf_rms, _) = diff(&tiled.conf.data, &whole.conf.data, &valid);
    println!(
        "{label}: reported pixels rms {rms:.3e} max {max:.3e} | all pixels rms {rms_all:.3e} \
         max {max_all:.3e} | conf rms (reported) {conf_rms:.3e}"
    );
    assert!(rms < 1e-3, "{label}: displacement rms {rms}");
    assert!(conf_rms < 1e-3, "{label}: confidence rms {conf_rms}");
    let strict = strict_mask(&whole, cfg, &valid);
    let (_, strict_max) = disp_diff(&tiled, &whole, &strict);
    println!("{label}: reported pixels 32 px from the border and from low confidence, max {strict_max:.3e}");
    assert!(strict_max < 1e-3, "{label}: max difference {strict_max}");
    if bands {
        let valid_tiled = pipeline_valid(&tiled.conf, cfg);
        check_bands(
            &format!("{label} (tiles of {TILE})"),
            &valid_tiled,
            &tiled.ux.data,
            &tiled.uy.data,
            &field,
            N,
            cfg,
        );
    }
}

#[test]
fn tiled_refine_equals_untiled_for_a_small_motion() {
    check("small motion", (0.0, 0.0), &Config::default(), true);
}

/// Real data moves by up to ~200 px: the second image's tile window has to follow the motion.
#[test]
fn tiled_refine_equals_untiled_for_a_large_motion() {
    let cfg = Config {
        max_disp: 64.0,
        ..Config::default()
    };
    check("large motion", (37.3, -21.6), &cfg, true);
}

/// Beyond the margin of a tile (64 px) the second image's window must be displaced to contain the
/// pixels the tile lands on at all.
#[test]
fn tiled_refine_equals_untiled_for_a_motion_beyond_the_tile_margin() {
    let cfg = Config {
        max_disp: 128.0,
        ..Config::default()
    };
    check("motion beyond the margin", (91.7, -67.3), &cfg, true);
}

#[test]
fn an_image_within_one_tile_is_refine_exactly() {
    let cfg = Config::default();
    let n = 160;
    let pat = synth::make_pattern(n, n, 3, 80.0);
    let f = synth::uniform_strain(0.01, -0.003, (80.0, 80.0), (2.4, -1.1));
    let a = synth::render(&pat, n, n, None);
    let b = synth::render(&pat, n, n, Some(&f));
    let (ux, uy) = synth::displacement(&f, n, n);
    let whole = refine(&a, &b, &ux, &uy, &cfg);
    for tile in [n, 1024] {
        let tiled = refine_tiled(&a, &b, &ux, &uy, &cfg, tile);
        assert_eq!(tiled.ux.data, whole.ux.data, "tile {tile}");
        assert_eq!(tiled.uy.data, whole.uy.data, "tile {tile}");
        assert_eq!(tiled.conf.data, whole.conf.data, "tile {tile}");
        assert_eq!(tiled.resid.data, whole.resid.data, "tile {tile}");
    }
}

/// Peak memory and time of `track_pair` on a large synthetic pair (tiled fine stage, banded
/// strain and guard). Not run by default:
/// `OPTFLOW_LARGE=3000x2000 /usr/bin/time -l cargo test --release -p optflow-core --test
/// tile_equivalence large_pair_report -- --ignored --nocapture` (the size defaults to 3000x2000).
#[test]
#[ignore]
fn large_pair_report() {
    use optflow_core::track::track_pair;
    let size = std::env::var("OPTFLOW_LARGE").unwrap_or_else(|_| "3000x2000".to_string());
    let (w, h) = size
        .split_once('x')
        .map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))
        .expect("OPTFLOW_LARGE=<width>x<height>");
    let t = std::time::Instant::now();
    let field = synth::uniform_strain(
        0.005,
        -0.002,
        (w as f64 / 2.0, h as f64 / 2.0),
        (14.3, -9.1),
    );
    let (a, b) = {
        let pat = synth::make_pattern(w, h, 1, 100.0);
        (
            synth::render(&pat, w, h, None),
            synth::render(&pat, w, h, Some(&field)),
        )
    };
    println!("rendered {w}x{h} in {:.1} s", t.elapsed().as_secs_f64());
    let cfg = Config::default();
    let t = std::time::Instant::now();
    let r = track_pair(&a, &b, &cfg, None, None).unwrap();
    println!(
        "track_pair {w}x{h}: {:.1} s (coarse {:.1}, fine {:.1}, guard + strain {:.1})",
        t.elapsed().as_secs_f64(),
        r.timing.coarse,
        r.timing.fine,
        r.timing.strain
    );
    let (mut sum, mut n, mut worst) = (0.0f64, 0usize, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let u = r.ux.data[y * w + x];
            if u.is_finite() {
                let (tx, _) = field.displacement(x as f64, y as f64);
                let d = f64::from(u) - tx;
                sum += d * d;
                n += 1;
                worst = worst.max(d.abs());
            }
        }
    }
    println!(
        "valid pixels {n} of {} ({:.1} %), ux error vs truth: rms {:.3e}, max {:.3e}",
        w * h,
        100.0 * n as f64 / (w * h) as f64,
        (sum / n.max(1) as f64).sqrt(),
        worst
    );
}

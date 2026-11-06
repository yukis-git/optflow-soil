//! Reported pixels next to a coarse-stage failure are accurate (spec 2.3, trusted pre-warp). Port
//! of `test_reported_pixels_next_to_a_coarse_stage_failure_are_accurate` of
//! `python/tests/test_track.py`.
//!
//! Translation (45.85, -33.65) px at 256 px with `max_disp` 64: the coarsest POC window cannot
//! hold the motion, so more than a fifth of the coarse field is off by over a pixel (up to ~100
//! px). The fine stage cannot repair those pixels, but it must not let their displacement leak
//! into the pre-warp of the good ones. Before the trusted pre-warp, 625 reported pixels beside
//! the failure of this setup were off by up to 0.165 px at confidence 0.94 to 1.0; with it the
//! error of the reported pixels is RMS 2.9e-4 and at most 1.6e-3 px. The pattern is this crate's
//! test generator (seed 2, margin 80), not numpy's, so the setup is not Python's seed 0, which
//! the generators do not share. Run with `-- --nocapture` to see the numbers.
mod common;
use common::bands::pipeline_valid;
use common::synth;
use common::{bits, on_one_thread};
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::poc::{coarse_flow, grid_sample};
use optflow_core::tile::refine_tiled;
use optflow_core::track::track_pair;

const N: usize = 256;

struct Setup {
    a: Image<f32>,
    b: Image<f32>,
    field: synth::Affine,
    cfg: Config,
    /// The dense field `track_pair` starts the fine stage from.
    start: (Image<f32>, Image<f32>),
}

fn setup() -> Setup {
    let c = N as f64 / 2.0;
    let field = synth::uniform_strain(0.01, -0.003, (c, c), (45.85, -33.65));
    let pat = synth::make_pattern(N, N, 2, 80.0);
    let a = synth::render(&pat, N, N, None);
    let b = synth::render(&pat, N, N, Some(&field));
    let cfg = Config {
        max_disp: 64.0,
        ..Config::default()
    };
    let g = coarse_flow(&a, &b, &cfg, None);
    let (mut ux, mut uy) = (Image::new(N, N), Image::new(N, N));
    for y in 0..N {
        for x in 0..N {
            let (u, v) = grid_sample(&g, x as f64, y as f64);
            ux.data[y * N + x] = u as f32;
            uy.data[y * N + x] = v as f32;
        }
    }
    Setup {
        a,
        b,
        field,
        cfg,
        start: (ux, uy),
    }
}

/// `|u - truth|` per pixel.
fn errors(field: &synth::Affine, ux: &[f32], uy: &[f32]) -> Vec<f64> {
    (0..N * N)
        .map(|i| {
            let (tx, ty) = field.displacement((i % N) as f64, (i / N) as f64);
            (f64::from(ux[i]) - tx).hypot(f64::from(uy[i]) - ty)
        })
        .collect()
}

/// RMS and maximum of `e` over the selected pixels, and how many there are.
fn rms_max(e: &[f64], keep: &[bool]) -> (f64, f64, usize) {
    let sel: Vec<f64> = e
        .iter()
        .zip(keep)
        .filter(|(_, &k)| k)
        .map(|(&v, _)| v)
        .collect();
    let rms = (sel.iter().map(|v| v * v).sum::<f64>() / sel.len().max(1) as f64).sqrt();
    (rms, sel.iter().copied().fold(0.0, f64::max), sel.len())
}

/// The case keeps exercising a coarse failure: more than a fifth of the start field is off by
/// more than a pixel.
fn assert_coarse_failure(s: &Setup) {
    let e = errors(&s.field, &s.start.0.data, &s.start.1.data);
    let share = e.iter().filter(|&&v| v > 1.0).count() as f64 / (N * N) as f64;
    println!(
        "coarse error > 1 px on {:.1} % of the pixels",
        100.0 * share
    );
    assert!(share > 0.2, "the coarse stage no longer fails: {share}");
}

#[test]
fn reported_pixels_next_to_a_coarse_stage_failure_are_accurate() {
    let s = setup();
    assert_coarse_failure(&s);
    let r = track_pair(&s.a, &s.b, &s.cfg, None, None).unwrap();
    let e = errors(&s.field, &r.ux.data, &r.uy.data);
    let ok: Vec<bool> = r.ux.data.iter().map(|v| v.is_finite()).collect();
    let (rms, max, n) = rms_max(&e, &ok);
    let bad = e.iter().zip(&ok).filter(|(&v, &k)| k && v > 1e-2).count();
    println!("track_pair: {n} reported pixels, rms {rms:.2e} max {max:.2e}, {bad} above 1e-2");
    assert!(n > N * N / 10, "only {n} reported pixels");
    assert!(
        rms <= 1e-3 && max <= 1e-2,
        "rms {rms} max {max} ({bad} above 1e-2)"
    );
}

/// The same through the tiled fine stage (4 tiles of 128 px with the 64 px margin): each tile
/// applies the pre-warp rule to its own data, the fill's reach (about 40 px) is within the margin.
/// That is an assumption and not a bound: it holds when every reported pixel has a trusted pixel
/// within about `R` = 12.2 px (the `prewarp` module doc), as here.
#[test]
fn reported_pixels_next_to_a_coarse_stage_failure_are_accurate_in_tiles() {
    let s = setup();
    let flow = refine_tiled(&s.a, &s.b, &s.start.0, &s.start.1, &s.cfg, 128);
    let e = errors(&s.field, &flow.ux.data, &flow.uy.data);
    let ok = pipeline_valid(&flow.conf, &s.cfg);
    let (rms, max, n) = rms_max(&e, &ok);
    println!("refine_tiled: {n} reported pixels, rms {rms:.2e} max {max:.2e}");
    assert!(n > N * N / 10, "only {n} reported pixels");
    assert!(rms <= 1e-3 && max <= 1e-2, "rms {rms} max {max}");
}

/// The pre-warp rule runs here on thousands of untrusted pixels (EDT, feature transform and fill, in
/// parallel by lines and rows): one thread and the default pool give the same bits, untiled and
/// tiled.
#[test]
fn the_pre_warp_rule_does_not_depend_on_the_thread_count() {
    let s = setup();
    let pair = || track_pair(&s.a, &s.b, &s.cfg, None, None).unwrap();
    let tiled = || refine_tiled(&s.a, &s.b, &s.start.0, &s.start.1, &s.cfg, 128);
    let (many, many_tiled) = (pair(), tiled());
    let (one, one_tiled) = on_one_thread(|| (pair(), tiled()));
    for (x, y) in [
        (&many.ux, &one.ux),
        (&many.uy, &one.uy),
        (&many.conf, &one.conf),
        (&many.raw_ux, &one.raw_ux),
        (&many.raw_uy, &one.raw_uy),
        (&many_tiled.ux, &one_tiled.ux),
        (&many_tiled.uy, &one_tiled.uy),
        (&many_tiled.conf, &one_tiled.conf),
        (&many_tiled.resid, &one_tiled.resid),
    ] {
        assert_eq!(bits(x), bits(y));
    }
    assert!(many.ux.data.iter().filter(|v| v.is_finite()).count() > N * N / 10);
}

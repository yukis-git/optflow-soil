//! The border rule of the fine stage (spec 2.4), port of the Python tests of the same names in
//! `python/tests/test_track.py` and `test_phaseflow.py`. Within `B = 4 sigma_w` of the first
//! image's border, and where the match is within `B` of the second image's border or outside it,
//! the confidence is 0 while the edge guard is on. Run with `-- --nocapture` for the tables.
mod common;
use common::bands::{check_bands, pipeline_valid};
use common::synth;
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::phaseflow::refine;
use optflow_core::tile::refine_tiled;
use optflow_core::track::track_pair;

const N: usize = 512;

/// A pair of `N` x `N` speckle images related by `translation + uniform_strain(0.01, -0.003)`
/// about the image centre.
fn pair(t: (f64, f64)) -> (Image<f32>, Image<f32>, synth::Affine) {
    let c = N as f64 / 2.0;
    let field = synth::uniform_strain(0.01, -0.003, (c, c), t);
    let pat = synth::make_pattern(N, N, 0, 120.0);
    (
        synth::render(&pat, N, N, None),
        synth::render(&pat, N, N, Some(&field)),
        field,
    )
}

fn check_track_pair(label: &str, t: (f64, f64), cfg: &Config) {
    let (a, b, field) = pair(t);
    let r = track_pair(&a, &b, cfg, None, None).unwrap();
    let valid: Vec<bool> = r.ux.data.iter().map(|v| v.is_finite()).collect();
    // the helper that the tiled tests use is the pipeline's rule: same pixels (a confidence of
    // exactly conf_min in f32 but below it in f64 is the only possible difference)
    let rule = pipeline_valid(&r.conf, cfg);
    let differ = valid.iter().zip(&rule).filter(|(a, b)| a != b).count();
    assert!(
        differ <= 2,
        "{label}: pipeline_valid differs at {differ} pixels"
    );
    check_bands(label, &valid, &r.ux.data, &r.uy.data, &field, N, cfg);
}

#[test]
fn valid_pixels_next_to_the_border_are_as_accurate_as_the_interior() {
    // before the rule, valid pixels 13-32 px from the border had rms 1e-2 .. 8e-4 px and up to
    // 0.11 px at confidence 0.997
    check_track_pair("border", (0.0, 0.0), &Config::default());
}

#[test]
fn valid_pixels_next_to_matches_that_leave_image_2_are_as_accurate_as_the_interior() {
    // large translation: the matches of the right and upper strips leave image 2 or come within B
    // of its border; before the rule the pixels beside them had errors of up to 4.7e-2 px at
    // confidence 0.992
    let cfg = Config {
        max_disp: 64.0,
        ..Config::default()
    };
    check_track_pair("matches leave image 2", (37.3, -21.6), &cfg);
}

const M: usize = 192;

/// Tile size of the tiled variants of the unit tests: 2 x 2 tiles on the `M` x `M` pair.
const TILE: usize = 128;

/// `refine` (`tile` is `None`) or `refine_tiled` of a pure translation `t` on an `M` x `M` pair,
/// started from the exact field.
fn conf_for_translation(t: (f64, f64), cfg: &Config, tile: Option<usize>) -> Image<f32> {
    let pat = synth::make_pattern(M, M, 11, 80.0);
    let field = synth::uniform_strain(0.0, 0.0, (0.0, 0.0), t);
    let a = synth::render(&pat, M, M, None);
    let b = synth::render(&pat, M, M, Some(&field));
    let (ux, uy) = synth::displacement(&field, M, M);
    match tile {
        None => refine(&a, &b, &ux, &uy, cfg).conf,
        Some(n) => refine_tiled(&a, &b, &ux, &uy, cfg, n).conf,
    }
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn column(c: &Image<f32>, x: usize) -> Vec<f32> {
    (40..M - 40).map(|y| c.get(x, y)).collect()
}

fn row(c: &Image<f32>, y: usize) -> Vec<f32> {
    (40..M - 40).map(|x| c.get(x, y)).collect()
}

#[test]
fn confidence_is_zero_within_four_sigma_w_of_the_image_border() {
    border_zone_of_image_1(None);
}

#[test]
fn confidence_is_zero_within_four_sigma_w_of_the_image_border_tiled() {
    border_zone_of_image_1(Some(TILE));
}

fn border_zone_of_image_1(tile: Option<usize>) {
    let cfg = Config::default();
    let b = (4.0 * cfg.sigma_w) as usize;
    assert!(cfg.edge_guard_px() > 0.0 && b == 16);
    let conf = conf_for_translation((0.4, 0.3), &cfg, tile);
    // distance B - 1 from the border: 0; distance B + 1, in good texture: confident
    for (strip, inner) in [
        (column(&conf, b - 1), column(&conf, b + 1)),
        (column(&conf, M - b), column(&conf, M - b - 2)),
        (row(&conf, b - 1), row(&conf, b + 1)),
        (row(&conf, M - b), row(&conf, M - b - 2)),
    ] {
        assert!(strip.iter().all(|&c| c == 0.0));
        assert!(median(inner) > 0.5);
    }
    for y in 0..M {
        for x in 0..M {
            if x.min(y).min(M - 1 - x).min(M - 1 - y) < b {
                assert_eq!(conf.get(x, y), 0.0, "({x}, {y})");
            }
        }
    }
}

#[test]
fn confidence_is_zero_where_the_match_is_within_four_sigma_w_of_the_border_of_image_2() {
    border_zone_of_image_2(None);
}

#[test]
fn confidence_is_zero_where_the_match_is_within_four_sigma_w_of_the_border_of_image_2_tiled() {
    border_zone_of_image_2(Some(TILE));
}

fn border_zone_of_image_2(tile: Option<usize>) {
    // x' = x + 30.5: the match of column x lies at x + 30.5 in image 2
    let b = 16.0;
    let conf = conf_for_translation((30.5, 0.0), &Config::default(), tile);
    let near = |x: usize| x as f64 + 30.5 > (M - 1) as f64 - b;
    let first_near = (0..M).find(|&x| near(x)).unwrap();
    assert!(first_near > 100 && first_near < M - 1);
    for x in first_near..M {
        assert!(column(&conf, x).iter().all(|&c| c == 0.0), "column {x}");
    }
    // the last column whose match is B or more inside
    assert!(median(column(&conf, first_near - 1)) > 0.5);
}

#[test]
fn border_rule_is_off_with_the_guard_off() {
    guard_off(None);
}

#[test]
fn border_rule_is_off_with_the_guard_off_tiled() {
    guard_off(Some(TILE));
}

fn guard_off(tile: Option<usize>) {
    // edge_guard = 0 keeps the old rule: zero only where the match leaves image 2. Translation
    // (-1.5, 1.7) takes columns 0, 1 and the last two rows out of the image, columns 2, 3 and the
    // rows above stay in.
    let cfg = Config {
        edge_guard: 0.0,
        ..Config::default()
    };
    let conf = conf_for_translation((-1.5, 1.7), &cfg, tile);
    for y in 0..M {
        assert!(conf.get(0, y) == 0.0 && conf.get(1, y) == 0.0);
    }
    for x in 0..M {
        assert!(conf.get(x, M - 2) == 0.0 && conf.get(x, M - 1) == 0.0);
    }
    assert!(median(column(&conf, 2)) > 0.5 && median(column(&conf, 3)) > 0.5);
    assert!(median(row(&conf, M - 3)) > 0.5 && median(row(&conf, M - 4)) > 0.5);
}

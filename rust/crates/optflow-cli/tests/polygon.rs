//! `polygon_mask` against masks drawn by Pillow's `ImageDraw.polygon(fill=1)` (the fixture is
//! written by `tests/fixtures/make_polygon_fixtures.py`).

use optflow_cli::polygon::{fill_polygon, polygon_mask};
use optflow_core::image::Image;
use serde_json::Value;

fn fixtures() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/polygon_cases.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture file"))
        .expect("fixture json")
}

fn unpack(hex: &str, w: usize, h: usize) -> Vec<bool> {
    let bytes: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    (0..w * h)
        .map(|i| bytes[i / 8] >> (7 - i % 8) & 1 == 1)
        .collect()
}

fn shape(v: &Value) -> (usize, usize) {
    let s = v["shape"].as_array().unwrap();
    (
        s[0].as_u64().unwrap() as usize,
        s[1].as_u64().unwrap() as usize,
    )
}

/// Panics with the number and first position of the differing pixels.
fn assert_same(got: &Image<bool>, want: &[bool], what: &str) {
    assert_eq!(got.data.len(), want.len(), "{what}: size");
    let bad: Vec<usize> = (0..want.len())
        .filter(|&i| got.data[i] != want[i])
        .collect();
    assert!(
        bad.is_empty(),
        "{what}: {} of {} pixels differ, first at (x={}, y={})",
        bad.len(),
        want.len(),
        bad.first().map_or(0, |i| i % got.w),
        bad.first().map_or(0, |i| i / got.w)
    );
}

#[test]
fn simple_polygons_match_pillow_pixel_for_pixel() {
    let f = fixtures();
    let cases = f["random"].as_array().unwrap();
    assert!(cases.len() >= 150);
    for (n, c) in cases.iter().enumerate() {
        let (h, w) = shape(c);
        let pts: Vec<(i32, i32)> = c["pts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32))
            .collect();
        let want = unpack(c["mask"].as_str().unwrap(), w, h);
        assert_same(
            &fill_polygon(w, h, &pts),
            &want,
            &format!("case {n} {pts:?}"),
        );
    }
}

#[test]
fn scaled_float_vertices_are_truncated_like_pillow() {
    let f = fixtures();
    for (n, c) in f["scaled"].as_array().unwrap().iter().enumerate() {
        let (h, w) = shape(c);
        let scale = c["scale"].as_u64().unwrap() as usize;
        let pts: Vec<(f64, f64)> = c["pts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
            .collect();
        let want = unpack(c["mask"].as_str().unwrap(), w, h);
        assert_same(
            &polygon_mask(w, h, &pts, scale),
            &want,
            &format!("scaled case {n} {pts:?}"),
        );
    }
}

#[test]
fn example_roi_matches_pillow_at_full_half_and_quarter_size() {
    let f = fixtures();
    let roi: Vec<(f64, f64)> = f["roi"]["polygon"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
        .collect();
    for c in f["roi"]["cases"].as_array().unwrap() {
        let (h, w) = shape(c);
        let scale = c["scale"].as_u64().unwrap() as usize;
        let mut want = vec![false; w * h];
        for (y, row) in c["runs"].as_array().unwrap().iter().enumerate() {
            for run in row.as_array().unwrap().chunks(2) {
                let (a, b) = (
                    run[0].as_u64().unwrap() as usize,
                    run[1].as_u64().unwrap() as usize,
                );
                want[y * w + a..=y * w + b].fill(true);
            }
        }
        assert_eq!(
            want.iter().filter(|&&v| v).count() as u64,
            c["area"].as_u64().unwrap()
        );
        assert_same(
            &polygon_mask(w, h, &roi, scale),
            &want,
            &format!("ROI at scale {scale}"),
        );
    }
}

#[test]
fn degenerate_polygons_do_not_panic() {
    for pts in [
        vec![],
        vec![(3, 3)],
        vec![(1, 1), (6, 6)],
        vec![(2, 2), (2, 2), (2, 2)],
        vec![(-9, -9), (99, 99), (-9, 99)],
    ] {
        let m = fill_polygon(8, 8, &pts);
        assert_eq!(m.data.len(), 64);
    }
    // a single point and a segment are drawn like Pillow does
    assert!(fill_polygon(8, 8, &[(3, 3), (3, 3), (3, 3)]).data[3 * 8 + 3]);
}

/// The mask as one string per row, `#` for a set pixel.
fn rows(m: &Image<bool>) -> Vec<String> {
    m.data
        .chunks(m.w)
        .map(|r| r.iter().map(|&v| if v { '#' } else { '.' }).collect())
        .collect()
}

#[test]
fn extreme_i32_vertices_are_filled_without_overflow() {
    // The edge arithmetic (x1 - x0, y - y0, y + 1, ...) is done in i64: in i32 it overflows, which
    // panics in a debug build and silently wraps in a release build. The masks below are what the
    // geometry says; wrapped arithmetic gives other pixels (for these polygons: a filled image
    // for `thin` and `far`, an empty one for `thin_2e9`, one row for `triangle`), so this fails in
    // `cargo test --release` as well as in a debug build.
    let (lo, hi) = (i32::MIN, i32::MAX);
    let (w, h) = (8, 4);
    let all = vec!["########"; 4];
    let none = vec!["........"; 4];
    let column_4 = vec!["....#..."; 4];

    // a polygon around the image fills it
    let m = fill_polygon(w, h, &[(lo, lo), (hi, lo), (hi, hi), (lo, hi)]);
    assert_eq!(rows(&m), all, "around");
    // a thin triangle with its apex far above and its base far below is one pixel wide (x 3.5 to
    // 4.5) on the rows of the image, which are half way between the apex and the base
    let m = fill_polygon(w, h, &[(4, lo), (5, hi), (3, hi)]);
    assert_eq!(rows(&m), column_4, "thin");
    let m = fill_polygon(
        w,
        h,
        &[(4, -2_000_000_000), (5, 2_000_000_000), (3, 2_000_000_000)],
    );
    assert_eq!(rows(&m), column_4, "thin_2e9");
    // a triangle whose long edges leave the image at the left: row 2 and 3 are inside
    let m = fill_polygon(w, h, &[(lo, 0), (hi, 3), (0, hi)]);
    assert_eq!(
        rows(&m),
        ["........", "........", "########", "########"],
        "triangle"
    );
    // a triangle entirely to the right of the image
    let pts = [
        (1_500_000_000, -2_000_000_000),
        (2_000_000_000, 2_000_000_000),
        (1_600_000_000, 5),
    ];
    assert_eq!(rows(&fill_polygon(w, h, &pts)), none, "far");
    // more extreme shapes that must not panic
    for pts in [
        vec![(hi, hi), (lo, lo), (hi, lo)],
        vec![(0, lo), (0, hi), (hi, 4), (lo, 4)],
    ] {
        assert_eq!(fill_polygon(8, 8, &pts).data.len(), 64);
    }
}

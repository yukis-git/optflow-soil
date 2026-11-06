//! Parity of the NaN-aware Savitzky-Golay gradient, the strain fields, the exact EDT and the
//! 4-connected component filter with the Python fixtures (`python/optflow_ref/strain.py`,
//! `scipy.ndimage.distance_transform_edt`, `track._large_components`).
//! Run with `-- --nocapture` to see the measured errors.
mod common;
use optflow_core::edt::{distance_to_invalid, large_components, nearest_source};
use optflow_core::image::Image;
use optflow_core::strain::{displacement_gradient, strain_fields, Gradient};

/// Largest difference to the fixtures accepted. Measured (`-- --nocapture`): 1.5e-8 at most over the
/// gradient and the strain fields, 0 for the EDT; the tolerance is 13 times the largest of them.
const TOL: f64 = 2e-7;

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    assert_eq!(shape.len(), 2, "{name}: expected a 2-D fixture");
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

/// Asserts that `got` and the fixture `name` have NaN at exactly the same pixels and that the
/// finite pixels agree within `TOL`. Returns (max |diff| over finite pixels, NaN count).
fn check_field(name: &str, got: &Image<f32>) -> (f64, usize) {
    let (shape, want) = common::load_f64(name);
    assert_eq!((got.h, got.w), (shape[0], shape[1]), "{name}: shape");
    let (mut max, mut nans, mut finite) = (0.0f64, 0usize, 0usize);
    for (i, (&g, &w)) in got.data.iter().zip(&want).enumerate() {
        assert_eq!(
            g.is_nan(),
            w.is_nan(),
            "{name}: NaN mismatch at pixel {i} (got {g}, want {w})"
        );
        if g.is_nan() {
            nans += 1;
        } else {
            finite += 1;
            max = max.max((f64::from(g) - w).abs());
        }
    }
    assert!(finite > 0, "{name}: no finite pixel compared");
    assert!(max < TOL, "{name}: max |diff| = {max:.3e} >= {TOL:e}");
    println!("{name:>12}: max |diff| = {max:.3e}  NaN pixels = {nans} (positions identical)");
    (max, nans)
}

fn fixture_gradient() -> Gradient {
    Gradient {
        dudx: img_f32("strain_dudx"),
        dudy: img_f32("strain_dudy"),
        dvdx: img_f32("strain_dvdx"),
        dvdy: img_f32("strain_dvdy"),
    }
}

#[test]
fn strain_gradient_matches_python() {
    let (ux, uy) = (img_f32("strain_ux"), img_f32("strain_uy"));
    assert!(
        ux.data.iter().any(|v| v.is_nan()),
        "fixture needs a NaN hole"
    );
    let g = displacement_gradient(&ux, &uy, 15);
    let (_, nans) = check_field("strain_dudx", &g.dudx);
    assert!(nans > 0, "the NaN hole must propagate into the gradient");
    check_field("strain_dudy", &g.dudy);
    check_field("strain_dvdx", &g.dvdx);
    check_field("strain_dvdy", &g.dvdy);
}

#[test]
fn strain_matches_python_from_rust_gradient() {
    let (ux, uy) = (img_f32("strain_ux"), img_f32("strain_uy"));
    let s = strain_fields(&displacement_gradient(&ux, &uy, 15));
    check_field("strain_Exx", &s.exx_gl);
    check_field("strain_Eyy", &s.eyy_gl);
    check_field("strain_Exy", &s.exy_gl);
    check_field("strain_gmax", &s.gmax);
    check_field("strain_vol", &s.vol);
    // the small-strain fields are plain copies / halves of the gradient
    check_field("strain_dudx", &s.exx);
    check_field("strain_dvdy", &s.eyy);
}

#[test]
fn strain_matches_python_from_python_gradient() {
    // isolates `strain_fields` from the gradient solver: same inputs as the Python function
    let g = fixture_gradient();
    let s = strain_fields(&g);
    check_field("strain_Exx", &s.exx_gl);
    check_field("strain_Eyy", &s.eyy_gl);
    check_field("strain_Exy", &s.exy_gl);
    check_field("strain_gmax", &s.gmax);
    check_field("strain_vol", &s.vol);
    for ((&a, &b), &c) in s.exy.data.iter().zip(&g.dudy.data).zip(&g.dvdx.data) {
        assert_eq!(a.is_nan(), b.is_nan() || c.is_nan());
        if !a.is_nan() {
            assert!((f64::from(a) - 0.5 * (f64::from(b) + f64::from(c))).abs() < 1e-6);
        }
    }
}

#[test]
fn edt_matches_scipy() {
    let (shape, v) = common::load_f64("edt_valid");
    let valid = Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x != 0.0).collect());
    assert!(
        valid.data.iter().any(|&b| !b),
        "fixture needs invalid pixels"
    );
    let got = distance_to_invalid(&valid);
    let (_, want) = common::load_f64("edt_dist");
    let max = got
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max);
    println!("edt: max |diff| = {max:.3e}");
    assert!(max < TOL, "edt max |diff| = {max:.3e}");
}

/// The feature transform (`distance_transform_edt(~src, return_indices=True)`) of four random
/// source sets with many equidistant nearest sources: the index of the nearest source is scipy's,
/// ties included (spec 2.3).
#[test]
fn nearest_source_matches_scipy_indices_including_ties() {
    let (shape, src_v) = common::load_f64("edt_ft_src");
    let (_, idx_v) = common::load_f64("edt_ft_idx");
    assert_eq!(shape.len(), 3);
    let (n, h, w) = (shape[0], shape[1], shape[2]);
    let mut tied = 0usize;
    for k in 0..n {
        let range = k * h * w..(k + 1) * h * w;
        let src = Image::from_vec(
            w,
            h,
            src_v[range.clone()].iter().map(|&x| x != 0.0).collect(),
        );
        let got = nearest_source(&src).expect("the fixture has sources");
        let want: Vec<u32> = idx_v[range].iter().map(|&x| x as u32).collect();
        assert_eq!(got, want, "source set {k}");
        // pixels whose nearest source is not unique: the fixture must actually exercise the tie rule
        let sources: Vec<(i64, i64)> = (0..h * w)
            .filter(|&p| src.data[p])
            .map(|p| ((p % w) as i64, (p / w) as i64))
            .collect();
        for p in 0..h * w {
            let (x, y) = ((p % w) as i64, (p / w) as i64);
            let d2: Vec<i64> = sources
                .iter()
                .map(|&(sx, sy)| (x - sx).pow(2) + (y - sy).pow(2))
                .collect();
            let best = *d2.iter().min().unwrap();
            tied += (d2.iter().filter(|&&d| d == best).count() > 1) as usize;
        }
    }
    println!("nearest_source: {n} sets of {h}x{w}, {tied} pixels with tied nearest sources");
    assert!(tied > 200, "only {tied} tied pixels: weak fixture");
}

fn mask(rows: &[&str]) -> Image<bool> {
    let w = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == w));
    let data = rows
        .iter()
        .flat_map(|r| r.bytes().map(|c| c == b'#'))
        .collect();
    Image::from_vec(w, rows.len(), data)
}

#[test]
fn edt_large_components_match_scipy_label() {
    // Expected values come from scipy.ndimage.label (default 4-connected structure) with the
    // `_large_components` logic of python/optflow_ref/track.py. Component sizes in label order:
    // [3 3 3 3 6 1 3 1 1 2]; diagonal neighbours such as (row 3, col 5) and (row 4, col 4) stay
    // in separate components.
    let b = mask(&[
        "##..#...#.",
        "#...##..##",
        "..#.......",
        ".##..###..",
        "....#.#...",
        "#.#...##.#",
        "#.........",
        "#..##.....",
    ]);
    let cases: [(f64, [&str; 8]); 4] = [
        (
            1.0,
            [
                "##..#...#.",
                "#...##..##",
                "..#.......",
                ".##..###..",
                "....#.#...",
                "#.#...##.#",
                "#.........",
                "#..##.....",
            ],
        ),
        (
            3.0,
            [
                "##..#...#.",
                "#...##..##",
                "..#.......",
                ".##..###..",
                "......#...",
                "#.....##..",
                "#.........",
                "#.........",
            ],
        ),
        (
            4.5,
            [
                "..........",
                "..........",
                "..........",
                ".....###..",
                "......#...",
                "......##..",
                "..........",
                "..........",
            ],
        ),
        (
            100.0,
            [
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
            ],
        ),
    ];
    for (min_area, want) in cases {
        let got = large_components(&b, min_area);
        assert_eq!(got, mask(&want), "min_area = {min_area}");
    }
    // no foreground at all: nothing to keep
    let empty = Image::<bool>::new(4, 3);
    assert_eq!(large_components(&empty, 1.0), empty);
}

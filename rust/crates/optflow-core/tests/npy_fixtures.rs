//! Reads the Python-written fixtures with `npy::read`, checks `npy::write_f32` reproduces numpy's
//! bytes exactly, and tests the shared helpers in `common`.
mod common;

use optflow_core::npy::{self, NpyData};

#[test]
fn npy_every_fixture_parses_and_rewrites_byte_identical() {
    let (mut f32_count, mut f64_count) = (0, 0);
    for entry in std::fs::read_dir(common::fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "npy") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let arr = npy::read(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        match &arr.data {
            // numpy wrote these as <f4: our writer must reproduce the file byte for byte
            NpyData::F32(data) => {
                assert_eq!(
                    npy::write_f32(&arr.shape, data),
                    bytes,
                    "{}",
                    path.display()
                );
                f32_count += 1;
            }
            // a few inputs are kept as <f8 (interp_x, interp_y): they only need to parse
            NpyData::F64(_) => f64_count += 1,
            _ => panic!("{}: expected <f4 or <f8 data", path.display()),
        }
    }
    assert!(f32_count >= 74, "only {f32_count} <f4 fixtures found");
    assert!(f64_count >= 2, "only {f64_count} <f8 fixtures found");
}

#[test]
fn npy_fixture_shapes_and_c_order() {
    let (shape, img) = common::load_f64("interp_img");
    assert_eq!(shape, vec![40, 50]);
    assert!(img.iter().all(|v| v.is_finite()));
    let (shape, xs) = common::load_f64("interp_x");
    assert_eq!(shape, vec![400]);
    assert!(matches!(common::load("interp_x").data, NpyData::F64(_)));
    assert!(xs.iter().all(|v| (-3.0..=53.0).contains(v)));
    assert_eq!(common::load_f64("poc_strip_ux").0, vec![3, 29]);
    assert_eq!(common::load_f64("seq_frames").0, vec![4, 128, 128]);
    assert_eq!(common::load_f64("gabor_r_re").0, vec![12, 64, 64]); // 3 scales x 4 orientations

    // strain_ux[y][x] = 0.01 x + 0.002 y + 1e-4 x y + 0.3 sin(x / 9), stored row by row
    let (shape, ux) = common::load_f64("strain_ux");
    assert_eq!(shape, vec![60, 70]);
    let (x, y) = (3usize, 2usize);
    let want = 0.01 * 3.0 + 0.002 * 2.0 + 1e-4 * 6.0 + 0.3 * (3.0f64 / 9.0).sin();
    assert!((ux[y * 70 + x] - want).abs() < 1e-6);
    // the NaN hole sux[20:26, 30:37] survives the round trip through numpy
    assert_eq!(ux.iter().filter(|v| v.is_nan()).count(), 6 * 7);
    assert!(ux[20 * 70 + 30].is_nan() && ux[25 * 70 + 36].is_nan() && !ux[26 * 70 + 36].is_nan());
}

#[test]
fn npy_meta_json_has_expected_keys() {
    let meta = common::meta();
    assert_eq!(
        meta["poc_shift"]["window"],
        serde_json::json!([32, 96, 32, 96])
    );
    assert!(meta["poc_shift"]["peak"].as_f64().unwrap() > 0.5);
    assert!(!meta["poc_grid"]["xs"].as_array().unwrap().is_empty());
    assert_eq!(meta["poc_strip"]["max_disp"], 100);
    assert_eq!(meta["poc_strip"]["n_levels"], 2);
    assert_eq!(meta["poc_strip"]["ys"].as_array().unwrap().len(), 3);
    assert_eq!(meta["level_window"].as_array().unwrap().len(), 9);
    assert_eq!(meta["n_levels"].as_array().unwrap().len(), 16);
    assert_eq!(meta["capture_range"].as_array().unwrap().len(), 14);
    assert_eq!(meta["seq_reference"].as_array().unwrap().len(), 3);
    // the coarse stage with a guess (same grid as poc_grid) and the stripe pair of refine
    assert_eq!(meta["poc_init"], meta["poc_grid"]);
    assert_eq!(meta["poc_fallback"], meta["poc_grid"]);
    assert_eq!(meta["refine_stripe"]["init"].as_array().unwrap().len(), 2);
}

/// The generator records which versions wrote the fixtures. No test depends on the values: they say
/// where a diff after a regeneration on another machine may come from.
#[test]
fn npy_meta_json_records_the_versions_that_wrote_the_fixtures() {
    let meta = common::meta();
    for key in ["python", "numpy", "scipy"] {
        let v = meta["provenance"][key]
            .as_str()
            .unwrap_or_else(|| panic!("meta.json has no provenance.{key}"));
        assert!(v.starts_with(|c: char| c.is_ascii_digit()), "{key}: {v:?}");
    }
}

#[test]
fn npy_fixtures_stay_under_eight_megabytes() {
    let total: u64 = std::fs::read_dir(common::fixture_dir())
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(total < 8_000_000, "fixtures are {total} bytes");
}

#[test]
fn npy_common_rms_diff_basic() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f64, 2.0, 3.0, 6.0];
    assert!((common::rms_diff(&a, &b, None) - 1.0).abs() < 1e-12); // sqrt(4 / 4)
    let mask = [true, true, true, false];
    assert_eq!(common::rms_diff(&a, &b, Some(&mask)), 0.0);
    assert!(common::rms_diff(&[], &[], None).is_nan());
}

#[test]
fn npy_common_rms_diff_is_strict_about_one_sided_non_finite() {
    // one side non-finite -> NaN, whichever side and whichever kind of non-finite
    assert!(common::rms_diff(&[1.0, f32::NAN], &[1.0, 2.0], None).is_nan());
    assert!(common::rms_diff(&[1.0, 2.0], &[1.0, f64::INFINITY], None).is_nan());
    assert!(common::rms_diff(&[f32::NEG_INFINITY, 2.0], &[1.0, 2.0], None).is_nan());
    // pairs where both sides are non-finite are skipped: sqrt((0 + 4) / 2)
    let a = [1.0f32, f32::NAN, 3.0];
    let b = [1.0f64, f64::NAN, 5.0];
    assert!((common::rms_diff(&a, &b, None) - 2.0f64.sqrt()).abs() < 1e-12);
    // (inf, NaN) is also "both non-finite": skipped, not an error
    assert_eq!(
        common::rms_diff(&[1.0, f32::INFINITY], &[1.0, f64::NAN], None),
        0.0
    );
    // a one-sided value outside the mask does not count, inside it does
    let a = [1.0f32, f32::NAN, 3.0];
    let b = [1.0f64, 2.0, 3.0];
    assert_eq!(common::rms_diff(&a, &b, Some(&[true, false, true])), 0.0);
    assert!(common::rms_diff(&a, &b, Some(&[true, true, true])).is_nan());
    // nothing compared (all masked out, or only doubly non-finite pairs) -> NaN
    assert!(common::rms_diff(&a, &b, Some(&[false; 3])).is_nan());
    assert!(common::rms_diff(&[f32::NAN], &[f64::NAN], None).is_nan());
}

#[test]
fn npy_common_rms_diff_common_skips_every_non_finite_pair() {
    let a = [1.0f32, f32::NAN, 3.0, f32::INFINITY, 5.0];
    let b = [2.0f64, 9.0, f64::NAN, 1.0, 5.0];
    // only pairs 0 and 4 are finite on both sides: sqrt((1 + 0) / 2)
    assert!((common::rms_diff_common(&a, &b, None) - 0.5f64.sqrt()).abs() < 1e-12);
    let mask = [false, true, true, true, true];
    assert_eq!(common::rms_diff_common(&a, &b, Some(&mask)), 0.0);
    assert!(common::rms_diff_common(&a, &b, Some(&[false; 5])).is_nan());
    assert!(common::rms_diff_common(&[f32::NAN], &[1.0], None).is_nan());
}

#[test]
fn npy_common_nan_agreement() {
    let a = [1.0f32, f32::NAN, 3.0, f32::INFINITY];
    let b = [1.0f64, f64::NAN, f64::NAN, f64::INFINITY];
    // pairs 0, 1, 3 agree about finiteness, pair 2 does not
    assert_eq!(common::nan_agreement(&a, &b, None), 0.75);
    assert!(
        (common::nan_agreement(&a, &b, Some(&[true, true, true, false])) - 2.0 / 3.0).abs() < 1e-12
    );
    assert_eq!(
        common::nan_agreement(&a, &b, Some(&[true, true, false, true])),
        1.0
    );
    assert_eq!(common::nan_agreement(&[1.0], &[2.0], None), 1.0);
    assert!(common::nan_agreement(&a, &b, Some(&[false; 4])).is_nan());
    assert!(common::nan_agreement(&[], &[], None).is_nan());
}

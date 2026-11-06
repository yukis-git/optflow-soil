//! The automatic ROI crop (`--crop {auto,off}`, `--crop-margin N`) of the `optflow` binary: the
//! outputs of `auto` and `off` agree inside the ROI, `crop.json`, the full-size outputs and the
//! border warning.

mod common;

use std::path::Path;
use std::process::Output;

use common::*;
use optflow_cli::polygon::polygon_mask;

const N: usize = 768;
/// A strip of about 30 % of the image, in the middle.
const ROI: [(f64, f64); 4] = [(84.0, 236.0), (684.0, 236.0), (684.0, 531.0), (84.0, 531.0)];
/// Margin 189 = ceil(32) + ceil(4 * 4) + ceil(12.2) + 32 + (64 / 2 + 16) * 2^1; A = 16 * 2^1 = 32
/// rounds y0 = 236 - 189 = 47 down to 32; x0 and the bottom are clipped (y1 = 532 + 189 = 721).
const EXPECTED_CROP_JSON: &str =
    "{\n  \"x0\": 0,\n  \"y0\": 32,\n  \"x1\": 768,\n  \"y1\": 721,\n  \"margin\": 189,\n  \"mode\": \"auto\"\n}";
const OFF_JSON: &str = "{\n  \"mode\": \"off\"\n}";
const WARNING: &str = "warning: the material moves close to the crop border \u{2014} increase --crop-margin or use --crop off";
/// The sand strip of the second and third scenes: 384 x 288 images.
const STRIP: [(f64, f64); 4] = [(130.0, 90.0), (250.0, 90.0), (250.0, 200.0), (130.0, 200.0)];

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn write_roi(path: &Path, poly: &[(f64, f64)]) {
    let pts: Vec<String> = poly.iter().map(|(x, y)| format!("[{x}, {y}]")).collect();
    std::fs::write(path, format!("{{\"polygon\": [{}]}}", pts.join(", "))).unwrap();
}

fn roi_mask(w: usize, h: usize, poly: &[(f64, f64)]) -> Vec<bool> {
    polygon_mask(w, h, poly, 1).data
}

/// `n` frames `f0.png`, `f1.png`... of a pattern moving by `dx` px per frame.
fn write_frames(dir: &Path, w: usize, h: usize, seed: u64, n: usize, dx: f64) {
    let p = make_pattern(w, h, seed);
    for k in 0..n {
        write_png16(
            &dir.join(format!("f{k}.png")),
            w,
            h,
            &render(&p, w, h, dx * k as f64, 0.0),
        );
    }
}

/// `optflow <cmd> <--option path>... <extra>...`
fn run(cmd: &str, paths: &[(&str, &Path)], extra: &[&str]) -> Output {
    let mut args = vec![cmd];
    for (option, path) in paths {
        args.push(option);
        args.push(s(path));
    }
    args.extend_from_slice(extra);
    optflow(&args)
}

/// `pair` of `dir/f0.png` and `dir/f1.png` with the ROI `dir/roi.json`.
fn pair(dir: &Path, out: &Path, extra: &[&str]) -> Output {
    let paths = [
        ("--ref", dir.join("f0.png")),
        ("--def", dir.join("f1.png")),
        ("--roi", dir.join("roi.json")),
        ("--out", out.to_path_buf()),
    ];
    let paths: Vec<(&str, &Path)> = paths.iter().map(|(o, p)| (*o, p.as_path())).collect();
    run("pair", &paths, extra)
}

/// `seq` of the frames in `dir` with the ROI `dir/roi.json`.
fn seq(dir: &Path, out: &Path, extra: &[&str]) -> Output {
    run(
        "seq",
        &[
            ("--frames", dir),
            ("--roi", &dir.join("roi.json")),
            ("--out", out),
        ],
        extra,
    )
}

fn field(dir: &Path, name: &str) -> (Vec<usize>, Vec<f64>) {
    load_npy(&dir.join(format!("{name}.npy")))
}

fn text(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// RMS difference of `a` and `b` over the ROI pixels where both are finite, how many there are,
/// and the share of the ROI pixels where both agree on being NaN or not.
fn compare(a: &[f64], b: &[f64], roi: &[bool]) -> (f64, usize, f64) {
    let (mut sum, mut both, mut agree, mut total) = (0.0, 0usize, 0usize, 0usize);
    for i in (0..roi.len()).filter(|&i| roi[i]) {
        total += 1;
        agree += usize::from(a[i].is_nan() == b[i].is_nan());
        if a[i].is_finite() && b[i].is_finite() {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
            both += 1;
        }
    }
    (
        (sum / both.max(1) as f64).sqrt(),
        both,
        agree as f64 / total as f64,
    )
}

/// Checks that `auto` and `off` (output directories of the same pair) agree inside `roi`.
fn assert_auto_matches_off(auto: &Path, off: &Path, roi: &[bool], (h, w): (usize, usize)) {
    assert_auto_near_off(auto, off, roi, (h, w), 0.999);
}

/// [`assert_auto_matches_off`] with the least share `min_agree` of the ROI pixels where both agree on NaN.
fn assert_auto_near_off(
    auto: &Path,
    off: &Path,
    roi: &[bool],
    (h, w): (usize, usize),
    min_agree: f64,
) {
    for name in [
        "ux", "uy", "conf", "Exx", "Eyy", "Exy", "gmax", "vol", "valid",
    ] {
        assert_eq!(field(auto, name).0, [h, w], "{name}");
        assert_eq!(field(off, name).0, [h, w], "{name}");
    }
    let n_roi = roi.iter().filter(|&&b| b).count();
    for name in ["ux", "uy"] {
        let (rms, both, agree) = compare(&field(auto, name).1, &field(off, name).1, roi);
        assert!(
            both * 2 > n_roi,
            "{name}: only {both} of {n_roi} ROI pixels valid in both"
        );
        assert!(rms <= 1e-3, "{name}: RMS difference {rms}");
        assert!(agree >= min_agree, "{name}: NaN agreement {agree}");
    }
}

#[test]
fn pair_crop_auto_matches_crop_off_inside_the_roi() {
    let dir = scratch("crop_pair");
    let p = make_pattern(N, N, 11);
    write_png16(&dir.join("f0.png"), N, N, &render(&p, N, N, 0.0, 0.0));
    write_png16(&dir.join("f1.png"), N, N, &render(&p, N, N, 2.0, -1.5));
    write_roi(&dir.join("roi.json"), &ROI);
    let (auto, off) = (dir.join("auto"), dir.join("off"));
    let o = pair(&dir, &auto, &["--set", "outer_iters=4"]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(!stderr(&o).contains("crop border"), "{}", stderr(&o));
    let o = pair(&dir, &off, &["--set", "outer_iters=4", "--crop", "off"]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));

    let roi = roi_mask(N, N, &ROI);
    let share = roi.iter().filter(|&&b| b).count() as f64 / (N * N) as f64;
    assert!((0.25..0.35).contains(&share), "{share}");
    assert_auto_matches_off(&auto, &off, &roi, (N, N));
    let (_, ux) = field(&auto, "ux");
    let (_, uy) = field(&auto, "uy");
    let in_roi = |f: &[f64]| median_of_finite((0..N * N).filter(|&i| roi[i]).map(|i| f[i]));
    assert!((in_roi(&ux) - 2.0).abs() < 0.02 && (in_roi(&uy) + 1.5).abs() < 0.02);

    assert_eq!(text(&auto.join("crop.json")), EXPECTED_CROP_JSON);
    assert_eq!(text(&off.join("crop.json")), OFF_JSON);

    // full-size outputs: NaN outside the crop, confidence 0, valid 0
    let outside = |i: usize| !(32..721).contains(&(i / N));
    for name in ["ux", "uy", "Exx", "Eyy", "Exy", "gmax", "vol"] {
        let (_, f) = field(&auto, name);
        assert!(
            (0..N * N).filter(|&i| outside(i)).all(|i| f[i].is_nan()),
            "{name}"
        );
    }
    let (_, conf) = field(&auto, "conf");
    let (_, valid) = field(&auto, "valid");
    assert!((0..N * N)
        .filter(|&i| outside(i))
        .all(|i| conf[i] == 0.0 && valid[i] == 0.0));
    assert!(valid.contains(&1.0));
    for (v, u) in valid.iter().zip(&ux) {
        assert_eq!(*v == 1.0, u.is_finite());
    }
}

#[test]
fn crop_json_says_off_without_a_roi_and_the_margin_applies_as_given() {
    let dir = scratch("crop_json");
    write_frames(&dir, 128, 128, 2, 2, 0.5);
    write_roi(
        &dir.join("roi.json"),
        &[(30.0, 30.0), (100.0, 30.0), (100.0, 100.0), (30.0, 100.0)],
    );
    let out = dir.join("out");
    let o = run(
        "pair",
        &[
            ("--ref", &dir.join("f0.png")),
            ("--def", &dir.join("f1.png")),
            ("--out", &out),
        ],
        &["--set", "outer_iters=2"],
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_eq!(text(&out.join("crop.json")), OFF_JSON);
    // the ROI covers x, y in 30..=100; margin 10 -> 20, A = 32 -> 0; x1 = y1 = 101 + 10
    let o = pair(
        &dir,
        &out,
        &["--set", "outer_iters=2", "--crop-margin", "10"],
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let want = "{\n  \"x0\": 0,\n  \"y0\": 0,\n  \"x1\": 111,\n  \"y1\": 111,\n  \"margin\": 10,\n  \"mode\": \"auto\"\n}";
    assert_eq!(text(&out.join("crop.json")), want);
    assert_eq!(field(&out, "ux").0, [128, 128]);
}

#[test]
fn bad_crop_options_are_errors_with_status_2() {
    let dir = scratch("crop_errors");
    write_frames(&dir, 96, 96, 2, 2, 0.5);
    write_roi(
        &dir.join("roi.json"),
        &[(10.0, 10.0), (80.0, 10.0), (80.0, 80.0)],
    );
    let out = dir.join("out");
    let o = pair(&dir, &out, &["--crop", "maybe"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o)
            .contains("error: argument --crop: invalid choice: 'maybe' (choose from auto, off)"),
        "{}",
        stderr(&o)
    );
    let o = pair(&dir, &out, &["--crop-margin", "x"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("error: argument --crop-margin: invalid int value: 'x'"));
    let o = pair(&dir, &out, &["--crop-margin", "-3"]);
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(
        stderr(&o).trim(),
        "error: --crop-margin must be >= 0, got -3"
    );
    // a crop below poc_win: x, y in 40..=50 with margin 0 give 32..51 after the alignment
    write_roi(
        &dir.join("tiny.json"),
        &[(40.0, 40.0), (50.0, 40.0), (50.0, 50.0)],
    );
    let o = run(
        "pair",
        &[
            ("--ref", &dir.join("f0.png")),
            ("--def", &dir.join("f1.png")),
            ("--roi", &dir.join("tiny.json")),
            ("--out", &out),
        ],
        &["--crop-margin", "0"],
    );
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).contains("error: the crop 19x19 is smaller than poc_win (64)"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn pair_warns_when_the_material_moves_farther_than_the_margin() {
    let dir = scratch("crop_warn_pair");
    write_frames(&dir, 384, 288, 5, 2, 30.0);
    write_roi(&dir.join("roi.json"), &STRIP);
    let out = dir.join("out");
    let base = ["--set", "outer_iters=2", "--set", "max_disp=64"];
    let with = |extra: &[&str]| pair(&dir, &out, &[&base[..], extra].concat());
    let o = with(&["--crop-margin", "20"]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_eq!(stderr(&o).matches(WARNING).count(), 1, "{}", stderr(&o));
    for extra in [&["--crop", "off"][..], &[]] {
        let o = with(extra);
        assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
        assert!(
            !stderr(&o).contains("crop border"),
            "{extra:?}: {}",
            stderr(&o)
        );
    }
}

#[test]
fn seq_warns_once_when_the_material_moves_farther_than_the_margin() {
    let dir = scratch("crop_warn_seq");
    write_frames(&dir, 384, 288, 5, 4, 20.0);
    write_roi(&dir.join("roi.json"), &STRIP);
    let out = dir.join("out");
    let o = seq(
        &dir,
        &out,
        &[
            "--set",
            "outer_iters=2",
            "--set",
            "max_disp=64",
            "--crop-margin",
            "20",
        ],
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_eq!(stderr(&o).matches(WARNING).count(), 1, "{}", stderr(&o));
    assert!(out.join("frame_00003/ux.npy").exists());
}

#[test]
fn seq_crop_writes_full_size_frames_that_match_crop_off() {
    let dir = scratch("crop_seq");
    write_frames(&dir, 512, 384, 5, 3, 1.5);
    let poly = [
        (200.0, 150.0),
        (300.0, 150.0),
        (300.0, 240.0),
        (200.0, 240.0),
    ];
    write_roi(&dir.join("roi.json"), &poly);
    let (auto, off) = (dir.join("auto"), dir.join("off"));
    let base = ["--set", "outer_iters=2"];
    let o = seq(
        &dir,
        &auto,
        &[&base[..], &["--crop-margin", "100"]].concat(),
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(!stderr(&o).contains("crop border"), "{}", stderr(&o));
    let o = seq(&dir, &off, &[&base[..], &["--crop", "off"]].concat());
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    // x: 200 - 100 = 100 -> 96; y: 150 - 100 = 50 -> 32; x1 = 301 + 100, y1 = 241 + 100
    let want = "{\n  \"x0\": 96,\n  \"y0\": 32,\n  \"x1\": 401,\n  \"y1\": 341,\n  \"margin\": 100,\n  \"mode\": \"auto\"\n}";
    assert_eq!(text(&auto.join("crop.json")), want);
    assert_eq!(text(&off.join("crop.json")), OFF_JSON);
    let roi = roi_mask(512, 384, &poly);
    for k in [1, 2] {
        let (a, b) = (
            auto.join(format!("frame_{k:05}")),
            off.join(format!("frame_{k:05}")),
        );
        assert_auto_matches_off(&a, &b, &roi, (384, 512));
        let (_, ux) = field(&a, "ux");
        let median = median_of_finite((0..512 * 384).filter(|&i| roi[i]).map(|i| ux[i]));
        assert!(
            (median - 1.5 * k as f64).abs() < 0.05,
            "frame {k}: {median}"
        );
    }
    // the default margin of a sequence (2 * 189) reaches the image border on every side
    let dflt = dir.join("dflt");
    let o = seq(&dir, &dflt, &base);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let want = "{\n  \"x0\": 0,\n  \"y0\": 0,\n  \"x1\": 512,\n  \"y1\": 384,\n  \"margin\": 378,\n  \"mode\": \"auto\"\n}";
    assert_eq!(text(&dflt.join("crop.json")), want);
}

#[test]
fn a_frame_of_another_size_is_reported_as_without_a_crop() {
    let dir = scratch("crop_shape");
    write_frames(&dir, 192, 160, 3, 2, 1.0);
    let other = make_pattern(200, 160, 3);
    write_png16(
        &dir.join("f2.png"),
        200,
        160,
        &render(&other, 200, 160, 0.0, 0.0),
    );
    write_roi(
        &dir.join("roi.json"),
        &[(40.0, 40.0), (150.0, 40.0), (150.0, 120.0), (40.0, 120.0)],
    );
    let out = dir.join("out");
    let off = seq(&dir, &out, &["--set", "outer_iters=2", "--crop", "off"]);
    let auto = seq(&dir, &out, &["--set", "outer_iters=2"]);
    assert_eq!((off.status.code(), auto.status.code()), (Some(2), Some(2)));
    let last = |o: &Output| stderr(o).lines().last().unwrap().to_string();
    assert_eq!(
        last(&auto),
        "error: frame 2 has shape (160, 200), expected (160, 192)"
    );
    assert_eq!(last(&auto), last(&off));
}

/// The number of the frame in the error counts frames from the third one on: the fourth frame of
/// another size is "frame 3", whatever the crop.
#[test]
fn the_frame_number_of_a_shape_error_follows_the_sequence() {
    let dir = scratch("crop_shape_late");
    write_frames(&dir, 192, 160, 3, 3, 1.0);
    let other = make_pattern(200, 160, 3);
    write_png16(
        &dir.join("f3.png"),
        200,
        160,
        &render(&other, 200, 160, 0.0, 0.0),
    );
    write_roi(
        &dir.join("roi.json"),
        &[(40.0, 40.0), (150.0, 40.0), (150.0, 120.0), (40.0, 120.0)],
    );
    // with the crop (the frames are cut by the iterator that counts them) and without
    for (what, crop) in [("auto", "auto"), ("off", "off")] {
        let out = dir.join(format!("out_{what}"));
        let o = seq(&dir, &out, &["--set", "outer_iters=2", "--crop", crop]);
        assert_eq!(o.status.code(), Some(2), "{what}: {}", stderr(&o));
        let last = stderr(&o).lines().last().unwrap().to_string();
        assert_eq!(
            last, "error: frame 3 has shape (160, 200), expected (160, 192)",
            "--crop {what}"
        );
    }
}

//! End-to-end tests of the `optflow` binary: `pair` and `seq` on synthetic images with a known
//! translation, the output files, the error contract (`error: ...` on stderr, status 2) and video
//! input through ffmpeg (skipped when ffmpeg is not installed).

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;
use optflow_core::config::Config;

/// `n` frames of a pattern moving by `(dx, dy)` px per frame, as 16-bit PNGs `f0.png`, `f1.png`...
fn write_frames(dir: &Path, w: usize, h: usize, seed: u64, n: usize, (dx, dy): (f64, f64)) {
    let p = make_pattern(w, h, seed);
    for k in 0..n {
        let img = render(&p, w, h, dx * k as f64, dy * k as f64);
        write_png16(&dir.join(format!("f{k}.png")), w, h, &img);
    }
}

/// The coarse stage of a 96 x 96 image covers +-16 px: with this `max_disp` a run of the tests'
/// 96 x 96 frames has no capture-range warning on stderr (the default 32 would add one), which the
/// tests that look at all of stderr need.
const NO_CAPTURE_WARNING: [&str; 2] = ["--set", "max_disp=16"];

/// `args` and the `--set` of [`NO_CAPTURE_WARNING`], for the runs whose whole stderr is checked.
fn quiet<'a>(args: &[&'a str]) -> Vec<&'a str> {
    let mut all = args.to_vec();
    all.extend(NO_CAPTURE_WARNING);
    all
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn pair_tracks_a_translation_and_leaves_the_roi_exterior_nan() {
    let dir = scratch("pair_translation");
    let (w, h) = (160, 160);
    let p = make_pattern(w, h, 2);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 1.25, -0.5));
    std::fs::write(
        dir.join("roi.json"),
        r#"{"polygon": [[20, 20], [140, 20], [140, 140], [20, 140]]}"#,
    )
    .unwrap();
    let out = dir.join("out");
    let o = optflow(&[
        "pair",
        "--ref",
        s(&dir.join("a.png")),
        "--def",
        s(&dir.join("b.png")),
        "--roi",
        s(&dir.join("roi.json")),
        "--out",
        s(&out),
        "--set",
        "outer_iters=4",
        "--png",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(
        stdout(&o).starts_with("done: {'coarse': "),
        "{}",
        stdout(&o)
    );
    let err = stderr(&o);
    assert!(err.contains("warning") && err.contains("--png"), "{err}");
    assert!(!out.join("maps.png").exists());

    for name in [
        "ux", "uy", "conf", "Exx", "Eyy", "Exy", "gmax", "vol", "valid",
    ] {
        assert!(out.join(format!("{name}.npy")).exists(), "{name}.npy");
    }
    let (shape, ux) = load_npy(&out.join("ux.npy"));
    assert_eq!(shape, [h, w]);
    assert!(ux[5 * w + 5].is_nan(), "outside the ROI");
    let (_, uy) = load_npy(&out.join("uy.npy"));
    assert!((median_of_finite(ux.iter().copied()) - 1.25).abs() < 0.01);
    assert!((median_of_finite(uy.iter().copied()) + 0.5).abs() < 0.01);
    let (vshape, valid) = load_npy(&out.join("valid.npy"));
    assert_eq!(vshape, [h, w]);
    assert!(valid.contains(&1.0));
    for (v, u) in valid.iter().zip(&ux) {
        assert_eq!(*v == 1.0, u.is_finite());
    }
    let raw = std::fs::read(out.join("valid.npy")).unwrap();
    assert!(
        raw.windows(8).any(|w| w == b"'|u1', '"),
        "valid.npy is uint8"
    );

    let timing: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("timing.json")).unwrap()).unwrap();
    let mut keys: Vec<&str> = timing
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["coarse", "fine", "strain"]);
    let cfg = Config {
        outer_iters: 4,
        ..Config::default()
    };
    assert_eq!(
        std::fs::read_to_string(out.join("config.json")).unwrap(),
        cfg.to_json()
    );
}

#[test]
fn config_json_is_written_as_python_writes_it() {
    // the images are missing, so the run fails after config.json is written
    let dir = scratch("config_json");
    let o = optflow(&[
        "pair",
        "--ref",
        s(&dir.join("x.png")),
        "--def",
        s(&dir.join("y.png")),
        "--out",
        s(&dir.join("o")),
        "--set",
        "sigma_w=6",
        "--set",
        "scales=0.8,1.2",
        "--set",
        "outer_iters=4",
        "--set",
        "edge_guard_min_area=1e-7",
        "--set",
        "lam=9.5",
    ]);
    assert_eq!(o.status.code(), Some(2));
    let want = "{\n  \"grid_step\": 16,\n  \"poc_win\": 64,\n  \"max_disp\": 32.0,\n  \"poc_peak_min\": 0.1,\n  \"median_thresh\": 2.0,\n  \"lam\": 9.5,\n  \"scales\": [\n    0.8,\n    1.2\n  ],\n  \"n_orient\": 4,\n  \"bandwidth\": 0.5,\n  \"dc_cut\": 2.0,\n  \"sigma_w\": 6.0,\n  \"shape_order\": 1,\n  \"sigma_s\": 4.0,\n  \"outer_iters\": 4,\n  \"inner_iters\": 2,\n  \"conf_min\": 0.2,\n  \"edge_guard\": -1.0,\n  \"edge_guard_min_area\": 1e-07,\n  \"phase_tol\": 0.4,\n  \"strain_win\": 15,\n  \"ref_update_conf\": 0.5,\n  \"seq_mode\": \"chain\",\n  \"fb_max\": 10,\n  \"fb_radius\": -1.0,\n  \"fb_min_support\": 20,\n  \"fb_tol\": 0.5,\n  \"tavg\": 0,\n  \"tavg_align\": false,\n  \"tavg_model\": \"affine\",\n  \"prewarp_freeze_at\": 0\n}";
    assert_eq!(
        std::fs::read_to_string(dir.join("o/config.json")).unwrap(),
        want
    );
}

#[test]
fn scale_two_box_averages_the_images_first() {
    let dir = scratch("pair_scale2");
    let (w, h) = (320, 320);
    let p = make_pattern(w, h, 5);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 2.5, 0.0));
    let out = dir.join("out");
    let o = optflow(&[
        "pair",
        "--ref",
        s(&dir.join("a.png")),
        "--def",
        s(&dir.join("b.png")),
        "--out",
        s(&out),
        "--scale",
        "2",
        "--set",
        "outer_iters=4",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let (shape, ux) = load_npy(&out.join("ux.npy"));
    assert_eq!(shape, [160, 160]);
    assert!((median_of_finite(ux.iter().copied()) - 1.25).abs() < 0.02);
}

#[test]
fn seq_tracks_a_directory_of_frames_back_to_frame_zero() {
    let dir = scratch("seq_dir");
    write_frames(&dir, 160, 160, 3, 3, (0.8, 0.4));
    let out = dir.join("out");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir),
        "--out",
        s(&out),
        "--set",
        "outer_iters=4",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let text = stdout(&o);
    assert!(text.contains("frame 1: reference 0 mean conf "), "{text}");
    assert!(text.contains("frame 2: reference "), "{text}");
    for (k, want) in [(1, 0.8), (2, 1.6)] {
        let (shape, ux) = load_npy(&out.join(format!("frame_{k:05}/ux.npy")));
        assert_eq!(shape, [160, 160]);
        assert!(
            (centre_median(&ux, 160, 40, 120) - want).abs() < 0.05,
            "frame {k}"
        );
        assert!(out.join(format!("frame_{k:05}/timing.json")).exists());
    }
    let timing = std::fs::read_to_string(out.join("frame_00002/timing.json")).unwrap();
    assert!(
        timing.contains("\"reference\"") && timing.contains("\"mean_conf\""),
        "{timing}"
    );
    assert!(out.join("config.json").exists());
}

#[test]
fn seq_start_stop_and_step_select_frames() {
    let dir = scratch("seq_range");
    write_frames(&dir, 128, 128, 4, 5, (0.5, 0.0));
    let run = |extra: &[&str], out: &str| {
        let out = dir.join(out);
        let mut args = vec![
            "seq",
            "--frames",
            s(&dir),
            "--out",
            s(&out),
            "--set",
            "outer_iters=2",
        ];
        args.extend_from_slice(extra);
        let o = optflow(&args);
        (o, out)
    };
    // frames 1, 3: the second is 2 x 0.5 = 1 px from the first
    let (o, out) = run(&["--start", "1", "--step", "2"], "a");
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let (_, ux) = load_npy(&out.join("frame_00001/ux.npy"));
    assert!((median_of_finite(ux.iter().copied()) - 1.0).abs() < 0.05);
    assert!(!out.join("frame_00002").exists());
    // a negative stop counts from the end of the file list, as in Python: f0, f1, f2, f3
    let (o, out) = run(&["--stop=-1"], "b");
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(out.join("frame_00003").exists() && !out.join("frame_00004").exists());
}

#[test]
fn every_failure_is_an_error_line_and_status_2() {
    let dir = scratch("errors");
    let (w, h) = (96, 96);
    let p = make_pattern(w, h, 2);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 0.5, 0.2));
    write_png16(
        &dir.join("wide.png"),
        128,
        h,
        &render(&make_pattern(128, h, 2), 128, h, 0.0, 0.0),
    );
    write_png16(
        &dir.join("tiny.png"),
        32,
        32,
        &render(&make_pattern(32, 32, 2), 32, 32, 0.0, 0.0),
    );
    std::fs::write(dir.join("bad.png"), b"not an image").unwrap();
    // a header that claims 13 GB of pixels
    std::fs::write(dir.join("bomb.bmp"), bomb_bmp(65_535, 65_535)).unwrap();
    // grey float32: tifffile reads it, this tool does not
    let gray_f32 = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/images/gray_f32.tif"
    );
    std::fs::write(dir.join("cfg_bad.json"), r#"{"grid_step": 2.5}"#).unwrap();
    std::fs::write(dir.join("roi_nokey.json"), r#"{"outline": []}"#).unwrap();
    std::fs::write(
        dir.join("roi_far.json"),
        r#"{"polygon": [[500, 500], [600, 500], [600, 600]]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("roi_huge.json"),
        r#"{"polygon": [[0, 0], [1e10, 5], [0, 9]]}"#,
    )
    .unwrap();
    let (a, b) = (
        s(&dir.join("a.png")).to_string(),
        s(&dir.join("b.png")).to_string(),
    );
    let out = s(&dir.join("out")).to_string();
    let pair = |extra: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = ["pair", "--ref", &a, "--def", &b, "--out", &out]
            .map(String::from)
            .to_vec();
        v.extend(extra.iter().map(|x| x.to_string()));
        v
    };
    let with = |p: &str| dir.join(p).to_str().unwrap().to_string();
    let cases: Vec<(Vec<String>, &str)> = vec![
        (vec![], "the following arguments are required: cmd"),
        (vec!["fly".into()], "invalid choice: 'fly'"),
        (
            vec!["pair".into(), "--out".into(), out.clone()],
            "the following arguments are required: --ref, --def",
        ),
        (
            pair(&["--scale", "x"]),
            "argument --scale: invalid int value: 'x'",
        ),
        (
            pair(&["--tile", "512"]),
            "unrecognized arguments: --tile 512",
        ),
        (
            pair(&["--set", "sigma_x=3"]),
            "unknown config key 'sigma_x'; valid keys: grid_step, poc_win",
        ),
        (
            pair(&["--set", "outer_iters"]),
            "--set expects key=value (e.g. --set sigma_w=6), got 'outer_iters'",
        ),
        (
            pair(&["--set", "outer_iters=2.5"]),
            "--set outer_iters: cannot read '2.5' as int",
        ),
        (
            pair(&["--set", "outer_iters=2.0"]),
            "--set outer_iters: cannot read '2.0' as int",
        ),
        (
            pair(&["--set", "sigma_w=abc"]),
            "--set sigma_w: cannot read 'abc' as float",
        ),
        (
            pair(&["--set", "scales=0.6,x"]),
            "cannot read '0.6,x' as a comma-separated list of numbers",
        ),
        (pair(&["--set", "lam=nan"]), "finite"),
        (
            pair(&["--set", "conf_min=2"]),
            "conf_min and poc_peak_min must lie in [0, 1]",
        ),
        (pair(&["--config", &with("cfg_bad.json")]), "grid_step"),
        (pair(&["--config", &with("no_cfg.json")]), "no_cfg.json"),
        (pair(&["--scale", "0"]), "scale must be >= 1, got 0"),
        (pair(&["--scale", "-2"]), "scale must be >= 1, got -2"),
        (pair(&["--roi", &with("no_roi.json")]), "no_roi.json"),
        (pair(&["--roi", &with("roi_nokey.json")]), "polygon"),
        (pair(&["--roi", &with("roi_far.json")]), "ROI mask is empty"),
        (
            pair(&["--roi", &with("roi_huge.json")]),
            "roi_huge.json: vertex [10000000000.0,5] is out of range",
        ),
        (
            pair(&["--set", "grid_step=99999999999999999999"]),
            "--set grid_step: '99999999999999999999' is out of range",
        ),
        (
            pair(&[
                "--set",
                "grid_step=999999999999999999999999999999999999999999x",
            ]),
            "--set grid_step: cannot read '999999999999999999999999999999999999999999x' as int",
        ),
        (
            vec![
                "pair".into(),
                "--ref".into(),
                with("missing.png"),
                "--def".into(),
                b.clone(),
                "--out".into(),
                out.clone(),
            ],
            "missing.png",
        ),
        (
            vec![
                "pair".into(),
                "--ref".into(),
                a.clone(),
                "--def".into(),
                with("bad.png"),
                "--out".into(),
                out.clone(),
            ],
            "bad.png",
        ),
        (
            vec![
                "pair".into(),
                "--ref".into(),
                with("bomb.bmp"),
                "--def".into(),
                b.clone(),
                "--out".into(),
                out.clone(),
            ],
            "bomb.bmp: image too large to read",
        ),
        (
            vec![
                "pair".into(),
                "--ref".into(),
                a.clone(),
                "--def".into(),
                gray_f32.into(),
                "--out".into(),
                out.clone(),
            ],
            "gray_f32.tif: unsupported pixel type",
        ),
        (
            vec![
                "pair".into(),
                "--ref".into(),
                a.clone(),
                "--def".into(),
                with("wide.png"),
                "--out".into(),
                out.clone(),
            ],
            "image sizes differ: (96, 96) vs (96, 128)",
        ),
        (
            vec![
                "pair".into(),
                "--ref".into(),
                with("tiny.png"),
                "--def".into(),
                with("tiny.png"),
                "--out".into(),
                out.clone(),
            ],
            "image smaller than poc_win (64): (32, 32)",
        ),
        (
            vec![
                "seq".into(),
                "--frames".into(),
                with("nothing_*.png"),
                "--out".into(),
                out.clone(),
            ],
            "no images found for",
        ),
        (
            vec![
                "seq".into(),
                "--frames".into(),
                s(&dir).into(),
                "--step".into(),
                "0".into(),
                "--out".into(),
                out.clone(),
            ],
            "step must be >= 1, got 0",
        ),
        (
            vec![
                "seq".into(),
                "--frames".into(),
                s(&dir).into(),
                "--start=-1".into(),
                "--out".into(),
                out.clone(),
            ],
            "start must be >= 0, got -1",
        ),
        (
            vec![
                "seq".into(),
                "--frames".into(),
                s(&dir).into(),
                "--start".into(),
                "50".into(),
                "--out".into(),
                out.clone(),
            ],
            "no frames to track in",
        ),
        (
            vec![
                "seq".into(),
                "--frames".into(),
                with("a.png"),
                "--out".into(),
                out.clone(),
            ],
            "need at least two frames to track, got 1",
        ),
        (
            vec![
                "seq".into(),
                "--frames".into(),
                with("missing.mp4"),
                "--out".into(),
                out.clone(),
            ],
            "missing.mp4",
        ),
    ];
    for (args, want) in cases {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let o = optflow(&refs);
        let err = stderr(&o);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {err}");
        assert!(err.contains("error: "), "{args:?}: {err}");
        assert!(err.contains(want), "{args:?}: wanted {want:?} in {err}");
        assert!(
            !err.contains("panicked") && !err.contains("backtrace"),
            "{args:?}: {err}"
        );
    }
}

/// What `validate` says about `grid_step` and the edge guard, as the command line prints it.
const GUARD_ERROR: &str = "error: grid_step <= 4096 and an edge guard (edge_guard, or the automatic one from lam, scales and bandwidth) <= 256 px required\n";

#[test]
fn an_absurd_grid_step_with_a_roi_is_a_config_error_not_a_panic() {
    // 2^63 << 1 wrapped to 0 in the crop alignment and divided by zero
    let dir = scratch("absurd_grid_step");
    write_frames(&dir, 128, 128, 2, 3, (0.5, 0.2));
    std::fs::write(
        dir.join("roi.json"),
        r#"{"polygon": [[20, 20], [100, 20], [100, 100], [20, 100]]}"#,
    )
    .unwrap();
    let out = dir.join("out");
    let (f0, f1, roi) = (dir.join("f0.png"), dir.join("f1.png"), dir.join("roi.json"));
    for grid_step in [
        "9223372036854775808",
        "18446744073709551615",
        "4611686018427387904",
        "4097",
    ] {
        let set = format!("grid_step={grid_step}");
        let pair = [
            "pair",
            "--ref",
            s(&f0),
            "--def",
            s(&f1),
            "--roi",
            s(&roi),
            "--out",
            s(&out),
            "--set",
            &set,
        ];
        let seq = [
            "seq",
            "--frames",
            s(&dir),
            "--roi",
            s(&roi),
            "--out",
            s(&out),
            "--set",
            &set,
        ];
        for args in [&pair[..], &seq[..]] {
            let o = optflow(args);
            assert_eq!(o.status.code(), Some(2), "{args:?}: {}", stderr(&o));
            assert_eq!(stderr(&o), GUARD_ERROR, "{args:?}");
        }
    }
}

#[test]
fn seq_stops_with_an_error_when_a_later_frame_cannot_be_read() {
    let dir = scratch("seq_bad_frame");
    write_frames(&dir, 96, 96, 2, 2, (0.5, 0.0));
    std::fs::write(dir.join("f2.png"), b"not an image").unwrap();
    let out = dir.join("out");
    let o = optflow(&quiet(&[
        "seq",
        "--frames",
        s(&dir),
        "--out",
        s(&out),
        "--set",
        "outer_iters=2",
    ]));
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).starts_with("error: ") && stderr(&o).contains("f2.png"),
        "{}",
        stderr(&o)
    );
    assert!(
        out.join("frame_00001/ux.npy").exists(),
        "frames before the failure are kept"
    );
}

#[test]
fn help_prints_the_usage_and_succeeds() {
    for args in [
        &["--help"][..],
        &["pair", "-h"],
        &["seq", "--frames", "x", "--help"],
    ] {
        let o = optflow(args);
        assert_eq!(o.status.code(), Some(0));
        assert!(stdout(&o).contains("usage: optflow {pair,seq}"));
    }
}

/// Encodes `n` frames of a pattern moving `(dx, dy)` per frame as an MJPG AVI.
fn make_mjpg_video(dir: &Path, n: usize, delta: (f64, f64)) -> PathBuf {
    make_mjpg_video_sized(dir, 160, 160, n, delta)
}

fn make_mjpg_video_sized(
    dir: &Path,
    w: usize,
    h: usize,
    n: usize,
    (dx, dy): (f64, f64),
) -> PathBuf {
    let p = make_pattern(w, h, 3);
    let frames = dir.join("frames");
    std::fs::create_dir_all(&frames).unwrap();
    for k in 0..n {
        write_png8(
            &frames.join(format!("g{}.png", k + 1)),
            w,
            h,
            &render(&p, w, h, dx * k as f64, dy * k as f64),
        );
    }
    let video = dir.join("s.avi");
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-framerate", "10", "-i"])
        .arg(frames.join("g%d.png"))
        .args(["-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj420p"])
        .arg(&video)
        .status()
        .unwrap();
    assert!(st.success());
    video
}

#[test]
fn seq_reads_an_mjpg_video_through_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    }
    let dir = scratch("seq_video");
    let video = make_mjpg_video(&dir, 3, (0.8, 0.4));
    let out = dir.join("out");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&video),
        "--out",
        s(&out),
        "--set",
        "outer_iters=4",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let (shape, ux) = load_npy(&out.join("frame_00002/ux.npy"));
    assert_eq!(shape, [160, 160]);
    assert!((centre_median(&ux, 160, 40, 120) - 1.6).abs() < 0.05);
    // --start / --stop work on frame indices of the video, as for images
    let out2 = dir.join("out2");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&video),
        "--start",
        "1",
        "--out",
        s(&out2),
        "--set",
        "outer_iters=2",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(out2.join("frame_00001").exists() && !out2.join("frame_00002").exists());
    let o = optflow(&[
        "seq",
        "--frames",
        s(&video),
        "--stop",
        "1",
        "--out",
        s(&dir.join("out3")),
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("need at least two frames to track, got 1"));
}

/// The first frame of a video, as the tracker would read it.
fn first_frame(video: &Path) -> optflow_cli::io::Gray {
    optflow_cli::video::iter_video(video, 0, None, 1)
        .unwrap()
        .next()
        .expect("a frame")
        .unwrap()
}

#[test]
fn a_rotated_video_has_the_pixels_of_the_unrotated_one_turned_upright() {
    // The turn is made by explicit filters and must be the one OpenCV (the Python reference) makes:
    // for an MJPEG clip made with `-display_rotation` the frames are exact permutations of the
    // unrotated clip's (even sizes: no chroma siting changes). Checked once against cv2 by hand: the
    // same clips give OpenCV's frames bit for bit.
    if !have_ffmpeg() {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    }
    let dir = scratch("video_rotations");
    let (w, h) = (128, 80);
    let video = make_mjpg_video_sized(&dir, w, h, 2, (0.8, 0.0));
    let base = first_frame(&video);
    assert_eq!((base.w, base.h), (w, h));
    // upright[y][x] of a w' x h' frame, from the stored frame
    let counterclockwise = |x: usize, y: usize| base.data[x * w + (w - 1 - y)];
    let clockwise = |x: usize, y: usize| base.data[(h - 1 - x) * w + y];
    let half = |x: usize, y: usize| base.data[(h - 1 - y) * w + (w - 1 - x)];
    let same = |x: usize, y: usize| base.data[y * w + x];
    type Pick<'a> = &'a dyn Fn(usize, usize) -> f64;
    let cases: [(&str, (usize, usize), Pick); 6] = [
        ("0", (w, h), &same),
        ("90", (h, w), &counterclockwise),
        ("-90", (h, w), &clockwise),
        ("270", (h, w), &clockwise),
        ("180", (w, h), &half),
        // not a quarter turn: OpenCV leaves the frames as they are
        ("45", (w, h), &same),
    ];
    for (rotation, (uw, uh), pick) in cases {
        let rotated = dir.join(format!("rot{rotation}.mov"));
        let st = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-display_rotation", rotation, "-i"])
            .arg(&video)
            .args(["-c", "copy"])
            .arg(&rotated)
            .status()
            .unwrap();
        assert!(st.success(), "{rotation}");
        let g = first_frame(&rotated);
        assert_eq!((g.w, g.h), (uw, uh), "{rotation}: size");
        let want: Vec<f64> = (0..uh)
            .flat_map(|y| (0..uw).map(move |x| (x, y)))
            .map(|(x, y)| pick(x, y))
            .collect();
        assert!(
            g.data == want,
            "{rotation}: the pixels are not the turned stored frame"
        );
    }
}

#[test]
fn a_rotated_video_is_read_upright() {
    if !have_ffmpeg() {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    }
    let dir = scratch("seq_rotated");
    let video = make_mjpg_video_sized(&dir, 128, 80, 3, (0.8, 0.0));
    let rotated = dir.join("rot.mov");
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-display_rotation", "90", "-i"])
        .arg(&video)
        .args(["-c", "copy"])
        .arg(&rotated)
        .status()
        .unwrap();
    assert!(st.success());
    let out = dir.join("out");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&rotated),
        "--out",
        s(&out),
        "--set",
        "outer_iters=2",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let (shape, _) = load_npy(&out.join("frame_00001/ux.npy"));
    assert_eq!(shape, [128, 80], "rows x columns of the upright frames");
}

#[test]
fn a_file_that_is_not_a_video_is_an_error() {
    if !have_ffmpeg() {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    }
    let dir = scratch("seq_notvideo");
    std::fs::write(dir.join("junk.mp4"), b"this is not a video").unwrap();
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir.join("junk.mp4")),
        "--out",
        s(&dir.join("o")),
    ]);
    assert_eq!(o.status.code(), Some(2));
    let err = stderr(&o);
    assert!(
        err.starts_with("error: ") && err.contains("junk.mp4"),
        "{err}"
    );
}

#[test]
fn missing_ffmpeg_is_reported_clearly() {
    let dir = scratch("seq_no_ffmpeg");
    std::fs::write(dir.join("v.mp4"), b"x").unwrap();
    let empty = dir.join("empty_path");
    std::fs::create_dir_all(&empty).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_optflow"))
        .env("PATH", &empty)
        .args([
            "seq",
            "--frames",
            s(&dir.join("v.mp4")),
            "--out",
            s(&dir.join("o")),
        ])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert_eq!(
        stderr(&o),
        "error: ffmpeg not found on PATH (needed for video input)\n"
    );
}

#[test]
fn seq_png_warns_and_writes_no_maps() {
    let dir = scratch("seq_png");
    write_frames(&dir, 96, 96, 2, 2, (0.5, 0.0));
    let out = dir.join("out");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir),
        "--out",
        s(&out),
        "--set",
        "outer_iters=2",
        "--png",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let err = stderr(&o);
    assert!(err.contains("warning") && err.contains("--png"), "{err}");
    assert!(out.join("frame_00001/ux.npy").exists());
    assert!(!out.join("maps.png").exists() && !out.join("frame_00001/maps.png").exists());
    // without the flag there is no warning
    let o = optflow(&quiet(&[
        "seq",
        "--frames",
        s(&dir),
        "--out",
        s(&dir.join("out2")),
        "--set",
        "outer_iters=2",
    ]));
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_eq!(stderr(&o), "");
}

#[cfg(unix)]
#[test]
fn an_argument_that_is_not_utf8_is_an_error_line() {
    use std::os::unix::ffi::OsStrExt;
    let bad = std::ffi::OsStr::from_bytes(b"x\xff.png");
    let o = Command::new(env!("CARGO_BIN_EXE_optflow"))
        .args(["pair", "--ref"])
        .arg(bad)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let err = stderr(&o);
    assert!(
        err.starts_with("error: argument 3 is not valid UTF-8: ") && err.contains("x\\xFF.png"),
        "{err}"
    );
    assert_eq!(err.lines().count(), 1, "{err}");
}

/// Runs `optflow` with stdout connected to a pipe whose reading end is already closed.
#[cfg(unix)]
fn optflow_with_closed_stdout(args: &[&str]) -> std::process::Output {
    let (reader, writer) = std::io::pipe().unwrap();
    drop(reader);
    Command::new(env!("CARGO_BIN_EXE_optflow"))
        .args(args)
        .stdout(writer)
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap()
}

#[cfg(unix)]
#[test]
fn a_closed_stdout_ends_the_run_quietly() {
    // `optflow ... | head -1`: no panic message, the status of a process killed by SIGPIPE
    let o = optflow_with_closed_stdout(&["--help"]);
    assert_eq!(o.status.code(), Some(141), "{}", stderr(&o));
    assert_eq!(stderr(&o), "");

    // the results are written before the line that finds the pipe closed
    let dir = scratch("closed_stdout");
    let (w, h) = (96, 96);
    let p = make_pattern(w, h, 2);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 0.5, 0.2));
    let out = dir.join("out");
    let o = optflow_with_closed_stdout(&quiet(&[
        "pair",
        "--ref",
        s(&dir.join("a.png")),
        "--def",
        s(&dir.join("b.png")),
        "--out",
        s(&out),
        "--set",
        "outer_iters=2",
    ]));
    assert_eq!(o.status.code(), Some(141), "{}", stderr(&o));
    assert_eq!(stderr(&o), "");
    assert!(out.join("ux.npy").exists());
}

/// Shell scripts that stand in for `ffprobe` (a `w` x `h` stream, with the display rotation
/// `$FAKE_ROTATION` if that is set) and `ffmpeg` in a directory for the PATH of a run.
/// `ffmpeg -version` reports `$FAKE_VERSION`; a decoding run records its arguments in `$FAKE_LOG`,
/// writes the raw RGB24 frames in `$FAKE_FRAMES` and exits with `$FAKE_STATUS` (after a message on
/// stderr if that is not 0). With `$FAKE_PIDFILE` set it writes its process id there first, with
/// `$FAKE_LINGER` set it then stays alive that many seconds after the last frame (as ffmpeg does
/// while the pipe is full).
#[cfg(unix)]
fn fake_ffmpeg_dir(dir: &Path, w: usize, h: usize) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let script = |name: &str, body: &str| {
        let path = bin.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    };
    script(
        "ffprobe",
        &format!(
            "printf 'width={w}\\nheight={h}\\n'\n\
             if [ -n \"$FAKE_ROTATION\" ]; then printf 'rotation=%s\\n' \"$FAKE_ROTATION\"; fi\n"
        ),
    );
    script(
        "ffmpeg",
        r#"if [ "$1" = "-version" ]; then
  echo "ffmpeg version $FAKE_VERSION Copyright (c) the FFmpeg developers"
  exit 0
fi
printf '%s\n' "$@" > "$FAKE_LOG"
if [ -n "$FAKE_PIDFILE" ]; then echo $$ > "$FAKE_PIDFILE"; fi
cat "$FAKE_FRAMES"
if [ -n "$FAKE_LINGER" ]; then exec sleep "$FAKE_LINGER"; fi
if [ "$FAKE_STATUS" != 0 ]; then echo "boom: decoder crashed" >&2; fi
exit "$FAKE_STATUS"
"#,
    );
    bin
}

/// `n` frames of a moving pattern as the raw RGB24 bytes ffmpeg would write.
#[cfg(unix)]
fn raw_rgb24_frames(w: usize, h: usize, n: usize, (dx, dy): (f64, f64)) -> Vec<u8> {
    let p = make_pattern(w, h, 3);
    let mut bytes = Vec::new();
    for k in 0..n {
        for v in render(&p, w, h, dx * k as f64, dy * k as f64) {
            let level = (v * 255.0).round() as u8;
            bytes.extend_from_slice(&[level; 3]);
        }
    }
    bytes
}

/// One frame per entry of `shifts`: the pattern moved by that many pixels in x, as raw RGB24.
#[cfg(unix)]
fn raw_rgb24_shifted(w: usize, h: usize, shifts: &[f64]) -> Vec<u8> {
    let p = make_pattern(w, h, 3);
    let mut bytes = Vec::new();
    for &dx in shifts {
        for v in render(&p, w, h, dx, 0.0) {
            let level = (v * 255.0).round() as u8;
            bytes.extend_from_slice(&[level; 3]);
        }
    }
    bytes
}

/// A stream for the fake tools: ffprobe reports `stored` (width, height) with the display rotation
/// `rotation` (none if empty), and ffmpeg delivers `frames` (width, height) frames.
#[cfg(unix)]
struct FakeStream<'a> {
    stored: (usize, usize),
    rotation: &'a str,
    frames: (usize, usize),
}

/// The plain 96 x 96 stream without a rotation.
#[cfg(unix)]
const PLAIN: FakeStream<'static> = FakeStream {
    stored: (96, 96),
    rotation: "",
    frames: (96, 96),
};

/// Runs `seq` on a 96 x 96 video with the fake tools first on the PATH, with the extra arguments
/// `extra`.
#[cfg(unix)]
fn seq_with_fake_ffmpeg(
    dir: &Path,
    version: &str,
    status: &str,
    out: &Path,
    extra: &[&str],
) -> std::process::Output {
    seq_with_fake_stream(dir, version, status, out, extra, &PLAIN)
}

/// As [`seq_with_fake_ffmpeg`], for the stream `stream` (three frames of a pattern moving by half a
/// pixel in x).
#[cfg(unix)]
fn seq_with_fake_stream(
    dir: &Path,
    version: &str,
    status: &str,
    out: &Path,
    extra: &[&str],
    stream: &FakeStream,
) -> std::process::Output {
    let frames = raw_rgb24_frames(stream.frames.0, stream.frames.1, 3, (0.5, 0.0));
    seq_with_fake_run(dir, version, status, out, extra, stream, &frames, &[])
}

/// [`seq_with_fake_ffmpeg`] with the raw frames the fake ffmpeg writes and extra environment
/// variables for it (see [`fake_ffmpeg_dir`]).
#[cfg(unix)]
fn seq_with_fake_frames(
    dir: &Path,
    version: &str,
    status: &str,
    out: &Path,
    extra: &[&str],
    frames: &[u8],
    env: &[(&str, &str)],
) -> std::process::Output {
    seq_with_fake_run(dir, version, status, out, extra, &PLAIN, frames, env)
}

/// The run behind the helpers above: the fake tools for `stream`, the raw `frames` and the extra
/// environment variables `env`.
#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
fn seq_with_fake_run(
    dir: &Path,
    version: &str,
    status: &str,
    out: &Path,
    extra: &[&str],
    stream: &FakeStream,
    frames: &[u8],
    env: &[(&str, &str)],
) -> std::process::Output {
    let rotation = stream.rotation;
    let bin = fake_ffmpeg_dir(dir, stream.stored.0, stream.stored.1);
    std::fs::write(dir.join("frames.raw"), frames).unwrap();
    std::fs::write(dir.join("v.mp4"), b"x").unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_optflow"))
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .env("FAKE_VERSION", version)
            .env("FAKE_ROTATION", rotation)
            .env("FAKE_STATUS", status)
            .env("FAKE_LOG", dir.join("args.log"))
            .env("FAKE_FRAMES", dir.join("frames.raw"))
            .envs(env.iter().copied())
            .args(["seq", "--frames", s(&dir.join("v.mp4")), "--out", s(out)])
            .args(["--set", "outer_iters=2"])
            .args(extra)
            .output()
            .unwrap()
    };
    // a script that was written a moment ago can be "busy" while another test forks
    let mut o = run();
    for _ in 0..5 {
        if !stderr(&o).contains("Text file busy") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        o = run();
    }
    o
}

#[cfg(unix)]
#[test]
fn ffmpeg_failing_mid_stream_is_reported_after_the_frames_before_it() {
    let dir = scratch("seq_ffmpeg_fails");
    let out = dir.join("out");
    let o = seq_with_fake_ffmpeg(&dir, "7.1.1", "1", &out, &NO_CAPTURE_WARNING);
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    assert_eq!(
        stderr(&o),
        "error: ffmpeg failed to decode the video: boom: decoder crashed\n"
    );
    // the three frames that arrived were tracked and saved before the failure was reported
    for k in [1, 2] {
        assert!(
            out.join(format!("frame_{k:05}/ux.npy")).exists(),
            "frame {k}"
        );
    }
    assert!(!out.join("frame_00003").exists());
    assert!(stdout(&o).contains("frame 2: reference "), "{}", stdout(&o));
    let log = std::fs::read_to_string(dir.join("args.log")).unwrap();
    assert!(log.lines().any(|l| l == "-fps_mode"), "{log}");
}

#[cfg(unix)]
#[test]
fn a_rotated_stream_is_turned_by_a_filter_and_ffmpeg_is_told_not_to() {
    // stored 128 x 80 (width x height) with the display rotation of a phone held upright: the
    // frames are 80 x 128, and the one decision (the probe) gives both the size we read and the filter
    for (rotation, filter, frames) in [
        ("-90", "transpose=clock,", (80, 128)),
        ("90", "transpose=cclock,", (80, 128)),
        ("-180", "hflip,vflip,", (128, 80)),
        ("0", "", (128, 80)),
        ("45", "", (128, 80)),
        ("", "", (128, 80)),
    ] {
        let dir = scratch(&format!("seq_ffmpeg_rotation_{rotation}"));
        let out = dir.join("out");
        let o = seq_with_fake_stream(
            &dir,
            "7.1.1",
            "0",
            &out,
            &["--set", "max_disp=16"],
            &FakeStream {
                stored: (128, 80),
                rotation,
                frames,
            },
        );
        assert_eq!(o.status.code(), Some(0), "{rotation}: {}", stderr(&o));
        let (shape, _) = load_npy(&out.join("frame_00002/ux.npy"));
        assert_eq!(shape, [frames.1, frames.0], "{rotation}: rows x columns");
        let log = std::fs::read_to_string(dir.join("args.log")).unwrap();
        let args: Vec<&str> = log.lines().collect();
        let input = args.iter().position(|a| *a == "-i").expect(&log);
        let no_auto = args.iter().position(|a| *a == "-noautorotate").expect(&log);
        assert!(no_auto < input, "an input option goes before -i: {log}");
        let vf = args.iter().position(|a| *a == "-vf").expect(&log);
        assert_eq!(
            args[vf + 1],
            format!("{filter}scale=in_color_matrix=bt601:flags=neighbor+accurate_rnd,format=rgb24"),
            "{rotation}"
        );
    }
}

#[cfg(unix)]
#[test]
fn ffmpeg_before_5_1_gets_vsync_instead_of_fps_mode() {
    let dir = scratch("seq_ffmpeg_old");
    let out = dir.join("out");
    let o = seq_with_fake_ffmpeg(
        &dir,
        "4.4.2-0ubuntu0.22.04.1",
        "0",
        &out,
        &NO_CAPTURE_WARNING,
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(out.join("frame_00002/ux.npy").exists());
    let log = std::fs::read_to_string(dir.join("args.log")).unwrap();
    let args: Vec<&str> = log.lines().collect();
    assert!(!args.contains(&"-fps_mode"), "{log}");
    let at = args.iter().position(|a| *a == "-vsync").expect(&log);
    assert_eq!(args[at + 1], "passthrough");
}

#[cfg(unix)]
#[test]
fn a_video_prints_the_capture_warning_once() {
    // 96 x 96 frames, three of them, so two results: the warning comes with the first only
    let dir = scratch("seq_ffmpeg_capture_warning");
    let out = dir.join("out");
    let o = seq_with_fake_ffmpeg(&dir, "7.1.1", "0", &out, &["--set", "max_disp=64"]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(out.join("frame_00002/ux.npy").exists());
    let err = stderr(&o);
    assert_eq!(err.lines().count(), 1, "{err}");
    assert!(
        err.starts_with(
            "warning: the coarse stage covers about \u{b1}16 px in x and \u{b1}16 px in y, \
             less than max_disp 64;"
        ),
        "{err}"
    );
}

/// `--start` and `--step` of a video keep the frames `start`, `start + step`, ...: with the pattern
/// moved only in frame 4, the pair (1, 4) is 1 px apart, whereas the pair (2, 5), which a step
/// counted from the wrong side of `start` would keep, does not move.
#[cfg(unix)]
#[test]
fn video_start_and_step_keep_every_step_th_frame_from_start() {
    let dir = scratch("seq_ffmpeg_step");
    let out = dir.join("out");
    let frames = raw_rgb24_shifted(96, 96, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let o = seq_with_fake_frames(
        &dir,
        "7.1.1",
        "0",
        &out,
        &["--start", "1", "--step", "3", "--set", "max_disp=16"],
        &frames,
        &[],
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(!out.join("frame_00002").exists(), "{}", stdout(&o));
    let (shape, ux) = load_npy(&out.join("frame_00001/ux.npy"));
    assert_eq!(shape, [96, 96]);
    let moved = centre_median(&ux, 96, 24, 72);
    assert!((moved - 1.0).abs() < 0.1, "median ux {moved}");
}

/// ffmpeg is stopped and reaped as soon as the last wanted frame is read: one that would go on
/// (it has more frames, or waits for its pipe to be read) does not outlive the run.
#[cfg(unix)]
#[test]
fn ffmpeg_is_stopped_when_the_last_wanted_frame_is_read() {
    let dir = scratch("seq_ffmpeg_stopped");
    let out = dir.join("out");
    let pidfile = dir.join("ffmpeg.pid");
    let frames = raw_rgb24_frames(96, 96, 3, (0.5, 0.0));
    let o = seq_with_fake_frames(
        &dir,
        "7.1.1",
        "0",
        &out,
        &["--stop", "2", "--set", "max_disp=16"],
        &frames,
        &[("FAKE_PIDFILE", s(&pidfile)), ("FAKE_LINGER", "30")],
    );
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(out.join("frame_00001/ux.npy").exists() && !out.join("frame_00002").exists());
    let pid = std::fs::read_to_string(&pidfile).unwrap();
    let alive = Command::new("sh")
        .args(["-c", &format!("kill -0 {} 2>/dev/null", pid.trim())])
        .status()
        .unwrap()
        .success();
    if alive {
        // do not leave the sleeping process behind
        let _ = Command::new("kill").arg(pid.trim()).status();
    }
    assert!(!alive, "ffmpeg (pid {}) outlived the run", pid.trim());
}

/// A stream that ffprobe reports with no size is reported as such, by name (the zero sizes are
/// the unit tests of `parse_probe`; this checks that the message reaches the user).
#[cfg(unix)]
#[test]
fn a_video_stream_without_a_size_is_no_video_stream() {
    let dir = scratch("seq_ffprobe_zero");
    let out = dir.join("out");
    let stream = FakeStream {
        stored: (0, 96),
        rotation: "",
        frames: (96, 96),
    };
    let o = seq_with_fake_stream(&dir, "7.1.1", "0", &out, &[], &stream);
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    assert_eq!(
        stderr(&o),
        format!(
            "error: no video stream in {}\n",
            dir.join("v.mp4").display()
        )
    );
}

#[test]
fn seq_hybrid_writes_fb_npy_and_the_hybrid_timing_and_chain_does_not() {
    let dir = scratch("seq_hybrid");
    write_frames(&dir, 160, 160, 3, 3, (0.8, 0.4));
    let run = |extra: &[&str], out: &str| {
        let out = dir.join(out);
        let mut args = vec![
            "seq",
            "--frames",
            s(&dir),
            "--out",
            s(&out),
            "--set",
            "outer_iters=4",
        ];
        args.extend_from_slice(extra);
        let o = optflow(&args);
        assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
        out
    };
    let chain = run(&[], "chain");
    assert!(!chain.join("frame_00001/fb.npy").exists());
    let hybrid = run(&["--set", "seq_mode=hybrid"], "hybrid");
    for k in [1, 2] {
        let bytes = std::fs::read(hybrid.join(format!("frame_{k:05}/fb.npy"))).unwrap();
        let text = String::from_utf8_lossy(&bytes[..128]).to_string();
        assert!(
            text.contains("'<i4'") && text.contains("(160, 160)"),
            "{text}"
        );
        let timing =
            std::fs::read_to_string(hybrid.join(format!("frame_{k:05}/timing.json"))).unwrap();
        for key in [
            "n_fb_max",
            "fallback_frac",
            "fb_gate_rejected",
            "reacquired",
        ] {
            assert!(timing.contains(&format!("\"{key}\"")), "{timing}");
        }
        // a clean translation: everything valid is direct (fb = 0), and the displacement is the chain's
        let fb = &bytes[bytes.len() - 160 * 160 * 4..];
        let centre = (80 * 160 + 80) * 4;
        assert_eq!(&fb[centre..centre + 4], &[0, 0, 0, 0]);
        let (_, a) = load_npy(&chain.join(format!("frame_{k:05}/ux.npy")));
        let (_, b) = load_npy(&hybrid.join(format!("frame_{k:05}/ux.npy")));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            if x.is_finite() && y.is_finite() {
                assert_eq!(x, y);
            }
        }
    }
    let cfg = std::fs::read_to_string(hybrid.join("config.json")).unwrap();
    assert!(cfg.contains("\"seq_mode\": \"hybrid\""), "{cfg}");
}

#[test]
fn seq_tavg_averages_a_window_around_every_sampled_frame() {
    let dir = scratch("seq_tavg");
    write_frames(&dir, 160, 160, 5, 7, (0.8, 0.0));
    let run = |extra: &[&str], out: &str| {
        let out = dir.join(out);
        let mut args = vec![
            "seq",
            "--frames",
            s(&dir),
            "--out",
            s(&out),
            "--start",
            "1",
            "--step",
            "2",
            "--set",
            "outer_iters=4",
        ];
        args.extend_from_slice(extra);
        let o = optflow(&args);
        assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
        out
    };
    let plain = run(&[], "plain");
    let zero = run(&["--set", "tavg=0"], "zero");
    let avg = run(&["--set", "tavg=1"], "avg");
    let aligned = run(&["--set", "tavg=1", "--set", "tavg_align=true"], "aligned");
    for k in [1, 2] {
        let name = format!("frame_{k:05}/ux.npy");
        assert_eq!(
            std::fs::read(plain.join(&name)).unwrap(),
            std::fs::read(zero.join(&name)).unwrap(),
            "tavg=0 is the plain path"
        );
        assert_ne!(
            std::fs::read(plain.join(&name)).unwrap(),
            std::fs::read(avg.join(&name)).unwrap(),
            "tavg=1 changes the frames"
        );
        // the windows (0..2), (2..4), (4..6) are symmetric about the sampled frames 1, 3, 5: the
        // displacement between samples is still 2 x 0.8 per step
        for dir in [&avg, &aligned] {
            let (_, ux) = load_npy(&dir.join(&name));
            let got = centre_median(&ux, 160, 40, 120);
            assert!((got - 1.6 * k as f64).abs() < 0.08, "frame {k}: {got}");
        }
    }
}

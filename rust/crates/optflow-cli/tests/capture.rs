//! The warning of the `optflow` binary when the coarse stage is designed to capture less than
//! `max_disp` (`capture_range` of `optflow-core`): one stderr line per run, exit status 0, about the
//! image actually tracked (after `--scale` and the crop). Python's command line says the same.

mod common;

use std::path::Path;
use std::process::Output;

use common::*;
use optflow_core::poc::capture_range;

const WARNING_128_512: &str = "warning: the coarse stage covers about \u{b1}64 px in x and \u{b1}64 px in y, less than max_disp 128; larger motions may be lost (use a larger image or crop margin, or a smaller max_disp)";

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// `n` square frames `f0.png`, `f1.png`... of a pattern moving by `(1.5, -0.5)` px per frame.
fn write_frames(dir: &Path, size: usize, n: usize) {
    let p = make_pattern(size, size, 2);
    for k in 0..n {
        let img = render(&p, size, size, 1.5 * k as f64, -0.5 * k as f64);
        write_png16(&dir.join(format!("f{k}.png")), size, size, &img);
    }
}

fn pair(dir: &Path, out: &str, extra: &[&str]) -> Output {
    let (f0, f1, out) = (dir.join("f0.png"), dir.join("f1.png"), dir.join(out));
    let mut args = vec![
        "pair",
        "--ref",
        s(&f0),
        "--def",
        s(&f1),
        "--out",
        s(&out),
        "--set",
        "outer_iters=1",
    ];
    args.extend_from_slice(extra);
    optflow(&args)
}

fn warnings(o: &Output) -> Vec<String> {
    stderr(o)
        .lines()
        .filter(|l| l.starts_with("warning"))
        .map(str::to_string)
        .collect()
}

#[test]
fn pair_warns_once_when_max_disp_is_beyond_the_coarse_capture_range() {
    let dir = scratch("capture_pair_512");
    write_frames(&dir, 512, 2);
    assert_eq!(capture_range(128.0, 64, 512, 512), (64.0, 64.0));
    let o = pair(&dir, "out", &["--set", "max_disp=128"]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_eq!(warnings(&o), [WARNING_128_512], "{}", stderr(&o));

    // the defaults sit exactly at the capture range of a 512 x 512 image (L = 1, a 64 px window:
    // +-32 px) on the same pair
    assert_eq!(capture_range(32.0, 64, 512, 512), (32.0, 32.0));
    let o = pair(&dir, "out_default", &[]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(!stderr(&o).contains("warning"), "{}", stderr(&o));
}

#[test]
fn the_warning_is_about_the_image_actually_tracked_after_scale_and_crop() {
    let dir = scratch("capture_scale_crop");
    write_frames(&dir, 200, 2);
    std::fs::write(
        dir.join("roi.json"),
        r#"{"polygon": [[60, 60], [140, 60], [140, 140], [60, 140]]}"#,
    )
    .unwrap();
    let roi = s(&dir.join("roi.json")).to_string();
    let msg = |c: u32| {
        format!("warning: the coarse stage covers about \u{b1}{c} px in x and \u{b1}{c} px in y, less than max_disp 64;")
    };
    let has = |o: &Output, c: u32| {
        assert_eq!(o.status.code(), Some(0), "{}", stderr(o));
        let w = warnings(o);
        assert!(
            w.iter().any(|l| l.starts_with(&msg(c))),
            "want \u{b1}{c}: {w:?}"
        );
    };
    let set = ["--set", "max_disp=64"];
    // the whole 200 x 200 image: L = 2, a ±32 px capture range
    has(&pair(&dir, "a", &set), 32);
    // 100 x 100: a ±16 px capture range
    has(&pair(&dir, "b", &[set[0], set[1], "--scale", "2"]), 16);
    // --roi crops to 151 x 151 (margin 10): a ±16 px capture range; --crop off tracks all 200 x 200
    has(
        &pair(
            &dir,
            "c",
            &[set[0], set[1], "--roi", &roi, "--crop-margin", "10"],
        ),
        16,
    );
    has(
        &pair(&dir, "d", &[set[0], set[1], "--roi", &roi, "--crop", "off"]),
        32,
    );
}

#[test]
fn seq_warns_once_for_the_whole_run() {
    let dir = scratch("capture_seq");
    write_frames(&dir, 96, 4);
    let run = |max_disp: &str| {
        optflow(&[
            "seq",
            "--frames",
            s(&dir),
            "--out",
            s(&dir.join("out")),
            "--set",
            "outer_iters=1",
            "--set",
            max_disp,
        ])
    };
    let o = run("max_disp=64");
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let w = warnings(&o);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("covers about \u{b1}16 px in x and \u{b1}16 px in y, less than max_disp 64;"),
        "{w:?}"
    );
    assert!(dir.join("out/frame_00003/ux.npy").exists());
    // 96 x 96 with max_disp 8: L = 0, a 64 px window = +-16 px
    let o = run("max_disp=8");
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(!stderr(&o).contains("warning"), "{}", stderr(&o));
}

#[test]
fn seq_with_an_roi_crop_warns_once_about_the_crop() {
    // 200 x 200 frames, four of them, so three results; the ROI crops them to 151 x 151 (margin 10):
    // a ±16 px capture range, where the whole frame (L = 2) would have ±32 px. The crop may add its
    // own warning (the material nears its border): only the capture warning is counted.
    let dir = scratch("capture_seq_crop");
    write_frames(&dir, 200, 4);
    std::fs::write(
        dir.join("roi.json"),
        r#"{"polygon": [[60, 60], [140, 60], [140, 140], [60, 140]]}"#,
    )
    .unwrap();
    let out = dir.join("out");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir),
        "--out",
        s(&out),
        "--set",
        "outer_iters=1",
        "--set",
        "max_disp=64",
        "--roi",
        s(&dir.join("roi.json")),
        "--crop-margin",
        "10",
    ]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert!(out.join("frame_00003/ux.npy").exists());
    let coarse: Vec<String> = warnings(&o)
        .into_iter()
        .filter(|l| l.starts_with("warning: the coarse stage"))
        .collect();
    assert_eq!(coarse.len(), 1, "{}", stderr(&o));
    assert!(
        coarse[0]
            .contains("covers about \u{b1}16 px in x and \u{b1}16 px in y, less than max_disp 64;"),
        "{coarse:?}"
    );
}

#[test]
fn a_seq_that_fails_at_frame_2_prints_the_warning_then_the_error() {
    // frame 1 is tracked before the unreadable frame 2 is reached: the warning comes with that
    // result and the error line closes the run, in this order and nothing else
    let dir = scratch("capture_seq_fails_late");
    write_frames(&dir, 96, 2);
    std::fs::write(dir.join("f2.png"), b"not an image").unwrap();
    let out = dir.join("out");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir),
        "--out",
        s(&out),
        "--set",
        "outer_iters=1",
        "--set",
        "max_disp=64",
    ]);
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    assert!(out.join("frame_00001/ux.npy").exists());
    let err = stderr(&o);
    let lines: Vec<&str> = err.lines().collect();
    assert_eq!(lines.len(), 2, "{err}");
    assert!(
        lines[0].starts_with(
            "warning: the coarse stage covers about \u{b1}16 px in x and \u{b1}16 px in y, \
             less than max_disp 64;"
        ),
        "{err}"
    );
    assert!(
        lines[1].starts_with("error: ") && lines[1].contains("f2.png"),
        "{err}"
    );
}

#[test]
fn max_disp_is_written_like_python_format_g() {
    let dir = scratch("capture_format_g");
    write_frames(&dir, 96, 2);
    for (given, shown) in [
        ("64", "64"),
        ("100.5", "100.5"),
        ("1e6", "1e+06"),
        ("123456.7", "123457"),
        ("40", "40"),
        ("2.5e5", "250000"),
    ] {
        let o = pair(&dir, "out", &["--set", &format!("max_disp={given}")]);
        assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
        assert_eq!(
            warnings(&o),
            [format!("warning: the coarse stage covers about \u{b1}16 px in x and \u{b1}16 px in y, less than max_disp {shown}; larger motions may be lost (use a larger image or crop margin, or a smaller max_disp)")],
            "max_disp={given}"
        );
    }
}

#[test]
fn a_run_that_fails_prints_its_error_only() {
    // the warning comes with the first result: a pair of different sizes (it would warn) only fails
    let dir = scratch("capture_fails");
    write_frames(&dir, 96, 1);
    write_png16(&dir.join("f1.png"), 96, 80, &vec![0.5; 96 * 80]);
    let o = pair(&dir, "out", &["--set", "max_disp=64"]);
    assert_eq!(o.status.code(), Some(2));
    let err = stderr(&o);
    assert!(
        err.starts_with("error:") && err.lines().count() == 1,
        "{err}"
    );
}

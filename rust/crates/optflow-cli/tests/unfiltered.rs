//! The displacement and strain without the confidence filter (`Cli::unfiltered`, a library-only
//! option of the desktop app): what is written, and that nothing else depends on it.

mod common;

use std::path::Path;

use common::*;
use optflow_cli::args::{parse, Cli, Parsed};
use optflow_cli::polygon::polygon_mask;
use optflow_cli::run;

const SIZE: usize = 128;
const POLY: [(f64, f64); 5] = [
    (20.0, 22.0),
    (108.0, 20.0),
    (110.0, 104.0),
    (64.0, 90.0),
    (22.0, 106.0),
];

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// `n` frames of a pattern moving by (0.7, -0.4) px per frame, the left part of it replaced by
/// another one from frame 2 on (so that some pixels fail the confidence), and `roi.json`.
fn scene(name: &str, n: usize) -> std::path::PathBuf {
    let dir = scratch(name);
    let (p, q) = (make_pattern(SIZE, SIZE, 3), make_pattern(SIZE, SIZE, 9));
    for k in 0..n {
        let (tx, ty) = (0.7 * k as f64, -0.4 * k as f64);
        let a = render(&p, SIZE, SIZE, tx, ty);
        let b = render(&q, SIZE, SIZE, tx, ty);
        let img: Vec<f64> = (0..SIZE * SIZE)
            .map(|i| if k >= 2 && i % SIZE > 56 { b[i] } else { a[i] })
            .collect();
        write_png16(&dir.join(format!("f{k}.png")), SIZE, SIZE, &img);
    }
    let pts: Vec<String> = POLY.iter().map(|(x, y)| format!("[{x}, {y}]")).collect();
    std::fs::write(
        dir.join("roi.json"),
        format!("{{\"polygon\": [{}]}}", pts.join(", ")),
    )
    .unwrap();
    dir
}

fn cli(dir: &Path, out: &Path, extra: &[&str]) -> Cli {
    let mut args: Vec<String> = [
        "seq",
        "--frames",
        s(dir),
        "--roi",
        s(&dir.join("roi.json")),
        "--set",
        "outer_iters=2",
        "--out",
        s(out),
    ]
    .iter()
    .map(|a| a.to_string())
    .collect();
    args.extend(extra.iter().map(|a| a.to_string()));
    match parse(&args).unwrap() {
        Parsed::Run(c) => *c,
        Parsed::Help => panic!("help"),
    }
}

fn bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{}/{name}: {e}", dir.display()))
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A sequence with the filter (the default) and without it, with `extra` options and the
/// reference moving when `hybrid`.
fn check(name: &str, extra: &[&str]) {
    let dir = scene(name, 5);
    let (on, off) = (dir.join("on"), dir.join("off"));
    let c_on = cli(&dir, &on, extra);
    assert!(!c_on.unfiltered, "off the command line only");
    run::run(&c_on).unwrap();
    let mut c_off = cli(&dir, &off, extra);
    c_off.unfiltered = true;
    c_off.raw = true;
    run::run(&c_off).unwrap();
    let mut c_raw = cli(&dir, &dir.join("raw"), extra);
    c_raw.raw = true;
    run::run(&c_raw).unwrap();

    // only a run without the filter leaves a note of it in its folder
    assert!(off.join("confidence_filter.json").is_file());
    assert!(!on.join("confidence_filter.json").exists());
    let roi = polygon_mask(SIZE, SIZE, &POLY, 1).data;
    let mut dropped = 0usize;
    for k in 1..5 {
        let (a, b, r) = (
            on.join(format!("frame_{k:05}")),
            off.join(format!("frame_{k:05}")),
            dir.join("raw").join(format!("frame_{k:05}")),
        );
        // the tracking does not depend on the option: these files are the same bytes
        let tracking = ["conf.npy", "valid.npy"]
            .into_iter()
            .chain(a.join("fb.npy").exists().then_some("fb.npy"));
        for f in tracking {
            assert!(bytes(&a, f) == bytes(&b, f), "frame {k}: {f}");
        }
        assert!(bytes(&r, "raw_ux.npy") == bytes(&b, "raw_ux.npy"), "{k}");
        assert!(bytes(&r, "raw_uy.npy") == bytes(&b, "raw_uy.npy"), "{k}");
        let ld = |d: &Path, n: &str| load_npy(&d.join(n)).1;
        let (ux_on, ux_off, raw) = (ld(&a, "ux.npy"), ld(&b, "ux.npy"), ld(&b, "raw_ux.npy"));
        let (uy_on, uy_off, rawy) = (ld(&a, "uy.npy"), ld(&b, "uy.npy"), ld(&b, "raw_uy.npy"));
        for i in 0..SIZE * SIZE {
            if roi[i] {
                // every pixel of the ROI is the raw field
                assert_eq!(ux_off[i].to_bits(), raw[i].to_bits(), "frame {k} ux {i}");
                assert_eq!(uy_off[i].to_bits(), rawy[i].to_bits(), "frame {k} uy {i}");
            } else {
                assert!(ux_off[i].is_nan() && uy_off[i].is_nan(), "outside the ROI");
            }
            if ux_on[i].is_finite() {
                // what passes the filter is the same value
                assert_eq!(ux_on[i].to_bits(), ux_off[i].to_bits(), "frame {k} {i}");
                assert_eq!(uy_on[i].to_bits(), uy_off[i].to_bits(), "frame {k} {i}");
            } else if roi[i] && ux_off[i].is_finite() {
                dropped += 1;
            }
        }
        // valid.npy is what the filter passes, as ever
        let valid = ld(&b, "valid.npy");
        assert!((0..SIZE * SIZE).all(|i| (valid[i] == 1.0) == ux_on[i].is_finite()));
        // strain from the written field: finite wherever the whole window is finite
        let half = 7; // strain_win 15
        let (exx, gmax, vol) = (ld(&b, "Exx.npy"), ld(&b, "gmax.npy"), ld(&b, "vol.npy"));
        let mut n_checked = 0;
        for y in half..SIZE - half {
            for x in half..SIZE - half {
                let all = (y - half..=y + half).all(|yy| {
                    (x - half..=x + half).all(|xx| {
                        ux_off[yy * SIZE + xx].is_finite() && uy_off[yy * SIZE + xx].is_finite()
                    })
                });
                if all {
                    let i = y * SIZE + x;
                    assert!(
                        exx[i].is_finite() && gmax[i].is_finite() && vol[i].is_finite(),
                        "frame {k}: strain at ({x}, {y})"
                    );
                    n_checked += 1;
                }
            }
        }
        assert!(n_checked > 500, "frame {k}: only {n_checked} windows");
    }
    assert!(
        dropped > 0,
        "{name}: the filter must drop some ROI pixels in this scene"
    );
}

#[test]
fn unfiltered_chain_with_a_crop() {
    check("unf_chain", &["--set", "seq_mode=chain"]);
}

#[test]
fn unfiltered_hybrid_without_a_crop() {
    check(
        "unf_hybrid",
        &[
            "--crop",
            "off",
            "--set",
            "seq_mode=hybrid",
            "--set",
            "ref_update_conf=0.6",
        ],
    );
}

#[test]
fn unfiltered_with_averaging_and_a_pair() {
    check(
        "unf_tavg",
        &["--set", "tavg=1", "--set", "ref_update_conf=0.6"],
    );
    // a pair too
    let dir = scene("unf_pair", 2);
    let out = |n: &str| dir.join(n);
    let args = |o: &Path| {
        vec![
            "pair".to_string(),
            "--ref".into(),
            s(&dir.join("f0.png")).into(),
            "--def".into(),
            s(&dir.join("f1.png")).into(),
            "--roi".into(),
            s(&dir.join("roi.json")).into(),
            "--set".into(),
            "outer_iters=2".into(),
            "--out".into(),
            s(o).into(),
        ]
    };
    let parse_pair = |o: &Path| match parse(&args(o)).unwrap() {
        Parsed::Run(c) => *c,
        Parsed::Help => panic!("help"),
    };
    run::run(&parse_pair(&out("on"))).unwrap();
    let mut c = parse_pair(&out("off"));
    c.unfiltered = true;
    run::run(&c).unwrap();
    for f in ["conf.npy", "valid.npy"] {
        assert!(bytes(&out("on"), f) == bytes(&out("off"), f), "{f}");
    }
    let roi = polygon_mask(SIZE, SIZE, &POLY, 1).data;
    let (a, b) = (
        load_npy(&out("on").join("ux.npy")).1,
        load_npy(&out("off").join("ux.npy")).1,
    );
    for i in 0..SIZE * SIZE {
        assert_eq!(b[i].is_finite(), roi[i] && b[i].is_finite());
        if a[i].is_finite() {
            assert_eq!(a[i].to_bits(), b[i].to_bits());
        }
    }
    assert!(!same(&a, &b));
}

#[test]
fn the_crack_model_has_no_unfiltered_output() {
    let dir = scene("unf_crack", 3);
    let mut c = cli(&dir, &dir.join("out"), &["--set", "crack_model=xdic"]);
    c.unfiltered = true;
    let e = run::run(&c).unwrap_err();
    assert!(format!("{e:#}").contains("confidence filter"), "{e:#}");
}

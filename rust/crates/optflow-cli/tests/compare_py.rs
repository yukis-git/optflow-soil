//! `optflow_core::compare::robust` gives the numbers and the verdict of
//! `rust/scripts/compare_outputs.py --tier robust` (the script is the reference).

mod common;

use std::path::Path;

use common::*;
use optflow_core::compare::{robust, Fields};

fn load_f32(dir: &Path, n: &str) -> Vec<f32> {
    load_npy(&dir.join(format!("{n}.npy")))
        .1
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn with_fields<T>(dir: &Path, f: impl FnOnce(&Fields) -> T) -> T {
    let (ux, uy, conf) = (
        load_f32(dir, "ux"),
        load_f32(dir, "uy"),
        load_f32(dir, "conf"),
    );
    let strain: Vec<Vec<f32>> = ["Exx", "Eyy", "Exy", "gmax", "vol"]
        .iter()
        .map(|n| load_f32(dir, n))
        .collect();
    f(&Fields {
        ux: &ux,
        uy: &uy,
        conf: &conf,
        strain: strain.iter().map(Vec::as_slice).collect(),
    })
}

/// The number after `key` in `line`.
fn after(line: &str, key: &str) -> f64 {
    let rest = line
        .split(key)
        .nth(1)
        .unwrap_or_else(|| panic!("{key} in {line}"));
    rest.split_whitespace().next().unwrap().parse().unwrap()
}

#[test]
fn rust_judge_matches_compare_outputs_py() {
    let Some(python) = std::env::var_os("OPTFLOW_PYTHON") else {
        eprintln!("skipped: OPTFLOW_PYTHON");
        return;
    };
    let dir = scratch("compare_py");
    let (w, h) = (96, 96);
    let p = make_pattern(w, h, 2);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 0.5, 0.2));
    let (a, b) = (dir.join("a.png"), dir.join("b.png"));
    for (name, sigma) in [("r0", "sigma_w=4.0"), ("r1", "sigma_w=4.2")] {
        let out = dir.join(name);
        let o = optflow(&[
            "pair",
            "--ref",
            a.to_str().unwrap(),
            "--def",
            b.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--set",
            "max_disp=16",
            "--set",
            sigma,
        ]);
        assert!(o.status.success(), "{}", stderr(&o));
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/compare_outputs.py");
    // identical outputs pass; slightly different sigma_w gives a verdict of either kind
    for (x, y) in [("r0", "r0"), ("r0", "r1")] {
        let (dx, dy) = (dir.join(x), dir.join(y));
        let o = std::process::Command::new(&python)
            .args([
                script.to_str().unwrap(),
                dx.to_str().unwrap(),
                dy.to_str().unwrap(),
                "--tier",
                "robust",
            ])
            .output()
            .expect("run compare_outputs.py");
        let text = stdout(&o);
        let line = |p: &str| {
            text.lines()
                .find(|l| l.starts_with(p))
                .unwrap_or_else(|| panic!("{p} in {text}"))
                .to_string()
        };
        let r = with_fields(&dx, |fa| with_fields(&dy, |fb| robust(fa, fb)));
        let (lx, ly) = (line("ux:"), line("uy:"));
        for (got, line, key) in [
            (r.ux_trim, &lx, "trim99.9"),
            (r.ux_p99, &lx, "p99"),
            (r.uy_trim, &ly, "trim99.9"),
            (r.uy_p99, &ly, "p99"),
        ] {
            let want = after(line, key);
            // the script prints 3 significant digits (rounding error up to 5e-3 relative)
            assert!(
                (got - want).abs() <= 5e-3 * want.abs().max(1e-300) + 1e-12,
                "{x} vs {y} {key}: {got} vs {want} ({line})"
            );
        }
        let verdict = line("RESULT").ends_with("pass");
        assert_eq!(verdict, r.ok(), "{x} vs {y}: {r:?}\n{text}");
        eprintln!(
            "{x} vs {y}: {} {r:?}",
            if verdict { "pass" } else { "fail" }
        );
    }
}

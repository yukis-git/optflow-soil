//! `OPTFLOW_ITER_TRACE` on the GPU engine: off by default, and on it changes no bit of the result
//! and writes one line per fine stage. The traced run is this test binary run again with the
//! variable set (it is read once per process).
#![cfg(target_os = "macos")]
#[path = "../../optflow-core/tests/common/synth.rs"]
mod synth;

use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::iter_trace;
use optflow_core::track::{track_pair_on, TILE};
use optflow_gpu::GpuEngine;

/// Set in the traced run: the file its bits go to.
const CHILD: &str = "OPTFLOW_ITER_TRACE_TEST_BITS";

fn pair(w: usize, h: usize) -> (Image<f32>, Image<f32>) {
    let f = synth::uniform_strain(0.01, -0.003, (w as f64 / 2.0, h as f64 / 2.0), (1.3, -0.7));
    let p = synth::make_pattern(w, h, 1, 32.0);
    (
        synth::render(&p, w, h, None),
        synth::render(&p, w, h, Some(&f)),
    )
}

/// The bits of a pair refined in one window and of one refined in two tiles.
fn run() -> Vec<u8> {
    let gpu = GpuEngine::new().expect("a Metal device");
    [(256, 256), (TILE + 76, 160)]
        .into_iter()
        .flat_map(|(w, h)| {
            let (a, b) = pair(w, h);
            let r = track_pair_on(&gpu, &a, &b, &Config::default(), None, None).unwrap();
            [&r.ux, &r.uy, &r.conf]
                .into_iter()
                .flat_map(|im| im.data.iter().flat_map(|v| v.to_bits().to_le_bytes()))
                .collect::<Vec<u8>>()
        })
        .collect()
}

#[test]
fn traced_child() {
    let Some(out) = std::env::var_os(CHILD) else {
        return; // only in the run that `trace_is_off_by_default_and_changes_no_bit` starts
    };
    assert!(iter_trace::enabled());
    std::fs::write(out, run()).unwrap();
}

#[test]
fn trace_is_off_by_default_and_changes_no_bit() {
    if std::env::var_os(CHILD).is_some() || std::env::var_os("OPTFLOW_ITER_TRACE").is_some() {
        return;
    }
    assert!(!iter_trace::enabled());
    let want = run();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (trace, got) = (
        dir.join("iter-trace-gpu.jsonl"),
        dir.join("iter-trace-gpu.bin"),
    );
    let _ = std::fs::remove_file(&trace);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "traced_child", "--test-threads=1"])
        .env("OPTFLOW_ITER_TRACE", &trace)
        .env(CHILD, &got)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(
        std::fs::read(&got).unwrap() == want,
        "the trace changed the result"
    );
    let text = std::fs::read_to_string(&trace).unwrap();
    let tiles: Vec<usize> = text
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert_eq!(v["iters"].as_array().unwrap().len(), 8);
            v["tiles"].as_array().unwrap().len()
        })
        .collect();
    assert_eq!(tiles, [1, 2]);
}

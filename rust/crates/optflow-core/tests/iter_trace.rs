//! `OPTFLOW_ITER_TRACE`: off by default, and on it changes no bit of the result and writes one
//! line per fine stage. The traced run is this test binary run again with the variable set (it
//! is read once per process).
mod common;
use common::{bits, synth};
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::iter_trace;
use optflow_core::track::{track_pair, TILE};

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

fn cfg() -> Config {
    Config {
        outer_iters: 3,
        ..Config::default()
    }
}

/// The bits of a pair refined in one window and of one refined in two tiles.
fn run() -> Vec<u32> {
    [(128, 128), (TILE + 76, 160)]
        .into_iter()
        .flat_map(|(w, h)| {
            let (a, b) = pair(w, h);
            let r = track_pair(&a, &b, &cfg(), None, None).unwrap();
            [bits(&r.ux), bits(&r.uy), bits(&r.conf)].concat()
        })
        .collect()
}

#[test]
fn traced_child() {
    let Some(out) = std::env::var_os(CHILD) else {
        return; // only in the run that `trace_is_off_by_default_and_changes_no_bit` starts
    };
    assert!(iter_trace::enabled());
    let b: Vec<u8> = run().iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, b).unwrap();
}

#[test]
fn trace_is_off_by_default_and_changes_no_bit() {
    if std::env::var_os(CHILD).is_some() || std::env::var_os("OPTFLOW_ITER_TRACE").is_some() {
        return;
    }
    assert!(!iter_trace::enabled());
    let want: Vec<u8> = run().iter().flat_map(|v| v.to_le_bytes()).collect();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (trace, got) = (
        dir.join("iter-trace-cpu.jsonl"),
        dir.join("iter-trace-cpu.bin"),
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
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    for (line, tiles) in lines.iter().zip([1, 2]) {
        assert_eq!(line["tiles"].as_array().unwrap().len(), tiles);
        let iters = line["iters"].as_array().unwrap();
        assert_eq!(iters.len(), cfg().outer_iters);
        for it in iters {
            assert!(it["n"].as_u64().unwrap() > 0);
            let v = |k: &str| it[k].as_f64().unwrap();
            let (max, p99, rms) = (v("max"), v("p99"), v("rms"));
            assert!(max >= p99 && max >= rms, "{it}");
        }
    }
}

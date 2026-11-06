//! `track_pair` with the CPU engine and with another engine on the same pair: per field, the
//! agreement against the limits of the GPU spec §5.
//!
//!     cargo run --release -p optflow-core --example engine_compare -- --a A.npy --b B.npy
//!         [--set key=value ...] [--engine model|perturbed]
//!
//! `A.npy`, `B.npy` are 2-D float32 arrays (`rust/scripts/dump_synth_case.py` writes them and a
//! `set.txt` of `key=value` lines for `--set`; the value is JSON text). `--engine model` is the
//! incremental-f32 model of the GPU numerics, `perturbed` the same with the stored responses and
//! reference phase jittered by one f32 ulp (the stand-in for a different FFT). The GPU engine is
//! compared by `bench_gpu --compare-cpu` of optflow-gpu. Prints per field the RMS of the difference
//! over the pixels finite in both, the largest difference and the agreement of `isfinite(ux)`,
//! then one line `GATE ok|fail`. The exit code is 0 either way.

use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, FineEngine};
use optflow_core::gpu_model::{ModelEngine, PerturbedModelEngine};
use optflow_core::image::Image;
use optflow_core::npy::{self, NpyData};
use optflow_core::track::{track_pair_on, TrackResult};

/// The limits of the GPU spec §5: RMS of ux, uy (px), of conf, of the strains, and the least
/// agreement of the validity mask.
const LIMIT_U: f64 = 1e-5;
const LIMIT_CONF: f64 = 1e-5;
const LIMIT_STRAIN: f64 = 1e-6;
const LIMIT_VALID: f64 = 0.999;

fn load(path: &str) -> Result<Image<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let arr = npy::read(&bytes).map_err(|e| format!("{path}: {e}"))?;
    let [h, w] = arr.shape[..] else {
        return Err(format!("{path}: not a 2-D array ({:?})", arr.shape));
    };
    match arr.data {
        NpyData::F32(v) => Ok(Image::from_vec(w, h, v)),
        _ => Err(format!("{path}: not float32")),
    }
}

/// RMS and largest |difference| over the pixels finite in both.
fn rms_max(x: &[f32], y: &[f32]) -> (f64, f64) {
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for (a, b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            let d = f64::from(*a) - f64::from(*b);
            sum += d * d;
            max = max.max(d.abs());
            n += 1;
        }
    }
    (if n == 0 { 0.0 } else { (sum / n as f64).sqrt() }, max)
}

/// Same, for conf, which is finite everywhere (a pair that is not finite in both is a failure).
fn rms_max_all(x: &[f32], y: &[f32]) -> (f64, f64) {
    if x.iter()
        .zip(y)
        .any(|(a, b)| !(a.is_finite() && b.is_finite()))
    {
        return (f64::INFINITY, f64::INFINITY);
    }
    rms_max(x, y)
}

fn run(args: &[String]) -> Result<(), String> {
    let (mut a, mut b, mut engine) = (None, None, "model".to_string());
    let mut sets = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--a" => a = Some(value("--a")?),
            "--b" => b = Some(value("--b")?),
            "--set" => sets.push(value("--set")?),
            "--engine" => engine = value("--engine")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let (a, b) = (a.ok_or("--a is required")?, b.ok_or("--b is required")?);
    let entries: Vec<String> = sets
        .iter()
        .map(|s| {
            let (k, v) = s
                .split_once('=')
                .ok_or(format!("--set {s}: not key=value"))?;
            Ok(format!("\"{k}\": {v}"))
        })
        .collect::<Result<_, String>>()?;
    let cfg =
        Config::from_json(&format!("{{{}}}", entries.join(", "))).map_err(|e| e.to_string())?;
    let other: Box<dyn FineEngine> = match engine.as_str() {
        "model" => Box::new(ModelEngine),
        "perturbed" => Box::new(PerturbedModelEngine(1)),
        e => return Err(format!("--engine {e}: expected model or perturbed")),
    };
    let (i1, i2) = (load(&a)?, load(&b)?);
    if (i1.w, i1.h) != (i2.w, i2.h) {
        return Err("the two images differ in size".into());
    }
    let run = |e: &dyn FineEngine| -> Result<TrackResult, String> {
        track_pair_on(e, &i1, &i2, &cfg, None, None).map_err(|e| e.to_string())
    };
    let want = run(&CpuEngine)?;
    let got = run(other.as_ref())?;

    let strains = |r: &TrackResult| {
        [
            ("exx", r.strain.exx.data.clone()),
            ("eyy", r.strain.eyy.data.clone()),
            ("exy", r.strain.exy.data.clone()),
            ("gmax", r.strain.gmax.data.clone()),
            ("vol", r.strain.vol.data.clone()),
        ]
    };
    let mut ok = true;
    let mut line = |name: &str, (rms, max): (f64, f64), limit: f64| {
        ok &= rms <= limit;
        println!("{name:5} rms {rms:.3e}  max {max:.3e}  (limit {limit:.0e})");
    };
    line("ux", rms_max(&got.ux.data, &want.ux.data), LIMIT_U);
    line("uy", rms_max(&got.uy.data, &want.uy.data), LIMIT_U);
    line(
        "conf",
        rms_max_all(&got.conf.data, &want.conf.data),
        LIMIT_CONF,
    );
    for ((name, g), (_, w)) in strains(&got).iter().zip(strains(&want).iter()) {
        line(name, rms_max(g, w), LIMIT_STRAIN);
    }
    let agree = got
        .ux
        .data
        .iter()
        .zip(&want.ux.data)
        .filter(|(p, q)| p.is_finite() == q.is_finite())
        .count();
    let valid = agree as f64 / want.ux.data.len() as f64;
    ok &= valid >= LIMIT_VALID;
    println!("valid agreement {valid:.5}  (limit {LIMIT_VALID})");
    println!("GATE {}", if ok { "ok" } else { "fail" });
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("engine_compare: {e}");
    }
}

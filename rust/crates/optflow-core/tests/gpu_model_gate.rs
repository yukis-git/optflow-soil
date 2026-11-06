//! Gate of the GPU spec §4.6: the incremental-f32 numerics (`ModelEngine`) against the CPU's f64
//! path, with the limits of §5, judged at track level. Every case runs twice: with the CPU's
//! responses and reference phase as they are (`plain`), and with each of those values scaled by
//! `1 + e`, `|e| <= 2^-23` (`perturbed`, the stand-in for a GPU FFT with other low bits). Run with
//! `-- --nocapture` for the table; `--ignored` adds the 512^2 cases. A failing case does not stop
//! the others: the test fails at its end with the list.
mod common;
use common::synth;
use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, FineEngine};
use optflow_core::gpu_model::{ModelEngine, PerturbedModelEngine};
use optflow_core::image::Image;
use optflow_core::track::{track_pair_on, track_sequence_on, TrackResult};

struct Diff {
    ux: f64,
    uy: f64,
    conf: f64,
    strain: f64,
    max_u: f64,
    valid: f64,
}

/// RMS and largest difference over the pixels finite in both.
fn rms_max(x: &[f32], y: &[f32]) -> (f64, f64) {
    let d: Vec<f64> = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| f64::from(*a) - f64::from(*b))
        .collect();
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let rms = (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt();
    (rms, d.iter().fold(0.0, |m, v| m.max(v.abs())))
}

fn diff(a: &TrackResult, b: &TrackResult) -> Diff {
    let s = [
        (&a.strain.exx, &b.strain.exx),
        (&a.strain.eyy, &b.strain.eyy),
        (&a.strain.exy, &b.strain.exy),
        (&a.strain.gmax, &b.strain.gmax),
        (&a.strain.vol, &b.strain.vol),
    ];
    let agree =
        a.ux.data
            .iter()
            .zip(&b.ux.data)
            .filter(|(p, q)| p.is_finite() == q.is_finite())
            .count();
    let (ux, mx) = rms_max(&a.ux.data, &b.ux.data);
    Diff {
        ux,
        uy: rms_max(&a.uy.data, &b.uy.data).0,
        conf: rms_max(&a.conf.data, &b.conf.data).0,
        strain: s
            .iter()
            .map(|(p, q)| rms_max(&p.data, &q.data).0)
            .fold(0.0, f64::max),
        max_u: mx,
        valid: agree as f64 / a.ux.data.len() as f64,
    }
}

fn engines() -> [(&'static str, Box<dyn FineEngine>); 2] {
    [
        ("plain", Box::new(ModelEngine)),
        ("perturbed", Box::new(PerturbedModelEngine(1))),
    ]
}

fn report(name: &str, mode: &str, d: &Diff) -> Option<String> {
    let ok = d.ux <= 1e-5 && d.uy <= 1e-5 && d.conf <= 1e-5 && d.strain <= 1e-6 && d.valid >= 0.999;
    println!(
        "{name:20} {mode:9} ux {:.2e} uy {:.2e} conf {:.2e} strain {:.2e} max|dux| {:.2e} valid {:.5} GATE {}",
        d.ux, d.uy, d.conf, d.strain, d.max_u, d.valid,
        if ok { "ok" } else { "fail" }
    );
    (!ok).then(|| format!("{name} [{mode}]"))
}

fn gate(name: &str, a: &Image<f32>, b: &Image<f32>, cfg: &Config) -> Vec<String> {
    let want = track_pair_on(&CpuEngine, a, b, cfg, None, None).unwrap();
    let mut failed = Vec::new();
    for (mode, e) in engines() {
        let got = track_pair_on(e.as_ref(), a, b, cfg, None, None).unwrap();
        failed.extend(report(name, mode, &diff(&got, &want)));
    }
    failed
}

fn speckle(
    n: usize,
    exx: f64,
    eyy: f64,
    t: (f64, f64),
    seed: u64,
    margin: f64,
) -> (Image<f32>, Image<f32>) {
    let c = n as f64 / 2.0;
    let f = synth::uniform_strain(exx, eyy, (c, c), t);
    let p = synth::make_pattern(n, n, seed, margin);
    (
        synth::render(&p, n, n, None),
        synth::render(&p, n, n, Some(&f)),
    )
}

fn finish(failed: Vec<String>) {
    assert!(failed.is_empty(), "gate failed: {failed:?}");
}

#[test]
fn gate_strain_10_percent() {
    let (a, b) = speckle(256, 0.1, -0.05, (1.3, -0.7), 1, 32.0);
    finish(gate("strain 10%", &a, &b, &Config::default()));
}

#[test]
fn gate_coarse_failure() {
    let (a, b) = speckle(256, 0.01, -0.003, (45.85, -33.65), 2, 80.0);
    let cfg = Config {
        max_disp: 64.0,
        ..Config::default()
    };
    finish(gate("coarse failure", &a, &b, &cfg));
}

#[test]
#[ignore]
fn gate_512_pairs() {
    let (a, b) = speckle(512, 0.01, -0.003, (3.4, -2.1), 1, 32.0);
    let mut failed = gate("512 (3.4,-2.1)", &a, &b, &Config::default());
    let (a, b) = speckle(512, 0.01, -0.003, (91.7, -67.3), 1, 32.0);
    let cfg = Config {
        max_disp: 128.0,
        ..Config::default()
    };
    failed.extend(gate("512 (91.7,-67.3)", &a, &b, &cfg));
    finish(failed);
}

fn fixture(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

/// `refine_raw` level on the repository fixtures: ux, uy only.
#[test]
fn gate_fixtures_order_1_and_2() {
    let (a, b) = (fixture("refine_a"), fixture("refine_b"));
    let (ux, uy) = (fixture("refine_init_ux"), fixture("refine_init_uy"));
    let mut failed = Vec::new();
    for order in [1, 2] {
        let cfg = Config {
            shape_order: order,
            ..Config::default()
        };
        let want = CpuEngine
            .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
            .unwrap();
        for (mode, e) in engines() {
            let got = e.refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg).unwrap();
            let rms = |x: &Image<f64>, y: &Image<f64>| {
                let d: Vec<f64> = x
                    .data
                    .iter()
                    .zip(&y.data)
                    .filter(|(p, q)| p.is_finite() && q.is_finite())
                    .map(|(p, q)| p - q)
                    .collect();
                let rms = (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt();
                (rms, d.iter().fold(0.0f64, |m, v| m.max(v.abs())))
            };
            let ((ex, mx), (ey, my)) = (rms(&got.ux, &want.ux), rms(&got.uy, &want.uy));
            let ok = ex <= 1e-5 && ey <= 1e-5;
            println!(
                "{:20} {mode:9} ux {ex:.2e} uy {ey:.2e} max|dux| {mx:.2e} max|duy| {my:.2e} GATE {}",
                format!("fixture order {order}"),
                if ok { "ok" } else { "fail" }
            );
            if !ok {
                failed.push(format!("fixture order {order} [{mode}]"));
            }
        }
    }
    finish(failed);
}

#[test]
fn gate_sequence() {
    let n = 128;
    let c = n as f64 / 2.0;
    let p = synth::make_pattern(n, n, 3, 40.0);
    let frames: Vec<Image<f32>> = std::iter::once(synth::render(&p, n, n, None))
        .chain((1..=3).map(|k| {
            let k = f64::from(k);
            let f = synth::uniform_strain(0.002 * k, -0.0006 * k, (c, c), (1.0 * k, 0.5 * k));
            synth::render(&p, n, n, Some(&f))
        }))
        .collect();
    let cfg = Config::default();
    let run = |e: &dyn FineEngine| -> Vec<(usize, TrackResult)> {
        track_sequence_on(e, frames.clone().into_iter(), &cfg, None)
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let want = run(&CpuEngine);
    assert_eq!(want.len(), 3);
    let mut failed = Vec::new();
    for (mode, e) in engines() {
        let got = run(e.as_ref());
        assert_eq!(got.len(), want.len());
        for ((k, g), (_, w)) in got.iter().zip(&want) {
            failed.extend(report(&format!("sequence frame {k}"), mode, &diff(g, w)));
        }
    }
    finish(failed);
}

//! `GpuEngine` against the CPU (the limits of GPU spec §5), against `ModelEngine` (the same
//! numerics on the CPU, differing only through the GPU FFT), and against itself (determinism).
//! Run with `-- --nocapture` for the measured differences.
#![cfg(target_os = "macos")]
#[path = "../../optflow-core/tests/common/synth.rs"]
mod synth;

use std::path::PathBuf;

use optflow_core::compare;
use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, FineEngine, Prepared};
use optflow_core::gpu_model::ModelEngine;
use optflow_core::image::Image;
use optflow_core::npy::{self, NpyData};
use optflow_core::phaseflow::RawFlow;
use optflow_core::track::{track_pair_on, track_sequence_on, TrackResult};
use optflow_gpu::GpuEngine;

fn speckle_wh(w: usize, h: usize, t: (f64, f64)) -> (Image<f32>, Image<f32>) {
    let f = synth::uniform_strain(0.01, -0.003, (w as f64 / 2.0, h as f64 / 2.0), t);
    let p = synth::make_pattern(w, h, 1, 32.0);
    (
        synth::render(&p, w, h, None),
        synth::render(&p, w, h, Some(&f)),
    )
}

fn speckle(n: usize, t: (f64, f64)) -> (Image<f32>, Image<f32>) {
    speckle_wh(n, n, t)
}

fn fixture(name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../optflow-core/tests/fixtures")
        .join(format!("{name}.npy"));
    let a = npy::read(&std::fs::read(&path).unwrap()).unwrap();
    let v = match a.data {
        NpyData::F64(v) => v,
        NpyData::F32(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U8(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U16(v) => v.into_iter().map(f64::from).collect(),
        NpyData::Bool(v) => v.into_iter().map(|b| f64::from(u8::from(b))).collect(),
    };
    (a.shape, v)
}

fn image(name: &str) -> Image<f32> {
    let (s, v) = fixture(name);
    Image::from_vec(s[1], s[0], v.iter().map(|&x| x as f32).collect())
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

fn fields(r: &TrackResult) -> compare::Fields<'_> {
    compare::Fields {
        ux: &r.ux.data,
        uy: &r.uy.data,
        conf: &r.conf.data,
        strain: vec![
            &r.strain.exx.data,
            &r.strain.eyy.data,
            &r.strain.exy.data,
            &r.strain.gmax.data,
            &r.strain.vol.data,
        ],
    }
}

/// `check(name, got, want)`: the five metrics of §5 for two track results; false on a miss.
fn check(name: &str, got: &TrackResult, want: &TrackResult) -> bool {
    let s = |r: &TrackResult| {
        [
            r.strain.exx.data.clone(),
            r.strain.eyy.data.clone(),
            r.strain.exy.data.clone(),
            r.strain.gmax.data.clone(),
            r.strain.vol.data.clone(),
        ]
    };
    let (sg, sw) = (s(got), s(want));
    let strain = sg
        .iter()
        .zip(&sw)
        .map(|(a, b)| rms_max(a, b).0)
        .fold(0.0, f64::max);
    let agree = got
        .ux
        .data
        .iter()
        .zip(&want.ux.data)
        .filter(|(p, q)| p.is_finite() == q.is_finite())
        .count() as f64
        / got.ux.data.len() as f64;
    let (ux, mx) = rms_max(&got.ux.data, &want.ux.data);
    let (uy, my) = rms_max(&got.uy.data, &want.uy.data);
    let (conf, _) = rms_max(&got.conf.data, &want.conf.data);
    let r = compare::robust(&fields(got), &fields(want));
    let ok = r.ok();
    println!(
        "{name:28} robust: ux trim {:.2e} p99 {:.2e} uy trim {:.2e} p99 {:.2e} conf {:.2e} strain trim {:.2e} valid {:.5} {} | plain: ux {ux:.2e} uy {uy:.2e} conf {conf:.2e} strain {strain:.2e} max|dux| {mx:.2e} max|duy| {my:.2e} valid {agree:.5}",
        r.ux_trim,
        r.ux_p99,
        r.uy_trim,
        r.uy_p99,
        r.conf_rms_both,
        r.strain_trim_max,
        r.valid_agree,
        if ok { "ok" } else { "FAIL" }
    );
    ok
}

/// Distance from the border below which the raw flow is not compared: the guard zone of the
/// pipeline (`B` + guard = 28 px) that track results discard. Near the corners of a small image
/// the f32 incremental numerics differ from the f64 CPU path by 1e-2 px even for `ModelEngine`
/// (measured; see the report), so no engine can meet the limits there.
const INTERIOR: usize = 28;

/// RMS of the differences of the raw flows (f64) away from the border, over the pixels finite
/// in both.
fn raw_diff(a: &RawFlow, b: &RawFlow) -> (f64, f64, f64, f64) {
    let (w, h) = (a.ux.w, a.ux.h);
    let rms = |x: &Image<f64>, y: &Image<f64>| {
        let d: Vec<f64> = (0..w * h)
            .filter(|i| {
                let (px, py) = (i % w, i / w);
                px >= INTERIOR && py >= INTERIOR && px + INTERIOR < w && py + INTERIOR < h
            })
            .map(|i| (x.data[i], y.data[i]))
            .filter(|(p, q)| p.is_finite() && q.is_finite())
            .map(|(p, q)| p - q)
            .collect();
        let r = (d.iter().map(|v| v * v).sum::<f64>() / d.len().max(1) as f64).sqrt();
        (r, d.iter().fold(0.0f64, |m, v| m.max(v.abs())))
    };
    let ((rx, mx), (ry, _)) = (rms(&a.ux, &b.ux), rms(&a.uy, &b.uy));
    (rx, ry, mx, rms(&a.resid, &b.resid).0)
}

/// The robust tier for raw flows: (trim 99.9 % RMS, p99) of |a - b| in the interior (`INTERIOR`),
/// over the pixels finite in both.
fn raw_tail(x: &Image<f64>, y: &Image<f64>) -> (f64, f64) {
    let (w, h) = (x.w, x.h);
    let mask: Vec<bool> = (0..w * h)
        .map(|i| {
            let (px, py) = (i % w, i / w);
            px >= INTERIOR && py >= INTERIOR && px + INTERIOR < w && py + INTERIOR < h
        })
        .collect();
    compare::tail(&x.data, &y.data, Some(&mask))
}

/// Raw flows agree: trim of ux and uy (and of the residual with `resid`) is <= 1e-5. No p99 limit
/// here: raw flows carry low-confidence pixels unmasked; the spec's p99 limit is for final outputs.
fn raw_ok(a: &RawFlow, b: &RawFlow, resid: bool) -> bool {
    let t = |x, y| raw_tail(x, y).0 <= 1e-5;
    t(&a.ux, &b.ux) && t(&a.uy, &b.uy) && (!resid || t(&a.resid, &b.resid))
}

fn bits(im: &Image<f32>) -> Vec<u32> {
    im.data.iter().map(|v| v.to_bits()).collect()
}

fn gpu() -> GpuEngine {
    GpuEngine::new().expect("a Metal device")
}

#[test]
fn refine_raw_matches_the_model_and_the_cpu() {
    let gpu = gpu();
    let (a, b) = speckle(128, (3.4, -2.1));
    let (ux, uy) = (
        Image::from_vec(128, 128, vec![3.0f32; 128 * 128]),
        Image::from_vec(128, 128, vec![-2.0f32; 128 * 128]),
    );
    for order in [1, 2] {
        let cfg = Config {
            shape_order: order,
            ..Config::default()
        };
        let want = CpuEngine
            .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
            .unwrap();
        let model = ModelEngine
            .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
            .unwrap();
        let got = gpu.refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg).unwrap();
        let (mx, my, mm, mr) = raw_diff(&got, &model);
        let (cx, cy, cm, cr) = raw_diff(&got, &want);
        println!("order {order} tails (trim, p99): gpu-model ux {:?} uy {:?} | gpu-cpu ux {:?} uy {:?} resid {:?}", raw_tail(&got.ux, &model.ux), raw_tail(&got.uy, &model.uy), raw_tail(&got.ux, &want.ux), raw_tail(&got.uy, &want.uy), raw_tail(&got.resid, &want.resid));
        println!(
            "order {order}: gpu-model ux RMS {mx:.2e} uy {my:.2e} max {mm:.2e} resid {mr:.2e} | gpu-cpu ux RMS {cx:.2e} uy {cy:.2e} max {cm:.2e} resid {cr:.2e}"
        );
        assert!(raw_ok(&got, &model, false), "gpu vs model, order {order}");
        assert!(raw_ok(&got, &want, true), "gpu vs cpu, order {order}");
    }
}

#[test]
fn degenerate_stripes_run_the_reset_paths() {
    let gpu = gpu();
    let (a, b) = (image("refine_stripe_a"), image("refine_stripe_b"));
    let (ux, uy) = (
        Image::from_vec(a.w, a.h, vec![0.7f32; a.w * a.h]),
        Image::from_vec(a.w, a.h, vec![-0.4f32; a.w * a.h]),
    );
    for order in [1, 2] {
        let cfg = Config {
            shape_order: order,
            ..Config::default()
        };
        let want = CpuEngine
            .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
            .unwrap();
        let model = ModelEngine
            .refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg)
            .unwrap();
        let got = gpu.refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg).unwrap();
        let (mx, my, mm, _) = raw_diff(&got, &model);
        let (cx, cy, cm, _) = raw_diff(&got, &want);
        println!(
            "stripes order {order}: gpu-model ux {mx:.2e} uy {my:.2e} max {mm:.2e} | gpu-cpu ux {cx:.2e} uy {cy:.2e} max {cm:.2e}"
        );
        assert!(raw_ok(&got, &model, false), "gpu vs model, order {order}");
        assert!(raw_ok(&got, &want, false), "gpu vs cpu, order {order}");
    }
}

fn track_case(t: (f64, f64), md: f64) -> bool {
    let gpu = gpu();
    let (a, b) = speckle(512, t);
    let cfg = Config {
        max_disp: md,
        ..Config::default()
    };
    let want = track_pair_on(&CpuEngine, &a, &b, &cfg, None, None).unwrap();
    let model = track_pair_on(&ModelEngine, &a, &b, &cfg, None, None).unwrap();
    let got = track_pair_on(&gpu, &a, &b, &cfg, None, None).unwrap();
    let again = track_pair_on(&gpu, &a, &b, &cfg, None, None).unwrap();
    let ok = check(&format!("512 {t:?} gpu-cpu"), &got, &want);
    check(&format!("512 {t:?} gpu-model"), &got, &model);
    check(&format!("512 {t:?} model-cpu"), &model, &want);
    assert_eq!(bits(&got.ux), bits(&again.ux), "deterministic ux");
    assert_eq!(bits(&got.uy), bits(&again.uy), "deterministic uy");
    assert_eq!(bits(&got.conf), bits(&again.conf), "deterministic conf");
    ok
}

#[test]
fn track_pair_matches_the_cpu_and_is_deterministic() {
    assert!(track_case((3.4, -2.1), 32.0));
}

/// Large displacement: the coarse stage fails in a part of the image and the pre-warp rule is
/// exercised; the reference pack is the CPU's (the GPU one's rounding is amplified at one spot).
#[test]
fn track_pair_with_a_large_displacement_matches_the_cpu() {
    assert!(track_case((91.7, -67.3), 128.0));
}

#[test]
fn tiled_refinement_matches_the_cpu() {
    // wider than TILE = 1024: track_pair_on takes the tiled path
    let gpu = gpu();
    let (a, b) = speckle_wh(1100, 200, (2.6, -1.3));
    let cfg = Config::default();
    let want = track_pair_on(&CpuEngine, &a, &b, &cfg, None, None).unwrap();
    let got = track_pair_on(&gpu, &a, &b, &cfg, None, None).unwrap();
    assert!(check("1100x200 tiled gpu-cpu", &got, &want));
}

#[test]
fn sequence_matches_the_cpu() {
    let gpu = gpu();
    let n = 128;
    let c = n as f64 / 2.0;
    let p = synth::make_pattern(n, n, 3, 40.0);
    let frames: Vec<Image<f32>> = std::iter::once(synth::render(&p, n, n, None))
        .chain((1..=2).map(|k| {
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
    let got = run(&gpu);
    assert_eq!(got.len(), want.len());
    let mut ok = true;
    for ((k, g), (_, w)) in got.iter().zip(&want) {
        ok &= check(&format!("sequence frame {k}"), g, w);
    }
    assert!(ok);
}

/// `seq_mode = "hybrid"` on the GPU engine against the CPU engine, on the frames of the hybrid
/// parity fixture (reference moves, fallback, gate, re-acquisition): the two-tier criterion of
/// `check` per frame, and the carried-frames map `fb` equal on all but a handful of pixels.
#[test]
fn hybrid_sequence_matches_the_cpu() {
    let gpu = gpu();
    let (s, v) = fixture("hyb_frames");
    let (n, h, w) = (s[0], s[1], s[2]);
    let frames: Vec<Image<f32>> = (0..n)
        .map(|k| {
            let px = &v[k * h * w..(k + 1) * h * w];
            Image::from_vec(w, h, px.iter().map(|&x| x as f32).collect())
        })
        .collect();
    let cfg = Config {
        seq_mode: "hybrid".to_string(),
        ref_update_conf: 0.8,
        ..Config::default()
    };
    let run = |e: &dyn FineEngine| -> Vec<(usize, TrackResult)> {
        track_sequence_on(e, frames.clone().into_iter(), &cfg, None)
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let want = run(&CpuEngine);
    let got = run(&gpu);
    assert_eq!(got.len(), want.len());
    let mut ok = true;
    for ((k, g), (_, w_)) in got.iter().zip(&want) {
        // frames with only a few valid pixels have no strain to compare (a NaN metric of `check`,
        // not a difference): the displacement and confidence tiers decide
        let r = check(&format!("hybrid frame {k}"), g, w_);
        let (ux, _) = rms_max(&g.ux.data, &w_.ux.data);
        let (uy, _) = rms_max(&g.uy.data, &w_.uy.data);
        let (cf, _) = rms_max(&g.conf.data, &w_.conf.data);
        let agree =
            g.ux.data
                .iter()
                .zip(&w_.ux.data)
                .filter(|(a, b)| a.is_finite() == b.is_finite())
                .count() as f64
                / g.ux.data.len() as f64;
        let few = g.ux.data.iter().filter(|v| v.is_finite()).count() < 400;
        ok &= (r || few) && ux < 1e-4 && uy < 1e-4 && cf < 1e-4 && agree >= 0.999;
        let (fg, fw) = (g.fb.as_ref().unwrap(), w_.fb.as_ref().unwrap());
        let same = fg.data.iter().zip(&fw.data).filter(|(a, b)| a == b).count() as f64
            / fg.data.len() as f64;
        println!(
            "hybrid frame {k}: fb equal {same:.5}, reference {:?}/{:?}",
            g.timing.reference, w_.timing.reference
        );
        ok &= same >= 0.99 && g.timing.reference == w_.timing.reference;
    }
    assert!(ok);
}

#[test]
fn the_repository_fixtures_match_the_cpu() {
    let gpu = gpu();
    let (a, b) = (image("pair_a"), image("pair_b"));
    let (s, v) = fixture("pair_roi");
    let roi = Image::from_vec(s[1], s[0], v.iter().map(|&x| x > 0.5).collect());
    let cfg = Config::default();
    let want = track_pair_on(&CpuEngine, &a, &b, &cfg, Some(&roi), None).unwrap();
    let got = track_pair_on(&gpu, &a, &b, &cfg, Some(&roi), None).unwrap();
    let mut ok = check("pair_a/b + roi gpu-cpu", &got, &want);
    let want = track_pair_on(&CpuEngine, &a, &b, &cfg, None, None).unwrap();
    let got = track_pair_on(&gpu, &a, &b, &cfg, None, None).unwrap();
    ok &= check("pair_a/b gpu-cpu", &got, &want);
    let (ra, rb) = (image("refine_a"), image("refine_b"));
    let (ux, uy) = (image("refine_init_ux"), image("refine_init_uy"));
    for order in [1, 2] {
        let cfg = Config {
            shape_order: order,
            ..Config::default()
        };
        let want = CpuEngine
            .refine_raw(&ra, &rb, (0, 0), &ux, &uy, &cfg)
            .unwrap();
        let got = gpu.refine_raw(&ra, &rb, (0, 0), &ux, &uy, &cfg).unwrap();
        let (x, y, m, _) = raw_diff(&got, &want);
        println!("refine fixture order {order}: ux {x:.2e} uy {y:.2e} max {m:.2e}");
        ok &= raw_ok(&got, &want, false);
    }
    assert!(ok);
}

#[test]
fn check_window_refuses_what_does_not_fit() {
    let gpu = gpu();
    let cfg = Config::default();
    assert!(gpu.check_window(1152, 1152, &cfg).is_ok());
    let err = gpu.check_window(40_000, 40_000, &cfg).unwrap_err();
    println!("{err}");
    assert!(err.0.contains("more than the device allows"));
}

/// Wall time of `track_pair` at 512^2 (run with `--ignored --nocapture`; the load matters).
#[test]
#[ignore = "timing"]
fn track_pair_wall_time() {
    let gpu = gpu();
    let (a, b) = speckle(512, (3.4, -2.1));
    let cfg = Config::default();
    track_pair_on(&gpu, &a, &b, &cfg, None, None).unwrap(); // warm-up: pipelines, bank
    for (name, e) in [("cpu", &CpuEngine as &dyn FineEngine), ("gpu", &gpu)] {
        let t = std::time::Instant::now();
        track_pair_on(e, &a, &b, &cfg, None, None).unwrap();
        println!(
            "track_pair 512^2 {name}: {:.2} s",
            t.elapsed().as_secs_f64()
        );
    }
}

/// Windows whose second image differs in size every time (as on the tiled and sequence paths)
/// must not grow the buffer pool without bound.
#[test]
fn the_buffer_pool_stays_bounded_over_windows_of_different_sizes() {
    let gpu = gpu();
    let (a, b) = speckle(96, (1.2, -0.8));
    let (ux, uy) = (
        Image::from_vec(96, 96, vec![1.0f32; 96 * 96]),
        Image::from_vec(96, 96, vec![-1.0f32; 96 * 96]),
    );
    let cfg = Config {
        outer_iters: 1,
        inner_iters: 1,
        ..Config::default()
    };
    let mut first = 0;
    for k in 0..12usize {
        // a different crop of the second image each time
        let (w2, h2) = (96 + 7 * k, 90 + 11 * k);
        let mut c = Image::<f32>::new(w2, h2);
        for y in 0..h2 {
            for x in 0..w2 {
                c.data[y * w2 + x] = b.data[(y % 96) * 96 + (x % 96)];
            }
        }
        gpu.refine_raw(&a, &c, (0, 0), &ux, &uy, &cfg).unwrap();
        let pooled = gpu.pooled_bytes();
        println!("window {k}: i2 {w2}x{h2}, pooled {pooled} bytes");
        if k == 0 {
            first = pooled;
        }
        assert!(
            pooled <= 2 * first.max(1) + (1 << 20),
            "pool grew: {pooled}"
        );
    }
}

/// The GPU result is bit-for-bit the same as the one recorded before an internal-only change.
/// Record with OPTFLOW_RECORD_BITS=1 (writes gpu-bits-512.bin in CARGO_TARGET_TMPDIR), then
/// compare. With no record (a fresh target directory) and no OPTFLOW_RECORD_BITS the test
/// compares nothing: it prints a SKIP line, writes the record and returns.
#[test]
fn internal_changes_keep_the_gpu_bits() {
    let (a, b) = speckle(512, (1.3, -0.7));
    let cfg = Config::default();
    let r = track_pair_on(&gpu(), &a, &b, &cfg, None, None).unwrap();
    let bits: Vec<u32> = [&r.ux, &r.uy, &r.conf]
        .iter()
        .flat_map(|im| im.data.iter().map(|v| v.to_bits()))
        .collect();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu-bits-512.bin");
    let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let record = std::env::var_os("OPTFLOW_RECORD_BITS").is_some();
    if record || !path.exists() {
        if !record {
            eprintln!(
                "SKIP: no recorded GPU bits at {}; record with OPTFLOW_RECORD_BITS=1",
                path.display()
            );
        }
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("recorded {}", path.display());
        return;
    }
    assert!(std::fs::read(&path).unwrap() == bytes, "GPU bits changed");
}

/// A window refined with its reference pack made ahead (`prepare_extra`, as the tiled driver's
/// prefetch does) gives the bits of `refine_raw`; `prepare_extra` does not wait for, or disturb,
/// a refinement running on another thread (it never takes the engine's lock).
#[test]
fn a_prepared_reference_pack_gives_the_bits_of_refine_raw() {
    let gpu = gpu();
    let (a, b) = speckle_wh(300, 220, (1.3, -0.7));
    let (ux, uy) = (
        Image::from_vec(300, 220, vec![1.0f32; 300 * 220]),
        Image::from_vec(300, 220, vec![-0.5f32; 300 * 220]),
    );
    let cfg = Config::default();
    let want = gpu.refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg).unwrap();
    assert_eq!(gpu.reference_packs(), (1, 0), "refine_raw makes its pack");
    // prepare while the same window is refined on another thread, a few times over
    let (extras, during) = std::thread::scope(|sc| {
        let busy = sc.spawn(|| {
            (0..3)
                .map(|_| gpu.refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg).unwrap())
                .collect::<Vec<_>>()
        });
        let extras: Vec<_> = (0..3).map(|_| gpu.prepare_extra(&a, &cfg)).collect();
        (extras, busy.join().unwrap())
    });
    let f64bits = |im: &Image<f64>| im.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let same = |got: &RawFlow, what: &str| {
        assert_eq!(f64bits(&got.ux), f64bits(&want.ux), "{what}: ux");
        assert_eq!(f64bits(&got.uy), f64bits(&want.uy), "{what}: uy");
        assert_eq!(
            f64bits(&got.energy),
            f64bits(&want.energy),
            "{what}: energy"
        );
        assert_eq!(f64bits(&got.resid), f64bits(&want.resid), "{what}: resid");
    };
    for got in &during {
        same(got, "refine_raw while preparing");
    }
    assert_eq!(gpu.reference_packs(), (4, 0));
    for extra in extras {
        assert!(
            extra.is_some(),
            "the GPU engine prepares the reference pack"
        );
        let p = Prepared {
            i1: a.clone(),
            i2: b.clone(),
            offset: (0, 0),
            ux: ux.clone(),
            uy: uy.clone(),
            extra,
        };
        same(&gpu.refine_prepared(p, &cfg).unwrap(), "refine_prepared");
    }
    // the three prepared packs were used, none made again
    assert_eq!(gpu.reference_packs(), (4, 3));
    // an extra of another type is ignored
    let p = Prepared {
        i1: a.clone(),
        i2: b.clone(),
        offset: (0, 0),
        ux: ux.clone(),
        uy: uy.clone(),
        extra: Some(Box::new(7u8)),
    };
    same(&gpu.refine_prepared(p, &cfg).unwrap(), "foreign extra");
    assert_eq!(
        gpu.reference_packs(),
        (5, 3),
        "a foreign extra makes the pack"
    );
}

/// `release_caches` empties the pool and drops the banks; a window refined afterwards gives the
/// bits of the first one (the banks and buffers are made again).
#[test]
fn releasing_the_caches_keeps_the_bits_and_empties_the_pool() {
    let gpu = gpu();
    let (a, b) = speckle_wh(300, 220, (1.3, -0.7));
    let (ux, uy) = (
        Image::from_vec(300, 220, vec![1.0f32; 300 * 220]),
        Image::from_vec(300, 220, vec![-0.5f32; 300 * 220]),
    );
    let cfg = Config::default();
    let f64bits = |im: &Image<f64>| im.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let run = || {
        let r = gpu.refine_raw(&a, &b, (0, 0), &ux, &uy, &cfg).unwrap();
        [&r.ux, &r.uy, &r.energy, &r.resid].map(f64bits)
    };
    let want = run();
    assert!(gpu.pooled_bytes() > 0, "a refinement leaves buffers pooled");
    gpu.release_caches();
    assert_eq!(gpu.pooled_bytes(), 0);
    assert_eq!(run(), want, "refined again after the release");
    gpu.release_caches();
    assert_eq!(gpu.pooled_bytes(), 0);
}

/// Two lanes (two tiles refined at once, the default) give the bits of one, on a pair of three
/// tiles and on a sequence of two-tile frames (the lanes and their banks reused across pairs).
#[test]
fn two_lanes_give_the_bits_of_one() {
    let (one, two) = (
        GpuEngine::with_lanes(1).expect("a Metal device"),
        GpuEngine::with_lanes(2).expect("a Metal device"),
    );
    assert_eq!((one.lanes(), two.lanes()), (1, 2));
    assert_eq!(GpuEngine::with_lanes(0).unwrap().lanes(), 1);
    assert_eq!(GpuEngine::with_lanes(5).unwrap().lanes(), 2);
    let cfg = Config::default();
    let result_bits = |r: &TrackResult| -> Vec<Vec<u32>> {
        [&r.ux, &r.uy, &r.conf].iter().map(|im| bits(im)).collect()
    };
    let (a, b) = speckle_wh(2100, 300, (2.6, -1.3));
    let pair = |e: &GpuEngine| result_bits(&track_pair_on(e, &a, &b, &cfg, None, None).unwrap());
    assert!(pair(&one) == pair(&two), "pair: two lanes changed the bits");

    let (w, h) = (1100, 200);
    let p = synth::make_pattern(w, h, 3, 40.0);
    let c = (w as f64 / 2.0, h as f64 / 2.0);
    let frames: Vec<Image<f32>> = std::iter::once(synth::render(&p, w, h, None))
        .chain((1..=2).map(|k| {
            let k = f64::from(k);
            let f = synth::uniform_strain(0.002 * k, -0.0006 * k, c, (1.0 * k, 0.5 * k));
            synth::render(&p, w, h, Some(&f))
        }))
        .collect();
    let seq = |e: &GpuEngine| -> Vec<Vec<Vec<u32>>> {
        track_sequence_on(e, frames.clone().into_iter(), &cfg, None)
            .map(|r| result_bits(&r.unwrap().1))
            .collect()
    };
    let (s1, s2) = (seq(&one), seq(&two));
    assert_eq!(s1.len(), 2);
    assert!(s1 == s2, "sequence: two lanes changed the bits");
}

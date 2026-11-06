//! Parity of `track::track_pair` and `track::track_sequence` with the Python reference
//! (`python/optflow_ref/track.py`). The pair has an ROI and a crack (a stripe whose content
//! differs between the images), so the guard is active; the sequence forces a reference update
//! at every frame. Run with `-- --nocapture` to see the measured differences.
mod common;
use common::synth;
use common::{bits, on_one_thread};
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::tile::refine_tiled;
use optflow_core::track::{track_pair, track_sequence, TrackResult};

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    assert_eq!(shape.len(), 2, "{name}: not a 2-D array");
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

fn mask(name: &str) -> Image<bool> {
    let (shape, v) = common::load_f64(name);
    assert_eq!(shape.len(), 2, "{name}: not a 2-D array");
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x > 0.5).collect())
}

fn finite_count(a: &[f32]) -> usize {
    a.iter().filter(|v| v.is_finite()).count()
}

#[test]
fn track_pair_with_roi_and_crack_matches_python() {
    let (a, b, roi) = (img_f32("pair_a"), img_f32("pair_b"), mask("pair_roi"));
    let r = track_pair(&a, &b, &Config::default(), Some(&roi), None).unwrap();
    let (w, h) = (a.w, a.h);
    for im in [
        &r.ux,
        &r.uy,
        &r.conf,
        &r.raw_ux,
        &r.raw_uy,
        &r.strain.exx_gl,
    ] {
        assert_eq!((im.w, im.h), (w, h));
    }
    let (_, ux_py) = common::load_f64("pair_ux");
    let (_, uy_py) = common::load_f64("pair_uy");
    let (_, conf_py) = common::load_f64("pair_conf");
    let (_, exx_py) = common::load_f64("pair_Exx");
    let (_, gmax_py) = common::load_f64("pair_gmax");

    let agree = common::nan_agreement(&r.ux.data, &ux_py, None);
    let rms_ux = common::rms_diff_common(&r.ux.data, &ux_py, None);
    let rms_uy = common::rms_diff_common(&r.uy.data, &uy_py, None);
    let rms_exx = common::rms_diff_common(&r.strain.exx_gl.data, &exx_py, None);
    let rms_gmax = common::rms_diff_common(&r.strain.gmax.data, &gmax_py, None);
    let rms_conf = common::rms_diff(&r.conf.data, &conf_py, None);
    let n_valid = finite_count(&r.ux.data);
    println!(
        "pair: nan_agreement {agree:.5}  rms ux {rms_ux:.3e} uy {rms_uy:.3e}  Exx {rms_exx:.3e}  \
         gmax {rms_gmax:.3e}  conf {rms_conf:.3e}  ({n_valid} valid pixels of {})",
        w * h
    );
    // the valid pixels start B + guard = 28 px from the border (and the ROI and the crack take
    // more); measured 4885 of 25600, the bound is 70 % of it
    assert!(n_valid > 3400, "too few valid pixels: {n_valid}");
    assert!(agree >= 0.995, "nan_agreement {agree}");
    assert!(
        rms_ux < 1e-3 && rms_uy < 1e-3,
        "rms ux {rms_ux} uy {rms_uy}"
    );
    assert!(rms_exx < 1e-5, "rms Exx {rms_exx}");
    assert!(rms_gmax < 1e-5, "rms gmax {rms_gmax}");
    assert!(rms_conf < 1e-3, "rms conf {rms_conf}");
    // the ROI exterior is invalid, and the unguarded confidence is not NaN
    for i in 0..w * h {
        assert!(roi.data[i] || r.ux.data[i].is_nan());
        assert!(r.conf.data[i].is_finite());
    }
    assert!(r.timing.reference.is_none() && r.timing.mean_conf.is_none());
}

fn sequence_frames() -> Vec<Image<f32>> {
    let (shape, v) = common::load_f64("seq_frames");
    assert_eq!(shape.len(), 3);
    let (n, h, w) = (shape[0], shape[1], shape[2]);
    (0..n)
        .map(|k| {
            Image::from_vec(
                w,
                h,
                v[k * h * w..(k + 1) * h * w]
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn track_sequence_with_forced_reference_updates_matches_python() {
    let cfg = Config {
        ref_update_conf: 2.0,
        ..Config::default()
    };
    let frames = sequence_frames();
    let (h, w) = (frames[0].h, frames[0].w);
    let (shape, ux_py) = common::load_f64("seq_ux");
    let (_, uy_py) = common::load_f64("seq_uy");
    assert_eq!(shape[0], frames.len() - 1);
    let out: Vec<(usize, TrackResult)> = track_sequence(frames.into_iter(), &cfg, None)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(out.iter().map(|(k, _)| *k).collect::<Vec<_>>(), [1, 2, 3]);
    let want_ref: Vec<usize> = common::meta()["seq_reference"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let got_ref: Vec<usize> = out
        .iter()
        .map(|(_, r)| r.timing.reference.unwrap())
        .collect();
    assert_eq!(got_ref, want_ref);
    for (k, r) in &out {
        let lo = (k - 1) * h * w;
        let (wx, wy) = (&ux_py[lo..lo + h * w], &uy_py[lo..lo + h * w]);
        let agree = common::nan_agreement(&r.ux.data, wx, None);
        let rms_ux = common::rms_diff(&r.ux.data, wx, None);
        let rms_uy = common::rms_diff(&r.uy.data, wy, None);
        println!(
            "seq frame {k}: reference {:?}  nan_agreement {agree:.5}  rms ux {rms_ux:.3e}  \
             uy {rms_uy:.3e}  mean_conf {:.4}  valid {}",
            r.timing.reference,
            r.timing.mean_conf.unwrap(),
            finite_count(&r.ux.data)
        );
        // measured 4970, 4899 and 4690 of 16384 (the valid pixels start 28 px from the border):
        // the bound h*w/4 = 4096 is 87 % of the smallest
        assert!(
            finite_count(&r.ux.data) > h * w / 4,
            "frame {k}: too few valid pixels"
        );
        assert!(
            rms_ux < 1e-3 && rms_uy < 1e-3,
            "frame {k}: rms {rms_ux} {rms_uy}"
        );
    }
}

/// The link-confidence (C1) fixture: a foreign block at rows and columns 56..72 in frames 1 to 3,
/// a translation of (2k, k) px, the reference moved on at every frame. The stored expectation is
/// that of frames 2 and 3, where only the min over the legs keeps the block's invalidity.
#[test]
fn track_sequence_keeps_the_invalidity_of_the_frame0_link_like_python() {
    let cfg = Config {
        ref_update_conf: 2.0, // the reference moves on at every frame
        ..Config::default()
    };
    let (shape, v) = common::load_f64("seqc1_frames");
    let (n, h, w) = (shape[0], shape[1], shape[2]);
    let frames: Vec<Image<f32>> = (0..n)
        .map(|k| {
            let px = &v[k * h * w..(k + 1) * h * w];
            Image::from_vec(w, h, px.iter().map(|&x| x as f32).collect())
        })
        .collect();
    let (_, ux_py) = common::load_f64("seqc1_ux");
    let (_, uy_py) = common::load_f64("seqc1_uy");
    let (_, conf_py) = common::load_f64("seqc1_conf");
    let out: Vec<(usize, TrackResult)> = track_sequence(frames.into_iter(), &cfg, None)
        .collect::<Result<_, _>>()
        .unwrap();
    let refs: Vec<usize> = out
        .iter()
        .map(|(_, r)| r.timing.reference.unwrap())
        .collect();
    assert_eq!(refs, [0, 1, 2]);
    // frame-0 pixels that move into the block on the 0 -> 1 leg (translation (2, 1))
    let gone: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (56..72).contains(&(y + 1)) && (56..72).contains(&(x + 2))
        })
        .collect();
    for (k, r) in out.iter().filter(|(k, _)| *k >= 2) {
        let lo = (k - 2) * h * w;
        let (wx, wy, wc) = (
            &ux_py[lo..lo + h * w],
            &uy_py[lo..lo + h * w],
            &conf_py[lo..lo + h * w],
        );
        let agree = common::nan_agreement(&r.ux.data, wx, None);
        let rms_ux = common::rms_diff(&r.ux.data, wx, None);
        let rms_uy = common::rms_diff(&r.uy.data, wy, None);
        let rms_conf = common::rms_diff(&r.conf.data, wc, None);
        let nan_in_gone = (0..w * h)
            .filter(|&i| gone[i] && r.ux.data[i].is_nan())
            .count() as f64
            / gone.iter().filter(|&&g| g).count() as f64;
        println!(
            "C1 frame {k}: nan_agreement {agree:.5}  rms ux {rms_ux:.3e} uy {rms_uy:.3e} \
             conf {rms_conf:.3e}  NaN share of the pixels that moved into the block {nan_in_gone:.3}"
        );
        assert!(agree >= 0.995, "frame {k}: nan_agreement {agree}");
        assert!(
            rms_ux < 1e-3 && rms_uy < 1e-3,
            "frame {k}: rms {rms_ux} {rms_uy}"
        );
        assert!(rms_conf < 1e-3, "frame {k}: conf rms {rms_conf}");
        // later legs match the block with itself and look confident: only the min over the legs
        // keeps the first leg's invalidity
        assert!(nan_in_gone >= 0.95, "frame {k}: NaN share {nan_in_gone}");
    }
}

fn seqc1_frames() -> Vec<Image<f32>> {
    let (shape, v) = common::load_f64("seqc1_frames");
    let (n, h, w) = (shape[0], shape[1], shape[2]);
    (0..n)
        .map(|k| {
            let px = &v[k * h * w..(k + 1) * h * w];
            Image::from_vec(w, h, px.iter().map(|&x| x as f32).collect())
        })
        .collect()
}

/// Runs the sequence and compares the reference and the mean confidence of every frame with the
/// `meta.json` entry `key`/`name`; `tol(k)` is the tolerance of the mean of frame `k + 1`. The mean
/// of a frame is the one of the leg that produced it.
fn check_mean_conf(
    key: &str,
    name: &str,
    frames: Vec<Image<f32>>,
    cfg: &Config,
    tol: impl Fn(usize) -> f64,
) {
    let want = &common::meta()[key][name];
    let want_ref: Vec<usize> = want["reference"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_u64().unwrap() as usize)
        .collect();
    let want_mean: Vec<f64> = want["mean_conf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.as_f64().unwrap())
        .collect();
    let out: Vec<(usize, TrackResult)> = track_sequence(frames.into_iter(), cfg, None)
        .collect::<Result<_, _>>()
        .unwrap();
    let got_ref: Vec<usize> = out
        .iter()
        .map(|(_, r)| r.timing.reference.unwrap())
        .collect();
    let got_mean: Vec<f64> = out
        .iter()
        .map(|(_, r)| r.timing.mean_conf.unwrap())
        .collect();
    println!("{key}/{name}: reference {got_ref:?} (python {want_ref:?})");
    println!("{key}/{name}: mean_conf {got_mean:.8?} (python {want_mean:.8?})");
    assert_eq!(got_ref, want_ref, "{key}/{name}: reference");
    assert_eq!(got_mean.len(), want_mean.len(), "{key}/{name}: frames");
    for (k, (g, p)) in got_mean.iter().zip(&want_mean).enumerate() {
        assert!(
            (g - p).abs() < tol(k),
            "{key}/{name} frame {}: mean_conf {g} vs Python {p}",
            k + 1
        );
    }
}

/// The mean confidence that decides the reference update (spec 2.6) is Python's: sequences on the
/// stored frames with the default `ref_update_conf` (`meta.json`: `seq_mean_conf`). "seq" and
/// "seqc1" keep the reference at frame 0 (the mean is taken over the reportable pixels: over all
/// pixels it was 0.45 to 0.55 there, and on "seqc1" under 0.5), so `u0r` is zero there and the
/// means are pinned at 1e-6 (measured: 1e-8). "mixed" (seq 0, seq 1, seqc1 0, seqc1 1: unrelated
/// images in the middle) moves the reference twice.
///
/// After the unrelated pairs of "mixed" the means are pinned at 1e-3 only, from frame 2 on: those
/// legs match nothing and are chaotic. Their raw fields differ by whole pixels between the
/// implementations (f32 here, f64 in Python), and the mean of the next legs is taken at the
/// positions that field composes to; a relative input perturbation of 1e-6 moves Python's own
/// frame 2 and 3 values by 1.5e-7 and 3.0e-6 (reviewer's measurement). The references and the
/// means to 1e-3 still pin the rule (the mean over all pixels would be 0.0074 and 0.675 there).
/// The fractional `u0r` is pinned at 1e-6 by the forced-update test below.
#[test]
fn track_sequence_mean_conf_and_reference_updates_match_python() {
    let seq = sequence_frames();
    let seqc1 = seqc1_frames();
    let mixed = vec![
        seq[0].clone(),
        seq[1].clone(),
        seqc1[0].clone(),
        seqc1[1].clone(),
    ];
    let cfg = Config::default();
    check_mean_conf("seq_mean_conf", "seq", seq, &cfg, |_| 1e-6);
    check_mean_conf("seq_mean_conf", "seqc1", seqc1, &cfg, |_| 1e-6);
    check_mean_conf("seq_mean_conf", "mixed", mixed, &cfg, |k| {
        if k > 0 {
            1e-3
        } else {
            1e-6
        }
    });
}

/// The same frames with `ref_update_conf = 2`: the reference moves on at every frame, so from
/// frame 2 on the mean confidence of a leg is taken at `X + u0r(X)` with a non-zero, fractional
/// `u0r` (the displacement of the earlier legs, which match well: the legs agree with Python to
/// 8e-7 px). Pinned at 1e-6 together with the reference (`meta.json`: `seq_mean_conf_forced`).
#[test]
fn track_sequence_mean_conf_at_fractional_positions_matches_python() {
    let cfg = Config {
        ref_update_conf: 2.0,
        ..Config::default()
    };
    check_mean_conf(
        "seq_mean_conf_forced",
        "seq",
        sequence_frames(),
        &cfg,
        |_| 1e-6,
    );
    check_mean_conf(
        "seq_mean_conf_forced",
        "seqc1",
        seqc1_frames(),
        &cfg,
        |_| 1e-6,
    );
}

/// Python's `test_sequence_without_roi_does_not_update_the_reference_on_an_ideal_translation`:
/// 128 x 128 frames, an ideal translation of (5, 2.5) px per frame, no ROI, the default config.
/// The 16 px border strips and the matches that leave image 2 have confidence 0 by the border
/// rule (spec 2.4) and used to pull the mean under `ref_update_conf = 0.5`, so that the reference
/// moved on at every frame.
#[test]
fn track_sequence_without_roi_does_not_update_the_reference_on_an_ideal_translation() {
    let n = 128;
    let pat = synth::make_pattern(n, n, 3, 80.0);
    let frames: Vec<Image<f32>> = (0..3)
        .map(|k| {
            let f = synth::uniform_strain(0.0, 0.0, (0.0, 0.0), (5.0 * k as f64, 2.5 * k as f64));
            synth::render(&pat, n, n, Some(&f))
        })
        .collect();
    let out: Vec<(usize, TrackResult)> =
        track_sequence(frames.into_iter(), &Config::default(), None)
            .collect::<Result<_, _>>()
            .unwrap();
    let refs: Vec<usize> = out
        .iter()
        .map(|(_, r)| r.timing.reference.unwrap())
        .collect();
    let means: Vec<f64> = out
        .iter()
        .map(|(_, r)| r.timing.mean_conf.unwrap())
        .collect();
    println!("ideal translation: reference {refs:?}, mean_conf {means:.4?}");
    assert_eq!(refs, [0, 0]);
    assert!(means.iter().all(|&m| m > 0.9), "mean_conf {means:?}");
}

#[test]
fn track_pair_result_does_not_depend_on_the_thread_count() {
    let (a, b, roi) = (img_f32("pair_a"), img_f32("pair_b"), mask("pair_roi"));
    let run = || track_pair(&a, &b, &Config::default(), Some(&roi), None).unwrap();
    let many = run();
    let one = on_one_thread(run);
    for (x, y) in [
        (&many.ux, &one.ux),
        (&many.uy, &one.uy),
        (&many.conf, &one.conf),
        (&many.strain.exx_gl, &one.strain.exx_gl),
        (&many.strain.gmax, &one.strain.gmax),
        (&many.raw_ux, &one.raw_ux),
    ] {
        assert_eq!(bits(x), bits(y));
    }
}

/// The tiled fine stage (9 tiles of 128 px on a 384 x 384 pair, each tile a `refine_raw` run and
/// the confidence formed from the joined cores) is bit-identical on one thread and on the
/// default pool.
#[test]
fn refine_tiled_result_does_not_depend_on_the_thread_count() {
    let n = 384;
    let field = synth::uniform_strain(0.01, -0.003, (192.0, 192.0), (3.4, -2.1));
    let pat = synth::make_pattern(n, n, 4, 80.0);
    let (a, b) = (
        synth::render(&pat, n, n, None),
        synth::render(&pat, n, n, Some(&field)),
    );
    let (mut ux, mut uy) = synth::displacement(&field, n, n);
    ux.data.iter_mut().for_each(|v| *v += 0.3);
    uy.data.iter_mut().for_each(|v| *v -= 0.3);
    let cfg = Config::default();
    let run = || refine_tiled(&a, &b, &ux, &uy, &cfg, 128);
    let many = run();
    let one = on_one_thread(run);
    for (x, y) in [
        (&many.ux, &one.ux),
        (&many.uy, &one.uy),
        (&many.conf, &one.conf),
        (&many.resid, &one.resid),
    ] {
        assert_eq!(bits(x), bits(y));
    }
    assert!(
        many.conf.data.iter().any(|&c| c > 0.5),
        "no confident pixel"
    );
}

/// Three frames with a reference update at every frame (so the link confidence, the composition
/// and a pair with a dense guess all run): every output of every frame is bit-identical on one
/// thread and on the default pool.
#[test]
fn track_sequence_result_does_not_depend_on_the_thread_count() {
    let cfg = Config {
        ref_update_conf: 2.0,
        ..Config::default()
    };
    let frames: Vec<Image<f32>> = sequence_frames().into_iter().take(3).collect();
    let run = || -> Vec<(usize, TrackResult)> {
        track_sequence(frames.clone().into_iter(), &cfg, None)
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let many = run();
    let one = on_one_thread(run);
    assert_eq!(many.len(), 2);
    assert_eq!(
        many.iter()
            .map(|(_, r)| r.timing.reference)
            .collect::<Vec<_>>(),
        [Some(0), Some(1)]
    );
    for ((k, m), (_, o)) in many.iter().zip(&one) {
        for (x, y) in [
            (&m.ux, &o.ux),
            (&m.uy, &o.uy),
            (&m.conf, &o.conf),
            (&m.strain.exx_gl, &o.strain.exx_gl),
            (&m.strain.gmax, &o.strain.gmax),
            (&m.raw_ux, &o.raw_ux),
            (&m.raw_uy, &o.raw_uy),
        ] {
            assert_eq!(bits(x), bits(y), "frame {k}");
        }
        assert_eq!(m.timing.mean_conf, o.timing.mean_conf, "frame {k}");
    }
}

/// With a good dense guess (the previous frame's field, as `track_sequence` passes it) the coarse
/// stage searches only for a correction and the fine stage starts from guess + correction; the
/// answer must be the cold start's wherever both are valid.
#[test]
fn track_pair_with_a_dense_init_agrees_with_the_cold_start() {
    let cfg = Config::default();
    let f = sequence_frames();
    let p01 = track_pair(&f[0], &f[1], &cfg, None, None).unwrap();
    let cold = track_pair(&f[0], &f[2], &cfg, None, None).unwrap();
    let warm = track_pair(&f[0], &f[2], &cfg, None, Some((&p01.raw_ux, &p01.raw_uy))).unwrap();
    let to64 = |im: &Image<f32>| im.data.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>();
    let agree = common::nan_agreement(&warm.ux.data, &to64(&cold.ux), None);
    let rms_x = common::rms_diff_common(&warm.ux.data, &to64(&cold.ux), None);
    let rms_y = common::rms_diff_common(&warm.uy.data, &to64(&cold.uy), None);
    println!(
        "init vs cold: nan_agreement {agree:.5}  rms ux {rms_x:.3e} uy {rms_y:.3e}  valid cold {} warm {}",
        finite_count(&cold.ux.data),
        finite_count(&warm.ux.data)
    );
    // 128 x 128 frames: the valid pixels start B + guard = 28 px from the border, 72 x 72 = 5184 at
    // most; measured 4760, the bound is 69 % of it
    assert!(finite_count(&cold.ux.data) > 3300);
    assert!(agree >= 0.995, "nan_agreement {agree}");
    assert!(rms_x < 1e-3 && rms_y < 1e-3, "rms {rms_x} {rms_y}");
}

/// Python's `test_sequence_output_is_guarded_on_the_frame0_grid`: the ROI exterior is NaN in
/// every frame's output and the guard erodes the ROI by its distance (12.2 px by default).
#[test]
fn track_sequence_output_is_guarded_on_the_frame0_grid() {
    let cfg = Config::default();
    let frames = sequence_frames();
    let (w, h) = (frames[0].w, frames[0].h);
    let (lo, hi) = (24usize, 104usize);
    let roi = Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| (lo..hi).contains(&(i % w)) && (lo..hi).contains(&(i / w)))
            .collect(),
    );
    let out: Vec<(usize, TrackResult)> = track_sequence(frames.into_iter(), &cfg, Some(&roi))
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(out.iter().map(|(k, _)| *k).collect::<Vec<_>>(), [1, 2, 3]);
    let guard = cfg.edge_guard_px();
    assert!((6.0..21.0).contains(&guard));
    for (k, r) in &out {
        for i in 0..w * h {
            assert!(roi.data[i] || (r.ux.data[i].is_nan() && r.uy.data[i].is_nan()));
        }
        // a pixel d inside the ROI edge is NaN for d + 1 <= guard, the pixels deeper in are valid
        for (d, finite) in [(5usize, false), (20, true)] {
            let c = h / 2;
            for (x, y) in [(lo + d, c), (hi - 1 - d, c), (c, lo + d), (c, hi - 1 - d)] {
                assert_eq!(
                    r.ux.data[y * w + x].is_finite(),
                    finite,
                    "frame {k}, {d} px inside the edge at ({x}, {y})"
                );
            }
        }
        assert!(r.ux.get(w / 2, h / 2).is_finite());
        assert_eq!(
            r.timing.reference,
            Some(0),
            "frame {k}: no reference update expected"
        );
    }
}

// ---- seq_mode = "hybrid" (v3)

fn hybrid_frames() -> Vec<Image<f32>> {
    let (shape, v) = common::load_f64("hyb_frames");
    let (n, h, w) = (shape[0], shape[1], shape[2]);
    (0..n)
        .map(|k| {
            let px = &v[k * h * w..(k + 1) * h * w];
            Image::from_vec(w, h, px.iter().map(|&x| x as f32).collect())
        })
        .collect()
}

/// A translation with a foreign block that frames 2 to 4 show at one fixed position, tracked in
/// `seq_mode = "hybrid"` (the reference moves, pixels are carried by the adjacent leg, the gate
/// rejects some, the direct leg re-acquires others) against the Python reference: reference,
/// mean confidence and the shares of `timing`, the carried-frames map `fb` and the displacement
/// (and for variant A the confidence). Variant B has `fb_max = 2`, so the recovery without a
/// fallback value is used.
fn check_hybrid(name: &str, fb_max: i64, with_conf: bool) {
    let cfg = Config {
        seq_mode: "hybrid".to_string(),
        ref_update_conf: 0.8,
        fb_max,
        ..Config::default()
    };
    let frames = hybrid_frames();
    let (h, w) = (frames[0].h, frames[0].w);
    let (shape, ux_py) = common::load_f64(&format!("hyb{name}_ux"));
    let (_, fb_py) = common::load_f64(&format!("hyb{name}_fb"));
    assert_eq!(shape[0], frames.len() - 1);
    let want = common::meta()["hybrid"][name].as_array().unwrap().clone();
    let out: Vec<(usize, TrackResult)> = track_sequence(frames.into_iter(), &cfg, None)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(out.len(), want.len());
    for ((k, r), t) in out.iter().zip(&want) {
        let lo = (k - 1) * h * w;
        let wx = &ux_py[lo..lo + h * w];
        let fb = r.fb.as_ref().expect("hybrid results carry fb");
        let fb_same = (0..h * w)
            .filter(|&i| f64::from(fb.data[i]) == fb_py[lo + i])
            .count() as f64
            / (h * w) as f64;
        let agree = common::nan_agreement(&r.ux.data, wx, None);
        let rms_ux = common::rms_diff_common(&r.ux.data, wx, None);
        let tm = &r.timing;
        println!(
            "hybrid {name} frame {k}: reference {:?}  nan_agreement {agree:.5}  rms ux {rms_ux:.3e}  \
             fb equal {fb_same:.5}  fallback {:.4} (py {:.4})  rejected {:.4} (py {:.4})  \
             reacquired {:.4} (py {:.4})  n_fb_max {:?}  valid {}",
            tm.reference,
            tm.fallback_frac.unwrap(),
            t["fallback_frac"].as_f64().unwrap(),
            tm.fb_gate_rejected.unwrap(),
            t["fb_gate_rejected"].as_f64().unwrap(),
            tm.reacquired.unwrap(),
            t["reacquired"].as_f64().unwrap(),
            tm.n_fb_max,
            finite_count(&r.ux.data)
        );
        assert_eq!(
            tm.reference,
            Some(t["reference"].as_u64().unwrap() as usize),
            "{name} frame {k}: reference"
        );
        assert_eq!(
            tm.n_fb_max,
            Some(t["n_fb_max"].as_i64().unwrap()),
            "{name} frame {k}: n_fb_max"
        );
        assert!(
            (tm.mean_conf.unwrap() - t["mean_conf"].as_f64().unwrap()).abs() < 1e-4,
            "{name} frame {k}: mean_conf"
        );
        for key in ["fallback_frac", "fb_gate_rejected", "reacquired"] {
            let got = match key {
                "fallback_frac" => tm.fallback_frac,
                "fb_gate_rejected" => tm.fb_gate_rejected,
                _ => tm.reacquired,
            }
            .unwrap();
            assert!(
                (got - t[key].as_f64().unwrap()).abs() < 2e-3,
                "{name} frame {k}: {key} {got} vs {}",
                t[key]
            );
        }
        assert!(fb_same >= 0.999, "{name} frame {k}: fb equal {fb_same}");
        assert!(agree >= 0.995, "{name} frame {k}: nan_agreement {agree}");
        assert!(rms_ux < 1e-3, "{name} frame {k}: rms ux {rms_ux}");
        if with_conf {
            let (_, conf_py) = common::load_f64("hybA_conf");
            let rms_conf = common::rms_diff(&r.conf.data, &conf_py[lo..lo + h * w], None);
            assert!(rms_conf < 1e-3, "{name} frame {k}: conf rms {rms_conf}");
        }
    }
}

#[test]
fn hybrid_sequence_matches_python() {
    check_hybrid("A", 10, true);
}

#[test]
fn hybrid_sequence_with_short_fallback_matches_python() {
    check_hybrid("B", 2, false);
}

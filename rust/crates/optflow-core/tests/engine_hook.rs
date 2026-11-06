//! The fine-stage engine hook: the CPU engine through `track_pair_on` / `track_sequence_on` is
//! `track_pair` / `track_sequence` bit for bit, and an engine's error stops the pipeline.
mod common;
use common::{bits, synth};
use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, EngineError, FineEngine};
use optflow_core::image::Image;
use optflow_core::phaseflow::RawFlow;
use optflow_core::track::{
    track_pair, track_pair_on, track_sequence, track_sequence_on, TrackError,
};

fn pair(n: usize) -> (Image<f32>, Image<f32>) {
    let c = n as f64 / 2.0;
    let f = synth::uniform_strain(0.01, -0.003, (c, c), (3.4, -2.1));
    let p = synth::make_pattern(n, n, 1, 32.0);
    (
        synth::render(&p, n, n, None),
        synth::render(&p, n, n, Some(&f)),
    )
}

#[test]
fn cpu_engine_is_track_pair_bit_for_bit() {
    let (a, b) = pair(128);
    let cfg = Config {
        outer_iters: 2,
        ..Config::default()
    };
    let want = track_pair(&a, &b, &cfg, None, None).unwrap();
    let got = track_pair_on(&CpuEngine, &a, &b, &cfg, None, None).unwrap();
    assert_eq!(bits(&got.ux), bits(&want.ux));
    assert_eq!(bits(&got.uy), bits(&want.uy));
    assert_eq!(bits(&got.conf), bits(&want.conf));
    assert_eq!(bits(&got.strain.exx), bits(&want.strain.exx));
}

#[test]
fn cpu_engine_is_track_sequence_bit_for_bit() {
    let (a, b) = pair(96);
    let frames = vec![a.clone(), b.clone(), a];
    let cfg = Config {
        outer_iters: 2,
        ..Config::default()
    };
    let want: Vec<_> = track_sequence(frames.clone().into_iter(), &cfg, None).collect();
    let got: Vec<_> = track_sequence_on(&CpuEngine, frames.into_iter(), &cfg, None).collect();
    assert_eq!(want.len(), got.len());
    assert_eq!(want.len(), 2);
    for (w, g) in want.iter().zip(&got) {
        let ((wk, w), (gk, g)) = (w.as_ref().unwrap(), g.as_ref().unwrap());
        assert_eq!(wk, gk);
        assert_eq!(bits(&g.ux), bits(&w.ux));
        assert_eq!(bits(&g.conf), bits(&w.conf));
    }
}

struct Failing;
impl FineEngine for Failing {
    fn refine_raw(
        &self,
        _: &Image<f32>,
        _: &Image<f32>,
        _: (i64, i64),
        _: &Image<f32>,
        _: &Image<f32>,
        _: &Config,
    ) -> Result<RawFlow, EngineError> {
        Err(EngineError("device lost".into()))
    }
}

#[test]
fn an_engine_error_is_a_track_error() {
    let (a, b) = pair(96);
    let err = track_pair_on(&Failing, &a, &b, &Config::default(), None, None).unwrap_err();
    assert!(
        matches!(err, TrackError::Engine(ref m) if m == "device lost"),
        "{err:?}"
    );
}

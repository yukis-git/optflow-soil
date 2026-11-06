//! Error against a known displacement, grouped by the distance from the image borders (spec 2.4),
//! over the pixels the pipeline reports.
#![allow(dead_code)]

use optflow_core::config::Config;
use optflow_core::edt::{distance_to_invalid, large_components};
use optflow_core::image::Image;

use super::synth::Affine;

/// Band edges [px]: the first band starts at the nearest valid pixel (`B + guard` = 28 px at the
/// defaults), the last is open.
pub const EDGES: [f64; 2] = [32.0, 48.0];

/// The pixels `track_pair` reports for a confidence map without ROI: `conf >= conf_min`, farther
/// than the guard from every large low-confidence component (`edge_guard_min_area`) and at least
/// the guard from the image border (`track::invalid_pixels`, which is private; the `f32`
/// confidence stands in for the `f64` one).
pub fn pipeline_valid(conf: &Image<f32>, cfg: &Config) -> Vec<bool> {
    let (w, h) = (conf.w, conf.h);
    let low = Image::from_vec(
        w,
        h,
        conf.data
            .iter()
            .map(|&c| f64::from(c) < cfg.conf_min)
            .collect(),
    );
    let guard = cfg.edge_guard_px();
    assert!(guard > 0.0, "pipeline_valid: the guard is off");
    let src = large_components(&low, cfg.edge_guard_min_area_px());
    let near_source: Vec<bool> = if src.data.iter().any(|&s| s) {
        let ok = Image::from_vec(w, h, src.data.iter().map(|&s| !s).collect());
        // scipy's distance is the root of an integer: restore it before comparing (track::within)
        distance_to_invalid(&ok)
            .data
            .iter()
            .map(|&d| (f64::from(d) * f64::from(d)).round().sqrt() <= guard)
            .collect()
    } else {
        vec![false; w * h]
    };
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let edge = x.min(y).min(w - 1 - x).min(h - 1 - y);
            !low.data[i] && !near_source[i] && (edge as f64) >= guard
        })
        .collect()
}

/// Per pixel of image 1, the smaller of its distance from image 1's border and the distance of
/// its true match from image 2's border (negative where the match leaves image 2).
pub fn border_distance(field: &Affine, w: usize, h: usize) -> Vec<f64> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let (tx, ty) = field.fwd(x, y);
            let (mx, my) = ((w - 1) as f64, (h - 1) as f64);
            [x, y, mx - x, my - y, tx, ty, mx - tx, my - ty]
                .into_iter()
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

pub struct Band {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
    pub rms: f64,
    pub max: f64,
}

/// `|u - truth|` over the valid pixels (`ux` is read as is: a NaN is not valid), in the bands of
/// `dist` below [`EDGES`] and above.
pub fn bands(
    valid: &[bool],
    ux: &[f32],
    uy: &[f32],
    field: &Affine,
    w: usize,
    dist: &[f64],
) -> Vec<Band> {
    let edges = [f64::NEG_INFINITY, EDGES[0], EDGES[1], f64::INFINITY];
    let mut out: Vec<(usize, f64, f64)> = vec![(0, 0.0, 0.0); 3];
    for i in 0..valid.len() {
        if !valid[i] {
            continue;
        }
        let (tx, ty) = field.displacement((i % w) as f64, (i / w) as f64);
        let e = (f64::from(ux[i]) - tx).hypot(f64::from(uy[i]) - ty);
        let k = (0..3).find(|&k| dist[i] < edges[k + 1]).unwrap();
        out[k].0 += 1;
        out[k].1 += e * e;
        out[k].2 = out[k].2.max(e);
    }
    let first = (0..valid.len())
        .filter(|&i| valid[i])
        .map(|i| dist[i])
        .fold(f64::INFINITY, f64::min);
    out.iter()
        .enumerate()
        .map(|(k, &(n, sq, max))| Band {
            lo: if k == 0 { first } else { edges[k] },
            hi: edges[k + 1],
            count: n,
            rms: (sq / n.max(1) as f64).sqrt(),
            max,
        })
        .collect()
}

/// Prints the table and asserts what the spec states: nothing is reported closer than
/// `B + guard - 1` to a border, and every band has rms <= 1e-3 px and max <= 1e-2 px.
pub fn check_bands(
    label: &str,
    valid: &[bool],
    ux: &[f32],
    uy: &[f32],
    field: &Affine,
    w: usize,
    cfg: &Config,
) {
    let h = valid.len() / w;
    let dist = border_distance(field, w, h);
    let table = bands(valid, ux, uy, field, w, &dist);
    for b in &table {
        println!(
            "{label}: [{:.0}, {}) {:7} px  rms {:.2e}  max {:.2e}",
            b.lo, b.hi, b.count, b.rms, b.max
        );
    }
    let nearest = table[0].lo;
    let floor = 4.0 * cfg.sigma_w + cfg.edge_guard_px() - 1.0;
    assert!(
        nearest >= floor,
        "{label}: a valid pixel {nearest:.2} px from a border (< {floor:.2})"
    );
    for b in &table {
        assert!(b.count > 1000, "{label}: only {} pixels in a band", b.count);
        assert!(
            b.rms <= 1e-3 && b.max <= 1e-2,
            "{label}: band [{:.0}, {}) rms {:.2e} max {:.2e}",
            b.lo,
            b.hi,
            b.rms,
            b.max
        );
    }
}

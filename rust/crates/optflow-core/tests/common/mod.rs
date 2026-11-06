//! Shared helpers for the parity tests: load the Python-written fixtures and compare against them.
#![allow(dead_code)]

use std::path::PathBuf;

use optflow_core::image::Image;
use optflow_core::npy::{self, NpyArray, NpyData};

pub mod bands;
pub mod synth;

/// The bit patterns of an image, for bit-for-bit comparisons.
pub fn bits(im: &Image<f32>) -> Vec<u32> {
    im.data.iter().map(|v| v.to_bits()).collect()
}

/// `f` on a pool of one thread.
pub fn on_one_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(f)
}

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Reads `tests/fixtures/<name>.npy` as parsed by `npy::read`. Panics with the file name on failure.
pub fn load(name: &str) -> NpyArray {
    let path = fixture_dir().join(format!("{name}.npy"));
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    npy::read(&bytes).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// Shape and values of a fixture, widened to `f64` whatever dtype it was stored in.
pub fn load_f64(name: &str) -> (Vec<usize>, Vec<f64>) {
    let NpyArray { shape, data } = load(name);
    let values = match data {
        NpyData::F64(v) => v,
        NpyData::F32(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U8(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U16(v) => v.into_iter().map(f64::from).collect(),
        NpyData::Bool(v) => v.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect(),
    };
    (shape, values)
}

/// Contents of `tests/fixtures/meta.json`.
pub fn meta() -> serde_json::Value {
    let path = fixture_dir().join("meta.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// The `(a, b)` pairs selected by `mask` (all pairs when `None`), widened to `f64`.
fn masked_pairs<'a>(
    a: &'a [f32],
    b: &'a [f64],
    mask: Option<&'a [bool]>,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    assert_eq!(a.len(), b.len(), "length mismatch");
    if let Some(m) = mask {
        assert_eq!(m.len(), a.len(), "mask length mismatch");
    }
    a.iter()
        .zip(b)
        .enumerate()
        .filter(move |(i, _)| mask.is_none_or(|m| m[*i]))
        .map(|(_, (&x, &y))| (f64::from(x), y))
}

fn rms(sum: f64, count: usize) -> f64 {
    if count == 0 {
        f64::NAN
    } else {
        (sum / count as f64).sqrt()
    }
}

/// Strict root-mean-square of `a - b` over the pixels selected by `mask` (all when `None`).
/// Pairs where both values are non-finite are skipped. If exactly one value of a pair is
/// non-finite the result is NaN, so a one-sided NaN or infinity can never pass a `<=` check.
/// NaN too when nothing is compared.
pub fn rms_diff(a: &[f32], b: &[f64], mask: Option<&[bool]>) -> f64 {
    let (mut sum, mut count) = (0.0f64, 0usize);
    for (x, y) in masked_pairs(a, b, mask) {
        match (x.is_finite(), y.is_finite()) {
            (true, true) => {
                let d = x - y;
                sum += d * d;
                count += 1;
            }
            (false, false) => {}
            _ => return f64::NAN,
        }
    }
    rms(sum, count)
}

/// Root-mean-square of `a - b` over the masked pairs where both values are finite, silently
/// skipping every pair with a non-finite value. Use only together with `nan_agreement`, which
/// reports how often the two sides disagree about finiteness. NaN when no pair is counted.
pub fn rms_diff_common(a: &[f32], b: &[f64], mask: Option<&[bool]>) -> f64 {
    let (mut sum, mut count) = (0.0f64, 0usize);
    for (x, y) in masked_pairs(a, b, mask) {
        if x.is_finite() && y.is_finite() {
            let d = x - y;
            sum += d * d;
            count += 1;
        }
    }
    rms(sum, count)
}

/// Fraction of the masked pixels (all when `None`) where `a` and `b` are both finite or both
/// non-finite. NaN when no pixel is selected.
pub fn nan_agreement(a: &[f32], b: &[f64], mask: Option<&[bool]>) -> f64 {
    let (mut agree, mut total) = (0usize, 0usize);
    for (x, y) in masked_pairs(a, b, mask) {
        agree += usize::from(x.is_finite() == y.is_finite());
        total += 1;
    }
    if total == 0 {
        f64::NAN
    } else {
        agree as f64 / total as f64
    }
}

//! Per-frame statistics (the time-series of the earlier CSV export) and the colour range.

use crate::fields::Fields;

/// One row of the time series.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameStats {
    pub k: usize,
    /// Share of the ROI pixels with a valid displacement: the tracking rate.
    pub valid_frac: f64,
    pub mean_conf: f64,
    pub ux_median: f64,
    pub uy_median: f64,
    pub uy_max: f64,
    pub g50: f64,
    pub g95: f64,
    pub g99: f64,
    pub vol_p50: f64,
}

/// numpy's `percentile` (linear interpolation) of an unsorted slice; NaN when empty.
pub fn percentile(values: &mut [f32], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    percentile_sorted(values, q)
}

/// [`percentile`] of a slice that is sorted already; NaN when empty.
fn percentile_sorted(values: &[f32], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let pos = q / 100.0 * (values.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let f = pos - lo as f64;
    values[lo] as f64 * (1.0 - f) + values[hi] as f64 * f
}

/// Statistics of frame `k`. `roi` is the ROI mask (row-major, `w * h`); without one the whole
/// image counts.
pub fn frame_stats(k: usize, f: &Fields, roi: Option<&[bool]>) -> FrameStats {
    let n = f.w * f.h;
    let in_roi = |i: usize| roi.is_none_or(|m| m[i]);
    let (mut roi_n, mut valid_n) = (0usize, 0usize);
    let (mut ux, mut uy, mut g, mut v) = (vec![], vec![], vec![], vec![]);
    let mut conf_sum = 0.0f64;
    for i in 0..n {
        if !in_roi(i) {
            continue;
        }
        roi_n += 1;
        if f.is_valid(i) {
            valid_n += 1;
            // a valid pixel can still lack a strain value (NaN near the edge of the valid area)
            ux.push(f.ux[i]);
            if f.uy[i].is_finite() {
                uy.push(f.uy[i]);
            }
            if f.gmax[i].is_finite() {
                g.push(f.gmax[i]);
            }
            if f.vol[i].is_finite() {
                v.push(f.vol[i]);
            }
            conf_sum += f.conf[i] as f64;
        }
    }
    // a 26-megapixel frame has millions of values here: one sort per list
    g.sort_unstable_by(|a, b| a.total_cmp(b));
    FrameStats {
        k,
        valid_frac: if roi_n > 0 {
            valid_n as f64 / roi_n as f64
        } else {
            f64::NAN
        },
        mean_conf: if valid_n > 0 {
            conf_sum / valid_n as f64
        } else {
            f64::NAN
        },
        ux_median: percentile(&mut ux, 50.0),
        uy_median: percentile(&mut uy, 50.0),
        uy_max: uy.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64,
        g50: percentile_sorted(&g, 50.0),
        g95: percentile_sorted(&g, 95.0),
        g99: percentile_sorted(&g, 99.0),
        vol_p50: percentile(&mut v, 50.0),
    }
}

/// Which map the overlay shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    Gmax,
    Ux,
    Uy,
    Conf,
    /// Crack opening displacement (crack runs only).
    Cod,
    /// Crack sliding displacement (crack runs only).
    Csd,
}

impl Overlay {
    pub const ALL: [Overlay; 6] = [
        Overlay::Gmax,
        Overlay::Ux,
        Overlay::Uy,
        Overlay::Conf,
        Overlay::Cod,
        Overlay::Csd,
    ];

    /// Whether the maps of `f` hold this overlay.
    pub fn available(self, f: &Fields) -> bool {
        !matches!(self, Overlay::Cod | Overlay::Csd) || f.cod.is_some()
    }

    pub fn label(self) -> &'static str {
        match self {
            Overlay::Gmax => "gmax (最大せん断ひずみ)",
            Overlay::Ux => "ux (x変位 px)",
            Overlay::Uy => "uy (y変位 px)",
            Overlay::Conf => "有効/信頼度",
            Overlay::Cod => "COD (亀裂の開口 px)",
            Overlay::Csd => "CSD (亀裂の滑り px)",
        }
    }

    pub fn symmetric(self) -> bool {
        matches!(self, Overlay::Ux | Overlay::Uy | Overlay::Csd)
    }

    pub fn map(self, f: &Fields) -> &[f32] {
        match self {
            Overlay::Gmax => &f.gmax,
            Overlay::Ux => &f.ux,
            Overlay::Uy => &f.uy,
            Overlay::Conf => &f.conf,
            Overlay::Cod => f.cod.as_deref().unwrap_or(&f.ux),
            Overlay::Csd => f.csd.as_deref().unwrap_or(&f.ux),
        }
    }
}

/// The automatic colour range of `kind` for frame `f` (over the valid pixels): gmax 0 to its 99th
/// percentile, displacements symmetric about 0 up to the larger of their 2nd and 98th
/// percentiles, confidence 0 to 1.
pub fn auto_range(kind: Overlay, f: &Fields) -> (f32, f32) {
    if kind == Overlay::Conf {
        return (0.0, 1.0);
    }
    let map = kind.map(f);
    let mut v: Vec<f32> = (0..f.w * f.h)
        .filter(|&i| f.is_valid(i))
        .map(|i| map[i])
        .filter(|x| x.is_finite())
        .collect();
    if v.is_empty() {
        return (0.0, 1.0);
    }
    if matches!(kind, Overlay::Gmax | Overlay::Cod) {
        return (0.0, (percentile(&mut v, 99.0) as f32).max(1e-6));
    }
    let p2 = percentile(&mut v, 2.0);
    let m = p2.abs().max(percentile_sorted(&v, 98.0).abs()) as f32;
    (-m.max(1e-6), m.max(1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(ux: Vec<f32>) -> Fields {
        let n = ux.len();
        Fields {
            w: n,
            h: 1,
            uy: ux.iter().map(|x| x * 2.0).collect(),
            conf: vec![0.5; n],
            gmax: (0..n).map(|i| i as f32).collect(),
            vol: vec![0.0; n],
            fb: None,
            cod: None,
            csd: None,
            ux,
        }
    }

    #[test]
    fn frame_stats_percentiles_are_those_of_the_unsorted_values() {
        // an order that sorting matters for, with a NaN strain on a valid pixel
        let n = 101usize;
        let mut f = fields((0..n).map(|i| ((i * 37) % n) as f32).collect());
        f.gmax = (0..n).map(|i| ((i * 53) % n) as f32 * 0.5).collect();
        f.gmax[3] = f32::NAN;
        let s = frame_stats(1, &f, None);
        let mut g: Vec<f32> = f.gmax.iter().copied().filter(|v| v.is_finite()).collect();
        assert_eq!(s.g50, percentile(&mut g.clone(), 50.0));
        assert_eq!(s.g95, percentile(&mut g.clone(), 95.0));
        assert_eq!(s.g99, percentile(&mut g, 99.0));
        let mut ux = f.ux.clone();
        assert_eq!(s.ux_median, percentile(&mut ux, 50.0));
        let mut uy = f.uy.clone();
        assert_eq!(s.uy_median, percentile(&mut uy, 50.0));
        assert_eq!(s.uy_max, 200.0);
        // both range ends of a displacement map, from one sort
        let (lo, hi) = auto_range(Overlay::Ux, &f);
        let mut v = f.ux.clone();
        let m = percentile(&mut v.clone(), 2.0)
            .abs()
            .max(percentile(&mut v, 98.0).abs()) as f32;
        assert_eq!((lo, hi), (-m, m));
    }

    #[test]
    fn percentile_is_numpys() {
        let mut v = vec![4.0f32, 1.0, 3.0, 2.0];
        assert_eq!(percentile(&mut v, 50.0), 2.5);
        assert_eq!(percentile(&mut v, 0.0), 1.0);
        assert_eq!(percentile(&mut v, 100.0), 4.0);
        assert!((percentile(&mut v, 95.0) - 3.85).abs() < 1e-12);
        assert!(percentile(&mut [], 50.0).is_nan());
    }

    #[test]
    fn valid_share_counts_the_roi_only() {
        let f = fields(vec![1.0, f32::NAN, 3.0, f32::NAN]);
        let roi = [true, true, true, false];
        let s = frame_stats(1, &f, Some(&roi));
        assert!((s.valid_frac - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(s.ux_median, 2.0);
        assert_eq!(s.uy_max, 6.0);
        assert_eq!(s.mean_conf, 0.5);
        assert!((frame_stats(1, &f, None).valid_frac - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ranges() {
        let f = fields(vec![-2.0, 1.0, f32::NAN, 2.0]);
        let (lo, hi) = auto_range(Overlay::Ux, &f);
        assert_eq!(lo, -hi);
        assert!(hi > 1.9 && hi <= 2.0);
        assert_eq!(auto_range(Overlay::Conf, &f), (0.0, 1.0));
        assert_eq!(auto_range(Overlay::Gmax, &f).0, 0.0);
    }
}

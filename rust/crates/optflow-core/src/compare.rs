//! The robust tier of the GPU spec 5 (rev 3) for two results, as `rust/scripts/compare_outputs.py`
//! (which stays the reference: change both together).
#![allow(clippy::cast_precision_loss)]

/// The fields of one result.
pub struct Fields<'a> {
    pub ux: &'a [f32],
    pub uy: &'a [f32],
    pub conf: &'a [f32],
    /// Exx, Eyy, Exy, gmax, vol (any subset; compared pairwise by position)
    pub strain: Vec<&'a [f32]>,
}

/// The metrics of the robust tier; NaN where there was nothing to compare.
#[derive(Debug, Clone, Copy)]
pub struct Robust {
    pub ux_trim: f64,
    pub ux_p99: f64,
    pub ux_frac: f64,
    pub uy_trim: f64,
    pub uy_p99: f64,
    pub uy_frac: f64,
    pub conf_rms_both: f64,
    pub strain_trim_max: f64,
    pub valid_agree: f64,
}

impl Robust {
    /// The limits of spec 9.3 (NaN compares false, so an empty comparison fails).
    pub fn ok(&self) -> bool {
        self.ux_trim <= 1e-5
            && self.ux_p99 <= 1e-5
            && self.uy_trim <= 1e-5
            && self.uy_p99 <= 1e-5
            && self.ux_frac <= 5e-3
            && self.uy_frac <= 5e-3
            && self.conf_rms_both <= 2e-5
            && self.strain_trim_max <= 2e-6
            && self.valid_agree >= 0.999
    }
}

fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }
}

/// `(p99, trim, frac)` of the differences `d`: p99 of |d|, RMS of the smallest 99.9 %, fraction
/// of |d| above 1e-4 (all NaN for an empty `d`).
fn tail3(d: &[f64]) -> (f64, f64, f64) {
    let mut s: Vec<f64> = d.iter().map(|v| v.abs()).collect();
    if s.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let frac = s.iter().filter(|&&v| v > 1e-4).count() as f64 / n as f64;
    (
        s[(n as f64 * 0.99) as usize],
        rms(&s[..(n as f64 * 0.999) as usize]),
        frac,
    )
}

/// For raw flows (no conf, no strain): (trim 99.9 % RMS, p99) of |a - b| over pixels finite in
/// both, optionally only where `mask` is true.
pub fn tail(a: &[f64], b: &[f64], mask: Option<&[bool]>) -> (f64, f64) {
    let d: Vec<f64> = (0..a.len().min(b.len()))
        .filter(|&i| mask.is_none_or(|m| m[i]) && a[i].is_finite() && b[i].is_finite())
        .map(|i| a[i] - b[i])
        .collect();
    let (p99, trim, _) = tail3(&d);
    (trim, p99)
}

/// The robust tier of two results.
pub fn robust(a: &Fields, b: &Fields) -> Robust {
    let n = a.ux.len();
    let both: Vec<bool> = (0..n)
        .map(|i| a.ux[i].is_finite() && b.ux[i].is_finite())
        .collect();
    let valid_agree = (0..n)
        .filter(|&i| a.ux[i].is_finite() == b.ux[i].is_finite())
        .count() as f64
        / n as f64;
    let diff = |x: &[f32], y: &[f32], keep: &dyn Fn(usize) -> bool| -> Vec<f64> {
        (0..n)
            .filter(|&i| keep(i))
            .map(|i| f64::from(x[i]) - f64::from(y[i]))
            .collect()
    };
    let in_both = |i: usize| both[i];
    let (ux_p99, ux_trim, ux_frac) = tail3(&diff(a.ux, b.ux, &in_both));
    let (uy_p99, uy_trim, uy_frac) = tail3(&diff(a.uy, b.uy, &in_both));
    let cf: Vec<f64> = diff(a.conf, b.conf, &in_both)
        .into_iter()
        .filter(|v| v.is_finite())
        .collect();
    let mut strain_trim_max = 0.0f64;
    for (x, y) in a.strain.iter().zip(&b.strain) {
        let d = diff(x, y, &|i| x[i].is_finite() && y[i].is_finite());
        let t = tail3(&d).1;
        // NaN must stick (f64::max would drop it)
        strain_trim_max = if t.is_nan() || strain_trim_max.is_nan() {
            f64::NAN
        } else {
            strain_trim_max.max(t)
        };
    }
    Robust {
        ux_trim,
        ux_p99,
        ux_frac,
        uy_trim,
        uy_p99,
        uy_frac,
        conf_rms_both: rms(&cf),
        strain_trim_max,
        valid_agree,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields<'a>(ux: &'a [f32], uy: &'a [f32], conf: &'a [f32], s: &'a [f32]) -> Fields<'a> {
        Fields {
            ux,
            uy,
            conf,
            strain: vec![s],
        }
    }

    #[test]
    fn identical_fields_pass_with_zero_metrics() {
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let r = robust(&fields(&v, &v, &v, &v), &fields(&v, &v, &v, &v));
        assert_eq!(
            (r.ux_trim, r.ux_p99, r.ux_frac, r.conf_rms_both),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(r.valid_agree, 1.0);
        assert!(r.ok());
    }

    #[test]
    fn a_few_outliers_are_trimmed_but_many_fail() {
        let a = vec![0.0f32; 10_000];
        let mut b = a.clone();
        for v in b.iter_mut().take(5) {
            *v = 1e-2; // 0.05 %: inside the trimmed 0.1 % and the 0.5 % budget
        }
        let r = robust(&fields(&a, &a, &a, &a), &fields(&b, &a, &a, &a));
        assert!(
            r.ux_trim == 0.0 && r.ux_p99 == 0.0 && r.ok(),
            "{}",
            r.ux_trim
        );
        for v in b.iter_mut().take(100) {
            *v = 1e-2; // 1 %: p99 and frac fail
        }
        let r = robust(&fields(&a, &a, &a, &a), &fields(&b, &a, &a, &a));
        assert!(!r.ok());
    }

    #[test]
    fn validity_disagreement_and_nan_metrics_fail() {
        let a = vec![0.0f32; 100];
        let mut b = a.clone();
        b[0] = f32::NAN;
        let r = robust(&fields(&a, &a, &a, &a), &fields(&b, &a, &a, &a));
        assert!((r.valid_agree - 0.99).abs() < 1e-12 && !r.ok());
        let n = vec![f32::NAN; 100];
        assert!(!robust(&fields(&n, &n, &n, &n), &fields(&n, &n, &n, &n)).ok());
    }
}

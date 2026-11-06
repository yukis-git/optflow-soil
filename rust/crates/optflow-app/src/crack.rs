//! The crack lines of a frame (`crack_lines.json`, written by the CLI for `crack_model = xdic`).

use std::fmt::Write as _;
use std::path::Path;

use crate::fields::frame_dir;

/// Samples along one line: positions, arc length and the opening / sliding displacement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Jump {
    pub pts: Vec<[f64; 2]>,
    pub s: Vec<f64>,
    pub cod: Vec<f64>,
    pub csd: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrackLines {
    pub lines: Vec<Vec<[f64; 2]>>,
    pub bands: Vec<Vec<[f64; 2]>>,
    pub rejected: Vec<Vec<[f64; 2]>>,
    pub jumps: Vec<Jump>,
}

fn nums(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
        .unwrap_or_default()
}

fn polylines(v: &serde_json::Value) -> Vec<Vec<[f64; 2]>> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|l| {
                    l.as_array()
                        .map(|p| {
                            p.iter()
                                .map(|q| {
                                    let c = nums(q);
                                    [
                                        c.first().copied().unwrap_or(f64::NAN),
                                        c.get(1).copied().unwrap_or(f64::NAN),
                                    ]
                                })
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parses the text of `crack_lines.json`.
pub fn parse(text: &str) -> Option<CrackLines> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    Some(CrackLines {
        lines: polylines(&v["lines"]),
        bands: polylines(&v["bands"]),
        rejected: polylines(&v["rejected"]),
        jumps: v["jumps"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|j| Jump {
                        pts: polylines(&serde_json::json!([j["pts"].clone()]))
                            .pop()
                            .unwrap_or_default(),
                        s: nums(&j["s"]),
                        cod: nums(&j["cod"]),
                        csd: nums(&j["csd"]),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// The crack lines of frame `k` of a run, if the run has them.
pub fn load(run: &Path, k: usize) -> Option<CrackLines> {
    parse(&std::fs::read_to_string(frame_dir(run, k).join("crack_lines.json")).ok()?)
}

/// Median of the finite values (NaN if none).
pub fn median(v: &[f64]) -> f64 {
    let mut f: Vec<f32> = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|&x| x as f32)
        .collect();
    crate::stats::percentile(&mut f, 50.0)
}

impl CrackLines {
    /// Median COD and CSD over all samples of all lines.
    pub fn medians(&self) -> (f64, f64) {
        let cod: Vec<f64> = self
            .jumps
            .iter()
            .flat_map(|j| j.cod.iter().copied())
            .collect();
        let csd: Vec<f64> = self
            .jumps
            .iter()
            .flat_map(|j| j.csd.iter().copied())
            .collect();
        (median(&cod), median(&csd))
    }
}

/// CSV rows (`CRACK_LINES_HEADER` columns) of the samples of frame `k`.
pub fn samples_csv(k: usize, c: &CrackLines) -> String {
    let mut s = String::new();
    for (li, j) in c.jumps.iter().enumerate() {
        for i in 0..j.pts.len() {
            let g = |v: &[f64]| {
                v.get(i)
                    .copied()
                    .filter(|x| x.is_finite())
                    .map_or("nan".to_string(), |x| format!("{x:.4}"))
            };
            let _ = writeln!(
                s,
                "{k},{li},{:.2},{:.2},{},{},{}",
                j.pts[i][0],
                j.pts[i][1],
                g(&j.s),
                g(&j.cod),
                g(&j.csd)
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = r#"{"lines": [[[1.0, 2.0], [3.0, 4.0]]], "bands": [], "rejected": [[[0, 0], [1, 1]]],
        "jumps": [{"pts": [[1.0, 2.0], [3.0, 4.0]], "n": [[1,0],[1,0]], "s": [0.0, 2.8], "cod": [1.5, null], "csd": [0.25, 0.5]}]}"#;

    #[test]
    fn parses_the_cli_file() {
        let c = parse(TEXT).unwrap();
        assert_eq!(c.lines, vec![vec![[1.0, 2.0], [3.0, 4.0]]]);
        assert_eq!(c.rejected.len(), 1);
        assert_eq!(c.jumps[0].pts.len(), 2);
        assert!(c.jumps[0].cod[1].is_nan());
        assert_eq!(c.medians(), (1.5, 0.375));
        assert!(parse("not json").is_none());
    }

    #[test]
    fn sample_rows() {
        let c = parse(TEXT).unwrap();
        let rows = samples_csv(3, &c);
        assert_eq!(
            rows.lines().next().unwrap(),
            "3,0,1.00,2.00,0.0000,1.5000,0.2500"
        );
        assert!(rows.lines().nth(1).unwrap().contains("nan"));
    }
}

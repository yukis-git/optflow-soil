//! A headless end-to-end check that uses the same library path as the window: a synthetic image
//! sequence (integer arithmetic only, so every platform gets identical input bytes), a ROI, the
//! recommended settings, a run, and a report of the statistics with full precision. CI runs it on
//! Windows and the results are compared with the macOS ones.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use anyhow::{bail, Result};

use crate::fields::{list_frames, Fields};
use crate::job::{run_job, JobSpec, Msg};
use crate::settings::{Preset, Settings};
use crate::stats::{frame_stats, FrameStats};

pub const W: usize = 160;
pub const H: usize = 120;

/// Smooth noise from an integer LCG and three box blurs; frame `i` is the same picture moved by
/// `(i, i / 2)` whole pixels.
pub fn synth_frames(n: usize) -> Vec<Vec<u8>> {
    let (bw, bh) = (W + n + 8, H + n + 8);
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut v: Vec<u32> = (0..bw * bh)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u32
        })
        .collect();
    for _ in 0..3 {
        let src = v.clone();
        for y in 0..bh {
            for x in 0..bw {
                let (mut s, mut c) = (0u32, 0u32);
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < bw && (yy as usize) < bh {
                            s += src[yy as usize * bw + xx as usize];
                            c += 1;
                        }
                    }
                }
                v[y * bw + x] = s / c;
            }
        }
    }
    let (lo, hi) = (*v.iter().min().unwrap(), *v.iter().max().unwrap());
    let norm: Vec<u8> = v
        .iter()
        .map(|&p| (((p - lo) as u64 * 255) / (hi - lo).max(1) as u64) as u8)
        .collect();
    (0..n)
        .map(|i| {
            let (ox, oy) = (i, i / 2);
            let mut f = Vec::with_capacity(W * H);
            for y in 0..H {
                for x in 0..W {
                    f.push(norm[(y + oy) * bw + x + ox]);
                }
            }
            f
        })
        .collect()
}

/// Writes the synthetic sequence as `f000.png`, ... into `dir`.
pub fn write_images(dir: &Path, n: usize) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (i, f) in synth_frames(n).into_iter().enumerate() {
        let im = image::GrayImage::from_raw(W as u32, H as u32, f).expect("size");
        im.save(dir.join(format!("f{i:03}.png")))?;
    }
    Ok(())
}

/// Runs the whole thing in `out` (`images/` and `run/` inside it) and returns the report text.
pub fn run_smoke(out: &Path) -> Result<String> {
    let n = 6;
    let images = out.join("images");
    write_images(&images, n)?;
    let spec = JobSpec {
        source: images.to_string_lossy().into_owned(),
        is_video: false,
        source_count: None,
        roi: vec![(20.0, 15.0), (140.0, 18.0), (142.0, 105.0), (22.0, 102.0)],
        start: 0,
        stop: None,
        step: 1,
        settings: Settings::preset(Preset::Recommended),
        out: out.join("run"),
    };
    let mut stats: Vec<FrameStats> = vec![];
    let mut last = None;
    run_job(&spec, &AtomicBool::new(false), &mut |m| match m {
        Msg::Frame { stats: s } => stats.push(s),
        Msg::Finished | Msg::Failed(_) | Msg::Cancelled => last = Some(m),
        _ => {}
    });
    match last {
        Some(Msg::Finished) => {}
        other => bail!("run did not finish: {other:?}"),
    }
    let ks = list_frames(&spec.out);
    if ks.len() != n - 1 || stats.len() != n - 1 {
        bail!(
            "expected {} frames, found {} (stats {})",
            n - 1,
            ks.len(),
            stats.len()
        );
    }
    let mut report = String::new();
    for k in ks {
        let f = Fields::load(&spec.out, k)?;
        let roi = optflow_cli::polygon::polygon_mask(f.w, f.h, &spec.roi, 1).data;
        let s = frame_stats(k, &f, Some(&roi));
        writeln!(
            report,
            "k={k} valid={:e} conf={:e} ux_med={:e} uy_med={:e} g50={:e} g95={:e} g99={:e}",
            s.valid_frac, s.mean_conf, s.ux_median, s.uy_median, s.g50, s.g95, s.g99
        )?;
    }
    std::fs::write(out.join("smoke_report.txt"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_run_tracks_the_known_motion() {
        let out = std::env::temp_dir().join(format!("optflow-app-smoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let report = run_smoke(&out).unwrap();
        // the picture of frame i is cut at (i, i / 2), so content moves by (-i, -(i / 2)); the
        // recommended settings average 3 frames each side (clamped at the ends), which puts the
        // last result at -2, -1 px: frames 2..5 against 0..3
        let last = report.lines().last().unwrap();
        let get = |key: &str| -> f64 {
            last.split_whitespace()
                .find_map(|t| t.strip_prefix(key))
                .unwrap()
                .parse()
                .unwrap()
        };
        assert!(get("valid=") > 0.5, "{last}");
        assert!(get("g99=").is_finite(), "{last}");
        assert!((get("ux_med=") + 2.0).abs() < 0.3, "{last}");
        assert!((get("uy_med=") + 1.0).abs() < 0.3, "{last}");
        let _ = std::fs::remove_dir_all(&out);
    }
}

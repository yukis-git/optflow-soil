//! Export: CSV of displacements and strain per frame (the columns of the earlier export,
//! `results/video_project/run3_strongest/csv`), the time series, and a PNG of the current view.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

use crate::colormap::{color, Scale};
use crate::fields::Fields;
use crate::source::Gray8;
use crate::stats::{FrameStats, Overlay};

/// Grid spacing [px] of the field CSV.
pub const GRID: usize = 10;

/// How frame numbers map to the video/image index and time.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub start: i64,
    pub step: i64,
    pub fps: f64,
}

impl Timing {
    pub fn source_frame(&self, k: usize) -> i64 {
        self.start + k as i64 * self.step
    }
    pub fn time_s(&self, k: usize) -> f64 {
        self.source_frame(k) as f64 / self.fps
    }
}

pub const FIELDS_HEADER: &str =
    "frame_index,time_s,x0_px,y0_px,ux_px,uy_px,x_px,y_px,gmax,vol_detF_minus_1,conf,n_fallback\n";
/// The header of the field CSV of a crack run: two more columns.
pub const FIELDS_HEADER_CRACK: &str = "frame_index,time_s,x0_px,y0_px,ux_px,uy_px,x_px,y_px,gmax,vol_detF_minus_1,conf,n_fallback,cod_px,csd_px\n";
pub const CRACK_LINES_HEADER: &str = "frame_index,line_index,x_px,y_px,s_px,cod_px,csd_px\n";
pub const SERIES_HEADER: &str = "frame_index,video_frame,time_s,valid_frac_roi,mean_conf_valid,ux_median_px,uy_median_px,uy_max_px,gmax_p50,gmax_p95,gmax_p99,vol_p50\n";

/// Appends the rows of frame `k`: the valid pixels on the `GRID` px lattice (multiples of `GRID`),
/// row by row.
pub fn append_field_rows(out: &mut String, k: usize, t: &Timing, f: &Fields) {
    for y in (0..f.h).step_by(GRID) {
        for x in (0..f.w).step_by(GRID) {
            let i = y * f.w + x;
            if !f.is_valid(i) {
                continue;
            }
            let _ = writeln!(
                out,
                "{k},{},{x},{y},{:.3},{:.3},{:.2},{:.2},{:.5},{:.6},{:.3},{}",
                num(t.time_s(k)),
                f.ux[i],
                f.uy[i],
                x as f32 + f.ux[i],
                y as f32 + f.uy[i],
                f.gmax[i],
                f.vol[i],
                f.conf[i],
                f.fb.as_ref().map_or(0, |fb| fb[i])
            );
            if let (Some(cod), Some(csd)) = (&f.cod, &f.csd) {
                // the row just written ends with its newline: add the two columns before it
                out.pop();
                let _ = writeln!(out, ",{},{}", opt(cod[i] as f64, 4), opt(csd[i] as f64, 4));
            }
        }
    }
}

/// A number with at most 6 decimals and no trailing zeros (`0.5`, `1`, `0.033333`).
fn num(x: f64) -> String {
    let s = format!("{x:.6}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn opt(x: f64, decimals: usize) -> String {
    if x.is_finite() {
        format!("{x:.decimals$}")
    } else {
        "nan".into()
    }
}

/// The time-series CSV text.
pub fn series_csv(stats: &[FrameStats], t: &Timing) -> String {
    let mut s = String::from(SERIES_HEADER);
    for st in stats {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            st.k,
            t.source_frame(st.k),
            num(t.time_s(st.k)),
            opt(st.valid_frac, 5),
            opt(st.mean_conf, 4),
            opt(st.ux_median, 3),
            opt(st.uy_median, 3),
            opt(st.uy_max, 3),
            opt(st.g50, 5),
            opt(st.g95, 5),
            opt(st.g99, 5),
            opt(st.vol_p50, 6),
        );
    }
    s
}

/// Writes `fields_grid10px.csv` and `timeseries.csv` into `dest` from the frames `ks` of `run`
/// (one frame in memory at a time). `progress(i, n)` is called after every frame.
pub fn export_csv(
    run: &Path,
    dest: &Path,
    ks: &[usize],
    stats: &[FrameStats],
    t: &Timing,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dest).with_context(|| format!("cannot create {}", dest.display()))?;
    let path = dest.join("fields_grid10px.csv");
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(&path).with_context(|| format!("cannot write {}", path.display()))?,
    );
    let crack_run = match ks.first() {
        Some(&k) => Fields::load(run, k)?.cod.is_some(),
        None => false,
    };
    w.write_all(
        if crack_run {
            FIELDS_HEADER_CRACK
        } else {
            FIELDS_HEADER
        }
        .as_bytes(),
    )?;
    let mut crack_csv = String::from(CRACK_LINES_HEADER);
    for (i, &k) in ks.iter().enumerate() {
        let f = Fields::load(run, k)?;
        if let Some(c) = crate::crack::load(run, k) {
            crack_csv.push_str(&crate::crack::samples_csv(k, &c));
        }
        let mut rows = String::new();
        append_field_rows(&mut rows, k, t, &f);
        w.write_all(rows.as_bytes())?;
        progress(i + 1, ks.len());
    }
    w.flush()?;
    if crack_run {
        let cp = dest.join("crack_lines.csv");
        std::fs::write(&cp, crack_csv).with_context(|| format!("cannot write {}", cp.display()))?;
    }
    let sp = dest.join("timeseries.csv");
    std::fs::write(&sp, series_csv(stats, t))
        .with_context(|| format!("cannot write {}", sp.display()))?;
    Ok(())
}

/// The overlay's alpha byte for an opacity in 0..=1.
pub fn overlay_alpha(alpha: f32) -> u8 {
    (alpha.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The colour scale of an overlay.
pub fn overlay_scale(kind: Overlay) -> Scale {
    if kind.symmetric() {
        Scale::Diverging
    } else {
        Scale::Sequential
    }
}

/// The RGBA of pixel `i` of the overlay; `map` is `kind.map(f)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn overlay_pixel(
    f: &Fields,
    map: &[f32],
    kind: Overlay,
    scale: Scale,
    lo: f32,
    hi: f32,
    a: u8,
    i: usize,
) -> [u8; 4] {
    if f.is_valid(i) && map[i].is_finite() {
        let c = color(scale, map[i], lo, hi);
        [c[0], c[1], c[2], a]
    } else if kind == Overlay::Conf {
        [200, 30, 30, a / 2]
    } else {
        [0; 4]
    }
}

/// The RGBA overlay pixel `o` over the grey value `g`.
#[inline]
pub fn blend(g: u8, o: &[u8]) -> [u8; 3] {
    let a = o[3] as u32;
    let mix = |c: u8| ((c as u32 * a + g as u32 * (255 - a) + 127) / 255) as u8;
    [mix(o[0]), mix(o[1]), mix(o[2])]
}

/// The RGBA pixels of the overlay of `kind` for `f` over the range `lo..=hi` at opacity `alpha`
/// (0 to 1); invalid pixels are transparent, except for `Overlay::Conf`, which tints them red.
pub fn overlay_rgba(f: &Fields, kind: Overlay, lo: f32, hi: f32, alpha: f32) -> Vec<u8> {
    let a = overlay_alpha(alpha);
    let scale = overlay_scale(kind);
    let map = kind.map(f);
    let mut out = vec![0u8; f.w * f.h * 4];
    for i in 0..f.w * f.h {
        out[4 * i..4 * i + 4].copy_from_slice(&overlay_pixel(f, map, kind, scale, lo, hi, a, i));
    }
    out
}

/// The picture shown, as RGB at the image's own resolution: the grey frame, the overlay blended
/// on it and the ROI outline.
pub fn compose_rgb(bg: &Gray8, overlay: Option<&[u8]>, roi: &[(f64, f64)]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bg.w * bg.h * 3);
    for (i, &g) in bg.data.iter().enumerate() {
        let p = match overlay {
            Some(o) => blend(g, &o[4 * i..4 * i + 4]),
            None => [g, g, g],
        };
        rgb.extend_from_slice(&p);
    }
    if roi.len() >= 2 {
        let n = roi.len();
        let closed = n >= 3;
        for j in 0..(if closed { n } else { n - 1 }) {
            line(
                &mut rgb,
                bg.w,
                bg.h,
                roi[j],
                roi[(j + 1) % n],
                [255, 220, 0],
            );
        }
    }
    rgb
}

/// Draws the polyline `pts` (open) in `rgb_color` on RGB pixels of size `w` x `h`.
pub fn draw_polyline(rgb: &mut [u8], w: usize, h: usize, pts: &[[f64; 2]], rgb_color: [u8; 3]) {
    for p in pts.windows(2) {
        line(rgb, w, h, (p[0][0], p[0][1]), (p[1][0], p[1][1]), rgb_color);
    }
}

fn line(rgb: &mut [u8], w: usize, h: usize, a: (f64, f64), b: (f64, f64), c: [u8; 3]) {
    let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let (x, y) = (
            (a.0 + (b.0 - a.0) * t).round(),
            (a.1 + (b.1 - a.1) * t).round(),
        );
        if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 3;
            rgb[i..i + 3].copy_from_slice(&c);
        }
    }
}

/// Saves RGB pixels as a PNG.
pub fn save_png(path: &Path, w: usize, h: usize, rgb: Vec<u8>) -> Result<()> {
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb).context("bad picture size")?;
    img.save(path)
        .with_context(|| format!("cannot write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Fields {
        let (w, h) = (25, 12);
        let n = w * h;
        let mut ux = vec![0.5f32; n];
        ux[0] = f32::NAN; // (0,0) invalid
        Fields {
            w,
            h,
            uy: vec![-0.25; n],
            conf: vec![0.9; n],
            gmax: vec![0.01; n],
            vol: vec![0.002; n],
            fb: Some(vec![2; n]),
            cod: None,
            csd: None,
            ux,
        }
    }

    #[test]
    fn field_rows_follow_the_earlier_export() {
        let t = Timing {
            start: 0,
            step: 15,
            fps: 30.0,
        };
        let mut s = String::new();
        append_field_rows(&mut s, 1, &t, &tiny());
        let rows: Vec<&str> = s.lines().collect();
        // x in {0,10,20}, y in {0,10}: six lattice points, (0,0) invalid
        assert_eq!(rows.len(), 5);
        assert_eq!(
            rows[0],
            "1,0.5,10,0,0.500,-0.250,10.50,-0.25,0.01000,0.002000,0.900,2"
        );
        assert_eq!(FIELDS_HEADER.trim().split(',').count(), 12);
    }

    #[test]
    fn series_row_matches_the_earlier_columns() {
        let t = Timing {
            start: 0,
            step: 15,
            fps: 30.0,
        };
        let st = FrameStats {
            k: 1,
            valid_frac: 0.8891,
            mean_conf: 0.69,
            ux_median: -0.756,
            uy_median: -0.591,
            uy_max: 2.128,
            g50: 0.04156,
            g95: 0.09924,
            g99: 0.13354,
            vol_p50: 0.000379,
        };
        let csv = series_csv(&[st], &t);
        assert_eq!(
            csv.lines().nth(1).unwrap(),
            "1,15,0.5,0.88910,0.6900,-0.756,-0.591,2.128,0.04156,0.09924,0.13354,0.000379"
        );
    }

    #[test]
    fn overlay_and_compose() {
        let f = tiny();
        let o = overlay_rgba(&f, Overlay::Gmax, 0.0, 0.02, 1.0);
        assert_eq!(o[3], 0); // invalid pixel transparent
        assert_eq!(o[7], 255);
        let bg = Gray8 {
            w: f.w,
            h: f.h,
            data: vec![100; f.w * f.h],
        };
        let rgb = compose_rgb(&bg, Some(&o), &[(1.0, 1.0), (20.0, 1.0), (20.0, 8.0)]);
        assert_eq!(rgb.len(), f.w * f.h * 3);
        assert_eq!(&rgb[0..3], &[100, 100, 100]);
        assert_eq!(&rgb[(f.w + 1) * 3..(f.w + 1) * 3 + 3], &[255, 220, 0]);
    }

    #[test]
    fn png_round_trip() {
        let p = std::env::temp_dir().join(format!("optflow-app-{}.png", std::process::id()));
        save_png(&p, 2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let im = image::open(&p).unwrap().to_rgb8();
        assert_eq!(im.into_raw(), vec![1, 2, 3, 4, 5, 6]);
        let _ = std::fs::remove_file(&p);
    }
}

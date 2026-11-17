//! The change from the first computed frame to the last: every result is relative to the first
//! frame, so the last result is that change. Written as pictures of gmax, ux, uy and the
//! confidence over the first frame, the CSV rows of the last result, and a summary in Japanese.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use optflow_cli::args::Device;

use crate::export::{self, Timing};
use crate::fields::{list_frames, Fields};
use crate::source::Gray8;
use crate::stats::{auto_range, frame_stats, Overlay};

/// The folder of a completed run that holds these files.
pub const DIR: &str = "first_last";
pub const CSV: &str = "first_last_grid10px.csv";
pub const SUMMARY: &str = "summary.txt";

/// What the summary says besides what the run folder holds.
pub struct Params<'a> {
    /// The input (video file or image folder).
    pub source: &'a str,
    /// The first computed frame, which the pictures lie on.
    pub bg: &'a Gray8,
    pub roi: &'a [(f64, f64)],
    /// The reduction of the run (`scale`); the ROI mask of the maps is made with it.
    pub scale: usize,
    pub timing: Timing,
    pub device: Device,
    /// The confidence filter was on.
    pub filter_on: bool,
    /// The run finished; otherwise the last result it has is described, and the summary says so.
    pub complete: bool,
}

/// The four pictures: file name, map, and the colour range of each.
const PICTURES: [(&str, Overlay); 4] = [
    ("gmax.png", Overlay::Gmax),
    ("ux.png", Overlay::Ux),
    ("uy.png", Overlay::Uy),
    ("conf.png", Overlay::Conf),
];

/// Writes the pictures, the CSV and the summary of the last result of `run` into `dest`.
pub fn write(run: &Path, dest: &Path, p: &Params) -> Result<()> {
    let ks = list_frames(run);
    let Some(&last) = ks.last() else {
        bail!("no results in {}", run.display());
    };
    let f = Fields::load(run, last)?;
    std::fs::create_dir_all(dest).with_context(|| format!("cannot create {}", dest.display()))?;
    let crack = crate::crack::load(run, last);
    let same_size = (f.w, f.h) == (p.bg.w, p.bg.h);
    let mut ranges = vec![];
    for (name, kind) in PICTURES {
        let (lo, hi) = auto_range(kind, &f);
        ranges.push((name, kind, lo, hi));
        if same_size {
            let o = export::overlay_rgba(&f, kind, lo, hi, 0.7);
            let mut rgb = export::compose_rgb(p.bg, Some(&o), p.roi);
            if let Some(c) = &crack {
                for l in &c.bands {
                    export::draw_polyline(&mut rgb, f.w, f.h, l, [255, 0, 255]);
                }
                for l in &c.lines {
                    export::draw_polyline(&mut rgb, f.w, f.h, l, [0, 255, 255]);
                }
            }
            export::save_png(&dest.join(name), f.w, f.h, rgb)?;
        }
    }
    let mut csv = String::from(if f.cod.is_some() {
        export::FIELDS_HEADER_CRACK
    } else {
        export::FIELDS_HEADER
    });
    export::append_field_rows(&mut csv, last, &p.timing, &f);
    let path = dest.join(CSV);
    std::fs::write(&path, csv).with_context(|| format!("cannot write {}", path.display()))?;
    let path = dest.join(SUMMARY);
    std::fs::write(&path, summary(&f, last, &ranges, same_size, p))
        .with_context(|| format!("cannot write {}", path.display()))
}

/// The ROI mask of the maps, if there is a ROI.
fn roi_mask(f: &Fields, p: &Params) -> Option<Vec<bool>> {
    (p.roi.len() >= 3)
        .then(|| optflow_cli::polygon::polygon_mask(f.w, f.h, p.roi, p.scale.max(1)).data)
}

/// Mean and largest `|u|` [px] over the pixels of the ROI that have a displacement.
pub fn displacement_size(f: &Fields, roi: Option<&[bool]>) -> (f64, f64) {
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for i in 0..f.w * f.h {
        if roi.is_some_and(|m| !m[i]) || !f.has_value(i) || !f.uy[i].is_finite() {
            continue;
        }
        let u = f64::from(f.ux[i]).hypot(f64::from(f.uy[i]));
        sum += u;
        max = max.max(u);
        n += 1;
    }
    if n == 0 {
        (f64::NAN, f64::NAN)
    } else {
        (sum / n as f64, max)
    }
}

fn summary(
    f: &Fields,
    last: usize,
    ranges: &[(&str, Overlay, f32, f32)],
    pictures: bool,
    p: &Params,
) -> String {
    let t = &p.timing;
    let mask = roi_mask(f, p);
    let st = frame_stats(last, f, mask.as_deref());
    let (mean_u, max_u) = displacement_size(f, mask.as_deref());
    let mut s = String::new();
    let _ = writeln!(
        s,
        "最初と最後のフレームの間の変化(最初の計算フレームを基準とした、最後の結果)"
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "入力: {}", p.source);
    let _ = writeln!(
        s,
        "最初の計算フレーム: ソース {} 枚目({:.2} 秒)",
        t.source_frame(0),
        t.time_s(0)
    );
    let _ = writeln!(
        s,
        "最後の計算フレーム: ソース {} 枚目({:.2} 秒)",
        t.source_frame(last),
        t.time_s(last)
    );
    let _ = writeln!(s, "計算したフレーム数: {} 枚(結果 {} 個)", last + 1, last);
    let _ = writeln!(s, "計算装置: {}", crate::device::label(p.device));
    let _ = writeln!(
        s,
        "信頼度フィルター: {}",
        if p.filter_on { "ON" } else { "OFF" }
    );
    if !p.complete {
        let _ = writeln!(
            s,
            "注意: この解析は最後まで終わっていません。計算できた最後のフレーム(結果 {last})までを書き出しています。"
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "色の範囲(画像の自動設定):");
    for (name, kind, lo, hi) in ranges {
        let unit = match kind {
            Overlay::Ux | Overlay::Uy => " px",
            _ => "",
        };
        let _ = writeln!(s, "  {name}: {lo:.4} 〜 {hi:.4}{unit}");
    }
    if !pictures {
        let _ = writeln!(
            s,
            "  (縮小率 2 以上の結果は入力画像と大きさが異なるため、PNG は書き出していません)"
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "有効率: {:.1} %", st.valid_frac * 100.0);
    let _ = writeln!(
        s,
        "gmax の p50 / p95 / p99: {:.4} / {:.4} / {:.4}",
        st.g50, st.g95, st.g99
    );
    let _ = writeln!(
        s,
        "変位の大きさ |u| の平均 / 最大: {mean_u:.3} / {max_u:.3} px"
    );
    if !p.filter_on {
        let _ = writeln!(
            s,
            "(信頼度フィルター OFF: gmax と |u| は ROI 内の全画素で計算しています。有効率は信頼度で数えています)"
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displacement_size_counts_the_roi_pixels_with_a_value() {
        let f = Fields {
            w: 4,
            h: 1,
            ux: vec![3.0, f32::NAN, 0.0, 30.0],
            uy: vec![4.0, 1.0, 0.0, 40.0],
            conf: vec![1.0; 4],
            gmax: vec![0.0; 4],
            vol: vec![0.0; 4],
            fb: None,
            cod: None,
            csd: None,
            valid: None,
        };
        let roi = [true, true, true, false];
        assert_eq!(displacement_size(&f, Some(&roi)), (2.5, 5.0));
        assert_eq!(displacement_size(&f, None).1, 50.0);
    }

    use crate::job::{run_job, write_first_last, JobSpec, Msg};
    use crate::settings::{Preset, Settings};
    use crate::stats::FrameStats;
    use std::sync::atomic::AtomicBool;

    fn run_in(base: &Path, name: &str, settings: Settings) -> (JobSpec, Vec<FrameStats>) {
        let images = base.join("images");
        if !images.exists() {
            crate::smoke::write_images(&images, 5).unwrap();
        }
        let spec = JobSpec {
            source: images.to_string_lossy().into_owned(),
            is_video: false,
            source_count: None,
            roi: vec![(20.0, 15.0), (140.0, 18.0), (142.0, 105.0), (22.0, 102.0)],
            start: 0,
            stop: None,
            step: 1,
            settings,
            out: base.join(name),
            autosave: None,
            resume: false,
            fps: 4.0,
        };
        let (mut stats, mut last) = (vec![], None);
        run_job(&spec, &AtomicBool::new(false), &mut |m| match m {
            Msg::Frame { stats: s } => stats.push(s),
            Msg::Finished | Msg::Failed(_) | Msg::Cancelled => last = Some(m),
            _ => {}
        });
        assert!(matches!(last, Some(Msg::Finished)), "{last:?}");
        (spec, stats)
    }

    #[test]
    fn a_finished_run_leaves_the_change_from_the_first_to_the_last_frame() {
        let base = std::env::temp_dir().join(format!("optflow-app-fl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        for filter in [true, false] {
            let name = if filter { "on" } else { "off" };
            let settings = Settings {
                conf_filter: filter,
                ..Settings::preset(Preset::Recommended)
            };
            let (spec, stats) = run_in(&base, name, settings);
            let d = spec.out.join(DIR);
            for f in ["gmax.png", "ux.png", "uy.png", "conf.png", CSV, SUMMARY] {
                assert!(d.join(f).is_file(), "{name}: {f}");
            }
            let last = stats.last().unwrap();
            // the CSV is the rows of the last result in the field CSV of the whole run
            let out = base.join(format!("csv-{name}"));
            let ks: Vec<usize> = stats.iter().map(|s| s.k).collect();
            let t = Timing {
                start: 0,
                step: 1,
                fps: 4.0,
            };
            export::export_csv(&spec.out, &out, &ks, &stats, &t, &mut |_, _| {}).unwrap();
            let all = std::fs::read_to_string(out.join("fields_grid10px.csv")).unwrap();
            let want: String = all
                .lines()
                .filter(|l| l.split(',').next() == Some(&last.k.to_string()))
                .map(|l| format!("{l}\n"))
                .collect();
            let got = std::fs::read_to_string(d.join(CSV)).unwrap();
            assert_eq!(got, format!("{}{want}", export::FIELDS_HEADER), "{name}");
            assert!(want.lines().count() > 20);
            // the summary has the numbers of the frame
            let text = std::fs::read_to_string(d.join(SUMMARY)).unwrap();
            assert!(
                text.contains(&format!("有効率: {:.1} %", last.valid_frac * 100.0)),
                "{text}"
            );
            assert!(
                text.contains(&format!(
                    "{:.4} / {:.4} / {:.4}",
                    last.g50, last.g95, last.g99
                )),
                "{text}"
            );
            assert!(text.contains(&format!(
                "最後の計算フレーム: ソース {} 枚目({:.2} 秒)",
                last.k,
                last.k as f64 / 4.0
            )));
            assert!(text.contains("最初の計算フレーム: ソース 0 枚目(0.00 秒)"));
            assert!(text.contains(&format!("計算したフレーム数: {} 枚", last.k + 1)));
            assert!(text.contains(&format!(
                "信頼度フィルター: {}",
                if filter { "ON" } else { "OFF" }
            )));
            assert!(text.contains("計算装置: CPU"));
            assert!(!text.contains("最後まで終わっていません"));
            let f = Fields::load(&spec.out, last.k).unwrap();
            let roi = optflow_cli::polygon::polygon_mask(f.w, f.h, &spec.roi, 1).data;
            let (mean, max) = displacement_size(&f, Some(&roi));
            assert!(text.contains(&format!("{mean:.3} / {max:.3} px")), "{text}");
            for (n, k) in [("gmax.png", Overlay::Gmax), ("ux.png", Overlay::Ux)] {
                let (lo, hi) = auto_range(k, &f);
                assert!(text.contains(&format!("{n}: {lo:.4} 〜 {hi:.4}")), "{text}");
            }
            // the pictures are the size of the input
            let im = image::open(d.join("gmax.png")).unwrap();
            assert_eq!((im.width() as usize, im.height() as usize), (f.w, f.h));
            // the same files for the run on show, to a folder of the user's choice, and a stopped
            // run says so
            let other = base.join(format!("chosen-{name}"));
            write_first_last(&spec, &other, false).unwrap();
            let again = std::fs::read_to_string(other.join(SUMMARY)).unwrap();
            assert!(again.contains("最後まで終わっていません"));
            assert_eq!(again.replace("注意: この解析は最後まで終わっていません。計算できた最後のフレーム(結果 4)までを書き出しています。\n", ""), text);
            assert!(
                std::fs::read(other.join("gmax.png")).unwrap()
                    == std::fs::read(d.join("gmax.png")).unwrap()
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}

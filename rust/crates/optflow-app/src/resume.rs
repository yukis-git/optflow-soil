//! What makes a stopped run resumable: what identifies it, how a checkpoint found in a folder
//! compares with what the window would run now, and the statistics kept with the checkpoint.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::Result;
use optflow_cli::checkpoint::{self, Meta, Summary, FILE};
use optflow_core::checkpoint::{CheckpointError, Sum, FORMAT};

use crate::job::{roi_json, JobSpec};
use crate::settings::Settings;
use crate::stats::FrameStats;

/// One way in which a checkpoint differs from the run asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mismatch {
    /// The file list, a size or an update time.
    Input,
    Roi,
    Range,
    /// The name of the setting.
    Setting(String),
    /// Another version of the app or another platform; the numbers may differ.
    Build,
}

impl Mismatch {
    pub fn text(&self) -> String {
        match self {
            Mismatch::Input => {
                "入力(ファイルの一覧・大きさ・更新日時)が中断時と異なります。".into()
            }
            Mismatch::Roi => "ROI が中断時と異なります。".into(),
            Mismatch::Range => "開始・終了・間隔が中断時と異なります。".into(),
            Mismatch::Setting(name) => format!("設定 {name} が中断時と異なります。"),
            Mismatch::Build => {
                "中断時とアプリの版または動作環境が異なるため、同じ結果を保証できません。".into()
            }
        }
    }

    /// Whether going back to the values of the checkpoint removes this difference.
    pub fn restorable(&self) -> bool {
        matches!(self, Mismatch::Roi | Mismatch::Range | Mismatch::Setting(_))
    }
}

/// The keys of the settings in the metadata.
const SETTING_KEYS: [&str; 9] = [
    "seq_mode",
    "tavg",
    "prewarp_freeze_at",
    "ref_update_conf",
    "max_disp",
    "conf_min",
    "sigma_w",
    "scale",
    "crack",
];

fn setting_values(s: &Settings) -> [String; 9] {
    [
        s.seq_mode.clone(),
        s.tavg.to_string(),
        s.prewarp_freeze_at.to_string(),
        format!("{}", s.ref_update_conf),
        format!("{}", s.max_disp),
        format!("{}", s.conf_min),
        format!("{}", s.sigma_w),
        s.scale.to_string(),
        s.crack.to_string(),
    ]
}

/// Paths, sizes and update times of the input, as a count and a checksum.
pub fn input_identity(source: &str, is_video: bool) -> Result<(usize, u64)> {
    let paths = if is_video {
        vec![source.to_string()]
    } else {
        optflow_cli::io::list_images(source)?
    };
    let mut bytes = Vec::with_capacity(paths.len() * 64);
    for p in &paths {
        let m = std::fs::metadata(p)?;
        let t = m
            .modified()?
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        bytes.extend_from_slice(p.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&m.len().to_le_bytes());
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    let mut sum = Sum::new();
    sum.update(&bytes);
    Ok((paths.len(), sum.finish()))
}

/// What identifies `spec`, without looking at the input files.
pub fn plain_meta(spec: &JobSpec) -> Meta {
    let mut m = vec![
        ("source".to_string(), spec.source.clone()),
        ("is_video".to_string(), spec.is_video.to_string()),
        ("start".to_string(), spec.start.to_string()),
        (
            "stop".to_string(),
            spec.stop.map_or("-".to_string(), |s| s.to_string()),
        ),
        ("step".to_string(), spec.step.to_string()),
        ("roi".to_string(), roi_key(&spec.roi)),
        ("format".to_string(), FORMAT.to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
        (
            "platform".to_string(),
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        ),
        (
            // the prototype switch of the pre-warp changes the numbers
            "prewarp_env".to_string(),
            std::env::var("OPTFLOW_PREWARP_FREEZE_AT").unwrap_or_default(),
        ),
    ];
    for (k, v) in SETTING_KEYS.iter().zip(setting_values(&spec.settings)) {
        m.push((format!("set.{k}"), v));
    }
    Meta(m)
}

/// The ROI as the run sees it: fewer than 3 vertices are no ROI.
fn roi_key(roi: &[(f64, f64)]) -> String {
    if roi.len() >= 3 {
        roi_json(roi)
    } else {
        String::new()
    }
}

/// [`plain_meta`] with the identity of the input and the time of the save.
pub fn full_meta(spec: &JobSpec, total: Option<usize>) -> Result<Meta> {
    let mut m = plain_meta(spec);
    let (count, digest) = input_identity(&spec.source, spec.is_video)?;
    m.0.push(("input_files".to_string(), count.to_string()));
    m.0.push(("input_digest".to_string(), format!("{digest:016x}")));
    m.0.push((
        "total".to_string(),
        total.map_or("-".to_string(), |t| t.to_string()),
    ));
    Ok(m)
}

/// How `saved` differs from `now`, which need not hold the input's identity.
pub fn compare(saved: &Meta, now: &Meta) -> Vec<Mismatch> {
    let differs = |key: &str| saved.get(key) != now.get(key);
    let mut out = vec![];
    if ["format", "version", "platform", "prewarp_env"]
        .iter()
        .any(|k| differs(k))
    {
        out.push(Mismatch::Build);
    }
    let input_known = now.get("input_digest").is_some();
    if differs("source")
        || differs("is_video")
        || (input_known && (differs("input_files") || differs("input_digest")))
    {
        out.push(Mismatch::Input);
    }
    if differs("roi") {
        out.push(Mismatch::Roi);
    }
    if differs("start") || differs("stop") || differs("step") {
        out.push(Mismatch::Range);
    }
    for k in SETTING_KEYS {
        if differs(&format!("set.{k}")) {
            out.push(Mismatch::Setting(k.to_string()));
        }
    }
    out
}

/// The values in the window that a checkpoint brings back.
#[derive(Clone, Debug, PartialEq)]
pub struct Saved {
    pub source: String,
    pub is_video: bool,
    pub roi: Vec<(f64, f64)>,
    pub start: i64,
    pub stop: Option<i64>,
    pub step: i64,
    pub settings: Settings,
}

/// Reads [`Saved`] back from the metadata.
pub fn saved_values(m: &Meta) -> Option<Saved> {
    let set = |k: &str| m.get(&format!("set.{k}"));
    let settings = Settings {
        seq_mode: set("seq_mode")?.to_string(),
        tavg: set("tavg")?.parse().ok()?,
        prewarp_freeze_at: set("prewarp_freeze_at")?.parse().ok()?,
        ref_update_conf: set("ref_update_conf")?.parse().ok()?,
        max_disp: set("max_disp")?.parse().ok()?,
        conf_min: set("conf_min")?.parse().ok()?,
        sigma_w: set("sigma_w")?.parse().ok()?,
        scale: set("scale")?.parse().ok()?,
        crack: set("crack")?.parse().ok()?,
    };
    let roi = match m.get("roi")? {
        "" => vec![],
        text => {
            let v: serde_json::Value = serde_json::from_str(text).ok()?;
            v.get("polygon")?
                .as_array()?
                .iter()
                .map(|p| Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?)))
                .collect::<Option<Vec<_>>>()?
        }
    };
    Some(Saved {
        source: m.get("source")?.to_string(),
        is_video: m.get("is_video")?.parse().ok()?,
        roi,
        start: m.get("start")?.parse().ok()?,
        stop: match m.get("stop")? {
            "-" => None,
            s => Some(s.parse().ok()?),
        },
        step: m.get("step")?.parse().ok()?,
        settings,
    })
}

/// A checkpoint found in a folder.
#[derive(Debug)]
pub struct Found {
    pub dir: PathBuf,
    /// What it says, or why it cannot be used.
    pub summary: Result<Summary, String>,
    pub saved_at: Option<std::time::SystemTime>,
}

/// A Japanese message for a checkpoint that cannot be used.
pub fn describe_error(e: &CheckpointError) -> String {
    match e {
        CheckpointError::Format(n) => format!(
            "再開用データの形式(版 {n})がこのアプリ(版 {FORMAT})と異なるため再開できません。"
        ),
        CheckpointError::Mismatch(_) => {
            format!("入力が中断時と一致しないため再開できません。({e})")
        }
        CheckpointError::Magic | CheckpointError::Damaged(_) | CheckpointError::Io(_) => {
            format!("再開用データが壊れているか読めないため再開できません。({e})")
        }
    }
}

/// The checkpoints in `base` itself and in its `run_*` folders, the newest first.
pub fn find(base: &Path) -> Vec<Found> {
    let mut dirs = vec![base.to_path_buf()];
    dirs.extend(
        std::fs::read_dir(base)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("run_"))
            }),
    );
    let mut found: Vec<Found> = dirs
        .into_iter()
        .filter_map(|dir| {
            let path = dir.join(FILE);
            if !path.is_file() {
                return None;
            }
            Some(Found {
                summary: checkpoint::summary(&path).map_err(|e| describe_error(&e)),
                saved_at: std::fs::metadata(&path).and_then(|m| m.modified()).ok(),
                dir,
            })
        })
        .collect();
    found.sort_by_key(|f| std::cmp::Reverse(f.saved_at));
    found
}

const STAT_COLUMNS: usize = 10;

/// The time series as raw little-endian doubles, ten to a row.
pub fn encode_stats(series: &[FrameStats]) -> Vec<u8> {
    let mut out = Vec::with_capacity(series.len() * STAT_COLUMNS * 8);
    for s in series {
        for v in [
            s.k as f64,
            s.valid_frac,
            s.mean_conf,
            s.ux_median,
            s.uy_median,
            s.uy_max,
            s.g50,
            s.g95,
            s.g99,
            s.vol_p50,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

pub fn decode_stats(b: &[u8]) -> Vec<FrameStats> {
    b.as_chunks::<{ STAT_COLUMNS * 8 }>()
        .0
        .iter()
        .map(|row| {
            let (values, _) = row.as_chunks::<8>();
            let v = |i: usize| f64::from_le_bytes(values[i]);
            FrameStats {
                k: v(0) as usize,
                valid_frac: v(1),
                mean_conf: v(2),
                ux_median: v(3),
                uy_median: v(4),
                uy_max: v(5),
                g50: v(6),
                g95: v(7),
                g99: v(8),
                vol_p50: v(9),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Preset;

    fn spec() -> JobSpec {
        JobSpec {
            source: "/in/a.mp4".into(),
            is_video: true,
            source_count: Some(10),
            roi: vec![(1.0, 2.5), (30.0, 4.0), (5.0, 60.0)],
            start: 2,
            stop: Some(9),
            step: 3,
            settings: Settings::preset(Preset::Recommended),
            out: PathBuf::from("/out/run_001"),
            autosave: None,
            resume: false,
        }
    }

    #[test]
    fn every_setting_and_the_roi_are_compared() {
        let base = plain_meta(&spec());
        assert!(compare(&base, &base).is_empty());
        let check = |edit: &dyn Fn(&mut JobSpec), want: Mismatch| {
            let mut s = spec();
            edit(&mut s);
            assert_eq!(compare(&base, &plain_meta(&s)), vec![want]);
        };
        check(&|s| s.roi[1].0 = 31.0, Mismatch::Roi);
        check(&|s| s.roi.clear(), Mismatch::Roi);
        check(&|s| s.start = 3, Mismatch::Range);
        check(&|s| s.stop = None, Mismatch::Range);
        check(&|s| s.step = 1, Mismatch::Range);
        check(&|s| s.source = "/in/b.mp4".into(), Mismatch::Input);
        let setting = |n: &str| Mismatch::Setting(n.to_string());
        check(
            &|s| s.settings.seq_mode = "chain".into(),
            setting("seq_mode"),
        );
        check(&|s| s.settings.tavg = 1, setting("tavg"));
        check(
            &|s| s.settings.prewarp_freeze_at = 0,
            setting("prewarp_freeze_at"),
        );
        check(
            &|s| s.settings.ref_update_conf = 0.4,
            setting("ref_update_conf"),
        );
        check(&|s| s.settings.max_disp = 33.0, setting("max_disp"));
        check(&|s| s.settings.conf_min = 0.11, setting("conf_min"));
        check(&|s| s.settings.sigma_w = 3.5, setting("sigma_w"));
        check(&|s| s.settings.scale = 2, setting("scale"));
        check(&|s| s.settings.crack = true, setting("crack"));
        // the output folder and the autosave interval do not change results
        check_same(&base);
    }

    fn check_same(base: &Meta) {
        let mut s = spec();
        s.out = PathBuf::from("/elsewhere");
        s.autosave = Some(optflow_cli::checkpoint::Policy::default());
        assert!(compare(base, &plain_meta(&s)).is_empty());
    }

    #[test]
    fn another_build_is_refused_and_cannot_be_restored() {
        let base = plain_meta(&spec());
        for key in ["format", "version", "platform", "prewarp_env"] {
            let mut other = base.clone();
            other.0.iter_mut().find(|(k, _)| k == key).unwrap().1 = "other".into();
            let got = compare(&base, &other);
            assert_eq!(got, vec![Mismatch::Build], "{key}");
            assert!(!got[0].restorable());
        }
    }

    #[test]
    fn the_window_gets_back_what_was_saved() {
        let mut s = spec();
        s.settings.sigma_w = 0.1 + 0.2;
        s.roi.push((0.1 + 0.2, 7.0));
        let v = saved_values(&plain_meta(&s)).unwrap();
        assert_eq!(v.roi, s.roi);
        assert_eq!(v.settings, s.settings);
        assert_eq!((v.start, v.stop, v.step), (2, Some(9), 3));
        assert_eq!((v.source.as_str(), v.is_video), ("/in/a.mp4", true));
        // no ROI, no end
        s.roi = vec![(1.0, 1.0)];
        s.stop = None;
        let v = saved_values(&plain_meta(&s)).unwrap();
        assert!(v.roi.is_empty() && v.stop.is_none());
    }

    #[test]
    fn the_input_identity_sees_a_new_file_and_a_changed_one() {
        let d = std::env::temp_dir().join(format!("optflow-app-ident-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        crate::smoke::write_images(&d, 3).unwrap();
        let src = d.to_string_lossy().into_owned();
        let a = input_identity(&src, false).unwrap();
        assert_eq!(a, input_identity(&src, false).unwrap());
        assert_eq!(a.0, 3);
        // the same size and another time
        let f = d.join("f001.png");
        let file = std::fs::File::options().write(true).open(&f).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1000))
            .unwrap();
        let b = input_identity(&src, false).unwrap();
        assert_ne!(a.1, b.1);
        // another size
        file.set_len(file.metadata().unwrap().len() + 1).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1000))
            .unwrap();
        let c = input_identity(&src, false).unwrap();
        assert_ne!(b.1, c.1);
        drop(file);
        // one more file
        std::fs::copy(d.join("f000.png"), d.join("f009.png")).unwrap();
        assert_eq!(input_identity(&src, false).unwrap().0, 4);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_time_series_round_trips_with_nan() {
        let s = FrameStats {
            k: 7,
            valid_frac: 0.25,
            mean_conf: f64::NAN,
            ux_median: -1.5,
            uy_median: 2.5,
            uy_max: 3.5,
            g50: 0.1,
            g95: 0.2,
            g99: 0.3,
            vol_p50: f64::NAN,
        };
        let back = decode_stats(&encode_stats(&[s.clone(), s.clone()]));
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].k, 7);
        assert_eq!(back[1].valid_frac, 0.25);
        assert!(back[0].mean_conf.is_nan());
    }

    #[test]
    fn a_checkpoint_is_found_in_the_base_and_in_its_runs() {
        let d = std::env::temp_dir().join(format!("optflow-app-find-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("run_001")).unwrap();
        std::fs::create_dir_all(d.join("other")).unwrap();
        assert!(find(&d).is_empty());
        std::fs::write(d.join("run_001").join(FILE), b"not a checkpoint").unwrap();
        std::fs::write(d.join("other").join(FILE), b"not a checkpoint").unwrap();
        let got = find(&d);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].dir, d.join("run_001"));
        assert!(got[0].summary.is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}

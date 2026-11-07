//! Running the tracker on a worker thread through the library (`optflow_cli::run::run_with`),
//! with progress messages and cancellation.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use optflow_cli::args::{Cli, Command, CropMode, Device};
use optflow_cli::checkpoint::{self, Checkpointing, Policy, SaveInfo};
use optflow_cli::run::{run_checkpointed, run_with, Hooks};
use optflow_core::image::Image;
use optflow_core::track::TrackResult;

use crate::fields::Fields;
use crate::messages;
use crate::power;
use crate::resume;
use crate::settings::Settings;
use crate::stats::{frame_stats, FrameStats};

/// Everything a run needs.
#[derive(Clone, Debug)]
pub struct JobSpec {
    pub source: String,
    pub is_video: bool,
    /// Video frames, if known (for the progress total).
    pub source_count: Option<usize>,
    /// ROI vertices in full-resolution pixels (at least 3 for a ROI; fewer: whole image).
    pub roi: Vec<(f64, f64)>,
    pub start: i64,
    pub stop: Option<i64>,
    pub step: i64,
    pub settings: Settings,
    /// The run folder (created; `frame_NNNNN` folders are written into it).
    pub out: PathBuf,
    /// Save checkpoints: periodically by this policy and when stopped. `None`: no checkpoints. A
    /// crack run has none in any case.
    pub autosave: Option<Policy>,
    /// Continue from the checkpoint in `out`.
    pub resume: bool,
}

/// Messages from the worker to the window.
#[derive(Debug)]
pub enum Msg {
    Started {
        total: Option<usize>,
    },
    Frame {
        stats: FrameStats,
    },
    /// A resumed run: the results that exist already.
    Restored {
        stats: Vec<FrameStats>,
    },
    /// A checkpoint was written.
    Saved(SaveInfo),
    Warn(String),
    Finished,
    Cancelled,
    Failed(String),
}

/// The default output folder for an input: `optflow_出力` beside it. The working directory is no
/// default to rely on: started from Finder it is `/`, and a double click on Windows gives the
/// exe's folder.
pub fn default_out(input: &Path) -> Option<PathBuf> {
    let parent = input.parent().filter(|p| !p.as_os_str().is_empty())?;
    Some(parent.join("optflow_出力"))
}

/// The next unused `run_NNN` folder name under `base` (not created).
pub fn next_run_dir(base: &Path) -> PathBuf {
    (1..)
        .map(|i| base.join(format!("run_{i:03}")))
        .find(|p| !p.exists())
        .expect("an unused run folder exists")
}

/// How many results a run of `spec` produces (sampled frames minus one), if it can be told.
pub fn expected_results(spec: &JobSpec) -> Option<usize> {
    let n = if spec.is_video {
        spec.source_count
    } else {
        optflow_cli::io::list_images(&spec.source)
            .ok()
            .map(|v| v.len())
    }?;
    let picked = optflow_cli::io::slice_indices(n, spec.start, spec.stop, spec.step).len();
    Some(picked.saturating_sub(1))
}

/// Bytes one result takes in the output folder. Whatever the ROI, every map is written at the full
/// (scaled) image size: 8 `<f4` maps, the uint8 `valid`, `fb` (`<i4`) in hybrid runs, the 5
/// crack maps (4 `<f4` and a mask) in crack runs.
pub fn output_bytes_per_result(w: usize, h: usize, settings: &Settings) -> u64 {
    let scale = settings.scale.max(1) as usize;
    let px = (w / scale) as u64 * (h / scale) as u64;
    let mut per_px = 8 * 4 + 1;
    if settings.seq_mode == "hybrid" {
        per_px += 4;
    }
    if settings.crack {
        per_px += 4 * 4 + 1;
    }
    px * per_px
}

/// The largest checkpoint a run can leave, in bytes: what cannot be recomputed from the input, at
/// the (scaled) image size; a ROI makes it smaller. Per pixel: two displacements, the link
/// confidence (a double), the previous leg and its confidence; a hybrid run also has two bit masks,
/// the counter of carried frames, the carried displacement of pixels in fallback and the adjacent
/// leg. Much of this collapses while nothing has moved or fallen back. Crack runs have none.
pub fn checkpoint_bytes(w: usize, h: usize, settings: &Settings) -> u64 {
    if settings.crack {
        return 0;
    }
    let scale = settings.scale.max(1) as usize;
    let px = (w / scale) as u64 * (h / scale) as u64;
    let chain_bytes = 8 + 8 + 8 + 4;
    if settings.seq_mode == "hybrid" {
        px * (chain_bytes + 1 + 8 + 8) + px / 4
    } else {
        px * chain_bytes
    }
}

/// `1.2 GB`, `850 MB` (decimal, as file managers show them).
pub fn fmt_bytes(n: u64) -> String {
    if n >= 999_500_000 {
        format!("{:.1} GB", n as f64 / 1e9)
    } else {
        format!("{} MB", (n as f64 / 1e6).round().max(1.0) as u64)
    }
}

/// The note beside the start button. `results`: how many results the run gives, if known.
pub fn output_note(w: usize, h: usize, settings: &Settings, results: Option<usize>) -> String {
    let one = output_bytes_per_result(w, h, settings);
    let mut note = match results {
        Some(n) if n >= 1 => format!(
            "保存サイズの目安: 1 フレームあたり約 {} × {n} = 約 {}(ROI によらず画像全体の大きさで保存します)",
            fmt_bytes(one),
            fmt_bytes(one * n as u64)
        ),
        _ => format!(
            "保存サイズの目安: 1 フレームあたり約 {}(ROI によらず画像全体の大きさで保存します)",
            fmt_bytes(one)
        ),
    };
    let ck = checkpoint_bytes(w, h, settings);
    if ck > 0 {
        note.push_str(&format!(
            "。再開用データは解析中に最大 約 {}(書き込み中は一時的に 2 倍、完了後に削除)",
            fmt_bytes(ck)
        ));
    }
    note
}

/// The ROI file the CLI reads, `{"polygon": [[x, y], ...]}`.
pub fn roi_json(points: &[(f64, f64)]) -> String {
    let pts: Vec<String> = points.iter().map(|(x, y)| format!("[{x}, {y}]")).collect();
    format!("{{\"polygon\": [{}]}}\n", pts.join(", "))
}

/// The command line the app stands for (so a run can be repeated with `optflow seq`).
pub fn cli_for(spec: &JobSpec) -> Cli {
    let roi = (spec.roi.len() >= 3).then(|| spec.out.join("roi.json"));
    Cli {
        command: Command::Seq {
            frames: spec.source.clone(),
            start: spec.start,
            stop: spec.stop,
            step: spec.step,
        },
        roi,
        mask: None,
        scale: spec.settings.scale,
        crop: CropMode::Auto,
        crop_margin: None,
        device: Device::Cpu,
        config: None,
        set: spec.settings.overrides(),
        out: spec.out.clone(),
        png: false,
        raw: false,
    }
}

struct WorkerHooks<'a> {
    spec: &'a JobSpec,
    cancel: &'a AtomicBool,
    send: &'a mut dyn FnMut(Msg),
    roi: Option<Vec<bool>>,
    roi_size: (usize, usize),
    /// The time series so far, kept with the checkpoints.
    series: Vec<FrameStats>,
}

impl Hooks for WorkerHooks<'_> {
    fn say(&mut self, _text: &str) -> Result<()> {
        Ok(())
    }

    fn warn(&mut self, text: &str) {
        (self.send)(Msg::Warn(text.to_string()));
    }

    fn frame_done(&mut self, k: usize, _res: &TrackResult) {
        // the saved (full-size) maps are the source of the statistics, as they are of the display
        let Ok(f) = Fields::load(&self.spec.out, k) else {
            return;
        };
        if self.spec.roi.len() >= 3 && self.roi_size != (f.w, f.h) {
            let m: Image<bool> = optflow_cli::polygon::polygon_mask(
                f.w,
                f.h,
                &self.spec.roi,
                self.spec.settings.scale.max(1) as usize,
            );
            self.roi = Some(m.data);
            self.roi_size = (f.w, f.h);
        }
        let stats = frame_stats(k, &f, self.roi.as_deref());
        self.series.push(stats.clone());
        (self.send)(Msg::Frame { stats });
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn checkpoint_extra(&self) -> Vec<u8> {
        resume::encode_stats(&self.series)
    }

    fn checkpoint_saved(&mut self, info: &SaveInfo) {
        (self.send)(Msg::Saved(*info));
    }
}

/// Whether the checkpoint in `spec.out` can be continued by `spec`; the reason in Japanese if not.
/// The results so far on success.
fn check_resume(spec: &JobSpec) -> Result<Vec<FrameStats>, String> {
    if spec.settings.crack {
        return Err(CRACK_NO_RESUME.to_string());
    }
    let path = spec.out.join(checkpoint::FILE);
    let found = checkpoint::summary(&path).map_err(|e| resume::describe_error(&e))?;
    let now = resume::full_meta(spec, None).map_err(|e| messages::describe(&e))?;
    let diff = resume::compare(&found.meta, &now);
    if !diff.is_empty() {
        let text: Vec<String> = diff.iter().map(resume::Mismatch::text).collect();
        return Err(format!("再開できません。{}", text.join("")));
    }
    let extra = checkpoint::extra(&path).map_err(|e| resume::describe_error(&e))?;
    let mut stats = resume::decode_stats(&extra);
    stats.retain(|s| s.k <= found.head.k);
    Ok(stats)
}

/// Why a crack run is not saved.
pub const CRACK_NO_RESUME: &str = "亀裂モデルでは再開できません。";

/// Runs `spec` on the calling thread, sending messages through `send`; the last one is
/// `Finished`, `Cancelled` or `Failed`.
pub fn run_job(spec: &JobSpec, cancel: &AtomicBool, send: &mut dyn FnMut(Msg)) {
    send(Msg::Started {
        total: expected_results(spec),
    });
    // before anything is written: a refused resume leaves the folder as it is
    let mut series = vec![];
    if spec.resume {
        match check_resume(spec) {
            Ok(stats) => {
                series.clone_from(&stats);
                send(Msg::Restored { stats });
            }
            Err(text) => {
                send(Msg::Failed(text));
                return;
            }
        }
    }
    let prepare = || -> Result<()> {
        std::fs::create_dir_all(&spec.out)
            .with_context(|| format!("cannot create {}", spec.out.display()))?;
        if spec.roi.len() >= 3 {
            let p = spec.out.join("roi.json");
            std::fs::write(&p, roi_json(&spec.roi))
                .with_context(|| format!("cannot write {}", p.display()))?;
        }
        Ok(())
    };
    if let Err(e) = prepare() {
        send(Msg::Failed(messages::describe(&e)));
        return;
    }
    let policy = spec.autosave.filter(|_| !spec.settings.crack);
    let meta = match policy.map(|_| resume::full_meta(spec, expected_results(spec))) {
        Some(Ok(m)) => Some(m),
        Some(Err(e)) => {
            send(Msg::Failed(messages::describe(&e)));
            return;
        }
        None => None,
    };
    let cli = cli_for(spec);
    let result = {
        let mut hooks = WorkerHooks {
            spec,
            cancel,
            send: &mut *send,
            roi: None,
            roi_size: (0, 0),
            series,
        };
        match (policy, meta) {
            (Some(policy), Some(meta)) => {
                let mut ckpt = Checkpointing::new(spec.out.clone(), policy, meta, spec.resume);
                run_checkpointed(&cli, &mut hooks, &mut ckpt)
            }
            _ => run_with(&cli, &mut hooks),
        }
    };
    match result {
        Ok(()) => send(Msg::Finished),
        Err(e) if e.downcast_ref::<optflow_cli::run::Cancelled>().is_some() => send(Msg::Cancelled),
        Err(e) => {
            let text = match e.downcast_ref::<optflow_core::checkpoint::CheckpointError>() {
                Some(c) => resume::describe_error(c),
                None => messages::describe(&e),
            };
            send(Msg::Failed(text));
        }
    }
}

/// A running worker thread.
pub struct Running {
    pub cancel: Arc<AtomicBool>,
    pub handle: JoinHandle<()>,
}

/// Starts `spec` on a new thread; `repaint` is called after every message so the window wakes.
/// A panic in the worker becomes a `Failed` message.
pub fn spawn(spec: JobSpec, tx: Sender<Msg>, repaint: Arc<dyn Fn() + Send + Sync>) -> Running {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    let handle = std::thread::spawn(move || {
        let _awake = power::keep_running("optflow analysis");
        let mut send = |m: Msg| {
            let _ = tx.send(m);
            repaint();
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(&spec, &flag, &mut send)
        }));
        if outcome.is_err() {
            send(Msg::Failed(
                "内部エラーが発生しました(処理を続けられません)。入力と設定を確認してもう一度試してください。".into(),
            ));
        }
    });
    Running { cancel, handle }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Preset;

    #[test]
    fn roi_json_is_what_the_cli_reads() {
        let text = roi_json(&[(1.0, 2.5), (3.0, 4.0), (5.0, 6.0)]);
        let dir = std::env::temp_dir().join(format!("optflow-app-roi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("r.json"), text).unwrap();
        let pts = optflow_cli::io::load_roi(&dir.join("r.json")).unwrap();
        assert_eq!(pts, vec![(1.0, 2.5), (3.0, 4.0), (5.0, 6.0)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_out_is_beside_the_input() {
        assert_eq!(
            default_out(Path::new("/データ/写真 1")),
            Some(PathBuf::from("/データ/optflow_出力"))
        );
        assert_eq!(default_out(Path::new("video.mp4")), None);
    }

    #[test]
    fn next_run_dir_skips_existing() {
        let base = std::env::temp_dir().join(format!("optflow-app-run-{}", std::process::id()));
        std::fs::create_dir_all(base.join("run_001")).unwrap();
        assert_eq!(next_run_dir(&base), base.join("run_002"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_size_note_reads_well() {
        let mut s = Settings::preset(Preset::Standard);
        s.seq_mode = "chain".into();
        assert_eq!(output_bytes_per_result(6240, 4160, &s), 6240 * 4160 * 33);
        assert_eq!(fmt_bytes(856_000_000), "856 MB");
        assert_eq!(fmt_bytes(2_568_000_000), "2.6 GB");
        assert!(output_note(6240, 4160, &s, Some(3)).contains("約 857 MB × 3 = 約 2.6 GB"));
        assert!(!output_note(6240, 4160, &s, None).contains('×'));
        assert!(!output_note(6240, 4160, &s, Some(0)).contains('×'));
        s.scale = 2;
        assert_eq!(output_bytes_per_result(6240, 4160, &s), 3120 * 2080 * 33);
    }

    /// The estimate against what a real run leaves in its folder.
    #[test]
    fn the_size_estimate_matches_a_real_run() {
        let out = std::env::temp_dir().join(format!("optflow-app-size-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        crate::smoke::run_smoke(&out).unwrap();
        let run = out.join("run");
        let ks = crate::fields::list_frames(&run);
        assert!(!ks.is_empty());
        let settings = Settings::preset(Preset::Recommended);
        let want = output_bytes_per_result(crate::smoke::W, crate::smoke::H, &settings) as f64;
        for k in ks {
            let dir = crate::fields::frame_dir(&run, k);
            let got: u64 = std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .map(|e| e.metadata().unwrap().len())
                .sum();
            let ratio = got as f64 / want;
            assert!((0.9..1.1).contains(&ratio), "frame {k}: {got} vs {want}");
        }
        let _ = std::fs::remove_dir_all(&out);
    }

    /// The same library call as `optflow seq` with the same settings: same bytes in every map.
    #[test]
    fn a_run_writes_the_bytes_of_the_command_line() {
        let base = std::env::temp_dir().join(format!("optflow-app-bytes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let images = base.join("images");
        crate::smoke::write_images(&images, 4).unwrap();
        let spec = JobSpec {
            source: images.to_string_lossy().into_owned(),
            is_video: false,
            source_count: None,
            roi: vec![(20.0, 15.0), (140.0, 18.0), (142.0, 105.0), (22.0, 102.0)],
            start: 0,
            stop: None,
            step: 1,
            settings: Settings::preset(Preset::Recommended),
            out: base.join("app"),
            autosave: None,
            resume: false,
        };
        let mut last = None;
        run_job(&spec, &AtomicBool::new(false), &mut |m| last = Some(m));
        assert!(matches!(last, Some(Msg::Finished)), "{last:?}");
        let mut cli = cli_for(&spec);
        cli.out = base.join("cli");
        optflow_cli::run::run(&cli).unwrap();
        let ks = crate::fields::list_frames(&spec.out);
        assert_eq!(ks, crate::fields::list_frames(&cli.out));
        assert_eq!(ks.len(), 3);
        for k in ks {
            let (a, c) = (
                crate::fields::frame_dir(&spec.out, k),
                crate::fields::frame_dir(&cli.out, k),
            );
            let mut n = 0;
            for e in std::fs::read_dir(&a).unwrap().flatten() {
                let name = e.file_name();
                if name.to_string_lossy().ends_with(".npy") {
                    let same =
                        std::fs::read(e.path()).unwrap() == std::fs::read(c.join(&name)).unwrap();
                    assert!(same, "frame {k} {name:?}");
                    n += 1;
                }
            }
            assert!(n >= 9, "frame {k}: only {n} maps");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_missing_source_fails_with_a_message_not_a_panic() {
        let spec = JobSpec {
            source: "/no/such/folder".into(),
            is_video: false,
            source_count: None,
            roi: vec![],
            start: 0,
            stop: None,
            step: 1,
            settings: Settings::preset(Preset::Standard),
            out: std::env::temp_dir().join(format!("optflow-app-fail-{}", std::process::id())),
            autosave: None,
            resume: false,
        };
        let mut msgs = vec![];
        run_job(&spec, &AtomicBool::new(false), &mut |m| msgs.push(m));
        assert!(
            matches!(msgs.last(), Some(Msg::Failed(t)) if t.contains("画像が見つかりません")),
            "{msgs:?}"
        );
        let _ = std::fs::remove_dir_all(&spec.out);
    }
}

//! A run stopped and resumed writes the bytes of an uninterrupted run, for every output file except
//! the wall-clock fields of `timing.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use optflow_app::fields::list_frames;
use optflow_app::job::{run_job, JobSpec, Msg, CRACK_NO_RESUME};
use optflow_app::settings::{Preset, Settings};
use optflow_cli::checkpoint::{self, Policy, FILE, TMP};

const W: usize = 200;
const H: usize = 160;
const FRAMES: usize = 14;

/// Every frame, every time.
const EVERY_FRAME: Policy = Policy {
    interval_s: 0.0,
    max_cost: f64::INFINITY,
};
/// Only when stopped.
const ON_STOP: Policy = Policy {
    interval_s: f64::INFINITY,
    max_cost: 0.05,
};

fn noise(seed: u64, bw: usize, bh: usize) -> Vec<u32> {
    let mut state = seed;
    let mut v: Vec<u32> = (0..bw * bh)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u32
        })
        .collect();
    for _ in 0..2 {
        let src = v.clone();
        for y in 2..bh - 2 {
            for x in 2..bw - 2 {
                let mut s = 0;
                for dy in 0..5 {
                    for dx in 0..5 {
                        s += src[(y + dy - 2) * bw + x + dx - 2];
                    }
                }
                v[y * bw + x] = s / 25;
            }
        }
    }
    v
}

/// A moving texture that a second one slowly replaces in part of the picture, so that the reference
/// moves and pixels fall back to the adjacent leg.
fn write_input(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let (bw, bh) = (W + 3 * FRAMES + 8, H + FRAMES + 8);
    let a = noise(0x2545_F491_4F6C_DD1D, bw, bh);
    let b = noise(0x1234_5678_9ABC_DEF1, bw, bh);
    for i in 0..FRAMES {
        let (ox, oy) = (2 * i, i);
        let im = image::GrayImage::from_fn(W as u32, H as u32, |x, y| {
            let (x, y) = (x as usize, y as usize);
            let j = (y + oy) * bw + x + ox;
            // the right part decorrelates: 16 steps from the texture to the other one
            let mix = if x > W / 2 { (2 * i).min(16) as u32 } else { 0 };
            let v = (a[j] * (16 - mix) + b[j] * mix) / 16;
            image::Luma([(v.saturating_sub(90) * 4).min(255) as u8])
        });
        im.save(dir.join(format!("f{i:03}.png"))).unwrap();
    }
}

struct Work {
    base: PathBuf,
    input: PathBuf,
}

impl Work {
    fn new(name: &str) -> Work {
        let base =
            std::env::temp_dir().join(format!("optflow-resume-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let input = base.join("input");
        write_input(&input);
        Work { base, input }
    }

    fn spec(&self, name: &str, settings: Settings, roi: bool) -> JobSpec {
        JobSpec {
            source: self.input.to_string_lossy().into_owned(),
            is_video: false,
            source_count: None,
            roi: if roi {
                vec![
                    (12.0, 10.0),
                    (188.0, 14.0),
                    (190.0, 150.0),
                    (100.0, 120.0),
                    (15.0, 148.0),
                ]
            } else {
                vec![]
            },
            start: 0,
            stop: None,
            step: 1,
            settings,
            out: self.base.join(name),
            autosave: None,
            resume: false,
        }
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

#[derive(Debug, PartialEq)]
enum End {
    Finished,
    Cancelled,
    Failed(String),
}

/// Runs `spec`, stopping after frame `stop_at`; how it ended and the frames it reported.
fn run(spec: &JobSpec, stop_at: Option<usize>) -> (End, Vec<usize>, Vec<usize>) {
    let cancel = AtomicBool::new(false);
    let mut end = None;
    let (mut frames, mut restored) = (vec![], vec![]);
    run_job(spec, &cancel, &mut |m| match m {
        Msg::Frame { stats } => {
            if Some(stats.k) == stop_at {
                cancel.store(true, Ordering::Relaxed);
            }
            frames.push(stats.k);
        }
        Msg::Restored { stats } => restored = stats.iter().map(|s| s.k).collect(),
        Msg::Finished => end = Some(End::Finished),
        Msg::Cancelled => end = Some(End::Cancelled),
        Msg::Failed(t) => end = Some(End::Failed(t)),
        _ => {}
    });
    (end.expect("a last message"), frames, restored)
}

fn tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut todo = vec![root.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                todo.push(p);
            } else {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                out.insert(rel, std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// `timing.json` without what the clock gives: the seconds of each stage.
fn without_clock(bytes: &[u8]) -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    let m = v.as_object_mut().unwrap();
    assert!(m.contains_key("coarse"));
    for k in ["coarse", "fine", "strain", "xdic_detect", "xdic"] {
        m.remove(k);
    }
    v
}

fn assert_same_tree(got: &Path, want: &Path, what: &str) {
    let (a, b) = (tree(got), tree(want));
    let names = |t: &BTreeMap<String, Vec<u8>>| t.keys().cloned().collect::<Vec<_>>();
    assert_eq!(names(&a), names(&b), "{what}: file list");
    for (name, bytes) in &a {
        if name.ends_with("timing.json") {
            assert_eq!(
                without_clock(bytes),
                without_clock(&b[name]),
                "{what}: {name}"
            );
        } else {
            assert!(bytes == &b[name], "{what}: {name} differs");
        }
    }
}

/// The reference and the carried count of every frame of a finished run.
fn timings(run: &Path) -> Vec<(usize, i64)> {
    list_frames(run)
        .into_iter()
        .map(|k| {
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(run.join(format!("frame_{k:05}")).join("timing.json")).unwrap(),
            )
            .unwrap();
            (
                v["reference"].as_u64().unwrap() as usize,
                v["n_fb_max"].as_i64().unwrap_or(0),
            )
        })
        .collect()
}

fn hybrid_settings() -> Settings {
    // the recommended preset (tavg 3, prewarp_freeze_at 3), with a reference that moves here
    Settings {
        ref_update_conf: 0.6,
        ..Settings::preset(Preset::Recommended)
    }
}

fn chain_settings(tavg: i64) -> Settings {
    Settings {
        ref_update_conf: 0.6,
        tavg,
        prewarp_freeze_at: 3,
        ..Settings::preset(Preset::Standard)
    }
}

/// Stops at each of `stops` in turn, resuming after each, then finishes; compares with `whole`.
fn stop_and_resume(work: &Work, name: &str, base: &JobSpec, whole: &Path, stops: &[usize]) {
    let dir = format!("{name}-{stops:?}");
    let mut spec = base.clone();
    spec.out = work.base.join(&dir);
    spec.autosave = Some(ON_STOP);
    let mut done = 0;
    for (i, &stop) in stops.iter().enumerate() {
        spec.resume = i > 0;
        let (end, frames, restored) = run(&spec, Some(stop));
        assert_eq!(end, End::Cancelled, "{dir}: stop {stop}");
        assert_eq!(frames, (done + 1..=stop).collect::<Vec<_>>(), "{dir}");
        assert_eq!(restored, (1..=done).collect::<Vec<_>>(), "{dir}");
        assert!(spec.out.join(FILE).is_file(), "{dir}: stop {stop}");
        let s = checkpoint::summary(&spec.out.join(FILE)).unwrap();
        assert_eq!(s.head.k, stop);
        done = stop;
    }
    spec.resume = true;
    let (end, frames, restored) = run(&spec, None);
    assert_eq!(end, End::Finished, "{dir}");
    assert_eq!(restored.len(), done, "{dir}");
    assert_eq!(frames.last().copied().unwrap_or(done), FRAMES - 1, "{dir}");
    assert!(!spec.out.join(FILE).exists(), "{dir}: the run finished");
    assert_same_tree(&spec.out, whole, &dir);
    let _ = std::fs::remove_dir_all(&spec.out);
}

/// The stops of a mode: right after frame 1, right after each of the first reference moves, around
/// the middle, at the last frame, and twice in a row.
fn exercise(work: &Work, name: &str, spec: &JobSpec) {
    let (end, _, _) = run(spec, None);
    assert_eq!(end, End::Finished, "{name}");
    let t = timings(&spec.out);
    let moves: Vec<usize> = (1..t.len())
        .filter(|&i| t[i].0 != t[i - 1].0)
        .map(|i| i + 1)
        .collect();
    assert!(!moves.is_empty(), "{name}: the reference must move: {t:?}");
    if spec.settings.seq_mode == "hybrid" {
        assert!(
            t.iter().any(|x| x.1 > 0),
            "{name}: some pixels must fall back: {t:?}"
        );
    }
    let last = t.len();
    let mut cases: Vec<Vec<usize>> = vec![vec![1], vec![last / 2], vec![last - 1], vec![last]];
    for &m in moves.iter().take(2) {
        cases.push(vec![m]);
    }
    cases.push(vec![1, 2]);
    cases.push(vec![moves[0], last - 1]);
    cases.push(vec![2, last / 2, last - 2]);
    for c in cases {
        stop_and_resume(work, name, spec, &spec.out, &c);
    }
    // saving at every frame changes nothing either
    let mut every = spec.clone();
    every.out = work.base.join(format!("{name}-every"));
    every.autosave = Some(EVERY_FRAME);
    assert_eq!(run(&every, None).0, End::Finished);
    assert_same_tree(
        &every.out,
        &spec.out,
        &format!("{name} saving at every frame"),
    );
    let _ = std::fs::remove_dir_all(&every.out);
    let _ = std::fs::remove_dir_all(&spec.out);
}

#[test]
fn hybrid_with_averaging_and_a_roi() {
    let w = Work::new("hybrid-roi");
    exercise(&w, "h", &w.spec("whole", hybrid_settings(), true));
}

#[test]
fn hybrid_without_averaging_or_roi() {
    let w = Work::new("hybrid-plain");
    let s = Settings {
        tavg: 0,
        prewarp_freeze_at: 0,
        ..hybrid_settings()
    };
    exercise(&w, "h", &w.spec("whole", s, false));
}

#[test]
fn chain_with_averaging() {
    let w = Work::new("chain-tavg");
    exercise(&w, "c", &w.spec("whole", chain_settings(2), false));
}

#[test]
fn chain_with_a_roi_and_no_averaging() {
    let w = Work::new("chain-roi");
    exercise(&w, "c", &w.spec("whole", chain_settings(0), true));
}

#[test]
fn a_range_with_a_step_and_a_scale() {
    let w = Work::new("range");
    let mut spec = w.spec("whole", hybrid_settings(), true);
    (spec.start, spec.stop, spec.step) = (1, Some(13), 2);
    // a stop inside the video: the averaging window of the last sample reaches past it
    assert_eq!(run(&spec, None).0, End::Finished);
    let last = *list_frames(&spec.out).last().unwrap();
    assert_eq!(last, 5, "samples 1, 3, ..., 11");
    let mut part = spec.clone();
    part.out = w.base.join("part");
    part.autosave = Some(ON_STOP);
    assert_eq!(run(&part, Some(2)).0, End::Cancelled);
    part.resume = true;
    assert_eq!(run(&part, None).0, End::Finished);
    assert_same_tree(&part.out, &spec.out, "range");
    // a scaled run: its ROI and maps are smaller than the input
    let scaled = Settings {
        scale: 2,
        ..chain_settings(1)
    };
    let mut spec = w.spec("whole2", scaled, false);
    spec.settings.max_disp = 16.0;
    assert_eq!(run(&spec, None).0, End::Finished);
    let mut part = spec.clone();
    part.out = w.base.join("part2");
    part.autosave = Some(ON_STOP);
    assert_eq!(run(&part, Some(4)).0, End::Cancelled);
    part.resume = true;
    assert_eq!(run(&part, None).0, End::Finished);
    assert_same_tree(&part.out, &spec.out, "scaled");
}

#[test]
fn a_video_input() {
    if !optflow_cli::video::tools_available() {
        eprintln!("skipped: no ffmpeg here");
        return;
    }
    let w = Work::new("video");
    let video = w.base.join("input.avi");
    let made = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-framerate", "10", "-i"])
        .arg(w.input.join("f%03d.png"))
        .args(["-c:v", "ffv1", "-pix_fmt", "gray"])
        .arg(&video)
        .status()
        .unwrap();
    assert!(made.success());
    let mut spec = w.spec("whole", hybrid_settings(), true);
    spec.source = video.to_string_lossy().into_owned();
    spec.is_video = true;
    assert_eq!(run(&spec, None).0, End::Finished);
    for stop in [1, 6] {
        let mut part = spec.clone();
        part.out = w.base.join(format!("part{stop}"));
        part.autosave = Some(ON_STOP);
        assert_eq!(run(&part, Some(stop)).0, End::Cancelled);
        part.resume = true;
        assert_eq!(run(&part, None).0, End::Finished);
        assert_same_tree(&part.out, &spec.out, &format!("video, stop {stop}"));
    }
}

/// Run by [`killed_runs_resume`] in a process of its own, to be killed.
#[test]
#[ignore = "a helper of killed_runs_resume"]
fn helper_run_until_killed() {
    let Ok(dir) = std::env::var("OPTFLOW_RESUME_HELPER") else {
        return;
    };
    let w = Work {
        base: PathBuf::from(&dir),
        input: Path::new(&dir).join("input"),
    };
    let mut spec = w.spec("killed", hybrid_settings(), true);
    spec.autosave = Some(EVERY_FRAME);
    run(&spec, None);
    // a kill that came late leaves the folder to the test
    std::mem::forget(w);
}

#[test]
fn killed_runs_resume() {
    let w = Work::new("kill");
    let whole = w.spec("whole", hybrid_settings(), true);
    assert_eq!(run(&whole, None).0, End::Finished);
    let mut spec = w.spec("killed", hybrid_settings(), true);
    spec.autosave = Some(EVERY_FRAME);
    // kills at a checkpoint of 2, then of 7 frames, at different moments of the following work
    for (round, target) in [2usize, 7].into_iter().enumerate() {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "helper_run_until_killed", "--ignored"])
            .env("OPTFLOW_RESUME_HELPER", &w.base)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let ck = spec.out.join(FILE);
        let t0 = std::time::Instant::now();
        loop {
            assert!(
                t0.elapsed().as_secs() < 300,
                "no checkpoint of {target} frames"
            );
            if let Ok(s) = checkpoint::summary(&ck) {
                if s.head.k >= target {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if round == 1 {
            std::thread::sleep(std::time::Duration::from_millis(40));
        }
        child.kill().unwrap();
        child.wait().unwrap();
        let saved = checkpoint::summary(&ck).unwrap().head.k;
        assert!(saved < FRAMES - 1, "the kill came too late: {saved}");
        // the next attempt of this loop starts from scratch: it must not see the old folder
        if round == 0 {
            std::fs::remove_dir_all(&spec.out).unwrap();
        }
    }
    // frames after the checkpoint may be there, whole or in part: make the last one partial
    let saved = checkpoint::summary(&spec.out.join(FILE)).unwrap().head.k;
    let next = spec.out.join(format!("frame_{:05}", saved + 1));
    std::fs::create_dir_all(&next).unwrap();
    std::fs::write(next.join("ux.npy"), b"cut short").unwrap();
    std::fs::remove_file(next.join("gmax.npy")).ok();
    // and a checkpoint that was being written when the power went
    std::fs::write(spec.out.join(TMP), b"half of a checkpoint").unwrap();
    spec.resume = true;
    assert_eq!(run(&spec, None).0, End::Finished);
    assert!(!spec.out.join(TMP).exists());
    assert_same_tree(&spec.out, &whole.out, "killed");
}

#[test]
fn what_differs_is_refused_and_nothing_is_deleted() {
    let w = Work::new("refuse");
    let mut spec = w.spec("run", chain_settings(1), true);
    spec.autosave = Some(ON_STOP);
    assert_eq!(run(&spec, Some(3)).0, End::Cancelled);
    let before = tree(&spec.out);
    spec.resume = true;
    let refused = |s: &JobSpec, needle: &str| {
        let (end, frames, _) = run(s, None);
        let End::Failed(text) = end else {
            panic!("{needle}: {end:?}")
        };
        assert!(text.contains(needle), "{text}");
        assert!(frames.is_empty(), "{needle}");
        assert!(tree(&s.out) == before, "{needle}: the folder changed");
    };
    let mut s = spec.clone();
    s.roi[0].0 += 1.0;
    refused(&s, "ROI");
    let mut s = spec.clone();
    s.settings.sigma_w += 0.5;
    refused(&s, "sigma_w");
    let mut s = spec.clone();
    s.settings.seq_mode = "hybrid".into();
    refused(&s, "seq_mode");
    let mut s = spec.clone();
    s.step = 2;
    refused(&s, "開始・終了・間隔");
    // another update time of a file that was already used, then a new file
    let f = w.input.join("f002.png");
    let t = std::fs::metadata(&f).unwrap().modified().unwrap();
    let file = std::fs::File::options().write(true).open(&f).unwrap();
    file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1000))
        .unwrap();
    refused(&spec, "入力");
    file.set_modified(t).unwrap();
    std::fs::copy(w.input.join("f000.png"), w.input.join("f999.png")).unwrap();
    refused(&spec, "入力");
    std::fs::remove_file(w.input.join("f999.png")).unwrap();
    // a checkpoint cut short, or of another format
    let ck = spec.out.join(FILE);
    let good = std::fs::read(&ck).unwrap();
    std::fs::write(&ck, &good[..good.len() - 1]).unwrap();
    let (end, _, _) = run(&spec, None);
    assert!(
        matches!(&end, End::Failed(t) if t.contains("壊れて")),
        "{end:?}"
    );
    let mut other = good.clone();
    other[8] = 99;
    std::fs::write(&ck, &other).unwrap();
    let (end, _, _) = run(&spec, None);
    assert!(
        matches!(&end, End::Failed(t) if t.contains("形式")),
        "{end:?}"
    );
    // a flipped bit in an array is found when it is read, before any frame is tracked
    let mut flipped = good.clone();
    *flipped.last_mut().unwrap() ^= 1;
    std::fs::write(&ck, &flipped).unwrap();
    let (end, frames, _) = run(&spec, None);
    assert!(
        matches!(&end, End::Failed(t) if t.contains("壊れて")),
        "{end:?}"
    );
    assert!(frames.is_empty());
    // and it still works once all is as it was
    std::fs::write(&ck, &good).unwrap();
    assert_eq!(run(&spec, None).0, End::Finished);
}

#[test]
fn an_input_that_gives_other_bits_is_refused_by_the_frame_checksums() {
    let w = Work::new("bits");
    let mut spec = w.spec("run", chain_settings(0), false);
    spec.autosave = Some(ON_STOP);
    assert_eq!(run(&spec, Some(3)).0, End::Cancelled);
    // another picture with the old name, size and time: only the content can tell
    let f = w.input.join("f003.png");
    let meta = std::fs::metadata(&f).unwrap();
    let (len, t) = (meta.len(), meta.modified().unwrap());
    let mut bytes = std::fs::read(&f).unwrap();
    let other = std::fs::read(w.input.join("f000.png")).unwrap();
    bytes = other.into_iter().chain(bytes).take(len as usize).collect();
    std::fs::write(&f, &bytes).unwrap();
    let file = std::fs::File::options().write(true).open(&f).unwrap();
    file.set_modified(t).unwrap();
    spec.resume = true;
    let (end, frames, _) = run(&spec, None);
    assert!(matches!(&end, End::Failed(_)), "{end:?}");
    assert!(frames.is_empty());
}

#[test]
fn a_crack_run_stops_cleanly_and_cannot_be_resumed() {
    let w = Work::new("crack");
    let s = Settings {
        crack: true,
        ..chain_settings(0)
    };
    let mut spec = w.spec("run", s, true);
    spec.autosave = Some(EVERY_FRAME);
    assert_eq!(run(&spec, Some(2)).0, End::Cancelled);
    assert!(!spec.out.join(FILE).exists());
    assert_eq!(list_frames(&spec.out), vec![1, 2]);
    spec.resume = true;
    let (end, _, _) = run(&spec, None);
    assert_eq!(end, End::Failed(CRACK_NO_RESUME.to_string()));
}

#[test]
fn a_failed_autosave_does_not_stop_the_run_and_keeps_the_last_checkpoint() {
    let w = Work::new("nosave");
    let whole = w.spec("whole", chain_settings(0), false);
    assert_eq!(run(&whole, None).0, End::Finished);
    let mut spec = w.spec("run", chain_settings(0), false);
    spec.autosave = Some(EVERY_FRAME);
    // from frame 4 on the temporary name is taken by a folder
    let tmp = spec.out.join(TMP);
    let cancel = AtomicBool::new(false);
    let mut warnings = 0;
    let mut end = None;
    run_job(&spec, &cancel, &mut |m| match m {
        Msg::Frame { stats } if stats.k == 3 => {
            std::fs::create_dir_all(&tmp).unwrap();
        }
        Msg::Frame { stats } if stats.k == 6 => cancel.store(true, Ordering::Relaxed),
        Msg::Warn(t) if t.contains("checkpoint not saved") => warnings += 1,
        Msg::Finished | Msg::Cancelled => end = Some(matches!(m, Msg::Cancelled)),
        _ => {}
    });
    assert_eq!(end, Some(true));
    assert_eq!(warnings, 4, "frames 3 to 6");
    let s = checkpoint::summary(&spec.out.join(FILE)).unwrap();
    assert_eq!(s.head.k, 2);
    std::fs::remove_dir_all(&tmp).unwrap();
    spec.resume = true;
    assert_eq!(run(&spec, None).0, End::Finished);
    assert_same_tree(&spec.out, &whole.out, "after a failed autosave");
}

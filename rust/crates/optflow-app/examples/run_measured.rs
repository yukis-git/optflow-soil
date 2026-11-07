//! Measurements for the speed and checkpoint sections of the notes: a run through the same library
//! path as the window, with or without the view thread at work, printing per-frame times and every
//! checkpoint. Not part of the app.
//!
//! `run_measured SOURCE OUT [--start N] [--stop N] [--roi x,y;x,y;...] [--live] [--every] [--off]
//!  [--stop-at K] [--resume] [--chain] [--standard] [--scale N] [--tavg N] [--gpu]`

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use optflow_app::job::{run_job, JobSpec, Msg};
use optflow_app::settings::{Preset, Settings};
use optflow_app::source::Source;
use optflow_app::stats::Overlay;
use optflow_app::view::{ViewKey, ViewWorker, MAX_TEXTURE_SIDE};
use optflow_cli::checkpoint::Policy;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (source, out) = (args[0].clone(), PathBuf::from(&args[1]));
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let int = |name: &str| value(name).map(|v| v.parse::<i64>().unwrap());
    let roi = value("--roi").map_or(vec![], |t| {
        t.split(';')
            .map(|p| {
                let (x, y) = p.split_once(',').unwrap();
                (x.parse().unwrap(), y.parse().unwrap())
            })
            .collect()
    });
    let src = Source::open(std::path::Path::new(&source)).unwrap();
    let mut settings = Settings::preset(Preset::Recommended);
    if flag("--standard") {
        settings = Settings::preset(Preset::Standard);
    }
    if let Some(n) = int("--scale") {
        settings.scale = n;
    }
    if flag("--chain") {
        settings.seq_mode = "chain".into();
    }
    if flag("--gpu") {
        settings.device = optflow_cli::args::Device::Gpu;
    }
    if let Some(t) = int("--tavg") {
        settings.tavg = t;
    }
    let autosave = if flag("--off") {
        None
    } else if flag("--every") {
        Some(Policy {
            interval_s: 0.0,
            max_cost: f64::INFINITY,
        })
    } else {
        Some(Policy::default())
    };
    let spec = JobSpec {
        source: src.path.clone(),
        is_video: src.is_video,
        source_count: src.count,
        roi,
        start: int("--start").unwrap_or(0),
        stop: int("--stop"),
        step: 1,
        settings,
        out: out.clone(),
        autosave,
        resume: flag("--resume"),
    };
    let stop_at = int("--stop-at").map(|k| k as usize);
    let live = flag("--live");
    let worker = live.then(|| ViewWorker::start(Arc::new(|| {})));
    let bg = live.then(|| Arc::new(src.frame(spec.start).unwrap()));
    let cancel = AtomicBool::new(false);
    let t0 = Instant::now();
    let mut last = t0;
    let (mut shown, mut asked) = (0, 0);
    run_job(&spec, &cancel, &mut |m| match m {
        Msg::Frame { stats } => {
            let now = Instant::now();
            println!(
                "frame {} {:.3} valid {:.4}",
                stats.k,
                (now - last).as_secs_f64(),
                stats.valid_frac
            );
            last = now;
            if let (Some(w), Some(bg)) = (&worker, &bg) {
                // the window's request for the newest frame; the answers are taken without waiting
                while let Ok(r) = w.rx.try_recv() {
                    if r.is_ok() {
                        shown += 1;
                    }
                }
                let key = ViewKey::new(
                    out.clone(),
                    stats.k,
                    Overlay::Gmax,
                    true,
                    0.0,
                    0.0,
                    0.7,
                    MAX_TEXTURE_SIDE,
                );
                w.request(Arc::clone(bg), key);
                asked += 1;
            }
            if Some(stats.k) == stop_at {
                cancel.store(true, Ordering::Relaxed);
            }
        }
        Msg::Restored { stats } => {
            println!(
                "restored {} at {:.3}",
                stats.len(),
                t0.elapsed().as_secs_f64()
            );
            last = Instant::now();
        }
        Msg::Saved(i) => println!("saved {} {} {:.3}", i.k, i.bytes, i.secs),
        Msg::Finished => println!("finished"),
        Msg::Cancelled => println!("cancelled"),
        Msg::Failed(t) => println!("failed {t}"),
        Msg::Warn(t) => println!("warn {t}"),
        Msg::Started { .. } => {}
    });
    println!("total {:.3}", t0.elapsed().as_secs_f64());
    if live {
        println!("view asked {asked} shown {shown}");
    }
    if flag("--time-stats") {
        // what the worker does after each saved frame, on the tracker's thread
        let k = optflow_app::fields::list_frames(&out)[0];
        for _ in 0..3 {
            let t = Instant::now();
            let f = optflow_app::fields::Fields::load(&out, k).unwrap();
            let load = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let s = optflow_app::stats::frame_stats(k, &f, None);
            println!(
                "frame_done: load {load:.3} s, statistics {:.3} s ({})",
                t.elapsed().as_secs_f64(),
                s.k
            );
        }
        // the work of the view thread for one frame
        let bg = src.frame(spec.start).unwrap();
        for _ in 0..3 {
            let mut cache = optflow_app::view::ViewCache::default();
            let key = ViewKey::new(
                out.clone(),
                k,
                Overlay::Gmax,
                true,
                0.0,
                0.0,
                0.7,
                MAX_TEXTURE_SIDE,
            );
            let t = Instant::now();
            let r = optflow_app::view::render(&mut cache, &bg, &key).unwrap();
            println!(
                "view thread: {:.3} s for one picture (reduced by {})",
                t.elapsed().as_secs_f64(),
                r.factor
            );
        }
    }
}

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
//! optflow-app: the egui front end. All logic lives in the library; this file is layout and event
//! handling.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{self, Color32, RichText, Stroke};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use optflow_app::colormap::{color, Scale};
use optflow_app::export::{self, Timing};
use optflow_app::fields::Fields;
use optflow_app::job::{self, JobSpec, Msg, Running};
use optflow_app::resume::{self, Mismatch};
use optflow_app::roi_edit::{self, Cursor, Drag, Handle};
use optflow_app::settings::{Preset, Settings};
use optflow_app::source::{Gray8, Source};
use optflow_app::stats::{FrameStats, Overlay};
use optflow_app::view::{self, ViewKey, ViewWorker};
use optflow_app::{fonts, messages, power};
use optflow_cli::checkpoint::{self, Policy, SaveInfo, Summary};

/// How far from a ROI handle the pointer still takes it, in screen pixels at any zoom.
const HANDLE_RADIUS: f32 = 9.0;

/// A stopped run found in the output folder.
struct Offer {
    dir: PathBuf,
    /// What it says, or why it cannot be used.
    summary: Result<Summary, String>,
    /// The input is not what it was (checked when the folder is looked at and again on resume).
    input_changed: bool,
}

/// What the start button asks when a stopped run is there.
#[derive(PartialEq)]
enum Ask {
    None,
    FreshRun,
}

/// The scripted drive of the window (`--selftest-interactive`): real pointer events go through the
/// same input path as a mouse's.
#[derive(Default)]
struct Script {
    source: PathBuf,
    out: PathBuf,
    stage: usize,
    wait: u32,
    /// Input events still to send, one group per frame.
    events: std::collections::VecDeque<Vec<egui::Event>>,
    t0: Option<Instant>,
    /// Frame at which to press stop.
    stop_at: usize,
    /// Quit after the stop, as a user closing the window.
    quit_after_stop: bool,
    /// Open the input with a stopped run in the folder: nothing is drawn, the run goes on.
    reopen: bool,
    checked_refusal: bool,
}

/// Test mode (`--selftest SOURCE "x,y;x,y;..." OUTDIR`): the window does what a user would do,
/// driven by code instead of clicks, and saves screenshots of the window.
struct Auto {
    source: PathBuf,
    roi: Vec<(f64, f64)>,
    out: PathBuf,
    crack: bool,
    /// Use the "標準" preset instead of "推奨".
    standard: bool,
    shot_cod: bool,
    shot_running: bool,
    stage: usize,
    /// Frames to wait before the next stage (lets a screenshot reach the screen).
    wait: u32,
    t0: Instant,
}

struct App {
    source: Option<Source>,
    first: Option<Arc<Gray8>>,
    first_tex: Option<egui::TextureHandle>,
    roi: Vec<(f64, f64)>,
    start: i64,
    /// 0: to the end.
    stop: i64,
    step: i64,
    fps: f64,
    settings: Settings,
    base_out: PathBuf,
    /// The user chose `base_out` in the dialog; opening an input no longer moves it.
    out_picked: bool,
    // run
    rx: Option<Receiver<Msg>>,
    running: Option<Running>,
    spec: Option<JobSpec>,
    total: Option<usize>,
    t0: Option<Instant>,
    elapsed_final: f64,
    series: Vec<FrameStats>,
    warnings: Vec<String>,
    error: Option<String>,
    notice: Option<String>,
    // results
    run_dir: Option<PathBuf>,
    cur: usize,
    fields: Option<Arc<Fields>>,
    crack: Option<optflow_app::crack::CrackLines>,
    overlay: Overlay,
    auto_scale: bool,
    lo: f32,
    hi: f32,
    alpha: f32,
    tex: Option<egui::TextureHandle>,
    /// Prepares the shown picture off this thread.
    view: ViewWorker,
    /// The picture last asked for, and the last one that came back for what was asked.
    view_asked: Option<ViewKey>,
    view_done: Option<ViewKey>,
    export_rx: Option<Receiver<Result<String, String>>>,
    ffmpeg_ok: bool,
    auto: Option<Auto>,
    script: Option<Script>,
    // ROI editing
    roi_drag: Option<Drag>,
    /// The ROI was drawn by a drag: a click no longer adds vertices.
    roi_closed: bool,
    /// Where the picture is on the screen, and the screen pixels per image pixel.
    image_area: Option<(egui::Rect, f32)>,
    // stop, autosave, resume
    /// Minutes between autosaves; 0: only when stopped.
    autosave_min: f64,
    last_save: Option<(Instant, SaveInfo)>,
    stop_asked: bool,
    offer: Option<Offer>,
    ask: Ask,
    /// Results that were already there when this run began.
    resumed_frames: usize,
    // live view
    /// Show each new result while the run goes on.
    live: bool,
    /// The shown frame follows the newest one.
    follow: bool,
    /// Where the buttons are, for the scripted drive.
    btn_start: Option<egui::Rect>,
    btn_stop: Option<egui::Rect>,
    btn_resume: Option<egui::Rect>,
    btn_restore: Option<egui::Rect>,
    btn_ask_cancel: Option<egui::Rect>,
}

fn setup_fonts(ctx: &egui::Context) -> Option<PathBuf> {
    let (path, bytes) = fonts::load_first(&fonts::candidates())?;
    let mut defs = egui::FontDefinitions::default();
    defs.font_data
        .insert("cjk".into(), Arc::new(egui::FontData::from_owned(bytes)));
    for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        defs.families.entry(fam).or_default().push("cjk".into());
    }
    ctx.set_fonts(defs);
    Some(path)
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, auto: Option<Auto>, script: Option<Script>) -> App {
        setup_fonts(&cc.egui_ctx);
        let base_out = std::env::current_dir()
            .ok()
            .filter(|p| p.is_dir())
            .map(|p| p.join("optflow_出力"))
            .unwrap_or_else(|| PathBuf::from("optflow_出力"));
        let c = cc.egui_ctx.clone();
        let view = ViewWorker::start(Arc::new(move || c.request_repaint()));
        App {
            source: None,
            first: None,
            first_tex: None,
            roi: vec![],
            start: 0,
            stop: 0,
            step: 1,
            fps: 1.0,
            settings: Settings::preset(Preset::Recommended),
            base_out,
            out_picked: false,
            rx: None,
            running: None,
            spec: None,
            total: None,
            t0: None,
            elapsed_final: 0.0,
            series: vec![],
            warnings: vec![],
            error: None,
            notice: None,
            run_dir: None,
            cur: 0,
            fields: None,
            crack: None,
            overlay: Overlay::Gmax,
            auto_scale: true,
            lo: 0.0,
            hi: 1.0,
            alpha: 0.7,
            tex: None,
            view,
            view_asked: None,
            view_done: None,
            export_rx: None,
            ffmpeg_ok: optflow_cli::video::tools_available(),
            auto,
            script,
            roi_drag: None,
            roi_closed: false,
            image_area: None,
            autosave_min: 5.0,
            last_save: None,
            stop_asked: false,
            offer: None,
            ask: Ask::None,
            resumed_frames: 0,
            live: true,
            follow: true,
            btn_start: None,
            btn_stop: None,
            btn_resume: None,
            btn_restore: None,
            btn_ask_cancel: None,
        }
    }

    fn set_source(
        &mut self,
        ctx: &egui::Context,
        src: Result<(Source, Option<(i64, i64)>), anyhow::Error>,
    ) {
        match src {
            Ok((s, range)) => {
                self.error = None;
                if !self.out_picked {
                    if let Some(p) = job::default_out(Path::new(&s.path)) {
                        self.base_out = p;
                    }
                }
                self.fps = s.fps;
                if let Some((a, b)) = range {
                    self.start = a;
                    self.stop = b;
                } else {
                    self.start = 0;
                    self.stop = 0;
                }
                self.roi.clear();
                self.roi_closed = false;
                self.roi_drag = None;
                self.close_results();
                if self.load_first_frame(ctx, &s) {
                    self.source = Some(s);
                    self.look_for_stopped_run(ctx, true);
                }
            }
            Err(e) => self.error = Some(messages::describe(&e)),
        }
    }

    /// The picture for ROI drawing; ROI coordinates stay in full-size pixels.
    fn load_first_frame(&mut self, ctx: &egui::Context, s: &Source) -> bool {
        match s.frame(self.start) {
            Ok(g) => {
                let factor = view::display_factor(g.w, g.h, max_texture_side(ctx));
                let (w, h, data) = view::shrink_gray(&g, factor);
                let img = egui::ColorImage::from_gray([w, h], &data);
                self.first_tex = Some(ctx.load_texture("first", img, egui::TextureOptions::LINEAR));
                self.first = Some(Arc::new(g));
                true
            }
            Err(e) => {
                self.error = Some(messages::describe(&e));
                false
            }
        }
    }

    /// Looks in the output folder for a stopped run and offers it. With `adopt`, and a ROI not yet
    /// drawn, the window takes over the input, ROI, range and settings of that run, so that going
    /// on takes one click; anything else stays, and the offer says what differs.
    fn look_for_stopped_run(&mut self, ctx: &egui::Context, adopt: bool) {
        let found = resume::find(&self.base_out);
        // the one of this input first, else the newest
        let of_this_input = |f: &resume::Found| {
            let source = self.source.as_ref().map(|s| s.path.as_str());
            f.summary
                .as_ref()
                .is_ok_and(|s| s.meta.get("source").is_some() && s.meta.get("source") == source)
        };
        let pick = found
            .iter()
            .position(of_this_input)
            .or(if found.is_empty() { None } else { Some(0) });
        self.offer = pick.and_then(|i| found.into_iter().nth(i)).map(|f| {
            let input_changed = f.summary.as_ref().is_ok_and(|s| {
                let (Some(src), Some(video)) = (s.meta.get("source"), s.meta.get("is_video"))
                else {
                    return true;
                };
                match resume::input_identity(src, video == "true") {
                    Ok((n, d)) => {
                        s.meta.get("input_files") != Some(n.to_string().as_str())
                            || s.meta.get("input_digest") != Some(format!("{d:016x}").as_str())
                    }
                    Err(_) => true,
                }
            });
            Offer {
                dir: f.dir,
                summary: f.summary,
                input_changed,
            }
        });
        if adopt && self.roi.is_empty() {
            let usable = self
                .offer
                .as_ref()
                .is_some_and(|o| o.summary.is_ok() && !o.input_changed);
            if usable {
                self.adopt_offer(ctx);
            }
        }
    }

    /// Brings the values of the offered run into the window.
    fn adopt_offer(&mut self, ctx: &egui::Context) {
        let Some(saved) = self
            .offer
            .as_ref()
            .and_then(|o| o.summary.as_ref().ok())
            .and_then(|s| resume::saved_values(&s.meta))
        else {
            return;
        };
        let same_input = self.source.as_ref().is_some_and(|s| s.path == saved.source);
        let start_moved = self.start != saved.start;
        if !same_input {
            match Source::open(Path::new(&saved.source)) {
                Ok(s) => {
                    self.fps = s.fps;
                    self.source = Some(s);
                }
                Err(e) => {
                    self.error = Some(messages::describe(&e));
                    return;
                }
            }
        }
        self.start = saved.start;
        self.stop = saved.stop.unwrap_or(0);
        self.step = saved.step;
        self.settings = saved.settings;
        self.roi = saved.roi;
        self.roi_closed = self.roi.len() >= 3;
        if !same_input || start_moved {
            if let Some(s) = self.source.clone() {
                self.load_first_frame(ctx, &s);
            }
        }
        self.notice =
            Some("中断した解析が見つかりました。ROI と設定を中断時のものにしました。".into());
    }

    /// How the offered run differs from what the window would run now.
    fn offer_mismatches(&self) -> Vec<Mismatch> {
        let (Some(o), Some(spec)) = (&self.offer, self.spec(None)) else {
            return vec![];
        };
        let Ok(s) = &o.summary else {
            return vec![];
        };
        let mut diff = resume::compare(&s.meta, &resume::plain_meta(&spec));
        if o.input_changed && !diff.contains(&Mismatch::Input) {
            diff.insert(0, Mismatch::Input);
        }
        diff
    }

    fn close_results(&mut self) {
        self.run_dir = None;
        self.fields = None;
        self.crack = None;
        self.series.clear();
        self.tex = None;
        self.view_asked = None;
        self.view_done = None;
        self.warnings.clear();
        self.notice = None;
        self.last_save = None;
        self.resumed_frames = 0;
    }

    /// The run of what the window shows, into `out` or a new run folder.
    fn spec(&self, out: Option<PathBuf>) -> Option<JobSpec> {
        let s = self.source.as_ref()?;
        Some(JobSpec {
            source: s.path.clone(),
            is_video: s.is_video,
            source_count: s.count,
            roi: self.roi.clone(),
            start: self.start,
            stop: (self.stop > 0).then_some(self.stop),
            step: self.step.max(1),
            settings: self.settings.clone(),
            out: out.unwrap_or_else(|| job::next_run_dir(&self.base_out)),
            autosave: Some(Policy {
                interval_s: if self.autosave_min > 0.0 {
                    self.autosave_min * 60.0
                } else {
                    f64::INFINITY
                },
                ..Policy::default()
            }),
            resume: false,
        })
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        let Some(spec) = self.spec(None) else { return };
        self.launch(ctx, spec);
    }

    /// Continues the offered run.
    fn resume_run(&mut self, ctx: &egui::Context) {
        let Some(dir) = self.offer.as_ref().map(|o| o.dir.clone()) else {
            return;
        };
        let Some(mut spec) = self.spec(Some(dir)) else {
            return;
        };
        spec.resume = true;
        self.launch(ctx, spec);
    }

    fn launch(&mut self, ctx: &egui::Context, spec: JobSpec) {
        self.close_results();
        self.offer = None;
        self.ask = Ask::None;
        self.stop_asked = false;
        self.follow = true;
        self.error = None;
        let (tx, rx) = channel();
        let c = ctx.clone();
        self.running = Some(job::spawn(
            spec.clone(),
            tx,
            Arc::new(move || c.request_repaint()),
        ));
        self.rx = Some(rx);
        self.run_dir = Some(spec.out.clone());
        self.spec = Some(spec);
        self.total = None;
        self.t0 = Some(Instant::now());
        self.cur = 0;
    }

    fn poll(&mut self) {
        let Some(rx) = self.rx.take() else { return };
        let mut done = false;
        let mut stopped = false;
        while let Ok(m) = rx.try_recv() {
            match m {
                Msg::Started { total } => self.total = total,
                Msg::Frame { stats } => {
                    if self.follow {
                        self.cur = self.series.len();
                    }
                    self.series.push(stats);
                }
                Msg::Restored { stats } => {
                    self.resumed_frames = stats.len();
                    self.series = stats;
                    self.cur = self.series.len().saturating_sub(1);
                }
                Msg::Saved(info) => self.last_save = Some((Instant::now(), info)),
                Msg::Warn(w) => self.warnings.push(w),
                Msg::Finished => {
                    self.notice = Some("完了しました。".into());
                    done = true;
                }
                Msg::Cancelled => {
                    let crack = self.spec.as_ref().is_some_and(|s| s.settings.crack);
                    self.notice = Some(if crack {
                        format!(
                            "中止しました(ここまでの結果は表示・書き出しできます)。{}",
                            job::CRACK_NO_RESUME
                        )
                    } else {
                        "中止しました。途中経過を保存したので、続きから再開できます(ここまでの結果は表示・書き出しもできます)。"
                            .into()
                    });
                    stopped = true;
                    done = true;
                }
                Msg::Failed(t) => {
                    self.error = Some(t);
                    done = true;
                }
            }
        }
        if done {
            self.elapsed_final = self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
            if let Some(r) = self.running.take() {
                let _ = r.handle.join();
            }
            self.stop_asked = false;
            if self.follow {
                self.cur = self.series.len().saturating_sub(1);
            }
            if stopped || self.error.is_some() {
                let dir = self.spec.as_ref().map(|s| s.out.clone());
                self.offer_from(dir);
            }
        } else {
            self.rx = Some(rx);
        }
    }

    /// Offers the checkpoint of `dir` if it has one.
    fn offer_from(&mut self, dir: Option<PathBuf>) {
        self.offer = dir.and_then(|dir| {
            let path = dir.join(checkpoint::FILE);
            path.is_file().then(|| Offer {
                summary: checkpoint::summary(&path).map_err(|e| resume::describe_error(&e)),
                dir,
                input_changed: false,
            })
        });
    }

    fn timing(&self) -> Timing {
        Timing {
            start: self.start,
            step: self.step.max(1),
            fps: if self.fps > 0.0 { self.fps } else { 1.0 },
        }
    }

    /// Takes in what the view worker finished and asks for the selected picture if it changed.
    /// Never reads or computes anything itself: a 6240x4160 frame takes seconds.
    fn refresh_view(&mut self, ctx: &egui::Context) {
        while let Ok(r) = self.view.rx.try_recv() {
            match r {
                Ok(r) => {
                    // one overtaken by a newer request still beats the older picture on show
                    if self.run_dir.as_ref() != Some(&r.key.run) {
                        continue;
                    }
                    if self.view_asked.as_ref() == Some(&r.key) {
                        if !self.overlay.available(&r.fields) {
                            self.overlay = Overlay::Gmax;
                        }
                        if r.key.auto {
                            (self.lo, self.hi) = (r.lo, r.hi);
                        }
                        self.view_done = Some(r.key.clone());
                    }
                    let options = if r.factor > 1 {
                        egui::TextureOptions::LINEAR
                    } else {
                        egui::TextureOptions::NEAREST
                    };
                    self.tex = r.image.map(|i| {
                        let img = egui::ColorImage::from_rgb([i.w, i.h], &i.rgb);
                        ctx.load_texture("view", img, options)
                    });
                    self.crack = r.crack;
                    self.fields = Some(r.fields);
                }
                Err((key, text)) => {
                    if self.view_asked.as_ref() == Some(&key) {
                        self.error = Some(text);
                        self.view_done = Some(key);
                    }
                }
            }
        }
        let (Some(run), Some(st), Some(bg)) = (
            self.run_dir.clone(),
            self.series.get(self.cur),
            self.first.as_ref(),
        ) else {
            return;
        };
        let key = ViewKey::new(
            run,
            st.k,
            self.overlay,
            self.auto_scale,
            self.lo,
            self.hi,
            self.alpha,
            max_texture_side(ctx),
        );
        if self.view_asked.as_ref() != Some(&key) {
            self.view.request(Arc::clone(bg), key.clone());
            self.view_asked = Some(key);
        }
    }

    fn view_busy(&self) -> bool {
        self.view_asked != self.view_done
    }

    fn save_png(&mut self, ctx: &egui::Context) {
        let (Some(f), Some(bg)) = (self.fields.clone(), self.first.clone()) else {
            return;
        };
        if (f.w, f.h) != (bg.w, bg.h) {
            self.notice =
                Some("縮小率 2 以上の結果は入力画像と大きさが異なるため PNG にできません。".into());
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("view.png")
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };
        let (overlay, lo, hi, alpha) = (self.overlay, self.lo, self.hi, self.alpha);
        let (roi, crack) = (self.roi.clone(), self.crack.clone());
        let (tx, rx) = channel();
        let c = ctx.clone();
        std::thread::spawn(move || {
            let _awake = power::keep_running("optflow export");
            let r = write_png(&path, &f, &bg, overlay, lo, hi, alpha, &roi, crack.as_ref())
                .map(|()| format!("PNGを保存しました: {}", path.display()))
                .map_err(|e| format!("PNGを保存できません: {e:#}"));
            let _ = tx.send(r);
            c.request_repaint();
        });
        self.export_rx = Some(rx);
        self.notice = Some("PNGを保存中...".into());
    }

    fn start_csv_export(&mut self, ctx: &egui::Context, dest: PathBuf) {
        let Some(run) = self.run_dir.clone() else {
            return;
        };
        let ks: Vec<usize> = self.series.iter().map(|s| s.k).collect();
        let series = self.series.clone();
        let t = self.timing();
        let (tx, rx) = channel();
        let c = ctx.clone();
        std::thread::spawn(move || {
            let _awake = power::keep_running("optflow export");
            let r = export::export_csv(&run, &dest, &ks, &series, &t, &mut |_, _| {})
                .map(|()| format!("CSVを書き出しました: {}", dest.display()))
                .map_err(|e| format!("CSVを書き出せません: {e:#}"));
            let _ = tx.send(r);
            c.request_repaint();
        });
        self.export_rx = Some(rx);
        self.notice = Some("CSVを書き出し中...".into());
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let busy = self.running.is_some();
        ui.heading("1. 入力");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("動画を開く…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter(
                            "動画",
                            &["mp4", "mov", "avi", "mkv", "m4v", "webm", "mts", "wmv"],
                        )
                        .pick_file()
                    {
                        let r = Source::open(&p).map(|s| (s, None));
                        self.set_source(ctx, r);
                    }
                }
                if ui.button("画像フォルダ…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        let r = Source::open(&p).map(|s| (s, None));
                        self.set_source(ctx, r);
                    }
                }
                if ui.button("画像を選択(複数)…").clicked() {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("画像", &["png", "tif", "tiff", "jpg", "jpeg", "bmp"])
                        .pick_files()
                    {
                        let r = Source::from_files(&files).map(|(s, a, b)| (s, Some((a, b))));
                        self.set_source(ctx, r);
                    }
                }
            });
        });
        if !self.ffmpeg_ok {
            ui.colored_label(
                Color32::from_rgb(200, 120, 0),
                "ffmpeg が見つかりません(動画は使えません。画像フォルダは使えます)。",
            );
        }
        if let Some(s) = &self.source {
            let name = std::path::Path::new(&s.path)
                .file_name()
                .map_or(s.path.clone(), |n| n.to_string_lossy().into_owned());
            ui.label(format!(
                "{}: {} ({}×{} px, {})",
                if s.is_video { "動画" } else { "画像列" },
                name,
                s.w,
                s.h,
                s.count
                    .map_or("枚数不明".to_string(), |n| format!("{n} フレーム"))
            ));
        }
        ui.separator();
        ui.heading("2. 範囲と領域(ROI)");
        ui.add_enabled_ui(!busy, |ui| {
            egui::Grid::new("range").num_columns(2).show(ui, |ui| {
                ui.label("開始フレーム");
                ui.add(egui::DragValue::new(&mut self.start).range(0..=10_000_000));
                ui.end_row();
                ui.label("終了フレーム(0=最後まで)");
                ui.add(egui::DragValue::new(&mut self.stop).range(0..=10_000_000));
                ui.end_row();
                ui.label("間隔(フレームおき)");
                ui.add(egui::DragValue::new(&mut self.step).range(1..=100_000));
                ui.end_row();
                ui.label("フレームレート(時間軸用)");
                ui.add(
                    egui::DragValue::new(&mut self.fps)
                        .range(0.001..=10_000.0)
                        .speed(0.1),
                );
                ui.end_row();
            });
            ui.add(
                egui::Label::new(
                    "画像上をドラッグすると四角形の ROI を作ります(クリックで頂点を順に置くこともできます)。\
                     作った後は角と辺の中央の点をドラッグして変更できます。\
                     辺をダブルクリックで頂点を追加、頂点を右クリックで削除します。",
                )
                .wrap(),
            );
            ui.horizontal(|ui| {
                if ui.button("1点戻す").clicked() {
                    self.roi.pop();
                    if self.roi.len() < 3 {
                        self.roi_closed = false;
                    }
                }
                if ui.button("ROIをクリア").clicked() {
                    self.roi.clear();
                    self.roi_closed = false;
                }
                ui.label(format!("頂点 {}", self.roi.len()));
            });
            if self.roi.len() < 3 {
                ui.label("ROI が 3 点未満の間は画像全体を追跡します。");
            }
            if roi_edit::self_intersects(&self.roi) {
                ui.colored_label(
                    Color32::from_rgb(200, 120, 0),
                    "ROI の辺が交差しています(交差した部分は内外が交互になります)。",
                );
            }
            if !self.series.is_empty() && ui.button("結果の表示を閉じて ROI を編集").clicked() {
                self.close_results();
            }
        });
        ui.separator();
        ui.heading("3. 設定");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                for p in [Preset::Recommended, Preset::Standard] {
                    let on = self.settings.matching_preset() == Some(p);
                    if ui.selectable_label(on, p.label()).clicked() {
                        let scale = self.settings.scale;
                        self.settings = Settings::preset(p);
                        self.settings.scale = scale;
                    }
                }
                if self.settings.matching_preset().is_none() {
                    ui.label("(カスタム)");
                }
            });
            egui::CollapsingHeader::new("詳細設定").show(ui, |ui| {
                let s = &mut self.settings;
                egui::Grid::new("adv").num_columns(2).show(ui, |ui| {
                    ui.label("系列モード");
                    egui::ComboBox::from_id_salt("seq_mode")
                        .selected_text(s.seq_mode.clone())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut s.seq_mode, "hybrid".to_string(), "hybrid");
                            ui.selectable_value(&mut s.seq_mode, "chain".to_string(), "chain");
                        });
                    ui.end_row();
                    ui.label("時間平均 tavg(前後フレーム数)");
                    ui.add(egui::DragValue::new(&mut s.tavg).range(0..=20));
                    ui.end_row();
                    ui.label("prewarp_freeze_at(0=無効)");
                    ui.add(egui::DragValue::new(&mut s.prewarp_freeze_at).range(0..=20));
                    ui.end_row();
                    ui.label("ref_update_conf");
                    ui.add(
                        egui::DragValue::new(&mut s.ref_update_conf)
                            .range(0.0..=1.0)
                            .speed(0.01),
                    );
                    ui.end_row();
                    ui.label("max_disp [px]");
                    ui.add(egui::DragValue::new(&mut s.max_disp).range(1.0..=2000.0));
                    ui.end_row();
                    ui.label("conf_min");
                    ui.add(
                        egui::DragValue::new(&mut s.conf_min)
                            .range(0.0..=1.0)
                            .speed(0.01),
                    );
                    ui.end_row();
                    ui.label("sigma_w");
                    ui.add(
                        egui::DragValue::new(&mut s.sigma_w)
                            .range(0.5..=100.0)
                            .speed(0.1),
                    );
                    ui.end_row();
                    ui.label("縮小率(整数)");
                    ui.add(egui::DragValue::new(&mut s.scale).range(1..=16));
                    ui.end_row();
                    ui.label("自動保存の間隔 [分](0=中止時のみ)");
                    ui.add(
                        egui::DragValue::new(&mut self.autosave_min)
                            .range(0.0..=1440.0)
                            .speed(0.5),
                    );
                    ui.end_row();
                });
                ui.label("縮小率 2 以上では変位は縮小後の画素単位になります。");
                ui.label("自動保存は実行時間の約 5 % を超えないように間隔を延ばします。結果には影響しません。");
            });
            // ---- CRACK TOGGLE ---------------------------------------------------------------
            ui.checkbox(&mut self.settings.crack, "亀裂モデル（試験的）");
            if self.settings.crack {
                ui.add(
                    egui::Label::new(
                        "亀裂か帯かの自動判定は十分には検証されていません(縮尺により結果が変わり、余裕が小さい場合があります)。\
                         幅の広い亀裂(穴が約 60 px を超えるもの)は検出の対象外です。\
                         追跡する範囲(ROI で切り出した範囲)は約 830 万画素(3840x2160)以下が必要で、100 万画素あたり約 1.1 GB のメモリを使います。",
                    )
                    .wrap(),
                );
            }
            // ---- end of crack toggle --------------------------------------------------------
        });
        ui.separator();
        ui.heading("4. 実行");
        ui.label("出力先");
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(self.base_out.display().to_string()).wrap());
        });
        ui.add_enabled_ui(!busy, |ui| {
            if ui.button("出力先を変更…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.base_out = p;
                    self.out_picked = true;
                    self.look_for_stopped_run(ctx, true);
                }
            }
        });
        if let Some(text) = self.output_estimate() {
            ui.add(egui::Label::new(text).wrap());
        }
        if !busy {
            self.resume_card(ui, ctx);
        }
        ui.horizontal(|ui| {
            let start = ui.add_enabled(
                !busy && self.source.is_some(),
                egui::Button::new(RichText::new("▶ 解析開始").strong()),
            );
            self.btn_start = Some(start.rect);
            if start.clicked() {
                if self.offer.as_ref().is_some_and(|o| o.summary.is_ok()) {
                    self.ask = Ask::FreshRun;
                } else {
                    self.start_run(ctx);
                }
            }
            let stop = ui.add_enabled(busy && !self.stop_asked, egui::Button::new("■ 中止"));
            self.btn_stop = Some(stop.rect);
            if stop.clicked() {
                if let Some(r) = &self.running {
                    r.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                self.stop_asked = true;
            }
        });
        if self.stop_asked {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("現在のフレームが終わり次第、途中経過を保存して止まります…");
            });
        }
        if let Some((at, info)) = &self.last_save {
            ui.label(format!(
                "最終保存: {} 前(フレーム {} まで、{}、{:.1} 秒)",
                fmt_secs(at.elapsed().as_secs_f64()),
                info.k,
                job::fmt_bytes(info.bytes),
                info.secs
            ));
        } else if busy {
            if self.settings.crack {
                ui.label(job::CRACK_NO_RESUME);
            } else {
                ui.label("最終保存: まだありません");
            }
        }
        if busy {
            let done = self.series.len();
            let el = self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
            match self.total {
                Some(n) if n > 0 => {
                    ui.add(
                        egui::ProgressBar::new((done as f32 / n as f32).min(1.0))
                            .text(format!("{done} / {n} フレーム")),
                    );
                    let new = done.saturating_sub(self.resumed_frames);
                    let eta = if new > 0 {
                        el / new as f64 * (n - done.min(n)) as f64
                    } else {
                        f64::NAN
                    };
                    ui.label(format!(
                        "経過 {} / 残り {}",
                        fmt_secs(el),
                        if eta.is_finite() {
                            fmt_secs(eta)
                        } else {
                            "計算中".into()
                        }
                    ));
                }
                _ => {
                    ui.add(
                        egui::ProgressBar::new(0.0)
                            .animate(true)
                            .text(format!("{done} フレーム処理済み")),
                    );
                    ui.label(format!("経過 {}", fmt_secs(el)));
                }
            }
            if self.total.is_none_or(|n| done < n) {
                ui.label(format!("現在: フレーム {}", done + 1));
            }
        } else if !self.series.is_empty() {
            ui.label(format!("所要時間 {}", fmt_secs(self.elapsed_final)));
        }
        if let Some(e) = &self.error {
            ui.colored_label(Color32::from_rgb(210, 40, 40), e);
        }
        if let Some(n) = &self.notice {
            ui.label(n);
        }
        for w in &self.warnings {
            ui.colored_label(Color32::from_rgb(200, 120, 0), w);
        }
        ui.separator();
        ui.heading("5. 書き出し");
        let has = self.run_dir.is_some() && !self.series.is_empty() && !busy;
        ui.add_enabled_ui(has && self.export_rx.is_none(), |ui| {
            if ui.button("CSV を書き出す…(変位・ひずみ・時系列)").clicked() {
                if let Some(d) = rfd::FileDialog::new().pick_folder() {
                    self.start_csv_export(ctx, d);
                }
            }
            let ready = !self.view_busy();
            if ui
                .add_enabled(ready, egui::Button::new("表示中の画像を PNG で保存…"))
                .clicked()
            {
                self.save_png(ctx);
            }
        });
        if let Some(r) = &self.run_dir {
            ui.add(egui::Label::new(format!("結果フォルダ: {}", r.display())).wrap());
        }
    }

    /// The offer of a stopped run in the output folder.
    fn resume_card(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(o) = &self.offer else { return };
        let name = o
            .dir
            .file_name()
            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
        let mut go = false;
        let mut restore = false;
        let diff = self.offer_mismatches();
        let summary = match &o.summary {
            Ok(s) => Some((s.head.k, s.meta.get("total").map(str::to_string), s.bytes)),
            Err(e) => {
                ui.colored_label(
                    Color32::from_rgb(210, 40, 40),
                    format!("{name}: {e}(データは消していません。新しく解析できます。)"),
                );
                None
            }
        };
        if let Some((k, total, bytes)) = summary {
            ui.group(|ui| {
                let of = total
                    .filter(|t| t != "-")
                    .map_or(String::new(), |t| format!(" / {t}"));
                ui.label(format!(
                    "中断した解析があります: {name}(フレーム {k}{of} まで、{})",
                    job::fmt_bytes(bytes)
                ));
                let can = diff.is_empty();
                let b = ui.add_enabled(can, egui::Button::new(RichText::new("▶ 再開").strong()));
                self.btn_resume = Some(b.rect);
                go = b.clicked();
                if !can {
                    for d in &diff {
                        ui.colored_label(Color32::from_rgb(210, 40, 40), d.text());
                    }
                    ui.label("このままでは再開できません。データは消していません。「解析開始」で新しく解析できます。");
                    if diff.iter().all(Mismatch::restorable) {
                        let b = ui.button("ROI と設定を中断時のものに戻す");
                        self.btn_restore = Some(b.rect);
                        restore = b.clicked();
                    }
                }
            });
        }
        if restore {
            self.adopt_offer(ctx);
        }
        if go {
            self.resume_run(ctx);
        }
    }

    /// The question of the start button when a stopped run is there.
    fn ask_window(&mut self, ctx: &egui::Context) {
        if self.ask != Ask::FreshRun {
            return;
        }
        let name = self
            .offer
            .as_ref()
            .and_then(|o| o.dir.file_name())
            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
        let can_resume = self.offer_mismatches().is_empty();
        let next = job::next_run_dir(&self.base_out);
        let next = next
            .file_name()
            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
        let mut choice = None;
        egui::Window::new("解析の開始")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(format!(
                    "このフォルダには中断した解析 {name} があります。\n新しいフォルダ {next} に最初から解析しますか?既存の結果は消しません。"
                ));
                ui.horizontal(|ui| {
                    if ui.button("最初から解析").clicked() {
                        choice = Some(0);
                    }
                    if ui.add_enabled(can_resume, egui::Button::new("続きから再開")).clicked() {
                        choice = Some(1);
                    }
                    let cancel = ui.button("やめる");
                    self.btn_ask_cancel = Some(cancel.rect);
                    if cancel.clicked() {
                        choice = Some(2);
                    }
                });
            });
        match choice {
            Some(0) => self.start_run(ctx),
            Some(1) => self.resume_run(ctx),
            Some(_) => self.ask = Ask::None,
            None => {}
        }
    }

    /// The size note beside the start button: the maps are written at the full image size, so a
    /// long run on large photos fills a disk whatever the ROI.
    fn output_estimate(&self) -> Option<String> {
        let s = self.source.as_ref()?;
        let results = s.count.map(|n| {
            let stop = (self.stop > 0).then_some(self.stop);
            let picked = optflow_cli::io::slice_indices(n, self.start, stop, self.step.max(1));
            picked.len().saturating_sub(1)
        });
        Some(job::output_note(s.w, s.h, &self.settings, results))
    }

    fn image_panel(&mut self, ui: &mut egui::Ui) {
        let Some(first) = &self.first else {
            ui.centered_and_justified(|ui| {
                ui.label("左の「1. 入力」で動画または画像フォルダを開いてください。");
            });
            return;
        };
        let avail = ui.available_size();
        let (iw, ih) = (first.w as f32, first.h as f32);
        let s = (avail.x / iw)
            .min((avail.y - 24.0).max(50.0) / ih)
            .max(0.05);
        let size = egui::vec2(iw * s, ih * s);
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        self.image_area = Some((rect, s));
        let painter = ui.painter_at(rect);
        let first_size = (first.w, first.h);
        let tex = if self.run_dir.is_some() && self.tex.is_some() {
            self.tex.as_ref()
        } else {
            self.first_tex.as_ref()
        };
        if let Some(t) = tex {
            painter.image(
                t.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        let to_screen = |p: (f64, f64)| rect.min + egui::vec2(p.0 as f32 * s, p.1 as f32 * s);
        let editing = self.running.is_none() && self.run_dir.is_none()
            || self.series.is_empty() && self.running.is_none();
        let radius = f64::from(HANDLE_RADIUS / s);
        let to_image = |pos: egui::Pos2| {
            let v = (pos - rect.min) / s;
            (f64::from(v.x), f64::from(v.y))
        };
        let mut hover_handle = None;
        if editing {
            hover_handle = resp
                .hover_pos()
                .and_then(|p| roi_edit::hit_test(&self.roi, to_image(p), radius));
            if resp.drag_started_by(egui::PointerButton::Primary) {
                let origin = ui.ctx().input(|i| i.pointer.press_origin());
                if let Some(o) = origin {
                    self.roi_drag = roi_edit::begin_drag(&self.roi, to_image(o), radius);
                }
            }
            if let (Some(d), true) = (self.roi_drag, resp.dragged_by(egui::PointerButton::Primary))
            {
                if let Some(pos) = resp.interact_pointer_pos() {
                    roi_edit::apply(&mut self.roi, d, to_image(pos), first_size);
                }
            }
            if let (Some(d), true) = (self.roi_drag, resp.drag_stopped()) {
                roi_edit::finish(&mut self.roi, d);
                self.roi_closed = self.roi.len() >= 3 || self.roi_closed;
                self.roi_drag = None;
            }
            let at = resp.interact_pointer_pos().map(to_image);
            if let (true, Some(p)) = (resp.double_clicked(), at) {
                if roi_edit::hit_test(&self.roi, p, radius).is_none() {
                    if let Some(i) = roi_edit::near_edge(&self.roi, p, radius) {
                        roi_edit::insert_on_edge(&mut self.roi, i, p, first_size);
                    }
                }
            } else if let (true, Some(p)) = (resp.clicked(), at) {
                // a click near what is there is for that; away from it, it places the next vertex
                // of a polygon being drawn point by point
                if !self.roi_closed
                    && roi_edit::hit_test(&self.roi, p, radius).is_none()
                    && roi_edit::near_edge(&self.roi, p, radius).is_none()
                {
                    self.roi.push(roi_edit::clamp(p, first_size));
                }
            }
            if let (true, Some(p)) = (resp.secondary_clicked(), at) {
                if let Some(Handle::Vertex(i)) = roi_edit::hit_test(&self.roi, p, radius) {
                    roi_edit::delete_vertex(&mut self.roi, i);
                }
            }
            let icon = match (&self.roi_drag, hover_handle) {
                (Some(Drag::NewRect { .. }), _) => Some(egui::CursorIcon::Crosshair),
                (Some(_), _) => Some(egui::CursorIcon::Grabbing),
                (None, Some(h)) => Some(match roi_edit::cursor(&self.roi, h) {
                    Cursor::Grab => egui::CursorIcon::Grab,
                    Cursor::ResizeHorizontal => egui::CursorIcon::ResizeHorizontal,
                    Cursor::ResizeVertical => egui::CursorIcon::ResizeVertical,
                    Cursor::ResizeNeSw => egui::CursorIcon::ResizeNeSw,
                    Cursor::ResizeNwSe => egui::CursorIcon::ResizeNwSe,
                }),
                (None, None) if self.roi.is_empty() && resp.hovered() => {
                    Some(egui::CursorIcon::Crosshair)
                }
                _ => None,
            };
            if let Some(icon) = icon {
                ui.ctx().set_cursor_icon(icon);
            }
        }
        let pts: Vec<egui::Pos2> = self.roi.iter().map(|&p| to_screen(p)).collect();
        let stroke = Stroke::new(2.0, Color32::from_rgb(255, 220, 0));
        for w in pts.windows(2) {
            painter.line_segment([w[0], w[1]], stroke);
        }
        if pts.len() >= 3 {
            painter.line_segment([pts[pts.len() - 1], pts[0]], stroke);
        }
        for (i, p) in pts.iter().enumerate() {
            let on = hover_handle == Some(Handle::Vertex(i));
            painter.circle_filled(
                *p,
                if on { 6.0 } else { 4.5 },
                Color32::from_rgb(255, 80, 0),
            );
            if editing {
                painter.circle_stroke(
                    *p,
                    if on { 6.0 } else { 4.5 },
                    Stroke::new(1.0, Color32::WHITE),
                );
            }
        }
        if editing {
            for i in 0..roi_edit::edge_count(&self.roi) {
                let m = to_screen(roi_edit::midpoint(&self.roi, i));
                let half = if hover_handle == Some(Handle::Edge(i)) {
                    5.0
                } else {
                    3.5
                };
                let r = egui::Rect::from_center_size(m, egui::vec2(2.0 * half, 2.0 * half));
                painter.rect_filled(r, 0.0, Color32::from_rgb(255, 220, 0));
                painter.rect_stroke(
                    r,
                    0.0,
                    Stroke::new(1.0, Color32::BLACK),
                    egui::StrokeKind::Outside,
                );
            }
        }
        if let (Some(c), true) = (&self.crack, self.run_dir.is_some()) {
            for (set, col) in [
                (&c.bands, Color32::from_rgb(255, 0, 255)),
                (&c.lines, Color32::from_rgb(0, 255, 255)),
            ] {
                for l in set {
                    let v: Vec<egui::Pos2> = l.iter().map(|p| to_screen((p[0], p[1]))).collect();
                    for w in v.windows(2) {
                        painter.line_segment([w[0], w[1]], Stroke::new(2.0, col));
                    }
                }
            }
        }
        if editing {
            ui.label(if self.roi.is_empty() {
                "ドラッグで ROI を作成"
            } else {
                "角・辺の点をドラッグで変更 / 辺をダブルクリックで頂点追加 / 頂点を右クリックで削除"
            });
        }
    }

    fn results_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.series.is_empty() {
            return;
        }
        let n = self.series.len();
        let st = self.series[self.cur.min(n - 1)].clone();
        let t = self.timing();
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("有効率 {:.1} %", st.valid_frac * 100.0)).size(26.0).strong());
            let mean = self.series.iter().map(|s| s.valid_frac).sum::<f64>() / n as f64;
            let min = self.series.iter().map(|s| s.valid_frac).fold(f64::INFINITY, f64::min);
            ui.label(format!("全フレーム平均 {:.1} %  最小 {:.1} %   平均信頼度 {:.2}   gmax p50/p95/p99 = {:.4} / {:.4} / {:.4}", mean * 100.0, min * 100.0, st.mean_conf, st.g50, st.g95, st.g99));
        });
        if let Some(c) = &self.crack {
            let (cod, csd) = c.medians();
            ui.label(format!(
                "亀裂: 線 {} 本(シアン) / 連続した帯 {} 本(マゼンタ) / COD 中央値 {} px / CSD 中央値 {} px",
                c.lines.len(),
                c.bands.len(),
                if cod.is_finite() { format!("{cod:.3}") } else { "-".into() },
                if csd.is_finite() { format!("{csd:.3}") } else { "-".into() },
            ));
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("フレーム");
            let mut c = self.cur.min(n - 1);
            ui.spacing_mut().slider_width = 240.0;
            if ui
                .add(
                    egui::Slider::new(&mut c, 0..=n - 1)
                        .custom_formatter(|v, _| format!("{}", v as usize + 1))
                        .custom_parser(|s| s.parse::<f64>().ok().map(|v| v - 1.0)),
                )
                .changed()
            {
                self.cur = c;
                self.follow = false;
            }
            if self.running.is_some() {
                let mut on = self.live && self.follow;
                if ui.checkbox(&mut on, "最新を表示").changed() {
                    self.live = on;
                    self.follow = on;
                    if on {
                        self.cur = n - 1;
                    }
                }
            }
            ui.label(format!(
                "第{}結果 / ソース {} 枚目 / {:.2} 秒",
                st.k,
                t.source_frame(st.k),
                t.time_s(st.k)
            ));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("表示");
            egui::ComboBox::from_id_salt("overlay")
                .selected_text(self.overlay.label())
                .show_ui(ui, |ui| {
                    for o in Overlay::ALL {
                        ui.selectable_value(&mut self.overlay, o, o.label());
                    }
                });
            ui.label("不透明度");
            ui.add(egui::Slider::new(&mut self.alpha, 0.0..=1.0));
            ui.checkbox(&mut self.auto_scale, "色範囲を自動");
            ui.add_enabled_ui(!self.auto_scale && self.overlay != Overlay::Conf, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.lo)
                        .speed(0.001)
                        .prefix("最小 "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.hi)
                        .speed(0.001)
                        .prefix("最大 "),
                );
            });
        });
        if self.view_busy() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("表示を更新中…");
            });
        }
        // colour bar
        let (rect, _) = ui.allocate_exact_size(egui::vec2(260.0, 14.0), egui::Sense::hover());
        let scale = if self.overlay.symmetric() {
            Scale::Diverging
        } else {
            Scale::Sequential
        };
        let steps = 64;
        for i in 0..steps {
            let a = i as f32 / steps as f32;
            let c = color(scale, self.lo + (self.hi - self.lo) * a, self.lo, self.hi);
            let r = egui::Rect::from_min_max(
                egui::pos2(rect.min.x + rect.width() * a, rect.min.y),
                egui::pos2(
                    rect.min.x + rect.width() * (a + 1.0 / steps as f32),
                    rect.max.y,
                ),
            );
            ui.painter()
                .rect_filled(r, 0.0, Color32::from_rgb(c[0], c[1], c[2]));
        }
        ui.label(format!("{:.4}  …  {:.4}", self.lo, self.hi));

        let cur_k = st.k as f64;
        let series = &self.series;
        ui.columns(2, |cols| {
            Plot::new("valid")
                .height(150.0)
                .include_y(0.0)
                .include_y(100.0)
                .x_axis_label("結果番号")
                .y_axis_label("有効率 [%]")
                .show(&mut cols[0], |p| {
                    p.line(Line::new(
                        "有効率",
                        PlotPoints::from_iter(
                            series.iter().map(|s| [s.k as f64, s.valid_frac * 100.0]),
                        ),
                    ));
                    p.vline(VLine::new("現在", cur_k));
                });
            Plot::new("gmax")
                .height(150.0)
                .include_y(0.0)
                .x_axis_label("結果番号")
                .y_axis_label("gmax")
                .show(&mut cols[1], |p| {
                    p.line(Line::new(
                        "p50",
                        PlotPoints::from_iter(series.iter().map(|s| [s.k as f64, s.g50])),
                    ));
                    p.line(Line::new(
                        "p95",
                        PlotPoints::from_iter(series.iter().map(|s| [s.k as f64, s.g95])),
                    ));
                    p.line(Line::new(
                        "p99",
                        PlotPoints::from_iter(series.iter().map(|s| [s.k as f64, s.g99])),
                    ));
                    p.vline(VLine::new("現在", cur_k));
                });
        });
        let _ = ctx;
    }

    /// One step of the scripted test (see [`Auto`]).
    fn auto_step(&mut self, ctx: &egui::Context) {
        let Some(mut a) = self.auto.take() else {
            return;
        };
        // nothing else keeps a window that nobody touches ticking
        if a.stage != 2 {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
        if a.wait > 0 {
            a.wait -= 1;
            self.auto = Some(a);
            return;
        }
        if a.stage >= 3 && self.view_busy() {
            self.auto = Some(a);
            return;
        }
        let shoot = |a: &mut Auto, name: &str| {
            let path = a.out.join(format!("{name}.png"));
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                egui::ScreenshotCallback::new(move |img| {
                    let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                    if let Some(im) =
                        image::RgbaImage::from_raw(img.width() as u32, img.height() as u32, rgba)
                    {
                        let _ = im.save(&path);
                        eprintln!("selftest: saved {}", path.display());
                    }
                }),
            ));
            a.wait = 8;
        };
        match a.stage {
            0 => {
                let r = Source::open(&a.source).map(|s| (s, None));
                self.set_source(ctx, r);
                self.roi = a.roi.clone();
                shoot(&mut a, "01_roi");
                a.stage = 1;
            }
            1 => {
                if a.standard {
                    self.settings = Settings::preset(Preset::Standard);
                }
                self.settings.crack = a.crack;
                self.base_out = a.out.join("optflow_出力");
                self.start_run(ctx);
                a.stage = 2;
            }
            2 => {
                if self.running.is_none() {
                    eprintln!(
                        "selftest: run ended: error={:?} frames={} at {:.1} s",
                        self.error,
                        self.series.len(),
                        a.t0.elapsed().as_secs_f64()
                    );
                    a.stage = 3;
                } else if !a.shot_running {
                    // otherwise as a user's window: only the repaints `logic` asks for in a run
                    a.shot_running = true;
                    shoot(&mut a, "02_running");
                }
            }
            3 => {
                self.cur = self.series.len().saturating_sub(1);
                self.overlay = Overlay::Gmax;
                a.wait = 6;
                a.stage = 4;
            }
            4 => {
                eprintln!(
                    "selftest: first result view at {:.1} s",
                    a.t0.elapsed().as_secs_f64()
                );
                shoot(&mut a, "03_gmax");
                a.stage = 5;
            }
            5 => {
                self.overlay = Overlay::Ux;
                a.wait = 6;
                a.stage = 6;
            }
            6 => {
                shoot(&mut a, "04_ux");
                a.stage = 7;
            }
            7 => {
                self.overlay = Overlay::Conf;
                a.wait = 6;
                a.stage = 8;
            }
            8 => {
                shoot(&mut a, "05_conf");
                a.stage = 9;
            }
            9 if a.crack && self.overlay != Overlay::Cod => {
                self.overlay = Overlay::Cod;
                a.wait = 6;
            }
            9 if a.crack && !a.shot_cod => {
                a.shot_cod = true;
                shoot(&mut a, "06_cod");
            }
            9 => {
                let dest = a.out.join("export");
                self.start_csv_export(ctx, dest);
                // PNG of the current view, without the dialog
                if let (Some(f), Some(bg)) = (&self.fields, &self.first) {
                    let _ = std::fs::create_dir_all(a.out.join("export"));
                    let _ = write_png(
                        &a.out.join("export").join("view.png"),
                        f,
                        bg,
                        self.overlay,
                        self.lo,
                        self.hi,
                        self.alpha,
                        &self.roi,
                        self.crack.as_ref(),
                    );
                }
                eprintln!(
                    "selftest: export started at {:.1} s",
                    a.t0.elapsed().as_secs_f64()
                );
                a.stage = 10;
            }
            // until the CSV thread reports back
            10 => {
                if self.export_rx.is_none() {
                    a.wait = 5;
                    a.stage = 11;
                }
            }
            _ => {
                eprintln!(
                    "selftest: done at {:.1} s; {:?}; run folder {:?}",
                    a.t0.elapsed().as_secs_f64(),
                    self.notice,
                    self.run_dir
                );
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        self.auto = Some(a);
    }
}

/// The events of a drag: one group per frame, so the interface sees it as a person's.
fn drag_events(
    from: egui::Pos2,
    to: egui::Pos2,
    button: egui::PointerButton,
) -> Vec<Vec<egui::Event>> {
    let press = |pressed, pos| egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let mut v = vec![
        vec![egui::Event::PointerMoved(from)],
        vec![press(true, from)],
    ];
    for i in 1..=12 {
        let p = from + (to - from) * (i as f32 / 12.0);
        v.push(vec![egui::Event::PointerMoved(p)]);
    }
    v.push(vec![press(false, to)]);
    v.push(vec![]);
    v
}

fn click_events(
    pos: egui::Pos2,
    button: egui::PointerButton,
    times: usize,
) -> Vec<Vec<egui::Event>> {
    let press = |pressed| egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let mut v = vec![vec![egui::Event::PointerMoved(pos)], vec![]];
    for _ in 0..times {
        v.push(vec![press(true)]);
        v.push(vec![press(false)]);
    }
    v.push(vec![]);
    v
}

impl App {
    fn screenshot(ctx: &egui::Context, out: &Path, name: &str) {
        let path = out.join(format!("{name}.png"));
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
            egui::ScreenshotCallback::new(move |img| {
                let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                if let Some(im) =
                    image::RgbaImage::from_raw(img.width() as u32, img.height() as u32, rgba)
                {
                    let _ = im.save(&path);
                    eprintln!("script: saved {}", path.display());
                }
            }),
        ));
    }

    /// One step of the scripted drive of the window (see [`Script`]). Every action on the ROI and
    /// every button press is a pointer event through egui.
    fn script_step(&mut self, ctx: &egui::Context) {
        let Some(mut s) = self.script.take() else {
            return;
        };
        ctx.request_repaint_after(std::time::Duration::from_millis(30));
        let t0 = *s.t0.get_or_insert_with(Instant::now);
        if s.wait > 0 || !s.events.is_empty() {
            s.wait = s.wait.saturating_sub(1);
            self.script = Some(s);
            return;
        }
        let at = |app: &App, x: f64, y: f64| {
            let (rect, zoom) = app.image_area.expect("the picture is on the screen");
            rect.min + egui::vec2(x as f32 * zoom, y as f32 * zoom)
        };
        let primary = egui::PointerButton::Primary;
        let log = |app: &App, what: &str| {
            eprintln!(
                "script: {what}: roi {:?} frames {} at {:.1} s",
                app.roi,
                app.series.len(),
                t0.elapsed().as_secs_f64()
            );
        };
        match s.stage {
            0 if s.reopen => {
                self.base_out = s.out.join("optflow_出力");
                self.out_picked = true;
                let r = Source::open(&s.source).map(|src| (src, None));
                self.set_source(ctx, r);
                log(self, "opened again");
                eprintln!("script: notice {:?}", self.notice);
                assert!(self.offer.is_some(), "the stopped run is offered");
                assert!(self.roi.len() >= 3, "its ROI came back");
                s.wait = 10;
                s.stage = 18;
            }
            0 => {
                self.base_out = s.out.join("optflow_出力");
                self.out_picked = true;
                let r = Source::open(&s.source).map(|src| (src, None));
                self.set_source(ctx, r);
                self.settings = optflow_app::settings::Settings::preset(
                    optflow_app::settings::Preset::Recommended,
                );
                s.wait = 10;
                s.stage = 1;
            }
            1 => {
                Self::screenshot(ctx, &s.out, "01_opened");
                s.wait = 8;
                s.stage = 2;
            }
            2 => {
                let first = self.first.clone().expect("a source is open");
                let (w, h) = (first.w as f64, first.h as f64);
                s.events.extend(drag_events(
                    at(self, 0.2 * w, 0.2 * h),
                    at(self, 0.8 * w, 0.8 * h),
                    primary,
                ));
                s.stage = 3;
                s.wait = 3;
            }
            3 => {
                log(self, "after the drag on the empty picture");
                assert_eq!(self.roi.len(), 4, "the drag made a rectangle");
                Self::screenshot(ctx, &s.out, "02_rect");
                s.wait = 8;
                s.stage = 4;
            }
            4 => {
                let c = self.roi[2];
                let first = self.first.clone().expect("a source is open");
                let (dx, dy) = (0.08 * first.w as f64, 0.06 * first.h as f64);
                s.events.extend(drag_events(
                    at(self, c.0, c.1),
                    at(self, c.0 + dx, c.1 + dy),
                    primary,
                ));
                s.stage = 5;
                s.wait = 3;
            }
            5 => {
                log(self, "after the corner drag");
                Self::screenshot(ctx, &s.out, "03_corner");
                s.wait = 8;
                s.stage = 6;
            }
            6 => {
                let m = roi_edit::midpoint(&self.roi, 0);
                let first = self.first.clone().expect("a source is open");
                s.events.extend(drag_events(
                    at(self, m.0, m.1),
                    at(self, m.0 + 40.0, m.1 - 0.1 * first.h as f64),
                    primary,
                ));
                s.stage = 7;
                s.wait = 3;
            }
            7 => {
                log(self, "after the edge drag");
                assert_eq!(self.roi[0].1, self.roi[1].1, "the top edge stayed level");
                Self::screenshot(ctx, &s.out, "04_edge");
                s.wait = 8;
                s.stage = 8;
            }
            8 => {
                // a double click at a quarter of the left edge inserts a vertex
                let (a, b) = (self.roi[3], self.roi[0]);
                let p = (a.0 + 0.25 * (b.0 - a.0), a.1 + 0.25 * (b.1 - a.1));
                s.events
                    .extend(click_events(at(self, p.0, p.1), primary, 2));
                s.stage = 9;
                s.wait = 3;
            }
            9 => {
                log(self, "after the double click on an edge");
                assert_eq!(self.roi.len(), 5, "the double click inserted a vertex");
                Self::screenshot(ctx, &s.out, "05_inserted");
                s.wait = 8;
                s.stage = 10;
            }
            10 => {
                let p = self.roi[4];
                s.events.extend(click_events(
                    at(self, p.0, p.1),
                    egui::PointerButton::Secondary,
                    1,
                ));
                s.stage = 11;
                s.wait = 3;
            }
            11 => {
                log(self, "after the right click on a vertex");
                assert_eq!(self.roi.len(), 4, "the right click removed it");
                // a press inside, away from any handle, moves nothing
                let before = self.roi.clone();
                let c = (
                    (before[0].0 + before[2].0) / 2.0,
                    (before[0].1 + before[2].1) / 2.0,
                );
                s.events.extend(drag_events(
                    at(self, c.0, c.1),
                    at(self, c.0 + 30.0, c.1 + 30.0),
                    primary,
                ));
                s.stage = 12;
                s.wait = 3;
            }
            12 => {
                log(self, "after a drag inside");
                let start = self.btn_start.expect("the start button is on the screen");
                s.events.extend(click_events(start.center(), primary, 1));
                s.stage = 13;
                s.wait = 3;
            }
            13 => {
                if self.running.is_some()
                    && !self.series.is_empty()
                    && !self.view_busy()
                    && self.tex.is_some()
                {
                    log(self, "the live map is up");
                    Self::screenshot(ctx, &s.out, "06_live_gmax");
                    s.wait = 8;
                    s.stage = 14;
                }
            }
            14 => {
                self.overlay = Overlay::Ux;
                s.wait = 10;
                s.stage = 15;
            }
            15 => {
                if !self.view_busy() {
                    Self::screenshot(ctx, &s.out, "07_live_ux");
                    s.wait = 8;
                    s.stage = 16;
                }
            }
            16 => {
                if self.series.len() >= s.stop_at {
                    let stop = self.btn_stop.expect("the stop button is on the screen");
                    log(self, "pressing stop");
                    s.events.extend(click_events(stop.center(), primary, 1));
                    s.stage = 17;
                    s.wait = 3;
                }
            }
            17 => {
                if self.running.is_none() {
                    log(self, "stopped");
                    s.wait = 10;
                    s.stage = if s.quit_after_stop { 23 } else { 18 };
                }
            }
            18 => {
                if !self.view_busy() {
                    eprintln!(
                        "script: notice {:?}; last save {:?}",
                        self.notice,
                        self.last_save.as_ref().map(|l| l.1)
                    );
                    let name = if s.reopen {
                        "10_reopened"
                    } else {
                        "08_stopped"
                    };
                    Self::screenshot(ctx, &s.out, name);
                    s.wait = 8;
                    s.stage = 19;
                }
            }
            // opened again: first what is refused, what asks, and the way back
            19 if s.reopen && !s.checked_refusal => {
                let c = self.roi[2];
                s.events.extend(drag_events(
                    at(self, c.0, c.1),
                    at(self, c.0 - 90.0, c.1),
                    primary,
                ));
                s.checked_refusal = true;
                s.stage = 100;
                s.wait = 3;
            }
            100 => {
                let diff = self.offer_mismatches();
                eprintln!("script: the ROI moved, the offer says {diff:?}");
                assert_eq!(diff, vec![Mismatch::Roi], "a moved ROI is refused");
                Self::screenshot(ctx, &s.out, "12_refused");
                s.wait = 8;
                s.stage = 101;
            }
            101 => {
                let b = self.btn_start.expect("the start button is on the screen");
                s.events.extend(click_events(b.center(), primary, 1));
                s.stage = 102;
                s.wait = 6;
            }
            102 => {
                assert!(self.ask == Ask::FreshRun, "a fresh run asks first");
                assert!(self.running.is_none());
                Self::screenshot(ctx, &s.out, "13_asks_first");
                s.wait = 8;
                s.stage = 103;
            }
            103 => {
                let b = self.btn_ask_cancel.expect("the question is on the screen");
                s.events.extend(click_events(b.center(), primary, 1));
                s.stage = 104;
                s.wait = 6;
            }
            104 => {
                assert!(self.ask == Ask::None && self.running.is_none());
                let b = self.btn_restore.expect("the way back is on the screen");
                s.events.extend(click_events(b.center(), primary, 1));
                s.stage = 105;
                s.wait = 6;
            }
            105 => {
                assert!(self.offer_mismatches().is_empty(), "the values came back");
                log(self, "back to the values of the stopped run");
                s.wait = 6;
                s.stage = 19;
            }
            19 => {
                assert!(self.offer.is_some(), "stopped: the run is offered");
                let b = self.btn_resume.expect("the resume button is on the screen");
                s.events.extend(click_events(b.center(), primary, 1));
                s.stage = 20;
                s.wait = 3;
            }
            20 => {
                if self.running.is_some() {
                    log(self, "resumed");
                    s.stage = 21;
                }
            }
            21 => {
                if self.running.is_none() {
                    log(self, "finished");
                    eprintln!("script: error {:?}; notice {:?}", self.error, self.notice);
                    s.wait = 10;
                    s.stage = 22;
                }
            }
            22 => {
                if !self.view_busy() {
                    let name = if s.reopen {
                        "11_reopened_finished"
                    } else {
                        "09_finished"
                    };
                    Self::screenshot(ctx, &s.out, name);
                    s.wait = 10;
                    s.stage = 23;
                }
            }
            _ => {
                eprintln!(
                    "script: done at {:.1} s; run folder {:?}",
                    t0.elapsed().as_secs_f64(),
                    self.run_dir
                );
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        self.script = Some(s);
    }
}

/// The full-size PNG of a view (the export; never reduced).
#[allow(clippy::too_many_arguments)]
fn write_png(
    path: &Path,
    f: &Fields,
    bg: &Gray8,
    overlay: Overlay,
    lo: f32,
    hi: f32,
    alpha: f32,
    roi: &[(f64, f64)],
    crack: Option<&optflow_app::crack::CrackLines>,
) -> anyhow::Result<()> {
    if (f.w, f.h) != (bg.w, bg.h) {
        anyhow::bail!("the maps are not the size of the first frame");
    }
    let o = export::overlay_rgba(f, overlay, lo, hi, alpha);
    let mut rgb = export::compose_rgb(bg, Some(&o), roi);
    draw_crack(&mut rgb, f.w, f.h, crack);
    export::save_png(path, f.w, f.h, rgb)
}

/// The longest side a texture of this app may have: its own limit or the driver's.
fn max_texture_side(ctx: &egui::Context) -> usize {
    ctx.input(|i| i.max_texture_side)
        .min(view::MAX_TEXTURE_SIDE)
}

fn draw_crack(rgb: &mut [u8], w: usize, h: usize, c: Option<&optflow_app::crack::CrackLines>) {
    let Some(c) = c else { return };
    for l in &c.bands {
        export::draw_polyline(rgb, w, h, l, [255, 0, 255]);
    }
    for l in &c.lines {
        export::draw_polyline(rgb, w, h, l, [0, 255, 255]);
    }
}

fn fmt_secs(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}時間{:02}分{:02}秒", s / 3600, s / 60 % 60, s % 60)
    } else if s >= 60 {
        format!("{}分{:02}秒", s / 60, s % 60)
    } else {
        format!("{s}秒")
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if let Some(rx) = &self.export_rx {
            if let Ok(r) = rx.try_recv() {
                self.notice = Some(match r {
                    Ok(t) | Err(t) => t,
                });
                self.export_rx = None;
            }
        }
        if self.running.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        if self.auto.is_some() {
            self.auto_step(ctx);
        }
        self.script_step(ctx);
        // while a run goes on the picture is made on the view thread and the newest request wins, so
        // frames are skipped when it lags; the tracker never waits for it
        if self.running.is_none() || self.live {
            self.cur = self.cur.min(self.series.len().saturating_sub(1));
            self.refresh_view(ctx);
        }
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        if let Some(s) = &mut self.script {
            if let Some(ev) = s.events.pop_front() {
                raw.events.extend(ev);
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        egui::Panel::left("controls")
            .default_size(350.0)
            .size_range(300.0..=420.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, &ctx));
            });
        egui::Panel::bottom("results")
            .resizable(true)
            .min_size(120.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.results_panel(ui, &ctx));
            });
        egui::CentralPanel::default().show(ui, |ui| self.image_panel(ui));
        self.ask_window(&ctx);
    }
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [flag, out] = args.as_slice() {
        if flag == "--headless-smoke" {
            // no window: the CI check of a build that cannot show one
            let code = match optflow_app::smoke::run_smoke(std::path::Path::new(out)) {
                Ok(_) => 0,
                Err(e) => {
                    let _ = std::fs::create_dir_all(out);
                    let _ = std::fs::write(
                        std::path::Path::new(out).join("smoke_error.txt"),
                        format!("{e:#}"),
                    );
                    1
                }
            };
            std::process::exit(code);
        }
    }
    let auto = match args.as_slice() {
        [flag, src, roi, out, ..] if flag == "--selftest" => {
            let crack = args.get(4).is_some_and(|a| a == "crack");
            let standard = args.get(5).is_some_and(|a| a == "standard");
            let roi = roi
                .split(';')
                .filter_map(|p| {
                    let (x, y) = p.split_once(',')?;
                    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
                })
                .collect();
            let out = PathBuf::from(out);
            let _ = std::fs::create_dir_all(&out);
            Some(Auto {
                source: PathBuf::from(src),
                roi,
                out,
                stage: 0,
                wait: 5,
                t0: Instant::now(),
                crack,
                standard,
                shot_cod: false,
                shot_running: false,
            })
        }
        _ => None,
    };
    let script = match args.as_slice() {
        [flag, src, out, stop_at, rest @ ..] if flag == "--selftest-interactive" => {
            let out = PathBuf::from(out);
            let _ = std::fs::create_dir_all(&out);
            Some(Script {
                source: PathBuf::from(src),
                out,
                wait: 5,
                stop_at: stop_at.parse().unwrap_or(2),
                quit_after_stop: rest.iter().any(|a| a == "quit"),
                reopen: rest.iter().any(|a| a == "reopen"),
                ..Script::default()
            })
        }
        _ => None,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 840.0])
            .with_title("optflow 土のひずみ追跡"),
        ..Default::default()
    };
    let result = eframe::run_native(
        "optflow-app",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, auto, script)))),
    );
    if let Err(e) = &result {
        // a window that cannot be opened (typically no OpenGL 2.0 driver) must not vanish silently
        let text = format!(
            "ウィンドウを開けませんでした。グラフィックスドライバーが OpenGL 2.0 以上に対応している必要があります。\n({e})"
        );
        let _ = std::fs::write(std::env::temp_dir().join("optflow-app-error.txt"), &text);
        if std::env::var_os("OPTFLOW_APP_NO_DIALOG").is_none() {
            rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Error)
                .set_title("optflow-app")
                .set_description(&text)
                .show();
        }
    }
    result
}

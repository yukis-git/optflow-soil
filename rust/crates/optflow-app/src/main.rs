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
use optflow_app::settings::{Preset, Settings};
use optflow_app::source::{Gray8, Source};
use optflow_app::stats::{FrameStats, Overlay};
use optflow_app::view::{self, ViewKey, ViewWorker};
use optflow_app::{fonts, messages, power};

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
    fn new(cc: &eframe::CreationContext<'_>, auto: Option<Auto>) -> App {
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
                self.close_results();
                match s.frame(self.start) {
                    Ok(g) => {
                        // the picture for ROI drawing; ROI coordinates stay in full-size pixels
                        let factor = view::display_factor(g.w, g.h, max_texture_side(ctx));
                        let (w, h, data) = view::shrink_gray(&g, factor);
                        let img = egui::ColorImage::from_gray([w, h], &data);
                        self.first_tex =
                            Some(ctx.load_texture("first", img, egui::TextureOptions::LINEAR));
                        self.first = Some(Arc::new(g));
                        self.source = Some(s);
                    }
                    Err(e) => {
                        self.error = Some(messages::describe(&e));
                    }
                }
            }
            Err(e) => self.error = Some(messages::describe(&e)),
        }
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
    }

    fn spec(&self) -> Option<JobSpec> {
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
            out: job::next_run_dir(&self.base_out),
        })
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        let Some(spec) = self.spec() else { return };
        self.close_results();
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
        let Some(rx) = &self.rx else { return };
        let mut done = false;
        while let Ok(m) = rx.try_recv() {
            match m {
                Msg::Started { total } => self.total = total,
                Msg::Frame { stats } => {
                    self.cur = self.series.len();
                    self.series.push(stats);
                }
                Msg::Warn(w) => self.warnings.push(w),
                Msg::Finished => {
                    self.notice = Some("完了しました。".into());
                    done = true;
                }
                Msg::Cancelled => {
                    self.notice =
                        Some("中止しました(ここまでの結果は表示・書き出しできます)。".into());
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
            self.rx = None;
            if let Some(r) = self.running.take() {
                let _ = r.handle.join();
            }
            self.cur = self.series.len().saturating_sub(1);
        }
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
            ui.label("画像上をクリックして ROI の頂点を追加します。");
            ui.horizontal(|ui| {
                if ui.button("1点戻す").clicked() {
                    self.roi.pop();
                }
                if ui.button("ROIをクリア").clicked() {
                    self.roi.clear();
                }
                ui.label(format!("頂点 {}", self.roi.len()));
            });
            if self.roi.len() < 3 {
                ui.label("ROI が 3 点未満の間は画像全体を追跡します。");
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
                });
                ui.label("縮小率 2 以上では変位は縮小後の画素単位になります。");
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
                }
            }
        });
        if let Some(text) = self.output_estimate() {
            ui.add(egui::Label::new(text).wrap());
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !busy && self.source.is_some(),
                    egui::Button::new(RichText::new("▶ 解析開始").strong()),
                )
                .clicked()
            {
                self.start_run(ctx);
            }
            if ui.add_enabled(busy, egui::Button::new("■ 中止")).clicked() {
                if let Some(r) = &self.running {
                    r.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
        if busy {
            let done = self.series.len();
            let el = self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
            match self.total {
                Some(n) if n > 0 => {
                    ui.add(
                        egui::ProgressBar::new((done as f32 / n as f32).min(1.0))
                            .text(format!("{done} / {n} フレーム")),
                    );
                    let eta = if done > 0 {
                        el / done as f64 * (n - done.min(n)) as f64
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
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
        let painter = ui.painter_at(rect);
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
        if editing && resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let v = (pos - rect.min) / s;
                self.roi.push((
                    v.x.clamp(0.0, iw - 1.0) as f64,
                    v.y.clamp(0.0, ih - 1.0) as f64,
                ));
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
        for p in &pts {
            painter.circle_filled(*p, 3.5, Color32::from_rgb(255, 80, 0));
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
            ui.label("クリックで ROI の頂点を追加");
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
        if self.running.is_none() {
            self.cur = self.cur.min(self.series.len().saturating_sub(1));
            self.refresh_view(ctx);
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
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 840.0])
            .with_title("optflow 土のひずみ追跡"),
        ..Default::default()
    };
    let result = eframe::run_native(
        "optflow-app",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, auto)))),
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

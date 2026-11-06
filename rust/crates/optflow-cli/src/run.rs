//! The `pair` and `seq` commands (`_run` of `python/optflow_ref/__main__.py`).

use std::borrow::Cow;
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::rc::Rc;

use anyhow::{anyhow, bail, Context, Result};
use optflow_core::engine::{CpuEngine, FineEngine};
use optflow_core::image::Image;
use optflow_core::poc::capture_range;
use optflow_core::track::{
    track_pair_owned_with, track_sequence_with, StrainOpts, Timing, TrackError, TrackResult,
};

use crate::args::{Cli, Command, CropMode, Device};
use crate::crop::{self, Crop};
use crate::io;
use crate::mask;
use crate::polygon::polygon_mask;
use crate::say;
use crate::settings;
use crate::tavg;
use crate::video;

/// One decoded and downscaled frame, or why there is none.
type Frame = Result<Image<f32>>;

fn load(path: &Path, scale: i64) -> Result<Image<f32>> {
    Ok(io::downscale(io::read_image(path)?, scale)?.into_f32())
}

fn roi_mask(roi: Option<&Path>, w: usize, h: usize, scale: i64) -> Result<Option<Image<bool>>> {
    let Some(path) = roi else {
        return Ok(None);
    };
    let points = io::load_roi(path)?;
    // scale >= 1 here: downscale has already rejected the rest
    Ok(Some(polygon_mask(w, h, &points, scale as usize)))
}

/// The ROI mask (`--roi` polygon or `--mask` file) of a `w` x `h` image, if there is one.
fn region(cli: &Cli, w: usize, h: usize) -> Result<Option<Image<bool>>> {
    match &cli.mask {
        Some(path) => Ok(Some(mask::load_mask(path, w, h)?)),
        None => roi_mask(cli.roi.as_deref(), w, h, cli.scale),
    }
}

/// A map of the result of tracking `crop` as one of the full `w` x `h` image (`Crop::place`).
fn placed<'a>(
    crop: Option<&Crop>,
    (w, h): (usize, usize),
    im: &'a Image<f32>,
    fill: f32,
) -> Cow<'a, Image<f32>> {
    match crop {
        Some(c) => c.place(im, fill, w, h),
        None => Cow::Borrowed(im),
    }
}

/// `io::save_result`, and with `--raw` the unmasked displacement as `raw_ux.npy`, `raw_uy.npy`;
/// `res` is the result of tracking `crop` of the `size` image and is written as full-size maps
/// (NaN outside the crop, confidence 0).
fn save(
    cli: &Cli,
    dir: &Path,
    res: &TrackResult,
    crop: Option<&Crop>,
    size: (usize, usize),
) -> Result<()> {
    io::save_result_placed(dir, res, cli.device == Device::Gpu, &|im, fill| {
        placed(crop, size, im, fill)
    })?;
    if let Some(fb) = &res.fb {
        // hybrid sequences: the frames each pixel was carried by the adjacent leg (0 = direct)
        let as_f32 = Image::from_vec(fb.w, fb.h, fb.data.iter().map(|&v| v as f32).collect());
        io::write_npy_i32(&dir.join("fb.npy"), &placed(crop, size, &as_f32, 0.0))?;
    }
    if let Some(crack) = &res.crack {
        let off = crop.map_or((0.0, 0.0), |c| (c.x0 as f64, c.y0 as f64));
        io::save_crack(dir, crack, &|im, fill| placed(crop, size, im, fill), off)?;
    }
    if cli.raw {
        for (name, im) in [("raw_ux", &res.raw_ux), ("raw_uy", &res.raw_uy)] {
            let path = dir.join(format!("{name}.npy"));
            io::write_npy_f32(&path, &placed(crop, size, im, f32::NAN))?;
        }
    }
    Ok(())
}

/// The engine of the fine stage for `--device`; a missing GPU is an error, never a fallback.
fn engine(d: Device) -> Result<Box<dyn FineEngine>> {
    match d {
        Device::Cpu => Ok(Box::new(CpuEngine)),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        Device::Gpu => Ok(Box::new(optflow_gpu::GpuEngine::new()?)),
        #[cfg(not(all(feature = "gpu", target_os = "macos")))]
        Device::Gpu => bail!("--device gpu needs a macOS build with --features gpu"),
    }
}

/// Python's `repr` of the timing dict, as `_run` prints it.
fn timing_text(t: &Timing, d: Device) -> String {
    let mut text = format!("{{'coarse': {:?}", t.coarse);
    if d == Device::Gpu {
        text.push_str(", 'device': 'gpu'");
    }
    text.push_str(&format!(", 'fine': {:?}, 'strain': {:?}", t.fine, t.strain));
    if let Some(r) = t.reference {
        text.push_str(&format!(", 'reference': {r}"));
    }
    if let Some(c) = t.mean_conf {
        text.push_str(&format!(", 'mean_conf': {c:?}"));
    }
    text.push('}');
    text
}

/// The crop of this run (`None`: track the whole image); writes `crop.json`.
fn plan_crop(
    cli: &Cli,
    cfg: &optflow_core::config::Config,
    mask: Option<&Image<bool>>,
    seq: bool,
) -> Result<Option<Crop>> {
    let crop = match (cli.crop, mask) {
        (CropMode::Auto, Some(m)) => Some(crop::plan(m, cfg, cli.crop_margin, seq)?),
        _ => None,
    };
    let path = cli.out.join("crop.json");
    let text = crop
        .as_ref()
        .map_or_else(|| crop::OFF_JSON.to_string(), Crop::json);
    fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(crop)
}

/// `im` cut to the crop, if there is one (the full image is dropped); `im` itself when the crop
/// is the whole image.
fn cut(crop: Option<&Crop>, im: Image<f32>) -> Image<f32> {
    match crop {
        Some(c) if !c.is_whole(im.w, im.h) => c.apply(&im),
        _ => im,
    }
}

/// The ROI of what is tracked: `mask` cut to the crop (the full mask is dropped).
fn cut_mask(crop: Option<&Crop>, mask: Option<Image<bool>>) -> Option<Image<bool>> {
    match (crop, mask) {
        (Some(c), Some(m)) if !c.is_whole(m.w, m.h) => Some(c.apply(&m)),
        (_, m) => m,
    }
}

/// Python's `format(x, "g")`: six significant digits, trailing zeros dropped, the exponent form
/// (`1e+06`, `2.5e-07`) below 1e-4 and from 1e6 on.
fn fmt_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0");
    }
    // `{:.5e}` rounds the exact value to six digits (ties to even), as Python does
    let sci = format!("{:.5e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let trim = |t: &str| {
        if t.contains('.') {
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            t.to_string()
        }
    };
    if (-4..6).contains(&exp) {
        format!(
            "{sign}{}",
            trim(&format!("{:.*}", (5 - exp) as usize, x.abs()))
        )
    } else {
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}e{esign}{:02}", trim(mantissa), exp.abs())
    }
}

/// The strain maps the output files hold: no small-strain maps (`exx`, `eyy`, `exy`) are written,
/// so they are not computed.
const WRITTEN_STRAIN: StrainOpts = StrainOpts { small: false };

/// The warning line when the coarse stage of the `h` x `w` image actually tracked (after `--scale`
/// and the crop) is designed to capture less than `max_disp` along an axis, else `None`. Printed
/// once per run after the first successful tracking, so a run that fails prints only its error
/// (`_capture_warning` of `python/optflow_ref/__main__.py`).
fn capture_warning(cfg: &optflow_core::config::Config, h: usize, w: usize) -> Option<String> {
    let (cy, cx) = capture_range(cfg.max_disp, cfg.poc_win, h, w);
    (cy.min(cx) < cfg.max_disp).then(|| {
        format!(
            "warning: the coarse stage covers about \u{b1}{} px in x and \u{b1}{} px in y, \
             less than max_disp {}; larger motions may be lost \
             (use a larger image or crop margin, or a smaller max_disp)",
            fmt_g(cx),
            fmt_g(cy),
            fmt_g(cfg.max_disp)
        )
    })
}

/// What a caller other than the command line (the desktop app) hooks into a run: where the
/// messages go, a look at every saved frame, and a request to stop. The defaults are the command
/// line's behaviour ([`CliHooks`]).
pub trait Hooks {
    /// A line of normal output (`frame 3: ...`, `done: ...`).
    fn say(&mut self, text: &str) -> Result<()> {
        say(text)
    }
    /// A warning line.
    fn warn(&mut self, text: &str) {
        eprintln!("{text}");
    }
    /// Frame `k` of a sequence has been tracked and written.
    fn frame_done(&mut self, _k: usize, _res: &TrackResult) {}
    /// Whether the run should stop before the next frame ([`Cancelled`] is its error).
    fn cancelled(&self) -> bool {
        false
    }
}

/// The hooks of the command line: stdout, stderr, never cancelled.
pub struct CliHooks;

impl Hooks for CliHooks {}

/// The run was stopped by [`Hooks::cancelled`].
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Runs the command line `cli`.
pub fn run(cli: &Cli) -> Result<()> {
    run_with(cli, &mut CliHooks)
}

/// [`run`] with `hooks` in place of stdout, stderr and the stop condition of the command line.
pub fn run_with(cli: &Cli, hooks: &mut dyn Hooks) -> Result<()> {
    let cfg = settings::config(cli.config.as_deref(), &cli.set)?;
    if cli.roi.is_some() && cli.mask.is_some() {
        bail!("--roi and --mask are mutually exclusive");
    }
    crop::check_margin(cli.crop_margin)?;
    if cfg.is_xdic() && cli.device == Device::Gpu {
        // the per-side solve needs the slope of the baseline solve, which the GPU engine does not
        // return: refuse before reading any image (the core says the same, per engine)
        bail!(
            "crack_model 'xdic' runs on the CPU only (the per-side solve needs the slope of the \
             baseline solve, which the GPU engine does not return); use --device cpu"
        );
    }
    let engine = engine(cli.device)?;
    fs::create_dir_all(&cli.out).with_context(|| format!("cannot create {}", cli.out.display()))?;
    let config_path = cli.out.join("config.json");
    fs::write(&config_path, settings::config_json(&cfg))
        .with_context(|| format!("cannot write {}", config_path.display()))?;
    if cli.png {
        hooks.warn(
            "warning: --png is accepted for compatibility but not supported: no maps are written",
        );
    }
    match &cli.command {
        Command::Pair {
            reference,
            deformed,
        } => {
            let i1 = load(reference, cli.scale)?;
            let i2 = load(deformed, cli.scale)?;
            let mask = region(cli, i1.w, i1.h)?;
            let (w, h) = (i1.w, i1.h);
            // images of different sizes are track_pair's error to report
            let crop = if (i2.w, i2.h) == (w, h) {
                plan_crop(cli, &cfg, mask.as_ref(), false)?
            } else {
                None
            };
            let (i1, i2) = (cut(crop.as_ref(), i1), cut(crop.as_ref(), i2));
            let mask = cut_mask(crop.as_ref(), mask);
            let note = capture_warning(&cfg, i1.h, i1.w);
            // the images are released once the fine stage is done, the ROI after the border check
            let res =
                track_pair_owned_with(&*engine, i1, i2, &cfg, mask.as_ref(), None, WRITTEN_STRAIN)?;
            if let Some(note) = &note {
                hooks.warn(note);
            }
            if let (Some(c), Some(m)) = (&crop, &mask) {
                if c.border_fraction(&res.raw_ux, &res.raw_uy, m, (w, h), &cfg)
                    > crop::WARN_FRACTION
                {
                    hooks.warn(crop::WARNING);
                }
            }
            drop(mask);
            save(cli, &cli.out, &res, crop.as_ref(), (w, h))?;
            hooks.say(&format!("done: {}\n", timing_text(&res.timing, cli.device)))
        }
        Command::Seq {
            frames,
            start,
            stop,
            step,
        } => run_seq(cli, &*engine, &cfg, frames, *start, *stop, *step, hooks),
    }
}

/// The frames `start, start + step, ... < stop` of a video, directory or glob pattern, downscaled
/// by `scale`, decoded one at a time.
fn raw_frames(
    source: &str,
    start: i64,
    stop: Option<i64>,
    step: i64,
    scale: i64,
) -> Result<Box<dyn Iterator<Item = Frame>>> {
    io::check_range(start, step)?;
    if io::is_video(source) {
        let gray = video::iter_video(Path::new(source), start, stop, step)?;
        Ok(Box::new(
            gray.map(move |g| Ok(io::downscale(g?, scale)?.into_f32())),
        ))
    } else {
        let paths = io::list_images(source)?;
        let picked = io::slice_indices(paths.len(), start, stop, step);
        Ok(Box::new(
            picked
                .into_iter()
                .map(move |i| load(Path::new(&paths[i]), scale)),
        ))
    }
}

/// The frames of a sequence (`iter_frames` of `python/optflow_ref/io.py`): those of
/// [`raw_frames`], or with `cfg.tavg = m > 0` each of them the mean of the frames `i - m ..= i + m`
/// ([`tavg`]).
fn frames(
    cli: &Cli,
    cfg: &optflow_core::config::Config,
    source: &str,
    start: i64,
    stop: Option<i64>,
    step: i64,
) -> Result<Box<dyn Iterator<Item = Frame>>> {
    let scale = cli.scale;
    if cfg.tavg == 0 {
        return raw_frames(source, start, stop, step, scale);
    }
    io::check_range(start, step)?;
    let m = cfg.tavg;
    let lo = (start - m).max(0);
    let raw = raw_frames(source, lo, stop.map(|s| s + m), 1, scale)?;
    let (mask_file, roi_file) = (cli.mask.clone(), cli.roi.clone());
    let mask: tavg::MaskFn = Box::new(move |w, h| match &mask_file {
        Some(path) => Ok(Some(mask::load_mask(path, w, h)?)),
        None => roi_mask(roi_file.as_deref(), w, h, scale),
    });
    Ok(Box::new(tavg::Averaged::new(
        raw,
        usize::try_from(start).unwrap_or(0),
        stop,
        usize::try_from(step).unwrap_or(1),
        usize::try_from(m).unwrap_or(0),
        cfg.tavg_align.then(|| cfg.clone()),
        mask,
    )))
}

/// Hands the frames that decode to the tracker, cut to the crop; the first failure ends the stream
/// and is kept to be reported afterwards.
struct Frames<I> {
    inner: I,
    failure: Rc<RefCell<Option<anyhow::Error>>>,
    crop: Option<Crop>,
    /// Width and height of the first frame, which every other frame must have.
    size: (usize, usize),
    /// Index of the next frame of the sequence.
    index: usize,
}

impl<I: Iterator<Item = Frame>> Iterator for Frames<I> {
    type Item = Image<f32>;

    fn next(&mut self) -> Option<Image<f32>> {
        if self.failure.borrow().is_some() {
            return None;
        }
        let f = match self.inner.next()? {
            Ok(f) => f,
            Err(e) => {
                *self.failure.borrow_mut() = Some(e);
                return None;
            }
        };
        let k = self.index;
        self.index += 1;
        if self.crop.is_some() && (f.w, f.h) != self.size {
            // what track_sequence reports for it, with the sizes of the whole frames
            *self.failure.borrow_mut() = Some(anyhow!(TrackError::FrameShape {
                k,
                got: (f.h, f.w),
                expected: (self.size.1, self.size.0),
            }));
            return None;
        }
        Some(cut(self.crop.as_ref(), f))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_seq(
    cli: &Cli,
    engine: &dyn FineEngine,
    cfg: &optflow_core::config::Config,
    source: &str,
    start: i64,
    stop: Option<i64>,
    step: i64,
    hooks: &mut dyn Hooks,
) -> Result<()> {
    let mut stream = frames(cli, cfg, source, start, stop, step)?;
    let Some(first) = stream.next() else {
        bail!("no frames to track in {source} (check --start, --stop and --step)");
    };
    let first = first?;
    let Some(second) = stream.next() else {
        bail!("need at least two frames to track, got 1");
    };
    let second = second?;
    let (w, h) = (first.w, first.h);
    let mask = region(cli, w, h)?;
    let crop = plan_crop(cli, cfg, mask.as_ref(), true)?;
    let cmask = cut_mask(crop.as_ref(), mask);
    if crop.is_some() && (second.w, second.h) != (w, h) {
        // what track_sequence reports for it, with the sizes of the whole frames
        bail!(TrackError::FrameShape {
            k: 1,
            got: (second.h, second.w),
            expected: (h, w)
        });
    }
    let failure = Rc::new(RefCell::new(None));
    let first = cut(crop.as_ref(), first);
    let mut note = capture_warning(cfg, first.h, first.w);
    let chain = std::iter::once(first)
        .chain(std::iter::once(cut(crop.as_ref(), second)))
        .chain(Frames {
            inner: stream,
            failure: Rc::clone(&failure),
            crop,
            size: (w, h),
            index: 2,
        });
    let mut warned = false;
    for item in track_sequence_with(engine, chain, cfg, cmask.as_ref(), WRITTEN_STRAIN) {
        let (k, res) = item?;
        if let Some(note) = note.take() {
            hooks.warn(&note);
        }
        if let (Some(c), Some(m)) = (&crop, &cmask) {
            if !warned
                && c.border_fraction(&res.raw_ux, &res.raw_uy, m, (w, h), cfg) > crop::WARN_FRACTION
            {
                hooks.warn(crop::WARNING);
                warned = true;
            }
        }
        let dir = cli.out.join(format!("frame_{k:05}"));
        save(cli, &dir, &res, crop.as_ref(), (w, h))?;
        hooks.say(&format!(
            "frame {k}: reference {} mean conf {:.2}\n",
            res.timing
                .reference
                .map_or_else(|| "None".to_string(), |r| r.to_string()),
            res.timing.mean_conf.unwrap_or(f64::NAN)
        ))?;
        hooks.frame_done(k, &res);
        if hooks.cancelled() {
            return Err(anyhow::Error::new(Cancelled));
        }
    }
    if let Some(e) = failure.borrow_mut().take() {
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use optflow_core::config::Config;

    #[test]
    fn fmt_g_is_pythons_format_g() {
        // the strings are `format(x, "g")` of CPython
        for (x, want) in [
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (64.0, "64"),
            (128.0, "128"),
            (0.5, "0.5"),
            (100.5, "100.5"),
            (123456.7, "123457"),
            (999999.5, "1e+06"),
            (999999.4, "999999"),
            (1e6, "1e+06"),
            (1234567.0, "1.23457e+06"),
            (1e5, "100000"),
            (100000.4, "100000"),
            (99999.95, "99999.9"),
            (1e-4, "0.0001"),
            (1.5e-5, "1.5e-05"),
            (0.00012345678, "0.000123457"),
            (1e16, "1e+16"),
            (4194304.0, "4.1943e+06"),
            (1048576.0, "1.04858e+06"),
            (65536.0, "65536"),
            (33554432.0, "3.35544e+07"),
            (2.5e-7, "2.5e-07"),
            (0.1 + 0.2, "0.3"),
            (5e-324, "4.94066e-324"),
            (f64::MAX, "1.79769e+308"),
            (1e-7, "1e-07"),
            (12345.65, "12345.6"),
            // exact ties (representable in binary) round half to even
            (1234565.0, "1.23456e+06"),
            (1234575.0, "1.23458e+06"),
            (100000.5, "100000"),
            (123456.5, "123456"),
            (2.5, "2.5"),
            (3.0000004, "3"),
            (1e22, "1e+22"),
            (123456789.0, "1.23457e+08"),
            (0.1, "0.1"),
            (7e-5, "7e-05"),
            (-64.0, "-64"),
            (-1e6, "-1e+06"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ] {
            assert_eq!(fmt_g(x), want, "{x:e}");
        }
    }

    #[test]
    fn capture_warning_is_issued_only_below_max_disp() {
        let cfg = |max_disp: f64| Config {
            max_disp,
            ..Config::default()
        };
        // 512 x 512, max_disp 128: L = 3 and a 32 px window cover +-64 px
        assert_eq!(
            capture_warning(&cfg(128.0), 512, 512).as_deref(),
            Some(
                "warning: the coarse stage covers about \u{b1}64 px in x and \u{b1}64 px in y, \
                 less than max_disp 128; larger motions may be lost \
                 (use a larger image or crop margin, or a smaller max_disp)"
            )
        );
        // the defaults reach max_disp exactly: no warning
        assert_eq!(capture_warning(&cfg(32.0), 512, 512), None);
        // a strip: +-256 px in x, +-64 px in y; x first in the text
        let w = capture_warning(&cfg(128.0), 421, 3002).unwrap();
        assert!(
            w.contains("about \u{b1}256 px in x and \u{b1}64 px in y, less than max_disp 128;"),
            "{w}"
        );
        // the smaller axis decides: 256 in x is enough for 100, 64 in y is not
        assert!(capture_warning(&cfg(100.0), 421, 3002).is_some());
        assert_eq!(capture_warning(&cfg(64.0), 421, 3002), None);
    }
}

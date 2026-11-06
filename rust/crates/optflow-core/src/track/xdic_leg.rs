//! A leg (image pair) with `crack_model = "xdic"`: `_track_pair_xdic` of
//! `python/optflow_ref/track.py` (the docstrings there are the specification), the carried crack
//! geometry of the sequence legs (`_hint`, `_same_ref_hint`, `_roi_in`) and the frame-0 outputs
//! of a sequence frame (`_xdic_out`).

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use super::*;
use crate::phaseflow::{refine_raw_prepared, Prepared};
use crate::xdic::detect::detect_lines;
use crate::xdic::geom::near_lines;
use crate::xdic::outputs::{
    advance_lines, finish_core, finish_outputs, is_crack, jump_magnitude, lines_to_frame0,
    model_gain, FinishCore, ModelGain,
};
use crate::xdic::params::DET_JMIN;
use crate::xdic::solve::{refine_xdic_on, SideData};
use crate::xdic::{CrackLines, CrackOut, Flow, Poly};

/// What a crack-model leg carries beyond a plain [`Pair`].
pub(super) struct XLeg {
    /// The crack lines after the solve, in the coordinates of the leg's reference.
    pub lines: Vec<Poly>,
    /// The same lines advanced into the leg's target frame (the carried geometry of the next leg).
    pub lines_next: Vec<Poly>,
    pub bands: Vec<Poly>,
    pub rejected: Vec<Poly>,
    /// The validity and COD / CSD samples of the final solve, and its `f64` displacement (the
    /// outputs of a pair; None without lines).
    pub fin: Option<(FinishCore, Vec<f64>, Vec<f64>)>,
    /// Seconds: the detector, and the solve with the checks (`timing["xdic_detect"]`,
    /// `timing["xdic"]`).
    pub t_detect: f64,
    pub t_xdic: f64,
}

impl XLeg {
    pub(super) fn has_lines(&self) -> bool {
        !self.lines.is_empty()
    }

    /// The crack outputs of a result without a line: only the bands and rejected lines (None
    /// when there are none).
    pub(super) fn no_line_crack(&self) -> Option<Box<CrackOut>> {
        (!self.bands.is_empty() || !self.rejected.is_empty()).then(|| {
            Box::new(CrackOut {
                maps: None,
                lines: CrackLines {
                    lines: Vec::new(),
                    lines_next: Some(Vec::new()),
                    jumps: Vec::new(),
                    bands: self.bands.clone(),
                    rejected: self.rejected.clone(),
                },
            })
        })
    }

    /// `crack_lines` of a pair result.
    fn crack_lines(&self) -> CrackLines {
        CrackLines {
            lines: self.lines.clone(),
            lines_next: Some(self.lines_next.clone()),
            jumps: self
                .fin
                .as_ref()
                .map(|f| f.0.jumps.clone())
                .unwrap_or_default(),
            bands: self.bands.clone(),
            rejected: self.rejected.clone(),
        }
    }
}

fn f64s(a: &[f32]) -> Vec<f64> {
    a.iter().map(|&v| f64::from(v)).collect()
}

/// The ROI of the frame 0 in the coordinates of a reference that frame 0 maps into by `u`
/// (first-order inverse, nearest pixel); None without an ROI or when it comes out empty
/// (`_roi_in`).
pub(super) fn roi_in(
    mask: Option<&Image<bool>>,
    u: &(Image<f32>, Image<f32>),
) -> Option<Image<bool>> {
    let m = mask?;
    let (w, h) = (m.w, m.h);
    let nz = |v: f32| if v.is_nan() { 0.0 } else { f64::from(v) };
    let data: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let xs = (x - nz(u.0.data[i]))
                .round_ties_even()
                .clamp(0.0, (w - 1) as f64) as usize;
            let ys = (y - nz(u.1.data[i]))
                .round_ties_even()
                .clamp(0.0, (h - 1) as f64) as usize;
            m.data[ys * w + xs]
        })
        .collect();
    data.iter().any(|&b| b).then(|| Image::from_vec(w, h, data))
}

/// The carried geometry for a leg whose reference is the target frame of `p` (`_hint`).
pub(super) fn hint(p: Option<&Pair>) -> Option<Vec<Poly>> {
    p?.xleg.as_ref().map(|x| x.lines_next.clone())
}

/// The carried geometry for a leg with the reference of `p` (`_same_ref_hint`).
pub(super) fn same_ref_hint(p: Option<&Pair>) -> Option<Vec<Poly>> {
    p?.xleg.as_ref().map(|x| x.lines.clone())
}

/// The coarse stage and the baseline fine stage of a pair for `crack_model = "xdic"`: the dense
/// start field as in [`pair_flow_on`], then the whole-image fine stage with its slope, as a
/// `Flow` with the confidence of the reference's `refine` (0 in the border strips).
fn baseline(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    cfg: &Config,
    init: Option<(&Image<f32>, &Image<f32>)>,
) -> Result<(Flow, Option<Prepared>, f64, f64), TrackError> {
    let (w, h) = (i1.w, i1.h);
    let t = prof::start(Stage::Coarse);
    let g = coarse_flow(i1, i2, cfg, init);
    let source = start_grid(&g, init);
    let whole = Rect {
        x0: 0,
        y0: 0,
        x1: w,
        y1: h,
    };
    let (gx, gy) = dense_start_on(&source, init, w, whole);
    let coarse = t.elapsed();
    prof::add(Stage::Coarse, t);
    let t = Instant::now();
    let (prep, (raw, slope)) = if engine.is_cpu_code() {
        let prep = Prepared::new(i1, i2, cfg);
        let out = refine_raw_prepared(&prep, w, h, (0, 0), &gx, &gy, cfg);
        (Some(prep), out)
    } else {
        (None, engine.refine_raw_with_slope(i1, i2, &gx, &gy, cfg)?)
    };
    let n = w * h;
    let scale = texture_scale(&raw.energy.data);
    let b = border_zone(cfg);
    let conf: Vec<f64> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            confidence(
                raw.energy.data[i],
                raw.resid.data[i],
                reportable(
                    x,
                    y,
                    x as f64 + raw.ux.data[i],
                    y as f64 + raw.uy.data[i],
                    w,
                    h,
                    b,
                ),
                scale,
                cfg.phase_tol,
            )
        })
        .collect();
    let fine = t.elapsed();
    Ok((
        Flow {
            w,
            h,
            ux: raw.ux.data,
            uy: raw.uy.data,
            conf,
            resid: raw.resid.data,
            slope,
        },
        prep,
        coarse.as_secs_f64(),
        fine.as_secs_f64(),
    ))
}

/// [`pair_flow_on`] for `crack_model = "xdic"` (`track_pair` of the reference up to the validity
/// decision, with `_track_pair_xdic`): the carried lines `hint` (crack polylines in `i1`'s
/// coordinates) plus the new lines of the detector on the baseline result, then the per-side
/// re-solve; lines without a jump in their own solution are dropped, detected lines for which
/// that solution does not explain the images better than the baseline are bands. `mask` is the
/// ROI in `i1`'s coordinates. No line: the baseline result.
pub(super) fn pair_flow_xdic(
    engine: &dyn FineEngine,
    i1: &Image<f32>,
    i2: &Image<f32>,
    cfg: &Config,
    init: Option<(&Image<f32>, &Image<f32>)>,
    mask: Option<&Image<bool>>,
    hint: Option<&[Poly]>,
) -> Result<Pair, TrackError> {
    if !engine.returns_slope() {
        return Err(TrackError::Engine(
            "crack_model 'xdic' runs on the CPU only (the per-side solve needs the slope of the \
             baseline solve, which this engine does not return); use --device cpu"
                .to_string(),
        ));
    }
    let (w, h) = (i1.w, i1.h);
    let n = w * h;
    if n > crate::xdic::MAX_PIXELS {
        return Err(TrackError::Engine(format!(
            "crack_model 'xdic' takes images of at most {} pixels (it holds about 1.1 GB per million \
             pixels, its baseline iterations run on the whole image), got {w}x{h}; limit the \
             region of interest",
            crate::xdic::MAX_PIXELS
        )));
    }
    let (fl0, prep, coarse, fine) = baseline(engine, i1, i2, cfg, init)?;
    let conf_min = cfg.conf_min;
    let t_detect = Instant::now();
    let low0: Vec<bool> = fl0.conf.iter().map(|&c| c < conf_min).collect();
    let (found, hint_lines) = prof::time(Stage::XdicDetect, || {
        let low_img = Image::from_vec(w, h, low0.clone());
        let bad = invalid_pixels(&low_img, cfg, mask, stage_bands(cfg, h, band_rows(w)).0);
        let f32s = |u: &[f64]| {
            Image::from_vec(
                w,
                h,
                (0..n)
                    .map(|i| if bad.data[i] { f32::NAN } else { u[i] as f32 })
                    .collect(),
            )
        };
        let (m0x, m0y) = (f32s(&fl0.ux), f32s(&fl0.uy));
        let strain = strain_banded(
            &m0x,
            &m0y,
            cfg.strain_win,
            stage_bands(cfg, h, band_rows(w)).1,
            StrainOpts { small: false },
        );
        let mx: Vec<f64> = (0..n)
            .map(|i| if bad.data[i] { f64::NAN } else { fl0.ux[i] })
            .collect();
        let my: Vec<f64> = (0..n)
            .map(|i| if bad.data[i] { f64::NAN } else { fl0.uy[i] })
            .collect();
        let (found, _) = detect_lines(&mx, &my, Some(&strain.gmax.data), w, h, cfg, mask);
        let hint_lines: Vec<Poly> = hint.map(|h| h.to_vec()).unwrap_or_default();
        (found, hint_lines)
    });
    // carried lines first; a detected line is added unless it lies along a carried one (most of
    // it within R_B)
    let rb = cfg.xdic_rb_px();
    let new: Vec<Poly> = found
        .into_iter()
        .filter(|l| !near_lines(l, &hint_lines, rb))
        .collect();
    let mut carried: Vec<bool> = vec![true; hint_lines.len()];
    carried.extend(std::iter::repeat_n(false, new.len()));
    let mut lines: Vec<Poly> = hint_lines;
    lines.extend(new);
    let t_detect = t_detect.elapsed().as_secs_f64();
    let mut bands: Vec<Poly> = Vec::new();
    let mut rejected: Vec<Poly> = Vec::new();
    let t_solve = Instant::now();
    let mut last: Option<(Flow, FinishCore)> = None;
    // built when the first line needs it: what is shared by the solves of the leg
    let mut shared: Option<(Prepared, SideData)> = None;
    let mut prep = prep;
    while !lines.is_empty() {
        let (pr, data) = shared.get_or_insert_with(|| {
            let pr = prep.take().unwrap_or_else(|| Prepared::new(i1, i2, cfg));
            let data = prof::time(Stage::XdicResponses, || SideData::new(i1, &pr));
            (pr, data)
        });
        let (flx, new_lines, _geo) = refine_xdic_on(pr, data, i1, &fl0, cfg, &lines);
        lines = new_lines;
        let low: Vec<bool> = flx.conf.iter().map(|&c| c < conf_min).collect();
        let core = prof::time(Stage::XdicFinish, || {
            finish_core(&flx.ux, &flx.uy, &low, cfg, mask, &lines, w, h)
        });
        // a crack must show a jump in its own per-side solution: lines whose median |J| is below
        // the detector's j_min (or that have no finite jump) are dropped and the rest is solved
        // again; so are the detected lines that are bands
        let jump_ok: Vec<bool> = core
            .jumps
            .iter()
            .map(|j| jump_magnitude(j) >= DET_JMIN)
            .collect();
        let gains: Vec<Option<ModelGain>> = if carried.iter().all(|&c| c) {
            vec![None; lines.len()]
        } else {
            prof::time(Stage::XdicFinish, || {
                model_gain(i1, &pr.spline2, &fl0, &flx, &lines, cfg, mask, 4.0)
                    .into_iter()
                    .map(Some)
                    .collect()
            })
        };
        let model_ok: Vec<bool> = carried
            .iter()
            .zip(&gains)
            .map(|(&c, m)| c || m.as_ref().is_some_and(is_crack))
            .collect();
        let done = jump_ok.iter().all(|&b| b) && model_ok.iter().all(|&b| b);
        if done {
            last = Some((flx, core));
            break;
        }
        rejected.extend(
            lines
                .iter()
                .zip(&jump_ok)
                .filter(|(_, &k)| !k)
                .map(|(l, _)| l.clone()),
        );
        bands.extend(
            lines
                .iter()
                .zip(&jump_ok)
                .zip(&model_ok)
                .filter(|((_, &j), &k)| j && !k)
                .map(|((l, _), _)| l.clone()),
        );
        let keep: Vec<bool> = jump_ok
            .iter()
            .zip(&model_ok)
            .map(|(&j, &k)| j && k)
            .collect();
        carried = carried
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(&c, _)| c)
            .collect();
        lines = lines
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(l, _)| l.clone())
            .collect();
    }
    let t_xdic = t_solve.elapsed().as_secs_f64();
    let Some((flx, core)) = last.filter(|_| !lines.is_empty()) else {
        // no line: the baseline result
        let (ux, uy) = (
            Image::from_vec(w, h, fl0.ux.iter().map(|&v| v as f32).collect()),
            Image::from_vec(w, h, fl0.uy.iter().map(|&v| v as f32).collect()),
        );
        return Ok(Pair {
            raw_ux: ux,
            raw_uy: uy,
            conf: Image::from_vec(w, h, fl0.conf.iter().map(|&c| c as f32).collect()),
            low: Image::from_vec(w, h, low0),
            coarse,
            fine,
            xleg: Some(Box::new(XLeg {
                lines: Vec::new(),
                lines_next: Vec::new(),
                bands,
                rejected,
                fin: None,
                t_detect,
                t_xdic,
            })),
        });
    };
    let lines_next = prof::time(Stage::XdicFinish, || {
        advance_lines(&lines, &flx.ux, &flx.uy, w, h)
    });
    let low: Vec<bool> = flx.conf.iter().map(|&c| c < conf_min).collect();
    Ok(Pair {
        raw_ux: Image::from_vec(w, h, flx.ux.iter().map(|&v| v as f32).collect()),
        raw_uy: Image::from_vec(w, h, flx.uy.iter().map(|&v| v as f32).collect()),
        conf: Image::from_vec(w, h, flx.conf.iter().map(|&c| c as f32).collect()),
        low: Image::from_vec(w, h, low),
        coarse,
        fine,
        xleg: Some(Box::new(XLeg {
            lines,
            lines_next,
            bands,
            rejected,
            fin: Some((core, flx.ux, flx.uy)),
            t_detect,
            t_xdic,
        })),
    })
}

/// The `Timing` fields of the leg's crack stages.
pub(super) fn leg_times(p: &Pair) -> (Option<f64>, Option<f64>) {
    match &p.xleg {
        Some(x) => (Some(x.t_detect), Some(x.t_xdic)),
        None => (None, None),
    }
}

/// The result of a pair with crack lines: `ux`, `uy` masked, the one-sided strain, the maps and
/// the lines.
pub(super) fn pair_result_with_lines(
    pair: Pair,
    cfg: &Config,
    opts: StrainOpts,
    t0: Instant,
) -> TrackResult {
    let x = pair
        .xleg
        .as_ref()
        .expect("a pair with lines has its crack leg");
    let (core, ux, uy) = x
        .fin
        .as_ref()
        .expect("a pair with lines has its final solve");
    let (mx, my, strain, maps) = prof::time(Stage::XdicFinish, || {
        finish_outputs(core, ux, uy, cfg, opts.small)
    });
    let lines = x.crack_lines();
    let timing = Timing {
        coarse: pair.coarse,
        fine: pair.fine,
        strain: t0.elapsed().as_secs_f64(),
        xdic_detect: Some(x.t_detect),
        xdic: Some(x.t_xdic),
        ..Timing::default()
    };
    TrackResult {
        ux: mx,
        uy: my,
        conf: pair.conf,
        strain,
        raw_ux: pair.raw_ux,
        raw_uy: pair.raw_uy,
        timing,
        fb: None,
        crack: Some(Box::new(CrackOut {
            maps: Some(maps),
            lines,
        })),
    }
}

/// Frame-0 outputs of a sequence frame for `crack_model = "xdic"`: the leg's lines mapped back to
/// frame 0, and the validity, strain and crack maps on the frame-0 grid from the displacement
/// `(ux, uy)` there (`_xdic_out`). None when the leg has no line.
pub(super) fn frame0_out(
    leg: &Pair,
    u0r: &(Image<f32>, Image<f32>),
    ux: &Image<f32>,
    uy: &Image<f32>,
    low: &[bool],
    cfg: &Config,
    mask: Option<&Image<bool>>,
    opts: StrainOpts,
) -> Option<(Image<f32>, Image<f32>, Strain, Box<CrackOut>)> {
    let x = leg.xleg.as_ref().filter(|x| x.has_lines())?;
    let (w, h) = (ux.w, ux.h);
    let (a, b) = (f64s(&u0r.0.data), f64s(&u0r.1.data));
    let lines0 = prof::time(Stage::XdicFinish, || {
        lines_to_frame0(&x.lines, Some((&a, &b)), w, h, 6.0)
    });
    let (cx, cy) = (f64s(&ux.data), f64s(&uy.data));
    let core = prof::time(Stage::XdicFinish, || {
        finish_core(&cx, &cy, low, cfg, mask, &lines0, w, h)
    });
    let (mx, my, strain, maps) = prof::time(Stage::XdicFinish, || {
        finish_outputs(&core, &cx, &cy, cfg, opts.small)
    });
    let cl = CrackLines {
        lines: lines0,
        lines_next: None,
        jumps: core.jumps,
        bands: Vec::new(),
        rejected: Vec::new(),
    };
    Some((
        mx,
        my,
        strain,
        Box::new(CrackOut {
            maps: Some(maps),
            lines: cl,
        }),
    ))
}

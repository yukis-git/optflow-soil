//! What the window shows: pictures reduced to a texture size, and the worker thread that prepares
//! them, because on a 6240x4160 input reading the maps, the colour range and the colours take
//! seconds. Display only: the saved maps, the CSVs and the PNG export never pass through here.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use anyhow::Result;
use rayon::prelude::*;

use crate::colormap::Scale;
use crate::crack::{self, CrackLines};
use crate::export;
use crate::fields::Fields;
use crate::messages;
use crate::source::Gray8;
use crate::stats::{auto_range, Overlay};

/// The longest texture side. A window on a 4K screen shows at most about this many pixels, and
/// some drivers refuse more than 4096 or 8192 (the window also asks the driver for its limit).
pub const MAX_TEXTURE_SIDE: usize = 2560;

/// The whole-number reduction that brings the longer side down to `max_side`.
pub fn display_factor(w: usize, h: usize, max_side: usize) -> usize {
    w.max(h).div_ceil(max_side.max(1)).max(1)
}

/// Block means of a grey picture, `factor` pixels to a side: `(w, h, data)`.
pub fn shrink_gray(bg: &Gray8, factor: usize) -> (usize, usize, Vec<u8>) {
    let (w2, h2) = (bg.w.div_ceil(factor), bg.h.div_ceil(factor));
    let mut out = vec![0u8; w2 * h2];
    out.par_chunks_mut(w2.max(1))
        .enumerate()
        .for_each(|(y2, row)| {
            for (x2, o) in row.iter_mut().enumerate() {
                let (mut s, mut n) = (0u32, 0u32);
                for y in y2 * factor..((y2 + 1) * factor).min(bg.h) {
                    for x in x2 * factor..((x2 + 1) * factor).min(bg.w) {
                        s += bg.data[y * bg.w + x] as u32;
                        n += 1;
                    }
                }
                *o = ((s + n / 2) / n) as u8;
            }
        });
    (w2, h2, out)
}

/// A picture for the window, RGB.
pub struct ViewImage {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

/// The overlay over `bg`, as `export::overlay_rgba` and `export::compose_rgb` make it for the PNG,
/// reduced by block means of `factor` pixels to a side (1: exactly those bytes).
pub fn compose_view(
    bg: &Gray8,
    f: &Fields,
    kind: Overlay,
    lo: f32,
    hi: f32,
    alpha: f32,
    factor: usize,
) -> ViewImage {
    let a = export::overlay_alpha(alpha);
    let scale: Scale = export::overlay_scale(kind);
    let map = kind.map(f);
    let (w2, h2) = (f.w.div_ceil(factor), f.h.div_ceil(factor));
    let mut rgb = vec![0u8; w2 * h2 * 3];
    rgb.par_chunks_mut((w2 * 3).max(1))
        .enumerate()
        .for_each(|(y2, row)| {
            for (x2, o) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let (mut s, mut n) = ([0u32; 3], 0u32);
                for y in y2 * factor..((y2 + 1) * factor).min(f.h) {
                    for x in x2 * factor..((x2 + 1) * factor).min(f.w) {
                        let i = y * f.w + x;
                        let p = export::overlay_pixel(f, map, kind, scale, lo, hi, a, i);
                        let c = export::blend(bg.data[i], &p);
                        for (t, v) in s.iter_mut().zip(c) {
                            *t += v as u32;
                        }
                        n += 1;
                    }
                }
                for (v, t) in o.iter_mut().zip(s) {
                    *v = ((t + n / 2) / n) as u8;
                }
            }
        });
    ViewImage { w: w2, h: h2, rgb }
}

/// Everything the shown picture depends on. Range values count only without the automatic range.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewKey {
    pub run: PathBuf,
    pub k: usize,
    pub overlay: Overlay,
    pub auto: bool,
    pub lo: u32,
    pub hi: u32,
    pub alpha: u32,
    pub max_side: usize,
}

impl ViewKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run: PathBuf,
        k: usize,
        overlay: Overlay,
        auto: bool,
        lo: f32,
        hi: f32,
        alpha: f32,
        max_side: usize,
    ) -> ViewKey {
        let bits = |v: f32| if auto { 0 } else { v.to_bits() };
        ViewKey {
            run,
            k,
            overlay,
            auto,
            lo: bits(lo),
            hi: bits(hi),
            alpha: alpha.to_bits(),
            max_side,
        }
    }
}

/// A finished picture. `image` is `None` when the maps are not the size of the first frame.
pub struct ViewResult {
    pub key: ViewKey,
    pub fields: Arc<Fields>,
    pub crack: Option<CrackLines>,
    pub lo: f32,
    pub hi: f32,
    pub image: Option<ViewImage>,
    /// Reduction of `image` from the maps.
    pub factor: usize,
}

/// What the worker keeps between requests: the one frame in memory and its last colour range.
#[derive(Default)]
pub struct ViewCache {
    loaded: Option<(PathBuf, usize, Arc<Fields>, Option<CrackLines>)>,
    range: Option<(PathBuf, usize, Overlay, (f32, f32))>,
}

/// Prepares the picture of `key`.
pub fn render(cache: &mut ViewCache, bg: &Gray8, key: &ViewKey) -> Result<ViewResult> {
    let hit =
        |l: &(PathBuf, usize, Arc<Fields>, Option<CrackLines>)| l.0 == key.run && l.1 == key.k;
    if !cache.loaded.as_ref().is_some_and(hit) {
        cache.loaded = None; // the old maps go before the new ones arrive
        let f = Fields::load(&key.run, key.k)?;
        cache.loaded = Some((
            key.run.clone(),
            key.k,
            Arc::new(f),
            crack::load(&key.run, key.k),
        ));
    }
    let (_, _, f, crack) = cache.loaded.as_ref().expect("loaded above");
    let f = Arc::clone(f);
    let (lo, hi) = if key.auto {
        match &cache.range {
            Some((r, k, o, v)) if *r == key.run && *k == key.k && *o == key.overlay => *v,
            _ => {
                let v = auto_range(key.overlay, &f);
                cache.range = Some((key.run.clone(), key.k, key.overlay, v));
                v
            }
        }
    } else {
        (f32::from_bits(key.lo), f32::from_bits(key.hi))
    };
    let factor = display_factor(f.w, f.h, key.max_side);
    let image = ((bg.w, bg.h) == (f.w, f.h)).then(|| {
        let alpha = f32::from_bits(key.alpha);
        compose_view(bg, &f, key.overlay, lo, hi, alpha, factor)
    });
    Ok(ViewResult {
        key: key.clone(),
        fields: f,
        crack: crack.clone(),
        lo,
        hi,
        image,
        factor,
    })
}

/// The picture-making thread. Only the latest request counts: while a slider moves, the ones in
/// between are skipped. It ends with this value.
pub struct ViewWorker {
    tx: Sender<(Arc<Gray8>, ViewKey)>,
    pub rx: Receiver<Result<ViewResult, (ViewKey, String)>>,
}

impl ViewWorker {
    pub fn start(repaint: Arc<dyn Fn() + Send + Sync>) -> ViewWorker {
        let (tx, jobs) = channel::<(Arc<Gray8>, ViewKey)>();
        let (done, rx) = channel();
        std::thread::spawn(move || {
            let mut cache = ViewCache::default();
            while let Ok(mut job) = jobs.recv() {
                while let Ok(newer) = jobs.try_recv() {
                    job = newer;
                }
                let (bg, key) = job;
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    render(&mut cache, &bg, &key)
                }));
                let out = match out {
                    Ok(Ok(r)) => Ok(r),
                    Ok(Err(e)) => Err((key, messages::describe(&e))),
                    Err(_) => {
                        cache = ViewCache::default();
                        Err((key, "結果を表示できません(内部エラー)。".to_string()))
                    }
                };
                if done.send(out).is_err() {
                    return;
                }
                repaint();
            }
        });
        ViewWorker { tx, rx }
    }

    pub fn request(&self, bg: Arc<Gray8>, key: ViewKey) {
        let _ = self.tx.send((bg, key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optflow_core::image::Image;

    fn sample_fields(w: usize, h: usize) -> Fields {
        let n = w * h;
        let ramp: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let mut ux = ramp.clone();
        ux[0] = f32::NAN;
        ux[n / 2] = f32::NAN;
        Fields {
            w,
            h,
            ux,
            uy: ramp.clone(),
            conf: ramp.clone(),
            gmax: ramp.clone(),
            vol: ramp,
            fb: None,
            cod: None,
            csd: None,
        }
    }

    fn sample_bg(w: usize, h: usize) -> Gray8 {
        Gray8 {
            w,
            h,
            data: (0..w * h).map(|i| (i * 7 % 256) as u8).collect(),
        }
    }

    #[test]
    fn the_factor_brings_the_longer_side_within_the_limit() {
        assert_eq!(display_factor(6240, 4160, MAX_TEXTURE_SIDE), 3);
        assert_eq!(display_factor(4160, 6240, MAX_TEXTURE_SIDE), 3);
        assert_eq!(display_factor(2560, 100, MAX_TEXTURE_SIDE), 1);
        assert_eq!(display_factor(2561, 100, MAX_TEXTURE_SIDE), 2);
        assert_eq!(display_factor(160, 120, MAX_TEXTURE_SIDE), 1);
        assert_eq!(display_factor(16000, 100, 2048), 8);
    }

    #[test]
    fn at_factor_one_the_view_is_the_bytes_of_the_png_export() {
        let (bg, f) = (sample_bg(17, 11), sample_fields(17, 11));
        for kind in [Overlay::Gmax, Overlay::Ux, Overlay::Conf] {
            let o = export::overlay_rgba(&f, kind, 0.1, 0.9, 0.7);
            let full = export::compose_rgb(&bg, Some(&o), &[]);
            let v = compose_view(&bg, &f, kind, 0.1, 0.9, 0.7, 1);
            assert_eq!((v.w, v.h), (17, 11));
            assert_eq!(v.rgb, full, "{kind:?}");
        }
    }

    #[test]
    fn reduced_views_are_block_means_including_short_last_blocks() {
        let (bg, f) = (sample_bg(7, 5), sample_fields(7, 5));
        let o = export::overlay_rgba(&f, Overlay::Gmax, 0.0, 1.0, 0.5);
        let full = export::compose_rgb(&bg, Some(&o), &[]);
        let v = compose_view(&bg, &f, Overlay::Gmax, 0.0, 1.0, 0.5, 3);
        assert_eq!((v.w, v.h), (3, 2));
        let mean = |xs: std::ops::Range<usize>, ys: std::ops::Range<usize>, c: usize| {
            let (mut s, mut n) = (0u32, 0u32);
            for y in ys {
                for x in xs.clone() {
                    s += full[3 * (y * 7 + x) + c] as u32;
                    n += 1;
                }
            }
            ((s + n / 2) / n) as u8
        };
        for c in 0..3 {
            assert_eq!(v.rgb[c], mean(0..3, 0..3, c));
            assert_eq!(v.rgb[3 * (3 + 2) + c], mean(6..7, 3..5, c));
        }
    }

    #[test]
    fn the_shrunken_grey_picture_is_block_means() {
        let bg = Gray8 {
            w: 3,
            h: 2,
            data: vec![0, 10, 200, 20, 30, 100],
        };
        assert_eq!(shrink_gray(&bg, 1), (3, 2, bg.data.clone()));
        assert_eq!(shrink_gray(&bg, 2), (2, 1, vec![15, 150]));
    }

    fn write_run(run: &std::path::Path, k: usize, w: usize, h: usize, v: f32) {
        let dir = crate::fields::frame_dir(run, k);
        std::fs::create_dir_all(&dir).unwrap();
        let data: Vec<f32> = (0..w * h).map(|i| v + i as f32).collect();
        for name in ["ux", "uy", "conf", "gmax", "vol"] {
            let im = Image::from_vec(w, h, data.clone());
            optflow_cli::io::write_npy_f32(&dir.join(format!("{name}.npy")), &im).unwrap();
        }
    }

    #[test]
    fn render_follows_the_key_and_a_size_mismatch_gives_no_picture() {
        let run = std::env::temp_dir().join(format!("optflow-app-view-{}", std::process::id()));
        write_run(&run, 1, 8, 6, 0.0);
        write_run(&run, 2, 8, 6, 100.0);
        let bg = sample_bg(8, 6);
        let mut cache = ViewCache::default();
        let key = |k, auto, lo: f32, hi: f32| {
            ViewKey::new(run.clone(), k, Overlay::Gmax, auto, lo, hi, 0.5, 4)
        };
        let a = render(&mut cache, &bg, &key(1, true, 0.0, 0.0)).unwrap();
        assert_eq!(a.factor, 2);
        assert_eq!(a.image.as_ref().map(|i| (i.w, i.h)), Some((4, 3)));
        assert_eq!((a.lo, a.hi), auto_range(Overlay::Gmax, &a.fields));
        // the same frame is not read again, another one is
        let b = render(&mut cache, &bg, &key(1, false, 2.0, 3.0)).unwrap();
        assert!(Arc::ptr_eq(&a.fields, &b.fields));
        assert_eq!((b.lo, b.hi), (2.0, 3.0));
        let c = render(&mut cache, &bg, &key(2, true, 0.0, 0.0)).unwrap();
        assert!(!Arc::ptr_eq(&a.fields, &c.fields));
        assert!(c.lo == 0.0 && c.hi >= 100.0, "{} {}", c.lo, c.hi);
        // a scaled run: maps smaller than the first frame
        let small = sample_bg(4, 3);
        assert!(render(&mut cache, &small, &key(2, true, 0.0, 0.0))
            .unwrap()
            .image
            .is_none());
        assert!(render(&mut cache, &bg, &key(9, true, 0.0, 0.0)).is_err());
        let _ = std::fs::remove_dir_all(&run);
    }

    #[test]
    fn the_worker_answers_and_reports_a_missing_frame() {
        let run = std::env::temp_dir().join(format!("optflow-app-vw-{}", std::process::id()));
        write_run(&run, 1, 4, 4, 0.0);
        let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let w = Arc::clone(&woken);
        let worker = ViewWorker::start(Arc::new(move || {
            w.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }));
        let bg = Arc::new(sample_bg(4, 4));
        let key = ViewKey::new(run.clone(), 1, Overlay::Ux, true, 0.0, 0.0, 0.7, 2560);
        worker.request(Arc::clone(&bg), key.clone());
        let got = worker
            .rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap();
        assert!(matches!(got, Ok(r) if r.key == key));
        let bad = ViewKey::new(run.clone(), 7, Overlay::Ux, true, 0.0, 0.0, 0.7, 2560);
        worker.request(bg, bad.clone());
        let got = worker
            .rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap();
        assert!(matches!(got, Err((k, _)) if k == bad));
        // an idle window only repaints when asked: every answer must wake it
        for _ in 0..1000 {
            if woken.load(std::sync::atomic::Ordering::SeqCst) == 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(woken.load(std::sync::atomic::Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&run);
    }
}

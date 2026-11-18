//! 「動画で書き出す…」: the computed results of a run as an MP4, one video frame per result. Each
//! frame is what the canvas shows (the quantity over the first frame, the ROI outline, the crack
//! lines) with one colour range for the whole video, below it a strip with the colour bar and the
//! frame number and time. Raw RGB frames are piped to `ffmpeg` one at a time: no PNG files, and
//! only one result in memory.

use std::io::Write;
use std::path::Path;
use std::process::Stdio;

use anyhow::{anyhow, bail, Context, Result};

use crate::colormap::color;
use crate::crack::CrackLines;
use crate::export::{self, Timing};
use crate::fields::Fields;
use crate::source::Gray8;
use crate::stats::{auto_range, Overlay};

/// An H.264 (or fallback) encoder of the installed `ffmpeg`, with its arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoder {
    pub name: &'static str,
    pub args: &'static [&'static str],
}

const LIBX264: Encoder = Encoder {
    name: "libx264",
    args: &["-c:v", "libx264", "-crf", "18", "-preset", "medium"],
};
const VIDEOTOOLBOX: Encoder = Encoder {
    name: "h264_videotoolbox",
    args: &["-c:v", "h264_videotoolbox", "-b:v", "20M"],
};
const MPEG4: Encoder = Encoder {
    name: "mpeg4",
    args: &["-c:v", "mpeg4", "-q:v", "2"],
};

/// The encoder to use given the output of `ffmpeg -encoders`: libx264 (crf 18), else
/// h264_videotoolbox on macOS, else mpeg4 at a high quality.
pub fn pick_encoder(listing: &str, macos: bool) -> Option<Encoder> {
    let has = |name: &str| {
        listing.lines().any(|l| {
            let mut it = l.split_whitespace();
            matches!((it.next(), it.next()), (Some(flags), Some(n)) if n == name && flags.starts_with('V'))
        })
    };
    if has("libx264") {
        Some(LIBX264)
    } else if macos && has("h264_videotoolbox") {
        Some(VIDEOTOOLBOX)
    } else if has("mpeg4") {
        Some(MPEG4)
    } else {
        None
    }
}

/// Asks the `ffmpeg` of the app (`ffmpeg -encoders`, once) for an encoder.
pub fn detect_encoder() -> Result<Encoder> {
    let out = optflow_cli::video::tool_command("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .context("ffmpeg を実行できません")?;
    let text = String::from_utf8_lossy(&out.stdout);
    pick_encoder(&text, cfg!(target_os = "macos")).ok_or_else(|| {
        anyhow!("この ffmpeg には動画を書き出せるエンコーダー(libx264 など)がありません")
    })
}

/// The name and unit of a quantity as written on the video (ASCII: the video has no font).
pub fn label(kind: Overlay) -> (&'static str, &'static str) {
    match kind {
        Overlay::Gmax => ("gmax", ""),
        Overlay::Ux => ("ux", "px"),
        Overlay::Uy => ("uy", "px"),
        Overlay::Conf => ("conf", ""),
        Overlay::Cod => ("COD", "px"),
        Overlay::Csd => ("CSD", "px"),
    }
}

/// A number for the colour bar: a few significant digits.
fn num(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if !(0.001..10000.0).contains(&a) {
        format!("{v:.1e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

/// A 5x7 bitmap font: the characters the strip needs. Rows top to bottom, 5 bits each, the leftmost
/// pixel in bit 4.
fn glyph(c: char) -> [u8; 7] {
    let g = |rows: [&str; 7]| {
        let mut o = [0u8; 7];
        for (i, r) in rows.iter().enumerate() {
            for (j, ch) in r.bytes().enumerate() {
                if ch == b'#' {
                    o[i] |= 1 << (4 - j);
                }
            }
        }
        o
    };
    match c {
        '0' => g([
            ".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###.",
        ]),
        '1' => g([
            "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
        ]),
        '2' => g([
            ".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####",
        ]),
        '3' => g([
            ".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###.",
        ]),
        '4' => g([
            "...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#.",
        ]),
        '5' => g([
            "#####", "#....", "####.", "....#", "....#", "#...#", ".###.",
        ]),
        '6' => g([
            "..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###.",
        ]),
        '7' => g([
            "#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#...",
        ]),
        '8' => g([
            ".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###.",
        ]),
        '9' => g([
            ".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##..",
        ]),
        '.' => g([
            ".....", ".....", ".....", ".....", ".....", ".##..", ".##..",
        ]),
        '-' => g([
            ".....", ".....", ".....", "#####", ".....", ".....", ".....",
        ]),
        '+' => g([
            ".....", "..#..", "..#..", "#####", "..#..", "..#..", ".....",
        ]),
        '=' => g([
            ".....", ".....", "#####", ".....", "#####", ".....", ".....",
        ]),
        '[' => g([
            ".###.", ".#...", ".#...", ".#...", ".#...", ".#...", ".###.",
        ]),
        ']' => g([
            ".###.", "...#.", "...#.", "...#.", "...#.", "...#.", ".###.",
        ]),
        '/' => g([
            "....#", "....#", "...#.", "..#..", ".#...", "#....", "#....",
        ]),
        ':' => g([
            ".....", ".##..", ".##..", ".....", ".##..", ".##..", ".....",
        ]),
        'a' => g([
            ".....", ".....", ".###.", "....#", ".####", "#...#", ".####",
        ]),
        'c' => g([
            ".....", ".....", ".###.", "#....", "#....", "#...#", ".###.",
        ]),
        'd' => g([
            "....#", "....#", ".####", "#...#", "#...#", "#...#", ".####",
        ]),
        'e' => g([
            ".....", ".....", ".###.", "#...#", "#####", "#....", ".###.",
        ]),
        'f' => g([
            "..##.", ".#..#", ".#...", "###..", ".#...", ".#...", ".#...",
        ]),
        'g' => g([
            ".....", ".####", "#...#", "#...#", ".####", "....#", ".###.",
        ]),
        'm' => g([
            ".....", ".....", "##.#.", "#.#.#", "#.#.#", "#...#", "#...#",
        ]),
        'n' => g([
            ".....", ".....", "#.##.", "##..#", "#...#", "#...#", "#...#",
        ]),
        'o' => g([
            ".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.",
        ]),
        'p' => g([
            ".....", ".....", "####.", "#...#", "####.", "#....", "#....",
        ]),
        'r' => g([
            ".....", ".....", "#.##.", "##..#", "#....", "#....", "#....",
        ]),
        's' => g([
            ".....", ".....", ".####", "#....", ".###.", "....#", "####.",
        ]),
        't' => g([
            ".#...", ".#...", "###..", ".#...", ".#...", ".#..#", "..##.",
        ]),
        'u' => g([
            ".....", ".....", "#...#", "#...#", "#...#", "#..##", ".##.#",
        ]),
        'x' => g([
            ".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#",
        ]),
        'y' => g([
            ".....", "#...#", "#...#", ".####", "....#", "....#", ".###.",
        ]),
        'C' => g([
            ".###.", "#...#", "#....", "#....", "#....", "#...#", ".###.",
        ]),
        'D' => g([
            "####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####.",
        ]),
        'O' => g([
            ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
        ]),
        'S' => g([
            ".###.", "#...#", "#....", ".###.", "....#", "#...#", ".###.",
        ]),
        _ => [0; 7],
    }
}

/// Draws `text` at (`x`, `y`) in white, each font pixel `sc` pixels wide. Returns the width drawn.
fn text(rgb: &mut [u8], w: usize, h: usize, x: usize, y: usize, sc: usize, s: &str) -> usize {
    let mut cx = x;
    for ch in s.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits >> (4 - col) & 1 == 1 {
                    for dy in 0..sc {
                        for dx in 0..sc {
                            let (px, py) = (cx + col * sc + dx, y + row * sc + dy);
                            if px < w && py < h {
                                let i = (py * w + px) * 3;
                                rgb[i..i + 3].copy_from_slice(&[255, 255, 255]);
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * sc;
    }
    cx - x
}

/// The width of `s` in the font at pixel size `sc`.
fn text_width(sc: usize, s: &str) -> usize {
    6 * sc * s.chars().count()
}

/// What every frame of a video has in common.
pub struct Style<'a> {
    pub bg: &'a Gray8,
    pub roi: &'a [(f64, f64)],
    pub kind: Overlay,
    pub lo: f32,
    pub hi: f32,
    pub alpha: f32,
}

/// The size of the video frame for a picture of `w` x `h`: the picture and the strip below it,
/// padded to even sides (yuv420p).
pub fn frame_size(w: usize, h: usize) -> (usize, usize) {
    let sc = (h / 540).max(1);
    let (fw, fh) = (w, h + 15 * sc);
    (fw + fw % 2, fh + fh % 2)
}

/// One video frame as RGB: the canvas picture of result `f`, and the strip (the colour bar with its
/// range and unit, and `caption`).
pub fn frame_rgb(
    st: &Style,
    f: &Fields,
    crack: Option<&CrackLines>,
    caption: &str,
) -> Result<(usize, usize, Vec<u8>)> {
    if (f.w, f.h) != (st.bg.w, st.bg.h) {
        bail!("結果が入力画像と大きさが異なります(縮小率 2 以上)");
    }
    let (w, h) = (f.w, f.h);
    let o = export::overlay_rgba(f, st.kind, st.lo, st.hi, st.alpha);
    let mut pic = export::compose_rgb(st.bg, Some(&o), st.roi);
    if let Some(c) = crack {
        for l in &c.bands {
            export::draw_polyline(&mut pic, w, h, l, [255, 0, 255]);
        }
        for l in &c.lines {
            export::draw_polyline(&mut pic, w, h, l, [0, 255, 255]);
        }
    }
    let sc = (h / 540).max(1);
    let (fw, fh) = frame_size(w, h);
    let mut rgb = vec![0u8; fw * fh * 3];
    for y in 0..h {
        rgb[y * fw * 3..y * fw * 3 + w * 3].copy_from_slice(&pic[y * w * 3..(y + 1) * w * 3]);
    }
    // the strip: dark, below the picture
    for px in rgb[h * fw * 3..].as_chunks_mut::<3>().0.iter_mut() {
        px.copy_from_slice(&[24, 24, 24]);
    }
    let (name, unit) = label(st.kind);
    let head = if unit.is_empty() {
        name.to_string()
    } else {
        format!("{name} [{unit}]")
    };
    let y0 = h + 4 * sc;
    let mut x = 4 * sc;
    x += text(&mut rgb, fw, fh, x, y0, sc, &head) + 6 * sc;
    let lo = num(st.lo);
    x += text(&mut rgb, fw, fh, x, y0, sc, &lo) + 4 * sc;
    let bw = (w / 3).max(40);
    let scale = export::overlay_scale(st.kind);
    for i in 0..bw {
        let v = st.lo + (st.hi - st.lo) * i as f32 / (bw - 1) as f32;
        let c = color(scale, v, st.lo, st.hi);
        for yy in y0..y0 + 7 * sc {
            if x + i < fw {
                let k = (yy * fw + x + i) * 3;
                rgb[k..k + 3].copy_from_slice(&c);
            }
        }
    }
    x += bw + 4 * sc;
    text(&mut rgb, fw, fh, x, y0, sc, &num(st.hi));
    let cw = text_width(sc, caption);
    text(
        &mut rgb,
        fw,
        fh,
        fw.saturating_sub(cw + 4 * sc),
        y0,
        sc,
        caption,
    );
    Ok((fw, fh, rgb))
}

/// What a running export reports: the stage and how far it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Reading every result once for the one colour range of the whole video.
    Range,
    /// Drawing and encoding.
    Encode,
}

/// The colour range common to all frames: the lowest low and the highest high of the automatic
/// ranges of the results.
pub fn common_range(kind: Overlay, ranges: impl IntoIterator<Item = (f32, f32)>) -> (f32, f32) {
    let mut it = ranges.into_iter();
    let Some(first) = it.next() else {
        return (0.0, 1.0);
    };
    let _ = kind;
    it.fold(first, |(a, b), (c, d)| (a.min(c), b.max(d)))
}

/// Writes the results `ks` of `run` as an MP4 at `dest`, `out_fps` frames per second. `range` is the
/// colour range, or `None` for the automatic one over all the results. `progress(stage, done, total)`
/// is called as it goes and returns `false` to cancel. The file is removed when it fails or is
/// cancelled; `Ok(false)` is a cancel.
#[allow(clippy::too_many_arguments)]
pub fn export_video(
    run: &Path,
    dest: &Path,
    ks: &[usize],
    t: &Timing,
    bg: &Gray8,
    roi: &[(f64, f64)],
    kind: Overlay,
    range: Option<(f32, f32)>,
    alpha: f32,
    out_fps: f64,
    enc: &Encoder,
    progress: &mut dyn FnMut(Stage, usize, usize) -> bool,
) -> Result<bool> {
    if ks.is_empty() {
        bail!("書き出す結果がありません");
    }
    let (lo, hi) = match range {
        Some(r) => r,
        None => {
            let mut v = Vec::with_capacity(ks.len());
            for (i, &k) in ks.iter().enumerate() {
                if !progress(Stage::Range, i, ks.len()) {
                    return Ok(false);
                }
                let f = Fields::load(run, k)?;
                v.push(auto_range(kind, &f));
            }
            common_range(kind, v)
        }
    };
    let (w, h) = (bg.w, bg.h);
    let (fw, fh) = frame_size(w, h);
    let st = Style {
        bg,
        roi,
        kind,
        lo,
        hi,
        alpha,
    };
    let fps = if out_fps.is_finite() && out_fps > 0.0 {
        out_fps
    } else {
        10.0
    };
    let mut cmd = optflow_cli::video::tool_command("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"])
        .args(["-pix_fmt", "rgb24", "-s", &format!("{fw}x{fh}")])
        .args(["-framerate", &format!("{fps}"), "-i", "-", "-an"])
        .args(enc.args)
        .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        .arg(dest)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().context("ffmpeg を起動できません")?;
    let mut stdin = child.stdin.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut stderr, &mut s);
        s
    });
    let mut outcome: Result<bool> = Ok(true);
    for (i, &k) in ks.iter().enumerate() {
        if !progress(Stage::Encode, i, ks.len()) {
            outcome = Ok(false);
            break;
        }
        let one = (|| -> Result<Vec<u8>> {
            let f = Fields::load(run, k)?;
            let crack = crate::crack::load(run, k);
            let caption = format!("frame {} t={:.2} s", t.source_frame(k), t.time_s(k));
            Ok(frame_rgb(&st, &f, crack.as_ref(), &caption)?.2)
        })();
        match one {
            Ok(rgb) => {
                if stdin.write_all(&rgb).is_err() {
                    outcome = Err(anyhow!("ffmpeg が止まりました"));
                    break;
                }
            }
            Err(e) => {
                outcome = Err(e);
                break;
            }
        }
    }
    drop(stdin);
    if !matches!(outcome, Ok(true)) {
        // a cancel or a failure: stop the encoder, do not wait for it to finish the file
        let _ = child.kill();
    }
    let status = child.wait();
    let errors = err_thread.join().unwrap_or_default();
    match (&outcome, status) {
        (Ok(true), Ok(s)) if s.success() => {
            let _ = progress(Stage::Encode, ks.len(), ks.len());
            Ok(true)
        }
        (Ok(true), _) => {
            let _ = std::fs::remove_file(dest);
            bail!("ffmpeg が失敗しました: {}", errors.trim());
        }
        (Ok(false), _) => {
            let _ = std::fs::remove_file(dest);
            Ok(false)
        }
        (Err(_), _) => {
            let _ = std::fs::remove_file(dest);
            let e = outcome.expect_err("matched above");
            if errors.trim().is_empty() {
                Err(e)
            } else {
                Err(e.context(errors.trim().to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{run_job, JobSpec, Msg};
    use crate::settings::{Preset, Settings};
    use std::sync::atomic::AtomicBool;

    #[test]
    fn the_encoder_is_the_first_that_exists() {
        let all = " V..... libx264              H.264\n V....D h264_videotoolbox   VideoToolbox\n V....D mpeg4   MPEG-4\n A....D aac   AAC\n";
        assert_eq!(pick_encoder(all, true).unwrap().name, "libx264");
        let no264 = " V....D h264_videotoolbox   VideoToolbox\n V....D mpeg4   MPEG-4\n";
        assert_eq!(pick_encoder(no264, true).unwrap().name, "h264_videotoolbox");
        // the VideoToolbox encoder is for macOS only
        assert_eq!(pick_encoder(no264, false).unwrap().name, "mpeg4");
        // an audio encoder of the same name does not count
        assert!(pick_encoder(" A....D mpeg4 x\n", false).is_none());
        assert!(pick_encoder("", true).is_none());
    }

    #[test]
    fn the_frame_is_even_and_has_a_strip_with_the_bar() {
        assert_eq!(frame_size(1920, 1080), (1920, 1080 + 30));
        assert_eq!(frame_size(101, 61), (102, 76));
        let (w, h) = (64, 48);
        let f = Fields {
            w,
            h,
            ux: vec![1.0; w * h],
            uy: vec![0.0; w * h],
            conf: vec![1.0; w * h],
            gmax: (0..w * h).map(|i| (i % w) as f32 / w as f32).collect(),
            vol: vec![0.0; w * h],
            fb: None,
            valid: None,
            cod: None,
            csd: None,
        };
        let bg = Gray8 {
            w,
            h,
            data: vec![100; w * h],
        };
        let st = Style {
            bg: &bg,
            roi: &[],
            kind: Overlay::Gmax,
            lo: 0.0,
            hi: 1.0,
            alpha: 1.0,
        };
        let (fw, fh, rgb) = frame_rgb(&st, &f, None, "frame 3 t=1.50 s").unwrap();
        assert_eq!((fw, fh), frame_size(w, h));
        assert_eq!(rgb.len(), fw * fh * 3);
        assert_eq!((fw % 2, fh % 2), (0, 0));
        // the picture is the canvas picture; the strip is below it and has bright (text, bar) pixels
        let o = export::overlay_rgba(&f, Overlay::Gmax, 0.0, 1.0, 1.0);
        let pic = export::compose_rgb(&bg, Some(&o), &[]);
        assert_eq!(&rgb[..w * 3], &pic[..w * 3]);
        let strip = &rgb[h * fw * 3..];
        assert!(strip.as_chunks::<3>().0.contains(&[255, 255, 255]), "text");
        assert!(
            strip
                .as_chunks::<3>()
                .0
                .iter()
                .any(|p| *p != [24, 24, 24] && *p != [255, 255, 255]),
            "bar"
        );
        // a result of another size than the first frame is refused
        let small = Gray8 {
            w: 8,
            h: 8,
            data: vec![0; 64],
        };
        let st2 = Style { bg: &small, ..st };
        assert!(frame_rgb(&st2, &f, None, "x").is_err());
    }

    #[test]
    fn one_range_covers_every_frame() {
        let r = common_range(Overlay::Gmax, [(0.0, 0.2), (0.0, 0.5), (0.0, 0.1)]);
        assert_eq!(r, (0.0, 0.5));
        let r = common_range(Overlay::Ux, [(-0.2, 0.2), (-0.7, 0.7)]);
        assert_eq!(r, (-0.7, 0.7));
    }

    fn ffprobe(args: &[&str], file: &Path) -> String {
        let out = optflow_cli::video::tool_command("ffprobe")
            .args(["-v", "error"])
            .args(args)
            .arg(file)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn a_run_becomes_an_mp4_of_the_right_size_count_and_speed() {
        if !optflow_cli::video::tools_available() {
            eprintln!("skipped: ffmpeg or ffprobe is not installed");
            return;
        }
        let enc = match detect_encoder() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipped: {e}");
                return;
            }
        };
        let base = std::env::temp_dir().join(format!("optflow-app-vid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let images = base.join("images");
        crate::smoke::write_images(&images, 5).unwrap();
        let spec = JobSpec {
            source: images.to_string_lossy().into_owned(),
            is_video: false,
            source_count: None,
            roi: vec![(20.0, 15.0), (140.0, 18.0), (142.0, 105.0), (22.0, 102.0)],
            start: 0,
            stop: None,
            step: 1,
            settings: Settings::preset(Preset::Recommended),
            out: base.join("run"),
            autosave: None,
            resume: false,
            fps: 4.0,
        };
        let mut last = None;
        run_job(&spec, &AtomicBool::new(false), &mut |m| {
            if matches!(m, Msg::Finished | Msg::Failed(_) | Msg::Cancelled) {
                last = Some(m);
            }
        });
        assert!(matches!(last, Some(Msg::Finished)), "{last:?}");
        let ks = crate::fields::list_frames(&spec.out);
        assert!(ks.len() >= 3);
        let bg = crate::source::Source::open(&images)
            .unwrap()
            .frame(0)
            .unwrap();
        let t = Timing {
            start: 0,
            step: 1,
            fps: 4.0,
        };
        let dest = base.join("out.mp4");
        let mut stages = vec![];
        let done = export_video(
            &spec.out,
            &dest,
            &ks,
            &t,
            &bg,
            &spec.roi,
            Overlay::Gmax,
            None,
            0.7,
            2.5,
            &enc,
            &mut |s, i, n| {
                stages.push((s, i, n));
                true
            },
        )
        .unwrap();
        assert!(done && dest.is_file());
        assert!(stages.contains(&(Stage::Range, 0, ks.len())));
        assert_eq!(stages.last(), Some(&(Stage::Encode, ks.len(), ks.len())));
        let (fw, fh) = frame_size(bg.w, bg.h);
        let size = ffprobe(
            &[
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
            ],
            &dest,
        );
        assert_eq!(size, format!("{fw},{fh}"));
        let fps = ffprobe(
            &[
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
            ],
            &dest,
        );
        assert_eq!(fps, "5/2", "{fps}");
        let frames = ffprobe(
            &[
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
            ],
            &dest,
        );
        assert_eq!(frames, ks.len().to_string());
        let pix = ffprobe(
            &[
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "csv=p=0",
            ],
            &dest,
        );
        assert_eq!(pix, "yuv420p");
        // a cancel leaves no file
        let gone = base.join("cancelled.mp4");
        let done = export_video(
            &spec.out,
            &gone,
            &ks,
            &t,
            &bg,
            &spec.roi,
            Overlay::Ux,
            Some((-1.0, 1.0)),
            0.7,
            5.0,
            &enc,
            &mut |s, i, _| !(s == Stage::Encode && i == 1),
        )
        .unwrap();
        assert!(!done);
        assert!(!gone.exists(), "the partial file is deleted");
        let _ = std::fs::remove_dir_all(&base);
    }
}

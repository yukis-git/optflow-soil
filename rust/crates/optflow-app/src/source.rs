//! The input: a video file or a folder of images, and its first frame for the ROI drawing.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use optflow_cli::{io, video};

use crate::messages::FFMPEG_MISSING;

/// An 8-bit grey picture.
pub struct Gray8 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

fn to_u8(g: io::Gray) -> Gray8 {
    Gray8 {
        w: g.w,
        h: g.h,
        data: g
            .data
            .iter()
            .map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect(),
    }
}

#[derive(Clone, Debug)]
pub struct Source {
    /// What the CLI gets as `--frames`: the video file or the image folder.
    pub path: String,
    pub is_video: bool,
    /// Frames in the source, if known.
    pub count: Option<usize>,
    /// Frames per second of a video; 1 for images (a time axis in frames until the user edits it).
    pub fps: f64,
    pub w: usize,
    pub h: usize,
    /// A video of constant frame rate that starts at time 0: its frames can be found by time (see
    /// [`Source::frame_preview`]). `None` for images and for any other video.
    pub seekable: Option<f64>,
}

fn as_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow!("このパスは扱えない文字を含んでいます: {}", p.display()))
}

impl Source {
    /// A video file, or an image folder.
    pub fn open(path: &Path) -> Result<Source> {
        let s = as_str(path)?;
        if path.is_file() && io::is_video(s) {
            if !video::tools_available() {
                bail!("{FFMPEG_MISSING}");
            }
            let m = video::probe_meta(path)?;
            Ok(Source {
                path: s.into(),
                is_video: true,
                count: m.frames,
                fps: m.fps,
                w: m.w,
                h: m.h,
                seekable: video::constant_rate(path),
            })
        } else {
            let files = io::list_images(s)?;
            let first = io::read_image(Path::new(&files[0]))?;
            Ok(Source {
                path: s.into(),
                is_video: false,
                count: Some(files.len()),
                fps: 1.0,
                w: first.w,
                h: first.h,
                seekable: None,
            })
        }
    }

    /// Several selected image files: their folder, and the range `start..stop` that covers them.
    pub fn from_files(files: &[PathBuf]) -> Result<(Source, i64, i64)> {
        if files.len() < 2 {
            bail!("画像を2枚以上選んでください。");
        }
        let dir = files[0]
            .parent()
            .ok_or_else(|| anyhow!("フォルダを特定できません。"))?;
        if files.iter().any(|f| f.parent() != Some(dir)) {
            bail!("選んだ画像は同じフォルダにある必要があります。");
        }
        let src = Source::open(dir)?;
        let all = io::list_images(&src.path)?;
        let idx: Vec<usize> = files
            .iter()
            .filter_map(|f| {
                let name = f.file_name()?.to_str()?;
                all.iter()
                    .position(|p| optflow_cli::glob::basename(p) == name)
            })
            .collect();
        let (Some(&lo), Some(&hi)) = (idx.iter().min(), idx.iter().max()) else {
            bail!("選んだ画像を読み込めませんでした。");
        };
        Ok((src, lo as i64, hi as i64 + 1))
    }

    /// Frame `n` for the slider preview: a video of constant frame rate is read by an input seek
    /// (a fraction of a second), any other source as [`Source::frame`]. Never for what a run reads
    /// or for the frame the ROI is drawn on.
    pub fn frame_preview(&self, n: i64) -> Result<Gray8> {
        match self.seekable {
            Some(fps) if self.is_video => Ok(to_u8(video::frame_at(
                Path::new(&self.path),
                n.max(0),
                fps,
            )?)),
            _ => self.frame(n),
        }
    }

    /// Frame number `start` of the source, for the ROI drawing.
    pub fn frame(&self, start: i64) -> Result<Gray8> {
        let start = start.max(0);
        if self.is_video {
            let mut it = video::iter_video(Path::new(&self.path), start, Some(start + 1), 1)?;
            let g = it
                .next()
                .ok_or_else(|| anyhow!("no frames to track in {}", self.path))??;
            Ok(to_u8(g))
        } else {
            let files = io::list_images(&self.path)?;
            let i = (start as usize).min(files.len() - 1);
            Ok(to_u8(io::read_image(Path::new(&files[i]))?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("optflow-app-src-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_images(dir: &Path, n: usize) {
        for i in 0..n {
            let im = image::GrayImage::from_fn(16, 12, |x, y| {
                image::Luma([(x * 10 + y + i as u32) as u8])
            });
            im.save(dir.join(format!("f{i:03}.png"))).unwrap();
        }
    }

    #[test]
    fn opens_a_folder_with_japanese_name_and_spaces() {
        let d = scratch("a").join("テスト 画像");
        std::fs::create_dir_all(&d).unwrap();
        write_images(&d, 4);
        let s = Source::open(&d).unwrap();
        assert_eq!((s.count, s.w, s.h, s.is_video), (Some(4), 16, 12, false));
        let f = s.frame(2).unwrap();
        assert_eq!((f.w, f.h), (16, 12));
        assert_eq!(f.data[0], 2);
        let _ = std::fs::remove_dir_all(d.parent().unwrap());
    }

    /// The frame the slider shows is the frame a run reads: the one found by number gives the
    /// pixels of decoding the video from its start.
    #[test]
    fn a_video_frame_found_by_number_is_the_frame_of_sequential_decoding() {
        if !video::tools_available() {
            eprintln!("skipped: no ffmpeg here");
            return;
        }
        let d = scratch("video");
        write_images(&d, 40);
        let avi = d.join("clip.avi");
        let made = std::process::Command::new(video::tool_path("ffmpeg"))
            .args(["-v", "error", "-y", "-framerate", "5", "-i"])
            .arg(d.join("f%03d.png"))
            .args(["-c:v", "ffv1", "-pix_fmt", "gray"])
            .arg(&avi)
            .status()
            .unwrap();
        assert!(made.success());
        let s = Source::open(&avi).unwrap();
        assert_eq!((s.is_video, s.w, s.h), (true, 16, 12));
        let all: Vec<Gray8> = video::iter_video(&avi, 0, None, 1)
            .unwrap()
            .map(|g| to_u8(g.unwrap()))
            .collect();
        assert_eq!(all.len(), 40);
        for n in [0usize, 1, 4, 39] {
            let f = s.frame(n as i64).unwrap();
            assert_eq!((f.w, f.h), (16, 12));
            assert!(f.data == all[n].data, "frame {n}");
        }
        // the slider's time seek gives the pixels of sequential decoding for every frame
        assert_eq!(s.seekable, Some(5.0), "a constant-rate video is seekable");
        for (n, want) in all.iter().enumerate() {
            let f = s.frame_preview(n as i64).unwrap();
            assert!(f.data == want.data, "seek, frame {n}");
        }
        // a variable-rate or late-starting stream is not seekable: it decodes from the start
        let mut other = s.clone();
        other.seekable = None;
        assert!(other.frame_preview(4).unwrap().data == all[4].data);
        // the frames differ from each other, so the check above can tell them apart
        assert!(all[0].data != all[4].data);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// By hand on a real video: `OPTFLOW_SEEK_VIDEO=/path/video.mp4 cargo test --release -p
    /// optflow-app --lib real_video -- --ignored --nocapture`. The seek gives the pixels of
    /// sequential decoding at the frames in `OPTFLOW_SEEK_FRAMES` (comma separated), with times.
    #[test]
    #[ignore]
    fn real_video_seek_matches_sequential_decoding() {
        let Some(path) = std::env::var_os("OPTFLOW_SEEK_VIDEO") else {
            eprintln!("skipped: OPTFLOW_SEEK_VIDEO is not set");
            return;
        };
        let frames: Vec<i64> = std::env::var("OPTFLOW_SEEK_FRAMES")
            .unwrap_or_else(|_| "0,1,2999,6000".into())
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let s = Source::open(Path::new(&path)).unwrap();
        eprintln!("seekable: {:?}, frames {:?}", s.seekable, s.count);
        assert!(s.seekable.is_some());
        let t = std::time::Instant::now();
        let mut want = std::collections::BTreeMap::new();
        let last = *frames.iter().max().unwrap();
        for (i, g) in video::iter_video(Path::new(&path), 0, Some(last + 1), 1)
            .unwrap()
            .enumerate()
        {
            if frames.contains(&(i as i64)) {
                want.insert(i as i64, to_u8(g.unwrap()));
            }
        }
        eprintln!(
            "sequential decoding of 0..={last}: {:.2} s",
            t.elapsed().as_secs_f64()
        );
        for n in frames {
            let t = std::time::Instant::now();
            let f = s.frame_preview(n).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let same = f.data == want[&n].data;
            eprintln!("frame {n}: seek {secs:.2} s, identical to sequential: {same}");
            assert!(same, "frame {n}");
        }
    }

    #[test]
    fn selected_files_become_a_range() {
        let d = scratch("b");
        write_images(&d, 6);
        let sel = vec![d.join("f001.png"), d.join("f004.png")];
        let (s, lo, hi) = Source::from_files(&sel).unwrap();
        assert_eq!((lo, hi, s.count), (1, 5, Some(6)));
        assert!(Source::from_files(&sel[..1]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn empty_folder_is_an_error() {
        let d = scratch("c");
        assert!(Source::open(&d).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}

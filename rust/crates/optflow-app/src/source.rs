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

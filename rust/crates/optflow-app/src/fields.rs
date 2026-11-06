//! Reading the per-frame result maps (`frame_NNNNN/*.npy`) back from the output folder. One frame
//! is loaded at a time, on demand.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

/// The folder of frame `k` inside a run folder.
pub fn frame_dir(run: &Path, k: usize) -> PathBuf {
    run.join(format!("frame_{k:05}"))
}

/// The frame numbers found in a run folder, ascending.
pub fn list_frames(run: &Path) -> Vec<usize> {
    let mut ks: Vec<usize> = std::fs::read_dir(run)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            name.strip_prefix("frame_")?.parse::<usize>().ok()
        })
        .collect();
    ks.sort_unstable();
    ks
}

/// The maps of one frame (all full-size, row-major; invalid pixels have a NaN `ux`).
pub struct Fields {
    pub w: usize,
    pub h: usize,
    pub ux: Vec<f32>,
    pub uy: Vec<f32>,
    pub conf: Vec<f32>,
    pub gmax: Vec<f32>,
    pub vol: Vec<f32>,
    /// Hybrid runs: consecutive frames each pixel was carried by the adjacent leg.
    pub fb: Option<Vec<i32>>,
    /// Crack runs (`crack_model = xdic`): opening and sliding displacement maps (NaN away from
    /// a crack line).
    pub cod: Option<Vec<f32>>,
    pub csd: Option<Vec<f32>>,
}

impl Fields {
    pub fn is_valid(&self, i: usize) -> bool {
        self.ux[i].is_finite()
    }

    pub fn load(run: &Path, k: usize) -> Result<Fields> {
        let dir = frame_dir(run, k);
        let (h, w, ux) = read_f32(&dir.join("ux.npy"))?;
        let get = |name: &str| -> Result<Vec<f32>> {
            let (h2, w2, d) = read_f32(&dir.join(name))?;
            if (h2, w2) != (h, w) {
                bail!("{name} has a different shape in {}", dir.display());
            }
            Ok(d)
        };
        let fb_path = dir.join("fb.npy");
        let fb = if fb_path.exists() {
            Some(read_i32(&fb_path)?.2)
        } else {
            None
        };
        let opt = |name: &str| -> Result<Option<Vec<f32>>> {
            if dir.join(name).exists() {
                Ok(Some(get(name)?))
            } else {
                Ok(None)
            }
        };
        Ok(Fields {
            w,
            h,
            cod: opt("cod.npy")?,
            csd: opt("csd.npy")?,
            uy: get("uy.npy")?,
            conf: get("conf.npy")?,
            gmax: get("gmax.npy")?,
            vol: get("vol.npy")?,
            ux,
            fb,
        })
    }
}

/// The `(h, w)` and data start of a 2-D npy file, checking the dtype.
fn parse(bytes: &[u8], descr: &str) -> Result<(usize, usize, usize)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let (hl, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => bail!("unsupported npy version {v}"),
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + hl)
            .ok_or_else(|| anyhow!("truncated npy"))?,
    )?;
    if !header.contains(descr) || header.contains("'fortran_order': True") {
        bail!("unexpected npy header {header:?} (wanted {descr})");
    }
    let shape = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| anyhow!("no shape in npy header"))?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;
    let [h, w] = dims[..] else {
        bail!("expected a 2-D array, got shape ({shape})");
    };
    if bytes.len() < start + hl + 4 * h * w {
        bail!("npy data is truncated");
    }
    Ok((h, w, start + hl))
}

fn read_f32(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let (h, w, at) = parse(&bytes, "<f4").with_context(|| path.display().to_string())?;
    let data = bytes[at..at + 4 * h * w]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok((h, w, data))
}

fn read_i32(path: &Path) -> Result<(usize, usize, Vec<i32>)> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let (h, w, at) = parse(&bytes, "<i4").with_context(|| path.display().to_string())?;
    let data = bytes[at..at + 4 * h * w]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    Ok((h, w, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use optflow_core::image::Image;

    #[test]
    fn reads_what_the_cli_writes() {
        let dir = std::env::temp_dir().join(format!("optflow-app-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let im = Image::from_vec(3, 2, vec![1.0f32, 2.0, f32::NAN, 4.0, 5.0, 6.0]);
        optflow_cli::io::write_npy_f32(&dir.join("a.npy"), &im).unwrap();
        optflow_cli::io::write_npy_i32(
            &dir.join("b.npy"),
            &Image::from_vec(3, 2, vec![0.0f32, 1.0, 2.0, 3.0, 4.0, -5.0]),
        )
        .unwrap();
        let (h, w, d) = read_f32(&dir.join("a.npy")).unwrap();
        assert_eq!((h, w), (2, 3));
        assert_eq!(d[0], 1.0);
        assert!(d[2].is_nan());
        assert_eq!(
            read_i32(&dir.join("b.npy")).unwrap().2,
            vec![0, 1, 2, 3, 4, -5]
        );
        assert!(read_i32(&dir.join("a.npy")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lists_frame_folders_in_order() {
        let dir = std::env::temp_dir().join(format!("optflow-app-ls-{}", std::process::id()));
        for k in [10, 2, 1] {
            std::fs::create_dir_all(frame_dir(&dir, k)).unwrap();
        }
        std::fs::create_dir_all(dir.join("other")).unwrap();
        assert_eq!(list_frames(&dir), vec![1, 2, 10]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

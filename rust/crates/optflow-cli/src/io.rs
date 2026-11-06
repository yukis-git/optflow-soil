//! Image, ROI and result I/O: the Rust side of `python/optflow_ref/io.py`.
//!
//! Images become grey levels in `[0, 1]` computed in `f64` exactly as `to_gray` does (8-bit
//! `/255`, 16-bit `/65535`, colour through `0.299 R + 0.587 G + 0.114 B`) and are stored as `f32`
//! after the optional box downscale. Which Pillow mode a file would open in decides some details
//! (a 16-bit RGB PNG is read by Pillow as 8-bit, a 16-bit RGB TIFF by tifffile in full), so the
//! container matters here too.

use std::borrow::Cow;
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use image::{
    DynamicImage, ExtendedColorType, ImageDecoder, ImageError, ImageFormat, ImageReader, Limits,
};
use optflow_core::image::Image;
use optflow_core::npy;
use optflow_core::track::TrackResult;

use crate::glob;
use crate::polygon::MAX_COORD;

pub const VIDEO_EXT: [&str; 5] = [".mp4", ".mov", ".avi", ".mkv", ".m4v"];
const IMAGE_EXT: [&str; 6] = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"];

/// Grey levels in `[0, 1]` in `f64`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f64>,
}

impl Gray {
    /// The `f32` image the tracker works on.
    pub fn into_f32(self) -> Image<f32> {
        let data = self.data.iter().map(|&v| v as f32).collect();
        Image::from_vec(self.w, self.h, data)
    }
}

/// `0.299 R + 0.587 G + 0.114 B`, evaluated left to right as numpy does.
pub fn luminance(r: f64, g: f64, b: f64) -> f64 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// `v / 255.0` for every byte value.
pub fn lut8() -> [f64; 256] {
    std::array::from_fn(|v| v as f64 / 255.0)
}

fn lower_ext_is(path: &str, exts: &[&str]) -> bool {
    let lower = path.to_lowercase();
    exts.iter().any(|e| lower.ends_with(e))
}

/// `os.path.splitext(source)[1].lower() in VIDEO_EXT`
pub fn is_video(source: &str) -> bool {
    is_video_with(source, glob::SEPARATORS)
}

/// [`is_video`] for the path separator set `seps` (a parameter so that both sets are tested on
/// every platform).
fn is_video_with(source: &str, seps: &[char]) -> bool {
    let name = glob::basename_with(source, seps);
    let stem_free = name.trim_start_matches('.');
    match stem_free.rfind('.') {
        Some(i) => VIDEO_EXT.contains(&stem_free[i..].to_lowercase().as_str()),
        None => false,
    }
}

/// The memory one image may take while it is decoded: 2 GiB. A 26 Mpx RGB16 frame needs about
/// 160 MB, so real data is far below it; a header that claims more (a decompression bomb) is an
/// error instead of an allocation that aborts the process.
const MAX_DECODE_BYTES: u64 = 2 << 30;

/// A decoding failure as one `error:` line. The formats and pixel types the decoders lack are
/// called unsupported (the Python reader takes more TIFF pixel types, e.g. grey float32), and an
/// exceeded memory limit is called too large.
fn decode_error(shown: &std::path::Display, e: &ImageError) -> anyhow::Error {
    match e {
        ImageError::Unsupported(_) => anyhow!("{shown}: unsupported pixel type: {e}"),
        ImageError::Limits(_) => anyhow!("{shown}: image too large to read: {e}"),
        _ => anyhow!("cannot decode image {shown}: {e}"),
    }
}

/// Reads an image as grey levels. Pillow's modes decide the arithmetic: 16-bit grey is `/65535`,
/// everything 8-bit `/255`, colour (alpha dropped) goes through the luminance formula, and modes
/// Pillow converts to RGB (1-bit, palette, grey + alpha) go through it as well. TIFF files go
/// through tifffile in Python: 16-bit colour keeps all 16 bits and float images are not scaled.
///
/// Two differences from Python, both an error line: a TIFF pixel type the decoder has no type for
/// (a grey float32 image, which tifffile reads) is "unsupported pixel type", and an image that
/// would need more than 2 GiB to decode is "too large to read".
pub fn read_image(path: &Path) -> Result<Gray> {
    let shown = path.display();
    let name = path.to_string_lossy();
    let is_tiff = lower_ext_is(&name, &[".tif", ".tiff"]);
    let file = fs::File::open(path).map_err(|e| anyhow!("cannot open image {shown}: {e}"))?;
    let mut reader = ImageReader::new(BufReader::new(file));
    if is_tiff {
        reader.set_format(ImageFormat::Tiff);
    } else {
        reader = reader
            .with_guessed_format()
            .map_err(|e| anyhow!("cannot read image {shown}: {e}"))?;
    }
    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    let format = reader
        .format()
        .ok_or_else(|| anyhow!("cannot identify image file {shown}"))?;
    let decoder = reader
        .into_decoder()
        .map_err(|e| decode_error(&shown, &e))?;
    // the limit above reaches the decoders' working memory, not the output buffer that
    // `from_decoder` allocates from the size in the header: check that one here
    let bytes = decoder.total_bytes();
    if bytes > MAX_DECODE_BYTES {
        let (w, h) = decoder.dimensions();
        bail!(
            "{shown}: image too large to read: {w}x{h} pixels need {bytes} bytes, \
             the limit is {MAX_DECODE_BYTES}"
        );
    }
    let original = decoder.original_color_type();
    let img = DynamicImage::from_decoder(decoder).map_err(|e| decode_error(&shown, &e))?;
    let one_bit = match format {
        ImageFormat::Png => png_is_one_bit_grey(path),
        _ => original == ExtendedColorType::L1,
    };
    gray_from(img, format, one_bit).map_err(|m| anyhow!("{shown}: {m}"))
}

/// Whether a PNG is 1-bit grey scale (Pillow mode "1"); the decoder reports it as 8-bit grey.
fn png_is_one_bit_grey(path: &Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 26];
    fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .is_ok_and(|()| head.starts_with(b"\x89PNG\r\n\x1a\n") && head[24] == 1 && head[25] == 0)
}

/// The grey levels of a decoded image; see [`read_image`].
fn gray_from(img: DynamicImage, format: ImageFormat, one_bit: bool) -> Result<Gray, String> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let png = format == ImageFormat::Png;
    let tiff = format == ImageFormat::Tiff;
    let l8 = lut8();
    // grey replicated into R, G, B and sent through the luminance formula (Pillow's convert("RGB"))
    let via_rgb = |g: f64| luminance(g, g, g);
    let data: Vec<f64> = match img {
        DynamicImage::ImageLuma8(b) => {
            if one_bit {
                if tiff {
                    return Err("unsupported pixel type bool".into());
                }
                b.as_raw()
                    .iter()
                    .map(|&v| via_rgb(l8[v as usize]))
                    .collect()
            } else {
                b.as_raw().iter().map(|&v| l8[v as usize]).collect()
            }
        }
        DynamicImage::ImageLuma16(b) => b.as_raw().iter().map(|&v| v as f64 / 65535.0).collect(),
        DynamicImage::ImageLumaA8(b) => {
            if tiff {
                return Err("unsupported image shape: grey + alpha TIFF".into());
            }
            b.as_raw()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| via_rgb(l8[p[0] as usize]))
                .collect()
        }
        DynamicImage::ImageLumaA16(b) => {
            if tiff {
                return Err("unsupported image shape: grey + alpha TIFF".into());
            }
            // Pillow reads 16-bit grey + alpha PNG as 8-bit "LA" (high byte)
            b.as_raw()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| via_rgb(l8[(p[0] >> 8) as usize]))
                .collect()
        }
        DynamicImage::ImageRgb8(b) => b
            .as_raw()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| luminance(l8[p[0] as usize], l8[p[1] as usize], l8[p[2] as usize]))
            .collect(),
        DynamicImage::ImageRgba8(b) => b
            .as_raw()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| luminance(l8[p[0] as usize], l8[p[1] as usize], l8[p[2] as usize]))
            .collect(),
        DynamicImage::ImageRgb16(b) => {
            let scale = |v: u16| {
                if png {
                    l8[(v >> 8) as usize]
                } else {
                    v as f64 / 65535.0
                }
            };
            b.as_raw()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| luminance(scale(p[0]), scale(p[1]), scale(p[2])))
                .collect()
        }
        DynamicImage::ImageRgba16(b) => {
            let scale = |v: u16| {
                if png {
                    l8[(v >> 8) as usize]
                } else {
                    v as f64 / 65535.0
                }
            };
            b.as_raw()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| luminance(scale(p[0]), scale(p[1]), scale(p[2])))
                .collect()
        }
        DynamicImage::ImageRgb32F(b) => b
            .as_raw()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| luminance(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect(),
        DynamicImage::ImageRgba32F(b) => b
            .as_raw()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| luminance(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect(),
        other => return Err(format!("unsupported pixel type {:?}", other.color())),
    };
    Ok(Gray { w, h, data })
}

/// Box-average downscale by an integer factor >= 1 (crops the remainder), `downscale` of io.py.
pub fn downscale(img: Gray, factor: i64) -> Result<Gray> {
    if factor < 1 {
        bail!("scale must be >= 1, got {factor}");
    }
    if factor == 1 {
        return Ok(img);
    }
    let f = usize::try_from(factor).unwrap_or(usize::MAX);
    let (ow, oh) = (img.w / f, img.h / f);
    let norm = (f as f64) * (f as f64);
    let mut data = Vec::with_capacity(ow * oh);
    for j in 0..oh {
        for i in 0..ow {
            let mut sum = 0.0;
            for dy in 0..f {
                let row = (j * f + dy) * img.w + i * f;
                sum += img.data[row..row + f].iter().sum::<f64>();
            }
            data.push(sum / norm);
        }
    }
    Ok(Gray { w: ow, h: oh, data })
}

/// The glob pattern of every entry of the directory `dir`: its trailing separators dropped, then
/// joined with `*` (`"a/"` and `"a"` give `"a/*"`). A root that is separators only keeps its first
/// one, so `"/"` gives `"/*"` and not `"*"` (which would be the current directory).
fn dir_pattern(dir: &str) -> String {
    let trimmed = dir.trim_end_matches(glob::SEPARATORS);
    let base = if trimmed.is_empty() {
        dir.get(..1).unwrap_or("")
    } else {
        trimmed
    };
    glob::join(base, "*")
}

/// `list_images`: the image files of a directory or glob pattern in plain string order.
pub fn list_images(source: &str) -> Result<Vec<String>> {
    let mut paths = if Path::new(source).is_dir() {
        glob::glob(&dir_pattern(source))
            .into_iter()
            .filter(|p| lower_ext_is(p, &IMAGE_EXT))
            .collect()
    } else {
        glob::glob(source)
    };
    if paths.is_empty() {
        bail!("no images found for {source}");
    }
    paths.sort();
    Ok(paths)
}

/// `start`, `start + step`, ... below `stop` as Python's `list[start:stop:step]` (`step >= 1`,
/// `start >= 0`; a negative `stop` counts from the end).
pub fn slice_indices(len: usize, start: i64, stop: Option<i64>, step: i64) -> Vec<usize> {
    let len_i = len as i64;
    let stop = match stop {
        None => len_i,
        Some(s) if s < 0 => (s + len_i).max(0),
        Some(s) => s.min(len_i),
    };
    let mut out = Vec::new();
    let mut i = start.min(len_i);
    while i < stop {
        out.push(i as usize);
        i = match i.checked_add(step) {
            Some(n) => n,
            None => break,
        };
    }
    out
}

/// `_check_range`
pub fn check_range(start: i64, step: i64) -> Result<()> {
    if step < 1 {
        bail!("step must be >= 1, got {step}");
    }
    if start < 0 {
        bail!("start must be >= 0, got {start}");
    }
    Ok(())
}

/// Vertices of the `"polygon"` entry of an ROI file, in full-resolution pixels. A coordinate
/// beyond `MAX_COORD` (2^30) in magnitude is an error: the polygon filler is not meant for such
/// values (and no image has such a size).
pub fn load_roi(path: &Path) -> Result<Vec<(f64, f64)>> {
    let shown = path.display();
    let text =
        fs::read_to_string(path).map_err(|e| anyhow!("cannot read ROI file {shown}: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| anyhow!("invalid ROI file {shown}: {e}"))?;
    let list = json
        .get("polygon")
        .and_then(|p| p.as_array())
        .ok_or_else(|| {
            anyhow!("invalid ROI file {shown}: expected {{\"polygon\": [[x, y], ...]}}")
        })?;
    list.iter()
        .map(|v| {
            let xy = v.as_array().filter(|a| a.len() == 2);
            match xy.map(|a| (a[0].as_f64(), a[1].as_f64())) {
                // the comparisons are false for NaN, so that one is rejected as well
                Some((Some(x), Some(y))) if x.abs() <= MAX_COORD && y.abs() <= MAX_COORD => {
                    Ok((x, y))
                }
                Some((Some(_), Some(_))) => Err(anyhow!(
                    "invalid ROI file {shown}: vertex {v} is out of range \
                     (coordinates must lie within +-2^30)"
                )),
                _ => Err(anyhow!(
                    "invalid ROI file {shown}: each vertex must be [x, y], got {v}"
                )),
            }
        })
        .collect()
}

/// A `|u1` npy file (format 1.0, header padded to 64 bytes as `numpy.save` writes it).
///
/// The header length is that of `npy::header_len` for the dict as written. numpy also reserves
/// room in the header for the first axis to grow (up to 21 digits, so a header of an array with
/// 15 or more dimensions comes out longer than its dict needs; `npy::write_f32` does this); this
/// function does not, so its bytes equal numpy's only for shapes of fewer than 15 dimensions. The
/// caller (`save_result`) passes 2-D shapes.
pub fn npy_u8_bytes(shape: &[usize], data: &[u8]) -> Vec<u8> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut out = npy_u8_header(shape);
    out.extend_from_slice(data);
    out
}

/// The header of [`npy_u8_bytes`]: what precedes the bytes of the array.
fn npy_u8_header(shape: &[usize]) -> Vec<u8> {
    let shape_text = match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let dict = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': {shape_text}, }}");
    let total = npy::header_len(dict.len());
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&((total - 10) as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(total - 1, b' ');
    out.push(b'\n');
    out
}

/// A `|b1` npy file (`numpy.save` of a bool array): the header of [`npy_u8_bytes`] with the dtype
/// `|b1`, then one byte per value.
pub fn npy_bool_bytes(shape: &[usize], data: &[bool]) -> Vec<u8> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut out = npy_u8_header(shape);
    let at = out
        .windows(5)
        .position(|w| w == b"'|u1'")
        .expect("the header names its dtype");
    out[at + 3] = b'b';
    out.extend(data.iter().map(|&b| u8::from(b)));
    out
}

/// The crack outputs of `crack_model = "xdic"` (the `crack`, `crack_nx`, `crack_ny`, `cod` and
/// `csd` maps of the reference's `save_result`, written when the result has a crack line) and
/// `crack_lines.json` (Rust only): the lines, the bands and rejected lines, and the COD / CSD
/// samples along each line. `place` puts the maps into the full image, `offset` is the crop's
/// `(x0, y0)`: the line coordinates are those of the image tracked, plus the offset.
pub fn save_crack(
    dir: &Path,
    crack: &optflow_core::xdic::CrackOut,
    place: &Place,
    offset: (f64, f64),
) -> Result<()> {
    use serde_json::json;
    if let Some(m) = &crack.maps {
        let as_f32 = Image::from_vec(
            m.mask.w,
            m.mask.h,
            m.mask
                .data
                .iter()
                .map(|&b| f32::from(u8::from(b)))
                .collect(),
        );
        let placed = place(&as_f32, 0.0);
        let flags: Vec<bool> = placed.data.iter().map(|&v| v > 0.5).collect();
        write_file(
            &dir.join("crack.npy"),
            &npy_bool_bytes(&[placed.h, placed.w], &flags),
        )?;
        for (name, im) in [
            ("crack_nx", &m.nx),
            ("crack_ny", &m.ny),
            ("cod", &m.cod),
            ("csd", &m.csd),
        ] {
            write_npy_f32(&dir.join(format!("{name}.npy")), &place(im, f32::NAN))?;
        }
    }
    let pts = |p: &[[f64; 2]]| -> Vec<[f64; 2]> {
        p.iter()
            .map(|q| [q[0] + offset.0, q[1] + offset.1])
            .collect()
    };
    let l = &crack.lines;
    let doc = json!({
        "lines": l.lines.iter().map(|p| pts(p)).collect::<Vec<_>>(),
        "bands": l.bands.iter().map(|p| pts(p)).collect::<Vec<_>>(),
        "rejected": l.rejected.iter().map(|p| pts(p)).collect::<Vec<_>>(),
        "jumps": l.jumps.iter().map(|j| json!({
            "pts": pts(&j.pts),
            "n": j.n,
            "s": j.s,
            "cod": j.cod,
            "csd": j.csd,
        })).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&doc).expect("crack lines serialise to JSON");
    write_file(&dir.join("crack_lines.json"), text.as_bytes())
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::write(path, bytes).with_context(|| format!("cannot write {}", path.display()))
}

/// Values per piece of a streamed npy file: 16 Ki values, 64 KiB of `f32`.
const PIECE: usize = 1 << 14;

/// Writes `header`, then `values` turned into bytes by `bytes` piece by piece, to `path`: the
/// file of `fs::write` of the whole buffer, without holding that buffer.
fn write_streamed<T: Copy>(
    path: &Path,
    header: &[u8],
    values: &[T],
    bytes: impl Fn(T, &mut Vec<u8>),
) -> Result<()> {
    let write = || -> std::io::Result<()> {
        let mut out = BufWriter::new(fs::File::create(path)?);
        out.write_all(header)?;
        let mut buf = Vec::with_capacity(4 * PIECE);
        for piece in values.chunks(PIECE) {
            buf.clear();
            for &v in piece {
                bytes(v, &mut buf);
            }
            out.write_all(&buf)?;
        }
        out.flush()
    };
    write().with_context(|| format!("cannot write {}", path.display()))
}

/// `im` as an `<f4` npy file (the bytes of `npy::write_f32`), written in pieces.
pub fn write_npy_f32(path: &Path, im: &Image<f32>) -> Result<()> {
    let header = npy::f32_header(&[im.h, im.w]);
    write_streamed(path, &header, &im.data, |v, buf| {
        buf.extend_from_slice(&v.to_le_bytes())
    })
}

/// `values` (a map of integers, held as `f32`, exact below 2^24) as an `<i4` npy file.
pub fn write_npy_i32(path: &Path, im: &Image<f32>) -> Result<()> {
    let mut header = npy::f32_header(&[im.h, im.w]);
    let at = header
        .windows(3)
        .position(|w| w == b"<f4")
        .expect("the header names its dtype");
    header[at..at + 3].copy_from_slice(b"<i4");
    write_streamed(path, &header, &im.data, |v, buf| {
        buf.extend_from_slice(&(v as i32).to_le_bytes())
    })
}

/// The `valid` file of `ux`: `|u1`, 1 where `ux` is finite (the bytes of `npy_u8_bytes`),
/// written in pieces.
fn write_valid(path: &Path, ux: &Image<f32>) -> Result<()> {
    let header = npy_u8_header(&[ux.h, ux.w]);
    write_streamed(path, &header, &ux.data, |v, buf| {
        buf.push(u8::from(v.is_finite()))
    })
}

/// Turns one map of a result into the map that is written: `place(map, fill)`, `fill` being the
/// value outside a crop (see `Crop::place`).
pub type Place<'p> = dyn Fn(&Image<f32>, f32) -> Cow<'_, Image<f32>> + 'p;

/// `save_result`: ux, uy (NaN = invalid), conf, the Green-Lagrange strain `Exx, Eyy, Exy`, `gmax`
/// and `vol` as `<f4`, `valid` (uint8, 1 where ux is finite) and `timing.json`.
pub fn save_result(dir: &Path, res: &TrackResult) -> Result<()> {
    save_result_placed(dir, res, false, &|im, _| Cow::Borrowed(im))
}

/// [`save_result`] with every map passed through `place` before it is written (the embedding of
/// a crop's result in the full image: NaN outside the crop, confidence 0); `valid` is that of
/// the placed `ux`. One placed map at a time is held, and each file is written in pieces. `gpu`
/// adds `"device": "gpu"` to `timing.json` (after `coarse`); a CPU run's file is unchanged.
pub fn save_result_placed(dir: &Path, res: &TrackResult, gpu: bool, place: &Place) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let s = &res.strain;
    let fields: [(&str, &Image<f32>, f32); 8] = [
        ("ux", &res.ux, f32::NAN),
        ("uy", &res.uy, f32::NAN),
        ("conf", &res.conf, 0.0),
        ("Exx", &s.exx_gl, f32::NAN),
        ("Eyy", &s.eyy_gl, f32::NAN),
        ("Exy", &s.exy_gl, f32::NAN),
        ("gmax", &s.gmax, f32::NAN),
        ("vol", &s.vol, f32::NAN),
    ];
    for (name, im, fill) in fields {
        let placed = place(im, fill);
        write_npy_f32(&dir.join(format!("{name}.npy")), &placed)?;
        if name == "ux" {
            write_valid(&dir.join("valid.npy"), &placed)?;
        }
    }
    let mut timing = serde_json::to_string_pretty(&res.timing).expect("timing serialises to JSON");
    if gpu {
        // after the first key, `coarse`; the pretty form puts every key on its own line
        let at = timing
            .find('\n')
            .and_then(|a| timing[a + 1..].find('\n').map(|b| a + 1 + b));
        let at = at.expect("timing has several keys");
        timing.insert_str(at + 1, "  \"device\": \"gpu\",\n");
    }
    write_file(&dir.join("timing.json"), timing.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_npy_files_have_the_bytes_of_the_in_memory_writers() {
        let dir = crate::unit_scratch("streamed_npy");
        // more values than a piece and not a multiple of it, with NaN, infinities and -0
        let (w, h) = (131usize, 127usize);
        assert!(w * h > PIECE && w * h % PIECE != 0);
        let data = (0..w * h)
            .map(|i| match i % 97 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => -0.0,
                3 => f32::NEG_INFINITY,
                _ => (i as f32 * 0.37).sin() * 1e3,
            })
            .collect();
        let big = Image::from_vec(w, h, data);
        for (name, im) in [
            ("big", &big),
            ("no_rows", &Image::new(9, 0)),
            ("no_columns", &Image::new(0, 4)),
            ("one_piece", &Image::new(PIECE, 1)),
        ] {
            let path = dir.join(format!("{name}.npy"));
            write_npy_f32(&path, im).unwrap();
            let want = npy::write_f32(&[im.h, im.w], &im.data);
            assert!(fs::read(&path).unwrap() == want, "{name}: f32 bytes differ");
            let path = dir.join(format!("{name}_valid.npy"));
            write_valid(&path, im).unwrap();
            let flags: Vec<u8> = im.data.iter().map(|v| u8::from(v.is_finite())).collect();
            let want = npy_u8_bytes(&[im.h, im.w], &flags);
            assert!(
                fs::read(&path).unwrap() == want,
                "{name}: valid bytes differ"
            );
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    fn gray(w: usize, h: usize, f: impl Fn(usize, usize) -> f64) -> Gray {
        Gray {
            w,
            h,
            data: (0..w * h).map(|i| f(i % w, i / w)).collect(),
        }
    }

    #[test]
    fn downscale_is_a_box_average_that_crops_the_remainder() {
        let g = gray(4, 4, |x, y| (4 * y + x) as f64);
        let d = downscale(g, 2).unwrap();
        assert_eq!((d.w, d.h), (2, 2));
        assert_eq!(d.data, [2.5, 4.5, 10.5, 12.5]);
        let g = gray(5, 3, |x, y| (x + y) as f64);
        let d = downscale(g, 2).unwrap();
        assert_eq!((d.w, d.h, d.data.len()), (2, 1, 2));
    }

    /// The mean of a block is its sum over `factor^2` (for 2 the sum of the factor with itself
    /// would do as well, which is why this uses 3 and 4).
    #[test]
    fn downscale_divides_by_the_block_area() {
        for f in [3usize, 4] {
            let g = gray(2 * f, 3 * f, |_, _| 1.0);
            let d = downscale(g, f as i64).unwrap();
            assert_eq!((d.w, d.h), (2, 3));
            assert!(d.data.iter().all(|&v| v == 1.0), "factor {f}: {:?}", d.data);
        }
        // blocks of 3 x 3 with the values 0..9 (mean 4) and 9..18 (mean 13)
        let g = gray(6, 3, |x, y| (3 * y + x % 3 + 9 * (x / 3)) as f64);
        assert_eq!(downscale(g, 3).unwrap().data, [4.0, 13.0]);
    }

    #[test]
    fn downscale_rejects_factors_below_one() {
        for f in [0, -1, -7] {
            let e = downscale(gray(4, 4, |_, _| 0.0), f).unwrap_err();
            assert_eq!(e.to_string(), format!("scale must be >= 1, got {f}"));
        }
        assert!(downscale(gray(4, 4, |_, _| 0.0), 1).is_ok());
    }

    #[test]
    fn luminance_of_pure_red_is_the_red_weight() {
        assert_eq!(luminance(1.0, 0.0, 0.0), 0.299);
    }

    #[test]
    fn the_extension_is_that_of_the_last_component_for_the_separators_given() {
        let posix: &[char] = &['/'];
        let windows: &[char] = &['/', '\\'];
        // a backslash is part of the name on POSIX and separates on Windows
        assert!(is_video_with("dir\\clip.mp4", posix));
        assert!(is_video_with("dir\\clip.mp4", windows));
        assert!(is_video_with("weird.mp4\\.mp4", posix)); // the name is "weird.mp4\\.mp4"
        assert!(!is_video_with("weird.mp4\\.mp4", windows)); // the name is ".mp4": no extension
        assert!(!is_video_with("a.mp4\\frames", posix));
        assert!(!is_video_with("a.mp4\\frames", windows));
        assert!(is_video_with("C:/dir.d\\x.MOV", windows));
    }

    #[test]
    fn video_files_are_recognised_by_extension() {
        for s in ["a.mp4", "dir/b.MOV", "c.Avi", "x.mkv", "y.m4v", "./d/e.mp4"] {
            assert!(is_video(s), "{s}");
        }
        for s in ["a.png", "dir.mp4/frames", ".mp4", "mp4", "frames/*.png", ""] {
            assert!(!is_video(s), "{s}");
        }
    }

    #[test]
    fn slice_indices_follow_python_slicing() {
        assert_eq!(slice_indices(5, 0, None, 1), [0, 1, 2, 3, 4]);
        assert_eq!(slice_indices(5, 1, Some(4), 2), [1, 3]);
        assert_eq!(slice_indices(5, 5, None, 1), Vec::<usize>::new());
        assert_eq!(slice_indices(5, 0, Some(-1), 1), [0, 1, 2, 3]);
        assert_eq!(slice_indices(5, 0, Some(-9), 1), Vec::<usize>::new());
        assert_eq!(slice_indices(5, 2, Some(99), 3), [2]);
        // a stop of 0 is the start of the list, not "-0" (the end)
        assert_eq!(slice_indices(5, 0, Some(0), 1), Vec::<usize>::new());
        assert_eq!(slice_indices(5, 0, Some(-5), 1), Vec::<usize>::new());
        assert_eq!(slice_indices(5, 0, Some(-4), 1), [0]);
    }

    /// 16-bit colour: a TIFF is read in full (`v / 65535`, by tifffile), a PNG as Pillow reads it,
    /// as 8 bits (the high byte).
    #[test]
    fn sixteen_bit_colour_is_scaled_by_65535_for_tiff_and_by_the_high_byte_for_png() {
        let l8 = lut8();
        let px = [40000u16, 20000, 60000];
        let want_tiff = luminance(
            px[0] as f64 / 65535.0,
            px[1] as f64 / 65535.0,
            px[2] as f64 / 65535.0,
        );
        let want_png = luminance(
            l8[(px[0] >> 8) as usize],
            l8[(px[1] >> 8) as usize],
            l8[(px[2] >> 8) as usize],
        );
        assert!(want_tiff != want_png);
        let rgb =
            || DynamicImage::ImageRgb16(image::ImageBuffer::from_raw(1, 1, px.to_vec()).unwrap());
        let rgba = || {
            let raw = vec![px[0], px[1], px[2], 12345];
            DynamicImage::ImageRgba16(image::ImageBuffer::from_raw(1, 1, raw).unwrap())
        };
        for (what, img) in [("rgb", rgb()), ("rgba", rgba())] {
            let tiff = gray_from(img.clone(), ImageFormat::Tiff, false).unwrap();
            assert_eq!(tiff.data, [want_tiff], "{what} tiff");
            let png = gray_from(img, ImageFormat::Png, false).unwrap();
            assert_eq!(png.data, [want_png], "{what} png");
        }
    }

    /// A one-dimensional shape is written as numpy does, `(3,)`, with the comma.
    #[test]
    fn npy_u8_one_dimensional_shape_has_the_trailing_comma() {
        let bytes = npy_u8_bytes(&[3], &[1, 2, 3]);
        let text = String::from_utf8_lossy(&bytes[10..]);
        assert!(
            text.starts_with("{'descr': '|u1', 'fortran_order': False, 'shape': (3,), }"),
            "{text}"
        );
        let back = npy::read(&bytes).unwrap();
        assert_eq!(back.shape, [3]);
    }

    /// `save_result` writes the maps unplaced: ux, uy, conf, the strain maps, `valid` and
    /// `timing.json`.
    #[test]
    fn save_result_writes_every_map_as_it_is() {
        use optflow_core::strain::Strain;
        use optflow_core::track::Timing;
        let (w, h) = (3usize, 2usize);
        // the NaN of ux is at pixel 4 and no other map has one there: `valid` can only be ux's
        let map = |seed: f32, nan_at: usize| {
            Image::from_vec(
                w,
                h,
                (0..w * h)
                    .map(|i| {
                        if i == nan_at {
                            f32::NAN
                        } else {
                            seed + i as f32 * 0.25
                        }
                    })
                    .collect(),
            )
        };
        let res = TrackResult {
            ux: map(1.0, 4),
            uy: map(2.0, 0),
            conf: map(3.0, 1),
            strain: Strain {
                exx_gl: map(4.0, 2),
                eyy_gl: map(5.0, 3),
                exy_gl: map(6.0, 5),
                exx: map(7.0, 0),
                eyy: map(8.0, 1),
                exy: map(9.0, 2),
                gmax: map(10.0, 3),
                vol: map(11.0, 5),
            },
            raw_ux: map(12.0, 0),
            raw_uy: map(13.0, 1),
            timing: Timing {
                coarse: 0.5,
                fine: 1.5,
                strain: 2.5,
                reference: None,
                mean_conf: None,
                ..Timing::default()
            },
            fb: None,
            crack: None,
        };
        let dir = crate::unit_scratch("save_result");
        save_result(&dir, &res).unwrap();
        let maps = [
            ("ux", &res.ux),
            ("uy", &res.uy),
            ("conf", &res.conf),
            ("Exx", &res.strain.exx_gl),
            ("Eyy", &res.strain.eyy_gl),
            ("Exy", &res.strain.exy_gl),
            ("gmax", &res.strain.gmax),
            ("vol", &res.strain.vol),
        ];
        for (name, im) in maps {
            let got = fs::read(dir.join(format!("{name}.npy"))).unwrap();
            let want = npy::write_f32(&[h, w], &im.data);
            // NaN is not equal to itself, so compare the bytes
            assert!(got == want, "{name}.npy");
        }
        let flags: Vec<u8> = res
            .ux
            .data
            .iter()
            .map(|v| u8::from(v.is_finite()))
            .collect();
        let want = npy_u8_bytes(&[h, w], &flags);
        assert!(
            fs::read(dir.join("valid.npy")).unwrap() == want,
            "valid.npy"
        );
        let timing = fs::read_to_string(dir.join("timing.json")).unwrap();
        assert!(timing.contains("\"coarse\": 0.5"), "{timing}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(not(windows))]
    fn dir_pattern_keeps_the_root_and_drops_trailing_separators() {
        assert_eq!(dir_pattern("/"), "/*");
        assert_eq!(dir_pattern("//"), "/*");
        assert_eq!(dir_pattern("a"), "a/*");
        assert_eq!(dir_pattern("a/"), "a/*");
        assert_eq!(dir_pattern("a//"), "a/*");
        assert_eq!(dir_pattern("/data/frames/"), "/data/frames/*");
    }

    #[test]
    fn list_images_sorts_plain_strings_and_filters_extensions() {
        let dir = crate::unit_scratch("list_images");
        for f in [
            "IMG_9.png",
            "IMG_10.png",
            "IMG_2.PNG",
            "a.JPG",
            "b.tiff",
            "c.bmp",
            "d.txt",
            ".h.png",
        ] {
            fs::write(dir.join(f), b"").unwrap();
        }
        let d = dir.to_str().unwrap();
        let names = |src: &str| -> Vec<String> {
            list_images(src)
                .unwrap()
                .iter()
                .map(|p| glob::basename(p).to_string())
                .collect()
        };
        // plain string order: uppercase before lowercase, IMG_10 before IMG_2 before IMG_9
        let want = [
            "IMG_10.png",
            "IMG_2.PNG",
            "IMG_9.png",
            "a.JPG",
            "b.tiff",
            "c.bmp",
        ];
        assert_eq!(names(d), want);
        assert_eq!(names(&format!("{d}/")), want);
        // a glob is not filtered by extension
        assert_eq!(names(&format!("{d}/*.txt")), ["d.txt"]);
        assert_eq!(
            names(&format!("{d}/IMG_*")),
            ["IMG_10.png", "IMG_2.PNG", "IMG_9.png"]
        );
        let e = list_images(&format!("{d}/nothing_*"))
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("no images found for "), "{e}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn roi_vertices_must_be_finite_numbers_within_2_pow_30() {
        let dir = crate::unit_scratch("load_roi");
        let load = |text: &str| {
            let path = dir.join("roi.json");
            fs::write(&path, text).unwrap();
            load_roi(&path)
        };
        let ok = load(r#"{"polygon": [[-1073741824, 0], [1073741824.0, 5.5], [0, 7]]}"#).unwrap();
        assert_eq!(ok, [(-1073741824.0, 0.0), (1073741824.0, 5.5), (0.0, 7.0)]);
        for text in [
            r#"{"polygon": [[1073741825, 0]]}"#,
            r#"{"polygon": [[0, -1073741825]]}"#,
            r#"{"polygon": [[0, 0], [1e10, 3]]}"#,
            r#"{"polygon": [[0, 1e300]]}"#,
            r#"{"polygon": [[0, 123456789012345678901234567890]]}"#,
        ] {
            let e = load(text).unwrap_err().to_string();
            assert!(e.contains("is out of range"), "{text}: {e}");
            assert!(e.contains("roi.json"), "{text}: {e}");
        }
        // other malformed vertices keep their own message
        let e = load(r#"{"polygon": [[1, "a"]]}"#).unwrap_err().to_string();
        assert!(e.contains("each vertex must be [x, y]"), "{e}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn range_check_messages() {
        assert_eq!(
            check_range(0, 0).unwrap_err().to_string(),
            "step must be >= 1, got 0"
        );
        assert_eq!(
            check_range(-1, 1).unwrap_err().to_string(),
            "start must be >= 0, got -1"
        );
        assert!(check_range(0, 1).is_ok());
    }

    #[test]
    fn npy_u8_matches_numpy_bytes() {
        // np.save of np.array([[1, 0, 1], [0, 1, 0]], np.uint8)
        let mut want = b"\x93NUMPY\x01\x00v\x00".to_vec();
        want.extend_from_slice(b"{'descr': '|u1', 'fortran_order': False, 'shape': (2, 3), }");
        want.resize(127, b' ');
        want.push(b'\n');
        want.extend_from_slice(&[1, 0, 1, 0, 1, 0]);
        assert_eq!(npy_u8_bytes(&[2, 3], &[1, 0, 1, 0, 1, 0]), want);
        let back = npy::read(&want).unwrap();
        assert_eq!(back.shape, [2, 3]);
    }
}

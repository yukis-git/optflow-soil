//! Helpers of the integration tests: speckle images with a known translation, image/video
//! writers, and running the `optflow` binary.
#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use image::{ImageBuffer, Luma};
use optflow_core::npy::{self, NpyData};

/// A fresh empty directory `target/tmp/<name>` for one test.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Gaussian blobs of radius 1.5 to 4 px covering about 60 % of `w` x `h` plus a margin; the same
/// pattern for the same seed (a port of `synth.make_pattern` with this file's own random numbers).
pub struct Pattern {
    cx: Vec<f64>,
    cy: Vec<f64>,
    r: Vec<f64>,
    a: Vec<f64>,
}

pub fn make_pattern(w: usize, h: usize, seed: u64) -> Pattern {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(12345);
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let (rmin, rmax, margin) = (1.5, 4.0, 40.0);
    let rmean = 0.5 * (rmin + rmax);
    let n = (0.6 * (h as f64 + 2.0 * margin) * (w as f64 + 2.0 * margin) / (PI * rmean * rmean))
        as usize;
    let mut p = Pattern {
        cx: vec![],
        cy: vec![],
        r: vec![],
        a: vec![],
    };
    for _ in 0..n {
        p.cx.push(-margin + (w as f64 + 2.0 * margin) * rnd());
        p.cy.push(-margin + (h as f64 + 2.0 * margin) * rnd());
        p.r.push(rmin + (rmax - rmin) * rnd());
        let sign = if rnd() < 0.5 { -1.0 } else { 1.0 };
        p.a.push((0.3 + 0.7 * rnd()) * sign);
    }
    p
}

/// The pattern moved by `(tx, ty)` px, grey values in `[0, 1]`, row-major.
pub fn render(p: &Pattern, w: usize, h: usize, tx: f64, ty: f64) -> Vec<f64> {
    let mut acc = vec![0.0f64; w * h];
    for k in 0..p.cx.len() {
        let (px, py) = (p.cx[k] + tx, p.cy[k] + ty);
        let rad = 5.2 * p.r[k] + 2.0;
        let (x0, x1) = (
            ((px - rad).floor().max(0.0)) as usize,
            (((px + rad).ceil().max(0.0)) as usize).min(w),
        );
        let (y0, y1) = (
            ((py - rad).floor().max(0.0)) as usize,
            (((py + rad).ceil().max(0.0)) as usize).min(h),
        );
        let inv2r2 = 1.0 / (2.0 * p.r[k] * p.r[k]);
        for y in y0..y1 {
            for x in x0..x1 {
                let (dx, dy) = (x as f64 - px, y as f64 - py);
                acc[y * w + x] += p.a[k] * (-(dx * dx + dy * dy) * inv2r2).exp();
            }
        }
    }
    acc.iter().map(|v| 0.5 + 0.25 * v.tanh()).collect()
}

pub fn write_png16(path: &Path, w: usize, h: usize, grey: &[f64]) {
    let data: Vec<u16> = grey.iter().map(|v| (v * 65535.0).round() as u16).collect();
    ImageBuffer::<Luma<u16>, _>::from_raw(w as u32, h as u32, data)
        .unwrap()
        .save(path)
        .unwrap();
}

pub fn write_png8(path: &Path, w: usize, h: usize, grey: &[f64]) {
    let data: Vec<u8> = grey.iter().map(|v| (v * 255.0).round() as u8).collect();
    ImageBuffer::<Luma<u8>, _>::from_raw(w as u32, h as u32, data)
        .unwrap()
        .save(path)
        .unwrap();
}

/// Runs `optflow` with `args`.
pub fn optflow(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_optflow"))
        .args(args)
        .output()
        .expect("run optflow")
}

pub fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

pub fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A `<f4` / `|u1` npy file as `(shape, values as f64)`.
pub fn load_npy(path: &Path) -> (Vec<usize>, Vec<f64>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let a = npy::read(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v = match a.data {
        NpyData::F32(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U8(v) => v.into_iter().map(f64::from).collect(),
        other => panic!("{}: unexpected dtype {other:?}", path.display()),
    };
    (a.shape, v)
}

pub fn median_of_finite(v: impl Iterator<Item = f64>) -> f64 {
    let mut x: Vec<f64> = v.filter(|v| v.is_finite()).collect();
    assert!(!x.is_empty(), "no finite values");
    x.sort_by(f64::total_cmp);
    x[x.len() / 2]
}

/// Whether `ffmpeg` and `ffprobe` run here.
pub fn have_ffmpeg() -> bool {
    ["ffmpeg", "ffprobe"].iter().all(|t| {
        Command::new(t)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

/// Median of the finite values of the square `[lo, hi)` x `[lo, hi)` of a `w`-wide field.
pub fn centre_median(field: &[f64], w: usize, lo: usize, hi: usize) -> f64 {
    median_of_finite((lo..hi).flat_map(|y| field[y * w + lo..y * w + hi].iter().copied()))
}

// ---- files whose headers claim far more than their data

/// CRC-32 (the checksum of the PNG chunks).
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A PNG whose header claims `w` x `h` pixels of 16-bit RGBA and whose data is a few bytes.
pub fn bomb_png(w: u32, h: u32) -> Vec<u8> {
    let chunk = |tag: &[u8; 4], data: &[u8]| {
        let mut body = tag.to_vec();
        body.extend_from_slice(data);
        let mut out = (data.len() as u32).to_be_bytes().to_vec();
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
        out
    };
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[16, 6, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend(chunk(b"IHDR", &ihdr));
    png.extend(chunk(
        b"IDAT",
        &[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01],
    ));
    png.extend(chunk(b"IEND", &[]));
    png
}

/// A 24-bit BMP whose header claims `w` x `h` pixels and has no pixel data.
pub fn bomb_bmp(w: u32, h: u32) -> Vec<u8> {
    let mut bmp = b"BM".to_vec();
    bmp.extend_from_slice(&54u32.to_le_bytes()); // file size
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
    bmp.extend_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER
    bmp.extend_from_slice(&w.to_le_bytes());
    bmp.extend_from_slice(&h.to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&[0; 24]); // no compression, sizes, resolution, palette
    bmp
}

/// A little-endian TIFF with one 8-bit grey strip whose header claims `w` x `h` pixels.
pub fn bomb_tiff(w: u32, h: u32) -> Vec<u8> {
    let entries: [(u16, u16, u32); 9] = [
        (256, 4, w),   // ImageWidth
        (257, 4, h),   // ImageLength
        (258, 3, 8),   // BitsPerSample
        (259, 3, 1),   // Compression: none
        (262, 3, 1),   // PhotometricInterpretation: BlackIsZero
        (273, 4, 200), // StripOffsets
        (277, 3, 1),   // SamplesPerPixel
        (278, 4, h),   // RowsPerStrip
        (279, 4, 4),   // StripByteCounts
    ];
    let mut tiff = b"II*\0".to_vec();
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, kind, value) in entries {
        tiff.extend_from_slice(&tag.to_le_bytes());
        tiff.extend_from_slice(&kind.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&value.to_le_bytes());
    }
    tiff.extend_from_slice(&0u32.to_le_bytes());
    tiff.resize(204, 0);
    tiff
}

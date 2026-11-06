//! The image reader against `python/optflow_ref/io.read_image` (tifffile for TIFF, Pillow
//! otherwise) on small files made by `tests/fixtures/make_decode_fixtures.py`.

mod common;

use std::path::Path;

use common::{bomb_bmp, bomb_png, bomb_tiff};
use optflow_cli::io::read_image;
use serde_json::Value;

/// A grey float32 TIFF: tifffile (Python) reads it, the `image` crate has no such pixel type.
const GRAY_F32: &str = "gray_f32.tif";

fn dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn expected() -> Value {
    let text = std::fs::read_to_string(dir().join("decode_expected.json")).expect("fixture file");
    serde_json::from_str(&text).expect("fixture json")
}

/// Largest and mean absolute difference between the Rust grey levels and Python's, the number of
/// pixels that differ at all and the number that still differ as `f32` (what the tracker reads).
fn compare(name: &str, want: &Value) -> (f64, f64, usize, usize) {
    let g = read_image(&dir().join("images").join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(
        (g.w, g.h),
        (
            want["w"].as_u64().unwrap() as usize,
            want["h"].as_u64().unwrap() as usize
        ),
        "{name}: size"
    );
    let gray = want["gray"].as_array().unwrap();
    assert_eq!(g.data.len(), gray.len(), "{name}: length");
    let (mut max, mut sum, mut differing, mut differing_f32) = (0.0f64, 0.0f64, 0, 0);
    for (a, b) in g.data.iter().zip(gray) {
        let b = b.as_f64().unwrap();
        let d = (a - b).abs();
        max = max.max(d);
        sum += d;
        differing += usize::from(d > 0.0);
        differing_f32 += usize::from(*a as f32 != b as f32);
    }
    (max, sum / g.data.len() as f64, differing, differing_f32)
}

#[test]
fn png_tiff_and_bmp_are_bit_identical_to_python() {
    let exp = expected();
    let mut checked = 0;
    let mut bmp = Vec::new();
    let mut failures = Vec::new();
    for (name, want) in exp.as_object().unwrap() {
        if name.ends_with(".jpg") || name == GRAY_F32 {
            continue; // the JPEG decoders differ slightly; the grey float32 TIFF is not readable here
        }
        let (max, _, mut differing, differing_f32) = compare(name, want);
        // Pillow reads an 8-bit BMP with a grey palette as mode "L" (level / 255); the decoder here
        // hands over RGB pixels, which go through the luminance formula: 1 ulp off in f64 for some
        // levels, the same after the cast to f32 (the 1-bit, 24-bit and 32-bit BMP files agree
        // exactly)
        if name == "gray8.bmp" {
            differing = differing_f32;
        }
        if differing != 0 {
            failures.push(format!(
                "{name}: {differing} pixels differ, max |d| = {max:e}"
            ));
        }
        checked += 1;
        if name.ends_with(".bmp") {
            bmp.push(name.as_str());
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(checked >= 19, "only {checked} files checked");
    // 24-bit, 32-bit, 8-bit grey palette and 1-bit
    assert_eq!(bmp.len(), 4, "{bmp:?}");
}

#[test]
fn jpeg_decoders_agree_within_a_few_levels() {
    // libjpeg-turbo (Pillow) and zune-jpeg (image crate) differ in the IDCT and upsampling rounding
    let exp = expected();
    for name in ["gray.jpg", "rgb.jpg"] {
        let (max, mean, _, _) = compare(name, &exp[name]);
        assert!(
            max <= 3.0 / 255.0,
            "{name}: max |d| = {} levels",
            max * 255.0
        );
        assert!(
            mean <= 1.0 / 255.0,
            "{name}: mean |d| = {} levels",
            mean * 255.0
        );
    }
}

#[test]
fn unreadable_files_are_errors_not_panics() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(tmp).unwrap();
    let junk = tmp.join("junk.png");
    std::fs::write(&junk, b"not an image").unwrap();
    assert!(read_image(&junk).is_err());
    let empty = tmp.join("empty.tif");
    std::fs::write(&empty, b"").unwrap();
    assert!(read_image(&empty).is_err());
    let e = read_image(&tmp.join("missing.png"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("missing.png"), "{e}");
}

#[test]
fn a_grey_float32_tiff_is_an_unsupported_pixel_type_error_not_a_panic() {
    // Python reads it, values as they are (this one reaches beyond [0, 1]); the TIFF decoder of the
    // `image` crate has no 32-bit grey type, and there is no `DynamicImage` for one either
    let exp = expected();
    let gray = exp[GRAY_F32]["gray"].as_array().unwrap();
    assert_eq!(gray.len(), 11 * 7);
    assert!(gray.iter().any(|v| v.as_f64().unwrap() < 0.0));
    let path = dir().join("images").join(GRAY_F32);
    let e = read_image(&path).unwrap_err().to_string();
    assert!(
        e.contains(GRAY_F32) && e.contains("unsupported pixel type"),
        "{e}"
    );
    assert!(!e.contains('\n'), "one line: {e}");
}

#[test]
fn a_header_that_claims_a_huge_image_is_an_error_not_an_allocation_abort() {
    // The output buffer is allocated from the header's size before any data is read: the claims
    // below are 80 GB (16-bit RGBA PNG), 13 GB (24-bit BMP), 10 GB (8-bit grey TIFF) and more, all
    // over the 2 GiB the reader allows itself
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(tmp).unwrap();
    let files = [
        ("bomb.png", bomb_png(100_000, 100_000)),
        ("bomb.bmp", bomb_bmp(65_535, 65_535)), // (the BMP decoder refuses a side over 65535 itself)
        ("bomb.tif", bomb_tiff(100_000, 100_000)),
        // 2^32 - 1 on both sides: the product does not fit in 32 bits
        ("bomb_max.png", bomb_png(u32::MAX, u32::MAX)),
    ];
    for (name, bytes) in files {
        let path = tmp.join(name);
        std::fs::write(&path, bytes).unwrap();
        let e = read_image(&path).unwrap_err().to_string();
        assert!(
            e.contains(&format!("{name}: image too large to read")),
            "{name}: {e}"
        );
        assert!(!e.contains('\n'), "{name}: one line: {e}");
    }
}

#[test]
fn a_header_below_the_limit_fails_for_lack_of_data_not_for_its_size() {
    // a header of 4000 x 3000 pixels (96 MB as 16-bit RGBA) is far below the limit: the file then fails
    // for lack of data
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(tmp).unwrap();
    let path = tmp.join("claims_12mpx.png");
    std::fs::write(&path, bomb_png(4000, 3000)).unwrap(); // 96 MB as 16-bit RGBA
    let e = read_image(&path).unwrap_err().to_string();
    assert!(
        e.contains("cannot decode image") && !e.contains("too large"),
        "{e}"
    );
}

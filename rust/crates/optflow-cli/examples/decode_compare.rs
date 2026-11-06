//! Compares the Rust image reader with Python's `read_image` on one file.
//!
//!     cargo run --release -p optflow-cli --example decode_compare -- IMAGE PYTHON_GRAY.npy
//!
//! `PYTHON_GRAY.npy` is `np.save(path, optflow_ref.io.read_image(IMAGE))` (float64). Prints the
//! largest and the mean absolute difference (in grey levels of 1/255) and the fractions of pixels
//! that differ at all and by one level or more. Used to judge the JPEG decoder of the `image`
//! crate against libjpeg-turbo (Pillow).

use std::path::Path;

use optflow_cli::io::read_image;
use optflow_core::npy::{self, NpyData};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [image, reference] = args.as_slice() else {
        eprintln!("usage: decode_compare IMAGE PYTHON_GRAY.npy");
        std::process::exit(2);
    };
    let ours = read_image(Path::new(image)).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let bytes = std::fs::read(reference).expect("reference file");
    let theirs = npy::read(&bytes).expect("npy");
    let NpyData::F64(theirs_data) = theirs.data else {
        panic!("the reference must be float64");
    };
    assert_eq!(theirs.shape, [ours.h, ours.w], "shapes differ");
    let n = ours.data.len() as f64;
    let (mut max, mut sum, mut differ, mut level) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (a, b) in ours.data.iter().zip(&theirs_data) {
        let d = (a - b).abs() * 255.0;
        max = max.max(d);
        sum += d;
        differ += usize::from(d > 0.0);
        level += usize::from(d >= 1.0 - 1e-9);
    }
    println!(
        "{}  {}x{}  max |d| {:.4} levels  mean |d| {:.5} levels  differing {:.4}%  differing by >= 1 level {:.4}%",
        Path::new(image).file_name().unwrap().to_string_lossy(),
        ours.w,
        ours.h,
        max,
        sum / n,
        100.0 * differ as f64 / n,
        100.0 * level as f64 / n
    );
}

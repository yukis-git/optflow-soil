//! Compares the frames the Rust video reader delivers with those of Python's `iter_video` (OpenCV).
//!
//!     cargo run --release -p optflow-cli --example video_compare -- VIDEO PYTHON_FRAMES.npy
//!
//! `PYTHON_FRAMES.npy` is `np.save(path, np.stack(list(optflow_ref.io.iter_video(VIDEO))))`
//! (float64, frames x rows x columns). Prints the frame count and, in grey levels of 1/255, the
//! largest and the mean absolute difference over all frames.

use std::path::Path;

use optflow_cli::video::iter_video;
use optflow_core::npy::{self, NpyData};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [video, reference] = args.as_slice() else {
        eprintln!("usage: video_compare VIDEO PYTHON_FRAMES.npy");
        std::process::exit(2);
    };
    let theirs = npy::read(&std::fs::read(reference).expect("reference file")).expect("npy");
    let NpyData::F64(data) = theirs.data else {
        panic!("the reference must be float64");
    };
    let [n, h, w] = theirs.shape[..] else {
        panic!("the reference must be (frames, rows, columns)");
    };
    let frames = iter_video(Path::new(video), 0, None, 1).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let (mut count, mut max, mut sum, mut differ) = (0usize, 0.0f64, 0.0f64, 0usize);
    for frame in frames {
        let frame = frame.unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        });
        assert_eq!((frame.h, frame.w), (h, w), "frame {count}: size");
        let want = &data[count * h * w..(count + 1) * h * w];
        for (a, b) in frame.data.iter().zip(want) {
            let d = (a - b).abs() * 255.0;
            max = max.max(d);
            sum += d;
            differ += usize::from(d > 0.0);
        }
        count += 1;
    }
    println!(
        "{}  frames Rust {count} / Python {n}  {w}x{h}  max |d| {max:.6} levels  mean |d| {:.6} levels  differing {:.4}%",
        Path::new(video).file_name().unwrap().to_string_lossy(),
        sum / (count * h * w) as f64,
        100.0 * differ as f64 / (count * h * w) as f64
    );
}

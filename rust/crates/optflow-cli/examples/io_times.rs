//! Times what surrounds the tracker in a run: reading and averaging the frames and writing one
//! result. For the speed notes; not part of the tool.
//!
//! `io_times IMAGE_DIR OUT TAVG`

use std::path::Path;
use std::time::Instant;

use optflow_cli::{io, tavg};
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::strain::Strain;
use optflow_core::track::{Timing, TrackResult};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, out, m): (&str, &Path, usize) =
        (&args[0], Path::new(&args[1]), args[2].parse().unwrap());
    let paths = io::list_images(dir).unwrap();
    let t = Instant::now();
    let one = io::read_image(Path::new(&paths[0])).unwrap();
    let (w, h) = (one.w, one.h);
    println!(
        "{w}x{h}: read and decode one frame {:.3} s",
        t.elapsed().as_secs_f64()
    );
    let raw_paths = paths.clone();
    let raw = raw_paths
        .into_iter()
        .map(|p| Ok(io::downscale(io::read_image(Path::new(&p))?, 1)?.into_f32()));
    let cfg = Config::default();
    let align = (cfg.tavg_align && m > 0).then(|| cfg.clone());
    println!(
        "tavg_align {} tavg_model {}",
        cfg.tavg_align, cfg.tavg_model
    );
    let avg = tavg::Averaged::new(
        Box::new(raw),
        0,
        None,
        1,
        m,
        align,
        Box::new(|_, _| Ok(None)),
    );
    let mut last = Instant::now();
    for (i, f) in avg.enumerate() {
        let f = f.unwrap();
        println!(
            "sample {i}: mean of up to {} frames {:.3} s",
            2 * m + 1,
            last.elapsed().as_secs_f64()
        );
        drop(f);
        last = Instant::now();
    }
    let n = w * h;
    let map = |v: f32| Image::from_vec(w, h, (0..n).map(|i| v + i as f32 * 1e-6).collect());
    let res = TrackResult {
        ux: map(1.0),
        uy: map(2.0),
        conf: map(0.5),
        strain: Strain {
            exx_gl: map(0.1),
            eyy_gl: map(0.2),
            exy_gl: map(0.3),
            exx: Image::new(0, 0),
            eyy: Image::new(0, 0),
            exy: Image::new(0, 0),
            gmax: map(0.4),
            vol: map(0.5),
        },
        raw_ux: map(1.0),
        raw_uy: map(2.0),
        timing: Timing::default(),
        fb: None,
        crack: None,
    };
    for _ in 0..3 {
        let t = Instant::now();
        io::save_result_placed(out, &res, false, &|im, _| std::borrow::Cow::Borrowed(im)).unwrap();
        println!(
            "write one result (8 maps and valid) {:.3} s",
            t.elapsed().as_secs_f64()
        );
    }
    let _ = std::fs::remove_dir_all(out);
}

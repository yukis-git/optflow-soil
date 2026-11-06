//! Memory profile of the command line tool on real images, with the counting allocator of
//! `optflow-core/examples/common/mem.rs` (peak per pipeline stage and tile; `MEM_TRACE=1` adds the
//! large blocks alive at the peak, `MEM_TIMELINE=1` the bytes at every stage boundary).
//!
//!     cargo run --release -p optflow-cli --example mem_cli -- cli pair --ref A --def B ... --out DIR
//!     cargo run --release -p optflow-cli --example mem_cli -- window X0,Y0,X1,Y1 A B CONFIG [ROI]
//!
//! `cli` runs the tool itself on the arguments that follow (everything the `optflow` binary does:
//! decoding, crop, tracking, embedding, writing). `window` decodes `A` and `B` as the tool does
//! (scale 1), cuts the rectangle `[X0, X1) x [Y0, Y1)` out of both (and out of the ROI polygon's
//! mask, if given) and runs `track_pair` on it with the configuration file `CONFIG`: the cost of
//! the tracking alone, e.g. of one 1024 x 1024 tile or of an ROI crop.

#[path = "../../optflow-core/examples/common/mem.rs"]
mod mem;

use std::path::Path;
use std::process::ExitCode;

use optflow_cli::crop::Crop;
use optflow_cli::io;
use optflow_cli::polygon::polygon_mask;
use optflow_cli::settings;
use optflow_core::track::track_pair;

#[global_allocator]
static ALLOC: mem::Counting = mem::Counting;

fn window(args: &[String]) -> ExitCode {
    let (rect, a, b, config) = match args {
        [r, a, b, c, ..] => (r, a, b, c),
        _ => {
            eprintln!("usage: mem_cli window X0,Y0,X1,Y1 A B CONFIG [ROI]");
            return ExitCode::from(2);
        }
    };
    let v: Vec<usize> = rect
        .split(',')
        .map(|s| s.parse().expect("X0,Y0,X1,Y1"))
        .collect();
    let crop = Crop {
        x0: v[0],
        y0: v[1],
        x1: v[2],
        y1: v[3],
        margin: 0,
    };
    let cfg = settings::config(Some(Path::new(config)), &[]).expect("config");
    let load = |p: &str| {
        let full = io::read_image(Path::new(p)).expect("image").into_f32();
        ((full.w, full.h), crop.apply(&full))
    };
    mem::phase("load");
    let ((w, h), i1) = load(a);
    let (_, i2) = load(b);
    let mask = args.get(4).map(|roi| {
        let points = io::load_roi(Path::new(roi)).expect("ROI");
        crop.apply(&polygon_mask(w, h, &points, 1))
    });
    // the profile of the tracking alone: the decoding's peak is not counted
    mem::reset_peak();
    mem::phase("(before track_pair)");
    let res = track_pair(&i1, &i2, &cfg, mask.as_ref(), None).expect("track_pair");
    mem::phase("(after track_pair)");
    let valid = res.ux.data.iter().filter(|v| v.is_finite()).count();
    println!(
        "window {}x{}: valid {valid} px, timing {:?}",
        i1.w, i1.h, res.timing
    );
    drop(res);
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    mem::start();
    let code = match args.first().map(String::as_str) {
        Some("cli") => optflow_cli::main_with(&args[1..]),
        Some("window") => window(&args[1..]),
        _ => {
            eprintln!("usage: mem_cli cli ARGS... | mem_cli window X0,Y0,X1,Y1 A B CONFIG [ROI]");
            return ExitCode::from(2);
        }
    };
    print!("{}", mem::report());
    if std::env::var("MEM_TIMELINE").is_ok_and(|v| v == "1") {
        print!("{}", mem::timeline());
    }
    code
}

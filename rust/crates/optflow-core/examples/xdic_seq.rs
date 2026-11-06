//! `track_sequence` with `crack_model = "xdic"` on `f32` frames in `.npy` files, for comparing the
//! result on a real video with the Python reference without rounding the frames to 16 bits.
//!
//!     cargo run --release -p optflow-core --example xdic_seq -- --frames DIR --mask ROI.npy \
//!         --out OUT [--set K=V ...] [--limit N]
//!
//! `DIR` holds `f000.npy`, `f001.npy`, ... (`<f4`), `ROI.npy` is a bool array. `OUT` gets
//! `ux.f32`, `uy.f32`, `gmax.f32`, `conf.f32` (the frames of the result one after the other, raw
//! little-endian `f32`, frame 1 first) and `lines.json` (per frame the lines, bands, rejected lines,
//! COD / CSD samples, the reference and the seconds since the start). The stage table of `prof`
//! goes to stdout. `RAYON_NUM_THREADS` sets the thread count.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::npy::{self, NpyData};
use optflow_core::prof::{self, STAGES};
use optflow_core::track::track_sequence;

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn load_f32(path: &Path) -> Image<f32> {
    let a = npy::read(&std::fs::read(path).expect("read frame")).expect("parse npy");
    let NpyData::F32(v) = a.data else {
        panic!("{}: need <f4", path.display())
    };
    Image::from_vec(a.shape[1], a.shape[0], v)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = arg_value(&args, "--frames").expect("--frames DIR");
    let out = Path::new(arg_value(&args, "--out").expect("--out DIR"));
    std::fs::create_dir_all(out).expect("create the output directory");
    let mask = arg_value(&args, "--mask").map(|p| {
        let a = npy::read(&std::fs::read(p).expect("read mask")).expect("parse npy");
        let NpyData::Bool(v) = a.data else {
            panic!("{p}: need a bool array")
        };
        Image::from_vec(a.shape[1], a.shape[0], v)
    });
    let mut cfg = Config {
        crack_model: "xdic".to_string(),
        ..Config::default()
    };
    for (i, a) in args.iter().enumerate() {
        if a == "--set" {
            let kv = args.get(i + 1).expect("--set K=V");
            let (k, v) = kv.split_once('=').expect("--set K=V");
            let mut obj: serde_json::Value = serde_json::from_str(&cfg.to_json()).unwrap();
            obj[k] = serde_json::from_str(v).unwrap_or_else(|_| serde_json::json!(v));
            cfg = Config::from_json(&obj.to_string()).unwrap_or_else(|e| panic!("--set {kv}: {e}"));
        }
    }
    let limit: usize = arg_value(&args, "--limit").map_or(usize::MAX, |s| s.parse().expect("N"));
    let names: Vec<_> = (0..limit)
        .map(|i| Path::new(dir).join(format!("f{i:03}.npy")))
        .take_while(|p| p.exists())
        .collect();
    let frames = names.iter().map(|p| load_f32(p));
    let mut w = ["ux", "uy", "gmax", "conf"]
        .map(|n| BufWriter::new(File::create(out.join(format!("{n}.f32"))).expect("create")));
    let mut doc = Vec::new();
    prof::reset();
    let t0 = Instant::now();
    for item in track_sequence(frames, &cfg, mask.as_ref()) {
        let (k, r) = item.expect("track");
        for (wr, im) in w.iter_mut().zip([&r.ux, &r.uy, &r.strain.gmax, &r.conf]) {
            for v in &im.data {
                wr.write_all(&v.to_le_bytes()).expect("write");
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let l = r.crack.as_ref().map(|c| &c.lines);
        let nlines = l.map_or(0, |l| l.lines.len());
        println!("frame {k} {secs:.2} s lines {nlines}");
        doc.push(serde_json::json!({
            "k": k,
            "secs": secs,
            "reference": r.timing.reference,
            "lines": l.map(|l| l.lines.clone()).unwrap_or_default(),
            "bands": l.map(|l| l.bands.clone()).unwrap_or_default(),
            "rejected": l.map(|l| l.rejected.clone()).unwrap_or_default(),
            "jumps": l.map(|l| l.jumps.iter().map(|j| serde_json::json!({
                "pts": j.pts, "s": j.s, "cod": j.cod, "csd": j.csd,
            })).collect::<Vec<_>>()).unwrap_or_default(),
        }));
    }
    let total = t0.elapsed().as_secs_f64();
    let text = serde_json::to_string(&doc).expect("serialise");
    std::fs::write(out.join("lines.json"), text).expect("write lines");
    println!(
        "total {total:.2} s, {} threads",
        rayon::current_num_threads()
    );
    for (s, v) in STAGES.iter().zip(&prof::snapshot()) {
        if *v > 0.0 {
            println!("  {:<18} {:8.2} s {:5.1} %", s.name(), v, 100.0 * v / total);
        }
    }
}

//! Per-stage timing of `track_pair` on synthetic speckle, and output dumps for checking that an
//! optimisation preserves the results.
//!
//!     cargo run --release -p optflow-core --example bench_pair -- [--sizes 512,1024] [--runs 5]
//!         [--warmup 1] [--label NAME] [--shift TX,TY] [--max-disp D]
//!     cargo run --release -p optflow-core --example bench_pair -- --files A.npy,B.npy [--xdic] [--runs 5]
//!         [--warmup 1] [--label NAME] [--max-disp D] [--set K=V ...]
//!     cargo run --release -p optflow-core --example bench_pair -- --dump DIR [--quick]
//!     cargo run --release -p optflow-core --example bench_pair -- --compare DIR_A DIR_B
//!     cargo run --release -p optflow-core --example bench_pair -- --mem [--sizes N] [--tiled]
//!
//! `RAYON_NUM_THREADS` sets the thread count. Timing: the pair is the speckle pattern of the
//! tests under `uniform_strain(0.01, -0.003)` about the centre plus the translation (3.4, -2.1);
//! `--shift` replaces that translation and `--max-disp` the coarse stage's range (a shift of 91.7,-67.3 with a
//! max-disp of 128 on 512 px makes the coarse stage fail over a large region: the worst case of the pre-warp rule);
//! `--warmup` untimed runs (default 1), then `--runs` runs, and every stage's minimum over the runs (wall time, see
//! `optflow_core::prof`). Each run also prints one `RESULT` line for interleaved A/B comparisons.
//!
//! `--files A.npy,B.npy` times the pair of two `f32` (or `f64`) images instead of the speckle,
//! `--xdic` with `crack_model = "xdic"` (its stages are the `xdic ...` lines of the table: detect,
//! responses of the references and of the per-side pre-warps, the per-side solve, the geometry
//! update, and the crack-or-band decision with the outputs; the baseline stages above them are
//! those of the baseline iterations of `refine_xdic` too). `--set K=V` overrides a config value
//! (numbers and strings, e.g. `--set xdic_iters=3`).
//!
//! `--dump DIR` writes the outputs (`.npy`, `f32`) of: the timing pairs at every size, the 512^2
//! `uniform_0.01` pair (no translation, like the Python benchmark case), a tiled refinement
//! (384^2, 9 tiles), and `refine` / `track_pair` / `track_sequence` on the repository fixtures.
//! `--compare` prints, per array, the NaN agreement, the RMS and largest difference over the
//! pixels finite in both, and whether the two are bit-identical. `--quick` skips the 1024^2 pair.
//!
//! `--mem` profiles the memory of one timing run (no warm-up) with the counting allocator of
//! `common/mem.rs`: the peak per stage and tile (`MEM_TRACE=1`: the large blocks alive at the
//! peak; `MEM_TIMELINE=1`: the bytes at every stage boundary). It takes one size: the footprint
//! maximum is that of the whole process, so each size needs a process of its own. The pair is
//! built after the profile starts and counts in its first window. `--tiled` runs the pair through
//! the tiled fine stage (`TILE` = 1024) by tracking an image of `n + 1` columns. The real images
//! go through the command line tool's `mem_cli` example, which shares the allocator.

#[path = "../tests/common/synth.rs"]
mod synth;

#[path = "common/mem.rs"]
mod mem;

#[global_allocator]
static ALLOC: mem::Counting = mem::Counting;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::npy::{self, NpyData};
use optflow_core::phaseflow::refine;
use optflow_core::prof::{self, STAGES};
use optflow_core::tile::refine_tiled;
use optflow_core::track::{track_pair, track_sequence, TrackResult};

fn speckle_pair(n: usize, t: (f64, f64)) -> (Image<f32>, Image<f32>) {
    let c = n as f64 / 2.0;
    let field = synth::uniform_strain(0.01, -0.003, (c, c), t);
    let pat = synth::make_pattern(n, n, 1, 32.0);
    (
        synth::render(&pat, n, n, None),
        synth::render(&pat, n, n, Some(&field)),
    )
}

const BENCH_T: (f64, f64) = (3.4, -2.1);

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--compare") {
        let (a, b) = (&args[i + 1], &args[i + 2]);
        compare(Path::new(a), Path::new(b));
        return;
    }
    if let Some(dir) = arg_value(&args, "--dump") {
        dump(Path::new(dir), args.iter().any(|a| a == "--quick"));
        return;
    }
    let sizes: Vec<usize> = arg_value(&args, "--sizes")
        .unwrap_or("512,1024")
        .split(',')
        .map(|s| s.parse().expect("--sizes: comma-separated integers"))
        .collect();
    let runs: usize = arg_value(&args, "--runs").map_or(5, |s| s.parse().expect("--runs"));
    let label = arg_value(&args, "--label").unwrap_or("");
    let warmup: usize = arg_value(&args, "--warmup").map_or(1, |s| s.parse().expect("--warmup"));
    let shift = arg_value(&args, "--shift").map_or(BENCH_T, |s| {
        let (x, y) = s.split_once(',').expect("--shift: TX,TY");
        (x.parse().expect("--shift"), y.parse().expect("--shift"))
    });
    let mut cfg = Config {
        max_disp: arg_value(&args, "--max-disp").map_or(Config::default().max_disp, |s| {
            s.parse().expect("--max-disp")
        }),
        ..Config::default()
    };
    if args.iter().any(|a| a == "--xdic") {
        cfg.crack_model = "xdic".to_string();
    }
    for (i, a) in args.iter().enumerate() {
        if a == "--set" {
            let kv = args.get(i + 1).expect("--set K=V");
            let (k, v) = kv.split_once('=').expect("--set K=V");
            // through the JSON of the config, so that any key and its type check apply
            let mut obj: serde_json::Value = serde_json::from_str(&cfg.to_json()).unwrap();
            obj[k] = serde_json::from_str(v).unwrap_or_else(|_| serde_json::json!(v));
            cfg = Config::from_json(&obj.to_string()).unwrap_or_else(|e| panic!("--set {kv}: {e}"));
        }
    }
    if let Some(files) = arg_value(&args, "--files") {
        let (a, b) = files.split_once(',').expect("--files A.npy,B.npy");
        bench_files(a, b, runs, warmup, label, &cfg);
        return;
    }
    if args.iter().any(|a| a == "--mem") {
        let [n] = sizes[..] else {
            eprintln!("--mem profiles one size per process: give --sizes N");
            std::process::exit(2);
        };
        mem_profile(n, args.iter().any(|a| a == "--tiled"), shift, &cfg);
        return;
    }
    for n in sizes {
        bench(n, runs, warmup, label, shift, &cfg);
    }
}

fn load_image(path: &str) -> Image<f32> {
    let a = npy::read(&std::fs::read(path).expect("read image")).expect("parse npy");
    assert_eq!(a.shape.len(), 2, "{path}: not a 2-D array");
    let v: Vec<f32> = match a.data {
        NpyData::F32(v) => v,
        NpyData::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        _ => panic!("{path}: need <f4 or <f8"),
    };
    Image::from_vec(a.shape[1], a.shape[0], v)
}

/// `bench` for two image files.
fn bench_files(a: &str, b: &str, runs: usize, warmup: usize, label: &str, cfg: &Config) {
    let (a, b) = (load_image(a), load_image(b));
    let threads = rayon::current_num_threads();
    for _ in 0..warmup {
        let _ = track_pair(&a, &b, cfg, None, None).unwrap();
    }
    let mut best_total = f64::INFINITY;
    let mut best = [f64::INFINITY; STAGES.len()];
    for _ in 0..runs {
        prof::reset();
        let t = Instant::now();
        let r = track_pair(&a, &b, cfg, None, None).unwrap();
        let total = t.elapsed().as_secs_f64();
        let stages = prof::snapshot();
        let lines = r.crack.as_ref().map_or(0, |c| c.lines.lines.len());
        std::hint::black_box(&r);
        let mut line = format!(
            "RESULT label={label} n={}x{} threads={threads} total={total:.4} lines={lines}",
            a.w, a.h
        );
        for (s, v) in STAGES.iter().zip(&stages) {
            line += &format!(" {}={v:.4}", s.name().replace(' ', "_"));
        }
        println!("{line}");
        best_total = best_total.min(total);
        for (b, v) in best.iter_mut().zip(&stages) {
            *b = b.min(*v);
        }
    }
    println!(
        "{}x{}, {threads} threads, min of {runs} runs: total {:.1} ms",
        a.w,
        a.h,
        best_total * 1e3
    );
    for (s, v) in STAGES.iter().zip(&best) {
        if *v > 0.0 {
            println!(
                "  {:<16} {:9.1} ms  {:5.1} %",
                s.name(),
                v * 1e3,
                100.0 * v / best_total
            );
        }
    }
}

/// The memory profile of one `track_pair` on the timing pair of size `n` (`tiled`: `n + 1`
/// columns, so that the fine stage runs in tiles).
fn mem_profile(n: usize, tiled: bool, shift: (f64, f64), cfg: &Config) {
    // before the pair is built, so that every block it frees later was counted
    mem::start();
    let (a, b) = if tiled {
        let w = n + 1;
        let c = n as f64 / 2.0;
        let field = synth::uniform_strain(0.01, -0.003, (c, c), shift);
        let pat = synth::make_pattern(w, n, 1, 32.0);
        (
            synth::render(&pat, w, n, None),
            synth::render(&pat, w, n, Some(&field)),
        )
    } else {
        speckle_pair(n, shift)
    };
    mem::phase("(after building the pair)");
    let t = Instant::now();
    let r = track_pair(&a, &b, cfg, None, None).unwrap();
    mem::phase("(after track_pair)");
    let total = t.elapsed().as_secs_f64();
    println!(
        "{}x{} {}: {total:.2} s, {} threads",
        a.w,
        a.h,
        if tiled { "tiled" } else { "untiled" },
        rayon::current_num_threads()
    );
    drop(r);
    print!("{}", mem::report());
    if std::env::var("MEM_TIMELINE").is_ok_and(|v| v == "1") {
        print!("{}", mem::timeline());
    }
}

fn bench(n: usize, runs: usize, warmup: usize, label: &str, shift: (f64, f64), cfg: &Config) {
    let (a, b) = speckle_pair(n, shift);
    let threads = rayon::current_num_threads();
    for _ in 0..warmup {
        let _ = track_pair(&a, &b, cfg, None, None).unwrap();
    }
    let mut best_total = f64::INFINITY;
    let mut best = [f64::INFINITY; STAGES.len()];
    for _ in 0..runs {
        prof::reset();
        let t = Instant::now();
        let r = track_pair(&a, &b, cfg, None, None).unwrap();
        let total = t.elapsed().as_secs_f64();
        let stages = prof::snapshot();
        std::hint::black_box(&r);
        let mut line = format!("RESULT label={label} n={n} threads={threads} total={total:.4}");
        for (s, v) in STAGES.iter().zip(&stages) {
            line += &format!(" {}={v:.4}", s.name().replace(' ', "_"));
        }
        println!("{line}");
        best_total = best_total.min(total);
        for (b, v) in best.iter_mut().zip(&stages) {
            *b = b.min(*v);
        }
    }
    println!(
        "{n}x{n}, {threads} threads, min of {runs} runs: total {:.1} ms",
        best_total * 1e3
    );
    for (s, v) in STAGES.iter().zip(&best) {
        println!(
            "  {:<16} {:9.1} ms  {:5.1} %",
            s.name(),
            v * 1e3,
            100.0 * v / best_total
        );
    }
}

// ---------------------------------------------------------------------------------------------
// dumps

fn save(dir: &Path, name: &str, im: &Image<f32>) {
    let bytes = npy::write_f32(&[im.h, im.w], &im.data);
    std::fs::write(dir.join(format!("{name}.npy")), bytes).expect("write dump");
}

fn save_track(dir: &Path, name: &str, r: &TrackResult) {
    for (part, im) in [
        ("ux", &r.ux),
        ("uy", &r.uy),
        ("conf", &r.conf),
        ("raw_ux", &r.raw_ux),
        ("raw_uy", &r.raw_uy),
        ("exx_gl", &r.strain.exx_gl),
        ("gmax", &r.strain.gmax),
    ] {
        save(dir, &format!("{name}_{part}"), im);
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load(name: &str) -> (Vec<usize>, Vec<f64>) {
    let bytes = std::fs::read(fixture_dir().join(format!("{name}.npy"))).expect("fixture");
    let a = npy::read(&bytes).expect("npy");
    let v = match a.data {
        NpyData::F64(v) => v,
        NpyData::F32(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U8(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U16(v) => v.into_iter().map(f64::from).collect(),
        NpyData::Bool(v) => v.into_iter().map(|b| f64::from(u8::from(b))).collect(),
    };
    (a.shape, v)
}

fn img(name: &str) -> Image<f32> {
    let (s, v) = load(name);
    Image::from_vec(s[1], s[0], v.iter().map(|&x| x as f32).collect())
}

fn frames(name: &str) -> Vec<Image<f32>> {
    let (s, v) = load(name);
    let (n, h, w) = (s[0], s[1], s[2]);
    (0..n)
        .map(|k| {
            let px = &v[k * h * w..(k + 1) * h * w];
            Image::from_vec(w, h, px.iter().map(|&x| x as f32).collect())
        })
        .collect()
}

fn dump(dir: &Path, quick: bool) {
    std::fs::create_dir_all(dir).expect("dump dir");
    let cfg = Config::default();
    for n in [512, 1024].into_iter().filter(|&n| !quick || n < 1024) {
        let (a, b) = speckle_pair(n, BENCH_T);
        save_track(
            dir,
            &format!("speckle{n}"),
            &track_pair(&a, &b, &cfg, None, None).unwrap(),
        );
    }
    let (a, b) = speckle_pair(512, (0.0, 0.0));
    save_track(
        dir,
        "uniform_0.01_512",
        &track_pair(&a, &b, &cfg, None, None).unwrap(),
    );

    // tiled refinement, as in parity_track's thread-count test
    let n = 384;
    let field = synth::uniform_strain(0.01, -0.003, (192.0, 192.0), (3.4, -2.1));
    let pat = synth::make_pattern(n, n, 4, 80.0);
    let (a, b) = (
        synth::render(&pat, n, n, None),
        synth::render(&pat, n, n, Some(&field)),
    );
    let (mut ux, mut uy) = synth::displacement(&field, n, n);
    ux.data.iter_mut().for_each(|v| *v += 0.3);
    uy.data.iter_mut().for_each(|v| *v -= 0.3);
    let f = refine_tiled(&a, &b, &ux, &uy, &cfg, 128);
    for (part, im) in [
        ("ux", &f.ux),
        ("uy", &f.uy),
        ("conf", &f.conf),
        ("resid", &f.resid),
    ] {
        save(dir, &format!("tiled384_{part}"), im);
    }

    // repository fixtures
    let (a, b) = (img("refine_a"), img("refine_b"));
    let (ux, uy) = (img("refine_init_ux"), img("refine_init_uy"));
    for order in [1, 2] {
        let c = Config {
            shape_order: order,
            ..Config::default()
        };
        let f = refine(&a, &b, &ux, &uy, &c);
        for (part, im) in [
            ("ux", &f.ux),
            ("uy", &f.uy),
            ("conf", &f.conf),
            ("resid", &f.resid),
        ] {
            save(dir, &format!("fx_refine_o{order}_{part}"), im);
        }
        let (sa, sb) = (img("refine_stripe_a"), img("refine_stripe_b"));
        let s0 = Image::from_vec(sa.w, sa.h, vec![0.7f32; sa.w * sa.h]);
        let s1 = Image::from_vec(sa.w, sa.h, vec![-0.4f32; sa.w * sa.h]);
        let f = refine(&sa, &sb, &s0, &s1, &c);
        for (part, im) in [("ux", &f.ux), ("uy", &f.uy), ("conf", &f.conf)] {
            save(dir, &format!("fx_stripe_o{order}_{part}"), im);
        }
    }
    let (a, b) = (img("refine_border_a"), img("refine_border_b"));
    let (w, h) = (a.w, a.h);
    let (mut ux, mut uy) = (Image::new(w, h), Image::new(w, h));
    for y in 0..h {
        for x in 0..w {
            ux.set(x, y, (0.004 * (x as f64 - 32.0) - 1.5 + 0.3) as f32);
            uy.set(x, y, (-0.002 * (y as f64 - 32.0) + 1.7 - 0.3) as f32);
        }
    }
    let c = Config {
        edge_guard: 0.0,
        ..Config::default()
    };
    let f = refine(&a, &b, &ux, &uy, &c);
    for (part, im) in [("ux", &f.ux), ("uy", &f.uy), ("conf", &f.conf)] {
        save(dir, &format!("fx_border_{part}"), im);
    }
    let (a, b) = (img("pair_a"), img("pair_b"));
    let (s, v) = load("pair_roi");
    let roi = Image::from_vec(s[1], s[0], v.iter().map(|&x| x > 0.5).collect());
    save_track(
        dir,
        "fx_pair",
        &track_pair(&a, &b, &cfg, Some(&roi), None).unwrap(),
    );
    let seq_cfg = Config {
        ref_update_conf: 2.0,
        ..Config::default()
    };
    for name in ["seq_frames", "seqc1_frames"] {
        for item in track_sequence(frames(name).into_iter(), &seq_cfg, None) {
            let (k, r) = item.unwrap();
            save_track(dir, &format!("fx_{name}_{k}"), &r);
        }
    }
    println!("dumped to {}", dir.display());
}

// ---------------------------------------------------------------------------------------------
// comparison

fn read_f32(path: &Path) -> Vec<f32> {
    let a = npy::read(&std::fs::read(path).expect("read dump")).expect("npy");
    match a.data {
        NpyData::F32(v) => v,
        _ => panic!("{}: not f32", path.display()),
    }
}

fn compare(a: &Path, b: &Path) {
    let mut names = BTreeMap::new();
    for e in std::fs::read_dir(a).expect("dir A") {
        let p = e.expect("entry").path();
        if p.extension().is_some_and(|x| x == "npy") {
            names.insert(p.file_name().unwrap().to_owned(), p);
        }
    }
    let (mut identical, mut worst_disp, mut worst_conf) = (0usize, 0.0f64, 0.0f64);
    for (name, pa) in &names {
        let pb = b.join(name);
        let (va, vb) = (read_f32(pa), read_f32(&pb));
        assert_eq!(va.len(), vb.len(), "{name:?}: sizes differ");
        let (mut agree, mut n, mut sq, mut max) = (0usize, 0usize, 0.0f64, 0.0f64);
        let mut same = true;
        for (&x, &y) in va.iter().zip(&vb) {
            same &= x.to_bits() == y.to_bits();
            if x.is_finite() == y.is_finite() {
                agree += 1;
            }
            if x.is_finite() && y.is_finite() {
                let d = f64::from(x) - f64::from(y);
                n += 1;
                sq += d * d;
                max = max.max(d.abs());
            }
        }
        let rms = (sq / n.max(1) as f64).sqrt();
        let name = name.to_string_lossy();
        if same {
            identical += 1;
        } else if name.contains("conf") {
            worst_conf = worst_conf.max(rms);
        } else if ["_ux", "_uy"].iter().any(|s| name.contains(s)) {
            worst_disp = worst_disp.max(rms);
        }
        println!(
            "{name:<34} {} nan-agree {:.6}  rms {rms:.3e}  max {max:.3e}",
            if same { "IDENTICAL" } else { "differs  " },
            agree as f64 / va.len().max(1) as f64
        );
    }
    println!(
        "{identical} of {} arrays bit-identical; worst rms: displacement {worst_disp:.3e} px, conf {worst_conf:.3e}",
        names.len()
    );
}

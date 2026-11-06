//! The macOS body of the `bench_gpu` example (Metal exists only there).

#[path = "../../../optflow-core/tests/common/synth.rs"]
mod synth;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, EngineError, FineEngine};
use optflow_core::image::Image;
use optflow_core::phaseflow::RawFlow;
use optflow_core::prof::{self, STAGES};
use optflow_core::track::{track_pair_on, TrackResult};
use optflow_gpu::GpuEngine;

/// An engine that adds the wall time of every `refine_raw` to `fine`.
struct Timed<'a> {
    inner: &'a dyn FineEngine,
    fine: Mutex<f64>,
}

impl FineEngine for Timed<'_> {
    fn refine_raw(
        &self,
        i1: &Image<f32>,
        i2: &Image<f32>,
        shift: (i64, i64),
        ux: &Image<f32>,
        uy: &Image<f32>,
        cfg: &Config,
    ) -> Result<RawFlow, EngineError> {
        let t = Instant::now();
        let r = self.inner.refine_raw(i1, i2, shift, ux, uy, cfg);
        *self.fine.lock().unwrap() += t.elapsed().as_secs_f64();
        r
    }

    fn check_window(&self, w: usize, h: usize, cfg: &Config) -> Result<(), EngineError> {
        self.inner.check_window(w, h, cfg)
    }
}

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn uptime() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

type Times = Vec<(&'static str, f64, usize)>;

/// Keeps the per-name minimum of `new` in `best`.
fn keep_min(best: &mut Times, new: Times) {
    for (k, s, c) in new {
        match best.iter_mut().find(|e| e.0 == k) {
            Some(e) => e.1 = e.1.min(s),
            None => best.push((k, s, c)),
        }
    }
}

fn print_table(title: &str, rows: &mut Times) {
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("{title} (min over the runs):");
    for (name, s, c) in rows.iter() {
        if *c > 0 {
            println!("  {name:<24} {:>9.3} ms  {:>5} calls", 1e3 * s, c);
        } else {
            println!("  {name:<24} {:>9.3} ms", 1e3 * s);
        }
    }
    println!(
        "  total {:.3} ms",
        1e3 * rows.iter().map(|r| r.1).sum::<f64>()
    );
}

/// Peak physical footprint of this process in bytes (macOS `ri_lifetime_max_phys_footprint`).
fn peak_footprint() -> u64 {
    // SAFETY: `rusage_info_v4` is plain old data, for which all zero bytes is a valid value.
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    // SAFETY: the buffer is a live `rusage_info_v4`, which is what flavor 4 (RUSAGE_INFO_V4) of
    // `proc_pid_rusage` fills in for the current process.
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            4,
            std::ptr::addr_of_mut!(info).cast::<libc::rusage_info_t>(),
        )
    };
    if rc == 0 {
        info.ri_lifetime_max_phys_footprint
    } else {
        0
    }
}

/// Measures of the difference of two fields (GPU spec section 5).
struct Diff {
    rms: f64,
    max: f64,
    /// Pixels that are finite in one field and not in the other.
    mismatched: usize,
    /// Pixels finite in both (an empty overlap fails the gate).
    n: usize,
}

/// RMS and largest absolute difference of `a` and `b` over the pixels finite in both, and the
/// number of pixels finite in one only.
fn diff(a: &Image<f32>, b: &Image<f32>) -> Diff {
    let (mut s, mut n, mut max, mut mismatched) = (0.0f64, 0usize, 0.0f64, 0usize);
    for (&x, &y) in a.data.iter().zip(&b.data) {
        if x.is_finite() && y.is_finite() {
            let d = f64::from(x) - f64::from(y);
            s += d * d;
            n += 1;
            max = max.max(d.abs());
        } else if x.is_finite() != y.is_finite() {
            mismatched += 1;
        }
    }
    Diff {
        rms: if n > 0 { (s / n as f64).sqrt() } else { 0.0 },
        max,
        mismatched,
        n,
    }
}

/// Prints the section 5 measures of GPU `g` against CPU `c`; true when the gate passes.
fn compare(n: usize, g: &TrackResult, c: &TrackResult) -> bool {
    let fields: [(&str, &Image<f32>, &Image<f32>, f64); 8] = [
        ("ux", &g.ux, &c.ux, 1e-5),
        ("uy", &g.uy, &c.uy, 1e-5),
        ("conf", &g.conf, &c.conf, 1e-5),
        ("Exx", &g.strain.exx, &c.strain.exx, 1e-6),
        ("Eyy", &g.strain.eyy, &c.strain.eyy, 1e-6),
        ("Exy", &g.strain.exy, &c.strain.exy, 1e-6),
        ("gmax", &g.strain.gmax, &c.strain.gmax, 1e-6),
        ("vol", &g.strain.vol, &c.strain.vol, 1e-6),
    ];
    let mut pass = true;
    let mut worst_strain: f64 = 0.0;
    println!("compare {n}^2, GPU against CPU (RMS over the pixels finite in both):");
    for (name, a, b, tol) in fields {
        let d = diff(a, b);
        let ok = d.n > 0 && d.rms <= tol && (name != "conf" || d.mismatched == 0);
        pass &= ok;
        if tol == 1e-6 {
            worst_strain = worst_strain.max(d.rms);
        }
        println!(
            "  {name:<5} rms {:.3e}  max {:.3e}  finite in one only {:>6}  tolerance {tol:.0e}  {}",
            d.rms,
            d.max,
            d.mismatched,
            if ok { "ok" } else { "FAIL" }
        );
    }
    let same =
        g.ux.data
            .iter()
            .zip(&c.ux.data)
            .filter(|(x, y)| x.is_finite() == y.is_finite())
            .count();
    let valid = same as f64 / g.ux.data.len() as f64;
    let valid_ok = valid >= 0.999;
    pass &= valid_ok;
    println!(
        "  valid agreement {valid:.6} (ux finite in both or neither)  need 0.999  {}",
        if valid_ok { "ok" } else { "FAIL" }
    );
    println!(
        "GATE n={n} {} (strain rms worst {worst_strain:.3e})",
        if pass { "pass" } else { "FAIL" }
    );
    pass
}

fn result_line(label: &str, device: &str, n: usize, total: f64, stages: &[f64]) -> String {
    let mut line = format!("RESULT label={label} device={device} n={n} total={total:.4}");
    for (s, v) in STAGES.iter().zip(stages) {
        line += &format!(" {}={v:.4}", s.name().replace(' ', "_"));
    }
    line
}

struct Opts {
    runs: usize,
    warmup: usize,
    label: String,
    shift: (f64, f64),
    gpu_only: bool,
    compare_cpu: bool,
    mem: bool,
    cfg: Config,
}

/// Times and (with `--compare-cpu`) compares the GPU and the CPU on the speckle pair of size
/// `n`; false when the comparison gate fails. `peak_alloc` takes the largest Metal allocation
/// seen.
fn bench_size(n: usize, gpu: &GpuEngine, o: &Opts, peak_alloc: &AtomicU64) -> bool {
    let c = n as f64 / 2.0;
    let field = synth::uniform_strain(0.01, -0.003, (c, c), o.shift);
    let pat = synth::make_pattern(n, n, 1, 32.0);
    let (a, b) = (
        synth::render(&pat, n, n, None),
        synth::render(&pat, n, n, Some(&field)),
    );
    let engines: Vec<(&str, &dyn FineEngine)> = if o.gpu_only {
        vec![("gpu", gpu)]
    } else {
        vec![("gpu", gpu), ("cpu", &CpuEngine)]
    };
    // warm-up: pipelines, the bank, the buffer pool
    for _ in 0..o.warmup {
        track_pair_on(gpu, &a, &b, &o.cfg, None, None).unwrap();
    }
    gpu.take_stage_times();
    gpu.take_kernel_times();
    println!("before: {}", uptime());
    let mut best = vec![(f64::INFINITY, f64::INFINITY); engines.len()];
    let (mut stages, mut kernels) = (Times::new(), Times::new());
    let mut walls = vec![Times::new(); engines.len()];
    let mut last: Vec<Option<TrackResult>> = engines.iter().map(|_| None).collect();
    for run in 0..o.runs {
        for (k, (name, e)) in engines.iter().enumerate() {
            let timed = Timed {
                inner: *e,
                fine: Mutex::new(0.0),
            };
            prof::reset();
            let t = Instant::now();
            let r = track_pair_on(&timed, &a, &b, &o.cfg, None, None).unwrap();
            let total = t.elapsed().as_secs_f64();
            let fine = *timed.fine.lock().unwrap();
            println!("run {run} {name}: track_pair {total:.3} s, fine stage {fine:.3} s");
            println!(
                "{}",
                result_line(&o.label, name, n, total, &prof::snapshot())
            );
            best[k] = (best[k].0.min(total), best[k].1.min(fine));
            let wall: Times = STAGES
                .iter()
                .zip(prof::snapshot())
                .filter(|(_, s)| *s > 0.0)
                .map(|(st, s)| (st.name(), s, 0))
                .collect();
            keep_min(&mut walls[k], wall);
            if k == 0 {
                keep_min(&mut stages, gpu.take_stage_times());
                keep_min(&mut kernels, gpu.take_kernel_times());
            }
            last[k] = Some(r);
        }
    }
    println!("after: {}", uptime());
    for ((name, _), (t, f)) in engines.iter().zip(&best) {
        println!("min {name} {n}^2: track_pair {t:.3} s, fine stage {f:.3} s");
    }
    if engines.len() == 2 {
        println!(
            "cpu/gpu: track_pair {:.2}x, fine stage {:.2}x",
            best[1].0 / best[0].0,
            best[1].1 / best[0].1
        );
    }
    for ((name, _), w) in engines.iter().zip(&mut walls) {
        print_table(&format!("{name} wall time per stage"), w);
    }
    if !stages.is_empty() {
        print_table("GPU time per stage", &mut stages);
    }
    if !kernels.is_empty() {
        print_table("GPU time per kernel", &mut kernels);
    }
    if o.mem {
        println!(
            "mem {n}^2: Metal allocated so far (max sampled) {:.1} MiB",
            peak_alloc.load(Ordering::Relaxed) as f64 / (1 << 20) as f64
        );
    }
    match (&last[0], last.get(1).and_then(Option::as_ref)) {
        (Some(g), Some(c)) if o.compare_cpu => compare(n, g, c),
        _ => true,
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sizes: Vec<usize> = arg(&args, "--sizes")
        .or_else(|| arg(&args, "--size"))
        .unwrap_or("512")
        .split(',')
        .map(|v| v.parse().expect("--sizes: comma-separated integers"))
        .collect();
    let mut o = Opts {
        runs: arg(&args, "--runs").map_or(3, |v| v.parse().unwrap()),
        warmup: arg(&args, "--warmup").map_or(1, |v| v.parse().unwrap()),
        label: arg(&args, "--label").unwrap_or("").to_string(),
        shift: arg(&args, "--shift").map_or((3.4, -2.1), |v| {
            let p: Vec<f64> = v.split(',').map(|x| x.parse().unwrap()).collect();
            (p[0], p[1])
        }),
        gpu_only: args.iter().any(|a| a == "--gpu-only"),
        compare_cpu: args.iter().any(|a| a == "--compare-cpu"),
        mem: args.iter().any(|a| a == "--mem"),
        cfg: Config::default(),
    };
    if let Some(d) = arg(&args, "--max-disp") {
        o.cfg.max_disp = d.parse().unwrap();
    }
    if o.compare_cpu && o.gpu_only {
        eprintln!("--compare-cpu needs the CPU run: drop --gpu-only");
        std::process::exit(2);
    }
    let gpu = GpuEngine::new().expect("a Metal device");

    // --mem: sample the Metal allocation while the runs go on
    let peak_alloc = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let all_pass = std::thread::scope(|sc| {
        if o.mem {
            sc.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    peak_alloc.fetch_max(gpu.allocated_bytes(), Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            });
        }
        let mut pass = true;
        for &n in &sizes {
            pass &= bench_size(n, &gpu, &o, &peak_alloc);
        }
        stop.store(true, Ordering::Relaxed);
        pass
    });
    if o.mem {
        peak_alloc.fetch_max(gpu.allocated_bytes(), Ordering::Relaxed);
        println!(
            "MEM metal_max_allocated={} MiB={:.1} process_peak_footprint={} MiB={:.1}",
            peak_alloc.load(Ordering::Relaxed),
            peak_alloc.load(Ordering::Relaxed) as f64 / (1 << 20) as f64,
            peak_footprint(),
            peak_footprint() as f64 / (1 << 20) as f64
        );
    }
    if !all_pass {
        std::process::exit(1);
    }
}

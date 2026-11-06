//! The macOS body of the `bench_kernels` example (Metal exists only there).

use optflow_core::config::Config;
use optflow_core::gabor::{gabor_bank, PAD};
use optflow_gpu::device::{Batch, Buffer, Gpu};
use optflow_gpu::fft::Fft2Plan;
use optflow_gpu::kernels::{self, MomentPlanGpu};
use optflow_gpu::GpuError;

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map_or(default, |v| v.parse().unwrap())
}

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn upload(gpu: &Gpu, v: &[f32]) -> Buffer {
    let mut b = gpu.buffer(4 * v.len()).unwrap();
    b.as_mut_slice::<f32>().copy_from_slice(v);
    b
}

struct Bufs {
    refpack: Buffer,
    responses: Buffer,
    field: [Buffer; 4],
    u: [Buffer; 4],
    u_next: [Buffer; 4],
    sums: Buffer,
    global: Buffer,
    taps: Buffer,
    r: usize,
    plan: MomentPlanGpu,
    energy: Buffer,
    resid: Buffer,
    slope: Buffer,
    erange: Buffer,
    img: Buffer,
    spec: Buffer,
    groups: Vec<Buffer>,
    ga: Buffer,
    gscratch: Buffer,
    fft: Fft2Plan,
}

type Step = fn(&mut Bufs, &mut Batch, usize, usize, usize) -> Result<(), GpuError>;

fn accumulate(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, f: usize) -> Result<(), GpuError> {
    let [sx, sy, csx, csy] = &mut s.field;
    let [a, bb, c, d] = &mut s.u;
    kernels::accumulate(
        b,
        &mut s.refpack,
        &mut s.responses,
        f,
        sx,
        sy,
        csx,
        csy,
        a,
        bb,
        c,
        d,
        w,
        h,
        &mut s.sums,
    )
}

fn moments(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, _: usize) -> Result<(), GpuError> {
    kernels::moments_global(
        b,
        &mut s.sums,
        w,
        h,
        &mut s.taps,
        s.r,
        &mut s.plan,
        &mut s.global,
    )
}

fn solve(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, _: usize) -> Result<(), GpuError> {
    let [a0, a1, a2, a3] = &mut s.u;
    let [o0, o1, o2, o3] = &mut s.u_next;
    kernels::solve_blocks(
        b,
        &mut s.sums,
        &mut s.global,
        [a0, a1, a2, a3],
        w,
        h,
        &mut s.taps,
        s.r,
        &mut s.plan,
        [o0, o1, o2, o3],
        &mut s.energy,
        &mut s.resid,
        &mut s.slope,
        &mut s.erange,
    )
}

fn forward(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, _: usize) -> Result<(), GpuError> {
    kernels::pad_reflect_complex(b, &mut s.img, w, h, &mut s.spec)?;
    s.fft.forward(b, &mut s.spec, 1)
}

fn multiply(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, f: usize) -> Result<(), GpuError> {
    let np = (w + 2 * PAD) * (h + 2 * PAD);
    for (g, filt) in s.groups.iter_mut().enumerate() {
        let cnt = (f - 4 * g).min(4);
        kernels::filter_multiply(b, &mut s.spec, filt, np, cnt, &mut s.ga)?;
    }
    Ok(())
}

fn inverse(s: &mut Bufs, b: &mut Batch, _: usize, _: usize, f: usize) -> Result<(), GpuError> {
    for g in 0..s.groups.len() {
        let cnt = (f - 4 * g).min(4);
        s.fft.inverse(b, &mut s.ga, cnt)?;
    }
    Ok(())
}

fn splines(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, f: usize) -> Result<(), GpuError> {
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    for g in 0..s.groups.len() {
        let cnt = (f - 4 * g).min(4);
        kernels::response_splines(
            b,
            &mut s.ga,
            &mut s.gscratch,
            (pw, ph),
            cnt,
            1.0 / (pw * ph) as f32,
            (4 * g, f),
            &mut s.responses,
        )?;
    }
    Ok(())
}

fn prefilter1(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, _: usize) -> Result<(), GpuError> {
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    kernels::prefilter_complex(b, &mut s.ga, &mut s.gscratch, pw, ph, 1)
}

fn prefilter4(s: &mut Bufs, b: &mut Batch, w: usize, h: usize, _: usize) -> Result<(), GpuError> {
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    kernels::prefilter_complex(b, &mut s.ga, &mut s.gscratch, pw, ph, 4)
}

pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n = arg(&args, "--size", 512);
    let reps = arg(&args, "--reps", 10);
    let batches = arg(&args, "--batches", 5);
    let gpu = Gpu::new().unwrap();
    let cfg = Config::default();
    let (w, h) = (n, n);
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let (np, npx) = (pw * ph, w * h);
    let bank = gabor_bank(ph, pw, &cfg);
    let f = bank.filters.len();
    // the reference pack: r random, gradients small, amplitude positive
    let mut rp = values(5 * npx * f, 1, -1.0, 1.0);
    for e in rp.chunks_mut(5) {
        e[4] = e[4].abs() + 0.1;
    }
    let (taps, r) = kernels::moment_taps_buffer(&gpu, cfg.sigma_w).unwrap();
    let plan = MomentPlanGpu::new(&gpu, cfg.shape_order).unwrap();
    let n_global = plan.n_global;
    let mut u: [Buffer; 4] = [0; 4].map(|_| gpu.buffer(4 * npx).unwrap());
    u[0].as_mut_slice::<f32>()
        .copy_from_slice(&values(npx, 3, -0.5, 0.5));
    u[2].as_mut_slice::<f32>()
        .copy_from_slice(&values(npx, 4, -0.5, 0.5));
    let groups = bank
        .filters
        .chunks(4)
        .map(|g| {
            let mut b = gpu.buffer(4 * np * g.len()).unwrap();
            for (d, s) in b.as_mut_slice::<f32>().chunks_mut(np).zip(g) {
                d.copy_from_slice(s);
            }
            b
        })
        .collect();
    let mut s = Bufs {
        refpack: upload(&gpu, &rp),
        responses: upload(&gpu, &values(2 * np * f, 2, -1.0, 1.0)),
        field: [0; 4].map(|_| gpu.buffer(4 * npx).unwrap()),
        u,
        u_next: [0; 4].map(|_| gpu.buffer(4 * npx).unwrap()),
        sums: gpu.buffer(28 * npx).unwrap(),
        global: gpu.buffer(4 * npx * n_global).unwrap(),
        taps,
        r,
        plan,
        energy: gpu.buffer(4 * npx).unwrap(),
        resid: gpu.buffer(4 * npx).unwrap(),
        slope: gpu.buffer(16 * npx).unwrap(),
        erange: gpu.buffer(12 * kernels::block_count(w, h)).unwrap(),
        img: upload(&gpu, &values(npx, 5, 0.0, 1.0)),
        spec: gpu.buffer(8 * np).unwrap(),
        groups,
        ga: gpu.buffer(8 * np * 4).unwrap(),
        gscratch: gpu.buffer(16 * np * 4).unwrap(),
        fft: Fft2Plan::new(&gpu, ph, pw).unwrap(),
    };
    let steps: [(&str, Step); 9] = [
        ("accumulate", accumulate),
        ("moments_global", moments),
        ("solve_blocks", solve),
        ("pad + forward FFT", forward),
        ("filter_multiply x3", multiply),
        ("inverse FFT x3", inverse),
        ("response_splines x3", splines),
        ("prefilter_complex 1", prefilter1),
        ("prefilter_complex 4", prefilter4),
    ];
    let up = std::process::Command::new("uptime").output().unwrap();
    println!("{}", String::from_utf8_lossy(&up.stdout).trim());
    let mut total = 0.0;
    for (name, step) in steps {
        let mut best = f64::INFINITY;
        for _ in 0..batches {
            let (res, t) = gpu
                .batch_timed(|b| (0..reps).try_for_each(|_| step(&mut s, b, w, h, f)))
                .unwrap();
            res.unwrap();
            best = best.min(t / reps as f64);
        }
        total += best;
        println!("{name:<22} {:>8.3} ms", 1e3 * best);
    }
    println!("{:<22} {:>8.3} ms", "sum", 1e3 * total);
}

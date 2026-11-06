//! The fused kernels of the performance pass against the unfused sequence they replace, bit for
//! bit.
#![cfg(target_os = "macos")]
use optflow_gpu::device::{Buffer, Gpu};
use optflow_gpu::kernels;

fn upload(gpu: &Gpu, v: &[f32]) -> Buffer {
    let mut b = gpu.buffer(4 * v.len()).unwrap();
    b.as_mut_slice::<f32>().copy_from_slice(v);
    b
}

/// Deterministic values of mixed sign and magnitude.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            (u - 0.5) * 2000.0
        })
        .collect()
}

fn bits(b: &Buffer) -> Vec<u32> {
    b.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

/// `response_splines` = `scale`, `prefilter_complex`, `interleave_filters`.
#[test]
fn response_splines_equal_scale_prefilter_interleave() {
    let gpu = Gpu::new().unwrap();
    for &(w, h, count, first, nf) in &[
        (40usize, 33usize, 4usize, 4usize, 12usize),
        (17, 64, 3, 0, 3),
        (1, 9, 2, 1, 5),
        (9, 1, 1, 0, 1),
    ] {
        let n = w * h;
        let data = values(2 * n * count, (w * 31 + h) as u64);
        let s = 1.0 / (n as f32);
        // unfused
        let mut a = upload(&gpu, &data);
        let mut scratch = gpu.buffer(16 * n * count).unwrap();
        let mut want = upload(&gpu, &values(2 * n * nf, 7));
        gpu.batch(|b| {
            kernels::scale(b, &mut a, 2 * n * count, s)?;
            kernels::prefilter_complex(b, &mut a, &mut scratch, w, h, count)?;
            kernels::interleave_filters(b, &mut a, w, h, count, first, nf, &mut want)
        })
        .unwrap()
        .unwrap();
        // fused
        let mut a = upload(&gpu, &data);
        let mut scratch = gpu.buffer(16 * n * count).unwrap();
        let mut got = upload(&gpu, &values(2 * n * nf, 7));
        gpu.batch(|b| {
            kernels::response_splines(
                b,
                &mut a,
                &mut scratch,
                (w, h),
                count,
                s,
                (first, nf),
                &mut got,
            )
        })
        .unwrap()
        .unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "{w}x{h} count {count} first {first}"
        );
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The local moments' three-kernel path (window radius up to 24) and the single-kernel path
/// (radius 25) against `local_solve_blocks`, bit for bit.
#[test]
fn local_solve_paths_are_bit_identical_near_the_radius_limit() {
    use optflow_core::gpu_model::df64::Df64;
    use optflow_core::gpu_model::solve::{local_solve_blocks, ModelSolveOut};
    use optflow_core::gpu_support::median;
    use optflow_gpu::kernels::MomentPlanGpu;

    let gpu = Gpu::new().unwrap();
    let (w, h) = (75usize, 61usize);
    let n = w * h;
    let mut rng = Rng(12345);
    let sums: Vec<[f32; 7]> = (0..n)
        .map(|_| {
            let mut s = [0.0f32; 7];
            for _ in 0..4 {
                let wk = rng.next();
                let (gx, gy, d) = (rng.next() - 0.5, rng.next() - 0.5, 0.3 * rng.next());
                s[0] += (wk * gx) * gx;
                s[1] += (wk * gx) * gy;
                s[2] += (wk * gy) * gy;
                s[3] += (wk * gx) * d;
                s[4] += (wk * gy) * d;
                s[5] += (wk * d) * d;
                s[6] += wk;
            }
            s
        })
        .collect();
    let mut ux = Vec::new();
    let mut uy = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f64, y as f64);
            ux.push(Df64::from_f64(0.4 + 0.01 * xf + 0.3 * (xf * 0.09).sin()));
            uy.push(Df64::from_f64(-0.4 + 0.007 * yf + 0.2 * (yf * 0.11).cos()));
        }
    }
    for (sigma, order) in [(5.5, 1), (5.5, 2), (6.0, 1), (6.0, 2), (6.2, 1), (6.2, 2)] {
        let mut want_ux = vec![Df64::from_f32(0.0); n];
        let mut want_uy = vec![Df64::from_f32(0.0); n];
        let mut want_e = vec![0.0f32; n];
        let mut want_r = vec![0.0f32; n];
        let mut want_s = vec![[0.0f32; 4]; n];
        local_solve_blocks(
            &sums,
            (&ux, &uy),
            w,
            h,
            sigma,
            order,
            ModelSolveOut {
                ux: &mut want_ux,
                uy: &mut want_uy,
                energy: &mut want_e,
                resid: &mut want_r,
                slope: &mut want_s,
            },
        );
        let mut flat = vec![0.0f32; 7 * n];
        for (i, s) in sums.iter().enumerate() {
            for (k, v) in s.iter().enumerate() {
                flat[k * n + i] = *v;
            }
        }
        let mut sb = upload(&gpu, &flat);
        let (mut taps, r) = kernels::moment_taps_buffer(&gpu, sigma).unwrap();
        let mut plan = MomentPlanGpu::new(&gpu, order).unwrap();
        let mut global = gpu.buffer(4 * n * plan.n_global).unwrap();
        let hi = |u: &[Df64]| u.iter().map(|d| d.hi).collect::<Vec<_>>();
        let lo = |u: &[Df64]| u.iter().map(|d| d.lo).collect::<Vec<_>>();
        let mut ui = [
            upload(&gpu, &hi(&ux)),
            upload(&gpu, &lo(&ux)),
            upload(&gpu, &hi(&uy)),
            upload(&gpu, &lo(&uy)),
        ];
        let mut uo = [0; 4].map(|_| gpu.buffer(4 * n).unwrap());
        let mut energy = gpu.buffer(4 * n).unwrap();
        let mut resid = gpu.buffer(4 * n).unwrap();
        let mut slope = gpu.buffer(16 * n).unwrap();
        let mut erange = gpu.buffer(12 * kernels::block_count(w, h)).unwrap();
        let [i0, i1, i2, i3] = &mut ui;
        let [o0, o1, o2, o3] = &mut uo;
        gpu.batch(|b| {
            kernels::moments_global(b, &mut sb, w, h, &mut taps, r, &mut plan, &mut global)?;
            kernels::solve_blocks(
                b,
                &mut sb,
                &mut global,
                [&mut *i0, &mut *i1, &mut *i2, &mut *i3],
                w,
                h,
                &mut taps,
                r,
                &mut plan,
                [&mut *o0, &mut *o1, &mut *o2, &mut *o3],
                &mut energy,
                &mut resid,
                &mut slope,
                &mut erange,
            )
        })
        .unwrap()
        .unwrap();
        let e: Vec<f64> = energy
            .as_slice::<f32>()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let threshold = 1e-9 * median(&e) + 1e-30;
        gpu.batch(|b| {
            kernels::reset_below(
                b,
                &mut energy,
                threshold,
                [&mut *i0, &mut *i1, &mut *i2, &mut *i3],
                [&mut *o0, &mut *o1, &mut *o2, &mut *o3],
                &mut resid,
                &mut slope,
                n,
            )
        })
        .unwrap()
        .unwrap();
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        let (gxh, gxl) = (o0.as_slice::<f32>(), o1.as_slice::<f32>());
        let (gyh, gyl) = (o2.as_slice::<f32>(), o3.as_slice::<f32>());
        let (ge, gr, gs) = (
            energy.as_slice::<f32>(),
            resid.as_slice::<f32>(),
            slope.as_slice::<f32>(),
        );
        let mut bad = 0;
        let mut solved = 0;
        for i in 0..n {
            let ok = same(gxh[i], want_ux[i].hi)
                && same(gxl[i], want_ux[i].lo)
                && same(gyh[i], want_uy[i].hi)
                && same(gyl[i], want_uy[i].lo)
                && same(ge[i], want_e[i])
                && same(gr[i], want_r[i])
                && (0..4).all(|k| same(gs[k * n + i], want_s[i][k]));
            bad += usize::from(!ok);
            solved += usize::from(want_r[i] != std::f64::consts::PI as f32);
        }
        println!("sigma {sigma} (r {r}) order {order}: {solved} solved, {bad} differ");
        assert!(solved > n / 2);
        assert_eq!(bad, 0, "sigma {sigma} order {order}");
    }
}

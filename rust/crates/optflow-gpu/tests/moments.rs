//! Global window moments, the block-anchored local solve and the median reset against
//! `gpu_model::solve::local_solve_blocks`, bit for bit.
#![cfg(target_os = "macos")]
use optflow_core::gpu_model::df64::Df64;
use optflow_core::gpu_model::solve::{
    local_solve_blocks, local_solve_blocks_ap, ModelAperture, ModelSolveOut,
};
use optflow_core::gpu_support::{median, Aperture};
use optflow_gpu::device::{Buffer, Gpu};
use optflow_gpu::kernels::{self, MomentPlanGpu, SolveAperture};

fn upload<T: bytemuck::Pod>(gpu: &Gpu, v: &[T]) -> Buffer {
    let mut b = gpu.buffer(std::mem::size_of_val(v)).unwrap();
    b.as_mut_slice::<T>().copy_from_slice(v);
    b
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn sym(&mut self) -> f32 {
        2.0 * self.next() - 1.0
    }
}

/// Per-pixel sums `[gxx, gxy, gyy, hx, hy, s_phiphi, s_w]` of a few random "filters" with the
/// formulas of the accumulate kernel (positive semi-definite `G`).
fn synth_sums(w: usize, h: usize, seed: u64) -> Vec<[f32; 7]> {
    let mut rng = Rng(seed);
    (0..w * h)
        .map(|_| {
            let mut s = [0.0f32; 7];
            for _ in 0..4 {
                let wk = rng.next();
                let (gx, gy, dphi) = (rng.sym(), rng.sym(), 0.3 * rng.sym());
                s[0] += (wk * gx) * gx;
                s[1] += (wk * gx) * gy;
                s[2] += (wk * gy) * gy;
                s[3] += (wk * gx) * dphi;
                s[4] += (wk * gy) * dphi;
                s[5] += (wk * dphi) * dphi;
                s[6] += wk;
            }
            s
        })
        .collect()
}

fn field(w: usize, h: usize, big: bool) -> (Vec<Df64>, Vec<Df64>) {
    let base = if big { 300.0 } else { 0.4 };
    let mut ux = Vec::new();
    let mut uy = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f64, y as f64);
            ux.push(Df64::from_f64(
                base + 0.01 * xf - 0.004 * yf + 0.3 * (xf * 0.09).sin(),
            ));
            uy.push(Df64::from_f64(
                -base + 0.007 * yf + 0.002 * xf + 0.2 * (yf * 0.11).cos(),
            ));
        }
    }
    (ux, uy)
}

fn planes(u: &[Df64]) -> (Vec<f32>, Vec<f32>) {
    (
        u.iter().map(|d| d.hi).collect(),
        u.iter().map(|d| d.lo).collect(),
    )
}

struct Run {
    ux: Vec<Df64>,
    uy: Vec<Df64>,
    energy: Vec<f32>,
    resid: Vec<f32>,
    slope: Vec<[f32; 4]>,
    erange: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    gpu: &Gpu,
    sums: &[[f32; 7]],
    u: (&[Df64], &[Df64]),
    (w, h): (usize, usize),
    sigma: f64,
    order: u32,
    reset: bool,
    ap: Option<SolveAperture>,
) -> Run {
    let n = w * h;
    let mut flat = vec![0.0f32; 7 * n];
    for (i, s) in sums.iter().enumerate() {
        for (k, v) in s.iter().enumerate() {
            flat[k * n + i] = *v;
        }
    }
    let mut sb = upload(gpu, &flat);
    let (mut taps, r) = kernels::moment_taps_buffer(gpu, sigma).unwrap();
    let mut plan = MomentPlanGpu::new(gpu, order).unwrap();
    let mut global = gpu.buffer(4 * n * plan.n_global).unwrap();
    let (xh, xl) = planes(u.0);
    let (yh, yl) = planes(u.1);
    let mut ui = [
        upload(gpu, &xh),
        upload(gpu, &xl),
        upload(gpu, &yh),
        upload(gpu, &yl),
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
        let (i, o) = (
            [&mut *i0, &mut *i1, &mut *i2, &mut *i3],
            [&mut *o0, &mut *o1, &mut *o2, &mut *o3],
        );
        let outs = (&mut energy, &mut resid, &mut slope, &mut erange);
        match ap {
            Some(a) => kernels::solve_blocks_ap(
                b,
                &mut sb,
                &mut global,
                i,
                w,
                h,
                &mut taps,
                r,
                &mut plan,
                o,
                outs,
                a,
            ),
            _ => kernels::solve_blocks(
                b,
                &mut sb,
                &mut global,
                i,
                w,
                h,
                &mut taps,
                r,
                &mut plan,
                o,
                outs.0,
                outs.1,
                outs.2,
                outs.3,
            ),
        }
    })
    .unwrap()
    .unwrap();
    if reset {
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
    }
    let df = |a: &Buffer, b: &Buffer| -> Vec<Df64> {
        a.as_slice::<f32>()
            .iter()
            .zip(b.as_slice::<f32>())
            .map(|(&hi, &lo)| Df64 { hi, lo })
            .collect()
    };
    let sl = slope.as_slice::<f32>();
    Run {
        ux: df(&uo[0], &uo[1]),
        uy: df(&uo[2], &uo[3]),
        energy: energy.as_slice::<f32>().to_vec(),
        resid: resid.as_slice::<f32>().to_vec(),
        slope: (0..n)
            .map(|i| [sl[i], sl[n + i], sl[2 * n + i], sl[3 * n + i]])
            .collect(),
        erange: erange.as_slice::<f32>().to_vec(),
    }
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// `nan`: 0 none, 1 NaN displacements (also at block centres), 2 also NaN sums (the median is
/// then NaN and the reset takes every pixel, as in the model).
fn check(w: usize, h: usize, sigma: f64, order: u32, big: bool, nan: u8, seed: u64) {
    let gpu = Gpu::new().unwrap();
    let mut sums = synth_sums(w, h, seed);
    let (mut ux, uy) = field(w, h, big);
    if nan >= 1 {
        ux[w * 3 + 3].hi = f32::NAN;
        ux[w * 8 + 8].hi = f32::NAN; // the centre of the first block (w, h >= 16)
        ux[w * 24 % (w * h) + 24 % w].lo = f32::NAN;
    }
    if nan == 2 {
        sums[w * 5 + 7] = [f32::NAN; 7];
        sums[w * (h / 2) + w / 2] = [f32::NAN; 7];
    }
    let n = w * h;
    let mut want = Run {
        ux: vec![Df64::from_f32(0.0); n],
        uy: vec![Df64::from_f32(0.0); n],
        energy: vec![0.0; n],
        resid: vec![0.0; n],
        slope: vec![[0.0; 4]; n],
        erange: Vec::new(),
    };
    local_solve_blocks(
        &sums,
        (&ux, &uy),
        w,
        h,
        sigma,
        order,
        ModelSolveOut {
            ux: &mut want.ux,
            uy: &mut want.uy,
            energy: &mut want.energy,
            resid: &mut want.resid,
            slope: &mut want.slope,
        },
    );
    let got = run_gpu(&gpu, &sums, (&ux, &uy), (w, h), sigma, order, true, None);
    let mut bad = [0usize; 4];
    let mut solved = 0;
    for i in 0..n {
        let eq = |a: Df64, b: Df64| same(a.hi, b.hi) && same(a.lo, b.lo);
        bad[0] += usize::from(!(eq(got.ux[i], want.ux[i]) && eq(got.uy[i], want.uy[i])));
        bad[1] += usize::from(!same(got.energy[i], want.energy[i]));
        bad[2] += usize::from(!same(got.resid[i], want.resid[i]));
        bad[3] += usize::from(!(0..4).all(|k| same(got.slope[i][k], want.slope[i][k])));
        solved += usize::from(want.resid[i] != std::f64::consts::PI as f32);
    }
    println!("{w}x{h} sigma {sigma} order {order}: {solved} solved, differing u/energy/resid/slope {bad:?}");
    if nan == 0 || sigma <= 4.0 && nan == 1 {
        assert!(solved > n / 2, "the test needs solved pixels");
    }
    assert_eq!(bad, [0; 4]);
    // energy range per block folded like `energy_range`
    let fold = got
        .erange
        .chunks(3)
        .fold((f64::INFINITY, f64::NEG_INFINITY, false), |a, c| {
            (
                a.0.min(f64::from(c[0])),
                a.1.max(f64::from(c[1])),
                a.2 | (c[2] != 0.0),
            )
        });
    let e64: Vec<f64> = want.energy.iter().map(|&e| f64::from(e)).collect();
    let cpu = e64.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, false),
        |(lo, hi, nn), &e| (lo.min(e), hi.max(e), nn | e.is_nan()),
    );
    assert_eq!(fold, cpu);
    assert_eq!(fold.2, nan == 2, "NaN energy flag");
}

/// The aperture prototypes (`OPTFLOW_APERTURE_CONF`, `OPTFLOW_APERTURE_RIDGE`) against the model:
/// the displacement, energy and slope bit for bit, and the residual too where only the ridge is
/// on (the confidence factor goes through `log`, whose precise Metal version need not round as
/// Rust's `ln`: there within 4 ulp). Settings this large make most pixels take them.
fn check_aperture(order: u32, conf: f64, ridge: f64, seed: u64) {
    let gpu = Gpu::new().unwrap();
    let (w, h, sigma) = (120, 100, 4.0);
    // gradients squeezed towards x by a factor from 0.05 (line-like) to 1 across the image
    let mut sums = synth_sums(w, h, seed);
    for (i, s) in sums.iter_mut().enumerate() {
        let k = 0.05 + 0.95 * (i % w) as f32 / w as f32;
        (s[1], s[2], s[4]) = (k * s[1], k * k * s[2], k * s[4]);
    }
    let (ux, uy) = field(w, h, false);
    let n = w * h;
    let ap = SolveAperture {
        conf,
        ridge,
        tol2: 0.16,
    };
    let solve = |ap: Option<ModelAperture>| {
        let mut r = Run {
            ux: vec![Df64::from_f32(0.0); n],
            uy: vec![Df64::from_f32(0.0); n],
            energy: vec![0.0; n],
            resid: vec![0.0; n],
            slope: vec![[0.0; 4]; n],
            erange: Vec::new(),
        };
        let out = ModelSolveOut {
            ux: &mut r.ux,
            uy: &mut r.uy,
            energy: &mut r.energy,
            resid: &mut r.resid,
            slope: &mut r.slope,
        };
        local_solve_blocks_ap(
            &sums,
            (&ux, &uy),
            w,
            h,
            sigma,
            order,
            ap.unwrap_or_default(),
            out,
        );
        r
    };
    let want = solve(Some(ModelAperture {
        ap: Aperture { conf, ridge },
        tol2: 0.16,
    }));
    let plain = solve(None);
    let got = run_gpu(
        &gpu,
        &sums,
        (&ux, &uy),
        (w, h),
        sigma,
        order,
        true,
        Some(ap),
    );
    let eq = |a: Df64, b: Df64| same(a.hi, b.hi) && same(a.lo, b.lo);
    let (mut bad, mut changed) = ([0usize; 4], [0usize; 2]);
    for i in 0..n {
        bad[0] += usize::from(!(eq(got.ux[i], want.ux[i]) && eq(got.uy[i], want.uy[i])));
        bad[1] += usize::from(!same(got.energy[i], want.energy[i]));
        let (g, e) = (got.resid[i], want.resid[i]);
        let ulps = (g.to_bits() as i64 - e.to_bits() as i64).unsigned_abs();
        bad[2] += usize::from(if conf > 0.0 { ulps > 4 } else { !same(g, e) });
        bad[3] += usize::from(!(0..4).all(|k| same(got.slope[i][k], want.slope[i][k])));
        changed[0] += usize::from(!eq(want.ux[i], plain.ux[i]) || !eq(want.uy[i], plain.uy[i]));
        changed[1] += usize::from(!same(want.resid[i], plain.resid[i]));
    }
    println!("aperture conf {conf} ridge {ridge} order {order}: changed u/resid {changed:?}, differing u/energy/resid/slope {bad:?}");
    assert_eq!(bad, [0; 4]);
    assert!(
        changed[1] > n / 4,
        "the settings should change most residuals"
    );
    if ridge > 0.0 {
        assert!(changed[0] > n / 4, "the ridge should move most pixels");
    } else {
        assert_eq!(
            changed[0], 0,
            "the confidence factor leaves the displacement"
        );
    }
}

#[test]
fn aperture_prototypes_match_the_model() {
    check_aperture(1, 0.3, 0.0, 12);
    check_aperture(1, 0.0, 0.3, 13);
    check_aperture(2, 0.3, 0.3, 14);
}

#[test]
fn order1_is_bit_identical() {
    check(120, 100, 4.0, 1, false, 0, 1);
}

#[test]
fn order2_is_bit_identical() {
    check(120, 100, 4.0, 2, false, 0, 2);
}

#[test]
fn large_displacements_and_nan_sums_are_bit_identical() {
    check(110, 90, 4.0, 1, true, 1, 3);
    check(110, 90, 4.0, 2, true, 1, 4);
    check(110, 90, 4.0, 1, true, 2, 9);
}

#[test]
fn window_wider_than_the_image_is_bit_identical() {
    check(37, 29, 12.0, 2, true, 0, 5);
    check(37, 29, 12.0, 1, true, 1, 10);
}

#[test]
fn largest_sigma_is_bit_identical() {
    check(40, 35, 256.0, 1, true, 0, 6);
    check(33, 21, 256.0, 2, false, 0, 7);
    check(33, 21, 256.0, 2, true, 1, 11);
}

#[test]
fn large_image_is_bit_identical() {
    check(1216, 400, 4.0, 2, true, 0, 8);
}

#[test]
fn reset_below_resets_only_the_low_energy_pixels() {
    let gpu = Gpu::new().unwrap();
    let n = 64;
    let mut energy = upload(&gpu, &(0..n).map(|i| i as f32).collect::<Vec<_>>());
    let mut e = energy.as_mut_slice::<f32>().to_vec();
    e[5] = f32::NAN;
    energy.as_mut_slice::<f32>().copy_from_slice(&e);
    let src = |v: f32| upload(&gpu, &vec![v; n]);
    let mut ui = [src(1.0), src(0.5), src(2.0), src(0.25)];
    let mut uo = [src(9.0), src(9.0), src(9.0), src(9.0)];
    let mut resid = src(0.1);
    let mut slope = upload(&gpu, &vec![7.0f32; 4 * n]);
    let [i0, i1, i2, i3] = &mut ui;
    let [o0, o1, o2, o3] = &mut uo;
    // 10.5 as f64 between f32 values 10 and 11: pixels 0..=10 reset
    gpu.batch(|b| {
        kernels::reset_below(
            b,
            &mut energy,
            10.5,
            [i0, i1, i2, i3],
            [o0, o1, o2, o3],
            &mut resid,
            &mut slope,
            n,
        )
    })
    .unwrap()
    .unwrap();
    for i in 0..n {
        let reset = i <= 10 || i == 5;
        assert_eq!(
            resid.as_slice::<f32>()[i] == std::f64::consts::PI as f32,
            reset,
            "{i}"
        );
        assert_eq!(uo[0].as_slice::<f32>()[i] == 1.0, reset);
        assert_eq!(uo[3].as_slice::<f32>()[i] == 0.25, reset);
        for k in 0..4 {
            assert_eq!(slope.as_slice::<f32>()[k * n + i] == 0.0, reset);
        }
    }
}

#[test]
fn wrappers_reject_bad_arguments() {
    let gpu = Gpu::new().unwrap();
    assert!(MomentPlanGpu::new(&gpu, 3).is_err());
    assert!(kernels::moment_taps_buffer(&gpu, 256.5).is_err());
    assert!(kernels::moment_taps_buffer(&gpu, 0.0).is_err());
    let (mut taps, r) = kernels::moment_taps_buffer(&gpu, 4.0).unwrap();
    let mut plan = MomentPlanGpu::new(&gpu, 1).unwrap();
    let mut sums = gpu.buffer(28 * 16).unwrap();
    let mut out = gpu.buffer(4 * 16 * plan.n_global).unwrap();
    let mut small = gpu.buffer(8).unwrap();
    gpu.batch(|b| {
        assert!(
            kernels::moments_global(b, &mut sums, 4, 4, &mut taps, r, &mut plan, &mut out).is_ok()
        );
        assert!(
            kernels::moments_global(b, &mut sums, 0, 4, &mut taps, r, &mut plan, &mut out).is_err()
        );
        assert!(
            kernels::moments_global(b, &mut sums, 4, 4, &mut taps, r + 1, &mut plan, &mut out)
                .is_err()
        );
        assert!(
            kernels::moments_global(b, &mut sums, 4, 4, &mut taps, r, &mut plan, &mut small)
                .is_err()
        );
        assert!(
            kernels::moments_global(b, &mut small, 4, 4, &mut taps, r, &mut plan, &mut out)
                .is_err()
        );
    })
    .unwrap();
}

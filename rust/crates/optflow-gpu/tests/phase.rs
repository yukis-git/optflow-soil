//! Reference phase, bank responses and accumulate kernels against their CPU oracles.
#![cfg(target_os = "macos")]
use num_complex::Complex32;
use optflow_core::config::Config;
use optflow_core::fft::Fft2;
use optflow_core::gabor::{gabor_bank, Bank, PAD};
use optflow_core::gpu_model::field::spline_coeffs_centred;
use optflow_core::gpu_support::{model_pixel_sums, ref_pack, responses};
use optflow_core::image::Image;
use optflow_gpu::device::{Buffer, Gpu};
use optflow_gpu::fft::Fft2Plan;
use optflow_gpu::kernels;

// phase.metal's PHASE_PAD
const _: () = assert!(PAD == 32);

#[path = "../../optflow-core/tests/common/synth.rs"]
mod synth;

fn upload<T: bytemuck::Pod>(gpu: &Gpu, v: &[T]) -> Buffer {
    let mut b = gpu.buffer(std::mem::size_of_val(v)).unwrap();
    b.as_mut_slice::<T>().copy_from_slice(v);
    b
}

fn speckle(w: usize, h: usize, seed: u64) -> Image<f32> {
    let p = synth::make_pattern(w, h, seed, 0.0);
    synth::render(&p, w, h, None)
}

/// pad -> forward FFT -> the padded spectrum in a buffer.
fn spectrum(gpu: &Gpu, plan: &Fft2Plan, img: &Image<f32>) -> Buffer {
    let (pw, ph) = (img.w + 2 * PAD, img.h + 2 * PAD);
    let mut src = upload(gpu, &img.data);
    let mut spec = gpu.buffer(8 * pw * ph).unwrap();
    gpu.batch(|b| {
        kernels::pad_reflect_complex(b, &mut src, img.w, img.h, &mut spec)?;
        plan.forward(b, &mut spec, 1)
    })
    .unwrap()
    .unwrap();
    spec
}

fn filters_buf(gpu: &Gpu, bank: &Bank) -> Buffer {
    let all: Vec<f32> = bank.filters.iter().flatten().copied().collect();
    upload(gpu, &all)
}

fn freqs_buf(gpu: &Gpu, bank: &Bank) -> Buffer {
    let all: Vec<f32> = bank.fx.iter().chain(&bank.fy).copied().collect();
    upload(gpu, &all)
}

/// The GPU reference pack of `img` (the flow of the pipeline).
fn gpu_ref_pack(gpu: &Gpu, img: &Image<f32>, bank: &Bank, split: bool) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let np = pw * ph;
    let f = bank.filters.len();
    let plan = Fft2Plan::new(gpu, ph, pw).unwrap();
    let mut spec = spectrum(gpu, &plan, img);
    let mut filt = filters_buf(gpu, bank);
    let mut freqs = freqs_buf(gpu, bank);
    let mut r = gpu.buffer(8 * np * f).unwrap();
    let mut rx = gpu.buffer(8 * np * f).unwrap();
    let mut ry = gpu.buffer(8 * np * f).unwrap();
    let mut out = gpu.buffer(4 * w * h * f * 5).unwrap();
    let s = 1.0 / (np as f32);
    gpu.batch(|b| {
        kernels::filter_multiply(b, &mut spec, &mut filt, np, f, &mut r)?;
        kernels::filter_multiply_deriv(b, &mut spec, &mut filt, &mut freqs, 0, pw, ph, f, &mut rx)?;
        kernels::filter_multiply_deriv(b, &mut spec, &mut filt, &mut freqs, 1, pw, ph, f, &mut ry)?;
        for buf in [&mut r, &mut rx, &mut ry] {
            plan.inverse(b, buf, f)?;
            kernels::scale(b, buf, 2 * np * f, s)?;
        }
        if split {
            return Ok(());
        }
        kernels::reference_pack(b, &mut r, &mut rx, &mut ry, w, h, f, 0, f, &mut out)
    })
    .unwrap()
    .unwrap();
    if split {
        // two groups of 6 filters, each from its own buffers
        assert_eq!(f % 2, 0);
        let half = 2 * np * (f / 2);
        for g in 0..2 {
            let part = |buf: &Buffer| upload(gpu, &buf.as_slice::<f32>()[g * half..(g + 1) * half]);
            let (mut a, mut bx, mut by) = (part(&r), part(&rx), part(&ry));
            gpu.batch(|b| {
                kernels::reference_pack(
                    b,
                    &mut a,
                    &mut bx,
                    &mut by,
                    w,
                    h,
                    f / 2,
                    g * (f / 2),
                    f,
                    &mut out,
                )
            })
            .unwrap()
            .unwrap();
        }
    }
    out.as_slice::<f32>().to_vec()
}

/// The GPU responses (node layout, padded grid) of `img`.
fn gpu_responses(gpu: &Gpu, img: &Image<f32>, bank: &Bank, split: bool) -> Vec<Complex32> {
    let (pw, ph) = (img.w + 2 * PAD, img.h + 2 * PAD);
    let np = pw * ph;
    let f = bank.filters.len();
    let plan = Fft2Plan::new(gpu, ph, pw).unwrap();
    let mut spec = spectrum(gpu, &plan, img);
    let mut filt = filters_buf(gpu, bank);
    let mut p = gpu.buffer(8 * np * f).unwrap();
    let mut scratch = gpu.buffer(16 * np * f).unwrap();
    let mut out = gpu.buffer(8 * np * f).unwrap();
    let s = 1.0 / (np as f32);
    gpu.batch(|b| {
        kernels::filter_multiply(b, &mut spec, &mut filt, np, f, &mut p)?;
        plan.inverse(b, &mut p, f)?;
        kernels::scale(b, &mut p, 2 * np * f, s)?;
        kernels::prefilter_complex(b, &mut p, &mut scratch, pw, ph, f)?;
        if split {
            return Ok(());
        }
        kernels::interleave_filters(b, &mut p, pw, ph, f, 0, f, &mut out)
    })
    .unwrap()
    .unwrap();
    if split {
        // two groups of 6 planes, each from its own buffer, to exercise `first`
        assert_eq!(f % 2, 0);
        let half = 2 * np * (f / 2);
        for g in 0..2 {
            let mut part = upload(gpu, &p.as_slice::<f32>()[g * half..(g + 1) * half]);
            gpu.batch(|b| {
                kernels::interleave_filters(b, &mut part, pw, ph, f / 2, g * (f / 2), f, &mut out)
            })
            .unwrap()
            .unwrap();
        }
    }
    out.as_slice::<Complex32>().to_vec()
}

fn bank_for(w: usize, h: usize) -> (Bank, Fft2) {
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    (gabor_bank(ph, pw, &Config::default()), Fft2::new(ph, pw))
}

#[test]
fn reference_pack_matches_the_cpu() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (128, 128);
    let img = speckle(w, h, 5);
    let (bank, fft) = bank_for(w, h);
    let f = bank.filters.len();
    assert_eq!(f, 12);
    let want = ref_pack(&img, &bank, &fft);
    let got = gpu_ref_pack(&gpu, &img, &bank, false);
    assert_eq!(got.len(), want.len());
    // RMS error over the pixels, relative to the RMS of the wanted values, per filter and
    // quantity (r.re and r.im together); gx, gy, amp only where |r| >= 0.3 * median |r|
    // (gx divides by |r|^2). The worst filter counts.
    let mut worst = [0.0f64; 4];
    for k in 0..f {
        let at = |v: &[f32], i: usize| -> [f64; 5] {
            let e = &v[(i * f + k) * 5..][..5];
            [e[0], e[1], e[2], e[3], e[4]].map(f64::from)
        };
        let mut amps: Vec<f64> = (0..w * h)
            .map(|i| at(&want, i)[0].hypot(at(&want, i)[1]))
            .collect();
        amps.sort_by(|a, b| a.total_cmp(b));
        let med = amps[amps.len() / 2];
        let (mut num, mut den) = ([0.0f64; 4], [0.0f64; 4]);
        for i in 0..w * h {
            let (g, c) = (at(&got, i), at(&want, i));
            let mask = c[0].hypot(c[1]) >= 0.3 * med;
            let mut acc = |slot: usize, gv: f64, cv: f64| {
                num[slot] += (gv - cv) * (gv - cv);
                den[slot] += cv * cv;
            };
            acc(0, g[0], c[0]);
            acc(0, g[1], c[1]);
            if mask {
                acc(1, g[2], c[2]);
                acc(1, g[3], c[3]);
                acc(2, g[4], c[4]);
            }
        }
        for v in 0..3 {
            worst[v] = worst[v].max((num[v] / den[v]).sqrt());
        }
    }
    println!(
        "worst RMS-relative error: r {:e}, gx/gy {:e}, amp {:e}",
        worst[0], worst[1], worst[2]
    );
    // the brief's 1e-6 is below what rustfft's own f32 transform reaches against f64 (1.2e-6, see
    // `gpu_reference_response_is_as_close_to_f64_as_rustfft`); 3e-6 is the FFT difference
    assert!(worst[0] <= 3e-6, "r: {worst:?}");
    assert!(worst[1] <= 1e-5 && worst[2] <= 1e-5, "{worst:?}");
}

#[test]
fn responses_match_the_cpu() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (128, 128);
    let img = speckle(w, h, 6);
    let (bank, fft) = bank_for(w, h);
    let f = bank.filters.len();
    let want = responses(&img, &bank, &fft);
    let got = gpu_responses(&gpu, &img, &bank, false);
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    let n = want.len() / f;
    for k in 0..f {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..n {
            num += f64::from((got[i * f + k] - want[i * f + k]).norm_sqr());
            den += f64::from(want[i * f + k].norm_sqr());
        }
        worst = worst.max((num / den).sqrt());
    }
    println!("worst RMS-relative coefficient error: {worst:e}");
    // FFT-bound as for r (the prefilter itself is bit-identical)
    assert!(worst <= 5e-6, "{worst:e}");
}

/// `reference_values` of the CPU (f64, rounded once), as in `gabor.rs`.
fn cpu_values(r: Complex32, rx: Complex32, ry: Complex32) -> [f32; 3] {
    let (re, im) = (f64::from(r.re), f64::from(r.im));
    let a2 = re * re + im * im + 1e-12;
    let grad = |d: Complex32| (re * f64::from(d.im) - im * f64::from(d.re)) / a2;
    [grad(rx) as f32, grad(ry) as f32, a2.sqrt() as f32]
}

#[test]
fn reference_values_agree_with_f64_for_identical_inputs() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (40, 30);
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let np = pw * ph;
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    // amplitudes from 1e-7 to 1e1
    let mut gen = |scale_exp: bool| -> Vec<Complex32> {
        (0..np)
            .map(|_| {
                let s = if scale_exp {
                    10f32.powf(rnd() * 4.0 - 3.0)
                } else {
                    1.0
                };
                Complex32::new(rnd() * s, rnd() * s)
            })
            .collect()
    };
    let r = gen(true);
    let rx = gen(false);
    let ry = gen(false);
    let mut rb = upload(&gpu, &r);
    let mut rxb = upload(&gpu, &rx);
    let mut ryb = upload(&gpu, &ry);
    let mut out = gpu.buffer(4 * w * h * 5).unwrap();
    gpu.batch(|b| kernels::reference_pack(b, &mut rb, &mut rxb, &mut ryb, w, h, 1, 0, 1, &mut out))
        .unwrap()
        .unwrap();
    let got = out.as_slice::<f32>();
    let mut differ = 0;
    let mut max_ulp = 0u32;
    for y in 0..h {
        for x in 0..w {
            let q = (y + PAD) * pw + x + PAD;
            let want = cpu_values(r[q], rx[q], ry[q]);
            let g = &got[(y * w + x) * 5..][..5];
            assert_eq!((g[0], g[1]), (r[q].re, r[q].im));
            for v in 0..3 {
                let (a, b) = (g[2 + v], want[v]);
                if a.to_bits() != b.to_bits() && !(a == 0.0 && b == 0.0) {
                    differ += 1;
                    let ulp = (f64::from(a) - f64::from(b)).abs()
                        / f64::from(f32::from_bits(b.abs().to_bits() + 1) - b.abs());
                    max_ulp = max_ulp.max(ulp.ceil() as u32);
                }
            }
        }
    }
    println!("{differ} of {} values differ, max {max_ulp} ulp", 3 * w * h);
    assert_eq!(differ, 0, "{differ} differ, {max_ulp} ulp");
}

fn to_df(v: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let hi: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    let lo = v
        .iter()
        .zip(&hi)
        .map(|(&x, &h)| (x - f64::from(h)) as f32)
        .collect();
    (hi, lo)
}

/// accumulate on the GPU against `pixel_sums` for the same inputs.
fn check_accumulate(
    gpu: &Gpu,
    w: usize,
    h: usize,
    u: impl Fn(usize, usize) -> (f64, f64),
    name: &str,
) {
    let img1 = speckle(w, h, 21);
    let img2 = speckle(w, h, 22);
    let (bank, fft) = bank_for(w, h);
    let f = bank.filters.len();
    let refpack = ref_pack(&img1, &bank, &fft);
    let resp = responses(&img2, &bank, &fft);
    let sx = Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| 1.5 * ((i % w) as f32 * 0.07).sin() + 0.3 * ((i / w) as f32 * 0.11).cos())
            .collect(),
    );
    let sy = Image::from_vec(
        w,
        h,
        (0..w * h)
            .map(|i| -2.0 * ((i / w) as f32 * 0.05).sin() + 0.2 * ((i % w) as f32 * 0.13).sin())
            .collect(),
    );
    let csx = spline_coeffs_centred(&sx);
    let csy = spline_coeffs_centred(&sy);
    let mut ux = vec![0.0f64; w * h];
    let mut uy = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            (ux[y * w + x], uy[y * w + x]) = u(x, y);
        }
    }
    let (mut uxh, uxl) = to_df(&ux);
    let (mut uyh, uyl) = to_df(&uy);
    // non-finite positions
    for (i, (a, b)) in [(NAN_AT, f32::NAN), (INF_AT, f32::INFINITY)]
        .into_iter()
        .enumerate()
    {
        uxh[a % (w * h)] = b;
        uyh[(a * 3 + i) % (w * h)] = b;
    }
    let pairs = |hi: &[f32], lo: &[f32]| -> Vec<(f32, f32)> {
        hi.iter().copied().zip(lo.iter().copied()).collect()
    };
    let want = model_pixel_sums(
        refpack.clone(),
        resp.clone(),
        f,
        [&sx, &sy, &csx, &csy],
        &pairs(&uxh, &uxl),
        &pairs(&uyh, &uyl),
    );
    let mut rp = upload(gpu, &refpack);
    let mut rs = upload(gpu, &resp);
    let mut bsx = upload(gpu, &sx.data);
    let mut bsy = upload(gpu, &sy.data);
    let mut bcx = upload(gpu, &csx.data);
    let mut bcy = upload(gpu, &csy.data);
    let (mut b1, mut b2, mut b3, mut b4) = (
        upload(gpu, &uxh),
        upload(gpu, &uxl),
        upload(gpu, &uyh),
        upload(gpu, &uyl),
    );
    let mut sums = gpu.buffer(28 * w * h).unwrap();
    gpu.batch(|b| {
        kernels::accumulate(
            b, &mut rp, &mut rs, f, &mut bsx, &mut bsy, &mut bcx, &mut bcy, &mut b1, &mut b2,
            &mut b3, &mut b4, w, h, &mut sums,
        )
    })
    .unwrap()
    .unwrap();
    let got = sums.as_slice::<f32>();
    let n = w * h;
    let mut bad_px = 0;
    let mut nan_px = 0;
    for (i, wv) in want.iter().enumerate() {
        if wv[0].is_nan() {
            nan_px += 1;
        }
        let same = (0..7).all(|s| {
            got[s * n + i].to_bits() == wv[s].to_bits()
                || (got[s * n + i].is_nan() && wv[s].is_nan())
        });
        if !same {
            bad_px += 1;
        }
    }
    println!("{name} {w}x{h}: {bad_px} of {n} pixels differ, {nan_px} NaN pixels");
    assert!(nan_px >= 2, "{name}: the test needs NaN pixels");
    assert_eq!(bad_px, 0, "{name}: {bad_px} of {n} pixels differ");
}

const NAN_AT: usize = 1234;
const INF_AT: usize = 2345;

#[test]
fn accumulate_is_bit_identical_to_pixel_sums() {
    let gpu = Gpu::new().unwrap();
    check_accumulate(
        &gpu,
        96,
        80,
        |x, y| {
            (
                0.8 * (x as f64 * 0.09).sin() + 0.25,
                0.6 * (y as f64 * 0.07).cos() - 0.4,
            )
        },
        "small u",
    );
}

#[test]
fn accumulate_with_large_displacements_is_bit_identical() {
    let gpu = Gpu::new().unwrap();
    check_accumulate(
        &gpu,
        384,
        320,
        |x, y| {
            (
                (x as f64 - 192.0) * 1.5 + 0.1234567,
                300.0 - 1.7 * y as f64 + 0.37 * (x as f64 * 0.01).sin(),
            )
        },
        "u ~ 300",
    );
}

#[test]
fn wrappers_reject_bad_arguments() {
    let gpu = Gpu::new().unwrap();
    let mut tiny = gpu.buffer(16).unwrap();
    let mut tiny2 = gpu.buffer(16).unwrap();
    let mut tiny3 = gpu.buffer(16).unwrap();
    let mut tiny4 = gpu.buffer(16).unwrap();
    gpu.batch(|b| {
        assert!(kernels::pad_reflect_complex(b, &mut tiny, 8, 8, &mut tiny2).is_err());
        assert!(kernels::pad_reflect_complex(b, &mut tiny, 0, 8, &mut tiny2).is_err());
        assert!(kernels::filter_multiply(b, &mut tiny, &mut tiny2, 8, 1, &mut tiny3).is_err());
        assert!(kernels::filter_multiply(b, &mut tiny, &mut tiny2, 1, 0, &mut tiny3).is_err());
        assert!(kernels::scale(b, &mut tiny, 8, 2.0).is_err());
        assert!(kernels::interleave_filters(b, &mut tiny, 4, 4, 2, 11, 12, &mut tiny2).is_err());
        assert!(kernels::reference_pack(
            b, &mut tiny, &mut tiny2, &mut tiny3, 4, 4, 1, 0, 1, &mut tiny4
        )
        .is_err());
    })
    .unwrap();
}

/// The responses `r` of every filter in f64 (rustfft f64 on the f32 filters), per filter the
/// cropped w x h values.
fn f64_responses(img: &Image<f32>, bank: &Bank) -> Vec<Vec<num_complex::Complex64>> {
    use num_complex::Complex64;
    use rustfft::FftPlanner;
    let (w, h) = (img.w, img.h);
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let mirror = |i: i64, n: i64| -> usize {
        let period = 2 * (n - 1);
        let m = i.rem_euclid(period);
        (if m >= n { period - m } else { m }) as usize
    };
    let mut data: Vec<Complex64> = (0..pw * ph)
        .map(|i| {
            let (x, y) = (
                mirror((i % pw) as i64 - PAD as i64, w as i64),
                mirror((i / pw) as i64 - PAD as i64, h as i64),
            );
            Complex64::new(f64::from(img.data[y * w + x]), 0.0)
        })
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft2 = |data: &mut Vec<Complex64>, inverse: bool, planner: &mut FftPlanner<f64>| {
        let (rp, cp) = if inverse {
            (planner.plan_fft_inverse(pw), planner.plan_fft_inverse(ph))
        } else {
            (planner.plan_fft_forward(pw), planner.plan_fft_forward(ph))
        };
        for row in data.chunks_mut(pw) {
            rp.process(row);
        }
        for x in 0..pw {
            let mut col: Vec<Complex64> = (0..ph).map(|y| data[y * pw + x]).collect();
            cp.process(&mut col);
            for y in 0..ph {
                data[y * pw + x] = col[y];
            }
        }
    };
    fft2(&mut data, false, &mut planner);
    bank.filters
        .iter()
        .map(|g| {
            let mut p: Vec<Complex64> = data
                .iter()
                .zip(g)
                .map(|(&s, &g)| s * f64::from(g))
                .collect();
            fft2(&mut p, true, &mut planner);
            let sc = 1.0 / (pw * ph) as f64;
            (0..w * h)
                .map(|i| p[(i / w + PAD) * pw + i % w + PAD] * sc)
                .collect()
        })
        .collect()
}

#[test]
fn gpu_reference_response_is_as_close_to_f64_as_rustfft() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (128, 128);
    let img = speckle(w, h, 5);
    let (bank, fft) = bank_for(w, h);
    let f = bank.filters.len();
    let exact = f64_responses(&img, &bank);
    let cpu = ref_pack(&img, &bank, &fft);
    let got = gpu_ref_pack(&gpu, &img, &bank, false);
    let (mut worst_gpu, mut worst_cpu) = (0.0f64, 0.0f64);
    for k in 0..f {
        let err = |v: &[f32]| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..w * h {
                let e = &v[(i * f + k) * 5..];
                let d = num_complex::Complex64::new(f64::from(e[0]), f64::from(e[1])) - exact[k][i];
                num += d.norm_sqr();
                den += exact[k][i].norm_sqr();
            }
            (num / den).sqrt()
        };
        worst_gpu = worst_gpu.max(err(&got));
        worst_cpu = worst_cpu.max(err(&cpu));
    }
    println!(
        "worst RMS-relative error of r vs f64: gpu {worst_gpu:e}, cpu (rustfft f32) {worst_cpu:e}"
    );
    assert!(
        worst_gpu <= 2.0 * worst_cpu.max(1e-7),
        "gpu {worst_gpu:e} cpu {worst_cpu:e}"
    );
}

#[test]
fn split_filter_groups_give_the_single_group_result() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (64, 48);
    let img = speckle(w, h, 9);
    let (bank, _) = bank_for(w, h);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&gpu_ref_pack(&gpu, &img, &bank, false)),
        bits(&gpu_ref_pack(&gpu, &img, &bank, true))
    );
    let (a, b) = (
        gpu_responses(&gpu, &img, &bank, false),
        gpu_responses(&gpu, &img, &bank, true),
    );
    assert_eq!(
        bits(bytemuck::cast_slice(&a)),
        bits(bytemuck::cast_slice(&b))
    );
}

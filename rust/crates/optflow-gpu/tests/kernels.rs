//! Spline prefilters, centred blur and warp kernels against their CPU oracles.
#![cfg(target_os = "macos")]
use num_complex::Complex32;
use optflow_core::gpu_model::df64::Df64;
use optflow_core::gpu_model::field::{blur_centred, spline_coeffs_centred, warp_exact};
use optflow_core::image::Image;
use optflow_core::interp::{warp, BSpline};
use optflow_gpu::device::{Buffer, Gpu};
use optflow_gpu::kernels;

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

fn big_field(w: usize, h: usize) -> Vec<f64> {
    (0..w * h)
        .map(|i| {
            300.0 + 0.01 * (i % w) as f64 - 0.003 * (i / w) as f64
                + 0.2 * ((i % w) as f64 * 0.05).sin()
        })
        .collect()
}

/// Number of differing values and the largest difference in ulps of the wanted value.
fn near_ties(got: &[f32], want: &[f32]) -> (usize, u32) {
    let mut n = 0;
    let mut max_ulp = 0;
    for (g, w) in got.iter().zip(want) {
        if g.to_bits() != w.to_bits() {
            n += 1;
            let ulp = (f64::from(*g) - f64::from(*w)).abs()
                / f64::from(f32::from_bits(w.abs().to_bits() + 1) - w.abs());
            max_ulp = max_ulp.max(ulp.ceil() as u32);
        }
    }
    (n, max_ulp)
}

#[test]
fn prefilter_real_matches_bspline_new() {
    let gpu = Gpu::new().unwrap();
    for (w, h) in [(96, 80), (100, 64), (67, 131)] {
        let img = speckle(w, h, 3);
        let want = BSpline::new(&img).coeffs;
        let mut src = upload(&gpu, &img.data);
        let mut scratch = gpu.buffer(8 * w * h).unwrap();
        let mut out = gpu.buffer(4 * w * h).unwrap();
        gpu.batch(|b| kernels::prefilter_real(b, &mut src, &mut scratch, &mut out, w, h))
            .unwrap()
            .unwrap();
        let (n, max_ulp) = near_ties(out.as_slice::<f32>(), &want.data);
        println!(
            "{w}x{h}: {n} of {} coefficients differ, max {max_ulp} ulp",
            w * h
        );
        assert!(
            n * 10_000 <= w * h && max_ulp <= 1,
            "{w}x{h}: {n} differ, {max_ulp} ulp"
        );
    }
}

#[test]
fn prefilter_complex_matches_bspline_new() {
    let gpu = Gpu::new().unwrap();
    let (w, h, count) = (90, 72, 12);
    let planes: Vec<Image<Complex32>> = (0..count)
        .map(|k| {
            let a = speckle(w, h, 10 + k as u64);
            let b = speckle(w, h, 50 + k as u64);
            Image::from_vec(
                w,
                h,
                a.data
                    .iter()
                    .zip(&b.data)
                    .map(|(&re, &im)| Complex32::new(re, im - 0.5))
                    .collect(),
            )
        })
        .collect();
    let all: Vec<Complex32> = planes.iter().flat_map(|p| p.data.iter().copied()).collect();
    let mut buf = upload(&gpu, &all);
    let mut scratch = gpu.buffer(16 * w * h * count).unwrap();
    gpu.batch(|b| kernels::prefilter_complex(b, &mut buf, &mut scratch, w, h, count))
        .unwrap()
        .unwrap();
    let got = buf.as_slice::<Complex32>();
    for (k, p) in planes.iter().enumerate() {
        let want = BSpline::new(p).coeffs;
        let g = &got[k * w * h..(k + 1) * w * h];
        let (re, im): (Vec<f32>, Vec<f32>) = want.data.iter().map(|c| (c.re, c.im)).unzip();
        let (gre, gim): (Vec<f32>, Vec<f32>) = g.iter().map(|c| (c.re, c.im)).unzip();
        for (lane, (got, want)) in [(&gre, &re), (&gim, &im)].into_iter().enumerate() {
            let (n, max_ulp) = near_ties(got, want);
            println!(
                "plane {k} lane {lane}: {n} of {} differ, max {max_ulp} ulp",
                w * h
            );
            assert!(
                n * 10_000 <= w * h && max_ulp <= 1,
                "plane {k} lane {lane}: {n}, {max_ulp} ulp"
            );
        }
    }
}

#[test]
fn prefilter_field_is_bit_identical_to_the_model() {
    let gpu = Gpu::new().unwrap();
    for (w, h) in [(70, 50), (128, 96)] {
        let s = Image::from_vec(w, h, big_field(w, h).iter().map(|&v| v as f32).collect());
        let want = spline_coeffs_centred(&s);
        let mut src = upload(&gpu, &s.data);
        let mut out = gpu.buffer(4 * w * h).unwrap();
        gpu.batch(|b| kernels::prefilter_field(b, &mut src, &mut out, w, h))
            .unwrap()
            .unwrap();
        let differ = out
            .as_slice::<f32>()
            .iter()
            .zip(&want.data)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differ, 0, "{w}x{h}: {differ} coefficients differ");
    }
}

#[test]
fn blur_field_is_bit_identical_to_the_model() {
    let gpu = Gpu::new().unwrap();
    for (w, h, sigma) in [
        (70, 50, 4.0),
        (128, 96, 2.5),
        (64, 64, 0.8),
        (40, 30, 256.0), // 2049 taps, far longer than the image
    ] {
        let u: Vec<Df64> = big_field(w, h).iter().map(|&v| Df64::from_f64(v)).collect();
        let want = blur_centred(&u, w, h, sigma);
        let hi: Vec<f32> = u.iter().map(|d| d.hi).collect();
        let lo: Vec<f32> = u.iter().map(|d| d.lo).collect();
        let (mut bh, mut bl) = (upload(&gpu, &hi), upload(&gpu, &lo));
        let mut taps = kernels::blur_taps(&gpu, sigma).unwrap();
        let mut mid = gpu.buffer(8 * w * h).unwrap();
        let mut out = gpu.buffer(4 * w * h).unwrap();
        gpu.batch(|b| {
            kernels::blur_field(b, &mut bh, &mut bl, &mut taps, &mut mid, &mut out, w, h)
        })
        .unwrap()
        .unwrap();
        let differ = out
            .as_slice::<f32>()
            .iter()
            .zip(&want.data)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differ, 0, "{w}x{h} sigma {sigma}: {differ} pixels differ");
    }
}

fn run_warp(
    gpu: &Gpu,
    coeffs: &Image<f32>,
    sx: &Image<f32>,
    sy: &Image<f32>,
    shift: (i64, i64),
) -> Vec<f32> {
    let (w, h) = (sx.w, sx.h);
    let mut c = upload(gpu, &coeffs.data);
    let (mut bx, mut by) = (upload(gpu, &sx.data), upload(gpu, &sy.data));
    let mut out = gpu.buffer(4 * w * h).unwrap();
    gpu.batch(|b| {
        kernels::warp(
            b,
            &mut c,
            (coeffs.w, coeffs.h),
            &mut bx,
            &mut by,
            (w, h),
            shift,
            &mut out,
        )
    })
    .unwrap()
    .unwrap();
    out.as_slice::<f32>().to_vec()
}

#[test]
fn warp_is_bit_identical_to_warp_exact() {
    let gpu = Gpu::new().unwrap();
    // window (w x h) inside a larger image 2 (w2 x h2), fractional, large and out-of-range shifts
    let (w2, h2, w, h) = (160, 130, 128, 100);
    let img = speckle(w2, h2, 7);
    let coeffs = BSpline::new(&img).coeffs;
    let n = w * h;
    let mut sx = Image::<f32>::new(w, h);
    let mut sy = Image::<f32>::new(w, h);
    for i in 0..n {
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        sx.data[i] = 6.37 + 0.013 * x - 0.4 * (y * 0.07).sin();
        sy.data[i] = -3.91 + 0.02 * y + 0.3 * (x * 0.05).cos();
    }
    // some extremes: far outside, NaN, exact integers, fractions near 1
    sx.data[5] = 1.0e9;
    sy.data[6] = -1.0e9;
    sx.data[7] = f32::NAN;
    sy.data[8] = f32::INFINITY;
    sx.data[9] = -300.0;
    sx.data[10] = 2.0;
    sy.data[10] = -5.0;
    sx.data[11] = 0.999_999_9;
    sy.data[11] = -0.000_000_1;
    for shift in [(0, 0), (16, 12), (-20, 9), (31, 30)] {
        let want = warp_exact(&coeffs, &sx, &sy, shift);
        let got = run_warp(&gpu, &coeffs, &sx, &sy, shift);
        let differ: Vec<usize> = (0..n)
            .filter(|&i| {
                got[i].to_bits() != want.data[i].to_bits()
                    && !(got[i].is_nan() && want.data[i].is_nan())
            })
            .collect();
        assert!(
            differ.len() * 100_000 <= n,
            "shift {shift:?}: {} of {n} pixels differ, first {:?}",
            differ.len(),
            differ.first().map(|&i| (i, got[i], want.data[i]))
        );
        println!("shift {shift:?}: {} differing pixels", differ.len());
    }
}

#[test]
fn warp_with_zero_shift_is_within_one_ulp_of_interp_warp() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (120, 96);
    let img = speckle(w, h, 21);
    let coeffs = BSpline::new(&img).coeffs;
    let mut ux = Image::<f32>::new(w, h);
    let mut uy = Image::<f32>::new(w, h);
    for i in 0..w * h {
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        ux.data[i] = 1.7 + 0.01 * x - 0.2 * (y * 0.1).sin();
        uy.data[i] = -2.3 + 0.015 * y + 0.2 * (x * 0.08).cos();
    }
    let want = warp(&img, &ux, &uy);
    let got = run_warp(&gpu, &coeffs, &ux, &uy, (0, 0));
    for (i, (g, wv)) in got.iter().zip(&want.data).enumerate() {
        let ulp = f32::from_bits(wv.abs().to_bits() + 1) - wv.abs();
        assert!((g - wv).abs() <= ulp, "pixel {i}: {g} vs {wv}");
    }
}

/// Prefilter of a w x h image on the GPU with the real, complex and field kernels against their
/// oracles (degenerate and large sizes).
fn check_prefilters(gpu: &Gpu, w: usize, h: usize) {
    let img = speckle(w, h, 5);
    let want = BSpline::new(&img).coeffs;
    let mut src = upload(gpu, &img.data);
    let mut scratch = gpu.buffer(8 * w * h).unwrap();
    let mut out = gpu.buffer(4 * w * h).unwrap();
    gpu.batch(|b| kernels::prefilter_real(b, &mut src, &mut scratch, &mut out, w, h))
        .unwrap()
        .unwrap();
    let (n, ulp) = near_ties(out.as_slice::<f32>(), &want.data);
    assert!(
        n * 10_000 <= w * h && ulp <= 1,
        "real {w}x{h}: {n} differ, {ulp} ulp"
    );

    let cdata: Vec<Complex32> = img
        .data
        .iter()
        .map(|&v| Complex32::new(v, 1.0 - v))
        .collect();
    let cwant = BSpline::new(&Image::from_vec(w, h, cdata.clone())).coeffs;
    let mut cb = upload(gpu, &cdata);
    let mut cs = gpu.buffer(16 * w * h).unwrap();
    gpu.batch(|b| kernels::prefilter_complex(b, &mut cb, &mut cs, w, h, 1))
        .unwrap()
        .unwrap();
    let g: Vec<f32> = cb
        .as_slice::<Complex32>()
        .iter()
        .flat_map(|c| [c.re, c.im])
        .collect();
    let wv: Vec<f32> = cwant.data.iter().flat_map(|c| [c.re, c.im]).collect();
    let (n, ulp) = near_ties(&g, &wv);
    assert!(
        n * 10_000 <= 2 * w * h && ulp <= 1,
        "complex {w}x{h}: {n} differ, {ulp} ulp"
    );

    let s = Image::from_vec(w, h, big_field(w, h).iter().map(|&v| v as f32).collect());
    let fwant = spline_coeffs_centred(&s);
    let mut sb = upload(gpu, &s.data);
    let mut fo = gpu.buffer(4 * w * h).unwrap();
    gpu.batch(|b| kernels::prefilter_field(b, &mut sb, &mut fo, w, h))
        .unwrap()
        .unwrap();
    let differ = fo
        .as_slice::<f32>()
        .iter()
        .zip(&fwant.data)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differ, 0, "field {w}x{h}");
}

#[test]
fn prefilters_handle_degenerate_sizes() {
    let gpu = Gpu::new().unwrap();
    for (w, h) in [
        (1, 1),
        (1, 7),
        (7, 1),
        (2, 2),
        (2, 9),
        (9, 2),
        (1, 2),
        (2, 1),
    ] {
        check_prefilters(&gpu, w, h);
    }
}

#[test]
fn prefilters_and_warp_at_the_pipeline_maximum() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (1216, 300);
    check_prefilters(&gpu, w, h);
    let img = speckle(w, h, 9);
    let coeffs = BSpline::new(&img).coeffs;
    let n = w * h;
    let mut sx = Image::<f32>::new(w, h);
    let mut sy = Image::<f32>::new(w, h);
    for i in 0..n {
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        sx.data[i] = 11.3 + 0.01 * x - 0.5 * (y * 0.03).sin();
        sy.data[i] = -7.7 + 0.02 * y + 0.4 * (x * 0.02).cos();
    }
    let want = warp_exact(&coeffs, &sx, &sy, (3, -2));
    let got = run_warp(&gpu, &coeffs, &sx, &sy, (3, -2));
    let differ = (0..n)
        .filter(|&i| got[i].to_bits() != want.data[i].to_bits())
        .count();
    assert!(differ * 100_000 <= n, "{differ} of {n} pixels differ");
}

#[test]
fn undersized_buffers_are_rejected() {
    let gpu = Gpu::new().unwrap();
    let mut small = gpu.buffer(16).unwrap();
    let mut out = gpu.buffer(4 * 64).unwrap();
    let r = gpu
        .batch(|b| kernels::prefilter_field(b, &mut small, &mut out, 8, 8))
        .unwrap();
    assert!(r.is_err());
    let r = gpu
        .batch(|b| kernels::prefilter_field(b, &mut out, &mut small, 0, 8))
        .unwrap();
    assert!(r.is_err());
}

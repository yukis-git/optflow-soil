//! The Metal FFT against an f64 rustfft reference.
#![cfg(target_os = "macos")]
use num_complex::Complex32;
use optflow_gpu::device::Gpu;
use optflow_gpu::fft::{factorise, Factorisation, Fft2Plan};
use rustfft::FftPlanner;

fn reference_2d(h: usize, w: usize, data: &[Complex32], inverse: bool) -> Vec<Complex32> {
    let mut p = FftPlanner::<f64>::new();
    let (row, col) = if inverse {
        (p.plan_fft_inverse(w), p.plan_fft_inverse(h))
    } else {
        (p.plan_fft_forward(w), p.plan_fft_forward(h))
    };
    let mut a: Vec<num_complex::Complex64> = data
        .iter()
        .map(|c| num_complex::Complex64::new(c.re.into(), c.im.into()))
        .collect();
    for r in a.chunks_mut(w) {
        row.process(r);
    }
    for x in 0..w {
        let mut c: Vec<_> = (0..h).map(|y| a[y * w + x]).collect();
        col.process(&mut c);
        for y in 0..h {
            a[y * w + x] = c[y];
        }
    }
    a.iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect()
}

fn test_data(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| {
            Complex32::new(
                ((i * 7919) % 1000) as f32 / 500.0 - 1.0,
                ((i as f32) * 0.013).sin(),
            )
        })
        .collect()
}

/// RMS error of the GPU transform relative to the RMS of the f64 reference.
fn rel_error(gpu: &Gpu, h: usize, w: usize, inverse: bool) -> f64 {
    let n = h * w;
    let data = test_data(n);
    let want = reference_2d(h, w, &data, inverse);
    let plan = Fft2Plan::new(gpu, h, w).unwrap();
    let mut buf = gpu.buffer(8 * n).unwrap();
    buf.as_mut_slice::<Complex32>().copy_from_slice(&data);
    gpu.batch(|b| {
        if inverse {
            plan.inverse(b, &mut buf, 1)
        } else {
            plan.forward(b, &mut buf, 1)
        }
    })
    .unwrap()
    .unwrap();
    let got = buf.as_slice::<Complex32>();
    let norm = (want.iter().map(|c| c.norm_sqr() as f64).sum::<f64>() / n as f64).sqrt();
    let err = (got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).norm_sqr() as f64)
        .sum::<f64>()
        / n as f64)
        .sqrt();
    err / norm.max(1e-30)
}

#[test]
fn factorisation_covers_the_pipeline_sizes() {
    assert!(matches!(factorise(576), Factorisation::Direct(_))); // 2^6 3^2
    assert!(matches!(factorise(1216), Factorisation::Direct(_))); // 2^6 19
    assert!(matches!(factorise(1061), Factorisation::Bluestein { .. })); // prime
    match factorise(1061) {
        Factorisation::Bluestein { n, m } => assert_eq!((n, m), (1061, 4096)),
        _ => unreachable!(),
    }
    match factorise(576) {
        Factorisation::Direct(r) => assert_eq!(r.iter().product::<usize>(), 576),
        _ => unreachable!(),
    }
}

#[test]
fn fft2_matches_an_f64_reference_within_1e_minus_6_relative() {
    let gpu = Gpu::new().unwrap();
    for &(h, w) in &[
        (64, 64),
        (576, 576),
        (96, 1216),
        (784, 120),
        (67, 1061),
        (1, 37),
        (37, 1),
        (1152, 1216),
    ] {
        for inverse in [false, true] {
            let rel = rel_error(&gpu, h, w, inverse);
            eprintln!("{h}x{w} inverse={inverse}: rel {rel:.2e}");
            assert!(rel <= 1e-6, "{h}x{w} inverse={inverse}: rel {rel:.2e}");
        }
    }
}

#[test]
fn planes_are_independent_and_count_selects_how_many() {
    let gpu = Gpu::new().unwrap();
    let (h, w) = (24, 40);
    let n = h * w;
    let plan = Fft2Plan::new(&gpu, h, w).unwrap();
    let planes: Vec<Vec<Complex32>> = (0..3)
        .map(|k| test_data(n).iter().map(|c| c * (k as f32 + 1.0)).collect())
        .collect();
    let mut buf = gpu.buffer(8 * n * 3).unwrap();
    for (k, p) in planes.iter().enumerate() {
        buf.as_mut_slice::<Complex32>()[k * n..(k + 1) * n].copy_from_slice(p);
    }
    // transform only the first two planes
    gpu.batch(|b| plan.forward(b, &mut buf, 2))
        .unwrap()
        .unwrap();
    let got = buf.as_slice::<Complex32>();
    for (k, p) in planes.iter().enumerate().take(2) {
        let want = reference_2d(h, w, p, false);
        let norm: f64 = want.iter().map(|c| c.norm_sqr() as f64).sum::<f64>().sqrt();
        let err: f64 = got[k * n..(k + 1) * n]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).norm_sqr() as f64)
            .sum::<f64>()
            .sqrt();
        assert!(err <= 1e-6 * norm, "plane {k}: {:.2e}", err / norm);
    }
    assert_eq!(&got[2 * n..], &planes[2][..], "third plane untouched");
}

/// Rough timing: `cargo test --release -p optflow-gpu --test fft -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_12_inverse_576_squared() {
    let gpu = Gpu::new().unwrap();
    let plan = Fft2Plan::new(&gpu, 576, 576).unwrap();
    let mut buf = gpu.buffer(8 * 576 * 576 * 12).unwrap();
    buf.as_mut_slice::<Complex32>()
        .copy_from_slice(&test_data(576 * 576 * 12));
    for _ in 0..2 {
        gpu.batch(|b| plan.inverse(b, &mut buf, 12))
            .unwrap()
            .unwrap();
    }
    let t = std::time::Instant::now();
    let reps = 10;
    for _ in 0..reps {
        gpu.batch(|b| plan.inverse(b, &mut buf, 12))
            .unwrap()
            .unwrap();
    }
    eprintln!(
        "12 inverse 576x576: {:.2} ms per batch",
        t.elapsed().as_secs_f64() * 1e3 / reps as f64
    );
}

#[test]
fn every_length_uses_the_smallest_threadgroup_variant() {
    use optflow_gpu::fft::kernel_name_for;
    assert_eq!(kernel_name_for(1024, false), "fft_lines_direct_1k");
    assert_eq!(kernel_name_for(1100, false), "fft_lines_direct_2k");
    assert_eq!(kernel_name_for(2048, false), "fft_lines_direct_2k");
    assert_eq!(kernel_name_for(2049, false), "fft_lines_direct_4k");
    assert_eq!(kernel_name_for(2048, true), "fft_lines_bluestein_2k");
    assert_eq!(kernel_name_for(4096, true), "fft_lines_bluestein_4k");
}

/// Relative RMS error of the 2-D FFT against the f64 reference, for the lengths of the pipeline
/// (the full-frame tiles: 6240 x 4160 in 1024 tiles with the default 64 margin, plus 2 x 32 of
/// reflect padding, gives 192, 224, 1152 and 1216; the 1024² bench gives 1088) and some awkward
/// ones; the new passes must be no worse than 1.5 x the error of the previous implementation
/// (recorded below, measured at e6cb85c by this test's `--nocapture` output before the change,
/// the larger of the two directions).
#[test]
fn fft2_error_is_no_worse_than_before_on_the_pipeline_sizes() {
    let gpu = Gpu::new().unwrap();
    // (h, w, previous relative RMS error)
    let cases: &[(usize, usize, f64)] = &[
        (1152, 1152, 1.840e-7),
        (1216, 1152, 1.937e-7),
        (1216, 1216, 1.923e-7),
        (192, 1216, 1.843e-7),
        (1216, 224, 1.767e-7),
        (1088, 1088, 1.800e-7),
        (576, 576, 1.832e-7),
        (1061, 96, 1.896e-7),
    ];
    for &(h, w, before) in cases {
        for inverse in [false, true] {
            let rel = rel_error(&gpu, h, w, inverse);
            println!("{h}x{w} inverse={inverse}: rel {rel:.3e} (before {before:.3e})");
            assert!(
                rel <= 1.5 * before,
                "{h}x{w} inverse={inverse}: {rel:.3e} > 1.5 x {before:.3e}"
            );
        }
    }
}

/// Every column pass (strided, tiles of 2 to 16 columns, transposed) against the f64
/// reference, including tiles past the last column, single-stage lines and Bluestein rows.
#[test]
fn every_column_pass_matches_the_f64_reference() {
    use optflow_gpu::fft::ColumnPass;
    let gpu = Gpu::new().unwrap();
    let passes = [
        ColumnPass::Strided,
        ColumnPass::Tiled(2),
        ColumnPass::Tiled(4),
        ColumnPass::Tiled(8),
        ColumnPass::Tiled(16),
        ColumnPass::Transposed,
    ];
    for &(h, w) in &[
        (8, 40),
        (5, 7),
        (24, 40),
        (192, 100),
        (224, 1061),
        (576, 64),
        (1216, 40),
        (1061, 24),
        (1, 37),
        (37, 1),
        // every register kernel variant and odd-prime stage: 1001 = 7 11 13 (r19 1k, tiles of
        // 2), 248 = 8 31 (r31 1k, tiles of 16), 208 = 16 13, 513 = 27 19, 3328 = 2^8 13 (r19 4k),
        // 2208 = 2^5 3 23 (r31 4k), 4096 (r8 4k), 1984 = 2^6 31 (r31 2k), 116 = 4 29
        (1001, 8),
        (248, 8),
        (208, 8),
        (513, 8),
        (4, 3328),
        (4, 2208),
        (8, 4096),
        (8, 1984),
        (116, 8),
    ] {
        let n = h * w;
        let data = test_data(n);
        for inverse in [false, true] {
            let want = reference_2d(h, w, &data, inverse);
            let norm = (want.iter().map(|c| c.norm_sqr() as f64).sum::<f64>() / n as f64).sqrt();
            for pass in passes {
                let Ok(plan) = Fft2Plan::with_column_pass(&gpu, h, w, pass) else {
                    // tiles that do not fit the length (or a Bluestein column) are refused
                    assert!(matches!(pass, ColumnPass::Tiled(_)), "{h}x{w} {pass:?}");
                    continue;
                };
                let mut buf = gpu.buffer(8 * n).unwrap();
                buf.as_mut_slice::<Complex32>().copy_from_slice(&data);
                gpu.batch(|b| {
                    if inverse {
                        plan.inverse(b, &mut buf, 1)
                    } else {
                        plan.forward(b, &mut buf, 1)
                    }
                })
                .unwrap()
                .unwrap();
                let got = buf.as_slice::<Complex32>();
                let err = (got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).norm_sqr() as f64)
                    .sum::<f64>()
                    / n as f64)
                    .sqrt();
                let rel = err / norm.max(1e-30);
                assert!(
                    rel <= 1e-6,
                    "{h}x{w} {pass:?} inverse={inverse}: rel {rel:.2e}"
                );
            }
        }
    }
}

#[test]
fn the_default_column_pass_tiles_the_short_columns() {
    use optflow_gpu::fft::ColumnPass;
    let gpu = Gpu::new().unwrap();
    let pass = |h: usize| Fft2Plan::new(&gpu, h, 64).unwrap().column_pass();
    assert_eq!(pass(192), ColumnPass::Tiled(16)); // full-frame bottom tiles
    assert_eq!(pass(256), ColumnPass::Tiled(16));
    assert_eq!(pass(512), ColumnPass::Tiled(8));
    assert_eq!(pass(576), ColumnPass::Tiled(4));
    assert_eq!(pass(1024), ColumnPass::Strided); // 8 8 8 2: 512 threads, 4 x 512 > 1024
    assert_eq!(pass(1152), ColumnPass::Strided); // 576 threads: no tile of 4 fits
    assert_eq!(pass(1216), ColumnPass::Strided); // longer than 1024
    assert_eq!(pass(1061), ColumnPass::Strided); // Bluestein
}

/// The same input gives bit-identical output, run after run (no order-dependent reductions).
#[test]
fn fft2_is_deterministic() {
    let gpu = Gpu::new().unwrap();
    for &(h, w) in &[(1088, 1088), (192, 1216), (576, 576), (67, 1061)] {
        let n = h * w;
        let data = test_data(4 * n);
        let plan = Fft2Plan::new(&gpu, h, w).unwrap();
        let run = || {
            let mut buf = gpu.buffer(8 * 4 * n).unwrap();
            buf.as_mut_slice::<Complex32>().copy_from_slice(&data);
            gpu.batch(|b| {
                plan.forward(b, &mut buf, 4)?;
                plan.inverse(b, &mut buf, 4)
            })
            .unwrap()
            .unwrap();
            buf.as_slice::<u32>().to_vec()
        };
        let a = run();
        let b = run();
        assert!(a == b, "{h}x{w}: two runs differ");
    }
}

#[test]
fn register_kernels_by_length_and_largest_radix() {
    use optflow_gpu::fft::{regs_kernel_name_for, tiled_kernel_name_for};
    assert_eq!(regs_kernel_name_for(1152, 8), "fft_rows_regs_r8_2k");
    assert_eq!(regs_kernel_name_for(1088, 17), "fft_rows_regs_r19_2k");
    assert_eq!(regs_kernel_name_for(1024, 19), "fft_rows_regs_r19_1k");
    assert_eq!(regs_kernel_name_for(4096, 8), "fft_rows_regs_r8_4k");
    assert_eq!(regs_kernel_name_for(23 * 64, 23), "fft_rows_regs_r31_2k");
    assert_eq!(tiled_kernel_name_for(7), "fft_cols_tiled_r8");
    assert_eq!(tiled_kernel_name_for(31), "fft_cols_tiled_r31");
}

/// `plan.inverse_filtered` of `count` filters of `spec` (h x w) with `scale`, read back.
fn run_inverse_filtered(
    gpu: &Gpu,
    h: usize,
    w: usize,
    spec: &[Complex32],
    filt: &[f32],
    count: usize,
    scale: f32,
) -> Vec<Complex32> {
    let plan = Fft2Plan::new(gpu, h, w).unwrap();
    inverse_filtered_with(gpu, &plan, h * w, spec, filt, count, scale)
}

fn inverse_filtered_with(
    gpu: &Gpu,
    plan: &Fft2Plan,
    n: usize,
    spec: &[Complex32],
    filt: &[f32],
    count: usize,
    scale: f32,
) -> Vec<Complex32> {
    let mut sb = gpu.buffer(8 * n).unwrap();
    sb.as_mut_slice::<Complex32>().copy_from_slice(spec);
    let mut fb = gpu.buffer(4 * n * count).unwrap();
    fb.as_mut_slice::<f32>().copy_from_slice(filt);
    let mut out = gpu.buffer(8 * n * count).unwrap();
    gpu.batch(|b| plan.inverse_filtered(b, &mut sb, &mut fb, n, count, &mut out, scale))
        .unwrap()
        .unwrap();
    out.as_slice::<Complex32>().to_vec()
}

/// sqrt of the sum of squares of the difference and of the reference.
fn rel_parts(got: &[Complex32], want: &[Complex32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).norm_sqr() as f64)
        .sum::<f64>()
        .sqrt();
    let norm = want.iter().map(|c| c.norm_sqr() as f64).sum::<f64>().sqrt();
    (err, norm)
}

#[test]
fn inverse_filtered_equals_multiply_then_inverse_then_scale_within_rounding() {
    let gpu = Gpu::new().unwrap();
    let (h, w, count) = (96, 80, 4);
    let n = h * w;
    let spec = test_data(n);
    let filt: Vec<f32> = (0..count * n)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    // reference: f64 multiply, f64 inverse DFT, scale 1/n
    let mut want = Vec::new();
    for k in 0..count {
        let prod: Vec<Complex32> = (0..n).map(|i| spec[i] * filt[k * n + i]).collect();
        let r = reference_2d(h, w, &prod, true);
        want.extend(r.iter().map(|c| c / n as f32));
    }
    let got = run_inverse_filtered(&gpu, h, w, &spec, &filt, count, 1.0 / n as f32);
    let (err, norm) = rel_parts(&got, &want);
    assert!(err <= 1e-6 * norm, "{:.2e}", err / norm);
}

/// The fused transform gives the same bits as `filter_multiply`, `inverse` and then `x * scale`
/// on every f32 (the order `response_splines` used), on every path: register, tiled, strided
/// and transposed columns, Bluestein rows and columns, single lines and a single value.
#[test]
fn inverse_filtered_is_bit_identical_to_multiply_inverse_then_scale() {
    use optflow_gpu::fft::ColumnPass;
    use optflow_gpu::kernels;
    let gpu = Gpu::new().unwrap();
    let passes = [
        ColumnPass::Strided,
        ColumnPass::Tiled(2),
        ColumnPass::Tiled(4),
        ColumnPass::Tiled(16),
        ColumnPass::Transposed,
    ];
    let count = 3;
    for &(h, w) in &[
        (96, 80),
        (5, 7),
        (192, 1216),
        (576, 64),
        (1088, 16),
        (224, 1061),
        (1061, 24),
        (8, 4096),
        (1, 37),
        (37, 1),
        (1, 1),
        // every fft_rows_regs_mul and fft_lines_bluestein_mul variant: 248 = 8 31 (r31 1k),
        // 1984 = 2^6 31 (r31 2k), 2208 = 2^5 3 23 (r31 4k), 1001 = 7 11 13 (r19 1k),
        // 3328 = 2^8 13 (r19 4k), 1152 (r8 2k), 541 (prime: Bluestein 2k)
        (4, 248),
        (4, 1984),
        (4, 2208),
        (4, 1001),
        (4, 3328),
        (4, 1152),
        (4, 541),
        (248, 8), // columns in fft_cols_tiled_r31
    ] {
        let n = h * w;
        let spec = test_data(n);
        let filt: Vec<f32> = (0..count * n)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.3)
            .collect();
        let scale = 1.0 / n as f32;
        for pass in passes {
            let Ok(plan) = Fft2Plan::with_column_pass(&gpu, h, w, pass) else {
                assert!(matches!(pass, ColumnPass::Tiled(_)), "{h}x{w} {pass:?}");
                continue;
            };
            let got = inverse_filtered_with(&gpu, &plan, n, &spec, &filt, count, scale);
            let mut sb = gpu.buffer(8 * n).unwrap();
            sb.as_mut_slice::<Complex32>().copy_from_slice(&spec);
            let mut fb = gpu.buffer(4 * n * count).unwrap();
            fb.as_mut_slice::<f32>().copy_from_slice(&filt);
            let mut want = gpu.buffer(8 * n * count).unwrap();
            gpu.batch(|b| {
                kernels::filter_multiply(b, &mut sb, &mut fb, n, count, &mut want)?;
                plan.inverse(b, &mut want, count)
            })
            .unwrap()
            .unwrap();
            let want: Vec<u32> = want
                .as_slice::<f32>()
                .iter()
                .map(|x| (x * scale).to_bits())
                .collect();
            let got: Vec<u32> = got
                .iter()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect();
            let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert!(
                diff == 0,
                "{h}x{w} {pass:?}: {diff} of {} values differ",
                got.len()
            );
        }
    }
}

#[test]
fn inverse_filtered_refuses_wrong_sizes() {
    let gpu = Gpu::new().unwrap();
    let (h, w) = (8, 6);
    let n = h * w;
    let plan = Fft2Plan::new(&gpu, h, w).unwrap();
    let mut spec = gpu.buffer(8 * n).unwrap();
    let mut filt = gpu.buffer(4 * n * 2).unwrap();
    let mut out = gpu.buffer(8 * n * 2).unwrap();
    gpu.batch(|b| {
        // np is not h w
        assert!(plan
            .inverse_filtered(b, &mut spec, &mut filt, n + 1, 1, &mut out, 1.0)
            .is_err());
        // three filters do not fit the filter or output buffers
        assert!(plan
            .inverse_filtered(b, &mut spec, &mut filt, n, 3, &mut out, 1.0)
            .is_err());
        // zero filters: nothing to do
        assert!(plan
            .inverse_filtered(b, &mut spec, &mut filt, n, 0, &mut out, 1.0)
            .is_ok());
    })
    .unwrap();
}

#[test]
fn filter_product_kernels_by_length_and_largest_radix() {
    use optflow_gpu::fft::{bluestein_mul_kernel_name_for, regs_mul_kernel_name_for};
    assert_eq!(regs_mul_kernel_name_for(1152, 8), "fft_rows_regs_mul_r8_2k");
    assert_eq!(
        regs_mul_kernel_name_for(1088, 17),
        "fft_rows_regs_mul_r19_2k"
    );
    assert_eq!(
        regs_mul_kernel_name_for(1024, 19),
        "fft_rows_regs_mul_r19_1k"
    );
    assert_eq!(regs_mul_kernel_name_for(4096, 8), "fft_rows_regs_mul_r8_4k");
    assert_eq!(
        regs_mul_kernel_name_for(23 * 64, 23),
        "fft_rows_regs_mul_r31_2k"
    );
    assert_eq!(
        bluestein_mul_kernel_name_for(128),
        "fft_lines_bluestein_mul_1k"
    );
    assert_eq!(
        bluestein_mul_kernel_name_for(2048),
        "fft_lines_bluestein_mul_2k"
    );
    assert_eq!(
        bluestein_mul_kernel_name_for(4096),
        "fft_lines_bluestein_mul_4k"
    );
}

#[test]
fn print_the_threadgroup_limits_of_the_fft_pipelines() {
    use optflow_gpu::fft::{
        bluestein_mul_kernel_name_for, kernel_name_for, regs_kernel_name_for,
        regs_mul_kernel_name_for, tiled_kernel_name_for,
    };
    let gpu = Gpu::new().unwrap();
    for maxr in [8, 19, 31] {
        for len in [1024, 2048, 4096] {
            for name in [
                regs_kernel_name_for(len, maxr),
                regs_mul_kernel_name_for(len, maxr),
            ] {
                eprintln!("{name}: {}", gpu.max_threads(name).unwrap());
            }
        }
        let name = tiled_kernel_name_for(maxr);
        eprintln!("{name}: {}", gpu.max_threads(name).unwrap());
    }
    for len in [1024, 2048, 4096] {
        for name in [
            kernel_name_for(len, true),
            bluestein_mul_kernel_name_for(len),
        ] {
            eprintln!("{name}: {}", gpu.max_threads(name).unwrap());
        }
    }
}

/// Every line length 2..=1280 as the columns (h) and as the rows (w) of a 2-D transform, against
/// the f64 reference: covers each direct/Bluestein plan and each column pass the thread limits
/// of the pipelines can select.
#[test]
fn every_length_up_to_1280_matches_the_f64_reference() {
    let gpu = Gpu::new().unwrap();
    let mut bad = Vec::new();
    for n in 2..=1280usize {
        for (h, w) in [(n, 6), (6, n)] {
            for inverse in [false, true] {
                let rel = rel_error(&gpu, h, w, inverse);
                if rel > 1e-6 || rel.is_nan() {
                    bad.push(format!("{h}x{w} inverse={inverse}: rel {rel:.2e}"));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} failures: {:?}",
        bad.len(),
        &bad[..bad.len().min(20)]
    );
}

//! Parity of the Gabor filter bank, `pad_reflect` and `reference_phase` with the Python reference
//! (`python/optflow_ref/phaseflow.py`). Run with `-- --nocapture` to see the measured errors.
//! `Config` is tested in `parity_config.rs`.
mod common;
use num_complex::Complex32;
use optflow_core::config::Config;
use optflow_core::fft::Fft2;
use optflow_core::gabor::*;
use optflow_core::image::Image;

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    Image::from_vec(w, h, v.iter().map(|&x| x as f32).collect())
}

#[test]
fn gabor_responses_and_gradients_match_python() {
    let img = img_f32("gabor_img");
    let (n_h, n_w) = (img.h, img.w);
    let cfg = Config::default();
    let bank = gabor_bank(n_h + 2 * PAD, n_w + 2 * PAD, &cfg);
    let fft = Fft2::new(bank.h, bank.w);
    let got = reference_phase(&img, &bank, &fft);

    let (shape, r_re) = common::load_f64("gabor_r_re");
    let (_, r_im) = common::load_f64("gabor_r_im");
    let (_, gx_py) = common::load_f64("gabor_gx");
    let (_, gy_py) = common::load_f64("gabor_gy");
    let n_filters = shape[0];
    let px = n_h * n_w;
    assert_eq!(n_filters, 12);
    assert_eq!(got.len(), n_filters);
    assert_eq!(bank.filters.len(), n_filters);

    let (mut worst_rel, mut worst_grad) = (0.0f64, 0.0f64);
    for (k, rp) in got.iter().enumerate() {
        assert_eq!((rp.r.h, rp.r.w), (n_h, n_w));
        let sl = k * px..(k + 1) * px;
        let (re, im) = (&r_re[sl.clone()], &r_im[sl.clone()]);
        let amp_py: Vec<f64> = re
            .iter()
            .zip(im)
            .map(|(a, b)| (a * a + b * b + 1e-12).sqrt())
            .collect();
        let max_amp = amp_py.iter().copied().fold(0.0, f64::max);
        let max_err = (0..px)
            .map(|i| (f64::from(rp.r.data[i].re) - re[i]).hypot(f64::from(rp.r.data[i].im) - im[i]))
            .fold(0.0, f64::max);
        let rel = max_err / max_amp;

        let mut grad = 0.0f64;
        let mut n_strong = 0usize;
        for i in 0..px {
            if amp_py[i] > 0.05 * max_amp {
                n_strong += 1;
                grad = grad
                    .max((f64::from(rp.gx.data[i]) - gx_py[sl.start + i]).abs())
                    .max((f64::from(rp.gy.data[i]) - gy_py[sl.start + i]).abs());
                // amplitude is stored too: compare it with |r_py| on the same pixels
                assert!((f64::from(rp.amp.data[i]) - amp_py[i]).abs() < 1e-4 * max_amp);
            }
        }
        println!(
            "filter {k:2}: max rel response err = {rel:.3e}, max gradient err = {grad:.3e} rad/px ({n_strong} of {px} px, max |r| = {max_amp:.3})"
        );
        assert!(n_strong > px / 20, "filter {k}: too few strong pixels");
        assert!(rel < 1e-4, "filter {k}: response error {rel}");
        assert!(grad < 1e-3, "filter {k}: gradient error {grad}");
        worst_rel = worst_rel.max(rel);
        worst_grad = worst_grad.max(grad);
    }
    println!("worst over 12 filters: rel response {worst_rel:.3e}, gradient {worst_grad:.3e}");
}

#[test]
fn gabor_bank_layout_and_dc_cut() {
    let cfg = Config::default();
    let (h, w) = (45, 70);
    let bank = gabor_bank(h, w, &cfg);
    assert_eq!((bank.h, bank.w), (h, w));
    assert_eq!(bank.filters.len(), cfg.scales.len() * cfg.n_orient);
    assert!(bank.filters.iter().all(|f| f.len() == h * w));
    assert_eq!((bank.fx.len(), bank.fy.len()), (w, h));
    // DC (index 0) is removed by the high-pass factor, filters never exceed 1
    assert!(bank.filters.iter().all(|f| f[0] == 0.0));
    assert!(bank
        .filters
        .iter()
        .flatten()
        .all(|&v| (0.0..=1.0 + 1e-6).contains(&v)));
    // numpy fftfreq: non-negative frequencies first, then the negative ones; 2*pi scaled
    let tau = std::f64::consts::TAU;
    assert_eq!(bank.fx[0], 0.0);
    assert!((f64::from(bank.fx[1]) - tau / 70.0).abs() < 1e-7);
    assert!((f64::from(bank.fx[35]) + tau * 35.0 / 70.0).abs() < 1e-6); // even w: Nyquist is negative
    assert!((f64::from(bank.fx[69]) + tau / 70.0).abs() < 1e-7);
    assert!((f64::from(bank.fy[22]) - tau * 22.0 / 45.0).abs() < 1e-6); // odd h: last positive
    assert!((f64::from(bank.fy[23]) + tau * 22.0 / 45.0).abs() < 1e-6);
}

#[test]
fn gabor_pad_reflect_matches_numpy() {
    // np.pad([[1,2,3],[4,5,6]], 3, mode="reflect"), padding wider than the image
    let img = Image::from_vec(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let p = pad_reflect(&img, 3);
    assert_eq!((p.w, p.h), (9, 8));
    let even = [5.0f32, 6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0, 5.0];
    let odd = [2.0f32, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0];
    for y in 0..8 {
        let want = if y % 2 == 0 { &even } else { &odd };
        assert_eq!(&p.data[y * 9..(y + 1) * 9], want, "row {y}");
    }
    // the edge pixel is not repeated; pad 0 is the identity
    let img = Image::from_vec(4, 3, (1..=12).map(|v| v as f32).collect());
    assert_eq!(pad_reflect(&img, 0), img);
    let p = pad_reflect(&img, 2);
    assert_eq!((p.w, p.h), (8, 7));
    assert_eq!(
        &p.data[2 * 8..3 * 8],
        &[3.0f32, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
    );
    assert_eq!(
        &p.data[..8],
        &[11.0f32, 10.0, 9.0, 10.0, 11.0, 12.0, 11.0, 10.0]
    );
}

/// A plane wave `cos(k . x)` at the centre frequency of a filter has, in the interior, the phase
/// gradient `k` in that filter's response. Checks the x/y axis convention on a non-square image
/// whose sides are not powers of two.
#[test]
fn gabor_plane_wave_phase_gradient_equals_wave_vector() {
    let cfg = Config::default();
    let (h, w) = (45usize, 70usize);
    let bank = gabor_bank(h + 2 * PAD, w + 2 * PAD, &cfg);
    let fft = Fft2::new(bank.h, bank.w);
    let wave = 2.0 * std::f64::consts::PI / (cfg.lam * cfg.scales[1]); // middle scale
    for k in 0..cfg.n_orient {
        let th = std::f64::consts::PI * k as f64 / cfg.n_orient as f64;
        let (kx, ky) = (wave * th.cos(), wave * th.sin());
        let data: Vec<f32> = (0..h * w)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (kx * x + ky * y + 0.3).cos() as f32
            })
            .collect();
        let img = Image::from_vec(w, h, data);
        let rp = &reference_phase(&img, &bank, &fft)[cfg.n_orient + k];
        let mut worst = 0.0f64;
        for y in 12..h - 12 {
            for x in 12..w - 12 {
                let i = y * w + x;
                worst = worst
                    .max((f64::from(rp.gx.data[i]) - kx).abs())
                    .max((f64::from(rp.gy.data[i]) - ky).abs());
            }
        }
        println!(
            "orientation {k}: wave vector ({kx:.4}, {ky:.4}), worst gradient error {worst:.3e}"
        );
        assert!(worst < 1e-2, "orientation {k}: {worst}");
    }
}

#[test]
#[should_panic(expected = "reference_phase")]
fn gabor_reference_phase_rejects_mismatched_bank() {
    let cfg = Config::default();
    let bank = gabor_bank(100, 100, &cfg);
    let img = Image::from_vec(50, 50, vec![0.0f32; 2500]); // needs a 114x114 bank
    let _ = reference_phase(&img, &bank, &Fft2::new(100, 100));
}

#[test]
fn gabor_responses_of_a_constant_image_vanish() {
    let cfg = Config::default();
    let (h, w) = (20, 30);
    let bank = gabor_bank(h + 2 * PAD, w + 2 * PAD, &cfg);
    let fft = Fft2::new(bank.h, bank.w);
    let img = Image::from_vec(w, h, vec![7.0f32; h * w]);
    for rp in reference_phase(&img, &bank, &fft) {
        assert!(rp.r.data.iter().all(|c: &Complex32| c.norm() < 1e-5));
        assert!(rp.amp.data.iter().all(|&a| (a - 1e-6).abs() < 1e-5));
    }
}

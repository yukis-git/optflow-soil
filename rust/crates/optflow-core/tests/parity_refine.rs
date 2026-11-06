//! Parity of the fine stage `phaseflow::refine` (order 1 and 2) with the Python reference
//! (`python/optflow_ref/phaseflow.py`), plus the public helpers `window_moment` and
//! `shape_unknowns`. Run with `-- --nocapture` to see the measured errors; the ignored
//! `refine_speed_report` prints timings (`cargo test --release refine_speed_report -- --ignored
//! --nocapture`, once with `RAYON_NUM_THREADS=1`).
mod common;
use common::bits;
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::interp::gauss_kernel;
use optflow_core::phaseflow::{refine, shape_unknowns, window_moment, Flow};

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    assert_eq!(shape.len(), 2, "{name}: not a 2-D array");
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

struct Inputs {
    a: Image<f32>,
    b: Image<f32>,
    ux: Image<f32>,
    uy: Image<f32>,
}

fn inputs() -> Inputs {
    Inputs {
        a: img_f32("refine_a"),
        b: img_f32("refine_b"),
        ux: img_f32("refine_init_ux"),
        uy: img_f32("refine_init_uy"),
    }
}

fn run(inp: &Inputs, order: u32) -> Flow {
    let cfg = Config {
        shape_order: order,
        ..Config::default()
    };
    refine(&inp.a, &inp.b, &inp.ux, &inp.uy, &cfg)
}

/// Largest |a - b| over the masked pairs; NaN if exactly one of a pair is non-finite.
fn max_abs_diff(a: &[f32], b: &[f64], mask: &[bool]) -> f64 {
    let mut worst = 0.0f64;
    for ((&x, &y), &m) in a.iter().zip(b).zip(mask) {
        if m {
            let d = (f64::from(x) - y).abs();
            if d.is_nan() {
                return f64::NAN;
            }
            worst = worst.max(d);
        }
    }
    worst
}

fn check_order(order: u32) {
    let inp = inputs();
    let flow = run(&inp, order);
    let (h, w) = (inp.a.h, inp.a.w);
    for im in [&flow.ux, &flow.uy, &flow.conf, &flow.resid] {
        assert_eq!((im.w, im.h), (w, h));
    }
    let (_, ux_py) = common::load_f64(&format!("refine_o{order}_ux"));
    let (_, uy_py) = common::load_f64(&format!("refine_o{order}_uy"));
    let (_, conf_py) = common::load_f64(&format!("refine_o{order}_conf"));
    let (_, resid_py) = common::load_f64(&format!("refine_o{order}_resid"));
    let mask: Vec<bool> = conf_py.iter().map(|&c| c > 0.2).collect();
    let n_mask = mask.iter().filter(|&&m| m).count();
    assert!(
        n_mask * 2 > w * h,
        "fixture has too few confident pixels ({n_mask}) for a meaningful comparison"
    );

    let rms_ux = common::rms_diff(&flow.ux.data, &ux_py, Some(&mask));
    let rms_uy = common::rms_diff(&flow.uy.data, &uy_py, Some(&mask));
    let max_u =
        max_abs_diff(&flow.ux.data, &ux_py, &mask).max(max_abs_diff(&flow.uy.data, &uy_py, &mask));
    let rms_conf = common::rms_diff(&flow.conf.data, &conf_py, None);
    let max_conf = max_abs_diff(&flow.conf.data, &conf_py, &vec![true; w * h]);
    let rms_resid = common::rms_diff(&flow.resid.data, &resid_py, Some(&mask));
    let max_resid = max_abs_diff(&flow.resid.data, &resid_py, &mask);
    let rust_conf_hi = flow.conf.data.iter().filter(|&&c| c > 0.2).count();
    println!(
        "order {order}: confident px (conf_py > 0.2) {n_mask}, rust conf > 0.2 {rust_conf_hi}\n  \
         disp rms ux {rms_ux:.3e} uy {rms_uy:.3e} max {max_u:.3e} px\n  \
         conf rms {rms_conf:.3e} max {max_conf:.3e}\n  \
         resid rms {rms_resid:.3e} max {max_resid:.3e} rad"
    );
    assert!(rms_ux < 1e-3, "ux rms {rms_ux}");
    assert!(rms_uy < 1e-3, "uy rms {rms_uy}");
    assert!(rms_conf < 1e-3, "conf rms {rms_conf}");
    assert!(rms_resid < 1e-3, "resid rms {rms_resid}");
}

#[test]
fn refine_order1_matches_python() {
    check_order(1);
}

#[test]
fn refine_order2_matches_python() {
    check_order(2);
}

#[test]
fn refine_result_does_not_depend_on_the_thread_count() {
    let inp = inputs();
    let many = run(&inp, 1);
    let one = common::on_one_thread(|| run(&inp, 1));
    assert_eq!(bits(&many.ux), bits(&one.ux));
    assert_eq!(bits(&many.uy), bits(&one.uy));
    assert_eq!(bits(&many.conf), bits(&one.conf));
    assert_eq!(bits(&many.resid), bits(&one.resid));
}

/// Confidence at the border columns and rows with the guard off (Task 8b): with `edge_guard = 0`
/// the border rule is off and only the matches that leave the second image get confidence 0.
/// Fixture `refine_border` (64 x 64, translation about (-1.5, 1.7), made by
/// `python/bench/make_parity_fixtures.py`): columns 0 and 1 and the last two rows leave the image
/// (landing margins of 0.2 to 0.3 px), columns 2 and 3 stay in. The conf of the border pixels
/// agrees with Python's pixel by pixel, not only in the rms: a step of the landing test that
/// went the other way would show as a difference of 0.5 to 1.
#[test]
fn refine_border_columns_and_rows_match_python_with_the_guard_off() {
    let (a, b) = (img_f32("refine_border_a"), img_f32("refine_border_b"));
    let (w, h) = (a.w, a.h);
    let m = &common::meta()["refine_border"];
    let num = |k: &str, i: usize| m[k][i].as_f64().unwrap();
    // the exact field (a uniform strain about `center` plus the translation `t`), started off by `init_offset`
    let (mut ux, mut uy) = (Image::new(w, h), Image::new(w, h));
    for y in 0..h {
        for x in 0..w {
            let u = m["exx"].as_f64().unwrap() * (x as f64 - num("center", 0))
                + num("t", 0)
                + num("init_offset", 0);
            let v = m["eyy"].as_f64().unwrap() * (y as f64 - num("center", 1))
                + num("t", 1)
                + num("init_offset", 1);
            ux.set(x, y, u as f32);
            uy.set(x, y, v as f32);
        }
    }
    let cfg = Config {
        edge_guard: 0.0,
        ..Config::default()
    };
    let flow = refine(&a, &b, &ux, &uy, &cfg);
    let (_, conf_py) = common::load_f64("refine_border_conf");
    let (_, ux_py) = common::load_f64("refine_border_ux");
    let max_conf = max_abs_diff(&flow.conf.data, &conf_py, &vec![true; w * h]);
    let mask: Vec<bool> = conf_py.iter().map(|&c| c > 0.2).collect();
    let max_u = max_abs_diff(&flow.ux.data, &ux_py, &mask);
    let border: Vec<bool> = (0..w * h)
        .map(|i| (i % w) < 4 || (i / w) >= h - 4)
        .collect();
    let max_conf_border = max_abs_diff(&flow.conf.data, &conf_py, &border);
    println!(
        "refine_border: conf max |diff| {max_conf:.3e} (border columns 0-3 and last 4 rows {max_conf_border:.3e}), \
         ux max |diff| over confident pixels {max_u:.3e}"
    );
    assert!(max_conf < 1e-3, "conf max diff {max_conf}");
    assert!(max_u < 1e-3, "ux max diff {max_u}");
    for y in 0..h {
        assert!(flow.conf.get(0, y) == 0.0 && flow.conf.get(1, y) == 0.0);
    }
    for x in 0..w {
        assert!(flow.conf.get(x, h - 2) == 0.0 && flow.conf.get(x, h - 1) == 0.0);
    }
    let med = |v: Vec<f32>| {
        let mut v = v;
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    assert!(med((16..48).map(|y| flow.conf.get(2, y)).collect()) > 0.5);
    assert!(med((16..48).map(|x| flow.conf.get(x, h - 3)).collect()) > 0.5);
}

#[test]
fn refine_confidence_is_zero_where_the_match_leaves_the_image() {
    // a constant offset pushes the right-hand columns past x = W - 1
    let mut inp = inputs();
    for v in &mut inp.ux.data {
        *v += 40.0;
    }
    let flow = run(&inp, 1);
    let (mx, my) = ((flow.ux.w - 1) as f64, (flow.ux.h - 1) as f64);
    let mut outside = 0;
    for y in 0..flow.ux.h {
        for x in 0..flow.ux.w {
            let (tx, ty) = (
                x as f64 + f64::from(flow.ux.get(x, y)),
                y as f64 + f64::from(flow.uy.get(x, y)),
            );
            // a margin keeps the f32 rounding of ux/uy from mattering at the border
            if !(-1e-3..=mx + 1e-3).contains(&tx) || !(-1e-3..=my + 1e-3).contains(&ty) {
                outside += 1;
                assert_eq!(flow.conf.get(x, y), 0.0, "({x}, {y}) -> ({tx}, {ty})");
            }
        }
    }
    assert!(outside > 1000, "only {outside} pixels left the image");
    assert!(flow.conf.data.iter().all(|c| (0.0..=1.0).contains(c)));
}

#[test]
fn refine_pure_stripe_texture_has_no_confidence() {
    // intensity varies along x only: the aperture problem leaves uy undetermined everywhere
    let n = 96;
    let stripes = Image::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| 0.5 + 0.2 * ((i % n) as f64 / 2.0).sin() as f32)
            .collect(),
    );
    let zero = Image::from_vec(n, n, vec![0.0f32; n * n]);
    for order in [1, 2] {
        let cfg = Config {
            shape_order: order,
            ..Config::default()
        };
        let flow = refine(&stripes, &stripes, &zero, &zero, &cfg);
        let worst = flow.conf.data.iter().copied().fold(0.0f32, f32::max);
        println!("order {order}: stripes, max conf {worst:.3e}");
        assert!(flow.conf.data.iter().all(|&c| c < 0.2), "max conf {worst}");
        assert!(flow
            .ux
            .data
            .iter()
            .chain(&flow.uy.data)
            .all(|v| v.is_finite()));
    }
}

#[test]
fn refine_shape_unknowns_lists_ux_uy_first_then_the_offset_powers() {
    assert_eq!(
        shape_unknowns(1),
        vec![
            (0, (0, 0)),
            (1, (0, 0)),
            (0, (1, 0)),
            (1, (1, 0)),
            (0, (0, 1)),
            (1, (0, 1))
        ]
    );
    let q = shape_unknowns(2);
    assert_eq!(q.len(), 12);
    assert_eq!(&q[..6], &shape_unknowns(1)[..]);
    assert_eq!(
        &q[6..],
        &[
            (0, (2, 0)),
            (1, (2, 0)),
            (0, (1, 1)),
            (1, (1, 1)),
            (0, (0, 2)),
            (1, (0, 2))
        ]
    );
}

/// Direct evaluation of `sum_q W(q - p) (qx - px)^px (qy - py)^py a(q)` with replicated edges.
fn window_moment_direct(a: &Image<f64>, sigma: f64, px: u32, py: u32) -> Image<f64> {
    let k = gauss_kernel(sigma);
    let r = (k.len() / 2) as i64;
    let mut out = Image::new(a.w, a.h);
    for y in 0..a.h as i64 {
        for x in 0..a.w as i64 {
            let mut acc = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (qx, qy) = (
                        (x + dx).clamp(0, a.w as i64 - 1) as usize,
                        (y + dy).clamp(0, a.h as i64 - 1) as usize,
                    );
                    acc += k[(dy + r) as usize]
                        * k[(dx + r) as usize]
                        * (dx as f64).powi(px as i32)
                        * (dy as f64).powi(py as i32)
                        * a.get(qx, qy);
                }
            }
            out.set(x as usize, y as usize, acc);
        }
    }
    out
}

#[test]
fn refine_window_moment_matches_direct_sum() {
    // deterministic pseudo-random values; image smaller than the kernel in one direction
    let (w, h) = (23usize, 9usize);
    let mut s = 12345u64;
    let data: Vec<f64> = (0..w * h)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect();
    let a = Image::from_vec(w, h, data);
    for (px, py) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2)] {
        let got = window_moment(&a, 1.5, px, py);
        let want = window_moment_direct(&a, 1.5, px, py);
        let worst = got
            .data
            .iter()
            .zip(&want.data)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-11, "moment ({px}, {py}): max |diff| {worst}");
    }
}

#[test]
#[ignore = "timing report; run with --ignored --nocapture"]
fn refine_speed_report() {
    let inp = inputs();
    let _ = run(&inp, 1); // warm-up
    let mut times: Vec<f64> = (0..7)
        .map(|_| {
            let t = std::time::Instant::now();
            let _ = run(&inp, 1);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    println!(
        "refine 128x128 order 1, {} rayon threads: min {:.1} ms, median {:.1} ms",
        rayon::current_num_threads(),
        times[0],
        times[times.len() / 2]
    );
}

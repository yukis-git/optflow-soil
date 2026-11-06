//! Parity of `phaseflow::refine` on a pure stripe texture with the Python reference
//! (`python/optflow_ref/phaseflow.py`). The image `0.5 + 0.2 sin(x / 2)` constrains one displacement
//! component only (the aperture problem): every local system is rank deficient, so the fine stage
//! must report "no information" (confidence below `conf_min`, the input displacement kept) and
//! neither fail nor produce a confident field. Run with `-- --nocapture` to see the measured
//! differences.
mod common;
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::phaseflow::refine;

fn img_f32(name: &str) -> Image<f32> {
    let (shape, v) = common::load_f64(name);
    assert_eq!(shape.len(), 2, "{name}: not a 2-D array");
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

fn check_order(order: u32) {
    let (a, b) = (img_f32("refine_stripe_a"), img_f32("refine_stripe_b"));
    assert_eq!((a.w, a.h), (64, 64));
    // the fixture starts every pixel from the same non-zero displacement, so "keeps the input
    // displacement" cannot be mistaken for "returns zero"
    let init = common::meta()["refine_stripe"]["init"].clone();
    let (ix, iy) = (
        init[0].as_f64().unwrap() as f32,
        init[1].as_f64().unwrap() as f32,
    );
    let ux0 = Image::from_vec(a.w, a.h, vec![ix; a.w * a.h]);
    let uy0 = Image::from_vec(a.w, a.h, vec![iy; a.w * a.h]);
    let cfg = Config {
        shape_order: order,
        ..Config::default()
    };
    let flow = refine(&a, &b, &ux0, &uy0, &cfg);
    for im in [&flow.ux, &flow.uy, &flow.conf] {
        assert_eq!((im.w, im.h), (a.w, a.h));
    }

    let conf_max = flow.conf.data.iter().copied().fold(0.0f32, f32::max);
    println!(
        "order {order}: max conf = {conf_max:.3e} (conf_min {})",
        cfg.conf_min
    );
    assert!(
        flow.conf.data.iter().all(|&c| f64::from(c) < cfg.conf_min),
        "order {order}: a confident pixel on a stripe texture, max conf {conf_max}"
    );

    for (name, got) in [("ux", &flow.ux), ("uy", &flow.uy), ("conf", &flow.conf)] {
        let (_, want) = common::load_f64(&format!("refine_stripe_o{order}_{name}"));
        // NaN (non-finite) positions must be the same ones; Python has none here
        let agree = common::nan_agreement(&got.data, &want, None);
        assert_eq!(
            agree, 1.0,
            "order {order} {name}: non-finite positions differ"
        );
        let rms = common::rms_diff_common(&got.data, &want, None);
        println!("order {order} {name}: rms = {rms:.3e}");
        let tol = if name == "conf" { 1e-6 } else { 1e-3 };
        assert!(rms <= tol, "order {order} {name}: rms {rms}");
    }
    // the displacement is the input one (the degenerate pixels keep it)
    assert!(flow.ux.data.iter().all(|&v| (v - ix).abs() < 1e-3));
    assert!(flow.uy.data.iter().all(|&v| (v - iy).abs() < 1e-3));
}

#[test]
fn refine_on_stripes_order_1_has_no_confidence_and_matches_python() {
    check_order(1);
}

#[test]
fn refine_on_stripes_order_2_has_no_confidence_and_matches_python() {
    check_order(2);
}

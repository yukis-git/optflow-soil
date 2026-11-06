//! Parity of `crack_model = "xdic"` with the Python reference (`python/optflow_ref/xdic.py` and
//! `_track_pair_xdic` of `track.py`). The fixtures (`xd_*.npy`, `xdic_meta.json`) come from
//! `python/bench/make_xdic_fixtures.py`: crack pairs at normal angle 0 and 30 degrees with opening 1
//! and 6 px, the ShearBand and weak-texture negative controls, a step field for the detector, a
//! chain and a hybrid sequence with a widening opening, and the frame-0 mapping of a line in a gap.
//! Run with `-- --nocapture` to see the measured differences.
//!
//! The discrete results (the number of lines, bands and rejected lines, which pixels are on the
//! line, the references of a sequence) must match exactly; the continuous ones (line vertices,
//! displacement, COD / CSD) to the tolerances below. Rust holds images, filter responses and spline
//! coefficients in `f32`, the reference in `f64`; the measured differences (line vertices a few
//! 1e-6 px, displacement 3e-7 px rms, which is mostly the float32 of the fixtures) are far below
//! the 1e-3 px of the verification spec, and the bounds here are 10 to 30 times the measured
//! values, so that a regression shows.
#![allow(clippy::needless_range_loop)]
mod common;
use common::fixture_dir;
use optflow_core::config::Config;
use optflow_core::image::Image;
use optflow_core::npy::NpyData;
use optflow_core::npy::{self, NpyArray};
use optflow_core::track::{track_pair, track_sequence, TrackResult};
use optflow_core::xdic::detect::{detect_lines, jump_test};
use optflow_core::xdic::geom::{near_lines, polyline_fields, resample};
use optflow_core::xdic::outputs::{advance_lines, lines_to_frame0};
use optflow_core::xdic::Poly;
use serde_json::Value;

/// The fixtures of this file live in `tests/fixtures/xdic/` (u16 images, bool and f32 maps).
fn load(name: &str) -> NpyArray {
    let path = fixture_dir().join("xdic").join(format!("{name}.npy"));
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    npy::read(&bytes).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

fn load_f64(name: &str) -> (Vec<usize>, Vec<f64>) {
    let a = load(name);
    let v = match a.data {
        NpyData::F64(v) => v,
        NpyData::F32(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U8(v) => v.into_iter().map(f64::from).collect(),
        NpyData::U16(v) => v.into_iter().map(f64::from).collect(),
        NpyData::Bool(v) => v.into_iter().map(|b| f64::from(u8::from(b))).collect(),
    };
    (a.shape, v)
}

#[test]
fn the_xdic_fixtures_stay_small() {
    let total: u64 = std::fs::read_dir(fixture_dir().join("xdic"))
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(total < 5_000_000, "xdic fixtures are {total} bytes");
}

fn meta() -> Value {
    let text = std::fs::read_to_string(fixture_dir().join("xdic").join("xdic_meta.json"))
        .expect("xdic_meta.json");
    serde_json::from_str(&text).unwrap()
}

/// A u16 fixture as the reference read it: `(v / 65535)` rounded to float32.
fn img16(name: &str) -> Image<f32> {
    let a = load(name);
    let NpyData::U16(v) = a.data else {
        panic!("{name}: not u16")
    };
    let (h, w) = (a.shape[a.shape.len() - 2], a.shape[a.shape.len() - 1]);
    let n = h * w;
    let frame = |k: usize| {
        Image::from_vec(
            w,
            h,
            v[k * n..(k + 1) * n]
                .iter()
                .map(|&x| (f64::from(x) / 65535.0) as f32)
                .collect(),
        )
    };
    frame(0)
}

fn frames16(name: &str) -> Vec<Image<f32>> {
    let a = load(name);
    let NpyData::U16(v) = a.data else {
        panic!("{name}: not u16")
    };
    let (k, h, w) = (a.shape[0], a.shape[1], a.shape[2]);
    (0..k)
        .map(|i| {
            Image::from_vec(
                w,
                h,
                v[i * h * w..(i + 1) * h * w]
                    .iter()
                    .map(|&x| (f64::from(x) / 65535.0) as f32)
                    .collect(),
            )
        })
        .collect()
}

fn f32img(name: &str) -> Image<f32> {
    let (shape, v) = load_f64(name);
    Image::from_vec(shape[1], shape[0], v.iter().map(|&x| x as f32).collect())
}

fn poly(v: &Value) -> Poly {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
        .collect()
}

fn polys(v: &Value) -> Vec<Poly> {
    v.as_array().unwrap().iter().map(poly).collect()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

/// Largest distance between corresponding vertices (infinite if the shapes differ).
fn max_vertex_diff(a: &[Poly], b: &[Poly]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut worst = 0.0f64;
    for (p, q) in a.iter().zip(b) {
        if p.len() != q.len() {
            return f64::INFINITY;
        }
        for (u, v) in p.iter().zip(q) {
            worst = worst.max((u[0] - v[0]).hypot(u[1] - v[1]));
        }
    }
    worst
}

fn xcfg() -> Config {
    Config {
        crack_model: "xdic".to_string(),
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------------------------
// geometry and detector

#[test]
fn polyline_fields_match_python() {
    let m = meta();
    let lines = polys(&m["units"]["poly"]["lines"]);
    let f = polyline_fields(&lines, 64, 64);
    let want = |k: &str| load_f64(&format!("xd_poly_{k}")).1;
    let (phi, nx, ny, tau, lid, inter) = (
        want("phi"),
        want("nx"),
        want("ny"),
        want("tau"),
        want("lid"),
        want("interior"),
    );
    let (mut dphi, mut dn, mut dtau) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..64 * 64 {
        assert_eq!(f.interior[i], inter[i] > 0.5, "interior at {i}");
        assert_eq!(f64::from(f.lid[i]), lid[i], "lid at {i}");
        dphi = dphi.max((f.phi[i] - phi[i]).abs());
        dn = dn.max((f.nx[i] - nx[i]).abs()).max((f.ny[i] - ny[i]).abs());
        dtau = dtau.max((f.tau[i] - tau[i]).abs());
    }
    println!("polyline_fields: max |dphi| {dphi:.2e}  |dn| {dn:.2e}  |dtau| {dtau:.2e}");
    // the fixtures hold float32 of the float64 fields (rounding 6e-8 relative)
    assert!(dphi < 1e-5 && dn < 1e-6 && dtau < 1e-5);
}

#[test]
fn resample_advance_and_near_lines_match_python() {
    let m = meta();
    let u = &m["units"];
    let lines = polys(&u["poly"]["lines"]);
    let (pts, nrm, s) = resample(&lines[0], 7.0);
    let want = &u["resample"];
    let wp = poly(&want["pts"]);
    let wn = poly(&want["n"]);
    let ws = nums(&want["s"]);
    assert_eq!(pts.len(), wp.len());
    for i in 0..pts.len() {
        assert!((pts[i][0] - wp[i][0]).abs() < 1e-12 && (pts[i][1] - wp[i][1]).abs() < 1e-12);
        assert!((nrm[i][0] - wn[i][0]).abs() < 1e-12 && (nrm[i][1] - wn[i][1]).abs() < 1e-12);
        assert!((s[i] - ws[i]).abs() < 1e-12);
    }
    let (w, h) = (64usize, 64usize);
    let ax: Vec<f64> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            0.5 + 0.03 * x - 0.01 * y + if x > 30.0 { 1.5 } else { 0.0 }
        })
        .collect();
    let ay: Vec<f64> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            -0.4 + 0.02 * y + 0.005 * x
        })
        .collect();
    let adv = advance_lines(&lines, &ax, &ay, w, h);
    let d = max_vertex_diff(&adv, &polys(&u["advance"]["lines"]));
    println!("advance_lines: max vertex diff {d:.2e}");
    assert!(d < 1e-12, "{d}");
    let shifted =
        |dx: f64| -> Vec<Poly> { vec![lines[0].iter().map(|p| [p[0] + dx, p[1] + dx]).collect()] };
    let want_near = u["near"].as_array().unwrap();
    assert_eq!(
        near_lines(&lines[0], &shifted(2.0), 5.0),
        want_near[0].as_bool().unwrap()
    );
    assert_eq!(
        near_lines(&lines[0], &shifted(12.0), 5.0),
        want_near[1].as_bool().unwrap()
    );
}

#[test]
fn frame0_mapping_of_a_line_in_a_wide_gap_matches_python() {
    let m = meta();
    let (h, w) = (256usize, 256usize);
    let mut worst = 0.0f64;
    for case in m["units"]["wide"].as_array().unwrap() {
        let o = case["o"].as_f64().unwrap();
        let ux: Vec<f64> = (0..w * h)
            .map(|i| if (i % w) >= 128 { o } else { 0.0 })
            .collect();
        let uy = vec![-3.0f64; w * h];
        let p = poly(&case["P"]);
        let got = lines_to_frame0(&[p], Some((&ux, &uy)), w, h, 6.0);
        let want = poly(&case["X"]);
        for (a, b) in got[0].iter().zip(&want) {
            worst = worst.max((a[0] - b[0]).hypot(a[1] - b[1]));
        }
        // the point of the case: any P inside the gap maps to the crack at x = 128
        assert!(
            got[0].iter().all(|q| (q[0] - 128.0).abs() <= 1.0),
            "{:?}",
            got[0]
        );
    }
    println!("lines_to_frame0 (gaps of 16 and 30 px): max diff {worst:.2e}");
    assert!(worst < 1e-9, "{worst}");
}

#[test]
fn detector_on_a_step_field_matches_python() {
    let m = meta();
    let st = &m["units"]["step"];
    let (ux, uy) = (f32img("xd_step_ux"), f32img("xd_step_uy"));
    let (w, h) = (ux.w, ux.h);
    let a: Vec<f64> = ux.data.iter().map(|&v| f64::from(v)).collect();
    let b: Vec<f64> = uy.data.iter().map(|&v| f64::from(v)).collect();
    let valid: Vec<bool> = (0..w * h)
        .map(|i| a[i].is_finite() && b[i].is_finite())
        .collect();
    let px: Vec<i64> = st["pts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p[0].as_i64().unwrap())
        .collect();
    let py: Vec<i64> = st["pts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p[1].as_i64().unwrap())
        .collect();
    let jt = jump_test(
        &a,
        &b,
        &valid,
        w,
        h,
        &px,
        &py,
        36.0,
        16,
        8.0,
        30.0,
        4.0 * std::f64::consts::PI * 16.0,
    );
    let wj = st["J"].as_array().unwrap();
    let wz = nums(&st["z"]);
    let mut dj = 0.0f64;
    let mut dz = 0.0f64;
    for i in 0..px.len() {
        let nan = wz[i] == -999.0;
        assert_eq!(jt.z[i].is_nan(), nan, "point {i}");
        if !nan {
            dj = dj.max((jt.j[i][0] - wj[i][0].as_f64().unwrap()).abs());
            dj = dj.max((jt.j[i][1] - wj[i][1].as_f64().unwrap()).abs());
            dz = dz.max(((jt.z[i] - wz[i]) / wz[i]).abs());
        }
    }
    println!("jump_test: max |dJ| {dj:.2e}  max relative dz {dz:.2e}");
    assert!(dj < 1e-9 && dz < 1e-9, "{dj} {dz}");
    let cfg = xcfg();
    let (lines, info) = detect_lines(&a, &b, None, w, h, &cfg, None);
    let want = polys(&st["lines"]);
    let d = max_vertex_diff(&lines, &want);
    println!(
        "detect_lines: {} line(s), candidates {} (python {}), accepted {} (python {}), max vertex diff {d:.2e}",
        lines.len(),
        info.n_cand,
        st["n_cand"],
        info.n_acc,
        st["n_acc"]
    );
    assert_eq!(info.n_cand as u64, st["n_cand"].as_u64().unwrap());
    assert_eq!(info.n_acc as u64, st["n_acc"].as_u64().unwrap());
    assert!(d < 1e-6, "{d}");
}

// ---------------------------------------------------------------------------------------------
// pairs

fn crack_lines(
    r: &TrackResult,
) -> (
    Vec<Poly>,
    Vec<Poly>,
    Vec<Poly>,
    Vec<optflow_core::xdic::JumpLine>,
) {
    match &r.crack {
        Some(c) => (
            c.lines.lines.clone(),
            c.lines.bands.clone(),
            c.lines.rejected.clone(),
            c.lines.jumps.clone(),
        ),
        None => (vec![], vec![], vec![], vec![]),
    }
}

/// One crack pair: returns (nan agreement, rms ux, rms uy, rms conf, line vertex diff, cod diff).
fn check_pair(key: &str) -> (f64, f64, f64, f64, f64, f64) {
    let m = meta();
    let want = &m[format!("pair_{key}")];
    let (a, b) = (img16(&format!("xd_{key}_a")), img16(&format!("xd_{key}_b")));
    let r = track_pair(&a, &b, &xcfg(), None, None).unwrap();
    let (lines, bands, rejected, jumps) = crack_lines(&r);
    // discrete results: exact
    assert_eq!(
        lines.len() as u64,
        want["n_lines"].as_u64().unwrap(),
        "{key}: number of lines"
    );
    assert_eq!(
        bands.len(),
        want["bands"].as_array().unwrap().len(),
        "{key}: bands"
    );
    assert_eq!(
        rejected.len(),
        want["rejected"].as_array().unwrap().len(),
        "{key}: rejected"
    );
    let crack_px = r
        .crack
        .as_ref()
        .and_then(|c| c.maps.as_ref())
        .map_or(0, |m| m.mask.data.iter().filter(|&&b| b).count());
    assert_eq!(
        crack_px as u64,
        want["crack_px"].as_u64().unwrap(),
        "{key}: pixels on the line"
    );
    let (_, ux_py) = load_f64(&format!("xd_{key}_ux"));
    let (_, uy_py) = load_f64(&format!("xd_{key}_uy"));
    let (_, conf_py) = load_f64(&format!("xd_{key}_conf"));
    let agree = common::nan_agreement(&r.ux.data, &ux_py, None);
    let rux = common::rms_diff_common(&r.ux.data, &ux_py, None);
    let ruy = common::rms_diff_common(&r.uy.data, &uy_py, None);
    let rconf = common::rms_diff(&r.conf.data, &conf_py, None);
    let dl = max_vertex_diff(&lines, &polys(&want["lines"]));
    let mut dcod = 0.0f64;
    let wj = want["jumps"].as_array().unwrap();
    assert_eq!(jumps.len(), wj.len(), "{key}: jump lines");
    for (j, w) in jumps.iter().zip(wj) {
        let (wc, ws) = (nums(&w["cod"]), nums(&w["csd"]));
        assert_eq!(j.cod.len(), wc.len(), "{key}: samples");
        for i in 0..wc.len() {
            assert_eq!(
                j.cod[i].is_nan(),
                wc[i].is_nan(),
                "{key}: cod finite at {i}"
            );
            if wc[i].is_finite() {
                dcod = dcod
                    .max((j.cod[i] - wc[i]).abs())
                    .max((j.csd[i] - ws[i]).abs());
            }
        }
    }
    println!(
        "pair {key}: lines {}  nan_agreement {agree:.5}  rms ux {rux:.2e} uy {ruy:.2e} conf {rconf:.2e}  \
         line vertex diff {dl:.2e}  COD/CSD diff {dcod:.2e}",
        lines.len()
    );
    (agree, rux, ruy, rconf, dl, dcod)
}

macro_rules! pair_test {
    ($name:ident, $key:expr) => {
        #[test]
        fn $name() {
            let (agree, rux, ruy, rconf, dl, dcod) = check_pair($key);
            assert!(agree >= 0.995, "nan agreement {agree}");
            assert!(rux < 1e-5 && ruy < 1e-5, "rms {rux} {ruy}");
            assert!(rconf < 1e-5, "rms conf {rconf}");
            assert!(dl < 1e-4, "line vertices {dl}");
            assert!(dcod < 1e-4, "COD / CSD {dcod}");
        }
    };
}
pair_test!(crack_pair_a0_o1_matches_python, "a0_o1");
pair_test!(crack_pair_a0_o6_matches_python, "a0_o6");
pair_test!(crack_pair_a30_o1_matches_python, "a30_o1");
pair_test!(crack_pair_a30_o6_matches_python, "a30_o6");

#[test]
fn one_sided_strain_of_the_slipping_crack_matches_python() {
    let (a, b) = (img16("xd_a30_o6_a"), img16("xd_a30_o6_b"));
    let r = track_pair(&a, &b, &xcfg(), None, None).unwrap();
    let (_, exx_py) = load_f64("xd_a30_o6_Exx");
    let rms = common::rms_diff_common(&r.strain.exx_gl.data, &exx_py, None);
    let agree = common::nan_agreement(&r.strain.exx_gl.data, &exx_py, None);
    let (_, cod_py) = load_f64("xd_a30_o6_cod");
    let cod = &r.crack.as_ref().unwrap().maps.as_ref().unwrap().cod;
    let dcod = common::rms_diff_common(&cod.data, &cod_py, None);
    println!("a30_o6: Exx rms {rms:.2e} (nan agreement {agree:.5}), COD map rms {dcod:.2e}");
    assert!(rms < 1e-6 && agree >= 0.995, "{rms} {agree}");
    assert!(dcod < 1e-5, "{dcod}");
}

#[test]
fn negative_controls_are_bands_or_nothing_and_leave_the_baseline_alone() {
    let m = meta();
    for key in ["weak", "band"] {
        let want = &m[format!("pair_{key}")];
        let (a, b) = (img16(&format!("xd_{key}_a")), img16(&format!("xd_{key}_b")));
        let r = track_pair(&a, &b, &xcfg(), None, None).unwrap();
        let base = track_pair(&a, &b, &Config::default(), None, None).unwrap();
        let (lines, bands, rejected, _) = crack_lines(&r);
        println!(
            "negative {key}: lines {} bands {} rejected {} (python {} {} {})",
            lines.len(),
            bands.len(),
            rejected.len(),
            want["n_lines"],
            want["bands"].as_array().unwrap().len(),
            want["rejected"].as_array().unwrap().len()
        );
        assert_eq!(
            lines.len() as u64,
            want["n_lines"].as_u64().unwrap(),
            "{key}"
        );
        assert_eq!(
            bands.len(),
            want["bands"].as_array().unwrap().len(),
            "{key}"
        );
        assert_eq!(
            rejected.len(),
            want["rejected"].as_array().unwrap().len(),
            "{key}"
        );
        if lines.is_empty() {
            // no crack: the baseline result, bit for bit
            assert!(common::bits(&r.ux) == common::bits(&base.ux), "{key}: ux");
            assert!(common::bits(&r.uy) == common::bits(&base.uy), "{key}: uy");
            assert!(
                common::bits(&r.conf) == common::bits(&base.conf),
                "{key}: conf"
            );
            assert!(
                r.crack.as_ref().is_none_or(|c| c.maps.is_none()),
                "{key}: no crack maps"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// sequences

fn check_sequence(mode: &str) {
    let m = meta();
    let want = m[format!("seq_{mode}")].as_array().unwrap().clone();
    let frames = frames16("xd_seq_frames");
    let cfg = Config {
        crack_model: "xdic".to_string(),
        seq_mode: mode.to_string(),
        ref_update_conf: 0.99,
        ..Config::default()
    };
    let out: Vec<(usize, TrackResult)> = track_sequence(frames.into_iter(), &cfg, None)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(out.len(), want.len());
    let mut worst_line = 0.0f64;
    let mut worst_cod = 0.0f64;
    for ((k, r), w) in out.iter().zip(&want) {
        assert_eq!(*k as u64, w["k"].as_u64().unwrap());
        assert_eq!(
            r.timing.reference.unwrap() as u64,
            w["reference"].as_u64().unwrap(),
            "{mode} frame {k}: reference"
        );
        let (lines, _, _, jumps) = crack_lines(r);
        assert_eq!(
            lines.len() as u64,
            w["n_lines"].as_u64().unwrap(),
            "{mode} frame {k}: lines"
        );
        let d = max_vertex_diff(&lines, &polys(&w["lines"]));
        worst_line = worst_line.max(d);
        let wj = w["jumps"].as_array().unwrap();
        assert_eq!(jumps.len(), wj.len(), "{mode} frame {k}: jump lines");
        for (j, wjl) in jumps.iter().zip(wj) {
            let wc = nums(&wjl["cod"]);
            assert_eq!(j.cod.len(), wc.len(), "{mode} frame {k}: samples");
            for i in 0..wc.len() {
                assert_eq!(
                    j.cod[i].is_nan(),
                    wc[i].is_nan(),
                    "{mode} frame {k}: cod finite"
                );
                if wc[i].is_finite() {
                    worst_cod = worst_cod.max((j.cod[i] - wc[i]).abs());
                }
            }
        }
        println!(
            "{mode} frame {k}: reference {}  lines {}  vertex diff {d:.2e}",
            r.timing.reference.unwrap(),
            lines.len()
        );
    }
    let (_, last) = out.last().unwrap();
    let (_, ux_py) = load_f64(&format!("xd_seq_{mode}_ux"));
    let (_, uy_py) = load_f64(&format!("xd_seq_{mode}_uy"));
    let agree = common::nan_agreement(&last.ux.data, &ux_py, None);
    let rux = common::rms_diff_common(&last.ux.data, &ux_py, None);
    let ruy = common::rms_diff_common(&last.uy.data, &uy_py, None);
    println!("{mode}: last frame nan_agreement {agree:.5}  rms ux {rux:.2e} uy {ruy:.2e}  max line diff {worst_line:.2e}  max COD diff {worst_cod:.2e}");
    assert!(agree >= 0.995, "{agree}");
    assert!(rux < 1e-5 && ruy < 1e-5, "{rux} {ruy}");
    assert!(worst_line < 1e-4, "{worst_line}");
    assert!(worst_cod < 1e-4, "{worst_cod}");
}

#[test]
fn chain_sequence_with_a_widening_gap_matches_python() {
    check_sequence("chain");
}

#[test]
fn hybrid_sequence_with_a_widening_gap_matches_python() {
    check_sequence("hybrid");
}

/// A crack at x = 80 opened by 8 px, a hole of the field in the last row around the crack (the end
/// vertex of a line beyond the image has no finite sample within the search), one far pixel that
/// sets max |u0r|.
fn frame0_step_field(max_u: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 160usize;
    let mut ux: Vec<f64> = (0..n * n)
        .map(|i| if i % n >= 80 { 8.0 } else { 0.0 })
        .collect();
    let mut uy = vec![0.0f64; n * n];
    for x in 65..105 {
        ux[(n - 1) * n + x] = f64::NAN;
        uy[(n - 1) * n + x] = f64::NAN;
    }
    ux[20 * n + 150] = max_u;
    (ux, uy)
}

/// max |u0r| on either side of an integer changes the search line by 1 px. Regression: the end
/// vertex jumped 0.4 px (and 0.351 px on the chain fixture, between x86_64 and aarch64).
#[test]
fn frame0_mapping_does_not_depend_on_the_length_of_the_search() {
    let p: Poly = (0..11)
        .map(|k| [84.1, -2.0 + 16.3 * f64::from(k)])
        .collect();
    let map = |m: f64| {
        let (a, b) = frame0_step_field(m);
        lines_to_frame0(std::slice::from_ref(&p), Some((&a, &b)), 160, 160, 6.0)
    };
    let (a, b) = (map(10.9995), map(11.0005));
    assert!(max_vertex_diff(&a, &b) < 1e-9, "{a:?} {b:?}");
    // also the vertex without a sample: it moves with its neighbour
    assert!(a[0].iter().all(|q| (q[0] - 80.0).abs() < 1.0), "{a:?}");
}

/// Regression: leaving the 1 px cell switched between `P - u` and the zero of `g`, up to 1 px.
#[test]
fn frame0_mapping_is_continuous_in_the_line() {
    let n = 160usize;
    let ux: Vec<f64> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f64, (i / n) as f64);
            0.01 * x + 0.005 * y + if x >= 80.0 { 6.0 } else { 0.0 }
        })
        .collect();
    let uy: Vec<f64> = (0..n * n).map(|i| 0.002 * (i % n) as f64).collect();
    let at = |t: f64| {
        let p: Poly = (0..9)
            .map(|k| {
                let k = f64::from(k) / 8.0;
                [83.0 + k - t, 10.0 + 140.0 * k]
            })
            .collect();
        lines_to_frame0(&[p], Some((&ux, &uy)), n, n, 6.0)
    };
    let mut worst = 0.0f64;
    let mut prev = at(-5.0);
    for k in 1..1000 {
        let cur = at(-5.0 + 0.01 * f64::from(k));
        worst = worst.max(max_vertex_diff(&prev, &cur));
        prev = cur;
    }
    assert!(worst < 0.06, "{worst}");
}

/// The chain sequence with noise of +-1e-6 on the frames: no vertex moves by more than the noise
/// amplified by the smooth part of the method (measured: at most 1e-4 px). The two seeds moved
/// the end vertex of frame 3 by 0.351 px before the mapping was made continuous.
#[test]
fn a_tiny_perturbation_of_the_frames_gives_a_tiny_line_change() {
    let cfg = Config {
        crack_model: "xdic".to_string(),
        seq_mode: "chain".to_string(),
        ref_update_conf: 0.99,
        ..Config::default()
    };
    let run = |frames: Vec<Image<f32>>| -> Vec<Vec<Poly>> {
        track_sequence(frames.into_iter(), &cfg, None)
            .map(|r| crack_lines(&r.unwrap().1).0)
            .collect()
    };
    let base = run(frames16("xd_seq_frames"));
    for seed in [4u64, 16] {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF;
        let mut frames = frames16("xd_seq_frames");
        for v in frames.iter_mut().flat_map(|f| f.data.iter_mut()) {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            *v += 1e-6 * (2.0 * ((st >> 40) as f32 / (1u64 << 24) as f32) - 1.0);
        }
        let got = run(frames);
        for (k, (a, b)) in got.iter().zip(&base).enumerate() {
            let d = max_vertex_diff(a, b);
            assert!(d < 1e-3, "seed {seed}, frame {}: {d}", k + 1);
        }
    }
}

/// An engine that does not return the slope of the baseline solve (the GPU engine is one) is
/// refused with a clear error, before any work.
/// An image with a side above the 1024 px of the tiles of the baseline: one line along the middle
/// row of a pair whose two halves moved apart by 3 px, found and measured as on a small image.
#[test]
fn an_image_longer_than_a_tile_is_solved() {
    use common::synth::{make_pattern, render, uniform_strain};
    let (w, h) = (1100, 140);
    let pat = make_pattern(w, h, 11, 32.0);
    let a = render(&pat, w, h, None);
    let shifted =
        |t: (f64, f64)| render(&pat, w, h, Some(&uniform_strain(0.0, 0.0, (0.0, 0.0), t)));
    let (up, down) = (shifted((0.5, -1.0)), shifted((0.5, 2.0)));
    let data = (0..w * h)
        .map(|i| {
            if i / w < h / 2 {
                up.data[i]
            } else {
                down.data[i]
            }
        })
        .collect();
    let b = Image::from_vec(w, h, data);
    let r = track_pair(&a, &b, &xcfg(), None, None).unwrap();
    let c = r.crack.as_ref().expect("a crack");
    assert_eq!(c.lines.lines.len(), 1);
    let span = |k: usize| {
        let v = c.lines.lines[0].iter().map(|p| p[k]);
        v.clone().fold(f64::NEG_INFINITY, f64::max) - v.fold(f64::INFINITY, f64::min)
    };
    assert!(span(0) > 900.0, "line along x: {}", span(0));
    assert!(span(1) < 8.0, "line along x: {}", span(1));
    let mut cod: Vec<f64> = c.lines.jumps[0]
        .cod
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    assert!(cod.len() > 300, "{} COD samples", cod.len());
    cod.sort_by(f64::total_cmp);
    let m = cod[cod.len() / 2];
    assert!((m.abs() - 3.0).abs() < 0.1, "median COD {m}");
    let finite = r.ux.data.iter().filter(|v| v.is_finite()).count();
    assert!(
        finite as f64 > 0.5 * (w * h) as f64,
        "{finite} valid pixels"
    );
}

#[test]
fn an_image_above_the_area_limit_is_refused() {
    let (w, h) = (3841, 2160);
    let im = Image::from_vec(w, h, vec![0.5f32; w * h]);
    let err = track_pair(&im, &im, &xcfg(), None, None).unwrap_err();
    assert!(err.to_string().contains("at most"), "{err}");
}

#[test]
fn an_engine_without_the_slope_is_refused() {
    use optflow_core::engine::{CpuEngine, EngineError, FineEngine};
    use optflow_core::phaseflow::RawFlow;
    struct NoSlope;
    impl FineEngine for NoSlope {
        fn refine_raw(
            &self,
            i1: &Image<f32>,
            i2: &Image<f32>,
            shift: (i64, i64),
            ux: &Image<f32>,
            uy: &Image<f32>,
            cfg: &Config,
        ) -> Result<RawFlow, EngineError> {
            CpuEngine.refine_raw(i1, i2, shift, ux, uy, cfg)
        }
    }
    let (a, b) = (img16("xd_a0_o6_a"), img16("xd_a0_o6_b"));
    let err = optflow_core::track::track_pair_on(&NoSlope, &a, &b, &xcfg(), None, None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("runs on the CPU only") && err.contains("--device cpu"),
        "{err}"
    );
    // the default config is untouched by the refusal
    assert!(
        optflow_core::track::track_pair_on(&NoSlope, &a, &b, &Config::default(), None, None)
            .is_ok()
    );
}

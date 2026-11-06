//! `--mask PATH` (an ROI as a .npy or .png file) and `--raw` (also write `raw_ux.npy`,
//! `raw_uy.npy`) of the `optflow` binary.

mod common;

use std::path::Path;
use std::process::Output;

use common::*;
use optflow_cli::io::npy_u8_bytes;
use optflow_cli::polygon::polygon_mask;
use optflow_core::npy;

const POLY: [(f64, f64); 4] = [(24.0, 24.0), (104.0, 24.0), (104.0, 104.0), (24.0, 104.0)];
const ARRAYS: [&str; 9] = [
    "ux", "uy", "conf", "Exx", "Eyy", "Exy", "gmax", "vol", "valid",
];

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn text(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

/// `n` frames `f0.png`, ... of a pattern moving by (0.7, -0.4) px per frame, and `roi.json`.
fn scene(name: &str, size: usize, n: usize) -> std::path::PathBuf {
    let dir = scratch(name);
    let p = make_pattern(size, size, 3);
    for k in 0..n {
        let img = render(&p, size, size, 0.7 * k as f64, -0.4 * k as f64);
        write_png16(&dir.join(format!("f{k}.png")), size, size, &img);
    }
    let pts: Vec<String> = POLY.iter().map(|(x, y)| format!("[{x}, {y}]")).collect();
    std::fs::write(
        dir.join("roi.json"),
        format!("{{\"polygon\": [{}]}}", pts.join(", ")),
    )
    .unwrap();
    dir
}

/// `optflow pair` of `dir/f0.png`, `dir/f1.png` with the options `extra` and the output `out`.
fn pair(dir: &Path, out: &str, extra: &[&str]) -> Output {
    let (a, b, o) = (dir.join("f0.png"), dir.join("f1.png"), dir.join(out));
    let mut args = vec![
        "pair",
        "--ref",
        s(&a),
        "--def",
        s(&b),
        "--set",
        "outer_iters=2",
        "--out",
        s(&o),
    ];
    args.extend_from_slice(extra);
    optflow(&args)
}

fn ok(o: &Output) {
    assert_eq!(o.status.code(), Some(0), "{}", stderr(o));
}

/// The last line of stderr, which is the error message.
fn error_line(o: &Output) -> String {
    assert_eq!(o.status.code(), Some(2), "{}", stderr(o));
    stderr(o).lines().last().unwrap().to_string()
}

/// A `|b1` npy file (format 1.0).
fn npy_bool(shape: &[usize], data: &[bool]) -> Vec<u8> {
    let mut bytes = npy_u8_bytes(
        shape,
        &data.iter().map(|&b| u8::from(b)).collect::<Vec<_>>(),
    );
    let at = bytes.windows(4).position(|w| w == b"'|u1").unwrap();
    bytes[at + 2] = b'b'; // '|u1' -> '|b1'
    bytes[at + 3] = b'1';
    bytes
}

/// A `|b1` npy file of the 2-D array `c_order` (`h` rows) in Fortran order, as
/// `numpy.save(f, np.asfortranarray(a))` writes it: `fortran_order` True, the columns one after
/// the other.
fn npy_bool_fortran(h: usize, w: usize, c_order: &[bool]) -> Vec<u8> {
    let dict = format!("{{'descr': '|b1', 'fortran_order': True, 'shape': ({h}, {w}), }}");
    let total = npy::header_len(dict.len());
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&u16::try_from(total - 10).unwrap().to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    bytes.resize(total - 1, b' ');
    bytes.push(b'\n');
    bytes.extend((0..w).flat_map(|x| (0..h).map(move |y| u8::from(c_order[y * w + x]))));
    bytes
}

fn polygon(w: usize, h: usize) -> Vec<bool> {
    polygon_mask(w, h, &POLY, 1).data
}

#[test]
fn mask_file_gives_the_outputs_of_the_same_roi() {
    let dir = scene("mask_same", 128, 2);
    let m = polygon(128, 128);
    std::fs::write(dir.join("bool.npy"), npy_bool(&[128, 128], &m)).unwrap();
    let u8s: Vec<u8> = m.iter().map(|&b| 7 * u8::from(b)).collect(); // nonzero = inside
    std::fs::write(dir.join("u8.npy"), npy_u8_bytes(&[128, 128], &u8s)).unwrap();
    let grey: Vec<f64> = m.iter().map(|&b| f64::from(u8::from(b))).collect();
    write_png8(&dir.join("m.png"), 128, 128, &grey);
    ok(&pair(&dir, "roi", &["--roi", s(&dir.join("roi.json"))]));
    for file in ["bool.npy", "u8.npy", "m.png"] {
        let out = format!("mask_{file}");
        ok(&pair(&dir, &out, &["--mask", s(&dir.join(file))]));
        for name in ARRAYS {
            let want = std::fs::read(dir.join("roi").join(format!("{name}.npy"))).unwrap();
            let got = std::fs::read(dir.join(&out).join(format!("{name}.npy"))).unwrap();
            assert!(want == got, "{file}: {name}.npy differs");
        }
        assert_eq!(
            text(&dir.join(&out).join("crop.json")),
            text(&dir.join("roi/crop.json"))
        );
    }
}

#[test]
fn a_fortran_order_mask_gives_the_outputs_of_the_same_mask_in_c_order() {
    let dir = scene("mask_fortran", 128, 2);
    // a rectangle that is not symmetric about the diagonal: x 24..104, y 40..100
    let (w, h) = (128, 128);
    let m: Vec<bool> = (0..w * h)
        .map(|i| (24..104).contains(&(i % w)) && (40..100).contains(&(i / w)))
        .collect();
    std::fs::write(dir.join("c.npy"), npy_bool(&[h, w], &m)).unwrap();
    std::fs::write(dir.join("f.npy"), npy_bool_fortran(h, w, &m)).unwrap();
    for file in ["c.npy", "f.npy"] {
        ok(&pair(
            &dir,
            &format!("out_{file}"),
            &["--mask", s(&dir.join(file))],
        ));
    }
    for name in ARRAYS {
        let c = std::fs::read(dir.join("out_c.npy").join(format!("{name}.npy"))).unwrap();
        let f = std::fs::read(dir.join("out_f.npy").join(format!("{name}.npy"))).unwrap();
        assert!(c == f, "{name}.npy differs");
    }
    let (shape, ux) = load_npy(&dir.join("out_f.npy/ux.npy"));
    assert_eq!(shape, [h, w]);
    assert!(
        (0..w * h).all(|i| m[i] || ux[i].is_nan()),
        "ux outside the mask"
    );
    assert!(
        (0..w * h).any(|i| m[i] && ux[i].is_finite()),
        "ux inside the mask"
    );
}

#[test]
fn a_two_block_mask_is_cropped_to_the_joint_bounding_box() {
    let dir = scratch("mask_two_blocks");
    let (w, h) = (256, 256);
    let p = make_pattern(w, h, 4);
    write_png16(&dir.join("f0.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("f1.png"), w, h, &render(&p, w, h, 0.9, 0.3));
    let mut m = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            m[y * w + x] = ((90..150).contains(&y) && (80..140).contains(&x))
                || ((160..220).contains(&y) && (150..210).contains(&x));
        }
    }
    std::fs::write(dir.join("m.npy"), npy_bool(&[h, w], &m)).unwrap();
    ok(&pair(
        &dir,
        "out",
        &["--mask", s(&dir.join("m.npy")), "--crop-margin", "40"],
    ));
    // x0 = 80 - 40 = 40 -> 32 (A = 32), x1 = 210 + 40; y0 = 90 - 40 = 50 -> 32, y1 = 260 is clipped to 256
    let want = "{\n  \"x0\": 32,\n  \"y0\": 32,\n  \"x1\": 250,\n  \"y1\": 256,\n  \"margin\": 40,\n  \"mode\": \"auto\"\n}";
    assert_eq!(text(&dir.join("out/crop.json")), want);
    let (shape, ux) = load_npy(&dir.join("out/ux.npy"));
    assert_eq!(shape, [h, w]);
    let inside = (0..w * h).filter(|&i| m[i]);
    assert!(inside.clone().filter(|&i| ux[i].is_finite()).count() * 5 > inside.count());
    assert!((0..w * h).all(|i| m[i] || ux[i].is_nan()));
    let valid_in = |y: std::ops::Range<usize>, x: std::ops::Range<usize>| {
        y.flat_map(|y| x.clone().map(move |x| y * w + x))
            .any(|i| ux[i].is_finite())
    };
    assert!(valid_in(90..150, 80..140) && valid_in(160..220, 150..210));
}

#[test]
fn mask_errors_are_error_lines_with_status_2() {
    let dir = scene("mask_errors", 128, 2);
    let m = polygon(128, 128);
    std::fs::write(dir.join("m.npy"), npy_bool(&[128, 128], &m)).unwrap();
    let small: Vec<bool> = (0..64 * 64).map(|i| m[(i / 64) * 128 + i % 64]).collect();
    std::fs::write(dir.join("small.npy"), npy_bool(&[64, 64], &small)).unwrap();
    std::fs::write(dir.join("line.npy"), npy_bool(&[128], &m[..128])).unwrap();
    let floats: Vec<f32> = m.iter().map(|&b| f32::from(u8::from(b))).collect();
    std::fs::write(dir.join("f.npy"), npy::write_f32(&[128, 128], &floats)).unwrap();
    std::fs::write(dir.join("m.txt"), "1").unwrap();
    std::fs::write(
        dir.join("empty.npy"),
        npy_bool(&[128, 128], &vec![false; 128 * 128]),
    )
    .unwrap();
    write_png8(&dir.join("small.png"), 96, 64, &vec![1.0; 96 * 64]);
    std::fs::write(dir.join("bad.png"), b"not a png").unwrap();
    let p = |name: &str| s(&dir.join(name)).to_string();
    let run = |extra: &[&str]| error_line(&pair(&dir, "out", extra));

    assert_eq!(
        run(&["--roi", &p("roi.json"), "--mask", &p("m.npy")]),
        "error: --roi and --mask are mutually exclusive"
    );
    assert_eq!(
        run(&["--mask", &p("small.npy")]),
        format!(
            "error: mask file {} has shape (64, 64), expected (128, 128)",
            p("small.npy")
        )
    );
    assert_eq!(
        run(&["--mask", &p("line.npy")]),
        format!(
            "error: mask file {} has shape (128,), expected (128, 128)",
            p("line.npy")
        )
    );
    assert_eq!(
        run(&["--mask", &p("f.npy")]),
        format!(
            "error: mask file {} must hold bool or uint8, got float32",
            p("f.npy")
        )
    );
    assert_eq!(
        run(&["--mask", &p("m.txt")]),
        format!(
            "error: mask file {} must be a .npy or .png file",
            p("m.txt")
        )
    );
    assert_eq!(
        run(&["--mask", &p("small.png")]),
        format!(
            "error: mask file {} has shape (64, 96), expected (128, 128)",
            p("small.png")
        )
    );
    assert_eq!(
        run(&["--mask", &p("empty.npy")]),
        "error: ROI mask is empty"
    );
    assert_eq!(
        run(&["--mask", &p("empty.npy"), "--crop", "off"]),
        "error: ROI mask is empty"
    );
    for name in ["none.npy", "bad.png"] {
        let line = run(&["--mask", &p(name)]);
        assert!(
            line.starts_with("error: cannot read mask file ") && line.contains(&p(name)),
            "{line}"
        );
    }
}

#[test]
fn mask_applies_to_the_scaled_image_and_to_sequences() {
    let dir = scene("mask_scaled", 128, 3);
    let mut m = vec![false; 64 * 64];
    for y in 12..52 {
        for x in 12..52 {
            m[y * 64 + x] = true;
        }
    }
    std::fs::write(dir.join("m.npy"), npy_bool(&[64, 64], &m)).unwrap();
    ok(&pair(
        &dir,
        "scaled",
        &[
            "--scale",
            "2",
            "--mask",
            s(&dir.join("m.npy")),
            "--crop",
            "off",
        ],
    ));
    let (shape, ux) = load_npy(&dir.join("scaled/ux.npy"));
    assert_eq!(shape, [64, 64]);
    assert!((0..64 * 64).all(|i| m[i] || ux[i].is_nan()));
    assert!(ux.iter().any(|v| v.is_finite()));
    let out = dir.join("seq");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir),
        "--mask",
        s(&dir.join("m.npy")),
        "--scale",
        "2",
        "--set",
        "outer_iters=2",
        "--out",
        s(&out),
    ]);
    ok(&o);
    assert_eq!(load_npy(&out.join("frame_00002/ux.npy")).0, [64, 64]);
}

#[test]
fn raw_writes_the_unthresholded_field_of_a_pair() {
    let dir = scene("raw_pair", 128, 2);
    for crop in ["off", "auto"] {
        let out = format!("out_{crop}");
        ok(&pair(
            &dir,
            &out,
            &["--roi", s(&dir.join("roi.json")), "--crop", crop, "--raw"],
        ));
        let d = dir.join(&out);
        let ((sx, ux), (_, uy)) = (load_npy(&d.join("ux.npy")), load_npy(&d.join("uy.npy")));
        let ((rs, rx), (_, ry)) = (
            load_npy(&d.join("raw_ux.npy")),
            load_npy(&d.join("raw_uy.npy")),
        );
        assert_eq!((sx, rs), (vec![128, 128], vec![128, 128]));
        let ok_px: Vec<usize> = (0..128 * 128).filter(|&i| ux[i].is_finite()).collect();
        assert!(!ok_px.is_empty());
        assert!(ok_px.iter().all(|&i| rx[i] == ux[i] && ry[i] == uy[i]));
        let finite = rx.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite > ok_px.len(),
            "{crop}: the exterior and the guard are kept"
        );
        assert!((median_of_finite(ok_px.iter().map(|&i| rx[i])) - 0.7).abs() < 0.05);
        assert!((median_of_finite(ok_px.iter().map(|&i| ry[i])) + 0.4).abs() < 0.05);
        let raw = std::fs::read(d.join("raw_ux.npy")).unwrap();
        assert!(
            raw.windows(8).any(|w| w == b"'<f4', '"),
            "raw_ux.npy is <f4"
        );
    }
    ok(&pair(&dir, "plain", &["--roi", s(&dir.join("roi.json"))]));
    assert!(!dir.join("plain/raw_ux.npy").exists() && !dir.join("plain/raw_uy.npy").exists());
}

#[test]
fn raw_is_written_per_frame_of_a_sequence() {
    let dir = scene("raw_seq", 128, 3);
    let out = dir.join("seq");
    let o = optflow(&[
        "seq",
        "--frames",
        s(&dir),
        "--roi",
        s(&dir.join("roi.json")),
        "--set",
        "outer_iters=2",
        "--raw",
        "--out",
        s(&out),
    ]);
    ok(&o);
    for k in [1usize, 2] {
        let d = out.join(format!("frame_{k:05}"));
        let (shape, rx) = load_npy(&d.join("raw_ux.npy"));
        let (_, ux) = load_npy(&d.join("ux.npy"));
        assert_eq!(shape, [128, 128]);
        let ok_px: Vec<usize> = (0..128 * 128).filter(|&i| ux[i].is_finite()).collect();
        assert!(ok_px.iter().all(|&i| rx[i] == ux[i]));
        assert!((median_of_finite(ok_px.iter().map(|&i| rx[i])) - 0.7 * k as f64).abs() < 0.05);
        assert!(d.join("raw_uy.npy").exists());
    }
    assert!(!out.join("raw_ux.npy").exists());
}

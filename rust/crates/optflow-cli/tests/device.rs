//! --device: cpu is the default and leaves timing.json as it was; gpu needs the gpu feature;
//! with it, the outputs match the CPU's within the GPU spec's limits.

mod common;

use common::*;

#[test]
fn an_unknown_device_is_a_usage_error() {
    let o = optflow(&[
        "pair", "--ref", "a.png", "--def", "b.png", "--out", "o", "--device", "tpu",
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).contains("argument --device: invalid choice: 'tpu' (choose from cpu, gpu)"),
        "{}",
        stderr(&o)
    );
}

#[cfg(not(feature = "gpu"))]
#[test]
fn gpu_without_the_feature_is_an_error() {
    let dir = scratch("device_no_feature");
    let (w, h) = (96, 96);
    let p = make_pattern(w, h, 2);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 0.5, 0.2));
    let out = dir.join("out");
    let o = optflow(&[
        "pair",
        "--ref",
        dir.join("a.png").to_str().unwrap(),
        "--def",
        dir.join("b.png").to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--device",
        "gpu",
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).contains("error: --device gpu needs a macOS build with --features gpu"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn cpu_is_the_default_and_timing_json_has_no_device_key() {
    let dir = scratch("device_cpu_default");
    let (w, h) = (96, 96);
    let p = make_pattern(w, h, 2);
    write_png16(&dir.join("a.png"), w, h, &render(&p, w, h, 0.0, 0.0));
    write_png16(&dir.join("b.png"), w, h, &render(&p, w, h, 0.5, 0.2));
    let (a, b) = (dir.join("a.png"), dir.join("b.png"));
    for (name, extra) in [("plain", None), ("cpu", Some("cpu"))] {
        let out = dir.join(name);
        let mut args = vec![
            "pair",
            "--ref",
            a.to_str().unwrap(),
            "--def",
            b.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--set",
            "max_disp=16",
        ];
        if let Some(d) = extra {
            args.extend(["--device", d]);
        }
        let o = optflow(&args);
        assert!(o.status.success(), "{}", stderr(&o));
        assert!(!stdout(&o).contains("device"), "{}", stdout(&o));
        let t = std::fs::read_to_string(out.join("timing.json")).unwrap();
        assert!(!t.contains("device"), "{t}");
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
mod gpu {
    use super::*;
    use std::path::Path;

    struct Loaded {
        ux: Vec<f32>,
        uy: Vec<f32>,
        conf: Vec<f32>,
        strain: Vec<Vec<f32>>,
    }

    impl Loaded {
        fn view(&self) -> optflow_core::compare::Fields<'_> {
            optflow_core::compare::Fields {
                ux: &self.ux,
                uy: &self.uy,
                conf: &self.conf,
                strain: self.strain.iter().map(Vec::as_slice).collect(),
            }
        }
    }

    /// The maps of two result directories agree in the robust tier of the GPU spec (the limits
    /// of `optflow_core::compare`, as `compare_outputs.py --tier robust`).
    fn assert_match(cpu: &Path, gpu: &Path) {
        let load = |d: &Path, n: &str| -> Vec<f32> {
            load_npy(&d.join(format!("{n}.npy")))
                .1
                .into_iter()
                .map(|v| v as f32)
                .collect()
        };
        let fields = |d: &Path| {
            let strain: Vec<Vec<f32>> = ["Exx", "Eyy", "Exy", "gmax", "vol"]
                .iter()
                .map(|n| load(d, n))
                .collect();
            Loaded {
                ux: load(d, "ux"),
                uy: load(d, "uy"),
                conf: load(d, "conf"),
                strain,
            }
        };
        let (c, g) = (fields(cpu), fields(gpu));
        let r = optflow_core::compare::robust(&c.view(), &g.view());
        assert!(r.ok(), "{r:?}");
        let (c, g) = (load(cpu, "valid"), load(gpu, "valid"));
        let same = c.iter().zip(&g).filter(|(x, y)| x == y).count();
        assert!(same as f64 / c.len() as f64 >= 0.999, "valid agreement");
    }

    /// The keys of timing.json in file order.
    fn timing_keys(dir: &Path) -> Vec<String> {
        // serde_json's map sorts, so take the top-level keys from the indented text
        std::fs::read_to_string(dir.join("timing.json"))
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("  \""))
            .filter_map(|l| l.split_once('"').map(|(k, _)| k.to_string()))
            .collect()
    }

    fn device_of(dir: &Path) -> Option<String> {
        let t: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("timing.json")).unwrap())
                .unwrap();
        t.get("device").map(|d| d.as_str().unwrap().to_string())
    }

    #[test]
    fn gpu_pair_and_seq_match_cpu_and_record_the_device() {
        if let Err(why) = optflow_cli::run::gpu_check() {
            eprintln!("skipped: no usable Metal GPU here ({why})");
            return;
        }
        let dir = scratch("device_gpu");
        let (w, h) = (256, 256);
        let p = make_pattern(w, h, 3);
        for k in 0..3 {
            let img = render(&p, w, h, 0.8 * k as f64, -0.4 * k as f64);
            write_png16(&dir.join(format!("f{k}.png")), w, h, &img);
        }
        let f = |k: usize| dir.join(format!("f{k}.png")).to_str().unwrap().to_string();

        // pair
        for d in ["cpu", "gpu"] {
            let out = dir.join(format!("pair_{d}"));
            let o = optflow(&[
                "pair",
                "--ref",
                &f(0),
                "--def",
                &f(1),
                "--out",
                out.to_str().unwrap(),
                "--device",
                d,
            ]);
            assert!(o.status.success(), "{d}: {}", stderr(&o));
            if d == "gpu" {
                assert!(stdout(&o).contains("'device': 'gpu'"), "{}", stdout(&o));
            }
        }
        let (c, g) = (dir.join("pair_cpu"), dir.join("pair_gpu"));
        assert_match(&c, &g);
        assert_eq!(timing_keys(&c), ["coarse", "fine", "strain"]);
        assert_eq!(timing_keys(&g), ["coarse", "device", "fine", "strain"]);
        assert_eq!(device_of(&g).as_deref(), Some("gpu"));
        assert_eq!(device_of(&c), None);

        // seq
        for d in ["cpu", "gpu"] {
            let out = dir.join(format!("seq_{d}"));
            let pattern = format!("{}/f*.png", dir.to_str().unwrap());
            let o = optflow(&[
                "seq",
                "--frames",
                &pattern,
                "--out",
                out.to_str().unwrap(),
                "--device",
                d,
            ]);
            assert!(o.status.success(), "{d}: {}", stderr(&o));
        }
        for k in [1, 2] {
            let name = format!("frame_{k:05}");
            let (c, g) = (
                dir.join("seq_cpu").join(&name),
                dir.join("seq_gpu").join(&name),
            );
            assert_match(&c, &g);
            assert_eq!(device_of(&g).as_deref(), Some("gpu"));
            assert_eq!(device_of(&c), None);
        }
    }
}

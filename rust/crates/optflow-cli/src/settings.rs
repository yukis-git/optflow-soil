//! Configuration of a run: the `--config` file and the `--set key=value` overrides, with the
//! rules and messages of `_parse_override` / `_config` in `python/optflow_ref/__main__.py`.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use optflow_core::config::{py_str_repr, Config};

/// Field names of [`Config`] in declaration order (as `dataclasses.fields` lists them).
pub const KEYS: [&str; 33] = [
    "grid_step",
    "poc_win",
    "max_disp",
    "poc_peak_min",
    "median_thresh",
    "lam",
    "scales",
    "n_orient",
    "bandwidth",
    "dc_cut",
    "sigma_w",
    "shape_order",
    "sigma_s",
    "outer_iters",
    "inner_iters",
    "conf_min",
    "edge_guard",
    "edge_guard_min_area",
    "phase_tol",
    "strain_win",
    "ref_update_conf",
    "seq_mode",
    "fb_max",
    "fb_radius",
    "fb_min_support",
    "fb_tol",
    "tavg",
    "tavg_align",
    "tavg_model",
    "prewarp_freeze_at",
    "crack_model",
    "xdic_RB",
    "xdic_iters",
];

/// Python's `repr(str)`, as [`optflow_core::config::py_str_repr`] writes it.
pub fn py_repr(s: &str) -> String {
    py_str_repr(s)
}

/// The text of `config.json`: [`Config::to_json`], which spells floats like Python's `json.dumps`
/// (`1e-07`, not `1e-7`), so the file is byte for byte what the Python tool writes.
pub fn config_json(cfg: &Config) -> String {
    cfg.to_json()
}

/// The most figures Python's `int(text)` converts (`sys.get_int_max_str_digits()`, 4300 unless the
/// interpreter is configured otherwise): more, leading zeros included, raise `ValueError`
/// ("Exceeds the limit"), which the Python command line reports as "cannot read ... as int".
const PY_INT_MAX_STR_DIGITS: usize = 4300;

/// Python's `int(text)` for the plain decimal forms: surrounding blanks, a sign, ASCII figures
/// with single underscores between them (`1_6`), at most [`PY_INT_MAX_STR_DIGITS`] figures. A
/// number beyond the `i128` range is an integer too and comes back as the limit of its sign; text
/// that is not a number at all, or has too many figures, is `None`, however long its leading
/// figures are.
fn parse_int(text: &str) -> Option<i128> {
    let text = text.trim();
    let (negative, figures) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let well_formed = !figures.starts_with('_')
        && !figures.ends_with('_')
        && !figures.contains("__")
        && figures.bytes().all(|b| b.is_ascii_digit() || b == b'_');
    if figures.is_empty() || !well_formed {
        return None;
    }
    let figures = figures.replace('_', "");
    if figures.len() > PY_INT_MAX_STR_DIGITS {
        return None;
    }
    let signed = if negative {
        format!("-{figures}")
    } else {
        figures
    };
    Some(
        signed
            .parse()
            .unwrap_or(if negative { i128::MIN } else { i128::MAX }),
    )
}

/// Python's `float(text)` for the plain forms (`1e3`, `.5`, `inf`, `nan`, surrounding blanks).
fn parse_float(text: &str) -> Option<f64> {
    text.trim().parse().ok()
}

/// An integer field: Python holds any `int`. A negative value becomes 0 (it fails the range rules
/// of `validate` exactly as 0 does) and one above `u64::MAX` is `None` (reported as out of range:
/// `validate` could not tell it from a smaller one, e.g. for the power-of-two rule of `poc_win`).
fn field_value(n: i128) -> Option<u64> {
    u64::try_from(n.max(0)).ok()
}

/// Applies one `key=value` override, parsed by the type of the field.
pub fn apply_override(cfg: &mut Config, kv: &str) -> Result<()> {
    let Some((k, v)) = kv.split_once('=') else {
        bail!(
            "--set expects key=value (e.g. --set sigma_w=6), got {}",
            py_repr(kv)
        );
    };
    if !KEYS.contains(&k) {
        bail!(
            "unknown config key {}; valid keys: {}",
            py_repr(k),
            KEYS.join(", ")
        );
    }
    let cannot = |kind: &str| anyhow!("--set {k}: cannot read {} as {kind}", py_repr(v));
    match k {
        "grid_step" | "poc_win" | "n_orient" | "shape_order" | "outer_iters" | "inner_iters"
        | "strain_win" => {
            let n = parse_int(v).ok_or_else(|| cannot("int"))?;
            let n = field_value(n)
                .ok_or_else(|| anyhow!("--set {k}: {} is out of range", py_repr(v)))?;
            let size = usize::try_from(n).unwrap_or(usize::MAX);
            match k {
                "grid_step" => cfg.grid_step = size,
                "poc_win" => cfg.poc_win = size,
                "n_orient" => cfg.n_orient = size,
                // shape_order is a u32; a larger value is still not 1 or 2 for `validate`
                "shape_order" => cfg.shape_order = u32::try_from(n).unwrap_or(u32::MAX),
                "outer_iters" => cfg.outer_iters = size,
                "inner_iters" => cfg.inner_iters = size,
                _ => cfg.strain_win = size,
            }
        }
        "fb_max" | "fb_min_support" | "tavg" | "prewarp_freeze_at" | "xdic_iters" => {
            // Python holds any int; the range rules of `validate` give the same verdict for the
            // value clamped to the i64 range
            let n = parse_int(v).ok_or_else(|| cannot("int"))?;
            let n = i64::try_from(n).unwrap_or(if n < 0 { i64::MIN } else { i64::MAX });
            match k {
                "fb_max" => cfg.fb_max = n,
                "fb_min_support" => cfg.fb_min_support = n,
                "tavg" => cfg.tavg = n,
                "xdic_iters" => cfg.xdic_iters = n,
                _ => cfg.prewarp_freeze_at = n,
            }
        }
        "tavg_align" => {
            cfg.tavg_align = match v.to_lowercase().as_str() {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => return Err(cannot("true or false")),
            }
        }
        "seq_mode" => cfg.seq_mode = v.to_string(),
        "tavg_model" => cfg.tavg_model = v.to_string(),
        "crack_model" => cfg.crack_model = v.to_string(),
        "scales" => {
            let list: Option<Vec<f64>> = v.split(',').map(parse_float).collect();
            cfg.scales = list.ok_or_else(|| cannot("a comma-separated list of numbers"))?;
        }
        _ => {
            let x = parse_float(v).ok_or_else(|| cannot("float"))?;
            match k {
                "max_disp" => cfg.max_disp = x,
                "poc_peak_min" => cfg.poc_peak_min = x,
                "median_thresh" => cfg.median_thresh = x,
                "lam" => cfg.lam = x,
                "bandwidth" => cfg.bandwidth = x,
                "dc_cut" => cfg.dc_cut = x,
                "sigma_w" => cfg.sigma_w = x,
                "sigma_s" => cfg.sigma_s = x,
                "conf_min" => cfg.conf_min = x,
                "edge_guard" => cfg.edge_guard = x,
                "edge_guard_min_area" => cfg.edge_guard_min_area = x,
                "phase_tol" => cfg.phase_tol = x,
                "fb_radius" => cfg.fb_radius = x,
                "fb_tol" => cfg.fb_tol = x,
                "xdic_RB" => cfg.xdic_rb = x,
                _ => cfg.ref_update_conf = x,
            }
        }
    }
    Ok(())
}

/// The run's configuration: defaults or the `--config` file, then the overrides in order, then
/// `validate()`.
pub fn config(path: Option<&Path>, overrides: &[String]) -> Result<Config> {
    let mut cfg = match path {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .map_err(|e| anyhow!("cannot read config {}: {e}", p.display()))?;
            Config::from_json(&text)?
        }
        None => Config::default(),
    };
    for kv in overrides {
        apply_override(&mut cfg, kv)?;
    }
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_of(kv: &str) -> String {
        let mut cfg = Config::default();
        apply_override(&mut cfg, kv).unwrap_err().to_string()
    }

    #[test]
    fn keys_are_the_config_fields() {
        let json = serde_json::to_value(Config::default()).unwrap();
        let mut fields: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let mut keys = KEYS.to_vec();
        fields.sort_unstable();
        keys.sort_unstable();
        assert_eq!(fields, keys);
    }

    #[test]
    fn overrides_parse_by_field_type() {
        let cfg = config(
            None,
            &[
                "scales=0.8,1.2".into(),
                "outer_iters=3".into(),
                "max_disp=64".into(),
                "lam=9.5".into(),
                "sigma_s= 5 ".into(),
            ],
        )
        .unwrap();
        assert_eq!(cfg.scales, vec![0.8, 1.2]);
        assert_eq!(cfg.outer_iters, 3);
        assert_eq!(cfg.max_disp, 64.0);
        assert_eq!(cfg.lam, 9.5);
        assert_eq!(cfg.sigma_s, 5.0);
    }

    /// Every key sets its own field and nothing else (a key left out of the match would fall
    /// through to the catch-all of its type and set another field).
    #[test]
    fn each_override_sets_its_own_field_only() {
        use serde_json::json;
        let table = [
            ("grid_step", "7", json!(7)),
            ("poc_win", "128", json!(128)),
            ("max_disp", "33.5", json!(33.5)),
            ("poc_peak_min", "0.123", json!(0.123)),
            ("median_thresh", "3.25", json!(3.25)),
            ("lam", "0.75", json!(0.75)),
            ("scales", "0.5,1.5,2.5", json!([0.5, 1.5, 2.5])),
            ("n_orient", "5", json!(5)),
            ("bandwidth", "0.31", json!(0.31)),
            ("dc_cut", "0.07", json!(0.07)),
            ("sigma_w", "2.5", json!(2.5)),
            ("shape_order", "2", json!(2)),
            ("sigma_s", "3.5", json!(3.5)),
            ("outer_iters", "9", json!(9)),
            ("inner_iters", "4", json!(4)),
            ("conf_min", "0.33", json!(0.33)),
            ("edge_guard", "11.5", json!(11.5)),
            ("edge_guard_min_area", "77.5", json!(77.5)),
            ("phase_tol", "0.45", json!(0.45)),
            ("strain_win", "21", json!(21)),
            ("ref_update_conf", "0.66", json!(0.66)),
            ("seq_mode", "hybrid", json!("hybrid")),
            ("fb_max", "7", json!(7)),
            ("fb_radius", "12.5", json!(12.5)),
            ("fb_min_support", "9", json!(9)),
            ("fb_tol", "0.31", json!(0.31)),
            ("tavg", "3", json!(3)),
            ("tavg_align", "True", json!(true)),
            ("tavg_model", "dense", json!("dense")),
            ("prewarp_freeze_at", "3", json!(3)),
            ("crack_model", "xdic", json!("xdic")),
            ("xdic_RB", "21.5", json!(21.5)),
            ("xdic_iters", "4", json!(4)),
        ];
        assert_eq!(table.iter().map(|t| t.0).collect::<Vec<_>>(), KEYS);
        let before = serde_json::to_value(Config::default()).unwrap();
        for (key, text, want) in table {
            assert_ne!(before[key], want, "{key}: the test value is the default");
            let mut cfg = Config::default();
            apply_override(&mut cfg, &format!("{key}={text}")).unwrap();
            let after = serde_json::to_value(&cfg).unwrap();
            for (field, value) in after.as_object().unwrap() {
                if field == key {
                    assert_eq!(value, &want, "--set {key}={text}");
                } else {
                    assert_eq!(value, &before[field], "--set {key}={text} changed {field}");
                }
            }
        }
    }

    #[test]
    fn bad_overrides_have_python_messages() {
        assert_eq!(
            err_of("outer_iters"),
            "--set expects key=value (e.g. --set sigma_w=6), got 'outer_iters'"
        );
        let unknown = err_of("sigma_x=3");
        assert!(unknown.starts_with("unknown config key 'sigma_x'; valid keys: grid_step, poc_win"));
        assert!(
            unknown.ends_with("tavg_model, prewarp_freeze_at, crack_model, xdic_RB, xdic_iters")
        );
        assert_eq!(
            err_of("outer_iters=2.5"),
            "--set outer_iters: cannot read '2.5' as int"
        );
        assert_eq!(
            err_of("outer_iters=2.0"),
            "--set outer_iters: cannot read '2.0' as int"
        );
        assert_eq!(
            err_of("sigma_w=abc"),
            "--set sigma_w: cannot read 'abc' as float"
        );
        assert_eq!(
            err_of("scales=0.6,x"),
            "--set scales: cannot read '0.6,x' as a comma-separated list of numbers"
        );
        assert_eq!(
            err_of("tavg_align=yes"),
            "--set tavg_align: cannot read 'yes' as true or false"
        );
        assert_eq!(err_of("tavg=1.5"), "--set tavg: cannot read '1.5' as int");
        assert_eq!(
            err_of("scales="),
            "--set scales: cannot read '' as a comma-separated list of numbers"
        );
    }

    #[test]
    fn overrides_are_validated_afterwards() {
        let msg = |kv: &str| config(None, &[kv.to_string()]).unwrap_err().to_string();
        assert!(msg("lam=nan").contains("finite"));
        assert!(msg("conf_min=2").contains("conf_min"));
        assert!(msg("grid_step=-1").contains("grid_step >= 1"));
        assert!(msg("outer_iters=-4").contains("outer_iters"));
        assert!(msg("poc_win=-64").contains("poc_win"));
    }

    #[test]
    fn huge_integers_are_out_of_range_not_zero() {
        // a value the field can hold is kept; one above u64::MAX is an error naming the key
        let mut cfg = Config::default();
        apply_override(&mut cfg, "grid_step=18446744073709551615").unwrap();
        assert_eq!(cfg.grid_step, usize::MAX);
        for kv in [
            "grid_step=18446744073709551616",
            "poc_win=99999999999999999999",
            "outer_iters=340282366920938463463374607431768211455",
            "strain_win=999999999999999999999999999999999999999999",
        ] {
            let (k, v) = kv.split_once('=').unwrap();
            assert_eq!(err_of(kv), format!("--set {k}: '{v}' is out of range"));
        }
        // shape_order is a u32: a larger value that fits u64 is not 1 or 2 for `validate`
        let msg = config(None, &["shape_order=4294967297".to_string()]).unwrap_err();
        assert_eq!(msg.to_string(), "shape_order must be 1 or 2");
        // negative integers, however large, are integers that fail the range rules
        for kv in [
            "grid_step=-5",
            "grid_step=-99999999999999999999999999999999999999999",
        ] {
            let msg = config(None, &[kv.to_string()]).unwrap_err().to_string();
            assert!(msg.contains("grid_step >= 1"), "{kv}: {msg}");
        }
    }

    #[test]
    fn huge_integers_with_trailing_garbage_cannot_be_read() {
        // Python's int() raises ValueError for these, whatever their size
        let nines = "9".repeat(40);
        for v in [
            format!("{nines}x"),
            format!("-{nines}x"),
            format!("{nines}.5"),
            format!("{nines}e3"),
            format!("{nines}__0"),
            format!("{nines}_"),
            format!("{nines} 1"),
            format!("{nines}-"),
            format!("{}z", "1".repeat(1000)),
            "_".to_string() + &nines,
            format!("+-{nines}"),
        ] {
            assert_eq!(
                err_of(&format!("grid_step={v}")),
                format!("--set grid_step: cannot read {} as int", py_repr(&v)),
                "{v}"
            );
        }
        // blanks around a huge integer, a sign and single underscores are fine: out of range
        for v in [
            format!(" {nines} "),
            format!("+{nines}"),
            format!("{nines}_0"),
            format!("9_{}", &nines[1..]),
        ] {
            assert_eq!(
                err_of(&format!("poc_win={v}")),
                format!("--set poc_win: {} is out of range", py_repr(&v)),
                "{v}"
            );
        }
    }

    #[test]
    fn integers_with_more_than_4300_figures_cannot_be_read() {
        // Python's int() raises ValueError ("Exceeds the limit (4300 digits)") for these, which
        // its command line reports as "cannot read": leading zeros and a sign do not matter, the
        // underscores are not figures
        // (the messages are thousands of characters long: a failure shows their ends only)
        let tail = |m: &str| m[m.len().saturating_sub(30)..].to_string();
        let ones = "1".repeat(4301);
        for v in [
            ones.clone(),
            format!("-{ones}"),
            format!("+{ones}"),
            format!(" {ones} "),
            format!("{}1", "0".repeat(4300)),
            "0".repeat(4301),
            "1_".repeat(4300) + "1",
        ] {
            let want = format!("--set grid_step: cannot read {} as int", py_repr(&v));
            let got = err_of(&format!("grid_step={v}"));
            assert!(got == want, "{} chars: ...{}", v.len(), tail(&got));
        }
        // 4300 figures are still an integer: out of range for the field, or one that fails the
        // range rules of validate (a negative one), or small (leading zeros)
        let ones = "1".repeat(4300);
        for v in [ones.clone(), format!("+{ones}"), "1_".repeat(4299) + "1"] {
            let want = format!("--set grid_step: {} is out of range", py_repr(&v));
            let got = err_of(&format!("grid_step={v}"));
            assert!(got == want, "{} chars: ...{}", v.len(), tail(&got));
        }
        let msg = config(None, &[format!("grid_step=-{ones}")]).unwrap_err();
        assert!(msg.to_string().contains("grid_step >= 1"), "{msg}");
        let mut cfg = Config::default();
        apply_override(&mut cfg, &format!("grid_step={}7", "0".repeat(4299))).unwrap();
        assert_eq!(cfg.grid_step, 7);
    }

    #[test]
    fn integers_may_have_single_underscores_between_their_figures() {
        // int("1_6") is 16 in Python
        let mut cfg = Config::default();
        apply_override(&mut cfg, "grid_step=1_6").unwrap();
        assert_eq!(cfg.grid_step, 16);
        apply_override(&mut cfg, "outer_iters= +1_0_0 ").unwrap();
        assert_eq!(cfg.outer_iters, 100);
        for v in ["_16", "16_", "1__6", "-_1", "1_-6", "_", "+", ""] {
            assert_eq!(
                err_of(&format!("grid_step={v}")),
                format!("--set grid_step: cannot read '{v}' as int"),
                "{v}"
            );
        }
    }

    #[test]
    fn config_json_uses_python_float_spelling() {
        let cfg = Config {
            edge_guard_min_area: 1e-7,
            phase_tol: 2.5e-5,
            ..Config::default()
        };
        let text = config_json(&cfg);
        assert!(text.contains("\"edge_guard_min_area\": 1e-07,"), "{text}");
        assert!(text.contains("\"phase_tol\": 2.5e-05,"), "{text}");
        assert!(text.contains("\"grid_step\": 16,"), "{text}");
        assert_eq!(config_json(&Config::default()), Config::default().to_json());
    }

    #[test]
    fn python_repr_of_strings() {
        assert_eq!(py_repr("a"), "'a'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("it's \"x\""), "'it\\'s \"x\"'");
        assert_eq!(py_repr("a\\b\n\t\u{1}"), "'a\\\\b\\n\\t\\x01'");
        // unprintable characters beyond ASCII are escaped as well
        assert_eq!(py_repr("a\u{200b}\u{85}\u{e9}"), "'a\\u200b\\x85\u{e9}'");
    }
}

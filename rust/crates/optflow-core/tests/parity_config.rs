//! Parity of `Config` JSON reading, writing and validation with the Python reference
//! (`python/optflow_ref/config.py`). `tests/fixtures/config_cases.json` is written by
//! `python/bench/make_parity_fixtures.py`: the default config text, texts Python rejects with the
//! exact message it raises, and the valid texts with the `to_json` text Python prints for them.
mod common;
use optflow_core::config::{Config, ConfigError};
use serde_json::{Map, Value};

fn cases() -> Value {
    let path = common::fixture_dir().join("config_cases.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

fn group<'a>(cases: &'a Value, name: &str) -> &'a [Value] {
    cases[name]
        .as_array()
        .unwrap_or_else(|| panic!("config_cases.json has no list {name:?}"))
}

fn str_of<'a>(case: &'a Value, key: &str) -> &'a str {
    case[key]
        .as_str()
        .unwrap_or_else(|| panic!("no string {key:?} in {case}"))
}

#[test]
fn config_default_json_matches_python() {
    let cases = cases();
    let python = str_of(&cases, "default_json");
    // the same text, not only the same values: both print with indent 2
    assert_eq!(Config::default().to_json(), python);
    assert_eq!(Config::from_json(python), Ok(Config::default()));
    let parsed: Value = serde_json::from_str(python).unwrap();
    assert_eq!(parsed.as_object().unwrap().len(), 30);
}

#[test]
fn config_from_json_errors_match_python_messages() {
    let cases = cases();
    let list = group(&cases, "from_json");
    assert!(list.len() >= 120, "only {} cases", list.len());
    for case in list {
        let (text, message) = (str_of(case, "text"), str_of(case, "message"));
        match Config::from_json(text) {
            Ok(c) => panic!("{text}: accepted as {c:?}, Python says {message:?}"),
            Err(e) => assert_eq!(e.to_string(), message, "text {text}"),
        }
    }
}

/// A non-finite or ordinary number of a `validate` case (`"nan"`, `"inf"`, `"-inf"` are strings
/// because JSON cannot carry them).
fn number(v: &Value) -> f64 {
    match v {
        Value::String(s) => match s.as_str() {
            "nan" => f64::NAN,
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            other => panic!("unknown number token {other:?}"),
        },
        Value::Number(n) => n.as_f64().unwrap(),
        other => panic!("not a number: {other}"),
    }
}

fn config_with(set: &Map<String, Value>) -> Config {
    let mut c = Config::default();
    for (key, v) in set {
        match key.as_str() {
            "max_disp" => c.max_disp = number(v),
            "poc_peak_min" => c.poc_peak_min = number(v),
            "median_thresh" => c.median_thresh = number(v),
            "lam" => c.lam = number(v),
            "bandwidth" => c.bandwidth = number(v),
            "dc_cut" => c.dc_cut = number(v),
            "sigma_w" => c.sigma_w = number(v),
            "sigma_s" => c.sigma_s = number(v),
            "conf_min" => c.conf_min = number(v),
            "edge_guard" => c.edge_guard = number(v),
            "edge_guard_min_area" => c.edge_guard_min_area = number(v),
            "phase_tol" => c.phase_tol = number(v),
            "ref_update_conf" => c.ref_update_conf = number(v),
            "fb_radius" => c.fb_radius = number(v),
            "fb_tol" => c.fb_tol = number(v),
            "scales" => c.scales = v.as_array().unwrap().iter().map(number).collect(),
            "grid_step" => c.grid_step = v.as_u64().unwrap() as usize,
            "poc_win" => c.poc_win = v.as_u64().unwrap() as usize,
            "n_orient" => c.n_orient = v.as_u64().unwrap() as usize,
            "shape_order" => c.shape_order = v.as_u64().unwrap() as u32,
            "outer_iters" => c.outer_iters = v.as_u64().unwrap() as usize,
            "inner_iters" => c.inner_iters = v.as_u64().unwrap() as usize,
            "strain_win" => c.strain_win = v.as_u64().unwrap() as usize,
            other => panic!("unknown field {other:?}"),
        }
    }
    c
}

#[test]
fn config_validate_with_non_finite_values_matches_python_messages() {
    let cases = cases();
    let list = group(&cases, "validate");
    // every float field with each of nan, inf, -inf, plus scales and the ordering cases
    assert!(list.len() >= 50, "only {} cases", list.len());
    for case in list {
        let set = case["set"].as_object().unwrap();
        let message = str_of(case, "message");
        match config_with(set).validate() {
            Ok(()) => panic!("{set:?}: accepted, Python says {message:?}"),
            Err(e) => {
                assert!(matches!(e, ConfigError::Invalid(_)), "{set:?}: {e:?}");
                assert_eq!(e.to_string(), message, "config {set:?}");
            }
        }
    }
}

#[test]
fn config_json_with_non_standard_literals_is_rejected() {
    // Python's json module reads NaN, Infinity and 1e999; serde_json does not, and the Rust reader
    // stays a strict JSON parser. Both sides reject the text, only the wording differs.
    let cases = cases();
    let list = group(&cases, "non_standard_json");
    assert!(!list.is_empty());
    for case in list {
        let text = str_of(case, "text");
        assert!(
            matches!(Config::from_json(text), Err(ConfigError::Json(_))),
            "{text}"
        );
    }
}

#[test]
fn config_malformed_json_is_a_syntax_error() {
    let cases = cases();
    let list = group(&cases, "syntax");
    assert!(!list.is_empty());
    for case in list {
        let text = str_of(case, "text");
        assert!(
            matches!(Config::from_json(text), Err(ConfigError::Json(_))),
            "{text:?}"
        );
    }
}

#[test]
fn config_valid_texts_read_and_print_like_python() {
    let cases = cases();
    let list = group(&cases, "valid");
    assert!(list.len() >= 5);
    for case in list {
        let (text, python) = (str_of(case, "text"), str_of(case, "to_json"));
        let cfg = Config::from_json(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        // the same text, float spelling included (1e-07, not 1e-7)
        assert_eq!(cfg.to_json(), python, "text {text}");
        assert_eq!(Config::from_json(&cfg.to_json()), Ok(cfg), "text {text}");
    }
}

// The texts below are what Python's `Config(...).to_json()` (json.dumps, indent 2) prints, from a
// scratch run of python/optflow_ref/config.py on the same field values.

/// Floats at the edges of Python's repr: exponent notation `e-07` / `e+22`, 17 digits, `-0.0`.
const PYTHON_EXTREME_FLOATS: &str = r#"{
  "grid_step": 16,
  "poc_win": 64,
  "max_disp": 1e+22,
  "poc_peak_min": 1e-07,
  "median_thresh": 5e-324,
  "lam": 1.7976931348623157e+308,
  "scales": [
    0.30000000000000004,
    1e-05,
    123456789012345.6,
    1e+16
  ],
  "n_orient": 4,
  "bandwidth": 1.5e-05,
  "dc_cut": -0.0,
  "sigma_w": 9.99e-05,
  "shape_order": 1,
  "sigma_s": 1234567.0,
  "outer_iters": 8,
  "inner_iters": 2,
  "conf_min": 0.0001,
  "edge_guard": 2.5e-10,
  "edge_guard_min_area": -3e-06,
  "phase_tol": 1000000000000000.0,
  "strain_win": 15,
  "ref_update_conf": 0.1,
  "seq_mode": "chain",
  "fb_max": 10,
  "fb_radius": -1.0,
  "fb_min_support": 20,
  "fb_tol": 0.5,
  "tavg": 0,
  "tavg_align": false,
  "tavg_model": "affine",
  "prewarp_freeze_at": 0
}"#;

/// Long reprs and the switch between positional and exponent notation at 1e16 and 1e-5.
const PYTHON_LONG_REPRS: &str = r#"{
  "grid_step": 16,
  "poc_win": 64,
  "max_disp": 1.2345678901234567e+19,
  "poc_peak_min": 0.30000000000000004,
  "median_thresh": 0.6666666666666666,
  "lam": 1e+100,
  "scales": [
    1.5e+16,
    9007199254740992.0,
    1.2345678901234568e+17,
    0.00012345
  ],
  "n_orient": 4,
  "bandwidth": 1e-100,
  "dc_cut": 0.0001,
  "sigma_w": 255.99999999999997,
  "shape_order": 1,
  "sigma_s": 1e-15,
  "outer_iters": 8,
  "inner_iters": 2,
  "conf_min": 5e-05,
  "edge_guard": 1e-07,
  "edge_guard_min_area": 100.0,
  "phase_tol": 4.35e-06,
  "strain_win": 15,
  "ref_update_conf": 1e-300,
  "seq_mode": "chain",
  "fb_max": 10,
  "fb_radius": -1.0,
  "fb_min_support": 20,
  "fb_tol": 0.5,
  "tavg": 0,
  "tavg_align": false,
  "tavg_model": "affine",
  "prewarp_freeze_at": 0
}"#;

/// `json.dumps` writes the non-finite floats as `NaN`, `Infinity`, `-Infinity` and an empty list
/// as `[]`.
const PYTHON_NON_FINITE: &str = r#"{
  "grid_step": 16,
  "poc_win": 64,
  "max_disp": NaN,
  "poc_peak_min": Infinity,
  "median_thresh": -Infinity,
  "lam": 8.0,
  "scales": [],
  "n_orient": 4,
  "bandwidth": 0.5,
  "dc_cut": 2.0,
  "sigma_w": 4.0,
  "shape_order": 1,
  "sigma_s": 4.0,
  "outer_iters": 8,
  "inner_iters": 2,
  "conf_min": 0.2,
  "edge_guard": -1.0,
  "edge_guard_min_area": -1.0,
  "phase_tol": 0.4,
  "strain_win": 15,
  "ref_update_conf": 0.5,
  "seq_mode": "chain",
  "fb_max": 10,
  "fb_radius": -1.0,
  "fb_min_support": 20,
  "fb_tol": 0.5,
  "tavg": 0,
  "tavg_align": false,
  "tavg_model": "affine",
  "prewarp_freeze_at": 0
}"#;

/// Every field different from its default.
const PYTHON_OTHER_FIELDS: &str = r#"{
  "grid_step": 8,
  "poc_win": 128,
  "max_disp": 40.5,
  "poc_peak_min": 0.25,
  "median_thresh": 3.5,
  "lam": 9.0,
  "scales": [
    0.7,
    1.1
  ],
  "n_orient": 6,
  "bandwidth": 0.6,
  "dc_cut": 2.5,
  "sigma_w": 5.0,
  "shape_order": 2,
  "sigma_s": 3.0,
  "outer_iters": 5,
  "inner_iters": 3,
  "conf_min": 0.3,
  "edge_guard": 12.0,
  "edge_guard_min_area": 50.0,
  "phase_tol": 0.5,
  "strain_win": 9,
  "ref_update_conf": 0.6,
  "seq_mode": "hybrid",
  "fb_max": 6,
  "fb_radius": 12.5,
  "fb_min_support": 8,
  "fb_tol": 0.25,
  "tavg": 2,
  "tavg_align": true,
  "tavg_model": "dense",
  "prewarp_freeze_at": 3
}"#;

#[test]
fn config_to_json_spells_extreme_floats_like_python() {
    let extreme = Config {
        max_disp: 1e22,
        poc_peak_min: 1e-7,
        median_thresh: 5e-324,
        lam: 1.7976931348623157e308,
        scales: vec![0.1 + 0.2, 1e-5, 123456789012345.6, 1e16],
        bandwidth: 1.5e-5,
        dc_cut: -0.0,
        sigma_w: 9.99e-5,
        sigma_s: 1234567.0,
        conf_min: 0.0001,
        edge_guard: 2.5e-10,
        edge_guard_min_area: -3e-6,
        phase_tol: 1e15,
        ref_update_conf: 0.1,
        ..Config::default()
    };
    assert_eq!(extreme.to_json(), PYTHON_EXTREME_FLOATS);

    let long = Config {
        max_disp: 1.2345678901234567e19,
        poc_peak_min: 0.1 + 0.2,
        median_thresh: 2.0 / 3.0,
        lam: 1e100,
        scales: vec![1.5e16, 9007199254740993.0, 123456789012345678.0, 0.00012345],
        bandwidth: 1e-100,
        dc_cut: 1e-4,
        // read back by from_json below, so within the upper bounds of validate (sigma_w <= 256)
        sigma_w: 255.99999999999997,
        sigma_s: 1e-15,
        conf_min: 5e-5,
        // an explicit guard: the automatic one (lam = 1e100, bandwidth = 1e-100) is over the bound
        edge_guard: 1e-7,
        edge_guard_min_area: 100.0,
        phase_tol: 4.35e-6,
        ref_update_conf: 1e-300,
        ..Config::default()
    };
    assert_eq!(long.to_json(), PYTHON_LONG_REPRS);

    let non_finite = Config {
        max_disp: f64::NAN,
        poc_peak_min: f64::INFINITY,
        median_thresh: f64::NEG_INFINITY,
        scales: vec![],
        ..Config::default()
    };
    assert_eq!(non_finite.to_json(), PYTHON_NON_FINITE);

    let other = Config {
        grid_step: 8,
        poc_win: 128,
        max_disp: 40.5,
        poc_peak_min: 0.25,
        median_thresh: 3.5,
        lam: 9.0,
        scales: vec![0.7, 1.1],
        n_orient: 6,
        bandwidth: 0.6,
        dc_cut: 2.5,
        sigma_w: 5.0,
        shape_order: 2,
        sigma_s: 3.0,
        outer_iters: 5,
        inner_iters: 3,
        conf_min: 0.3,
        edge_guard: 12.0,
        edge_guard_min_area: 50.0,
        phase_tol: 0.5,
        strain_win: 9,
        ref_update_conf: 0.6,
        seq_mode: "hybrid".to_string(),
        fb_max: 6,
        fb_radius: 12.5,
        fb_min_support: 8,
        fb_tol: 0.25,
        tavg: 2,
        tavg_align: true,
        tavg_model: "dense".to_string(),
        prewarp_freeze_at: 3,
        // the crack keys stay at their defaults: Python leaves them out of the JSON
        ..Config::default()
    };
    assert_eq!(other.to_json(), PYTHON_OTHER_FIELDS);
    // and it reads back to the same config, field by field
    assert_eq!(Config::from_json(PYTHON_OTHER_FIELDS), Ok(other));
    assert_eq!(
        Config::from_json(PYTHON_LONG_REPRS).unwrap().to_json(),
        PYTHON_LONG_REPRS
    );
}

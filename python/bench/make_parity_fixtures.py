"""Write Python-reference inputs and expected outputs for the Rust parity tests.

Output: rust/crates/optflow-core/tests/fixtures/<name>.npy (float32, to keep the fixtures small), meta.json
and config_cases.json (Config JSON texts with the error messages Python raises for them).
Re-run whenever python/optflow_ref changes; the Rust tests compare against these files.
"""
import json
import os
import platform
import sys
from dataclasses import fields

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from optflow_ref import interp, phaseflow, poc, strain, synth, track  # noqa: E402
from optflow_ref.config import Config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "rust", "crates", "optflow-core", "tests", "fixtures")


def save(name, a):
    a = np.asarray(a)
    if a.dtype == np.float64 or a.dtype == np.int64:
        a = a.astype(np.float32)
    np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(a))


def grid_dict(g):
    return {"xs": [int(v) for v in g.xs], "ys": [int(v) for v in g.ys]}


# the float and int fields of Config in declaration order (scales, the tuple, is neither): a new field is
# picked up here, so the generated cases cover it
FLOAT_FIELDS = [f.name for f in fields(Config) if f.type is float]
INT_FIELDS = [f.name for f in fields(Config) if f.type is int]
NON_FINITE = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _message(fn):
    """The message of the ValueError that fn() must raise."""
    try:
        fn()
    except ValueError as e:
        return str(e)
    raise AssertionError("expected a ValueError")


def _decode(v):
    """Config values of a 'validate' case: the strings nan / inf / -inf stand for the floats JSON cannot hold."""
    if isinstance(v, list):
        return tuple(_decode(x) for x in v)
    return NON_FINITE[v] if isinstance(v, str) else v


def config_cases():
    """Config JSON texts and Python's exact error messages, for the Rust config tests.

    from_json: standard-JSON texts that Config.from_json must reject; Rust returns the same message.
    validate: configs with non-finite values, which JSON text cannot carry (nan / inf / -inf are strings here);
    Rust builds the Config directly and calls validate().
    non_standard_json: texts with the NaN / Infinity / 1e999 literals Python's json module accepts; Rust's
    parser rejects them as syntax errors, so only the fact of the rejection is compared.
    syntax: malformed texts (Python's JSONDecodeError wording differs from serde_json's).
    valid: texts Python accepts, with the to_json text of the parsed config.
    """
    from_json = []
    for name in FLOAT_FIELDS:
        from_json.append(json.dumps({name: "x"}))
    for name in INT_FIELDS:
        from_json += [json.dumps({name: 2.5}), '{"%s": true}' % name]
    from_json += [
        # numbers that are not integers, with Python's repr in the message
        '{"poc_win": 64.0}', '{"strain_win": 15.0}', '{"grid_step": 1e16}', '{"grid_step": 1e-5}',
        '{"grid_step": 1e15}', '{"grid_step": 123456789.125}', '{"grid_step": 0.0001}', '{"grid_step": 1.5e300}',
        '{"poc_win": -0.0}', '{"grid_step": 5e-324}', '{"grid_step": 1.7976931348623157e308}',
        '{"outer_iters": null}', '{"inner_iters": "2"}', '{"strain_win": [15]}', '{"n_orient": false}',
        # wrong type for a float field, each with a different repr
        '{"lam": "8"}', '{"lam": true}', '{"lam": null}', '{"lam": [8]}', '{"lam": {"a": null, "b": [1, 2.5]}}',
        '{"max_disp": "it\'s"}', '{"bandwidth": "say \\"hi\\""}', '{"conf_min": "a\\nb\\t\\\\"}',
        '{"phase_tol": "\u00e9"}', '{"ref_update_conf": false}', '{"lam": "it\'s \\"q\\""}', '{"lam": ""}',
        # scales
        '{"scales": 1.0}', '{"scales": [1.0, "a"]}', '{"scales": [true]}', '{"scales": "0.6,1.0"}',
        '{"scales": null}', '{"scales": [[1.0]]}', '{"scales": {}}', '{"scales": [1.0, null]}',
        # the first bad key in the text is reported, a repeated key keeps its first position and last value
        '{"lam": "x", "grid_step": 2.5}', '{"grid_step": 2.5, "lam": "x"}', '{"lam": "x", "grid_step": 2.5, "lam": 3}',
        # not an object
        '[1, 2]', '3', '"str"', 'null', 'true', '[]',
        # unknown keys come first and are sorted
        '{"lambda": 3, "alpha": 1}', '{"it\'s": 1}', '{"lam": 8, "zzz": 1, "Zeta": 2}', '{"lam": "x", "foo": 1}',
        # ranges
        '{"scales": []}', '{"poc_win": 48}', '{"poc_win": 8}', '{"poc_win": 0}', '{"poc_win": -64}',
        '{"grid_step": 0}', '{"grid_step": -1}', '{"strain_win": 14}', '{"strain_win": 3}', '{"strain_win": -7}',
        '{"dc_cut": 0}', '{"dc_cut": -1.0}', '{"phase_tol": 0}', '{"phase_tol": -0.4}',
        '{"shape_order": 0}', '{"shape_order": 3}', '{"shape_order": -1}', '{"shape_order": 4294967297}',
        '{"lam": 4}', '{"sigma_w": 0}', '{"sigma_s": -1}', '{"scales": [0.3, 1.0]}', '{"lam": -8}',
        '{"bandwidth": 0}', '{"bandwidth": -0.5}', '{"max_disp": 0}', '{"max_disp": -3}', '{"median_thresh": 0}',
        '{"median_thresh": -1}', '{"n_orient": 0}', '{"outer_iters": 0}', '{"inner_iters": 0}', '{"n_orient": -2}',
        '{"conf_min": -0.1}', '{"conf_min": 1.5}', '{"conf_min": 2}', '{"poc_peak_min": -0.1}',
        '{"poc_peak_min": 1.1}', '{"poc_peak_min": 2}',
        # several violations at once: the first rule in Python's order wins
        '{"scales": [], "poc_win": 48}', '{"poc_win": 48, "grid_step": 0}', '{"grid_step": 0, "dc_cut": 0}',
        '{"dc_cut": 0, "phase_tol": 0}', '{"phase_tol": 0, "shape_order": 3}', '{"shape_order": 3, "lam": 4}',
        '{"lam": 4, "bandwidth": 0}', '{"bandwidth": 0, "n_orient": 0}', '{"n_orient": 0, "conf_min": 2}',
        '{"max_disp": 0, "bandwidth": 0, "median_thresh": 0}', '{"scales": [0.3], "conf_min": 2}',
        '{"conf_min": 2, "poc_peak_min": -1}', '{"outer_iters": 0, "inner_iters": 0}',
        # sequence-mode, hybrid, temporal-averaging and freeze keys: types, ranges and the order of the rules
        '{"seq_mode": 3}', '{"seq_mode": null}', '{"seq_mode": "x"}', '{"seq_mode": "Hybrid"}', '{"seq_mode": ""}',
        '{"seq_mode": "it\'s"}', '{"seq_mode": "q\\u00e9\\n"}', '{"tavg_model": true}', '{"tavg_model": "linear"}',
        '{"tavg_align": 1}', '{"tavg_align": "true"}', '{"tavg_align": null}', '{"fb_radius": 0}', '{"fb_radius": 0.0}',
        '{"fb_tol": 0}', '{"fb_tol": -0.1}', '{"fb_max": -1}', '{"fb_min_support": -1}', '{"tavg": -1}', '{"tavg": 65}',
        '{"tavg": 1000000000000}', '{"prewarp_freeze_at": -1}', '{"prewarp_freeze_at": 2.0}',
        '{"seq_mode": "x", "scales": []}', '{"scales": [], "fb_max": -1}', '{"fb_max": -1, "fb_tol": 0}',
        '{"fb_tol": 0, "tavg": 65}', '{"tavg": 65, "tavg_model": "x"}', '{"tavg_model": "x", "prewarp_freeze_at": -1}',
        '{"prewarp_freeze_at": -1, "poc_win": 2048}', '{"fb_max": -1, "conf_min": 2}', '{"poc_win": 2048, "tavg": 65}',
        # upper bounds (the last rule of validate): each just above its limit; integers from 2**63 up to the largest
        # the Rust reader takes (u64::MAX) give the same message on both sides (a larger one is "an integer in range")
        '{"poc_win": 2048}', '{"strain_win": 257}', '{"sigma_w": 256.0001}', '{"sigma_s": 257}', '{"sigma_w": 1e300}',
        '{"n_orient": 22}', '{"n_orient": 65, "scales": [1.0]}', '{"n_orient": 17, "scales": [0.6, 1.0, 1.6, 2.0]}',
        '{"n_orient": 1, "scales": [%s]}' % ", ".join(["1.0"] * 65),
        '{"poc_win": 9223372036854775808}', '{"poc_win": 4611686018427387904}',
        '{"strain_win": 18446744073709551615}', '{"strain_win": 9223372036854775809}',
        '{"n_orient": 9223372036854775808}', '{"n_orient": 18446744073709551615}',
        # the bounds come last: a config that also breaks an earlier rule gets that message
        '{"poc_win": 2048, "grid_step": 0}', '{"poc_win": 3000}', '{"strain_win": 256}', '{"strain_win": 301, "conf_min": 2}',
        '{"sigma_w": 1000, "lam": 4}', '{"sigma_s": 1000, "bandwidth": 0}', '{"n_orient": 22, "median_thresh": 0}',
        '{"poc_win": 2048, "n_orient": 0}', '{"scales": [], "poc_win": 2048}', '{"poc_win": 2048, "strain_win": 257, "sigma_w": 300}',
        # grid_step <= 4096 and a resolved edge guard <= 256 (the rule after the bounds): just above each limit, the
        # automatic guard (3 lam max(scales) / (2 pi bandwidth)) too, integers up to u64::MAX
        '{"grid_step": 4097}', '{"grid_step": 9223372036854775808}', '{"grid_step": 18446744073709551615}',
        '{"edge_guard": 256.5}', '{"edge_guard": 256.0001}', '{"edge_guard": 1e300}', '{"bandwidth": 0.001}',
        '{"lam": 268.2, "scales": [1.0]}', '{"grid_step": 4097, "edge_guard": 300}',
        # the guard rule comes after the bounds and every other rule
        '{"grid_step": 5000, "poc_win": 2048}', '{"grid_step": 5000, "sigma_w": 300}', '{"edge_guard": 1e300, "strain_win": 301}',
        '{"grid_step": 9223372036854775808, "scales": []}', '{"grid_step": 9223372036854775808, "poc_win": 3000}',
        '{"grid_step": 5000, "conf_min": 2}', '{"edge_guard": 300, "bandwidth": 0}',
    ]
    cases = {"default_json": Config().to_json(), "from_json": [], "validate": [], "non_standard_json": [],
             "syntax": [], "valid": []}
    for text in from_json:
        cases["from_json"].append({"text": text, "message": _message(lambda t=text: Config.from_json(t))})

    validate = [{name: tok} for name in FLOAT_FIELDS for tok in NON_FINITE]
    validate += [{"scales": s} for s in ([0.6, "nan", 1.6], [1.0, "inf"], [1.0, "-inf"], ["nan", 1.0], ["inf"])]
    # (the loop above covers: nan passes the `<= 0` checks and is caught by the finite check, -inf trips the earlier
    # range check of dc_cut, phase_tol, sigma_w, sigma_s and lam)
    validate += [
        # order of the finite checks: the field list, then scales, then the ranges
        {"lam": "nan", "max_disp": "inf"}, {"scales": ["nan"], "lam": "nan"}, {"scales": ["inf"], "bandwidth": 0.0},
        {"bandwidth": "nan", "max_disp": 0.0}, {"conf_min": "nan", "poc_peak_min": 2.0},
        {"scales": [], "lam": "nan"}, {"lam": "nan", "poc_win": 48}, {"edge_guard": "inf", "n_orient": 0},
        # upper bounds, set on the Config directly (huge ints that JSON text would give "an integer in range" for)
        {"sigma_w": 256.0001}, {"sigma_s": 1e300}, {"strain_win": 257}, {"poc_win": 2048}, {"poc_win": 2**63},
        {"n_orient": 22}, {"n_orient": 2**63}, {"n_orient": 2**64 - 1}, {"n_orient": 2**64 - 1, "scales": [1.0]},
        {"n_orient": 1, "scales": [1.0] * 65}, {"strain_win": 2**64 - 1}, {"poc_win": 2**62},
        {"sigma_w": "inf", "poc_win": 2048}, {"sigma_w": "nan", "n_orient": 100}, {"scales": ["inf"], "n_orient": 100},
        {"strain_win": 301, "conf_min": 2.0}, {"n_orient": 100, "bandwidth": 0.0},
        # grid_step and the resolved edge guard (explicit or automatic, even overflowing to inf)
        {"grid_step": 4097}, {"grid_step": 2**31}, {"grid_step": 2**63}, {"grid_step": 2**64 - 1},
        {"edge_guard": 256.0001}, {"edge_guard": 1e300}, {"bandwidth": 0.01}, {"bandwidth": 1e-300},
        {"lam": 268.2, "scales": [1.0]}, {"lam": 1e308, "scales": [10.0]}, {"grid_step": 4097, "edge_guard": 300.0},
        {"grid_step": 2**63, "poc_win": 2048}, {"grid_step": 2**63, "n_orient": 22}, {"grid_step": 2**63, "scales": []},
        {"edge_guard": 1e300, "conf_min": 2.0},
    ]
    for spec in validate:
        kw = {k: _decode(v) for k, v in spec.items()}
        cases["validate"].append({"set": spec, "message": _message(lambda kw=kw: Config(**kw).validate())})

    for text in ('{"lam": NaN}', '{"max_disp": Infinity}', '{"scales": [1.0, -Infinity]}', '{"grid_step": NaN}',
                 '{"lam": 1e999}', '{"scales": [1e999]}', '{"conf_min": -Infinity}'):
        cases["non_standard_json"].append({"text": text, "message": _message(lambda t=text: Config.from_json(t))})
    for text in ('{', '{"lam": }', '', "{'lam': 1}", '{"lam": 1,}'):
        cases["syntax"].append({"text": text, "message": _message(lambda t=text: Config.from_json(t))})

    for text in ('{}', '{"lam": 10, "max_disp": 64, "scales": [0.5, 1, 2], "outer_iters": 4}',
                 '{"conf_min": 0, "poc_peak_min": 1, "n_orient": 1, "outer_iters": 1, "inner_iters": 1}',
                 '{"edge_guard": -1, "edge_guard_min_area": -1, "ref_update_conf": 2}',
                 '{"scales": [0.375, 1.0]}', '{"lam": 1e-5, "max_disp": 1e16, "scales": [400000, 1e6]}',
                 '{"poc_win": 128, "grid_step": 8, "strain_win": 5, "shape_order": 2}',
                 '{"max_disp": 1e-7, "dc_cut": 5e-324, "sigma_w": 256.0}',
                 # every upper bound exactly at its limit
                 '{"poc_win": 1024, "strain_win": 255, "sigma_w": 256, "sigma_s": 256.0}',
                 '{"n_orient": 64, "scales": [1.0]}', '{"n_orient": 16, "scales": [0.6, 1.0, 1.6, 2.0]}',
                 '{"n_orient": 1, "scales": [%s]}' % ", ".join(["1.0"] * 64),
                 # grid_step and the resolved edge guard exactly at their limits
                 '{"grid_step": 4096, "edge_guard": 256}', '{"grid_step": 4096}', '{"edge_guard": 256.0}',
                 '{"lam": 268.0, "scales": [1.0]}', '{"edge_guard": 256, "bandwidth": 1e-9}',
                 '{"seq_mode": "hybrid", "fb_max": 0, "fb_radius": 12.5, "fb_min_support": 0, "fb_tol": 0.25}',
                 '{"seq_mode": "hybrid", "tavg": 3, "tavg_align": true, "tavg_model": "dense", "prewarp_freeze_at": 3}',
                 '{"tavg": 64, "fb_radius": -3, "fb_tol": 1e-9, "prewarp_freeze_at": 1000000}'):
        cases["valid"].append({"text": text, "to_json": Config.from_json(text).to_json()})
    return cases


def main():
    os.makedirs(OUT, exist_ok=True)
    # which interpreter and libraries wrote the fixtures (the Rust tests ignore it); a diff of the numbers after a
    # regeneration on another machine is then explained or ruled out by the versions
    meta = {"provenance": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    rng = np.random.default_rng(0)

    # --- interp: B-spline sampling (incl. out-of-range coords -> clamped), warp, Gaussian blur
    img = rng.normal(size=(40, 50))
    x = rng.uniform(-3, 53, 400)
    y = rng.uniform(-3, 43, 400)
    save("interp_img", img)
    # sample coordinates stay float64: the Rust tests feed them to the sampler at full precision
    np.save(os.path.join(OUT, "interp_x.npy"), np.ascontiguousarray(x))
    np.save(os.path.join(OUT, "interp_y.npy"), np.ascontiguousarray(y))
    save("interp_sample", interp.sample(interp.bspline_coeffs(img), x, y))
    ux = rng.uniform(-2, 2, img.shape); uy = rng.uniform(-2, 2, img.shape)
    save("interp_warp_ux", ux); save("interp_warp_uy", uy); save("interp_warp", interp.warp(img, ux, uy))
    for s in (1.0, 2.0, 4.0):
        save(f"interp_blur_{s:g}", interp.gauss_blur(img, s))

    # --- gabor bank responses on a small synthetic image (default config)
    n = 64
    gimg = synth.render(synth.make_pattern(n, n, seed=3), n, n)
    cfg = Config()
    bank, fx, fy = phaseflow.gabor_bank((n + 2 * phaseflow.PAD, n + 2 * phaseflow.PAD), cfg.lam, cfg.scales,
                                        cfg.n_orient, cfg.bandwidth, cfg.dc_cut)
    ref = phaseflow.reference_phase(gimg, bank, fx, fy)
    save("gabor_img", gimg)
    save("gabor_r_re", np.stack([r.real for r, _, _, _ in ref]))
    save("gabor_r_im", np.stack([r.imag for r, _, _, _ in ref]))
    save("gabor_gx", np.stack([g for _, g, _, _ in ref]))
    save("gabor_gy", np.stack([g for _, _, g, _ in ref]))

    # --- POC: one window shift, a full coarse_flow, a narrow strip with a large shift, level_window table
    n = 128
    p = synth.make_pattern(n, n, seed=5)
    a = synth.render(p, n, n); b = synth.render(p, n, n, synth.translation(5.3, -2.7))
    save("poc_a", a); save("poc_b", b)
    dx, dy, pk = poc.poc_shift(a[32:96, 32:96], b[32:96, 32:96])
    meta["poc_shift"] = {"window": [32, 96, 32, 96], "dx": dx, "dy": dy, "peak": pk}
    g = poc.coarse_flow(a, b, Config())
    assert g.ux.size > 0 and len(g.ys) > 0 and len(g.xs) > 0
    save("poc_grid_ux", g.ux); save("poc_grid_uy", g.uy); save("poc_grid_peak", g.peak)
    meta["poc_grid"] = grid_dict(g)
    ps = synth.make_pattern(96, 512, seed=8, margin=200)
    sa = synth.render(ps, 96, 512); sb = synth.render(ps, 96, 512, synth.translation(60.4, 1.1))
    save("poc_strip_a", sa); save("poc_strip_b", sb)
    gs = poc.coarse_flow(sa, sb, Config(max_disp=100))
    assert gs.ux.size > 0 and len(gs.ys) > 0 and len(gs.xs) > 0
    save("poc_strip_ux", gs.ux); save("poc_strip_uy", gs.uy)
    meta["poc_strip"] = grid_dict(gs) | {"max_disp": 100}
    meta["poc_strip"]["n_levels"] = poc.n_levels(100, 64, (96, 512))
    shapes = [(500, 500), (40, 600), (600, 40), (10, 600), (96, 96), (95, 95), (24, 256), (105, 636), (26, 159)]
    meta["level_window"] = [[h, w, poc.level_window((h, w), 64)] for h, w in shapes]
    meta["n_levels"] = [[d, h, w, poc.n_levels(d, 64, (h, w))]
                        for d in (16, 64, 100, 200) for h, w in [(1000, 1000), (100, 1000), (40, 1000), (420, 2544)]]
    # (max_disp, win, h, w, cy, cx): the window coarse_flow uses at the coarsest level, not that of shape >> L (95 x 95)
    meta["capture_range"] = [[d, win, h, w, *poc.capture_range(d, win, (h, w))]
                             for d, win, h, w in [(128, 64, 512, 512), (32, 64, 512, 512), (16, 64, 1000, 1000),
                                                  (128, 64, 421, 3002), (32, 64, 95, 95), (100, 64, 96, 512),
                                                  (128, 64, 40, 1000), (500, 64, 700, 333), (200, 64, 420, 2544),
                                                  (64, 32, 200, 150), (8, 32, 1000, 1000), (1e6, 64, 96, 96),
                                                  (0.5, 128, 300, 300), (300, 128, 1000, 400)]]

    # --- coarse_flow with a dense initial guess (sloped, near the true shift of the poc_a / poc_b pair), and
    # with a blank second image where no level finds a peak: the result is the guess resampled level by level
    n = 128
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ig_x = (4.0 + 0.02 * xx).astype(np.float32); ig_y = (-2.0 + 0.01 * yy).astype(np.float32)
    save("poc_init_guess_ux", ig_x); save("poc_init_guess_uy", ig_y)
    guess = (ig_x.astype(np.float64), ig_y.astype(np.float64))
    gi = poc.coarse_flow(a, b, Config(), init_dense=guess)  # a, b: the poc_a / poc_b pair from above
    save("poc_init_ux", gi.ux); save("poc_init_uy", gi.uy); save("poc_init_peak", gi.peak)
    meta["poc_init"] = grid_dict(gi)
    gf = poc.coarse_flow(a, np.full((n, n), 0.5), Config(), init_dense=guess)
    assert (gf.peak == 0).all()
    save("poc_fallback_ux", gf.ux); save("poc_fallback_uy", gf.uy); save("poc_fallback_peak", gf.peak)
    meta["poc_fallback"] = grid_dict(gf)

    # --- coarse_flow where the guess decides the result: the poc_strip pair (true shift (60.4, 1.1)) with max_disp = 16 has
    # one level and a 64 px window, so a shift of 60 px is out of reach without a guess (POC finds |d| < 32). The sloped
    # dense guess ux = 45 + 0.05 x, uy = -3 + 0.12 y is 14 to 8 px (x) and 4 to 2 px (y) off the truth. The poc_init
    # pair above converges to the same grid with or without its guess; this one differs by up to 85 px. The guess is
    # not stored (it would be two 96 x 512 arrays): the Rust test builds it from these parameters, in float64 rounded
    # to float32 as here.
    far = {"max_disp": 16, "guess_x": [45.0, 0.05], "guess_y": [-3.0, 0.12]}
    yy, xx = np.mgrid[0:96, 0:512].astype(float)
    far_guess = ((far["guess_x"][0] + far["guess_x"][1] * xx).astype(np.float32).astype(np.float64),
                 (far["guess_y"][0] + far["guess_y"][1] * yy).astype(np.float32).astype(np.float64))
    far_cfg = Config(max_disp=far["max_disp"])
    gfar = poc.coarse_flow(sa, sb, far_cfg, init_dense=far_guess)
    gnone = poc.coarse_flow(sa, sb, far_cfg)
    assert np.abs(gfar.ux - gnone.ux).max() > 50  # the guess changes the result
    # ... at every grid point by more than 10 px: the Rust test (parity_poc.rs) counts the points where its own
    # no-guess result is farther than 10 px from gfar
    assert np.hypot(gfar.ux - gnone.ux, gfar.uy - gnone.uy).min() > 10
    assert (gfar.peak > 0.9).sum() >= 40 and np.abs(gfar.ux[gfar.peak > 0.9] - 60.4).max() < 0.1
    save("poc_far_ux", gfar.ux); save("poc_far_uy", gfar.uy); save("poc_far_peak", gfar.peak)
    meta["poc_far"] = grid_dict(gfar) | far

    # --- refine (fine stage) from a near-exact initial field: order 1 and 2
    n = 128
    p = synth.make_pattern(n, n, seed=7)
    f = synth.uniform_strain(0.02, -0.01, (64, 64), t=(1.3, -0.8))
    a = synth.render(p, n, n); b = synth.render(p, n, n, f)
    gx, gy = synth.displacement(f, n, n)
    save("refine_a", a); save("refine_b", b); save("refine_init_ux", gx + 0.3); save("refine_init_uy", gy - 0.3)
    for order in (1, 2):
        fl = phaseflow.refine(a, b, gx + 0.3, gy - 0.3, Config(shape_order=order))
        save(f"refine_o{order}_ux", fl.ux); save(f"refine_o{order}_uy", fl.uy)
        save(f"refine_o{order}_conf", fl.conf); save(f"refine_o{order}_resid", fl.resid)

    # --- refine on a pure stripe texture (aperture problem): confidence stays below conf_min, no NaN / error
    xs_ = np.arange(64)[None, :] * np.ones((64, 1))
    stripe_a, stripe_b = 0.5 + 0.2 * np.sin(xs_ / 2), 0.5 + 0.2 * np.sin((xs_ - 1.0) / 2)
    save("refine_stripe_a", stripe_a); save("refine_stripe_b", stripe_b)
    # the degenerate pixels keep the input displacement: a non-zero constant start makes that visible
    stripe_init = (0.7, -0.4)
    meta["refine_stripe"] = {"init": list(stripe_init), "shift": 1.0}
    init_x, init_y = np.full(stripe_a.shape, stripe_init[0]), np.full(stripe_a.shape, stripe_init[1])
    for order in (1, 2):
        fl = phaseflow.refine(stripe_a, stripe_b, init_x, init_y, Config(shape_order=order))
        assert fl.conf.max() < Config().conf_min and np.isfinite(fl.ux).all() and np.isfinite(fl.uy).all()
        save(f"refine_stripe_o{order}_ux", fl.ux); save(f"refine_stripe_o{order}_uy", fl.uy)
        save(f"refine_stripe_o{order}_conf", fl.conf)

    # --- refine with the guard off at the image border (Task 8b): with edge_guard = 0 the border rule is off and only
    # the matches that leave image 2 get confidence 0, so the conf of the border columns and rows is pinned. A
    # translation of about (-1.5, 1.7) takes columns 0, 1 and the last two rows out of the image (the landing test has
    # margins of 0.2-0.3 px, far above the f32 / f64 differences), columns 2, 3 stay in. 64 x 64 keeps it small.
    n = 64
    p = synth.make_pattern(n, n, seed=13)
    bf = {"exx": 0.004, "eyy": -0.002, "center": [32.0, 32.0], "t": [-1.5, 1.7], "init_offset": [0.3, -0.3]}
    f = synth.uniform_strain(bf["exx"], bf["eyy"], tuple(bf["center"]), t=tuple(bf["t"]))
    a = synth.render(p, n, n); b = synth.render(p, n, n, f)
    gx, gy = synth.displacement(f, n, n)
    cfg0 = Config(edge_guard=0.0)
    fl = phaseflow.refine(a, b, gx + bf["init_offset"][0], gy + bf["init_offset"][1], cfg0)
    assert (fl.conf[:, :2] == 0).all() and (fl.conf[-2:, :] == 0).all()
    assert np.median(fl.conf[16:48, 2]) > 0.5 and np.median(fl.conf[n - 3, 16:48]) > 0.5
    save("refine_border_a", a); save("refine_border_b", b)
    save("refine_border_ux", fl.ux); save("refine_border_uy", fl.uy); save("refine_border_conf", fl.conf)
    meta["refine_border"] = bf

    # --- strain on a smooth field with NaN holes; EDT of a mask
    yy, xx = np.mgrid[0:60, 0:70].astype(float)
    sux = 0.01 * xx + 0.002 * yy + 1e-4 * xx * yy + 0.3 * np.sin(xx / 9.0)
    suy = -0.003 * xx + 1e-4 * yy**2 + 0.2 * np.cos(yy / 7.0)
    sux[20:26, 30:37] = np.nan; suy[20:26, 30:37] = np.nan
    save("strain_ux", sux); save("strain_uy", suy)
    grads = strain.displacement_gradient(sux, suy, 15)
    for k, gr in zip(("dudx", "dudy", "dvdx", "dvdy"), grads):
        save(f"strain_{k}", gr)
    sf = strain.strain_fields(*grads)
    for k in ("Exx", "Eyy", "Exy", "gmax", "vol"):
        save(f"strain_{k}", sf[k])
    valid = rng.uniform(size=(50, 60)) > 0.03
    valid[10:20, 25:30] = False
    save("edt_valid", valid.astype(np.float64)); save("edt_dist", distance_transform_edt(valid))

    # --- EDT feature transform (distance_transform_edt(..., return_indices=True)), as prewarp uses it (spec 2.3): four
    # random source sets of rising density, 26 rows x 30 columns, with many pixels that have several equidistant nearest
    # sources; idx is the row-major index iy * 30 + ix of the nearest source (ties: scipy's rule, see Rust edt.rs)
    ft_rng = np.random.default_rng(12345)
    ft_src = np.stack([ft_rng.uniform(size=(26, 30)) < p for p in (0.004, 0.02, 0.06, 0.2)])
    assert ft_src.reshape(4, -1).any(axis=1).all()
    ft_idx = np.empty(ft_src.shape, np.int64)
    for k, src in enumerate(ft_src):
        _, (iy, ix) = distance_transform_edt(~src, return_indices=True)
        ft_idx[k] = iy * src.shape[1] + ix
    save("edt_ft_src", ft_src.astype(np.float64)); save("edt_ft_idx", ft_idx)

    # --- track_pair with ROI and a crack (guard active), and a 4-frame sequence with forced reference updates
    n = 160
    p = synth.make_pattern(n, n, seed=21)
    a = synth.render(p, n, n); f = synth.translation(6.3, 1.7); b = synth.render(p, n, n, f)
    other = synth.render(synth.make_pattern(n, n, seed=77), n, n)
    b[:, 72:88] = other[:, 72:88]
    roi = np.zeros((n, n), bool); roi[10:150, 10:150] = True
    r = track.track_pair(a, b, Config(), mask=roi)
    save("pair_a", a); save("pair_b", b); save("pair_roi", roi.astype(np.float64))
    for k, v in (("ux", r.ux), ("uy", r.uy), ("conf", r.conf), ("Exx", r.strain["Exx"]), ("gmax", r.strain["gmax"])):
        save(f"pair_{k}", v)
    n = 128
    p = synth.make_pattern(n, n, seed=31)
    frames = [synth.render(p, n, n, synth.uniform_strain(0.003 * k, 0.0, (64, 64), t=(1.5 * k, 0.7 * k)))
              for k in range(4)]
    save("seq_frames", np.stack(frames))
    out = list(track.track_sequence(frames, Config(ref_update_conf=2.0)))
    save("seq_ux", np.stack([res.ux for _, res in out])); save("seq_uy", np.stack([res.uy for _, res in out]))
    meta["seq_reference"] = [res.timing["reference"] for _, res in out]

    # --- track_sequence link confidence (C1, spec 2.6): a foreign block that frames 1..3 show at one fixed position
    # (absent from frame 0) breaks the 0 -> 1 match of the frame-0 pixels that move into it; the reference moves on
    # every frame (ref_update_conf = 2), so the later legs match the block with itself and look confident, and only
    # the min over the legs keeps those pixels invalid. Frames 2 and 3 are the ones that show it, so only their
    # outputs are stored (size budget); the reference is computed from the float32 frames that are stored.
    n = 128
    p = synth.make_pattern(n, n, seed=21)
    c1_frames = [synth.render(p, n, n, synth.translation(2.0 * k, 1.0 * k)) for k in range(4)]
    other = synth.render(synth.make_pattern(n, n, seed=77), n, n)
    for k in (1, 2, 3):
        c1_frames[k][56:72, 56:72] = other[56:72, 56:72]
    c1_frames = [f.astype(np.float32).astype(np.float64) for f in c1_frames]
    save("seqc1_frames", np.stack(c1_frames))
    out = list(track.track_sequence(c1_frames, Config(ref_update_conf=2.0)))
    assert [res.timing["reference"] for _, res in out] == [0, 1, 2]
    out = out[1:]  # k = 2, 3
    save("seqc1_ux", np.stack([res.ux for _, res in out])); save("seqc1_uy", np.stack([res.uy for _, res in out]))
    save("seqc1_conf", np.stack([res.conf for _, res in out]))

    # --- mean_conf of the reference update (spec 2.6): sequences run with the default ref_update_conf (0.5) on the float32
    # frames stored above (nothing new is stored). "seq" and "seqc1" keep the reference at frame 0: the mean is taken over
    # the reportable pixels only (0.92-1.00 here; over all pixels it was 0.45-0.55 and on seqc1 fell under 0.5, moving the
    # reference at every frame). "mixed" is seq 0, seq 1, seqc1 0, seqc1 1: unrelated pairs of images in the middle, so the
    # mean of the 0 -> 2 leg is ~0 and the reference moves to frame 1, then to frame 2. timing["mean_conf"] is that of the
    # leg that produced the frame (after a reference move: of the re-tracked leg).
    stored = {k: list(np.load(os.path.join(OUT, f"{k}_frames.npy")).astype(np.float64)) for k in ("seq", "seqc1")}
    mean_conf_sets = {"seq": stored["seq"], "seqc1": stored["seqc1"],
                      "mixed": [stored["seq"][0], stored["seq"][1], stored["seqc1"][0], stored["seqc1"][1]]}
    meta["seq_mean_conf"] = {}
    for name, fr in mean_conf_sets.items():
        res = [r for _, r in track.track_sequence(fr, Config())]
        meta["seq_mean_conf"][name] = {"reference": [r.timing["reference"] for r in res],
                                       "mean_conf": [r.timing["mean_conf"] for r in res]}
    # The same two sequences with ref_update_conf = 2 (the forced-update runs of the cases above, here on the float32
    # frames that are stored): the reference moves on at every frame, so from frame 2 on the leg's mean is taken at
    # X + u0r(X) with a non-zero, fractional u0r. The legs match well, so these means can be pinned tightly (1e-6),
    # unlike the unrelated-image legs of "mixed", which are chaotic.
    meta["seq_mean_conf_forced"] = {}
    for name in ("seq", "seqc1"):
        res = [r for _, r in track.track_sequence(stored[name], Config(ref_update_conf=2.0))]
        assert [r.timing["reference"] for r in res] == [0, 1, 2]
        meta["seq_mean_conf_forced"][name] = {"reference": [r.timing["reference"] for r in res],
                                              "mean_conf": [r.timing["mean_conf"] for r in res]}

    # --- seq_mode = "hybrid" (v3): a translation with a foreign block that frames 2..4 show at one fixed position. The
    # reference moves (ref_update_conf 0.8: frames 3, 5, 6), pixels are carried by the adjacent leg, the spatial gate
    # rejects some, and the direct leg re-acquires others when the block is gone (frames 4 and 7). "hybA" is the default
    # hybrid configuration; "hybB" has fb_max = 2, so the carried values expire and the recovery without a fallback value
    # (gated like a fallback candidate) is used. Frames, ux, conf and fb of A and ux, fb of B are stored.
    n = 112
    p = synth.make_pattern(n, n, seed=21)
    other = synth.render(synth.make_pattern(n, n, seed=77), n, n)
    hy_frames = [synth.render(p, n, n, synth.translation(1.5 * k, 0.7 * k)) for k in range(8)]
    for k in (2, 3, 4):
        hy_frames[k][44:72, 44:72] = other[44:72, 44:72]
    hy_frames = [f.astype(np.float32).astype(np.float64) for f in hy_frames]
    save("hyb_frames", np.stack(hy_frames))
    meta["hybrid"] = {}
    for name, kw in (("A", {}), ("B", {"fb_max": 2})):
        out = list(track.track_sequence(hy_frames, Config(seq_mode="hybrid", ref_update_conf=0.8, **kw)))
        save(f"hyb{name}_ux", np.stack([res.ux for _, res in out]))
        save(f"hyb{name}_fb", np.stack([res.fb for _, res in out]))
        if name == "A":
            save("hybA_conf", np.stack([res.conf for _, res in out]))
        keys = ("reference", "mean_conf", "n_fb_max", "fallback_frac", "fb_gate_rejected", "reacquired")
        meta["hybrid"][name] = [{k: res.timing[k] for k in keys} for _, res in out]

    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=float)
    with open(os.path.join(OUT, "config_cases.json"), "w") as fh:
        json.dump(config_cases(), fh, indent=1)
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"wrote {len(os.listdir(OUT))} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

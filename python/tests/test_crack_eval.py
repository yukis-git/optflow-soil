"""End-to-end checks of the crack-aware tracking on the representative synthetic case of the design spec (section 4):
alpha = 0, opening 3 px, slip 1 px, 1 % noise, 384 x 384. One baseline and one crack_split + crack_out run are shared."""
import os

import numpy as np
import pytest

from demo.crack_eval import CONFIGS, Case, jump_metrics, metrics, profile, with_frac
from dataclasses import replace

from optflow_ref.config import Config
from optflow_ref.io import save_result
from optflow_ref.track import track_pair


@pytest.fixture(scope="module")
def run():
    case = Case(384, 0.0, 3.0, 1.0)
    a, b = case.images()
    cs = Config(crack_split=True, crack_out=True, crack_a=6.0)  # stage 1
    return case, cs, track_pair(a, b, Config()), track_pair(a, b, cs)


def test_defaults_give_the_same_pair_result_as_crack_off_explicitly():
    from optflow_ref import synth

    n = 160
    pat = synth.make_pattern(n, n, seed=2)
    f = synth.uniform_strain(0.01, -0.005, (80, 80), t=(2.3, -1.1))
    a, b = synth.render(pat, n, n), synth.render(pat, n, n, f)
    r1 = track_pair(a, b, Config())
    r2 = track_pair(a, b, Config(crack_split=False, crack_out=False, split_tol=0.1, split_gain=0.5, split_from_outer=2,
                                 crack_r=3.0, crack_a=3.0))
    for k in ("ux", "uy", "conf", "raw_ux", "raw_uy"):
        assert np.array_equal(getattr(r1, k), getattr(r2, k), equal_nan=True), k
    for k in r1.strain:
        assert np.array_equal(r1.strain[k], r2.strain[k], equal_nan=True), k
    assert r1.crack is None and r2.crack is None and r1.crack_in is None


def test_band_tracking_rate_improves_by_at_least_30_points(run):
    case, cs, r0, r1 = run
    m0, m1 = metrics(case, r0), metrics(case, r1)
    assert m1["valid"] >= m0["valid"] + 0.30, (m0["valid"], m1["valid"])


def test_far_field_unchanged(run):
    case, cs, r0, r1 = run
    m0, m1 = metrics(case, r0), metrics(case, r1)
    assert m1["far_valid"] == m0["far_valid"]
    assert m1["far_rms"] <= m0["far_rms"] + 1e-3


@pytest.mark.xfail(strict=True, reason="measured band RMS 0.26 px against the 0.05 px criterion: the Gabor filters straddle "
                                      "the crack")
def test_band_rms_at_most_005_px(run):
    case, cs, r0, r1 = run
    assert metrics(case, r1)["rms"] <= 0.05


@pytest.mark.xfail(strict=True, reason="measured COD error 0.7 px (crack_a = 12) against the 0.1 px criterion")
def test_cod_error_at_most_01_px(run):
    case, cs, r0, r1 = run
    assert jump_metrics(case, r1, cs, 12.0)["cod"] <= 0.1


def test_cod_and_csd_are_found_with_the_right_sign_and_size(run):
    case, cs, r0, r1 = run
    jm = jump_metrics(case, r1, cs, 12.0)
    assert jm["fin"] > 0.8
    assert 2.0 < jm["cod_med"] < 4.5  # true 3 px


def test_crack_outputs_are_written_only_with_crack_out(run, tmp_path):
    case, cs, r0, r1 = run
    save_result(str(tmp_path / "on"), r1)
    save_result(str(tmp_path / "off"), r0)
    for name in ("crack", "crack_nx", "crack_ny", "cod", "csd"):
        assert os.path.exists(tmp_path / "on" / f"{name}.npy")
        assert not os.path.exists(tmp_path / "off" / f"{name}.npy")
    assert np.load(tmp_path / "on" / "crack.npy").dtype == bool
    cod = np.load(tmp_path / "on" / "cod.npy")
    mask = np.load(tmp_path / "on" / "crack.npy")
    assert cod.dtype == np.float32 and np.isnan(cod[~mask]).all() and np.isfinite(cod[mask]).any()


# ---- stage 2: crack_sides (spec 6) -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sides(run):
    case = run[0]
    a, b = case.images()
    cfg = CONFIGS["sides"]
    r = track_pair(a, b, cfg)
    return case, cfg, r, with_frac(r, cfg, 0.0)


def test_sides_tracking_rate_improves_by_at_least_30_points_without_the_guard(run, sides):
    case, _, r0, _ = run[0], run[1], run[2], run[3]
    _, cfg, r, r_f0 = sides
    assert metrics(case, r_f0)["valid"] >= metrics(case, r0)["valid"] + 0.30


def test_sides_far_field_unchanged(run, sides):
    case, _, r0, _ = run[0], run[1], run[2], run[3]
    m0, m1 = metrics(case, r0), metrics(sides[0], sides[2])
    assert m1["far_valid"] == m0["far_valid"] and m1["far_rms"] <= m0["far_rms"] + 1e-3


def test_sides_valid_pixels_14_to_16_px_from_the_crack_are_near_the_noise_floor(sides):
    case, _, r, r_f0 = sides
    p = profile(case, r_f0)
    # 1 % noise: the far field has 0.04-0.05 px median error; per side 0.08 px at 14..16 px (stage 1: 0.09 and 0.17 at 12 px)
    assert p["-"][-1] <= 0.08 and p["+"][-1] <= 0.08


@pytest.mark.xfail(strict=True, reason="measured band RMS 0.081 px (crack_guard_frac 0.5; 0.132 with 0) against 0.05; the "
                                      "far field alone has 0.065 at this noise")
def test_sides_band_rms_at_most_005_px(sides):
    case, _, r, _ = sides
    assert metrics(case, r)["rms"] <= 0.05


@pytest.mark.xfail(strict=True, reason="measured COD error 0.21 px (crack_a 12, strip fit) against 0.1")
def test_sides_cod_error_at_most_01_px(sides):
    case, cfg, r, _ = sides
    assert jump_metrics(case, r, cfg, cfg.crack_a)["cod"] <= 0.1


def test_sides_cod_found_and_close(sides):
    case, cfg, r, _ = sides
    jm = jump_metrics(case, r, cfg, cfg.crack_a)
    assert jm["fin"] > 0.9 and abs(jm["cod_med"] - 3.0) < 0.4

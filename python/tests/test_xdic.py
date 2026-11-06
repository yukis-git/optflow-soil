"""crack_model "xdic" (option B, optflow_ref/xdic.py): config, geometry, detector, per-side solve, sequences."""
import json

import numpy as np
import pytest

from optflow_ref import phaseflow as pf
from optflow_ref import synth, xdic
from optflow_ref.config import Config
from optflow_ref.interp import warp
from optflow_ref.track import track_pair, track_sequence


# ------------------------------------------------------------------------------------------------------------- config
@pytest.mark.parametrize("kw", [dict(crack_model="on"), dict(crack_model=1), dict(xdic_RB=0.0), dict(xdic_RB=float("nan")),
                                dict(xdic_RB=300.0), dict(xdic_iters=-1), dict(xdic_iters=9), dict(xdic_iters=1.5),
                                dict(xdic_iters=True), dict(crack_model="xdic", crack_split=True),
                                dict(crack_model="xdic", crack_split=True, crack_sides=True)])
def test_xdic_keys_validated(kw):
    with pytest.raises(ValueError):
        Config(**kw).validate()


def test_xdic_keys_left_out_of_the_json_at_default_and_round_trip():
    d = json.loads(Config().to_json())
    assert not {"crack_model", "xdic_RB", "xdic_iters"} & set(d)
    c = Config(crack_model="xdic", xdic_iters=3)
    d2 = json.loads(c.to_json())
    assert d2["crack_model"] == "xdic" and d2["xdic_iters"] == 3 and "xdic_RB" not in d2
    assert Config.from_json(c.to_json()) == c
    assert Config().xdic_rb_px() == pytest.approx(Config().auto_edge_guard_px() + 16.0)
    assert Config(xdic_RB=20.0).xdic_rb_px() == 20.0


# ------------------------------------------------------------------------------------------------------------- geometry
def test_polyline_fields_signed_distance_interior_and_normal():
    P = np.array([[10.0, 5.0], [10.0, 25.0], [30.0, 45.0]])  # down, then diagonal
    f = xdic.polyline_fields([P], (50, 40))
    yy, xx = np.mgrid[0:50, 0:40].astype(float)
    # left normal of (0, 1) is (-1, 0): phi > 0 for x < 10 along the first segment
    assert f["phi"][15, 4] == pytest.approx(6.0) and f["phi"][15, 13] == pytest.approx(-3.0)
    assert f["nx"][15, 4] == pytest.approx(-1.0) and f["ny"][15, 4] == pytest.approx(0.0)
    assert f["interior"][15, 4] and not f["interior"][2, 10]  # beyond the start
    assert f["lid"][15, 4] == 0 and f["tau"][15, 4] == pytest.approx(10.0)
    # second segment: unsigned distance to the line through it
    d = np.abs((xx - 10) * (-20) + (yy - 25) * 20) / np.hypot(20, 20)
    m = f["interior"] & (yy > 32) & (yy < 40) & (np.abs(xx - (yy - 15)) < 4)
    assert np.allclose(np.abs(f["phi"][m]), d[m])


def test_advance_and_frame0_mapping_of_lines():
    H = W = 60
    P = np.array([[30.0, 0.0], [30.0, 59.0]])
    ux = np.where(np.arange(W)[None, :] < 30, 1.0, 3.0) * np.ones((H, W))  # opening 2 px across x = 30
    uy = np.full((H, W), -1.0)
    nxt = xdic.advance_lines([P], ux, uy)[0]
    assert np.allclose(nxt[:, 0], 32.0) and np.allclose(nxt[:, 1], P[:, 1] - 1.0)  # mean motion of the two faces
    back = xdic.lines_to_frame0([nxt], (ux, uy))[0]
    assert np.allclose(back, P, atol=1e-6)
    assert np.array_equal(xdic.lines_to_frame0([P], (np.zeros((H, W)), np.zeros((H, W))))[0], P)


# ------------------------------------------------------------------------------------------------------------- detector
def _step_field(N=160, j=(2.0, 0.6), hole=12, noise=0.03, seed=0):
    """A displacement field with a jump across x = 80.3, invalid within `hole` px of it, plus white noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    s = xx - 80.3
    ux = 0.5 + 0.002 * yy + np.where(s >= 0, j[0], 0.0) + noise * rng.standard_normal((N, N))
    uy = -0.3 + np.where(s >= 0, j[1], 0.0) + noise * rng.standard_normal((N, N))
    bad = (np.abs(s) < hole) | (np.minimum.reduce([xx, yy, N - 1 - xx, N - 1 - yy]) < 28)
    return np.where(bad, np.nan, ux), np.where(bad, np.nan, uy)


def test_detector_finds_the_jump_line_and_not_a_smooth_field():
    cfg = Config(crack_model="xdic")
    ux, uy = _step_field()
    lines, info = xdic.detect_lines(ux, uy, None, cfg)
    assert len(lines) == 1 and info["n_acc"] > 10
    P = lines[0]
    inside = (P[:, 1] > 30) & (P[:, 1] < 130)
    assert np.abs(P[inside, 0] - 80.3).max() < 2.0
    assert P[:, 1].min() < 0 and P[:, 1].max() > 159  # extended to the border
    J, z, _ = xdic.jump_test(ux, uy, np.isfinite(ux), np.array([80]), np.array([80]), kappa=4 * np.pi * 16)
    assert J[0, 0] == pytest.approx(2.0, abs=0.05) and J[0, 1] == pytest.approx(0.6, abs=0.05) and z[0] > 6
    ux0, uy0 = _step_field(j=(0.0, 0.0))
    assert xdic.detect_lines(ux0, uy0, None, cfg)[0] == []


# ------------------------------------------------------------------------------------------------------------- solve
def test_masked_reference_removes_the_anchor_of_the_paint_edge():
    """One per-side step about a pre-warp that is 0.3 px off: the plain phase gradient keeps most of the offset at the
    painted line (the paint edge does not move), the masked-derivative Jacobian recovers it."""
    N = 128
    pat = synth.make_pattern(N, N, seed=3)
    a = synth.degrade(synth.render(pat, N, N), seed=1, noise=0.01)
    b = synth.degrade(synth.render(pat, N, N), seed=2, noise=0.01)  # zero motion
    cfg = Config()
    bank, fx, fy = pf.gabor_bank((N + 2 * pf.PAD, N + 2 * pf.PAD), cfg.lam, cfg.scales, cfg.n_orient, cfg.bandwidth, cfg.dc_cut)
    xx = np.mgrid[0:N, 0:N][1].astype(float)
    fill = xx >= 64 - xdic.MIX_PX
    m1 = a.mean()
    sx, sy = np.full((N, N), 0.3), np.zeros((N, N))
    J2 = np.where(fill, m1, warp(b, sx, sy))
    resp = [pf._crop(np.fft.ifft2(np.fft.fft2(pf._pad(J2)) * G)) for G in bank]
    err = {}
    for name, ref in (("masked", xdic.masked_reference(a, fill, m1, bank, fx, fy)),
                      ("plain", pf.reference_phase(np.where(fill, m1, a), bank, fx, fy))):
        sk = pf._solve_full(*xdic._moments_grid(ref, resp, sx, sy, (~fill).astype(float)), sx, sy, cfg.sigma_w, 1)
        err[name] = float(np.median(np.abs(sk[0][30:98, 59])))  # 3 px from the line
    assert err["masked"] < 0.1 < err["plain"]


@pytest.fixture(scope="module")
def crack_pair():
    N = 192
    c = (N / 2.0, N / 2.0)
    f = synth.Crack(c, 0.0, 3.0, 1.0)
    pat = synth.make_pattern(N, N, seed=21)
    a = synth.degrade(synth.render(pat, N, N), seed=1, noise=0.01)
    b = synth.degrade(synth.render(pat, N, N, f), seed=2, noise=0.01)
    r = track_pair(a, b, Config(crack_model="xdic"))
    return f, a, b, r


def test_xdic_pair_tracks_up_to_the_line_and_reads_the_jump(crack_pair):
    f, a, b, r = crack_pair
    N = a.shape[0]
    gx, gy = synth.displacement(f, N, N)
    yy, xx = np.mgrid[0:N, 0:N]
    d = np.abs(xx - N / 2)
    mid = (yy > 50) & (yy < N - 50)
    e = np.hypot(r.ux - gx, r.uy - gy)
    near = mid & (d >= 2) & (d <= 6)
    assert np.isfinite(e[near]).mean() > 0.9 and np.nanmedian(e[near]) < 0.2
    assert len(r.crack_lines["lines"]) == 1
    P = r.crack_lines["lines"][0]
    assert np.abs(P[(P[:, 1] > 50) & (P[:, 1] < N - 50), 0] - N / 2).max() < 1.5
    j = r.crack_lines["jumps"][0]
    m = (j["pts"][:, 1] > 50) & (j["pts"][:, 1] < N - 50)
    assert np.nanmedian(j["cod"][m]) == pytest.approx(3.0, abs=0.15)
    assert np.nanmedian(j["csd"][m]) == pytest.approx(1.0, abs=0.15)
    assert r.crack["mask"].any() and set(r.crack) >= {"mask", "nx", "ny", "cod", "csd"}
    base = track_pair(a, b, Config())
    far = mid & (d > 40) & np.isfinite(base.ux)
    assert np.nanmax(np.abs(r.ux[far] - base.ux[far])) < 2e-2  # the far field is the ordinary solve


def test_shear_band_is_reported_as_band_not_crack():
    N = 192
    pat = synth.make_pattern(N, N, seed=21)
    f = synth.ShearBand(3.0, 6.0, (N / 2, N / 2), 90.0)
    a = synth.degrade(synth.render(pat, N, N), seed=1, noise=0.01)
    b = synth.degrade(synth.render(pat, N, N, f), seed=2, noise=0.01)
    r = track_pair(a, b, Config(crack_model="xdic"))
    base = track_pair(a, b, Config())
    assert r.crack is None and (r.crack_lines is None or r.crack_lines["lines"] == [])
    assert np.array_equal(np.isfinite(r.ux), np.isfinite(base.ux)) and np.allclose(r.ux, base.ux, equal_nan=True)


def test_default_track_pair_unchanged_by_the_xdic_keys():
    N = 96
    pat = synth.make_pattern(N, N, seed=4)
    a = synth.render(pat, N, N)
    b = synth.render(pat, N, N, synth.translation(0.7, -0.4))
    r0 = track_pair(a, b, Config())
    r1 = track_pair(a, b, Config(crack_model="off", xdic_RB=12.0, xdic_iters=5))
    assert np.array_equal(r0.ux, r1.ux, equal_nan=True) and np.array_equal(r0.conf, r1.conf)
    assert r0.crack_lines is None and r1.crack_lines is None


def test_xdic_sequence_carries_the_line_in_frame0_coordinates():
    N = 160
    pat = synth.make_pattern(N, N, seed=21)
    c = (N / 2.0, N / 2.0)
    fields = [synth.Crack(c, 0.0, 2.0 * k, 0.0) for k in range(3)]
    frames = [synth.degrade(synth.render(pat, N, N, f), seed=100 + k, noise=0.01) for k, f in enumerate(fields)]
    for mode in ("chain", "hybrid"):
        out = list(track_sequence(frames, Config(crack_model="xdic", seq_mode=mode)))
        k, r = out[-1]
        assert k == 2 and r.crack_lines is not None and len(r.crack_lines["lines"]) == 1
        P = r.crack_lines["lines"][0]
        assert np.abs(P[(P[:, 1] > 40) & (P[:, 1] < N - 40), 0] - N / 2).max() < 1.5
        cod = np.concatenate([j["cod"] for j in r.crack_lines["jumps"]])
        assert np.nanmedian(cod) == pytest.approx(4.0, abs=0.2)


# ------------------------------------------------------------------------------------- large openings, reference with a gap
@pytest.mark.parametrize("o", [16.0, 30.0])
@pytest.mark.parametrize("off", [0.0, 0.25, -0.25])
def test_frame0_mapping_of_a_line_in_a_wide_gap(o, off):
    """u0r jumps by o at the crack of frame 0 (x = 128). The old fixed point X0 = P - mean u0r(X0 +- 6 n) was a 2-cycle once
    o > 12 px and ended half a gap off, even with P in the middle of the gap; any P inside the gap maps to the crack."""
    H = W = 256
    xx = np.mgrid[0:H, 0:W][1].astype(float)
    ux = np.where(xx >= 128, o, 0.0)
    uy = np.full((H, W), -3.0)
    P = np.array([[128 + o / 2 + off * o, 20.0], [128 + o / 2 + off * o, 230.0]])
    X = xdic.lines_to_frame0([P], (ux, uy))[0]
    assert np.abs(X[:, 0] - 128).max() <= 1.0
    assert np.allclose(X[:, 1], P[:, 1] + 3.0)


def _gap_pair(N, o_r, d, seed=21):
    """Two frames of a straight crack (x = N / 2 on the - side) already open by o_r in the first one, opening by d more."""
    pat = synth.make_pattern(N, N, seed=seed)
    c = (N / 2.0, N / 2.0)
    a = synth.degrade(synth.render(pat, N, N, synth.Crack(c, 0.0, o_r, 0.0)), seed=1, noise=0.01)
    b = synth.degrade(synth.render(pat, N, N, synth.Crack(c, 0.0, o_r + d, 0.0)), seed=2, noise=0.01)
    return a, b


@pytest.mark.parametrize("hint_off", [0.0, 5.0])
def test_geometry_update_is_not_moved_by_the_noise_inside_a_gap(hint_off):
    """Gap of 20 px in the reference: inside the +-8 px band both sides explain every pixel, so the old update moved the line
    by the noise (+-7 px after the first update from the exact middle). With hint_off = 5 the + face enters the band: the
    line may only move towards the middle."""
    N, o_r = 192, 20.0
    a, b = _gap_pair(N, o_r, 2.0)
    mid = N / 2.0 + o_r / 2
    P = np.array([[mid + hint_off, -2.0], [mid + hint_off, N + 2.0]])
    r = track_pair(a, b, Config(crack_model="xdic"), crack_hint=[P])
    Q = r.crack_lines["lines"][0]
    q = Q[(Q[:, 1] > 40) & (Q[:, 1] < N - 40), 0] - mid
    assert q.min() >= -1.0 and q.max() <= hint_off + 1.0
    assert np.nanmedian(r.crack_lines["jumps"][0]["cod"]) == pytest.approx(2.0, abs=0.2)


def test_hybrid_sequence_keeps_line_and_cod_past_a_wide_opening_across_reference_moves():
    """Opening 0 -> 24 px, the reference moves every frame (ref_update_conf 0.99): before the fixes the frame-0 line was half
    a gap off from a gap of 14 px on and the COD collapsed (3.4 px at a true 20 px)."""
    N = 160
    pat = synth.make_pattern(N, N, seed=21)
    c = (N / 2.0, N / 2.0)
    fields = [synth.Crack(c, 0.0, 4.0 * k, 0.0) for k in range(7)]
    frames = [synth.degrade(synth.render(pat, N, N, f), seed=100 + k, noise=0.01) for k, f in enumerate(fields)]
    out = list(track_sequence(frames, Config(crack_model="xdic", seq_mode="hybrid", ref_update_conf=0.99)))
    assert out[-1][1].timing["reference"] == 5
    for k, r in out[2:]:
        P = r.crack_lines["lines"][0]
        assert np.abs(P[(P[:, 1] > 40) & (P[:, 1] < N - 40), 0] - N / 2).max() < 2.5, k
        cod = np.concatenate([j["cod"] for j in r.crack_lines["jumps"]])
        assert np.nanmedian(cod) == pytest.approx(4.0 * k, abs=0.4), k


# --------------------------------------------------------------------------------------------- crack or band: model comparison
def _pair(field, N=192, seed=21):
    pat = synth.make_pattern(N, N, seed=seed)
    return (synth.degrade(synth.render(pat, N, N), seed=1, noise=0.01),
            synth.degrade(synth.render(pat, N, N, field), seed=2, noise=0.01))


def test_sharp_continuous_band_is_a_band_though_the_baseline_residual_is_high():
    """delta 6 px over tanh width 3 px: the baseline phase residual at the line is 2.9 x typical, which the former test (ratio
    > 2) took for a crack and replaced the field by a two-sided one; the per-side solve explains the images no better."""
    N = 192
    a, b = _pair(synth.ShearBand(6.0, 3.0, (N / 2, N / 2), 90.0), N)
    r = track_pair(a, b, Config(crack_model="xdic"))
    base = track_pair(a, b, Config())
    assert r.crack is None and r.crack_lines["lines"] == [] and len(r.crack_lines["bands"]) == 1
    assert np.array_equal(r.ux, base.ux, equal_nan=True) and np.array_equal(r.uy, base.uy, equal_nan=True)


def test_opening_of_a_gap_already_in_the_reference_is_a_crack():
    """A gap of 8 px in the reference opening by 3 px: the baseline tracks the dark band (confidence 0.85, residual 2 x typical),
    which the former test called a band; the per-side solve removes most of the excess intensity residual."""
    N = 192
    a, b = _gap_pair(N, 8.0, 3.0)
    r = track_pair(a, b, Config(crack_model="xdic"))
    assert len(r.crack_lines["lines"]) == 1
    P = r.crack_lines["lines"][0]
    assert np.abs(P[(P[:, 1] > 40) & (P[:, 1] < N - 40), 0] - (N / 2 + 4.0)).max() < 2.0
    assert np.nanmedian(r.crack_lines["jumps"][0]["cod"]) == pytest.approx(3.0, abs=0.25)


def test_is_crack_rule():
    assert xdic.is_crack(dict(gain=6.0, e0=6.1, e1=1.9, mean=1.2, n=500))
    assert xdic.is_crack(dict(gain=1.6, e0=13.0, e1=8.8, mean=4.0, n=500))  # the gap's gross mismatch
    assert not xdic.is_crack(dict(gain=1.2, e0=3.0, e1=2.6, mean=1.1, n=500))  # no better than the baseline
    assert not xdic.is_crack(dict(gain=5.0, e0=1.4, e1=1.08, mean=1.7, n=500))  # the baseline stands within the noise
    assert not xdic.is_crack(dict(gain=0.0, e0=0.0, e1=0.0, mean=0.0, n=3))  # too few pixels


def _frame0_step_field(max_u):
    """A crack at x = 80 opened by 8 px, a hole of the field in the last row around the crack (so the end vertex of a line
    beyond the image has no finite sample within the search), one far pixel that sets max |u0r|."""
    H = W = 160
    xx = np.mgrid[0:H, 0:W][1].astype(float)
    ux = np.where(xx >= 80, 8.0, 0.0)
    uy = np.zeros((H, W))
    ux[H - 1, 65:105] = np.nan
    uy[H - 1, 65:105] = np.nan
    ux[20, 150] = max_u
    return ux, uy


def test_frame0_mapping_does_not_depend_on_the_length_of_the_search():
    # max |u0r| on either side of an integer changes the search line by 1 px; regression: the end vertex jumped 0.4 px
    P = np.column_stack([np.full(11, 84.1), np.linspace(-2.0, 161.0, 11)])
    a = xdic.lines_to_frame0([P], _frame0_step_field(10.9995))[0]
    b = xdic.lines_to_frame0([P], _frame0_step_field(11.0005))[0]
    assert np.abs(a - b).max() < 1e-9
    assert np.abs(a[:, 0] - 80.0).max() < 1.0  # also the vertex without a sample: it moves with its neighbour


def test_frame0_mapping_is_continuous_in_the_line():
    # regression: leaving the 1 px cell switched between P - u and the zero of g, a jump of up to 1 px
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ux = 0.01 * xx + 0.005 * yy + np.where(xx >= 80, 6.0, 0.0)
    uy = 0.002 * xx
    P = np.column_stack([np.linspace(83.0, 84.0, 9), np.linspace(10.0, 150.0, 9)])
    n = np.array([-1.0, 0.0])
    X = np.stack([xdic.lines_to_frame0([P + t * n], (ux, uy))[0] for t in np.arange(-5.0, 5.0, 0.01)])
    assert np.hypot(*np.moveaxis(np.diff(X, axis=0), 2, 0)).max() < 0.06

"""seq_mode = "hybrid" (direct-first): direct value wherever the direct link is confident, bounded adjacent fallback
elsewhere, recovery, and the unchanged chain default."""
import hashlib

import numpy as np
import pytest

from optflow_ref import synth
from optflow_ref import track as T
from optflow_ref.config import Config
from optflow_ref.track import Result, track_sequence


def test_config_hybrid_keys_roundtrip_and_validation():
    c = Config(seq_mode="hybrid", fb_max=4, fb_radius=12.0, fb_min_support=5, fb_tol=0.3, tavg=3, tavg_align=True, tavg_model="dense")
    assert Config.from_json(c.to_json()) == c
    d = Config()
    assert (d.seq_mode, d.fb_max) == ("chain", 10)
    assert (d.fb_radius, d.fb_min_support, d.fb_tol, d.tavg, d.tavg_align, d.tavg_model) == (-1.0, 20, 0.5, 0, False, "affine")
    assert d.fb_radius_px() == 16.0 and c.fb_radius_px() == 12.0
    for bad in ({"seq_mode": "x"}, {"fb_max": -1}, {"fb_tol": 0.0}, {"fb_radius": 0.0}, {"fb_min_support": -1},
                {"tavg": -1}, {"tavg": 65}, {"tavg_model": "x"}):
        with pytest.raises(ValueError):
            Config(**bad).validate()
    for text in ('{"tavg_align": 1}', '{"tavg": 1.5}', '{"seq_mode": 1}', '{"seq_mode": "other"}', '{"fb_max": 1.5}', '{"sigma_ref": 0.05}'):
        with pytest.raises(ValueError):
            Config.from_json(text)


H = W = 96
PATCH = (slice(30, 50), slice(30, 50))
INNER = (slice(5, -5), slice(5, -5))
CFG = dict(edge_guard=0.0, strain_win=5)


def _frames(n):
    return [np.full((H, W), float(k)) for k in range(n)]


def _scripted(monkeypatch, conf_of, disp_of=None):
    """Replace track_pair: frames are constants k; the pair (i, j) has displacement disp_of(i, j) (default j - i,
    i.e. the texture moves 1 px in x per frame) and confidence conf_of(i, j) (scalar or HxW array)."""
    calls = []

    def fake(I1, I2, cfg, mask=None, init=None):
        i, j = int(I1[0, 0]), int(I2[0, 0])
        calls.append((i, j))
        ux = np.broadcast_to(np.asarray(j - i if disp_of is None else disp_of(i, j), dtype=np.float64), (H, W)).copy()
        c = np.broadcast_to(np.asarray(conf_of(i, j), dtype=np.float64), (H, W)).copy()
        return Result(ux, 0 * ux, c, {}, ux, 0 * ux, {"coarse": 0.0, "fine": 0.0})

    monkeypatch.setattr(T, "track_pair", fake)
    return calls


def _patch_conf(low, bad_leg):
    """0.9 everywhere except `low` on the patch for the legs bad_leg(i, j) selects."""
    def conf_of(i, j):
        c = np.full((H, W), 0.9)
        if bad_leg(i, j):
            c[PATCH] = low
        return c
    return conf_of


def test_direct_good_pixels_equal_chain_exactly(monkeypatch):
    conf = _patch_conf(0.0, lambda i, j: i == 0 and j >= 2)
    _scripted(monkeypatch, conf)
    frames = _frames(5)
    kw = dict(CFG, ref_update_conf=0.0)  # the reference stays at frame 0 in both modes
    chain = dict(track_sequence(frames, Config(**kw)))
    hyb = dict(track_sequence(frames, Config(seq_mode="hybrid", **kw)))
    for k in range(1, 5):
        good = hyb[k].fb == 0
        assert good.sum() == H * W - (400 if k >= 2 else 0)
        assert np.array_equal(hyb[k].raw_ux[good], chain[k].raw_ux[good])
        assert np.array_equal(hyb[k].raw_uy[good], chain[k].raw_uy[good])
        assert np.array_equal(hyb[k].conf[good], chain[k].conf[good])
        fin = np.isfinite(chain[k].ux)
        assert np.array_equal(hyb[k].ux[good & fin], chain[k].ux[good & fin], equal_nan=True)
        assert hyb[k].timing["reference"] == 0


def test_all_good_real_images_hybrid_is_bit_identical_to_chain():
    N = 160
    pat = synth.make_pattern(N, N, seed=3)
    frames = [synth.render(pat, N, N, synth.uniform_strain(0.002 * k, -0.001 * k, (N / 2, N / 2), t=(1.1 * k, 0.5 * k)))
              for k in range(4)]
    chain = dict(track_sequence(frames, Config()))
    hyb = dict(track_sequence(frames, Config(seq_mode="hybrid")))
    for k in chain:
        v = np.isfinite(chain[k].ux)
        assert v.mean() > 0.2 and (hyb[k].fb[v] == 0).all()  # border strips are not direct-good: they fall back and stay invalid
        assert np.array_equal(hyb[k].ux[v], chain[k].ux[v]) and np.array_equal(hyb[k].uy[v], chain[k].uy[v])  # hybrid may be valid on more pixels
        for a in ("conf", "raw_ux", "raw_uy"):
            assert np.array_equal(getattr(hyb[k], a)[v], getattr(chain[k], a)[v]), (k, a)


def test_static_overlay_cannot_pull_a_direct_good_pixel(monkeypatch):
    # on the patch the adjacent legs see a static overlay: displacement 0 at confidence 0.99; the direct legs see the truth
    _scripted(monkeypatch, lambda i, j: 0.99 if (j - i == 1 and i > 0) else 0.9, disp_of=_overlay_disp)
    out = dict(track_sequence(_frames(6), Config(seq_mode="hybrid", **CFG, ref_update_conf=0.0)))
    assert (out[1].raw_ux[PATCH] == 0).all()  # k = 1: the only pair is the adjacent one
    for k in range(2, 6):
        assert (out[k].fb == 0).all()
        assert np.allclose(out[k].raw_ux[INNER], k)  # the patch included: direct value, not the overlay's 0


def _overlay_disp(i, j):
    d = np.full((H, W), float(j - i))
    if j - i == 1:
        d[PATCH] = 0.0
    return d


def test_fallback_counter_expiry(monkeypatch):
    # the direct leg (0 -> j, j >= 2) is blind on the patch; adjacent legs are fine (0.9)
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j >= 2))
    out = dict(track_sequence(_frames(7), Config(seq_mode="hybrid", fb_max=3, **CFG, ref_update_conf=0.0)))
    assert out[1].fb[PATCH].max() == 0
    for k, n in ((2, 1), (3, 2), (4, 3)):
        assert (out[k].fb[PATCH] == n).all() and (out[k].fb[~_patch()] == 0).all()
        assert np.isfinite(out[k].ux[40, 40]) and out[k].ux[40, 40] == pytest.approx(k)  # carried: u_{k-1} + 1
        assert out[k].conf[40, 40] == pytest.approx(0.9 * (1 - n / 4))
        assert out[k].timing["fallback_frac"] == pytest.approx(400 / (H * W))
    for k in (5, 6):  # n_fb = 4 > fb_max: invalid, the counter stays capped, no adjacent leg is needed any more
        assert np.isnan(out[k].ux[40, 40]) and np.isfinite(out[k].ux[70, 70])
        assert (out[k].fb[PATCH] == 4).all() and out[k].conf[40, 40] == 0.0
        assert out[k].timing["fallback_frac"] == 0.0


def _patch():
    m = np.zeros((H, W), bool)
    m[PATCH] = True
    return m


def test_fallback_needs_confident_adjacent_leg(monkeypatch):
    # direct and adjacent legs are both blind on the patch from frame 2: no value is trusted
    _scripted(monkeypatch, _patch_conf(0.1, lambda i, j: j >= 2))
    out = dict(track_sequence(_frames(4), Config(seq_mode="hybrid", **CFG, ref_update_conf=0.0)))
    assert np.isnan(out[2].ux[40, 40]) and np.isnan(out[3].ux[40, 40]) and np.isfinite(out[3].ux[70, 70])
    assert (out[2].fb[PATCH] == 1).all() and (out[3].fb[PATCH] == 2).all()


def test_recovery_after_a_low_frame_where_chain_stays_invalid(monkeypatch):
    # every leg ending at frame 3 is blind on the patch
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: j == 3))
    frames = _frames(6)
    out = dict(track_sequence(frames, Config(seq_mode="hybrid", **CFG, ref_update_conf=0.0)))
    assert np.isnan(out[3].ux[40, 40])  # adjacent leg 2 -> 3 is blind too: nothing to carry it
    assert (out[3].fb[PATCH] == 1).all()
    for k in (4, 5):
        assert (out[k].fb == 0).all() and np.isfinite(out[k].ux[40, 40]) and out[k].ux[40, 40] == pytest.approx(k)
    chain = dict(track_sequence(frames, Config(**CFG, ref_update_conf=2.0)))  # min over the chain
    assert np.isnan(chain[4].ux[40, 40]) and np.isnan(chain[5].ux[40, 40])


def test_recovery_fallback_bridges_a_blind_direct_leg(monkeypatch):
    # only the direct leg 0 -> 3 is blind on the patch: carried at frame 3 by the adjacent leg, direct again at 4
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j == 3))
    out = dict(track_sequence(_frames(6), Config(seq_mode="hybrid", **CFG, ref_update_conf=0.0)))
    assert (out[3].fb[PATCH] == 1).all() and out[3].ux[40, 40] == pytest.approx(3.0)
    for k in (4, 5):
        assert (out[k].fb == 0).all() and out[k].ux[40, 40] == pytest.approx(k)


def test_pixel_in_fallback_when_the_reference_moves_is_not_an_anchor(monkeypatch):
    # direct leg from frame 0 blind on the patch at frame 3; the reference moves to 3 at frame 4 (mean conf test)
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j >= 3))
    cfg = Config(seq_mode="hybrid", fb_max=5, **CFG, ref_update_conf=0.0)
    gen = track_sequence(_frames(9), cfg)
    out = {}
    for k, r in gen:
        out[k] = r
        if k == 3:
            cfg.ref_update_conf = 2.0  # from the next frame the reference moves to k-1
    assert (out[3].fb[PATCH] == 1).all() and out[3].timing["reference"] == 0
    assert out[4].timing["reference"] == 3
    # c0K = 0 on the patch: it stays in fallback (carried by the adjacent leg) although the new direct leg is confident
    assert (out[4].fb[PATCH] == 2).all() and out[4].ux[40, 40] == pytest.approx(4.0)
    assert out[7].ux[40, 40] == pytest.approx(7.0) and (out[7].fb[PATCH] == 5).all()  # n_fb = fb_max: still valid
    assert np.isnan(out[8].ux[40, 40]) and (out[8].fb[PATCH] == 6).all()
    # outside the patch the reference move changes nothing
    assert (out[7].fb[5:-5, 5:-15][~_patch()[5:-5, 5:-15]] == 0).all() and out[7].ux[70, 70] == pytest.approx(7.0)


def test_chain_default_is_unchanged():
    """sha256 of 5-frame chain outputs recorded on 4f722cf (before the hybrid mode existed)."""
    N = 192
    pat = synth.make_pattern(N, N, seed=21)
    frames = [synth.render(pat, N, N, synth.uniform_strain(0.002 * k, -0.001 * k, (N / 2, N / 2), t=(1.3 * k, 0.6 * k)))
              for k in range(5)]
    h = hashlib.sha256()
    for cfg in (Config(), Config(ref_update_conf=2.0)):
        for _, r in track_sequence(frames, cfg):
            assert r.fb is None
            for a in (r.ux, r.uy, r.conf, r.raw_ux, r.raw_uy):
                h.update(np.ascontiguousarray(a).tobytes())
    assert h.hexdigest() == "ce9c9bdbb05e56098647aeb9b8dac8e605be4ef5845f33b53beb5aef2a83620b"


# ---- v3: spatial-consistency gate of the fallback, recovery across reference moves ----

def _overlay_adjacent_disp(i, j):
    """Adjacent legs from frame 2 on see a static overlay on the patch (displacement 0)."""
    d = np.full((H, W), float(j - i))
    if j - i == 1 and j >= 2:
        d[PATCH] = 0.0
    return d


def _overlay_conf(i, j):
    c = np.full((H, W), 0.9)
    if i == 0 and j >= 2:
        c[PATCH] = 0.0  # the direct leg is blind there
    elif j - i == 1:
        c[PATCH] = 0.99  # the overlay matches confidently at zero displacement
    return c


def test_gate_rejects_a_fallback_that_locks_on_a_static_overlay(monkeypatch):
    _scripted(monkeypatch, _overlay_conf, disp_of=_overlay_adjacent_disp)
    frames = _frames(5)
    v2 = dict(track_sequence(frames, Config(seq_mode="hybrid", fb_min_support=0, **CFG, ref_update_conf=0.0)))
    v3 = dict(track_sequence(frames, Config(seq_mode="hybrid", **CFG, ref_update_conf=0.0)))
    for k in (2, 3, 4):
        assert np.isfinite(v2[k].ux[40, 40]) and v2[k].ux[40, 40] == pytest.approx(1.0)  # v2: stuck at the overlay's zero motion (true k) and valid
        assert np.isnan(v3[k].ux[40, 40])  # v3: disagrees with the direct-good neighbours (displacement k), invalid
        # outside the patch nothing changes; the column x = 49 is carried by a leg sampled just outside the overlay (correct)
        assert np.isfinite(v3[k].ux[70, 70]) and np.isnan(v3[k].ux[40, 30:49]).all()
        assert (v3[k].fb[PATCH] == k - 1).all()  # the counter keeps counting


def test_gate_accepts_a_consistent_fallback_within_fb_tol(monkeypatch):
    # the adjacent legs are 0.3 px short on the patch: inside fb_tol = 0.5 valid, outside (fb_tol = 0.2) invalid
    def disp(i, j):
        d = np.full((H, W), float(j - i))
        if i > 0 and j - i == 1:
            d[PATCH] -= 0.3
        return d
    conf = _patch_conf(0.0, lambda i, j: i == 0 and j >= 2)
    _scripted(monkeypatch, conf, disp_of=disp)
    ok = dict(track_sequence(_frames(3), Config(seq_mode="hybrid", **CFG, ref_update_conf=0.0)))
    assert ok[2].ux[40, 40] == pytest.approx(1.7)  # u_1 + (1 - 0.3) = 1.7, model 2: |dx| = 0.3 <= 0.5
    tight = dict(track_sequence(_frames(3), Config(seq_mode="hybrid", fb_tol=0.2, **CFG, ref_update_conf=0.0)))
    assert np.isnan(tight[2].ux[40, 40])


def test_gate_needs_fb_min_support_direct_good_neighbours_within_the_radius(monkeypatch):
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j >= 2))
    out = dict(track_sequence(_frames(3), Config(seq_mode="hybrid", fb_radius=3.0, **CFG, ref_update_conf=0.0)))
    assert np.isnan(out[2].ux[40, 40])  # centre of the 20 x 20 patch: nothing direct-good within 3 px
    assert np.isfinite(out[2].ux[30, 40])  # first patch row: the 7 x 7 window holds 3 good rows = 21 >= 20 pixels
    assert np.isnan(out[2].ux[31, 40])  # one row deeper: 14 < 20
    assert out[2].fb[40, 40] == 1


def test_gate_off_equals_v2(monkeypatch):
    _scripted(monkeypatch, _overlay_conf, disp_of=_overlay_adjacent_disp)
    a = dict(track_sequence(_frames(4), Config(seq_mode="hybrid", fb_min_support=0, **CFG, ref_update_conf=0.0)))
    assert np.isfinite(a[3].ux[40, 40])  # no gate: the v2 rule alone (adjacent conf, counter)


INNER_R = (slice(5, -15), slice(5, -15))  # away from the borders the shifted frames leave


def _toggled(cfg, k_on, k_off):
    """track_sequence generator on scripted frames whose ref_update_conf is 2.0 (reference moves) only for the
    frame after k_on; the value is restored after k_off. Returns the dict of results."""
    def run(n):
        out = {}
        for k, r in track_sequence(_frames(n), cfg):
            out[k] = r
            if k == k_on:
                cfg.ref_update_conf = 2.0
            if k == k_off:
                cfg.ref_update_conf = 0.0
        return out
    return run


def test_recovery_across_a_reference_move_when_direct_agrees_with_the_fallback(monkeypatch):
    # direct leg from frame 0 blind on the patch for j >= 3; the reference moves to frame 3 at frame 4 and stays there
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j >= 3))
    cfg = Config(seq_mode="hybrid", fb_max=6, **CFG, ref_update_conf=0.0)
    out = _toggled(cfg, 3, 4)(9)
    assert (out[3].fb[PATCH] == 1).all() and out[4].timing["reference"] == 3
    # frame 4: reference == k-1, the direct leg is the adjacent leg: no independent evidence, still fallback
    assert (out[4].fb[PATCH] == 2).all() and out[4].ux[40, 40] == pytest.approx(4.0)
    # frame 5: the leg 3 -> 5 is confident and agrees with the carried value (5): direct-good again, counter reset
    for k in (5, 6, 7, 8):
        assert (out[k].fb[INNER_R] == 0).all() and out[k].ux[40, 40] == pytest.approx(k)
        assert out[k].conf[40, 40] == pytest.approx(0.9)


def test_no_recovery_when_the_direct_leg_disagrees_with_the_fallback(monkeypatch):
    # the leg 3 -> j is confident on the patch but 1 px off (a static overlay seen by the longer leg): the fallback value
    # (carried by the adjacent legs) is kept, the pixel is not re-acquired and expires at fb_max
    def disp(i, j):
        d = np.full((H, W), float(j - i))
        if i == 3 and j >= 5:
            d[PATCH] += 1.0
        return d
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j >= 3), disp_of=disp)
    cfg = Config(seq_mode="hybrid", fb_max=4, **CFG, ref_update_conf=0.0)
    out = _toggled(cfg, 3, 4)(9)
    for k, n in ((5, 3), (6, 4)):
        assert (out[k].fb[30:50, 30:44] == n).all() and out[k].ux[40, 40] == pytest.approx(k)  # fallback value, not the direct's
    # frame 7: n_fb would be 5 > fb_max, so no fallback value is left to disagree with; the confident direct leg is 1 px off
    # the local model of the direct-good neighbours (v3 gate on this branch too): not accepted, the pixel stays invalid
    assert (out[7].fb[30:50, 30:44] == 5).all() and np.isnan(out[7].ux[40, 40])
    # gate off (v2 rule + recovery): the confident direct value is accepted
    _scripted(monkeypatch, _patch_conf(0.0, lambda i, j: i == 0 and j >= 3), disp_of=disp)
    cfg = Config(seq_mode="hybrid", fb_max=4, fb_min_support=0, **CFG, ref_update_conf=0.0)
    off = _toggled(cfg, 3, 4)(9)
    assert (off[7].fb[30:50, 30:44] == 0).all() and off[7].ux[40, 40] == pytest.approx(8.0)


def test_recovery_when_no_fallback_value_exists(monkeypatch):
    # both the direct leg from frame 0 and the adjacent legs 3 -> 4, 4 -> 5 are blind on the patch; the leg 3 -> 5 is
    # confident: with no fallback value to disagree with, the direct estimate is accepted
    def conf(i, j):
        c = np.full((H, W), 0.9)
        if (i == 0 and j >= 3) or (i, j) in ((3, 4), (4, 5)):
            c[PATCH] = 0.0
        return c
    _scripted(monkeypatch, conf)
    cfg = Config(seq_mode="hybrid", fb_max=6, **CFG, ref_update_conf=0.0)
    out = _toggled(cfg, 3, 4)(8)
    assert np.isnan(out[4].ux[40, 40]) and (out[4].fb[PATCH] == 2).all()
    assert (out[5].fb[INNER_R] == 0).all() and out[5].ux[40, 40] == pytest.approx(5.0)

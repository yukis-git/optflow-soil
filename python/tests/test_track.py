import numpy as np
import pytest

from optflow_ref import synth
from optflow_ref.config import Config
from optflow_ref.metrics import disp_errors, eval_mask, strain_errors, true_strain
from optflow_ref.poc import coarse_flow, grid_sample
from optflow_ref.track import _finish, _mean_conf, compose, track_pair, track_sequence

N = 256


@pytest.fixture(scope="module")
def pat():
    return synth.make_pattern(N, N, seed=21)


def test_pair_large_motion_and_strain(pat):
    f = synth.uniform_strain(0.02, -0.01, (N / 2, N / 2), t=(18.3, -9.6))
    a = synth.render(pat, N, N)
    b = synth.render(pat, N, N, f)
    r = track_pair(a, b, Config(max_disp=32))
    m = eval_mask((N, N), f, 40)
    gx, gy = synth.displacement(f, N, N)
    assert disp_errors(r.ux, r.uy, gx, gy, m)["rms"] < 0.02
    se = strain_errors(r.strain, true_strain(f, (N, N)), m)
    assert se["Exx"] < 3e-4 and se["Eyy"] < 3e-4


def test_pair_rejects_size_mismatch():
    with pytest.raises(ValueError, match="sizes differ"):
        track_pair(np.zeros((10, 10)), np.zeros((10, 12)), Config())


def test_compose_of_translations():
    z = np.zeros((20, 20))
    cx, cy = compose(z + 1.0, z + 2.0, z + 0.5, z - 1.0)
    assert np.nanmax(np.abs(cx[:10, :10] - 1.5)) < 1e-12 and np.nanmax(np.abs(cy[:10, :10] - 1.0)) < 1e-12


def test_sequence_accumulates_without_drift(pat):
    frames = [synth.render(pat, N, N, synth.uniform_strain(0.004 * k, 0.0, (N / 2, N / 2), t=(1.5 * k, 0.7 * k)))
              for k in range(6)]
    out = list(track_sequence(frames, Config()))
    assert [k for k, _ in out] == [1, 2, 3, 4, 5]
    f5 = synth.uniform_strain(0.02, 0.0, (N / 2, N / 2), t=(7.5, 3.5))
    gx, gy = synth.displacement(f5, N, N)
    r5 = out[-1][1]
    assert disp_errors(r5.ux, r5.uy, gx, gy, eval_mask((N, N), f5, 40))["rms"] < 0.02


@pytest.mark.filterwarnings("error")
def test_sequence_reference_update_path(pat):
    fields = [synth.translation(2.0 * k, 1.0 * k) for k in range(4)]
    frames = [synth.render(pat, N, N, f) for f in fields]
    out = list(track_sequence(frames, Config(ref_update_conf=2.0)))  # forces an update every frame
    assert [r.timing["reference"] for _, r in out] == [0, 1, 2]
    gx, gy = synth.displacement(fields[3], N, N)
    r = out[-1][1]
    assert disp_errors(r.ux, r.uy, gx, gy, eval_mask((N, N), fields[3], 40))["rms"] < 0.02


def _crack_pair(pat, n=N):
    """Translation with a 16 px stripe whose content changes between the frames (a crack)."""
    a = synth.render(pat, n, n)
    f = synth.translation(12.3, 1.7)
    b = synth.render(pat, n, n, f)
    other = synth.render(synth.make_pattern(n, n, seed=77), n, n)
    c0 = n // 2 - 8
    b[:, c0:c0 + 16] = other[:, c0:c0 + 16]
    return a, b, synth.displacement(f, n, n)


def test_errors_stay_local_to_a_crack():
    # with the guard off, pixels 3-15 px from the invalid stripe show how far the filters smear the
    # discontinuity. On this pattern dc_cut = 2 / 3 / 4 give 0.27 / 0.38 / 0.44 px RMS there.
    from scipy.ndimage import distance_transform_edt

    n = 192
    a, b, (gx, gy) = _crack_pair(synth.make_pattern(n, n, seed=11), n)
    r = track_pair(a, b, Config(edge_guard=0))
    valid = np.isfinite(r.ux)
    d = distance_transform_edt(valid)
    band = valid & (d >= 3) & (d < 15)
    band[:32] = band[-32:] = False
    band[:, :32] = band[:, -32:] = False
    e = np.hypot(r.ux - gx, r.uy - gy)[band]
    assert np.sqrt(np.mean(e**2)) < 0.33


@pytest.fixture(scope="module")
def crack_runs(pat):
    a, b, gt = _crack_pair(pat)
    return gt, track_pair(a, b, Config(edge_guard=0)), track_pair(a, b, Config())


def test_edge_guard_removes_contaminated_pixels(crack_runs):
    from scipy.ndimage import distance_transform_edt, label

    (gx, gy), r_off, r = crack_runs
    cfg = Config()
    # Result.conf is the unguarded confidence: the guard changes only ux / uy, never conf. The one difference
    # between the runs is the border rule of refine (spec 2.4), which is part of the guard setting: conf is 0
    # within B of image 1's border and where the match is within B of image 2's, the same elsewhere
    B = 4 * cfg.sigma_w
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    d = np.minimum.reduce([xx, yy, N - 1 - xx, N - 1 - yy, xx + gx, yy + gy, N - 1 - xx - gx, N - 1 - yy - gy])
    assert (d >= B + 1).sum() > 0.4 * N * N and (d < B - 1).sum() > 0.2 * N * N
    deep = d >= B + 1  # (a pixel inside the crack has an arbitrary estimated match, which the rule may also zero)
    same = r.conf == r_off.conf
    assert same[deep & (r_off.conf >= cfg.conf_min)].all() and (r.conf[deep & ~same] == 0).all()
    assert (r.conf[d < B - 1] == 0).all()
    # guard sources: 4-connected low-confidence components of at least edge_guard_min_area (here the crack)
    lab, _ = label(r.conf < cfg.conf_min)
    big = np.bincount(lab.ravel()) >= cfg.edge_guard_min_area_px()
    big[0] = False
    src = big[lab]
    assert src.any()
    near = distance_transform_edt(~src) <= cfg.edge_guard_px()
    assert np.isnan(r.ux[near]).all()
    inner = np.zeros((N, N), bool)
    inner[32:-32, 32:-32] = True
    ok = np.isfinite(r.ux) & inner
    e = np.hypot(r.ux - gx, r.uy - gy)[ok]
    assert ok.sum() > 0.5 * inner.sum() and np.sqrt(np.mean(e**2)) < 0.05


def test_image_border_is_a_guard_source(crack_runs):
    # without a ROI the 1-px frame outside the image acts like the exterior of a ROI: pixels closer to the
    # border than the guard distance are invalid (unguarded they are valid but 0.02 px off, interior 0.0003).
    # With the guard on, refine also gives confidence 0 within B = 4 sigma_w of the border (spec 2.4): that strip
    # is a large low-confidence component and gets the guard band, so nothing closer than B + guard - 1 is valid
    _, r_off, r = crack_runs
    cfg = Config()
    g = cfg.edge_guard_px()
    B = 4 * cfg.sigma_w
    yy, xx = np.mgrid[0:N, 0:N]
    to_border = np.minimum.reduce([xx, yy, N - 1 - xx, N - 1 - yy])
    assert np.isnan(r.ux[to_border < g]).all()
    assert np.isnan(r.ux[to_border < B + g - 1]).all()
    assert np.isfinite(r_off.ux[2:5, 20:80]).all()  # the unguarded run keeps them (guard off -> no border source)
    assert 12 < g < 13
    assert np.isfinite(r.ux[28:35, 40:90]).mean() > 0.95  # just beyond B + guard (27.2 px)


def test_finish_marks_pixels_closer_to_the_image_frame_than_the_guard_distance():
    # _finish on its own: the frame just outside the image is a guard source, min(x, y, W-1-x, H-1-y) < guard.
    # conf is 1 everywhere (no low-confidence component, no ROI), so the only source is the frame. The guard
    # distances are larger than B = 4 sigma_w = 16 px, the width of the strip refine zeroes at the border, so
    # that strip cannot account for the result: this is the rule itself, not a consequence of the border rule.
    H, W = 70, 90
    yy, xx = np.mgrid[0:H, 0:W]
    to_frame = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy])
    ones = np.ones((H, W))
    for guard in (20.0, 20.5, 25.0):
        cfg = Config(edge_guard=guard)
        assert guard > 4 * cfg.sigma_w
        timing = {}
        mx, my, strain = _finish(2 * ones, -ones, ones, cfg, None, timing)
        assert np.array_equal(np.isfinite(mx), to_frame >= guard), guard
        assert np.array_equal(np.isfinite(my), np.isfinite(mx))
        assert (mx[np.isfinite(mx)] == 2.0).all() and (my[np.isfinite(my)] == -1.0).all()
        assert timing["strain"] >= 0 and strain["Exx"].shape == (H, W)
    # a ROI that covers the whole image changes nothing
    cfg = Config(edge_guard=20.0)
    mx_roi = _finish(ones, ones, ones, cfg, np.ones((H, W), bool), {})[0]
    assert np.array_equal(np.isfinite(mx_roi), to_frame >= 20)
    # automatic distance (edge_guard < 0): 3 x the spatial sigma of the longest filter, 12.2 px at the defaults
    cfg = Config()
    mx = _finish(ones, ones, ones, cfg, None, {})[0]
    assert np.array_equal(np.isfinite(mx), to_frame >= cfg.edge_guard_px())
    # edge_guard = 0 switches the guard, and with it the frame source, off
    assert np.isfinite(_finish(ones, ones, ones, Config(edge_guard=0), None, {})[0]).all()


def test_guard_ignores_isolated_low_confidence_speckles():
    # 2 % noise leaves a few small low-confidence clusters; guarding them (old rule) cost ~1 % coverage
    n = 256
    p = synth.make_pattern(n, n, seed=0)
    f = synth.uniform_strain(0.01, -0.003, (n / 2, n / 2))
    a = synth.degrade(synth.render(p, n, n), seed=1, noise=0.02, bits=8)
    b = synth.degrade(synth.render(p, n, n, f), seed=2, noise=0.02, bits=8)
    gx, gy = synth.displacement(f, n, n)
    r = track_pair(a, b, Config())
    assert disp_errors(r.ux, r.uy, gx, gy, eval_mask((n, n), f, 32))["valid_frac"] > 0.995


def test_guard_spares_neighbours_of_a_small_changed_blob():
    n = 192
    p = synth.make_pattern(n, n, seed=11)
    a = synth.render(p, n, n)
    b = synth.render(p, n, n, synth.translation(12.3, 1.7))
    other = synth.render(synth.make_pattern(n, n, seed=77), n, n)
    b[100:106, 100:106] = other[100:106, 100:106]  # 36 px^2 changed -> a ~132 px^2 low-confidence
    # component, still below edge_guard_min_area (~330 px^2): it is not a guard source
    r = track_pair(a, b, Config())
    # columns whose match lies within B + guard (28 px) of image 2's right border are invalid by the border rule
    # (x + 12.3 > 191 - 16 - 12.2 from x = 151 on), so the region ends at column 147
    assert np.isfinite(r.ux[32:-32, 32:-44]).mean() > 0.98


def test_sequence_keeps_invalidity_of_the_frame0_link(pat):
    # A foreign block that frames 1..3 show at one fixed image position (absent from frame 0) breaks the
    # 0 -> 1 match for the frame-0 pixels that move into it. After the reference moves on (forced every
    # frame) the later legs match the block with itself and look confident: the invalidity of the first
    # leg must survive in the 0 -> k output (spec 2.6).
    from scipy.ndimage import binary_dilation

    fields = [synth.translation(2.0 * k, 1.0 * k) for k in range(4)]
    frames = [synth.render(pat, N, N, f) for f in fields]
    other = synth.render(synth.make_pattern(N, N, seed=77), N, N)
    blk = np.s_[110:140, 110:140]
    for k in (1, 2, 3):
        frames[k] = frames[k].copy()
        frames[k][blk] = other[blk]
    out = dict(track_sequence(frames, Config(ref_update_conf=2.0)))
    assert [out[k].timing["reference"] for k in (1, 2, 3)] == [0, 1, 2]

    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    inner = np.zeros((N, N), bool)
    inner[114:136, 114:136] = True  # the block, shrunk by 4 px
    gone = np.zeros((N, N), bool)  # frame-0 pixels that move into the block on the 0 -> 1 leg
    gone[(yy + 1.0 >= 114) & (yy + 1.0 < 136) & (xx + 2.0 >= 114) & (xx + 2.0 < 136)] = True
    near = binary_dilation(inner, iterations=30)  # everything the block can disturb
    for k in (2, 3):
        r = out[k]
        assert np.isnan(r.ux[gone]).mean() >= 0.95, (k, np.isnan(r.ux[gone]).mean())
        gx, gy = synth.displacement(fields[k], N, N)
        far = eval_mask((N, N), fields[k], 40) & ~near
        assert disp_errors(r.ux, r.uy, gx, gy, far)["rms"] < 0.02


def test_pair_on_stripe_texture_is_all_invalid():
    x = np.arange(128)[None, :] * np.ones((128, 1))
    r = track_pair(0.5 + 0.2 * np.sin(x / 2), 0.5 + 0.2 * np.sin((x - 1.0) / 2), Config())
    assert np.isnan(r.ux).all() and (r.conf < Config().conf_min).all()


def _roi(n=N, lo=48, hi=208):
    m = np.zeros((n, n), bool)
    m[lo:hi, lo:hi] = True
    return m


def _assert_roi_eroded(ux, g=Config().edge_guard_px()):
    # exterior pixels beside the ROI edge are guard sources: a pixel d inside the edge is NaN for
    # d + 1 <= guard (12.2 px), the pixels deeper inside are valid
    assert 5 + 1 <= g < 20 + 1
    for d, finite in ((5, False), (20, True)):
        for line in (ux[128, 48 + d], ux[128, 207 - d], ux[48 + d, 128], ux[207 - d, 128]):
            assert np.isfinite(line) == finite, (d, line)
    assert np.isfinite(ux[100:156, 100:156]).all()


def test_roi_exterior_is_eroded_by_the_guard(pat):
    a = synth.render(pat, N, N)
    b = synth.render(pat, N, N, synth.translation(3.2, 1.1))
    r = track_pair(a, b, Config(), mask=_roi())
    _assert_roi_eroded(r.ux)
    assert np.isnan(r.ux[~_roi()]).all() and np.isnan(r.uy[~_roi()]).all()


def test_sequence_output_is_guarded_on_the_frame0_grid(pat):
    frames = [synth.render(pat, N, N, synth.translation(2.0 * k, 1.0 * k)) for k in range(3)]
    out = list(track_sequence(frames, Config(), mask=_roi()))
    assert [k for k, _ in out] == [1, 2]
    for _, r in out:
        _assert_roi_eroded(r.ux)
        assert np.isnan(r.ux[~_roi()]).all()


def test_track_pair_rejects_small_image_and_wrong_mask_shape():
    with pytest.raises(ValueError, match="smaller than poc_win"):
        track_pair(np.zeros((32, 128)), np.zeros((32, 128)), Config())
    with pytest.raises(ValueError, match="smaller than poc_win"):
        track_pair(np.zeros((128, 48)), np.zeros((128, 48)), Config())
    with pytest.raises(ValueError, match="mask"):
        track_pair(np.zeros((64, 64)), np.zeros((64, 64)), Config(), mask=np.ones((64, 65), bool))


def test_track_sequence_rejects_bad_input():
    z = np.zeros((64, 64))
    with pytest.raises(ValueError, match="no frames"):
        list(track_sequence([], Config()))
    with pytest.raises(ValueError, match="frame 1"):
        list(track_sequence([z, np.zeros((64, 48))], Config()))
    with pytest.raises(ValueError, match="mask"):
        list(track_sequence([z, z], Config(), mask=np.ones((64, 48), bool)))
    with pytest.raises(ValueError, match="empty"):
        list(track_sequence([z, z], Config(), mask=np.zeros((64, 64), bool)))


# ---- accuracy next to the image borders (spec 2.4) ----

N512 = 512
BAND_EDGES = (32, 48)  # px; the first band starts at the first valid pixel, B + guard = 28.2 px at the defaults


def _border_distance(field, n=N512):
    """Per pixel of image 1: min over the distance of the pixel to the border of image 1 and the distance of
    its true match X' to the border of image 2 (negative where the match leaves image 2)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    x, y = field.fwd(xx, yy)
    d1 = np.minimum.reduce([xx, yy, n - 1 - xx, n - 1 - yy])
    d2 = np.minimum.reduce([x, y, n - 1 - x, n - 1 - y])
    return np.minimum(d1, d2)


def _band_table(r, field, dist, n=N512):
    """[(label, count, rms, max)] of the error of the valid pixels in the bands of `dist`."""
    gx, gy = synth.displacement(field, n, n)
    valid = np.isfinite(r.ux)
    err = np.hypot(r.ux - gx, r.uy - gy)
    edges = (-np.inf,) + BAND_EDGES + (np.inf,)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = valid & (dist >= lo) & (dist < hi)
        e = err[sel]
        rows.append((f"[{max(lo, dist[valid].min()):.0f}, {hi})", int(sel.sum()),
                     float(np.sqrt(np.mean(e**2))) if e.size else float("nan"), float(e.max()) if e.size else float("nan")))
    return rows


def _check_bands(r, field, dist, cfg):
    rows = _band_table(r, field, dist)
    print("\n".join(f"  {lab:>10}: {n:7d} px  rms {rms:.2e}  max {mx:.2e}" for lab, n, rms, mx in rows))
    valid = np.isfinite(r.ux)
    # nothing closer than B + guard - 1 to a border (guard-1 because the nearest low-confidence pixel is at B - 1)
    assert dist[valid].min() >= 4 * cfg.sigma_w + cfg.edge_guard_px() - 1
    for lab, n, rms, mx in rows:
        assert n > 1000, (lab, n)
        assert rms <= 1e-3 and mx <= 1e-2, (lab, rms, mx)


@pytest.fixture(scope="module")
def pat512():
    return synth.make_pattern(N512, N512, seed=21)


def test_valid_pixels_next_to_the_border_are_as_accurate_as_the_interior(pat512):
    # before the rule, valid pixels 13-32 px from the border had rms 1e-2 .. 8e-4 px and up to 0.11 px at conf 0.997
    cfg = Config()
    c = N512 / 2
    f = synth.uniform_strain(0.01, -0.003, (c, c))
    r = track_pair(synth.render(pat512, N512, N512), synth.render(pat512, N512, N512, f), cfg)
    _check_bands(r, f, _border_distance(f), cfg)


def test_valid_pixels_next_to_matches_that_leave_image_2_are_as_accurate_as_the_interior(pat512):
    # large translation: the matches of the right and upper strips of image 1 leave image 2 or come within B of its
    # border; before the rule the pixels beside them had errors of up to 4.7e-2 px at conf 0.992
    cfg = Config(max_disp=64.0)
    c = N512 / 2
    f = synth.uniform_strain(0.01, -0.003, (c, c), t=(37.3, -21.6))
    r = track_pair(synth.render(pat512, N512, N512), synth.render(pat512, N512, N512, f), cfg)
    _check_bands(r, f, _border_distance(f), cfg)


# ---- confident wrong pixels next to a coarse-stage failure (spec 2.3) ----

def test_reported_pixels_next_to_a_coarse_stage_failure_are_accurate():
    # Translation (45.85, -33.65) at 256 px with max_disp 64: the coarsest POC window cannot hold the motion, so more than
    # a fifth of the coarse field is off by over a pixel (up to ~100 px). The fine stage cannot repair those pixels, but it
    # must not let their displacement leak into the pre-warp of the good ones: before the trusted pre-warp, 1437 reported
    # pixels beside the failure were off by up to 0.40 px at confidence 0.94-1.0.
    n = 256
    p = synth.make_pattern(n, n, seed=0)
    f = synth.uniform_strain(0.01, -0.003, (n / 2, n / 2), t=(45.85, -33.65))
    a, b = synth.render(p, n, n), synth.render(p, n, n, f)
    gx, gy = synth.displacement(f, n, n)
    cfg = Config(max_disp=64)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx, cy = grid_sample(coarse_flow(a, b, cfg), xx, yy)
    assert (np.hypot(cx - gx, cy - gy) > 1.0).mean() > 0.2  # the case keeps exercising a coarse failure
    r = track_pair(a, b, cfg)
    ok = np.isfinite(r.ux)
    assert ok.sum() > 0.1 * n * n
    e = np.hypot(r.ux - gx, r.uy - gy)[ok]
    assert np.sqrt(np.mean(e**2)) <= 1e-3 and e.max() <= 1e-2, (np.sqrt(np.mean(e**2)), e.max(), (e > 1e-2).sum())


def test_mean_conf_averages_over_reportable_pixels_only():
    # The border strip (conf 0 by the border rule of refine) is a geometric exclusion, not a tracking-quality signal.
    n, B = 128, 4 * Config().sigma_w
    cfg = Config()
    conf = np.zeros((n, n))
    conf[16:-16, 16:-16] = 0.9
    flow = type("Leg", (), {"conf": conf, "raw_ux": np.zeros((n, n)), "raw_uy": np.zeros((n, n))})
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    z = (np.zeros((n, n)), np.zeros((n, n)))
    assert B == 16
    assert _mean_conf(flow, z, None, xx, yy, cfg) == pytest.approx(0.9)
    # inside the reportable zone a low confidence does count
    low = conf.copy()
    low[16:-16, 16:64] = 0.1
    flow_low = type("Leg", (), {"conf": low, "raw_ux": flow.raw_ux, "raw_uy": flow.raw_uy})
    assert _mean_conf(flow_low, z, None, xx, yy, cfg) == pytest.approx((48 * 0.1 + 48 * 0.9) / 96)
    # a zero inside the reportable zone is a tracking failure, not a border effect: it counts
    hole = conf.copy()
    hole[40:60, 40:60] = 0.0
    flow_hole = type("Leg", (), {"conf": hole, "raw_ux": flow.raw_ux, "raw_uy": flow.raw_uy})
    assert _mean_conf(flow_hole, z, None, xx, yy, cfg) == pytest.approx(0.9 * (96 * 96 - 400) / (96 * 96))
    # the ROI restricts the average further; a ROI entirely in the strip has no reportable pixel
    roi = np.zeros((n, n), bool)
    roi[:, :80] = True
    assert _mean_conf(flow_low, z, roi, xx, yy, cfg) == pytest.approx((48 * 0.1 + 16 * 0.9) / 64)
    strip = np.zeros((n, n), bool)
    strip[:, :10] = True
    assert _mean_conf(flow, z, strip, xx, yy, cfg) == 0.0
    # matches that leave image 2 are not reportable either (border rule), and u0r moves the frame-0 grid into the
    # leg's reference: pixels whose frame-0 position falls outside the leg's image do not count
    shifted = type("Leg", (), {"conf": conf, "raw_ux": np.full((n, n), 8.0), "raw_uy": np.zeros((n, n))})
    conf_shift = conf.copy()
    conf_shift[:, -24:] = 0.0  # x + 8 > 127 - 16 for x >= 104: zero by the border rule
    shifted.conf = conf_shift
    assert _mean_conf(shifted, z, None, xx, yy, cfg) == pytest.approx(0.9)
    u0r = (np.full((n, n), 5.0), np.full((n, n), 0.0))
    assert _mean_conf(flow, u0r, None, xx, yy, cfg) == pytest.approx(0.9)


def test_sequence_without_roi_does_not_update_the_reference_on_an_ideal_translation():
    # 128 x 128, ideal translation, no ROI, default config: the 16 px border strips and the matches that leave
    # image 2 are zero confidence by the border rule (spec 2.4) and used to pull the mean under ref_update_conf = 0.5
    # (0.47 at frame 2 here, while the reportable pixels have 1.00), so the reference moved on at every frame.
    n = 128
    p = synth.make_pattern(n, n, seed=3)
    frames = [synth.render(p, n, n, synth.translation(5.0 * k, 2.5 * k)) for k in range(3)]
    out = dict(track_sequence(frames, Config()))
    assert [out[k].timing["reference"] for k in (1, 2)] == [0, 0]
    assert all(out[k].timing["mean_conf"] > 0.9 for k in (1, 2))

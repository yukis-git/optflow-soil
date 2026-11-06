import numpy as np
import pytest

from optflow_ref import synth
from optflow_ref.config import Config
from optflow_ref.metrics import disp_errors, eval_mask
from optflow_ref.interp import gauss_blur
from optflow_ref.phaseflow import local_solve, prewarp, refine

N = 192


@pytest.fixture(scope="module")
def pat():
    p = synth.make_pattern(N, N, seed=11)
    return p, synth.render(p, N, N)


def _run(pat, field, init_err=0.3):
    p, a = pat
    b = synth.render(p, N, N, field)
    gx, gy = synth.displacement(field, N, N)
    fl = refine(a, b, gx + init_err, gy - init_err, Config())
    return fl, disp_errors(fl.ux, fl.uy, gx, gy, eval_mask((N, N), field, 32))


@pytest.mark.parametrize("t", [(0.25, 0.5), (0.75, -0.3)])
def test_subpixel_translation(pat, t):
    _, e = _run(pat, synth.translation(*t))
    assert e["rms"] < 0.005


@pytest.mark.parametrize("eps", [0.001, 0.05])
def test_uniform_strain_does_not_bias(pat, eps):
    _, e = _run(pat, synth.uniform_strain(eps, -eps / 2, (N / 2, N / 2)))
    assert e["rms"] < 0.01


def test_confidence_low_on_untextured_patch(pat):
    p, a = pat
    b = synth.render(p, N, N, synth.translation(0.5, 0.5))
    a = a.copy(); b = b.copy()
    a[80:112, 80:112] = 0.5
    b[80:112, 80:112] = 0.5
    fl = refine(a, b, np.full((N, N), 0.5), np.full((N, N), 0.5), Config())
    assert np.median(fl.conf[90:102, 90:102]) < 0.2
    assert np.median(fl.conf[30:60, 30:60]) > 0.5


def test_noise_error_near_lower_bound(pat):
    from optflow_ref.metrics import crlb_rms

    p, a0 = pat
    f = synth.uniform_strain(0.02, -0.01, (N / 2, N / 2), t=(0.4, 0.3))
    a = synth.degrade(a0, seed=1, noise=0.01, bits=8)
    b = synth.degrade(synth.render(p, N, N, f), seed=2, noise=0.01, bits=8)
    gx, gy = synth.displacement(f, N, N)
    fl = refine(a, b, gx + 0.3, gy - 0.3, Config())
    e = disp_errors(fl.ux, fl.uy, gx, gy, eval_mask((N, N), f, 32))
    assert e["rms"] < 1.3 * crlb_rms(a0, 0.01, Config().sigma_w)


def test_quadratic_model_resolves_shear_band_better(pat):
    from scipy.ndimage import gaussian_filter

    p, a = pat
    f = synth.ShearBand(4.0, 10.0, (N / 2, N / 2), angle_deg=10)
    b = synth.render(p, N, N, f)
    gx, gy = synth.displacement(f, N, N)
    init = (gaussian_filter(gx, 8) + 0.2, gaussian_filter(gy, 8) - 0.2)
    m = eval_mask((N, N), f, 40)
    r1 = refine(a, b, *init, Config(shape_order=1))
    r2 = refine(a, b, *init, Config(shape_order=2))
    e1 = disp_errors(r1.ux, r1.uy, gx, gy, m)["rms"]
    e2 = disp_errors(r2.ux, r2.uy, gx, gy, m)["rms"]
    assert e2 < 0.6 * e1


def test_confidence_rejects_wrong_match(pat):
    p, a = pat
    b = synth.render(p, N, N, synth.translation(0.5, 0.5))
    other = synth.render(synth.make_pattern(N, N, seed=99), N, N)  # unrelated texture
    good = refine(a, b, np.full((N, N), 0.5), np.full((N, N), 0.5), Config())
    bad = refine(a, other, np.zeros((N, N)), np.zeros((N, N)), Config())
    inner = np.s_[40:-40, 40:-40]
    assert np.median(good.conf[inner]) > 0.8 and np.median(good.resid[inner]) < 0.05
    assert np.median(bad.conf[inner]) < 0.2


def test_brightness_offset_and_illumination_ramp_do_not_bias(pat):
    p, a = pat
    f = synth.uniform_strain(0.01, -0.003, (N / 2, N / 2))
    b = synth.degrade(synth.render(p, N, N, f), gain=1.2, offset=-0.05, grad=(0.2, 0.1))
    gx, gy = synth.displacement(f, N, N)
    fl = refine(a, b, gx + 0.3, gy - 0.3, Config())
    assert disp_errors(fl.ux, fl.uy, gx, gy, eval_mask((N, N), f, 32))["rms"] < 0.01


def test_confidence_stays_high_on_noisy_correct_match(pat):
    p, a = pat
    f = synth.translation(0.4, 0.3)
    a2 = synth.degrade(a, seed=1, noise=0.02, bits=8)
    b = synth.degrade(synth.render(p, N, N, f), seed=2, noise=0.02, bits=8)
    fl = refine(a2, b, np.full((N, N), 0.4), np.full((N, N), 0.3), Config())
    assert np.median(fl.conf[40:-40, 40:-40]) > 0.5



def test_noise_efficiency_at_2_percent(pat):
    """Averaged over noise draws the error stays within 20 % of the lower bound: the DC cut of
    the filter bank (dc_cut = 2) costs ~12 % here; a cut twice as wide would cost more."""
    from optflow_ref.metrics import crlb_rms

    p, a0 = pat
    f = synth.uniform_strain(0.02, -0.01, (N / 2, N / 2), t=(0.4, 0.3))
    b0 = synth.render(p, N, N, f)
    gx, gy = synth.displacement(f, N, N)
    m = eval_mask((N, N), f, 32)
    ratios = []
    for sd in (1, 3, 5):
        a = synth.degrade(a0, seed=sd, noise=0.02, bits=8)
        b = synth.degrade(b0, seed=sd + 1, noise=0.02, bits=8)
        fl = refine(a, b, gx + 0.3, gy - 0.3, Config())
        ratios.append(disp_errors(fl.ux, fl.uy, gx, gy, m)["rms"] / crlb_rms(a0, 0.02, Config().sigma_w))
    assert np.mean(ratios) < 1.2


def _stripes(n=128, shift=1.0):
    x = np.arange(n)[None, :] * np.ones((n, 1))
    return 0.5 + 0.2 * np.sin(x / 2), 0.5 + 0.2 * np.sin((x - shift) / 2)


@pytest.mark.parametrize("order", [1, 2])
def test_one_dimensional_texture_gives_low_confidence_not_an_error(order):
    # a pure stripe texture constrains one displacement component only: the local systems are singular
    # (aperture problem). That must read as "no information", not as a LinAlgError or a confident field.
    a, b = _stripes()
    z = np.zeros(a.shape)
    fl = refine(a, b, z, z, Config(shape_order=order))
    assert np.isfinite(fl.ux).all() and np.isfinite(fl.uy).all()
    assert fl.conf.max() < Config().conf_min


def _conf_for_translation(pat, t, **cfg_kw):
    """Confidence of refine on a pure translation t, started from the exact field."""
    p, a = pat
    b = synth.render(p, N, N, synth.translation(*t))
    gx, gy = synth.displacement(synth.translation(*t), N, N)
    return refine(a, b, gx, gy, Config(**cfg_kw)).conf


def test_confidence_is_zero_within_four_sigma_w_of_the_image_border(pat):
    # Gabor responses within the guard of the border come from reflect padding and the window averages them
    # over 4 sigma_w: the padded texture is self-consistent, so the residual cannot see it (spec 2.4)
    cfg = Config()
    B = int(4 * cfg.sigma_w)
    assert cfg.edge_guard_px() > 0 and B == 16
    conf = _conf_for_translation(pat, (0.4, 0.3))
    mid = slice(40, N - 40)  # rows / columns far from the other borders
    for strip, inner in ((conf[mid, B - 1], conf[mid, B + 1]), (conf[mid, N - B], conf[mid, N - B - 2]),
                         (conf[B - 1, mid], conf[B + 1, mid]), (conf[N - B, mid], conf[N - B - 2, mid])):
        assert (strip == 0).all()  # distance B - 1 from the border
        assert np.median(inner) > 0.5  # distance B + 1, in good texture
    assert (conf[:B, :] == 0).all() and (conf[-B:, :] == 0).all()
    assert (conf[:, :B] == 0).all() and (conf[:, -B:] == 0).all()


def test_confidence_is_zero_where_the_match_is_within_four_sigma_w_of_the_border_of_image_2(pat):
    # X' = X + (30.5, 0): the match of column X lies at X + 30.5 in image 2
    B = 16
    conf = _conf_for_translation(pat, (30.5, 0.0))
    xs = np.arange(N)
    rows = slice(40, N - 40)
    near = (xs + 30.5) > N - 1 - B  # matches within B of the right border of image 2 (or beyond it)
    assert near.any() and (~near)[B + 4:].any()
    assert (conf[rows][:, near] == 0).all()
    last_ok = np.nonzero(~near)[0].max()
    assert np.median(conf[rows, last_ok]) > 0.5  # the last column whose match is B or more inside
    assert last_ok + 1 == np.nonzero(near)[0].min()


def test_border_rule_is_off_with_the_guard_off(pat):
    # edge_guard = 0 keeps the old rule: zero only where the match leaves image 2. Pinned at the border columns
    # and rows (Rust parity fixture 'refine_border'): translation (-1.5, 1.7) takes columns 0, 1 and the last
    # two rows out of the image, columns 2, 3 and the rows above stay in.
    conf = _conf_for_translation(pat, (-1.5, 1.7), edge_guard=0.0)
    mid = slice(40, N - 40)
    assert (conf[:, :2] == 0).all() and (conf[-2:, :] == 0).all()
    assert np.median(conf[mid, 2]) > 0.5 and np.median(conf[mid, 3]) > 0.5
    assert np.median(conf[N - 3, mid]) > 0.5 and np.median(conf[N - 4, mid]) > 0.5


# ---- trusted first-order pre-warp (spec 2.3) ----

def _affine_field(h, w):
    """An affine displacement field and its exact slope (dux/dx, duy/dx, dux/dy, duy/dy)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ux = 3.0 + 0.02 * xx - 0.01 * yy
    uy = -2.0 + 0.005 * xx + 0.015 * yy
    slope = np.empty((h, w, 4))
    slope[..., 0], slope[..., 1], slope[..., 2], slope[..., 3] = 0.02, 0.005, -0.01, 0.015
    return ux, uy, slope


def test_prewarp_replaces_untrusted_pixels_by_the_first_order_extrapolation_of_the_trusted_ones():
    # A block where the coarse stage failed (conf 0, displacement +100 px) and, around it, a ring of pixels that
    # look confident (conf 1) but sit closer than the filter reach (12.2 px at the defaults) to the block and
    # carry garbage too: the guard band counts as untrusted. Whatever the untrusted values are, they are
    # replaced by u(N) + grad u(N) (X - N) from the nearest trusted pixel N, which reproduces an affine field exactly.
    from scipy.ndimage import distance_transform_edt

    cfg = Config()
    h, w = 96, 112
    ux, uy, slope = _affine_field(h, w)
    conf = np.ones((h, w))
    block = np.zeros((h, w), bool)
    block[30:60, 40:80] = True
    conf[block] = 0.0
    ring = ~block & (distance_transform_edt(~block) <= 8.0)
    assert 8.0 < cfg.edge_guard_px() and (~block & ~ring & (distance_transform_edt(~block) > cfg.edge_guard_px())).any()
    bad_x, bad_y = ux.copy(), uy.copy()
    bad_x[block], bad_y[block] = 100.0, 100.0
    bad_x[ring], bad_y[ring] = 50.0, -50.0
    sx, sy = prewarp(bad_x, bad_y, slope, conf, cfg)
    assert np.abs(sx - gauss_blur(ux, cfg.sigma_s)).max() < 1e-12
    assert np.abs(sy - gauss_blur(uy, cfg.sigma_s)).max() < 1e-12


def test_prewarp_without_untrusted_pixels_is_the_plain_blur():
    cfg = Config()
    rng = np.random.default_rng(0)
    ux, uy = rng.normal(size=(2, 64, 80))
    slope = rng.normal(size=(64, 80, 4))
    sx, sy = prewarp(ux, uy, slope, np.ones((64, 80)), cfg)  # every pixel confident
    assert np.array_equal(sx, gauss_blur(ux, cfg.sigma_s)) and np.array_equal(sy, gauss_blur(uy, cfg.sigma_s))
    sx, sy = prewarp(ux, uy, slope, np.zeros((64, 80)), cfg)  # no pixel confident: nothing to extrapolate from
    assert np.array_equal(sx, gauss_blur(ux, cfg.sigma_s)) and np.array_equal(sy, gauss_blur(uy, cfg.sigma_s))
    conf = np.ones((64, 80))
    conf[10, 10] = cfg.conf_min - 1e-9  # a single low pixel untrusts its neighbourhood within the filter reach
    sx, _ = prewarp(ux, uy, slope, conf, cfg)
    assert not np.array_equal(sx, gauss_blur(ux, cfg.sigma_s))
    conf[10, 10] = cfg.conf_min  # conf_min itself is confident
    assert np.array_equal(prewarp(ux, uy, slope, conf, cfg)[0], gauss_blur(ux, cfg.sigma_s))


def _model_sums(ux, uy):
    """Per-pixel sums over four filters (unequal weights and phase gradient lengths) of a field that the shape
    function can follow exactly: c_k = g_k . u(q), the right-hand side of g_q . P_p(q - p) = dphi_q + g_q . u_q for a
    perfect match (dphi = 0 is not needed: the model only sees c)."""
    h, w = ux.shape
    gxx, gxy, gyy, gxc, gyc, gcc, gw = (np.zeros((h, w)) for _ in range(7))
    for ang, length, wt in ((0.0, 1.0, 1.0), (45.0, 0.7, 2.0), (90.0, 1.3, 0.5), (135.0, 0.9, 1.5)):
        gx, gy = length * np.cos(np.radians(ang)), length * np.sin(np.radians(ang))
        c = gx * ux + gy * uy
        gxx += wt * gx * gx; gxy += wt * gx * gy; gyy += wt * gy * gy
        gxc += wt * gx * c; gyc += wt * gy * c; gcc += wt * c * c
        gw += wt
    return gxx, gxy, gyy, gxc, gyc, gcc, gw


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("name,coef", [("shear", (0.0, 0.0, 0.03, 0.0)), ("rotation", (0.0, 0.02, -0.02, 0.0)),
                                       ("general", (0.01, 0.03, -0.02, 0.005))])
def test_local_solve_slope_is_dux_dx_duy_dx_dux_dy_duy_dy(order, name, coef):
    # The 5th output of local_solve is the affine part of the shape function at the window centre, in the order
    # (dux/dx, duy/dx, dux/dy, duy/dy) that prewarp reads (g[..., 0] * dx + g[..., 2] * dy for ux). Pinned against the
    # analytic gradient of a sheared, a rotated and a general field (four distinct coefficients: a swapped pair is
    # caught); for order 2 a quadratic term makes the gradient vary over the image, and the slope is the one at p.
    a1, b1, a2, b2 = coef
    q = 4e-4 if order == 2 else 0.0
    h, w = 64, 72
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ux = 2.0 + a1 * xx + a2 * yy + q * xx * xx
    uy = -1.5 + b1 * xx + b2 * yy + q * xx * yy
    sums = _model_sums(ux, uy)
    gux, guy, energy, resid, slope = local_solve(*sums, ux + 0.25, uy - 0.25, Config().sigma_w, order)
    assert slope.shape == (h, w, 4)
    inner = (slice(20, -20), slice(20, -20))  # farther than the window radius 4 sigma_w = 16 from the edges
    want = np.stack([a1 + 2 * q * xx, b1 + q * yy, a2 + 0 * xx, b2 + q * xx], axis=-1)  # d/dx, d/dx, d/dy, d/dy
    assert np.abs(slope[inner] - want[inner]).max() < 1e-8
    assert np.abs(gux[inner] - ux[inner]).max() < 1e-7 and np.abs(guy[inner] - uy[inner]).max() < 1e-7
    assert resid[inner].max() < 1e-3  # (sum W w c^2 - theta . b cancels to ~1e-9 of the sum, so sqrt gives ~3e-5)


@pytest.mark.parametrize("order", [1, 2])
def test_local_solve_slope_is_zero_where_the_system_is_degenerate(order):
    h, w = 40, 40
    z = np.zeros((h, w))
    gux, guy, energy, resid, slope = local_solve(z, z, z, z, z, z, z, z + 0.7, z - 0.4, 4.0, order)
    assert np.array_equal(slope, np.zeros((h, w, 4)))
    assert np.array_equal(gux, z + 0.7) and np.array_equal(resid, np.full((h, w), np.pi))


def test_half_moments_match_brute_force():
    from optflow_ref.interp import gauss_kernel
    from optflow_ref.phaseflow import _DIRS, HalfMoments

    a = np.random.default_rng(0).normal(size=(50, 56))
    hm = HalfMoments(a.shape, 3.0)
    k = gauss_kernel(3.0)
    r = len(k) // 2
    ap = np.pad(a, r, mode="edge")
    y, x = 11, 40
    for h in (0, 1, 2, 3, 4, 5, 6, 7):
        for pw in ((0, 0), (1, 1), (2, 0), (0, 1)):
            acc = 0.0
            for ty in range(-r, r + 1):
                for tx in range(-r, r + 1):
                    s = _DIRS[h][0] * tx + _DIRS[h][1] * ty
                    acc += k[ty + r] * k[tx + r] * (1.0 if s > 0 else 0.5 if s == 0 else 0.0) * tx ** pw[0] * ty ** pw[1] * ap[y + ty + r, x + tx + r]
            assert hm("a", a, pw, h)[y, x] == pytest.approx(acc, abs=1e-10)


def test_crack_split_off_is_the_default_and_leaves_flow_unchanged(pat):
    p, a = pat
    f = synth.uniform_strain(0.01, -0.005, (N / 2, N / 2), t=(1.3, -0.6))
    b = synth.render(p, N, N, f)
    gx, gy = synth.displacement(f, N, N)
    f1 = refine(a, b, gx + 0.2, gy - 0.2, Config())
    f2 = refine(a, b, gx + 0.2, gy - 0.2, Config(crack_split=False, split_tol=0.1, split_gain=0.5, split_from_outer=3))
    for k in ("ux", "uy", "conf", "resid", "resid_full", "slope", "split"):
        assert np.array_equal(getattr(f1, k), getattr(f2, k)), k
    assert (f1.split == -1).all() and f1.split.dtype == np.int8 and f1.slope.shape == (N, N, 4)
    # a smooth field has no residual above split_tol: turning the split on adopts (almost) nothing and changes nothing much
    f3 = refine(a, b, gx + 0.2, gy - 0.2, Config(crack_split=True))
    assert (f3.split >= 0).mean() < 0.01
    assert np.nanmax(np.abs(f3.ux - f1.ux)[40:-40, 40:-40]) < 0.01


def test_crack_split_picks_the_half_window_that_avoids_the_crack():
    n = 192
    p = synth.make_pattern(n, n, seed=4)
    a = synth.render(p, n, n)
    f = synth.Crack((n / 2, n / 2), 0.0, 3.0, 0.0)
    b = synth.render(p, n, n, f)
    gx, gy = synth.displacement(f, n, n)
    cfg = Config(crack_split=True)
    fl = refine(a, b, gx + 0.1, gy + 0.1, cfg)
    base = refine(a, b, gx + 0.1, gy + 0.1, Config())
    rows = slice(60, 130)
    # left of the crack the window points left (h = 4), right of it right (h = 0)
    assert (fl.split[rows, 88:92] == 4).mean() > 0.8
    assert (fl.split[rows, 102:106] == 0).mean() > 0.8
    assert (fl.split[rows, 20:60] == -1).all()
    # the residual of the full window is kept; the adopted residual is lower and the pixels become confident
    ad = fl.split[rows, 88:92] >= 0
    assert (fl.resid_full[rows, 88:92][ad] > 0.25).all()
    assert (fl.resid[rows, 88:92][ad] < 0.7 * fl.resid_full[rows, 88:92][ad]).all()
    assert fl.conf[rows, 88:92].mean() > base.conf[rows, 88:92].mean() + 0.2


def test_crack_sides_labels_the_two_sides_and_invalidates_the_band():
    n = 192
    p = synth.make_pattern(n, n, seed=4)
    a = synth.render(p, n, n)
    f = synth.Crack((n / 2, n / 2), 0.0, 3.0, 0.0)
    b = synth.render(p, n, n, f)
    gx, gy = synth.displacement(f, n, n)
    cfg = Config(crack_split=True, crack_sides=True)
    fl = refine(a, b, gx + 0.1, gy + 0.1, cfg)
    rows = slice(60, 130)
    left, right = fl.split[rows, 66:86], fl.split[rows, 106:126]
    assert (left == 9).mean() > 0.95 and (right == 10).mean() > 0.95  # side k is labelled 8 + k
    assert fl.crack_band[rows, 94:98].all() and not fl.crack_band[rows, :60].any()
    assert (fl.conf[fl.crack_band] < 1e-6).all()
    assert fl.split_global.shape == fl.split.shape and (fl.split_global < 8).all()
    # without crack_sides the band and labels are absent and split is split_global
    fl0 = refine(a, b, gx + 0.1, gy + 0.1, Config(crack_split=True))
    assert not fl0.crack_band.any() and np.array_equal(fl0.split, fl0.split_global)
    e = np.hypot(fl.ux - gx, fl.uy - gy)
    e0 = np.hypot(fl0.ux - gx, fl0.uy - gy)
    sel = np.zeros((n, n), bool)
    sel[rows, 76:84] = True
    sel[rows, 108:116] = True  # 12..20 px from the crack
    ok = sel & (fl.conf > 0.2)
    assert ok.sum() > 300 and np.median(e[ok]) < np.median(e0[ok])

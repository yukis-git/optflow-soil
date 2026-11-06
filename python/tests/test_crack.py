import numpy as np
import pytest

from optflow_ref import crack
from optflow_ref.config import Config
from optflow_ref.strain import displacement_gradient


def _ci(alpha_deg, n=120, x0=60.0, y0=60.0, gap=3.0):
    """Synthetic half-window bookkeeping of a straight crack through (x0, y0) with normal angle alpha: the full-window
    residual is high within 14 px of it, the pixels 5..13 px from it chose the half window pointing away from it."""
    a = np.deg2rad(alpha_deg)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    s = np.cos(a) * (xx - x0) + np.sin(a) * (yy - y0)
    split = np.full((n, n), -1, np.int8)
    away = (np.abs(s) >= gap + 2) & (np.abs(s) <= 13)
    # quantise the outward normal +-n to the nearest of the 8 directions (0 = +x, 1 = +x+y, ... counter-clockwise in y down)
    h_plus = int(np.rint(alpha_deg / 45.0)) % 8
    split[away & (s > 0)] = h_plus
    split[away & (s < 0)] = (h_plus + 4) % 8
    resid_full = np.where(np.abs(s) < 14, 0.6, 0.1)
    resid = np.where(split >= 0, 0.1, resid_full)
    return dict(resid_full=resid_full, resid=resid, split=split), s


@pytest.mark.parametrize("alpha", [0.0, 30.0])
def test_geometry_finds_centre_line_and_normal(alpha):
    cfg = Config(crack_split=True, crack_out=True)
    ci, s = _ci(alpha)
    geo = crack.geometry(ci, cfg)
    assert geo["C"].sum() > 3000
    line = geo["line"]
    assert line.sum() > 40 and np.abs(s[line]).max() < 2.5  # on the crack
    a = np.deg2rad(alpha)
    inner = geo["C"].copy()
    inner[:25], inner[-25:], inner[:, :25], inner[:, -25:] = False, False, False, False  # the line ends at the image border
    nx, ny = geo["nx"][inner], geo["ny"][inner]
    assert np.abs(nx * np.sin(a) - ny * np.cos(a)).max() < 0.05  # parallel to n (tilt < 3 deg) ...
    assert (nx * np.cos(a) + ny * np.sin(a)).min() > 0.99  # ... and oriented like it (+ side along n)
    assert np.isnan(geo["nx"][~geo["C"]]).all()


def test_geometry_without_split_pixels_is_empty():
    ci, _ = _ci(0.0)
    ci["split"][:] = -1
    geo = crack.geometry(ci, Config(crack_split=True))
    assert geo["C"].any() and not geo["line"].any() and geo["iy"] is None
    cod, csd = crack.jump(np.zeros((120, 120)), np.zeros((120, 120)), (np.zeros((120, 120)),) * 4, geo, Config())
    assert np.isnan(cod).all() and np.isnan(csd).all()


def test_small_residual_spots_are_not_cracks():
    ci, _ = _ci(0.0)
    ci["resid_full"][5:9, 5:9] = 1.0
    geo = crack.geometry(ci, Config(crack_split=True))
    assert not geo["C"][5:9, 5:9].any()


@pytest.mark.parametrize("alpha", [0.0, 30.0])
def test_jump_recovers_cod_and_csd_with_one_sided_gradients(alpha):
    cfg = Config(crack_split=True, crack_out=True, crack_a=8.0)
    ci, s = _ci(alpha)
    a = np.deg2rad(alpha)
    n, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    yy, xx = np.mgrid[0:120, 0:120].astype(float)
    J = 2.5 * n - 0.8 * t
    # each side: a linear field (uniform strain on the + side only) plus the jump on the + side
    ux = np.where(s > 0, 0.002 * (xx - 60) + J[0], 0.0)
    uy = np.where(s > 0, -0.001 * (yy - 60) + J[1], 0.0)
    ux[np.abs(s) < 4], uy[np.abs(s) < 4] = np.nan, np.nan  # the gap is invalid
    geo = crack.geometry(ci, cfg)
    grad = displacement_gradient(ux, uy, cfg.strain_win, half=crack.strain_half_map(geo, cfg.strain_win))
    cod, csd = crack.jump(ux, uy, grad, geo, cfg)
    tang = -np.sin(a) * (xx - 60) + np.cos(a) * (yy - 60)
    sel = geo["C"] & np.isfinite(cod) & (np.abs(tang) < 30)  # the linear part of the field makes the jump vary along the crack
    assert sel.sum() > 500
    # u+ at the crack of the + side field is J (its linear part vanishes at the crack to within the line position, ~1 px)
    assert np.abs(cod[sel] - 2.5).max() < 0.1
    assert np.abs(csd[sel] + 0.8).max() < 0.1
    assert np.isnan(cod[~geo["C"]]).all()


def test_strain_half_map_points_away_from_the_crack():
    ci, s = _ci(0.0)
    geo = crack.geometry(ci, Config(crack_split=True))
    half = crack.strain_half_map(geo, 15)
    yy, xx = np.mgrid[0:120, 0:120]
    right = (xx > 60 + 3) & (xx < 60 + 7) & (yy > 40) & (yy < 80)
    left = (xx < 60 - 3) & (xx > 60 - 7) & (yy > 40) & (yy < 80)
    assert (half[right] == 0).mean() > 0.9 and (half[left] == 4).mean() > 0.9
    assert (half[(np.abs(xx - 60) > 9)] == -1).all()


def test_sample_leg_nearest_and_outside():
    n = 20
    ci = dict(resid_full=np.arange(n * n, dtype=float).reshape(n, n), resid=np.zeros((n, n)), split=np.ones((n, n), np.int8))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    out = crack.sample_leg(ci, (np.full((n, n), 2.4), np.full((n, n), -1.0)), xx, yy)
    assert out["resid_full"][5, 5] == ci["resid_full"][4, 7]
    assert out["resid_full"][0, 0] == 0.0 and out["split"][0, 0] == -1  # y = -1 is outside
    assert out["resid_full"][0, n - 1] == 0.0  # x = n - 1 + 2.4 outside

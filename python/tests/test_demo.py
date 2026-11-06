"""The footing-settlement demo: unit tests of the displacement field and the truth, and end-to-end smoke tests
(small scene, a few frames, the Rust CLI)."""
import json
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from demo import figures, report, run_demo
from demo.field import (FootingField, GridField, analytic_budget, band_fwhm, basis_arrays, field_summary, prandtl_extent,
                        prepeak_heave, prepeak_parts, second_order_area, singular_discs, surface_budget, surface_profile)
from demo.footing import Renderer, Truth, make_scene
from optflow_ref import synth
from optflow_ref.strain import displacement_gradient, strain_fields

RUST_BIN = os.environ.get("OPTFLOW_BIN") or os.path.join(
    os.path.dirname(__file__), "..", "..", "rust", "target", "release", "optflow")
SCENES = ("small", "full")


def _params(size):
    """The scene's field at s/B = 0.2, the full scene's final state (the small scene's own default stops earlier, at s/B =
    0.133, because its wedge is narrower than the tracker's filters)."""
    p = make_scene(size).params
    return replace(p, s_final=0.2 * p.B)


def _soil_points(sc, n=4000, seed=0):
    """Random points of the initial soil domain (frame-0 coordinates), denser near the footing."""
    rng = np.random.default_rng(seed)
    p = sc.params
    X = p.xc + rng.normal(0.0, 0.45 * p.half_box, n).clip(-p.half_box, p.half_box)
    Y = p.y0 + rng.uniform(0.0, p.depth, n)
    return X, Y


@pytest.mark.parametrize("size", SCENES)
def test_fwd_inv_round_trip(size):
    sc = make_scene(size)
    X, Y = _soil_points(sc)
    for s in (0.0, 0.4 * sc.params.s_final, sc.params.s_final):
        f = FootingField(sc.params, s)
        x, y = f.fwd(X, Y)
        Xr, Yr = f.inv(x, y)
        assert np.abs(Xr - X).max() < 1e-6 and np.abs(Yr - Y).max() < 1e-6
        # and the other way round, from arbitrary pixel positions
        Xi, Yi = f.inv(X, Y)
        xr, yr = f.fwd(Xi, Yi)
        assert np.abs(xr - X).max() < 1e-6 and np.abs(yr - Y).max() < 1e-6


def _soil_grid(p, step=1.0, depth=None):
    """Cell centres of the initial soil domain (the container's interior)."""
    d = p.depth if depth is None else depth
    Y, X = np.mgrid[p.y0 + 0.5 * step:p.y0 + d:step, p.xc - p.half_box + 0.5 * step:p.xc + p.half_box:step]
    return X, Y


def test_time_history_splits_the_settlement_into_a_diffuse_part_and_the_mechanism():
    p = make_scene("full").params
    s = np.linspace(0.0, p.s_final, 301)
    se, sm = p.s_e(s), p.s_m(s)
    np.testing.assert_allclose(se + sm, s, atol=1e-12)
    assert se[0] == 0.0 and sm[0] == 0.0
    assert (np.diff(se) >= 0).all() and (np.diff(sm) >= 0).all()  # both only grow
    assert se.max() < p.sp  # the diffuse part saturates at s_p = 0.06 B
    k = 5  # small settlements: s_m ~ s^3 / (3 s_p^2), the bands emerge gradually
    assert sm[k] == pytest.approx(s[k] ** 3 / (3.0 * p.sp ** 2), rel=0.02)
    assert sm[-1] > 2.0 * se[-1]  # at the last frame the mechanism dominates


@pytest.mark.parametrize("size", SCENES)
def test_deformation_ranges_det_f_volumetric_strain_of_bands_and_pre_peak_part(size):
    """Check 3 and check 2.  det F >= 0.3 everywhere in the soil at every stage (so the map is invertible), also at the
    singular points; the volumetric strain det F - 1 of the shear bands and the fan is at most 0.13 (away from the footing's
    edges and the wedge's tip, where it is bounded by 0.8); outside those discs it is at least -0.15 (the cross term of the
    pre-peak shear with the band shear, -0.14 on the fan's outer band).  The pre-peak part alone, s_e U_e, has the TRUE
    det F - 1 in [-0.005, +0.0125]: its first-order divergence is <= 0 everywhere (-kappa s q under the footing, down to
    -0.0025), so the positive part is the second-order term det(grad u) of the edge band (Psi Psi'' K K'': the shear coupling),
    confined to a strip beside the footing's edges."""
    p = _params(size)
    X, Y = _soil_grid(p, 1.0 if size == "full" else 0.5)
    G = GridField(p, X, Y)
    disc = singular_discs(p, X, Y)
    for s in np.linspace(0.0, p.s_final, 7):
        _, _, (a, b, c, d) = G.at(s)
        det = (1 + a) * (1 + d) - b * c
        assert det.min() >= 0.3, (s, det.min())
    vol = det - 1.0  # the last frame
    assert vol[~disc].max() <= 0.13, vol[~disc].max()
    assert vol.max() <= 0.8, vol.max()
    assert vol[~disc].min() >= -0.15, vol[~disc].min()  # the apex disc (excluded) goes down to -0.24
    se = float(p.s_e(p.s_final))
    B = G.B
    vol_e = (1 + se * B["ex_x"]) * (1 + se * B["ey_y"]) - se * se * B["ex_y"] * B["ey_x"] - 1.0
    assert -0.005 <= vol_e.min() and vol_e.max() <= 0.0125, (vol_e.min(), vol_e.max())
    div_e = se * (B["ex_x"] + B["ey_y"])  # first order
    assert div_e.max() <= 1e-12 and div_e.min() >= -0.003, (div_e.min(), div_e.max())
    # the part above +0.002 is the edge band's second-order term: beside the footing's edges, at moderate depth
    we = p.ue_edge_hw
    pos = vol_e > 0.002
    xs, ys = np.abs(X - p.xc)[pos], (Y - p.y0)[pos]
    assert pos.mean() < 0.04 and (vol_e > 0.005).mean() < 0.025
    assert xs.min() >= p.b - 0.25 * we and xs.max() <= p.b + 1.5 * we and ys.max() <= 0.5 * p.ue_depth_k, (xs.min(), xs.max(), ys.max())
    gmax = strain_fields(*G.at(p.s_final)[2])["gmax"]
    assert 1.0 < np.percentile(gmax, 99.9) < 3.0  # engineering shear strain 1 - 1.8 in the bands (Green-Lagrange gmax)


@pytest.mark.parametrize("size", SCENES)
def test_field_summary_reports_the_ranges_of_the_field(size):
    p = _params(size)
    fs = field_summary(p, step=1.0 if size == "full" else 0.5)
    assert fs["det_f_min"] >= 0.3 and fs["vol_band_max"] <= 0.13 and fs["vol_disc_max"] <= 0.8
    assert -0.005 <= fs["vol_pre_min"] and fs["vol_pre_max"] <= 0.0125
    assert 1.0 < fs["gmax_p999"] <= fs["gmax_max"] < 3.0
    # the facts the report quotes: the edge discs dilate, the apex disc compresses (both regularisation artefacts); the
    # pre-peak part compresses by at most 0.5 % (the mean slightly), its excess over +0.002 is a small share of the soil,
    # and its largest shear strain is at the footing's edges (a few times that under the footing's middle)
    assert fs["vol_edge_max"] == fs["vol_disc_max"] > 0.3 and fs["vol_apex_min"] < -0.1
    assert fs["vol_pre_min"] < fs["vol_pre_mean"] < 0.0 and fs["vol_pre_share_below"] == 0.0
    assert 0.0 < fs["vol_pre_share_above5"] < fs["vol_pre_share_above"] < 0.04
    assert fs["div_iso_max"] < 1e-12
    assert 0.2 < fs["gmax_pre_max"] < 0.45 and 0.05 < fs["gmax_pre_middle_max"] < 0.15
    assert fs["gmax_pre_max"] > 2.5 * fs["gmax_pre_middle_max"]
    # the extremes lie where they are said to: the minimum under the footing at depth, the maximum beside the edge
    xmin, ymin = fs["vol_pre_min_at"]
    xmax, ymax = fs["vol_pre_max_at"]
    assert abs(xmin) < p.b and 0.3 * p.ue_depth_k < ymin < 0.7 * p.ue_depth_k
    assert p.b < abs(xmax) < p.b + 1.5 * p.ue_edge_hw and 0.1 * p.ue_depth_k < ymax < 0.4 * p.ue_depth_k
    # the same point as an independent evaluation of the strain there
    x, y = p.xc + fs["det_f_min_at"][0], p.y0 + fs["det_f_min_at"][1]
    a, b, c, d = FootingField(p, p.s_final).grad(np.array([x]), np.array([y]))
    assert float(((1 + a) * (1 + d) - b * c)[0]) == pytest.approx(fs["det_f_min"], abs=1e-12)


@pytest.mark.parametrize("size", SCENES)
def test_displacement_is_zero_at_walls_bottom_and_far_away(size):
    p = _params(size)
    s = p.s_final
    f = FootingField(p, s)
    t = np.linspace(0.0, 1.0, 400)
    walls = [(np.full_like(t, p.xc - p.half_box), p.y0 + t * p.depth), (np.full_like(t, p.xc + p.half_box), p.y0 + t * p.depth),
             (p.xc + (2 * t - 1) * p.half_box, np.full_like(t, p.y0 + p.depth))]
    for X, Y in walls:
        ux, uy = f.disp(X, Y)
        assert np.abs(ux).max() < 1e-9 and np.abs(uy).max() < 1e-9
    # and the strain vanishes there too (the field is C1 up to the boundary)
    for X, Y in walls:
        assert max(np.abs(g).max() for g in f.grad(X, Y)) < 1e-9
    ux, uy = f.disp(np.array([p.xc + 0.9 * p.half_box]), np.array([p.y0 + 0.9 * p.depth]))
    assert np.hypot(ux, uy).max() < 1e-3 * s


@pytest.mark.parametrize("size", SCENES)
def test_mechanism_wedge_interior_moves_by_s_m_straight_down(size):
    """Check 4.  In the wedge, at least 2 w from its sides (and outside the neighbourhoods of the edges and the apex,
    where the bands widen), U_m is exactly (0, 1) and strain-free.  The soil at the footing's base follows the footing
    exactly (s_e G_y + s_m = s) in the middle of the base, up to about b - 2 w, and lags near the edges."""
    p = _params(size)
    X, Y = _soil_grid(p, 1.0 if size == "full" else 0.5)
    B = basis_arrays(p, X, Y)
    xp, yp = X - p.xc, Y - p.y0
    d_side = (np.abs(xp) - p.b) * np.sin(np.deg2rad(p.alpha_deg)) + yp * np.cos(np.deg2rad(p.alpha_deg))  # < 0 in the wedge
    m = (d_side <= -2.0 * p.w) & ~singular_discs(p, X, Y)
    assert m.sum() > 100
    assert np.abs(B["mx"][m]).max() < 1e-9 and np.abs(B["my"][m] - 1.0).max() < 1e-9
    for name in ("mx_x", "mx_y", "my_x", "my_y"):
        assert np.abs(B[name][m]).max() < 1e-9
    # the footing's own settlement: the soil at the base follows it in the middle (|x'| <= b - 2 w) ...
    xs = np.linspace(-(p.b - 2.0 * p.w), p.b - 2.0 * p.w, 9)
    for k in (1, 3):
        s = p.s_final * k / 3
        ux, uy = FootingField(p, s).disp(p.xc + xs, np.full_like(xs, p.y0))
        assert np.abs(ux).max() < 1e-9 and np.abs(uy - s).max() < 1e-9, (size, k)
    # ... and lags at the edge (the regularised corner), by a good fraction of the settlement, and moves outward
    ux, uy = FootingField(p, p.s_final).disp(np.array([p.xc + p.b]), np.array([p.y0]))
    assert 0.2 * p.s_final < p.s_final - uy[0] < 0.7 * p.s_final and ux[0] > 0.0


@pytest.mark.parametrize("size", SCENES)
def test_mechanism_is_mirror_symmetric(size):
    p = make_scene(size).params
    X, Y = _soil_points(make_scene(size), n=500, seed=5)
    Bp, Bm = basis_arrays(p, X, Y), basis_arrays(p, 2 * p.xc - X, Y)
    for name, sign in (("mx", -1), ("my", 1), ("ex", -1), ("ey", 1)):
        np.testing.assert_allclose(Bm[name], sign * Bp[name], atol=1e-12)
        np.testing.assert_allclose(Bm[name + "_x"], -sign * Bp[name + "_x"], atol=1e-12)
        np.testing.assert_allclose(Bm[name + "_y"], sign * Bp[name + "_y"], atol=1e-12)


def _thin_band_params(size="full"):
    """The scene's field with thin bands and no regularisation: it differs from the sharp mechanism only within a
    few px of the interfaces, so the jumps can be measured a few band widths away."""
    return replace(make_scene(size).params, w=1.5, rc=1.5, edge_widen=0.0, apex_widen=0.0)


def _jump(p, pt, n, d=None):
    """U_m(pt + d n) - U_m(pt - d n), d = 3 w by default (n unit normal, pointing from the first side to the second):
    its component across and along the interface (tangent = n rotated by 90 deg), in the frame x' right / y' down."""
    d = 3.0 * p.w if d is None else d
    pts = np.array([[pt[0] + d * n[0], pt[1] + d * n[1]], [pt[0] - d * n[0], pt[1] - d * n[1]]])
    B = basis_arrays(p, p.xc + pts[:, 0], p.y0 + pts[:, 1])
    jump = np.array([B["mx"][0] - B["mx"][1], B["my"][0] - B["my"][1]])
    return float(jump @ np.array(n)), float(abs(jump @ np.array([-n[1], n[0]])))


def test_mechanism_velocity_jumps_make_the_dilation_angle_psi_with_every_interface_and_open_it():
    """Check 1.  Kinematic admissibility of U_m: across W|F (along E-A), F|stationary (the log spiral) and
    P|stationary (C-X_e) the velocity jump, measured 3 w on either side along the normal, makes the angle psi with
    the interface (+-0.5 deg) and opens it (the normal component separates the two sides: dilation); five or more
    points on each, away from the corners."""
    p = _thin_band_params()
    psi = np.deg2rad(p.psi_deg)
    a, bp, b = np.deg2rad(p.alpha_deg), np.deg2rad(p.beta_deg), p.b
    E = np.array([b, 0.0])
    rA, th_max = p.r_a, np.deg2rad(p.theta_deg)
    angles, normals = [], []
    # W|F along E-A: from the wedge into the fan
    n = (np.sin(a), np.cos(a))
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        pt = E + t * rA * np.array([-np.cos(a), np.sin(a)])
        jn, jt = _jump(p, pt, n)
        angles.append(("W|F", np.rad2deg(np.arctan2(jn, jt)))), normals.append(jn)
    # F|stationary: the log spiral r = r_A exp(theta tan psi) around E; normal = e_r + tan(psi) e_phi (outward)
    for th_deg in (10.0, 20.0, 30.0, 40.0, 50.0):
        th = np.deg2rad(th_deg)
        phi = np.pi - a - th
        R = rA * np.exp(th * np.tan(psi))
        er, ephi = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
        nv = er + np.tan(psi) * ephi
        nv /= np.linalg.norm(nv)
        jn, jt = _jump(p, E + R * er, tuple(nv))
        angles.append(("F|S", np.rad2deg(np.arctan2(jn, jt)))), normals.append(jn)
    # P|stationary along C-X_e: from the passive wedge outward
    bet = np.deg2rad(p.beta_in_deg)
    C = np.array([b + p.r_theta * np.cos(bet), p.r_theta * np.sin(bet)])
    Xe = np.array([p.xe, 0.0])
    n = (np.sin(bp), np.cos(bp))
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        jn, jt = _jump(p, C + t * (Xe - C), n)
        angles.append(("P|S", np.rad2deg(np.arctan2(jn, jt)))), normals.append(jn)
    assert len(angles) == 15
    for name, ang in angles:
        assert abs(ang - p.psi_deg) < 0.5, (name, ang)
    assert min(normals) > 0.0  # all open


def _sharp_um_right(p, xp, yp):
    """U_m of the sharp mechanism (zero-width bands) at points of the right half (x' >= 0) by zone, and the signed
    distances to the interfaces (W|F line, spiral, F|P line, outer slip line)."""
    a, bp, ps, bet = (np.deg2rad(v) for v in (p.alpha_deg, p.beta_deg, p.psi_deg, p.beta_in_deg))
    dx, dy = xp - p.b, yp
    r = np.hypot(dx, dy)
    theta = (np.pi - a) - np.arctan2(dy, dx)
    d_ea = dx * np.sin(a) + dy * np.cos(a)
    d_sp = (r - p.r_a * np.exp(theta * np.tan(ps))) * np.cos(ps)
    d_ec = dx * np.sin(bet) - dy * np.cos(bet)
    d_cx = (xp - p.b - p.r_theta * np.cos(bet)) * np.sin(bp) + (yp - p.r_theta * np.sin(bet)) * np.cos(bp)
    in_w = d_ea < 0
    in_p = (d_ec > 0) & (d_cx < 0)
    in_f = (d_ea >= 0) & (d_ec < 0) & (d_sp <= 0)
    f = p.f0 * np.exp(theta * np.tan(ps))
    ux = np.where(in_f, f * dy / r, 0.0) + np.where(in_p, p.f_theta * np.sin(bet), 0.0)
    uy = np.where(in_w, 1.0, 0.0) + np.where(in_f, -f * dx / r, 0.0) + np.where(in_p, -p.f_theta * np.cos(bet), 0.0)
    return ux, uy, (d_ea, d_sp, d_ec, d_cx), (in_w, in_f, in_p)


@pytest.mark.parametrize("size", SCENES)
def test_shipped_field_equals_the_sharp_mechanism_away_from_the_interfaces(size):
    """The jump test above uses thin bands (w = 1.5, no regularisation); this one checks the field that ships (default
    band width, corner and apex regularisation): at points farther than a band width from every interface and outside
    the edge and apex discs it is the sharp piecewise mechanism (rigid wedge, fan, passive wedge, stationary soil)."""
    p = make_scene(size).params
    rng = np.random.default_rng(7)
    n = 60000
    xp = rng.uniform(0.0, p.xe + 4 * p.w, n)
    yp = rng.uniform(0.5, p.depth_mech + 4 * p.w, n)
    ux_s, uy_s, ds, zones = _sharp_um_right(p, xp, yp)
    X, Y = p.xc + xp, p.y0 + yp
    far = np.all([np.abs(d) > 1.02 * p.w for d in ds], axis=0) & ~singular_discs(p, X, Y)
    B = basis_arrays(p, X, Y)
    assert far.sum() > 0.3 * n
    for z, name in zip(zones, ("wedge", "fan", "passive wedge")):
        assert (far & z).sum() > 500, name
    assert (far & ~zones[0] & ~zones[1] & ~zones[2]).sum() > 500  # and the stationary soil
    assert np.abs(B["mx"] - ux_s)[far].max() < 1e-6 and np.abs(B["my"] - uy_s)[far].max() < 1e-6
    # strain-free in the rigid zones, as the sharp mechanism is
    rigid = far & (zones[0] | zones[2] | ~(zones[0] | zones[1] | zones[2]))
    for name in ("mx_x", "mx_y", "my_x", "my_y"):
        assert np.abs(B[name][rigid]).max() < 1e-6


@pytest.mark.parametrize("size", SCENES)
def test_corner_regularisation_reaches_three_band_widths_from_the_edge_and_no_further(size):
    """The regularisation of the footing edge (taper of the fan within 2 r_c, widening of the W|F band within edge_radius w)
    changes the field up to max(2 r_c, edge_radius w) = 3 w from the edge, and not beyond: compared with the field without
    it (r_c -> 0, no widening)."""
    p = make_scene(size).params
    q = replace(p, rc=1e-6, edge_widen=0.0)
    reach = max(2.0 * p.rc, p.edge_radius * p.w)
    assert reach == pytest.approx(3.0 * p.w)
    rng = np.random.default_rng(3)
    n = 80000
    ang = rng.uniform(0.0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform((0.2 * reach) ** 2, (3.0 * reach) ** 2, n))
    xp, yp = p.b + rad * np.cos(ang), rad * np.sin(ang)
    keep = yp > 0.5
    xp, yp, rad = xp[keep], yp[keep], rad[keep]
    X, Y = p.xc + xp, p.y0 + yp
    A, Bq = basis_arrays(p, X, Y), basis_arrays(q, X, Y)
    diff = np.maximum(np.abs(A["mx"] - Bq["mx"]), np.abs(A["my"] - Bq["my"]))
    apex = np.hypot(xp, yp - p.ha) < (p.apex_radius + 2.0) * p.w  # the apex widening is the same in both fields, but keep clear of it
    beyond = (rad > reach * 1.0001) & ~apex
    assert beyond.sum() > 1000 and diff[beyond].max() < 1e-9, diff[beyond].max()
    between = (rad > 2.0 * p.rc * 1.05) & (rad < reach * 0.98) & ~apex
    assert between.sum() > 500 and diff[between].max() > 0.01, diff[between].max()  # still changed between 2 r_c and 3 w
    inside = (rad < 0.8 * p.rc) & ~apex
    assert diff[inside].max() > 0.1


def test_fan_to_passive_wedge_has_no_jump_and_the_closed_forms_hold():
    """F|P along E-C is continuous (f(Theta) c(Theta) = U_P) and the geometric closed forms are consistent."""
    p = _thin_band_params()
    bet = np.deg2rad(p.beta_in_deg)
    b = p.b
    C = np.array([b + p.r_theta * np.cos(bet), p.r_theta * np.sin(bet)])
    E = np.array([b, 0.0])
    # sample across E-C: normal (sin beta, -cos beta), 3 w on either side
    n = np.array([np.sin(bet), -np.cos(bet)])
    for t in (0.2, 0.5, 0.8):
        pt = E + t * (C - E)
        # the fan's velocity varies with the angle, so the difference across E-C grows linearly with the distance:
        # extrapolate it to the interface from two distances
        (n1, t1), (n2, t2) = _jump(p, pt, tuple(n), 1.5 * p.w), _jump(p, pt, tuple(n), 3.0 * p.w)
        assert abs(2 * n1 - n2) < 0.01 and abs(2 * t1 - t2) < 0.01, (t, 2 * n1 - n2, 2 * t1 - t2)
    ps, a = np.deg2rad(p.psi_deg), np.deg2rad(p.alpha_deg)
    assert p.beta_in_deg == pytest.approx(90.0 - p.beta_deg - p.psi_deg)
    assert p.theta_deg == pytest.approx(180.0 - p.alpha_deg - p.beta_in_deg)
    assert p.heave_flux == pytest.approx(p.b + p.dilation)  # b + D = |X_e - E| f(Theta) cos(beta)


@pytest.mark.parametrize("size", SCENES)
def test_exit_distance_and_depth_match_the_closed_forms(size):
    """Check 5.  The mechanism's exit at the surface (the half-height of the passive wedge's heave) and its depth (the
    half-way point of the fan's velocity across the log spiral's deepest point) against the closed forms."""
    p = make_scene(size).params
    # exit: U_m,y along the surface across X_e; the passive wedge's plateau is -f(Theta) cos(beta)
    x = np.arange(p.xe - 3 * p.w, p.xe + 3 * p.w, 0.01)
    B = basis_arrays(p, p.xc + x, np.full_like(x, p.y0))
    plateau = -p.f_theta * np.cos(np.deg2rad(p.beta_in_deg))
    half = x[np.argmin(np.abs(B["my"] - 0.5 * plateau))]
    assert abs(half - p.xe) < 0.5, (half, p.xe)
    # depth: |U_m| along the vertical line through the spiral's deepest point
    ps, a = np.deg2rad(p.psi_deg), np.deg2rad(p.alpha_deg)
    th = np.clip(np.pi / 2 - a + ps, 0.0, np.deg2rad(p.theta_deg))
    phi = np.pi - a - th
    R = p.r_a * np.exp(th * np.tan(ps))
    xd = p.b + R * np.cos(phi)
    y = np.arange(p.depth_mech - 3 * p.w, p.depth_mech + 3 * p.w, 0.01)
    B = basis_arrays(p, np.full_like(y, p.xc + xd), p.y0 + y)
    speed = np.hypot(B["mx"], B["my"])
    inside = p.f0 * np.exp(th * np.tan(ps))
    ydeep = y[np.argmin(np.abs(speed - 0.5 * inside))]
    assert abs(ydeep - p.depth_mech) < 0.5, (ydeep, p.depth_mech)
    # the mechanism stays clear of the container: the exit is at least 120 px (full size) from the wall, the bottom
    # at least 1 B below the mechanism
    assert p.half_box - p.xe >= 120.0 * (p.B / 150.0) and p.depth - p.depth_mech >= p.B


@pytest.mark.parametrize("size", SCENES)
def test_volume_budget(size):
    """Check 2 and check 5.  The volumetric strain integrates to the closed-form budget (pre-peak compression -kappa B s_e
    plus 2 D s_m), the finite-displacement part is the surface boundary integral, and the exact area change is that of the
    deformed surface (two independent computations: the grid sum of det F - 1, and the area under the deformed ground
    surface).  The heave bookkeeping net heave = B s + net_dv - overlap follows from the definitions (it is not a validation);
    the substantive checks are that the overlap is small, that the trough beside the footing is small, and that the soil does
    not lose area (the dilation of the bands beats the small pre-peak compression)."""
    p = _params(size)
    X, Y = _soil_grid(p, 1.0 if size == "full" else 0.5)
    dA = (1.0 if size == "full" else 0.5) ** 2
    G = GridField(p, X, Y)
    ux, uy, (a, b, c, d) = G.at(p.s_final)
    lin = float(((a + d) * dA).sum())
    exact = float((((1 + a) * (1 + d) - b * c - 1.0) * dA).sum())
    bud = analytic_budget(p, p.s_final)
    se = float(p.s_e(p.s_final))
    assert bud["pre_peak"] == pytest.approx(-p.ue_kappa * p.B * se, rel=1e-12)
    # first order: the pre-peak part alone integrates to -kappa B s_e (U_iso has no divergence, U_c takes the area g K_c(0)),
    # and the whole to the closed form, to 3 % of the dilation term
    B = G.B
    lin_e = float((se * (B["ex_x"] + B["ey_y"]) * dA).sum())
    assert lin_e == pytest.approx(bud["pre_peak"], rel=1e-2)
    assert abs(lin - bud["total"]) < 0.03 * bud["dilation"], (lin, bud)
    # the finite-displacement part is a boundary integral of the surface's displacement
    so = second_order_area(p, p.s_final)
    assert exact - lin == pytest.approx(so, rel=2e-2, abs=0.002 * abs(bud["dilation"]))
    # and the exact area change is the area under the deformed surface
    sb = surface_budget(p, p.s_final)
    assert exact == pytest.approx(sb["net_dv"], rel=3e-3, abs=0.01 * abs(bud["dilation"]))
    # the soil does not lose area at the last frame (check 4): the dilation beats the compression and the second-order term
    assert sb["net_dv"] > 0.0 and exact > 0.0
    # bookkeeping of the heave: an identity of the definitions
    assert sb["net_heave"] == pytest.approx(sb["intrusion_area"] + sb["net_dv"] - sb["contact_overlap"], rel=1e-9, abs=1e-6)
    assert sb["heave_area"] - sb["trough_area"] == pytest.approx(sb["net_heave"], rel=1e-9)
    # the substantive parts: the soil overlapping the footing at its edges is small, the trough beside it (soil dragged down
    # next to the edge, the regularised corner) is small, and the heave is at least what the footing pushed in
    assert 0.0 <= sb["contact_overlap"] < 0.08 * sb["intrusion_area"]
    assert sb["trough_area"] < 0.12 * sb["intrusion_area"]
    assert sb["heave_max"] > 0.15 * p.s_final * 0.5
    assert sb["net_heave"] > 0.9 * sb["intrusion_area"] and sb["heave_area"] > sb["net_heave"] > 0.0
    assert -bud["pre_peak"] < 0.1 * sb["intrusion_area"]  # nearly isochoric: the pre-peak compression is a small part of the intrusion
    # the integrals do not depend on the quadrature step (the footing's edges are nodes of the resampled profile)
    fine = surface_budget(p, p.s_final, step=0.02)
    for key in ("net_heave", "heave_area", "trough_area", "contact_overlap", "net_dv"):
        assert sb[key] == pytest.approx(fine[key], rel=2e-3, abs=0.5), key


@pytest.mark.parametrize("size", SCENES)
def test_prepeak_flow_is_a_stream_function_flow_divergence_free_to_first_order(size):
    """Check 1.  U_iso is the flow of the stream function Phi = Psi(x') K(y'), u_x = Phi_y, u_y = -Phi_x: its first-order
    divergence is zero to round-off on a grid (the analytic gradients); Psi is the integral of -u_y at the surface (an
    independent quadrature gives u_x = Psi K'); on the footing's base u_y = 1 and u_x = 0 (rough base) for |x'| <= x_s, u_x = 0
    along the whole surface, the soil under the footing spreads outward at depth, the flow is zero beyond b + L_h and below L_k."""
    p = _params(size)
    X, Y = _soil_grid(p, 1.0 if size == "full" else 0.5)
    P = prepeak_parts(p, X, Y)
    assert np.abs(P["isox_x"] + P["isoy_y"]).max() < 1e-12
    assert np.abs(P["isox_x"]).max() > 1e-3  # (not trivially zero)
    # the base and the surface
    xs = np.linspace(-p.ue_band_start, p.ue_band_start, 15)
    S = prepeak_parts(p, p.xc + xs, np.full_like(xs, p.y0))
    assert np.abs(S["isoy"] - 1.0).max() < 1e-12  # (u_y = -Psi' K = 1 at the base)
    assert np.abs(S["isox"]).max() < 1e-12 and np.abs(S["cy"] - 1.0).max() < 1e-12
    xw = np.linspace(-p.half_box, p.half_box, 801)
    assert np.abs(prepeak_parts(p, p.xc + xw, np.full_like(xw, p.y0))["isox"]).max() < 1e-12  # K'(0) = 0
    # spreading outward under the footing at depth: u_x = -x' K' >= 0 for x' > 0 (and odd)
    y_d = p.y0 + 0.2 * p.ue_depth_k
    xr = np.linspace(0.05 * p.b, 0.9 * p.ue_band_start, 9)
    Pr, Pl = prepeak_parts(p, p.xc + xr, np.full_like(xr, y_d)), prepeak_parts(p, p.xc - xr, np.full_like(xr, y_d))
    assert (Pr["isox"] > 0).all() and np.abs(Pl["isox"] + Pr["isox"]).max() < 1e-12 and np.abs(Pl["isoy"] - Pr["isoy"]).max() < 1e-12
    # Psi from an independent quadrature of the surface profile: Psi(x) = int_0^x Psi' = -int_0^x u_y(x', 0) dx'; then
    # u_x(x, y) = Psi(x) K'(y) with K = smootherstep(1 - y / L_k) (its derivative at the depth y_d)
    xq = np.linspace(0.0, p.b + p.ue_lobe + 20.0, 20001)
    uy0 = prepeak_parts(p, p.xc + xq, np.full_like(xq, p.y0))["isoy"]
    psi = -np.concatenate([[0.0], np.cumsum(0.5 * (uy0[1:] + uy0[:-1]) * np.diff(xq))])
    c = 1.0 - 0.2
    kprime = -30.0 * c * c * (1.0 - c) ** 2 / p.ue_depth_k
    xt = np.array([0.2 * p.b, 0.9 * p.b, p.b, p.b + 0.5 * p.ue_edge_hw, p.b + p.ue_edge_hw, p.b + 0.5 * p.B, p.b + 0.9 * p.ue_lobe])
    uxt = prepeak_parts(p, p.xc + xt, np.full_like(xt, y_d))["isox"]
    assert np.abs(uxt - np.interp(xt, xq, psi) * kprime).max() < 1e-4 * p.b * abs(kprime)
    assert abs(psi[-1]) < 1e-3 * p.b  # the loop closes: Psi = 0 beyond the lobe
    # the lobe beside the footing is a lift (u_y < 0 for |x'| above the band), the flow vanishes beyond b + L_h and below L_k
    xl = np.linspace(p.ue_band_start + 2.0 * p.ue_edge_hw + 1e-6, p.b + 0.95 * p.ue_lobe, 50)
    assert (prepeak_parts(p, p.xc + xl, np.full_like(xl, p.y0))["isoy"] < 0).all()
    zero = (np.abs(X - p.xc) >= p.b + p.ue_lobe) | (Y - p.y0 >= max(p.ue_depth_k, p.ue_comp_depth))
    for name in ("isox", "isoy", "cy"):
        assert np.abs(P[name][zero]).max() < 1e-12, name


@pytest.mark.parametrize("size", SCENES)
def test_prepeak_heave_area_equals_one_minus_kappa_times_the_intrusion(size):
    """Check 3.  The pre-peak part alone, at s = s_p (s_e = s_p tanh 1): the net area lifted beside the footing (|x'| > b,
    both sides) equals (1 - kappa) B s_e (first order) within 2 % (what the footing pushes in minus the share U_c absorbs;
    the small remainder is the part of the downward flux that sits in the edge band and in U_c's edge); the pre-peak surface
    has no horizontal displacement, so the lift in the deformed coordinates is the same; and over the whole surface the net
    area is exactly -kappa B s_e."""
    p = _params(size)
    se = float(p.s_e(p.sp))
    x = np.unique(np.round(np.concatenate([np.arange(-p.half_box, p.half_box, 0.02), [-p.b, p.b]]), 9))
    B = basis_arrays(p, p.xc + x, np.full_like(x, p.y0))
    assert np.abs(B["ex"]).max() < 1e-12
    h = -se * B["ey"]
    out = np.abs(x) >= p.b - 1e-9
    seg = out[1:] & out[:-1]
    net_out = float(np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(x) * seg))
    assert net_out == pytest.approx((1.0 - p.ue_kappa) * p.B * se, rel=0.02)
    assert net_out < (1.0 - p.ue_kappa) * p.B * se  # (the edge band's share under the footing makes it a little smaller)
    net_all = float(np.trapezoid(h, x))
    assert net_all == pytest.approx(-p.ue_kappa * p.B * se, rel=1e-3)
    ph = prepeak_heave(p)
    assert ph["net_outside"] * se == pytest.approx(net_out, rel=1e-3) and ph["net_total"] * se == pytest.approx(net_all, rel=1e-3)
    assert ph["lag_inside"] < 0.02 * p.B  # the soil under the footing keeps up with it: the lag area is < 2 % of B s_e


@pytest.mark.parametrize("size", SCENES)
def test_final_state_has_no_deep_trough_net_area_gain_and_quiet_walls(size):
    """Check 4, at s_final: (a) the ground surface rises above the initial level within 0.3 B of the footing's edge, and the
    trough beside the footing is shallow and narrow (the soil dragged down at the edge, not a bowl: the surface returns to
    the initial level 0.16 B from the edge, where the first version's bowl needed 1 B); (b) the soil's net area change is >= 0;
    (c) no compression patch at the container's walls or bottom: |det F - 1| < 1e-3 within 50 px (full size; scaled) of them."""
    p = _params(size)
    Xm, xd, h = surface_profile(p, p.s_final, step=0.05)
    beyond = xd > p.b
    near = beyond & (xd < p.b + 0.3 * p.B)
    assert h[near].max() > 0.0  # (a)
    assert (xd[beyond & (h > 0.0)].min() - p.b) < 0.3 * p.B
    assert np.interp(p.b, xd, h) > -0.7 * p.s_final  # the edge sits above the footing's base level -s ...
    assert h[xd > p.b + 0.3 * p.B].min() > -0.1 * p.s_final  # ... and nowhere beyond 0.3 B is the surface more than 0.1 s below the initial level
    assert surface_budget(p, p.s_final)["net_dv"] > 0.0  # (b)
    step = 1.0 if size == "full" else 0.5
    X, Y = _soil_grid(p, step)
    _, _, (a, b, c, d) = GridField(p, X, Y).at(p.s_final)
    det1 = (1 + a) * (1 + d) - b * c - 1.0
    m = 50.0 * p.B / 150.0
    wall = (np.abs(X - p.xc) > p.half_box - m) | (Y > p.y0 + p.depth - m)
    assert wall.sum() > 1000 and np.abs(det1[wall]).max() < 1e-3, np.abs(det1[wall]).max()  # (c)
    # and the displacement itself vanishes at the walls (see test_displacement_is_zero_at_walls_bottom_and_far_away)


@pytest.mark.parametrize("size", SCENES)
def test_report_heave_is_the_surface_heave_of_the_field(size):
    """Check 7.  The heave that summary.md reports (report.heave_facts and the physics table) is the highest point of the
    field's deformed ground surface outside the footing, found here by brute force on a fine grid, with the extent of the
    plateau it sits on."""
    p = _params(size)
    X = np.arange(p.xc - p.half_box, p.xc + p.half_box, 0.01)
    ux, uy = FootingField(p, p.s_final).disp(X, np.full_like(X, p.y0))
    xd, h = X + ux - p.xc, -uy
    out = np.abs(xd) > p.b
    hv = report.heave_facts(p)
    assert hv["heave_max"] == pytest.approx(float(h[out].max()), abs=2e-3)
    assert abs(hv["heave_max_x"]) == pytest.approx(abs(float(xd[out][np.argmax(h[out])])), abs=1.0)
    # the top: within 1 % of the maximum, containing the maximum: a ridge beside the footing (the pre-peak lobe on the passive
    # wedge's lift), inside the pre-peak lobe and well inside the mechanism's exit; further out the surface is the passive
    # wedge's plateau (U_m alone, beyond the pre-peak lobe), lower than the ridge
    lo, hi = hv["heave_plateau"]
    top = (h >= 0.99 * h[out].max()) & out & (xd > 0)
    assert lo == pytest.approx(float(xd[top].min()), abs=0.1) and hi == pytest.approx(float(xd[top].max()), abs=0.1)
    assert lo <= abs(hv["heave_max_x"]) <= hi and hi - lo > 0.05 * p.B
    assert p.b < lo and hi < p.b + p.ue_lobe < p.xe
    assert 0.3 * hv["heave_max"] < hv["h_wedge"] < 0.8 * hv["heave_max"]
    row = dict((r[0], r[1]) for r in report.physics_table(p))["隆起と体積収支"]
    assert f"最大 {float(h[out].max()):.1f} px 隆起" in row
    assert f"B·s = {p.B * p.s_final:.0f} px²" in row
    assert f"x' = {lo:.0f}〜{hi:.0f} px" in row
    assert hv["heave_max"] > 0.2 * p.s_final * 0.5  # not the 1.7 px of the first version's wrong figure (about 6 % of s)
    # the table has all its rows, filled with numbers of the field; no unfilled placeholders, no unresolved citation marks
    rows = report.physics_table(p)
    assert [r[0] for r in rows][:2] == ["破壊機構と角度", "広がり"] and len(rows) >= 8
    for item, mine, lit in rows:
        assert "{" not in mine + lit and "nan" not in (mine + lit).lower() and "TODO" not in mine + lit, item
        assert len(lit) > 60  # every row has a literature cell
    # every author-year cited in the text is in the reference list, and every reference is cited in the text
    import re
    text = " ".join(m + l for _, m, l in rows) + " " + report.sp_sensitivity_text(p) + " " + report.deviation_text(p)
    cited = set(re.findall(r"([A-Z][A-Za-z\u00c0-\u017f\-]+(?: et al\.| & [A-Z][A-Za-z\u00c0-\u017f\-]+)?) ((?:19|20)\d\d)", text))
    listed = set()
    for key, _ in report.REFERENCES:
        m = re.fullmatch(r"(.+?) \(((?:19|20)\d\d)\)", key)
        assert m, key
        listed.add((m.group(1), m.group(2)))
    assert cited - listed == set(), f"cited but not in the reference list: {cited - listed}"
    assert listed - cited == set(), f"in the reference list but never cited: {listed - cited}"
    assert "TODO" not in report.sp_sensitivity_text(p)


@pytest.mark.parametrize("size", SCENES)
def test_report_numbers_that_the_table_quotes_follow_from_the_field(size):
    """The Prandtl extents (own computation) are the closed forms of Params for Prandtl's angles; the band width is the
    measured half-maximum width of gmax (a little below the linear estimate 1.08 w); the mechanism carries 24 % of the
    settlement at s = s_p and half of it from s/B about 0.12; a smaller diffuse part gives less compression and more heave."""
    p = _params(size)
    ex, dp, ap = prandtl_extent(45.0)
    assert (ex, dp, ap) == pytest.approx((12.11, 3.00, 1.207), abs=0.01)
    assert prandtl_extent(30.0) == pytest.approx((4.79, 1.585, 0.866), abs=0.01)
    assert p.xe / p.B < prandtl_extent(30.0)[0] and p.depth_mech / p.B < prandtl_extent(30.0)[1]  # well inside the classical extent
    fw = band_fwhm(p)
    assert 0.85 * 1.08 * p.w < fw < 1.03 * 1.08 * p.w, (fw, 1.08 * p.w)
    assert p.s_m(p.sp) / p.sp == pytest.approx(1.0 - np.tanh(1.0))
    assert 0.10 < report.half_mechanism_settlement(p) / p.B < 0.13
    base = report.heave_facts(p)
    sens = report.sp_sensitivity(p)
    assert [r["frac"] for r in sens] == [0.04, 0.03]
    I = base["intrusion_area"]
    prev_share, prev_min = base["heave_area"] / I, field_summary(p, step=1.0)["vol_pre_min"]
    for r in sens:
        assert r["heave_share"] > prev_share and r["vol_pre_min"] > prev_min  # more heave, less compression
        assert r["det_f_min"] > 0.3 and r["gmax_p999"] > field_summary(p, step=1.0)["gmax_p999"]  # but stronger bands
        prev_share, prev_min = r["heave_share"], r["vol_pre_min"]


@pytest.mark.parametrize("size", SCENES)
def test_displacement_history_is_monotone_where_it_should_be(size):
    sc = make_scene(size)
    p = sc.params
    Y, X = np.mgrid[p.y0:p.y0 + p.depth + 0.1:3.0, p.xc - p.half_box:p.xc + p.half_box + 0.1:3.0]
    G = GridField(p, X, Y)
    rigid = (np.abs(G.B["mx"]) < 1e-9) & (np.abs(G.B["my"] - 1.0) < 1e-9)
    prev_uy = None
    for s in np.linspace(0.0, p.s_final, 13):
        ux, uy, _ = G.at(s)
        if prev_uy is not None:
            assert (uy[rigid] >= prev_uy[rigid] - 1e-9).all()  # the wedge only goes down
        prev_uy = uy
    assert np.abs(np.hypot(ux, uy)).max() == pytest.approx(p.s_final, rel=1e-9)  # the footing's own displacement is the largest
    assert uy.min() < -0.1 * p.s_final * 0.5  # the free surface beside the footing is lifted (and not only by the diffuse part)


@pytest.mark.parametrize("size", SCENES)
def test_gradient_matches_finite_differences(size):
    sc = make_scene(size)
    p = sc.params
    f = FootingField(p, 0.8 * p.s_final)
    X, Y = _soil_points(sc, n=1500, seed=3)
    h = 1e-4
    ux = lambda X, Y: f.disp(X, Y)[0]
    uy = lambda X, Y: f.disp(X, Y)[1]
    num = ((ux(X + h, Y) - ux(X - h, Y)) / (2 * h), (ux(X, Y + h) - ux(X, Y - h)) / (2 * h),
           (uy(X + h, Y) - uy(X - h, Y)) / (2 * h), (uy(X, Y + h) - uy(X, Y - h)) / (2 * h))
    for a, b in zip(f.grad(X, Y), num):
        np.testing.assert_allclose(a, b, atol=2e-6)


def test_truth_strain_follows_the_trackers_definitions():
    """E = (F^T F - I) / 2 with F = I + grad u, gmax = difference of its principal values, vol = det F - 1: the
    truth of the demo (through strain.strain_fields) against an independent matrix computation."""
    sc = make_scene("small")
    tr = Truth(sc)
    t = tr.frame(sc.params.s_final)
    G = tr.grid
    _, _, (a, b, c, d) = G.at(sc.params.s_final)
    F = np.stack([np.stack([1 + a, b], -1), np.stack([c, 1 + d], -1)], -2)  # (..., 2, 2), F = [[1+dudX, dudY], [dvdX, 1+dvdY]]
    E = 0.5 * (np.swapaxes(F, -1, -2) @ F - np.eye(2))
    ev = np.linalg.eigvalsh(E)
    np.testing.assert_allclose(t["Exx"], E[..., 0, 0], atol=1e-12)
    np.testing.assert_allclose(t["Eyy"], E[..., 1, 1], atol=1e-12)
    np.testing.assert_allclose(t["Exy"], E[..., 0, 1], atol=1e-12)
    np.testing.assert_allclose(t["gmax"], ev[..., 1] - ev[..., 0], atol=1e-10)
    np.testing.assert_allclose(t["vol"], np.linalg.det(F) - 1.0, atol=1e-12)


def test_truth_strain_on_an_affine_special_case():
    """The same definitions on a synth.py affine field (a known closed form): the grad() convention of the footing
    field is that of synth's fields, so strain_fields gives the same numbers for both."""
    f = synth.uniform_strain(0.05, -0.02, (10.0, 10.0), exy=0.03)
    X, Y = np.array([3.0, 20.0]), np.array([4.0, 8.0])
    st = strain_fields(*f.grad(X, Y))
    F = f.A
    E = 0.5 * (F.T @ F - np.eye(2))
    np.testing.assert_allclose(st["Exx"], E[0, 0])
    np.testing.assert_allclose(st["Exy"], E[0, 1])
    np.testing.assert_allclose(st["vol"], np.linalg.det(F) - 1.0)
    ev = np.linalg.eigvalsh(E)
    np.testing.assert_allclose(st["gmax"], ev[1] - ev[0])


def test_analytic_gradient_agrees_with_the_trackers_numerical_one_in_smooth_regions():
    """strain.displacement_gradient (local quadratic fit) of the true displacement field reproduces the analytic
    gradient away from the shear bands, in the same order (du/dX, du/dY, dv/dX, dv/dY).  The tolerance 3e-3 is the truncation
    error of the quadratic fit (window 15) where the pre-peak flow varies on the scale of L_k, not an error of the analytic
    gradient (see test_gradient_matches_finite_differences)."""
    sc = make_scene("full")
    p = sc.params
    Y, X = np.mgrid[p.y0 + 200:p.y0 + 330, p.xc + 100:p.xc + 260].astype(np.float64)  # below the mechanism
    G = GridField(p, X, Y)
    ux, uy, gr = G.at(p.s_final)
    num = displacement_gradient(ux, uy, 15)
    sl = np.s_[10:-10, 10:-10]
    for a, b in zip(gr, num):
        np.testing.assert_allclose(a[sl], b[sl], atol=3e-3)


def test_renderer_inverse_map_is_accurate():
    """The renderer solves the inverse map at every pixel (Newton, exact Jacobian, warm start from the previous frame): the
    material point it returns maps forward onto the pixel, from a cold start and from a warm start."""
    for size in SCENES:
        sc = make_scene(size)
        r = Renderer(sc)
        p = sc.params
        soil = r.py >= p.y0 + 1.0  # above the surface there is no soil (and the extended field is not used)
        for s in (0.5 * p.s_final, p.s_final):  # the second call starts from the first one's solution
            X, Y = r.inverse(s)
            x, y = FootingField(p, s).fwd(X, Y)
            assert np.abs(x - r.px)[soil].max() < 1e-7 and np.abs(y - r.py)[soil].max() < 1e-7, (size, s)
        assert not hasattr(r, "COARSE")


def test_invert_displacement_of_the_animation():
    """figures.invert_displacement recovers the material coordinates from a displacement given on the frame-0 grid."""
    sc = make_scene("small")
    p = sc.params
    s = 0.6 * p.s_final
    f = FootingField(p, s)
    yy, xx = np.mgrid[0:sc.height, 0:sc.width].astype(np.float64)
    ux, uy = f.disp(xx, yy)
    Y, X = np.mgrid[int(p.y0 + 10):int(p.y0 + 70), int(p.xc - 90):int(p.xc + 90)].astype(np.float64)
    Xe, Ye = f.inv(X, Y)
    Xi, Yi = figures.invert_displacement(ux, uy, X, Y, tol=1e-4)
    # the bilinear samples of a smooth field are accurate to ~1e-2 px
    assert np.nanmax(np.abs(Xi - Xe)) < 0.05 and np.nanmax(np.abs(Yi - Ye)) < 0.05


def test_scene_roi_lies_in_the_soil_below_the_surface():
    for size in SCENES:
        sc = make_scene(size)
        poly = np.array(sc.roi_polygon())
        p = sc.params
        assert poly[:, 0].min() > sc.box_l and poly[:, 0].max() < sc.box_r
        assert poly[:, 1].min() > p.y0 + 2 and poly[:, 1].max() < sc.y_bottom
        x0, y0, x1, y1 = sc.roi_bbox()
        assert 0 <= x0 < x1 <= sc.width and 0 <= y0 < y1 <= sc.height


def test_prepare_out_replaces_only_its_own_files(tmp_path):
    out = tmp_path / "o"
    (out / "track" / "frame_00001").mkdir(parents=True)
    (out / "track" / "config.json").write_text("{}")
    for name in ("summary.md", "overview.png", "footing.mp4"):
        (out / name).write_text("old")
    (out / "notes.txt").write_text("mine")
    (out / "mydir").mkdir()
    run_demo.prepare_out(str(out))
    assert (out / "notes.txt").read_text() == "mine" and (out / "mydir").is_dir()
    assert not (out / "summary.md").exists() and not (out / "track").exists() and not (out / "footing.mp4").exists()
    # a "track" directory that is not this demo's output is refused, not deleted
    (out / "track").mkdir()
    (out / "track" / "precious.dat").write_text("x")
    with pytest.raises(ValueError, match="does not look like"):
        run_demo.prepare_out(str(out))
    assert (out / "track" / "precious.dat").exists()
    with pytest.raises(ValueError):
        run_demo.prepare_out(run_demo.REPO)


def test_default_output_directory_is_demo_out_at_the_repository_root():
    assert run_demo.parse_args([]).out == os.path.join(run_demo.REPO, "demo_out")
    with open(os.path.join(run_demo.REPO, ".gitignore")) as f:
        assert "demo_out/" in f.read().split()


# ---------------------------------------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN),
                    reason="the Rust optflow binary is not built (cd rust && cargo build --release -p optflow-cli)")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_demo_end_to_end_smoke(tmp_path):
    """Small scene, 5 frames, 3 px settlement: video -> Rust CLI with the ROI and the automatic crop -> comparison with
    the truth -> figures, animation, summary.  Accuracy targets are those of a quiet, small run."""
    cv2 = pytest.importorskip("cv2")
    out = tmp_path / "demo"
    m = run_demo.run(["--size", "small", "--frames", "5", "--settlement", "3", "--out", str(out), "--bin", RUST_BIN,
                      "--no-build", "--quiet"])
    for name in ("footing.mp4", "roi.json", "truth.npz", "overview.png", "curves.png", "strain_animation.mp4", "summary.md",
                 "metrics.json", "per_frame.csv", "cli_stdout.txt", "track/config.json", "track/crop.json"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert (out / "cli_stderr.txt").read_text() == ""  # no warning of the tracker
    assert not any(n.startswith("frame_") for n in os.listdir(out / "track"))  # the fields are deleted by default
    S = m["summary"]
    assert S["frames"] == 5
    # (0.063 with the pre-peak edge band; 0.045-0.048 for seeds 1-3 with the first version's wide bowl.  In this 0.3-scale scene
    # the edge band is only 12 px wide, as narrow as the tracker's filters, so the shear pixels next to the footing's edges
    # carry most of the error: band RMS 0.14 px against 0.05 px outside the bands at the last frame)
    assert S["disp_rms_worst"] < 0.08, S
    assert S["valid_frac_eff_min"] > 0.9, S  # of the ROI beyond the tracker's guard distance from its edge
    assert S["valid_frac_min"] > 0.5, S
    # the estimate is the median u_y over the wedge pixels: compared with the true median over the same pixels (the soil in
    # the wedge settles a little less than the footing: the pre-peak flow decays with depth).  That soil is strained by
    # up to about 4 % vertically (isochorically: it spreads sideways), and the tracker's estimate lags the truth by up to 4 %
    # of the settlement in this tiny wedge
    assert abs(S["settlement_est_final"] - S["settlement_ref_final"]) < 0.15, S
    assert 0.0 < S["settlement_true_final"] - S["settlement_ref_final"] < 0.5 * S["settlement_true_final"], S
    assert S["settlement_err_max_abs"] < 0.15, S
    assert not m["tracker"]["capture_warning"] and not m["tracker"]["crop_warning"], m["tracker"]["stderr"]
    assert m["tracker"]["wall_seconds"] > 0 and len(m["frames"]) == 4
    # the video has all frames, the animation too (side by side: twice the cropped width)
    for name, n in (("footing.mp4", 5), ("strain_animation.mp4", 5)):
        cap = cv2.VideoCapture(str(out / name))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n, name
        cap.release()
    assert json.loads((out / "roi.json").read_text())["polygon"]
    text = (out / "summary.md").read_text()
    assert "沈下" in text and "再実行" in text and "nan" not in text.lower()
    with np.load(out / "truth.npz") as z:
        assert z["settlement"].shape == (5,) and z["settlement"][-1] == pytest.approx(3.0)


def _strict_json(path):
    """metrics.json parsed as strict JSON (no NaN / Infinity literals)."""
    def bad(c):
        raise ValueError(f"non-finite number {c} in {path}")
    with open(path) as f:
        return json.load(f, parse_constant=bad)


def _run_small(tmp_path, *extra):
    out = tmp_path / "demo"
    m = run_demo.run(["--size", "small", *extra, "--out", str(out), "--bin", RUST_BIN, "--no-build", "--quiet"])
    return out, m


@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN), reason="the Rust optflow binary is not built")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_documented_small_default_runs_clean(tmp_path):
    """`python -m demo.run_demo --size small` (6 frames, 6 px): the wedge stays valid to the last frame, so every
    number of summary.md and metrics.json is finite; no tracker warning."""
    pytest.importorskip("cv2")
    out, m = _run_small(tmp_path)
    S = m["summary"]
    assert S["frames"] == 6 and S["settlement_true_final"] == pytest.approx(6.0)
    assert S["rigid_valid_frac_final"] > 0.9 and np.isfinite(S["settlement_est_final"]), S
    assert abs(S["settlement_est_final"] - S["settlement_ref_final"]) < 0.5, S  # the tiny wedge: the tracker lags the strained soil by a few %
    assert (out / "cli_stderr.txt").read_text() == ""
    text = (out / "summary.md").read_text()
    assert "nan" not in text.lower() and "TODO" not in text and "参考文献" in text
    for key, _ in report.REFERENCES:  # the reference list is printed in full
        assert key.split(" (")[0].split(",")[0].split(" ")[0] in text
    _strict_json(out / "metrics.json")


@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN), reason="the Rust optflow binary is not built")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_small_run_that_loses_the_wedge_writes_clean_outputs(tmp_path):
    """A settlement for which the strong bands swallow the small scene's whole wedge at the last frame (9 px, s/B = 0.2):
    the estimate is lost there; summary.md must not print 'nan', metrics.json stays strict JSON (null), and the headline
    numbers fall back to the last frame that has an estimate."""
    pytest.importorskip("cv2")
    out, m = _run_small(tmp_path, "--settlement", "9", "--no-animation")
    S = m["summary"]
    text = (out / "summary.md").read_text()
    assert "nan" not in text.lower()
    j = _strict_json(out / "metrics.json")
    if S["rigid_valid_frac_final"] == 0.0:  # (as measured; the assertions below are about that case)
        assert not np.isfinite(S["settlement_est_final"]) and j["summary"]["settlement_est_final"] is None
        assert S["settlement_last_valid_frame"] is not None and np.isfinite(S["settlement_est_last_valid"])
        assert "最終フレームでは楔の画素がすべて無効" in text


def test_summary_falls_back_to_the_last_frame_with_an_estimate():
    from types import SimpleNamespace

    from demo import evaluate as ev_mod

    def row(k, est, ref=1.0):
        return {"k": k, "settlement_true": float(k), "settlement_ref": ref, "settlement_est": est, "settlement_err": est - ref,
                "disp_rms": 0.01 * k, "disp_p99": 0.1, "disp_max": 0.2, "valid_frac": 0.9, "valid_frac_eff": 0.95, "ref_update": False,
                "gmax_rms_band": 0.1, "gmax_rms_out": 0.01, "gmax_peak_est": 0.5, "gmax_peak_true": 1.0, "gmax_peak_true_all": 1.5,
                "rigid_valid_frac": 0.0 if np.isnan(est) else 1.0}

    ev = SimpleNamespace(rows=[row(1, 0.9), row(2, 1.8, 2.0), row(3, float("nan"), float("nan"))], band=np.ones((2, 2), bool),
                         interior=np.ones((3, 3), bool), roi=np.ones((4, 4), bool))
    S = ev_mod.summarize(ev)
    assert np.isnan(S["settlement_est_final"]) and S["settlement_last_valid_frame"] == 2
    assert S["settlement_est_last_valid"] == 1.8 and S["settlement_ref_last_valid"] == 2.0
    assert S["settlement_err_max_abs"] == pytest.approx(0.2) and S["rigid_valid_frac_final"] == 0.0
    assert report._n(float("nan"), ".2f") == "-" and report._n(1.234, ".2f") == "1.23"
    clean = run_demo._json_clean({"a": float("nan"), "b": [np.float64("inf"), np.int64(3), np.bool_(True)], "c": np.array([1.0, np.nan])})
    assert clean == {"a": None, "b": [None, 3, True], "c": [1.0, None]}
    json.dumps(clean, allow_nan=False)


def test_settlement_beyond_the_valid_range_is_refused_at_start_up():
    """The frozen mechanism folds the soil over for s/B above about 0.3 (det F < 0.3): refused before anything is rendered."""
    for size in SCENES:
        p = make_scene(size).params
        run_demo.check_field(replace(p, s_final=0.2 * p.B))  # fine
        with pytest.raises(ValueError, match="too large"):
            run_demo.check_field(replace(p, s_final=0.45 * p.B))
    assert run_demo.main(["--size", "small", "--settlement", "40", "--out", "/nonexistent/never_created"]) == 2


def test_reuse_refuses_tracker_output_of_another_field(tmp_path):
    """--reuse compares every parameter of the displacement field, not only the settlement: the tracker output of an earlier
    version of the field must not be compared with the new truth."""
    sc = make_scene("small")
    out = tmp_path / "o"
    (out / "track" / "frame_00005").mkdir(parents=True)
    (out / "track" / "frame_00005" / "raw_ux.npy").write_bytes(b"x")
    (out / "footing.mp4").write_bytes(b"x")
    (out / "cli_stdout.txt").write_text("")
    field = dict(sc.params.__dict__)
    meta = {"scene": {"width": sc.width, "height": sc.height, "n_frames": sc.n_frames, "field": field}, "tracker": {}}
    (out / "metrics.json").write_text(json.dumps(meta))
    assert run_demo.load_previous(str(out), sc)["scene"]["width"] == sc.width  # the same field: accepted
    meta["scene"]["field"] = {**field, "alpha_deg": 58.0}
    (out / "metrics.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="another displacement field.*alpha_deg"):
        run_demo.load_previous(str(out), sc)
    del meta["scene"]["field"]["w"]  # an older metrics.json without some parameter
    (out / "metrics.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="another displacement field"):
        run_demo.load_previous(str(out), sc)

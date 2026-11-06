"""Displacement field of a rigid strip footing pushed into dense sand: a kinematically admissible failure mechanism
(rigid wedge, log-spiral fan, passive wedge, with dilation) on top of a diffuse, nearly isochoric pre-peak flow.

Kinematics only: there are no stresses and no load here, no peak-load criterion; the settlement history s(t) and the
scale s_p of the pre-peak part are prescribed.  The field is stylised, not calibrated to a measurement.

Everything is expressed in the footing frame: x' to the right from the footing's centre line, y' downwards from the
initial soil surface (= the footing base); b = B / 2 is the footing's half width.  The field is

    u(X; s) = s_e(s) U_e(X) + s_m(s) U_m(X)

* Time history (progressive failure).  s_e(s) = s_p tanh(s / s_p), s_m(s) = s - s_e(s), s_p = 0.06 B.  For small
  settlements s_m ~ s^3 / (3 s_p^2): the bands emerge gradually; for large s the diffuse part saturates (s_e -> s_p).
  The footing settles by s, and the soil at its base follows: it moves by (0, s_e + s_m) = (0, s), exactly in the
  middle of the base (to 1e-9 px for |x'| up to about b - 2 w); within about two band widths of the footing's edges
  the soil lags (the regularised corner: 12 px at the edge for s = 30 px, B = 150 px).

* U_m, the failure mechanism: a kinematically admissible rigid-block + log-spiral-fan mechanism for a non-associated
  (dilatant) Mohr-Coulomb material, i.e. every velocity jump is inclined at psi to its interface (Michalowski 1997).
  It is not an upper bound (for psi < phi the limit-analysis theorems do not hold), not optimised, and not the Prandtl
  solution: alpha and beta' come from shear-band orientation theory, psi is a run-average.  Right-hand side; the left
  one is its mirror image (x' -> -x', u_x -> -u_x):

        free surface  ----E=======footing=======E'-----------------------X_e---   y' = 0
                          \\ \\       W      / /        P  (passive wedge, rigid)    /
                           \\  \\          / /    ______..----------------'  <- outer slip line C-X_e (beta')
                            \\   F  fan  /   .-'       C
                             \\  (log spiral)  .-'
                              A  <- apex      spiral R(theta) = r_A exp(theta tan psi) around E

  - W, the wedge under the footing (side angle alpha, apex A at depth b tan alpha), moves by (0, 1) (per unit s_m).
  - F, the fan around the footing edge E, between the line E-A and the line E-C (angle Theta = 180 - alpha - beta),
    bounded outward by the log spiral R(theta) = r_A exp(theta tan psi): U_F = f(theta) c(theta) with the
    circumferential direction c and f(theta) = f0 exp(theta tan psi), f0 = cos(alpha - psi) / cos(psi).
  - P, the passive wedge (E, C, X_e): rigid, U_P = f(Theta) c(Theta), inclined by psi to the outer slip line C-X_e.
  - beta' (`beta_deg`) is the outer slip line's angle to the horizontal and beta = 90 - beta' - psi.
  Every velocity jump (W|F along E-A, F|rest along the spiral, P|rest along C-X_e) makes the angle psi with its
  interface and opens it (dilation); F|P along E-C has no jump; the fan interior dilates at f tan(psi) / r.
  Per side and per unit s_m the pushed-in volume plus the dilation, b + D, leaves as surface heave: b + D =
  |X_e - E| f(Theta) cos(beta).

  The interfaces are bands: the zone memberships change over |d| < w (smootherstep, C2); the zone velocities are
  blended with a partition of unity (the stationary soil takes the weight the zones leave).  Two points of the
  mechanism are singular and regularised, both only locally:
    - the footing edge E, where W, F and P meet and the fan's shear rate f / r diverges: within r < 2 r_c the fan
      weight is tapered by smootherstep(r / (2 r_c)) (the soil flows around the edge), and the W|F band widens
      to (1 + edge_widen) w at E.  The effect reaches max(2 r_c, edge_radius w) = 3 w = 39 px from E (not only 2 r_c);
      beyond that the field is exactly the unregularised one;
    - the apex A, where the wedge's tip pushes into stationary soil (without regularisation det F < 0 there): the
      W|F and spiral bands widen to (1 + apex_widen) w within apex_radius w of A (a rounded tip).
  In the unregularised fan grad U_m has rank one, so det F - 1 = s_m f tan(psi) / r exactly (the band dilation).  The
  large volumetric strain near E is a cross term of the taper's radial gradient with the fan's angular shear rate, s_m^2 t t' f^2 / r
  (positive), plus the gradients of the band weights; near the apex it goes negative.  Mechanism alone, s_final: +0.56 at E (inside
  3 w), -0.18 at the apex (inside 7 w, det F >= 0.82), and at most +0.104 elsewhere.  Total field (with the pre-peak part, whose
  shear couples with the bands): +0.75 at E, -0.23 at the apex (det F >= 0.769 everywhere), at most +0.113 in the bands, and as
  low as -0.14 beside the fan's outer band (-0.0005 for the mechanism alone).  The values at E and A are regularisation artefacts,
  confined to about 3 w of E and 7 w of A; elsewhere the volumetric strain is the band dilation (<= about 0.11).

* U_e, the pre-peak part: a diffuse, nearly isochoric flow with a strain concentration at the footing's edges, U_e =
  (1 - kappa) U_iso + kappa U_c.  Dense sand contracts very little before the peak (about 0.1 % at 4.9 kPa and below 0.5 % at 392 kPa confining
  pressure in plane strain element tests, read by eye from Tatsuoka et al. 1986, Figs 6a, 7a).  The model therefore lets most of
  what the footing pushes in come back out as heave beside it.  That is a MODELLING ASSUMPTION: no measurement of the heave /
  intrusion ratio was found in the literature check.
  - U_iso, a stream-function flow: Phi(x', y') = Psi(x') K(y'), u_x = Phi_y = Psi K', u_y = -Phi_x = -Psi' K (per unit s_e,
    y' down).  It is divergence-free to first order, exactly (u_x,x = Psi' K' = -u_y,y).  K(y') = smootherstep(1 - y' / L_k)
    falls from 1 at the surface to 0 at L_k (C2, K'(0) = 0): u_x = 0 along the whole surface and the base is rough.  Under the
    footing (|x'| <= x_s = b - w_e / 2) Psi = -x', so u_y = K: the soil moves with the footing at the base, and u_x = -x' K' >= 0
    spreads it outward at depth.  Beside the footing Psi returns to 0: Psi' = -1 + (1 + n0) R1 - n0 R2, R1 =
    smootherstep((x' - x_s) / (2 w_e)) (the edge band, centred b + w_e / 2, where the vertical flow turns from down to up: a
    strain concentration, u_y,x up to (1 + n0) 1.875 / (2 w_e) s_e) and R2 = smootherstep((x' - x_2) / L_2) (the heave lobe fades
    out between the band's end x_2 and b + L_h, where Psi = 0 and the field vanishes, clear of the walls).  n0 follows from
    closing the loop (the integral of Psi' over the half line is 0): the lobe beside the footing takes back the flux that went
    down, so its net area is (1 - kappa) B s_e minus a small lag (the edge band starts a little under the footing, and U_c's edge).
  - U_c, a compressive bulb: u_y = g(|x'|) K_c(y'), vertical only.  g = 1 under the footing and 0 beyond it (smootherstep over
    w_c around the edge), K_c = 1 at the surface, a constant slope (uniform compression) down to y_a, then a smooth fade to 0 at
    L_c.  The base moves with the footing ((1 - kappa) + kappa = 1) and the bulb does not lift the surface beside it.  It
    absorbs kappa B s_e of the intrusion (the integral of g is B): the only first-order volume change of U_e, -kappa B s_e.
  kappa is the largest weight for which det F - 1 of s_p U_e (the true determinant, not the linearisation) stays >= -0.005.

  DEVIATIONS FROM THE BRIEF (accepted by the controller; all measured, see field_summary):
  (1) det F - 1 of s_p U_e is [-0.0049, +0.0122], not within [-0.005, +0.002] (nor [-0.01, +0.005] in the edge discs).  The minimum
      is at (x', y') = (-55, 165) under the footing; the maximum, +0.0122, at (-95, 63) beside the edge, outside the 3 w discs (inside
      them +0.0097).  The first-order divergence of s_p U_e is <= 0 everywhere (min -0.0025), so the positive part is purely
      second order (the shear coupling below).  It exceeds +0.002 over 3.3 % of the soil (|x'| 74-120, y' 10-138) and +0.005 over
      2.0 %.  +0.002 is out of reach for a stream-function flow with an edge strain concentration unless w_e > 100 px.
  (2) w_e = 3 w = 39 px, not about w: a narrow band makes the second-order term grow like 1 / w_e (w_e = w: +0.033, edge-band gmax
      1.15, and -0.29 beside the bands instead of -0.14).
  (3) kappa = 0.07, not 0.1-0.2: it is the largest weight keeping the minimum >= -0.005 (kappa 0.1: -0.0059; 0.2: -0.0092).
  The second-order term of the isochoric flow, det(grad u) = det Hess(Phi) = -(Psi' K')^2 + Psi Psi'' K K'', is about -0.002
  under the footing, and the bulb adds kappa s_p q ~ -0.003.  The cross term Psi Psi'' K K'' is the shear coupling of the edge
  band (u_y,x ~ Psi'') with the depth variation of the outward flow (u_x,y = Psi K''): positive, at depth 20-120 px along the
  edges, proportional to 1 / w_e (+0.033 for w_e = w), which is why the band is w_e = 3 w wide (+0.012 at most); `field_summary`
  gives the ranges and where they are.  The pre-peak shear also couples with the mechanism's band shear (cross terms of the
  two fields' gradients), more strongly the narrower the band.  The field is exactly zero beyond |x'| = b + L_h and below
  L_k and L_c, hence at the container's walls and bottom.

The shape functions U_e, U_m do not depend on s: a whole sequence is a cheap linear combination (see GridField).
The gradients are exact (forward-mode automatic differentiation with the small dual-number class below).  Above the
initial surface (y' < 0, outside the soil) the field is continued as constant in y' so that the inverse map is
well-defined for the pixels of the heave.

Reference: Michalowski, R.L. (1997) An estimate of the influence of soil weight on bearing capacity using limit
analysis.  Soils and Foundations 37(4), 57-64.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------------------------------------
# forward-mode automatic differentiation: value and the two partial derivatives (wrt X and wrt Y)
# ---------------------------------------------------------------------------------------------------------
class D:
    """Dual number with two partials; operands may also be plain floats or arrays (constants)."""

    __slots__ = ("v", "dx", "dy")

    def __init__(self, v, dx, dy):
        self.v, self.dx, self.dy = v, dx, dy

    def __add__(self, o):
        if isinstance(o, D):
            return D(self.v + o.v, self.dx + o.dx, self.dy + o.dy)
        return D(self.v + o, self.dx, self.dy)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, D):
            return D(self.v - o.v, self.dx - o.dx, self.dy - o.dy)
        return D(self.v - o, self.dx, self.dy)

    def __rsub__(self, o):
        return D(o - self.v, -self.dx, -self.dy)

    def __neg__(self):
        return D(-self.v, -self.dx, -self.dy)

    def __mul__(self, o):
        if isinstance(o, D):
            return D(self.v * o.v, self.dx * o.v + self.v * o.dx, self.dy * o.v + self.v * o.dy)
        return D(self.v * o, self.dx * o, self.dy * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, D):
            inv = 1.0 / o.v
            v = self.v * inv
            return D(v, (self.dx - v * o.dx) * inv, (self.dy - v * o.dy) * inv)
        return D(self.v / o, self.dx / o, self.dy / o)

    def __rtruediv__(self, o):
        inv = 1.0 / self.v
        v = o * inv
        return D(v, -v * self.dx * inv, -v * self.dy * inv)


def _unary(x, f, df):
    v = f(x.v)
    d = df(x.v, v)
    return D(v, d * x.dx, d * x.dy)


def d_exp(x):
    return _unary(x, np.exp, lambda a, e: e)


def d_sqrt(x):
    return _unary(x, np.sqrt, lambda a, r: 0.5 / r)


def d_abs(x):
    return _unary(x, np.abs, lambda a, r: np.sign(a))


def d_atan2(y, x):
    """atan2(y, x) of two dual numbers (finite at the origin: value 0, partials 0)."""
    den = x.v * x.v + y.v * y.v + 1e-30
    return D(np.arctan2(y.v, x.v), (x.v * y.dx - y.v * x.dx) / den, (x.v * y.dy - y.v * x.dy) / den)


def sstep(t):
    """Smootherstep of t clipped to [0, 1]: 6t^5 - 15t^4 + 10t^3, C2, flat at both ends."""
    return _unary(t, lambda a: (lambda c: c * c * c * (10.0 + c * (-15.0 + 6.0 * c)))(np.clip(a, 0.0, 1.0)),
                  lambda a, r: (lambda c: 30.0 * c * c * (1.0 - c) ** 2)(np.clip(a, 0.0, 1.0)))


def sstep_prime(t):
    """The derivative of the smootherstep, 30 c^2 (1 - c)^2 with c = t clipped to [0, 1] (its own derivative is
    60 c (1 - c) (1 - 2 c) inside (0, 1))."""
    def f(a):
        c = np.clip(a, 0.0, 1.0)
        return 30.0 * c * c * (1.0 - c) ** 2

    def df(a, r):
        c = np.clip(a, 0.0, 1.0)
        return np.where((a > 0.0) & (a < 1.0), 60.0 * c * (1.0 - c) * (1.0 - 2.0 * c), 0.0)
    return _unary(t, f, df)


# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Geometry [px] and parameters of the mechanism.  Defaults: the full-size scene (1280 x 720), B = 150 px."""

    xc: float = 640.0  # footing centre line (x of the container's centre)
    y0: float = 300.0  # initial soil surface (footing base)
    b: float = 75.0  # footing half width (B = 2 b = 150 px)
    depth: float = 350.0  # soil depth: the container bottom is at y0 + depth
    half_box: float = 560.0  # distance from the centre line to the container walls
    alpha_deg: float = 60.0  # wedge side angle to the horizontal: 45 + (phi + psi_peak) / 4, phi = 45, psi_peak = 15
    beta_deg: float = 30.0  # beta': outer slip line's angle to the horizontal: 45 - (phi + psi_peak) / 4
    psi_deg: float = 4.0  # kinematic dilation angle: the run-average dilation of the bands
    w: float = 13.0  # band half width: memberships change over |d| < w (FWHM of the strain profile ~ 1.08 w)
    rc: float = 13.0  # corner radius: the footing edge is regularised within 2 rc
    apex_widen: float = 1.0  # the W|F and spiral bands widen by (1 + apex_widen) at the apex (rounded wedge tip)
    apex_radius: float = 6.0  # ... within apex_radius * w of the apex
    edge_widen: float = 2.0  # the W|F band widens by (1 + edge_widen) at the footing edge
    edge_radius: float = 3.0  # ... within edge_radius * w of the edge
    s_final: float = 30.0  # settlement at the last frame (s / B = 0.2)
    sp_frac: float = 0.06  # s_p / B: the pre-peak settlement scale
    # pre-peak part U_e = (1 - kappa) U_iso + kappa U_c (see the module docstring)
    ue_kappa: float = 0.07  # kappa: the share of the pre-peak intrusion that the compressive bulb U_c absorbs (the largest
    # one for which det F - 1 of s_p U_e stays >= -0.005 with this bulb)
    ue_lh: float = 1.6  # L_h / B: the heave lobe beside the footing reaches b + L_h (clear of the walls: b + L_h = 315 px)
    ue_lk: float = 2.2  # L_k / B: depth over which the stream-function flow decays
    ue_we: float = 3.0  # w_e / w: half width of the edge band, where the vertical flow turns from down (-1) to up
    ue_shift: float = 0.5  # the edge band starts at b - (1 - ue_shift) w_e (the heave area beside the footing, not the lag)
    ue_lc: float = 2.2  # L_c / B: depth of the compressive bulb
    ue_flat: float = 0.5  # y_a / L_c: the bulb's compression is uniform down to y_a, then fades smoothly
    ue_cw: float = 1.0  # w_c / w: half width of the bulb's lateral edge (centred on the footing's edge)

    # ---- closed forms of the mechanism ------------------------------------------------------------------
    @property
    def B(self):
        return 2.0 * self.b

    @property
    def beta_in_deg(self):
        """beta = 90 - beta' - psi: the passive wedge's angle at the footing edge."""
        return 90.0 - self.beta_deg - self.psi_deg

    @property
    def theta_deg(self):
        """Fan angle Theta = 180 - alpha - beta."""
        return 180.0 - self.alpha_deg - self.beta_in_deg

    @property
    def ha(self):
        """Depth of the wedge's apex."""
        return self.b * np.tan(np.deg2rad(self.alpha_deg))

    @property
    def r_a(self):
        return self.b / np.cos(np.deg2rad(self.alpha_deg))

    @property
    def r_theta(self):
        """Spiral radius at the fan's outer edge (the point C)."""
        return self.r_a * np.exp(np.deg2rad(self.theta_deg) * np.tan(np.deg2rad(self.psi_deg)))

    @property
    def xe(self):
        """Distance of the outer slip line's exit X_e at the surface from the centre line."""
        return self.b + self.r_theta * np.cos(np.deg2rad(self.psi_deg)) / np.sin(np.deg2rad(self.beta_deg))

    @property
    def f0(self):
        return np.cos(np.deg2rad(self.alpha_deg - self.psi_deg)) / np.cos(np.deg2rad(self.psi_deg))

    @property
    def f_theta(self):
        """Speed of the passive wedge per unit footing velocity."""
        return self.f0 * np.exp(np.deg2rad(self.theta_deg) * np.tan(np.deg2rad(self.psi_deg)))

    @property
    def depth_mech(self):
        """Depth of the mechanism: the deepest point of the log spiral, r_A exp(th tan psi) cos(psi) at th* = 90 - alpha
        + psi (limited to the fan)."""
        ps = np.deg2rad(self.psi_deg)
        th = np.clip(np.pi / 2 - np.deg2rad(self.alpha_deg) + ps, 0.0, np.deg2rad(self.theta_deg))
        phi = np.pi - np.deg2rad(self.alpha_deg) - th
        return float(self.r_a * np.exp(th * np.tan(ps)) * np.sin(phi))

    @property
    def heave_flux(self):
        """Surface heave per side and unit s_m, |X_e - E| f(Theta) cos(beta) [px^2 per px]."""
        return (self.xe - self.b) * self.f_theta * np.cos(np.deg2rad(self.beta_in_deg))

    @property
    def dilation(self):
        """D: the dilation of the bands and the fan interior per side and unit s_m: heave_flux - b."""
        return self.heave_flux - self.b

    # ---- time history ------------------------------------------------------------------------------------
    @property
    def sp(self):
        return self.sp_frac * self.B

    def s_e(self, s):
        """The diffuse (pre-peak) part of the settlement: s_p tanh(s / s_p)."""
        s = np.maximum(s, 0.0)
        return self.sp * np.tanh(s / self.sp)

    def s_m(self, s):
        """The mechanism part: s - s_e(s)."""
        s = np.maximum(s, 0.0)
        return s - self.s_e(s)

    # ---- the pre-peak part -------------------------------------------------------------------------------
    @property
    def ue_lobe(self):
        """L_h: the heave lobe beside the footing reaches b + L_h from the centre line (zero field beyond)."""
        return self.ue_lh * self.B

    @property
    def ue_depth_k(self):
        """L_k: depth below which the stream-function flow is zero."""
        return self.ue_lk * self.B

    @property
    def ue_edge_hw(self):
        """w_e: half width of the edge band of the stream function (px)."""
        return self.ue_we * self.w

    @property
    def ue_band_start(self):
        """Where the edge band starts (the vertical flow is down, u_y = 1, for |x'| below it)."""
        return self.b - (1.0 - self.ue_shift) * self.ue_edge_hw

    @property
    def ue_n0(self):
        """Height of the heave lobe's start per unit s_e (the vertical flow just beyond the edge band is up, by n0).  It
        follows from the loop closing: the downward flux B-ish under the footing returns as the lobe, the integral of
        Psi' over the half line is 0."""
        c = self.ue_band_start + self.ue_edge_hw  # centre of the band
        l2 = self.b + self.ue_lobe - (c + self.ue_edge_hw)  # the lobe's decline
        return c / (self.ue_edge_hw + 0.5 * l2)

    @property
    def ue_comp_depth(self):
        """L_c: depth below which the compressive bulb is zero."""
        return self.ue_lc * self.B

    @property
    def ue_half_support(self):
        return self.b + self.ue_lobe


# ---------------------------------------------------------------------------------------------------------
# geometry constants (floats) of a Params
# ---------------------------------------------------------------------------------------------------------
class _Geo:
    pass


@lru_cache(maxsize=16)
def _geometry(p: Params):
    g = _Geo()
    a, bp, ps = np.deg2rad(p.alpha_deg), np.deg2rad(p.beta_deg), np.deg2rad(p.psi_deg)
    bet = np.deg2rad(p.beta_in_deg)
    g.b, g.w, g.rc = p.b, p.w, p.rc
    g.apex_k, g.apex_r = p.apex_widen, p.apex_radius * p.w
    g.edge_k, g.edge_r = p.edge_widen, p.edge_radius * p.w
    g.ha = p.ha
    g.alpha, g.sa, g.ca = a, np.sin(a), np.cos(a)
    g.sbp, g.cbp = np.sin(bp), np.cos(bp)
    g.sbet, g.cbet = np.sin(bet), np.cos(bet)
    g.tanpsi, g.cospsi = np.tan(ps), np.cos(ps)
    g.rA, g.f0, g.fth = p.r_a, p.f0, p.f_theta
    g.Cx = p.b + p.r_theta * g.cbet
    g.Cy = p.r_theta * g.sbet
    g.UPx, g.UPy = g.fth * g.sbet, -g.fth * g.cbet  # passive wedge velocity (right-hand side)
    # the region where U_m / U_e can differ from zero (a margin beyond the bands)
    g.m_xlim = p.xe + 2.0 * p.w / g.sbp + 2.0
    g.m_ylim = p.depth_mech + 2.0 * p.w + 2.0
    # the pre-peak part: edge band [e_xs, e_x2] of Psi', the lobe's decline [e_x2, e_end], the loop-closing height n0
    g.e_kappa = p.ue_kappa
    g.e_we = p.ue_edge_hw
    g.e_xs = p.ue_band_start
    g.e_x2 = g.e_xs + 2.0 * g.e_we
    g.e_end = p.b + p.ue_lobe
    g.e_l2 = g.e_end - g.e_x2
    g.e_n0 = p.ue_n0
    g.e_lk = p.ue_depth_k
    g.e_lc = p.ue_comp_depth
    g.e_ya = p.ue_flat * g.e_lc
    g.e_q = 2.0 / (g.e_lc + g.e_ya)  # the bulb's uniform compression rate per unit s_e of its settlement
    g.e_cw = p.ue_cw * p.w
    g.e_xlim = g.e_end + 1.0
    g.e_ylim = max(g.e_lk, g.e_lc) + 1.0
    return g


# ---------------------------------------------------------------------------------------------------------
# U_m: the failure mechanism
# ---------------------------------------------------------------------------------------------------------
def _side(xr, yp, g, wl):
    """Zone weights and fan velocity of the right-hand side in the frame of that side (xr to the side's outer
    direction): a = S(d_EA) (outside the wedge), the fan weight wF, the passive-wedge weight wP and the fan velocity
    (UFx, UFy).  wl: half width of the W|F and spiral bands (a dual number: wider near the apex and the edge).  Within
    2 r_c of the footing edge E the fan weight is tapered by smootherstep(r / (2 r_c)): the soil there flows around
    the edge instead of following the (singular) fan field.  The passive wedge is rigid, hence not singular: its weight
    is not tapered, and not multiplied by `a` either (it is bounded by E-C and C-X_e only), which keeps the surface
    integral of u_y, i.e. the first-order volume budget, within 2 % of the closed form."""
    w = g.w
    dx = xr - g.b
    dy = yp
    a = sstep((dx * g.sa + dy * g.ca + wl) / (2.0 * wl))  # outside the line E-A (W|F band)
    r = d_sqrt(dx * dx + dy * dy + 1e-9)
    theta = (np.pi - g.alpha) - d_atan2(dy, dx)  # fan angle, measured from E-A
    ex = d_exp(theta * g.tanpsi)
    d_sp = (r - ex * g.rA) * g.cospsi  # signed distance to the log spiral (positive outside the fan)
    s_sp = sstep((wl - d_sp) / (2.0 * wl))  # inside the spiral
    c_ec = sstep((dx * g.sbet - dy * g.cbet + w) * (0.5 / w))  # beyond the line E-C (passive wedge side)
    d_cx = (xr - g.Cx) * g.sbp + (yp - g.Cy) * g.cbp
    s_cx = sstep((w - d_cx) * (0.5 / w))  # inside the line C-X_e
    taper = sstep(r * (0.5 / g.rc))  # corner regularisation: 0 at E, 1 from 2 r_c on
    wF = a * (1.0 - c_ec) * s_sp * taper
    wP = c_ec * s_cx
    f = ex * g.f0
    UFx, UFy = f * dy / r, -(f * dx / r)
    return a, wF, wP, UFx, UFy


def _um(xp, yp, g):
    """(u_x, u_y) of the mechanism per unit s_m; xp, yp dual numbers (yp >= 0)."""
    rho = d_sqrt(xp * xp + (yp - g.ha) * (yp - g.ha) + 1e-9)  # distance to the apex
    rer = d_sqrt((xp - g.b) * (xp - g.b) + yp * yp + 1e-9)  # distance to the footing edges
    rel = d_sqrt((xp + g.b) * (xp + g.b) + yp * yp + 1e-9)
    wl = (sstep(1.0 - rho * (1.0 / g.apex_r)) * g.apex_k
          + (sstep(1.0 - rer * (1.0 / g.edge_r)) + sstep(1.0 - rel * (1.0 / g.edge_r))) * g.edge_k) * g.w + g.w
    # (the W|F and spiral bands widen toward the singular points: a rounded wedge tip and footing edge)
    aR, wFR, wPR, UFxR, UFyR = _side(xp, yp, g, wl)
    aL, wFL, wPL, UFxL, UFyL = _side(-xp, yp, g, wl)
    wW = (1.0 - aR) * (1.0 - aL)
    # partition of unity: the stationary soil takes what the zones leave (weight 1 - sum); where the zones overlap
    # (the two fans around the apex) the weights are normalised, smoothly: T = 1 + e sstep(e / 0.3), e = sum - 1
    e = wW + wFR + wPR + wFL + wPL - 1.0
    T = 1.0 + e * sstep(e * (1.0 / 0.3))
    nx = wFR * UFxR + wPR * g.UPx - (wFL * UFxL + wPL * g.UPx)
    ny = wW + wFR * UFyR + wPR * g.UPy + wFL * UFyL + wPL * g.UPy
    return nx / T, ny / T


# ---------------------------------------------------------------------------------------------------------
# U_e: the pre-peak part, (1 - kappa) U_iso + kappa U_c
# ---------------------------------------------------------------------------------------------------------
def _phi_int(u):
    """Antiderivative of the smootherstep: u^4 (2.5 - 3 u + u^2); Phi(1) = 1/2."""
    return u ** 4 * (2.5 - 3.0 * u + u * u)


def _ramp_integral(a, x0, length):
    """Integral from -infinity of smootherstep((a - x0) / length): 0 below x0, length Phi(u) inside, (a - x0) - length / 2
    beyond x0 + length (plain float arrays)."""
    u = (a - x0) / length
    return length * (_phi_int(np.clip(u, 0.0, 1.0)) + np.maximum(u - 1.0, 0.0))


def _smootherstep(u):
    c = np.clip(u, 0.0, 1.0)
    return c * c * c * (10.0 + c * (-15.0 + 6.0 * c))


def _psi_pos(a, g):
    """The stream function Psi(x') of U_iso for x' >= 0 (a >= 0, float array): -x' under the footing, then
    Psi' = -1 + (1 + n0) R1 - n0 R2 with R1 = smootherstep((x' - x_s) / (2 w_e)) (the edge band: the vertical flow turns from
    down to up) and R2 = smootherstep((x' - x_2) / L_2) (the heave lobe fades out), so Psi = -x' + (1 + n0) int R1 - n0 int R2,
    and Psi = 0 for x' >= b + L_h."""
    return (-a + (1.0 + g.e_n0) * _ramp_integral(a, g.e_xs, 2.0 * g.e_we) - g.e_n0 * _ramp_integral(a, g.e_x2, g.e_l2))


def _psi_pos_prime(a, g):
    return (-1.0 + (1.0 + g.e_n0) * _smootherstep((a - g.e_xs) / (2.0 * g.e_we))
            - g.e_n0 * _smootherstep((a - g.e_x2) / g.e_l2))


def _comp_profile(yp, g):
    """K_c(y'): the compressive bulb's vertical profile, 1 at the surface and 0 from L_c on; its slope is the constant -q down
    to y_a and fades out smoothly (smootherstep) to 0 at L_c: K_c = 1 - q (y - int_0^y smootherstep((t - y_a) / (L_c - y_a)) dt)."""
    ya, lc, q = g.e_ya, g.e_lc, g.e_q

    def f(a):
        return 1.0 - q * (a - _ramp_integral(a, ya, lc - ya))

    def df(a, r):
        return -q * (1.0 - _smootherstep((a - ya) / (lc - ya)))
    return _unary(yp, f, df)


def _ue_parts(xp, yp, g):
    """(u_x, u_y) of U_iso (per unit s_e of it) and u_y of U_c (U_c has no horizontal component); xp, yp dual numbers.

    U_iso: stream function Phi = Psi(x') K(y'), u_x = Phi_y = Psi K', u_y = -Phi_x = -Psi' K, with K(y') = smootherstep(1 - y'/L_k):
    divergence-free (u_x,x = Psi' K' = -u_y,y), u_y = 1 and u_x = 0 along the footing's base (K(0) = 1, K'(0) = 0: a rough base)
    and u_x = 0 along the whole surface, Psi odd so u_x odd and u_y even.
    U_c: u_y = g(|x'|) K_c(y'), g = 1 under the footing, 0 beyond it, its edge (half width w_c) centred on the footing's edge,
    so that the base moves with the footing (1 - kappa) + kappa = 1 and U_c does not lift the surface beside the footing."""
    ax = d_abs(xp)
    psi_p = (sstep((ax - g.e_xs) * (1.0 / (2.0 * g.e_we))) * (1.0 + g.e_n0)
             - sstep((ax - g.e_x2) * (1.0 / g.e_l2)) * g.e_n0 - 1.0)  # Psi' (even)
    psi = _unary(xp, lambda a: np.sign(a) * _psi_pos(np.abs(a), g), lambda a, r: _psi_pos_prime(np.abs(a), g))  # Psi (odd)
    c = 1.0 - yp * (1.0 / g.e_lk)
    kk = sstep(c)
    kp = sstep_prime(c) * (-1.0 / g.e_lk)  # K'
    gx = sstep(1.0 - (ax - (g.b - g.e_cw)) * (1.0 / (2.0 * g.e_cw)))
    return psi * kp, -(psi_p * kk), gx * _comp_profile(yp, g)


def _ue(xp, yp, g):
    """(u_x, u_y) of the pre-peak part per unit s_e: (1 - kappa) U_iso + kappa U_c."""
    iso_x, iso_y, c_y = _ue_parts(xp, yp, g)
    k = g.e_kappa
    return iso_x * (1.0 - k), iso_y * (1.0 - k) + c_y * k


# ---------------------------------------------------------------------------------------------------------
def _evaluate(p: Params, X, Y, fns):
    """Evaluate dual-number fields at arbitrary points.  fns: list of (names, xlim, ylim, fn) where fn(xp, yp) returns one dual
    number per name; the field is zero outside |x'| < xlim, y' < ylim.  Returns {name: values, name_x: d/dX, name_y: d/dY}.
    X, Y: absolute pixel coordinates; above the initial surface (y' < 0) the field is continued as constant in y'."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    shape = np.broadcast(X, Y).shape
    Xf = np.broadcast_to(X, shape).ravel()
    Yf = np.broadcast_to(Y, shape).ravel()
    n = Xf.size
    out = {}
    for names, _, _, _ in fns:
        for k in names:
            out[k] = np.zeros(n)
            out[k + "_x"] = np.zeros(n)
            out[k + "_y"] = np.zeros(n)
    xp0 = Xf - p.xc
    yr = Yf - p.y0
    yp0 = np.maximum(yr, 0.0)  # the field is continued as constant above the surface
    dyv = (yr >= 0.0).astype(np.float64)
    for names, xlim, ylim, fn in fns:
        idx = np.nonzero((np.abs(xp0) < xlim) & (yp0 < ylim))[0]
        if idx.size == 0:
            continue
        xp = D(xp0[idx], np.ones(idx.size), np.zeros(idx.size))
        yp = D(yp0[idx], np.zeros(idx.size), dyv[idx])
        for k, f in zip(names, fn(xp, yp)):
            out[k][idx] = f.v
            out[k + "_x"][idx] = f.dx
            out[k + "_y"][idx] = f.dy
    return {k: v.reshape(shape) for k, v in out.items()}


def basis_arrays(p: Params, X, Y):
    """Shape functions on arbitrary points: dict of arrays 'ex', 'ey' (U_e, coefficient of s_e), 'mx', 'my' (U_m,
    coefficient of s_m) and their partials wrt X and Y ('ex_x', 'ex_y', ...).  X, Y: absolute pixel coordinates."""
    g = _geometry(p)
    return _evaluate(p, X, Y, [(("ex", "ey"), g.e_xlim, g.e_ylim, lambda xp, yp: _ue(xp, yp, g)),
                               (("mx", "my"), g.m_xlim, g.m_ylim, lambda xp, yp: _um(xp, yp, g))])


def prepeak_parts(p: Params, X, Y):
    """The two parts of U_e separately, each per unit of its own weight: 'isox', 'isoy' (the stream-function flow U_iso,
    u_x = Phi_y, u_y = -Phi_x, exactly divergence-free) and 'cy' (the compressive bulb U_c, vertical only), with their
    partials ('isox_x', ...).  U_e = (1 - kappa) U_iso + kappa U_c."""
    g = _geometry(p)
    return _evaluate(p, X, Y, [(("isox", "isoy", "cy"), g.e_xlim, g.e_ylim, lambda xp, yp: _ue_parts(xp, yp, g))])


class FootingField:
    """Deformation x = X + u(X; s) of the footing scene at settlement s [px]; the interface of synth.py's fields
    (fwd, inv, grad)."""

    def __init__(self, params: Params, s: float):
        self.p = params
        self.s = float(s)
        self.s_e = float(params.s_e(self.s))
        self.s_m = float(params.s_m(self.s))

    def _combine(self, B):
        se, sm = self.s_e, self.s_m
        ux = se * B["ex"] + sm * B["mx"]
        uy = se * B["ey"] + sm * B["my"]
        gr = (se * B["ex_x"] + sm * B["mx_x"], se * B["ex_y"] + sm * B["mx_y"],
              se * B["ey_x"] + sm * B["my_x"], se * B["ey_y"] + sm * B["my_y"])
        return ux, uy, gr

    def disp(self, X, Y):
        B = basis_arrays(self.p, X, Y)
        ux, uy, _ = self._combine(B)
        return ux, uy

    def fwd(self, X, Y):
        ux, uy = self.disp(X, Y)
        return X + ux, Y + uy

    def grad(self, X, Y):
        """(du/dX, du/dY, dv/dX, dv/dY) with u = x - X."""
        _, _, gr = self._combine(basis_arrays(self.p, X, Y))
        return gr

    def inv(self, x, y, X0=None, tol=1e-10, maxit=40):
        """Material point X with X + u(X) = x: Newton's method (exact Jacobian), damped when the step is large."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X0 is None:
            X, Y = x.copy(), y.copy()
        else:
            X, Y = X0[0].copy(), X0[1].copy()
        for _ in range(maxit):
            ux, uy, (a, b_, c, d) = self._combine(basis_arrays(self.p, X, Y))
            rx, ry = X + ux - x, Y + uy - y
            j11, j12, j21, j22 = 1.0 + a, b_, c, 1.0 + d
            det = j11 * j22 - j12 * j21
            dX = (j22 * rx - j12 * ry) / det
            dY = (-j21 * rx + j11 * ry) / det
            step = np.maximum(np.abs(dX), np.abs(dY))
            t = np.minimum(1.0, 8.0 / np.maximum(step, 1e-300))  # never move more than 8 px at once
            X, Y = X - t * dX, Y - t * dY
            if float(np.max(step)) < tol:
                break
        return X, Y


class GridField:
    """FootingField restricted to a fixed set of reference points (e.g. the frame-0 pixel grid): the shape functions
    are evaluated once, then every settlement costs a few array operations."""

    def __init__(self, params: Params, X, Y):
        self.p = params
        self.B = basis_arrays(params, X, Y)

    def at(self, s):
        """(ux, uy, (dudx, dudy, dvdx, dvdy)) of settlement s on the stored points."""
        return FootingField(self.p, s)._combine(self.B)


# ---------------------------------------------------------------------------------------------------------
# the ground surface and the volume budget (for the report, the curves and the tests)
# ---------------------------------------------------------------------------------------------------------
def surface_profile(p: Params, s: float, step: float = 0.1):
    """The initial surface (y' = 0) after settlement s: material positions X' (from the centre line), deformed
    positions x' and the height h above the initial surface level (h = -u_y, positive up)."""
    X = np.arange(p.xc - p.half_box, p.xc + p.half_box + 0.5 * step, step)
    ux, uy = FootingField(p, s).disp(X, np.full_like(X, p.y0))
    return X - p.xc, X - p.xc + ux, -uy


def _trapz(h, xd, sel):
    """Trapezoid rule of h over the nodes xd for the segments whose two ends are selected."""
    seg = sel[1:] & sel[:-1]
    return float(np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(xd) * seg))


def surface_budget(p: Params, s: float, step: float = 0.1):
    """Surface heave and the (exact) volume budget of the true field at settlement s, in px^2 (areas of the 2-D scene).

    The deformed surface is a graph over x' (checked); the soil's area change is the area under it (the container's
    walls and bottom do not move): net_dv = int h dx'.  The profile is resampled on a uniform grid of the deformed
    position x' that has the footing's edges |x'| = b as nodes, so the integrals are accurate to O(step^2).  Outside the
    footing (|x'| > b): heave_area is the positive part of h, trough_area the negative part, net_heave their difference;
    heave_max is the highest point and heave_plateau the extent of the part within 1 % of it (the passive wedge
    translating: a plateau, not a ridge).  intrusion_area = B s is what the footing pushes in.  Under the footing the
    soil surface follows the footing's base except within about two band widths of the edges, where it lags (the
    regularised corner, i.e. soil above the footing's base level, overlapping the rigid footing): contact_overlap =
    int (h + s) dx' over |x'| <= b.  Exactly (a bookkeeping identity of the definitions, up to quadrature),
    net_heave = intrusion_area + net_dv - contact_overlap.  h_edge is the surface height at the footing's edge, and
    zero_cross the distance from the centre line at which the surface returns to the initial level."""
    Xm, xd, h0 = surface_profile(p, s, step)
    if not np.all(np.diff(xd) > 0):
        raise ValueError("the deformed surface is not a graph over x'")
    xg = np.unique(np.round(np.concatenate([np.arange(xd[0], xd[-1] + 0.5 * step, step), [-p.b, p.b]]), 9))
    xg = xg[(xg >= xd[0] - 1e-9) & (xg <= xd[-1] + 1e-9)]
    h = np.interp(xg, xd, h0)
    outside = np.abs(xg) >= p.b - 1e-9  # the nodes +-b belong to both
    inside = np.abs(xg) <= p.b + 1e-9
    i = int(np.argmax(np.where(np.abs(xg) > p.b + 1e-9, h, -np.inf)))
    hm = float(h[i])
    near = (h >= 0.99 * hm) & (np.abs(xg) > p.b) & (np.sign(xg) == np.sign(xg[i]))
    lo = hi = i
    while lo > 0 and near[lo - 1]:
        lo -= 1
    while hi < len(xg) - 1 and near[hi + 1]:
        hi += 1
    right = (xg > p.b) & (h > 0.0)
    return {"s": float(s), "heave_max": hm, "heave_max_x": float(xg[i]),
            "heave_plateau": (float(abs(xg[lo])), float(abs(xg[hi]))) if xg[i] > 0 else (float(abs(xg[hi])), float(abs(xg[lo]))),
            "net_heave": _trapz(h, xg, outside), "heave_area": _trapz(np.maximum(h, 0.0), xg, outside),
            "trough_area": _trapz(np.maximum(-h, 0.0), xg, outside), "intrusion_area": p.B * float(s),
            "contact_overlap": _trapz(h + s, xg, inside), "net_dv": _trapz(h, xg, np.ones_like(outside)),
            "h_edge": float(np.interp(p.b, xg, h)), "zero_cross": float(xg[right].min()) if right.any() else float("nan")}


def prepeak_heave(p: Params, step: float = 0.02):
    """The surface of the pre-peak part alone, per unit s_e (first order; U_e has no horizontal flow at the surface, so the
    surface points do not move sideways): h(x') = -u_y(x', 0).  Returns the net area of the lift beside the footing (|x'| > b,
    both sides: (1 - kappa) B up to the part of the downward flux that the edge band and U_c's edge take), its positive part,
    the area of the dragged-down soil beside the footing, the lag under the footing (int (1 - u_y) dx' over |x'| < b, soil that
    overlaps the rigid footing), the net area of the whole surface (-kappa B: the compression) and the highest point."""
    xm = p.ue_half_support + 5.0
    x = np.unique(np.round(np.concatenate([np.arange(-xm, xm + 0.5 * step, step), [-p.b, p.b]]), 9))
    B = basis_arrays(p, p.xc + x, np.full_like(x, p.y0))
    ux, h = B["ex"], -B["ey"]
    out, inside = np.abs(x) >= p.b - 1e-9, np.abs(x) <= p.b + 1e-9
    return {"net_outside": _trapz(h, x, out), "pos_outside": _trapz(np.maximum(h, 0.0), x, out),
            "drag_outside": _trapz(np.maximum(-h, 0.0), x, out), "lag_inside": _trapz(1.0 + h, x, inside),
            "net_total": _trapz(h, x, np.ones_like(out)), "h_max": float(h.max()), "h_max_x": float(abs(x[np.argmax(h)])),
            "ux_surface_max": float(np.abs(ux).max())}


def analytic_budget(p: Params, s: float):
    """The first-order (small-displacement) volume budget in closed form: pre-peak compression -kappa B s_e (U_iso is
    divergence-free, and U_c pushes the area B s_e it takes at the base into the soil below it: int g dx' = B), band + fan
    dilation 2 D s_m, and their sum.  It is the integral of the volumetric strain div(u) over the soil."""
    se, sm = float(p.s_e(s)), float(p.s_m(s))
    pre = -p.ue_kappa * p.B * se
    dil = 2.0 * p.dilation * sm
    return {"pre_peak": pre, "dilation": dil, "total": pre + dil}


def second_order_area(p: Params, s: float, step: float = 0.05):
    """The finite-displacement part of the area change, int det(grad u) dA = (1/2) int (u_x du_y - u_y du_x) along the
    surface (a boundary integral: it is minus the area enclosed by the surface's displacement hodograph; the
    container's walls and bottom do not move).  The linear-in-s field is not an exact finite motion, so the area
    change is first-order budget + this term: int (det F - 1) dA = analytic_budget total + second_order_area."""
    X = np.arange(p.xc - p.half_box, p.xc + p.half_box + 0.5 * step, step)
    ux, uy = FootingField(p, s).disp(X, np.full_like(X, p.y0))
    return float(0.5 * np.trapezoid(ux * np.gradient(uy, X) - uy * np.gradient(ux, X), X))


def _disc_masks(p: Params, X, Y):
    xp, yp = X - p.xc, Y - p.y0
    near_edge = np.hypot(np.abs(xp) - p.b, yp) < p.edge_radius * p.w
    near_apex = np.hypot(xp, yp - p.ha) < (p.apex_radius + 1.0) * p.w
    return near_edge, near_apex


def singular_discs(p: Params, X, Y):
    """Mask of the neighbourhoods of the footing edges and the wedge's apex (the points where the mechanism is
    singular and regularised): the volumetric strain there is a regularisation artefact."""
    near_edge, near_apex = _disc_masks(p, X, Y)
    return near_edge | near_apex


def _green_lagrange_gmax(a, b, c, d):
    """gmax (difference of the principal Green-Lagrange strains, the tracker's definition) from the displacement
    gradient (du/dX, du/dY, dv/dX, dv/dY)."""
    exx, eyy, exy = a + 0.5 * (a * a + c * c), d + 0.5 * (b * b + d * d), 0.5 * (b + c + a * b + c * d)
    return np.sqrt((exx - eyy) ** 2 + 4.0 * exy ** 2)


def field_summary(p: Params, s=None, step: float = 1.0):
    """Ranges of the true field at settlement s (default: the last frame) on a grid of `step` px over the whole soil:
    the smallest det F and where, the volumetric strain det F - 1 of the bands and the fan (outside the singular discs
    around the footing's edges and the wedge's apex; the minimum also for the mechanism alone, to show how much of it is the
    cross term with the pre-peak shear), inside the edge discs (maximum) and the apex disc (minimum), and that of the
    pre-peak part alone s_e U_e: the extremes and where they are, the extremes inside the edge discs (radius
    edge_radius w) and outside them, the mean, the shares of the soil outside [-0.005, +0.002] and above +0.005, the largest
    first-order divergence of U_iso (zero), and the largest gmax of the pre-peak part (in all, and under the footing's
    middle |x'| < b / 2, y' > 0).  Also the largest Green-Lagrange gmax of the whole field (difference of the principal
    strains, the tracker's definition)."""
    s = p.s_final if s is None else float(s)
    Y, X = np.mgrid[p.y0 + 0.5 * step:p.y0 + p.depth:step, p.xc - p.half_box + 0.5 * step:p.xc + p.half_box:step]
    G = GridField(p, X, Y)
    _, _, (a, b, c, d) = G.at(s)
    det = (1 + a) * (1 + d) - b * c
    edge, apex = _disc_masks(p, X, Y)
    disc = edge | apex
    se = float(p.s_e(s))
    B = G.B
    ea, eb, ec, ed = se * B["ex_x"], se * B["ex_y"], se * B["ey_x"], se * B["ey_y"]
    vol_e = (1 + ea) * (1 + ed) - eb * ec - 1.0
    gmax = _green_lagrange_gmax(a, b, c, d)
    gmax_e = _green_lagrange_gmax(ea, eb, ec, ed)
    sm = float(p.s_m(s))
    det_m = (1 + sm * B["mx_x"]) * (1 + sm * B["my_y"]) - sm * sm * B["mx_y"] * B["my_x"] - 1.0  # the mechanism alone
    P = prepeak_parts(p, X, Y)
    i = np.unravel_index(int(np.argmin(det)), det.shape)
    imax, imin = np.unravel_index(int(np.argmax(vol_e)), vol_e.shape), np.unravel_index(int(np.argmin(vol_e)), vol_e.shape)
    middle = (np.abs(X - p.xc) < 0.5 * p.b) & (Y > p.y0)
    return {"s": s, "det_f_min": float(det.min()), "det_f_min_at": (float(X[i] - p.xc), float(Y[i] - p.y0)),
            "vol_band_max": float((det - 1.0)[~disc].max()), "vol_band_min": float((det - 1.0)[~disc].min()),
            "vol_band_min_mech": float(det_m[~disc].min()),
            "vol_disc_max": float((det - 1.0)[disc].max()), "vol_edge_max": float((det - 1.0)[edge].max()),
            "vol_apex_min": float((det - 1.0)[apex].min()),
            "vol_pre_min": float(vol_e.min()), "vol_pre_max": float(vol_e.max()), "vol_pre_mean": float(vol_e.mean()),
            "vol_pre_min_at": (float(X[imin] - p.xc), float(Y[imin] - p.y0)),
            "vol_pre_max_at": (float(X[imax] - p.xc), float(Y[imax] - p.y0)),
            "vol_pre_edge_min": float(vol_e[edge].min()), "vol_pre_edge_max": float(vol_e[edge].max()),
            "vol_pre_out_min": float(vol_e[~edge].min()), "vol_pre_out_max": float(vol_e[~edge].max()),
            "vol_pre_share_below": float((vol_e < -0.005).mean()), "vol_pre_share_above": float((vol_e > 0.002).mean()),
            "vol_pre_share_above5": float((vol_e > 0.005).mean()),
            "div_iso_max": float(np.abs(P["isox_x"] + P["isoy_y"]).max()),
            "gmax_pre_max": float(gmax_e.max()), "gmax_pre_middle_max": float(gmax_e[middle].max()),
            "gmax_p999": float(np.percentile(gmax, 99.9)), "gmax_max": float(gmax.max())}


def band_fwhm(p: Params, s=None, t: float = 0.5, half: float = 4.0, dstep: float = 0.02):
    """Full width at half maximum of the gmax profile across the W|F band, measured along the normal through the point
    at the fraction t of the edge E - A [px].  (The linear estimate 1.08 w is 14.0 px for w = 13 px; the real profile of
    the blended velocities is a little narrower.)"""
    s = p.s_final if s is None else float(s)
    a = np.deg2rad(p.alpha_deg)
    ex, ey = p.b, 0.0
    px, py = ex + t * p.r_a * (-np.cos(a)), ey + t * p.r_a * np.sin(a)
    d = np.arange(-half * p.w, half * p.w + 0.5 * dstep, dstep)
    X, Y = p.xc + px + d * np.sin(a), p.y0 + py + d * np.cos(a)
    g = _green_lagrange_gmax(*FootingField(p, s).grad(X, Y))
    k = int(np.argmax(g))
    half_max = 0.5 * g[k]
    lo = k
    while lo > 0 and g[lo - 1] >= half_max:
        lo -= 1
    hi = k
    while hi < len(g) - 1 and g[hi + 1] >= half_max:
        hi += 1
    # linear interpolation of the two crossings
    xl = d[lo] if lo == 0 else d[lo - 1] + (half_max - g[lo - 1]) / (g[lo] - g[lo - 1]) * dstep
    xh = d[hi] if hi == len(g) - 1 else d[hi] + (g[hi] - half_max) / (g[hi] - g[hi + 1]) * dstep
    return float(xh - xl)


def prandtl_extent(phi_deg: float):
    """Extent of the classical Prandtl mechanism for the friction angle phi (wedge angle 45 + phi/2, outer slip line
    45 - phi/2, associated flow psi = phi, so the fan angle is 90 deg): (exit distance from the centre line / B,
    depth / B, apex depth / B), by the closed forms of `Params`."""
    q = Params(b=1.0, alpha_deg=45.0 + 0.5 * phi_deg, beta_deg=45.0 - 0.5 * phi_deg, psi_deg=phi_deg)
    return float(q.xe / q.B), float(q.depth_mech / q.B), float(q.ha / q.B)

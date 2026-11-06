"""Synthetic speckle images with exactly known displacement fields.

The pattern is a sum of Gaussian blobs evaluated analytically at inverse-mapped
coordinates, so the deformed image contains no interpolation error.
Displacements are Lagrangian: material point X moves to x = X + u(X).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pattern:
    cx: np.ndarray
    cy: np.ndarray
    r: np.ndarray
    a: np.ndarray


def make_pattern(h, w, seed=0, density=0.6, rmin=1.5, rmax=4.0, margin=80) -> Pattern:
    rng = np.random.default_rng(seed)
    rmean = 0.5 * (rmin + rmax)
    n = int(density * (h + 2 * margin) * (w + 2 * margin) / (np.pi * rmean**2))
    return Pattern(
        cx=rng.uniform(-margin, w + margin, n),
        cy=rng.uniform(-margin, h + margin, n),
        r=rng.uniform(rmin, rmax, n),
        a=rng.uniform(0.3, 1.0, n) * rng.choice([-1.0, 1.0], n),
    )


class Affine:
    """x = c + A (X - c) + t."""

    def __init__(self, A, t=(0.0, 0.0), c=(0.0, 0.0)):
        self.A = np.asarray(A, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.Ainv = np.linalg.inv(self.A)

    def fwd(self, X, Y):
        dx, dy = X - self.c[0], Y - self.c[1]
        A = self.A
        return (self.c[0] + A[0, 0] * dx + A[0, 1] * dy + self.t[0],
                self.c[1] + A[1, 0] * dx + A[1, 1] * dy + self.t[1])

    def inv(self, x, y):
        dx, dy = x - self.c[0] - self.t[0], y - self.c[1] - self.t[1]
        B = self.Ainv
        return self.c[0] + B[0, 0] * dx + B[0, 1] * dy, self.c[1] + B[1, 0] * dx + B[1, 1] * dy

    def grad(self, X, Y):
        """(du/dX, du/dY, dv/dX, dv/dY) on the reference grid."""
        G = self.A - np.eye(2)
        one = np.ones(np.shape(X))
        return G[0, 0] * one, G[0, 1] * one, G[1, 0] * one, G[1, 1] * one


def translation(tx, ty):
    return Affine(np.eye(2), (tx, ty))


def rotation(deg, center):
    t = np.deg2rad(deg)
    return Affine([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], c=center)


def uniform_strain(exx, eyy, center, exy=0.0, t=(0.0, 0.0)):
    return Affine([[1 + exx, exy], [exy, 1 + eyy]], t=t, c=center)


def simple_shear(gamma, center):
    return Affine([[1.0, gamma], [0.0, 1.0]], c=center)


class ShearBand:
    """Displacement 0.5*delta*tanh(s/width) along a straight band; s = distance across it."""

    def __init__(self, delta, width, center, angle_deg=0.0):
        a = np.deg2rad(angle_deg)
        self.tx, self.ty = np.cos(a), np.sin(a)  # along the band
        self.nx, self.ny = -np.sin(a), np.cos(a)  # across the band
        self.delta, self.width = delta, width
        self.c = np.asarray(center, dtype=np.float64)

    def _s(self, X, Y):
        return self.nx * (X - self.c[0]) + self.ny * (Y - self.c[1])

    def fwd(self, X, Y):
        f = 0.5 * self.delta * np.tanh(self._s(X, Y) / self.width)
        return X + f * self.tx, Y + f * self.ty

    def inv(self, x, y):
        f = 0.5 * self.delta * np.tanh(self._s(x, y) / self.width)  # s is unchanged by the motion
        return x - f * self.tx, y - f * self.ty

    def grad(self, X, Y):
        g = 0.5 * self.delta / self.width / np.cosh(self._s(X, Y) / self.width) ** 2
        return g * self.tx * self.nx, g * self.tx * self.ny, g * self.ty * self.nx, g * self.ty * self.ny


class Crack:
    """Straight crack with opening and slip (a displacement discontinuity).

    The line passes through `center` with unit normal n = (cos alpha, sin alpha). Negative side (n . (X - c) < 0):
    x = A_minus(X). Positive side: x = A_plus(X) + J, J = opening n + slip t, t = (-n_y, n_x), opening >= 0. A_minus,
    A_plus are Affine maps (None = identity), so a uniform strain can be given to one side only.
    inv returns NaN for points of the deformed image that no side maps to: the gap."""

    def __init__(self, center, alpha_deg, opening, slip, minus=None, plus=None):
        a = np.deg2rad(alpha_deg)
        self.c = np.asarray(center, dtype=np.float64)
        self.n = np.array([np.cos(a), np.sin(a)])
        self.t = np.array([-self.n[1], self.n[0]])
        self.J = opening * self.n + slip * self.t
        self.minus = minus if minus is not None else Affine(np.eye(2))
        self.plus = plus if plus is not None else Affine(np.eye(2))
        # how far (beyond the blob radius) a deformed blob can reach from its mapped centre: the jump plus slack
        self.reach = float(np.hypot(*self.J)) + 2.0

    def _s(self, X, Y):
        return self.n[0] * (X - self.c[0]) + self.n[1] * (Y - self.c[1])

    def fwd(self, X, Y):
        xm, ym = self.minus.fwd(X, Y)
        xp, yp = self.plus.fwd(X, Y)
        neg = self._s(X, Y) < 0
        return np.where(neg, xm, xp + self.J[0]), np.where(neg, ym, yp + self.J[1])

    def inv(self, x, y):
        Xm, Ym = self.minus.inv(x, y)
        Xp, Yp = self.plus.inv(x - self.J[0], y - self.J[1])
        in_m = self._s(Xm, Ym) < 0
        in_p = self._s(Xp, Yp) >= 0
        return (np.where(in_m, Xm, np.where(in_p, Xp, np.nan)),
                np.where(in_m, Ym, np.where(in_p, Yp, np.nan)))

    def grad(self, X, Y):
        gm, gp = self.minus.grad(X, Y), self.plus.grad(X, Y)
        neg = self._s(X, Y) < 0
        return tuple(np.where(neg, a, b) for a, b in zip(gm, gp))


def displacement(field, h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = field.fwd(xx, yy)
    return x - xx, y - yy


def render(pattern: Pattern, h, w, field=None, ss=4, gap_value=0.1) -> np.ndarray:
    """Image of the pattern after deformation `field` (None = reference), values in [0, 1].

    Sub-samples whose inverse map is NaN (the gap of a Crack) are painted gap_value (a final image value, not
    passed through the tanh); a pixel mixes its gap and pattern sub-samples by their share. A field that never
    returns NaN gives exactly the image it always did."""
    o = (np.arange(ss) + 0.5) / ss - 0.5
    ys = (np.arange(h)[:, None] + o[None, :]).ravel()
    xs = (np.arange(w)[:, None] + o[None, :]).ravel()
    Ys, Xs = np.meshgrid(ys, xs, indexing="ij")
    if field is None:
        X, Y = Xs, Ys
        px, py = pattern.cx, pattern.cy
    else:
        X, Y = field.inv(Xs, Ys)
        px, py = field.fwd(pattern.cx, pattern.cy)
    gap = ~(np.isfinite(X) & np.isfinite(Y))
    has_gap = bool(gap.any())
    if has_gap:
        X, Y = np.where(gap, 1e9, X), np.where(gap, 1e9, Y)  # far from every blob: no contribution
    reach = getattr(field, "reach", 0.0)
    acc = np.zeros_like(Xs)
    for k in range(len(pattern.cx)):
        rad = 5.2 * pattern.r[k] + 2.0 + reach
        j0 = max(int(np.floor((px[k] - rad + 0.5) * ss)), 0)
        j1 = min(int(np.ceil((px[k] + rad + 0.5) * ss)), w * ss)
        i0 = max(int(np.floor((py[k] - rad + 0.5) * ss)), 0)
        i1 = min(int(np.ceil((py[k] + rad + 0.5) * ss)), h * ss)
        if j0 >= j1 or i0 >= i1:
            continue
        dx = X[i0:i1, j0:j1] - pattern.cx[k]
        dy = Y[i0:i1, j0:j1] - pattern.cy[k]
        acc[i0:i1, j0:j1] += pattern.a[k] * np.exp(-(dx * dx + dy * dy) / (2 * pattern.r[k] ** 2))
    if not has_gap:
        img = acc.reshape(h, ss, w, ss).mean(axis=(1, 3))
        return 0.5 + 0.25 * np.tanh(img)
    n_gap = gap.reshape(h, ss, w, ss).sum(axis=(1, 3))
    frac = n_gap / float(ss * ss)
    img = acc.reshape(h, ss, w, ss).sum(axis=(1, 3)) / np.maximum(ss * ss - n_gap, 1)
    return (1.0 - frac) * (0.5 + 0.25 * np.tanh(img)) + frac * gap_value


def degrade(img, seed=0, noise=0.0, gain=1.0, offset=0.0, grad=(0.0, 0.0), bits=None):
    """Sensor noise (std as a fraction of full range), illumination change and quantisation."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    shade = 1.0 + grad[0] * (xx / w - 0.5) + grad[1] * (yy / h - 0.5)
    out = img * gain * shade + offset
    if noise > 0:
        out = out + np.random.default_rng(seed).normal(0.0, noise, img.shape)
    out = np.clip(out, 0.0, 1.0)
    if bits:
        q = 2**bits - 1
        out = np.round(out * q) / q
    return out


def scratches(h, w, n, seed, amp=(0.03, 0.08)):
    """Static thin bright lines (scratches on a glass wall): Gaussian cross-section (sigma 0.7 px),
    uniform orientation, length 50-300 px. Added identically to every frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w))
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        th = rng.uniform(0, np.pi)
        length = rng.uniform(50, 300)
        tx, ty = np.cos(th), np.sin(th)
        d = (xx - cx) * (-ty) + (yy - cy) * tx
        s = (xx - cx) * tx + (yy - cy) * ty
        img += rng.uniform(*amp) * np.exp(-d**2 / (2 * 0.7**2)) * (np.abs(s) < length / 2)
    return img

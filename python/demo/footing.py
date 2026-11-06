"""Synthetic model test of a strip footing settling into sand: scene geometry, renderer, video writer and truth.

The scene (full size 1280 x 720 px): a soil box (static walls with fiducial marks, static background), sand
(random speckle) and a rigid footing at the centre of the soil surface that is pushed down at a constant rate.
The soil follows the analytic displacement field of `demo.field` (a kinematically admissible failure mechanism:
rigid wedge, log-spiral fan, passive wedge with dilation, on top of a diffuse, nearly isochoric pre-peak flow); the frames are
rendered through the exact inverse map, so the truth is known to rendering accuracy:

* u(X, t) on the frame-0 grid (Lagrangian, X = position in frame 0) and the strain of that field, defined as the
  tracker defines them (Green-Lagrange E = (F^T F - I) / 2, gmax, vol of `optflow_ref.strain.strain_fields`);
* the settlement of every frame.

Pixel convention: pixel (row i, column j) has its centre at image coordinates (x, y) = (j, i), as everywhere in
optflow_ref.
"""
import shutil
import subprocess
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi

from optflow_ref.strain import strain_fields

from .field import FootingField, GridField, Params  # noqa: F401  (re-exported)

LUMA = np.array([0.299, 0.587, 0.114])
SS = 2  # supersampling of the sand texture table


# ---------------------------------------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scene:
    width: int
    height: int
    params: Params  # displacement field, in absolute pixel coordinates
    n_frames: int
    wall: float  # wall thickness
    foot_h: float  # height of the footing block
    rod_w: float  # width of the loading rod above it
    seed: int = 1
    noise: float = 0.004  # camera noise, standard deviation as a fraction of the full range
    flicker: float = 0.006  # per-frame gain fluctuation (standard deviation)
    fps: float = 15.0

    @property
    def box_l(self):
        return self.params.xc - self.params.half_box

    @property
    def box_r(self):
        return self.params.xc + self.params.half_box

    @property
    def y_surface(self):
        return self.params.y0

    @property
    def y_bottom(self):
        return self.params.y0 + self.params.depth

    def settlement(self, k):
        """Settlement of frame k: constant penetration rate (displacement-controlled test), 0 at frame 0."""
        return self.params.s_final * k / max(self.n_frames - 1, 1)

    def settlements(self):
        return np.array([self.settlement(k) for k in range(self.n_frames)])

    def roi_polygon(self):
        """ROI on frame 0: the soil inside the container, a small margin from the walls and the bottom, the top
        edge a few px below the initial surface (so the footing is excluded)."""
        m = max(4.0, 0.012 * self.width)
        top = self.y_surface + max(3.0, 0.02 * self.height)
        l, r, b = self.box_l + m, self.box_r - m, self.y_bottom - m
        return [[l, top], [r, top], [r, b], [l, b]]

    def roi_bbox(self):
        """Integer bounding box (x0, y0, x1, y1), half-open, of the ROI."""
        pts = np.array(self.roi_polygon())
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
        return int(x0), int(y0), int(min(x1, self.width)), int(min(y1, self.height))


def make_scene(size="full", n_frames=None, s_final=None, seed=1) -> Scene:
    """size 'full': 1280 x 720, settlement 30 px (s/B = 0.2) in 60 frames.  'small': 384 x 216, a geometry scaled by 0.3
    (footing, soil, container, band width, corner radius; the band width stays >= 4 px), 6 frames, settlement 6 px
    (s/B = 0.133), for smoke tests.  The small scene's wedge is only about 60 px wide, narrower than the tracker's filters,
    so a settlement of 9 px (s/B = 0.2, the full scene's final state) makes its strong bands swallow the whole wedge:
    the estimate is then lost at the last frame (and the tracker lags the strained soil by 5 % well before)."""
    if size == "full":
        sc, W, H, n_def, s_def = 1.0, 1280, 720, 60, 30.0
    elif size == "small":
        sc, W, H, n_def, s_def = 0.3, 384, 216, 6, 6.0
    else:
        raise ValueError(f"size must be 'full' or 'small', got {size!r}")
    base = Params()
    sf = s_def if s_final is None else float(s_final)
    w = max(4.0, base.w * sc)  # the band width: scaled, but never narrower than 4 px
    p = replace(base, xc=W / 2.0, y0=300.0 * sc, b=base.b * sc, depth=base.depth * sc, half_box=base.half_box * sc,
                w=w, rc=w, s_final=sf)
    return Scene(width=W, height=H, params=p, n_frames=int(n_frames or n_def), wall=30.0 * sc, foot_h=110.0 * sc,
                 rod_w=44.0 * sc, seed=seed)


# ---------------------------------------------------------------------------------------------------------
# textures and static layers
# ---------------------------------------------------------------------------------------------------------
def _norm(a):
    return (a - a.mean()) / a.std()


def sand_texture(h, w, rng):
    """Sand-like random speckle on a SS-times finer grid than the image, values around 0.5: band-limited
    noise (wavelengths 5-20 px), sparse coarse grains, slow blotches."""
    H, W = h * SS, w * SS
    n = rng.standard_normal((H, W)).astype(np.float32)
    fine = _norm(ndi.gaussian_filter(n, 1.2 * SS) - ndi.gaussian_filter(n, 3.4 * SS))
    blot = _norm(ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 14 * SS))
    imp = np.zeros((H, W), np.float32)
    n_g = int(0.004 * h * w)
    imp[rng.integers(0, H, n_g), rng.integers(0, W, n_g)] = rng.choice([-1.0, 1.0], n_g)
    grain = ndi.gaussian_filter(imp, 1.9 * SS)
    grain /= max(float(np.abs(grain).max()), 1e-9)
    return (0.52 + 0.095 * np.tanh(1.5 * fine) + 0.03 * blot + 0.16 * grain).astype(np.float32)


def steel_texture(h, w, rng):
    """Brushed steel of the footing, at image resolution; static in the footing's frame."""
    n = rng.standard_normal((h, w)).astype(np.float32)
    return (0.46 + 0.035 * _norm(ndi.gaussian_filter(n, (0.6, 5.0))) + 0.02 * _norm(ndi.gaussian_filter(n, 2.0))).astype(np.float32)


def _tint(rgb):
    c = np.asarray(rgb, dtype=np.float64)
    return c / float(LUMA @ c)  # luminance 1: composing L * tint keeps L the luminance of the frame


SAND_TINT, STEEL_TINT = _tint((0.93, 0.80, 0.60)), _tint((0.80, 0.86, 0.98))
WALL_TINT, BACK_TINT = _tint((0.82, 0.88, 0.92)), _tint((0.92, 0.94, 0.90))


def _rect_alpha(x, y, xl, xr, yt, yb):
    """Coverage of an axis-aligned rectangle at pixel centres: a 1 px ramp over the edge."""
    d = np.minimum.reduce([x - xl, xr - x, y - yt, yb - y])
    return np.clip(d + 0.5, 0.0, 1.0)


@dataclass
class Layers:
    """Everything that does not depend on the settlement, plus the tables the renderer samples."""
    L: np.ndarray  # static luminance (background, walls, fiducials), (H, W) float32
    T: np.ndarray  # static tint, (H, W, 3) float32
    sand: np.ndarray  # spline coefficients of the sand texture table
    steel: np.ndarray  # spline coefficients of the steel texture
    gain_map: np.ndarray  # static slow illumination gradient, (H, W)
    marks: list  # fiducial mark centres (x, y)


def build_layers(scene: Scene) -> Layers:
    rng = np.random.default_rng(scene.seed)
    W, H, p = scene.width, scene.height, scene.params
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # back wall: light, with a slow vertical gradient and a faint texture
    back = 0.74 - 0.06 * yy / H + 0.012 * _norm(ndi.gaussian_filter(rng.standard_normal((H, W)), 3.0))
    L = back
    T = np.broadcast_to(BACK_TINT, (H, W, 3)).copy()
    # container frame: left and right walls, the bottom
    wl, wr, yb, wt = scene.box_l, scene.box_r, scene.y_bottom, scene.wall
    frame_a = np.maximum.reduce([_rect_alpha(xx, yy, wl - wt, wl, p.y0 - 80 * H / 720, yb + wt),
                                 _rect_alpha(xx, yy, wr, wr + wt, p.y0 - 80 * H / 720, yb + wt),
                                 _rect_alpha(xx, yy, wl - wt, wr + wt, yb, yb + wt)])
    wall_L = 0.62 + 0.02 * _norm(ndi.gaussian_filter(rng.standard_normal((H, W)), 1.5))
    # dark lines along the inner and outer edges of the frame
    edge = np.exp(-(np.minimum.reduce([np.abs(xx - wl), np.abs(xx - wr)]) / 1.2) ** 2) * ((yy > p.y0 - 80 * H / 720) & (yy < yb))
    edge += np.exp(-((yy - yb) / 1.2) ** 2) * ((xx > wl) & (xx < wr))
    wall_L = wall_L - 0.35 * np.clip(edge, 0, 1)
    L = frame_a * wall_L + (1 - frame_a) * L
    T = frame_a[..., None] * WALL_TINT + (1 - frame_a[..., None]) * T
    # fiducial marks on the walls: dark disc with a bright centre (static)
    r = max(3.0, 7.0 * H / 720)
    marks = []
    for f in (0.12, 0.45, 0.80):
        marks += [(wl - wt / 2, p.y0 + f * p.depth), (wr + wt / 2, p.y0 + f * p.depth)]
    for f in (0.30, 0.70):
        marks.append((p.xc + (f - 0.5) * 2 * p.half_box, yb + wt / 2))
    for mx, my in marks:
        d = np.hypot(xx - mx, yy - my)
        disc = np.clip(r + 0.5 - d, 0, 1)
        dot = np.clip(0.38 * r + 0.5 - d, 0, 1)
        L = L * (1 - disc) + disc * (0.10 + 0.85 * dot)
    gain_map = 1.0 + 0.04 * (xx / W - 0.5) - 0.03 * (yy / H - 0.5)  # static slow illumination gradient
    sand = ndi.spline_filter(sand_texture(H, W, rng), order=3, output=np.float32)
    steel = ndi.spline_filter(steel_texture(H, W, rng), order=3, output=np.float32)
    return Layers(L.astype(np.float32), T.astype(np.float32), sand, steel, gain_map.astype(np.float32), marks)


# ---------------------------------------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------------------------------------
class Renderer:
    """Renders frames of a scene: the soil through the inverse map, the rigid footing on top."""

    def __init__(self, scene: Scene, layers: Layers = None):
        self.scene = scene
        self._layers = layers  # built on first use (the inverse map alone does not need them)
        p = scene.params
        # soil box in pixels: columns [c0, c1), rows [r0, r1); r0 leaves room for the heave above y0
        self.c0, self.c1 = int(np.ceil(scene.box_l)), int(np.floor(scene.box_r)) + 1
        self.r0 = int(np.floor(p.y0 - 0.06 * p.depth - 4))
        self.r1 = int(np.floor(scene.y_bottom)) + 1
        ys = np.arange(self.r0, self.r1, dtype=np.float64)
        xs = np.arange(self.c0, self.c1, dtype=np.float64)
        self.px, self.py = np.meshgrid(xs, ys)  # pixel centres of the soil region
        self._prev = None

    @property
    def layers(self) -> Layers:
        if self._layers is None:
            self._layers = build_layers(self.scene)
        return self._layers

    def inverse(self, s):
        """Material coordinates (X, Y) of every pixel of the soil region at settlement s: Newton's method with the exact
        Jacobian at every pixel, started from the previous frame's solution (about 0.6 s per full-size frame).  (Solving
        it every 2nd pixel and interpolating was tried: the field's footing edges and wedge tip are too fine for that,
        0.1 px errors at the passive wedge's exit.)"""
        f = FootingField(self.scene.params, s)
        X, Y = f.inv(self.px, self.py, X0=self._prev, tol=1e-9)
        self._prev = (X, Y)
        return X, Y

    def frame(self, k, s, rng):
        """uint8 RGB frame k at settlement s; rng: numpy Generator for this frame's noise."""
        sc, p, ly = self.scene, self.scene.params, self.layers
        H, W = sc.height, sc.width
        L = ly.L.copy()
        T = ly.T.copy()
        # soil: sand texture at the material position, inside the initial soil domain (anti-aliased edges)
        X, Y = self.inverse(s)
        crd = np.array([(SS * Y + (SS - 1) / 2.0).ravel(), (SS * X + (SS - 1) / 2.0).ravel()])
        tex = ndi.map_coordinates(ly.sand, crd, order=3, prefilter=False, mode="nearest").reshape(X.shape)
        a = np.minimum(np.clip(Y - p.y0 + 0.5, 0, 1), np.clip(sc.y_bottom - Y + 0.5, 0, 1))
        a = np.minimum(a, np.clip(X - sc.box_l + 0.5, 0, 1))
        a = np.minimum(a, np.clip(sc.box_r - X + 0.5, 0, 1))
        sl = np.s_[self.r0:self.r1, self.c0:self.c1]
        L[sl] = a * tex + (1 - a) * L[sl]
        T[sl] = a[..., None] * SAND_TINT + (1 - a[..., None]) * T[sl]
        self._footing(L, T, s)
        gain = 1.0 + sc.flicker * rng.standard_normal()
        img = L * ly.gain_map * gain
        img = img + sc.noise * rng.standard_normal((H, W)).astype(np.float32)
        rgb = np.clip(img[..., None] * T, 0.0, 1.0)
        return np.rint(rgb * 255.0).astype(np.uint8)

    def _footing(self, L, T, s):
        """The rigid footing: a block that moved down by s and the loading rod above it."""
        sc, p, ly = self.scene, self.scene.params, self.layers
        x0, x1 = int(p.xc - p.b) - 2, int(p.xc + p.b) + 3
        y_top, y_bot = 0, int(p.y0 + s + 3)
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y_top, y_bot, dtype=np.float64)
        x, y = np.meshgrid(xs, ys)
        base = p.y0 + s
        block = _rect_alpha(x, y, p.xc - p.b, p.xc + p.b, base - sc.foot_h, base)
        rod = _rect_alpha(x, y, p.xc - sc.rod_w / 2, p.xc + sc.rod_w / 2, -10.0, base - sc.foot_h + 1.0)
        a = np.maximum(block, rod)
        yf = y - s  # footing frame: the texture and the marks move with the footing
        y_off = max(40.0, p.s_final + 10.0)  # keeps the texture's row index positive
        st = ndi.map_coordinates(ly.steel, np.array([(yf + y_off).ravel(), x.ravel()]), order=3, prefilter=False,
                                 mode="nearest").reshape(x.shape)
        # dark outline, tick marks on the rod, a bright bolt row on the block
        edge = np.minimum.reduce([x - (p.xc - p.b), (p.xc + p.b) - x, y - (base - sc.foot_h), base - y])
        edge = np.where(block > 0.5, edge, 99.0)
        st = st - 0.18 * np.exp(-((edge - 1.5) / 1.5) ** 2)
        tick = (np.abs(((yf + 1000.0) % (40.0 * sc.foot_h / 110.0 + 1e-9)) - 0.0) < 1.6) & (rod > 0.5) & (block < 0.5)
        st = np.where(tick, st - 0.25, st)
        for bx in (-0.8, -0.4, 0.0, 0.4, 0.8):
            d = np.hypot(x - (p.xc + bx * p.b), yf - (-0.5 * sc.foot_h))
            st = st + 0.22 * np.clip(3.2 * sc.foot_h / 110.0 + 0.5 - d, 0, 1)
        sl = np.s_[y_top:y_bot, x0:x1]
        L[sl] = a * st + (1 - a) * L[sl]
        T[sl] = a[..., None] * STEEL_TINT + (1 - a[..., None]) * T[sl]


# ---------------------------------------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------------------------------------
def ffmpeg_path():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg not found on PATH (needed to write and read the demo videos)")
    return exe


class VideoWriter:
    """H.264 (libx264, yuv420p) video written through a pipe to ffmpeg; frames are uint8 RGB (H, W, 3)."""

    def __init__(self, path, width, height, fps=15.0, crf=12):
        self.path = path
        self.proc = subprocess.Popen(
            [ffmpeg_path(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {err.strip()[-400:]}")

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait()


def generate_video(scene: Scene, path, progress=None):
    """Render all frames of the scene into an H.264 video; returns the settlement of every frame and the
    first-frame RGB image (for quick looks)."""
    r = Renderer(scene)
    s_k = scene.settlements()
    first = None
    with VideoWriter(path, scene.width, scene.height, scene.fps) as vw:
        for k, s in enumerate(s_k):
            rgb = r.frame(k, float(s), np.random.default_rng([scene.seed, 1000 + k]))
            if first is None:
                first = rgb
            vw.write(rgb)
            if progress:
                progress(k, len(s_k))
    return s_k, first


# ---------------------------------------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------------------------------------
TRUTH_KEYS = ("ux", "uy", "Exx", "Eyy", "Exy", "gmax", "vol")


class Truth:
    """True displacement and strain on the frame-0 pixel grid inside the ROI's bounding box (x0, y0, x1, y1)."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.bbox = scene.roi_bbox()
        x0, y0, x1, y1 = self.bbox
        Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        self.grid = GridField(scene.params, X, Y)

    def rigid_mask(self, tol=1e-6):
        """Pixels of the bounding box inside the rigid wedge under the footing, where the mechanism part U_m of the field is
        exactly the translation (0, 1) (value and gradient within `tol`): there the soil follows the footing, apart from
        the slowly varying pre-peak flow U_e (u_y = K(y') < 1 below the base).  The median u_y of the estimate over these pixels is the
        estimated settlement (compared with the true median over the same pixels)."""
        B = self.grid.B
        m = (np.abs(B["mx"]) < tol) & (np.abs(B["my"] - 1.0) < tol)
        for name in ("mx", "my"):
            m &= (np.abs(B[name + "_x"]) < tol) & (np.abs(B[name + "_y"]) < tol)  # and no strain
        return m

    def frame(self, s):
        """dict ux, uy (px), Exx, Eyy, Exy, gmax, vol of settlement s, shape of the bounding box."""
        ux, uy, gr = self.grid.at(s)
        st = strain_fields(*gr)
        out = {"ux": ux, "uy": uy}
        out.update({k: st[k] for k in ("Exx", "Eyy", "Exy", "gmax", "vol")})
        return out

    def save(self, path, frames=None):
        """Write the settlement history and the settlement-independent shape functions (float32, compressed): with
        them any frame's truth is s_e Psi_e + s_m Psi_m (see demo.field; s_e, s_m are stored too).  frames: indices whose
        fields are stored in full as well (a few frames; the whole sequence would be gigabytes)."""
        sc = self.scene
        s_k = sc.settlements()
        data = {"settlement": s_k, "s_e": sc.params.s_e(s_k), "s_m": sc.params.s_m(s_k), "bbox": np.array(self.bbox),
                "xc": sc.params.xc, "y0": sc.params.y0}
        data.update({f"basis_{k}": v.astype(np.float32) for k, v in self.grid.B.items()})
        for k in frames or ():
            for name, a in self.frame(float(sc.settlement(k))).items():
                data[f"frame{k:05d}_{name}"] = a.astype(np.float32)
        np.savez_compressed(path, **data)

"""Automatic ROI crop for the command line: track on the ROI's bounding box plus a margin instead of
the whole image, and write full-size outputs (NaN outside the crop, confidence 0, valid 0).

The rectangle is in pixels of the scaled image, half-open: [x0, x1) x [y0, y1). It starts at the ROI
mask's bounding box, grows by the margin on every side, has x0 and y0 rounded down to a multiple of
A = grid_step * 2**L (L = the pyramid depth POC uses for this config and image, so the coarse grid of
the crop lies on the coarse grid of the whole image) and is clipped to the image.
"""
import json
import math
from dataclasses import dataclass

import numpy as np

from .phaseflow import PAD
from .poc import n_levels
from .track import Result

WARNING = ("warning: the material moves close to the crop border — "
           "increase --crop-margin or use --crop off")
WARN_FRACTION = 0.01  # warn when more than this share of the ROI pixels is lost at the crop border
OFF_JSON = json.dumps({"mode": "off"}, indent=2)


def check_margin(margin) -> None:
    if margin is not None and margin < 0:
        raise ValueError(f"--crop-margin must be >= 0, got {margin}")


def default_margin(cfg, shape, seq=False) -> int:
    """Largest displacement + the fine stage's averaging window (4 sigma_w) + the guard band + the
    response padding (phaseflow.PAD) + the footprint of the coarse stage: its grid points sit
    poc_win / 2 inside the image at the coarsest pyramid level and the outlier filter looks one grid
    step further, (poc_win / 2 + grid_step) * 2**L at full resolution, so the coarse grid of the crop
    sees the same context as in the whole image of `shape` = (height, width). A sequence needs twice
    that: the material moves cumulatively."""
    L = n_levels(cfg.max_disp, cfg.poc_win, shape)
    m = (math.ceil(cfg.max_disp) + math.ceil(4 * cfg.sigma_w) + math.ceil(cfg.edge_guard_px()) + PAD
         + (cfg.poc_win // 2 + cfg.grid_step) * 2**L)
    return 2 * m if seq else m


@dataclass(frozen=True)
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int
    margin: int

    @property
    def shape(self):
        return self.y1 - self.y0, self.x1 - self.x0

    def apply(self, a):
        """A copy of the part of a full-size array (H, W, ...) inside the crop."""
        return a[self.y0:self.y1, self.x0:self.x1].copy()

    def json(self) -> str:
        return json.dumps({"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                           "margin": self.margin, "mode": "auto"}, indent=2)

    def embed(self, res, shape) -> Result:
        """A track.Result of the crop as one of the full image `shape`: NaN outside the crop, confidence 0."""
        def put(a, fill):
            full = np.full(shape, fill, dtype=a.dtype)
            full[self.y0:self.y1, self.x0:self.x1] = a
            return full

        return Result(put(res.ux, np.nan), put(res.uy, np.nan), put(res.conf, 0.0),
                      {k: put(v, np.nan) for k, v in res.strain.items()},
                      put(res.raw_ux, np.nan), put(res.raw_uy, np.nan), res.timing,
                      None if res.fb is None else put(res.fb, 0),
                      None if res.crack is None else {k: put(v, False if v.dtype == bool else np.nan)
                                                      for k, v in res.crack.items()})


def plan(mask, cfg, margin=None, seq=False) -> Crop:
    """The crop for an ROI mask (H, W). margin=None: default_margin(cfg, mask.shape, seq)."""
    H, W = mask.shape
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        raise ValueError("ROI mask is empty")
    rows = np.flatnonzero(mask.any(axis=1))
    check_margin(margin)
    if min(H, W) < cfg.poc_win:
        raise ValueError(f"image smaller than poc_win ({cfg.poc_win}): {(H, W)}")
    if margin is None:
        margin = default_margin(cfg, mask.shape, seq)
    A = cfg.grid_step * 2 ** n_levels(cfg.max_disp, cfg.poc_win, mask.shape)
    x0 = max(int(cols[0]) - margin, 0) // A * A
    y0 = max(int(rows[0]) - margin, 0) // A * A
    c = Crop(x0, y0, min(int(cols[-1]) + 1 + margin, W), min(int(rows[-1]) + 1 + margin, H), margin)
    h, w = c.shape
    if min(h, w) < cfg.poc_win:
        raise ValueError(f"the crop {w}x{h} is smaller than poc_win ({cfg.poc_win}): "
                         "increase --crop-margin or use --crop off")
    return c


def border_fraction(raw_ux, raw_uy, mask, crop, shape, cfg) -> float:
    """Share of the ROI pixels whose match lies within B = 4 sigma_w of the crop border (or outside it),
    which phaseflow's border rule sets to confidence 0. raw_ux, raw_uy: the unmasked displacement of the
    crop, mask: its ROI. Only sides the crop introduced count (not those on the border of the image of
    `shape`, which are lost without a crop as well); 0.0 when there are none."""
    H, W = shape
    sides = (crop.x0 > 0, crop.y0 > 0, crop.x1 < W, crop.y1 < H)
    if not any(sides) or not mask.any():
        return 0.0
    h, w = mask.shape
    B = 4.0 * cfg.sigma_w if cfg.edge_guard_px() > 0 else 0.0  # as phaseflow.refine
    yy, xx = np.mgrid[0:h, 0:w]
    tx, ty = xx + raw_ux, yy + raw_uy
    ok = np.isfinite(tx) & np.isfinite(ty)  # NaN compares false: no match counts as outside
    for introduced, dist in zip(sides, (tx, ty, w - 1 - tx, h - 1 - ty)):
        if introduced:
            ok &= dist >= B
    return float(np.count_nonzero(mask & ~ok)) / float(np.count_nonzero(mask))

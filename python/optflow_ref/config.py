"""Processing parameters shared by the Python reference and the Rust implementation."""
import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_finite(v) -> bool:
    return _is_number(v) and math.isfinite(v)


# Python-only keys (the Rust config does not know them): see Config.to_json
CRACK_KEYS = ("crack_split", "split_tol", "split_gain", "split_from_outer", "crack_out", "crack_r", "crack_a",
              "crack_sides", "side_from_outer", "crack_guard_frac", "side_band", "crack_model", "xdic_RB", "xdic_iters")


@dataclass
class Config:
    # coarse stage: windowed phase-only correlation on a grid
    grid_step: int = 16
    poc_win: int = 64
    max_disp: float = 32.0  # expected max displacement [px]; sets the pyramid depth
    poc_peak_min: float = 0.1
    median_thresh: float = 2.0
    # fine stage: local phase of a complex Gabor bank
    lam: float = 8.0
    scales: tuple = (0.6, 1.0, 1.6)  # filter wavelengths are lam * scales
    n_orient: int = 4
    bandwidth: float = 0.5
    # DC cut of the filter bank: high-pass width = dc_cut * lam * max(scales) / (2 pi). Larger keeps
    # more low-frequency texture (better white-noise efficiency) but widens the filters' footprint,
    # spreading contamination from cracks and edges further.
    dc_cut: float = 2.0
    sigma_w: float = 4.0
    shape_order: int = 1  # local displacement model: 1 affine, 2 quadratic (sharper bands, more noise)
    sigma_s: float = 4.0
    outer_iters: int = 8
    inner_iters: int = 2
    # validity and strain
    conf_min: float = 0.2
    # guard distance [px]: pixels closer than this to a guard source are set invalid, because the
    # filters straddle the discontinuity there. Guard sources: large low-confidence components (>=
    # edge_guard_min_area, e.g. cracks, holes), everything outside the ROI, and the outside of the
    # image (a pixel is invalid when min(x, y, W-1-x, H-1-y) < edge_guard). Isolated small
    # low-confidence spots are not sources. < 0: automatic = 3 x the spatial sigma of the
    # longest-wavelength filter; 0: off. While the guard is on, phaseflow.refine also gives confidence 0 to
    # the pixels within B = 4 sigma_w of image 1's border and to those whose match is within B of image 2's
    # border or outside it (the responses there come from reflect padding, which the residual cannot see):
    # those strips are large low-confidence components and get the guard band as well, so the result starts
    # ~B + guard (28 px at the defaults) from a border. With the guard off only matches outside image 2 get 0.
    edge_guard: float = -1.0
    # only invalid regions at least this large [px^2] are guarded (plus the ROI exterior): isolated
    # low-confidence speckles do not contaminate their neighbours, cracks and large holes do.
    # < 0: automatic = 2 x (longest wavelength)^2 (~330 px^2 at the defaults).
    edge_guard_min_area: float = -1.0
    # [rad] conf = texture * exp(-(phase residual / phase_tol)^2). Correct matches give a residual
    # of ~0.07 / 0.15 / 0.2 rad at 1 % / 2 % noise / real photos; wrong ones 0.7-1.6 rad.
    phase_tol: float = 0.4
    strain_win: int = 15
    # sequences
    ref_update_conf: float = 0.5
    # sequence mode: "chain" (a direct match against a reference frame that moves when the mean confidence drops;
    # the output confidence is the minimum over the chain) or "hybrid" (direct-first: the chain value wherever the
    # direct link is confident, else the adjacent match k-1 -> k carries the pixel for at most fb_max consecutive
    # frames).
    seq_mode: str = "chain"
    fb_max: int = 10
    # hybrid v3 (report section "v3"): a fallback pixel is valid only if its displacement agrees (within fb_tol px)
    # with an affine model fitted to the direct-good pixels within fb_radius px (-1: 4 sigma_w) of it, of which at least
    # fb_min_support are needed. fb_min_support = 0 switches the gate off (= v2).
    fb_radius: float = -1.0
    fb_min_support: int = 20
    fb_tol: float = 0.5
    # temporal averaging front-end of `seq` (report section "temporal averaging"): each sampled frame is the mean of
    # the 2 tavg + 1 frames around it (clipped to the video); tavg_align first aligns every neighbour to the centre frame
    # with an affine motion fitted to the coarse POC flow. tavg = 0: off, frames untouched.
    tavg: int = 0
    tavg_align: bool = False
    # motion model of the alignment: "affine" (one robust affine fit to the coarse POC flow inside the ROI) or "dense"
    # (the coarse POC flow itself, bilinearly interpolated: follows localised deformation, noisier)
    tavg_model: str = "affine"
    # Rust only (the Python reference does not implement it and ignores the value): the low-confidence pixels of the
    # pre-warp of outer iteration K are kept for every later outer iteration of the window (env OPTFLOW_PREWARP_FREEZE_AT). 0: off.
    prewarp_freeze_at: int = 0
    # crack handling (Python reference only, off by default, experimental).
    # crack_split: where the full-window phase residual exceeds split_tol [rad], the local solve also tries 8 half
    # windows (the half plane n_h . d >= 0) and adopts the best one if its residual is below split_gain x the full
    # one; from outer iteration split_from_outer on. crack_out: also report the crack band, its normal, COD and CSD
    # and compute the strain one-sided near the crack; crack_r [px] is the radius the normal is averaged over,
    # crack_a [px] the distance from the crack centre at which the displacement on either side is read.
    crack_split: bool = False
    split_tol: float = 0.25
    split_gain: float = 0.7
    split_from_outer: int = 0
    crack_out: bool = False
    crack_r: float = 8.0
    crack_a: float = 12.0
    # stage 2 (spec section 6): crack_sides also builds the filter responses, the pre-warp and the LS weights per side of
    # the crack from outer iteration side_from_outer on (needs crack_split); crack_guard_frac: share of the edge guard
    # that still applies around the low-confidence region of a crack (was the constant CRACK_GUARD_FRAC = 0.5).
    crack_sides: bool = False
    side_from_outer: int = 2
    crack_guard_frac: float = 0.5
    # which residual defines the crack band C (spec 6.1) of crack_sides: "full" = resid_full > split_tol (a band of +-14 px
    # around the crack, all of it invalid), "post" = the residual after the half-window selection (+-8 px)
    side_band: str = "post"
    # crack_model (optflow_ref/xdic.py): "xdic"
    # detects crack lines as jumps of the baseline field and re-solves the pixels within xdic_RB px of a line per side of
    # it (filters, pre-warp and LS weights of one side only, the other side filled with a constant); the line is moved
    # xdic_iters times by reassigning the pixels near it to the side whose motion explains their phase better.
    # xdic_RB < 0: automatic = auto_edge_guard_px() + 4 sigma_w (28 px at the defaults). Not combined with crack_split.
    crack_model: str = "off"
    xdic_RB: float = -1.0
    xdic_iters: int = 2

    def to_json(self) -> str:
        """The config as JSON. The Python-only crack keys are left out while they have their default value, so the JSON
        of a default (or crack-free) config is exactly what it was before they existed and the Rust CLI, which rejects
        unknown keys, still reads it; from_json restores the defaults of missing keys."""
        d = asdict(self)
        default = type(self)()
        for k in CRACK_KEYS:
            if d[k] == getattr(default, k):
                del d[k]
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Config":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        defaults = cls()
        for key, v in data.items():
            d = getattr(defaults, key)
            if isinstance(d, tuple):
                if not isinstance(v, list) or not all(_is_number(x) for x in v):
                    raise ValueError(f"config key {key!r} must be a list of numbers, got {v!r}")
                data[key] = tuple(float(x) for x in v)
            elif isinstance(d, bool):
                if not isinstance(v, bool):
                    raise ValueError(f"config key {key!r} must be true or false, got {v!r}")
            elif isinstance(d, str):
                if not isinstance(v, str):
                    raise ValueError(f"config key {key!r} must be a string, got {v!r}")
            elif isinstance(d, int):
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError(f"config key {key!r} must be an integer, got {v!r}")
            elif not _is_number(v):
                raise ValueError(f"config key {key!r} must be a number, got {v!r}")
            else:
                data[key] = float(v)
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.seq_mode not in ("chain", "hybrid"):
            raise ValueError(f"seq_mode must be 'chain' or 'hybrid', got {self.seq_mode!r}")
        if not self.scales:
            raise ValueError("scales must not be empty")
        if self.poc_win < 16 or self.poc_win & (self.poc_win - 1):
            raise ValueError("poc_win must be a power of two >= 16")
        if self.grid_step < 1 or self.strain_win < 5 or self.strain_win % 2 == 0:
            raise ValueError("grid_step >= 1 and strain_win odd >= 5 required")
        if self.dc_cut <= 0:
            raise ValueError("dc_cut must be positive")
        if self.phase_tol <= 0:
            raise ValueError("phase_tol must be positive")
        if self.shape_order not in (1, 2):
            raise ValueError("shape_order must be 1 or 2")
        if self.lam * min(self.scales) < 3 or self.sigma_w <= 0 or self.sigma_s <= 0:
            raise ValueError("shortest wavelength lam*min(scales) must be >= 3 px; sigmas must be positive")
        for name in ("max_disp", "poc_peak_min", "median_thresh", "lam", "bandwidth", "dc_cut", "sigma_w",
                     "sigma_s", "conf_min", "edge_guard", "edge_guard_min_area", "phase_tol", "ref_update_conf"):
            if not _is_finite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        if not all(_is_finite(x) for x in self.scales):
            raise ValueError("scales must be finite numbers")
        if self.bandwidth <= 0 or self.max_disp <= 0 or self.median_thresh <= 0:
            raise ValueError("bandwidth, max_disp and median_thresh must be positive")
        if self.n_orient < 1 or self.outer_iters < 1 or self.inner_iters < 1:
            raise ValueError("n_orient, outer_iters and inner_iters must be >= 1")
        if not 0 <= self.conf_min <= 1 or not 0 <= self.poc_peak_min <= 1:
            raise ValueError("conf_min and poc_peak_min must lie in [0, 1]")
        if self.fb_max < 0:
            raise ValueError("fb_max must be >= 0")
        if not _is_finite(self.fb_radius) or not _is_finite(self.fb_tol) or self.fb_tol <= 0 or self.fb_radius == 0:
            raise ValueError("fb_radius must be finite and nonzero (negative: 4 sigma_w), fb_tol finite and positive")
        if self.fb_min_support < 0 or self.tavg < 0 or self.tavg > 64:
            raise ValueError("fb_min_support must be >= 0 and tavg in [0, 64]")
        if self.tavg_model not in ("affine", "dense"):
            raise ValueError(f"tavg_model must be 'affine' or 'dense', got {self.tavg_model!r}")
        if not isinstance(self.tavg_align, bool):
            raise ValueError("tavg_align must be true or false")
        if self.prewarp_freeze_at < 0:
            raise ValueError("prewarp_freeze_at must be >= 0")
        for name in ("crack_split", "crack_out", "crack_sides"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if not _is_finite(self.split_tol) or self.split_tol <= 0:
            raise ValueError("split_tol must be finite and positive")
        if not _is_finite(self.split_gain) or not 0 < self.split_gain <= 1:
            raise ValueError("split_gain must lie in (0, 1]")
        if not 0 <= self.split_from_outer < self.outer_iters:
            raise ValueError("split_from_outer must lie in [0, outer_iters)")
        if not _is_finite(self.crack_r) or self.crack_r <= 0 or not _is_finite(self.crack_a) or self.crack_a <= 0:
            raise ValueError("crack_r and crack_a must be finite and positive")
        if self.crack_sides and not self.crack_split:
            raise ValueError("crack_sides needs crack_split")
        if self.crack_sides and not 0 < self.side_from_outer < self.outer_iters:
            raise ValueError("side_from_outer must lie in (0, outer_iters)")
        if self.side_band not in ("full", "post"):
            raise ValueError(f"side_band must be 'full' or 'post', got {self.side_band!r}")
        if not _is_finite(self.crack_guard_frac) or not 0 <= self.crack_guard_frac <= 1:
            raise ValueError("crack_guard_frac must lie in [0, 1]")
        if self.crack_model not in ("off", "xdic"):
            raise ValueError(f"crack_model must be 'off' or 'xdic', got {self.crack_model!r}")
        if not _is_finite(self.xdic_RB) or self.xdic_RB == 0 or self.xdic_RB > 256:
            raise ValueError("xdic_RB must be finite, nonzero and <= 256 (negative: automatic)")
        if isinstance(self.xdic_iters, bool) or not isinstance(self.xdic_iters, int) or not 0 <= self.xdic_iters <= 8:
            raise ValueError("xdic_iters must be an integer in [0, 8]")
        if self.crack_model == "xdic" and (self.crack_split or self.crack_sides):
            raise ValueError("crack_model 'xdic' cannot be combined with crack_split / crack_sides")
        # upper bounds that keep the memory per tile sane (an absurd value would abort on allocation); last, so
        # every message above keeps its place
        if (self.n_orient * len(self.scales) > 64 or self.sigma_w > 256 or self.sigma_s > 256
                or self.strain_win > 255 or self.poc_win > 1024):
            raise ValueError("at most 64 filters (n_orient * len(scales)); sigma_w, sigma_s <= 256, "
                             "strain_win <= 255 and poc_win <= 1024 required")
        # the grid step enters the crop alignment grid_step * 2**L, the guard sets the tile margin: both bounded
        # so that no product or sum of them can overflow a 64-bit size; after the bounds, so their message keeps
        # its place
        if self.grid_step > 4096 or self.edge_guard_px() > 256:
            raise ValueError("grid_step <= 4096 and an edge guard (edge_guard, or the automatic one from lam, "
                             "scales and bandwidth) <= 256 px required")

    def xdic_rb_px(self) -> float:
        """Resolved R_B [px] of crack_model "xdic": the half width of the strip around a crack line solved per side."""
        if self.xdic_RB > 0:
            return float(self.xdic_RB)
        return self.auto_edge_guard_px() + 4.0 * self.sigma_w

    def fb_radius_px(self) -> float:
        """Resolved support radius [px] of the hybrid fallback gate."""
        return float(self.fb_radius) if self.fb_radius > 0 else 4.0 * self.sigma_w

    def edge_guard_min_area_px(self) -> float:
        """Resolved minimum area [px^2] of an invalid region that gets a guard band."""
        if self.edge_guard_min_area >= 0:
            return float(self.edge_guard_min_area)
        return 2.0 * (self.lam * max(self.scales)) ** 2

    def auto_edge_guard_px(self) -> float:
        """The automatic guard distance [px]: 3 x the spatial sigma of the longest-wavelength filter, the reach of
        a discontinuity into the phase. Independent of the edge_guard the user chose."""
        return 3.0 * self.lam * max(self.scales) / (2 * np.pi * self.bandwidth)

    def edge_guard_px(self) -> float:
        """Resolved guard distance in pixels."""
        if self.edge_guard >= 0:
            return float(self.edge_guard)
        return self.auto_edge_guard_px()

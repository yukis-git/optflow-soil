"""Synthetic benchmark cases (spec section 4.2)."""
from dataclasses import dataclass, field

import numpy as np

from optflow_ref import synth
from optflow_ref.config import Config

CRACK_SEED = 77  # pattern that replaces the central band of the deformed frame in the crack case


@dataclass
class Case:
    name: str
    group: str
    params: dict
    fields: list  # true deformation of frames 1..n relative to frame 0
    cfg: Config = field(default_factory=Config)
    margin: int = 32
    noise: float = 0.0
    bits: int = 0
    illum: dict = field(default_factory=dict)  # degrade() kwargs applied to the deformed frames
    video: bool = False
    overlay: dict = field(default_factory=dict)  # synth.scratches() kwargs, added to every frame (incl. the reference)
    crack_width: int = 0  # > 0: central vertical band of this width in the deformed frame shows another pattern


def all_cases(n):
    c = (n / 2, n / 2)
    out = []
    for s in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        out.append(Case(f"shift_{s:.1f}", "subpixel", {"tx": float(s), "ty": 0.3}, [synth.translation(s, 0.3)]))
    for d in (0.5, 2.0, 5.0):
        out.append(Case(f"rot_{d}", "rotation", {"deg": d}, [synth.rotation(d, c)]))
    for e in (1e-4, 1e-3, 1e-2, 1e-1):
        out.append(Case(f"uniform_{e:g}", "uniform", {"exx": e, "eyy": -0.3 * e},
                        [synth.uniform_strain(e, -0.3 * e, c)]))
    for g in (1e-3, 1e-2, 1e-1):
        out.append(Case(f"shear_{g:g}", "simple_shear", {"gamma": g}, [synth.simple_shear(g, c)]))
    for d, w in ((2, 2), (2, 5), (4, 10), (6, 5), (4, 20)):
        for order in (1, 2):
            out.append(Case(f"band_D{d}_w{w}_o{order}", "shear_band",
                            {"delta": d, "width": w, "shape_order": order, "angle": 10},
                            [synth.ShearBand(d, w, c, angle_deg=10)], Config(shape_order=order), margin=40))
    for nz in (0.005, 0.01, 0.02):
        out.append(Case(f"noise_{nz:g}", "noise", {"noise": nz}, [synth.uniform_strain(0.01, -0.003, c)],
                        noise=nz, bits=8))
    out.append(Case("illumination", "illumination", {"gain": 1.2, "offset": -0.05, "grad": [0.2, 0.1]},
                    [synth.uniform_strain(0.01, -0.003, c)], illum={"gain": 1.2, "offset": -0.05, "grad": (0.2, 0.1)}))
    out.append(Case("static_scratches", "overlay", {"n_scratches": 25, "amp": [0.03, 0.08], "exx": 0.01, "eyy": -0.003,
                                                    "tx": 20.3, "ty": 3.1},
                    [synth.uniform_strain(0.01, -0.003, c, t=(20.3, 3.1))], Config(max_disp=64), margin=40,
                    overlay={"n": 25, "seed": 1}))
    out.append(Case("crack", "crack", {"width": 16, "tx": 12.3, "ty": 1.7, "other_pattern_seed": CRACK_SEED},
                    [synth.translation(12.3, 1.7)], margin=40, crack_width=16))
    out.append(Case("large_motion", "large", {"tx": 40.3, "ty": -25.7, "exx": 0.02, "eyy": -0.01},
                    [synth.uniform_strain(0.02, -0.01, c, t=(40.3, -25.7))], Config(max_disp=64), margin=40))
    sc = n / 512  # the 512 px setup scaled to the image size: the coarse stage fails on the motion (spec 2.3)
    t = (91.7 * sc, -67.3 * sc)
    out.append(Case("coarse_failure", "coarse_failure", {"tx": t[0], "ty": t[1], "exx": 0.01, "eyy": -0.003,
                                                          "max_disp": 128 * sc},
                    [synth.uniform_strain(0.01, -0.003, c, t=t)], Config(max_disp=128 * sc), margin=40))
    seq = [synth.uniform_strain(0.001 * k, -0.0003 * k, c, t=(1.0 * k, 0.5 * k)) for k in range(1, 21)]
    out.append(Case("sequence_20", "sequence", {"frames": 20, "step_px": 1.0, "step_strain": 0.001}, seq, margin=40))
    out.append(Case("video_20", "sequence", {"frames": 20, "codec": "MJPG"}, seq, margin=40, video=True))
    return out

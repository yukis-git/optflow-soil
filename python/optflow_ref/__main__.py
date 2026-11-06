"""Command line: python -m optflow_ref {pair,seq} ...  (mirrors the planned Rust CLI)."""
import argparse
import itertools
import os
import sys
from dataclasses import fields

import numpy as np

from .config import Config
from .crop import OFF_JSON, WARN_FRACTION, WARNING, border_fraction, check_margin, plan
from .io import iter_frames, load_mask, load_roi, polygon_mask, read_image, downscale, save_result
from .poc import capture_range
from .track import track_pair, track_sequence
from .viz import save_maps


def _parse_override(cfg, kv):
    """Apply one `key=value` override, parsed by the type of the field's default."""
    if "=" not in kv:
        raise ValueError(f"--set expects key=value (e.g. --set sigma_w=6), got {kv!r}")
    k, v = kv.split("=", 1)
    valid = [f.name for f in fields(Config)]
    if k not in valid:
        raise ValueError(f"unknown config key {k!r}; valid keys: {', '.join(valid)}")
    cur = getattr(Config(), k)
    try:
        if isinstance(cur, bool):
            if v.lower() not in ("true", "false", "1", "0"):
                raise ValueError
            val = v.lower() in ("true", "1")
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in v.split(","))
        else:
            val = type(cur)(v)  # int("2.5") and int("2.0") fail: integer fields need an integer
    except ValueError:
        kind = ("a comma-separated list of numbers" if isinstance(cur, tuple)
                else "true or false" if isinstance(cur, bool) else f"{type(cur).__name__}")
        raise ValueError(f"--set {k}: cannot read {v!r} as {kind}") from None
    setattr(cfg, k, val)


def _config(path, overrides):
    if path:
        with open(path) as f:
            cfg = Config.from_json(f.read())
    else:
        cfg = Config()
    for kv in overrides:
        _parse_override(cfg, kv)
    cfg.validate()
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optflow_ref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pair", "seq"):
        p = sub.add_parser(name)
        if name == "pair":
            p.add_argument("--ref", required=True)
            p.add_argument("--def", dest="deformed", required=True)
        else:
            p.add_argument("--frames", required=True, help="video file, image directory or glob")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int, default=None)
            p.add_argument("--step", type=int, default=1)
        p.add_argument("--roi", help="JSON with {\"polygon\": [[x, y], ...]} in full-resolution pixels")
        p.add_argument("--mask", help="instead of --roi: a .npy (bool or uint8) or PNG file, nonzero = inside, "
                                      "with the shape of the scaled image")
        p.add_argument("--scale", type=int, default=1, help="integer downscale factor")
        p.add_argument("--crop", choices=("auto", "off"), default="auto",
                       help="auto: with --roi, track only the ROI's bounding box plus a margin (outputs keep the "
                            "full size, NaN outside); off: the whole image")
        p.add_argument("--crop-margin", type=int, default=None, metavar="N",
                       help="crop margin in pixels of the scaled image (default: max_disp + 4 sigma_w + guard "
                            "+ 32 + the coarse window footprint (poc_win/2 + grid_step) * 2^levels, doubled "
                            "for seq)")
        p.add_argument("--config")
        p.add_argument("--set", action="append", default=[], help="override, e.g. --set sigma_w=6")
        p.add_argument("--out", required=True)
        p.add_argument("--png", action="store_true", help="also write overlay maps")
        p.add_argument("--raw", action="store_true", help="also write raw_ux.npy and raw_uy.npy (the displacement "
                                                          "before the confidence threshold, the ROI and the guard)")
    a = ap.parse_args(argv)
    try:
        return _run(a)
    except (ValueError, OSError) as e:  # OSError: missing or unreadable file, undecodable image
        print(f"error: {e}", file=sys.stderr)
        return 2


def _region(a, shape):
    """The ROI mask (--roi polygon or --mask file) of an image of `shape`, or None."""
    if a.mask:
        return load_mask(a.mask, shape)
    return polygon_mask(shape, load_roi(a.roi), a.scale) if a.roi else None


def _plan_crop(a, cfg, mask, seq):
    """The crop of this run (None: track the whole image); writes crop.json."""
    crop = plan(mask, cfg, a.crop_margin, seq) if a.crop == "auto" and mask is not None else None
    with open(os.path.join(a.out, "crop.json"), "w") as f:
        f.write(OFF_JSON if crop is None else crop.json())
    return crop


def _capture_warning(cfg, shape):
    """The warning line when the coarse stage of the image actually tracked (`shape` = (height, width) after
    --scale and the crop) is designed to capture less than max_disp along an axis, else None. Printed once
    per run after the first successful tracking, so a run that fails prints only its error."""
    cy, cx = capture_range(cfg.max_disp, cfg.poc_win, shape)
    if min(cy, cx) >= cfg.max_disp:
        return None
    return (f"warning: the coarse stage covers about ±{cx:g} px in x and ±{cy:g} px in y, "
            f"less than max_disp {cfg.max_disp:g}; larger motions may be lost "
            "(use a larger image or crop margin, or a smaller max_disp)")


def _run(a):
    cfg = _config(a.config, a.set)
    if a.roi and a.mask:
        raise ValueError("--roi and --mask are mutually exclusive")
    check_margin(a.crop_margin)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    if a.cmd == "pair":
        I1 = downscale(read_image(a.ref), a.scale)
        I2 = downscale(read_image(a.deformed), a.scale)
        mask = _region(a, I1.shape)
        # images of different sizes are track_pair's error to report
        crop = _plan_crop(a, cfg, mask, seq=False) if I1.shape == I2.shape else None
        note = _capture_warning(cfg, I1.shape if crop is None else crop.shape)
        if crop is None:
            res = track_pair(I1, I2, cfg, mask=mask)
            if note:
                print(note, file=sys.stderr)
        else:
            cmask = crop.apply(mask)
            res = track_pair(crop.apply(I1), crop.apply(I2), cfg, mask=cmask)
            if note:
                print(note, file=sys.stderr)
            if border_fraction(res.raw_ux, res.raw_uy, cmask, crop, I1.shape, cfg) > WARN_FRACTION:
                print(WARNING, file=sys.stderr)
            res = crop.embed(res, I1.shape)
        save_result(a.out, res, a.raw)
        if a.png:
            save_maps(I1, res, os.path.join(a.out, "maps.png"))
        print(f"done: {res.timing}")
        return 0
    frames = iter_frames(a.frames, a.start, a.stop, a.step, a.scale, cfg.tavg, cfg.tavg_align, cfg,
                         lambda shape: _region(a, shape))
    first = next(frames, None)
    if first is None:
        raise ValueError(f"no frames to track in {a.frames} (check --start, --stop and --step)")
    second = next(frames, None)
    if second is None:
        raise ValueError("need at least two frames to track, got 1")
    mask = _region(a, first.shape)
    crop = _plan_crop(a, cfg, mask, seq=True)
    cmask = mask if crop is None else crop.apply(mask)

    def chain():
        for k, f in enumerate(itertools.chain((first, second), frames)):
            if crop is not None:
                if f.shape != first.shape:  # the error track_sequence gives, with the shapes of the whole frames
                    raise ValueError(f"frame {k} has shape {f.shape}, expected {first.shape}")
                f = crop.apply(f)
            yield f

    note = _capture_warning(cfg, first.shape if crop is None else crop.shape)
    warned = False
    for k, res in track_sequence(chain(), cfg, mask=cmask):
        if note:
            print(note, file=sys.stderr)
            note = None
        if crop is not None:
            if not warned and border_fraction(res.raw_ux, res.raw_uy, cmask, crop, first.shape, cfg) > WARN_FRACTION:
                print(WARNING, file=sys.stderr)
                warned = True
            res = crop.embed(res, first.shape)
        d = os.path.join(a.out, f"frame_{k:05d}")
        save_result(d, res, a.raw)
        if a.png:
            save_maps(first, res, os.path.join(d, "maps.png"), title=f"(0 -> {k})")
        print(f"frame {k}: reference {res.timing['reference']} mean conf {res.timing['mean_conf']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

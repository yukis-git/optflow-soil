"""Image, video, ROI and result I/O. All images become float64 grey levels in [0, 1]."""
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def to_gray(a: np.ndarray) -> np.ndarray:
    """uint8/uint16/float array (H, W) or (H, W, 3|4) -> float64 luminance in [0, 1]."""
    if a.dtype == np.uint8:
        f = a.astype(np.float64) / 255.0
    elif a.dtype == np.uint16:
        f = a.astype(np.float64) / 65535.0
    elif np.issubdtype(a.dtype, np.floating):
        f = a.astype(np.float64)
    else:
        raise ValueError(f"unsupported pixel type {a.dtype}")
    if f.ndim == 3:
        f = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    elif f.ndim != 2:
        raise ValueError(f"unsupported image shape {a.shape}")
    return f


def read_image(path: str) -> np.ndarray:
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        return to_gray(tifffile.imread(path))
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            return to_gray(np.asarray(im).astype(np.uint16))
        if im.mode not in ("L", "RGB", "RGBA"):
            im = im.convert("RGB")
        return to_gray(np.asarray(im))


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-average downscale by an integer factor >= 1 (crops the remainder)."""
    if factor < 1:
        raise ValueError(f"scale must be >= 1, got {factor}")
    if factor == 1:
        return img
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def _check_range(start, step):
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")


def iter_video(path, start=0, stop=None, step=1):
    _check_range(start, step)
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        idx = 0
        while stop is None or idx < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start and (idx - start) % step == 0:
                yield to_gray(frame[..., ::-1])  # BGR -> RGB
            idx += 1
    finally:
        cap.release()


def list_images(source: str):
    """Image files of a directory or glob pattern, sorted by plain string order (so IMG_10 comes
    before IMG_9; zero-pad frame numbers). The Rust implementation must sort the same way."""
    if os.path.isdir(source):
        paths = sorted(p for p in glob.glob(os.path.join(source, "*"))
                       if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")))
    else:
        paths = sorted(glob.glob(source))
    if not paths:
        raise ValueError(f"no images found for {source}")
    return paths


def _raw_frames(source, lo, hi):
    """Every frame with index lo <= i < hi (hi None: to the end) of a video file, directory or glob, undownscaled."""
    if os.path.splitext(source)[1].lower() in VIDEO_EXT:
        return iter_video(source, lo, hi, 1)
    return (read_image(p) for p in list_images(source)[lo:hi])


def iter_frames(source: str, start=0, stop=None, step=1, scale=1, tavg=0, align=False, cfg=None, mask_fn=None):
    """Frames from a video file, a directory of images or a glob pattern; frame indices
    start, start + step, ... < stop (step >= 1, start >= 0, checked on the first next()).

    tavg = m > 0: each of these frames is the mean of the frames i - m .. i + m (clipped to the video; the window may
    reach past `stop`), taken after the integer downscale. align: the neighbours are first aligned to frame i with an
    affine motion from the coarse POC flow (tavg.align_to; cfg gives its parameters, default Config(); mask_fn(shape)
    returns the ROI mask of a downscaled frame, to which the motion fit is restricted). tavg = 0 returns the frames
    untouched."""
    _check_range(start, step)
    if tavg < 0:
        raise ValueError(f"tavg must be >= 0, got {tavg}")
    if tavg:
        yield from _iter_averaged(source, start, stop, step, scale, tavg, align, cfg, mask_fn)
        return
    if os.path.splitext(source)[1].lower() in VIDEO_EXT:
        gen = iter_video(source, start, stop, step)
    else:
        gen = (read_image(p) for p in list_images(source)[start:stop:step])
    for f in gen:
        yield downscale(f, scale)


def _iter_averaged(source, start, stop, step, scale, m, align, cfg, mask_fn):
    from .tavg import average_window
    buf = {}  # frame index -> downscaled frame, only the indices a pending window still needs
    nxt, last = start, -1  # next sampled index, last frame index read

    def emit(s):
        lo, hi = max(0, s - m), min(s + m, last)
        fr = [buf[i] for i in range(lo, hi + 1)]
        return average_window(fr, s - lo, align, cfg, mask_fn(fr[0].shape) if align and mask_fn else None)

    def due(i_ready):
        return (stop is None or nxt < stop) and nxt <= i_ready

    for i, f in enumerate(_raw_frames(source, max(0, start - m), None if stop is None else stop + m), start=max(0, start - m)):
        buf[i], last = downscale(f, scale), i
        while due(i - m):  # the window of `nxt` is complete
            yield emit(nxt)
            nxt += step
            for j in [j for j in buf if j < nxt - m]:
                del buf[j]
    while due(last):  # the video ended: windows clipped at the last frame
        yield emit(nxt)
        nxt += step


def polygon_mask(shape, points, scale=1) -> np.ndarray:
    """Boolean ROI from polygon vertices [[x, y], ...] given in full-resolution pixels."""
    im = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([(x / scale, y / scale) for x, y in points], fill=1)
    return np.asarray(im, dtype=bool)


def load_roi(path: str):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("polygon"), list):
        raise ValueError(f"ROI file {path} has no 'polygon' list")
    return data["polygon"]


def _shape_text(shape) -> str:
    return str(tuple(int(n) for n in shape))


def load_mask(path: str, shape) -> np.ndarray:
    """Boolean ROI from a .npy file (bool or uint8) or a PNG: nonzero = inside. It lives in pixels of the
    scaled image, so its shape must be `shape` = (height, width) of that image."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        try:
            a = np.load(path, allow_pickle=False)
        except OSError as e:
            raise ValueError(f"cannot read mask file {path}: {e}") from None
        except ValueError as e:
            raise ValueError(f"invalid mask file {path}: {e}") from None
        if a.dtype not in (np.bool_, np.uint8):
            raise ValueError(f"mask file {path} must hold bool or uint8, got {a.dtype}")
    elif ext == ".png":
        try:
            with Image.open(path) as im:
                if im.mode in ("1", "L", "I;16", "I"):
                    a = np.asarray(im)
                else:  # colour: inside where any channel is nonzero (alpha is ignored)
                    a = np.asarray(im.convert("RGB")).any(axis=2)
        except OSError as e:
            raise ValueError(f"cannot read mask file {path}: {e}") from None
    else:
        raise ValueError(f"mask file {path} must be a .npy or .png file")
    if a.shape != tuple(shape):
        raise ValueError(f"mask file {path} has shape {_shape_text(a.shape)}, expected {_shape_text(shape)}")
    return a != 0


def save_result(out_dir: str, res, raw: bool = False) -> None:
    """Write ux, uy (NaN = invalid), conf, the strain fields (float32) and valid (uint8, 1 where ux is
    finite). conf is the unguarded confidence (see track.Result): thresholding it at conf_min does not
    reproduce valid, which also excludes the ROI exterior and the guard bands. raw: also write raw_ux and
    raw_uy, the displacement before the confidence threshold, the ROI and the guard (Result.raw_ux/raw_uy)."""
    os.makedirs(out_dir, exist_ok=True)
    arrays = {"ux": res.ux, "uy": res.uy, "conf": res.conf}
    arrays.update({k: res.strain[k] for k in ("Exx", "Eyy", "Exy", "gmax", "vol")})
    if raw:
        arrays.update({"raw_ux": res.raw_ux, "raw_uy": res.raw_uy})
    for name, a in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), np.ascontiguousarray(a, dtype="<f4"))
    np.save(os.path.join(out_dir, "valid.npy"), np.isfinite(res.ux).astype(np.uint8))
    if res.crack is not None:  # crack_out: the band (bool), its normal, opening and slip (NaN outside the band)
        np.save(os.path.join(out_dir, "crack.npy"), res.crack["mask"].astype(bool))
        for name, key in (("crack_nx", "nx"), ("crack_ny", "ny"), ("cod", "cod"), ("csd", "csd")):
            np.save(os.path.join(out_dir, f"{name}.npy"), np.ascontiguousarray(res.crack[key], dtype="<f4"))
    if res.fb is not None:  # hybrid sequences: consecutive frames each pixel was carried by the adjacent leg (0 = direct)
        np.save(os.path.join(out_dir, "fb.npy"), np.ascontiguousarray(res.fb, dtype="<i4"))
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(res.timing, f, indent=2)

"""Helpers to run the Rust `optflow` CLI from the benchmark scripts: 16-bit PNG frames, one process per
call measured with /usr/bin/time -l (wall time, peak RSS), and loading of the .npy fields it writes."""
import datetime
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DEFAULT_BIN = os.path.join(REPO, "rust", "target", "release", "optflow")
# the .npy files the CLI writes: those of save_result, and with --raw raw_ux, raw_uy (Result.raw_ux, raw_uy of the
# Python reference: the displacement before the confidence threshold, the ROI and the guard)
FIELD_NAMES = ("ux", "uy", "conf", "Exx", "Eyy", "Exy", "gmax", "vol", "raw_ux", "raw_uy")
STRAIN_NAMES = ("Exx", "Eyy", "Exy", "gmax", "vol")
TIME_BIN = "/usr/bin/time"


def binary(path=None):
    p = path or os.environ.get("OPTFLOW_BIN") or DEFAULT_BIN
    if not os.path.isfile(p):
        raise FileNotFoundError(f"optflow binary not found: {p} (cd rust && cargo build --release -p optflow-cli)")
    return os.path.abspath(p)


def quantize16(gray):
    """float grey in [0, 1] -> uint16 as written to the PNG (round to 1/65535)."""
    g = np.asarray(gray, dtype=np.float64)
    if g.min() < -1e-9 or g.max() > 1 + 1e-9:
        raise ValueError(f"grey levels leave [0, 1] ({g.min():.4f}..{g.max():.4f}): not representable in a PNG")
    return np.round(np.clip(g, 0.0, 1.0) * 65535.0).astype(np.uint16)


def dequantize16(q):
    """What a reader makes of the 16-bit PNG (value / 65535)."""
    return q.astype(np.float64) / 65535.0


def write_png16(path, gray):
    import cv2

    if not cv2.imwrite(path, quantize16(gray)):
        raise OSError(f"cannot write {path}")


_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)
# macOS: what Activity Monitor shows (includes compressed pages), next to the maximum RSS
_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M)
_TIMES = re.compile(r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", re.M)
BUILD_FILES = (".rs", ".toml", ".lock")  # tracked files under rust/ that go into the binary (not fixtures, docs)


@functools.lru_cache(maxsize=None)
def binary_info(bin_path=None):
    """Provenance of the CLI binary: sha256, size, mtime, and whether it is older than the newest build input
    (a tracked .rs / .toml / .lock file under rust/, by mtime; None when that cannot be determined). The `git`
    entry of a record is the source revision only; this says which binary actually ran. A stale binary prints
    a warning once."""
    p = binary(bin_path)
    st = os.stat(p)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    info = {"path": p, "sha256": h.hexdigest(), "size": st.st_size, "mtime": st.st_mtime,
            "mtime_iso": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "bin_stale": None}
    try:
        files = subprocess.check_output(["git", "ls-files", "rust"], text=True, cwd=REPO).split()
        mt = {f: os.path.getmtime(os.path.join(REPO, f)) for f in files
              if f.endswith(BUILD_FILES) and os.path.exists(os.path.join(REPO, f))}
        newest = max(mt, key=mt.get)
        info.update(newest_source=newest, newest_source_mtime=mt[newest], bin_stale=bool(st.st_mtime < mt[newest]))
        if info["bin_stale"]:
            newest_iso = datetime.datetime.fromtimestamp(mt[newest]).isoformat(timespec="seconds")
            print(f"warning: {p} (built {info['mtime_iso']}) is older than {newest} ({newest_iso}); "
                  f"rebuild: cd rust && cargo build --release -p optflow-cli", file=sys.stderr, flush=True)
    except Exception:
        pass
    return info


def bin_fields(bin_path=None):
    """The record entries that describe the binary: bin (path, sha256, size, mtime, ...) and bin_stale."""
    info = dict(binary_info(bin_path))
    return {"bin_stale": info.pop("bin_stale"), "bin": info}


def run_cli(args, threads=None, bin_path=None, device=None):
    """Run `optflow <args>` under /usr/bin/time -l. threads = RAYON_NUM_THREADS (None: all threads).
    device = "gpu" adds `--device gpu` (a binary built with `--features gpu`; the CLI exits with 2 otherwise).
    Returns wall_seconds (process start to exit, measured here), max_rss_bytes, peak_footprint_bytes (macOS),
    user/sys CPU seconds, stdout and the CLI's own stderr (its warnings). A non-zero exit raises RuntimeError with the
    CLI's message."""
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    if threads:
        env["RAYON_NUM_THREADS"] = str(int(threads))
    if device == "gpu":
        args = [*args, "--device", "gpu"]
    timed = os.path.exists(TIME_BIN)
    cmd = ([TIME_BIN, "-l"] if timed else []) + [binary(bin_path), *map(str, args)]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t
    tm = _TIMES.search(p.stderr)
    own = p.stderr[:tm.start()] if tm else p.stderr  # the CLI's own messages, before the time(1) statistics
    if p.returncode != 0:
        raise RuntimeError(f"optflow {' '.join(map(str, args))} failed ({p.returncode}): {own.strip()[-600:]}")
    rss = _RSS.search(p.stderr)
    fp = _FOOTPRINT.search(p.stderr)
    return {"wall_seconds": wall, "max_rss_bytes": int(rss.group(1)) if rss else None,
            "peak_footprint_bytes": int(fp.group(1)) if fp else None,
            "user_seconds": float(tm.group(2)) if tm else None, "sys_seconds": float(tm.group(3)) if tm else None,
            "stdout": p.stdout, "stderr": own.strip()}


def algo_seconds(timing):
    """Sum of the stage timers (coarse + fine + strain): the algorithm time without any I/O."""
    return float(timing["coarse"] + timing["fine"] + timing["strain"])


def load_result(d, only=None):
    """Result-like object from a CLI output directory: float64 ux, uy, conf, raw_ux, raw_uy, strain dict, timing
    dict. only: names to load (default all; raw_ux, raw_uy only when the run had --raw); the rest are None."""
    names = FIELD_NAMES if only is None else only
    if only is None:
        names = [k for k in names if not k.startswith("raw_") or os.path.exists(os.path.join(d, f"{k}.npy"))]
    a = {k: np.load(os.path.join(d, f"{k}.npy")).astype(np.float64) for k in names}
    with open(os.path.join(d, "timing.json")) as f:
        timing = json.load(f)
    return SimpleNamespace(ux=a.get("ux"), uy=a.get("uy"), conf=a.get("conf"), raw_ux=a.get("raw_ux"),
                           raw_uy=a.get("raw_uy"), strain={k: a[k] for k in STRAIN_NAMES if k in a}, timing=timing)

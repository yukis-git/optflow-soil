"""Run the synthetic benchmark; one JSON per case in results/.

--impl python runs the reference in-process. --impl rust writes the frames as 16-bit PNG (the video case as
the same MJPG video), runs `optflow pair|seq` on them and evaluates the .npy fields it writes with the same
code. Fields are kept under <out>/fields/<label>/<case>/ (label: python, rust, rust-t1, rust-gpu, python-q16) for the
parity table of report.py; sequences keep the first, middle and last frame.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import rustcli  # noqa: E402
from bench.cases import CRACK_SEED, all_cases  # noqa: E402
from optflow_ref import synth  # noqa: E402
from optflow_ref.io import iter_frames, save_result  # noqa: E402
from optflow_ref.metrics import crlb_rms, disp_errors, eval_mask, strain_errors, true_strain  # noqa: E402
from optflow_ref.track import track_pair, track_sequence  # noqa: E402


REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def git_rev(path="python"):
    """Short hash of the last commit that touched `path` (python/ or rust/), plus "-dirty" when it has
    uncommitted changes (the results then do not correspond to a commit)."""
    try:
        def git(*args):
            return subprocess.check_output(["git", *args, "--", path], text=True, cwd=REPO).strip()

        return git("log", "-1", "--format=%h") + ("-dirty" if git("status", "--porcelain") else "")
    except Exception:
        return "unknown"


def env():
    import cv2
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "opencv": cv2.__version__, "machine": platform.machine(), "system": platform.system()}


SEED_REF, SEED_DEF = 1, 2  # noise seeds of the reference and of the (first) deformed frame


def crack_band(n, width):
    """Column range [lo, hi) of the central vertical crack band in the deformed frame."""
    lo = n // 2 - width // 2
    return lo, lo + width


def frame(pattern, n, f, case, seed):
    img = synth.render(pattern, n, n, f)
    if case.crack_width and f is not None:  # crack: another pattern shows through a central band
        lo, hi = crack_band(n, case.crack_width)
        img[:, lo:hi] = synth.render(synth.make_pattern(n, n, seed=CRACK_SEED), n, n)[:, lo:hi]
    if case.overlay:  # static pattern fixed to the camera (scratches on glass), identical in every frame
        img = img + synth.scratches(n, n, **case.overlay)
    if f is not None and case.illum:
        img = synth.degrade(img, **case.illum)
    if case.noise or case.bits:
        img = synth.degrade(img, seed=seed, noise=case.noise, bits=case.bits or None)
    return img


def evaluate(res, f, n, margin):
    m = eval_mask((n, n), f, margin)
    gx, gy = synth.displacement(f, n, n)
    return {"disp": disp_errors(res.ux, res.uy, gx, gy, m),
            "strain": strain_errors(res.strain, true_strain(f, (n, n)), m)}


def crack_metrics(res, f, n, case, near=40.0):
    """Errors near and far from the crack. Distance = |deformed x position - band centre|, i.e. measured in the
    deformed image where the band sits; pixels are those of the (guarded) result on the reference grid."""
    lo, hi = crack_band(n, case.crack_width)
    centre = 0.5 * (lo + hi - 1)
    m = eval_mask((n, n), f, case.margin)
    gx, gy = synth.displacement(f, n, n)
    dist = np.abs(np.arange(n, dtype=np.float64)[None, :] + gx - centre)
    fin = m & np.isfinite(res.ux) & np.isfinite(res.uy)
    e = np.hypot(res.ux - gx, res.uy - gy)
    out = {"band_columns": [lo, hi], "band_centre_x": centre, "near_px": near,
           "valid_frac": float(fin.sum() / m.sum())}  # guard included: it is applied inside track_pair
    for name, sel in (("near", dist < near), ("far", dist >= near)):
        v, s = fin & sel, m & sel
        out[name] = {"rms": float(np.sqrt(np.mean(e[v] ** 2))) if v.any() else float("nan"),
                     "valid_frac": float(v.sum() / s.sum()) if s.any() else float("nan"), "n_valid": int(v.sum())}
    return out


def write_video(frames, path):
    import cv2
    n = frames[0].shape[0]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (n, n), isColor=False)
    for f in frames:
        vw.write(np.round(f * 255).astype(np.uint8))
    vw.release()


def label_of(impl, threads=None, q16=False, device=None):
    """Name of an implementation variant: python, python-q16, rust, rust-t1 (RAYON_NUM_THREADS=1), rust-gpu
    (--device gpu)."""
    return impl + ("-gpu" if device == "gpu" else "") + (f"-t{threads}" if threads else "") + ("-q16" if q16 else "")


def keep_frames(n_frames):
    """Sequence frames whose fields are kept for the parity table: first, middle and last."""
    return sorted({1, (n_frames + 1) // 2, n_frames})


def base_record(case, n, seed, impl, label, threads=None, bin_path=None, device=None):
    rec = {"case": case.name, "group": case.group, "params": case.params, "impl": impl, "label": label, "size": n,
           "config": asdict(case.cfg), "git": git_rev("python" if impl == "python" else "rust"), "env": env(),
           "seeds": {"pattern": seed, "noise_ref": SEED_REF, "noise_deformed": SEED_DEF},  # sequences: DEF + k
           "date": datetime.date.today().isoformat(), "loadavg_start": list(os.getloadavg())}
    if impl == "rust":
        # threads None: all (RAYON default). git = the rust/ source revision; the binary that ran is in bin
        # (sha256, size, mtime) and bin_stale says it is older than a build input; git_python = python/ revision.
        rec.update(threads=threads, cpu_count=os.cpu_count(), git_python=git_rev("python"),
                   **rustcli.bin_fields(bin_path))
        rec["env"]["device"] = device or "cpu"  # where the CLI's fine stage ran (--device)
    return rec


def score_pair(rec, case, pattern, n, res):
    """Errors against the truth of a pair case (res: Result or a CLI result with ux, uy, strain)."""
    f = case.fields[0]
    rec.update(evaluate(res, f, n, case.margin))
    if case.noise:  # the bound refers to the noise-free reference image
        rec["crlb_rms"] = crlb_rms(synth.render(pattern, n, n), case.noise, case.cfg.sigma_w)
        rec["rms_over_crlb"] = rec["disp"]["rms"] / rec["crlb_rms"]
    if case.crack_width:
        rec["crack"] = crack_metrics(res, f, n, case)


def finish_record(rec, case, n):
    frames = max(1, len(case.fields))
    rec["mpx_per_s"] = n * n * frames / 1e6 / rec["seconds"]
    rec["algo_mpx_per_s"] = n * n * frames / 1e6 / rec["algo_seconds"]
    rec["loadavg_end"] = list(os.getloadavg())  # 1/5/15 min; loadavg_start is taken in base_record
    return rec


def stage_timing(timing):
    return {k: timing[k] for k in ("coarse", "fine", "strain")}


def run_python(case, pattern, n, seed=0, fdir=None, q16=False):
    """The reference. seconds = frame synthesis + tracking (evaluation and saving excluded); algo_seconds =
    sum of the coarse, fine and strain timers (sequences: summed over the frames). q16: the frames go
    through the 16-bit quantisation the PNG files of the Rust run have."""
    rec = base_record(case, n, seed, "python", label_of("python", q16=q16))
    if q16:
        rec["quantize16"] = True
    qf = (lambda a: rustcli.dequantize16(rustcli.quantize16(a))) if q16 and not case.video else (lambda a: a)
    ref = qf(frame(pattern, n, None, case, seed=SEED_REF))
    kept = []
    t = time.perf_counter()
    if case.group != "sequence":
        f = case.fields[0]
        res = track_pair(ref, qf(frame(pattern, n, f, case, seed=SEED_DEF)), case.cfg)
        rec["seconds"] = time.perf_counter() - t
        rec["timing"] = res.timing
        rec["algo_seconds"] = rustcli.algo_seconds(res.timing)
        score_pair(rec, case, pattern, n, res)
        kept.append((None, res))
    else:
        frames = [ref] + [qf(frame(pattern, n, f, case, seed=SEED_DEF + k)) for k, f in enumerate(case.fields)]
        if case.video:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "seq.avi")
                write_video(frames, path)
                frames = list(iter_frames(path))
        per, last, eval_s = [], None, 0.0
        keep = keep_frames(len(case.fields))
        for k, res in track_sequence(frames, case.cfg):
            te = time.perf_counter()
            e = evaluate(res, case.fields[k - 1], n, case.margin)
            eval_s += time.perf_counter() - te
            per.append({"frame": k, "rms": e["disp"]["rms"], "reference": res.timing["reference"],
                        "timing": stage_timing(res.timing)})
            if k in keep:
                kept.append((k, res))
            last = e
        rec["seconds"] = time.perf_counter() - t - eval_s
        rec.update(last)
        rec["per_frame"] = per
        rec["algo_seconds"] = sum(rustcli.algo_seconds(p["timing"]) for p in per)
    if fdir:
        if os.path.isdir(fdir):
            shutil.rmtree(fdir)
        for k, res in kept:
            save_result(fdir if k is None else os.path.join(fdir, f"frame_{k:05d}"), res)
        rec["fields_dir"] = fdir
    return finish_record(rec, case, n)


def run_rust(case, pattern, n, seed=0, threads=None, bin_path=None, fdir=None, device=None):
    """The CLI on PNG/MJPG input. seconds = writing the input + the whole process (start, decode, algorithm,
    .npy writing) + reading the .npy back; cli_seconds = the process alone; algo_seconds = the CLI's
    own timers (coarse + fine + strain; sequences: summed over the frames)."""
    rec = base_record(case, n, seed, "rust", label_of("rust", threads, device=device), threads, bin_path, device)
    pair = case.group != "sequence"
    ref = frame(pattern, n, None, case, seed=SEED_REF)
    if pair:
        frames = [ref, frame(pattern, n, case.fields[0], case, seed=SEED_DEF)]
    else:
        frames = [ref] + [frame(pattern, n, f, case, seed=SEED_DEF + k) for k, f in enumerate(case.fields)]
    if fdir and os.path.isdir(fdir):
        shutil.rmtree(fdir)
    with tempfile.TemporaryDirectory() as tmp:
        out = fdir or os.path.join(tmp, "out")
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w") as fh:
            fh.write(case.cfg.to_json())
        t = time.perf_counter()
        if pair:
            a, b = os.path.join(tmp, "ref.png"), os.path.join(tmp, "def.png")
            rustcli.write_png16(a, frames[0])
            rustcli.write_png16(b, frames[1])
            args = ["pair", "--ref", a, "--def", b]
        elif case.video:
            src = os.path.join(tmp, "seq.avi")
            write_video(frames, src)
            args = ["seq", "--frames", src]
        else:
            src = os.path.join(tmp, "frames")
            os.makedirs(src)
            for i, fr in enumerate(frames):  # zero-padded: the CLI sorts the names as plain strings
                rustcli.write_png16(os.path.join(src, f"f_{i:03d}.png"), fr)
            args = ["seq", "--frames", src]
        rec["input_write_seconds"] = time.perf_counter() - t
        cli = rustcli.run_cli([*args, "--config", cfg_path, "--out", out], threads, bin_path, device)
        rec.update(cli_seconds=cli["wall_seconds"], max_rss_bytes=cli["max_rss_bytes"],
                   peak_footprint_bytes=cli["peak_footprint_bytes"], cli_user_seconds=cli["user_seconds"], cli_sys_seconds=cli["sys_seconds"])
        read_s = 0.0
        if pair:
            t = time.perf_counter()
            res = rustcli.load_result(out)
            read_s = time.perf_counter() - t
            rec["timing"] = res.timing
            rec["algo_seconds"] = rustcli.algo_seconds(res.timing)
            score_pair(rec, case, pattern, n, res)
        else:
            per, last = [], None
            keep = keep_frames(len(case.fields))
            for k in range(1, len(case.fields) + 1):
                d = os.path.join(out, f"frame_{k:05d}")
                t = time.perf_counter()
                res = rustcli.load_result(d)
                read_s += time.perf_counter() - t
                last = evaluate(res, case.fields[k - 1], n, case.margin)
                per.append({"frame": k, "rms": last["disp"]["rms"], "reference": res.timing["reference"],
                            "timing": stage_timing(res.timing)})
                if fdir and k not in keep:
                    shutil.rmtree(d)
            rec.update(last)
            rec["per_frame"] = per
            rec["algo_seconds"] = sum(rustcli.algo_seconds(p["timing"]) for p in per)
        rec["output_read_seconds"] = read_s
        rec["seconds"] = rec["input_write_seconds"] + rec["cli_seconds"] + read_s
        if fdir:
            rec["fields_dir"] = fdir
    return finish_record(rec, case, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cases", default="", help="comma-separated substrings; empty = all")
    ap.add_argument("--pairs-only", action="store_true", help="skip the sequence and video cases")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", choices=("python", "rust"), default="python")
    ap.add_argument("--threads", type=int, default=None,
                    help="rust: RAYON_NUM_THREADS for the CLI process (default: all threads)")
    ap.add_argument("--bin", default=None, help="rust: optflow binary (default: $OPTFLOW_BIN or rust/target/release)")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                    help="rust: the CLI's --device (gpu: label rust-gpu; needs a binary built with --features gpu)")
    ap.add_argument("--no-save-fields", action="store_true",
                    help="do not keep <out>/fields/<label>/<case>/ (the parity table of report.py needs them)")
    ap.add_argument("--quantize16", action="store_true",
                    help="python: round the frames to 16 bit like the PNG files of the rust run (label python-q16)")
    a = ap.parse_args()
    if a.impl == "python" and (a.threads or a.bin):
        ap.error("--threads and --bin apply to --impl rust")
    if a.impl == "python" and a.device != "cpu":
        ap.error("--device applies to --impl rust")
    if a.impl == "rust" and a.quantize16:
        ap.error("--quantize16 applies to --impl python (the rust run always reads 16-bit PNG)")
    os.makedirs(a.out, exist_ok=True)
    pattern = synth.make_pattern(a.size, a.size, seed=a.seed)
    keys = [k for k in a.cases.split(",") if k]
    label = label_of(a.impl, a.threads, a.quantize16, a.device)
    for case in all_cases(a.size):
        if keys and not any(k in case.name for k in keys):
            continue
        if a.pairs_only and case.group == "sequence":
            continue
        fdir = None if a.no_save_fields else os.path.join(a.out, "fields", label, case.name)
        if a.impl == "python":
            rec = run_python(case, pattern, a.size, a.seed, fdir, a.quantize16)
        else:
            rec = run_rust(case, pattern, a.size, a.seed, a.threads, a.bin, fdir, a.device)
        if rec.get("fields_dir"):
            rec["fields_dir"] = os.path.relpath(rec["fields_dir"], a.out)
        path = os.path.join(a.out, f"{rec['date']}_{label}_{case.name}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        d = rec["disp"]
        print(f"{case.name:24s} rms {d['rms']:.4f} px  max {d['max']:.3f}  "
              f"Exx {rec['strain']['Exx']:.1e}  algo {rec['algo_seconds']:.2f}s  wall {rec['seconds']:.2f}s  "
              f"[{label}]", flush=True)


if __name__ == "__main__":
    main()

"""One-command demo: a synthetic model test of a footing settling into sand, tracked from the first frame's ROI.

    python -m demo.run_demo [--out DIR] [--size small|full] [--frames N] [--impl rust|python] [--no-animation]
    python python/demo/run_demo.py            (from the repository root)

Stages: generate the video and the truth -> ROI polygon -> track (Rust CLI `optflow seq --roi --crop auto`, or the
Python reference) -> compare with the truth -> overview.png, curves.png, strain_animation.mp4 -> summary.md.

--size small is a smoke test (384x216, 6 frames, 6 px): its wedge is narrower than the tracker's filters, so it is much
less accurate than the full size.  A settlement above s/B of about 0.3 is refused (the frozen mechanism folds the soil over).
"""
import argparse
import csv
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(PYTHON_DIR)
if PYTHON_DIR not in sys.path:  # `python python/demo/run_demo.py` from the repository root
    sys.path.insert(0, PYTHON_DIR)

import numpy as np  # noqa: E402

from bench import rustcli  # noqa: E402
from demo import evaluate as ev_mod  # noqa: E402
from demo import figures, report  # noqa: E402
from demo.field import field_summary  # noqa: E402
from demo.footing import Truth, generate_video, make_scene  # noqa: E402
from optflow_ref.config import Config  # noqa: E402
from optflow_ref.io import iter_video  # noqa: E402

DEFAULT_OUT = os.path.join(REPO, "demo_out")
# files and directories this demo writes into --out (and only these are replaced when it runs again)
OWN_FILES = ("footing.mp4", "roi.json", "scene.json", "truth.npz", "cli_stdout.txt", "cli_stderr.txt", "overview.png",
             "curves.png", "strain_animation.mp4", "summary.md", "metrics.json", "per_frame.csv")
OWN_DIRS = ("track",)


class Stages:
    """Wall-clock time of the stages, printed as they finish."""

    def __init__(self, quiet=False):
        self.t = {}
        self.quiet = quiet
        self._t0 = time.perf_counter()

    def run(self, name, label, fn):
        if not self.quiet:
            print(f"[{name}] {label} ...", flush=True)
        t = time.perf_counter()
        out = fn()
        self.t[name] = time.perf_counter() - t
        if not self.quiet:
            print(f"[{name}] done in {self.t[name]:.1f} s", flush=True)
        return out

    def total(self):
        return time.perf_counter() - self._t0


REUSED = ("footing.mp4", "roi.json", "truth.npz", "cli_stdout.txt", "cli_stderr.txt")


def prepare_out(out, reuse=False):
    """Create the output directory and remove what an earlier run of this demo left there (only its own files);
    with reuse, keep the video, the ROI and the tracker output."""
    out = os.path.abspath(out)
    home = os.path.expanduser("~")
    if out in (os.path.abspath(os.sep), os.path.abspath(home), REPO, PYTHON_DIR) or out.startswith(os.path.join(REPO, ".git")):
        raise ValueError(f"refusing to use {out} as the output directory")
    if os.path.exists(out) and not os.path.isdir(out):
        raise ValueError(f"{out} exists and is not a directory")
    os.makedirs(out, exist_ok=True)
    for name in OWN_FILES:
        p = os.path.join(out, name)
        if os.path.isfile(p) and not (reuse and name in REUSED + ("metrics.json",)):
            os.remove(p)
    for name in OWN_DIRS:
        p = os.path.join(out, name)
        if os.path.isdir(p) and not reuse:
            if os.path.isfile(os.path.join(p, "config.json")) or not os.listdir(p):
                shutil.rmtree(p)
            else:
                raise ValueError(f"{p} exists and does not look like tracker output of this demo; remove it first")
    return out


def ensure_binary(path=None, build=True):
    """Path of the release `optflow` binary (rustcli's search: --bin, $OPTFLOW_BIN, rust/target/release); built with
    cargo when it is missing and build is True."""
    try:
        return rustcli.binary(path)
    except FileNotFoundError:
        if not build or path:
            raise
    print("optflow binary not found: building it (cd rust && cargo build --release -p optflow-cli) ...", flush=True)
    subprocess.run(["cargo", "build", "--release", "-p", "optflow-cli"], cwd=os.path.join(REPO, "rust"), check=True)
    return rustcli.binary(path)


def run_tracker(a, mp4, roi_json, track_dir):
    """Run `optflow seq --frames mp4 --roi roi.json --crop auto --out track_dir --raw` (Rust), or the same command of
    the Python reference.  Returns a record: wall seconds, peak memory, the CLI's stdout / stderr, the binary."""
    args = ["seq", "--frames", mp4, "--roi", roi_json, "--crop", "auto", "--out", track_dir, "--raw"]
    shown = ["seq", "--frames", os.path.basename(mp4), "--roi", os.path.basename(roi_json), "--crop", "auto", "--out",
             os.path.basename(track_dir), "--raw"]  # as typed in the output directory
    for kv in a.set:
        args += ["--set", kv]
        shown += ["--set", kv]
    if a.impl == "rust":
        binary = ensure_binary(a.bin, build=not a.no_build)
        r = rustcli.run_cli(args, bin_path=binary)
        info = rustcli.binary_info(binary)
        rec = {"impl": "rust", "wall_seconds": r["wall_seconds"], "max_rss_bytes": r["max_rss_bytes"],
               "peak_footprint_bytes": r["peak_footprint_bytes"], "user_seconds": r["user_seconds"],
               "stdout": r["stdout"], "stderr": r["stderr"], "binary": info["path"], "binary_sha256": info["sha256"],
               "binary_stale": info["bin_stale"], "command": "optflow " + " ".join(shown)}
    else:
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "optflow_ref", *args], capture_output=True, text=True, cwd=PYTHON_DIR)
        wall = time.perf_counter() - t
        if p.returncode != 0:
            raise RuntimeError(f"python -m optflow_ref failed ({p.returncode}): {p.stderr.strip()[-600:]}")
        rec = {"impl": "python", "wall_seconds": wall, "max_rss_bytes": None, "peak_footprint_bytes": None,
               "user_seconds": None, "stdout": p.stdout, "stderr": p.stderr.strip(),
               "command": "python -m optflow_ref " + " ".join(shown)}
    rec["capture_warning"] = "the coarse stage covers" in rec["stderr"]
    rec["crop_warning"] = "close to the crop border" in rec["stderr"]
    return rec


def _json_clean(o):
    """Plain Python values for strict JSON: numpy scalars and arrays converted, NaN and infinity written as null."""
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


MIN_DET_F = 0.3  # the least det F the field may reach (the map must stay invertible, the truth meaningful)


def check_field(p):
    """Refuse a settlement for which the frozen mechanism folds the soil over (det F falls below MIN_DET_F: s/B above
    about 0.3).  Returns the field's summary."""
    fs = field_summary(p, step=1.0 if p.b >= 50 else 0.5)
    if fs["det_f_min"] < MIN_DET_F:
        raise ValueError(f"a settlement of {p.s_final:g} px (s/B = {p.s_final / p.B:.2f}) is too large for this field: "
                         f"det F falls to {fs['det_f_min']:.2f} (< {MIN_DET_F}) where the mechanism pushes into the soil, "
                         f"so the deformation is not invertible and the truth is meaningless; use s/B <= 0.3")
    return fs


def ffmpeg_version():
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "unknown"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="demo.run_demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: demo_out at the repository root)")
    ap.add_argument("--size", choices=("small", "full"), default="full", help="full: 1280x720, 60 frames, 30 px; "
                                                                             "small: 384x216, 6 frames (smoke test)")
    ap.add_argument("--frames", type=int, default=None, help="number of frames (default: 60 full, 6 small)")
    ap.add_argument("--settlement", type=float, default=None, help="settlement of the last frame [px] (default: 30 "
                                                                    "full, 6 small); refused above s/B of about 0.3 "
                                                                    "(det F < 0.3: 45 px full, 13 px small)")
    ap.add_argument("--impl", choices=("rust", "python"), default="rust", help="tracker: the Rust CLI (default) or "
                                                                                 "the slow Python reference")
    ap.add_argument("--no-animation", action="store_true", help="skip strain_animation.mp4")
    ap.add_argument("--seed", type=int, default=1, help="seed of the textures and the noise")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override for the tracker "
                                                                                    "(repeatable), e.g. max_disp=48")
    ap.add_argument("--bin", default=None, help="path of the optflow binary (default: $OPTFLOW_BIN or rust/target/release)")
    ap.add_argument("--no-build", action="store_true", help="do not build the Rust binary when it is missing")
    ap.add_argument("--keep-fields", action="store_true", help="keep the tracker's per-frame .npy output (about "
                                                               "30 MB per frame at full size); default: delete it")
    ap.add_argument("--reuse", action="store_true", help="re-make the figures, the animation and the summary from the "
                                                         "video and the tracker output of an earlier run in --out "
                                                         "(needs that run's --keep-fields); skips generation and tracking")
    ap.add_argument("--quiet", action="store_true", help="print nothing but errors")
    return ap.parse_args(argv)


def load_previous(out, scene):
    """metrics.json of an earlier run in `out`, checked against this run's scene (size, frame count and every parameter
    of the displacement field) and the files --reuse needs."""
    try:
        with open(os.path.join(out, "metrics.json")) as f:
            prev = json.load(f)
        with open(os.path.join(out, "cli_stdout.txt")) as f:
            prev["tracker"]["stdout"] = f.read()
    except OSError as e:
        raise ValueError(f"--reuse needs the output of an earlier run in {out}: {e}") from None
    sc = prev["scene"]
    if (sc["width"], sc["height"], sc["n_frames"]) != (scene.width, scene.height, scene.n_frames):
        raise ValueError("--reuse: the earlier run used another size or frame count")
    now = scene.params.__dict__
    old = sc.get("field", {})
    diff = [k for k, v in now.items() if k not in old or abs(float(old[k]) - float(v)) > 1e-9 * max(1.0, abs(float(v)))]
    if diff:
        raise ValueError("--reuse: the earlier run used another displacement field (" + ", ".join(
            f"{k}: {old.get(k)} -> {now[k]}" for k in diff[:4]) + (" ..." if len(diff) > 4 else "") +
            "); its tracker output does not belong to this truth")
    last = os.path.join(out, "track", f"frame_{scene.n_frames - 1:05d}", "raw_ux.npy")
    if not os.path.isfile(os.path.join(out, "footing.mp4")) or not os.path.isfile(last):
        raise ValueError("--reuse needs footing.mp4 and the tracker's per-frame output (run once with --keep-fields)")
    return prev


def generate_and_track(a, scene, out, mp4, roi_json, track_dir, st):
    """Stages 1 and 2: the video, the ROI and the truth, then the tracker."""

    def gen():
        step = max(scene.n_frames // 4, 1)

        def prog(k, n):
            if not a.quiet and k % step == 0:
                print(f"      rendering frame {k + 1}/{n}", flush=True)

        generate_video(scene, mp4, prog)
        with open(roi_json, "w") as f:
            json.dump({"polygon": scene.roi_polygon()}, f, indent=2)
        truth = Truth(scene)
        truth.save(os.path.join(out, "truth.npz"), frames=figures.selected_frames(scene.n_frames))
        return truth

    truth = st.run("generate", "synthetic video, ROI polygon and truth", gen)
    cli = st.run("track", f"tracking with {'the Rust CLI' if a.impl == 'rust' else 'the Python reference'}",
                 lambda: run_tracker(a, mp4, roi_json, track_dir))
    with open(os.path.join(out, "cli_stdout.txt"), "w") as f:
        f.write(cli["stdout"])
    with open(os.path.join(out, "cli_stderr.txt"), "w") as f:
        f.write(cli["stderr"] + ("\n" if cli["stderr"] else ""))
    if cli["capture_warning"] or cli["crop_warning"]:
        print("WARNING from the tracker:\n" + cli["stderr"], file=sys.stderr, flush=True)
    return truth, cli


def run(argv=None):
    """Run the whole demo; returns the metrics dict (also written to metrics.json)."""
    a = parse_args(argv)
    st = Stages(a.quiet)
    scene = make_scene(a.size, a.frames, a.settlement, a.seed)
    if scene.n_frames < 3:
        raise ValueError("need at least 3 frames")
    check_field(scene.params)
    out = prepare_out(a.out, reuse=a.reuse)
    mp4 = os.path.join(out, "footing.mp4")
    roi_json = os.path.join(out, "roi.json")
    track_dir = os.path.join(out, "track")
    p = scene.params
    if not a.quiet:
        print(f"demo: {scene.width}x{scene.height}, {scene.n_frames} frames, settlement 0 -> {p.s_final:g} px, "
              f"tracker {a.impl}, output {out}" + (" (reusing the video and the tracker output)" if a.reuse else ""),
              flush=True)

    if a.reuse:
        prev = load_previous(out, scene)
        truth = Truth(scene)
        cli = prev["tracker"]
        st.t.update({k: v for k, v in prev["stage_seconds"].items() if k in ("generate", "track")})
    else:
        truth, cli = generate_and_track(a, scene, out, mp4, roi_json, track_dir, st)

    # 3. evaluation
    sel = figures.selected_frames(scene.n_frames)
    ev = st.run("evaluate", "comparison with the truth",
                lambda: ev_mod.evaluate(scene, truth, track_dir, keep=sel))
    summ = ev_mod.summarize(ev)

    # 4. figures
    ref_img = next(iter_video(mp4, 0, 1))

    def figs():
        lim = figures.make_overview(ev, ref_img, os.path.join(out, "overview.png"),
                                    title=f"Footing settlement demo, {a.impl} tracker: estimate vs truth (frame-0 grid, "
                                          f"valid pixels, shared colour scales)\n"
                                          f"truth: a stylised, kinematically admissible field, not calibrated to an experiment")
        figures.make_curves(ev, os.path.join(out, "curves.png"), title="Footing settlement demo: per-frame accuracy")
        return lim

    lim = st.run("figures", "overview.png and curves.png", figs)

    # 5. animation
    if not a.no_animation:
        gmax_hi = lim["gmax"][1]
        st.run("animate", "strain_animation.mp4",
               lambda: figures.make_animation(scene, mp4, track_dir, os.path.join(out, "strain_animation.mp4"), gmax_hi,
                                              scene.roi_polygon()))

    # 6. summary and metrics
    cfg = Config.from_json(open(os.path.join(track_dir, "config.json")).read())
    timing = {k: sum(r[k] for r in ev.rows) for k in ("t_coarse", "t_fine", "t_strain")}
    stage_times = dict(st.t)
    stage_times["total"] = sum(stage_times.values()) if a.reuse else st.total()  # reuse: the stages of both runs
    metrics = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {"size": a.size, "frames": scene.n_frames, "settlement": p.s_final, "impl": a.impl, "seed": a.seed,
                 "set": a.set},
        "scene": {"width": scene.width, "height": scene.height, "n_frames": scene.n_frames, "fps": scene.fps,
                  "noise": scene.noise, "flicker": scene.flicker, "field": p.__dict__ | {"ha": p.ha, "xe": p.xe, "depth_mech": p.depth_mech, "theta_deg": p.theta_deg,
                                         "beta_in_deg": p.beta_in_deg, "dilation": p.dilation, "heave_flux": p.heave_flux},
                  "roi_polygon": scene.roi_polygon(), "roi_bbox": scene.roi_bbox()},
        "summary": summ, "error_facts": ev.extra, "stage_seconds": stage_times, "tracker": {k: v for k, v in cli.items() if k not in ("stdout",)},
        "tracker_algo_seconds": timing, "tracker_config": cfg.__dict__ | {"edge_guard_px": cfg.edge_guard_px()},
        "env": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                "machine": platform.machine(), "ffmpeg": ffmpeg_version(), "loadavg": list(os.getloadavg())},
        "crop": json.load(open(os.path.join(track_dir, "crop.json"))),
        "view_window": figures.view_window(scene), "selected_frames": sel, "limits": {k: list(v) for k, v in lim.items()},
        "frames": ev.rows,
    }
    with open(os.path.join(out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev.rows[0].keys()))
        w.writeheader()
        w.writerows(ev.rows)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(report.summary_md(scene, ev, metrics, a, out))
    stage_times["total"] = sum(v for k, v in stage_times.items() if k != "total") if a.reuse else st.total()
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(_json_clean(metrics), f, indent=2, allow_nan=False)
    if not (a.keep_fields or a.reuse):
        for name in os.listdir(track_dir):
            if name.startswith("frame_"):
                shutil.rmtree(os.path.join(track_dir, name))
    if not a.quiet:
        print_summary(summ, stage_times, cli, out)
    return metrics


def print_summary(summ, stages, cli, out):
    print("-" * 72)
    print("stage timings [s]: " + ", ".join(f"{k} {v:.1f}" for k, v in stages.items()))
    if np.isfinite(summ["settlement_est_final"]):
        print(f"settlement final: footing {summ['settlement_true_final']:.3f} px, soil in the wedge {summ['settlement_ref_final']:.3f} px, "
              f"estimated {summ['settlement_est_final']:.3f} px (max |estimate - soil| {summ['settlement_err_max_abs']:.3f} px)")
    elif summ["settlement_last_valid_frame"] is not None:
        print(f"settlement: no valid pixel in the wedge at the last frame; at frame {summ['settlement_last_valid_frame']}: footing "
              f"{summ['settlement_true_last_valid']:.3f} px, soil in the wedge {summ['settlement_ref_last_valid']:.3f} px, "
              f"estimated {summ['settlement_est_last_valid']:.3f} px (max |estimate - soil| {summ['settlement_err_max_abs']:.3f} px)")
    else:
        print("settlement: no valid pixel in the wedge at any frame")
    print(f"displacement error: RMS final {summ['disp_rms_final']:.4f} px, worst frame {summ['disp_rms_worst']:.4f} px "
          f"(frame {summ['disp_rms_worst_frame']}), p99 final {summ['disp_p99_final']:.3f} px, max final {summ['disp_max_final']:.3f} px")
    print(f"valid fraction of the ROI: min {summ['valid_frac_min']:.3f} (beyond the guard distance: "
          f"{summ['valid_frac_eff_min']:.3f}); reference updates: {summ['reference_updates']}")
    print(f"gmax RMS error final: in bands {summ['gmax_rms_band_final']:.3f}, outside {summ['gmax_rms_out_final']:.3f}; "
          f"99.9th percentile on the valid pixels: est {summ['gmax_peak_est_final']:.3f} vs true {summ['gmax_peak_true_final']:.3f} "
          f"(true, all pixels beyond the guard: {summ['gmax_peak_true_all_final']:.3f})")
    if cli["capture_warning"] or cli["crop_warning"]:
        print("tracker warning: " + cli["stderr"])
    print(f"outputs in {out}")


def main(argv=None):
    try:
        run(argv)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

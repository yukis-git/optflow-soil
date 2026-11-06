"""Accuracy and time of Python, Rust (1 thread / all threads), OpenCV DIS (medium, ultrafast) and Farneback on
synthetic pairs (default: uniform_0.01, noise_0.01 and large_motion at 512x512). Writes
results/<date>_speed_compare.json, which report.py turns into a table.

Accuracy = displacement RMS against the truth over the benchmark's evaluation region (eval_mask) and the pixels
with a finite estimate; valid fraction = share of the evaluation region with a finite estimate. Python and Rust
time = the implementation's own stage timers (coarse + fine + strain, I/O excluded); OpenCV time = wall time
of the flow call (uint8 conversion excluded). OpenCV flows are optical flow, not DIC: speed and accuracy
references only."""
import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import run_synth as rs  # noqa: E402
from bench import rustcli  # noqa: E402
from bench.cases import all_cases  # noqa: E402
from optflow_ref import synth  # noqa: E402
from optflow_ref.metrics import disp_errors, eval_mask  # noqa: E402

# the parameters of the OpenCV tutorial (the Python binding has no defaults of its own)
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
DEFAULT_CASES = "uniform_0.01,noise_0.01,large_motion"


def opencv_methods(cv2):
    def dis(preset):
        def run(a, b):
            return cv2.DISOpticalFlow_create(preset).calc(a, b, None)
        return run

    def farneback(a, b):
        return cv2.calcOpticalFlowFarneback(a, b, None, **FARNEBACK)

    return [("OpenCV DIS (MEDIUM)", dis(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)),
            ("OpenCV DIS (ULTRAFAST)", dis(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)),
            ("OpenCV Farneback (OpenCV tutorial parameters)", farneback)]


def best(rec_list, key):
    return min(r[key] for r in rec_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cases", default=DEFAULT_CASES, help="comma-separated case names (exact)")
    ap.add_argument("--repeats", type=int, default=3, help="runs of each Rust and OpenCV method (best time kept)")
    ap.add_argument("--py-repeats", type=int, default=1, help="runs of the Python reference (it takes ~25 s at 512)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    ap.add_argument("--bin", default=None, help="optflow binary (default: $OPTFLOW_BIN or rust/target/release)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import cv2

    names = [c for c in a.cases.split(",") if c]
    cases = [c for c in all_cases(a.size) if c.name in names]
    if sorted(c.name for c in cases) != sorted(names):
        ap.error(f"unknown case in {names}")
    os.makedirs(a.out, exist_ok=True)
    pattern = synth.make_pattern(a.size, a.size, seed=a.seed)
    n = a.size
    n_cpu = os.cpu_count()
    load_start = os.getloadavg()
    rows = []
    for case in cases:
        if case.group == "sequence":
            ap.error(f"{case.name} is a sequence; speed_compare takes pair cases")
        f = case.fields[0]
        gx, gy = synth.displacement(f, n, n)
        mask = eval_mask((n, n), f, case.margin)
        ref = rs.frame(pattern, n, None, case, seed=rs.SEED_REF)
        defo = rs.frame(pattern, n, f, case, seed=rs.SEED_DEF)

        def add(method, threads, secs, d, **extra):
            rows.append({"case": case.name, "method": method, "threads": threads, "disp_rms": d["rms"],
                         "valid_frac": d["valid_frac"], "seconds": min(secs), "seconds_all": secs,
                         "mpx_per_s": n * n / 1e6 / min(secs), **extra})
            print(f"{case.name:14s} {method:46s} rms {d['rms']:.4f} px  valid {d['valid_frac']:.4f}  "
                  f"{min(secs):9.4g} s  {n * n / 1e6 / min(secs):9.4g} Mpx/s", flush=True)

        py = [rs.run_python(case, pattern, n, a.seed) for _ in range(a.py_repeats)]
        add("Python", None, [r["algo_seconds"] for r in py], py[0]["disp"])
        for threads, name in ((1, "Rust (1 thread)"), (None, "Rust (all threads)")):
            rr = [rs.run_rust(case, pattern, n, a.seed, threads, a.bin) for _ in range(a.repeats)]
            add(name, threads or n_cpu, [r["algo_seconds"] for r in rr], rr[0]["disp"],
                wall_seconds=best(rr, "seconds"), cli_seconds=best(rr, "cli_seconds"),
                max_rss_bytes=max(r["max_rss_bytes"] or 0 for r in rr))
        u8 = [np.round(255.0 * np.clip(x, 0.0, 1.0)).astype(np.uint8) for x in (ref, defo)]
        for name, run in opencv_methods(cv2):
            secs, flow = [], None
            for _ in range(a.repeats):
                t = time.perf_counter()
                flow = run(*u8)
                secs.append(time.perf_counter() - t)
            add(name, cv2.getNumThreads(), secs, disp_errors(flow[..., 0].astype(np.float64),
                                                              flow[..., 1].astype(np.float64), gx, gy, mask))
        t_py = next(r["seconds"] for r in rows if r["case"] == case.name and r["method"] == "Python")
        for r in rows:
            if r["case"] == case.name:
                r["speedup_vs_python"] = t_py / r["seconds"]

    notes = [
        "DIS と Farneback はオプティカルフロー法で、DIC ではない。速度と精度の参考値としてのみ示す。",
        "OpenCV には uint8 フレーム（round(255·I)）を渡す。Python は浮動小数、Rust は 16bit PNG 経由の同じ画像を処理する。"
        "noise_0.01 は元から 8bit 量子化済みで同一の画像、他のケースでは OpenCV 側にだけ 8bit 量子化の誤差が加わる。",
        "OpenCV のフローも基準画素での変位（I1(x) ≈ I2(x + flow)）で、こちらの u と同じ規約（calcOpticalFlowFarneback と DIS の定義どおり）。",
        f"OpenCV {cv2.__version__}: cv2.getNumThreads() = {cv2.getNumThreads()}、useOptimized = {cv2.useOptimized()}。"
        f"Farneback の引数は OpenCV チュートリアルの値: {FARNEBACK}。DIS は DISOpticalFlow_create(プリセット) の既定値。",
        f"Rust (1 thread) = RAYON_NUM_THREADS=1、Rust (all threads) = 全スレッド（{n_cpu} 論理 CPU）。"
        f"Python は numpy/scipy の既定（自前の並列化なし）。",
        f"測定時のロードアベレージ（1/5/15 分）: 開始 {', '.join(f'{x:.1f}' for x in load_start)}、"
        f"終了 {', '.join(f'{x:.1f}' for x in os.getloadavg())}。",
    ]
    rec = {"date": datetime.date.today().isoformat(), "size": n, "cases": names, "rows": rows, "notes": notes,
           "time_basis": f"Python/Rust は自前タイマーの coarse+fine+strain（入出力除く、Rust と OpenCV は {a.repeats} 回の最良、"
                         f"Python は {a.py_repeats} 回）、OpenCV はフロー関数呼び出しの壁時計",
           "repeats": a.repeats, "py_repeats": a.py_repeats, "git": rs.git_rev("rust"), "git_python": rs.git_rev("python"),
           "env": rs.env(), "machine": platform.platform(), **rustcli.bin_fields(a.bin),
           "cv2": {"version": cv2.__version__, "threads": cv2.getNumThreads(), "optimized": cv2.useOptimized()},
           "farneback": FARNEBACK, "cpu_count": n_cpu, "loadavg_start": list(load_start),
           "loadavg_end": list(os.getloadavg())}
    path = os.path.join(a.out, f"{rec['date']}_speed_compare.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

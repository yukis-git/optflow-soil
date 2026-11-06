"""Build results/REPORT.md (+ figures) from the newest JSON of each case and implementation: accuracy table,
Python-vs-Rust parity table (from the kept fields), speed table, speed_compare table and real-data table."""
import argparse
import glob
import json
import math
import os
import statistics
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_DISP = 0.02
TARGET_STRAIN_SMALL = 1e-4  # eps <= 1 %
TARGET_STRAIN_10 = 3e-4  # eps = 10 %
TARGET_CRLB = 1.3
MIN_VALID = 0.99  # fraction of evaluation pixels that must carry a finite estimate
COARSE_FAILURE_MAX, COARSE_FAILURE_RMS = 1e-2, 1e-3  # reported pixels beside a coarse-stage failure [px]
PARITY_DISP = 1e-3  # plan-2 spec 6: Rust vs Python on every synthetic case [px]
PARITY_STRAIN = 1e-5
PARITY_NAN = 0.999  # share of pixels on which Python and Rust agree about valid / NaN
SPEED_RATIO = 30.0  # plan-2 spec 6: MEDIAN over the 512² pair cases of Rust (all threads) / Python, algorithm time
LOAD_DIFF_MAX = 3.0  # the speed verdict needs the two sides measured under similar load: median 1-minute loads
LOAD_MAX = 6.0  # that differ by at most this, and neither above this (otherwise the ratio is not judged)
MAIN_LABELS = ("python", "rust")  # rows of the accuracy table; the other labels are variants (rust-t1, python-q16)
LABEL_ORDER = {"python": 0, "rust": 1, "rust-t1": 2, "python-q16": 3, "rust-gpu": 4}
STAGES = ("coarse", "fine", "strain")


def label(r):
    return r.get("label", r["impl"])


def load(results):
    """Newest record of every (label, case); the file names start with the date."""
    newest = {}
    for p in sorted(glob.glob(os.path.join(results, "*.json"))):
        r = json.load(open(p))
        if "case" in r and "disp" in r:
            newest[(label(r), r["case"])] = r
    return newest


def load_other(results, pattern):
    """Records of the non-case JSON files matching pattern, oldest first."""
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results, pattern)))]


def verdict(r):
    d, s, g = r["disp"]["rms"], r["strain"], r["group"]
    if "codec" in r["params"]:  # lossy video: checks the input path, not the algorithm
        return "(記録のみ)"
    checks = []
    if g in ("subpixel", "rotation", "uniform", "simple_shear", "large", "illumination", "sequence"):
        checks.append(d <= TARGET_DISP)
    if g == "uniform":
        lim = TARGET_STRAIN_10 if r["params"]["exx"] >= 0.05 else TARGET_STRAIN_SMALL
        checks.append(all(math.isfinite(s[k]) and s[k] <= lim for k in ("Exx", "Eyy", "Exy")))
    if g == "noise":
        checks.append(r["rms_over_crlb"] <= TARGET_CRLB)
    if g == "coarse_failure":  # judged on the pixels it reports only; the valid fraction is recorded, not judged
        return "OK" if r["disp"]["max"] <= COARSE_FAILURE_MAX and d <= COARSE_FAILURE_RMS else "NG"
    if not checks:
        return "(記録のみ)"
    checks.append(r["disp"]["valid_frac"] >= MIN_VALID)
    return "OK" if all(checks) else "NG"


def fmt(x, f="{:.4f}"):
    return "—" if x is None else ("nan" if x != x else f.format(x))


def t4(x):
    """Seconds with 4 significant digits."""
    if x is None:
        return "—"
    if x != x:
        return "nan"
    return f"{x:.0f}" if abs(x) >= 1e4 else f"{x:.4g}"


def md_table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def algo_seconds(r):
    """Algorithm time (coarse + fine + strain; sequences summed over the frames); None when the record has no
    timers (a sequence recorded before the per-frame timers existed)."""
    if r.get("algo_seconds"):
        return r["algo_seconds"]
    per = r.get("per_frame")
    if per and all("timing" in p for p in per):
        return sum(sum(p["timing"][k] for k in STAGES) for p in per)
    t = r.get("timing")
    if t and all(k in t for k in STAGES):
        return sum(t[k] for k in STAGES)
    return None


def n_px(r):
    return r["size"] ** 2 * (r["params"]["frames"] if r["group"] == "sequence" else 1)


# ---------------------------------------------------------------- parity

FIELDS = ("ux", "uy", "Exx", "Eyy", "Exy")


class ParityError(Exception):
    """Two field sets that cannot be compared (missing, no common frame, different shapes)."""


def field_sets(root):
    """{name: dir} of the field sets below root: a pair case is root itself, a sequence its frame_* dirs."""
    if os.path.exists(os.path.join(root, "ux.npy")):
        return {"": root}
    return {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(root, "frame_*")))}


def load_fields(d):
    return {k: np.load(os.path.join(d, f"{k}.npy")).astype(np.float64) for k in FIELDS}


def short(d):
    return "/".join(d.rstrip("/").split("/")[-2:])


def worse(cur, v):
    """The larger of two error values; nan counts as the worst whichever of the two it is."""
    if cur != cur:
        return cur
    if v != v:
        return v
    return max(cur, v)


def parity(dir_a, dir_b):
    """Differences of two kept field sets (the worst over the common sequence frames): RMS of the displacement
    vector difference over the pixels valid in both, RMS of the Exx, Eyy, Exy differences over the pixels where
    both are finite, and how well the NaN patterns agree. Raises ParityError if the sets cannot be compared."""
    sa, sb = field_sets(dir_a), field_sets(dir_b)
    if not sa or not sb:
        raise ParityError("no fields in " + short(dir_a if not sa else dir_b))
    common = sorted(set(sa) & set(sb))
    if not common:
        raise ParityError(f"no common frame ({short(dir_a)}: {sorted(sa)}, {short(dir_b)}: {sorted(sb)})")
    out = None
    for name in common:
        a, b = load_fields(sa[name]), load_fields(sb[name])
        if a["ux"].shape != b["ux"].shape:
            raise ParityError(f"shapes differ {a['ux'].shape} vs {b['ux'].shape} (stale fields?) in "
                              f"{short(sa[name])} / {short(sb[name])}")
        va = np.isfinite(a["ux"]) & np.isfinite(a["uy"])
        vb = np.isfinite(b["ux"]) & np.isfinite(b["uy"])
        both = va & vb
        dd = np.hypot(a["ux"] - b["ux"], a["uy"] - b["uy"])[both]
        cur = {"n_both": int(both.sum()), "n_only_a": int((va & ~vb).sum()), "n_only_b": int((vb & ~va).sum()),
               "nan_agree": float((va == vb).mean()),
               "disp_rms": float(np.sqrt(np.mean(dd**2))) if dd.size else float("nan"),
               "disp_max": float(dd.max()) if dd.size else float("nan")}
        for k in ("Exx", "Eyy", "Exy"):
            m = both & np.isfinite(a[k]) & np.isfinite(b[k])
            cur[k] = float(np.sqrt(np.mean((a[k] - b[k])[m] ** 2))) if m.any() else float("nan")
        if out is None:
            out = cur
        else:  # worst frame per metric
            for k, v in cur.items():
                if k in ("nan_agree", "n_both"):
                    out[k] = min(out[k], v)
                elif k.startswith("n_only"):
                    out[k] = max(out[k], v)
                else:
                    out[k] = worse(out[k], v)
    return out


def parity_ok(p):
    """Spec 6 (plan 2) plus the NaN pattern: a nan anywhere fails (the comparisons are False for nan)."""
    return bool(p["disp_rms"] <= PARITY_DISP and all(p[k] <= PARITY_STRAIN for k in ("Exx", "Eyy", "Exy"))
                and p["nan_agree"] >= PARITY_NAN)


def parity_section(results, rows):
    """(lines, ok): ok is None when there is no Rust record at all, False when any case is missing, not compared,
    misses the tolerance or has a different verdict."""
    py = {c: r for (lab, c), r in rows.items() if lab == "python"}
    rs = {c: r for (lab, c), r in rows.items() if lab == "rust"}
    if not rs:
        return [], None
    cases = sorted(set(py) | set(rs), key=lambda c: ((py.get(c) or rs[c])["group"], c))
    lines = ["", "## Python と Rust の一致（合成ケース）", "",
             "フレームは Rust 側に 16bit PNG（動画ケースは同じ MJPG 動画）で渡す。保存した .npy（float32）の差。"
             "変位差 RMS = 両方で有効な画素での (ux, uy) 差ベクトルの RMS、ひずみ差 RMS = Exx, Eyy, Exy それぞれの RMS"
             "（両方で有限な画素）。NaN 一致 = 変位の有効/無効が一致する画素の割合（py のみ / rs のみ有効な画素数）。"
             "系列ケースは保存した先頭・中間・最終フレームのうち最悪値（nan はどの順でも最悪として残る）。"
             f"合格条件: 変位差 RMS ≤ {PARITY_DISP:g} px かつ ひずみ差 RMS ≤ {PARITY_STRAIN:g} かつ NaN 一致 ≥ {PARITY_NAN:g}"
             "（全ケース）。判定の一致は、両方の記録がある全ケースで、フィールドの有無によらず調べる。", ""]
    table, misses, vdiff, not_cmp, missing, n_cmp = [], [], [], [], [], 0
    dash = ["—"] * 7
    for case in cases:
        rp, rr = py.get(case), rs.get(case)
        vp, vr = (verdict(rp) if rp else "—"), (verdict(rr) if rr else "—")
        if rp is None or rr is None:
            missing.append((case, "python" if rp is None else "rust"))
            table.append([case, *dash, vp, vr, "**欠落**"])
            continue
        if vp != vr:
            vdiff.append((case, vp, vr))
        reason = f"size {rp['size']} vs {rr['size']}" if rp["size"] != rr["size"] else None
        p = None
        if reason is None:
            try:
                p = parity(os.path.join(results, "fields", "python", case), os.path.join(results, "fields", "rust", case))
            except ParityError as e:
                reason = str(e)
        if p is None:
            not_cmp.append((case, reason))
            table.append([case, *dash, vp, vr, "**未比較**"])
            continue
        n_cmp += 1
        ok = parity_ok(p)
        if not ok:
            misses.append((case, p))
        table.append([case, fmt(p["disp_rms"], "{:.2e}"), fmt(p["disp_max"], "{:.2e}"), fmt(p["Exx"], "{:.1e}"),
                      fmt(p["Eyy"], "{:.1e}"), fmt(p["Exy"], "{:.1e}"), f"{p['nan_agree']:.6f}",
                      f"{p['n_only_a']} / {p['n_only_b']}", vp, vr, "OK" if ok else "**NG**"])
    lines += md_table(["case", "変位差RMS [px]", "変位差最大 [px]", "Exx差RMS", "Eyy差RMS", "Exy差RMS", "NaN一致", "py/rs のみ有効",
                       "判定 py", "判定 rs", "一致"], table)
    m = len(cases)
    lines += ["", f"比較できた: {n_cmp} / {m} ケース（未比較 {len(not_cmp)}、欠落 {len(missing)}）。"
              f"合格: {n_cmp - len(misses)} / {n_cmp}。"]
    if misses:
        lines += ["", "合格条件を外したケース:"] + [
            f"- {c}: 変位差 RMS {p['disp_rms']:.2e} px, ひずみ差 RMS Exx {p['Exx']:.1e} / Eyy {p['Eyy']:.1e} / "
            f"Exy {p['Exy']:.1e}, NaN 一致 {p['nan_agree']:.6f}" for c, p in misses]
    if not_cmp:
        lines += ["", "未比較（両方の記録はあるが、比較できるフィールドがない）:"] + [f"- {c}: {why}" for c, why in not_cmp]
    if missing:
        lines += ["", "欠落（片方の実装の記録しかない）:"] + [f"- {c}: {side} の記録がない" for c, side in missing]
    n_both_rec = m - len(missing)
    lines += ["", f"判定の違い（両方の記録がある {n_both_rec} ケース）: " + ("なし（全ケースで同じ）" if not vdiff else f"{len(vdiff)} ケース")]
    lines += [f"- {c}: python {a} / rust {b}" for c, a, b in vdiff]

    def pq(c, a, b):
        ra, rb = rows[(a, c)], rows.get((b, c))
        if rb is None:
            raise ParityError(f"no {b} record")
        if ra["size"] != rb["size"]:
            raise ParityError(f"size {ra['size']} vs {rb['size']}")
        return parity(os.path.join(results, "fields", a, c), os.path.join(results, "fields", b, c))

    q, q_notes = [], []
    for c in sorted(c for (lab, c) in rows if lab == "python-q16"):
        try:
            q.append((c, pq(c, "python-q16", "python"), pq(c, "python-q16", "rust")))
        except ParityError as e:
            q_notes.append((c, str(e)))
    if q or q_notes:
        lines += ["", "16bit 量子化（PNG 往復）の影響: Python を浮動小数入力と 16bit 丸め入力（python-q16）で走らせた変位・ひずみの差。"
                  "右の 2 列は python-q16 と Rust の差（量子化を除いた実装差: float32 と演算順序）。", ""]
    if q:
        lines += md_table(["case", "変位差RMS [px]", "変位差最大 [px]", "Exx差RMS", "Eyy差RMS", "Exy差RMS",
                           "q16 vs Rust 変位差RMS [px]", "q16 vs Rust ひずみ差RMS(最大)"],
                          [[c, fmt(p["disp_rms"], "{:.2e}"), fmt(p["disp_max"], "{:.2e}"), fmt(p["Exx"], "{:.1e}"),
                            fmt(p["Eyy"], "{:.1e}"), fmt(p["Exy"], "{:.1e}"), fmt(r["disp_rms"], "{:.2e}"),
                            fmt(max(r["Exx"], r["Eyy"], r["Exy"]), "{:.1e}")] for c, p, r in q])
    lines += [f"- 未比較: {c}（{why}）" for c, why in q_notes]
    return lines, not (misses or vdiff or not_cmp or missing)


# ---------------------------------------------------------------- speed

def median_load(recs):
    """Median 1-minute load average over the start and the end of every record; None when no record carries one."""
    xs = [x[0] for r in recs for x in (r.get("loadavg_start"), r.get("loadavg_end")) if x]
    return statistics.median(xs) if xs else None


def loads_comparable(py_load, rs_load):
    """The two sides ran under similar, moderate load (a missing load cannot be shown to be, so it is not)."""
    return (py_load is not None and rs_load is not None
            and abs(py_load - rs_load) <= LOAD_DIFF_MAX and max(py_load, rs_load) <= LOAD_MAX)


def ratio_line(title, items, accept=True, loads=(None, None)):
    """One line of the Rust(all)/Python ratio statistic over [(case, ratio)]. accept: the line is judged against the
    target (pairs), otherwise it only says how many cases are at least 30x (reference). The judged figure is the
    MEDIAN ratio (>= SPEED_RATIO gives 達成; the controller's ruling, plan-2 spec 6); the minimum and the cases below
    SPEED_RATIO (all listed by name) are information and never decide. loads: the median 1-minute load (Python, Rust)
    of the records behind the ratios, shown next to them; a judged line gives no verdict when they are not comparable
    (differ by more than LOAD_DIFF_MAX, either above LOAD_MAX, or unknown)."""
    if not items:
        return f"{title}: 該当するケースなし（512² で Python と Rust の両方がそろったものがない）。"
    vals = sorted(v for _, v in items)
    median = statistics.median(vals)
    bad = [(c, v) for c, v in items if v < SPEED_RATIO]
    listed = ("（" + ", ".join(f"{c} {v:.1f}×" for c, v in bad) + "）") if bad else ""
    py_load, rs_load = loads
    shown = tuple("—" if x is None else f"{x:.1f}" for x in loads)
    head = (f"{title}（{len(vals)} 件）の比 Rust(全)/Python: 最小 {vals[0]:.1f}× / 中央値 {median:.1f}× / "
            f"最大 {vals[-1]:.1f}×。負荷（1 分の中央値）: Python {shown[0]} / Rust {shown[1]}。")
    if not accept:
        return head + f"{SPEED_RATIO:g}× 未満: {len(bad)} 件{listed}（合否の対象外）。"
    if not loads_comparable(py_load, rs_load):
        return head + f"負荷条件が揃っていないため判定しない（Python 負荷 {shown[0]} / Rust 負荷 {shown[1]}）"
    return (head + f"目標（判定は中央値）≥ {SPEED_RATIO:g}×: " + ("達成" if median >= SPEED_RATIO else f"**未達**（中央値 {median:.1f}×）")
            + f"。最小値と {SPEED_RATIO:g}× 未満のケースは参考（合否の対象外）: {SPEED_RATIO:g}× 未満 {len(bad)} 件{listed}。")


def speed_section(rows):
    cases = sorted({c for (lab, c) in rows if lab == "rust"}, key=lambda c: (rows[("rust", c)]["group"], c))
    table, pair_ratios, seq_ratios = [], [], []
    pair_recs, seq_recs = ([], []), ([], [])  # (Python, Rust) records behind each list of ratios
    for c in cases:
        py, rs, r1 = rows.get(("python", c)), rows[("rust", c)], rows.get(("rust-t1", c))
        ap, ar, a1 = (algo_seconds(x) if x else None for x in (py, rs, r1))

        def mpx(x, a):
            return None if a is None else n_px(x) / 1e6 / a

        mp, mr, m1 = mpx(py, ap) if py else None, mpx(rs, ar), mpx(r1, a1) if r1 else None
        ratio = None if mp is None or mr is None else mr / mp
        if ratio is not None and rs["size"] == 512:
            seq = rs["group"] == "sequence"
            (seq_ratios if seq else pair_ratios).append((c, ratio))
            for side, rec in zip(seq_recs if seq else pair_recs, (py, rs)):
                side.append(rec)
        table.append([c, str(rs["size"]), t4(ap), fmt(mp, "{:.3f}"), t4(a1), fmt(m1, "{:.3f}"), t4(ar), fmt(mr, "{:.3f}"),
                      t4(rs["seconds"]), fmt(ratio, "{:.1f}"),
                      fmt(None if (m1 is None or mp is None) else m1 / mp, "{:.1f}")])
    if not table:
        return []
    thr = next((x.get("cpu_count") for x in (rows[("rust", c)] for c in cases) if x.get("cpu_count")), None)
    lines = ["", "## 速度", "",
             "アルゴリズム時間 = timing の coarse + fine + strain の合計（入出力を除く。系列は全フレームの合計）。"
             "Mpx/s = 画素数 × フレーム数 / アルゴリズム時間。Rust(1) は RAYON_NUM_THREADS=1、Rust(全) は全スレッド"
             + (f"（{thr} 論理 CPU）" if thr else "") + "。壁時計(Rust) はプロセス起動・PNG/npy の入出力を含む。"
             "系列・動画ケースの timing は各フレームの最後の track_pair と strain だけ（参照フレーム更新時の再実行、"
             "合成・平均信頼度の計算は含まない）で、Python・Rust とも下限。その比は参考値。"
             "目標（合否）の対象は、512² のペアケースの比 Rust(全)/Python の中央値だけ（両側を同じ静かな時間帯に測ったもの）。"
             "ケースごとの最小値、30× 未満のケース、「他手法との比較」の 3 ケースの値は並べて示すが、判定しない。", ""]
    lines += md_table(["case", "サイズ", "Python [s]", "Python Mpx/s", "Rust(1) [s]", "Rust(1) Mpx/s", "Rust(全) [s]",
                       "Rust(全) Mpx/s", "壁時計(Rust全) [s]", "比 Rust(全)/Python", "比 Rust(1)/Python"], table)
    lines += ["", ratio_line("512² のペアケース（合否の対象）", pair_ratios, loads=tuple(map(median_load, pair_recs)))]
    if seq_ratios or any(rows[("rust", c)]["group"] == "sequence" and rows[("rust", c)]["size"] == 512 for c in cases):
        lines += ["", ratio_line("512² の系列・動画ケース（参考。下限タイマー）", seq_ratios, accept=False,
                                 loads=tuple(map(median_load, seq_recs)))]
    return lines


# ---------------------------------------------------------------- speed_compare

def compare_section(results):
    recs = load_other(results, "*_speed_compare.json")
    if not recs:
        return []
    r = recs[-1]
    lines = ["", "## 他手法との比較（精度と速度の参照）", "",
             f"{r['size']}² の合成ケース。変位 RMS は評価領域（縁から margin 画素以上内側）の有限画素で真値と比較、"
             f"有効率 = 評価領域のうち推定値が有限な割合。時間は {r['time_basis']}。", ""]
    lines += [f"- {n}" for n in r["notes"]]
    for case in r["cases"]:
        lines += ["", f"**{case}**", ""]
        lines += md_table(["手法", "スレッド", "変位RMS [px]", "有効率", "時間 [s]", "Mpx/s", "Python 比の速さ"],
                          [[m["method"], "—" if m["threads"] is None else str(m["threads"]),
                            fmt(m["disp_rms"], "{:.4f}"), fmt(m["valid_frac"], "{:.4f}"),
                            t4(m["seconds"]), fmt(m["mpx_per_s"], "{:.3f}"),
                            fmt(m.get("speedup_vs_python"), "{:.1f}×")] for m in r["rows"] if m["case"] == case])
    return lines


# ---------------------------------------------------------------- real data

def real_records(results):
    recs = {}
    for r in load_other(results, "*_real*.json"):
        recs[(label(r), r["scale"], r.get("mode", "both"))] = r  # oldest first -> newest wins
    return recs


def real_section(results):
    recs = real_records(results)
    if not recs:
        return []
    table = []
    for (lab, scale, mode), r in sorted(recs.items(), key=lambda kv: (LABEL_ORDER.get(kv[0][0], 9), kv[0][1])):
        x0, y0, x1, y1 = r["crop"]
        pairs = r.get("pairs", [])
        fb = r.get("forward_backward_1_2")
        cd = r.get("chain_vs_direct")
        cli = r.get("cli", {})
        rss = cli.get("max_rss_bytes", r.get("max_rss_bytes"))
        if cli:  # Rust: the largest of the CLI processes
            rss_cell = "—" if rss is None else f"{rss / 1e9:.2f}（最大の CLI: {cli.get('max_rss_call', '?')}）"
        else:  # Python: the whole run_real process
            rss_cell = "—" if rss is None else f"{rss / 1e9:.2f}（プロセス全体）"
        foot = cli.get("peak_footprint_bytes")
        split = " / ".join(f"{k}×{v['n']} {t4(v['wall_seconds'])}" for k, v in cli.get("by_kind", {}).items()) or "—"
        table.append([lab, str(scale), mode + ("※" if mode == "fullframe" else ""), f"{x1 - x0}×{y1 - y0}", t4(r["seconds"]),
                      t4(cli.get("wall_seconds")), t4(cli.get("algo_seconds")), split, rss_cell,
                      "—" if foot is None else f"{foot / 1e9:.2f}",
                      " / ".join(fmt(p["valid_frac"], "{:.4f}") for p in pairs) or "—",
                      "—" if not fb else f"{fb['median_px']:.3f} / {fb['p90_px']:.3f} / {fb['rms_px']:.3f}",
                      "—" if not cd else f"{cd['median_px']:.3f} / {cd['p90_px']:.3f} ({cd['compared_frac']:.3f})"])
    lines = ["", "## 実データ（実画像）", "",
             "ROI の砂の帯を縁（左右 640、上下 128 の全解像度画素）付きで切り出した範囲。時間 = 全処理の壁時計（Python: 系列＋全ペアの"
             "前後方向＋保存。Rust: CLI 呼び出しと Python 側の集計を含む）。CLI 計/アルゴリズム計 = Rust の各プロセスの壁時計"
             "（/usr/bin/time -l）と timing の合計、内訳 = 呼び出しの種類ごと（回数×壁時計 [s]）。"
             "有効率はペア 1→2 / 2→3 / 3→4。FB = 順逆方向の整合（ペア 1→2、中央値 / p90 / RMS [px]）、"
             "連鎖 vs 直接 = ペア 1→2→3→4 の連鎖と 1→4 の直接の差（中央値 / p90 [px]、比較できた割合）。", ""]
    lines += md_table(["impl", "scale", "mode", "切り出し [px]", "時間 [s]", "CLI計 [s]", "アルゴリズム計 [s]", "CLI内訳 [s]",
                       "ピークRSS [GB]", "フットプリント [GB]", "有効率", "FB [px]", "連鎖 vs 直接 [px]"], table)
    lines += ["", "ピークRSS は測っているものが違う: Python は run_real のプロセス全体（画像・全フィールドを保持）、Rust は最大の CLI プロセス"
              "1 つ（RSS = /usr/bin/time -l の maximum resident set size、フットプリント = macOS の peak memory footprint）。"
              "Rust 側のハーネス自体のメモリは含まない（記録の harness_max_rss_bytes）。"]
    if any(k[0].startswith("rust") for k in recs):
        lines += ["", "注: Rust の切り出しの行は、Python と同じ作業量（系列 3 ＋ 前方向 3 ＋ 後方向 3 のペア単位 9）。ROI は厳密な"
                  "マスク（`--mask`）で渡し、フレームは Python が切り出し済みなので CLI の自動切り出しは `--crop off`。FB の逆方向と連鎖は"
                  "CLI の `--raw` が書く未マスクの場（Result.raw_ux 相当）を使う。",
                  "", "注: Python と Rust の実データの比較には JPEG デコードの差（Task 9 の実測: 最大 1.6 階調、平均 0.009）が"
                  "含まれ得るが、欠陥ではない。run_real.py の切り出し範囲は Python がデコードして 16bit PNG で Rust に渡すため、"
                  "切り出しの実行にはこの差は入らない（全画面の pair は Rust がデコードする）。"]
    if any(k[2] == "fullframe" for k in recs):
        lines += ["", "※ fullframe の行は全画像（JPEG）と `--roi` を CLI に渡し、CLI の既定の自動切り出し（crop.json、出力は全画像の大きさ）で"
                  "追跡する。切り出しの行（縁が左右 640、上下 128）とは範囲も縁も違い、有効率の分母も ROI 全体なので比較できない。"
                  "時間とメモリの規模を見るための行。"]
    return lines


# ---------------------------------------------------------------- provenance

def provenance_section(results, rows):
    """Which Rust binaries produced the records (sha256), whether they were older than the sources, and how busy
    the machine was. Returns (lines, n_binaries, n_stale)."""
    recs = [r for r in rows.values() if r.get("impl") == "rust"]
    recs += [r for r in load_other(results, "*_speed_compare.json")[-1:] if "bin" in r or "bin_stale" in r]
    recs += [r for r in real_records(results).values() if r.get("impl") == "rust"]
    groups = {}
    for r in recs:
        b = r.get("bin") if isinstance(r.get("bin"), dict) else None  # older records: a path string
        key = b.get("sha256") if b else None
        g = groups.setdefault(key, {"bin": b, "n": 0, "stale": 0, "git": set(), "git_python": set()})
        g["n"] += 1
        g["stale"] += bool(r.get("bin_stale"))
        g["git"].add(str(r.get("git")))
        g["git_python"].add(str(r.get("git_python", "?")))
    lines = []
    n_stale = sum(g["stale"] for g in groups.values())
    if groups:
        lines += ["", "## 実行条件（Rust バイナリ）", "",
                  "`git` は rust/ のソースのリビジョンだけで、どのバイナリが走ったかは sha256 で分かる。stale = バイナリが rust/ の"
                  "ビルド入力（.rs / .toml / .lock）の最新の更新時刻より古い。", ""]
        lines += md_table(["sha256（先頭 12）", "サイズ [B]", "ビルド時刻", "記録数", "stale", "rust/ 版", "python/ 版"],
                          [[(k or "不明（記録なし）")[:12], str(g["bin"]["size"]) if g["bin"] else "—",
                            g["bin"]["mtime_iso"] if g["bin"] else "—", str(g["n"]), str(g["stale"]),
                            ", ".join(sorted(g["git"])), ", ".join(sorted(g["git_python"]))] for k, g in groups.items()])
        if len([k for k in groups if k]) > 1:
            lines += ["", f"**警告**: {len([k for k in groups if k])} 種類のバイナリの記録が混在している。"]
        if n_stale:
            lines += ["", f"**警告**: バイナリが rust/ のソースより古い記録が {n_stale} 件ある（再ビルドして測り直す）。"]
        if None in groups:
            lines += ["", f"**警告**: バイナリの情報がない Rust の記録が {groups[None]['n']} 件ある。"]
    loads = []
    for r in list(rows.values()) + list(real_records(results).values()):
        loads += [x[0] for x in (r.get("loadavg_start"), r.get("loadavg_end")) if x]
    if loads:
        loads.sort()
        lines += ["", f"実行中のロードアベレージ（1 分）: 最大 {loads[-1]:.1f} / 中央値 {loads[len(loads) // 2]:.1f}"
                  f"（{len(loads)} 点、各記録の開始時と終了時）。時間の比較は静かなマシンでの値だけを使う。"]
    return lines, len([k for k in groups if k]), n_stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    a = ap.parse_args()
    rows = load(a.results)
    figdir = os.path.join(a.results, "fig")
    os.makedirs(figdir, exist_ok=True)
    lines = ["# 精度検証レポート（合成データ）", "",
             f"目標: 変位 RMS ≤ {TARGET_DISP} px（ノイズなし）、一様ひずみ ≤ 1e-4（ε≤1%）/ ≤ 3e-4（ε=10%）、"
             f"ノイズ時 ≤ {TARGET_CRLB}×理論下限。", "",
             "時間: アルゴリズム = coarse + fine + strain の合計（入出力を除く）。壁時計 = Python はフレーム生成＋追跡、"
             "Rust は入力の書き出し・プロセス全体・npy 読み込み。", ""]
    shown = [r for (lab, c), r in rows.items() if lab in MAIN_LABELS]
    table, verdicts = [], {lab: [] for lab in MAIN_LABELS}
    for r in sorted(shown, key=lambda r: (LABEL_ORDER[label(r)], r["group"], r["case"])):
        d, s = r["disp"], r["strain"]
        v = verdict(r)
        verdicts[label(r)].append(v)
        extra = f" (×{r['rms_over_crlb']:.2f} 下限)" if "rms_over_crlb" in r else ""
        table.append([label(r), r["case"], f"{fmt(d['rms'])}{extra}", fmt(d["max"], "{:.3f}"),
                      f"{fmt(d['bias_x'], '{:+.4f}')}/{fmt(d['bias_y'], '{:+.4f}')}", fmt(s["Exx"], "{:.1e}"),
                      fmt(s["Eyy"], "{:.1e}"), fmt(s["gmax"], "{:.1e}"), fmt(d["valid_frac"], "{:.4f}"),
                      t4(algo_seconds(r)), t4(r["seconds"]), v])
    lines += md_table(["impl", "case", "変位RMS [px]", "最大 [px]", "偏り x/y [px]", "Exx RMS", "Eyy RMS", "γmax RMS", "有効率",
                       "アルゴリズム [s]", "壁時計 [s]", "判定"], table)
    summaries = {}
    for lab, vs in verdicts.items():
        if vs:
            summaries[lab] = f"{vs.count('OK')} OK / {vs.count('NG')} NG / {vs.count('(記録のみ)')} record-only"
    lines += [""] + [f"判定の集計（{lab}）: {s}" for lab, s in summaries.items()]
    n_ng = sum(vs.count("NG") for vs in verdicts.values())
    py = [r for (lab, c), r in rows.items() if lab == "python"]
    sub = sorted((r for r in py if r["group"] == "subpixel"), key=lambda r: r["params"]["tx"])
    if sub:
        plt.figure(figsize=(5, 3.5))
        plt.plot([r["params"]["tx"] for r in sub], [r["disp"]["bias_x"] for r in sub], "o-", label="bias x")
        plt.plot([r["params"]["tx"] for r in sub], [r["disp"]["rms"] for r in sub], "s--", label="RMS")
        plt.xlabel("sub-pixel shift [px]"); plt.ylabel("error [px]"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "subpixel.png"), dpi=120); plt.close()
        lines += ["", "![subpixel](fig/subpixel.png)"]
    band = [r for r in py if r["group"] == "shear_band"]
    if band:
        plt.figure(figsize=(5, 3.5))
        for order in (1, 2):
            rr = sorted((r for r in band if r["params"]["shape_order"] == order),
                        key=lambda r: (r["params"]["width"], r["params"]["delta"]))
            plt.plot([f"D{r['params']['delta']}/w{r['params']['width']}" for r in rr],
                     [r["disp"]["rms"] for r in rr], "o-", label=f"shape_order={order}")
        plt.ylabel("displacement RMS [px]"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "shear_band.png"), dpi=120); plt.close()
        lines += ["", "せん断帯（分解能の記録。合否なし）:", "", "![shear band](fig/shear_band.png)"]
    par, par_ok = parity_section(a.results, rows)
    lines += par
    lines += speed_section(rows)
    lines += compare_section(a.results)
    lines += real_section(a.results)
    prov, n_bin, n_stale = provenance_section(a.results, rows)
    lines += prov
    with open(os.path.join(a.results, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows")
    for lab, s in summaries.items():
        print(f"{lab}: {s}")
    if par_ok is None:
        print("parity: no Rust records")
    else:
        print("parity: " + ("every case compared, all within tolerance, verdicts equal" if par_ok
                            else "NOT CLEAN - see REPORT.md (misses, verdict differences, cases not compared or missing)"))
    if n_bin:
        print(f"rust binaries: {n_bin} distinct" + (f", {n_stale} STALE record(s)" if n_stale else "")
              + (" - MIXED, measure again with one binary" if n_bin > 1 else ""))
    return 1 if n_ng else (2 if par_ok is False else 0)  # 1: a verdict is NG; 2: the parity check is not clean


if __name__ == "__main__":
    sys.exit(main())

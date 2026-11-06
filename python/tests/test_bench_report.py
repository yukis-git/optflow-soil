"""The speed verdict of bench/report.py with fabricated records. The judged figure is the MEDIAN Rust(all)/Python ratio
over the 512² pair cases (>= 30 gives 達成; the minimum and the cases below 30 are listed but never decide), and it is
given only when the two sides were measured under comparable load (median 1-minute load averages within 3 of each
other, neither above 6)."""
import pytest

from bench import report

PAIR_LINE = "512² のペアケース"
SEQ_LINE = "512² の系列・動画ケース"
MET = "目標（判定は中央値）≥ 30×: 達成"
MISSED = "目標（判定は中央値）≥ 30×: **未達**"


def _rec(group, algo_seconds, load=None, load_end=None):
    """A record of a 512 x 512 case; load: 1-minute load average at the start (and at the end, load_end), None: no record."""
    r = {"group": group, "size": 512, "algo_seconds": algo_seconds, "seconds": algo_seconds, "params": {"frames": 4}}
    if load is not None:
        r["loadavg_start"] = [load, 1.0, 1.0]
        r["loadavg_end"] = [load if load_end is None else load_end, 1.0, 1.0]
    return r


def _rows(py_loads, rs_loads, ratio, group="pair", py_seconds=12.0):
    """One case per entry of the load lists (Python and Rust of a case are paired by position); the Rust time is
    chosen so that the ratio Rust(all)/Python of every case is `ratio`."""
    rows = {}
    for i, (lp, lr) in enumerate(zip(py_loads, rs_loads)):
        rows[("python", f"case{i}")] = _rec(group, py_seconds, lp)
        rows[("rust", f"case{i}")] = _rec(group, py_seconds / ratio, lr)
    return rows


def _line(rows, start):
    (line,) = [x for x in report.speed_section(rows) if x.startswith(start)]
    return line


def test_median_load_pools_the_start_and_the_end_of_every_record():
    recs = [_rec("pair", 1.0, 1.0, 5.0), _rec("pair", 1.0, 2.0, 2.0), _rec("pair", 1.0, 9.0, 9.0), {"group": "pair"}]
    assert report.median_load(recs) == 3.5  # 1, 5, 2, 2, 9, 9 (the record without a load adds nothing)
    assert report.median_load([{"group": "pair"}]) is None
    assert report.median_load([]) is None


@pytest.mark.parametrize("ratio, verdict", [(35.0, MET), (28.0, MISSED)])
def test_comparable_loads_keep_the_verdict_and_show_both_medians(ratio, verdict):
    line = _line(_rows([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], ratio), PAIR_LINE)
    assert "負荷（1 分の中央値）: Python 2.0 / Rust 3.0" in line
    assert verdict in line
    assert "判定しない" not in line
    assert f"{ratio:.1f}×" in line


@pytest.mark.parametrize("py_load, rs_load", [
    (9.0, 1.0),   # a loaded Python against a quiet Rust run: the 120x of Task 10
    (1.0, 9.0),   # the other way round
    (7.0, 7.0),   # equal, but both above 6
    (5.5, 1.5),   # both at most 6, but 4 apart
    (6.5, 6.0),   # close, one above 6
    (0.0, 3.5),   # more than 3 apart at low load
])
def test_incomparable_loads_give_no_verdict(py_load, rs_load):
    line = _line(_rows([py_load] * 3, [rs_load] * 3, 120.0), PAIR_LINE)
    assert f"負荷条件が揃っていないため判定しない（Python 負荷 {py_load:.1f} / Rust 負荷 {rs_load:.1f}）" in line
    assert "達成" not in line and "未達" not in line and "目標" not in line
    assert "120.0×" in line  # the ratios themselves are still shown


@pytest.mark.parametrize("py_load, rs_load", [
    (6.0, 6.0),   # exactly at the limit is allowed
    (1.0, 4.0),   # exactly 3 apart is allowed
    (4.0, 1.0),
])
def test_the_limits_themselves_are_comparable(py_load, rs_load):
    line = _line(_rows([py_load] * 2, [rs_load] * 2, 28.0), PAIR_LINE)
    assert "判定しない" not in line and "**未達**" in line


def test_the_median_not_the_maximum_decides():
    # one very busy Python record among quiet ones does not withhold the verdict
    py = [1.0, 1.0, 1.0, 1.0, 20.0]
    line = _line(_rows(py, [2.0] * 5, 35.0), PAIR_LINE)
    assert "Python 1.0 / Rust 2.0" in line and MET in line
    # ... and a majority of busy ones does
    line = _line(_rows([20.0] * 3 + [1.0] * 2, [2.0] * 5, 35.0), PAIR_LINE)
    assert "判定しない（Python 負荷 20.0 / Rust 負荷 2.0）" in line


def test_a_missing_load_record_gives_no_verdict():
    rows = _rows([1.0, 1.0], [None, None], 35.0)
    line = _line(rows, PAIR_LINE)
    assert "負荷条件が揃っていないため判定しない（Python 負荷 1.0 / Rust 負荷 —）" in line
    assert "達成" not in line


def test_the_sequence_line_shows_the_loads_and_never_gives_a_verdict():
    rows = _rows([9.0], [1.0], 20.0, group="sequence")
    line = _line(rows, SEQ_LINE)
    assert "負荷（1 分の中央値）: Python 9.0 / Rust 1.0" in line
    assert "合否の対象外" in line and "判定しない" not in line


def test_the_pair_loads_come_from_the_pair_records_only():
    rows = _rows([1.0, 1.0], [2.0, 2.0], 35.0)
    seq = _rows([9.0], [9.0], 20.0, group="sequence")
    rows.update({(lab, "seq0"): rec for (lab, _), rec in seq.items()})
    pair = _line(rows, PAIR_LINE)
    assert "Python 1.0 / Rust 2.0" in pair and MET in pair
    assert "Python 9.0 / Rust 9.0" in _line(rows, SEQ_LINE)


def _rows_by_ratio(ratios, py_load=1.0, rs_load=2.0):
    """One pair case per ratio Rust(all)/Python, measured under the given loads (comparable by default)."""
    rows = {}
    for i, ratio in enumerate(ratios):
        rows[("python", f"case{i}")] = _rec("pair", 12.0, py_load)
        rows[("rust", f"case{i}")] = _rec("pair", 12.0 / ratio, rs_load)
    return rows


def test_a_median_of_at_least_30_gives_the_target_although_one_case_is_below_30():
    # median 31, minimum 26: 達成; the case below 30 is still listed by name, as information
    line = _line(_rows_by_ratio([26.0, 31.0, 31.0, 33.0, 36.0]), PAIR_LINE)
    assert MET in line and "未達" not in line
    assert "最小 26.0× / 中央値 31.0× / 最大 36.0×" in line
    assert "30× 未満 1 件（case0 26.0×）" in line
    assert "参考（合否の対象外）" in line


def test_every_case_below_30_is_listed_by_name_next_to_a_met_target():
    line = _line(_rows_by_ratio([20.0, 27.5, 29.5, 31.0, 32.0, 33.0, 34.0]), PAIR_LINE)
    assert MET in line
    assert "30× 未満 3 件（case0 20.0×, case1 27.5×, case2 29.5×）" in line


def test_a_median_below_30_misses_the_target_although_the_maximum_is_far_above():
    line = _line(_rows_by_ratio([25.0, 28.0, 29.0, 40.0, 60.0]), PAIR_LINE)
    assert MISSED in line and "（中央値 29.0×）" in line and "達成" not in line
    assert "30× 未満 3 件（case0 25.0×, case1 28.0×, case2 29.0×）" in line


def test_no_case_below_30_says_none():
    line = _line(_rows_by_ratio([31.0, 35.0, 40.0]), PAIR_LINE)
    assert MET in line and "30× 未満 0 件。" in line


def test_the_median_at_exactly_30_is_met_and_an_even_count_uses_the_mean_of_the_middle_two():
    loads = (1.0, 2.0)
    assert MET in report.ratio_line("t", [("a", 29.0), ("b", 30.0), ("c", 31.0)], loads=loads)
    line = report.ratio_line("t", [("a", 29.0), ("b", 31.0)], loads=loads)
    assert MET in line and "中央値 30.0×" in line and "30× 未満 1 件（a 29.0×）" in line
    assert MISSED in report.ratio_line("t", [("a", 29.0), ("b", 30.0), ("c", 100.0), ("d", 29.5)], loads=loads)


def test_the_load_guard_still_withholds_the_verdict_for_a_median_of_at_least_30():
    line = _line(_rows_by_ratio([26.0, 31.0, 36.0], py_load=9.0, rs_load=1.0), PAIR_LINE)
    assert "負荷条件が揃っていないため判定しない（Python 負荷 9.0 / Rust 負荷 1.0）" in line
    assert "達成" not in line and "未達" not in line

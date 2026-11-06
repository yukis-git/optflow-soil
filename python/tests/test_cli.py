import json
import sys

import numpy as np
import pytest
from PIL import Image

from optflow_ref import synth
from optflow_ref.__main__ import main
from optflow_ref.config import Config


def _save(path, img):
    Image.fromarray(np.round(img * 65535).astype(np.uint16)).save(path)


def test_pair_cli_writes_outputs(tmp_path):
    p = synth.make_pattern(160, 160, seed=2)
    _save(tmp_path / "a.png", synth.render(p, 160, 160))
    _save(tmp_path / "b.png", synth.render(p, 160, 160, synth.translation(1.25, -0.5)))
    (tmp_path / "roi.json").write_text(json.dumps({"polygon": [[20, 20], [140, 20], [140, 140], [20, 140]]}))
    out = tmp_path / "out"
    assert main(["pair", "--ref", str(tmp_path / "a.png"), "--def", str(tmp_path / "b.png"),
                 "--roi", str(tmp_path / "roi.json"), "--out", str(out), "--set", "outer_iters=4", "--png"]) == 0
    ux = np.load(out / "ux.npy")
    assert ux.dtype == np.float32 and np.isnan(ux[5, 5])
    assert abs(np.nanmedian(ux) - 1.25) < 0.01
    assert (out / "maps.png").exists() and (out / "timing.json").exists()


def test_seq_cli_on_video(tmp_path):
    import cv2

    p = synth.make_pattern(160, 160, seed=3)
    path = str(tmp_path / "s.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 160), isColor=False)
    for k in range(3):
        vw.write(np.round(synth.render(p, 160, 160, synth.translation(0.8 * k, 0.4 * k)) * 255).astype(np.uint8))
    vw.release()
    out = tmp_path / "seq"
    assert main(["seq", "--frames", path, "--out", str(out), "--set", "outer_iters=4"]) == 0
    ux = np.load(out / "frame_00002" / "ux.npy")
    assert abs(np.nanmedian(ux[40:120, 40:120]) - 1.6) < 0.05


def _pair_args(tmp_path, *extra):
    p = synth.make_pattern(96, 96, seed=2)
    _save(tmp_path / "a.png", synth.render(p, 96, 96))
    _save(tmp_path / "b.png", synth.render(p, 96, 96, synth.translation(0.5, 0.2)))
    return ["pair", "--ref", str(tmp_path / "a.png"), "--def", str(tmp_path / "b.png"),
            "--out", str(tmp_path / "out"), *extra]


def test_pair_cli_writes_valid_mask(tmp_path):
    out = tmp_path / "out"
    assert main(_pair_args(tmp_path, "--set", "outer_iters=2")) == 0
    valid, ux = np.load(out / "valid.npy"), np.load(out / "ux.npy")
    assert valid.dtype == np.uint8 and valid.any() and (valid == np.isfinite(ux)).all()


@pytest.mark.parametrize("bad,msg", [
    ("sigma_x=3", "sigma_x"),  # unknown key: named, valid keys listed
    ("outer_iters", "key=value"),  # no '='
    ("outer_iters=2.5", "outer_iters"),  # integer field, non-integer text
    ("outer_iters=2.0", "outer_iters"),
    ("sigma_w=abc", "sigma_w"),
    ("scales=0.6,x", "scales"),
    ("lam=nan", "finite"),  # parsed, then rejected by validate()
    ("conf_min=2", "conf_min")])
def test_cli_rejects_bad_set_with_exit_status_2(tmp_path, capsys, bad, msg):
    assert main(_pair_args(tmp_path, "--set", bad)) == 2
    err = capsys.readouterr().err
    assert msg in err and "Traceback" not in err
    if bad.startswith("sigma_x"):
        assert "valid keys" in err and "sigma_w" in err


def test_cli_set_parses_by_field_type():
    from optflow_ref.__main__ import _config

    cfg = _config(None, ["scales=0.8,1.2", "outer_iters=3", "max_disp=64", "lam=9.5"])
    assert cfg.scales == (0.8, 1.2) and cfg.outer_iters == 3 and cfg.max_disp == 64.0 and cfg.lam == 9.5


# sys.get_int_max_str_digits() exists from Python 3.11 on; before that the test is skipped (the lambda gives 0)
@pytest.mark.skipif(
    getattr(sys, "get_int_max_str_digits", lambda: 0)() != 4300,
    reason="the Rust port copies the default limit, 4300",
)
def test_cli_set_integer_with_more_than_4300_figures_cannot_be_read(tmp_path, capsys):
    # int() raises ValueError ("Exceeds the limit (4300 digits)") for more figures, leading zeros included
    # (underscores and the sign are not figures); the command line reports it like any unreadable integer
    ones = "1" * 4301
    for v in (ones, "-" + ones, "+" + ones, f" {ones} ", "0" * 4300 + "1", "1_" * 4300 + "1"):
        err = _run_expecting_exit_2(_pair_args(tmp_path, "--set", f"grid_step={v}"), capsys)
        assert err == f"error: --set grid_step: cannot read {v!r} as int\n"
    # 4300 figures are read
    from optflow_ref.__main__ import _parse_override

    cfg = Config()
    for v, want in (("1" * 4300, int("1" * 4300)), ("-" + "1" * 4300, -int("1" * 4300)),
                    ("1_" * 4299 + "1", int("1" * 4300)), ("0" * 4299 + "7", 7)):
        _parse_override(cfg, f"grid_step={v}")
        assert cfg.grid_step == want


def test_cli_rejects_invalid_config_file(tmp_path, capsys):
    (tmp_path / "c.json").write_text('{"grid_step": 2.5}')
    assert main(_pair_args(tmp_path, "--config", str(tmp_path / "c.json"))) == 2
    assert "grid_step" in capsys.readouterr().err


def test_seq_cli_with_no_frames_is_a_clear_error(tmp_path, capsys):
    p = synth.make_pattern(96, 96, seed=2)
    for k in range(2):
        _save(tmp_path / f"f{k}.png", synth.render(p, 96, 96, synth.translation(0.5 * k, 0)))
    assert main(["seq", "--frames", str(tmp_path), "--start", "5", "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "no frames" in err and "Traceback" not in err
    assert main(["seq", "--frames", str(tmp_path / "nothing_*.png"), "--out", str(tmp_path / "o")]) == 2
    assert "no images found" in capsys.readouterr().err
    assert main(["seq", "--frames", str(tmp_path), "--step", "0", "--out", str(tmp_path / "o")]) == 2
    assert "step" in capsys.readouterr().err


def test_seq_cli_with_one_frame_is_a_clear_error(tmp_path, capsys):
    p = synth.make_pattern(96, 96, seed=2)
    _save(tmp_path / "f0.png", synth.render(p, 96, 96))
    assert main(["seq", "--frames", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "need at least two frames to track, got 1" in err and "Traceback" not in err
    for k in range(1, 3):  # two frames, but --stop leaves one
        _save(tmp_path / f"f{k}.png", synth.render(p, 96, 96, synth.translation(0.5 * k, 0)))
    assert main(["seq", "--frames", str(tmp_path), "--stop", "1", "--out", str(tmp_path / "o")]) == 2
    assert "need at least two frames to track, got 1" in capsys.readouterr().err


@pytest.mark.parametrize("scale", ["0", "-2"])
def test_cli_rejects_scale_below_one(tmp_path, capsys, scale):
    err = _run_expecting_exit_2(_pair_args(tmp_path, "--scale", scale), capsys)
    assert f"scale must be >= 1, got {scale}" in err


def test_cli_reports_missing_files_without_a_traceback(tmp_path, capsys):
    args = _pair_args(tmp_path)
    args[2] = str(tmp_path / "missing.png")  # --ref
    err = _run_expecting_exit_2(args, capsys)
    assert err.startswith("error: ") and "missing.png" in err
    err = _run_expecting_exit_2(_pair_args(tmp_path, "--roi", str(tmp_path / "no_roi.json")), capsys)
    assert err.startswith("error: ") and "no_roi.json" in err
    err = _run_expecting_exit_2(_pair_args(tmp_path, "--config", str(tmp_path / "no_cfg.json")), capsys)
    assert err.startswith("error: ") and "no_cfg.json" in err
    (tmp_path / "bad.png").write_bytes(b"not an image")
    args = _pair_args(tmp_path)
    args[4] = str(tmp_path / "bad.png")  # --def
    assert "Traceback" not in _run_expecting_exit_2(args, capsys)
    err = _run_expecting_exit_2(["seq", "--frames", str(tmp_path / "missing.avi"), "--out", str(tmp_path / "o")], capsys)
    assert err.startswith("error: ") and "missing.avi" in err


def test_cli_reports_a_roi_file_without_a_polygon(tmp_path, capsys):
    roi = tmp_path / "roi.json"
    roi.write_text('{"points": [[0, 0], [1, 1], [0, 1]]}')
    err = _run_expecting_exit_2(_pair_args(tmp_path, "--roi", str(roi)), capsys)
    assert err.strip() == f"error: ROI file {roi} has no 'polygon' list"


def _run_expecting_exit_2(args, capsys):
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err

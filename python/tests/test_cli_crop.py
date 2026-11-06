"""Automatic ROI crop of the command line (`--crop {auto,off}`, `--crop-margin N`)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from optflow_ref import synth
from optflow_ref.__main__ import main
from optflow_ref.config import Config
from optflow_ref.crop import WARNING, Crop, border_fraction, default_margin, plan
from optflow_ref.io import polygon_mask

N = 768
ROI = [[84, 236], [684, 236], [684, 531], [84, 531]]  # a strip of about 30 % of the image, in the middle
# margin 189 = ceil(32) + ceil(4 * 4) + ceil(12.2) + 32 + (64 / 2 + 16) * 2**1; A = 16 * 2**1 = 32 rounds
# y0 = 236 - 189 = 47 down to 32; x0 and the right side are clipped, y1 = 532 + 189 = 721
EXPECTED_CROP_JSON = '{\n  "x0": 0,\n  "y0": 32,\n  "x1": 768,\n  "y1": 721,\n  "margin": 189,\n  "mode": "auto"\n}'


def _save(path, img):
    Image.fromarray(np.round(img * 65535).astype(np.uint16)).save(path)


def _rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---- the rectangle

def test_default_margin_is_displacement_plus_window_plus_guard_plus_padding_plus_coarse_footprint():
    cfg = Config()
    # L = 1 level: (64 / 2 + 16) * 2 = 96 for the coarse stage
    assert default_margin(cfg, (768, 768)) == 32 + 16 + 13 + 32 + 96
    assert default_margin(cfg, (768, 768), seq=True) == 2 * 189
    # L = 3 levels: 48 * 8 = 384
    cfg = Config(max_disp=100.5, sigma_w=6.0, edge_guard=5.0)
    assert default_margin(cfg, (768, 768)) == 101 + 24 + 5 + 32 + 384
    # a small image gets fewer levels
    assert default_margin(Config(), (40, 40)) == 32 + 16 + 13 + 32 + 48


def test_rect_is_the_roi_bounding_box_plus_margin_aligned_down_to_the_coarse_grid():
    mask = np.zeros((N, N), bool)
    mask[300:400, 300:400] = True
    c = plan(mask, Config())
    assert (c.x0, c.y0, c.x1, c.y1, c.margin) == (96, 96, 589, 589, 189)  # L = 1, A = 32: 111 -> 96
    # max_disp = 128: L = 3 levels, A = 16 * 8 = 128, margin 128 + 16 + 13 + 32 + 48 * 8
    mask = np.zeros((2048, 2048), bool)
    mask[1000:1080, 1000:1080] = True
    c = plan(mask, Config(max_disp=128.0))
    assert c.margin == 573
    assert (c.x0, c.y0) == (384, 384)  # 1000 - 573 = 427 rounds down to 384
    assert (c.x1, c.y1) == (1080 + 573, 1080 + 573)  # only the start is aligned
    assert c.json() == json.dumps({"x0": 384, "y0": 384, "x1": 1653, "y1": 1653, "margin": 573, "mode": "auto"},
                                  indent=2)


def test_rect_is_clipped_to_the_image_and_a_given_margin_replaces_the_default():
    mask = np.zeros((300, 400), bool)
    mask[0:80, 350:400] = True
    c = plan(mask, Config())
    assert (c.x0, c.y0, c.x1, c.y1) == (160, 0, 400, 269)  # 350 - 189 = 161 -> 160
    c = plan(mask, Config(), margin=10)
    assert (c.x0, c.y0, c.x1, c.y1, c.margin) == (320, 0, 400, 90, 10)  # 340 -> 320
    assert plan(mask, Config(), seq=True).margin == 378


def test_plan_rejects_an_empty_roi_a_negative_margin_and_a_crop_below_poc_win():
    with pytest.raises(ValueError, match="ROI mask is empty"):
        plan(np.zeros((300, 400), bool), Config())
    mask = np.ones((300, 400), bool)
    with pytest.raises(ValueError, match=r"--crop-margin must be >= 0, got -1"):
        plan(mask, Config(), margin=-1)
    mask = np.zeros((300, 400), bool)
    mask[100:110, 100:110] = True
    with pytest.raises(ValueError, match=r"smaller than poc_win \(64\).*--crop-margin.*--crop off"):
        plan(mask, Config(), margin=0)
    with pytest.raises(ValueError, match=r"image smaller than poc_win \(64\): \(32, 40\)"):
        plan(np.ones((32, 40), bool), Config())


# ---- the border warning

def _warn_fraction(ux, mask, crop=Crop(100, 100, 200, 200, 20), shape=(400, 400), cfg=Config()):
    return border_fraction(ux, np.zeros_like(ux), mask, crop, shape, cfg)


def test_border_fraction_counts_roi_pixels_whose_match_is_within_4_sigma_w_of_the_crop_border():
    inner = np.zeros((100, 100), bool)
    inner[30:70, 30:70] = True
    still = np.zeros((100, 100))
    assert _warn_fraction(still, inner) == 0.0
    assert _warn_fraction(still + 25.0, inner) > 0.2  # x + 25 > 99 - 16 for x >= 59
    assert _warn_fraction(still + 5.0, inner) == 0.0
    assert _warn_fraction(np.full((100, 100), np.nan), inner) == 1.0  # no match at all counts as outside
    everything = np.ones((100, 100), bool)
    assert _warn_fraction(still, everything) == pytest.approx(1 - 68**2 / 100**2)
    # the guard off: only a match outside the crop counts
    assert _warn_fraction(still, everything, cfg=Config(edge_guard=0.0)) == 0.0
    assert _warn_fraction(still + 3.0, everything, cfg=Config(edge_guard=0.0)) == pytest.approx(3 / 100)


def test_border_fraction_ignores_crop_sides_that_are_image_borders():
    everything = np.ones((100, 100), bool)
    still = np.zeros((100, 100))
    # the crop is the whole image: nothing the crop could have changed
    assert _warn_fraction(still, everything, Crop(0, 0, 100, 100, 20), (100, 100)) == 0.0
    # only the right side is a crop border (the image is 100 high and 150 wide)
    f = _warn_fraction(still, everything, Crop(0, 0, 100, 100, 20), (100, 150))
    assert f == pytest.approx(16 / 100)  # x > 83 is within 16 of the right border
    # the left side is an image border, the others are not: x < 16 is not counted
    f = _warn_fraction(still, everything, Crop(0, 100, 100, 200, 20), (400, 400))
    assert f == pytest.approx(1 - 84 * 68 / 100**2)


# ---- pair

@pytest.fixture(scope="module")
def pair_runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("crop_pair")
    p = synth.make_pattern(N, N, seed=11)
    _save(d / "a.png", synth.render(p, N, N))
    _save(d / "b.png", synth.render(p, N, N, synth.uniform_strain(0.002, -0.001, (N / 2, N / 2), t=(2.0, -1.5))))
    (d / "roi.json").write_text(json.dumps({"polygon": ROI}))
    base = ["pair", "--ref", str(d / "a.png"), "--def", str(d / "b.png"), "--roi", str(d / "roi.json"),
            "--set", "outer_iters=4"]
    assert main([*base, "--out", str(d / "auto")]) == 0
    assert main([*base, "--crop", "off", "--out", str(d / "off")]) == 0
    return d / "auto", d / "off"


def test_pair_crop_auto_matches_crop_off_inside_the_roi(pair_runs):
    auto, off = pair_runs
    roi = polygon_mask((N, N), ROI)
    assert 0.25 < roi.mean() < 0.35
    for name in ("ux", "uy", "conf", "Exx", "Eyy", "Exy", "gmax", "vol", "valid"):
        assert np.load(auto / f"{name}.npy").shape == np.load(off / f"{name}.npy").shape == (N, N), name
    ua, ub = np.load(auto / "ux.npy"), np.load(off / "ux.npy")
    va, vb = np.load(auto / "uy.npy"), np.load(off / "uy.npy")
    both = roi & np.isfinite(ua) & np.isfinite(ub)
    assert both.sum() > 0.5 * roi.sum()
    assert _rms(ua[both], ub[both]) <= 1e-3 and _rms(va[both], vb[both]) <= 1e-3
    assert (np.isnan(ua) == np.isnan(ub))[roi].mean() >= 0.999
    assert abs(np.nanmedian(ua[roi]) - 2.0) < 0.1 and abs(np.nanmedian(va[roi]) + 1.5) < 0.1


def test_pair_crop_auto_writes_full_size_outputs_with_nan_outside_the_crop(pair_runs):
    auto, _ = pair_runs
    outside = np.ones((N, N), bool)
    outside[32:721, :] = False
    for name in ("ux", "uy", "Exx", "Eyy", "Exy", "gmax", "vol"):
        a = np.load(auto / f"{name}.npy")
        assert a.dtype == np.float32 and np.isnan(a[outside]).all(), name
    assert (np.load(auto / "conf.npy")[outside] == 0).all()
    valid = np.load(auto / "valid.npy")
    assert valid.dtype == np.uint8 and (valid[outside] == 0).all() and valid.any()
    assert (valid == np.isfinite(np.load(auto / "ux.npy"))).all()


def test_pair_crop_json(pair_runs):
    auto, off = pair_runs
    assert (auto / "crop.json").read_text() == EXPECTED_CROP_JSON
    assert (off / "crop.json").read_text() == '{\n  "mode": "off"\n}'


def test_crop_json_says_off_without_a_roi(tmp_path):
    p = synth.make_pattern(96, 96, seed=2)
    _save(tmp_path / "a.png", synth.render(p, 96, 96))
    _save(tmp_path / "b.png", synth.render(p, 96, 96, synth.translation(0.5, 0.2)))
    out = tmp_path / "out"
    assert main(["pair", "--ref", str(tmp_path / "a.png"), "--def", str(tmp_path / "b.png"), "--out", str(out),
                 "--set", "outer_iters=2"]) == 0
    assert json.loads((out / "crop.json").read_text()) == {"mode": "off"}


@pytest.mark.parametrize("args,msg", [
    (["--crop", "maybe"], "argument --crop: invalid choice: 'maybe' (choose from "),  # the list's quoting varies by version
    (["--crop-margin", "x"], "argument --crop-margin: invalid int value: 'x'"),
])
def test_cli_rejects_bad_crop_options(tmp_path, capsys, args, msg):
    with pytest.raises(SystemExit) as e:
        main(["pair", "--ref", "a", "--def", "b", "--out", str(tmp_path), *args])
    assert e.value.code == 2
    assert msg in capsys.readouterr().err


def test_cli_rejects_a_negative_crop_margin(tmp_path, capsys):
    p = synth.make_pattern(96, 96, seed=2)
    _save(tmp_path / "a.png", synth.render(p, 96, 96))
    (tmp_path / "roi.json").write_text(json.dumps({"polygon": [[10, 10], [80, 10], [80, 80]]}))
    assert main(["pair", "--ref", str(tmp_path / "a.png"), "--def", str(tmp_path / "a.png"), "--roi",
                 str(tmp_path / "roi.json"), "--crop-margin", "-3", "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert err.strip() == "error: --crop-margin must be >= 0, got -3"


GUARD_ERROR = ("error: grid_step <= 4096 and an edge guard (edge_guard, or the automatic one from lam, scales and "
               "bandwidth) <= 256 px required\n")


def _absurd_grid_step_runs(d):
    """pair and seq command lines with a ROI and the grid_step values the validate rule rejects."""
    p = synth.make_pattern(128, 128, seed=2)
    for k in range(3):
        _save(d / f"f{k}.png", synth.render(p, 128, 128, synth.translation(0.5 * k, 0.2 * k)))
    (d / "roi.json").write_text(json.dumps({"polygon": [[20, 20], [100, 20], [100, 100], [20, 100]]}))
    for grid_step in ("9223372036854775808", "18446744073709551615", "4611686018427387904", "4097"):
        for src in (["pair", "--ref", str(d / "f0.png"), "--def", str(d / "f1.png")], ["seq", "--frames", str(d)]):
            yield [*src, "--roi", str(d / "roi.json"), "--out", str(d / "out"), "--set", f"grid_step={grid_step}"]


def test_cli_rejects_an_absurd_grid_step_with_a_roi_as_a_config_error(tmp_path, capsys):
    # the Rust crop alignment grid_step * 2**L wrapped to 0 for 2**63 and divided by zero
    for args in _absurd_grid_step_runs(tmp_path):
        assert main(args) == 2, args
        assert capsys.readouterr().err == GUARD_ERROR, args


# ---- seq

def _write_frames(d, n, step, w=384, h=288, seed=5):
    p = synth.make_pattern(h, w, seed=seed)
    for k in range(n):
        _save(d / f"f{k}.png", synth.render(p, h, w, synth.translation(step * k, 0.0)))


def _seq_args(d, *extra):
    (d / "roi.json").write_text(json.dumps({"polygon": [[130, 90], [250, 90], [250, 200], [130, 200]]}))
    return ["seq", "--frames", str(d), "--roi", str(d / "roi.json"), "--out", str(d / "out"),
            "--set", "outer_iters=2", "--set", "max_disp=64", *extra]


def test_seq_writes_full_size_frames_and_one_crop_json(tmp_path, capsys):
    _write_frames(tmp_path, 3, 1.5)
    assert main(_seq_args(tmp_path)) == 0
    out = tmp_path / "out"
    c = json.loads((out / "crop.json").read_text())
    # margin 2 * (64 + 16 + 13 + 32 + (32 + 16) * 4) = 634 covers the whole 384 x 288 image; A = 16 * 4 = 64
    assert c == {"x0": 0, "y0": 0, "x1": 384, "y1": 288, "margin": 634, "mode": "auto"}
    for k in (1, 2):
        ux = np.load(out / f"frame_{k:05d}" / "ux.npy")
        assert ux.shape == (288, 384) and np.isnan(ux[5, 5])
        assert abs(np.nanmedian(ux[90:200, 130:250]) - 1.5 * k) < 0.05
    assert "crop border" not in capsys.readouterr().err


def test_seq_crop_saves_cropped_frames_and_matches_crop_off(tmp_path):
    _write_frames(tmp_path, 3, 1.5, w=512, h=384)
    roi = {"polygon": [[200, 150], [300, 150], [300, 240], [200, 240]]}
    (tmp_path / "roi.json").write_text(json.dumps(roi))
    base = ["seq", "--frames", str(tmp_path), "--roi", str(tmp_path / "roi.json"), "--set", "outer_iters=2"]
    assert main([*base, "--out", str(tmp_path / "auto"), "--crop-margin", "100"]) == 0
    assert main([*base, "--out", str(tmp_path / "off"), "--crop", "off"]) == 0
    c = json.loads((tmp_path / "auto" / "crop.json").read_text())
    assert c == {"x0": 96, "y0": 32, "x1": 401, "y1": 341, "margin": 100, "mode": "auto"}
    roi_mask = polygon_mask((384, 512), roi["polygon"])
    for k in (1, 2):
        a = np.load(tmp_path / "auto" / f"frame_{k:05d}" / "ux.npy")
        b = np.load(tmp_path / "off" / f"frame_{k:05d}" / "ux.npy")
        assert a.shape == b.shape == (384, 512)
        both = roi_mask & np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 0.5 * roi_mask.sum() and _rms(a[both], b[both]) <= 1e-3


def test_seq_warns_once_when_the_material_moves_farther_than_the_margin(tmp_path, capsys):
    _write_frames(tmp_path, 4, 20.0)
    assert main(_seq_args(tmp_path, "--crop-margin", "20")) == 0
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1, err
    assert WARNING == ("warning: the material moves close to the crop border — "
                       "increase --crop-margin or use --crop off")


def test_pair_warns_when_the_material_moves_farther_than_the_margin(tmp_path, capsys):
    _write_frames(tmp_path, 2, 30.0)
    roi = {"polygon": [[130, 90], [250, 90], [250, 200], [130, 200]]}
    (tmp_path / "roi.json").write_text(json.dumps(roi))
    args = ["pair", "--ref", str(tmp_path / "f0.png"), "--def", str(tmp_path / "f1.png"), "--roi",
            str(tmp_path / "roi.json"), "--set", "outer_iters=2", "--set", "max_disp=64",
            "--out", str(tmp_path / "out")]
    assert main([*args, "--crop-margin", "20"]) == 0
    assert capsys.readouterr().err.count(WARNING) == 1
    assert main([*args, "--crop", "off"]) == 0
    assert WARNING not in capsys.readouterr().err
    assert main(args) == 0  # the default margin holds the motion
    assert WARNING not in capsys.readouterr().err


# ---- the Rust command line

RUST_BIN = os.environ.get("OPTFLOW_BIN") or os.path.join(
    os.path.dirname(__file__), "..", "..", "rust", "target", "release", "optflow")


@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN),
                    reason="the Rust optflow binary is not built (cargo build --release -p optflow-cli)")
def test_python_and_rust_command_lines_agree_on_crop_json_and_output_shapes(tmp_path):
    d = tmp_path
    _write_frames(d, 3, 1.5, w=320, h=256)
    (d / "roi.json").write_text(json.dumps({"polygon": [[130, 100], [200, 100], [200, 160], [130, 160]]}))
    for extra in ([], ["--crop-margin", "100"], ["--crop", "off"]):
        for cmd, src in (("pair", ["--ref", str(d / "f0.png"), "--def", str(d / "f1.png")]),
                         ("seq", ["--frames", str(d)])):
            args = [cmd, *src, "--roi", str(d / "roi.json"), "--set", "outer_iters=2", *extra]
            py, rs = d / "py", d / "rs"
            shutil.rmtree(py, ignore_errors=True)
            shutil.rmtree(rs, ignore_errors=True)
            assert main([*args, "--out", str(py)]) == 0
            done = subprocess.run([RUST_BIN, *args, "--out", str(rs)], capture_output=True, text=True)
            assert done.returncode == 0, done.stderr
            assert (py / "crop.json").read_text() == (rs / "crop.json").read_text(), (cmd, extra)
            sub = "" if cmd == "pair" else "frame_00002"
            for name in ("ux", "conf", "Exx", "valid"):
                a, b = np.load(py / sub / f"{name}.npy"), np.load(rs / sub / f"{name}.npy")
                assert a.shape == b.shape == (256, 320) and a.dtype == b.dtype, (cmd, extra, name)


@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN),
                    reason="the Rust optflow binary is not built (cargo build --release -p optflow-cli)")
def test_python_and_rust_command_lines_agree_on_an_absurd_grid_step(tmp_path, capsys):
    for args in _absurd_grid_step_runs(tmp_path):
        assert main(args) == 2, args
        py_err = capsys.readouterr().err
        done = subprocess.run([RUST_BIN, *args], capture_output=True, text=True)
        assert done.returncode == 2, (args, done.stderr)
        assert done.stderr == py_err == GUARD_ERROR, args

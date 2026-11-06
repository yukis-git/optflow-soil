"""`--mask PATH` (an ROI as a .npy or .png file) and `--raw` (also write raw_ux.npy, raw_uy.npy)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from optflow_ref import synth
from optflow_ref.__main__ import main
from optflow_ref.io import polygon_mask

POLY = [[24, 24], [104, 24], [104, 104], [24, 104]]
ARRAYS = ("ux", "uy", "conf", "Exx", "Eyy", "Exy", "gmax", "vol", "valid")


def _save(path, img):
    Image.fromarray(np.round(img * 65535).astype(np.uint16)).save(path)


@pytest.fixture
def scene(tmp_path):
    """128 x 128 images f0.png, f1.png and the ROI polygon `roi.json`."""
    p = synth.make_pattern(128, 128, seed=3)
    for k in range(3):
        _save(tmp_path / f"f{k}.png", synth.render(p, 128, 128, synth.translation(0.7 * k, -0.4 * k)))
    (tmp_path / "roi.json").write_text(json.dumps({"polygon": POLY}))
    return tmp_path


def _pair(d, *extra, out="out"):
    return ["pair", "--ref", str(d / "f0.png"), "--def", str(d / "f1.png"), "--set", "outer_iters=2",
            *extra, "--out", str(d / out)]


def _same_outputs(a, b, names=ARRAYS):
    for name in names:
        x, y = np.load(a / f"{name}.npy"), np.load(b / f"{name}.npy")
        assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True), name


# ---- --mask

@pytest.mark.parametrize("kind", ["npy-bool", "npy-uint8", "png-L", "png-1bit"])
def test_mask_file_gives_the_outputs_of_the_same_roi(scene, kind):
    m = polygon_mask((128, 128), POLY)
    if kind == "npy-bool":
        np.save(scene / "m.npy", m)
    elif kind == "npy-uint8":
        np.save(scene / "m.npy", m.astype(np.uint8) * 7)  # nonzero = inside
    elif kind == "png-L":
        Image.fromarray(m.astype(np.uint8) * 255).save(scene / "m.png")
    else:
        Image.fromarray(m).save(scene / "m.png")
    path = scene / ("m.npy" if kind.startswith("npy") else "m.png")
    assert main(_pair(scene, "--roi", str(scene / "roi.json"), out="roi")) == 0
    assert main(_pair(scene, "--mask", str(path), out="mask")) == 0
    _same_outputs(scene / "roi", scene / "mask")
    assert (scene / "roi" / "crop.json").read_text() == (scene / "mask" / "crop.json").read_text()


def test_a_two_block_mask_is_cropped_to_the_joint_bounding_box(tmp_path):
    p = synth.make_pattern(256, 256, seed=4)
    _save(tmp_path / "f0.png", synth.render(p, 256, 256))
    _save(tmp_path / "f1.png", synth.render(p, 256, 256, synth.translation(0.9, 0.3)))
    m = np.zeros((256, 256), bool)
    m[90:150, 80:140] = True  # x 80..139, y 90..149
    m[160:220, 150:210] = True  # x 150..209, y 160..219
    np.save(tmp_path / "m.npy", m)
    assert main(_pair(tmp_path, "--mask", str(tmp_path / "m.npy"), "--crop-margin", "40")) == 0
    # x0 = 80 - 40 = 40 -> 32 (A = 32), x1 = 210 + 40; y0 = 90 - 40 = 50 -> 32, y1 = 220 + 40 is clipped to 256
    assert json.loads((tmp_path / "out" / "crop.json").read_text()) == {
        "x0": 32, "y0": 32, "x1": 250, "y1": 256, "margin": 40, "mode": "auto"}
    ux = np.load(tmp_path / "out" / "ux.npy")
    assert ux.shape == (256, 256) and not np.isfinite(ux[~m]).any()
    assert np.isfinite(ux[m]).sum() > 0.2 * m.sum()
    assert np.isfinite(ux[90:150, 80:140]).any() and np.isfinite(ux[160:220, 150:210]).any()


def _error(args, capsys):
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err.strip()


def test_mask_errors(scene, capsys):
    m = polygon_mask((128, 128), POLY)
    np.save(scene / "m.npy", m)
    pair = lambda *extra: _pair(scene, *extra)
    assert _error(pair("--roi", str(scene / "roi.json"), "--mask", str(scene / "m.npy")), capsys) == \
        "error: --roi and --mask are mutually exclusive"
    np.save(scene / "small.npy", m[:64, :64])
    assert _error(pair("--mask", str(scene / "small.npy")), capsys) == \
        f"error: mask file {scene / 'small.npy'} has shape (64, 64), expected (128, 128)"
    np.save(scene / "line.npy", m[0])
    assert _error(pair("--mask", str(scene / "line.npy")), capsys) == \
        f"error: mask file {scene / 'line.npy'} has shape (128,), expected (128, 128)"
    np.save(scene / "f.npy", m.astype(np.float32))
    assert _error(pair("--mask", str(scene / "f.npy")), capsys) == \
        f"error: mask file {scene / 'f.npy'} must hold bool or uint8, got float32"
    (scene / "m.txt").write_text("1")
    assert _error(pair("--mask", str(scene / "m.txt")), capsys) == \
        f"error: mask file {scene / 'm.txt'} must be a .npy or .png file"
    np.save(scene / "empty.npy", np.zeros((128, 128), bool))
    assert _error(pair("--mask", str(scene / "empty.npy")), capsys) == "error: ROI mask is empty"
    assert _error(pair("--mask", str(scene / "empty.npy"), "--crop", "off"), capsys) == "error: ROI mask is empty"
    assert str(scene / "none.npy") in _error(pair("--mask", str(scene / "none.npy")), capsys)
    (scene / "bad.png").write_bytes(b"not a png")
    assert str(scene / "bad.png") in _error(pair("--mask", str(scene / "bad.png")), capsys)
    Image.fromarray(np.ones((64, 96), np.uint8)).save(scene / "small.png")
    assert _error(pair("--mask", str(scene / "small.png")), capsys) == \
        f"error: mask file {scene / 'small.png'} has shape (64, 96), expected (128, 128)"


def test_mask_applies_to_the_scaled_image_and_to_sequences(scene):
    m = np.zeros((64, 64), bool)
    m[12:52, 12:52] = True
    np.save(scene / "m.npy", m)
    assert main(_pair(scene, "--scale", "2", "--mask", str(scene / "m.npy"), "--crop", "off")) == 0
    ux = np.load(scene / "out" / "ux.npy")
    assert ux.shape == (64, 64) and not np.isfinite(ux[~m]).any() and np.isfinite(ux[m]).any()
    assert main(["seq", "--frames", str(scene), "--mask", str(scene / "m.npy"), "--scale", "2",
                 "--set", "outer_iters=2", "--out", str(scene / "seq")]) == 0
    assert np.load(scene / "seq" / "frame_00002" / "ux.npy").shape == (64, 64)


# ---- --raw

@pytest.mark.parametrize("crop", ["off", "auto"])
def test_raw_writes_the_unthresholded_field_of_a_pair(scene, crop):
    assert main(_pair(scene, "--roi", str(scene / "roi.json"), "--crop", crop, "--raw")) == 0
    out = scene / "out"
    ux, uy = np.load(out / "ux.npy"), np.load(out / "uy.npy")
    rx, ry = np.load(out / "raw_ux.npy"), np.load(out / "raw_uy.npy")
    assert rx.dtype == ry.dtype == np.dtype("<f4") and rx.shape == ry.shape == (128, 128)
    ok = np.isfinite(ux)
    assert ok.any() and (rx[ok] == ux[ok]).all() and (ry[ok] == uy[ok]).all()  # valid pixels: the same numbers
    assert np.isfinite(rx).sum() > ok.sum()  # the exterior of the ROI and the guard bands are kept
    assert abs(np.nanmedian(rx[ok]) - 0.7) < 0.05 and abs(np.nanmedian(ry[ok]) + 0.4) < 0.05


def test_raw_is_off_by_default(scene):
    assert main(_pair(scene, "--roi", str(scene / "roi.json"))) == 0
    assert not (scene / "out" / "raw_ux.npy").exists() and not (scene / "out" / "raw_uy.npy").exists()


def test_raw_is_written_per_frame_of_a_sequence(scene):
    args = ["seq", "--frames", str(scene), "--roi", str(scene / "roi.json"), "--set", "outer_iters=2",
            "--raw", "--out", str(scene / "seq")]
    assert main(args) == 0
    for k in (1, 2):
        d = scene / "seq" / f"frame_{k:05d}"
        rx = np.load(d / "raw_ux.npy")
        ux = np.load(d / "ux.npy")
        assert rx.dtype == np.dtype("<f4") and rx.shape == (128, 128) and (rx[np.isfinite(ux)] == ux[np.isfinite(ux)]).all()
        assert abs(np.nanmedian(rx[np.isfinite(ux)]) - 0.7 * k) < 0.05
    assert not (scene / "seq" / "raw_ux.npy").exists()


# ---- the Rust command line

RUST_BIN = os.environ.get("OPTFLOW_BIN") or os.path.join(
    os.path.dirname(__file__), "..", "..", "rust", "target", "release", "optflow")
needs_rust = pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN),
                                reason="the Rust optflow binary is not built (cargo build --release -p optflow-cli)")


@needs_rust
def test_python_and_rust_command_lines_give_the_same_mask_errors(scene, capsys):
    m = polygon_mask((128, 128), POLY)
    np.save(scene / "m.npy", m)
    np.save(scene / "small.npy", m[:64, :64])
    np.save(scene / "line.npy", m[0])
    np.save(scene / "f.npy", m.astype(np.float32))
    np.save(scene / "i.npy", m.astype(np.int64))
    np.save(scene / "empty.npy", np.zeros((128, 128), bool))
    (scene / "m.txt").write_text("1")
    Image.fromarray(np.ones((64, 96), np.uint8)).save(scene / "small.png")
    (scene / "bad.png").write_bytes(b"not a png")
    cases = [["--roi", str(scene / "roi.json"), "--mask", str(scene / "m.npy")]]
    cases += [["--mask", str(scene / n)] for n in ("small.npy", "line.npy", "f.npy", "i.npy", "m.txt", "empty.npy",
                                                   "small.png")]
    for extra in cases:
        args = _pair(scene, *extra)
        assert main(args) == 2
        py = capsys.readouterr().err
        rust = subprocess.run([RUST_BIN, *args], capture_output=True, text=True)
        assert rust.returncode == 2 and rust.stderr.strip() == py.strip(), extra
    # a missing and an undecodable file: an error line naming the file, from both
    for name in ("none.npy", "bad.png"):
        args = _pair(scene, "--mask", str(scene / name))
        rust = subprocess.run([RUST_BIN, *args], capture_output=True, text=True)
        assert rust.returncode == 2 and str(scene / name) in rust.stderr and rust.stderr.startswith("error: ")


@needs_rust
def test_python_and_rust_command_lines_agree_on_mask_and_raw_outputs(scene):
    m = np.zeros((128, 128), bool)
    m[30:70, 30:60] = True
    m[80:110, 70:100] = True
    np.save(scene / "m.npy", m)
    for extra in (["--mask", str(scene / "m.npy"), "--raw"],
                  ["--mask", str(scene / "m.npy"), "--raw", "--crop", "off"],
                  ["--roi", str(scene / "roi.json"), "--raw", "--crop-margin", "20"]):
        for cmd, src in (("pair", ["--ref", str(scene / "f0.png"), "--def", str(scene / "f1.png")]),
                         ("seq", ["--frames", str(scene)])):
            args = [cmd, *src, "--set", "outer_iters=2", *extra]
            py, rs = scene / "py", scene / "rs"
            shutil.rmtree(py, ignore_errors=True)
            shutil.rmtree(rs, ignore_errors=True)
            assert main([*args, "--out", str(py)]) == 0
            done = subprocess.run([RUST_BIN, *args, "--out", str(rs)], capture_output=True, text=True)
            assert done.returncode == 0, done.stderr
            assert (py / "crop.json").read_text() == (rs / "crop.json").read_text(), (cmd, extra)
            sub = "" if cmd == "pair" else "frame_00002"
            for name in ("ux", "valid", "raw_ux", "raw_uy"):
                a, b = np.load(py / sub / f"{name}.npy"), np.load(rs / sub / f"{name}.npy")
                assert a.shape == b.shape == (128, 128) and a.dtype == b.dtype, (cmd, extra, name)
            # the raw fields agree to the accuracy of the two implementations where both are finite
            a, b = np.load(py / sub / "raw_ux.npy"), np.load(rs / sub / "raw_ux.npy")
            assert (np.isnan(a) == np.isnan(b)).mean() > 0.99
            both = np.isfinite(a) & np.isfinite(b)
            assert both.any() and np.abs(a[both] - b[both]).max() < 0.05, (cmd, extra)

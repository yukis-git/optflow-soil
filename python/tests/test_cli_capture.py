"""The command line warns when the coarse stage is designed to capture less than max_disp."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from optflow_ref import synth
from optflow_ref.__main__ import main
from optflow_ref.poc import capture_range

WARNING_128_512 = ("warning: the coarse stage covers about ±64 px in x and ±64 px in y, less than max_disp 128; "
                   "larger motions may be lost (use a larger image or crop margin, or a smaller max_disp)")


def _save(path, img):
    Image.fromarray(np.round(img * 65535).astype(np.uint16)).save(path)


def _write_frames(d, n, size, step=(1.5, -0.5), seed=2):
    """f0.png ... f{n-1}.png: a pattern of size x size moving by `step` px per frame."""
    p = synth.make_pattern(size, size, seed=seed)
    for k in range(n):
        _save(d / f"f{k}.png", synth.render(p, size, size, synth.translation(step[0] * k, step[1] * k)))


@pytest.fixture(scope="module")
def pair512(tmp_path_factory):
    d = tmp_path_factory.mktemp("pair512")
    _write_frames(d, 2, 512)
    return d


def _pair(d, out, *extra):
    return ["pair", "--ref", str(d / "f0.png"), "--def", str(d / "f1.png"), "--out", str(out),
            "--set", "outer_iters=1", *extra]


def test_pair_warns_once_when_max_disp_is_beyond_the_coarse_capture_range(pair512, tmp_path, capsys):
    assert capture_range(128.0, 64, (512, 512)) == (64.0, 64.0)
    assert main(_pair(pair512, tmp_path / "out", "--set", "max_disp=128")) == 0
    err = capsys.readouterr().err
    assert err.splitlines().count(WARNING_128_512) == 1 and err.count("warning") == 1, err


def test_pair_with_the_defaults_does_not_warn_on_the_same_pair(pair512, tmp_path, capsys):
    # the defaults sit exactly at the capture range of a 512 x 512 image (L = 1, a 64 px window: +-32 px)
    assert capture_range(32.0, 64, (512, 512)) == (32.0, 32.0)
    assert main(_pair(pair512, tmp_path / "out")) == 0
    assert "warning" not in capsys.readouterr().err


def test_the_warning_is_about_the_image_actually_tracked_after_scale_and_crop(tmp_path, capsys):
    _write_frames(tmp_path, 2, 200)
    (tmp_path / "roi.json").write_text(json.dumps({"polygon": [[60, 60], [140, 60], [140, 140], [60, 140]]}))
    args = _pair(tmp_path, tmp_path / "out", "--set", "max_disp=64")
    msg = "warning: the coarse stage covers about ±{c:g} px in x and ±{c:g} px in y, less than max_disp 64;"

    assert main(args) == 0  # the whole 200 x 200 image: L = 2, a 32 px window
    assert msg.format(c=32) in capsys.readouterr().err
    assert main([*args, "--scale", "2"]) == 0  # 100 x 100: a 16 px window
    assert msg.format(c=16) in capsys.readouterr().err
    # --roi crops to 151 x 151 (margin 10): a 16 px window; --crop off tracks all 200 x 200
    assert main([*args, "--roi", str(tmp_path / "roi.json"), "--crop-margin", "10"]) == 0
    assert msg.format(c=16) in capsys.readouterr().err
    assert main([*args, "--roi", str(tmp_path / "roi.json"), "--crop", "off"]) == 0
    assert msg.format(c=32) in capsys.readouterr().err


def test_seq_warns_once_for_the_whole_run(tmp_path, capsys):
    _write_frames(tmp_path, 4, 96)
    out = tmp_path / "out"
    args = ["seq", "--frames", str(tmp_path), "--out", str(out), "--set", "outer_iters=1", "--set", "max_disp=64"]
    assert main(args) == 0
    err = capsys.readouterr().err
    assert err.count("warning") == 1 and err.count("covers about ±16 px in x and ±16 px in y, less than max_disp 64") == 1
    assert (out / "frame_00003" / "ux.npy").exists()  # three frames tracked, one warning
    assert main(["seq", "--frames", str(tmp_path), "--out", str(out), "--set", "outer_iters=1", "--set", "max_disp=8"]) == 0
    assert "warning" not in capsys.readouterr().err  # 96 x 96 with max_disp 8: L = 0, a 64 px window = +-16 px


def test_seq_on_a_video_warns_once(tmp_path, capsys):
    import cv2

    p = synth.make_pattern(96, 96, seed=3)
    path = str(tmp_path / "s.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96), isColor=False)
    for k in range(3):  # three frames, two results: the warning comes with the first only
        vw.write(np.round(synth.render(p, 96, 96, synth.translation(0.8 * k, 0.4 * k)) * 255).astype(np.uint8))
    vw.release()
    out = tmp_path / "out"
    assert main(["seq", "--frames", path, "--out", str(out), "--set", "outer_iters=1", "--set", "max_disp=64"]) == 0
    err = capsys.readouterr().err
    assert err.count("warning") == 1 and err.count("covers about ±16 px in x and ±16 px in y, less than max_disp 64;") == 1, err
    assert (out / "frame_00002" / "ux.npy").exists()


def test_seq_with_an_roi_crop_warns_once_about_the_crop(tmp_path, capsys):
    # 200 x 200 frames, four of them, so three results; the ROI crops them to 151 x 151 (margin 10): a 16 px
    # window, where the whole frame (L = 2) would have a 32 px one. The crop may add its own warning (the
    # material nears its border): only the capture warning is counted.
    _write_frames(tmp_path, 4, 200)
    (tmp_path / "roi.json").write_text(json.dumps({"polygon": [[60, 60], [140, 60], [140, 140], [60, 140]]}))
    out = tmp_path / "out"
    assert main(["seq", "--frames", str(tmp_path), "--out", str(out), "--set", "outer_iters=1", "--set", "max_disp=64",
                 "--roi", str(tmp_path / "roi.json"), "--crop-margin", "10"]) == 0
    err = capsys.readouterr().err
    coarse = [line for line in err.splitlines() if line.startswith("warning: the coarse stage")]
    assert len(coarse) == 1 and "covers about ±16 px in x and ±16 px in y, less than max_disp 64;" in coarse[0], err
    assert (out / "frame_00003" / "ux.npy").exists()


def test_a_seq_that_fails_at_frame_2_prints_the_warning_then_the_error(tmp_path, capsys):
    # frame 1 is tracked before the unreadable frame 2 is reached: the warning comes with that result and the
    # error line closes the run, in this order and nothing else
    _write_frames(tmp_path, 2, 96)
    (tmp_path / "f2.png").write_bytes(b"not an image")
    out = tmp_path / "out"
    assert main(["seq", "--frames", str(tmp_path), "--out", str(out), "--set", "outer_iters=1", "--set", "max_disp=64"]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2, lines
    assert lines[0].startswith("warning: the coarse stage covers about ±16 px in x and ±16 px in y, less than max_disp 64;")
    assert lines[1].startswith("error: ") and "f2.png" in lines[1]
    assert (out / "frame_00001" / "ux.npy").exists()


def test_a_run_that_fails_prints_its_error_only(tmp_path, capsys):
    # the warning comes with the first result: a pair of different sizes (L = 2 here would warn) only fails
    _write_frames(tmp_path, 1, 96)
    _save(tmp_path / "f1.png", np.zeros((80, 96)))
    assert main(_pair(tmp_path, tmp_path / "out", "--set", "max_disp=64")) == 2
    err = capsys.readouterr().err
    assert "warning" not in err and err.startswith("error:") and err.count("\n") == 1


# ---- the Rust command line says the same

RUST_BIN = os.environ.get("OPTFLOW_BIN") or os.path.join(
    os.path.dirname(__file__), "..", "..", "rust", "target", "release", "optflow")


@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN),
                    reason="the Rust optflow binary is not built (cargo build --release -p optflow-cli)")
@pytest.mark.parametrize("max_disp", ["64", "100.5", "1e6", "123456.7", "40", "16", "8"])
def test_python_and_rust_command_lines_print_the_same_warning(tmp_path, capsys, max_disp):
    _write_frames(tmp_path, 3, 96)
    for cmd, src in (("pair", ["--ref", str(tmp_path / "f0.png"), "--def", str(tmp_path / "f1.png")]),
                     ("seq", ["--frames", str(tmp_path)])):
        args = [cmd, *src, "--set", "outer_iters=1", "--set", f"max_disp={max_disp}"]
        assert main([*args, "--out", str(tmp_path / "py")]) == 0
        py = capsys.readouterr().err
        done = subprocess.run([RUST_BIN, *args, "--out", str(tmp_path / "rs")], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert done.stderr == py, (cmd, max_disp)
        assert ("warning" in py) == (float(max_disp) > 16)  # on 96 x 96 the coarse stage covers +-16 px whatever L is

"""A video with a display rotation is read upright, by OpenCV (Python) and by ffmpeg (Rust) alike."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from optflow_ref import synth
from optflow_ref.__main__ import main

RUST_BIN = os.environ.get("OPTFLOW_BIN") or os.path.join(
    os.path.dirname(__file__), "..", "..", "rust", "target", "release", "optflow")
W, H = 128, 80  # as stored; the quarter turns make frames of H x W


def _clip(d, rotation):
    """A 3-frame MJPEG clip of a pattern moving 1.5 px per frame along x, with the display rotation `rotation`."""
    cv2 = pytest.importorskip("cv2")
    p = synth.make_pattern(H, W, seed=3)
    path = str(d / "src.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H), isColor=False)
    for k in range(3):
        vw.write(np.round(synth.render(p, H, W, synth.translation(1.5 * k, 0.0)) * 255).astype(np.uint8))
    vw.release()
    out = d / "rot.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", rotation, "-i", path, "-c", "copy", str(out)],
                   check=True)
    return str(out)


@pytest.mark.skipif(not os.access(RUST_BIN, os.X_OK) and not shutil.which(RUST_BIN),
                    reason="the Rust optflow binary is not built (cargo build --release -p optflow-cli)")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("rotation,upright", [("0", (H, W)), ("90", (W, H)), ("-90", (W, H)), ("180", (H, W)),
                                              ("45", (H, W))])  # 45: not a quarter turn, the frames stay as stored
def test_python_and_rust_track_a_rotated_video_the_same_way(tmp_path, rotation, upright):
    video = _clip(tmp_path, rotation)
    args = ["seq", "--frames", video, "--set", "outer_iters=2", "--set", "max_disp=16"]
    assert main([*args, "--out", str(tmp_path / "py")]) == 0
    done = subprocess.run([RUST_BIN, *args, "--out", str(tmp_path / "rs")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    ux = {}
    for who in ("py", "rs"):
        ux[who] = [np.load(tmp_path / who / "frame_00002" / f"{name}.npy") for name in ("ux", "uy")]
    for a, b in zip(*ux.values()):
        assert a.shape == b.shape == upright, rotation
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 1000  # of 10240 pixels: the 28 px border strips are not reported
        assert np.abs(a[both] - b[both]).max() < 5e-3, rotation
    # the translation (3, 0) of the stored frames is along x or along y of the upright ones
    mx, my = (float(np.nanmedian(a)) for a in ux["py"])
    along_x = rotation in ("0", "180", "45")
    assert abs(abs(mx) - (3.0 if along_x else 0.0)) < 0.1 and abs(abs(my) - (0.0 if along_x else 3.0)) < 0.1, (mx, my)

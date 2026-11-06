"""Temporal averaging front-end of `seq` (tavg, tavg_align)."""
import numpy as np
import pytest
from PIL import Image

from optflow_ref import synth
from optflow_ref.config import Config
from optflow_ref.io import iter_frames
from optflow_ref.tavg import align_to


def _const_dir(tmp_path, n, h=24, w=32):
    """n image files whose gray level is frame_index / 255 (exact in 8 bit)."""
    for i in range(n):
        Image.fromarray(np.full((h, w), i, np.uint8)).save(tmp_path / f"f{i:03d}.png")
    return str(tmp_path)


def _mean_levels(a, b):
    return np.mean(np.arange(a, b + 1)) / 255.0


def test_tavg_zero_is_the_plain_frame_source(tmp_path):
    d = _const_dir(tmp_path, 9)
    a = list(iter_frames(d, 1, None, 3, 2))
    b = list(iter_frames(d, 1, None, 3, 2, tavg=0, align=True))  # align is moot without averaging
    assert len(a) == len(b) == 3 and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_plain_average_windows_and_clipping(tmp_path):
    d = _const_dir(tmp_path, 16)
    got = [f[0, 0] for f in iter_frames(d, 0, None, 5, 1, tavg=2)]
    # sampled 0, 5, 10, 15: windows 0..2 (clipped), 3..7, 8..12, 13..15 (clipped at the last frame)
    assert got == pytest.approx([_mean_levels(0, 2), _mean_levels(3, 7), _mean_levels(8, 12), _mean_levels(13, 15)], abs=1e-12)


def test_plain_average_start_stop_and_scale(tmp_path):
    d = _const_dir(tmp_path, 20)
    fr = list(iter_frames(d, 2, 13, 5, 2, tavg=1))
    # sampled 2, 7, 12 (< stop 13): windows 1..3, 6..8, 11..13 (the window may reach past stop)
    assert [f[0, 0] for f in fr] == pytest.approx([_mean_levels(1, 3), _mean_levels(6, 8), _mean_levels(11, 13)], abs=1e-12)
    assert fr[0].shape == (12, 16)  # downscaled by 2


def test_plain_average_reduces_noise_by_sqrt_n(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, (40, 40))
    fr = [base + rng.normal(0, 0.02, base.shape) for _ in range(9)]
    for i, f in enumerate(fr):
        Image.fromarray(np.round(np.clip(f, 0, 1) * 65535).astype(np.uint16)).save(tmp_path / f"f{i}.png")
    out = next(iter_frames(str(tmp_path), 4, 5, 1, 1, tavg=4))
    assert out.std() > 0
    err = out - base
    assert err.std() == pytest.approx(0.02 / 3.0, rel=0.3)  # 9 frames


def _moving_frames(n, shifts, N=128, noise=0.0, seed=0):
    pat = synth.make_pattern(N, N, seed=3)
    rng = np.random.default_rng(seed)
    return [synth.render(pat, N, N, synth.Affine(np.eye(2), t=(sx, sy), c=(N / 2, N / 2))) + noise * rng.normal(size=(N, N))
            for sx, sy in shifts]


def test_align_to_undoes_a_translation():
    N = 128
    fr = _moving_frames(2, [(0.0, 0.0), (2.5, -1.5)], N)
    out = align_to(fr[0], fr[1], Config(max_disp=8.0))
    # fr[1] is fr[0] moved by (2.5, -1.5); aligned back it matches fr[0] away from the border
    m = (slice(20, -20), slice(20, -20))
    assert np.abs(out[m] - fr[0][m]).max() < 0.05
    assert np.abs(fr[1][m] - fr[0][m]).max() > 0.15


def test_dense_alignment_follows_a_local_deformation_that_an_affine_fit_misses():
    N = 256
    pat = synth.make_pattern(N, N, seed=3)
    f0 = synth.render(pat, N, N)
    f1 = synth.render(pat, N, N, synth.Affine(np.eye(2), t=(2.0, 0.0), c=(N / 2, N / 2)))
    other = np.where(np.arange(N)[None, :] >= 150, f1, f0)  # only the right part moves
    m = (slice(20, -20), slice(200, -12))
    dense = align_to(f0, other, Config(max_disp=8.0, tavg_model="dense"))
    affine = align_to(f0, other, Config(max_disp=8.0))
    err = lambda a: np.sqrt(np.mean((a[m] - f0[m]) ** 2))
    assert err(other) > 0.04 and err(dense) < 0.4 * err(affine) and err(dense) < 0.02


def test_align_to_returns_none_when_no_motion_can_be_estimated():
    flat = np.full((128, 128), 0.5)
    assert align_to(flat, flat + 0.0, Config(max_disp=8.0)) is None


def test_aligned_average_beats_plain_average_on_moving_content(tmp_path):
    shifts = [(0.6 * i, 0.3 * i) for i in range(7)]  # drift of 0.6 / 0.3 px per frame
    fr = _moving_frames(7, shifts, noise=0.0)
    for i, f in enumerate(fr):
        Image.fromarray(np.round(np.clip(f, 0, 1) * 65535).astype(np.uint16)).save(tmp_path / f"f{i}.png")
    centre = fr[3]
    m = (slice(20, -20), slice(20, -20))
    plain = next(iter_frames(str(tmp_path), 3, 4, 1, 1, tavg=3))
    aligned = next(iter_frames(str(tmp_path), 3, 4, 1, 1, tavg=3, align=True, cfg=Config(max_disp=8.0)))
    e_plain = np.sqrt(np.mean((plain[m] - centre[m]) ** 2))
    e_aligned = np.sqrt(np.mean((aligned[m] - centre[m]) ** 2))
    assert e_aligned < 0.3 * e_plain


def test_align_to_mask_restricts_the_motion_fit_to_the_region(tmp_path):
    N = 192
    mov = _moving_frames(7, [(0.7 * i, 0.0) for i in range(7)], N)
    stat = np.roll(_moving_frames(1, [(0.0, 0.0)], N)[0], 31, axis=0)  # a different texture, never moving
    fr = [np.where(np.arange(N)[None, :] < 112, stat, f) for f in mov]  # the left 112 columns are a static surround
    roi = np.zeros((N, N), bool)
    roi[:, 136:] = True
    m = (slice(20, -20), slice(140, -12))  # inside the moving part and the ROI
    err = lambda a: np.sqrt(np.mean((a[m] - fr[3][m]) ** 2))
    masked = align_to(fr[3], fr[0], Config(max_disp=8.0), roi)
    unmasked = align_to(fr[3], fr[0], Config(max_disp=8.0))
    assert err(fr[0]) > 0.04 and err(masked) < 2e-3 and err(unmasked) > 4e-3  # the static majority pulls the global fit
    # the same through the frame source: mask_fn is called with the shape of the downscaled frame
    for i, f in enumerate(fr):
        Image.fromarray(np.round(np.clip(f, 0, 1) * 65535).astype(np.uint16)).save(tmp_path / f"f{i}.png")
    shapes = []
    out = next(iter_frames(str(tmp_path), 3, 4, 1, 1, tavg=3, align=True, cfg=Config(max_disp=8.0),
                           mask_fn=lambda shape: shapes.append(shape) or roi))
    assert shapes == [(N, N)] * len(shapes) and shapes and out.shape == (N, N)


def test_cli_seq_tavg_off_is_bit_identical_and_on_changes_frames(tmp_path):
    from optflow_ref.__main__ import main
    N = 160
    pat = synth.make_pattern(N, N, seed=3)
    d = tmp_path / "frames"
    d.mkdir()
    for k in range(9):
        f = synth.render(pat, N, N, synth.uniform_strain(0.0005 * k, 0.0, (N / 2, N / 2), t=(0.3 * k, 0.1 * k)))
        Image.fromarray(np.round(f * 255).astype(np.uint8)).save(d / f"f{k:02d}.png")
    outs = {}
    for tag, extra in (("default", []), ("off", ["--set", "tavg=0"]), ("on", ["--set", "tavg=1"])):
        o = tmp_path / tag
        assert main(["seq", "--frames", str(d), "--step", "4", "--out", str(o), "--raw"] + extra) == 0
        outs[tag] = np.load(o / "frame_00001" / "raw_ux.npy")
    assert np.array_equal(outs["default"], outs["off"])
    assert not np.array_equal(outs["default"], outs["on"])

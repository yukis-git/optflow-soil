import numpy as np
import pytest

from optflow_ref import synth
from optflow_ref.config import Config
from optflow_ref.poc import Grid, clean, coarse_flow, grid_sample, n_levels, poc_shift


@pytest.fixture(scope="module")
def ref():
    p = synth.make_pattern(256, 256, seed=7)
    return p, synth.render(p, 256, 256)


def test_poc_shift_subpixel(ref):
    p, a = ref
    b = synth.render(p, 256, 256, synth.translation(2.3, -1.6))
    dx, dy, pk = poc_shift(a[96:160, 96:160], b[96:160, 96:160])
    assert abs(dx - 2.3) < 0.1 and abs(dy + 1.6) < 0.1 and pk > 0.3


def test_levels_from_max_disp():
    assert n_levels(16, 64, (1000, 1000)) == 0
    assert n_levels(64, 64, (1000, 1000)) == 2
    assert n_levels(64, 64, (100, 1000)) == 2  # rectangular 16 x 128 window at level 2
    assert n_levels(64, 64, (40, 1000)) == 0  # coarser levels leave no room for a 16 px window


def test_clean_replaces_outlier():
    ux = np.ones((5, 5)); ux[2, 2] = 9.0
    g = clean(Grid(np.arange(5), np.arange(5), ux, np.zeros((5, 5)), np.ones((5, 5))), 0.1, 2.0)
    assert g.ux[2, 2] == pytest.approx(1.0)


def test_grid_sample_bilinear():
    g = Grid(np.array([0, 10]), np.array([0, 10]), np.array([[0.0, 1.0], [2.0, 3.0]]), np.zeros((2, 2)), np.ones((2, 2)))
    ux, _ = grid_sample(g, np.array([5.0, -3.0]), np.array([5.0, 20.0]))
    np.testing.assert_allclose(ux, [1.5, 2.0])


def test_coarse_flow_large_translation(ref):
    p, a = ref
    b = synth.render(p, 256, 256, synth.translation(21.4, -13.2))
    g = coarse_flow(a, b, Config(max_disp=32))
    inner = np.s_[1:-2, 1:-2]
    assert np.abs(g.ux[inner] - 21.4).max() < 0.15
    assert np.abs(g.uy[inner] + 13.2).max() < 0.15


def test_level_window_rectangular_for_strips():
    from optflow_ref.poc import level_window

    assert level_window((500, 500), 64) == (64, 64)
    assert level_window((40, 600), 64) == (16, 256)
    assert level_window((600, 40), 64) == (256, 16)
    assert level_window((10, 600), 64) is None


def test_coarse_flow_large_motion_in_narrow_strip():
    # a 96 px tall strip moving 150 px sideways: square windows cannot reach this range
    p = synth.make_pattern(96, 1024, seed=8, margin=200)
    a = synth.render(p, 96, 1024)
    b = synth.render(p, 96, 1024, synth.translation(150.3, 2.2))
    g = coarse_flow(a, b, Config(max_disp=200))
    cols = g.xs > 150 + 64  # windows whose content is present in both images
    assert np.abs(g.ux[1:-1, cols] - 150.3).max() < 0.2
    assert np.abs(g.uy[1:-1, cols] - 2.2).max() < 0.2


def test_coarse_flow_uses_the_initial_guess(ref):
    # 100 px is beyond what max_disp=64 can find from scratch; the guess (e.g. the previous
    # frame's field) must carry the search, scaled down through the pyramid
    p, a = ref
    b = synth.render(p, 256, 256, synth.translation(100.0, -40.0))
    cfg = Config(max_disp=64)
    guess = (np.full((256, 256), 99.0), np.full((256, 256), -39.0))
    g = coarse_flow(a, b, cfg, init_dense=guess)
    err = np.hypot(g.ux - 100.0, g.uy + 40.0)
    assert (err < 0.2).mean() > 0.95
    g0 = coarse_flow(a, b, cfg)  # without the guess the shift is out of range
    assert (np.hypot(g0.ux - 100.0, g0.uy + 40.0) < 0.2).mean() < 0.5


def test_clean_without_a_valid_point_returns_the_fallback():
    xs, ys = np.arange(3) * 16, np.arange(4) * 16
    g = Grid(xs, ys, np.full((4, 3), 7.0), np.full((4, 3), -7.0), np.zeros((4, 3)))  # every peak weak
    fx, fy = np.full((4, 3), 2.5), np.full((4, 3), -1.5)
    out = clean(g, 0.1, 2.0, fallback=(fx, fy))
    np.testing.assert_array_equal(out.ux, fx)
    np.testing.assert_array_equal(out.uy, fy)
    assert (out.peak == 0).all()
    out = clean(g, 0.1, 2.0)  # without a guess: zeros
    assert (out.ux == 0).all() and (out.uy == 0).all()


def test_coarse_flow_keeps_the_guess_where_nothing_matches():
    n = 192
    a = synth.render(synth.make_pattern(n, n, seed=1), n, n)
    b = np.full((n, n), 0.5)  # blank image: no correlation peak anywhere, on any level
    g = coarse_flow(a, b, Config(max_disp=8), init_dense=(np.full((n, n), 3.0), np.full((n, n), -2.0)))
    assert (g.peak == 0).all()
    np.testing.assert_allclose(g.ux, 3.0)
    np.testing.assert_allclose(g.uy, -2.0)


def test_capture_range_is_the_window_over_four_times_two_to_the_levels():
    from optflow_ref.poc import capture_range

    assert capture_range(128, 64, (512, 512)) == (64.0, 64.0)  # L = 3, a 32 px window: only +-64 px, not 128
    assert capture_range(32, 64, (512, 512)) == (32.0, 32.0)  # the default: reaches max_disp exactly
    assert capture_range(16, 64, (1000, 1000)) == (16.0, 16.0)  # L = 0: the nominal window
    assert capture_range(8, 32, (1000, 1000)) == (8.0, 8.0)
    # a strip: the window is 32 x 128 at L = 3, so the axes differ; (cy, cx)
    cy, cx = capture_range(128, 64, (421, 3002))
    assert (cy, cx) == (64.0, 256.0)
    assert isinstance(cy, float) and isinstance(cx, float)
    # L = 0 although max_disp asks for more: no coarser level has a window (a 40 px tall image)
    assert capture_range(128, 64, (40, 1000)) == (16.0, 16.0)


def test_capture_range_takes_the_window_of_the_downsampled_level_not_of_shape_shifted_by_L():
    from optflow_ref.poc import capture_range, downsample, level_window

    # 95 >> 1 = 47 allows a 16 px window, but downsample keeps ceil(95 / 2) = 48 rows: 32 px fit
    assert level_window((95 >> 1, 95 >> 1), 64) == (16, 16)
    assert downsample(np.zeros((95, 95))).shape == (48, 48)
    assert level_window((48, 48), 64) == (32, 32)
    assert capture_range(32, 64, (95, 95)) == (16.0, 16.0)


@pytest.mark.parametrize("max_disp,shape", [(128, (512, 512)), (32, (512, 512)), (16, (1000, 1000)),
                                            (128, (421, 3002)), (32, (95, 95)), (100, (96, 512)),
                                            (500, (700, 333)), (128, (40, 1000))])
def test_capture_range_matches_the_coarsest_window_coarse_flow_uses(monkeypatch, max_disp, shape):
    from optflow_ref import poc
    from optflow_ref.poc import capture_range

    wins = []

    def stub(I1, I2, xs, ys, win, init=None, iters=3):  # records the window of each level, finds nothing
        wins.append(win)
        zero = np.zeros((len(ys), len(xs)))
        return Grid(xs, ys, zero, zero.copy(), zero.copy())

    monkeypatch.setattr(poc, "poc_grid", stub)
    a = np.zeros(shape)
    L = poc.n_levels(max_disp, 64, shape)
    poc.coarse_flow(a, a, Config(max_disp=max_disp))
    assert len(wins) == L + 1
    wy, wx = (64, 64) if L == 0 else wins[0]  # wins[0]: the coarsest level
    assert capture_range(max_disp, 64, shape) == (wy / 4 * 2**L, wx / 4 * 2**L)

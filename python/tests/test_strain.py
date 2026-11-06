import numpy as np

from optflow_ref.strain import displacement_gradient, strain_fields


def test_quadratic_field_gradient_exact():
    yy, xx = np.mgrid[0:60, 0:70].astype(float)
    ux = 0.01 * xx + 0.002 * yy + 1e-4 * xx * yy
    uy = -0.003 * xx + 1e-4 * yy**2
    dudx, dudy, dvdx, dvdy = displacement_gradient(ux, uy, 15)
    s = np.s_[10:-10, 10:-10]
    np.testing.assert_allclose(dudx[s], (0.01 + 1e-4 * yy)[s], atol=1e-10)
    np.testing.assert_allclose(dudy[s], (0.002 + 1e-4 * xx)[s], atol=1e-10)
    np.testing.assert_allclose(dvdx[s], -0.003, atol=1e-10)
    np.testing.assert_allclose(dvdy[s], (2e-4 * yy)[s], atol=1e-10)


def test_nan_holes_are_skipped_not_spread():
    yy, xx = np.mgrid[0:60, 0:60].astype(float)
    ux = 0.02 * xx
    ux[28:32, 28:32] = np.nan
    dudx, *_ = displacement_gradient(ux, 0 * ux, 15)
    np.testing.assert_allclose(dudx[20:40, 20:40], 0.02, atol=1e-10)
    big = np.full((60, 60), np.nan); big[:, :5] = 0.0
    assert np.isnan(displacement_gradient(big, big, 15)[0][30, 30])


def test_green_lagrange_of_simple_shear():
    s = strain_fields(np.array(0.0), np.array(0.2), np.array(0.0), np.array(0.0))
    assert s["Exx"] == 0 and np.isclose(s["Eyy"], 0.02) and np.isclose(s["Exy"], 0.1)
    assert np.isclose(s["vol"], 0.0)


def test_half_window_gradient_one_sided_across_a_jump():
    import numpy as np
    from optflow_ref.strain import displacement_gradient

    yy, xx = np.mgrid[0:60, 0:80].astype(float)
    # u = 0.01 x - 0.004 y with a jump of 2 px at x = 40.5 (between columns 40 and 41)
    ux = 0.01 * xx - 0.004 * yy + np.where(xx > 40.5, 2.0, 0.0)
    uy = 0.003 * xx + 0.002 * yy
    full = displacement_gradient(ux, uy, 15)
    assert abs(full[0][30, 41] - 0.01) > 0.05  # the full window across the jump is wrong
    half = np.full(ux.shape, -1, np.int8)
    half[:, 41:48] = 0  # n_0 = +x: window on the side away from the jump...
    half[:, 33:41] = 4  # ... and n_4 = -x on the left
    g = displacement_gradient(ux, uy, 15, half=half)
    for c in (34, 38, 40, 41, 44, 47):
        for k, ref in zip(range(4), (0.01, -0.004, 0.003, 0.002)):
            assert abs(g[k][30, c] - ref) < 1e-9, (c, k)
    # pixels with half = -1 are the plain full-window fit, bit for bit
    same = np.array_equal(g[0][:, :30], full[0][:, :30], equal_nan=True)
    assert same

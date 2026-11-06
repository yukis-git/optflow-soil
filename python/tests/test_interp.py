import numpy as np
from scipy import ndimage

from optflow_ref.interp import bspline_coeffs, gauss_blur, sample, warp


def test_sample_reproduces_grid_values():
    a = np.random.default_rng(0).normal(size=(20, 30))
    yy, xx = np.mgrid[0:20, 0:30].astype(float)
    np.testing.assert_allclose(sample(bspline_coeffs(a), xx, yy), a, atol=1e-9)


def test_sample_matches_scipy_mirror_spline_inside():
    a = np.random.default_rng(1).normal(size=(40, 50))
    rng = np.random.default_rng(2)
    x = rng.uniform(5, 44, 500)
    y = rng.uniform(5, 34, 500)
    ref = ndimage.map_coordinates(a, [y, x], order=3, mode="mirror")
    np.testing.assert_allclose(sample(bspline_coeffs(a), x, y), ref, atol=1e-8)


def test_complex_sampling_is_componentwise():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    x, y = np.array([3.3, 7.9]), np.array([4.1, 10.6])
    got = sample(bspline_coeffs(a), x, y)
    want = sample(bspline_coeffs(a.real), x, y) + 1j * sample(bspline_coeffs(a.imag), x, y)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_warp_shifts_smooth_image():
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    img = np.sin(0.3 * xx) * np.cos(0.2 * yy)
    out = warp(img, np.full_like(img, 0.4), np.full_like(img, -0.7))
    want = np.sin(0.3 * (xx + 0.4)) * np.cos(0.2 * (yy - 0.7))
    np.testing.assert_allclose(out[8:-8, 8:-8], want[8:-8, 8:-8], atol=2e-3)


def test_gauss_blur_matches_scipy():
    a = np.random.default_rng(4).normal(size=(30, 40))
    ref = ndimage.gaussian_filter(a, 2.0, mode="nearest", truncate=4.0)
    np.testing.assert_allclose(gauss_blur(a, 2.0), ref, atol=1e-12)

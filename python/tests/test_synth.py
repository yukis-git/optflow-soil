import numpy as np

from optflow_ref import synth


def test_affine_inverse():
    f = synth.uniform_strain(0.05, -0.02, (50, 40), exy=0.01, t=(3.0, -2.0))
    X, Y = np.array([1.0, 70.0]), np.array([5.0, 33.0])
    np.testing.assert_allclose(f.inv(*f.fwd(X, Y)), (X, Y), atol=1e-12)


def test_shear_band_inverse_and_gradient():
    f = synth.ShearBand(4.0, 5.0, (64, 64), angle_deg=20)
    X, Y = np.array([10.0, 60.0, 90.0]), np.array([20.0, 66.0, 100.0])
    np.testing.assert_allclose(f.inv(*f.fwd(X, Y)), (X, Y), atol=1e-12)
    h = 1e-5
    ux = lambda X, Y: f.fwd(X, Y)[0] - X
    num = (ux(X + h, Y) - ux(X - h, Y)) / (2 * h)
    np.testing.assert_allclose(f.grad(X, Y)[0], num, atol=1e-7)


def test_reference_render_is_deterministic_and_textured():
    p = synth.make_pattern(64, 64, seed=3)
    a = synth.render(p, 64, 64)
    b = synth.render(synth.make_pattern(64, 64, seed=3), 64, 64)
    np.testing.assert_array_equal(a, b)
    assert 0.0 <= a.min() and a.max() <= 1.0 and a.std() > 0.05


def test_integer_translation_shifts_pixels():
    p = synth.make_pattern(64, 64, seed=4)
    a = synth.render(p, 64, 64)
    b = synth.render(p, 64, 64, synth.translation(3.0, 2.0))
    np.testing.assert_allclose(b[2:, 3:], a[:-2, :-3], atol=1e-12)


def test_degrade_quantises_and_clips():
    img = np.linspace(0, 1, 100).reshape(10, 10)
    out = synth.degrade(img, gain=1.5, bits=8)
    assert out.max() == 1.0
    np.testing.assert_allclose(out * 255, np.round(out * 255), atol=1e-9)


def test_scratches_are_deterministic_bounded_and_nonzero():
    a = synth.scratches(128, 128, 10, seed=5)
    np.testing.assert_array_equal(a, synth.scratches(128, 128, 10, seed=5))
    assert not np.array_equal(a, synth.scratches(128, 128, 10, seed=6))
    assert a.min() >= 0.0 and a.max() < 0.2 and (a > 0).any()


def _crack(alpha=30.0, o=3.0, s=1.0, **kw):
    return synth.Crack((60.0, 50.0), alpha, o, s, **kw)


def test_crack_fwd_inv_consistent_on_both_sides():
    f = _crack(plus=synth.uniform_strain(0.01, -0.005, (60.0, 50.0)))
    yy, xx = np.mgrid[0:100, 0:120].astype(np.float64)
    x, y = f.fwd(xx, yy)
    Xb, Yb = f.inv(x, y)
    assert np.isfinite(Xb).all()
    assert np.abs(Xb - xx).max() < 1e-9 and np.abs(Yb - yy).max() < 1e-9
    side = f.n[0] * (xx - 60.0) + f.n[1] * (yy - 50.0)
    assert (side < -5).any() and (side > 5).any()


def test_crack_gap_is_nan_and_rendered_dark():
    f = _crack(alpha=0.0, o=4.0, s=0.0)
    x = np.array([[58.0, 59.0, 61.0, 63.0, 65.0, 62.0]])
    Xb, _ = f.inv(x, np.full_like(x, 50.0))
    assert np.isnan(Xb[0, 2]) and np.isnan(Xb[0, 3]) and np.isfinite(Xb[0, 1]) and np.isfinite(Xb[0, 4])
    pat = synth.make_pattern(100, 120, seed=3)
    img = synth.render(pat, 100, 120, f)
    # the gap is the columns whose centre lies in [60, 64): painted gap_value 0.1 (the pattern never goes below 0.25)
    assert np.allclose(img[:, 61:63], 0.1)
    assert img[:, 10:50].min() > 0.2


def test_crack_displacement_jump_equals_J():
    f = _crack(alpha=30.0, o=3.0, s=-1.5)
    e = 0.01
    P = np.array([60.0, 50.0])
    pm, pp = P - e * f.n, P + e * f.n  # a point on each side of the line, 2e apart across it
    xm, ym = f.fwd(*pm)
    xp, yp = f.fwd(*pp)
    jump = np.array([xp - xm, yp - ym]) - 2 * e * f.n
    assert np.allclose(jump, f.J, atol=1e-12)
    assert np.allclose(f.J, 3.0 * f.n - 1.5 * f.t)
    u = synth.displacement(f, 100, 120)
    assert u[0].shape == (100, 120)


def test_render_without_gap_unchanged_by_the_gap_code():
    pat = synth.make_pattern(64, 64, seed=5)
    f = synth.uniform_strain(0.01, 0.0, (32.0, 32.0), t=(1.2, -0.7))
    a = synth.render(pat, 64, 64, f)
    assert np.isfinite(a).all()
    # a crack with zero opening and no slip has no gap: it agrees with the identity (its larger blob window only adds tails < 1e-5)
    c0 = synth.Crack((32.0, 32.0), 0.0, 0.0, 0.0)
    assert np.abs(synth.render(pat, 64, 64, c0) - synth.render(pat, 64, 64, synth.Affine(np.eye(2)))).max() < 1e-5

import pytest

from optflow_ref.config import Config


def test_json_round_trip():
    c = Config(lam=10.0, max_disp=200.0)
    assert Config.from_json(c.to_json()) == c


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_json('{"lambda": 3}')


def test_validate_rejects_bad_window():
    with pytest.raises(ValueError):
        Config(poc_win=48).validate()
    with pytest.raises(ValueError):
        Config(lam=4.0).validate()
    with pytest.raises(ValueError):
        Config(strain_win=14).validate()


def test_validate_rejects_nonpositive_phase_tol():
    with pytest.raises(ValueError):
        Config(phase_tol=0.0).validate()


def test_edge_guard_resolution_and_dc_cut_validation():
    c = Config()
    assert c.edge_guard_px() == pytest.approx(3 * 8.0 * 1.6 / (2 * 3.141592653589793 * 0.5))
    assert Config(edge_guard=5).edge_guard_px() == 5.0
    assert Config(edge_guard=0).edge_guard_px() == 0.0
    with pytest.raises(ValueError):
        Config(dc_cut=0).validate()


def test_edge_guard_min_area_resolution():
    assert Config().edge_guard_min_area_px() == pytest.approx(2 * (8.0 * 1.6) ** 2)
    assert Config(edge_guard_min_area=0).edge_guard_min_area_px() == 0.0


FLOAT_FIELDS = ["max_disp", "poc_peak_min", "median_thresh", "lam", "bandwidth", "dc_cut", "sigma_w", "sigma_s",
                "conf_min", "edge_guard", "edge_guard_min_area", "phase_tol", "ref_update_conf"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("name", FLOAT_FIELDS)
def test_validate_rejects_non_finite_floats(name, bad):
    with pytest.raises(ValueError):
        Config(**{name: bad}).validate()


@pytest.mark.parametrize("scales", [(0.6, float("nan"), 1.6), (1.0, float("inf"))])
def test_validate_rejects_non_finite_scales(scales):
    with pytest.raises(ValueError):
        Config(scales=scales).validate()


@pytest.mark.parametrize("kw", [
    {"scales": ()}, {"bandwidth": 0.0}, {"bandwidth": -0.5}, {"max_disp": 0.0}, {"max_disp": -3.0},
    {"n_orient": 0}, {"outer_iters": 0}, {"inner_iters": 0}, {"conf_min": -0.1}, {"conf_min": 1.5},
    {"poc_peak_min": -0.1}, {"poc_peak_min": 1.1}, {"median_thresh": 0.0}])
def test_validate_rejects_out_of_range(kw):
    with pytest.raises(ValueError):
        Config(**kw).validate()


def test_validate_accepts_the_edges_of_the_ranges():
    Config(conf_min=0.0, poc_peak_min=1.0, n_orient=1, outer_iters=1, inner_iters=1).validate()
    Config(edge_guard=-1.0, edge_guard_min_area=-1.0, ref_update_conf=2.0).validate()


@pytest.mark.parametrize("text,key", [
    ('{"grid_step": 16.5}', "grid_step"), ('{"grid_step": 16.0}', "grid_step"), ('{"grid_step": true}', "grid_step"),
    ('{"grid_step": "16"}', "grid_step"), ('{"outer_iters": null}', "outer_iters"),
    ('{"lam": "8"}', "lam"), ('{"lam": true}', "lam"), ('{"lam": null}', "lam"), ('{"lam": [8]}', "lam"),
    ('{"scales": 1.0}', "scales"), ('{"scales": [1.0, "a"]}', "scales"), ('{"scales": [true]}', "scales"),
    ('{"scales": "0.6,1.0"}', "scales"), ('{"lam": NaN}', "lam"), ('{"max_disp": Infinity}', "max_disp"),
    ('{"scales": []}', "scales"), ('{"conf_min": 2}', "conf_min"), ('{"poc_win": 48}', "poc_win"),
    ('[1, 2]', "JSON object")])
def test_from_json_rejects_wrongly_typed_or_invalid_values(text, key):
    with pytest.raises(ValueError, match=key):
        Config.from_json(text)


def test_from_json_accepts_ints_for_floats_and_lists_for_scales():
    c = Config.from_json('{"lam": 10, "max_disp": 64, "scales": [0.5, 1, 2], "outer_iters": 4}')
    assert c == Config(lam=10.0, max_disp=64.0, scales=(0.5, 1.0, 2.0), outer_iters=4)
    assert isinstance(c.lam, float) and isinstance(c.scales, tuple)


BOUNDS_MSG = ("at most 64 filters (n_orient * len(scales)); sigma_w, sigma_s <= 256, strain_win <= 255 "
              "and poc_win <= 1024 required")


@pytest.mark.parametrize("kw", [
    {"n_orient": 16, "scales": (0.6, 1.0, 1.6, 2.0)},  # 4 x 16 = 64 filters
    {"n_orient": 64, "scales": (1.0,)}, {"n_orient": 1, "scales": (1.0,) * 64},
    {"sigma_w": 256.0}, {"sigma_s": 256.0}, {"sigma_w": 256, "sigma_s": 256},
    {"strain_win": 255}, {"poc_win": 1024}])
def test_validate_accepts_each_upper_bound_at_its_limit(kw):
    Config(**kw).validate()


@pytest.mark.parametrize("kw", [
    {"n_orient": 17, "scales": (0.6, 1.0, 1.6, 2.0)},  # 68 filters
    {"n_orient": 65, "scales": (1.0,)}, {"n_orient": 1, "scales": (1.0,) * 65}, {"n_orient": 22},  # 66
    {"n_orient": 2**63, "scales": (1.0,)},
    {"sigma_w": 256.0001}, {"sigma_s": 256.0001}, {"sigma_w": 1e300}, {"sigma_s": 1e300},
    {"strain_win": 257}, {"strain_win": 2**63 + 1},
    {"poc_win": 2048}, {"poc_win": 2**63}])
def test_validate_rejects_each_upper_bound_just_above_its_limit(kw):
    with pytest.raises(ValueError) as e:
        Config(**kw).validate()
    assert str(e.value) == BOUNDS_MSG


def test_upper_bound_rule_comes_last_so_earlier_messages_win():
    # one config that breaks an earlier rule and a bound: the earlier message, as before the bounds existed
    for kw, msg in [({"scales": (), "poc_win": 2048}, "scales must not be empty"),
                    ({"poc_win": 3000}, "poc_win must be a power of two >= 16"),
                    ({"strain_win": 256}, "grid_step >= 1 and strain_win odd >= 5 required"),
                    ({"sigma_w": 1e9, "lam": 4.0}, "shortest wavelength lam*min(scales) must be >= 3 px; "
                                                  "sigmas must be positive"),
                    ({"sigma_w": float("nan"), "poc_win": 2048}, "sigma_w must be a finite number"),
                    ({"sigma_s": float("inf")}, "sigma_s must be a finite number"),
                    ({"bandwidth": 0.0, "sigma_s": 1e9}, "bandwidth, max_disp and median_thresh must be positive"),
                    ({"n_orient": 0, "poc_win": 2048}, "n_orient, outer_iters and inner_iters must be >= 1"),
                    ({"conf_min": 2.0, "strain_win": 301}, "conf_min and poc_peak_min must lie in [0, 1]")]:
        with pytest.raises(ValueError) as e:
            Config(**kw).validate()
        assert str(e.value) == msg, kw


def test_from_json_and_the_cli_apply_the_upper_bounds():
    from optflow_ref.__main__ import _config

    with pytest.raises(ValueError) as e:
        Config.from_json('{"poc_win": 2048}')
    assert str(e.value) == BOUNDS_MSG
    with pytest.raises(ValueError) as e:
        Config.from_json('{"n_orient": 9223372036854775808}')  # 2**63
    assert str(e.value) == BOUNDS_MSG
    with pytest.raises(ValueError) as e:
        _config(None, ["scales=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", "n_orient=4"])  # 68 filters
    assert str(e.value) == BOUNDS_MSG
    assert Config.from_json('{"sigma_w": 256, "strain_win": 255, "poc_win": 1024}').poc_win == 1024


GUARD_MSG = ("grid_step <= 4096 and an edge guard (edge_guard, or the automatic one from lam, scales and bandwidth) "
             "<= 256 px required")


@pytest.mark.parametrize("kw", [
    {"grid_step": 4096}, {"edge_guard": 256.0}, {"edge_guard": 256}, {"grid_step": 4096, "edge_guard": 256.0},
    {"edge_guard": 0.0},  # off
    {"lam": 268.0, "scales": (1.0,)},  # automatic guard 3 * 268 / (2 pi * 0.5) = 255.9
    {"lam": 8.0, "scales": (1.0,), "bandwidth": 0.0477},  # 3 * 8 / (2 pi * 0.0477) = 80
    {"edge_guard": 256.0, "bandwidth": 1e-9}])  # an explicit guard is not the automatic one
def test_validate_accepts_grid_step_and_edge_guard_at_their_limits(kw):
    Config(**kw).validate()


@pytest.mark.parametrize("kw", [
    {"grid_step": 4097}, {"grid_step": 2**31}, {"grid_step": 2**63}, {"grid_step": 2**64 - 1}, {"grid_step": 10**30},
    {"edge_guard": 256.0001}, {"edge_guard": 1e300},
    {"lam": 268.2, "scales": (1.0,)},  # automatic guard 256.1
    {"bandwidth": 0.01},  # automatic guard 610
    {"bandwidth": 1e-300}, {"lam": 1e308, "scales": (10.0,)},  # the automatic guard overflows to inf
    {"grid_step": 4097, "edge_guard": 300.0}])
def test_validate_rejects_grid_step_and_edge_guard_just_above_their_limits(kw):
    with pytest.raises(ValueError) as e:
        Config(**kw).validate()
    assert str(e.value) == GUARD_MSG


def test_guard_rule_comes_after_the_upper_bounds():
    for kw, msg in [({"poc_win": 2048, "grid_step": 5000}, BOUNDS_MSG),
                    ({"sigma_w": 300.0, "edge_guard": 300.0}, BOUNDS_MSG),
                    ({"n_orient": 2**63, "grid_step": 2**63}, BOUNDS_MSG),
                    ({"grid_step": 2**63, "scales": ()}, "scales must not be empty"),
                    ({"grid_step": 2**63, "poc_win": 3000}, "poc_win must be a power of two >= 16"),
                    ({"edge_guard": 1e300, "conf_min": 2.0}, "conf_min and poc_peak_min must lie in [0, 1]")]:
        with pytest.raises(ValueError) as e:
            Config(**kw).validate()
        assert str(e.value) == msg, kw


def test_from_json_and_the_cli_apply_the_guard_rule():
    from optflow_ref.__main__ import _config

    for text in ('{"grid_step": 4097}', '{"grid_step": 9223372036854775808}', '{"edge_guard": 256.5}',
                 '{"bandwidth": 0.001}'):
        with pytest.raises(ValueError) as e:
            Config.from_json(text)
        assert str(e.value) == GUARD_MSG, text
    with pytest.raises(ValueError) as e:
        _config(None, ["grid_step=9223372036854775808"])
    assert str(e.value) == GUARD_MSG
    assert Config.from_json('{"grid_step": 4096, "edge_guard": 256}').grid_step == 4096


def test_crack_keys_default_off_and_round_trip():
    c = Config()
    assert (c.crack_split, c.crack_out, c.split_tol, c.split_gain, c.split_from_outer, c.crack_r, c.crack_a) == (
        False, False, 0.25, 0.7, 0, 8.0, 12.0)
    assert (c.crack_sides, c.side_from_outer, c.crack_guard_frac) == (False, 2, 0.5)
    c2 = Config(crack_split=True, crack_out=True, split_tol=0.3, split_gain=0.5, split_from_outer=2, crack_r=5.0, crack_a=4.0)
    assert Config.from_json(c2.to_json()) == c2
    with pytest.raises(ValueError, match="must be true or false"):
        Config.from_json('{"crack_split": 1}')
    with pytest.raises(ValueError, match="must be an integer"):
        Config.from_json('{"split_from_outer": 1.5}')


@pytest.mark.parametrize("kw", [dict(split_tol=0.0), dict(split_tol=float("nan")), dict(split_gain=0.0),
                                dict(split_gain=1.5), dict(split_from_outer=-1), dict(split_from_outer=8),
                                dict(crack_r=0.0), dict(crack_sides=True), dict(crack_split=True, crack_sides=True, side_from_outer=0),
                                dict(crack_split=True, crack_sides=True, side_from_outer=8), dict(crack_guard_frac=1.5),
                                dict(crack_guard_frac=-0.1), dict(crack_guard_frac=float('nan')), dict(crack_a=-1.0), dict(crack_a=float("inf")), dict(crack_split=1)])
def test_crack_keys_validated(kw):
    with pytest.raises(ValueError):
        Config(**kw).validate()


def test_to_json_leaves_out_default_crack_keys_so_the_rust_cli_still_reads_it():
    import json

    from optflow_ref.config import CRACK_KEYS

    d = json.loads(Config().to_json())
    assert not any(k in d for k in CRACK_KEYS)
    assert "prewarp_freeze_at" in d and d["seq_mode"] == "chain"
    d2 = json.loads(Config(crack_split=True, crack_a=4.0).to_json())
    assert d2["crack_split"] is True and d2["crack_a"] == 4.0 and "split_tol" not in d2
    assert Config.from_json(Config(crack_split=True, crack_a=4.0).to_json()) == Config(crack_split=True, crack_a=4.0)


def test_crack_sides_config_round_trip():
    c = Config(crack_split=True, crack_sides=True, side_from_outer=3, crack_guard_frac=0.0)
    c.validate()
    assert Config.from_json(c.to_json()) == c

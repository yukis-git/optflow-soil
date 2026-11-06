import os

import numpy as np
import pytest
from PIL import Image

from optflow_ref.io import downscale, iter_frames, list_images, load_roi, polygon_mask, read_image, save_result, to_gray


def test_to_gray_rgb_uint8():
    a = np.zeros((2, 2, 3), np.uint8); a[..., 0] = 255
    np.testing.assert_allclose(to_gray(a), 0.299)


def test_read_16bit_png(tmp_path):
    a = (np.arange(64).reshape(8, 8) * 1000).astype(np.uint16)
    Image.fromarray(a).save(tmp_path / "a.png")
    np.testing.assert_allclose(read_image(str(tmp_path / "a.png")), a / 65535.0)


def test_downscale_box():
    np.testing.assert_allclose(downscale(np.arange(16.0).reshape(4, 4), 2), [[2.5, 4.5], [10.5, 12.5]])


def test_polygon_mask_scaled():
    m = polygon_mask((10, 10), [[0, 0], [10, 0], [10, 10], [0, 10]], scale=2)
    assert m[:5, :5].all() and not m[8:, 8:].any()


def test_iter_frames_from_video(tmp_path):
    import cv2
    path = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for k in range(5):
        vw.write(np.full((24, 32, 3), 40 * k, np.uint8))
    vw.release()
    frames = list(iter_frames(path, step=2))
    assert len(frames) == 3 and frames[0].shape == (24, 32)
    assert frames[2].mean() > frames[0].mean()


def _write_images(tmp_path, names):
    for k, n in enumerate(names):
        Image.fromarray(np.full((8, 8), 10 * k, np.uint8)).save(tmp_path / n)


def test_list_images_uses_plain_string_order(tmp_path):
    _write_images(tmp_path, ["IMG_9.png", "IMG_10.png", "IMG_2.png"])
    assert [os.path.basename(p) for p in list_images(str(tmp_path))] == ["IMG_10.png", "IMG_2.png", "IMG_9.png"]


@pytest.mark.parametrize("kw", [{"step": 0}, {"step": -1}, {"start": -1}])
def test_iter_frames_rejects_bad_start_and_step(tmp_path, kw):
    _write_images(tmp_path, ["a.png", "b.png"])
    with pytest.raises(ValueError):
        list(iter_frames(str(tmp_path), **kw))


@pytest.mark.parametrize("kw", [{"step": 0}, {"start": -1}])
def test_iter_video_rejects_bad_start_and_step(tmp_path, kw):
    import cv2
    path = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for k in range(3):
        vw.write(np.full((24, 32, 3), 40 * k, np.uint8))
    vw.release()
    with pytest.raises(ValueError):
        list(iter_frames(path, **kw))


def test_save_result_writes_valid_mask(tmp_path):
    from optflow_ref.track import Result

    ux = np.array([[1.0, np.nan], [2.0, 3.0]])
    res = Result(ux, ux, np.full((2, 2), 0.9), {k: ux for k in ("Exx", "Eyy", "Exy", "gmax", "vol")}, ux, ux)
    save_result(str(tmp_path), res)
    v = np.load(tmp_path / "valid.npy")
    assert v.dtype == np.uint8 and v.tolist() == [[1, 0], [1, 1]]
    assert np.load(tmp_path / "conf.npy").min() > 0.8  # conf keeps the unguarded values


@pytest.mark.parametrize("factor", [0, -1])
def test_downscale_rejects_factor_below_one(factor):
    with pytest.raises(ValueError, match=f"scale must be >= 1, got {factor}"):
        downscale(np.zeros((4, 4)), factor)


def test_load_roi_reads_the_polygon(tmp_path):
    (tmp_path / "roi.json").write_text('{"polygon": [[1, 2], [30, 2], [30, 40]]}')
    assert load_roi(str(tmp_path / "roi.json")) == [[1, 2], [30, 2], [30, 40]]


@pytest.mark.parametrize("text", ['{}', '{"poly": [[0, 0], [1, 1], [0, 1]]}', '{"polygon": 3}', '{"polygon": null}', '[[0, 0]]'])
def test_load_roi_without_a_polygon_list_is_a_value_error(tmp_path, text):
    path = tmp_path / "roi.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=f"ROI file {path} has no 'polygon' list"):
        load_roi(str(path))

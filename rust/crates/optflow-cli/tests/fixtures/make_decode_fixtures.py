"""Writes small test images and decode_expected.json: what python/optflow_ref/io.read_image returns
for each (float64 grey levels, row-major), the reference for the Rust reader.

Run from the repository root:
    python/.venv/bin/python rust/crates/optflow-cli/tests/fixtures/make_decode_fixtures.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from optflow_ref.io import read_image  # noqa: E402

IMG = os.path.join(HERE, "images")
W, H = 11, 7
rng = np.random.default_rng(7)


def noise(dtype, *shape):
    hi = np.iinfo(dtype).max
    return rng.integers(0, hi + 1, size=shape, dtype=dtype)


def smooth(channels):
    y, x = np.mgrid[0:H, 0:W]
    base = [(8 * x + 5 * y + 40 * c) % 256 for c in range(channels)]
    return np.stack(base, axis=-1).astype(np.uint8) if channels > 1 else base[0].astype(np.uint8)


def main():
    import tifffile

    os.makedirs(IMG, exist_ok=True)
    out = {}

    def add(name):
        out[name] = {"w": W, "h": H, "gray": read_image(os.path.join(IMG, name)).ravel().tolist()}

    pil = {
        "gray8.png": Image.fromarray(noise(np.uint8, H, W)),
        "gray16.png": Image.fromarray(noise(np.uint16, H, W)),
        "rgb8.png": Image.fromarray(noise(np.uint8, H, W, 3)),
        "rgba8.png": Image.fromarray(noise(np.uint8, H, W, 4)),
        "la8.png": Image.fromarray(noise(np.uint8, H, W, 2), "LA"),
        "bit1.png": Image.fromarray(noise(np.uint8, H, W) > 127),
    }
    pal = Image.fromarray(noise(np.uint8, H, W) % 5, "P")
    pal.putpalette([c for k in range(5) for c in (50 * k, 255 - 40 * k, 17 * k + 3)])
    pil["palette.png"] = pal
    for name, im in pil.items():
        im.save(os.path.join(IMG, name))
    # 16-bit colour PNGs: Pillow cannot write them, so write the chunks with a minimal encoder
    for name, ch in (("rgb16.png", 3), ("rgba16.png", 4), ("la16.png", 2)):
        write_png16(os.path.join(IMG, name), noise(np.uint16, H, W, ch))
    tifffile.imwrite(os.path.join(IMG, "gray8.tif"), noise(np.uint8, H, W))
    tifffile.imwrite(os.path.join(IMG, "gray16.tif"), noise(np.uint16, H, W))
    tifffile.imwrite(os.path.join(IMG, "rgb8.tif"), noise(np.uint8, H, W, 3), photometric="rgb")
    tifffile.imwrite(os.path.join(IMG, "rgba8.tif"), noise(np.uint8, H, W, 4), photometric="rgb")
    tifffile.imwrite(os.path.join(IMG, "rgb16.tif"), noise(np.uint16, H, W, 3), photometric="rgb")
    tifffile.imwrite(os.path.join(IMG, "rgb_f32.tif"), rng.random((H, W, 3)).astype(np.float32), photometric="rgb")
    # grey float32: tifffile reads it and to_gray takes the values as they are (no scaling; these reach beyond [0, 1]).
    # Its own generator, so that the random numbers of the other files do not change
    tifffile.imwrite(os.path.join(IMG, "gray_f32.tif"),
                     (np.random.default_rng(8).random((H, W)) * 3 - 1).astype(np.float32))
    Image.fromarray(smooth(1)).save(os.path.join(IMG, "gray.jpg"), quality=95)
    Image.fromarray(smooth(3)).save(os.path.join(IMG, "rgb.jpg"), quality=95)
    # BMP (drawn last so that the random numbers of the files above do not change): Pillow reads
    # 24-bit as RGB, 8-bit with a grey palette as L, 32-bit as RGBA and 1-bit as mode "1"
    Image.fromarray(noise(np.uint8, H, W, 3)).save(os.path.join(IMG, "rgb8.bmp"))
    Image.fromarray(noise(np.uint8, H, W)).save(os.path.join(IMG, "gray8.bmp"))
    Image.fromarray(noise(np.uint8, H, W, 4), "RGBA").save(os.path.join(IMG, "rgba8.bmp"))
    Image.fromarray(noise(np.uint8, H, W) > 127).save(os.path.join(IMG, "bit1.bmp"))
    for name in sorted(os.listdir(IMG)):
        add(name)
    with open(os.path.join(HERE, "decode_expected.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(sorted(out), os.path.getsize(os.path.join(HERE, "decode_expected.json")))


def write_png16(path, a):
    """8 bytes signature + IHDR + IDAT + IEND for a 16-bit (H, W, C) array, C = 2, 3 or 4."""
    import struct
    import zlib

    h, w, c = a.shape
    color = {2: 4, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + a[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, color, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))


if __name__ == "__main__":
    main()

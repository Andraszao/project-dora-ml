#!/usr/bin/env python3
"""Regenerates the codec test fixtures with independent encoders (Pillow and ImageMagick, which wrap
libpng/libtiff), so DoraBio's decoders are checked against real-world files, not only against themselves.

    python3 make_fixtures.py        # run from this directory; needs Pillow, NumPy and ImageMagick's `convert`

Every image is 37x23 pixels with a pixel-value formula that FixtureCodecTests.cs recomputes exactly.
"""
import os
import subprocess

import numpy as np
from PIL import Image

W, H = 37, 23
y, x = np.mgrid[0:H, 0:W]
gray8 = ((x * 7 + y * 13) % 256).astype(np.uint8)
gray16 = ((x * 1723 + y * 4099) % 65536).astype(np.uint16)
alpha8 = ((x * y) % 256).astype(np.uint8)
rgb8 = np.stack([(x * 5) % 256, (y * 11) % 256, ((x + y) * 3) % 256], axis=-1).astype(np.uint8)
float32 = ((x - 18.0) * 0.25 + (y * y) * 0.125 - 3.5).astype(np.float32)

here = os.path.dirname(os.path.abspath(__file__))
out = lambda name: os.path.join(here, name)

# --- PNG (Pillow) ---
Image.fromarray(gray8, "L").save(out("gray8.png"))
Image.fromarray(gray16).save(out("gray16.png"))                 # mode I;16 -> 16-bit grayscale PNG
Image.fromarray(rgb8, "RGB").save(out("rgb8.png"))
Image.fromarray(np.stack([gray8, alpha8], axis=-1), "LA").save(out("gray_alpha8.png"))

# --- PNG using every scanline filter type ---
# libpng (via Pillow) only picked Sub and Paeth for these images, and ImageMagick ignores requests to force one
# filter. This hand-written encoder cycles row filters 0,1,2,3,4 so None/Up/Average are exercised too.
def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)

def write_png_all_filters(path, samples16):
    import struct, zlib
    bpp = 2
    rows = [samples16[r].astype(">u2").tobytes() for r in range(H)]
    raw = bytearray()
    for r, row in enumerate(rows):
        ftype = r % 5
        prev = rows[r - 1] if r > 0 else bytes(len(row))
        raw.append(ftype)
        for i, v in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][ftype]
            raw.append((v - pred) & 0xFF)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 16, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)

write_png_all_filters(out("gray16_allfilters.png"), gray16)

pgm = out("_gray16.pgm")
with open(pgm, "wb") as f:                                         # 16-bit PGM is big-endian by spec
    f.write(f"P5 {W} {H} 65535\n".encode())
    f.write(gray16.astype(">u2").tobytes())

# --- TIFF (Pillow/libtiff) ---
Image.fromarray(gray8, "L").save(out("gray8.tif"))
Image.fromarray(gray16).save(out("gray16.tif"))
Image.fromarray(gray16).save(out("gray16_deflate.tif"), compression="tiff_adobe_deflate")
Image.fromarray(gray16).save(out("gray16_packbits.tif"), compression="packbits")
Image.fromarray(gray16).save(out("gray16_lzw.tif"), compression="tiff_lzw")    # must be rejected clearly
Image.fromarray(float32, "F").save(out("float32.tif"))
Image.fromarray(rgb8, "RGB").save(out("rgb8.tif"))

# --- TIFF variants via ImageMagick/libtiff ---
subprocess.run(["convert", pgm, "-depth", "16", "-endian", "MSB", "-define", "tiff:rows-per-strip=5",
                out("gray16_bigendian_strips.tif")], check=True)
subprocess.run(["convert", pgm, "-depth", "16", "-compress", "zip", "-define", "tiff:predictor=2",
                out("gray16_deflate_predictor.tif")], check=True)
subprocess.run(["convert", out("gray8.png"), pgm, "-depth", "16", out("stack2.tif")], check=True)
os.remove(pgm)
print("fixtures written to", here)

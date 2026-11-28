#!/usr/bin/env python3
"""Regenerates brick 41's fixtures with an independent writer, tifffile (whose LZW encoder is imagecodecs', not DORA's):

    python3 -I make_pilot_fixtures.py [OUT_DIR]   # default: this directory; needs NumPy, tifffile 2026.10.15, imagecodecs 2026.11.11

  lzw_noise_u16_pred2.tif  96 x 96 uint16, LZW with the horizontal predictor, one strip. The pixels are a hash of (x, y), so
                           the strip is incompressible enough for its codes to reach 12 bits and the table to be reset:
                           PilotCoreTests asserts both, and recomputes every pixel.
  lzw_runs_u8.tif          64 x 64 uint8, LZW: a flat top half (long repeated strings, the "KwKwK" code) and a ramp below.
  ome_z4.tif               4 planes (5 x 4 uint16) with OME-XML metadata (axes ZYX): DORA must not take them as frames.
  imagej_calibrated_t3.tif ImageJ time-lapse, 3 frames (5 x 4 uint16), 0.65 um pixels and a 2 s frame interval: a denoised
                           copy must keep the ImageDescription verbatim and the pixel size.
The hash is (in uint32 arithmetic) h = ((x * 73856093) ^ (y * 19349663)) * 2654435761, pixel = h >> 16.
"""
import os
import sys

import numpy as np
import tifffile

here = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
out = lambda name: os.path.join(here, name)

y, x = np.mgrid[0:96, 0:96].astype(np.uint64)
h = ((x * 73856093) ^ (y * 19349663)) & 0xFFFFFFFF
h = (h * 2654435761) & 0xFFFFFFFF
noise = (h >> 16).astype(np.uint16)
tifffile.imwrite(out("lzw_noise_u16_pred2.tif"), noise, compression="lzw", predictor=True, rowsperstrip=96, photometric="minisblack")

ry, rx = np.mgrid[0:64, 0:64]
runs = np.where(ry < 32, 7, (rx * 3 + ry) % 256).astype(np.uint8)
tifffile.imwrite(out("lzw_runs_u8.tif"), runs, compression="lzw", rowsperstrip=64, photometric="minisblack")

py, px = np.mgrid[0:4, 0:5]
planes = np.stack([100 * k + 2 * py + px + 1 for k in range(4)]).astype(np.uint16)
tifffile.imwrite(out("ome_z4.tif"), planes, ome=True, photometric="minisblack", metadata={"axes": "ZYX"})

frames = np.stack([1000 * t + 2 * py + px + 1 for t in range(3)]).astype(np.uint16)
tifffile.imwrite(out("imagej_calibrated_t3.tif"), frames, imagej=True, resolution=(1 / 0.65, 1 / 0.65),
                 metadata={"axes": "TYX", "unit": "um", "finterval": 2.0})
print("wrote", ", ".join(sorted(f for f in os.listdir(here) if f.endswith(".tif"))))

#!/usr/bin/env python3
"""Regenerates the stack fixtures (brick 34) with independent writers: tifffile (ImageJ hyperstacks, tifffile's JSON
shape metadata, BigTIFF) and mrcfile (MRC2014), so DORA's stack readers are checked against files real tools write.

    python3 make_stack_fixtures.py    # run from this directory; needs NumPy, tifffile (2026.3.3) and mrcfile (1.5.4)

Every plane is 5 x 4 (width 5, height 4). Pixel (y, x) of the plane at position (t, z, c) is
v = 1000*t + 100*z + 10*c + 2*y + x + 1 (plus 0.25 in the float file), which StackFixtureTests.cs recomputes, so a
plane read from the wrong page, axis order or byte order can't pass.
"""
import os

import mrcfile
import numpy as np
import tifffile

H, W = 4, 5
here = os.path.dirname(os.path.abspath(__file__))
out = lambda name: os.path.join(here, name)


def value(t=0, z=0, c=0):
    y, x = np.mgrid[0:H, 0:W]
    return 1000 * t + 100 * z + 10 * c + 2 * y + x + 1


# ImageJ hyperstack, T=2, Z=3, C=2 (ImageJ stores channels fastest, then slices, then frames).
tzc = np.stack([np.stack([np.stack([value(t, z, c) for c in range(2)]) for z in range(3)]) for t in range(2)])
tifffile.imwrite(out("imagej_tzc.tif"), tzc.astype(np.uint16), imagej=True, metadata={"axes": "TZCYX"})
# (photometric="minisblack" below: without it tifffile stores a leading axis of 3 or 4 as planar RGB(A) samples of one page.)

# tifffile JSON metadata with axes: T=3 frames, C=2 channels, float32, big-endian.
tc = np.stack([np.stack([value(t=t, c=c) for c in range(2)]) for t in range(3)]).astype(np.float32) + 0.25
tifffile.imwrite(out("tifffile_tc_be.tif"), tc.astype(">f4"), photometric="minisblack", metadata={"axes": "TCYX"})

# tifffile JSON metadata without axes ({"shape": [...]} only): 5 planes along an unlabelled axis, int16.
z5 = np.stack([value(z=z) for z in range(5)]).astype(np.int16)
tifffile.imwrite(out("tifffile_noaxes.tif"), z5, photometric="minisblack", metadata={})

# BigTIFF, big-endian, 6 frames, Deflate-compressed with the horizontal predictor (as large time-lapses often are).
t6 = np.stack([value(t=t) for t in range(6)]).astype(">u2")
tifffile.imwrite(out("bigtiff_t6_be.tif"), t6, bigtiff=True, photometric="minisblack", compression="zlib",
                 predictor=True, metadata={"axes": "TYX"})

# MRC: 3 sections, uint16 (mode 6) little-endian, and int16 (mode 1) big-endian.
with mrcfile.new(out("mrc_z3_u16.mrc"), overwrite=True) as m:
    m.set_data(np.stack([value(z=z) for z in range(3)]).astype(np.uint16))
with mrcfile.new(out("mrc_z3_i16_be.mrc"), overwrite=True) as m:
    m.set_data(np.stack([value(z=z) for z in range(3)]).astype(">i2"))

for name in sorted(os.listdir(here)):
    if name.startswith(("imagej_", "tifffile_", "bigtiff_", "mrc_")):
        print(name, os.path.getsize(out(name)))

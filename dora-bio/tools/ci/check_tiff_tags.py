#!/usr/bin/env python3
"""Checks with Pillow, an independent reader, that a TIFF DORA denoised kept what it should of its input (brick 41).

    python3 -I tools/ci/check_tiff_tags.py INPUT.tif OUTPUT.tif [EXPECTED_DESCRIPTION]

The output must have the input's page count and size, float32 pages, the input's pixel size (XResolution, YResolution,
ResolutionUnit: tags 282, 283, 296) on every page, and on its first page a Software tag (305) that says which DORA and
which model made it. Its ImageDescription (270) must be the input's, verbatim, or EXPECTED_DESCRIPTION when given
(an OME-XML input's copy records the declared axes instead of the OME-XML). Exit 0 if all of that holds, 1 otherwise.
"""
import sys

from PIL import Image

DESCRIPTION, X_RESOLUTION, Y_RESOLUTION, RESOLUTION_UNIT, SOFTWARE = 270, 282, 283, 296, 305


def pages(path):
    """Yields each page's (size, mode, tags) in order."""
    with Image.open(path) as image:
        for i in range(getattr(image, "n_frames", 1)):
            image.seek(i)
            yield image.size, image.mode, dict(image.tag_v2)


def resolution(tags):
    """The page's pixel size as exact fractions, with TIFF's default unit (2, inch) when the tag is absent."""
    def ratio(tag):
        value = tags.get(tag)
        return None if value is None else (value.numerator, value.denominator)
    return ratio(X_RESOLUTION), ratio(Y_RESOLUTION), tags.get(RESOLUTION_UNIT, 2)


def main(input_path, output_path, expected_description=None):
    source, copy = list(pages(input_path)), list(pages(output_path))
    problems = []
    if len(copy) != len(source):
        problems.append(f"{len(copy)} pages, but the input has {len(source)}")
    for i, ((size, _, tags), (copy_size, copy_mode, copy_tags)) in enumerate(zip(source, copy)):
        if copy_size != size:
            problems.append(f"page {i}: size {copy_size}, the input's is {size}")
        if copy_mode != "F":
            problems.append(f"page {i}: mode {copy_mode}, expected F (float32)")
        if resolution(copy_tags) != resolution(tags):
            problems.append(f"page {i}: pixel size {resolution(copy_tags)}, the input's is {resolution(tags)}")
    if source and copy:
        want = expected_description if expected_description is not None else source[0][2].get(DESCRIPTION)
        got = copy[0][2].get(DESCRIPTION)
        if got != want:
            problems.append(f"ImageDescription {got!r}, expected {want!r}")
        software = copy[0][2].get(SOFTWARE, "")
        if not (software.startswith("DORA ") and "model sha256:" in software):
            problems.append(f"Software tag {software!r} doesn't say which DORA and model made it")
    for problem in problems:
        print(f"{output_path}: {problem}")
    if not problems:
        print(f"Pillow reads {output_path}: {len(copy)} pages, the expected description, the input's pixel size and DORA's Software tag")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))

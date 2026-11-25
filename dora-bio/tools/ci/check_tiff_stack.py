#!/usr/bin/env python3
"""Reads a multi-page TIFF written by DORA with Pillow, an independent reader, and checks every page.

    python3 -I tools/ci/check_tiff_stack.py STACK.tif PAGE_DIR

PAGE_DIR holds the same frames as single-page files, synthetic_000.tif, synthetic_001.tif, ... (as `dora-bio synth`
writes them). Pillow must find one page per file, of the same size and mode, with identical pixel values.
Exit 0 if all pages match, 1 otherwise.
"""
import sys
from pathlib import Path

from PIL import Image


def main(stack_path, page_dir):
    singles = sorted(Path(page_dir).glob("synthetic_*.tif"))
    problems = []
    with Image.open(stack_path) as stack:
        frames = getattr(stack, "n_frames", 1)
        if frames != len(singles):
            problems.append(f"Pillow sees {frames} pages, expected {len(singles)}")
        for i, single_path in enumerate(singles[:frames]):
            stack.seek(i)
            page = stack.copy()
            with Image.open(single_path) as single:
                single.load()
                if (page.size, page.mode) != (single.size, single.mode):
                    problems.append(f"page {i}: {page.size} {page.mode}, but {single_path.name} is {single.size} {single.mode}")
                elif page.tobytes() != single.tobytes():   # same mode, so identical bytes = identical values
                    problems.append(f"page {i}: pixel values differ from {single_path.name}")
    for p in problems:
        print(f"✗ {p}")
    if not problems:
        print(f"✓ Pillow reads all {len(singles)} pages of {Path(stack_path).name} exactly")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

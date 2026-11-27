"""Residual audit for unpaired data (the wild-data check before brick 41): what DORA removed, without ground truth.

    VENV/bin/python -I tools/experiments/wild_residual.py LABEL RAW.tif DENOISED.tif [RAW2 DENOISED2 ...] [--png OUT.png]

For each pair of stacks (or a folder of single images given as several pairs) the residual r = raw - denoised is pooled
over every plane, in the input's own units:
  - corr(r, denoised), signed: about 0 when DORA removed only zero-mean noise; positive when bright structure was
    shrunk (the residual then carries part of it); negative when the output overshoots bright structure.
    (Shot noise makes |r| grow with brightness, but not its sign, so the signed correlation stays near 0 for pure noise.)
  - lag-1 correlation of r along rows (x) and columns (y): near 0 for removed independent noise; positive when
    structure or correlated noise was removed.
  - net change: mean(r) / mean(raw); and the RMS of r as a fraction of the raw planes' 0.1-99.9 percentile range.
--png writes raw | denoised | residual of the first pair's middle plane: raw and denoised on one shared 0.1-99.9
percentile window, the residual on a symmetric window.
Needs numpy, tifffile and imagecodecs (tools/experiments/requirements-careamics.txt has the first two). Not gated: a
diagnostic whose numbers are recorded in docs/RESULTS.md with the command that made them.
"""
import sys

import numpy as np
import tifffile


def planes(path):
    a = tifffile.imread(path).astype(np.float64)
    return a.reshape((-1,) + a.shape[-2:])


def main(argv):
    png = None
    if "--png" in argv:
        i = argv.index("--png")
        png = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    label, paths = argv[0], argv[1:]
    if len(paths) < 2 or len(paths) % 2:
        sys.exit("give LABEL then RAW DENOISED pairs")
    sums = dict(n=0, rd=0.0, r=0.0, d=0.0, rr=0.0, dd=0.0, raw=0.0)
    lag = {"x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0]}   # sum r_i r_j, sum r_i^2, sum r_j^2 (residual mean removed per plane)
    rms, nplanes = [], 0
    for k in range(0, len(paths), 2):
        raw, den = planes(paths[k]), planes(paths[k + 1])
        if raw.shape != den.shape:
            sys.exit(f"{paths[k]} and {paths[k + 1]} differ in shape: {raw.shape} vs {den.shape}")
        for p in range(raw.shape[0]):
            x, d = raw[p], den[p]
            r = x - d
            sums["n"] += r.size
            sums["rd"] += float((r * d).sum()); sums["r"] += float(r.sum()); sums["d"] += float(d.sum())
            sums["rr"] += float((r * r).sum()); sums["dd"] += float((d * d).sum()); sums["raw"] += float(x.sum())
            rc = r - r.mean()
            for axis, (a, b) in {"x": (rc[:, :-1], rc[:, 1:]), "y": (rc[:-1, :], rc[1:, :])}.items():
                lag[axis][0] += float((a * b).sum()); lag[axis][1] += float((a * a).sum()); lag[axis][2] += float((b * b).sum())
            lo, hi = np.percentile(x, [0.1, 99.9])
            rms.append(float(np.sqrt((r * r).mean())) / max(hi - lo, 1e-12))
            nplanes += 1
        if png and k == 0:
            write_png(png, raw[raw.shape[0] // 2], den[den.shape[0] // 2])
    n = sums["n"]
    cov = sums["rd"] / n - (sums["r"] / n) * (sums["d"] / n)
    var_r = sums["rr"] / n - (sums["r"] / n) ** 2
    var_d = sums["dd"] / n - (sums["d"] / n) ** 2
    corr = cov / np.sqrt(max(var_r * var_d, 1e-300))
    lx, ly = (lag[a][0] / np.sqrt(max(lag[a][1] * lag[a][2], 1e-300)) for a in ("x", "y"))
    print(f"{label}: {nplanes} planes; corr(residual, output) {corr:+.3f}; residual lag-1 x {lx:+.3f}, y {ly:+.3f}; "
          f"net change {sums['r'] / max(sums['raw'], 1e-12):+.4f} of the mean; RMS change {np.mean(rms):.3f} of the range")


def write_png(path, raw, den):
    import imagecodecs
    lo, hi = np.percentile(raw, [0.1, 99.9])
    scale = lambda a: np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    r = raw - den
    s = max(float(np.percentile(np.abs(r), 99.5)), 1e-12)
    panels = [scale(raw), scale(den), np.clip(0.5 + r / (2 * s), 0, 1)]
    gap = np.ones((raw.shape[0], 4))
    image = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
    with open(path, "wb") as f:
        f.write(imagecodecs.png_encode((image * 255).round().astype(np.uint8)))


if __name__ == "__main__":
    main(sys.argv[1:])

"""Alignment sweep for a paired low/high-SNR time-lapse (before brick 40's pre-registration).

    python -I align_sweep.py LOW.tif HIGH.tif [--frames-per-block 8] [--blur 2]

For three blocks (early, middle, late) of the movie: blur both movies, then correlate low-SNR frame t with high-SNR
frame t + dt shifted by (dx, dy), over dt in -2..2 and dx, dy in -5..5, on the shared interior only (a 5 px border is
dropped from every frame, so no shift ever reads padding). Each block averages frames c .. c + n - 1. Reports the best
(dt, dx, dy), its correlation, the zero-offset correlation, a 50-frames-apart baseline, and the best neighbouring shift
(a near-tie with a neighbour means a sub-pixel offset that whole-pixel shifts can't remove).
"""
import argparse
import numpy as np
import tifffile

M = 5  # max |shift|


def blur(a, sigma):
    if sigma <= 0:
        return a
    r = int(3 * sigma)
    x = np.arange(-r, r + 1)
    k = np.exp(-x * x / (2 * sigma * sigma)); k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -2, a)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


class Pages:
    """Frame access through the TIFF's pages, for stacks tifffile can't map in one piece."""
    def __init__(self, tif):
        self.tif = tif
        first = tif.pages[0]
        self.shape = (len(tif.pages), *first.shape)
    def __getitem__(self, i):
        return self.tif.pages[int(i)].asarray()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("low"); ap.add_argument("high")
    ap.add_argument("--frames-per-block", type=int, default=8)
    ap.add_argument("--blur", type=float, default=2.0)
    ap.add_argument("--dynamic", type=int, default=0,
                    help="subtract each pixel's mean over this many frames around the block first, so only what changes in "
                         "time is compared (static anatomy correlates at any lag); 0 = raw frames")
    args = ap.parse_args()
    lt, ht = tifffile.TiffFile(args.low), tifffile.TiffFile(args.high)
    try:
        low, high = lt.series[0].asarray(out="memmap"), ht.series[0].asarray(out="memmap")
    except Exception:
        low, high = Pages(lt), Pages(ht)
    T, H, W = low.shape
    assert tuple(high.shape) == tuple(low.shape), (low.shape, high.shape)
    n = args.frames_per_block
    centres = [100, T // 2, T - 100 - n]
    print(f"{args.low.split('/')[-1]}: {T} frames of {W}x{H}, {n} frames per block, blur sigma {args.blur}")
    print(f"{'block':>6} {'best (dt,dx,dy)':>16} {'corr':>6} {'zero':>6} {'t+50':>6} {'2nd best shift':>16} {'corr':>6}")
    def window_mean(movie, c):
        if args.dynamic <= 0:
            return 0.0
        lo, hi = max(0, c - args.dynamic // 2), min(T, c + n + args.dynamic // 2)
        return np.mean([blur(np.asarray(movie[i], dtype=np.float64), args.blur) for i in range(lo, hi)], axis=0)

    if args.dynamic:
        print(f"dynamic: each pixel's mean over ~{args.dynamic} frames around the block subtracted from both movies")
    for c in centres:
        mean_low, mean_high = window_mean(low, c), window_mean(high, c)
        L = np.stack([blur(np.asarray(low[c + i], dtype=np.float64), args.blur) - mean_low for i in range(n)])
        inner = (slice(None), slice(M, H - M), slice(M, W - M))
        scores = {}
        for dt in range(-2, 3):
            Hs = np.stack([blur(np.asarray(high[c + dt + i], dtype=np.float64), args.blur) - mean_high for i in range(n)])
            for dy in range(-M, M + 1):
                for dx in range(-M, M + 1):
                    shifted = Hs[:, M + dy:H - M + dy, M + dx:W - M + dx]
                    scores[(dt, dx, dy)] = corr(L[inner], shifted)
        best = max(scores, key=scores.get)
        dt0, dx0, dy0 = best
        neighbours = {k: v for k, v in scores.items() if k[0] == dt0 and max(abs(k[1] - dx0), abs(k[2] - dy0)) == 1}
        second = max(neighbours, key=neighbours.get)
        far = np.stack([blur(np.asarray(high[min(T - 1, c + 50 + i)], dtype=np.float64), args.blur) - mean_high for i in range(n)])
        base = corr(L[inner], far[inner])
        print(f"{c:>6} {str(best):>16} {scores[best]:6.3f} {scores[(0, 0, 0)]:6.3f} {base:6.3f} {str(second):>16} {neighbours[second]:6.3f}")


if __name__ == "__main__":
    main()

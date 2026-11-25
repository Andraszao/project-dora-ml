using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>
/// Estimates an image's Poisson–Gaussian <see cref="NoiseModel"/> (variance = gain · intensity + intercept) from the
/// noisy image alone, in its own units, so that the <see cref="Anscombe"/> transform can be applied to it.
///
/// Method, after Foi et al. (IEEE TIP 2008), simplified: cut the image into 16×16 tiles and, for each, measure its mean
/// and the variance left after removing its best-fit plane (<see cref="NoiseDiagnostics"/> does the same to judge
/// correlation). Sort the tiles by mean into 12 bands of equal size. In each band, cell edges and other structure only
/// ever add variance, so take the band's lower quartile of tile variances, which ignores the busiest three quarters,
/// and divide out the quartile's known bias for pure noise. (A lower quantile can be asked for; see the parameter.) Fit a straight line variance = gain · mean + intercept
/// through the bands, drop bands far off it, and fit again.
/// </summary>
/// <remarks>
/// One image cannot separate the camera's offset from its read noise: both enter only the intercept, as
/// read² − gain · offset. The transform needs nothing more. Tiles that touch a clipped value (0, or the top of an 8-,
/// 12- or 16-bit range) are skipped, since clipping removes noise.
/// </remarks>
public static class NoiseCalibration
{
    /// <summary>Number of brightness bands the tiles are sorted into.</summary>
    public const int Bands = 12;

    /// <summary>Fewest usable tiles an estimate needs (3 per band).</summary>
    public const int MinTiles = 3 * Bands;

    /// <summary>Which quantile of each band's tile variances is trusted by default.</summary>
    public const double DefaultQuantile = 0.25;

    // Standard-normal quantiles for the supported choices (Wilson–Hilferty needs z for the bias correction below).
    private static readonly Dictionary<double, double> NormalQuantile = new() { [0.05] = -1.6449, [0.1] = -1.2816, [0.25] = -0.6745, [0.5] = 0 };

    /// <summary>Estimates the noise model of a noisy image in its original intensity units (not normalized).</summary>
    /// <param name="image">A noisy image; at least 36 usable 16×16 tiles (e.g. 96×96).</param>
    /// <param name="quantile">
    /// Which quantile of each band's tile variances to trust: 0.05, 0.1, 0.25 or 0.5. Lower ignores more structure
    /// (images whose bright regions are all cells) at the cost of more scatter. The bias it introduces for pure noise
    /// is divided out.
    /// </param>
    /// <exception cref="ArgumentException">Too few usable tiles, or an unsupported quantile.</exception>
    public static NoiseModel Estimate(GrayImage image, double quantile = DefaultQuantile)
    {
        if (!NormalQuantile.TryGetValue(quantile, out double z))
            throw new ArgumentException("Supported quantiles: 0.05, 0.1, 0.25, 0.5.", nameof(quantile));
        // That quantile of a chi-square variance estimate with k = 16² − 3 degrees of freedom, as a fraction of the true
        // variance (Wilson–Hilferty): (1 − 2/(9k) + z·√(2/(9k)))³, e.g. 0.939 for the lower quartile.
        double c = 2.0 / (9 * (NoiseDiagnostics.TileSize * NoiseDiagnostics.TileSize - 3));
        double bias = Math.Pow(1 - c + z * Math.Sqrt(c), 3);
        float top = image.Pixels.Max();
        bool clippedTop = top is 255f or 4095f or 65535f;
        const int t = NoiseDiagnostics.TileSize;
        const double dof = t * t - 3;   // the plane fit uses 3 degrees of freedom

        var tiles = new List<(double Mean, double Variance)>();
        for (int y0 = 0; y0 + t <= image.Height; y0 += t)
            for (int x0 = 0; x0 + t <= image.Width; x0 += t)
            {
                double mean = 0;
                bool clipped = false;
                for (int y = y0; y < y0 + t && !clipped; y++)
                    for (int x = x0; x < x0 + t; x++)
                    {
                        float v = image[x, y];
                        if (v <= 0 || (clippedTop && v >= top)) { clipped = true; break; }
                        mean += v;
                    }
                if (clipped) continue;
                var residual = NoiseDiagnostics.PlaneResidual(image, x0, y0);
                double sumSquares = 0;
                foreach (double r in residual) sumSquares += r * r;
                tiles.Add((mean / (t * t), sumSquares / dof));
            }
        if (tiles.Count < MinTiles)
            throw new ArgumentException($"Need at least {MinTiles} unclipped {t}×{t} tiles to estimate the noise; this image has {tiles.Count}.");

        // Equal-count brightness bands: (median mean, debiased lower-quartile variance) per band.
        tiles.Sort((a, b) => a.Mean.CompareTo(b.Mean));
        var bands = new List<(double Mean, double Variance)>();
        for (int b = 0; b < Bands; b++)
        {
            var band = tiles.Skip(b * tiles.Count / Bands).Take((b + 1) * tiles.Count / Bands - b * tiles.Count / Bands).ToList();
            var variances = band.Select(x => x.Variance).OrderBy(v => v).ToList();
            bands.Add((band[band.Count / 2].Mean, Quantile(variances, quantile) / bias));
        }

        var (gain, intercept) = FitLine(bands);
        // One pass of outlier rejection: bands whose structure survived the quartile pull the line up.
        var residuals = bands.Select(b => Math.Abs(b.Variance - (gain * b.Mean + intercept))).ToList();
        double cut = 3 * Math.Max(Quantile(residuals.OrderBy(r => r).ToList(), 0.5), 1e-12);
        var kept = bands.Where((b, i) => residuals[i] <= cut).ToList();
        if (kept.Count >= Bands / 2) (gain, intercept) = FitLine(kept);

        // No measurable signal dependence (pure Gaussian noise): a tiny gain turns the transform into a plain scaling.
        if (gain <= 0)
        {
            double variance = Quantile(bands.Select(b => b.Variance).OrderBy(v => v).ToList(), 0.5);
            return new NoiseModel(1e-6 * Math.Sqrt(variance), variance);
        }
        return new NoiseModel(gain, intercept);
    }

    /// <summary>Ordinary least squares for variance = gain · mean + intercept.</summary>
    private static (double Gain, double Intercept) FitLine(IReadOnlyList<(double Mean, double Variance)> points)
    {
        double mx = points.Average(p => p.Mean), my = points.Average(p => p.Variance), sxy = 0, sxx = 0;
        foreach (var (x, y) in points)
        {
            sxy += (x - mx) * (y - my);
            sxx += (x - mx) * (x - mx);
        }
        double gain = sxx > 0 ? sxy / sxx : 0;
        return (gain, my - gain * mx);
    }

    private static double Quantile(List<double> sorted, double q)
    {
        double position = q * (sorted.Count - 1);
        int low = (int)Math.Floor(position), high = Math.Min(low + 1, sorted.Count - 1);
        return sorted[low] + (position - low) * (sorted[high] - sorted[low]);
    }
}

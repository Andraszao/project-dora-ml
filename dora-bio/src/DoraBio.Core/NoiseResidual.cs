using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>How correlated a set's real noise is between neighbouring pixels, in one region of its images.</summary>
/// <param name="Region">"background", "interior", "edges" or "all".</param>
/// <param name="Pixels">Pixels in the region, over all pairs.</param>
/// <param name="Right1">Correlation with the pixel one to the right (along image rows).</param>
/// <param name="Down1">Correlation with the pixel one below (along image columns).</param>
/// <param name="Right2">Correlation with the pixel two to the right.</param>
/// <param name="Down2">Correlation with the pixel two below.</param>
public sealed record ResidualCorrelation(string Region, long Pixels, double Right1, double Down1, double Right2, double Down2);

/// <summary>
/// Measures a set's real noise correlation from its high-SNR references (added after brick 38, for its Tribolium
/// question; a diagnostic, not part of any pre-registered reading). Per pair the reference is matched to the noisy image
/// by least squares (m = a·reference + b, so exposure and offset differences cancel), and the residual
/// r = (noisy − m) / √max(m, ε), with ε = 1% of m's 99.9th percentile, evens out shot noise's intensity-dependent variance.
/// Correlations between r at a pixel and at its neighbours are pooled over all pairs and reported per region, where a pair
/// of pixels counts only if both lie in the region: <b>background</b> (m in the image's lowest 40%), <b>interior</b> (m in its
/// top 25%) and <b>edges</b> (the 20% of pixels with the steepest reference gradient, taken out of the other two). Detector or
/// scan correlation shows in the background, along one axis if it comes from scanning; a noisy image misaligned with its
/// reference leaves structure at the edges, along both axes.
/// </summary>
public static class NoiseResidual
{
    /// <summary>The regions reported, in order.</summary>
    public static readonly string[] Regions = ["background", "interior", "edges", "all"];

    private static readonly (int Dx, int Dy)[] Offsets = [(1, 0), (0, 1), (2, 0), (0, 2)];

    /// <summary>Pools the residual correlations of every pair (see the class remarks).</summary>
    /// <exception cref="ArgumentException">A pair's images differ in size.</exception>
    public static List<ResidualCorrelation> Measure(IEnumerable<(GrayImage Noisy, GrayImage Reference)> pairs)
    {
        // sums[region, offset] = (Σ ab, Σ a², Σ b²); pixels[region]
        var sums = new double[Regions.Length, Offsets.Length, 3];
        var pixels = new long[Regions.Length];
        foreach (var (noisy, reference) in pairs)
        {
            if (noisy.Width != reference.Width || noisy.Height != reference.Height)
                throw new ArgumentException("A noisy image and its reference differ in size.");
            var (residual, region) = Prepare(noisy, reference);
            int w = noisy.Width, h = noisy.Height;
            for (int i = 0; i < residual.Length; i++)
            {
                if (region[i] < Regions.Length - 1) pixels[region[i]]++;   // the middle band belongs to no named region
                pixels[Regions.Length - 1]++;
            }
            for (int o = 0; o < Offsets.Length; o++)
            {
                var (dx, dy) = Offsets[o];
                for (int y = 0; y + dy < h; y++)
                    for (int x = 0; x + dx < w; x++)
                    {
                        int i = y * w + x, j = (y + dy) * w + x + dx;
                        double a = residual[i], b = residual[j];
                        Add(sums, Regions.Length - 1, o, a, b);
                        if (region[i] == region[j] && region[i] < Regions.Length - 1) Add(sums, region[i], o, a, b);
                    }
            }
        }
        var result = new List<ResidualCorrelation>();
        for (int r = 0; r < Regions.Length; r++)
        {
            double C(int o) => sums[r, o, 1] > 0 && sums[r, o, 2] > 0 ? sums[r, o, 0] / Math.Sqrt(sums[r, o, 1] * sums[r, o, 2]) : double.NaN;
            result.Add(new ResidualCorrelation(Regions[r], pixels[r], C(0), C(1), C(2), C(3)));
        }
        return result;
    }

    private static void Add(double[,,] sums, int region, int offset, double a, double b)
    {
        sums[region, offset, 0] += a * b;
        sums[region, offset, 1] += a * a;
        sums[region, offset, 2] += b * b;
    }

    /// <summary>The normalized, mean-free residual and each pixel's region (index into <see cref="Regions"/>; "all" = none).</summary>
    internal static (double[] Residual, int[] Region) Prepare(GrayImage noisy, GrayImage reference)
    {
        int n = noisy.Length, w = noisy.Width, h = noisy.Height;
        double mr = 0, mn = 0;
        for (int i = 0; i < n; i++) { mr += reference.Pixels[i]; mn += noisy.Pixels[i]; }
        mr /= n; mn /= n;
        double cov = 0, var = 0;
        for (int i = 0; i < n; i++)
        {
            double dr = reference.Pixels[i] - mr;
            cov += dr * (noisy.Pixels[i] - mn);
            var += dr * dr;
        }
        double a = var > 0 ? cov / var : 0, b = mn - a * mr;
        var m = new double[n];
        for (int i = 0; i < n; i++) m[i] = a * reference.Pixels[i] + b;
        double eps = Math.Max(1e-6, 0.01 * Quantile(m, 0.999));
        var residual = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++) { residual[i] = (noisy.Pixels[i] - m[i]) / Math.Sqrt(Math.Max(m[i], eps)); mean += residual[i]; }
        mean /= n;
        for (int i = 0; i < n; i++) residual[i] -= mean;

        var gradient = new double[n];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double gx = m[y * w + Math.Min(x + 1, w - 1)] - m[y * w + Math.Max(x - 1, 0)];
                double gy = m[Math.Min(y + 1, h - 1) * w + x] - m[Math.Max(y - 1, 0) * w + x];
                gradient[y * w + x] = Math.Sqrt(gx * gx + gy * gy);
            }
        double low = Quantile(m, 0.40), high = Quantile(m, 0.75), steep = Quantile(gradient, 0.80);
        var region = new int[n];
        for (int i = 0; i < n; i++)
            region[i] = gradient[i] >= steep && gradient[i] > 0 ? 2 : m[i] <= low ? 0 : m[i] >= high ? 1 : Regions.Length - 1;
        return (residual, region);
    }

    private static double Quantile(double[] values, double q)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        return sorted[(int)Math.Clamp(Math.Round(q * (sorted.Length - 1)), 0, sorted.Length - 1)];
    }
}

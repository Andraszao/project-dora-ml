using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>
/// Fidelity of one denoised image against its high-SNR reference, in the terms scientists measure (brick 32): are
/// regional intensities still proportional, and were spots invented or erased?
/// </summary>
/// <param name="Slope">Proportionality: k in output ∝ reference^k, fitted over the foreground tiles of both images with
/// their backgrounds subtracted. 1 means every ratio between two regions is preserved; below 1 the output compresses
/// contrast (dim regions brightened relative to bright ones), above 1 it exaggerates it. A square-root distortion gives
/// 0.5, a square 2.</param>
/// <param name="RatioError">The median, over foreground tiles, of |output / (g · reference) − 1|, where g is the overall
/// gain (the geometric mean of the tile ratios): the typical error of a regional intensity measurement, as a fraction.</param>
/// <param name="SpotsFound">Recall: the fraction of the reference's spots that the output has within 2 px. Low = erased.</param>
/// <param name="SpotsReal">Precision: the fraction of the output's spots that the reference has within 2 px. Low = invented
/// (or noise that looks like spots).</param>
/// <param name="ForegroundTiles">Tiles the intensity measures were computed over.</param>
/// <param name="ReferenceSpots">Spots detected in the reference.</param>
/// <param name="OutputSpots">Spots detected in the output.</param>
/// <param name="MatchedSpots">Output spots paired with a reference spot.</param>
public sealed record FidelityScore(double Slope, double RatioError, double SpotsFound, double SpotsReal,
    int ForegroundTiles, int ReferenceSpots, int OutputSpots, int MatchedSpots);

/// <summary>
/// Measurements of whether denoising distorts what scientists quantify.
/// <para><b>Intensities</b> are what a biologist measures with regions of interest: background-subtracted regional means,
/// compared as ratios. So both images are background-subtracted and scaled by their own 0.1st and 99.9th percentiles
/// (the same normalization as the reference in <see cref="ExtBenchmark"/>), and compared over 16 px tiles whose
/// reference mean is at least 0.1 (the foreground). A power law output ∝ reference^k is fitted to the tile means in log
/// space. A gain cancels out, so only a non-linear distortion, such as a denoiser that flattens bright structures more
/// than dim ones, moves k away from 1 or adds ratio error. (Matching by least squares with an offset, as for PSNR,
/// would hide the direction of a distortion: a square-root and a square both come out near 0.9.)</para>
/// <para><b>Spots</b> (puncta, foci, small vesicles) are blobs of 2–4 px. Here the output is matched to the reference by
/// least squares as for PSNR, so one threshold serves both. Spots are strict local maxima, in a 5×5 window, of a
/// difference of Gaussians (σ = 1 minus σ = 2), above five robust standard deviations (1.4826 × MAD) over the median of
/// the reference's response. Output spots are paired greedily, strongest first, with the nearest unpaired reference spot
/// within 2 px.</para>
/// </summary>
public static class Fidelity
{
    /// <summary>Side of the tiles intensities are measured on, in pixels.</summary>
    public const int TileSize = 16;

    /// <summary>Tiles whose reference mean (on the 0–1 reference scale) is at least this count as foreground.</summary>
    public const double ForegroundLevel = 0.1;

    /// <summary>Maximum distance, in pixels, between an output spot and the reference spot it is paired with.</summary>
    public const double MatchRadius = 2.0;

    /// <summary>Measures <paramref name="output"/> (any units) against <paramref name="clean"/> (any units).</summary>
    /// <exception cref="ArgumentException">The images differ in size.</exception>
    public static FidelityScore Measure(GrayImage output, GrayImage clean)
    {
        var reference = ExtBenchmark.NormalizeReference(clean);
        var (slope, ratio, tiles) = Intensities(ExtBenchmark.NormalizeReference(output), reference);
        var matched = ExtBenchmark.MatchToReference(output, reference);
        var dogReference = DifferenceOfGaussians(reference);
        double threshold = RobustThreshold(dogReference.Pixels);
        var refSpots = Spots(dogReference, threshold);
        var outSpots = Spots(DifferenceOfGaussians(matched), threshold);
        int pairs = Pair(outSpots, refSpots);
        return new FidelityScore(slope, ratio,
            refSpots.Count == 0 ? double.NaN : pairs / (double)refSpots.Count,
            outSpots.Count == 0 ? double.NaN : pairs / (double)outSpots.Count,
            tiles, refSpots.Count, outSpots.Count, pairs);
    }

    /// <summary>
    /// Pools per-image scores into one per dataset: spot counts and tile statistics are summed over images (so larger
    /// images weigh more, and an image with few spots doesn't swing the result), slope and ratio error averaged.
    /// </summary>
    public static FidelityScore Pool(IReadOnlyList<FidelityScore> scores)
    {
        int refs = scores.Sum(s => s.ReferenceSpots), outs = scores.Sum(s => s.OutputSpots), pairs = scores.Sum(s => s.MatchedSpots);
        static double Mean(IEnumerable<double> values) { var v = values.Where(double.IsFinite).ToList(); return v.Count == 0 ? double.NaN : v.Average(); }
        return new FidelityScore(Mean(scores.Select(s => s.Slope)), Mean(scores.Select(s => s.RatioError)),
            refs == 0 ? double.NaN : pairs / (double)refs, outs == 0 ? double.NaN : pairs / (double)outs,
            scores.Sum(s => s.ForegroundTiles), refs, outs, pairs);
    }

    private static (double Slope, double RatioError, int Tiles) Intensities(GrayImage output, GrayImage reference)
    {
        var logs = new List<(double R, double O)>();
        for (int ty = 0; ty + TileSize <= reference.Height; ty += TileSize)
            for (int tx = 0; tx + TileSize <= reference.Width; tx += TileSize)
            {
                double r = 0, o = 0;
                for (int y = ty; y < ty + TileSize; y++)
                    for (int x = tx; x < tx + TileSize; x++)
                    {
                        r += reference[x, y];
                        o += output[x, y];
                    }
                r /= TileSize * TileSize;
                o /= TileSize * TileSize;
                if (r < ForegroundLevel) continue;
                logs.Add((Math.Log(r), Math.Log(Math.Max(o, 1e-3))));
            }
        if (logs.Count < 2) return (double.NaN, double.NaN, logs.Count);
        double mr = logs.Average(p => p.R), mo = logs.Average(p => p.O), cov = 0, var = 0;
        foreach (var (r, o) in logs) { cov += (r - mr) * (o - mo); var += (r - mr) * (r - mr); }
        // Gain g = geometric mean of output/reference; the ratio error is what's left after it.
        double logGain = mo - mr;
        var errors = logs.Select(p => Math.Abs(Math.Exp(p.O - p.R - logGain) - 1)).Order().ToList();
        return (var > 0 ? cov / var : double.NaN, errors[errors.Count / 2], logs.Count);
    }

    /// <summary>Gaussian blur σ = 1 minus σ = 2: responds to blobs 2–4 px across, and not to flat regions or slow gradients.</summary>
    internal static GrayImage DifferenceOfGaussians(GrayImage image)
    {
        var fine = Blur(image, 1.0);
        var coarse = Blur(image, 2.0);
        var result = new float[image.Length];
        for (int i = 0; i < result.Length; i++) result[i] = fine.Pixels[i] - coarse.Pixels[i];
        return new GrayImage(image.Width, image.Height, result);
    }

    /// <summary>Separable Gaussian blur (radius 3σ), with edges extended by repeating the border pixel.</summary>
    internal static GrayImage Blur(GrayImage image, double sigma)
    {
        int radius = (int)Math.Ceiling(3 * sigma), w = image.Width, h = image.Height;
        var kernel = new double[2 * radius + 1];
        double total = 0;
        for (int k = -radius; k <= radius; k++) total += kernel[k + radius] = Math.Exp(-k * k / (2 * sigma * sigma));
        for (int k = 0; k < kernel.Length; k++) kernel[k] /= total;
        var rows = new float[image.Length];
        var result = new float[image.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double acc = 0;
                for (int k = -radius; k <= radius; k++) acc += kernel[k + radius] * image[Math.Clamp(x + k, 0, w - 1), y];
                rows[y * w + x] = (float)acc;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double acc = 0;
                for (int k = -radius; k <= radius; k++) acc += kernel[k + radius] * rows[Math.Clamp(y + k, 0, h - 1) * w + x];
                result[y * w + x] = (float)acc;
            }
        return new GrayImage(w, h, result);
    }

    /// <summary>Median plus five robust standard deviations (1.4826 × the median absolute deviation).</summary>
    internal static double RobustThreshold(ReadOnlySpan<float> values)
    {
        var sorted = values.ToArray();
        Array.Sort(sorted);
        double median = sorted[sorted.Length / 2];
        var deviations = sorted.Select(v => Math.Abs(v - median)).ToArray();
        Array.Sort(deviations);
        return median + 5 * 1.4826 * deviations[deviations.Length / 2];
    }

    /// <summary>Strict local maxima in a 5×5 window above the threshold, strongest first.</summary>
    internal static List<(int X, int Y, float Response)> Spots(GrayImage dog, double threshold)
    {
        var spots = new List<(int, int, float)>();
        for (int y = 0; y < dog.Height; y++)
            for (int x = 0; x < dog.Width; x++)
            {
                float v = dog[x, y];
                if (v <= threshold) continue;
                bool peak = true;
                for (int dy = -2; dy <= 2 && peak; dy++)
                    for (int dx = -2; dx <= 2; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= dog.Width || ny >= dog.Height) continue;
                        if (dog[nx, ny] >= v) { peak = false; break; }
                    }
                if (peak) spots.Add((x, y, v));
            }
        spots.Sort((a, b) => b.Item3.CompareTo(a.Item3));
        return spots;
    }

    /// <summary>Greedy pairing, strongest output spot first, with the nearest unpaired reference spot within the radius.</summary>
    private static int Pair(List<(int X, int Y, float Response)> output, List<(int X, int Y, float Response)> reference)
    {
        var used = new bool[reference.Count];
        int pairs = 0;
        foreach (var o in output)
        {
            int best = -1;
            double bestDistance = MatchRadius * MatchRadius + 1e-9;
            for (int i = 0; i < reference.Count; i++)
            {
                if (used[i]) continue;
                double dx = o.X - reference[i].X, dy = o.Y - reference[i].Y, d = dx * dx + dy * dy;
                if (d <= bestDistance) { bestDistance = d; best = i; }
            }
            if (best < 0) continue;
            used[best] = true;
            pairs++;
        }
        return pairs;
    }
}

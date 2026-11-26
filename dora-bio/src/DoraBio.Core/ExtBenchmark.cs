using DoraBio.Core.Data;
using DoraBio.Core.IO;

namespace DoraBio.Core;

/// <summary>Scores for one external test image: the noisy input, the bilateral filter and the method under test.</summary>
/// <param name="Name">The image's file name.</param>
/// <param name="NoisyPsnr">PSNR (dB) of the brightness-matched noisy input.</param>
/// <param name="NoisySsim">SSIM of the brightness-matched noisy input.</param>
/// <param name="BilateralPsnr">PSNR of the brightness-matched bilateral filter.</param>
/// <param name="BilateralSsim">SSIM of the brightness-matched bilateral filter.</param>
/// <param name="MethodPsnr">PSNR of the brightness-matched method output.</param>
/// <param name="MethodSsim">SSIM of the brightness-matched method output.</param>
public sealed record ExtScore(string Name, double NoisyPsnr, double NoisySsim, double BilateralPsnr, double BilateralSsim,
    double MethodPsnr, double MethodSsim);

/// <summary>
/// Scores denoisers on the external datasets' test images with the protocol of the CARE paper (Weigert et al., Nature
/// Methods 2018) and its CSBDeep code, so DORA and CARE or Noise2Void are compared exactly as their authors compare
/// methods (quality pass Q18).
/// <para>The high-SNR reference was taken with more light, so it is 3–11× brighter than the noisy input, and every
/// method outputs a different brightness scale. A raw PSNR would mostly measure that offset. Instead:</para>
/// <code>
/// reference r  = (clean − p0.1) / (p99.9 − p0.1)          its own 0.1st and 99.9th percentiles
/// output x     → a·x + b, with a, b = argmin Σ (a·x + b − r)²     (least squares: CSBDeep's normalize_minmse)
///                a = cov(x, r) / var(x),  b = mean(r) − a·mean(x)
/// score        = PSNR(a·x + b, r) with data range 1, and SSIM likewise
/// </code>
/// <para>The noisy input and the bilateral filter go through the same match, so every column is on equal terms: a
/// method can't gain by being brighter, only by being closer in shape.</para>
/// </summary>
public static class ExtBenchmark
{
    /// <summary>The reference scaled to [0, 1] by its own 0.1st and 99.9th percentiles (values outside are kept).</summary>
    public static GrayImage NormalizeReference(GrayImage clean) =>
        Normalization.Apply(clean, Normalization.FromPercentiles(clean.Pixels, Normalization.DefaultLowPercentile, Normalization.DefaultHighPercentile));

    /// <summary>The least-squares affine match a·x + b of <paramref name="output"/> to <paramref name="reference"/>.</summary>
    /// <exception cref="ArgumentException">The images differ in size.</exception>
    public static GrayImage MatchToReference(GrayImage output, GrayImage reference)
    {
        if (output.Width != reference.Width || output.Height != reference.Height)
            throw new ArgumentException("Output and reference differ in size.");
        int n = output.Length;
        double mx = 0, mr = 0;
        for (int i = 0; i < n; i++) { mx += output.Pixels[i]; mr += reference.Pixels[i]; }
        mx /= n;
        mr /= n;
        double cov = 0, var = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = output.Pixels[i] - mx;
            cov += dx * (reference.Pixels[i] - mr);
            var += dx * dx;
        }
        double a = var > 0 ? cov / var : 0, b = mr - a * mx;
        var matched = new float[n];
        for (int i = 0; i < n; i++) matched[i] = (float)(a * output.Pixels[i] + b);
        return new GrayImage(output.Width, output.Height, matched);
    }

    /// <summary>PSNR (data range 1) and SSIM of an output against a normalized reference, after the affine match.</summary>
    public static (double Psnr, double Ssim) Score(GrayImage output, GrayImage normalizedReference)
    {
        var matched = MatchToReference(output, normalizedReference);
        return (Metrics.Psnr(matched.Pixels, normalizedReference.Pixels), Metrics.Ssim(matched, normalizedReference));
    }

    /// <summary>
    /// Scores <paramref name="method"/> on every test image of a paired dataset, alongside the noisy input and the
    /// bilateral filter.
    /// </summary>
    /// <param name="root">The data/ext folder.</param>
    /// <param name="dataset">A dataset with ground truth.</param>
    /// <param name="method">Given an image's file name and its noisy pixels (original units), returns the output in any
    /// units (the match removes scale and offset).</param>
    public static List<ExtScore> Evaluate(string root, ExtDataset dataset, Func<string, GrayImage, GrayImage> method) =>
        Evaluate(ExtDatasets.PlanPairs(root, dataset, ExtSplit.Test), method);

    /// <summary>The same protocol on any planned pairs, e.g. <see cref="ExtDatasets.ValidationPlan"/>.</summary>
    public static List<ExtScore> Evaluate(IReadOnlyList<FmdDataset.PlannedPair> plan, Func<string, GrayImage, GrayImage> method)
    {
        var scores = new List<ExtScore>();
        foreach (var p in plan)
        {
            var noisy = ImageLoader.Load(p.NoisyPath);
            var reference = NormalizeReference(ImageLoader.Load(p.CleanPath));
            var input = Normalization.Apply(noisy, Normalization.FromPercentiles(noisy.Pixels));
            var (np, ns) = Score(noisy, reference);
            var (bp, bs) = Score(BilateralBaseline.FilterAuto(input), reference);
            var (mp, ms) = Score(method(Path.GetFileName(p.NoisyPath), noisy), reference);
            scores.Add(new ExtScore(Path.GetFileName(p.NoisyPath), np, ns, bp, bs, mp, ms));
        }
        return scores;
    }
}

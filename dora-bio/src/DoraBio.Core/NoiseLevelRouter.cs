using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>
/// Picks one of two models per image by how noisy the image is (quality pass Q10): a model trained on the noisiest
/// frames for noisy images, and a model trained on every noise level for clean ones. The measure is Immerkær's noise σ
/// (<see cref="Metrics.EstimateNoiseSigma"/>) of the image as the network sees it, after percentile normalization, so
/// it is relative to the image's own intensity range and needs no calibration.
/// </summary>
public static class NoiseLevelRouter
{
    /// <summary>Noise σ of an image in original units, measured after the same normalization DORA applies at inference.</summary>
    public static double Sigma(GrayImage noisy) =>
        Metrics.EstimateNoiseSigma(Normalization.Apply(noisy, Normalization.FromPercentiles(noisy.Pixels)));

    /// <summary>
    /// The threshold that maximizes the mean PSNR when images with σ above it go to model A and the rest to model B.
    /// Candidates are the midpoints between consecutive σ values (plus "everything to A" and "everything to B"), so
    /// the result is exact for the given images. Ties go to the lower threshold (more images to A).
    /// </summary>
    /// <param name="images">Per image: its σ, and the PSNR model A and model B reach on it.</param>
    /// <returns>The threshold and the mean PSNR it gives.</returns>
    public static (double Threshold, double MeanPsnr) ChooseThreshold(IReadOnlyList<(double Sigma, double PsnrA, double PsnrB)> images)
    {
        if (images.Count == 0) throw new ArgumentException("Need at least one image.", nameof(images));
        var sorted = images.OrderBy(i => i.Sigma).ToList();
        // Threshold below every σ: all images to A. Moving it up past image k sends image k to B.
        double total = sorted.Sum(i => i.PsnrA), best = total, bestThreshold = sorted[0].Sigma - 1e-9;
        for (int k = 0; k < sorted.Count; k++)
        {
            total += sorted[k].PsnrB - sorted[k].PsnrA;
            double threshold = k + 1 < sorted.Count ? 0.5 * (sorted[k].Sigma + sorted[k + 1].Sigma) : sorted[k].Sigma + 1e-9;
            if (total > best + 1e-12)
            {
                best = total;
                bestThreshold = threshold;
            }
        }
        return (bestThreshold, best / sorted.Count);
    }
}

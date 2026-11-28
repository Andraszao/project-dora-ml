using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>
/// Classical edge-preserving bilateral filter. It's shown next to DORA's output so users can see what a
/// learned denoiser adds over a well-known hand-designed one. It is not a training target.
/// </summary>
public static class BilateralBaseline
{
    /// <summary>
    /// Replaces each pixel with a weighted mean of its neighbors. The weights fall off with distance
    /// (spatial term) and with intensity difference (range term):
    /// <c>w = exp(−d²/2σs²) · exp(−(v − center)²/2σr²)</c>.
    /// Across an edge the intensities differ a lot, so the far side gets almost no weight and the edge stays sharp.
    /// </summary>
    /// <param name="image">Image to filter (unchanged); typically normalized.</param>
    /// <param name="sigmaSpatial">Gaussian falloff with distance, in pixels.</param>
    /// <param name="sigmaRange">Gaussian falloff with intensity difference, in the image's own units.</param>
    /// <remarks>The window radius is ⌈2σs⌉. Borders are handled by clamping coordinates (edge pixels repeat).</remarks>
    public static GrayImage Filter(GrayImage image, float sigmaSpatial, float sigmaRange)
    {
        if (sigmaSpatial <= 0 || sigmaRange <= 0) throw new ArgumentOutOfRangeException(nameof(sigmaSpatial));
        int radius = Math.Max(1, (int)MathF.Ceiling(2f * sigmaSpatial));
        int w = image.Width, h = image.Height;
        var src = image.Pixels;
        var result = new GrayImage(w, h);
        var dst = result.Pixels;

        int span = 2 * radius + 1;
        var spatial = new float[span * span];
        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++)
                spatial[(dy + radius) * span + dx + radius] = MathF.Exp(-(dx * dx + dy * dy) / (2f * sigmaSpatial * sigmaSpatial));
        float rangeScale = -1f / (2f * sigmaRange * sigmaRange);

        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                float center = src[y * w + x];
                float weightSum = 0f, valueSum = 0f;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int sy = Math.Clamp(y + dy, 0, h - 1);
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int sx = Math.Clamp(x + dx, 0, w - 1);
                        float v = src[sy * w + sx];
                        float diff = v - center;
                        float weight = spatial[(dy + radius) * span + dx + radius] * MathF.Exp(diff * diff * rangeScale);
                        weightSum += weight;
                        valueSum += weight * v;
                    }
                }
                dst[y * w + x] = valueSum / weightSum;
            }
        });
        return result;
    }

    /// <summary>Picks the range sigma from the estimated noise level (2σ), a common rule of thumb.</summary>
    public static GrayImage FilterAuto(GrayImage image, float sigmaSpatial = 1.5f)
    {
        float noise = (float)Metrics.EstimateNoiseSigma(image);
        var (min, max) = image.Range();
        float sigmaRange = Math.Max(2f * noise, 1e-4f * Math.Max(1e-6f, max - min));
        return Filter(image, sigmaSpatial, sigmaRange);
    }
}

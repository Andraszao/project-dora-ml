namespace DoraBio.Core.Data;

/// <summary>Affine map from sensor units to normalized intensity: (v − Low) / (High − Low).</summary>
/// <param name="Low">Sensor value that maps to 0.</param>
/// <param name="High">Sensor value that maps to 1. Must differ from <paramref name="Low"/>.</param>
/// <remarks>
/// Keep the map next to the image: the CLI uses <see cref="Invert"/> to write denoised output back in the
/// original intensity units, so downstream quantification still works on real numbers.
/// </remarks>
public readonly record struct IntensityMap(float Low, float High)
{
    /// <summary>Width of the mapped range in sensor units (High − Low).</summary>
    public float Scale => High - Low;

    /// <summary>Sensor units → normalized (Low → 0, High → 1; values outside the range are not clamped).</summary>
    public float Apply(float value) => (value - Low) / Scale;

    /// <summary>Normalized → sensor units; the exact inverse of <see cref="Apply"/>.</summary>
    public float Invert(float normalized) => normalized * Scale + Low;
}

/// <summary>
/// Percentile normalization for microscopy data. Unlike v1's 8-bit pipeline, nothing is quantized,
/// and by default nothing is clamped either: clipping the noise tails would bias a denoiser
/// toward darker or brighter values.
/// </summary>
/// <remarks>
/// Why percentiles rather than min/max? A single hot pixel or cosmic-ray hit would set the max and squash
/// everything else into a sliver of the range. The 0.1% and 99.9% quantiles ignore such outliers.
/// </remarks>
public static class Normalization
{
    /// <summary>Quantile mapped to 0 by default (0.1%).</summary>
    public const double DefaultLowPercentile = 0.001;

    /// <summary>Quantile mapped to 1 by default (99.9%).</summary>
    public const double DefaultHighPercentile = 0.999;

    /// <summary>
    /// Maps the <paramref name="lowFraction"/> and <paramref name="highFraction"/> quantiles to 0 and 1.
    /// A flat image (High == Low) gets a unit-width range, so the map never divides by zero.
    /// </summary>
    public static IntensityMap FromPercentiles(
        ReadOnlySpan<float> values,
        double lowFraction = DefaultLowPercentile,
        double highFraction = DefaultHighPercentile)
    {
        if (values.IsEmpty) throw new ArgumentException("No values to normalize.", nameof(values));
        if (!(lowFraction >= 0 && lowFraction < highFraction && highFraction <= 1))
            throw new ArgumentOutOfRangeException(nameof(lowFraction), "Need 0 ≤ low < high ≤ 1.");

        float[] sorted = values.ToArray();
        Array.Sort(sorted);
        float low = Quantile(sorted, lowFraction);
        float high = Quantile(sorted, highFraction);
        if (high - low < 1e-6f) high = low + 1f;
        return new IntensityMap(low, high);
    }

    /// <summary>Linear-interpolated quantile of an already sorted array (the "type 7" definition NumPy uses).</summary>
    public static float Quantile(ReadOnlySpan<float> sorted, double fraction)
    {
        if (sorted.IsEmpty) throw new ArgumentException("Empty input.", nameof(sorted));
        double position = fraction * (sorted.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        double t = position - lower;
        return (float)(sorted[lower] + (sorted[upper] - sorted[lower]) * t);
    }

    /// <summary>Returns a new image with <paramref name="map"/> applied to every pixel.</summary>
    /// <param name="image">Source image (unchanged).</param>
    /// <param name="map">The mapping to apply.</param>
    /// <param name="clamp">Clamp results to [0, 1]. Off by default; only useful for display, never for training.</param>
    public static GrayImage Apply(GrayImage image, IntensityMap map, bool clamp = false)
    {
        var result = new GrayImage(image.Width, image.Height);
        var src = image.Pixels;
        var dst = result.Pixels;
        for (int i = 0; i < src.Length; i++)
        {
            float v = map.Apply(src[i]);
            dst[i] = clamp ? Math.Clamp(v, 0f, 1f) : v;
        }
        return result;
    }

    /// <summary>Returns a new image mapped back from normalized intensities to sensor units.</summary>
    public static GrayImage Invert(GrayImage normalized, IntensityMap map)
    {
        var result = new GrayImage(normalized.Width, normalized.Height);
        for (int i = 0; i < normalized.Length; i++)
            result.Pixels[i] = map.Invert(normalized.Pixels[i]);
        return result;
    }
}

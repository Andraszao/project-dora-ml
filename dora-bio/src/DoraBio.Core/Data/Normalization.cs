namespace DoraBio.Core.Data;

/// <summary>Affine map from sensor units to normalized intensity: (v − Low) / (High − Low).</summary>
public readonly record struct IntensityMap(float Low, float High)
{
    public float Scale => High - Low;
    public float Apply(float value) => (value - Low) / Scale;
    public float Invert(float normalized) => normalized * Scale + Low;
}

/// <summary>
/// Percentile normalization for microscopy data. Unlike v1's 8-bit pipeline, nothing is quantized,
/// and by default nothing is clamped either: clipping the noise tails would bias a denoiser
/// toward darker or brighter values.
/// </summary>
public static class Normalization
{
    public const double DefaultLowPercentile = 0.001;
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

    public static GrayImage Invert(GrayImage normalized, IntensityMap map)
    {
        var result = new GrayImage(normalized.Width, normalized.Height);
        for (int i = 0; i < normalized.Length; i++)
            result.Pixels[i] = map.Invert(normalized.Pixels[i]);
        return result;
    }
}

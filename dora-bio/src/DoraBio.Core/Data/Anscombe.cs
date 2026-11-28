namespace DoraBio.Core.Data;

/// <summary>
/// Poisson–Gaussian noise model in camera units: the noise variance at mean intensity m is Gain · m + Intercept.
/// </summary>
/// <remarks>
/// A camera records ADU = offset + gain · Poisson(λ) + Normal(0, read²), whose mean is m = offset + gain · λ and whose
/// variance is gain² · λ + read² = gain · (m − offset) + read². So Intercept = read² − gain · offset (plus 1/12 for
/// rounding to whole ADU). One image reveals Gain and Intercept but can't separate offset from read noise, and the
/// variance-stabilizing transform needs only the two it reveals.
/// </remarks>
/// <param name="Gain">ADU per photon: how fast the noise variance grows with intensity. Must be positive.</param>
/// <param name="Intercept">Variance extrapolated to intensity 0 (often negative: the camera offset).</param>
public sealed record NoiseModel(double Gain, double Intercept)
{
    /// <summary>Noise variance at mean intensity <paramref name="mean"/>.</summary>
    public double Variance(double mean) => Gain * mean + Intercept;

    /// <summary>The model of a camera with known settings, e.g. <see cref="SyntheticOptions"/>.</summary>
    /// <param name="gain">ADU per photon.</param>
    /// <param name="offset">Dark offset in ADU.</param>
    /// <param name="readNoise">Read-noise standard deviation in ADU.</param>
    /// <param name="rounded">Whether the camera rounds to whole ADU (adds 1/12 ADU² of variance).</param>
    public static NoiseModel FromCamera(double gain, double offset, double readNoise, bool rounded = true) =>
        new(gain, readNoise * readNoise - gain * offset + (rounded ? 1.0 / 12 : 0));
}

/// <summary>
/// The generalized Anscombe transform (GAT): turns Poisson–Gaussian noise, whose variance grows with intensity, into
/// noise of roughly unit variance everywhere, and back.
///
/// Forward: f(z) = (2 / g) · √(g·z + ⅜·g² + b), for a <see cref="NoiseModel"/> with gain g and intercept b (0 where the
/// root's argument is negative). A first-order expansion gives Var f(z) ≈ f′(m)² · Var z = (g·m + b)/(g·m + ⅜g² + b) ≈ 1.
///
/// Inverse: denoising in the transformed domain estimates E[f(z)], not f(E[z]). Because f is concave they differ (by
/// Jensen's inequality), so the algebraic inverse f⁻¹ is biased low at low photon counts. The exact unbiased inverse
/// maps E[f(z)] back to E[z]; Mäkitalo and Foi (IEEE TIP 2011, 2013) give a closed-form approximation of it, used here.
/// </summary>
public static class Anscombe
{
    private static readonly double Sqrt32 = Math.Sqrt(1.5);

    /// <summary>f at zero photons: below it, the unbiased inverse falls back to the algebraic one (they meet there).</summary>
    public static readonly double ZeroPhotonValue = 2 * Math.Sqrt(0.375);

    /// <summary>The forward transform of one value.</summary>
    public static double Forward(double z, NoiseModel model)
    {
        double g = model.Gain;
        return 2 / g * Math.Sqrt(Math.Max(0, g * z + 0.375 * g * g + model.Intercept));
    }

    /// <summary>f⁻¹: exact for values whose root argument was non-negative (round trip ≤ 1e-4 relative).</summary>
    public static double InverseAlgebraic(double d, NoiseModel model) =>
        model.Gain * (d * d / 4 - 0.375) - model.Intercept / model.Gain;

    /// <summary>
    /// Maps an estimate of E[f(z)] back to E[z] in camera units. In units of photons (gain 1), the Mäkitalo–Foi closed
    /// form is ¼D² + ¼√(3/2)·D⁻¹ − (11/8)·D⁻² + (5/8)·√(3/2)·D⁻³ − ⅛ − σ̃², which this applies via the model's gain and
    /// intercept. Below <see cref="ZeroPhotonValue"/> the series is not valid; the algebraic inverse takes over there, and
    /// the two agree exactly at that point, so the result is continuous.
    /// </summary>
    public static double InverseUnbiased(double d, NoiseModel model)
    {
        if (d < ZeroPhotonValue) return InverseAlgebraic(d, model);
        double photons = d * d / 4 + Sqrt32 / (4 * d) - 11 / (8 * d * d) + 5 * Sqrt32 / (8 * d * d * d) - 0.125;
        return model.Gain * photons - model.Intercept / model.Gain;
    }

    /// <summary>The forward transform of a whole image.</summary>
    public static GrayImage Forward(GrayImage image, NoiseModel model)
    {
        var result = new GrayImage(image.Width, image.Height);
        for (int i = 0; i < image.Length; i++) result.Pixels[i] = (float)Forward(image.Pixels[i], model);
        return result;
    }

    /// <summary>The unbiased inverse of a whole image (for a denoised image in the transformed domain).</summary>
    public static GrayImage InverseUnbiased(GrayImage image, NoiseModel model)
    {
        var result = new GrayImage(image.Width, image.Height);
        for (int i = 0; i < image.Length; i++) result.Pixels[i] = (float)InverseUnbiased(image.Pixels[i], model);
        return result;
    }
}

using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>
/// The condition a noise-conditioned network (<see cref="Nn.NetConditioning.NoiseLevel"/>, quality pass Q13) is given
/// for each frame: s = ln(σ̂ / 0.05), where σ̂ is Immerkær's noise σ (<see cref="Metrics.EstimateNoiseSigma"/>) of the
/// whole normalized frame the network sees. The log makes equal ratios of noise equal steps (0 at σ̂ = 0.05, about ±1.4
/// for 4× more or less noise), which suits a small network's input.
///
/// <para>It is always measured on a whole frame, never on a tile or a training patch: <see cref="TiledInference"/>
/// measures it once and gives every tile the same value (so tiled output still equals whole-image output), and training
/// measures each training image once and gives all its patches that value. So the network learns the estimator's
/// errors along with its signal, exactly as it will meet them.</para>
/// </summary>
public static class NoiseConditioning
{
    /// <summary>The σ̂ at which s = 0.</summary>
    public const double ReferenceSigma = 0.05;

    /// <summary>s for a whole normalized frame.</summary>
    public static float Of(GrayImage normalizedFrame) => FromSigma(Metrics.EstimateNoiseSigma(normalizedFrame));

    /// <summary>s for a given σ̂ (floored at 1e-4, so a constant image gives a finite value).</summary>
    public static float FromSigma(double sigma) => (float)Math.Log(Math.Max(sigma, 1e-4) / ReferenceSigma);
}

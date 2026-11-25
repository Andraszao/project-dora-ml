using DoraBio.Core.Data;
using Xunit;
using Xunit.Abstractions;

namespace DoraBio.Core.Tests;

/// <summary>
/// The generalized Anscombe transform, on Poisson–Gaussian samples drawn exactly as <see cref="SyntheticFluorescence"/>
/// draws them (ADU = offset + gain · Poisson(λ) + Normal(0, read²), rounded), with known camera settings.
/// </summary>
[Trait(Bricks.Trait, Bricks.Vst)]
public class VstTests
{
    private readonly ITestOutputHelper _output;
    public VstTests(ITestOutputHelper output) => _output = output;

    private const double Gain = 1.5, Offset = 100, Read = 3;
    private static readonly NoiseModel Camera = NoiseModel.FromCamera(Gain, Offset, Read);

    /// <summary>Knuth's Poisson sampler (exact; fine for the λ ≤ 200 used here) plus Box–Muller read noise.</summary>
    private static double[] Samples(double lambda, int count, int seed)
    {
        var rng = new Random(seed);
        var result = new double[count];
        double limit = Math.Exp(-lambda);
        for (int i = 0; i < count; i++)
        {
            int k = 0;
            for (double p = rng.NextDouble(); p > limit; p *= rng.NextDouble()) k++;
            double gauss = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            result[i] = Math.Round(Offset + Gain * k + Read * gauss);
        }
        return result;
    }

    [Fact]
    public void TheAlgebraicInverseUndoesTheForwardTransform()
    {
        foreach (double z in new[] { 95.0, 100, 101.5, 130, 1000, 40000 })
        {
            double back = Anscombe.InverseAlgebraic(Anscombe.Forward(z, Camera), Camera);
            Assert.True(Math.Abs(back - z) <= 1e-4 * Math.Max(1, Math.Abs(z)), $"{z} → {back}");
        }
    }

    [Theory]
    [InlineData(5, 0.85, 1.10)]
    [InlineData(10, 0.92, 1.08)]
    [InlineData(25, 0.95, 1.05)]
    [InlineData(100, 0.95, 1.05)]
    [InlineData(200, 0.95, 1.05)]
    public void TheTransformedNoiseHasUnitVarianceAtEveryBrightness(double photons, double low, double high)
    {
        // Untransformed, the variance here ranges from about 20 to 450 ADU². Transformed, it should be about 1 at all of them.
        var f = Samples(photons, 40_000, seed: (int)photons).Select(z => Anscombe.Forward(z, Camera)).ToArray();
        double mean = f.Average(), variance = f.Sum(v => (v - mean) * (v - mean)) / (f.Length - 1);
        _output.WriteLine($"λ = {photons}: raw variance {Camera.Variance(Offset + Gain * photons):0.0} ADU², transformed {variance:0.000}");
        Assert.InRange(variance, low, high);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(20)]
    public void TheUnbiasedInverseRecoversTheMeanWhereTheAlgebraicOneIsBiased(double photons)
    {
        // A perfect denoiser in the transformed domain outputs E[f(z)]. Mapping that back must give E[z].
        double truth = Offset + Gain * photons;
        double expected = Samples(photons, 200_000, seed: 100 + (int)photons).Average(z => Anscombe.Forward(z, Camera));
        double unbiased = Anscombe.InverseUnbiased(expected, Camera);
        double algebraic = Anscombe.InverseAlgebraic(expected, Camera);
        _output.WriteLine($"λ = {photons}: truth {truth:0.000}, unbiased {unbiased:0.000}, algebraic {algebraic:0.000}");
        Assert.InRange(unbiased - truth, -0.05 * Gain * photons - 0.05, 0.05 * Gain * photons + 0.05);
        Assert.True(Math.Abs(unbiased - truth) < Math.Abs(algebraic - truth), "the unbiased inverse must beat the algebraic one");
    }

    [Fact]
    public void TheUnbiasedInverseIsContinuousAtZeroPhotons()
    {
        double d = Anscombe.ZeroPhotonValue;
        Assert.Equal(Anscombe.InverseAlgebraic(d, Camera), Anscombe.InverseUnbiased(d, Camera), 9);
        Assert.Equal(Anscombe.InverseUnbiased(d - 1e-9, Camera), Anscombe.InverseUnbiased(d + 1e-9, Camera), 6);
    }

    [Fact]
    public void ImagesRoundTripThroughTheTransform()
    {
        var frame = SyntheticFluorescence.Generate(64, 64, seed: 3,
            new SyntheticOptions { GainAduPerPhoton = Gain, CameraOffsetAdu = Offset, ReadNoiseAdu = Read });
        var model = NoiseModel.FromCamera(Gain, Offset, Read);
        var forward = Anscombe.Forward(frame.Clean, model);
        for (int i = 0; i < forward.Length; i++)
            Assert.Equal(frame.Clean.Pixels[i], Anscombe.InverseAlgebraic(forward.Pixels[i], model), 2);
        var back = Anscombe.InverseUnbiased(forward, model);
        Assert.Equal(frame.Clean.Width, back.Width);
    }
}

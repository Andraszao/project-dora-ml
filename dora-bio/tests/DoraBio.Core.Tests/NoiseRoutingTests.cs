using DoraBio.Core.Data;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>Noise-level routing: the threshold search is exact, and the noise measure tracks how noisy a frame is.</summary>
[Trait(Bricks.Trait, Bricks.NoiseRouting)]
public class NoiseRoutingTests
{
    [Fact]
    public void TheChosenThresholdMaximizesTheRoutedMeanExactly()
    {
        // A wins on the noisy images (σ ≥ 0.3), B on the clean ones: the best split is between 0.2 and 0.3.
        var images = new List<(double, double, double)> { (0.05, 30, 33), (0.1, 31, 32), (0.2, 30, 31), (0.3, 29, 27), (0.5, 26, 24) };
        var (threshold, mean) = NoiseLevelRouter.ChooseThreshold(images);
        Assert.InRange(threshold, 0.2, 0.3);
        Assert.Equal((33 + 32 + 31 + 29 + 26) / 5.0, mean, 9);

        // Brute force over every split agrees.
        double best = double.MinValue;
        foreach (double t in new[] { -1.0, 0.07, 0.15, 0.25, 0.4, 1.0 })
            best = Math.Max(best, images.Average(i => i.Item1 > t ? i.Item2 : i.Item3));
        Assert.Equal(best, mean, 9);
    }

    [Fact]
    public void EverythingCanGoToOneModel()
    {
        var aAlways = new List<(double, double, double)> { (0.1, 30, 29), (0.2, 31, 30) };
        var (low, meanA) = NoiseLevelRouter.ChooseThreshold(aAlways);
        Assert.True(low < 0.1);
        Assert.Equal(30.5, meanA, 9);
        var bAlways = new List<(double, double, double)> { (0.1, 29, 30), (0.2, 30, 31) };
        var (high, meanB) = NoiseLevelRouter.ChooseThreshold(bAlways);
        Assert.True(high > 0.2);
        Assert.Equal(30.5, meanB, 9);
        Assert.Throws<ArgumentException>(() => NoiseLevelRouter.ChooseThreshold([]));
    }

    [Fact]
    public void MorePhotonsMeanALowerNoiseSigma()
    {
        // Same scene, 5 → 80 photons at peak: relative noise must fall steadily.
        var sigmas = new[] { 5.0, 20, 80 }
            .Select(peak => NoiseLevelRouter.Sigma(SyntheticFluorescence.Generate(128, 128, seed: 3, new SyntheticOptions { PeakPhotons = peak }).Noisy))
            .ToList();
        Assert.True(sigmas[0] > sigmas[1] && sigmas[1] > sigmas[2], string.Join(", ", sigmas));
    }
}

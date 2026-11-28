using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Quality pass Q19 (brick 37): the routing threshold is kept or re-chosen on validation images by a fixed rule (the
/// owner's first guard), before any test image is scored; and, since Q20 adopted the design, routed Noise2Void training
/// (<see cref="RoutedNoise2Void"/>) splits images by the noise check and trains the right network for each kind. The
/// recorded verdicts are checked by gate 37.
/// </summary>
[Trait(Bricks.Trait, Bricks.Routing)]
public class RoutingTests
{
    [Fact]
    public void AThresholdThatSeparatesEveryImageIsKept()
    {
        var samples = new[] { (0.26, false), (0.16, false), (0.56, true), (0.57, true) };
        var (threshold, reason) = NoiseDiagnostics.ChooseThreshold(samples);
        Assert.Equal(0.4, threshold);
        Assert.StartsWith("kept", reason);
    }

    [Fact]
    public void AMisroutingThresholdMovesToTheMiddleOfTheValidationGap()
    {
        var samples = new[] { (0.10, false), (0.44, false), (0.50, true), (0.60, true) };   // 0.4 sends 0.44 the wrong way
        var (threshold, reason) = NoiseDiagnostics.ChooseThreshold(samples);
        Assert.Equal(0.47, threshold, 9);
        Assert.StartsWith("moved", reason);
    }

    [Fact]
    public void WithoutAGapTheFewestMisroutesWinAndTiesGoToTheNearestValue()
    {
        // 0.30 (correlated) sits below 0.35 (independent): no threshold separates them, every choice misroutes one.
        var samples = new[] { (0.10, false), (0.35, false), (0.30, true), (0.70, true) };
        var (threshold, _) = NoiseDiagnostics.ChooseThreshold(samples);
        Assert.Equal(1, samples.Count(s => (s.Item1 >= threshold) != s.Item2));
        Assert.Equal(0.4, threshold);                                // 0.4 also misroutes just one, and it is the nearest
        var worse = new[] { (0.10, false), (0.20, true), (0.45, false), (0.50, false), (0.80, true) };
        var (moved, _) = NoiseDiagnostics.ChooseThreshold(worse);    // 0.4 misroutes 3 (0.20, 0.45, 0.50); 0.65 misroutes 1
        Assert.Equal(0.65, moved, 9);
        Assert.Throws<ArgumentException>(() => NoiseDiagnostics.ChooseThreshold(Array.Empty<(double, bool)>()));
    }

    [Fact]
    public void ImagesSplitByVerdictAndUnjudgedOnesCountAsIndependent()
    {
        var images = Enumerable.Range(0, 4).Select(i => new ImagePair(new GrayImage(4, 4), null, $"g{i}", $"i{i}")).ToList();
        var (independent, correlated) = RoutedNoise2Void.Split(images, new bool?[] { true, false, null, true });
        Assert.Equal(["i1", "i2"], independent.Select(p => p.Name));
        Assert.Equal(["i0", "i3"], correlated.Select(p => p.Name));
        Assert.Throws<ArgumentException>(() => RoutedNoise2Void.Split(images, new bool?[] { true }));
        Assert.Equal(Path.Combine("runs", "w.correlated.bin"), RoutedNoise2Void.CorrelatedPath(Path.Combine("runs", "w.bin")));
    }

    [Fact]
    public void TheRecipesAreQ19sAdoptedNetworks()
    {
        var baseConfig = new TrainerConfig { MaxEpochs = 30, Seed = 2026 };
        var independent = RoutedNoise2Void.Independent(baseConfig);
        Assert.Equal(NetConfig.UNet(16, 2), independent.Network);
        Assert.Equal((ValidationSignal.Loss, int.MaxValue, true), (independent.Trainer.Selection, independent.Trainer.EarlyStopPatience, independent.Trainer.KeepLastEpoch));
        Assert.Null(independent.Footprint);
        var correlated = RoutedNoise2Void.Correlated(baseConfig);
        Assert.Equal(NetConfig.Default, correlated.Network);
        Assert.Equal((ValidationSignal.Auto, 6, false), (correlated.Trainer.Selection, correlated.Trainer.EarlyStopPatience, correlated.Trainer.KeepLastEpoch));
        Assert.Equal(StructuredMasking.DefaultFootprint, correlated.Footprint);
        Assert.All(new[] { independent, correlated }, r => Assert.Equal((TrainingMode.Noise2Void, 30, 2026), (r.Trainer.Mode, r.Trainer.MaxEpochs, r.Trainer.Seed)));
    }

    [Fact]
    public void AMixedSetTrainsOneNetworkPerKindOfNoise()
    {
        var frame = NoiseDiagnosticsTests.Frame;
        var correlatedImage = NoiseDiagnosticsTests.SmoothedNoise(alongY: true);
        var raw = new List<ImagePair>
        {
            new(frame.Noisy, null, "a", "independent-a"), new(frame.Noisy.Crop(0, 0, 192, 192), null, "b", "independent-b"),
            new(correlatedImage, null, "c", "correlated-c"), new(correlatedImage.Crop(32, 32, 192, 192), null, "d", "correlated-d"),
        };
        var images = raw.Select(p => p.Normalize().Pair).ToList();
        string dir = TestUtil.TempDirectory(), outPath = Path.Combine(dir, "w.bin");
        var config = new TrainerConfig { MaxEpochs = 1, StepsPerEpoch = 1, BatchSize = 2, PatchSize = 32, ValidationPatches = 2, Seed = 3 };
        var trained = RoutedNoise2Void.Train(images, config, outPath, "test data", minimumImages: 1);   // 2 + 2: below the default minimum
        Assert.Equal([NoiseKind.Independent, NoiseKind.Correlated], trained.Select(t => t.Kind));
        Assert.Equal([2, 2], trained.Select(t => t.Images));
        var (unet, unetInfo) = WeightsFile.Load(outPath);
        var (dilated, dilatedInfo) = WeightsFile.Load(RoutedNoise2Void.CorrelatedPath(outPath));
        Assert.Equal((NetArchitecture.UNet, NetArchitecture.Dilated), (unet.Config.Architecture, dilated.Config.Architecture));
        Assert.Contains("says independent", unetInfo.Provenance);
        Assert.Contains("structured blind spots", dilatedInfo.Provenance);

        var onlyIndependent = RoutedNoise2Void.Train(images.Take(2).ToList(), config, Path.Combine(dir, "single.bin"), "test data");
        Assert.Equal(Path.Combine(dir, "single.bin"), Assert.Single(onlyIndependent).WeightsPath);   // one kind: one file
    }
}

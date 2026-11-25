using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>The test_mix benchmark's protocol, checked on the miniature FMD tree (no real data needed).</summary>
[Trait(Bricks.Trait, Bricks.FmdEval)]
public class FmdBenchmarkTests : IDisposable
{
    private readonly string _root = TestUtil.TempDirectory();

    public FmdBenchmarkTests() => FmdTestTree.Build(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    /// <summary>A network whose weights are all zero predicts zero noise, so its output is its input.</summary>
    private static DenoisingNet IdentityNet()
    {
        var config = new NetConfig(4, [1, 1]);
        return DenoisingNet.FromParameters(config, new float[DenoisingNet.CountParameters(config)]);
    }

    [Fact]
    public void AnIdentityModelScoresExactlyLikeTheNoisyInput()
    {
        // Normalize → identity → invert → clip must round-trip 8-bit data, so "do nothing" equals the noisy baseline.
        var scores = FmdBenchmark.Evaluate(IdentityNet(), Path.Combine(_root, "test_mix"), "raw");
        Assert.Equal(4, scores.Count);
        Assert.All(scores, s =>
        {
            Assert.Equal(s.NoisyPsnr, s.DoraPsnr, 3);
            Assert.Equal(s.NoisySsim, s.DoraSsim, 5);
            Assert.True(double.IsFinite(s.NoisyPsnr));
        });
    }

    [Fact]
    public void EvaluatesEveryNoiseLevelFolder()
    {
        // (The miniature tree's levels share one noise pattern, so this checks plumbing, not that avg16 is cleaner.)
        foreach (string level in DoraBio.Core.Data.FmdDataset.Levels)
        {
            var summary = FmdBenchmark.Summarize(FmdBenchmark.Evaluate(IdentityNet(), Path.Combine(_root, "test_mix"), level));
            Assert.Equal(4, summary[^1].Count);
            Assert.Equal(summary[^1].Mean.NoisyPsnr, summary[^1].Mean.DoraPsnr, 3);
        }
    }

    /// <summary>A flat image with independent uniform noise (±10), plus the same noise averaged over two horizontal neighbours.</summary>
    private static (GrayImage Clean, GrayImage Independent, GrayImage Smeared) NoisyImages(int size = 192)
    {
        var rng = new Random(7);
        var noise = new float[size * (size + 1)];
        for (int i = 0; i < noise.Length; i++) noise[i] = (float)(rng.NextDouble() * 20 - 10);
        var clean = new GrayImage(size, size);
        var independent = new GrayImage(size, size);
        var smeared = new GrayImage(size, size);
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                clean[x, y] = 100;
                independent[x, y] = 100 + noise[y * size + x];
                smeared[x, y] = 100 + 0.5f * (noise[y * size + x] + noise[y * size + x + 1]);
            }
        return (clean, independent, smeared);
    }

    [Fact]
    public void NoiseCorrelationIsZeroForIndependentNoiseAndAHalfForTwoTapAveraging()
    {
        var (clean, independent, smeared) = NoisyImages();
        Assert.InRange(FmdBenchmark.NoiseCorrelation(independent, clean, 1, 0), -0.03, 0.03);
        Assert.InRange(FmdBenchmark.NoiseCorrelation(independent, clean, 0, 1), -0.03, 0.03);

        // n'(x) = (n(x) + n(x+1)) / 2 shares one of its two terms with n'(x+1): ρ = 0.5. Nothing is shared vertically
        // or two pixels apart.
        Assert.InRange(FmdBenchmark.NoiseCorrelation(smeared, clean, 1, 0), 0.47, 0.53);
        Assert.InRange(FmdBenchmark.NoiseCorrelation(smeared, clean, 0, 1), -0.03, 0.03);
        Assert.InRange(FmdBenchmark.NoiseCorrelation(smeared, clean, 2, 0), -0.03, 0.03);

        // The image content cancels: only noisy − clean is measured, so structured images measure the same.
        var structured = new GrayImage(clean.Width, clean.Height);
        var structuredNoisy = new GrayImage(clean.Width, clean.Height);
        for (int i = 0; i < clean.Length; i++)
        {
            structured.Pixels[i] = i % 97;
            structuredNoisy.Pixels[i] = smeared.Pixels[i] - 100 + i % 97;
        }
        Assert.Equal(FmdBenchmark.NoiseCorrelation(smeared, clean, 1, 0),
            FmdBenchmark.NoiseCorrelation(structuredNoisy, structured, 1, 0), 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => FmdBenchmark.NoiseCorrelation(smeared, clean, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => FmdBenchmark.NoiseCorrelation(smeared, clean, -1, 0));
        Assert.Throws<ArgumentException>(() => FmdBenchmark.NoiseCorrelation(smeared, new GrayImage(4, 4), 1, 0));
    }

    [Fact]
    public void MeasuresNoiseOfEveryTestImageAndGroupsLikeTheScores()
    {
        var stats = FmdBenchmark.MeasureNoise(Path.Combine(_root, "test_mix"), "raw");
        Assert.Equal(4, stats.Count);
        Assert.All(stats, s =>
        {
            Assert.True(s.NoiseSd > 0);
            Assert.InRange(s.Right1, -1, 1);
            Assert.InRange(s.Down1, -1, 1);
        });
        var summary = FmdBenchmark.SummarizeNoise(stats);
        Assert.Equal(["Confocal", "WideField", "all"], summary.Select(s => s.Group));
        Assert.Equal([2, 2, 4], summary.Select(s => s.Count));
        Assert.Equal(stats.Average(s => s.Right1), summary[^1].Mean.Right1, 10);
    }

    [Fact]
    public void ARoutedEvaluationPicksAModelPerImage()
    {
        // Routing to the same model for every image must score exactly like the single-model call.
        int calls = 0;
        var net = IdentityNet();
        var routed = FmdBenchmark.Evaluate(noisy => { calls++; Assert.True(noisy.Length > 0); return net; },
            Path.Combine(_root, "test_mix"), "raw");
        var single = FmdBenchmark.Evaluate(net, Path.Combine(_root, "test_mix"), "raw");
        Assert.Equal(4, calls);
        Assert.Equal(single, routed);
    }

    [Fact]
    public void SummaryGroupsByModalityThenAll()
    {
        var summary = FmdBenchmark.Summarize(FmdBenchmark.Evaluate(IdentityNet(), Path.Combine(_root, "test_mix"), "raw"));
        Assert.Equal(["Confocal", "WideField", "all"], summary.Select(s => s.Group));
        Assert.Equal([2, 2, 4], summary.Select(s => s.Count));
        Assert.Equal("TwoPhoton", FmdBenchmark.ModalityOf("TwoPhoton_MICE"));
    }
}

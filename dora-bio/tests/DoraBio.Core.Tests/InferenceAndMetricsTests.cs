using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

[Trait(Bricks.Trait, Bricks.Inference)]
public class InferenceAndMetricsTests
{
    [Theory]
    [InlineData(100, 77, 32)]   // ragged edge tiles in both directions
    [InlineData(64, 64, 16)]    // tile core smaller than the receptive radius
    [InlineData(20, 9, 128)]    // single tile larger than the image
    public void TiledInferenceMatchesWholeImage(int width, int height, int tileCore)
    {
        var net = DenoisingNet.Create(new NetConfig(6, [1, 3, 2, 1]), seed: 2);
        var rng = new Random(1);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.3f;
        var image = TestUtil.RandomImage(width, height, 3);

        var whole = new float[image.Length];
        net.Predict(image.Pixels, height, width, whole, net.CreateWorkspace());
        var tiled = TiledInference.Denoise(net, image, tileCore);

        for (int i = 0; i < whole.Length; i++)
            Assert.True(Math.Abs(whole[i] - tiled.Pixels[i]) <= 1e-5f, $"pixel {i}: {whole[i]} vs {tiled.Pixels[i]}");
    }

    [Fact]
    public void PsnrKnownValues()
    {
        var a = new float[100];
        var b = Enumerable.Repeat(0.1f, 100).ToArray();
        Assert.Equal(20.0, Metrics.Psnr(a, b), 4);                      // MSE 0.01, range 1 → 20 dB
        Assert.Equal(double.PositiveInfinity, Metrics.Psnr(a, a));
        Assert.Equal(20.0 + 20 * Math.Log10(2), Metrics.Psnr(a, b, dataRange: 2), 4);
    }

    [Fact]
    public void SsimIsOneForIdenticalAndDropsWithNoise()
    {
        var clean = TestUtil.Pattern(48, 40, (x, y) => (float)(0.5 + 0.4 * Math.Sin(x * 0.3) * Math.Cos(y * 0.2)));
        Assert.Equal(1.0, Metrics.Ssim(clean, clean), 9);

        double light = Metrics.Ssim(TestUtil.AddGaussianNoise(clean, 0.02, 1), clean);
        double heavy = Metrics.Ssim(TestUtil.AddGaussianNoise(clean, 0.2, 1), clean);
        Assert.True(light > heavy, $"{light} should exceed {heavy}");
        Assert.InRange(light, 0.5, 1.0);
    }

    [Fact]
    public void NoiseEstimateRecoversWhiteNoiseLevel()
    {
        var smooth = TestUtil.Pattern(256, 256, (x, y) => (float)(0.3 + 0.2 * Math.Sin(x * 0.05) + 0.1 * Math.Cos(y * 0.03)));
        foreach (double sigma in new[] { 0.01, 0.05, 0.1 })
        {
            double estimate = Metrics.EstimateNoiseSigma(TestUtil.AddGaussianNoise(smooth, sigma, 7));
            Assert.InRange(estimate / sigma, 0.9, 1.1);
        }
    }

    [Fact]
    public void BilateralKeepsFlatRegionsFlatAndReducesNoise()
    {
        var flat = TestUtil.Pattern(64, 64, (_, _) => 0.4f);
        Assert.All(BilateralBaseline.Filter(flat, 1.5f, 0.1f).Pixels, v => Assert.Equal(0.4f, v, 5));

        var noisy = TestUtil.AddGaussianNoise(flat, 0.05, 3);
        var filtered = BilateralBaseline.FilterAuto(noisy);
        Assert.True(TestUtil.StdDev(filtered.Pixels) < 0.5 * TestUtil.StdDev(noisy.Pixels));
    }

    [Fact]
    public void BilateralPreservesAStepEdge()
    {
        var step = TestUtil.Pattern(32, 32, (x, _) => x < 16 ? 0.1f : 0.9f);
        var filtered = BilateralBaseline.Filter(step, 2f, 0.05f);
        Assert.Equal(0.1f, filtered[15, 10], 2);
        Assert.Equal(0.9f, filtered[16, 10], 2);
    }
}

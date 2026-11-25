using DoraBio.Core.Data;
using Xunit;
using Xunit.Abstractions;

namespace DoraBio.Core.Tests;

/// <summary>
/// Noise-model estimation from one image, on synthetic frames from cameras with known settings. Pass: gain within 10%,
/// and the predicted noise variance within 10% at the 10th, 50th and 90th percentiles of the frame's intensities.
/// (Offset and read noise can't be separated from one image; only their combination, the intercept, is checked, via
/// the variance curve.)
/// </summary>
[Trait(Bricks.Trait, Bricks.Calibration)]
public class CalibrationTests
{
    private readonly ITestOutputHelper _output;
    public CalibrationTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(1.0, 100, 2.0, 25, 1)]     // the generator's defaults
    [InlineData(1.0, 100, 2.0, 25, 2)]
    [InlineData(2.5, 500, 5.0, 25, 1)]     // high gain, big offset
    [InlineData(0.5, 0, 1.5, 100, 1)]      // sub-unity gain, no offset, bright
    [InlineData(4.0, 1000, 10.0, 10, 2)]   // dim and noisy
    [InlineData(1.0, 100, 2.0, 200, 2)]    // very bright
    public void RecoversTheCamerasNoiseModelFromOneFrame(double gain, double offset, double read, double peak, int seed)
    {
        var frame = SyntheticFluorescence.Generate(512, 512, seed,
            new SyntheticOptions { GainAduPerPhoton = gain, CameraOffsetAdu = offset, ReadNoiseAdu = read, PeakPhotons = peak });
        var truth = NoiseModel.FromCamera(gain, offset, read);
        var estimate = NoiseCalibration.Estimate(frame.Noisy);

        Assert.InRange(estimate.Gain / gain, 0.9, 1.1);
        var sorted = frame.Clean.Pixels.OrderBy(v => v).ToArray();
        foreach (double q in new[] { 0.1, 0.5, 0.9 })
        {
            double m = sorted[(int)(q * (sorted.Length - 1))];
            double ratio = estimate.Variance(m) / truth.Variance(m);
            _output.WriteLine($"p{q * 100:0}: intensity {m:0.0}, variance true {truth.Variance(m):0.00}, estimated {estimate.Variance(m):0.00}");
            Assert.InRange(ratio, 0.9, 1.1);
        }
    }

    [Fact]
    public void ScalesWithTheIntensityUnits()
    {
        // Variance = g·m + b in ADU; in units c times larger it must be (c·g)·m′ + c²·b.
        var frame = SyntheticFluorescence.Generate(256, 256, seed: 4);
        var scaled = new GrayImage(256, 256);
        for (int i = 0; i < scaled.Length; i++) scaled.Pixels[i] = frame.Noisy.Pixels[i] * 3f;
        var a = NoiseCalibration.Estimate(frame.Noisy);
        var b = NoiseCalibration.Estimate(scaled);
        Assert.Equal(3 * a.Gain, b.Gain, 6);
        Assert.Equal(9 * a.Intercept, b.Intercept, 3);
    }

    [Fact]
    public void SaturatedTilesAreIgnored()
    {
        // Clipping at the top of a 16-bit range removes noise; those tiles must not drag the estimate down.
        var frame = SyntheticFluorescence.Generate(512, 512, seed: 5);
        var saturated = frame.Noisy.Clone();
        for (int y = 0; y < 160; y++)
            for (int x = 0; x < 512; x++)
                saturated[x, y] = 65535f;
        var clean = NoiseCalibration.Estimate(frame.Noisy);
        var withClipping = NoiseCalibration.Estimate(saturated);
        Assert.InRange(withClipping.Gain / clean.Gain, 0.85, 1.15);
    }

    [Fact]
    public void PureGaussianNoiseGivesAFlatModel()
    {
        var rng = new Random(6);
        var image = new GrayImage(256, 256);
        for (int i = 0; i < image.Length; i++)
            image.Pixels[i] = (float)(500 + 200 * (i % 256) / 256.0 + 4 * Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        var model = NoiseCalibration.Estimate(image);
        // The variance is 16 everywhere: whatever the fit, it must predict about 16 across the intensity range.
        foreach (double m in new[] { 520.0, 600, 680 }) Assert.InRange(model.Variance(m), 14.4, 17.6);
    }

    [Fact]
    public void RefusesImagesTooSmallToCalibrate()
    {
        Assert.Throws<ArgumentException>(() => NoiseCalibration.Estimate(new GrayImage(64, 64)));
    }
}

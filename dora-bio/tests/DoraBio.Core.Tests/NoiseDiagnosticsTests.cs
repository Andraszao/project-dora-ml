using DoraBio.Core.Data;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// The blind noise-correlation check, on synthetic fluorescence frames whose noise structure is known exactly:
/// the generator's own pixel-independent Poisson–Gaussian noise, and that same noise smoothed with a [1, 2, 1] / 4
/// kernel, whose true one-pixel correlation is (2·1 + 1·2) / (1 + 4 + 1) = 0.667 (FMD widefield measures 0.6).
/// </summary>
[Trait(Bricks.Trait, Bricks.NoiseDiagnostic)]
public class NoiseDiagnosticsTests
{
    private static readonly SyntheticFluorescence.Frame Frame = SyntheticFluorescence.Generate(256, 256, seed: 11);

    /// <summary>The frame's clean image plus its own noise smoothed along x (and, if asked, along y).</summary>
    private static GrayImage SmoothedNoise(bool alongY)
    {
        var (noisy, clean) = (Frame.Noisy, Frame.Clean);
        int w = noisy.Width, h = noisy.Height;
        var noise = new float[noisy.Length];
        for (int i = 0; i < noise.Length; i++) noise[i] = noisy.Pixels[i] - clean.Pixels[i];
        float At(float[] n, int x, int y) => n[Math.Clamp(y, 0, h - 1) * w + Math.Clamp(x, 0, w - 1)];

        var smoothed = new float[noise.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                smoothed[y * w + x] = 0.25f * At(noise, x - 1, y) + 0.5f * At(noise, x, y) + 0.25f * At(noise, x + 1, y);
        if (alongY)
        {
            var copy = (float[])smoothed.Clone();
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    smoothed[y * w + x] = 0.25f * At(copy, x, y - 1) + 0.5f * At(copy, x, y) + 0.25f * At(copy, x, y + 1);
        }

        var result = new GrayImage(w, h);
        for (int i = 0; i < result.Length; i++) result.Pixels[i] = clean.Pixels[i] + smoothed[i];
        return result;
    }

    [Fact]
    public void IndependentNoiseIsNotFlagged()
    {
        var d = NoiseDiagnostics.Diagnose(Frame.Noisy);
        Assert.False(d.Correlated, d.ToString());
        Assert.InRange(d.Right1, -0.15, 0.15);
        Assert.InRange(d.Down1, -0.15, 0.15);
        Assert.InRange(d.Diagonal1, -0.15, 0.15);
        Assert.Equal(256 / 16 * (256 / 16) / 4, d.TilesUsed);   // the flattest quarter of 256 tiles
    }

    [Fact]
    public void SmoothedNoiseIsFlaggedAndItsCorrelationRecovered()
    {
        var d = NoiseDiagnostics.Diagnose(SmoothedNoise(alongY: true));
        Assert.True(d.Correlated, d.ToString());
        // Truth: 0.667 at one pixel, 0.667² = 0.44 diagonally, 0.167 at two pixels. Small tiles bias the estimate low.
        Assert.InRange(d.Right1, 0.5, 0.75);
        Assert.InRange(d.Down1, 0.5, 0.75);
        Assert.InRange(d.Diagonal1, 0.25, 0.5);
        Assert.InRange(d.Right2, 0.0, 0.25);
        Assert.Contains("CORRELATED", d.ToString());
    }

    [Fact]
    public void TheDirectionOfTheCorrelationIsResolved()
    {
        var d = NoiseDiagnostics.Diagnose(SmoothedNoise(alongY: false));
        Assert.True(d.Correlated, d.ToString());
        Assert.InRange(d.Right1, 0.5, 0.75);
        Assert.InRange(d.Down1, -0.15, 0.15);
    }

    [Fact]
    public void TheVerdictDoesNotDependOnIntensityUnits()
    {
        // Correlation is scale-free, and the per-tile plane absorbs any offset (camera offset, normalization).
        var scaled = new GrayImage(Frame.Noisy.Width, Frame.Noisy.Height);
        for (int i = 0; i < scaled.Length; i++) scaled.Pixels[i] = Frame.Noisy.Pixels[i] * 7f + 300f;
        var a = NoiseDiagnostics.Diagnose(Frame.Noisy);
        var b = NoiseDiagnostics.Diagnose(scaled);
        Assert.Equal(a.Right1, b.Right1, 3);
        Assert.Equal(a.Down1, b.Down1, 3);
        Assert.Equal(a.Correlated, b.Correlated);
    }

    [Fact]
    public void RefusesImagesWithTooFewUsableTiles()
    {
        Assert.Throws<ArgumentException>(() => NoiseDiagnostics.Diagnose(new GrayImage(31, 31)));
        Assert.Throws<ArgumentException>(() => NoiseDiagnostics.Diagnose(new GrayImage(128, 128)));   // constant: nothing to measure
    }
}

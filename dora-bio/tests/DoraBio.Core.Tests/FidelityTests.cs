using DoraBio.Core.Data;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 32: the fidelity measures, checked on synthetic ground truth where the right answer is known exactly. A perfect
/// output scores perfectly whatever its brightness; compressed contrast lowers the slope and exaggerated contrast raises
/// it; invented spots lower "real";
/// erased spots lower "found", each by the exact count.
/// </summary>
[Trait(Bricks.Trait, Bricks.QuantFidelity)]
public class FidelityTests
{
    private const int Size = 96;

    /// <summary>Gaussian spots (σ = 1.2 px) on a dim background, at the given grid cells (12 px apart).</summary>
    private static GrayImage SpotImage(IEnumerable<(int Gx, int Gy)> cells, float background = 0.05f)
    {
        var pixels = Enumerable.Repeat(background, Size * Size).ToArray();
        foreach (var (gx, gy) in cells)
        {
            int cx = 8 + 12 * gx, cy = 8 + 12 * gy;
            for (int y = Math.Max(0, cy - 5); y <= Math.Min(Size - 1, cy + 5); y++)
                for (int x = Math.Max(0, cx - 5); x <= Math.Min(Size - 1, cx + 5); x++)
                    pixels[y * Size + x] += (float)Math.Exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * 1.2 * 1.2));
        }
        return new GrayImage(Size, Size, pixels);
    }

    private static readonly (int, int)[] Grid = Enumerable.Range(0, 7).SelectMany(gx => Enumerable.Range(0, 7).Select(gy => (gx, gy))).ToArray();

    /// <summary>Smoothly varying foreground: every tile is above the foreground level, with a wide range of brightness.</summary>
    private static GrayImage Ramp() =>
        TestUtil.Pattern(Size, Size, (x, y) => (x + y) / (2f * (Size - 1)));

    [Fact]
    public void APerfectOutputScoresPerfectlyWhateverItsBrightness()
    {
        var clean = SpotImage(Grid);
        foreach (var (a, b) in new[] { (1f, 0f), (40f, 7f), (0.01f, -3f) })
        {
            var output = new GrayImage(Size, Size, clean.Pixels.Select(v => a * v + b).ToArray());
            var s = Fidelity.Measure(output, clean);
            Assert.Equal(49, s.ReferenceSpots);
            Assert.Equal(1.0, s.SpotsFound, 6);
            Assert.Equal(1.0, s.SpotsReal, 6);
        }
        var ramp = Ramp();
        var r = Fidelity.Measure(new GrayImage(Size, Size, ramp.Pixels.Select(v => 3f * v + 0.5f).ToArray()), ramp);
        Assert.InRange(r.ForegroundTiles, 30, 36);   // the darkest corner tiles are background
        Assert.Equal(1.0, r.Slope, 3);
        Assert.Equal(0.0, r.RatioError, 3);

        // A gain left over after normalization isn't distortion either: 30 hot pixels in a background tile move the
        // output's 99.9th percentile, so its foreground comes out about 5x dimmer than the reference's.
        var hot = (float[])ramp.Pixels.Clone();
        for (int k = 0; k < 30; k++) hot[(8 + k / 6) * Size + 8 + k % 6] = 5f;   // inside the top-left (background) tile, away from its darkest pixels
        var h = Fidelity.Measure(new GrayImage(Size, Size, hot), ramp);
        Assert.Equal(1.0, h.Slope, 2);
        Assert.Equal(0.0, h.RatioError, 2);
    }

    [Fact]
    public void ContrastDistortionsMoveTheSlopeTheRightWay()
    {
        // In background-subtracted units a square root compresses contrast (k = 0.5) and a square exaggerates it (k = 2),
        // and a gain changes nothing.
        var ramp = Ramp();
        var compressed = Fidelity.Measure(new GrayImage(Size, Size, ramp.Pixels.Select(v => 5f * MathF.Sqrt(v)).ToArray()), ramp);
        Assert.InRange(compressed.Slope, 0.45, 0.65);   // 0.61: tile averaging and the output's own small background offset
        Assert.True(compressed.RatioError > 0.05, $"ratio error {compressed.RatioError:0.000}");
        var exaggerated = Fidelity.Measure(new GrayImage(Size, Size, ramp.Pixels.Select(v => v * v).ToArray()), ramp);
        Assert.InRange(exaggerated.Slope, 1.7, 2.1);
    }

    [Fact]
    public void InventedAndErasedSpotsAreCountedExactly()
    {
        var reference = SpotImage(Grid.Take(40));
        var invented = Fidelity.Measure(SpotImage(Grid), reference);            // 9 spots that aren't there
        Assert.Equal((40, 49, 40), (invented.ReferenceSpots, invented.OutputSpots, invented.MatchedSpots));
        Assert.Equal(1.0, invented.SpotsFound, 6);
        Assert.Equal(40.0 / 49, invented.SpotsReal, 6);

        var erased = Fidelity.Measure(SpotImage(Grid.Take(30)), reference);      // 10 real spots gone
        Assert.Equal(30.0 / 40, erased.SpotsFound, 6);
        Assert.Equal(1.0, erased.SpotsReal, 6);

        // A spot shifted by 1 px still counts as found; by 4 px it doesn't.
        var shifted = new GrayImage(Size, Size, new float[Size * Size]);
        var src = SpotImage(Grid.Take(1));
        for (int y = 0; y < Size; y++) for (int x = 1; x < Size; x++) shifted.Pixels[y * Size + x] = src[x - 1, y];
        Assert.Equal(1, Fidelity.Measure(shifted, src).MatchedSpots);
    }

    [Fact]
    public void PoolingSumsCountsAndAveragesRates()
    {
        var a = new FidelityScore(1.0, 0.1, 1.0, 0.5, 10, 4, 8, 4);
        var b = new FidelityScore(0.8, 0.3, 0.0, double.NaN, 6, 4, 0, 0);
        var pooled = Fidelity.Pool([a, b]);
        Assert.Equal((16, 8, 8, 4), (pooled.ForegroundTiles, pooled.ReferenceSpots, pooled.OutputSpots, pooled.MatchedSpots));
        Assert.Equal(0.5, pooled.SpotsFound, 6);
        Assert.Equal(0.5, pooled.SpotsReal, 6);
        Assert.Equal(0.9, pooled.Slope, 6);
        Assert.Equal(0.2, pooled.RatioError, 6);
    }
}

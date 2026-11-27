using DoraBio.Core.Data;
using DoraBio.Core.IO;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 40's scoring: per-frame CARE matching, the dynamic and steady masks (S = √max(blurred reference, 0), top 10%),
/// and a block scored as a standalone sequence that never reads a frame outside it.
/// </summary>
[Trait(Bricks.Trait, Bricks.TimeLapseModes)]
public class SequenceBenchmarkTests
{
    [Fact]
    public void TheMasksFollowTheirFormulas()
    {
        // S clamps before the square root: a negative blurred reference gives 0, not NaN.
        var negative = new GrayImage(8, 8, Enumerable.Repeat(-4f, 64).ToArray());
        Assert.All(SequenceBenchmark.Stabilized(negative), v => Assert.Equal(0f, v));
        var flat = new GrayImage(8, 8, Enumerable.Repeat(9f, 64).ToArray());
        Assert.All(SequenceBenchmark.Stabilized(flat), v => Assert.Equal(3f, v, 5));

        // Constant velocity in S: the dynamic change (deviation from the neighbours' midpoint) is zero, the steady one isn't.
        float[] before = [1, 2, 3], current = [2, 4, 3], after = [3, 6, 3];
        Assert.Equal([0f, 0f, 0f], SequenceBenchmark.DynamicChange(before, current, after));
        Assert.Equal([2f, 4f, 0f], SequenceBenchmark.SteadyChange(before, current, after));
        // A transient (up at t, back down after) shows in both.
        Assert.Equal([4f], SequenceBenchmark.DynamicChange([1f], [5f], [1f]));
        Assert.Equal([8f], SequenceBenchmark.SteadyChange([1f], [5f], [1f]));
    }

    [Fact]
    public void TopFractionTakesTheLargestTenthAndItsTies()
    {
        var values = Enumerable.Range(0, 200).Select(i => (float)((i * 37) % 200)).ToArray();   // 0 … 199, shuffled
        var mask = SequenceBenchmark.TopFraction(values);
        Assert.Equal(20, mask.Count(m => m));
        for (int i = 0; i < values.Length; i++) Assert.Equal(values[i] >= 180, mask[i]);
        var ties = Enumerable.Repeat(1f, 50).Concat(Enumerable.Repeat(0f, 50)).ToArray();
        Assert.Equal(50, SequenceBenchmark.TopFraction(ties).Count(m => m));   // the whole tie at the threshold
        // Over every pixel, the masked PSNR is the ordinary PSNR.
        var a = TestUtil.RandomPlane(64, 1);
        var b = TestUtil.RandomPlane(64, 2);
        Assert.Equal(Metrics.Psnr(a, b), SequenceBenchmark.MaskedPsnr(a, b, Enumerable.Repeat(true, 64).ToArray()), 5);
    }

    [Fact]
    public void TheDiagnosticsSeeTransientsSmearsAndFlicker()
    {
        // A still scene in which bright 3 × 3 spots flash for one frame each; two movies of it with independent noise.
        const int w = 64, h = 48, count = 40;
        var scene = SyntheticFluorescence.GenerateSequence(w, h, 1, seed: 3)[0].Clean;
        float peak = scene.Pixels.Max();
        var rng = new Random(7);
        var truth = new List<float[]>();
        for (int t = 0; t < count; t++)
        {
            var frame = (float[])scene.Pixels.Clone();
            for (int s = 0; s < 6; s++)
            {
                int cx = 2 + rng.Next(w - 4), cy = 2 + rng.Next(h - 4);
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) frame[(cy + dy) * w + cx + dx] += 0.6f * peak;
            }
            truth.Add(frame);
        }
        float scale = truth.SelectMany(f => f).Max();
        GrayImage Noisy(float[] clean, float gain, float offset, float sigma, int seed)
        {
            var r = new Random(seed);
            return new GrayImage(w, h, clean.Select(v => gain * v + offset + sigma * scale * (float)(r.NextDouble() + r.NextDouble() + r.NextDouble() - 1.5)).ToArray());
        }
        var low = truth.Select((f, t) => Noisy(f, 1f, 100f, 0.1f, 1000 + t)).ToList();
        var high = truth.Select((f, t) => Noisy(f, 5f, 20f, 0.2f, 2000 + t)).ToList();
        string dir = TestUtil.TempDirectory();
        File.WriteAllBytes(Path.Combine(dir, "low.tif"), TiffCodec.EncodeStack(low, TiffSampleType.Float32));
        File.WriteAllBytes(Path.Combine(dir, "high.tif"), TiffCodec.EncodeStack(high, TiffSampleType.Float32));
        using var lowStack = ImageStack.Open(Path.Combine(dir, "low.tif"));
        using var highStack = ImageStack.Open(Path.Combine(dir, "high.tif"));
        int Index(GrayImage raw) => low.FindIndex(f => ReferenceEquals(f.Pixels, raw.Pixels) || f.Pixels.AsSpan().SequenceEqual(raw.Pixels));
        var (audit, r) = SequenceDiagnostics.Run(lowStack, highStack, new FrameRange(0, count - 1),
        [
            (raw, _) => raw,                                                                          // the noisy input
            (raw, _) => new GrayImage(w, h, truth[Index(raw)].Select(v => 2 * v + 9).ToArray()),      // the truth, other units
            (_, nb) => new GrayImage(w, h, nb[0].Pixels.Zip(nb[1].Pixels, (a, b) => 0.5f * (a + b)).ToArray()),   // never sees t
            (raw, _) => { int t = Index(raw); return new GrayImage(w, h, truth[t].Select(v => v * (t % 2 == 0 ? 1.3f : 0.7f)).ToArray()); },
        ]);
        // Brick 32's pixel-percentile slope on the time averages reads the noisy input as non-proportional when contrast is low
        // (here 101–134 over a flat background); the tile-mean version doesn't, and still catches a real square-root compression.
        Assert.InRange(r[0].TileSlope, 0.9, 1.1);
        Assert.InRange(r[1].TileSlope, 0.9, 1.1);
        Assert.InRange(r[2].TileSlope, 0.9, 1.1);
        var mean = new float[w * h];
        for (int t = 0; t < count; t++) for (int q = 0; q < w * h; q++) mean[q] += truth[t][q] / count;
        float floor = mean.Order().ElementAt((int)(0.05 * (w * h - 1)));
        var compressed = new GrayImage(w, h, mean.Select(v => MathF.Sqrt(Math.Max(0f, v - floor))).ToArray());
        Assert.InRange(SequenceDiagnostics.TileFidelity(compressed, new GrayImage(w, h, mean)).Slope, 0.35, 0.75);   // well below 1
        Assert.Equal(1.0, r[0].TransientKept, 6);   // relative to the noisy input, by definition
        Assert.Equal(1.0, r[0].MotionKept, 6);
        Assert.InRange(r[1].TransientKept, 0.8, 1.2);
        Assert.InRange(r[1].MotionKept, 0.8, 1.2);
        Assert.True(Math.Abs(r[1].MeanFidelity.Slope - 1) > 0.2 && Math.Abs(r[1].TileSlope - 1) < 0.1,
            $"pixel-percentile slope {r[1].MeanFidelity.Slope:0.000} (moved by residual noise), tile slope {r[1].TileSlope:0.000}");
        Assert.InRange(r[2].TransientKept, -0.8, -0.55);  // a one-frame flash becomes a dip in the neighbours' mean: −2/3
        Assert.InRange(r[2].MotionKept, -0.65, -0.35);    // and −1/2 of its first differences
        Assert.True(r[1].GainJitterDb < 0.5, $"the truth shouldn't jitter: {r[1].GainJitterDb:0.00} dB");
        Assert.True(r[3].GainJitterDb > 3, $"a gain alternating ±30% should: {r[3].GainJitterDb:0.00} dB");
        Assert.True(r[1].Flicker < 0.01, $"the truth shouldn't flicker: {r[1].Flicker:0.000}");
        Assert.True(r[3].Flicker > 0.05, $"a gain alternating ±30% should: {r[3].Flicker:0.000}");
        Assert.True(r[1].TraceCorrelation > r[0].TraceCorrelation);
        Assert.True(audit.DynamicInMask > 0.2, $"the flashes are real change: {audit.DynamicInMask:0.000}");
        Assert.Equal(0.0, r[0].Change, 6);

        // Without flashes the two movies share no change, so the audit finds none.
        var stillLow = Enumerable.Range(0, 12).Select(t => Noisy(scene.Pixels, 1f, 100f, 0.1f, 3000 + t)).ToList();
        var stillHigh = Enumerable.Range(0, 12).Select(t => Noisy(scene.Pixels, 5f, 20f, 0.2f, 4000 + t)).ToList();
        File.WriteAllBytes(Path.Combine(dir, "slow.tif"), TiffCodec.EncodeStack(stillLow, TiffSampleType.Float32));
        File.WriteAllBytes(Path.Combine(dir, "shigh.tif"), TiffCodec.EncodeStack(stillHigh, TiffSampleType.Float32));
        using var a = ImageStack.Open(Path.Combine(dir, "slow.tif"));
        using var b = ImageStack.Open(Path.Combine(dir, "shigh.tif"));
        var (still, _) = SequenceDiagnostics.Run(a, b, new FrameRange(0, 11), [(raw, _) => raw]);
        Assert.InRange(still.DynamicInMask, -0.1, 0.1);   // 10 interior frames: about ±0.02 per standard error
        Assert.InRange(still.DynamicAll, -0.1, 0.1);
    }

    [Fact]
    public void ABlockIsScoredAsAStandaloneSequence()
    {
        string dir = TestUtil.TempDirectory();
        var sequence = SyntheticFluorescence.GenerateSequence(48, 40, 10, seed: 5, driftPerFrame: 1);
        var clean = sequence.Select(f => f.Clean).ToList();
        var noisy = sequence.Select(f => f.Noisy).ToList();
        string Write(string name, IReadOnlyList<GrayImage> frames)
        {
            string path = Path.Combine(dir, name);
            File.WriteAllBytes(path, TiffCodec.EncodeStack(frames, TiffSampleType.Float32));
            return path;
        }
        // The same movie with every frame outside the block (3 … 6) replaced: the scores must not move.
        var other = Enumerable.Range(0, 10).Select(t => t is >= 3 and <= 6 ? noisy[t] : TestUtil.RandomImage(48, 40, 100 + t)).ToList();
        var otherClean = Enumerable.Range(0, 10).Select(t => t is >= 3 and <= 6 ? clean[t] : TestUtil.RandomImage(48, 40, 200 + t)).ToList();
        var block = new FrameRange(3, 6);
        List<SequenceScore> Score(string n, string r)
        {
            using var a = ImageStack.Open(n);
            using var b = ImageStack.Open(r);
            // Methods: the noisy frame; an affine copy of the clean frame (a perfect output in other units); and the
            // neighbours' mean, which shows which neighbours were handed over.
            return SequenceBenchmark.Evaluate(a, b, block,
            [
                (raw, _) => raw,
                (raw, _) => new GrayImage(raw.Width, raw.Height, clean[noisy.FindIndex(f => f.Pixels.SequenceEqual(raw.Pixels))].Pixels.Select(v => 3 * v + 7).ToArray()),
                (_, nb) => new GrayImage(nb[0].Width, nb[0].Height, nb[0].Pixels.Zip(nb[1].Pixels, (x, y) => x + y).ToArray()),
            ]);
        }
        var scores = Score(Write("noisy.tif", noisy), Write("clean.tif", clean));
        var outside = Score(Write("noisy2.tif", other), Write("clean2.tif", otherClean));
        for (int m = 0; m < scores.Count; m++)
        {
            Assert.Equal(4, scores[m].Frames.Count);
            Assert.Equal([3, 4, 5, 6], scores[m].Frames.Select(f => f.Frame));
            Assert.Equal(scores[m].Psnr, outside[m].Psnr);
            Assert.Equal(scores[m].DynamicPsnr, outside[m].DynamicPsnr);
            Assert.Equal(scores[m].SteadyPsnr, outside[m].SteadyPsnr);
        }
        Assert.True(scores[1].Psnr > 60, $"a perfect output in other units scores {scores[1].Psnr:0.0} dB");   // the match removes 3·x + 7
        Assert.True(scores[1].DynamicPsnr > 60);
        Assert.InRange(scores[1].Fidelity.Slope, 0.99, 1.01);
        Assert.True(scores[0].Psnr < scores[1].Psnr);
        // At the block's first frame the neighbours are (4, 4), reflected inside the block, not (2, 4).
        using (var a = ImageStack.Open(Path.Combine(dir, "noisy.tif")))
        using (var b = ImageStack.Open(Path.Combine(dir, "clean.tif")))
        {
            var seen = new List<(GrayImage, GrayImage)>();
            SequenceBenchmark.Evaluate(a, b, block, [(raw, nb) => { seen.Add((nb[0], nb[1])); return raw; }]);
            Assert.Equal(noisy[4].Pixels, seen[0].Item1.Pixels);
            Assert.Equal(noisy[4].Pixels, seen[0].Item2.Pixels);
            Assert.Equal(noisy[5].Pixels, seen[^1].Item1.Pixels);   // and at its last frame (6): (5, 5)
            Assert.Equal(noisy[5].Pixels, seen[^1].Item2.Pixels);
            Assert.Throws<ArgumentException>(() => SequenceBenchmark.Evaluate(a, b, new FrameRange(8, 10), [(raw, _) => raw]));
        }
    }
}

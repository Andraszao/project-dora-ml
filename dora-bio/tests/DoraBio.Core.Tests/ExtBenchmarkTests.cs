using DoraBio.Core.Data;
using DoraBio.Core.IO;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Quality pass Q18: CARE's scoring protocol. The least-squares match removes any brightness scale and offset, so a
/// method can't gain by being brighter or darker, only by being closer in shape; the reference is normalized by its own
/// percentiles; and supervised training on an external set normalizes each image of a pair by its own percentiles.
/// </summary>
[Trait(Bricks.Trait, Bricks.ExtBench)]
public class ExtBenchmarkTests
{
    private static GrayImage Image(int seed, float scale = 1f, float offset = 0f) =>
        new(32, 24, TestUtil.RandomPlane(32 * 24, seed).Select(v => v * scale + offset).ToArray());

    [Fact]
    public void TheMatchUndoesAnyScaleAndOffsetExactly()
    {
        var reference = Image(1);
        var output = new GrayImage(32, 24, reference.Pixels.Select(v => 7.5f * v - 3f).ToArray());
        var matched = ExtBenchmark.MatchToReference(output, reference);
        Assert.True(matched.Pixels.Zip(reference.Pixels).Max(p => Math.Abs(p.First - p.Second)) < 1e-5f);
        Assert.True(ExtBenchmark.Score(output, reference).Psnr > 90);
    }

    [Fact]
    public void ScoresDoNotDependOnTheOutputsBrightness()
    {
        var reference = ExtBenchmark.NormalizeReference(Image(2, 3000f, 100f));
        var output = TestUtil.AddGaussianNoise(reference, 0.1, 3);
        var (psnr, ssim) = ExtBenchmark.Score(output, reference);
        foreach (var (a, b) in new[] { (4f, 0f), (0.25f, 10f), (-2f, 5f) })
        {
            var rescaled = new GrayImage(output.Width, output.Height, output.Pixels.Select(v => a * v + b).ToArray());
            var (p, s) = ExtBenchmark.Score(rescaled, reference);
            Assert.Equal(psnr, p, 3);
            Assert.Equal(ssim, s, 4);
        }
        // And the match is the best affine fit: no other scale gets closer.
        var matched = ExtBenchmark.MatchToReference(output, reference);
        double best = Metrics.Mse(matched.Pixels, reference.Pixels);
        foreach (float tweak in new[] { 0.98f, 1.02f })
            Assert.True(Metrics.Mse(matched.Pixels.Select(v => v * tweak).ToArray(), reference.Pixels) > best);
    }

    [Fact]
    public void TheReferenceIsScaledByItsOwnPercentiles()
    {
        var clean = Image(4, 5000f, 200f);
        var normalized = ExtBenchmark.NormalizeReference(clean);
        var sorted = normalized.Pixels.Order().ToArray();
        Assert.Equal(0f, Normalization.Quantile(sorted, 0.001), 3);
        Assert.Equal(1f, Normalization.Quantile(sorted, 0.999), 3);
    }

    [Fact]
    public void SupervisedExtTrainingNormalizesEachImageOfAPairByItsOwnPercentiles()
    {
        // A clean partner 4x brighter than its noisy image, like the DeepBacs pairs: both must come out on [0, 1].
        string root = TestUtil.TempDirectory();
        var mreb = ExtDatasets.Get("deepbacs-mreb");
        var noisy = Image(5, 1000f, 50f);
        var clean = Image(6, 4000f, 200f);
        foreach (var (folder, image) in new[] { (mreb.TrainNoisy, noisy), (mreb.TrainClean!, clean) })
        {
            Directory.CreateDirectory(Path.Combine(root, mreb.Name, folder));
            File.WriteAllBytes(Path.Combine(root, mreb.Name, folder, "m1.tif"), TiffCodec.Encode(image, TiffSampleType.Float32));
        }
        var pair = Assert.Single(ExtDatasets.LoadTraining(root, mreb, supervised: true));
        foreach (var image in new[] { pair.Noisy, pair.Clean! })
        {
            var sorted = image.Pixels.Order().ToArray();
            Assert.Equal(0f, Normalization.Quantile(sorted, 0.001), 3);
            Assert.Equal(1f, Normalization.Quantile(sorted, 0.999), 3);
        }
        var n2v = Assert.Single(ExtDatasets.LoadTraining(root, mreb, supervised: false));
        Assert.Null(n2v.Clean);
        Assert.Equal("deepbacs-mreb/m1", n2v.GroupId);
        Directory.Delete(root, recursive: true);
    }
}

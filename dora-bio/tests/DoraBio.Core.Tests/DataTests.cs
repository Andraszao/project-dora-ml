using DoraBio.Core.Data;
using Xunit;

namespace DoraBio.Core.Tests;

[Trait(Bricks.Trait, Bricks.Data)]
public class DataTests
{
    [Fact]
    public void DihedralProducesEightDistinctLosslessTransforms()
    {
        const int size = 5;
        var src = Enumerable.Range(0, size * size).Select(i => (float)i).ToArray();
        var seen = new HashSet<string>();
        for (int id = 0; id < Dihedral.Count; id++)
        {
            var dst = new float[src.Length];
            Dihedral.Apply(src, dst, size, id);
            Assert.Equal(src.OrderBy(v => v), dst.OrderBy(v => v));   // a permutation: nothing lost or duplicated
            seen.Add(string.Join(",", dst));
        }
        Assert.Equal(8, seen.Count);

        var identity = new float[src.Length];
        Dihedral.Apply(src, identity, size, 0);
        Assert.Equal(src, identity);
    }

    [Fact]
    public void FourQuarterTurnsAreIdentity()
    {
        const int size = 4;
        var a = TestUtil.RandomPlane(size * size, 9);
        var b = new float[a.Length];
        var current = a;
        for (int i = 0; i < 4; i++)
        {
            Dihedral.Apply(current, b, size, 1);
            current = (float[])b.Clone();
        }
        Assert.Equal(a, current);
    }

    [Fact]
    public void BlindSpotMasksStratifiedPixelsWithNeighborValues()
    {
        const int size = 32;
        var original = TestUtil.RandomPlane(size * size, 1);
        var masked = new float[original.Length];
        var mask = new float[original.Length];
        var rng = new Random(2);

        int count = BlindSpot.Apply(original, masked, mask, size, 0.02f, rng);

        Assert.Equal(count, (int)mask.Sum());
        Assert.InRange(count, 16, 36);   // one per 7×7 cell
        for (int i = 0; i < original.Length; i++)
        {
            if (mask[i] == 0f)
            {
                Assert.Equal(original[i], masked[i]);
                continue;
            }
            // The masked value is copied from a neighbor within radius 2, never from the pixel itself.
            int y = i / size, x = i % size;
            bool fromNeighbor = false;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++)
                {
                    int ny = y + dy, nx = x + dx;
                    if ((dy != 0 || dx != 0) && ny >= 0 && nx >= 0 && ny < size && nx < size && original[ny * size + nx] == masked[i])
                        fromNeighbor = true;
                }
            Assert.True(fromNeighbor);
        }
    }

    [Fact]
    public void BlindSpotDrawsAFreshMaskEachCall()
    {
        // The proposal's starter code seeded its own Random inside the function, so every step got the same mask.
        const int size = 16;
        var original = TestUtil.RandomPlane(size * size, 1);
        var rng = new Random(3);
        var m1 = new float[original.Length];
        var m2 = new float[original.Length];
        BlindSpot.Apply(original, new float[original.Length], m1, size, 0.05f, rng);
        BlindSpot.Apply(original, new float[original.Length], m2, size, 0.05f, rng);
        Assert.NotEqual(m1, m2);
    }

    [Fact]
    public void PercentileNormalizationMatchesNumpyQuantiles()
    {
        // np.quantile(np.arange(101), [0.001, 0.999]) == [0.1, 99.9]
        var values = Enumerable.Range(0, 101).Select(i => (float)i).ToArray();
        var map = Normalization.FromPercentiles(values);
        Assert.Equal(0.1f, map.Low, 4);
        Assert.Equal(99.9f, map.High, 4);
        Assert.Equal(0.5f, map.Apply(50f), 5);
        Assert.Equal(50f, map.Invert(map.Apply(50f)), 4);
    }

    [Fact]
    public void NormalizationDoesNotClampByDefault()
    {
        var image = new GrayImage(4, 1, [0f, 10f, 20f, 1000f]);
        var map = new IntensityMap(0f, 20f);
        Assert.Equal(50f, Normalization.Apply(image, map).Pixels[3]);
        Assert.Equal(1f, Normalization.Apply(image, map, clamp: true).Pixels[3]);
    }

    [Fact]
    public void FlatImageNormalizesWithoutDividingByZero()
    {
        var map = Normalization.FromPercentiles(new float[] { 5f, 5f, 5f });
        Assert.True(float.IsFinite(map.Apply(5f)));
    }

    [Fact]
    public void SplitKeepsFieldsOfViewTogether()
    {
        var images = new List<ImagePair>();
        for (int fov = 0; fov < 5; fov++)
            for (int frame = 0; frame < 3; frame++)
                images.Add(new ImagePair(new GrayImage(8, 8), null, $"fov{fov}", $"fov{fov}/{frame}"));

        var (train, val) = DatasetSplit.Split(images, 0.2, seed: 1);
        Assert.Equal(15, train.Count + val.Count);
        Assert.Equal(3, val.Count);   // one whole FOV
        Assert.Empty(train.Select(t => t.GroupId).Intersect(val.Select(v => v.GroupId)));
    }

    [Fact]
    public void SingleFovSplitsSpatially()
    {
        var image = new ImagePair(TestUtil.RandomImage(40, 50, 1), null, "only", "only");
        var (train, val) = DatasetSplit.Split([image], 0.2, seed: 1);
        Assert.Equal(40, train[0].Noisy.Height);
        Assert.Equal(10, val[0].Noisy.Height);
        Assert.Equal(image.Noisy[3, 45], val[0].Noisy[3, 5]);
    }

    [Fact]
    public void SupervisedSamplerAlignsNoisyAndCleanUnderAugmentation()
    {
        var clean = TestUtil.RandomImage(40, 40, 1);
        var noisy = new GrayImage(40, 40, clean.Pixels.Select(v => v + 10f).ToArray());
        var sampler = new PatchSampler([new ImagePair(noisy, clean, "a", "a")], 16, TrainingMode.Supervised, 0.02f, new Random(1));
        var batch = new TrainingBatch(8, 16);
        sampler.Fill(batch);
        for (int i = 0; i < batch.Inputs.Length; i++)
        {
            Assert.Equal(batch.Targets[i] + 10f, batch.Inputs[i], 4);
            Assert.Equal(1f, batch.Masks[i]);
        }
    }

    [Fact]
    public void SyntheticFramesHavePoissonGaussianStatistics()
    {
        var options = new SyntheticOptions { PeakPhotons = 40, ReadNoiseAdu = 2, CameraOffsetAdu = 100 };
        var frame = SyntheticFluorescence.Generate(128, 128, seed: 5, options);

        // Residual variance should be about photons + readNoise² across the image (gain 1).
        double residualVariance = 0, expectedVariance = 0;
        for (int i = 0; i < frame.Noisy.Length; i++)
        {
            double expected = frame.Clean.Pixels[i];
            double r = frame.Noisy.Pixels[i] - expected;
            residualVariance += r * r;
            expectedVariance += (expected - 100) + 4;
        }
        Assert.InRange(residualVariance / expectedVariance, 0.9, 1.1);
        Assert.All(frame.Noisy.Pixels, v => Assert.Equal(MathF.Round(v), v));   // integer camera counts
    }

    [Fact]
    public void PoissonSamplerHasCorrectMeanAndVariance()
    {
        var rng = new Random(1);
        foreach (double lambda in new[] { 0.5, 7.0, 100.0 })
        {
            var samples = Enumerable.Range(0, 40_000).Select(_ => SyntheticFluorescence.SamplePoisson(lambda, rng)).ToArray();
            double mean = samples.Average();
            double variance = samples.Select(s => (s - mean) * (s - mean)).Average();
            Assert.InRange(mean / lambda, 0.97, 1.03);
            Assert.InRange(variance / lambda, 0.93, 1.07);
        }
    }
}

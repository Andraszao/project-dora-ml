using DoraBio.Core.Data;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Structured blind spots: the footprint shapes, the masking invariant (every hidden pixel's value comes from a pixel
/// that no footprint hides), and the plumbing that carries a footprint from the noise check to training and validation.
/// </summary>
[Trait(Bricks.Trait, Bricks.StructuredMask)]
public class StructuredMaskTests
{
    [Fact]
    public void FootprintsHaveTheDocumentedShapesAndSurviveEveryDihedralTransform()
    {
        Assert.Equal(8, new MaskFootprint(FootprintShape.Square, 1).Offsets().Length);
        Assert.Equal(12, new MaskFootprint(FootprintShape.Diamond, 2).Offsets().Length);
        Assert.Equal(24, new MaskFootprint(FootprintShape.Square, 2).Offsets().Length);
        foreach (var footprint in new[] { MaskFootprint.Parse("square-1"), MaskFootprint.Parse("diamond-2"), MaskFootprint.Parse("square-3") })
        {
            var set = footprint.Offsets().ToHashSet();
            Assert.DoesNotContain((0, 0), set);
            // Closed under mirroring and transposition, hence under all 8 flips and quarter turns.
            Assert.All(set, o => Assert.Contains((-o.Dx, o.Dy), set));
            Assert.All(set, o => Assert.Contains((o.Dy, o.Dx), set));
            Assert.Equal(footprint, MaskFootprint.Parse(footprint.ToString()));
        }
        Assert.Throws<ArgumentException>(() => MaskFootprint.Parse("circle-2"));
        Assert.Throws<ArgumentException>(() => MaskFootprint.Parse("square-9"));
    }

    [Theory]
    [InlineData("diamond-2", 64, 0.02f)]
    [InlineData("square-1", 32, 0.05f)]
    [InlineData("square-2", 48, 0.02f)]
    public void EveryHiddenPixelIsFilledFromAPixelThatNoFootprintHides(string footprintName, int size, float ratio)
    {
        var footprint = MaskFootprint.Parse(footprintName);
        // Each pixel's value is its own index, so every copied value names its source pixel exactly. (Random floats
        // won't do: among 4,096 of them a repeat is about as likely as not.)
        var original = Enumerable.Range(0, size * size).Select(i => (float)i).ToArray();
        var indexOf = original.Select((v, i) => (v, i)).ToDictionary(t => t.v, t => t.i);
        var masked = new float[original.Length];
        var loss = new float[original.Length];

        int centres = BlindSpot.Apply(original, masked, loss, size, ratio, new Random(9), footprint: footprint);
        Assert.Equal(centres, (int)loss.Sum());

        var hidden = new HashSet<int>();
        for (int i = 0; i < loss.Length; i++)
        {
            if (loss[i] == 0f) continue;
            int y = i / size, x = i % size;
            hidden.Add(i);
            foreach (var (dx, dy) in footprint.Offsets())
                if (y + dy >= 0 && y + dy < size && x + dx >= 0 && x + dx < size) hidden.Add((y + dy) * size + x + dx);
        }

        for (int i = 0; i < original.Length; i++)
        {
            if (!hidden.Contains(i)) { Assert.Equal(original[i], masked[i]); continue; }
            Assert.True(indexOf.TryGetValue(masked[i], out int source), "a hidden pixel's value must be copied from the patch");
            Assert.DoesNotContain(source, hidden);   // never from a centre or from inside any footprint
        }
        Assert.True(hidden.Count > 3 * centres, "the footprint hides far more than the centres");
    }

    [Fact]
    public void WithoutAFootprintTheBlindSpotIsExactlyThePlainOne()
    {
        const int size = 32;
        var original = TestUtil.RandomPlane(size * size, 1);
        var (a, b) = (new float[original.Length], new float[original.Length]);
        var (ma, mb) = (new float[original.Length], new float[original.Length]);
        BlindSpot.Apply(original, a, ma, size, 0.02f, new Random(4));
        BlindSpot.Apply(original, b, mb, size, 0.02f, new Random(4), footprint: null);
        Assert.Equal(a, b);
        Assert.Equal(ma, mb);
    }

    [Fact]
    public void AFootprintThatHidesTheWholePatchIsRefused()
    {
        var original = TestUtil.RandomPlane(16, 1);
        Assert.Throws<ArgumentException>(() => BlindSpot.Apply(original, new float[16], new float[16], 4, 0.5f, new Random(1),
            footprint: new MaskFootprint(FootprintShape.Square, 4)));
    }

    [Fact]
    public void AutoModeStructuresOnlyTheImagesWithCorrelatedNoise()
    {
        var independent = new ImagePair(NoiseDiagnosticsTests.Frame.Noisy, NoiseDiagnosticsTests.Frame.Clean, "a", "independent");
        var correlated = new ImagePair(NoiseDiagnosticsTests.SmoothedNoise(alongY: true), NoiseDiagnosticsTests.Frame.Clean, "b", "correlated");
        ImagePair[] images = [independent, correlated];

        var auto = StructuredMasking.Assign(images, StructuredMaskMode.Auto);
        Assert.Null(auto[0].Footprint);
        Assert.Equal(StructuredMasking.DefaultFootprint, auto[1].Footprint);
        Assert.All(StructuredMasking.Assign(images, StructuredMaskMode.Always), p => Assert.NotNull(p.Footprint));
        Assert.All(StructuredMasking.Assign(images, StructuredMaskMode.Off), p => Assert.Null(p.Footprint));
        Assert.Equal([false, true], StructuredMasking.Diagnose(images).Select(v => v!.Value));

        // The footprint travels with the image through normalization and the spatial train/validation split.
        var (normalized, _) = auto[1].Normalize();
        Assert.Equal(auto[1].Footprint, normalized.Footprint);
        Assert.Equal(auto[1].Footprint, auto[1].Crop(0, 0, 64, 64, ":val").Footprint);
    }

    [Fact]
    public void AdaptToSampleRunsWithStructuredBlindSpots()
    {
        // What the Godot app does on an image whose noise check says correlated.
        var (image, _) = new ImagePair(NoiseDiagnosticsTests.SmoothedNoise(alongY: true), null, "s", "s").Normalize();
        var net = DoraBio.Core.Nn.DenoisingNet.Create(new DoraBio.Core.Nn.NetConfig(8, [1, 2, 4, 2, 1]), seed: 1);
        var adapter = new OnlineAdapter(net, learningRate: 3e-3f, patchSize: 32, batchSize: 8);
        var losses = adapter.Adapt(image.Noisy, 60, footprint: StructuredMasking.DefaultFootprint);
        Assert.Equal(60, losses.Count);
        Assert.All(losses, l => Assert.True(double.IsFinite(l) && l > 0));
        Assert.True(losses.Skip(50).Average() < losses.Take(10).Average(), "adaptation should lower the blind-spot loss");
    }

    [Fact]
    public void TrainingAndValidationPatchesUseTheImagesFootprint()
    {
        var image = new ImagePair(TestUtil.RandomImage(96, 96, seed: 2), null, "fov", "img");
        var structured = image with { Footprint = new MaskFootprint(FootprintShape.Diamond, 2) };

        int Changed(ImagePair pair)
        {
            var sampler = new PatchSampler([pair], 32, TrainingMode.Noise2Void, 0.02f, new Random(3));
            var batch = new TrainingBatch(4, 32);
            sampler.Fill(batch);
            int changed = 0, centres = 0;
            for (int i = 0; i < batch.Inputs.Length; i++)
            {
                if (batch.Inputs[i] != batch.Targets[i]) changed++;
                if (batch.Masks[i] == 1f) centres++;
            }
            return changed / Math.Max(1, centres);   // hidden pixels per blind spot
        }
        Assert.True(Changed(image) <= 1);       // plain: only the centres change
        Assert.True(Changed(structured) >= 5);  // diamond-2: up to 13 per centre, fewer at overlaps and edges

        var config = new TrainerConfig { Mode = TrainingMode.Noise2Void, PatchSize = 32, ValidationPatches = 2 };
        var validation = new ValidationSet([structured], config);
        Assert.Equal(2, validation.PatchSources.Count);
        Assert.True(validation.HiddenPixelsPerBlindSpot() >= 5);
    }
}

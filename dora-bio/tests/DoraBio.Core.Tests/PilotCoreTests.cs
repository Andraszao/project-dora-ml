using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 41 (pilot readiness, core and CLI): stacks follow their axes (a time-lapse model's neighbours are the same channel
/// and slice one frame away, never the next page), Z-stacks are refused for time-lapse modes and unlabelled planes are
/// taken as frames with a notice, nothing is silently dropped (every page is read, an ImageJ stack over 4 GB is refused),
/// and routing is decided once per stack with no network trained on a handful of images.
/// </summary>
[Trait(Bricks.Trait, Bricks.PilotCore)]
public class PilotCoreTests
{
    // imagej_tzc.tif (tifffile): T=2, Z=3, C=2 planes of 5 x 4; pixel (0, 0) of the plane at (t, z, c) is 1000t + 100z + 10c + 1.
    private static (int T, int Z, int C) Position(GrayImage plane)
    {
        int v = (int)plane.Pixels[0] - 1;
        return (v / 1000, v / 100 % 10, v / 10 % 10);
    }

    /// <summary>A 5-plane stack ImageJ labels as slices (what Fiji saves for a plain, non-hyperstack stack).</summary>
    private static string ZStack()
    {
        string path = Path.Combine(TestUtil.TempDirectory(), "zstack.tif");
        var planes = Enumerable.Range(0, 5).Select(k => TestUtil.RandomImage(16, 12, 40 + k)).ToList();
        File.WriteAllBytes(path, TiffCodec.EncodeStack(planes, TiffSampleType.Float32, description: "ImageJ=1.54f\nimages=5\nslices=5\n"));
        return path;
    }

    [Fact]
    public void NeighboursAreTheSameChannelAndSliceOneFrameAway()
    {
        using var stack = ImageStack.Open(TestUtil.Fixture("imagej_tzc.tif"));
        Assert.Equal("TZC", stack.Layout.Axes);
        var time = TimeAxis.Resolve(stack.Layout);
        Assert.Equal(('T', (string?)null), (time.Axis, time.Notice));
        for (int p = 0; p < stack.PlaneCount; p++)
        {
            var (t, z, c) = Position(stack.ReadPlane(p));
            foreach (int q in TimeAxis.NeighbourPages(stack.Layout, 'T', p))
                Assert.Equal((1 - t, z, c), Position(stack.ReadPlane(q)));   // T = 2: each frame's neighbours are the other frame
        }
        // Six series (3 slices x 2 channels), each the two frames of one slice and channel, labelled by them.
        var series = TimeAxis.Series(stack.Layout, 'T').ToList();
        Assert.Equal(6, series.Count);
        Assert.Contains(series, s => s.Label == "Z=2,C=1");
        foreach (var (label, pages) in series)
        {
            var positions = pages.Select(p => Position(stack.ReadPlane(p))).ToList();
            Assert.Equal([0, 1], positions.Select(x => x.T));
            Assert.Single(positions.Select(x => (x.Z, x.C)).Distinct());
            Assert.Equal($"Z={positions[0].Z},C={positions[0].C}", label);
        }
        // A longer T axis reflects at the ends, other axes fixed: T=5, C=2 (channels fastest, as ImageJ stores them).
        var layout = new StackLayout("TC", [5, 2], StackMetadata.ImageJ);
        Assert.Equal([3, 3], TimeAxis.NeighbourPages(layout, 'T', 1));   // t=0, c=1: neighbours (1, 1) → pages 3, 3
        Assert.Equal([2, 6], TimeAxis.NeighbourPages(layout, 'T', 4));   // t=2, c=0: pages 2 and 6, never the adjacent page 5
        Assert.Equal([7, 7], TimeAxis.NeighbourPages(layout, 'T', 9));   // t=4, c=1: (3, 3) → page 7
    }

    [Fact]
    public void ZStacksAreRefusedAndUnlabelledPlanesAreFramesWithANotice()
    {
        var z = Assert.Throws<ArgumentException>(() => TimeAxis.Resolve(new StackLayout("Z", [7], StackMetadata.ImageJ)));
        Assert.Contains("--axes TYX", z.Message);
        Assert.Contains("Stack to Hyperstack", z.Message);
        Assert.Throws<ArgumentException>(() => TimeAxis.Resolve(new StackLayout("C", [3], StackMetadata.ImageJ)));
        Assert.Throws<ArgumentException>(() => TimeAxis.Resolve(new StackLayout("", [], StackMetadata.None)));
        var unlabelled = TimeAxis.Resolve(new StackLayout("I", [9], StackMetadata.None));
        Assert.Equal('I', unlabelled.Axis);
        Assert.Equal("axes unlabelled: planes treated as T (pass --axes ZYX if this is a Z-stack)", unlabelled.Notice);
        Assert.Throws<ArgumentException>(() => ImageStack.Open(TestUtil.Fixture("imagej_tzc.tif"), "TZC"));   // axes must end in YX
    }

    [Fact]
    public void AHyperstackDenoisesInPageOrderWithEachChannelsOwnNeighbours()
    {
        var net = DenoisingNet.Create(new NetConfig(4, [1, 1]) with { InputFrames = 3 }, 3);
        var denoiser = new FrameDenoiser(net, tileCore: 16);
        string dir = TestUtil.TempDirectory(), output = Path.Combine(dir, "out.tif");
        var notices = new List<string>();
        int pages = BatchDenoising.DenoiseFile(denoiser, TestUtil.Fixture("imagej_tzc.tif"), output, onNotice: notices.Add);
        Assert.Equal(12, pages);
        Assert.Empty(notices);   // labelled frames: nothing to assume
        using var stack = ImageStack.Open(TestUtil.Fixture("imagej_tzc.tif"));
        using var written = TiffStackReader.Open(output);
        Assert.Equal(12, written.PageCount);
        for (int p = 0; p < 12; p++)
        {
            var neighbours = TimeAxis.NeighbourPages(stack.Layout, 'T', p).Select(stack.ReadPlane).ToArray();
            Assert.Equal(denoiser.Denoise(stack.ReadPlane(p), neighbours).Output.Pixels, written.ReadPage(p).Pixels);
        }
        // A stack labelled as slices is refused for this model, unless declared as frames; a single-frame model denoises it.
        string z = ZStack(), other = Path.Combine(dir, "other.tif");   // (output is still open above)
        var refused = Assert.Throws<ArgumentException>(() => BatchDenoising.DenoiseFile(denoiser, z, other));
        Assert.Contains("labelled Z", refused.Message);
        Assert.Equal(5, BatchDenoising.DenoiseFile(denoiser, z, other, axes: "TYX"));
        var single = new FrameDenoiser(DenoisingNet.Create(new NetConfig(4, [1, 1]), 3), tileCore: 16);
        Assert.Equal(5, BatchDenoising.DenoiseFile(single, z, other));
        // A declaration that contradicts the file's own labels is an error, not a silent relabelling.
        Assert.Throws<InvalidDataException>(() => BatchDenoising.DenoiseFile(denoiser, TestUtil.Fixture("imagej_tzc.tif"), other, axes: "ZYX"));
    }

    [Fact]
    public void TimeLapseTrainingSplitsAHyperstackIntoItsSeries()
    {
        var images = Sequences.Load(TestUtil.Fixture("imagej_tzc.tif"), TrainingMode.Noise2Void, 3);
        Assert.Equal(12, images.Count);   // 6 series of 2 frames
        using var stack = ImageStack.Open(TestUtil.Fixture("imagej_tzc.tif"));
        GrayImage Normalized(int t, int z, int c)
        {
            var raw = stack.ReadPlane(stack.PlaneIndex(t, z, c));
            return Normalization.Apply(raw, Normalization.FromPercentiles(raw.Pixels));
        }
        foreach (var image in images)
        {
            // Names: "imagej_tzc[Z=z,C=c]/t000t"; the context is the other frame of the same slice and channel.
            var name = image.Name;
            int z = name[name.IndexOf("Z=", StringComparison.Ordinal) + 2] - '0', c = name[name.IndexOf("C=", StringComparison.Ordinal) + 2] - '0';
            int t = name[^1] - '0';
            Assert.Equal(Normalized(t, z, c).Pixels, image.Noisy.Pixels);
            Assert.All(image.Context!, n => Assert.Equal(Normalized(1 - t, z, c).Pixels, n.Pixels));
        }
        Assert.Throws<ArgumentException>(() => Sequences.Load(ZStack(), TrainingMode.Noise2Void, 3));
        Assert.Equal(5, Sequences.Load(ZStack(), TrainingMode.Noise2Void, 3, axes: "TYX").Count);
    }

    [Fact]
    public void EveryPageIsReadAndAnImageJStackOver4GbIsRefused()
    {
        string dir = TestUtil.TempDirectory();
        File.Copy(TestUtil.Fixture("imagej_tzc.tif"), Path.Combine(dir, "stack.tif"));
        File.WriteAllBytes(Path.Combine(dir, "single.tif"), TiffCodec.Encode(TestUtil.RandomImage(5, 4, 1) is var r ? r : r));
        var images = DatasetLoader.LoadDirectory(dir);
        Assert.Equal(13, images.Count);   // 12 planes of the stack and the single image
        Assert.Equal(12, images.Count(i => i.GroupId == "stack"));   // one FOV: a split keeps them together
        Assert.Contains(images, i => i.Name == "stack.tif#p11");
        Assert.Contains(images, i => i.Name == "single.tif");

        // ImageJ writes a stack over 4 GB as one page directory followed by raw planes, with images=N in its header:
        // reading that as one page would silently drop the rest, so every path refuses it.
        string big = Path.Combine(dir, "big", "imagej_big.tif");
        Directory.CreateDirectory(Path.GetDirectoryName(big)!);
        File.WriteAllBytes(big, TiffCodec.EncodeStack([TestUtil.RandomImage(8, 8, 2)], description: "ImageJ=1.54f\nimages=3\nslices=3\n"));
        Assert.Throws<NotSupportedException>(() => BatchDenoising.CountPages(big));
        Assert.Throws<NotSupportedException>(() => DatasetLoader.Planes(big));
        var denoiser = new FrameDenoiser(DenoisingNet.Create(new NetConfig(4, [1, 1]), 3), tileCore: 16);
        Assert.Throws<NotSupportedException>(() => BatchDenoising.DenoiseFile(denoiser, big, Path.Combine(dir, "out.tif")));
        Assert.Throws<NotSupportedException>(() => DatasetLoader.LoadDirectory(Path.GetDirectoryName(big)!));
    }

    [Fact]
    public void TheWriterKeepsAnImageDescriptionAndIsOtherwiseUnchanged()
    {
        var pages = new[] { TestUtil.RandomImage(6, 5, 1), TestUtil.RandomImage(6, 5, 2) };
        const string header = "ImageJ=1.54f\nimages=2\nframes=2\n";
        var bytes = TiffCodec.EncodeStack(pages, TiffSampleType.Float32, description: header);
        using var reader = new TiffStackReader(new MemoryStream(bytes));
        Assert.Equal(header, reader.Description);
        Assert.Equal(pages[1].Pixels, reader.ReadPage(1).Pixels);
        using var stack = ImageStack.FromTiff(new TiffStackReader(new MemoryStream(bytes)));
        Assert.Equal(("T", 2L), (stack.Layout.Axes, stack.Layout.PlaneCount));
        // Without a description the file is byte for byte what it was before.
        Assert.Equal(TiffCodec.Encode(pages[0]), TiffCodec.EncodeStack([pages[0]]));
    }

    [Fact]
    public void RoutingIsOneDecisionPerStackAndNeverOneNetworkPerFewImages()
    {
        static List<ImagePair> Images(int n, string prefix) =>
            Enumerable.Range(0, n).Select(i => new ImagePair(new GrayImage(4, 4), null, $"{prefix}{i}", $"{prefix}{i}")).ToList();
        var (independent, correlated, note) = RoutedNoise2Void.Coherent(Images(29, "i"), Images(1, "c"));
        Assert.Equal((30, 0), (independent.Count, correlated.Count));
        Assert.Contains("1 of 30 images look correlated", note);
        (independent, correlated, note) = RoutedNoise2Void.Coherent(Images(2, "i"), Images(40, "c"));   // 2 < 3: they join
        Assert.Equal((0, 42), (independent.Count, correlated.Count));
        Assert.NotNull(note);
        (independent, correlated, note) = RoutedNoise2Void.Coherent(Images(10, "i"), Images(10, "c"));
        Assert.Equal((10, 10, (string?)null), (independent.Count, correlated.Count, note));
        Assert.Equal(2, RoutedNoise2Void.Coherent(Images(2, "i"), Images(2, "c"), minimumImages: 1).Correlated.Count);

        // Denoising: a 5-plane stack with one correlated-looking plane goes entirely to the main model, and the reverse
        // entirely to the correlated-noise model (the majority of the sampled planes decides, once).
        var plain = NoiseDiagnosticsTests.Frame.Noisy;
        var smooth = NoiseDiagnosticsTests.SmoothedNoise(alongY: true);
        var denoiser = new FrameDenoiser(DenoisingNet.Create(new NetConfig(4, [1, 1]), 1), DenoisingNet.Create(new NetConfig(4, [1, 1]), 2), tileCore: 64);
        foreach (var (stackPlanes, expectCorrelated) in new[] { (new[] { plain, plain, smooth, plain, plain }, false), (new[] { smooth, smooth, plain, smooth, smooth }, true) })
        {
            using var stack = ImageStack.FromTiff(new TiffStackReader(new MemoryStream(TiffCodec.EncodeStack(stackPlanes, TiffSampleType.Float32))));
            using var sink = new TiffStackWriter(new MemoryStream(), TiffSampleType.Float32);
            var used = new List<bool>();
            var notices = new List<string>();
            BatchDenoising.DenoiseStack(denoiser, stack, sink, onPage: (_, _, r) => used.Add(r.UsedCorrelatedModel), onNotice: notices.Add);
            Assert.All(used, u => Assert.Equal(expectCorrelated, u));
            Assert.Contains(notices, n => n.StartsWith("noise check on 5 of 5 planes", StringComparison.Ordinal));
        }
    }
}

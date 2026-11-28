using System.Security.Cryptography;
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
        File.WriteAllBytes(path, TiffCodec.EncodeStack(planes, TiffSampleType.Float32, tags: new TiffPageTags { Description = "ImageJ=1.54f\nimages=5\nslices=5\n" }));
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
        File.WriteAllBytes(big, TiffCodec.EncodeStack([TestUtil.RandomImage(8, 8, 2)], tags: new TiffPageTags { Description = "ImageJ=1.54f\nimages=3\nslices=3\n" }));
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
        var bytes = TiffCodec.EncodeStack(pages, TiffSampleType.Float32, tags: new TiffPageTags { Description = header });
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

    [Fact]
    public void TheContentHashIsTheModelNotItsProvenanceOrSaveTime()
    {
        // Gate 8 compares the app's trained model with the CLI's by this hash; the files differ in their save time.
        var net = DenoisingNet.Create(new NetConfig(4, [1, 2, 1]), seed: 3);
        string hash = WeightsFile.ContentHash(net);
        Assert.Matches("^sha256:[0-9a-f]{64}$", hash);
        var reloaded = WeightsFile.Load(WeightsFile.Serialize(net, "another description", new DateTime(2020, 1, 2, 0, 0, 0, DateTimeKind.Utc))).Net;
        Assert.Equal(hash, WeightsFile.ContentHash(reloaded));
        var changed = net.Clone();
        changed.Parameters[^1] = BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(changed.Parameters[^1]) + 1);   // one ulp
        Assert.NotEqual(hash, WeightsFile.ContentHash(changed));
        Assert.NotEqual(hash, WeightsFile.ContentHash(DenoisingNet.Create(new NetConfig(4, [1, 3, 1]), seed: 3)));
    }

    [Fact]
    public void FineTuningTheAppsModelTrainsThatNetworkNotTheRoutedPair()
    {
        // The app fine-tunes the model on screen (TrainRequest.InitNet), which, like --init, is a recipe choice: one
        // network of that architecture, never the routed default, and the caller's network is left as it was.
        var noisyOnly = SyntheticFluorescence.GenerateDataset(4, 64, seed: 5).Select(p => p with { Clean = null }).ToList();
        var current = DenoisingNet.Create(new NetConfig(4, [1, 2, 1]), seed: 9);
        var before = (float[])current.Parameters.Clone();
        string dir = TestUtil.TempDirectory();
        try
        {
            var lines = new List<string>();
            var models = Pipeline.Train(new TrainRequest
            {
                Dataset = noisyOnly, DatasetName = "test", Out = Path.Combine(dir, "w.bin"), InitNet = current,
                Epochs = 1, Steps = 2, Batch = 2, Patch = 32,
            }, lines.Add);
            var model = Assert.Single(models);
            Assert.Null(model.Kind);
            var info = WeightsFile.Load(model.WeightsPath).Info;
            Assert.Equal(current.Config.ToString(), info.Config.ToString());
            // Provenance records how it was trained: DORA's build, the settings with the seed, and the data's fingerprint.
            Assert.Contains($"trained by DORA {Pipeline.Version}: Noise2Void, route off, up to 1 epochs × 2 steps × batch 2 @ 32px", info.Provenance);
            Assert.Contains("seed 1234", info.Provenance);
            Assert.Contains($"data {Pipeline.DataFingerprint(noisyOnly)} (4 images)", info.Provenance);
            Assert.NotEqual(Pipeline.DataFingerprint(noisyOnly), Pipeline.DataFingerprint(noisyOnly.Take(3).ToList()));
            Assert.Equal(before, current.Parameters);
            Assert.DoesNotContain(lines, l => l.Contains("routed", StringComparison.Ordinal));
            Assert.Contains(lines, l => l.StartsWith("📚 4 images from 'test' (no clean images)", StringComparison.Ordinal));

            // Without a recipe choice the same images take the routed default, exactly as `dora-bio train` does.
            var routed = Pipeline.Train(new TrainRequest
            {
                Dataset = noisyOnly, DatasetName = "test", Out = Path.Combine(dir, "r.bin"), Epochs = 1, Steps = 2, Batch = 2, Patch = 32,
            });
            Assert.Equal(NoiseKind.Independent, Assert.Single(routed).Kind);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Pixel (x, y) of lzw_noise_u16_pred2.tif: a hash, so the strip doesn't compress (make_pilot_fixtures.py).</summary>
    private static float NoiseHash(int x, int y)
    {
        uint h = ((uint)x * 73856093u) ^ ((uint)y * 19349663u);
        h *= 2654435761u;
        return h >> 16;
    }

    [Fact]
    public void LzwFromAnIndependentEncoderDecodesExactlyThroughEveryCodeWidthAndTableReset()
    {
        // imagecodecs' encoder (via tifffile), horizontal predictor, one 18 KB strip of incompressible pixels: its codes
        // grow from 9 to 12 bits and the table is cleared and rebuilt along the way.
        var bytes = File.ReadAllBytes(TestUtil.Fixture("lzw_noise_u16_pred2.tif"));
        var image = TiffCodec.Decode(bytes);
        Assert.Equal((96, 96), (image.Width, image.Height));
        for (int y = 0; y < 96; y++)
            for (int x = 0; x < 96; x++)
                Assert.Equal(NoiseHash(x, y), image.Pixels[y * 96 + x]);
        var strip = Assert.Single(TiffCodec.StoredStrips(bytes, 0));
        Lzw.Decode(strip, 96 * 96 * 2, out var stats);
        Assert.Equal(12, stats.MaxCodeWidth);
        Assert.True(stats.TableResets >= 1, $"the table should be reset at least once, was {stats.TableResets}");

        // Long runs: strings that repeat themselves (the code being defined is the one read, "KwKwK").
        var runs = TiffCodec.Decode(File.ReadAllBytes(TestUtil.Fixture("lzw_runs_u8.tif")));
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
                Assert.Equal(y < 32 ? 7 : (x * 3 + y) % 256, runs.Pixels[y * 64 + x]);

        // Refusals: old-style (pre-TIFF 6.0, least-significant bit first) LZW, and a strip that stops short.
        var old = Assert.Throws<NotSupportedException>(() => Lzw.Decode([0x00, 0x01, 0x02, 0x03], 4));
        Assert.Contains("old-style LZW", old.Message);
        Assert.Throws<InvalidDataException>(() => Lzw.Decode(strip.AsSpan(0, strip.Length / 2), 96 * 96 * 2));
    }

    [Fact]
    public void OtherCompressionsAreRefusedByNameWithAPlainConversion()
    {
        string message = TiffCodec.UnsupportedCompression(50000);
        Assert.Contains("TIFF compression 50000 (Zstandard)", message);
        Assert.Contains("tifffile.imwrite(dst, tifffile.imread(src))", message);
        Assert.Contains("--axes ZYX", message);
        Assert.Contains("(JPEG 2000)", TiffCodec.UnsupportedCompression(34712));
        Assert.StartsWith("TIFF compression 12345 is not supported.", TiffCodec.UnsupportedCompression(12345));
    }

    [Fact]
    public void OmeXmlStacksAreRefusedUntilTheirAxisIsDeclared()
    {
        // tifffile's OME-TIFF writer: 4 planes that OME-XML labels Z. DORA doesn't read OME-XML, and must not let the
        // planes fall through to "unlabelled, so frames" (brick 41).
        string ome = TestUtil.Fixture("ome_z4.tif");
        var error = Assert.Throws<NotSupportedException>(() => ImageStack.Open(ome));
        Assert.Contains("OME-XML", error.Message);
        Assert.Contains("--axes", error.Message);
        Assert.Throws<NotSupportedException>(() => BatchDenoising.CountPages(ome));
        Assert.Throws<NotSupportedException>(() => DatasetLoader.Planes(ome));
        using (var volume = ImageStack.Open(ome, "ZYX"))
        {
            Assert.Equal(("Z", StackMetadata.OmeXml), (volume.Layout.Axes, volume.Layout.Source));
            Assert.Throws<ArgumentException>(() => TimeAxis.Resolve(volume.Layout));
            // A float32 copy must not claim the input's OME pixel type; it records the declared axes instead.
            Assert.Equal("""{"shape":[4,4,5],"axes":"ZYX"}""", volume.OutputTags(null).Description);
        }
        using (var movie = ImageStack.Open(ome, "TYX")) Assert.Equal('T', TimeAxis.Resolve(movie.Layout).Axis);
        Assert.Equal(4, BatchDenoising.CountPages(ome, "ZYX"));

        // Denoised with --axes ZYX, the copy reopens as a volume with no declaration (and without OME-XML).
        string dir = TestUtil.TempDirectory();
        try
        {
            var denoiser = new FrameDenoiser(DenoisingNet.Create(new NetConfig(4, [1, 1]), 3), tileCore: 16);
            string copy = Path.Combine(dir, "ome_denoised.tif");
            Assert.Equal(4, BatchDenoising.DenoiseFile(denoiser, ome, copy, axes: "ZYX"));
            using var reopened = ImageStack.Open(copy);
            Assert.Equal(("Z", StackMetadata.Tifffile), (reopened.Layout.Axes, reopened.Layout.Source));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
        // Planes nobody labelled stay unlabelled in the copy.
        using var plain = ImageStack.Open(TestUtil.Fixture("stack2.tif"));
        Assert.Equal(StackMetadata.None, plain.Layout.Source);
        Assert.Null(plain.OutputTags(null).Description);
    }

    [Fact]
    public void ADenoisedCopyKeepsTheImageJHeaderAndPixelSizeAndSaysWhichDoraMadeIt()
    {
        // tifffile's ImageJ writer: 3 frames, 0.65 µm pixels (resolution 20/13 per unit, unit=um in the header), 2 s apart.
        string input = TestUtil.Fixture("imagej_calibrated_t3.tif");
        string dir = TestUtil.TempDirectory();
        try
        {
            var net = DenoisingNet.Create(new NetConfig(4, [1, 2, 1]), seed: 1);
            string output = Path.Combine(dir, "out.tif"), baseline = Path.Combine(dir, "baseline.tif");
            BatchDenoising.DenoiseFile(new FrameDenoiser(net, tileCore: 16), input, output, baseline);
            using var source = TiffStackReader.Open(input);
            foreach (string written in new[] { output, baseline })
            {
                using var copy = TiffStackReader.Open(written);
                Assert.Equal(source.Description, copy.Description);   // verbatim: frames, unit and interval as ImageJ wrote them
                Assert.Equal(new TiffResolution(20, 13, 20, 13, 1), copy.Resolution);
                Assert.Equal(3, copy.PageCount);
                for (int p = 1; p < 3; p++) Assert.Equal(copy.Resolution, copy.PageResolution(p));
                Assert.StartsWith($"DORA {Pipeline.Version}; ", copy.Software);
                Assert.DoesNotContain("tifffile", copy.Software);
            }
            using (var denoised = TiffStackReader.Open(output))
            {
                Assert.Contains($"model {WeightsFile.ContentHash(net)}", denoised.Software);
                Assert.Contains("single-frame", denoised.Software);
            }
            using (var filtered = TiffStackReader.Open(baseline)) Assert.Contains("bilateral baseline", filtered.Software);

            // The description goes on page 0 only, the Software on page 0 only; the pixels are what they were without tags.
            var plain = TiffCodec.EncodeStack([TestUtil.RandomImage(6, 5, 3)], TiffSampleType.Float32);
            var tagged = TiffCodec.EncodeStack([TestUtil.RandomImage(6, 5, 3)], TiffSampleType.Float32,
                tags: new TiffPageTags { Description = "ImageJ=1.54f\n", Software = "x", Resolution = new TiffResolution(3, 1, 3, 1, 3) });
            Assert.Equal(TiffCodec.Decode(plain).Pixels, TiffCodec.Decode(tagged).Pixels);
            using var short_ = new TiffStackReader(new MemoryStream(tagged));
            Assert.Equal("x", short_.Software);   // 2 bytes with its NUL: stored inline in the entry, as TIFF requires
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ATifffileHeaderIsKeptOnlyWhereItCantContradictTheFloatCopy()
    {
        // Shape and axes only: verbatim. Other keys (a dtype the float32 copy no longer has): just shape and axes.
        // A shape that isn't the pages' (an RGB input, whose copy is gray): dropped.
        Assert.Equal("{\"shape\": [2, 4, 5], \"axes\": \"TYX\"}", ImageStack.TifffileHeaderFor("{\"shape\": [2, 4, 5], \"axes\": \"TYX\"}", 5, 4));
        Assert.Equal("{\"shape\":[2,4,5],\"axes\":\"TYX\"}",
            ImageStack.TifffileHeaderFor("{\"shape\": [2, 4, 5], \"axes\": \"TYX\", \"dtype\": \"uint16\"}", 5, 4));
        Assert.Equal("{\"shape\":[2,4,5]}", ImageStack.TifffileHeaderFor("{\"shape\": [2, 4, 5], \"exposure\": 0.1}", 5, 4));
        Assert.Null(ImageStack.TifffileHeaderFor("{\"shape\": [4, 5, 3], \"axes\": \"YXS\"}", 5, 4));
        Assert.Null(ImageStack.TifffileHeaderFor("{\"shape\": [2, 8, 8]}", 5, 4));
    }

    [Fact]
    public void TheResidualCheckTellsRemovedNoiseFromRemovedStructure()
    {
        var frame = SyntheticFluorescence.Generate(96, 96, seed: 21);
        // A perfect denoiser leaves the noise itself: zero-mean, unrelated to the output and between neighbours.
        var perfect = new ResidualCheck();
        perfect.Add(frame.Noisy, frame.Clean);
        Assert.InRange(perfect.ResidualOutputCorrelation, -0.1, 0.1);
        Assert.InRange(perfect.LagX, -0.1, 0.1);
        Assert.InRange(perfect.LagY, -0.1, 0.1);
        Assert.InRange(perfect.NetChange, -0.01, 0.01);
        Assert.True(perfect.RmsChange > 0);

        // One that also takes away a fifth of the signal: the residual carries structure, and the output is darker.
        var shrunk = new GrayImage(96, 96);
        for (int i = 0; i < shrunk.Length; i++) shrunk.Pixels[i] = 0.8f * frame.Clean.Pixels[i];
        var lossy = new ResidualCheck();
        lossy.Add(frame.Noisy, shrunk);
        // (How far above 0 depends on the noise: here 0.2 of the signal's spread against the photon noise.)
        Assert.True(lossy.ResidualOutputCorrelation > perfect.ResidualOutputCorrelation + 0.15, $"correlation {lossy.ResidualOutputCorrelation}");
        Assert.True(lossy.LagX > perfect.LagX + 0.05 && lossy.LagY > perfect.LagY + 0.05, $"neighbour correlation {lossy.LagX}, {lossy.LagY}");
        Assert.InRange(lossy.NetChange, 0.15, 0.25);

        // Nothing removed: nothing changed.
        var none = new ResidualCheck();
        none.Add(frame.Noisy, frame.Noisy);
        Assert.Equal((0.0, 0.0), (none.NetChange, none.RmsChange));
        Assert.Throws<ArgumentException>(() => none.Add(frame.Noisy, new GrayImage(95, 96)));
    }

    [Fact]
    public void TheReportPictureCentresTheResidualOnMidGrayAndClipsItAtThe99thPercentile()
    {
        // 200 pixels: residual 0 in the top half, ±1 alternating in the bottom half, and one hot pixel of +1000. The
        // 99th percentile of |residual| is 1, so ±1 spans the whole gray range; the hot pixel only saturates itself.
        const int w = 20, h = 10, gap = 4;
        var raw = new GrayImage(w, h);
        var denoised = new GrayImage(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                denoised.Pixels[i] = 100 + (i % 7);
                raw.Pixels[i] = denoised.Pixels[i] + (y < h / 2 ? 0 : (x % 2 == 0 ? 1 : -1));
            }
        raw.Pixels[0] += 1000;
        var png = PngDecoder.Decode(ResidualCheck.Picture(raw, denoised));
        Assert.Equal((3 * w + 2 * gap, h), (png.Width, png.Height));
        float Residual(int x, int y) => png.Pixels[y * png.Width + 2 * (w + gap) + x];
        Assert.Equal(128, Residual(5, 2));     // zero is mid-gray (127.5 rounds to even)
        Assert.Equal(255, Residual(4, 7));     // +1: DORA removed brightness here
        Assert.Equal(0, Residual(5, 7));       // −1: DORA added brightness here
        Assert.Equal(255, Residual(0, 0));     // the hot pixel, clipped
        Assert.Equal(255, png.Pixels[w]);      // the gap between panels is white
    }

    [Fact]
    public void ADenoiseReportIsOneSelfContainedPageAboutTheRun()
    {
        string dir = TestUtil.TempDirectory();
        try
        {
            // A real-world name with markup characters in it: escaped in the page, never markup.
            string input = Path.Combine(dir, "WT_GFP&RFP_#2.tif"), output = Path.Combine(dir, "out.tif"), report = Path.Combine(dir, "r.html");
            File.Copy(TestUtil.Fixture("imagej_calibrated_t3.tif"), input);
            var net = DenoisingNet.Create(new NetConfig(4, [1, 1]), seed: 3);
            var model = new LoadedModel(net, new ModelInfo(net.Config, "2.0.0-bio", "a test model <not real>", DateTime.UnixEpoch));
            var lines = new List<string>();
            Pipeline.Denoise(new DenoiseRequest { Input = input, Output = output, Model = model, Report = report, TileCore = 16 }, lines.Add);
            Assert.Contains(lines, l => l.StartsWith("📝 report → ", StringComparison.Ordinal));
            string html = File.ReadAllText(report);

            // Self-contained: the picture is embedded, and nothing is fetched (no external source, link or script).
            Assert.Contains("data:image/png;base64,", html);
            Assert.DoesNotMatch("(src|href)=\"(?!data:)", html);
            Assert.DoesNotContain("<script", html);
            Assert.DoesNotContain("<link", html);
            // What was done, to what, with what.
            Assert.Contains(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(input))).ToLowerInvariant(), html);
            Assert.Contains("16-bit unsigned integer, uncompressed", html);
            Assert.Contains("T=3", html);
            Assert.Contains(System.Net.WebUtility.HtmlEncode("0.65 × 0.65 um per pixel"), html);
            Assert.Contains("the ImageJ header, verbatim", html);
            Assert.Contains(WeightsFile.ContentHash(net), html);
            Assert.Contains(System.Net.WebUtility.HtmlEncode($"DORA {Pipeline.Version}; denoised single-frame"), html);
            Assert.Contains("a test model &lt;not real&gt;", html);   // text from the files is escaped, never markup
            Assert.Contains("WT_GFP&amp;RFP_#2.tif", html);
            Assert.DoesNotContain("GFP&RFP", html);

            // Numbers don't follow the machine's locale (a comma would be a decimal point in some labs' reports).
            var culture = System.Globalization.CultureInfo.CurrentCulture;
            try
            {
                System.Globalization.CultureInfo.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
                Pipeline.Denoise(new DenoiseRequest { Input = input, Output = output, Model = model, Report = report, TileCore = 16 });
            }
            finally
            {
                System.Globalization.CultureInfo.CurrentCulture = culture;
            }
            string german = File.ReadAllText(report);
            Assert.Matches(@"3 planes in \d+\.\d s", german);
            Assert.Matches(@"x [+-]\d\.\d{3}, y [+-]\d\.\d{3}", german);

            // A folder is refused before any work: one report describes one image or stack.
            Assert.Throws<ArgumentException>(() => Pipeline.Denoise(new DenoiseRequest
            {
                Input = dir, Output = Path.Combine(dir, "folder_out"), Model = model, Report = Path.Combine(dir, "x.html"),
            }));
            Assert.False(Directory.Exists(Path.Combine(dir, "folder_out")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void CancellingStopsTrainingWithinAStepAndDenoisingWithinATile()
    {
        // The app waits a few seconds for its job when it quits (DoraBioBridge._ExitTree): that only works if a cancel
        // lands within a step or a tile, not at the end of an epoch or a frame. An epoch of a million steps (hours) and a
        // 2048² frame through the U-Net (seconds) are each cancelled mid-flight and must stop promptly.
        var images = SyntheticFluorescence.GenerateDataset(4, 64, seed: 5).Select(p => p with { Clean = null }).ToList();
        string dir = TestUtil.TempDirectory();
        try
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            using (var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(500)))
            {
                var trained = Pipeline.Train(new TrainRequest
                {
                    Dataset = images, DatasetName = "test", Out = Path.Combine(dir, "w.bin"), Route = "off", Arch = "dilated",
                    Epochs = 1, Steps = 1_000_000, Batch = 2, Patch = 32,
                }, cancellation: cancel.Token);
                Assert.Equal("cancelled", Assert.Single(trained).Result.StopReason);
            }
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), $"training took {clock.Elapsed.TotalSeconds:0.0} s to stop");

            // Cancelled before training starts (while the data loads): nothing is trained.
            using (var already = new CancellationTokenSource())
            {
                already.Cancel();
                Assert.ThrowsAny<OperationCanceledException>(() => Pipeline.Train(new TrainRequest
                {
                    Dataset = images, DatasetName = "test", Out = Path.Combine(dir, "never.bin"), Epochs = 1, Steps = 1, Batch = 2, Patch = 32,
                }, cancellation: already.Token));
                Assert.False(File.Exists(Path.Combine(dir, "never.bin")));
            }

            string big = Path.Combine(dir, "big.tif");
            File.WriteAllBytes(big, TiffCodec.Encode(TestUtil.RandomImage(2048, 2048, 7), TiffSampleType.Float32));
            var unet = DenoisingNet.Create(NetConfig.UNet(16, 2), seed: 1);
            var model = new LoadedModel(unet, new ModelInfo(unet.Config, "2.0.0-bio", "test", DateTime.UnixEpoch));
            clock.Restart();
            using (var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(300)))
                Assert.ThrowsAny<OperationCanceledException>(() => Pipeline.Denoise(new DenoiseRequest
                {
                    Input = big, Output = Path.Combine(dir, "big_out.tif"), Model = model, TileCore = 64,
                }, cancellation: cancel.Token));
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(5), $"denoising took {clock.Elapsed.TotalSeconds:0.0} s to stop");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

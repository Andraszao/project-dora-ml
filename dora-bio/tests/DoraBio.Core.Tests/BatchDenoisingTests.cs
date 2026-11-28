using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 28: stacks and folders denoised one frame at a time. The streaming reader decodes every page exactly as the
/// whole-file decoder does; the streaming writer leaves a valid TIFF after every page; a page of a stack comes out
/// bit for bit as the same frame on its own; and no frame outlives the next one, so memory doesn't grow with the
/// number of pages.
/// </summary>
[Trait(Bricks.Trait, Bricks.CliBatch)]
public class BatchDenoisingTests
{
    private static DenoisingNet TinyNet() => DenoisingNet.Create(new NetConfig(4, [1, 2, 1]), seed: 3);

    private static List<GrayImage> Frames(int count, int width, int height) =>
        Enumerable.Range(0, count).Select(i => SyntheticFluorescence.Generate(width + 3 * i, height, 100 + i).Noisy).ToList();

    private static void AssertSame(GrayImage expected, GrayImage actual)
    {
        Assert.Equal((expected.Width, expected.Height), (actual.Width, actual.Height));
        Assert.Equal(expected.Pixels, actual.Pixels);
    }

    [Theory]
    [InlineData(TiffSampleType.UInt8, false)]
    [InlineData(TiffSampleType.UInt16, true)]
    [InlineData(TiffSampleType.Float32, false)]
    [InlineData(TiffSampleType.Float32, true)]
    public void StreamedPagesDecodeExactlyAsTheWholeFile(TiffSampleType type, bool deflate)
    {
        var bytes = TiffCodec.EncodeStack(Frames(3, 21, 13), type, deflate);
        using var reader = new TiffStackReader(new MemoryStream(bytes));
        Assert.Equal(3, reader.PageCount);
        for (int p = 0; p < 3; p++) AssertSame(TiffCodec.Decode(bytes, p), reader.ReadPage(p));
        Assert.Throws<ArgumentOutOfRangeException>(() => reader.ReadPage(3));
    }

    [Theory]
    [InlineData("stack2.tif")]                    // ImageMagick: an 8-bit page and a 16-bit page
    [InlineData("gray16_bigendian_strips.tif")]   // big-endian, several strips
    [InlineData("gray16_deflate_predictor.tif")]
    public void StreamedFixturesFromOtherWritersDecodeExactly(string fixture)
    {
        var bytes = File.ReadAllBytes(TestUtil.Fixture(fixture));
        using var reader = TiffStackReader.Open(TestUtil.Fixture(fixture));
        Assert.Equal(TiffCodec.CountPages(bytes), reader.PageCount);
        for (int p = 0; p < reader.PageCount; p++) AssertSame(TiffCodec.Decode(bytes, p), reader.ReadPage(p));
    }

    [Fact]
    public void EveryPrefixOfAStreamedStackIsAValidTiff()
    {
        var frames = Frames(4, 9, 7);
        var stream = new MemoryStream();
        using (var writer = new TiffStackWriter(stream, TiffSampleType.Float32, leaveOpen: true))
            for (int k = 0; k < frames.Count; k++)
            {
                writer.Append(frames[k]);
                var sofar = stream.ToArray();
                Assert.Equal(k + 1, TiffCodec.CountPages(sofar));
                AssertSame(frames[k], TiffCodec.Decode(sofar, k));
            }
        Assert.Equal(TiffCodec.EncodeStack(frames, TiffSampleType.Float32), stream.ToArray());
    }

    [Fact]
    public void WriterRefusesStreamsItCannotPatch()
    {
        Assert.Throws<ArgumentException>(() => new TiffStackWriter(new MemoryStream([1, 2, 3])));
        Assert.Throws<ArgumentException>(() => new TiffStackWriter(new MemoryStream(new byte[8], writable: false)));
    }

    [Fact]
    public void APageChainThatLoopsIsRejected()
    {
        var bytes = TiffCodec.EncodeStack(Frames(2, 8, 8));
        uint ifd0 = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4));
        uint ifd1 = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan((int)ifd0 + 2 + BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan((int)ifd0)) * 12));
        int tail1 = (int)ifd1 + 2 + BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan((int)ifd1)) * 12;
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(tail1), ifd0);   // page 2 points back to page 1
        Assert.Throws<InvalidDataException>(() => TiffCodec.CountPages(bytes));
        Assert.Throws<InvalidDataException>(() => new TiffStackReader(new MemoryStream(bytes)));
    }

    [Fact]
    public void EachPageOfAStackComesOutAsTheSameFrameOnItsOwn()
    {
        string dir = TestUtil.TempDirectory();
        var frames = Frames(3, 40, 33);
        File.WriteAllBytes(Path.Combine(dir, "stack.tif"), TiffCodec.EncodeStack(frames));
        var denoiser = new FrameDenoiser(TinyNet(), tileCore: 16);

        int pages = BatchDenoising.DenoiseFile(denoiser, Path.Combine(dir, "stack.tif"), Path.Combine(dir, "out.tif"), Path.Combine(dir, "base.tif"));
        Assert.Equal(3, pages);
        var output = File.ReadAllBytes(Path.Combine(dir, "out.tif"));
        var baseline = File.ReadAllBytes(Path.Combine(dir, "base.tif"));
        Assert.Equal(3, TiffCodec.CountPages(output));
        for (int p = 0; p < 3; p++)
        {
            string single = Path.Combine(dir, $"frame{p}.tif");
            File.WriteAllBytes(single, TiffCodec.Encode(frames[p]));
            BatchDenoising.DenoiseFile(denoiser, single, single + ".out.tif", single + ".base.tif");
            AssertSame(TiffCodec.Decode(File.ReadAllBytes(single + ".out.tif")), TiffCodec.Decode(output, p));
            AssertSame(TiffCodec.Decode(File.ReadAllBytes(single + ".base.tif")), TiffCodec.Decode(baseline, p));
            // ...and as the in-memory pipeline gives, so the file round trip loses nothing (float32 output).
            var direct = denoiser.Denoise(frames[p], withBaseline: true);
            AssertSame(direct.Output, TiffCodec.Decode(output, p));
            AssertSame(direct.Baseline!, TiffCodec.Decode(baseline, p));
        }
        Directory.Delete(dir, recursive: true);
    }

    [Fact]
    public void PngInputsDenoiseAsTheirDecodedFrame()
    {
        string dir = TestUtil.TempDirectory();
        var denoiser = new FrameDenoiser(TinyNet(), tileCore: 16);
        string png = TestUtil.Fixture("gray16.png");
        Assert.Equal(1, BatchDenoising.CountPages(png));
        Assert.Equal(1, BatchDenoising.DenoiseFile(denoiser, png, Path.Combine(dir, "out.tif")));
        AssertSame(denoiser.Denoise(ImageLoader.Load(png)).Output, TiffCodec.Decode(File.ReadAllBytes(Path.Combine(dir, "out.tif"))));
        Directory.Delete(dir, recursive: true);
    }

    [Fact]
    public void OutputsMayNotOverwriteTheInput()
    {
        string dir = TestUtil.TempDirectory();
        string input = Path.Combine(dir, "in.tif");
        var bytes = TiffCodec.Encode(Frames(1, 16, 16)[0]);
        File.WriteAllBytes(input, bytes);
        var denoiser = new FrameDenoiser(TinyNet());
        Assert.Throws<ArgumentException>(() => BatchDenoising.DenoiseFile(denoiser, input, input));
        Assert.Throws<ArgumentException>(() => BatchDenoising.DenoiseFile(denoiser, input, Path.Combine(dir, "out.tif"), Path.Combine(dir, ".", "in.tif")));
        Assert.Equal(bytes, File.ReadAllBytes(input));
        Assert.Throws<ArgumentException>(() => BatchDenoising.PlanFolder(dir, dir + Path.DirectorySeparatorChar));
        Directory.Delete(dir, recursive: true);
    }

    [Fact]
    public void FolderPlansAreOrderedAndRefuseClashes()
    {
        string dir = TestUtil.TempDirectory(), output = Path.Combine(dir, "out");
        foreach (var name in new[] { "b.tif", "a.TIFF", "notes.txt", "c.png" }) File.WriteAllBytes(Path.Combine(dir, name), [0]);
        Directory.CreateDirectory(Path.Combine(dir, "sub"));
        File.WriteAllBytes(Path.Combine(dir, "sub", "d.tif"), [0]);   // subfolders are not searched

        var plan = BatchDenoising.PlanFolder(dir, output);
        Assert.Equal(["a.TIFF", "b.tif", "c.png"], plan.Select(p => Path.GetFileName(p.Input)));
        Assert.Equal(["a.tif", "b.tif", "c.tif"], plan.Select(p => Path.GetFileName(p.Output)));
        Assert.All(plan, p => Assert.Equal(output, Path.GetDirectoryName(p.Output)));

        File.WriteAllBytes(Path.Combine(dir, "b.png"), [0]);   // b.png and b.tif would both write b.tif
        var clash = Assert.Throws<ArgumentException>(() => BatchDenoising.PlanFolder(dir, output));
        Assert.Contains("b.png", clash.Message);
        Assert.Throws<DirectoryNotFoundException>(() => BatchDenoising.PlanFolder(Path.Combine(dir, "missing"), output));
        Directory.CreateDirectory(Path.Combine(dir, "empty"));
        Assert.Throws<ArgumentException>(() => BatchDenoising.PlanFolder(Path.Combine(dir, "empty"), output));
        Directory.Delete(dir, recursive: true);
    }

    [Fact]
    public void CancellingMidStackLeavesTheFinishedPagesReadable()
    {
        string dir = TestUtil.TempDirectory();
        File.WriteAllBytes(Path.Combine(dir, "stack.tif"), TiffCodec.EncodeStack(Frames(5, 24, 24)));
        using var cancel = new CancellationTokenSource();
        Assert.ThrowsAny<OperationCanceledException>(() => BatchDenoising.DenoiseFile(new FrameDenoiser(TinyNet()),
            Path.Combine(dir, "stack.tif"), Path.Combine(dir, "out.tif"), onPage: (page, _, _) => { if (page == 1) cancel.Cancel(); },
            cancellation: cancel.Token));
        Assert.Equal(2, TiffCodec.CountPages(File.ReadAllBytes(Path.Combine(dir, "out.tif"))));
        Directory.Delete(dir, recursive: true);
    }

    [Fact]
    public void NoFrameOutlivesTheNextOne()
    {
        // The memory bound, checked directly rather than through noisy process statistics: weak references to every
        // page's input, output and baseline, and a full collection after each page. When page k finishes, nothing
        // from pages before k − 1 may still be reachable (in practice not even k − 1), so memory holds a bounded
        // number of frames however long the stack is.
        string dir = TestUtil.TempDirectory();
        const int pages = 8;
        File.WriteAllBytes(Path.Combine(dir, "stack.tif"), TiffCodec.EncodeStack(Frames(pages, 64, 48)));
        var tracked = new List<WeakReference>[pages];
        var leaked = new List<int>();

        BatchDenoising.DenoiseFile(new FrameDenoiser(TinyNet(), tileCore: 32), Path.Combine(dir, "stack.tif"),
            Path.Combine(dir, "out.tif"), Path.Combine(dir, "base.tif"), onPage: (page, raw, result) =>
            {
                tracked[page] = Track(raw, result);
                FullCollection();
                for (int earlier = 0; earlier < page - 1; earlier++)
                    if (tracked[earlier].Any(w => w.IsAlive)) leaked.Add(earlier);
            });
        FullCollection();
        Assert.Empty(leaked);
        Assert.All(tracked, refs => Assert.DoesNotContain(refs, w => w.IsAlive));
        Assert.Equal(pages, TiffCodec.CountPages(File.ReadAllBytes(Path.Combine(dir, "out.tif"))));
        Directory.Delete(dir, recursive: true);
    }

    // Not inlined, so no hidden temporary in the caller keeps the frames alive.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static List<WeakReference> Track(GrayImage raw, FrameResult result) =>
        [new(raw), new(raw.Pixels), new(result), new(result.Output.Pixels), new(result.Baseline!.Pixels)];

    private static void FullCollection()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}

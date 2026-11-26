using System.Buffers.Binary;
using System.Text.Json;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 34: the benchmark data's formats and layouts. Fixtures come from independent writers (tifffile for ImageJ
/// hyperstacks, JSON shapes and BigTIFF; mrcfile for MRC; see Fixtures/make_stack_fixtures.py). Each plane's pixels
/// encode its own position, v = 1000·t + 100·z + 10·c + 2·y + x + 1, so a plane read from the wrong page, axis order or
/// byte order can't pass. Gate 34 checks the real downloads with <c>stack-check</c>.
/// </summary>
[Trait(Bricks.Trait, Bricks.ExtData2)]
public class StackTests
{
    private const int W = 5, H = 4;

    private static void AssertPlane(GrayImage plane, int t = 0, int z = 0, int c = 0, float offset = 0)
    {
        Assert.Equal((W, H), (plane.Width, plane.Height));
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                Assert.Equal(1000 * t + 100 * z + 10 * c + 2 * y + x + 1 + offset, plane[x, y]);
    }

    [Fact]
    public void ImageJHyperstackPlanesComeBackAtTheirPositions()
    {
        using var stack = ImageStack.Open(TestUtil.Fixture("imagej_tzc.tif"));
        Assert.Equal(("TIFF", "TZC", StackMetadata.ImageJ), (stack.Format, stack.Layout.Axes, stack.Layout.Source));
        Assert.Equal([2, 3, 2], stack.Layout.Shape);
        Assert.Equal(12, stack.PlaneCount);
        for (int t = 0; t < 2; t++)
            for (int z = 0; z < 3; z++)
                for (int c = 0; c < 2; c++)
                    AssertPlane(stack.ReadPlane(stack.PlaneIndex(t: t, z: z, c: c)), t, z, c);
        var volume = stack.ReadVolume(t: 1, c: 1);
        Assert.Equal(3, volume.Length);
        for (int z = 0; z < 3; z++) AssertPlane(volume[z], 1, z, 1);
    }

    [Fact]
    public void TifffileShapeAndAxesReadFromJsonInBigEndianFloat()
    {
        using var stack = ImageStack.Open(TestUtil.Fixture("tifffile_tc_be.tif"));
        Assert.Equal(("TC", StackMetadata.Tifffile), (stack.Layout.Axes, stack.Layout.Source));
        Assert.Equal([3, 2], stack.Layout.Shape);
        for (int t = 0; t < 3; t++)
            for (int c = 0; c < 2; c++)
                AssertPlane(stack.ReadPlane(stack.PlaneIndex(t: t, c: c)), t: t, c: c, offset: 0.25f);
        Assert.Equal(3, stack.Frames(c: 1).Count());
    }

    [Fact]
    public void AShapeWithoutAxesIsUnlabelledUntilDeclared()
    {
        using (var stack = ImageStack.Open(TestUtil.Fixture("tifffile_noaxes.tif")))
        {
            Assert.Equal(("I", StackMetadata.Tifffile), (stack.Layout.Axes, stack.Layout.Source));
            Assert.Equal([5], stack.Layout.Shape);
        }
        using var declared = ImageStack.Open(TestUtil.Fixture("tifffile_noaxes.tif"), "ZYX");
        Assert.Equal("ZYX", declared.Layout.FullAxes);
        var volume = declared.ReadVolume();
        Assert.Equal(5, volume.Length);
        for (int z = 0; z < 5; z++) AssertPlane(volume[z], z: z);
    }

    [Fact]
    public void BigTiffReadsPageByPageFromDiskAndFromMemory()
    {
        string path = TestUtil.Fixture("bigtiff_t6_be.tif");
        using (var stack = ImageStack.Open(path))
        {
            Assert.Equal(("BigTIFF", "T"), (stack.Format, stack.Layout.Axes));
            Assert.Equal([6], stack.Layout.Shape);
            int t = 0;
            foreach (var frame in stack.Frames()) AssertPlane(frame, t: t++);
            Assert.Equal(6, t);
        }
        var bytes = File.ReadAllBytes(path);
        Assert.True(TiffCodec.HasTiffSignature(bytes));
        Assert.Equal(6, TiffCodec.CountPages(bytes));
        AssertPlane(TiffCodec.Decode(bytes, 4), t: 4);
        AssertPlane(ImageLoader.Load(path));                  // single-image loading takes page 0
        Assert.Equal(6, BatchDenoising.CountPages(path));     // and `denoise` streams BigTIFF stacks like any other
    }

    [Theory]
    [InlineData("mrc_z3_u16.mrc", true)]
    [InlineData("mrc_z3_i16_be.mrc", false)]
    public void MrcSectionsReadInEitherByteOrder(string file, bool littleEndian)
    {
        using (var mrc = MrcReader.Open(TestUtil.Fixture(file)))
        {
            Assert.Equal((W, H, 3, littleEndian), (mrc.Width, mrc.Height, mrc.PlaneCount, mrc.LittleEndian));
        }
        using var stack = ImageStack.Open(TestUtil.Fixture(file), "PYX");
        Assert.Equal(("MRC", "P", StackMetadata.Mrc), (stack.Format, stack.Layout.Axes, stack.Layout.Source));
        for (int p = 0; p < 3; p++) AssertPlane(stack.ReadPlane(stack.PlaneIndex(p: p)), z: p);
    }

    [Fact]
    public void TruncatedOrForeignMrcFilesAreRejected()
    {
        var bytes = File.ReadAllBytes(TestUtil.Fixture("mrc_z3_u16.mrc"));
        Assert.Throws<InvalidDataException>(() => new MrcReader(new MemoryStream(bytes[..^2])));   // last sample cut short
        var badMode = (byte[])bytes.Clone();
        BinaryPrimitives.WriteInt32LittleEndian(badMode.AsSpan(12), 4);                           // mode 4: complex
        Assert.Throws<NotSupportedException>(() => new MrcReader(new MemoryStream(badMode)));
        Assert.Throws<InvalidDataException>(() => new MrcReader(new MemoryStream(new byte[100])));
    }

    /// <summary>A minimal little-endian BigTIFF, written by hand: two 3×2 uint16 pages, every value inline.</summary>
    private static byte[] HandMadeBigTiff(ushort offsetSize = 8, bool loop = false)
    {
        const int entries = 8, ifdBytes = 8 + entries * 20 + 8, pixelBytes = 12;
        var file = new byte[16 + 2 * (pixelBytes + ifdBytes)];
        "II"u8.CopyTo(file);
        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(2), 43);
        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(4), offsetSize);
        long at = 16, previousNext = 8;
        for (int page = 0; page < 2; page++)
        {
            long strip = at;
            for (int i = 0; i < 6; i++) BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan((int)strip + 2 * i), (ushort)(100 * page + i));
            long ifd = strip + pixelBytes;
            BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan((int)previousNext), (ulong)ifd);
            BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan((int)ifd), entries);
            var tags = new (ushort Tag, ushort Type, ulong Value)[]
            {
                (256, 3, 3), (257, 3, 2), (258, 3, 16), (259, 3, 1), (262, 3, 1),
                (273, 16, (ulong)strip),          // StripOffsets as LONG8, inline in the 8-byte value field
                (278, 3, 2), (279, 16, pixelBytes),
            };
            for (int e = 0; e < entries; e++)
            {
                var entry = file.AsSpan((int)ifd + 8 + e * 20, 20);
                BinaryPrimitives.WriteUInt16LittleEndian(entry, tags[e].Tag);
                BinaryPrimitives.WriteUInt16LittleEndian(entry[2..], tags[e].Type);
                BinaryPrimitives.WriteUInt64LittleEndian(entry[4..], 1);
                if (tags[e].Type == 3) BinaryPrimitives.WriteUInt16LittleEndian(entry[12..], (ushort)tags[e].Value);
                else BinaryPrimitives.WriteUInt64LittleEndian(entry[12..], tags[e].Value);
            }
            previousNext = ifd + 8 + entries * 20;
            at = ifd + ifdBytes;
        }
        if (loop) BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan((int)previousNext), 16 + pixelBytes);   // back to page 0
        return file;
    }

    [Fact]
    public void HandMadeBigTiffDecodesAndCorruptionIsCaught()
    {
        var file = HandMadeBigTiff();
        Assert.Equal(2, TiffCodec.CountPages(file));
        Assert.Equal([0f, 1, 2, 3, 4, 5], TiffCodec.Decode(file, 0).Pixels);
        Assert.Equal([100f, 101, 102, 103, 104, 105], TiffCodec.Decode(file, 1).Pixels);
        using (var reader = new TiffStackReader(new MemoryStream(file)))
        {
            Assert.True(reader.IsBigTiff);
            Assert.Equal((3, 2), reader.PageSize(1));
            Assert.Equal([100f, 101, 102, 103, 104, 105], reader.ReadPage(1).Pixels);
            Assert.Null(reader.Description);
        }
        Assert.Throws<InvalidDataException>(() => TiffCodec.CountPages(HandMadeBigTiff(offsetSize: 4)));
        Assert.Throws<InvalidDataException>(() => TiffCodec.CountPages(HandMadeBigTiff(loop: true)));
        var outside = HandMadeBigTiff();
        BinaryPrimitives.WriteUInt64LittleEndian(outside.AsSpan(8), (ulong)outside.Length + 8);   // first IFD past the end
        Assert.Throws<InvalidDataException>(() => TiffCodec.CountPages(outside));
        var hugeCount = HandMadeBigTiff();
        BinaryPrimitives.WriteUInt64LittleEndian(hugeCount.AsSpan(16 + 12), ulong.MaxValue);       // entry count of page 0
        Assert.Throws<InvalidDataException>(() => TiffCodec.Decode(hugeCount));
    }

    [Theory]
    [InlineData("ImageJ=1.53c\nimages=6\nchannels=2\nslices=3\n", 6, "ZC", new[] { 3, 2 })]
    [InlineData("ImageJ=1.53c\nimages=5706\nframes=5706\ncf=0\nc0=-32768.0\n", 5706, "T", new[] { 5706 })]
    [InlineData("ImageJ=1.11a\nimages=16\nchannels=16\nhyperstack=true\n", 16, "C", new[] { 16 })]
    [InlineData("ImageJ=1.53c\nimages=24\nchannels=2\nslices=3\nframes=4\n", 24, "TZC", new[] { 4, 3, 2 })]
    [InlineData("{\"shape\": [48, 773, 665]}", 48, "I", new[] { 48 })]
    [InlineData("{\"shape\": [7, 2, 30, 40], \"axes\": \"TCYX\"}", 14, "TC", new[] { 7, 2 })]
    [InlineData("{\"shape\": [3, 30, 40, 3], \"axes\": \"ZYXS\"}", 3, "Z", new[] { 3 })]
    [InlineData(null, 9, "I", new[] { 9 })]
    [InlineData("a camera's free text", 1, "", new int[0])]
    public void LayoutComesFromTheDescription(string? description, int pages, string axes, int[] shape)
    {
        var layout = StackLayout.FromTiff(description, pages);
        Assert.Equal(axes, layout.Axes);
        Assert.Equal(shape, layout.Shape);
        Assert.Equal(pages, layout.PlaneCount);
    }

    [Fact]
    public void ContradictoryMetadataIsAnErrorNotAGuess()
    {
        Assert.Throws<InvalidDataException>(() => StackLayout.FromTiff("ImageJ=1.53c\nimages=7\nchannels=2\nslices=3\n", 7));
        Assert.Throws<InvalidDataException>(() => StackLayout.FromTiff("ImageJ=1.53c\nimages=6\nchannels=2\nslices=3\n", 5));
        Assert.Throws<NotSupportedException>(() => StackLayout.FromTiff("ImageJ=1.53c\nimages=6\nslices=6\n", 1));
        Assert.Throws<InvalidDataException>(() => StackLayout.FromTiff("{\"shape\": [4, 30, 40]}", 3));
        Assert.Throws<InvalidDataException>(() => StackLayout.FromTiff("{\"shape\": [4, 30, 40], \"axes\": \"ZYXQ\"}", 4));
        Assert.Throws<InvalidDataException>(() => StackLayout.FromTiff("{\"shape\": oops}", 4));
    }

    [Fact]
    public void DeclaredAxesMustFitTheFile()
    {
        var w2sLike = StackLayout.FromTiff("ImageJ=1.11a\nimages=360\nchannels=3\nslices=120\n", 360);
        Assert.Equal("ZC", w2sLike.Axes);                     // ImageJ calls W2S's fields of view "slices"
        var declared = w2sLike.Declare("SCYX");
        Assert.Equal(("SC", 120, 3), (declared.Axes, declared.Size('S'), declared.Size('C')));
        Assert.Equal(1, declared.Size('T'));
        Assert.Throws<InvalidDataException>(() => w2sLike.Declare("ZYX"));
        Assert.Throws<ArgumentException>(() => w2sLike.Declare("SC"));
        Assert.Throws<ArgumentException>(() => w2sLike.Declare("SSYX"));
        Assert.Throws<ArgumentException>(() => w2sLike.Declare("SQYX"));
        using var stack = ImageStack.Open(TestUtil.Fixture("imagej_tzc.tif"));
        Assert.Throws<ArgumentOutOfRangeException>(() => stack.PlaneIndex(t: 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => stack.PlaneIndex(s: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => stack.ReadPlane(12));
    }

    private static void WriteStack(string path, int planes, float value, int size = 8)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var pages = Enumerable.Range(0, planes).Select(p => new GrayImage(size, size, Enumerable.Repeat(value + p, size * size).ToArray())).ToList();
        File.WriteAllBytes(path, TiffCodec.EncodeStack(pages));
    }

    [Fact]
    public void PairsAreFoundByPatternAndPathSubstitution()
    {
        string root = TestUtil.TempDirectory();
        string trib = Path.Combine(root, "mdc25-tribolium");
        foreach (var name in new[] { "00010.tif", "00002.tif" })
        {
            WriteStack(Path.Combine(trib, "noisy", name), 16, 1);
            WriteStack(Path.Combine(trib, "gt", name), 16, 50);
        }
        WriteStack(Path.Combine(trib, "noisy", "deeper", "ignored.tif"), 16, 1);   // '*' stays within one folder
        var dataset = ExtStacks.Get("mdc25-tribolium");
        var pairs = ExtStacks.Plan(root, dataset);
        Assert.Equal(["00002.tif", "00010.tif"], pairs.Select(p => Path.GetFileName(p.Noisy)));
        Assert.All(pairs, p => Assert.Equal(Path.Combine(trib, "gt", Path.GetFileName(p.Noisy)), p.Clean));
        var summary = Assert.Single(ExtStacks.Inspect(root, dataset));
        Assert.Equal((2, 32L), (summary.Files, summary.Planes));
        Assert.Equal(["8×8"], summary.PlaneSizes);

        WriteStack(Path.Combine(trib, "gt", "00002.tif"), 15, 50);                 // a partner with a different shape
        Assert.Throws<InvalidDataException>(() => ExtStacks.Inspect(root, dataset));
        File.Delete(Path.Combine(trib, "gt", "00002.tif"));
        Assert.Throws<InvalidDataException>(() => ExtStacks.Plan(root, dataset));  // and none at all

        string cad = Path.Combine(root, "deepcad-neutrophils");
        WriteStack(Path.Combine(cad, "01_n_6x6x3_lowSNR.tif"), 3, 1);
        WriteStack(Path.Combine(cad, "01_n_6x6x3_highSNR.tif"), 3, 9);
        var frames = Assert.Single(ExtStacks.Plan(root, ExtStacks.Get("deepcad-neutrophils")));
        Assert.EndsWith("01_n_6x6x3_highSNR.tif", frames.Clean);

        string sr = Path.Combine(root, "biosr-ccps", "CCPs", "Cell_001", "Level_01");
        Directory.CreateDirectory(sr);
        File.Copy(TestUtil.Fixture("mrc_z3_u16.mrc"), Path.Combine(sr, "RawSIMData.mrc"));
        File.Copy(TestUtil.Fixture("mrc_z3_i16_be.mrc"), Path.Combine(sr, "RawSIMData_gt.mrc"));
        var sim = Assert.Single(ExtStacks.Inspect(root, ExtStacks.Get("biosr-ccps")));
        Assert.Equal((1, 3L, "MRC"), (sim.Files, sim.Planes, sim.Formats.Single()));
    }

    [Theory]
    [InlineData("noisy/*.tif", "noisy/a.tif", true)]
    [InlineData("noisy/*.tif", "noisy/sub/a.tif", false)]
    [InlineData("noisy/*.tif", "xnoisy/a.tif", false)]
    [InlineData("*/Cell_*/Level_*/RawSIMData.mrc", "ER/Cell_007/Level_03/RawSIMData.mrc", true)]
    [InlineData("*/Cell_*/Level_*/RawSIMData.mrc", "ER/Cell_007/Level_03/RawSIMData_gt.mrc", false)]
    [InlineData("*_lowSNR.tif", "01_x_lowSNR.tif", true)]
    [InlineData("noisy.tiff", "noisy.tiff", true)]
    public void GlobsMatchWithinOneFolder(string pattern, string path, bool matches) =>
        Assert.Equal(matches, ExtStacks.Glob(pattern).IsMatch(path));

    [Fact]
    public void RegistriesAgreeWithTheManifestAndItsLicencePolicy()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "ext_manifest.json")));
        var datasets = doc.RootElement.GetProperty("datasets").EnumerateArray().ToList();
        string Str(JsonElement d, string key) => d.GetProperty(key).GetString()!;

        var downloadable = datasets.Where(d => !d.TryGetProperty("fetched_by", out _)).Select(d => Str(d, "name")).ToList();
        var registered = ExtDatasets.All.Select(d => d.Name).Concat(ExtStacks.All.Select(d => d.Name)).ToList();
        Assert.Equal(downloadable.Order(), registered.Order());                    // each in exactly one registry
        Assert.Equal(registered.Count, registered.Distinct().Count());

        foreach (var d in datasets)
        {
            string name = Str(d, "name"), licence = Str(d, "licence"), use = Str(d, "use");
            if (use == "train") Assert.True(licence is "CC-BY-4.0" or "CC0-1.0", $"{name}: {licence} data can't train a shipped model");
            else Assert.Equal("eval-only", use);
            foreach (var f in d.GetProperty("files").EnumerateArray())
            {
                Assert.Matches("^[0-9a-f]{32}$", Str(f, "md5"));
                Assert.True(f.GetProperty("size").GetInt64() > 0);
                if (Str(d, "tier") == "core") Assert.Matches("^[0-9a-f]{64}$", Str(f, "sha256"));   // the default fetch is fully pinned
            }
        }
        var fmd = datasets.Single(d => Str(d, "name") == "fmd");
        Assert.Equal(("CC-BY-SA-4.0", "eval-only"), (Str(fmd, "licence"), Str(fmd, "use")));
        Assert.Equal(["deepbacs-bsubtilis", "deepbacs-mreb", "deepbacs-nucleoid", "mdc24-w2s", "zerocost-care2d"],
            datasets.Where(d => Str(d, "tier") == "core").Select(d => Str(d, "name")).Order());
    }
}

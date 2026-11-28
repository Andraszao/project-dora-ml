using System.Buffers.Binary;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Multi-page TIFF writing: every sample type, with and without Deflate, round-trips through the reader page by page,
/// and a one-page stack is the single-page file. (Gate 26 also checks a written stack with Pillow, an independent reader.)
/// </summary>
[Trait(Bricks.Trait, Bricks.TiffStackWrite)]
public class TiffStackTests
{
    private static List<GrayImage> Pages(int count, int width, int height, float scale) =>
        Enumerable.Range(0, count).Select(p => new GrayImage(width, height,
            TestUtil.RandomPlane(width * height, 10 + p).Select(v => v * scale).ToArray())).ToList();

    [Theory]
    [InlineData(TiffSampleType.UInt8, false)]
    [InlineData(TiffSampleType.UInt8, true)]
    [InlineData(TiffSampleType.UInt16, false)]
    [InlineData(TiffSampleType.UInt16, true)]
    [InlineData(TiffSampleType.Float32, false)]
    [InlineData(TiffSampleType.Float32, true)]
    public void ThreePageStacksRoundTripInEverySampleType(TiffSampleType type, bool deflate)
    {
        float scale = type switch { TiffSampleType.UInt8 => 255, TiffSampleType.UInt16 => 60000, _ => 3.5f };
        var pages = Pages(3, 23, 17, scale);
        var bytes = TiffCodec.EncodeStack(pages, type, deflate);

        Assert.Equal(3, TiffCodec.CountPages(bytes));
        for (int p = 0; p < 3; p++)
        {
            var decoded = TiffCodec.Decode(bytes, p);
            Assert.Equal((23, 17), (decoded.Width, decoded.Height));
            for (int i = 0; i < decoded.Length; i++)
            {
                float expected = type == TiffSampleType.Float32 ? pages[p].Pixels[i] : MathF.Round(pages[p].Pixels[i]);
                Assert.Equal(expected, decoded.Pixels[i]);
            }
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => TiffCodec.Decode(bytes, 3));
    }

    [Fact]
    public void PagesMayDifferInSize()
    {
        var pages = new List<GrayImage> { Pages(1, 8, 4, 100)[0], Pages(1, 5, 9, 100)[0] };
        var bytes = TiffCodec.EncodeStack(pages);
        Assert.Equal((8, 4), (TiffCodec.Decode(bytes, 0).Width, TiffCodec.Decode(bytes, 0).Height));
        Assert.Equal((5, 9), (TiffCodec.Decode(bytes, 1).Width, TiffCodec.Decode(bytes, 1).Height));
    }

    [Fact]
    public void AOnePageStackIsTheSinglePageLayout()
    {
        // Header | strip at offset 8 | IFD on a word boundary, whose next-IFD offset is 0: what Encode always wrote.
        var image = Pages(1, 3, 3, 1000)[0];   // 9 UInt16 samples = 18 bytes, so the IFD starts at 26
        var single = TiffCodec.Encode(image, TiffSampleType.UInt16);
        Assert.Equal(single, TiffCodec.EncodeStack([image], TiffSampleType.UInt16));
        Assert.Equal(26u, BinaryPrimitives.ReadUInt32LittleEndian(single.AsSpan(4)));
        int entries = BinaryPrimitives.ReadUInt16LittleEndian(single.AsSpan(26));
        Assert.Equal(0u, BinaryPrimitives.ReadUInt32LittleEndian(single.AsSpan(26 + 2 + entries * 12)));
        Assert.Equal(26 + 2 + entries * 12 + 4, single.Length);

        var odd = Pages(1, 3, 3, 200)[0];      // 9 UInt8 samples: the IFD needs one pad byte, at 18
        Assert.Equal(18u, BinaryPrimitives.ReadUInt32LittleEndian(TiffCodec.Encode(odd, TiffSampleType.UInt8).AsSpan(4)));
    }

    [Fact]
    public void AnEmptyStackIsRefused()
    {
        Assert.Throws<ArgumentException>(() => TiffCodec.EncodeStack([]));
    }
}

using System.Buffers.Binary;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Decoders are checked against files written by Pillow, ImageMagick and libtiff (Fixtures/make_fixtures.py).
/// Every fixture is 37×23 with a known per-pixel formula.
/// </summary>
public class CodecTests
{
    private const int W = 37, H = 23;

    private static float Gray8(int x, int y) => (x * 7 + y * 13) % 256;
    private static float Gray16(int x, int y) => (x * 1723 + y * 4099) % 65536;
    private static float Rgb8Mean(int x, int y) => ((x * 5) % 256 + (y * 11) % 256 + ((x + y) * 3) % 256) / 3f;
    private static float Float32(int x, int y) => (x - 18f) * 0.25f + y * y * 0.125f - 3.5f;

    private static void AssertImage(GrayImage image, Func<int, int, float> expected, float tolerance = 0f)
    {
        Assert.Equal(W, image.Width);
        Assert.Equal(H, image.Height);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                Assert.True(Math.Abs(image[x, y] - expected(x, y)) <= tolerance,
                    $"pixel ({x},{y}) = {image[x, y]}, expected {expected(x, y)}");
    }

    private static GrayImage LoadFixture(string name) => ImageLoader.Load(TestUtil.Fixture(name));

    [Theory]
    [InlineData("gray8.png")]
    [InlineData("gray_alpha8.png")]
    [InlineData("gray8.tif")]
    public void Decodes8BitGray(string file) => AssertImage(LoadFixture(file), Gray8);

    [Theory]
    [InlineData("gray16.png")]
    [InlineData("gray16_allfilters.png")]
    [InlineData("gray16.tif")]
    [InlineData("gray16_deflate.tif")]
    [InlineData("gray16_packbits.tif")]
    [InlineData("gray16_bigendian_strips.tif")]
    [InlineData("gray16_deflate_predictor.tif")]
    public void Decodes16BitGray(string file) => AssertImage(LoadFixture(file), Gray16);

    [Theory]
    [InlineData("rgb8.png")]
    [InlineData("rgb8.tif")]
    public void AveragesRgbToGray(string file) => AssertImage(LoadFixture(file), Rgb8Mean, 1e-4f);

    [Fact]
    public void DecodesFloat32Tiff() => AssertImage(LoadFixture("float32.tif"), Float32);

    [Fact]
    public void ReadsEveryPageOfAStack()
    {
        var bytes = File.ReadAllBytes(TestUtil.Fixture("stack2.tif"));
        Assert.Equal(2, TiffCodec.CountPages(bytes));
        // ImageMagick rescales the 8-bit page to 16 bits (×257).
        AssertImage(TiffCodec.Decode(bytes, page: 0), (x, y) => Gray8(x, y) * 257f);
        AssertImage(TiffCodec.Decode(bytes, page: 1), Gray16);
        Assert.Throws<ArgumentOutOfRangeException>(() => TiffCodec.Decode(bytes, page: 2));
    }

    [Fact]
    public void LzwIsRejectedWithAnActionableMessage()
    {
        var error = Assert.Throws<NotSupportedException>(() => LoadFixture("gray16_lzw.tif"));
        Assert.Contains("Deflate", error.Message);
    }

    [Fact]
    public void Hdf5IsRejectedClearly()
    {
        var bytes = new byte[] { 0x89, (byte)'H', (byte)'D', (byte)'F', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0 };
        var error = Assert.Throws<NotSupportedException>(() => ImageLoader.Decode(bytes, "data.h5"));
        Assert.Contains("HDF5", error.Message);
    }

    [Fact]
    public void CorruptPngCrcIsDetected()
    {
        var bytes = File.ReadAllBytes(TestUtil.Fixture("gray8.png"));
        bytes[^20] ^= 0xFF;   // inside the IDAT chunk
        Assert.Throws<InvalidDataException>(() => PngDecoder.Decode(bytes));
    }

    [Theory]
    [InlineData(TiffSampleType.UInt8, false)]
    [InlineData(TiffSampleType.UInt16, false)]
    [InlineData(TiffSampleType.UInt16, true)]
    [InlineData(TiffSampleType.Float32, false)]
    [InlineData(TiffSampleType.Float32, true)]
    public void TiffRoundTrips(TiffSampleType type, bool deflate)
    {
        var image = type switch
        {
            TiffSampleType.UInt8 => TestUtil.Pattern(W, H, Gray8),
            TiffSampleType.UInt16 => TestUtil.Pattern(W, H, Gray16),
            _ => TestUtil.Pattern(W, H, Float32),
        };
        var decoded = TiffCodec.Decode(TiffCodec.Encode(image, type, deflate));
        Assert.Equal(image.Pixels, decoded.Pixels);
    }

    [Fact]
    public void PackBitsHandlesLiteralAndRepeatRuns()
    {
        // Example from the TIFF 6.0 spec, section 9.
        byte[] packed = [0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7, 0xAA];
        byte[] expected = [0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22,
                           0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA];
        Assert.Equal(expected, TiffCodec.UnpackBits(packed, expected.Length));
    }

    [Fact]
    public void PaethMatchesSpecTieBreaking()
    {
        Assert.Equal(10, PngDecoder.Paeth(10, 10, 10));
        Assert.Equal(20, PngDecoder.Paeth(10, 20, 10));   // p = 20 → b is closest
        Assert.Equal(10, PngDecoder.Paeth(10, 20, 20));   // p = 10 → a
        Assert.Equal(20, PngDecoder.Paeth(10, 20, 5));    // p = 25: distances 15, 5, 20 → b
        Assert.Equal(0, PngDecoder.Paeth(0, 20, 30));     // p = −10: distances 10, 30, 40 → a
        Assert.Equal(15, PngDecoder.Paeth(20, 10, 15));   // p = 15: distances 5, 5, 0 → c
    }

    [Fact]
    public void WeightsRoundTripExactly()
    {
        var net = DenoisingNet.Create(new NetConfig(5, [1, 2, 1]), seed: 3);
        var bytes = WeightsFile.Serialize(net, "unit test", new DateTime(2026, 1, 2, 3, 4, 5, DateTimeKind.Utc));
        var (loaded, info) = WeightsFile.Load(bytes);

        Assert.Equal(net.Config, loaded.Config);
        Assert.Equal(net.Parameters, loaded.Parameters);
        Assert.Equal("unit test", info.Provenance);
        Assert.Equal(WeightsFile.ModelVersion, info.ModelVersion);
        Assert.Equal(new DateTime(2026, 1, 2, 3, 4, 5, DateTimeKind.Utc), info.SavedUtc);
        Assert.Equal(8 + 2 + 1 + 9 + 1 + 9 + 8 + 4 + 4 + 12 + 4 + net.ParameterCount * 4 + 4, bytes.Length);
    }

    [Fact]
    public void WeightsCorruptionAndV1FilesAreRejected()
    {
        var net = DenoisingNet.Create(new NetConfig(3, [1, 1]), seed: 3);
        var bytes = WeightsFile.Serialize(net, "x", DateTime.UtcNow);
        bytes[bytes.Length / 2] ^= 0x01;
        Assert.Contains("CRC", Assert.Throws<InvalidDataException>(() => WeightsFile.Load(bytes)).Message);

        var v1Json = "{\"model_version\":\"1.0.0\",\"weights_input_hidden\":[]}"u8.ToArray();
        Assert.Contains("v1", Assert.Throws<InvalidDataException>(() => WeightsFile.Load(v1Json)).Message);
    }

    [Fact]
    public void WeightsFileIsLittleEndianFloat()
    {
        var net = DenoisingNet.Create(new NetConfig(3, [1, 1]), seed: 3);
        net.Parameters[0] = 1.5f;
        var bytes = WeightsFile.Serialize(net, "", DateTime.UnixEpoch);
        int paramStart = bytes.Length - 4 - net.ParameterCount * 4;
        Assert.Equal(1.5f, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(paramStart)));
    }
}

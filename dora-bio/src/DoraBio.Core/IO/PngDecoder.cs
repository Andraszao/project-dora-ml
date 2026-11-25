using System.Buffers.Binary;
using System.IO.Compression;
using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>
/// Dependency-free PNG decoder producing grayscale sample values. The FMD fluorescence dataset
/// (Zhang et al., CVPR 2019) ships as PNG.
/// Supports 8- and 16-bit gray, gray+alpha, RGB and RGBA (color is averaged to gray, alpha is ignored),
/// all five scanline filters, and CRC checks on every chunk.
/// Not supported: interlacing, palettes, bit depths below 8.
/// </summary>
public static class PngDecoder
{
    private static ReadOnlySpan<byte> Signature => [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    public static bool HasPngSignature(ReadOnlySpan<byte> data) => data.Length >= 8 && data[..8].SequenceEqual(Signature);

    public static GrayImage Decode(ReadOnlySpan<byte> data)
    {
        if (!HasPngSignature(data)) throw new InvalidDataException("Not a PNG file.");

        int width = 0, height = 0, bitDepth = 0, colorType = -1;
        using var idat = new MemoryStream();
        int pos = 8;
        bool sawEnd = false;

        while (pos + 12 <= data.Length && !sawEnd)
        {
            int length = checked((int)BinaryPrimitives.ReadUInt32BigEndian(data[pos..]));
            if (pos + 12 + length > data.Length) throw new InvalidDataException("PNG chunk runs past the end of the file.");
            var typeAndData = data.Slice(pos + 4, 4 + length);
            var chunkData = typeAndData[4..];
            uint storedCrc = BinaryPrimitives.ReadUInt32BigEndian(data[(pos + 8 + length)..]);
            if (Crc32.Compute(typeAndData) != storedCrc)
                throw new InvalidDataException($"PNG chunk '{System.Text.Encoding.ASCII.GetString(typeAndData[..4])}' failed its CRC check.");

            switch (System.Text.Encoding.ASCII.GetString(typeAndData[..4]))
            {
                case "IHDR":
                    width = checked((int)BinaryPrimitives.ReadUInt32BigEndian(chunkData));
                    height = checked((int)BinaryPrimitives.ReadUInt32BigEndian(chunkData[4..]));
                    bitDepth = chunkData[8];
                    colorType = chunkData[9];
                    if (chunkData[12] != 0) throw new NotSupportedException("Interlaced PNG is not supported.");
                    break;
                case "IDAT":
                    idat.Write(chunkData);
                    break;
                case "IEND":
                    sawEnd = true;
                    break;
            }
            pos += 12 + length;
        }

        if (width <= 0 || height <= 0) throw new InvalidDataException("PNG has no valid IHDR chunk.");
        if (colorType == 3) throw new NotSupportedException("Palette PNG is not supported.");
        int channels = colorType switch
        {
            0 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            _ => throw new InvalidDataException($"Unknown PNG color type {colorType}."),
        };
        if (bitDepth is not (8 or 16)) throw new NotSupportedException($"{bitDepth}-bit PNG is not supported.");

        int bytesPerSample = bitDepth / 8;
        int bytesPerPixel = channels * bytesPerSample;
        int rowBytes = width * bytesPerPixel;
        byte[] filtered = Inflate(idat.ToArray(), (rowBytes + 1) * height);
        byte[] pixels = Unfilter(filtered, height, rowBytes, bytesPerPixel);

        int colorChannels = colorType is 2 or 6 ? 3 : 1;
        var image = new GrayImage(width, height);
        for (int i = 0; i < width * height; i++)
        {
            int baseOffset = i * bytesPerPixel;
            double sum = 0;
            for (int c = 0; c < colorChannels; c++)
            {
                int o = baseOffset + c * bytesPerSample;
                sum += bytesPerSample == 1 ? pixels[o] : BinaryPrimitives.ReadUInt16BigEndian(pixels.AsSpan(o));
            }
            image.Pixels[i] = (float)(sum / colorChannels);
        }
        return image;
    }

    internal static byte[] Unfilter(byte[] filtered, int height, int rowBytes, int bytesPerPixel)
    {
        var output = new byte[height * rowBytes];
        for (int y = 0; y < height; y++)
        {
            int filterType = filtered[y * (rowBytes + 1)];
            var src = filtered.AsSpan(y * (rowBytes + 1) + 1, rowBytes);
            var cur = output.AsSpan(y * rowBytes, rowBytes);
            var prev = y > 0 ? output.AsSpan((y - 1) * rowBytes, rowBytes) : Span<byte>.Empty;

            for (int i = 0; i < rowBytes; i++)
            {
                int a = i >= bytesPerPixel ? cur[i - bytesPerPixel] : 0;
                int b = y > 0 ? prev[i] : 0;
                int c = y > 0 && i >= bytesPerPixel ? prev[i - bytesPerPixel] : 0;
                int predictor = filterType switch
                {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => (a + b) >> 1,
                    4 => Paeth(a, b, c),
                    _ => throw new InvalidDataException($"Unknown PNG filter type {filterType} on row {y}."),
                };
                cur[i] = (byte)(src[i] + predictor);
            }
        }
        return output;
    }

    internal static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        return pb <= pc ? b : c;
    }

    private static byte[] Inflate(byte[] compressed, int expectedLength)
    {
        using var input = new MemoryStream(compressed);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        var output = new byte[expectedLength];
        int read = 0;
        while (read < expectedLength)
        {
            int n = zlib.Read(output, read, expectedLength - read);
            if (n == 0) throw new InvalidDataException("PNG image data is truncated.");
            read += n;
        }
        return output;
    }
}

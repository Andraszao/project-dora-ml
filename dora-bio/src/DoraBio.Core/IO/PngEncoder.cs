using System.Buffers.Binary;
using System.IO.Compression;

namespace DoraBio.Core.IO;

/// <summary>
/// Writes 8-bit grayscale PNG (brick 41: the picture in a denoise report). Display only: the pixels are already
/// windowed to 0..255, so this is never a data path (rule 3). Rows use filter 0 (none); the whole image is one zlib
/// stream in one IDAT chunk, which every PNG reader accepts.
/// </summary>
public static class PngEncoder
{
    private static readonly byte[] Signature = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>Encodes <paramref name="pixels"/> (row-major, one byte per pixel) as a grayscale PNG.</summary>
    /// <exception cref="ArgumentException">The sizes don't match the pixel count.</exception>
    public static byte[] EncodeGray8(ReadOnlySpan<byte> pixels, int width, int height)
    {
        if (width < 1 || height < 1 || (long)width * height != pixels.Length)
            throw new ArgumentException($"A {width}×{height} PNG needs {(long)width * height} pixels, not {pixels.Length}.");
        var filtered = new byte[(long)(width + 1) * height];
        for (int y = 0; y < height; y++) pixels.Slice(y * width, width).CopyTo(filtered.AsSpan(y * (width + 1) + 1));   // filter byte 0

        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true)) zlib.Write(filtered);

        using var png = new MemoryStream();
        png.Write(Signature);
        var header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
        header[8] = 8;    // bit depth
        header[9] = 0;    // grayscale; compression, filter and interlace methods 0
        Chunk(png, "IHDR", header);
        Chunk(png, "IDAT", compressed.ToArray());
        Chunk(png, "IEND", []);
        return png.ToArray();
    }

    private static void Chunk(Stream png, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> word = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(word, data.Length);
        png.Write(word);
        var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);
        png.Write(typeBytes);
        png.Write(data);
        uint crc = Crc32.Update(Crc32.Update(0xFFFFFFFFu, typeBytes), data) ^ 0xFFFFFFFFu;
        BinaryPrimitives.WriteUInt32BigEndian(word, crc);
        png.Write(word);
    }
}

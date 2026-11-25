using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>Picks a decoder from the file's magic bytes, not its extension.</summary>
public static class ImageLoader
{
    private static readonly string[] Extensions = [".tif", ".tiff", ".png"];

    public static IReadOnlyList<string> SupportedExtensions => Extensions;

    public static bool IsSupported(string path) =>
        Extensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    public static GrayImage Load(string path) => Decode(File.ReadAllBytes(path), path);

    /// <param name="nameHint">Only used in error messages.</param>
    public static GrayImage Decode(ReadOnlySpan<byte> data, string? nameHint = null)
    {
        if (TiffCodec.HasTiffSignature(data)) return TiffCodec.Decode(data);
        if (PngDecoder.HasPngSignature(data)) return PngDecoder.Decode(data);

        string name = nameHint ?? "input";
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == (byte)'H' && data[2] == (byte)'D' && data[3] == (byte)'F')
            throw new NotSupportedException($"{name}: HDF5 is not supported yet. Export frames as TIFF or PNG.");
        throw new NotSupportedException($"{name}: unrecognized image format. Supported: 8/16/32-bit TIFF and PNG.");
    }
}

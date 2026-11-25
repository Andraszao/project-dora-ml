using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>Picks a decoder from the file's magic bytes, not its extension.</summary>
/// <remarks>
/// Sniffing the content means a mislabeled file (a PNG saved as ".tif") still loads. A truly unsupported
/// format fails with a message that says what to do, rather than a confusing parse error deep in a decoder.
/// The Godot bridge uses <see cref="Decode"/> on bytes from Godot's FileAccess, so res:// paths work too.
/// </remarks>
public static class ImageLoader
{
    private static readonly string[] Extensions = [".tif", ".tiff", ".png"];

    /// <summary>File extensions the dataset loader picks up when scanning folders (lower case, with dot).</summary>
    public static IReadOnlyList<string> SupportedExtensions => Extensions;

    /// <summary>True if <paramref name="path"/> has a supported extension (case-insensitive). Used to filter folder listings.</summary>
    public static bool IsSupported(string path) =>
        Extensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    /// <summary>Reads and decodes a file from disk. Returns raw sample values (no normalization).</summary>
    public static GrayImage Load(string path) => Decode(File.ReadAllBytes(path), path);

    /// <summary>Decodes an in-memory TIFF or PNG file. Returns raw sample values (no normalization).</summary>
    /// <param name="data">The complete file contents.</param>
    /// <param name="nameHint">Only used in error messages.</param>
    /// <exception cref="NotSupportedException">Unknown format, HDF5, or an unsupported TIFF/PNG variant.</exception>
    public static GrayImage Decode(ReadOnlySpan<byte> data, string? nameHint = null)
    {
        if (TiffCodec.HasTiffSignature(data)) return TiffCodec.Decode(data);
        if (PngDecoder.HasPngSignature(data)) return PngDecoder.Decode(data);

        string name = nameHint ?? "input";
        // HDF5 signature: \x89 'H' 'D' 'F' \r \n \x1a \n. Named explicitly because the original v2 plan assumed .h5 input.
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == (byte)'H' && data[2] == (byte)'D' && data[3] == (byte)'F')
            throw new NotSupportedException($"{name}: HDF5 is not supported yet. Export frames as TIFF or PNG.");
        throw new NotSupportedException($"{name}: unrecognized image format. Supported: 8/16/32-bit TIFF and PNG.");
    }
}

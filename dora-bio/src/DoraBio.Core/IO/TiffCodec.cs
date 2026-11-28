using System.Buffers.Binary;
using System.IO.Compression;
using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>Sample format used when writing a TIFF.</summary>
public enum TiffSampleType
{
    /// <summary>8-bit unsigned integer; values are rounded and clamped to 0–255.</summary>
    UInt8,

    /// <summary>16-bit unsigned integer, the usual raw-camera format; values are rounded and clamped to 0–65535.</summary>
    UInt16,

    /// <summary>32-bit IEEE float, stored exactly. Use for denoised output and anything normalized.</summary>
    Float32,
}

/// <summary>
/// Dependency-free baseline TIFF codec for grayscale scientific images.
///
/// Reads: classic TIFF and BigTIFF (files over 4 GB), little- or big-endian; strips (not tiles); 8/16/32-bit
/// unsigned or signed integers and 32-bit floats; uncompressed, LZW (5, brick 41), Deflate (8 / 32946) or PackBits
/// compression; horizontal predictor (2) for integers; BlackIsZero, WhiteIsZero and RGB (averaged to gray); any page
/// of a multi-page stack, its ImageDescription text (where ImageJ and tifffile record a stack's shape), pixel size and
/// Software tag. Not supported (a clear <see cref="NotSupportedException"/> naming the codec and a plain conversion is
/// thrown): Zstandard, JPEG, JPEG 2000 and the other compressions, old-style (pre-6.0) LZW, tiles, palettes, planar
/// RGB. Most microscope software can save uncompressed, LZW or Deflate TIFF.
///
/// Writes: little-endian, one strip per page, uncompressed or Deflate, UInt8/UInt16/Float32; single pages or
/// multi-page stacks (<see cref="EncodeStack"/>), with an optional ImageDescription, Software tag and pixel size
/// (<see cref="TiffPageTags"/>).
///
/// <para><b>TIFF in one paragraph.</b> An 8-byte header gives the byte order ("II" = little-endian,
/// "MM" = big-endian), the magic number 42, and the offset of the first IFD (image file directory). An IFD
/// is a count followed by 12-byte entries (tag, type, value count, then the value itself if it fits in 4 bytes,
/// otherwise an offset to it), and then the offset of the next IFD (0 = last page). The pixel data lives in
/// "strips" of whole rows, located by the StripOffsets and StripByteCounts tags. Everything else
/// (bit depth, compression, sample format) is just more tags.</para>
/// </summary>
public static class TiffCodec
{
    // Tag numbers from the TIFF 6.0 specification (and the SampleFormat extension).
    private const ushort TagImageWidth = 256, TagImageLength = 257, TagBitsPerSample = 258, TagCompression = 259,
        TagPhotometric = 262, TagStripOffsets = 273, TagSamplesPerPixel = 277, TagRowsPerStrip = 278,
        TagStripByteCounts = 279, TagPlanarConfig = 284, TagPredictor = 317, TagTileWidth = 322, TagSampleFormat = 339,
        TagImageDescription = 270, TagXResolution = 282, TagYResolution = 283, TagResolutionUnit = 296, TagSoftware = 305;

    // Compression codes. 32946 is the older "Adobe Deflate" code, still written by some tools; the data is identical.
    private const ushort CompressionNone = 1, CompressionLzw = 5, CompressionDeflate = 8,
        CompressionPackBits = 32773, CompressionAdobeDeflate = 32946;

    /// <summary>True if the data starts with a TIFF header (classic, magic 42, or BigTIFF, magic 43) in either byte order.</summary>
    public static bool HasTiffSignature(ReadOnlySpan<byte> data) =>
        data.Length >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] is 42 or 43 && data[3] == 0) ||
                             (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] is 42 or 43));

    /// <summary>Number of pages (IFDs) in the file. Microscope stacks store one frame or z-slice per page.</summary>
    public static int CountPages(ReadOnlySpan<byte> data)
    {
        var reader = OpenHeader(data, out long ifd);
        return PageDirectories(reader, ifd).Count;
    }

    /// <summary>
    /// Offsets of every page's IFD, in order, following the next-IFD chain from <paramref name="first"/>. A chain that
    /// points back to an IFD it already visited would never end, so it's rejected as a corrupt file.
    /// </summary>
    private static List<long> PageDirectories(TiffReader reader, long first)
    {
        var offsets = new List<long>();
        var seen = new HashSet<long>();
        for (long ifd = first; ifd != 0; ifd = reader.NextIfd(ifd))
        {
            if (!seen.Add(ifd)) throw new InvalidDataException("TIFF page directories form a loop.");
            offsets.Add(ifd);
        }
        return offsets;
    }

    /// <summary>Decodes one page into raw sample values (no normalization).</summary>
    /// <param name="data">The whole TIFF file.</param>
    /// <param name="page">Zero-based page index for multi-page stacks.</param>
    /// <exception cref="NotSupportedException">The file uses a feature this decoder deliberately omits (see class docs).</exception>
    /// <exception cref="InvalidDataException">The file is truncated or inconsistent.</exception>
    public static GrayImage Decode(ReadOnlySpan<byte> data, int page = 0)
    {
        var reader = OpenHeader(data, out long ifd);
        for (int p = 0; p < page; p++)
        {
            ifd = reader.NextIfd(ifd);
            if (ifd == 0) throw new ArgumentOutOfRangeException(nameof(page), $"TIFF has only {p + 1} page(s).");
        }
        return DecodePage(reader, ifd);
    }

    /// <summary>Writes a single-page, single-strip, little-endian grayscale TIFF.</summary>
    /// <param name="image">Pixel values in the units to store (integers are rounded and clamped for UInt8/UInt16).</param>
    /// <param name="sampleType">Storage format. Use <see cref="TiffSampleType.Float32"/> to keep values exactly.</param>
    /// <param name="deflate">Compress the strip with zlib/Deflate (lossless; readable by ImageJ/Fiji, Python and libtiff).</param>
    /// <returns>The complete file contents. File layout: header | pixel strip | IFD.</returns>
    public static byte[] Encode(GrayImage image, TiffSampleType sampleType = TiffSampleType.UInt16, bool deflate = false) =>
        EncodeStack([image], sampleType, deflate);

    /// <summary>
    /// Writes a multi-page stack (time-lapse frames or z-slices), one page per image, in order. Pages may differ in
    /// size. Layout: header | strip₀ | IFD₀ | strip₁ | IFD₁ | …, where each IFD's last 4 bytes hold the offset of the
    /// next IFD (0 after the last page). A one-page stack is byte for byte what <see cref="Encode"/> writes.
    /// </summary>
    /// <param name="pages">The images, first page first (at least one).</param>
    /// <param name="sampleType">Storage format for every page.</param>
    /// <param name="deflate">Compress each page's strip with zlib/Deflate.</param>
    /// <param name="tags">Optional description, software and resolution tags (see <see cref="TiffPageTags"/>), or null for none.</param>
    /// <returns>The complete file contents.</returns>
    /// <exception cref="ArgumentException">No pages, or the file would exceed TIFF's 4 GB offset limit.</exception>
    public static byte[] EncodeStack(IReadOnlyList<GrayImage> pages, TiffSampleType sampleType = TiffSampleType.UInt16, bool deflate = false,
        TiffPageTags? tags = null)
    {
        if (pages.Count == 0) throw new ArgumentException("A TIFF needs at least one page.", nameof(pages));
        using var file = new MemoryStream();
        using (var writer = new TiffStackWriter(file, sampleType, deflate, leaveOpen: true, tags: tags))
            foreach (var page in pages) writer.Append(page);
        return file.ToArray();
    }

    /// <summary>
    /// One page's pixel strip, its tag values that don't fit an IFD entry, and its IFD (next-IFD offset 0), with the strip
    /// to be written at <paramref name="stripOffset"/> and the values block right after it and its word-alignment pad, at
    /// <c>ExtraOffset</c>. The block holds, in tag order, the ImageDescription and Software text (ASCII, NUL-terminated)
    /// and the two resolution rationals; a value of 4 bytes or fewer sits inline in its entry, as TIFF requires.
    /// </summary>
    internal static (byte[] Strip, byte[] Extra, uint ExtraOffset, byte[] Ifd) EncodePage(GrayImage image, TiffSampleType sampleType, bool deflate,
        uint stripOffset, TiffPageTags? tags = null)
    {
        int bytesPerSample = sampleType switch { TiffSampleType.UInt8 => 1, TiffSampleType.UInt16 => 2, _ => 4 };
        var pixelBytes = new byte[image.Length * bytesPerSample];
        for (int i = 0; i < image.Length; i++)
        {
            float v = image.Pixels[i];
            switch (sampleType)
            {
                case TiffSampleType.UInt8:
                    pixelBytes[i] = (byte)Math.Clamp(MathF.Round(v), 0, byte.MaxValue);
                    break;
                case TiffSampleType.UInt16:
                    BinaryPrimitives.WriteUInt16LittleEndian(pixelBytes.AsSpan(i * 2), (ushort)Math.Clamp(MathF.Round(v), 0, ushort.MaxValue));
                    break;
                default:
                    BinaryPrimitives.WriteSingleLittleEndian(pixelBytes.AsSpan(i * 4), v);
                    break;
            }
        }

        byte[] strip = deflate ? Deflate(pixelBytes) : pixelBytes;
        uint extraOffset = (uint)((stripOffset + (long)strip.Length + 1) / 2 * 2);
        using var extra = new MemoryStream();
        // An entry's value field: the bytes themselves when they fit (4 bytes, little-endian order), else their offset.
        uint Place(byte[] bytes)
        {
            if (bytes.Length <= 4)
            {
                Span<byte> inline = stackalloc byte[4];
                bytes.CopyTo(inline);
                return BinaryPrimitives.ReadUInt32LittleEndian(inline);
            }
            long at = extraOffset + extra.Length;
            if (at + bytes.Length > uint.MaxValue) throw new ArgumentException("Stack is too large for a TIFF (4 GB).");
            extra.Write(bytes);
            if (extra.Length % 2 == 1) extra.WriteByte(0);   // the next value (and the IFD) starts on a word boundary
            return (uint)at;
        }
        static byte[] Text(string text, System.Text.Encoding encoding) => [.. encoding.GetBytes(text), 0];
        static byte[] Rational(uint numerator, uint denominator)
        {
            var bytes = new byte[8];
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, numerator);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), denominator);
            return bytes;
        }

        // Tags stay in ascending order, as TIFF requires.
        var list = new List<(ushort Tag, ushort Type, uint Count, uint Value)>
        {
            (TagImageWidth, 4, 1, (uint)image.Width),
            (TagImageLength, 4, 1, (uint)image.Height),
            (TagBitsPerSample, 3, 1, (uint)(bytesPerSample * 8)),
            (TagCompression, 3, 1, deflate ? CompressionDeflate : CompressionNone),
            (TagPhotometric, 3, 1, 1),               // BlackIsZero
        };
        if (tags?.Description is { } description)
        {
            var bytes = Text(description, System.Text.Encoding.Latin1);
            list.Add((TagImageDescription, 2, (uint)bytes.Length, Place(bytes)));
        }
        list.AddRange(
        [
            (TagStripOffsets, 4, 1, stripOffset),
            (TagSamplesPerPixel, 3, 1, 1),
            (TagRowsPerStrip, 4, 1, (uint)image.Height),
            (TagStripByteCounts, 4, 1, (uint)strip.Length),
        ]);
        if (tags?.Resolution is { } resolution)
        {
            list.Add((TagXResolution, 5, 1, Place(Rational(resolution.XNumerator, resolution.XDenominator))));
            list.Add((TagYResolution, 5, 1, Place(Rational(resolution.YNumerator, resolution.YDenominator))));
            list.Add((TagResolutionUnit, 3, 1, resolution.Unit));
        }
        if (tags?.Software is { } software)
        {
            var bytes = Text(software, System.Text.Encoding.ASCII);
            list.Add((TagSoftware, 2, (uint)bytes.Length, Place(bytes)));
        }
        list.Add((TagSampleFormat, 3, 1, sampleType == TiffSampleType.Float32 ? 3u : 1u));
        var entries = list.ToArray();

        var ifd = new byte[2 + entries.Length * 12 + 4];
        BinaryPrimitives.WriteUInt16LittleEndian(ifd, (ushort)entries.Length);
        for (int e = 0; e < entries.Length; e++)
        {
            var span = ifd.AsSpan(2 + e * 12, 12);
            BinaryPrimitives.WriteUInt16LittleEndian(span, entries[e].Tag);
            BinaryPrimitives.WriteUInt16LittleEndian(span[2..], entries[e].Type);
            BinaryPrimitives.WriteUInt32LittleEndian(span[4..], entries[e].Count);
            if (entries[e].Type == 3) BinaryPrimitives.WriteUInt16LittleEndian(span[8..], (ushort)entries[e].Value);
            else BinaryPrimitives.WriteUInt32LittleEndian(span[8..], entries[e].Value);
        }
        return (strip, extra.ToArray(), extraOffset, ifd);   // next-IFD offset: 0 until the writer patches it
    }

    /// <summary>Validates the header and returns a reader for its byte order and variant, plus the first IFD's offset.</summary>
    private static TiffReader OpenHeader(ReadOnlySpan<byte> data, out long firstIfd)
    {
        bool little = CheckHeader(data, out bool big);
        var reader = new TiffReader(data, little, big);
        firstIfd = reader.FirstIfd();
        return reader;
    }

    /// <summary>
    /// Validates the first bytes of a file (16 are enough); returns true for little-endian ("II"). BigTIFF (magic 43)
    /// must declare 8-byte offsets, the only size the format defines.
    /// </summary>
    internal static bool CheckHeader(ReadOnlySpan<byte> header, out bool big)
    {
        if (!HasTiffSignature(header)) throw new InvalidDataException("Not a TIFF file.");
        bool little = header[0] == 'I';
        big = (little ? header[2] : header[3]) == 43;
        if (big && (header.Length < 16 || (little ? BinaryPrimitives.ReadUInt16LittleEndian(header[4..]) : BinaryPrimitives.ReadUInt16BigEndian(header[4..])) != 8))
            throw new InvalidDataException("BigTIFF header is truncated or declares an offset size other than 8.");
        return little;
    }

    /// <summary>Offsets of every page's IFD in a TIFF on a seekable stream (the stream's position is changed).</summary>
    internal static List<long> PageDirectories(Stream stream, out bool littleEndian, out bool big)
    {
        Span<byte> header = stackalloc byte[16];
        stream.Position = 0;
        int read = stream.ReadAtLeast(header, header.Length, throwOnEndOfStream: false);
        littleEndian = CheckHeader(header[..read], out big);
        var reader = new TiffReader(stream, littleEndian, big);
        return PageDirectories(reader, reader.FirstIfd());
    }

    /// <summary>The pixel size of the page whose IFD is at <paramref name="ifd"/> (tags 282, 283, 296), or null without both resolutions.</summary>
    internal static TiffResolution? Resolution(Stream stream, bool littleEndian, bool big, long ifd)
    {
        var r = new TiffReader(stream, littleEndian, big);
        if (r.Rational(ifd, TagXResolution) is not { } x || r.Rational(ifd, TagYResolution) is not { } y) return null;
        uint unit = r.Tag(ifd, TagResolutionUnit).FirstOrDefault(2u);   // TIFF's default: inch
        return new TiffResolution(x.Numerator, x.Denominator, y.Numerator, y.Denominator, (ushort)Math.Min(unit, ushort.MaxValue));
    }

    /// <summary>The Software tag (305) of the page whose IFD is at <paramref name="ifd"/>, or null.</summary>
    internal static string? Software(Stream stream, bool littleEndian, bool big, long ifd) =>
        new TiffReader(stream, littleEndian, big).Text(ifd, TagSoftware);

    /// <summary>The ImageDescription tag (270) of the page whose IFD is at <paramref name="ifd"/>, or null.</summary>
    internal static string? Description(Stream stream, bool littleEndian, bool big, long ifd) =>
        new TiffReader(stream, littleEndian, big).Text(ifd, TagImageDescription);

    /// <summary>Width and height of the page whose IFD is at <paramref name="ifd"/>, read from its tags alone.</summary>
    internal static (int Width, int Height) PageSize(Stream stream, bool littleEndian, bool big, long ifd)
    {
        var r = new TiffReader(stream, littleEndian, big);
        uint w = r.Tag(ifd, TagImageWidth, required: true)[0], h = r.Tag(ifd, TagImageLength, required: true)[0];
        if (w is 0 or > int.MaxValue || h is 0 or > int.MaxValue) throw new InvalidDataException("TIFF page has an impossible size.");
        return ((int)w, (int)h);
    }

    /// <summary>Decodes the page whose IFD is at <paramref name="ifd"/>, reading only that page's bytes from the stream.</summary>
    internal static GrayImage DecodePage(Stream stream, bool littleEndian, bool big, long ifd) =>
        DecodePage(new TiffReader(stream, littleEndian, big), ifd);

    /// <summary>
    /// Decodes the page described by the IFD at <paramref name="ifd"/> in four steps: read the tags, reject
    /// unsupported variants up front, then reassemble and decompress the strips into one raw byte buffer.
    /// Finally undo the predictor and convert samples to float gray values.
    /// </summary>
    private static GrayImage DecodePage(TiffReader r, long ifd)
    {
        // 1. Tags. Missing optional tags take their TIFF-spec defaults.
        int width = (int)r.Tag(ifd, TagImageWidth, required: true)[0];
        int height = (int)r.Tag(ifd, TagImageLength, required: true)[0];
        int samplesPerPixel = (int)(r.Tag(ifd, TagSamplesPerPixel).FirstOrDefault(1u));
        int bits = (int)r.Tag(ifd, TagBitsPerSample).FirstOrDefault(1u);
        uint compression = r.Tag(ifd, TagCompression).FirstOrDefault(CompressionNone);
        uint photometric = r.Tag(ifd, TagPhotometric).FirstOrDefault(1u);
        uint planar = r.Tag(ifd, TagPlanarConfig).FirstOrDefault(1u);
        uint predictor = r.Tag(ifd, TagPredictor).FirstOrDefault(1u);
        uint sampleFormat = r.Tag(ifd, TagSampleFormat).FirstOrDefault(1u);
        uint rowsPerStrip = r.Tag(ifd, TagRowsPerStrip).FirstOrDefault((uint)height);

        // 2. Fail fast, with an actionable message, on anything outside the supported subset.
        if (r.Tag(ifd, TagTileWidth).Length > 0) throw new NotSupportedException("Tiled TIFFs are not supported; save as strips.");
        if (bits is not (8 or 16 or 32)) throw new NotSupportedException($"{bits}-bit TIFF samples are not supported.");
        if (sampleFormat == 3 && bits != 32) throw new NotSupportedException("Only 32-bit floating-point TIFF is supported.");
        if (sampleFormat is not (1 or 2 or 3)) throw new NotSupportedException($"TIFF SampleFormat {sampleFormat} is not supported.");
        if (photometric is not (0 or 1 or 2)) throw new NotSupportedException($"TIFF photometric interpretation {photometric} (e.g. palette) is not supported.");
        if (samplesPerPixel > 1 && planar != 1) throw new NotSupportedException("Planar-separated multichannel TIFF is not supported.");
        if (predictor is not (1 or 2)) throw new NotSupportedException($"TIFF predictor {predictor} is not supported.");
        if (predictor == 2 && sampleFormat == 3) throw new NotSupportedException("Horizontal predictor on float TIFF is not supported.");
        if (compression is not (CompressionNone or CompressionLzw or CompressionDeflate or CompressionAdobeDeflate or CompressionPackBits))
            throw new NotSupportedException(UnsupportedCompression(compression));

        // 3. Strips: each holds RowsPerStrip whole rows (the last may hold fewer), stored or compressed on its own.
        //    Decode them in order into one contiguous buffer of height × rowBytes.
        int bytesPerSample = bits / 8;
        int rowSamples = width * samplesPerPixel;
        int rowBytes = rowSamples * bytesPerSample;
        var raw = new byte[(long)rowBytes * height];

        long[] offsets = r.Offsets(ifd, TagStripOffsets, required: true);
        long[] counts = r.Offsets(ifd, TagStripByteCounts, required: true);
        if (offsets.Length != counts.Length) throw new InvalidDataException("TIFF strip offset/count mismatch.");

        int written = 0;
        for (int s = 0; s < offsets.Length && written < raw.Length; s++)
        {
            int stripRows = (int)Math.Min(rowsPerStrip, (uint)(height - written / rowBytes));
            int expected = stripRows * rowBytes;
            var stored = r.Slice(offsets[s], counts[s]);
            byte[] decoded = compression switch
            {
                CompressionNone => stored.ToArray(),
                CompressionLzw => Lzw.Decode(stored, expected),
                CompressionPackBits => UnpackBits(stored, expected),
                _ => Inflate(stored),
            };
            if (decoded.Length < expected) throw new InvalidDataException($"TIFF strip {s} is truncated.");
            decoded.AsSpan(0, expected).CopyTo(raw.AsSpan(written));
            written += expected;
        }
        if (written != raw.Length) throw new InvalidDataException("TIFF strips do not cover the whole image.");

        // 4. Undo the predictor (it applies after decompression), then convert samples to gray floats.
        if (predictor == 2) UndoHorizontalPredictor(raw, height, rowSamples, samplesPerPixel, bytesPerSample, r.LittleEndian);

        var image = new GrayImage(width, height);
        double whiteLevel = sampleFormat == 1 ? Math.Pow(2, bits) - 1 : 0;
        int channelsToAverage = photometric == 2 ? Math.Min(3, samplesPerPixel) : 1;

        for (int i = 0; i < width * height; i++)
        {
            double sum = 0;
            for (int c = 0; c < channelsToAverage; c++)
                sum += ReadSample(raw, (i * samplesPerPixel + c) * bytesPerSample, bits, sampleFormat, r.LittleEndian);
            double value = sum / channelsToAverage;
            if (photometric == 0 && sampleFormat == 1) value = whiteLevel - value;   // WhiteIsZero
            image.Pixels[i] = (float)value;
        }
        return image;
    }

    /// <summary>Reads one sample of the given width and format (1 = unsigned, 2 = signed, 3 = float) in the file's byte order.</summary>
    private static double ReadSample(byte[] raw, int offset, int bits, uint format, bool little)
    {
        var s = raw.AsSpan(offset);
        return (bits, format) switch
        {
            (8, 2) => (sbyte)s[0],
            (8, _) => s[0],
            (16, 2) => little ? BinaryPrimitives.ReadInt16LittleEndian(s) : BinaryPrimitives.ReadInt16BigEndian(s),
            (16, _) => little ? BinaryPrimitives.ReadUInt16LittleEndian(s) : BinaryPrimitives.ReadUInt16BigEndian(s),
            (32, 3) => little ? BinaryPrimitives.ReadSingleLittleEndian(s) : BinaryPrimitives.ReadSingleBigEndian(s),
            (32, 2) => little ? BinaryPrimitives.ReadInt32LittleEndian(s) : BinaryPrimitives.ReadInt32BigEndian(s),
            _ => little ? BinaryPrimitives.ReadUInt32LittleEndian(s) : BinaryPrimitives.ReadUInt32BigEndian(s),
        };
    }

    /// <summary>
    /// The refusal for a compression DORA doesn't decode: its code, its name when it's a known one (e.g. 50000,
    /// Zstandard), and the plain conversion that fixes it. The copy tifffile writes that way has no axis labels, hence the
    /// <c>--axes</c> advice.
    /// </summary>
    internal static string UnsupportedCompression(uint compression)
    {
        string? name = compression switch
        {
            2 or 3 or 4 => "CCITT fax",
            6 or 7 => "JPEG",
            33003 or 33005 or 34712 => "JPEG 2000",
            34887 => "LERC",
            34925 => "LZMA",
            34933 => "PNG",
            50000 => "Zstandard",
            50001 => "WebP",
            50002 => "JPEG XL",
            _ => null,
        };
        return $"TIFF compression {compression}{(name is null ? "" : $" ({name})")} is not supported. Save an uncompressed copy, e.g. in " +
               "Python: tifffile.imwrite(dst, tifffile.imread(src)), and denoise that; for a stack, add --axes TYX (a time-lapse) or " +
               "--axes ZYX (a volume), since the copy has no axis labels.";
    }

    /// <summary>The stored (still compressed) bytes of each strip of a page, in order: for tests of the decompressors.</summary>
    internal static List<byte[]> StoredStrips(byte[] file, int page)
    {
        var r = OpenHeader(file, out long ifd);
        for (int p = 0; p < page; p++) ifd = r.NextIfd(ifd);
        long[] offsets = r.Offsets(ifd, TagStripOffsets, required: true);
        long[] counts = r.Offsets(ifd, TagStripByteCounts, required: true);
        var strips = new List<byte[]>(offsets.Length);
        for (int s = 0; s < offsets.Length; s++) strips.Add(r.Slice(offsets[s], counts[s]).ToArray());
        return strips;
    }

    /// <summary>Predictor 2 stores each sample as the difference from the previous sample of the same channel.</summary>
    private static void UndoHorizontalPredictor(byte[] raw, int height, int rowSamples, int spp, int bytesPerSample, bool little)
    {
        for (int y = 0; y < height; y++)
        {
            var row = raw.AsSpan(y * rowSamples * bytesPerSample, rowSamples * bytesPerSample);
            for (int i = spp; i < rowSamples; i++)
            {
                switch (bytesPerSample)
                {
                    case 1:
                        row[i] = (byte)(row[i] + row[i - spp]);
                        break;
                    case 2:
                        ushort prev2 = little ? BinaryPrimitives.ReadUInt16LittleEndian(row[((i - spp) * 2)..]) : BinaryPrimitives.ReadUInt16BigEndian(row[((i - spp) * 2)..]);
                        ushort cur2 = little ? BinaryPrimitives.ReadUInt16LittleEndian(row[(i * 2)..]) : BinaryPrimitives.ReadUInt16BigEndian(row[(i * 2)..]);
                        ushort sum2 = (ushort)(cur2 + prev2);
                        if (little) BinaryPrimitives.WriteUInt16LittleEndian(row[(i * 2)..], sum2);
                        else BinaryPrimitives.WriteUInt16BigEndian(row[(i * 2)..], sum2);
                        break;
                    default:
                        uint prev4 = little ? BinaryPrimitives.ReadUInt32LittleEndian(row[((i - spp) * 4)..]) : BinaryPrimitives.ReadUInt32BigEndian(row[((i - spp) * 4)..]);
                        uint cur4 = little ? BinaryPrimitives.ReadUInt32LittleEndian(row[(i * 4)..]) : BinaryPrimitives.ReadUInt32BigEndian(row[(i * 4)..]);
                        uint sum4 = unchecked(cur4 + prev4);
                        if (little) BinaryPrimitives.WriteUInt32LittleEndian(row[(i * 4)..], sum4);
                        else BinaryPrimitives.WriteUInt32BigEndian(row[(i * 4)..], sum4);
                        break;
                }
            }
        }
    }

    /// <summary>
    /// PackBits run-length decoding (TIFF 6.0 §9). Each control byte n means: 0..127 → copy the next n+1
    /// bytes literally; −127..−1 → repeat the next byte 1−n times; −128 → no-op.
    /// </summary>
    internal static byte[] UnpackBits(ReadOnlySpan<byte> src, int expectedLength)
    {
        var output = new byte[expectedLength];
        int i = 0, o = 0;
        while (o < expectedLength && i < src.Length)
        {
            int n = (sbyte)src[i++];
            if (n >= 0)
            {
                int run = Math.Min(n + 1, Math.Min(expectedLength - o, src.Length - i));
                src.Slice(i, run).CopyTo(output.AsSpan(o));
                i += n + 1;
                o += run;
            }
            else if (n != -128 && i < src.Length)
            {
                int run = Math.Min(1 - n, expectedLength - o);
                output.AsSpan(o, run).Fill(src[i++]);
                o += run;
            }
        }
        if (o < expectedLength) throw new InvalidDataException("PackBits data ended early.");
        return output;
    }

    /// <summary>Decompresses a zlib-wrapped Deflate stream (TIFF compression 8 and 32946).</summary>
    private static byte[] Inflate(ReadOnlySpan<byte> compressed)
    {
        using var input = new MemoryStream(compressed.ToArray());
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);
        return output.ToArray();
    }

    /// <summary>Compresses to a zlib-wrapped Deflate stream, the inverse of <see cref="Inflate"/>.</summary>
    private static byte[] Deflate(byte[] data)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Optimal, leaveOpen: true))
            zlib.Write(data);
        return output.ToArray();
    }

    /// <summary>
    /// Bounds-checked reads in the file's byte order. Every offset in a TIFF comes from the file itself, so every access
    /// is checked: a corrupt or malicious file raises <see cref="InvalidDataException"/> rather than reading out of
    /// bounds. Reads come from either the whole file in memory or a seekable stream. In
    /// stream mode each read seeks and copies just the bytes asked for, so a page can be decoded without loading the
    /// rest of a stack (<see cref="TiffStackReader"/>). Offsets are checked against the file length before anything is
    /// allocated, so a corrupt count can't trigger an allocation larger than the file.
    /// <para>Classic TIFF and BigTIFF differ only in field widths: BigTIFF's IFD starts with an 8-byte entry count, its
    /// entries are 20 bytes (tag 2, type 2, count 8, value-or-offset 8), and it adds 8-byte integer types (16–18).</para>
    /// </summary>
    private readonly ref struct TiffReader
    {
        private readonly ReadOnlySpan<byte> _data;
        private readonly Stream? _stream;
        private readonly long _length;

        /// <summary>True for "II" files, false for "MM" (big-endian) files.</summary>
        public bool LittleEndian { get; }

        /// <summary>True for BigTIFF (magic 43): 8-byte offsets and counts.</summary>
        public bool Big { get; }

        public TiffReader(ReadOnlySpan<byte> data, bool littleEndian, bool big)
        {
            _data = data;
            _stream = null;
            _length = data.Length;
            (LittleEndian, Big) = (littleEndian, big);
        }

        public TiffReader(Stream stream, bool littleEndian, bool big)
        {
            _data = default;
            _stream = stream;
            _length = stream.Length;
            (LittleEndian, Big) = (littleEndian, big);
        }

        private int CountSize => Big ? 8 : 2;
        private int EntrySize => Big ? 20 : 12;
        private int ValueSize => Big ? 8 : 4;

        /// <summary>Unsigned 16-bit value at <paramref name="offset"/>, in the file's byte order.</summary>
        public ushort U16(long offset) => Read16(Slice(offset, 2));

        /// <summary>Unsigned 32-bit value at <paramref name="offset"/>, in the file's byte order.</summary>
        public uint U32(long offset) => Read32(Slice(offset, 4));

        private ushort Read16(ReadOnlySpan<byte> s) =>
            LittleEndian ? BinaryPrimitives.ReadUInt16LittleEndian(s) : BinaryPrimitives.ReadUInt16BigEndian(s);

        private uint Read32(ReadOnlySpan<byte> s) =>
            LittleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(s) : BinaryPrimitives.ReadUInt32BigEndian(s);

        private ulong Read64(ReadOnlySpan<byte> s) =>
            LittleEndian ? BinaryPrimitives.ReadUInt64LittleEndian(s) : BinaryPrimitives.ReadUInt64BigEndian(s);

        /// <summary>A file position or size read from the file, rejected if it can't be one.</summary>
        private long Position(ulong value) =>
            value <= (ulong)_length ? (long)value : throw new InvalidDataException("TIFF offset points outside the file.");

        /// <summary>A checked view (span mode) or copy (stream mode) of <paramref name="length"/> bytes at <paramref name="offset"/>.</summary>
        public ReadOnlySpan<byte> Slice(long offset, long length)
        {
            if (offset < 0 || length < 0 || offset > _length || length > _length - offset)
                throw new InvalidDataException("TIFF offset points outside the file.");
            if (_stream is null) return _data.Slice((int)offset, (int)length);
            if (length > Array.MaxLength) throw new InvalidDataException("TIFF block is too large to read.");
            var buffer = new byte[length];
            _stream.Position = offset;
            _stream.ReadExactly(buffer);
            return buffer;
        }

        /// <summary>Offset of the first IFD, from the header (bytes 4–7, or 8–15 in BigTIFF).</summary>
        public long FirstIfd() => Big ? Position(Read64(Slice(8, 8))) : U32(4);

        /// <summary>Number of entries in the IFD at <paramref name="ifd"/>.</summary>
        private long EntryCount(long ifd) => Big ? Position(Read64(Slice(ifd, 8))) : U16(ifd);

        /// <summary>Offset of the IFD after this one (0 if this is the last page): it follows the entry table.</summary>
        public long NextIfd(long ifd)
        {
            long at = ifd + CountSize + EntryCount(ifd) * EntrySize;
            return Big ? Position(Read64(Slice(at, 8))) : U32(at);
        }

        /// <summary>Bytes per value of a TIFF field type (0 = a type no tag DORA reads should have).</summary>
        private static int TypeSize(ushort type) => type switch
        {
            1 or 2 or 6 or 7 => 1,
            3 or 8 => 2,
            4 or 9 or 13 => 4,
            5 or 10 or 16 or 17 or 18 => 8,   // RATIONAL and SRATIONAL are two 32-bit integers
            _ => 0,
        };

        /// <summary>
        /// Finds a tag in the IFD. Entry layout: tag (2 bytes), type (2), count (4 or 8), value-or-offset (4 or 8). The
        /// values sit inline in the last field when count × size fits there, otherwise that field holds their offset.
        /// The entry table is read as one block (one seek in stream mode, even for thousands of strip offsets).
        /// </summary>
        private bool Find(long ifd, ushort tag, out ushort type, out long count, out long valuesAt)
        {
            long n = EntryCount(ifd);
            var table = Slice(ifd + CountSize, n * EntrySize);
            for (int e = 0; e < n; e++)
            {
                var entry = table.Slice(e * EntrySize, EntrySize);
                if (Read16(entry) != tag) continue;

                type = Read16(entry[2..]);
                int size = TypeSize(type);
                if (size == 0) throw new InvalidDataException($"TIFF tag {tag} has unexpected type {type}.");
                count = Position(Big ? Read64(entry[4..]) : Read32(entry[4..]));
                long bytes = count * size;   // count ≤ file length, so this can't overflow
                long entryAt = ifd + CountSize + (long)e * EntrySize;
                long field = entryAt + 4 + ValueSize;   // after tag, type and count
                valuesAt = bytes <= ValueSize ? field : Position(Big ? Read64(entry[12..]) : Read32(entry[8..]));
                return true;
            }
            (type, count, valuesAt) = (0, 0, 0);
            return false;
        }

        /// <summary>Values of an integer tag (BYTE/SHORT/LONG, and LONG8 in BigTIFF); empty if absent.</summary>
        private ulong[] Values(long ifd, ushort tag, bool required)
        {
            if (!Find(ifd, tag, out ushort type, out long count, out long valuesAt))
            {
                if (required) throw new InvalidDataException($"TIFF is missing required tag {tag}.");
                return [];
            }
            if (type is 2 or 5 or 10) throw new InvalidDataException($"TIFF tag {tag} has unexpected type {type}.");
            int size = TypeSize(type);
            var block = Slice(valuesAt, count * size);
            var values = new ulong[count];
            for (int k = 0; k < count; k++)
                values[k] = size switch
                {
                    1 => block[k],
                    2 => Read16(block.Slice(k * 2)),
                    4 => Read32(block.Slice(k * 4)),
                    _ => Read64(block.Slice(k * 8)),
                };
            return values;
        }

        /// <summary>Values of an integer tag that must fit 32 bits (sizes, bit depths, codes); empty if absent.</summary>
        public uint[] Tag(long ifd, ushort tag, bool required = false) =>
            Array.ConvertAll(Values(ifd, tag, required),
                v => v <= uint.MaxValue ? (uint)v : throw new InvalidDataException($"TIFF tag {tag} has an out-of-range value."));

        /// <summary>Values of a tag holding file positions or byte counts (strip offsets and counts); empty if absent.</summary>
        public long[] Offsets(long ifd, ushort tag, bool required = false)
        {
            var values = Values(ifd, tag, required);
            var result = new long[values.Length];
            for (int k = 0; k < values.Length; k++) result[k] = Position(values[k]);
            return result;
        }

        /// <summary>A RATIONAL tag's first value (numerator, denominator), or null if it's absent or of another type.</summary>
        public (uint Numerator, uint Denominator)? Rational(long ifd, ushort tag)
        {
            if (!Find(ifd, tag, out ushort type, out long count, out long valuesAt) || type != 5 || count < 1) return null;
            var block = Slice(valuesAt, 8);
            return (Read32(block), Read32(block[4..]));
        }

        /// <summary>An ASCII tag's text up to its first NUL (null if absent or not ASCII). Read as UTF-8, which ASCII is.</summary>
        public string? Text(long ifd, ushort tag)
        {
            if (!Find(ifd, tag, out ushort type, out long count, out long valuesAt) || type != 2) return null;
            var bytes = Slice(valuesAt, count);
            int end = bytes.IndexOf((byte)0);
            return System.Text.Encoding.UTF8.GetString(end < 0 ? bytes : bytes[..end]);
        }
    }
}

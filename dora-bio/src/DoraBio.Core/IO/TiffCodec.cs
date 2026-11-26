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
/// Reads: little- or big-endian; strips (not tiles); 8/16/32-bit unsigned or signed integers and 32-bit floats;
/// uncompressed, Deflate (8 / 32946) or PackBits compression; horizontal predictor (2) for integers;
/// BlackIsZero, WhiteIsZero and RGB (averaged to gray); any page of a multi-page stack.
/// Not supported (a clear <see cref="NotSupportedException"/> is thrown): LZW, JPEG, tiles, BigTIFF, palettes,
/// planar RGB. Most microscope software can save uncompressed or Deflate TIFF.
///
/// Writes: little-endian, one strip per page, uncompressed or Deflate, UInt8/UInt16/Float32; single pages or
/// multi-page stacks (<see cref="EncodeStack"/>).
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
        TagStripByteCounts = 279, TagPlanarConfig = 284, TagPredictor = 317, TagTileWidth = 322, TagSampleFormat = 339;

    // Compression codes. 32946 is the older "Adobe Deflate" code, still written by some tools; the data is identical.
    private const ushort CompressionNone = 1, CompressionLzw = 5, CompressionDeflate = 8,
        CompressionPackBits = 32773, CompressionAdobeDeflate = 32946;

    /// <summary>True if the data starts with a classic TIFF header in either byte order.</summary>
    public static bool HasTiffSignature(ReadOnlySpan<byte> data) =>
        data.Length >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] == 42 && data[3] == 0) ||
                             (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == 42));

    /// <summary>Number of pages (IFDs) in the file. Microscope stacks store one frame or z-slice per page.</summary>
    public static int CountPages(ReadOnlySpan<byte> data)
    {
        var reader = OpenHeader(data, out uint ifd);
        return PageDirectories(reader, ifd).Count;
    }

    /// <summary>
    /// Offsets of every page's IFD, in order, following the next-IFD chain from <paramref name="first"/>. A chain that
    /// points back to an IFD it already visited would never end, so it's rejected as a corrupt file.
    /// </summary>
    private static List<uint> PageDirectories(TiffReader reader, uint first)
    {
        var offsets = new List<uint>();
        var seen = new HashSet<uint>();
        for (uint ifd = first; ifd != 0; ifd = reader.NextIfd(ifd))
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
        var reader = OpenHeader(data, out uint ifd);
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
    /// <returns>The complete file contents.</returns>
    /// <exception cref="ArgumentException">No pages, or the file would exceed TIFF's 4 GB offset limit.</exception>
    public static byte[] EncodeStack(IReadOnlyList<GrayImage> pages, TiffSampleType sampleType = TiffSampleType.UInt16, bool deflate = false)
    {
        if (pages.Count == 0) throw new ArgumentException("A TIFF needs at least one page.", nameof(pages));
        using var file = new MemoryStream();
        using (var writer = new TiffStackWriter(file, sampleType, deflate, leaveOpen: true))
            foreach (var page in pages) writer.Append(page);
        return file.ToArray();
    }

    /// <summary>One page's pixel strip and its IFD (next-IFD offset 0), with the strip to be written at <paramref name="stripOffset"/>.</summary>
    internal static (byte[] Strip, byte[] Ifd) EncodePage(GrayImage image, TiffSampleType sampleType, bool deflate, uint stripOffset)
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
        var entries = new (ushort Tag, ushort Type, uint Value)[]
        {
            (TagImageWidth, 4, (uint)image.Width),
            (TagImageLength, 4, (uint)image.Height),
            (TagBitsPerSample, 3, (uint)(bytesPerSample * 8)),
            (TagCompression, 3, deflate ? CompressionDeflate : CompressionNone),
            (TagPhotometric, 3, 1),                  // BlackIsZero
            (TagStripOffsets, 4, stripOffset),
            (TagSamplesPerPixel, 3, 1),
            (TagRowsPerStrip, 4, (uint)image.Height),
            (TagStripByteCounts, 4, (uint)strip.Length),
            (TagSampleFormat, 3, sampleType == TiffSampleType.Float32 ? 3u : 1u),
        };

        var ifd = new byte[2 + entries.Length * 12 + 4];
        BinaryPrimitives.WriteUInt16LittleEndian(ifd, (ushort)entries.Length);
        for (int e = 0; e < entries.Length; e++)
        {
            var span = ifd.AsSpan(2 + e * 12, 12);
            BinaryPrimitives.WriteUInt16LittleEndian(span, entries[e].Tag);
            BinaryPrimitives.WriteUInt16LittleEndian(span[2..], entries[e].Type);
            BinaryPrimitives.WriteUInt32LittleEndian(span[4..], 1);
            if (entries[e].Type == 3) BinaryPrimitives.WriteUInt16LittleEndian(span[8..], (ushort)entries[e].Value);
            else BinaryPrimitives.WriteUInt32LittleEndian(span[8..], entries[e].Value);
        }
        return (strip, ifd);   // next-IFD offset: 0 until EncodeStack patches it
    }

    /// <summary>Validates the 8-byte header and returns a reader for its byte order, plus the first IFD's offset.</summary>
    private static TiffReader OpenHeader(ReadOnlySpan<byte> data, out uint firstIfd)
    {
        var reader = new TiffReader(data, littleEndian: CheckHeader(data));
        firstIfd = reader.U32(4);
        return reader;
    }

    /// <summary>Validates the first bytes of a file; returns true for little-endian ("II").</summary>
    internal static bool CheckHeader(ReadOnlySpan<byte> header)
    {
        // BigTIFF uses magic 43 and 64-bit offsets. Reject it explicitly rather than misreading it.
        if (header.Length >= 4 && (header[2] == 43 || header[3] == 43))
            throw new NotSupportedException("BigTIFF files are not supported.");
        if (!HasTiffSignature(header)) throw new InvalidDataException("Not a TIFF file.");
        return header[0] == 'I';
    }

    /// <summary>Offsets of every page's IFD in a TIFF on a seekable stream (the stream's position is changed).</summary>
    internal static List<uint> PageDirectories(Stream stream, out bool littleEndian)
    {
        Span<byte> header = stackalloc byte[8];
        stream.Position = 0;
        int read = stream.ReadAtLeast(header, header.Length, throwOnEndOfStream: false);
        littleEndian = CheckHeader(header[..read]);
        var reader = new TiffReader(stream, littleEndian);
        return PageDirectories(reader, reader.U32(4));
    }

    /// <summary>Decodes the page whose IFD is at <paramref name="ifd"/>, reading only that page's bytes from the stream.</summary>
    internal static GrayImage DecodePage(Stream stream, bool littleEndian, uint ifd) =>
        DecodePage(new TiffReader(stream, littleEndian), ifd);

    /// <summary>
    /// Decodes the page described by the IFD at <paramref name="ifd"/> in four steps: read the tags, reject
    /// unsupported variants up front, then reassemble and decompress the strips into one raw byte buffer.
    /// Finally undo the predictor and convert samples to float gray values.
    /// </summary>
    private static GrayImage DecodePage(TiffReader r, uint ifd)
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
        if (compression == CompressionLzw) throw new NotSupportedException("LZW-compressed TIFF is not supported; re-save uncompressed or with Deflate/ZIP.");
        if (compression is not (CompressionNone or CompressionDeflate or CompressionAdobeDeflate or CompressionPackBits))
            throw new NotSupportedException($"TIFF compression {compression} is not supported.");

        // 3. Strips: each holds RowsPerStrip whole rows (the last may hold fewer), stored or compressed on its own.
        //    Decode them in order into one contiguous buffer of height × rowBytes.
        int bytesPerSample = bits / 8;
        int rowSamples = width * samplesPerPixel;
        int rowBytes = rowSamples * bytesPerSample;
        var raw = new byte[(long)rowBytes * height];

        uint[] offsets = r.Tag(ifd, TagStripOffsets, required: true);
        uint[] counts = r.Tag(ifd, TagStripByteCounts, required: true);
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
    /// allocated, so a corrupt count can't trigger a huge allocation.
    /// </summary>
    private readonly ref struct TiffReader
    {
        private readonly ReadOnlySpan<byte> _data;
        private readonly Stream? _stream;
        private readonly long _length;

        /// <summary>True for "II" files, false for "MM" (big-endian) files.</summary>
        public bool LittleEndian { get; }

        public TiffReader(ReadOnlySpan<byte> data, bool littleEndian)
        {
            _data = data;
            _stream = null;
            _length = data.Length;
            LittleEndian = littleEndian;
        }

        public TiffReader(Stream stream, bool littleEndian)
        {
            _data = default;
            _stream = stream;
            _length = stream.Length;
            LittleEndian = littleEndian;
        }

        /// <summary>Unsigned 16-bit value at <paramref name="offset"/>, in the file's byte order.</summary>
        public ushort U16(long offset) => Read16(Slice(offset, 2));

        /// <summary>Unsigned 32-bit value at <paramref name="offset"/>, in the file's byte order.</summary>
        public uint U32(long offset) => Read32(Slice(offset, 4));

        private ushort Read16(ReadOnlySpan<byte> s) =>
            LittleEndian ? BinaryPrimitives.ReadUInt16LittleEndian(s) : BinaryPrimitives.ReadUInt16BigEndian(s);

        private uint Read32(ReadOnlySpan<byte> s) =>
            LittleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(s) : BinaryPrimitives.ReadUInt32BigEndian(s);

        /// <summary>A checked view (span mode) or copy (stream mode) of <paramref name="length"/> bytes at <paramref name="offset"/>.</summary>
        public ReadOnlySpan<byte> Slice(long offset, long length)
        {
            if (offset < 0 || length < 0 || offset + length > _length)
                throw new InvalidDataException("TIFF offset points outside the file.");
            if (_stream is null) return _data.Slice((int)offset, (int)length);
            if (length > Array.MaxLength) throw new InvalidDataException("TIFF block is too large to read.");
            var buffer = new byte[length];
            _stream.Position = offset;
            _stream.ReadExactly(buffer);
            return buffer;
        }

        /// <summary>Offset of the IFD after this one (0 if this is the last page): it follows the entry table.</summary>
        public uint NextIfd(uint ifd) => U32(ifd + 2 + U16(ifd) * 12L);

        /// <summary>Values of an integer tag (BYTE/SHORT/LONG); empty if absent.</summary>
        /// <remarks>
        /// Entry layout: tag (2 bytes), type (2), count (4), value-or-offset (4). The values sit inline in the
        /// last 4 bytes when count × size ≤ 4, otherwise those bytes hold their file offset. The entry table and
        /// each value array are read as one block (one seek in stream mode, even for thousands of strip offsets).
        /// </remarks>
        public uint[] Tag(uint ifd, ushort tag, bool required = false)
        {
            int count = U16(ifd);
            var table = Slice(ifd + 2, count * 12L);
            for (int e = 0; e < count; e++)
            {
                var entry = table.Slice(e * 12, 12);
                if (Read16(entry) != tag) continue;

                ushort type = Read16(entry[2..]);
                uint n = Read32(entry[4..]);
                int size = type switch { 1 or 6 or 7 => 1, 3 or 8 => 2, 4 or 9 => 4, _ => 0 };
                if (size == 0) throw new InvalidDataException($"TIFF tag {tag} has unexpected type {type}.");

                long entryAt = ifd + 2 + e * 12L;
                long valuesAt = (long)n * size <= 4 ? entryAt + 8 : Read32(entry[8..]);
                var block = Slice(valuesAt, (long)n * size);
                var values = new uint[n];
                for (int k = 0; k < n; k++)
                    values[k] = size switch
                    {
                        1 => block[k],
                        2 => Read16(block.Slice(k * 2)),
                        _ => Read32(block.Slice(k * 4)),
                    };
                return values;
            }
            if (required) throw new InvalidDataException($"TIFF is missing required tag {tag}.");
            return [];
        }
    }
}

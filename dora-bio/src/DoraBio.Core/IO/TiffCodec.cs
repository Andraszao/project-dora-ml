using System.Buffers.Binary;
using System.IO.Compression;
using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

public enum TiffSampleType
{
    UInt8,
    UInt16,
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
/// Writes: little-endian, single strip, uncompressed or Deflate, UInt8/UInt16/Float32.
/// </summary>
public static class TiffCodec
{
    private const ushort TagImageWidth = 256, TagImageLength = 257, TagBitsPerSample = 258, TagCompression = 259,
        TagPhotometric = 262, TagStripOffsets = 273, TagSamplesPerPixel = 277, TagRowsPerStrip = 278,
        TagStripByteCounts = 279, TagPlanarConfig = 284, TagPredictor = 317, TagTileWidth = 322, TagSampleFormat = 339;

    private const ushort CompressionNone = 1, CompressionLzw = 5, CompressionDeflate = 8,
        CompressionPackBits = 32773, CompressionAdobeDeflate = 32946;

    public static bool HasTiffSignature(ReadOnlySpan<byte> data) =>
        data.Length >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] == 42 && data[3] == 0) ||
                             (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == 42));

    public static int CountPages(ReadOnlySpan<byte> data)
    {
        var reader = OpenHeader(data, out uint ifd);
        int pages = 0;
        while (ifd != 0)
        {
            pages++;
            ifd = reader.NextIfd(ifd);
        }
        return pages;
    }

    /// <summary>Decodes one page into raw sample values (no normalization).</summary>
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

    public static byte[] Encode(GrayImage image, TiffSampleType sampleType = TiffSampleType.UInt16, bool deflate = false)
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
        const int dataOffset = 8;
        int ifdOffset = dataOffset + strip.Length;
        if (ifdOffset % 2 == 1) ifdOffset++;   // IFDs start on a word boundary

        var entries = new (ushort Tag, ushort Type, uint Value)[]
        {
            (TagImageWidth, 4, (uint)image.Width),
            (TagImageLength, 4, (uint)image.Height),
            (TagBitsPerSample, 3, (uint)(bytesPerSample * 8)),
            (TagCompression, 3, deflate ? CompressionDeflate : CompressionNone),
            (TagPhotometric, 3, 1),                  // BlackIsZero
            (TagStripOffsets, 4, dataOffset),
            (TagSamplesPerPixel, 3, 1),
            (TagRowsPerStrip, 4, (uint)image.Height),
            (TagStripByteCounts, 4, (uint)strip.Length),
            (TagSampleFormat, 3, sampleType == TiffSampleType.Float32 ? 3u : 1u),
        };

        var file = new byte[ifdOffset + 2 + entries.Length * 12 + 4];
        "II"u8.CopyTo(file);
        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(2), 42);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(4), (uint)ifdOffset);
        strip.CopyTo(file, dataOffset);

        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(ifdOffset), (ushort)entries.Length);
        for (int e = 0; e < entries.Length; e++)
        {
            var span = file.AsSpan(ifdOffset + 2 + e * 12, 12);
            BinaryPrimitives.WriteUInt16LittleEndian(span, entries[e].Tag);
            BinaryPrimitives.WriteUInt16LittleEndian(span[2..], entries[e].Type);
            BinaryPrimitives.WriteUInt32LittleEndian(span[4..], 1);
            if (entries[e].Type == 3) BinaryPrimitives.WriteUInt16LittleEndian(span[8..], (ushort)entries[e].Value);
            else BinaryPrimitives.WriteUInt32LittleEndian(span[8..], entries[e].Value);
        }
        // Next-IFD offset stays 0: single page.
        return file;
    }

    private static TiffReader OpenHeader(ReadOnlySpan<byte> data, out uint firstIfd)
    {
        if (data.Length >= 4 && (data[2] == 43 || data[3] == 43))
            throw new NotSupportedException("BigTIFF files are not supported.");
        if (!HasTiffSignature(data)) throw new InvalidDataException("Not a TIFF file.");
        var reader = new TiffReader(data, littleEndian: data[0] == 'I');
        firstIfd = reader.U32(4);
        return reader;
    }

    private static GrayImage DecodePage(TiffReader r, uint ifd)
    {
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

    private static byte[] Inflate(ReadOnlySpan<byte> compressed)
    {
        using var input = new MemoryStream(compressed.ToArray());
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);
        return output.ToArray();
    }

    private static byte[] Deflate(byte[] data)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Optimal, leaveOpen: true))
            zlib.Write(data);
        return output.ToArray();
    }

    private readonly ref struct TiffReader
    {
        private readonly ReadOnlySpan<byte> _data;
        public bool LittleEndian { get; }

        public TiffReader(ReadOnlySpan<byte> data, bool littleEndian)
        {
            _data = data;
            LittleEndian = littleEndian;
        }

        public ushort U16(long offset)
        {
            var s = Slice(offset, 2);
            return LittleEndian ? BinaryPrimitives.ReadUInt16LittleEndian(s) : BinaryPrimitives.ReadUInt16BigEndian(s);
        }

        public uint U32(long offset)
        {
            var s = Slice(offset, 4);
            return LittleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(s) : BinaryPrimitives.ReadUInt32BigEndian(s);
        }

        public ReadOnlySpan<byte> Slice(long offset, long length)
        {
            if (offset < 0 || length < 0 || offset + length > _data.Length)
                throw new InvalidDataException("TIFF offset points outside the file.");
            return _data.Slice((int)offset, (int)length);
        }

        public uint NextIfd(uint ifd) => U32(ifd + 2 + U16(ifd) * 12L);

        /// <summary>Values of an integer tag (BYTE/SHORT/LONG); empty if absent.</summary>
        public uint[] Tag(uint ifd, ushort tag, bool required = false)
        {
            int count = U16(ifd);
            for (int e = 0; e < count; e++)
            {
                long entry = ifd + 2 + e * 12L;
                if (U16(entry) != tag) continue;

                ushort type = U16(entry + 2);
                uint n = U32(entry + 4);
                int size = type switch { 1 or 6 or 7 => 1, 3 or 8 => 2, 4 or 9 => 4, _ => 0 };
                if (size == 0) throw new InvalidDataException($"TIFF tag {tag} has unexpected type {type}.");

                long valuesAt = n * size <= 4 ? entry + 8 : U32(entry + 8);
                var values = new uint[n];
                for (int k = 0; k < n; k++)
                    values[k] = size switch
                    {
                        1 => Slice(valuesAt + k, 1)[0],
                        2 => U16(valuesAt + k * 2L),
                        _ => U32(valuesAt + k * 4L),
                    };
                return values;
            }
            if (required) throw new InvalidDataException($"TIFF is missing required tag {tag}.");
            return [];
        }
    }
}

using System.Buffers.Binary;
using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>
/// Reads MRC files (the electron- and light-microscopy volume format; BioSR stores raw SIM frames in it) one plane at a
/// time from a seekable stream. Supported modes: 0 (int8), 1 (int16), 2 (float32), 6 (uint16) and 12 (float16).
///
/// <para><b>MRC in one paragraph.</b> A 1024-byte header of 4-byte words: NX, NY, NZ (columns, rows, sections) and
/// MODE (the sample type) come first; word 23 (NSYMBT) is the size of an optional extended header that follows; bytes
/// 212–215 (MACHST) give the byte order (0x44 = little-endian, 0x11 = big-endian). The samples follow, x fastest,
/// then y, then z, with no padding. Old files may leave MACHST zero; they are read little-endian unless that gives an
/// impossible size and big-endian doesn't.</para>
/// Rows are returned in file order (MRC's first row is conventionally the bottom of the image, so a viewer may show it
/// flipped against a TIFF of the same data; noisy and clean planes are read the same way, so pairs stay aligned).
/// </summary>
public sealed class MrcReader : IDisposable
{
    private const int HeaderBytes = 1024;
    private readonly Stream _stream;
    private readonly bool _leaveOpen, _littleEndian;
    private readonly long _dataStart;
    private readonly int _bytesPerSample;

    /// <summary>Opens an MRC file on a stream and checks that its header and size agree.</summary>
    /// <param name="stream">A readable, seekable stream holding the whole file.</param>
    /// <param name="leaveOpen">Leave the stream open when the reader is disposed.</param>
    /// <exception cref="InvalidDataException">Not an MRC file, or the file is shorter than its header says.</exception>
    /// <exception cref="NotSupportedException">A sample mode other than 0, 1, 2, 6 or 12 (e.g. complex data).</exception>
    public MrcReader(Stream stream, bool leaveOpen = false)
    {
        if (!stream.CanRead || !stream.CanSeek) throw new ArgumentException("An MRC stream must be readable and seekable.", nameof(stream));
        _stream = stream;
        _leaveOpen = leaveOpen;
        try
        {
            var header = new byte[HeaderBytes];
            stream.Position = 0;
            if (stream.ReadAtLeast(header, HeaderBytes, throwOnEndOfStream: false) < HeaderBytes)
                throw new InvalidDataException("Not an MRC file: shorter than the 1024-byte header.");
            _littleEndian = header[212] switch
            {
                0x44 => true,
                0x11 => false,
                _ => Plausible(header, little: true) || !Plausible(header, little: false),
            };
            Width = Word(header, 0);
            Height = Word(header, 1);
            PlaneCount = Word(header, 2);
            Mode = Word(header, 3);
            int extended = Word(header, 23);
            _bytesPerSample = Mode switch
            {
                0 => 1,
                1 or 6 or 12 => 2,
                2 => 4,
                _ => throw new NotSupportedException($"MRC mode {Mode} is not supported (DORA reads modes 0, 1, 2, 6 and 12)."),
            };
            if (!Plausible(header, _littleEndian) || extended < 0)
                throw new InvalidDataException($"Not an MRC file: implausible header ({Width} × {Height} × {PlaneCount}, extended header {extended}).");
            _dataStart = HeaderBytes + (long)extended;
            long needed = _dataStart + PlaneBytes * PlaneCount;
            if (stream.Length < needed)
                throw new InvalidDataException($"MRC file is truncated: its header needs {needed} bytes, the file has {stream.Length}.");
        }
        catch when (!leaveOpen)
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>Opens an MRC file for plane-by-plane reading.</summary>
    public static MrcReader Open(string path) =>
        new(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16, FileOptions.RandomAccess));

    /// <summary>True if the data starts like an MRC file this reader accepts (checked by plausibility; MRC has no magic number).</summary>
    public static bool LooksLikeMrc(ReadOnlySpan<byte> header) =>
        header.Length >= HeaderBytes && (Plausible(header, true) || Plausible(header, false)) &&
        (Word(header, 3, true) is 0 or 1 or 2 or 6 or 12 || Word(header, 3, false) is 0 or 1 or 2 or 6 or 12);

    /// <summary>Plane width (NX, columns).</summary>
    public int Width { get; }

    /// <summary>Plane height (NY, rows).</summary>
    public int Height { get; }

    /// <summary>Number of planes (NZ: z-sections, frames or raw SIM patterns, depending on the file).</summary>
    public int PlaneCount { get; }

    /// <summary>The MRC sample mode (0 int8, 1 int16, 2 float32, 6 uint16, 12 float16).</summary>
    public int Mode { get; }

    /// <summary>True if the samples are little-endian.</summary>
    public bool LittleEndian => _littleEndian;

    private long PlaneBytes => (long)Width * Height * _bytesPerSample;

    /// <summary>Reads one plane as raw sample values (no normalization).</summary>
    /// <exception cref="ArgumentOutOfRangeException">No such plane.</exception>
    public GrayImage ReadPlane(int plane)
    {
        if ((uint)plane >= (uint)PlaneCount)
            throw new ArgumentOutOfRangeException(nameof(plane), $"MRC file has {PlaneCount} plane(s).");
        var raw = new byte[PlaneBytes];
        _stream.Position = _dataStart + plane * PlaneBytes;
        _stream.ReadExactly(raw);
        var image = new GrayImage(Width, Height);
        var px = image.Pixels;
        bool le = _littleEndian;
        for (int i = 0; i < px.Length; i++)
        {
            var s = raw.AsSpan(i * _bytesPerSample);
            px[i] = Mode switch
            {
                0 => (sbyte)s[0],
                1 => le ? BinaryPrimitives.ReadInt16LittleEndian(s) : BinaryPrimitives.ReadInt16BigEndian(s),
                6 => le ? BinaryPrimitives.ReadUInt16LittleEndian(s) : BinaryPrimitives.ReadUInt16BigEndian(s),
                12 => (float)(le ? BinaryPrimitives.ReadHalfLittleEndian(s) : BinaryPrimitives.ReadHalfBigEndian(s)),
                _ => le ? BinaryPrimitives.ReadSingleLittleEndian(s) : BinaryPrimitives.ReadSingleBigEndian(s),
            };
        }
        return image;
    }

    private int Word(ReadOnlySpan<byte> header, int index) => Word(header, index, _littleEndian);

    private static int Word(ReadOnlySpan<byte> header, int index, bool little) =>
        little ? BinaryPrimitives.ReadInt32LittleEndian(header[(index * 4)..]) : BinaryPrimitives.ReadInt32BigEndian(header[(index * 4)..]);

    /// <summary>Sizes that a real file could have: positive, a plane under 2³¹ samples, a sane extended header.</summary>
    private static bool Plausible(ReadOnlySpan<byte> header, bool little)
    {
        int nx = Word(header, 0, little), ny = Word(header, 1, little), nz = Word(header, 2, little), ext = Word(header, 23, little);
        return nx > 0 && ny > 0 && nz > 0 && (long)nx * ny <= Array.MaxLength / 4 && ext >= 0 && ext < (1 << 30);
    }

    /// <summary>Closes the stream unless it was opened with <c>leaveOpen</c>.</summary>
    public void Dispose()
    {
        if (!_leaveOpen) _stream.Dispose();
    }
}

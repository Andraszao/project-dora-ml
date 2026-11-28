using System.Buffers.Binary;
using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>
/// Reads a TIFF one page at a time from a seekable stream, so a stack (time-lapse frames or z-slices) can be processed
/// frame by frame without ever holding the whole file in memory. Opening it walks the chain of page directories (IFDs)
/// and keeps only their offsets; <see cref="ReadPage"/> then reads just that page's tags and strips. Pages decode
/// exactly as <see cref="TiffCodec.Decode"/> decodes them from the whole file, with the same supported subset.
/// BigTIFF works the same way, so time-lapses over 4 GB stream too.
/// </summary>
public sealed class TiffStackReader : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _leaveOpen, _littleEndian, _big;
    private readonly List<long> _pages;

    /// <summary>Opens a TIFF on a stream.</summary>
    /// <param name="stream">A readable, seekable stream holding the whole TIFF (offsets in a TIFF are absolute).</param>
    /// <param name="leaveOpen">Leave the stream open when the reader is disposed.</param>
    /// <exception cref="ArgumentException">The stream can't read or seek.</exception>
    /// <exception cref="InvalidDataException">Not a TIFF, or its page directories are corrupt.</exception>
    public TiffStackReader(Stream stream, bool leaveOpen = false)
    {
        if (!stream.CanRead || !stream.CanSeek) throw new ArgumentException("A TIFF stream must be readable and seekable.", nameof(stream));
        _stream = stream;
        _leaveOpen = leaveOpen;
        try
        {
            _pages = TiffCodec.PageDirectories(stream, out _littleEndian, out _big);
        }
        catch when (!leaveOpen)
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>Opens a TIFF file for page-by-page reading.</summary>
    public static TiffStackReader Open(string path) =>
        new(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16, FileOptions.RandomAccess));

    /// <summary>Number of pages (1 for an ordinary single-image TIFF).</summary>
    public int PageCount => _pages.Count;

    /// <summary>True if the file is a BigTIFF (64-bit offsets, written for files over 4 GB).</summary>
    public bool IsBigTiff => _big;

    /// <summary>
    /// The first page's ImageDescription text, or null. ImageJ and tifffile write a stack's shape there
    /// (<see cref="StackLayout"/> reads it).
    /// </summary>
    public string? Description => _pages.Count == 0 ? null : TiffCodec.Description(_stream, _littleEndian, _big, _pages[0]);

    /// <summary>The first page's pixel size (XResolution, YResolution, ResolutionUnit), or null if it has none.</summary>
    public TiffResolution? Resolution => _pages.Count == 0 ? null : PageResolution(0);

    /// <summary>A page's pixel size, or null if it has none.</summary>
    /// <exception cref="ArgumentOutOfRangeException">No such page.</exception>
    public TiffResolution? PageResolution(int page)
    {
        if ((uint)page >= (uint)_pages.Count)
            throw new ArgumentOutOfRangeException(nameof(page), $"TIFF has only {_pages.Count} page(s).");
        return TiffCodec.Resolution(_stream, _littleEndian, _big, _pages[page]);
    }

    /// <summary>The first page's Software tag (305), or null: for a DORA output, the version, model and mode that made it.</summary>
    public string? Software => _pages.Count == 0 ? null : TiffCodec.Software(_stream, _littleEndian, _big, _pages[0]);

    /// <summary>A page's width and height, from its tags (nothing is decoded).</summary>
    /// <exception cref="ArgumentOutOfRangeException">No such page.</exception>
    public (int Width, int Height) PageSize(int page)
    {
        if ((uint)page >= (uint)_pages.Count)
            throw new ArgumentOutOfRangeException(nameof(page), $"TIFF has only {_pages.Count} page(s).");
        return TiffCodec.PageSize(_stream, _littleEndian, _big, _pages[page]);
    }

    /// <summary>Decodes one page into raw sample values (no normalization).</summary>
    /// <param name="page">Zero-based page index.</param>
    /// <exception cref="ArgumentOutOfRangeException">No such page.</exception>
    /// <exception cref="NotSupportedException">The page uses a feature the decoder deliberately omits (see <see cref="TiffCodec"/>).</exception>
    /// <exception cref="InvalidDataException">The page is truncated or inconsistent.</exception>
    public GrayImage ReadPage(int page)
    {
        if ((uint)page >= (uint)_pages.Count)
            throw new ArgumentOutOfRangeException(nameof(page), $"TIFF has only {_pages.Count} page(s).");
        return TiffCodec.DecodePage(_stream, _littleEndian, _big, _pages[page]);
    }

    /// <summary>Closes the stream unless it was opened with <c>leaveOpen</c>.</summary>
    public void Dispose()
    {
        if (!_leaveOpen) _stream.Dispose();
    }
}

/// <summary>
/// Writes a multi-page TIFF to a seekable stream one page at a time, so a denoised stack goes to disk frame by frame
/// instead of being assembled in memory. Layout, as in <see cref="TiffCodec.EncodeStack"/> (which uses this class):
/// header | strip₀ | IFD₀ | strip₁ | IFD₁ | …. Each IFD is written with a next-IFD offset of 0, and that offset is
/// patched in place when the next page arrives. So after every <see cref="Append"/> the file on disk is a complete,
/// valid TIFF of the pages so far: a run that stops halfway leaves a readable partial stack.
/// </summary>
public sealed class TiffStackWriter : IDisposable
{
    private readonly Stream _stream;
    private readonly TiffSampleType _sampleType;
    private readonly bool _deflate, _leaveOpen;
    private readonly TiffPageTags? _tags;   // description and software for the first page, resolution for every page
    private long _pointerAt = 4;   // where the offset of the next IFD goes: the header for page 0, then each IFD's tail

    /// <summary>Starts a TIFF on an empty stream: writes the 8-byte little-endian header.</summary>
    /// <param name="stream">A writable, seekable, empty stream.</param>
    /// <param name="sampleType">Storage format for every page.</param>
    /// <param name="deflate">Compress each page's strip with zlib/Deflate.</param>
    /// <param name="leaveOpen">Leave the stream open when the writer is disposed.</param>
    /// <param name="tags">
    /// Optional tags (brick 41): the ImageDescription and Software go on the first page (where ImageJ and tifffile read a
    /// stack's shape, so a denoised hyperstack reopens with its input's channels, slices and frames), the resolution on
    /// every page.
    /// </param>
    /// <exception cref="ArgumentException">The stream can't write or seek, or isn't empty.</exception>
    public TiffStackWriter(Stream stream, TiffSampleType sampleType = TiffSampleType.UInt16, bool deflate = false, bool leaveOpen = false,
        TiffPageTags? tags = null)
    {
        _tags = tags;
        if (!stream.CanWrite || !stream.CanSeek) throw new ArgumentException("A TIFF stream must be writable and seekable.", nameof(stream));
        if (stream.Length != 0) throw new ArgumentException("A TIFF must be written to an empty stream (its offsets are absolute).", nameof(stream));
        (_stream, _sampleType, _deflate, _leaveOpen) = (stream, sampleType, deflate, leaveOpen);
        Span<byte> header = stackalloc byte[8];
        "II"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[2..], 42);
        stream.Write(header);   // the first-IFD offset (bytes 4–7) is patched once IFD₀'s position is known
    }

    /// <summary>Creates (or overwrites) a TIFF file to write pages into.</summary>
    public static TiffStackWriter Create(string path, TiffSampleType sampleType = TiffSampleType.UInt16, bool deflate = false,
        TiffPageTags? tags = null) =>
        new(new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None, bufferSize: 1 << 16), sampleType, deflate,
            tags: tags);

    /// <summary>Pages written so far. A writer disposed before its first page leaves a header-only file, which no reader accepts.</summary>
    public int PageCount { get; private set; }

    /// <summary>Appends one page (pages may differ in size).</summary>
    /// <exception cref="ArgumentException">The file would exceed classic TIFF's 4 GB offset limit.</exception>
    public void Append(GrayImage page)
    {
        long stripAt = _stream.Length;
        if (stripAt > uint.MaxValue) throw new ArgumentException("Stack is too large for a TIFF (4 GB).", nameof(page));
        var tags = PageCount == 0 ? _tags : _tags?.Resolution is { } resolution ? new TiffPageTags { Resolution = resolution } : null;
        var (strip, extra, extraAt, ifd) = TiffCodec.EncodePage(page, _sampleType, _deflate, (uint)stripAt, tags);
        _stream.Position = stripAt;
        _stream.Write(strip);
        if (_stream.Length % 2 == 1) _stream.WriteByte(0);   // tag values and IFDs start on a word boundary
        if (extra.Length > 0)
        {
            if (_stream.Length != extraAt) throw new InvalidOperationException("TIFF tag values offset mismatch.");
            _stream.Write(extra);   // already an even number of bytes
        }
        long ifdOffset = _stream.Length;
        if (ifdOffset + ifd.Length > uint.MaxValue) throw new ArgumentException("Stack is too large for a TIFF (4 GB).", nameof(page));
        _stream.Write(ifd);
        Patch(_pointerAt, (uint)ifdOffset);
        _pointerAt = ifdOffset + ifd.Length - 4;   // stays 0 if this is the last page
        PageCount++;
    }

    private void Patch(long at, uint value)
    {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        long end = _stream.Position;
        _stream.Position = at;
        _stream.Write(bytes);
        _stream.Position = end;
    }

    /// <summary>Flushes, and closes the stream unless it was opened with <c>leaveOpen</c>.</summary>
    public void Dispose()
    {
        _stream.Flush();
        if (!_leaveOpen) _stream.Dispose();
    }
}

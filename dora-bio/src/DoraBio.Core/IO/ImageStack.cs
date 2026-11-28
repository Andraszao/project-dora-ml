using System.Globalization;
using System.Text.Json;
using DoraBio.Core.Data;

namespace DoraBio.Core.IO;

/// <summary>Where a stack's shape was read from.</summary>
public enum StackMetadata
{
    /// <summary>No shape metadata: every page (or MRC section) is one plane along a single axis.</summary>
    None,

    /// <summary>ImageJ's ImageDescription ("ImageJ=…", images/channels/slices/frames).</summary>
    ImageJ,

    /// <summary>tifffile's JSON ImageDescription ({"shape": […], "axes": "…"}).</summary>
    Tifffile,

    /// <summary>An MRC header (NZ sections).</summary>
    Mrc,

    /// <summary>
    /// OME-XML in the ImageDescription (brick 41). DORA doesn't read it, so the planes are unlabelled; a stack of more
    /// than one plane is refused until <c>--axes</c> names its axis (<see cref="StackLayout.OmeXmlMessage"/>), so an
    /// OME Z-stack can't pass for an unlabelled time-lapse.
    /// </summary>
    OmeXml,
}

/// <summary>
/// The shape of a stack's planes: one letter per non-YX axis, outermost first, as the planes are stored (the last axis
/// varies fastest from one page to the next). Axes of size 1 are left out. Letters: <c>T</c> time, <c>Z</c> depth,
/// <c>C</c> channel, <c>S</c> separate fields of view, <c>P</c> raw SIM patterns, <c>I</c> planes whose meaning the
/// file doesn't say.
/// </summary>
/// <param name="Axes">Axis letters (empty for a single plane), e.g. "TZC" or "SC".</param>
/// <param name="Shape">The size of each axis, in the same order.</param>
/// <param name="Source">Where the shape came from.</param>
public sealed record StackLayout(string Axes, int[] Shape, StackMetadata Source)
{
    /// <summary>The axis letters a layout may use.</summary>
    public const string Letters = "TZCSPI";

    /// <summary>The refusal for a multi-plane TIFF whose planes are described in OME-XML and no axes were declared.</summary>
    public const string OmeXmlMessage =
        "this TIFF describes its planes in OME-XML, which DORA doesn't read, so it can't tell frames from slices or channels. " +
        "Pass --axes TYX if the planes are one time-lapse (ZYX for a volume, CYX for channels), or, for a hyperstack, open it " +
        "in Fiji (Bio-Formats) and save it as TIFF, which writes the ImageJ header DORA reads";

    /// <summary>Number of planes (the product of the shape; 1 for a single plane).</summary>
    public long PlaneCount => Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>The axes as usually written, including the image axes: e.g. "TZCYX".</summary>
    public string FullAxes => Axes + "YX";

    /// <summary>Size of <paramref name="axis"/> (1 if the stack doesn't have it).</summary>
    public int Size(char axis)
    {
        int k = Axes.IndexOf(axis);
        return k < 0 ? 1 : Shape[k];
    }

    /// <summary>Human-readable shape, e.g. "S=120 × C=3".</summary>
    public override string ToString() =>
        Axes.Length == 0 ? "single plane" : string.Join(" × ", Axes.Select((a, k) => $"{a}={Shape[k]}"));

    /// <summary>
    /// Reads a TIFF stack's layout from its ImageDescription (ImageJ or tifffile) and checks it against the number of
    /// pages actually in the file. Without either kind of metadata, every page is one plane along axis I.
    /// </summary>
    /// <param name="description">The first page's ImageDescription, or null.</param>
    /// <param name="pageCount">Pages in the file.</param>
    /// <exception cref="InvalidDataException">The metadata contradicts itself or the page count.</exception>
    /// <exception cref="NotSupportedException">An ImageJ stack stored with a single page directory (ImageJ does this
    /// for files over 4 GB); re-save it as a BigTIFF.</exception>
    /// <remarks>OME-XML is recognized (<see cref="StackMetadata.OmeXml"/>) but not read: its planes come back unlabelled.</remarks>
    public static StackLayout FromTiff(string? description, int pageCount)
    {
        if (pageCount < 1) throw new InvalidDataException("A TIFF stack needs at least one page.");
        var trimmed = description?.TrimStart();
        if (trimmed is not null && trimmed.StartsWith("ImageJ=", StringComparison.Ordinal))
            return FromImageJ(trimmed, pageCount);
        // Checked before the unlabelled fallback: OME-XML labels its planes, just not in a form DORA reads.
        if (trimmed is not null && (trimmed.StartsWith("<?xml", StringComparison.Ordinal) || trimmed.Contains("<OME", StringComparison.Ordinal)))
            return Unlabelled(pageCount, StackMetadata.OmeXml);
        if (trimmed is not null && trimmed.StartsWith('{') && trimmed.Contains("\"shape\"", StringComparison.Ordinal))
            return FromTifffile(trimmed, pageCount);
        return Unlabelled(pageCount, StackMetadata.None);
    }

    /// <summary>The layout of an MRC file: its sections along one axis (I until declared otherwise).</summary>
    public static StackLayout FromMrc(int sections) => Unlabelled(sections, StackMetadata.Mrc);

    private static StackLayout Unlabelled(int planes, StackMetadata source) =>
        planes == 1 ? new("", [], source) : new("I", [planes], source);

    /// <summary>
    /// ImageJ's hyperstack order is CZT: channels vary fastest, then slices, then frames. So the stored shape is
    /// [frames, slices, channels], outermost first. "images" must equal their product and the page count.
    /// </summary>
    private static StackLayout FromImageJ(string description, int pageCount)
    {
        var values = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var line in description.Split('\n'))
        {
            int eq = line.IndexOf('=');
            if (eq > 0 && long.TryParse(line[(eq + 1)..].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out long v))
                values[line[..eq].Trim()] = v;
        }
        long Get(string key) => values.TryGetValue(key, out long v) ? v : 1;
        long frames = Get("frames"), slices = Get("slices"), channels = Get("channels");
        long images = values.TryGetValue("images", out long n) ? n : frames * slices * channels;
        if (frames < 1 || slices < 1 || channels < 1 || frames * slices * channels != images)
            throw new InvalidDataException(
                $"ImageJ metadata is inconsistent: images={images} but channels={channels} × slices={slices} × frames={frames}.");
        if (images != pageCount)
        {
            if (pageCount == 1 && images > 1)
                throw new NotSupportedException(
                    $"This ImageJ stack declares {images} images but stores one page directory (ImageJ does that for files " +
                    "over 4 GB). Re-save it as a BigTIFF, e.g. with Fiji's Bio-Formats exporter or tifffile.");
            throw new InvalidDataException($"ImageJ metadata says {images} images, but the file has {pageCount} pages.");
        }
        return Compact("TZC", [frames, slices, channels], StackMetadata.ImageJ);
    }

    /// <summary>tifffile records the full array shape (image axes last) and, when it was given one, an axes string.</summary>
    private static StackLayout FromTifffile(string description, int pageCount)
    {
        int[] shape;
        string? axes = null;
        try
        {
            using var doc = JsonDocument.Parse(description);
            shape = doc.RootElement.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            if (doc.RootElement.TryGetProperty("axes", out var a) && a.ValueKind == JsonValueKind.String) axes = a.GetString();
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException or KeyNotFoundException)
        {
            throw new InvalidDataException($"tifffile metadata is not valid JSON with a shape: {e.Message}");
        }
        if (axes is not null && axes.Length != shape.Length)
            throw new InvalidDataException($"tifffile metadata has shape [{string.Join(", ", shape)}] but axes '{axes}'.");
        int image = axes is not null && axes.EndsWith('S') ? 3 : 2;   // a trailing S is samples per pixel (RGB), part of a page
        if (shape.Length < image || shape.Any(d => d < 1))
            throw new InvalidDataException($"tifffile metadata has an impossible shape [{string.Join(", ", shape)}].");
        var outer = shape[..^image].Select(d => (long)d).ToArray();
        string letters = axes is null
            ? new string('I', outer.Length)
            : new string(axes[..^image].Select(c => Letters.Contains(c) ? c : 'I').ToArray());
        var layout = Compact(letters, outer, StackMetadata.Tifffile);
        if (layout.PlaneCount != pageCount)
            throw new InvalidDataException($"tifffile metadata describes {layout.PlaneCount} planes, but the file has {pageCount} pages.");
        if (layout.Axes.Distinct().Count() != layout.Axes.Length)   // e.g. two unlabelled axes: merge rather than guess
            return new("I", [pageCount], StackMetadata.Tifffile);
        return layout;
    }

    /// <summary>Drops axes of size 1.</summary>
    private static StackLayout Compact(string letters, long[] sizes, StackMetadata source)
    {
        var keep = Enumerable.Range(0, letters.Length).Where(k => sizes[k] != 1).ToArray();
        if (keep.Any(k => sizes[k] > int.MaxValue)) throw new InvalidDataException("Stack axis is too long.");
        return new(new string(keep.Select(k => letters[k]).ToArray()), keep.Select(k => (int)sizes[k]).ToArray(), source);
    }

    /// <summary>
    /// The same planes under the axis letters someone who knows the data declares (ImageJ calls fields of view
    /// "slices", and a 3D patch may be stored as "channels"). The declared axes must have exactly as many non-image
    /// axes as the file, in the same order; their sizes come from the file.
    /// </summary>
    /// <param name="declared">Axes ending in "YX", e.g. "SCYX", "ZYX" or "TYX"; just "YX" for a single plane.</param>
    /// <exception cref="ArgumentException">Malformed axes.</exception>
    /// <exception cref="InvalidDataException">The file's layout has a different number of axes.</exception>
    public StackLayout Declare(string declared)
    {
        if (!declared.EndsWith("YX", StringComparison.Ordinal))
            throw new ArgumentException($"Axes '{declared}' must end in YX (the image axes).", nameof(declared));
        string letters = declared[..^2];
        if (letters.Any(c => !Letters.Contains(c)) || letters.Distinct().Count() != letters.Length)
            throw new ArgumentException($"Axes '{declared}': use each of {Letters} at most once, then YX.", nameof(declared));
        if (letters.Length != Axes.Length)
            throw new InvalidDataException(
                $"The file's planes are laid out as {this} ({Source}), which doesn't fit the declared axes {declared}.");
        return this with { Axes = letters };
    }
}

/// <summary>
/// A multi-plane image file opened for plane-by-plane reading: a TIFF stack (classic or BigTIFF, with ImageJ or tifffile
/// shape metadata) or an MRC file. Only the planes asked for are read, so time-lapses of thousands of frames and
/// volumes of gigabytes open instantly and stream in bounded memory. Planes are addressed by axis
/// (<see cref="PlaneIndex"/>), so a caller asks for "frame 12 of channel 1" rather than computing page numbers.
/// </summary>
public sealed class ImageStack : IDisposable
{
    private readonly TiffStackReader? _tiff;
    private readonly MrcReader? _mrc;

    private ImageStack(TiffStackReader? tiff, MrcReader? mrc, StackLayout layout, int width, int height, string format)
    {
        (_tiff, _mrc, Layout, Width, Height, Format) = (tiff, mrc, layout, width, height, format);
    }

    /// <summary>
    /// Opens a stack. The layout comes from the file's own metadata; <paramref name="axes"/> (e.g. "TYX") names the
    /// axes when the caller knows better than the file, and must agree with it in shape.
    /// </summary>
    /// <param name="path">A .tif/.tiff or .mrc file.</param>
    /// <param name="axes">Declared axes ending in "YX", or null to keep the file's own labels.</param>
    /// <exception cref="InvalidDataException">Not a supported stack, or its metadata contradicts itself or the declaration.</exception>
    /// <exception cref="NotSupportedException">A TIFF/MRC variant DORA deliberately doesn't read.</exception>
    public static ImageStack Open(string path, string? axes = null)
    {
        if (axes is not null && !axes.EndsWith("YX", StringComparison.Ordinal))
            throw new ArgumentException($"--axes '{axes}' must end in YX, e.g. TYX, ZYX or TZCYX.");
        var head = new byte[1024];
        int read;
        using (var probe = File.OpenRead(path)) read = probe.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        if (TiffCodec.HasTiffSignature(head.AsSpan(0, read)))
        {
            var tiff = TiffStackReader.Open(path);
            try
            {
                return FromTiff(tiff, axes, Path.GetFileName(path));
            }
            catch
            {
                tiff.Dispose();
                throw;
            }
        }
        if (Path.GetExtension(path).Equals(".mrc", StringComparison.OrdinalIgnoreCase) && MrcReader.LooksLikeMrc(head.AsSpan(0, read)))
        {
            var mrc = MrcReader.Open(path);
            try
            {
                var layout = StackLayout.FromMrc(mrc.PlaneCount);
                if (axes is not null) layout = layout.Declare(axes);
                return new ImageStack(null, mrc, layout, mrc.Width, mrc.Height, "MRC");
            }
            catch
            {
                mrc.Dispose();
                throw;
            }
        }
        throw new InvalidDataException($"{Path.GetFileName(path)}: not a TIFF or MRC stack.");
    }

    /// <summary>
    /// Wraps an open TIFF reader (it is disposed with the stack), its layout read from its own metadata as in
    /// <see cref="Open"/>, or declared by <paramref name="axes"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">The metadata contradicts itself, the page count or the declaration.</exception>
    /// <exception cref="NotSupportedException">An ImageJ stack over 4 GB that stores one page directory, or a multi-plane
    /// OME-XML TIFF without <paramref name="axes"/>.</exception>
    public static ImageStack FromTiff(TiffStackReader tiff, string? axes = null) => FromTiff(tiff, axes, null);

    private static ImageStack FromTiff(TiffStackReader tiff, string? axes, string? name)
    {
        var layout = StackLayout.FromTiff(tiff.Description, tiff.PageCount);
        if (layout.Source == StackMetadata.OmeXml && layout.PlaneCount > 1 && axes is null)
            throw new NotSupportedException($"{(name is null ? "" : name + ": ")}{StackLayout.OmeXmlMessage}.");
        if (axes is not null) layout = layout.Declare(axes);
        var (w, h) = tiff.PageSize(0);
        return new ImageStack(tiff, null, layout, w, h, tiff.IsBigTiff ? "BigTIFF" : "TIFF");
    }

    /// <summary>The planes' axes and shape.</summary>
    public StackLayout Layout { get; }

    /// <summary>
    /// The tags a denoised copy of this stack keeps (brick 41), never ones that would contradict a single-sample float32
    /// copy: an ImageJ header verbatim (it names no pixel type; its frames, unit and display range still hold); a tifffile
    /// header when its shape is this stack's (see <see cref="TifffileHeaderFor"/>); never OME-XML, which records the
    /// input's pixel type and files. A file without either header (an OME-XML stack, or plain pages) whose axes were
    /// declared gets a tifffile header with them (see <see cref="DeclaredHeader"/>), so the copy reopens with the axes the
    /// user named. Also the first page's pixel size. <paramref name="software"/> says which DORA made it.
    /// </summary>
    public TiffPageTags OutputTags(string? software) => new()
    {
        Description = Layout.Source switch
        {
            StackMetadata.ImageJ => _tiff?.Description,
            StackMetadata.Tifffile => TifffileHeaderFor(_tiff?.Description, Width, Height),
            _ => DeclaredHeader(Layout, Width, Height),
        },
        Software = software,
        Resolution = _tiff?.Resolution,
    };

    /// <summary>The first page's storage (bit depth, sample format, compression, predictor), for a TIFF; null for MRC.</summary>
    public TiffPageFormat? Storage => _tiff?.PageFormat(0);

    /// <summary>The first page's resolution tags, for a TIFF that has them.</summary>
    public TiffResolution? Resolution => _tiff?.Resolution;

    /// <summary>
    /// The first page's pixel size in words, e.g. "0.65 × 0.65 um per pixel", or null when the file records none
    /// (tifffile's default 1 × 1 with no unit counts as none). TIFF stores pixels per unit; ImageJ writes unit 1 (none)
    /// and names its unit in the description ("unit=um").
    /// </summary>
    public string? PixelSize
    {
        get
        {
            if (Resolution is not { X: > 0, Y: > 0 } r || (r.Unit == 1 && r.X == 1 && r.Y == 1)) return null;
            string? named = Layout.Source == StackMetadata.ImageJ
                ? _tiff?.Description?.Split('\n').FirstOrDefault(l => l.StartsWith("unit=", StringComparison.Ordinal))?[5..].Trim()
                : null;
            string unit = r.Unit switch { 2 => "inch", 3 => "cm", _ => string.IsNullOrEmpty(named) ? "units (no unit recorded)" : named };
            return string.Create(CultureInfo.InvariantCulture, $"{1 / r.X:G4} × {1 / r.Y:G4} {unit} per pixel");
        }
    }

    /// <summary>What a denoised copy does with this stack's ImageDescription (<see cref="OutputTags"/>), in words, for the report.</summary>
    public string DescriptionCarried => Layout.Source switch
    {
        StackMetadata.ImageJ => "the ImageJ header, verbatim",
        StackMetadata.Tifffile => TifffileHeaderFor(_tiff?.Description, Width, Height) is null
            ? "none: the tifffile header's shape doesn't describe the single-sample float32 copy"
            : "the tifffile header (shape and axes only)",
        _ => (DeclaredHeader(Layout, Width, Height) is null ? "none" : "the declared axes, as a tifffile header") +
             (Layout.Source == StackMetadata.OmeXml ? " (the OME-XML isn't copied: it states the input's pixel type)" : ""),
    };

    /// <summary>
    /// tifffile's JSON header for planes whose axes were declared (any letter but I, the unlabelled axis), e.g.
    /// {"shape": [4, 64, 96], "axes": "ZYX"}; null when nothing was declared, so an unlabelled input stays unlabelled
    /// rather than gaining labels nobody gave it.
    /// </summary>
    internal static string? DeclaredHeader(StackLayout layout, int width, int height) =>
        layout.Axes.All(c => c == 'I') ? null
            : JsonSerializer.Serialize(new { shape = layout.Shape.Append(height).Append(width).ToArray(), axes = layout.Axes + "YX" });

    /// <summary>
    /// tifffile's JSON header for a copy: verbatim when it holds only "shape" and "axes" and its shape ends in this
    /// stack's height and width (no samples axis: an RGB input's copy is gray); rewritten with just those two keys when it
    /// holds others (some tools add a dtype or their own fields, which the copy needn't match); null when its shape
    /// doesn't describe the copy's pages.
    /// </summary>
    internal static string? TifffileHeaderFor(string? description, int width, int height)
    {
        if (description is null) return null;
        try
        {
            using var doc = JsonDocument.Parse(description);
            var root = doc.RootElement;
            int[] shape = root.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            string? axes = root.TryGetProperty("axes", out var a) && a.ValueKind == JsonValueKind.String ? a.GetString() : null;
            if (shape.Length < 2 || shape[^2] != height || shape[^1] != width || axes?.EndsWith('S') == true) return null;
            if (root.EnumerateObject().All(p => p.Name is "shape" or "axes")) return description;
            return axes is null ? JsonSerializer.Serialize(new { shape }) : JsonSerializer.Serialize(new { shape, axes });
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException or KeyNotFoundException)
        {
            return null;
        }
    }

    /// <summary>"TIFF", "BigTIFF" or "MRC".</summary>
    public string Format { get; }

    /// <summary>Width of the first plane (planes of a TIFF may in principle differ; DORA's datasets don't).</summary>
    public int Width { get; }

    /// <summary>Height of the first plane.</summary>
    public int Height { get; }

    /// <summary>Number of planes in the file.</summary>
    public int PlaneCount => (int)Layout.PlaneCount;

    /// <summary>
    /// The storage index of the plane at the given position on each axis (axes the stack doesn't have must be 0):
    /// row-major over <see cref="StackLayout.Shape"/>, which is how both ImageJ and tifffile order pages.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A position is outside its axis.</exception>
    public int PlaneIndex(int t = 0, int z = 0, int c = 0, int s = 0, int p = 0, int i = 0)
    {
        int index = 0;
        foreach (var (axis, value) in new[] { ('T', t), ('Z', z), ('C', c), ('S', s), ('P', p), ('I', i) })
        {
            if (value < 0 || value >= Layout.Size(axis))
                throw new ArgumentOutOfRangeException(axis.ToString(), $"{axis}={value} is outside the stack ({Layout}).");
        }
        var position = new Dictionary<char, int> { ['T'] = t, ['Z'] = z, ['C'] = c, ['S'] = s, ['P'] = p, ['I'] = i };
        for (int k = 0; k < Layout.Axes.Length; k++) index = index * Layout.Shape[k] + position[Layout.Axes[k]];
        return index;
    }

    /// <summary>Reads one plane by storage index, as raw sample values (no normalization).</summary>
    /// <exception cref="ArgumentOutOfRangeException">No such plane.</exception>
    public GrayImage ReadPlane(int index)
    {
        if ((uint)index >= (uint)PlaneCount) throw new ArgumentOutOfRangeException(nameof(index), $"The stack has {PlaneCount} planes.");
        return _tiff is not null ? _tiff.ReadPage(index) : _mrc!.ReadPlane(index);
    }

    /// <summary>Every Z plane at the given position on the other axes (one plane if the stack has no Z axis).</summary>
    public GrayImage[] ReadVolume(int t = 0, int c = 0, int s = 0, int p = 0) =>
        Enumerable.Range(0, Layout.Size('Z')).Select(z => ReadPlane(PlaneIndex(t, z, c, s, p))).ToArray();

    /// <summary>The T planes at the given position on the other axes, read one at a time as they're enumerated.</summary>
    public IEnumerable<GrayImage> Frames(int z = 0, int c = 0, int s = 0, int p = 0)
    {
        for (int t = 0; t < Layout.Size('T'); t++) yield return ReadPlane(PlaneIndex(t, z, c, s, p));
    }

    /// <summary>Closes the file.</summary>
    public void Dispose()
    {
        _tiff?.Dispose();
        _mrc?.Dispose();
    }
}

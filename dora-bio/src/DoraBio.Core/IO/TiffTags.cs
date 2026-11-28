namespace DoraBio.Core.IO;

/// <summary>
/// A page's pixel size as TIFF records it: XResolution and YResolution (tags 282, 283) are pixels per unit, as
/// rationals, and ResolutionUnit (296) says which unit: 1 none, 2 inch, 3 centimetre. ImageJ and Fiji store a
/// calibrated pixel size this way (with the unit name, e.g. micron, in their ImageDescription).
/// </summary>
/// <param name="XNumerator">XResolution numerator.</param>
/// <param name="XDenominator">XResolution denominator.</param>
/// <param name="YNumerator">YResolution numerator.</param>
/// <param name="YDenominator">YResolution denominator.</param>
/// <param name="Unit">ResolutionUnit (2 when the file has none, as TIFF defines).</param>
public readonly record struct TiffResolution(uint XNumerator, uint XDenominator, uint YNumerator, uint YDenominator, ushort Unit)
{
    /// <summary>Pixels per unit along x (0 for a zero denominator).</summary>
    public double X => XDenominator == 0 ? 0 : (double)XNumerator / XDenominator;

    /// <summary>Pixels per unit along y.</summary>
    public double Y => YDenominator == 0 ? 0 : (double)YNumerator / YDenominator;

    /// <summary>E.g. "5.2 × 5.2 pixels per unit (none)".</summary>
    public override string ToString()
    {
        string unit = Unit switch { 1 => "unit (none)", 2 => "inch", 3 => "cm", _ => $"unit {Unit}" };
        return string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{X:G6} × {Y:G6} pixels per {unit}");
    }
}

/// <summary>
/// How a page's pixels are stored (brick 41, for <c>denoise --report</c>): the tags a reader decodes them by.
/// </summary>
/// <param name="BitsPerSample">BitsPerSample (258): 8, 16 or 32.</param>
/// <param name="SampleFormat">SampleFormat (339): 1 unsigned integer, 2 signed integer, 3 floating point.</param>
/// <param name="SamplesPerPixel">SamplesPerPixel (277): 1 for gray, 3 or 4 for RGB(A) (DORA reads their luminance).</param>
/// <param name="Compression">Compression (259): 1 none, 5 LZW, 8 or 32946 Deflate, 32773 PackBits.</param>
/// <param name="Predictor">Predictor (317): 1 none, 2 horizontal differencing.</param>
public sealed record TiffPageFormat(int BitsPerSample, uint SampleFormat, int SamplesPerPixel, uint Compression, uint Predictor)
{
    /// <summary>E.g. "16-bit unsigned integer, LZW with the horizontal predictor".</summary>
    public override string ToString()
    {
        string kind = SampleFormat switch { 2 => "signed integer", 3 => "floating point", _ => "unsigned integer" };
        string samples = SamplesPerPixel == 1 ? "" : $", {SamplesPerPixel} samples per pixel";
        string predictor = Predictor == 2 ? " with the horizontal predictor" : "";
        return $"{BitsPerSample}-bit {kind}{samples}, {TiffCodec.CompressionName(Compression)}{predictor}";
    }
}

/// <summary>
/// The tags a written page carries besides its pixels (brick 41): what an output keeps of its input, and what says
/// which DORA made it. The description and software go on the first page, the resolution on every page.
/// </summary>
public sealed record TiffPageTags
{
    /// <summary>
    /// ImageDescription (270), written as Latin-1: the input's ImageJ or tifffile header, verbatim, so the output reopens
    /// with the input's channels, slices, frames and calibration. Null for none.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Software (305), ASCII: DORA's version, the model's content hash and how it denoised. Provenance goes here rather
    /// than into the ImageDescription, which ImageJ parses as its own key=value block.
    /// </summary>
    public string? Software { get; init; }

    /// <summary>The pixel size (282, 283, 296), or null for none.</summary>
    public TiffResolution? Resolution { get; init; }
}

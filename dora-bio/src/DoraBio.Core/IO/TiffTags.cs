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

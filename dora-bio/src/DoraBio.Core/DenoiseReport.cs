using System.Globalization;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using DoraBio.Core.IO;

namespace DoraBio.Core;

/// <summary>Everything a denoise report states (brick 41), gathered by <see cref="Pipeline.Denoise"/>.</summary>
public sealed record DenoiseReportData
{
    /// <summary>The input file.</summary>
    public required string Input { get; init; }

    /// <summary>The denoised output file.</summary>
    public required string Output { get; init; }

    /// <summary>The bilateral baseline file, if one was written.</summary>
    public string? Baseline { get; init; }

    /// <summary>"TIFF", "BigTIFF", "MRC" or "PNG".</summary>
    public required string Format { get; init; }

    /// <summary>Plane size.</summary>
    public required (int Width, int Height) Size { get; init; }

    /// <summary>The planes' layout as DORA used it (after any declared axes).</summary>
    public required StackLayout Layout { get; init; }

    /// <summary>The axes the user declared, or null.</summary>
    public string? DeclaredAxes { get; init; }

    /// <summary>The first page's storage (bit depth, sample format, compression, predictor), for a TIFF.</summary>
    public TiffPageFormat? Storage { get; init; }

    /// <summary>The first page's pixel size in words (<see cref="ImageStack.PixelSize"/>), or null if the file records none.</summary>
    public string? PixelSize { get; init; }

    /// <summary>The output's Software tag.</summary>
    public string? Software { get; init; }

    /// <summary>Whether the input's ImageDescription was carried over (and in what form).</summary>
    public required string DescriptionCarried { get; init; }

    /// <summary>The main model.</summary>
    public required LoadedModel Model { get; init; }

    /// <summary>The correlated-noise model, if any.</summary>
    public LoadedModel? CorrelatedModel { get; init; }

    /// <summary>Planes that went to the correlated-noise model.</summary>
    public int CorrelatedPlanes { get; init; }

    /// <summary>Planes denoised.</summary>
    public required int Planes { get; init; }

    /// <summary>Tile core size.</summary>
    public required int TileCore { get; init; }

    /// <summary>Variance-stabilized models.</summary>
    public bool Vst { get; init; }

    /// <summary>Wall-clock seconds.</summary>
    public required double Seconds { get; init; }

    /// <summary>Peak resident memory, bytes.</summary>
    public required long PeakMemory { get; init; }

    /// <summary>Every notice the run raised.</summary>
    public required IReadOnlyList<string> Notices { get; init; }

    /// <summary>The residual statistics over every plane.</summary>
    public required ResidualCheck Residual { get; init; }

    /// <summary>The picture of the middle plane (PNG bytes), or null.</summary>
    public byte[]? Picture { get; init; }

    /// <summary>Which plane the picture shows.</summary>
    public int PicturePlane { get; init; }
}

/// <summary>
/// <c>denoise --report FILE.html</c> (brick 41): one self-contained HTML page (styles inline, the picture embedded) that
/// says what was done to an image, with what, and what to look at before trusting it, so a pilot user can keep it with
/// the data or send it to a colleague. Nothing in it needs DORA, the network or the original files to read.
/// </summary>
public static class DenoiseReport
{
    /// <summary>How big a brightness change has to be before the report flags it for intensity measurements.</summary>
    public const double NetChangeFlag = 0.02;

    /// <summary>Writes the report to <paramref name="path"/>.</summary>
    public static void Write(string path, DenoiseReportData d) => File.WriteAllText(path, Render(d), new UTF8Encoding(false));

    /// <summary>The report's HTML.</summary>
    public static string Render(DenoiseReportData d)
    {
        var ci = CultureInfo.InvariantCulture;
        string E(string? s) => WebUtility.HtmlEncode(s ?? "");
        var html = new StringBuilder();
        void Row(string key, string value) => html.Append(ci, $"<tr><th>{E(key)}</th><td>{value}</td></tr>\n");
        var r = d.Residual;
        bool brightness = Math.Abs(r.NetChange) > NetChangeFlag;

        html.Append("""
            <!doctype html>
            <html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
            <title>DORA denoise report</title>
            <style>
            body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; padding: 0 1rem; color: #1b1b1b; background: #fff; }
            h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; }
            table { border-collapse: collapse; width: 100%; } th { text-align: left; width: 16rem; vertical-align: top; color: #555; font-weight: 600; }
            td, th { padding: .2rem .5rem .2rem 0; } code { font-size: .9em; word-break: break-all; }
            .flag { background: #fff4d6; border-left: 4px solid #d99a00; padding: .5rem .8rem; }
            .ok { background: #eaf6ea; border-left: 4px solid #3a8a3a; padding: .5rem .8rem; }
            .strip { max-width: 100%; } .strip img { display: block; width: 100%; height: auto; image-rendering: pixelated; }
            .labels { display: grid; font-size: .85rem; color: #555; text-align: center; }
            @media (prefers-color-scheme: dark) { body { background: #141414; color: #e8e8e8; } th { color: #aaa; }
              .flag { background: #3a2f10; } .ok { background: #16301a; } h2 { border-color: #333; } }
            </style></head><body>
            """);
        html.Append(ci, $"<h1>DORA denoise report: {E(Path.GetFileName(d.Input))}</h1>\n");
        html.Append(ci, $"<p>Made {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC by DORA {E(Pipeline.Version)}.</p>\n");

        html.Append("<h2>Verdict</h2>\n");
        string routing = d.CorrelatedModel is null ? "one model (no correlated-noise model given)"
            : $"{d.CorrelatedPlanes} of {d.Planes} planes went to the correlated-noise model, the rest to the main one";
        html.Append(ci, $"<p>{d.Planes} plane{(d.Planes == 1 ? "" : "s")} denoised in {d.Seconds:0.0} s; {E(routing)}.</p>\n");
        foreach (string notice in d.Notices) html.Append(ci, $"<p class=\"flag\">{E(notice)}</p>\n");

        html.Append("<h2>Look at the residual before you measure</h2>\n");
        if (d.Picture is { } png)
        {
            // Shown at a whole multiple of its size (nearest-neighbour, never smoothed: blurred grain would look correlated),
            // up to the page width; a picture wider than the page is scaled down to fit. Only integers here, so no culture.
            int panel = d.Size.Width, gap = ResidualCheck.PictureGap, total = 3 * panel + 2 * gap, scale = Math.Max(1, 900 / total);
            html.Append($"<div class=\"strip\" style=\"width: {scale * total}px\"><div class=\"labels\" style=\"grid-template-columns: " +
                            $"{panel}fr {gap}fr {panel}fr {gap}fr {panel}fr\"><span>Raw</span><span></span><span>Denoised</span><span></span>" +
                            $"<span>Residual (raw − denoised)</span></div>\n<img alt=\"raw, denoised and residual of plane {d.PicturePlane + 1}\" " +
                            $"src=\"data:image/png;base64,{Convert.ToBase64String(png)}\"></div>\n");
            string shown = scale == 1 ? "actual size, unsmoothed" : $"{scale}×{scale} screen pixels per image pixel, unsmoothed";
            html.Append($"<p>Plane {d.PicturePlane + 1} of {d.Planes} ({shown}). Raw and " +
                            "denoised share one display window. In the residual, mid-gray is 0, brighter means DORA removed brightness there " +
                            "and darker that it added some; it is clipped at the 99th percentile of its size.</p>\n");
        }
        html.Append("<p>The residual should look like noise: grain with no edges, outlines or texture. Anything you can recognize in it is " +
                    "signal DORA removed, so check those structures in the raw image before measuring them.</p>\n<table>\n");
        Row("Residual vs output, correlation", Num(r.ResidualOutputCorrelation, "+0.000;-0.000") +
            " (about 0 when only noise was removed; positive: bright structure was shrunk; negative: it was exaggerated)");
        Row("Residual, neighbour correlation", $"x {Num(r.LagX, "+0.000;-0.000")}, y {Num(r.LagY, "+0.000;-0.000")} (about 0 for independent noise; " +
                                               "positive: structure, or noise correlated between pixels, was removed)");
        Row("Net brightness change", Math.Abs(r.NetChange) < 0.0005 ? "none (under 0.05% of the mean)"
            : (r.NetChange * -100).ToString("+0.0;-0.0", ci) + "% of the mean (the output is " + (r.NetChange > 0 ? "darker" : "brighter") + ")");
        Row("Typical change per pixel", (r.RmsChange * 100).ToString("0.0", ci) + "% of the raw intensity range (RMS)");
        html.Append("</table>\n");
        string flag = (NetChangeFlag * 100).ToString("0", ci);
        html.Append(brightness
            ? $"<p class=\"flag\">The output's mean brightness differs from the input's by more than {flag}%. If you compare intensities " +
              "between images or conditions, denoise them all with this model and compare like with like, or measure on the raw data.</p>\n"
            : $"<p class=\"ok\">The mean brightness changed by less than {flag}%.</p>\n");

        html.Append("<h2>Input</h2>\n<table>\n");
        Row("File", $"<code>{E(Path.GetFullPath(d.Input))}</code>");
        Row("SHA-256", $"<code>{Sha256(d.Input)}</code>");
        Row("Format", E(d.Format) + (d.Storage is { } s ? $", {E(s.ToString())}" : ""));
        Row("Plane size", $"{d.Size.Width} × {d.Size.Height} pixels");
        Row("Planes and axes", E(d.Layout.ToString()) + $" ({E(SourceName(d.Layout.Source))}" +
                               (d.DeclaredAxes is { } a ? $"; declared --axes {E(a)}" : "") + ")");
        Row("Pixel size", d.PixelSize is { } size ? E(size) : "not recorded");
        html.Append("</table>\n<h2>Output</h2>\n<table>\n");
        Row("Denoised", $"<code>{E(Path.GetFullPath(d.Output))}</code>, float32 in the input's intensity units");
        Row("SHA-256", $"<code>{Sha256(d.Output)}</code>");
        if (d.Baseline is { } b) Row("Bilateral baseline", $"<code>{E(Path.GetFullPath(b))}</code> (for comparison; not DORA's output)");
        Row("Description carried over", E(d.DescriptionCarried) + (d.PixelSize is null ? "" : "; the pixel size on every page"));
        string vst = d.Vst ? ", variance-stabilized" : "", planesText = d.Planes == 1 ? "1 plane" : $"{d.Planes} planes";
        Row("This run", string.Create(ci, $"{planesText} in {d.Seconds:0.0} s, tile core {d.TileCore}px{vst}; peak memory {d.PeakMemory / (1024.0 * 1024.0):0} MB (this process)"));
        Row("Software tag", $"<code>{E(d.Software)}</code>");
        html.Append("</table>\n<h2>Model</h2>\n<table>\n");
        Row("Main model", E(d.Model.Info.Provenance));
        Row("Its content hash", $"<code>{E(WeightsFile.ContentHash(d.Model.Net))}</code>");
        Row("Architecture", E(d.Model.Info.Config.ToString()));
        if (d.CorrelatedModel is { } c)
        {
            Row("Correlated-noise model", E(c.Info.Provenance));
            Row("Its content hash", $"<code>{E(WeightsFile.ContentHash(c.Net))}</code>");
        }
        html.Append("</table>\n<h2>What this report can't tell you</h2>\n" +
                    "<p>Without a clean reference image nobody can measure how close the output is to the truth. The checks above catch " +
                    "the common failures (removed structure, changed brightness); they can't prove their absence. Keep the raw data.</p>\n" +
                    "</body></html>\n");
        return html.ToString();
    }

    private static string SourceName(StackMetadata source) => source switch
    {
        StackMetadata.ImageJ => "from the ImageJ header",
        StackMetadata.Tifffile => "from the tifffile header",
        StackMetadata.Mrc => "from the MRC header",
        StackMetadata.OmeXml => "OME-XML, not read by DORA",
        _ => "no axis metadata",
    };

    private static string Num(double v, string format) => v.ToString(format, CultureInfo.InvariantCulture);

    private static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}

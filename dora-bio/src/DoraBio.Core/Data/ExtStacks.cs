using System.Text.RegularExpressions;
using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>
/// One group of stack files in a dataset: the noisy files matching a pattern, each with an optional clean partner
/// whose path is the noisy path with one substring replaced.
/// </summary>
/// <param name="Noisy">Pattern relative to <c>data/ext/&lt;dataset&gt;/</c>, '/'-separated; <c>*</c> matches within one
/// folder or file name, e.g. <c>noisy/*.tif</c> or <c>*/Cell_*/Level_*/RawSIMData.mrc</c>.</param>
/// <param name="CleanFrom">Substring of a noisy path to replace to get its clean partner, or null (no ground truth).</param>
/// <param name="CleanTo">Its replacement.</param>
/// <param name="Axes">The planes' axes, ending in YX (e.g. "ZYX"), as the dataset's authors describe them.</param>
/// <param name="Files">Noisy files the published dataset has in this group.</param>
/// <param name="Planes">Noisy planes in the group, or null where they vary and weren't counted from the published metadata.</param>
public sealed record ExtStackGroup(string Noisy, string? CleanFrom, string? CleanTo, string Axes, int Files, long? Planes);

/// <summary>A benchmark dataset stored as stacks (3D volumes, time-lapses, multichannel or raw SIM files).</summary>
/// <param name="Name">Folder name under data/ext; the same name as in tools/ext_manifest.json.</param>
/// <param name="Description">What is imaged and how it is stored.</param>
/// <param name="Groups">Its groups of files.</param>
public sealed record ExtStackDataset(string Name, string Description, IReadOnlyList<ExtStackGroup> Groups)
{
    /// <summary>True if noisy files come with clean partners.</summary>
    public bool HasGroundTruth => Groups.Any(g => g.CleanFrom is not null);
}

/// <summary>A noisy stack file and its clean partner (null when the dataset has no ground truth).</summary>
public sealed record StackPair(string Noisy, string? Clean, ExtStackGroup Group);

/// <summary>What a dataset's files hold, read from their headers (no pixels are decoded).</summary>
/// <param name="Files">Noisy files found.</param>
/// <param name="Planes">Noisy planes in those files.</param>
/// <param name="PlaneSizes">Distinct plane sizes, e.g. "512×512".</param>
/// <param name="Formats">Distinct formats ("TIFF", "BigTIFF", "MRC").</param>
/// <param name="HeadersOpened">Noisy files whose headers were opened (all of them, or a spread-out sample; see
/// <see cref="ExtStacks.Inspect"/>).</param>
public sealed record StackSummary(int Files, long Planes, IReadOnlyList<string> PlaneSizes, IReadOnlyList<string> Formats, int HeadersOpened);

/// <summary>
/// The stack datasets of PATHWAY.md's road to state of the art (tools/ext_manifest.json), and how to find and check
/// their files. Counts come from the published metadata (Zenodo listings and ImageJ headers, read 2026-11-26) and are
/// checked against the files whenever a dataset is downloaded.
/// </summary>
public static class ExtStacks
{
    /// <summary>Every stack dataset, in manifest order.</summary>
    public static IReadOnlyList<ExtStackDataset> All { get; } =
    [
        new("mdc24-w2s", "W2S widefield: 120 fields of view × 3 channels, noisy only (int16 512²)",
            [new("noisy.tiff", null, null, "SCYX", 1, 360)]),
        new("mdc24-support", "SUPPORT Penicillium: one volume of 1001 z-slices, noisy only (uint16 1024²)",
            [new("noisy.tiff", null, null, "ZYX", 1, 1001)]),
        new("mdc24-jump", "JUMP Cell Painting: 517 fields of view × 4 channels, added Gaussian noise, noisy only (float32 540²)",
            [new("noisy.tiff", null, null, "SCYX", 1, 2068)]),
        new("mdc25-tribolium", "Tribolium: 15,500 paired 3D patches of 16 × 64² (float32)",
            [new("noisy/*.tif", "noisy/", "gt/", "ZYX", 15500, 15500L * 16)]),
        new("mdc25-planaria", "Planaria: 17,900 paired 3D patches of 16 × 64² and 45 paired volumes",
            [new("small_crops/noisy/*.tif", "/noisy/", "/gt/", "ZYX", 17900, 17900L * 16),
             new("large_crops/noisy/*.tif", "/noisy/", "/gt/", "ZYX", 45, null)]),
        new("biosr-ccps", "BioSR+ clathrin-coated pits: 48 cells × 8 signal levels, 9 raw SIM patterns each (uint16 MRC)",
            [new("*/Cell_*/Level_*/RawSIMData.mrc", "RawSIMData.mrc", "RawSIMData_gt.mrc", "PYX", 384, 384L * 9)]),
        new("biosr-er", "BioSR+ endoplasmic reticulum: 72 cells, 6–8 signal levels, 9 raw SIM patterns each (uint16 MRC)",
            [new("*/Cell_*/Level_*/RawSIMData.mrc", "RawSIMData.mrc", "RawSIMData_gt.mrc", "PYX", 462, 462L * 9)]),
        new("biosr-microtubules", "BioSR+ microtubules: 43 cells × 8 signal levels, 9 raw SIM patterns each (uint16 MRC)",
            [new("*/Cell_*/Level_*/RawSIMData.mrc", "RawSIMData.mrc", "RawSIMData_gt.mrc", "PYX", 344, 344L * 9)]),
        new("biosr-f-actin", "BioSR+ F-actin: 26 cells, 11–15 signal levels, 9 raw SIM patterns each (uint16 MRC)",
            [new("*/Cell_*/Level_*/RawSIMData.mrc", "RawSIMData.mrc", "RawSIMData_gt.mrc", "PYX", 330, 330L * 9)]),
        new("biosr-myosin-iia", "BioSR+ Myosin-IIA: 41 cells, 8–10 signal levels, 9 raw SIM patterns each (uint16 MRC)",
            [new("*/Cell_*/Level_*/RawSIMData.mrc", "RawSIMData.mrc", "RawSIMData_gt.mrc", "PYX", 360, 360L * 9)]),
        new("deepcad-zebrafish-regions", "DeepCAD-RT zebrafish brain regions: 9,800 two-photon frames, low/high SNR (BigTIFF)",
            [new("*_lowSNR.tif", "_lowSNR", "_highSNR", "TYX", 1, 9800)]),
        new("deepcad-neutrophils", "DeepCAD-RT mouse neutrophils (motile): 6,440 + 5,706 frames, low/high SNR",
            [new("*_lowSNR.tif", "_lowSNR", "_highSNR", "TYX", 2, 6440 + 5706)]),
        new("deepcad-zebrafish-ot", "DeepCAD-RT zebrafish optic tectum: 6,955 two-photon frames, low/high SNR (BigTIFF)",
            [new("*_lowSNR.tif", "_lowSNR", "_highSNR", "TYX", 1, 6955)]),
    ];

    /// <summary>The stack dataset called <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">No such dataset (the message lists the known names).</exception>
    public static ExtStackDataset Get(string name) =>
        All.FirstOrDefault(d => d.Name == name)
        ?? throw new ArgumentException($"Unknown stack dataset '{name}'. Known: {string.Join(", ", All.Select(d => d.Name))}.");

    /// <summary>True once <c>tools/fetch_ext.sh</c> has verified and unpacked the dataset under <paramref name="root"/>.</summary>
    public static bool IsPresent(string root, ExtStackDataset dataset) => File.Exists(Path.Combine(root, dataset.Name, ".complete"));

    /// <summary>
    /// Every noisy file of every group, in ordinal path order, with its clean partner.
    /// </summary>
    /// <exception cref="InvalidDataException">A noisy file has no clean partner where the group has ground truth.</exception>
    public static List<StackPair> Plan(string root, ExtStackDataset dataset)
    {
        string dir = Path.Combine(root, dataset.Name);
        var files = Directory.Exists(dir)
            ? Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories)
                .Select(f => Path.GetRelativePath(dir, f).Replace('\\', '/'))
                .Order(StringComparer.Ordinal).ToList()
            : [];
        var pairs = new List<StackPair>();
        foreach (var group in dataset.Groups)
        {
            var pattern = Glob(group.Noisy);
            foreach (var rel in files.Where(f => pattern.IsMatch(f)))
            {
                string? clean = null;
                if (group.CleanFrom is not null)
                {
                    int at = rel.IndexOf(group.CleanFrom, StringComparison.Ordinal);
                    string partner = at < 0 ? rel : rel[..at] + group.CleanTo + rel[(at + group.CleanFrom.Length)..];
                    clean = Path.Combine(dir, partner);
                    if (at < 0 || !File.Exists(clean))
                        throw new InvalidDataException($"{dataset.Name}: {rel} has no clean partner ({partner}).");
                }
                pairs.Add(new StackPair(Path.Combine(dir, rel), clean, group));
            }
        }
        return pairs;
    }

    /// <summary>
    /// Checks every pair's files and returns what was found per group (no pixels are read): each noisy file must have the
    /// group's declared axes, and its clean partner the same layout and plane size. Every pair is checked to exist. A
    /// group of more than <paramref name="sampleLimit"/> files has the headers of an evenly spread sample opened (always
    /// including the first and last), and its plane count is files × planes per file, which holds only if every file in
    /// the group has the same byte size and every sampled file the same layout; otherwise every header is opened. (On a
    /// cold disk, opening the 31,000 headers of MDC25 Tribolium one by one took 20 minutes.)
    /// </summary>
    /// <exception cref="InvalidDataException">A file disagrees with its declared axes or with its partner.</exception>
    public static List<StackSummary> Inspect(string root, ExtStackDataset dataset, int sampleLimit = 200)
    {
        if (sampleLimit < 2) throw new ArgumentOutOfRangeException(nameof(sampleLimit), "A sample needs at least the first and last file.");
        var pairs = Plan(root, dataset);
        var result = new List<StackSummary>();
        foreach (var group in dataset.Groups)
        {
            var members = pairs.Where(p => p.Group == group).ToList();
            bool uniform = members.Count > sampleLimit
                           && members.Select(p => new FileInfo(p.Noisy).Length).Distinct().Count() == 1
                           && members.Where(p => p.Clean is not null).Select(p => new FileInfo(p.Clean!).Length).Distinct().Count() <= 1;
            var opened = uniform
                ? Enumerable.Range(0, sampleLimit).Select(k => (int)((long)k * (members.Count - 1) / (sampleLimit - 1))).Distinct().Select(i => members[i]).ToList()
                : members;
            long planes = 0;
            var layouts = new HashSet<string>(StringComparer.Ordinal);
            var sizes = new SortedSet<string>(StringComparer.Ordinal);
            var formats = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var pair in opened)
            {
                using var noisy = Open(pair.Noisy, group.Axes);
                planes += noisy.PlaneCount;
                layouts.Add(noisy.Layout.ToString());
                sizes.Add($"{noisy.Width}×{noisy.Height}");
                formats.Add(noisy.Format);
                if (pair.Clean is null) continue;
                using var clean = Open(pair.Clean, group.Axes);
                if (clean.Layout.ToString() != noisy.Layout.ToString() || clean.Width != noisy.Width || clean.Height != noisy.Height)
                    throw new InvalidDataException(
                        $"{Path.GetFileName(pair.Clean)}: {clean.Layout}, {clean.Width}×{clean.Height}, but its noisy partner has " +
                        $"{noisy.Layout}, {noisy.Width}×{noisy.Height}.");
            }
            if (uniform && layouts.Count == 1 && sizes.Count == 1)
                planes = planes / opened.Count * members.Count;   // same bytes and same layout in the sample: one shape
            else if (uniform)
                return Inspect(root, dataset, sampleLimit: int.MaxValue);   // the sample disagrees: open everything
            result.Add(new StackSummary(members.Count, planes, sizes.ToList(), formats.ToList(), opened.Count));
        }
        return result;
    }

    private static ImageStack Open(string path, string axes)
    {
        try
        {
            return ImageStack.Open(path, axes);
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"{Path.GetFileName(path)}: {e.Message}", e);
        }
    }

    /// <summary>A '/'-separated glob as a whole-path regex: <c>*</c> matches any run of characters except '/'.</summary>
    internal static Regex Glob(string pattern) =>
        new("^" + string.Join("[^/]*", pattern.Split('*').Select(Regex.Escape)) + "$", RegexOptions.CultureInvariant);
}

using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>
/// Writes the derived 2D sets of <see cref="ExtDatasets.Derived"/> from their stack datasets (brick 38): every plane of
/// the selected noisy files and of their clean partners becomes a float32 TIFF, raw values unchanged, in the downloaded
/// sets' train/test × low_SNR/high_SNR layout. The selections are the scoreboard's pre-registered splits. Derived data is
/// as much the source's as the original (same licence, never committed).
/// </summary>
public static class ExtDerived
{
    /// <summary>True once <see cref="Export"/> has finished the set under <paramref name="root"/>.</summary>
    public static bool IsPresent(string root, ExtDataset derived) => File.Exists(Path.Combine(root, derived.Name, ".complete"));

    /// <summary>Exports a derived set (skipped if already complete). Returns (training planes, test planes).</summary>
    /// <exception cref="ArgumentException">Not a derived set, or its source isn't downloaded.</exception>
    public static (int Train, int Test) Export(string root, ExtDataset derived)
    {
        if (derived.Source is null) throw new ArgumentException($"{derived.Name} is not a derived set.", nameof(derived));
        var source = ExtStacks.Get(derived.Source);
        if (!ExtStacks.IsPresent(root, source)) throw new ArgumentException($"{derived.Name} needs {source.Name} (tools/fetch_ext.sh {source.Name}).");
        string dir = Path.Combine(root, derived.Name);
        if (IsPresent(root, derived)) return (derived.TrainCount, derived.TestCount);
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        var pairs = ExtStacks.Plan(root, source);
        var (train, test) = Select(derived.Name, pairs);
        int nTrain = Write(train, Path.Combine(dir, "train"), derived.Name);
        int nTest = Write(test, Path.Combine(dir, "test"), derived.Name);
        if (nTrain != derived.TrainCount || nTest != derived.TestCount)
            throw new InvalidDataException($"{derived.Name}: wrote {nTrain}/{nTest} planes, expected {derived.TrainCount}/{derived.TestCount}.");
        File.WriteAllText(Path.Combine(dir, ".complete"), $"{nTrain} {nTest}\n");
        return (nTrain, nTest);
    }

    /// <summary>The pre-registered splits (PATHWAY.md, scoreboard): by patch index for Tribolium, by cell for BioSR+.</summary>
    internal static (List<StackPair> Train, List<StackPair> Test) Select(string name, IReadOnlyList<StackPair> pairs) => name switch
    {
        "mdc25-tribolium-2d" => (pairs.Take(2000).ToList(), pairs.Skip(15250).Take(250).ToList()),
        "biosr-ccps-2d" => (pairs.Where(p => CellOf(p) <= 40).ToList(), pairs.Where(p => CellOf(p) is >= 41 and <= 48).ToList()),
        _ => throw new ArgumentException($"No export rule for {name}."),
    };

    /// <summary>The cell number of a BioSR+ file (…/Cell_017/Level_03/RawSIMData.mrc → 17).</summary>
    internal static int CellOf(StackPair pair)
    {
        string cell = pair.Noisy.Replace('\\', '/').Split('/').First(s => s.StartsWith("Cell_", StringComparison.Ordinal));
        return int.Parse(cell[5..], System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>The derived file stem of a source file: <c>group__</c> plus whatever names the plane.</summary>
    internal static string StemOf(string name, StackPair pair)
    {
        var parts = pair.Noisy.Replace('\\', '/').Split('/');
        return name == "biosr-ccps-2d"
            ? $"c{CellOf(pair):000}__{parts.First(s => s.StartsWith("Level_", StringComparison.Ordinal)).Replace("Level_", "L")}"
            : Path.GetFileNameWithoutExtension(parts[^1]) + "__";
    }

    /// <summary>Writes every plane of <paramref name="pairs"/> (noisy to low_SNR, clean to high_SNR under <paramref name="dir"/>).</summary>
    internal static int Write(IReadOnlyList<StackPair> pairs, string dir, string name)
    {
        string low = Path.Combine(dir, "low_SNR"), high = Path.Combine(dir, "high_SNR");
        Directory.CreateDirectory(low);
        Directory.CreateDirectory(high);
        int written = 0;
        foreach (var pair in pairs)
        {
            using var noisy = ImageStack.Open(pair.Noisy, pair.Group.Axes);
            using var clean = ImageStack.Open(pair.Clean!, pair.Group.Axes);
            string stem = StemOf(name, pair);
            string axis = pair.Group.Axes[..^2];   // the one non-image axis: Z for Tribolium, P for BioSR+
            for (int k = 0; k < noisy.PlaneCount; k++)
            {
                string file = stem.EndsWith("__", StringComparison.Ordinal) ? $"{stem}{axis.ToLowerInvariant()}{k:00}.tif" : $"{stem}_{axis.ToLowerInvariant()}{k}.tif";
                File.WriteAllBytes(Path.Combine(low, file), TiffCodec.Encode(noisy.ReadPlane(k), TiffSampleType.Float32));
                File.WriteAllBytes(Path.Combine(high, file), TiffCodec.Encode(clean.ReadPlane(k), TiffSampleType.Float32));
                written++;
            }
        }
        return written;
    }
}

using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>
/// Loads a training directory into normalized <see cref="ImagePair"/>s. The layout is detected automatically:
///
/// 1. Paired:   dir/noisy/NAME.tif with optional dir/clean/NAME.tif (matched by file name).
///              This is what <c>dora-bio synth</c> writes.
/// 2. FMD:      a dataset root of category folders, or a single category (dir/raw/FOV/*.png + dir/gt/FOV/avg50.png),
///              or the flat test_mix layout. See <see cref="FmdDataset"/>, which also keeps FMD's test FOV out of training.
/// 3. Flat:     dir/*.tif|png, noisy only, for Noise2Void. Each file is its own FOV.
/// </summary>
/// <remarks>
/// The generic loader is gated by brick 5 (training). The FMD layouts are brick 12 (fmd-loader).
/// </remarks>
public static class DatasetLoader
{
    /// <summary>Detects the layout, loads every image, checks sizes, and normalizes each pair by its noisy image's percentiles.</summary>
    /// <param name="dir">Dataset folder.</param>
    /// <param name="fmd">What to load from an FMD folder (frames per FOV, noise levels, categories). Defaults exclude the test FOV.</param>
    /// <exception cref="DirectoryNotFoundException"><paramref name="dir"/> doesn't exist.</exception>
    /// <exception cref="InvalidDataException">No supported images found, or a clean image doesn't match its noisy image's size.</exception>
    public static List<ImagePair> LoadDirectory(string dir, FmdOptions? fmd = null)
    {
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException(dir);

        if (FmdDataset.IsCategory(dir) || FmdDataset.IsRoot(dir))
            return FmdDataset.Load(FmdDataset.PlanTraining(dir, fmd ?? new FmdOptions()));
        if (FmdDataset.IsTestMix(dir))
            return FmdDataset.Load(FmdDataset.PlanTestMix(dir));

        string noisyDir = Path.Combine(dir, "noisy"), cleanDir = Path.Combine(dir, "clean");
        List<ImagePair> raw = Directory.Exists(noisyDir)
            ? LoadPaired(noisyDir, cleanDir)
            : ImageFiles(dir).Select(f => new ImagePair(ImageLoader.Load(f), null, Path.GetFileNameWithoutExtension(f), Path.GetFileName(f))).ToList();

        if (raw.Count == 0)
            throw new InvalidDataException($"No .tif/.tiff/.png images found in {dir} (looked for noisy/, FMD raw/+gt/, or flat files).");
        foreach (var pair in raw) pair.Validate();
        return raw.Select(p => p.Normalize().Pair).ToList();
    }

    /// <summary>noisy/NAME.* paired with clean/NAME.* by file stem; a noisy image without a match gets no clean image.</summary>
    private static List<ImagePair> LoadPaired(string noisyDir, string cleanDir)
    {
        var result = new List<ImagePair>();
        foreach (string file in ImageFiles(noisyDir))
        {
            string stem = Path.GetFileNameWithoutExtension(file);
            string? cleanFile = Directory.Exists(cleanDir)
                ? ImageFiles(cleanDir).FirstOrDefault(c => Path.GetFileNameWithoutExtension(c) == stem)
                : null;
            result.Add(new ImagePair(ImageLoader.Load(file), cleanFile is null ? null : ImageLoader.Load(cleanFile), stem, Path.GetFileName(file)));
        }
        return result;
    }

    /// <summary>Supported image files in <paramref name="dir"/>, sorted ordinally so load order (and seeded splits) is stable.</summary>
    private static IEnumerable<string> ImageFiles(string dir) =>
        Directory.GetFiles(dir).Where(ImageLoader.IsSupported).OrderBy(f => f, StringComparer.Ordinal);
}

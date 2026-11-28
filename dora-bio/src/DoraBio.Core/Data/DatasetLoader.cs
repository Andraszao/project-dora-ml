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
///
/// A multi-page TIFF (or MRC) contributes every plane, each an image of its file's FOV, so a split keeps a stack's planes
/// together (brick 41: until then only page 0 was read, silently). Paired stacks pair page k with page k.
/// </summary>
/// <remarks>
/// The generic loader is gated by brick 5 (training). The FMD layouts are brick 12 (fmd-loader).
/// </remarks>
public static class DatasetLoader
{
    /// <summary>Detects the layout, loads every image, checks sizes, and normalizes each pair by its noisy image's percentiles.</summary>
    /// <param name="dir">Dataset folder.</param>
    /// <param name="fmd">What to load from an FMD folder (frames per FOV, noise levels, categories). Defaults exclude the test FOV.</param>
    /// <param name="beforeNormalize">Optional per-pair step in original units, before normalization (e.g. variance stabilization).</param>
    /// <exception cref="DirectoryNotFoundException"><paramref name="dir"/> doesn't exist.</exception>
    /// <exception cref="InvalidDataException">No supported images found, or a clean image doesn't match its noisy image's size.</exception>
    public static List<ImagePair> LoadDirectory(string dir, FmdOptions? fmd = null, Func<ImagePair, ImagePair>? beforeNormalize = null)
    {
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException(dir);

        if (FmdDataset.IsCategory(dir) || FmdDataset.IsRoot(dir))
            return FmdDataset.Load(FmdDataset.PlanTraining(dir, fmd ?? new FmdOptions()), beforeNormalize);
        if (FmdDataset.IsTestMix(dir))
            return FmdDataset.Load(FmdDataset.PlanTestMix(dir), beforeNormalize);

        string noisyDir = Path.Combine(dir, "noisy"), cleanDir = Path.Combine(dir, "clean");
        List<ImagePair> raw = Directory.Exists(noisyDir)
            ? LoadPaired(noisyDir, cleanDir)
            : ImageFiles(dir).SelectMany(f =>
            {
                var planes = Planes(f);
                return planes.Select((plane, k) => new ImagePair(plane, null, Path.GetFileNameWithoutExtension(f), PlaneName(f, k, planes.Count)));
            }).ToList();

        if (raw.Count == 0)
            throw new InvalidDataException($"No .tif/.tiff/.png images found in {dir} (looked for noisy/, FMD raw/+gt/, or flat files).");
        foreach (var pair in raw) pair.Validate();
        return raw.Select(p => (beforeNormalize?.Invoke(p) ?? p).Normalize().Pair).ToList();
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
            var noisy = Planes(file);
            var clean = cleanFile is null ? null : Planes(cleanFile);
            if (clean is not null && clean.Count != noisy.Count)
                throw new InvalidDataException($"{Path.GetFileName(file)} has {noisy.Count} planes but its clean partner has {clean.Count}.");
            for (int k = 0; k < noisy.Count; k++)
                result.Add(new ImagePair(noisy[k], clean?[k], stem, PlaneName(file, k, noisy.Count)));
        }
        return result;
    }

    /// <summary>
    /// Every plane of an image file: a TIFF or MRC stack plane by plane (its page count checked against its metadata, so an
    /// ImageJ stack over 4 GB that stores one page directory is refused, not read as one plane), a PNG as one plane.
    /// A single-page file is decoded exactly as before.
    /// </summary>
    public static List<GrayImage> Planes(string file)
    {
        var head = new byte[1024];
        int read;
        using (var probe = File.OpenRead(file)) read = probe.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        bool stackFormat = TiffCodec.HasTiffSignature(head.AsSpan(0, read)) || MrcReader.LooksLikeMrc(head.AsSpan(0, read));
        if (!stackFormat) return [ImageLoader.Load(file)];
        using var stack = ImageStack.Open(file);
        if (stack.PlaneCount == 1 && stack.Format != "MRC") return [ImageLoader.Load(file)];
        return Enumerable.Range(0, stack.PlaneCount).Select(stack.ReadPlane).ToList();
    }

    /// <summary>The file name, plus "#p" and the plane number for a stack's planes.</summary>
    private static string PlaneName(string file, int plane, int planes) =>
        planes == 1 ? Path.GetFileName(file) : $"{Path.GetFileName(file)}#p{plane}";

    /// <summary>Supported image files in <paramref name="dir"/>, sorted ordinally so load order (and seeded splits) is stable.</summary>
    private static IEnumerable<string> ImageFiles(string dir) =>
        Directory.GetFiles(dir).Where(ImageLoader.IsSupported).OrderBy(f => f, StringComparer.Ordinal);
}

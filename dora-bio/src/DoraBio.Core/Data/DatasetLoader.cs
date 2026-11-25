using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>
/// Loads a training directory into normalized <see cref="ImagePair"/>s. The layout is detected automatically:
///
/// 1. Paired:  dir/noisy/NAME.tif with optional dir/clean/NAME.tif (matched by file name).
///             This is what <c>dora-bio synth</c> writes.
/// 2. FMD:     dir/raw/FOV/*.png with dir/gt/FOV/avg50.png, the published layout of the Fluorescence Microscopy
///             Denoising dataset (Zhang et al., CVPR 2019). Up to <c>maxFramesPerFov</c> raw frames per FOV.
///             Matches the dataset's documented layout but has not been tested against the real download.
/// 3. Flat:    dir/*.tif|png, noisy only, for Noise2Void. Each file is its own FOV.
/// </summary>
public static class DatasetLoader
{
    public static List<ImagePair> LoadDirectory(string dir, int maxFramesPerFov = 4)
    {
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException(dir);

        string noisyDir = Path.Combine(dir, "noisy"), cleanDir = Path.Combine(dir, "clean");
        string rawDir = Path.Combine(dir, "raw"), gtDir = Path.Combine(dir, "gt");

        List<ImagePair> raw;
        if (Directory.Exists(noisyDir)) raw = LoadPaired(noisyDir, cleanDir);
        else if (Directory.Exists(rawDir) && Directory.Exists(gtDir)) raw = LoadFmd(rawDir, gtDir, maxFramesPerFov);
        else raw = ImageFiles(dir).Select(f => new ImagePair(ImageLoader.Load(f), null, Path.GetFileNameWithoutExtension(f), Path.GetFileName(f))).ToList();

        if (raw.Count == 0)
            throw new InvalidDataException($"No .tif/.tiff/.png images found in {dir} (looked for noisy/, raw/+gt/, or flat files).");
        foreach (var pair in raw) pair.Validate();
        return raw.Select(p => p.Normalize().Pair).ToList();
    }

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

    private static List<ImagePair> LoadFmd(string rawDir, string gtDir, int maxFramesPerFov)
    {
        var result = new List<ImagePair>();
        foreach (string fovDir in Directory.GetDirectories(rawDir).OrderBy(d => d, StringComparer.Ordinal))
        {
            string fov = Path.GetFileName(fovDir);
            string? gtFile = Directory.Exists(Path.Combine(gtDir, fov))
                ? ImageFiles(Path.Combine(gtDir, fov)).FirstOrDefault()
                : null;
            var clean = gtFile is null ? null : ImageLoader.Load(gtFile);
            foreach (string frame in ImageFiles(fovDir).Take(maxFramesPerFov))
                result.Add(new ImagePair(ImageLoader.Load(frame), clean, $"fov-{fov}", $"{fov}/{Path.GetFileName(frame)}"));
        }
        return result;
    }

    private static IEnumerable<string> ImageFiles(string dir) =>
        Directory.GetFiles(dir).Where(ImageLoader.IsSupported).OrderBy(f => f, StringComparer.Ordinal);
}

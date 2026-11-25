using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>Which part of the FMD dataset to load for training.</summary>
public sealed record FmdOptions
{
    /// <summary>Category folders to use (e.g. "Confocal_FISH"); null means every category under the root.</summary>
    public IReadOnlyList<string>? Categories { get; init; }

    /// <summary>
    /// Noisy frames taken per FOV and noise level (the first N in file-name order). FMD has 50 per FOV and level.
    /// They are repeated captures of the same scene, so more frames add noise variety but little new structure.
    /// </summary>
    public int FramesPerFov { get; init; } = 4;

    /// <summary>
    /// Noise levels to train on: "raw" (single captures) and/or "avg2", "avg4", "avg8", "avg16" (averages of 2–16
    /// captures, so progressively less noise). Mixing levels is the real-data form of v1's "progressive noise" idea.
    /// </summary>
    public IReadOnlyList<string> NoiseLevels { get; init; } = ["raw"];

    /// <summary>
    /// FOVs never used for training. By default this is FMD's test FOV (<see cref="FmdDataset.TestFov"/>), which
    /// test_mix is drawn from; training on it would leak the test set into the model.
    /// </summary>
    public IReadOnlyCollection<int> ExcludedFovs { get; init; } = [FmdDataset.TestFov];
}

/// <summary>
/// The Fluorescence Microscopy Denoising dataset (Zhang et al., CVPR 2019; CC BY-SA 4.0), laid out as
/// <c>tools/fetch_fmd.sh</c> extracts it. Layouts verified against the real download:
/// <code>
/// &lt;root&gt;/&lt;Category&gt;/raw/&lt;fov&gt;/*.png        50 single noisy captures per FOV (FOVs 1–20)
/// &lt;root&gt;/&lt;Category&gt;/avgN/&lt;fov&gt;/*.png       50 averages of N captures (N = 2, 4, 8, 16): less noise
/// &lt;root&gt;/&lt;Category&gt;/gt/&lt;fov&gt;/avg50.png      ground truth: the average of all 50 captures
/// &lt;root&gt;/test_mix/{raw,avgN,gt}/&lt;Category&gt;_&lt;k&gt;.png   benchmark set: 4 images per category, all from FOV 19
/// </code>
/// Loading happens in two steps. <see cref="PlanTraining"/> and <see cref="PlanTestMix"/> only list files (fast, no
/// decoding), so checks such as "no test FOV in the training plan" are cheap. <see cref="Load"/> then decodes and
/// normalizes the pairs.
/// </summary>
public static class FmdDataset
{
    /// <summary>The FOV that test_mix is drawn from (verified pixel-for-pixel by <see cref="Verify"/>).</summary>
    public const int TestFov = 19;

    /// <summary>Noise-level folder names, noisiest first.</summary>
    public static IReadOnlyList<string> Levels { get; } = ["raw", "avg2", "avg4", "avg8", "avg16"];

    /// <summary>One noisy/clean file pair to load.</summary>
    /// <param name="NoisyPath">Noisy image file.</param>
    /// <param name="CleanPath">Ground-truth image file.</param>
    /// <param name="GroupId">FOV identity, e.g. "Confocal_FISH/7": the unit of the train/validation split.</param>
    /// <param name="Name">Display name, e.g. "Confocal_FISH/raw/7/HV140_P100510000.png".</param>
    public sealed record PlannedPair(string NoisyPath, string CleanPath, string GroupId, string Name);

    /// <summary>True for a category folder (raw/ and gt/ holding numbered FOV folders).</summary>
    public static bool IsCategory(string dir) =>
        Directory.Exists(Path.Combine(dir, "raw")) && Directory.Exists(Path.Combine(dir, "gt")) &&
        Directory.GetDirectories(Path.Combine(dir, "raw")).Length > 0;

    /// <summary>True for the test_mix layout (raw/ and gt/ holding image files directly, matched by name).</summary>
    public static bool IsTestMix(string dir) =>
        Directory.Exists(Path.Combine(dir, "raw")) && Directory.Exists(Path.Combine(dir, "gt")) &&
        Directory.GetFiles(Path.Combine(dir, "raw")).Any(ImageLoader.IsSupported);

    /// <summary>True for a dataset root: a folder containing at least one category folder.</summary>
    public static bool IsRoot(string dir) => Directory.Exists(dir) && Directory.GetDirectories(dir).Any(IsCategory);

    /// <summary>
    /// Lists the training pairs for a dataset root or a single category, in a deterministic order (categories and
    /// files ordinal, FOVs numeric). Excluded FOVs never appear. Every noisy frame of a FOV pairs with that FOV's
    /// ground truth.
    /// </summary>
    /// <exception cref="InvalidDataException">Missing ground truth, a missing noise level, or no categories.</exception>
    public static List<PlannedPair> PlanTraining(string rootOrCategory, FmdOptions options)
    {
        if (options.FramesPerFov < 1) throw new ArgumentOutOfRangeException(nameof(options), "FramesPerFov must be at least 1.");
        if (options.NoiseLevels.Count == 0 || options.NoiseLevels.Any(l => !Levels.Contains(l)))
            throw new ArgumentException($"Noise levels must be some of: {string.Join(", ", Levels)}.", nameof(options));

        var categories = IsCategory(rootOrCategory)
            ? [rootOrCategory]
            : Directory.GetDirectories(rootOrCategory).Where(IsCategory).OrderBy(d => d, StringComparer.Ordinal).ToList();
        if (options.Categories is { } wanted)
        {
            var unknown = wanted.Except(categories.Select(Path.GetFileName)!, StringComparer.Ordinal).ToList();
            if (unknown.Count > 0) throw new InvalidDataException($"Unknown FMD categories: {string.Join(", ", unknown)}.");
            categories = categories.Where(c => wanted.Contains(Path.GetFileName(c), StringComparer.Ordinal)).ToList();
        }
        if (categories.Count == 0) throw new InvalidDataException($"No FMD category folders (raw/ + gt/) under {rootOrCategory}.");

        var plan = new List<PlannedPair>();
        foreach (string category in categories)
        {
            string name = Path.GetFileName(category);
            foreach (int fov in Fovs(category).Where(f => !options.ExcludedFovs.Contains(f)))
            {
                string gtDir = Path.Combine(category, "gt", fov.ToString());
                string clean = ImageFiles(gtDir).FirstOrDefault()
                               ?? throw new InvalidDataException($"No ground truth image in {gtDir}.");
                foreach (string level in options.NoiseLevels)
                {
                    string levelDir = Path.Combine(category, level, fov.ToString());
                    if (!Directory.Exists(levelDir)) throw new InvalidDataException($"Missing noise level folder {levelDir}.");
                    foreach (string frame in ImageFiles(levelDir).Take(options.FramesPerFov))
                        plan.Add(new PlannedPair(frame, clean, $"{name}/{fov}", $"{name}/{level}/{fov}/{Path.GetFileName(frame)}"));
                }
            }
        }
        return plan;
    }

    /// <summary>
    /// Lists the test_mix benchmark pairs at one noise level: level/&lt;Category&gt;_&lt;k&gt;.png with
    /// gt/&lt;Category&gt;_&lt;k&gt;.png. The group is the category, since every test image is FOV 19.
    /// </summary>
    public static List<PlannedPair> PlanTestMix(string testMixDir, string level = "raw")
    {
        if (!Levels.Contains(level)) throw new ArgumentException($"Unknown noise level '{level}'.", nameof(level));
        string levelDir = Path.Combine(testMixDir, level), gtDir = Path.Combine(testMixDir, "gt");
        if (!Directory.Exists(levelDir) || !Directory.Exists(gtDir))
            throw new InvalidDataException($"{testMixDir} is not a test_mix folder (needs {level}/ and gt/).");

        var plan = new List<PlannedPair>();
        foreach (string noisy in ImageFiles(levelDir))
        {
            string file = Path.GetFileName(noisy);
            string clean = Path.Combine(gtDir, file);
            if (!File.Exists(clean)) throw new InvalidDataException($"test_mix image {file} has no ground truth in {gtDir}.");
            plan.Add(new PlannedPair(noisy, clean, CategoryOf(file), $"test_mix/{level}/{file}"));
        }
        if (plan.Count == 0) throw new InvalidDataException($"No images in {levelDir}.");
        return plan;
    }

    /// <summary>Decodes and normalizes planned pairs in parallel (output order = plan order).</summary>
    /// <remarks>
    /// Each ground-truth file is decoded once and shared by all its frames before normalization. Every pair is
    /// normalized by its own noisy image's percentiles, as at inference time.
    /// </remarks>
    /// <param name="plan">Pairs from <see cref="PlanTraining"/> or <see cref="PlanTestMix"/>.</param>
    /// <param name="beforeNormalize">Optional per-pair step in original units, before normalization.</param>
    public static List<ImagePair> Load(IReadOnlyList<PlannedPair> plan, Func<ImagePair, ImagePair>? beforeNormalize = null)
    {
        var cleanCache = plan.Select(p => p.CleanPath).Distinct()
            .AsParallel().Select(path => (path, image: ImageLoader.Load(path)))
            .ToDictionary(x => x.path, x => x.image);
        var result = new ImagePair[plan.Count];
        Parallel.For(0, plan.Count, i =>
        {
            var p = plan[i];
            var pair = new ImagePair(ImageLoader.Load(p.NoisyPath), cleanCache[p.CleanPath], p.GroupId, p.Name);
            pair.Validate();
            result[i] = (beforeNormalize?.Invoke(pair) ?? pair).Normalize().Pair;
        });
        return result.ToList();
    }

    /// <summary>"Confocal_BPAE_B_3.png" → "Confocal_BPAE_B".</summary>
    public static string CategoryOf(string testMixFile)
    {
        string stem = Path.GetFileNameWithoutExtension(testMixFile);
        int underscore = stem.LastIndexOf('_');
        return underscore > 0 && int.TryParse(stem[(underscore + 1)..], out _) ? stem[..underscore] : stem;
    }

    /// <summary>
    /// Checks a dataset root: category structure, test_mix structure, and the leakage guard. Every test_mix ground
    /// truth must be pixel-identical to its category's FOV <see cref="TestFov"/> ground truth, and the default training
    /// plan must contain no test-FOV file. Returns one line per problem (empty = OK) and a summary.
    /// </summary>
    public static (List<string> Problems, string Summary) Verify(string root)
    {
        var problems = new List<string>();
        var categories = Directory.GetDirectories(root).Where(IsCategory).OrderBy(d => d, StringComparer.Ordinal).ToList();
        if (categories.Count == 0) return (["no category folders"], "");

        int fovTotal = 0, frameTotal = 0;
        foreach (string category in categories)
        {
            string name = Path.GetFileName(category);
            var fovs = Fovs(category).ToList();
            fovTotal += fovs.Count;
            if (!fovs.Contains(TestFov)) problems.Add($"{name}: no FOV {TestFov} (the test FOV)");
            foreach (int fov in fovs)
            {
                int gtCount = ImageFiles(Path.Combine(category, "gt", fov.ToString())).Count();
                if (gtCount != 1) problems.Add($"{name}/gt/{fov}: {gtCount} images (expected 1)");
                foreach (string level in Levels)
                {
                    string dir = Path.Combine(category, level, fov.ToString());
                    int n = Directory.Exists(dir) ? ImageFiles(dir).Count() : 0;
                    if (n == 0) problems.Add($"{name}/{level}/{fov}: missing or empty");
                    frameTotal += n;
                }
            }
        }

        string testMix = Path.Combine(root, "test_mix");
        int testCount = 0;
        if (!IsTestMix(testMix)) problems.Add("test_mix: missing or not in the raw/ + gt/ layout");
        else
        {
            foreach (string level in Levels)
                if (PlanTestMix(testMix, level).Count != PlanTestMix(testMix, "raw").Count)
                    problems.Add($"test_mix/{level}: image count differs from raw/");
            var tests = PlanTestMix(testMix, "raw");
            testCount = tests.Count;
            foreach (var t in tests)
            {
                string categoryGt = Path.Combine(root, t.GroupId, "gt", TestFov.ToString());
                string? gtFile = Directory.Exists(categoryGt) ? ImageFiles(categoryGt).FirstOrDefault() : null;
                if (gtFile is null) { problems.Add($"{t.Name}: no {t.GroupId}/gt/{TestFov} to compare with"); continue; }
                if (!ImageLoader.Load(t.CleanPath).Pixels.AsSpan().SequenceEqual(ImageLoader.Load(gtFile).Pixels))
                    problems.Add($"{t.Name}: ground truth is not FOV {TestFov}'s; excluding FOV {TestFov} would not prevent leakage");
            }
        }

        var leaked = PlanTraining(root, new FmdOptions()).Where(p => p.GroupId.EndsWith($"/{TestFov}", StringComparison.Ordinal)).ToList();
        if (leaked.Count > 0) problems.Add($"default training plan contains {leaked.Count} test-FOV files");

        string summary = $"{categories.Count} categories, {fovTotal} FOVs, {frameTotal} noisy frames over {Levels.Count} noise levels; " +
                         $"test_mix: {testCount} images, all matching FOV {TestFov}; default training plan excludes FOV {TestFov}";
        return (problems, summary);
    }

    private static IEnumerable<int> Fovs(string category) =>
        Directory.GetDirectories(Path.Combine(category, "raw"))
            .Select(d => int.TryParse(Path.GetFileName(d), out int f) ? f : -1)
            .Where(f => f >= 0)
            .OrderBy(f => f);

    private static IEnumerable<string> ImageFiles(string dir) =>
        Directory.Exists(dir)
            ? Directory.GetFiles(dir).Where(ImageLoader.IsSupported).OrderBy(f => f, StringComparer.Ordinal)
            : [];
}

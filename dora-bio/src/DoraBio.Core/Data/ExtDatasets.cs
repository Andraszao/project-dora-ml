using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>Which half of an external dataset: the authors' training images or their held-out test images.</summary>
public enum ExtSplit
{
    /// <summary>The images the dataset's authors trained on.</summary>
    Train,

    /// <summary>The images the authors held out for testing. DORA never trains on them.</summary>
    Test,
}

/// <summary>
/// One public denoising dataset of the field-validation track (PATHWAY.md, FV), as <c>tools/fetch_ext.sh</c> lays it
/// out under <c>data/ext/&lt;Name&gt;/</c>. Folders are relative to that directory. A noisy image and its clean partner
/// share a file name.
/// </summary>
/// <param name="Name">Folder name under data/ext and the name the CLI takes.</param>
/// <param name="Description">What is imaged, and the file format.</param>
/// <param name="Doi">The Zenodo record's DOI.</param>
/// <param name="Citation">The publication to credit (CC BY 4.0 requires attribution).</param>
/// <param name="TrainNoisy">Folder of noisy training images.</param>
/// <param name="TrainClean">Folder of their clean partners, or null when the dataset has no ground truth.</param>
/// <param name="TestNoisy">Folder of noisy test images, or null when there is no test split.</param>
/// <param name="TestClean">Folder of their clean partners, or null.</param>
/// <param name="TrainCount">Noisy training images in the published dataset.</param>
/// <param name="TestCount">Noisy test images in the published dataset.</param>
public sealed record ExtDataset(string Name, string Description, string Doi, string Citation,
    string TrainNoisy, string? TrainClean, string? TestNoisy, string? TestClean, int TrainCount, int TestCount)
{
    /// <summary>True if noisy images come with clean (high-SNR) partners, so DORA's output can be scored.</summary>
    public bool HasGroundTruth => TrainClean is not null;
}

/// <summary>The external datasets DORA is benchmarked on, and how to plan and load their image pairs.</summary>
public static class ExtDatasets
{
    private const string DeepBacs = "Spahn et al., \"DeepBacs for multi-task bacterial image analysis using open-source deep " +
                                    "learning approaches\", Communications Biology 5, 688 (2022)";
    private const string ZeroCost = "von Chamier et al., \"Democratising deep learning for microscopy with ZeroCostDL4Mic\", " +
                                    "Nature Communications 12, 2276 (2021)";

    /// <summary>Every dataset <c>tools/fetch_ext.sh</c> downloads, in its order.</summary>
    public static IReadOnlyList<ExtDataset> All { get; } =
    [
        new("deepbacs-mreb", "E. coli MreB-sfGFP, live cells, paired low/high SNR, 16-bit 512²", "10.5281/zenodo.6460867",
            DeepBacs, "train/low_SNR", "train/high_SNR", "test/low_SNR", "test/high_SNR", 30, 5),
        new("deepbacs-nucleoid", "E. coli H-NS-mScarlet-I, fixed cells, paired low/high SNR, 8-bit 512²", "10.5281/zenodo.5551112",
            DeepBacs, "train/low_SNR", "train/high_SNR", "test/low_SNR", "test/high_SNR", 28, 2),
        new("deepbacs-bsubtilis", "B. subtilis FtsZ-GFP frames, no ground truth, 16-bit 1024²", "10.5281/zenodo.5551135",
            DeepBacs, "train", null, null, null, 3, 0),
        new("zerocost-care2d", "CARE (2D) example, paired low/high SNR, float32 1024²", "10.5281/zenodo.3713330",
            ZeroCost, "CARE (2D)/Training - Low SNR images", "CARE (2D)/Training - High SNR images",
            "CARE (2D)/Test - Low SNR images", "CARE (2D)/Test - High SNR images", 22, 2),
    ];

    /// <summary>The dataset called <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">No such dataset (the message lists the known names).</exception>
    public static ExtDataset Get(string name) =>
        All.FirstOrDefault(d => d.Name == name)
        ?? throw new ArgumentException($"Unknown external dataset '{name}'. Known: {string.Join(", ", All.Select(d => d.Name))}.");

    /// <summary>True once <c>tools/fetch_ext.sh</c> has verified and extracted the dataset under <paramref name="root"/>.</summary>
    public static bool IsPresent(string root, ExtDataset dataset) => File.Exists(Path.Combine(root, dataset.Name, ".complete"));

    /// <summary>The noisy images of a split, in ordinal file-name order (empty if the dataset has no such split).</summary>
    public static List<string> PlanNoisy(string root, ExtDataset dataset, ExtSplit split)
    {
        string? folder = split == ExtSplit.Train ? dataset.TrainNoisy : dataset.TestNoisy;
        return folder is null ? [] : TiffFiles(Path.Combine(root, dataset.Name, folder));
    }

    /// <summary>
    /// Noisy/clean pairs of a split, matched by file name, in ordinal order. Each file is its own field of view
    /// (<see cref="ImagePair.GroupId"/> = dataset/stem), as in the published train/test splits.
    /// </summary>
    /// <exception cref="InvalidOperationException">The dataset has no ground truth.</exception>
    /// <exception cref="InvalidDataException">An image has no partner of the same name.</exception>
    public static List<FmdDataset.PlannedPair> PlanPairs(string root, ExtDataset dataset, ExtSplit split)
    {
        if (!dataset.HasGroundTruth) throw new InvalidOperationException($"{dataset.Name} has no ground truth to pair with.");
        string dir = Path.Combine(root, dataset.Name);
        var (noisyFolder, cleanFolder) = split == ExtSplit.Train ? (dataset.TrainNoisy, dataset.TrainClean!) : (dataset.TestNoisy!, dataset.TestClean!);
        var noisy = TiffFiles(Path.Combine(dir, noisyFolder));
        var clean = TiffFiles(Path.Combine(dir, cleanFolder)).Select(Path.GetFileName).ToHashSet(StringComparer.Ordinal);
        var unmatched = noisy.Select(Path.GetFileName).Where(n => !clean.Contains(n)).ToList();
        unmatched.AddRange(clean.Except(noisy.Select(Path.GetFileName), StringComparer.Ordinal)!);
        if (unmatched.Count > 0)
            throw new InvalidDataException($"{dataset.Name} {split}: no partner for {string.Join(", ", unmatched.Take(5))}.");
        return noisy.Select(path =>
        {
            string file = Path.GetFileName(path);
            return new FmdDataset.PlannedPair(path, Path.Combine(dir, cleanFolder, file),
                $"{dataset.Name}/{Path.GetFileNameWithoutExtension(file)}", $"{dataset.Name}/{file}");
        }).ToList();
    }

    /// <summary>
    /// The training pairs that a training run with <paramref name="seed"/> holds out for validation
    /// (<see cref="DatasetSplit.Split"/> with <paramref name="fraction"/>). The split depends only on the set of groups
    /// (one per image here) and the seed, so this reproduces it exactly without loading anything. Quality passes choose
    /// settings on these images and touch the test split only once (quality pass Q12).
    /// </summary>
    public static List<FmdDataset.PlannedPair> ValidationPlan(string root, ExtDataset dataset, int seed, double fraction = 0.2)
    {
        var pairs = PlanPairs(root, dataset, ExtSplit.Train);
        var placeholder = new GrayImage(1, 1);
        var (_, held) = DatasetSplit.Split(pairs.Select(p => new ImagePair(placeholder, null, p.GroupId, p.Name)).ToList(), fraction, seed);
        var heldOut = held.Select(p => p.GroupId).ToHashSet();
        return pairs.Where(p => heldOut.Contains(p.GroupId)).ToList();
    }

    /// <summary>Loads and normalizes a split's pairs (each by its noisy image's percentiles, as at inference).</summary>
    public static List<ImagePair> LoadPairs(string root, ExtDataset dataset, ExtSplit split) =>
        FmdDataset.Load(PlanPairs(root, dataset, split));

    /// <summary>
    /// The training split, normalized for training (quality pass Q18's recipe).
    /// <list type="bullet">
    /// <item><b>Supervised:</b> the pairs, with noisy and clean each normalized by <i>its own</i> 0.1st and 99.9th
    /// percentiles, CARE's convention. Here the clean partner is 3–11× brighter than the noisy image; normalizing both by
    /// the noisy image's map (DORA's FMD convention, where both are at one scale) would ask the network to learn that
    /// gain too.</item>
    /// <item><b>Noise2Void:</b> the noisy images alone, as a user without ground truth would have them.</item>
    /// </list>
    /// Each file is its own field of view, so the trainer's validation split holds out whole images. The test split is
    /// never touched.
    /// </summary>
    public static List<ImagePair> LoadTraining(string root, ExtDataset dataset, bool supervised)
    {
        if (supervised)
        {
            var plan = PlanPairs(root, dataset, ExtSplit.Train);
            var pairs = new ImagePair[plan.Count];
            Parallel.For(0, plan.Count, i =>
            {
                var p = plan[i];
                var noisy = ImageLoader.Load(p.NoisyPath);
                var clean = ImageLoader.Load(p.CleanPath);
                var pair = new ImagePair(Normalization.Apply(noisy, Normalization.FromPercentiles(noisy.Pixels)),
                    Normalization.Apply(clean, Normalization.FromPercentiles(clean.Pixels)), p.GroupId, p.Name);
                pair.Validate();
                pairs[i] = pair;
            });
            return pairs.ToList();
        }
        return PlanNoisy(root, dataset, ExtSplit.Train).AsParallel().AsOrdered().Select(path =>
        {
            var noisy = ImageLoader.Load(path);
            string file = Path.GetFileName(path);
            return new ImagePair(Normalization.Apply(noisy, Normalization.FromPercentiles(noisy.Pixels)), null,
                $"{dataset.Name}/{Path.GetFileNameWithoutExtension(file)}", $"{dataset.Name}/{file}");
        }).ToList();
    }

    private static List<string> TiffFiles(string dir)
    {
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException($"{dir}: missing (run tools/fetch_ext.sh).");
        return Directory.EnumerateFiles(dir)
            .Where(f => f.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase))
            .Order(StringComparer.Ordinal).ToList();
    }
}

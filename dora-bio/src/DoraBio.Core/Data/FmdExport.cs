using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>
/// FMD in the folder layout of the external sets, for other tools (brick 38's scoreboard): the noisy frames of DORA's
/// default training plan (<see cref="FmdDataset.PlanTraining"/> with default <see cref="FmdOptions"/>: 4 raw frames per
/// FOV, FOV 19 never) go to <c>train/low_SNR</c>, and test_mix's raw images to <c>test/low_SNR</c>, all as float32 TIFFs
/// with the 8-bit values unchanged. Another tool trains on the first and predicts the second; <c>fmd-eval
/// --predictions</c> then scores the outputs against test_mix's ground truth by <see cref="PredictionName"/>.
/// </summary>
public static class FmdExport
{
    /// <summary>True once <see cref="Export"/> has finished under <paramref name="outDir"/>.</summary>
    public static bool IsPresent(string outDir) => File.Exists(Path.Combine(outDir, ".complete"));

    /// <summary>The file a test image's output is expected in: its name with a .tif extension.</summary>
    public static string PredictionName(FmdDataset.PlannedPair test) => Path.GetFileNameWithoutExtension(test.NoisyPath) + ".tif";

    /// <summary>A training frame's exported name: its plan name with '/' replaced by '_' (unique, since plan names are).</summary>
    public static string TrainingName(FmdDataset.PlannedPair frame) =>
        Path.GetFileNameWithoutExtension(frame.Name.Replace('/', '_')) + ".tif";

    /// <summary>Exports (skipped if already complete). Returns (training frames, test images).</summary>
    /// <param name="root">The FMD root (with test_mix/).</param>
    /// <param name="outDir">Where to write.</param>
    public static (int Train, int Test) Export(string root, string outDir)
    {
        var train = FmdDataset.PlanTraining(root, new FmdOptions());
        var test = FmdDataset.PlanTestMix(Path.Combine(root, "test_mix"), "raw");
        if (IsPresent(outDir)) return (train.Count, test.Count);
        if (Directory.Exists(outDir)) Directory.Delete(outDir, recursive: true);
        Write(train, Path.Combine(outDir, "train", "low_SNR"), TrainingName);
        Write(test, Path.Combine(outDir, "test", "low_SNR"), PredictionName);
        File.WriteAllText(Path.Combine(outDir, ".complete"), $"{train.Count} {test.Count}\n");
        return (train.Count, test.Count);
    }

    private static void Write(IReadOnlyList<FmdDataset.PlannedPair> pairs, string dir, Func<FmdDataset.PlannedPair, string> name)
    {
        Directory.CreateDirectory(dir);
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in pairs)
        {
            string file = name(p);
            if (!names.Add(file)) throw new InvalidDataException($"Two images export to {file}.");
            File.WriteAllBytes(Path.Combine(dir, file), TiffCodec.Encode(ImageLoader.Load(p.NoisyPath), TiffSampleType.Float32));
        }
    }
}

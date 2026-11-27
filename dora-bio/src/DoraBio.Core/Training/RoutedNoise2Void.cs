using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

/// <summary>Which half of a routed Noise2Void recipe a network serves.</summary>
public enum NoiseKind
{
    /// <summary>Images whose noise check says the noise is independent from pixel to pixel (or that are too small to judge).</summary>
    Independent,

    /// <summary>Images whose noise check says the noise is correlated between neighbouring pixels.</summary>
    Correlated,
}

/// <summary>One network of a routed recipe: its architecture, training settings and blind-spot footprint.</summary>
/// <param name="Kind">The images it serves.</param>
/// <param name="Network">Architecture.</param>
/// <param name="Trainer">Training settings.</param>
/// <param name="Footprint">Structured blind-spot footprint, or null for plain blind spots.</param>
public sealed record RoutedRecipe(NoiseKind Kind, NetConfig Network, TrainerConfig Trainer, MaskFootprint? Footprint);

/// <summary>One trained network of a routed run.</summary>
/// <param name="Kind">The images it serves.</param>
/// <param name="WeightsPath">Where its weights were written.</param>
/// <param name="Images">Training images it saw.</param>
/// <param name="Result">The trainer's result.</param>
public sealed record RoutedNetwork(NoiseKind Kind, string WeightsPath, int Images, TrainingResult Result);

/// <summary>
/// DORA's default Noise2Void recipe since Q20 (quality pass Q19's routed design, adopted once CARE 2D's floor was
/// dropped). Every training image gets the noise check (<see cref="NoiseDiagnostics"/>): images it calls independent
/// train the U-Net under Q2's rule C (16 base features, depth 2, all epochs, last epoch kept, no ground truth in the
/// choice), images it calls correlated train P3's network (the dilated stack with diamond-2 structured blind spots, today's
/// epoch rule). <c>denoise --weights-correlated</c> then routes each frame the same way. On FMD this scored 33.24 dB on
/// raw test_mix without any ground truth (docs/RESULTS.md).
/// </summary>
public static class RoutedNoise2Void
{
    /// <summary>The network for images with independent noise, built on <paramref name="baseConfig"/>'s budget and seed.</summary>
    public static RoutedRecipe Independent(TrainerConfig baseConfig) => new(NoiseKind.Independent, NetConfig.UNet(16, 2),
        baseConfig with { Mode = TrainingMode.Noise2Void, Selection = ValidationSignal.Loss, EarlyStopPatience = int.MaxValue, KeepLastEpoch = true },
        null);

    /// <summary>The network for images with correlated noise (P3's), built on <paramref name="baseConfig"/>'s budget and seed.</summary>
    public static RoutedRecipe Correlated(TrainerConfig baseConfig) => new(NoiseKind.Correlated, NetConfig.Default,
        baseConfig with { Mode = TrainingMode.Noise2Void }, StructuredMasking.DefaultFootprint);

    /// <summary>Splits images by their noise-check verdicts: true → correlated; false or null (too small to judge) → independent.</summary>
    /// <exception cref="ArgumentException">Not one verdict per image.</exception>
    public static (List<ImagePair> Independent, List<ImagePair> Correlated) Split(IReadOnlyList<ImagePair> images, IReadOnlyList<bool?> verdicts)
    {
        if (verdicts.Count != images.Count) throw new ArgumentException("One verdict per image is needed.", nameof(verdicts));
        var independent = new List<ImagePair>();
        var correlated = new List<ImagePair>();
        for (int i = 0; i < images.Count; i++) (verdicts[i] == true ? correlated : independent).Add(images[i]);
        return (independent, correlated);
    }

    /// <summary>Where the correlated-image network goes next to <paramref name="outPath"/>: <c>w.bin</c> → <c>w.correlated.bin</c>.</summary>
    public static string CorrelatedPath(string outPath) =>
        Path.Combine(Path.GetDirectoryName(outPath) ?? "", Path.GetFileNameWithoutExtension(outPath) + ".correlated" + Path.GetExtension(outPath));

    /// <summary>
    /// Trains the routed recipe: one network per kind of image present. A set of one kind gives one network at
    /// <paramref name="outPath"/>; a mixed set gives the independent-image network at <paramref name="outPath"/> and the
    /// correlated-image one at <paramref name="correlatedPath"/> (default <see cref="CorrelatedPath"/>).
    /// </summary>
    /// <param name="images">Normalized training images.</param>
    /// <param name="baseConfig">Budget, batch, patch, learning rate, masking and seed shared by both networks.</param>
    /// <param name="outPath">Weights path.</param>
    /// <param name="description">What the data is, for the weights' provenance and the journey log.</param>
    /// <param name="journalPath">Journey log, or null.</param>
    /// <param name="onEpoch">Called after every epoch with the kind being trained.</param>
    /// <param name="correlatedPath">Path for the correlated-image network of a mixed set.</param>
    /// <param name="verdicts">Noise-check verdicts if already computed.</param>
    /// <exception cref="ArgumentException">No images.</exception>
    public static List<RoutedNetwork> Train(IReadOnlyList<ImagePair> images, TrainerConfig baseConfig, string outPath, string description,
        string? journalPath = null, Action<NoiseKind, EpochReport>? onEpoch = null, string? correlatedPath = null, IReadOnlyList<bool?>? verdicts = null)
    {
        if (images.Count == 0) throw new ArgumentException("No training images.", nameof(images));
        verdicts ??= StructuredMasking.Diagnose(images);
        var (independent, correlated) = Split(images, verdicts);
        bool mixed = independent.Count > 0 && correlated.Count > 0;
        var trained = new List<RoutedNetwork>();
        foreach (var (recipe, part) in new[] { (Independent(baseConfig), independent), (Correlated(baseConfig), correlated) })
        {
            if (part.Count == 0) continue;
            string path = recipe.Kind == NoiseKind.Correlated && mixed ? correlatedPath ?? CorrelatedPath(outPath) : outPath;
            var data = recipe.Footprint is null ? part : StructuredMasking.Assign(part, StructuredMaskMode.Always, recipe.Footprint);
            string kind = recipe.Kind == NoiseKind.Independent
                ? "routed (Q20): images whose noise check says independent; U-Net under Q2's rule C"
                : $"routed (Q20): images whose noise check says correlated; structured blind spots ({recipe.Footprint})";
            var net = DenoisingNet.Create(recipe.Network, recipe.Trainer.Seed);
            var result = TrainingJob.Run(net, data, recipe.Trainer,
                new TrainingJobOptions { DatasetDescription = $"{description}; {part.Count} of {images.Count} images, {kind}", WeightsPath = path, JournalPath = journalPath },
                report => onEpoch?.Invoke(recipe.Kind, report));
            trained.Add(new RoutedNetwork(recipe.Kind, path, part.Count, result));
        }
        return trained;
    }
}

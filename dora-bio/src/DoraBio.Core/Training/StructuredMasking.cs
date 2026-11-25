using DoraBio.Core.Data;

namespace DoraBio.Core.Training;

/// <summary>When Noise2Void uses structured blind spots (<see cref="MaskFootprint"/>) instead of plain ones.</summary>
public enum StructuredMaskMode
{
    /// <summary>Plain blind spots everywhere (classic Noise2Void).</summary>
    Off,

    /// <summary>Structured blind spots on the images whose noise <see cref="NoiseDiagnostics"/> finds correlated.</summary>
    Auto,

    /// <summary>Structured blind spots on every image.</summary>
    Always,
}

/// <summary>
/// Decides which training images get structured blind spots. It only sets <see cref="ImagePair.Footprint"/>, which
/// <see cref="PatchSampler"/> and the validation set then honour, so the trainer itself needs no special case.
/// </summary>
public static class StructuredMasking
{
    /// <summary>
    /// The footprint used unless one is given. It must cover the offsets where the noise is correlated; on FMD widefield
    /// that's one pixel across and diagonally and two pixels across, which a radius-2 diamond does
    /// (see docs/RESULTS.md for how it was chosen).
    /// </summary>
    public static readonly MaskFootprint DefaultFootprint = new(FootprintShape.Diamond, 2);

    /// <summary>
    /// Runs the noise check on every image in parallel: true for correlated noise, false for independent, null when
    /// an image is too small to judge (see <see cref="NoiseDiagnostics.MinTiles"/>).
    /// </summary>
    public static bool?[] Diagnose(IReadOnlyList<ImagePair> images)
    {
        var verdicts = new bool?[images.Count];
        Parallel.For(0, images.Count, i =>
        {
            try { verdicts[i] = NoiseDiagnostics.Diagnose(images[i].Noisy).Correlated; }
            catch (ArgumentException) { verdicts[i] = null; }
        });
        return verdicts;
    }

    /// <summary>Returns the images with <see cref="ImagePair.Footprint"/> set according to <paramref name="mode"/>.</summary>
    /// <param name="images">Training images (normalized or not; the noise check is scale-free).</param>
    /// <param name="mode">Off, Auto (flagged images only) or Always.</param>
    /// <param name="footprint">Footprint for the structured images; <see cref="DefaultFootprint"/> if null.</param>
    /// <param name="verdicts">Results of <see cref="Diagnose"/> for these images, if already computed (Auto only).</param>
    public static List<ImagePair> Assign(IReadOnlyList<ImagePair> images, StructuredMaskMode mode,
        MaskFootprint? footprint = null, bool?[]? verdicts = null)
    {
        footprint ??= DefaultFootprint;
        if (mode == StructuredMaskMode.Auto)
        {
            verdicts ??= Diagnose(images);
            if (verdicts.Length != images.Count) throw new ArgumentException("One verdict per image is needed.", nameof(verdicts));
        }
        return images.Select((image, i) => image with
        {
            Footprint = mode switch
            {
                StructuredMaskMode.Always => footprint,
                StructuredMaskMode.Auto when verdicts![i] == true => footprint,
                _ => null,
            },
        }).ToList();
    }
}

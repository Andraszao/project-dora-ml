namespace DoraBio.Core.Data;

/// <summary>What the network is trained to reproduce.</summary>
public enum TrainingMode
{
    /// <summary>Self-supervised blind-spot training on noisy images alone.</summary>
    Noise2Void,

    /// <summary>Noisy → clean regression. Every image needs ground truth.</summary>
    Supervised,

    /// <summary>
    /// Noise2Noise from neighbouring frames (brick 39): the same regression as supervised, but the target is the next or
    /// previous noisy frame of a time-lapse (<see cref="ImagePair.Clean"/> holds it). Single-frame input only, so the
    /// target never appears in the input. Validation reports the loss only: a noisy target can't give a PSNR.
    /// </summary>
    Noise2Noise,
}

/// <summary>A mini-batch of square patches, stored sample-major: index = sample * Size² + y * Size + x.</summary>
/// <remarks>Allocated once and refilled every step by <see cref="PatchSampler.Fill"/>, so training allocates nothing per step.</remarks>
public sealed class TrainingBatch
{
    /// <summary>Number of patches in the batch.</summary>
    public int Count { get; }

    /// <summary>Patch side length in pixels.</summary>
    public int Size { get; }

    /// <summary>Pixels per patch (Size²).</summary>
    public int Plane => Size * Size;

    /// <summary>Frames per input: 1, or the multi-frame network's input frames (each sample's input is that many planes).</summary>
    public int InputFrames { get; }

    /// <summary>All network inputs, patch after patch (InputFrames planes each).</summary>
    public float[] Inputs { get; }

    /// <summary>All targets the outputs are compared against, patch after patch.</summary>
    public float[] Targets { get; }

    /// <summary>All per-pixel loss weights (0 or 1), patch after patch.</summary>
    public float[] Masks { get; }

    /// <summary>
    /// Per sample: the condition (<see cref="NoiseConditioning.Of"/>) of the whole training image the patch was cut from.
    /// Only noise-conditioned networks read it.
    /// </summary>
    public float[] Conditions { get; }

    /// <summary>
    /// Allocates a batch of <paramref name="count"/> patches of <paramref name="size"/>×<paramref name="size"/> pixels,
    /// with <paramref name="inputFrames"/> input planes per patch.
    /// </summary>
    public TrainingBatch(int count, int size, int inputFrames = 1)
    {
        if (count <= 0 || size <= 0 || inputFrames <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        Count = count;
        Size = size;
        InputFrames = inputFrames;
        Inputs = new float[count * inputFrames * Plane];
        Targets = new float[count * Plane];
        Masks = new float[count * Plane];
        Conditions = new float[count];
    }

    /// <summary>Input of sample <paramref name="i"/>: <see cref="InputFrames"/> planes (a view, not a copy).</summary>
    public Span<float> Input(int i) => Inputs.AsSpan(i * InputFrames * Plane, InputFrames * Plane);

    /// <summary>Target plane of sample <paramref name="i"/> (a view, not a copy).</summary>
    public Span<float> Target(int i) => Targets.AsSpan(i * Plane, Plane);

    /// <summary>Loss-mask plane of sample <paramref name="i"/> (a view, not a copy).</summary>
    public Span<float> Mask(int i) => Masks.AsSpan(i * Plane, Plane);
}

/// <summary>
/// Draws random patches with random dihedral augmentation, choosing each image with probability
/// proportional to its area.
/// Noise2Void: input = masked noisy, target = noisy, mask = blind spots (structured if the image has a
/// <see cref="ImagePair.Footprint"/>).
/// Supervised and Noise2Noise: input = noisy, target = clean (or the neighbouring noisy frame), mask = all ones.
/// Multi-frame images (<see cref="ImagePair.Context"/>): the neighbours get the same window and symmetry and fill the
/// other input planes unmasked; blind spots go into the centre plane only.
/// </summary>
public sealed class PatchSampler
{
    private readonly IReadOnlyList<ImagePair> _images;
    private readonly double[] _cumulativeArea;
    private readonly Random _rng;
    private readonly float[] _noisyPatch;
    private readonly float[] _cleanPatch;
    private readonly float[] _scratch;
    private readonly float[] _conditions;   // per image, measured on first use (NaN until then)
    private readonly int _plane;

    /// <summary>Side length of the square patches this sampler produces.</summary>
    public int PatchSize { get; }

    /// <summary>Whether batches are built for Noise2Void or supervised training.</summary>
    public TrainingMode Mode { get; }

    /// <summary>Frames per input (the images' <see cref="ImagePair.FrameCount"/>; all must agree).</summary>
    public int InputFrames { get; }

    /// <summary>Fraction of pixels blind-spotted per patch (Noise2Void only).</summary>
    public float MaskRatio { get; }

    /// <summary>How patches are augmented (lossless dihedral by default).</summary>
    public AugmentationMode Augmentation { get; }

    /// <param name="images">Normalized training images. Images smaller than the patch are skipped.</param>
    /// <param name="patchSize">Patch side length (at least 8).</param>
    /// <param name="mode">Noise2Void or supervised. Supervised requires a clean image for every usable pair.</param>
    /// <param name="maskRatio">Fraction of pixels to blind-spot per patch in Noise2Void mode (e.g. 0.02).</param>
    /// <param name="rng">Owned by the caller and reused for every draw, so a seeded run is reproducible.</param>
    /// <param name="augmentation">Dihedral (default) or the v1-style rotate-and-scale comparison mode.</param>
    public PatchSampler(IReadOnlyList<ImagePair> images, int patchSize, TrainingMode mode, float maskRatio, Random rng,
        AugmentationMode augmentation = AugmentationMode.Dihedral)
    {
        if (patchSize < 8) throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be at least 8.");
        // v1-style rotation and scaling reads a footprint wider than the patch, so images must be large enough for it.
        int minSide = augmentation == AugmentationMode.V1RotateScale ? 2 * V1Augment.Reach(patchSize) + 1 : patchSize;
        var usable = images.Where(i => i.Noisy.Width >= minSide && i.Noisy.Height >= minSide).ToList();
        if (usable.Count == 0)
            throw new ArgumentException($"No image is at least {minSide}x{minSide}; lower the patch size.");
        if (mode == TrainingMode.Supervised && usable.Any(i => !i.HasClean))
            throw new ArgumentException("Supervised training needs a clean image for every training image.");
        if (mode == TrainingMode.Noise2Noise && usable.Any(i => !i.HasClean))
            throw new ArgumentException("Noise2Noise training needs a neighbouring frame (the target) for every training image.");
        if (mode == TrainingMode.Noise2Noise && usable.Any(i => i.Context is not null))
            throw new ArgumentException("Noise2Noise takes one input frame: its target must never be part of the input.");
        foreach (var image in usable) image.Validate();
        InputFrames = usable[0].FrameCount;
        if (usable.Any(i => i.FrameCount != InputFrames)) throw new ArgumentException("Every training image must have the same number of frames.");
        if (InputFrames > 1 && augmentation == AugmentationMode.V1RotateScale)
            throw new ArgumentException("v1-style augmentation works on single frames only.");

        _images = usable;
        _rng = rng;
        PatchSize = patchSize;
        Mode = mode;
        MaskRatio = maskRatio;
        Augmentation = augmentation;

        _cumulativeArea = new double[usable.Count];
        double total = 0;
        for (int i = 0; i < usable.Count; i++)
        {
            total += (double)usable[i].Noisy.Width * usable[i].Noisy.Height;
            _cumulativeArea[i] = total;
        }

        int plane = patchSize * patchSize;
        _plane = plane;
        _noisyPatch = new float[plane];
        _cleanPatch = new float[plane];
        _scratch = new float[plane];
        _conditions = new float[usable.Count];
        Array.Fill(_conditions, float.NaN);
    }

    /// <summary>Overwrites every sample in <paramref name="batch"/> with a freshly drawn, augmented patch.</summary>
    public void Fill(TrainingBatch batch)
    {
        if (batch.Size != PatchSize) throw new ArgumentException("Batch patch size does not match the sampler.");
        if (batch.InputFrames != InputFrames) throw new ArgumentException("Batch frame count does not match the sampler.");
        for (int i = 0; i < batch.Count; i++)
            batch.Conditions[i] = FillSample(batch.Input(i), batch.Target(i), batch.Mask(i));
    }

    /// <summary>
    /// Draws one patch: pick an image, pick a position, pick one of the 8 symmetries, and apply the same
    /// symmetry to noisy and clean so they stay aligned. Then build input, target and mask for the mode.
    /// Returns the condition of the image the patch came from: measured once per image on the whole image (never on
    /// the patch), and without touching the random generator, so runs of plain networks are unchanged by it.
    /// </summary>
    private float FillSample(Span<float> input, Span<float> target, Span<float> mask)
    {
        int index = PickImage();
        var image = _images[index];
        if (float.IsNaN(_conditions[index])) _conditions[index] = NoiseConditioning.Of(image.Noisy);
        if (Augmentation == AugmentationMode.V1RotateScale)
        {
            // v1's recipe: one random angle and scale, applied identically to noisy and clean.
            int reach = V1Augment.Reach(PatchSize);
            double cx = reach + _rng.NextDouble() * (image.Noisy.Width - 1 - 2 * reach);
            double cy = reach + _rng.NextDouble() * (image.Noisy.Height - 1 - 2 * reach);
            double angle = (_rng.NextDouble() * 2 - 1) * V1Augment.MaxAngleDegrees * Math.PI / 180;
            double scale = V1Augment.MinScale + _rng.NextDouble() * (V1Augment.MaxScale - V1Augment.MinScale);
            V1Augment.RotateScaleNearest(image.Noisy, cx, cy, angle, scale, PatchSize, _noisyPatch);
            if (Mode != TrainingMode.Noise2Void)
                V1Augment.RotateScaleNearest(image.Clean!, cx, cy, angle, scale, PatchSize, _cleanPatch);
        }
        else
        {
            int x = _rng.Next(image.Noisy.Width - PatchSize + 1);
            int y = _rng.Next(image.Noisy.Height - PatchSize + 1);
            int transform = _rng.Next(Dihedral.Count);
            ExtractAugmented(image.Noisy, x, y, transform, _noisyPatch);
            if (Mode != TrainingMode.Noise2Void) ExtractAugmented(image.Clean!, x, y, transform, _cleanPatch);
            // Neighbouring frames: the same window and symmetry, into the planes before and after the centre. They draw
            // nothing from the random generator, so single-frame runs are unchanged.
            if (image.Context is { } context)
                for (int k = 0; k < context.Length; k++)
                {
                    int channel = k < context.Length / 2 ? k : k + 1;
                    ExtractAugmented(context[k], x, y, transform, input.Slice(channel * _plane, _plane));
                }
        }

        var centerInput = input.Slice(InputFrames / 2 * _plane, _plane);   // the whole input when single-frame
        if (Mode != TrainingMode.Noise2Void)
        {
            _noisyPatch.CopyTo(centerInput);
            _cleanPatch.CopyTo(target);
            mask.Fill(1f);
        }
        else
        {
            _noisyPatch.CopyTo(target);
            BlindSpot.Apply(_noisyPatch, centerInput, mask, PatchSize, MaskRatio, _rng, footprint: image.Footprint);
        }
        return _conditions[index];
    }

    /// <summary>
    /// Area-weighted choice: a uniform draw on [0, total area) is located in the cumulative-area table, so
    /// every pixel in the dataset is equally likely to start a patch, however the images vary in size.
    /// </summary>
    private int PickImage()
    {
        double r = _rng.NextDouble() * _cumulativeArea[^1];
        int index = Array.BinarySearch(_cumulativeArea, r);
        if (index < 0) index = ~index;
        return Math.Min(index, _images.Count - 1);
    }

    /// <summary>Copies the PatchSize² window at (x, y) into scratch, then writes its dihedral transform to <paramref name="destination"/>.</summary>
    private void ExtractAugmented(GrayImage source, int x, int y, int transform, Span<float> destination)
    {
        for (int row = 0; row < PatchSize; row++)
            source.Pixels.AsSpan((y + row) * source.Width + x, PatchSize).CopyTo(_scratch.AsSpan(row * PatchSize, PatchSize));
        Dihedral.Apply(_scratch, destination, PatchSize, transform);
    }
}

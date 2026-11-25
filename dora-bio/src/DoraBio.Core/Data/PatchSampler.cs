namespace DoraBio.Core.Data;

/// <summary>What the network is trained to reproduce.</summary>
public enum TrainingMode
{
    /// <summary>Self-supervised blind-spot training on noisy images alone.</summary>
    Noise2Void,

    /// <summary>Noisy → clean regression. Every image needs ground truth.</summary>
    Supervised,
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

    /// <summary>All network inputs, patch after patch.</summary>
    public float[] Inputs { get; }

    /// <summary>All targets the outputs are compared against, patch after patch.</summary>
    public float[] Targets { get; }

    /// <summary>All per-pixel loss weights (0 or 1), patch after patch.</summary>
    public float[] Masks { get; }

    /// <summary>Allocates a batch of <paramref name="count"/> patches of <paramref name="size"/>×<paramref name="size"/> pixels.</summary>
    public TrainingBatch(int count, int size)
    {
        if (count <= 0 || size <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        Count = count;
        Size = size;
        Inputs = new float[count * Plane];
        Targets = new float[count * Plane];
        Masks = new float[count * Plane];
    }

    /// <summary>Input plane of sample <paramref name="i"/> (a view, not a copy).</summary>
    public Span<float> Input(int i) => Inputs.AsSpan(i * Plane, Plane);

    /// <summary>Target plane of sample <paramref name="i"/> (a view, not a copy).</summary>
    public Span<float> Target(int i) => Targets.AsSpan(i * Plane, Plane);

    /// <summary>Loss-mask plane of sample <paramref name="i"/> (a view, not a copy).</summary>
    public Span<float> Mask(int i) => Masks.AsSpan(i * Plane, Plane);
}

/// <summary>
/// Draws random patches with random dihedral augmentation, choosing each image with probability
/// proportional to its area.
/// Noise2Void: input = masked noisy, target = noisy, mask = blind spots.
/// Supervised: input = noisy, target = clean, mask = all ones.
/// </summary>
public sealed class PatchSampler
{
    private readonly IReadOnlyList<ImagePair> _images;
    private readonly double[] _cumulativeArea;
    private readonly Random _rng;
    private readonly float[] _noisyPatch;
    private readonly float[] _cleanPatch;
    private readonly float[] _scratch;

    /// <summary>Side length of the square patches this sampler produces.</summary>
    public int PatchSize { get; }

    /// <summary>Whether batches are built for Noise2Void or supervised training.</summary>
    public TrainingMode Mode { get; }

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
        foreach (var image in usable) image.Validate();

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
        _noisyPatch = new float[plane];
        _cleanPatch = new float[plane];
        _scratch = new float[plane];
    }

    /// <summary>Overwrites every sample in <paramref name="batch"/> with a freshly drawn, augmented patch.</summary>
    public void Fill(TrainingBatch batch)
    {
        if (batch.Size != PatchSize) throw new ArgumentException("Batch patch size does not match the sampler.");
        for (int i = 0; i < batch.Count; i++)
            FillSample(batch.Input(i), batch.Target(i), batch.Mask(i));
    }

    /// <summary>
    /// Draws one patch: pick an image, pick a position, pick one of the 8 symmetries, and apply the same
    /// symmetry to noisy and clean so they stay aligned. Then build input, target and mask for the mode.
    /// </summary>
    private void FillSample(Span<float> input, Span<float> target, Span<float> mask)
    {
        var image = PickImage();
        if (Augmentation == AugmentationMode.V1RotateScale)
        {
            // v1's recipe: one random angle and scale, applied identically to noisy and clean.
            int reach = V1Augment.Reach(PatchSize);
            double cx = reach + _rng.NextDouble() * (image.Noisy.Width - 1 - 2 * reach);
            double cy = reach + _rng.NextDouble() * (image.Noisy.Height - 1 - 2 * reach);
            double angle = (_rng.NextDouble() * 2 - 1) * V1Augment.MaxAngleDegrees * Math.PI / 180;
            double scale = V1Augment.MinScale + _rng.NextDouble() * (V1Augment.MaxScale - V1Augment.MinScale);
            V1Augment.RotateScaleNearest(image.Noisy, cx, cy, angle, scale, PatchSize, _noisyPatch);
            if (Mode == TrainingMode.Supervised)
                V1Augment.RotateScaleNearest(image.Clean!, cx, cy, angle, scale, PatchSize, _cleanPatch);
        }
        else
        {
            int x = _rng.Next(image.Noisy.Width - PatchSize + 1);
            int y = _rng.Next(image.Noisy.Height - PatchSize + 1);
            int transform = _rng.Next(Dihedral.Count);
            ExtractAugmented(image.Noisy, x, y, transform, _noisyPatch);
            if (Mode == TrainingMode.Supervised) ExtractAugmented(image.Clean!, x, y, transform, _cleanPatch);
        }

        if (Mode == TrainingMode.Supervised)
        {
            _noisyPatch.CopyTo(input);
            _cleanPatch.CopyTo(target);
            mask.Fill(1f);
        }
        else
        {
            _noisyPatch.CopyTo(target);
            BlindSpot.Apply(_noisyPatch, input, mask, PatchSize, MaskRatio, _rng);
        }
    }

    /// <summary>
    /// Area-weighted choice: a uniform draw on [0, total area) is located in the cumulative-area table, so
    /// every pixel in the dataset is equally likely to start a patch, however the images vary in size.
    /// </summary>
    private ImagePair PickImage()
    {
        double r = _rng.NextDouble() * _cumulativeArea[^1];
        int index = Array.BinarySearch(_cumulativeArea, r);
        if (index < 0) index = ~index;
        return _images[Math.Min(index, _images.Count - 1)];
    }

    /// <summary>Copies the PatchSize² window at (x, y) into scratch, then writes its dihedral transform to <paramref name="destination"/>.</summary>
    private void ExtractAugmented(GrayImage source, int x, int y, int transform, Span<float> destination)
    {
        for (int row = 0; row < PatchSize; row++)
            source.Pixels.AsSpan((y + row) * source.Width + x, PatchSize).CopyTo(_scratch.AsSpan(row * PatchSize, PatchSize));
        Dihedral.Apply(_scratch, destination, PatchSize, transform);
    }
}

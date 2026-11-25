namespace DoraBio.Core.Data;

public enum TrainingMode
{
    /// <summary>Self-supervised blind-spot training on noisy images alone.</summary>
    Noise2Void,

    /// <summary>Noisy → clean regression. Every image needs ground truth.</summary>
    Supervised,
}

/// <summary>A mini-batch of square patches, stored sample-major: index = sample * Size² + y * Size + x.</summary>
public sealed class TrainingBatch
{
    public int Count { get; }
    public int Size { get; }
    public int Plane => Size * Size;
    public float[] Inputs { get; }
    public float[] Targets { get; }
    public float[] Masks { get; }

    public TrainingBatch(int count, int size)
    {
        if (count <= 0 || size <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        Count = count;
        Size = size;
        Inputs = new float[count * Plane];
        Targets = new float[count * Plane];
        Masks = new float[count * Plane];
    }

    public Span<float> Input(int i) => Inputs.AsSpan(i * Plane, Plane);
    public Span<float> Target(int i) => Targets.AsSpan(i * Plane, Plane);
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

    public int PatchSize { get; }
    public TrainingMode Mode { get; }
    public float MaskRatio { get; }

    public PatchSampler(IReadOnlyList<ImagePair> images, int patchSize, TrainingMode mode, float maskRatio, Random rng)
    {
        if (patchSize < 8) throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be at least 8.");
        var usable = images.Where(i => i.Noisy.Width >= patchSize && i.Noisy.Height >= patchSize).ToList();
        if (usable.Count == 0)
            throw new ArgumentException($"No image is at least {patchSize}x{patchSize}; lower the patch size.");
        if (mode == TrainingMode.Supervised && usable.Any(i => !i.HasClean))
            throw new ArgumentException("Supervised training needs a clean image for every training image.");
        foreach (var image in usable) image.Validate();

        _images = usable;
        _rng = rng;
        PatchSize = patchSize;
        Mode = mode;
        MaskRatio = maskRatio;

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

    public void Fill(TrainingBatch batch)
    {
        if (batch.Size != PatchSize) throw new ArgumentException("Batch patch size does not match the sampler.");
        for (int i = 0; i < batch.Count; i++)
            FillSample(batch.Input(i), batch.Target(i), batch.Mask(i));
    }

    private void FillSample(Span<float> input, Span<float> target, Span<float> mask)
    {
        var image = PickImage();
        int x = _rng.Next(image.Noisy.Width - PatchSize + 1);
        int y = _rng.Next(image.Noisy.Height - PatchSize + 1);
        int transform = _rng.Next(Dihedral.Count);

        ExtractAugmented(image.Noisy, x, y, transform, _noisyPatch);

        if (Mode == TrainingMode.Supervised)
        {
            ExtractAugmented(image.Clean!, x, y, transform, _cleanPatch);
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

    private ImagePair PickImage()
    {
        double r = _rng.NextDouble() * _cumulativeArea[^1];
        int index = Array.BinarySearch(_cumulativeArea, r);
        if (index < 0) index = ~index;
        return _images[Math.Min(index, _images.Count - 1)];
    }

    private void ExtractAugmented(GrayImage source, int x, int y, int transform, Span<float> destination)
    {
        for (int row = 0; row < PatchSize; row++)
            source.Pixels.AsSpan((y + row) * source.Width + x, PatchSize).CopyTo(_scratch.AsSpan(row * PatchSize, PatchSize));
        Dihedral.Apply(_scratch, destination, PatchSize, transform);
    }
}

namespace DoraBio.Core.Data;

/// <summary>
/// A noisy image with optional clean ground truth (for example an average of many frames).
/// <paramref name="GroupId"/> names the field of view (FOV): frames of the same FOV are never split
/// between training and validation, so the validation scores aren't inflated by leakage.
/// </summary>
/// <param name="Noisy">The image the network sees.</param>
/// <param name="Clean">Optional ground truth of the same size. Used for supervised training and for PSNR/SSIM scoring.</param>
/// <param name="GroupId">Field-of-view identifier; pairs with the same id always land on the same side of the split.</param>
/// <param name="Name">Display name for logs and error messages (usually the file name).</param>
public sealed record ImagePair(GrayImage Noisy, GrayImage? Clean, string GroupId, string Name)
{
    /// <summary>True when ground truth is available.</summary>
    public bool HasClean => Clean is not null;

    /// <summary>
    /// Structured blind-spot footprint for Noise2Void on this image, or null for a plain blind spot (the right choice
    /// when its noise is independent between pixels). <see cref="Training.StructuredMasking"/> sets it on the images whose
    /// noise <see cref="NoiseDiagnostics"/> finds correlated. Ignored by supervised training.
    /// </summary>
    public MaskFootprint? Footprint { get; init; }

    /// <summary>
    /// For a multi-frame network (brick 39): the frame's neighbours in time order, without the frame itself, e.g.
    /// [t−1, t+1] for three frames. Reflected at a sequence's ends (<see cref="Sequences.FrameIndices"/>), never
    /// clamped, because a copy of frame t in a neighbour channel would show the network the very pixels its blind spot
    /// hides. Each is normalized by its own percentiles, as at inference. Null for single-frame training.
    /// </summary>
    public GrayImage[]? Context { get; init; }

    /// <summary>Frames the network sees for this image: 1 plus the context frames.</summary>
    public int FrameCount => 1 + (Context?.Length ?? 0);

    /// <summary>Throws if the clean image's or a context frame's size differs from the noisy one's.</summary>
    public void Validate()
    {
        if (Clean is not null && (Clean.Width != Noisy.Width || Clean.Height != Noisy.Height))
            throw new ArgumentException(
                $"'{Name}': clean image is {Clean.Width}x{Clean.Height} but noisy is {Noisy.Width}x{Noisy.Height}.");
        if (Context is { } context)
        {
            if (context.Length % 2 != 0) throw new ArgumentException($"'{Name}': context frames must be as many before as after.");
            if (context.Any(c => c.Width != Noisy.Width || c.Height != Noisy.Height))
                throw new ArgumentException($"'{Name}': a neighbouring frame differs in size from the frame.");
        }
    }

    /// <summary>Crops both images to the same rectangle and appends <paramref name="suffix"/> to the group id and name.</summary>
    /// <remarks>
    /// The suffix gives the train and validation halves of a spatially split FOV different group ids,
    /// so they stay apart.
    /// </remarks>
    public ImagePair Crop(int x, int y, int w, int h, string suffix) => this with
    {
        Noisy = Noisy.Crop(x, y, w, h),
        Clean = Clean?.Crop(x, y, w, h),
        Context = Context?.Select(c => c.Crop(x, y, w, h)).ToArray(),
        GroupId = GroupId + suffix,
        Name = Name + suffix,
    };

    /// <summary>
    /// Normalizes both images with the map computed from the noisy image, because only the noisy image
    /// exists at inference time. Clean and noisy therefore stay on the same intensity scale.
    /// </summary>
    public (ImagePair Pair, IntensityMap Map) Normalize(
        double lowFraction = Normalization.DefaultLowPercentile,
        double highFraction = Normalization.DefaultHighPercentile)
    {
        var map = Normalization.FromPercentiles(Noisy.Pixels, lowFraction, highFraction);
        var pair = this with
        {
            Noisy = Normalization.Apply(Noisy, map),
            Clean = Clean is null ? null : Normalization.Apply(Clean, map),
            // Neighbouring frames are normalized by their own percentiles, as each frame is at inference.
            Context = Context?.Select(c => Normalization.Apply(c, Normalization.FromPercentiles(c.Pixels, lowFraction, highFraction))).ToArray(),
        };
        return (pair, map);
    }
}

/// <summary>Train/validation split by field of view.</summary>
public static class DatasetSplit
{
    /// <summary>
    /// Holds out about <paramref name="validationFraction"/> of the FOV groups, chosen by a seeded shuffle.
    /// With only one group, the images are split spatially instead: the bottom rows go to validation.
    /// </summary>
    public static (List<ImagePair> Train, List<ImagePair> Validation) Split(
        IReadOnlyList<ImagePair> images, double validationFraction, int seed)
    {
        if (images.Count == 0) throw new ArgumentException("Dataset is empty.", nameof(images));
        if (!(validationFraction > 0 && validationFraction < 1))
            throw new ArgumentOutOfRangeException(nameof(validationFraction));

        var groups = images.Select(i => i.GroupId).Distinct().OrderBy(g => g, StringComparer.Ordinal).ToList();
        if (groups.Count >= 2)
        {
            var rng = new Random(seed);
            var shuffled = groups.OrderBy(_ => rng.Next()).ToList();
            int valCount = Math.Clamp((int)Math.Round(groups.Count * validationFraction), 1, groups.Count - 1);
            var valGroups = shuffled.Take(valCount).ToHashSet();
            return (images.Where(i => !valGroups.Contains(i.GroupId)).ToList(),
                    images.Where(i => valGroups.Contains(i.GroupId)).ToList());
        }

        var train = new List<ImagePair>();
        var validation = new List<ImagePair>();
        foreach (var image in images)
        {
            int valRows = Math.Max(1, (int)Math.Round(image.Noisy.Height * validationFraction));
            int trainRows = image.Noisy.Height - valRows;
            if (trainRows < 1) throw new ArgumentException($"'{image.Name}' is too small to split.");
            train.Add(image.Crop(0, 0, image.Noisy.Width, trainRows, ":train"));
            validation.Add(image.Crop(0, trainRows, image.Noisy.Width, valRows, ":val"));
        }
        return (train, validation);
    }
}

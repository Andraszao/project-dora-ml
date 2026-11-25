namespace DoraBio.Core.Data;

/// <summary>
/// A noisy image with optional clean ground truth (for example an average of many frames).
/// <paramref name="GroupId"/> names the field of view (FOV): frames of the same FOV are never split
/// between training and validation, so the validation scores aren't inflated by leakage.
/// </summary>
public sealed record ImagePair(GrayImage Noisy, GrayImage? Clean, string GroupId, string Name)
{
    public bool HasClean => Clean is not null;

    public void Validate()
    {
        if (Clean is not null && (Clean.Width != Noisy.Width || Clean.Height != Noisy.Height))
            throw new ArgumentException(
                $"'{Name}': clean image is {Clean.Width}x{Clean.Height} but noisy is {Noisy.Width}x{Noisy.Height}.");
    }

    public ImagePair Crop(int x, int y, int w, int h, string suffix) =>
        new(Noisy.Crop(x, y, w, h), Clean?.Crop(x, y, w, h), GroupId + suffix, Name + suffix);

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

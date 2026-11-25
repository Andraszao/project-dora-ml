namespace DoraBio.Core.Data;

/// <summary>How training patches are augmented.</summary>
public enum AugmentationMode
{
    /// <summary>
    /// The 8 lossless 90° rotations and flips (<see cref="Dihedral"/>). Pixels move intact, so noise statistics are
    /// unchanged. This is the default.
    /// </summary>
    Dihedral,

    /// <summary>
    /// v1's geometric augmentation, reproduced for comparison: a random rotation of ±15° and scaling by 0.85–1.15,
    /// resampled with nearest neighbour exactly as v1's MNISTLoader did. Resampling duplicates and drops pixels,
    /// so neighbouring pixels can share one noise sample, which is precisely what Noise2Void's blind spot must not
    /// see. Kept only to measure that effect (docs/RESULTS.md); not recommended for training.
    /// </summary>
    V1RotateScale,
}

/// <summary>v1-style rotate-and-scale patch extraction (see <see cref="AugmentationMode.V1RotateScale"/>).</summary>
public static class V1Augment
{
    /// <summary>v1's rotation range: ±15°.</summary>
    public const double MaxAngleDegrees = 15;

    /// <summary>v1's scale range: 0.85–1.15.</summary>
    public const double MinScale = 0.85, MaxScale = 1.15;

    /// <summary>
    /// Pixels from a patch's center to the farthest source pixel it can read, for the worst-case scale and any
    /// rotation (half the patch diagonal, divided by the smallest scale, plus one for rounding).
    /// </summary>
    public static int Reach(int size) => (int)Math.Ceiling(size * Math.Sqrt(2) / (2 * MinScale)) + 1;

    /// <summary>
    /// Fills a size×size patch: patch pixel (i, j) reads the source at the center plus the offset (j − c, i − c)
    /// scaled by 1/<paramref name="scale"/> and rotated by <paramref name="angleRadians"/> (c = (size − 1)/2),
    /// rounded to the nearest pixel.
    /// </summary>
    /// <param name="source">Image to sample. The whole rotated footprint must lie inside it (see <see cref="Reach"/>).</param>
    /// <param name="centerX">Source x coordinate of the patch center.</param>
    /// <param name="centerY">Source y coordinate of the patch center.</param>
    /// <param name="angleRadians">Rotation angle.</param>
    /// <param name="scale">Zoom factor: above 1 magnifies (more duplicated pixels), below 1 shrinks (more dropped pixels).</param>
    /// <param name="size">Patch side length.</param>
    /// <param name="destination">Receives size² values, row-major.</param>
    public static void RotateScaleNearest(GrayImage source, double centerX, double centerY, double angleRadians, double scale,
        int size, Span<float> destination)
    {
        if (destination.Length != size * size) throw new ArgumentException("Destination must be size×size.");
        double c = (size - 1) / 2.0, cos = Math.Cos(angleRadians), sin = Math.Sin(angleRadians);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
            {
                double u = (j - c) / scale, v = (i - c) / scale;
                int sx = (int)Math.Round(centerX + u * cos - v * sin, MidpointRounding.AwayFromZero);
                int sy = (int)Math.Round(centerY + u * sin + v * cos, MidpointRounding.AwayFromZero);
                if ((uint)sx >= (uint)source.Width || (uint)sy >= (uint)source.Height)
                    throw new ArgumentOutOfRangeException(nameof(centerX), "Rotated patch footprint leaves the image.");
                destination[i * size + j] = source[sx, sy];
            }
    }
}

namespace DoraBio.Core.Data;

/// <summary>
/// The 8 lossless symmetries of a square (4 rotations, each with or without a mirror).
/// Microscopy has no preferred orientation, so these are free augmentations. Unlike v1's rotations they
/// move whole pixels and never interpolate, so noise statistics stay exactly the same.
/// </summary>
public static class Dihedral
{
    /// <summary>Number of distinct transforms (valid ids are 0..7).</summary>
    public const int Count = 8;

    /// <summary>
    /// dst[y, x] = src[map(y, x)] for transform <paramref name="id"/> in 0..7.
    /// Transforms 0–3 are 0/90/180/270° rotations; 4–7 mirror first, then rotate.
    /// </summary>
    public static void Apply(ReadOnlySpan<float> src, Span<float> dst, int size, int id)
    {
        if ((uint)id >= Count) throw new ArgumentOutOfRangeException(nameof(id));
        if (src.Length != size * size || dst.Length != size * size)
            throw new ArgumentException("Dihedral transforms need square planes of the given size.");
        if (src.Overlaps(dst)) throw new ArgumentException("Source and destination must not overlap.");

        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                var (sy, sx) = SourceCoordinates(y, x, size, id);
                dst[y * size + x] = src[sy * size + sx];
            }
    }

    /// <summary>
    /// Where destination pixel (y, x) is read from. Starting from (y, x), optionally mirror horizontally,
    /// then rotate 90° (id &amp; 3) times. Each step is a bijection on the grid, so the result is a pure
    /// permutation of pixels.
    /// </summary>
    internal static (int Y, int X) SourceCoordinates(int y, int x, int size, int id)
    {
        int last = size - 1;
        int a = y, b = x;
        if (id >= 4) b = last - b;            // mirror
        for (int r = 0; r < (id & 3); r++)    // rotate 90° per step
            (a, b) = (b, last - a);
        return (a, b);
    }
}

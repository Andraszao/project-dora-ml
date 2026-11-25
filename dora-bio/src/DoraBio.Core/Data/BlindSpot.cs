namespace DoraBio.Core.Data;

/// <summary>
/// Noise2Void blind-spot masking (Krull et al., 2019).
///
/// Some pixels are chosen and each is replaced in the input by a random neighbor's value. The loss is computed
/// only at those pixels, against their original noisy values. The network can't see a masked pixel's own
/// noise, so it can't learn the identity. With pixel-independent, zero-mean noise, the best it can do is
/// predict the clean signal. This lets DORA learn from noisy images alone, with no clean ground truth.
///
/// Masked pixels are stratified (one per grid cell), so they spread evenly over the patch.
/// </summary>
public static class BlindSpot
{
    /// <summary>
    /// Writes the masked input and a 0/1 loss mask for one square patch. Returns how many pixels were masked.
    /// Pass a long-lived <see cref="Random"/>: a new mask is wanted on every training step.
    /// </summary>
    public static int Apply(
        ReadOnlySpan<float> original,
        Span<float> maskedInput,
        Span<float> lossMask,
        int size,
        float maskRatio,
        Random rng,
        int neighborhoodRadius = 2)
    {
        if (original.Length != size * size || maskedInput.Length != size * size || lossMask.Length != size * size)
            throw new ArgumentException("Planes must be size×size.");
        if (!(maskRatio > 0 && maskRatio <= 0.5f)) throw new ArgumentOutOfRangeException(nameof(maskRatio));
        if (neighborhoodRadius < 1) throw new ArgumentOutOfRangeException(nameof(neighborhoodRadius));
        if (size < 2) throw new ArgumentOutOfRangeException(nameof(size), "Patch must be at least 2×2.");

        original.CopyTo(maskedInput);
        lossMask.Clear();

        int cell = Math.Max(1, (int)Math.Round(Math.Sqrt(1.0 / maskRatio)));
        int masked = 0;

        for (int cellY = 0; cellY < size; cellY += cell)
            for (int cellX = 0; cellX < size; cellX += cell)
            {
                int y = cellY + rng.Next(Math.Min(cell, size - cellY));
                int x = cellX + rng.Next(Math.Min(cell, size - cellX));
                var (ny, nx) = RandomNeighbor(y, x, size, neighborhoodRadius, rng);

                maskedInput[y * size + x] = original[ny * size + nx];
                lossMask[y * size + x] = 1f;
                masked++;
            }

        return masked;
    }

    private static (int Y, int X) RandomNeighbor(int y, int x, int size, int radius, Random rng)
    {
        while (true)
        {
            int ny = Math.Clamp(y + rng.Next(-radius, radius + 1), 0, size - 1);
            int nx = Math.Clamp(x + rng.Next(-radius, radius + 1), 0, size - 1);
            if (ny != y || nx != x) return (ny, nx);
        }
    }
}

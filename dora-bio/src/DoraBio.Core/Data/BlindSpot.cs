using System.Buffers;

namespace DoraBio.Core.Data;

/// <summary>Shape of the neighbourhood a structured blind spot hides together with the pixel itself.</summary>
public enum FootprintShape
{
    /// <summary>All offsets with |dx| ≤ r and |dy| ≤ r: a (2r+1)² square.</summary>
    Square,

    /// <summary>All offsets with |dx| + |dy| ≤ r: a diamond (radius 2 covers the 12 nearest neighbours).</summary>
    Diamond,
}

/// <summary>
/// Which neighbours a structured blind spot hides along with its centre (structured Noise2Void, Broaddus et al. 2020).
/// The footprint should cover every offset at which the noise is noticeably correlated: on FMD widefield that is one
/// pixel across and diagonally (0.6 and 0.38) and two pixels across (0.2). Both shapes are symmetric under the
/// dihedral augmentation, so a footprint stays valid after any flip or quarter turn.
/// </summary>
/// <param name="Shape">Square or diamond.</param>
/// <param name="Radius">1 to 4 pixels.</param>
public sealed record MaskFootprint(FootprintShape Shape, int Radius)
{
    /// <summary>Every hidden offset except the centre, row by row. Square(1) has 8, Diamond(2) 12, Square(2) 24.</summary>
    public (int Dx, int Dy)[] Offsets()
    {
        if (Radius is < 1 or > 4) throw new ArgumentOutOfRangeException(nameof(Radius), "Footprint radius must be 1 to 4.");
        var offsets = new List<(int, int)>();
        for (int dy = -Radius; dy <= Radius; dy++)
            for (int dx = -Radius; dx <= Radius; dx++)
                if ((dx != 0 || dy != 0) && (Shape == FootprintShape.Square || Math.Abs(dx) + Math.Abs(dy) <= Radius))
                    offsets.Add((dx, dy));
        return offsets.ToArray();
    }

    /// <summary>"square-1", "diamond-2", …: the form <see cref="Parse"/> reads and the CLI prints.</summary>
    public override string ToString() => $"{Shape.ToString().ToLowerInvariant()}-{Radius}";

    /// <summary>Reads "square-1", "diamond-2", etc.</summary>
    /// <exception cref="ArgumentException">Not of the form shape-radius with a known shape and radius 1 to 4.</exception>
    public static MaskFootprint Parse(string text)
    {
        var parts = text.Split('-');
        if (parts.Length == 2 && Enum.TryParse<FootprintShape>(parts[0], ignoreCase: true, out var shape)
            && int.TryParse(parts[1], out int radius) && radius is >= 1 and <= 4)
            return new MaskFootprint(shape, radius);
        throw new ArgumentException($"Unknown footprint '{text}' (use square-1..4 or diamond-1..4).");
    }
}

/// <summary>
/// Noise2Void blind-spot masking (Krull et al., 2019).
///
/// Some pixels are chosen and each is replaced in the input by a random neighbor's value. The loss is computed
/// only at those pixels, against their original noisy values. The network can't see a masked pixel's own
/// noise, so it can't learn the identity. With pixel-independent, zero-mean noise, the best it can do is
/// predict the clean signal. This lets DORA learn from noisy images alone, with no clean ground truth.
///
/// Masked pixels are stratified (one per grid cell), so they spread evenly over the patch.
///
/// That argument fails if neighbouring pixels' noise is correlated: the neighbours (and the neighbour copied into the
/// hole) then tell the network part of the hidden pixel's noise, and it learns to keep it. A
/// <see cref="MaskFootprint"/> fixes this by hiding the correlated neighbours too (structured Noise2Void).
/// </summary>
public static class BlindSpot
{
    /// <summary>
    /// Writes the masked input and a 0/1 loss mask for one square patch. Returns how many pixels carry loss.
    /// Pass a long-lived <see cref="Random"/>: a new mask is wanted on every training step.
    /// </summary>
    /// <param name="original">The noisy patch, size×size.</param>
    /// <param name="maskedInput">Receives the network input: the patch with its blind spots filled in.</param>
    /// <param name="lossMask">Receives 1 at blind-spot centres (where the loss is computed) and 0 elsewhere.</param>
    /// <param name="size">Patch side length.</param>
    /// <param name="maskRatio">Fraction of pixels that are blind-spot centres (one per stratification cell).</param>
    /// <param name="rng">Source of the mask positions and replacement values.</param>
    /// <param name="neighborhoodRadius">Plain blind spots copy a random neighbour within this radius.</param>
    /// <param name="footprint">
    /// Null for a plain blind spot. Otherwise each centre and its footprint are hidden, and every hidden pixel takes the
    /// value of a random pixel that is hidden by no footprint, so no replacement comes from near any blind spot.
    /// </param>
    public static int Apply(
        ReadOnlySpan<float> original,
        Span<float> maskedInput,
        Span<float> lossMask,
        int size,
        float maskRatio,
        Random rng,
        int neighborhoodRadius = 2,
        MaskFootprint? footprint = null)
    {
        if (original.Length != size * size || maskedInput.Length != size * size || lossMask.Length != size * size)
            throw new ArgumentException("Planes must be size×size.");
        if (!(maskRatio > 0 && maskRatio <= 0.5f)) throw new ArgumentOutOfRangeException(nameof(maskRatio));
        if (neighborhoodRadius < 1) throw new ArgumentOutOfRangeException(nameof(neighborhoodRadius));
        if (size < 2) throw new ArgumentOutOfRangeException(nameof(size), "Patch must be at least 2×2.");

        if (footprint is not null) return ApplyStructured(original, maskedInput, lossMask, size, maskRatio, rng, footprint);

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

    /// <summary>
    /// Structured blind spots. Pass 1 picks the centres exactly as the plain version does (one per stratification
    /// cell). Pass 2 marks each centre and its footprint as hidden. Pass 3 fills every hidden pixel with the value of a
    /// uniformly random pixel that nothing hides: by the footprint's choice, its noise is uncorrelated with every
    /// centre's. Copying from far away (instead of from a neighbour) also keeps a centre's own noisy value from
    /// reappearing next to it when two footprints overlap.
    /// </summary>
    private static int ApplyStructured(ReadOnlySpan<float> original, Span<float> maskedInput, Span<float> lossMask,
        int size, float maskRatio, Random rng, MaskFootprint footprint)
    {
        var offsets = footprint.Offsets();
        int plane = size * size;
        original.CopyTo(maskedInput);
        lossMask.Clear();
        bool[] hidden = ArrayPool<bool>.Shared.Rent(plane);
        try
        {
            Array.Clear(hidden, 0, plane);
            int cell = Math.Max(1, (int)Math.Round(Math.Sqrt(1.0 / maskRatio)));
            int centres = 0, hiddenCount = 0;
            for (int cellY = 0; cellY < size; cellY += cell)
                for (int cellX = 0; cellX < size; cellX += cell)
                {
                    int y = cellY + rng.Next(Math.Min(cell, size - cellY));
                    int x = cellX + rng.Next(Math.Min(cell, size - cellX));
                    lossMask[y * size + x] = 1f;
                    centres++;
                    Hide(y, x);
                    foreach (var (dx, dy) in offsets)
                        if (y + dy >= 0 && y + dy < size && x + dx >= 0 && x + dx < size) Hide(y + dy, x + dx);
                }

            if (hiddenCount == plane)
                throw new ArgumentException($"Footprint {footprint} at mask ratio {maskRatio} hides the whole {size}×{size} patch.");
            for (int i = 0; i < plane; i++)
            {
                if (!hidden[i]) continue;
                int q;
                do q = rng.Next(plane); while (hidden[q]);
                maskedInput[i] = original[q];
            }
            return centres;

            void Hide(int y, int x)
            {
                if (hidden[y * size + x]) return;
                hidden[y * size + x] = true;
                hiddenCount++;
            }
        }
        finally
        {
            ArrayPool<bool>.Shared.Return(hidden);
        }
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

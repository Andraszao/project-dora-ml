using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core;

/// <summary>
/// Denoises images of any size in parallel tiles. Each tile is read with a margin equal to the network's
/// receptive radius, and only the tile's core is written back. Margins are clipped at the image edge, so
/// border pixels see the same zero padding as in whole-image inference. The result is identical to a single
/// whole-image pass, but uses bounded memory and all CPU cores.
/// </summary>
/// <remarks>
/// Why this is exact: an output pixel depends only on input pixels within <see cref="NetConfig.ReceptiveRadius"/>.
/// A core pixel's whole neighborhood lies inside its window, so it is computed from exactly the same
/// values as in a whole-image pass. Pixels near a window's inner edge are wrong (they see artificial zero
/// padding), but they are in the margin and get discarded. At the true image border the window is clipped,
/// so the padding there is the same padding the whole-image pass would use.
/// <c>TiledInferenceMatchesWholeImage</c> pins this down.
///
/// <para>A U-Net needs one more rule. Pooling makes it shift-equivariant only for shifts that are multiples of
/// 2^Depth (<see cref="NetConfig.TileAlignment"/>), so every core and every window starts on that grid: the core size
/// and the margin are rounded up to multiples of it. A window clipped at the right or bottom image edge then gets
/// zero-padded inside <see cref="DenoisingNet.Predict"/> to exactly the edge a whole-image pass pads to, because its
/// start is on the grid. <c>TiledUNetInferenceMatchesWholeImage</c> pins this down.</para>
/// </remarks>
public static class TiledInference
{
    /// <summary>
    /// Default core size in pixels. With the default 22 px margin each window is up to 172², which balances
    /// overhead (margins are recomputed by neighboring tiles) against memory and parallelism.
    /// </summary>
    public const int DefaultTileCore = 128;

    /// <summary>Denoises a normalized image, tile by tile, on all cores.</summary>
    /// <param name="net">The trained network. Only read, so concurrent tiles can share it.</param>
    /// <param name="input">Normalized noisy image (any size).</param>
    /// <param name="tileCore">Side length of the region each tile writes (at least 8).</param>
    /// <param name="cancellation">Stops scheduling further tiles. Throws <see cref="OperationCanceledException"/>.</param>
    /// <returns>The denoised image, same size and scale as <paramref name="input"/>.</returns>
    public static GrayImage Denoise(DenoisingNet net, GrayImage input, int tileCore = DefaultTileCore, CancellationToken cancellation = default)
    {
        if (tileCore < 8) throw new ArgumentOutOfRangeException(nameof(tileCore));
        int align = net.Config.TileAlignment;
        tileCore = RoundUp(tileCore, align);
        int w = input.Width, h = input.Height, margin = RoundUp(net.Config.ReceptiveRadius, align);
        var output = new GrayImage(w, h);

        // Cores tile the image without overlap; the last row and column of cores may be smaller.
        var tiles = new List<(int X0, int Y0, int X1, int Y1)>();
        for (int y = 0; y < h; y += tileCore)
            for (int x = 0; x < w; x += tileCore)
                tiles.Add((x, y, Math.Min(w, x + tileCore), Math.Min(h, y + tileCore)));

        var pool = new WorkspacePool(net);
        var options = new ParallelOptions { CancellationToken = cancellation };
        Parallel.For(0, tiles.Count, options, () => pool.Rent(), (t, _, workspace) =>
        {
            // Window = core grown by the margin on every side, clipped to the image.
            var (cx0, cy0, cx1, cy1) = tiles[t];
            int wx0 = Math.Max(0, cx0 - margin), wy0 = Math.Max(0, cy0 - margin);
            int wx1 = Math.Min(w, cx1 + margin), wy1 = Math.Min(h, cy1 + margin);
            int ww = wx1 - wx0, wh = wy1 - wy0;

            var window = new float[ww * wh];
            for (int row = 0; row < wh; row++)
                input.Pixels.AsSpan((wy0 + row) * w + wx0, ww).CopyTo(window.AsSpan(row * ww, ww));

            var denoised = new float[ww * wh];
            net.Predict(window, wh, ww, denoised, workspace);

            // Keep only the core. Each tile writes a disjoint region, so no locking is needed.
            for (int y = cy0; y < cy1; y++)
                denoised.AsSpan((y - wy0) * ww + (cx0 - wx0), cx1 - cx0).CopyTo(output.Pixels.AsSpan(y * w + cx0, cx1 - cx0));
            return workspace;
        }, pool.Return);

        return output;
    }

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;
}

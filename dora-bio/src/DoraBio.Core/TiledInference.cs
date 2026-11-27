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
    public static GrayImage Denoise(DenoisingNet net, GrayImage input, int tileCore = DefaultTileCore, CancellationToken cancellation = default) =>
        Denoise(net, input, tileCore, cancellation, new WorkspacePool(net));

    /// <summary>
    /// Denoises the centre frame of a multi-frame input (brick 39) with a network of <see cref="NetConfig.InputFrames"/>
    /// frames. Every tile takes the same window from every frame, so the spatial margin covers every frame channel and
    /// tiling stays exact.
    /// </summary>
    /// <param name="net">A network with as many input frames as <paramref name="frames"/> holds.</param>
    /// <param name="frames">Normalized frames in time order, all the same size; the centre one is denoised.</param>
    /// <param name="tileCore">Side length of the region each tile writes (at least 8).</param>
    /// <param name="cancellation">Stops scheduling further tiles. Throws <see cref="OperationCanceledException"/>.</param>
    public static GrayImage Denoise(DenoisingNet net, IReadOnlyList<GrayImage> frames, int tileCore = DefaultTileCore,
        CancellationToken cancellation = default) =>
        Denoise(net, frames, tileCore, cancellation, new WorkspacePool(net), float.NaN);

    /// <summary>
    /// As the public overload, with workspaces taken from (and returned to) <paramref name="pool"/>, so a caller that
    /// denoises many frames (<see cref="FrameDenoiser"/>) allocates the tens of MB of activations per thread once,
    /// not once per frame.
    /// </summary>
    internal static GrayImage Denoise(DenoisingNet net, GrayImage input, int tileCore, CancellationToken cancellation, WorkspacePool pool) =>
        Denoise(net, input, tileCore, cancellation, pool, net.Config.IsConditioned ? NoiseConditioning.Of(input) : float.NaN);

    /// <summary>Multi-frame inference with workspaces from <paramref name="pool"/> (see the public overload).</summary>
    internal static GrayImage Denoise(DenoisingNet net, IReadOnlyList<GrayImage> frames, int tileCore, CancellationToken cancellation,
        WorkspacePool pool) => Denoise(net, frames, tileCore, cancellation, pool, float.NaN);

    /// <summary>
    /// As above with an explicit condition for a noise-conditioned network. Normally it is measured on the whole
    /// <paramref name="input"/>; evaluations pass a deliberately wrong one to test whether the network uses it (Q13).
    /// </summary>
    internal static GrayImage Denoise(DenoisingNet net, GrayImage input, int tileCore, CancellationToken cancellation,
        WorkspacePool pool, float condition) => Denoise(net, [input], tileCore, cancellation, pool, condition);

    private static GrayImage Denoise(DenoisingNet net, IReadOnlyList<GrayImage> frames, int tileCore, CancellationToken cancellation,
        WorkspacePool pool, float condition)
    {
        if (tileCore < 8) throw new ArgumentOutOfRangeException(nameof(tileCore));
        if (frames.Count != net.Config.InputFrames)
            throw new ArgumentException(net.Config.InputFrames == 1
                ? "This network denoises single frames."
                : $"This network needs {net.Config.InputFrames} frames (the frame and its neighbours), got {frames.Count}.", nameof(frames));
        var input = frames[net.Config.CenterFrame];
        if (frames.Any(f => f.Width != input.Width || f.Height != input.Height))
            throw new ArgumentException("Every frame must have the same size.", nameof(frames));
        int align = net.Config.TileAlignment;
        tileCore = RoundUp(tileCore, align);
        int w = input.Width, h = input.Height, margin = RoundUp(net.Config.ReceptiveRadius, align);
        var output = new GrayImage(w, h);

        // Cores tile the image without overlap; the last row and column of cores may be smaller.
        var tiles = new List<(int X0, int Y0, int X1, int Y1)>();
        for (int y = 0; y < h; y += tileCore)
            for (int x = 0; x < w; x += tileCore)
                tiles.Add((x, y, Math.Min(w, x + tileCore), Math.Min(h, y + tileCore)));

        // Every workspace is sized once for the largest window, so tiles of different sizes (edges are smaller)
        // never make it reallocate.
        int maxWindowW = Math.Min(w, tileCore + 2 * margin), maxWindowH = Math.Min(h, tileCore + 2 * margin);
        // At most one task per core: Parallel.For would otherwise let the thread pool add workers over a long run,
        // and each new worker rents (and the pool then keeps) another workspace of tens of MB.
        var options = new ParallelOptions { CancellationToken = cancellation, MaxDegreeOfParallelism = Environment.ProcessorCount };
        Parallel.For(0, tiles.Count, options, () =>
        {
            var workspace = pool.Rent();
            workspace.Reserve(maxWindowH, maxWindowW);
            return workspace;
        }, (t, _, workspace) =>
        {
            // Window = core grown by the margin on every side, clipped to the image.
            var (cx0, cy0, cx1, cy1) = tiles[t];
            int wx0 = Math.Max(0, cx0 - margin), wy0 = Math.Max(0, cy0 - margin);
            int wx1 = Math.Min(w, cx1 + margin), wy1 = Math.Min(h, cy1 + margin);
            int ww = wx1 - wx0, wh = wy1 - wy0;

            var (windowBuffer, denoisedBuffer) = workspace.TileBuffers(frames.Count * maxWindowW * maxWindowH);
            var window = windowBuffer.AsSpan(0, frames.Count * ww * wh);
            for (int f = 0; f < frames.Count; f++)   // the same window from every frame, one plane after another
                for (int row = 0; row < wh; row++)
                    frames[f].Pixels.AsSpan((wy0 + row) * w + wx0, ww).CopyTo(window.Slice(f * ww * wh + row * ww, ww));

            var denoised = denoisedBuffer.AsSpan(0, ww * wh);
            // A conditioned network gets the whole frame's condition in every tile, so tiles still agree with a
            // whole-image pass (a condition measured per tile would differ from tile to tile).
            net.Predict(window, wh, ww, denoised, workspace, condition);

            // Keep only the core. Each tile writes a disjoint region, so no locking is needed.
            for (int y = cy0; y < cy1; y++)
                denoised.Slice((y - wy0) * ww + (cx0 - wx0), cx1 - cx0).CopyTo(output.Pixels.AsSpan(y * w + cx0, cx1 - cx0));
            return workspace;
        }, pool.Return);

        return output;
    }

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;
}

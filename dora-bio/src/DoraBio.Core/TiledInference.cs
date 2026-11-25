using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core;

/// <summary>
/// Denoises images of any size in parallel tiles. Each tile is read with a margin equal to the network's
/// receptive radius, and only the tile's core is written back. Margins are clipped at the image edge, so
/// border pixels see the same zero padding as in whole-image inference. The result is identical to a single
/// whole-image pass, but uses bounded memory and all CPU cores.
/// </summary>
public static class TiledInference
{
    public const int DefaultTileCore = 128;

    public static GrayImage Denoise(DenoisingNet net, GrayImage input, int tileCore = DefaultTileCore, CancellationToken cancellation = default)
    {
        if (tileCore < 8) throw new ArgumentOutOfRangeException(nameof(tileCore));
        int w = input.Width, h = input.Height, margin = net.Config.ReceptiveRadius;
        var output = new GrayImage(w, h);

        var tiles = new List<(int X0, int Y0, int X1, int Y1)>();
        for (int y = 0; y < h; y += tileCore)
            for (int x = 0; x < w; x += tileCore)
                tiles.Add((x, y, Math.Min(w, x + tileCore), Math.Min(h, y + tileCore)));

        var pool = new WorkspacePool(net);
        var options = new ParallelOptions { CancellationToken = cancellation };
        Parallel.For(0, tiles.Count, options, () => pool.Rent(), (t, _, workspace) =>
        {
            var (cx0, cy0, cx1, cy1) = tiles[t];
            int wx0 = Math.Max(0, cx0 - margin), wy0 = Math.Max(0, cy0 - margin);
            int wx1 = Math.Min(w, cx1 + margin), wy1 = Math.Min(h, cy1 + margin);
            int ww = wx1 - wx0, wh = wy1 - wy0;

            var window = new float[ww * wh];
            for (int row = 0; row < wh; row++)
                input.Pixels.AsSpan((wy0 + row) * w + wx0, ww).CopyTo(window.AsSpan(row * ww, ww));

            var denoised = new float[ww * wh];
            net.Predict(window, wh, ww, denoised, workspace);

            for (int y = cy0; y < cy1; y++)
                denoised.AsSpan((y - wy0) * ww + (cx0 - wx0), cx1 - cx0).CopyTo(output.Pixels.AsSpan(y * w + cx0, cx1 - cx0));
            return workspace;
        }, pool.Return);

        return output;
    }
}

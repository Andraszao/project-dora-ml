using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>
/// How correlated an image's noise looks between neighbouring pixels, estimated from the noisy image alone
/// (see <see cref="NoiseDiagnostics.Diagnose"/>). Correlations are Pearson coefficients: 0 means independent noise,
/// as Noise2Void assumes, and 1 means neighbours carry identical noise.
/// </summary>
/// <param name="Right1">Estimated correlation with the pixel one to the right.</param>
/// <param name="Down1">Estimated correlation with the pixel one below.</param>
/// <param name="Diagonal1">Estimated correlation with the pixel one right and one down.</param>
/// <param name="Right2">Estimated correlation with the pixel two to the right (underestimated: see remarks on Diagnose).</param>
/// <param name="Down2">Estimated correlation with the pixel two below (underestimated likewise).</param>
/// <param name="TilesUsed">How many flat tiles the estimate is the median over.</param>
/// <param name="Correlated">True when Right1 or Down1 reaches <see cref="NoiseDiagnostics.CorrelatedThreshold"/>.</param>
public sealed record NoiseDiagnosis(double Right1, double Down1, double Diagonal1, double Right2, double Down2,
    int TilesUsed, bool Correlated)
{
    /// <summary>One-line, human-readable summary for logs and the UI.</summary>
    public override string ToString() => Correlated
        ? $"noise looks CORRELATED between neighbouring pixels (right {Right1:0.00}, down {Down1:0.00}): plain Noise2Void " +
          "will keep part of it. Prefer supervised training, or structured masking."
        : $"noise looks independent between pixels (right {Right1:0.00}, down {Down1:0.00}): fine for Noise2Void.";
}

/// <summary>
/// Checks Noise2Void's key assumption, pixel-independent noise, on a single noisy image with no ground truth.
///
/// Method: cut the image into 16×16 tiles and remove each tile's best-fit plane (mean and linear slope), so slow
/// intensity gradients don't count as correlation. Signal (edges, cells) adds strong positive correlation, so keep
/// only the flattest quarter of the tiles (lowest residual variance). In fluorescence images those are mostly
/// background, where what's left is noise. Measure each kept tile's correlation between pixel pairs at a given
/// offset, and report the median over tiles, which shrugs off the few tiles that still contain signal.
/// </summary>
/// <remarks>
/// Checked on FMD's 48 test images against the ground-truth measurement (<see cref="FmdBenchmark.NoiseCorrelation"/>):
/// widefield images estimate 0.56–0.58 at one pixel (truth 0.58–0.65), and every confocal and two-photon image
/// estimates 0.26 or less (truth 0.24 or less). The threshold of 0.4 sits in the gap. Small tiles keep signal out,
/// but they bias the estimate low, increasingly with distance: at two pixels the widefield estimate is about half the
/// truth (0.10 vs 0.20). Larger tiles reduce the bias but let dense structure in (32 px tiles read 0.41 on a densely
/// stained confocal sample), which makes the verdict unreliable.
/// </remarks>
public static class NoiseDiagnostics
{
    /// <summary>Tile side length in pixels.</summary>
    public const int TileSize = 16;

    /// <summary>One-pixel correlation at or above which noise counts as correlated.</summary>
    public const double CorrelatedThreshold = 0.4;

    /// <summary>Fewest usable tiles a diagnosis needs (a 64×64 image has 16).</summary>
    public const int MinTiles = 4;

    private static readonly (int Dx, int Dy)[] Offsets = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)];

    /// <summary>Estimates the noise correlation of one noisy image. Works in any intensity units.</summary>
    /// <param name="image">A noisy image at least 32×32 pixels (more is better: a 512×512 frame gives 1,024 tiles).</param>
    /// <exception cref="ArgumentException">Too few non-constant tiles to estimate anything.</exception>
    public static NoiseDiagnosis Diagnose(GrayImage image)
    {
        int tilesX = image.Width / TileSize, tilesY = image.Height / TileSize;
        var residuals = new List<(double Variance, int Index, double[] Pixels)>();
        for (int ty = 0; ty < tilesY; ty++)
            for (int tx = 0; tx < tilesX; tx++)
            {
                var r = PlaneResidual(image, tx * TileSize, ty * TileSize);
                double variance = 0;
                foreach (double v in r) variance += v * v;
                variance /= r.Length;
                if (variance > 1e-12) residuals.Add((variance, residuals.Count, r));   // skip constant (e.g. clipped) tiles
            }
        if (residuals.Count < MinTiles)
            throw new ArgumentException($"Need at least {MinTiles} non-constant {TileSize}×{TileSize} tiles to judge the noise; " +
                $"this {image.Width}×{image.Height} image has {residuals.Count}.");

        // Flattest quarter, ties broken by position so the result never depends on sort stability.
        var flat = residuals.OrderBy(t => t.Variance).ThenBy(t => t.Index)
            .Take(Math.Max(MinTiles, residuals.Count / 4)).ToList();
        var medians = Offsets.Select(o => Median(flat.Select(t => TileCorrelation(t.Pixels, o.Dx, o.Dy)).ToList())).ToArray();
        return new NoiseDiagnosis(medians[0], medians[1], medians[2], medians[3], medians[4], flat.Count,
            Math.Max(medians[0], medians[1]) >= CorrelatedThreshold);
    }

    /// <summary>
    /// The tile minus its least-squares plane a + b·x + c·y. On a square grid with centred coordinates the three
    /// basis functions are orthogonal, so each coefficient is a separate projection: a = mean, b = Σv·x / Σx², c likewise.
    /// </summary>
    internal static double[] PlaneResidual(GrayImage image, int x0, int y0)
    {
        const double c = (TileSize - 1) / 2.0;
        var r = new double[TileSize * TileSize];
        double mean = 0, sx = 0, sy = 0, xx = 0;
        for (int y = 0; y < TileSize; y++)
            for (int x = 0; x < TileSize; x++)
            {
                double v = image[x0 + x, y0 + y];
                r[y * TileSize + x] = v;
                mean += v;
                sx += v * (x - c);
                sy += v * (y - c);
            }
        for (int i = 0; i < TileSize; i++) xx += (i - c) * (i - c);
        xx *= TileSize;   // Σ over the whole tile of (x - c)²; the same for y
        mean /= r.Length;
        double b = sx / xx, d = sy / xx;
        for (int y = 0; y < TileSize; y++)
            for (int x = 0; x < TileSize; x++)
                r[y * TileSize + x] -= mean + b * (x - c) + d * (y - c);
        return r;
    }

    /// <summary>Pearson correlation of the residual with itself shifted by (dx, dy), over the pairs inside the tile.</summary>
    private static double TileCorrelation(double[] r, int dx, int dy)
    {
        double cross = 0, a2 = 0, b2 = 0;
        for (int y = 0; y + dy < TileSize; y++)
            for (int x = 0; x + dx < TileSize; x++)
            {
                double a = r[y * TileSize + x], b = r[(y + dy) * TileSize + x + dx];
                cross += a * b;
                a2 += a * a;
                b2 += b * b;
            }
        return a2 > 0 && b2 > 0 ? cross / Math.Sqrt(a2 * b2) : 0;
    }

    private static double Median(List<double> values)
    {
        values.Sort();
        int n = values.Count;
        return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    }
}

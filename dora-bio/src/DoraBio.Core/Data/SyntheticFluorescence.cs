namespace DoraBio.Core.Data;

/// <summary>Knobs for <see cref="SyntheticFluorescence"/>. Photon counts are per pixel at the brightest structure.</summary>
public sealed record SyntheticOptions
{
    public double PeakPhotons { get; init; } = 25;
    public double ReadNoiseAdu { get; init; } = 2.0;
    public double CameraOffsetAdu { get; init; } = 100;
    public double GainAduPerPhoton { get; init; } = 1.0;
    public double Autofluorescence { get; init; } = 0.08;
    public int Cells { get; init; } = 6;
    public int Filaments { get; init; } = 10;
    public int Puncta { get; init; } = 40;
}

/// <summary>
/// Procedural fluorescence-like scenes with physically motivated Poisson–Gaussian noise
/// (photon shot noise plus camera read noise), for tests, demos and smoke-training.
///
/// This is SYNTHETIC data. It looks loosely like widefield images of cells (soft cell bodies, bright
/// puncta, thin filaments, a smooth autofluorescence background), but it is not a substitute for real
/// microscope data. Weights trained on it should be labeled as such.
/// </summary>
public static class SyntheticFluorescence
{
    /// <summary>A noisy 16-bit-range frame and its noise-free expectation, both in camera units (ADU).</summary>
    public sealed record Frame(GrayImage Noisy, GrayImage Clean);

    public static Frame Generate(int width, int height, int seed, SyntheticOptions? options = null)
    {
        var o = options ?? new SyntheticOptions();
        var rng = new Random(seed);
        var signal = RenderSignal(width, height, rng, o);

        var noisy = new GrayImage(width, height);
        var clean = new GrayImage(width, height);
        for (int i = 0; i < signal.Length; i++)
        {
            double lambda = Math.Max(0, signal[i]) * o.PeakPhotons;
            double photons = SamplePoisson(lambda, rng);
            double adu = o.CameraOffsetAdu + o.GainAduPerPhoton * photons + o.ReadNoiseAdu * SampleGaussian(rng);
            noisy.Pixels[i] = (float)Math.Clamp(Math.Round(adu), 0, ushort.MaxValue);   // what a 16-bit camera records
            clean.Pixels[i] = (float)(o.CameraOffsetAdu + o.GainAduPerPhoton * lambda);
        }
        return new Frame(noisy, clean);
    }

    /// <summary>Generates <paramref name="count"/> independent FOVs as normalized <see cref="ImagePair"/>s.</summary>
    public static List<ImagePair> GenerateDataset(int count, int size, int seed, SyntheticOptions? options = null)
    {
        var result = new List<ImagePair>(count);
        for (int i = 0; i < count; i++)
        {
            var frame = Generate(size, size, seed + 7919 * i, options);
            var (pair, _) = new ImagePair(frame.Noisy, frame.Clean, $"synthetic-{i:D3}", $"synthetic-{i:D3}").Normalize();
            result.Add(pair);
        }
        return result;
    }

    /// <summary>Noise-free intensity field, roughly in [0, 1].</summary>
    private static double[] RenderSignal(int width, int height, Random rng, SyntheticOptions o)
    {
        var field = new double[width * height];
        double scale = Math.Min(width, height);

        // Smooth autofluorescence: a constant floor plus a few very wide blobs.
        for (int i = 0; i < field.Length; i++) field[i] = o.Autofluorescence * 0.5;
        for (int k = 0; k < 3; k++)
            AddGaussian(field, width, height, rng.NextDouble() * width, rng.NextDouble() * height,
                        scale * (0.25 + 0.25 * rng.NextDouble()), o.Autofluorescence);

        for (int c = 0; c < o.Cells; c++) AddCell(field, width, height, rng, scale);
        for (int f = 0; f < o.Filaments; f++) AddFilament(field, width, height, rng, scale);
        for (int p = 0; p < o.Puncta; p++)
            AddGaussian(field, width, height, rng.NextDouble() * width, rng.NextDouble() * height,
                        0.8 + 0.7 * rng.NextDouble(), 0.5 + 0.5 * rng.NextDouble());

        for (int i = 0; i < field.Length; i++) field[i] = Math.Min(field[i], 1.2);
        return field;
    }

    private static void AddCell(double[] field, int width, int height, Random rng, double scale)
    {
        double cx = rng.NextDouble() * width, cy = rng.NextDouble() * height;
        double rx = scale * (0.06 + 0.09 * rng.NextDouble());
        double ry = rx * (0.6 + 0.6 * rng.NextDouble());
        double angle = rng.NextDouble() * Math.PI;
        double cos = Math.Cos(angle), sin = Math.Sin(angle);
        double body = 0.25 + 0.15 * rng.NextDouble();
        double reach = Math.Max(rx, ry) * 1.4;

        ForEachPixelNear(width, height, cx, cy, reach, (x, y, i) =>
        {
            double dx = x - cx, dy = y - cy;
            double u = (dx * cos + dy * sin) / rx, v = (-dx * sin + dy * cos) / ry;
            double r = Math.Sqrt(u * u + v * v);
            double membrane = 1.0 / (1.0 + Math.Exp((r - 1.0) / 0.08));   // soft-edged body
            double nucleus = Math.Exp(-r * r / (2 * 0.22 * 0.22));         // brighter core
            field[i] += body * membrane + 0.3 * nucleus;
        });
    }

    private static void AddFilament(double[] field, int width, int height, Random rng, double scale)
    {
        double x = rng.NextDouble() * width, y = rng.NextDouble() * height;
        double heading = rng.NextDouble() * 2 * Math.PI;
        double amplitude = 0.35 + 0.35 * rng.NextDouble();
        int steps = (int)(scale * (0.3 + 0.5 * rng.NextDouble()));
        const double sigma = 0.8;

        for (int s = 0; s < steps; s++)
        {
            ForEachPixelNear(width, height, x, y, 3 * sigma, (px, py, i) =>
            {
                double d2 = (px - x) * (px - x) + (py - y) * (py - y);
                field[i] = Math.Max(field[i], field[i] * 0.5 + amplitude * Math.Exp(-d2 / (2 * sigma * sigma)));
            });
            heading += 0.15 * SampleGaussian(rng);
            x += Math.Cos(heading);
            y += Math.Sin(heading);
        }
    }

    private static void AddGaussian(double[] field, int width, int height, double cx, double cy, double sigma, double amplitude) =>
        ForEachPixelNear(width, height, cx, cy, 3.5 * sigma, (x, y, i) =>
        {
            double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            field[i] += amplitude * Math.Exp(-d2 / (2 * sigma * sigma));
        });

    private static void ForEachPixelNear(int width, int height, double cx, double cy, double radius, Action<int, int, int> visit)
    {
        int x0 = Math.Max(0, (int)Math.Floor(cx - radius)), x1 = Math.Min(width - 1, (int)Math.Ceiling(cx + radius));
        int y0 = Math.Max(0, (int)Math.Floor(cy - radius)), y1 = Math.Min(height - 1, (int)Math.Ceiling(cy + radius));
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
                visit(x, y, y * width + x);
    }

    internal static double SampleGaussian(Random rng)
    {
        // Box–Muller; 1 − U avoids log(0).
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    internal static double SamplePoisson(double lambda, Random rng)
    {
        if (lambda <= 0) return 0;
        if (lambda > 60)   // normal approximation is accurate here and avoids long loops
            return Math.Max(0, Math.Round(lambda + Math.Sqrt(lambda) * SampleGaussian(rng)));

        // Knuth's multiplication method.
        double limit = Math.Exp(-lambda), product = 1.0;
        int k = 0;
        do { k++; product *= rng.NextDouble(); } while (product > limit);
        return k - 1;
    }
}

using DoraBio.Core.Data;
using DoraBio.Core.IO;

namespace DoraBio.Core;

/// <summary>
/// What DORA removed, judged without ground truth (brick 41, from the wild-data check): the residual r = raw − denoised,
/// pooled over every plane as they stream past, in the input's own units. Removed noise is zero-mean, unrelated to the
/// image and, for independent noise, unrelated between neighbouring pixels; removed signal is none of these.
/// </summary>
public sealed class ResidualCheck
{
    private long _n;
    private double _r, _d, _rd, _rr, _dd, _raw;
    private readonly double[] _lagX = new double[3], _lagY = new double[3];   // Σ a·b, Σ a², Σ b² of the mean-removed residual
    private double _rmsShareSum;
    private int _planes;

    /// <summary>Adds one plane and its denoised output (same size).</summary>
    /// <exception cref="ArgumentException">The sizes differ.</exception>
    public void Add(GrayImage raw, GrayImage denoised)
    {
        if (raw.Width != denoised.Width || raw.Height != denoised.Height)
            throw new ArgumentException("A plane and its output must have the same size.");
        int w = raw.Width, h = raw.Height, n = raw.Length;
        var r = new float[n];
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            double ri = raw.Pixels[i] - denoised.Pixels[i], di = denoised.Pixels[i];
            r[i] = (float)ri;
            mean += ri;
            _r += ri; _d += di; _rd += ri * di; _rr += ri * ri; _dd += di * di; _raw += raw.Pixels[i];
        }
        _n += n;
        mean /= n;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double a = r[y * w + x] - mean;
                if (x + 1 < w) Accumulate(_lagX, a, r[y * w + x + 1] - mean);
                if (y + 1 < h) Accumulate(_lagY, a, r[(y + 1) * w + x] - mean);
            }
        var sorted = (float[])raw.Pixels.Clone();
        Array.Sort(sorted);
        double range = Normalization.Quantile(sorted, 0.999) - Normalization.Quantile(sorted, 0.001);
        double rms = Math.Sqrt(r.Sum(v => (double)v * v) / n);
        _rmsShareSum += rms / Math.Max(range, 1e-12);
        _planes++;
    }

    private static void Accumulate(double[] lag, double a, double b)
    {
        lag[0] += a * b;
        lag[1] += a * a;
        lag[2] += b * b;
    }

    /// <summary>White columns between the panels of <see cref="Picture"/> (the report lays its labels out by it).</summary>
    public const int PictureGap = 4;

    /// <summary>Planes added.</summary>
    public int Planes => _planes;

    /// <summary>
    /// Signed correlation of the residual with the output: about 0 when only zero-mean noise was removed; positive when
    /// bright structure was shrunk (the residual carries part of it); negative when the output overshoots it.
    /// </summary>
    public double ResidualOutputCorrelation
    {
        get
        {
            double cov = _rd / _n - (_r / _n) * (_d / _n), vr = _rr / _n - Math.Pow(_r / _n, 2), vd = _dd / _n - Math.Pow(_d / _n, 2);
            return cov / Math.Sqrt(Math.Max(vr * vd, 1e-300));
        }
    }

    /// <summary>Lag-1 correlation of the residual along rows (x): near 0 for removed independent noise.</summary>
    public double LagX => _lagX[0] / Math.Sqrt(Math.Max(_lagX[1] * _lagX[2], 1e-300));

    /// <summary>Lag-1 correlation of the residual down columns (y).</summary>
    public double LagY => _lagY[0] / Math.Sqrt(Math.Max(_lagY[1] * _lagY[2], 1e-300));

    /// <summary>The net brightness change, mean(residual) / mean(raw): positive means the output is darker.</summary>
    public double NetChange => _raw == 0 ? 0 : _r / _raw;

    /// <summary>RMS of the residual as a share of each plane's 0.1–99.9 percentile range, averaged over planes.</summary>
    public double RmsChange => _planes == 0 ? 0 : _rmsShareSum / _planes;

    /// <summary>
    /// Raw | denoised | residual side by side as an 8-bit grayscale PNG, for the report. Raw and denoised share the raw
    /// plane's 0.1–99.9 percentile window; the residual is centred on zero (mid-gray), positive brighter, negative darker,
    /// clipped at the 99th percentile of |residual|, so one hot pixel can't flatten it to gray.
    /// </summary>
    public static byte[] Picture(GrayImage raw, GrayImage denoised)
    {
        int w = raw.Width, h = raw.Height, gap = PictureGap, total = 3 * w + 2 * gap;
        var sorted = (float[])raw.Pixels.Clone();
        Array.Sort(sorted);
        double lo = Normalization.Quantile(sorted, 0.001), hi = Normalization.Quantile(sorted, 0.999), span = Math.Max(hi - lo, 1e-12);
        var magnitude = new float[raw.Length];
        for (int i = 0; i < raw.Length; i++) magnitude[i] = Math.Abs(raw.Pixels[i] - denoised.Pixels[i]);
        Array.Sort(magnitude);
        double bound = Math.Max(Normalization.Quantile(magnitude, 0.99), 1e-12);
        var pixels = new byte[total * h];
        Array.Fill(pixels, (byte)255);
        static byte Byte(double v) => (byte)Math.Round(Math.Clamp(v, 0, 1) * 255);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x, row = y * total;
                pixels[row + x] = Byte((raw.Pixels[i] - lo) / span);
                pixels[row + w + gap + x] = Byte((denoised.Pixels[i] - lo) / span);
                pixels[row + 2 * (w + gap) + x] = Byte(0.5 + (raw.Pixels[i] - denoised.Pixels[i]) / (2 * bound));
            }
        return PngEncoder.EncodeGray8(pixels, total, h);
    }
}

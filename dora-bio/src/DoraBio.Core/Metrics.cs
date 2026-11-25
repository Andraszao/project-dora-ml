using DoraBio.Core.Data;

namespace DoraBio.Core;

/// <summary>Image-quality metrics. All inputs are assumed to be on the same (normalized) scale.</summary>
public static class Metrics
{
    public static double Mse(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length || a.IsEmpty) throw new ArgumentException("Inputs must be non-empty and equal length.");
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - (double)b[i];
            sum += d * d;
        }
        return sum / a.Length;
    }

    /// <summary>Peak signal-to-noise ratio in dB. Identical inputs give +∞.</summary>
    public static double Psnr(ReadOnlySpan<float> prediction, ReadOnlySpan<float> reference, double dataRange = 1.0) =>
        PsnrFromMse(Mse(prediction, reference), dataRange);

    public static double PsnrFromMse(double mse, double dataRange = 1.0) =>
        mse <= 0 ? double.PositiveInfinity : 10.0 * Math.Log10(dataRange * dataRange / mse);

    /// <summary>
    /// Mean structural similarity (Wang et al., 2004) with the standard 11×11 Gaussian window (σ = 1.5),
    /// K1 = 0.01 and K2 = 0.03. Only window positions that lie fully inside the image are used.
    /// Images smaller than 11 px use the largest odd window that fits.
    /// </summary>
    public static double Ssim(GrayImage a, GrayImage b, double dataRange = 1.0)
    {
        if (a.Width != b.Width || a.Height != b.Height) throw new ArgumentException("Images must be the same size.");
        int window = Math.Min(11, Math.Min(a.Width, a.Height));
        if (window % 2 == 0) window--;
        if (window < 3) throw new ArgumentException("Images must be at least 3×3 for SSIM.");

        double[] kernel = GaussianKernel(window, 1.5);
        double c1 = Math.Pow(0.01 * dataRange, 2), c2 = Math.Pow(0.03 * dataRange, 2);
        int w = a.Width, h = a.Height;

        double[] x = Array.ConvertAll(a.Pixels, v => (double)v);
        double[] y = Array.ConvertAll(b.Pixels, v => (double)v);
        double[] xx = new double[x.Length], yy = new double[x.Length], xy = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            xx[i] = x[i] * x[i];
            yy[i] = y[i] * y[i];
            xy[i] = x[i] * y[i];
        }

        var muX = FilterValid(x, w, h, kernel, out int ow, out int oh);
        var muY = FilterValid(y, w, h, kernel, out _, out _);
        var sXX = FilterValid(xx, w, h, kernel, out _, out _);
        var sYY = FilterValid(yy, w, h, kernel, out _, out _);
        var sXY = FilterValid(xy, w, h, kernel, out _, out _);

        double total = 0;
        for (int i = 0; i < ow * oh; i++)
        {
            double mx = muX[i], my = muY[i];
            double vx = sXX[i] - mx * mx, vy = sYY[i] - my * my, cov = sXY[i] - mx * my;
            total += (2 * mx * my + c1) * (2 * cov + c2) / ((mx * mx + my * my + c1) * (vx + vy + c2));
        }
        return total / (ow * oh);
    }

    /// <summary>
    /// Estimates the standard deviation of additive white noise without a clean reference (Immerkær, 1996).
    /// It convolves the image with a Laplacian-difference mask that cancels smooth structure.
    /// Real fluorescence noise depends on the signal (Poisson), so treat this as a rough image-wide figure,
    /// good for comparing before and after denoising, not as a calibrated measurement.
    /// </summary>
    public static double EstimateNoiseSigma(GrayImage image)
    {
        int w = image.Width, h = image.Height;
        if (w < 3 || h < 3) throw new ArgumentException("Image must be at least 3×3.");
        var p = image.Pixels;
        double sum = 0;
        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                int i = y * w + x;
                double response =
                      p[i - w - 1] - 2 * p[i - w] + p[i - w + 1]
                    - 2 * p[i - 1] + 4 * p[i] - 2 * p[i + 1]
                    + p[i + w - 1] - 2 * p[i + w] + p[i + w + 1];
                sum += Math.Abs(response);
            }
        return Math.Sqrt(Math.PI / 2) * sum / (6.0 * (w - 2) * (h - 2));
    }

    private static double[] GaussianKernel(int size, double sigma)
    {
        var k = new double[size];
        int r = size / 2;
        double total = 0;
        for (int i = 0; i < size; i++)
        {
            k[i] = Math.Exp(-((i - r) * (i - r)) / (2 * sigma * sigma));
            total += k[i];
        }
        for (int i = 0; i < size; i++) k[i] /= total;
        return k;
    }

    /// <summary>Separable filtering over "valid" positions only (no padding).</summary>
    private static double[] FilterValid(double[] src, int w, int h, double[] kernel, out int outW, out int outH)
    {
        int n = kernel.Length;
        outW = w - n + 1;
        outH = h - n + 1;
        var horizontal = new double[outW * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < outW; x++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += src[y * w + x + k] * kernel[k];
                horizontal[y * outW + x] = s;
            }
        var result = new double[outW * outH];
        for (int y = 0; y < outH; y++)
            for (int x = 0; x < outW; x++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += horizontal[(y + k) * outW + x] * kernel[k];
                result[y * outW + x] = s;
            }
        return result;
    }
}

using System.Numerics;
using System.Runtime.InteropServices;

namespace DoraBio.Core.Nn;

/// <summary>
/// Vectorized row kernels used by the convolution layer.
/// The vector and scalar paths do the same multiply-then-add per element (.NET does not fuse them),
/// so a value gets the same result whichever path processes it. Tiled inference relies on this to
/// match full-image inference exactly.
/// </summary>
internal static class Simd
{
    /// <summary>y[i] += a * x[i]</summary>
    public static void Axpy(float a, ReadOnlySpan<float> x, Span<float> y)
    {
        if (x.Length != y.Length) throw new ArgumentException("Length mismatch.");

        int i = 0;
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var va = new Vector<float>(a);
            var vx = MemoryMarshal.Cast<float, Vector<float>>(x);
            var vy = MemoryMarshal.Cast<float, Vector<float>>(y);
            for (int j = 0; j < vx.Length; j++)
                vy[j] += va * vx[j];
            i = vx.Length * Vector<float>.Count;
        }
        for (; i < x.Length; i++)
            y[i] += a * x[i];
    }

    /// <summary>Σ x[i] * y[i], accumulated in float lanes then reduced.</summary>
    public static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (x.Length != y.Length) throw new ArgumentException("Length mismatch.");

        int i = 0;
        float sum = 0f;
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var acc = Vector<float>.Zero;
            var vx = MemoryMarshal.Cast<float, Vector<float>>(x);
            var vy = MemoryMarshal.Cast<float, Vector<float>>(y);
            for (int j = 0; j < vx.Length; j++)
                acc += vx[j] * vy[j];
            sum = Vector.Sum(acc);
            i = vx.Length * Vector<float>.Count;
        }
        for (; i < x.Length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    public static float Sum(ReadOnlySpan<float> x)
    {
        float sum = 0f;
        foreach (float v in x) sum += v;
        return sum;
    }
}

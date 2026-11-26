using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace DoraBio.Core.Nn;

/// <summary>
/// 512-bit versions of the blocked kernels whose SIMD lanes are pixels (systems brick S2, brick 36). In those kernels
/// each lane computes one output pixel from scalar weights and that pixel's own inputs, so a lane does exactly what the
/// 256-bit lane (and the scalar reference) does: the same terms, in the same order, each a separate multiply then add.
/// Doubling the lanes changes which pixels share an instruction, never a value. (The weight-gradient kernel sums
/// across pixels, so its lane width is part of its rounding; it stays on <c>Vector&lt;float&gt;</c>.)
/// </summary>
internal sealed partial class ConvLayer
{
    private const int Wide = 16;   // floats per Vector512

    /// <summary>
    /// True where 512-bit vectors are accelerated (AVX-512) and the wide kernels may be used. The environment variable
    /// <c>DORA_WIDE_KERNELS=0</c> turns them off (results are identical either way; only speed changes).
    /// </summary>
    internal static bool WideKernels { get; set; } =
        Vector512.IsHardwareAccelerated && Environment.GetEnvironmentVariable("DORA_WIDE_KERNELS") != "0";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Load512(ref float origin, int index) =>
        Unsafe.ReadUnaligned<Vector512<float>>(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref origin, index)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store512(ref float origin, int index, Vector512<float> value) =>
        Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref origin, index)), value);

    /// <summary>
    /// <see cref="Forward"/> on 16-pixel strips: blocks of 8 output channels in registers, each input vector (one tap of
    /// one input channel) loaded once for all 8. Rows narrower than 16 pixels go to the 256-bit kernel. Bit-identical to
    /// <see cref="ForwardReference"/> per output value.
    /// </summary>
    private void ForwardWide(ReadOnlySpan<float> parameters, ReadOnlySpan<float> padded, int height, int width, Span<float> output)
    {
        int d = Dilation, pw = PaddedWidth(width), paddedPlane = PaddedHeight(height) * pw, plane = height * width;
        int ic9 = InChannels * Taps;
        ref float input = ref MemoryMarshal.GetReference(padded);
        ref float result = ref MemoryMarshal.GetReference(output);
        ref float weights = ref MemoryMarshal.GetReference(parameters);
        _ = padded[PaddedLength(height, width) - 1];
        _ = output[OutChannels * plane - 1];
        // Offsets of the 9 taps from a pixel's top-left tap, in the reference's order (ky, then kx).
        int t1 = d, t2 = 2 * d, t3 = d * pw, t4 = t3 + d, t5 = t3 + 2 * d, t6 = 2 * d * pw, t7 = t6 + d, t8 = t6 + 2 * d;

        int oc = 0;
        for (; oc + Block <= OutChannels; oc += Block)
        {
            int wBlock = WeightOffset + oc * ic9;
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x = x + Wide >= width ? width : Math.Min(x + Wide, width - Wide))
                {
                    var a0 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc));
                    var a1 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 1));
                    var a2 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 2));
                    var a3 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 3));
                    var a4 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 4));
                    var a5 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 5));
                    var a6 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 6));
                    var a7 = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc + 7));
                    for (int ic = 0; ic < InChannels; ic++)
                    {
                        int row = ic * paddedPlane + y * pw + x, w = wBlock + ic * Taps;
                        Tap(ref input, row, ref weights, w, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t1, ref weights, w + 1, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t2, ref weights, w + 2, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t3, ref weights, w + 3, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t4, ref weights, w + 4, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t5, ref weights, w + 5, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t6, ref weights, w + 6, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t7, ref weights, w + 7, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref input, row + t8, ref weights, w + 8, ic9, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                    }
                    int o = oc * plane + y * width + x;
                    Store512(ref result, o, a0);
                    Store512(ref result, o + plane, a1);
                    Store512(ref result, o + 2 * plane, a2);
                    Store512(ref result, o + 3 * plane, a3);
                    Store512(ref result, o + 4 * plane, a4);
                    Store512(ref result, o + 5 * plane, a5);
                    Store512(ref result, o + 6 * plane, a6);
                    Store512(ref result, o + 7 * plane, a7);
                }
        }

        for (; oc < OutChannels; oc++)   // leftover channels (the 1-channel output layer): one accumulator per strip
        {
            int w0 = WeightOffset + oc * ic9;
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x = x + Wide >= width ? width : Math.Min(x + Wide, width - Wide))
                {
                    var a = Vector512.Create(Unsafe.Add(ref weights, BiasOffset + oc));
                    for (int ic = 0; ic < InChannels; ic++)
                    {
                        int row = ic * paddedPlane + y * pw + x, w = w0 + ic * Taps;
                        a += Vector512.Create(Unsafe.Add(ref weights, w)) * Load512(ref input, row);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 1)) * Load512(ref input, row + t1);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 2)) * Load512(ref input, row + t2);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 3)) * Load512(ref input, row + t3);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 4)) * Load512(ref input, row + t4);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 5)) * Load512(ref input, row + t5);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 6)) * Load512(ref input, row + t6);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 7)) * Load512(ref input, row + t7);
                        a += Vector512.Create(Unsafe.Add(ref weights, w + 8)) * Load512(ref input, row + t8);
                    }
                    Store512(ref result, oc * plane + y * width + x, a);
                }
        }
    }

    /// <summary>
    /// The input-gradient blocks of <see cref="InputGradients"/> on 16-pixel strips: each input-gradient value gathers
    /// W·G over output channels, then taps (the reference's order), from the 2d-padded copy of G, for 8 input channels in
    /// registers. Returns the first input channel it didn't cover (the leftovers go to the scalar path, as before).
    /// </summary>
    private int InputGradientBlocksWide(ref float weights, ref float g, ref float result, int height, int width, int gw, int gPlane)
    {
        int d = Dilation, pw = PaddedWidth(width), paddedPlane = PaddedHeight(height) * pw;
        // Offset of tap (ky, kx) from G's position for output pixel (Y, X): (2 − ky)·d rows and (2 − kx)·d columns.
        int t0 = 2 * d * gw + 2 * d, t1 = 2 * d * gw + d, t2 = 2 * d * gw, t3 = d * gw + 2 * d, t4 = d * gw + d, t5 = d * gw,
            t6 = 2 * d, t7 = d, t8 = 0;
        int ic = 0;
        for (; ic + Block <= InChannels; ic += Block)
            for (int Y = d; Y < d + height; Y++)
            {
                int end = d + width;
                for (int X = d; X < end; X = X + Wide >= end ? end : Math.Min(X + Wide, end - Wide))
                {
                    Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default, a6 = default, a7 = default;
                    for (int oc = 0; oc < OutChannels; oc++)
                    {
                        int w = WeightOffset + (oc * InChannels + ic) * Taps, at = oc * gPlane + Y * gw + X;
                        Tap(ref g, at + t0, ref weights, w, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t1, ref weights, w + 1, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t2, ref weights, w + 2, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t3, ref weights, w + 3, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t4, ref weights, w + 4, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t5, ref weights, w + 5, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t6, ref weights, w + 6, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t7, ref weights, w + 7, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                        Tap(ref g, at + t8, ref weights, w + 8, Taps, ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7);
                    }
                    int o = ic * paddedPlane + Y * pw + X;
                    Store512(ref result, o, a0);
                    Store512(ref result, o + paddedPlane, a1);
                    Store512(ref result, o + 2 * paddedPlane, a2);
                    Store512(ref result, o + 3 * paddedPlane, a3);
                    Store512(ref result, o + 4 * paddedPlane, a4);
                    Store512(ref result, o + 5 * paddedPlane, a5);
                    Store512(ref result, o + 6 * paddedPlane, a6);
                    Store512(ref result, o + 7 * paddedPlane, a7);
                }
            }
        return ic;
    }

    /// <summary>
    /// One tap into 8 channels' accumulators: load the input vector once, then for channel k multiply by the scalar weight
    /// at <paramref name="t"/> + k·<paramref name="ic9"/> and add (forward: 8 output channels; input gradient: 8 input channels).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Tap(ref float input, int at, ref float weights, int t, int ic9,
        ref Vector512<float> a0, ref Vector512<float> a1, ref Vector512<float> a2, ref Vector512<float> a3,
        ref Vector512<float> a4, ref Vector512<float> a5, ref Vector512<float> a6, ref Vector512<float> a7)
    {
        var v = Load512(ref input, at);
        a0 += Vector512.Create(Unsafe.Add(ref weights, t)) * v;
        a1 += Vector512.Create(Unsafe.Add(ref weights, t + ic9)) * v;
        a2 += Vector512.Create(Unsafe.Add(ref weights, t + 2 * ic9)) * v;
        a3 += Vector512.Create(Unsafe.Add(ref weights, t + 3 * ic9)) * v;
        a4 += Vector512.Create(Unsafe.Add(ref weights, t + 4 * ic9)) * v;
        a5 += Vector512.Create(Unsafe.Add(ref weights, t + 5 * ic9)) * v;
        a6 += Vector512.Create(Unsafe.Add(ref weights, t + 6 * ic9)) * v;
        a7 += Vector512.Create(Unsafe.Add(ref weights, t + 7 * ic9)) * v;
    }
}

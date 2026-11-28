using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DoraBio.Core.Nn;

/// <summary>
/// The register-blocked kernels (see the class remarks in ConvLayer.cs). Each works on blocks of <see cref="Block"/>
/// channels and strips of <c>Vector&lt;float&gt;.Count</c> pixels, keeping the block's accumulators in registers, and
/// falls back to one channel at a time and to scalar code for leftovers. Every output value receives exactly the
/// terms the reference kernel gives it, in the same order, each as a separate multiply then add.
/// </summary>
internal sealed partial class ConvLayer
{
    /// <summary>Channels per register block: 8 accumulators plus an input vector and a weight fit in 16 AVX registers.</summary>
    private const int Block = 8;

    private static int V => Vector<float>.Count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Load(ref float origin, int index) =>
        Unsafe.ReadUnaligned<Vector<float>>(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref origin, index)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(ref float origin, int index, Vector<float> value) =>
        Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref origin, index)), value);

    /// <summary>output[oc] = bias[oc] + Σ_ic Σ_taps w · shifted(padded[ic]), bit-identical to <see cref="ForwardReference"/>.</summary>
    /// <remarks>
    /// For a block of 8 output channels and one strip of pixels, each input vector (one tap of one input channel) is
    /// loaded once and multiplied into all 8 accumulators. Per output value the terms are: bias, then input channel by
    /// input channel, tap by tap, which is the reference order.
    /// </remarks>
    public void Forward(ReadOnlySpan<float> parameters, ReadOnlySpan<float> padded, int height, int width, Span<float> output)
    {
        if (WideKernels && width >= Wide)
        {
            ForwardWide(parameters, padded, height, width, output);   // 512-bit lanes, same per-pixel arithmetic
            return;
        }
        int d = Dilation, pw = PaddedWidth(width), paddedPlane = PaddedHeight(height) * pw, plane = height * width;
        int ic9 = InChannels * Taps;
        ref float input = ref MemoryMarshal.GetReference(padded);
        ref float result = ref MemoryMarshal.GetReference(output);
        ref float weights = ref MemoryMarshal.GetReference(parameters);
        _ = padded[PaddedLength(height, width) - 1];      // one bounds check up front for the unchecked loads below
        _ = output[OutChannels * plane - 1];
        int t1 = d, t2 = 2 * d, t3 = d * pw, t4 = t3 + d, t5 = t3 + 2 * d, t6 = 2 * d * pw, t7 = t6 + d, t8 = t6 + 2 * d;

        int oc = 0;
        for (; oc + Block <= OutChannels; oc += Block)
        {
            int wBlock = WeightOffset + oc * ic9;
            for (int y = 0; y < height; y++)
            {
                if (width < V)
                {
                    for (int x0 = 0; x0 < width; x0++)
                        for (int k = 0; k < Block; k++)
                            Unsafe.Add(ref result, (oc + k) * plane + y * width + x0) =
                                ForwardScalar(ref weights, ref input, oc + k, y, x0, d, pw, paddedPlane);
                    continue;
                }
                // Strips of V pixels; the last one is moved back to end at the row's end. Pixels it shares with the
                // previous strip are computed twice, to exactly the same value, which beats a slow scalar tail.
                for (int x = 0; x < width; x = x + V >= width ? width : Math.Min(x + V, width - V))
                {
                    var a0 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc));
                    var a1 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 1));
                    var a2 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 2));
                    var a3 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 3));
                    var a4 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 4));
                    var a5 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 5));
                    var a6 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 6));
                    var a7 = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc + 7));
                    for (int ic = 0; ic < InChannels; ic++)
                    {
                        // The 9 taps unrolled, offsets precomputed, in the reference's order (ky, then kx).
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
                    Store(ref result, o, a0);
                    Store(ref result, o + plane, a1);
                    Store(ref result, o + 2 * plane, a2);
                    Store(ref result, o + 3 * plane, a3);
                    Store(ref result, o + 4 * plane, a4);
                    Store(ref result, o + 5 * plane, a5);
                    Store(ref result, o + 6 * plane, a6);
                    Store(ref result, o + 7 * plane, a7);
                }
            }
        }

        for (; oc < OutChannels; oc++)   // leftover channels (the 1-channel output layer): one accumulator per strip
        {
            int w0 = WeightOffset + oc * ic9;
            for (int y = 0; y < height; y++)
            {
                if (width < V)
                {
                    for (int x0 = 0; x0 < width; x0++)
                        Unsafe.Add(ref result, oc * plane + y * width + x0) = ForwardScalar(ref weights, ref input, oc, y, x0, d, pw, paddedPlane);
                    continue;
                }
                for (int x = 0; x < width; x = x + V >= width ? width : Math.Min(x + V, width - V))
                {
                    var a = new Vector<float>(Unsafe.Add(ref weights, BiasOffset + oc));
                    for (int ic = 0; ic < InChannels; ic++)
                    {
                        int row = ic * paddedPlane + y * pw + x, w = w0 + ic * Taps;
                        for (int ky = 0; ky < Kernel; ky++)
                            for (int kx = 0; kx < Kernel; kx++)
                                a += new Vector<float>(Unsafe.Add(ref weights, w + ky * Kernel + kx)) * Load(ref input, row + ky * d * pw + kx * d);
                    }
                    Store(ref result, oc * plane + y * width + x, a);
                }
            }
        }
    }

    /// <summary>One tap into 8 channels' accumulators, <c>Vector&lt;float&gt;</c> lanes (see the 512-bit overload).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Tap(ref float input, int at, ref float weights, int t, int ic9,
        ref Vector<float> a0, ref Vector<float> a1, ref Vector<float> a2, ref Vector<float> a3,
        ref Vector<float> a4, ref Vector<float> a5, ref Vector<float> a6, ref Vector<float> a7)
    {
        var v = Load(ref input, at);
        a0 += new Vector<float>(Unsafe.Add(ref weights, t)) * v;
        a1 += new Vector<float>(Unsafe.Add(ref weights, t + ic9)) * v;
        a2 += new Vector<float>(Unsafe.Add(ref weights, t + 2 * ic9)) * v;
        a3 += new Vector<float>(Unsafe.Add(ref weights, t + 3 * ic9)) * v;
        a4 += new Vector<float>(Unsafe.Add(ref weights, t + 4 * ic9)) * v;
        a5 += new Vector<float>(Unsafe.Add(ref weights, t + 5 * ic9)) * v;
        a6 += new Vector<float>(Unsafe.Add(ref weights, t + 6 * ic9)) * v;
        a7 += new Vector<float>(Unsafe.Add(ref weights, t + 7 * ic9)) * v;
    }

    /// <summary>One output value, in the reference order (scalar float math, identical to a SIMD lane).</summary>
    private float ForwardScalar(ref float weights, ref float input, int oc, int y, int x, int d, int pw, int paddedPlane)
    {
        float s = Unsafe.Add(ref weights, BiasOffset + oc);
        for (int ic = 0; ic < InChannels; ic++)
        {
            int row = ic * paddedPlane + y * pw + x, w = WeightOffset + (oc * InChannels + ic) * Taps;
            for (int ky = 0; ky < Kernel; ky++)
                for (int kx = 0; kx < Kernel; kx++)
                    s += Unsafe.Add(ref weights, w + ky * Kernel + kx) * Unsafe.Add(ref input, row + ky * d * pw + kx * d);
        }
        return s;
    }

    /// <summary>
    /// Bit-identical to <see cref="BackwardReference"/>: bias and weight gradients are added to <paramref name="gradients"/>;
    /// if <paramref name="gradPadded"/> is non-empty, its interior receives ∂loss/∂input (its zero-padding border is
    /// left at zero rather than filled, since callers discard it).
    /// </summary>
    /// <remarks>
    /// Weight gradients: the reference computes each ∂loss/∂W[oc,ic,tap] as a sum over rows of
    /// <see cref="Simd.Dot"/>(gradient row, input row). Here 8 output channels share each input-row load, but every one
    /// of them keeps that exact reduction (lane sums, then <c>Vector.Sum</c>, then the scalar tail, then the row total).
    /// Input gradients: the reference scatters W·G into the input gradient tap by tap. Here each input-gradient value
    /// gathers the same terms (output channel by output channel, tap by tap) from a zero-padded copy of G, with 8 input
    /// channels in registers. The padding only ever adds +0 to an accumulator that started at +0, which can't change it.
    /// </remarks>
    /// <param name="parameters">The network's flat parameter array (read-only here).</param>
    /// <param name="padded">The padded input saved by the forward pass.</param>
    /// <param name="gradOutput">∂loss/∂output, [OutChannels, h, w].</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="gradients">Gradient buffer with the same layout as <paramref name="parameters"/>; added to, not overwritten.</param>
    /// <param name="gradPadded">Receives ∂loss/∂(padded input), or pass empty to skip it (the first layer, whose input is data).</param>
    public void Backward(
        ReadOnlySpan<float> parameters,
        ReadOnlySpan<float> padded,
        ReadOnlySpan<float> gradOutput,
        int height,
        int width,
        Span<float> gradients,
        Span<float> gradPadded)
    {
        int plane = height * width;
        for (int oc = 0; oc < OutChannels; oc++)
            gradients[BiasOffset + oc] += Simd.Sum(gradOutput.Slice(oc * plane, plane));
        WeightGradients(padded, gradOutput, height, width, gradients);
        if (!gradPadded.IsEmpty) InputGradients(parameters, gradOutput, height, width, gradPadded);
    }

    private void WeightGradients(ReadOnlySpan<float> padded, ReadOnlySpan<float> gradOutput, int height, int width, Span<float> gradients)
    {
        int d = Dilation, pw = PaddedWidth(width), paddedPlane = PaddedHeight(height) * pw, plane = height * width;
        int vectors = width >= V ? width / V : 0, tail = vectors * V;
        ref float input = ref MemoryMarshal.GetReference(padded);
        ref float grad = ref MemoryMarshal.GetReference(gradOutput);
        _ = padded[PaddedLength(height, width) - 1];
        _ = gradOutput[OutChannels * plane - 1];
        Span<float> rowSums = stackalloc float[Block];
        Span<float> totals = stackalloc float[Block];

        for (int oc = 0; oc < OutChannels; oc += Block)
        {
            int count = Math.Min(Block, OutChannels - oc);
            for (int ic = 0; ic < InChannels; ic++)
                for (int ky = 0; ky < Kernel; ky++)
                    for (int kx = 0; kx < Kernel; kx++)
                    {
                        int shift = ic * paddedPlane + ky * d * pw + kx * d;
                        totals.Clear();
                        for (int y = 0; y < height; y++)
                        {
                            int inRow = shift + y * pw, gRow = oc * plane + y * width;
                            if (count == Block && vectors > 0)
                            {
                                Vector<float> s0 = default, s1 = default, s2 = default, s3 = default, s4 = default, s5 = default, s6 = default, s7 = default;
                                for (int j = 0; j < vectors; j++)
                                {
                                    var v = Load(ref input, inRow + j * V);
                                    int g = gRow + j * V;
                                    s0 += Load(ref grad, g) * v;
                                    s1 += Load(ref grad, g + plane) * v;
                                    s2 += Load(ref grad, g + 2 * plane) * v;
                                    s3 += Load(ref grad, g + 3 * plane) * v;
                                    s4 += Load(ref grad, g + 4 * plane) * v;
                                    s5 += Load(ref grad, g + 5 * plane) * v;
                                    s6 += Load(ref grad, g + 6 * plane) * v;
                                    s7 += Load(ref grad, g + 7 * plane) * v;
                                }
                                rowSums[0] = Vector.Sum(s0); rowSums[1] = Vector.Sum(s1); rowSums[2] = Vector.Sum(s2); rowSums[3] = Vector.Sum(s3);
                                rowSums[4] = Vector.Sum(s4); rowSums[5] = Vector.Sum(s5); rowSums[6] = Vector.Sum(s6); rowSums[7] = Vector.Sum(s7);
                            }
                            else
                            {
                                for (int k = 0; k < count; k++)
                                {
                                    var s = Vector<float>.Zero;
                                    for (int j = 0; j < vectors; j++)
                                        s += Load(ref grad, gRow + k * plane + j * V) * Load(ref input, inRow + j * V);
                                    rowSums[k] = vectors > 0 ? Vector.Sum(s) : 0f;
                                }
                            }
                            // Scalar tail of each row's dot product, then the row total, as in Simd.Dot and the reference.
                            for (int k = 0; k < count; k++)
                            {
                                float sum = rowSums[k];
                                for (int x = tail; x < width; x++)
                                    sum += Unsafe.Add(ref grad, gRow + k * plane + x) * Unsafe.Add(ref input, inRow + x);
                                totals[k] += sum;
                            }
                        }
                        for (int k = 0; k < count; k++)
                            gradients[WeightOffset + ((oc + k) * InChannels + ic) * Taps + ky * Kernel + kx] += totals[k];
                    }
        }
    }

    private void InputGradients(ReadOnlySpan<float> parameters, ReadOnlySpan<float> gradOutput, int height, int width, Span<float> gradPadded)
    {
        int d = Dilation, pw = PaddedWidth(width), paddedPlane = PaddedHeight(height) * pw, plane = height * width;
        gradPadded[..PaddedLength(height, width)].Clear();

        // G zero-padded by 2d on every side, so G(Y − ky·d, X − kx·d) is always a valid read for interior (Y, X).
        int gw = width + 4 * d, gPlane = (height + 4 * d) * gw;
        float[] gPad = ArrayPool<float>.Shared.Rent(OutChannels * gPlane);
        try
        {
            Array.Clear(gPad, 0, OutChannels * gPlane);
            for (int oc = 0; oc < OutChannels; oc++)
                for (int y = 0; y < height; y++)
                    gradOutput.Slice(oc * plane + y * width, width).CopyTo(gPad.AsSpan(oc * gPlane + (y + 2 * d) * gw + 2 * d, width));

            ref float g = ref MemoryArrayStart(gPad);
            ref float weights = ref MemoryMarshal.GetReference(parameters);
            ref float result = ref MemoryMarshal.GetReference(gradPadded);
            _ = parameters[WeightOffset + WeightCount - 1];
            int ic = 0;
            if (WideKernels && width >= Wide)
                ic = InputGradientBlocksWide(ref weights, ref g, ref result, height, width, gw, gPlane);   // same per-pixel arithmetic
            for (; ic + Block <= InChannels; ic += Block)
                for (int Y = d; Y < d + height; Y++)
                {
                    if (width < V)
                    {
                        for (int X0 = d; X0 < d + width; X0++)
                            for (int k = 0; k < Block; k++)
                                Unsafe.Add(ref result, (ic + k) * paddedPlane + Y * pw + X0) = InputGradientScalar(ref weights, ref g, ic + k, Y, X0, d, gw, gPlane);
                        continue;
                    }
                    int end = d + width;
                    for (int X = d; X < end; X = X + V >= end ? end : Math.Min(X + V, end - V))
                    {
                        Vector<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default, a6 = default, a7 = default;
                        for (int oc = 0; oc < OutChannels; oc++)
                        {
                            int w = WeightOffset + (oc * InChannels + ic) * Taps;
                            for (int ky = 0; ky < Kernel; ky++)
                                for (int kx = 0; kx < Kernel; kx++)
                                {
                                    var v = Load(ref g, oc * gPlane + (Y + (2 - ky) * d) * gw + X + (2 - kx) * d);
                                    int t = w + ky * Kernel + kx;
                                    a0 += new Vector<float>(Unsafe.Add(ref weights, t)) * v;
                                    a1 += new Vector<float>(Unsafe.Add(ref weights, t + Taps)) * v;
                                    a2 += new Vector<float>(Unsafe.Add(ref weights, t + 2 * Taps)) * v;
                                    a3 += new Vector<float>(Unsafe.Add(ref weights, t + 3 * Taps)) * v;
                                    a4 += new Vector<float>(Unsafe.Add(ref weights, t + 4 * Taps)) * v;
                                    a5 += new Vector<float>(Unsafe.Add(ref weights, t + 5 * Taps)) * v;
                                    a6 += new Vector<float>(Unsafe.Add(ref weights, t + 6 * Taps)) * v;
                                    a7 += new Vector<float>(Unsafe.Add(ref weights, t + 7 * Taps)) * v;
                                }
                        }
                        int o = ic * paddedPlane + Y * pw + X;
                        Store(ref result, o, a0);
                        Store(ref result, o + paddedPlane, a1);
                        Store(ref result, o + 2 * paddedPlane, a2);
                        Store(ref result, o + 3 * paddedPlane, a3);
                        Store(ref result, o + 4 * paddedPlane, a4);
                        Store(ref result, o + 5 * paddedPlane, a5);
                        Store(ref result, o + 6 * paddedPlane, a6);
                        Store(ref result, o + 7 * paddedPlane, a7);
                    }
                }

            for (; ic < InChannels; ic++)
                for (int Y = d; Y < d + height; Y++)
                    for (int X = d; X < d + width; X++)
                        Unsafe.Add(ref result, ic * paddedPlane + Y * pw + X) = InputGradientScalar(ref weights, ref g, ic, Y, X, d, gw, gPlane);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gPad);
        }
    }

    /// <summary>One input-gradient value, gathering the reference's terms in its order (output channel, then tap).</summary>
    private float InputGradientScalar(ref float weights, ref float g, int ic, int Y, int X, int d, int gw, int gPlane)
    {
        float s = 0f;
        for (int oc = 0; oc < OutChannels; oc++)
        {
            int w = WeightOffset + (oc * InChannels + ic) * Taps;
            for (int ky = 0; ky < Kernel; ky++)
                for (int kx = 0; kx < Kernel; kx++)
                    s += Unsafe.Add(ref weights, w + ky * Kernel + kx) * Unsafe.Add(ref g, oc * gPlane + (Y + (2 - ky) * d) * gw + X + (2 - kx) * d);
        }
        return s;
    }

    private static ref float MemoryArrayStart(float[] array) => ref MemoryMarshal.GetArrayDataReference(array);
}

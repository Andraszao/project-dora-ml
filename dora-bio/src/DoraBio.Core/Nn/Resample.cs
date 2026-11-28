namespace DoraBio.Core.Nn;

/// <summary>
/// 2×2 average pooling and 2× nearest-neighbour upsampling on planar activations [channels, height, width]
/// (index = (c · height + y) · width + x, as in <see cref="ConvLayer"/>). They move a U-Net between resolutions.
///
/// Both are linear, so each backward pass is simply the transpose (adjoint) of its forward pass:
/// <code>
/// pool:      out(y, x) = ¼ · Σ_{i,j ∈ {0,1}} in(2y+i, 2x+j)        backward: gIn(2y+i, 2x+j) = ¼ · gOut(y, x)
/// upsample:  out(y, x) = in(⌊y/2⌋, ⌊x/2⌋)                          backward: gIn(y, x) = Σ_{i,j} gOut(2y+i, 2x+j)
/// </code>
/// Average rather than max pooling: it has no ties or argmax to remember, its gradient is exact everywhere, and for
/// denoising it averages noise down instead of picking out its peaks.
/// </summary>
internal static class Resample
{
    /// <summary>[C, h, w] → [C, h/2, w/2], each output the mean of a 2×2 block.</summary>
    /// <param name="input">Input activations, channels × height × width.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="height">Input height (even).</param>
    /// <param name="width">Input width (even).</param>
    /// <param name="output">Receives channels × height/2 × width/2 values.</param>
    public static void AvgPool2(ReadOnlySpan<float> input, int channels, int height, int width, Span<float> output)
    {
        CheckEven(height, width);
        int oh = height / 2, ow = width / 2;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < oh; y++)
            {
                var top = input.Slice((c * height + 2 * y) * width, width);
                var bottom = input.Slice((c * height + 2 * y + 1) * width, width);
                var row = output.Slice((c * oh + y) * ow, ow);
                for (int x = 0; x < ow; x++)
                    row[x] = 0.25f * (top[2 * x] + top[2 * x + 1] + bottom[2 * x] + bottom[2 * x + 1]);
            }
    }

    /// <summary>Transpose of <see cref="AvgPool2"/>: spreads each output gradient as ¼ over its 2×2 block. Overwrites.</summary>
    /// <param name="gradOutput">Gradient at the pooled output, channels × height/2 × width/2.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="height">Input (full-resolution) height (even).</param>
    /// <param name="width">Input (full-resolution) width (even).</param>
    /// <param name="gradInput">Receives the gradient at the input, channels × height × width.</param>
    public static void AvgPool2Backward(ReadOnlySpan<float> gradOutput, int channels, int height, int width, Span<float> gradInput)
    {
        CheckEven(height, width);
        int oh = height / 2, ow = width / 2;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < oh; y++)
            {
                var g = gradOutput.Slice((c * oh + y) * ow, ow);
                var top = gradInput.Slice((c * height + 2 * y) * width, width);
                var bottom = gradInput.Slice((c * height + 2 * y + 1) * width, width);
                for (int x = 0; x < ow; x++)
                {
                    float v = 0.25f * g[x];
                    top[2 * x] = top[2 * x + 1] = bottom[2 * x] = bottom[2 * x + 1] = v;
                }
            }
    }

    /// <summary>[C, h, w] → [C, 2h, 2w], each input value copied to a 2×2 block.</summary>
    /// <param name="input">Low-resolution activations, channels × height × width.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="height">Input (low-resolution) height.</param>
    /// <param name="width">Input (low-resolution) width.</param>
    /// <param name="output">Receives channels × 2·height × 2·width values.</param>
    public static void Upsample2(ReadOnlySpan<float> input, int channels, int height, int width, Span<float> output)
    {
        int oh = 2 * height, ow = 2 * width;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < height; y++)
            {
                var row = input.Slice((c * height + y) * width, width);
                var top = output.Slice((c * oh + 2 * y) * ow, ow);
                for (int x = 0; x < width; x++) top[2 * x] = top[2 * x + 1] = row[x];
                top.CopyTo(output.Slice((c * oh + 2 * y + 1) * ow, ow));
            }
    }

    /// <summary>Transpose of <see cref="Upsample2"/>: each input gradient is the sum over its 2×2 block. Overwrites.</summary>
    /// <param name="gradOutput">Gradient at the upsampled output, channels × 2·height × 2·width.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="height">Input (low-resolution) height.</param>
    /// <param name="width">Input (low-resolution) width.</param>
    /// <param name="gradInput">Receives the gradient at the low-resolution input, channels × height × width.</param>
    public static void Upsample2Backward(ReadOnlySpan<float> gradOutput, int channels, int height, int width, Span<float> gradInput)
    {
        int oh = 2 * height, ow = 2 * width;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < height; y++)
            {
                var top = gradOutput.Slice((c * oh + 2 * y) * ow, ow);
                var bottom = gradOutput.Slice((c * oh + 2 * y + 1) * ow, ow);
                var row = gradInput.Slice((c * height + y) * width, width);
                for (int x = 0; x < width; x++)
                    row[x] = top[2 * x] + top[2 * x + 1] + bottom[2 * x] + bottom[2 * x + 1];
            }
    }

    private static void CheckEven(int height, int width)
    {
        if (height <= 0 || width <= 0 || height % 2 != 0 || width % 2 != 0)
            throw new ArgumentException($"2×2 pooling needs even, positive sizes; got {width}×{height}.");
    }
}

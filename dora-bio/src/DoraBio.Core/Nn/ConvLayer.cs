namespace DoraBio.Core.Nn;

/// <summary>
/// One 3×3 convolution with "same" zero padding and optional dilation, for a single sample.
///
/// The layer owns no arrays. Its weights and bias live in the network's flat parameter array at
/// <see cref="WeightOffset"/>, so the optimizer and the weights file each handle one contiguous float[].
/// Weight layout: [outChannel][inChannel][ky][kx], followed by [outChannel] biases.
///
/// Activations are planar, channel-major: index = (c * height + y) * width + x.
/// Every inner loop is a SIMD axpy or dot product over one image row.
///
/// <para><b>The math</b> (d = dilation, P = the input zero-padded by d on every side):</para>
/// <code>
/// forward:   out[oc](y, x)   = b[oc] + Σ_ic Σ_ky,kx  W[oc,ic,ky,kx] · P[ic](y + ky·d, x + kx·d)
///
/// backward, given G = ∂loss/∂out:
///   ∂loss/∂b[oc]             = Σ_y,x G[oc](y, x)
///   ∂loss/∂W[oc,ic,ky,kx]    = Σ_y,x G[oc](y, x) · P[ic](y + ky·d, x + kx·d)     (one dot product per row)
///   ∂loss/∂P[ic](y+ky·d, x+kx·d) += W[oc,ic,ky,kx] · G[oc](y, x)                  (one axpy per row)
/// </code>
/// The last line is the "transposed convolution": every output gradient is scattered back to the input
/// pixels that produced it. Gradients that land in the padding border are discarded by <see cref="Unpad"/>,
/// because padding is a constant zero, not a learnable input.
/// </summary>
internal sealed class ConvLayer
{
    /// <summary>Kernel side length. Fixed at 3; larger receptive fields come from dilation, not bigger kernels.</summary>
    public const int Kernel = 3;
    private const int Taps = Kernel * Kernel;

    /// <summary>Channels read per pixel (1 for the network's first layer).</summary>
    public int InChannels { get; }

    /// <summary>Channels written per pixel (1 for the network's last layer).</summary>
    public int OutChannels { get; }

    /// <summary>Spacing between kernel taps: 1 is an ordinary 3×3 kernel, 2 samples every other pixel, and so on.</summary>
    public int Dilation { get; }

    /// <summary>Index of this layer's first weight in the network's flat parameter array.</summary>
    public int WeightOffset { get; }

    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="dilation">Tap spacing (≥ 1). The layer adds <c>dilation</c> pixels of context on each side.</param>
    /// <param name="weightOffset">Where this layer's weights start in the flat parameter array.</param>
    public ConvLayer(int inChannels, int outChannels, int dilation, int weightOffset)
    {
        if (inChannels <= 0 || outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        InChannels = inChannels;
        OutChannels = outChannels;
        Dilation = dilation;
        WeightOffset = weightOffset;
    }

    /// <summary>Number of weights: OutChannels × InChannels × 9.</summary>
    public int WeightCount => OutChannels * InChannels * Taps;

    /// <summary>Index of the first bias, directly after the weights.</summary>
    public int BiasOffset => WeightOffset + WeightCount;

    /// <summary>Weights plus one bias per output channel.</summary>
    public int ParameterCount => WeightCount + OutChannels;

    /// <summary>Inputs feeding each output value (InChannels × 9). Used to scale the initial weights.</summary>
    public int FanIn => InChannels * Taps;

    /// <summary>Row length of the padded input buffer.</summary>
    public int PaddedWidth(int width) => width + 2 * Dilation;

    /// <summary>Number of rows in the padded input buffer.</summary>
    public int PaddedHeight(int height) => height + 2 * Dilation;

    /// <summary>Floats needed for the padded input of an image of this size.</summary>
    public int PaddedLength(int height, int width) => InChannels * PaddedHeight(height) * PaddedWidth(width);

    /// <summary>Copies a [InChannels, h, w] input into a zero-bordered [InChannels, h+2d, w+2d] buffer.</summary>
    /// <remarks>
    /// Padding once up front means the inner loops never test for image edges: every tap reads a valid
    /// address, and out-of-image taps read the zero border. The padded copy is kept in the workspace,
    /// because the backward pass needs the same values again to compute weight gradients.
    /// </remarks>
    public void Pad(ReadOnlySpan<float> input, int height, int width, Span<float> padded)
    {
        int d = Dilation, pw = PaddedWidth(width), ph = PaddedHeight(height);
        padded = padded[..PaddedLength(height, width)];
        padded.Clear();
        for (int c = 0; c < InChannels; c++)
            for (int y = 0; y < height; y++)
                input.Slice((c * height + y) * width, width)
                     .CopyTo(padded.Slice((c * ph + y + d) * pw + d, width));
    }

    /// <summary>Inverse of <see cref="Pad"/>: drops the border, keeping the [InChannels, h, w] interior.</summary>
    public void Unpad(ReadOnlySpan<float> padded, int height, int width, Span<float> output)
    {
        int d = Dilation, pw = PaddedWidth(width), ph = PaddedHeight(height);
        for (int c = 0; c < InChannels; c++)
            for (int y = 0; y < height; y++)
                padded.Slice((c * ph + y + d) * pw + d, width)
                      .CopyTo(output.Slice((c * height + y) * width, width));
    }

    /// <summary>output[oc] = bias[oc] + Σ_ic Σ_taps w * shifted(padded[ic]). Output is [OutChannels, h, w].</summary>
    /// <remarks>
    /// Loop order: for each (output channel, input channel, tap), the whole tap contributes
    /// <c>weight × (the input plane shifted by the tap offset)</c> to the output plane, one row at a time.
    /// Each row is a contiguous SIMD axpy. That's much faster than gathering 9 × InChannels values per pixel.
    /// </remarks>
    public void Forward(ReadOnlySpan<float> parameters, ReadOnlySpan<float> padded, int height, int width, Span<float> output)
    {
        int d = Dilation, pw = PaddedWidth(width), ph = PaddedHeight(height);
        int plane = height * width, paddedPlane = ph * pw;

        for (int oc = 0; oc < OutChannels; oc++)
        {
            var outPlane = output.Slice(oc * plane, plane);
            outPlane.Fill(parameters[BiasOffset + oc]);

            for (int ic = 0; ic < InChannels; ic++)
            {
                var inPlane = padded.Slice(ic * paddedPlane, paddedPlane);
                int wBase = WeightOffset + (oc * InChannels + ic) * Taps;

                for (int ky = 0; ky < Kernel; ky++)
                    for (int kx = 0; kx < Kernel; kx++)
                    {
                        float weight = parameters[wBase + ky * Kernel + kx];
                        // Offset of this tap inside the padded plane: output (y, x) reads padded (y + ky·d, x + kx·d).
                        int shift = ky * d * pw + kx * d;
                        for (int y = 0; y < height; y++)
                            Simd.Axpy(weight, inPlane.Slice(shift + y * pw, width), outPlane.Slice(y * width, width));
                    }
            }
        }
    }

    /// <summary>
    /// Accumulates (+=) weight and bias gradients into <paramref name="gradients"/>, which uses the parameter layout.
    /// If <paramref name="gradPadded"/> is non-empty it is overwritten with dLoss/dPaddedInput, in the padded layout.
    /// Callers crop it with <see cref="Unpad"/>; values on the border belong to the zero padding and are discarded.
    /// </summary>
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
        int d = Dilation, pw = PaddedWidth(width), ph = PaddedHeight(height);
        int plane = height * width, paddedPlane = ph * pw;
        bool wantInputGrad = !gradPadded.IsEmpty;
        if (wantInputGrad) gradPadded[..PaddedLength(height, width)].Clear();

        for (int oc = 0; oc < OutChannels; oc++)
        {
            var gOutPlane = gradOutput.Slice(oc * plane, plane);
            // The bias is added to every output pixel, so its gradient is the sum of all output gradients.
            gradients[BiasOffset + oc] += Simd.Sum(gOutPlane);

            for (int ic = 0; ic < InChannels; ic++)
            {
                var inPlane = padded.Slice(ic * paddedPlane, paddedPlane);
                int wBase = WeightOffset + (oc * InChannels + ic) * Taps;

                for (int ky = 0; ky < Kernel; ky++)
                    for (int kx = 0; kx < Kernel; kx++)
                    {
                        int tap = wBase + ky * Kernel + kx;
                        int shift = ky * d * pw + kx * d;   // same tap offset as in Forward

                        // ∂loss/∂W: correlate the output gradient with the input this tap saw.
                        float gw = 0f;
                        for (int y = 0; y < height; y++)
                            gw += Simd.Dot(gOutPlane.Slice(y * width, width), inPlane.Slice(shift + y * pw, width));
                        gradients[tap] += gw;

                        if (!wantInputGrad) continue;
                        // ∂loss/∂input: send each output gradient back, weighted, to the pixel it read.
                        float weight = parameters[tap];
                        var gInPlane = gradPadded.Slice(ic * paddedPlane, paddedPlane);
                        for (int y = 0; y < height; y++)
                            Simd.Axpy(weight, gOutPlane.Slice(y * width, width), gInPlane.Slice(shift + y * pw, width));
                    }
            }
        }
    }
}

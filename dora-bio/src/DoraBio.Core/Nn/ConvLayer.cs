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
/// </summary>
internal sealed class ConvLayer
{
    public const int Kernel = 3;
    private const int Taps = Kernel * Kernel;

    public int InChannels { get; }
    public int OutChannels { get; }
    public int Dilation { get; }
    public int WeightOffset { get; }

    public ConvLayer(int inChannels, int outChannels, int dilation, int weightOffset)
    {
        if (inChannels <= 0 || outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        InChannels = inChannels;
        OutChannels = outChannels;
        Dilation = dilation;
        WeightOffset = weightOffset;
    }

    public int WeightCount => OutChannels * InChannels * Taps;
    public int BiasOffset => WeightOffset + WeightCount;
    public int ParameterCount => WeightCount + OutChannels;
    public int FanIn => InChannels * Taps;

    public int PaddedWidth(int width) => width + 2 * Dilation;
    public int PaddedHeight(int height) => height + 2 * Dilation;
    public int PaddedLength(int height, int width) => InChannels * PaddedHeight(height) * PaddedWidth(width);

    /// <summary>Copies a [InChannels, h, w] input into a zero-bordered [InChannels, h+2d, w+2d] buffer.</summary>
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
            gradients[BiasOffset + oc] += Simd.Sum(gOutPlane);

            for (int ic = 0; ic < InChannels; ic++)
            {
                var inPlane = padded.Slice(ic * paddedPlane, paddedPlane);
                int wBase = WeightOffset + (oc * InChannels + ic) * Taps;

                for (int ky = 0; ky < Kernel; ky++)
                    for (int kx = 0; kx < Kernel; kx++)
                    {
                        int tap = wBase + ky * Kernel + kx;
                        int shift = ky * d * pw + kx * d;

                        float gw = 0f;
                        for (int y = 0; y < height; y++)
                            gw += Simd.Dot(gOutPlane.Slice(y * width, width), inPlane.Slice(shift + y * pw, width));
                        gradients[tap] += gw;

                        if (!wantInputGrad) continue;
                        float weight = parameters[tap];
                        var gInPlane = gradPadded.Slice(ic * paddedPlane, paddedPlane);
                        for (int y = 0; y < height; y++)
                            Simd.Axpy(weight, gOutPlane.Slice(y * width, width), gInPlane.Slice(shift + y * pw, width));
                    }
            }
        }
    }
}

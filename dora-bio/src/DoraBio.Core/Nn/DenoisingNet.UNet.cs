namespace DoraBio.Core.Nn;

/// <summary>
/// The U-Net half of <see cref="DenoisingNet"/> (<see cref="NetArchitecture.UNet"/>).
///
/// <para>Shape, with D = Depth, Cₗ = Features · 2ˡ channels and level l at 1/2ˡ resolution:</para>
/// <code>
/// encoder   l = 0..D    eₗ = relu(conv(relu(conv(inₗ))))       in₀ = image (1 ch),  inₗ = avgpool(eₗ₋₁)
///                       (level D is the bottleneck)
/// decoder   l = D−1..0  dₗ = relu(conv(relu(conv([ up(dₗ₊₁) ; eₗ ]))))   d_D = e_D,  [a ; b] stacks channels
/// output    r = conv(d₀)  (1 ch, linear), and denoised = image − r, as in the dilated network
/// </code>
/// <para>Layer order in the parameter array: enc(0,a), enc(0,b), …, enc(D,a), enc(D,b), dec(D−1,a), dec(D−1,b), …,
/// dec(0,a), dec(0,b), final. All convolutions are 3×3 with dilation 1.</para>
///
/// <para>Backward runs the same graph in reverse. The only new rules are the transposes of pooling and upsampling
/// (<see cref="Resample"/>) and the skip connection: each eₗ feeds two consumers (the next encoder level through the
/// pool, and the decoder through the concatenation), so its gradient is the sum of both. The decoder sweep stores the
/// skip part per level; the encoder sweep adds it to the pooling part before applying eₗ's ReLU mask.</para>
/// </summary>
public sealed partial class DenoisingNet
{
    private int Depth => Config.Depth;
    private int Channels(int level) => Config.Features << level;
    private int Enc(int level, int k) => 2 * level + k;
    private int Dec(int level, int k) => 2 * (Depth + 1) + 2 * (Depth - 1 - level) + k;
    private int FinalLayer => 4 * Depth + 2;

    /// <summary>Resolution level of each layer, in parameter order.</summary>
    private int LevelOf(int layer)
    {
        if (layer == FinalLayer) return 0;
        if (layer < 2 * (Depth + 1)) return layer / 2;
        return Depth - 1 - (layer - 2 * (Depth + 1)) / 2;
    }

    /// <summary>U-Net layers in parameter order: (input channels, output channels, dilation 1).</summary>
    private static List<(int In, int Out, int Dilation)> UNetLayerShapes(NetConfig config)
    {
        int f = config.Features, d = config.Depth;
        var shapes = new List<(int, int, int)>();
        for (int l = 0; l <= d; l++)
        {
            shapes.Add((l == 0 ? 1 : f << (l - 1), f << l, 1));
            shapes.Add((f << l, f << l, 1));
        }
        for (int l = d - 1; l >= 0; l--)
        {
            shapes.Add(((f << (l + 1)) + (f << l), f << l, 1));
            shapes.Add((f << l, f << l, 1));
        }
        shapes.Add((f, 1, 1));
        return shapes;
    }

    /// <summary>Forward pass for sides that are multiples of 2^Depth; returns the predicted noise (1 channel).</summary>
    private Span<float> ForwardUNet(ReadOnlySpan<float> input, int height, int width, Workspace ws)
    {
        ws.Ensure(height, width);
        ReadOnlySpan<float> current = input;
        for (int l = 0; l <= Depth; l++)
        {
            int hl = height >> l, wl = width >> l;
            if (l > 0)
            {
                int hp = height >> (l - 1), wp = width >> (l - 1);
                var previous = ws.Activations[Enc(l - 1, 1)].AsSpan(0, Channels(l - 1) * hp * wp);
                var pooled = ws.Pooled[l].AsSpan(0, Channels(l - 1) * hl * wl);
                Resample.AvgPool2(previous, Channels(l - 1), hp, wp, pooled);
                current = pooled;
            }
            current = ConvRelu(Enc(l, 0), current, hl, wl, ws);
            current = ConvRelu(Enc(l, 1), current, hl, wl, ws);
        }

        for (int l = Depth - 1; l >= 0; l--)
        {
            int hl = height >> l, wl = width >> l, plane = hl * wl;
            var stacked = ws.Cat[l].AsSpan(0, (Channels(l + 1) + Channels(l)) * plane);
            Resample.Upsample2(current, Channels(l + 1), height >> (l + 1), width >> (l + 1), stacked[..(Channels(l + 1) * plane)]);
            ws.Activations[Enc(l, 1)].AsSpan(0, Channels(l) * plane).CopyTo(stacked[(Channels(l + 1) * plane)..]);
            current = ConvRelu(Dec(l, 0), stacked, hl, wl, ws);
            current = ConvRelu(Dec(l, 1), current, hl, wl, ws);
        }

        var final = _layers[FinalLayer];
        final.Pad(current, height, width, ws.Padded[FinalLayer]);
        var residual = ws.Activations[FinalLayer].AsSpan(0, height * width);
        ConvForward(final, ws.Padded[FinalLayer], height, width, residual);
        return residual;
    }

    /// <summary>One 3×3 convolution at a level, followed by ReLU; keeps its padded input and output for backward.</summary>
    private Span<float> ConvRelu(int layer, ReadOnlySpan<float> input, int height, int width, Workspace ws)
    {
        var conv = _layers[layer];
        conv.Pad(input, height, width, ws.Padded[layer]);
        var output = ws.Activations[layer].AsSpan(0, conv.OutChannels * height * width);
        ConvForward(conv, ws.Padded[layer], height, width, output);
        Relu(output);
        return output;
    }

    /// <summary>
    /// Backward pass: ∂loss/∂(predicted noise) in, every layer's weight and bias gradients added to
    /// <paramref name="gradients"/>. Three scratch buffers take turns; nothing is allocated.
    /// </summary>
    private void BackwardUNet(int height, int width, ReadOnlySpan<float> gradResidual, Span<float> gradients, Workspace ws)
    {
        float[] a = ws.GradA, b = ws.GradB, c = ws.GradC;

        // Output layer → gradient at d₀.
        var gDeep = ConvInputGradient(FinalLayer, gradResidual, height, width, gradients, ws, a);
        MaskRelu(gDeep, ws.Activations[Dec(0, 1)]);

        // Decoder, shallow to deep. At each level the stacked input's gradient splits into the upsampled half (sent
        // down through the transpose of upsampling) and the skip half (kept for the encoder sweep).
        for (int l = 0; l < Depth; l++)
        {
            int hl = height >> l, wl = width >> l, plane = hl * wl;
            var g1 = ConvInputGradient(Dec(l, 1), gDeep, hl, wl, gradients, ws, b);
            MaskRelu(g1, ws.Activations[Dec(l, 0)]);
            var gStacked = ConvInputGradient(Dec(l, 0), g1, hl, wl, gradients, ws, c);

            int upLength = Channels(l + 1) * plane;
            gStacked[upLength..].CopyTo(ws.SkipGrad[l]);
            int hd = height >> (l + 1), wd = width >> (l + 1);
            var gLower = a.AsSpan(0, Channels(l + 1) * hd * wd);
            Resample.Upsample2Backward(gStacked[..upLength], Channels(l + 1), hd, wd, gLower);
            MaskRelu(gLower, ws.Activations[l + 1 < Depth ? Dec(l + 1, 1) : Enc(Depth, 1)]);
            gDeep = gLower;
        }

        // Bottleneck and encoder, deep to shallow. gOut is the gradient at eₗ: from the bottleneck, it arrives already
        // masked; at shallower levels it is the pooling path's gradient plus the stored skip gradient, masked once.
        var gOut = gDeep;
        for (int l = Depth; l >= 0; l--)
        {
            int hl = height >> l, wl = width >> l;
            if (l < Depth)
            {
                Simd.Axpy(1f, ws.SkipGrad[l].AsSpan(0, gOut.Length), gOut);
                MaskRelu(gOut, ws.Activations[Enc(l, 1)]);
            }
            var gA = ConvInputGradient(Enc(l, 1), gOut, hl, wl, gradients, ws, b);
            MaskRelu(gA, ws.Activations[Enc(l, 0)]);

            if (l == 0)
            {
                // The first layer's input is the image itself, so no input gradient is needed.
                ConvBackward(_layers[Enc(0, 0)], ws.Padded[Enc(0, 0)], gA, hl, wl, gradients, Span<float>.Empty);
                break;
            }
            var gPooled = ConvInputGradient(Enc(l, 0), gA, hl, wl, gradients, ws, c);
            int hp = height >> (l - 1), wp = width >> (l - 1);
            var gPrevious = a.AsSpan(0, Channels(l - 1) * hp * wp);
            Resample.AvgPool2Backward(gPooled, Channels(l - 1), hp, wp, gPrevious);
            gOut = gPrevious;
        }
    }

    /// <summary>Adds one layer's parameter gradients and returns the gradient at its (unpadded) input, written to <paramref name="into"/>.</summary>
    private Span<float> ConvInputGradient(int layer, ReadOnlySpan<float> gradOutput, int height, int width,
        Span<float> gradients, Workspace ws, float[] into)
    {
        var conv = _layers[layer];
        ConvBackward(conv, ws.Padded[layer], gradOutput, height, width, gradients, ws.GradPadded);
        var gradInput = into.AsSpan(0, conv.InChannels * height * width);
        conv.Unpad(ws.GradPadded, height, width, gradInput);
        return gradInput;
    }

    /// <summary>Back through a ReLU: zero the gradient wherever the stored activation relu(z) is not positive.</summary>
    private static void MaskRelu(Span<float> gradient, float[] activation)
    {
        for (int i = 0; i < gradient.Length; i++)
            if (activation[i] <= 0f) gradient[i] = 0f;
    }

    public sealed partial class Workspace
    {
        private int _unetHeight, _unetWidth;

        /// <summary>U-Net, per level l ≥ 1: the pooled input of that level's first convolution.</summary>
        internal float[][] Pooled { get; private set; } = [];

        /// <summary>U-Net, per decoder level: [up(dₗ₊₁) ; eₗ], the stacked input of its first convolution.</summary>
        internal float[][] Cat { get; private set; } = [];

        /// <summary>U-Net, per decoder level: the gradient reaching eₗ through the skip connection.</summary>
        internal float[][] SkipGrad { get; private set; } = [];

        /// <summary>U-Net: the third scratch buffer for gradients flowing backward.</summary>
        internal float[] GradC { get; private set; } = [];

        /// <summary>
        /// Sizes every U-Net buffer for the largest height and width seen so far (each layer at its own level's
        /// resolution). A buffer sized for H×W serves any smaller h×w, since slices use the actual sizes.
        /// </summary>
        private void EnsureUNet(int height, int width)
        {
            if (height <= _unetHeight && width <= _unetWidth) return;
            _unetHeight = Math.Max(_unetHeight, height);
            _unetWidth = Math.Max(_unetWidth, width);
            int h = _unetHeight, w = _unetWidth, depth = _net.Config.Depth;
            var layers = _net._layers;
            int Plane(int level) => (h >> level) * (w >> level);

            Padded = layers.Select((l, i) => new float[l.PaddedLength(h >> _net.LevelOf(i), w >> _net.LevelOf(i))]).ToArray();
            Activations = layers.Select((l, i) => new float[l.OutChannels * Plane(_net.LevelOf(i))]).ToArray();
            Pooled = Enumerable.Range(0, depth + 1).Select(l => l == 0 ? [] : new float[_net.Channels(l - 1) * Plane(l)]).ToArray();
            Cat = Enumerable.Range(0, depth).Select(l => new float[(_net.Channels(l + 1) + _net.Channels(l)) * Plane(l)]).ToArray();
            SkipGrad = Enumerable.Range(0, depth).Select(l => new float[_net.Channels(l) * Plane(l)]).ToArray();
            int scratch = layers.Select((l, i) => Math.Max(l.InChannels, l.OutChannels) * Plane(_net.LevelOf(i))).Max();
            GradA = new float[scratch];
            GradB = new float[scratch];
            GradC = new float[scratch];
            GradPadded = new float[layers.Select((l, i) => l.PaddedLength(h >> _net.LevelOf(i), w >> _net.LevelOf(i))).Max()];
            LossGradient = new float[h * w];
        }
    }
}

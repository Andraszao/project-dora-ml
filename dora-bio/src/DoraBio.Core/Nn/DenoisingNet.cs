namespace DoraBio.Core.Nn;

/// <summary>
/// Architecture of the denoiser. Layer i is a 3×3 convolution with dilation <c>Dilations[i]</c>.
/// The first layer maps 1 → Features channels, middle layers Features → Features, the last Features → 1.
/// </summary>
/// <param name="Features">Channels in every hidden layer. More features means more capacity and proportionally slower runs.</param>
/// <param name="Dilations">
/// One entry per layer (at least 2). Layer i adds <c>Dilations[i]</c> pixels of context on each side, so a
/// 1-2-4-8-4-2-1 schedule sees 22 px in every direction (a 45 px receptive field) with only 7 layers.
/// </param>
public sealed record NetConfig(int Features, int[] Dilations)
{
    /// <summary>24 features, 7 layers, receptive field 45×45 px, about 26.5k parameters.</summary>
    public static NetConfig Default => new(24, [1, 2, 4, 8, 4, 2, 1]);

    /// <summary>Number of convolution layers (= <c>Dilations.Length</c>).</summary>
    public int LayerCount => Dilations.Length;

    /// <summary>Pixels of context each output pixel sees on each side, i.e. the receptive field is 2r+1.</summary>
    /// <remarks>Also the tile margin <see cref="TiledInference"/> needs for exact results.</remarks>
    public int ReceptiveRadius => Dilations.Sum();

    /// <summary>Throws if the configuration can't describe a network (non-positive sizes, fewer than 2 layers).</summary>
    public void Validate()
    {
        if (Features <= 0) throw new ArgumentOutOfRangeException(nameof(Features), "Features must be positive.");
        if (Dilations is null || Dilations.Length < 2) throw new ArgumentException("Need at least 2 layers.", nameof(Dilations));
        if (Dilations.Any(d => d <= 0)) throw new ArgumentException("Dilations must be positive.", nameof(Dilations));
    }

    /// <summary>Human-readable summary, e.g. "7 conv layers × 24 features, dilations [1,2,4,8,4,2,1], receptive field 45px".</summary>
    public override string ToString() =>
        $"{LayerCount} conv layers × {Features} features, dilations [{string.Join(",", Dilations)}], receptive field {2 * ReceptiveRadius + 1}px";

    /// <summary>Equal when the feature count and every dilation match.</summary>
    /// <remarks>
    /// Records compare array members by reference, which would make two identical configs unequal.
    /// Weight loading and <see cref="DenoisingNet.CopyParametersFrom"/> rely on content equality.
    /// </remarks>
    public bool Equals(NetConfig? other) =>
        other is not null && Features == other.Features && Dilations.AsSpan().SequenceEqual(other.Dilations);

    /// <summary>Content-based hash, consistent with <see cref="Equals(NetConfig?)"/>.</summary>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Features);
        foreach (int d in Dilations) hash.Add(d);
        return hash.ToHashCode();
    }
}

/// <summary>
/// Fully-convolutional residual denoiser (DnCNN-style): the network predicts the noise, and the
/// denoised image is <c>input − predictedNoise</c>. Hidden layers use ReLU; the last layer is linear.
///
/// Because it is fully convolutional it works on any image size, and tiled inference with
/// margin ≥ <see cref="NetConfig.ReceptiveRadius"/> matches whole-image inference exactly.
/// The network is single-threaded per sample; callers get parallelism by running samples or tiles
/// concurrently, each with its own <see cref="Workspace"/>.
///
/// <para>Why predict the noise rather than the clean image? The noise is small, zero-mean and statistically
/// simple, while the image can be anything. The identity ("change nothing") is a good starting point, and
/// residual learning makes it the default: if the network outputs 0, the input passes through unchanged.</para>
/// </summary>
public sealed class DenoisingNet
{
    private readonly ConvLayer[] _layers;

    /// <summary>The architecture these parameters belong to.</summary>
    public NetConfig Config { get; }

    /// <summary>All weights and biases, layer by layer. Mutated in place by the optimizer.</summary>
    public float[] Parameters { get; }

    /// <summary>Total number of learnable values (weights + biases).</summary>
    public int ParameterCount => Parameters.Length;

    internal IReadOnlyList<ConvLayer> Layers => _layers;

    private DenoisingNet(NetConfig config, float[]? parameters)
    {
        config.Validate();
        Config = config;

        int offset = 0;
        _layers = new ConvLayer[config.LayerCount];
        for (int i = 0; i < config.LayerCount; i++)
        {
            int inC = i == 0 ? 1 : config.Features;
            int outC = i == config.LayerCount - 1 ? 1 : config.Features;
            _layers[i] = new ConvLayer(inC, outC, config.Dilations[i], offset);
            offset += _layers[i].ParameterCount;
        }

        if (parameters is not null && parameters.Length != offset)
            throw new ArgumentException($"Architecture needs {offset} parameters but {parameters.Length} were given.");
        Parameters = parameters ?? new float[offset];
    }

    /// <summary>How many parameters a network with this architecture has (used to validate weight files).</summary>
    public static int CountParameters(NetConfig config) => new DenoisingNet(config, null).ParameterCount;

    /// <summary>
    /// He-uniform initialization (bound √(6/fanIn), giving variance 2/fanIn for ReLU), biases zero.
    /// The last layer is scaled by 0.1 so an untrained network starts close to the identity
    /// (predicted noise ≈ 0), which makes early training stable.
    /// </summary>
    public static DenoisingNet Create(NetConfig config, int seed = 42)
    {
        var net = new DenoisingNet(config, null);
        var rng = new Random(seed);
        for (int i = 0; i < net._layers.Length; i++)
        {
            var layer = net._layers[i];
            float bound = MathF.Sqrt(6f / layer.FanIn) * (i == net._layers.Length - 1 ? 0.1f : 1f);
            for (int k = 0; k < layer.WeightCount; k++)
                net.Parameters[layer.WeightOffset + k] = (rng.NextSingle() * 2f - 1f) * bound;
        }
        return net;
    }

    /// <summary>Wraps existing parameters (e.g. from a weights file). The array is used directly, not copied.</summary>
    /// <exception cref="ArgumentException">The array length doesn't match the architecture.</exception>
    public static DenoisingNet FromParameters(NetConfig config, float[] parameters) => new(config, parameters);

    /// <summary>Deep copy: same architecture, independent parameter array.</summary>
    /// <remarks>The Godot bridge trains a clone, so the model on screen stays usable until training finishes.</remarks>
    public DenoisingNet Clone() => new(Config, (float[])Parameters.Clone());

    /// <summary>Overwrites this network's parameters with <paramref name="other"/>'s (same architecture required).</summary>
    public void CopyParametersFrom(DenoisingNet other)
    {
        if (!Config.Equals(other.Config)) throw new ArgumentException("Architectures differ.");
        other.Parameters.CopyTo(Parameters, 0);
    }

    /// <summary>Allocates scratch buffers for one thread. Reuse it across calls; it grows to the largest image seen.</summary>
    public Workspace CreateWorkspace() => new(this);

    /// <summary>denoised = input − network(input), for one [height, width] plane.</summary>
    /// <param name="input">Normalized noisy plane, row-major, <paramref name="height"/>×<paramref name="width"/> values.</param>
    /// <param name="height">Plane height in pixels.</param>
    /// <param name="width">Plane width in pixels.</param>
    /// <param name="output">Receives the denoised plane (same size). May not alias <paramref name="input"/>.</param>
    /// <param name="workspace">This thread's scratch buffers (from <see cref="CreateWorkspace"/>).</param>
    public void Predict(ReadOnlySpan<float> input, int height, int width, Span<float> output, Workspace workspace)
    {
        CheckPlane(input, height, width);
        var residual = Forward(input, height, width, workspace);
        for (int i = 0; i < residual.Length; i++)
            output[i] = input[i] - residual[i];
    }

    /// <summary>
    /// Forward and backward pass for one sample with a masked squared-error loss:
    /// loss = scale · Σ mask·(denoised − target)². Gradients are added to <paramref name="gradients"/>
    /// (parameter layout), so a mini-batch can sum per-sample contributions.
    /// An empty <paramref name="mask"/> means every pixel counts.
    /// Returns the unscaled Σ mask·(denoised − target)².
    /// </summary>
    /// <param name="input">Network input: the noisy patch, or the blind-spot-masked patch in Noise2Void mode.</param>
    /// <param name="target">What the output should match: the clean patch (supervised) or the original noisy patch (Noise2Void).</param>
    /// <param name="mask">Per-pixel loss weight (0 or 1): every pixel in supervised mode, only the blind spots in Noise2Void.</param>
    /// <param name="height">Patch height.</param>
    /// <param name="width">Patch width.</param>
    /// <param name="scale">
    /// Multiplies the loss and its gradient. The trainer passes 1 / (masked pixels in the whole batch), so the
    /// summed gradient is the gradient of the batch-mean squared error.
    /// </param>
    /// <param name="gradients">Gradient accumulator in parameter layout; this sample's gradient is added to it.</param>
    /// <param name="workspace">This thread's scratch buffers.</param>
    public double AccumulateGradients(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> target,
        ReadOnlySpan<float> mask,
        int height,
        int width,
        float scale,
        Span<float> gradients,
        Workspace workspace)
    {
        CheckPlane(input, height, width);
        CheckPlane(target, height, width);
        if (!mask.IsEmpty) CheckPlane(mask, height, width);
        if (gradients.Length != ParameterCount) throw new ArgumentException("Gradient buffer has the wrong length.");

        var residual = Forward(input, height, width, workspace);
        var gradResidual = workspace.LossGradient.AsSpan(0, residual.Length);

        // Loss gradient with respect to the network's raw output (the predicted noise r):
        //   denoised = input − r,  loss_i = scale · m_i · (denoised_i − target_i)²
        //   ∂loss_i/∂r_i = scale · m_i · 2 · (denoised_i − target_i) · (−1)
        // Pixels outside the mask contribute nothing, which is what makes Noise2Void's blind spots work.
        double sse = 0;
        for (int i = 0; i < residual.Length; i++)
        {
            float m = mask.IsEmpty ? 1f : mask[i];
            if (m == 0f) { gradResidual[i] = 0f; continue; }
            float diff = input[i] - residual[i] - target[i];
            sse += m * diff * (double)diff;
            gradResidual[i] = -2f * scale * m * diff;
        }

        Backward(height, width, gradResidual, gradients, workspace);
        return sse;
    }

    /// <summary>
    /// Runs every layer, keeping each layer's padded input and output in the workspace for the backward pass.
    /// Returns the last layer's output: the predicted noise, one channel.
    /// </summary>
    private Span<float> Forward(ReadOnlySpan<float> input, int height, int width, Workspace ws)
    {
        ws.Ensure(height, width);
        int plane = height * width;
        ReadOnlySpan<float> current = input;

        for (int l = 0; l < _layers.Length; l++)
        {
            var layer = _layers[l];
            layer.Pad(current, height, width, ws.Padded[l]);
            var output = ws.Activations[l].AsSpan(0, layer.OutChannels * plane);
            layer.Forward(Parameters, ws.Padded[l], height, width, output);
            if (l < _layers.Length - 1) Relu(output);
            current = output;
        }
        return ws.Activations[^1].AsSpan(0, plane);
    }

    /// <summary>
    /// Backpropagates ∂loss/∂(predicted noise) through every layer, from last to first, adding each layer's
    /// weight and bias gradients to <paramref name="gradients"/>. Two buffers take turns holding "gradient at
    /// this layer's output" and "gradient at its input", so nothing is allocated per call.
    /// </summary>
    /// <remarks>
    /// Unlike v1, the weights are not touched here. All gradients are computed with the forward-pass weights,
    /// and the optimizer applies them afterwards, once per mini-batch.
    /// </remarks>
    private void Backward(int height, int width, ReadOnlySpan<float> gradResidual, Span<float> gradients, Workspace ws)
    {
        int plane = height * width;
        float[] gradOut = ws.GradA, gradIn = ws.GradB;
        gradResidual.CopyTo(gradOut);

        for (int l = _layers.Length - 1; l >= 0; l--)
        {
            var layer = _layers[l];
            var gOut = gradOut.AsSpan(0, layer.OutChannels * plane);

            if (l == 0)
            {
                // The input is data, not a parameter, so its gradient is not needed.
                layer.Backward(Parameters, ws.Padded[l], gOut, height, width, gradients, Span<float>.Empty);
                break;
            }

            layer.Backward(Parameters, ws.Padded[l], gOut, height, width, gradients, ws.GradPadded);
            var gIn = gradIn.AsSpan(0, layer.InChannels * plane);
            layer.Unpad(ws.GradPadded, height, width, gIn);

            // Back through the previous layer's ReLU: relu'(z) is 1 where the unit was active and 0 where it
            // was clipped. The stored activation is relu(z), so "activation > 0" is the same test.
            var previousActivation = ws.Activations[l - 1].AsSpan(0, gIn.Length);
            for (int i = 0; i < gIn.Length; i++)
                if (previousActivation[i] <= 0f) gIn[i] = 0f;

            (gradOut, gradIn) = (gradIn, gradOut);
        }
    }

    private static void Relu(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            if (values[i] < 0f) values[i] = 0f;
    }

    private static void CheckPlane(ReadOnlySpan<float> plane, int height, int width)
    {
        if (height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (plane.Length != height * width)
            throw new ArgumentException($"Expected a {width}x{height} plane ({height * width} values), got {plane.Length}.");
    }

    /// <summary>
    /// Scratch buffers for one forward/backward pass. Not thread-safe: use one per thread.
    /// Grows automatically to the largest image size it has seen.
    /// </summary>
    public sealed class Workspace
    {
        private readonly DenoisingNet _net;
        private int _capacityPixels;
        private int _capacityPadded;

        /// <summary>Per layer: that layer's zero-padded input (needed again for weight gradients).</summary>
        internal float[][] Padded { get; private set; } = [];

        /// <summary>Per layer: that layer's output after ReLU (the last layer's is the raw predicted noise).</summary>
        internal float[][] Activations { get; private set; } = [];

        /// <summary>Ping-pong buffer for gradients flowing backward (Features × pixels).</summary>
        internal float[] GradA { get; private set; } = [];

        /// <summary>The other ping-pong buffer.</summary>
        internal float[] GradB { get; private set; } = [];

        /// <summary>A layer's input gradient in padded layout, before <see cref="ConvLayer.Unpad"/>.</summary>
        internal float[] GradPadded { get; private set; } = [];

        /// <summary>∂loss/∂(predicted noise), one value per pixel.</summary>
        internal float[] LossGradient { get; private set; } = [];

        internal Workspace(DenoisingNet net) => _net = net;

        /// <summary>Grows the buffers if this image is larger than any seen before; otherwise does nothing.</summary>
        internal void Ensure(int height, int width)
        {
            int pixels = height * width;
            int padded = _net._layers.Max(l => l.PaddedLength(height, width));
            if (pixels <= _capacityPixels && padded <= _capacityPadded) return;

            _capacityPixels = Math.Max(_capacityPixels, pixels);
            _capacityPadded = Math.Max(_capacityPadded, padded);
            int features = _net.Config.Features;

            Padded = _net._layers.Select(_ => new float[_capacityPadded]).ToArray();
            Activations = _net._layers.Select(l => new float[l.OutChannels * _capacityPixels]).ToArray();
            GradA = new float[features * _capacityPixels];
            GradB = new float[features * _capacityPixels];
            GradPadded = new float[_capacityPadded];
            LossGradient = new float[_capacityPixels];
        }
    }
}

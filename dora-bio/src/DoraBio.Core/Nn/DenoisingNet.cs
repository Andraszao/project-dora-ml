namespace DoraBio.Core.Nn;

/// <summary>
/// Architecture of the denoiser. Layer i is a 3×3 convolution with dilation <c>Dilations[i]</c>.
/// The first layer maps 1 → Features channels, middle layers Features → Features, the last Features → 1.
/// </summary>
public sealed record NetConfig(int Features, int[] Dilations)
{
    /// <summary>24 features, 7 layers, receptive field 45×45 px, about 26.5k parameters.</summary>
    public static NetConfig Default => new(24, [1, 2, 4, 8, 4, 2, 1]);

    public int LayerCount => Dilations.Length;

    /// <summary>Pixels of context each output pixel sees on each side, i.e. the receptive field is 2r+1.</summary>
    public int ReceptiveRadius => Dilations.Sum();

    public void Validate()
    {
        if (Features <= 0) throw new ArgumentOutOfRangeException(nameof(Features), "Features must be positive.");
        if (Dilations is null || Dilations.Length < 2) throw new ArgumentException("Need at least 2 layers.", nameof(Dilations));
        if (Dilations.Any(d => d <= 0)) throw new ArgumentException("Dilations must be positive.", nameof(Dilations));
    }

    public override string ToString() =>
        $"{LayerCount} conv layers × {Features} features, dilations [{string.Join(",", Dilations)}], receptive field {2 * ReceptiveRadius + 1}px";

    // Records compare arrays by reference; compare by content instead.
    public bool Equals(NetConfig? other) =>
        other is not null && Features == other.Features && Dilations.AsSpan().SequenceEqual(other.Dilations);

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
/// </summary>
public sealed class DenoisingNet
{
    private readonly ConvLayer[] _layers;

    public NetConfig Config { get; }

    /// <summary>All weights and biases, layer by layer. Mutated in place by the optimizer.</summary>
    public float[] Parameters { get; }

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

    public static DenoisingNet FromParameters(NetConfig config, float[] parameters) => new(config, parameters);

    public DenoisingNet Clone() => new(Config, (float[])Parameters.Clone());

    public void CopyParametersFrom(DenoisingNet other)
    {
        if (!Config.Equals(other.Config)) throw new ArgumentException("Architectures differ.");
        other.Parameters.CopyTo(Parameters, 0);
    }

    public Workspace CreateWorkspace() => new(this);

    /// <summary>denoised = input − network(input), for one [height, width] plane.</summary>
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

        double sse = 0;
        for (int i = 0; i < residual.Length; i++)
        {
            float m = mask.IsEmpty ? 1f : mask[i];
            if (m == 0f) { gradResidual[i] = 0f; continue; }
            float diff = input[i] - residual[i] - target[i];
            sse += m * diff * (double)diff;
            // d/dResidual of (input − residual − target)² = −2·diff
            gradResidual[i] = -2f * scale * m * diff;
        }

        Backward(height, width, gradResidual, gradients, workspace);
        return sse;
    }

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

            // Back through the previous layer's ReLU.
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

        internal float[][] Padded { get; private set; } = [];
        internal float[][] Activations { get; private set; } = [];
        internal float[] GradA { get; private set; } = [];
        internal float[] GradB { get; private set; } = [];
        internal float[] GradPadded { get; private set; } = [];
        internal float[] LossGradient { get; private set; } = [];

        internal Workspace(DenoisingNet net) => _net = net;

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

namespace DoraBio.Core.Nn;

/// <summary>Which network family a <see cref="NetConfig"/> describes.</summary>
public enum NetArchitecture
{
    /// <summary>A stack of dilated 3×3 convolutions at full resolution (DnCNN-style). Exact to tile at any offset.</summary>
    Dilated,

    /// <summary>
    /// A U-Net: two 3×3 convolutions per level, 2×2 average pooling down, nearest upsampling up, and skip connections
    /// that concatenate each encoder level onto the decoder at the same resolution.
    /// </summary>
    UNet,
}

/// <summary>What a network is told about each frame besides its pixels.</summary>
public enum NetConditioning
{
    /// <summary>Nothing: the network infers everything from the pixels it sees.</summary>
    None,

    /// <summary>
    /// The frame's estimated noise level (<see cref="NoiseConditioning"/>), applied as FiLM modulation of every hidden
    /// convolution (quality pass Q13; see DenoisingNet.Film.cs).
    /// </summary>
    NoiseLevel,
}

/// <summary>
/// Architecture of the denoiser.
/// <list type="bullet">
/// <item><b>Dilated</b> (the default): layer i is a 3×3 convolution with dilation <c>Dilations[i]</c>. The first layer
/// maps 1 → Features channels, middle layers Features → Features, the last Features → 1.</item>
/// <item><b>U-Net</b> (<see cref="UNet"/>): <see cref="Depth"/> pooling steps; level l has Features · 2ˡ channels.
/// <c>Dilations</c> is empty.</item>
/// </list>
/// </summary>
/// <param name="Features">
/// Dilated: channels in every hidden layer. U-Net: channels at full resolution (doubling at each level down).
/// More features means more capacity and proportionally slower runs.
/// </param>
/// <param name="Dilations">
/// Dilated only: one entry per layer (at least 2). Layer i adds <c>Dilations[i]</c> pixels of context on each side, so a
/// 1-2-4-8-4-2-1 schedule sees 22 px in every direction (a 45 px receptive field) with only 7 layers.
/// </param>
public sealed record NetConfig(int Features, int[] Dilations)
{
    /// <summary>24 features, 7 layers, receptive field 45×45 px, about 26.5k parameters.</summary>
    public static NetConfig Default => new(24, [1, 2, 4, 8, 4, 2, 1]);

    /// <summary>Dilated (default) or U-Net.</summary>
    public NetArchitecture Architecture { get; init; } = NetArchitecture.Dilated;

    /// <summary>U-Net only: number of 2×2 pooling steps (1 to 4). Image sides are padded to multiples of 2^Depth.</summary>
    public int Depth { get; init; }

    /// <summary>None (the default), or <see cref="NetConditioning.NoiseLevel"/> for a noise-conditioned network.</summary>
    public NetConditioning Conditioning { get; init; }

    /// <summary>True if the network needs a per-frame condition (see <see cref="NoiseConditioning"/>).</summary>
    public bool IsConditioned => Conditioning != NetConditioning.None;

    /// <summary>A U-Net with <paramref name="features"/> channels at full resolution and <paramref name="depth"/> levels below it.</summary>
    /// <remarks>16 features and depth 2 cost about as much compute per pixel as <see cref="Default"/>, with 4.5× the parameters.</remarks>
    public static NetConfig UNet(int features = 16, int depth = 2) =>
        new(features, []) { Architecture = NetArchitecture.UNet, Depth = depth };

    /// <summary>Number of convolution layers: <c>Dilations.Length</c>, or 4·Depth + 3 for a U-Net.</summary>
    public int LayerCount => Architecture == NetArchitecture.UNet ? 4 * Depth + 3 : Dilations.Length;

    /// <summary>Pixels of context each output pixel sees on each side, at most, i.e. the receptive field is 2r+1.</summary>
    /// <remarks>
    /// With <see cref="TileAlignment"/>, this sets the margin <see cref="TiledInference"/> needs for exact results.
    /// U-Net: a 3×3 convolution at level l reaches one level-l cell (2ˡ pixels) further on each side, and so can each
    /// pooling and upsampling step, which is a safe upper bound: Σ over those layers of 2ˡ.
    /// </remarks>
    public int ReceptiveRadius
    {
        get
        {
            if (Architecture != NetArchitecture.UNet) return Dilations.Sum();
            int r = 1;   // the final full-resolution convolution
            for (int l = 0; l < Depth; l++) r += (2 + 2) * (1 << l) + (1 << l) + (1 << l);   // encoder + decoder convs, pool, upsample
            return r + 2 * (1 << Depth);                                                       // the bottleneck's two convs
        }
    }

    /// <summary>
    /// Tile origins must be multiples of this (1 for the dilated stack, 2^Depth for a U-Net), because pooling makes a
    /// U-Net shift-equivariant only for shifts that are multiples of 2^Depth.
    /// </summary>
    public int TileAlignment => Architecture == NetArchitecture.UNet ? 1 << Depth : 1;

    /// <summary>Throws if the configuration can't describe a network (non-positive sizes, too few layers or levels).</summary>
    public void Validate()
    {
        if (Features <= 0) throw new ArgumentOutOfRangeException(nameof(Features), "Features must be positive.");
        if (!Enum.IsDefined(Conditioning)) throw new ArgumentOutOfRangeException(nameof(Conditioning));
        if (Architecture == NetArchitecture.UNet)
        {
            if (Depth is < 1 or > 4) throw new ArgumentOutOfRangeException(nameof(Depth), "U-Net depth must be 1 to 4.");
            if (Dilations is null || Dilations.Length != 0) throw new ArgumentException("A U-Net has no dilations.", nameof(Dilations));
            return;
        }
        if (Dilations is null || Dilations.Length < 2) throw new ArgumentException("Need at least 2 layers.", nameof(Dilations));
        if (Dilations.Any(d => d <= 0)) throw new ArgumentException("Dilations must be positive.", nameof(Dilations));
    }

    /// <summary>Human-readable summary, e.g. "7 conv layers × 24 features, dilations [1,2,4,8,4,2,1], receptive field 45px".</summary>
    public override string ToString() => (Architecture == NetArchitecture.UNet
        ? $"U-Net, depth {Depth}, {string.Join("→", Enumerable.Range(0, Depth + 1).Select(l => Features << l))} features, " +
          $"{LayerCount} conv layers, receptive field up to {2 * ReceptiveRadius + 1}px"
        : $"{LayerCount} conv layers × {Features} features, dilations [{string.Join(",", Dilations)}], receptive field {2 * ReceptiveRadius + 1}px")
        + (IsConditioned ? ", noise-conditioned (FiLM)" : "");

    /// <summary>Equal when the architecture, feature count, depth, conditioning and every dilation match.</summary>
    /// <remarks>
    /// Records compare array members by reference, which would make two identical configs unequal.
    /// Weight loading and <see cref="DenoisingNet.CopyParametersFrom"/> rely on content equality.
    /// </remarks>
    public bool Equals(NetConfig? other) =>
        other is not null && Architecture == other.Architecture && Features == other.Features && Depth == other.Depth
        && Conditioning == other.Conditioning && Dilations.AsSpan().SequenceEqual(other.Dilations);

    /// <summary>Content-based hash, consistent with <see cref="Equals(NetConfig?)"/>.</summary>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Architecture);
        hash.Add(Features);
        hash.Add(Depth);
        hash.Add(Conditioning);
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
///
/// <para>The same class runs both architectures in <see cref="NetConfig"/>: the dilated stack (this file) and the
/// U-Net (DenoisingNet.UNet.cs). Callers never need to know which one they hold. A U-Net works on sides that are
/// multiples of 2^Depth; other sizes are zero-padded at the bottom and right inside <see cref="Predict"/> and
/// <see cref="AccumulateGradients"/>, with the padding excluded from the loss.</para>
/// </summary>
public sealed partial class DenoisingNet
{
    private readonly ConvLayer[] _layers;

    /// <summary>The architecture these parameters belong to.</summary>
    public NetConfig Config { get; }

    /// <summary>All weights and biases, layer by layer. Mutated in place by the optimizer.</summary>
    public float[] Parameters { get; }

    /// <summary>Total number of learnable values (weights + biases).</summary>
    public int ParameterCount => Parameters.Length;

    internal IReadOnlyList<ConvLayer> Layers => _layers;

    /// <summary>
    /// Tests only: run every convolution through the reference row kernels instead of the register-blocked ones, to
    /// prove the two give bit-identical networks (brick 23). Per instance, so parallel tests can't affect each other.
    /// </summary>
    internal bool ReferenceKernels { get; set; }

    // The kernels read ws.Active: Parameters itself, or a conditioned network's effective parameters for this call.
    private void ConvForward(ConvLayer layer, ReadOnlySpan<float> padded, int height, int width, Span<float> output, Workspace ws)
    {
        if (ReferenceKernels) layer.ForwardReference(ws.Active, padded, height, width, output);
        else layer.Forward(ws.Active, padded, height, width, output);
    }

    private void ConvBackward(ConvLayer layer, ReadOnlySpan<float> padded, ReadOnlySpan<float> gradOutput, int height, int width,
        Span<float> gradients, Span<float> gradPadded, Workspace ws)
    {
        if (ReferenceKernels) layer.BackwardReference(ws.Active, padded, gradOutput, height, width, gradients, gradPadded);
        else layer.Backward(ws.Active, padded, gradOutput, height, width, gradients, gradPadded);
    }

    private DenoisingNet(NetConfig config, float[]? parameters)
    {
        config.Validate();
        Config = config;

        var shapes = config.Architecture == NetArchitecture.UNet ? UNetLayerShapes(config) : DilatedLayerShapes(config);
        int offset = 0;
        _layers = new ConvLayer[shapes.Count];
        for (int i = 0; i < shapes.Count; i++)
        {
            _layers[i] = new ConvLayer(shapes[i].In, shapes[i].Out, shapes[i].Dilation, offset);
            offset += _layers[i].ParameterCount;
        }
        offset += SetUpConditioning(offset);

        if (parameters is not null && parameters.Length != offset)
            throw new ArgumentException($"Architecture needs {offset} parameters but {parameters.Length} were given.");
        Parameters = parameters ?? new float[offset];
    }

    /// <summary>Dilated stack: 1 → Features → … → Features → 1 channels, one layer per dilation.</summary>
    private static List<(int In, int Out, int Dilation)> DilatedLayerShapes(NetConfig config) =>
        Enumerable.Range(0, config.Dilations.Length)
            .Select(i => (i == 0 ? 1 : config.Features, i == config.Dilations.Length - 1 ? 1 : config.Features, config.Dilations[i]))
            .ToList();

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
        net.InitializeConditioning(rng);   // after the convolutions, so they match a plain network with the same seed
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
    /// <param name="condition">
    /// Noise-conditioned networks only: the whole frame's <see cref="NoiseConditioning.Of"/> (required; ignored otherwise).
    /// </param>
    public void Predict(ReadOnlySpan<float> input, int height, int width, Span<float> output, Workspace workspace,
        float condition = float.NaN)
    {
        CheckPlane(input, height, width);
        int a = Config.TileAlignment;
        if (height % a == 0 && width % a == 0)
        {
            var residual = Forward(input, height, width, workspace, condition);
            for (int i = 0; i < residual.Length; i++)
                output[i] = input[i] - residual[i];
            return;
        }

        // U-Net on a size that isn't a multiple of 2^Depth: zero-pad at the bottom and right, then crop.
        int ph = RoundUp(height, a), pw = RoundUp(width, a);
        var padded = workspace.AlignedInput(ph * pw);
        PadTo(input, height, width, padded, pw);
        var paddedResidual = Forward(padded, ph, pw, workspace, condition);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                output[y * width + x] = input[y * width + x] - paddedResidual[y * pw + x];
    }

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

    /// <summary>Copies an h×w plane into the top-left of a zeroed plane that is <paramref name="paddedWidth"/> wide.</summary>
    private static void PadTo(ReadOnlySpan<float> plane, int height, int width, Span<float> padded, int paddedWidth)
    {
        padded.Clear();
        for (int y = 0; y < height; y++)
            plane.Slice(y * width, width).CopyTo(padded.Slice(y * paddedWidth, width));
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
    /// <param name="condition">
    /// Noise-conditioned networks only: the condition of the frame the sample comes from (required; ignored otherwise).
    /// </param>
    public double AccumulateGradients(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> target,
        ReadOnlySpan<float> mask,
        int height,
        int width,
        float scale,
        Span<float> gradients,
        Workspace workspace,
        float condition = float.NaN)
    {
        CheckPlane(input, height, width);
        CheckPlane(target, height, width);
        if (!mask.IsEmpty) CheckPlane(mask, height, width);
        if (gradients.Length != ParameterCount) throw new ArgumentException("Gradient buffer has the wrong length.");

        int a = Config.TileAlignment;
        if (height % a != 0 || width % a != 0)
        {
            // Pad input, target and mask; the padded pixels get mask 0, so they add nothing to the loss or gradients.
            int ph = RoundUp(height, a), pw = RoundUp(width, a);
            var (bi, bt, bm) = workspace.AlignedTraining(ph * pw);
            Span<float> pi = bi.AsSpan(0, ph * pw), pt = bt.AsSpan(0, ph * pw), pm = bm.AsSpan(0, ph * pw);
            PadTo(input, height, width, pi, pw);
            PadTo(target, height, width, pt, pw);
            if (mask.IsEmpty)
            {
                pm.Clear();
                for (int y = 0; y < height; y++) pm.Slice(y * pw, width).Fill(1f);
            }
            else PadTo(mask, height, width, pm, pw);
            return AccumulateGradients(pi, pt, pm, ph, pw, scale, gradients, workspace, condition);
        }

        var residual = Forward(input, height, width, workspace, condition);
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

        if (!Config.IsConditioned)
        {
            Backward(height, width, gradResidual, gradients, workspace);
            return sse;
        }
        // Conditioned: backpropagate to the effective parameters, then on to the real ones (DenoisingNet.Film.cs).
        var effectiveGradient = workspace.EffectiveGradient.AsSpan();
        effectiveGradient.Clear();
        Backward(height, width, gradResidual, effectiveGradient, workspace);
        ConditioningBackward(effectiveGradient, gradients, workspace);
        return sse;
    }

    /// <summary>
    /// Runs every layer, keeping each layer's padded input and output in the workspace for the backward pass.
    /// Returns the last layer's output: the predicted noise, one channel.
    /// </summary>
    private Span<float> Forward(ReadOnlySpan<float> input, int height, int width, Workspace ws, float condition)
    {
        ws.Active = ActiveParameters(condition, ws);
        if (Config.Architecture == NetArchitecture.UNet) return ForwardUNet(input, height, width, ws);
        ws.Ensure(height, width);
        int plane = height * width;
        ReadOnlySpan<float> current = input;

        for (int l = 0; l < _layers.Length; l++)
        {
            var layer = _layers[l];
            layer.Pad(current, height, width, ws.Padded[l]);
            var output = ws.Activations[l].AsSpan(0, layer.OutChannels * plane);
            ConvForward(layer, ws.Padded[l], height, width, output, ws);
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
        if (Config.Architecture == NetArchitecture.UNet) { BackwardUNet(height, width, gradResidual, gradients, ws); return; }
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
                ConvBackward(layer, ws.Padded[l], gOut, height, width, gradients, Span<float>.Empty, ws);
                break;
            }

            ConvBackward(layer, ws.Padded[l], gOut, height, width, gradients, ws.GradPadded, ws);
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
    public sealed partial class Workspace
    {
        private readonly DenoisingNet _net;
        private int _capacityPixels;
        private int _capacityPadded;
        private float[] _alignedInput = [], _alignedTarget = [], _alignedMask = [];
        private float[] _tileIn = [], _tileOut = [];

        /// <summary>The parameters the kernels read during the current pass (set by Forward).</summary>
        internal float[] Active { get; set; } = [];

        /// <summary>
        /// Sizes every buffer for images up to <paramref name="height"/> × <paramref name="width"/> (rounded up to the
        /// tile alignment, as <see cref="Predict"/> pads U-Net inputs). Buffers only grow, so calling this once for the
        /// largest tile means a workspace reused across tiles of several sizes, or across frames, allocates nothing
        /// more. Every buffer is cleared or overwritten before it's read, so its size never changes a result.
        /// </summary>
        internal void Reserve(int height, int width)
        {
            int align = _net.Config.TileAlignment;
            Ensure((height + align - 1) / align * align, (width + align - 1) / align * align);
        }

        /// <summary>Input and output windows for one tile of <see cref="TiledInference"/>, at least <paramref name="length"/> long.</summary>
        internal (float[] Input, float[] Output) TileBuffers(int length)
        {
            if (_tileIn.Length < length) (_tileIn, _tileOut) = (new float[length], new float[length]);
            return (_tileIn, _tileOut);
        }

        /// <summary>A zero-padding buffer for <see cref="Predict"/> on sizes that aren't multiples of 2^Depth.</summary>
        internal Span<float> AlignedInput(int length)
        {
            if (_alignedInput.Length < length) _alignedInput = new float[length];
            return _alignedInput.AsSpan(0, length);
        }

        /// <summary>Padding buffers for <see cref="AccumulateGradients"/> on unaligned sizes (input, target, mask), at least <paramref name="length"/> long.</summary>
        internal (float[] Input, float[] Target, float[] Mask) AlignedTraining(int length)
        {
            AlignedInput(length);
            if (_alignedTarget.Length < length)
            {
                _alignedTarget = new float[length];
                _alignedMask = new float[length];
            }
            return (_alignedInput, _alignedTarget, _alignedMask);
        }

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
            if (_net.Config.Architecture == NetArchitecture.UNet) { EnsureUNet(height, width); return; }
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

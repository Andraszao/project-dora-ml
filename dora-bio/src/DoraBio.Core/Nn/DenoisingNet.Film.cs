namespace DoraBio.Core.Nn;

/// <summary>
/// Noise conditioning (quality pass Q13, <see cref="NetConditioning.NoiseLevel"/>): every hidden convolution is
/// modulated per channel by one number per frame, the condition s from <see cref="NoiseConditioning"/>.
///
/// <para>A small network shared by all layers turns s into a scale γ and a shift β for every channel (FiLM):</para>
/// <code>
/// h = relu(W₁ s + b₁)                          H = 16 hidden units
/// o = W₂ h + b₂                                2C outputs; C = the channels of all hidden convolutions together
/// γ_c = 1 + o[c],   β_c = o[C + c]             (each layer owns a run of C)
/// z_c = γ_c · conv(x)_c + β_c, then ReLU       the final 1-channel convolution is not modulated
/// </code>
/// <para><b>Folded into the convolution.</b> γ·(W∗x + b) + β = (γW)∗x + (γb + β), so a forward pass builds
/// <i>effective</i> parameters once per call and runs the ordinary kernels on them. The backward pass collects the
/// gradient with respect to the effective parameters, g′, and then applies the chain rule:</para>
/// <code>
/// ∂L/∂W_c = γ_c g′_W,c      ∂L/∂b_c = γ_c g′_b,c      ∂L/∂γ_c = Σ_k W_c,k g′_W,c,k + b_c g′_b,c      ∂L/∂β_c = g′_b,c
/// ∂L/∂o → ∂L/∂W₂ = ∂L/∂o · hᵀ,  ∂L/∂b₂ = ∂L/∂o,  ∂L/∂h = W₂ᵀ ∂L/∂o,  then through the ReLU to W₁ (· s) and b₁
/// </code>
/// <para>So every invariant of the plain network holds: the kernels and their bit-identical blocked versions are
/// untouched, the activation buffers are the same, and with W₂ = b₂ = 0 (how <see cref="Create"/> starts them) γ = 1 and
/// β = 0 exactly, so an untrained conditioned network computes bit for bit what the plain one with the same seed does.
/// Building the effective parameters costs one multiply per weight per call, nothing next to the convolutions.</para>
/// <para>Layout: the generator's parameters follow all the convolutions' in <see cref="Parameters"/>:
/// W₁[H], b₁[H], W₂[2C × H] (row j holds output j's weights), b₂[2C].</para>
/// </summary>
public sealed partial class DenoisingNet
{
    /// <summary>Hidden units of the conditioning network (fixed; stored in weights files for checking).</summary>
    public const int FilmHidden = 16;

    private int _convParameters;      // parameters of all convolutions; the generator's start here
    private int _filmChannels;        // C
    private int[] _filmOffset = [];   // per layer: its first channel in o, or −1 if it isn't modulated

    private int FilmW1 => _convParameters;
    private int FilmB1 => FilmW1 + FilmHidden;
    private int FilmW2 => FilmB1 + FilmHidden;
    private int FilmB2 => FilmW2 + 2 * _filmChannels * FilmHidden;

    /// <summary>Lays out the generator after the convolutions; returns its parameter count (0 when unconditioned).</summary>
    private int SetUpConditioning(int convParameters)
    {
        _convParameters = convParameters;
        _filmOffset = new int[_layers.Length];
        Array.Fill(_filmOffset, -1);
        if (!Config.IsConditioned) return 0;
        for (int l = 0; l < _layers.Length - 1; l++)
        {
            _filmOffset[l] = _filmChannels;
            _filmChannels += _layers[l].OutChannels;
        }
        return 2 * FilmHidden + 2 * _filmChannels * FilmHidden + 2 * _filmChannels;
    }

    /// <summary>Generator initialization: W₁ He-uniform for one input (±√6), b₁ uniform in ±1 so the ReLU kinks spread
    /// over the range s takes, and W₂ = b₂ = 0 so training starts from the plain network.</summary>
    private void InitializeConditioning(Random rng)
    {
        if (!Config.IsConditioned) return;
        for (int i = 0; i < FilmHidden; i++)
        {
            Parameters[FilmW1 + i] = (rng.NextSingle() * 2f - 1f) * MathF.Sqrt(6f);
            Parameters[FilmB1 + i] = rng.NextSingle() * 2f - 1f;
        }
    }

    /// <summary>The parameters the kernels should read for this condition: <see cref="Parameters"/> itself for a plain
    /// network, otherwise the workspace's effective parameters, built here (the generator's activations are kept for
    /// <see cref="ConditioningBackward"/>).</summary>
    internal float[] ActiveParameters(float condition, Workspace ws)
    {
        if (!Config.IsConditioned) return Parameters;
        if (float.IsNaN(condition))
            throw new ArgumentException("This network is noise-conditioned: pass the frame's condition (NoiseConditioning.Of).");
        var p = Parameters;
        ws.EnsureConditioning(_convParameters, 2 * _filmChannels);
        ws.Condition = condition;
        for (int i = 0; i < FilmHidden; i++)
        {
            float pre = p[FilmW1 + i] * condition + p[FilmB1 + i];
            ws.FilmPre[i] = pre;
            ws.FilmHiddenOut[i] = pre > 0f ? pre : 0f;
        }
        for (int j = 0; j < 2 * _filmChannels; j++)
        {
            float acc = p[FilmB2 + j];
            int row = FilmW2 + j * FilmHidden;
            for (int i = 0; i < FilmHidden; i++) acc += p[row + i] * ws.FilmHiddenOut[i];
            ws.FilmOut[j] = acc;
        }

        var effective = ws.Effective;
        for (int l = 0; l < _layers.Length; l++)
        {
            var layer = _layers[l];
            int off = _filmOffset[l];
            if (off < 0)
            {
                p.AsSpan(layer.WeightOffset, layer.ParameterCount).CopyTo(effective.AsSpan(layer.WeightOffset));
                continue;
            }
            int fanIn = layer.FanIn;
            for (int oc = 0; oc < layer.OutChannels; oc++)
            {
                float gamma = 1f + ws.FilmOut[off + oc], beta = ws.FilmOut[_filmChannels + off + oc];
                int w = layer.WeightOffset + oc * fanIn, b = layer.BiasOffset + oc;
                for (int k = 0; k < fanIn; k++) effective[w + k] = gamma * p[w + k];
                effective[b] = gamma * p[b] + beta;
            }
        }
        return effective;
    }

    /// <summary>Chain rule from the effective parameters' gradient <paramref name="effectiveGradient"/> to the real
    /// parameters (convolutions and generator), added to <paramref name="gradients"/>. Uses the activations that
    /// <see cref="ActiveParameters"/> stored for this sample.</summary>
    internal void ConditioningBackward(ReadOnlySpan<float> effectiveGradient, Span<float> gradients, Workspace ws)
    {
        var p = Parameters;
        var dOut = ws.FilmGradOut;
        for (int l = 0; l < _layers.Length; l++)
        {
            var layer = _layers[l];
            int off = _filmOffset[l];
            if (off < 0)
            {
                for (int k = layer.WeightOffset; k < layer.WeightOffset + layer.ParameterCount; k++) gradients[k] += effectiveGradient[k];
                continue;
            }
            int fanIn = layer.FanIn;
            for (int oc = 0; oc < layer.OutChannels; oc++)
            {
                float gamma = 1f + ws.FilmOut[off + oc];
                int w = layer.WeightOffset + oc * fanIn, b = layer.BiasOffset + oc;
                float dGamma = 0f;
                for (int k = 0; k < fanIn; k++)
                {
                    float g = effectiveGradient[w + k];
                    gradients[w + k] += gamma * g;
                    dGamma += p[w + k] * g;
                }
                float gb = effectiveGradient[b];
                gradients[b] += gamma * gb;
                dOut[off + oc] = dGamma + p[b] * gb;
                dOut[_filmChannels + off + oc] = gb;
            }
        }

        Span<float> dHidden = stackalloc float[FilmHidden];
        dHidden.Clear();
        for (int j = 0; j < 2 * _filmChannels; j++)
        {
            float d = dOut[j];
            gradients[FilmB2 + j] += d;
            int row = FilmW2 + j * FilmHidden;
            for (int i = 0; i < FilmHidden; i++)
            {
                gradients[row + i] += d * ws.FilmHiddenOut[i];
                dHidden[i] += p[row + i] * d;
            }
        }
        for (int i = 0; i < FilmHidden; i++)
        {
            if (ws.FilmPre[i] <= 0f) continue;   // back through the ReLU
            gradients[FilmW1 + i] += dHidden[i] * ws.Condition;
            gradients[FilmB1 + i] += dHidden[i];
        }
    }

    public sealed partial class Workspace
    {
        /// <summary>Conditioned networks: the effective convolution parameters for the current condition.</summary>
        internal float[] Effective { get; private set; } = [];

        /// <summary>Conditioned networks: the gradient with respect to <see cref="Effective"/> for one sample.</summary>
        internal float[] EffectiveGradient { get; private set; } = [];

        /// <summary>Conditioned networks: the generator's pre-activations, hidden outputs and outputs for the current condition.</summary>
        internal float[] FilmPre { get; } = new float[FilmHidden];

        /// <inheritdoc cref="FilmPre"/>
        internal float[] FilmHiddenOut { get; } = new float[FilmHidden];

        /// <inheritdoc cref="FilmPre"/>
        internal float[] FilmOut { get; private set; } = [];

        /// <summary>Conditioned networks: ∂loss/∂o, the generator outputs' gradient.</summary>
        internal float[] FilmGradOut { get; private set; } = [];

        /// <summary>The condition the effective parameters were built for.</summary>
        internal float Condition { get; set; }

        /// <summary>Allocates the conditioning buffers once.</summary>
        internal void EnsureConditioning(int convParameters, int outputs)
        {
            if (Effective.Length == convParameters) return;
            Effective = new float[convParameters];
            EffectiveGradient = new float[convParameters];
            FilmOut = new float[outputs];
            FilmGradOut = new float[outputs];
        }
    }
}

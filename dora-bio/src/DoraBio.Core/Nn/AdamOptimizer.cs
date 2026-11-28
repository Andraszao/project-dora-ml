namespace DoraBio.Core.Nn;

/// <summary>
/// Adam with bias correction and optional global-norm gradient clipping.
///
/// The moment buffers and step count persist for the optimizer's whole lifetime. This matters for online
/// adaptation: the v1 GDScript app made a fresh optimizer on every feedback click, so every update ran at
/// t = 1, where Adam reduces to <c>lr · sign(gradient)</c> and ignores how large the error is.
///
/// <para>Per parameter, for step t with gradient g (Kingma &amp; Ba, 2015):</para>
/// <code>
/// m ← β1·m + (1 − β1)·g            running mean of gradients ("momentum")
/// v ← β2·v + (1 − β2)·g²           running mean of squared gradients (per-parameter scale)
/// m̂ = m / (1 − β1ᵗ),  v̂ = v / (1 − β2ᵗ)   bias correction: m and v start at 0
/// p ← p − lr · m̂ / (√v̂ + ε)
/// </code>
/// </summary>
public sealed class AdamOptimizer
{
    private readonly float[] _m;
    private readonly float[] _v;

    /// <summary>Step size. Mutable so the trainer can decay it on plateaus without losing the moment state.</summary>
    public float LearningRate { get; set; }

    /// <summary>Decay of the gradient mean (momentum). 0.9 is the standard value.</summary>
    public float Beta1 { get; init; } = 0.9f;

    /// <summary>Decay of the squared-gradient mean. 0.999 is the standard value.</summary>
    public float Beta2 { get; init; } = 0.999f;

    /// <summary>Added to √v̂ so parameters with near-zero gradient history don't divide by zero.</summary>
    public float Epsilon { get; init; } = 1e-8f;

    /// <summary>If set, gradients are rescaled so their global L2 norm is at most this value.</summary>
    /// <remarks>
    /// Clipping the whole gradient vector keeps its direction and only limits the step length on unusually bad
    /// batches. This replaces v1's per-weight clamp of each update to ±1.0, which distorted the direction.
    /// </remarks>
    public float? ClipNorm { get; init; }

    /// <summary>Number of updates applied so far (Adam's t).</summary>
    public long Step { get; private set; }

    /// <summary>Length of the parameter vector this optimizer was created for.</summary>
    public int ParameterCount => _m.Length;

    /// <summary>Creates an optimizer with zeroed moment buffers for <paramref name="parameterCount"/> parameters.</summary>
    /// <param name="parameterCount">Must match the length of every parameter and gradient array passed to <see cref="Apply"/>.</param>
    /// <param name="learningRate">Initial step size (positive).</param>
    public AdamOptimizer(int parameterCount, float learningRate)
    {
        if (parameterCount <= 0) throw new ArgumentOutOfRangeException(nameof(parameterCount));
        if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        _m = new float[parameterCount];
        _v = new float[parameterCount];
        LearningRate = learningRate;
    }

    /// <summary>Applies one update in place. Returns the gradient norm before clipping.</summary>
    /// <param name="parameters">Updated in place.</param>
    /// <param name="gradients">∂loss/∂parameters for this step (already averaged over the mini-batch).</param>
    public double Apply(Span<float> parameters, ReadOnlySpan<float> gradients)
    {
        if (parameters.Length != _m.Length || gradients.Length != _m.Length)
            throw new ArgumentException("Parameter/gradient length does not match the optimizer.");

        double normSquared = 0;
        foreach (float g in gradients) normSquared += g * (double)g;
        double norm = Math.Sqrt(normSquared);
        float clipScale = ClipNorm is { } clip && norm > clip ? (float)(clip / norm) : 1f;

        Step++;
        float correction1 = 1f - MathF.Pow(Beta1, Step);
        float correction2 = 1f - MathF.Pow(Beta2, Step);

        for (int i = 0; i < parameters.Length; i++)
        {
            float g = gradients[i] * clipScale;
            _m[i] = Beta1 * _m[i] + (1f - Beta1) * g;
            _v[i] = Beta2 * _v[i] + (1f - Beta2) * g * g;
            float mHat = _m[i] / correction1;
            float vHat = _v[i] / correction2;
            parameters[i] -= LearningRate * mHat / (MathF.Sqrt(vHat) + Epsilon);
        }
        return norm;
    }
}

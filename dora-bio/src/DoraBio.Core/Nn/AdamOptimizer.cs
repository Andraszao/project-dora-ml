namespace DoraBio.Core.Nn;

/// <summary>
/// Adam with bias correction and optional global-norm gradient clipping.
///
/// The moment buffers and step count persist for the optimizer's whole lifetime. This matters for online
/// adaptation: the v1 GDScript app made a fresh optimizer on every feedback click, so every update ran at
/// t = 1, where Adam reduces to <c>lr · sign(gradient)</c> and ignores how large the error is.
/// </summary>
public sealed class AdamOptimizer
{
    private readonly float[] _m;
    private readonly float[] _v;

    public float LearningRate { get; set; }
    public float Beta1 { get; init; } = 0.9f;
    public float Beta2 { get; init; } = 0.999f;
    public float Epsilon { get; init; } = 1e-8f;

    /// <summary>If set, gradients are rescaled so their global L2 norm is at most this value.</summary>
    public float? ClipNorm { get; init; }

    /// <summary>Number of updates applied so far (Adam's t).</summary>
    public long Step { get; private set; }

    public int ParameterCount => _m.Length;

    public AdamOptimizer(int parameterCount, float learningRate)
    {
        if (parameterCount <= 0) throw new ArgumentOutOfRangeException(nameof(parameterCount));
        if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        _m = new float[parameterCount];
        _v = new float[parameterCount];
        LearningRate = learningRate;
    }

    /// <summary>Applies one update in place. Returns the gradient norm before clipping.</summary>
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

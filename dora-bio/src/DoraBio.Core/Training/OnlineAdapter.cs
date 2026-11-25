using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

/// <summary>
/// "Adapt to this sample": a few self-supervised Noise2Void steps on the image the user is looking at,
/// so a pretrained model can tune itself to this microscope's noise without any ground truth.
///
/// One adapter lives as long as the loaded model. Its Adam moments and step count carry over between
/// calls, unlike v1's feedback learning (see <see cref="AdamOptimizer"/>). Nothing is written to disk
/// here; the host decides when to save a checkpoint.
/// </summary>
public sealed class OnlineAdapter
{
    private readonly DenoisingNet _net;
    private readonly AdamOptimizer _optimizer;
    private readonly BatchStepper _stepper;
    private readonly Random _rng;
    private readonly float _maskRatio;

    public int PatchSize { get; }
    public long OptimizerStep => _optimizer.Step;
    public float LearningRate => _optimizer.LearningRate;

    public OnlineAdapter(DenoisingNet net, float learningRate = 1e-4f, int patchSize = 64, int batchSize = 8,
        float maskRatio = 0.02f, int seed = 7)
    {
        _net = net;
        _optimizer = new AdamOptimizer(net.ParameterCount, learningRate) { ClipNorm = 1.0f };
        _stepper = new BatchStepper(net, batchSize, patchSize);
        _rng = new Random(seed);
        _maskRatio = maskRatio;
        PatchSize = patchSize;
    }

    /// <summary>Runs <paramref name="steps"/> blind-spot steps on a normalized image; returns the loss per step.</summary>
    public IReadOnlyList<double> Adapt(GrayImage normalizedImage, int steps, CancellationToken cancellation = default)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        if (normalizedImage.Width < PatchSize || normalizedImage.Height < PatchSize)
            throw new ArgumentException($"Image must be at least {PatchSize}×{PatchSize} to adapt on.");

        var sampler = new PatchSampler([new ImagePair(normalizedImage, null, "current", "current")],
            PatchSize, TrainingMode.Noise2Void, _maskRatio, _rng);
        var losses = new List<double>(steps);
        for (int s = 0; s < steps && !cancellation.IsCancellationRequested; s++)
            losses.Add(_stepper.Step(sampler, _optimizer));
        return losses;
    }
}

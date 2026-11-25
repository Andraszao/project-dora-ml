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

    /// <summary>Side of the patches drawn from the image. The image must be at least this large.</summary>
    public int PatchSize { get; }

    /// <summary>Total adaptation steps taken so far, across all <see cref="Adapt"/> calls (Adam's t).</summary>
    public long OptimizerStep => _optimizer.Step;

    /// <summary>The adaptation step size.</summary>
    public float LearningRate => _optimizer.LearningRate;

    /// <param name="net">The model to adapt. It's modified in place, so take a clone if the original must survive.</param>
    /// <param name="learningRate">
    /// Deliberately about 10× below the training rate: adaptation should nudge a trained model toward this
    /// microscope's noise, not overwrite what it learned.
    /// </param>
    /// <param name="patchSize">Patch side; the bridge lowers it for small images.</param>
    /// <param name="batchSize">Patches per step.</param>
    /// <param name="maskRatio">Blind-spot fraction per patch.</param>
    /// <param name="seed">Seeds patch positions and masks.</param>
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
    /// <param name="normalizedImage">The sample on screen, normalized like every DORA input.</param>
    /// <param name="steps">Optimizer steps to run.</param>
    /// <param name="cancellation">Checked between steps.</param>
    /// <param name="footprint">
    /// Structured blind spots for an image whose noise is correlated (see <see cref="NoiseDiagnostics"/>), or null for
    /// plain ones. One image means one microscope, so this avoids the mixing problem RESULTS.md describes for P3.
    /// </param>
    public IReadOnlyList<double> Adapt(GrayImage normalizedImage, int steps, CancellationToken cancellation = default,
        MaskFootprint? footprint = null)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        if (normalizedImage.Width < PatchSize || normalizedImage.Height < PatchSize)
            throw new ArgumentException($"Image must be at least {PatchSize}×{PatchSize} to adapt on.");

        var sampler = new PatchSampler([new ImagePair(normalizedImage, null, "current", "current") { Footprint = footprint }],
            PatchSize, TrainingMode.Noise2Void, _maskRatio, _rng);
        var losses = new List<double>(steps);
        for (int s = 0; s < steps && !cancellation.IsCancellationRequested; s++)
            losses.Add(_stepper.Step(sampler, _optimizer));
        return losses;
    }
}

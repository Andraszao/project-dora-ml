using DoraBio.Core.Data;

namespace DoraBio.Core.Training;

public sealed record TrainerConfig
{
    public TrainingMode Mode { get; init; } = TrainingMode.Noise2Void;
    public int PatchSize { get; init; } = 64;
    public int BatchSize { get; init; } = 16;
    public int StepsPerEpoch { get; init; } = 50;
    public int MaxEpochs { get; init; } = 30;

    public float LearningRate { get; init; } = 1e-3f;
    public float MinLearningRate { get; init; } = 1e-5f;
    /// <summary>Multiplier applied when the validation score plateaus. Below 1, so the rate only decays.</summary>
    public float LearningRateDecay { get; init; } = 0.85f;
    /// <summary>Epochs without a new best score before each decay.</summary>
    public int PlateauPatience { get; init; } = 2;
    /// <summary>Epochs without a new best score before training stops.</summary>
    public int EarlyStopPatience { get; init; } = 6;
    public float? GradientClipNorm { get; init; } = 1.0f;

    /// <summary>Fraction of pixels masked per patch in Noise2Void mode.</summary>
    public float MaskRatio { get; init; } = 0.02f;
    public double ValidationFraction { get; init; } = 0.2;
    public int ValidationPatches { get; init; } = 16;
    public int Seed { get; init; } = 1234;

    /// <summary>
    /// Progressive targets are measured as PSNR gain (dB) of DORA's output over the noisy input on the
    /// validation set. That needs clean validation images; without them the milestones are skipped.
    /// Gains travel across datasets better than absolute PSNR, which depends heavily on the noise level.
    /// </summary>
    public bool EnableProgressiveTargets { get; init; } = true;
    public double InitialTargetGainDb { get; init; } = 2.0;
    public double TargetIncrementDb { get; init; } = 2.0;
    public double MaxTargetGainDb { get; init; } = 8.0;
    public int TargetHitsNeeded { get; init; } = 2;
    public bool StopOnMastery { get; init; } = true;

    public void Validate()
    {
        if (PatchSize < 16) throw new ArgumentOutOfRangeException(nameof(PatchSize), "Patch size must be at least 16.");
        if (BatchSize < 1 || StepsPerEpoch < 1 || MaxEpochs < 1) throw new ArgumentOutOfRangeException(nameof(BatchSize));
        if (LearningRate <= 0 || MinLearningRate <= 0) throw new ArgumentOutOfRangeException(nameof(LearningRate));
        if (!(LearningRateDecay > 0 && LearningRateDecay < 1)) throw new ArgumentOutOfRangeException(nameof(LearningRateDecay));
        if (PlateauPatience < 1 || EarlyStopPatience < 1) throw new ArgumentOutOfRangeException(nameof(PlateauPatience));
        if (ValidationPatches < 1) throw new ArgumentOutOfRangeException(nameof(ValidationPatches));
    }
}

/// <summary>One epoch's results. <see cref="ValPsnr"/> and the PSNR gain are null without clean validation data.</summary>
public sealed record EpochReport(
    int Epoch,
    double TrainLoss,
    double ValLoss,
    double? ValPsnr,
    double? ValSsim,
    double? ValPsnrGain,
    float LearningRate,
    bool IsBest,
    IReadOnlyList<string> Events,
    TimeSpan Elapsed);

public sealed record TrainingResult(
    IReadOnlyList<EpochReport> History,
    int BestEpoch,
    double BestScore,
    double? NoisyBaselinePsnr,
    string StopReason,
    bool Mastered)
{
    public EpochReport? Best => History.FirstOrDefault(e => e.Epoch == BestEpoch);
}

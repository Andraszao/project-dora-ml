using DoraBio.Core.Data;

namespace DoraBio.Core.Training;

/// <summary>What chooses the best epoch, drives plateau decay and decides early stopping (quality pass Q2).</summary>
public enum ValidationSignal
{
    /// <summary>
    /// Validation PSNR when every validation patch has a clean partner, else the validation loss. The behaviour up to
    /// Q2: in Noise2Void mode it lets ground truth, when the dataset happens to have it, choose the weights.
    /// </summary>
    Auto,

    /// <summary>
    /// Always the validation loss: the masked Noise2Void loss against noisy pixels, or the MSE against clean in
    /// supervised mode. A self-supervised run then never uses ground truth to choose its weights.
    /// </summary>
    Loss,
}

/// <summary>
/// Every training knob in one immutable record. Defaults are tuned for 256–512 px fluorescence frames on a
/// laptop-class CPU. Override with <c>with { … }</c>. It's serialized into the journey log, so each session
/// records exactly how it was trained.
/// </summary>
public sealed record TrainerConfig
{
    // ---- What to learn and on what --------------------------------------------------------------

    /// <summary>Noise2Void (noisy images only) or supervised (needs clean targets for every image).</summary>
    public TrainingMode Mode { get; init; } = TrainingMode.Noise2Void;

    /// <summary>
    /// Side of the square training patches, in pixels (at least 16). Patches must be well above the receptive
    /// field (45 px by default) so most pixels see real context rather than zero padding.
    /// </summary>
    public int PatchSize { get; init; } = 64;

    /// <summary>Patches per optimizer step. Gradients are averaged over the batch.</summary>
    public int BatchSize { get; init; } = 16;

    /// <summary>
    /// Optimizer steps per "epoch". Patches are sampled at random, so an epoch is a fixed amount of work between
    /// validations, not a pass over the data.
    /// </summary>
    public int StepsPerEpoch { get; init; } = 50;

    /// <summary>Hard upper bound on epochs; mastery or early stopping usually ends training first.</summary>
    public int MaxEpochs { get; init; } = 30;

    // ---- Optimization ---------------------------------------------------------------------------

    /// <summary>Starting Adam step size.</summary>
    public float LearningRate { get; init; } = 1e-3f;

    /// <summary>Floor for plateau decay; the rate never goes below this.</summary>
    public float MinLearningRate { get; init; } = 1e-5f;

    /// <summary>Multiplier applied when the validation score plateaus. Below 1, so the rate only decays.</summary>
    public float LearningRateDecay { get; init; } = 0.85f;

    /// <summary>Epochs without a new best score before each decay.</summary>
    public int PlateauPatience { get; init; } = 2;

    /// <summary>Epochs without a new best score before training stops. <see cref="int.MaxValue"/> never stops early.</summary>
    public int EarlyStopPatience { get; init; } = 6;

    /// <summary>The signal behind best-epoch selection, plateau decay and early stopping.</summary>
    public ValidationSignal Selection { get; init; } = ValidationSignal.Auto;

    /// <summary>
    /// Keep the final epoch's weights instead of restoring the best epoch's. The best epoch is still tracked and
    /// reported; it just isn't restored. Needs no held-out signal to pick the weights (Q2's candidate C).
    /// </summary>
    public bool KeepLastEpoch { get; init; }

    /// <summary>Global gradient-norm cap per step (null = no clipping). Guards against rare explosive batches.</summary>
    public float? GradientClipNorm { get; init; } = 1.0f;

    // ---- Noise2Void and validation --------------------------------------------------------------

    /// <summary>Patch augmentation. Lossless dihedral by default; <see cref="AugmentationMode.V1RotateScale"/> exists to measure v1's recipe.</summary>
    public AugmentationMode Augmentation { get; init; } = AugmentationMode.Dihedral;

    /// <summary>Fraction of pixels masked per patch in Noise2Void mode.</summary>
    /// <remarks>
    /// 2% follows the Noise2Void paper. More masked pixels give more loss terms per patch, but leave more
    /// holes in the input.
    /// </remarks>
    public float MaskRatio { get; init; } = 0.02f;

    /// <summary>Share of fields of view held out for validation (by FOV group; spatial split if there is only one).</summary>
    public double ValidationFraction { get; init; } = 0.2;

    /// <summary>
    /// Number of fixed validation patches scored after every epoch, spread over the held-out fields of view (one per
    /// FOV before any FOV gets a second). 48 covers all 46 held-out FMD FOVs.
    /// </summary>
    public int ValidationPatches { get; init; } = 48;

    /// <summary>
    /// Seeds the split, patch sampling, masking and validation patches. Together with
    /// <see cref="DoraBio.Core.Nn.DenoisingNet.Create"/>'s seed, it makes a run bit-for-bit reproducible.
    /// </summary>
    public int Seed { get; init; } = 1234;

    // ---- Progressive targets (DORA's milestone loop) --------------------------------------------

    /// <summary>
    /// Progressive targets are measured as PSNR gain (dB) of DORA's output over the noisy input on the
    /// validation set. That needs clean validation images; without them the milestones are skipped.
    /// Gains travel across datasets better than absolute PSNR, which depends heavily on the noise level.
    /// </summary>
    public bool EnableProgressiveTargets { get; init; } = true;

    /// <summary>First target, in dB of PSNR gain over the noisy input.</summary>
    public double InitialTargetGainDb { get; init; } = 2.0;

    /// <summary>How much each escalation raises the target, in dB.</summary>
    public double TargetIncrementDb { get; init; } = 2.0;

    /// <summary>Final target. Holding it for <see cref="TargetHitsNeeded"/> consecutive epochs is mastery.</summary>
    public double MaxTargetGainDb { get; init; } = 8.0;

    /// <summary>Consecutive epochs at or above the target needed to escalate (or, at the maximum, to master).</summary>
    public int TargetHitsNeeded { get; init; } = 2;

    /// <summary>End training as soon as mastery is reached (as v1 did).</summary>
    public bool StopOnMastery { get; init; } = true;

    /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> for settings that can't train.</summary>
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
/// <param name="Epoch">1-based epoch number.</param>
/// <param name="TrainLoss">Mean batch loss over the epoch's steps (masked MSE on training patches).</param>
/// <param name="ValLoss">Validation loss: the blind-spot MSE against noisy pixels (Noise2Void) or MSE against clean (supervised).</param>
/// <param name="ValPsnr">Validation PSNR of DORA's output against clean, in dB.</param>
/// <param name="ValSsim">Validation SSIM against clean (1 = identical structure).</param>
/// <param name="ValPsnrGain">ValPsnr minus the noisy input's own PSNR: how many dB DORA adds. Drives the targets.</param>
/// <param name="LearningRate">Learning rate in effect at the end of the epoch (after any decay).</param>
/// <param name="IsBest">True if this epoch set a new best validation score (and its weights were kept).</param>
/// <param name="Events">Human-readable milestones: new best, plateau decay, target hit/raised/mastered.</param>
/// <param name="Elapsed">Wall-clock time since training started.</param>
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

/// <summary>
/// Outcome of <see cref="Trainer.Train"/>. The network now holds the weights from <see cref="BestEpoch"/>, or the last
/// epoch's with <see cref="TrainerConfig.KeepLastEpoch"/>.
/// </summary>
/// <param name="History">Every epoch's report, in order.</param>
/// <param name="BestEpoch">Epoch whose weights were restored (0 if no epoch finished).</param>
/// <param name="BestScore">That epoch's score: validation PSNR if clean data exists and <see cref="TrainerConfig.Selection"/> is
/// Auto, else the negated validation loss.</param>
/// <param name="NoisyBaselinePsnr">PSNR of the unprocessed noisy validation patches, the "do nothing" reference.</param>
/// <param name="StopReason">"mastered", "early_stop", "max_epochs" or "cancelled".</param>
/// <param name="Mastered">Whether the final progressive target was held.</param>
public sealed record TrainingResult(
    IReadOnlyList<EpochReport> History,
    int BestEpoch,
    double BestScore,
    double? NoisyBaselinePsnr,
    string StopReason,
    bool Mastered)
{
    /// <summary>The report of <see cref="BestEpoch"/>, or null if no epoch finished.</summary>
    public EpochReport? Best => History.FirstOrDefault(e => e.Epoch == BestEpoch);

    /// <summary>The report of the final epoch, or null if no epoch finished.</summary>
    public EpochReport? Last => History.Count == 0 ? null : History[^1];

    /// <summary>The parameters after the final epoch, before any restore (null if no epoch finished).</summary>
    public float[]? LastParameters { get; init; }
}

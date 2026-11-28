using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

/// <summary>
/// Mini-batch trainer. Training is CPU-bound and synchronous; hosts (CLI, Godot bridge) run it on a background
/// thread and receive per-epoch reports through <c>onEpoch</c>, which is called on the training thread.
///
/// Per step: sample a batch of augmented patches, compute every sample's gradient in parallel (one buffer per
/// sample, summed in a fixed order so runs are reproducible), average over the batch's masked pixels, then
/// apply one Adam update.
/// Per epoch: score the fixed validation set, keep a copy of the best weights, lower the learning rate on
/// plateaus, update the progressive targets, and stop early when the score stalls.
/// When training ends (including on cancellation), the best weights are copied back into the network.
/// </summary>
public sealed class Trainer
{
    private readonly DenoisingNet _net;
    private readonly TrainerConfig _config;

    /// <param name="net">Trained in place. Pass a clone if the original must stay usable meanwhile.</param>
    /// <param name="config">Validated immediately; invalid settings throw here, not mid-run.</param>
    public Trainer(DenoisingNet net, TrainerConfig config)
    {
        config.Validate();
        _net = net;
        _config = config;
    }

    /// <summary>
    /// Runs training to completion, cancellation, mastery or early stop. The network ends up holding its best weights
    /// (or its last, with <see cref="TrainerConfig.KeepLastEpoch"/>).
    /// </summary>
    /// <param name="dataset">Normalized image pairs. Split into train and validation by FOV inside this call.</param>
    /// <param name="onEpoch">Called on the training thread after every epoch (log it, persist it, or marshal it to a UI).</param>
    /// <param name="cancellation">Checked between steps; the current step finishes, then the best weights are restored.</param>
    public TrainingResult Train(
        IReadOnlyList<ImagePair> dataset,
        Action<EpochReport>? onEpoch = null,
        CancellationToken cancellation = default)
    {
        var (trainImages, valImages) = DatasetSplit.Split(dataset, _config.ValidationFraction, _config.Seed);
        return Train(trainImages, valImages, onEpoch, cancellation);
    }

    /// <summary>
    /// As <see cref="Train(IReadOnlyList{ImagePair}, Action{EpochReport}?, CancellationToken)"/>, with the validation
    /// images given instead of split off by FOV: for a held-out block chosen in advance, such as brick 40's contiguous
    /// validation frames of a time-lapse.
    /// </summary>
    /// <param name="trainImages">Normalized training images.</param>
    /// <param name="valImages">Normalized validation images (never trained on).</param>
    /// <param name="onEpoch">Called on the training thread after every epoch.</param>
    /// <param name="cancellation">Checked between steps.</param>
    public TrainingResult Train(
        IReadOnlyList<ImagePair> trainImages,
        IReadOnlyList<ImagePair> valImages,
        Action<EpochReport>? onEpoch = null,
        CancellationToken cancellation = default)
    {
        // Setup: build the sampler and the fixed validation set, and create a fresh optimizer.
        // Targets only exist when validation has clean data (PSNR gain needs ground truth).
        var cfg = _config;
        CheckFrames(trainImages.Concat(valImages));
        if (trainImages.Count == 0 || valImages.Count == 0) throw new ArgumentException("Training and validation images are both needed.");
        var rng = new Random(cfg.Seed);
        var sampler = new PatchSampler(trainImages, cfg.PatchSize, cfg.Mode, cfg.MaskRatio, rng, cfg.Augmentation);
        var validation = new ValidationSet(valImages, cfg);
        var optimizer = new AdamOptimizer(_net.ParameterCount, cfg.LearningRate) { ClipNorm = cfg.GradientClipNorm };
        var targets = cfg.EnableProgressiveTargets && validation.HasClean
            ? new ProgressiveTargets(cfg.InitialTargetGainDb, cfg.TargetIncrementDb, cfg.MaxTargetGainDb, cfg.TargetHitsNeeded)
            : null;

        var stepper = new BatchStepper(_net, cfg.BatchSize, cfg.PatchSize);
        var bestParameters = (float[])_net.Parameters.Clone();
        double bestScore = double.NegativeInfinity;
        int bestEpoch = 0, epochsSinceBest = 0;
        var history = new List<EpochReport>();
        string stopReason = "max_epochs";
        var clock = System.Diagnostics.Stopwatch.StartNew();

        for (int epoch = 1; epoch <= cfg.MaxEpochs; epoch++)
        {
            // 1. Train: StepsPerEpoch mini-batch updates (cancellation is checked between steps).
            double lossSum = 0;
            int steps = 0;
            for (; steps < cfg.StepsPerEpoch && !cancellation.IsCancellationRequested; steps++)
                lossSum += stepper.Step(sampler, optimizer);
            if (cancellation.IsCancellationRequested && steps == 0) { stopReason = "cancelled"; break; }

            // 2. Score on the fixed validation patches. With ground truth, PSNR is the score. Without it,
            //    the blind-spot loss is the only signal (negated so that higher is better in both cases).
            var score = validation.Evaluate(_net);
            double primary = cfg.Selection == ValidationSignal.Loss ? -score.Loss : score.Psnr ?? -score.Loss;
            var events = new List<string>();

            // 3. Keep the best weights. On a plateau, decay the learning rate every PlateauPatience stale epochs.
            bool isBest = primary > bestScore + 1e-6;
            if (isBest)
            {
                bestScore = primary;
                bestEpoch = epoch;
                epochsSinceBest = 0;
                Array.Copy(_net.Parameters, bestParameters, bestParameters.Length);
                events.Add("⭐ new best");
            }
            else
            {
                epochsSinceBest++;
                if (epochsSinceBest % cfg.PlateauPatience == 0 && optimizer.LearningRate > cfg.MinLearningRate)
                {
                    float old = optimizer.LearningRate;
                    optimizer.LearningRate = Math.Max(cfg.MinLearningRate, old * cfg.LearningRateDecay);
                    events.Add($"🐢 plateau: lr {old:G3} → {optimizer.LearningRate:G3}");
                }
            }

            // 4. Progressive targets: is DORA's improvement over the raw input big enough yet?
            double? gain = score.Psnr - validation.NoisyPsnr;
            TargetEvent? targetEvent = null;
            if (targets is not null && gain is { } g)
            {
                targetEvent = targets.Evaluate(g);
                events.Add(targetEvent.ToString());
            }

            var report = new EpochReport(epoch, lossSum / Math.Max(1, steps), score.Loss, score.Psnr, score.Ssim, gain,
                optimizer.LearningRate, isBest, events, clock.Elapsed);
            history.Add(report);
            onEpoch?.Invoke(report);

            // 5. Stop conditions, in priority order.
            if (cancellation.IsCancellationRequested) { stopReason = "cancelled"; break; }
            if (cfg.StopOnMastery && targetEvent?.Kind == TargetEventKind.Mastered) { stopReason = "mastered"; break; }
            if (epochsSinceBest >= cfg.EarlyStopPatience) { stopReason = "early_stop"; break; }
        }

        // Hand back the best weights seen, not whatever the last epoch happened to leave (v1 saved the last),
        // unless the config asks for the last epoch's.
        float[]? last = history.Count > 0 ? (float[])_net.Parameters.Clone() : null;
        if (bestEpoch > 0 && !cfg.KeepLastEpoch) Array.Copy(bestParameters, _net.Parameters, bestParameters.Length);
        return new TrainingResult(history, bestEpoch, bestScore, validation.NoisyPsnr, stopReason, targets?.IsMastered ?? false)
        {
            LastParameters = last,
        };
    }

    /// <summary>
    /// Every image must carry the frames the network's window needs (<see cref="NetConfig.WindowFrames"/>), and a
    /// skip-centre network trains only in skip-centre mode (and the other way round): any other mode would put frame t,
    /// its target, into the network's input planes.
    /// </summary>
    private void CheckFrames(IEnumerable<ImagePair> images)
    {
        var net = _net.Config;
        if (net.SkipsCentre != (_config.Mode == TrainingMode.SkipCentre))
            throw new ArgumentException(net.SkipsCentre
                ? "A skip-centre network trains only in skip-centre mode (input t − 1, t + 1; target t)."
                : "Skip-centre training needs a skip-centre network (NetConfig.SkipCentre).");
        if (images.Any(i => i.FrameCount != net.WindowFrames))
            throw new ArgumentException(net.WindowFrames == 1
                ? "This network takes single frames, but the training images carry neighbouring frames."
                : $"This network takes {net.InputFrames} frames; every training image needs {net.WindowFrames - 1} neighbouring frames.");
    }
}

/// <summary>One mini-batch gradient step. Shared by <see cref="Trainer"/> and <see cref="OnlineAdapter"/>.</summary>
internal sealed class BatchStepper
{
    private readonly DenoisingNet _net;
    private readonly TrainingBatch _batch;
    private readonly float[][] _sampleGradients;
    private readonly double[] _sampleLoss;
    private readonly float[] _gradients;
    private readonly WorkspacePool _pool;

    /// <summary>Preallocates the batch, one gradient buffer per sample, and the summed gradient.</summary>
    public BatchStepper(DenoisingNet net, int batchSize, int patchSize)
    {
        _net = net;
        _batch = new TrainingBatch(batchSize, patchSize, net.Config.InputFrames);
        _sampleGradients = Enumerable.Range(0, batchSize).Select(_ => new float[net.ParameterCount]).ToArray();
        _sampleLoss = new double[batchSize];
        _gradients = new float[net.ParameterCount];
        _pool = new WorkspacePool(net);
    }

    /// <summary>Workspaces this stepper's pool has created (at most one per core over a whole run; for tests).</summary>
    internal int WorkspacesCreated => _pool.Created;

    /// <summary>Returns the batch loss: mean squared error over masked pixels.</summary>
    public double Step(PatchSampler sampler, AdamOptimizer optimizer)
    {
        sampler.Fill(_batch);

        // Normalize by the masked pixels in the whole batch, so the summed gradients are those of the
        // batch-mean loss, whether a pixel is one of 2% blind spots or one of 100% supervised pixels.
        double maskTotal = 0;
        foreach (float m in _batch.Masks) maskTotal += m;
        float scale = (float)(1.0 / Math.Max(1.0, maskTotal));
        int size = _batch.Size;

        // Each sample writes only its own gradient buffer, so the parallel loop needs no locks. Summing the
        // buffers afterwards in sample order (not completion order) keeps runs bit-for-bit reproducible.

        Parallel.For(0, _batch.Count, WorkspacePool.OnePerCore(), () => _pool.Rent(), (i, _, ws) =>
        {
            Array.Clear(_sampleGradients[i]);
            _sampleLoss[i] = _net.AccumulateGradients(_batch.Input(i), _batch.Target(i), _batch.Mask(i),
                size, size, scale, _sampleGradients[i], ws, _batch.Conditions[i]);
            return ws;
        }, _pool.Return);

        Array.Clear(_gradients);
        double sse = 0;
        for (int i = 0; i < _batch.Count; i++)
        {
            Simd.Axpy(1f, _sampleGradients[i], _gradients);
            sse += _sampleLoss[i];
        }
        optimizer.Apply(_net.Parameters, _gradients);
        return sse * scale;
    }
}

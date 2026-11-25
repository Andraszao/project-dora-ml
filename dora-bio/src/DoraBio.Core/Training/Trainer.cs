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

    public Trainer(DenoisingNet net, TrainerConfig config)
    {
        config.Validate();
        _net = net;
        _config = config;
    }

    public TrainingResult Train(
        IReadOnlyList<ImagePair> dataset,
        Action<EpochReport>? onEpoch = null,
        CancellationToken cancellation = default)
    {
        var cfg = _config;
        var (trainImages, valImages) = DatasetSplit.Split(dataset, cfg.ValidationFraction, cfg.Seed);
        var rng = new Random(cfg.Seed);
        var sampler = new PatchSampler(trainImages, cfg.PatchSize, cfg.Mode, cfg.MaskRatio, rng);
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
            double lossSum = 0;
            int steps = 0;
            for (; steps < cfg.StepsPerEpoch && !cancellation.IsCancellationRequested; steps++)
                lossSum += stepper.Step(sampler, optimizer);
            if (cancellation.IsCancellationRequested && steps == 0) { stopReason = "cancelled"; break; }

            var score = validation.Evaluate(_net);
            double primary = score.Psnr ?? -score.Loss;   // higher is better
            var events = new List<string>();

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

            if (cancellation.IsCancellationRequested) { stopReason = "cancelled"; break; }
            if (cfg.StopOnMastery && targetEvent?.Kind == TargetEventKind.Mastered) { stopReason = "mastered"; break; }
            if (epochsSinceBest >= cfg.EarlyStopPatience) { stopReason = "early_stop"; break; }
        }

        if (bestEpoch > 0) Array.Copy(bestParameters, _net.Parameters, bestParameters.Length);
        return new TrainingResult(history, bestEpoch, bestScore, validation.NoisyPsnr, stopReason, targets?.IsMastered ?? false);
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

    public BatchStepper(DenoisingNet net, int batchSize, int patchSize)
    {
        _net = net;
        _batch = new TrainingBatch(batchSize, patchSize);
        _sampleGradients = Enumerable.Range(0, batchSize).Select(_ => new float[net.ParameterCount]).ToArray();
        _sampleLoss = new double[batchSize];
        _gradients = new float[net.ParameterCount];
        _pool = new WorkspacePool(net);
    }

    /// <summary>Returns the batch loss: mean squared error over masked pixels.</summary>
    public double Step(PatchSampler sampler, AdamOptimizer optimizer)
    {
        sampler.Fill(_batch);
        double maskTotal = 0;
        foreach (float m in _batch.Masks) maskTotal += m;
        float scale = (float)(1.0 / Math.Max(1.0, maskTotal));
        int size = _batch.Size;

        Parallel.For(0, _batch.Count, () => _pool.Rent(), (i, _, ws) =>
        {
            Array.Clear(_sampleGradients[i]);
            _sampleLoss[i] = _net.AccumulateGradients(_batch.Input(i), _batch.Target(i), _batch.Mask(i),
                size, size, scale, _sampleGradients[i], ws);
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

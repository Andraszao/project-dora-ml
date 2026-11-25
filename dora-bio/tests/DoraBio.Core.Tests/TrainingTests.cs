using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;
using Xunit.Abstractions;

namespace DoraBio.Core.Tests;

[Trait(Bricks.Trait, Bricks.Training)]
public class ProgressiveTargetTests
{
    [Fact]
    public void ClimbsToTheMaximumAndDeclaresMastery()
    {
        var targets = new ProgressiveTargets(initial: 2, increment: 2, maximum: 6, hitsNeeded: 2);
        var kinds = new[] { 2.5, 2.5, 4.1, 4.2, 6.0, 6.3 }.Select(v => targets.Evaluate(v).Kind).ToList();
        Assert.Equal(new[]
        {
            TargetEventKind.Hit, TargetEventKind.Escalated,     // 2 → 4
            TargetEventKind.Hit, TargetEventKind.Escalated,     // 4 → 6
            TargetEventKind.Hit, TargetEventKind.Mastered,      // held the maximum: v1 never managed this
        }, kinds);
        Assert.True(targets.IsMastered);
        Assert.Equal(6, targets.CurrentTarget);
    }

    [Fact]
    public void AMissResetsTheStreak()
    {
        var targets = new ProgressiveTargets(2, 2, 8, 2);
        targets.Evaluate(3);
        var miss = targets.Evaluate(1);
        Assert.Equal(TargetEventKind.Missed, miss.Kind);
        Assert.Equal(0, targets.Hits);
        Assert.Equal(TargetEventKind.Hit, targets.Evaluate(3).Kind);   // has to start over
    }

    [Fact]
    public void IncrementIsCappedAtMaximum()
    {
        var targets = new ProgressiveTargets(2, 5, 4, 1);
        Assert.Equal(TargetEventKind.Escalated, targets.Evaluate(3).Kind);
        Assert.Equal(4, targets.CurrentTarget);
    }
}

[Trait(Bricks.Trait, Bricks.Training)]
public class TrainingTests
{
    private readonly ITestOutputHelper _output;
    public TrainingTests(ITestOutputHelper output) => _output = output;

    private static readonly NetConfig SmallNet = new(8, [1, 2, 4, 2, 1]);

    private static TrainerConfig QuickConfig(TrainingMode mode) => new()
    {
        Mode = mode,
        PatchSize = 32,
        BatchSize = 8,
        StepsPerEpoch = 25,
        MaxEpochs = 8,
        LearningRate = 3e-3f,
        ValidationPatches = 8,
        EarlyStopPatience = 8,
        InitialTargetGainDb = 1,
        TargetIncrementDb = 1,
        MaxTargetGainDb = 20,
        Seed = 3,
    };

    private static List<ImagePair> SyntheticData() =>
        SyntheticFluorescence.GenerateDataset(count: 5, size: 96, seed: 10, new SyntheticOptions { PeakPhotons = 15 });

    [Theory]
    [InlineData(TrainingMode.Supervised, 3.0)]
    [InlineData(TrainingMode.Noise2Void, 1.5)]
    public void LearnsToDenoiseSyntheticFluorescence(TrainingMode mode, double minimumGainDb)
    {
        var net = DenoisingNet.Create(SmallNet, seed: 1);
        var result = new Trainer(net, QuickConfig(mode)).Train(SyntheticData(),
            r => _output.WriteLine($"{mode} epoch {r.Epoch}: loss {r.TrainLoss:G4} val {r.ValLoss:G4} psnr {r.ValPsnr:0.00} gain {r.ValPsnrGain:0.00} | {string.Join("; ", r.Events)}"));

        var best = result.Best!;
        _output.WriteLine($"noisy baseline {result.NoisyBaselinePsnr:0.00} dB, best {best.ValPsnr:0.00} dB at epoch {best.Epoch}");
        Assert.NotNull(result.NoisyBaselinePsnr);
        Assert.True(best.ValPsnrGain >= minimumGainDb, $"gain {best.ValPsnrGain:0.00} dB < {minimumGainDb} dB");
        Assert.Contains(result.History, r => r.Events.Any(e => e.Contains("target")));
    }

    [Fact]
    public void ValidationPatchesCoverEveryHeldOutFovBeforeRepeatingOne()
    {
        // Listed FOV by FOV, the way real datasets are. P2's validation took every patch from the first FOVs in the list.
        var images = new List<ImagePair>();
        foreach (var fov in Enumerable.Range(0, 6))
            foreach (var frame in Enumerable.Range(0, 3))
                images.Add(new ImagePair(TestUtil.RandomImage(32, 32, seed: fov * 10 + frame), null, $"fov{fov}", $"fov{fov}/{frame}"));

        var six = new ValidationSet(images, QuickConfig(TrainingMode.Noise2Void) with { ValidationPatches = 6 });
        Assert.Equal(6, six.PatchSources.Distinct().Count());

        var fifteen = new ValidationSet(images, QuickConfig(TrainingMode.Noise2Void) with { ValidationPatches = 15 });
        var counts = fifteen.PatchSources.GroupBy(s => s).Select(g => g.Count()).ToList();
        Assert.Equal(6, counts.Count);
        Assert.InRange(counts.Min(), 2, 3);   // 15 patches over 6 FOVs: every FOV gets 2 or 3
        Assert.InRange(counts.Max(), 2, 3);
    }

    [Fact]
    public void TrainingIsReproducibleForAFixedSeed()
    {
        var config = QuickConfig(TrainingMode.Supervised) with { MaxEpochs = 2, StepsPerEpoch = 5 };
        var a = DenoisingNet.Create(SmallNet, seed: 1);
        var b = DenoisingNet.Create(SmallNet, seed: 1);
        new Trainer(a, config).Train(SyntheticData());
        new Trainer(b, config).Train(SyntheticData());
        Assert.Equal(a.Parameters, b.Parameters);
    }

    [Fact]
    public void BestWeightsAreRestoredAndCancellationStopsCleanly()
    {
        using var cts = new CancellationTokenSource();
        var net = DenoisingNet.Create(SmallNet, seed: 1);
        var result = new Trainer(net, QuickConfig(TrainingMode.Supervised)).Train(SyntheticData(), r =>
        {
            if (r.Epoch == 2) cts.Cancel();
        }, cts.Token);
        Assert.Equal("cancelled", result.StopReason);
        Assert.Equal(2, result.History.Count);
        Assert.InRange(result.BestEpoch, 1, 2);
    }

    [Fact]
    public void LearningRateOnlyDecays()
    {
        var config = QuickConfig(TrainingMode.Supervised) with { LearningRate = 0.5f, PlateauPatience = 1, MaxEpochs = 4, StepsPerEpoch = 3 };
        var result = new Trainer(DenoisingNet.Create(SmallNet, 1), config).Train(SyntheticData());
        var rates = result.History.Select(h => h.LearningRate).ToList();
        for (int i = 1; i < rates.Count; i++) Assert.True(rates[i] <= rates[i - 1]);
    }

    [Fact]
    public void TrainingJobWritesOneJourneySessionWithEveryEpochAndLoadableWeights()
    {
        string dir = TestUtil.TempDirectory();
        string journal = Path.Combine(dir, "journey.json"), weights = Path.Combine(dir, "w.bin");
        var config = QuickConfig(TrainingMode.Supervised) with { MaxEpochs = 3, StepsPerEpoch = 4, EnableProgressiveTargets = false };
        var options = new TrainingJobOptions { DatasetDescription = "5 synthetic FOVs", WeightsPath = weights, JournalPath = journal };

        var net = DenoisingNet.Create(SmallNet, 1);
        var result = TrainingJob.Run(net, SyntheticData(), config, options);
        TrainingJob.Run(DenoisingNet.Create(SmallNet, 2), SyntheticData(), config, options);

        var log = JourneyLog.Open(journal);
        Assert.Equal(2, log.Document.Sessions.Count);
        var first = log.Document.Sessions[0];
        Assert.Equal("completed", first.Status);
        Assert.Equal(result.History.Count, first.Epochs.Count);
        Assert.Equal(result.BestEpoch, first.BestEpoch);
        Assert.Equal(TrainingMode.Supervised, first.Config!.Mode);
        Assert.Equal(SmallNet, first.Network);

        var (loaded, info) = WeightsFile.Load(weights);
        Assert.Equal(SmallNet, loaded.Config);
        Assert.Contains("5 synthetic FOVs", info.Provenance);
        Directory.Delete(dir, recursive: true);
    }

    [Fact]
    public void OnlineAdapterKeepsOptimizerStateAcrossCalls()
    {
        var image = SyntheticData()[0].Noisy;
        var adapter = new OnlineAdapter(DenoisingNet.Create(SmallNet, 1), patchSize: 32, batchSize: 4);
        adapter.Adapt(image, 3);
        adapter.Adapt(image, 4);
        Assert.Equal(7, adapter.OptimizerStep);
    }

    [Fact]
    public void OnlineAdaptationReducesBlindSpotLossOnTheSample()
    {
        var image = SyntheticData()[0].Noisy;
        var adapter = new OnlineAdapter(DenoisingNet.Create(SmallNet, 1), learningRate: 3e-3f, patchSize: 32, batchSize: 8);
        var losses = adapter.Adapt(image, 60);
        double early = losses.Take(10).Average(), late = losses.Skip(50).Average();
        _output.WriteLine($"blind-spot loss: first 10 steps {early:G4}, last 10 steps {late:G4}");
        Assert.True(late < early);
    }

    [Fact]
    public void DatasetLoaderReadsPairedAndFlatLayouts()
    {
        string dir = TestUtil.TempDirectory();
        Directory.CreateDirectory(Path.Combine(dir, "paired", "noisy"));
        Directory.CreateDirectory(Path.Combine(dir, "paired", "clean"));
        Directory.CreateDirectory(Path.Combine(dir, "flat"));
        for (int i = 0; i < 3; i++)
        {
            var frame = SyntheticFluorescence.Generate(40, 40, i);
            File.WriteAllBytes(Path.Combine(dir, "paired", "noisy", $"f{i}.tif"), TiffCodec.Encode(frame.Noisy));
            if (i < 2) File.WriteAllBytes(Path.Combine(dir, "paired", "clean", $"f{i}.tif"), TiffCodec.Encode(frame.Clean, TiffSampleType.Float32));
            File.WriteAllBytes(Path.Combine(dir, "flat", $"f{i}.tif"), TiffCodec.Encode(frame.Noisy));
        }

        var paired = DatasetLoader.LoadDirectory(Path.Combine(dir, "paired"));
        Assert.Equal(3, paired.Count);
        Assert.Equal(2, paired.Count(p => p.HasClean));
        Assert.InRange(paired[0].Noisy.Pixels.Average(), -0.5, 1.5);   // normalized

        var flat = DatasetLoader.LoadDirectory(Path.Combine(dir, "flat"));
        Assert.Equal(3, flat.Count);
        Assert.All(flat, p => Assert.False(p.HasClean));
        Directory.Delete(dir, recursive: true);
    }
}

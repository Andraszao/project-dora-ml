using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Quality pass Q2 (brick 35): what chooses a training run's weights. <see cref="ValidationSignal.Auto"/> picks the
/// epoch with the best validation PSNR whenever clean images exist (so a Noise2Void run on FMD is chosen with ground
/// truth); <see cref="ValidationSignal.Loss"/> always picks the lowest validation loss; <c>KeepLastEpoch</c> keeps
/// the final weights. The recorded verdict is checked by gate 35 against the pre-registered rule.
/// </summary>
[Trait(Bricks.Trait, Bricks.NoiseStop)]
public class StoppingTests
{
    private static readonly NetConfig SmallNet = new(8, [1, 2, 4, 2, 1]);

    /// <summary>Noise2Void on synthetic data that has clean partners, so validation reports both loss and PSNR.</summary>
    private static TrainerConfig Config(ValidationSignal selection, bool keepLast = false) => new()
    {
        Mode = TrainingMode.Noise2Void,
        PatchSize = 32,
        BatchSize = 8,
        StepsPerEpoch = 20,
        MaxEpochs = 30,                // long enough to plateau, so best, last and PSNR-best differ
        LearningRate = 3e-3f,
        ValidationPatches = 8,
        EarlyStopPatience = int.MaxValue,
        EnableProgressiveTargets = false,
        Selection = selection,
        KeepLastEpoch = keepLast,
        Seed = 9,                      // a seed where PSNR and the masked loss pick very different epochs (29 against 19)
    };

    private static List<ImagePair> Data() =>
        SyntheticFluorescence.GenerateDataset(count: 5, size: 96, seed: 10, new SyntheticOptions { PeakPhotons = 15 });

    private static int ArgBest(IReadOnlyList<EpochReport> history, Func<EpochReport, double> score)
    {
        int best = 0;
        double bestScore = double.NegativeInfinity;
        foreach (var r in history)
            if (score(r) > bestScore + 1e-6) (best, bestScore) = (r.Epoch, score(r));
        return best;
    }

    [Fact]
    public void TheSelectionSignalDecidesTheBestEpoch()
    {
        var auto = new Trainer(DenoisingNet.Create(SmallNet, 1), Config(ValidationSignal.Auto)).Train(Data());
        var loss = new Trainer(DenoisingNet.Create(SmallNet, 1), Config(ValidationSignal.Loss)).Train(Data());
        Assert.All(auto.History, r => Assert.NotNull(r.ValPsnr));                       // clean partners exist
        Assert.Equal(ArgBest(auto.History, r => r.ValPsnr!.Value), auto.BestEpoch);      // Auto: PSNR with ground truth
        Assert.Equal(ArgBest(loss.History, r => -r.ValLoss), loss.BestEpoch);           // Loss: never ground truth
        Assert.Equal(-loss.Best!.ValLoss, loss.BestScore, 9);
        // In these runs the two signals disagree, so the test can tell them apart (it would pass either way otherwise).
        Assert.NotEqual(ArgBest(auto.History, r => r.ValPsnr!.Value), ArgBest(auto.History, r => -r.ValLoss));
        Assert.NotEqual(ArgBest(loss.History, r => r.ValPsnr!.Value), ArgBest(loss.History, r => -r.ValLoss));
    }

    [Fact]
    public void PatienceOffRunsTheWholeBudget()
    {
        var result = new Trainer(DenoisingNet.Create(SmallNet, 1), Config(ValidationSignal.Loss)).Train(Data());
        Assert.Equal(("max_epochs", 30), (result.StopReason, result.History.Count));
        var stopping = new Trainer(DenoisingNet.Create(SmallNet, 1), Config(ValidationSignal.Loss) with { EarlyStopPatience = 1 })
            .Train(Data());
        Assert.Equal("early_stop", stopping.StopReason);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheNetworkKeepsTheBestOrTheLastWeightsAndTheLastAreAlwaysReported(bool keepLast)
    {
        var net = DenoisingNet.Create(SmallNet, 1);
        var snapshots = new Dictionary<int, float[]>();
        var result = new Trainer(net, Config(ValidationSignal.Loss, keepLast))
            .Train(Data(), r => snapshots[r.Epoch] = (float[])net.Parameters.Clone());
        Assert.NotEqual(result.BestEpoch, result.History.Count);                       // so best and last differ here
        Assert.Equal(snapshots[result.History.Count], result.LastParameters);
        Assert.Equal(snapshots[keepLast ? result.History.Count : result.BestEpoch], net.Parameters);
    }

    [Fact]
    public void TrainingJobSavesTheKeptWeightsAndOptionallyTheLast()
    {
        string dir = TestUtil.TempDirectory();
        var options = new TrainingJobOptions
        {
            DatasetDescription = "synthetic",
            WeightsPath = Path.Combine(dir, "kept.bin"),
            LastWeightsPath = Path.Combine(dir, "last.bin"),
        };
        var net = DenoisingNet.Create(SmallNet, 1);
        var result = TrainingJob.Run(net, Data(), Config(ValidationSignal.Loss), options);
        var (kept, keptInfo) = WeightsFile.Load(options.WeightsPath);
        var (last, lastInfo) = WeightsFile.Load(options.LastWeightsPath);
        Assert.Equal(net.Parameters, kept.Parameters);
        Assert.Equal(result.LastParameters, last.Parameters);
        Assert.Contains($"best epoch {result.BestEpoch};", keptInfo.Provenance);
        Assert.Contains($"last epoch {result.History.Count};", lastInfo.Provenance);

        var lastOnly = TrainingJob.Run(DenoisingNet.Create(SmallNet, 1), Data(), Config(ValidationSignal.Loss, keepLast: true),
            options with { WeightsPath = Path.Combine(dir, "keptlast.bin"), LastWeightsPath = null });
        var (keptLast, keptLastInfo) = WeightsFile.Load(Path.Combine(dir, "keptlast.bin"));
        Assert.Equal(lastOnly.LastParameters, keptLast.Parameters);
        Assert.Contains($"last epoch {lastOnly.History.Count};", keptLastInfo.Provenance);
    }
}

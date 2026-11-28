using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>The benchmark measures what it says, reports sane numbers, and leaves the model it was given untouched.</summary>
[Trait(Bricks.Trait, Bricks.Bench)]
public class BenchTests
{
    [Fact]
    public void DenoiseTimingsAreSaneAndLabelled()
    {
        var net = DenoisingNet.Create(new NetConfig(4, [1, 2, 1]), seed: 1);
        var t = PerformanceBench.Denoise(net, size: 48, runs: 3, tileCore: 16);
        Assert.Equal("denoise", t.Name);
        Assert.Equal("dilated", t.Architecture);
        Assert.Equal("48x48, tile 16", t.Detail);
        Assert.Equal(3, t.Runs);
        Assert.InRange(t.MinSeconds, 1e-9, t.MedianSeconds);
    }

    [Theory]
    [InlineData(TrainingMode.Noise2Void, "train-step-n2v")]
    [InlineData(TrainingMode.Supervised, "train-step-supervised")]
    public void TrainStepTimingsLeaveTheModelUnchanged(TrainingMode mode, string name)
    {
        var net = DenoisingNet.Create(NetConfig.UNet(2, 1), seed: 1);
        var before = (float[])net.Parameters.Clone();
        var t = PerformanceBench.TrainStep(net, mode, batch: 2, patch: 16, runs: 2);
        Assert.Equal(name, t.Name);
        Assert.Equal("unet", t.Architecture);
        Assert.Equal("batch 2 @ 16px", t.Detail);
        Assert.InRange(t.MinSeconds, 1e-9, t.MedianSeconds);
        Assert.Equal(before, net.Parameters);   // a clone was trained
    }

    [Fact]
    public void RejectsZeroRuns()
    {
        var net = DenoisingNet.Create(new NetConfig(2, [1, 1]), seed: 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => PerformanceBench.Denoise(net, 16, runs: 0));
    }
}

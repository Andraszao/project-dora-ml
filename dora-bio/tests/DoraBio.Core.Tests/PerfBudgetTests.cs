using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>The reference-kernel timings that gate 24 compares against really run the reference kernels, on a copy.</summary>
[Trait(Bricks.Trait, Bricks.PerfBudget)]
public class PerfBudgetTests
{
    [Fact]
    public void ReferenceTimingsAreLabelledAndLeaveTheModelOnTheBlockedKernels()
    {
        var net = DenoisingNet.Create(new NetConfig(8, [1, 2, 1]), seed: 1);
        var denoise = PerformanceBench.Denoise(net, size: 32, runs: 2, tileCore: 16, referenceKernels: true);
        var step = PerformanceBench.TrainStep(net, TrainingMode.Noise2Void, batch: 2, patch: 16, runs: 2, referenceKernels: true);
        Assert.Equal("reference", denoise.Kernels);
        Assert.Equal("reference", step.Kernels);
        Assert.Equal("blocked", PerformanceBench.Denoise(net, 32, 1, 16).Kernels);
        Assert.False(net.ReferenceKernels);
    }

    [Fact]
    public void TheFullSetIncludesTheReferenceTimingsTheBudgetNeeds()
    {
        // Gate 24's speedup checks divide these by the blocked timings of the same operation.
        var names = PerformanceBench.Run(quick: true).Select(t => (t.Name, t.Architecture, t.Kernels)).ToHashSet();
        Assert.Contains(("denoise", "dilated", "reference"), names);
        Assert.Contains(("train-step-n2v", "dilated", "reference"), names);
        Assert.Equal(8, names.Count);
    }
}

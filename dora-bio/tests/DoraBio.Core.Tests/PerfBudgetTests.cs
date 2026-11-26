using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// The reference-kernel timings that gate 24 compares against really run the reference kernels, on a copy, and the
/// full set has every timing the budget's speedup checks divide by.
/// </summary>
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
        // On AVX-512 machines brick 36 adds the 256-bit comparisons its speedup floor divides by.
        var narrow = new[] { ("denoise", "dilated", "blocked-256"), ("train-step-n2v", "dilated", "blocked-256"), ("denoise", "unet", "blocked-256") };
        if (ConvLayer.WideKernels) Assert.All(narrow, n => Assert.Contains(n, names));
        Assert.Equal(ConvLayer.WideKernels ? 11 : 8, names.Count);
    }
}

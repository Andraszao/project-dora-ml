using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Quality pass Q19 (brick 37): the routing threshold is kept or re-chosen on validation images by a fixed rule (the
/// owner's first guard), before any test image is scored. The recorded verdict is checked by gate 37.
/// </summary>
[Trait(Bricks.Trait, Bricks.Routing)]
public class RoutingTests
{
    [Fact]
    public void AThresholdThatSeparatesEveryImageIsKept()
    {
        var samples = new[] { (0.26, false), (0.16, false), (0.56, true), (0.57, true) };
        var (threshold, reason) = NoiseDiagnostics.ChooseThreshold(samples);
        Assert.Equal(0.4, threshold);
        Assert.StartsWith("kept", reason);
    }

    [Fact]
    public void AMisroutingThresholdMovesToTheMiddleOfTheValidationGap()
    {
        var samples = new[] { (0.10, false), (0.44, false), (0.50, true), (0.60, true) };   // 0.4 sends 0.44 the wrong way
        var (threshold, reason) = NoiseDiagnostics.ChooseThreshold(samples);
        Assert.Equal(0.47, threshold, 9);
        Assert.StartsWith("moved", reason);
    }

    [Fact]
    public void WithoutAGapTheFewestMisroutesWinAndTiesGoToTheNearestValue()
    {
        // 0.30 (correlated) sits below 0.35 (independent): no threshold separates them, every choice misroutes one.
        var samples = new[] { (0.10, false), (0.35, false), (0.30, true), (0.70, true) };
        var (threshold, _) = NoiseDiagnostics.ChooseThreshold(samples);
        Assert.Equal(1, samples.Count(s => (s.Item1 >= threshold) != s.Item2));
        Assert.Equal(0.4, threshold);                                // 0.4 also misroutes just one, and it is the nearest
        var worse = new[] { (0.10, false), (0.20, true), (0.45, false), (0.50, false), (0.80, true) };
        var (moved, _) = NoiseDiagnostics.ChooseThreshold(worse);    // 0.4 misroutes 3 (0.20, 0.45, 0.50); 0.65 misroutes 1
        Assert.Equal(0.65, moved, 9);
        Assert.Throws<ArgumentException>(() => NoiseDiagnostics.ChooseThreshold(Array.Empty<(double, bool)>()));
    }
}

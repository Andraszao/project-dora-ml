using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>The held-out benchmark treats both architectures alike (checked on the miniature FMD tree).</summary>
[Trait(Bricks.Trait, Bricks.ArchEval)]
public class ArchEvalTests : IDisposable
{
    private readonly string _root = TestUtil.TempDirectory();

    public ArchEvalTests() => FmdTestTree.Build(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void ZeroWeightNetworksOfEitherArchitectureScoreExactlyLikeTheNoisyInput()
    {
        // Zero weights predict zero noise, so the output is the input. The 16×16 test images aren't a problem for the
        // U-Net: its sides must be multiples of 4, which 16 is; tiling and padding are covered by bricks 19 and 20.
        foreach (var config in new[] { new NetConfig(4, [1, 1]), NetConfig.UNet(4, 2) })
        {
            var net = DenoisingNet.FromParameters(config, new float[DenoisingNet.CountParameters(config)]);
            var scores = FmdBenchmark.Evaluate(net, Path.Combine(_root, "test_mix"), "raw");
            Assert.Equal(4, scores.Count);
            Assert.All(scores, s => Assert.Equal(s.NoisyPsnr, s.DoraPsnr, 3));
        }
    }

    [Fact]
    public void TheDefaultArchitectureIsOneOfTheTwoMeasured()
    {
        // Gate 21 checks that NetConfig.Default follows the rule set before measuring; here, only that it is valid.
        NetConfig.Default.Validate();
        Assert.Contains(NetConfig.Default.Architecture, new[] { NetArchitecture.Dilated, NetArchitecture.UNet });
    }
}

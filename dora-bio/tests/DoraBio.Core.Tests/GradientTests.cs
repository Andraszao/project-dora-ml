using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Checks backpropagation against central finite differences, in double precision over float32 math.
/// This is what proves the training gradients are right, instead of just hoping they are.
/// </summary>
[Trait(Bricks.Trait, Bricks.Nn)]
public class GradientTests
{
    [Theory]
    [InlineData(3, new[] { 1, 1 }, 6, 7)]          // shallowest possible net, non-square plane
    [InlineData(4, new[] { 1, 2, 1 }, 9, 8)]       // dilation inside
    [InlineData(3, new[] { 2, 3, 1, 1 }, 7, 11)]   // dilation wider than half the image
    public void AnalyticGradientsMatchFiniteDifferences(int features, int[] dilations, int height, int width)
    {
        var net = DenoisingNet.Create(new NetConfig(features, dilations), seed: 11);
        // Make the last layer non-trivial so gradients flow everywhere.
        var rng = new Random(5);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.2f;

        int plane = height * width;
        var input = TestUtil.RandomPlane(plane, 1);
        var target = TestUtil.RandomPlane(plane, 2);
        var mask = TestUtil.RandomPlane(plane, 3).Select(v => v > 0.6f ? 1f : 0f).ToArray();
        var ws = net.CreateWorkspace();
        float scale = 1f / mask.Sum();

        var analytic = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, mask, height, width, scale, analytic, ws);

        double Loss()
        {
            var scratch = new float[net.ParameterCount];
            return net.AccumulateGradients(input, target, mask, height, width, scale, scratch, ws) * scale;
        }

        // Check a spread of parameters across every layer, including all biases of the last layer.
        var indices = Enumerable.Range(0, net.ParameterCount).Where(i => i % 7 == 0 || i >= net.ParameterCount - 3).ToList();
        const float h = 1e-2f;
        int checkedCount = 0;
        foreach (int i in indices)
        {
            float original = net.Parameters[i];
            net.Parameters[i] = original + h;
            double plus = Loss();
            net.Parameters[i] = original - h;
            double minus = Loss();
            net.Parameters[i] = original;

            double numeric = (plus - minus) / (2 * h);
            double a = analytic[i];
            double tolerance = 2e-3 + 2e-2 * Math.Max(Math.Abs(numeric), Math.Abs(a));
            Assert.True(Math.Abs(numeric - a) <= tolerance,
                $"param {i}: analytic {a:G6} vs numeric {numeric:G6}");
            checkedCount++;
        }
        Assert.True(checkedCount > 10);
        Assert.Contains(analytic, g => Math.Abs(g) > 1e-4);   // the check is not vacuous
    }

    [Fact]
    public void GradientsAccumulateAcrossCalls()
    {
        var net = DenoisingNet.Create(new NetConfig(3, [1, 1]), seed: 3);
        var input = TestUtil.RandomPlane(25, 1);
        var target = TestUtil.RandomPlane(25, 2);
        var ws = net.CreateWorkspace();

        var once = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, [], 5, 5, 1f, once, ws);
        var twice = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, [], 5, 5, 1f, twice, ws);
        net.AccumulateGradients(input, target, [], 5, 5, 1f, twice, ws);

        for (int i = 0; i < once.Length; i++) Assert.Equal(2 * once[i], twice[i], 4);
    }

    [Fact]
    public void UntrainedNetworkIsCloseToIdentity()
    {
        var net = DenoisingNet.Create(NetConfig.Default, seed: 1);
        var input = TestUtil.RandomPlane(32 * 32, 4);
        var output = new float[input.Length];
        net.Predict(input, 32, 32, output, net.CreateWorkspace());
        double maxDiff = input.Zip(output, (a, b) => Math.Abs(a - b)).Max();
        Assert.InRange(maxDiff, 0, 0.5);   // last layer starts at 0.1× scale
    }

    [Fact]
    public void DefaultConfigMatchesDocumentedSize()
    {
        Assert.Equal(26_497, DenoisingNet.CountParameters(NetConfig.Default));
        Assert.Equal(22, NetConfig.Default.ReceptiveRadius);
    }
}

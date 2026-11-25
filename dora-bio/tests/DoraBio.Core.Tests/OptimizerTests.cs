using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

public class OptimizerTests
{
    [Fact]
    public void AdamMinimizesQuadratic()
    {
        var p = new float[] { 3f, -2f, 0.5f };
        var target = new float[] { 1f, 1f, -1f };
        var adam = new AdamOptimizer(3, 0.05f);
        var g = new float[3];
        for (int step = 0; step < 2000; step++)
        {
            for (int i = 0; i < 3; i++) g[i] = 2 * (p[i] - target[i]);
            adam.Apply(p, g);
        }
        for (int i = 0; i < 3; i++) Assert.Equal(target[i], p[i], 2);
        Assert.Equal(2000, adam.Step);
    }

    [Fact]
    public void FirstStepIsSignStepWhichIsWhyStateMustPersist()
    {
        // At t = 1, bias-corrected Adam moves every parameter by about lr·sign(g), whatever |g| is.
        // This was v1's feedback-learning behavior. The test documents it, so reusing an optimizer
        // (OnlineAdapter) is understood as a correctness requirement.
        var p = new float[] { 0f, 0f };
        var adam = new AdamOptimizer(2, 0.01f);
        adam.Apply(p, new float[] { 1000f, 0.001f });
        Assert.Equal(-0.01f, p[0], 5);
        Assert.Equal(-0.01f, p[1], 4);
    }

    [Fact]
    public void PersistentStateScalesUpdatesWithGradientHistory()
    {
        var p = new float[] { 0f, 0f };
        var adam = new AdamOptimizer(2, 0.01f);
        for (int i = 0; i < 50; i++) adam.Apply(p, new float[] { 1f, 1f });
        // A gradient 100× smaller than the history now moves the parameter far less than a full lr step.
        float before = p[1];
        adam.Apply(p, new float[] { 1f, 0.01f });
        Assert.True(Math.Abs(p[1] - before) < 0.01f * 0.95f);
    }

    [Fact]
    public void ClipNormBoundsTheGradient()
    {
        var p = new float[] { 0f, 0f };
        var adam = new AdamOptimizer(2, 0.1f) { ClipNorm = 1f };
        double norm = adam.Apply(p, new float[] { 30f, 40f });
        Assert.Equal(50, norm, 6);
    }
}

using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Pooling and upsampling: hand-checked values, the adjoint identities that make each backward pass the exact transpose
/// of its forward pass, and a finite-difference check of a nonlinear loss through both (the brick-1 harness's method).
/// </summary>
[Trait(Bricks.Trait, Bricks.PoolUpsample)]
public class ResampleTests
{
    [Fact]
    public void PoolingAveragesEach2x2BlockAndUpsamplingCopiesIt()
    {
        // One 4×4 channel: values 0..15 row by row. Block means: (0+1+4+5)/4 = 2.5, 4.5, 10.5, 12.5.
        var input = Enumerable.Range(0, 16).Select(i => (float)i).ToArray();
        var pooled = new float[4];
        Resample.AvgPool2(input, 1, 4, 4, pooled);
        Assert.Equal([2.5f, 4.5f, 10.5f, 12.5f], pooled);

        var up = new float[16];
        Resample.Upsample2(pooled, 1, 2, 2, up);
        Assert.Equal([2.5f, 2.5f, 4.5f, 4.5f, 2.5f, 2.5f, 4.5f, 4.5f, 10.5f, 10.5f, 12.5f, 12.5f, 10.5f, 10.5f, 12.5f, 12.5f], up);

        Assert.Throws<ArgumentException>(() => Resample.AvgPool2(new float[15], 1, 5, 3, new float[4]));
    }

    [Theory]
    [InlineData(1, 4, 6)]
    [InlineData(3, 8, 2)]
    [InlineData(2, 10, 14)]
    public void EachBackwardPassIsTheTransposeOfItsForwardPass(int channels, int height, int width)
    {
        // For a linear map A, backward must be Aᵀ: ⟨A·x, y⟩ = ⟨x, Aᵀ·y⟩ for every x and y.
        int full = channels * height * width, half = full / 4;
        var x = TestUtil.RandomPlane(full, 1, -1, 1);
        var y = TestUtil.RandomPlane(half, 2, -1, 1);
        var ax = new float[half];
        var aty = new float[full];
        Resample.AvgPool2(x, channels, height, width, ax);
        Resample.AvgPool2Backward(y, channels, height, width, aty);
        Assert.Equal(Dot(ax, y), Dot(x, aty), 4);

        var lowX = TestUtil.RandomPlane(half, 3, -1, 1);
        var highY = TestUtil.RandomPlane(full, 4, -1, 1);
        var upX = new float[full];
        var upTy = new float[half];
        Resample.Upsample2(lowX, channels, height / 2, width / 2, upX);
        Resample.Upsample2Backward(highY, channels, height / 2, width / 2, upTy);
        Assert.Equal(Dot(upX, highY), Dot(lowX, upTy), 4);
    }

    [Fact]
    public void GradientsOfANonlinearLossThroughBothMatchFiniteDifferences()
    {
        // L(x) = ½ Σ (v ⊙ up(pool(x)))³ / 3 is nonlinear, so the check exercises the chain rule, not just linearity.
        const int c = 2, h = 6, w = 8;
        int full = c * h * w;
        var x = TestUtil.RandomPlane(full, 5, -1, 1);
        var v = TestUtil.RandomPlane(full, 6, -1, 1);

        double Loss(float[] input)
        {
            var p = new float[full / 4];
            var u = new float[full];
            Resample.AvgPool2(input, c, h, w, p);
            Resample.Upsample2(p, c, h / 2, w / 2, u);
            return u.Select((value, i) => Math.Pow(v[i] * value, 3) / 6).Sum();
        }

        // Analytic: dL/du_i = ½ v_i³ u_i², then back through upsample and pooling.
        var pooled = new float[full / 4];
        var upsampled = new float[full];
        Resample.AvgPool2(x, c, h, w, pooled);
        Resample.Upsample2(pooled, c, h / 2, w / 2, upsampled);
        var gU = upsampled.Select((u, i) => 0.5f * v[i] * v[i] * v[i] * u * u).ToArray();
        var gP = new float[full / 4];
        var gX = new float[full];
        Resample.Upsample2Backward(gU, c, h / 2, w / 2, gP);
        Resample.AvgPool2Backward(gP, c, h, w, gX);

        const float step = 1e-2f;
        for (int i = 0; i < full; i += 3)
        {
            float original = x[i];
            x[i] = original + step; double plus = Loss(x);
            x[i] = original - step; double minus = Loss(x);
            x[i] = original;
            double numeric = (plus - minus) / (2 * step);
            Assert.True(Math.Abs(numeric - gX[i]) <= 1e-3 + 1e-2 * Math.Abs(numeric), $"x[{i}]: analytic {gX[i]:G6} vs numeric {numeric:G6}");
        }
        Assert.Contains(gX, g => Math.Abs(g) > 1e-3);
    }

    private static double Dot(float[] a, float[] b) => a.Zip(b, (p, q) => (double)p * q).Sum();
}

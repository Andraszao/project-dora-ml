using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 36 (systems S1 + S2): the 512-bit kernels give exactly the reference kernels' values, and the roofline's
/// bookkeeping is right. The wide kernels are forced on here, so they're checked even on machines without AVX-512, where
/// .NET runs <c>Vector512</c> in software with the same per-lane arithmetic. Shapes cover strips that end ragged (widths
/// 16–40), every dilation the networks use, block and leftover channel counts, and rows too narrow for 16 lanes.
/// </summary>
[Trait(Bricks.Trait, Bricks.WideKernels)]
public class WideKernelTests
{
    private static float[] Random(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() - 0.5) * scale).ToArray();
    }

    private static void WithWide(bool wide, Action action)
    {
        bool before = ConvLayer.WideKernels;
        ConvLayer.WideKernels = wide;
        try { action(); }
        finally { ConvLayer.WideKernels = before; }
    }

    public static TheoryData<int, int, int, int, int> Shapes()
    {
        var data = new TheoryData<int, int, int, int, int>();
        foreach (var (cin, cout) in new[] { (1, 24), (24, 24), (24, 1), (9, 17), (16, 16) })
            foreach (int d in new[] { 1, 2, 4, 8 })
                foreach (var (h, w) in new[] { (5, 16), (7, 17), (3, 31), (6, 40), (4, 12) })
                    data.Add(cin, cout, d, h, w);
        return data;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void WideForwardIsBitIdenticalToTheReference(int cin, int cout, int d, int height, int width)
    {
        var layer = new ConvLayer(cin, cout, d, weightOffset: 3);
        var parameters = Random(3 + layer.ParameterCount, seed: cin * 100 + cout);
        var padded = new float[layer.PaddedLength(height, width)];
        layer.Pad(Random(cin * height * width, seed: d), height, width, padded);
        var expected = new float[cout * height * width];
        var actual = new float[expected.Length];
        layer.ForwardReference(parameters, padded, height, width, expected);
        WithWide(true, () => layer.Forward(parameters, padded, height, width, actual));
        Assert.Equal(expected, actual);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void WideBackwardIsBitIdenticalToTheReference(int cin, int cout, int d, int height, int width)
    {
        var layer = new ConvLayer(cin, cout, d, weightOffset: 0);
        var parameters = Random(layer.ParameterCount, seed: cin + 7 * cout);
        var padded = new float[layer.PaddedLength(height, width)];
        layer.Pad(Random(cin * height * width, seed: 2 * d), height, width, padded);
        var gradOutput = Random(cout * height * width, seed: 11 * d);
        var expectedGrads = new float[layer.ParameterCount];
        var actualGrads = new float[layer.ParameterCount];
        var expectedInput = new float[padded.Length];
        var actualInput = new float[padded.Length];
        layer.BackwardReference(parameters, padded, gradOutput, height, width, expectedGrads, expectedInput);
        WithWide(true, () => layer.Backward(parameters, padded, gradOutput, height, width, actualGrads, actualInput));
        Assert.Equal(expectedGrads, actualGrads);
        // The reference fills the padding border of the input gradient too; callers discard it, so compare the interior.
        for (int c = 0; c < cin; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int i = (c * layer.PaddedHeight(height) + y + d) * layer.PaddedWidth(width) + x + d;
                    Assert.Equal(expectedInput[i], actualInput[i]);
                }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WholeNetworksGiveTheSameOutputWithAndWithoutWideKernels(bool unet)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.UNet() : NetConfig.Default, seed: 4);
        var image = TestUtil.RandomImage(53, 37, seed: 9);
        GrayImage narrow = null!, wide = null!;
        WithWide(false, () => narrow = TiledInference.Denoise(net, image, tileCore: 24));
        WithWide(true, () => wide = TiledInference.Denoise(net, image, tileCore: 24));
        Assert.Equal(narrow.Pixels, wide.Pixels);
    }

    [Fact]
    public void RooflineCountsConvolutionWorkAndPicksTheNearerRoof()
    {
        Assert.Equal(2.0 * 24 * 24 * 9 * 172 * 172, Roofline.ConvFlops(24, 24, 172, 172, backward: false));
        Assert.Equal(2 * Roofline.ConvFlops(16, 32, 8, 9, backward: false), Roofline.ConvFlops(16, 32, 8, 9, backward: true));
        Assert.Equal("compute", new KernelFigure("fwd-512", "x", 1, 1, 1, 1, 50, ComputeShare: 0.8, BandwidthShare: 0.05).Bound);
        Assert.Equal("memory", new KernelFigure("fwd-256", "x", 1, 1, 1, 1, 2, ComputeShare: 0.2, BandwidthShare: 0.7).Bound);
    }
}

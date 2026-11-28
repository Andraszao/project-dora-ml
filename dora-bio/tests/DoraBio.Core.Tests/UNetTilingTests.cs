using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>Tiled U-Net inference must equal one whole-image pass, for ragged sizes and awkward tile sizes.</summary>
[Trait(Bricks.Trait, Bricks.UNetTiling)]
public class UNetTilingTests
{
    [Theory]
    [InlineData(2, 100, 77, 32)]    // ragged edge tiles in both directions
    [InlineData(2, 70, 53, 10)]     // core size not a multiple of 4 (rounded up to 12), unaligned image
    [InlineData(2, 64, 64, 16)]     // core much smaller than the receptive radius
    [InlineData(3, 97, 66, 24)]     // depth 3: everything on the 8-pixel grid
    [InlineData(2, 21, 9, 128)]     // one tile bigger than the image
    [InlineData(1, 5, 3, 8)]        // tinier than a pooling block in one direction
    public void TiledUNetInferenceMatchesWholeImage(int depth, int width, int height, int tileCore)
    {
        var net = DenoisingNet.Create(NetConfig.UNet(3, depth), seed: 2);
        var rng = new Random(1);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.3f;
        var image = TestUtil.RandomImage(width, height, 3);

        var whole = new float[image.Length];
        net.Predict(image.Pixels, height, width, whole, net.CreateWorkspace());
        var tiled = TiledInference.Denoise(net, image, tileCore);

        for (int i = 0; i < whole.Length; i++)
            Assert.True(Math.Abs(whole[i] - tiled.Pixels[i]) <= 1e-5f, $"pixel ({i % width}, {i / width}): {whole[i]} vs {tiled.Pixels[i]}");
    }

    [Fact]
    public void ATileStartingOffTheGridWouldNotMatch()
    {
        // Why the alignment rule exists: shifting the image by one pixel (not a multiple of 4) changes a U-Net's
        // output beyond the shift itself, so tiles that started off the grid could not reproduce the whole image.
        var net = DenoisingNet.Create(NetConfig.UNet(3, 2), seed: 2);
        var rng = new Random(1);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.3f;
        const int size = 128;   // the 27 px receptive field around the probe stays inside both crops
        var image = TestUtil.RandomImage(size + 4, size, 3);
        float[] Run(int offset)
        {
            var crop = image.Crop(offset, 0, size, size);
            var output = new float[size * size];
            net.Predict(crop.Pixels, size, size, output, net.CreateWorkspace());
            return output;
        }
        // The same image pixel (64, 64) in both crops, far from every crop edge.
        double DifferenceAfterShift(int shift) => Math.Abs(Run(0)[64 * size + 64] - Run(shift)[64 * size + 64 - shift]);
        Assert.True(DifferenceAfterShift(4) < 1e-5, "a shift by 4 (on the grid) must not change interior outputs");
        Assert.True(DifferenceAfterShift(1) > 1e-4, "a shift by 1 (off the grid) does change them");
    }
}

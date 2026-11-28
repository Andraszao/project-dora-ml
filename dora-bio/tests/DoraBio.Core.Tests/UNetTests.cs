using System.Buffers.Binary;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// The U-Net: whole-network gradients against finite differences (brick 1's method), its size and starting point,
/// weights format 2 for both architectures with format 1 still readable, and that it learns through the real trainer.
/// </summary>
[Trait(Bricks.Trait, Bricks.UNet)]
public class UNetTests
{
    // A finite difference is only valid if the step doesn't cross a ReLU kink. On small images with lots of zero padding
    // (where a unit's pre-activation is just its bias) a 1e-2 step sometimes does: at those parameters the difference
    // converges to the analytic value as the step shrinks (−0.0224 at 1e-2, −0.02538 at 1e-3, analytic −0.02538). So a
    // parameter passes if either a 1e-2 or a 1e-3 step agrees. A wrong gradient disagrees at both.
    [Theory]
    [InlineData(2, 1, 8, 8)]     // one pooling level, square
    [InlineData(2, 2, 8, 12)]    // two levels, non-square
    [InlineData(3, 2, 4, 8)]     // two levels down to a 1×2 bottleneck
    [InlineData(2, 2, 6, 10)]    // not a multiple of 4: the zero-padding path and its loss mask
    public void WholeNetworkGradientsMatchFiniteDifferences(int features, int depth, int height, int width)
    {
        var net = DenoisingNet.Create(NetConfig.UNet(features, depth), seed: 11);
        var rng = new Random(5);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.2f;

        int plane = height * width;
        var input = TestUtil.RandomPlane(plane, 1);
        var target = TestUtil.RandomPlane(plane, 2);
        var mask = TestUtil.RandomPlane(plane, 3).Select(v => v > 0.5f ? 1f : 0f).ToArray();
        var ws = net.CreateWorkspace();
        float scale = 1f / mask.Sum();

        var analytic = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, mask, height, width, scale, analytic, ws);
        double Loss() => net.AccumulateGradients(input, target, mask, height, width, scale, new float[net.ParameterCount], ws) * scale;

        double Numeric(int i, float step)
        {
            float original = net.Parameters[i];
            net.Parameters[i] = original + step; double plus = Loss();
            net.Parameters[i] = original - step; double minus = Loss();
            net.Parameters[i] = original;
            return (plus - minus) / (2 * step);
        }
        bool Agrees(double numeric, double a) => Math.Abs(numeric - a) <= 2e-3 + 2e-2 * Math.Max(Math.Abs(numeric), Math.Abs(a));

        // A spread of parameters through every layer (encoder, bottleneck, decoder with its skip inputs, output).
        int checkedCount = 0;
        for (int i = 0; i < net.ParameterCount; i += 5)
        {
            double a = analytic[i], coarse = Numeric(i, 1e-2f);
            if (!Agrees(coarse, a))
            {
                double fine = Numeric(i, 1e-3f);
                Assert.True(Agrees(fine, a), $"param {i} of {net.ParameterCount}: analytic {a:G6} vs numeric {coarse:G6} (1e-2), {fine:G6} (1e-3)");
            }
            checkedCount++;
        }
        Assert.True(checkedCount > 20);
        Assert.True(analytic.Count(g => Math.Abs(g) > 1e-4) > net.ParameterCount / 4, "gradients must reach most parameters");
    }

    [Fact]
    public void UnalignedSizesBehaveExactlyLikeTheZeroPaddedImage()
    {
        // 6×10 isn't a multiple of 4. Internally it is zero-padded to 8×12 with mask 0 on the padding; doing that by hand
        // must give bit-identical loss, gradients and output.
        var net = DenoisingNet.Create(NetConfig.UNet(2, 2), seed: 7);
        const int h = 6, w = 10, ph = 8, pw = 12;
        var input = TestUtil.RandomPlane(h * w, 1);
        var target = TestUtil.RandomPlane(h * w, 2);
        var mask = TestUtil.RandomPlane(h * w, 3).Select(v => v > 0.5f ? 1f : 0f).ToArray();
        float[] Pad(float[] plane)
        {
            var padded = new float[ph * pw];
            for (int y = 0; y < h; y++) Array.Copy(plane, y * w, padded, y * pw, w);
            return padded;
        }

        var gInternal = new float[net.ParameterCount];
        var gManual = new float[net.ParameterCount];
        double lossInternal = net.AccumulateGradients(input, target, mask, h, w, 0.5f, gInternal, net.CreateWorkspace());
        double lossManual = net.AccumulateGradients(Pad(input), Pad(target), Pad(mask), ph, pw, 0.5f, gManual, net.CreateWorkspace());
        Assert.Equal(lossManual, lossInternal);
        Assert.Equal(gManual, gInternal);

        var outInternal = new float[h * w];
        var outManual = new float[ph * pw];
        net.Predict(input, h, w, outInternal, net.CreateWorkspace());
        net.Predict(Pad(input), ph, pw, outManual, net.CreateWorkspace());
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                Assert.Equal(outManual[y * pw + x], outInternal[y * w + x]);
    }

    [Fact]
    public void TheDefaultUNetHasTheDocumentedSizeAndStartsNearTheIdentity()
    {
        var config = NetConfig.UNet();
        Assert.Equal(118_113, DenoisingNet.CountParameters(config));   // 16→32→64 features, worked out by hand in RESULTS.md
        Assert.Equal(11, config.LayerCount);
        Assert.Equal(4, config.TileAlignment);
        Assert.Equal(27, config.ReceptiveRadius);
        Assert.Contains("U-Net, depth 2, 16→32→64", config.ToString());

        var net = DenoisingNet.Create(config, seed: 1);
        var input = TestUtil.RandomPlane(32 * 32, 4);
        var output = new float[input.Length];
        net.Predict(input, 32, 32, output, net.CreateWorkspace());
        Assert.InRange(input.Zip(output, (x, y) => Math.Abs(x - y)).Max(), 0, 0.5);
    }

    [Fact]
    public void ConfigsCompareByContentAndRejectNonsense()
    {
        Assert.Equal(NetConfig.UNet(8, 3), NetConfig.UNet(8, 3));
        Assert.NotEqual(NetConfig.UNet(8, 3), NetConfig.UNet(8, 2));
        Assert.NotEqual<NetConfig>(NetConfig.UNet(24, 2), new NetConfig(24, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => NetConfig.UNet(8, 0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => NetConfig.UNet(8, 5).Validate());
        Assert.Throws<ArgumentException>(() => (NetConfig.UNet(8, 2) with { Dilations = [1, 1] }).Validate());
    }

    [Fact]
    public void WeightsFormat2RoundTripsBothArchitecturesAndFormat1StillLoads()
    {
        foreach (var config in new[] { NetConfig.UNet(4, 2), new NetConfig(5, [1, 2, 1]) })
        {
            var net = DenoisingNet.Create(config, seed: 3);
            var (loaded, info) = WeightsFile.Load(WeightsFile.Serialize(net, "round trip", DateTime.UnixEpoch));
            Assert.Equal(config, loaded.Config);
            Assert.Equal(config, info.Config);
            Assert.Equal(net.Parameters, loaded.Parameters);
        }

        // A format-1 file is format 2 without the architecture byte. Build one from a format-2 file and load it.
        var dilated = DenoisingNet.Create(new NetConfig(3, [1, 1]), seed: 4);
        var v2 = WeightsFile.Serialize(dilated, "old", DateTime.UnixEpoch);
        int archAt = 8 + 2 + 1 + WeightsFile.ModelVersion.Length + 1 + "old".Length + 8;
        var v1 = v2[..archAt].Concat(v2[(archAt + 1)..^4]).ToArray();
        BinaryPrimitives.WriteUInt16LittleEndian(v1.AsSpan(8), 1);
        var crc = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(crc, Crc32.Compute(v1));
        var (old, oldInfo) = WeightsFile.Load(v1.Concat(crc).ToArray());
        Assert.Equal(dilated.Config, old.Config);
        Assert.Equal(dilated.Parameters, old.Parameters);
        Assert.Equal("old", oldInfo.Provenance);
    }

    [Fact]
    public void AUNetLearnsToDenoiseThroughTheTrainer()
    {
        var data = SyntheticFluorescence.GenerateDataset(count: 5, size: 96, seed: 10, new SyntheticOptions { PeakPhotons = 15 });
        var net = DenoisingNet.Create(NetConfig.UNet(4, 2), seed: 1);
        var config = new TrainerConfig
        {
            Mode = TrainingMode.Supervised, PatchSize = 32, BatchSize = 8, StepsPerEpoch = 25, MaxEpochs = 6,
            LearningRate = 3e-3f, ValidationPatches = 8, EarlyStopPatience = 6, Seed = 3, EnableProgressiveTargets = false,
        };
        var result = new Trainer(net, config).Train(data);
        Assert.NotNull(result.Best);
        Assert.True(result.Best!.ValPsnrGain > 2.0, $"PSNR gain {result.Best.ValPsnrGain:0.00} dB");
    }
}

using System.Buffers.Binary;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Quality pass Q13: the noise-conditioned (FiLM) network. Whole-network finite-difference gradients through the
/// generator; an untrained conditioned network is the plain one bit for bit; the condition really changes the output;
/// tiled inference gives every tile the whole frame's condition, so it still equals whole-image inference; weights
/// format 3 round-trips while plain networks keep writing format 2; training patches carry their frame's condition.
/// </summary>
[Trait(Bricks.Trait, Bricks.NoiseConditioning)]
public class NoiseConditioningTests
{
    private static NetConfig Conditioned(NetConfig c) => c with { Conditioning = NetConditioning.NoiseLevel };

    /// <summary>A conditioned network whose generator is far from its zero start, so γ and β really vary with s.</summary>
    private static DenoisingNet Perturbed(NetConfig config, int seed)
    {
        var net = DenoisingNet.Create(config, seed);
        var rng = new Random(seed + 1);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.2f;
        return net;
    }

    [Theory]
    [InlineData(true, 8, 12)]    // U-Net, two pooling levels
    [InlineData(true, 6, 10)]    // U-Net on an unaligned size: the zero-padding path
    [InlineData(false, 9, 7)]    // the dilated stack
    public void WholeNetworkGradientsMatchFiniteDifferencesThroughTheGenerator(bool unet, int height, int width)
    {
        var plain = unet ? NetConfig.UNet(2, 2) : new NetConfig(3, [1, 2, 1]);
        var net = Perturbed(Conditioned(plain), 11);
        int convParameters = DenoisingNet.CountParameters(plain);
        const float condition = 0.7f;

        int pixels = height * width;
        var input = TestUtil.RandomPlane(pixels, 1);
        var target = TestUtil.RandomPlane(pixels, 2);
        var mask = TestUtil.RandomPlane(pixels, 3).Select(v => v > 0.5f ? 1f : 0f).ToArray();
        var ws = net.CreateWorkspace();
        float scale = 1f / mask.Sum();

        var analytic = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, mask, height, width, scale, analytic, ws, condition);
        double Loss() => net.AccumulateGradients(input, target, mask, height, width, scale, new float[net.ParameterCount], ws, condition) * scale;
        double Numeric(int i, float step)
        {
            float original = net.Parameters[i];
            net.Parameters[i] = original + step; double plus = Loss();
            net.Parameters[i] = original - step; double minus = Loss();
            net.Parameters[i] = original;
            return (plus - minus) / (2 * step);
        }
        bool Agrees(double numeric, double a) => Math.Abs(numeric - a) <= 2e-3 + 2e-2 * Math.Max(Math.Abs(numeric), Math.Abs(a));

        // Every 5th convolution parameter, every W₁/b₁ entry, and every 3rd entry of W₂/b₂.
        var indices = Enumerable.Range(0, convParameters).Where(i => i % 5 == 0)
            .Concat(Enumerable.Range(convParameters, 2 * DenoisingNet.FilmHidden))
            .Concat(Enumerable.Range(convParameters + 2 * DenoisingNet.FilmHidden,
                net.ParameterCount - convParameters - 2 * DenoisingNet.FilmHidden).Where(i => i % 3 == 0)).ToList();
        foreach (int i in indices)
        {
            double a = analytic[i], coarse = Numeric(i, 1e-2f);
            if (Agrees(coarse, a)) continue;
            double fine = Numeric(i, 1e-3f);
            Assert.True(Agrees(fine, a), $"param {i} of {net.ParameterCount} (generator from {convParameters}): analytic {a:G6} vs numeric {coarse:G6} (1e-2), {fine:G6} (1e-3)");
        }
        var generator = analytic.AsSpan(convParameters).ToArray();
        Assert.True(generator.Count(g => Math.Abs(g) > 1e-5) > generator.Length / 4, "gradients must reach the generator");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheChainRuleThroughTheGeneratorMatchesFiniteDifferencesTightly(bool unet)
    {
        // The whole-network check above runs near float precision, so its absolute tolerance is larger than many of the
        // generator's gradients: it proves they flow, not that every factor is right. This checks the chain rule alone,
        // tightly. For a fixed vector v, f(θ) = ⟨v, effective parameters(θ)⟩ is smooth and cheap, and its gradient must
        // be exactly what the backward pass gives for an effective-parameter gradient of v.
        var plain = unet ? NetConfig.UNet(2, 2) : new NetConfig(3, [1, 2, 1]);
        var net = Perturbed(Conditioned(plain), 21);
        int convParameters = DenoisingNet.CountParameters(plain);
        const float condition = 0.7f;
        var v = TestUtil.RandomPlane(convParameters, 22, -1f, 1f);
        var ws = net.CreateWorkspace();
        double F()
        {
            var effective = net.ActiveParameters(condition, ws);
            double f = 0;
            for (int k = 0; k < convParameters; k++) f += v[k] * (double)effective[k];
            return f;
        }
        F();
        var analytic = new float[net.ParameterCount];
        net.ConditioningBackward(v, analytic, ws);
        double Numeric(int i, float step)
        {
            float original = net.Parameters[i];
            net.Parameters[i] = original + step; double plus = F();
            net.Parameters[i] = original - step; double minus = F();
            net.Parameters[i] = original;
            return (plus - minus) / (2 * step);
        }
        bool Agrees(double numeric, double a) => Math.Abs(numeric - a) <= 1e-3 + 2e-3 * Math.Max(Math.Abs(numeric), Math.Abs(a));
        for (int i = 0; i < net.ParameterCount; i++)
        {
            double a = analytic[i], coarse = Numeric(i, 1e-2f);
            if (Agrees(coarse, a)) continue;
            double fine = Numeric(i, 1e-3f);   // a step across one of the generator's ReLU kinks
            Assert.True(Agrees(fine, a), $"param {i} (generator from {convParameters}): analytic {a:G6} vs numeric {coarse:G6} (1e-2), {fine:G6} (1e-3)");
        }
        var generator = analytic.AsSpan(convParameters).ToArray();
        Assert.True(generator.Count(g => Math.Abs(g) > 0.05) > generator.Length / 4, "generator gradients must be large next to the tolerance");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AnUntrainedConditionedNetworkIsThePlainOneBitForBit(bool unet)
    {
        var plainConfig = unet ? NetConfig.UNet(4, 2) : new NetConfig(6, [1, 2, 4, 2, 1]);
        var plain = DenoisingNet.Create(plainConfig, seed: 3);
        var conditioned = DenoisingNet.Create(Conditioned(plainConfig), seed: 3);
        int convParameters = plain.ParameterCount;
        Assert.Equal(plain.Parameters, conditioned.Parameters.AsSpan(0, convParameters).ToArray());

        const int h = 20, w = 24;
        var input = TestUtil.RandomPlane(h * w, 4);
        var target = TestUtil.RandomPlane(h * w, 5);
        var expected = new float[h * w];
        plain.Predict(input, h, w, expected, plain.CreateWorkspace());
        var expectedGradient = new float[convParameters];
        plain.AccumulateGradients(input, target, [], h, w, 0.01f, expectedGradient, plain.CreateWorkspace());
        foreach (float condition in new[] { -2f, 0f, 1.5f })
        {
            var output = new float[h * w];
            conditioned.Predict(input, h, w, output, conditioned.CreateWorkspace(), condition);
            Assert.Equal(expected, output);
            var gradient = new float[conditioned.ParameterCount];
            conditioned.AccumulateGradients(input, target, [], h, w, 0.01f, gradient, conditioned.CreateWorkspace(), condition);
            Assert.Equal(expectedGradient, gradient.AsSpan(0, convParameters).ToArray());
        }
    }

    [Fact]
    public void TheConditionChangesTheOutputAndIsRequired()
    {
        var net = Perturbed(Conditioned(NetConfig.UNet(4, 2)), 7);
        const int h = 16, w = 16;
        var input = TestUtil.RandomPlane(h * w, 8);
        var low = new float[h * w];
        var high = new float[h * w];
        var ws = net.CreateWorkspace();
        net.Predict(input, h, w, low, ws, -1f);
        net.Predict(input, h, w, high, ws, 1f);
        Assert.True(low.Zip(high).Max(p => Math.Abs(p.First - p.Second)) > 1e-3f, "the condition must matter");

        Assert.Throws<ArgumentException>(() => net.Predict(input, h, w, low, ws));
        Assert.Throws<ArgumentException>(() => net.AccumulateGradients(input, input, [], h, w, 1f, new float[net.ParameterCount], ws));
        var plain = DenoisingNet.Create(NetConfig.UNet(4, 2), 7);
        plain.Predict(input, h, w, low, plain.CreateWorkspace());   // a plain network needs none
    }

    [Theory]
    [InlineData(70, 53, 16)]
    [InlineData(33, 41, 8)]
    public void TiledInferenceGivesEveryTileTheWholeFramesCondition(int width, int height, int core)
    {
        var net = Perturbed(Conditioned(NetConfig.UNet(4, 2)), 9);
        var image = TestUtil.AddGaussianNoise(TestUtil.Pattern(width, height, (x, y) => (x * 7 + y * 3) % 17 / 17f), 0.1, 2);
        var tiled = TiledInference.Denoise(net, image, core);
        var whole = new float[image.Length];
        net.Predict(image.Pixels, height, width, whole, net.CreateWorkspace(), NoiseConditioning.Of(image));
        Assert.True(whole.Zip(tiled.Pixels).Max(p => Math.Abs(p.First - p.Second)) <= 1e-5f);
    }

    [Fact]
    public void ConditionIsTheLogOfTheFramesNoiseRelativeToTheReference()
    {
        Assert.Equal(0f, NoiseConditioning.FromSigma(NoiseConditioning.ReferenceSigma), 6);
        Assert.Equal(MathF.Log(2f), NoiseConditioning.FromSigma(2 * NoiseConditioning.ReferenceSigma), 5);
        Assert.True(float.IsFinite(NoiseConditioning.FromSigma(0)));
        var image = TestUtil.AddGaussianNoise(new GrayImage(64, 64), 0.1, 3);
        Assert.Equal(NoiseConditioning.FromSigma(Metrics.EstimateNoiseSigma(image)), NoiseConditioning.Of(image));
    }

    [Fact]
    public void WeightsFormat3RoundTripsAndPlainNetworksStayOnFormat2()
    {
        var net = Perturbed(Conditioned(NetConfig.UNet(4, 2)), 12);
        var bytes = WeightsFile.Serialize(net, "q13 test", new DateTime(2026, 11, 26, 0, 0, 0, DateTimeKind.Utc));
        Assert.Equal(3, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8)));
        var (loaded, info) = WeightsFile.Load(bytes);
        Assert.Equal(NetConditioning.NoiseLevel, info.Config.Conditioning);
        Assert.Equal(net.Config, loaded.Config);
        Assert.Equal(net.Parameters, loaded.Parameters);
        Assert.Contains("noise-conditioned", net.Config.ToString());

        var plain = WeightsFile.Serialize(DenoisingNet.Create(NetConfig.UNet(4, 2)), "plain", DateTime.UnixEpoch);
        Assert.Equal(2, BinaryPrimitives.ReadUInt16LittleEndian(plain.AsSpan(8)));
        Assert.NotEqual(NetConfig.UNet(4, 2), Conditioned(NetConfig.UNet(4, 2)));
    }

    [Fact]
    public void TrainingPatchesCarryTheirWholeFramesCondition()
    {
        // Three frames at three noise levels; every sampled patch must carry exactly its source frame's condition.
        var images = new[] { 0.02, 0.08, 0.3 }.Select((sigma, i) => new ImagePair(
            TestUtil.AddGaussianNoise(TestUtil.Pattern(48, 40, (x, y) => (x + y) % 9 / 9f), sigma, 20 + i), null, $"f{i}", $"f{i}")).ToList();
        var allowed = images.Select(p => NoiseConditioning.Of(p.Noisy)).ToHashSet();
        Assert.Equal(3, allowed.Count);
        var sampler = new PatchSampler(images, 16, TrainingMode.Noise2Void, 0.02f, new Random(4));
        var batch = new TrainingBatch(24, 16);
        sampler.Fill(batch);
        Assert.All(batch.Conditions, c => Assert.Contains(c, allowed));
        Assert.True(batch.Conditions.Distinct().Count() >= 2);
    }

    [Fact]
    public void AConditionedUNetLearnsThroughTheTrainerReproducibly()
    {
        // Two noise levels in one dataset, the situation conditioning is for.
        var data = SyntheticFluorescence.GenerateDataset(count: 3, size: 96, seed: 10, new SyntheticOptions { PeakPhotons = 6 })
            .Concat(SyntheticFluorescence.GenerateDataset(count: 3, size: 96, seed: 40, new SyntheticOptions { PeakPhotons = 60 })).ToList();
        var config = new TrainerConfig
        {
            Mode = TrainingMode.Supervised, PatchSize = 32, BatchSize = 8, StepsPerEpoch = 25, MaxEpochs = 6,
            LearningRate = 3e-3f, ValidationPatches = 8, EarlyStopPatience = 6, Seed = 3, EnableProgressiveTargets = false,
        };
        DenoisingNet Run(out TrainingResult result)
        {
            var net = DenoisingNet.Create(Conditioned(NetConfig.UNet(4, 2)), seed: 1);
            result = new Trainer(net, config).Train(data);
            return net;
        }
        var first = Run(out var result);
        var second = Run(out _);
        Assert.Equal(first.Parameters, second.Parameters);
        Assert.NotNull(result.Best);
        Assert.True(result.Best!.ValPsnrGain > 2.0, $"PSNR gain {result.Best.ValPsnrGain:0.00} dB");
    }
}

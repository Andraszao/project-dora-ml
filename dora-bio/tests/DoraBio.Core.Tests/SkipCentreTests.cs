using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 40 (Track A): skip-centre Noise2Noise. Its input is (t − 1, t + 1) and never frame t, at any index of any
/// sequence, in training, validation and inference; its output is the neighbours' mean minus a correction; gradients,
/// tiling, streaming and the weights file (format 5) hold as for every other network. Also the per-frame validation
/// blocks that give every arm the same frames and positions.
/// </summary>
[Trait(Bricks.Trait, Bricks.TimeLapseModes)]
public class SkipCentreTests
{
    private static NetConfig Dilated => NetConfig.SkipCentre(new NetConfig(4, [1, 2, 1]));

    [Fact]
    public void NoInputIndexIsEverTheTargetFrame()
    {
        Assert.Equal([1, 1], Sequences.NeighbourIndices(0, 5));   // reflected at the start
        Assert.Equal([1, 3], Sequences.NeighbourIndices(2, 5));
        Assert.Equal([3, 3], Sequences.NeighbourIndices(4, 5));   // and at the end
        Assert.Equal([1, 1], Sequences.NeighbourIndices(0, 2));   // T = 2: (1, 1), then (0, 0)
        Assert.Equal([0, 0], Sequences.NeighbourIndices(1, 2));
        Assert.Throws<ArgumentException>(() => Sequences.NeighbourIndices(0, 1));   // T = 1: no neighbour at all
        for (int count = 2; count <= 12; count++)
            for (int t = 0; t < count; t++)
            {
                var indices = Sequences.NeighbourIndices(t, count);
                Assert.DoesNotContain(t, indices);
                Assert.All(indices, i => Assert.InRange(i, 0, count - 1));
                Assert.All(indices, i => Assert.Equal(1, Math.Abs(i - t)));   // always an adjacent frame
            }

        // Training images: interior frames only, each the target with its two true neighbours as context.
        var sequence = SyntheticFluorescence.GenerateSequence(24, 20, 6, seed: 3).Select(f => f.Noisy).ToList();
        var images = Sequences.SkipCentre(sequence, "s");
        Assert.Equal(4, images.Count);
        var normalized = sequence.Select(f => Normalization.Apply(f, Normalization.FromPercentiles(f.Pixels))).ToList();
        for (int k = 0; k < images.Count; k++)
        {
            int t = k + 1;
            Assert.Equal(normalized[t].Pixels, images[k].Noisy.Pixels);
            Assert.Equal(normalized[t - 1].Pixels, images[k].Context![0].Pixels);
            Assert.Equal(normalized[t + 1].Pixels, images[k].Context![1].Pixels);
            Assert.All(images[k].Context!, c => Assert.NotEqual(images[k].Noisy.Pixels, c.Pixels));
        }
        Assert.Throws<ArgumentException>(() => Sequences.SkipCentre(sequence.Take(2).ToList(), "short"));   // no interior frame
    }

    [Fact]
    public void TheSamplerNeverWritesTheTargetIntoTheInput()
    {
        // Neighbours are constant planes and the frame is noise: every input value must be one of the constants, and the
        // target the frame's own patch with every pixel in the loss.
        var frame = TestUtil.RandomImage(48, 48, 3);
        GrayImage Constant(float v) => new(48, 48, Enumerable.Repeat(v, 48 * 48).ToArray());
        var images = new List<ImagePair> { new(frame, null, "g", "n") { Context = [Constant(0.25f), Constant(0.75f)] } };
        var sampler = new PatchSampler(images, 32, TrainingMode.SkipCentre, 0.02f, new Random(1));
        Assert.Equal(2, sampler.InputFrames);
        var batch = new TrainingBatch(8, 32, 2);
        sampler.Fill(batch);
        var framePixels = frame.Pixels.ToHashSet();
        for (int i = 0; i < batch.Count; i++)
        {
            var input = batch.Input(i);
            Assert.All(input[..batch.Plane].ToArray(), v => Assert.Equal(0.25f, v));
            Assert.All(input[batch.Plane..].ToArray(), v => Assert.Equal(0.75f, v));
            Assert.All(batch.Mask(i).ToArray(), m => Assert.Equal(1f, m));
            Assert.All(batch.Target(i).ToArray(), v => Assert.Contains(v, framePixels));
        }
        // Without both neighbours, or with a frame count the network doesn't take, it refuses.
        Assert.Throws<ArgumentException>(() => new PatchSampler([new ImagePair(frame, null, "g", "n")], 32, TrainingMode.SkipCentre, 0.02f, new Random(1)));
        Assert.Throws<ArgumentException>(() => sampler.Fill(new TrainingBatch(2, 32, 3)));
    }

    [Theory]
    [InlineData(false, 16, 16)]
    [InlineData(true, 16, 16)]
    [InlineData(true, 13, 17)]
    public void TheResidualIsSubtractedFromTheNeighboursMean(bool unet, int height, int width)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.SkipCentre(NetConfig.UNet(4, 2)) : Dilated, 1);
        Array.Clear(net.Parameters);   // predicts no correction: the output must be ½(before + after), exactly
        int plane = height * width;
        var input = TestUtil.RandomPlane(2 * plane, 9);
        var output = new float[plane];
        net.Predict(input, height, width, output, net.CreateWorkspace());
        for (int i = 0; i < plane; i++) Assert.Equal(0.5f * (input[i] + input[plane + i]), output[i]);
        Assert.Throws<ArgumentException>(() => net.Predict(TestUtil.RandomPlane(3 * plane, 1), height, width, output, net.CreateWorkspace()));
    }

    [Theory]
    [InlineData(false, 8, 9)]
    [InlineData(true, 8, 8)]
    [InlineData(true, 7, 9)]   // unaligned: both frames padded, the base computed before padding
    public void SkipCentreGradientsMatchFiniteDifferences(bool unet, int height, int width)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.SkipCentre(NetConfig.UNet(3, 1)) : Dilated, seed: 11);
        var rng = new Random(5);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.2f;
        int plane = height * width;
        var input = TestUtil.RandomPlane(2 * plane, 1);
        var target = TestUtil.RandomPlane(plane, 2);
        var mask = Enumerable.Repeat(1f, plane).ToArray();   // every pixel, as in skip-centre training
        var ws = net.CreateWorkspace();
        float scale = 1f / plane;
        var analytic = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, mask, height, width, scale, analytic, ws);
        double Loss() => net.AccumulateGradients(input, target, mask, height, width, scale, new float[net.ParameterCount], ws) * scale;

        int firstWeights = net.Layers[0].WeightCount;
        var indices = Enumerable.Range(0, net.ParameterCount).Where(i => i < firstWeights || i % 5 == 0).ToList();
        const float h = 1e-3f;
        foreach (int i in indices)
        {
            float original = net.Parameters[i];
            net.Parameters[i] = original + h;
            double plus = Loss();
            net.Parameters[i] = original - h;
            double minus = Loss();
            net.Parameters[i] = original;
            double numeric = (plus - minus) / (2 * h), a = analytic[i];
            Assert.True(Math.Abs(numeric - a) <= 2e-3 + 2e-2 * Math.Max(Math.Abs(numeric), Math.Abs(a)), $"param {i}: analytic {a:G6} vs numeric {numeric:G6}");
        }
        // Both input frames feed the network: first-layer weights of each get gradient.
        Assert.Equal(net.Config.Features * 2 * 9, firstWeights);
        Assert.True(Enumerable.Range(0, firstWeights).Any(i => Math.Abs(analytic[i]) > 1e-4 && i % 18 / 9 == 0), "no gradient reaches frame t − 1");
        Assert.True(Enumerable.Range(0, firstWeights).Any(i => Math.Abs(analytic[i]) > 1e-4 && i % 18 / 9 == 1), "no gradient reaches frame t + 1");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TiledEqualsWholeFrame(bool unet)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.SkipCentre(NetConfig.UNet(4, 2)) : Dilated, 5);
        const int w = 70, h = 53;
        var frames = Enumerable.Range(0, 2).Select(f => TestUtil.RandomImage(w, h, 20 + f)).ToList();
        var whole = new float[w * h];
        net.Predict(frames.SelectMany(f => f.Pixels).ToArray(), h, w, whole, net.CreateWorkspace());
        var tiled = TiledInference.Denoise(net, frames, tileCore: 16);
        for (int i = 0; i < whole.Length; i++) Assert.True(Math.Abs(whole[i] - tiled.Pixels[i]) <= 1e-5f, $"pixel {i}: {whole[i]} vs {tiled.Pixels[i]}");
        Assert.Throws<ArgumentException>(() => TiledInference.Denoise(net, frames.Take(1).ToList()));
        Assert.Throws<ArgumentException>(() => TiledInference.Denoise(net, [frames[0], frames[1], frames[0]]));
    }

    [Fact]
    public void SkipCentreFilesAreFormatFiveAndOthersUnchanged()
    {
        var stamp = new DateTime(2026, 11, 27, 0, 0, 0, DateTimeKind.Utc);
        Assert.Equal(2, BitConverter.ToUInt16(WeightsFile.Serialize(DenoisingNet.Create(NetConfig.Default, 1), "p", stamp), 8));
        Assert.Equal(4, BitConverter.ToUInt16(WeightsFile.Serialize(DenoisingNet.Create(NetConfig.UNet(4, 1) with { InputFrames = 3 }, 1), "p", stamp), 8));
        foreach (var config in new[] { Dilated, NetConfig.SkipCentre(NetConfig.UNet(4, 1)) })
        {
            var net = DenoisingNet.Create(config, 7);
            Assert.Equal(config.Features * 2 * 9 + config.Features, net.Layers[0].ParameterCount);   // only the first layer differs
            var file = WeightsFile.Serialize(net, "skip-centre", stamp);
            Assert.Equal(5, BitConverter.ToUInt16(file, 8));
            var (loaded, info) = WeightsFile.Load(file);
            Assert.Equal(config, info.Config);
            Assert.True(loaded.Config.SkipsCentre);
            Assert.Equal(2, loaded.Config.InputFrames);
            Assert.Equal(3, loaded.Config.WindowFrames);
            Assert.Equal(net.Parameters, loaded.Parameters);
            Assert.Contains("skip-centre", info.Config.ToString());

            // A format-5 file whose residual-base byte says "centre frame" is refused, not loaded as a centre network.
            int baseByte = file.Length - 4 - 4 * net.ParameterCount - 4 - 1;
            Assert.Equal((byte)NetResidualBase.NeighbourMean, file[baseByte]);
            var tampered = (byte[])file.Clone();
            tampered[baseByte] = (byte)NetResidualBase.CentreFrame;
            BitConverter.GetBytes(Crc32.Compute(tampered.AsSpan(0, tampered.Length - 4))).CopyTo(tampered, tampered.Length - 4);
            Assert.Throws<InvalidDataException>(() => WeightsFile.Load(tampered));
        }
        // Two frames are only a skip-centre network; a skip-centre network takes exactly two.
        Assert.Throws<ArgumentOutOfRangeException>(() => (NetConfig.Default with { InputFrames = 2 }).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => (Dilated with { InputFrames = 3 }).Validate());
        Assert.Throws<ArgumentException>(() => NetConfig.SkipCentre(NetConfig.UNet() with { Conditioning = NetConditioning.NoiseLevel }).Validate());
        Assert.NotEqual(Dilated, new NetConfig(4, [1, 2, 1]) with { InputFrames = 2 });   // the residual base is part of equality
    }

    [Fact]
    public void InferenceNeverReadsTheFramesPixels()
    {
        // Frame t sets only its intensity map (from its percentiles) and the noise check. Shuffling its pixels keeps
        // both and changes every pixel, so if any of them reached the network the output would change.
        var net = DenoisingNet.Create(Dilated, 3);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] *= 3f;   // a network that does something
        var denoiser = new FrameDenoiser(net, tileCore: 16);
        var raw = SyntheticFluorescence.GenerateSequence(40, 32, 3, seed: 2).Select(f => f.Noisy).ToList();
        var order = new Random(4);
        var shuffled = raw[1].Pixels.OrderBy(_ => order.Next()).ToArray();
        var original = denoiser.Denoise(raw[1], [raw[0], raw[2]]).Output;
        var permuted = denoiser.Denoise(new GrayImage(40, 32, shuffled), [raw[0], raw[2]]).Output;
        Assert.Equal(original.Pixels, permuted.Pixels);
        Assert.NotEqual(raw[1].Pixels, shuffled);
        Assert.Throws<ArgumentException>(() => denoiser.Denoise(raw[1]));                     // needs its neighbours
        Assert.Throws<ArgumentException>(() => denoiser.Denoise(raw[1], [raw[0]]));           // both of them
    }

    [Fact]
    public void StacksStreamWithAThreeFrameWindow()
    {
        var net = DenoisingNet.Create(Dilated, 3);
        var denoiser = new FrameDenoiser(net, tileCore: 16);
        foreach (int pages in new[] { 2, 3, 6 })   // the shortest stack, the shortest with an interior frame, and longer
        {
            var raw = SyntheticFluorescence.GenerateSequence(40, 32, pages, seed: pages).Select(f => f.Noisy).ToList();
            using var input = new TiffStackReader(new MemoryStream(TiffCodec.EncodeStack(raw, TiffSampleType.UInt16)));
            using var stream = new MemoryStream();
            using (var output = new TiffStackWriter(stream, TiffSampleType.Float32, leaveOpen: true))
                Assert.Equal(pages, BatchDenoising.DenoiseStack(denoiser, input, output));
            using var written = new TiffStackReader(new MemoryStream(stream.ToArray()));
            for (int t = 0; t < pages; t++)   // each page equals its frame denoised from its reflected neighbours
            {
                var n = Sequences.NeighbourIndices(t, pages);
                Assert.Equal(denoiser.Denoise(raw[t], [raw[n[0]], raw[n[1]]]).Output.Pixels, written.ReadPage(t).Pixels);
            }
        }
        using var single = new TiffStackReader(new MemoryStream(TiffCodec.EncodeStack([TestUtil.RandomImage(32, 32, 1)], TiffSampleType.Float32)));
        using var nowhere = new TiffStackWriter(new MemoryStream(), TiffSampleType.Float32);
        Assert.Throws<ArgumentException>(() => BatchDenoising.DenoiseStack(denoiser, single, nowhere));   // T = 1 refused
    }

    [Fact]
    public void ValidationBlocksGiveEveryArmTheSameFramesAndPositions()
    {
        var block = SyntheticFluorescence.GenerateSequence(80, 72, 12, seed: 6).Select(f => f.Noisy).ToList();
        var config = new TrainerConfig { PatchSize = 32, ValidationPatches = 7, Seed = 2026 };
        var arms = new (TrainingMode Mode, int Frames)[]
            { (TrainingMode.Noise2Void, 1), (TrainingMode.Noise2Noise, 1), (TrainingMode.Noise2Void, 3), (TrainingMode.SkipCentre, 2) };
        var sets = arms.Select(a =>
        {
            var images = Sequences.ValidationFrames(block, a.Mode, a.Frames, "movie", firstFrame: 1210);
            Assert.Equal(10, images.Count);   // interior frames 1211 … 1220 only
            Assert.Equal("movie/v1211", images[0].GroupId);
            return new ValidationSet(images, config with { Mode = a.Mode });
        }).ToList();
        foreach (var set in sets.Skip(1))
        {
            Assert.Equal(sets[0].PatchSources, set.PatchSources);
            Assert.Equal(sets[0].PatchOrigins, set.PatchOrigins);
        }
        Assert.Equal(7, sets[0].PatchSources.Distinct().Count());   // seven different frames
        // Noise2Noise validates against t + 1, inside the block; skip-centre against the frame itself.
        var pairs = Sequences.ValidationFrames(block, TrainingMode.Noise2Noise, 1, "movie");
        var normalized = block.Select(f => Normalization.Apply(f, Normalization.FromPercentiles(f.Pixels))).ToList();
        Assert.Equal(normalized[2].Pixels, pairs[0].Clean!.Pixels);
        Assert.Throws<ArgumentException>(() => Sequences.ValidationFrames(block, TrainingMode.SkipCentre, 3, "movie"));
        Assert.Throws<ArgumentException>(() => Sequences.ValidationFrames(block.Take(2).ToList(), TrainingMode.Noise2Void, 1, "movie"));

        // A zero network outputs the neighbours' mean, so skip-centre's validation loss is the mean of the per-patch MSEs
        // of ½(t − 1 + t + 1) against t, whatever the positions: here constant frames make it exact.
        GrayImage Constant(float v) => new(48, 48, Enumerable.Repeat(v, 48 * 48).ToArray());
        var flat = new ValidationSet([new ImagePair(Constant(0.1f), null, "g", "n") { Context = [Constant(0.25f), Constant(0.75f)] }],
            new TrainerConfig { PatchSize = 32, ValidationPatches = 3, Mode = TrainingMode.SkipCentre });
        var zero = DenoisingNet.Create(Dilated, 1);
        Array.Clear(zero.Parameters);
        Assert.Equal(0.16, flat.Evaluate(zero).Loss, 6);
        Assert.Null(flat.Evaluate(zero).Psnr);   // no ground truth: the reference never enters validation
    }

    [Fact]
    public void TheTrainerRefusesModeAndNetworkMismatches()
    {
        var sequence = SyntheticFluorescence.GenerateSequence(40, 40, 6, seed: 2).Select(f => f.Noisy).ToList();
        var skip = Sequences.SkipCentre(sequence, "s");
        var config = new TrainerConfig { Mode = TrainingMode.SkipCentre, MaxEpochs = 1, StepsPerEpoch = 1, BatchSize = 2, PatchSize = 16, ValidationPatches = 2 };
        // A centre-frame 3-frame network in skip-centre mode would get frame t into its centre plane: refused.
        Assert.Throws<ArgumentException>(() => new Trainer(DenoisingNet.Create(new NetConfig(4, [1, 1]) with { InputFrames = 3 }, 1), config).Train(skip));
        // A skip-centre network in any other mode: refused too.
        var spatiotemporal = Sequences.Spatiotemporal(sequence, 3, "s");
        Assert.Throws<ArgumentException>(() => new Trainer(DenoisingNet.Create(Dilated, 1), config with { Mode = TrainingMode.Noise2Void }).Train(spatiotemporal));
        // The matching pair trains, with the validation frames given or split by blocks.
        var result = new Trainer(DenoisingNet.Create(Dilated, 1), config).Train(skip.Take(3).ToList(), skip.Skip(3).ToList());
        Assert.Single(result.History);
    }

    [Fact]
    public void SkipCentreLearnsBeyondTheNeighboursMean()
    {
        // On a still, noisy synthetic scene the neighbours' mean alone (an untrained network) halves the noise variance;
        // training must do better than that with a shallow network. A sanity check that the mode learns, not a quality bar.
        var options = new SyntheticOptions { PeakPhotons = 10 };
        var train = Enumerable.Range(0, 3).SelectMany(s =>
            Sequences.SkipCentre(SyntheticFluorescence.GenerateSequence(64, 64, 8, 100 + s, 0, options).Select(f => f.Noisy).ToList(), $"s{s}")).ToList();
        var net = DenoisingNet.Create(NetConfig.SkipCentre(new NetConfig(8, [1, 1])), 9);
        var untrained = net.Clone();
        Array.Clear(untrained.Parameters);
        new Trainer(net, new TrainerConfig
        {
            Mode = TrainingMode.SkipCentre, MaxEpochs = 30, StepsPerEpoch = 10, BatchSize = 8, PatchSize = 32, Seed = 3, ValidationPatches = 4,
            EarlyStopPatience = int.MaxValue, KeepLastEpoch = true, Selection = ValidationSignal.Loss, EnableProgressiveTargets = false,
        }).Train(train);
        var test = SyntheticFluorescence.GenerateSequence(64, 64, 6, 999, 0, options);
        double Score(DenoisingNet model)
        {
            var denoiser = new FrameDenoiser(model, tileCore: 32);
            double mse = 0;
            for (int t = 0; t < test.Count; t++)
            {
                var n = Sequences.NeighbourIndices(t, test.Count);
                var output = denoiser.Denoise(test[t].Noisy, [test[n[0]].Noisy, test[n[1]].Noisy]).Output;
                var map = Normalization.FromPercentiles(test[t].Noisy.Pixels);
                mse += Metrics.Mse(Normalization.Apply(output, map).Pixels, Normalization.Apply(test[t].Clean, map).Pixels);
            }
            return Metrics.PsnrFromMse(mse / test.Count);
        }
        double trained = Score(net), mean = Score(untrained);
        Assert.True(trained > mean + 1.0, $"trained {trained:0.00} dB vs the neighbours' mean {mean:0.00} dB");
    }
}

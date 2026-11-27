using System.Runtime.CompilerServices;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 39: Noise2Noise from neighbouring frames and the spatiotemporal blind spot. A multi-frame network only widens its
/// first layer and subtracts its residual from the centre frame; endpoints are reflected, never clamped; the blind spot
/// touches the centre frame only; Noise2Noise never sees its target; tiling and streaming stay exact.
/// </summary>
[Trait(Bricks.Trait, Bricks.Temporal)]
public class TemporalTests
{
    private static NetConfig Dilated3 => new NetConfig(4, [1, 2, 1]) with { InputFrames = 3 };

    private static float[] RandomFrames(int frames, int plane, int seed) => TestUtil.RandomPlane(frames * plane, seed);

    [Fact]
    public void EndpointsAreReflectedNeverClamped()
    {
        Assert.Equal([1, 0, 1], Sequences.FrameIndices(0, 5, 3));
        Assert.Equal([1, 2, 3], Sequences.FrameIndices(2, 5, 3));
        Assert.Equal([3, 4, 3], Sequences.FrameIndices(4, 5, 3));
        Assert.Equal([1, 0, 1], Sequences.FrameIndices(0, 2, 3));   // the shortest sequence: (1, 0, 1) then (0, 1, 0)
        Assert.Equal([0, 1, 0], Sequences.FrameIndices(1, 2, 3));
        Assert.Equal([4], Sequences.FrameIndices(4, 5, 1));
        Assert.Throws<ArgumentException>(() => Sequences.FrameIndices(0, 1, 3));   // T = 1: nothing to reflect
        // Five frames would reflect t − 2 at t = 1 onto frame 1, the target itself: refused, here and in the network.
        Assert.Throws<ArgumentOutOfRangeException>(() => Sequences.FrameIndices(1, 6, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => (NetConfig.Default with { InputFrames = 5 }).Validate());
        // The property that matters: no channel but the centre ever holds the target frame, at any position or length.
        foreach (int frames in new[] { 1, 3 })
            for (int count = Sequences.MinimumFrames(frames); count <= 9; count++)
                for (int t = 0; t < count; t++)
                {
                    var indices = Sequences.FrameIndices(t, count, frames);
                    Assert.Equal(t, indices[frames / 2]);
                    Assert.DoesNotContain(t, indices.Where((_, j) => j != frames / 2));
                }

        // And at the pixel level: the training images' neighbours are other frames, never a copy of their own.
        var sequence = SyntheticFluorescence.GenerateSequence(24, 20, 4, seed: 3).Select(f => f.Noisy).ToList();
        var images = Sequences.Spatiotemporal(sequence, 3, "s");
        Assert.All(images, image => Assert.All(image.Context!, c => Assert.NotEqual(image.Noisy.Pixels, c.Pixels)));
        Assert.Equal(images[1].Noisy.Pixels, images[0].Context![0].Pixels);   // t = 0 sees frame 1 on both sides
        Assert.Equal(images[1].Noisy.Pixels, images[0].Context![1].Pixels);
    }

    [Fact]
    public void SingleFrameNetworksAreUnchangedAndMultiFrameFilesRoundTrip()
    {
        // A single-frame network is the same network as before, byte for byte in its weights file (format 2).
        var plain = DenoisingNet.Create(NetConfig.Default, 42);
        Assert.Equal(plain.Parameters, DenoisingNet.Create(NetConfig.Default with { InputFrames = 1 }, 42).Parameters);
        var stamp = new DateTime(2026, 11, 27, 0, 0, 0, DateTimeKind.Utc);
        var bytes = WeightsFile.Serialize(plain, "p", stamp);
        Assert.Equal(2, BitConverter.ToUInt16(bytes, 8));
        Assert.Equal(26_497, plain.ParameterCount);

        foreach (var config in new[] { Dilated3, NetConfig.UNet(4, 1) with { InputFrames = 3 } })
        {
            var net = DenoisingNet.Create(config, 7);
            int firstLayer = net.Layers[0].ParameterCount;
            Assert.Equal(config.Features * 3 * 9 + config.Features, firstLayer);   // only the first layer widens
            var file = WeightsFile.Serialize(net, "three frames", stamp);
            Assert.Equal(4, BitConverter.ToUInt16(file, 8));
            var (loaded, info) = WeightsFile.Load(file);
            Assert.Equal(config, info.Config);
            Assert.Equal(3, loaded.Config.InputFrames);
            Assert.Equal(net.Parameters, loaded.Parameters);
            Assert.Contains("3 input frames", info.Config.ToString());
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => (NetConfig.Default with { InputFrames = 2 }).Validate());
        Assert.Throws<ArgumentException>(() => (NetConfig.UNet() with { InputFrames = 3, Conditioning = NetConditioning.NoiseLevel }).Validate());
    }

    [Theory]
    [InlineData(false, 8, 9)]
    [InlineData(true, 8, 8)]
    [InlineData(true, 7, 9)]   // unaligned: every frame is padded
    public void MultiFrameGradientsMatchFiniteDifferences(bool unet, int height, int width)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.UNet(3, 1) with { InputFrames = 3 } : Dilated3, seed: 11);
        var rng = new Random(5);
        for (int i = 0; i < net.ParameterCount; i++) net.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.2f;
        int plane = height * width;
        var input = RandomFrames(3, plane, 1);
        var target = TestUtil.RandomPlane(plane, 2);
        var mask = TestUtil.RandomPlane(plane, 3).Select(v => v > 0.5f ? 1f : 0f).ToArray();
        var ws = net.CreateWorkspace();
        float scale = 1f / mask.Sum();
        var analytic = new float[net.ParameterCount];
        net.AccumulateGradients(input, target, mask, height, width, scale, analytic, ws);
        double Loss() => net.AccumulateGradients(input, target, mask, height, width, scale, new float[net.ParameterCount], ws) * scale;

        // Every first-layer weight (all three frames) plus a spread of the rest.
        int firstWeights = net.Layers[0].WeightCount;
        var indices = Enumerable.Range(0, net.ParameterCount).Where(i => i < firstWeights || i % 5 == 0).ToList();
        const float h = 1e-3f;   // 1e-2 crosses a ReLU kink in the unaligned U-Net case; 1e-3 doesn't
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
        // The neighbouring frames really feed the network: their first-layer weights get gradient too.
        int perFrame = net.Layers[0].WeightCount / 3;
        Assert.Contains(analytic.Take(firstWeights), g => Math.Abs(g) > 1e-4);
        Assert.True(Enumerable.Range(0, firstWeights).Any(i => Math.Abs(analytic[i]) > 1e-4 && i % (3 * 9) / 9 != 1), "no gradient reaches a neighbouring frame");
        Assert.True(perFrame > 0);
    }

    [Theory]
    [InlineData(false, 16, 16)]
    [InlineData(true, 16, 16)]
    [InlineData(true, 13, 17)]
    public void TheResidualIsSubtractedFromTheCentreFrame(bool unet, int height, int width)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.UNet(4, 2) with { InputFrames = 3 } : Dilated3, 1);
        Array.Clear(net.Parameters);   // predicts zero noise: the output must be the centre frame, exactly
        int plane = height * width;
        var input = RandomFrames(3, plane, 9);
        var output = new float[plane];
        net.Predict(input, height, width, output, net.CreateWorkspace());
        Assert.Equal(input.AsSpan(plane, plane).ToArray(), output);
        Assert.Throws<ArgumentException>(() => net.Predict(input.AsSpan(0, plane), height, width, output, net.CreateWorkspace()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TiledEqualsWholeFrameWithThreeFrames(bool unet)
    {
        var net = DenoisingNet.Create(unet ? NetConfig.UNet(4, 2) with { InputFrames = 3 } : Dilated3, 5);
        const int w = 70, h = 53;
        var frames = Enumerable.Range(0, 3).Select(f => TestUtil.RandomImage(w, h, 20 + f)).ToList();
        var whole = new float[w * h];
        net.Predict(frames.SelectMany(f => f.Pixels).ToArray(), h, w, whole, net.CreateWorkspace());
        var tiled = TiledInference.Denoise(net, frames, tileCore: 16);
        for (int i = 0; i < whole.Length; i++) Assert.True(Math.Abs(whole[i] - tiled.Pixels[i]) <= 1e-5f, $"pixel {i}: {whole[i]} vs {tiled.Pixels[i]}");
        Assert.Throws<ArgumentException>(() => TiledInference.Denoise(net, frames.Take(1).ToList()));
    }

    [Fact]
    public void TheBlindSpotTouchesTheCentreFrameOnly()
    {
        // Neighbours are constant planes, so any change to them would show; the centre frame is noise.
        var noisy = TestUtil.RandomImage(48, 48, 3);
        GrayImage Constant(float v) => new(48, 48, Enumerable.Repeat(v, 48 * 48).ToArray());
        foreach (var footprint in new MaskFootprint?[] { null, StructuredMasking.DefaultFootprint })
        {
            var images = new List<ImagePair> { new(noisy, null, "g", "n") { Context = [Constant(0.25f), Constant(0.75f)], Footprint = footprint } };
            var sampler = new PatchSampler(images, 32, TrainingMode.Noise2Void, 0.05f, new Random(1));
            var batch = new TrainingBatch(6, 32, 3);
            sampler.Fill(batch);
            int plane = batch.Plane, changedCentres = 0;
            for (int i = 0; i < batch.Count; i++)
            {
                var input = batch.Input(i);
                Assert.All(input[..plane].ToArray(), v => Assert.Equal(0.25f, v));
                Assert.All(input[(2 * plane)..].ToArray(), v => Assert.Equal(0.75f, v));
                var center = input.Slice(plane, plane);
                var target = batch.Target(i);
                var mask = batch.Mask(i);
                for (int p = 0; p < plane; p++)
                    if (mask[p] == 1f && center[p] != target[p]) changedCentres++;
                Assert.Equal(target.ToArray().Length, plane);
            }
            Assert.True(changedCentres > 0, "no blind spot was applied to the centre frame");
        }
        Assert.Throws<ArgumentException>(() => new PatchSampler([new ImagePair(noisy, null, "g", "n")], 32, TrainingMode.Noise2Void, 0.05f, new Random(1))
            .Fill(new TrainingBatch(2, 32, 3)));   // batch and images disagree on frames
    }

    [Fact]
    public void Noise2NoiseNeverPutsItsTargetInTheInput()
    {
        var sequence = SyntheticFluorescence.GenerateSequence(32, 32, 5, seed: 4).Select(f => f.Noisy).ToList();
        var pairs = Sequences.NeighbourPairs(sequence, "s");
        Assert.Equal(8, pairs.Count);
        Assert.All(pairs, p => { Assert.NotNull(p.Clean); Assert.Null(p.Context); Assert.NotEqual(p.Noisy.Pixels, p.Clean!.Pixels); });
        Assert.All(pairs, p => Assert.Equal(1, p.FrameCount));
        Assert.Throws<ArgumentException>(() => Sequences.NeighbourPairs(sequence.Take(1).ToList(), "one"));   // T = 1
        Assert.Throws<ArgumentException>(() => Sequences.Load("x.tif", TrainingMode.Noise2Noise, 3));
        var withContext = new ImagePair(sequence[0], sequence[1], "g", "n") { Context = [sequence[2], sequence[3]] };
        Assert.Throws<ArgumentException>(() => new PatchSampler([withContext], 16, TrainingMode.Noise2Noise, 0.02f, new Random(1)));
    }

    [Fact]
    public void SyntheticSequencesAreStillOrDriftAsBuilt()
    {
        var still = SyntheticFluorescence.GenerateSequence(40, 30, 4, seed: 8);
        Assert.All(still, f => Assert.Equal(still[0].Clean.Pixels, f.Clean.Pixels));   // one scene
        Assert.NotEqual(still[0].Noisy.Pixels, still[1].Noisy.Pixels);                // independent noise
        var moving = SyntheticFluorescence.GenerateSequence(40, 30, 4, seed: 8, driftPerFrame: 3);
        for (int y = 0; y < 30; y++)
            for (int x = 0; x + 3 < 40; x++)   // frame t+1 is frame t moved 3 px right
                Assert.Equal(moving[1].Clean.Pixels[y * 40 + x], moving[2].Clean.Pixels[y * 40 + x + 3]);
    }

    [Fact]
    public void StacksStreamWithAWindowOfThreeFrames()
    {
        var net = DenoisingNet.Create(Dilated3, 3);
        var denoiser = new FrameDenoiser(net, tileCore: 16);
        foreach (int pages in new[] { 2, 5 })   // the shortest stack, and a longer one
        {
            var raw = SyntheticFluorescence.GenerateSequence(40, 32, pages, seed: pages).Select(f => f.Noisy).ToList();
            using var input = new TiffStackReader(new MemoryStream(TiffCodec.EncodeStack(raw, TiffSampleType.UInt16)));
            using var stream = new MemoryStream();
            using (var output = new TiffStackWriter(stream, TiffSampleType.Float32, leaveOpen: true))
                Assert.Equal(pages, BatchDenoising.DenoiseStack(denoiser, input, output));
            using var written = new TiffStackReader(new MemoryStream(stream.ToArray()));
            Assert.Equal(pages, written.PageCount);
            for (int t = 0; t < pages; t++)   // each page equals its frame denoised with its (reflected) neighbours
            {
                var indices = Sequences.FrameIndices(t, pages, 3);
                var expected = denoiser.Denoise(raw[t], [raw[indices[0]], raw[indices[2]]]).Output;
                Assert.Equal(expected.Pixels, written.ReadPage(t).Pixels);
            }
        }

        // Nothing older than the window: once frame t + 1 has been read, frame t − 2 can be collected.
        const int count = 7;
        var frames = SyntheticFluorescence.GenerateSequence(48, 40, count, seed: 1).Select(f => f.Noisy).ToList();
        using var reader = new TiffStackReader(new MemoryStream(TiffCodec.EncodeStack(frames, TiffSampleType.UInt16)));
        frames = null!;
        var tracked = new List<WeakReference>[count];
        var leaked = new List<int>();
        using (var sink = new TiffStackWriter(new MemoryStream(), TiffSampleType.Float32))
            BatchDenoising.DenoiseStack(denoiser, reader, sink, onPage: (page, r, result) =>
            {
                tracked[page] = Track(r, result);
                FullCollection();
                for (int earlier = 0; earlier < page - 2; earlier++)
                    if (tracked[earlier].Any(w => w.IsAlive)) leaked.Add(earlier);
            });
        Assert.Empty(leaked);

        // A single page has no neighbours to reflect: refused, not clamped.
        using var single = new TiffStackReader(new MemoryStream(TiffCodec.EncodeStack([TestUtil.RandomImage(32, 32, 1)], TiffSampleType.Float32)));
        using var nowhere = new TiffStackWriter(new MemoryStream(), TiffSampleType.Float32);
        Assert.Throws<ArgumentException>(() => BatchDenoising.DenoiseStack(denoiser, single, nowhere));
        Assert.Throws<ArgumentException>(() => denoiser.Denoise(TestUtil.RandomImage(32, 32, 1)));   // a 3-frame model needs neighbours
    }

    [Fact]
    public void TimeLapseModesLearnAndMotionCostsNoise2Noise()
    {
        // Noisy synthetic time-lapses and a deliberately shallow network (5 × 5 receptive field), so spatial smoothing
        // can't do the neighbouring frames' job. On a still scene both new modes beat single-frame Noise2Void; when the
        // scene drifts, Noise2Noise's lead turns into a loss (it blurs along the motion): the smear brick 40 has to weigh.
        // The blind spot learns slowly (its loss sees 2% of the pixels), so the budget is 2,000 steps.
        var options = new SyntheticOptions { PeakPhotons = 10 };
        double Score(int drift, TrainingMode mode, int frames)
        {
            var train = Enumerable.Range(0, 4).SelectMany(s =>
            {
                var seq = SyntheticFluorescence.GenerateSequence(64, 64, 8, 100 + s, drift, options).Select(f => f.Noisy).ToList();
                return mode == TrainingMode.Noise2Noise ? Sequences.NeighbourPairs(seq, $"s{s}") : Sequences.Spatiotemporal(seq, frames, $"s{s}");
            }).ToList();
            var net = DenoisingNet.Create(new NetConfig(8, [1, 1]) with { InputFrames = frames }, 9);
            new Trainer(net, new TrainerConfig
            {
                Mode = mode, MaxEpochs = 100, StepsPerEpoch = 20, BatchSize = 8, PatchSize = 32, Seed = 3, ValidationPatches = 4,
                EarlyStopPatience = int.MaxValue, KeepLastEpoch = true, Selection = ValidationSignal.Loss, EnableProgressiveTargets = false,
            }).Train(train);
            var test = SyntheticFluorescence.GenerateSequence(64, 64, 6, 999, drift, options);
            var denoiser = new FrameDenoiser(net, tileCore: 32);
            double mse = 0;
            for (int t = 0; t < test.Count; t++)
            {
                var indices = Sequences.FrameIndices(t, test.Count, frames);
                var output = denoiser.Denoise(test[t].Noisy, frames == 1 ? null : [test[indices[0]].Noisy, test[indices[2]].Noisy]).Output;
                var map = Normalization.FromPercentiles(test[t].Noisy.Pixels);
                mse += Metrics.Mse(Normalization.Apply(output, map).Pixels, Normalization.Apply(test[t].Clean, map).Pixels);
            }
            return Metrics.PsnrFromMse(mse / test.Count);
        }
        double n2v = Score(0, TrainingMode.Noise2Void, 1), spatiotemporal = Score(0, TrainingMode.Noise2Void, 3), n2n = Score(0, TrainingMode.Noise2Noise, 1);
        Assert.True(spatiotemporal > n2v + 0.5, $"still: 3-frame blind spot {spatiotemporal:0.00} dB vs single-frame {n2v:0.00}");
        Assert.True(n2n > n2v + 0.5, $"still: Noise2Noise {n2n:0.00} dB vs single-frame Noise2Void {n2v:0.00}");
        double n2vMoving = Score(4, TrainingMode.Noise2Void, 1), n2nMoving = Score(4, TrainingMode.Noise2Noise, 1);
        Assert.True(n2nMoving - n2vMoving < n2n - n2v - 1.0,
            $"Noise2Noise's lead should shrink with motion: still {n2n - n2v:+0.00;-0.00} dB, drifting {n2nMoving - n2vMoving:+0.00;-0.00} dB");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static List<WeakReference> Track(GrayImage raw, FrameResult result) => [new(raw), new(raw.Pixels), new(result), new(result.Output.Pixels)];

    private static void FullCollection()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}

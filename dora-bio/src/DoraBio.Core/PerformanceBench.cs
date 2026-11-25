using System.Diagnostics;
using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;

namespace DoraBio.Core;

/// <summary>One measured operation (see <see cref="PerformanceBench"/>).</summary>
/// <param name="Name">"denoise", "train-step-n2v" or "train-step-supervised".</param>
/// <param name="Architecture">"dilated" or "unet".</param>
/// <param name="Detail">What exactly was timed, e.g. "512x512, tile 128" or "batch 16 @ 64px".</param>
/// <param name="MedianSeconds">Median wall time of the timed runs (after one untimed warm-up run).</param>
/// <param name="MinSeconds">Fastest timed run.</param>
/// <param name="Runs">Number of timed runs.</param>
/// <param name="Kernels">"blocked" (the kernels DORA runs) or "reference" (the original row kernels, for comparison).</param>
public sealed record BenchTiming(string Name, string Architecture, string Detail, double MedianSeconds, double MinSeconds, int Runs,
    string Kernels = "blocked");

/// <summary>
/// Repeatable timings of the two things that dominate DORA's run time: denoising a frame (tiled, all cores) and one
/// training step (a mini-batch forward and backward pass plus the Adam update). Inputs are synthetic and seeded, so
/// every run measures the same work. Each measurement does one warm-up run (JIT compilation, buffer growth), then
/// reports the median and minimum of the timed runs.
/// </summary>
public static class PerformanceBench
{
    /// <summary>Times <see cref="TiledInference.Denoise"/> on a seeded synthetic frame.</summary>
    /// <param name="net">The model (only read).</param>
    /// <param name="size">Frame side length in pixels.</param>
    /// <param name="runs">Timed runs (at least 1).</param>
    /// <param name="tileCore">Tile core size, as in normal inference.</param>
    /// <param name="referenceKernels">Time the reference row kernels instead (on a copy of the model).</param>
    public static BenchTiming Denoise(DenoisingNet net, int size, int runs, int tileCore = TiledInference.DefaultTileCore,
        bool referenceKernels = false)
    {
        var frame = SyntheticFluorescence.Generate(size, size, seed: 17);
        var (pair, _) = new ImagePair(frame.Noisy, null, "bench", "bench").Normalize();
        var model = net;
        if (referenceKernels)
        {
            model = net.Clone();
            model.ReferenceKernels = true;
        }
        var times = Time(runs, () => TiledInference.Denoise(model, pair.Noisy, tileCore));
        return Summarize("denoise", net, $"{size}x{size}, tile {tileCore}", times, referenceKernels);
    }

    /// <summary>Times one optimizer step (sample a batch, forward and backward for every patch, Adam update).</summary>
    /// <param name="net">The model; a clone is trained, so it is left unchanged.</param>
    /// <param name="mode">Noise2Void (blind-spot loss) or supervised.</param>
    /// <param name="batch">Patches per step.</param>
    /// <param name="patch">Patch side length.</param>
    /// <param name="runs">Timed steps (at least 1).</param>
    /// <param name="referenceKernels">Time the reference row kernels instead.</param>
    public static BenchTiming TrainStep(DenoisingNet net, TrainingMode mode, int batch, int patch, int runs, bool referenceKernels = false)
    {
        var data = SyntheticFluorescence.GenerateDataset(count: 4, size: Math.Max(128, 2 * patch), seed: 23);
        var clone = net.Clone();
        clone.ReferenceKernels = referenceKernels;
        var sampler = new PatchSampler(data, patch, mode, 0.02f, new Random(1));
        var stepper = new BatchStepper(clone, batch, patch);
        var optimizer = new AdamOptimizer(clone.ParameterCount, 1e-3f) { ClipNorm = 1f };
        var times = Time(runs, () => stepper.Step(sampler, optimizer));
        string name = mode == TrainingMode.Noise2Void ? "train-step-n2v" : "train-step-supervised";
        return Summarize(name, net, $"batch {batch} @ {patch}px", times, referenceKernels);
    }

    /// <summary>
    /// The standard set: denoise and both kinds of training step, for the default dilated network and the default
    /// U-Net, plus the dilated network's denoise and Noise2Void step on the reference kernels, so the blocked kernels'
    /// speedup can be checked on whatever machine runs this (gate 24). Full: a 512×512 frame (5 runs) and batch 16 at
    /// 64 px (10 steps). Quick (for gate 22): 128×128 and batch 4 at 32 px, 3 runs each.
    /// </summary>
    public static IReadOnlyList<BenchTiming> Run(bool quick)
    {
        int size = quick ? 128 : 512, batch = quick ? 4 : 16, patch = quick ? 32 : 64;
        int denoiseRuns = quick ? 3 : 5, stepRuns = quick ? 3 : 10;
        var results = new List<BenchTiming>();
        // The standard dilated network (24 features, 1-2-4-8-4-2-1) and the default U-Net, whichever is the default.
        foreach (var config in new[] { new NetConfig(24, [1, 2, 4, 8, 4, 2, 1]), NetConfig.UNet() })
        {
            var net = DenoisingNet.Create(config, seed: 5);
            results.Add(Denoise(net, size, denoiseRuns));
            results.Add(TrainStep(net, TrainingMode.Noise2Void, batch, patch, stepRuns));
            results.Add(TrainStep(net, TrainingMode.Supervised, batch, patch, stepRuns));
            if (config.Architecture == NetArchitecture.Dilated)
            {
                // The speedup gate 24 checks is measured with blocked and reference runs interleaved (A, B, A, B, …),
                // so noise and drift on a shared machine hit both sides alike. These replace the plain timings above.
                var (blocked, reference) = DenoisePair(net, size, denoiseRuns);
                results[results.FindIndex(t => t.Name == "denoise" && t.Architecture == "dilated")] = blocked;
                results.Add(reference);
                (blocked, reference) = TrainStepPair(net, TrainingMode.Noise2Void, batch, patch, stepRuns);
                results[results.FindIndex(t => t.Name == "train-step-n2v" && t.Architecture == "dilated")] = blocked;
                results.Add(reference);
            }
        }
        return results;
    }

    /// <summary>Denoise with the blocked and the reference kernels, runs interleaved; returns (blocked, reference).</summary>
    /// <param name="net">The model (only read; the reference runs use a copy).</param>
    /// <param name="size">Frame side length.</param>
    /// <param name="runs">Timed runs of each.</param>
    public static (BenchTiming Blocked, BenchTiming Reference) DenoisePair(DenoisingNet net, int size, int runs)
    {
        var frame = SyntheticFluorescence.Generate(size, size, seed: 17);
        var (pair, _) = new ImagePair(frame.Noisy, null, "bench", "bench").Normalize();
        var reference = net.Clone();
        reference.ReferenceKernels = true;
        var (a, b) = TimePair(runs,
            () => TiledInference.Denoise(net, pair.Noisy),
            () => TiledInference.Denoise(reference, pair.Noisy));
        string detail = $"{size}x{size}, tile {TiledInference.DefaultTileCore}";
        return (Summarize("denoise", net, detail, a, false), Summarize("denoise", net, detail, b, true));
    }

    /// <summary>Training steps with the blocked and the reference kernels, runs interleaved; returns (blocked, reference).</summary>
    /// <param name="net">The model; both sides train their own copy.</param>
    /// <param name="mode">Noise2Void or supervised.</param>
    /// <param name="batch">Patches per step.</param>
    /// <param name="patch">Patch side length.</param>
    /// <param name="runs">Timed steps of each.</param>
    public static (BenchTiming Blocked, BenchTiming Reference) TrainStepPair(DenoisingNet net, TrainingMode mode, int batch, int patch, int runs)
    {
        Action Stepper(bool referenceKernels)
        {
            var data = SyntheticFluorescence.GenerateDataset(count: 4, size: Math.Max(128, 2 * patch), seed: 23);
            var clone = net.Clone();
            clone.ReferenceKernels = referenceKernels;
            var sampler = new PatchSampler(data, patch, mode, 0.02f, new Random(1));
            var stepper = new BatchStepper(clone, batch, patch);
            var optimizer = new AdamOptimizer(clone.ParameterCount, 1e-3f) { ClipNorm = 1f };
            return () => stepper.Step(sampler, optimizer);
        }
        var (a, b) = TimePair(runs, Stepper(false), Stepper(true));
        string name = mode == TrainingMode.Noise2Void ? "train-step-n2v" : "train-step-supervised";
        string detail = $"batch {batch} @ {patch}px";
        return (Summarize(name, net, detail, a, false), Summarize(name, net, detail, b, true));
    }

    private static (double[] A, double[] B) TimePair(int runs, Action a, Action b)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        a();
        b();   // warm both up first
        var ta = new double[runs];
        var tb = new double[runs];
        for (int i = 0; i < runs; i++)
        {
            var clock = Stopwatch.StartNew();
            a();
            ta[i] = clock.Elapsed.TotalSeconds;
            clock.Restart();
            b();
            tb[i] = clock.Elapsed.TotalSeconds;
        }
        return (ta, tb);
    }

    private static double[] Time(int runs, Action action)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        action();   // warm-up: JIT, workspace growth, caches
        var times = new double[runs];
        for (int i = 0; i < runs; i++)
        {
            var clock = Stopwatch.StartNew();
            action();
            times[i] = clock.Elapsed.TotalSeconds;
        }
        return times;
    }

    private static BenchTiming Summarize(string name, DenoisingNet net, string detail, double[] times, bool referenceKernels)
    {
        var sorted = times.OrderBy(t => t).ToArray();
        double median = sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : 0.5 * (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]);
        string architecture = net.Config.Architecture == NetArchitecture.UNet ? "unet" : "dilated";
        return new BenchTiming(name, architecture, detail, median, sorted[0], sorted.Length, referenceKernels ? "reference" : "blocked");
    }
}

using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;

namespace DoraBio.Core;

/// <summary>What this machine can do at most, measured (see <see cref="Roofline.MeasurePeaks"/>).</summary>
/// <param name="Cores">Threads used for the all-core figures (<see cref="Environment.ProcessorCount"/>).</param>
/// <param name="VectorBits">Width of <see cref="Vector{T}"/> in this process: what DORA's kernels actually use.</param>
/// <param name="Compute">GFLOP/s by instruction mix and width, one thread and all threads.</param>
/// <param name="Bandwidth">Read bandwidth in GB/s by working-set size (one thread for the cache levels, all for DRAM).</param>
public sealed record MachinePeaks(int Cores, int VectorBits, IReadOnlyList<PeakFigure> Compute, IReadOnlyList<PeakFigure> Bandwidth);

/// <summary>One measured peak.</summary>
/// <param name="Name">e.g. "mul+add 256-bit" or "read 1 MB".</param>
/// <param name="OneThread">With one thread (GFLOP/s or GB/s).</param>
/// <param name="AllThreads">With every core busy, per machine.</param>
public sealed record PeakFigure(string Name, double OneThread, double AllThreads);

/// <summary>One convolution kernel measured on one thread against the machine's peaks.</summary>
/// <param name="Kernel">"forward" or "backward" (weight and input gradients).</param>
/// <param name="Shape">Channels, dilation and image size, e.g. "24→24 d2 @ 172²".</param>
/// <param name="Gflop">Floating-point work per call (a multiply and an add count as 2).</param>
/// <param name="Seconds">Median time per call.</param>
/// <param name="Gflops">Achieved GFLOP/s.</param>
/// <param name="Megabytes">Bytes the call must touch at least once (inputs, outputs, weights), in MB.</param>
/// <param name="Intensity">FLOPs per byte touched.</param>
/// <param name="ComputeShare">Achieved GFLOP/s over the one-thread mul+add peak at the kernel's own vector width.</param>
/// <param name="BandwidthShare">Achieved bytes/s over the read bandwidth of the cache level the working set fits in.</param>
public sealed record KernelFigure(string Kernel, string Shape, double Gflop, double Seconds, double Gflops, double Megabytes,
    double Intensity, double ComputeShare, double BandwidthShare)
{
    /// <summary>"compute" if the kernel is closer to the compute roof than to the bandwidth roof, else "memory".</summary>
    public string Bound => ComputeShare >= BandwidthShare ? "compute" : "memory";
}

/// <summary>Where a stack run's time goes, per 512² page (see <see cref="Roofline.MeasureStack"/>).</summary>
/// <param name="Pages">Pages timed.</param>
/// <param name="DecodeSeconds">Reading and decoding a page.</param>
/// <param name="InferSeconds">Tiled inference on the normalized frame.</param>
/// <param name="OtherProcessingSeconds">Normalization and mapping back to the input's units (the rest of <see cref="FrameDenoiser.Denoise"/>).</param>
/// <param name="EncodeWriteSeconds">Encoding and appending the output page.</param>
public sealed record StackBreakdown(int Pages, double DecodeSeconds, double InferSeconds, double OtherProcessingSeconds, double EncodeWriteSeconds)
{
    /// <summary>Total per page.</summary>
    public double TotalSeconds => DecodeSeconds + InferSeconds + OtherProcessingSeconds + EncodeWriteSeconds;
}

/// <summary>
/// Systems brick S1 (brick 36): how close DORA's kernels come to what this CPU can do, so the next optimization targets
/// the real limit. The roofline model: a kernel can't run faster than the machine's arithmetic peak, nor than its
/// bytes moved divided by the bandwidth of wherever its data lives. Peaks are measured here with the kernels' own
/// instruction mix (a separate multiply and add, never fused, because the blocked kernels must stay bit-identical to the
/// reference) at 256 and 512 bits, with fused multiply-add for comparison, and read bandwidth at working sets that fit
/// L1, L2, L3 and DRAM.
/// </summary>
public static class Roofline
{
    private const int Chains = 12;   // independent chains: enough to hide multiply + add latency on two vector ports

    /// <summary>Measures compute and bandwidth peaks (a few seconds).</summary>
    public static MachinePeaks MeasurePeaks()
    {
        var compute = new List<PeakFigure>
        {
            Peak("mul+add 256-bit", () => MulAdd256(), Vector256.IsHardwareAccelerated),
            Peak("mul+add 512-bit", () => MulAdd512(), Vector512.IsHardwareAccelerated),
            Peak("FMA 256-bit", () => Fma256(), Fma.IsSupported),
            Peak("FMA 512-bit", () => Fma512(), Avx512F.IsSupported),
        };
        var bandwidth = new List<PeakFigure>();
        foreach (var (label, bytes) in new[] { ("read 16 KB (L1)", 16 << 10), ("read 1 MB (L2)", 1 << 20), ("read 64 MB (L3)", 64 << 20), ("read 1 GB (DRAM)", 1 << 30) })
            bandwidth.Add(ReadBandwidth(label, bytes, allThreadsShareOneArray: bytes >= (64 << 20)));
        return new MachinePeaks(Environment.ProcessorCount, Vector<float>.Count * 32, compute.Where(c => c.OneThread > 0).ToList(), bandwidth);
    }

    /// <summary>GFLOP/s of one run of <paramref name="kernel"/> (which returns its FLOP count), on one thread and on all.</summary>
    private static PeakFigure Peak(string name, Func<double> kernel, bool supported)
    {
        if (!supported) return new PeakFigure(name, 0, 0);
        kernel();   // JIT and warm up
        double one = Best(3, () => kernel());
        int n = Environment.ProcessorCount;
        double all = Best(3, () =>
        {
            double flops = 0;
            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = n }, _ => { double f = kernel(); lock (Locker) flops += f; });
            return flops;
        });
        return new PeakFigure(name, one, all);
    }

    private static readonly object Locker = new();

    /// <summary>Best rate over <paramref name="runs"/> runs of <paramref name="work"/> (which returns units of work), per second, in giga-units.</summary>
    private static double Best(int runs, Func<double> work)
    {
        double best = 0;
        for (int r = 0; r < runs; r++)
        {
            var clock = Stopwatch.StartNew();
            double units = work();
            best = Math.Max(best, units / clock.Elapsed.TotalSeconds / 1e9);
        }
        return best;
    }

    private const int PeakIterations = 4_000_000;

    private static double MulAdd256()
    {
        var x = Vector256.Create(1.0000001f);
        var w = Vector256.Create(0.9999999f);
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default,
            a6 = default, a7 = default, a8 = default, a9 = default, a10 = default, a11 = default;
        for (int i = 0; i < PeakIterations; i++)
        {
            // acc = acc · w + x: a multiply that depends on the accumulator (so it can't be hoisted out of the loop, as
            // a loop-invariant x · w would be), then a separate add. The JIT never fuses them, like the blocked kernels.
            a0 = a0 * w + x; a1 = a1 * w + x; a2 = a2 * w + x; a3 = a3 * w + x; a4 = a4 * w + x; a5 = a5 * w + x;
            a6 = a6 * w + x; a7 = a7 * w + x; a8 = a8 * w + x; a9 = a9 * w + x; a10 = a10 * w + x; a11 = a11 * w + x;
        }
        Sink(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        return 2.0 * 8 * Chains * PeakIterations;
    }

    private static double MulAdd512()
    {
        var x = Vector512.Create(1.0000001f);
        var w = Vector512.Create(0.9999999f);
        Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default,
            a6 = default, a7 = default, a8 = default, a9 = default, a10 = default, a11 = default;
        for (int i = 0; i < PeakIterations; i++)
        {
            // acc = acc · w + x: a multiply that depends on the accumulator (so it can't be hoisted out of the loop, as
            // a loop-invariant x · w would be), then a separate add. The JIT never fuses them, like the blocked kernels.
            a0 = a0 * w + x; a1 = a1 * w + x; a2 = a2 * w + x; a3 = a3 * w + x; a4 = a4 * w + x; a5 = a5 * w + x;
            a6 = a6 * w + x; a7 = a7 * w + x; a8 = a8 * w + x; a9 = a9 * w + x; a10 = a10 * w + x; a11 = a11 * w + x;
        }
        Sink(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        return 2.0 * 16 * Chains * PeakIterations;
    }

    private static double Fma256()
    {
        var x = Vector256.Create(1.0000001f);
        var w = Vector256.Create(0.9999999f);
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default,
            a6 = default, a7 = default, a8 = default, a9 = default, a10 = default, a11 = default;
        for (int i = 0; i < PeakIterations; i++)
        {
            a0 = Fma.MultiplyAdd(x, w, a0); a1 = Fma.MultiplyAdd(x, w, a1); a2 = Fma.MultiplyAdd(x, w, a2);
            a3 = Fma.MultiplyAdd(x, w, a3); a4 = Fma.MultiplyAdd(x, w, a4); a5 = Fma.MultiplyAdd(x, w, a5);
            a6 = Fma.MultiplyAdd(x, w, a6); a7 = Fma.MultiplyAdd(x, w, a7); a8 = Fma.MultiplyAdd(x, w, a8);
            a9 = Fma.MultiplyAdd(x, w, a9); a10 = Fma.MultiplyAdd(x, w, a10); a11 = Fma.MultiplyAdd(x, w, a11);
        }
        Sink(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        return 2.0 * 8 * Chains * PeakIterations;
    }

    private static double Fma512()
    {
        var x = Vector512.Create(1.0000001f);
        var w = Vector512.Create(0.9999999f);
        Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default, a4 = default, a5 = default,
            a6 = default, a7 = default, a8 = default, a9 = default, a10 = default, a11 = default;
        for (int i = 0; i < PeakIterations; i++)
        {
            a0 = Avx512F.FusedMultiplyAdd(x, w, a0); a1 = Avx512F.FusedMultiplyAdd(x, w, a1); a2 = Avx512F.FusedMultiplyAdd(x, w, a2);
            a3 = Avx512F.FusedMultiplyAdd(x, w, a3); a4 = Avx512F.FusedMultiplyAdd(x, w, a4); a5 = Avx512F.FusedMultiplyAdd(x, w, a5);
            a6 = Avx512F.FusedMultiplyAdd(x, w, a6); a7 = Avx512F.FusedMultiplyAdd(x, w, a7); a8 = Avx512F.FusedMultiplyAdd(x, w, a8);
            a9 = Avx512F.FusedMultiplyAdd(x, w, a9); a10 = Avx512F.FusedMultiplyAdd(x, w, a10); a11 = Avx512F.FusedMultiplyAdd(x, w, a11);
        }
        Sink(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        return 2.0 * 16 * Chains * PeakIterations;
    }

    private static volatile float _sink;

    private static void Sink<T>(T v) where T : struct => _sink = v.GetHashCode();

    /// <summary>
    /// Read bandwidth with a working set of <paramref name="bytes"/>: one thread sweeping its own array, and all threads
    /// sweeping their own arrays (cache levels) or one shared array split between them (DRAM-sized sets).
    /// </summary>
    private static PeakFigure ReadBandwidth(string name, int bytes, bool allThreadsShareOneArray)
    {
        int n = Environment.ProcessorCount, floats = bytes / 4;
        long target = 2L << 30;                              // sweep about 2 GB per measurement
        int sweeps = (int)Math.Max(1, target / bytes);
        var own = Enumerable.Range(0, allThreadsShareOneArray ? 1 : n).Select(_ => Fill(floats)).ToArray();
        double one = Best(3, () => { for (int s = 0; s < sweeps; s++) SumVector(own[0]); return (double)sweeps * bytes; });
        double all = Best(3, () =>
        {
            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = n }, t =>
            {
                if (allThreadsShareOneArray)
                {
                    int chunk = floats / n;
                    for (int s = 0; s < sweeps; s++) SumVector(own[0].AsSpan(t * chunk, chunk));
                }
                else
                    for (int s = 0; s < sweeps; s++) SumVector(own[t]);
            });
            return allThreadsShareOneArray ? (double)sweeps * bytes : (double)sweeps * bytes * n;
        });
        return new PeakFigure(name, one, all);
    }

    private static float[] Fill(int floats)
    {
        var a = new float[floats];
        for (int i = 0; i < a.Length; i++) a[i] = i & 7;
        return a;
    }

    private static void SumVector(ReadOnlySpan<float> data)
    {
        var v = System.Runtime.InteropServices.MemoryMarshal.Cast<float, Vector256<float>>(data);
        Vector256<float> s0 = default, s1 = default, s2 = default, s3 = default, s4 = default, s5 = default, s6 = default, s7 = default;
        int i = 0;
        for (; i + 8 <= v.Length; i += 8)
        {
            s0 += v[i]; s1 += v[i + 1]; s2 += v[i + 2]; s3 += v[i + 3];
            s4 += v[i + 4]; s5 += v[i + 5]; s6 += v[i + 6]; s7 += v[i + 7];
        }
        for (; i < v.Length; i++) s0 += v[i];
        Sink(s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7);
    }

    /// <summary>
    /// Times the blocked convolution kernels on one thread for the shapes DORA actually runs: the dilated network's
    /// layers on an inference tile (128 px core plus its 22 px margin on each side), and the U-Net's levels and the
    /// dilated network's middle layer on a 64 px training patch (forward and backward).
    /// </summary>
    /// <param name="peaks">From <see cref="MeasurePeaks"/>.</param>
    /// <param name="runs">Timed calls per shape (median reported).</param>
    public static List<KernelFigure> MeasureKernels(MachinePeaks peaks, int runs = 7)
    {
        int tile = TiledInference.DefaultTileCore + 2 * 22;
        var shapes = new (string Kind, int In, int Out, int Dilation, int Size)[]
        {
            ("forward", 1, 24, 1, tile), ("forward", 24, 24, 2, tile), ("forward", 24, 24, 8, tile), ("forward", 24, 1, 1, tile),
            ("forward", 16, 16, 1, 64), ("forward", 32, 32, 1, 32), ("forward", 64, 64, 1, 16),
            ("backward", 24, 24, 2, 64), ("backward", 16, 16, 1, 64), ("backward", 32, 32, 1, 32), ("backward", 64, 64, 1, 16),
        };
        double PeakFor(int bits) => peaks.Compute.FirstOrDefault(c => c.Name == $"mul+add {bits}-bit")?.OneThread ?? double.NaN;
        var results = new List<KernelFigure>();
        bool wide = ConvLayer.WideKernels;
        try
        {
            foreach (var (kind, cin, cout, d, size) in shapes)
                foreach (bool useWide in wide && kind == "forward" ? new[] { false, true } : [wide])
                {
                    ConvLayer.WideKernels = useWide;
                    // Backward runs its input gradients 512 bits wide where it can, but its weight gradients stay on
                    // Vector<float>; it is compared with the narrower peak.
                    string label = kind != "forward" ? kind : useWide ? "fwd-512" : "fwd-256";
                    double peak = PeakFor(kind == "forward" && useWide ? 512 : peaks.VectorBits);
                    results.Add(MeasureKernel(label, cin, cout, d, size, runs, peak, peaks.Bandwidth));
                }
        }
        finally
        {
            ConvLayer.WideKernels = wide;
        }
        return results;
    }

    private static KernelFigure MeasureKernel(string kind, int cin, int cout, int d, int size, int runs, double computePeak,
        IReadOnlyList<PeakFigure> bandwidth)
    {
        var layer = new ConvLayer(cin, cout, d, weightOffset: 0);
        var rng = new Random(7);
        var parameters = Enumerable.Range(0, layer.ParameterCount).Select(_ => (float)(rng.NextDouble() - 0.5) * 0.1f).ToArray();
        var input = Enumerable.Range(0, cin * size * size).Select(_ => (float)rng.NextDouble()).ToArray();
        var padded = new float[layer.PaddedLength(size, size)];
        layer.Pad(input, size, size, padded);
        var output = new float[cout * size * size];
        var gradOut = Enumerable.Range(0, cout * size * size).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var gradients = new float[layer.ParameterCount];
        var gradPadded = new float[padded.Length];
        Action call = kind.StartsWith("fwd", StringComparison.Ordinal)
            ? () => layer.Forward(parameters, padded, size, size, output)
            : () => layer.Backward(parameters, padded, gradOut, size, size, gradients, gradPadded);
        call();
        var times = new double[runs];
        for (int r = 0; r < runs; r++)
        {
            var clock = Stopwatch.StartNew();
            call();
            times[r] = clock.Elapsed.TotalSeconds;
        }
        Array.Sort(times);
        double seconds = times[runs / 2];
        double flops = ConvFlops(cin, cout, size, size, backward: !kind.StartsWith("fwd", StringComparison.Ordinal));
        double bytes = 4.0 * (padded.Length + output.Length + layer.ParameterCount);
        if (kind == "backward") bytes += 4.0 * (padded.Length + layer.ParameterCount);   // + gradient of the input, of the weights
        double gflops = flops / seconds / 1e9, gbs = bytes / seconds / 1e9;
        // The bandwidth roof is the level the working set fits in (per-core L2 is 2 MB on the machine this was built on;
        // the figures carry their own labels, so pick by size rather than by name).
        var level = bytes <= 32 << 10 ? bandwidth[0] : bytes <= 1.5 * (1 << 20) ? bandwidth[1] : bytes <= 128 << 20 ? bandwidth[2] : bandwidth[3];
        string shape = $"{cin}→{cout} d{d} @ {size}²";
        return new KernelFigure(kind, shape, flops / 1e9, seconds, gflops, bytes / 1e6, flops / bytes, gflops / computePeak, gbs / level.OneThread);
    }

    /// <summary>
    /// FLOPs of one 3×3 convolution call: in × out × 9 multiply-adds per pixel, 2 FLOPs each; a backward call computes both
    /// the weight and the input gradients, so twice that (the bias gradient's sums are negligible and not counted).
    /// </summary>
    internal static double ConvFlops(int inChannels, int outChannels, int height, int width, bool backward) =>
        (backward ? 4.0 : 2.0) * inChannels * outChannels * 9 * height * width;

    /// <summary>
    /// Denoises a synthetic uint16 stack page by page the way <c>dora-bio denoise</c> does, timing each stage: decode,
    /// tiled inference, the rest of the per-frame processing (normalization and mapping back), and encode + write.
    /// </summary>
    /// <param name="net">The model.</param>
    /// <param name="pages">Pages in the synthetic stack.</param>
    /// <param name="size">Page side length.</param>
    public static StackBreakdown MeasureStack(DenoisingNet net, int pages = 8, int size = 512)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"dora-roofline-{Environment.ProcessId}");
        Directory.CreateDirectory(dir);
        string inPath = Path.Combine(dir, "in.tif"), outPath = Path.Combine(dir, "out.tif");
        try
        {
            using (var writer = TiffStackWriter.Create(inPath))
                for (int p = 0; p < pages; p++)
                {
                    var frame = SyntheticFluorescence.Generate(size, size, seed: 100 + p).Noisy;
                    writer.Append(frame);
                }
            var denoiser = new FrameDenoiser(net);
            double decode = 0, process = 0, infer = 0, write = 0;
            using var reader = TiffStackReader.Open(inPath);
            using var output = TiffStackWriter.Create(outPath, TiffSampleType.Float32);
            denoiser.Denoise(reader.ReadPage(0));   // warm up (JIT, workspaces)
            for (int p = 0; p < pages; p++)
            {
                var clock = Stopwatch.StartNew();
                var raw = reader.ReadPage(p);
                decode += clock.Elapsed.TotalSeconds;
                clock.Restart();
                var result = denoiser.Denoise(raw);
                process += clock.Elapsed.TotalSeconds;
                var (normalized, _) = new ImagePair(raw, null, "p", "p").Normalize();
                clock.Restart();
                TiledInference.Denoise(net, normalized.Noisy);
                infer += clock.Elapsed.TotalSeconds;
                clock.Restart();
                output.Append(result.Output);
                write += clock.Elapsed.TotalSeconds;
            }
            return new StackBreakdown(pages, decode / pages, infer / pages, Math.Max(0, process - infer) / pages, write / pages);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}

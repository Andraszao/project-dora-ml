using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;

namespace DoraBio.Core;

/// <summary>One denoised frame, in the input's original intensity units.</summary>
/// <param name="Output">The denoised frame.</param>
/// <param name="Baseline">The bilateral baseline, if it was asked for.</param>
/// <param name="UsedCorrelatedModel">True if the noise check sent this frame to the correlated-noise model.</param>
/// <param name="NoiseModel">The noise model calibrated from the frame, when variance stabilization was used.</param>
/// <param name="SigmaBefore">Estimated noise σ of the normalized input.</param>
/// <param name="SigmaAfter">Estimated noise σ of the normalized output.</param>
/// <param name="BaselineSigma">Estimated noise σ of the normalized baseline, if there is one.</param>
public sealed record FrameResult(GrayImage Output, GrayImage? Baseline, bool UsedCorrelatedModel, NoiseModel? NoiseModel,
    double SigmaBefore, double SigmaAfter, double? BaselineSigma);

/// <summary>
/// Denoises one frame in original units, the way <c>dora-bio denoise</c> always has:
/// percentile normalization; optionally the noise check that sends a frame to a second model trained for correlated
/// noise (P3); optionally variance stabilization (P3); tiled inference; then the inverse normalization. A single image,
/// every page of a stack and every file of a folder all go through <see cref="Denoise(GrayImage, bool, CancellationToken)"/>, so a page of a stack comes
/// out exactly as the same frame denoised on its own. Each frame is normalized by its own percentiles, as a single
/// image would be. The networks are only read, so one instance can serve several threads.
/// </summary>
public sealed class FrameDenoiser
{
    /// <summary>Configures the per-frame pipeline.</summary>
    /// <param name="net">The model for frames whose noise looks independent (or every frame, without a second model).</param>
    /// <param name="correlatedNet">Optional model for frames whose noise check says correlated.</param>
    /// <param name="stabilize">Denoise in the variance-stabilized domain (for models trained with <c>--vst</c>).</param>
    /// <param name="tileCore">Tile core size for <see cref="TiledInference"/>.</param>
    public FrameDenoiser(DenoisingNet net, DenoisingNet? correlatedNet = null, bool stabilize = false, int tileCore = TiledInference.DefaultTileCore)
    {
        (Net, CorrelatedNet, Stabilize, TileCore) = (net, correlatedNet, stabilize, tileCore);
        if (correlatedNet is not null && correlatedNet.Config.InputFrames != net.Config.InputFrames)
            throw new ArgumentException("Both models of a routed pair must take the same number of frames.");
        if (stabilize && net.Config.InputFrames > 1)
            throw new ArgumentException("Variance stabilization works with single-frame models only.");
        _pool = new WorkspacePool(net);
        _correlatedPool = correlatedNet is null ? null : new WorkspacePool(correlatedNet);
    }

    // Inference workspaces (tens of MB per thread) are kept for the denoiser's lifetime and shared by every frame, so a
    // long stack allocates them once instead of once per frame. Without this, each 512² frame left about 390 MB of
    // garbage, and the process's peak memory climbed with the stack's length before the collector caught up.
    private readonly WorkspacePool _pool;
    private readonly WorkspacePool? _correlatedPool;

    /// <summary>The main model.</summary>
    public DenoisingNet Net { get; }

    /// <summary>The model for correlated noise, if any.</summary>
    public DenoisingNet? CorrelatedNet { get; }

    /// <summary>True to denoise through the generalized Anscombe transform.</summary>
    public bool Stabilize { get; }

    /// <summary>Tile core size for inference.</summary>
    public int TileCore { get; }

    /// <summary>
    /// The noise check (<see cref="NoiseDiagnostics"/>) on a raw frame: true if its noise looks correlated between
    /// neighbouring pixels. Frames too small for the check count as independent.
    /// </summary>
    public static bool LooksCorrelated(GrayImage noisy)
    {
        try { return NoiseDiagnostics.Diagnose(noisy).Correlated; }
        catch (ArgumentException) { return false; }
    }

    /// <summary>Denoises one frame.</summary>
    /// <param name="raw">The frame in its original units.</param>
    /// <param name="withBaseline">Also compute the bilateral baseline.</param>
    /// <param name="cancellation">Checked between tiles.</param>
    public FrameResult Denoise(GrayImage raw, bool withBaseline = false, CancellationToken cancellation = default) =>
        Denoise(raw, null, withBaseline, cancellation);

    /// <summary>
    /// Denoises one frame of a time-lapse with a multi-frame model (brick 39): <paramref name="context"/> holds its
    /// neighbours in time order without the frame itself (for 3 frames: t−1, t+1; reflected at a sequence's ends,
    /// <see cref="Sequences.FrameIndices"/>). Each frame is normalized by its own percentiles, as in training, and the
    /// output is mapped back with the frame's own map. Single-frame models take no context.
    /// </summary>
    /// <param name="raw">The frame, in original units.</param>
    /// <param name="context">Its neighbouring frames (raw units), or null for a single-frame model.</param>
    /// <param name="withBaseline">Also run the bilateral filter.</param>
    /// <param name="cancellation">Stops between tiles.</param>
    /// <exception cref="ArgumentException">The model and the number of frames given don't match.</exception>
    public FrameResult Denoise(GrayImage raw, IReadOnlyList<GrayImage>? context, bool withBaseline = false, CancellationToken cancellation = default)
    {
        int needed = Net.Config.InputFrames - 1;
        if ((context?.Count ?? 0) != needed)
            throw new ArgumentException(needed == 0
                ? "This model denoises single frames; it takes no neighbouring frames."
                : $"This model takes {Net.Config.InputFrames} frames: denoise a time-lapse stack, so each frame comes with its neighbours.");
        var map = Normalization.FromPercentiles(raw.Pixels);
        var input = Normalization.Apply(raw, map);
        bool correlated = CorrelatedNet is not null && LooksCorrelated(raw);
        var (net, pool) = correlated ? (CorrelatedNet!, _correlatedPool!) : (Net, _pool);

        GrayImage output, denoised;
        NoiseModel? model = null;
        if (Stabilize)
        {
            (output, var m) = VarianceStabilization.Denoise(net, raw, TileCore, cancellation, pool);
            model = m;
            denoised = Normalization.Apply(output, map);
        }
        else if (context is { Count: > 0 })
        {
            // Neighbours around the frame, each normalized by its own percentiles, as in training.
            var frames = new List<GrayImage>(context.Count + 1);
            frames.AddRange(context.Take(context.Count / 2).Select(NormalizeOwn));
            frames.Add(input);
            frames.AddRange(context.Skip(context.Count / 2).Select(NormalizeOwn));
            denoised = TiledInference.Denoise(net, frames, TileCore, cancellation, pool);
            output = Normalization.Invert(denoised, map);
        }
        else
        {
            denoised = TiledInference.Denoise(net, input, TileCore, cancellation, pool);
            output = Normalization.Invert(denoised, map);
        }

        GrayImage? baseline = null;
        double? baselineSigma = null;
        if (withBaseline)
        {
            var filtered = BilateralBaseline.FilterAuto(input);
            baselineSigma = Metrics.EstimateNoiseSigma(filtered);
            baseline = Normalization.Invert(filtered, map);
        }
        return new FrameResult(output, baseline, correlated, model,
            Metrics.EstimateNoiseSigma(input), Metrics.EstimateNoiseSigma(denoised), baselineSigma);
    }

    private static GrayImage NormalizeOwn(GrayImage frame) => Normalization.Apply(frame, Normalization.FromPercentiles(frame.Pixels));
}

/// <summary>
/// Denoising many frames: every page of a TIFF stack, or every image in a folder, streamed one frame at a time.
/// Only the current frame (input, output and baseline) is held in memory, whatever the number of pages: each page is
/// read from disk just before it is denoised and written to disk right after, and nothing keeps a reference to it
/// afterwards. Outputs are float32 TIFF in the input's units; a stack's output is a stack with the same page order.
/// Because the writer patches the page chain after every page, a run that stops early (an error, or cancellation)
/// leaves valid TIFFs holding the pages finished so far.
/// </summary>
public static class BatchDenoising
{
    /// <summary>Called after each page: its index, the raw frame read from disk, and the result written out.</summary>
    public delegate void PageDone(int page, GrayImage raw, FrameResult result);

    /// <summary>Denoises every page of <paramref name="input"/> into <paramref name="output"/>, in order.</summary>
    /// <param name="denoiser">The per-frame pipeline.</param>
    /// <param name="input">The stack (a single-page TIFF is a one-page stack).</param>
    /// <param name="output">Receives one denoised page per input page.</param>
    /// <param name="baseline">Optional: receives the bilateral baseline of each page.</param>
    /// <param name="onPage">Optional progress callback.</param>
    /// <param name="cancellation">Checked before each page and between tiles.</param>
    /// <returns>The number of pages denoised.</returns>
    public static int DenoiseStack(FrameDenoiser denoiser, TiffStackReader input, TiffStackWriter output, TiffStackWriter? baseline = null,
        PageDone? onPage = null, CancellationToken cancellation = default)
    {
        if (denoiser.Net.Config.InputFrames > 1) return DenoiseWindowed(denoiser, input, output, baseline, onPage, cancellation);
        long allocatedAtLastCollection = GC.GetTotalAllocatedBytes();
        for (int p = 0; p < input.PageCount; p++)
        {
            cancellation.ThrowIfCancellationRequested();
            var raw = input.ReadPage(p);
            var result = denoiser.Denoise(raw, baseline is not null, cancellation);
            output.Append(result.Output);
            baseline?.Append(result.Baseline!);
            onPage?.Invoke(p, raw, result);

            CollectIfDue(ref allocatedAtLastCollection);
        }
        return input.PageCount;
    }

    /// <summary>Page garbage allowed to build up between the full collections <see cref="DenoiseStack"/> requests.</summary>
    private const long CollectEveryBytes = 64L << 20;

    /// <summary>
    /// Each page leaves a few MB of finished frame buffers, all large-object-heap arrays, which the runtime only reclaims
    /// in full collections. On a machine with free memory it lets them pile up for hundreds of MB first (a 160-page 512²
    /// stack peaked at 850 MB). Collecting once per <see cref="CollectEveryBytes"/> of page garbage keeps the peak near the
    /// live working set, and costs nothing measurable at that spacing.
    /// </summary>
    private static void CollectIfDue(ref long allocatedAtLastCollection)
    {
        if (GC.GetTotalAllocatedBytes() - allocatedAtLastCollection <= CollectEveryBytes) return;
        GC.Collect();
        allocatedAtLastCollection = GC.GetTotalAllocatedBytes();
    }

    /// <summary>
    /// A multi-frame model on a stack (brick 39): page t is denoised with its neighbours (<see cref="Sequences.FrameIndices"/>,
    /// reflected at both ends), so frame t is written once t + 1 has been read. Only the model's window of pages is kept:
    /// nothing older than t − k (k = frames / 2) outlives the step that reads t + k.
    /// </summary>
    private static int DenoiseWindowed(FrameDenoiser denoiser, TiffStackReader input, TiffStackWriter output, TiffStackWriter? baseline,
        PageDone? onPage, CancellationToken cancellation)
    {
        int frames = denoiser.Net.Config.InputFrames, k = frames / 2, count = input.PageCount;
        if (count < Sequences.MinimumFrames(frames))
            throw new ArgumentException($"This model takes {frames} frames; the stack has {count} page(s), too few to give each frame neighbours.");
        var window = new Dictionary<int, GrayImage>();
        GrayImage Page(int i)
        {
            if (!window.TryGetValue(i, out var page)) window[i] = page = input.ReadPage(i);
            return page;
        }
        long allocatedAtLastCollection = GC.GetTotalAllocatedBytes();
        for (int t = 0; t < count; t++)
        {
            cancellation.ThrowIfCancellationRequested();
            foreach (int old in window.Keys.Where(i => i < t - k).ToList()) window.Remove(old);   // older than the window
            var indices = Sequences.FrameIndices(t, count, frames);
            var raw = Page(t);
            var context = indices.Where((_, j) => j != k).Select(Page).ToArray();
            var result = denoiser.Denoise(raw, context, baseline is not null, cancellation);
            output.Append(result.Output);
            baseline?.Append(result.Baseline!);
            onPage?.Invoke(t, raw, result);
            CollectIfDue(ref allocatedAtLastCollection);
        }
        return count;
    }

    /// <summary>
    /// Denoises one image file (TIFF, single page or stack, or PNG) into a float32 TIFF with the same number of pages.
    /// </summary>
    /// <param name="denoiser">The per-frame pipeline.</param>
    /// <param name="inputPath">The image to denoise.</param>
    /// <param name="outputPath">Where to write the denoised TIFF (overwritten). Must not be the input.</param>
    /// <param name="baselinePath">Optional: where to write the bilateral baseline.</param>
    /// <param name="onPage">Optional progress callback.</param>
    /// <param name="cancellation">Checked before each page and between tiles.</param>
    /// <returns>The number of pages denoised.</returns>
    /// <exception cref="ArgumentException">An output path is the input path.</exception>
    public static int DenoiseFile(FrameDenoiser denoiser, string inputPath, string outputPath, string? baselinePath = null,
        PageDone? onPage = null, CancellationToken cancellation = default)
    {
        string inputFull = Path.GetFullPath(inputPath);
        foreach (var target in new[] { outputPath, baselinePath })
            if (target is not null && Path.GetFullPath(target) == inputFull)
                throw new ArgumentException($"{target}: the output would overwrite the input.");

        using var reader = OpenAsStack(inputPath);
        using var output = TiffStackWriter.Create(outputPath, TiffSampleType.Float32);
        using var baseline = baselinePath is null ? null : TiffStackWriter.Create(baselinePath, TiffSampleType.Float32);
        return DenoiseStack(denoiser, reader, output, baseline, onPage, cancellation);
    }

    /// <summary>Number of frames in an image file: its page count for a TIFF, 1 for a PNG. Reads only the page directories.</summary>
    public static int CountPages(string path)
    {
        using var reader = OpenAsStack(path);
        return reader.PageCount;
    }

    /// <summary>
    /// Opens any supported image as a stack: TIFFs stream page by page; a PNG (always one frame) is decoded and
    /// re-wrapped as a one-page in-memory TIFF, which stores its values exactly as float32.
    /// </summary>
    private static TiffStackReader OpenAsStack(string path)
    {
        Span<byte> head = stackalloc byte[4];
        using (var probe = File.OpenRead(path)) head = head[..probe.ReadAtLeast(head, head.Length, throwOnEndOfStream: false)];
        if (TiffCodec.HasTiffSignature(head)) return TiffStackReader.Open(path);
        var frame = ImageLoader.Load(path);   // PNG, or a clear NotSupportedException
        return new TiffStackReader(new MemoryStream(TiffCodec.Encode(frame, TiffSampleType.Float32)));
    }

    /// <summary>
    /// Plans a folder run: every supported image directly inside <paramref name="inputDir"/> (not in subfolders),
    /// in ordinal name order, mapped to <c>outputDir/&lt;name&gt;.tif</c>.
    /// </summary>
    /// <exception cref="DirectoryNotFoundException">The input folder doesn't exist.</exception>
    /// <exception cref="ArgumentException">The folders are the same, the folder has no supported images, or two inputs
    /// would write the same output (e.g. <c>a.png</c> and <c>a.tif</c>).</exception>
    public static IReadOnlyList<(string Input, string Output)> PlanFolder(string inputDir, string outputDir)
    {
        if (!Directory.Exists(inputDir)) throw new DirectoryNotFoundException($"{inputDir}: no such folder.");
        if (Path.GetFullPath(inputDir).TrimEnd(Path.DirectorySeparatorChar) == Path.GetFullPath(outputDir).TrimEnd(Path.DirectorySeparatorChar))
            throw new ArgumentException("The output folder must differ from the input folder, so no input is overwritten.");
        var plan = Directory.EnumerateFiles(inputDir)
            .Where(ImageLoader.IsSupported)
            .Order(StringComparer.Ordinal)
            .Select(f => (Input: f, Output: Path.Combine(outputDir, Path.GetFileNameWithoutExtension(f) + ".tif")))
            .ToList();
        if (plan.Count == 0) throw new ArgumentException($"{inputDir}: no .tif, .tiff or .png images.");
        var clashes = plan.GroupBy(p => p.Output, StringComparer.OrdinalIgnoreCase).Where(g => g.Count() > 1).ToList();
        if (clashes.Count > 0)
            throw new ArgumentException("These inputs would write the same output: " +
                string.Join("; ", clashes.Select(g => string.Join(", ", g.Select(p => Path.GetFileName(p.Input))))));
        return plan;
    }
}

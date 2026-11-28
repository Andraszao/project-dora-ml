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
        if (correlatedNet is not null && (correlatedNet.Config.InputFrames != net.Config.InputFrames
            || correlatedNet.Config.ResidualBase != net.Config.ResidualBase))
            throw new ArgumentException("Both models of a routed pair must take the same frames.");
        if (stabilize && net.Config.WindowFrames > 1)
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
    /// output is mapped back with the frame's own map. Single-frame models take no context. A skip-centre model
    /// (brick 40) takes the same two neighbours and only them: the frame sets the intensity map and the noise check,
    /// as its target did in training, but never enters the network's input.
    /// </summary>
    /// <param name="raw">The frame, in original units.</param>
    /// <param name="context">Its neighbouring frames (raw units), or null for a single-frame model.</param>
    /// <param name="withBaseline">Also run the bilateral filter.</param>
    /// <param name="cancellation">Stops between tiles.</param>
    /// <exception cref="ArgumentException">The model and the number of frames given don't match.</exception>
    /// <param name="correlated">The routing decision when a caller makes it once for a whole stack (brick 41); null to run the
    /// noise check on this frame.</param>
    public FrameResult Denoise(GrayImage raw, IReadOnlyList<GrayImage>? context, bool withBaseline = false, CancellationToken cancellation = default,
        bool? correlated = null)
    {
        int needed = Net.Config.WindowFrames - 1;
        if ((context?.Count ?? 0) != needed)
            throw new ArgumentException(needed == 0
                ? "This model denoises single frames; it takes no neighbouring frames."
                : Net.Config.SkipsCentre
                    ? "This skip-centre model denoises a frame from its two neighbours: denoise a time-lapse stack, so each frame comes with them."
                    : $"This model takes {Net.Config.InputFrames} frames: denoise a time-lapse stack, so each frame comes with its neighbours.");
        var map = Normalization.FromPercentiles(raw.Pixels);
        var input = Normalization.Apply(raw, map);
        bool routeCorrelated = CorrelatedNet is not null && (correlated ?? LooksCorrelated(raw));
        var (net, pool) = routeCorrelated ? (CorrelatedNet!, _correlatedPool!) : (Net, _pool);

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
            if (!net.Config.SkipsCentre) frames.Add(input);   // a skip-centre network never sees the frame itself
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
        return new FrameResult(output, baseline, routeCorrelated, model,
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

    /// <summary>
    /// Denoises every page of an open TIFF reader into <paramref name="output"/>, in order: the same path as the
    /// <see cref="ImageStack"/> overload, with the stack's axes read from the TIFF's own metadata.
    /// </summary>
    /// <param name="denoiser">The per-frame pipeline.</param>
    /// <param name="input">The stack (a single-page TIFF is a one-page stack). The caller still owns and disposes it.</param>
    /// <param name="output">Receives one denoised page per input page.</param>
    /// <param name="baseline">Optional: receives the bilateral baseline of each page.</param>
    /// <param name="onPage">Optional progress callback.</param>
    /// <param name="cancellation">Checked before each page and between tiles.</param>
    /// <returns>The number of pages denoised.</returns>
    public static int DenoiseStack(FrameDenoiser denoiser, TiffStackReader input, TiffStackWriter output, TiffStackWriter? baseline = null,
        PageDone? onPage = null, CancellationToken cancellation = default) =>
        // Not disposed here: disposing the wrapper would dispose the caller's reader.
        DenoiseStack(denoiser, ImageStack.FromTiff(input), output, baseline, onPage, null, cancellation);

    /// <summary>
    /// Denoises every plane of <paramref name="input"/> into <paramref name="output"/>, in storage order, so the output
    /// has the input's page order. A multi-frame model gets each plane's neighbours along the stack's time axis with every
    /// other axis held fixed (<see cref="TimeAxis"/>); <paramref name="onNotice"/> receives the notice when unlabelled
    /// planes are taken as frames.
    /// </summary>
    /// <param name="denoiser">The per-frame pipeline.</param>
    /// <param name="input">The stack, with its axes (from the file, or declared when opened).</param>
    /// <param name="output">Receives one denoised page per plane.</param>
    /// <param name="baseline">Optional: receives the bilateral baseline of each plane.</param>
    /// <param name="onPage">Optional progress callback.</param>
    /// <param name="onNotice">Optional: called with notices (unlabelled axes taken as T).</param>
    /// <param name="cancellation">Checked before each plane and between tiles.</param>
    /// <returns>The number of planes denoised.</returns>
    /// <exception cref="ArgumentException">A multi-frame model on a stack with no time axis (a Z-stack, a single plane, …).</exception>
    public static int DenoiseStack(FrameDenoiser denoiser, ImageStack input, TiffStackWriter output, TiffStackWriter? baseline = null,
        PageDone? onPage = null, Action<string>? onNotice = null, CancellationToken cancellation = default) =>
        DenoisePlanned(denoiser, input, output, baseline, onPage, Plan(denoiser, input, onNotice), cancellation);

    /// <summary>
    /// What a stack's run decided before its first plane (brick 41): the routing decision and the time axis. Made once,
    /// so its notices are raised once and the output's Software tag (written with page 0) can state it.
    /// </summary>
    /// <param name="Correlated">The routing decision (null without a second model).</param>
    /// <param name="Flagged">Sampled planes whose noise check said correlated.</param>
    /// <param name="Sampled">Planes the noise check sampled (0 without a second model).</param>
    /// <param name="Time">The time axis, for a multi-frame model; null for a single-frame one.</param>
    internal sealed record StackPlan(bool? Correlated, int Flagged, int Sampled, TimeAxisChoice? Time);

    /// <summary>Makes <paramref name="input"/>'s <see cref="StackPlan"/>, raising its notices.</summary>
    /// <exception cref="ArgumentException">A multi-frame model on a stack without a time axis.</exception>
    internal static StackPlan Plan(FrameDenoiser denoiser, ImageStack input, Action<string>? onNotice)
    {
        TimeAxisChoice? time = null;
        if (denoiser.Net.Config.WindowFrames > 1)
        {
            time = TimeAxis.Resolve(input.Layout);
            if (time.Notice is { } notice) onNotice?.Invoke(notice);
        }
        var (correlated, flagged, sampled) = StackVerdict(denoiser, input, onNotice);
        return new StackPlan(correlated, flagged, sampled, time);
    }

    private static int DenoisePlanned(FrameDenoiser denoiser, ImageStack input, TiffStackWriter output, TiffStackWriter? baseline,
        PageDone? onPage, StackPlan plan, CancellationToken cancellation)
    {
        bool? correlated = plan.Correlated;
        if (plan.Time is { } time) return DenoiseWindowed(denoiser, input, output, baseline, onPage, time, cancellation, correlated);
        long allocatedAtLastCollection = GC.GetTotalAllocatedBytes();
        for (int p = 0; p < input.PlaneCount; p++)
        {
            cancellation.ThrowIfCancellationRequested();
            var raw = input.ReadPlane(p);
            var result = denoiser.Denoise(raw, null, baseline is not null, cancellation, correlated);
            output.Append(result.Output);
            baseline?.Append(result.Baseline!);
            onPage?.Invoke(p, raw, result);
            CollectIfDue(ref allocatedAtLastCollection);
        }
        return input.PlaneCount;
    }

    /// <summary>Planes of a stack the routing decision samples, evenly spread (all of them in a shorter stack).</summary>
    public const int RoutingSamples = 9;

    /// <summary>
    /// One routing decision for a whole stack (brick 41), when there is a second model to route to: the noise check on up to
    /// <see cref="RoutingSamples"/> evenly spread planes, and the majority wins (a tie stays with the main model). Every plane
    /// then goes to the same model, so a time-lapse can't switch networks, and with them its intensity response, from one
    /// frame to the next. A single image is still decided on its own (the same check on its one plane). Null without a second model.
    /// </summary>
    private static (bool? Correlated, int Flagged, int Sampled) StackVerdict(FrameDenoiser denoiser, ImageStack input, Action<string>? onNotice)
    {
        if (denoiser.CorrelatedNet is null) return (null, 0, 0);
        int n = input.PlaneCount, samples = Math.Min(RoutingSamples, n);
        int flagged = 0;
        for (int k = 0; k < samples; k++)
            if (FrameDenoiser.LooksCorrelated(input.ReadPlane(samples == 1 ? 0 : (int)((long)k * (n - 1) / (samples - 1))))) flagged++;
        bool correlated = 2 * flagged > samples;
        if (n > 1)
            onNotice?.Invoke($"noise check on {samples} of {n} planes: {flagged} look correlated, so every plane uses the " +
                             (correlated ? "correlated-noise model" : "main model"));
        return (correlated, flagged, samples);
    }

    /// <summary>Page garbage allowed to build up between the full collections <c>DenoiseStack</c> requests.</summary>
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
    /// A multi-frame model on a stack (bricks 39–41): plane p is denoised with its neighbours along the stack's time axis,
    /// every other axis held fixed and the series reflected at its ends (<see cref="TimeAxis.NeighbourPages"/>), and written in
    /// storage order, so a hyperstack keeps its page order. Each neighbour is read again from the file when it's needed
    /// (planes are random access), so nothing beyond the plane and its two neighbours is held, whatever the number of
    /// channels or slices between consecutive frames (rule 14). A skip-centre model (brick 40) uses the same neighbours; only
    /// its input leaves the plane itself out.
    /// </summary>
    private static int DenoiseWindowed(FrameDenoiser denoiser, ImageStack input, TiffStackWriter output, TiffStackWriter? baseline,
        PageDone? onPage, TimeAxisChoice time, CancellationToken cancellation, bool? correlated)
    {
        long allocatedAtLastCollection = GC.GetTotalAllocatedBytes();
        for (int p = 0; p < input.PlaneCount; p++)
        {
            cancellation.ThrowIfCancellationRequested();
            var raw = input.ReadPlane(p);
            var context = TimeAxis.NeighbourPages(input.Layout, time.Axis, p).Select(input.ReadPlane).ToArray();
            var result = denoiser.Denoise(raw, context, baseline is not null, cancellation, correlated);
            output.Append(result.Output);
            baseline?.Append(result.Baseline!);
            onPage?.Invoke(p, raw, result);
            CollectIfDue(ref allocatedAtLastCollection);
        }
        return input.PlaneCount;
    }

    /// <summary>
    /// Denoises one image file (TIFF or BigTIFF, single page or stack; MRC; or PNG) into a float32 TIFF with one page per
    /// plane, in the input's page order. The stack's axes come from its metadata, or from <paramref name="axes"/>.
    /// </summary>
    /// <param name="denoiser">The per-frame pipeline.</param>
    /// <param name="inputPath">The image to denoise.</param>
    /// <param name="outputPath">Where to write the denoised TIFF (overwritten). Must not be the input.</param>
    /// <param name="baselinePath">Optional: where to write the bilateral baseline.</param>
    /// <param name="onPage">Optional progress callback.</param>
    /// <param name="cancellation">Checked before each page and between tiles.</param>
    /// <param name="axes">Declared axes ending in YX (e.g. "TYX"), or null to use the file's own labels.</param>
    /// <param name="onNotice">Optional: called with notices (unlabelled axes taken as T).</param>
    /// <returns>The number of pages denoised.</returns>
    /// <exception cref="ArgumentException">An output path is the input path, or a multi-frame model meets a stack without a time axis.</exception>
    /// <exception cref="NotSupportedException">A file DORA deliberately doesn't read, e.g. an ImageJ stack over 4 GB with one page directory.</exception>
    public static int DenoiseFile(FrameDenoiser denoiser, string inputPath, string outputPath, string? baselinePath = null,
        PageDone? onPage = null, CancellationToken cancellation = default, string? axes = null, Action<string>? onNotice = null)
    {
        string inputFull = Path.GetFullPath(inputPath);
        foreach (var target in new[] { outputPath, baselinePath })
            if (target is not null && Path.GetFullPath(target) == inputFull)
                throw new ArgumentException($"{target}: the output would overwrite the input.");

        using var stack = OpenAsStack(inputPath, axes);
        var plan = Plan(denoiser, stack, onNotice);
        using var output = TiffStackWriter.Create(outputPath, TiffSampleType.Float32, tags: stack.OutputTags(Software(denoiser, plan)));
        using var baseline = baselinePath is null ? null
            : TiffStackWriter.Create(baselinePath, TiffSampleType.Float32, tags: stack.OutputTags(BaselineSoftware));
        return DenoisePlanned(denoiser, stack, output, baseline, onPage, plan, cancellation);
    }

    /// <summary>
    /// The Software tag (305) of a denoised file (brick 41): DORA's build, how it denoised (single-frame, or the window
    /// along which axis), each model's content hash and, with a second model, the routing decision and the noise check
    /// behind it. ASCII; provenance goes here so the ImageDescription can stay the input's.
    /// </summary>
    internal static string Software(FrameDenoiser denoiser, StackPlan plan)
    {
        var config = denoiser.Net.Config;
        string mode = plan.Time is not { } time ? "single-frame"
            : config.SkipsCentre ? $"skip-centre Noise2Noise, neighbours along {time.Axis}"
            : $"{config.WindowFrames}-frame window along {time.Axis}";
        var parts = new List<string> { $"DORA {Pipeline.Version}", $"denoised {mode}", $"model {WeightsFile.ContentHash(denoiser.Net)}" };
        if (denoiser.CorrelatedNet is { } second)
            parts.Add($"correlated-noise model {WeightsFile.ContentHash(second)} {(plan.Correlated == true ? "used" : "not used")} " +
                      $"(noise check correlated on {plan.Flagged} of {plan.Sampled} sampled planes)");
        if (denoiser.Stabilize) parts.Add("variance-stabilized");
        return string.Join("; ", parts);
    }

    /// <summary>The Software tag of a bilateral baseline file: what made it, and that it isn't DORA's output.</summary>
    internal static string BaselineSoftware => $"DORA {Pipeline.Version}; bilateral baseline for comparison (not DORA's output)";

    /// <summary>
    /// Number of planes in an image file: every plane of a TIFF or MRC stack (checked against its metadata, so an ImageJ
    /// stack over 4 GB that stores one page directory is refused rather than counted as one), 1 for a PNG.
    /// </summary>
    /// <param name="path">The image file.</param>
    /// <param name="axes">Declared axes ending in YX, or null (an OME-XML stack needs them; see <see cref="StackLayout.OmeXmlMessage"/>).</param>
    public static int CountPages(string path, string? axes = null)
    {
        using var stack = OpenAsStack(path, axes);
        return stack.PlaneCount;
    }

    /// <summary>
    /// Opens any supported image as a stack with its axes: TIFF and MRC through <see cref="ImageStack"/> (which checks the
    /// metadata against the pages); a PNG (always one frame) is decoded and re-wrapped as a one-page in-memory TIFF, which
    /// stores its values exactly as float32.
    /// </summary>
    internal static ImageStack OpenAsStack(string path, string? axes)
    {
        var head = new byte[1024];   // an MRC header is 1024 bytes
        int read;
        using (var probe = File.OpenRead(path)) read = probe.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        if (TiffCodec.HasTiffSignature(head.AsSpan(0, read)) || MrcReader.LooksLikeMrc(head.AsSpan(0, read))) return ImageStack.Open(path, axes);
        var frame = ImageLoader.Load(path);   // PNG, or a clear NotSupportedException
        return ImageStack.FromTiff(new TiffStackReader(new MemoryStream(TiffCodec.Encode(frame, TiffSampleType.Float32))), axes);
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

using System.Diagnostics;
using System.Globalization;
using DoraBio.Core;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;

namespace DoraBio.Cli;

/// <summary>
/// Headless front end to DoraBio.Core: everything the Godot app does, minus the UI.
/// Run with no arguments for usage.
/// </summary>
public static class Program
{
    private const string Usage = """
        dora-bio: DORA v2 fluorescence denoiser (headless)

        Commands:
          synth    --out DIR [--count 8] [--size 256 | --width W --height H] [--seed 1] [--peak 25] [--stack FILE.tif]
                   (--stack also writes the noisy frames as one multi-page uint16 TIFF)
                   [--sequence N [--drift D]]  (instead: COUNT time-lapses of N frames, one scene each, moving D px right
                    per frame, as DIR/noisy/seq_000.tif stacks with DIR/clean/ truth; for train --frames / --mode n2n)
                   Write a synthetic dataset: DIR/noisy/*.tif (uint16) and DIR/clean/*.tif (float32).
          train    --data DIR --out WEIGHTS.bin [--mode n2v|supervised|n2n] [--epochs 30] [--steps 50]
                   [--batch 16] [--patch 64] [--lr 0.001] [--journey FILE.json] [--seed 1234]
                   [--arch dilated|unet] [--features 24 (dilated) / 16 (unet)] [--depth 2 (unet)]
                   [--condition none|noise]  (noise: tell the network each frame's noise level, FiLM; quality pass Q13)
                   [--init WEIGHTS.bin]   (fine-tune existing weights instead of starting fresh)
                   [--mask-ratio 0.02]    (Noise2Void: fraction of pixels hidden per patch)
                   [--route auto|off] [--out-correlated FILE.bin]  (Noise2Void default since Q20: route each image by the noise
                    check to a U-Net (independent noise) or a structured-blind-spot network (correlated); a mixed set writes
                    two files, the second for denoise --weights-correlated. Any recipe option below means --route off)
                   [--select auto|loss] [--patience 6] [--keep best|last] [--save-last FILE.bin]
                    (what picks the epoch: auto = validation PSNR when clean images exist, else the loss; loss = always the
                     validation loss, so Noise2Void never uses ground truth; patience 0 = never stop early; quality pass Q2)
                   [--describe TEXT]      (what the data is; stored in the weights' provenance)
                   [--augment dihedral|v1] (v1 = v1's ±15° rotation + 0.85-1.15 scale, nearest neighbour; for comparison)
                   [--struct off|auto|always] [--footprint diamond-2]  (Noise2Void: structured blind spots for
                    correlated noise; auto = only on images whose noise check flags them)
                   [--subset correlated|independent]  (train only on the images whose noise check says so)
                   [--vst]  (Noise2Void in the variance-stabilized domain; apply the model with --vst too)
                   Time-lapses (brick 39; --data is a stack file or a folder of stacks, every plane a frame):
                    [--frames 3]  (Noise2Void with the spatiotemporal blind spot: input t-1, t, t+1, blind spots in t only)
                    [--mode n2n]  (Noise2Noise from neighbouring frames: t -> t+1 and t+1 -> t; one input frame only)
                    [--mode skip-centre]  (skip-centre Noise2Noise, brick 40: input t-1, t+1, target t, never t itself;
                     the output is their mean minus the network's correction)
                    [--frame-block 32]  (frames per validation group, so one long time-lapse still splits by time)
                    [--frame-range A-B]  (train on frames A..B of one stack, a standalone sequence; also for single frames)
                    [--axes TYX|TZCYX…]  (the stack's axes if its metadata is missing or wrong: each channel and slice is
                     its own time series, a Z-stack is refused, and unlabelled planes are taken as frames)
                    [--validation-range C-D]  (validate on frames C..D of the same stack instead of splitting by blocks:
                     one patch source per interior frame, the same frames and positions for every mode; with Noise2Void
                     it needs --route off)
                   FMD folders: [--fmd-frames 4] [--fmd-levels raw,avg2,avg4,avg8,avg16] [--fmd-categories A,B]
                   External sets: --ext NAME [--data data/ext]  (its training split from tools/fetch_ext.sh; supervised
                    pairs are normalized each by their own percentiles; --mode n2v uses the noisy images alone)
                   (FOV 19, FMD's test FOV, is always excluded from training)
          denoise  --weights WEIGHTS.bin --in IMAGE --out OUT.tif [--baseline BASELINE.tif] [--tile 128]
                   (IMAGE may be a multi-page TIFF stack: OUT.tif is then a stack with the same pages; or a folder:
                    --out and --baseline are then folders, and every image in it is denoised. Frames stream one at a time)
                   [--weights-correlated W2.bin]  (used instead if the image's noise check says correlated)
                   [--vst]  (for models trained with --vst)
                   [--axes TYX|ZYX|TZCYX…]  (the stack's axes when its metadata is missing or wrong; a time-lapse model
                    steps along T with every other axis fixed, refuses a Z-stack, and takes unlabelled planes as T)
                   [--report REPORT.html]  (one self-contained page: input, model, settings, and the residual to look at
                    before measuring; one image or stack, not a folder)
                   Output is float32 TIFF in the input's original intensity units, one page per input page, in order.
          eval     --weights WEIGHTS.bin --data DIR
                   Per-image PSNR/SSIM of noisy, bilateral and DORA against clean (needs clean images).
          info     --weights WEIGHTS.bin   |   info --default-arch   (prints dilated or unet)
                   |   info --default-condition   (supervised training's default --condition: none or noise)
                   |   info --default-timelapse   (train --data STACK with no --mode/--frames: none, n2n, frames-3 or skip-centre)
          bench    [--quick] [--json OUT.json]
                   Timings: denoise a frame and one training step, for both architectures (see PerformanceBench).
          bench --roofline [--json OUT.json]
                   This CPU's measured peaks (mul+add and FMA at 256/512 bits, read bandwidth per cache level), each
                   convolution kernel's GFLOP/s and bytes against them, and where a stack run's time goes (~1 min).
          diagnose --in IMAGE
                   Does the image's noise look independent between pixels, as Noise2Void assumes? (no ground truth needed)
          stack-check [--root data/ext] [--dataset NAME]
                   Check every downloaded stack dataset (3D, time-lapse, multichannel, raw SIM) against its published
                   counts and declared axes, and that each clean partner matches its noisy file. Exit 1 on any disagreement.
          stack-info --in FILE.tif|FILE.mrc [--axes TZCYX]
                   What a stack holds: format (TIFF, BigTIFF, MRC), axes, shape and plane size, from its metadata.
          stack-export --dataset mdc25-tribolium-2d|biosr-ccps-2d [--root data/ext]
                   Write a derived 2D set from its downloaded stack set (the scoreboard's splits, brick 38); then train
                   --ext and ext-eval take it like a downloaded set.
          noise-residual (--dataset NAME [--root data/ext] | --fmd data/fmd | --stack LOW.tif --reference HIGH.tif
                    [--frames 24] [--offset dt,dx,dy]) [--only PREFIX]
                   The real noise correlation of a set's test pairs, per axis and region (background, interiors, edges),
                   from (noisy - matched reference) / sqrt(reference); next to the blind check's verdicts on the same images.
          seq-eval --stack LOW.tif --reference HIGH.tif --frames A-B [--weights W.bin [--label TEXT]] [--per-frame FILE.tsv]
                   A time-lapse block against its frame-aligned high-SNR movie (brick 40), frame by frame with CARE's
                   protocol, the block a standalone sequence (reflected at its ends): PSNR, SSIM, PSNR on the pixels that
                   change most (dynamic: |S_t - (S_t-1 + S_t+1)/2|; steady: |S_t - S_t-1| + |S_t - S_t+1|; top 10%,
                   S = sqrt(max(blurred reference, 0))) and brick 32's fidelity, for the noisy input and the model.
          seq-diagnose --stack LOW.tif --reference HIGH.tif --frames A-B --weights W1.bin,W2.bin,... --labels L1,L2,...
                   Post-hoc checks of brick 40's guards (not part of its reading): how much real change its masks hold
                   (low- vs high-SNR signed changes), the share of real transients and motion each model keeps relative
                   to the raw frames, time-averaged fidelity, gain jitter and flicker over the block, tile-trace correlation.
          fmd-export --root data/fmd --out DIR
                   FMD for other tools: DORA's default training frames (4 raw per FOV, never FOV 19) to DIR/train/low_SNR,
                   test_mix raw to DIR/test/low_SNR, float32 TIFF; score their outputs with fmd-eval --predictions.
          ext-check [--root data/ext]
                   Verify the public paired datasets from tools/fetch_ext.sh: counts, pairing, decoding, brightness.
          fidelity (--dataset NAME [--root data/ext] | --fmd data/fmd [--level raw]) (--weights W.bin | --predictions DIR [--label TEXT])
                   What scientists measure, vs the high-SNR reference: are regional intensities still proportional
                   (slope 1.00) and how far off (ratio error), and are spots kept (found) and not invented (real)?
          ext-eval --dataset NAME (--weights WEIGHTS.bin | --predictions DIR [--label TEXT]) [--root data/ext] [--precise]
                   [--split test|validation --seed S]  (validation: the training images a run with seed S held out)
                   Score a model, or another tool's outputs (DIR/<test file name>), on an external set's test images with
                   CARE's protocol: reference normalized by its 0.1-99.9 percentiles, outputs matched by least squares.
          fmd-check --root data/fmd
                   Verify a downloaded FMD dataset: structure, test_mix, and no train/test leakage.
          fmd-eval  --weights WEIGHTS.bin --root data/fmd [--levels raw,avg2,...] [--per-image] [--precise]
                   [--predictions DIR [--label TEXT]]  (instead of --weights: score another tool's outputs on test_mix raw,
                    DIR/<test image name>.tif in 8-bit units, clipped to 0-255 like DORA's)
                   [--validation-seed S]  (score the FOVs a training run with --seed S held out, one frame each, not test_mix)
                   [--sigma-scale K]  (noise-conditioned models: tell them K × each image's measured noise σ)
                   [--weights-correlated W2.bin]  (images whose noise check says correlated go to W2 instead)
                   [--weights-clean W2.bin --noise-threshold S]  (images with noise σ ≤ S go to W2 instead)
          fmd-route --weights A.bin --weights-clean B.bin --root data/fmd [--seed 2026]
                   Chooses the noise threshold for --weights-clean on the validation FOVs of a training run (never test_mix).
                   [--vst]  (for models trained with --vst)
                   PSNR/SSIM (8-bit scale) on FMD's held-out test_mix: noisy vs bilateral vs DORA, per microscope type.
          fmd-noise --root data/fmd [--levels raw,avg2,...]
                   How correlated test_mix's real noise is between neighbouring pixels (Noise2Void assumes 0).
          fmd-diagnose --root data/fmd [--levels raw] [--validation-seed S] [--choose-threshold]
                   Checks `diagnose` (noisy image only) against fmd-noise's ground truth; exit 1 on any disagreement.
        """;

    /// <summary>Entry point: <c>dora-bio &lt;command&gt; [--option value]…</c>.</summary>
    /// <returns>0 on success, 1 for usage (no arguments), 2 for a user-facing error (bad option, missing or unsupported file).</returns>
    /// <remarks>
    /// Expected failures (bad input) print one "error: …" line instead of a stack trace. Anything else is a bug and
    /// is allowed to crash loudly. Numbers are parsed and printed with the invariant culture, so a German-locale
    /// machine still reads "0.001" correctly.
    /// </remarks>
    public static int Main(string[] args)
    {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Console.WriteLine(Usage);
            return args.Length == 0 ? 1 : 0;
        }

        try
        {
            var options = Options.Parse(args.Skip(1));
            options.RejectUnknown();
            return args[0] switch
            {
                "synth" => Synth(options),
                "train" => Train(options),
                "denoise" => Denoise(options),
                "eval" => Eval(options),
                "info" => Info(options),
                "fmd-check" => FmdCheck(options),
                "ext-check" => ExtCheck(options),
                "stack-check" => StackCheck(options),
                "stack-info" => StackInfo(options),
                "stack-export" => StackExport(options),
                "fmd-export" => FmdExportCommand(options),
                "noise-residual" => NoiseResidualCommand(options),
                "seq-eval" => SeqEval(options),
                "seq-diagnose" => SeqDiagnose(options),
                "ext-eval" => ExtEval(options),
                "fidelity" => FidelityCommand(options),
                "fmd-eval" => FmdEval(options),
                "fmd-noise" => FmdNoise(options),
                "fmd-diagnose" => FmdDiagnose(options),
                "fmd-route" => FmdRoute(options),
                "diagnose" => Diagnose(options),
                "bench" => Bench(options),
                _ => throw new ArgumentException($"Unknown command '{args[0]}'."),
            };
        }
        catch (Exception e) when (e is ArgumentException or FileNotFoundException or DirectoryNotFoundException
                                       or InvalidDataException or NotSupportedException or FormatException)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 2;
        }
    }

    /// <summary>
    /// <c>synth</c>: writes noisy (uint16) and clean (float32) TIFF pairs in the noisy/ + clean/ layout that
    /// <see cref="DatasetLoader"/> reads. Each frame uses seed + 7919·i, so frames differ but the set is reproducible.
    /// </summary>
    private static int Synth(Options o)
    {
        string outDir = o.Required("out");
        int count = o.Int("count", 8), size = o.Int("size", 256), seed = o.Int("seed", 1);
        int width = o.Int("width", size), height = o.Int("height", size);
        var synthOptions = new SyntheticOptions { PeakPhotons = o.Double("peak", 25) };
        Directory.CreateDirectory(Path.Combine(outDir, "noisy"));
        Directory.CreateDirectory(Path.Combine(outDir, "clean"));
        if (o.Has("sequence"))
        {
            // Time-lapses (brick 39): one scene per sequence, every frame with its own noise, the scene drifting right.
            int frames = o.Int("sequence", 8), drift = o.Int("drift", 0);
            if (frames < 1) throw new ArgumentException("--sequence must be at least 1.");
            if (drift < 0) throw new ArgumentException("--drift must be 0 or more.");
            for (int i = 0; i < count; i++)
            {
                var sequence = SyntheticFluorescence.GenerateSequence(width, height, frames, seed + 7919 * i, drift, synthOptions);
                string name = $"seq_{i:D3}.tif";
                File.WriteAllBytes(Path.Combine(outDir, "noisy", name), TiffCodec.EncodeStack(sequence.Select(f => f.Noisy).ToList(), TiffSampleType.UInt16));
                File.WriteAllBytes(Path.Combine(outDir, "clean", name), TiffCodec.EncodeStack(sequence.Select(f => f.Clean).ToList(), TiffSampleType.Float32));
            }
            Console.WriteLine($"🎞 wrote {count} synthetic time-lapse(s) of {frames} {width}×{height} frames (drift {drift} px/frame, peak {synthOptions.PeakPhotons} photons) to {outDir}");
            return 0;
        }

        var noisyFrames = new List<GrayImage>();
        for (int i = 0; i < count; i++)
        {
            var frame = SyntheticFluorescence.Generate(width, height, seed + 7919 * i, synthOptions);
            string name = $"synthetic_{i:D3}.tif";
            File.WriteAllBytes(Path.Combine(outDir, "noisy", name), TiffCodec.Encode(frame.Noisy, TiffSampleType.UInt16));
            File.WriteAllBytes(Path.Combine(outDir, "clean", name), TiffCodec.Encode(frame.Clean, TiffSampleType.Float32));
            noisyFrames.Add(frame.Noisy);
        }
        if (o.StringOrNull("stack") is { } stackPath)
        {
            File.WriteAllBytes(stackPath, TiffCodec.EncodeStack(noisyFrames, TiffSampleType.UInt16));
            Console.WriteLine($"🗂 wrote the {count} noisy frames as one {count}-page stack: {stackPath}");
        }
        Console.WriteLine($"🧪 wrote {count} synthetic {width}×{height} frames (peak {synthOptions.PeakPhotons} photons) to {outDir}");
        return 0;
    }

    /// <summary>
    /// <c>train</c>: loads a dataset folder and trains through <see cref="TrainingJob"/>, which handles journaling and
    /// best-weight checkpointing. Mode defaults to supervised only when every image has a clean partner.
    /// <c>--init</c> fine-tunes existing weights instead of starting from a fresh network.
    /// </summary>
    private static int Train(Options o)
    {
        // The recipe, routing and mode decisions live in Pipeline.Train (brick 41), which the Godot app calls too; an
        // option left out is null here, which takes the same default as before.
        Pipeline.Train(new TrainRequest
        {
            Data = o.StringOrNull("data"),
            Ext = o.StringOrNull("ext"),
            Out = o.Required("out"),
            OutCorrelated = o.StringOrNull("out-correlated"),
            Journey = o.StringOrNull("journey"),
            Describe = o.StringOrNull("describe"),
            SaveLast = o.StringOrNull("save-last"),
            Mode = o.StringOrNull("mode"),
            Frames = o.IntOrNull("frames"),
            FrameRange = o.StringOrNull("frame-range") is { } fr ? FrameRange.Parse(fr) : null,
            ValidationRange = o.StringOrNull("validation-range") is { } vr ? FrameRange.Parse(vr) : null,
            FrameBlock = o.IntOrNull("frame-block"),
            Axes = o.StringOrNull("axes"),
            Vst = o.Has("vst"),
            FmdFrames = o.IntOrNull("fmd-frames"),
            FmdLevels = o.StringOrNull("fmd-levels"),
            FmdCategories = o.StringOrNull("fmd-categories"),
            Epochs = o.IntOrNull("epochs"),
            Steps = o.IntOrNull("steps"),
            Batch = o.IntOrNull("batch"),
            Augment = o.StringOrNull("augment"),
            Patch = o.IntOrNull("patch"),
            MaskRatio = o.DoubleOrNull("mask-ratio"),
            LearningRate = o.DoubleOrNull("lr"),
            Seed = o.IntOrNull("seed"),
            Patience = o.IntOrNull("patience"),
            Select = o.StringOrNull("select"),
            Keep = o.StringOrNull("keep"),
            Route = o.StringOrNull("route"),
            Init = o.StringOrNull("init"),
            Arch = o.StringOrNull("arch"),
            Features = o.IntOrNull("features"),
            Depth = o.IntOrNull("depth"),
            Condition = o.StringOrNull("condition"),
            Struct = o.StringOrNull("struct"),
            Footprint = o.StringOrNull("footprint"),
            Subset = o.StringOrNull("subset"),
        }, Console.WriteLine);
        return 0;
    }

    /// <summary>
    /// <c>denoise</c>: normalize, denoise with tiled inference, then map back to the input's own intensity units and
    /// write float32 TIFF. Writing float rather than 16-bit keeps sub-count precision, which downstream intensity
    /// measurements need. <c>--baseline</c> also writes the bilateral result for comparison.
    /// <para>
    /// The input can be one image, a multi-page TIFF stack (the output is a stack with the same pages), or a folder
    /// (every image in it, written to the <c>--out</c> folder under the same names). Stacks and folders stream one frame
    /// at a time through <see cref="BatchDenoising"/>, so memory stays at one frame's worth whatever their length; every
    /// frame goes through the same <see cref="FrameDenoiser"/> as a single image, so it comes out identical.
    /// </para>
    /// </summary>
    private static int Denoise(Options o)
    {
        Pipeline.Denoise(new DenoiseRequest
        {
            Input = o.Required("in"),
            Output = o.Required("out"),
            Weights = o.Required("weights"),
            CorrelatedWeights = o.StringOrNull("weights-correlated"),
            Baseline = o.StringOrNull("baseline"),
            Axes = o.StringOrNull("axes"),
            Vst = o.Has("vst"),
            TileCore = o.Int("tile", TiledInference.DefaultTileCore),
            Report = o.StringOrNull("report"),
        }, Console.WriteLine);
        return 0;
    }

    /// <summary>
    /// <c>eval</c>: per-image PSNR/SSIM of noisy, bilateral and DORA against clean. Run it on FOVs the model never
    /// trained on; images from the training set score optimistically.
    /// </summary>
    private static int Eval(Options o)
    {
        var (net, info) = WeightsFile.Load(o.Required("weights"));
        var dataset = DatasetLoader.LoadDirectory(o.Required("data")).Where(d => d.HasClean).ToList();
        if (dataset.Count == 0) throw new ArgumentException("eval needs clean images (a clean/ or gt/ folder).");

        Console.WriteLine($"🧬 model: {info.Provenance}");
        Console.WriteLine($"{"image",-28} {"noisy",14} {"bilateral",14} {"DORA",14}   (PSNR dB / SSIM)");
        double[] totals = new double[6];
        foreach (var pair in dataset)
        {
            var clean = pair.Clean!;
            var bilateral = BilateralBaseline.FilterAuto(pair.Noisy);
            var dora = TiledInference.Denoise(net, pair.Noisy);
            double[] row =
            [
                Metrics.Psnr(pair.Noisy.Pixels, clean.Pixels), Metrics.Ssim(pair.Noisy, clean),
                Metrics.Psnr(bilateral.Pixels, clean.Pixels), Metrics.Ssim(bilateral, clean),
                Metrics.Psnr(dora.Pixels, clean.Pixels), Metrics.Ssim(dora, clean),
            ];
            for (int i = 0; i < 6; i++) totals[i] += row[i];
            Console.WriteLine($"{Truncate(pair.Name, 28),-28} {row[0],6:0.00}/{row[1],-6:0.000} {row[2],6:0.00}/{row[3],-6:0.000} {row[4],6:0.00}/{row[5],-6:0.000}");
        }
        var m = totals.Select(t => t / dataset.Count).ToArray();
        Console.WriteLine($"{"mean",-28} {m[0],6:0.00}/{m[1],-6:0.000} {m[2],6:0.00}/{m[3],-6:0.000} {m[4],6:0.00}/{m[5],-6:0.000}");
        Console.WriteLine("Note: images in the training set score optimistically; evaluate on held-out FOVs.");
        return 0;
    }

    /// <summary>
    /// FMD loading options from <c>--fmd-frames</c>, <c>--fmd-levels</c> and <c>--fmd-categories</c>. The test FOV
    /// exclusion is deliberately not exposed: the CLI never trains on FMD's test FOV.
    /// </summary>
    private static FmdOptions FmdOptionsFrom(Options o) => new()
    {
        FramesPerFov = o.Int("fmd-frames", 4),
        NoiseLevels = o.String("fmd-levels", "raw").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
        Categories = o.StringOrNull("fmd-categories")?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
    };

    /// <summary>
    /// <c>fmd-eval</c>: deterministic benchmark on test_mix (see <see cref="FmdBenchmark"/>). The output is designed to be
    /// pasted verbatim into docs/RESULTS.md, where gate 13 re-runs it and requires an exact match.
    /// </summary>
    private static int FmdEval(Options o)
    {
        if (o.StringOrNull("predictions") is { } predictions) return FmdEvalPredictions(o, predictions);
        var (net, _) = WeightsFile.Load(o.Required("weights"));
        // --weights-correlated: a second model for the images whose noise check says correlated (P3's routed design).
        // --weights-clean + --noise-threshold: a second model for images whose noise σ is at or below the threshold (Q10).
        var correlatedNet = o.StringOrNull("weights-correlated") is { } c ? WeightsFile.Load(c).Net : null;
        var cleanNet = o.StringOrNull("weights-clean") is { } q ? WeightsFile.Load(q).Net : null;
        if (correlatedNet is not null && cleanNet is not null)
            throw new ArgumentException("Use either --weights-correlated or --weights-clean, not both.");
        double noiseThreshold = cleanNet is null ? 0 : o.Double("noise-threshold", double.NaN);
        if (cleanNet is not null && double.IsNaN(noiseThreshold))
            throw new ArgumentException("--weights-clean needs --noise-threshold (dora-bio fmd-route chooses one on validation data).");
        int routed = 0;
        DenoisingNet Pick(GrayImage noisy)
        {
            var second = correlatedNet ?? cleanNet;
            bool useSecond = correlatedNet is not null ? FrameDenoiser.LooksCorrelated(noisy) : cleanNet is not null && NoiseLevelRouter.Sigma(noisy) <= noiseThreshold;
            if (!useSecond) return net;
            routed++;
            return second!;
        }
        string testMix = Path.Combine(o.Required("root"), "test_mix");
        var levels = o.String("levels", "raw").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        double sigmaScale = o.Double("sigma-scale", 1);
        if (!(sigmaScale > 0)) throw new ArgumentException("--sigma-scale must be positive.");
        int? validationSeed = o.Has("validation-seed") ? o.Int("validation-seed", 0) : null;
        string scope = validationSeed is { } vs ? $"validation (seed {vs})" : "test_mix";
        string scaled = sigmaScale == 1 ? "" : $", condition from {sigmaScale:0.###}× the measured noise σ";
        foreach (string level in levels)
        {
            routed = 0;
            var scores = validationSeed is { } seed
                ? FmdBenchmark.EvaluatePlan(Pick, FmdBenchmark.ValidationPlan(o.Required("root"), seed, [level]), stabilize: o.Has("vst"), sigmaScale: sigmaScale)
                : FmdBenchmark.Evaluate(Pick, testMix, level, stabilize: o.Has("vst"), sigmaScale: sigmaScale);
            Console.WriteLine($"{scope} level {level}: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale{scaled}");
            Console.WriteLine($"{"modality",-10} {"n",3}  {"noisy",-15} {"bilateral",-15} {"DORA",-15}");
            foreach (var (group, count, m) in FmdBenchmark.Summarize(scores))
                Console.WriteLine($"{group,-10} {count,3}  {m.NoisyPsnr,6:0.00} / {m.NoisySsim:0.000}  {m.BilateralPsnr,6:0.00} / {m.BilateralSsim:0.000}  {m.DoraPsnr,6:0.00} / {m.DoraSsim:0.000}");
            if (correlatedNet is not null)
                Console.WriteLine($"routed by the noise check: {routed} of {scores.Count} images to the correlated-noise model");
            if (cleanNet is not null)
                Console.WriteLine($"routed by noise level (σ ≤ {noiseThreshold:0.00000}): {routed} of {scores.Count} images to the clean-input model");
            if (o.Has("per-image"))
                foreach (var s in scores)
                    Console.WriteLine($"  {s.Name,-40} {s.NoisyPsnr,6:0.00} {s.BilateralPsnr,6:0.00} {s.DoraPsnr,6:0.00}");
            if (o.Has("precise"))   // for a bar that sits within rounding of the 2-decimal table (Q2's FMD guard)
            {
                var all = FmdBenchmark.Summarize(scores)[^1].Mean;
                Console.WriteLine($"all, unrounded: noisy {all.NoisyPsnr:0.0000} dB, bilateral {all.BilateralPsnr:0.0000} dB, DORA {all.DoraPsnr:0.0000} dB");
            }
        }
        return 0;
    }

    /// <summary>
    /// <c>fmd-eval --predictions DIR</c>: another tool's outputs for test_mix raw (named by <see cref="FmdExport.PredictionName"/>),
    /// scored with FMD's protocol and printed in fmd-eval's table (brick 38's scoreboard).
    /// </summary>
    private static int FmdEvalPredictions(Options o, string dir)
    {
        if (o.Has("weights")) throw new ArgumentException("Use either --weights or --predictions, not both.");
        if (o.String("levels", "raw") != "raw") throw new ArgumentException("--predictions scores test_mix raw only.");
        var plan = FmdDataset.PlanTestMix(Path.Combine(o.Required("root"), "test_mix"), "raw");
        var scores = FmdBenchmark.EvaluatePredictions(p => ImageLoader.Load(Path.Combine(dir, FmdExport.PredictionName(p))), plan);
        Console.WriteLine("test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale");
        Console.WriteLine($"method: {o.String("label", "outputs in " + Path.GetFileName(Path.TrimEndingDirectorySeparator(dir)))}");
        Console.WriteLine($"{"modality",-10} {"n",3}  {"noisy",-15} {"bilateral",-15} {"method",-15}");
        foreach (var (group, count, m) in FmdBenchmark.Summarize(scores))
            Console.WriteLine($"{group,-10} {count,3}  {m.NoisyPsnr,6:0.00} / {m.NoisySsim:0.000}  {m.BilateralPsnr,6:0.00} / {m.BilateralSsim:0.000}  {m.DoraPsnr,6:0.00} / {m.DoraSsim:0.000}");
        if (o.Has("per-image"))
            foreach (var s in scores)
                Console.WriteLine($"  {s.Name,-40} {s.NoisyPsnr,6:0.00} {s.BilateralPsnr,6:0.00} {s.DoraPsnr,6:0.00}");
        if (o.Has("precise"))
        {
            var all = FmdBenchmark.Summarize(scores)[^1].Mean;
            Console.WriteLine($"all, unrounded: noisy {all.NoisyPsnr:0.0000} dB, bilateral {all.BilateralPsnr:0.0000} dB, method {all.DoraPsnr:0.0000} dB");
        }
        return 0;
    }

    /// <summary>
    /// <c>noise-residual</c>: the real noise correlation of a set's test pairs per axis and region, from its references
    /// (<see cref="NoiseResidual"/>), next to what the blind check says about the same noisy images. Added after brick 38 for
    /// its Tribolium question; deterministic, so docs/RESULTS.md records it verbatim and gate 38 re-runs it.
    /// </summary>
    private static int NoiseResidualCommand(Options o)
    {
        // Each entry loads one (noisy, reference) pair when measured, so only one pair is in memory at a time.
        List<(string Name, Func<(GrayImage Noisy, GrayImage Reference)> Load)> plan;
        string scope;
        if (o.StringOrNull("fmd") is { } fmdRoot)
        {
            plan = FmdDataset.PlanTestMix(Path.Combine(fmdRoot, "test_mix"), "raw")
                .Select(p => (Path.GetFileName(p.NoisyPath), (Func<(GrayImage, GrayImage)>)(() => (ImageLoader.Load(p.NoisyPath), ImageLoader.Load(p.CleanPath))))).ToList();
            scope = "FMD test_mix raw";
        }
        else if (o.StringOrNull("stack") is { } stackPath)
        {
            // A time-lapse and its high-SNR movie (before brick 40): frames sampled evenly over the movie; --offset dt,dx,dy
            // pairs noisy frame t with reference frame t + dt moved by (dx, dy), both cropped to their shared interior.
            string referencePath = o.Required("reference");
            var offset = o.String("offset", "0,0,0").Split(',').Select(v => int.Parse(v.Trim(), CultureInfo.InvariantCulture)).ToArray();
            if (offset.Length != 3) throw new ArgumentException("--offset is dt,dx,dy (three integers).");
            int dt = offset[0], dx = offset[1], dy = offset[2];
            using var probe = ImageStack.Open(stackPath);
            int frames = probe.PlaneCount, samples = Math.Min(o.Int("frames", 24), frames);
            using (var reference = ImageStack.Open(referencePath))
                if (reference.PlaneCount != frames || reference.Width != probe.Width || reference.Height != probe.Height)
                    throw new ArgumentException("The stack and its reference must have the same frames and size.");
            int width = probe.Width - Math.Abs(dx), height = probe.Height - Math.Abs(dy);
            var indices = Enumerable.Range(0, samples).Select(k => (int)((long)k * (frames - 1) / Math.Max(1, samples - 1)))
                .Where(t => t + dt >= 0 && t + dt < frames).Distinct().ToList();
            plan = indices.Select(t => ($"t{t}", (Func<(GrayImage, GrayImage)>)(() =>
            {
                using var noisyStack = ImageStack.Open(stackPath);
                using var referenceStack = ImageStack.Open(referencePath);
                // Reference pixel (x + dx, y + dy) pairs with noisy pixel (x, y); the shared interior drops |dx|, |dy| edges.
                var noisy = noisyStack.ReadPlane(t).Crop(Math.Max(0, -dx), Math.Max(0, -dy), width, height);
                var matched = referenceStack.ReadPlane(t + dt).Crop(Math.Max(0, dx), Math.Max(0, dy), width, height);
                return (noisy, matched);
            }))).ToList();
            scope = $"{Path.GetFileName(stackPath)}, {indices.Count} frames sampled evenly, reference offset dt {dt}, dx {dx}, dy {dy}";
        }
        else
        {
            var dataset = ExtDatasets.Get(o.Required("dataset"));
            plan = ExtDatasets.PlanPairs(o.String("root", "data/ext"), dataset, ExtSplit.Test)
                .Select(p => (Path.GetFileName(p.NoisyPath), (Func<(GrayImage, GrayImage)>)(() => (ImageLoader.Load(p.NoisyPath), ImageLoader.Load(p.CleanPath))))).ToList();
            scope = $"{dataset.Name} test";
        }
        if (o.StringOrNull("only") is { } prefix)
        {
            plan = plan.Where(p => p.Name.StartsWith(prefix, StringComparison.Ordinal)).ToList();
            scope += $" ({prefix}*)";
        }
        if (plan.Count == 0) throw new ArgumentException("No pairs to measure.");
        int called = 0, judged = 0;
        var right = new List<double>();
        var down = new List<double>();
        IEnumerable<(GrayImage, GrayImage)> Pairs()
        {
            foreach (var p in plan)
            {
                var (noisy, referenceImage) = p.Load();
                try
                {
                    var d = NoiseDiagnostics.Diagnose(noisy);
                    judged++;
                    if (d.Correlated) called++;
                    right.Add(d.Right1);
                    down.Add(d.Down1);
                }
                catch (ArgumentException) { }   // too few flat tiles to judge
                yield return (noisy, referenceImage);
            }
        }
        var rows = NoiseResidual.Measure(Pairs());
        Console.WriteLine($"{scope}, {plan.Count} pairs: correlation of the noise residual (noisy − matched reference) / √reference with a neighbour");
        Console.WriteLine($"{"region",-11} {"pixels",10}  {"right 1",7} {"down 1",7} {"right 2",7} {"down 2",7}");
        foreach (var r in rows)
            Console.WriteLine($"{r.Region,-11} {r.Pixels,10}  {r.Right1,7:0.000} {r.Down1,7:0.000} {r.Right2,7:0.000} {r.Down2,7:0.000}");
        static double Median(List<double> v) { if (v.Count == 0) return double.NaN; v.Sort(); return v[v.Count / 2]; }
        Console.WriteLine($"blind check (noisy image only, threshold {NoiseDiagnostics.CorrelatedThreshold:0.00}): {called} of {judged} judged images called correlated " +
                          $"({plan.Count - judged} too small to judge); median estimate right 1 {Median(right):0.000}, down 1 {Median(down):0.000}");
        return 0;
    }

    /// <summary><c>fmd-export</c>: FMD's training frames and test_mix raw in the external sets' layout (<see cref="FmdExport"/>).</summary>
    private static int FmdExportCommand(Options o)
    {
        string outDir = o.Required("out");
        var (train, test) = FmdExport.Export(o.Required("root"), outDir);
        Console.WriteLine($"FMD exported to {outDir}: {train} training frames (train/low_SNR), {test} test images (test/low_SNR)");
        return 0;
    }

    /// <summary><c>stack-export</c>: a derived 2D set from its stack set (<see cref="ExtDerived"/>).</summary>
    private static int StackExport(Options o)
    {
        string root = o.String("root", "data/ext");
        var dataset = ExtDatasets.Get(o.Required("dataset"));
        var (train, test) = ExtDerived.Export(root, dataset);
        Console.WriteLine($"{dataset.Name} (from {dataset.Source}): {train} training and {test} test planes in {Path.Combine(root, dataset.Name)}");
        return 0;
    }

    /// <summary>
    /// <c>fmd-noise</c>: correlation of test_mix's real noise between neighbouring pixels, per microscope type (see
    /// <see cref="FmdBenchmark.NoiseCorrelation"/>). Like fmd-eval, the output is recorded verbatim in docs/RESULTS.md
    /// and re-checked by gate 13.
    /// </summary>
    private static int FmdNoise(Options o)
    {
        string testMix = Path.Combine(o.Required("root"), "test_mix");
        var levels = o.String("levels", "raw").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (string level in levels)
        {
            Console.WriteLine($"test_mix level {level}: correlation of the noise (noisy - ground truth) with a neighbour, mean over images");
            Console.WriteLine($"{"modality",-10} {"n",3}  {"right 1",7} {"down 1",7} {"right 2",7}  {"noise SD",8}");
            foreach (var (group, count, m) in FmdBenchmark.SummarizeNoise(FmdBenchmark.MeasureNoise(testMix, level)))
                Console.WriteLine($"{group,-10} {count,3}  {m.Right1,7:0.000} {m.Down1,7:0.000} {m.Right2,7:0.000}  {m.NoiseSd,8:0.00}");
        }
        return 0;
    }

    /// <summary>
    /// <c>fmd-check</c>: structure of every category and of test_mix, plus the leakage guard. Every test_mix
    /// ground truth must equal its category's FOV 19 ground truth pixel for pixel, and the default training plan
    /// must contain no FOV 19 file. Exit 1 on any problem (used by bricks.sh gate 12).
    /// </summary>
    private static int FmdCheck(Options o)
    {
        var (problems, summary) = FmdDataset.Verify(o.Required("root"));
        foreach (string p in problems) Console.WriteLine($"✗ {p}");
        if (problems.Count > 0) return 1;
        Console.WriteLine($"✓ {summary}");
        return 0;
    }

    /// <summary>
    /// <c>bench</c>: <see cref="PerformanceBench.Run"/>, printed as a table and optionally written as JSON
    /// (schema 2, checked by tools/ci/check_bench.py; gate 24 compares it with tools/ci/perf_budget.json).
    /// </summary>
    private static int Bench(Options o)
    {
        if (o.Has("roofline")) return BenchRoofline(o);
        bool quick = o.Has("quick");
        var timings = PerformanceBench.Run(quick);
        Console.WriteLine($"{"operation",-22} {"architecture",-12} {"kernels",-10} {"detail",-20} {"median s",9} {"min s",9}");
        foreach (var t in timings)
            Console.WriteLine($"{t.Name,-22} {t.Architecture,-12} {t.Kernels,-10} {t.Detail,-20} {t.MedianSeconds,9:0.0000} {t.MinSeconds,9:0.0000}");

        if (o.StringOrNull("json") is { } path)
        {
            var report = new Dictionary<string, object>
            {
                ["schema"] = 2,
                ["model_version"] = WeightsFile.ModelVersion,
                ["quick"] = quick,
                ["machine"] = new Dictionary<string, object>
                {
                    ["cores"] = Environment.ProcessorCount,
                    ["simd_floats"] = System.Numerics.Vector<float>.Count,
                    ["simd_accelerated"] = System.Numerics.Vector.IsHardwareAccelerated,
                    ["os"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                    ["dotnet"] = Environment.Version.ToString(),
                },
                ["timings"] = timings.Select(t => new Dictionary<string, object>
                {
                    ["name"] = t.Name, ["architecture"] = t.Architecture, ["detail"] = t.Detail,
                    ["median_seconds"] = t.MedianSeconds, ["min_seconds"] = t.MinSeconds, ["runs"] = t.Runs, ["kernels"] = t.Kernels,
                }).ToList(),
            };
            File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"📝 {path}");
        }
        return 0;
    }

    /// <summary><c>diagnose</c>: the noise check for one image, from the noisy image alone.</summary>
    private static int Diagnose(Options o)
    {
        var planes = DatasetLoader.Planes(o.Required("in"));
        if (planes.Count > 1)
        {
            // A stack: every plane is checked (brick 41; page 0 alone used to stand in for the whole file).
            var flagged = new List<int>();
            int judged = 0;
            for (int k = 0; k < planes.Count; k++)
            {
                try { if (NoiseDiagnostics.Diagnose(planes[k]).Correlated) flagged.Add(k); judged++; }
                catch (ArgumentException) { }   // too few flat tiles to judge
            }
            Console.WriteLine($"{Path.GetFileName(o.Required("in"))}: {planes.Count} planes of {planes[0].Width}×{planes[0].Height}; " +
                              $"{judged} judged, {flagged.Count} with noise correlated between pixels" +
                              (flagged.Count == 0 ? "" : $" (planes {string.Join(", ", flagged.Take(20))}{(flagged.Count > 20 ? ", …" : "")})"));
            return 0;
        }
        var image = planes[0];
        var d = NoiseDiagnostics.Diagnose(image);
        Console.WriteLine($"{Path.GetFileName(o.Required("in"))}: {image.Width}×{image.Height}, median over {d.TilesUsed} flat {NoiseDiagnostics.TileSize}px tiles");
        Console.WriteLine($"noise correlation: right 1 {d.Right1:0.000}  down 1 {d.Down1:0.000}  diagonal {d.Diagonal1:0.000}  right 2 {d.Right2:0.000}  down 2 {d.Down2:0.000}");
        Console.WriteLine((d.Correlated ? "⚠️ " : "✓ ") + d);
        return 0;
    }

    /// <summary>
    /// <c>fmd-diagnose</c>: on every test_mix image, the blind check (noisy image only) must reach the same verdict as
    /// the ground-truth measurement, using the same threshold on the true one-pixel correlation. Exit 1 if any image
    /// disagrees (bricks.sh gate 14).
    /// </summary>
    private static int FmdDiagnose(Options o)
    {
        string testMix = Path.Combine(o.Required("root"), "test_mix");
        int? validationSeed = o.Has("validation-seed") ? o.Int("validation-seed", 0) : null;
        string scope = validationSeed is { } vs ? $"validation (seed {vs})" : "test_mix";
        int disagreements = 0;
        var samples = new List<(double Estimate, bool Correlated)>();
        foreach (string level in o.String("levels", "raw").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var plan = validationSeed is { } seed
                ? FmdBenchmark.ValidationPlan(o.Required("root"), seed, [level])
                : FmdDataset.PlanTestMix(testMix, level);
            var truth = FmdBenchmark.MeasureNoise(plan).ToDictionary(s => s.Name);
            var rows = new List<(string Modality, bool Truth, bool Blind, double Estimate)>();
            foreach (var p in plan)
            {
                var d = NoiseDiagnostics.Diagnose(ImageLoader.Load(p.NoisyPath));
                var t = truth[p.Name];
                bool truthCorrelated = Math.Max(t.Right1, t.Down1) >= NoiseDiagnostics.CorrelatedThreshold;
                rows.Add((FmdBenchmark.ModalityOf(p.GroupId), truthCorrelated, d.Correlated, Math.Max(d.Right1, d.Down1)));
                samples.Add((Math.Max(d.Right1, d.Down1), truthCorrelated));
                if (truthCorrelated != d.Correlated)
                {
                    disagreements++;
                    Console.WriteLine($"✗ {p.Name}: blind estimate {Math.Max(d.Right1, d.Down1):0.000}, truth {Math.Max(t.Right1, t.Down1):0.000}");
                }
            }
            Console.WriteLine($"{scope} level {level}: correlated-noise verdicts, blind (noisy image only) vs ground truth");
            Console.WriteLine($"{"modality",-10} {"n",3}  {"truth",5} {"blind",5}  {"blind estimate min-max",22}");
            foreach (var g in rows.GroupBy(r => r.Modality).OrderBy(g => g.Key, StringComparer.Ordinal))
                Console.WriteLine($"{g.Key,-10} {g.Count(),3}  {g.Count(r => r.Truth),5} {g.Count(r => r.Blind),5}  {$"{g.Min(r => r.Estimate):0.000}-{g.Max(r => r.Estimate):0.000}",22}");
        }
        Console.WriteLine(disagreements == 0 ? "✓ every verdict agrees with the ground truth" : $"✗ {disagreements} verdicts disagree");
        if (o.Has("choose-threshold"))   // Q19's guard: the routing threshold these images support (validation only)
        {
            var (threshold, reason) = NoiseDiagnostics.ChooseThreshold(samples);
            Console.WriteLine($"routing threshold: {threshold:0.000} ({reason})");
            return 0;
        }
        return disagreements == 0 ? 0 : 1;
    }

    /// <summary>
    /// <c>fmd-route</c>: on the FOVs a training run with <c>--seed</c> held out for validation (all five noise levels, one
    /// frame each), scores model A and model B on every image, measures its noise σ, and chooses the threshold that
    /// maximizes the mean PSNR of routing (σ above → A, otherwise → B). test_mix is never touched. Deterministic, so the
    /// output can be recorded and re-checked (quality pass Q10, gate 27).
    /// </summary>
    private static int FmdRoute(Options o)
    {
        var a = WeightsFile.Load(o.Required("weights")).Net;
        var b = WeightsFile.Load(o.Required("weights-clean")).Net;
        var plan = FmdBenchmark.ValidationPlan(o.Required("root"), o.Int("seed", 2026), FmdDataset.Levels);
        var sigmas = plan.Select(p => NoiseLevelRouter.Sigma(ImageLoader.Load(p.NoisyPath))).ToList();
        var scoresA = FmdBenchmark.EvaluatePlan(_ => a, plan);
        var scoresB = FmdBenchmark.EvaluatePlan(_ => b, plan);
        var images = plan.Select((_, i) => (sigmas[i], scoresA[i].DoraPsnr, scoresB[i].DoraPsnr)).ToList();
        var (threshold, mean) = NoiseLevelRouter.ChooseThreshold(images);

        Console.WriteLine($"validation FOVs held out by seed {o.Int("seed", 2026)}: {plan.Select(p => p.GroupId).Distinct().Count()} FOVs × 5 levels = {plan.Count} images");
        Console.WriteLine($"{"level",-6} {"n",3}  {"sigma median",12}  {"PSNR A",7} {"PSNR B",7} {"routed",7}  {"to B",4}");
        foreach (string level in FmdDataset.Levels)
        {
            var idx = plan.Select((p, i) => (p, i)).Where(t => t.p.Name.Split('/')[1] == level).Select(t => t.i).ToList();
            var s = idx.Select(i => sigmas[i]).OrderBy(v => v).ToList();
            double routedMean = idx.Average(i => sigmas[i] > threshold ? scoresA[i].DoraPsnr : scoresB[i].DoraPsnr);
            Console.WriteLine($"{level,-6} {idx.Count,3}  {s[s.Count / 2],12:0.00000}  {idx.Average(i => scoresA[i].DoraPsnr),7:0.00} {idx.Average(i => scoresB[i].DoraPsnr),7:0.00} {routedMean,7:0.00}  {idx.Count(i => sigmas[i] <= threshold),4}");
        }
        Console.WriteLine($"all    {plan.Count,3}  {"",12}  {scoresA.Average(x => x.DoraPsnr),7:0.00} {scoresB.Average(x => x.DoraPsnr),7:0.00} {mean,7:0.00}  {images.Count(i => i.Item1 <= threshold),4}");
        Console.WriteLine($"chosen noise threshold: {threshold:0.00000}");
        return 0;
    }

    /// <summary>
    /// <c>ext-eval</c>: scores a DORA model, or the saved outputs of another tool (CAREamics, a vendor's denoiser), on an
    /// external dataset's test images with CARE's protocol (<see cref="ExtBenchmark"/>), next to the noisy input and the
    /// bilateral filter. Every test image is listed, since the published test splits hold only 2–5 images. The output is
    /// deterministic, so docs/RESULTS.md records it verbatim and gate 31 re-runs it.
    /// </summary>
    private static int ExtEval(Options o)
    {
        string root = o.String("root", "data/ext");
        var dataset = ExtDatasets.Get(o.Required("dataset"));
        if (!dataset.HasGroundTruth) throw new ArgumentException($"{dataset.Name} has no ground truth to score against.");
        Func<string, GrayImage, GrayImage> method;
        string label;
        if (o.StringOrNull("weights") is { } weights)
        {
            var (net, info) = WeightsFile.Load(weights);
            // --weights-correlated: images the noise check calls correlated go to a second network (P3's routing, Q19).
            var correlated = o.StringOrNull("weights-correlated") is { } c ? WeightsFile.Load(c) : default;
            var denoiser = new FrameDenoiser(net, correlated.Net);
            method = (_, noisy) => denoiser.Denoise(noisy).Output;
            label = correlated.Net is null ? info.Provenance : $"{info.Provenance} | correlated images: {correlated.Info.Provenance}";
        }
        else if (o.StringOrNull("predictions") is { } dir)
        {
            method = (name, _) => ImageLoader.Load(Path.Combine(dir, name));
            label = o.String("label", "outputs in " + Path.GetFileName(Path.TrimEndingDirectorySeparator(dir)));
        }
        else throw new ArgumentException("ext-eval needs --weights or --predictions.");

        var (plan, scope) = ExtPlan(o, root, dataset);
        var scores = ExtBenchmark.Evaluate(plan, method);
        Console.WriteLine($"{dataset.Name} {scope}, {scores.Count} images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization");
        Console.WriteLine($"method: {label}");
        Console.WriteLine($"{"image",-12} {"noisy",-15} {"bilateral",-15} {"method",-15}");
        foreach (var s in scores)
            Console.WriteLine($"{s.Name,-12} {s.NoisyPsnr,6:0.00} / {s.NoisySsim:0.000}  {s.BilateralPsnr,6:0.00} / {s.BilateralSsim:0.000}  {s.MethodPsnr,6:0.00} / {s.MethodSsim:0.000}");
        Console.WriteLine($"{"mean",-12} {scores.Average(s => s.NoisyPsnr),6:0.00} / {scores.Average(s => s.NoisySsim):0.000}  " +
                          $"{scores.Average(s => s.BilateralPsnr),6:0.00} / {scores.Average(s => s.BilateralSsim):0.000}  " +
                          $"{scores.Average(s => s.MethodPsnr),6:0.00} / {scores.Average(s => s.MethodSsim):0.000}");
        if (o.Has("precise"))   // for margins that could sit within rounding of the 2-decimal mean (brick 38's reading)
            Console.WriteLine($"mean, unrounded: noisy {scores.Average(s => s.NoisyPsnr):0.0000} dB, bilateral {scores.Average(s => s.BilateralPsnr):0.0000} dB, " +
                              $"method {scores.Average(s => s.MethodPsnr):0.0000} dB");
        return 0;
    }

    /// <summary>
    /// The pairs <c>ext-eval</c> and <c>fidelity</c> score: the test split by default, or with <c>--split validation
    /// --seed S</c> the training images a run with that seed held out (quality passes choose settings there).
    /// </summary>
    private static (List<FmdDataset.PlannedPair> Plan, string Scope) ExtPlan(Options o, string root, ExtDataset dataset) =>
        o.String("split", "test") switch
        {
            "test" => (ExtDatasets.PlanPairs(root, dataset, ExtSplit.Test), "test"),
            "validation" => (ExtDatasets.ValidationPlan(root, dataset, o.Int("seed", 2026)), $"validation (seed {o.Int("seed", 2026)})"),
            var other => throw new ArgumentException($"Unknown --split '{other}' (use test or validation)."),
        };

    /// <summary>
    /// <c>seq-eval</c>: scores a block of a time-lapse against its high-SNR movie (<see cref="SequenceBenchmark"/>), for the
    /// noisy input and, with <c>--weights</c>, a model applied as a standalone stack of that block would be: each frame with
    /// its neighbours reflected at the block's ends, never read from outside it. Deterministic, so brick 40's recorded
    /// blocks are re-run by gate 40.
    /// </summary>
    private static int SeqEval(Options o)
    {
        string stackPath = o.Required("stack"), referencePath = o.Required("reference");
        var block = FrameRange.Parse(o.Required("frames"));
        using var noisy = ImageStack.Open(stackPath);
        using var reference = ImageStack.Open(referencePath);
        var methods = new List<Func<GrayImage, GrayImage[], GrayImage>> { (raw, _) => raw };
        var names = new List<string> { "noisy" };
        string label = "";
        if (o.StringOrNull("weights") is { } weights)
        {
            var (net, info) = WeightsFile.Load(weights);
            var denoiser = new FrameDenoiser(net);
            methods.Add(net.Config.WindowFrames == 1
                ? (raw, _) => denoiser.Denoise(raw).Output
                : (raw, neighbours) => denoiser.Denoise(raw, neighbours).Output);
            names.Add("method");
            label = o.String("label", info.Provenance);
        }
        StreamWriter? perFrame = o.StringOrNull("per-frame") is { } tsv ? new StreamWriter(tsv) : null;
        perFrame?.WriteLine("method\tframe\tpsnr\tssim\tdynamic_psnr\tsteady_psnr\tslope");
        var scores = SequenceBenchmark.Evaluate(noisy, reference, block, methods, (m, s) =>
            perFrame?.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{names[m]}\t{s.Frame}\t{s.Psnr:0.0000}\t{s.Ssim:0.0000}\t{s.DynamicPsnr:0.0000}\t{s.SteadyPsnr:0.0000}\t{s.Fidelity.Slope:0.0000}")));
        perFrame?.Dispose();

        Console.WriteLine($"🎞 frames {block} of {Path.GetFileName(stackPath)} against {Path.GetFileName(referencePath)}: {block.Count} frames, " +
                          "CARE's protocol per frame, the block a standalone sequence");
        if (label.Length > 0) Console.WriteLine($"method: {label}");
        Console.WriteLine("         PSNR      SSIM    dynamic   steady   slope   ratio-err  spots-found  spots-real");
        for (int m = 0; m < scores.Count; m++)
        {
            var s = scores[m];
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{names[m],-7} {s.Psnr,8:0.0000} {s.Ssim,8:0.0000} {s.DynamicPsnr,9:0.0000} {s.SteadyPsnr,8:0.0000} {s.Fidelity.Slope,7:0.000} {s.Fidelity.RatioError,9:0.000} {s.Fidelity.SpotsFound,11:0.000} {s.Fidelity.SpotsReal,11:0.000}"));
        }
        return 0;
    }

    /// <summary>
    /// <c>seq-diagnose</c>: brick 40's post-hoc diagnostics (<see cref="SequenceDiagnostics"/>) for the noisy input and every
    /// model given, in one pass over the block. Deterministic, so docs/RESULTS.md records it verbatim.
    /// </summary>
    private static int SeqDiagnose(Options o)
    {
        string stackPath = o.Required("stack"), referencePath = o.Required("reference");
        var block = FrameRange.Parse(o.Required("frames"));
        var weights = o.Required("weights").Split(',');
        var labels = o.String("labels", string.Join(",", weights.Select(Path.GetFileNameWithoutExtension))).Split(',');
        if (labels.Length != weights.Length) throw new ArgumentException("--labels needs one label per --weights file.");
        using var noisy = ImageStack.Open(stackPath);
        using var reference = ImageStack.Open(referencePath);
        var methods = new List<Func<GrayImage, GrayImage[], GrayImage>> { (raw, _) => raw };
        foreach (var path in weights)
        {
            var denoiser = new FrameDenoiser(WeightsFile.Load(path).Net);
            methods.Add(denoiser.Net.Config.WindowFrames == 1
                ? (raw, _) => denoiser.Denoise(raw).Output
                : (raw, neighbours) => denoiser.Denoise(raw, neighbours).Output);
        }
        var (audit, readouts) = SequenceDiagnostics.Run(noisy, reference, block, methods);
        Console.WriteLine($"🔬 frames {block} of {Path.GetFileName(stackPath)} against {Path.GetFileName(referencePath)} (post hoc, after brick 40)");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"mask audit, low- vs high-SNR signed change: dynamic {audit.DynamicInMask:0.000} in its mask, {audit.DynamicAll:0.000} over all pixels; " +
            $"first difference {audit.SteadyInMask:0.000} in its mask, {audit.SteadyAll:0.000} over all pixels"));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"shared change, noisy input vs reference, raw units, all pixels: midpoint {audit.RawDynamic:0.0000} ± {audit.RawDynamicSe:0.0000}, " +
            $"first difference {audit.RawSteady:0.0000} ± {audit.RawSteadySe:0.0000} (jackknife over {SequenceDiagnostics.Chunks} frame chunks)"));
        Console.WriteLine("          change  mean-slope  mean-ratio-err  tile-slope  tile-ratio-err  transients-kept (se)  motion-kept (se)  gain-jitter-dB  flicker  trace-corr");
        string[] names = ["noisy", .. labels];
        for (int j = 0; j < readouts.Count; j++)
        {
            var r = readouts[j];
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{names[j],-8} {r.Change,7:0.0000} {r.MeanFidelity.Slope,11:0.000} {r.MeanFidelity.RatioError,15:0.000} {r.TileSlope,11:0.000} {r.TileRatioError,15:0.000} {r.TransientKept,13:0.000} ({r.TransientKeptSe:0.000}) {r.MotionKept,9:0.000} ({r.MotionKeptSe:0.000}) {r.GainJitterDb,15:0.000} {r.Flicker,8:0.000} {r.TraceCorrelation,11:0.000}"));
        }
        return 0;
    }

    /// <summary>
    /// <c>fidelity</c>: the measurements of brick 32 (<see cref="Fidelity"/>) for the noisy input, the bilateral filter and a
    /// model or another tool's saved outputs, pooled over an external set's test images or FMD's test_mix at one noise
    /// level. Deterministic, so docs/RESULTS.md records it verbatim and gate 32 re-runs it.
    /// </summary>
    private static int FidelityCommand(Options o)
    {
        List<(string Name, string NoisyPath, string CleanPath)> plan;
        string scope;
        if (o.StringOrNull("fmd") is { } fmdRoot)
        {
            string level = o.String("level", "raw");
            plan = FmdDataset.PlanTestMix(Path.Combine(fmdRoot, "test_mix"), level).Select(p => (p.Name, p.NoisyPath, p.CleanPath)).ToList();
            scope = $"FMD test_mix level {level}";
        }
        else
        {
            var dataset = ExtDatasets.Get(o.Required("dataset"));
            var (pairs, split) = ExtPlan(o, o.String("root", "data/ext"), dataset);
            plan = pairs.Select(p => (Path.GetFileName(p.NoisyPath), p.NoisyPath, p.CleanPath)).ToList();
            scope = $"{dataset.Name} {split}";
        }

        Func<string, GrayImage, GrayImage> method;
        string label;
        if (o.StringOrNull("weights") is { } weights)
        {
            var (net, info) = WeightsFile.Load(weights);
            // --weights-correlated: images the noise check calls correlated go to a second network (P3's routing, Q19).
            var correlated = o.StringOrNull("weights-correlated") is { } c ? WeightsFile.Load(c) : default;
            var denoiser = new FrameDenoiser(net, correlated.Net);
            method = (_, noisy) => denoiser.Denoise(noisy).Output;
            label = correlated.Net is null ? info.Provenance : $"{info.Provenance} | correlated images: {correlated.Info.Provenance}";
        }
        else if (o.StringOrNull("predictions") is { } dir)
        {
            method = (name, _) => ImageLoader.Load(Path.Combine(dir, name));
            label = o.String("label", "outputs in " + Path.GetFileName(Path.TrimEndingDirectorySeparator(dir)));
        }
        else throw new ArgumentException("fidelity needs --weights or --predictions.");

        var rows = new List<FidelityScore>[3];
        for (int r = 0; r < 3; r++) rows[r] = [];
        foreach (var (name, noisyPath, cleanPath) in plan)
        {
            var noisy = ImageLoader.Load(noisyPath);
            var clean = ImageLoader.Load(cleanPath);
            var input = Normalization.Apply(noisy, Normalization.FromPercentiles(noisy.Pixels));
            rows[0].Add(Fidelity.Measure(noisy, clean));
            rows[1].Add(Fidelity.Measure(BilateralBaseline.FilterAuto(input), clean));
            rows[2].Add(Fidelity.Measure(method(name, noisy), clean));
        }
        var pooled = rows.Select(Fidelity.Pool).ToArray();
        Console.WriteLine($"{scope}, {plan.Count} images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)");
        Console.WriteLine($"method: {label}");
        Console.WriteLine($"{"",-10} {"slope",6} {"ratio err",9} {"spots found",11} {"spots real",10}");
        string[] names = ["noisy", "bilateral", "method"];
        for (int r = 0; r < 3; r++)
        {
            var p = pooled[r];
            Console.WriteLine($"{names[r],-10} {p.Slope,6:0.00} {p.RatioError * 100,8:0.0}% {p.SpotsFound * 100,10:0.0}% {p.SpotsReal * 100,9:0.0}%");
        }
        Console.WriteLine($"over {pooled[2].ForegroundTiles} foreground tiles and {pooled[2].ReferenceSpots} reference spots");
        return 0;
    }

    /// <summary>
    /// <c>ext-check</c>: for every external dataset that <c>tools/fetch_ext.sh</c> has downloaded, checks that the train
    /// and test splits hold the published numbers of images, that every noisy image has a clean partner of the same size,
    /// and that every file decodes. Also reports each set's brightness ratio, clean over noisy: the high-SNR images were
    /// taken with more light, so they are brighter, which is why the benchmark scores outputs after a least-squares
    /// brightness match. Exit 1 on any inconsistency; datasets not downloaded are listed, not failed.
    /// </summary>
    private static int ExtCheck(Options o)
    {
        string root = o.String("root", "data/ext");
        int problems = 0, missing = 0;
        Console.WriteLine($"external datasets in {root}");
        Console.WriteLine($"{"name",-20} {"train",7} {"test",7}  {"size",-11} {"noisy max",9}  clean/noisy brightness");
        foreach (var d in ExtDatasets.All)
        {
            if (!ExtDatasets.IsPresent(root, d))
            {
                Console.WriteLine($"{d.Name,-20} not downloaded (tools/fetch_ext.sh {d.Name})");
                missing++;
                continue;
            }
            var train = ExtDatasets.PlanNoisy(root, d, ExtSplit.Train);
            var test = ExtDatasets.PlanNoisy(root, d, ExtSplit.Test);
            var sizes = new SortedSet<string>(StringComparer.Ordinal);
            var ratios = new List<double>();
            double noisyMax = 0;
            void Note(string problem) { Console.WriteLine($"  ✗ {d.Name}: {problem}"); problems++; }
            if (train.Count != d.TrainCount) Note($"{train.Count} training images, expected {d.TrainCount}");
            if (test.Count != d.TestCount) Note($"{test.Count} test images, expected {d.TestCount}");
            if (d.HasGroundTruth)
            {
                foreach (var split in new[] { ExtSplit.Train, ExtSplit.Test })
                    foreach (var p in ExtDatasets.PlanPairs(root, d, split))
                    {
                        var noisy = ImageLoader.Load(p.NoisyPath);
                        var clean = ImageLoader.Load(p.CleanPath);
                        if ((noisy.Width, noisy.Height) != (clean.Width, clean.Height)) Note($"{p.Name}: noisy and clean differ in size");
                        sizes.Add($"{noisy.Width}×{noisy.Height}");
                        noisyMax = Math.Max(noisyMax, noisy.Pixels.Max());
                        double noisyMean = noisy.Pixels.Average(v => (double)v);
                        if (noisyMean > 0) ratios.Add(clean.Pixels.Average(v => (double)v) / noisyMean);
                    }
            }
            else
                foreach (var path in train.Concat(test))
                {
                    var noisy = ImageLoader.Load(path);
                    sizes.Add($"{noisy.Width}×{noisy.Height}");
                    noisyMax = Math.Max(noisyMax, noisy.Pixels.Max());
                }
            ratios.Sort();
            string brightness = ratios.Count == 0 ? "(no ground truth)" : $"{ratios[ratios.Count / 2]:0.00}× (median over {ratios.Count} pairs)";
            Console.WriteLine($"{d.Name,-20} {$"{train.Count}/{d.TrainCount}",7} {$"{test.Count}/{d.TestCount}",7}  {string.Join(",", sizes),-11} {noisyMax,9:0.##}  {brightness}");
        }
        if (problems > 0) { Console.WriteLine($"✗ {problems} problem(s)"); return 1; }
        Console.WriteLine(missing == 0 ? $"✓ all {ExtDatasets.All.Count} datasets present and consistent"
                                       : $"✓ the {ExtDatasets.All.Count - missing} present datasets are consistent; {missing} not downloaded");
        return 0;
    }

    /// <summary>
    /// <c>stack-check</c>: for every stack dataset that is downloaded, opens every file's header and checks it against
    /// the published counts and declared axes (<see cref="ExtStacks"/>), checks each clean partner has the same layout,
    /// and decodes the first and last plane of the first file. Exit 1 on any disagreement; datasets not downloaded are
    /// listed, not failed.
    /// </summary>
    private static int StackCheck(Options o)
    {
        string root = o.String("root", "data/ext");
        string? only = o.StringOrNull("dataset");
        var datasets = only is null ? ExtStacks.All : [ExtStacks.Get(only)];
        int problems = 0, missing = 0;
        Console.WriteLine($"stack datasets in {root}");
        Console.WriteLine($"{"name",-26} {"files",11} {"planes",15}  {"axes",-5} {"plane size",-12} {"format",-8} first plane (min / mean / max)");
        foreach (var d in datasets)
        {
            if (!ExtStacks.IsPresent(root, d))
            {
                Console.WriteLine($"{d.Name,-26} not downloaded (tools/fetch_ext.sh {d.Name})");
                missing++;
                continue;
            }
            try
            {
                var summaries = ExtStacks.Inspect(root, d);
                var first = ExtStacks.Plan(root, d)[0];
                for (int g = 0; g < d.Groups.Count; g++)
                {
                    var (group, sum) = (d.Groups[g], summaries[g]);
                    string stats = "";
                    if (g == 0)
                    {
                        using var stack = ImageStack.Open(first.Noisy, group.Axes);
                        var plane = stack.ReadPlane(0);
                        stack.ReadPlane(stack.PlaneCount - 1);
                        if (first.Clean is not null)
                        {
                            using var clean = ImageStack.Open(first.Clean, group.Axes);
                            clean.ReadPlane(0);
                            clean.ReadPlane(clean.PlaneCount - 1);
                        }
                        stats = $"{plane.Pixels.Min():0.##} / {plane.Pixels.Average(v => (double)v):0.##} / {plane.Pixels.Max():0.##}";
                    }
                    string planes = group.Planes is long p ? $"{sum.Planes}/{p}" : $"{sum.Planes}";
                    string label = d.Groups.Count == 1 ? d.Name : $"{d.Name}[{g}]";
                    string sampled = sum.HeadersOpened < sum.Files ? $"  (headers of {sum.HeadersOpened} spread-out files; all {sum.Files} the same size)" : "";
                    Console.WriteLine($"{label,-26} {$"{sum.Files}/{group.Files}",11} {planes,15}  {group.Axes,-5} {string.Join(",", sum.PlaneSizes),-12} " +
                                      $"{string.Join(",", sum.Formats),-8} {stats}{sampled}");
                    if (sum.Files != group.Files) { Console.WriteLine($"  ✗ {label}: {sum.Files} files, expected {group.Files}"); problems++; }
                    if (group.Planes is long expected && sum.Planes != expected)
                    {
                        Console.WriteLine($"  ✗ {label}: {sum.Planes} planes, expected {expected}");
                        problems++;
                    }
                }
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException)
            {
                if (d.RefusedWith is { } expected && e.Message.Contains(expected, StringComparison.Ordinal))
                {
                    Console.WriteLine($"{d.Name,-26} refused, as recorded: {e.Message}");
                    continue;
                }
                Console.WriteLine($"  ✗ {d.Name}: {e.Message}");
                problems++;
                continue;
            }
            if (d.RefusedWith is { } refusal)
            {
                Console.WriteLine($"  ✗ {d.Name}: read without error, but it is recorded as refused ('{refusal}')");
                problems++;
            }
        }
        if (problems > 0) { Console.WriteLine($"✗ {problems} problem(s)"); return 1; }
        int present = datasets.Count - missing;
        Console.WriteLine(missing == 0 ? $"✓ all {datasets.Count} stack datasets present and consistent"
                                       : $"✓ the {present} present stack dataset(s) are consistent; {missing} not downloaded");
        return 0;
    }

    /// <summary>
    /// <c>stack-info</c>: what a TIFF or MRC stack holds, from its header and metadata: format, axes, shape, plane size
    /// and where the shape came from, plus the first plane's range. <c>--axes</c> checks a declaration against the file.
    /// </summary>
    private static int StackInfo(Options o)
    {
        string path = o.Required("in");
        using var stack = ImageStack.Open(path, o.StringOrNull("axes"));
        var plane = stack.ReadPlane(0);
        Console.WriteLine($"{Path.GetFileName(path)}: {stack.Format}, {stack.Layout.FullAxes}, {stack.Layout} " +
                          $"({stack.PlaneCount} planes of {stack.Width}×{stack.Height}; shape from {stack.Layout.Source})");
        Console.WriteLine($"first plane: min {plane.Pixels.Min():0.###}, mean {plane.Pixels.Average(v => (double)v):0.###}, max {plane.Pixels.Max():0.###}");
        return 0;
    }

    /// <summary>
    /// <c>bench --roofline</c> (systems brick S1): the machine's measured peaks, every convolution kernel shape DORA runs
    /// set against them, and a stack run's time per stage, so speed work targets the real limit.
    /// </summary>
    private static int BenchRoofline(Options o)
    {
        var peaks = Roofline.MeasurePeaks();
        Console.WriteLine($"machine: {peaks.Cores} threads, Vector<float> = {peaks.VectorBits} bits, AVX-512 {(System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated ? "yes" : "no")}");
        Console.WriteLine($"{"peak",-22} {"1 thread",10} {"all threads",12}");
        foreach (var c in peaks.Compute) Console.WriteLine($"{c.Name,-22} {c.OneThread,7:0.0} GF {c.AllThreads,9:0.0} GF");
        foreach (var b in peaks.Bandwidth) Console.WriteLine($"{b.Name,-22} {b.OneThread,7:0.0} GB {b.AllThreads,9:0.0} GB");
        var kernels = Roofline.MeasureKernels(peaks);
        Console.WriteLine($"{"kernel",-9} {"shape",-20} {"GFLOP/s",8} {"ms",8} {"MB",7} {"flop/B",7} {"of compute",10} {"of bandwidth",12}  bound");
        foreach (var k in kernels)
            Console.WriteLine($"{k.Kernel,-9} {k.Shape,-20} {k.Gflops,8:0.0} {k.Seconds * 1e3,8:0.00} {k.Megabytes,7:0.00} {k.Intensity,7:0.0} {k.ComputeShare,10:P0} {k.BandwidthShare,12:P0}  {k.Bound}");
        var stack = Roofline.MeasureStack(DenoisingNet.Create(NetConfig.Default, seed: 5));
        Console.WriteLine($"stack run, per 512² page ({stack.Pages} pages): decode {stack.DecodeSeconds * 1e3:0.0} ms, inference {stack.InferSeconds * 1e3:0.0} ms, " +
                          $"normalize + map back {stack.OtherProcessingSeconds * 1e3:0.0} ms, encode + write {stack.EncodeWriteSeconds * 1e3:0.0} ms " +
                          $"(inference {stack.InferSeconds / stack.TotalSeconds:P0})");
        if (o.StringOrNull("json") is { } path)
        {
            var report = new { schema = 1, peaks, kernels = kernels.Select(k => new { k.Kernel, k.Shape, k.Gflop, k.Seconds, k.Gflops, k.Megabytes, k.Intensity, k.ComputeShare, k.BandwidthShare, k.Bound }), stack };
            File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"📝 {path}");
        }
        return 0;
    }

    /// <summary><c>info</c>: prints a weights file's version, architecture, size, save time and provenance.</summary>
    private static int Info(Options o)
    {
        if (o.Has("default-condition"))
        {
            Console.WriteLine(Recipes.DefaultCondition);
            return 0;
        }
        if (o.Has("default-n2v-route"))
        {
            Console.WriteLine(Recipes.DescribeN2vRoute());
            return 0;
        }
        if (o.Has("default-timelapse"))
        {
            Console.WriteLine(Recipes.DefaultTimeLapse);
            return 0;
        }
        if (o.Has("default-n2v"))
        {
            var n2v = Recipes.DefaultN2v;
            Console.WriteLine($"{n2v.Arch} select={n2v.Select} patience={n2v.Patience} keep={n2v.Keep}");
            return 0;
        }
        if (o.Has("default-arch"))
        {
            Console.WriteLine(NetConfig.Default.Architecture == NetArchitecture.UNet ? "unet" : "dilated");
            return 0;
        }
        var (net, info) = WeightsFile.Load(o.Required("weights"));
        Console.WriteLine($"model version : {info.ModelVersion}");
        Console.WriteLine($"architecture  : {info.Config}");
        Console.WriteLine($"parameters    : {net.ParameterCount:N0}");
        Console.WriteLine($"saved (UTC)   : {info.SavedUtc:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"provenance    : {info.Provenance}");
        Console.WriteLine($"content hash  : {WeightsFile.ContentHash(net)}");
        return 0;
    }

    /// <summary>Shortens <paramref name="s"/> to <paramref name="n"/> characters, keeping the end (file names differ at the end).</summary>
    private static string Truncate(string s, int n) => s.Length <= n ? s : "…" + s[^(n - 1)..];

    /// <summary>
    /// Minimal "--key value" parser with no dependencies. Every option takes one value, or none when it is a flag
    /// (the next token is another option, or there is none). Unknown keys are refused (<see cref="RejectUnknown"/>).
    /// </summary>
    private sealed class Options
    {
        private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);

        public static Options Parse(IEnumerable<string> args)
        {
            // "--key value", or a bare "--flag" when the next token is another option or there is none (--vst, --per-image).
            var o = new Options();
            var tokens = args.ToList();
            for (int i = 0; i < tokens.Count; i++)
            {
                if (!tokens[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Unexpected argument '{tokens[i]}'.");
                string key = tokens[i][2..];
                bool hasValue = i + 1 < tokens.Count && !tokens[i + 1].StartsWith("--", StringComparison.Ordinal);
                o._values[key] = hasValue ? tokens[++i] : "true";
            }
            return o;
        }

        /// <summary>
        /// Every option some command reads (brick 41): anything else is refused before any work starts, so a typo such as
        /// <c>--epoch 60</c> can't silently run with the default instead. Gate 6 checks the list against the names this
        /// file actually reads.
        /// </summary>
        public static readonly IReadOnlySet<string> Known = new HashSet<string>(StringComparer.Ordinal)
        {
        "arch", "augment", "axes", "baseline", "batch", "choose-threshold", "condition", "count", "data",
        "dataset", "default-arch", "default-condition", "default-n2v", "default-n2v-route", "default-timelapse",
        "depth", "describe", "drift", "epochs", "ext", "features", "fmd", "fmd-categories", "fmd-frames",
        "fmd-levels", "footprint", "frame-block", "frame-range", "frames", "height", "in", "init", "journey",
        "json", "keep", "label", "labels", "level", "levels", "lr", "mask-ratio", "mode", "noise-threshold",
        "offset", "only", "out", "out-correlated", "patch", "patience", "peak", "per-frame", "per-image",
        "precise", "predictions", "quick", "reference", "report", "roofline", "root", "route", "save-last", "seed",
        "select", "sequence", "sigma-scale", "size", "split", "stack", "steps", "struct", "subset", "tile",
        "validation-range", "validation-seed", "vst", "weights", "weights-clean", "weights-correlated", "width",
        };

        /// <summary>Throws if an option isn't one any command reads, naming the closest known one.</summary>
        public void RejectUnknown()
        {
            foreach (string key in _values.Keys)
            {
                if (Known.Contains(key)) continue;
                string? closest = Known.MinBy(k => Distance(k, key));
                string hint = closest is not null && Distance(closest, key) <= 2 ? $" Did you mean --{closest}?" : "";
                throw new ArgumentException($"Unknown option --{key}.{hint} (dora-bio help lists them.)");
            }
        }

        /// <summary>Levenshtein distance, for the "did you mean" hint.</summary>
        private static int Distance(string a, string b)
        {
            var d = new int[a.Length + 1, b.Length + 1];
            for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
            for (int j = 0; j <= b.Length; j++) d[0, j] = j;
            for (int i = 1; i <= a.Length; i++)
                for (int j = 1; j <= b.Length; j++)
                    d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            return d[a.Length, b.Length];
        }

        public bool Has(string key) => _values.ContainsKey(key);
        public string Required(string key) => _values.TryGetValue(key, out var v) ? v : throw new ArgumentException($"--{key} is required.");
        public string? StringOrNull(string key) => _values.GetValueOrDefault(key);
        public string String(string key, string fallback) => _values.GetValueOrDefault(key, fallback);

        public int Int(string key, int fallback) =>
            _values.TryGetValue(key, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

        public double Double(string key, double fallback) =>
            _values.TryGetValue(key, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : fallback;

        /// <summary>The option's integer value, or null when it wasn't given (so the pipeline's default applies).</summary>
        public int? IntOrNull(string key) => _values.TryGetValue(key, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : null;

        /// <summary>The option's numeric value, or null when it wasn't given.</summary>
        public double? DoubleOrNull(string key) =>
            _values.TryGetValue(key, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : null;
    }
}

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
          synth    --out DIR [--count 8] [--size 256 | --width W --height H] [--seed 1] [--peak 25]
                   Write a synthetic dataset: DIR/noisy/*.tif (uint16) and DIR/clean/*.tif (float32).
          train    --data DIR --out WEIGHTS.bin [--mode n2v|supervised] [--epochs 30] [--steps 50]
                   [--batch 16] [--patch 64] [--lr 0.001] [--features 24] [--journey FILE.json] [--seed 1234]
                   [--init WEIGHTS.bin]   (fine-tune existing weights instead of starting fresh)
                   [--describe TEXT]      (what the data is; stored in the weights' provenance)
                   [--augment dihedral|v1] (v1 = v1's ±15° rotation + 0.85-1.15 scale, nearest neighbour; for comparison)
                   FMD folders: [--fmd-frames 4] [--fmd-levels raw,avg2,avg4,avg8,avg16] [--fmd-categories A,B]
                   (FOV 19, FMD's test FOV, is always excluded from training)
          denoise  --weights WEIGHTS.bin --in IMAGE --out OUT.tif [--baseline BASELINE.tif] [--tile 128]
                   Output is float32 TIFF in the input's original intensity units.
          eval     --weights WEIGHTS.bin --data DIR
                   Per-image PSNR/SSIM of noisy, bilateral and DORA against clean (needs clean images).
          info     --weights WEIGHTS.bin
          fmd-check --root data/fmd
                   Verify a downloaded FMD dataset: structure, test_mix, and no train/test leakage.
          fmd-eval  --weights WEIGHTS.bin --root data/fmd [--levels raw,avg2,...] [--per-image]
                   PSNR/SSIM (8-bit scale) on FMD's held-out test_mix: noisy vs bilateral vs DORA, per microscope type.
          fmd-noise --root data/fmd [--levels raw,avg2,...]
                   How correlated test_mix's real noise is between neighbouring pixels (Noise2Void assumes 0).
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
            return args[0] switch
            {
                "synth" => Synth(options),
                "train" => Train(options),
                "denoise" => Denoise(options),
                "eval" => Eval(options),
                "info" => Info(options),
                "fmd-check" => FmdCheck(options),
                "fmd-eval" => FmdEval(options),
                "fmd-noise" => FmdNoise(options),
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

        for (int i = 0; i < count; i++)
        {
            var frame = SyntheticFluorescence.Generate(width, height, seed + 7919 * i, synthOptions);
            string name = $"synthetic_{i:D3}.tif";
            File.WriteAllBytes(Path.Combine(outDir, "noisy", name), TiffCodec.Encode(frame.Noisy, TiffSampleType.UInt16));
            File.WriteAllBytes(Path.Combine(outDir, "clean", name), TiffCodec.Encode(frame.Clean, TiffSampleType.Float32));
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
        string dataDir = o.Required("data");
        var dataset = DatasetLoader.LoadDirectory(dataDir, FmdOptionsFrom(o));
        var mode = o.String("mode", dataset.All(d => d.HasClean) ? "supervised" : "n2v") switch
        {
            "n2v" or "noise2void" => TrainingMode.Noise2Void,
            "supervised" => TrainingMode.Supervised,
            var m => throw new ArgumentException($"Unknown mode '{m}' (use n2v or supervised)."),
        };

        var config = new TrainerConfig
        {
            Mode = mode,
            MaxEpochs = o.Int("epochs", 30),
            StepsPerEpoch = o.Int("steps", 50),
            BatchSize = o.Int("batch", 16),
            Augmentation = o.String("augment", "dihedral") switch
            {
                "dihedral" => AugmentationMode.Dihedral,
                "v1" => AugmentationMode.V1RotateScale,
                var a => throw new ArgumentException($"Unknown --augment '{a}' (use dihedral or v1)."),
            },
            PatchSize = o.Int("patch", 64),
            LearningRate = (float)o.Double("lr", 1e-3),
            Seed = o.Int("seed", 1234),
        };

        DenoisingNet net;
        if (o.Has("init"))
        {
            net = WeightsFile.Load(o.Required("init")).Net;
            Console.WriteLine($"🔁 fine-tuning {o.Required("init")}");
        }
        else
        {
            var defaults = NetConfig.Default;
            net = DenoisingNet.Create(new NetConfig(o.Int("features", defaults.Features), defaults.Dilations), config.Seed);
        }

        int withClean = dataset.Count(d => d.HasClean);
        string cleanUse = withClean == 0 ? "no clean images"
            : mode == TrainingMode.Noise2Void ? $"{withClean} clean images used only to score validation"
            : $"{withClean} clean targets";
        string description = o.StringOrNull("describe")
            ?? $"{dataset.Count} images from '{Path.GetFileName(Path.GetFullPath(dataDir))}' ({cleanUse})";
        Console.WriteLine($"🧬 {net.Config}, {net.ParameterCount:N0} parameters");
        Console.WriteLine($"📚 {description}; mode {mode}; {config.MaxEpochs} epochs × {config.StepsPerEpoch} steps × batch {config.BatchSize} @ {config.PatchSize}px");

        var clock = Stopwatch.StartNew();
        var result = TrainingJob.Run(net, dataset, config,
            new TrainingJobOptions { DatasetDescription = description, WeightsPath = o.Required("out"), JournalPath = o.StringOrNull("journey") },
            report => Console.WriteLine(FormatEpoch(report)));

        Console.WriteLine($"✅ stopped: {result.StopReason} after {result.History.Count} epochs in {clock.Elapsed:mm\\:ss}");
        if (result.Best is { } best)
        {
            string psnr = best.ValPsnr is { } p ? $", val PSNR {p:0.00} dB vs noisy {result.NoisyBaselinePsnr:0.00} dB" : "";
            Console.WriteLine($"💾 best epoch {best.Epoch} (val loss {best.ValLoss:G4}{psnr}) saved to {o.Required("out")}");
        }
        return 0;
    }

    /// <summary>
    /// <c>denoise</c>: normalize, denoise with tiled inference, then map back to the input's own intensity units and
    /// write float32 TIFF. Writing float rather than 16-bit keeps sub-count precision, which downstream intensity
    /// measurements need. <c>--baseline</c> also writes the bilateral result for comparison.
    /// </summary>
    private static int Denoise(Options o)
    {
        var (net, info) = WeightsFile.Load(o.Required("weights"));
        var raw = ImageLoader.Load(o.Required("in"));
        var map = Normalization.FromPercentiles(raw.Pixels);
        var input = Normalization.Apply(raw, map);

        var clock = Stopwatch.StartNew();
        var denoised = TiledInference.Denoise(net, input, o.Int("tile", TiledInference.DefaultTileCore));
        double seconds = clock.Elapsed.TotalSeconds;

        File.WriteAllBytes(o.Required("out"), TiffCodec.Encode(Normalization.Invert(denoised, map), TiffSampleType.Float32));
        Console.WriteLine($"🧬 model: {info.Provenance}");
        Console.WriteLine($"✨ denoised {raw.Width}×{raw.Height} in {seconds:0.00}s → {o.Required("out")}");
        Console.WriteLine($"   estimated noise σ (normalized): {Metrics.EstimateNoiseSigma(input):0.0000} → {Metrics.EstimateNoiseSigma(denoised):0.0000}");

        if (o.StringOrNull("baseline") is { } baselinePath)
        {
            var baseline = BilateralBaseline.FilterAuto(input);
            File.WriteAllBytes(baselinePath, TiffCodec.Encode(Normalization.Invert(baseline, map), TiffSampleType.Float32));
            Console.WriteLine($"   bilateral baseline σ: {Metrics.EstimateNoiseSigma(baseline):0.0000} → {baselinePath}");
        }
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
        var (net, _) = WeightsFile.Load(o.Required("weights"));
        string testMix = Path.Combine(o.Required("root"), "test_mix");
        var levels = o.String("levels", "raw").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (string level in levels)
        {
            var scores = FmdBenchmark.Evaluate(net, testMix, level);
            Console.WriteLine($"test_mix level {level}: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale");
            Console.WriteLine($"{"modality",-10} {"n",3}  {"noisy",-15} {"bilateral",-15} {"DORA",-15}");
            foreach (var (group, count, m) in FmdBenchmark.Summarize(scores))
                Console.WriteLine($"{group,-10} {count,3}  {m.NoisyPsnr,6:0.00} / {m.NoisySsim:0.000}  {m.BilateralPsnr,6:0.00} / {m.BilateralSsim:0.000}  {m.DoraPsnr,6:0.00} / {m.DoraSsim:0.000}");
            if (o.Has("per-image"))
                foreach (var s in scores)
                    Console.WriteLine($"  {s.Name,-40} {s.NoisyPsnr,6:0.00} {s.BilateralPsnr,6:0.00} {s.DoraPsnr,6:0.00}");
        }
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

    /// <summary><c>info</c>: prints a weights file's version, architecture, size, save time and provenance.</summary>
    private static int Info(Options o)
    {
        var (net, info) = WeightsFile.Load(o.Required("weights"));
        Console.WriteLine($"model version : {info.ModelVersion}");
        Console.WriteLine($"architecture  : {info.Config}");
        Console.WriteLine($"parameters    : {net.ParameterCount:N0}");
        Console.WriteLine($"saved (UTC)   : {info.SavedUtc:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"provenance    : {info.Provenance}");
        return 0;
    }

    /// <summary>One console line per epoch: losses, PSNR/SSIM (if clean data exists), learning rate, time and milestone events.</summary>
    private static string FormatEpoch(EpochReport r)
    {
        string psnr = r.ValPsnr is { } p ? $" | val PSNR {p:0.00} dB (+{r.ValPsnrGain:0.00}) SSIM {r.ValSsim:0.000}" : "";
        string events = r.Events.Count > 0 ? "  " + string.Join("  ", r.Events) : "";
        return $"📊 epoch {r.Epoch,3} | train {r.TrainLoss:0.00000} | val {r.ValLoss:0.00000}{psnr} | lr {r.LearningRate:G3} | {r.Elapsed:mm\\:ss}{events}";
    }

    /// <summary>Shortens <paramref name="s"/> to <paramref name="n"/> characters, keeping the end (file names differ at the end).</summary>
    private static string Truncate(string s, int n) => s.Length <= n ? s : "…" + s[^(n - 1)..];

    /// <summary>
    /// Minimal "--key value" parser with no dependencies. Every option takes exactly one value; there are no flags.
    /// Unknown keys are accepted and ignored, so typos surface as "required" errors or as defaults.
    /// </summary>
    private sealed class Options
    {
        private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);

        public static Options Parse(IEnumerable<string> args)
        {
            var o = new Options();
            using var e = args.GetEnumerator();
            while (e.MoveNext())
            {
                if (!e.Current.StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Unexpected argument '{e.Current}'.");
                string key = e.Current[2..];
                if (!e.MoveNext()) throw new ArgumentException($"--{key} needs a value.");
                o._values[key] = e.Current;
            }
            return o;
        }

        public bool Has(string key) => _values.ContainsKey(key);
        public string Required(string key) => _values.TryGetValue(key, out var v) ? v : throw new ArgumentException($"--{key} is required.");
        public string? StringOrNull(string key) => _values.GetValueOrDefault(key);
        public string String(string key, string fallback) => _values.GetValueOrDefault(key, fallback);

        public int Int(string key, int fallback) =>
            _values.TryGetValue(key, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

        public double Double(string key, double fallback) =>
            _values.TryGetValue(key, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : fallback;
    }
}

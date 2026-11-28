using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;

namespace DoraBio.Core;

/// <summary>
/// The one orchestration entry point (brick 41): <c>dora-bio denoise</c> and <c>dora-bio train</c> are thin mappings
/// from options to a <see cref="DenoiseRequest"/> or <see cref="TrainRequest"/>, and the Godot app builds the same
/// records, so a pilot user's app and a script on the lab server read, route, train and write identically (gate 8
/// checks byte-identical outputs and identical trained weights). Everything a run prints goes to <c>log</c>, one line
/// at a time, so the CLI writes it to the console and the app to its log panel, word for word.
/// </summary>
public static class Pipeline
{
    /// <summary>
    /// DORA's version as built, e.g. "2.0.0-bio+&lt;git commit&gt;" (the SDK appends the commit when building in a git
    /// checkout). It goes into every output's Software tag and every trained model's provenance.
    /// </summary>
    public static string Version { get; } =
        typeof(Pipeline).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

    /// <summary>
    /// Denoises one image file into a float32 TIFF with one page per plane, or every image in a folder into a folder
    /// (see <see cref="BatchDenoising"/> for the per-plane pipeline and its bounded memory).
    /// </summary>
    /// <param name="request">What to denoise, with which models.</param>
    /// <param name="log">Receives the run's report, one line at a time.</param>
    /// <param name="cancellation">Checked before each plane and between tiles.</param>
    /// <exception cref="ArgumentException">No model, an output path that is the input, or a stack the model can't take.</exception>
    public static DenoiseOutcome Denoise(DenoiseRequest request, Action<string>? log = null, CancellationToken cancellation = default)
    {
        void Log(string line) => log?.Invoke(line);
        var main = request.Model ?? (request.Weights is { } w ? LoadedModel.Load(w) : throw new ArgumentException("--weights is required."));
        var correlated = request.CorrelatedModel ?? (request.CorrelatedWeights is { } c ? LoadedModel.Load(c) : null);
        if (correlated is not null) Log($"🧬 correlated-noise model: {correlated.Info.Provenance}");
        var denoiser = new FrameDenoiser(main.Net, correlated?.Net, request.Vst, request.TileCore);
        Log($"🧬 model: {main.Info.Provenance}");
        string input = request.Input, output = request.Output;
        string? baseline = request.Baseline;
        var notices = new List<string>();
        void Notice(string notice)
        {
            notices.Add(notice);
            Log($"🧭 {notice}");
        }

        var clock = Stopwatch.StartNew();
        if (Directory.Exists(input))
        {
            var plan = BatchDenoising.PlanFolder(input, output);
            Directory.CreateDirectory(output);
            if (baseline is not null) Directory.CreateDirectory(baseline);
            int pages = 0, routedTotal = 0;
            foreach (var (file, target) in plan)
            {
                cancellation.ThrowIfCancellationRequested();
                string? baselineTarget = baseline is null ? null : Path.Combine(baseline, Path.GetFileName(target));
                int routed = 0;
                int n = BatchDenoising.DenoiseFile(denoiser, file, target, baselineTarget, (_, _, r) => routed += r.UsedCorrelatedModel ? 1 : 0,
                    cancellation, request.Axes, Notice);
                pages += n;
                routedTotal += routed;
                string routedText = correlated is null ? "" : $", {routed} by the correlated-noise model";
                Log($"   {Path.GetFileName(file)} → {target} ({n} page{(n == 1 ? "" : "s")}{routedText})");
            }
            Log($"✨ denoised {plan.Count} files, {pages} pages in {clock.Elapsed.TotalSeconds:0.00}s → {output}");
            Log(PeakMemoryLine());
            return new DenoiseOutcome(plan.Count, pages, routedTotal, clock.Elapsed.TotalSeconds, notices);
        }

        int total = BatchDenoising.CountPages(input, request.Axes);
        int routedPages = 0;
        BatchDenoising.PageDone report = total == 1
            ? (_, raw, r) =>
            {
                routedPages += r.UsedCorrelatedModel ? 1 : 0;
                if (r.UsedCorrelatedModel) Log("🔎 noise check: correlated, so the --weights-correlated model is used");
                Log($"✨ denoised {raw.Width}×{raw.Height} in {clock.Elapsed.TotalSeconds:0.00}s → {output}");
                Log($"   estimated noise σ (normalized): {r.SigmaBefore:0.0000} → {r.SigmaAfter:0.0000}");
                if (r.NoiseModel is { } m) Log($"📐 noise model: variance = {m.Gain:G4} · intensity {m.Intercept:+0.###;-0.###}");
                if (r.BaselineSigma is { } b) Log($"   bilateral baseline σ: {b:0.0000} → {baseline}");
            }
            : (page, raw, r) =>
            {
                routedPages += r.UsedCorrelatedModel ? 1 : 0;
                string model = r.UsedCorrelatedModel ? " (correlated-noise model)" : "";
                Log($"   page {page + 1}/{total}: {raw.Width}×{raw.Height}, noise σ {r.SigmaBefore:0.0000} → {r.SigmaAfter:0.0000}{model}");
            };
        BatchDenoising.DenoiseFile(denoiser, input, output, baseline, report, cancellation, request.Axes, Notice);
        double seconds = clock.Elapsed.TotalSeconds;
        if (total > 1)
        {
            Log($"✨ denoised {total} pages in {seconds:0.00}s ({seconds / total:0.00}s per page) → {output}" +
                (baseline is null ? "" : $", baseline → {baseline}"));
            if (correlated is not null) Log($"   {routedPages} of {total} pages used the correlated-noise model");
            Log(PeakMemoryLine());
        }
        return new DenoiseOutcome(1, total, routedPages, seconds, notices);
    }

    /// <summary>The process's peak resident memory so far (the OS high-water mark), as stack and folder runs report it.</summary>
    public static string PeakMemoryLine()
    {
        using var self = Process.GetCurrentProcess();
        return $"   peak memory: {self.PeakWorkingSet64 / (1024.0 * 1024.0):0} MB";
    }

    /// <summary>
    /// Trains as <c>dora-bio train</c> does: loads the data, resolves the mode and recipe (settings left null take
    /// <see cref="Recipes"/>' defaults), trains one network or the routed pair, and writes the weights.
    /// </summary>
    /// <param name="r">What to train on and how.</param>
    /// <param name="log">Receives the run's report, one line at a time, epochs included.</param>
    /// <param name="onEpoch">Also called after each epoch with its report (and the noise kind when routed), for a chart.</param>
    /// <param name="cancellation">Stops after the current step; the weights kept so far are still written.</param>
    /// <returns>Every network written, the main (or independent-noise) one first.</returns>
    /// <exception cref="ArgumentException">Contradictory or missing settings, named as the CLI options.</exception>
    public static IReadOnlyList<TrainedModel> Train(TrainRequest r, Action<string>? log = null, Action<NoiseKind?, EpochReport>? onEpoch = null,
        CancellationToken cancellation = default)
    {
        void Log(string line) => log?.Invoke(line);
        int sources = (r.Dataset is not null ? 1 : 0) + (r.Ext is not null ? 1 : 0);
        if (sources > 1) throw new ArgumentException("Train on an external dataset (--ext) or on images in memory, not both.");
        string? modeOption = r.Mode;
        int? defaultFrames = null;
        if (modeOption is null && r.Frames is null && r.Ext is null && r.Dataset is null && r.Data is { } dataPath && File.Exists(dataPath))
        {
            // A stack file with no mode: brick 40's verdict decides (Recipes.DefaultTimeLapse).
            if (Recipes.DefaultTimeLapse == "none")
                throw new ArgumentException($"{dataPath} is a stack file: train a time-lapse with --mode n2n, --mode skip-centre or " +
                                            "--frames 3 (brick 40 made none of them the default), or give a folder of images.");
            (modeOption, defaultFrames) = Recipes.DefaultTimeLapse switch
            {
                "n2n" => ("n2n", (int?)null),
                "skip-centre" => ("skip-centre", null),
                "frames-3" => ("n2v", 3),
                var d => throw new InvalidOperationException($"Unknown DefaultTimeLapse '{d}'."),
            };
            Log($"🎞 {Path.GetFileName(dataPath)} is a stack: brick 40's default time-lapse mode, {Recipes.DefaultTimeLapse}");
        }
        bool neighbourPairs = modeOption is "n2n" or "noise2noise";
        bool skipCentre = modeOption is "skip-centre";
        int frames = defaultFrames ?? r.Frames ?? (skipCentre ? 2 : 1);
        if (neighbourPairs && frames != 1)
            throw new ArgumentException("Noise2Noise takes one input frame (--frames 1): with several, its target would be part of the input.");
        if (skipCentre && frames != 2)
            throw new ArgumentException("Skip-centre Noise2Noise takes two input frames (t − 1, t + 1); leave --frames out.");
        bool timeLapse = frames > 1 || neighbourPairs || skipCentre || r.FrameRange is not null;
        if (r.ValidationRange is { } v0 && r.FrameRange is { } f0 && v0.First <= f0.Last && f0.First <= v0.Last)
            throw new ArgumentException($"--validation-range {v0} overlaps --frame-range {f0}: validation frames must never be trained on.");
        if (r.ValidationRange is not null && r.FrameRange is null)
            throw new ArgumentException("--validation-range needs --frame-range, so the training frames are named too.");
        List<ImagePair>? validation = null;
        List<ImagePair> dataset;
        if (timeLapse)
        {
            // Time-lapses (brick 39): a stack file or a folder of stacks; every plane is a frame.
            if (r.Ext is not null || r.Vst || r.FmdFrames is not null || r.FmdLevels is not null || r.FmdCategories is not null || r.Dataset is not null)
                throw new ArgumentException("--frames and --mode n2n train on time-lapse stacks (--data FILE.tif or a folder of stacks).");
            if (!neighbourPairs && !skipCentre && modeOption is not (null or "n2v" or "noise2void"))
                throw new ArgumentException("--frames 3 is the spatiotemporal blind spot, a Noise2Void mode (--mode n2v).");
            var sequenceMode = neighbourPairs ? TrainingMode.Noise2Noise : skipCentre ? TrainingMode.SkipCentre : TrainingMode.Noise2Void;
            string data = Required(r.Data, "data");
            dataset = Sequences.Load(data, sequenceMode, frames, r.FrameBlock ?? Sequences.DefaultBlockSize, r.FrameRange,
                r.Axes, notice => Log($"🧭 {notice}"));
            if (r.ValidationRange is { } range)
            {
                validation = [];
                foreach (var (label, planes, _) in Sequences.ReadSeries(data, null, r.Axes, range, null))
                {
                    string name = Path.GetFileNameWithoutExtension(data);
                    validation.AddRange(Sequences.ValidationFrames(planes, sequenceMode, frames, label.Length == 0 ? name : $"{name}[{label}]", range.First));
                }
                Log($"🧪 validation: frames {range} of {Path.GetFileName(data)}, {validation.Count} interior frames as patch sources");
            }
        }
        else if (r.Ext is { } extName)
        {
            // An external dataset's training split (field validation, Q18): supervised on its pairs, or Noise2Void on its
            // noisy images alone. Its test split is never loaded here.
            var ext = ExtDatasets.Get(extName);
            bool supervised = (r.Mode ?? (ext.HasGroundTruth ? "supervised" : "n2v")) == "supervised";
            if (supervised && !ext.HasGroundTruth) throw new ArgumentException($"{ext.Name} has no ground truth for supervised training.");
            if (r.Vst) throw new ArgumentException("--vst isn't supported with --ext.");
            dataset = ExtDatasets.LoadTraining(r.Data ?? "data/ext", ext, supervised);
        }
        else if (r.Dataset is { } inMemory)
        {
            if (r.Vst) throw new ArgumentException("--vst stabilizes images as they're read; images in memory are already normalized.");
            if (inMemory.Count == 0) throw new ArgumentException("No training images.");
            dataset = [.. inMemory];
        }
        else
        {
            var fmd = new FmdOptions
            {
                FramesPerFov = r.FmdFrames ?? 4,
                NoiseLevels = (r.FmdLevels ?? "raw").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
                Categories = r.FmdCategories?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
            };
            dataset = DatasetLoader.LoadDirectory(Required(r.Data, "data"), fmd, r.Vst ? VarianceStabilization.Stabilize : null);
        }
        var mode = timeLapse ? (neighbourPairs ? TrainingMode.Noise2Noise : skipCentre ? TrainingMode.SkipCentre : TrainingMode.Noise2Void)
            : (r.Mode ?? (dataset.All(d => d.HasClean) ? "supervised" : "n2v")) switch
            {
                "n2v" or "noise2void" => TrainingMode.Noise2Void,
                "supervised" => TrainingMode.Supervised,
                var m => throw new ArgumentException($"Unknown mode '{m}' (use n2v, supervised, n2n or skip-centre)."),
            };
        string source = r.Ext ?? (r.Dataset is not null ? r.DatasetName ?? "memory" : Path.GetFileName(Path.GetFullPath(Required(r.Data, "data"))));

        var config = new TrainerConfig
        {
            Mode = mode,
            MaxEpochs = r.Epochs ?? 30,
            StepsPerEpoch = r.Steps ?? 50,
            BatchSize = r.Batch ?? 16,
            Augmentation = (r.Augment ?? "dihedral") switch
            {
                "dihedral" => AugmentationMode.Dihedral,
                "v1" => AugmentationMode.V1RotateScale,
                var a => throw new ArgumentException($"Unknown --augment '{a}' (use dihedral or v1)."),
            },
            PatchSize = r.Patch ?? 64,
            MaskRatio = (float)(r.MaskRatio ?? 0.02),
            LearningRate = (float)(r.LearningRate ?? 1e-3),
            Seed = r.Seed ?? 1234,
        };
        bool n2vMode = mode == TrainingMode.Noise2Void;
        int patience = r.Patience ?? (n2vMode ? Recipes.DefaultN2v.Patience : 6);
        if (patience < 0) throw new ArgumentException("--patience must be 0 (never stop early) or more.");
        config = config with
        {
            EarlyStopPatience = patience == 0 ? int.MaxValue : patience,
            Selection = (r.Select ?? (n2vMode ? Recipes.DefaultN2v.Select : "auto")) switch
            {
                "auto" => ValidationSignal.Auto,
                "loss" => ValidationSignal.Loss,
                var v => throw new ArgumentException($"Unknown --select '{v}' (use auto or loss)."),
            },
            KeepLastEpoch = (r.Keep ?? (n2vMode ? Recipes.DefaultN2v.Keep : "best")) switch
            {
                "best" => false,
                "last" => true,
                var k => throw new ArgumentException($"Unknown --keep '{k}' (use best or last)."),
            },
        };

        // Q20: with no recipe option, Noise2Void routes each training image by the noise check (RoutedNoise2Void).
        string route = r.Route ?? (n2vMode && !r.ChoosesRecipe ? Recipes.DefaultN2vRoute : "off");
        if (route == "auto")
        {
            if (!n2vMode) throw new ArgumentException("--route auto is for Noise2Void training.");
            if (validation is not null) throw new ArgumentException("--validation-range trains one network: add --route off (and the recipe you want).");
            var verdicts = StructuredMasking.Diagnose(dataset);
            ReportNoiseCheck(verdicts, Log);
            string routedDescription = r.Describe ?? $"{dataset.Count} images from '{source}'";
            Log($"🔀 routed Noise2Void (Q20): independent noise → U-Net under rule C, correlated noise → dilated network with " +
                $"structured blind spots; {config.MaxEpochs} epochs × {config.StepsPerEpoch} steps × batch {config.BatchSize} @ {config.PatchSize}px");
            var routedClock = Stopwatch.StartNew();
            var trained = RoutedNoise2Void.Train(dataset, config, r.Out, $"{routedDescription}; {TrainingRecord(config, route, dataset)}", r.Journey,
                (kind, report) =>
                {
                    Log($"[{kind.ToString().ToLowerInvariant()}] {FormatEpoch(report)}");
                    onEpoch?.Invoke(kind, report);
                },
                r.OutCorrelated, verdicts, onNotice: notice => Log($"🔀 {notice}"), cancellation: cancellation);
            var models = trained.Select(t => new TrainedModel(t.Kind, t.WeightsPath, t.Images, t.Result, t.Kind == NoiseKind.Independent)).ToList();
            foreach (var t in models)
                Log($"💾 {t.Kind.ToString()!.ToLowerInvariant()} noise: {t.Images} images, " +
                    $"{(t.KeptLast ? "last" : "best")} epoch {t.Kept?.Epoch} saved to {t.WeightsPath}");
            if (models.Count == 2)
                Log($"   denoise with: --weights {models[0].WeightsPath} --weights-correlated {models[1].WeightsPath}");
            Log($"✅ routed training done in {(int)routedClock.Elapsed.TotalMinutes:00}:{routedClock.Elapsed.Seconds:00}");
            return models;
        }
        if (route != "off") throw new ArgumentException($"Unknown --route '{route}' (use auto or off).");

        DenoisingNet net;
        if (r.Init is not null || r.InitNet is not null)
        {
            if (r.Condition is not null) throw new ArgumentException("--condition comes from the --init weights; leave it out when fine-tuning.");
            if (r.InitNet is { } given) net = given.Clone();
            else
            {
                net = WeightsFile.Load(r.Init!).Net;
                Log($"🔁 fine-tuning {r.Init}");
            }
        }
        else
        {
            var defaults = NetConfig.Default;
            string arch = r.Arch ?? (n2vMode ? Recipes.DefaultN2v.Arch : defaults.Architecture == NetArchitecture.UNet ? "unet" : "dilated");
            var architecture = arch switch
            {
                "dilated" => new NetConfig(r.Features ?? 24, [1, 2, 4, 8, 4, 2, 1]),
                "unet" => NetConfig.UNet(r.Features ?? 16, r.Depth ?? 2),
                _ => throw new ArgumentException($"Unknown --arch '{arch}' (use dilated or unet)."),
            };
            architecture = (r.Condition ?? (mode == TrainingMode.Supervised ? Recipes.DefaultCondition : "none")) switch
            {
                "none" => architecture,
                "noise" => architecture with { Conditioning = NetConditioning.NoiseLevel },
                var c => throw new ArgumentException($"Unknown --condition '{c}' (use none or noise)."),
            };
            architecture = skipCentre ? NetConfig.SkipCentre(architecture) : architecture with { InputFrames = frames };
            architecture.Validate();
            net = DenoisingNet.Create(architecture, config.Seed);
        }

        if (r.Vst && mode != TrainingMode.Noise2Void)
            throw new ArgumentException("--vst is for Noise2Void: supervised targets are clean, so their noise needs no stabilizing.");
        string structure = r.Vst ? "; variance-stabilized (generalized Anscombe, each image calibrated on its own)" : "";
        // Subset: train only on the images the noise check calls correlated (or independent), for P3's routed design.
        if (r.Subset is { } subset)
        {
            bool wantCorrelated = subset switch
            {
                "correlated" => true,
                "independent" => false,
                _ => throw new ArgumentException($"Unknown --subset '{subset}' (use correlated or independent)."),
            };
            var all = StructuredMasking.Diagnose(dataset);
            dataset = dataset.Where((_, i) => all[i] == wantCorrelated).ToList();
            if (dataset.Count == 0) throw new ArgumentException($"No training image's noise check says {subset}.");
            structure += $"; only the {dataset.Count} images whose noise check says {subset}";
            Log($"🔎 --subset {subset}: training on {dataset.Count} of {all.Length} images");
        }
        if (mode == TrainingMode.Noise2Void)
        {
            var verdicts = StructuredMasking.Diagnose(dataset);
            ReportNoiseCheck(verdicts, Log);
            var structMode = (r.Struct ?? "off") switch
            {
                "off" => StructuredMaskMode.Off,
                "auto" => StructuredMaskMode.Auto,
                "always" => StructuredMaskMode.Always,
                var s => throw new ArgumentException($"Unknown --struct '{s}' (use off, auto or always)."),
            };
            var footprint = r.Footprint is { } f ? MaskFootprint.Parse(f) : StructuredMasking.DefaultFootprint;
            dataset = StructuredMasking.Assign(dataset, structMode, footprint, verdicts);
            int structured = dataset.Count(d => d.Footprint is not null);
            if (structured > 0)
            {
                structure += $"; structured blind spots ({footprint}) on {structured} of {dataset.Count} images";
                Log($"🧩 structured blind spots ({footprint}) on {structured} of {dataset.Count} training images");
            }
        }

        int withClean = dataset.Count(d => d.HasClean);
        string cleanUse = withClean == 0 ? "no clean images"
            : mode == TrainingMode.Noise2Void ? $"{withClean} clean images used only to score validation"
            : mode == TrainingMode.Noise2Noise ? $"{withClean} neighbouring-frame pairs, no ground truth"
            : mode == TrainingMode.SkipCentre ? $"{withClean} clean images used only to score validation"
            : $"{withClean} clean targets";
        string description = (r.Describe ?? $"{dataset.Count} images from '{source}' ({cleanUse})") + structure;
        Log($"🧬 {net.Config}, {net.ParameterCount:N0} parameters");
        Log($"📚 {description}; mode {mode}; {config.MaxEpochs} epochs × {config.StepsPerEpoch} steps × batch {config.BatchSize} @ {config.PatchSize}px");

        var clock = Stopwatch.StartNew();
        var result = TrainingJob.Run(net, dataset, config,
            new TrainingJobOptions
            {
                DatasetDescription = $"{description}; {TrainingRecord(config, route, dataset)}", WeightsPath = r.Out, JournalPath = r.Journey,
                LastWeightsPath = r.SaveLast, Validation = validation,
            },
            report =>
            {
                Log(FormatEpoch(report));
                onEpoch?.Invoke(null, report);
            },
            cancellation);

        // Total minutes, so a run past an hour doesn't wrap (mm:ss alone would print 05:00 for 1 h 5 min).
        Log($"✅ stopped: {result.StopReason} after {result.History.Count} epochs in {(int)clock.Elapsed.TotalMinutes:00}:{clock.Elapsed.Seconds:00}");
        var model = new TrainedModel(null, r.Out, dataset.Count, result, config.KeepLastEpoch);
        if (model.Kept is { } kept)
        {
            string psnr = kept.ValPsnr is { } p ? $", val PSNR {p:0.00} dB vs noisy {result.NoisyBaselinePsnr:0.00} dB" : "";
            Log($"💾 {(config.KeepLastEpoch ? "last" : "best")} epoch {kept.Epoch} (val loss {kept.ValLoss:G4}{psnr}) saved to {r.Out}");
        }
        if (r.SaveLast is { } lastPath && result.Last is { } last)
            Log($"💾 last epoch {last.Epoch} (val loss {last.ValLoss:G4}) saved to {lastPath}");
        return [model];
    }

    /// <summary>
    /// What a weights file's provenance records about its training (brick 41, after the description): the DORA build, every
    /// setting the run resolved (seed included) and a fingerprint of the training data as loaded, so a model can be traced
    /// to its data and retrained. Text only: the weights format is unchanged.
    /// </summary>
    internal static string TrainingRecord(TrainerConfig c, string route, IReadOnlyList<ImagePair> data) =>
        string.Create(CultureInfo.InvariantCulture,
            $"trained by DORA {Version}: {c.Mode}, route {route}, up to {c.MaxEpochs} epochs × {c.StepsPerEpoch} steps × batch {c.BatchSize} " +
            $"@ {c.PatchSize}px, lr {c.LearningRate:G4}, mask ratio {c.MaskRatio:G3}, {c.Augmentation} augmentation, seed {c.Seed}; " +
            $"data {DataFingerprint(data)} ({data.Count} images)");

    /// <summary>
    /// A fingerprint of training data as the trainer sees it: SHA-256 over every image's name, size and pixels (noisy, and
    /// clean when it has one), in order; "sha256:" and the first 16 hex digits. The same files give the same value whatever
    /// their format or path.
    /// </summary>
    public static string DataFingerprint(IReadOnlyList<ImagePair> data)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var size = new byte[8];
        void Add(GrayImage image)
        {
            BitConverter.TryWriteBytes(size, image.Width);
            BitConverter.TryWriteBytes(size.AsSpan(4), image.Height);
            sha.AppendData(size);
            sha.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(image.Pixels.AsSpan()));
        }
        foreach (var pair in data)
        {
            sha.AppendData(System.Text.Encoding.UTF8.GetBytes(pair.Name + "\0"));
            Add(pair.Noisy);
            if (pair.Clean is { } clean) Add(clean);
        }
        return "sha256:" + Convert.ToHexString(sha.GetHashAndReset())[..16].ToLowerInvariant();
    }

    /// <summary>One line per epoch: losses, PSNR/SSIM (if clean data exists), learning rate, time and milestone events.</summary>
    public static string FormatEpoch(EpochReport r)
    {
        string psnr = r.ValPsnr is { } p ? $" | val PSNR {p:0.00} dB (+{r.ValPsnrGain:0.00}) SSIM {r.ValSsim:0.000}" : "";
        string events = r.Events.Count > 0 ? "  " + string.Join("  ", r.Events) : "";
        return $"📊 epoch {r.Epoch,3} | train {r.TrainLoss:0.00000} | val {r.ValLoss:0.00000}{psnr} | lr {r.LearningRate:G3} | {r.Elapsed:mm\\:ss}{events}";
    }

    /// <summary>The training set's noise check, and what to do when Noise2Void's assumption fails.</summary>
    private static void ReportNoiseCheck(bool?[] verdicts, Action<string> log)
    {
        int judged = verdicts.Count(v => v is not null), correlated = verdicts.Count(v => v == true);
        log($"🔎 noise check: {correlated} of {judged} training images have noise correlated between pixels");
        if (correlated == 0) return;
        log("⚠️ Noise2Void assumes independent noise and will keep part of the correlated noise in those images.");
        log(correlated == judged
            ? "   Use --struct auto (structured blind spots: +3.8 dB on FMD widefield), or --mode supervised if you have clean targets."
            : "   Mixed data: one network does worse on the independent images. Train two, with --subset independent and with " +
              "--subset correlated --struct auto, then route with --weights-correlated (see docs/RESULTS.md, P3).");
    }

    private static string Required(string? value, string option) => value ?? throw new ArgumentException($"--{option} is required.");
}

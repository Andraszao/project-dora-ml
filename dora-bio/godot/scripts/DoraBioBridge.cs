using DoraBio.Core;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Godot;
using GodotFileAccess = Godot.FileAccess;

namespace DoraBio.GodotBridge;

/// <summary>
/// The only Godot-facing class: it connects the GDScript UI to DoraBio.Core.
///
/// Threading contract:
///   • Public methods are called from the main thread (GDScript).
///   • Heavy work (denoise, adapt, train) runs on the .NET thread pool, one job at a time (<see cref="IsBusy"/>).
///   • Every signal is emitted on the main thread via CallDeferred, so GDScript handlers can touch nodes.
///
/// From GDScript, C# members keep their PascalCase names:
///   bridge.LoadImage(path); bridge.DenoiseFinished.connect(handler)
/// </summary>
[GlobalClass]
public partial class DoraBioBridge : Node
{
    // ---- Inspector settings -----------------------------------------------------------------------

    /// <summary>Where trained or adapted weights are saved and loaded first. user:// stays writable in exported builds.</summary>
    [Export] public string UserWeightsPath { get; set; } = "user://dora_bio_weights.bin";

    /// <summary>Read-only fallback model shipped inside the project (trained on synthetic data).</summary>
    [Export] public string BundledWeightsPath { get; set; } = "res://models/dora_bio_weights.bin";

    /// <summary>Training-history JSON written by every training run started from the app.</summary>
    [Export] public string JournalPath { get; set; } = "user://dora_bio_journey.json";

    /// <summary>Tile core size for inference (see <see cref="TiledInference.DefaultTileCore"/>).</summary>
    [Export] public int TileCore { get; set; } = TiledInference.DefaultTileCore;

    /// <summary>Step size for "Adapt to sample"; kept small so adaptation refines rather than overwrites the model.</summary>
    [Export] public float AdaptLearningRate { get; set; } = 1e-4f;

    /// <summary>Where each denoise writes its result: the same float32 TIFF <c>dora-bio denoise</c> writes (brick 41).</summary>
    [Export] public string DenoisedPath { get; set; } = "user://dora_bio_denoised.tif";

    /// <summary>Where each denoise writes the bilateral baseline it is compared with.</summary>
    [Export] public string BaselinePath { get; set; } = "user://dora_bio_baseline.tif";

    // ---- Signals (GDScript: bridge.<Name>.connect(handler)). All are emitted on the main thread. ----

    /// <summary>The loaded model changed (startup, training, adaptation, save). <c>description</c> = <see cref="DescribeModel"/>.</summary>
    [Signal] public delegate void ModelChangedEventHandler(string description);

    /// <summary>
    /// An image was decoded and normalized. <c>raw</c> is a 32-bit float (FORMAT_RF) image of the normalized data.
    /// <c>displayLow</c>/<c>displayHigh</c> are its 0.5%/99.5% quantiles, the suggested shared display window.
    /// <c>description</c> gives name, size and raw intensity range.
    /// </summary>
    [Signal] public delegate void ImageLoadedEventHandler(Image raw, float displayLow, float displayHigh, string description);

    /// <summary>
    /// Denoising finished. <c>denoised</c> and <c>baseline</c> (bilateral) are FORMAT_RF images on the same scale as the raw image.
    /// The three <c>noise*</c> values are Immerkær noise-σ estimates (raw, DORA, bilateral), and <c>seconds</c>
    /// is the denoising run's wall-clock time (DORA and the baseline, files written).
    /// </summary>
    [Signal] public delegate void DenoiseFinishedEventHandler(Image denoised, Image baseline, float noiseRaw, float noiseDenoised, float noiseBaseline, float seconds);

    /// <summary>
    /// A training epoch finished. <c>valPsnr</c> and <c>psnrGain</c> are NaN without clean validation data.
    /// <c>events</c> is the joined milestone text (new best, plateau, target hit/raised/mastered).
    /// </summary>
    [Signal] public delegate void EpochCompletedEventHandler(int epoch, float valLoss, float valPsnr, float psnrGain, float learningRate, string events);

    /// <summary>Training ended (any reason). <c>summary</c> says why and how well; the new weights are already loaded and saved.</summary>
    [Signal] public delegate void TrainingFinishedEventHandler(string summary);

    /// <summary>
    /// Adaptation finished. Blind-spot loss at the first and last step: a drop means the model fit this sample's noise.
    /// Weights are now unsaved.
    /// </summary>
    [Signal] public delegate void AdaptFinishedEventHandler(int steps, float firstLoss, float lastLoss);

    /// <summary>A background job started (<c>busy</c> = true, <c>activity</c> names it) or ended (false, ""). Use it to disable buttons.</summary>
    [Signal] public delegate void BusyChangedEventHandler(bool busy, string activity);

    /// <summary>Informational message for the log view (may contain emoji).</summary>
    [Signal] public delegate void LogEventHandler(string message);

    /// <summary>A request failed or was refused (busy, no image, decode error…). <c>message</c> is user-facing.</summary>
    [Signal] public delegate void JobFailedEventHandler(string message);

    // ---- State: touched only on the main thread, except that a running job reads its own captured copies ----

    private DenoisingNet _net = DenoisingNet.Create(NetConfig.Default);
    private ModelInfo? _info;
    private LoadedModel? _correlated;   // routed training's second network (correlated noise), or null
    private string _modelSource = "untrained";
    private string? _modelFile;         // the weights file _net was loaded from, as an OS path (null: res:// or untrained)
    private OnlineAdapter? _adapter;    // created on first Adapt, discarded when the model changes
    private GrayImage? _image;          // normalized
    private IntensityMap? _map;         // the loaded image's normalization, so outputs read back share its display scale
    private string? _imageFile;         // the loaded image as an OS path the pipeline can open (res:// is copied to user://)
    private NoiseDiagnosis? _noise;     // the loaded image's noise check; null if it was too small to judge
    private string _imageName = "";
    private CancellationTokenSource? _cancellation;
    private Task? _job;                 // the running job, so leaving the tree can wait for it
    private bool _busy;
    private bool _unsaved;

    /// <summary>True while a denoise, adapt or training job runs. Only one job runs at a time.</summary>
    public bool IsBusy() => _busy;

    /// <summary>True once an image has been loaded successfully.</summary>
    public bool HasImage() => _image is not null;

    /// <summary>True after adaptation changed the weights since the last save or load.</summary>
    public bool HasUnsavedChanges() => _unsaved;

    /// <summary>
    /// The loaded image's noise check (<see cref="NoiseDiagnostics"/>): whether its noise looks independent between
    /// pixels, as Noise2Void and "Adapt to sample" assume. Empty if no image is loaded or it was too small to judge.
    /// </summary>
    public string DescribeNoise() => _noise?.ToString() ?? "";

    /// <summary>Loads weights only. Datasets load lazily, when training is requested.</summary>
    public override void _Ready() => LoadBestAvailableModel();

    /// <summary>
    /// Stops any running job when the node leaves the tree (e.g. the window closes) and waits for it, at most 10 s:
    /// a job still running when the engine shuts down would call into a torn-down Godot and crash the process.
    /// Jobs check the cancellation between tiles and steps, so the wait is short.
    /// </summary>
    public override void _ExitTree()
    {
        _cancellation?.Cancel();
        _job?.Wait(TimeSpan.FromSeconds(10));
    }

    /// <summary>One line about the current model, for the UI header.</summary>
    public string DescribeModel()
    {
        string provenance = _info?.Provenance ?? "untrained network: train or adapt before trusting the output";
        string routed = _correlated is null ? "" : " · + correlated-noise network";
        return $"{_modelSource} · {_net.ParameterCount:N0} params · {provenance}{routed}{(_unsaved ? " · unsaved changes" : "")}";
    }

    /// <summary>
    /// The content hash of the current main network (architecture and weights, not provenance or save time), as
    /// <c>dora-bio info</c> prints it: equal hashes mean the app and the CLI hold the same model.
    /// </summary>
    public string ModelHash() => WeightsFile.ContentHash(_net);

    /// <summary>The weights file the current model came from, as an OS path, or "" when it isn't a file on disk (inside an exported package, or untrained).</summary>
    public string ModelFile() => _modelFile ?? "";

    /// <summary>Where the last denoised image was written (float32 TIFF in the input's units), as an OS path.</summary>
    public string DenoisedFile() => ProjectSettings.GlobalizePath(DenoisedPath);

    /// <summary>Where the last bilateral baseline was written, as an OS path.</summary>
    public string BaselineFile() => ProjectSettings.GlobalizePath(BaselinePath);

    /// <summary>
    /// Loads user:// weights if they exist, else the bundled res:// model, else a fresh network.
    /// Only weights are read here. Datasets are not touched until training is requested (v1 loaded MNIST on every launch).
    /// </summary>
    public void LoadBestAvailableModel()
    {
        if (_busy) return;
        foreach (string path in new[] { UserWeightsPath, BundledWeightsPath })
        {
            if (!GodotFileAccess.FileExists(path)) continue;
            try
            {
                var (net, info) = WeightsFile.Load(GodotFileAccess.GetFileAsBytes(path));
                if (net.Config.InputFrames > 1)
                {
                    // A time-lapse model (brick 39) needs each frame's neighbours; the app works on single images.
                    WriteLog($"⚠️ {path} takes {net.Config.InputFrames} frames (a time-lapse model); use the dora-bio CLI on stacks");
                    continue;
                }
                // Routed training (brick 41: the app trains what the CLI trains) leaves a second network beside the first.
                string companion = CompanionPath(path);
                LoadedModel? correlated = GodotFileAccess.FileExists(companion)
                    ? LoadedModelFrom(companion)
                    : null;
                SetModel(net, info, path, correlated);
                return;
            }
            catch (Exception e) when (e is InvalidDataException or ArgumentException)
            {
                WriteLog($"⚠️ could not load {path}: {e.Message}");
            }
        }
        SetModel(DenoisingNet.Create(NetConfig.Default), null, "untrained");
    }

    /// <summary>Decodes and normalizes an image (res://, user:// or absolute path). Fast enough to run synchronously.</summary>
    public bool LoadImage(string path)
    {
        if (_busy) return false;
        try
        {
            byte[] bytes = GodotFileAccess.GetFileAsBytes(path);
            if (bytes.Length == 0) throw new FileNotFoundException($"Could not read {path} ({GodotFileAccess.GetOpenError()}).");
            var raw = ImageLoader.Decode(bytes, path);
            var map = Normalization.FromPercentiles(raw.Pixels);
            _image = Normalization.Apply(raw, map);
            _map = map;
            _imageFile = OsFile(path, bytes);
            _imageName = path.GetFile();

            var sorted = (float[])_image.Pixels.Clone();
            Array.Sort(sorted);
            float low = Normalization.Quantile(sorted, 0.005), high = Normalization.Quantile(sorted, 0.995);
            var (rawMin, rawMax) = raw.Range();
            // Milliseconds even for large frames, so it runs here rather than as a job.
            try { _noise = NoiseDiagnostics.Diagnose(_image); }
            catch (ArgumentException) { _noise = null; }
            EmitSignal(SignalName.ImageLoaded, ToGodotImage(_image), low, high,
                $"{_imageName} · {raw.Width}×{raw.Height} · raw range {rawMin:0.#}–{rawMax:0.#}");
            WriteLog(_noise is null ? "🔎 noise check skipped: image too small"
                : (_noise.Correlated ? "⚠️ " : "🔎 ") + _noise);
            return true;
        }
        catch (Exception e)
        {
            EmitSignal(SignalName.JobFailed, $"Could not load {path}: {e.Message}");
            return false;
        }
    }

    /// <summary>
    /// Runs DORA and the bilateral baseline on the loaded image through <see cref="Pipeline.Denoise"/>, the path
    /// <c>dora-bio denoise --baseline</c> takes: the same routing, the same float32 TIFFs (written to
    /// <see cref="DenoisedPath"/> and <see cref="BaselinePath"/>), byte for byte (gate 8). The panels then show what was
    /// written, on the raw image's display scale.
    /// </summary>
    public void StartDenoise()
    {
        if (_image is null || _map is null || _imageFile is null) { EmitSignal(SignalName.JobFailed, "Load an image first."); return; }
        var request = new DenoiseRequest
        {
            Input = _imageFile,
            Output = DenoisedFile(),
            Baseline = BaselineFile(),
            Model = new LoadedModel(_net, CurrentInfo()),
            CorrelatedModel = _correlated,
            TileCore = TileCore,
        };
        var image = _image;
        var map = _map.Value;
        RunJob("Denoising", token =>
        {
            var outcome = Pipeline.Denoise(request, line => Defer(() => WriteLog(line)), token);
            var denoised = Normalization.Apply(ImageLoader.Load(request.Output), map);
            var baseline = Normalization.Apply(ImageLoader.Load(request.Baseline!), map);
            float noiseRaw = (float)Metrics.EstimateNoiseSigma(image);
            float noiseDenoised = (float)Metrics.EstimateNoiseSigma(denoised);
            float noiseBaseline = (float)Metrics.EstimateNoiseSigma(baseline);
            return () => EmitSignal(SignalName.DenoiseFinished, ToGodotImage(denoised), ToGodotImage(baseline),
                noiseRaw, noiseDenoised, noiseBaseline, (float)outcome.Seconds);
        });
    }

    /// <summary>The current model's info, with any unsaved adaptation noted in its provenance.</summary>
    private ModelInfo CurrentInfo()
    {
        var info = _info ?? new ModelInfo(_net.Config, WeightsFile.ModelVersion, "untrained network", DateTime.UtcNow);
        return _adapter is { OptimizerStep: > 0 } a ? info with { Provenance = $"{info.Provenance}; + {a.OptimizerStep} online N2V steps" } : info;
    }

    /// <summary>
    /// Self-supervised Noise2Void fine-tuning on the loaded image. The optimizer state persists across calls
    /// for the lifetime of the loaded model. Nothing is saved until <see cref="SaveCheckpoint"/>.
    /// </summary>
    public void StartAdapt(int steps)
    {
        if (_image is null) { EmitSignal(SignalName.JobFailed, "Load an image first."); return; }
        var image = _image;
        int patch = Math.Min(64, Math.Min(image.Width, image.Height) / 16 * 16);
        if (patch < 16) { EmitSignal(SignalName.JobFailed, "Image is too small to adapt on (need at least 16×16)."); return; }
        _adapter ??= new OnlineAdapter(_net, AdaptLearningRate, patchSize: patch);
        var adapter = _adapter;
        // Correlated noise: structured blind spots, which hide the correlated neighbours too (P3, docs/RESULTS.md).
        var footprint = _noise?.Correlated == true ? StructuredMasking.DefaultFootprint : null;
        if (footprint is not null)
            WriteLog($"🧩 this image's noise is correlated between pixels, so adaptation uses structured blind spots ({footprint})");

        RunJob("Adapting to sample", token =>
        {
            var losses = adapter.Adapt(image, Math.Max(1, steps), token, footprint);
            return () =>
            {
                _unsaved = true;
                if (losses.Count > 0)
                    EmitSignal(SignalName.AdaptFinished, losses.Count, (float)losses[0], (float)losses[^1]);
                EmitSignal(SignalName.ModelChanged, DescribeModel());
            };
        });
    }

    /// <summary>
    /// Fine-tunes (or, <paramref name="fromScratch"/>, trains anew) on a folder in any layout <see cref="DatasetLoader"/>
    /// understands: <c>dora-bio train --data FOLDER --epochs N --steps 30 --batch 8</c>, plus <c>--lr 3e-4</c> and the
    /// current weights as <c>--init</c> when fine-tuning (see <see cref="StartTraining"/>).
    /// </summary>
    public void StartTrainingOnFolder(string folder, int epochs, bool fromScratch)
    {
        string path = ProjectSettings.GlobalizePath(folder);
        StartTraining($"Training on {path.GetFile()}", () => new TrainRequest { Data = path, Out = "" }, epochs, fromScratch);
    }

    /// <summary>
    /// Generates synthetic fluorescence data in memory and trains on it: a demo that needs no download. The data's seed
    /// is fixed, so two runs of the demo train the same model.
    /// </summary>
    public void StartSyntheticTraining(int epochs, bool fromScratch)
    {
        StartTraining("Training on synthetic data",
            () => new TrainRequest
            {
                Dataset = SyntheticFluorescence.GenerateDataset(8, 192, seed: SyntheticSeed),
                DatasetName = "8 synthetic 192px FOVs (generated in-app)",
                Out = "",
            },
            epochs, fromScratch);
    }

    /// <summary>Seed of the synthetic demo data.</summary>
    private const int SyntheticSeed = 41;

    /// <summary>Requests cancellation of the running job (if any). Training stops after its current step and keeps its best weights.</summary>
    public void Cancel() => _cancellation?.Cancel();

    /// <summary>Saves the current weights (including online adaptation) to <see cref="UserWeightsPath"/>.</summary>
    public bool SaveCheckpoint()
    {
        if (_busy) return false;
        try
        {
            string path = ProjectSettings.GlobalizePath(UserWeightsPath);
            WeightsFile.Save(path, _net, CurrentInfo().Provenance);
            _info = WeightsFile.Load(path).Info;
            _modelSource = UserWeightsPath;
            _modelFile = path;
            _unsaved = false;
            WriteLog($"💾 saved weights to {path}");
            EmitSignal(SignalName.ModelChanged, DescribeModel());
            return true;
        }
        catch (Exception e)
        {
            EmitSignal(SignalName.JobFailed, $"Save failed: {e.Message}");
            return false;
        }
    }

    /// <summary>
    /// Shared body of both training entry points: <see cref="Pipeline.Train"/>, the code behind <c>dora-bio train</c>, so
    /// the app picks the mode, recipe and routing the CLI picks (gate 8 checks the trained weights match). The app only
    /// sets what it needs: 30 steps × batch 8 per epoch (frequent progress updates), the user's epoch count, and when
    /// fine-tuning a copy of the current model (<c>--init</c>) at a lower learning rate (3e-4 rather than 1e-3). Weights
    /// are written to <see cref="UserWeightsPath"/> (and, when training is routed and finds both kinds of noise, its
    /// <c>.correlated</c> companion); the on-screen model stays as it was until training wrote new weights.
    /// </summary>
    private void StartTraining(string activity, Func<TrainRequest> data, int epochs, bool fromScratch)
    {
        string weightsPath = ProjectSettings.GlobalizePath(UserWeightsPath);
        string journalPath = ProjectSettings.GlobalizePath(JournalPath);
        var init = fromScratch ? null : _net;   // Pipeline.Train trains a copy

        RunJob(activity, token =>
        {
            // Built here, not on the main thread: generating or reading the data can take a while.
            var request = data() with
            {
                Out = weightsPath,
                Journey = journalPath,
                Epochs = Math.Max(1, epochs),
                Steps = 30,
                Batch = 8,
                LearningRate = fromScratch ? 1e-3 : 3e-4,
                InitNet = init,
            };
            var models = Pipeline.Train(request, line => Defer(() => WriteLog(line)),
                (_, report) => Defer(() => EmitSignal(SignalName.EpochCompleted, report.Epoch, (float)report.ValLoss,
                    (float)(report.ValPsnr ?? double.NaN), (float)(report.ValPsnrGain ?? double.NaN),
                    report.LearningRate, string.Join("  ", report.Events))),
                token);

            var main = models[0];
            if (main.Kept is null)
                return () => EmitSignal(SignalName.TrainingFinished, $"stopped: {main.Result.StopReason} before any epoch finished; model unchanged");
            var trained = LoadedModel.Load(main.WeightsPath);
            var correlated = models.Count > 1 ? LoadedModel.Load(models[1].WeightsPath) : null;
            // A single network replaces any routed pair saved earlier: don't pair it with a stale companion on the next launch.
            string companion = RoutedNoise2Void.CorrelatedPath(weightsPath);
            if (correlated is null && File.Exists(companion)) File.Delete(companion);
            var kept = main.Kept;
            return () =>
            {
                SetModel(trained.Net, trained.Info, UserWeightsPath, correlated, main.WeightsPath);
                EmitSignal(SignalName.TrainingFinished,
                    $"stopped: {main.Result.StopReason} after {main.Result.History.Count} epochs; kept {(main.KeptLast ? "last" : "best")} epoch {kept.Epoch}" +
                    (kept.ValPsnrGain is { } g ? $" (+{g:0.00} dB over noisy)" : "") +
                    (correlated is null ? "" : "; + a correlated-noise network") +
                    $"; weights saved to {UserWeightsPath}");
            };
        });
    }

    /// <summary>Installs a model and announces it. Main thread only.</summary>
    private void SetModel(DenoisingNet net, ModelInfo? info, string source, LoadedModel? correlated = null, string? file = null)
    {
        _net = net;
        _info = info;
        _correlated = correlated;
        _modelSource = source;
        _modelFile = file ?? (source.StartsWith("res://", StringComparison.Ordinal) || source.StartsWith("user://", StringComparison.Ordinal)
            ? ProjectSettings.GlobalizePath(source) : null);
        if (_modelFile is not null && !File.Exists(_modelFile)) _modelFile = null;   // res:// inside an exported package
        _adapter = null;      // optimizer state belongs to the previous weights
        _unsaved = false;
        EmitSignal(SignalName.ModelChanged, DescribeModel());
    }

    /// <summary>The <c>.correlated</c> companion of a weights path (res://, user:// or an OS path).</summary>
    private static string CompanionPath(string path)
    {
        string file = path.GetFile();
        string stem = file.GetBaseName(), extension = file.GetExtension();
        return path[..^file.Length] + stem + ".correlated" + (extension.Length > 0 ? "." + extension : "");
    }

    /// <summary>Loads a weights file through Godot's file access, so res:// paths inside an exported package work.</summary>
    private static LoadedModel LoadedModelFrom(string path)
    {
        var (net, info) = WeightsFile.Load(GodotFileAccess.GetFileAsBytes(path));
        return new LoadedModel(net, info);
    }

    /// <summary>
    /// An OS path for an image the pipeline can open: user:// and OS paths as they are; a res:// file (inside the
    /// package once exported, so not a file) is copied to user://inputs first.
    /// </summary>
    private static string OsFile(string path, byte[] bytes)
    {
        if (!path.StartsWith("res://", StringComparison.Ordinal)) return ProjectSettings.GlobalizePath(path);
        string target = ProjectSettings.GlobalizePath("user://inputs/" + path.GetFile());
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        File.WriteAllBytes(target, bytes);
        return target;
    }

    /// <summary>Runs <paramref name="work"/> on the thread pool, then its returned continuation on the main thread.</summary>
    private void RunJob(string activity, Func<CancellationToken, Action> work)
    {
        if (_busy) { EmitSignal(SignalName.JobFailed, "DORA is busy; wait or cancel first."); return; }
        _busy = true;
        _cancellation = new CancellationTokenSource();
        var token = _cancellation.Token;
        EmitSignal(SignalName.BusyChanged, true, activity);

        _job = Task.Run(() =>
        {
            Action onMainThread;
            try
            {
                onMainThread = work(token);
            }
            catch (OperationCanceledException)
            {
                onMainThread = () => WriteLog($"⏹ {activity} cancelled");
            }
            catch (Exception e)
            {
                onMainThread = () => EmitSignal(SignalName.JobFailed, $"{activity} failed: {e.Message}");
            }

            Defer(() =>
            {
                // Become idle *before* the continuation emits result signals, so their handlers can
                // start the next job (e.g. the UI re-denoises right after an adaptation finishes).
                _busy = false;
                _cancellation?.Dispose();
                _cancellation = null;
                EmitSignal(SignalName.BusyChanged, false, "");
                onMainThread();
            });
        });
    }

    /// <summary>Queues an action for the main thread, skipping it if this node has been freed meanwhile.</summary>
    private void Defer(Action action) => Callable.From(() =>
    {
        if (IsInstanceValid(this)) action();
    }).CallDeferred();

    private void WriteLog(string message) => EmitSignal(SignalName.Log, message);

    /// <summary>32-bit float single-channel image: no 8-bit quantization between the model and the screen.</summary>
    private static Image ToGodotImage(GrayImage image)
    {
        var bytes = new byte[image.Length * sizeof(float)];
        Buffer.BlockCopy(image.Pixels, 0, bytes, 0, bytes.Length);
        return Image.CreateFromData(image.Width, image.Height, false, Image.Format.Rf, bytes);
    }
}

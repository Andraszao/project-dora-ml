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
    /// is DORA's inference time.
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
    private string _modelSource = "untrained";
    private OnlineAdapter? _adapter;    // created on first Adapt, discarded when the model changes
    private GrayImage? _image;          // normalized
    private string _imageName = "";
    private CancellationTokenSource? _cancellation;
    private bool _busy;
    private bool _unsaved;

    /// <summary>True while a denoise, adapt or training job runs. Only one job runs at a time.</summary>
    public bool IsBusy() => _busy;

    /// <summary>True once an image has been loaded successfully.</summary>
    public bool HasImage() => _image is not null;

    /// <summary>True after adaptation changed the weights since the last save or load.</summary>
    public bool HasUnsavedChanges() => _unsaved;

    /// <summary>Loads weights only. Datasets load lazily, when training is requested.</summary>
    public override void _Ready() => LoadBestAvailableModel();

    /// <summary>Asks any running job to stop when the node leaves the tree (e.g. the window closes).</summary>
    public override void _ExitTree() => _cancellation?.Cancel();

    /// <summary>One line about the current model, for the UI header.</summary>
    public string DescribeModel()
    {
        string provenance = _info?.Provenance ?? "untrained network: train or adapt before trusting the output";
        return $"{_modelSource} · {_net.ParameterCount:N0} params · {provenance}{(_unsaved ? " · unsaved changes" : "")}";
    }

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
                SetModel(net, info, path);
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
            _imageName = path.GetFile();

            var sorted = (float[])_image.Pixels.Clone();
            Array.Sort(sorted);
            float low = Normalization.Quantile(sorted, 0.005), high = Normalization.Quantile(sorted, 0.995);
            var (rawMin, rawMax) = raw.Range();
            EmitSignal(SignalName.ImageLoaded, ToGodotImage(_image), low, high,
                $"{_imageName} · {raw.Width}×{raw.Height} · raw range {rawMin:0.#}–{rawMax:0.#}");
            return true;
        }
        catch (Exception e)
        {
            EmitSignal(SignalName.JobFailed, $"Could not load {path}: {e.Message}");
            return false;
        }
    }

    /// <summary>Runs DORA (tiled, all cores) and the bilateral baseline on the loaded image.</summary>
    public void StartDenoise()
    {
        if (_image is null) { EmitSignal(SignalName.JobFailed, "Load an image first."); return; }
        var net = _net;
        var image = _image;
        RunJob("Denoising", token =>
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var denoised = TiledInference.Denoise(net, image, TileCore, token);
            float seconds = (float)clock.Elapsed.TotalSeconds;
            var baseline = BilateralBaseline.FilterAuto(image);
            float noiseRaw = (float)Metrics.EstimateNoiseSigma(image);
            float noiseDenoised = (float)Metrics.EstimateNoiseSigma(denoised);
            float noiseBaseline = (float)Metrics.EstimateNoiseSigma(baseline);
            return () => EmitSignal(SignalName.DenoiseFinished, ToGodotImage(denoised), ToGodotImage(baseline),
                noiseRaw, noiseDenoised, noiseBaseline, seconds);
        });
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

        RunJob("Adapting to sample", token =>
        {
            var losses = adapter.Adapt(image, Math.Max(1, steps), token);
            return () =>
            {
                _unsaved = true;
                if (losses.Count > 0)
                    EmitSignal(SignalName.AdaptFinished, losses.Count, (float)losses[0], (float)losses[^1]);
                EmitSignal(SignalName.ModelChanged, DescribeModel());
            };
        });
    }

    /// <summary>Fine-tunes (or retrains) on a folder in any layout <see cref="DatasetLoader"/> understands.</summary>
    public void StartTrainingOnFolder(string folder, int epochs, bool fromScratch)
    {
        string path = ProjectSettings.GlobalizePath(folder);
        StartTraining($"Training on {path.GetFile()}", () => DatasetLoader.LoadDirectory(path),
            $"folder '{path.GetFile()}'", epochs, fromScratch);
    }

    /// <summary>Generates synthetic fluorescence data in memory and trains on it: a demo that needs no download.</summary>
    public void StartSyntheticTraining(int epochs, bool fromScratch)
    {
        StartTraining("Training on synthetic data", () => SyntheticFluorescence.GenerateDataset(8, 192, seed: System.Environment.TickCount),
            "8 synthetic 192px FOVs (generated in-app)", epochs, fromScratch);
    }

    /// <summary>Requests cancellation of the running job (if any). Training stops after its current step and keeps its best weights.</summary>
    public void Cancel() => _cancellation?.Cancel();

    /// <summary>Saves the current weights (including online adaptation) to <see cref="UserWeightsPath"/>.</summary>
    public bool SaveCheckpoint()
    {
        if (_busy) return false;
        try
        {
            string provenance = (_info?.Provenance ?? "untrained") + (_adapter is { OptimizerStep: > 0 } a ? $"; + {a.OptimizerStep} online N2V steps" : "");
            string path = ProjectSettings.GlobalizePath(UserWeightsPath);
            WeightsFile.Save(path, _net, provenance);
            _info = WeightsFile.Load(path).Info;
            _modelSource = UserWeightsPath;
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
    /// Shared body of both training entry points. Trains a <em>copy</em> of the current model, so the on-screen model
    /// stays intact if training is cancelled or fails, and swaps it in only when an epoch produced weights.
    /// Fine-tuning uses a lower learning rate (3e-4) than training from scratch (1e-3).
    /// Small per-epoch settings (30 steps × batch 8) keep the UI's progress updates frequent.
    /// </summary>
    private void StartTraining(string activity, Func<IReadOnlyList<ImagePair>> loadData, string description, int epochs, bool fromScratch)
    {
        var net = fromScratch ? DenoisingNet.Create(_net.Config, seed: System.Environment.TickCount) : _net.Clone();
        string weightsPath = ProjectSettings.GlobalizePath(UserWeightsPath);
        string journalPath = ProjectSettings.GlobalizePath(JournalPath);

        RunJob(activity, token =>
        {
            var dataset = loadData();   // lazy: datasets are only read when training is requested
            bool supervised = dataset.All(d => d.HasClean);
            var config = new TrainerConfig
            {
                Mode = supervised ? TrainingMode.Supervised : TrainingMode.Noise2Void,
                MaxEpochs = Math.Max(1, epochs),
                StepsPerEpoch = 30,
                BatchSize = 8,
                PatchSize = Math.Min(64, dataset.Min(d => Math.Min(d.Noisy.Width, d.Noisy.Height)) / 16 * 16),
                LearningRate = fromScratch ? 1e-3f : 3e-4f,
            };
            Defer(() => WriteLog($"📚 {dataset.Count} images, {config.Mode}, up to {config.MaxEpochs} epochs"));

            var result = TrainingJob.Run(net, dataset, config,
                new TrainingJobOptions { DatasetDescription = description, WeightsPath = weightsPath, JournalPath = journalPath },
                report => Defer(() => EmitSignal(SignalName.EpochCompleted, report.Epoch, (float)report.ValLoss,
                    (float)(report.ValPsnr ?? double.NaN), (float)(report.ValPsnrGain ?? double.NaN),
                    report.LearningRate, string.Join("  ", report.Events))),
                token);

            return () =>
            {
                if (result.BestEpoch > 0)
                {
                    SetModel(net, WeightsFile.Load(weightsPath).Info, UserWeightsPath);
                    EmitSignal(SignalName.TrainingFinished,
                        $"stopped: {result.StopReason} after {result.History.Count} epochs; best epoch {result.BestEpoch}" +
                        (result.Best?.ValPsnrGain is { } g ? $" (+{g:0.00} dB over noisy)" : "") +
                        $"; weights saved to {UserWeightsPath}");
                }
                else
                {
                    EmitSignal(SignalName.TrainingFinished, $"stopped: {result.StopReason} before any epoch finished; model unchanged");
                }
            };
        });
    }

    /// <summary>Installs a model and announces it. Main thread only.</summary>
    private void SetModel(DenoisingNet net, ModelInfo? info, string source)
    {
        _net = net;
        _info = info;
        _modelSource = source;
        _adapter = null;      // optimizer state belongs to the previous weights
        _unsaved = false;
        EmitSignal(SignalName.ModelChanged, DescribeModel());
    }

    /// <summary>Runs <paramref name="work"/> on the thread pool, then its returned continuation on the main thread.</summary>
    private void RunJob(string activity, Func<CancellationToken, Action> work)
    {
        if (_busy) { EmitSignal(SignalName.JobFailed, "DORA is busy; wait or cancel first."); return; }
        _busy = true;
        _cancellation = new CancellationTokenSource();
        var token = _cancellation.Token;
        EmitSignal(SignalName.BusyChanged, true, activity);

        Task.Run(() =>
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

using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;

namespace DoraBio.Core;

/// <summary>A network with what its weights file said about it (from <see cref="WeightsFile.Load(string)"/>).</summary>
/// <param name="Net">The network.</param>
/// <param name="Info">Its architecture, version, provenance and save time.</param>
public sealed record LoadedModel(DenoisingNet Net, ModelInfo Info)
{
    /// <summary>Loads a weights file.</summary>
    public static LoadedModel Load(string path)
    {
        var (net, info) = WeightsFile.Load(path);
        return new LoadedModel(net, info);
    }
}

/// <summary>
/// One <c>denoise</c> run, the same for the CLI and the app (brick 41): an image file (TIFF or BigTIFF, single page or
/// stack; MRC; PNG) to a float32 TIFF, or a folder of them to a folder. Each model is given either as a weights file or
/// already loaded (the app reads its bundled model from inside its package, where there is no file path).
/// </summary>
public sealed record DenoiseRequest
{
    /// <summary>An image file or a folder of images.</summary>
    public required string Input { get; init; }

    /// <summary>The output file, or for a folder the output folder (same file names).</summary>
    public required string Output { get; init; }

    /// <summary>The main model's weights file. Ignored when <see cref="Model"/> is given.</summary>
    public string? Weights { get; init; }

    /// <summary>The main model, already loaded.</summary>
    public LoadedModel? Model { get; init; }

    /// <summary>Optional second model for planes whose noise check says correlated, as a file.</summary>
    public string? CorrelatedWeights { get; init; }

    /// <summary>The second model, already loaded.</summary>
    public LoadedModel? CorrelatedModel { get; init; }

    /// <summary>Optional: also write the bilateral baseline here (a file, or a folder for a folder input).</summary>
    public string? Baseline { get; init; }

    /// <summary>Declared axes ending in YX (e.g. "TYX"), or null to use the file's own labels.</summary>
    public string? Axes { get; init; }

    /// <summary>The models were trained on variance-stabilized data.</summary>
    public bool Vst { get; init; }

    /// <summary>Tile core size for tiled inference.</summary>
    public int TileCore { get; init; } = TiledInference.DefaultTileCore;

    /// <summary>
    /// Optional: write a self-contained HTML report here (<see cref="DenoiseReport"/>): what was done, with which model,
    /// and the residual to look at before trusting the output. One image or stack only, not a folder.
    /// </summary>
    public string? Report { get; init; }
}

/// <summary>What a <see cref="Pipeline.Denoise"/> run did.</summary>
/// <param name="Files">Files denoised (1 for a single file).</param>
/// <param name="Planes">Planes denoised over all files.</param>
/// <param name="CorrelatedPlanes">Planes that went to the correlated-noise model.</param>
/// <param name="Seconds">Wall-clock time of the denoising (models already loaded).</param>
/// <param name="Notices">Every notice raised (e.g. unlabelled axes taken as T), in order.</param>
public sealed record DenoiseOutcome(int Files, int Planes, int CorrelatedPlanes, double Seconds, IReadOnlyList<string> Notices);

/// <summary>
/// One <c>train</c> run, the same for the CLI and the app (brick 41). Every setting left null takes the recipe the
/// quality passes chose (<see cref="Recipes"/>), exactly as a CLI option left out does; the CLI builds this record from
/// its options and nothing else. The data is a folder or stack file (<see cref="Data"/>), an external dataset
/// (<see cref="Ext"/>) or images already in memory (<see cref="Dataset"/>, the app's synthetic demo).
/// </summary>
public sealed record TrainRequest
{
    /// <summary>A folder of images (any layout <see cref="DatasetLoader"/> reads), a stack file or a folder of stacks
    /// (time-lapse modes), or with <see cref="Ext"/> the external datasets' root (default data/ext).</summary>
    public string? Data { get; init; }

    /// <summary>An external dataset's name: train on its training split.</summary>
    public string? Ext { get; init; }

    /// <summary>Normalized images already in memory, instead of <see cref="Data"/>.</summary>
    public IReadOnlyList<ImagePair>? Dataset { get; init; }

    /// <summary>What to call in-memory <see cref="Dataset"/> images in the description (default "memory").</summary>
    public string? DatasetName { get; init; }

    /// <summary>Where the weights are written (the independent-noise network when training is routed).</summary>
    public required string Out { get; init; }

    /// <summary>Routed training: where the correlated-noise network goes (default: next to <see cref="Out"/>).</summary>
    public string? OutCorrelated { get; init; }

    /// <summary>Journey log file, or null.</summary>
    public string? Journey { get; init; }

    /// <summary>Overrides the description stored in the weights' provenance.</summary>
    public string? Describe { get; init; }

    /// <summary>Also write the last epoch's weights here.</summary>
    public string? SaveLast { get; init; }

    /// <summary>n2v, supervised, n2n or skip-centre; null decides from the data (supervised when every image has a clean target).</summary>
    public string? Mode { get; init; }

    /// <summary>Input frames (3: the spatiotemporal blind spot).</summary>
    public int? Frames { get; init; }

    /// <summary>Train on these frames of a time-lapse only.</summary>
    public FrameRange? FrameRange { get; init; }

    /// <summary>Validate on these frames (disjoint from <see cref="FrameRange"/>).</summary>
    public FrameRange? ValidationRange { get; init; }

    /// <summary>Frames per time-lapse block (the unit of the train/validation split).</summary>
    public int? FrameBlock { get; init; }

    /// <summary>Declared axes of the stacks, ending in YX.</summary>
    public string? Axes { get; init; }

    /// <summary>Variance-stabilize each image (Noise2Void only).</summary>
    public bool Vst { get; init; }

    /// <summary>FMD layout: raw frames per FOV.</summary>
    public int? FmdFrames { get; init; }

    /// <summary>FMD layout: comma-separated noise levels.</summary>
    public string? FmdLevels { get; init; }

    /// <summary>FMD layout: comma-separated microscope categories.</summary>
    public string? FmdCategories { get; init; }

    /// <summary>Most epochs (default 30).</summary>
    public int? Epochs { get; init; }

    /// <summary>Steps per epoch (default 50).</summary>
    public int? Steps { get; init; }

    /// <summary>Batch size (default 16).</summary>
    public int? Batch { get; init; }

    /// <summary>Augmentation: dihedral (default) or v1.</summary>
    public string? Augment { get; init; }

    /// <summary>Patch size in pixels (default 64).</summary>
    public int? Patch { get; init; }

    /// <summary>Noise2Void mask ratio (default 0.02).</summary>
    public double? MaskRatio { get; init; }

    /// <summary>Learning rate (default 1e-3).</summary>
    public double? LearningRate { get; init; }

    /// <summary>Seed for the network and the sampler (default 1234).</summary>
    public int? Seed { get; init; }

    /// <summary>Early-stop patience; 0 never stops early.</summary>
    public int? Patience { get; init; }

    /// <summary>What picks the kept epoch: auto or loss.</summary>
    public string? Select { get; init; }

    /// <summary>Which weights are kept: best or last.</summary>
    public string? Keep { get; init; }

    /// <summary>Noise2Void routing: auto or off.</summary>
    public string? Route { get; init; }

    /// <summary>Fine-tune this weights file instead of starting from scratch.</summary>
    public string? Init { get; init; }

    /// <summary>Fine-tune this network (a copy is trained; the caller's stays as it was). Counts as <see cref="Init"/>.</summary>
    public DenoisingNet? InitNet { get; init; }

    /// <summary>Network: dilated or unet.</summary>
    public string? Arch { get; init; }

    /// <summary>Feature channels.</summary>
    public int? Features { get; init; }

    /// <summary>U-Net depth.</summary>
    public int? Depth { get; init; }

    /// <summary>Conditioning: none or noise.</summary>
    public string? Condition { get; init; }

    /// <summary>Structured blind spots: off, auto or always.</summary>
    public string? Struct { get; init; }

    /// <summary>Structured blind-spot footprint (e.g. diamond-2).</summary>
    public string? Footprint { get; init; }

    /// <summary>Train only on the images the noise check calls correlated or independent.</summary>
    public string? Subset { get; init; }

    /// <summary>
    /// Whether a recipe option (<see cref="Recipes.RecipeOptions"/>) was set, which turns Noise2Void's default routing off,
    /// as giving one of those options to the CLI does.
    /// </summary>
    public bool ChoosesRecipe =>
        Vst || new object?[] { Arch, Features, Depth, Select, Patience, Keep, Struct, Footprint, Subset, Init, InitNet, Condition, SaveLast }
            .Any(option => option is not null);
}

/// <summary>One network a <see cref="Pipeline.Train"/> run wrote.</summary>
/// <param name="Kind">The noise it was routed to, or null when training wasn't routed.</param>
/// <param name="WeightsPath">Where its weights are.</param>
/// <param name="Images">How many training images it saw.</param>
/// <param name="Result">Its training history.</param>
/// <param name="KeptLast">Whether the last epoch was kept (rather than the best).</param>
public sealed record TrainedModel(NoiseKind? Kind, string WeightsPath, int Images, TrainingResult Result, bool KeptLast)
{
    /// <summary>The epoch whose weights were written.</summary>
    public EpochReport? Kept => KeptLast ? Result.Last : Result.Best;
}

using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

/// <summary>One evaluation of the validation set.</summary>
/// <param name="Loss">Mean blind-spot MSE against noisy pixels (Noise2Void) or MSE against clean (supervised).</param>
/// <param name="Psnr">PSNR of DORA's output on the unmasked input against clean, in dB; null without ground truth.</param>
/// <param name="Ssim">Mean SSIM against clean; null without ground truth.</param>
public sealed record ValidationScore(double Loss, double? Psnr, double? Ssim);

/// <summary>
/// A fixed set of validation patches from held-out FOVs, extracted once with a fixed seed so every epoch
/// is scored on the same pixels.
/// Loss: Noise2Void mode uses the masked blind-spot loss against the noisy pixels; supervised mode uses MSE
/// against clean. PSNR and SSIM are measured on the unmasked noisy input against clean, when clean exists.
/// </summary>
internal sealed class ValidationSet
{
    private readonly List<Patch> _patches = new();
    private readonly int _size;
    private readonly TrainingMode _mode;

    /// <summary>
    /// One fixed validation patch. The blind-spot mask is drawn once, so the Noise2Void loss is comparable
    /// from epoch to epoch.
    /// </summary>
    /// <param name="Noisy">The frame's noisy patch (the Noise2Void target).</param>
    /// <param name="Input">The network's unmasked input: Noisy, or for a multi-frame network every frame's patch in time order.</param>
    /// <param name="MaskedInput">Input with the blind spots in the centre frame.</param>
    /// <param name="Mask">The blind spots.</param>
    /// <param name="Clean">Ground truth (or, for Noise2Noise, the neighbouring frame it is trained towards), or null.</param>
    /// <param name="Condition">The whole frame's noise condition.</param>
    private sealed record Patch(float[] Noisy, float[] Input, float[] MaskedInput, float[] Mask, float[]? Clean, float Condition);

    /// <summary>True when every validation patch has ground truth, so PSNR, SSIM and progressive targets are available.</summary>
    public bool HasClean { get; }

    /// <summary>PSNR of the noisy input itself against clean, i.e. the "do nothing" baseline.</summary>
    public double? NoisyPsnr { get; }

    /// <summary>The field of view (<see cref="ImagePair.GroupId"/>) each patch was cut from, in patch order.</summary>
    internal IReadOnlyList<string> PatchSources { get; }

    /// <summary>Each patch's top-left corner in its image, in patch order (with <see cref="PatchSources"/>: where every patch came from).</summary>
    internal IReadOnlyList<(int X, int Y)> PatchOrigins { get; }

    /// <summary>
    /// Extracts <see cref="TrainerConfig.ValidationPatches"/> patches spread over the held-out fields of view: the FOVs
    /// are visited in a seeded random order, one patch each (from a random frame of that FOV), starting over once every
    /// FOV has one. Positions and choices come from a seed derived from <see cref="TrainerConfig.Seed"/>.
    /// </summary>
    /// <remarks>
    /// Until P3 the patches cycled through the images in list order. Datasets are listed by category, so on FMD all
    /// 16 patches came from 4 FOVs of a single confocal category, and every training decision (best epoch, learning
    /// rate decay, early stopping, targets) was made without seeing a two-photon or widefield image.
    /// </remarks>
    public ValidationSet(IReadOnlyList<ImagePair> images, TrainerConfig config)
    {
        _mode = config.Mode;
        int minSide = images.Min(i => Math.Min(i.Noisy.Width, i.Noisy.Height));
        _size = Math.Min(config.PatchSize, minSide);
        if (_size < 16) throw new ArgumentException("Validation images must be at least 16×16.");

        var rng = new Random(config.Seed ^ 0x5EED);
        var fovs = images.GroupBy(i => i.GroupId).OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => g.ToList()).OrderBy(_ => rng.Next()).ToList();
        var sources = new List<string>();
        var origins = new List<(int X, int Y)>();
        for (int k = 0; k < config.ValidationPatches; k++)
        {
            var fov = fovs[k % fovs.Count];
            var image = fov[rng.Next(fov.Count)];
            sources.Add(image.GroupId);
            int x = rng.Next(image.Noisy.Width - _size + 1);
            int y = rng.Next(image.Noisy.Height - _size + 1);
            origins.Add((x, y));
            var noisy = image.Noisy.Crop(x, y, _size, _size).Pixels;
            var clean = image.Clean?.Crop(x, y, _size, _size).Pixels;
            bool skipCentre = _mode == TrainingMode.SkipCentre;
            if (skipCentre && image.Context is not { Length: 2 })
                throw new ArgumentException("Skip-centre validation needs each frame's two neighbours (t − 1, t + 1) as its context.");
            int plane = noisy.Length, frames = skipCentre ? 2 : image.FrameCount, center = frames / 2;
            var input = frames == 1 ? noisy : new float[frames * plane];
            var masked = new float[frames * plane];
            var mask = new float[plane];
            if (image.Context is { } context)   // neighbours, unmasked, around the centre frame (skip-centre: planes 0 and 1)
                for (int c = 0; c < context.Length; c++)
                {
                    int channel = skipCentre || c < context.Length / 2 ? c : c + 1;
                    var window = context[c].Crop(x, y, _size, _size).Pixels;
                    window.CopyTo(input, channel * plane);
                    window.CopyTo(masked, channel * plane);
                }
            if (frames > 1 && !skipCentre) noisy.CopyTo(input, center * plane);
            // The same blind spot as in training, so the Noise2Void validation loss scores what training optimizes. It is
            // drawn in every mode, even where it goes unused (skip-centre draws it into a scratch plane, never the input),
            // so the generator advances alike and every mode gets the same frames and positions from the same images.
            Span<float> blindTarget = skipCentre ? new float[plane] : masked.AsSpan(center * plane, plane);
            BlindSpot.Apply(noisy, blindTarget, mask, _size, config.MaskRatio, rng, footprint: image.Footprint);
            // The whole frame's condition, as in training and inference (read only by conditioned networks).
            _patches.Add(new Patch(noisy, input, masked, mask, clean, NoiseConditioning.Of(image.Noisy)));
        }

        PatchSources = sources;
        PatchOrigins = origins;
        bool allTargets = _patches.All(p => p.Clean is not null);
        if (_mode is TrainingMode.Supervised or TrainingMode.Noise2Noise && !allTargets)
            throw new ArgumentException(_mode == TrainingMode.Supervised
                ? "Supervised training needs clean validation images."
                : "Noise2Noise training needs a neighbouring frame for every validation image.");
        // A Noise2Noise target is another noisy frame, not ground truth: it gives a loss but no PSNR.
        HasClean = allTargets && _mode != TrainingMode.Noise2Noise;
        if (HasClean)
            NoisyPsnr = Metrics.PsnrFromMse(_patches.Average(p => Metrics.Mse(p.Noisy, p.Clean!)));
    }

    /// <summary>Mean number of input pixels the blind spots changed per blind spot (1 for plain, more for structured).</summary>
    internal double HiddenPixelsPerBlindSpot()
    {
        double changed = 0, centres = 0;
        foreach (var p in _patches)
            for (int i = 0; i < p.Noisy.Length; i++)
            {
                int centerOffset = p.MaskedInput.Length / p.Noisy.Length / 2 * p.Noisy.Length;
                if (p.MaskedInput[centerOffset + i] != p.Noisy[i]) changed++;
                if (p.Mask[i] == 1f) centres++;
            }
        return changed / Math.Max(1, centres);
    }

    /// <summary>Scores the network on every patch in parallel. Read-only with respect to the network.</summary>
    public ValidationScore Evaluate(DenoisingNet net)
    {
        var pool = new WorkspacePool(net);
        var losses = new double[_patches.Count];
        var mses = new double[_patches.Count];
        var ssims = new double[_patches.Count];

        Parallel.For(0, _patches.Count, () => pool.Rent(), (i, _, ws) =>
        {
            var p = _patches[i];
            var output = new float[p.Noisy.Length];

            if (_mode == TrainingMode.SkipCentre)
            {
                // Every pixel against the frame itself, which the input never held.
                net.Predict(p.Input, _size, _size, output, ws, p.Condition);
                losses[i] = Metrics.Mse(output, p.Noisy);
            }
            if (_mode == TrainingMode.Noise2Void)
            {
                net.Predict(p.MaskedInput, _size, _size, output, ws, p.Condition);
                double sse = 0, count = 0;
                for (int j = 0; j < output.Length; j++)
                {
                    if (p.Mask[j] == 0f) continue;
                    double d = output[j] - p.Noisy[j];
                    sse += d * d;
                    count++;
                }
                losses[i] = sse / Math.Max(1, count);
            }

            if (p.Clean is not null)
            {
                net.Predict(p.Input, _size, _size, output, ws, p.Condition);
                mses[i] = Metrics.Mse(output, p.Clean);
                ssims[i] = Metrics.Ssim(new GrayImage(_size, _size, output), new GrayImage(_size, _size, p.Clean));
                if (_mode is TrainingMode.Supervised or TrainingMode.Noise2Noise) losses[i] = mses[i];
            }
            return ws;
        }, pool.Return);

        return new ValidationScore(
            losses.Average(),
            HasClean ? Metrics.PsnrFromMse(mses.Average()) : null,
            HasClean ? ssims.Average() : null);
    }
}

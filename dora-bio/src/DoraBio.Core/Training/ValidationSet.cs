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
    private sealed record Patch(float[] Noisy, float[] MaskedInput, float[] Mask, float[]? Clean);

    /// <summary>True when every validation patch has ground truth, so PSNR, SSIM and progressive targets are available.</summary>
    public bool HasClean { get; }

    /// <summary>PSNR of the noisy input itself against clean, i.e. the "do nothing" baseline.</summary>
    public double? NoisyPsnr { get; }

    /// <summary>
    /// Extracts <see cref="TrainerConfig.ValidationPatches"/> patches, cycling through the held-out images,
    /// at positions drawn from a seed derived from <see cref="TrainerConfig.Seed"/>.
    /// </summary>
    public ValidationSet(IReadOnlyList<ImagePair> images, TrainerConfig config)
    {
        _mode = config.Mode;
        int minSide = images.Min(i => Math.Min(i.Noisy.Width, i.Noisy.Height));
        _size = Math.Min(config.PatchSize, minSide);
        if (_size < 16) throw new ArgumentException("Validation images must be at least 16×16.");

        var rng = new Random(config.Seed ^ 0x5EED);
        for (int k = 0; k < config.ValidationPatches; k++)
        {
            var image = images[k % images.Count];
            int x = rng.Next(image.Noisy.Width - _size + 1);
            int y = rng.Next(image.Noisy.Height - _size + 1);
            var noisy = image.Noisy.Crop(x, y, _size, _size).Pixels;
            var clean = image.Clean?.Crop(x, y, _size, _size).Pixels;
            var masked = new float[noisy.Length];
            var mask = new float[noisy.Length];
            BlindSpot.Apply(noisy, masked, mask, _size, config.MaskRatio, rng);
            _patches.Add(new Patch(noisy, masked, mask, clean));
        }

        HasClean = _patches.All(p => p.Clean is not null);
        if (_mode == TrainingMode.Supervised && !HasClean)
            throw new ArgumentException("Supervised training needs clean validation images.");
        if (HasClean)
            NoisyPsnr = Metrics.PsnrFromMse(_patches.Average(p => Metrics.Mse(p.Noisy, p.Clean!)));
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

            if (_mode == TrainingMode.Noise2Void)
            {
                net.Predict(p.MaskedInput, _size, _size, output, ws);
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
                net.Predict(p.Noisy, _size, _size, output, ws);
                mses[i] = Metrics.Mse(output, p.Clean);
                ssims[i] = Metrics.Ssim(new GrayImage(_size, _size, output), new GrayImage(_size, _size, p.Clean));
                if (_mode == TrainingMode.Supervised) losses[i] = mses[i];
            }
            return ws;
        }, pool.Return);

        return new ValidationScore(
            losses.Average(),
            HasClean ? Metrics.PsnrFromMse(mses.Average()) : null,
            HasClean ? ssims.Average() : null);
    }
}

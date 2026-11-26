using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;

namespace DoraBio.Core;

/// <summary>Scores for one test_mix image, on FMD's 8-bit scale (0–255).</summary>
/// <param name="Name">Image name, e.g. "test_mix/raw/Confocal_FISH_2.png".</param>
/// <param name="Category">FMD category, e.g. "Confocal_FISH".</param>
/// <param name="Modality">Microscope type: "Confocal", "TwoPhoton" or "WideField".</param>
/// <param name="NoisyPsnr">PSNR of the unprocessed noisy image (the "do nothing" baseline).</param>
/// <param name="NoisySsim">SSIM of the unprocessed noisy image.</param>
/// <param name="BilateralPsnr">PSNR after the classical bilateral filter.</param>
/// <param name="BilateralSsim">SSIM after the bilateral filter.</param>
/// <param name="DoraPsnr">PSNR after DORA.</param>
/// <param name="DoraSsim">SSIM after DORA.</param>
public sealed record FmdScore(string Name, string Category, string Modality,
    double NoisyPsnr, double NoisySsim, double BilateralPsnr, double BilateralSsim, double DoraPsnr, double DoraSsim);

/// <summary>
/// How strongly one test image's real noise is correlated between neighbouring pixels (see
/// <see cref="FmdBenchmark.NoiseCorrelation"/>). Noise2Void assumes all three correlations are 0.
/// </summary>
/// <param name="Name">Image name, e.g. "test_mix/raw/WideField_BPAE_B_1.png".</param>
/// <param name="Modality">Microscope type: "Confocal", "TwoPhoton" or "WideField".</param>
/// <param name="Right1">Correlation with the pixel one to the right.</param>
/// <param name="Down1">Correlation with the pixel one below.</param>
/// <param name="Right2">Correlation with the pixel two to the right.</param>
/// <param name="NoiseSd">Standard deviation of the noise (noisy − ground truth), in 8-bit units.</param>
public sealed record NoiseStats(string Name, string Modality, double Right1, double Down1, double Right2, double NoiseSd);

/// <summary>
/// Benchmarks a model on FMD's held-out test_mix set (FOV 19 of every category, never trained on).
///
/// Protocol: each noisy image is normalized exactly as at inference time (its own 0.1/99.9% percentiles),
/// denoised, mapped back to the original 8-bit units, and clipped to [0, 255]. It is then compared with the
/// 50-frame-average ground truth. PSNR and SSIM use a data range of 255, the convention for FMD's 8-bit images,
/// so numbers are comparable with other FMD results. The bilateral filter and the raw input are scored the same way.
/// Everything is deterministic, so a recorded result can be re-checked exactly (bricks.sh gate 13).
/// </summary>
public static class FmdBenchmark
{
    /// <summary>FMD images are 8-bit.</summary>
    public const double DataRange = 255;

    /// <summary>Microscope type from a category name: "TwoPhoton_MICE" → "TwoPhoton".</summary>
    public static string ModalityOf(string category)
    {
        int underscore = category.IndexOf('_');
        return underscore > 0 ? category[..underscore] : category;
    }

    /// <summary>Scores every test_mix image at one noise level ("raw" is the hardest).</summary>
    /// <param name="net">The model to evaluate.</param>
    /// <param name="testMixDir">The test_mix folder.</param>
    /// <param name="level">Noise level: raw, avg2, avg4, avg8 or avg16.</param>
    /// <param name="tileCore">Tile core size for inference.</param>
    public static List<FmdScore> Evaluate(DenoisingNet net, string testMixDir, string level = "raw",
        int tileCore = TiledInference.DefaultTileCore) => Evaluate(_ => net, testMixDir, level, tileCore);

    /// <summary>
    /// Like <see cref="Evaluate(DenoisingNet, string, string, int)"/>, but <paramref name="pickNet"/> chooses the model for
    /// each image from the noisy image alone (for example by its noise check), the way a deployed system would have to.
    /// </summary>
    /// <param name="pickNet">Given an image's noisy pixels (original units), returns the model that denoises it.</param>
    /// <param name="testMixDir">The test_mix folder.</param>
    /// <param name="level">Noise level: raw, avg2, avg4, avg8 or avg16.</param>
    /// <param name="tileCore">Tile core size for inference.</param>
    /// <param name="stabilize">
    /// For models trained with variance stabilization: calibrate each image (<see cref="NoiseCalibration"/>), denoise
    /// its <see cref="Anscombe"/> transform, and map the result back with the unbiased inverse. The noisy and bilateral
    /// columns are unaffected.
    /// </param>
    /// <param name="sigmaScale">Noise-conditioned models only: multiply each image's measured noise σ by this before it
    /// becomes the condition. 1 is the honest value; 2 and 0.5 test whether the model uses its condition at all (Q13).</param>
    public static List<FmdScore> Evaluate(Func<GrayImage, DenoisingNet> pickNet, string testMixDir, string level = "raw",
        int tileCore = TiledInference.DefaultTileCore, bool stabilize = false, double sigmaScale = 1) =>
        EvaluatePlan(pickNet, FmdDataset.PlanTestMix(testMixDir, level), tileCore, stabilize, sigmaScale);

    /// <summary>
    /// The same protocol on any planned pairs, e.g. <see cref="ValidationPlan"/>: per image, normalize, denoise with the
    /// model <paramref name="pickNet"/> picks, map back to 8-bit units, and score against ground truth.
    /// </summary>
    /// <param name="pickNet">Given an image's noisy pixels (original units), returns the model that denoises it.</param>
    /// <param name="plan">The pairs to score.</param>
    /// <param name="tileCore">Tile core size for inference.</param>
    /// <param name="stabilize">Denoise through the variance-stabilized domain (models trained with --vst).</param>
    /// <param name="sigmaScale">Noise-conditioned models only: scales the measured σ behind each image's condition.</param>
    public static List<FmdScore> EvaluatePlan(Func<GrayImage, DenoisingNet> pickNet, IReadOnlyList<FmdDataset.PlannedPair> plan,
        int tileCore = TiledInference.DefaultTileCore, bool stabilize = false, double sigmaScale = 1)
    {
        var scores = new List<FmdScore>();
        foreach (var p in plan)
        {
            var noisy = ImageLoader.Load(p.NoisyPath);
            var clean = ImageLoader.Load(p.CleanPath);
            var map = Normalization.FromPercentiles(noisy.Pixels);
            var input = Normalization.Apply(noisy, map);
            var net = pickNet(noisy);
            var dora = stabilize
                ? DenoiseStabilized(net, noisy, tileCore)
                : ToSensorUnits(net.Config.IsConditioned && sigmaScale != 1
                    ? TiledInference.Denoise(net, input, tileCore, default, new WorkspacePool(net),
                        NoiseConditioning.FromSigma(Metrics.EstimateNoiseSigma(input) * sigmaScale))
                    : TiledInference.Denoise(net, input, tileCore), map);
            var bilateral = ToSensorUnits(BilateralBaseline.FilterAuto(input), map);

            string category = p.GroupId.Split('/')[0];
            scores.Add(new FmdScore(p.Name, category, ModalityOf(category),
                Metrics.Psnr(noisy.Pixels, clean.Pixels, DataRange), Metrics.Ssim(noisy, clean, DataRange),
                Metrics.Psnr(bilateral.Pixels, clean.Pixels, DataRange), Metrics.Ssim(bilateral, clean, DataRange),
                Metrics.Psnr(dora.Pixels, clean.Pixels, DataRange), Metrics.Ssim(dora, clean, DataRange)));
        }
        return scores;
    }

    /// <summary>
    /// Mean scores per modality and overall, in a fixed order (Confocal, TwoPhoton, WideField, then "all").
    /// PSNR is averaged per image, as FMD results usually are.
    /// </summary>
    public static List<(string Group, int Count, FmdScore Mean)> Summarize(IReadOnlyList<FmdScore> scores) =>
        ByModality(scores, s => s.Modality).Select(g => (g.Group, g.Items.Count, new FmdScore(g.Group, g.Group, g.Group,
            g.Items.Average(s => s.NoisyPsnr), g.Items.Average(s => s.NoisySsim),
            g.Items.Average(s => s.BilateralPsnr), g.Items.Average(s => s.BilateralSsim),
            g.Items.Average(s => s.DoraPsnr), g.Items.Average(s => s.DoraSsim)))).ToList();

    /// <summary>
    /// Pearson correlation between the noise at each pixel and the noise <paramref name="dx"/> pixels right and
    /// <paramref name="dy"/> pixels down, where noise n = noisy − clean with its mean removed:
    /// ρ = Σ n(x,y)·n(x+dx,y+dy) / √(Σ n(x,y)² · Σ n(x+dx,y+dy)²), over every pair of pixels inside the image.
    /// </summary>
    /// <remarks>
    /// Noise2Void's blind spot hides a pixel from the network and asks it to predict that pixel from its neighbours.
    /// That only removes noise if the neighbours carry no information about the hidden pixel's noise, i.e. ρ ≈ 0 at
    /// every nonzero offset. If ρ is large, the network learns to predict the noise too, and keeps it. For scale:
    /// averaging independent noise over two horizontal neighbours gives ρ = 0.5 at (1, 0) and 0 at (2, 0).
    /// </remarks>
    /// <param name="noisy">The noisy image.</param>
    /// <param name="clean">Its ground truth, same size.</param>
    /// <param name="dx">Horizontal offset, at least 0 and less than the width.</param>
    /// <param name="dy">Vertical offset, at least 0 and less than the height. (dx, dy) must not be (0, 0).</param>
    public static double NoiseCorrelation(GrayImage noisy, GrayImage clean, int dx, int dy)
    {
        if (noisy.Width != clean.Width || noisy.Height != clean.Height)
            throw new ArgumentException("Noisy and clean images differ in size.");
        if (dx < 0 || dy < 0 || (dx == 0 && dy == 0) || dx >= noisy.Width || dy >= noisy.Height)
            throw new ArgumentOutOfRangeException(nameof(dx), "Offsets must be non-negative, smaller than the image, and not both 0.");

        var noise = new double[noisy.Length];
        double mean = 0;
        for (int i = 0; i < noise.Length; i++) mean += noise[i] = noisy.Pixels[i] - clean.Pixels[i];
        mean /= noise.Length;

        int w = noisy.Width;
        double cross = 0, sumA = 0, sumB = 0;
        for (int y = 0; y + dy < noisy.Height; y++)
            for (int x = 0; x + dx < w; x++)
            {
                double a = noise[y * w + x] - mean, b = noise[(y + dy) * w + x + dx] - mean;
                cross += a * b;
                sumA += a * a;
                sumB += b * b;
            }
        return sumA > 0 && sumB > 0 ? cross / Math.Sqrt(sumA * sumB) : 0;
    }

    /// <summary>Noise correlations and noise level of every test_mix image at one noise level (no model involved).</summary>
    /// <param name="testMixDir">The test_mix folder.</param>
    /// <param name="level">Noise level: raw, avg2, avg4, avg8 or avg16.</param>
    public static List<NoiseStats> MeasureNoise(string testMixDir, string level = "raw") =>
        MeasureNoise(FmdDataset.PlanTestMix(testMixDir, level));

    /// <summary>Noise correlations and noise level of every image of a plan (e.g. <see cref="ValidationPlan"/>).</summary>
    public static List<NoiseStats> MeasureNoise(IReadOnlyList<FmdDataset.PlannedPair> plan)
    {
        var stats = new List<NoiseStats>();
        foreach (var p in plan)
        {
            var noisy = ImageLoader.Load(p.NoisyPath);
            var clean = ImageLoader.Load(p.CleanPath);
            double mean = 0, squares = 0;
            for (int i = 0; i < noisy.Length; i++)
            {
                double n = noisy.Pixels[i] - clean.Pixels[i];
                mean += n;
                squares += n * n;
            }
            mean /= noisy.Length;
            double sd = Math.Sqrt(Math.Max(0, squares / noisy.Length - mean * mean));
            stats.Add(new NoiseStats(p.Name, ModalityOf(p.GroupId), NoiseCorrelation(noisy, clean, 1, 0),
                NoiseCorrelation(noisy, clean, 0, 1), NoiseCorrelation(noisy, clean, 2, 0), sd));
        }
        return stats;
    }

    /// <summary>Mean noise statistics per modality and overall, in the same order as <see cref="Summarize"/>.</summary>
    public static List<(string Group, int Count, NoiseStats Mean)> SummarizeNoise(IReadOnlyList<NoiseStats> stats) =>
        ByModality(stats, s => s.Modality).Select(g => (g.Group, g.Items.Count, new NoiseStats(g.Group, g.Group,
            g.Items.Average(s => s.Right1), g.Items.Average(s => s.Down1), g.Items.Average(s => s.Right2),
            g.Items.Average(s => s.NoiseSd)))).ToList();

    /// <summary>
    /// The FOVs a training run held out for validation, at the given noise levels: exactly the groups
    /// <see cref="Training.Trainer"/> withholds for <paramref name="seed"/> and <paramref name="fraction"/> (the split depends
    /// only on the set of FOVs, not on how many frames or levels were loaded). For choosing settings without touching
    /// test_mix. FOV 19 is never among them.
    /// </summary>
    /// <param name="root">The FMD root.</param>
    /// <param name="seed">The training run's seed.</param>
    /// <param name="levels">Noise levels to include for each held-out FOV.</param>
    /// <param name="framesPerFov">Frames per FOV and level.</param>
    /// <param name="fraction">The training run's validation fraction.</param>
    public static List<FmdDataset.PlannedPair> ValidationPlan(string root, int seed, IReadOnlyList<string> levels,
        int framesPerFov = 1, double fraction = 0.2)
    {
        var groups = FmdDataset.PlanTraining(root, new FmdOptions()).Select(p => p.GroupId).Distinct().ToList();
        var placeholder = new GrayImage(1, 1);
        var (_, held) = DatasetSplit.Split(groups.Select(g => new ImagePair(placeholder, null, g, g)).ToList(), fraction, seed);
        var heldOut = held.Select(p => p.GroupId).ToHashSet();
        return FmdDataset.PlanTraining(root, new FmdOptions { FramesPerFov = framesPerFov, NoiseLevels = levels.ToArray() })
            .Where(p => heldOut.Contains(p.GroupId)).ToList();
    }

    /// <summary>Groups by modality in ordinal order, then appends everything as "all".</summary>
    private static List<(string Group, List<T> Items)> ByModality<T>(IReadOnlyList<T> items, Func<T, string> modality)
    {
        var groups = items.GroupBy(modality).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => (g.Key, g.ToList())).ToList();
        groups.Add(("all", items.ToList()));
        return groups;
    }

    /// <summary>Calibrate, transform, normalize, denoise, then undo each step (unbiased inverse last), clipped to 8 bits.</summary>
    private static GrayImage DenoiseStabilized(DenoisingNet net, GrayImage noisy, int tileCore)
    {
        var (output, _) = VarianceStabilization.Denoise(net, noisy, tileCore);
        for (int i = 0; i < output.Length; i++) output.Pixels[i] = (float)Math.Clamp(output.Pixels[i], 0, DataRange);
        return output;
    }

    /// <summary>Normalized → original units, clipped to the 8-bit range (evaluation only; training never clips).</summary>
    private static GrayImage ToSensorUnits(GrayImage normalized, IntensityMap map)
    {
        var result = Normalization.Invert(normalized, map);
        for (int i = 0; i < result.Length; i++) result.Pixels[i] = (float)Math.Clamp(result.Pixels[i], 0, DataRange);
        return result;
    }
}

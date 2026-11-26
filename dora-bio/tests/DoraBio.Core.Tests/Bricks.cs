using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Test-gate names for the brick-by-brick build (see ../../bricks.sh and dora-bio/CLAUDE.md).
/// Every test class carries exactly one brick trait. A brick's gate runs only its own tests;
/// bricks.sh runs the gates in order, so lower bricks are proven before higher ones are trusted.
/// </summary>
internal static class Bricks
{
    public const string Trait = "Brick";

    public const string Scaffold = "0-scaffold";   // project wiring: every test class belongs to a brick

    public const string Nn = "1-nn";               // ConvLayer, DenoisingNet, AdamOptimizer, Simd
    public const string Data = "2-data";           // GrayImage, Normalization, Dihedral, BlindSpot, PatchSampler, Synthetic
    public const string IO = "3-io";               // TiffCodec, PngDecoder, ImageLoader, WeightsFile
    public const string Inference = "4-inference"; // Metrics, TiledInference, BilateralBaseline
    public const string Training = "5-training";   // Trainer, ValidationSet, ProgressiveTargets, JourneyLog, OnlineAdapter, DatasetLoader
    public const string Fmd = "12-fmd";            // FmdDataset: real FMD layouts, test-FOV exclusion, leakage guard
    public const string FmdEval = "13-fmd-eval";   // FmdBenchmark: test_mix protocol (8-bit scale, per-modality summary)
    public const string NoiseDiagnostic = "14-n2v-diagnostic";   // NoiseDiagnostics: correlated noise, from the noisy image alone
    public const string StructuredMask = "15-structured-mask";   // structured blind spots for correlated noise
    public const string Vst = "16-vst";                          // generalized Anscombe transform and its inverse
    public const string Calibration = "17-calibration";          // gain, offset and read noise from the image
    public const string PoolUpsample = "18-pool-upsample";       // 2×2 average pooling and nearest upsampling
    public const string UNet = "19-unet";                        // U-Net architecture, weights format 2
    public const string UNetTiling = "20-unet-tiling";           // exact tiled inference for the U-Net
    public const string ArchEval = "21-arch-eval";               // dilated CNN vs U-Net on FMD
    public const string Bench = "22-bench";                      // timings for denoise and training steps
    public const string BlockedConv = "23-blocked-conv";         // register-blocked convolution kernels
    public const string PerfBudget = "24-perf-budget";           // time budgets that fail on regressions
    public const string TiffStackWrite = "26-tiff-stack-write";  // multi-page TIFF writer
    public const string NoiseRouting = "27-q10-routing";         // quality pass Q10: route by noise level
    public const string CliBatch = "28-cli-batch";               // streamed stack and folder denoising
    public const string NoiseConditioning = "29-q13-conditioning"; // quality pass Q13: FiLM noise conditioning
    public const string ExtData = "30-ext-data";                 // public paired datasets (field validation)
    public const string ExtBench = "31-q18-ext-bench";           // quality pass Q18: DORA vs CARE and Noise2Void
    public const string QuantFidelity = "32-quant-fidelity";     // proportionality, bias, invented/erased spots
    public const string SelfSupervisedRecipe = "33-q12-n2v-recipe"; // quality pass Q12: the Noise2Void recipe
}

[Trait(Bricks.Trait, Bricks.Scaffold)]
public class BrickWiringTests
{
    private static readonly HashSet<string> Known =
        [Bricks.Scaffold, Bricks.Nn, Bricks.Data, Bricks.IO, Bricks.Inference, Bricks.Training, Bricks.Fmd, Bricks.FmdEval,
         Bricks.NoiseDiagnostic, Bricks.StructuredMask, Bricks.Vst, Bricks.Calibration,
         Bricks.PoolUpsample, Bricks.UNet, Bricks.UNetTiling, Bricks.ArchEval,
         Bricks.Bench, Bricks.BlockedConv, Bricks.PerfBudget, Bricks.TiffStackWrite, Bricks.NoiseRouting, Bricks.CliBatch, Bricks.NoiseConditioning, Bricks.ExtData, Bricks.ExtBench, Bricks.QuantFidelity, Bricks.SelfSupervisedRecipe];

    /// <summary>An untagged test would silently run in no gate, so treat a missing or unknown brick as a build error.</summary>
    [Xunit.Fact]
    public void EveryTestClassBelongsToExactlyOneKnownBrick()
    {
        var testClasses = typeof(BrickWiringTests).Assembly.GetTypes()
            .Where(t => t.GetMethods().Any(m => m.GetCustomAttributes(typeof(Xunit.FactAttribute), inherit: true).Length > 0))
            .ToList();
        Xunit.Assert.NotEmpty(testClasses);

        foreach (var type in testClasses)
        {
            var bricks = type.GetCustomAttributesData()
                .Where(a => a.AttributeType == typeof(Xunit.TraitAttribute) && (string?)a.ConstructorArguments[0].Value == Bricks.Trait)
                .Select(a => (string?)a.ConstructorArguments[1].Value)
                .ToList();
            Xunit.Assert.True(bricks.Count == 1, $"{type.Name} must have exactly one [Trait(Bricks.Trait, ...)], found {bricks.Count}.");
            Xunit.Assert.True(Known.Contains(bricks[0]!), $"{type.Name} uses unknown brick '{bricks[0]}'.");
        }
    }
}


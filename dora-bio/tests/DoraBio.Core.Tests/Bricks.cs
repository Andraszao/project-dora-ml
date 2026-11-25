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
}

[Trait(Bricks.Trait, Bricks.Scaffold)]
public class BrickWiringTests
{
    private static readonly HashSet<string> Known =
        [Bricks.Scaffold, Bricks.Nn, Bricks.Data, Bricks.IO, Bricks.Inference, Bricks.Training, Bricks.Fmd];

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


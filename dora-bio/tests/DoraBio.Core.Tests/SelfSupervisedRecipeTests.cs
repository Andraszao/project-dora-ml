using DoraBio.Core.Data;
using DoraBio.Core.IO;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Quality pass Q12: settings are chosen on the training images a run holds out for validation, so the validation plan
/// must be exactly the trainer's own split, and the test split must never be part of it.
/// </summary>
[Trait(Bricks.Trait, Bricks.SelfSupervisedRecipe)]
public class SelfSupervisedRecipeTests
{
    [Theory]
    [InlineData(2026)]
    [InlineData(7)]
    public void TheValidationPlanIsExactlyWhatTheTrainerHoldsOut(int seed)
    {
        string root = TestUtil.TempDirectory();
        var mreb = ExtDatasets.Get("deepbacs-mreb");
        for (int i = 1; i <= 12; i++)
            foreach (var (folder, value) in new[] { (mreb.TrainNoisy, 10f), (mreb.TrainClean!, 40f), (mreb.TestNoisy!, 10f), (mreb.TestClean!, 40f) })
            {
                if (folder.StartsWith("test") && i > 3) continue;
                Directory.CreateDirectory(Path.Combine(root, mreb.Name, folder));
                var image = TestUtil.RandomImage(8, 8, 100 * i + (int)value);
                File.WriteAllBytes(Path.Combine(root, mreb.Name, folder, $"{(folder.StartsWith("test") ? "t" : "m")}{i}.tif"), TiffCodec.Encode(image, TiffSampleType.Float32));
            }

        foreach (bool supervised in new[] { true, false })
        {
            var images = ExtDatasets.LoadTraining(root, mreb, supervised);
            var (_, trainerHeldOut) = DatasetSplit.Split(images, 0.2, seed);
            var plan = ExtDatasets.ValidationPlan(root, mreb, seed);
            Assert.Equal(trainerHeldOut.Select(p => p.GroupId).Order(), plan.Select(p => p.GroupId).Order());
            Assert.NotEmpty(plan);
            Assert.All(plan, p => Assert.Contains("train", p.NoisyPath));
        }
        Directory.Delete(root, recursive: true);
    }
}

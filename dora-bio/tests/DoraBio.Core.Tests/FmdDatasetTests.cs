using DoraBio.Core.Data;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// FMD loader tests on a miniature dataset that mirrors the real download's layout (checked against the real 8.5 GB
/// tree): categories with raw/ and avgN/ holding numbered FOV folders, gt/FOV/avg50.png, and a flat test_mix copied
/// from FOV 19. No real data is needed, so these run everywhere, CI included.
/// </summary>
[Trait(Bricks.Trait, Bricks.Fmd)]
public class FmdDatasetTests : IDisposable
{
    private const int Size = 16;
    private static readonly string[] Categories = ["Confocal_FISH", "WideField_BPAE_B"];
    private static readonly int[] FovIds = [1, 2, 3, 19, 20];
    private readonly string _root = TestUtil.TempDirectory();

    public FmdDatasetTests()
    {
        foreach (string category in Categories)
            foreach (int fov in FovIds)
            {
                // Each FOV gets a distinct ground truth, so mix-ups between FOVs are detectable.
                byte Truth(int x, int y) => (byte)(20 + 7 * fov + x + y);
                TestUtil.WritePng(Path.Combine(_root, category, "gt", fov.ToString(), "avg50.png"), Size, Size, Truth);
                foreach (string level in FmdDataset.Levels)
                    for (int frame = 0; frame < 3; frame++)
                        TestUtil.WritePng(Path.Combine(_root, category, level, fov.ToString(), $"HV_P{frame:D3}.png"),
                            Size, Size, (x, y) => (byte)(Truth(x, y) + ((x * 31 + y * 17 + frame * 13) % 9) - 4));
            }

        // test_mix: two images per category, copied from FOV 19, named <Category>_<k>.png, like the real one.
        foreach (string category in Categories)
            for (int k = 1; k <= 2; k++)
            {
                File.Copy(Path.Combine(_root, category, "gt", "19", "avg50.png"), Target("gt"));
                foreach (string level in FmdDataset.Levels)
                    File.Copy(Path.Combine(_root, category, level, "19", $"HV_P{k:D3}.png"), Target(level));

                string Target(string folder)
                {
                    string path = Path.Combine(_root, "test_mix", folder, $"{category}_{k}.png");
                    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                    return path;
                }
            }
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void DefaultTrainingPlanExcludesTheTestFovAndOrdersFovsNumerically()
    {
        var plan = FmdDataset.PlanTraining(_root, new FmdOptions());

        Assert.DoesNotContain(plan, p => p.GroupId.EndsWith("/19"));
        Assert.DoesNotContain(plan, p => p.NoisyPath.Contains($"{Path.DirectorySeparatorChar}19{Path.DirectorySeparatorChar}"));
        // 2 categories × 4 FOVs × 3 frames available (FramesPerFov 4 caps at what exists) × raw only.
        Assert.Equal(2 * 4 * 3, plan.Count);
        var fovOrder = plan.Where(p => p.GroupId.StartsWith("Confocal_FISH/")).Select(p => p.GroupId).Distinct().ToList();
        Assert.Equal(["Confocal_FISH/1", "Confocal_FISH/2", "Confocal_FISH/3", "Confocal_FISH/20"], fovOrder);   // not 1,19,2,20,3
        Assert.All(plan, p => Assert.EndsWith("avg50.png", p.CleanPath));
        Assert.All(plan, p => Assert.Contains($"{Path.DirectorySeparatorChar}raw{Path.DirectorySeparatorChar}", p.NoisyPath));
    }

    [Fact]
    public void PlanHonorsFramesLevelsAndCategories()
    {
        var plan = FmdDataset.PlanTraining(_root, new FmdOptions
        {
            Categories = ["WideField_BPAE_B"],
            FramesPerFov = 2,
            NoiseLevels = ["raw", "avg4", "avg16"],
        });
        Assert.Equal(1 * 4 * 2 * 3, plan.Count);
        Assert.All(plan, p => Assert.StartsWith("WideField_BPAE_B/", p.GroupId));
        Assert.Equal(["raw", "avg4", "avg16"], plan.Select(p => p.Name.Split('/')[1]).Distinct());

        Assert.Throws<InvalidDataException>(() => FmdDataset.PlanTraining(_root, new FmdOptions { Categories = ["Nope"] }));
        Assert.Throws<ArgumentException>(() => FmdDataset.PlanTraining(_root, new FmdOptions { NoiseLevels = ["avg3"] }));
    }

    [Fact]
    public void ExcludingNothingIncludesTheTestFov()
    {
        // The guard is a default, not hard-coded: making it explicit keeps the leakage risk visible.
        var plan = FmdDataset.PlanTraining(_root, new FmdOptions { ExcludedFovs = [] });
        Assert.Contains(plan, p => p.GroupId.EndsWith("/19"));
    }

    [Fact]
    public void TestMixPairsByNameAtTheRequestedLevel()
    {
        var plan = FmdDataset.PlanTestMix(Path.Combine(_root, "test_mix"), "avg8");
        Assert.Equal(4, plan.Count);
        Assert.All(plan, p => Assert.Equal(Path.GetFileName(p.NoisyPath), Path.GetFileName(p.CleanPath)));
        Assert.All(plan, p => Assert.Contains($"{Path.DirectorySeparatorChar}avg8{Path.DirectorySeparatorChar}", p.NoisyPath));
        Assert.Equal(["Confocal_FISH", "WideField_BPAE_B"], plan.Select(p => p.GroupId).Distinct());
        Assert.Equal("TwoPhoton_BPAE_G", FmdDataset.CategoryOf("TwoPhoton_BPAE_G_3.png"));
    }

    [Fact]
    public void VerifyPassesOnAConsistentTreeAndCatchesLeakage()
    {
        var (problems, summary) = FmdDataset.Verify(_root);
        Assert.Empty(problems);
        Assert.Contains("2 categories, 10 FOVs", summary);

        // If test_mix came from a different FOV, excluding FOV 19 would not prevent leakage: Verify must say so.
        File.Copy(Path.Combine(_root, "Confocal_FISH", "gt", "3", "avg50.png"),
                  Path.Combine(_root, "test_mix", "gt", "Confocal_FISH_1.png"), overwrite: true);
        (problems, _) = FmdDataset.Verify(_root);
        Assert.Contains(problems, p => p.Contains("Confocal_FISH_1.png") && p.Contains("not FOV 19"));
    }

    [Fact]
    public void LoaderDetectsRootCategoryAndTestMix()
    {
        var all = DatasetLoader.LoadDirectory(_root);
        var one = DatasetLoader.LoadDirectory(Path.Combine(_root, "Confocal_FISH"));
        var test = DatasetLoader.LoadDirectory(Path.Combine(_root, "test_mix"));

        Assert.Equal(24, all.Count);
        Assert.Equal(12, one.Count);
        Assert.Equal(4, test.Count);
        Assert.All(all.Concat(one).Concat(test), p => Assert.True(p.HasClean));
        Assert.DoesNotContain(all, p => p.GroupId.EndsWith("/19"));
        // Normalized: the noisy image's 0.1–99.9% quantiles map to about 0 and 1.
        Assert.All(all, p => Assert.InRange(p.Noisy.Pixels.Average(), -0.5, 1.5));
    }
}

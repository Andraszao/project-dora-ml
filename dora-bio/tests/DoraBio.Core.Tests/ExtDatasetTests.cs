using DoraBio.Core.Data;
using DoraBio.Core.IO;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 30: the external datasets' layouts (miniature copies here; gate 30 checks the real downloads). Pairs are matched
/// by file name, never by position, so a missing or extra file is an error rather than a silent misalignment.
/// </summary>
[Trait(Bricks.Trait, Bricks.ExtData)]
public class ExtDatasetTests
{
    private static void Write(string path, float value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, TiffCodec.Encode(new GrayImage(8, 8, Enumerable.Repeat(value, 64).ToArray()), TiffSampleType.UInt16));
    }

    private static string MiniatureMreb(params string[] names)
    {
        string root = TestUtil.TempDirectory();
        foreach (var (name, i) in names.Select((n, i) => (n, i)))
        {
            Write(Path.Combine(root, "deepbacs-mreb", "train", "low_SNR", name), 100 + i);
            Write(Path.Combine(root, "deepbacs-mreb", "train", "high_SNR", name), 400 + i);
        }
        return root;
    }

    [Fact]
    public void PairsMatchByFileNameInOrdinalOrder()
    {
        string root = MiniatureMreb("m2.tif", "m10.tif", "m1.tif");
        var plan = ExtDatasets.PlanPairs(root, ExtDatasets.Get("deepbacs-mreb"), ExtSplit.Train);
        Assert.Equal(["m1.tif", "m10.tif", "m2.tif"], plan.Select(p => Path.GetFileName(p.NoisyPath)));
        Assert.All(plan, p => Assert.Equal(Path.GetFileName(p.NoisyPath), Path.GetFileName(p.CleanPath)));
        Assert.All(plan, p => Assert.Contains("high_SNR", p.CleanPath));
        Assert.Equal(["deepbacs-mreb/m1", "deepbacs-mreb/m10", "deepbacs-mreb/m2"], plan.Select(p => p.GroupId));

        // The clean partner is the image of the same name, whatever the listing order.
        var pairs = ExtDatasets.LoadPairs(root, ExtDatasets.Get("deepbacs-mreb"), ExtSplit.Train);
        Assert.Equal(3, pairs.Count);
        Directory.Delete(root, recursive: true);
    }

    [Fact]
    public void AnImageWithoutAPartnerIsAnError()
    {
        string root = MiniatureMreb("m1.tif", "m2.tif");
        File.Delete(Path.Combine(root, "deepbacs-mreb", "train", "high_SNR", "m2.tif"));
        Write(Path.Combine(root, "deepbacs-mreb", "train", "high_SNR", "m3.tif"), 1);
        var error = Assert.Throws<InvalidDataException>(() => ExtDatasets.PlanPairs(root, ExtDatasets.Get("deepbacs-mreb"), ExtSplit.Train));
        Assert.Contains("m2.tif", error.Message);
        Assert.Contains("m3.tif", error.Message);
        Directory.Delete(root, recursive: true);
    }

    [Fact]
    public void FolderNamesWithSpacesAndDatasetsWithoutGroundTruthWork()
    {
        string root = TestUtil.TempDirectory();
        var care = ExtDatasets.Get("zerocost-care2d");
        Write(Path.Combine(root, care.Name, care.TestNoisy!, "23.tif"), 5);
        Write(Path.Combine(root, care.Name, care.TestClean!, "23.tif"), 50);
        Assert.Single(ExtDatasets.PlanPairs(root, care, ExtSplit.Test));

        var bsub = ExtDatasets.Get("deepbacs-bsubtilis");
        Write(Path.Combine(root, bsub.Name, "train", "fr_1.tif"), 7);
        Assert.False(bsub.HasGroundTruth);
        Assert.Single(ExtDatasets.PlanNoisy(root, bsub, ExtSplit.Train));
        Assert.Empty(ExtDatasets.PlanNoisy(root, bsub, ExtSplit.Test));
        Assert.Throws<InvalidOperationException>(() => ExtDatasets.PlanPairs(root, bsub, ExtSplit.Train));

        Assert.False(ExtDatasets.IsPresent(root, care));
        File.WriteAllText(Path.Combine(root, care.Name, ".complete"), "sha");
        Assert.True(ExtDatasets.IsPresent(root, care));
        Directory.Delete(root, recursive: true);
    }

    [Fact]
    public void EveryDatasetIsDescribedAndNamedUniquely()
    {
        Assert.Equal(4, ExtDatasets.All.Count);
        Assert.Equal(ExtDatasets.All.Count, ExtDatasets.All.Select(d => d.Name).Distinct().Count());
        Assert.All(ExtDatasets.All, d =>
        {
            Assert.StartsWith("10.5281/zenodo.", d.Doi);
            Assert.False(string.IsNullOrWhiteSpace(d.Citation));
            Assert.True(d.TrainCount > 0);
            // A paired dataset has a clean folder for every noisy one; an unpaired one has none at all.
            if (!d.HasGroundTruth) Assert.Null(d.TestClean);
            if (d.HasGroundTruth && d.TestCount > 0) Assert.NotNull(d.TestClean);
        });
        var unknown = Assert.Throws<ArgumentException>(() => ExtDatasets.Get("nope"));
        Assert.Contains("deepbacs-mreb", unknown.Message);
    }
}

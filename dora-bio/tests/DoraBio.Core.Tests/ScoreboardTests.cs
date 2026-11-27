using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// Brick 38, the self-supervised scoreboard: the 2D sets derived from stack sets (pre-registered splits, planes of one
/// patch or cell kept in one validation group), FMD exported for other tools, and other tools' FMD outputs scored with
/// FMD's own protocol. Gate 38 checks the recorded results and their reading.
/// </summary>
[Trait(Bricks.Trait, Bricks.Scoreboard)]
public class ScoreboardTests
{
    [Fact]
    public void DerivedSetsComeFromDownloadableStackSetsAndAreFoundByName()
    {
        Assert.Equal(["mdc25-tribolium-2d", "biosr-ccps-2d"], ExtDatasets.Derived.Select(d => d.Name));
        foreach (var d in ExtDatasets.Derived)
        {
            Assert.Contains(d.Source, ExtStacks.All.Select(s => s.Name));
            Assert.DoesNotContain(d.Name, ExtDatasets.All.Select(s => s.Name));   // never in the manifest's registry
            Assert.Same(d, ExtDatasets.Get(d.Name));
            Assert.True(d.HasGroundTruth);
            Assert.Equal(("train/low_SNR", "train/high_SNR", "test/low_SNR", "test/high_SNR"), (d.TrainNoisy, d.TrainClean, d.TestNoisy, d.TestClean));
        }
        Assert.All(ExtDatasets.All, d => Assert.Null(d.Source));
        Assert.Throws<ArgumentException>(() => ExtDerived.Export(TestUtil.TempDirectory(), ExtDatasets.Get("deepbacs-mreb")));
        Assert.Throws<ArgumentException>(() => ExtDerived.Export(TestUtil.TempDirectory(), ExtDatasets.Get("biosr-ccps-2d")));   // source missing
    }

    [Theory]
    [InlineData("00001__z03.tif", "set/00001")]
    [InlineData("c041__L01_p0.tif", "set/c041")]
    [InlineData("m31.tif", "set/m31")]
    [InlineData("__odd.tif", "set/__odd")]
    public void TheGroupIsTheStemBeforeADoubleUnderscore(string file, string group) =>
        Assert.Equal(group, ExtDatasets.GroupOf(ExtDatasets.Get("deepbacs-mreb") with { Name = "set" }, file));

    [Fact]
    public void PlanesOfOnePatchNeverStraddleTheValidationSplit()
    {
        string root = TestUtil.TempDirectory();
        var set = ExtDatasets.Get("mdc25-tribolium-2d");
        for (int patch = 1; patch <= 10; patch++)
            for (int z = 0; z < 3; z++)
                foreach (string folder in new[] { "low_SNR", "high_SNR" })
                {
                    string path = Path.Combine(root, set.Name, "train", folder, $"{patch:00000}__z{z:00}.tif");
                    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                    File.WriteAllBytes(path, TiffCodec.Encode(TestUtil.RandomImage(8, 8, patch * 10 + z), TiffSampleType.Float32));
                }
        var plan = ExtDatasets.PlanPairs(root, set, ExtSplit.Train);
        Assert.Equal(10, plan.Select(p => p.GroupId).Distinct().Count());
        var held = ExtDatasets.ValidationPlan(root, set, 2026);
        Assert.NotEmpty(held);
        Assert.All(held.GroupBy(p => p.GroupId), g => Assert.Equal(3, g.Count()));   // a patch is held out whole or not at all
        var images = ExtDatasets.LoadTraining(root, set, supervised: false);
        Assert.Equal(plan.Select(p => p.GroupId), images.Select(i => i.GroupId));
        Directory.Delete(root, recursive: true);
    }

    [Fact]
    public void TheSplitsAreThePreRegisteredOnes()
    {
        var tribolium = ExtStacks.Get("mdc25-tribolium");
        var patches = Enumerable.Range(1, 15500).Select(i => new StackPair($"/d/mdc25-tribolium/noisy/{i:00000}.tif", $"/d/mdc25-tribolium/gt/{i:00000}.tif", tribolium.Groups[0])).ToList();
        var (train, test) = ExtDerived.Select("mdc25-tribolium-2d", patches);
        Assert.Equal(Enumerable.Range(1, 2000), train.Select(p => int.Parse(Path.GetFileNameWithoutExtension(p.Noisy))));
        Assert.Equal(Enumerable.Range(15251, 250), test.Select(p => int.Parse(Path.GetFileNameWithoutExtension(p.Noisy))));
        Assert.Equal("00042__", ExtDerived.StemOf("mdc25-tribolium-2d", patches[41]));

        var biosr = ExtStacks.Get("biosr-ccps");
        var cells = (from c in Enumerable.Range(1, 48) from l in Enumerable.Range(1, 8)
                     select new StackPair($"/d/biosr-ccps/CCPs/Cell_{c:000}/Level_{l:00}/RawSIMData.mrc", $"/d/biosr-ccps/CCPs/Cell_{c:000}/Level_{l:00}/RawSIMData_gt.mrc", biosr.Groups[0])).ToList();
        (train, test) = ExtDerived.Select("biosr-ccps-2d", cells);
        Assert.Equal(320, train.Count);
        Assert.Equal(64, test.Count);
        Assert.Equal(Enumerable.Range(41, 8), test.Select(ExtDerived.CellOf).Distinct());
        Assert.True(train.All(p => ExtDerived.CellOf(p) <= 40));
        Assert.Equal("c041__L03", ExtDerived.StemOf("biosr-ccps-2d", test[2]));
        Assert.Throws<ArgumentException>(() => ExtDerived.Select("deepbacs-mreb", cells));
    }

    [Fact]
    public void EveryPlaneBecomesAFileWithItsRawValues()
    {
        string root = TestUtil.TempDirectory();
        StackPair Stack(string noisyPath, string cleanPath, ExtStackGroup group, int planes)
        {
            foreach (var (path, offset) in new[] { (noisyPath, 0), (cleanPath, 1000) })
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                using var writer = TiffStackWriter.Create(path, TiffSampleType.Float32);
                for (int k = 0; k < planes; k++) writer.Append(new GrayImage(4, 4, Enumerable.Repeat(offset + 10f * k + 0.5f, 16).ToArray()));
            }
            return new StackPair(noisyPath, cleanPath, group);
        }
        var tribolium = Stack(Path.Combine(root, "src", "noisy", "00007.tif"), Path.Combine(root, "src", "gt", "00007.tif"), ExtStacks.Get("mdc25-tribolium").Groups[0], 16);
        Assert.Equal(16, ExtDerived.Write([tribolium], Path.Combine(root, "t"), "mdc25-tribolium-2d"));
        Assert.Equal(Enumerable.Range(0, 16).Select(z => $"00007__z{z:00}.tif"), Directory.GetFiles(Path.Combine(root, "t", "low_SNR")).Select(Path.GetFileName).Order(StringComparer.Ordinal));
        Assert.Equal(130.5f, ImageLoader.Load(Path.Combine(root, "t", "low_SNR", "00007__z13.tif")).Pixels[5]);
        Assert.Equal(1130.5f, ImageLoader.Load(Path.Combine(root, "t", "high_SNR", "00007__z13.tif")).Pixels[5]);

        var cell = Stack(Path.Combine(root, "src", "CCPs", "Cell_041", "Level_02", "RawSIMData.tif"),
            Path.Combine(root, "src", "CCPs", "Cell_041", "Level_02", "RawSIMData_gt.tif"), ExtStacks.Get("biosr-ccps").Groups[0], 9);
        Assert.Equal(9, ExtDerived.Write([cell], Path.Combine(root, "b"), "biosr-ccps-2d"));
        Assert.Equal(Enumerable.Range(0, 9).Select(k => $"c041__L02_p{k}.tif"), Directory.GetFiles(Path.Combine(root, "b", "low_SNR")).Select(Path.GetFileName).Order(StringComparer.Ordinal));
        Assert.Equal(1080.5f, ImageLoader.Load(Path.Combine(root, "b", "high_SNR", "c041__L02_p8.tif")).Pixels[0]);
        Directory.Delete(root, recursive: true);
    }

    [Fact]
    public void FmdExportsDorasTrainingFramesAndTestMixUnderUniqueNames()
    {
        string root = TestUtil.TempDirectory(), output = Path.Combine(root, "export");
        FmdTestTree.Build(root);
        var (train, test) = FmdExport.Export(root, output);
        var plan = FmdDataset.PlanTraining(root, new FmdOptions());
        Assert.Equal(plan.Count, train);
        Assert.Equal(2 * 4 * 3, train);   // 2 categories × FOVs 1, 2, 3, 20 (never 19) × the 3 frames there
        Assert.Equal(4, test);
        Assert.Equal(plan.Select(FmdExport.TrainingName).Order(StringComparer.Ordinal),
            Directory.GetFiles(Path.Combine(output, "train", "low_SNR")).Select(Path.GetFileName).Order(StringComparer.Ordinal));
        Assert.DoesNotContain(Directory.GetFiles(Path.Combine(output, "train", "low_SNR")), f => f.Contains("_19_", StringComparison.Ordinal));
        Assert.Equal("Confocal_FISH_raw_2_HV_P001.tif", FmdExport.TrainingName(plan.Single(p => p.Name == "Confocal_FISH/raw/2/HV_P001.png")));
        var testMix = FmdDataset.PlanTestMix(Path.Combine(root, "test_mix"));
        Assert.Equal("Confocal_FISH_1.tif", FmdExport.PredictionName(testMix[0]));
        Assert.Equal(ImageLoader.Load(testMix[0].NoisyPath).Pixels, ImageLoader.Load(Path.Combine(output, "test", "low_SNR", "Confocal_FISH_1.tif")).Pixels);

        File.Delete(Path.Combine(output, "test", "low_SNR", "Confocal_FISH_1.tif"));
        Assert.Equal((train, test), FmdExport.Export(root, output));   // complete: not written again
        Assert.False(File.Exists(Path.Combine(output, "test", "low_SNR", "Confocal_FISH_1.tif")));
        Directory.Delete(root, recursive: true);
    }

    [Fact]
    public void OutsideOutputsAreScoredLikeDorasOnFmd()
    {
        string root = TestUtil.TempDirectory();
        FmdTestTree.Build(root);
        var plan = FmdDataset.PlanTestMix(Path.Combine(root, "test_mix"));
        var dora = FmdBenchmark.EvaluatePlan(_ => DenoisingNet.Create(NetConfig.Default, 3), plan);
        var asNoisy = FmdBenchmark.EvaluatePredictions(p => ImageLoader.Load(p.NoisyPath), plan);
        Assert.Equal(dora.Select(s => (s.Name, s.Category, s.NoisyPsnr, s.NoisySsim, s.BilateralPsnr, s.BilateralSsim)),
            asNoisy.Select(s => (s.Name, s.Category, s.NoisyPsnr, s.NoisySsim, s.BilateralPsnr, s.BilateralSsim)));
        Assert.All(asNoisy, s => Assert.Equal((s.NoisyPsnr, s.NoisySsim), (s.DoraPsnr, s.DoraSsim)));

        // Outputs are clipped to the 8-bit range, as DORA's are.
        GrayImage Constant(FmdDataset.PlannedPair p, float v) { var n = ImageLoader.Load(p.NoisyPath); return new GrayImage(n.Width, n.Height, Enumerable.Repeat(v, n.Length).ToArray()); }
        Assert.Equal(FmdBenchmark.EvaluatePredictions(p => Constant(p, 255), plan).Select(s => s.DoraPsnr),
            FmdBenchmark.EvaluatePredictions(p => Constant(p, 4000), plan).Select(s => s.DoraPsnr));
        Assert.Throws<InvalidDataException>(() => FmdBenchmark.EvaluatePredictions(_ => new GrayImage(3, 3), plan));
        Directory.Delete(root, recursive: true);
    }
}

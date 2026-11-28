using DoraBio.Core.Data;

namespace DoraBio.Core.Tests;

/// <summary>
/// Builds a miniature FMD dataset of real PNG files with the real download's layout: categories Confocal_FISH and
/// WideField_BPAE_B, FOVs 1, 2, 3, 19 and 20, all five noise levels (3 frames each), gt/FOV/avg50.png, and a flat
/// test_mix with two images per category copied from FOV 19.
/// </summary>
internal static class FmdTestTree
{
    public const int Size = 16;
    public static readonly string[] Categories = ["Confocal_FISH", "WideField_BPAE_B"];
    public static readonly int[] FovIds = [1, 2, 3, 19, 20];

    public static void Build(string root)
    {
        foreach (string category in Categories)
            foreach (int fov in FovIds)
            {
                // Each FOV gets a distinct ground truth, so mix-ups between FOVs are detectable.
                byte Truth(int x, int y) => (byte)(20 + 7 * fov + x + y);
                TestUtil.WritePng(Path.Combine(root, category, "gt", fov.ToString(), "avg50.png"), Size, Size, Truth);
                foreach (string level in FmdDataset.Levels)
                    for (int frame = 0; frame < 3; frame++)
                        TestUtil.WritePng(Path.Combine(root, category, level, fov.ToString(), $"HV_P{frame:D3}.png"),
                            Size, Size, (x, y) => (byte)(Truth(x, y) + ((x * 31 + y * 17 + frame * 13) % 9) - 4));
            }

        // test_mix: two images per category, copied from FOV 19, named <Category>_<k>.png, like the real one.
        foreach (string category in Categories)
            for (int k = 1; k <= 2; k++)
            {
                File.Copy(Path.Combine(root, category, "gt", "19", "avg50.png"), Target("gt"));
                foreach (string level in FmdDataset.Levels)
                    File.Copy(Path.Combine(root, category, level, "19", $"HV_P{k:D3}.png"), Target(level));

                string Target(string folder)
                {
                    string path = Path.Combine(root, "test_mix", folder, $"{category}_{k}.png");
                    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                    return path;
                }
            }
    }
}

using DoraBio.Core.Data;

namespace DoraBio.Core.Tests;

internal static class TestUtil
{
    public static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    public static float[] RandomPlane(int length, int seed, float min = 0f, float max = 1f)
    {
        var rng = new Random(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = min + (max - min) * rng.NextSingle();
        return values;
    }

    public static GrayImage RandomImage(int width, int height, int seed) => new(width, height, RandomPlane(width * height, seed));

    public static GrayImage Pattern(int width, int height, Func<int, int, float> f)
    {
        var image = new GrayImage(width, height);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                image[x, y] = f(x, y);
        return image;
    }

    public static GrayImage AddGaussianNoise(GrayImage image, double sigma, int seed)
    {
        var rng = new Random(seed);
        var noisy = image.Clone();
        for (int i = 0; i < noisy.Length; i++) noisy.Pixels[i] += (float)(sigma * SyntheticFluorescence.SampleGaussian(rng));
        return noisy;
    }

    public static double StdDev(ReadOnlySpan<float> values)
    {
        double mean = 0;
        foreach (float v in values) mean += v;
        mean /= values.Length;
        double sum = 0;
        foreach (float v in values) sum += (v - mean) * (v - mean);
        return Math.Sqrt(sum / values.Length);
    }

    public static string TempDirectory()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dorabio-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}

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

    /// <summary>Writes an 8-bit grayscale PNG (filter 0, zlib), the format FMD ships in.</summary>
    public static void WritePng(string path, int width, int height, Func<int, int, byte> pixel)
    {
        var raw = new byte[height * (width + 1)];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                raw[y * (width + 1) + 1 + x] = pixel(x, y);
        using var compressed = new MemoryStream();
        using (var z = new System.IO.Compression.ZLibStream(compressed, System.IO.Compression.CompressionLevel.Fastest, leaveOpen: true))
            z.Write(raw);

        using var png = new MemoryStream();
        png.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        void Chunk(string type, byte[] data)
        {
            var typeAndData = System.Text.Encoding.ASCII.GetBytes(type).Concat(data).ToArray();
            var len = new byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(len, (uint)data.Length);
            var crc = new byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(crc, DoraBio.Core.IO.Crc32.Compute(typeAndData));
            png.Write(len); png.Write(typeAndData); png.Write(crc);
        }
        var ihdr = new byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(ihdr, (uint)width);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(4), (uint)height);
        ihdr[8] = 8;   // bit depth; color type 0 (gray), compression/filter/interlace 0
        Chunk("IHDR", ihdr);
        Chunk("IDAT", compressed.ToArray());
        Chunk("IEND", []);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, png.ToArray());
    }

    public static string TempDirectory()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dorabio-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}

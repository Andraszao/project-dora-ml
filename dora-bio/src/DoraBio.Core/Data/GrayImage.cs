namespace DoraBio.Core.Data;

/// <summary>
/// A single-channel image stored row-major as 32-bit floats.
/// Values are whatever the caller puts in: raw sensor units (ADU) straight from a loader,
/// or normalized intensities after <see cref="Normalization"/>. Nothing here clamps or quantizes.
/// </summary>
/// <remarks>
/// Pixel (x, y) is at <c>Pixels[y * Width + x]</c>. This is the same layout the network uses for a
/// one-channel plane, so a <see cref="GrayImage"/> can be passed to it without copying.
/// </remarks>
public sealed class GrayImage
{
    /// <summary>Width in pixels (number of columns).</summary>
    public int Width { get; }

    /// <summary>Height in pixels (number of rows).</summary>
    public int Height { get; }

    /// <summary>The pixel values, row-major. Exposed directly for speed; callers may read and write it.</summary>
    public float[] Pixels { get; }

    /// <summary>Creates a zero-filled image, or wraps <paramref name="pixels"/> without copying it.</summary>
    /// <param name="width">Width in pixels (positive).</param>
    /// <param name="height">Height in pixels (positive).</param>
    /// <param name="pixels">Optional row-major data of length width × height, used as-is.</param>
    public GrayImage(int width, int height, float[]? pixels = null)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), $"Image size must be positive, got {width}x{height}.");
        if (pixels is not null && pixels.Length != width * height)
            throw new ArgumentException($"Expected {width * height} pixels, got {pixels.Length}.", nameof(pixels));

        Width = width;
        Height = height;
        Pixels = pixels ?? new float[width * height];
    }

    /// <summary>Total number of pixels (Width × Height).</summary>
    public int Length => Pixels.Length;

    /// <summary>The value at column <paramref name="x"/>, row <paramref name="y"/>.</summary>
    public float this[int x, int y]
    {
        get => Pixels[y * Width + x];
        set => Pixels[y * Width + x] = value;
    }

    /// <summary>A writable view of row <paramref name="y"/> (no copy).</summary>
    public Span<float> Row(int y) => Pixels.AsSpan(y * Width, Width);

    /// <summary>Deep copy with an independent pixel array.</summary>
    public GrayImage Clone() => new(Width, Height, (float[])Pixels.Clone());

    /// <summary>Copies the rectangle [x, x+w) × [y, y+h) into a new image.</summary>
    public GrayImage Crop(int x, int y, int w, int h)
    {
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > Width || y + h > Height)
            throw new ArgumentOutOfRangeException(nameof(x), $"Crop {x},{y} {w}x{h} is outside {Width}x{Height}.");

        var result = new GrayImage(w, h);
        for (int row = 0; row < h; row++)
            Pixels.AsSpan((y + row) * Width + x, w).CopyTo(result.Row(row));
        return result;
    }

    /// <summary>The smallest and largest pixel values.</summary>
    public (float Min, float Max) Range()
    {
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in Pixels)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return (min, max);
    }
}

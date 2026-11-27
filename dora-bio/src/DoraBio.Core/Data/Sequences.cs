using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>
/// Time-lapse sequences for brick 39: which frames a multi-frame network sees for each frame, and the training images
/// for its two modes.
/// <list type="bullet">
/// <item><b>Spatiotemporal blind spot</b> (Noise2Void with 3 input frames): input (t−1, t, t+1), blind spots in frame t
/// only, target frame t's hidden pixels. The neighbours' noise is independent of frame t's, so they help without leaking.</item>
/// <item><b>Noise2Noise from neighbouring frames</b>: input frame t, target frame t+1 (and t+1 → t), every pixel in
/// the loss. Single-frame only: with several input frames the target would sit in the input, and the network could copy it.</item>
/// </list>
/// Validation is held out by blocks of frames (<c>name/b…</c> groups), so one long time-lapse still splits by time.
/// </summary>
public static class Sequences
{
    /// <summary>Frames per validation group: frames t with the same t / BlockSize share a group.</summary>
    public const int DefaultBlockSize = 32;

    /// <summary>The shortest sequence a network with <paramref name="inputFrames"/> frames can reflect at both ends.</summary>
    public static int MinimumFrames(int inputFrames) => inputFrames / 2 + 1;

    /// <summary>
    /// The frames a network with <paramref name="inputFrames"/> input frames sees for frame <paramref name="t"/> of a
    /// <paramref name="frameCount"/>-frame sequence, in time order with t in the middle. Missing neighbours are reflected
    /// (t − 1 at t = 0 is frame 1; t + 1 at the last frame is the one before it), never clamped: a clamped copy of frame t in
    /// a neighbour channel would carry the very pixels the blind spot hides, noise and all.
    /// </summary>
    /// <remarks>
    /// Only 1 or 3 frames: with 5, reflecting t − 2 at t = 1 lands on frame 1, which is t itself, so a neighbour channel
    /// would carry the target frame. A wider window needs a padding rule that never repeats the centre frame.
    /// </remarks>
    /// <exception cref="ArgumentException">The sequence is too short to reflect (fewer than <see cref="MinimumFrames"/>).</exception>
    public static int[] FrameIndices(int t, int frameCount, int inputFrames)
    {
        if (inputFrames is not (1 or 3))
            throw new ArgumentOutOfRangeException(nameof(inputFrames), "1 or 3 frames: wider windows would reflect the centre frame into a neighbour.");
        if (t < 0 || t >= frameCount) throw new ArgumentOutOfRangeException(nameof(t));
        int k = inputFrames / 2;
        if (frameCount < MinimumFrames(inputFrames))
            throw new ArgumentException($"A {inputFrames}-frame network needs sequences of at least {MinimumFrames(inputFrames)} frames; this one has {frameCount}.");
        var indices = new int[inputFrames];
        for (int j = 0; j < inputFrames; j++)
        {
            int i = t - k + j;
            if (i < 0) i = -i;
            if (i >= frameCount) i = 2 * (frameCount - 1) - i;
            indices[j] = i;
        }
        return indices;
    }

    /// <summary>
    /// Training images for the spatiotemporal blind spot: one per frame, its neighbours (<see cref="FrameIndices"/>) as
    /// <see cref="ImagePair.Context"/>. Every frame is normalized by its own percentiles, as at inference; a clean frame
    /// by its noisy frame's map.
    /// </summary>
    /// <param name="frames">The noisy frames in time order (raw units).</param>
    /// <param name="inputFrames">The network's input frames (odd).</param>
    /// <param name="name">The sequence's name (group ids and names start with it).</param>
    /// <param name="clean">Optional ground-truth frames, for validation scores only.</param>
    /// <param name="blockSize">Frames per validation group.</param>
    public static List<ImagePair> Spatiotemporal(IReadOnlyList<GrayImage> frames, int inputFrames, string name,
        IReadOnlyList<GrayImage>? clean = null, int blockSize = DefaultBlockSize)
    {
        if (clean is not null && clean.Count != frames.Count) throw new ArgumentException("One clean frame per noisy frame is needed.", nameof(clean));
        var (normalized, maps) = NormalizeEach(frames);
        var result = new List<ImagePair>(frames.Count);
        int center = inputFrames / 2;
        for (int t = 0; t < frames.Count; t++)
        {
            var indices = FrameIndices(t, frames.Count, inputFrames);
            result.Add(new ImagePair(normalized[t], clean is null ? null : Normalization.Apply(clean[t], maps[t]),
                $"{name}/b{t / blockSize}", $"{name}/t{t:0000}")
            {
                Context = inputFrames == 1 ? null : indices.Where((_, j) => j != center).Select(i => normalized[i]).ToArray(),
            });
        }
        return result;
    }

    /// <summary>
    /// Training images for Noise2Noise from neighbouring frames: (t → t+1) and (t+1 → t) for every adjacent pair, the
    /// target in <see cref="ImagePair.Clean"/>. A pair belongs to the validation group of its earlier frame.
    /// </summary>
    /// <exception cref="ArgumentException">Fewer than 2 frames.</exception>
    public static List<ImagePair> NeighbourPairs(IReadOnlyList<GrayImage> frames, string name, int blockSize = DefaultBlockSize)
    {
        if (frames.Count < 2) throw new ArgumentException($"Noise2Noise needs at least 2 frames; '{name}' has {frames.Count}.");
        var (normalized, _) = NormalizeEach(frames);
        var result = new List<ImagePair>(2 * (frames.Count - 1));
        for (int t = 0; t + 1 < frames.Count; t++)
        {
            string group = $"{name}/b{t / blockSize}";
            result.Add(new ImagePair(normalized[t], normalized[t + 1], group, $"{name}/t{t:0000}->t{t + 1:0000}"));
            result.Add(new ImagePair(normalized[t + 1], normalized[t], group, $"{name}/t{t + 1:0000}->t{t:0000}"));
        }
        return result;
    }

    /// <summary>
    /// Loads time-lapses for training: a stack file, or a folder of them (in <c>noisy/</c>, with optional same-named
    /// ground-truth stacks in <c>clean/</c>, or directly in the folder). Every plane of a stack, in file order, is a frame.
    /// </summary>
    /// <param name="path">A stack file or a folder.</param>
    /// <param name="mode">Noise2Void (spatiotemporal, with <paramref name="inputFrames"/>) or Noise2Noise (one input frame).</param>
    /// <param name="inputFrames">The network's input frames: 1 for Noise2Noise.</param>
    /// <param name="blockSize">Frames per validation group.</param>
    public static List<ImagePair> Load(string path, TrainingMode mode, int inputFrames, int blockSize = DefaultBlockSize)
    {
        if (mode == TrainingMode.Noise2Noise && inputFrames != 1)
            throw new ArgumentException("Noise2Noise takes one input frame: with several, its target would be part of the input.");
        if (mode == TrainingMode.Supervised) throw new ArgumentException("Time-lapse training is Noise2Void or Noise2Noise.");
        var stacks = new List<(string Noisy, string? Clean)>();
        if (File.Exists(path)) stacks.Add((path, null));
        else if (Directory.Exists(path))
        {
            string noisyDir = Path.Combine(path, "noisy"), cleanDir = Path.Combine(path, "clean");
            string dir = Directory.Exists(noisyDir) ? noisyDir : path;
            foreach (string file in Directory.GetFiles(dir).Where(ImageLoader.IsSupported).Order(StringComparer.Ordinal))
            {
                string partner = Path.Combine(cleanDir, Path.GetFileName(file));
                stacks.Add((file, Directory.Exists(noisyDir) && File.Exists(partner) ? partner : null));
            }
            if (stacks.Count == 0) throw new InvalidDataException($"No .tif/.tiff stacks in {dir}.");
        }
        else throw new FileNotFoundException($"{path}: no such file or folder.");

        var result = new List<ImagePair>();
        foreach (var (noisy, clean) in stacks)
        {
            string name = Path.GetFileNameWithoutExtension(noisy);
            var frames = ReadPlanes(noisy);
            result.AddRange(mode == TrainingMode.Noise2Noise
                ? NeighbourPairs(frames, name, blockSize)
                : Spatiotemporal(frames, inputFrames, name, clean is null ? null : ReadPlanes(clean), blockSize));
        }
        return result;
    }

    /// <summary>Every plane of a stack file, in file order.</summary>
    public static List<GrayImage> ReadPlanes(string path)
    {
        using var stack = ImageStack.Open(path);
        return Enumerable.Range(0, stack.PlaneCount).Select(stack.ReadPlane).ToList();
    }

    private static (GrayImage[] Normalized, IntensityMap[] Maps) NormalizeEach(IReadOnlyList<GrayImage> frames)
    {
        var maps = frames.Select(f => Normalization.FromPercentiles(f.Pixels)).ToArray();
        return (frames.Select((f, i) => Normalization.Apply(f, maps[i])).ToArray(), maps);
    }
}

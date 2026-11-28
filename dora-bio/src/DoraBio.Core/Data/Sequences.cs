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
/// <item><b>Skip-centre Noise2Noise</b> (brick 40, TeD-inspired): input (t−1, t+1), target frame t, every pixel in the
/// loss; frame t is never part of the input (<see cref="SkipCentre"/>).</item>
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
    /// The neighbours a skip-centre network (brick 40) sees for frame <paramref name="t"/>: (t − 1, t + 1), reflected at
    /// the ends exactly as <see cref="FrameIndices"/> reflects a 3-frame window, so (1, 1) at t = 0 and (T − 2, T − 2) at the
    /// last frame. Neither is ever t: reflection moves an index by at least one frame, which a test checks for every t of
    /// every length. A 2-frame sequence works ((1, 1), then (0, 0)); a single frame has no neighbour and is refused.
    /// </summary>
    /// <exception cref="ArgumentException">The sequence has fewer than 2 frames.</exception>
    public static int[] NeighbourIndices(int t, int frameCount)
    {
        var window = FrameIndices(t, frameCount, 3);
        return [window[0], window[2]];
    }

    /// <summary>
    /// Training images for skip-centre Noise2Noise (brick 40): one per <b>interior</b> frame t = 1 … T − 2, the frame as
    /// <see cref="ImagePair.Noisy"/> (the target, which the sampler never puts into the input) and (t − 1, t + 1) as
    /// <see cref="ImagePair.Context"/>. The two end frames aren't targets: their reflected input would be one neighbour
    /// twice, an input training otherwise never shows. Inference still denoises them (<see cref="NeighbourIndices"/>).
    /// Every frame is normalized by its own percentiles, as at inference.
    /// </summary>
    /// <param name="frames">The noisy frames in time order (raw units).</param>
    /// <param name="name">The sequence's name (group ids and names start with it).</param>
    /// <param name="clean">Optional ground-truth frames, for validation scores only.</param>
    /// <param name="blockSize">Frames per validation group.</param>
    /// <exception cref="ArgumentException">Fewer than 3 frames, so no interior frame.</exception>
    public static List<ImagePair> SkipCentre(IReadOnlyList<GrayImage> frames, string name,
        IReadOnlyList<GrayImage>? clean = null, int blockSize = DefaultBlockSize)
    {
        if (frames.Count < 3) throw new ArgumentException($"Skip-centre training needs at least 3 frames (an interior target); '{name}' has {frames.Count}.");
        if (clean is not null && clean.Count != frames.Count) throw new ArgumentException("One clean frame per noisy frame is needed.", nameof(clean));
        var (normalized, maps) = NormalizeEach(frames);
        var result = new List<ImagePair>(frames.Count - 2);
        for (int t = 1; t + 1 < frames.Count; t++)
            result.Add(new ImagePair(normalized[t], clean is null ? null : Normalization.Apply(clean[t], maps[t]),
                $"{name}/b{t / blockSize}", $"{name}/t{t:0000}")
            {
                Context = [normalized[t - 1], normalized[t + 1]],
            });
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
    /// <param name="mode">
    /// Noise2Void (spatiotemporal, with <paramref name="inputFrames"/>), Noise2Noise (one input frame) or skip-centre
    /// (two input frames, t − 1 and t + 1).
    /// </param>
    /// <param name="inputFrames">The network's input frames: 1 for Noise2Noise, 2 for skip-centre.</param>
    /// <param name="blockSize">Frames per validation group.</param>
    /// <param name="range">
    /// Only these frames of a single stack file, as a standalone sequence (brick 40's training block); null for every frame.
    /// </param>
    /// <param name="axes">The stack's axes when its metadata is missing or wrong (e.g. "TYX"), or null for the file's own.</param>
    /// <param name="onNotice">Optional: called with notices (unlabelled planes taken as frames).</param>
    public static List<ImagePair> Load(string path, TrainingMode mode, int inputFrames, int blockSize = DefaultBlockSize, FrameRange? range = null,
        string? axes = null, Action<string>? onNotice = null)
    {
        if (range is not null && !File.Exists(path)) throw new ArgumentException($"A frame range needs a single stack file; {path} isn't one.");
        if (mode == TrainingMode.Noise2Noise && inputFrames != 1)
            throw new ArgumentException("Noise2Noise takes one input frame: with several, its target would be part of the input.");
        if (mode == TrainingMode.SkipCentre && inputFrames != 2)
            throw new ArgumentException("Skip-centre Noise2Noise takes two input frames, t − 1 and t + 1.");
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
        var notices = new HashSet<string>();
        foreach (var (noisy, clean) in stacks)
        {
            string name = Path.GetFileNameWithoutExtension(noisy);
            // One sequence per time series: a hyperstack's channels and slices are never each other's frames (brick 41).
            foreach (var (label, frames, truth) in ReadSeries(noisy, clean, axes, range, n => { if (notices.Add(n)) onNotice?.Invoke(n); }))
            {
                string series = label.Length == 0 ? name : $"{name}[{label}]";
                result.AddRange(mode switch
                {
                    TrainingMode.Noise2Noise => NeighbourPairs(frames, series, blockSize),
                    TrainingMode.SkipCentre => SkipCentre(frames, series, truth, blockSize),
                    _ => Spatiotemporal(frames, inputFrames, series, truth, blockSize),
                });
            }
        }
        return result;
    }

    /// <summary>
    /// A stack's time series (<see cref="TimeAxis.Series"/>), each read as frames in time order, restricted to
    /// <paramref name="range"/> of the time axis if given, with the matching frames of an optional ground-truth stack.
    /// </summary>
    /// <param name="noisy">The stack file.</param>
    /// <param name="clean">An optional ground-truth stack of the same layout, or null.</param>
    /// <param name="axes">Declared axes, or null for the file's own.</param>
    /// <param name="range">Frames of the time axis to read, or null for all.</param>
    /// <param name="onNotice">Called with the notice when unlabelled planes are taken as frames.</param>
    /// <exception cref="ArgumentException">No time axis, or the range runs past a series.</exception>
    public static IEnumerable<(string Label, List<GrayImage> Frames, List<GrayImage>? Truth)> ReadSeries(string noisy, string? clean, string? axes,
        FrameRange? range, Action<string>? onNotice)
    {
        using var stack = ImageStack.Open(noisy, axes);
        using var truthStack = clean is null ? null : ImageStack.Open(clean, axes);
        if (truthStack is not null && truthStack.Layout.FullAxes != stack.Layout.FullAxes)
            throw new ArgumentException($"{clean}'s axes ({truthStack.Layout}) differ from {noisy}'s ({stack.Layout}).");
        var time = TimeAxis.Resolve(stack.Layout);
        if (time.Notice is { } notice) onNotice?.Invoke(notice);
        foreach (var (label, pages) in TimeAxis.Series(stack.Layout, time.Axis))
        {
            var chosen = pages;
            if (range is { } r)
            {
                if (r.Last >= pages.Length)
                    throw new ArgumentOutOfRangeException(nameof(range), $"{noisy} has {pages.Length} frames per series; frames {r} run past them.");
                chosen = pages[r.First..(r.Last + 1)];
            }
            yield return (label, chosen.Select(stack.ReadPlane).ToList(), truthStack is null ? null : chosen.Select(truthStack.ReadPlane).ToList());
        }
    }

    /// <summary>Every plane of a stack file, in file order.</summary>
    public static List<GrayImage> ReadPlanes(string path)
    {
        using var stack = ImageStack.Open(path);
        return Enumerable.Range(0, stack.PlaneCount).Select(stack.ReadPlane).ToList();
    }

    /// <summary>
    /// Frames <paramref name="range"/>.First … Last (inclusive, 0-based) of a stack file, read one plane at a time, so a
    /// block of a long time-lapse is loaded without the rest (brick 40). The block is then a standalone sequence: its own
    /// first and last frames are its ends, and nothing outside it is read.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The range is empty or runs past the stack.</exception>
    public static List<GrayImage> ReadPlanes(string path, FrameRange range)
    {
        using var stack = ImageStack.Open(path);
        if (range.Last >= stack.PlaneCount)
            throw new ArgumentOutOfRangeException(nameof(range), $"{path} has {stack.PlaneCount} frames; frames {range} run past it.");
        return Enumerable.Range(range.First, range.Count).Select(stack.ReadPlane).ToList();
    }

    /// <summary>
    /// Validation images for a held-out block of a time-lapse (brick 40): one per <b>interior</b> frame t = 1 … T − 2 of
    /// the block, each its own group (<c>name/v</c> + the frame's number in the movie), built for the arm being trained:
    /// <list type="bullet">
    /// <item>Noise2Void, 1 input frame: the frame alone; 3 input frames: with (t − 1, t + 1) as context.</item>
    /// <item>Noise2Noise: the frame, with frame t + 1 as its target (one direction only, so one image per frame).</item>
    /// <item>Skip-centre: the frame as target, (t − 1, t + 1) as context.</item>
    /// </list>
    /// Interior frames only, so every arm's inputs and targets lie inside the block. Because every arm gets the same
    /// number of images, the same groups and the same sizes, <see cref="Training.ValidationSet"/> draws the same frames and
    /// positions for all of them from the same seed: the fixed subset the pre-registration names.
    /// </summary>
    /// <param name="frames">The block's noisy frames (raw units), in time order.</param>
    /// <param name="mode">The arm's training mode.</param>
    /// <param name="inputFrames">The arm's network input frames (1 or 3 for Noise2Void, 1 for Noise2Noise, 2 for skip-centre).</param>
    /// <param name="name">The sequence's name.</param>
    /// <param name="firstFrame">The block's first frame in the movie, so names and groups carry movie frame numbers.</param>
    /// <exception cref="ArgumentException">Fewer than 3 frames, or a mode and frame count that don't go together.</exception>
    public static List<ImagePair> ValidationFrames(IReadOnlyList<GrayImage> frames, TrainingMode mode, int inputFrames, string name, int firstFrame = 0)
    {
        if (frames.Count < 3) throw new ArgumentException($"A validation block needs at least 3 frames (an interior one); '{name}' has {frames.Count}.");
        bool ok = mode switch
        {
            TrainingMode.Noise2Void => inputFrames is 1 or 3,
            TrainingMode.Noise2Noise => inputFrames == 1,
            TrainingMode.SkipCentre => inputFrames == 2,
            _ => false,
        };
        if (!ok) throw new ArgumentException($"No time-lapse validation for mode {mode} with {inputFrames} input frame(s).");
        var (normalized, _) = NormalizeEach(frames);
        var result = new List<ImagePair>(frames.Count - 2);
        for (int t = 1; t + 1 < frames.Count; t++)
        {
            string id = $"{name}/v{firstFrame + t:0000}";
            var frame = normalized[t];
            result.Add(mode switch
            {
                TrainingMode.Noise2Noise => new ImagePair(frame, normalized[t + 1], id, id),
                _ when inputFrames == 1 => new ImagePair(frame, null, id, id),
                _ => new ImagePair(frame, null, id, id) { Context = [normalized[t - 1], normalized[t + 1]] },
            });
        }
        return result;
    }

    private static (GrayImage[] Normalized, IntensityMap[] Maps) NormalizeEach(IReadOnlyList<GrayImage> frames)
    {
        var maps = frames.Select(f => Normalization.FromPercentiles(f.Pixels)).ToArray();
        return (frames.Select((f, i) => Normalization.Apply(f, maps[i])).ToArray(), maps);
    }
}

/// <summary>A block of frames First … Last of a time-lapse, both inclusive and 0-based (brick 40's training, validation and test blocks).</summary>
/// <param name="First">The block's first frame.</param>
/// <param name="Last">The block's last frame.</param>
public readonly record struct FrameRange(int First, int Last)
{
    /// <summary>Number of frames in the block.</summary>
    public int Count => Last - First + 1;

    /// <summary>Parses "A-B" (for example "1370-1669").</summary>
    /// <exception cref="FormatException">Not two non-negative integers A ≤ B separated by a hyphen.</exception>
    public static FrameRange Parse(string text)
    {
        var parts = text.Split('-');
        if (parts.Length != 2 || !int.TryParse(parts[0], out int first) || !int.TryParse(parts[1], out int last) || first < 0 || last < first)
            throw new FormatException($"'{text}' is not a frame range: write FIRST-LAST, e.g. 1370-1669 (0-based, both included).");
        return new FrameRange(first, last);
    }

    /// <summary>"First-Last".</summary>
    public override string ToString() => $"{First}-{Last}";
}

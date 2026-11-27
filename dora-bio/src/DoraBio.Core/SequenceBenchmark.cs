using DoraBio.Core.Data;
using DoraBio.Core.IO;

namespace DoraBio.Core;

/// <summary>One time-lapse frame scored against its high-SNR reference frame (brick 40).</summary>
/// <param name="Frame">The frame's index in the movie.</param>
/// <param name="Psnr">PSNR (data range 1) after the whole frame's least-squares match (CARE's protocol, <see cref="ExtBenchmark"/>).</param>
/// <param name="Ssim">SSIM after the same match.</param>
/// <param name="DynamicPsnr">PSNR over the frame's dynamic mask (<see cref="SequenceBenchmark.DynamicChange"/>), after the same match.</param>
/// <param name="SteadyPsnr">PSNR over the frame's steady-motion mask (<see cref="SequenceBenchmark.SteadyChange"/>).</param>
/// <param name="Fidelity">Brick 32's fidelity of the frame (slope, ratio error, spots).</param>
public sealed record SequenceFrameScore(int Frame, double Psnr, double Ssim, double DynamicPsnr, double SteadyPsnr, FidelityScore Fidelity);

/// <summary>A block of a time-lapse scored frame by frame (brick 40): means over its frames, fidelity pooled.</summary>
/// <param name="Block">The frames scored.</param>
/// <param name="Psnr">Mean of the per-frame PSNRs.</param>
/// <param name="Ssim">Mean of the per-frame SSIMs.</param>
/// <param name="DynamicPsnr">Mean of the per-frame dynamic-pixel PSNRs.</param>
/// <param name="SteadyPsnr">Mean of the per-frame steady-motion PSNRs.</param>
/// <param name="Fidelity">The frames' fidelity pooled as for a set's test images (<see cref="Fidelity.Pool"/>).</param>
/// <param name="Frames">Every frame's scores, in time order.</param>
public sealed record SequenceScore(FrameRange Block, double Psnr, double Ssim, double DynamicPsnr, double SteadyPsnr, FidelityScore Fidelity,
    IReadOnlyList<SequenceFrameScore> Frames);

/// <summary>
/// Scores a block of a time-lapse against its frame-aligned high-SNR movie, as brick 40 pre-registered (PATHWAY.md):
/// every frame with CARE's protocol (<see cref="ExtBenchmark"/>: the reference scaled by its own 0.1/99.9 percentiles,
/// the output matched to it by least squares over the whole frame), then the mean over the block's frames.
/// <para><b>Where things change.</b> Two pixel masks per frame, chosen on the reference alone, after a σ = 1 Gaussian
/// blur and a square root, so shot noise, whose variance grows with brightness, doesn't pull bright still pixels in:
/// S = √max(blurred reference, 0), on the reference's raw values. Then</para>
/// <code>
/// dynamic change at t = |S_t − ½(S_{t−1} + S_{t+1})|      transient onsets, peaks, acceleration (the guard metric)
/// steady change at t  = |S_t − S_{t−1}| + |S_t − S_{t+1}|   also constant-velocity drift (reported only)
/// mask               = the pixels whose change is at least the frame's 90th percentile (its top 10%)
/// </code>
/// <para>and the PSNR is computed over the <b>unblurred</b> normalized reference inside the mask, after the whole frame's
/// match. The block is a standalone sequence: S_{t−1} at its first frame is S of its second, and S_{t+1} at its last is S
/// of the one before (<see cref="Sequences.FrameIndices"/>), the same reflection a model's inference uses, so nothing
/// outside the block is read.</para>
/// </summary>
public static class SequenceBenchmark
{
    /// <summary>The fraction of each frame's pixels a change mask holds.</summary>
    public const double MaskFraction = 0.10;

    /// <summary>σ of the Gaussian blur applied to the reference before the square root, in pixels.</summary>
    public const double BlurSigma = 1.0;

    /// <summary>S = √max(blurred reference, 0), from a reference frame in its raw units.</summary>
    public static float[] Stabilized(GrayImage reference)
    {
        var blurred = Fidelity.Blur(reference, BlurSigma).Pixels;
        var s = new float[blurred.Length];
        for (int i = 0; i < s.Length; i++) s[i] = MathF.Sqrt(Math.Max(blurred[i], 0f));
        return s;
    }

    /// <summary>|S_t − ½(S_{t−1} + S_{t+1})| per pixel: the deviation from the neighbours' midpoint.</summary>
    public static float[] DynamicChange(ReadOnlySpan<float> before, ReadOnlySpan<float> current, ReadOnlySpan<float> after)
    {
        var change = new float[current.Length];
        for (int i = 0; i < change.Length; i++) change[i] = MathF.Abs(current[i] - 0.5f * (before[i] + after[i]));
        return change;
    }

    /// <summary>|S_t − S_{t−1}| + |S_t − S_{t+1}| per pixel: any change, constant-velocity drift included.</summary>
    public static float[] SteadyChange(ReadOnlySpan<float> before, ReadOnlySpan<float> current, ReadOnlySpan<float> after)
    {
        var change = new float[current.Length];
        for (int i = 0; i < change.Length; i++) change[i] = MathF.Abs(current[i] - before[i]) + MathF.Abs(current[i] - after[i]);
        return change;
    }

    /// <summary>
    /// The pixels whose value is at least the k-th largest, k = round(<paramref name="fraction"/> · n): the top fraction,
    /// plus any pixel tied with the threshold (so the mask doesn't depend on how a sort orders ties).
    /// </summary>
    public static bool[] TopFraction(ReadOnlySpan<float> values, double fraction = MaskFraction)
    {
        int n = values.Length, k = Math.Clamp((int)Math.Round(fraction * n), 1, n);
        var sorted = values.ToArray();
        Array.Sort(sorted);
        float threshold = sorted[n - k];
        var mask = new bool[n];
        for (int i = 0; i < n; i++) mask[i] = values[i] >= threshold;
        return mask;
    }

    /// <summary>PSNR (data range 1) of <paramref name="matched"/> against <paramref name="reference"/> over the masked pixels only.</summary>
    public static double MaskedPsnr(ReadOnlySpan<float> matched, ReadOnlySpan<float> reference, ReadOnlySpan<bool> mask)
    {
        double sse = 0;
        long count = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (!mask[i]) continue;
            double d = matched[i] - reference[i];
            sse += d * d;
            count++;
        }
        return Metrics.PsnrFromMse(sse / Math.Max(1, count));
    }

    /// <summary>Scores one frame's output (any units) against its raw reference frame, with the two masks given.</summary>
    public static SequenceFrameScore ScoreFrame(int frame, GrayImage output, GrayImage reference, bool[] dynamicMask, bool[] steadyMask)
    {
        var normalized = ExtBenchmark.NormalizeReference(reference);
        var matched = ExtBenchmark.MatchToReference(output, normalized);
        return new SequenceFrameScore(frame, Metrics.Psnr(matched.Pixels, normalized.Pixels), Metrics.Ssim(matched, normalized),
            MaskedPsnr(matched.Pixels, normalized.Pixels, dynamicMask), MaskedPsnr(matched.Pixels, normalized.Pixels, steadyMask),
            Fidelity.Measure(output, reference));
    }

    /// <summary>
    /// Scores every frame of <paramref name="block"/> for each of <paramref name="methods"/>, with the same masks. Frames are
    /// read one at a time with a window of three (the block's own neighbours, reflected at its ends), so a long movie is
    /// never held in memory.
    /// </summary>
    /// <param name="noisy">The low-SNR movie.</param>
    /// <param name="reference">Its frame-aligned high-SNR movie (same size, at least as many frames).</param>
    /// <param name="block">The frames to score.</param>
    /// <param name="methods">
    /// Each, given frame t (raw units) and its reflected neighbours within the block (t − 1, t + 1, raw units), returns the
    /// output for frame t in any units. A single-frame method ignores the neighbours; the identity scores the noisy input.
    /// </param>
    /// <param name="onFrame">Optional: called with each method's index and its scores for every frame as they are computed.</param>
    /// <returns>One score per method, in order.</returns>
    /// <exception cref="ArgumentException">The movies differ in frame size, or the block runs past either of them.</exception>
    public static List<SequenceScore> Evaluate(ImageStack noisy, ImageStack reference, FrameRange block,
        IReadOnlyList<Func<GrayImage, GrayImage[], GrayImage>> methods, Action<int, SequenceFrameScore>? onFrame = null)
    {
        if (block.Last >= noisy.PlaneCount || block.Last >= reference.PlaneCount)
            throw new ArgumentException($"Frames {block} run past the movie ({noisy.PlaneCount} frames) or its reference ({reference.PlaneCount}).");
        if (block.Count < 2) throw new ArgumentException("A block needs at least 2 frames, so every frame has a neighbour to reflect.");
        var noisyWindow = new Dictionary<int, GrayImage>();
        var stabilizedWindow = new Dictionary<int, float[]>();
        var referenceWindow = new Dictionary<int, GrayImage>();
        GrayImage Reference(int i)
        {
            if (!referenceWindow.TryGetValue(i, out var r)) referenceWindow[i] = r = reference.ReadPlane(i);
            return r;
        }
        GrayImage Noisy(int i)
        {
            if (!noisyWindow.TryGetValue(i, out var f)) noisyWindow[i] = f = noisy.ReadPlane(i);
            return f;
        }
        float[] S(int i)
        {
            if (!stabilizedWindow.TryGetValue(i, out var s)) stabilizedWindow[i] = s = Stabilized(Reference(i));
            return s;
        }

        var frames = methods.Select(_ => new List<SequenceFrameScore>(block.Count)).ToList();
        for (int k = 0; k < block.Count; k++)
        {
            int t = block.First + k;
            foreach (var window in new System.Collections.IDictionary[] { noisyWindow, stabilizedWindow, referenceWindow })
                foreach (int old in window.Keys.Cast<int>().Where(i => i < t - 1).ToList()) window.Remove(old);
            var w = Sequences.FrameIndices(k, block.Count, 3).Select(i => block.First + i).ToArray();
            var raw = Noisy(t);
            var reference_t = Reference(t);
            if (raw.Width != reference_t.Width || raw.Height != reference_t.Height)
                throw new ArgumentException($"Frame {t}: the movie is {raw.Width}x{raw.Height} but its reference is {reference_t.Width}x{reference_t.Height}.");
            var dynamicMask = TopFraction(DynamicChange(S(w[0]), S(t), S(w[2])));
            var steadyMask = TopFraction(SteadyChange(S(w[0]), S(t), S(w[2])));
            GrayImage[] neighbours = [Noisy(w[0]), Noisy(w[2])];
            for (int m = 0; m < methods.Count; m++)
            {
                var score = ScoreFrame(t, methods[m](raw, neighbours), reference_t, dynamicMask, steadyMask);
                frames[m].Add(score);
                onFrame?.Invoke(m, score);
            }
        }
        return frames.Select(f => new SequenceScore(block, f.Average(s => s.Psnr), f.Average(s => s.Ssim), f.Average(s => s.DynamicPsnr),
            f.Average(s => s.SteadyPsnr), Fidelity.Pool(f.Select(s => s.Fidelity).ToList()), f)).ToList();
    }
}

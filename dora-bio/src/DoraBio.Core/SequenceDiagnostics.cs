using DoraBio.Core.Data;
using DoraBio.Core.IO;

namespace DoraBio.Core;

/// <summary>How well the brick 40 guards see real change: the masks' real-change content, measured between the two movies.</summary>
/// <param name="DynamicInMask">Correlation of the signed midpoint change S_t − ½(S_{t−1} + S_{t+1}) between the low- and
/// high-SNR movies, inside the pre-registered dynamic mask (top 10% of the high-SNR movie's |change|).</param>
/// <param name="DynamicAll">The same over every pixel.</param>
/// <param name="SteadyInMask">Correlation of the signed first difference S_t − S_{t−1} between the movies, inside the top 10%
/// of the high-SNR movie's |first difference| (the one-sided half of the pre-registered steady mask).</param>
/// <param name="SteadyAll">The same over every pixel.</param>
/// <param name="RawDynamic">Correlation of the midpoint change between the noisy input and the reference in raw units, every
/// pixel of every interior frame: the shared transient signal the retention ratios divide by.</param>
/// <param name="RawDynamicSe">Its jackknife standard error over <see cref="SequenceDiagnostics.Chunks"/> contiguous frame chunks.</param>
/// <param name="RawSteady">The same for first differences.</param>
/// <param name="RawSteadySe">Its jackknife standard error.</param>
public sealed record MaskAudit(double DynamicInMask, double DynamicAll, double SteadyInMask, double SteadyAll,
    double RawDynamic, double RawDynamicSe, double RawSteady, double RawSteadySe);

/// <summary>One method's post-hoc readouts on a time-lapse block (<see cref="SequenceDiagnostics"/>).</summary>
/// <param name="Change">RMS of (output − noisy input) per frame, in units of the input frame's 0.1–99.9 percentile range.</param>
/// <param name="MeanFidelity">Brick 32's fidelity of the block's time-averaged output against the time-averaged reference.</param>
/// <param name="TileSlope">Proportionality as in brick 32 (log-log slope over foreground tiles), but on 16 px tile means of the
/// time averages, each image's background and scale taken from its tile means' 5th and 99th percentiles instead of single
/// pixels' 0.1st and 99.9th, which residual noise moves (<see cref="SequenceDiagnostics.TileFidelity"/>).</param>
/// <param name="TileRatioError">The matching ratio error on the same tiles.</param>
/// <param name="TransientKept">Real frame-t transients kept, relative to the noisy input (1 = as much as the raw frames carry).</param>
/// <param name="TransientKeptSe">Jackknife standard error of <paramref name="TransientKept"/> over contiguous frame chunks.</param>
/// <param name="MotionKept">Real frame-to-frame changes kept, relative to the noisy input.</param>
/// <param name="MotionKeptSe">Its jackknife standard error.</param>
/// <param name="GainJitterDb">How much worse one affine match for the whole block fits than one match per frame, in dB.</param>
/// <param name="Flicker">Standard deviation over frames of (matched output's frame mean − reference's frame mean), as a fraction
/// of the time-averaged reference's 1st-to-99th percentile range.</param>
/// <param name="TraceCorrelation">Mean, over foreground 16 px tiles, of the correlation between the output's and the reference's
/// tile-mean time series.</param>
public sealed record SequenceReadout(double Change, FidelityScore MeanFidelity, double TileSlope, double TileRatioError,
    double TransientKept, double TransientKeptSe, double MotionKept, double MotionKeptSe, double GainJitterDb, double Flicker,
    double TraceCorrelation);

/// <summary>
/// Post-hoc diagnostics for brick 40 (added after its run, at the owner's request, to check whether its two guards could
/// see what they guard; not part of the pre-registered reading). The two movies' noises are independent, which makes three
/// things measurable without trusting a noisy reference pixel by pixel:
/// <list type="bullet">
/// <item><b>Mask audit.</b> Inside a mask chosen on the high-SNR movie alone, the correlation between the two movies' signed
/// changes is the share of real change in it: independent noise doesn't correlate.</item>
/// <item><b>Transients and motion kept.</b> With D the midpoint change x_t − ½(x_{t−1} + x_{t+1}) of an output and of the
/// reference (raw units, every pixel of every interior frame), cov(D_out, D_ref) has the reference's noise averaged out, so it
/// measures the real change the output carries. Divided by the output's gain g (the slope of its time-averaged image on the
/// reference's, where time-averaging removes both noises) and by the same quantity for the noisy input, it says what fraction
/// of real transients survive: about 1 for an output that keeps them; for one-frame flashes, −2/3 for the neighbours' mean
/// ½(x_{t−1} + x_{t+1}), which never sees frame t and shows a flash at t as a dip. "Motion kept" is the same with the first
/// difference x_t − x_{t−1} (−1/2 for the neighbours' mean on one-frame flashes).</item>
/// <item><b>Time-averaged fidelity.</b> Brick 32's slope and ratio error on the block's time-averaged output and reference:
/// averaging the same frames on both sides keeps a proportional output proportional, and removes most reference noise.</item>
/// </list>
/// Gain stability compares one least-squares affine match for the whole block (raw units) with one per frame; flicker and
/// tile traces follow the frame means and 16 px tile means over time. The block is a standalone sequence, as in
/// <see cref="SequenceBenchmark"/>.
/// </summary>
public static class SequenceDiagnostics
{
    /// <summary>Side of the tiles whose mean traces are compared, in pixels.</summary>
    public const int TileSize = 16;

    /// <summary>Contiguous frame chunks for the jackknife standard errors (neighbouring frames share signal, so chunks, not frames).</summary>
    public const int Chunks = 10;

    /// <summary>
    /// Runs every readout for each method over <paramref name="block"/>; the first method should be the identity (the
    /// noisy input), which the retention readouts are relative to.
    /// </summary>
    /// <param name="noisy">The low-SNR movie.</param>
    /// <param name="reference">Its frame-aligned high-SNR movie.</param>
    /// <param name="block">The frames (at least 4).</param>
    /// <param name="methods">As in <see cref="SequenceBenchmark.Evaluate"/>; method 0 must return the noisy frame itself.</param>
    /// <returns>The mask audit, and one readout per method.</returns>
    public static (MaskAudit Audit, List<SequenceReadout> Readouts) Run(ImageStack noisy, ImageStack reference, FrameRange block,
        IReadOnlyList<Func<GrayImage, GrayImage[], GrayImage>> methods)
    {
        if (block.Count < 4) throw new ArgumentException("The diagnostics need at least 4 frames.");
        if (block.Last >= noisy.PlaneCount || block.Last >= reference.PlaneCount)
            throw new ArgumentException($"Frames {block} run past the movie or its reference.");
        int m = methods.Count, n = block.Count;
        var first = reference.ReadPlane(block.First);
        int w = first.Width, h = first.Height, plane = w * h;
        int tilesX = w / TileSize, tilesY = h / TileSize, tiles = tilesX * tilesY;

        var timeSum = new double[m + 1][];   // index m: the reference
        for (int k = 0; k <= m; k++) timeSum[k] = new double[plane];
        var tileTrace = new double[m + 1][];
        for (int k = 0; k <= m; k++) tileTrace[k] = new double[tiles * n];
        var frameMean = new double[m + 1, n];
        var frameStats = new Moments[m, n];          // per frame: x = output, y = reference
        var change = new double[m];
        var dynamicCov = new Moments[m, Chunks];     // x = output's midpoint change, y = reference's; per frame chunk
        var steadyCov = new Moments[m, Chunks];      // first differences
        int Chunk(int k) => Math.Min(Chunks - 1, k * Chunks / n);
        Moments auditDynMask = default, auditDynAll = default, auditStdMask = default, auditStdAll = default;

        // Sliding windows of three frames (k − 2, k − 1, k): outputs, reference, and √ of the blurred movies.
        var outputs = new float[m][][];
        for (int k = 0; k < m; k++) outputs[k] = new float[3][];
        var refs = new float[3][];
        var sRef = new float[3][];
        var sLow = new float[3][];
        var noisyWindow = new Dictionary<int, GrayImage>();
        GrayImage Noisy(int i)
        {
            if (!noisyWindow.TryGetValue(i, out var f)) noisyWindow[i] = f = noisy.ReadPlane(i);
            return f;
        }

        for (int k = 0; k < n; k++)
        {
            int t = block.First + k, slot = k % 3;
            foreach (int old in noisyWindow.Keys.Where(i => i < t - 1).ToList()) noisyWindow.Remove(old);
            var window = Sequences.FrameIndices(k, n, 3).Select(i => block.First + i).ToArray();
            var raw = Noisy(t);
            var refFrame = k == 0 ? first : reference.ReadPlane(t);
            if (raw.Width != w || raw.Height != h || refFrame.Width != w || refFrame.Height != h)
                throw new ArgumentException($"Frame {t}: every frame of both movies must be {w}x{h}.");
            refs[slot] = refFrame.Pixels;
            sRef[slot] = SequenceBenchmark.Stabilized(refFrame);
            sLow[slot] = SequenceBenchmark.Stabilized(raw);
            var map = Normalization.FromPercentiles(raw.Pixels);
            double range = Math.Max(1e-12, map.High - map.Low);
            GrayImage[] neighbours = [Noisy(window[0]), Noisy(window[2])];
            for (int j = 0; j < m; j++)
            {
                var output = methods[j](raw, neighbours).Pixels;
                if (output.Length != plane) throw new ArgumentException("A method returned a frame of another size.");
                outputs[j][slot] = output;
                double sq = 0;
                var stats = new Moments();
                for (int i = 0; i < plane; i++)
                {
                    double d = (output[i] - raw.Pixels[i]) / range;
                    sq += d * d;
                    stats.Add(output[i], refFrame.Pixels[i]);
                    timeSum[j][i] += output[i];
                }
                change[j] += Math.Sqrt(sq / plane) / n;
                frameStats[j, k] = stats;
                frameMean[j, k] = stats.MeanX;
                AddTiles(output, w, tilesX, tilesY, tileTrace[j], k, n);
            }
            double refMean = 0;
            for (int i = 0; i < plane; i++) { timeSum[m][i] += refFrame.Pixels[i]; refMean += refFrame.Pixels[i]; }
            frameMean[m, k] = refMean / plane;
            AddTiles(refFrame.Pixels, w, tilesX, tilesY, tileTrace[m], k, n);

            if (k >= 1)   // first differences at k
            {
                int prev = (k - 1) % 3;
                var steadyMask = SequenceBenchmark.TopFraction(Abs(Diff(sRef[slot], sRef[prev])));
                for (int i = 0; i < plane; i++)
                {
                    double a = sLow[slot][i] - sLow[prev][i], b = sRef[slot][i] - sRef[prev][i];
                    auditStdAll.Add(a, b);
                    if (steadyMask[i]) auditStdMask.Add(a, b);
                }
                for (int j = 0; j < m; j++)
                    for (int i = 0; i < plane; i++)
                        steadyCov[j, Chunk(k)].Add(outputs[j][slot][i] - outputs[j][prev][i], refs[slot][i] - refs[prev][i]);
            }
            if (k >= 2)   // midpoint changes at k − 1 (an interior frame, so no reflection enters)
            {
                int before = (k - 2) % 3, mid = (k - 1) % 3;
                var dynamicMask = SequenceBenchmark.TopFraction(SequenceBenchmark.DynamicChange(sRef[before], sRef[mid], sRef[slot]));
                for (int i = 0; i < plane; i++)
                {
                    double a = sLow[mid][i] - 0.5 * (sLow[before][i] + sLow[slot][i]);
                    double b = sRef[mid][i] - 0.5 * (sRef[before][i] + sRef[slot][i]);
                    auditDynAll.Add(a, b);
                    if (dynamicMask[i]) auditDynMask.Add(a, b);
                }
                for (int j = 0; j < m; j++)
                {
                    float[] o0 = outputs[j][before], o1 = outputs[j][mid], o2 = outputs[j][slot];
                    for (int i = 0; i < plane; i++)
                        dynamicCov[j, Chunk(k - 1)].Add(o1[i] - 0.5 * (o0[i] + o2[i]), refs[mid][i] - 0.5 * (refs[before][i] + refs[slot][i]));
                }
            }
        }

        // Gains: the slope of each time-averaged output on the time-averaged reference.
        var refAverage = Average(timeSum[m], n, w, h);
        var gains = new double[m];
        var readoutFidelity = new FidelityScore[m];
        var tileFidelity = new (double Slope, double RatioError, int Tiles)[m];
        for (int j = 0; j < m; j++)
        {
            var average = Average(timeSum[j], n, w, h);
            var g = new Moments();
            for (int i = 0; i < plane; i++) g.Add(refAverage.Pixels[i], average.Pixels[i]);
            gains[j] = g.Covariance / Math.Max(1e-30, g.VarianceX);
            readoutFidelity[j] = Fidelity.Measure(average, refAverage);
            tileFidelity[j] = TileFidelity(average, refAverage);
        }
        // Retention ratios, on all chunks and leaving each chunk out (the gains come from the whole block's averages).
        double Ratio(Moments[,] cov, int j, int skip) => Pool(cov, j, skip).Covariance / gains[j] / (Pool(cov, 0, skip).Covariance / gains[0]);
        (double Value, double Se) Retention(Moments[,] cov, int j) =>
            (Ratio(cov, j, -1), Jackknife(c => Ratio(cov, j, c)));
        (double Value, double Se) RawCorrelation(Moments[,] cov) => (Pool(cov, 0, -1).Correlation, Jackknife(c => Pool(cov, 0, c).Correlation));

        // Foreground tiles: reference time-average tile mean at least Fidelity.ForegroundLevel on its own 0–1 scale.
        var normalizedAverage = ExtBenchmark.NormalizeReference(refAverage);
        var foreground = new List<int>();
        for (int ty = 0; ty < tilesY; ty++)
            for (int tx = 0; tx < tilesX; tx++)
            {
                double s = 0;
                for (int y = ty * TileSize; y < (ty + 1) * TileSize; y++)
                    for (int x = tx * TileSize; x < (tx + 1) * TileSize; x++) s += normalizedAverage.Pixels[y * w + x];
                if (s / (TileSize * TileSize) >= Fidelity.ForegroundLevel) foreground.Add(ty * tilesX + tx);
            }

        var sortedAverage = refAverage.Pixels.Order().ToArray();
        double refRange = sortedAverage[(int)(0.99 * (plane - 1))] - sortedAverage[(int)(0.01 * (plane - 1))];
        var readouts = new List<SequenceReadout>(m);
        for (int j = 0; j < m; j++)
        {
            // One affine match for the whole block against one per frame, both by least squares in raw units.
            var all = new Moments();
            double perFrameSse = 0;
            for (int k = 0; k < n; k++) { all = all.Merge(frameStats[j, k]); perFrameSse += frameStats[j, k].BestAffineSse(); }
            var (a, b) = all.BestAffine();
            double globalSse = 0;
            var residual = new double[n];
            for (int k = 0; k < n; k++)
            {
                globalSse += frameStats[j, k].AffineSse(a, b);
                residual[k] = a * frameMean[j, k] + b - frameMean[m, k];
            }
            double jitter = 10 * Math.Log10(Math.Max(1e-30, globalSse) / Math.Max(1e-30, perFrameSse));
            double trace = foreground.Count == 0 ? double.NaN
                : foreground.Average(tile => Correlation(tileTrace[j].AsSpan(tile * n, n), tileTrace[m].AsSpan(tile * n, n)));
            var (transient, transientSe) = Retention(dynamicCov, j);
            var (motion, motionSe) = Retention(steadyCov, j);
            readouts.Add(new SequenceReadout(change[j], readoutFidelity[j], tileFidelity[j].Slope, tileFidelity[j].RatioError,
                transient, transientSe, motion, motionSe,
                jitter, StandardDeviation(residual) / Math.Max(1e-30, refRange), trace));
        }
        var (rawDynamic, rawDynamicSe) = RawCorrelation(dynamicCov);
        var (rawSteady, rawSteadySe) = RawCorrelation(steadyCov);
        return (new MaskAudit(auditDynMask.Correlation, auditDynAll.Correlation, auditStdMask.Correlation, auditStdAll.Correlation,
            rawDynamic, rawDynamicSe, rawSteady, rawSteadySe), readouts);
    }

    /// <summary>Method <paramref name="j"/>'s sums over every chunk but <paramref name="skip"/> (−1: all of them).</summary>
    private static Moments Pool(Moments[,] cov, int j, int skip)
    {
        var total = new Moments();
        for (int c = 0; c < cov.GetLength(1); c++) if (c != skip) total = total.Merge(cov[j, c]);
        return total;
    }

    /// <summary>Jackknife standard error over <see cref="Chunks"/> leave-one-chunk-out estimates: √((G − 1)/G · Σ (θ₍₋g₎ − θ̄)²).</summary>
    private static double Jackknife(Func<int, double> leaveOut)
    {
        var values = Enumerable.Range(0, Chunks).Select(leaveOut).ToArray();
        double mean = values.Average();
        return Math.Sqrt((Chunks - 1.0) / Chunks * values.Sum(v => (v - mean) * (v - mean)));
    }

    /// <summary>
    /// Brick 32's proportionality on 16 px tile means: each image's background is its tile means' 5th percentile and its scale
    /// their 99th percentile above that; tiles whose reference value is at least <see cref="Fidelity.ForegroundLevel"/> of the
    /// scale are fitted as output ∝ reference^k in log space, and the ratio error is the median |output / (g · reference) − 1|
    /// after the overall gain g. Tile means average 256 pixels, so single noisy pixels no longer set background or scale.
    /// </summary>
    internal static (double Slope, double RatioError, int Tiles) TileFidelity(GrayImage output, GrayImage reference)
    {
        int w = reference.Width, tilesX = w / TileSize, tilesY = reference.Height / TileSize, tiles = tilesX * tilesY;
        var o = new double[tiles];
        var r = new double[tiles];
        AddTiles(output.Pixels, w, tilesX, tilesY, o, 0, 1);
        AddTiles(reference.Pixels, w, tilesX, tilesY, r, 0, 1);
        static double Percentile(double[] v, double q) { var s = v.Order().ToArray(); return s[(int)Math.Round(q * (s.Length - 1))]; }
        double bo = Percentile(o, 0.05), br = Percentile(r, 0.05);
        double so = Math.Max(1e-30, Percentile(o, 0.99) - bo), sr = Math.Max(1e-30, Percentile(r, 0.99) - br);
        var logs = new List<(double R, double O)>();
        for (int i = 0; i < tiles; i++)
        {
            double rv = (r[i] - br) / sr;
            if (rv < Fidelity.ForegroundLevel) continue;
            logs.Add((Math.Log(rv), Math.Log(Math.Max((o[i] - bo) / so, 1e-3))));
        }
        if (logs.Count < 2) return (double.NaN, double.NaN, logs.Count);
        double mr = logs.Average(p => p.R), mo = logs.Average(p => p.O), cov = 0, var = 0;
        foreach (var (lr, lo) in logs) { cov += (lr - mr) * (lo - mo); var += (lr - mr) * (lr - mr); }
        double logGain = mo - mr;
        var errors = logs.Select(p => Math.Abs(Math.Exp(p.O - p.R - logGain) - 1)).Order().ToList();
        return (var > 0 ? cov / var : double.NaN, errors[errors.Count / 2], logs.Count);
    }

    private static void AddTiles(float[] frame, int w, int tilesX, int tilesY, double[] traces, int k, int n)
    {
        for (int ty = 0; ty < tilesY; ty++)
            for (int tx = 0; tx < tilesX; tx++)
            {
                double s = 0;
                for (int y = ty * TileSize; y < (ty + 1) * TileSize; y++)
                    for (int x = tx * TileSize; x < (tx + 1) * TileSize; x++) s += frame[y * w + x];
                traces[(ty * tilesX + tx) * n + k] = s / (TileSize * TileSize);
            }
    }

    private static GrayImage Average(double[] sum, int n, int w, int h)
    {
        var pixels = new float[sum.Length];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = (float)(sum[i] / n);
        return new GrayImage(w, h, pixels);
    }

    private static float[] Diff(float[] a, float[] b)
    {
        var d = new float[a.Length];
        for (int i = 0; i < d.Length; i++) d[i] = a[i] - b[i];
        return d;
    }

    private static float[] Abs(float[] values)
    {
        for (int i = 0; i < values.Length; i++) values[i] = MathF.Abs(values[i]);
        return values;
    }

    private static double StandardDeviation(double[] values)
    {
        double mean = values.Average();
        return Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / values.Length);
    }

    private static double Correlation(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var moments = new Moments();
        for (int i = 0; i < a.Length; i++) moments.Add(a[i], b[i]);
        return moments.Correlation;
    }

    /// <summary>Running sums for means, (co)variances and least-squares fits of y on x.</summary>
    private struct Moments
    {
        private double _n, _x, _y, _xx, _xy, _yy;

        public void Add(double x, double y) { _n++; _x += x; _y += y; _xx += x * x; _xy += x * y; _yy += y * y; }

        public readonly Moments Merge(Moments o) => new()
            { _n = _n + o._n, _x = _x + o._x, _y = _y + o._y, _xx = _xx + o._xx, _xy = _xy + o._xy, _yy = _yy + o._yy };

        public readonly double MeanX => _x / Math.Max(1, _n);

        public readonly double Covariance => (_xy - _x * _y / _n) / _n;

        public readonly double VarianceX => (_xx - _x * _x / _n) / _n;

        private readonly double VarianceY => (_yy - _y * _y / _n) / _n;

        public readonly double Correlation => Covariance / Math.Sqrt(Math.Max(1e-300, VarianceX * VarianceY));

        /// <summary>a, b minimizing Σ (a·x + b − y)².</summary>
        public readonly (double A, double B) BestAffine()
        {
            double a = VarianceX > 0 ? Covariance / VarianceX : 0;
            return (a, (_y - a * _x) / _n);
        }

        /// <summary>Σ (a·x + b − y)², from the sums alone.</summary>
        public readonly double AffineSse(double a, double b) =>
            a * a * _xx + 2 * a * b * _x + _n * b * b - 2 * a * _xy - 2 * b * _y + _yy;

        public readonly double BestAffineSse()
        {
            var (a, b) = BestAffine();
            return Math.Max(0, AffineSse(a, b));
        }
    }
}

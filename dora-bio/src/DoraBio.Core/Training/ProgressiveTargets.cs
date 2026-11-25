namespace DoraBio.Core.Training;

public enum TargetEventKind
{
    Hit,
    Escalated,
    Mastered,
    Missed,
}

public sealed record TargetEvent(TargetEventKind Kind, double Value, double Target, double NewTarget, int Hits, int HitsNeeded)
{
    public override string ToString() => Kind switch
    {
        TargetEventKind.Hit => $"🎉 target {Target:0.0} dB hit ({Hits}/{HitsNeeded})",
        TargetEventKind.Escalated => $"🚀 target raised {Target:0.0} → {NewTarget:0.0} dB",
        TargetEventKind.Mastered => $"👑 mastery: held the final {Target:0.0} dB target",
        _ => Hits > 0 ? $"📉 missed {Target:0.0} dB, streak reset" : $"target {Target:0.0} dB not reached yet",
    };
}

/// <summary>
/// DORA's gamified milestone loop, carried over from v1 with its quirks fixed.
/// Each epoch's score is checked against the current target. <see cref="HitsNeeded"/> consecutive hits raise
/// the target by <see cref="Increment"/>, up to <see cref="Maximum"/>. Holding the maximum for
/// <see cref="HitsNeeded"/> epochs is mastery.
///
/// Fixed versus v1: the maximum really is the last target (v1 never set its 98% max as a target),
/// and hitting a target no longer raises the learning rate. The trainer lowers it on plateaus instead.
/// </summary>
public sealed class ProgressiveTargets
{
    public double Increment { get; }
    public double Maximum { get; }
    public int HitsNeeded { get; }

    public double CurrentTarget { get; private set; }
    public int Hits { get; private set; }
    public bool IsMastered { get; private set; }

    public ProgressiveTargets(double initial, double increment, double maximum, int hitsNeeded)
    {
        if (increment <= 0) throw new ArgumentOutOfRangeException(nameof(increment));
        if (maximum < initial) throw new ArgumentOutOfRangeException(nameof(maximum), "Maximum must be ≥ initial.");
        if (hitsNeeded < 1) throw new ArgumentOutOfRangeException(nameof(hitsNeeded));
        CurrentTarget = initial;
        Increment = increment;
        Maximum = maximum;
        HitsNeeded = hitsNeeded;
    }

    public TargetEvent Evaluate(double value)
    {
        double target = CurrentTarget;
        if (IsMastered)
            return new TargetEvent(TargetEventKind.Mastered, value, target, target, Hits, HitsNeeded);

        if (value < target)
        {
            int hadHits = Hits;
            Hits = 0;
            return new TargetEvent(TargetEventKind.Missed, value, target, target, hadHits, HitsNeeded);
        }

        Hits++;
        if (Hits < HitsNeeded)
            return new TargetEvent(TargetEventKind.Hit, value, target, target, Hits, HitsNeeded);

        if (target >= Maximum - 1e-9)
        {
            IsMastered = true;
            return new TargetEvent(TargetEventKind.Mastered, value, target, target, Hits, HitsNeeded);
        }

        CurrentTarget = Math.Min(Maximum, target + Increment);
        Hits = 0;
        return new TargetEvent(TargetEventKind.Escalated, value, target, CurrentTarget, HitsNeeded, HitsNeeded);
    }
}

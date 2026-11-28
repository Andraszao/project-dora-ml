namespace DoraBio.Core.Training;

/// <summary>What happened when an epoch's score was checked against the current target.</summary>
public enum TargetEventKind
{
    /// <summary>Reached the target; the streak grows but isn't long enough to escalate yet.</summary>
    Hit,

    /// <summary>Reached the target enough times in a row; the target has been raised.</summary>
    Escalated,

    /// <summary>Held the final target for the required streak (or had already done so).</summary>
    Mastered,

    /// <summary>Below the target; any streak resets to zero.</summary>
    Missed,
}

/// <summary>The outcome of one <see cref="ProgressiveTargets.Evaluate"/> call.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="Value">The score that was evaluated (dB of PSNR gain).</param>
/// <param name="Target">The target in effect for this evaluation.</param>
/// <param name="NewTarget">The target after this evaluation (differs from <paramref name="Target"/> only on escalation).</param>
/// <param name="Hits">Streak length counted by this evaluation (for a miss: the streak that was lost).</param>
/// <param name="HitsNeeded">Streak length required to escalate.</param>
public sealed record TargetEvent(TargetEventKind Kind, double Value, double Target, double NewTarget, int Hits, int HitsNeeded)
{
    /// <summary>One-line, emoji-decorated message in DORA's narrating style, for logs and the UI.</summary>
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
    /// <summary>How much each escalation raises the target.</summary>
    public double Increment { get; }

    /// <summary>The final target. Escalation is capped here, so the maximum is always reachable as a target.</summary>
    public double Maximum { get; }

    /// <summary>Consecutive hits needed to escalate, or at the maximum, to master.</summary>
    public int HitsNeeded { get; }

    /// <summary>The target the next score is compared against.</summary>
    public double CurrentTarget { get; private set; }

    /// <summary>Length of the current streak of hits at <see cref="CurrentTarget"/>.</summary>
    public int Hits { get; private set; }

    /// <summary>True once the final target has been held; later evaluations just report mastery.</summary>
    public bool IsMastered { get; private set; }

    /// <param name="initial">First target.</param>
    /// <param name="increment">Raise per escalation (positive).</param>
    /// <param name="maximum">Final target (at least <paramref name="initial"/>).</param>
    /// <param name="hitsNeeded">Streak length required (at least 1).</param>
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

    /// <summary>Checks one epoch's score against the current target and advances the state machine.</summary>
    /// <remarks>
    /// Below target: miss, streak resets. At or above: the streak grows. When it reaches
    /// <see cref="HitsNeeded"/>, either raise the target to min(target + increment, maximum) and reset the
    /// streak, or, if the target already is the maximum, declare mastery.
    /// </remarks>
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

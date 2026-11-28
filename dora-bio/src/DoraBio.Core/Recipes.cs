namespace DoraBio.Core;

/// <summary>
/// What training does when a setting is left out: the defaults the quality passes decided. They live in the core
/// (brick 41) so the CLI and the app train the same recipe from the same request; <c>dora-bio info --default-…</c>
/// prints them, which is how gates 29, 35, 37 and 40 check them against the recorded verdicts.
/// </summary>
public static class Recipes
{
    /// <summary>
    /// Supervised training's default <c>--condition</c>. Quality pass Q13 decides it (gate 29 checks that it follows the
    /// recorded verdict): "noise" only if the conditioned U-Net met its pre-registered bar.
    /// </summary>
    public const string DefaultCondition = "none";

    /// <summary>
    /// <c>train --mode n2v</c>'s defaults: network, what picks the epoch, early-stop patience and which weights are kept.
    /// Quality pass Q2 decides them (gate 35 checks that they follow the recorded verdict).
    /// </summary>
    public static readonly (string Arch, string Select, int Patience, string Keep) DefaultN2v = ("dilated", "auto", 6, "best");

    /// <summary>
    /// <c>train --mode n2v</c>'s default <c>--route</c> when no recipe option is given: "auto" (route each training image by
    /// the noise check, <see cref="Training.RoutedNoise2Void"/>) since Q20 adopted Q19's routed design; "off" trains one
    /// network with <see cref="DefaultN2v"/>. Gate 37 checks that it follows the recorded Q20 verdict.
    /// </summary>
    public const string DefaultN2vRoute = "auto";

    /// <summary>What <see cref="DefaultN2vRoute"/> trains, as <c>info --default-n2v-route</c> prints it.</summary>
    public static string DescribeN2vRoute() => DefaultN2vRoute == "auto"
        ? "auto: independent → unet16 select=loss patience=0 keep=last; correlated → dilated24 struct=diamond-2 select=auto patience=6 keep=best"
        : "off";

    /// <summary>
    /// What training on a stack file does with neither a mode nor a frame count (brick 40's adoption rule,
    /// pre-registered in PATHWAY.md; gate 40 checks it against the recorded verdict): "none" refuses and asks for a mode,
    /// "n2n" is one-sided Noise2Noise (A2), "frames-3" the 3-frame spatiotemporal blind spot (A3), "skip-centre"
    /// skip-centre Noise2Noise (A4).
    /// </summary>
    public static readonly string DefaultTimeLapse = "none";   // readonly, not const: a const would make the other branches unreachable

    /// <summary>
    /// Options that choose a recipe by hand. When one is given, Noise2Void trains the one network it describes instead of
    /// the routed default (<see cref="DefaultN2vRoute"/>).
    /// </summary>
    public static readonly IReadOnlyList<string> RecipeOptions =
        ["arch", "features", "depth", "select", "patience", "keep", "struct", "footprint", "subset", "init", "condition", "save-last", "vst"];
}

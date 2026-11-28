using DoraBio.Core.IO;

namespace DoraBio.Core.Data;

/// <summary>The axis a time-lapse model steps along in a stack, and the notice to show when it had to be assumed.</summary>
/// <param name="Axis">'T', or 'I' for a stack whose planes carry no label (treated as frames).</param>
/// <param name="Notice">Null when the file labels its frames; otherwise the line to print and record in the report.</param>
public sealed record TimeAxisChoice(char Axis, string? Notice);

/// <summary>
/// Which planes of a stack are a time-lapse model's neighbours (brick 41): t − 1 and t + 1 along the time axis, with every
/// other axis (channel, depth, field of view) held fixed, reflected at the series' ends as <see cref="Sequences.FrameIndices"/>
/// reflects a window. Pages are stored row-major over the layout's axes (ImageJ: channels fastest, then slices, then frames),
/// so in a 2-channel movie page t's neighbour is two pages away, not one; reading pages in file order would hand the
/// network the other channel as "the next frame".
/// <list type="bullet">
/// <item>A labelled T axis is the time axis.</item>
/// <item>Planes with no label (one axis, I) are taken as frames, and <see cref="UnlabelledNotice"/> says so.</item>
/// <item>A stack labelled Z with no T is a volume: refused, with the ways to say otherwise (<see cref="ZStackMessage"/>).</item>
/// <item>Anything else without a time axis (channels only, fields of view) is refused.</item>
/// </list>
/// </summary>
public static class TimeAxis
{
    /// <summary>Printed, and written into the report, when unlabelled planes are taken as frames.</summary>
    public const string UnlabelledNotice = "axes unlabelled: planes treated as T (pass --axes ZYX if this is a Z-stack)";

    /// <summary>The refusal for a stack whose planes are labelled Z (depth) and which has no T axis.</summary>
    public const string ZStackMessage =
        "this stack's planes are labelled Z (a volume), and a time-lapse model needs frames. If they are frames, pass " +
        "--axes TYX (TZYX, TCYX, … for a hyperstack), or fix the file in Fiji: Image › Hyperstacks › Stack to Hyperstack… " +
        "(Slices = 1, Frames = the plane count), or Re-order Hyperstack's Channels/Slices/Frames if Z and T are swapped";

    /// <summary>The time axis of <paramref name="layout"/> for a model that needs neighbouring frames.</summary>
    /// <exception cref="ArgumentException">The stack has no time axis (see the class remarks), or a single frame.</exception>
    public static TimeAxisChoice Resolve(StackLayout layout)
    {
        if (layout.Size('T') > 1) return new('T', null);
        if (layout.Axes == "I") return new('I', UnlabelledNotice);
        if (layout.Size('Z') > 1) throw new ArgumentException($"{ZStackMessage} ({layout}).");
        if (layout.PlaneCount <= 1)
            throw new ArgumentException("A time-lapse model needs at least 2 frames; this is a single plane.");
        throw new ArgumentException($"This stack has no time axis ({layout}), and a time-lapse model needs frames; pass --axes with a T axis if it has one.");
    }

    /// <summary>
    /// The stack's time series: one per position of the other axes (each channel, each slice, …), each the pages along
    /// <paramref name="axis"/> in time order, labelled by the other axes' positions (e.g. "Z=2,C=1"; empty when the stack
    /// has no other axis).
    /// </summary>
    public static IEnumerable<(string Label, int[] Pages)> Series(StackLayout layout, char axis)
    {
        int a = layout.Axes.IndexOf(axis);
        if (a < 0) throw new ArgumentException($"The stack has no {axis} axis ({layout}).", nameof(axis));
        int stride = 1;
        for (int k = layout.Axes.Length - 1; k > a; k--) stride *= layout.Shape[k];
        int n = layout.Shape[a];
        for (long first = 0; first < layout.PlaneCount; first++)
        {
            if (first / stride % n != 0) continue;   // not at t = 0: part of a series already listed
            var pages = Enumerable.Range(0, n).Select(t => (int)first + t * stride).ToArray();
            var label = new List<string>();
            long rest = first;
            var positions = new int[layout.Axes.Length];
            for (int k = layout.Axes.Length - 1; k >= 0; k--) { positions[k] = (int)(rest % layout.Shape[k]); rest /= layout.Shape[k]; }
            for (int k = 0; k < layout.Axes.Length; k++) if (k != a) label.Add($"{layout.Axes[k]}={positions[k]}");
            yield return (string.Join(",", label), pages);
        }
    }

    /// <summary>
    /// The pages holding <paramref name="page"/>'s neighbours (t − 1, t + 1) along <paramref name="axis"/>, every other
    /// axis held fixed, reflected at the series' ends: (1, 1) at the first frame, (n − 2, n − 2) at the last.
    /// </summary>
    /// <exception cref="ArgumentException">The axis isn't in the layout, or has a single position.</exception>
    public static int[] NeighbourPages(StackLayout layout, char axis, int page)
    {
        int a = layout.Axes.IndexOf(axis);
        if (a < 0) throw new ArgumentException($"The stack has no {axis} axis ({layout}).", nameof(axis));
        if ((uint)page >= (ulong)layout.PlaneCount) throw new ArgumentOutOfRangeException(nameof(page));
        // Row-major: the stride of axis a is the product of the sizes after it.
        int stride = 1;
        for (int k = layout.Axes.Length - 1; k > a; k--) stride *= layout.Shape[k];
        int n = layout.Shape[a], t = page / stride % n;
        var neighbours = Sequences.NeighbourIndices(t, n);
        return [page + (neighbours[0] - t) * stride, page + (neighbours[1] - t) * stride];
    }
}

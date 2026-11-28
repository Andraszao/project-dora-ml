using System.Collections.Concurrent;

namespace DoraBio.Core.Nn;

/// <summary>
/// Reuses <see cref="DenoisingNet.Workspace"/> buffers across parallel loops to avoid GC churn.
/// A workspace holds several MB of activations, and allocating one per tile or per training step
/// would put heavy pressure on the garbage collector.
/// </summary>
/// <remarks>Typical use: <c>Parallel.For(0, n, pool.Rent, (i, _, ws) => { …; return ws; }, pool.Return)</c>.</remarks>
internal sealed class WorkspacePool
{
    private readonly DenoisingNet _net;
    private readonly ConcurrentBag<DenoisingNet.Workspace> _free = new();

    /// <param name="net">The network whose architecture the workspaces are sized for.</param>
    public WorkspacePool(DenoisingNet net) => _net = net;

    private int _created;

    /// <summary>Workspaces this pool has created so far (each one stays in the pool): at most one per worker that ever ran.</summary>
    public int Created => Volatile.Read(ref _created);

    /// <summary>Takes a free workspace, or creates one if none is free. Thread-safe.</summary>
    public DenoisingNet.Workspace Rent()
    {
        if (_free.TryTake(out var ws)) return ws;
        Interlocked.Increment(ref _created);
        return _net.CreateWorkspace();
    }

    /// <summary>
    /// The options for a loop that rents from a pool: at most one worker per core. Without the cap the thread pool adds
    /// workers over a long run, and each one rents (and the pool then keeps) another workspace of tens of MB.
    /// </summary>
    public static ParallelOptions OnePerCore(CancellationToken cancellation = default) =>
        new() { CancellationToken = cancellation, MaxDegreeOfParallelism = Environment.ProcessorCount };

    /// <summary>Gives a workspace back for reuse. Thread-safe.</summary>
    public void Return(DenoisingNet.Workspace workspace) => _free.Add(workspace);
}

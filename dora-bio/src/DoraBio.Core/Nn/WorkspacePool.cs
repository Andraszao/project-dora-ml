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

    /// <summary>Takes a free workspace, or creates one if none is free. Thread-safe.</summary>
    public DenoisingNet.Workspace Rent() => _free.TryTake(out var ws) ? ws : _net.CreateWorkspace();

    /// <summary>Gives a workspace back for reuse. Thread-safe.</summary>
    public void Return(DenoisingNet.Workspace workspace) => _free.Add(workspace);
}

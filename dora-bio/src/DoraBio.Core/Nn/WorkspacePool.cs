using System.Collections.Concurrent;

namespace DoraBio.Core.Nn;

/// <summary>Reuses <see cref="DenoisingNet.Workspace"/> buffers across parallel loops to avoid GC churn.</summary>
internal sealed class WorkspacePool
{
    private readonly DenoisingNet _net;
    private readonly ConcurrentBag<DenoisingNet.Workspace> _free = new();

    public WorkspacePool(DenoisingNet net) => _net = net;

    public DenoisingNet.Workspace Rent() => _free.TryTake(out var ws) ? ws : _net.CreateWorkspace();

    public void Return(DenoisingNet.Workspace workspace) => _free.Add(workspace);
}

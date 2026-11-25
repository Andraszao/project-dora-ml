using System.Text.Json;
using System.Text.Json.Serialization;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

public sealed class JourneySession
{
    public int SessionNumber { get; set; }
    public string Status { get; set; } = "running";
    public DateTime StartedUtc { get; set; }
    public DateTime? EndedUtc { get; set; }
    public double DurationSeconds { get; set; }
    public string Dataset { get; set; } = "";
    public NetConfig? Network { get; set; }
    public TrainerConfig? Config { get; set; }
    public double? NoisyBaselinePsnr { get; set; }
    public List<EpochReport> Epochs { get; set; } = new();
    public int? BestEpoch { get; set; }
    public string? StopReason { get; set; }
    public bool Mastered { get; set; }
}

public sealed class JourneyDocument
{
    public int FormatVersion { get; set; } = 1;
    public DateTime? LastUpdatedUtc { get; set; }
    public List<JourneySession> Sessions { get; set; } = new();
}

/// <summary>
/// Human-readable JSON history of every training session, the v2 replacement for dora_training_journey.json.
///
/// Fixed versus v1: a session is opened only when training actually starts, and is never opened for
/// inference-only launches. Every epoch is recorded as it finishes. The file is rewritten after each epoch
/// (via temp file and rename), so even an interrupted run leaves a readable "running" or "failed" record.
/// </summary>
public sealed class JourneyLog
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() },
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public string FilePath { get; }
    public JourneyDocument Document { get; }

    private JourneyLog(string filePath, JourneyDocument document)
    {
        FilePath = filePath;
        Document = document;
    }

    public static JourneyLog Open(string filePath)
    {
        if (!File.Exists(filePath)) return new JourneyLog(filePath, new JourneyDocument());
        var document = JsonSerializer.Deserialize<JourneyDocument>(File.ReadAllText(filePath), JsonOptions)
                       ?? throw new InvalidDataException($"Could not read journey log {filePath}.");
        return new JourneyLog(filePath, document);
    }

    public JourneySession BeginSession(string dataset, NetConfig network, TrainerConfig config)
    {
        var session = new JourneySession
        {
            SessionNumber = Document.Sessions.Count == 0 ? 1 : Document.Sessions.Max(s => s.SessionNumber) + 1,
            StartedUtc = DateTime.UtcNow,
            Dataset = dataset,
            Network = network,
            Config = config,
        };
        Document.Sessions.Add(session);
        Save();
        return session;
    }

    public void RecordEpoch(JourneySession session, EpochReport report)
    {
        session.Epochs.Add(report);
        session.DurationSeconds = (DateTime.UtcNow - session.StartedUtc).TotalSeconds;
        Save();
    }

    public void Complete(JourneySession session, TrainingResult result)
    {
        session.Status = "completed";
        session.EndedUtc = DateTime.UtcNow;
        session.DurationSeconds = (session.EndedUtc.Value - session.StartedUtc).TotalSeconds;
        session.NoisyBaselinePsnr = result.NoisyBaselinePsnr;
        session.BestEpoch = result.BestEpoch;
        session.StopReason = result.StopReason;
        session.Mastered = result.Mastered;
        Save();
    }

    public void Fail(JourneySession session, Exception error)
    {
        session.Status = $"failed: {error.Message}";
        session.EndedUtc = DateTime.UtcNow;
        session.DurationSeconds = (session.EndedUtc.Value - session.StartedUtc).TotalSeconds;
        Save();
    }

    public void Save()
    {
        Document.LastUpdatedUtc = DateTime.UtcNow;
        string fullPath = Path.GetFullPath(FilePath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        string temp = fullPath + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(Document, JsonOptions));
        File.Move(temp, fullPath, overwrite: true);
    }
}

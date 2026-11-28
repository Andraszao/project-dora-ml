using System.Text.Json;
using System.Text.Json.Serialization;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

/// <summary>One training run as stored in the journey log. Plain settable properties, so it round-trips through JSON.</summary>
public sealed class JourneySession
{
    /// <summary>1-based, increasing across the file's lifetime (never reused after edits).</summary>
    public int SessionNumber { get; set; }

    /// <summary>"running" while training, then "completed" or "failed: &lt;message&gt;".</summary>
    /// <remarks>A session left as "running" means the process died mid-training (crash, kill, power loss).</remarks>
    public string Status { get; set; } = "running";

    /// <summary>When training began.</summary>
    public DateTime StartedUtc { get; set; }

    /// <summary>When training ended; null while running.</summary>
    public DateTime? EndedUtc { get; set; }

    /// <summary>Wall-clock seconds so far, updated after every epoch (v1 always wrote 0 here).</summary>
    public double DurationSeconds { get; set; }

    /// <summary>Free-text description of the training data.</summary>
    public string Dataset { get; set; } = "";

    /// <summary>Architecture that was trained.</summary>
    public NetConfig? Network { get; set; }

    /// <summary>Every training setting used.</summary>
    public TrainerConfig? Config { get; set; }

    /// <summary>PSNR of the unprocessed noisy validation data, the reference that the epoch gains are measured from.</summary>
    public double? NoisyBaselinePsnr { get; set; }

    /// <summary>Each epoch's report, appended as it finishes.</summary>
    public List<EpochReport> Epochs { get; set; } = new();

    /// <summary>Epoch whose weights were kept; set on completion.</summary>
    public int? BestEpoch { get; set; }

    /// <summary>Why training ended (see <see cref="TrainingResult.StopReason"/>); set on completion.</summary>
    public string? StopReason { get; set; }

    /// <summary>Whether the final progressive target was held.</summary>
    public bool Mastered { get; set; }
}

/// <summary>Root object of the journey JSON file.</summary>
public sealed class JourneyDocument
{
    /// <summary>Schema version of this file, for future migrations.</summary>
    public int FormatVersion { get; set; } = 1;

    /// <summary>When the file was last written.</summary>
    public DateTime? LastUpdatedUtc { get; set; }

    /// <summary>All sessions, oldest first.</summary>
    public List<JourneySession> Sessions { get; set; } = new();
}

/// <summary>
/// Human-readable JSON history of every training session, the v2 replacement for dora_training_journey.json.
///
/// Fixed versus v1: a session is opened only when training actually starts, and is never opened for
/// inference-only launches. Every epoch is recorded as it finishes. The file is rewritten after each epoch
/// (via temp file and rename), so even an interrupted run leaves a readable "running" or "failed" record.
/// </summary>
/// <remarks>
/// Lifecycle: <see cref="Open"/>, then <see cref="BeginSession"/>, then <see cref="RecordEpoch"/> once per epoch,
/// then <see cref="Complete"/> or <see cref="Fail"/>. <see cref="TrainingJob"/> drives this, so callers rarely
/// touch it directly. Not thread-safe; use it from the training thread only.
/// </remarks>
public sealed class JourneyLog
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,                                        // meant to be read by people, too
        Converters = { new JsonStringEnumConverter() },              // "Noise2Void", not 0
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,   // PSNR can be +∞ for identical images
    };

    /// <summary>Path of the JSON file (created on the first save if it doesn't exist).</summary>
    public string FilePath { get; }

    /// <summary>The in-memory document; written back by <see cref="Save"/>.</summary>
    public JourneyDocument Document { get; }

    private JourneyLog(string filePath, JourneyDocument document)
    {
        FilePath = filePath;
        Document = document;
    }

    /// <summary>Loads an existing journey file, or starts an empty one if the file doesn't exist yet.</summary>
    /// <exception cref="InvalidDataException">The file exists but isn't a journey document.</exception>
    public static JourneyLog Open(string filePath)
    {
        if (!File.Exists(filePath)) return new JourneyLog(filePath, new JourneyDocument());
        var document = JsonSerializer.Deserialize<JourneyDocument>(File.ReadAllText(filePath), JsonOptions)
                       ?? throw new InvalidDataException($"Could not read journey log {filePath}.");
        return new JourneyLog(filePath, document);
    }

    /// <summary>Appends a new "running" session and saves immediately, so even a crash in epoch 1 leaves a trace.</summary>
    /// <param name="dataset">Description of the training data.</param>
    /// <param name="network">Architecture being trained.</param>
    /// <param name="config">Training settings.</param>
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

    /// <summary>Appends one epoch's report to <paramref name="session"/>, updates its duration, and saves.</summary>
    public void RecordEpoch(JourneySession session, EpochReport report)
    {
        session.Epochs.Add(report);
        session.DurationSeconds = (DateTime.UtcNow - session.StartedUtc).TotalSeconds;
        Save();
    }

    /// <summary>Marks the session completed with its final results and saves.</summary>
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

    /// <summary>Marks the session failed with the exception message and saves. The caller rethrows.</summary>
    public void Fail(JourneySession session, Exception error)
    {
        session.Status = $"failed: {error.Message}";
        session.EndedUtc = DateTime.UtcNow;
        session.DurationSeconds = (session.EndedUtc.Value - session.StartedUtc).TotalSeconds;
        Save();
    }

    /// <summary>Writes the whole document atomically: write a temp file, then rename it over the old one.</summary>
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

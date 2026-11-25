using System.Globalization;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

public sealed record TrainingJobOptions
{
    /// <summary>Short description of the data, stored in the journey log and in the weights' provenance.</summary>
    public required string DatasetDescription { get; init; }

    /// <summary>Where to write the best weights. Written at each new best epoch and again at the end. Null skips saving.</summary>
    public string? WeightsPath { get; init; }

    /// <summary>Journey log file. Null skips logging.</summary>
    public string? JournalPath { get; init; }
}

/// <summary>
/// Shared training entry point for the CLI and the Godot bridge, so both get the same lifecycle:
/// open a journey session, train, record every epoch, checkpoint the best weights, then close the session.
/// </summary>
public static class TrainingJob
{
    public static TrainingResult Run(
        DenoisingNet net,
        IReadOnlyList<ImagePair> dataset,
        TrainerConfig config,
        TrainingJobOptions options,
        Action<EpochReport>? onEpoch = null,
        CancellationToken cancellation = default)
    {
        var journal = options.JournalPath is null ? null : JourneyLog.Open(options.JournalPath);
        var session = journal?.BeginSession(options.DatasetDescription, net.Config, config);

        try
        {
            var result = new Trainer(net, config).Train(dataset, report =>
            {
                if (journal is not null) journal.RecordEpoch(session!, report);
                if (report.IsBest && options.WeightsPath is not null)
                    WeightsFile.Save(options.WeightsPath, net, Provenance(config, options, report, final: false));
                onEpoch?.Invoke(report);
            }, cancellation);

            // The trainer has restored the best weights into the network; save those.
            if (options.WeightsPath is not null && result.Best is { } best)
                WeightsFile.Save(options.WeightsPath, net, Provenance(config, options, best, final: true));
            journal?.Complete(session!, result);
            return result;
        }
        catch (Exception e)
        {
            journal?.Fail(session!, e);
            throw;
        }
    }

    private static string Provenance(TrainerConfig config, TrainingJobOptions options, EpochReport report, bool final)
    {
        var text = FormattableString.Invariant(
            $"{config.Mode} on {options.DatasetDescription}; {(final ? "best" : "checkpoint at")} epoch {report.Epoch}; val loss {report.ValLoss:G4}");
        if (report.ValPsnr is { } psnr)
            text += string.Create(CultureInfo.InvariantCulture, $"; val PSNR {psnr:0.00} dB (+{report.ValPsnrGain ?? 0:0.00} dB over noisy)");
        return text;
    }
}

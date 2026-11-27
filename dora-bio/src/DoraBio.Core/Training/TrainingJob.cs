using System.Globalization;
using DoraBio.Core.Data;
using DoraBio.Core.IO;
using DoraBio.Core.Nn;

namespace DoraBio.Core.Training;

/// <summary>Where a <see cref="TrainingJob"/> writes its outputs, and how it describes the data.</summary>
public sealed record TrainingJobOptions
{
    /// <summary>Short description of the data, stored in the journey log and in the weights' provenance.</summary>
    public required string DatasetDescription { get; init; }

    /// <summary>Where to write the best weights. Written at each new best epoch and again at the end. Null skips saving.</summary>
    public string? WeightsPath { get; init; }

    /// <summary>Journey log file. Null skips logging.</summary>
    public string? JournalPath { get; init; }

    /// <summary>Also write the final epoch's weights here (whatever <see cref="WeightsPath"/> keeps). Null skips it.</summary>
    public string? LastWeightsPath { get; init; }

    /// <summary>
    /// Validation images chosen in advance (normalized), such as brick 40's held-out block of a time-lapse; then the
    /// whole dataset trains. Null (the default) splits the dataset by FOV, as always.
    /// </summary>
    public IReadOnlyList<ImagePair>? Validation { get; init; }
}

/// <summary>
/// Shared training entry point for the CLI and the Godot bridge, so both get the same lifecycle:
/// open a journey session, train, record every epoch, checkpoint the best weights, then close the session.
/// </summary>
public static class TrainingJob
{
    /// <summary>Trains <paramref name="net"/> in place and persists the results according to <paramref name="options"/>.</summary>
    /// <param name="net">The network to train (ends up holding its best weights).</param>
    /// <param name="dataset">Normalized image pairs.</param>
    /// <param name="config">Training settings.</param>
    /// <param name="options">Output paths and dataset description.</param>
    /// <param name="onEpoch">Called on the training thread after each epoch has been journaled and checkpointed.</param>
    /// <param name="cancellation">Stops after the current step. The session is still completed and the best weights saved.</param>
    /// <remarks>
    /// Saving weights at each new best epoch, not only at the end, means a crash or kill mid-run loses at most
    /// the epochs since the last improvement. If training throws, the journey session is marked failed and the
    /// exception is rethrown.
    /// </remarks>
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
            void Epoch(EpochReport report)
            {
                if (journal is not null) journal.RecordEpoch(session!, report);
                if (report.IsBest && !config.KeepLastEpoch && options.WeightsPath is not null)
                    WeightsFile.Save(options.WeightsPath, net, Provenance(config, options, report, "checkpoint at"));
                onEpoch?.Invoke(report);
            }
            var trainer = new Trainer(net, config);
            var result = options.Validation is null
                ? trainer.Train(dataset, Epoch, cancellation)
                : trainer.Train(dataset, options.Validation, Epoch, cancellation);

            // The trainer has restored the weights it keeps into the network (the best epoch's, or the last's); save those.
            var kept = config.KeepLastEpoch ? result.Last : result.Best;
            if (options.WeightsPath is not null && kept is not null)
                WeightsFile.Save(options.WeightsPath, net, Provenance(config, options, kept, config.KeepLastEpoch ? "last" : "best"));
            if (options.LastWeightsPath is not null && result is { LastParameters: { } lastParameters, Last: { } lastReport })
            {
                var copy = DenoisingNet.Create(net.Config, seed: 0);
                Array.Copy(lastParameters, copy.Parameters, lastParameters.Length);
                WeightsFile.Save(options.LastWeightsPath, copy, Provenance(config, options, lastReport, "last"));
            }
            journal?.Complete(session!, result);
            return result;
        }
        catch (Exception e)
        {
            journal?.Fail(session!, e);
            throw;
        }
    }

    /// <summary>
    /// The text stored in the weights file, e.g. "Noise2Void on …; best epoch 25; val loss 0.0104; val PSNR 28.67 dB
    /// (+8.11 dB over noisy)". Formatted with the invariant culture, so the file reads the same on every machine.
    /// </summary>
    private static string Provenance(TrainerConfig config, TrainingJobOptions options, EpochReport report, string which)
    {
        var text = FormattableString.Invariant(
            $"{config.Mode} on {options.DatasetDescription}; {which} epoch {report.Epoch}; val loss {report.ValLoss:G4}");
        if (report.ValPsnr is { } psnr)
            text += string.Create(CultureInfo.InvariantCulture, $"; val PSNR {psnr:0.00} dB (+{report.ValPsnrGain ?? 0:0.00} dB over noisy)");
        return text;
    }
}

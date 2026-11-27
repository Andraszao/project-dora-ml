using System.Buffers.Binary;
using System.Text;
using DoraBio.Core.Nn;

namespace DoraBio.Core.IO;

/// <summary>Information stored alongside the parameters.</summary>
/// <param name="Config">Architecture the parameters belong to.</param>
/// <param name="ModelVersion">Format/model generation that wrote the file, e.g. "2.0.0-bio".</param>
/// <param name="Provenance">
/// Free text saying what the weights were trained on and how well they scored. Shown in the UI and by
/// <c>dora-bio info</c>, so users can tell synthetic-data weights from weights trained on their own data.
/// </param>
/// <param name="SavedUtc">When the file was written.</param>
public sealed record ModelInfo(NetConfig Config, string ModelVersion, string Provenance, DateTime SavedUtc);

/// <summary>
/// Compact binary model file (about 4 bytes per parameter, roughly 106 KB for the default network).
///
/// Layout, little-endian:
///   "DORABIO\0"                         8-byte magic
///   u16      format version (2; 3 for a noise-conditioned network; 4 for a multi-frame network; format 1 is still read)
///   string   model version, e.g. "2.0.0-bio"   (BinaryWriter length-prefixed UTF-8)
///   string   provenance: free text saying what the weights were trained on
///   i64      saved time, UTC ticks
///   u8       architecture: 0 = dilated, 1 = U-Net                    (format 2 only)
///   dilated: i32 features, i32 layer count, i32[layer count] dilations
///   U-Net:   i32 features, i32 depth
///   format 3 only: u8 conditioning (1 = noise level, FiLM), u16 generator hidden units (16)
///   format 4 only: u8 input frames (3; brick 39's spatiotemporal blind spot)
///   i32      parameter count, f32[parameter count] parameters
///   u32      CRC-32 of every preceding byte
/// Format 1 is format 2 without the architecture byte, so it always describes a dilated network (the bundled model,
/// and every file written before P4). Plain networks keep writing format 2, so their files are unchanged by Q13 and by
/// brick 39 (single-frame networks never write format 4).
/// </summary>
public static class WeightsFile
{
    /// <summary>Model version written into new files. Readers accept any "2.x" version.</summary>
    public const string ModelVersion = "2.0.0-bio";
    private const ushort FormatVersion = 2, ConditionedFormat = 3, MultiFrameFormat = 4;
    private static ReadOnlySpan<byte> Magic => "DORABIO\0"u8;

    /// <summary>Serializes the network to a stream (timestamped now).</summary>
    /// <param name="stream">Destination; written at its current position and left open.</param>
    /// <param name="net">The network whose architecture and parameters are stored.</param>
    /// <param name="provenance">What the weights were trained on (see <see cref="ModelInfo.Provenance"/>).</param>
    public static void Save(Stream stream, DenoisingNet net, string provenance)
    {
        byte[] bytes = Serialize(net, provenance, DateTime.UtcNow);
        stream.Write(bytes);
    }

    /// <summary>Saves to a file atomically, creating the directory if needed.</summary>
    /// <param name="path">Destination file; replaced only once the new file is fully written.</param>
    /// <param name="net">The network whose architecture and parameters are stored.</param>
    /// <param name="provenance">What the weights were trained on (see <see cref="ModelInfo.Provenance"/>).</param>
    public static void Save(string path, DenoisingNet net, string provenance)
    {
        // Write to a temp file and then rename, so a crash never leaves a half-written model behind.
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        string temp = fullPath + ".tmp";
        File.WriteAllBytes(temp, Serialize(net, provenance, DateTime.UtcNow));
        File.Move(temp, fullPath, overwrite: true);
    }

    /// <summary>Reads and validates a weights file from disk.</summary>
    public static (DenoisingNet Net, ModelInfo Info) Load(string path) => Load(File.ReadAllBytes(path));

    /// <summary>
    /// Parses and validates an in-memory weights file: magic, CRC, version, architecture, and that the parameter
    /// count matches the architecture. Returns a ready-to-use network.
    /// </summary>
    /// <exception cref="InvalidDataException">Not a v2 weights file, corrupted, or internally inconsistent.</exception>
    public static (DenoisingNet Net, ModelInfo Info) Load(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < Magic.Length + 6 || !bytes[..Magic.Length].SequenceEqual(Magic))
            throw new InvalidDataException("Not a DoraBio weights file (bad magic). v1 .dat JSON weights cannot be loaded.");

        uint stored = BinaryPrimitives.ReadUInt32LittleEndian(bytes[^4..]);
        if (Crc32.Compute(bytes[..^4]) != stored)
            throw new InvalidDataException("Weights file is corrupted (CRC mismatch).");

        using var reader = new BinaryReader(new MemoryStream(bytes[..^4].ToArray()), Encoding.UTF8);
        reader.ReadBytes(Magic.Length);

        ushort format = reader.ReadUInt16();
        if (format is not (1 or FormatVersion or ConditionedFormat or MultiFrameFormat)) throw new InvalidDataException($"Unsupported weights format version {format}.");

        string modelVersion = reader.ReadString();
        if (!modelVersion.StartsWith("2.", StringComparison.Ordinal))
            throw new InvalidDataException($"Weights are model version {modelVersion}; this build reads 2.x.");

        string provenance = reader.ReadString();
        var saved = new DateTime(reader.ReadInt64(), DateTimeKind.Utc);

        var architecture = format == 1 ? NetArchitecture.Dilated : (NetArchitecture)reader.ReadByte();
        NetConfig config;
        int features = reader.ReadInt32();
        if (architecture == NetArchitecture.UNet)
            config = NetConfig.UNet(features, reader.ReadInt32());
        else if (architecture == NetArchitecture.Dilated)
        {
            int layers = reader.ReadInt32();
            if (layers is < 2 or > 64) throw new InvalidDataException($"Implausible layer count {layers}.");
            var dilations = new int[layers];
            for (int i = 0; i < layers; i++) dilations[i] = reader.ReadInt32();
            config = new NetConfig(features, dilations);
        }
        else throw new InvalidDataException($"Unknown architecture {(int)architecture}.");
        if (format == ConditionedFormat)
        {
            var conditioning = (NetConditioning)reader.ReadByte();
            if (conditioning != NetConditioning.NoiseLevel) throw new InvalidDataException($"Unknown conditioning {(int)conditioning}.");
            int hidden = reader.ReadUInt16();
            if (hidden != DenoisingNet.FilmHidden)
                throw new InvalidDataException($"Conditioning network has {hidden} hidden units; this build supports {DenoisingNet.FilmHidden}.");
            config = config with { Conditioning = conditioning };
        }
        if (format == MultiFrameFormat)
        {
            int frames = reader.ReadByte();
            if (frames < 3) throw new InvalidDataException($"A multi-frame weights file says {frames} input frames.");
            config = config with { InputFrames = frames };
        }
        try { config.Validate(); }
        catch (ArgumentException e) { throw new InvalidDataException($"Invalid architecture in weights file: {e.Message}"); }

        int count = reader.ReadInt32();
        int expected = DenoisingNet.CountParameters(config);
        if (count != expected)
            throw new InvalidDataException($"Weights hold {count} parameters but the stored architecture needs {expected}.");

        var parameters = new float[count];
        for (int i = 0; i < count; i++) parameters[i] = reader.ReadSingle();
        if (reader.BaseStream.Position != reader.BaseStream.Length)
            throw new InvalidDataException("Unexpected trailing data in weights file.");

        return (DenoisingNet.FromParameters(config, parameters), new ModelInfo(config, modelVersion, provenance, saved));
    }

    /// <summary>Builds the complete file contents, CRC included. Internal so tests can pin the timestamp.</summary>
    internal static byte[] Serialize(DenoisingNet net, string provenance, DateTime savedUtc)
    {
        using var memory = new MemoryStream();
        using (var writer = new BinaryWriter(memory, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(Magic);
            writer.Write(net.Config.IsConditioned ? ConditionedFormat : net.Config.InputFrames > 1 ? MultiFrameFormat : FormatVersion);
            writer.Write(ModelVersion);
            writer.Write(provenance ?? "");
            writer.Write(savedUtc.ToUniversalTime().Ticks);
            writer.Write((byte)net.Config.Architecture);
            writer.Write(net.Config.Features);
            if (net.Config.Architecture == NetArchitecture.UNet) writer.Write(net.Config.Depth);
            else
            {
                writer.Write(net.Config.LayerCount);
                foreach (int d in net.Config.Dilations) writer.Write(d);
            }
            if (net.Config.IsConditioned)
            {
                writer.Write((byte)net.Config.Conditioning);
                writer.Write((ushort)DenoisingNet.FilmHidden);
            }
            if (net.Config.InputFrames > 1) writer.Write((byte)net.Config.InputFrames);
            writer.Write(net.ParameterCount);
            foreach (float p in net.Parameters) writer.Write(p);
        }
        uint crc = Crc32.Compute(memory.GetBuffer().AsSpan(0, (int)memory.Length));
        Span<byte> crcBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(crcBytes, crc);
        memory.Write(crcBytes);
        return memory.ToArray();
    }
}

namespace DoraBio.Core.IO;

/// <summary>Standard CRC-32 (IEEE 802.3, the one PNG and zip use).</summary>
/// <remarks>
/// Table-driven, reflected form with polynomial 0xEDB88320. Used to verify every PNG chunk and to protect
/// weights files against truncation or bit rot. Written out here because the core takes no dependencies
/// (System.IO.Hashing would be a NuGet package).
/// </remarks>
internal static class Crc32
{
    private static readonly uint[] Table = BuildTable();

    /// <summary>Precomputes the CRC of every possible byte value, so each byte later costs one lookup.</summary>
    private static uint[] BuildTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }

    /// <summary>CRC-32 of <paramref name="data"/> (initial value and final XOR 0xFFFFFFFF, as PNG specifies).</summary>
    public static uint Compute(ReadOnlySpan<byte> data) => Update(0xFFFFFFFFu, data) ^ 0xFFFFFFFFu;

    /// <summary>Feeds more bytes into a running (pre-inversion) CRC state, for checksumming data in pieces.</summary>
    public static uint Update(uint crc, ReadOnlySpan<byte> data)
    {
        foreach (byte b in data)
            crc = Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc;
    }
}

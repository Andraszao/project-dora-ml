namespace DoraBio.Core.IO;

/// <summary>What decoding one LZW strip went through: the widest code read and how often the table was reset.</summary>
/// <param name="MaxCodeWidth">Widest code read, in bits (9 to 12).</param>
/// <param name="TableResets">Clear codes after the first (each starts the table over at 9-bit codes).</param>
internal readonly record struct LzwStats(int MaxCodeWidth, int TableResets);

/// <summary>
/// TIFF's LZW (compression 5, TIFF 6.0 section 13), as libtiff decodes it (brick 41: Fiji and many acquisition
/// programs write it). Codes are packed most-significant bit first and start 9 bits wide; 256 clears the table and
/// 257 ends the strip; new entries start at 258, and the code width grows to 10, 11 and 12 bits one entry early, when
/// the next free entry reaches 511, 1023 and 2047. The encoder clears the table before it overflows 4096 entries.
/// The predictor (tag 317) is undone afterwards, as for every compression.
/// <para>
/// Old-style LZW (before TIFF 6.0: codes packed least-significant bit first, written by some 1990s software) is
/// refused rather than guessed at; libtiff tells it apart by its first two bytes, and so does this.
/// </para>
/// </summary>
internal static class Lzw
{
    private const int Clear = 256, EndOfInformation = 257, FirstFree = 258, MaxWidth = 12, TableSize = 1 << MaxWidth;

    /// <summary>Decodes one strip into exactly <paramref name="expected"/> bytes.</summary>
    /// <exception cref="NotSupportedException">Old-style (pre-TIFF 6.0) LZW.</exception>
    /// <exception cref="InvalidDataException">A corrupt code stream, or one that ends before <paramref name="expected"/> bytes.</exception>
    public static byte[] Decode(ReadOnlySpan<byte> data, int expected) => Decode(data, expected, out _);

    /// <summary>Decodes one strip and reports the code widths and table resets it went through (for tests).</summary>
    public static byte[] Decode(ReadOnlySpan<byte> data, int expected, out LzwStats stats)
    {
        if (data.Length >= 2 && data[0] == 0 && (data[1] & 1) != 0)
            throw new NotSupportedException("This TIFF uses old-style LZW (from before TIFF 6.0), which DORA doesn't read; re-save it " +
                                            "uncompressed or with LZW from current software (Fiji, tifffile).");
        // Entry k is entry prefix[k] followed by the byte suffix[k]; first[k] is its first byte and length[k] its length.
        var prefix = new short[TableSize];
        var suffix = new byte[TableSize];
        var first = new byte[TableSize];
        var length = new int[TableSize];
        for (int k = 0; k < 256; k++) (suffix[k], first[k], length[k], prefix[k]) = ((byte)k, (byte)k, 1, -1);

        var output = new byte[expected];
        int written = 0, next = FirstFree, width = 9, maxWidth = 9, clears = 0, previous = -1;
        long bit = 0;

        while (written < expected)
        {
            int code = ReadCode(data, ref bit, width);
            if (code == EndOfInformation) break;
            if (code == Clear)
            {
                if (bit > 9) clears++;   // a strip starts with a clear too; count only the resets after it
                next = FirstFree;
                width = 9;
                do code = ReadCode(data, ref bit, width); while (code == Clear);
                if (code == EndOfInformation) break;
                if (code > 255) throw new InvalidDataException("Corrupt LZW data: the first code after a clear isn't a byte.");
                written = Emit(code, output, written, prefix, suffix, length);
                previous = code;
                continue;
            }
            if (previous < 0) throw new InvalidDataException("Corrupt LZW data: the strip doesn't start with a clear code.");
            if (code > next) throw new InvalidDataException($"Corrupt LZW data: code {code} isn't in the table yet.");
            if (next < TableSize)
            {
                // The new entry is the previous string plus the first byte of this one, or of itself when this code is the
                // entry being defined (the "KwKwK" case, code == next).
                prefix[next] = (short)previous;
                suffix[next] = code == next ? first[previous] : first[code];
                first[next] = first[previous];
                length[next] = length[previous] + 1;
                next++;
                if (next >= (1 << width) - 1 && width < MaxWidth) maxWidth = Math.Max(maxWidth, ++width);
            }
            written = Emit(code, output, written, prefix, suffix, length);
            previous = code;
        }
        if (written < expected)
            throw new InvalidDataException($"LZW strip is truncated: {written} of {expected} bytes.");
        stats = new LzwStats(maxWidth, clears);
        return output;
    }

    /// <summary>The next <paramref name="width"/>-bit code, most significant bit first; the end code once the data runs out.</summary>
    private static int ReadCode(ReadOnlySpan<byte> data, ref long bit, int width)
    {
        if (bit + width > (long)data.Length * 8) return EndOfInformation;   // a strip that stops without its end code ends here
        int code = 0;
        for (int b = 0; b < width; b++, bit++) code = (code << 1) | ((data[(int)(bit >> 3)] >> (7 - (int)(bit & 7))) & 1);
        return code;
    }

    /// <summary>
    /// Writes entry <paramref name="code"/> at <paramref name="written"/>, walking its prefix chain from the last byte back;
    /// bytes past the end of the strip (a last string may run over) are dropped. Returns the new write position.
    /// </summary>
    private static int Emit(int code, byte[] output, int written, short[] prefix, byte[] suffix, int[] length)
    {
        int end = written + length[code];
        for (int at = end - 1, k = code; at >= written; at--, k = prefix[k])
            if (at < output.Length) output[at] = suffix[k];
        return Math.Min(end, output.Length);
    }
}

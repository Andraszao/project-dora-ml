using DoraBio.Core.Data;
using DoraBio.Core.Nn;
using DoraBio.Core.Training;
using Xunit;

namespace DoraBio.Core.Tests;

/// <summary>
/// The register-blocked kernels must be bit-identical to the reference row kernels: same terms, same order, same
/// separate multiply and add. Compared bit for bit (not with a tolerance), for single layers of awkward shapes, whole
/// networks of both architectures, and a complete training run.
/// </summary>
[Trait(Bricks.Trait, Bricks.BlockedConv)]
public class BlockedConvTests
{
    private static void AssertBitIdentical(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: reference {expected[i]:R} vs blocked {actual[i]:R}");
    }

    [Theory]
    [InlineData(1, 24, 1, 9, 13)]     // first dilated layer: one input channel
    [InlineData(24, 24, 2, 17, 8)]    // a middle layer, width exactly one SIMD strip
    [InlineData(24, 1, 4, 11, 21)]    // output layer: one channel, ragged width
    [InlineData(3, 5, 1, 4, 3)]       // fewer channels than a block, narrower than a strip
    [InlineData(16, 32, 1, 12, 19)]   // U-Net encoder shape
    [InlineData(48, 16, 3, 10, 10)]   // U-Net decoder shape (stacked skip input)
    [InlineData(9, 8, 2, 7, 33)]      // exactly one output block, odd input channels, several strips plus a tail
    public void ALayerMatchesTheReferenceBitForBit(int inChannels, int outChannels, int dilation, int height, int width)
    {
        var layer = new ConvLayer(inChannels, outChannels, dilation, weightOffset: 5);
        var parameters = TestUtil.RandomPlane(5 + layer.ParameterCount, 1, -1, 1);
        var input = TestUtil.RandomPlane(inChannels * height * width, 2, -1, 1);
        var padded = new float[layer.PaddedLength(height, width)];
        layer.Pad(input, height, width, padded);

        var reference = new float[outChannels * height * width];
        var blocked = new float[reference.Length];
        layer.ForwardReference(parameters, padded, height, width, reference);
        layer.Forward(parameters, padded, height, width, blocked);
        AssertBitIdentical(reference, blocked, "forward");

        // Backward adds into existing gradients, so start both from the same non-zero buffer.
        var gradOut = TestUtil.RandomPlane(reference.Length, 3, -1, 1);
        var gRef = TestUtil.RandomPlane(parameters.Length, 4, -1, 1);
        var gNew = (float[])gRef.Clone();
        var inRef = new float[padded.Length];
        var inNew = new float[padded.Length];
        layer.BackwardReference(parameters, padded, gradOut, height, width, gRef, inRef);
        layer.Backward(parameters, padded, gradOut, height, width, gNew, inNew);
        AssertBitIdentical(gRef, gNew, "parameter gradients");

        // Only the interior of the input gradient is meaningful (callers discard the padding border).
        var interiorRef = new float[input.Length];
        var interiorNew = new float[input.Length];
        layer.Unpad(inRef, height, width, interiorRef);
        layer.Unpad(inNew, height, width, interiorNew);
        AssertBitIdentical(interiorRef, interiorNew, "input gradient");

        // And skipping the input gradient (the first layer) must not change the parameter gradients.
        var gSkip = TestUtil.RandomPlane(parameters.Length, 4, -1, 1);
        layer.Backward(parameters, padded, gradOut, height, width, gSkip, Span<float>.Empty);
        AssertBitIdentical(gRef, gSkip, "parameter gradients without input gradient");
    }

    [Theory]
    [MemberData(nameof(Networks))]
    public void WholeNetworksMatchTheReferenceBitForBit(NetConfig config)
    {
        const int height = 37, width = 29;   // not a multiple of 4: the U-Net's padding path too
        var blocked = DenoisingNet.Create(config, seed: 3);
        var rng = new Random(2);
        for (int i = 0; i < blocked.ParameterCount; i++) blocked.Parameters[i] += (rng.NextSingle() - 0.5f) * 0.2f;
        var reference = blocked.Clone();
        reference.ReferenceKernels = true;

        var input = TestUtil.RandomPlane(height * width, 4);
        var target = TestUtil.RandomPlane(height * width, 5);
        var mask = TestUtil.RandomPlane(height * width, 6).Select(v => v > 0.7f ? 1f : 0f).ToArray();

        var outRef = new float[input.Length];
        var outNew = new float[input.Length];
        reference.Predict(input, height, width, outRef, reference.CreateWorkspace());
        blocked.Predict(input, height, width, outNew, blocked.CreateWorkspace());
        AssertBitIdentical(outRef, outNew, "prediction");

        var gRef = new float[reference.ParameterCount];
        var gNew = new float[blocked.ParameterCount];
        double lossRef = reference.AccumulateGradients(input, target, mask, height, width, 0.01f, gRef, reference.CreateWorkspace());
        double lossNew = blocked.AccumulateGradients(input, target, mask, height, width, 0.01f, gNew, blocked.CreateWorkspace());
        Assert.Equal(lossRef, lossNew);
        AssertBitIdentical(gRef, gNew, "gradients");
    }

    public static IEnumerable<object[]> Networks() =>
    [
        [NetConfig.Default],
        [NetConfig.UNet(8, 2)],
    ];

    [Fact]
    public void ATrainingRunEndsWithBitIdenticalWeights()
    {
        var data = SyntheticFluorescence.GenerateDataset(count: 4, size: 64, seed: 10);
        var config = new TrainerConfig
        {
            Mode = TrainingMode.Noise2Void, PatchSize = 32, BatchSize = 4, StepsPerEpoch = 5, MaxEpochs = 2,
            ValidationPatches = 4, Seed = 9,
        };
        var blocked = DenoisingNet.Create(new NetConfig(8, [1, 2, 1]), seed: 1);
        var reference = blocked.Clone();
        reference.ReferenceKernels = true;
        new Trainer(blocked, config).Train(data);
        new Trainer(reference, config).Train(data);
        AssertBitIdentical(reference.Parameters, blocked.Parameters, "trained weights");
    }
}

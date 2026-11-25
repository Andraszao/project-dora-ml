using DoraBio.Core.Data;
using DoraBio.Core.Nn;

namespace DoraBio.Core;

/// <summary>
/// Optional preprocessing for Noise2Void: denoise each image in its variance-stabilized domain. Every image is
/// calibrated on its own (<see cref="NoiseCalibration"/>), transformed with the generalized <see cref="Anscombe"/>
/// transform, normalized and denoised; the result is mapped back with the unbiased inverse. A model trained this way
/// must also be applied this way.
/// </summary>
public static class VarianceStabilization
{
    /// <summary>
    /// Transforms a pair in its original units, using the noise model estimated from its noisy image. The clean image,
    /// if any, gets the same (algebraic) transform, so validation PSNR is measured in the transformed domain.
    /// </summary>
    public static ImagePair Stabilize(ImagePair pair)
    {
        var model = NoiseCalibration.Estimate(pair.Noisy);
        return pair with
        {
            Noisy = Anscombe.Forward(pair.Noisy, model),
            Clean = pair.Clean is null ? null : Anscombe.Forward(pair.Clean, model),
        };
    }

    /// <summary>Denoises an image in original units through the stabilized domain; returns the result and the model used.</summary>
    /// <param name="net">A model trained on stabilized images.</param>
    /// <param name="noisy">The noisy image in its original units.</param>
    /// <param name="tileCore">Tile core size for inference.</param>
    /// <param name="cancellation">Checked between tiles.</param>
    public static (GrayImage Denoised, NoiseModel Model) Denoise(DenoisingNet net, GrayImage noisy,
        int tileCore = TiledInference.DefaultTileCore, CancellationToken cancellation = default)
    {
        var model = NoiseCalibration.Estimate(noisy);
        var transformed = Anscombe.Forward(noisy, model);
        var map = Normalization.FromPercentiles(transformed.Pixels);
        var denoised = Normalization.Invert(TiledInference.Denoise(net, Normalization.Apply(transformed, map), tileCore, cancellation), map);
        return (Anscombe.InverseUnbiased(denoised, model), model);
    }
}

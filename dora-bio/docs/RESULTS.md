# DORA v2-bio on real fluorescence microscopy (FMD)

First measurements of DORA on real microscope images: phase P2 of [`PATHWAY.md`](../PATHWAY.md).
Measured 2026-11-25 on a 4-vCPU cloud container (no GPU).

## Data

**FMD**: Zhang, Zhu, Nichols, Wang, Zhang, Smith, Howard, *"A Poisson-Gaussian Denoising Dataset with Real
Fluorescence Microscopy Images"*, CVPR 2019 ([arXiv:1812.10366](https://arxiv.org/abs/1812.10366)). Licensed
**CC BY-SA 4.0**. Images are not in this repo; `tools/fetch_fmd.sh` downloads them (about 9 GB) with pinned checksums.

- 12 categories over 3 microscope types: confocal (5), two-photon (4), widefield (3). Samples are BPAE cells
  (3 stains), zebrafish (FISH) and mouse brain (MICE).
- 20 fields of view (FOVs) per category, each captured 50 times. Ground truth is the average of all 50 captures.
  Each FOV also comes at 4 lower noise levels (averages of 2, 4, 8 and 16 captures). Everything is 8-bit, 512×512.
- **Test set:** `test_mix`, 48 images, 4 per category (20 confocal, 16 two-photon, 12 widefield). `dora-bio fmd-check`
  verifies that every test image is FOV 19 of its category, pixel for pixel. **FOV 19 is never used for training**
  (`FmdOptions.ExcludedFovs`, enforced by gate 12), so every number below is on held-out fields of view.

## Protocol

- **Metric:** PSNR and SSIM against the 50-frame ground truth, on the 8-bit scale (data range 255), averaged over
  images. Each noisy image is normalized exactly as at inference (its own 0.1/99.9% percentiles), denoised, mapped
  back to 8-bit units and clipped to 0–255. The classical bilateral filter (noise-adaptive) and the raw noisy input
  are scored the same way. Code: `FmdBenchmark`; command: `dora-bio fmd-eval`.
- **Training (every run):** seed 2026, at most 30 epochs × 50 Adam steps × batch 16 × 64 px patches, so every run
  gets the same budget of optimizer steps; plateau LR decay, early stopping, and best-epoch weights restored.
  Validation uses a fifth of the *training* FOVs, never FOV 19. Same 26,497-parameter network as the bundled model.
  CPU only, about 4–5 minutes per run. Raw-frame runs use 4 raw frames of each of the 228 training FOVs
  (12 categories × 19 FOVs, 912 images).
- **Reproduce:** `tools/fetch_fmd.sh && tools/experiments/p2_fmd.sh`. Training is deterministic, so the same seed
  gives the same weights and the same tables.

## Results at FMD's hardest noise level (single raw captures)

Mean PSNR (dB) / SSIM against ground truth on the 48 held-out test_mix images. Higher is better; best per column in bold.

| Model | Trained on | Confocal (20) | Two-photon (16) | Widefield (12) | All 48 |
|---|---|---|---|---|---|
| Noisy input | | 29.13 / 0.701 | 26.32 / 0.473 | 25.23 / 0.374 | 27.22 / 0.543 |
| Bilateral filter (classical) | | 31.74 / 0.872 | 31.07 / 0.816 | 26.53 / 0.466 | 30.22 / 0.752 |
| DORA, bundled model (Noise2Void) | 10 **synthetic** frames | 32.70 / 0.846 | 31.98 / 0.815 | 31.45 / 0.775 | 32.15 / 0.818 |
| DORA, Noise2Void | FMD raw frames, no ground truth | 33.80 / 0.881 | 31.80 / 0.797 | 27.78 / 0.553 | 31.63 / 0.771 |
| DORA, supervised | FMD raw frames → ground truth | **34.26 / 0.897** | **33.37 / 0.874** | **32.28 / 0.830** | **33.47 / 0.873** |

What this shows:

- **With ground truth to learn from, DORA beats the classical filter on every microscope type:** +3.25 dB overall
  (+6.25 dB over the noisy input), and +5.75 dB on widefield, where the bilateral filter barely helps.
- **Without ground truth (Noise2Void), real data is a mixed result.** It beats the synthetic model on confocal
  (+1.10 dB) but loses 3.67 dB on widefield, so overall it lands *below* the model that never saw a real image
  (31.63 vs 32.15 dB). The next section measures why.
- **The bundled synthetic model generalizes better than expected:** +4.9 dB over noisy and +1.9 dB over bilateral
  on real data it never saw. On confocal, though, its SSIM (0.846) is below the bilateral filter's (0.872).
- **These rankings are for the noisiest input.** On FMD's cleaner levels they change, and on 16-frame averages
  every DORA model does worse than leaving the image alone ([graded noise](#2-graded-noise-progressive-noise)).

## Why Noise2Void fails on widefield: the noise is correlated

Noise2Void hides a pixel from the network and asks it to predict that pixel from its neighbours. That removes the
noise only if the neighbours carry no information about the hidden pixel's noise. If neighbouring pixels' noise is
correlated, the network learns to predict the noise as well, and keeps it. `dora-bio fmd-noise` measures this
directly on the test images: noise = noisy − ground truth, and the table gives its correlation with the pixel one to
the right, one below, and two to the right (0 = independent, as Noise2Void assumes; 0.5 = two-pixel smoothing):

<!-- fmd-noise -->
```text
test_mix level raw: correlation of the noise (noisy - ground truth) with a neighbour, mean over images
modality     n  right 1  down 1 right 2  noise SD
Confocal    20    0.093   0.039   0.034      9.73
TwoPhoton   16    0.091   0.041   0.038     12.74
WideField   12    0.615   0.603   0.198     14.04
all         48    0.223   0.180   0.077     11.81
```
<!-- /fmd-noise -->

Confocal and two-photon noise is close to independent. **Widefield noise is 0.6 correlated with both neighbours and
still 0.2 two pixels away**, the signature of an image that was smoothed or resampled somewhere between the camera
and the released file (FMD doesn't document a cause, and none is claimed here). That is exactly the failure P3's
`n2v-diagnostic` brick was planned to detect, and P2 shows that it happens on real data. Supervised training doesn't
depend on the assumption, which is why it wins most on widefield (32.28 vs 27.78 dB). The synthetic model never saw
this camera's correlated noise, so it can't have learned to reproduce it, which is the likely reason it does well
on widefield too.

## v1's ideas, measured

v1 (the GDScript digit recognizer at the repo root) trained with two data tricks, geometric augmentation and
"progressive noise". Each was tested here against v2's default, with the same network, seed and step budget.

### 1. Geometric augmentation: ±15° rotation and 0.85–1.15 scaling

v1 rotated and rescaled every training digit with nearest-neighbour sampling. v2's default is the 8 flips and
quarter turns (dihedral), which move pixels without resampling them. `--augment v1` (`V1Augment`) applies v1's
exact ranges and sampling, identically to the noisy patch and its target.

| Training | Augmentation | Confocal | Two-photon | Widefield | All 48 |
|---|---|---|---|---|---|
| Noise2Void | dihedral (v2 default) | 33.80 / 0.881 | 31.80 / 0.797 | 27.78 / 0.553 | 31.63 / 0.771 |
| Noise2Void | v1 rotate + scale | 33.52 / 0.873 | 31.53 / 0.771 | 28.04 / 0.565 | 31.49 / 0.762 |
| Supervised | dihedral (v2 default) | 34.26 / 0.897 | 33.37 / 0.874 | 32.28 / 0.830 | 33.47 / 0.873 |
| Supervised | v1 rotate + scale | 34.34 / 0.898 | 33.34 / 0.876 | 33.10 / 0.850 | 33.69 / 0.878 |

- **With Noise2Void it makes no clear difference** (−0.14 dB overall, too small for one seed to resolve). It had
  been expected to hurt, because nearest-neighbour upscaling duplicates pixels and so correlates their noise. It
  didn't, measurably.
- **With supervised training it helps a little overall (+0.22 dB) and clearly on widefield (+0.82 dB).** One
  possible reason: resampled training inputs have correlated noise, like widefield's. That is a hypothesis, not a
  measurement.
- Both v1-augmented runs were still improving when the 30-epoch cap stopped them (best epochs 27 and 28), while the
  dihedral runs had stopped early on a plateau. So the augmentation partly acts as regularization, and a longer
  budget could widen the gap.

**Verdict:** worth having as an option for supervised training. It shouldn't become the default without a
multi-seed check.

### 2. Graded noise ("progressive noise")

v1 added synthetic noise at five levels to copies of its training digits. FMD has the real version: every FOV at
5 noise levels (single captures, then averages of 2, 4, 8 and 16). The graded runs train on all 5 levels
(2 frames of each, 2,280 images) instead of 4 raw frames (912 images), with the same step budget. If graded noise
works as intended, the model should hold up on raw input and do better on the cleaner inputs.

Mean PSNR (dB) / SSIM over all 48 test images at each test noise level, from single captures (raw, hardest) to
16-frame averages (avg16, cleanest). Each model was trained once; only the test input changes.

| Model | raw (1 capture) | avg2 | avg4 | avg8 | avg16 |
|---|---|---|---|---|---|
| Noisy input | 27.22 / 0.543 | 30.08 / 0.680 | 32.86 / 0.798 | 36.03 / 0.889 | 39.70 / 0.949 |
| Bilateral filter | 30.22 / 0.752 | 32.45 / 0.830 | 34.88 / 0.894 | 37.39 / 0.935 | 40.57 / 0.964 |
| DORA, bundled synthetic model | 32.15 / 0.818 | 33.21 / 0.864 | 34.18 / 0.885 | 34.86 / 0.891 | 35.50 / 0.896 |
| DORA Noise2Void, raw frames only | 31.63 / 0.771 | 33.25 / 0.840 | 34.94 / 0.892 | 36.48 / 0.923 | 38.09 / 0.941 |
| DORA Noise2Void, all 5 levels | 31.04 / 0.737 | 32.85 / 0.819 | 34.73 / 0.880 | 36.39 / 0.918 | 38.18 / 0.942 |
| DORA supervised, raw frames only | 33.47 / 0.873 | 34.07 / 0.896 | 35.10 / 0.908 | 35.36 / 0.908 | 35.88 / 0.911 |
| DORA supervised, all 5 levels | 33.34 / 0.855 | 34.56 / 0.896 | 36.28 / 0.926 | 37.48 / 0.938 | 38.86 / 0.948 |

- **For supervised training, graded noise is a clear win.** Trained on raw frames only, the supervised model is the
  best at raw but falls apart on cleaner inputs: at avg16 it scores 35.88 dB, 3.8 dB *below the unprocessed input*,
  because it keeps smoothing as hard as single captures needed. Trained on all five levels, it gives up 0.13 dB at
  raw and gains 0.5 to 3.0 dB at every other level, which makes it the best DORA model at every level except raw.
  The cost isn't spread evenly: at raw it gains 0.58 dB on confocal and loses 1.55 dB on widefield.
- **For Noise2Void it doesn't help:** 0.2 to 0.6 dB worse from raw to avg4, and level with raw-only training at
  avg8 and avg16. Noise2Void trained on raw frames already copes better with clean inputs than raw-only supervised
  training does (38.09 vs 35.88 dB at avg16).
- **No DORA model knows when to stop.** At avg16, averaged over all 48 images, every DORA model scores below the
  noisy input it was given, while the bilateral filter still improves on it slightly. The harm comes from confocal
  and two-photon images. On widefield, the all-levels supervised model still helps at avg16 (40.61 vs 38.74 dB).
  The bundled synthetic model does the most damage on clean inputs (35.50 vs 39.70 dB at avg16). DORA is built for
  noisy images; on images that are already clean it can make things worse.

**Verdict:** keep graded noise for supervised training (`--fmd-levels raw,avg2,avg4,avg8,avg16`). There's no reason to
use it with Noise2Void.

### 3. The optimizer

Not re-measured, because v1's optimizer problems were defects rather than design choices that need a benchmark:
Adam was restarted on every feedback click (which makes the first step a fixed-size sign step), each weight's
update was clamped to ±1, and the learning rate *rose* after successes. Every run above uses v2's replacement:
persistent Adam state, one clip on the whole gradient's norm, and a learning rate that only decays on plateaus.
v1's rising-rate schedule is the one choice that could be measured the same way; that hasn't been done. What did
carry over is v1's milestone loop (progressive targets). It ran in every training here and never declared mastery
on real data (see below).

## What these results do and don't show

- **One seed per configuration.** Differences of a few tenths of a dB may be run-to-run variation rather than real
  effects. No multi-seed spread was measured.
- **Short CPU training of a small network.** These are not attempts at the FMD state of the art. Published FMD
  baselines use larger networks trained much longer on GPUs. The comparisons here are *within* DORA: same network,
  same budget, one thing changed at a time.
- **8-bit data.** FMD was quantized before release; real 16-bit camera data may behave differently.
- **A fixed step budget, and some runs were still improving.** Two runs stopped early on a validation plateau;
  the four with v1 augmentation or graded noise ran to the 30-epoch cap with their best epoch at 27 or 28. Those
  four might gain from longer training, so their comparisons here are at equal cost, not at convergence.
- **Progressive targets on real data:** no run reached "mastery" (holding the final 8 dB target). The supervised
  runs got as far as the 6 or 8 dB target; Noise2Void stalled at 4 or 6 dB. (The targets use validation patches on
  the normalized scale, so their dB values aren't comparable with the test tables.)
- **Weights trained on FMD are not published** (CC BY-SA terms for derived models are a separate decision). The
  bundled model stays the synthetic one.

## Reproducibility check (gate 13)

The blocks above marked `fmd-noise` and below marked `fmd-eval-synthetic` are verbatim program output.
`bricks.sh` gate 13 re-runs both whenever the data is present and requires an exact match. The block below is
`dora-bio fmd-eval --weights godot/models/dora_bio_weights.bin --root data/fmd --levels raw` for the bundled
synthetic model:

<!-- fmd-eval-synthetic -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   32.70 / 0.846
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   31.98 / 0.815
WideField   12   25.23 / 0.374   26.53 / 0.466   31.45 / 0.775
all         48   27.22 / 0.543   30.22 / 0.752   32.15 / 0.818
```
<!-- /fmd-eval-synthetic -->

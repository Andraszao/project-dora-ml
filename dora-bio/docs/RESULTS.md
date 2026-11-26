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
- **Reproduce:** `tools/fetch_fmd.sh && tools/experiments/p2_fmd.sh` at commit `d2be16e`. Training is
  deterministic, so the same seed gives the same weights and the same tables at that commit.
- **Known flaw in these runs, found in P3:** the 16 validation patches all came from 4 FOVs of one category
  (Confocal_BPAE_B), because they cycled through the held-out images in list order. So every training decision
  (best epoch, learning-rate decay, early stopping, progressive targets) was made on confocal data alone. The test
  numbers are unaffected in kind (test_mix was never involved), but some runs might have stopped at a better epoch
  for widefield or two-photon. Since P3 the validation patches cover every held-out FOV; P3 retrains its baselines.

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

## P3: correlated noise and structured blind spots

Phase P3 of [`PATHWAY.md`](../PATHWAY.md), measured 2026-11-25 with P2's budget and seed (30 epochs × 50 steps ×
batch 16 × 64 px, seed 2026, 4 raw frames per FOV, FOV 19 never trained on) and P3's validation fix.
Reproduce: `tools/experiments/p3_struct.sh`.

### Spotting correlated noise without ground truth (brick 14)

`dora-bio diagnose --in IMAGE` estimates the noise correlation from the noisy image alone: 16 px tiles, each tile's
best-fit plane removed, the flattest quarter kept, the median correlation per offset. Noise counts as correlated at
0.4 or more one pixel away. On the 48 raw test images it agrees with the ground-truth measurement every time:
widefield estimates 0.557–0.578 (truth 0.58–0.65), confocal 0.262 at most and two-photon 0.155 at most. Gate 14
re-checks this whenever the data is present. On the 912 training images it flags exactly the 228 widefield ones.
The Godot app runs the check on every loaded image.

Limits: small tiles keep cell structure out but bias the estimate low, about half the truth at two pixels. On
cleaner inputs the check false-alarms on densely stained images: on 16-frame averages it reads 0.57 on
Confocal_BPAE_G, where even the flattest tiles contain structure. (There the ground truth isn't reliable either: the
leftover noise is so small that the difference from the 50-frame average likely includes drift between captures.)

### The validation set was one category (fixed)

P2's 16 validation patches all came from 4 FOVs of Confocal_BPAE_B, so every training decision (best epoch,
learning-rate decay, early stopping) was made on confocal data alone. They now cover all 46 held-out FOVs (48 patches).
Retrained with the fix, the same supervised recipe scores **33.75 dB** on test_mix (was 33.47), with widefield
**33.02** (was 32.28). Plain Noise2Void barely moves: 31.68 (was 31.63).

### Structured blind spots (brick 15)

A structured blind spot hides the pixel's correlated neighbours too, and fills every hidden pixel from a pixel that
no footprint hides. It only goes on the training images the noise check flags. **The bar was set before anything was
built and never changed** (PATHWAY.md): vs one plain Noise2Void network on test_mix (raw), widefield +1 dB or more,
confocal and two-photon −0.2 dB at worst.

The footprint was chosen on validation PSNR (held-out *training* FOVs), never on test_mix: square-1 26.78 dB,
**diamond-2 26.87 dB**, square-2 26.76 dB, against 25.39 dB for plain blind spots. Diamond-2 hides the 12 nearest
neighbours, which covers every offset where widefield noise is correlated by 0.17 or more.

| Noise2Void on raw test_mix | Networks | Confocal (20) | Two-photon (16) | Widefield (12) | All 48 |
|---|---|---|---|---|---|
| Plain blind spots | 1 | 33.77 / 0.881 | 31.85 / 0.800 | 27.97 / 0.563 | 31.68 / 0.774 |
| Design 1: structured blind spots on the flagged images | 1 | 32.62 / 0.855 | 31.75 / 0.812 | 31.38 / 0.792 | 32.02 / 0.825 |
| **Design 2: two networks, routed by the noise check** | 2 | **34.09 / 0.886** | **32.46 / 0.819** | **31.76 / 0.807** | **32.97 / 0.844** |
| Supervised, for reference (needs ground truth) | 1 | 34.38 / 0.898 | 33.52 / 0.878 | 33.02 / 0.852 | 33.75 / 0.880 |

- **Design 1 missed the bar.** Widefield gained 3.41 dB, but confocal lost 1.15 dB although no confocal image was
  masked differently. One network has to learn when to trust a pixel's nearest neighbours, and it can't fully tell
  the two kinds of image apart.
- **Design 2 meets it:** widefield +3.79 dB, confocal +0.32 dB, two-photon +0.61 dB. One network trains with plain
  blind spots on the images the noise check calls independent, the other with structured ones (diamond-2) on the
  images it calls correlated. Each test image goes to the network its own noise check picks (12 of 48 routed to the
  correlated one, exactly the widefield images). Confocal and two-photon gain too: their network never has to
  accommodate widefield.
- **Disclosed:** design 2 was devised after seeing design 1's test results, so test_mix was looked at twice under the
  same bar. It also costs twice the training (two networks, each with the full step budget).
- **Without any ground truth, Noise2Void is now within 0.8 dB of supervised training** (32.97 vs 33.75 dB), and the
  widefield gap shrinks from 5.05 to 1.26 dB. It beats the bilateral filter by 2.75 dB and the bundled synthetic
  model by 0.82 dB.
- Every footprint candidate was also scored on test_mix, for completeness only (not used to choose): with design 1,
  square-1 gives 31.89 dB overall (widefield 30.69), square-2 32.16 (31.42).

### At cleaner inputs, structured blind spots hurt

Mean PSNR (dB) / SSIM over all 48 test images at each test noise level:

| Model | raw (1 capture) | avg2 | avg4 | avg8 | avg16 |
|---|---|---|---|---|---|
| Noisy input | 27.22 / 0.543 | 30.08 / 0.680 | 32.86 / 0.798 | 36.03 / 0.889 | 39.70 / 0.949 |
| Bilateral filter | 30.22 / 0.752 | 32.45 / 0.830 | 34.88 / 0.894 | 37.39 / 0.935 | 40.57 / 0.964 |
| Noise2Void, plain (1 network) | 31.68 / 0.774 | 33.21 / 0.841 | 34.89 / 0.891 | 36.33 / 0.920 | 37.89 / 0.938 |
| Noise2Void, routed (2 networks) | 32.97 / 0.844 | 33.85 / 0.879 | 34.67 / 0.885 | 35.13 / 0.894 | 35.49 / 0.884 |
| Supervised, raw frames only (reference) | 33.75 / 0.880 | 34.19 / 0.895 | 35.04 / 0.903 | 35.32 / 0.903 | 35.83 / 0.904 |

The routed pair wins at raw and avg2 and loses from avg4 up, for two reasons. A structured network can't use the
nearest pixels, so it over-smooths once the noise is small: on widefield at avg16 it scores 36.13 dB, where plain
Noise2Void scores 40.10. And the noise check's false alarms grow as the noise shrinks, so more dense confocal and
two-photon images get routed to that network (16 of 48 at avg4 and avg8, 20 at avg16).

**Verdict:** structured blind spots are worth it for noisy images with correlated noise, the case Noise2Void is
for. They are opt-in rather than the default, because they lose on cleaner inputs. The CLI recommends
`--struct auto` when the noise check flags the training images, and the two-network recipe when the data is mixed.
"Adapt to sample" uses them automatically on images the check flags (one image is one microscope, so the mixing
problem doesn't arise).

### Variance stabilization and calibration (bricks 16–17)

The generalized Anscombe transform turns Poisson–Gaussian noise, whose variance grows with brightness, into noise of
unit variance, and Mäkitalo and Foi's unbiased inverse maps the denoised result back. It needs each image's noise
model, variance = gain · intensity + intercept, which `NoiseCalibration` estimates from the noisy image alone.
On synthetic frames from six camera settings it finds the gain within 5.1% and the variance curve within 6.8%.

**On real FMD images the calibration is only reliable for widefield.** Checked against ground truth on *training*
FOVs 1–3 of every category (never test_mix): the variance predicted from the noisy image alone, divided by the
variance measured against ground truth in 5 brightness bands, and the noise variance left after the transform,
which should be 1:

| | Predicted / measured variance (median, middle 80%) | Variance after the transform (mean, range) |
|---|---|---|
| Confocal | 1.09 (−0.28 to 1.52) | 0.41 (0.17–0.77) |
| Two-photon | 1.12 (0.89–1.40) | 0.71 (0.39–0.92) |
| Widefield | 0.91 (0.72–1.17) | 1.16 (0.91–1.61) |

Dim confocal images are the problem. Their measured variance curve bends at low intensity (3.5 at mean 4 on
Confocal_BPAE_B, where a line through the brighter bands is already below zero), and their bright regions are nearly
all cell structure, which steepens the fitted line. The transform then clamps much of the dark background to zero.
Trusting a lower quantile of each band's tiles (0.10 or 0.05 instead of 0.25) helps only slightly (confocal 0.52), so
the default stays. A straight Poisson–Gaussian line doesn't describe FMD's 8-bit confocal noise at low intensities.

**With the transform, Noise2Void loses.** The bar for a win was set before running it (PATHWAY.md, P3 exit): the
all-48 PSNR must rise at raw and drop by no more than 0.2 dB at any lower level, compared with the same design
without the transform. It fails for both designs:

| Raw test_mix | Confocal (20) | Two-photon (16) | Widefield (12) | All 48 |
|---|---|---|---|---|
| Plain Noise2Void | 33.77 / 0.881 | 31.85 / 0.800 | 27.97 / 0.563 | 31.68 / 0.774 |
| Plain, with the transform | 33.35 / 0.886 | 31.28 / 0.783 | 28.30 / 0.631 | 31.40 / 0.788 |
| Routed pair | 34.09 / 0.886 | 32.46 / 0.819 | 31.76 / 0.807 | 32.97 / 0.844 |
| Routed pair, with the transform | 33.60 / 0.885 | 31.99 / 0.809 | 32.23 / 0.806 | 32.72 / 0.840 |

| All 48, PSNR (dB) | raw | avg2 | avg4 | avg8 | avg16 |
|---|---|---|---|---|---|
| Plain Noise2Void | 31.68 | 33.21 | 34.89 | 36.33 | 37.89 |
| Plain, with the transform | 31.40 | 33.01 | 34.18 | 35.72 | 36.91 |
| Routed pair | 32.97 | 33.85 | 34.67 | 35.13 | 35.49 |
| Routed pair, with the transform | 32.72 | 33.68 | 34.00 | 34.81 | 35.22 |

The split follows the calibration exactly: widefield, the one modality it calibrates well, gains 0.33 dB (plain) and
0.47 dB (routed), while confocal and two-photon lose about 0.5 dB. **Verdict:** the transform stays opt-in (`--vst`,
Noise2Void only, and a model trained with it must be applied with it). What limits it here is calibrating dim 8-bit
images from one frame, not the transform. A calibrated camera (dark and flat frames) or a curve-based transform would
be the way to revisit it.

## P4: a U-Net against the dilated network

Phase P4 of [`PATHWAY.md`](../PATHWAY.md), measured 2026-11-25. The U-Net has 16 → 32 → 64 features over two pooling
levels (11 convolutions, 118,113 parameters, about the dilated network's compute per pixel); the dilated network has
26,497. Both get exactly P3's budget, seed, data and validation (30 epochs × 50 steps × batch 16 × 64 px, seed 2026,
4 raw frames per FOV); the dilated runs are P3's `sup_plain` and `n2v_plain`. Reproduce: `tools/experiments/p4_arch.sh`.

**The rule was set before measuring** (PATHWAY.md): the U-Net becomes the default only if it beats the dilated network
on all-48 test_mix PSNR (raw) for supervised *and* Noise2Void training, and is at most 3× slower.

| Raw test_mix | Confocal (20) | Two-photon (16) | Widefield (12) | All 48 |
|---|---|---|---|---|
| Supervised, dilated | 34.38 / 0.898 | 33.52 / 0.878 | 33.02 / 0.852 | 33.75 / 0.880 |
| **Supervised, U-Net** | **35.55 / 0.927** | **33.87 / 0.893** | **34.51 / 0.893** | **34.73 / 0.907** |
| Noise2Void, dilated | 33.77 / 0.881 | 31.85 / 0.800 | 27.97 / 0.563 | 31.68 / 0.774 |
| Noise2Void, U-Net | 32.99 / 0.859 | 31.40 / 0.767 | 27.98 / 0.571 | 31.21 / 0.756 |

| All 48, PSNR (dB) | raw | avg2 | avg4 | avg8 | avg16 |
|---|---|---|---|---|---|
| Supervised, dilated | 33.75 | 34.19 | 35.04 | 35.32 | 35.83 |
| Supervised, U-Net | 34.73 | 35.04 | 35.44 | 35.33 | 35.28 |
| Noise2Void, dilated | 31.68 | 33.21 | 34.89 | 36.33 | 37.89 |
| Noise2Void, U-Net | 31.21 | 32.50 | 34.29 | 35.22 | 36.27 |

Speed: a 512×512 frame takes 0.79 s with the U-Net and 0.58 s with the dilated network (median of 5 tiled runs on 4
cores), **1.36× slower** <!-- p4-slowdown: 1.36 -->. Training took 5:42 for 30 epochs against 4:40 (1.22×).

- **With ground truth, the U-Net is clearly better:** +0.98 dB overall and better on every microscope type (widefield
  +1.49, confocal +1.17, two-photon +0.35), with SSIM 0.907 against 0.880. It is the best DORA model so far (34.73 dB,
  4.5 dB above the bilateral filter), and it was still improving when the 30-epoch cap stopped it (best epoch 28).
  It also helps on cleaner inputs up to avg8, though like every model it falls below the untouched image there.
- **Without ground truth (Noise2Void), it is worse:** −0.47 dB overall. Its run stopped early at epoch 10 with its
  best epoch at 4, while the dilated run kept improving until epoch 23. That points at training (the U-Net may need a
  smaller learning rate or more care with blind spots) rather than capacity, but this wasn't tested. On the
  validation set its best score was in fact slightly higher (25.56 vs 25.39 dB), so 48 validation patches can't
  reliably tell these two apart.
- **Verdict, by the rule set in advance: the dilated network stays the default**, because the U-Net doesn't win for
  Noise2Void. `--arch unet` is available and is the better choice when ground truth exists. Making the default depend
  on the training mode would be a new rule, chosen after seeing these numbers, so it isn't adopted here.

## Quality pass Q1: the best supervised recipe

The first quality pass (brick 25), pre-registered in [`PATHWAY.md`](../PATHWAY.md) before any run and measured
2026-11-25. Reproduce: `tools/experiments/q1_supervised.sh`.

**Candidate:** the U-Net (16 → 64 features, depth 2) trained supervised on all five FMD noise levels (2 frames of
each, 2,280 images) for up to 60 epochs. **Ablation:** the same at 30 epochs. **Baseline:** P4's U-Net trained on raw
frames only for 30 epochs, the best supervised model so far. **Bar:** at raw, all-48 PSNR no more than 0.1 dB lower; the
mean over avg2–avg16 at least 0.5 dB higher; no lower level more than 0.1 dB worse.

| All 48, PSNR (dB) | raw | avg2 | avg4 | avg8 | avg16 |
|---|---|---|---|---|---|
| Noisy input | 27.22 | 30.08 | 32.86 | 36.03 | 39.70 |
| Bilateral filter | 30.22 | 32.45 | 34.88 | 37.39 | 40.57 |
| Baseline: U-Net, raw frames, 30 epochs | **34.73** | 35.04 | 35.44 | 35.33 | 35.28 |
| Candidate: U-Net, all levels, 60 epochs (best epoch 46) | 34.26 | **35.16** | **37.02** | **38.32** | **40.15** |
| Ablation: U-Net, all levels, 30 epochs | 34.03 | 35.05 | 36.84 | 38.28 | 40.15 |

- **The bar is missed at raw:** the candidate is 0.47 dB below the baseline there, against a tolerance of 0.1. The other
  two conditions hold easily (mean +2.39 dB over avg2–avg16, every level better). **By the rule, nothing ships**:
  supervised FMD training keeps defaulting to raw frames. <!-- q1-verdict: not-adopted -->
- **But it's the first DORA model that doesn't damage clean images.** Averaged over all 48, every model before it
  scored below the untouched 16-frame averages (35.28 dB for the baseline against 39.70). This one scores 40.15, above
  the untouched image though still below the bilateral filter (40.57). It also beats the bilateral filter from avg2 to
  avg8, by up to 2.7 dB.
- **What the extra epochs bought:** +0.23 dB at raw and almost nothing elsewhere (the ablation). The all-level training
  is what costs the 0.47 dB at raw: the network shares its capacity across five noise levels.
- **What it suggests (backlog, not adopted here):** choose the model per image from its estimated noise level, the
  raw-trained model for noisy images and the all-level model for clean ones, the way P3 routed by noise correlation.
  If the noise estimate is good enough, that could keep 34.73 at raw and gain the 2–5 dB on clean inputs.

## Quality pass Q10: routing each image by its noise level

The second quality pass (brick 27), pre-registered in `PATHWAY.md` (commit `9361f10`) before any of its code existed.
Reproduce: `tools/experiments/q10_routing.sh`. **Idea:** keep Q1's gains on clean inputs without its cost on noisy ones,
by sending noisy images to A (P4's U-Net trained on raw frames) and clean ones to B (Q1's all-level U-Net). The router is
Immerkær's noise σ of the normalized image. **Its threshold was chosen on the 46 validation FOVs both models held out**
(all five levels, 230 images), never on test_mix. **Bar:** at raw ≥ A − 0.1 dB (34.63); over avg2–avg16 ≥ B − 0.2 dB (37.46).

| Validation, PSNR (dB) | raw | avg2 | avg4 | avg8 | avg16 | all 230 |
|---|---|---|---|---|---|---|
| A: raw-trained U-Net | **35.31** | 35.93 | 36.05 | 36.14 | 36.27 | 35.94 |
| B: all-level U-Net | 34.97 | **36.30** | **37.63** | **39.02** | **40.86** | **37.76** |

- **The best threshold on validation sends every image to B.** A is better only on raw frames there, by 0.34 dB, and
  σ can't pick those images out, because how noisy an image looks depends on the sample as much as on the averaging.
  Even a router that knew each image's true noise level would gain only 0.06 dB over "always B" on validation.
- **On test_mix, routing scores 34.28 dB at raw** (8 of the 48 raw images went to A), against the 34.63 needed; the
  lower levels match B (37.68 on average). **The bar is missed, so nothing ships:** `dora-bio denoise` doesn't get the
  routing option; the evaluation tooling (`fmd-route`, `fmd-eval --weights-clean`) stays. <!-- q10-verdict: not-adopted -->
- **What it teaches:** when the noise level of the data isn't known in advance, the all-level model alone is the better
  single supervised model: it costs 0.3–0.5 dB on the noisiest frames and gains 1–5 dB on everything cleaner. Whether
  that trade is worth taking is a choice about what DORA's users image, not something these bars can settle.

## Quality pass Q13: telling the network how noisy each image is

The third quality pass (brick 29), pre-registered in [`PATHWAY.md`](../PATHWAY.md#q13-pre-registration) (commit
`c14066f`) before any of its code existed. Reproduce: `tools/experiments/q13_conditioning.sh`. **Idea:** Q1's
all-level U-Net loses at raw because one network must serve five noise levels. Give it each frame's noise level and
let it adapt: FiLM modulation (a per-channel scale and shift) of every hidden convolution, driven by
s = ln(σ̂ / 0.05) of the whole frame through a 16-unit generator that starts at zero, so an untrained conditioned
U-Net is the plain one bit for bit. **Recipe:** exactly Q1's (all five levels, 2 frames each, up to 60 epochs), with
seeds 2026, 2027 and 2028 for both the conditioned and the plain recipe. **Bar** (seed 2026, all 48): raw ≥ 34.63,
avg2–avg16 mean ≥ 37.56, no level more than 0.2 below Q1's all-level model, and a three-seed mean raw gain over the
plain recipe of at least 0.2 dB.

| test_mix, all 48, PSNR (dB) | raw | avg2 | avg4 | avg8 | avg16 | avg2–avg16 mean |
|---|---|---|---|---|---|---|
| Plain all-level U-Net, seed 2026 (Q1's candidate) | 34.26 | 35.16 | 37.02 | 38.32 | 40.15 | 37.66 |
| Plain, seed 2027 | 34.55 | 35.44 | 37.14 | 38.51 | 40.51 | 37.90 |
| Plain, seed 2028 | 34.48 | 35.34 | 37.14 | 38.38 | 40.37 | 37.81 |
| **Conditioned, seed 2026** (decides the bar; stopped early at epoch 33, best 27) | 33.92 | 35.03 | 36.82 | 38.27 | 40.24 | 37.59 |
| Conditioned, seed 2027 | 34.44 | 35.29 | 37.04 | 38.34 | 40.25 | 37.73 |
| Conditioned, seed 2028 | 34.36 | 35.23 | 37.03 | 38.31 | 40.36 | 37.73 |
| Three-seed means: plain / conditioned | 34.43 / 34.24 | | | | | 37.79 / 37.68 |

- **The bar is missed, by a wide margin.** Seed 2026 scores 33.92 at raw against the 34.63 needed, 0.34 below the plain
  model with the same seed. Over three seeds, conditioning *loses* 0.19 dB at raw (the bar wanted +0.2), and 0.11 dB
  on the cleaner levels. It is behind the plain recipe for every seed, on test_mix and on each seed's own validation
  FOVs (by 0.12 to 0.51 dB). **By the rule, nothing ships:** `--condition noise` stays opt-in and supervised training's
  default stays `none`. <!-- q13-verdict: not-adopted -->
- **The network does use its condition.** On the seed-2026 model's validation FOVs (all five levels, 46 images each),
  telling it twice or half the measured noise costs up to 2.0 dB, mostly on the noisiest frames:

  | Validation, condition from | raw | avg2 | avg4 | avg8 | avg16 |
  |---|---|---|---|---|---|
  | 1× the measured σ (honest) | **34.40** | **35.89** | **37.32** | **38.81** | **40.77** |
  | 2× | 33.17 | 34.94 | 36.68 | 38.36 | 40.42 |
  | 0.5× | 32.45 | 34.44 | 36.39 | 38.34 | 40.74 |

  So the failure is not that the network ignores s. It learned to lean on it, but s most likely carries nothing the plain
  U-Net can't already read from the pixels within its 55 px view, and it brings the estimator's errors with it: as Q10 found, an
  image's σ̂ depends on its structure as much as on its noise, and a factor-2 error costs a dB.
- **What it teaches.** (1) **Seed spread is about 0.3 dB.** The plain all-level recipe ranges 34.26–34.55 at raw over
  three seeds, so single-run differences of a few tenths of a dB (several earlier verdicts rest on one seed) are within
  run-to-run noise. (2) **Q1's raw cost was partly bad luck.** Its seed-2026 run was the worst of the three: the
  three-seed mean puts the all-level recipe 0.30 dB below the raw-trained U-Net's single run (34.73), not 0.47.
  (3) Early stopping on 48 validation patches can cut a run off at epoch 33 that would have improved to 60 (Q5).

## Quality pass Q18: DORA against CARE and Noise2Void on public data

The first comparison with the tools biologists already use (brick 31, field validation), pre-registered in
[`PATHWAY.md`](../PATHWAY.md#q18-pre-registration-dora-against-care-and-noise2void-on-public-data-ext-bench) (commit
`be5228b`) before any of its code existed. Reproduce: `tools/fetch_ext.sh`, then install
`tools/experiments/requirements-careamics.txt` into a virtualenv and run
`CAREAMICS_PYTHON=VENV/bin/python tools/experiments/q18_ext.sh`.

**Data:** three public paired sets (CC BY 4.0). DeepBacs *E. coli* MreB (16-bit, 30 training pairs, 5 test) and
nucleoid (8-bit, 28 and 2), from Spahn et al., *Communications Biology* 2022. The ZeroCostDL4Mic CARE 2D example (float32,
22 and 2), from von Chamier et al., *Nature Communications* 2021. **Scoring:** CARE's protocol (Weigert et al., *Nature
Methods* 2018): the high-SNR reference normalized by its own 0.1st and 99.9th percentiles, every output matched to it
by least-squares scale and offset, and PSNR and SSIM after that, the noisy input included. **Baselines:** CAREamics 0.3.5
(the library behind the napari Noise2Void and CARE plugins), default 2D configurations, 64 px patches, batch 16, the
same step budget as DORA's matching recipe, on the same 4-core CPU.

| Test PSNR (dB) / SSIM, mean over each test split | MreB (5) | Nucleoid (2) | CARE 2D (2) | Mean |
|---|---|---|---|---|
| Noisy input | 21.67 / 0.368 | 27.39 / 0.808 | 19.28 / 0.461 | 22.78 |
| Bilateral filter | 21.75 / 0.381 | 27.53 / 0.823 | 19.30 / 0.469 | 22.86 |
| DORA out of the box: bundled synthetic model | 23.59 / 0.513 | 32.34 / 0.952 | 19.23 / 0.436 | 25.05 |
| DORA out of the box: FMD-trained U-Net (P4) | 26.10 / 0.758 | 33.50 / 0.971 | 18.91 / 0.385 | 26.17 |
| **DORA Noise2Void** (dilated, 26k parameters) | 25.83 / 0.697 | 33.71 / 0.961 | 19.30 / 0.467 | **26.28** |
| **CAREamics Noise2Void** (U-Net, 334k parameters) | **27.01** / 0.814 | **34.70** / 0.979 | 19.29 / 0.466 | **27.00** |
| **DORA supervised** (U-Net, 118k parameters) | **27.37** / 0.827 | **35.45** / 0.982 | **20.13** / 0.539 | **27.65** |
| **CAREamics CARE** (U-Net, 334k parameters) | 27.35 / 0.818 | 35.30 / 0.981 | 20.08 / 0.520 | **27.58** |

- **The bar is missed, on Noise2Void.** Averaged over the three sets, DORA supervised is **+0.07 dB** against CARE (the
  bar allowed −0.5) and DORA Noise2Void is **−0.72 dB** against CAREamics Noise2Void (also −0.5 allowed). **By the rule:
  not competitive.** Q12 (capacity and training budget) moves ahead of the lab pilot, and the README states the gap.
  <!-- q18-verdict: not-competitive -->
- **With ground truth, DORA matches CARE.** DORA's supervised U-Net is ahead on all three sets (by 0.02, 0.15 and
  0.05 dB, and on SSIM by 0.001–0.019), on the same CPU and budget, with a network a third the size. That's within
  noise, so the fair reading is parity, not a win. CARE stopped early under its default early stopping, after 9–13
  epochs; DORA's supervised runs stopped at 28–39.
- **Without ground truth, DORA's default is about 1 dB behind where Noise2Void works.** On the two DeepBacs sets,
  whose noise DORA's check finds independent between pixels (correlation −0.02 and −0.01), CAREamics Noise2Void is ahead
  by 1.18 and 0.99 dB. The likeliest cause is network size: DORA's default Noise2Void network is the 26k-parameter
  dilated stack, 12.6× smaller than CAREamics' U-Net. Other defaults differ too: masking 2% of pixels against 0.2%,
  holding out 20% of the training images against 8 patches, and predicting with the best checkpoint against the last.
  Which of these matters is untested (on FMD, a U-Net trained by Noise2Void lost at a 30-epoch budget: P4, Q2).
- **Where Noise2Void can't work, DORA says so.** On CARE 2D neither Noise2Void does anything (19.30 and 19.29 against
  19.28 untouched), while supervised training gains 0.8 dB. DORA's noise check finds its noise correlated between
  neighbouring pixels (0.58), from the noisy image alone, and tells the user to prefer supervised training or structured
  masking. CAREamics gives no such warning. (Structured blind spots weren't part of this pass.)
- **Out of the box, without training on these sets,** the FMD-trained U-Net beats DORA's own Noise2Void on MreB (26.10)
  and comes within 0.2 dB on nucleoid. But it falls below the untouched image on CARE 2D. The synthetic bundled model
  gains 1.9 and 5.0 dB on the DeepBacs sets.
- **Speed on this CPU** (not part of the bar):
  - **Training, Noise2Void:** DORA 2:33–2:56, CAREamics 187–230 s.
  - **Training, supervised:** DORA 2:49–4:04, CARE 76–92 s (it stops much earlier).
  - **Inference:** DORA about 0.27 s per 512² frame (`bench`), CAREamics about 0.5 s per 512² image in these runs.
- **What it doesn't show.** Three small test splits (5, 2 and 2 images, every image listed in the blocks below) and
  one seed per method; the Q13 seed spread was about 0.3 dB. Both tools ran their CPU defaults with an equal step
  budget, not the longer GPU training their authors used, so either could score higher with more training.

## Quality pass Q12: closing the self-supervised gap

Brick 33, the first pass of the road to state of the art, pre-registered in
[`PATHWAY.md`](../PATHWAY.md#q12-pre-registration-closing-the-self-supervised-gap-brick-33) (commit `1a16b2f`) before any
of its code existed. Reproduce: `tools/experiments/q12_n2v.sh`. **Question:** can DORA's Noise2Void close Q18's gap
(26.28 against CAREamics' 27.00 dB) and brick 32's contrast compression (MreB slope 0.75) with a bigger network, other
masking or a longer budget? **Selection** happened on validation images: the fifth of each training split a seed-2026
run holds out, scored with CARE's protocol. The pre-registered rule picked from the scores below automatically
(`tools/experiments/q12_n2v.sh`), and the test splits were scored once, at the end.

<!-- q12-selection -->
```text
step 1: network (mean validation PSNR over 3 sets, MreB validation slope)
dilated 26.777 0.74
unet16 27.177 0.92
unet32 27.070 0.91
chosen network: unet16
step 2: masking
unet16 27.177 0.92
unet16_m02 26.453 0.87
chosen masking: 2%
step 3: budget (1 of 3 winning runs ended at the 30-epoch cap)
budget step skipped (fewer than two runs at the cap)
final recipe: unet16
final arguments: --arch unet --features 16 --depth 2 --epochs 30
```
<!-- /q12-selection -->

| Test (PSNR dB; MreB slope / ratio error / spots real) | MreB | Nucleoid | CARE 2D | Mean | MreB fidelity |
|---|---|---|---|---|---|
| Today's default: dilated, 26k parameters (Q18) | 25.83 | 33.71 | 19.30 | 26.28 | 0.75 / 8.9% / 34% |
| **Q12's recipe: U-Net 16 → 64, 118k parameters, 2% masking, 30 × 50 steps** | **26.65** | **34.14** | 19.30 | **26.697** | **0.93 / 6.6%** / 41% |
| CAREamics Noise2Void (Q18) | 27.01 | 34.70 | 19.29 | 27.00 | 0.86 / 7.4% / 56% |
| **FMD guard** (test_mix raw; needs ≥ 31.48) | | | | | U-Net **31.21** (the default: 31.68) |

- **Not adopted, and the FMD guard decides it.** On FMD's raw frames the chosen U-Net stops early (best epoch 4 of 10)
  and scores 31.21 dB, 0.47 below the default. That is the same run as P4's U-Net Noise2Void, the problem Q2 already
  recorded. The bar itself is missed by 0.003 dB (26.697 against 26.70), which is inside run-to-run noise but still a
  miss. The "improved anyway" branch would have applied (+0.42 dB, slope up), but it too requires the guard. **By the
  rule, nothing ships:** `train --mode n2v` keeps the dilated network. <!-- q12-verdict: not-adopted -->
- **What the U-Net fixes, on the public sets.**
  - **The gap:** it closes about 60% of it (+0.42 of the 0.72 dB).
  - **Fidelity:** the slope goes from 0.75 to 0.93 (CAREamics: 0.86), and the ratio error from 8.9% to 6.6%, now lower
    than the raw frame's 7.1% and CAREamics' 7.4%.
  - **Spots:** 41% of its spots are real, up from 34% but short of CAREamics' 56%.
  - So intensities measured on it are trustworthy where the default's weren't.
- **What didn't help.**
  - **The 0.47M U-Net**, CAREamics' size, was slightly worse on validation (27.07 against 27.18). It stopped early
    (best epoch 9) on two of the three sets.
  - **CAREamics' 0.2% masking** was 0.7 dB worse in DORA's setup (26.45 against 27.18).
  - **The budget step** was skipped: only one of the three winning runs reached the 30-epoch cap.
- **What blocks the U-Net is its early stop under Noise2Void.** It stopped at epoch 9 on nucleoid, and at epoch 4 on
  FMD, where it then loses to the small network. Noise2Void validates on masked pixels against noisy values, a loss
  whose floor is the noise itself. Small real gains hide under it, so early stopping fires long before the denoising
  stops improving (P4, Q2, Q5). That is the next pass's target: how a U-Net trained by Noise2Void should decide when to
  stop.

## Fidelity: what scientists measure (brick 32)

PSNR rewards being close on average. A biologist measures specific things: the brightness of a region relative to
another (after subtracting background), and whether a punctum is really there. `dora-bio fidelity` measures both against
the high-SNR reference (`Fidelity.cs`; reproduce: `tools/experiments/fidelity.sh`).

- **Slope** (1.00 is perfect): k in output ∝ reference^k over the foreground tiles (16 px, reference mean ≥ 0.1), with
  both images background-subtracted and scaled by their own percentiles. Below 1, the output compresses contrast: bright
  regions are dimmed relative to dim ones, so a measured ratio between two cells comes out too small.
- **Ratio error** (0% is perfect): the typical error of one tile's intensity once the overall gain is taken out.
- **Spots found / real** (100% is perfect): blobs of 2–4 px detected in both images by the same detector and threshold
  (set on the reference), paired within 2 px. "Found" is the share of the reference's spots the output keeps; "real" is
  the share of the output's spots that the reference has.

| Slope / ratio error / spots found / spots real | |
|---|---|
| **FMD test_mix, raw (48 images)** | |
| Noisy input | 0.83 / 7.5% / 65.4% / 14.3% |
| Bundled synthetic model | Noise2Void / on / 10 / SYNTHETIC |
| Supervised, dilated (P3) | Supervised / on / FMD / (12 |
| Supervised, U-Net (P4) | **Supervised / on / FMD / (12** |
| Supervised, all-level U-Net (Q1) | Supervised / on / FMD / (12 |
| Noise2Void, dilated (P3) | Noise2Void / on / FMD / (12 |
| Noise2Void, U-Net (P4) | Noise2Void / on / FMD / (12 |
| **DeepBacs MreB (5 images)** | |
| Noisy input | 1.00 / 7.1% / 55.2% / 7.1% |
| DORA Noise2Void | Noise2Void / on / deepbacs-mreb / training |
| CAREamics Noise2Void | CAREamics / 0.3.5 / Noise2Void, / default |
| DORA supervised | Supervised / on / deepbacs-mreb / training |
| CAREamics CARE | CAREamics / 0.3.5 / CARE, / default |
| **DeepBacs nucleoid (2 images)** | |
| Noisy input | 1.00 / 3.0% / 21.2% / 14.2% |
| DORA Noise2Void | Noise2Void / on / deepbacs-nucleoid / training |
| CAREamics Noise2Void | CAREamics / 0.3.5 / Noise2Void, / default |
| DORA supervised | Supervised / on / deepbacs-nucleoid / training |
| CAREamics CARE | CAREamics / 0.3.5 / CARE, / default |
| **CARE 2D (2 images)** | |
| Noisy input | 0.75 / 31.0% / 4.2% / 35.3% |
| DORA Noise2Void | Noise2Void / on / zerocost-care2d / training |
| CAREamics Noise2Void | CAREamics / 0.3.5 / Noise2Void, / default |
| DORA supervised | Supervised / on / zerocost-care2d / training |
| CAREamics CARE | CAREamics / 0.3.5 / CARE, / default |

- **DORA's supervised U-Net keeps measurements honest on FMD.** On FMD's raw frames: slope 1.02, the lowest ratio error
  (3.9%), and 72% of its spots are real (the noisy input: 14%). On MreB it compresses contrast slightly (0.94) where CARE
  slightly exaggerates it (1.03), with a lower ratio error than CARE (6.7% against 7.4%).
- **DORA's Noise2Void distorts intensities.** On MreB its slope is 0.75 and its ratio error (8.9%) is *worse than the
  untouched image's* (7.1%), so a ratio measured on its output is less accurate than one measured on the raw frame.
  CAREamics' Noise2Void distorts less (0.86, 7.4%), and twice as many of its spots are real (56% against 34%). On FMD
  both of DORA's Noise2Void networks compress contrast (0.86–0.88) and leave most noise peaks in place (23–27% of spots
  real). This is a second target for Q12, next to the 1 dB PSNR gap: a better self-supervised model has to keep the
  slope near 1, not just raise the PSNR.
- **What these numbers can't show.**
  - The high-SNR references are not noise-free, so even a perfect denoiser would show some ratio error, and some of the
    reference's "spots" are noise peaks, which keeps "found" well below 100% for every method.
  - A noisy output has many peaks (on MreB about 13 per real spot), so some pair with reference spots by chance; its
    "found" is inflated.
  - Nucleoids are not punctate, so the spot columns say little there.
  - On CARE 2D even the untouched image is far from its reference (slope 0.75, ratio error 31%), so the pairs differ by
    more than noise (acquisition or registration). A supervised model learns that difference; CARE learns it with less
    distortion (0.95) than DORA (0.67).

## What these results do and don't show

- **Mostly one seed per configuration.** Q13 ran the all-level U-Net recipe with three seeds: its raw score spread
  over 0.29 dB (34.26–34.55). Treat single-run differences under about 0.3 dB as run-to-run variation, not effects.
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

## Reproducibility checks (gates 13, 15, 21, 25, 27, 29, 31, 32 and 33)

The blocks marked `fmd-noise` above and the blocks below are verbatim program output. `bricks.sh` gate 13 re-runs
the first two whenever the data is present and requires an exact match. The block below is
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

Gate 15 checks the bar on the last two blocks, and re-runs all three when the P3 weights are present (they come
from `tools/experiments/p3_struct.sh` and stay in the gitignored `runs/p3/`). Plain Noise2Void,
`fmd-eval --weights runs/p3/n2v_plain.bin`:

<!-- p3-n2v-plain -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   33.77 / 0.881
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   31.85 / 0.800
WideField   12   25.23 / 0.374   26.53 / 0.466   27.97 / 0.563
all         48   27.22 / 0.543   30.22 / 0.752   31.68 / 0.774
```
<!-- /p3-n2v-plain -->

Design 1, one network with structured blind spots on the flagged images, `fmd-eval --weights runs/p3/n2v_struct.bin`:

<!-- p3-n2v-shared -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   32.62 / 0.855
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   31.75 / 0.812
WideField   12   25.23 / 0.374   26.53 / 0.466   31.38 / 0.792
all         48   27.22 / 0.543   30.22 / 0.752   32.02 / 0.825
```
<!-- /p3-n2v-shared -->

Design 2, two networks routed by the noise check,
`fmd-eval --weights runs/p3/n2v_indep.bin --weights-correlated runs/p3/n2v_corr.bin`:

<!-- p3-n2v-routed -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   34.09 / 0.886
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   32.46 / 0.819
WideField   12   25.23 / 0.374   26.53 / 0.466   31.76 / 0.807
all         48   27.22 / 0.543   30.22 / 0.752   32.97 / 0.844
routed by the noise check: 12 of 48 images to the correlated-noise model
```
<!-- /p3-n2v-routed -->

Gate 21 applies the P4 rule to the four blocks below (and the slowdown marker above), checks that `NetConfig.Default`
follows it, and re-runs the blocks when the weights are present (`runs/p3/` for the dilated network, `runs/p4/` for
the U-Net, from `tools/experiments/p4_arch.sh`).

<!-- p4-dncnn-sup -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   34.38 / 0.898
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.52 / 0.878
WideField   12   25.23 / 0.374   26.53 / 0.466   33.02 / 0.852
all         48   27.22 / 0.543   30.22 / 0.752   33.75 / 0.880
```
<!-- /p4-dncnn-sup -->

<!-- p4-unet-sup -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.55 / 0.927
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.87 / 0.893
WideField   12   25.23 / 0.374   26.53 / 0.466   34.51 / 0.893
all         48   27.22 / 0.543   30.22 / 0.752   34.73 / 0.907
```
<!-- /p4-unet-sup -->

<!-- p4-dncnn-n2v -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   33.77 / 0.881
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   31.85 / 0.800
WideField   12   25.23 / 0.374   26.53 / 0.466   27.97 / 0.563
all         48   27.22 / 0.543   30.22 / 0.752   31.68 / 0.774
```
<!-- /p4-dncnn-n2v -->

<!-- p4-unet-n2v -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   32.99 / 0.859
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   31.40 / 0.767
WideField   12   25.23 / 0.374   26.53 / 0.466   27.98 / 0.571
all         48   27.22 / 0.543   30.22 / 0.752   31.21 / 0.756
```
<!-- /p4-unet-n2v -->

Gate 25 applies Q1's pre-registered rule to the blocks below (and the baseline's raw row in `p4-unet-sup`), checks that
the verdict recorded above matches it, and re-runs the blocks when the weights are present (`runs/p4/unet_sup.bin`,
`runs/q1/`, from `tools/experiments/q1_supervised.sh`). Baseline at the lower levels:

<!-- q1-baseline -->
```text
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   35.95 / 0.913
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.73 / 0.902
WideField   12   28.19 / 0.540   29.46 / 0.631   35.25 / 0.910
all         48   30.08 / 0.680   32.45 / 0.830   35.04 / 0.909
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   35.68 / 0.904
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   34.59 / 0.905
WideField   12   31.23 / 0.702   32.45 / 0.773   36.20 / 0.918
all         48   32.86 / 0.798   34.88 / 0.894   35.44 / 0.908
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   35.41 / 0.900
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   34.00 / 0.876
WideField   12   34.79 / 0.833   36.01 / 0.878   36.98 / 0.922
all         48   36.03 / 0.889   37.39 / 0.935   35.33 / 0.898
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   34.64 / 0.890
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   34.20 / 0.867
WideField   12   38.74 / 0.923   39.91 / 0.944   37.77 / 0.924
all         48   39.70 / 0.949   40.57 / 0.964   35.28 / 0.891
```
<!-- /q1-baseline -->

Candidate (all levels, 60 epochs):

<!-- q1-candidate -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.31 / 0.924
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.73 / 0.886
WideField   12   25.23 / 0.374   26.53 / 0.466   33.21 / 0.861
all         48   27.22 / 0.543   30.22 / 0.752   34.26 / 0.895
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   37.00 / 0.945
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.44 / 0.898
WideField   12   28.19 / 0.540   29.46 / 0.631   34.36 / 0.890
all         48   30.08 / 0.680   32.45 / 0.830   35.16 / 0.916
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.59 / 0.958
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.67 / 0.932
WideField   12   31.23 / 0.702   32.45 / 0.773   36.19 / 0.918
all         48   32.86 / 0.798   34.88 / 0.894   37.02 / 0.939
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.14 / 0.969
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.24 / 0.941
WideField   12   34.79 / 0.833   36.01 / 0.878   38.06 / 0.940
all         48   36.03 / 0.889   37.39 / 0.935   38.32 / 0.952
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   40.97 / 0.975
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.04 / 0.957
WideField   12   38.74 / 0.923   39.91 / 0.944   40.27 / 0.957
all         48   39.70 / 0.949   40.57 / 0.964   40.15 / 0.965
```
<!-- /q1-candidate -->

Ablation (all levels, 30 epochs):

<!-- q1-ablation -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.15 / 0.918
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.58 / 0.879
WideField   12   25.23 / 0.374   26.53 / 0.466   32.78 / 0.841
all         48   27.22 / 0.543   30.22 / 0.752   34.03 / 0.886
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   36.76 / 0.938
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.43 / 0.894
WideField   12   28.19 / 0.540   29.46 / 0.631   34.36 / 0.884
all         48   30.08 / 0.680   32.45 / 0.830   35.05 / 0.910
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.34 / 0.952
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.44 / 0.922
WideField   12   31.23 / 0.702   32.45 / 0.773   36.21 / 0.915
all         48   32.86 / 0.798   34.88 / 0.894   36.84 / 0.933
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   39.85 / 0.964
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.14 / 0.930
WideField   12   34.79 / 0.833   36.01 / 0.878   38.50 / 0.938
all         48   36.03 / 0.889   37.39 / 0.935   38.28 / 0.946
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   40.70 / 0.971
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   38.89 / 0.951
WideField   12   38.74 / 0.923   39.91 / 0.944   40.93 / 0.956
all         48   39.70 / 0.949   40.57 / 0.964   40.15 / 0.961
```
<!-- /q1-ablation -->

Gate 27 applies Q10's rule to the routed block below (A's raw score from `p4-unet-sup`, B's lower levels from
`q1-candidate`), checks the recorded verdict, and re-runs both blocks when the weights are present. Threshold choice on
the validation FOVs, `fmd-route --weights runs/p4/unet_sup.bin --weights-clean runs/q1/unet_levels60.bin`:

<!-- q10-validation -->
```text
validation FOVs held out by seed 2026: 46 FOVs × 5 levels = 230 images
level    n  sigma median   PSNR A  PSNR B  routed  to B
raw     46       0.03951    35.31   34.97   34.97    46
avg2    46       0.03072    35.93   36.30   36.30    46
avg4    46       0.02342    36.05   37.63   37.63    46
avg8    46       0.01827    36.14   39.02   39.02    46
avg16   46       0.01430    36.27   40.86   40.86    46
all    230                  35.94   37.76   37.76   230
chosen noise threshold: 0.07627
```
<!-- /q10-validation -->

Routed on test_mix with that threshold:

<!-- q10-routed -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.31 / 0.924
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.79 / 0.889
WideField   12   25.23 / 0.374   26.53 / 0.466   33.21 / 0.861
all         48   27.22 / 0.543   30.22 / 0.752   34.28 / 0.896
routed by noise level (σ ≤ 0.07627): 40 of 48 images to the clean-input model
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   37.00 / 0.945
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.66 / 0.902
WideField   12   28.19 / 0.540   29.46 / 0.631   34.36 / 0.890
all         48   30.08 / 0.680   32.45 / 0.830   35.23 / 0.917
routed by noise level (σ ≤ 0.07627): 41 of 48 images to the clean-input model
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.59 / 0.958
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.67 / 0.932
WideField   12   31.23 / 0.702   32.45 / 0.773   36.19 / 0.918
all         48   32.86 / 0.798   34.88 / 0.894   37.02 / 0.939
routed by noise level (σ ≤ 0.07627): 48 of 48 images to the clean-input model
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.14 / 0.969
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.24 / 0.941
WideField   12   34.79 / 0.833   36.01 / 0.878   38.06 / 0.940
all         48   36.03 / 0.889   37.39 / 0.935   38.32 / 0.952
routed by noise level (σ ≤ 0.07627): 48 of 48 images to the clean-input model
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   40.97 / 0.975
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.04 / 0.957
WideField   12   38.74 / 0.923   39.91 / 0.944   40.27 / 0.957
all         48   39.70 / 0.949   40.57 / 0.964   40.15 / 0.965
routed by noise level (σ ≤ 0.07627): 48 of 48 images to the clean-input model
```
<!-- /q10-routed -->

Quality pass Q13 (gate 29 checks the verdict from these blocks and Q1's, and re-runs all five when `runs/q13/` is
present). Each is `fmd-eval --weights runs/q13/<name>.bin --root data/fmd --levels raw,avg2,avg4,avg8,avg16`; the
seed-2026 plain run is Q1's `q1-candidate` above. The validation table in the Q13 section comes from the same command
with `--validation-seed 2026 --sigma-scale 1|2|0.5` (`runs/q13/mechanism_x*.txt`; not part of the bar, so not re-run).

Conditioned, seed 2026:

<!-- q13-cond-2026 -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   34.92 / 0.919
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.69 / 0.889
WideField   12   25.23 / 0.374   26.53 / 0.466   32.55 / 0.834
all         48   27.22 / 0.543   30.22 / 0.752   33.92 / 0.888
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   36.72 / 0.942
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.49 / 0.895
WideField   12   28.19 / 0.540   29.46 / 0.631   34.25 / 0.883
all         48   30.08 / 0.680   32.45 / 0.830   35.03 / 0.912
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.30 / 0.953
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.52 / 0.928
WideField   12   31.23 / 0.702   32.45 / 0.773   36.10 / 0.914
all         48   32.86 / 0.798   34.88 / 0.894   36.82 / 0.935
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.02 / 0.964
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.24 / 0.938
WideField   12   34.79 / 0.833   36.01 / 0.878   38.07 / 0.938
all         48   36.03 / 0.889   37.39 / 0.935   38.27 / 0.949
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   41.19 / 0.970
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.01 / 0.954
WideField   12   38.74 / 0.923   39.91 / 0.944   40.31 / 0.957
all         48   39.70 / 0.949   40.57 / 0.964   40.24 / 0.961
```
<!-- /q13-cond-2026 -->

Conditioned, seed 2027:

<!-- q13-cond-2027 -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.37 / 0.925
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.88 / 0.890
WideField   12   25.23 / 0.374   26.53 / 0.466   33.64 / 0.872
all         48   27.22 / 0.543   30.22 / 0.752   34.44 / 0.900
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   37.01 / 0.945
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.56 / 0.898
WideField   12   28.19 / 0.540   29.46 / 0.631   34.73 / 0.899
all         48   30.08 / 0.680   32.45 / 0.830   35.29 / 0.918
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.59 / 0.957
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.60 / 0.930
WideField   12   31.23 / 0.702   32.45 / 0.773   36.38 / 0.921
all         48   32.86 / 0.798   34.88 / 0.894   37.04 / 0.939
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.26 / 0.968
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.30 / 0.940
WideField   12   34.79 / 0.833   36.01 / 0.878   37.84 / 0.941
all         48   36.03 / 0.889   37.39 / 0.935   38.34 / 0.952
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   41.51 / 0.975
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.00 / 0.955
WideField   12   38.74 / 0.923   39.91 / 0.944   39.83 / 0.958
all         48   39.70 / 0.949   40.57 / 0.964   40.25 / 0.964
```
<!-- /q13-cond-2027 -->

Conditioned, seed 2028:

<!-- q13-cond-2028 -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.28 / 0.926
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.85 / 0.890
WideField   12   25.23 / 0.374   26.53 / 0.466   33.50 / 0.872
all         48   27.22 / 0.543   30.22 / 0.752   34.36 / 0.900
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   37.05 / 0.946
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.51 / 0.895
WideField   12   28.19 / 0.540   29.46 / 0.631   34.50 / 0.898
all         48   30.08 / 0.680   32.45 / 0.830   35.23 / 0.917
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.67 / 0.959
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.58 / 0.929
WideField   12   31.23 / 0.702   32.45 / 0.773   36.23 / 0.921
all         48   32.86 / 0.798   34.88 / 0.894   37.03 / 0.940
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.53 / 0.972
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.35 / 0.941
WideField   12   34.79 / 0.833   36.01 / 0.878   37.22 / 0.941
all         48   36.03 / 0.889   37.39 / 0.935   38.31 / 0.954
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   41.92 / 0.979
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.43 / 0.961
WideField   12   38.74 / 0.923   39.91 / 0.944   39.02 / 0.958
all         48   39.70 / 0.949   40.57 / 0.964   40.36 / 0.968
```
<!-- /q13-cond-2028 -->

Plain, seed 2027:

<!-- q13-plain-2027 -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.58 / 0.927
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.83 / 0.889
WideField   12   25.23 / 0.374   26.53 / 0.466   33.77 / 0.877
all         48   27.22 / 0.543   30.22 / 0.752   34.55 / 0.902
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   37.24 / 0.947
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.69 / 0.902
WideField   12   28.19 / 0.540   29.46 / 0.631   34.77 / 0.900
all         48   30.08 / 0.680   32.45 / 0.830   35.44 / 0.920
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.80 / 0.960
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.64 / 0.933
WideField   12   31.23 / 0.702   32.45 / 0.773   36.37 / 0.920
all         48   32.86 / 0.798   34.88 / 0.894   37.14 / 0.941
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.43 / 0.971
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.52 / 0.944
WideField   12   34.79 / 0.833   36.01 / 0.878   37.99 / 0.938
all         48   36.03 / 0.889   37.39 / 0.935   38.51 / 0.954
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   41.51 / 0.978
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.46 / 0.960
WideField   12   38.74 / 0.923   39.91 / 0.944   40.22 / 0.956
all         48   39.70 / 0.949   40.57 / 0.964   40.51 / 0.966
```
<!-- /q13-plain-2027 -->

Plain, seed 2028:

<!-- q13-plain-2028 -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   35.55 / 0.926
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   33.88 / 0.888
WideField   12   25.23 / 0.374   26.53 / 0.466   33.51 / 0.878
all         48   27.22 / 0.543   30.22 / 0.752   34.48 / 0.902
test_mix level avg2: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   32.11 / 0.805   34.43 / 0.919   37.23 / 0.948
TwoPhoton   16   28.95 / 0.627   32.22 / 0.868   33.57 / 0.900
WideField   12   28.19 / 0.540   29.46 / 0.631   34.55 / 0.902
all         48   30.08 / 0.680   32.45 / 0.830   35.34 / 0.920
test_mix level avg4: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   34.86 / 0.882   36.66 / 0.947   38.84 / 0.961
TwoPhoton   16   31.58 / 0.764   34.46 / 0.917   35.67 / 0.933
WideField   12   31.23 / 0.702   32.45 / 0.773   36.25 / 0.922
all         48   32.86 / 0.798   34.88 / 0.894   37.14 / 0.942
test_mix level avg8: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   37.97 / 0.937   39.28 / 0.967   40.49 / 0.972
TwoPhoton   16   34.53 / 0.871   36.06 / 0.938   36.44 / 0.943
WideField   12   34.79 / 0.833   36.01 / 0.878   37.43 / 0.940
all         48   36.03 / 0.889   37.39 / 0.935   38.38 / 0.954
test_mix level avg16: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   41.08 / 0.970   41.66 / 0.979   41.61 / 0.979
TwoPhoton   16   38.70 / 0.941   39.70 / 0.962   39.50 / 0.961
WideField   12   38.74 / 0.923   39.91 / 0.944   39.47 / 0.956
all         48   39.70 / 0.949   40.57 / 0.964   40.37 / 0.967
```
<!-- /q13-plain-2028 -->

Quality pass Q18 (gate 31 checks the verdict from these blocks and re-runs all twelve when `data/ext` and `runs/q18/`
are present). DORA's are `ext-eval --dataset <set> --weights runs/q18/dora_<n2v|sup>_<set>.bin`. CAREamics' are
`ext-eval --dataset <set> --predictions runs/q18/careamics_<n2v|care>/<set> --label …`, which re-scores its saved outputs
(re-training CAREamics needs the virtualenv). The out-of-the-box rows come from the same command (`runs/q18/eval_*`).

DORA Noise2Void, deepbacs-mreb:

<!-- q18-dora-n2v-mreb -->
```text
deepbacs-mreb test, 5 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Noise2Void on deepbacs-mreb training split, noisy images only: DORA Noise2Void, default recipe; best epoch 28; val loss 0.007466
image        noisy           bilateral       method         
m31.tif       21.19 / 0.328   21.30 / 0.346   25.39 / 0.682
m32.tif       20.48 / 0.282   20.72 / 0.319   24.59 / 0.654
m33.tif       22.40 / 0.423   22.41 / 0.424   26.62 / 0.733
m34.tif       21.81 / 0.359   21.84 / 0.365   25.90 / 0.698
m35.tif       22.48 / 0.449   22.48 / 0.449   26.65 / 0.719
mean          21.67 / 0.368   21.75 / 0.381   25.83 / 0.697
```
<!-- /q18-dora-n2v-mreb -->

CAREamics Noise2Void, deepbacs-mreb:

<!-- q18-careamics-n2v-mreb -->
```text
deepbacs-mreb test, 5 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU
image        noisy           bilateral       method         
m31.tif       21.19 / 0.328   21.30 / 0.346   26.59 / 0.813
m32.tif       20.48 / 0.282   20.72 / 0.319   25.75 / 0.791
m33.tif       22.40 / 0.423   22.41 / 0.424   27.75 / 0.827
m34.tif       21.81 / 0.359   21.84 / 0.365   27.04 / 0.818
m35.tif       22.48 / 0.449   22.48 / 0.449   27.91 / 0.821
mean          21.67 / 0.368   21.75 / 0.381   27.01 / 0.814
```
<!-- /q18-careamics-n2v-mreb -->

DORA supervised, deepbacs-mreb:

<!-- q18-dora-sup-mreb -->
```text
deepbacs-mreb test, 5 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Supervised on deepbacs-mreb training pairs: DORA supervised U-Net; best epoch 31; val loss 0.002245; val PSNR 26.49 dB (+6.43 dB over noisy)
image        noisy           bilateral       method         
m31.tif       21.19 / 0.328   21.30 / 0.346   26.92 / 0.824
m32.tif       20.48 / 0.282   20.72 / 0.319   26.12 / 0.812
m33.tif       22.40 / 0.423   22.41 / 0.424   28.04 / 0.841
m34.tif       21.81 / 0.359   21.84 / 0.365   27.41 / 0.831
m35.tif       22.48 / 0.449   22.48 / 0.449   28.34 / 0.828
mean          21.67 / 0.368   21.75 / 0.381   27.37 / 0.827
```
<!-- /q18-dora-sup-mreb -->

CAREamics CARE, deepbacs-mreb:

<!-- q18-careamics-care-mreb -->
```text
deepbacs-mreb test, 5 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU
image        noisy           bilateral       method         
m31.tif       21.19 / 0.328   21.30 / 0.346   26.88 / 0.809
m32.tif       20.48 / 0.282   20.72 / 0.319   26.18 / 0.801
m33.tif       22.40 / 0.423   22.41 / 0.424   28.08 / 0.837
m34.tif       21.81 / 0.359   21.84 / 0.365   27.36 / 0.817
m35.tif       22.48 / 0.449   22.48 / 0.449   28.24 / 0.824
mean          21.67 / 0.368   21.75 / 0.381   27.35 / 0.818
```
<!-- /q18-careamics-care-mreb -->

DORA Noise2Void, deepbacs-nucleoid:

<!-- q18-dora-n2v-nucleoid -->
```text
deepbacs-nucleoid test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Noise2Void on deepbacs-nucleoid training split, noisy images only: DORA Noise2Void, default recipe; best epoch 21; val loss 0.001445
image        noisy           bilateral       method         
m29.tif       26.76 / 0.762   26.95 / 0.782   33.15 / 0.955
m30.tif       28.01 / 0.854   28.11 / 0.865   34.26 / 0.968
mean          27.39 / 0.808   27.53 / 0.823   33.71 / 0.961
```
<!-- /q18-dora-n2v-nucleoid -->

CAREamics Noise2Void, deepbacs-nucleoid:

<!-- q18-careamics-n2v-nucleoid -->
```text
deepbacs-nucleoid test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU
image        noisy           bilateral       method         
m29.tif       26.76 / 0.762   26.95 / 0.782   34.14 / 0.976
m30.tif       28.01 / 0.854   28.11 / 0.865   35.25 / 0.982
mean          27.39 / 0.808   27.53 / 0.823   34.70 / 0.979
```
<!-- /q18-careamics-n2v-nucleoid -->

DORA supervised, deepbacs-nucleoid:

<!-- q18-dora-sup-nucleoid -->
```text
deepbacs-nucleoid test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Supervised on deepbacs-nucleoid training pairs: DORA supervised U-Net; best epoch 22; val loss 0.0003181; val PSNR 34.97 dB (+7.18 dB over noisy)
image        noisy           bilateral       method         
m29.tif       26.76 / 0.762   26.95 / 0.782   34.88 / 0.979
m30.tif       28.01 / 0.854   28.11 / 0.865   36.03 / 0.985
mean          27.39 / 0.808   27.53 / 0.823   35.45 / 0.982
```
<!-- /q18-dora-sup-nucleoid -->

CAREamics CARE, deepbacs-nucleoid:

<!-- q18-careamics-care-nucleoid -->
```text
deepbacs-nucleoid test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU
image        noisy           bilateral       method         
m29.tif       26.76 / 0.762   26.95 / 0.782   34.76 / 0.977
m30.tif       28.01 / 0.854   28.11 / 0.865   35.84 / 0.985
mean          27.39 / 0.808   27.53 / 0.823   35.30 / 0.981
```
<!-- /q18-careamics-care-nucleoid -->

DORA Noise2Void, zerocost-care2d:

<!-- q18-dora-n2v-care2d -->
```text
zerocost-care2d test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Noise2Void on zerocost-care2d training split, noisy images only: DORA Noise2Void, default recipe; best epoch 28; val loss 0.0003379
image        noisy           bilateral       method         
23.tif        18.76 / 0.457   18.78 / 0.465   18.78 / 0.463
24.tif        19.80 / 0.465   19.83 / 0.473   19.83 / 0.471
mean          19.28 / 0.461   19.30 / 0.469   19.30 / 0.467
```
<!-- /q18-dora-n2v-care2d -->

CAREamics Noise2Void, zerocost-care2d:

<!-- q18-careamics-n2v-care2d -->
```text
zerocost-care2d test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU
image        noisy           bilateral       method         
23.tif        18.76 / 0.457   18.78 / 0.465   18.77 / 0.462
24.tif        19.80 / 0.465   19.83 / 0.473   19.81 / 0.470
mean          19.28 / 0.461   19.30 / 0.469   19.29 / 0.466
```
<!-- /q18-careamics-n2v-care2d -->

DORA supervised, zerocost-care2d:

<!-- q18-dora-sup-care2d -->
```text
zerocost-care2d test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Supervised on zerocost-care2d training pairs: DORA supervised U-Net; best epoch 33; val loss 0.01024; val PSNR 19.90 dB (+1.61 dB over noisy)
image        noisy           bilateral       method         
23.tif        18.76 / 0.457   18.78 / 0.465   19.71 / 0.540
24.tif        19.80 / 0.465   19.83 / 0.473   20.55 / 0.538
mean          19.28 / 0.461   19.30 / 0.469   20.13 / 0.539
```
<!-- /q18-dora-sup-care2d -->

CAREamics CARE, zerocost-care2d:

<!-- q18-careamics-care-care2d -->
```text
zerocost-care2d test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU
image        noisy           bilateral       method         
23.tif        18.76 / 0.457   18.78 / 0.465   19.82 / 0.522
24.tif        19.80 / 0.465   19.83 / 0.473   20.35 / 0.517
mean          19.28 / 0.461   19.30 / 0.469   20.08 / 0.520
```
<!-- /q18-careamics-care-care2d -->

Fidelity (gate 32 re-runs these when their data and weights are present). Each is `dora-bio fidelity` with the arguments
in `tools/experiments/fidelity.sh`:

<!-- fidelity-fmd-bundled -->
```text
FMD test_mix level raw, 48 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on 10 SYNTHETIC 256px fluorescence-like frames (dora-bio synth --seed 1); clean images used only to score validation; best epoch 25; val loss 0.01039; val PSNR 28.67 dB (+8.11 dB over noisy)
            slope ratio err spots found spots real
noisy        0.83      7.5%       65.4%      14.3%
bilateral    0.93      6.0%       64.9%      26.1%
method       0.86      5.6%       51.0%      44.2%
over 23096 foreground tiles and 34412 reference spots
```
<!-- /fidelity-fmd-bundled -->

<!-- fidelity-fmd-sup-dilated -->
```text
FMD test_mix level raw, 48 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Supervised on FMD (12 categories, FOV 19 excluded): supervised raw -> ground truth; best epoch 30; val loss 0.001538; val PSNR 28.13 dB (+6.10 dB over noisy)
            slope ratio err spots found spots real
noisy        0.83      7.5%       65.4%      14.3%
bilateral    0.93      6.0%       64.9%      26.1%
method       0.97      4.5%       51.2%      65.7%
over 23096 foreground tiles and 34412 reference spots
```
<!-- /fidelity-fmd-sup-dilated -->

<!-- fidelity-fmd-sup-unet -->
```text
FMD test_mix level raw, 48 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Supervised on FMD (12 categories, FOV 19 excluded): U-Net, supervised raw -> ground truth; best epoch 28; val loss 0.001168; val PSNR 29.33 dB (+7.30 dB over noisy)
            slope ratio err spots found spots real
noisy        0.83      7.5%       65.4%      14.3%
bilateral    0.93      6.0%       64.9%      26.1%
method       1.02      3.9%       58.3%      71.5%
over 23096 foreground tiles and 34412 reference spots
```
<!-- /fidelity-fmd-sup-unet -->

<!-- fidelity-fmd-sup-unet-levels -->
```text
FMD test_mix level raw, 48 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Supervised on FMD (12 categories, FOV 19 excluded): U-Net, supervised, all 5 noise levels, 60 epochs; best epoch 46; val loss 0.0009593; val PSNR 30.18 dB (+5.38 dB over noisy)
            slope ratio err spots found spots real
noisy        0.83      7.5%       65.4%      14.3%
bilateral    0.93      6.0%       64.9%      26.1%
method       1.00      3.9%       59.4%      63.5%
over 23096 foreground tiles and 34412 reference spots
```
<!-- /fidelity-fmd-sup-unet-levels -->

<!-- fidelity-fmd-n2v-dilated -->
```text
FMD test_mix level raw, 48 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on FMD (12 categories, FOV 19 excluded): Noise2Void, raw frames, plain blind spots; best epoch 23; val loss 0.004518; val PSNR 25.39 dB (+3.36 dB over noisy)
            slope ratio err spots found spots real
noisy        0.83      7.5%       65.4%      14.3%
bilateral    0.93      6.0%       64.9%      26.1%
method       0.88      5.7%       64.8%      27.4%
over 23096 foreground tiles and 34412 reference spots
```
<!-- /fidelity-fmd-n2v-dilated -->

<!-- fidelity-fmd-n2v-unet -->
```text
FMD test_mix level raw, 48 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on FMD (12 categories, FOV 19 excluded): U-Net, Noise2Void, raw frames, plain blind spots; best epoch 4; val loss 0.004745; val PSNR 25.56 dB (+3.53 dB over noisy)
            slope ratio err spots found spots real
noisy        0.83      7.5%       65.4%      14.3%
bilateral    0.93      6.0%       64.9%      26.1%
method       0.86      6.1%       64.0%      23.3%
over 23096 foreground tiles and 34412 reference spots
```
<!-- /fidelity-fmd-n2v-unet -->

<!-- fidelity-deepbacs-mreb-dora-n2v -->
```text
deepbacs-mreb test, 5 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on deepbacs-mreb training split, noisy images only: DORA Noise2Void, default recipe; best epoch 28; val loss 0.007466
            slope ratio err spots found spots real
noisy        1.00      7.1%       55.2%       7.1%
bilateral    1.01      7.3%       54.6%       8.1%
method       0.75      8.9%       45.6%      34.2%
over 711 foreground tiles and 2534 reference spots
```
<!-- /fidelity-deepbacs-mreb-dora-n2v -->

<!-- fidelity-deepbacs-mreb-dora-sup -->
```text
deepbacs-mreb test, 5 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Supervised on deepbacs-mreb training pairs: DORA supervised U-Net; best epoch 31; val loss 0.002245; val PSNR 26.49 dB (+6.43 dB over noisy)
            slope ratio err spots found spots real
noisy        1.00      7.1%       55.2%       7.1%
bilateral    1.01      7.3%       54.6%       8.1%
method       0.94      6.7%       38.5%      67.2%
over 711 foreground tiles and 2534 reference spots
```
<!-- /fidelity-deepbacs-mreb-dora-sup -->

<!-- fidelity-deepbacs-mreb-careamics-n2v -->
```text
deepbacs-mreb test, 5 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU
            slope ratio err spots found spots real
noisy        1.00      7.1%       55.2%       7.1%
bilateral    1.01      7.3%       54.6%       8.1%
method       0.86      7.4%       43.8%      56.2%
over 711 foreground tiles and 2534 reference spots
```
<!-- /fidelity-deepbacs-mreb-careamics-n2v -->

<!-- fidelity-deepbacs-mreb-careamics-care -->
```text
deepbacs-mreb test, 5 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU
            slope ratio err spots found spots real
noisy        1.00      7.1%       55.2%       7.1%
bilateral    1.01      7.3%       54.6%       8.1%
method       1.03      7.4%       41.4%      66.2%
over 711 foreground tiles and 2534 reference spots
```
<!-- /fidelity-deepbacs-mreb-careamics-care -->

<!-- fidelity-deepbacs-nucleoid-dora-n2v -->
```text
deepbacs-nucleoid test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on deepbacs-nucleoid training split, noisy images only: DORA Noise2Void, default recipe; best epoch 21; val loss 0.001445
            slope ratio err spots found spots real
noisy        1.00      3.0%       21.2%      14.2%
bilateral    1.01      3.1%       19.6%      14.0%
method       0.93      3.6%       34.5%      11.5%
over 262 foreground tiles and 3738 reference spots
```
<!-- /fidelity-deepbacs-nucleoid-dora-n2v -->

<!-- fidelity-deepbacs-nucleoid-dora-sup -->
```text
deepbacs-nucleoid test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Supervised on deepbacs-nucleoid training pairs: DORA supervised U-Net; best epoch 22; val loss 0.0003181; val PSNR 34.97 dB (+7.18 dB over noisy)
            slope ratio err spots found spots real
noisy        1.00      3.0%       21.2%      14.2%
bilateral    1.01      3.1%       19.6%      14.0%
method       0.98      2.8%       21.2%      11.9%
over 262 foreground tiles and 3738 reference spots
```
<!-- /fidelity-deepbacs-nucleoid-dora-sup -->

<!-- fidelity-deepbacs-nucleoid-careamics-n2v -->
```text
deepbacs-nucleoid test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU
            slope ratio err spots found spots real
noisy        1.00      3.0%       21.2%      14.2%
bilateral    1.01      3.1%       19.6%      14.0%
method       0.97      3.1%       22.2%      12.8%
over 262 foreground tiles and 3738 reference spots
```
<!-- /fidelity-deepbacs-nucleoid-careamics-n2v -->

<!-- fidelity-deepbacs-nucleoid-careamics-care -->
```text
deepbacs-nucleoid test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU
            slope ratio err spots found spots real
noisy        1.00      3.0%       21.2%      14.2%
bilateral    1.01      3.1%       19.6%      14.0%
method       0.99      3.0%       17.8%      14.9%
over 262 foreground tiles and 3738 reference spots
```
<!-- /fidelity-deepbacs-nucleoid-careamics-care -->

<!-- fidelity-zerocost-care2d-dora-n2v -->
```text
zerocost-care2d test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on zerocost-care2d training split, noisy images only: DORA Noise2Void, default recipe; best epoch 28; val loss 0.0003379
            slope ratio err spots found spots real
noisy        0.75     31.0%        4.2%      35.3%
bilateral    0.76     31.3%        4.3%      34.4%
method       0.75     31.1%        4.3%      35.0%
over 7297 foreground tiles and 2584 reference spots
```
<!-- /fidelity-zerocost-care2d-dora-n2v -->

<!-- fidelity-zerocost-care2d-dora-sup -->
```text
zerocost-care2d test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Supervised on zerocost-care2d training pairs: DORA supervised U-Net; best epoch 33; val loss 0.01024; val PSNR 19.90 dB (+1.61 dB over noisy)
            slope ratio err spots found spots real
noisy        0.75     31.0%        4.2%      35.3%
bilateral    0.76     31.3%        4.3%      34.4%
method       0.67     22.8%        6.4%      41.4%
over 7297 foreground tiles and 2584 reference spots
```
<!-- /fidelity-zerocost-care2d-dora-sup -->

<!-- fidelity-zerocost-care2d-careamics-n2v -->
```text
zerocost-care2d test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: CAREamics 0.3.5 Noise2Void, default 2D config, 30 x 50 steps x batch 16, CPU
            slope ratio err spots found spots real
noisy        0.75     31.0%        4.2%      35.3%
bilateral    0.76     31.3%        4.3%      34.4%
method       0.80     33.0%        5.5%      31.9%
over 7297 foreground tiles and 2584 reference spots
```
<!-- /fidelity-zerocost-care2d-careamics-n2v -->

<!-- fidelity-zerocost-care2d-careamics-care -->
```text
zerocost-care2d test, 2 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: CAREamics 0.3.5 CARE, default 2D config, 60 x 50 steps x batch 16, CPU
            slope ratio err spots found spots real
noisy        0.75     31.0%        4.2%      35.3%
bilateral    0.76     31.3%        4.3%      34.4%
method       0.95     22.3%        1.5%      46.0%
over 7297 foreground tiles and 2584 reference spots
```
<!-- /fidelity-zerocost-care2d-careamics-care -->

Quality pass Q12 (gate 33 checks the verdict and the selection log against the pre-registered rule, and re-runs these
when `data/ext`, `data/fmd` and `runs/q12/` are present). Test splits: `ext-eval --dataset <set> --weights
runs/q12/unet16_<set>.bin`; MreB fidelity: `fidelity --dataset deepbacs-mreb --weights runs/q12/unet16_deepbacs-mreb.bin`;
FMD guard: `fmd-eval --weights runs/q12/fmd_guard.bin --root data/fmd --levels raw`.

<!-- q12-test-mreb -->
```text
deepbacs-mreb test, 5 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Noise2Void on deepbacs-mreb training split, noisy images only: Q12 candidate unet16 (--arch unet --features 16 --depth 2); best epoch 22; val loss 0.007226
image        noisy           bilateral       method         
m31.tif       21.19 / 0.328   21.30 / 0.346   26.22 / 0.754
m32.tif       20.48 / 0.282   20.72 / 0.319   25.41 / 0.744
m33.tif       22.40 / 0.423   22.41 / 0.424   27.39 / 0.767
m34.tif       21.81 / 0.359   21.84 / 0.365   26.68 / 0.758
m35.tif       22.48 / 0.449   22.48 / 0.449   27.54 / 0.753
mean          21.67 / 0.368   21.75 / 0.381   26.65 / 0.755
```
<!-- /q12-test-mreb -->

<!-- q12-test-nucleoid -->
```text
deepbacs-nucleoid test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Noise2Void on deepbacs-nucleoid training split, noisy images only: Q12 candidate unet16 (--arch unet --features 16 --depth 2); best epoch 9; val loss 0.001382
image        noisy           bilateral       method         
m29.tif       26.76 / 0.762   26.95 / 0.782   33.55 / 0.962
m30.tif       28.01 / 0.854   28.11 / 0.865   34.73 / 0.974
mean          27.39 / 0.808   27.53 / 0.823   34.14 / 0.968
```
<!-- /q12-test-nucleoid -->

<!-- q12-test-care2d -->
```text
zerocost-care2d test, 2 images: PSNR (dB) / SSIM vs the high-SNR reference, CARE's normalization
method: Noise2Void on zerocost-care2d training split, noisy images only: Q12 candidate unet16 (--arch unet --features 16 --depth 2); best epoch 30; val loss 0.00032
image        noisy           bilateral       method         
23.tif        18.76 / 0.457   18.78 / 0.465   18.77 / 0.464
24.tif        19.80 / 0.465   19.83 / 0.473   19.83 / 0.472
mean          19.28 / 0.461   19.30 / 0.469   19.30 / 0.468
```
<!-- /q12-test-care2d -->

<!-- q12-fidelity-mreb -->
```text
deepbacs-mreb test, 5 images: fidelity vs the high-SNR reference (1.00 / 0% / 100% / 100% is perfect)
method: Noise2Void on deepbacs-mreb training split, noisy images only: Q12 candidate unet16 (--arch unet --features 16 --depth 2); best epoch 22; val loss 0.007226
            slope ratio err spots found spots real
noisy        1.00      7.1%       55.2%       7.1%
bilateral    1.01      7.3%       54.6%       8.1%
method       0.93      6.6%       43.8%      41.2%
over 711 foreground tiles and 2534 reference spots
```
<!-- /q12-fidelity-mreb -->

<!-- q12-fmd-guard -->
```text
test_mix level raw: mean PSNR (dB) / SSIM vs ground truth, 8-bit scale
modality     n  noisy           bilateral       DORA           
Confocal    20   29.13 / 0.701   31.74 / 0.872   32.99 / 0.859
TwoPhoton   16   26.32 / 0.473   31.07 / 0.816   31.40 / 0.767
WideField   12   25.23 / 0.374   26.53 / 0.466   27.98 / 0.571
all         48   27.22 / 0.543   30.22 / 0.752   31.21 / 0.756
```
<!-- /q12-fmd-guard -->

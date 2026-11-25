# DORA v2.0-bio

**Self-supervised fluorescence-microscopy denoising, built from scratch in C# and shown in Godot 4.5.**

This folder is a ground-up rebuild of Project DORA (the v1 MNIST digit app at the repo root, which is
left untouched). DORA v2 learns to remove photon shot noise and camera noise from microscope images.
With **Noise2Void** it learns from the noisy images alone, with no clean ground truth, and it can keep
adapting to the sample on screen. Like v1, every line of the ML is readable code with no ML framework.

![DORA v2 UI: raw, bilateral baseline and DORA output on a synthetic sample](docs/screenshot.png)

*Real render of the Godot app (Xvfb + Mesa). The sample image is **synthetic**: see the honesty notes below.*

## What's verified

All numbers come from this repo, run on a 4-core cloud VM with no GPU.

| Check | Result |
|---|---|
| Core unit tests (`dotnet test`) | 70 passing in ~4 s, including finite-difference gradient checks (mutation-tested) |
| Brick gates (`./bricks.sh`) | all 11 green; documentation of every public member is enforced by the build |
| Gates can fail (`./bricks.sh --self-test`) | one planted bug per gate 0–7, all 9 caught (after a clean control copy passes) |
| CI (GitHub Actions) | all gates incl. headless Godot, on PRs into `main`, pushes to `main` and manual runs (docs-only changes skipped): [run #1](https://github.com/Andraszao/project-dora-ml/actions/runs/45617118232) green in 150 s of runner time, 0 billable minutes (public repo) |
| Codecs vs. independent encoders | TIFF/PNG fixtures written by Pillow, ImageMagick and libtiff decode exactly |
| Tiled vs. whole-image inference | identical (≤1e-5) for ragged tiles, tiles smaller than the receptive field, etc. |
| Training reproducibility | two runs with the same seed produce bit-identical weights |
| Bundled model training | Noise2Void on 10 synthetic 256 px frames, reached the 8 dB "mastery" target at epoch 25 in 5 min |
| Held-out evaluation (4 unseen synthetic FOVs) | noisy **20.89 dB / SSIM 0.223** → bilateral **28.43 / 0.677** → DORA **30.21 / 0.809** |
| Godot 4.5.1 .NET smoke test (headless) | loads model and sample, denoises off-thread, adapts, stays responsive; exit code 0 |
| Speed | 512×384 frame in ~0.5 s (CLI and in-editor) |

## Quick start

**Run the app.** Requires the **.NET build** of Godot 4.5.x and the .NET 8 SDK.

```bash
cd dora-bio/godot
dotnet build            # or press Build in the Godot editor
godot --path .          # or open dora-bio/godot/project.godot in the editor and press F5
```

Click **Load sample**, then **Denoise**. **Adapt to sample** runs self-supervised fine-tuning on the image on
screen. **Train (synthetic)** generates data in memory and keeps training. **Train on folder…** trains on your
own data. **Save weights** writes `user://dora_bio_weights.bin`.

**Build brick by brick.** The project is a stack of 11 bricks (scaffold → nn → data → io → inference → training →
cli → godot-build → godot-smoke → ci-core → ci-godot). Each has a gate that must pass before the next is trusted,
and CI runs them all on PRs into `main`:

```bash
cd dora-bio
./bricks.sh --list       # what each gate checks
./bricks.sh              # run gates 0..8 in order, stop at the first red one
./bricks.sh --only 5     # iterate on a single brick
./bricks.sh --self-test  # prove the gates can fail: one planted bug per gate, each must turn red
```

**Headless (CLI).** Works anywhere .NET 8 runs, with no Godot needed:

```bash
cd dora-bio
dotnet test DoraBio.sln
alias dora-bio="dotnet run --project tools/DoraBio.Cli -c Release --"
dora-bio synth   --out data/synth --count 10 --size 256
dora-bio train   --data data/synth --out runs/w.bin --mode n2v --journey runs/journey.json
dora-bio eval    --weights runs/w.bin --data data/heldout
dora-bio denoise --weights godot/models/dora_bio_weights.bin --in my_image.tif --out clean.tif --baseline bilateral.tif
```

**Your own data.** Use TIFF (8/16/32-bit) or PNG. A training folder can be:
- `noisy/` with optional `clean/`, matched by file name. Supervised training if every image has a clean match, otherwise Noise2Void.
- The FMD dataset layout: `raw/<fov>/*.png` with `gt/<fov>/avg50.png`. This follows the published layout but **hasn't been tested on the real download yet**.
- A flat folder of noisy images (Noise2Void only).

## How it works

```
Godot (UI, main thread)                          DoraBio.Core (pure net8.0, no Godot, no NuGet)
┌───────────────────────────────┐   calls      ┌───────────────────────────────────────────────┐
│ Main.tscn + MicroscopyViewport│ ───────────▶ │ ImageLoader (TIFF/PNG) → Normalization (float) │
│   .gd (widgets only)          │              │ DenoisingNet: 7× dilated 3×3 conv, residual    │
│ contrast_stretch.gdshader     │              │ TiledInference (parallel, exact)               │
│   (Rf float textures, shared  │   signals    │ Trainer: mini-batch Adam, N2V blind spots,     │
│    window, keep-aspect)       │ ◀─────────── │   FOV-held-out validation, PSNR/SSIM, LR decay │
│ DoraBioBridge.cs ─ Task.Run ─▶│ (CallDeferred│ OnlineAdapter (persistent Adam), JourneyLog    │
└───────────────────────────────┘  to main)    └───────────────────────────────────────────────┘
```

- **Model:** a DnCNN-style residual CNN. It predicts the noise, and the output is `input − noise`. Dilations
  1-2-4-8-4-2-1 give a 45 px receptive field with 26,497 parameters, stored in a 106 KB binary file with a CRC.
- **Noise2Void:** about 2% of pixels per patch are swapped for a random neighbor's value, and the loss is computed
  only at those pixels. The network can't copy a pixel's own noise, so it learns the signal.
- **Progressive targets (DORA's milestone loop):** targets are PSNR *gains* over the noisy input (+2, +4, +6,
  +8 dB). Two hits in a row raise the target, and holding the last one is mastery. Unlike v1, plateaus
  *lower* the learning rate (×0.85), and the best weights are kept.
- **Journey log:** `user://dora_bio_journey.json` gets one session per training run, with every epoch recorded.

## v1 problems fixed in v2

| v1 (GDScript, `02dd999`) | v2 |
|---|---|
| "Batch size" only changed logging; updates were per sample | real mini-batches, with per-sample gradients summed in a fixed order |
| Hidden-layer backprop used already-updated output weights | full backward pass before one optimizer step; verified with finite differences |
| Fresh Adam on every feedback click, so updates were `lr·sign(g)` | `OnlineAdapter` keeps Adam state for as long as the model is loaded |
| Learning rate *multiplied up* on success | learning rate only decays on plateaus; best weights restored and checkpointed |
| 98% max target was unreachable | the maximum is the last target, and holding it is mastery |
| Training accuracy reported as if it were validation | validation held out by field of view; PSNR/SSIM against clean data |
| Dataset loaded and augmented on every launch | only weights load at startup; data loads when training starts |
| Journey log: empty epochs, duplicate sessions | session opens only when training starts, every epoch saved, ends `completed` or `failed` |
| 8-bit RGBA canvas, non-square panel squash | float32 end to end, `Image.Format.Rf`, keep-aspect display |
| Synchronous training froze the window | every heavy job runs on the thread pool, with cancel |
| Fallback path skipped signal wiring; `get_node("../…")` paths | `@export` node references; one bridge class |

## Changes from the original v2 proposal

The plan this rebuild started from had a few errors. Where the build differs from it:

- **No HDF5 / PureHDF.** The FMD dataset ships as PNG, not `.h5`, and the core takes no dependencies.
  HDF5 input gets a clear "not supported" error. Export frames to TIFF, or add a reader later in a separate project.
- **Signal names from GDScript are PascalCase** (`bridge.DenoiseFinished`), not `inference_finished`.
- **Starter-code bugs fixed:** `SubtractResidual` clamped only its scalar tail (outputs are now never clamped).
  The blind-spot mask used a fixed seed, so every step got the same mask (now a fresh mask each step).
  `Conv2D` had no backward pass (now forward and backward, gradient-checked).
- **Residual dilated CNN instead of a U-Net** for this first version: much less code to verify, an exact tiling
  guarantee, and a 45 px receptive field. A U-Net is a reasonable next step.
- **Progressive targets on PSNR gain**, not fixed 24/26/28/30 dB, because absolute PSNR depends mostly on the
  noise level, so fixed targets wouldn't carry over between datasets.

## Roadmap

[`PATHWAY.md`](PATHWAY.md) lays out phases P0–P10, from today's foundation (P0, done) through CI, validation on
real FMD data, noise-model fidelity, a U-Net, performance, stacks, analysis tools, model management, more formats,
and a release. Each phase is a set of gated bricks.

## Honesty notes and limitations

- **Everything shipped here is synthetic.** `godot/samples/synthetic_cells.tif` and the bundled weights come from
  `SyntheticFluorescence` (procedural cells, filaments and puncta with Poisson–Gaussian noise). The held-out
  numbers above are on synthetic data. **Performance on real microscopes has not been measured.** Use
  **Adapt to sample**, or train on your data, and check with `dora-bio eval` on held-out FOVs that have ground truth.
- Noise2Void assumes pixel-independent, zero-mean noise. Correlated noise (some sCMOS patterns, structured
  artifacts) breaks this assumption and leaves residual structure.
- The "estimated noise σ" in the UI is Immerkær's white-noise estimate: a rough image-wide figure, not calibrated.
- C# Godot projects **cannot export to the web** (Godot 4.5), so v2 targets desktop and mobile only.
- CPU only. That's fine for 512² frames and small models; large stacks or 3D volumes would need GPU work.

## Layout

See [`CLAUDE.md`](CLAUDE.md) for the file map, the architecture rules, and the Godot⇄C# gotchas, written for
AI coding sessions but useful for humans too.

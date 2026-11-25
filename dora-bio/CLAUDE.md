# dora-bio/: DORA v2.0-bio (context and rules for AI sessions)

This folder is a **separate rebuild** of Project DORA as a fluorescence-microscopy denoiser. The v1 digit
recognizer at the repo root (GDScript, MNIST) is untouched and unrelated at runtime. `dora-bio/.gdignore`
keeps v1's Godot project from importing anything in here.

## What it is

A self-supervised image denoiser: a small dilated residual CNN (DnCNN-style, 7 layers × 24 features,
45 px receptive field, 26,497 parameters), trained with **Noise2Void** (noisy images only) or supervised
(noisy → clean pairs). All math is hand-written C# with SIMD row kernels and no ML framework.
A Godot 4.5 .NET app shows raw, classical (bilateral) and DORA output side by side. It can adapt to the
loaded sample in-app, or train on synthetic data or on a folder.

## Layout

```
DoraBio.sln                    core + tests + CLI (no Godot)
src/DoraBio.Core/              ALL ML/IO code. net8.0. No Godot, no NuGet dependencies.
  Nn/        ConvLayer (3×3 dilated conv fwd/bwd), DenoisingNet, AdamOptimizer, Simd, WorkspacePool
  Data/      GrayImage, Normalization, Dihedral, BlindSpot, PatchSampler, ImagePair/DatasetSplit,
             DatasetLoader, SyntheticFluorescence
  IO/        TiffCodec, PngDecoder, ImageLoader, WeightsFile, Crc32
  Training/  Trainer (+BatchStepper), ValidationSet, ProgressiveTargets, TrainingJob, JourneyLog,
             OnlineAdapter, TrainerConfig
  Metrics.cs, TiledInference.cs, BilateralBaseline.cs
tests/DoraBio.Core.Tests/      xunit. Fixtures/ written by Pillow/ImageMagick (make_fixtures.py)
tools/DoraBio.Cli/             `dora-bio` CLI: synth | train | denoise | eval | info
godot/                         Godot 4.5 .NET project (open THIS folder in Godot)
  DoraBio.Godot.csproj         Godot.NET.Sdk/4.5.1, ProjectReference → src/DoraBio.Core
  scripts/DoraBioBridge.cs     the ONLY Godot↔core boundary ([GlobalClass] Node)
  scripts/MicroscopyViewport.gd  UI wiring only
  scenes/Main.tscn, shaders/contrast_stretch.gdshader
  samples/synthetic_cells.tif  SYNTHETIC demo frame (uint16)
  models/dora_bio_weights.bin  bundled weights, trained on SYNTHETIC data only
  tests/smoke_test.tscn        headless end-to-end check
```

## Commands (all verified in a Linux container without a GPU)

```bash
cd dora-bio
dotnet test DoraBio.sln                                   # about 70 tests, ~30 s
dotnet run --project tools/DoraBio.Cli -c Release -- synth --out data/synth --count 10 --size 256
dotnet run --project tools/DoraBio.Cli -c Release -- train --data data/synth --out runs/w.bin --mode n2v
cd godot && dotnet build && godot --headless --path . --import \
  && godot --headless --path . res://tests/smoke_test.tscn  # exit code 0 = pass
```

`godot` must be the **.NET** build of Godot 4.5.x. `data/` and `runs/` are gitignored.

## Rules: don't regress these

1. **Core stays pure.** No `using Godot` and no NuGet packages in `src/DoraBio.Core`. Anything Godot-specific
   (FileAccess, res:// and user:// paths, Image, signals, threading) belongs in `DoraBioBridge.cs`.
2. **Never block Godot's main thread.** Denoise, adapt and train go through `DoraBioBridge.RunJob`
   (thread pool, one job at a time). Results return via `CallDeferred`. The bridge becomes idle *before* the
   continuation emits result signals, so handlers may start the next job.
3. **No 8-bit anywhere in the data path.** Images are `float` end to end. Display uses `Image.Format.Rf`
   textures with a shared min/max window in `contrast_stretch.gdshader` and `KeepAspectCentered`.
   Never use `Image.LoadFromFile` for microscopy data, and never clamp normalized intensities in training
   or inference (clamping the noise tails biases a denoiser).
4. **Gradients are tested.** Any change to `ConvLayer`, `DenoisingNet` or the loss must keep
   `GradientTests` (finite differences) passing. Mutation-checked: a mirrored-tap bug fails all 3 cases.
5. **Tiled inference must equal whole-image inference** (`TiledInference` margin = receptive radius,
   clipped at image edges). Covered by `TiledInferenceMatchesWholeImage`.
6. **Optimizer state persists** across online adaptation (`OnlineAdapter`). A fresh Adam at t = 1 is a sign-step
   (v1's bug). Adaptation never saves implicitly. The user clicks Save, and the UI shows "unsaved changes".
7. **Learning rate only decays** (×0.85 after `PlateauPatience` epochs without a new best). The best weights
   are snapshotted and restored at the end, and checkpointed to disk at each new best.
8. **Validation is held out by FOV** (`ImagePair.GroupId`). Frames of one field of view never straddle the split.
9. **Journey log lifecycle:** a session opens only when training starts. Every epoch is appended and saved.
   The session ends as `completed` or `failed: …`. Inference-only launches write nothing.
10. **Label synthetic things as synthetic.** The bundled sample and weights come from `SyntheticFluorescence`.
    Don't describe them as real microscope data, and don't claim real-data accuracy that hasn't been measured.

## Godot ⇄ C# gotchas

- From GDScript, C# members keep **PascalCase**: `bridge.StartDenoise()`, `bridge.DenoiseFinished.connect(f)`.
  Signal names drop the `EventHandler` suffix (`DenoiseFinishedEventHandler` → `DenoiseFinished`).
- A C# signal named `X` generates an event `X`, so no other member may be called `X` (that's why it's `WriteLog`).
- `Godot.Environment` shadows `System.Environment` in bridge code; qualify it.
- Children are `_Ready` before parents. The UI calls `bridge.DescribeModel()` after connecting signals,
  because `ModelChanged` was already emitted during the bridge's `_Ready`.
- C# Godot projects **cannot export to the web** (Godot 4.5). Desktop and mobile only.
- Save paths: user weights `user://dora_bio_weights.bin`, journey `user://dora_bio_journey.json`. The bundled
  model at `res://models/` is a read-only fallback.

## File formats

- **Weights** (`WeightsFile`): `"DORABIO\0"`, u16 format = 1, model version `"2.0.0-bio"`, provenance string,
  saved-UTC ticks, features, layer count, dilations[], parameter count, float32 LE[], CRC-32. v1 JSON `.dat`
  files are rejected with a clear message.
- **Images in:** TIFF (8/16/32-bit int/float; none/Deflate/PackBits; predictor 2; LE/BE; strips; multi-page)
  and PNG (8/16-bit gray/RGB(A), all filters). Not supported, each with a clear error: LZW, tiled TIFF,
  BigTIFF, palettes, interlaced PNG, **HDF5**.
- **Datasets** (`DatasetLoader`): `noisy/`+`clean/`; FMD `raw/<fov>/`+`gt/<fov>/` (written to the published
  layout, **untested against the real download**); or a flat folder (Noise2Void only).

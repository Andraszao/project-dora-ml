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
  Nn/        ConvLayer (3×3 dilated conv fwd/bwd), DenoisingNet (+ .UNet.cs: the U-Net graph; .Film.cs: noise conditioning), Resample (2×2 pool,
             upsample), AdamOptimizer, Simd, WorkspacePool
  Data/      GrayImage, Normalization, Anscombe (variance stabilization), Dihedral, V1Augment, BlindSpot (+MaskFootprint),
             PatchSampler, ImagePair/DatasetSplit,
             DatasetLoader, FmdDataset (real FMD layouts + test-FOV exclusion), ExtDatasets (the public paired sets in data/ext),
             SyntheticFluorescence
  IO/        TiffCodec, TiffStack (page-by-page reader and writer), PngDecoder, ImageLoader, WeightsFile, Crc32
  Training/  Trainer (+BatchStepper), ValidationSet, ProgressiveTargets, TrainingJob, JourneyLog,
             OnlineAdapter, TrainerConfig
  Metrics.cs, TiledInference.cs, BilateralBaseline.cs, FmdBenchmark.cs (held-out FMD scores, noise correlation),
  BatchDenoising.cs (FrameDenoiser: the one per-frame pipeline; stacks and folders streamed frame by frame),
  NoiseDiagnostics.cs (is the noise pixel-independent? from the noisy image alone),
  NoiseCalibration.cs (variance = gain · intensity + intercept, from the noisy image alone),
  NoiseConditioning.cs (the per-frame condition a noise-conditioned network gets: ln(σ̂ / 0.05) of the whole frame)
tests/DoraBio.Core.Tests/      xunit. Fixtures/ written by Pillow/ImageMagick (make_fixtures.py)
tools/DoraBio.Cli/             `dora-bio` CLI: synth | train | denoise | eval | info | bench | diagnose | fmd-check | fmd-eval |
                               fmd-noise | fmd-diagnose | fmd-route
tools/fetch_fmd.sh             downloads + verifies the real FMD dataset into data/fmd/ (gitignored)
tools/fetch_ext.sh             downloads + verifies the public paired datasets of the field-validation track into data/ext/
                               (DeepBacs, ZeroCostDL4Mic CARE; CC BY 4.0, gitignored); tools/extract_zip.py checks every entry
tools/experiments/p2_fmd.sh    every P2 training run and evaluation behind docs/RESULTS.md
docs/RESULTS.md                measured results on real microscope data (FMD), with reproducible blocks
godot/                         Godot 4.5 .NET project (open THIS folder in Godot)
  DoraBio.Godot.csproj         Godot.NET.Sdk/4.5.1, ProjectReference → src/DoraBio.Core
  scripts/DoraBioBridge.cs     the ONLY Godot↔core boundary ([GlobalClass] Node)
  scripts/MicroscopyViewport.gd  UI wiring only
  scenes/Main.tscn, shaders/contrast_stretch.gdshader
  samples/synthetic_cells.tif  SYNTHETIC demo frame (uint16)
  models/dora_bio_weights.bin  bundled weights, trained on SYNTHETIC data only
  tests/smoke_test.tscn        headless end-to-end check
```

## Development pathway

Work follows [`PATHWAY.md`](PATHWAY.md): phases **P0 → P10**, each a group of bricks with gates. P0 (bricks 0–8),
P1 (bricks 9–10, CI), P2 (bricks 11–13, real FMD data), P3 (bricks 14–17, noise-model fidelity), P4 (bricks 18–21,
U-Net) and P5 (bricks 22–24, performance) are done. **The current phase is P6 (stacks, time-lapse and batch)**:
bricks 25–30 are done (30 opened the field-validation track, FV); next is `ext-bench`.
Pick up the next unfinished brick of the current phase, follow the phase protocol there, and update its status table when a phase closes. Read `docs/RESULTS.md` before
changing training or the model: it records what actually works on real data.

## Brick-by-brick workflow (how this project is built: follow it)

The build is a stack of bricks. **Each brick depends only on the bricks below it, and has a gate that must pass
before anything is built on top.** `./bricks.sh` runs the gates in order and stops at the first failure.

| # | Brick | Source | Gate |
|---|---|---|---|
| 0 | scaffold | `*.sln`, `*.csproj`, `Directory.Build.props` | solution builds with warnings as errors; every test class has exactly one brick trait |
| 1 | nn | `Nn/` | `Brick=1-nn`: finite-difference gradient checks, Adam behavior |
| 2 | data | `Data/` (except `DatasetLoader`) | `Brick=2-data`: dihedral, blind-spot, normalization, sampler, synthetic noise statistics |
| 3 | io | `IO/` | `Brick=3-io`: decoders vs Pillow/ImageMagick/libtiff fixtures, weights round-trip and CRC |
| 4 | inference | `Metrics.cs`, `TiledInference.cs`, `BilateralBaseline.cs` | `Brick=4-inference`: tiled == whole image, PSNR/SSIM/noise σ |
| 5 | training | `Training/`, `Data/DatasetLoader.cs` (needs IO) | `Brick=5-training`: learns on synthetic data, reproducible, journey lifecycle |
| 6 | cli | `tools/DoraBio.Cli` | synth → train → info → denoise → eval end to end; bundled model and sample still work |
| 7 | godot-build | `godot/*.csproj`, `scripts/*.cs` | Godot C# assembly compiles against Godot.NET.Sdk 4.5.1; every member in `godot/scripts/*.cs` is documented |
| 8 | godot-smoke | `godot/scenes`, `scripts/*.gd`, `shaders` | headless Godot smoke test (needs `GODOT=…`, otherwise **SKIPPED**, which is not a pass; `REQUIRE_GODOT=1` makes it a FAIL) |
| 9 | ci-core | `.github/workflows/dora-bio.yml` (job `gates`), `tools/ci/`, `bricks.sh --self-test` | workflow structure check + self-test: one planted bug per gate 0–7, each must turn its gate red |
| 10 | ci-godot | workflow job `godot-smoke` | CI's Godot version = Godot.NET.Sdk version; download SHA-512 pinned and verified; gate 8 mandatory in CI |
| 11 | fmd-fetch | `tools/fetch_fmd.sh` | all 13 FMD archives pinned (Drive id, size, SHA-256); with data: every archive's digest and image count match |
| 12 | fmd-loader | `Data/FmdDataset.cs` | `Brick=12-fmd` (miniature real-layout tree); with data: `dora-bio fmd-check` (structure + test_mix = FOV 19 pixel for pixel, FOV 19 never in training) |
| 13 | fmd-eval | `FmdBenchmark.cs`, `tools/experiments/`, `docs/RESULTS.md` | `Brick=13-fmd-eval`; RESULTS.md exists; with data: its recorded synthetic-model evaluation and noise-correlation blocks reproduce exactly |
| 14 | n2v-diagnostic | `NoiseDiagnostics.cs`, CLI `diagnose` / `fmd-diagnose`, the bridge's noise check on load | `Brick=14-n2v-diagnostic` (independent vs smoothed synthetic noise); with data: the blind verdict matches the ground-truth verdict on all 48 raw test images |
| 15 | structured-mask | `Data/BlindSpot.cs` (`MaskFootprint`), `ImagePair.Footprint`, `Training/StructuredMasking.cs`, CLI `--struct` / `--subset` / `--weights-correlated` | `Brick=15-structured-mask`; the recorded routed two-network results meet the bar set in advance (widefield ≥ +1 dB, confocal and two-photon ≥ −0.2 dB vs plain Noise2Void); with data + P3 weights: all three recorded blocks reproduce |
| 16 | vst | `Data/Anscombe.cs` (`NoiseModel`, generalized Anscombe transform, Mäkitalo–Foi unbiased inverse) | `Brick=16-vst`: variance 0.98–1.05 after the transform at 5–200 photons, exact round trip, the unbiased inverse recovers the mean where the algebraic one is biased |
| 17 | calibration | `NoiseCalibration.cs` | `Brick=17-calibration`: from one synthetic frame, gain within 10% and the predicted noise variance within 10% across its intensity range, for 6 camera settings |
| 18 | pool-upsample | `Nn/Resample.cs` | `Brick=18-pool-upsample`: hand-checked values, adjoint identities, finite-difference gradients |
| 19 | unet | `Nn/DenoisingNet.UNet.cs`, `NetConfig.UNet`, `IO/WeightsFile.cs` (format 2) | `Brick=19-unet`: whole-network finite-difference gradients (4 shapes, incl. unaligned), unaligned = hand-padded bit for bit, format 2 round-trips both architectures, format 1 loads, learns through the trainer |
| 20 | unet-tiling | `TiledInference.cs` | `Brick=20-unet-tiling`: tiled U-Net inference equals whole-image inference for ragged sizes |
| 21 | arch-eval | `tools/experiments/p4_arch.sh`, `docs/RESULTS.md` | `Brick=21-arch-eval`; the default architecture follows the rule set before measuring; with data + weights: the recorded blocks reproduce |
| 22 | bench | `PerformanceBench.cs`, CLI `bench`, `tools/ci/check_bench.py` | `Brick=22-bench`; `dora-bio bench --quick --json` passes the schema-1 check |
| 23 | blocked-conv | `Nn/ConvLayer.cs` (register-blocked kernels) | `Brick=23-blocked-conv`: bit-identical to the reference row kernels; bar set before building: ≥ 1.5× faster on the 512² denoise and the training step |
| 24 | perf-budget | `tools/ci/perf_budget.json` | `Brick=24-perf-budget`; a full `bench` stays within the budgets |
| 25 | q1-recipe (quality pass) | `tools/experiments/q1_supervised.sh`, `docs/RESULTS.md` | Q1's recorded verdict follows its pre-registered rule (`<!-- q1-verdict -->`); with data + weights the blocks reproduce |
| 26 | tiff-stack-write | `IO/TiffCodec.cs` (`EncodeStack`), CLI `synth --stack`, `tools/ci/check_tiff_stack.py` | `Brick=26-tiff-stack-write` (3-page round trips, every sample type, ±Deflate); Pillow reads a written stack exactly (SKIPPED without Pillow; CI installs it) |
| 27 | q10-routing (quality pass) | `NoiseLevelRouter.cs`, `FmdBenchmark.ValidationPlan`, CLI `fmd-route` / `fmd-eval --weights-clean`, `tools/experiments/q10_routing.sh` | `Brick=27-q10-routing`; Q10's recorded verdict follows its pre-registered rule; with data + weights the blocks reproduce |
| 28 | cli-batch | `BatchDenoising.cs` (`FrameDenoiser`, `BatchDenoising`), `IO/TiffStack.cs` (`TiffStackReader`/`TiffStackWriter`), CLI `denoise` on stacks and folders | `Brick=28-cli-batch`: streamed pages decode exactly; every prefix of a written stack is a valid TIFF; a stack page equals the frame on its own; no frame outlives the next (weak references); the CLI denoises a 3-page stack and a folder (pages = files, by Pillow), and 40 pages peak ≤ 160 MB above 3 |
| 29 | q13-conditioning (quality pass) | `Nn/DenoisingNet.Film.cs` (FiLM generator, effective parameters), `NoiseConditioning.cs`, `NetConfig.Conditioning`, weights format 3, CLI `train --condition`, `fmd-eval --validation-seed / --sigma-scale`, `tools/experiments/q13_conditioning.sh` | `Brick=29-q13-conditioning`: whole-network and tight chain-rule finite differences, untrained = plain bit for bit, exact tiling with the whole frame's condition; Q13's recorded verdict and supervised training's default `--condition` follow the pre-registered rule; with data + weights the blocks reproduce |
| 30 | ext-data (field validation) | `tools/fetch_ext.sh`, `tools/extract_zip.py`, `Data/ExtDatasets.cs`, CLI `ext-check` | `Brick=30-ext-data` (pairs match by name, never by position); every archive pinned (record, size, SHA-256); with data: every `.complete` matches its pin and `ext-check` finds the published counts, partners of equal size, and files that decode |

Gates 11–15 and 21 need the dataset (the reproductions in gates 15 and 21 also need the weights from `tools/experiments/`) (`tools/fetch_fmd.sh`, about 9 GB into the gitignored `data/fmd/`). Without it, their
data-dependent part reports SKIPPED (exit code 77) while the pins, unit tests and RESULTS.md checks still run. CI has
no data and must not download it.

Rules:
1. **Work on one brick at a time.** Put new code in the lowest brick it belongs to, and give it tests tagged
   `[Trait(Bricks.Trait, Bricks.X)]`. Iterate with `./bricks.sh --only N`.
2. **A brick is done when `./bricks.sh N` is green**, meaning gate N and every gate below it. Then commit
   (`brick N (<name>): …`) before starting the next brick. Never build on a red gate.
3. **Dependencies only point down.** A change to a lower brick re-opens every brick above it, so run the full
   `./bricks.sh` before pushing.
4. **To add a brick:** add a constant to `tests/DoraBio.Core.Tests/Bricks.cs` (and its `Known` set), add a `gate_N`
   with `NAMES`/`CHECKS` entries in `bricks.sh`, add a planted bug for it to `MUTATIONS` in `bricks.sh` (so the
   self-test proves the new gate can fail), and add a row to this table.
5. A gate must never pass vacuously. Test gates fail if they match zero tests, gate 0 fails on an untagged
   test class, and `./bricks.sh --self-test` (part of gate 9, so it runs in every full run and in CI) plants
   one bug per gate 0–7 and requires each to turn red. If code moves on and a planted bug no longer applies,
   the self-test fails and says so: update `MUTATIONS`.
6. **Documentation is part of the brick.** Every public member needs XML docs (`/// <summary>`, plus `<param>`
   for record parameters). The compiler enforces this for the core and CLI (gate 0) and `bricks.sh` enforces it
   for `godot/scripts/*.cs` (gate 7). Beyond the tags, comment the *why* and the math: formulas for numeric
   code, file-format primers for codecs, invariants (exact tiling, reproducible gradient sums) where they're
   relied on. GDScript uses `##` doc comments on exports and functions.
7. **Alternate building and improving.** From brick 25 on, every other brick is a quality pass: the top item of the
   quality backlog in `PATHWAY.md`, pre-registered there (candidate, bar, adoption rule) before anything is measured,
   chosen on validation data, recorded in `docs/RESULTS.md`, and gated so the shipped defaults follow the evidence.
   A missed bar still completes the pass; record it and move on.
8. **CI is the shared gate, and it's spent thoughtfully.** The full gate set (Godot gate 8 mandatory) runs on
   GitHub Actions for PRs into `main`, pushes to `main` and manual "Run workflow", never for docs-only changes.
   Gate 9 enforces that policy. On working branches the local `./bricks.sh` is the gate, so run it (with
   `GODOT=…`) before every push and never push red. Don't trigger CI runs that a local run already answers.
   If CI goes red, fixing it comes before anything else.

## Commands (all verified in a Linux container without a GPU)

```bash
cd dora-bio
./bricks.sh                                               # all gates in order (~2 min incl. self-test; GODOT=… for gate 8)
./bricks.sh --only 3                                      # iterate on one brick
./bricks.sh --self-test                                   # plant one bug per gate; each must turn red (~1 min)
tools/fetch_fmd.sh                                        # FMD dataset, ~9 GB, into data/fmd/ (needs drive.usercontent.google.com)
tools/fetch_ext.sh                                        # public paired sets (DeepBacs, CARE 2D), ~740 MB, into data/ext/ (needs zenodo.org)
dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll fmd-eval --weights W.bin --root data/fmd   # held-out test_mix
tools/experiments/p2_fmd.sh                               # retrain + evaluate every P2 model (~45 min CPU)
tools/experiments/p3_struct.sh                            # P3: plain vs structured blind spots (~50 min CPU)
tools/experiments/p3_vst.sh                               # P3: with and without variance stabilization (~16 min CPU)
tools/experiments/p4_arch.sh                              # P4: U-Net vs dilated network (~21 min CPU)
dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll fmd-noise --root data/fmd   # is the real noise pixel-independent?
dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll diagnose --in IMAGE          # the same question, without ground truth
dotnet tools/DoraBio.Cli/bin/Release/net8.0/dora-bio.dll bench --json bench.json   # timings (~1 min; --quick: seconds)
dotnet test DoraBio.sln                                   # all tests at once, ~5 s
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
   The register-blocked kernels (`ConvLayer.Blocked.cs`) must stay **bit-identical** to the reference row kernels
   (`ForwardReference`/`BackwardReference`): same terms, same order per output value, separate multiply and add.
   Never use a fused multiply-add there; it would change results and break the recorded FMD reproductions.
5. **Tiled inference must equal whole-image inference** (`TiledInference` margin = receptive radius,
   clipped at image edges). Covered by `TiledInferenceMatchesWholeImage`.
6. **Optimizer state persists** across online adaptation (`OnlineAdapter`). A fresh Adam at t = 1 is a sign-step
   (v1's bug). Adaptation never saves implicitly. The user clicks Save, and the UI shows "unsaved changes".
7. **Learning rate only decays** (×0.85 after `PlateauPatience` epochs without a new best). The best weights
   are snapshotted and restored at the end, and checkpointed to disk at each new best.
8. **Validation is held out by FOV** (`ImagePair.GroupId`). Frames of one field of view never straddle the split.
9. **Journey log lifecycle:** a session opens only when training starts. Every epoch is appended and saved.
   The session ends as `completed` or `failed: …`. Inference-only launches write nothing.
10. **Structured blind spots and variance stabilization are opt-in** (`--struct auto`, `--vst`): on FMD the first
    wins on noisy input and loses on clean input (and one network trained on mixed data loses on the independent
    images), and the second lost overall (docs/RESULTS.md, P3). Don't make either a default without new held-out
    evidence. Choose footprints and other settings on validation data, never on test_mix. Noise conditioning
    (`--condition noise`, Q13) is opt-in for the same reason: it lost to the plain U-Net on every seed.
    The same goes for the architecture: by P4's rule the dilated network is the default (the U-Net won supervised,
    +0.98 dB, but lost Noise2Void); gate 21 fails if `NetConfig.Default` disagrees with the recorded results.
11. **Never train on FMD's test FOV (19).** `FmdOptions.ExcludedFovs` defaults to it, the CLI doesn't expose it, and
    gate 12 proves test_mix is FOV 19 and absent from the training plan. Report results on test_mix only.
12. **Don't commit FMD or external data, or weights trained on them, without a license decision.** The data is CC BY-SA 4.0;
    data/ and runs/ are gitignored. Credit Zhang et al., CVPR 2019 wherever FMD numbers appear.
13. **Label synthetic things as synthetic.** The bundled sample and weights come from `SyntheticFluorescence`.
    Don't describe them as real microscope data, and don't claim real-data accuracy that hasn't been measured.
14. **Stacks stream; nothing holds a past frame.** `BatchDenoising` reads a page, denoises it, writes it, and lets it
    go; `FrameDenoiser` keeps the inference workspaces for its lifetime instead of reallocating them per frame. Gate 28
    checks both (weak references, and peak memory flat in stack length). Every frame, stack page or single image, goes
    through `FrameDenoiser.Denoise`, so they can't drift apart.
15. **A noise-conditioned network's condition is per frame, never per tile or patch.** `TiledInference` measures it once
    on the whole normalized frame and gives every tile the same value; training measures each training image once. A
    per-tile value would differ between tiles and break exact tiling (the same goes for anything global: percentile
    normalization, or channel attention that pools over the image). FiLM is folded into the convolution weights
    (`DenoisingNet.Film.cs`), so the kernels and their bit-identical blocked versions are untouched.

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

- **Weights** (`WeightsFile`): `"DORABIO\0"`, u16 format = 2 (3 for a noise-conditioned network), model version
  `"2.0.0-bio"`, provenance string, saved-UTC ticks, architecture byte (0 dilated, 1 U-Net), then features + layer
  count + dilations[] (dilated) or features + depth (U-Net), format 3 only: conditioning byte + u16 generator hidden
  units (16), parameter count, float32 LE[], CRC-32. Plain networks still write format 2, byte for byte as before. Format 1 (no architecture byte, always dilated:
  the bundled model) still loads. v1 JSON `.dat` files are rejected with a clear message.
- **Images out:** TIFF, uint8/uint16/float32, uncompressed or Deflate, single pages or multi-page stacks
  (`TiffCodec.EncodeStack`, or `TiffStackWriter` page by page: the file is a valid TIFF after every page).
- **Images in:** TIFF (8/16/32-bit int/float; none/Deflate/PackBits; predictor 2; LE/BE; strips; multi-page)
  and PNG (8/16-bit gray/RGB(A), all filters). Not supported, each with a clear error: LZW, tiled TIFF,
  BigTIFF, palettes, interlaced PNG, **HDF5**.
- **Datasets** (`DatasetLoader`): `noisy/`+`clean/`; FMD (`FmdDataset`: the dataset root, one category, or
  `test_mix`, checked against the real download by `dora-bio fmd-check`); or a flat folder (Noise2Void only).

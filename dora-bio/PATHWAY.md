# DORA v2.0-bio development pathway: P0 → P10

The plan from today's state to a released, real-data-validated tool. It extends the
[brick-by-brick workflow](CLAUDE.md#brick-by-brick-workflow-how-this-project-is-built-follow-it): **each phase is a
group of bricks, and each brick adds a gate to `bricks.sh`.** Phases are ordered so the biggest unknowns are
settled first, and so every phase builds only on phases that are already green.

## Status

| Phase | Theme | Risk it removes | Status |
|---|---|---|---|
| **P0** | Foundation | "does any of it work?" | ✅ done (`41e2c6f`): bricks 0–8 green, docs enforced |
| **P1** | Continuous gates (CI) | regressions slipping in between sessions | ✅ done: bricks 9–10, [CI run #1](https://github.com/Andraszao/project-dora-ml/actions/runs/45617118232) green |
| **P2** | Real-data truth (FMD) | performance on real microscopes is unmeasured | ✅ done: bricks 11–13, [results](docs/RESULTS.md) on 48 held-out FMD images |
| **P3** | Noise-model fidelity | Noise2Void's assumptions failing on real cameras (measured in P2 on widefield) | ✅ done: bricks 14–17, [results](docs/RESULTS.md#p3-correlated-noise-and-structured-blind-spots) |
| **P4** | Model capacity (U-Net) | **the small CNN being the ceiling** | ⏭ next |
| **P5** | Performance | slowness on large frames and stacks | planned |
| **P6** | Stacks, time-lapse, batch | single 2-D frames only | planned |
| **P7** | Analysis UX | can see but can't measure or export | planned |
| **P8** | Model and session management | adaptation lost on restart; no way back to a known-good model | planned |
| **P9** | Format coverage | "my file won't open" (LZW, OME-TIFF, HDF5) | planned |
| **P10** | Release | only runs from the editor or CLI | planned |

## Phase protocol (applies to every phase)

1. **Open:** add the phase's gates to `bricks.sh` first, as failing tests or explicit SKIPs with a reason, so
   "done" is defined before any code is written. Mark the phase "in progress" in the table above.
2. **Build:** one brick at a time. Iterate with `./bricks.sh --only N`, finish with `./bricks.sh N`, and commit
   `brick N (<name>): …` only when green. Docs are part of the brick (rule 6 in `CLAUDE.md`).
3. **Close:** full `./bricks.sh` green (with `GODOT=…`), README "What's verified" updated with **measured**
   numbers and the command that produced them, this table updated, then a commit `P<n> done: …`.
4. **Never start P(n+1) while P(n) is red.** Lower phases may be reopened. A reopened phase re-runs every gate above it.

Brick numbers below continue from P0's 0–8. They're indicative; renumber if a phase needs more or fewer bricks.

---

## P0: Foundation ✅

Done. Pure-C# core, 70 tests (finite-difference gradient checks, codec fixtures from independent encoders, exact
tiling, reproducible training), CLI, Godot 4.5 .NET shell, headless smoke test, compiler-enforced docs.
**Gate:** `./bricks.sh` bricks 0–8 green. **Baseline numbers** (4 vCPU, no GPU): 512×384 denoise about 0.5 s;
training step about 0.3 s (batch 16 × 64 px); held-out *synthetic* eval: noisy 20.89 dB → bilateral 28.43 → DORA 30.21.

## P1: Continuous gates (CI)

**Why now:** every later phase changes behavior. CI turns "never build on red" from a habit into a hard rule,
and catches regressions made outside these sessions.

✅ **Done.** What was built (local gate → remote proof):

| Brick | Delivers | Gate |
|---|---|---|
| 9 `ci-core` | `.github/workflows/dora-bio.yml` job `gates`: runs the **full** `./bricks.sh` (read-only token, timeout, superseded runs cancelled) under the spend policy below. `./bricks.sh --self-test` plants one known bug per gate 0–7 in scratch copies and requires each gate to turn red, after an unmutated control copy passes | `tools/ci/check_workflow.py … core` (itself shown to reject 3 broken workflows) + self-test: all 9 planted bugs caught |
| 10 `ci-godot` | job `godot-smoke` (`needs: gates`): official Godot 4.5.1 .NET build, verified against the release's SHA-512, cached; runs gate 8 with `REQUIRE_GODOT=1` so it can never pass by skipping | `check_workflow.py … godot`: Godot version = Godot.NET.Sdk version, checksum pinned and verified, gate 8 mandatory |

**Exit, as met:** CI run #1 green on both jobs (gates job 1 min 47 s including the self-test; Godot job 39 s, including the Godot download). The plan said to prove red-ability with a
broken commit on a throwaway branch. That was replaced by the self-test, which proves on **every** CI run that
each gate can fail, without broken commits, and without pushing to branches other than the working branch.
**Spend policy (enforced by gate 9):** CI runs on PRs into `main`, pushes to `main` and manual "Run workflow",
never for docs-only changes (`*.md`), and cancels superseded runs. On working branches the local `./bricks.sh` is
the gate. **Measured cost:** run #1 used 150 s of runner time, which is 3 billable minutes on a private repo since
each job rounds up. GitHub reported **0 billable minutes** because the repo is public and standard runners are
free there.

## P2: Real-data truth (FMD)

**Why now:** this is the biggest open question. All quality numbers so far are on synthetic data, and the FMD
loader matches the published layout but has never seen the real download.

| Brick | Delivers | Gate |
|---|---|---|
| 11 `fmd-fetch` | `tools/fetch_fmd.sh` (download + checksum), or written instructions if the host blocks scripted download | script verifies checksums; SKIPPED with a reason if the data is absent |
| 12 `fmd-loader` | `DatasetLoader` run against the real FMD tree, fixed if reality differs from the docs | loader test on the real layout (SKIPPED without data) |
| 13 `fmd-eval` | `docs/RESULTS.md`: per microscope type (confocal / two-photon / widefield), noisy vs bilateral vs DORA PSNR/SSIM on **held-out FOVs**, for (a) the synthetic model, (b) N2V trained on raw only, (c) supervised with avg50 ground truth | numbers recorded with the commit hash and exact commands |

**Exit:** README limitations rewritten from measured results, good or bad.
**Needs your decision:** check FMD's license before committing any frames or FMD-trained weights. If
redistribution isn't allowed, the bundled model stays synthetic and the doc says how to train your own.

**Outcome:** done. The license is CC BY-SA 4.0, so no FMD frames or FMD-trained weights are committed and the bundled
model stays synthetic. The measured results, including tests of v1's augmentation and graded-noise ideas, are in
[`docs/RESULTS.md`](docs/RESULTS.md). Supervised DORA beats the bilateral filter on all three microscope types.
Noise2Void fails on widefield because its noise is correlated, which reshaped P3.

## P3: Noise-model fidelity

**Why:** Noise2Void assumes pixel-independent, zero-mean noise. Real cameras add gain and offset structure
(Poisson–Gaussian) and sometimes spatially correlated noise (sCMOS read-out patterns), and those break it quietly.

**P2 measured it** ([RESULTS.md](docs/RESULTS.md#why-noise2void-fails-on-widefield-the-noise-is-correlated)): FMD's
widefield noise is 0.6 correlated between neighbouring pixels, and Noise2Void trained on it scores 27.78 dB against
32.28 dB for supervised training of the same network. So the diagnostic moves first, and a fix joins the phase.
P2 also found that every model makes already-clean inputs (16-frame averages) worse, so every brick here is judged
at all five FMD noise levels, not only on raw frames.

| Brick | Delivers | Gate |
|---|---|---|
| 14 `n2v-diagnostic` | noise autocorrelation estimated from the noisy image alone (no ground truth); warn in the CLI and UI when neighbouring pixels' noise is correlated | flags correlated synthetic noise, stays quiet on i.i.d. noise; with data: same verdict as the ground-truth measurement on every raw test_mix image (`dora-bio fmd-diagnose`) |
| 15 `structured-mask` | blind spot that also hides the correlated neighbours (structured Noise2Void masking, after Broaddus et al. 2020), applied automatically to the training images brick 14 flags, with its size picked on validation data (never on test_mix) | masking tests; **bar set before building:** on FMD test_mix (raw), +1 dB or more over plain Noise2Void on widefield, and no more than 0.2 dB lost on confocal or on two-photon. Gate 15 checks the bar on the recorded results and, with data and weights, that they reproduce |
| 15 follow-up (bar set 2026-11-25, after the first design missed) | the first design (one network, structured blind spots on the flagged images) met the widefield bar (+3.41 dB) but lost 1.15 dB on confocal. Second design: **two networks routed by the noise check**: plain Noise2Void trained on the images the check calls independent, structured (diamond-2, the footprint validation chose) on the ones it calls correlated; at inference each image goes to the network its own noise check picks | **the same bar, unchanged:** vs the one plain Noise2Void network, widefield +1 dB or more, confocal and two-photon −0.2 dB at worst. Disclosed in RESULTS.md as a second look at test_mix |
| 16 `vst` | generalized Anscombe transform (Poisson–Gaussian) with exact unbiased inverse, as an optional preprocessing step | after the transform, residual variance ≈ 1 across intensity bins on synthetic data; round-trip error ≤ 1e-4 |
| 17 `calibration` | estimate the noise model from one image: variance = gain · intensity + intercept, which is all the transform needs | on synthetic frames with several camera settings: gain within 10%, and the predicted variance within 10% across the image's intensity range. *(Restated before building: one image can't separate the camera offset from read noise, since both only enter the intercept as read² − gain · offset. The original "gain, offset and read noise each within 10%" was not measurable.)* |

**Exit:** FMD eval at every noise level with and without the structured mask and the transform recorded. Each
becomes a default only if it wins on held-out data. *For the transform, "wins" was set before running it: compared
with the same Noise2Void design without it (plain, and the routed pair), the all-48 test_mix PSNR must rise at raw
and drop by no more than 0.2 dB at any lower noise level.*

**Outcome:** done. The noise check matches ground truth on every FMD test image. Structured blind spots, as two
networks routed by the check, met their bar (widefield +3.79 dB, confocal +0.32, two-photon +0.61) and bring
Noise2Void within 0.8 dB of supervised training. They stay opt-in because they lose on cleaner inputs. The
transform missed its bar (−0.25 to −0.28 dB at raw): one-image calibration works on widefield but not on dim 8-bit
confocal, so it stays opt-in too. P3 also found and fixed a validation set that covered one category only.

## P4: Model capacity (U-Net)

**Why:** the 26k-parameter dilated CNN is fast and exact to tile, but it may be the quality ceiling. U-Nets are
the standard Noise2Void backbone.

| Brick | Delivers | Gate |
|---|---|---|
| 18 `pool-upsample` | 2×2 pooling and upsampling layers with forward/backward | finite-difference gradient checks (same harness as brick 1) |
| 19 `unet` | U-Net with skip connections; architecture tag in the weights file (format v2) that still loads v1-format files | gradient check of the whole net; old weight files still load |
| 20 `unet-tiling` | margins and padding-to-multiple-of-2ᵈ rules for exact tiling | `TiledInferenceMatchesWholeImage` passes for U-Net |
| 21 `arch-eval` | DnCNN vs U-Net on held-out FMD: quality vs speed | numbers in `docs/RESULTS.md` |

**Exit:** the default architecture is whichever wins on held-out real data, with the result documented.

## P5: Performance

**Why:** P6 (stacks) multiplies cost by the number of frames, so speed is measured and protected before that.

| Brick | Delivers | Gate |
|---|---|---|
| 22 `bench` | `dora-bio bench`: denoise and train-step timings as JSON | runs in gate 6; JSON schema validated |
| 23 `intra-image-parallel` | parallelize within a single large tile (by output channel); optional im2col + blocked matrix multiply | bit-exact against the current path on the tiling and gradient tests; faster in `bench` |
| 24 `perf-budget` | loose regression budgets in CI (e.g. 512² denoise ≤ 1.0 s, train step ≤ 0.5 s on the CI runner) | fails if over budget |

**Later, optional:** GPU inference via Godot compute shaders. Its gate would be numerical agreement with the CPU path.

## P6: Stacks, time-lapse and batch

| Brick | Delivers | Gate |
|---|---|---|
| 25 `tiff-stack-write` | multi-page TIFF writer (the reader already handles pages) | round-trip 3-page stacks in every sample type |
| 26 `cli-batch` | `denoise` over stacks and folders, streaming one frame at a time with bounded memory | gate 6 denoises a 3-page synthetic stack; peak memory checked |
| 27 `ui-stack` | page/frame slider in the UI, denoising the current frame or the whole stack (cancellable) | smoke test loads a stack, steps pages, denoises one frame |

## P7: Analysis UX

| Brick | Delivers | Gate |
|---|---|---|
| 28 `sync-view` | synchronized zoom and pan across the three panels | smoke test: same view transform on all panels after a zoom |
| 29 `measure` | line profile and ROI mean/σ on raw, baseline and DORA | profile values equal the image values exactly (test) |
| 30 `export` | "Export denoised…" writes float32 TIFF in original units | the UI export is byte-identical to `dora-bio denoise` with the same model and input |
| 31 `compare` | before/after wipe slider and colormaps (display only) | screenshot review recorded; data path untouched (gate 8) |

## P8: Model and session management

| Brick | Delivers | Gate |
|---|---|---|
| 32 `optimizer-checkpoint` | optional Adam-state block in weights files, so adaptation survives restarts | save → load → continue adapting is **bit-identical** to never stopping |
| 33 `model-picker` | bundled / user / history models in the UI; reset to bundled; undo last adaptation | smoke test: adapt → undo restores the exact parameters |
| 34 `journey-view` | journey log viewer (sessions, epochs, milestones) in the UI | renders a fixture journey file; malformed files fail with a message |

## P9: Format coverage

| Brick | Delivers | Gate |
|---|---|---|
| 35 `tiff-lzw` | LZW decode (the most common "won't open" case today) | libtiff/Pillow LZW fixtures decode exactly |
| 36 `tiff-tiled-big` | tiled TIFF and BigTIFF | fixtures from libtiff |
| 37 `ome-metadata` | OME-TIFF pixel size and channel names read, and carried into outputs | metadata round-trip test |
| 38 `hdf5` | HDF5 reader as a **separate** optional project (`DoraBio.Hdf5`) so the core stays dependency-free | fixture test; gate 0 asserts the core still has zero `PackageReference`s |

## P10: Release

| Brick | Delivers | Gate |
|---|---|---|
| 39 `export-presets` | Godot .NET export presets for Linux, Windows and macOS | exported Linux build runs the smoke scenario headless in CI |
| 40 `release-ci` | tag → CI builds, attaches binaries and the CLI to a **draft** GitHub release | draft release created on a test tag |
| 41 `user-docs` | user guide (install, first denoise, adapt, train, evaluate), changelog, data and weights license notes | docs link-checked; README quick start verified on the exported build |

**Exit:** v2.0.0 tagged. **Needs your decisions:** app signing and notarization (Apple and Windows accounts) and
the release license text.

---

## Parking lot (deliberately not on the path)

- **Web export:** not possible for C# Godot projects in 4.5. Revisit only if Godot adds it.
- **3-D (volumetric) denoising and GPU training:** large scope. Reconsider after P5 measurements.
- **v1 (repo root, MNIST):** frozen as the educational original. Its bugs are documented in the root `CLAUDE.md`, not fixed.
- **History rewrite into per-brick commits:** possible, but it needs a force-push and an explicit request.

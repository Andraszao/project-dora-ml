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
| **P4** | Model capacity (U-Net) | the small CNN being the ceiling | ✅ done: bricks 18–21, [results](docs/RESULTS.md#p4-a-u-net-against-the-dilated-network) |
| **P5** | Performance | slowness on large frames and stacks | ✅ done: bricks 22–24 (2.1× faster denoise, bit-identical) |
| **P6** | Stacks, time-lapse, batch | **single 2-D frames only** | 🚧 in progress: 25 (quality, Q1), 26 `tiff-stack-write`, 27 (quality, Q10), 28 `cli-batch`, 29 (quality, Q13) done; next `ui-stack`, then 31 (quality, Q12) |
| **P7** | Analysis UX | can see but can't measure or export | planned |
| **P8** | Model and session management | adaptation lost on restart; no way back to a known-good model | planned |
| **P9** | Format coverage | "my file won't open" (LZW, OME-TIFF, HDF5) | planned |
| **P10** | Release | only runs from the editor or CLI | planned |
| **FV** | Field validation (alongside P6–P10) | nobody outside this repo has used DORA or compared it with their current tool | 🚧 in progress: 30 `ext-data`, 31 `ext-bench` (Q18: supervised at parity with CARE, Noise2Void 0.7 dB behind), 32 `quant-fidelity` done; the pilot waits for Q12 |
| **SOTA** | [Road to state of the art](#road-to-state-of-the-art-stages-15-agreed-2026-11-26) (stages 1–5, bricks 33–49+) | DORA trails the standard self-supervised tool and has no broad scoreboard | 🚧 in progress: 33 (Q12: not adopted, blocked only by the U-Net's early stop on FMD) 34 `ext-data-2` (benchmark data pinned, stack readers) and 35 (Q2: not adopted by 0.0006 dB on FMD; longer Noise2Void training learns correlated noise) done; 36 (S1 + S2: roofline, 512-bit kernels, 1.3–1.6×) and 37 (Q19: 33.24 dB on FMD without ground truth; not adopted on CARE 2D's floor, then adopted under Q20, so `train --mode n2v` now routes) done; 38 (scoreboard: competitive with CAREamics N2V/N2V2 on all five sets, +1.76 dB on FMD) done; 39 (Noise2Noise from neighbouring frames + the 3-frame spatiotemporal blind spot, built and gated) done; next 40 (Track A: the time-lapse modes on two real two-photon movies, [pre-registered](#brick-40-pre-registration-time-lapse-modes-on-real-data-track-a); its skip-centre arm is built first), then 41–42 (Track B: N2V2's pieces and their ablation) |
| **SYS** | [Systems track](#systems-track-speed-big-data-and-the-viewport-as-an-instrument-proposed-2026-11-26) (S1–S10, alongside stages 1–5) | CPU kernels estimated at about a tenth of peak; stacks bigger than memory can't be browsed; the viewport shows results but measures nothing | 🚧 S1 + S2 done (brick 36: compute-bound everywhere, 512-bit kernels 1.3–1.6×, bit-identical); S3–S9 with stage 5; S10 (GPU inference, forward only, Godot app) approved |

## Phase protocol (applies to every phase)

1. **Open:** add the phase's gates to `bricks.sh` first, as failing tests or explicit SKIPs with a reason, so
   "done" is defined before any code is written. Mark the phase "in progress" in the table above.
2. **Build:** one brick at a time. Iterate with `./bricks.sh --only N`, finish with `./bricks.sh N`, and commit
   `brick N (<name>): …` only when green. Docs are part of the brick (rule 6 in `CLAUDE.md`).
3. **Close:** full `./bricks.sh` green (with `GODOT=…`), README "What's verified" updated with **measured**
   numbers and the command that produced them, this table updated, then a commit `P<n> done: …`.
4. **Never start P(n+1) while P(n) is red.** Lower phases may be reopened. A reopened phase re-runs every gate above it.
5. **Alternate building and improving (from brick 25 on).** After every build brick, the next brick is a **quality
   pass**: the most valuable open item in the [quality backlog](#quality-backlog-quality-passes), an improvement to
   DORA's results or a revision of an earlier decision. So the pathway reads build, quality, build, quality, …

### Quality-pass protocol

1. **Pick** the highest-value open backlog item (re-rank first if the last brick taught something new).
2. **Pre-register before measuring:** write the candidate change, the bar and the adoption rule into the item's row,
   and commit them with the pass. The rule says what ships if the bar is met *and* if it isn't.
3. **Measure honestly:** choose settings on validation data (held-out training FOVs), score test_mix once for the
   final comparison, and say so when a design was revised after seeing test results.
4. **Record** the numbers in `docs/RESULTS.md` as verbatim blocks. The pass's gate re-runs them when the data and
   weights are present, and checks that the shipped defaults and the recorded verdict follow the rule.
5. **A miss still completes the pass** ("measured, not adopted"), so the pathway never stalls on a hypothesis that
   didn't pan out. Whatever was learned goes back into the backlog.

Brick numbers below continue from P0's 0–8. Up to brick 24 they were fixed in advance. From P6 on, build bricks are
numbered as they're laid, with quality passes between them.

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

**Exit:** the default architecture is whichever wins on held-out real data, with the result documented. *Rule set
before building (2026-11-25): the U-Net becomes the default only if, at the same number of training steps (P3's
budget and seed), it beats the dilated CNN on all-48 test_mix PSNR (raw) for supervised AND plain Noise2Void
training, and denoises a 512×512 frame at most 3× slower. Gate 21 checks that the default follows the rule.*

**Outcome:** done. Supervised, the U-Net is the best DORA model so far: 34.73 dB against 33.75, better on all three
microscope types, at 1.36× the inference time. With Noise2Void it lost 0.47 dB (its run stopped early at epoch 10).
By the rule, **the dilated network stays the default**; `--arch unet` is there for supervised training. A per-mode
default would be a new rule chosen after seeing the data, so it was left for a decision.

## P5: Performance

**Why:** P6 (stacks) multiplies cost by the number of frames, so speed is measured and protected before that.

| Brick | Delivers | Gate |
|---|---|---|
| 22 `bench` | `dora-bio bench`: denoise and train-step timings as JSON (both architectures, both training modes) | runs in gate 22; JSON schema validated (`tools/ci/check_bench.py`) |
| 23 `blocked-conv` | register-blocked direct convolution: 8 output channels × 8 pixels of accumulators in registers, so each input value is loaded once per tap instead of once per output channel (forward, weight gradients, input gradients). Tiles are already parallel, so this attacks the real limit: memory traffic, at about 8% of the CPU's arithmetic peak | **bit-identical** to the current kernels, kept as a test reference (same per-pixel summation order, no fused multiply-add); **bar set before building:** at least 1.5× faster on the 512² denoise and on the training step, median of `bench` on the same machine |
| 24 `perf-budget` | loose time budgets (`tools/ci/perf_budget.json`), checked locally and in CI | fails if any `bench` timing is over budget |

**Later, optional:** GPU inference via Godot compute shaders. Its gate would be numerical agreement with the CPU path.

**Outcome:** done. Baseline (`dora-bio bench`, 4-core container): 512² denoise 0.573 s (dilated) / 0.701 s (U-Net),
training step 0.20–0.24 s. The register-blocked kernels are bit-identical (every recorded FMD result from P2–P4
reproduces exactly) and met the 1.5× bar: denoise 0.27 s (2.1×) / 0.30 s (2.4×), training step 0.12–0.14 s
(1.5–1.8×). A first attempt reached only 1.45× on the denoise; replacing the scalar row tail with an overlapping
last SIMD strip got it to 2.1×. The budget check times the blocked and reference kernels interleaved, because on
this shared machine back-to-back timings drifted by up to 25% (disabling tiered compilation didn't help, so it
wasn't kept).

## P6: Stacks, time-lapse and batch

| Brick | Delivers | Gate |
|---|---|---|
| 26 `tiff-stack-write` ✅ | multi-page TIFF writer (the reader already handles pages) | round-trip 3-page stacks in every sample type; Pillow reads a written stack exactly |
| 28 `cli-batch` ✅ | `denoise` over stacks and folders, streaming one frame at a time with bounded memory | gate 28: a 3-page stack and a folder through the CLI (the stack's pages equal the folder's files); no frame outlives the next; 40 pages of 512² peak at most 160 MB above 3 (measured: 182 → 269 MB; before the fix, workspaces reallocated per frame and 160 pages peaked at 850 MB) |
| `ui-stack` | page/frame slider in the UI, denoising the current frame or the whole stack (cancellable) | smoke test loads a stack, steps pages, denoises one frame |
| `train-stream` | training data streamed as a working set of frames read from disk (seeded, so runs stay reproducible), instead of every frame in RAM (Q1's 2,280 FMD frames are 2.3 GB of noisy pixels) | peak memory flat in dataset size; a streamed run equals an in-memory run with the same seed and schedule; needed before Q12 adds data |

## P7: Analysis UX

| Brick | Delivers | Gate |
|---|---|---|
| `sync-view` | synchronized zoom and pan across the three panels | smoke test: same view transform on all panels after a zoom |
| `viewport-chunks` | denoise what's on screen first, as game engines stream textures: one cheap whole-frame pass (normalization, and the noise condition for conditioned models), then chunks on the tile grid, nearest the viewport centre first, kept in a cache that the model or settings invalidate; the rest fills in the background | chunks finished in any order stitch to exactly the whole-image result (tiling is exact); cancelling stops within one chunk; smoke test pans and sees visible chunks finish first |
| `measure` | line profile and ROI mean/σ on raw, baseline and DORA | profile values equal the image values exactly (test) |
| `export` | "Export denoised…" writes float32 TIFF in original units | the UI export is byte-identical to `dora-bio denoise` with the same model and input |
| `compare` | before/after wipe slider and colormaps (display only) | screenshot review recorded; data path untouched (gate 8) |

## P8: Model and session management

| Brick | Delivers | Gate |
|---|---|---|
| `optimizer-checkpoint` | optional Adam-state block in weights files, so adaptation survives restarts | save → load → continue adapting is **bit-identical** to never stopping |
| `model-picker` | bundled / user / history models in the UI; reset to bundled; undo last adaptation | smoke test: adapt → undo restores the exact parameters |
| `journey-view` | journey log viewer (sessions, epochs, milestones) in the UI | renders a fixture journey file; malformed files fail with a message |

## P9: Format coverage

| Brick | Delivers | Gate |
|---|---|---|
| `tiff-lzw` | LZW decode (the most common "won't open" case today) | libtiff/Pillow LZW fixtures decode exactly |
| `tiff-tiled-big` | tiled TIFF and BigTIFF, and huge single frames read strip by strip and denoised one row of tiles at a time (a 30k × 30k mosaic is 3.6 GB as floats) | fixtures from libtiff; peak memory flat in frame size |
| `ome-metadata` | OME-TIFF pixel size and channel names read, and carried into outputs | metadata round-trip test |
| `hdf5` | HDF5 reader as a **separate** optional project (`DoraBio.Hdf5`) so the core stays dependency-free | fixture test; gate 0 asserts the core still has zero `PackageReference`s |

## P10: Release

| Brick | Delivers | Gate |
|---|---|---|
| `export-presets` | Godot .NET export presets for Linux, Windows and macOS | exported Linux build runs the smoke scenario headless in CI |
| `release-ci` | tag → CI builds, attaches binaries and the CLI to a **draft** GitHub release | draft release created on a test tag |
| `user-docs` | user guide (install, first denoise, adapt, train, evaluate), changelog, data and weights license notes | docs link-checked; README quick start verified on the exported build |

**Exit:** v2.0.0 tagged. **Needs your decisions:** app signing and notarization (Apple and Windows accounts) and
the release license text.

## FV: Field validation (runs alongside P6–P10)

Whether anyone would choose DORA can only be measured outside this repo: on other people's data, against the tools
they already use, and by asking them. Two sources. **Public datasets** that come with low- and high-SNR pairs, and in
some cases with the CARE or Noise2Void models trained on them (Zenodo, CC BY 4.0, checked 2026-11-26): DeepBacs
*E. coli* MreB (record 6460867, 0.22 GB) and nucleoid (5551112), DeepBacs *B. subtilis* (5551135), the ZeroCostDL4Mic
CARE 2D example (3713330), and the AI4Life 2025 denoising challenge sets (Tribolium 15615943, Planaria 15615592).
Also the OME sample files of native vendor formats (downloads.openmicroscopy.org). **Labs**, reached through
forum.image.sc, light-microscopy core facilities (BINA, ABRF, Euro-BioImaging) and first authors of recent live-cell
preprints. They give the answer that matters: which tool they would actually use, and why.

| Brick | Delivers | Gate |
|---|---|---|
| 30 `ext-data` ✅ | `tools/fetch_ext.sh`: the Zenodo records above, pinned (record id, file, size, SHA-256, licence), into the gitignored `data/ext/`; loaders for their low/high-SNR layouts | pins verified; with data: counts and pairing checked by `ext-check`. Measured on the four sets fetched so far (MreB, nucleoid, *B. subtilis*, CARE 2D): DORA's own reader opens all 194 TIFFs, including Micro-Manager OME-TIFF stacks and big-endian 16-bit; the high-SNR partners are 3.3–10.8× brighter than their low-SNR images, so outputs must be brightness-matched before scoring |
| 31 `ext-bench` (quality pass Q18) ✅ | DORA (self-supervised on the noisy images alone, and supervised on each set's own training split) against the bilateral filter, against the CARE models those records ship, and against Noise2Void and CARE run the way a Fiji or napari user would (CAREamics, CPU, default settings) | recorded blocks reproduce; the baselines' settings and versions are recorded with their numbers |
| 32 `quant-fidelity` ✅ | what scientists measure, beyond PSNR: intensity linearity (slope and offset of denoised against reference in ROIs), mean-intensity bias, and invented or erased structure (residuals at puncta and dim filaments in the high-SNR reference) | metrics exact on synthetic ground truth; recorded for 18 model/dataset pairs ([results](docs/RESULTS.md#fidelity-what-scientists-measure-brick-32)). DORA's supervised U-Net keeps ratios on FMD (slope 1.02, 3.9% error); DORA's Noise2Void compresses contrast (MreB slope 0.75, ratio error worse than the raw frame's) |
| `vendor-corpus` | the OME sample files (.nd2, .czi, .lif, .oib, OME-TIFF) as a test corpus: every file either opens or fails with a message that names its format and the conversion route | a recorded table of every sample file and its outcome, and no crash on any of them. P9's reader order follows what this shows |
| `pilot` | the outreach kit (an image.sc post, emails to core facilities and first authors, the data spec below), 3–5 consented lab datasets, a side-by-side report for each against the lab's current tool, and their answer to "which would you use?" | recorded per lab, with consent; lab data is never committed |

**Data spec for labs.**
- **Unprocessed 16-bit raw counts.** No 8-bit exports, no auto-contrast, deconvolution or background subtraction.
- **Low-photon data.** Live-cell time-lapses or deep z-stacks taken at reduced laser power, not bright fixed slides.
- **A high-SNR reference of the same sample, where they have one.** It shows whether DORA invents puncta or erases dim filaments.
- **Their downstream measurement** (Cellpose segmentation, TrackMate tracking, or intensity quantification), because
  that, not PSNR, decides whether denoising helped.
- **Practicalities:** pixel size and exposure (and camera gain and offset if known), a download link rather than
  attachments, and consent for what may be published (aggregate numbers only, unless they say otherwise).

## Road to state of the art (stages 1–5, agreed 2026-11-26)

**The goal:** at or above the best open tools on the benchmarks the field uses, on the user's CPU, with fidelity
(brick 32) reported next to PSNR. DORA's bet is a compact model adapted to the user's own sensor and sample in minutes,
not a giant zero-shot model; GPU time, if used at all, only pretrains its starting weights (stage 4). Each stage keeps
the build/quality cadence, and each quality pass is pre-registered as usual.

**Decisions (owner, 2026-11-26).**
- **GPU training is allowed as an optional, offline step.** A PyTorch script may pretrain base weights and export them in
  DORA's format. Inference and per-sample adaptation stay pure C# on the CPU, and end users never need Python.
- **The shipped model is trained only on synthetic data and CC BY 4.0 / CC0 datasets.** `ATTRIBUTIONS.md` is generated
  from the pinned manifest (`tools/ext_manifest.json`, since brick 34). FMD (CC BY-SA) is never used for training a shipped model; it stays
  a zero-shot evaluation benchmark. This settles the FMD licence question.
- **Amended (owner, 2026-11-26, S10): an optional GPU path for forward inference in the Godot app.** Only inference,
  only in the interactive app, opt-in; training, per-sample adaptation and every `dora-bio` CLI run stay on the CPU. The
  CPU engine is the reference: the GPU path must match it (max pixel difference < 1e-4 on the normalized scale and
  PSNR within 0.01 dB) and falls back to the CPU when no Vulkan RenderingDevice exists.
  **Renderer dependency (noted 2026-11-27):** the Godot app runs the GL Compatibility renderer, which has no
  RenderingDevice and only half-precision float render targets. S10 therefore either moves the app to Forward+ or
  Mobile (smoke test and screenshot re-checked under lavapipe) or works in Compatibility fragment shaders on half-float
  targets, which would struggle to meet the 1e-4 bound; S10's design brick decides, against that bound.
- **Challenge submissions wait for stage 4 (owner, 2026-11-27).** The 2024 AI4Life leaderboards (W2S, SUPPORT, JUMP,
  Hagen et al.; late submissions reportedly open, per the owner) get a DORA container only once stage 4's pretrained
  model exists; the container may run zero-shot or adapt per stack on the CPU within the runtime limit. Submitting
  needs the owner's Grand Challenge account.

**Benchmark data (checked on Zenodo, 2026-11-26; all CC BY 4.0).** The 2024 AI4Life challenge subsets: W2S (10925783,
0.19 GB), SUPPORT voltage imaging (10925939, 2.1 GB) and JUMP Cell Painting (10912386, 2.4 GB); the organisers describe
it as an *unsupervised* challenge (confirmed in brick 34: models train on single noisy images; four leaderboards,
structured noise from Hagen et al. and unstructured noise from W2S and JUMP, plus SUPPORT; its references stay with
the organisers, so the challenge sets come noisy-only). The 2025 AI4Life *supervised* challenge
training sets: Tribolium (15615943, 4.5 GB) and Planaria (15615592, 16.3 GB). BioSR+ (7115540, 9.3 GB, per-structure
archives). DeepCAD-RT two-photon time-lapses: mouse brain neutrophils (6296569, 12.7 GB, motile), zebrafish brain
regions (6293696) and optic tectum neurons (6339707). Aydin examples (18686251, 4 GB). Not usable for training: the
DeepCAD-RT *demo* data (GPL-licensed), and the Convallaria / mouse actin / mouse nuclei trio until its licence is found.
Dropped in brick 34: the Aydin examples. The record says CC BY 4.0, but its own file list gives other terms for many
files (CC BY-SA 4.0 for the OpenCell volumes, "PI-confirmed redistribution" for others), and its TIFFs are
zstd-compressed, which DORA doesn't decode.

| # | Kind | Brick | Gate or bar |
|---|---|---|---|
| **Stage 1** | | **Close the self-supervised gap (CPU only)** | |
| 33 | quality | **Q12:** a Noise2Void U-Net at CAREamics' size (about 0.5M parameters) against today's 26k dilated network, 0.2% against 2% masking, a longer budget; chosen on held-out training images | within 0.3 dB of CAREamics Noise2Void on the Q18 sets **and** MreB slope ≥ 0.9 |
| 34 ✅ | build | `ext-data-2`: the benchmark data above in `tools/fetch_ext.sh` (the large archives behind `--full`), loaders for 3D and time-lapse stacks, and the attribution manifest | pins, counts and licences recorded; gates stay fast without `--full`. **Done:** `tools/ext_manifest.json` holds all 17 datasets (22 files, 58 GB; Zenodo MD5 + size for each, SHA-256 pinned for every file downloaded); `ATTRIBUTIONS.md` is generated from it and gate-checked; the licence rule (only CC BY / CC0 may train a shipped model) is checked by script and test. Reading the real files changed the plan's picture: the 2025 sets are **training patches** (Tribolium: 15,500 pairs of 16 × 64²; Planaria: 17,900 such pairs plus 45 full volumes), not whole volumes; BioSR+ is **raw SIM** (9 pattern frames per file, MRC) at 8–15 signal levels; two of the three DeepCAD-RT records are **BigTIFF**; ImageJ labels W2S's fields of view "slices" and MDC25's z-slices "channels". So DORA now reads BigTIFF and MRC, and a dataset declares its axes (checked against the file's own shape). Checked on the downloads (`stack-check`, 25 s): W2S 360 planes (S=120 × C=3); Tribolium 15,500 pairs, 248,000 planes, each partner the same shape; BioSR+ CCPs 384 pairs × 9 patterns (512 × 256); zebrafish optic tectum 6,955 frames of 492² read from a 3.4 GB BigTIFF that carries no shape metadata, so its declared TYX is what labels it. The other nine sets are pinned by Zenodo MD5 and size, with their counts taken from the published listings, and get checked when downloaded |
| 35 ✅ | quality | **Q2: when a Noise2Void U-Net should stop** (added after Q12: its early stop on FMD is the one thing between the U-Net and the new default) | [pre-registered](#q2-pre-registration-when-a-noise2void-u-net-should-stop-brick-35): Q12's bar and FMD guard, plus a rule-only branch for the dilated default. **Measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q2-when-a-noise2void-u-net-should-stop)): keeping the last of 30 epochs (C) gives the U-Net 26.84 dB on the public sets and MreB slope 0.96, but 31.4794 on FMD against the 31.48 guard. Longer training gains on independent noise (confocal +0.84, two-photon +1.02) and loses on correlated widefield noise (−1.69), which the masked loss can't see |
| 36 ✅ | build | **S1 + S2 (systems track):** `perf-roofline` measures achieved GFLOP/s *and* memory bandwidth per kernel against this CPU's peaks; then the kernels are reworked for whichever bound dominates (512-bit `Vector512` width, or channel-blocked layout / cache tiling) | bit-identical to the reference kernels; speed-up measured for inference and training. **Done** ([results](docs/RESULTS.md#systems-s1--s2-brick-36-where-the-time-goes-and-512-bit-kernels)): every kernel compute-bound (34–75 FLOP/byte, 1–5% of bandwidth), inference 97% of a stack run, so width was the target, not layout; 512-bit forward and input-gradient kernels, bit-identical: 1.3–1.6× on denoise and training steps (interleaved) |
| 37 ✅ | quality | **Q19: the U-Net with routing** (added after Q2) | [pre-registered](#q19-pre-registration-the-u-net-with-routing-brick-37), with the owner's two guards: the routing threshold frozen on validation before FMD test is touched, and a per-modality floor next to the overall FMD bar (≥ 33.17 dB against P3's routed 32.97). **Measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q19-the-u-net-with-routing)): FMD **33.24 dB** without ground truth (confocal 34.45, two-photon 32.86, widefield 31.76: every floor met), public mean 26.78, MreB slope 0.96; only CARE 2D's per-set floor missed (19.12 against 19.20), where the structured network replaces a near-identity output on two test images with mismatched references |
| 38 ✅ | quality | **Self-supervised scoreboard:** DORA's best self-supervised recipe against the 2024 challenge leaderboard (W2S, SUPPORT, JUMP) and against CAREamics Noise2Void and N2V2 run here on the 2025 sets | [pre-registered](#scoreboard-pre-registration-doras-self-supervised-default-against-careamics-brick-38): DORA's routed default against CAREamics N2V and N2V2 on FMD, the three public sets, MDC25 Tribolium and BioSR+ CCPs (the 2024 challenge sets need a submission; DeepCAD-RT was thought to have no aligned reference, corrected afterwards: it has); a measurement that orders the next bricks. **Done** ([results](docs/RESULTS.md#the-self-supervised-scoreboard-brick-38)): competitive (within 0.3 dB of the better CAREamics variant) on all five sets: FMD **+1.76 dB** (widefield 31.76 against 27.20/28.06), Tribolium +0.12, BioSR+ −0.13, MreB −0.24, nucleoid −0.28 (CARE 2D −0.23, reported only); no targeted pass. N2V2 beat Noise2Void by ≥ 0.3 dB only on Tribolium, so Noise2Noise goes first in 39 |
| **Stage 2** | | **Better blind spots and frame pairs** | |
| 39 ✅ | build | **Noise2Noise** from neighbouring time-lapse frames, plus a spatiotemporal blind spot, **first** (the scoreboard's reading <!-- sb-next-brick: noise2noise -->: CAREamics' N2V2 beat its Noise2Void by ≥ 0.3 dB on one of five sets); then **N2V2** (blur pooling and median replacement, which remove Noise2Void's checkerboard artifacts) | gradient checks; the checkerboard test goes away. **Gate design (2026-11-27):** (1) finite-difference gradients for a first layer that takes several frames (both architectures); (2) single-frame networks unchanged bit for bit (weights formats 2/3 written byte for byte as before, every recorded reproduction still passing); a new weights format for multi-frame networks, older files still loading, `info` reporting the frame count; (3) tiled = whole-frame inference with 3-frame input on ragged sizes, the spatial halo carried across every frame channel (inference has no blind spot); (4) endpoints: frames are reflected (frame 1 stands in for t−1 at t = 0), never clamped, since repeating frame t into a neighbour channel would hand the network the very pixel its blind spot hides; a test proves no input channel at an endpoint contains the target frame's masked pixels, and Noise2Noise pairs never use a padded frame; (5) the blind spot hides the same positions in the target frame only, neighbouring frames untouched; (6) on synthetic still sequences, both new modes beat single-frame Noise2Void against the synthetic truth, and on drifting sequences the known Noise2Noise smear is reproduced and measured (a sanity check, not a quality bar); (7) stacks still stream: a 3-frame model holds a window of three frames and writes frame t once t+1 is read (rule 14 and gate 28's weak-reference test amended from "no past frame" to "nothing older than the model's window"); CLI `train --frames 3 --mode n2n|n2v` and `denoise` on a stack end to end **Done:** `train --mode n2n` (single input frame, symmetric adjacent pairs) and `train --frames 3` (spatiotemporal blind spot, routed like single-frame Noise2Void); only 1 or 3 input frames, because the endpoint test found that reflecting a 5-frame window puts frame t back into a neighbour channel. Synthetic sanity numbers (shallow 5 × 5 network, 2,000 steps, peak 10 photons): still scene, single-frame Noise2Void 25.64 dB, 3-frame blind spot 27.30, Noise2Noise 26.76; drifting 4 px/frame, 25.68 / 25.70 / 22.67, so Noise2Noise smears with motion and the blind spot doesn't. The blind spot learns slowly (its loss sees 2% of the pixels: with a tenth of the steps it was still behind single-frame), which brick 40's budget has to allow for |
| 40 | quality | **Time-lapse modes on real data (Track A):** single-frame Noise2Void, one-sided Noise2Noise, the 3-frame spatiotemporal blind spot and skip-centre Noise2Noise (TeD-inspired, built here: input t−1, t+1, target t) on DeepCAD-RT's zebrafish (mostly still) and neutrophil pair 02 (motile), scored against their aligned high-SNR movies | [pre-registered](#brick-40-pre-registration-time-lapse-modes-on-real-data-track-a): contiguous blocks with guard gaps, matched budgets, a dynamic-pixel guard; a mode that smears moving or transient signal doesn't ship as a default |
| 41 | build | **N2V2's pieces** (moved here from 39 by the owner on 2026-11-27, with Track B split off row 40): median replacement of masked pixels (a plain blind spot: the median of the other 24 pixels of its 5 × 5 window; a structured one: the median of the 12 pixels at \|dx\| + \|dy\| = 3, just outside diamond-2; pixels outside the patch left out), and a U-Net option without the global residual and the top skip | gradient checks; median replacement never reads a hidden pixel; the option round-trips through the weights file |
| 42 | quality | **N2V2 ablation (Track B):** the routed default against (B1) median replacement (both routed networks), (B2) no global residual and no top skip (the U-Net only; the dilated structured network has no skip), (B3) both; MreB primary (spot precision and recall side by side, slope, mean offset, the untrained network's output), FMD and Tribolium as guards (Tribolium and FMD widefield feel only B1) | to be pre-registered before 41's code is measured: precision up without recall, slope or offset getting worse |
| **Stage 3** | | **Context: 3D and time** | |
| 43 | build | **2.5D context:** neighbouring slices or frames (z−1, z, z+1) as input channels; exact volume tiling with a one-slice halo | tiled equals whole volume |
| 44 | quality | 2.5D against 2D on Tribolium and Planaria (3D) and the time-lapses | pre-registered |
| **Stage 4** | | **Stronger base weights (optional offline GPU)** | |
| 45 | build | **GPU training and export:** a PyTorch training script for DORA's architectures that writes DORA's weight format | the same weights give the same outputs in PyTorch and in DORA's C# inference |
| 46 | quality | **Supervised scoreboard:** NAFNet-style blocks (with TLC-style local pooling, so tiling stays exact) and wider networks, trained on the GPU, against the 2025 challenge leaderboard and CARE run here | pre-registered |
| 47 | build | `train-stream`: training data read from disk as a seeded working set. **Spec (agreed 2026-11-27):** two-level seeded sampling: the seed shuffles which shards are resident, patches are drawn within the resident window, and LRU only decides what is evicted (compact per-image descriptors, pre-allocated pixel buffers, as in the owner's Galadriel loader notes); the cache changes I/O, never the draw order. Also: a step whose gradient isn't finite is skipped, in the trainer and `OnlineAdapter` (no effect on finite gradients, so every recorded result still reproduces) | peak memory flat in dataset size; bit-identical weights to an in-memory run with the same draw order, with a cache cap small enough to force eviction; and windowed sampling costs no quality against uniform sampling (FMD within the seed spread), since it mixes images less per step |
| 48 | quality | **The shippable model:** one network pretrained on the permissive corpus (synthetic + CC BY / CC0), adapted per sample on the CPU; `ATTRIBUTIONS.md` generated. Ablation arm (agreed 2026-11-27, from v1's progressive-noise and occlusion training): a calibrated Poisson–Gaussian noise ladder (5 tiers) plus 2×2-block and streak occlusion in pretraining, testing whether it stops isolated outliers passing through the global residual and top skip (spot precision, brick 32) | beats today's bundled model on every benchmark and on FMD, which it never saw; becomes the default |
| **Stage 5** | | **State of the art as a tool** | |
| 49+ | builds | **ROI correction brush** (agreed 2026-11-27, v1's "Teach Dora" done right): "not a spot" / "real structure" strokes adapt through `OnlineAdapter` (persistent Adam, never a fresh t = 1 sign-step), one-click rollback, every stroke in the weights' provenance and the journey log. One stroke (mouse-down to mouse-up) is one logged command (ROI mask, mode, steps, learning rate, seed), never a wall-clock coalescing window, and replaying the log reproduces the adapted weights bit for bit (a gate); undo is an in-memory ring of snapshots that hold the weights *and* Adam's moments and step count (about 1.4 MB each for the 118k U-Net, 32 deep), so an undone stroke leaves no momentum behind; its quality pass is pre-registered before it ships: on MreB, do a few brushed ROIs raise held-out spot precision without lowering recall, the whole-image slope or PSNR on frames that weren't brushed (shared convolution weights make every stroke a global edit). **Track Changes**, merged into the "what was removed" view: the signed difference between any two saved states (raw, base, adapted, brush edits), computed exactly on the CPU in float32 and only coloured by a shader (sign, deadband, diverging LUT), with a per-ROI net intensity change; denoised − raw is a hint, not a verdict, since raw is noisy; gated under Xvfb + Mesa by colour classes on positive, negative and in-deadband patches (headless Godot never runs shaders). Live telemetry stays reference-free (residual whiteness, the noise check, estimated SNR gain): slope and spot precision need a reference and stay benchmark measurements. The delta-channel export waits for the OME-TIFF writer and a per-output provenance file. **Viewer requests** (with `viewport-chunks` / `ui-stack`): the bridge keeps one pending slot per request kind (latest wins) instead of refusing work while busy, and a newer request cancels a stale frame between tiles (`TiledInference` already takes a cancellation token); a cancelled frame is never shown as finished; gate: a headless test scrubs 13 → 14 → 15 → 16 during a busy job and only 16 runs, with the shown frame carrying the last request's id. Frontend only: the CLI and gates keep their deterministic pipelines. Then: uncertainty view; the fidelity report in the app; vendor formats (OME-TIFF first, then CZI/ND2, from `vendor-corpus`); `ui-stack`; `viewport-chunks`; a napari/Fiji bridge; release | per brick |

**Checkpoints.** The lab pilot starts as soon as stage 1 passes (Q12), not at the end. If Q12 misses, stage 1's
scoreboard (38) still runs, to show whether the gap is general or specific to the Q18 sets.

---

## Systems track: speed, big data and the viewport as an instrument (proposed 2026-11-26)

The owner proposed a three-layer design: a C++ GDExtension for I/O, memory and SIMD; the C# core for networks and
training; Godot for display, 3D and ROI tools. Checked against the code before planning:

- **The viewport already keeps the dynamic range.** Frames reach Godot as 32-bit float textures (`Image.Format.Rf`)
  and are contrast-stretched in a shader; nothing is reduced to 8 bits on the CPU. A 16-bit texture would halve the
  memory of a big stack, which matters for 3D, but it doesn't fix a precision problem.
- **C# already has what the C++ layer was for.** .NET 8 maps files into memory (`MemoryMappedFile`), works on
  unmanaged memory through spans and pointers with no GC involvement, and has AVX-512 (`Vector512`) and ARM NEON
  (`AdvSimd`) intrinsics. A C++ layer would add a third language and a native build per platform for abilities the
  core can reach in C#. **Not planned**; it comes back only if S1 finds a hot spot C# can't reach.
- **On the denoising path, compute dominates, not copying.** A 512² frame takes about 0.27 s to denoise, while reading
  and converting a 512² uint16 page takes milliseconds. Copies matter for *scrubbing* a large stack in the viewer,
  which is S3's and S4's job.
- **The CPU kernels have headroom.** A rough estimate from the parameter count (about 26k multiply-adds per pixel for
  the dilated network) puts today's speed near 50 GFLOP/s on 4 cores. That's roughly a tenth of this CPU's AVX-512 peak,
  and DORA's kernels use 256-bit vectors (`Vector<float>`'s default width). S1 measures it properly before anything
  is changed.
- **GPU compute through Godot is real but not "milliseconds everywhere".** A rough estimate for the 118k U-Net on a
  2048² frame (about 0.4 TFLOP) is tens to a few hundred ms on a discrete GPU and around a second on integrated
  graphics, against about 13 s on this CPU today. It would live in the Godot app only: a headless run has no
  RenderingDevice, so the CLI and servers keep the CPU path. It also crosses the owner decision of 2026-11-26
  (inference stays on the CPU), so it waits for a new decision.

| # | Brick | What | Gate |
|---|---|---|---|
| S1 | `perf-roofline` | `bench` reports achieved GFLOP/s **and memory bandwidth** per kernel (forward, backward, U-Net levels) against the machine's measured peaks (FMA throughput and streaming bandwidth), and a time breakdown of a stack run (decode, normalize, infer, encode, write) | numbers recorded; decides what S2 targets: vector width, or layout and cache tiling |
| S2 | `simd-512` | 512-bit kernels where the CPU has AVX-512 (`Vector512`), NEON on ARM, or a channel-blocked layout / cache tiling if S1 shows the kernels are memory-bound (the owner expects 1.5–2× from width alone, not 2×); Winograd F(2×2, 3×3) only as an optional, tolerance-tested path | bit-identical to the reference kernels (vectorized across pixels, so the summation order doesn't change); training and inference speed-up measured. Faster kernels also make every later quality pass cheaper |
| S3 | `mmap-stacks` | uncompressed TIFF/BigTIFF/MRC planes read through a memory map (`ImageStack` backend), so a 100 GB stack opens instantly and pages in what is viewed | resident memory flat while scrubbing a multi-GB stack; values identical to the stream reader |
| S4 | `viewport-chunks` (already planned) + **foveated order** | the tile queue sorted by distance from the viewport centre; tiles under the cursor first, the rest in the background | first visible tile within a stated budget; the finished frame identical to whole-frame inference (exact tiling holds) |
| S5 | `display-16bit` | R16 textures (half the memory of today's float ones) via `Texture2DRD`, with the contrast shader unchanged | displayed values equal the source within 1/65535; texture memory halved |
| S6 | `residual-lens` (the planned "what was removed" view) | a loupe showing raw, denoised, residual and the residual's local autocorrelation (from `NoiseDiagnostics`): removed signal shows up as structure, removed noise doesn't | on synthetic data, a deliberately erased filament lights up, and the residual of pure noise doesn't |
| S7 | `roi-probe` (the planned in-app fidelity report) | draw an ROI; get block-mean slope, bias and noise reduction of denoised against raw, live as Z/T is scrubbed, plus brick 32's metrics when a reference is loaded | without a reference it is labelled a consistency check (denoised should be an unbiased estimate of the raw mean), never "fidelity" |
| S8 | `live-adapt` | Noise2Void adaptation on a background thread with the preview refreshed every few epochs | the stop signal is whatever Q2 settles (no ground truth); the UI never blocks |
| S9 | `volume-view` | a 3D texture with maximum-intensity projection, ortho slices and clipping planes, for stage 3's 2.5D results | renders a known synthetic volume correctly; memory stated |
| S11 | `godot-net10` (agreed 2026-11-27): Godot 4.7/4.8 with .NET 10 for the app and the core together (the Godot project consumes the core, so they move as one; per the owner, Godot 4.5-mono can't target .NET 10, godotengine/godot#112701, and 4.8 moves its C# builds to .NET 10) | gates 7, 8 and 10 on the new Godot (SDK version, headless smoke test, CI's pinned download and SHA-512), the in-app noise check on load, and every recorded reproduction bit-identical; bench speed recorded. With it, a CPU-tier check: inference reproductions under `DOTNET_EnableAVX2=0` / `DOTNET_EnableAVX512F=0` (its kernels put pixels in the SIMD lanes, so they should match on any tier) |
| S10 | `gpu-infer` (**approved by the owner, 2026-11-26**: forward inference only, Godot app only, opt-in) | U-Net / dilated inference as RenderingDevice compute shaders in the Godot app; the CPU path stays the reference, the default for the CLI and headless runs, and the automatic fallback when no Vulkan RenderingDevice exists | max pixel difference < 1e-4 (normalized scale) and PSNR within 0.01 dB of the CPU on every architecture; speed recorded per GPU class |

**Order.** S1 and S2 come first: they're cheap, and faster kernels speed up every later experiment (training uses
the same convolutions). S3–S7 join stage 5 with `ui-stack`. S8 waits for Q2. S9 waits for stage 3. S10 is approved (see
the decisions above) and comes after the viewport bricks it serves. S1 and S2 are laid as one build brick (36), S1's
measurement first, so the build/quality cadence holds: 36 (S1 + S2), then 37 (Q19).

---

## Quality backlog (quality passes)

Ranked by expected value. Each item names the evidence it comes from. When a pass picks an item, its candidate,
bar and adoption rule are written here **before** it is measured.

| # | Item | Evidence | Status |
|---|---|---|---|
| Q1 | **Best supervised recipe**: U-Net + all five FMD noise levels + a longer budget | U-Net +0.98 dB supervised and still improving at the 30-epoch cap (P4); all-level training +0.5 to +3.0 dB on cleaner inputs for supervised models (P2) | ✅ brick 25, **measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q1-the-best-supervised-recipe)). Pre-registered bar: raw within 0.1 dB of 34.73, avg2–avg16 +0.5 dB on average, no level −0.1. Result: raw **−0.47** (34.26), avg2–avg16 **+2.39** on average (avg16 40.15, the first model above the untouched image). Raw missed, so nothing shipped |
| Q10 | **Noise-level routing**: pick the raw-trained or the all-level model per image from its estimated noise level | Q1: −0.47 dB at raw, +0.1 to +4.9 dB on cleaner inputs; P3's correlation routing worked | ✅ brick 27, **measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q10-routing-each-image-by-its-noise-level)). The threshold chosen on validation sends every image to B: A's raw edge there is 0.34 dB, and noise σ can't pick out the images where A wins (even level-perfect routing: +0.06 dB). test_mix raw 34.28 vs the 34.63 needed. Lesson: with unknown noise levels, the all-level model alone is the better single supervised model (−0.3 to −0.5 dB on raw, +1 to +5 dB elsewhere): **a trade for the owner to weigh (see Q4)** |
| Q18 | **DORA against CARE and Noise2Void on public data** (`ext-bench`, field validation): the same images, the same scoring (CARE's brightness-matched protocol), and the same patch budget for DORA and for CAREamics run on CPU | the viability question: nobody has compared DORA with the tools biologists already use | ✅ brick 31, **measured, not competitive** ([results](docs/RESULTS.md#quality-pass-q18-dora-against-care-and-noise2void-on-public-data)). Supervised: DORA's U-Net matches CAREamics CARE (+0.07 dB over three sets, ahead on each). Noise2Void: DORA's default (26k-parameter dilated network) is 0.72 dB behind on average and about 1 dB behind on the two sets where Noise2Void works; on CARE 2D neither works, and DORA's noise check says why (correlated noise). By the rule, Q12 moves ahead of the pilot |
| Q13 | **A noise-conditioned U-Net (FiLM)**: one all-level model that is told how noisy each image is, so it can change its response per image instead of averaging over five noise levels | Q1: the all-level model loses 0.47 dB at raw; Q10: switching between two fixed models can't win it back; FFDNet-style conditioning (FiLM modulation rather than an input noise map) adapts one model smoothly | ✅ brick 29, **measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q13-telling-the-network-how-noisy-each-image-is)). Seed 2026 raw 33.92 vs the 34.63 needed; over three seeds conditioning loses 0.19 dB at raw and 0.11 on cleaner levels, though the network does use its condition (a 2× wrong σ̂ costs 1.2 dB). Lessons: seed spread is about 0.3 dB, and Q1's raw cost was partly a bad seed (three-seed mean: 0.30 below the raw-trained U-Net, not 0.47) |
| Q12 | **A wider U-Net and a longer budget for all-level training** (24 or 32 base features, more epochs; optionally pretrained on synthetic Poisson–Gaussian pairs, then fine-tuned on FMD) | same; costs about 2–4× the compute. A 60-epoch run takes only about 6 minutes on this CPU, so the budget is cheap to raise. Pretraining on synthetic noise, then fine-tuning on real data, has been tried on FMD before | ✅ brick 33 (self-supervised part), **measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q12-closing-the-self-supervised-gap)). The U-Net (118k) won selection and, on the public sets, closed 60% of the gap (26.70 vs 26.28; CAREamics 27.00) and fixed the contrast compression (MreB slope 0.75 → 0.93, ratio error below the raw frame's). But the FMD guard failed: under Noise2Void the U-Net stops at epoch 4 on FMD and scores 31.21 (needed 31.48). Bigger (0.47M) and 0.2% masking were worse. The supervised all-level part moves to stage 4 |
| Q2 | **U-Net Noise2Void training**: its run stopped early (best epoch 4) and lost 0.47 dB | P4; **Q12**: the same early stop is the only thing between the U-Net and the new Noise2Void default (it wins the public sets and fixes fidelity). The logs show the FMD run chose its epoch on jittery validation PSNR while its masked loss was still falling | ✅ brick 35, **measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q2-when-a-noise2void-u-net-should-stop)). The chosen rule (last of 30 epochs) cleared the public bar (26.84) and slope bar (0.96) and missed the FMD guard by 0.0006 dB (31.4794). The early stop on FMD had been protecting widefield: on correlated noise the masked loss keeps falling while the network learns the noise (widefield 27.98 → 26.29). The rule-only branch failed too (dilated −0.37 dB on FMD) |
| Q20 | **CARE 2D as a gate** (added after Q19): its two test images' references don't match their inputs (brick 32: the untouched image has slope 0.75 and a 31% ratio error), every method scores 19.1–20.1 dB on it, and its per-set floor alone kept Q19's routed U-Net from adoption | Q18, brick 32, Q19 | ✅ **owner decision 2026-11-26, after Q19 (disclosed)**: CARE 2D stays in every table and mean, without a per-set floor. Q19's rule re-applied without it: **adopted** ([results](docs/RESULTS.md#q20-care-2d-reported-without-a-floor-owner-decision-after-q19)); `train --mode n2v` now routes by the noise check by default. Q19's own verdict stays "not adopted" in the record |
| Q19 | **The U-Net with routing**: Q2's U-Net (last of 30 epochs, no ground truth) for images the noise check calls independent, and a structured-blind-spot network for those it calls correlated (P3's routed design, 32.97 dB with the dilated networks) | Q2: the U-Net gains +0.84/+1.02 dB on confocal/two-photon and loses only on correlated widefield noise, which routing already handles | ✅ brick 37, **measured, not adopted** ([results](docs/RESULTS.md#quality-pass-q19-the-u-net-with-routing)): FMD 33.24 dB (all floors met), public mean 26.78, MreB slope 0.96; CARE 2D 19.12 against its 19.20 floor. Was: quality pass 37, [pre-registered](#q19-pre-registration-the-u-net-with-routing-brick-37). Guards set by the owner: the routing threshold is frozen on validation before FMD test is touched, and a per-modality floor sits next to the overall FMD guard, so a confocal/two-photon gain can't hide a widefield loss. Its validation rule also adds a per-set floor (Q2's four-set mean hid a 0.37 dB FMD validation loss) |
| Q21 | **A structured blind spot shaped to the measured correlation** (added after brick 38): Tribolium's noise correlates 0.75 one row apart and 0.30 two rows apart but 0.21 along a row ([diagnostic](docs/RESULTS.md#after-brick-38-why-the-noise-check-flags-every-tribolium-slice-post-hoc-diagnostic)), while diamond-2 hides neighbours in every direction; a footprint taller than it is wide (or one chosen from the noise check's own per-axis estimates) | brick 38's diagnostic; structured Noise2Void (Broaddus et al. 2020) masks along the correlated axis. Also seen in brick 39: the noise check called 12 of 18 synthetic 64 × 64 frames correlated although their noise is independent, so densely structured small frames can fool it (Tribolium's verdict was confirmed against its references); a small-frame false-positive check belongs here | open; chosen on validation data, guarded on FMD widefield (isotropic, where diamond-2 is right) |
| Q22 | **An SNR curriculum for per-stack adaptation** (from v1's progressive targets, learning rate decaying): adapt on the clearest patches first, unlock the rest when the residual (raw − denoised) is white enough, checkpoint at each step | owner proposal 2026-11-27; v2 already keeps the target ladder with a decaying rate | open, backlog |
| Q23 | **An auto-configured sensor profile**: the noise check's per-axis estimates and `NoiseCalibration`'s gain choose the blind-spot footprint, the learning-rate decay and the stop rule before training | owner proposal 2026-11-27 (v1 journey log / "genome" idea); Q21 would be its first component | open, backlog; settings chosen on validation data |
| Q24 | **Noise-addition training without blind spots (FM2S-style)**: median-filter estimate plus calibrated Poisson–Gaussian noise added to the noisy image, trained noisier-to-noisy, so no masked pixel is ever replaced by a neighbour | candidate against brick 38's spot-precision gap, next to brick 42's N2V2 arms (Track B) and under the same precision and recall floors. Reference supplied by the owner, verification pending: Liu et al., "FM2S: Towards Spatially-Correlated Noise Modeling in Zero-Shot Fluorescence Microscopy Image Denoising", arXiv:2412.10031 (2024; Machine Intelligence Research 23(1), 2026) | open, backlog |
| Q11 | **An all-level model that keeps raw quality**: weight raw frames more in the training mix (e.g. 4 raw frames plus 1 of each other level) | Q1/Q10: the all-level model's raw loss comes from sharing capacity across five levels | open |
| Q3 | **"Do no harm" on clean inputs**: every model scores below the untouched image on 16-frame averages (bundled model −4.2 dB) | P2, P3, P4 level tables; Q1's all-level U-Net is the first to beat the untouched image overall (40.15 vs 39.70) | open; Q10 is its best candidate |
| Q4 | **Per-mode default architecture and recipe** (U-Net, and possibly all noise levels, when training with ground truth) | P4 rule kept the dilated default; Q1/Q10: the all-level U-Net trades −0.3 to −0.5 dB on raw frames for +1 to +5 dB on cleaner ones; **needs the owner's decision** | waiting for a decision |
| Q5 | **Validation that can tell close models apart**: 48 patches ranked U-Net and dilated Noise2Void the wrong way round | P4 (25.56 vs 25.39 dB on validation, opposite on test) | open |
| Q6 | **Seed variance**: every comparison is one run with one seed | all phases | **partly measured in Q13**: the all-level U-Net's raw score spreads 0.29 dB over three seeds; verdicts that hinge on less (P3's, P4's) remain single-seed |
| Q7 | **Structured blind spots on clean inputs**: the routed pair loses from avg4 up | P3 | open (overlaps Q3) |
| Q8 | **A better synthetic generator for the bundled model** (correlated noise, more realistic structure), measured on FMD | the bundled model is synthetic-only by license; it scored 32.15 dB on real data and is the worst on clean inputs | open |
| Q9 | **Calibration on dim 8-bit images** (curve-based variance stabilization) | P3: variance after the transform 0.41 on confocal | open, low priority |
| Q14 | **Haar wavelet down- and upsampling in the U-Net** (as in MWCNN): keep the three detail bands that 2×2 average pooling throws away (average pooling already is the Haar low band, scaled), and invert the transform on the way up | Q12's question by other means: more detail reaches the ½ and ¼ levels without widening them. Exact tiling holds because the 2×2 blocks align with the U-Net's tile alignment of 4 | open |
| Q15 | **Structural reparameterization**: train each 3×3 convolution with parallel 1×1 and identity branches, then fold them into one 3×3 kernel | inference is unchanged (same kernels, same speed, same function class); it only helps optimization, with reported gains around 0.1–0.2 dB for small super-resolution networks; costs about 1.5× per training step | open, low priority |
| Q16 | **A detail-refinement stage for "Adapt to sample"** (Noise2Detail, MICCAI 2025) | a three-layer network trained on one noisy image, aimed at fast inference for smart microscopy. Caveat: refining against the noisy image leaks noise unless the first estimate is blind to each pixel | open, research |
| Q17 | **NAFNet-style blocks** (simple gates, channel attention) inside the U-Net | simple convolutional blocks rival transformers at a fraction of the cost. Channel attention pools over the whole image, which breaks exact tiling: pool over a fixed local window instead (the TLC fix), in training and at inference, and widen the tiling margin by half the window | open, after Q12 |


### Q13 pre-registration

Written before any of Q13's code exists. Nothing below is tuned on data; the only choice made on data is the
trainer's usual best-validation epoch.

- **Candidate.** P4's U-Net (16 → 32 → 64 features, depth 2) with FiLM modulation after each of its 10 hidden
  convolutions: z = γ(s) ⊙ conv(x) + β(s), then ReLU (the final 1-channel convolution is not modulated). The condition is
  one number per frame, s = ln(σ̂ / 0.05), where σ̂ is Immerkær's noise σ of the whole normalized input frame (the measure
  behind Q10's router). One small shared network maps s to every γ and β: 16 hidden ReLU units, γ = 1 + output,
  β = output. Its output layer starts at zero, so an untrained conditioned U-Net computes exactly the plain U-Net. That
  adds about 10.9k parameters and well under 1% of the compute.
- **Measuring σ̂.** It is measured once per frame, before tiling, so tiled output still equals whole-image output. In
  training it is measured on each full training frame (never per patch), so the network learns the estimator's errors
  as well as its signal. FiLM rather than an input noise map, because in a convolutional network a constant extra input
  channel only shifts the first layer's biases. FiLM reaches every resolution level.
- **Recipe.** Exactly Q1's candidate: supervised, all five levels (2 frames each), batch 16, 64 px patches, 50 steps ×
  up to 60 epochs, seed 2026, validation on held-out FOVs, best-validation epoch kept. The only change is the
  conditioning.
- **Mechanism check** (recorded, not part of the bar). On the 46 validation FOVs, evaluate with σ̂ doubled and halved. If
  PSNR moves less than 0.05 dB, the network has learned to ignore s, and a gain is not from conditioning.
- **Seeds.** Seed 2026 decides the bar, like for like with Q1 and Q10. Seeds 2027 and 2028 are also run, for both the
  conditioned and the plain all-level recipe. This is the first spread measured across seeds (toward Q6).
- **Bar** (test_mix, all 48, seed 2026). Raw ≥ 34.63 dB (the raw-trained U-Net's 34.73 − 0.1). The mean over avg2–avg16
  ≥ 37.56 (Q1's all-level 37.66 − 0.1). No level more than 0.2 dB below Q1's all-level model. **And robustness:**
  averaged over the three seeds, the conditioned recipe beats the plain all-level recipe at raw by at least 0.2 dB.
- **Adoption rule.**
  - **Bar met:** one model then matches the raw specialist on noisy frames and the all-level model elsewhere, which
    removes the trade Q4 is waiting on. `train --mode supervised` defaults to the conditioned U-Net (`--condition noise`)
    and, for FMD data, to all five levels, and Q4 closes for the supervised mode. Noise2Void and the bundled model keep
    the dilated network (P4's rule).
  - **Raw at least 0.2 dB above Q1's all-level model (≥ 34.46), but below 34.63:** "measured, not adopted, improvement
    recorded", and Q12 runs with conditioning on.
  - **Otherwise:** "measured, not adopted".
  - In both misses, conditioning stays an opt-in (`--condition noise`), like `--vst`.
- **Prior, stated now.** The gain may be small. Blind Gaussian denoisers trained on several noise levels come within
  about 0.1 dB of level-specific ones (DnCNN-B against DnCNN-S), so part of Q1's 0.47 dB may come from capacity (Q12)
  or from raw frames being only a fifth of the mix (Q11), not from the missing noise level.

### Q18 pre-registration: DORA against CARE and Noise2Void on public data (`ext-bench`)

Written before any of its code exists or anything is measured.

- **Datasets.** The three paired sets from `ext-data`: DeepBacs *E. coli* MreB (30 training pairs, 5 test), DeepBacs
  *E. coli* nucleoid (28, 2) and the ZeroCostDL4Mic CARE 2D example (22, 2). Only each set's own test images are
  scored, and nothing trains on them. (*B. subtilis* has no ground truth, so it isn't scored.)
- **Scoring: CARE's protocol**, because the high-SNR images are 3.3–10.8× brighter than the low-SNR ones. Each
  reference image is normalized by its 0.1st and 99.9th percentiles. Each output is then matched to it by the
  least-squares scale and offset (CSBDeep's `normalize_minmse`), and PSNR (data range 1) and SSIM are measured. Every
  method gets this treatment, the untouched noisy input included, so no method is helped or hurt by its brightness.
  A dataset's score is the mean over its test images; the headline is the mean over the three datasets.
- **Methods.** The training settings are fixed now.
  1. Noisy input and the bilateral filter.
  2. **DORA out of the box**, with no training on the dataset: the bundled synthetic model, and the FMD-trained
     supervised U-Net (`runs/p4/unet_sup.bin`; not shippable today, see the FMD licence decision).
  3. **DORA Noise2Void:** the default recipe (dilated network, 30 epochs × 50 steps, batch 16, 64 px patches, seed
     2026) on the training split's noisy images.
  4. **DORA supervised:** U-Net 16 → 64, depth 2, 60 epochs × 50 steps, batch 16, 64 px patches, seed 2026, on the
     training pairs. Each image of a pair is normalized by its own percentiles, CARE's convention.
  5. **CAREamics Noise2Void and CARE** (version pinned in `tools/experiments/requirements-careamics.txt`), their
     default 2D configurations with 64 px patches and batch 16, on CPU, trained on the same images. They get the same
     number of training patches as DORA's matching method: 24,000 for Noise2Void and 48,000 for CARE.
  Wall-clock training and inference times are recorded for every method on this machine; they're not part of the bar.
- **Bar.** DORA is **competitive** if, averaged over the three datasets, DORA supervised is within 0.5 dB of CAREamics
  CARE **and** DORA Noise2Void is within 0.5 dB of CAREamics Noise2Void.
- **Adoption rule.**
  - **Competitive:** the field-validation track goes on to the pilot with these numbers, and the README states them.
  - **Not competitive:** Q12 (capacity and training budget) moves ahead of the pilot, and the README states the gap.
  - Either way, the numbers are recorded and the README reports them.
- **Known weaknesses, stated now.** The test splits are tiny (5, 2 and 2 images), so per-image scores are reported
  too. One seed per method, and Q13 measured a seed spread of about 0.3 dB. The baselines are run here, on CPU, with
  an equal patch budget, not the GPU budgets their authors used, so their own published numbers may be higher.

### Q12 pre-registration: closing the self-supervised gap (brick 33)

Written before any of its code exists or anything is run.

- **The gap (Q18, brick 32).**
  - **PSNR:** DORA's default Noise2Void averages 26.28 dB over the three public sets, against 27.00 for CAREamics
    Noise2Void. DORA's is a 26k-parameter dilated network masking 2% of pixels for 30 × 50 steps; CAREamics' is a
    334k-parameter U-Net masking 0.2%.
  - **Fidelity:** DORA's compresses contrast on MreB (slope 0.75, against CAREamics' 0.86).
  - **Budget:** two of its three runs were still improving at the 30-epoch cap.
- **Candidates, chosen in three steps on validation images.** The validation images are the fifth of each training
  split that the trainer holds out with seed 2026, scored with CARE's protocol against their high-SNR partners. The
  test images aren't touched until the end.
  1. **Network:** the dilated stack (26k parameters, today's default), and the U-Net with 16 base features (118k) and
     with 32 (about 0.47M, CAREamics' size). All at 2% masking and 30 × 50 steps.
  2. **Masking**, for the chosen network: 2% against 0.2% of pixels.
  3. **Budget**, for the chosen network and masking: 30 against 90 epochs (× 50 steps). This step runs only if at least
     two of step 2's winning runs ended at the 30-epoch cap.
- **How each step chooses.** It keeps the candidate with the highest mean validation PSNR over the three sets among
  those whose MreB validation slope is at least 0.85; if none qualifies, the highest mean PSNR. Ties within 0.05 dB go
  to the cheaper candidate.
- **Fixed throughout:** batch 16, 64 px patches, seed 2026, Adam at 1e-3 with DORA's plateau decay and early stopping,
  dihedral augmentation, plain (unstructured) blind spots, and the noisy images of each training split only.
- **Bar** (on the test splits, after choosing): mean PSNR over the three sets ≥ 26.70 dB (CAREamics Noise2Void's 27.00
  − 0.3) **and** MreB test slope ≥ 0.90.
- **FMD guard.** The chosen recipe is also trained with P3's plain Noise2Void settings (raw frames, 4 per FOV, seed
  2026, FOV 19 excluded). It must score at least 31.48 dB on test_mix raw: P3's plain Noise2Void scored 31.68, so it may
  lose at most 0.2.
- **Adoption rule.**
  - **Bar met and the guard holds:** the chosen network, masking and budget become `train --mode n2v`'s defaults, and
    the lab pilot is unblocked.
  - **Bar missed, but better:** if the test mean still gains ≥ 0.3 dB on today's default (≥ 26.58), without lowering the
    MreB slope, and the guard holds, it becomes the default anyway ("improved, not competitive"). The pilot stays
    blocked.
  - **Otherwise:** not adopted.
  - The supervised default is untouched by this pass.
- **Known weaknesses, stated now.**
  - One seed, and Q13 measured a spread of about 0.3 dB.
  - The selection uses ground truth on held-out training images, which a user wouldn't have. But it picks one default
    for everyone, the way CAREamics' defaults were picked.
  - CARE 2D's noise is correlated, so no plain Noise2Void can do much there. It stays in the mean, as pre-registered
    for Q18.

### Q2 pre-registration: when a Noise2Void U-Net should stop (brick 35)

Written before any of its code exists or anything is run.

- **What Q12 left.** Q12's U-Net (16 base features, 118k parameters, 2% masking, 30 × 50 steps) closed 60% of the gap to
  CAREamics on the public sets (26.697 dB against the 26.70 bar) and fixed MreB's contrast (slope 0.93). The FMD guard
  then failed: 31.21 dB on test_mix raw, against the 31.48 needed.
- **What the logs show (read after Q12, before this pass).** The trainer picks its best epoch, and stops early, on
  validation **PSNR whenever the validation images have clean partners** (FMD), and on the masked Noise2Void loss
  otherwise (the public sets, which load noisy images only). On FMD the U-Net's masked validation loss fell at almost
  every epoch (0.00531 → 0.00436 over 10 epochs) and its SSIM rose (0.54 → 0.65), but its PSNR on 48 patches only
  jittered between 25.0 and 25.6 dB after epoch 2. It peaked at epoch 4, so training stopped at epoch 10 with epoch 4's
  weights. Q12's explanation (the masked loss's noise floor hides real gains) doesn't fit FMD: there the noisy signal
  was the PSNR, and the masked loss was the one still improving. On the public sets the loss did decide, and runs
  stopped at epoch 15 (nucleoid) and 28 (MreB).
  - A self-supervised recipe shouldn't choose its weights with ground truth anyway: a user training on their own
    images has none.
- **Candidates** (the Q12 network: U-Net, 16 base features, depth 2, 2% masking, at most 30 × 50 steps).
  - **A, today's rule:** best epoch and early stopping (patience 6) on validation PSNR when the validation images have
    clean partners, else on the masked loss.
  - **B, masked loss, full budget:** best epoch on the masked validation loss always, ground truth or not; no early
    stopping (all 30 epochs); the plateau learning-rate decay follows the same loss.
  - **C, last epoch:** B's run, keeping the final weights. It needs no held-out images at all.
- **Choosing, on validation only.** Mean validation PSNR over four sets: the three public sets (the fifth of each
  training split held out with seed 2026, scored with CARE's protocol, as in Q12) and FMD (the validation FOVs of a
  seed-2026 run, raw frames, `fmd-eval --validation-seed 2026`). The highest mean wins. Ties within 0.05 dB go to C,
  then B (fewer moving parts). A's runs already exist (Q12's `unet16_*` and `fmd_guard`). **If A wins, the pass ends
  there: not adopted, and the test sets aren't touched.**
- **Fixed throughout:** batch 16, 64 px patches, seed 2026, Adam at 1e-3 with DORA's plateau decay, dihedral
  augmentation, plain blind spots; public sets from their training splits' noisy images; FMD with P3's plain
  Noise2Void settings (raw frames, 4 per FOV, FOV 19 excluded).
- **Scored once, on test, for the winner:** the U-Net on the three public test splits (PSNR, and MreB's slope) and on
  FMD test_mix raw. The dilated network (today's default) is also trained and tested under the winning rule, for the
  third branch below.
- **Adoption rule.**
  - **Competitive:** public test mean ≥ 26.70 dB, MreB test slope ≥ 0.90 and FMD test_mix raw ≥ 31.48 (Q12's bar and
    guard). The U-Net and the winning rule become `train --mode n2v`'s defaults, and the lab pilot is unblocked.
    Verdict `adopted`.
  - **Improved:** public test mean ≥ 26.58 (+0.3 dB on today's 26.28), MreB slope ≥ 0.75 (not below today's) and the
    same FMD guard. Same defaults; the pilot stays blocked. Verdict `improved`.
  - **The rule alone:** otherwise, if the dilated network under the winning rule loses at most 0.10 dB against today
    on both the public test mean (≥ 26.18) and FMD test_mix raw (≥ 31.58), the rule becomes the Noise2Void default
    with the dilated network: model selection without ground truth, at no measured cost. Verdict `rule-only`.
  - **Otherwise:** not adopted. Verdict `not-adopted`.
  - Supervised training is untouched.
- **Known weaknesses, stated now.**
  - One seed; Q13 measured a spread of about 0.3 dB, which is as large as the margins here.
  - The guard needs +0.27 dB over the U-Net's 31.21, and the B/C choice uses ground truth on held-out images, as Q12
    did (it picks one default for everyone).
  - The rule changes when training stops, not what the network learns. If the U-Net under Noise2Void is simply worse
    on widefield FMD (it was level with the dilated network there: 27.98 against 27.97), longer training won't fix it.

### Q19 pre-registration: the U-Net with routing (brick 37)

Written before any of its code exists or anything is run.

- **Where it comes from.** Q2: Q12's U-Net under rule C (all 30 epochs, last epoch kept, no ground truth in the choice)
  gains where noise is independent from pixel to pixel (FMD confocal +0.84, two-photon +1.02 dB; public sets 26.84, MreB
  slope 0.96) and loses on widefield, whose noise is correlated (−1.69 dB), because the masked loss rewards it for
  learning that noise. P3: routing each image by DORA's noise check to one of two networks (plain blind spots for
  independent noise, structured diamond-2 blind spots for correlated noise) is DORA's best ground-truth-free result on
  FMD: **32.97 dB** on raw test_mix (confocal 34.09, two-photon 32.46, widefield 31.76).
- **The candidate: P3's routed pair with one network changed.** The network for images the noise check calls
  independent becomes the U-Net under Q2's rule C (16 base features, depth 2, 2% masking, `--select loss --patience 0
  --keep last`), trained only on those images, as P3's plain network was. The network for images it calls correlated
  is chosen on validation (below) between P3's own (dilated, diamond-2, `runs/p3/n2v_corr.bin`) and a U-Net with diamond-2
  blind spots under rule C, trained on the correlated images.
- **Guard 1 (owner): the routing threshold is frozen on validation.** Disclosed: the noise check's threshold of 0.4 was
  placed in brick 14 in the gap between FMD's *test* images' estimates. Before test is touched, every validation FOV
  (the FOVs a seed-2026 run holds out, raw frames) gets the noise check's estimate and the ground-truth correlation
  (noisy minus clean). If 0.4 puts every validation FOV on the side its ground truth says (truth ≥ 0.4 is correlated),
  it stays. Otherwise the threshold becomes the midpoint of the validation gap, or, with no gap, the value that
  misroutes the fewest validation FOVs (ties go to the value nearest 0.4). It is frozen before scoring test, and the same
  threshold routes training images, validation images and test images.
- **Choosing the correlated network: on validation, widefield only.** Mean validation PSNR over the validation FOVs whose
  (frozen) noise check says correlated. The higher wins; within 0.05 dB, P3's (no new network). No averaging across
  microscope types: Q2's four-set mean hid a 0.37 dB FMD validation loss.
- **Public sets, same recipe per set.** Each set's training images are routed by the frozen noise check: images called
  independent train the U-Net under rule C, images called correlated train the chosen correlated design with diamond-2
  blind spots, and each test image goes to its routed network. (If a set has images of only one kind, only that
  network is trained.) Scored with CARE's protocol as in Q18; MreB fidelity as in brick 32.
- **Fixed throughout:** batch 16, 64 px patches, seed 2026, Adam at 1e-3 with DORA's plateau decay, dihedral
  augmentation, at most 30 × 50 steps; FMD with P3's settings (raw frames, 4 per FOV, FOV 19 excluded); FMD scores
  unrounded (`fmd-eval --precise`).
- **Adoption rule (all on test, after the choices above).**
  - **FMD, against P3's routed pair:** overall ≥ 33.17 dB (32.97 + 0.2), **and** (guard 2, owner) every microscope type
    at least its P3 routed score − 0.1 dB: confocal ≥ 33.99, two-photon ≥ 32.36, widefield ≥ 31.66.
    *(Amended after the result by Q20, an owner decision: CARE 2D's per-set floor below was dropped; see the backlog.)*
  - **Public sets:** mean ≥ 26.70 dB and MreB slope ≥ 0.90 (Q12's bar), and no set more than 0.1 dB below today's
    Noise2Void default there (Q18: MreB 25.83, nucleoid 33.71, CARE 2D 19.30).
  - **All met: adopted.** `train --mode n2v` routes by the noise check: its training images that are called independent
    train the U-Net under rule C, the ones called correlated train the chosen correlated design (a mixed set yields two
    weights files, the second for `denoise --weights-correlated`), and `info --default-n2v` reports it. The lab pilot is
    unblocked.
  - **Otherwise: not adopted.** Nothing ships; the routed pair stays an opt-in recipe.
- **Known weaknesses, stated now.**
  - One seed, and the margins (0.1–0.2 dB) are within Q13's measured seed spread of about 0.3 dB.
  - FMD's test images were looked at for P3's routed pair (design 2 was devised after design 1's test results) and for
    the 0.4 threshold; this pass's baseline therefore carries that history.
  - Two networks cost twice the training of one.

### Scoreboard pre-registration: DORA's self-supervised default against CAREamics (brick 38)

Written before any of its code exists or anything is run. A measurement, not an adoption pass: nothing ships from it.
The reading below decides only the order of the next bricks.

- **What is compared.** DORA's default Noise2Void since Q20 (`train --mode n2v`, routed by the noise check) against
  CAREamics 0.3.5 Noise2Void and N2V2 (its blur-pool and median-replacement variant, `use_n2v2=True`), each with its
  default 2D configuration, plus the noisy input and the bilateral filter. Every method gets the same budget: 30 × 50
  steps, batch 16, 64 px patches, seed 2026, CPU only.
- **Sets: every benchmark DORA can score here.** Scored with CARE's protocol (reference normalized by its own 0.1/99.9
  percentiles, output matched by least squares), except FMD, which keeps its own protocol (8-bit PSNR on raw test_mix).
  - **FMD:** training on the raw training frames (4 per FOV, FOV 19 never), testing on test_mix raw.
  - **DeepBacs MreB, DeepBacs nucleoid, CARE 2D:** Q18's splits. CARE 2D is reported without a floor (Q20).
    CAREamics Noise2Void's numbers there are Q18's runs (same protocol), not re-run.
  - **MDC25 Tribolium** (3D patches of 16 × 64², used as 2D slices): training on the noisy slices of patches
    00001–02000 (32,000 slices); testing on patches 15251–15500 (4,000 slice pairs). The slices of one patch form one
    group, so DORA's validation split never separates them.
  - **BioSR+ clathrin-coated pits** (raw SIM frames, 9 per file, 512 × 256): training on the noisy frames of cells
    001–040 at all 8 signal levels (2,880 images); testing on cells 041–048 (576 pairs). One cell is one group. Scores
    are reported per signal level and overall.
- **Not scored, and why (stated now).** The 2024 challenge sets (W2S, SUPPORT, JUMP): the organisers hold the references,
  so scoring them needs a submission from the owner's account. DeepCAD-RT: its high-SNR movie is not a frame-aligned
  reference (blurred matched frames correlate 0.51–0.72 with the low-SNR ones, barely more than frames 50 apart).
  **Corrected 2026-11-27 (after the run):** that test was flawed (static anatomy correlates at any lag); compared on
  what changes in time, both DeepCAD-RT pairs here are synchronised with no spatial offset
  ([results](docs/RESULTS.md#before-brick-40-are-deepcad-rts-high-snr-movies-frame-aligned-references)).
  Planaria and the other four BioSR+ structures: not downloaded (disk); the script takes them when they are.
- **Reported:** PSNR and SSIM per set (BioSR+ also per level), MreB's fidelity (slope, ratio error, spots real), and
  wall-clock training time.
- **The reading, fixed now.**
  - DORA is **competitive** on a set when its PSNR is at least the better CAREamics variant's − 0.3 dB.
  - If CAREamics N2V2 beats CAREamics Noise2Void by ≥ 0.3 dB on at least two of the five sets with references
    (excluding CARE 2D), stage 2's N2V2 brick (39) stays next. Otherwise Noise2Noise from neighbouring frames goes first.
  - If DORA trails the better CAREamics variant by more than 0.3 dB on at least two of those five sets, the next quality
    pass targets the largest such gap.
- **Known weaknesses, stated now.** One seed (spread about 0.3 dB, as large as the margins). A CPU budget far below
  what CAREamics' authors train with, the same for everyone. Tribolium is 3D data scored in 2D (CARE's own model for it
  was 3D). BioSR+ raw SIM frames are usually reconstructed, not denoised, so they test the denoiser rather than the
  application.
- **Result (added after the run, 2026-11-27):** DORA competitive on all five sets (FMD +1.76 dB; worst MreB −0.24 and
  nucleoid −0.28); N2V2 ahead of Noise2Void by ≥ 0.3 dB on one set (Tribolium), so Noise2Noise goes first; no targeted
  pass. One deviation: Planaria and the other BioSR+ structures aren't taken by the script yet (each needs a split rule
  fixed in advance). [Results](docs/RESULTS.md#the-self-supervised-scoreboard-brick-38).

### Brick 40 pre-registration: time-lapse modes on real data (Track A)

Written before any of its code exists or anything is run, and approved by the owner on 2026-11-27 with the floors,
budget, splits and boundary rule below. The old row 40 is split: **Track A (this brick)** compares the time-lapse modes
on real time-lapses; **Track B** (the N2V2 ablation) needs N2V2's pieces built first, so it is bricks 41 (build) and 42
(its pass, pre-registered before 41's code is measured).

- **Data (fixed now).** DeepCAD-RT zebrafish optic tectum (492 × 492, 6,955 frames, mostly still, GCaMP6s calcium
  activity) and neutrophil pair 02 (512 × 512, 5,706 frames, motile cells), both two-photon, both measured synchronised
  with their high-SNR movies at offset (0, 0, 0) and with close-to-independent noise
  ([results](docs/RESULTS.md#before-brick-40-are-deepcad-rts-high-snr-movies-frame-aligned-references)), so frames are
  compared as they are, with no shift and no crop. Per movie, contiguous blocks with 10-frame guard gaps:
  - **training** frames 0–1199 (training holds its frames in memory until `train-stream`, brick 47, hence 1,200),
  - **validation** 1210–1359,
  - **test** 1370–1669: the primary block, the only one the adoption rule reads,
  - **late test** 5000–5299, neutrophil pair 02 only: **reported, never gating** (it asks whether a model trained on
    the first minutes still holds late in the movie, when cells have moved on).

  Frames outside these blocks are not used. **Every block is a standalone sequence:** training pairs and windows,
  validation patches, test inference and the test masks below all reflect at the block's own first and last frames
  (at test frame 1370, t − 1 is 1371; at 1669, t + 1 is 1668), exactly what standalone inference on a 300-frame stack
  does, and never read a frame outside the block (1369 or 1670), so no block sees the guard gap or its neighbour.
- **Arms, all on the same U-Net (16 base features, depth 2) under Q2's rule C (patience 0, last epoch kept; the
  validation loss never uses the reference), with the same budget: 60 epochs × 50 steps, batch 16, 64 px patches,
  seed 2026, CPU.** The noise check is expected to route both movies to the U-Net (it flagged 0 of 24 sampled frames of
  either); the run reports its verdict on the training frames, and any frame it would send to the correlated network is
  a deviation reported next to the scores, not a second network.
  - **A1, single-frame Noise2Void:** today's default.
  - **A2, one-sided Noise2Noise:** t → t + 1 and t + 1 → t, one input frame (brick 39's `--mode n2n`).
  - **A3, the 3-frame spatiotemporal blind spot:** input (t − 1, t, t + 1), blind spots in t only (`--frames 3`).
  - **A4, skip-centre Noise2Noise (TeD-inspired, not a replication of TeD):** input (t − 1, t + 1), target t, every
    pixel in the loss, frame t never in the input; the output is the neighbours' mean minus the network's correction.
    Built in this brick: weights format 5 with an explicit residual-base byte; training targets are the block's
    interior frames only (T ≥ 3); inference reflects at the ends ((I₁, I₁) at t = 0); T = 1 is refused.
- **Validation loss, on a fixed deterministic subset.** Per movie, 48 patches of 64 px whose frames, positions and
  blind-spot masks are drawn once with seed 2026 from the validation block's interior frames (1211–1358, so every arm's
  inputs and target lie inside the block); the same frames and positions for every arm and every epoch (each arm scores its own loss on them: blind-spot MSE for A1 and A3, MSE against the neighbouring
  frame t + 1 for A2, against frame t for A4). It decides nothing under rule C; it is the convergence record.
- **Convergence.** An arm whose validation loss at epoch 60 is more than 1% below its loss at epoch 50 is reported as
  **not converged**, next to its scores. If A3 is flagged, one longer A3 run (180 × 50 steps, otherwise identical) is a
  **post-hoc diagnostic**, recorded beside the arms and never entering the adoption rule; the comparison stays at one
  matched budget.
- **Scored per test frame against the high-SNR movie (CARE's protocol per frame, as in Q18), then averaged over the
  block's 300 frames:**
  - **PSNR and SSIM.**
  - **Dynamic-pixel PSNR (the guard metric).** The mask is chosen on the reference after a σ = 1 Gaussian blur and a
    square root, so shot noise, whose variance grows with brightness, doesn't pull bright still pixels in:
    S = √max(blurred reference, 0), on the reference's raw values. The change at frame t is
    |S_t − ½(S_{t−1} + S_{t+1})|, its deviation from the neighbours' midpoint, and the mask is that frame's top 10%.
    PSNR is computed on the **unblurred** reference pixels inside the mask, after the whole frame's CARE match. This
    isolates transient onsets, peaks and acceleration, where A4 cannot see frame t and A3 can.
  - **Steady-motion PSNR (reported only):** the same with the change |S_t − S_{t−1}| + |S_t − S_{t+1}|, which also
    catches constant-velocity drift, where A4 should beat A2.
  - **Brick 32's fidelity** on the test frames (slope, ratio error, spots found and real).
  - Wall-clock training time.
- **Adoption rule (test, once, primary blocks only).** An arm becomes what `train --data STACK` does when neither
  `--mode` nor `--frames` is given (today a stack file then isn't a training set) if, against A1, on **both** movies:
  - test PSNR ≥ A1 + 0.5 dB,
  - dynamic-pixel PSNR ≥ A1 − 0.3 dB,
  - intensity slope ≥ A1's − 0.03.

  If several qualify, the highest mean test PSNR over the two movies wins; within 0.1 dB of each other, the one with
  fewer input frames (A2, then A4, then A3). If none qualifies, nothing ships and the modes stay opt-in. Single-image
  training is untouched either way.
- **Reported regardless:** every arm on every metric and block, the late block included, and the motile-minus-still
  difference of each arm's PSNR gain over A1 (neutrophils minus zebrafish), the number the motion question is about.
- **Known weaknesses, stated now.** One seed (spread about 0.3 dB, below the 0.5 dB bar but not by much). Two movies,
  both two-photon. Only 1,670 frames of each, plus one late block of one movie. A CPU budget far below published methods'.
  The zebrafish frame rate, and so how many frames a calcium transient spans, isn't recorded with the data here. The
  neutrophils' "motile" label is the dataset's; how far cells move per frame isn't measured before the run. A top-10%
  mask always selects pixels, even in a frame where little changes; it measures where the most change is, not that
  there is any.

## Parking lot (deliberately not on the path)

- **Web export:** not possible for C# Godot projects in 4.5. Revisit only if Godot adds it.
- **3-D (volumetric) denoising and GPU training:** large scope. Reconsider after P5 measurements.
- **Transformer or attention hybrids** (Restormer, SwinIR, Uformer): state of the art on photos, but they need GPU
  training, much more data and a large hand-written implementation. Reconsider only once a GPU path exists and Q12–Q17
  have stopped paying off.
- **A wavelet sub-band loss** (stationary wavelet transform): unweighted, it is just the pixel loss again (the
  undecimated Haar transform is a tight frame), so it only matters as a heavier weight on high frequencies. That trades
  PSNR, which is every bar here, for sharpness. It also doesn't carry over to Noise2Void, whose loss is defined only at
  the masked pixels. Revisit if a sharpness metric joins the bars.
- **v1 (repo root, MNIST):** frozen as the educational original. Its bugs are documented in the root `CLAUDE.md`, not fixed.
- **History rewrite into per-brick commits:** possible, but it needs a force-push and an explicit request.

# Project DORA: context for LLM sessions

This file tells an AI coding session what this repository is and how it really behaves.
It was checked against the source at commit `02dd999`. **Where README.md, documentation.md or
LIMITATIONS.md disagree with the code, trust the code.** The differences are listed in §11.

> **Two projects live in this repo.** Everything below (§1–§14) describes **v1**, the GDScript MNIST digit
> app at the repo root. **`dora-bio/`** is a separate v2 rebuild: a C# fluorescence-microscopy denoiser
> with a Godot 4.5 .NET UI. It has its own rules in [`dora-bio/CLAUDE.md`](dora-bio/CLAUDE.md). See §15.

---

## 1. What it is

Project DORA ("The Learning Explorer") is a small **educational handwritten-digit recognizer built
in the Godot game engine, written entirely in GDScript**, with no ML libraries. The user draws a digit
on a canvas and clicks "What is this?". A hand-written fully-connected neural network (784→128→10)
predicts the digit and shows a confidence. The user can then confirm the prediction or correct it,
and the network immediately runs one backpropagation step on that drawing and saves its weights to disk.

At startup it trains on a subset of MNIST (1,000 images by default, augmented to 3,900) unless
saved weights already exist. **Trained weights are committed to the repo**, so a fresh clone goes
straight to inference. The project is "gamified": during batch training, accuracy targets go up
step by step and the learning rate is *raised* each time a target is hit ("success breeds success").
The code puts clarity and heavy console narration (emoji `print`s) ahead of speed or ML rigor.

- Author: Zeyphrus @ Inchworm Games (git author `zephyrus`). MIT license (LICENSE file).
- Repo: `Andraszao/project-dora-ml`. Version 1.0 (Oct 2025).
- Size: about 3,300 lines of GDScript across 6 scripts, plus 1 scene. No tests, no CI, no build step.

## 2. Stack and running

- **Godot 4.5** (`project.godot` → `config/features=PackedStringArray("4.5", "GL Compatibility")`;
  README says 4.4+). GL Compatibility renderer. Main scene: `scenes/Main.tscn` (uid `bvqxk8n7w6dkj`).
- Run: open the folder in the Godot editor and press F5. All output goes to the Godot console (stdout).
- **Godot is not installed in the cloud container**, so nothing here can be run or tested without
  installing it (headless Godot 4.5 would be needed to verify changes). There is no automated test suite.
- All configuration is done through `@export` vars on the `DrawingArea` node (script
  `DrawingCanvas.gd`) in the Godot Inspector. There are **55** exports, not the 67 the README claims.

## 3. Repository map

```
project.godot                 Godot project config (name, main scene, renderer)
scenes/Main.tscn              The only scene: UI layout (see §7)
scripts/
  DrawingCanvas.gd   (1412 L) extends Panel. THE ORCHESTRATOR: all config exports, startup/training
                              pipeline, drawing input, canvas→28×28 preprocessing, prediction UI,
                              feedback/online learning, progressive targets, session + journey persistence.
  NeuralNetwork.gd    (424 L) extends RefCounted. Weights, forward pass (predict), ReLU, softmax,
                              JSON save/load with version + architecture check. Also dead heuristic-weight code.
  NeuralNetworkTrainer.gd (577 L) RefCounted. Per-sample backprop, SGD and Adam, epoch loop,
                              early stopping, progress logging.
  MNISTLoader.gd      (361 L) RefCounted, all static. IDX file parsing, augmentation (rotate/scale/noise),
                              "progressive noise" + occlusion, ASCII digit visualizer.
  PatternClassifier.gd(304 L) RefCounted. Hand-tuned region-intensity heuristic classifier, used
                              only as a fallback when MNIST fails to load.
  MNISTWeights.gd     (234 L) RefCounted, static. Hand-crafted "pretrained" weights. UNUSED (see §12).
  *.gd.uid                    Godot 4 UID sidecar files (keep them with their scripts).
training_data/
  train-images.idx3-ubyte, train-labels.idx1-ubyte    ← the only files the code reads (res:// paths)
  t10k-*.idx?-ubyte                                    ← MNIST test set, never used
  */<same-name>/<file>                                 ← byte-identical duplicate copies, never used
dora_trained_weights.dat      Committed trained model (JSON, ~1.9 MB), loaded at startup
dora_milestone_weights.dat    Committed milestone backup (older format, no model_version). Never loaded.
dora_training_session.dat     Committed progressive-target/session state (JSON)
dora_training_journey.json    Committed "training journey" log (JSON). Contents are mostly empty, see §10.
README.md / LIMITATIONS.md / documentation.md   Human docs (partly aspirational, see §11)
icon.svg, LICENSE, .gitignore (weights lines are commented out on purpose), .gitattributes
```

The data is about 110 MB including the duplicates. Everything is committed directly, with no Git LFS.

## 4. Startup flow (`DrawingCanvas._ready`, all synchronous on the main thread)

1. `brush_size = default_brush_size`. If `reset_memory_on_start`, delete `dora_trained_weights.dat`
   (this deletes only that file, not the session, journey or milestone files).
2. If `auto_optimize_batch_size` (default **on**), overwrite `batch_size` with
   `max(50, min(cores*cpu_usage_limit*25, 150, training_samples/50))`. With the defaults this is **50**, not 100.
3. **Always** load MNIST (`MNISTLoader.load_training_batch(training_samples)`) and run augmentation and
   noise training, *even when saved weights exist and no training will happen*.
   - If loading fails: create a `PatternClassifier`, call `setup_canvas()`, then **`return` early, before
     the button signals are connected**, so the UI buttons do nothing in fallback mode (bug).
4. Optionally print an ASCII picture of the first training digit.
5. `NeuralNetwork.new()` (random init). Then `should_train = force_fresh_training or not load_weights_from_file()`.
6. If training: build a `NeuralNetworkTrainer`, configure Adam or SGD with an LR callback, early stopping and
   speed optimizations, then run `train_multiple_epochs(training_epochs, batch_size, progressive_callback?)`,
   then `save_weights_to_file()` and `save_training_session_state()`.
   Otherwise: `load_training_session_state()`, which restores the target level, epoch accuracies and learning_count.
7. If journey logging is enabled: `initialize_training_journey_logging()`. If `epoch_accuracies` is non-empty,
   `finalize_training_session()` and append a session to `dora_training_journey.json`.
8. `setup_canvas()`, connect the UI signals via hard-coded relative node paths (`../ButtonContainer/...` etc.).

Training blocks the main thread, so the window freezes during it. The docs estimate 30–60 minutes for the defaults.

## 5. The neural network (`NeuralNetwork.gd`)

- Layers: 784 inputs (28×28, values 0..1) → 128 hidden (ReLU) → 10 outputs (numerically stable softmax).
- Storage: plain GDScript arrays. `weights_input_hidden[784][128]` (indexed `[input][hidden]`),
  `weights_hidden_output[128][10]` (`[hidden][output]`), `bias_hidden[128]`, `bias_output[10]`.
  101,770 parameters in total.
- Init: uniform in `±sqrt(2/fan_in)`, biases 0. The code calls this "Xavier", but the bound is a He-style
  scale used as a uniform range.
- `predict(pixels: Array[float]) -> {prediction:int, confidence:float, probabilities:Array[float]}`.
  It returns `prediction=-1` if the input length is not 784. **Side effect:** it caches `last_hidden_activations`,
  which the trainer's backprop reads. So `predict()` must always run immediately before `backpropagate()` on the same input.
- Save format (JSON via `JSON.stringify`): `{model_version:"1.0.0", architecture:{input_size,hidden_size,output_size},
  weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, trained:true, timestamp}`.
  Load warns if the version is not in `COMPATIBLE_VERSIONS`, refuses if the architecture differs, and requires `trained:true`.

## 6. Batch training: what actually happens

### Dataset construction (defaults)
- 1,000 MNIST training images (the first N in the file, normalized byte/255).
- `augment_training_data(factor=3)`: the originals plus 2 augmented copies of each, for 3,000 total. Each copy gets a random
  rotation of ±15°, a random scale of 0.85–1.15 (both nearest-neighbor, about center 13.5), and noise of ±0.05.
  **These values are hard-coded.** The `max_rotation_degrees`, `scale_range_*` and `noise_level` exports are ignored.
- `add_progressive_noise_training(ratio 0.3, max 0.2, steps 5)`: adds 900 more samples (180 per step, noise
  levels 0.04, 0.08, …, 0.2). These are always made from **the first 180 original images**. Each has a hard-coded 30% chance
  of occlusion (random pixels, horizontal stripes, or two corner blocks). The `enable_partial_occlusion` export is ignored.
  This gives 3,900 samples in total.
- "Gaussian noise" in the docs is actually **uniform** noise (`randf_range(-l, l)`), clamped to 0..1.

### Optimization loop (`NeuralNetworkTrainer`)
- **Weights are updated after every single sample (online SGD/Adam). `batch_size` does not batch gradients.**
  It only controls how often a progress line is printed.
- Per sample: `predict` → cross-entropy loss → `backpropagate`. The output error is `probs - one_hot`.
  The output layer is updated first, and *then* the hidden error is computed using the **already-updated**
  output weights (a small departure from textbook backprop). After that the hidden layer is updated.
- Adam keeps standard m/v state per parameter, with bias correction. Each per-parameter update is clamped to ±1.0
  (the code calls this "gradient clipping"). An optional cache stores the bias-correction `pow()` results.
- On every backprop, the learning rate is re-read through a callback that returns `DrawingCanvas.adam_learning_rate`.
  That field is **multiplied** by the progressive-target rewards, so the **learning rate rises during training** instead of decaying.
- The reported "accuracy" is **training accuracy**, counted on the fly from each prediction made just before that sample's update,
  over the augmented training set. **There is no validation or test evaluation at all.** The t10k files are never read.

### Stopping conditions (checked after each epoch, in this order)
1. The progressive callback returns `false` once "mastery" is reached.
2. If early stopping is on (the default): stop when accuracy ≥ `target_accuracy_threshold` (99) or after
   `early_stopping_patience` (2) epochs without a new best.
3. If early stopping is off: legacy rule, stop when accuracy > 95%.
- The `save_best_weights` export is **not implemented**. The final weights are saved, not the best ones.

### Progressive targets (`_evaluate_progressive_target_per_epoch` in DrawingCanvas)
- The target starts at 80% and rises by 4% after 2 *consecutive* epochs that hit it. A miss resets the counter.
- Each hit multiplies the LR by `achievement_lr_boost` (1.05). Each escalation multiplies it by a further
  `escalation_lr_boost` (1.10) and writes `dora_milestone_weights.dat`.
- Escalation only happens if `current + increment <= max_target_accuracy`. With the defaults the targets go
  80, 84, 88, 92, **96**, and the next step (100) exceeds 98, so "mastery" is declared at 96. **98 is never an actual target.**
- Committed result (`dora_training_session.dat`): 9 epochs. Accuracies 82.5, 93.3, 94.8, 96.4, 96.4, 96.6, **96.9**, 96.8, 96.5.
  Early stopping ended the run. The LR grew from 0.002 to 0.00454 (2.27×). Batch size was 50, from the auto-optimizer.

## 7. Inference: canvas to prediction

UI (`Main.tscn`): a `CenterContainer` holds a `VBoxContainer` (minimum size 400×600). Its children are
Title "PROJECT: DORA 🗺️", Subtitle, `DrawingArea` (Panel with the script, minimum size 280×280), Instructions,
`Prediction` label, `FeedbackContainer` (hidden by default; contains FeedbackLabel, Yes/No buttons, and an "Actually it's:"
SpinBox 0–9 with a "Teach Dora" button), `BrushContainer` (HSlider 4–32 step 2, value label), and `ButtonContainer`
("Clear Canvas", "What is this?").

Pipeline (`_on_test_pressed` → `convert_to_ml_data` → `preprocess_like_mnist`):
1. The canvas is an RGBA8 `Image` the size of the Panel's *actual* size, black, painted with white circular brush dabs from
   left mouse button and motion input, and shown through a `StyleBoxTexture`.
2. Block-average it down to 28×28 grayscale, using separate x and y scale factors.
3. Find the bounding box (threshold 0.05) and scale its longer side to `min(20, max_dim)`. Small drawings are never upscaled.
   Then center it on (14,14) with nearest-neighbor sampling. This is bounding-box centering, not MNIST's center-of-mass centering.
4. Apply a piecewise contrast curve (>0.7→1.0, 0.3–0.7 boosted, 0.1–0.3 ×2, 0.05–0.1→0.2, otherwise 0). A Gaussian blur
   exists but is disabled with `if false:`.
5. `neural_network.predict()` gives the label text "I think that's a N! (X% confidence) 🎯", shows the feedback UI,
   and prints the top 3 to the console.

Unverified: the VBox is 400 px wide, so the Panel is probably 400×280 rather than square, which would squash drawings
horizontally in step 2. The Panel size at `_ready` may also differ from its laid-out size until the first Clear.
The code prints "Panel actual size" at startup, so check that line before relying on this.

## 8. Interactive (online) learning (`learn_from_feedback`)

- **Yes** → `learn_from_feedback(pixels, predicted_label, true)`. **Teach Dora** → `learn_from_feedback(pixels, spinbox_value, false)`.
  **No** only changes the label text, asking the user to use the correction row. The Yes and correction branches run
  **identical code**: one `predict` and one `backpropagate` toward the given label.
- Each call creates a **new trainer with fresh Adam state** (and re-allocates every m/v matrix). At t=1, bias-corrected Adam reduces to
  about `lr * sign(gradient)`, so every weight with a nonzero gradient moves by about ±`online_learning_rate` (0.0001),
  whatever the gradient's size. Weights from pixels that are 0 get no update.
- The "PROOF OF LEARNING" console print shows only `weights_hidden_output[0][label]` and `bias_output[label]`. If hidden
  neuron 0 is inactive for that input, the weight change it shows is 0.
- The full weights JSON (about 1.9 MB) is rewritten to disk after every feedback click.
- `learning_count` is only persisted by `save_training_session_state()`, which only runs after batch training. In practice,
  feedback counts are lost on restart, although the weight changes themselves are kept.

## 9. Persistence and paths

| File | Writer | Reader | Notes |
|---|---|---|---|
| `dora_trained_weights.dat` | `NeuralNetwork.save_weights_to_file()` after training and after every feedback | `load_weights_from_file()` at startup | JSON; deleted by `reset_memory_on_start` |
| `dora_milestone_weights.dat` | `_apply_escalation_bonus()` | nobody | committed copy is the older format (no version/architecture) |
| `dora_training_session.dat` | `save_training_session_state()` after training | `load_training_session_state()` when not training | JSON despite the .dat extension. `original_adam_learning_rate` is hard-coded to 0.002 |
| `dora_training_journey.json` | `save_training_journey_log()` | `load_training_journey_log()` | Tab-indented JSON |

- Save and load paths are **bare relative filenames**. They resolve against the process working directory, which is the project root
  when the game is launched from the editor. MNIST is read from `res://training_data/...`. In an exported build, `res://` is the
  read-only pack, so the IDX files would have to be included in the export.
- `restore_earned_learning_rate` (default off) controls whether a boosted LR from the session file is reapplied.

## 10. Known bugs and quirks (verified by reading the code)

1. **Fallback mode has dead buttons**: an early `return` in `_ready` skips signal connection when MNIST fails to load. The
   later `else: pattern_classifier = ...` branch (around line 321) can never run.
2. **The journey log records almost nothing**:
   - `initialize_training_journey_logging()` runs *after* training, so `log_parameter_change()` calls made during training
     write into a placeholder dict that is then overwritten.
   - `log_epoch_completion()` and `log_achievement_milestone()` are never called anywhere.
   - `finalize_training_session()` also runs on launches that only load weights, because `epoch_accuracies` is restored from
     the session file. Each such launch appends a duplicate session.
   - The committed JSON has 8 sessions, all with empty `epochs`, `achievements` and `parameter_changes`. Sessions 7 and 8
     duplicate session 6. `session_duration_minutes` is always 0.
3. MNIST loading and augmentation run on every launch, even when no training happens.
4. The learning rate grows during training (by design, but unusual). Hidden-layer backprop uses the already-updated output weights.
5. "Accuracy" figures everywhere are training-set accuracy on augmented data. There is no held-out evaluation.
6. Many settings are ignored (see §12). The brush slider range in the scene (4–32) differs from the exports (4–40), and the
   exports are never applied to the slider.
7. `_on_feedback_no` leaves the feedback UI open. The correction flow depends on the user finding the SpinBox row.

## 11. Docs vs. code

| Docs claim | Reality |
|---|---|
| Mini-batch gradient descent, batch size 100 | Per-sample updates. Batch size only affects logging, and the auto-optimizer sets it to 50 |
| Accuracy evaluated on a validation set | Training accuracy only. No split, test set unused |
| Gaussian noise; configurable rotation, scale and noise ranges | Uniform noise; ranges hard-coded |
| 67 export variables | 55 |
| Max target 98% | The effective final target is 96% with the default increment |
| Online LR "100x smaller" (docs) / "10x smaller" (code comment) | 0.0001 vs 0.002, which is 20× |
| Journey log records epochs, achievements and LR changes | It doesn't (see §10.2) |
| Milestone and best-weights saving give safety | Milestones are written but never read. Best-weights saving isn't implemented |
| Buttons "Test My Drawing" / "Actually it's…" | The scene has "What is this?", "✓ Yes!", "✗ No", and "Teach Dora" |
| "92–98% on user drawings" | Unmeasured. The best recorded figure is 96.9% *training* accuracy |
| README "Project Info: License: Educational use" | The LICENSE file is MIT |

## 12. Dead or unused code and settings

- `MNISTWeights.gd` is used only by `NeuralNetwork.load_pretrained_weights()`, which is never called.
  `NeuralNetwork._create_simple_pattern_weights()` and the `_enhance_*` helpers are also never called.
- `DrawingCanvas._show_intelligent_feedback()` is left over from a removed "cellular automata / grow a digit" feature.
  Comments mention an "exploration folder" that is not in the repo, and `get_current_training_parameters()` still logs
  `enable_cellular_automata: false`.
- These are never called: `get_training_journey_summary()`, `get_ml_data()`, `log_epoch_completion()`,
  `log_achievement_milestone()`, and `apply_gaussian_blur()` (reachable only through `if false`).
- These exports are declared but never read: `max_rotation_degrees`, `scale_range_min`, `scale_range_max`, `noise_level`,
  `enable_partial_occlusion`, `save_best_weights`, `show_confidence_threshold`, `brush_size_step`.

## 13. Conventions when editing

- GDScript 4 with tab indentation. Use typed arrays (`Array[float]`, `Array[Array]`) where the existing code does;
  the forward and backward functions expect `Array[float]` inputs. Each class is a global `class_name`. Non-node classes
  extend `RefCounted`, and loader and weight helpers are `static func`.
- Each function starts with a `"""triple-quoted"""` string acting as a docstring (a bare string statement; GDScript has no real
  docstrings). Exports use `##` doc comments, which show as Inspector tooltips, and are grouped with `@export_group` / `@export_subgroup`.
- The style is educational: explain the *why* in comments and narrate with emoji `print`s. "Dora" is personified as "she".
- UI wiring uses hard-coded relative `get_node("../...")` paths from `DrawingArea`. Renaming or moving nodes in `Main.tscn` breaks them.
- Changing `input_size`, `hidden_size` or `output_size` makes the committed weights fail the architecture check on load,
  which triggers retraining. Bump `MODEL_VERSION` and `COMPATIBLE_VERSIONS` if the save format changes.
- Do not delete the committed `.dat` and `.json` state files casually. Without the weights file, the next launch runs a long training session.

## 14. History

- `62bb410` (2025-10-12) initial commit (.gitattributes)
- `454652f` (2025-10-13) v1.0: all code, data, docs and trained state
- `101f743` (2025-10-13) LIMITATIONS.md tweak
- `02dd999` (2026-02-02) added documentation.md (function reference; partly describes intended rather than actual behavior)

## 15. dora-bio/: the v2 rebuild (separate project)

- **What:** a self-supervised (Noise2Void) fluorescence-microscopy denoiser. The ML is pure C#
  (`dora-bio/src/DoraBio.Core`, net8.0, no Godot, no NuGet) with a Godot 4.5 **.NET** UI in `dora-bio/godot/`.
  None of the v1 scripts, MNIST data or `.dat` weights are used. v1 weights are rejected by v2's loader.
- **Isolation:** `dora-bio/.gdignore` stops v1's Godot project from importing it, and `dora-bio/godot/project.godot`
  is a separate Godot project. Open that folder, not the repo root, to run v2.
- **Built brick by brick:** `cd dora-bio && ./bricks.sh` runs 39 ordered gates (scaffold → nn → data → io →
  inference → training → cli → godot-build → godot-smoke → ci-core → ci-godot → fmd-fetch → fmd-loader → fmd-eval →
  n2v-diagnostic → structured-mask → vst → calibration → pool-upsample → unet → unet-tiling → arch-eval → bench →
  blocked-conv → perf-budget → q1-recipe → tiff-stack-write → q10-routing → cli-batch → q13-conditioning → ext-data → q18-ext-bench → quant-fidelity → q12-n2v-recipe → ext-data-2 → q2-n2v-stop → roofline-wide → q19-routed → scoreboard) and stops at the first failure. Gates 11–13 check the real FMD dataset when `dora-bio/tools/fetch_fmd.sh` has
  downloaded it (about 9 GB, gitignored); without it, their data checks report SKIPPED.
  CI (`.github/workflows/dora-bio.yml`) runs them all, with headless Godot, on PRs into `main`, pushes to `main`
  and manual runs (not on working-branch pushes or docs-only changes). Work on one brick at a time
  and commit only on green (rules in `dora-bio/CLAUDE.md`). In the cloud container the .NET 8 SDK comes from
  `apt install dotnet-sdk-8.0`. Gate 8 needs the Godot 4.5 .NET binary (`GODOT=…`), otherwise it is reported SKIPPED.
- **Roadmap:** `dora-bio/PATHWAY.md` (phases P0–P10; P0–P5 done, P6 in progress). From brick 25 on, build bricks alternate with
  pre-registered quality passes (backlog Q1–Q17 at the end of PATHWAY.md).
- **Real-data results:** `dora-bio/docs/RESULTS.md`. On FMD's 48 held-out images, supervised DORA scores 33.75 dB
  against 30.22 for a bilateral filter (34.73 with the U-Net, `--arch unet`; the dilated network stays the default). Plain Noise2Void fails on widefield, whose noise is correlated between
  pixels; two Noise2Void networks routed by DORA's own noise check reach 32.97 dB without any ground truth, and 33.24 dB
  with the U-Net for independent noise (Q19). Since Q20 that routed design is `train --mode n2v`'s default. Brick 38's
  scoreboard puts that default within 0.3 dB of CAREamics' Noise2Void and N2V2 on all five scorable sets at the same CPU
  budget, 1.76 dB ahead on FMD; by its pre-registered reading Noise2Noise comes before N2V2.
- **Rules, file map and gotchas:** `dora-bio/CLAUDE.md`. **Limitations:** `dora-bio/README.md`.
  The bundled sample and weights are **synthetic**. FMD data and FMD-trained weights are never committed.
  Every external dataset (licence, pins, whether it may train a shipped model) is in `dora-bio/tools/ext_manifest.json`;
  `dora-bio/ATTRIBUTIONS.md` is generated from it.

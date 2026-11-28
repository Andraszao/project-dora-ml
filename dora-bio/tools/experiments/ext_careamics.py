"""Quality pass Q18 baselines: CAREamics Noise2Void and CARE on the external datasets, on CPU.

    python tools/experiments/ext_careamics.py --algorithm n2v|care --dataset NAME [--root data/ext] [--out runs/q18]
    python tools/experiments/ext_careamics.py --algorithm n2v --n2v2 --dataset NAME --out runs/scoreboard

--n2v2 (brick 38's scoreboard) trains N2V2 instead, CAREamics' blur-pool and median-replacement variant of Noise2Void
(create_n2v_config(use_n2v2=True)), and writes to <out>/careamics_n2v2/<dataset>/. The scoreboard's extra sets are the
2D sets `dora-bio stack-export` derives from stacks and FMD as `dora-bio fmd-export --out data/ext/fmd-raw` writes it
(Noise2Void only: no clean training images). Test images of one shape are predicted in chunks of at most 2**20 pixels:
CAREamics predicts each sample on its own with the training set's statistics either way, so this only saves its
per-call overhead (4,000 Tribolium slices).

Run it with a Python that has tools/experiments/requirements-careamics.txt installed (tools/experiments/q18_ext.sh
does). It trains on the dataset's training split exactly as PATHWAY.md's Q18 pre-registration fixes: CAREamics' default
2D configuration, 64 px patches, batch 16, and the same number of training patches as DORA's matching recipe
(Noise2Void 30 epochs x 50 steps, CARE 60 x 50). Then it predicts every test image and writes float32 TIFFs named like
the test images to <out>/careamics_<algorithm>/<dataset>/, which `dora-bio ext-eval --predictions` scores with the same
code as DORA's outputs. Timings and versions go to timing.json next to them.

The folder layouts below must match src/DoraBio.Core/Data/ExtDatasets.cs.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import tifffile
import torch
from careamics import CAREamist
from careamics.config import create_care_config, create_n2v_config
from lightning.pytorch import seed_everything

# name: (training noisy, training clean, test noisy), relative to data/ext/<name>
LAYOUTS = {
    "deepbacs-mreb": ("train/low_SNR", "train/high_SNR", "test/low_SNR"),
    "deepbacs-nucleoid": ("train/low_SNR", "train/high_SNR", "test/low_SNR"),
    "zerocost-care2d": ("CARE (2D)/Training - Low SNR images", "CARE (2D)/Training - High SNR images", "CARE (2D)/Test - Low SNR images"),
    "mdc25-tribolium-2d": ("train/low_SNR", "train/high_SNR", "test/low_SNR"),
    "biosr-ccps-2d": ("train/low_SNR", "train/high_SNR", "test/low_SNR"),
    "fmd-raw": ("train/low_SNR", None, "test/low_SNR"),
}
CHUNK_PIXELS = 2 ** 20
EPOCHS = {"n2v": 30, "care": 60}
STEPS_PER_EPOCH, BATCH, PATCH, SEED = 50, 16, 64, 2026


def tiffs(folder: pathlib.Path) -> list[pathlib.Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")), key=lambda p: p.name)


def stack(paths: list[pathlib.Path]) -> np.ndarray:
    return np.stack([tifffile.imread(p).astype(np.float32) for p in paths])


def chunks(paths: list[pathlib.Path]) -> list[list[pathlib.Path]]:
    """Consecutive test images of one shape, at most CHUNK_PIXELS pixels per chunk (always at least one image)."""
    result: list[list[pathlib.Path]] = []
    shape, pixels = None, 0
    for path in paths:
        with tifffile.TiffFile(path) as tif:
            this = tif.pages[0].shape
        size = int(np.prod(this))
        if not result or this != shape or pixels + size > CHUNK_PIXELS:
            result.append([])
            shape, pixels = this, 0
        result[-1].append(path)
        pixels += size
    return result


def outputs(prediction, count: int) -> list[np.ndarray]:
    """CAREamics' prediction of a stack of `count` images as `count` 2D arrays, whatever nesting this version returns."""
    if isinstance(prediction, tuple) and len(prediction) == 2 and isinstance(prediction[1], list):
        prediction = prediction[0]   # (arrays, source names)
    arrays = [np.asarray(p) for p in prediction] if isinstance(prediction, list) else [np.asarray(prediction)]
    planes = [plane for a in arrays for plane in np.asarray(a, dtype=np.float32).reshape(-1, *a.shape[-2:])]
    if len(planes) != count:
        raise ValueError(f"CAREamics returned {len(planes)} planes for {count} images")
    return planes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["n2v", "care"], required=True)
    parser.add_argument("--dataset", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--root", default="data/ext")
    parser.add_argument("--out", default="runs/q18")
    parser.add_argument("--n2v2", action="store_true", help="N2V2 (blur pool, median replacement) instead of Noise2Void")
    parser.add_argument("--smoke", action="store_true", help="1 epoch x 2 steps, only to check the script runs")
    args = parser.parse_args()
    if args.n2v2 and args.algorithm != "n2v":
        parser.error("--n2v2 is a Noise2Void variant")
    if LAYOUTS[args.dataset][1] is None and args.algorithm == "care":
        parser.error(f"{args.dataset} has no clean training images")
    name = "n2v2" if args.n2v2 else args.algorithm
    epochs, steps = (1, 2) if args.smoke else (EPOCHS[args.algorithm], STEPS_PER_EPOCH)

    seed_everything(SEED, workers=True)
    base = pathlib.Path(args.root) / args.dataset
    noisy_dir, clean_dir, test_dir = (base / f if f else None for f in LAYOUTS[args.dataset])
    train_noisy = tiffs(noisy_dir)
    out = pathlib.Path(args.out) / f"careamics_{name}" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    common = dict(experiment_name=f"q18_{name}_{args.dataset}", data_type="array", axes="SYX",
                  patch_size=[PATCH, PATCH], batch_size=BATCH, num_epochs=epochs, num_steps=steps)
    if args.algorithm == "n2v":
        config = create_n2v_config(**common, use_n2v2=args.n2v2)
        train = dict(train_data=stack(train_noisy))
    else:
        config = create_care_config(**common)
        train = dict(train_data=stack(train_noisy), train_data_target=stack([clean_dir / p.name for p in train_noisy]))

    careamist = CAREamist(config, work_dir=out / "work", enable_progress_bar=False)
    started = time.perf_counter()
    careamist.train(**train)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for chunk in chunks(tiffs(test_dir)):
        images = np.stack([tifffile.imread(p).astype(np.float32) for p in chunk])
        # A stack with a sample axis, so the axes match training's (CARE requires the same sample axis); CAREamics'
        # default prediction settings otherwise, as a user would get them.
        for path, prediction in zip(chunk, outputs(careamist.predict(images, axes="SYX", data_type="array"), len(chunk)), strict=True):
            tifffile.imwrite(out / path.name, prediction.astype(np.float32))
    predict_seconds = time.perf_counter() - started

    import careamics
    (out / "timing.json").write_text(json.dumps({
        "algorithm": name, "dataset": args.dataset, "train_seconds": round(train_seconds, 1),
        "predict_seconds": round(predict_seconds, 1), "test_images": len(tiffs(test_dir)),
        "epochs": epochs, "steps_per_epoch": steps, "batch": BATCH, "patch": PATCH,
        "careamics": careamics.__version__, "torch": torch.__version__, "threads": torch.get_num_threads(),
    }, indent=2) + "\n")
    print(f"{name} {args.dataset}: trained in {train_seconds:.0f} s, predicted in {predict_seconds:.1f} s -> {out}")


if __name__ == "__main__":
    main()

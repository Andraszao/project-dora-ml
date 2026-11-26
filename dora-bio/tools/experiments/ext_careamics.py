"""Quality pass Q18 baselines: CAREamics Noise2Void and CARE on the external datasets, on CPU.

    python tools/experiments/ext_careamics.py --algorithm n2v|care --dataset NAME [--root data/ext] [--out runs/q18]

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
}
EPOCHS = {"n2v": 30, "care": 60}
STEPS_PER_EPOCH, BATCH, PATCH, SEED = 50, 16, 64, 2026


def tiffs(folder: pathlib.Path) -> list[pathlib.Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")), key=lambda p: p.name)


def stack(paths: list[pathlib.Path]) -> np.ndarray:
    return np.stack([tifffile.imread(p).astype(np.float32) for p in paths])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["n2v", "care"], required=True)
    parser.add_argument("--dataset", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--root", default="data/ext")
    parser.add_argument("--out", default="runs/q18")
    parser.add_argument("--smoke", action="store_true", help="1 epoch x 2 steps, only to check the script runs")
    args = parser.parse_args()
    epochs, steps = (1, 2) if args.smoke else (EPOCHS[args.algorithm], STEPS_PER_EPOCH)

    seed_everything(SEED, workers=True)
    base = pathlib.Path(args.root) / args.dataset
    noisy_dir, clean_dir, test_dir = (base / f for f in LAYOUTS[args.dataset])
    train_noisy = tiffs(noisy_dir)
    out = pathlib.Path(args.out) / f"careamics_{args.algorithm}" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    common = dict(experiment_name=f"q18_{args.algorithm}_{args.dataset}", data_type="array", axes="SYX",
                  patch_size=[PATCH, PATCH], batch_size=BATCH, num_epochs=epochs, num_steps=steps)
    if args.algorithm == "n2v":
        config = create_n2v_config(**common)
        train = dict(train_data=stack(train_noisy))
    else:
        config = create_care_config(**common)
        train = dict(train_data=stack(train_noisy), train_data_target=stack([clean_dir / p.name for p in train_noisy]))

    careamist = CAREamist(config, work_dir=out / "work", enable_progress_bar=False)
    started = time.perf_counter()
    careamist.train(**train)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for path in tiffs(test_dir):
        image = tifffile.imread(path).astype(np.float32)
        # One-image stack, so the axes match training's (CARE requires the same sample axis); CAREamics' default
        # prediction settings otherwise, as a user would get them.
        prediction = careamist.predict(image[np.newaxis], axes="SYX", data_type="array")
        prediction = np.asarray(prediction[0] if isinstance(prediction, (list, tuple)) else prediction)
        while isinstance(prediction, np.ndarray) and prediction.dtype == object:
            prediction = np.asarray(prediction[0])
        tifffile.imwrite(out / path.name, np.squeeze(prediction).astype(np.float32))
    predict_seconds = time.perf_counter() - started

    import careamics
    (out / "timing.json").write_text(json.dumps({
        "algorithm": args.algorithm, "dataset": args.dataset, "train_seconds": round(train_seconds, 1),
        "predict_seconds": round(predict_seconds, 1), "test_images": len(tiffs(test_dir)),
        "epochs": epochs, "steps_per_epoch": steps, "batch": BATCH, "patch": PATCH,
        "careamics": careamics.__version__, "torch": torch.__version__, "threads": torch.get_num_threads(),
    }, indent=2) + "\n")
    print(f"{args.algorithm} {args.dataset}: trained in {train_seconds:.0f} s, predicted in {predict_seconds:.1f} s -> {out}")


if __name__ == "__main__":
    main()

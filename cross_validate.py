"""Repeated stratified 5-fold cross-validation on the development partition.

The development partition (training + validation images of the hold-out split,
85% of APTOS 2019) is divided into 5 stratified folds, and the procedure is
repeated with three seeds (42, 123, 456), giving 15 train / validation runs.
In each run the model is trained on 4 folds and early-stopped and evaluated on
the remaining fold. No hyperparameter is tuned inside the folds, and the test
set is never touched.

Only the proposed frozen configuration is supported: the backbone embeddings
of the development partition are extracted once and reused by every fold.

Outputs: fold assignments, per-fold metrics and their mean / standard deviation.

Example
-------
    python cross_validate.py --task multiclass --splits splits/aptos2019_split.csv \
                             --image-dir data/aptos2019/preprocessed --output-dir runs/cv_multiclass \
                             --feature-cache cache/aptos2019
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drgrading import config as C
from drgrading.data import FeatureDataset, FundusDataset, load_split, make_cv_folds
from drgrading.engine import fit, load_or_extract_features, make_loader, predict
from drgrading.losses import LOSSES, build_criterion
from drgrading.metrics import summarize
from drgrading.model import FUSIONS, build_model
from drgrading.utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=C.TASKS, required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branches", nargs="+", choices=("vit", "resnet"), default=["vit", "resnet"])
    parser.add_argument("--fusion", choices=FUSIONS, default="concat")
    parser.add_argument("--loss", choices=LOSSES, default="ce")
    parser.add_argument("--folds", type=int, default=C.CV_FOLDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(C.CV_SEEDS))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    batch_size = args.batch_size or C.BATCH_SIZE[args.task]
    class_names = C.CLASS_NAMES[args.task]

    # Development partition = training + validation images of the hold-out split
    dev_df = load_split(args.splits, ["train", "val"]).sort_values("image").reset_index(drop=True)
    dev_images = FundusDataset(dev_df, args.image_dir, args.task)

    set_seed(C.SEED)
    model = build_model(args.task, args.branches, args.fusion, "frozen", pretrained=True)
    cache = Path(args.feature_cache) / "dev.pt" if args.feature_cache else None
    features = load_or_extract_features(cache, model.extractor, dev_images, device,
                                        num_workers=args.num_workers, desc="Features (dev)")
    labels = dev_images.labels

    # Fold assignments (stratified on the five grades, shared by both tasks)
    folds = make_cv_folds(dev_df, args.seeds, n_splits=args.folds)
    folds.to_csv(out_dir / "fold_assignments.csv", index=False)
    position = {image: i for i, image in enumerate(dev_df["image"])}

    rows = []
    for seed in args.seeds:
        assignment = folds[folds["seed"] == seed]
        for fold in range(args.folds):
            val_idx = np.array([position[i] for i in assignment.loc[assignment["fold"] == fold, "image"]])
            train_idx = np.setdiff1d(np.arange(len(dev_df)), val_idx)

            set_seed(seed)
            model.classifier.reset_parameters()
            train_set = FeatureDataset(features, labels, train_idx)
            val_set = FeatureDataset(features, labels, val_idx)
            criterion = build_criterion(args.task, args.loss, labels[train_idx])
            print(f"\nSeed {seed} | fold {fold + 1}/{args.folds} | "
                  f"train {len(train_idx)} | val {len(val_idx)}")
            fit(model.classifier,
                make_loader(train_set, batch_size, shuffle=True, device=device, seed=seed),
                make_loader(val_set, batch_size, shuffle=False, device=device),
                criterion, device, args.task, max_epochs=args.max_epochs,
                patience=args.patience, verbose=False)

            probs, y = predict(model.classifier, make_loader(val_set, batch_size, False), device, args.task)
            result = summarize(args.task, y, probs, class_names)
            row = {"seed": seed, "fold": fold + 1, "accuracy": result["accuracy"],
                   "macro_f1": result["macro_f1"],
                   "balanced_accuracy": result["balanced_accuracy"]}
            if args.task == "multiclass":
                row["qwk"] = result["qwk"]
            rows.append(row)
            print("  " + " | ".join(f"{k} {v:.4f}" for k, v in row.items() if k not in ("seed", "fold")))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "cv_results.csv", index=False)
    metrics = [c for c in table.columns if c not in ("seed", "fold")]
    summary = {m: {"mean": table[m].mean(), "std": table[m].std(ddof=1)} for m in metrics}
    summary["n_runs"] = len(table)
    save_json(summary, out_dir / "cv_summary.json")

    print(f"\nCross-validation over {len(table)} runs:")
    for m in metrics:
        scale, unit = (1, "") if m == "qwk" else (100, " %")
        print(f"  {m:<18} {scale * summary[m]['mean']:.4f} +/- {scale * summary[m]['std']:.4f}{unit}")


if __name__ == "__main__":
    main()

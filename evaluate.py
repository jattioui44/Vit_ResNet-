"""Evaluate a trained checkpoint, or recompute metrics from a predictions file.

Examples
--------
Evaluate a checkpoint on the test set:
    python evaluate.py --checkpoint runs/multiclass/model.pt --splits splits/aptos2019_split.csv \
                       --image-dir data/aptos2019/preprocessed --split test --output-dir runs/multiclass/eval

Recompute every metric from an existing predictions CSV (no GPU needed):
    python evaluate.py --predictions runs/multiclass/predictions_test.csv --task multiclass \
                       --output-dir runs/multiclass/eval
"""

import argparse

import numpy as np
import pandas as pd

from drgrading import config as C
from drgrading.data import FundusDataset, load_split
from drgrading.engine import make_loader, predict, predictions_table
from drgrading.metrics import format_report, summarize, threshold_table
from drgrading.model import load_checkpoint
from drgrading.plots import plot_confusion_matrix
from drgrading.utils import ensure_dir, get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="model.pt written by train.py")
    source.add_argument("--predictions", help="predictions_*.csv written by train.py")
    parser.add_argument("--task", choices=C.TASKS, help="Required with --predictions")
    parser.add_argument("--splits")
    parser.add_argument("--image-dir")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def probs_from_table(table, task, class_names):
    """Read back the probability columns written by predictions_table()."""
    if task == "binary":
        return table["prob_DR"].to_numpy()
    columns = [f"prob_{name.replace(' ', '_')}" for name in class_names]
    return table[columns].to_numpy()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    if args.checkpoint:
        if not (args.splits and args.image_dir):
            raise SystemExit("--splits and --image-dir are required with --checkpoint")
        device = get_device(args.device)
        model, cfg = load_checkpoint(args.checkpoint, device)
        task = cfg["task"]
        class_names = C.CLASS_NAMES[task]
        df = load_split(args.splits, args.split)
        dataset = FundusDataset(df, args.image_dir, task)
        loader = make_loader(dataset, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, device=device)
        probs, labels = predict(model, loader, device, task)
        predictions_table(df["image"].tolist(), labels, probs, task, class_names) \
            .to_csv(out_dir / f"predictions_{args.split}.csv", index=False)
    else:
        if not args.task:
            raise SystemExit("--task is required with --predictions")
        task = args.task
        class_names = C.CLASS_NAMES[task]
        table = pd.read_csv(args.predictions, dtype={"image": str})
        labels = table["true"].to_numpy().astype(int)
        probs = probs_from_table(table, task, class_names)

    result = summarize(task, labels, np.asarray(probs), class_names, threshold=C.BINARY_THRESHOLD)
    if task == "binary":
        result["thresholds"] = threshold_table(labels, probs, prevalence=C.DR_PREVALENCE)
    save_json(result, out_dir / f"metrics_{args.split}.json")
    plot_confusion_matrix(result["confusion_matrix"], class_names,
                          out_dir / f"confusion_matrix_{args.split}.png")

    report = format_report(result, class_names, task)
    if task == "binary":
        report += "\n\nThreshold   Sens.   Spec.    PPV     NPV  (prevalence %.2f)" % C.DR_PREVALENCE
        for row in result["thresholds"]:
            report += (f"\n  {row['threshold']:.2f}    {100 * row['sensitivity']:5.1f}   "
                       f"{100 * row['specificity']:5.1f}   {100 * row['ppv']:5.1f}   {100 * row['npv']:5.1f}")
    (out_dir / f"report_{args.split}.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

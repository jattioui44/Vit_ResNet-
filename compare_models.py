"""Paired comparison of classifiers on the same test images (McNemar test).

Each comparison takes two prediction files written by train.py / evaluate.py
(columns image, true, pred). The McNemar statistic uses Edwards' continuity
correction, and the p-values of all comparisons given in one call are adjusted
together with the Holm-Bonferroni procedure.

Example (four pre-specified comparisons):
    python compare_models.py \
        --comparison "Binary: fusion vs ResNet50" runs/binary/predictions_test.csv runs/binary_resnet/predictions_test.csv \
        --comparison "Binary: fusion vs ViT"      runs/binary/predictions_test.csv runs/binary_vit/predictions_test.csv \
        --comparison "5-class: fusion vs ResNet50" runs/multiclass/predictions_test.csv runs/multiclass_resnet/predictions_test.csv \
        --comparison "5-class: fusion vs ViT"      runs/multiclass/predictions_test.csv runs/multiclass_vit/predictions_test.csv \
        --output results/mcnemar.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from drgrading.metrics import holm_adjust, mcnemar_test


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", nargs=3, action="append", required=True,
                        metavar=("NAME", "PREDICTIONS_A", "PREDICTIONS_B"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default=None, help="Optional CSV file for the results")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for name, path_a, path_b in args.comparison:
        a = pd.read_csv(path_a, dtype={"image": str})
        b = pd.read_csv(path_b, dtype={"image": str})
        merged = a.merge(b, on="image", suffixes=("_a", "_b"))
        if len(merged) != len(a) or len(merged) != len(b):
            raise SystemExit(f"{name}: the two files do not contain the same images")
        if (merged["true_a"] != merged["true_b"]).any():
            raise SystemExit(f"{name}: inconsistent ground-truth labels")
        test = mcnemar_test(merged["true_a"], merged["pred_a"], merged["pred_b"])
        acc_a = (merged["pred_a"] == merged["true_a"]).mean()
        acc_b = (merged["pred_b"] == merged["true_b"]).mean()
        rows.append({"comparison": name, "n": len(merged), "acc_a": acc_a, "acc_b": acc_b, **test})

    adjusted = holm_adjust([r["p"] for r in rows])
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
        row["significant"] = p_holm < args.alpha

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

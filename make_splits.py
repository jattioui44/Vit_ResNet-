"""Create the train / validation / test partition of a dataset.

APTOS 2019: stratified 85% / 15% hold-out, then 80% / 20% train / validation
inside the 85% development partition (seed 42). The same split is used for the
binary and the five-class task.

DDR: stratified 70% / 15% / 15% split of the gradable images (grade 5,
ungradable, is removed).

The output CSV has the columns image, label (0-4) and split (train, val, test)
and is the only input needed by the other scripts to rebuild the partitions.

Examples
--------
    python make_splits.py --dataset aptos --labels data/aptos2019/train.csv \
                          --output splits/aptos2019_split.csv
    python make_splits.py --dataset ddr --labels data/DDR/DR_grading \
                          --output splits/ddr_split.csv
"""

import argparse
from pathlib import Path

from drgrading.config import SEED
from drgrading.data import (describe_split, load_aptos_labels, load_ddr_labels,
                            make_holdout_split, make_three_way_split)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("aptos", "ddr"), required=True)
    parser.add_argument("--labels", required=True,
                        help="APTOS: path to train.csv. DDR: folder containing train/valid/test.txt")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "aptos":
        df = load_aptos_labels(args.labels)
        split = make_holdout_split(df, seed=args.seed)
    else:
        df = load_ddr_labels(args.labels)
        split = make_three_way_split(df, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output, index=False)

    print(f"Saved {len(split)} images to {args.output}\n")
    print("Five-class counts per partition:")
    print(describe_split(split, "multiclass").to_string())
    print("\nBinary counts per partition:")
    print(describe_split(split, "binary").to_string())


if __name__ == "__main__":
    main()

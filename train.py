"""Train the hybrid ViT-ResNet50 classifier (binary detection or five-class grading).

Proposed configuration (default arguments): both backbones frozen, concatenation
fusion, plain cross-entropy, Adam (lr 1e-4) with cosine decay to 1e-6, batch
size 32 (binary) or 16 (five-class), early stopping with patience 10 on the
validation loss, best checkpoint restored.

In the frozen regime the backbone embeddings are extracted once (batches of 64)
and the projection layers and the classification head are trained on them.
In the partial and full regimes the images go through the backbones at every
step.

The model is evaluated on the validation set at the end of training. The test
set is evaluated only when --evaluate-test is given; it should be used once,
for the final selected configuration.

Examples
--------
    python train.py --task binary     --splits splits/aptos2019_split.csv \
                    --image-dir data/aptos2019/preprocessed --output-dir runs/binary --evaluate-test
    python train.py --task multiclass --splits splits/aptos2019_split.csv \
                    --image-dir data/aptos2019/preprocessed --output-dir runs/multiclass --evaluate-test

Ablations (evaluated on the validation set):
    --branches vit                  ViT branch only
    --branches resnet               ResNet50 branch only
    --fusion weighted_sum | product | bilinear
    --regime partial | full         backbone fine-tuning
    --loss weighted_ce | focal      class-imbalance strategies
    --oversample                    random oversampling of minority classes
"""

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import Subset

from drgrading import config as C
from drgrading.data import FeatureDataset, FundusDataset, load_split, to_task_labels
from drgrading.engine import (fit, load_or_extract_features, make_loader, predict,
                              predictions_table)
from drgrading.losses import LOSSES, build_criterion, oversample_indices
from drgrading.metrics import format_report, summarize, threshold_table
from drgrading.model import FUSIONS, REGIMES, build_model, save_checkpoint
from drgrading.plots import plot_confusion_matrix, plot_training_curves
from drgrading.utils import count_parameters, ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=C.TASKS, required=True)
    parser.add_argument("--splits", required=True, help="Split CSV written by make_splits.py")
    parser.add_argument("--image-dir", required=True, help="Folder of preprocessed images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branches", nargs="+", choices=("vit", "resnet"), default=["vit", "resnet"])
    parser.add_argument("--fusion", choices=FUSIONS, default="concat")
    parser.add_argument("--regime", choices=REGIMES, default="frozen")
    parser.add_argument("--loss", choices=LOSSES, default="ce")
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=C.MIN_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Default: 32 for binary, 16 for five-class")
    parser.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--feature-cache", default=None,
                        help="Folder where frozen backbone features are cached (frozen regime)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Also evaluate the held-out test set")
    return parser.parse_args()


def evaluate_split(name, net, dataset, df, args, device, out_dir, batch_size):
    """Predict one partition and write metrics, predictions and confusion matrix."""
    class_names = C.CLASS_NAMES[args.task]
    loader = make_loader(dataset, batch_size, shuffle=False, device=device)
    probs, labels = predict(net, loader, device, args.task)
    result = summarize(args.task, labels, probs, class_names, threshold=C.BINARY_THRESHOLD)

    predictions_table(df["image"].tolist(), labels, probs, args.task, class_names,
                      threshold=C.BINARY_THRESHOLD).to_csv(out_dir / f"predictions_{name}.csv", index=False)
    if args.task == "binary":
        result["thresholds"] = threshold_table(labels, probs, prevalence=C.DR_PREVALENCE)
    save_json(result, out_dir / f"metrics_{name}.json")
    plot_confusion_matrix(result["confusion_matrix"], class_names, out_dir / f"confusion_matrix_{name}.png")

    report = format_report(result, class_names, args.task)
    (out_dir / f"report_{name}.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n=== {name} set ===\n{report}")
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    batch_size = args.batch_size or C.BATCH_SIZE[args.task]

    # Data
    split_df = load_split(args.splits)
    frames = {name: split_df[split_df["split"] == name].reset_index(drop=True)
              for name in ("train", "val", "test")}
    images = {name: FundusDataset(df, args.image_dir, args.task) for name, df in frames.items()}
    train_labels = to_task_labels(frames["train"]["label"], args.task)
    print(f"Device: {device} | train {len(images['train'])} | val {len(images['val'])} "
          f"| test {len(images['test'])}")

    # Model
    model = build_model(args.task, args.branches, args.fusion, args.regime, pretrained=True)
    n_trainable = count_parameters(model.classifier) + count_parameters(model.extractor)
    print(f"Trainable parameters: {n_trainable:,} "
          f"(classifier {count_parameters(model.classifier):,}, "
          f"backbones {count_parameters(model.extractor):,})")

    # Inputs: cached embeddings (frozen) or images (fine-tuning)
    if args.regime == "frozen":
        cache = Path(args.feature_cache) if args.feature_cache else None
        datasets = {}
        for name in ("train", "val", "test"):
            if name == "test" and not args.evaluate_test:
                continue
            features = load_or_extract_features(
                cache / f"{name}.pt" if cache else None, model.extractor, images[name],
                device, num_workers=args.num_workers, desc=f"Features ({name})")
            datasets[name] = FeatureDataset(features, images[name].labels)
        net = model.classifier
    else:
        datasets = dict(images)
        net = model

    train_set = datasets["train"]
    if args.oversample:
        indices = oversample_indices(train_labels, seed=args.seed)
        train_set = (FeatureDataset(train_set.features, train_set.labels, indices)
                     if args.regime == "frozen" else Subset(train_set, indices))
        print(f"Oversampled training set: {len(train_set)} samples")

    # Training (re-seeded so that the head initialisation does not depend on extraction)
    set_seed(args.seed)
    model.classifier.reset_parameters()
    criterion = build_criterion(args.task, args.loss, train_labels)
    workers = 0 if args.regime == "frozen" else args.num_workers
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=workers,
                               device=device, seed=args.seed)
    val_loader = make_loader(datasets["val"], batch_size, shuffle=False, num_workers=workers,
                             device=device)
    history = fit(net, train_loader, val_loader, criterion, device, args.task,
                  lr=args.lr, min_lr=args.min_lr, max_epochs=args.max_epochs,
                  patience=args.patience)

    # Save
    run_config = {**vars(args), "batch_size": batch_size, "trainable_parameters": n_trainable,
                  "branches": list(args.branches)}
    save_checkpoint(out_dir / "model.pt", model, run_config)
    save_json(run_config, out_dir / "config.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_training_curves(history, out_dir / "training_curves.png")

    # Evaluation
    evaluate_split("val", net, datasets["val"], frames["val"], args, device, out_dir, batch_size)
    if args.evaluate_test:
        evaluate_split("test", net, datasets["test"], frames["test"], args, device, out_dir, batch_size)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()

"""Attribution maps (Grad-CAM, Integrated Gradients, Saliency, SmoothGrad) for a trained model.

By default one test image per severity grade is drawn at random (seed 42) and
explained for the class predicted by the model. Specific images can be given
with --image-ids instead.

Outputs, in --output-dir:
    xai_grid.png                     input + four overlays, one row per image
    <image id>/input.png             preprocessed input
    <image id>/<method>.png          overlay of each method
    <image id>/<method>.npy          raw heatmap in [0, 1] (224 x 224)
    explained_images.csv             true class, predicted class and probability

Example
-------
    python explain.py --checkpoint runs/multiclass/model.pt --splits splits/aptos2019_split.csv \
                      --image-dir data/aptos2019/preprocessed --output-dir runs/multiclass/xai
"""

import argparse

import numpy as np
import pandas as pd
import torch

from drgrading import config as C
from drgrading.data import FundusDataset, load_split
from drgrading.engine import logits_to_probs
from drgrading.explainability import METHODS, Explainer, denormalize
from drgrading.model import load_checkpoint
from drgrading.plots import overlay_heatmap, plot_attribution_grid
from drgrading.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image-ids", nargs="+", default=None,
                        help="Images to explain (default: one random image per grade)")
    parser.add_argument("--only-correct", action="store_true",
                        help="Draw only among images classified correctly")
    parser.add_argument("--target", choices=("pred", "true"), default="pred",
                        help="Class explained: predicted (default) or ground truth")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--smoothgrad-samples", type=int, default=50)
    parser.add_argument("--smoothgrad-noise", type=float, default=0.15,
                        help="Noise level as a fraction of (max(x) - min(x))")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--internal-batch-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def predict_one(model, x, task):
    probs = logits_to_probs(model(x), task)[0].cpu().numpy()
    if task == "binary":
        p_dr = float(probs)
        return int(p_dr >= C.BINARY_THRESHOLD), np.array([1 - p_dr, p_dr])
    return int(probs.argmax()), probs


def select_images(df, dataset, model, task, device, only_correct, seed):
    """One randomly drawn image per class of the task, in class order."""
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    chosen = []
    for c in range(len(C.CLASS_NAMES[task])):
        candidates = np.flatnonzero(labels == c)
        rng.shuffle(candidates)
        for i in candidates:
            if only_correct:
                x = dataset[i][0].unsqueeze(0).to(device)
                if predict_one(model, x, task)[0] != c:
                    continue
            chosen.append(int(i))
            break
    return chosen


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    model, cfg = load_checkpoint(args.checkpoint, device)
    task = cfg["task"]
    class_names = C.CLASS_NAMES[task]
    methods = [m for m in args.methods if not (m == "Grad-CAM" and "resnet" not in cfg["branches"])]

    df = load_split(args.splits, args.split)
    dataset = FundusDataset(df, args.image_dir, task)
    if args.image_ids:
        index = {image: i for i, image in enumerate(dataset.images)}
        missing = [i for i in args.image_ids if i not in index]
        if missing:
            raise SystemExit(f"Images not in the {args.split} split: {missing}")
        selected = [index[i] for i in args.image_ids]
    else:
        selected = select_images(df, dataset, model, task, device, args.only_correct, args.seed)

    explainer = Explainer(model, task, smoothgrad_samples=args.smoothgrad_samples,
                          smoothgrad_noise=args.smoothgrad_noise, ig_steps=args.ig_steps,
                          internal_batch_size=args.internal_batch_size)

    grid_rows, records = [], []
    for i in selected:
        image_id = dataset.images[i]
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        y = int(y)
        pred, probs = predict_one(model, x, task)
        target = pred if args.target == "pred" else y
        print(f"{image_id}: true {class_names[y]} | predicted {class_names[pred]} "
              f"(p = {probs[pred]:.3f}) | explaining {class_names[target]}")

        heatmaps = explainer.explain(x, target, methods)
        image01 = denormalize(x)
        image_dir = ensure_dir(out_dir / image_id)
        _save_png(image01, image_dir / "input.png")

        overlays = {}
        for method, heatmap in heatmaps.items():
            slug = method.lower().replace(" ", "_").replace("-", "")
            np.save(image_dir / f"{slug}.npy", heatmap.astype(np.float32))
            overlays[method] = overlay_heatmap(image01, heatmap, alpha=args.alpha, cmap=args.cmap)
            _save_png(overlays[method], image_dir / f"{slug}.png")

        grid_rows.append({"image": image01, "overlays": overlays, "label": class_names[y]})
        records.append({"image": image_id, "true": class_names[y], "pred": class_names[pred],
                        "prob_pred": float(probs[pred]), "explained_class": class_names[target]})

    plot_attribution_grid(grid_rows, methods, out_dir / "xai_grid.png")
    pd.DataFrame(records).to_csv(out_dir / "explained_images.csv", index=False)
    print(f"\nAttribution maps written to {out_dir}")


def _save_png(image01, path):
    import matplotlib.pyplot as plt
    plt.imsave(path, np.clip(image01, 0, 1))


if __name__ == "__main__":
    main()

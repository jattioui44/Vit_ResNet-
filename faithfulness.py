"""Faithfulness of the attribution maps by pixel masking (perturbation test).

For every test image, the SmoothGrad map of the predicted class is computed.
The top-k% most salient pixels are then replaced by the mean pixel value of the
image and the model is re-evaluated. A random baseline masks the same number
of randomly chosen pixels (averaged over 10 random masks per image).

If the maps highlight regions the model actually relies on, masking salient
pixels must reduce accuracy much more than masking random pixels.

Example
-------
    python faithfulness.py --checkpoint runs/multiclass/model.pt --splits splits/aptos2019_split.csv \
                           --image-dir data/aptos2019/preprocessed --output-dir runs/multiclass/faithfulness
"""

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from drgrading import config as C
from drgrading.data import FundusDataset, load_split
from drgrading.engine import logits_to_probs
from drgrading.explainability import Explainer
from drgrading.model import load_checkpoint
from drgrading.utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--smoothgrad-samples", type=int, default=50)
    parser.add_argument("--smoothgrad-noise", type=float, default=0.15)
    parser.add_argument("--internal-batch-size", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=None, help="Limit for a quick check")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def predict_classes(model, x, task):
    probs = logits_to_probs(model(x), task)
    if task == "binary":
        return (probs >= C.BINARY_THRESHOLD).long().cpu().numpy()
    return probs.argmax(dim=1).cpu().numpy()


def apply_masks(x, masks):
    """Replace masked pixels by the per-channel mean of the image.

    x     : (1, 3, H, W) normalised image
    masks : (M, H, W) boolean array, True = pixel removed
    returns a batch (M, 3, H, W)
    """
    fill = x.mean(dim=(2, 3), keepdim=True)  # mean pixel value of this image
    m = torch.as_tensor(masks, device=x.device).unsqueeze(1).float()
    return x * (1 - m) + fill * m


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    model, cfg = load_checkpoint(args.checkpoint, device)
    task = cfg["task"]
    explainer = Explainer(model, task, smoothgrad_samples=args.smoothgrad_samples,
                          smoothgrad_noise=args.smoothgrad_noise,
                          internal_batch_size=args.internal_batch_size)

    df = load_split(args.splits, args.split)
    dataset = FundusDataset(df, args.image_dir, task)
    n_images = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))

    correct_clean = 0
    correct_salient = {r: 0 for r in args.ratios}
    correct_random = {r: 0.0 for r in args.ratios}

    for i in tqdm(range(n_images), desc="Perturbation test"):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        y = int(y)
        pred = int(predict_classes(model, x, task)[0])
        correct_clean += pred == y

        # Saliency ranking of the pixels (max over colour channels)
        attribution = explainer.smoothgrad_attribution(x, pred)
        saliency = attribution.detach().abs().amax(dim=1)[0].flatten().cpu().numpy()
        order = np.argsort(-saliency)
        n_pixels = saliency.size
        h, w = x.shape[2:]

        for r in args.ratios:
            k = int(round(r * n_pixels))
            salient = np.zeros(n_pixels, dtype=bool)
            salient[order[:k]] = True
            random_masks = np.zeros((args.random_repeats, n_pixels), dtype=bool)
            for j in range(args.random_repeats):
                random_masks[j, rng.choice(n_pixels, size=k, replace=False)] = True

            masks = np.concatenate([salient[None], random_masks]).reshape(-1, h, w)
            preds = predict_classes(model, apply_masks(x, masks), task)
            correct_salient[r] += int(preds[0] == y)
            correct_random[r] += float((preds[1:] == y).mean())

    clean_acc = correct_clean / n_images
    rows = []
    for r in args.ratios:
        masked_acc = correct_salient[r] / n_images
        random_acc = correct_random[r] / n_images
        rows.append({"masking_ratio": r, "accuracy_masked": masked_acc,
                     "accuracy_drop_pp": 100 * (masked_acc - clean_acc),
                     "accuracy_random": random_acc,
                     "random_drop_pp": 100 * (random_acc - clean_acc)})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "faithfulness.csv", index=False)
    save_json({"n_images": n_images, "clean_accuracy": clean_acc, "results": rows},
              out_dir / "faithfulness.json")

    print(f"\nUnmasked accuracy: {100 * clean_acc:.2f} % on {n_images} images")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

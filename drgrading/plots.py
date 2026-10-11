"""Figures: confusion matrices, training curves and attribution overlays."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to disk
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names, path, title="Confusion Matrix"):
    """Annotated confusion matrix (rows = true label, columns = predicted label)."""
    cm = np.asarray(cm)
    size = 6 if len(class_names) <= 2 else 8
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label", fontweight="bold")
    ax.set_ylabel("True label", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=12)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, path):
    """Loss and accuracy per epoch for the training and validation sets."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="Training")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay_heatmap(image01, heatmap, alpha=0.5, cmap="jet"):
    """Blend a [0, 1] heatmap over an RGB image in [0, 1]; warm colours = high attribution."""
    colored = plt.get_cmap(cmap)(np.clip(heatmap, 0, 1))[..., :3]
    return np.clip((1 - alpha) * image01 + alpha * colored, 0, 1)


def plot_attribution_grid(rows, method_names, path):
    """Grid of overlays: one row per image, columns = input + one per method.

    ``rows`` is a list of dicts with keys ``image`` (H, W, 3 in [0, 1]),
    ``overlays`` (dict method -> overlay) and ``label`` (text shown on the right).
    """
    n_rows, n_cols = len(rows), len(method_names) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1.5, 3 * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"])
        for c, method in enumerate(method_names, start=1):
            axes[r, c].imshow(row["overlays"][method])
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, -1].yaxis.set_label_position("right")
        axes[r, -1].set_ylabel(row["label"], rotation=0, ha="left", va="center", fontsize=11)
    for c, title in enumerate(["Input image", *method_names]):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

"""Feature extraction, training loop with early stopping, and prediction."""

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import FEATURE_BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, MIN_LEARNING_RATE, PATIENCE


def to_device(inputs, device):
    """Move a tensor or a dict of tensors to a device."""
    if isinstance(inputs, dict):
        return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    return inputs.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Frozen feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(extractor, dataset, device, batch_size=FEATURE_BATCH_SIZE, num_workers=4,
                     desc="Extracting features"):
    """Run the frozen backbones once over a dataset.

    Returns a dict branch -> float32 tensor (N, D) stored on the CPU.
    """
    extractor.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    chunks = {b: [] for b in extractor.branches}
    for x, _ in tqdm(loader, desc=desc, leave=False):
        out = extractor(x.to(device, non_blocking=True))
        for b in extractor.branches:
            chunks[b].append(out[b].float().cpu())
    return {b: torch.cat(v) for b, v in chunks.items()}


def load_or_extract_features(cache_file, extractor, dataset, device, num_workers=4, desc=None):
    """Return cached features when they match the dataset, otherwise extract and cache them.

    The cache records the image identifiers, the image folder and the branches so
    that a stale file (other split, other preprocessing) is never reused.
    """
    key = {"images": list(dataset.images), "image_dir": str(Path(dataset.image_dir).resolve()),
           "preprocess": repr(dataset.preprocess_kwargs)}
    if cache_file is not None and Path(cache_file).exists():
        cached = torch.load(cache_file, map_location="cpu")
        if cached.get("key") == key and all(b in cached["features"] for b in extractor.branches):
            return {b: cached["features"][b] for b in extractor.branches}

    features = extract_features(extractor, dataset, device, num_workers=num_workers,
                                desc=desc or "Extracting features")
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": key, "features": features}, cache_file)
    return features


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _accuracy(logits, targets, task):
    if task == "binary":
        preds = (logits.view(-1) >= 0).long()
    else:
        preds = logits.argmax(dim=1)
    return (preds == targets).float().sum().item()


def run_epoch(model, loader, criterion, device, task, optimizer=None):
    """One pass over a loader. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = to_device(inputs, device)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = targets.size(0)
            total_loss += loss.item() * n
            total_correct += _accuracy(logits.detach(), targets, task)
            total += n
    return total_loss / total, total_correct / total


def fit(model, train_loader, val_loader, criterion, device, task,
        lr=LEARNING_RATE, min_lr=MIN_LEARNING_RATE, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, verbose=True):
    """Adam + cosine decay, early stopping on the validation loss.

    The weights of the epoch with the lowest validation loss are restored at
    the end of training.

    Returns
    -------
    list of dict
        Per-epoch history (losses, accuracies, learning rate, time).
    """
    model.to(device)
    criterion.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, task, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, task)
        scheduler.step()

        history.append({"epoch": epoch, "lr": current_lr,
                        "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "seconds": time.time() - start})

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1

        if verbose:
            mark = "*" if improved else " "
            print(f"Epoch {epoch:3d} {mark} train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {current_lr:.2e}")

        if wait >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} "
                      f"(val loss {best_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def logits_to_probs(logits, task):
    """Sigmoid P(DR) of shape (N,) for the binary task, softmax (N, K) otherwise."""
    if task == "binary":
        return torch.sigmoid(logits.view(-1))
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def predict(model, loader, device, task):
    """Return (probabilities, labels) as numpy arrays."""
    model.to(device).eval()
    all_probs, all_labels = [], []
    for inputs, targets in loader:
        logits = model(to_device(inputs, device))
        all_probs.append(logits_to_probs(logits, task).float().cpu())
        all_labels.append(targets)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy().astype(int)


def make_loader(dataset, batch_size, shuffle, num_workers=0, device=None, seed=42):
    """DataLoader with a seeded generator so that shuffling is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = device is not None and torch.device(device).type == "cuda"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin, generator=generator)


def predictions_table(images, labels, probs, task, class_names, threshold=0.5):
    """Per-image predictions as a pandas DataFrame (used for McNemar tests)."""
    import pandas as pd

    probs = np.asarray(probs)
    if task == "binary":
        preds = (probs >= threshold).astype(int)
        df = pd.DataFrame({"image": images, "true": labels, "pred": preds, "prob_DR": probs})
    else:
        preds = probs.argmax(axis=1)
        df = pd.DataFrame({"image": images, "true": labels, "pred": preds})
        for i, name in enumerate(class_names):
            df[f"prob_{name.replace(' ', '_')}"] = probs[:, i]
    return df

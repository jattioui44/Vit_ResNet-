"""Evaluation metrics and statistical tests.

* Accuracy, precision, recall (sensitivity), specificity, F1 (one-vs-rest per class)
* 95% Wilson score intervals for proportions
* 95% delta-method intervals for F1
* Balanced accuracy and macro-averaged metrics
* Quadratic weighted kappa with a Fisher z-transformation interval
* Expected Calibration Error (10 equal-width bins)
* ROC AUC with a DeLong interval (binary task)
* McNemar test with Edwards continuity correction and Holm-Bonferroni adjustment
"""

import math

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_interval(successes, n, z=Z_95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def f1_delta_interval(precision, recall, n_p, n_r, z=Z_95):
    """Delta-method interval for F1 = 2PR / (P + R).

    SE(F1)^2 = (2R^2/(P+R)^2)^2 * P(1-P)/n_P + (2P^2/(P+R)^2)^2 * R(1-R)/n_R
    with n_P = TP + FP and n_R = class support.
    """
    if precision + recall == 0 or n_p == 0 or n_r == 0:
        return float("nan"), float("nan")
    f1 = 2 * precision * recall / (precision + recall)
    s = (precision + recall) ** 2
    d_p = 2 * recall ** 2 / s
    d_r = 2 * precision ** 2 / s
    var = d_p ** 2 * precision * (1 - precision) / n_p + d_r ** 2 * recall * (1 - recall) / n_r
    se = math.sqrt(var)
    return max(0.0, f1 - z * se), min(1.0, f1 + z * se)


def qwk_interval(kappa, n, z=Z_95):
    """Interval on the quadratic weighted kappa using the Fisher z-transformation."""
    if n <= 3 or abs(kappa) >= 1:
        return float("nan"), float("nan")
    centre = math.atanh(kappa)
    half = z / math.sqrt(n - 3)
    return math.tanh(centre - half), math.tanh(centre + half)


# ---------------------------------------------------------------------------
# DeLong AUC variance
# ---------------------------------------------------------------------------

def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_auc_interval(y_true, scores, z=Z_95):
    """ROC AUC and its DeLong 95% interval for a single classifier."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), (float("nan"), float("nan"))

    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    se = math.sqrt(max(var, 0.0))
    return float(auc), (max(0.0, auc - z * se), min(1.0, auc + z * se))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(confidences, correct, n_bins=10):
    """ECE = sum_m |B_m|/n * |acc(B_m) - conf(B_m)| over equal-width bins."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin |= confidences == 0.0
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def prediction_confidence(probs, task):
    """Confidence of the predicted class (max(p, 1-p) for the binary task)."""
    probs = np.asarray(probs)
    if task == "binary":
        return np.maximum(probs, 1.0 - probs)
    return probs.max(axis=1)


# ---------------------------------------------------------------------------
# Main summary
# ---------------------------------------------------------------------------

def per_class_metrics(y_true, y_pred, class_names):
    """One-vs-rest precision, recall, F1 and specificity with 95% intervals."""
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    total = cm.sum()
    rows = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        n_p, n_r, n_s = tp + fp, tp + fn, tn + fp
        precision = tp / n_p if n_p else 0.0
        recall = tp / n_r if n_r else 0.0
        specificity = tn / n_s if n_s else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[name] = {
            "support": int(n_r),
            "precision": precision, "precision_ci": wilson_interval(tp, n_p),
            "recall": recall, "recall_ci": wilson_interval(tp, n_r),
            "f1": f1, "f1_ci": f1_delta_interval(precision, recall, n_p, n_r),
            "specificity": specificity, "specificity_ci": wilson_interval(tn, n_s),
        }
    return rows, cm


def summarize(task, y_true, probs, class_names, threshold=0.5, n_bins=10):
    """Compute every reported metric for a set of predictions.

    Parameters
    ----------
    task : "binary" or "multiclass"
    y_true : array (N,) of integer labels
    probs : array (N,) of P(DR) for the binary task, or (N, K) softmax output
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    y_pred = predictions_from_probs(probs, task, threshold)
    n = len(y_true)

    per_class, cm = per_class_metrics(y_true, y_pred, class_names)
    correct = int((y_true == y_pred).sum())
    recalls = [per_class[c]["recall"] for c in class_names]

    result = {
        "n": n,
        "accuracy": correct / n,
        "accuracy_ci": wilson_interval(correct, n),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_precision": float(np.mean([per_class[c]["precision"] for c in class_names])),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in class_names])),
        "weighted_f1": float(np.average([per_class[c]["f1"] for c in class_names],
                                        weights=[per_class[c]["support"] for c in class_names])),
        "ece": expected_calibration_error(prediction_confidence(probs, task), y_true == y_pred, n_bins),
        "per_class": per_class,
        "confusion_matrix": cm,
    }

    if task == "binary":
        auc, auc_ci = delong_auc_interval(y_true, probs)
        tn, fp, fn, tp = cm.ravel()
        result.update({
            "auc": auc, "auc_ci": auc_ci,
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "false_negatives": int(fn), "false_positives": int(fp),
        })
    else:
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        result.update({"qwk": float(kappa), "qwk_ci": qwk_interval(kappa, n)})
    return result


def predictions_from_probs(probs, task, threshold=0.5):
    probs = np.asarray(probs)
    if task == "binary":
        return (probs >= threshold).astype(int)
    return probs.argmax(axis=1)


def threshold_table(y_true, prob_dr, thresholds=(0.3, 0.5, 0.7), prevalence=0.34):
    """Sensitivity, specificity, PPV and NPV of the binary model at several thresholds.

    PPV and NPV are computed for a given population prevalence with Bayes' rule
    rather than from the test-set prevalence.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_dr = np.asarray(prob_dr, dtype=float)
    rows = []
    for t in thresholds:
        y_pred = (prob_dr >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        ppv_den = sens * prevalence + (1 - spec) * (1 - prevalence)
        npv_den = spec * (1 - prevalence) + (1 - sens) * prevalence
        rows.append({
            "threshold": t, "sensitivity": sens, "specificity": spec,
            "ppv": sens * prevalence / ppv_den if ppv_den else float("nan"),
            "npv": spec * (1 - prevalence) / npv_den if npv_den else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# Paired comparison of two classifiers
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar chi-square test with Edwards continuity correction.

    b = cases where A is correct and B is wrong
    c = cases where A is wrong and B is correct
    chi2 = (|b - c| - 1)^2 / (b + c), 1 degree of freedom
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2), "p": float(stats.chi2.sf(chi2, df=1))}


def holm_adjust(pvalues):
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def format_report(result, class_names, task):
    """Human-readable table of a summarize() result."""
    def pct(v):
        return f"{100 * v:6.2f}"

    def ci(v):
        return f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"

    lines = [f"{'Class':<10} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Spec':>6}  {'Recall 95% CI':<16} {'Support':>7}"]
    for name in class_names:
        r = result["per_class"][name]
        lines.append(f"{name:<10} {pct(r['precision'])} {pct(r['recall'])} {pct(r['f1'])} "
                     f"{pct(r['specificity'])}  {ci(r['recall_ci']):<16} {r['support']:>7}")
    lines.append("")
    lines.append(f"Accuracy          : {pct(result['accuracy'])} % {ci(result['accuracy_ci'])}")
    lines.append(f"Balanced accuracy : {pct(result['balanced_accuracy'])} %")
    lines.append(f"Macro F1          : {result['macro_f1']:.4f}")
    lines.append(f"ECE               : {result['ece']:.4f}")
    if task == "binary":
        lines.append(f"AUC               : {result['auc']:.4f} "
                     f"[{result['auc_ci'][0]:.3f}, {result['auc_ci'][1]:.3f}]")
        lines.append(f"Sensitivity       : {pct(result['sensitivity'])} %")
        lines.append(f"Specificity       : {pct(result['specificity'])} %")
    else:
        lines.append(f"QWK               : {result['qwk']:.4f} "
                     f"[{result['qwk_ci'][0]:.3f}, {result['qwk_ci'][1]:.3f}]")
    lines.append("")
    lines.append("Confusion matrix (rows = true, columns = predicted):")
    lines.append(str(np.asarray(result["confusion_matrix"])))
    return "\n".join(lines)

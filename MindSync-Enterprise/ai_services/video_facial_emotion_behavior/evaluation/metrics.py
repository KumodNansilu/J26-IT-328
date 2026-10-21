"""Evaluation metrics: accuracy, macro/per-class P/R/F1, confusion matrix.

Pure sklearn/numpy helpers — no employee data, only FER2013 test arrays.
"""
from __future__ import annotations


def compute_metrics(y_true, y_pred, *, labels=None) -> dict:
    """Return accuracy, macro + per-class precision/recall/F1."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_labels = len(labels) if labels else int(max(y_true.max(),
                                                  y_pred.max()) + 1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_labels)), zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "macro_precision": float(prec.mean()),
           "macro_recall": float(rec.mean()),
           "macro_f1": float(f1.mean()),
           "per_class": {}}
    names = list(labels) if labels else [str(i) for i in range(n_labels)]
    for i, name in enumerate(names):
        out["per_class"][name] = {"precision": float(prec[i]),
                                  "recall": float(rec[i]),
                                  "f1": float(f1[i])}
    out["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(n_labels))).tolist()
    return out


def save_confusion_matrix_png(cm, labels, path: str):
    """Save confusion matrix heatmap PNG (for the final report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

"""Utility helpers to evaluate trained models on the BUSI ultrasound dataset."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model


def _ensure_directory(path: str) -> None:
    """Create *path* and parents if it does not already exist."""

    os.makedirs(path, exist_ok=True)


def _sorted_class_labels(generator) -> List[str]:
    """Return class labels ordered according to ``flow_from_directory`` indices."""

    return [label for label, _ in sorted(generator.class_indices.items(), key=lambda item: item[1])]


def get_predictions(model, data_generator):
    """Collect ground truth, class probabilities and predicted labels for *data_generator*."""

    data_generator.reset()
    probabilities = model.predict(data_generator, verbose=0)
    y_true = data_generator.classes
    y_pred = np.argmax(probabilities, axis=1)
    labels = _sorted_class_labels(data_generator)
    return y_true, y_pred, probabilities, labels


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: str,
    normalize: bool = False,
) -> None:
    """Plot and save a (optionally normalised) confusion matrix."""

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix (normalized)"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    labels: List[str],
    save_path: str,
) -> Dict[str, float]:
    """Plot ROC curves for each class along with a micro-average curve."""

    num_classes = len(labels)
    y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(8, 6))
    roc_aucs: Dict[str, float] = {}

    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, idx], y_scores[:, idx])
        class_auc = auc(fpr, tpr)
        roc_aucs[label] = float(class_auc)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {class_auc:.3f})")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    roc_aucs["micro_average"] = float(auc_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"Micro-average (AUC = {auc_micro:.3f})",
        linestyle="--",
        color="black",
    )

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return roc_aucs


def save_classification_report(report: str, save_path: str) -> None:
    """Persist the textual classification report to disk."""

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(report)


def evaluate_model(
    model_path: str,
    data_generator,
    results_dir: str = "results",
    run_name: str = "evaluation",
) -> Dict[str, object]:
    """Evaluate *model_path* on *data_generator* and persist artefacts.

    Returns a dictionary containing metrics and paths to the generated artefacts.
    """

    _ensure_directory(results_dir)

    model = load_model(model_path)
    y_true, y_pred, y_prob, labels = get_predictions(model, data_generator)

    metrics_summary: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

    # Compute ROC-AUC scores where possible.
    try:
        y_true_one_hot = label_binarize(y_true, classes=np.arange(len(labels)))
        metrics_summary["roc_auc_macro"] = float(
            roc_auc_score(y_true_one_hot, y_prob, average="macro", multi_class="ovr")
        )
        metrics_summary["roc_auc_weighted"] = float(
            roc_auc_score(y_true_one_hot, y_prob, average="weighted", multi_class="ovr")
        )
        metrics_summary["roc_auc_micro"] = float(
            roc_auc_score(y_true_one_hot, y_prob, average="micro", multi_class="ovr")
        )
    except ValueError:
        # Can happen when a class is missing from y_true (e.g., small test split).
        pass

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)

    cm_path = os.path.join(results_dir, f"{run_name}_confusion_matrix.png")
    cm_norm_path = os.path.join(results_dir, f"{run_name}_confusion_matrix_normalized.png")
    roc_path = os.path.join(results_dir, f"{run_name}_roc_curves.png")
    report_path = os.path.join(results_dir, f"{run_name}_classification_report.txt")
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")

    plot_confusion_matrix(y_true, y_pred, labels, cm_path, normalize=False)
    plot_confusion_matrix(y_true, y_pred, labels, cm_norm_path, normalize=True)
    roc_aucs = plot_roc_curves(y_true, y_prob, labels, roc_path)

    save_classification_report(report, report_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, indent=2)

    return {
        "metrics": metrics_summary,
        "roc_aucs": roc_aucs,
        "classification_report": report,
        "confusion_matrix_path": cm_path,
        "confusion_matrix_normalized_path": cm_norm_path,
        "roc_curves_path": roc_path,
        "classification_report_path": report_path,
        "metrics_path": metrics_path,
        "labels": labels,
    }

"""Evaluate binary predictions and save non-interactive reports."""

import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def write_evaluation(
    labels: np.ndarray, probabilities: np.ndarray, output_dir: Path
) -> dict:
    """Apply the historical threshold and save the resulting metrics and figures."""
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Evaluation requires one finite probability per recording.")

    predictions = probabilities > 0.5
    counts = confusion_matrix(labels, predictions, labels=[0, 1])
    totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts * 100.0, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0
    )
    report = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "threshold": 0.5,
        "label_order": [0, 1],
        "test_recordings": int(len(labels)),
        "confusion_matrix": counts.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for matrix, filename, value_format in (
        (counts, "confusion.png", "d"),
        (percentages, "confusion_percent.png", ".2f"),
    ):
        figure = Figure(figsize=(6, 6), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.set_title("Confusion matrix" if value_format == "d" else "Confusion matrix (%)")
        ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(
            ax=axis, cmap=colormaps["Blues"], colorbar=False, values_format=value_format
        )
        figure.savefig(output_dir / filename)
    return report


def write_training_loss(losses: list[float], output_dir: Path) -> None:
    """Save the training loss against the actual epoch number."""
    figure = Figure(figsize=(6, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(range(1, len(losses) + 1), losses)
    axis.set(xlabel="Epoch", ylabel="Training loss")
    figure.savefig(output_dir / "loss.png")

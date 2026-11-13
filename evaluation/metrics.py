"""
evaluation/metrics.py — Shared evaluation utilities.

Reused by both evaluate_rig.py and evaluate_cwru.py so metrics are
computed consistently across datasets.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, accuracy, and false alarm rate.

    Parameters
    ----------
    y_true : list of int
        Ground-truth labels (1 = anomaly / fault, 0 = normal / healthy).
    y_pred : list of int
        Predicted labels from the detector.
    positive_label : int
        Which integer represents the anomaly class (default 1).

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy, false_alarm_rate
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == positive_label) & (y_true == positive_label)))
    fp = int(np.sum((y_pred == positive_label) & (y_true != positive_label)))
    fn = int(np.sum((y_pred != positive_label) & (y_true == positive_label)))
    tn = int(np.sum((y_pred != positive_label) & (y_true != positive_label)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy        = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "accuracy":        round(accuracy, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Evaluation Results",
) -> None:
    """
    Pretty-print a table of metrics for multiple detectors or fault types.

    Parameters
    ----------
    results : dict
        { 'detector_or_fault_name': metrics_dict }
    title : str
        Header line.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = f"{'Name':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FAR':>6} {'Acc':>6}"
    print(header)
    print("-" * 60)
    for name, m in results.items():
        print(
            f"{name:<25} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} "
            f"{m['false_alarm_rate']:>6.3f} "
            f"{m['accuracy']:>6.3f}"
        )
    print("=" * 60)


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 100,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the threshold that maximises F1 by sweeping over candidate values.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores from the detector.
    labels : np.ndarray
        Ground-truth labels (1 = anomaly).
    n_steps : int
        Number of threshold candidates to try.

    Returns
    -------
    (best_threshold, best_metrics)
    """
    candidates = np.linspace(scores.min(), scores.max(), n_steps)
    best_f1, best_thresh, best_metrics = -1.0, candidates[0], {}

    for t in candidates:
        preds   = (scores >= t).astype(int)
        metrics = compute_metrics(labels.tolist(), preds.tolist())
        if metrics["f1"] > best_f1:
            best_f1, best_thresh, best_metrics = metrics["f1"], t, metrics

    logger.info(
        "Threshold sweep: best F1=%.4f at threshold=%.4f",
        best_f1, best_thresh,
    )
    return float(best_thresh), best_metrics

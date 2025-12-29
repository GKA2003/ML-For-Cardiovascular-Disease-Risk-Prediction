"""
src.evaluation

Phase 7: Model evaluation and comparison.

This module standardises:
- Metric computation (incl. sensitivity/specificity/PPV/NPV) at a chosen threshold.
- ROC + Precision–Recall curves across models.
- Calibration curves (+ Brier score).
- Optional decision curve analysis (net benefit).
- Optional uncertainty via stratified bootstrap CIs.
- Optional pairwise tests (DeLong for ROC-AUC; paired bootstrap for AP/other metrics).

Outputs
- Figures saved under reports/figures/evaluation/
- Tables saved under reports/tables/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    BASELINE_MODELS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    TUNED_MODELS_DIR,
    THRESHOLD_PRIMARY_STRATEGY,
)
from src.utils import convert_numpy_types, load_model, load_model_metadata, save_figure

module_logger = logging.getLogger(__name__)


# ----------------------------
# Configuration
# ----------------------------

@dataclass(frozen=True)
class Phase7Config:
    """Configuration knobs for Phase 7 evaluation."""
    fallback_threshold: float = 0.5

    # Bootstrap uncertainty
    bootstrap: bool = True
    n_bootstraps: int = 500
    ci_alpha: float = 0.05  # 95% CI
    random_state: int = RANDOM_SEED

    # Pairwise tests
    pairwise_tests: bool = True

    # Plots
    plot_roc: bool = True
    plot_pr: bool = True
    plot_calibration: bool = True
    plot_decision_curves: bool = False

    # Calibration curve bins
    calibration_bins: int = 10

    # Decision curve thresholds (lo, hi, n_points)
    decision_curve_thresholds: Tuple[float, float, int] = (0.01, 0.99, 99)

    # Output locations (subfolders under reports/figures)
    figures_subdir: str = "evaluation"


# ----------------------------
# Core helpers
# ----------------------------

def _to_numpy(y: Any) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.Index)):
        return y.values
    if isinstance(y, pd.DataFrame):
        return y.squeeze().values
    return np.asarray(y)


def safe_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1|x). Raises a helpful error if missing."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
        raise ValueError("predict_proba output shape not understood.")
    raise AttributeError(
        f"Model {type(model).__name__} does not implement predict_proba(). "
        "Ensure probability=True for SVC."
    )


def choose_threshold(
    *,
    explicit_threshold: Optional[float],
    metadata: Optional[Dict[str, Any]],
    config: Phase7Config,
    model_name: str | None = None
) -> float:
    """
    Resolve a model threshold.

    Priority:
    1) explicit_threshold (passed in thresholds map)
    2) metadata['chosen_threshold']
    3) metadata['threshold_optimisation'][<primary>_optimization]['threshold']
    4) config.fallback_threshold
    """
    is_calibrated = bool(model_name) and ("calibrated" in model_name)

    if is_calibrated and metadata:
        if metadata.get("chosen_threshold_calibrated") is not None:
            return float(metadata["chosen_threshold_calibrated"])

        pack_cal = metadata.get("threshold_optimisation_calibrated")
        if isinstance(pack_cal, dict):
            key = f"{THRESHOLD_PRIMARY_STRATEGY}_optimization"
            if key in pack_cal and "threshold" in pack_cal[key]:
                return float(pack_cal[key]["threshold"])

    if explicit_threshold is not None:
        return float(explicit_threshold)

    if metadata:
        if "chosen_threshold" in metadata and metadata["chosen_threshold"] is not None:
            try:
                return float(metadata["chosen_threshold"])
            except Exception:
                pass

        pack = metadata.get("threshold_optimisation")
        if isinstance(pack, dict):
            key = f"{THRESHOLD_PRIMARY_STRATEGY}_optimization"
            if key in pack and isinstance(pack[key], dict) and "threshold" in pack[key]:
                try:
                    return float(pack[key]["threshold"])
                except Exception:
                    pass

    return float(config.fallback_threshold)

def resolve_threshold_with_doc(
    *,
    explicit_threshold: Optional[float],
    metadata: Optional[Dict[str, Any]],
    config: Phase7Config,
    model_name: str | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (threshold, doc) where doc explains source and strategy.
    """
    doc = {
        "model_name": model_name,
        "strategy": THRESHOLD_PRIMARY_STRATEGY,
        "source": "fallback",
        "field": None,
        "score": None,
    }

    is_cal = bool(model_name) and ("_calibrated" in model_name)

    # calibrated fields first
    if is_cal and metadata:
        if metadata.get("chosen_threshold_calibrated") is not None:
            try:
                thr = float(metadata["chosen_threshold_calibrated"])
                doc.update({"source": "metadata", "field": "chosen_threshold_calibrated"})
                return thr, doc
            except Exception:
                pass

        pack_cal = metadata.get("threshold_optimisation_calibrated")
        if isinstance(pack_cal, dict):
            key = f"{THRESHOLD_PRIMARY_STRATEGY}_optimization"
            if key in pack_cal and isinstance(pack_cal[key], dict) and "threshold" in pack_cal[key]:
                try:
                    thr = float(pack_cal[key]["threshold"])
                    doc.update({"source": "metadata", "field": f"threshold_optimisation_calibrated.{key}.threshold"})
                    if "score" in pack_cal[key]:
                        doc["score"] = float(pack_cal[key]["score"])
                    return thr, doc
                except Exception:
                    pass

    # explicit override
    if explicit_threshold is not None:
        doc.update({"source": "explicit", "field": "explicit_threshold"})
        return float(explicit_threshold), doc

    # uncalibrated fields
    if metadata:
        if metadata.get("chosen_threshold") is not None:
            try:
                thr = float(metadata["chosen_threshold"])
                doc.update({"source": "metadata", "field": "chosen_threshold"})
                return thr, doc
            except Exception:
                pass

        pack = metadata.get("threshold_optimisation")
        if isinstance(pack, dict):
            key = f"{THRESHOLD_PRIMARY_STRATEGY}_optimization"
            if key in pack and isinstance(pack[key], dict) and "threshold" in pack[key]:
                try:
                    thr = float(pack[key]["threshold"])
                    doc.update({"source": "metadata", "field": f"threshold_optimisation.{key}.threshold"})
                    if "score" in pack[key]:
                        doc["score"] = float(pack[key]["score"])
                    return thr, doc
                except Exception:
                    pass

    # fallback
    return float(config.fallback_threshold), doc

def compute_threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Compute classification + clinical metrics at a threshold.

    Includes:
    - sensitivity (recall), specificity
    - PPV (precision), NPV
    - balanced accuracy, Youden's J
    """
    y_true = _to_numpy(y_true).astype(int)
    y_proba = _to_numpy(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    balanced_acc = (sensitivity + specificity) / 2.0
    youden_j = sensitivity + specificity - 1.0

    return {
        "threshold": float(threshold),
        "prevalence": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(specificity),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_acc),
        "youden_j": float(youden_j),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


def compute_probability_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Metrics that do not require a threshold."""
    y_true = _to_numpy(y_true).astype(int)
    y_proba = _to_numpy(y_proba).astype(float)

    # Calibration (binning)
    cal_u = calibration_ece_mce(y_true, y_proba, n_bins=10, strategy="uniform")
    cal_q = calibration_ece_mce(y_true, y_proba, n_bins=10, strategy="quantile")
    lin = calibration_intercept_slope(y_true, y_proba)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "ece_uniform": float(cal_u["ece"]),
        "mce_uniform": float(cal_u["mce"]),
        "ece_quantile": float(cal_q["ece"]),
        "mce_quantile": float(cal_q["mce"]),
        **lin
    }

def _clip_proba(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = _to_numpy(p).astype(float)
    return np.clip(p, eps, 1.0 - eps)


def calibration_ece_mce(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",  # "uniform" or "quantile"
) -> Dict[str, Any]:
    """
    Expected calibration error (ECE) and max calibration error (MCE) with binning.

    Returns:
      {
        "ece": float,
        "mce": float,
        "bins_df": pd.DataFrame([bin_low, bin_high, n, mean_pred, frac_pos, abs_gap])
      }
    """
    y_true = _to_numpy(y_true).astype(int)
    y_proba = _clip_proba(y_proba)

    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # quantile bins based on predicted probability distribution
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_proba, qs)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
        bin_edges = np.unique(bin_edges)
        # if all probs are similar, fallback to uniform
        if len(bin_edges) < 3:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # assign bins
    bin_ids = np.digitize(y_proba, bin_edges[1:-1], right=False)

    rows = []
    n_total = len(y_true)
    ece = 0.0
    mce = 0.0

    for b in range(len(bin_edges) - 1):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        p_bin = y_proba[idx]
        y_bin = y_true[idx]
        mean_pred = float(np.mean(p_bin))
        frac_pos = float(np.mean(y_bin))
        gap = abs(frac_pos - mean_pred)
        weight = len(idx) / max(n_total, 1)

        ece += weight * gap
        mce = max(mce, gap)

        rows.append({
            "bin_low": float(bin_edges[b]),
            "bin_high": float(bin_edges[b + 1]),
            "n": int(len(idx)),
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
            "abs_gap": float(gap),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins_df": pd.DataFrame(rows),
    }

def calibration_intercept_slope(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Logistic recalibration fit:
      logit(P(Y=1)) = intercept + slope * logit(p_hat)

    Ideal:
      intercept ~ 0, slope ~ 1
    """
    y_true = _to_numpy(y_true).astype(int)
    p = _clip_proba(y_proba)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y_true)

    return {
        "calibration_intercept": float(lr.intercept_[0]),
        "calibration_slope": float(lr.coef_[0][0]),
    }

# ----------------------------
# Bootstrap uncertainty
# ----------------------------

def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_true = _to_numpy(y_true).astype(int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return idx


def bootstrap_cis(
    *,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_bootstraps: int,
    ci_alpha: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Stratified bootstrap CIs for all reported metrics."""
    y_true = _to_numpy(y_true).astype(int)
    y_proba = _to_numpy(y_proba).astype(float)

    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}

    for _ in range(int(n_bootstraps)):
        idx = _stratified_bootstrap_indices(y_true, rng)
        yt = y_true[idx]
        yp = y_proba[idx]

        prob_metrics = compute_probability_metrics(yt, yp)
        thr_metrics = compute_threshold_metrics(yt, yp, threshold)

        merged = {**prob_metrics, **{k: v for k, v in thr_metrics.items() if k not in {"tp", "fp", "tn", "fn"}}}

        for k, v in merged.items():
            samples.setdefault(k, []).append(float(v))

    lo = 100.0 * (ci_alpha / 2.0)
    hi = 100.0 * (1.0 - ci_alpha / 2.0)

    ci: Dict[str, Dict[str, float]] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "lower": float(np.percentile(arr, lo)),
            "upper": float(np.percentile(arr, hi)),
        }
    return ci


# ----------------------------
# Pairwise tests
# ----------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    midranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        midranks[i:j] = mid
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def delong_roc_test(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> Dict[str, float]:
    """DeLong test for ROC-AUC difference between two correlated score vectors."""
    from scipy.stats import norm

    y_true = _to_numpy(y_true).astype(int)
    y_score_a = _to_numpy(y_score_a).astype(float)
    y_score_b = _to_numpy(y_score_b).astype(float)

    order = np.argsort(-y_true)  # positives first
    y_true_sorted = y_true[order]
    preds = np.vstack([y_score_a[order], y_score_b[order]])
    label_1_count = int(np.sum(y_true_sorted))

    aucs, cov = _fast_delong(preds, label_1_count)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])

    diff = auc_a - auc_b
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = np.sqrt(max(var, 1e-12))
    z = diff / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))

    return {"auc_a": auc_a, "auc_b": auc_b, "delta": float(diff), "z": float(z), "p_value": float(p)}


def paired_bootstrap_pvalue(
    *,
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int,
    seed: int,
) -> float:
    """Paired, stratified bootstrap p-value for a metric difference."""
    y_true = _to_numpy(y_true).astype(int)
    y_score_a = _to_numpy(y_score_a).astype(float)
    y_score_b = _to_numpy(y_score_b).astype(float)

    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(int(n_bootstraps)):
        idx = _stratified_bootstrap_indices(y_true, rng)
        yt = y_true[idx]
        a = y_score_a[idx]
        b = y_score_b[idx]
        diffs.append(metric_fn(yt, a) - metric_fn(yt, b))

    diffs = np.asarray(diffs, dtype=float)
    p_le = np.mean(diffs <= 0.0)
    p_ge = np.mean(diffs >= 0.0)
    return float(min(2.0 * min(p_le, p_ge), 1.0))


# ----------------------------
# Plots
# ----------------------------

def plot_roc_curves(y_true: np.ndarray, proba_by_model: Dict[str, np.ndarray], config: Phase7Config) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Phase 7)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return save_figure(fig, "phase7_roc_curves", config.figures_subdir)


def plot_pr_curves(y_true: np.ndarray, proba_by_model: Dict[str, np.ndarray], config: Phase7Config) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = float(np.mean(_to_numpy(y_true)))
    for name, y_proba in proba_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")

    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.6, label=f"Prevalence={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Phase 7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_figure(fig, "phase7_pr_curves", config.figures_subdir)


def plot_calibration_curves(y_true: np.ndarray, proba_by_model: Dict[str, np.ndarray], config: Phase7Config) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, y_proba in proba_by_model.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=config.calibration_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_proba)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves (Phase 7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_figure(fig, "phase7_calibration_curves", config.figures_subdir)


def _net_benefit(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    w = threshold / (1.0 - threshold)
    return (tp / n) - (fp / n) * w


def plot_decision_curves(y_true: np.ndarray, proba_by_model: Dict[str, np.ndarray], config: Phase7Config) -> Path:
    lo, hi, n = config.decision_curve_thresholds
    thresholds = np.linspace(lo, hi, int(n))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, np.zeros_like(thresholds), "k--", alpha=0.6, label="Treat none")

    prevalence = float(np.mean(_to_numpy(y_true)))
    treat_all = []
    for t in thresholds:
        w = t / (1.0 - t)
        treat_all.append(prevalence - (1 - prevalence) * w)
    ax.plot(thresholds, treat_all, "k:", alpha=0.8, label="Treat all")

    for name, y_proba in proba_by_model.items():
        nb = [_net_benefit(y_true, y_proba, t) for t in thresholds]
        ax.plot(thresholds, nb, linewidth=2, label=name)

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis (Phase 7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_figure(fig, "phase7_decision_curves", config.figures_subdir)


# ----------------------------
# Model discovery/loading
# ----------------------------

def _latest_file_in_dir(
    directory: Path,
    suffixes: Tuple[str, ...] = (".joblib", ".pkl"),
    *,
    require_phase: Optional[int] = None,
    include_substr: Optional[str] = None,
    exclude_substr: Optional[str] = None,
) -> Optional[Path]:
    if not directory.exists():
        return None

    candidates = sorted(
        [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix in suffixes
            and "metadata" not in p.name
            and (include_substr is None or include_substr in p.stem)
            and (exclude_substr is None or exclude_substr not in p.stem)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        return None

    # If we don't care about phase, just take the newest model file.
    if require_phase is None:
        return candidates[0]

    # If we do care about phase, scan newest->oldest and return the newest
    # candidate whose metadata declares the required phase.
    for p in candidates:
        md_path = p.with_name(p.stem + "_metadata.json")
        if not md_path.exists():
            continue

        try:
            # Quiet read: avoids duplicate "Metadata loaded ..." logs during discovery
            with open(md_path, "r", encoding="utf-8") as f:
                md = json.load(f)
        except Exception:
            continue

        phase_val = md.get("phase")
        try:
            if phase_val is not None and int(phase_val) == int(require_phase):
                return p
        except Exception:
            # If phase isn't parseable, ignore this candidate.
            continue

    return None


def load_models_for_phase7(
    *,
    baseline_results: Optional[Dict[str, Any]] = None,
    tuned_results: Optional[Dict[str, Any]] = None,
    include_calibrated: bool = True,
    include_uncalibrated: bool = True,
    config: Optional[Phase7Config] = None,
):
    """
    Build a model dict + thresholds map for Phase 7.

    Returns:
        (models, thresholds, metadata_map, path_map)
    """
    cfg = config or Phase7Config()

    models: Dict[str, Any] = {}
    thresholds: Dict[str, float] = {}
    metadata_map: Dict[str, Dict[str, Any]] = {}
    path_map: Dict[str, str] = {}

    # baseline
    if baseline_results and "model" in baseline_results:
        models["baseline_logistic"] = baseline_results["model"]
        md = {"threshold_optimisation": baseline_results.get("threshold_optimisation"), "chosen_threshold": None}
        metadata_map["baseline_logistic"] = md
        thresholds["baseline_logistic"] = choose_threshold(explicit_threshold=None, metadata=md, config=cfg, model_name="baseline_logistic")
        if baseline_results.get("model_path") is not None:
            path_map["baseline_logistic"] = str(baseline_results["model_path"])
    else:
        latest = _latest_file_in_dir(Path(BASELINE_MODELS_DIR))
        if latest is not None:
            models["baseline_logistic"] = load_model(latest)
            path_map["baseline_logistic"] = str(latest)
            md_path = latest.with_name(latest.stem + "_metadata.json")
            md = load_model_metadata(md_path) if md_path.exists() else {}
            metadata_map["baseline_logistic"] = md
            thresholds["baseline_logistic"] = choose_threshold(explicit_threshold=None, metadata=md, config=cfg, model_name="baseline_logistic")

    # tuned models (from Phase 6 return or auto-discover on disk)
    if tuned_results:
        for model_name, pack in tuned_results.items():
            base_path = pack.get("model_path")
            cal_path = pack.get("calibrated_model_path")

            # Prefer the correctly-named metadata keys produced by ModelTrainer
            md_uncal = pack.get("metadata_uncalibrated") or pack.get("metadata") or {}
            md_cal = pack.get("metadata_calibrated") or pack.get("metadata") or md_uncal

            if include_uncalibrated and base_path:
                key = f"{model_name}_tuned"
                models[key] = load_model(base_path)
                metadata_map[key] = md_uncal
                thresholds[key] = choose_threshold(
                    explicit_threshold=None, metadata=md_uncal, config=cfg, model_name=key
                )
                path_map[key] = str(base_path)

            if include_calibrated and cal_path:
                key = f"{model_name}_tuned_calibrated"
                models[key] = load_model(cal_path)
                metadata_map[key] = md_cal
                thresholds[key] = choose_threshold(
                    explicit_threshold=None, metadata=md_cal, config=cfg, model_name=key
                )
                path_map[key] = str(cal_path)

    else:
        tuned_root = Path(TUNED_MODELS_DIR)
        if tuned_root.exists():
            for model_dir in sorted([p for p in tuned_root.iterdir() if p.is_dir()]):

                if include_uncalibrated:
                    latest = _latest_file_in_dir(
                        model_dir,
                        require_phase=6,
                        exclude_substr="calibrated",
                    )
                    if latest is not None:
                        key = f"{model_dir.name}_tuned"
                        models[key] = load_model(latest)
                        path_map[key] = str(latest)
                        md_path = latest.with_name(latest.stem + "_metadata.json")
                        md = load_model_metadata(md_path) if md_path.exists() else {}
                        metadata_map[key] = md
                        thresholds[key] = choose_threshold(explicit_threshold=None, metadata=md, config=cfg, model_name=key)

                if include_calibrated:
                    latest_cal = _latest_file_in_dir(
                        model_dir,
                        require_phase=6,
                        include_substr="calibrated",
                    )

                    if latest_cal is not None:
                        key = f"{model_dir.name}_tuned_calibrated"
                        models[key] = load_model(latest_cal)
                        path_map[key] = str(latest_cal)
                        md_path = latest_cal.with_name(latest_cal.stem + "_metadata.json")
                        md = load_model_metadata(md_path) if md_path.exists() else {}
                        metadata_map[key] = md
                        thresholds[key] = choose_threshold(explicit_threshold=None, metadata=md, config=cfg, model_name=key)

    return models, thresholds, metadata_map, path_map


def decision_curve_table(y_true: np.ndarray, proba_by_model: Dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        row = {"threshold": float(t)}
        for name, p in proba_by_model.items():
            row[name] = float(_net_benefit(y_true, p, float(t)))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_and_compare_models(
    *,
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: Optional[Dict[str, float]] = None,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Phase7Config] = None,
) -> Dict[str, Any]:
    """
    Run Phase 7 evaluation:
    - computes per-model metrics (probability + threshold-based)
    - plots ROC/PR/calibration (+ optional decision curves)
    - saves comparison tables
    - optional bootstrap CIs and pairwise tests
    """
    cfg = config or Phase7Config()
    y_true = _to_numpy(y).astype(int)

    thresholds = thresholds or {}
    metadata_map = metadata_map or {}

    proba_by_model: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []

    calibration_bins_rows = []
    threshold_docs = []

    module_logger.info(f"Phase 7: evaluating {len(models)} model(s) on {len(y_true)} samples")

    for name, model in models.items():
        y_proba = safe_predict_proba(model, X)
        proba_by_model[name] = y_proba

        cal_bins_uniform = calibration_ece_mce(y_true, y_proba, n_bins=cfg.calibration_bins, strategy="uniform")["bins_df"]
        cal_bins_uniform["model"] = name
        cal_bins_uniform["binning"] = "uniform"

        cal_bins_quant = calibration_ece_mce(y_true, y_proba, n_bins=cfg.calibration_bins, strategy="quantile")["bins_df"]
        cal_bins_quant["model"] = name
        cal_bins_quant["binning"] = "quantile"

        # collect for saving later
        calibration_bins_rows.append(cal_bins_uniform)
        calibration_bins_rows.append(cal_bins_quant)

        thr, thr_doc = resolve_threshold_with_doc(
            explicit_threshold=thresholds.get(name),
            metadata=metadata_map.get(name),
            config=cfg,
            model_name=name,
        )

        prob_metrics = compute_probability_metrics(y_true, y_proba)
        thr_metrics = compute_threshold_metrics(y_true, y_proba, thr)

        rows.append({
            "model": name,
            **prob_metrics,
            **thr_metrics,
            "threshold_source": thr_doc["source"],
            "threshold_field": thr_doc["field"],
            "threshold_strategy": thr_doc["strategy"],
            "threshold_strategy_score": thr_doc["score"],
        })
        threshold_docs.append(thr_doc)

        if cfg.bootstrap:
            cis = bootstrap_cis(
                y_true=y_true,
                y_proba=y_proba,
                threshold=thr,
                n_bootstraps=cfg.n_bootstraps,
                ci_alpha=cfg.ci_alpha,
                seed=cfg.random_state,
            )
            for metric, pack in cis.items():
                ci_rows.append({"model": name, "metric": metric, **pack})

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    calibration_bins_path = None
    if calibration_bins_rows:
        cal_bins_df = pd.concat(calibration_bins_rows, ignore_index=True)
        calibration_bins_path = TABLES_DIR / f"phase7_calibration_bins_{timestamp}.csv"
        cal_bins_df.to_csv(calibration_bins_path, index=False)

    metrics_csv = TABLES_DIR / f"phase7_model_comparison_{timestamp}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    threshold_doc_path = TABLES_DIR / f"phase7_threshold_docs_{timestamp}.json"
    with open(threshold_doc_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(threshold_docs), f, indent=2)

    summary_json = TABLES_DIR / f"phase7_model_comparison_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(metrics_df.to_dict(orient="records")), f, indent=2)

    ci_csv = None
    ci_df = None
    if cfg.bootstrap and ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        ci_csv = TABLES_DIR / f"phase7_bootstrap_cis_{timestamp}.csv"
        ci_df.to_csv(ci_csv, index=False)

    fig_paths: Dict[str, str] = {}
    if cfg.plot_roc and proba_by_model:
        fig_paths["roc_curves"] = str(plot_roc_curves(y_true, proba_by_model, cfg))
    if cfg.plot_pr and proba_by_model:
        fig_paths["pr_curves"] = str(plot_pr_curves(y_true, proba_by_model, cfg))
    if cfg.plot_calibration and proba_by_model:
        fig_paths["calibration_curves"] = str(plot_calibration_curves(y_true, proba_by_model, cfg))
    if cfg.plot_decision_curves and proba_by_model:
        fig_paths["decision_curves"] = str(plot_decision_curves(y_true, proba_by_model, cfg))

    dca_path = None
    if cfg.plot_decision_curves and proba_by_model:
        lo, hi, n = cfg.decision_curve_thresholds
        thr_grid = np.linspace(lo, hi, int(n))
        dca_df = decision_curve_table(y_true, proba_by_model, thr_grid)
        dca_path = TABLES_DIR / f"phase7_decision_curve_data_{timestamp}.csv"
        dca_df.to_csv(dca_path, index=False)

    pairwise_paths: Dict[str, str] = {}
    if cfg.pairwise_tests and len(proba_by_model) >= 2:
        names = list(proba_by_model.keys())
        auc_tests = []
        ap_tests = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                try:
                    d = delong_roc_test(y_true, proba_by_model[a], proba_by_model[b])
                    auc_tests.append({"model_a": a, "model_b": b, **d})
                except Exception as e:
                    module_logger.warning(f"DeLong test failed for {a} vs {b}: {e}")

                try:
                    p_ap = paired_bootstrap_pvalue(
                        y_true=y_true,
                        y_score_a=proba_by_model[a],
                        y_score_b=proba_by_model[b],
                        metric_fn=lambda yt, yp: float(average_precision_score(yt, yp)),
                        n_bootstraps=min(cfg.n_bootstraps, 500),
                        seed=cfg.random_state,
                    )
                    ap_tests.append({
                        "model_a": a,
                        "model_b": b,
                        "ap_a": float(average_precision_score(y_true, proba_by_model[a])),
                        "ap_b": float(average_precision_score(y_true, proba_by_model[b])),
                        "delta": float(average_precision_score(y_true, proba_by_model[a]) -
                                       average_precision_score(y_true, proba_by_model[b])),
                        "p_value": float(p_ap),
                        "method": "paired_stratified_bootstrap",
                    })
                except Exception as e:
                    module_logger.warning(f"AP bootstrap test failed for {a} vs {b}: {e}")

        if auc_tests:
            auc_df = pd.DataFrame(auc_tests)
            auc_path = TABLES_DIR / f"phase7_pairwise_auc_delong_{timestamp}.csv"
            auc_df.to_csv(auc_path, index=False)
            pairwise_paths["auc_delong"] = str(auc_path)

        if ap_tests:
            ap_df = pd.DataFrame(ap_tests)
            ap_path = TABLES_DIR / f"phase7_pairwise_ap_bootstrap_{timestamp}.csv"
            ap_df.to_csv(ap_path, index=False)
            pairwise_paths["ap_bootstrap"] = str(ap_path)

    winner = None
    if not metrics_df.empty:
        metrics_df["_rank_key"] = metrics_df["roc_auc"] - 0.01 * metrics_df["brier_score"]
        winner = str(metrics_df.sort_values("_rank_key", ascending=False).iloc[0]["model"])
        metrics_df.drop(columns=["_rank_key"], inplace=True, errors="ignore")

    return {
        "metrics_table_path": str(metrics_csv),
        "metrics_json_path": str(summary_json),
        "bootstrap_ci_path": str(ci_csv) if ci_csv else None,
        "pairwise_test_paths": pairwise_paths,
        "figure_paths": fig_paths,
        "winner_model": winner,
        "metrics_df": metrics_df,
        "ci_df": ci_df,
        "calibration_bins_path": str(calibration_bins_path) if calibration_bins_path else None,
        "decision_curve_data_path": str(dca_path) if dca_path else None,
        "threshold_docs_path": str(threshold_doc_path)
    }

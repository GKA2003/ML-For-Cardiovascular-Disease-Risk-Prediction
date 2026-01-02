"""src.interpretation

Phase 8: Feature importance and model interpretation.

Deliverables implemented:
1) Phase 8 header table anchored to Phase 7 headline metrics for the final model set.
2) SHAP global explanations (TreeExplainer) for the primary model:
   extra_trees_tuned_calibrated
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import textwrap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import sparse as sp

from src.config import (
    RANDOM_SEED,
    TABLES_DIR,
    PHASE8_FINAL_MODEL_SET,
    PHASE8_PRIMARY_MODEL,
    PHASE8_SHAP_BACKGROUND_SIZE,
    PHASE8_SHAP_TOP_N,
    MODEL_CARDS_DIR,
    PHASE8_CALIBRATION_BINS,
    PHASE8_RISK_DECILES,
    PHASE8_LOCAL_HIGH,
    PHASE8_LOCAL_LOW,
    PHASE8_LOCAL_BORDERLINE,
    PHASE8_LOCAL_BORDERLINE_FALLBACK,
    PHASE8_LOCAL_TOP_CONTRIB,
    PHASE8_BOOTSTRAP, PHASE8_SUBGROUP_MIN,
    PHASE8_STABILITY_TOP_K, PHASE8_CATBOOST_EXPLAIN,
    PHASE8_CROSSMODEL_TOP_K, TARGET_COLUMN,
    PHASE7_EXPECTED_TEST_SIZE, PHASE8_ALL_ZERO_ALLOWLIST,
    PHASE8_SUMMARY_TOP_N
)
from src.utils import save_figure

logger = logging.getLogger(__name__)

PHASE7_METRICS_PREFIX = "phase7_model_comparison_"


def _latest_phase7_metrics_csv(tables_dir: Path = TABLES_DIR) -> Optional[Path]:
    candidates = sorted(tables_dir.glob(f"{PHASE7_METRICS_PREFIX}*.csv"))
    return candidates[-1] if candidates else None


def create_phase8_header_table(
    *,
    final_models: List[str] = PHASE8_FINAL_MODEL_SET,
    phase7_metrics_path: Optional[Path] = None,
    out_dir: Path = TABLES_DIR,
) -> Path:
    """
    Create the Phase 8 header table from the Phase 7 comparison CSV.

    Columns included (if present in Phase 7 output):
      AUC, AP, Brier, calibration intercept/slope, chosen threshold (+ threshold_strategy)
    """
    phase7_metrics_path = phase7_metrics_path or _latest_phase7_metrics_csv(out_dir)
    if phase7_metrics_path is None or not phase7_metrics_path.exists():
        raise FileNotFoundError(
            "No Phase 7 model comparison table found in reports/tables/. "
            "Run Phase 7 first (or pass phase7_metrics_path explicitly)."
        )

    df = pd.read_csv(phase7_metrics_path)

    want_cols = [
        "model",
        "roc_auc",
        "average_precision",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
        "threshold",
        "threshold_strategy",
    ]
    cols = [c for c in want_cols if c in df.columns]

    header = (
        df[df["model"].isin(final_models)][cols]
        .copy()
        .sort_values("model")
        .reset_index(drop=True)
    )

    missing = [m for m in final_models if m not in set(header["model"].tolist())]
    if missing:
        raise ValueError(
            "Phase 7 metrics CSV is missing some Phase 8 final models: " + ", ".join(missing)
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"phase8_header_metrics_{timestamp}.csv"
    header.to_csv(out_path, index=False)
    logger.info(f"Phase 8 header table saved to: {out_path}")
    return out_path

#SHAP helpers
def _is_pipeline(obj: Any) -> bool:
    return isinstance(obj, (Pipeline, ImbPipeline))


def _unwrap_calibrated_estimator(model: Any) -> Any:
    """
    Best-effort unwrap for CalibratedClassifierCV.

    Note: for cv != 'prefit', CalibratedClassifierCV stores multiple fitted
    estimators (one per fold). We pick fold 0's fitted estimator as a lightweight
    representative so TreeExplainer works.
    """
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            return model.calibrated_classifiers_[0].estimator
    return model


def _extract_preprocess_and_model_steps(estimator: Any) -> Tuple[Optional[Any], Any]:
    """
    Return (preprocess_step, final_model_estimator).

    Works with Phase 6 pipelines that look like:
      preprocess -> to_numpy -> (optional sampler) -> model
    """
    if _is_pipeline(estimator) and hasattr(estimator, "named_steps"):
        steps = estimator.named_steps
        preprocess = steps.get("preprocess")
        model_step = steps.get("model") or steps.get("clf") or steps.get("estimator")
        if model_step is None:
            model_step = list(steps.values())[-1]
        return preprocess, model_step
    return None, estimator


def _transform_X(preprocess: Optional[Any], X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transform X into the feature space expected by the downstream model."""
    if preprocess is None:
        return np.asarray(X), list(X.columns)

    Xt = preprocess.transform(X)
    if sp is not None and hasattr(sp, "issparse") and sp.issparse(Xt):
        Xt = Xt.toarray()

    try:
        feature_names = list(preprocess.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(np.asarray(Xt).shape[1])]

    return np.asarray(Xt), feature_names


def _ensure_shap_matrix(shap_values: Any) -> np.ndarray:
    """
    Normalise SHAP output to a 2D array (n_samples, n_features).

    Handles:
      - list of per-class arrays (uses positive class if available)
      - 3D arrays (n_samples, n_features, n_classes) -> select positive class
      - shap.Explanation objects
    """
    # shap.Explanation -> raw values
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # list of class-wise arrays -> pick class 1 if possible
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        arr = np.asarray(shap_values)

    # 3D -> choose positive class (index 1 if exists)
    if arr.ndim == 3:
        class_idx = 1 if arr.shape[-1] > 1 else 0
        arr = arr[..., class_idx]

    # Safety: enforce 2D
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)

    return arr

def _safe_filename(s: str) -> str:
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
    )

def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """
    Calibration-in-the-large (intercept) and calibration slope via:
      logit(p) -> logistic regression predicting y
    Ideal: intercept ~ 0, slope ~ 1.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    if len(np.unique(y)) < 2:
        return (np.nan, np.nan)

    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)

    # no-penalty if available, otherwise approximate with huge C
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    except Exception:
        lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=2000)

    lr.fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])

@dataclass(frozen=True)
class Phase8ShapResult:
    model_name: str
    beeswarm_path: Path
    bar_path: Path
    mean_abs_csv_path: Path
    shap_values: Optional[np.ndarray] = None
    X_test_transformed: Optional[pd.DataFrame] = None
    feature_names: Optional[List[str]] = None
    expected_value: Optional[float] = None

@dataclass(frozen=True)
class Phase8PermutationImportanceResult:
    model_name: str
    csv_path: Path
    bar_path: Path
    overlap_csv_path: Optional[Path]

def generate_phase8_shap_global_for_primary(
    *,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = PHASE8_PRIMARY_MODEL,
    background_size: int = PHASE8_SHAP_BACKGROUND_SIZE,
    top_n: int = PHASE8_SHAP_TOP_N,
    figures_subdir: str = "interpretation",
    out_tables_dir: Path = TABLES_DIR,
) -> Phase8ShapResult:
    """
    Generate:
      - SHAP beeswarm (top N)
      - SHAP bar (mean |SHAP|, top N)
      - CSV of mean(|SHAP|) for *all* features, ranked

    Compute kept light:
      background: 200–500-ish rows sampled from X_train (default 300)
      explain: all test rows (expected ~678 from Phase 7)
    """
    if len(X_test) == 0:
        raise ValueError("X_test is empty; cannot explain.")

    estimator = _unwrap_calibrated_estimator(model)
    preprocess, tree_model = _extract_preprocess_and_model_steps(estimator)

    bg_n = int(min(max(background_size, 1), len(X_train)))
    X_bg = X_train.sample(n=bg_n, random_state=RANDOM_SEED)

    X_bg_t, feat_names = _transform_X(preprocess, X_bg)
    X_test_t, _ = _transform_X(preprocess, X_test)

    X_bg_df = pd.DataFrame(X_bg_t, columns=feat_names)
    X_test_df = pd.DataFrame(X_test_t, columns=feat_names)

    explainer = shap.TreeExplainer(tree_model, data=X_bg_df, model_output="probability")
    shap_vals = _ensure_shap_matrix(explainer.shap_values(X_test_df))

    exp = explainer.expected_value
    if isinstance(exp, (list, np.ndarray)):
        exp = exp[1] if len(exp) > 1 else exp[0]
    exp = float(exp)

    # Beeswarm
    shap.summary_plot(shap_vals, X_test_df, show=False, max_display=top_n)
    beeswarm_path = save_figure(
        plt.gcf(), f"phase8_shap_beeswarm_top{top_n}_{model_name}", figures_subdir
    )

    # Bar
    shap.summary_plot(shap_vals, X_test_df, plot_type="bar", show=False, max_display=top_n)
    bar_path = save_figure(
        plt.gcf(), f"phase8_shap_bar_top{top_n}_{model_name}", figures_subdir
    )

    # mean(|SHAP|) table
    mean_abs = np.mean(np.abs(shap_vals), axis=0).reshape(-1)

    out_df = pd.DataFrame(
        {"feature": feat_names, "mean_abs_shap": mean_abs.astype(float)}
    ).sort_values("mean_abs_shap", ascending=False)

    out_df.insert(0, "rank", np.arange(1, len(out_df) + 1))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mean_abs_csv = out_tables_dir / f"phase8_shap_mean_abs_{model_name}_{ts}.csv"
    out_df.to_csv(mean_abs_csv, index=False)

    logger.info(f"Phase 8 SHAP outputs saved: {beeswarm_path}, {bar_path}, {mean_abs_csv}")
    return Phase8ShapResult(
        model_name=model_name,
        beeswarm_path=beeswarm_path,
        bar_path=bar_path,
        mean_abs_csv_path=mean_abs_csv,
        shap_values=shap_vals,
        X_test_transformed=X_test_df,
        feature_names=feat_names,
        expected_value=exp,
    )

def generate_phase8_permutation_importance_brier(
    *,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_repeats: int,
    top_n: int,
    figures_subdir: str = "interpretation",
    out_tables_dir: Path = TABLES_DIR,
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Permutation importance using Brier score (risk prediction).

    We optimise calibrated probability quality, so we use Brier.
    sklearn.permutation_importance expects "higher is better", so we score as NEGATIVE brier:
        score = -brier_score_loss(y, p)
    Then permutation importance (decrease in score) corresponds to an increase in Brier.
    """
    def neg_brier(estimator, X, y):
        p = estimator.predict_proba(X)[:, 1]
        return -brier_score_loss(y, p)

    pi = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring=neg_brier,
        n_repeats=int(n_repeats),
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    feat_names = list(X_test.columns)
    df = pd.DataFrame(
        {
            "feature": feat_names,
            # This is "increase in Brier" (because it's decrease in -Brier)
            "brier_increase_mean": pi.importances_mean.astype(float),
            "brier_increase_std": pi.importances_std.astype(float),
        }
    ).sort_values("brier_increase_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    # Required exact output name (no timestamp)
    csv_path = out_tables_dir / f"phase8_perm_importance_{model_name}.csv"
    df.to_csv(csv_path, index=False)

    # Bar plot (top N)
    top = df.head(int(top_n)).iloc[::-1]  # reverse for horizontal bar plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(top["feature"], top["brier_increase_mean"], xerr=top["brier_increase_std"])
    ax.set_xlabel("Increase in Brier score when permuted (higher = more important)")
    ax.set_title(f"Permutation importance (Brier) - top {top_n}\n{model_name}")

    bar_path = save_figure(fig, f"phase8_perm_importance_bar_top{top_n}_{model_name}", figures_subdir)
    logger.info(f"Permutation importance saved: {csv_path} and {bar_path}")

    return df, csv_path, bar_path


def create_phase8_shap_perm_overlap_table(
    *,
    shap_mean_abs_csv: Path,
    perm_importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 20,
    out_tables_dir: Path = TABLES_DIR,
) -> Path:
    """
    Convenience table for write-up: overlap between SHAP top-N and permutation top-N.
    """
    shap_df = pd.read_csv(shap_mean_abs_csv)
    shap_top = shap_df.head(top_n)[["rank", "feature", "mean_abs_shap"]].copy()
    shap_top = shap_top.rename(columns={"rank": "shap_rank", "mean_abs_shap": "shap_mean_abs"})

    perm_top = perm_importance_df.head(top_n)[["rank", "feature", "brier_increase_mean"]].copy()
    perm_top = perm_top.rename(columns={"rank": "perm_rank", "brier_increase_mean": "perm_brier_increase_mean"})

    shap_set = set(shap_top["feature"])
    perm_set = set(perm_top["feature"])
    overlap = shap_set.intersection(perm_set)

    # Long-form table: one row per feature in the union
    union = sorted(list(shap_set.union(perm_set)))
    out = pd.DataFrame({"feature": union})
    out["in_shap_topN"] = out["feature"].isin(shap_set)
    out["in_perm_topN"] = out["feature"].isin(perm_set)

    out = out.merge(shap_top, on="feature", how="left").merge(perm_top, on="feature", how="left")
    out = out.sort_values(
        by=["in_shap_topN", "in_perm_topN", "shap_rank", "perm_rank"],
        ascending=[False, False, True, True],
    )

    out_path = out_tables_dir / f"phase8_shap_perm_overlap_top{top_n}_{model_name}.csv"
    out.to_csv(out_path, index=False)

    logger.info(
        f"SHAP vs permutation overlap (top{top_n}): {len(overlap)} shared features. Table: {out_path}"
    )
    return out_path

def generate_phase8_shap_dependence_plots(
    *,
    shap_values: np.ndarray,
    X_test_transformed: pd.DataFrame,
    shap_mean_abs_csv: Path,
    model_name: str,
    top_k: int,
    max_interaction_plots: int,
    figures_subdir: str = "interpretation",
) -> list[Path]:
    """
    Dependence plots for top drivers (6–8 features).
    Also creates up to 1–2 interaction-focused plots ONLY if they look meaningful:
      - requires both features to have enough unique values to show a gradient
    """
    shap_rank_df = pd.read_csv(shap_mean_abs_csv)
    top_features = shap_rank_df["feature"].head(int(top_k)).tolist()

    out_paths: list[Path] = []

    # 1) Main dependence plots (interaction_index='auto' gives useful colouring cheaply)
    for feat in top_features:
        if feat not in X_test_transformed.columns:
            continue
        shap.dependence_plot(
            feat,
            shap_values,
            X_test_transformed,
            interaction_index="auto",
            show=False,
        )
        fig = plt.gcf()
        out_paths.append(
            save_figure(fig, f"phase8_shap_dependence_{model_name}", figures_subdir)
        )

    # 2) Up to 1–2 explicit interaction plots, only if meaningful
    try:
        feature_names = list(X_test_transformed.columns)
        made = 0
        for feat in top_features:
            if made >= int(max_interaction_plots):
                break
            if feat not in feature_names:
                continue

            i = feature_names.index(feat)
            # Pick top suggested interaction partner
            inter_order = shap.utils.approximate_interactions(i, shap_values, X_test_transformed)
            if len(inter_order) == 0:
                continue
            j = int(inter_order[0])
            inter_feat = feature_names[j]

            # “Meaningful” gate: avoid binary/near-constant features that won't show structure
            if X_test_transformed[feat].nunique() < 10 or X_test_transformed[inter_feat].nunique() < 10:
                continue

            shap.dependence_plot(
                feat,
                shap_values,
                X_test_transformed,
                interaction_index=inter_feat,
                show=False,
            )
            fig = plt.gcf()
            out_paths.append(
                save_figure(
                    fig,
                    f"phase8_shap_dependence_interaction_{model_name}",
                    figures_subdir,
                )
            )
            made += 1
    except Exception as e:
        logger.warning(f"Skipping interaction-focused dependence plots due to error: {e}")

    logger.info(f"Saved {len(out_paths)} SHAP dependence plots for {model_name}")
    return out_paths

def generate_phase8_pdp_ice(
    *,
    model: Any,
    X_test: pd.DataFrame,
    features: list[str],
    model_name: str,
    figures_subdir: str = "interpretation",
) -> list[Path]:
    """
    Optional classical view: PDP + ICE on probability of class 1.
    Kept small (2–3 features max).
    """
    out_paths: list[Path] = []
    for feat in features:
        if feat not in X_test.columns:
            continue
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            PartialDependenceDisplay.from_estimator(
                model,
                X_test,
                features=[feat],
                kind="both",                  # PDP + ICE
                response_method="predict_proba",
                target=1,
                ax=ax,
            )
            ax.set_title(f"PDP/ICE - {feat}\n{model_name}")
            out_paths.append(
                save_figure(fig, f"phase8_pdp_ice_{model_name}_{_safe_filename(feat)}", figures_subdir)
            )
        except Exception as e:
            logger.warning(f"Skipping PDP/ICE for {feat}: {e}")
    return out_paths

def generate_phase8_risk_deciles(
    *,
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    n_deciles: int = PHASE8_RISK_DECILES,
    figures_subdir: str = "interpretation",
    out_tables_dir: Path = TABLES_DIR,
) -> tuple[Path, Path]:
    """
    Deciles (quantiles) of predicted risk on the test split.
    Outputs:
      - CSV: n, mean_predicted_risk, observed_event_rate (per decile)
      - Plot: mean predicted vs observed rate across deciles (+ counts annotated)
    """
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true).astype(int), "y_proba": np.asarray(y_proba).astype(float)}
    )

    # qcut can drop bins if too many duplicate probs; keep robust
    df["decile"] = pd.qcut(df["y_proba"], q=int(n_deciles), labels=False, duplicates="drop") + 1

    g = df.groupby("decile", as_index=False).agg(
        n=("y_true", "size"),
        mean_predicted_risk=("y_proba", "mean"),
        observed_event_rate=("y_true", "mean"),
        min_predicted=("y_proba", "min"),
        max_predicted=("y_proba", "max"),
    )

    # fixed name for reproducibility (like permutation importance)
    csv_path = out_tables_dir / f"phase8_risk_deciles_{model_name}.csv"
    g.to_csv(csv_path, index=False)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(g["decile"], g["mean_predicted_risk"], marker="o", label="Mean predicted risk")
    ax.plot(g["decile"], g["observed_event_rate"], marker="o", label="Observed event rate")
    ax.set_xlabel("Risk decile (1 = lowest predicted risk)")
    ax.set_ylabel("Risk / event rate")
    ax.set_title(f"Risk deciles (test split) - {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # annotate counts
    for _, r in g.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["decile"], r["observed_event_rate"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    plot_path = save_figure(fig, f"phase8_risk_deciles_plot_{model_name}", figures_subdir)
    logger.info(f"Risk deciles saved: {csv_path} and {plot_path}")

    return csv_path, plot_path

def plot_phase8_reliability_with_counts(
    *,
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    n_bins: int = PHASE8_CALIBRATION_BINS,
    figures_subdir: str = "interpretation",
    out_tables_dir: Path = TABLES_DIR,
) -> tuple[Path, Path]:
    """
    Reliability diagram using uniform probability bins, with bin counts annotated.
    Also writes a bin summary CSV for write-up/debug.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # bin index 0..n_bins-1
    bin_idx = np.digitize(p, bins, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, int(n_bins) - 1)

    rows = []
    for b in range(int(n_bins)):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {"bin": b + 1, "n": 0, "mean_pred": np.nan, "frac_pos": np.nan, "bin_low": bins[b], "bin_high": bins[b + 1]}
            )
            continue
        rows.append(
            {
                "bin": b + 1,
                "n": n,
                "mean_pred": float(p[mask].mean()),
                "frac_pos": float(y[mask].mean()),
                "bin_low": float(bins[b]),
                "bin_high": float(bins[b + 1]),
            }
        )

    df_bins = pd.DataFrame(rows)
    bins_csv = out_tables_dir / f"phase8_reliability_bins_{model_name}.csv"
    df_bins.to_csv(bins_csv, index=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    plot_df = df_bins.dropna(subset=["mean_pred", "frac_pos"])
    ax.plot(plot_df["mean_pred"], plot_df["frac_pos"], marker="o", linewidth=2, label=model_name)

    # annotate counts at each point
    for _, r in plot_df.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_pred"], r["frac_pos"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    brier = brier_score_loss(y, p)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed event rate (bin)")
    ax.set_title(f"Reliability diagram (test split) - {model_name}\nBrier={brier:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_path = save_figure(fig, f"phase8_reliability_{model_name}", figures_subdir)
    logger.info(f"Reliability diagram saved: {fig_path} (bins CSV: {bins_csv})")

    return bins_csv, fig_path

def select_phase8_local_cases(
    *,
    X_test: pd.DataFrame,
    y_proba: np.ndarray,
    model_threshold: Optional[float],
    model_name: str,
) -> pd.DataFrame:
    """
    Select reproducible cases from the test split:
      - top N high risk
      - bottom N low risk
      - N borderline near threshold (or fallback)
    Returns a dataframe with test-row indices and predicted probabilities.
    """
    proba = pd.Series(np.asarray(y_proba).astype(float), index=X_test.index, name="predicted_proba")

    n_high = int(PHASE8_LOCAL_HIGH)
    n_low = int(PHASE8_LOCAL_LOW)
    n_border = int(PHASE8_LOCAL_BORDERLINE)

    high = proba.sort_values(ascending=False).head(n_high)
    low = proba.sort_values(ascending=True).head(n_low)

    ref = float(model_threshold) if model_threshold is not None else float(PHASE8_LOCAL_BORDERLINE_FALLBACK)

    remaining = proba.drop(index=high.index.union(low.index))
    border = (remaining - ref).abs().sort_values(ascending=True).head(n_border)

    rows = []
    for i, (idx, p) in enumerate(high.items(), start=1):
        rows.append({"case_group": "highest_risk", "case_rank": i, "row_index": int(idx), "predicted_proba": float(p), "reference_threshold": ref})
    for i, (idx, p) in enumerate(low.items(), start=1):
        rows.append({"case_group": "lowest_risk", "case_rank": i, "row_index": int(idx), "predicted_proba": float(p), "reference_threshold": ref})
    for i, (idx, p) in enumerate(border.items(), start=1):
        rows.append({"case_group": "borderline", "case_rank": i, "row_index": int(idx), "predicted_proba": float(p), "reference_threshold": ref})

    out = pd.DataFrame(rows)
    out["model"] = model_name
    return out


def _top_contributors_bullets(
    *,
    shap_row: np.ndarray,
    x_row: pd.Series,
    feature_names: List[str],
    top_k: int = PHASE8_LOCAL_TOP_CONTRIB,
) -> List[str]:
    """
    Return bullet strings for top-k SHAP contributors for a single case.
    """
    vals = np.asarray(shap_row).astype(float)
    top_idx = np.argsort(np.abs(vals))[-int(top_k):][::-1]

    bullets = []
    for j in top_idx:
        feat = feature_names[j]
        v = x_row.iloc[j]
        s = vals[j]
        direction = "↑ risk" if s > 0 else "↓ risk"
        bullets.append(f"- {feat} = {v:.4g} → SHAP {s:+.4f} ({direction})")
    return bullets


def generate_phase8_local_explanations(
    *,
    model_name: str,
    cases_df: pd.DataFrame,
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    X_test_transformed: pd.DataFrame,
    feature_names: List[str],
    expected_value: float,
    figures_subdir: str = "interpretation/local",
    out_tables_dir: Path = TABLES_DIR,
    out_model_cards_dir: Path = MODEL_CARDS_DIR,
) -> tuple[Path, List[Path], List[Path]]:
    """
    For each selected case:
      - SHAP waterfall plot
      - short markdown narrative (top contributors)
    Also saves phase8_local_cases_summary.csv (fixed name).
    """
    # map from row_index to position in test arrays
    index_to_pos = {int(idx): i for i, idx in enumerate(X_test.index.astype(int))}

    # fixed name requested
    summary_path = out_tables_dir / "phase8_local_cases_summary.csv"
    cases_df.to_csv(summary_path, index=False)

    fig_paths: List[Path] = []
    md_paths: List[Path] = []

    out_model_cards_dir.mkdir(parents=True, exist_ok=True)

    for _, row in cases_df.iterrows():
        row_idx = int(row["row_index"])
        group = str(row["case_group"])
        pred = float(row["predicted_proba"])

        if row_idx not in index_to_pos:
            logger.warning(f"Local case row_index not found in X_test index: {row_idx}")
            continue

        pos = index_to_pos[row_idx]

        shap_row = np.asarray(shap_values[pos]).astype(float)
        x_row = X_test_transformed.iloc[pos]

        # Build shap.Explanation for waterfall
        expl = shap.Explanation(
            values=shap_row,
            base_values=expected_value,
            data=x_row.values,
            feature_names=feature_names,
        )

        shap.plots.waterfall(expl, max_display=12, show=False)
        fig = plt.gcf()
        fig_path = save_figure(
            fig,
            f"phase8_local_waterfall_{model_name}_idx{row_idx}_{group}",
            figures_subdir,
        )
        fig_paths.append(fig_path)

        bullets = _top_contributors_bullets(
            shap_row=shap_row,
            x_row=x_row,
            feature_names=feature_names,
            top_k=PHASE8_LOCAL_TOP_CONTRIB,
        )

        md_path = out_model_cards_dir / f"phase8_local_case_{model_name}_idx{row_idx}_{group}.md"
        md_text = "\n".join(
            [
                f"# Phase 8 Local Case: {model_name}",
                "",
                f"- Test row index: **{row_idx}**",
                f"- Case group: **{group}**",
                f"- Predicted probability: **{pred:.4f}**",
                "",
                "## Top contributors (SHAP)",
                *bullets,
                "",
                f"Waterfall figure: {fig_path.name}",
            ]
        )
        md_path.write_text(md_text, encoding="utf-8")
        md_paths.append(md_path)

    logger.info(f"Local case summary saved: {summary_path}")
    logger.info(f"Saved {len(fig_paths)} waterfall plots and {len(md_paths)} markdown case notes.")
    return summary_path, fig_paths, md_paths

def generate_phase8_error_analysis_by_group(
    *,
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    out_tables_dir: Path = TABLES_DIR,
    figures_subdir: str = "interpretation",
    min_n: int = PHASE8_SUBGROUP_MIN,
) -> tuple[Path, Path]:
    """
    Compute performance + calibration by subgroup on the test split.
    Outputs:
      - reports/tables/phase8_error_analysis_by_group.csv (fixed name)
      - one plot: AP vs Brier for each subgroup level
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)

    # --- pick subgroup variables if present & non-constant ---
    subgroup_series: list[tuple[str, pd.Series]] = []

    # age bands
    if "age_group" in X_test.columns and X_test["age_group"].nunique() > 1:
        subgroup_series.append(("age_group", X_test["age_group"]))
    elif "age" in X_test.columns and X_test["age"].nunique() > 1:
        # clinically interpretable bins, fallback if no age_group
        bins = [0, 40, 50, 60, 70, 200]
        labels = ["<40", "40-49", "50-59", "60-69", "70+"]
        age_band = pd.cut(X_test["age"], bins=bins, labels=labels, include_lowest=True)
        if age_band.nunique() > 1:
            subgroup_series.append(("age_band", age_band))

    # sex
    for c in ["sex", "sex_encoded", "male", "gender"]:
        if c in X_test.columns and X_test[c].nunique() > 1:
            subgroup_series.append((c, X_test[c]))
            break

    # smoking
    for c in ["is_smoking_encoded", "currentSmoker", "smoker", "smoking", "is_smoker"]:
        if c in X_test.columns and X_test[c].nunique() > 1:
            subgroup_series.append((c, X_test[c]))
            break

    # diabetes
    for c in ["diabetes", "prevalentDiabetes", "diabetes_encoded"]:
        if c in X_test.columns and X_test[c].nunique() > 1:
            subgroup_series.append((c, X_test[c]))
            break

    rows = []

    # overall row
    ap_all = average_precision_score(y, p)
    brier_all = brier_score_loss(y, p)
    auc_all = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    ci_all, cs_all = _calibration_slope_intercept(y, p)

    rows.append(
        {
            "model": model_name,
            "group_var": "overall",
            "group_value": "all",
            "n": len(y),
            "event_rate": float(y.mean()),
            "mean_pred": float(p.mean()),
            "average_precision": float(ap_all),
            "brier": float(brier_all),
            "roc_auc": float(auc_all),
            "calibration_intercept": float(ci_all),
            "calibration_slope": float(cs_all),
        }
    )

    # subgroup rows
    for var, s in subgroup_series:
        s = s.astype(str)
        for val in sorted(s.unique()):
            mask = (s == val).to_numpy()
            n = int(mask.sum())
            if n < int(min_n):
                continue

            yy = y[mask]
            pp = p[mask]

            ap = average_precision_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
            brier = brier_score_loss(yy, pp) if n > 0 else np.nan
            auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
            ci, cs = _calibration_slope_intercept(yy, pp)

            rows.append(
                {
                    "model": model_name,
                    "group_var": var,
                    "group_value": val,
                    "n": n,
                    "event_rate": float(yy.mean()) if n > 0 else np.nan,
                    "mean_pred": float(pp.mean()) if n > 0 else np.nan,
                    "average_precision": float(ap) if ap == ap else np.nan,
                    "brier": float(brier) if brier == brier else np.nan,
                    "roc_auc": float(auc) if auc == auc else np.nan,
                    "calibration_intercept": float(ci) if ci == ci else np.nan,
                    "calibration_slope": float(cs) if cs == cs else np.nan,
                }
            )

    df = pd.DataFrame(rows).sort_values(["group_var", "group_value"]).reset_index(drop=True)

    # fixed name required
    csv_path = out_tables_dir / "phase8_error_analysis_by_group.csv"
    df.to_csv(csv_path, index=False)

    # one plot: AP vs Brier scatter
    plot_df = df[df["group_var"] != "overall"].dropna(subset=["average_precision", "brier"]).copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(plot_df["brier"], plot_df["average_precision"])

    for _, r in plot_df.iterrows():
        label = f"{r['group_var']}={r['group_value']} (n={int(r['n'])})"
        ax.annotate(label, (r["brier"], r["average_precision"]), textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.scatter([brier_all], [ap_all], marker="*", s=150)
    ax.annotate("overall", (brier_all, ap_all), textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("Brier (lower is better)")
    ax.set_ylabel("Average Precision (higher is better)")
    ax.set_title(f"Error analysis by subgroup (test split) - {model_name}")
    ax.grid(True, alpha=0.3)

    fig_path = save_figure(fig, f"phase8_error_analysis_by_group_{model_name}", figures_subdir)
    logger.info(f"Error analysis saved: {csv_path} and {fig_path}")

    return csv_path, fig_path

def bootstrap_phase8_shap_rank_stability(
    *,
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    n_boot: int = PHASE8_BOOTSTRAP,
    top_k: int = PHASE8_STABILITY_TOP_K,
    out_tables_dir: Path = TABLES_DIR,
) -> tuple[Path, Path]:
    """
    Bootstrap test split 5 times and recompute top-k SHAP ranks cheaply by resampling rows
    of the already computed SHAP matrix.

    Reports:
      - how often each feature appears in top-k
      - Spearman rank correlation vs original ranks (per bootstrap)
    """
    sv = np.asarray(shap_values)
    if sv.ndim != 2:
        raise ValueError(f"Expected 2D SHAP matrix, got shape {sv.shape}")

    n = sv.shape[0]
    rng = np.random.RandomState(RANDOM_SEED)

    base_mean = np.mean(np.abs(sv), axis=0)
    base_rank = pd.Series((-base_mean).argsort().argsort() + 1, index=feature_names)  # 1..N ranks

    appear = {f: 0 for f in feature_names}
    rankcorr_rows = []

    for b in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        boot_mean = np.mean(np.abs(sv[idx, :]), axis=0)

        boot_order = np.argsort(-boot_mean)
        top_feats = [feature_names[i] for i in boot_order[: int(top_k)]]
        for f in top_feats:
            appear[f] += 1

        boot_rank = pd.Series(((-boot_mean).argsort().argsort() + 1), index=feature_names)
        rho = float(base_rank.corr(boot_rank, method="spearman"))
        rankcorr_rows.append({"bootstrap": b + 1, "spearman_rank_corr_vs_original": rho})

    freq_df = (
        pd.DataFrame(
            {
                "feature": list(appear.keys()),
                "topk_appear_count": list(appear.values()),
                "topk_appear_pct": [v / float(n_boot) for v in appear.values()],
            }
        )
        .sort_values(["topk_appear_count", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    top10_freq_path = out_tables_dir / f"phase8_stability_shap_top{top_k}_{model_name}.csv"
    freq_df.to_csv(top10_freq_path, index=False)

    corr_df = pd.DataFrame(rankcorr_rows)
    corr_path = out_tables_dir / f"phase8_stability_shap_rankcorr_{model_name}.csv"
    corr_df.to_csv(corr_path, index=False)

    logger.info(f"SHAP stability saved: {top10_freq_path} and {corr_path}")
    return top10_freq_path, corr_path

def compute_phase8_tree_shap_mean_abs_light(
    *,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    explain_n: int,
    background_n: int,
    out_tables_dir: Path = TABLES_DIR,
    figures_subdir: str = "interpretation",
) -> tuple[Path, Path]:
    """
    Compute mean(|SHAP|) for a tree model, explaining a sample of test rows for speed.
    Returns (mean_abs_csv_path, bar_plot_path).
    """
    estimator = _unwrap_calibrated_estimator(model)
    preprocess, tree_model = _extract_preprocess_and_model_steps(estimator)

    bg_n = int(min(max(background_n, 1), len(X_train)))
    X_bg = X_train.sample(n=bg_n, random_state=RANDOM_SEED)

    ex_n = int(min(max(explain_n, 1), len(X_test)))
    X_ex = X_test.sample(n=ex_n, random_state=RANDOM_SEED)

    X_bg_t, feat_names = _transform_X(preprocess, X_bg)
    X_ex_t, _ = _transform_X(preprocess, X_ex)

    X_bg_df = pd.DataFrame(X_bg_t, columns=feat_names)
    X_ex_df = pd.DataFrame(X_ex_t, columns=feat_names)

    explainer = shap.TreeExplainer(tree_model, data=X_bg_df, model_output="probability")
    shap_vals = _ensure_shap_matrix(explainer.shap_values(X_ex_df))

    mean_abs = np.mean(np.abs(shap_vals), axis=0).reshape(-1)
    df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_tables_dir / f"phase8_shap_mean_abs_{model_name}_test{ex_n}_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # bar top-20 (quick visual)
    top = df.head(20).iloc[::-1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title(f"SHAP global importance (sampled) - {model_name} (n={ex_n})")
    bar_path = save_figure(fig, f"phase8_shap_bar_sampled_{model_name}_n{ex_n}", figures_subdir)

    logger.info(f"Light SHAP saved: {csv_path} and {bar_path}")
    return csv_path, bar_path


def compute_phase8_logistic_standardized_coeffs(
    *,
    model: Any,
    X_train: pd.DataFrame,
    model_name: str,
    out_tables_dir: Path = TABLES_DIR,
) -> Path:
    """
    Standardized coefficients in transformed feature space:
      coef_std = coef * std(X_train_transformed_feature)
    Also reports odds ratios = exp(coef).
    """
    estimator = model
    preprocess, lr = _extract_preprocess_and_model_steps(estimator)

    # unwrap if pipeline ended with CalibratedClassifierCV (unlikely for baseline)
    lr = _unwrap_calibrated_estimator(lr)

    if not hasattr(lr, "coef_"):
        raise ValueError("Provided baseline_logistic model does not expose coef_.")

    Xt, feat_names = _transform_X(preprocess, X_train)
    coef = np.asarray(lr.coef_).reshape(-1)

    # std for standardization
    std = np.asarray(Xt).std(axis=0, ddof=0)
    coef_std = coef * std
    odds_ratio = np.exp(coef)

    df = pd.DataFrame(
        {
            "feature": feat_names,
            "coef": coef.astype(float),
            "coef_std": coef_std.astype(float),
            "odds_ratio": odds_ratio.astype(float),
        }
    )
    df["abs_coef_std"] = np.abs(df["coef_std"])
    df = df.sort_values("abs_coef_std", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    out_path = out_tables_dir / f"phase8_logistic_coeffs_{model_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Logistic coefficients saved: {out_path}")
    return out_path


def create_phase8_cross_model_top10_comparison(
    *,
    extra_trees_shap_csv: Path,
    catboost_shap_csv: Path,
    logistic_coef_csv: Path,
    out_tables_dir: Path = TABLES_DIR,
) -> Path:
    """
    Single comparison table:
      rank 1..10, top feature per model, plus overlap counts appended as final rows.
    """
    et = pd.read_csv(extra_trees_shap_csv).head(PHASE8_CROSSMODEL_TOP_K)
    cb = pd.read_csv(catboost_shap_csv).head(PHASE8_CROSSMODEL_TOP_K)
    lg = pd.read_csv(logistic_coef_csv).head(PHASE8_CROSSMODEL_TOP_K)

    et_feats = et["feature"].tolist()
    cb_feats = cb["feature"].tolist()
    lg_feats = lg["feature"].tolist()

    et_set, cb_set, lg_set = set(et_feats), set(cb_feats), set(lg_feats)

    overlap_et_cb = len(et_set & cb_set)
    overlap_et_lg = len(et_set & lg_set)
    overlap_cb_lg = len(cb_set & lg_set)
    overlap_all3 = len(et_set & cb_set & lg_set)

    out = pd.DataFrame(
        {
            "rank": list(range(1, PHASE8_CROSSMODEL_TOP_K + 1)),
            "extra_trees_tuned_calibrated_top_feature": et_feats,
            "catboost_tuned_calibrated_top_feature": cb_feats,
            "baseline_logistic_top_feature": lg_feats,
        }
    )

    # Append overlap as extra rows (still "single table" CSV)
    footer = pd.DataFrame(
        [
            {
                "rank": "OVERLAP_ET_vs_CB",
                "extra_trees_tuned_calibrated_top_feature": overlap_et_cb,
                "catboost_tuned_calibrated_top_feature": "",
                "baseline_logistic_top_feature": "",
            },
            {
                "rank": "OVERLAP_ET_vs_LOG",
                "extra_trees_tuned_calibrated_top_feature": overlap_et_lg,
                "catboost_tuned_calibrated_top_feature": "",
                "baseline_logistic_top_feature": "",
            },
            {
                "rank": "OVERLAP_CB_vs_LOG",
                "extra_trees_tuned_calibrated_top_feature": overlap_cb_lg,
                "catboost_tuned_calibrated_top_feature": "",
                "baseline_logistic_top_feature": "",
            },
            {
                "rank": "OVERLAP_ALL_3",
                "extra_trees_tuned_calibrated_top_feature": overlap_all3,
                "catboost_tuned_calibrated_top_feature": "",
                "baseline_logistic_top_feature": "",
            },
        ]
    )

    out = pd.concat([out, footer], ignore_index=True)
    out_path = out_tables_dir / "phase8_cross_model_top10_comparison.csv"
    out.to_csv(out_path, index=False)

    logger.info(f"Cross-model comparison table saved: {out_path}")
    return out_path

def phase8_guardrail_check(
    *,
    train_engineered: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    expected_test_size: int = PHASE7_EXPECTED_TEST_SIZE,
    target_col: str = TARGET_COLUMN,
    all_zero_allowlist: Optional[list[str]] = None,
) -> None:
    """
    Guardrails to avoid interpreting the wrong test set.

    Checks:
      - expected row count (Phase 7 test size)
      - label/target column exists in engineered df
      - no all-zero engineered categorical columns in X_test (unless allowlisted)

    If violated: log warning and stop (raise RuntimeError).
    """
    all_zero_allowlist = all_zero_allowlist or PHASE8_ALL_ZERO_ALLOWLIST

    # 1) target exists
    if target_col not in train_engineered.columns:
        msg = f"Guardrail failed: target column '{target_col}' not found in train_engineered."
        logger.warning(msg)
        raise RuntimeError(msg)

    # 2) row count matches Phase 7 test size
    n = len(X_test)
    if n != int(expected_test_size):
        msg = (
            f"Guardrail failed: X_test has {n} rows but expected {expected_test_size} "
            f"(Phase 7 evaluated test size)."
        )
        logger.warning(msg)
        raise RuntimeError(msg)

    # 3) y alignment
    if len(y_test) != len(X_test):
        msg = f"Guardrail failed: y_test length {len(y_test)} != X_test length {len(X_test)}."
        logger.warning(msg)
        raise RuntimeError(msg)

    # 4) all-zero columns (unexpected engineered categorical dummies etc.)
    # treat NaN as 0 for this check
    zero_cols = []
    for c in X_test.columns:
        if c in set(all_zero_allowlist):
            continue
        s = X_test[c]
        # Only consider numeric/bool-like columns; ignore object columns (rare in engineered set)
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            if (s.fillna(0) == 0).all():
                zero_cols.append(c)

    if zero_cols:
        n_features = X_test.shape[1]
        frac = len(zero_cols) / max(1, n_features)

        # Stop only if the pattern is “too many zeros” (likely wrong encoding/test set)
        # Tune thresholds if needed, but these work well in practice.
        too_many = (len(zero_cols) >= 50) or (frac >= 0.20)

        msg = (
                f"Guardrail note: {len(zero_cols)} all-zero feature columns in X_test "
                f"({frac:.1%} of features): "
                + ", ".join(zero_cols[:50])
                + (" ..." if len(zero_cols) > 50 else "")
                + "\nThis can be normal for rare binaries / one-hot levels absent in the test split."
        )

        if too_many:
            logger.warning(
                msg + "\nStopping because the number/fraction is suspiciously high (possible wrong test set or encoding mismatch).")
            raise RuntimeError(msg)
        else:
            logger.warning(
                msg + "\nContinuing (small number of all-zero columns). If desired, allowlist them in PHASE8_ALL_ZERO_ALLOWLIST.")

    logger.info(
        f"Phase 8 guardrails passed (n_test={n}, target='{target_col}', all-zero cols=0)."
    )

def generate_phase8_what_we_learned_summary(
    *,
    model_name: str,
    shap_mean_abs_csv: Path,
    shap_values: np.ndarray,
    X_test_transformed: pd.DataFrame,
    out_tables_dir: Path = TABLES_DIR,
    top_n: int = PHASE8_SUMMARY_TOP_N,
) -> Path:
    """
    Concise Phase 8 summary table:
      feature + directionality + plausibility note.

    Directionality is inferred heuristically from SHAP vs feature values:
      - binary features: compare mean SHAP when feature==1 vs 0
      - continuous: Spearman correlation(feature, SHAP)
    """
    shap_df = pd.read_csv(shap_mean_abs_csv).head(int(top_n)).copy()
    feature_names = list(X_test_transformed.columns)

    # small clinical plausibility note map (edit freely)
    notes = {
        "age": "Older age is an established CHD risk factor.",
        "age_group": "Risk typically increases with age bands.",
        "totChol": "Higher total cholesterol is linked to CHD risk.",
        "chol": "Higher cholesterol is linked to CHD risk.",
        "sysBP": "Higher systolic BP / hypertension increases CHD risk.",
        "diaBP": "Higher diastolic BP / hypertension increases CHD risk.",
        "prevalentHyp": "Hypertension status is clinically plausible as a risk driver.",
        "currentSmoker": "Smoking increases cardiovascular risk.",
        "is_smoking_encoded": "Smoking status is clinically plausible as a risk driver.",
        "prevalentDiabetes": "Diabetes increases cardiovascular risk.",
        "diabetes": "Diabetes increases cardiovascular risk.",
        "sex": "Sex differences are observed; interpret with coding context.",
        "sex_encoded": "Sex differences are observed; interpret with coding context.",
        "education": "Likely socioeconomic proxy; association plausible but not causal.",
        "education_encoded": "Likely socioeconomic proxy; association plausible but not causal.",
        "risk_factor_count": "Composite risk burden; plausible but not mechanistic.",
    }

    rows = []
    sv = np.asarray(shap_values)
    for _, r in shap_df.iterrows():
        feat = r["feature"]
        if feat not in X_test_transformed.columns:
            continue
        j = feature_names.index(feat)

        x = X_test_transformed[feat].to_numpy()
        s = sv[:, j]

        direction = "mixed / non-linear"
        evidence = np.nan

        uniq = pd.Series(x).dropna().unique()
        if len(uniq) <= 2:
            # binary-ish: compare SHAP means
            m1 = float(np.mean(s[x == 1])) if np.any(x == 1) else np.nan
            m0 = float(np.mean(s[x == 0])) if np.any(x == 0) else np.nan
            if m1 == m1 and m0 == m0:
                delta = m1 - m0
                evidence = delta
                if delta > 0:
                    direction = "feature=1 increases predicted risk"
                elif delta < 0:
                    direction = "feature=1 decreases predicted risk"
                else:
                    direction = "no clear effect"
        else:
            # continuous-ish: Spearman corr
            try:
                corr = pd.Series(x).corr(pd.Series(s), method="spearman")
                evidence = float(corr)
                if corr is not None and abs(corr) >= 0.2:
                    direction = "higher values increase predicted risk" if corr > 0 else "higher values decrease predicted risk"
                else:
                    direction = "mixed / non-linear"
            except Exception:
                direction = "mixed / non-linear"

        plaus = notes.get(feat)
        if plaus is None:
            if "interaction" in str(feat).lower():
                plaus = "Engineered interaction term; interpret cautiously."
            else:
                plaus = "Clinically plausible (confirm with domain context)."

        rows.append(
            {
                "model": model_name,
                "feature": feat,
                "shap_rank": int(r["rank"]) if "rank" in r else None,
                "mean_abs_shap": float(r["mean_abs_shap"]) if "mean_abs_shap" in r else None,
                "directionality": direction,
                "direction_evidence": evidence,
                "clinical_plausibility_note": plaus,
            }
        )

    out = pd.DataFrame(rows)
    out_path = out_tables_dir / "phase8_what_we_learned_summary.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Phase 8 'what we learned' summary saved: {out_path}")
    return out_path

def _read_top_features_from_csv(path: Path, col: str = "feature", n: int = 10) -> list[str]:
    try:
        df = pd.read_csv(path)
        return df[col].head(n).astype(str).tolist()
    except Exception:
        return []


def update_phase8_model_card_and_chapter(
    *,
    model_name: str,
    artifacts: dict,
    out_dir: Path = MODEL_CARDS_DIR,
) -> tuple[Path, Path]:
    """
    Writes/updates:
      - model card for chosen model (markdown)
      - Phase 8 interpretation chapter (markdown)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull key artifact paths (strings -> Path)
    def P(k: str) -> Optional[Path]:
        v = artifacts.get(k)
        return Path(v) if v else None

    shap_mean_abs = P("shap_mean_abs_csv")
    perm_csv = P("perm_csv")
    overlap_csv = P("overlap_csv")
    risk_deciles_csv = P("risk_deciles_csv")
    reliability_bins_csv = P("reliability_bins_csv")
    subgroup_csv = P("subgroup_csv")
    stability_freq_csv = P("stability_freq_csv")
    stability_corr_csv = P("stability_corr_csv")
    local_cases_csv = P("local_cases_csv")
    cross_model_csv = P("cross_model_csv")

    # Read key summaries
    shap_top10 = _read_top_features_from_csv(shap_mean_abs) if shap_mean_abs else []
    perm_top10 = _read_top_features_from_csv(perm_csv) if perm_csv else []

    # Overlap count (if table exists, count rows where both in topN)
    overlap_text = "N/A"
    if overlap_csv and overlap_csv.exists():
        ov = pd.read_csv(overlap_csv)
        if "in_shap_topN" in ov.columns and "in_perm_topN" in ov.columns:
            overlap_text = f"{int((ov['in_shap_topN'] & ov['in_perm_topN']).sum())} shared features in top-N union table"

    # Calibration deciles quick finding
    decile_finding = "See risk decile table/plot."
    if risk_deciles_csv and risk_deciles_csv.exists():
        d = pd.read_csv(risk_deciles_csv)
        if len(d) >= 2 and {"observed_event_rate", "mean_predicted_risk"}.issubset(d.columns):
            bottom = float(d.iloc[0]["observed_event_rate"])
            top = float(d.iloc[-1]["observed_event_rate"])
            decile_finding = f"Observed event rate increases from ~{bottom:.3f} (lowest decile) to ~{top:.3f} (highest decile)."

    # Subgroup robustness highlight
    subgroup_finding = "See subgroup table/plot."
    if subgroup_csv and subgroup_csv.exists():
        sg = pd.read_csv(subgroup_csv)
        sg2 = sg[(sg["group_var"] != "overall") & sg["brier"].notna() & sg["average_precision"].notna()].copy()
        if len(sg2) > 0:
            worst_brier = sg2.sort_values("brier", ascending=False).head(1).iloc[0]
            best_ap = sg2.sort_values("average_precision", ascending=False).head(1).iloc[0]
            subgroup_finding = (
                f"Worst Brier subgroup: {worst_brier['group_var']}={worst_brier['group_value']} (Brier={worst_brier['brier']:.3f}, n={int(worst_brier['n'])}). "
                f"Best AP subgroup: {best_ap['group_var']}={best_ap['group_value']} (AP={best_ap['average_precision']:.3f}, n={int(best_ap['n'])})."
            )

    # Stability highlight
    stability_finding = "See stability CSVs."
    if stability_freq_csv and stability_freq_csv.exists():
        st = pd.read_csv(stability_freq_csv)
        top_stable = st[st["topk_appear_count"] >= 4].head(10)["feature"].astype(str).tolist()
        if top_stable:
            stability_finding = "Features appearing in top-k in ≥4/5 bootstraps: " + ", ".join(top_stable)

    # Local case quick summary
    local_finding = "See local case summary CSV."
    if local_cases_csv and local_cases_csv.exists():
        lc = pd.read_csv(local_cases_csv)
        local_finding = f"{len(lc)} reproducible cases selected (high/low/borderline). See case notes in reports/model_cards/."

    # Cross-model agreement highlight (overlap footer rows exist in your cross-model CSV)
    cross_finding = "See cross-model comparison table."
    if cross_model_csv and cross_model_csv.exists():
        cm = pd.read_csv(cross_model_csv)
        if "rank" in cm.columns:
            footer = cm[cm["rank"].astype(str).str.startswith("OVERLAP")]
            if len(footer) > 0:
                # small readable summary
                parts = []
                for _, rr in footer.iterrows():
                    parts.append(f"{rr['rank']}: {rr.iloc[1]}")
                cross_finding = " | ".join(parts)

    # Dependence plots list (you already save filepaths; we store count)
    dep_count = int(artifacts.get("dependence_plot_count", 0))

    # --- Write model card ---
    model_card_path = out_dir / f"model_card_{model_name}.md"

    model_card_md = "\n".join(
        [
            f"# Model Card - {model_name}",
            "",
            "## Phase 8: Interpretation update",
            "",
            "### Global drivers",
            "",
            "**Top drivers (SHAP mean |SHAP|, top 10):**",
            *(["- " + f for f in shap_top10] if shap_top10 else ["- (missing SHAP table)"]),
            "",
            "**Permutation importance sanity check (Brier, top 10):**",
            *(["- " + f for f in perm_top10] if perm_top10 else ["- (missing permutation table)"]),
            "",
            f"**SHAP vs permutation overlap:** {overlap_text}",
            "",
            "### How drivers behave (dependence)",
            f"- Generated dependence plots: {dep_count} (see reports/figures/interpretation/)",
            "",
            "### Calibration & risk stratification",
            f"- {decile_finding}",
            "- Reliability diagram includes bin counts; interpret sparse bins cautiously.",
            "",
            "### Local case studies",
            f"- {local_finding}",
            "",
            "### Subgroup robustness",
            f"- {subgroup_finding}",
            "",
            "### Explanation stability (cheap robustness)",
            f"- {stability_finding}",
            "",
            "### Limitations / safeguards",
            "",
            "- **Correlation ≠ causation:** feature importance reflects associations in this dataset, not causal effects.",
            "- **Correlated predictors:** SHAP/permutation can distribute importance across correlated variables.",
            "- **Data shift risk:** calibration and ranking may degrade under population shift or different measurement protocols.",
            "- **Constant / missing levels:** all-zero one-hot levels (or missing categories) can invalidate interpretation if not checked.",
            "- **External test encoding caveat:** ensure the external dataset uses identical preprocessing/encodings; otherwise feature meanings differ.",
            "",
        ]
    )

    model_card_path.write_text(model_card_md, encoding="utf-8")

    # --- Write chapter-style synthesis doc ---
    chapter_path = out_dir / f"phase8_interpretation_chapter_{model_name}.md"
    chapter_md = "\n".join(
        [
            f"# Phase 8 Interpretation Chapter - {model_name}",
            "",
            "## 1) Context (anchor to Phase 7)",
            f"- Header metrics table: {artifacts.get('header_table_path','(missing)')}",
            "",
            "## 2) Global importance",
            f"- SHAP summary plots + table: {artifacts.get('shap_fig_beeswarm','')}, {artifacts.get('shap_fig_bar','')}, {artifacts.get('shap_mean_abs_csv','')}",
            f"- Permutation importance (Brier): {artifacts.get('perm_csv','')}, {artifacts.get('perm_bar','')}",
            f"- Overlap table: {artifacts.get('overlap_csv','')}",
            "",
            "## 3) Feature effects / dependence",
            f"- Dependence plots generated: {dep_count}",
            "",
            "## 4) Calibration & risk stratification",
            f"- Risk deciles: {artifacts.get('risk_deciles_csv','')}, {artifacts.get('risk_deciles_plot','')}",
            f"- Reliability (with counts): {artifacts.get('reliability_bins_csv','')}, {artifacts.get('reliability_plot','')}",
            "",
            "## 5) Local explanations (case studies)",
            f"- Reproducible case list: {artifacts.get('local_cases_csv','')}",
            "- Waterfalls saved under reports/figures/interpretation/local/",
            "- Case notes saved under reports/model_cards/",
            "",
            "## 6) Error analysis & robustness",
            f"- Subgroup analysis: {artifacts.get('subgroup_csv','')}, {artifacts.get('subgroup_plot','')}",
            f"- Stability (bootstrap): {artifacts.get('stability_freq_csv','')}, {artifacts.get('stability_corr_csv','')}",
            "",
            "## 7) Cross-model agreement",
            f"- Cross-model table: {artifacts.get('cross_model_csv','')}",
            f"- CatBoost SHAP sampled: {artifacts.get('catboost_shap_csv','')}",
            f"- Logistic coefficients/ORs: {artifacts.get('logistic_coef_csv','')}",
            "",
            "## 8) Summary + limitations",
            "- Different model families may converge on similar predictors; where they diverge, check correlation structure and model capacity.",
            "- Treat interpretation as descriptive of this dataset and preprocessing pipeline; validate externally where possible.",
        ]
    )
    chapter_path.write_text(chapter_md, encoding="utf-8")

    logger.info(f"Model card written/updated: {model_card_path}")
    logger.info(f"Phase 8 chapter written: {chapter_path}")
    return model_card_path, chapter_path

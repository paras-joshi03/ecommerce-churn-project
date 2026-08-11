# ------------------------------------------------------------------
# explainability.py  -  Layer 8
#
# ExplainabilityEngine:
#   Uses SHAP TreeExplainer to produce global and local explanations
#   for XGBoost models (or XGBoost inside an imblearn Pipeline).
#
#   Global explanation: mean absolute SHAP value per feature across
#   all test customers. Shows which features drive churn overall.
#
#   Local explanation: SHAP values for a single customer. Shows
#   exactly which features pushed that individual's churn probability
#   up or down. Positive SHAP = increases churn risk. Negative = reduces.
#
#   SHAP values are cached after the first computation to avoid
#   recomputing on every dashboard interaction.
#
#   Handles SHAP version differences:
#   - SHAP >= 0.46 returns a 3D array (samples, features, classes).
#     We take index [:, :, 1] for the churn class.
#   - Older SHAP returns a list [class0_array, class1_array].
#     We take index [1].
#
# DriftMonitor:
#   Uses the Kolmogorov-Smirnov two-sample test to compare the
#   distribution of each feature between training data and new data.
#   A p-value below the significance threshold indicates that the
#   feature distribution has shifted since training.
#   When enough features drift, the model's predictions may no longer
#   be reliable and retraining should be considered.
# ------------------------------------------------------------------

import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from config import DRIFT_SIGNIFICANCE


def _extract_xgb(model):
    """
    Unwrap an XGBClassifier from an imblearn Pipeline if needed.
    SHAP TreeExplainer requires the raw XGBClassifier, not a Pipeline wrapper.
    """
    if isinstance(model, ImbPipeline):
        return model.named_steps.get("clf", None)
    return model


class ExplainabilityEngine:

    def __init__(self, model):
        xgb_model = _extract_xgb(model)
        if xgb_model is None:
            raise ValueError(
                "ExplainabilityEngine requires an XGBClassifier "
                "or a Pipeline with 'clf' as the model step."
            )
        self.explainer   = shap.TreeExplainer(xgb_model)
        self._shap_cache = {}

    # ------------------------------------------------------------------

    def _get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values and cache the result.
        Handles output format differences across SHAP versions.
        """
        cache_key = id(X)
        if cache_key not in self._shap_cache:
            raw = self.explainer.shap_values(X)

            if isinstance(raw, np.ndarray) and raw.ndim == 3:
                # SHAP >= 0.46: shape is (samples, features, classes)
                shap_values = raw[:, :, 1]
            elif isinstance(raw, list):
                # Older SHAP: list of [class0, class1]
                shap_values = raw[1]
            else:
                shap_values = raw

            self._shap_cache[cache_key] = shap_values

        return self._shap_cache[cache_key]

    # ------------------------------------------------------------------

    def global_explanation(self, X_test: pd.DataFrame):
        """
        Compute mean absolute SHAP value per feature.
        Higher values indicate features that consistently influence
        churn predictions across the entire customer base.

        Returns the ranked Series and a horizontal bar chart figure.
        """
        shap_values = self._get_shap_values(X_test)

        mean_abs = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_test.columns
        ).sort_values(ascending=False)

        top = mean_abs.head(12)
        color_map = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top)))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top.index[::-1], top.values[::-1], color=color_map[::-1], height=0.65)
        ax.set_xlabel("Mean absolute SHAP value", fontsize=10)
        ax.set_title("Global churn drivers across all customers", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        return mean_abs, fig

    # ------------------------------------------------------------------

    def local_explanation(self, X_test: pd.DataFrame, customer_index: int):
        """
        Explain the model's prediction for one specific customer.
        Returns a DataFrame ranking each feature by its contribution,
        the name of the top driver, and a bar chart figure.

        Positive SHAP impact means the feature increased churn probability.
        Negative means it reduced it.
        """
        shap_values = self._get_shap_values(X_test)

        impacts = pd.DataFrame({
            "Feature":       X_test.columns,
            "Feature_Value": X_test.iloc[customer_index].values,
            "SHAP_Impact":   shap_values[customer_index],
        }).sort_values("SHAP_Impact", key=abs, ascending=False)

        top_driver = impacts.iloc[0]["Feature"]

        top8   = impacts.head(8)
        colors = ["#c0392b" if v > 0 else "#27ae60" for v in top8["SHAP_Impact"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(
            top8["Feature"][::-1],
            top8["SHAP_Impact"][::-1],
            color=colors[::-1],
            height=0.6
        )
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP impact on churn probability", fontsize=9)
        ax.set_title("Why is this customer predicted to churn?", fontsize=10)
        plt.tight_layout()

        return impacts, top_driver, fig


# ------------------------------------------------------------------
# Drift Monitor
# ------------------------------------------------------------------

class DriftMonitor:
    """
    Monitors whether the distribution of input features has shifted
    between the training data and new incoming data.

    Uses the Kolmogorov-Smirnov two-sample test per feature.
    The KS test checks whether two samples come from the same distribution
    without assuming any particular shape (non-parametric).
    """

    def __init__(self):
        self.train_data   = None
        self.is_fitted    = False
        self.significance = DRIFT_SIGNIFICANCE

    def fit(self, X_train: pd.DataFrame):
        """Store training data distributions as the reference baseline."""
        self.train_data = X_train.copy()
        self.is_fitted  = True
        print("Drift monitor fitted on {} training samples.".format(len(X_train)))
        return self

    def check(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Run KS test for each feature and return a drift report.
        Features with p-value below the significance threshold are flagged.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before calling check().")

        records = []
        for col in self.train_data.columns:
            if col not in X_new.columns:
                continue

            train_vals = self.train_data[col].dropna().values
            new_vals   = X_new[col].dropna().values

            ks_stat, p_value = stats.ks_2samp(train_vals, new_vals)
            drifted          = p_value < self.significance

            records.append({
                "Feature": col,
                "KS_Stat": round(float(ks_stat), 4),
                "P_Value": round(float(p_value), 4),
                "Drifted": drifted,
                "Status":  "Drifted" if drifted else "Stable",
            })

        report = pd.DataFrame(records).sort_values("KS_Stat", ascending=False)

        n_drifted = report["Drifted"].sum()
        print("Drift check complete. {} of {} features show drift.".format(
            n_drifted, len(records)
        ))

        return report

    def retrain_recommended(self, drift_report: pd.DataFrame, threshold: int = 3) -> bool:
        """
        Return True if the number of drifted features exceeds the threshold.
        The default threshold of 3 means retraining is recommended when
        at least 3 features have shifted distribution.
        """
        return int(drift_report["Drifted"].sum()) >= threshold

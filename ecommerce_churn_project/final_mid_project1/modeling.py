# ------------------------------------------------------------------
# modeling.py  -  Layer 4
#
# Two components in this layer:
#
# ChurnModeler:
#   Trains XGBoost and Random Forest on SMOTE-balanced training data.
#   Produces a side-by-side comparison of results with and without SMOTE
#   so the impact of class balancing is visible in the dashboard.
#   The best model by Recall is stored as the primary model.
#
#   Why Recall is the primary metric:
#   Missing a customer who is about to churn is far more expensive than
#   incorrectly flagging a loyal customer. A missed churner means lost
#   revenue with no chance to intervene. A false positive means sending
#   one unnecessary retention offer, which is a small cost.
#
#   Why SMOTE only on training data:
#   Applying SMOTE before the train/test split would allow synthetic
#   samples to appear in both sets, making test metrics unrealistically
#   optimistic. SMOTE is applied after splitting so the test set always
#   contains only real customer records.
#
# SurvivalAnalyzer:
#   Fits a Kaplan-Meier estimator on Tenure and Churn columns.
#   Estimates the probability that a customer survives (does not churn)
#   at each month of tenure.
#   Works only when a Tenure column is available.
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, recall_score, roc_auc_score,
    precision_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from config import TEST_SIZE, RANDOM_SEED


class ChurnModeler:

    def __init__(self):
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.best_model      = None
        self.best_model_name = ""
        self.metrics         = {}
        self.smote_stats     = {}   # stores before/after class counts

    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data, apply SMOTE on training split only, train both models,
        record metrics with and without SMOTE, select best by Recall.

        Returns X_train, X_test, y_train, y_test.
        """
        print("Splitting data: {:.0f}% train, {:.0f}% test.".format(
            (1 - TEST_SIZE) * 100, TEST_SIZE * 100
        ))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y   # preserves class ratio in both splits
        )

        # Record class distribution before SMOTE.
        before_counts = y_train.value_counts().to_dict()

        # Apply SMOTE to training data only.
        # The test set is never touched so evaluation reflects real-world performance.
        print("Applying SMOTE to training data.")
        print("Class distribution before SMOTE: {}.".format(before_counts))

        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        after_counts = pd.Series(y_train_balanced).value_counts().to_dict()
        print("Class distribution after SMOTE:  {}.".format(after_counts))

        self.smote_stats = {
            "before": before_counts,
            "after":  after_counts,
        }

        # Train both models without SMOTE first to record baseline.
        print("Training XGBoost without SMOTE (baseline).")
        xgb_no_smote = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0
        )
        xgb_no_smote.fit(X_train, y_train)
        xgb_base_pred = xgb_no_smote.predict(X_test)

        self.metrics["XGBoost_No_SMOTE"] = self._compute_metrics(
            y_test, xgb_base_pred,
            xgb_no_smote.predict_proba(X_test)[:, 1]
        )

        # Train XGBoost with SMOTE.
        print("Training XGBoost with SMOTE.")
        self.xgb_model.fit(X_train_balanced, y_train_balanced)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]

        self.metrics["XGBoost_SMOTE"] = self._compute_metrics(
            y_test, xgb_pred, xgb_prob
        )

        # Train Random Forest with SMOTE.
        print("Training Random Forest with SMOTE.")
        self.rf_model.fit(X_train_balanced, y_train_balanced)
        rf_pred = self.rf_model.predict(X_test)
        rf_prob = self.rf_model.predict_proba(X_test)[:, 1]

        self.metrics["RandomForest_SMOTE"] = self._compute_metrics(
            y_test, rf_pred, rf_prob
        )

        # Select the best model by Recall.
        xgb_recall = self.metrics["XGBoost_SMOTE"]["recall"]
        rf_recall  = self.metrics["RandomForest_SMOTE"]["recall"]

        if xgb_recall >= rf_recall:
            self.best_model      = self.xgb_model
            self.best_model_name = "XGBoost"
            print("Best model: XGBoost (Recall = {:.4f}).".format(xgb_recall))
        else:
            self.best_model      = self.rf_model
            self.best_model_name = "RandomForest"
            print("Best model: Random Forest (Recall = {:.4f}).".format(rf_recall))

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------

    def _compute_metrics(self, y_true, y_pred, y_prob) -> dict:
        """Compute a standard set of classification metrics."""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
            "report":    classification_report(
                             y_true, y_pred, output_dict=True, zero_division=0
                         ),
            "confusion_matrix": cm.tolist(),
        }


# ------------------------------------------------------------------
# Survival Analysis - Kaplan-Meier estimator
# ------------------------------------------------------------------

class SurvivalAnalyzer:
    """
    Fits a Kaplan-Meier survival curve on customer Tenure data.
    The survival probability at time t is the probability that a
    customer has not churned by tenure month t.

    Requires Tenure (duration) and Churn (event flag) columns.
    Works on raw unscaled data.
    """

    def __init__(self, duration_col: str = "Tenure", event_col: str = "Churn"):
        self.duration_col = duration_col
        self.event_col    = event_col
        self.km_table     = None
        self.is_fitted    = False

    def fit(self, dataframe: pd.DataFrame):
        """
        Compute the Kaplan-Meier survival table.
        Each row in km_table represents one time point with:
          At_Risk       : customers still active at this tenure
          Events        : customers who churned at this tenure
          Survival_Prob : estimated probability of survival up to this point
        """
        if self.duration_col not in dataframe.columns:
            print("Survival analysis skipped: '{}' column not found.".format(
                self.duration_col
            ))
            return self

        if self.event_col not in dataframe.columns:
            print("Survival analysis skipped: '{}' column not found.".format(
                self.event_col
            ))
            return self

        data = dataframe[[self.duration_col, self.event_col]].dropna().copy()
        data.columns = ["T", "E"]
        data["T"]    = data["T"].astype(int)

        survival_prob = 1.0
        records       = []

        for t in sorted(data["T"].unique()):
            at_risk = (data["T"] >= t).sum()
            events  = ((data["T"] == t) & (data["E"] == 1)).sum()

            if at_risk > 0:
                survival_prob *= (1 - events / at_risk)

            records.append({
                "Tenure":        t,
                "At_Risk":       int(at_risk),
                "Events":        int(events),
                "Survival_Prob": round(survival_prob, 4),
                "Churn_Prob":    round(1 - survival_prob, 4),
            })

        self.km_table  = pd.DataFrame(records)
        self.is_fitted = True

        print("Survival model fitted across {} tenure points.".format(
            len(records)
        ))
        return self

    def survival_at(self, months: int) -> float:
        """Return estimated survival probability at a given tenure month."""
        if self.km_table is None:
            return float("nan")
        rows = self.km_table[self.km_table["Tenure"] <= months]
        return float(rows["Survival_Prob"].iloc[-1]) if len(rows) else 1.0

    def get_danger_zones(self, drop_threshold: float = 0.03) -> list:
        """
        Return tenure months where survival drops sharply.
        These are the months where churn acceleration is highest
        and proactive intervention would have the most impact.
        """
        if self.km_table is None:
            return []
        km          = self.km_table.copy()
        km["Drop"]  = km["Survival_Prob"].diff(-1).fillna(0)
        danger_rows = km[km["Drop"] >= drop_threshold]
        return danger_rows[["Tenure", "Survival_Prob", "Drop"]].to_dict("records")

    def summary(self) -> dict:
        """Key survival statistics for the dashboard."""
        if not self.is_fitted:
            return {}
        below_half = self.km_table[self.km_table["Survival_Prob"] <= 0.5]
        median     = float(below_half["Tenure"].iloc[0]) if len(below_half) else None
        return {
            "survival_at_6m":  self.survival_at(6),
            "survival_at_12m": self.survival_at(12),
            "survival_at_24m": self.survival_at(24),
            "median_tenure":   median,
        }

# ------------------------------------------------------------------
# model_selection.py  -  Layer 5
#
# Model Zoo: evaluates Logistic Regression, Random Forest,
# XGBoost, and LightGBM on the same train/test split.
#
# GridSearch: tunes XGBoost hyperparameters using cross-validation.
# SMOTE is placed inside the pipeline so it only operates on each
# training fold. Applying SMOTE outside the CV loop would let
# synthetic samples from the test fold influence training,
# making cross-validation results misleadingly optimistic.
#
# Final model selection is based on Recall because the cost of
# missing a churner outweighs the cost of a false positive.
# ------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from config import RANDOM_SEED, GRIDSEARCH_PARAMS, GRIDSEARCH_CV


class ModelSelector:

    def __init__(self):
        self.best_model      = None
        self.best_model_name = ""
        self.best_recall     = 0.0
        self.best_auc        = 0.0
        self.all_results     = {}
        self.best_params     = {}

    # ------------------------------------------------------------------

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, name: str):
        """
        Train a model with SMOTE applied to training data and evaluate on test set.
        SMOTE is applied here (not in GridSearch) because this is a quick
        single-run evaluation, not cross-validation.
        """
        smote = SMOTE(random_state=RANDOM_SEED)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        model.fit(X_balanced, y_balanced)

        predictions  = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        recall        = recall_score(y_test, predictions, zero_division=0)
        auc           = roc_auc_score(y_test, probabilities)

        self.all_results[name] = {
            "recall":  round(recall, 4),
            "roc_auc": round(auc, 4),
            "report":  classification_report(
                           y_test, predictions, output_dict=True, zero_division=0
                       ),
            "model": model,
        }

        print("{} - Recall: {:.4f}, ROC-AUC: {:.4f}".format(name, recall, auc))
        return recall, auc

    # ------------------------------------------------------------------

    def run_model_zoo(self, X_train, y_train, X_test, y_test):
        """
        Evaluate all four models and set the current best by Recall.
        """
        print("Running model zoo.")
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, random_state=RANDOM_SEED,
                eval_metric="logloss", verbosity=0
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=100, random_state=RANDOM_SEED,
                verbose=-1, n_jobs=-1
            ),
        }

        best_recall = -1
        best_name   = ""

        for name, model in models.items():
            recall, _ = self._evaluate_model(
                model, X_train, y_train, X_test, y_test, name
            )
            if recall > best_recall:
                best_recall = recall
                best_name   = name

        self.best_model_name = best_name
        self.best_recall     = self.all_results[best_name]["recall"]
        self.best_auc        = self.all_results[best_name]["roc_auc"]
        self.best_model      = self.all_results[best_name]["model"]

        print("Model zoo complete. Current best: {} (Recall = {:.4f}).".format(
            best_name, best_recall
        ))
        return self.best_model

    # ------------------------------------------------------------------

    def run_grid_search(self, X_train, y_train, X_test, y_test):
        """
        Tune XGBoost with GridSearchCV.
        SMOTE is placed inside an ImbPipeline so it is applied
        independently within each CV fold's training portion.
        This is the correct way to combine SMOTE with cross-validation.
        """
        print("Running GridSearch on XGBoost. This may take a few minutes.")

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("clf",   XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                verbosity=0
            )),
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=GRIDSEARCH_PARAMS,
            scoring="recall",
            cv=GRIDSEARCH_CV,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        tuned_model       = grid_search.best_estimator_
        tuned_predictions = tuned_model.predict(X_test)
        tuned_probs       = tuned_model.predict_proba(X_test)[:, 1]
        tuned_recall      = recall_score(y_test, tuned_predictions, zero_division=0)
        tuned_auc         = roc_auc_score(y_test, tuned_probs)

        self.best_params              = grid_search.best_params_
        self.all_results["XGBoost_Tuned"] = {
            "recall":  round(tuned_recall, 4),
            "roc_auc": round(tuned_auc, 4),
            "report":  classification_report(
                           y_test, tuned_predictions, output_dict=True, zero_division=0
                       ),
            "model": tuned_model,
        }

        print("GridSearch complete. Best params: {}.".format(self.best_params))
        print("Tuned XGBoost - Recall: {:.4f}, ROC-AUC: {:.4f}".format(
            tuned_recall, tuned_auc
        ))

        if tuned_recall >= self.best_recall:
            self.best_model      = tuned_model
            self.best_model_name = "XGBoost_Tuned"
            self.best_recall     = tuned_recall
            self.best_auc        = tuned_auc
            print("XGBoost_Tuned is now the best model.")

        return self.best_model

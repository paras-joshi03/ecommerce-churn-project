# ------------------------------------------------------------------
# model_store.py
#
# Saves and loads all trained artifacts so the model is trained once
# and reused across all prediction sessions.
#
# After training on the labelled dataset, all components are saved:
#   - The trained model (best from model zoo + grid search)
#   - The fitted FeatureEngineer (encoders, scaler, training medians)
#   - The fitted FeatureSelector (selected feature list)
#   - The fitted DriftMonitor (training distribution baseline)
#   - The fitted SurvivalAnalyzer (KM table)
#   - Training metadata (SMOTE stats, model name, metrics, etc.)
#
# In prediction modes, these are loaded directly.
# This means no retraining is needed when a business uploads new data.
# ------------------------------------------------------------------

import os
import joblib
from config import (
    SAVE_DIR, MODEL_PATH, ENGINEER_PATH, SELECTOR_PATH,
    DRIFT_PATH, SURVIVAL_PATH, META_PATH
)


class ModelStore:

    @staticmethod
    def save(model, engineer, selector, drift_monitor, survival, meta: dict):
        """
        Save all pipeline artifacts to the saved_model directory.
        Creates the directory if it does not exist.
        """
        os.makedirs(SAVE_DIR, exist_ok=True)

        joblib.dump(model,         MODEL_PATH)
        joblib.dump(engineer,      ENGINEER_PATH)
        joblib.dump(selector,      SELECTOR_PATH)
        joblib.dump(drift_monitor, DRIFT_PATH)
        joblib.dump(survival,      SURVIVAL_PATH)
        joblib.dump(meta,          META_PATH)

        print("All artifacts saved to '{}'.".format(SAVE_DIR))

    @staticmethod
    def load():
        """
        Load all saved artifacts.
        Returns a dict with all components, or raises an error
        if training has not been run yet.
        """
        missing = [
            path for path in [MODEL_PATH, ENGINEER_PATH, SELECTOR_PATH, META_PATH]
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Trained model not found. Please run the system in training mode "
                "first using your labelled dataset (with Churn column).\n"
                "Missing: {}".format(missing)
            )

        artifacts = {
            "model":    joblib.load(MODEL_PATH),
            "engineer": joblib.load(ENGINEER_PATH),
            "selector": joblib.load(SELECTOR_PATH),
            "meta":     joblib.load(META_PATH),
        }

        if os.path.exists(DRIFT_PATH):
            artifacts["drift_monitor"] = joblib.load(DRIFT_PATH)
        else:
            artifacts["drift_monitor"] = None

        if os.path.exists(SURVIVAL_PATH):
            artifacts["survival"] = joblib.load(SURVIVAL_PATH)
        else:
            artifacts["survival"] = None

        print("Artifacts loaded from '{}'.".format(SAVE_DIR))
        return artifacts

    @staticmethod
    def is_trained() -> bool:
        """Check whether a trained model exists on disk."""
        return all(os.path.exists(p) for p in [MODEL_PATH, ENGINEER_PATH, META_PATH])
